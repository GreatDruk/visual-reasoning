"""Learn per-pixel votes for partially visible instances from paired contest images."""
import argparse
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from scripts.research import occlusion_verifier as v
from kaggle_suite_common import amp, tune_batch

H, W = v.HEIGHT, v.WIDTH
EXTRA = Path(os.environ.get("FRAGMENT_INPUT", "."))
SIZES = (2,8,6)  # joint material x shape has six classes


def coords(height,width,device,step):
    y,x=torch.meshgrid(torch.arange(height,device=device),torch.arange(width,device=device),indexing="ij")
    return torch.stack((x,y),-1).float().add(.5).mul(step)


class FragmentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk=v.ResNetCenters(full=True)
        self.trunk.load_state_dict(v.load(v.PRIORS/"r50_img.pt"))
        self.trunk.head=nn.Identity()
        self.decode=nn.Sequential(nn.Conv2d(128,64,3,padding=1),nn.GroupNorm(8,64),nn.SiLU(),
                                  nn.Conv2d(64,5+sum(SIZES),1))

    def forward(self,x):
        b=self.trunk
        z=b.stem[:3]((x-b.mean)/b.std)
        a=b.layer1(b.stem[3](z))
        c=b.layer2(a)
        d=b.layer3(c)
        f=b.lateral4(b.layer4(d))
        for skip in (b.lateral3(d),b.lateral2(c),b.lateral1(a)):
            f=skip+F.interpolate(f,size=skip.shape[-2:],mode="bilinear",align_corners=False)
        f=F.interpolate(f,size=z.shape[-2:],mode="bilinear",align_corners=False)
        return self.decode(torch.cat((f,z),1)).float()


def images(split,names):
    with ThreadPoolExecutor(4) as pool:
        return np.stack(list(pool.map(v.read_rgb,[v.DATA/split/n for n in names])))


class Scenes:
    def __init__(self,device,rgb=None):
        with np.load(EXTRA/"fragment_train.npz") as a:
            names,box,mask,labels,keep=[a[k] for k in ("names","box","mask","labels","keep")]
        pixels=images("train",names) if rgb is None else rgb
        self.rgb=torch.from_numpy(pixels).to(device).permute(0,3,1,2)
        instance=np.zeros((len(names),H,W),np.uint8)
        for i in range(len(names)):
            order=np.argsort(box[i,:,1]+box[i,:,3])
            for j in order[keep[i,order]]:
                x,y,w,h=box[i,j]
                region=instance[i,y:y+h,x:x+w]
                region[cv2.resize(mask[i,j],(int(w),int(h)))>127]=j+1
        self.instances=torch.from_numpy(instance).to(device)
        self.box=torch.from_numpy(box.astype("f")).to(device)
        self.labels=torch.from_numpy(labels.astype("int64")).to(device)
        self.masks=torch.from_numpy(mask).to(device)
        self.donors=torch.from_numpy(np.stack(np.nonzero(keep & (labels>=0).all(-1)),1)).to(device)
        if not len(self.donors):
            raise ValueError("No confident train donors")
        self.anchors=torch.from_numpy(keep).to(device)
        self.grid=coords(H,W,device,1)
        self.device=device

    def pair(self,idx,*,tiny_fragments=False,donors=None,difficulty="mixed"):
        n=len(idx)
        image=self.rgb[idx].float()/255
        inst=self.instances[idx]
        device=image.device
        front=(torch.rand(n,self.box.shape[1],device=device)*self.anchors[idx]).argmax(1)
        donors=self.donors if donors is None else donors
        d,k=donors[torch.randint(len(donors),(n,),device=device)].T
        box=self.box[idx,front]
        source=self.box[d,k]
        extent=source[:,2:]*(.75+.35*torch.rand(n,1,device=device))
        offset=(torch.rand(n,2,device=device)-.5)*box[:,2:]*1.15
        offset[:,1]=-offset[:,1].abs()
        center=box[:,:2]+.5*box[:,2:]+offset
        origin=center-.5*extent
        uv=(self.grid-origin[:,None,None])/extent[:,None,None]
        alpha=F.grid_sample(self.masks[d,k,None].float()/255,uv*2-1,align_corners=False)[:,0]
        source_grid=(source[:,None,None,:2]+uv*source[:,None,None,2:])/image.new_tensor([W,H])
        patch=F.grid_sample(self.rgb[d].float()/255,source_grid*2-1,align_corners=False)
        visible=alpha*(inst==0)
        ratio=visible.sum((1,2))/alpha.sum((1,2)).clamp_min(1)
        native=inst[:,1::2,1::2].long()
        if tiny_fragments:
            upper=torch.where(torch.rand(n,device=device)<.5,.28,.85)
            added=(F.max_pool2d(visible[:,None],2)[:,0]>.5)&(native==0)
            ok=(ratio>.035)&(ratio<upper)&(added.sum((1,2))>=2)
            if difficulty=="tiny":
                pixels=(visible>.5).sum((1,2))
                ok=(pixels>=4)&(pixels<=20)&(ratio<=.28)&(added.sum((1,2))>=2)
            elif difficulty=="middle":
                ok=(ratio>.12)&(ratio<.70)&((visible>.5).sum((1,2))>20)
        else:
            added=visible[:,1::2,1::2]>.5
            ok=(ratio>.12)&(ratio<.85)
        ok &= (d!=idx)&self.anchors[idx].any(1)
        visible=visible*ok[:,None,None]
        composed=image+(patch-image)*visible[:,None]
        blank=torch.zeros_like(source[:,None])
        boxes=torch.cat((blank,self.box[idx],torch.cat((origin,extent),1)[:,None]),1)
        attributes=torch.cat((self.labels[idx,:1]*0-1,self.labels[idx],self.labels[d,k,None]),1)
        added=added&ok[:,None,None]
        new=torch.where(added,boxes.shape[1]-1,native)
        return (torch.cat((image,composed)),torch.cat((native,new)),boxes.repeat(2,1,1),
                attributes.repeat(2,1,1),added)


def objective(model,batch):
    rgb,inst,boxes,labels,added=batch
    with amp(rgb.device):
        raw=model(rgb)
    obj=raw[:,0]
    parts=raw.permute(0,2,3,1)
    present=inst>0
    row=torch.arange(len(raw),device=raw.device)[:,None,None]
    box=boxes[row,inst]
    label=labels[row,inst]
    center=box[...,:2]+.5*box[...,2:]
    xy=coords(*inst.shape[1:],raw.device,2)
    offset=(center-xy)/24
    size=(box[...,2:].clamp_min(1)/24).log()
    # Give each inserted fragment comparable gradient mass even when little is visible.
    extra=torch.cat((torch.zeros_like(added),added))
    weight=present.float()+extra.float()*2
    fg=F.binary_cross_entropy_with_logits(obj,present.float(),reduction="none")
    loss=(fg*(weight+.12)).mean()/(weight+.12).mean()
    regression=F.smooth_l1_loss(parts[...,1:5],torch.cat((offset,size),-1),reduction="none").sum(-1)
    loss+=(regression*weight).sum()/weight.sum().clamp_min(1)
    for g,scores in enumerate(raw[:,5:].split(SIZES,1)):
        target=label[...,g]
        ce=F.cross_entropy(scores,target,ignore_index=-1,reduction="none")
        loss+=(ce*weight).sum()/(weight*(target>=0)).sum().clamp_min(1)
    before,after=obj.chunk(2)
    loss+=.4*(F.softplus(2-after+before)*added).sum()/added.sum().clamp_min(1)
    return loss


def fit(device,epochs):
    data=Scenes(device)
    model=FragmentNet().to(device,memory_format=torch.channels_last)
    model.train()
    for module in model.trunk.modules():
        if isinstance(module,nn.BatchNorm2d):
            module.eval()
    opt=torch.optim.AdamW([{"params":model.trunk.parameters(),"lr":.00009},
                           {"params":model.decode.parameters(),"lr":.0007}])
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    def probe(n):
        try:
            loss=objective(model,data.pair(torch.arange(n,device=device)%len(data.rgb)))
            scaler.scale(loss).backward()
        finally:
            opt.zero_grad(set_to_none=True)
    sizes=(96,64,48,32,16,8) if device.type=="cuda" else (2,)
    batch=tune_batch("paired scenes per batch",sizes,probe,device)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    for epoch in range(epochs):
        history=[]
        for idx in torch.randperm(len(data.rgb),device=device).split(batch):
            opt.zero_grad(set_to_none=True)
            loss=objective(model,data.pair(idx))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(opt)
            scaler.update()
            history.append(loss.detach())
        sched.step()
        mean=float(torch.stack(history).mean())
        if not math.isfinite(mean):
            raise ValueError("Non-finite train loss")
        print(epoch+1,mean,flush=True)
    torch.save(model.state_dict(),v.OUT/"fragments.pt")


def proposals(raw):
    """Group foreground pixels by their predicted center; pool attributes of their voters."""
    b,_,h,w=raw.shape
    grid=coords(h,w,raw.device,2)
    z=raw.permute(0,2,3,1)
    center=(grid+24*z[...,1:3])/4-.5
    extent=24*z[...,3:5].clamp(-1.5,1.5).exp()
    # Weight a visible pixel by the inverse predicted footprint of the object.
    weight=raw[:,0].sigmoid().pow(2)*4/extent.prod(-1)
    attrs=[s.softmax(-1) for s in z[...,5:].split(SIZES,-1)]
    fields=torch.cat((torch.ones_like(weight[...,None]),center*4+2,extent,*attrs),-1)
    sums=raw.new_zeros(b,fields.shape[-1],h*w//4)
    lower=center.floor()
    for x,y in ((0,0),(1,0),(0,1),(1,1)):
        at=lower+raw.new_tensor([x,y])
        fraction=(1-(center-at).abs()).prod(-1)*weight
        u,vv=at.long().unbind(-1)
        fraction*=((u>=0)&(u<w//2)&(vv>=0)&(vv<h//2))
        index=(vv*(w//2)+u).clamp(0,h*w//4-1).flatten(1)
        sums.scatter_add_(2,index[:,None].expand(-1,sums.shape[1],-1),
                          (fields*fraction[...,None]).flatten(1,2).transpose(1,2))
    pooled=F.avg_pool2d(sums.view(b,-1,h//2,w//2),3,1,1)*9
    mass=pooled[:,0]
    peak=mass==F.max_pool2d(mass,5,1,2)
    idx=mass.masked_fill(~peak,-1).flatten(1).topk(20).indices
    f=pooled.flatten(2).gather(2,idx[:,None].expand(-1,pooled.shape[1],-1)).transpose(1,2)
    m=f[...,:1].clamp_min(1e-8)
    loc,wh,sz,col,joint=(f[...,1:]/m).split((2,2,*SIZES),-1)
    joint=joint.reshape(b,20,2,3)
    probs=torch.cat((sz,col,joint.sum(-1),joint.sum(-2)),-1)
    norm=raw.new_tensor([W,H])
    box=torch.cat(((loc-.5*wh)/norm,wh/norm),-1)
    xy=(loc+wh*raw.new_tensor([0,.38]))/norm*2-1
    confidence=torch.logit((1-torch.exp(-4*m)).clamp(1e-6,1-1e-6))
    confidence=confidence.masked_fill(~peak.flatten(1).gather(1,idx)[...,None],-20.)
    return torch.cat((confidence,confidence,probs.clamp_min(1e-7).log(),box,xy),-1).cpu().numpy()


@torch.inference_mode()
def extract(split,device):
    names=pd.read_csv(v.DATA/f"{split}.csv").image.unique()
    model=FragmentNet().to(device,memory_format=torch.channels_last).eval()
    model.load_state_dict(v.load(v.OUT/"fragments.pt"))
    original=v.ResNetCenters(full=True).to(device,memory_format=torch.channels_last).eval()
    original.load_state_dict(v.load(v.PRIORS/"r50_img.pt"))
    head=v.model_head("r50_img",device)
    out=[]
    diagnostics=[]
    checks=[]
    def forward(rgb):
        primary=v.detector_outputs(original,head,rgb)
        with amp(device):
            raw=model(rgb)
        return primary,proposals(raw)
    sizes=(192,128,96,64,32,16) if device.type=="cuda" else (4,)
    batch=tune_batch("scene inference",sizes,lambda n:forward(torch.zeros(n,3,H,W,device=device)),device)
    for start in range(0,len(names),batch):
        rgb=torch.from_numpy(images(split,names[start:start+batch])).to(device).permute(0,3,1,2).float()/255
        left,right=forward(rgb)
        merged=[v.merge(a,b) for a,b in zip(left,right)]
        for i,(x,_,ok) in enumerate(merged):
            present=x[ok,-3:-1]
            diagnostics.append([len(present),int((present[:,0]==0).sum())])
            if split=="val" and names[start+i]=="58281.jpg":
                for cx,cy in ((122.,119.),(144.,89.)):
                    cand=right[i]
                    centers=(cand[:,-6:-4]+.5*cand[:,-4:-2])*[W,H]
                    j=np.linalg.norm(centers-[cx,cy],axis=1).argmin()
                    checks.append([cx,cy,*centers[j],float(cand[j,0]),*cand[j,2:2+v.N_ATTRS]])
        out.append(tuple(map(np.stack,zip(*merged))))
    x,xy,valid=(np.concatenate(g) for g in zip(*out))
    (v.CACHE/"tables").mkdir(exist_ok=True,parents=True)
    np.savez(v.CACHE/"tables"/f"{split}.npz",names=names.astype(str),x=x,xy=xy,valid=valid)
    np.savez(v.OUT/f"{split}_proposals.npz",names=names.astype(str),counts=diagnostics,
             cylinder_candidates=np.asarray(checks))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("fit","val","test"))
    parser.add_argument("seed",type=int)
    parser.add_argument("--epochs",type=int,default=14)
    args=parser.parse_args()
    _,device=v.setup(args.seed)
    v.OUT.mkdir(exist_ok=True)
    if args.action=="fit":
        fit(device,args.epochs)
        extract("train",device)
        v.train("control",args.seed,12,device)
    else:
        extract(args.action,device)
        v.infer(args.action,"control",args.seed,device)
