"""Caption-supervised dense count integration with counterfactual count differences."""
import argparse
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from scripts.research.fragment_votes import Scenes, images, v, EXTRA
from kaggle_suite_common import amp, tune_batch
from native_centers import canonical_form, parse_program
from src.models.slot_program import EXACT, MORE, FEWER, SAME, EXIST, NONE, EXCLUSIVE_GROUPS
from src.twins import find_twins

TYPES=list(itertools.product(*EXCLUSIVE_GROUPS))
ACTIVE=("counting","comparison")


def queries(frame,device):
    scene,names=pd.factorize(frame.image)
    captions=[canonical_form(r.caption,r.category) for r in frame.itertuples()]
    pairs=find_twins(frame.image,captions,frame.category)
    rows=[]
    for a,b in pairs:
        if frame.category.iat[a] in (*ACTIVE,"negation","attribute"):
            rows.append((a,b))
    if not rows:
        raise ValueError("No supported caption pairs")
    rows=np.asarray(rows)
    table=[]
    def mask(words):
        return [set(words)<=set(t) for t in TYPES]
    for r in rows.ravel():
        p=parse_program(captions[r])
        if p is None or p.relation or p.op not in (EXACT,MORE,FEWER,SAME,EXIST,NONE):
            raise ValueError(captions[r])
        table.append((mask(p.a),mask(p.b),p.op,p.number))
    arrays=[torch.as_tensor(np.array(t),device=device) for t in zip(*table)]
    return np.asarray(names,dtype=str),rows,torch.as_tensor(scene[rows[:,0]],device=device),arrays


class Counter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=v.ResNetCenters(full=True)
        self.net.load_state_dict(v.load(v.PRIORS/"r50_img.pt"))
        # One correction for mass at each cell. An object's mass may be spread over many cells.
        self.mass=nn.Sequential(nn.Conv2d(20,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,1,1))
        nn.init.zeros_(self.mass[-1].weight)
        nn.init.zeros_(self.mass[-1].bias)

    def forward(self,rgb,local=False):
        with amp(rgb.device):
            raw=self.net(rgb)
            delta=self.mass(raw)
        raw=raw.float()
        heat=raw[:,:1]
        # Each cell spends a fraction of a unit according to its evidence relative to its neighborhood.
        e=(heat.clamp(-12,10)*1.5).exp()
        neighborhood=F.avg_pool2d(e,5,1,2,divisor_override=1)
        mass=heat.sigmoid()*e/neighborhood.clamp_min(1e-8)*(2*delta.float().tanh()).exp()
        attrs=[a.softmax(1).flatten(2) for a in raw[:,5:].split((2,8,2,3),1)]
        sz,col,mat,sh=attrs
        left=(sz[:,:,None]*col[:,None]).flatten(1,2)
        right=(mat[:,:,None]*sh[:,None]).flatten(1,2)
        histogram=((left*mass.flatten(2))@right.transpose(1,2)).flatten(1)
        return (histogram,None) if local else histogram


class TileCounter(nn.Module):
    """Regress the class histogram in each of 4 x 6 image regions, then add."""
    def __init__(self):
        super().__init__()
        from torchvision.models import convnext_tiny
        backbone=convnext_tiny(weights=None)
        backbone.load_state_dict(v.load(EXTRA/"convnext_tiny-983f1562.pth"))
        self.net=backbone.features
        self.laterals=nn.ModuleList(nn.Conv2d(width,64,1) for width in (96,192,384,768))
        self.readout=nn.Sequential(nn.Conv2d(64,96,3,padding=1),nn.GroupNorm(8,96),nn.GELU(),
                                   nn.Conv2d(96,97,1))
        nn.init.constant_(self.readout[-1].bias[0],-5.4)
        self.register_buffer("mean",torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer("std",torch.tensor([.229,.224,.225])[None,:,None,None])

    def forward(self,rgb,local=False):
        x=F.interpolate((rgb-self.mean)/self.std,size=(256,384),mode="bilinear",align_corners=False)
        with amp(rgb.device):
            levels=[]
            for i,layer in enumerate(self.net):
                x=layer(x)
                if i%2:
                    feat=self.laterals[len(levels)](x)
                    levels.append(F.interpolate(feat,size=(32,48),mode="bilinear",align_corners=False))
            raw=self.readout(sum(levels)).float()
        density=F.softplus(raw[:,:1])*raw[:,1:].softmax(1)
        tiles=F.avg_pool2d(density,8,8,divisor_override=1)
        hist=tiles.sum((-2,-1))
        return (hist,tiles) if local else hist


def tile_loss(pred,instance,label):
    # Each visible pixel gets 1/area of its instance. Its contributions across all
    # 24 regions sum to one even if the silhouette spans multiple regions.
    b,h,w=instance.shape
    ids=instance.flatten(1)
    num=label.shape[1]
    area=pred.new_zeros(b,num).scatter_add_(1,ids,torch.ones_like(ids,dtype=pred.dtype))
    known=(label>=0).all(-1)
    sz,col,joint=label.clamp_min(0).unbind(-1)
    classes=(sz*8+col)*6+joint
    region=(torch.arange(h,device=ids.device)[:,None]*4//h)*6+torch.arange(w,device=ids.device)*6//w
    slot=24*classes.gather(1,ids)+region.flatten()
    weights=(known.gather(1,ids)&(ids!=0))/area.gather(1,ids).clamp_min(1)
    target=pred.new_zeros(b,96*24).scatter_add_(1,slot,weights).view_as(pred)
    mass=pred.sum(1).clamp_min(1e-6)
    total=target.sum(1)
    # Weak train masks are incomplete. Give locations with observed objects
    # stronger supervision than nominal background.
    valid_weight=torch.where(total>.01,1.,.12)
    count=(F.smooth_l1_loss(mass,total,beta=.15,reduction="none")*valid_weight).sum()/valid_weight.sum()
    classes=-(target*(pred.clamp_min(1e-9).log()-mass[:,None].log())).sum()/total.sum().clamp_min(1)
    return count+classes


def truth(hist,table):
    ma,mb,op,num=table
    a=(hist*ma).sum(-1)
    b=(hist*mb).sum(-1)
    diff=a-b
    # Signed distance to the boundary of the integer decision, in units of objects.
    # A discrete Gaussian count likelihood (sigma .33). Its normalizer cancels
    # when comparing different numbers for the same filter.
    value=-1.125*(a-num).square()
    for code,case in ((MORE,diff-.5),(FEWER,-diff-.5),(SAME,.5-diff.abs()),(EXIST,a-.5),(NONE,.5-a)):
        value=torch.where(op==code,case,value)
    return value*4


def caption_scores(hist,idx,table):
    a=truth(hist.repeat_interleave(2,dim=0),[t[idx] for t in table]).view(-1,2)
    return a[:,0]-a[:,1]


def fit(seed,device,epochs,architecture="r50", *, scenes=None, initial=None, batch=None, rate_scale=1.):
    if scenes is None:
        scenes=Scenes(device)
    frame=pd.read_csv(v.DATA/"train.csv")
    names,rows,scene,table=queries(frame,device)
    with np.load(EXTRA/"fragment_train.npz") as a:
        assert np.array_equal(names,a["names"])
    target=torch.tensor(frame.label.to_numpy()[rows[:,0]],device=device,dtype=torch.float)
    model=(Counter() if architecture=="r50" else TileCounter()).to(device,memory_format=torch.channels_last)
    if initial is not None:
        model.load_state_dict(v.load(initial))
    for m in model.net.modules():
        if isinstance(m,nn.BatchNorm2d):m.eval()
    extra=[p for name,p in model.named_parameters() if not name.startswith("net.")]
    opt=torch.optim.AdamW([dict(params=model.net.parameters(),lr=.00006*rate_scale),
                           dict(params=extra,lr=.0006*rate_scale)])
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    def loss(idx):
        rgb,instance,_,labels,added=scenes.pair(idx)
        hist,tiles=model(rgb,local=True)
        before,after=hist.chunk(2)
        # Under augmentation only the inserted donor may change a count. Its class is known.
        sz,col,joint=labels[:len(idx),-1].T
        kind=(sz*8+col)*6+joint
        active=added.flatten(1).sum(1)>=4
        change=F.one_hot(kind,96).float()*active[:,None]
        counts=after-before
        # Object presence can change a count even when the original scene has no exact annotation.
        # Borderline insertions have no reliable target, skip these in the paired objective.
        usable=active | ~(rgb[:len(idx)]!=rgb[len(idx):]).flatten(1).any(1)
        paired_error=F.smooth_l1_loss(counts,change,reduction="none",beta=.15)
        pair=((paired_error*(1+10*change)).sum(1)*usable).sum()/usable.sum().clamp_min(1)
        si,q=(idx[:,None]==scene[None]).nonzero(as_tuple=True)
        auxiliary=.28*pair
        if tiles is not None:
            auxiliary=auxiliary+.6*tile_loss(tiles,instance,labels)
        if not len(q):return auxiliary
        picks=(2*q[:,None]+torch.arange(2,device=device)).flatten()
        logits=caption_scores(before[si],picks,table)
        factual=F.binary_cross_entropy_with_logits(logits,target[q])
        # True exact counts also constrain the absolute scale, including between adjacent twin numbers.
        chosen=2*q+(1-target[q]).long()
        exact=table[2][chosen]==EXACT
        count=(before[si]*table[0][chosen]).sum(1)
        error=F.smooth_l1_loss(count,table[3][chosen].float(),reduction="none",beta=.25)
        return factual+.6*(error*exact).sum()/exact.sum().clamp_min(1)+auxiliary
    def probe(n):
        try:scaler.scale(loss(torch.arange(n,device=device)%len(names))).backward()
        finally:opt.zero_grad(set_to_none=True)
    if batch is None:
        batch=tune_batch("paired count", (96,64,48,32,16,8) if device.type=="cuda" else (2,),probe,device)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    for epoch in range(epochs):
        losses=[]
        for idx in torch.randperm(len(names),device=device).split(batch):
            opt.zero_grad(set_to_none=True)
            objective=loss(idx)
            if not torch.isfinite(objective):raise ValueError("Non-finite loss")
            scaler.scale(objective).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(opt);scaler.update()
            losses.append(objective.detach())
        sched.step()
        print("epoch",epoch+1,"mean loss",float(torch.stack(losses).mean()),flush=True)
    torch.save(model.state_dict(),v.OUT/"counter.pt")


@torch.inference_mode()
def infer(split,seed,device,initial=False,architecture="r50"):
    frame=pd.read_csv(v.DATA/f"{split}.csv")
    names,rows,scene,table=queries(frame,device)
    model=(Counter() if architecture=="r50" else TileCounter()).to(device,memory_format=torch.channels_last).eval()
    if not initial:model.load_state_dict(v.load(v.OUT/"counter.pt"))
    hist=[]
    for start in range(0,len(names),64):
        rgb=torch.from_numpy(images(split,names[start:start+64])).to(device).permute(0,3,1,2).float()/255
        hist.append(model(rgb))
    hist=torch.cat(hist)
    logits=caption_scores(hist[scene],torch.arange(2*len(scene),device=device),table).clamp(-8,8).cpu().numpy()
    score=np.zeros(len(frame));score[rows[:,0]]=logits;score[rows[:,1]]=-logits
    active=frame.category.isin(ACTIVE)
    assert set(np.flatnonzero(active))<=set(rows.ravel()) and np.isfinite(score).all()
    np.savez(v.OUT/f"control_{seed}_{split}.npz",ids=frame.id.to_numpy(),score=score)
    np.savez_compressed(v.OUT/f"counts_{split}.npz",names=names.astype(str),hist=hist.cpu().numpy())


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("stage",choices=["fit","val","test","initial"])
    cli.add_argument("seed",type=int)
    cli.add_argument("--architecture",choices=["r50","convnext"],default="r50")
    cli.add_argument("--epochs",type=int,default=18)
    args=cli.parse_args();_,device=v.setup(args.seed);v.OUT.mkdir(exist_ok=True)
    if args.stage=="fit":fit(args.seed,device,args.epochs,args.architecture)
    else:infer("val" if args.stage=="initial" else args.stage,args.seed,device,args.stage=="initial",args.architecture)
