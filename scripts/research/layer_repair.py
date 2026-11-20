"""Train occlusion repair with original target pixels and differentiable caption counts."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit, softmax
import torch
from torch import nn
from torch.nn import functional as F

from scripts.research import rear_candidate as rear
from scripts.research.fragment_votes import Scenes, EXTRA, v
from kaggle_suite_common import amp
from native_centers import GROUPS, canonical_form, parse_program
from src.models.slot_program import EXACT, FEWER, MORE, SAME
from src.scene_posterior import _draws, _word_mask
from src.twins import find_twins

OUT=Path("layer_result")
SCALE=rear.SCALE
SIDE=rear.SIDE
ARMS=("verify","split")
SIZES=(2,8,6)
ATTR_CODES=np.asarray([(1<<a)|(1<<b)|(1<<c)|(1<<d)
                       for a in GROUPS[0] for b in GROUPS[1] for c in GROUPS[2] for d in GROUPS[3]])


def frames(front, candidate, arm):
    if arm=="verify":
        grid = rear.relative_grid(rear.view_grid(front.device),front,candidate)
    else:
        center=front[:,:2]+front[:,2:]*front.new_tensor([.5,.25])
        size=front[:,2:].amax(1,keepdim=True).clamp_min(16.)*1.90
        grid=center[:,None,None]+(rear.view_grid(front.device)-.5)*size[:,None,None]
    return grid


def prompts(image,grid,front,candidate,arm):
    mark0=rear.box_prompt(grid,front)
    mark1=rear.box_prompt(grid,candidate) if arm=="verify" else torch.zeros_like(mark0)
    return torch.cat((image,mark0[:,None],mark1[:,None]),1)


def crop(rgb,front,candidate,arm):
    grid=frames(front,candidate,arm)
    screen=grid/grid.new_tensor([v.WIDTH,v.HEIGHT])
    image=F.grid_sample(rgb.float()/255,2*screen-1,align_corners=False,padding_mode="border")
    return prompts(image,grid,front,candidate,arm),grid


class LayerNet(nn.Module):
    def __init__(self,arm):
        super().__init__()
        self.arm=arm

        def block(cin,cout,stride):
            return nn.Sequential(nn.Conv2d(cin,cout,3,stride,1,bias=False),
                                 nn.GroupNorm(8,cout),nn.SiLU(),
                                 nn.Conv2d(cout,cout,3,1,1,bias=False),nn.GroupNorm(8,cout),nn.SiLU())
        self.fine=block(5,32,1)
        self.mid=block(32,48,2)
        self.context=nn.Sequential(block(48,64,2),block(64,96,2))
        self.lateral=nn.Conv2d(48,24,1)
        self.mask=nn.Conv2d(56,3,1)  # front, visible rear and full rear
        self.readout=nn.Sequential(nn.Linear(96*4+3*4,128),nn.SiLU(),nn.Dropout(.10),nn.Linear(128,21))
        with torch.no_grad():
            self.readout[-1].bias[0]=-2.5  # an extra object is sparse in real scenes

    def forward(self,x):
        a=self.fine(x)
        b=self.mid(a)
        features=self.context(b)
        masks=self.mask(torch.cat((a,F.interpolate(self.lateral(b),scale_factor=2,mode="bilinear",
                                                  align_corners=False)),1)).float()
        view=torch.cat((F.adaptive_avg_pool2d(features,(2,2)).flatten(1),
                        F.adaptive_avg_pool2d(masks.sigmoid(),(2,2)).flatten(1)),1)
        return self.readout(view).float(),masks


class Occlude:
    """Paste the foreground over real target pixels. Positive and negative share the occluder."""
    def __init__(self,device,rgb,background):
        self.data=Scenes(device,rgb=rgb)
        self.bg=torch.from_numpy(background).to(device=device,dtype=torch.float32).permute(2,0,1)[None]/255
        s=self.data
        bank=s.labels[s.donors[:,0],s.donors[:,1],1]
        classes=[torch.where(bank==k)[0] for k in range(8)]
        width=max(map(len,classes))
        self.by_color=torch.stack([F.pad(c,(0,width-len(c))) for c in classes])
        self.count=torch.tensor([len(c) for c in classes],device=device)

    def paired(self,idx,arm):
        s=self.data
        device=idx.device
        n=len(idx)
        # An original target has reliable appearance and contour labels.
        candidates=s.anchors[idx]&(s.labels[idx]>=0).all(-1)
        tid=(torch.rand(candidates.shape,device=device)*candidates).argmax(1)
        target=s.box[idx,tid]
        color=s.labels[idx,tid,1].clamp_min(0)
        matched=self.by_color[color,(torch.rand(n,device=device)*self.count[color]).long()]
        pick=torch.where((torch.rand(n,device=device)<.70)&(self.count[color]>0),matched,
                         torch.randint(len(s.donors),(n,),device=device))
        src,j=s.donors[pick].T
        source=s.box[src,j]
        extent=source[:,2:]*(.85+.37*torch.rand(n,1,device=device))
        center=target[:,:2]+.5*target[:,2:]
        jitter=(torch.rand(n,2,device=device)-.5)*target[:,2:]*.90
        jitter[:,1]=jitter[:,1].abs()+.05*target[:,3]
        front=torch.cat((center+jitter-.5*extent,extent),1)
        candidate=target.clone()
        candidate[:,:2]+=(torch.rand(n,2,device=device)-.5)*target[:,2:]*.30
        orig,grid=crop(s.rgb[idx],front,candidate,arm)
        screen=grid/grid.new_tensor([v.WIDTH,v.HEIGHT])
        owner=F.grid_sample(s.instances[idx,None].float(),2*screen-1,mode="nearest",align_corners=False)[:,0]
        # Use the real contour as the target mask and keep the original RGB in the positive.
        target_mask=(owner==tid[:,None,None]+1).float()
        u=(grid-front[:,None,None,:2])/extent[:,None,None]
        alpha=F.grid_sample(s.masks[src,j,None].float()/255,2*u-1,align_corners=False)[:,0]
        observed=target_mask*(1-alpha)
        fraction=observed.sum((-1,-2))/target_mask.sum((-1,-2)).clamp_min(1)
        usable=candidates.any(1)&(src!=idx)&(fraction>=.035)&(fraction<=.57)&(observed.sum((-1,-2))>=7)
        usable&=(target_mask.sum((-1,-2))>=60)
        texture=(source[:,None,None,:2]+u*source[:,None,None,2:])/grid.new_tensor([v.WIDTH,v.HEIGHT])
        occ=F.grid_sample(s.rgb[src].float()/255,2*texture-1,align_corners=False)
        bg=F.grid_sample(self.bg.expand(n,-1,-1,-1),2*screen-1,align_corners=False,padding_mode="border")
        # Match background brightness to the scene using unoccupied pixels.
        clear=(owner==0).float()[:,None]
        shift=((orig[:,:3]-bg)*clear).sum((-1,-2),keepdim=True)/clear.sum((-1,-2),keepdim=True).clamp_min(1)
        fill=(bg+shift.clamp(-.12,.12)).clamp(0,1)
        erased=orig[:,:3]+(fill-orig[:,:3])*target_mask[:,None]
        trans=alpha[:,None]
        positive=orig.clone()
        negative=orig.clone()
        positive[:,:3]=(1-trans)*orig[:,:3]+trans*occ
        negative[:,:3]=(1-trans)*erased+trans*occ
        frame_size=(grid[:,-1,-1]-grid[:,0,0])*SIDE/(SIDE-1)
        corner=grid[:,0,0]-.5*frame_size/SIDE
        box=torch.cat(((center-corner)/frame_size,
                       (target[:,2:]/frame_size).clamp(.01,1)),1)
        masks=torch.stack((alpha,observed,target_mask),1)
        null=masks.clone();null[:,1:]=0
        return positive,negative,masks,null,s.labels[idx,tid],box,usable,fraction


def plan(table,p,arm):
    n,limit=table["pairs"].shape[:2]
    out=table["geometry"].copy()
    slots=table["pairs"][...,0].copy()
    valid=slots>=0
    if arm=="split":
        valid[:]=False
        slots[:]=-1
        for i in range(n):
            chosen=np.flatnonzero(table["valid"][i]&(p[i]>.85))[:limit]
            for j,k in enumerate(chosen):
                box=table["x"][i,k,v.BOX]*SCALE
                out[i,j,0]=box
                out[i,j,1]=box
                valid[i,j]=True
    return out,slots,valid


def withheld_pairs(table,p,windows,slots,valid):
    """Hide a confident rear detection from the real-scene table used for training."""
    n,limit=slots.shape
    extra=np.zeros((n,2,2,4),np.float32)
    omitted=np.full((n,2),-1,np.int16)
    for i in range(n):
        confident=np.flatnonzero(table["valid"][i]&(p[i]>.88))
        if len(confident)<2:
            continue
        boxes=table["x"][i,confident,v.BOX]*SCALE
        low=boxes[:,:2]
        high=low+boxes[:,2:]
        center=(low+high)/2
        extent=boxes[:,2:].prod(1)
        common=(np.minimum(high[:,None],high[None])-np.maximum(low[:,None],low[None])).clip(0).prod(2)
        cover=common/extent[:,None].clip(1)
        distance=np.linalg.norm(center[:,None]-center[None],axis=2)
        foot=low[:,1]+.88*boxes[:,3]
        eligible=(cover>.12)&(cover<.88)&(foot[:,None]+5<foot[None,:])&(distance>=6)
        pairs=np.argwhere(eligible)
        if len(pairs):
            order=np.argsort(-cover[eligible],kind="stable")
            for k,(a,b) in enumerate(pairs[order[:2]]):
                extra[i,k]=(boxes[a],boxes[b])
                omitted[i,k]=confident[a]
    print("withheld confident rear proposals",int((omitted>=0).sum()),flush=True)
    return (np.concatenate((windows,extra),axis=1),np.concatenate((slots,omitted),axis=1),
            np.concatenate((valid,omitted>=0),axis=1))


def marginals(frame,table,p,attrs,slots,eligible):
    """Expected truth with each possible added class, using original train captions."""
    ids={name:i for i,name in enumerate(table["names"])}
    rows=frame[frame.category.isin(["counting","comparison"])]
    jobs=[]
    for name,group in rows.groupby("image",sort=False):
        i=ids[name]
        if not eligible[i].any():
            continue
        use=table["valid"][i]
        u=_draws(int(use.sum()),8,421,256)
        present=u[...,0]<p[i,use]
        codes=np.zeros_like(present,dtype=np.int32)
        for j,g in enumerate(GROUPS):
            chosen=(u[...,j+1,None]>attrs[j][i,use].cumsum(-1)).sum(-1).clip(max=len(g)-1)
            codes|=1<<g[chosen]
        index=np.cumsum(use)-1
        for row in group.itertuples(index=False):
            program=parse_program(canonical_form(row.caption,row.category))
            if program is None or program.relation or program.op not in (EXACT,FEWER,MORE,SAME):
                continue
            wa,wb=_word_mask(program.a),_word_mask(program.b)
            ca=present&((codes&wa)==wa)
            cb=present&((codes&wb)==wb)
            extra_a=(ATTR_CODES&wa)==wa
            extra_b=(ATTR_CODES&wb)==wb
            def evaluate(a,b):
                if program.op==EXACT:return a==program.number
                if program.op==MORE:return a>b
                if program.op==FEWER:return a<b
                return a==b
            for j in np.flatnonzero(eligible[i]):
                a,b=ca.sum(1),cb.sum(1)
                slot=slots[i,j]
                if slot>=0:
                    a-=ca[:,index[slot]]
                    b-=cb[:,index[slot]]
                prior=evaluate(a,b).mean()
                gain=evaluate(a[:,None]+extra_a,b[:,None]+extra_b).mean(0)-prior
                # A caption that cannot see this hypothesis cannot train this candidate.
                if np.abs(gain).max()>=.075:
                    jobs.append((i,j,row.label,prior,gain.astype(np.float32)))
    if not jobs:
        return None
    scene,roi,label,prior,gain=zip(*jobs)
    return np.asarray(scene),np.asarray(roi),np.c_[label,prior,np.stack(gain)].astype(np.float32)


def joint_classes(pred):
    sz,col,cs=[s.softmax(-1) for s in pred[:,1:17].split(SIZES,1)]
    product=(sz[:,:,None,None]*col[:,None,:,None]*cs[:,None,None,:]).flatten(1)
    return sz,col,cs,product


def benchmark(probe,device):
    if device.type!="cuda":return 8
    winners=[]
    for n in (512,384,256,192,128,96,64,32):
        peak=0
        times=[]
        torch.cuda.reset_peak_memory_stats(device)
        try:
            for i in range(3):
                torch.cuda.synchronize(device);start=time.monotonic()
                probe(n)
                torch.cuda.synchronize(device)
                if i:times.append(time.monotonic()-start)
            peak=torch.cuda.max_memory_allocated(device)
            rate=n/np.median(times)
            print("bench",n,round(rate,1),"pairs/s",round(peak/2**30,2),"GiB",flush=True)
            if peak<.87*torch.cuda.get_device_properties(device).total_memory:winners.append((rate,n))
        except torch.cuda.OutOfMemoryError:
            print("OOM",n,flush=True)
        torch.cuda.empty_cache()
    if not winners:raise RuntimeError("All batch probes exceeded the memory budget")
    return max(winners)[1]


def fit(arm,seed,device,epochs):
    table=rear.collect("train")
    with np.load(EXTRA/"fragment_train.npz") as z:
        assert np.array_equal(z["names"],table["names"])
    pixels=np.empty((len(table["names"]),v.HEIGHT,v.WIDTH,3),dtype=np.uint8)
    for i in (0,1):pixels[i::2]=np.load(rear.SCRATCH/f"train_{i}_rgb.npy",mmap_mode="r")
    dataset=Occlude(device,pixels,np.load(EXTRA/"train_background.npy"))
    del pixels
    p,_,_,attrs,_=rear.bank_parts(table,"train",device)
    windows,slots,valid=plan(table,p,arm)
    windows,slots,valid=withheld_pairs(table,p,windows,slots,valid)
    constraints=marginals(pd.read_csv(v.DATA/"train.csv"),table,p,attrs,slots,valid)
    if constraints is None:raise ValueError("No informative training captions")
    scene,roi,truth=constraints
    sign=2*truth[:,0]-1
    correct=np.where(sign>0,truth[:,1],1-truth[:,1])
    improves=(sign[:,None]*truth[:,2:]).max(1)
    relevant=np.flatnonzero((correct<.70)&(improves>.15))
    print("train caption observations",len(truth),"unique scenes",len(np.unique(scene)),
          "observations with a repairable count deficit",len(relevant),flush=True)
    windows=torch.from_numpy(windows).to(device)
    scene,roi,truth=[torch.from_numpy(x).to(device) for x in (scene,roi,truth)]
    relevant=torch.from_numpy(relevant).to(device)
    model=LayerNet(arm).to(device,memory_format=torch.channels_last)
    opt=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.001)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda",init_scale=128.)
    factor=0.
    def objective(idx):
        x,z,m,n,labels,box,keep,visibility=dataset.paired(idx,arm)
        with amp(device):
            pred,mask=model(torch.cat((x,z)).contiguous(memory_format=torch.channels_last))
        positive,negative=pred.chunk(2)
        keep=keep.float()
        denom=keep.sum().clamp_min(1)
        pair=(F.softplus(-positive[:,0])+F.softplus(negative[:,0])
              +.25*F.softplus(1-positive[:,0]+negative[:,0]))
        loss=(pair*keep).sum()/denom
        mask_target=torch.cat((m,n))
        masks=F.binary_cross_entropy_with_logits(mask,mask_target,reduction="none")*(1+4*mask_target)
        loss+=.50*(masks.mean((1,2,3))*keep.repeat(2)).sum()/(2*denom)
        for k,logits in enumerate(positive[:,1:17].split(SIZES,1)):
            ce=F.cross_entropy(logits,labels[:,k],ignore_index=-1,reduction="none")
            loss+=.16*(ce*keep).sum()/denom
        loss+=.65*(F.smooth_l1_loss(positive[:,17:].sigmoid(),box.clamp(0,1),
                                      reduction="none",beta=.12).sum(1)*keep).sum()/denom
        # Train the same readout through the effect of its hypothesis on natural train captions.
        real_batch=max(8,len(idx)//2)
        k=torch.randint(len(scene),(real_batch,),device=device)
        if len(relevant):
            half=real_batch//2
            k[:half]=relevant[torch.randint(len(relevant),(half,),device=device)]
        i,j=scene[k],roi[k]
        rear_box,front_box=windows[i,j].unbind(1)
        real,_=crop(dataset.data.rgb[i],front_box,rear_box,arm)
        with amp(device):
            real_score,_=model(real.contiguous(memory_format=torch.channels_last))
        *_,distribution=joint_classes(real_score)
        label,prior=truth[k,:2].T
        gain=(truth[k,2:]*distribution).sum(-1)
        probability=.01+.98*(prior+real_score[:,0].sigmoid()*gain).clamp(0,1)
        loss+=factor*F.binary_cross_entropy(probability,label)
        return loss,keep.sum(),((visibility<.20)*keep).sum()

    def probe(n):
        try:
            scaler.scale(objective(torch.arange(n,device=device)%len(p))[0]).backward()
        finally:
            opt.zero_grad(set_to_none=True)
    with torch.random.fork_rng(devices=[0] if device.type=="cuda" else []):
        batch=benchmark(probe,device)
    print("selected",batch,"pairs per GPU",flush=True)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=.00008)
    for epoch in range(epochs):
        factor=.70*min(1.,epoch/6)
        sums=[]
        start=time.monotonic()
        for idx in torch.randperm(len(p),device=device).split(batch):
            opt.zero_grad(set_to_none=True)
            loss,n,hard=objective(idx)
            if not torch.isfinite(loss):raise ValueError("Non-finite model loss")
            scaler.scale(loss).backward();scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(opt);scaler.update()
            sums.append(torch.stack((loss.detach(),n,hard)))
        sched.step()
        stats=torch.stack(sums).sum(0).tolist()
        print(arm,seed,epoch+1,"loss",round(stats[0]/len(sums),4),
              "pairs",int(stats[1]),"visibility below 20%",int(stats[2]),
              "seconds",round(time.monotonic()-start,2),flush=True)
    torch.save(model.state_dict(),OUT/f"{arm}_{seed}.pt")


def material_priors(attrs):
    return (attrs[2][:,:,None]*attrs[3][:,None,:]).reshape(-1,6)


def change_score(programs, twins, p, attributes, positions, offsets, candidate, score, box):
    joint=material_priors(attributes)
    u=_draws(len(p)+int(candidate<0),8,41,2048)
    if candidate<0:
        candidate=len(p)
        p=np.r_[p,0.]
        attributes=[np.vstack([g,np.full(g.shape[1],1/g.shape[1])]) for g in attributes]
        joint=np.vstack([joint,np.full(6,1/6)])
        position=(box[:2]+box[2:]*[.5,.88])/SCALE[:2]*2-1
        positions=np.vstack([positions,position])
    baseline=rear.decode(programs,twins,p,attributes,joint,positions,offsets,u)
    presence=float(expit(score[0]))
    p[candidate]=.95*presence
    for group,old in zip(np.split(score[1:17],[2,10]),[attributes[0],attributes[1],joint]):
        old[candidate]=.25*old[candidate]+.75*softmax(group)
    return rear.decode(programs,twins,p,attributes,joint,positions,offsets,u)-baseline


@torch.inference_mode()
def predict(arm,split,seed,shard,device):
    frame=pd.read_csv(v.DATA/f"{split}.csv")
    group=frame.groupby("image",sort=False).indices
    captions=[canonical_form(row.caption,row.category) for row in frame.itertuples()]
    table=rear.collect(split,shard)
    p,xy,valid,attrs,offsets=rear.bank_parts(table,split,device)
    windows,slots,available=plan(table,p,arm)
    rgb=np.load(rear.SCRATCH/f"{split}_{shard}_rgb.npy",mmap_mode="r")
    model=LayerNet(arm).to(device,memory_format=torch.channels_last).eval()
    model.load_state_dict(v.load(OUT/f"{arm}_{seed}.pt"))
    si,sj=np.nonzero(available)
    predictions=[[],[],[],[]]
    background=torch.from_numpy(np.load(EXTRA/"train_background.npy")).to(device,dtype=torch.float32)
    background=background.permute(2,0,1)[None]/255
    for step in range(0,len(si),384 if device.type=="cuda" else 32):
        sl=slice(step,step+(384 if device.type=="cuda" else 32))
        pixels=torch.from_numpy(rgb[si[sl]]).to(device).permute(0,3,1,2)
        rear_box,front_box=torch.from_numpy(windows[si[sl],sj[sl]]).to(device).unbind(1)
        x,grid=crop(pixels,front_box,rear_box,arm)
        with amp(device):
            raw,masks=model(x.contiguous(memory_format=torch.channels_last))
        span=(grid[:,-1,-1]-grid[:,0,0])*SIDE/(SIDE-1)
        low=grid[:,0,0]-.5*span/SIDE
        ctr,size=raw[:,17:].sigmoid().split(2,1)
        recovered=torch.cat((low+(ctr-.5*size)*span,size*span),1)
        # Delete only the pixels attributed to the visible rear fragment while
        # holding the box prompts and the predicted foreground fixed.
        rear_pixels=(masks[:,1].sigmoid()>.55)&(masks[:,0].sigmoid()<.45)
        visible=rear_pixels.sum((-1,-2))
        inspect=(raw[:,0].sigmoid()>=.87)&(visible>=5)
        contrast=torch.zeros(len(raw),device=device)
        if inspect.any():
            points=grid[inspect]/grid.new_tensor([v.WIDTH,v.HEIGHT])
            blank=F.grid_sample(background.expand(len(points),-1,-1,-1),points*2-1,
                                padding_mode="border",align_corners=False)
            selected=x[inspect]
            keep=(masks[inspect,:2].sigmoid().amax(1)<.3).float()[:,None]
            adjust=((selected[:,:3]-blank)*keep).sum((-2,-1),keepdim=True)/keep.sum((-2,-1),keepdim=True).clamp_min(1)
            selected[:,:3]=torch.where(rear_pixels[inspect,None],(blank+adjust.clamp(-.12,.12)).clamp(0,1),selected[:,:3])
            with amp(device):
                erased,_=model(selected.contiguous(memory_format=torch.channels_last))
            contrast[inspect]=raw[inspect,0]-erased[:,0]
        predictions[0].append(raw.cpu().numpy())
        predictions[1].append(recovered.cpu().numpy())
        predictions[2].append(visible.cpu().numpy())
        predictions[3].append(contrast.cpu().numpy())
    output,boxes,support,contrast=[np.concatenate(x) if x else np.empty(0) for x in predictions]
    ids=[];effect=[];audit=[]
    for i,name in enumerate(table["names"]):
        candidates=np.flatnonzero(si==i)
        if not len(candidates):continue
        use=valid[i]
        plausible=[]
        for k in candidates:
            j=sj[k]
            b=windows[i,j,0] if arm=="verify" else boxes[k]
            slot=slots[i,j]
            if arm=="split":
                known=table["x"][i,:,v.BOX]*SCALE
                mid=b[:2]+.5*b[2:]
                distance=np.linalg.norm(known[:,:2]+.5*known[:,2:]-mid,axis=1)
                near=use&(distance<np.clip(.33*b[2:].max(),5,14))
                common=(np.minimum(known[:,:2]+known[:,2:],b[:2]+b[2:])-
                        np.maximum(known[:,:2],b[:2])).clip(0).prod(1)
                iou=common/(known[:,2:].prod(1)+b[2:].prod()-common).clip(1)
                # Close centers alone are expected under overlap. Suppress a
                # known object only when its box also matches closely.
                duplicate=(distance<np.clip(.16*b[2:].max(),3,7))&(iou>.60)
                if (near&duplicate&(p[i]>.52)).any():continue
                weak_match=near&(p[i]<=.52)
                if weak_match.any():slot=int(np.argmin(np.where(weak_match,distance,np.inf)))
                if not((mid>=0).all() and (mid<=SCALE[:2]).all() and b[2:].min()>=6):continue
            probability=float(expit(output[k,0]))
            if split=="val":
                audit.append(dict(image=str(name),probability=probability,contrast=float(contrast[k]),
                                  pixels=int(support[k]),box=b.tolist(),
                                  metal_cylinder=float(softmax(output[k,11:17])[-1])))
            if probability>=.87 and support[k]>=5 and contrast[k]>=.75:
                plausible.append((probability,k,slot,b))
        if not plausible:
            continue
        q,k,index,b=max(plausible,key=lambda row:row[0])
        if index>=0:
            if q*.95<=p[i,index]:continue
            index=int(np.cumsum(use)[index]-1)
        rows=group[name]
        pair=find_twins(frame.image.iloc[rows],[captions[j] for j in rows],frame.category.iloc[rows])
        programs=[parse_program(captions[j]) for j in rows]
        diff=change_score(programs,pair,p[i,use],[a[i,use] for a in attrs],xy[i,use],offsets,index,output[k],b)
        ids.extend(frame.id.iloc[rows]);effect.extend(diff)
    np.savez(OUT/f"{arm}_{seed}_{split}_{shard}.npz",ids=np.asarray(ids,dtype=np.int64),delta=np.asarray(effect))
    if split=="val":(OUT/f"{arm}_{seed}_{shard}_candidates.json").write_text(json.dumps(audit),encoding="utf8")
    print(arm,seed,split,shard,len(ids),"caption rows scored",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","fit","predict"))
    parser.add_argument("--arm",choices=ARMS,default="verify")
    parser.add_argument("--split",choices=("train","val","test"),default="train")
    parser.add_argument("--shard",type=int,choices=(0,1),default=0)
    parser.add_argument("--seed",type=int,default=421)
    parser.add_argument("--epochs",type=int,default=64)
    args=parser.parse_args()
    _,device=v.setup(args.seed)
    rear.SCRATCH.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    if args.stage=="prepare":rear.prepare(args.split,args.shard,device)
    elif args.stage=="fit":fit(args.arm,args.seed,device,args.epochs)
    else:predict(args.arm,args.split,args.seed,args.shard,device)
