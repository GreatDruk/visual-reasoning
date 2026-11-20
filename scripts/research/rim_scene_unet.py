"""Full-frame U-Net with independent object queries and visible-mask supervision."""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from scripts.research import detail_counts as d
from scripts.research.rim_fragments import OUT,block,gpu_util,load_bank,sync
from scripts.research.fragment_votes import images, v
from kaggle_suite_common import amp
from scripts.research.dense_counts import queries


class SceneUNet(d.MaskSet):
    def __init__(self):
        # Reuse the instance-query readout and matching with a train-only U-Net encoder.
        nn.Module.__init__(self)
        self.tokens=nn.Parameter(torch.randn(d.K,48)*.09)
        self.context=nn.Sequential(nn.Conv2d(48,48,3,stride=2,padding=1),nn.GroupNorm(8,48),nn.SiLU())
        self.pos=nn.Conv2d(2,48,1)
        self.cross=nn.MultiheadAttention(48,4,batch_first=True)
        self.self_attention=nn.MultiheadAttention(48,4,batch_first=True)
        self.norm=nn.LayerNorm(48)
        self.classify=nn.Linear(48,17)
        self.mask=nn.Linear(48,48)
        nn.init.constant_(self.classify.bias[0],-2)
        self.high=block(3,24)
        self.mid=nn.Sequential(block(24,40),block(40,40))
        self.low=nn.Sequential(block(40,64),block(64,64))
        self.deep=nn.Sequential(block(64,80),block(80,80))
        self.up2=block(80+64,64)
        self.up1=block(64+40+24,48)

    def features(self,rgb):
        p=F.avg_pool2d
        top=self.high(rgb)
        early=p(top,2)
        middle=self.mid(early)
        lower=self.low(p(middle,2))
        context=self.deep(p(lower,2))
        c=self.up2(torch.cat([lower,d.resize(context,lower.shape[-2:])],1))
        # Max pooling retains per-channel evidence from just one native pixel.
        return self.up1(torch.cat([F.max_pool2d(top,2),middle,d.resize(c,middle.shape[-2:])],1))


def fit(seed,device,epochs,limit, *, bank=None, batch=None, initial=None, rate_scale=1.):
    if bank is None:
        bank,_,startup=load_bank(device)
    else:
        startup={}
    torch.manual_seed(seed)
    ids=torch.arange(len(bank.rgb),device=device)[:limit]
    model=SceneUNet().to(device,memory_format=torch.channels_last)
    opt=torch.optim.AdamW(model.parameters(),lr=.0007,fused=device.type=="cuda")
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda",init_scale=128.)
    candidates=[]
    def step(idx,profile=False):
        start=time.perf_counter()
        rgb,instance,_,labels,added=bank.pair(idx,tiny_fragments=True)
        host=time.perf_counter()-start
        if profile:
            sync(device)
        prep=time.perf_counter()-start
        opt.zero_grad(set_to_none=True)
        with amp(device):
            raw,mask=model(rgb.contiguous(memory_format=torch.channels_last))
        timing={} if profile else None
        loss=d.mask_loss(raw,mask,instance,labels[:,1:],timing)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if profile:
            sync(device)
        return loss.detach(), added.sum(), prep,host,time.perf_counter()-start,timing

    widths=(8,16,32,48,64,96,128) if device.type=="cuda" else (min(2,len(ids)),)
    for n in (widths if batch is None else ()):
        if device.type=="cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            idx=ids[torch.arange(n,device=device)%len(ids)]
            step(idx,True)
            _,_,prep,host,total,timing=step(idx,True)
            mem=torch.cuda.max_memory_reserved() if device.type=="cuda" else 0
            cap=torch.cuda.get_device_properties(device).total_memory if device.type=="cuda" else 1
            candidates.append(dict(batch=n,crops_s=2*n/total,host_ms=1000*host,
                                   prepare_ms=1000*prep,reserved_gib=mem/2**30,vram_fraction=mem/cap,gpu=gpu_util(device),**timing))
            print(candidates[-1],flush=True)
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            print("scene-unet batch",n,"OOM",flush=True)
    if batch is None:
        viable=[r for r in candidates if r["vram_fraction"]<.91]
        if not viable:
            raise RuntimeError("Scene U-Net does not fit")
        fastest=max(v["crops_s"] for v in viable)
        batch=max(v["batch"] for v in viable if v["crops_s"]>=.97*fastest)
    torch.manual_seed(seed)
    model=SceneUNet().to(device,memory_format=torch.channels_last)
    if initial is not None:
        model.load_state_dict(v.load(initial))
    opt=torch.optim.AdamW(model.parameters(),lr=.0007*rate_scale,fused=device.type=="cuda")
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda",init_scale=128.)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=.00004*rate_scale)
    history=[]
    for ep in range(epochs):
        t=time.perf_counter()
        stats=[]
        for idx in ids[torch.randperm(len(ids),device=device)].split(batch):
            stats.append(step(idx)[:2])
        loss=torch.stack([s[0] for s in stats]).mean().item()
        inserted=torch.stack([s[1] for s in stats]).sum().item()
        row=dict(epoch=ep+1,loss=loss,inserted_pixels=inserted,elapsed_s=time.perf_counter()-t,gpu=gpu_util(device))
        if not np.isfinite(loss):
            raise ValueError("Non-finite training objective")
        history.append(row)
        schedule.step()
        print("scene-unet",seed,row,flush=True)
    torch.save(model.state_dict(),OUT/f"scene_unet_{seed}.pt")
    (OUT/f"scene_unet_{seed}.json").write_text(json.dumps(dict(batch=batch,probes=candidates,history=history,startup=startup)))


def compare_difference(members,ma,mb,op):
    # One instance has one joint class. Shared membership cancels in A-B.
    a=(members*(ma&~mb)[:,None]).sum(-1)
    b=(members*(mb&~ma)[:,None]).sum(-1)
    p=F.one_hot(torch.full_like(op,d.K),2*d.K+1).float()
    for x,y in zip(a.T,b.T):
        p=p*(1-x-y).clamp_min(0)[:,None]+F.pad(p[:,:-1],(1,0))*x[:,None]+F.pad(p[:,1:],(0,1))*y[:,None]
    val=torch.where(op==d.MORE,p[:,d.K+1:].sum(1),p[:,:d.K].sum(1))
    return torch.where(op==d.SAME,p[:,d.K],val)


@torch.inference_mode()
def infer(seed,device,split,limit,tag="scene_unet",output=OUT,frame=None,checkpoint=None,rgb_cache=None):
    if frame is None:
        frame=pd.read_csv(v.DATA/f"{split}.csv")
    if limit:
        frame=frame[frame.image.isin(frame.image.unique()[:limit])].reset_index(drop=True)
    ns,rows,si,table=queries(frame,device)
    ma,mb,op,num=table
    model=SceneUNet().to(device,memory_format=torch.channels_last).eval()
    model.load_state_dict(v.load(output/f"{tag}_{seed}.pt" if checkpoint is None else checkpoint))
    result=np.zeros(len(frame))
    proposals=[]
    batch=32 if device.type=="cuda" else 4
    for start in range(0,len(ns),batch):
        names=ns[start:start+batch]
        u=torch.where((si>=start)&(si<start+len(names)))[0]
        if not len(u):
            continue
        pixels=images(split,names) if rgb_cache is None else np.stack([rgb_cache[name] for name in names])
        rgb=torch.from_numpy(pixels).to(device).permute(0,3,1,2).float()/255
        with amp(device):
            objects,masks=model(rgb)
        m=d.mask_members(objects,masks)
        selected=(si[u]-start).repeat_interleave(2)
        query=(2*u[:,None]+torch.arange(2,device=device)).flatten()
        member=m[selected]
        a=d.poisson((member*ma[query,None]).sum(-1))
        b=d.poisson((member*mb[query,None]).sum(-1))
        main=d.statements(a,b,op[query],num[query])
        comp=compare_difference(member,ma[query],mb[query],op[query])
        cl=(op[query]==d.MORE)|(op[query]==d.FEWER)|(op[query]==d.SAME)
        main=torch.where(cl,torch.logit(comp.clamp(1e-5,1-1e-5)),main)
        s=main.reshape(-1,2).cpu().numpy()
        j=rows[u.cpu().numpy()]
        result[j[:,0]]=s[:,0]-s[:,1]
        result[j[:,1]]=-result[j[:,0]]
        if split=="val":
            proposals.extend(d.mask_boxes(names,objects,masks,include_attributes=True))
    np.savez_compressed(output/f"{tag}_{seed}_{split}.npz",ids=frame.id.to_numpy(),score=result)
    if split=="val":
        (output/f"{tag}_{seed}_objects.json").write_text(json.dumps(proposals))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("fit","val","test"))
    parser.add_argument("--seed",type=int,default=421)
    parser.add_argument("--epochs",type=int,default=24)
    parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    _,device=v.setup(args.seed)
    if args.stage=="fit":
        fit(args.seed,device,args.epochs,args.limit)
    else:
        infer(args.seed,device,args.stage,args.limit)
