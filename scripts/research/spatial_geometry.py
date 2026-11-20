"""Fit ground-plane relation uncertainty on training captions and replay frozen scenes."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from scripts.research import spatial_replay as replay
from scripts.research import rim_eval as rim
from src.models.slot_program import EXACT


def data(frame,names,field,target):
    lookup={name:i for i,name in enumerate(names)}
    queries=[]
    for j,r in enumerate(frame.itertuples(index=False)):
        if not target[j] or r.image not in lookup:
            continue
        p=rim.layer.parse_program(rim.layer.canonical_form(r.caption,r.category))
        if p is None or p.op!=EXACT or p.relation not in (1,2,3,4):
            continue
        modes=[np.concatenate([np.isin(group,words) if set(group)&set(words) else np.ones(len(group),bool)
                    for group in rim.GROUPS]) for words in (p.a,p.ref)]
        queries.append((j,lookup[r.image],p.relation,p.number,*modes))
    rows,idx,rel,num,subject,reference=map(np.asarray,zip(*queries))
    num=np.minimum(num,field["p"].shape[1])
    return (rows,torch.as_tensor(idx),torch.as_tensor(rel),torch.as_tensor(num),
            torch.as_tensor(subject),torch.as_tensor(reference))


def scene_tensors(fields):
    names=("p","x","xy","size","color","material","shape")
    return {k:torch.as_tensor(np.asarray(fields[k]),dtype=torch.float32) for k in names}


def exact(q,number):
    selected=torch.zeros(*q.shape[:2],q.shape[-1]+1,device=q.device)
    selected[:,:,0]=1
    for p in q.unbind(-1):
        selected=selected*(1-p[:,:,None])+F.pad(selected[:,:,:-1],(1,0))*p[:,:,None]
    return selected.gather(-1,number[:,None,None].expand(len(number),q.shape[1],1)).squeeze(2)


class Geometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift=nn.Parameter(torch.zeros(2,3))
        self.cross=nn.Parameter(torch.zeros(2))
        self.blend=nn.Parameter(torch.zeros(2))
        self.boundary=nn.Parameter(torch.zeros(4))
        self.width=nn.Parameter(torch.tensor([-3.0,-3.0]))

    def coords(self,x,xy,shape,learned):
        origin,extent=x.split(2,dim=-1)
        center=(origin+extent*x.new_tensor([.5,.88]))*2-1
        if not learned:
            return xy
        mix=.5+.4*self.blend.tanh()
        center=xy*(1-mix)+center*mix
        shift=shape@self.shift.tanh().T*.18
        scaled=(extent*shift)*2
        return center+scaled+center.flip(-1)*(.07*self.cross.tanh())

    def forward(self,f,rel,num,subject,reference,learned=True):
        p=f["p"].clamp(0.,1.)
        attrs=[f[k] for k in ("size","color","material","shape")]
        def test(mask):
            total=p
            for value,allowed in zip(attrs,mask.split(tuple(len(g) for g in rim.GROUPS),1)):
                total=total*(value*allowed[:,None]).sum(-1)
            return total
        sub,ref=test(subject),test(reference)
        xy=self.coords(f["x"],f["xy"],f["shape"],learned)
        axis=rel.ge(3).long()
        coord=xy.gather(-1,axis[:,None,None].expand(len(rel),xy.shape[1],1)).squeeze(-1)
        sign=torch.where((rel==1)|(rel==4),-1.,1.)
        distance=(coord[:,None,:]-coord[:,:,None])*sign[:,None,None]
        width=(self.width.exp()+.015)[axis] if learned else distance.new_full((len(rel),),.055)
        boundary=self.boundary[rel-1].tanh()*.08 if learned else torch.zeros_like(width)
        relation=((distance-boundary[:,None,None])/width[:,None,None]).sigmoid()
        relation=relation.masked_fill(torch.eye(xy.shape[1],dtype=torch.bool),0.)
        q=sub[:,None,:]*relation
        conditional=exact(q,num)
        weight=ref.square()
        any_ref=1-(1-ref).prod(-1)
        prob=(conditional*weight).sum(1)/weight.sum(1).clamp_min(1e-8)*any_ref
        return prob.clamp(1e-5,1-1e-5)


def pack(infile):
    with np.load(infile) as z:
        a={n:z[n] for n in z.files if n!="background"}
    a["x"]=a["features"][...,8:12]
    a["p"]=a["p"]*a["valid"]
    return a


def subset(f,idx):
    return {k:v[idx] for k,v in f.items()}


def fit(root,frame=None,teacher=None):
    if frame is None:
        frame=pd.read_csv(rim.v.DATA/"train.csv")
    a=pack(Path("artefacts/research/teacher_train.npz")) if teacher is None else teacher
    rows,idx,rel,num,sub,ref=data(frame,a["names"],a,frame.category.eq("spatial_counting").to_numpy())
    labels=torch.as_tensor(frame.label.to_numpy()[rows],dtype=torch.float32)
    fields=scene_tensors(a)
    model=Geometry()
    opt=torch.optim.AdamW(model.parameters(),lr=.025,weight_decay=.03)
    rng=torch.Generator().manual_seed(627)
    for ep in range(18):
        values=[]
        for selected in torch.randperm(len(rows),generator=rng).split(96):
            prob=model(subset(fields,idx[selected]),rel[selected],num[selected],sub[selected],ref[selected])
            loss=F.binary_cross_entropy(prob,labels[selected])+.02*sum(v.square().mean() for v in
                (model.shift,model.cross,model.blend,model.boundary))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            values.append(float(loss.detach()))
        if ep%5==4 or ep==17:
            print("geometry",ep+1,len(rows),np.mean(values),flush=True)
    torch.save(model.state_dict(),root/"spatial_geometry.pt")


@torch.inference_mode()
def infer(root,split,teacher=False):
    f,base,_,need=replay.frames(root,split)
    model=Geometry()
    model.load_state_dict(rim.v.load(root/"spatial_geometry.pt"))
    if teacher:
        q=pack(Path("artefacts/research/scene_silhouette.npz"))
        q["names"]=f.image.unique()
        tables=[q]
    else:
        tables=[]
        for z,(p,xy,valid,attr,off) in replay.tables_for(root,split):
            tables.append(dict(names=z["names"],p=p*valid,x=z["x"][...,rim.v.BOX],xy=xy,
                **dict(zip(("size","color","material","shape"),attr))))
    d=np.zeros(len(f))
    old=np.zeros(len(f))
    new=np.zeros(len(f))
    for tab in tables:
        r,idx,rel,num,sub,ref=data(f,tab["names"],tab,need)
        features=scene_tensors(tab)
        for segment in torch.arange(len(r)).split(96):
            z=subset(features,idx[segment])
            args=(z,rel[segment],num[segment],sub[segment],ref[segment])
            old[r[segment]]=torch.logit(model(*args,learned=False)).numpy()
            new[r[segment]]=torch.logit(model(*args)).numpy()
    # Compose the two probabilities only inside actual twin pairs.
    cap=[rim.layer.canonical_form(r.caption,r.category) for r in f.itertuples()]
    pairs=rim.layer.find_twins(f.image,cap,f.category)
    l,r=pairs.T
    selected=need[l]&need[r]
    l,r=l[selected],r[selected]
    for arr in (old,new):
        evidence=(arr[l]-arr[r])/2
        arr[l]=evidence
        arr[r]=-evidence
    d[need]=new[need]-old[need]
    where=root/"spatial"
    where.mkdir(exist_ok=True)
    key=f"{split}_geometry"+("_teacher" if teacher else "")
    np.savez_compressed(where/f"{key}.npz",ids=f.id.to_numpy(),base=old,score=new,delta=d,selected=need)
    print("geometry replay",split,"affected",int(np.count_nonzero(d)),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("fit","val","test"))
    p.add_argument("--root",type=Path,default=Path("artefacts/kaggle_suite/curriculum_run"))
    p.add_argument("--teacher",action="store_true")
    a=p.parse_args()
    torch.set_num_threads(2)
    fit(a.root) if a.stage=="fit" else infer(a.root,a.stage,a.teacher)

