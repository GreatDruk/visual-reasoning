"""Recover candidates near contours and replay uncertain spatial counts."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit, softmax
import torch

from scripts.research import curriculum_eval as e, curriculum_replay as typed, curriculum_train as train
from scripts.research import rim_fragments as rim
from scripts.research.visible_fragment_eval import score_metrics

DIRECTIONS=np.asarray(((.05,.16),(.50,.10),(.95,.16),(.06,.52),(.94,.52),(.50,.36)))
VARIANTS=("r50_mixed","unet_boundary")
FOCUS_MARGIN=3.
THRESHOLDS=(.15,.35,.55)


def frames(root,split):
    frame=pd.read_csv(rim.v.DATA/f"{split}.csv")
    if split=="val":
        base=e.r.read(e.r.EXTRA/"anchor_val.npz",frame,"score")
        typed_score=e.r.read(root/"shape_mat_val.npz",frame,"score")
    else:
        p=e.r.read(typed.anchor_path(root),frame,"probability")
        base=logit(p)
        a=pd.read_csv(root/"shape_mat.csv",float_precision="round_trip")
        if not a.id.is_unique or set(a.id)!=set(frame.id):
            raise ValueError("Invalid typed submission")
        typed_score=logit(a.set_index("id").loc[frame.id,"label"].to_numpy())
    target=frame.category.eq("spatial_counting").to_numpy()&(np.abs(base)<=FOCUS_MARGIN)
    return frame,base,typed_score,target


def posterior(root,split,shard,device):
    dst=root/"spatial"
    dst.mkdir(exist_ok=True)
    cache=(root.parent/"rim_run/rear_cache") if split=="val" else typed.locations(dst)[1]
    if split=="test":
        f,_,_,target=frames(root,split)
        typed.prepare(f,dst,shard,device,set(f.image[target]))
    return dst,cache


def proposals(root,split,shard,device):
    folder,cache=posterior(root,split,shard,device)
    f,_,_,target=frames(root,split)
    chosen=set(f.image[target])
    e.r.layer.rear.SCRATCH=cache
    path=folder/f"{split}_{shard}.npz"
    if path.is_file() and (folder/f"{split}_{shard}_rgb.npy").is_file():
        return
    arr=e.r.layer.rear.collect(split,shard)
    p,*_=e.r.layer.rear.bank_parts(arr,split,torch.device("cpu"))
    names=np.asarray([x for x in arr["names"] if x in chosen])
    bank={name:i for i,name in enumerate(names)}
    si=[]; boxes=[]; fronts=[]; source=[]
    for i,name in enumerate(arr["names"]):
        if name not in bank:
            continue
        available=arr["x"][i,arr["valid"][i]&(p[i]>=.70),rim.v.BOX]*rim.SCALE
        for b in available:
            if b[2:].min()<8:
                continue
            for xy in b[:2]+DIRECTIONS*b[2:]:
                si.append(bank[name])
                boxes.append(np.r_[xy-6,12,12])
                fronts.append(b)
                source.append(0.)
    if not len(names):
        raise ValueError("Empty spatial slice")
    np.save(folder/f"{split}_{shard}_rgb.npy",rim.images(split,names))
    np.savez_compressed(path,names=names,index=np.asarray(si,dtype=np.int64),
                        boxes=np.asarray(boxes,dtype=np.float32).reshape(-1,4),
                        front=np.asarray(fronts,dtype=np.float32).reshape(-1,4),source=np.asarray(source))
    print("contour candidates",split,shard,len(names),len(si),flush=True)


def infer(root,split,arm,seed,shard,device):
    folder=root/"spatial"
    rim.OUT=folder
    if (folder/f"{arm}_{seed}_{split}_{shard}.npz").exists():
        return
    a,b=arm.split("_",1)
    model=train.build(a,b,seed,device,pretrained=False)
    weights=root/"curriculum"
    ckpt=json.loads((weights/f"{arm}_{seed}.json").read_text())
    state=weights/(f"{arm}_{seed}.pt" if ckpt.get("fixed_checkpoint")
                   else f"{arm}_{seed}_{ckpt['selected_step']}.pt")
    model.load_state_dict(rim.v.load(state))
    rim.infer(arm,seed,split,shard,device,model.eval())


def distributions(x):
    size=softmax(x[1:3])
    color=softmax(x[3:11])
    joint=softmax(x[11:17]).reshape(2,3)
    return size,color,joint


def framed(z,mode):
    """Require two distinct windows for a physical fragment."""
    for part in z:
        for i,name in enumerate(part["names"]):
            rows=np.flatnonzero((part["index"]==i)&(part["pixels"]>=3)&(part["compactness"]>=.40))
            if not len(rows):
                continue
            rank=expit(part["data"][rows,0])
            chosen=[]
            for j in rank.argsort()[::-1]:
                k=rows[j]
                p=rank[j]
                if p<.12:
                    break
                peak=part["peak"][k]
                b=part["front"][k]
                mid=e.r.center(part["boxes"][k])
                near=np.linalg.norm(part["peak"][rows]-peak,axis=1)<=5.5
                views=np.linalg.norm(e.r.center(part["boxes"][rows])-mid,axis=1)>=4.
                aligned=np.asarray([rim.pairwise_iou(b,other)>=.70 for other in part["front"][rows]])
                allowed=near&views&aligned&(rank>=max(.12,p*.5))
                if mode=="one":
                    allowed[j]=True
                if not allowed.any() or any(np.linalg.norm(peak-p0["peak"])<6 for p0 in chosen):
                    continue
                agree=rows[np.r_[j,np.flatnonzero(allowed)]]
                d=part["data"][agree].mean(0)
                size,_,_=distributions(d)
                # A visible fragment is usually above the real ground contact.
                # Use the apparent object scale of the occluder as a bounded estimate.
                extent=np.clip(b[2:]*(.42+.24*size[1]),[7,8],[23,32])
                point=peak+np.asarray([0,.30*extent[1]])
                box=np.r_[point-.5*extent,extent]
                chosen.append(dict(data=d,box=box,front=b,pixels=6,compactness=1.,peak=peak,
                                   prob=float(p),source=float(p)))
            if chosen:
                yield str(name),chosen


def candidates(root,split,variant,mode):
    e.OUT=e.r.OUT=root/"spatial"
    if variant=="agreement":
        base=list(e.agree(list(VARIANTS),split))
    else:
        base=e.load(variant,split)
    return dict(framed(base,mode))


def tables_for(root,split):
    cache=(root.parent/"rim_run/rear_cache") if split=="val" else root/"spatial/replay/rear_cache"
    e.r.layer.rear.SCRATCH=cache
    return e.r.bank(split)


def nearby(box,candidate):
    return np.linalg.norm(e.r.center(box)-candidate["peak"])<=11


def replay(frame,base,active,proposals,tables,threshold,anchor_mode):
    ret=np.zeros(len(frame))
    directly=np.zeros(len(frame))
    grouped=frame.groupby("image",sort=False).indices
    for z,(p,xy,valid,attrs,offset) in tables:
        for i,name in enumerate(z["names"]):
            options=proposals.get(name,[])
            group=grouped[name]
            if not options or not active[group].any():
                continue
            kval=valid[i]
            b=z["x"][i,kval,rim.v.BOX]*rim.SCALE
            pres=p[i,kval]
            attr=[a[i,kval].copy() for a in attrs]
            pos=xy[i,kval].copy()
            chosen=None
            for o in options:
                if o["prob"]<threshold:
                    continue
                weak=np.asarray([nearby(box,o) for box in b])&(pres<.60)
                strong=(pres>=.60)&(rim.pairwise_iou(o["box"],b)>.36)
                if strong.any():
                    continue
                j=int(np.argmin(np.where(weak,np.linalg.norm(e.r.center(b)-o["peak"],axis=1),np.inf))) if weak.any() else -1
                if anchor_mode=="existing" and j<0:
                    continue
                chosen=(o,j)
                break
            if chosen is None:
                continue
            o,j=chosen
            captions=[e.r.layer.canonical_form(r.caption,r.category) for r in frame.iloc[group].itertuples()]
            twins=e.r.layer.find_twins(frame.image.iloc[group],captions,frame.category.iloc[group])
            programs=[e.r.layer.parse_program(cap) for cap in captions]
            if any(d is None for d in programs):
                continue
            if j<0:
                j=len(pres)
                pres=np.r_[pres,0.]
                attr=[np.vstack((a,np.full(a.shape[1],1/a.shape[1]))) for a in attr]
                where=(o["box"][:2]+o["box"][2:]*[.5,.83])/rim.SCALE[:2]*2-1
                pos=np.vstack((pos,where))
            joint=e.r.layer.material_priors(attr)
            u=e.r.layer._draws(len(pres),8,41,2048)
            old=e.r.layer.rear.decode(programs,twins,pres,attr,joint,pos,offset,u)
            a,c,m=distributions(o["data"])
            pres[j]=max(pres[j],min(.92,o["prob"]*2))
            attr[0][j]=a
            attr[1][j]=c
            joint[j]=m.flatten()
            new=e.r.layer.rear.decode(programs,twins,pres,attr,joint,pos,offset,u)
            ret[group]=(new-old)*active[group]
            directly[group]=new*active[group]
    return ret,directly


def evaluate(root,split):
    f,b,_,mask=frames(root,split)
    folder=root/"spatial"
    tb=tables_for(root,split)
    result={}
    selected=None
    if split=="test":
        plan=json.loads((folder/"validation.json").read_text(encoding="utf8"))["preferred"]
        selected={plan[k].split("@",1)[0].removeprefix("agree/").removeprefix("stack/")
                  for k in ("contour","both") if k in plan}
    for model in (*VARIANTS,"agreement"):
        for mode in ("one","views"):
            options=[(anchor,q) for anchor in ("existing","new") for q in THRESHOLDS
                     if selected is None or f"{model}/{mode}/{anchor}/{q}" in selected]
            if not options:
                continue
            props=candidates(root,split,model,mode)
            for anchor,q in options:
                delta,_=replay(f,b,mask,props,tb,q,anchor)
                result[f"{model}/{mode}/{anchor}/{q}"]=delta
    if not result:
        return
    np.savez_compressed(folder/f"{split}_branches.npz",ids=f.id.to_numpy(),
                        keys=np.asarray(list(result)),delta=np.stack(list(result.values())))
    print(split,"contour branch replays",len(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","infer","evaluate"))
    p.add_argument("--root",type=Path,default=Path("artefacts/kaggle_suite/curriculum_run"))
    p.add_argument("--split",choices=("val","test"),required=True)
    p.add_argument("--arm",choices=VARIANTS)
    p.add_argument("--seed",type=int,choices=e.SEEDS,default=421)
    p.add_argument("--shard",choices=(0,1),type=int,default=0)
    p.add_argument("--threads",type=int,default=3)
    a=p.parse_args()
    _,dev=rim.v.setup(a.seed)
    if dev.type=="cpu":
        torch.set_num_threads(a.threads)
    if a.stage=="prepare":
        proposals(a.root,a.split,a.shard,dev)
    elif a.stage=="infer":
        infer(a.root,a.split,a.arm,a.seed,a.shard,dev)
    else:
        evaluate(a.root,a.split)


if __name__=="__main__":
    main()

