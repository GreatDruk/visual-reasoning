"""Replay the exploratory ensemble from fitted curriculum and RIM checkpoints."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit, softmax
import torch

from scripts.research import curriculum_eval as e, curriculum_train as train
from scripts.research import rim_fragments as crops, rim_scene_unet as whole

LOCAL=("unet_noexact","r50_noexact","unet_boundary")
COLOR=r"\b(?:gray|red|blue|green|brown|purple|cyan|yellow) objects\b"
MATERIAL=r"\b(?:metal|rubber)\b"
MATERIAL_WEIGHT=1.125


def scopes(frame):
    comparison=frame.category.eq("comparison").to_numpy()
    color=comparison&(frame.caption.str.count(COLOR).to_numpy()==2)
    material=frame.category.eq("counting").to_numpy()&frame.caption.str.contains(MATERIAL).to_numpy()
    return color,comparison&~color,material


def locations(root):
    target=root/"replay"
    target.mkdir(exist_ok=True)
    posterior=target/"rear_cache"
    posterior.mkdir(exist_ok=True)
    return target,posterior


@torch.inference_mode()
def prepare(frame,root,shard,device,scene_names=None):
    target,posterior=locations(root)
    pb=posterior/f"test_{shard}.npz"
    path=target/f"test_{shard}.npz"
    if pb.is_file() and path.is_file() and (target/f"test_{shard}_rgb.npy").is_file():
        return
    v=crops.v
    if scene_names is None:
        _,shape,material=scopes(frame)
        scene_names=set(frame.image[shape|material])
    ns=frame.image.unique()
    names=np.array([n for n in ns[shard::2] if n in scene_names])
    start=time.monotonic()
    models=[]
    heads=[]
    for model,key in ((v.ResNetCenters(full=True),"r50_img"),(v.CenterDetector(),"thaw")):
        model.load_state_dict(v.load(v.PRIORS/f"{key}.pt"))
        models.append(model.to(device,memory_format=torch.channels_last).eval())
        heads.append(v.model_head(key,device))
    projection=v.projection().to(device)
    cache=np.lib.format.open_memmap(target/f"test_{shard}_rgb.npy",mode="w+",dtype=np.uint8,
                                    shape=(len(names),v.HEIGHT,v.WIDTH,3))
    indices=[]; boxes=[]; fronts=[]; likelihood=[]; tables=[]
    batch=32 if device.type=="cpu" else 160
    for i in range(0,len(names),batch):
        pixels=crops.images("test",names[i:i+batch])
        cache[i:i+len(pixels)]=pixels
        rgb=torch.from_numpy(pixels).to(device).permute(0,3,1,2).float()/255
        a=v.detector_outputs(models[0],heads[0],rgb)
        b=v.detector_outputs(models[1],heads[1],rgb,projection)
        for j,(left,right) in enumerate(zip(a,b),i):
            tables.append(e.r.layer.rear.prompts(*v.merge(left,right),left))
            for box,front,p in crops.candidate_peaks(left):
                indices.append(j)
                boxes.append(box)
                fronts.append(front)
                likelihood.append(p)
        if i%512==0:
            print("test",shard,i+len(pixels),len(names),flush=True)
    cache.flush()
    x,xy,valid,pairs,geometry=map(np.stack,zip(*tables))
    np.savez(pb,names=names,x=x,xy=xy,valid=valid,pairs=pairs,geometry=geometry)
    np.savez_compressed(path,names=names,index=np.asarray(indices,dtype=np.int64),
                        boxes=np.asarray(boxes,dtype=np.float32).reshape(-1,4),
                        front=np.asarray(fronts,dtype=np.float32).reshape(-1,4),source=likelihood)
    print("prepared test",shard,len(names),len(indices),round(time.monotonic()-start,1),flush=True)


def predict(frame,root,shard,device):
    out,_=locations(root)
    crops.OUT=out
    _,shape,material=scopes(frame)
    for name in LOCAL:
        arm,mode=name.split("_",1)
        images=set(frame.image[shape if name=="unet_noexact" else material])
        for seed in e.SEEDS:
            path=out/f"{name}_{seed}_test_{shard}.npz"
            if path.is_file():
                continue
            folder=root/"curriculum"
            info=json.loads((folder/f"{name}_{seed}.json").read_text())
            state=folder/(f"{name}_{seed}.pt" if info.get("fixed_checkpoint")
                          else f"{name}_{seed}_{info['selected_step']}.pt")
            model=train.build(arm,mode,seed,device,pretrained=False)
            model.load_state_dict(crops.v.load(state))
            crops.infer(name,seed,"test",shard,device,model.eval(),scene_names=images)


def predict_color(frame,root,seed,device):
    out,_=locations(root)
    color,_,_=scopes(frame)
    p=e.r.read(e.r.EXTRA/"anchor_test.npz",frame,"probability")
    needed=color&(np.abs(logit(p.clip(1e-15,1-1e-15)))<=3)
    if not needed.any():
        np.savez_compressed(out/f"scene_unet_{seed}_test.npz",ids=np.empty(0,dtype=frame.id.dtype),score=np.empty(0))
        return
    selected=frame.loc[frame.image.isin(frame.image[needed])].reset_index(drop=True)
    state=f"scene_unet_{seed}.pt"
    shutil.copy2(root.parent/"rim_run"/state,out/state)
    whole.infer(seed,device,"test",None,output=out,frame=selected)


def attributes(shape_name,material_name,split):
    for a,b in zip(e.load(shape_name,split),e.load(material_name,split)):
        if any(not np.array_equal(a[k],b[k]) for k in ("names","index","boxes","front")):
            raise ValueError("Shape and material predictions must use identical candidates")
        ja,jb=[softmax(p["data"][:,11:17],axis=1).reshape(-1,2,3) for p in (a,b)]
        shape=ja.sum(1)
        material=.25*ja.sum(2)+.75*jb.sum(2)
        out=dict(a,data=a["data"].copy())
        out["data"][:,11:17]=np.log((material[:,:,None]*shape[:,None,:]).reshape(-1,6).clip(1e-8))
        agreement=np.linalg.norm(a["peak"]-b["peak"],axis=1)<8
        out["pixels"]=np.minimum(a["pixels"],b["pixels"])*agreement
        out["compactness"]=np.minimum(a["compactness"],b["compactness"])
        out["data"][:,0]=(a["data"][:,0]+b["data"][:,0])/2
        yield out


def replay(frame,base,root,split,color_score=None):
    if split=="test":
        out,posterior=locations(root)
    else:
        out=root/"curriculum"
        posterior=root.parent/"rim_run/rear_cache"
    e.OUT=e.r.OUT=out
    e.r.layer.rear.SCRATCH=posterior
    tables=e.r.bank(split)
    c,s,m=scopes(frame)
    color=c&(np.abs(base)<=3)
    if color_score is None:
        parts=[]
        for seed in e.SEEDS:
            with np.load(out/f"scene_unet_{seed}_test.npz") as a:
                scores=pd.Series(a["score"],index=a["ids"])
                parts.append(scores.reindex(frame.id).to_numpy())
        color_score=np.mean(parts,axis=0)
        if not np.isfinite(color_score[color]).all():
            raise ValueError("Missing U-Net scores for color comparisons")
    d,_,_=e.r.branch(frame,base,e.r.by_image(e.load("unet_noexact",split)),.92,tables)
    h,_,_=e.r.branch(frame,base,e.r.by_image(attributes("r50_noexact","unet_boundary",split)),.5,tables)
    return np.select([color,s,m],[.5*(base+color_score),base+d,base+MATERIAL_WEIGHT*h],default=base)


def anchor_path(root):
    path=root/"anchor_test.npz"
    return path if path.is_file() else e.r.EXTRA/"anchor_test.npz"


def export(frame,root):
    base=e.r.read(anchor_path(root),frame,"probability")
    odds=logit(base.clip(1e-15,1-1e-15))
    score=replay(frame,odds,root,"test")
    p=np.where(odds==score,base,expit(score))
    template=pd.read_csv(crops.v.DATA/"sample_submission.csv")
    if not frame.id.is_unique or not template.id.is_unique or set(frame.id)!=set(template.id):
        raise ValueError("Test IDs do not match sample_submission.csv")
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError("Invalid probabilities")
    template["label"]=pd.Series(p,index=frame.id).loc[template.id].to_numpy()
    output=root/"shape_mat.csv"
    interim=output.with_suffix(".tmp")
    template.to_csv(interim,index=False,float_format="%.17g")
    pd.testing.assert_frame_equal(template,pd.read_csv(interim,float_precision="round_trip"))
    interim.replace(output)
    print(output,len(p),"rows",int((p!=base).sum()),"updated",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","predict","color","export"))
    p.add_argument("--root",type=Path,default=Path("artefacts/kaggle_suite/curriculum_run"))
    p.add_argument("--shard",type=int,choices=(0,1),default=0)
    p.add_argument("--seed",type=int,choices=e.SEEDS,default=421)
    a=p.parse_args()
    frame=pd.read_csv(crops.v.DATA/"test.csv")
    _,device=crops.v.setup(a.seed)
    if a.stage=="prepare":
        prepare(frame,a.root,a.shard,device)
    elif a.stage=="predict":
        predict(frame,a.root,a.shard,device)
    elif a.stage=="color":
        predict_color(frame,a.root,a.seed,device)
    else:
        export(frame,a.root)


if __name__=="__main__":
    main()

