"""Try replacing the separate material backbone with the existing scene detector."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit, softmax
from sklearn.metrics import roc_auc_score
import torch

from scripts.research import curriculum_eval as e, curriculum_replay as prior
from scripts.research import curriculum_train as train

ROOT=Path("artefacts/kaggle_suite/curriculum_run")
FOCUS=("58281.jpg","78612.jpg","26268.jpg","11105.jpg")


def load_base(root,split):
    frame=pd.read_csv(prior.crops.v.DATA/f"{split}.csv")
    if split=="val":
        p=e.r.read(e.r.EXTRA/"anchor_val.npz",frame,"score")
        full=e.r.read(root/"shape_mat_val.npz",frame,"score")
    else:
        p=logit(e.r.read(e.r.EXTRA/"anchor_test.npz",frame,"probability"))
        saved=pd.read_csv(root/"shape_mat.csv",float_precision="round_trip")
        if not saved.id.is_unique or set(saved.id)!=set(frame.id):
            raise ValueError("Wrong shape/material reference IDs")
        full=logit(saved.set_index("id").loc[frame.id,"label"].to_numpy())
    return frame,p,full


def recovered(proposals,tables):
    output={}
    for z,(p,xy,valid,attrs,offset) in tables:
        for i,name in enumerate(z["names"]):
            options=proposals.get(name,[])
            if not options:
                continue
            boxes=z["x"][i,:,prior.crops.v.BOX]*prior.crops.SCALE
            known=valid[i]
            if not known.any():
                continue
            out=[]
            for o in options:
                d=np.linalg.norm(e.r.center(boxes)-e.r.center(o["box"]),axis=1)
                overlaps=e.r.pairwise_iou(o["box"],boxes)
                ok=known&(d<8)&(overlaps>.30)
                if not ok.any():
                    continue
                j=int(np.argmin(np.where(ok,d,np.inf)))
                field=o["data"].copy()
                joint=softmax(field[11:17]).reshape(2,3)
                material=.25*attrs[2][i,j]+.75*joint.sum(1)
                shape=attrs[3][i,j]
                field[11:17]=np.log((material[:,None]*shape[None,:]).ravel().clip(1e-8))
                out.append(dict(o,data=field))
            if out:
                output[name]=out
    return output


def eval_split(root,split):
    f,base,full=load_base(root,split)
    if split=="val":
        e.OUT=e.r.OUT=root/"curriculum"
        e.r.layer.rear.SCRATCH=root.parent/"rim_run/rear_cache"
    else:
        e.OUT=e.r.OUT=root/"replay"
        e.r.layer.rear.SCRATCH=root/"replay/rear_cache"
    tables=e.r.bank(split)
    proposals=e.r.by_image(e.load("unet_boundary",split))
    correction=recovered(proposals,tables)
    _,shape_scope,active=prior.scopes(f)
    selected=np.where(active,base,full)
    choices={"material_off":selected,"shape_mat":full}
    for q in (.50,.65,.80):
        delta,_,_=e.r.branch(f,base,correction,q,tables)
        for w in (.65,1.0,1.125):
            choices[f"shared_{q}_{w}"]=np.where(active,base+w*delta,full)
    scope_tag="unet_boundary" if split=="val" else "unet_boundary_shape"
    if e.files(scope_tag,split):
        shape_proposals=proposals if split=="val" else e.r.by_image(e.load(scope_tag,split))
        for q in ((.70,.85,.92) if split=="val" else (.92,)):
            delta,_,_=e.r.branch(f,base,shape_proposals,q,tables)
            choices[f"single_local_{q}"]=np.where(shape_scope,base+delta,choices["shared_0.65_0.65"])
    return f,base,choices


def report(root):
    from scripts.research.visible_fragment_eval import score_metrics
    f,base,s=eval_split(root,"val")
    outside=~f.image.isin(FOCUS).to_numpy()
    stats={}
    for name,p in s.items():
        row=score_metrics(f,base,p)
        y=f.label.to_numpy()[outside]
        row["inversions_without_four"]=round((1-roc_auc_score(y,p[outside]))*sum(y)*sum(1-y))
        row["active"]=int(np.count_nonzero(p!=s["shape_mat"]))
        print(name,row["inversions"],row["errors"],row["inversions_without_four"],flush=True)
        stats[name]=row
    out=root/"views"
    out.mkdir(exist_ok=True)
    (out/"share_report.json").write_text(json.dumps(stats,indent=2),encoding="utf8")
    np.savez_compressed(out/"share_val.npz",ids=f.id.to_numpy(),keys=list(s),scores=np.stack(list(s.values())))


def predict_shape(root,shard):
    f,_,_=load_base(root,"test")
    _,shape,_=prior.scopes(f)
    names=set(f.image[shape])
    crops=prior.crops
    crops.OUT=root/"replay"
    for seed in e.SEEDS:
        _,device=crops.v.setup(seed)
        if device.type=="cpu":
            torch.set_num_threads(3)
        weights=root/"curriculum"
        ckpt=json.loads((weights/f"unet_boundary_{seed}.json").read_text())["selected_step"]
        model=train.build("unet","boundary",seed,device,pretrained=False)
        model.load_state_dict(crops.v.load(weights/f"unet_boundary_{seed}_{ckpt}.pt"))
        crops.infer("unet_boundary_shape",seed,"test",shard,device,model.eval(),scene_names=names)


def export(root,compact=False):
    f,_,values=eval_split(root,"test")
    key="single_local_0.92" if compact else "shared_0.65_0.65"
    old=pd.read_csv(root/"shape_mat.csv",float_precision="round_trip").set_index("id")
    changed=values[key]!=values["shape_mat"]
    prob=np.where(changed,expit(values[key]),old.loc[f.id,"label"].to_numpy())
    sample=pd.read_csv(prior.crops.v.DATA/"sample_submission.csv")
    if not f.id.is_unique or not sample.id.is_unique or set(f.id)!=set(sample.id):
        raise ValueError("Submission IDs do not match")
    sample["label"]=pd.Series(prob,index=f.id).loc[sample.id].to_numpy()
    path=root/("shape_compact.csv" if compact else "shape_shared.csv")
    temp=path.with_suffix(".tmp")
    sample.to_csv(temp,index=False,float_format="%.17g")
    read=pd.read_csv(temp,float_precision="round_trip")
    if not read.id.equals(sample.id) or not read.label.between(0.,1.).all():
        raise ValueError("Invalid compact submission")
    temp.replace(path)
    print(path,len(sample),"rows",int(changed.sum()),"local changes",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--export",action="store_true")
    p.add_argument("--compact",action="store_true")
    p.add_argument("--predict-shape",type=int,choices=(0,1))
    a=p.parse_args()
    if a.predict_shape is not None:
        predict_shape(a.root,a.predict_shape)
    elif a.export:
        export(a.root,a.compact)
    else:
        report(a.root)

