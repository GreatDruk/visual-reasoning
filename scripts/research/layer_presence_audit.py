"""Replay accepted verify hypotheses while keeping the original detector's classes."""
from itertools import product
from pathlib import Path
import io
import json
import zipfile

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import torch

from scripts.research import layer_repair as model
from scripts.research.layer_consensus_audit import RUN, CORE
from scripts.research.fragment_votes import images


def selected(archive,seed):
    parts=[]
    for shard in (0,1):
        prefix=f"layer_result/verify_{seed}"
        df=pd.DataFrame(json.loads(archive.read(f"{prefix}_{shard}_candidates.json")))
        passed=df[(df.probability>=.87)&(df.contrast>=.75)&(df.pixels>=5)]
        passed=passed.sort_values("probability",ascending=False,kind="stable").drop_duplicates("image")
        with np.load(io.BytesIO(archive.read(f"{prefix}_val_{shard}.npz"))) as z:
            parts.append((passed,z["ids"]))
    return pd.concat([x[0] for x in parts]).set_index("image"),np.concatenate([x[1] for x in parts])


def replay(frame):
    with zipfile.ZipFile(RUN/"layer_results.zip") as z:
        a,aid=selected(z,421);b,bid=selected(z,853)
    allowed=frame[frame.id.isin(aid)&frame.id.isin(bid)].image.unique()
    a,b=a.loc[allowed],b.loc[allowed]
    matches=np.max(np.abs(np.stack(a.box)-np.stack(b.box)),axis=1)<.001
    names=allowed[matches]
    dev=torch.device("cpu")
    v=model.v
    net=v.ResNetCenters(full=True).eval();native=v.CenterDetector().eval()
    net.load_state_dict(v.load(v.PRIORS/"r50_img.pt"));native.load_state_dict(v.load(v.PRIORS/"thaw.pt"))
    head,other=v.model_head("r50_img",dev),v.model_head("thaw",dev)
    projection=v.projection()
    captions=[model.canonical_form(row.caption,row.category) for row in frame.itertuples()]
    index=frame.groupby("image",sort=False).indices
    result=np.zeros(len(frame))
    support=np.zeros(len(frame))
    used=0
    with torch.inference_mode():
        for start in range(0,len(names),24):
            batch=names[start:start+24]
            pixels=torch.from_numpy(images("val",batch)).permute(0,3,1,2).float()/255
            x=v.detector_outputs(net,head,pixels)
            y=v.detector_outputs(native,other,pixels,projection)
            xx,xy,valid,pairs,geometry=map(np.stack,zip(*[model.rear.prompts(*v.merge(aa,bb),aa)
                                                        for aa,bb in zip(x,y)]))
            ps,z,ok,attributes,offset=model.rear.bank_parts(dict(x=xx,xy=xy,valid=valid),"val",dev)
            for i,name in enumerate(batch):
                used_slots=np.flatnonzero(pairs[i,:,0]>=0)
                if not len(used_slots):continue
                d=np.linalg.norm(geometry[i,used_slots,0]-np.asarray(a.loc[name,"box"]),axis=1)
                k=d.argmin()
                if d[k]>3.5:continue
                slot=pairs[i,used_slots[k],0]
                rowid=index[name]
                cap=[captions[r] for r in rowid]
                pipe=model.find_twins(frame.image.iloc[rowid],cap,frame.category.iloc[rowid])
                events=[model.parse_program(c) for c in cap]
                mask=ok[i];j=np.cumsum(mask)[slot]-1
                p=ps[i,mask];attrs=[a[i,mask] for a in attributes]
                distribution=model.material_priors(attrs)
                u=model._draws(len(p),8,41,2048)
                old=model.rear.decode(events,pipe,p,attrs,distribution,z[i,mask],offset,u)
                p[j]=.95*(a.loc[name,"probability"]+b.loc[name,"probability"])/2
                new=model.rear.decode(events,pipe,p,attrs,distribution,z[i,mask],offset,u)
                result[rowid]=new-old
                support[rowid]=min(a.loc[name,"contrast"],b.loc[name,"contrast"])
                used+=1
            if start%96==0:print("replay",start+len(batch),"/",len(names),flush=True)
    print("matched replay",used,"/",len(names),flush=True)
    return result*np.asarray(frame.category.isin(CORE)),support


def main():
    torch.set_num_threads(5)
    with zipfile.ZipFile(RUN/"layer_results.zip") as z:
        f=pd.read_csv(z.open("verify_result/val_rows.csv"))
    diff,support=replay(f)
    np.savez(RUN/"presence_only_val.npz",ids=f.id,delta=diff,contrast=support)
    with zipfile.ZipFile(RUN.parent/"rear_run/rear_results.zip") as z:
        h=pd.read_csv(z.open("verify_result/val_rows.csv"))
    assert h.id.equals(f.id)
    y=f.label.to_numpy()
    scene,names=pd.factorize(f.image)
    for name,base in [("union_tv",f.baseline.to_numpy()),("count_hidden",h.base.to_numpy())]:
        choices=[(0.,0.),*product([1.,2.,4.],[.1,.25,.5])]
        scores={k:base+np.where(support>=k[0],k[1]*diff,0) for k in choices}
        key=max(choices,key=lambda k:roc_auc_score(y,scores[k]))
        print(name,"baseline",roc_auc_score(y,base),"full-val best",key,roc_auc_score(y,scores[key]))
        for seed in (619,2401,4211):
            oof=base.copy()
            for tr,te in KFold(5,shuffle=True,random_state=seed).split(names):
                fit,out=np.isin(scene,tr),np.isin(scene,te)
                k=max(choices,key=lambda k:roc_auc_score(y[fit],scores[k][fit]))
                oof[out]=scores[k][out]
            print(seed,roc_auc_score(y,oof),int(((y==1)&(oof<=0)).sum()),flush=True)


if __name__=="__main__":
    main()
