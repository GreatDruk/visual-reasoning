"""Use additional views of the same detector to corroborate weak objects."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import expit, logit, logsumexp
import torch
from torch.nn import functional as F

from scripts.research import rim_eval as r
from scripts.research import rim_fragments as f
from kaggle_suite_common import table_predictions

OUT=Path("artefacts/kaggle_suite/curriculum_run")
ACTIVE=("counting","comparison","spatial_counting")
ARMS=("standard","offset","reflected","supported","strict")
SIZE=np.asarray([f.W,f.H])
GATES=(2.,3.,4.)


def inputs(root,split):
    ref=pd.read_csv(f.v.DATA/f"{split}.csv")
    if split=="val":
        p=r.read(f.EXTRA/"anchor_val.npz",ref,"score")
        anchor=r.read(root/"shape_mat_val.npz",ref,"score")
    else:
        p=logit(r.read(f.EXTRA/"anchor_test.npz",ref,"probability"))
        raw=pd.read_csv(root/"shape_mat.csv",float_precision="round_trip")
        if not raw.id.is_unique or set(raw.id)!=set(ref.id):
            raise ValueError("Invalid shape_mat.csv")
        anchor=logit(raw.set_index("id").loc[ref.id,"label"].to_numpy())
    focus=ref.category.isin(ACTIVE).to_numpy()&(np.abs(p)<=GATES[-1])
    return ref,p,anchor,focus


def points(peak):
    return (peak[:,-6:-4]+.5*peak[:,-4:-2])*SIZE


def pair(left,right):
    if not len(left) or not len(right):
        return {}
    dist=np.linalg.norm(points(left)[:,None]-points(right)[None],axis=2)
    dummy=np.full((len(left),len(left)),6.1)
    a,b=linear_sum_assignment(np.column_stack([dist,dummy]))
    return {int(i):int(j) for i,j in zip(a,b) if j<len(right) and dist[i,j]<=6.}


def match(a,b):
    a_shape=int(a[14:17].argmax())
    a_color=int(a[4:12].argmax())
    return a_shape==int(b[14:17].argmax()) and a_color==int(b[4:12].argmax())


def average(rows):
    data=np.mean(rows,axis=0)
    lo=2
    for n in (2,8,2,3):
        data[lo:lo+n]-=logsumexp(data[lo:lo+n])
        lo+=n
    return data


def corroborate(raw,off,flip):
    threshold=logit(.025)
    a,b,c=[z[z[:,0]>=threshold] for z in (raw,off,flip)]
    sb=pair(a,b)
    sc=pair(a,c)
    off_only=np.asarray([j for j in range(len(b)) if j not in sb.values()],dtype=np.int64)
    flip_only=np.asarray([j for j in range(len(c)) if j not in sc.values()],dtype=np.int64)
    missing=[(b[off_only[i]],c[flip_only[j]]) for i,j in pair(b[off_only],c[flip_only]).items()]
    versions=[]
    for cutoff in (.12,.28):
        rows=[]
        for i,first in enumerate(a):
            support=[]
            if i in sb:
                support.append(b[sb[i]])
            if i in sc:
                support.append(c[sc[i]])
            enough=[z for z in support if z[0]>=logit(cutoff) and match(first,z)]
            # The raw head was trained at the normal pixel phase. Require independent
            # corroboration and avoid altering already confidently recognized objects.
            if enough and first[0]<logit(.55) and np.mean([z[0] for z in enough])>first[0]+.50:
                rows.append(average([first,*enough]))
            else:
                rows.append(first)
        for x,y in missing:
            if min(x[0],y[0])>=logit(cutoff) and match(x,y):
                row=average([x,y])
                if any(np.linalg.norm(points(row[None])[0]-points(z[None])[0])<5.5 for z in rows):
                    continue
                rows.append(row)
        versions.append(np.asarray(rows,dtype=np.float32).reshape(-1,raw.shape[1]))
    return versions


def restore(raw,reflected=False,dx=0,dy=0):
    if reflected:
        raw[:,:,-6]=1-raw[:,:,-6]-raw[:,:,-4]
        raw[:,:,-2]*=-1
    shift=np.asarray([dx,dy],dtype=np.float32)/SIZE
    raw[:,:,-6:-4]-=shift
    raw[:,:,-2:]-=shift*2
    return raw


@torch.inference_mode()
def prepare(root,split,shard,device):
    frame,_,_,use=inputs(root,split)
    out=root/"views"
    out.mkdir(exist_ok=True)
    allowed=set(frame.image[use])
    ns=np.asarray([n for n in frame.image.unique()[shard::2] if n in allowed])
    path=out/f"{split}_{shard}.npz"
    if path.is_file():
        with np.load(path) as z:
            if np.array_equal(z["names"],ns):
                return
    v=f.v
    nets=[]
    heads=[]
    for tag,net in zip(("r50_img","thaw"),(v.ResNetCenters(full=True),v.CenterDetector())):
        net.load_state_dict(v.load(v.PRIORS/f"{tag}.pt"))
        nets.append(net.to(device,memory_format=torch.channels_last).eval())
        heads.append(v.model_head(tag,device))
    matrix=v.projection().to(device)
    parts={key:[] for key in ARMS}
    start=time.monotonic()
    batch=36 if device.type=="cpu" else 192
    with ThreadPoolExecutor(3) as pool:
        for j in range(0,len(ns),batch):
            raw=np.stack(list(pool.map(v.read_rgb,[v.DATA/split/n for n in ns[j:j+batch]])))
            rgb=torch.as_tensor(raw,device=device).permute(0,3,1,2).float().div_(255)
            normal=v.detector_outputs(nets[0],heads[0],rgb)
            shifted=F.pad(rgb,(2,0,2,0),mode="replicate")[:,:,:f.H,:f.W]
            offset=restore(v.detector_outputs(nets[0],heads[0],shifted),dx=2,dy=2)
            reflected=restore(v.detector_outputs(nets[0],heads[0],rgb.flip(-1)),reflected=True)
            native=v.detector_outputs(nets[1],heads[1],rgb,matrix)
            for raw,off,flip,other in zip(normal,offset,reflected,native):
                combined=corroborate(raw,off,flip)
                for key,values in zip(ARMS,(raw,off,flip,*combined)):
                    parts[key].append(v.merge(values,other))
    payload={"names":ns}
    for key,rows in parts.items():
        payload.update({f"{key}_{n}":np.stack(arr) for n,arr in zip(("x","xy","valid"),zip(*rows))})
    tmp=path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle,**payload)
    tmp.replace(path)
    print("shared detector pass",split,shard,len(ns),round(time.monotonic()-start,1),"s",flush=True)


@torch.inference_mode()
def scores(root,split,device):
    ff,original,base,mask=inputs(root,split)
    out=root/"views"
    model=f.v.Verifier(False)
    state="union_trainval.pt" if split=="test" else "union_train.pt"
    model.load_state_dict(f.v.load(f.v.PRIORS/state))
    model.to(device).eval()
    offset=model.relation_offset.detach().cpu().numpy()
    rows=ff.groupby("image",sort=False).indices
    caps=[r.layer.canonical_form(x.caption,x.category) for x in ff.itertuples()]
    programs=[r.layer.parse_program(x) for x in caps]
    score={k:np.zeros(len(ff)) for k in ARMS}
    for shard in (0,1):
        with np.load(out/f"{split}_{shard}.npz") as z:
            for k in ARMS:
                valid=z[f"{k}_valid"]
                p,xy,_,*attrs=table_predictions(model,tuple(z[f"{k}_{c}"] for c in ("x","xy","valid")),device)
                for i,name in enumerate(z["names"]):
                    idx=rows[name]
                    prog=[programs[j] for j in idx]
                    if any(o is None for o in prog) or not valid[i].any():
                        continue
                    twin=r.layer.find_twins(ff.image.iloc[idx],[caps[j] for j in idx],ff.category.iloc[idx])
                    use=valid[i]
                    attr=[a[i,use] for a in attrs]
                    mat=r.layer.material_priors(attr)
                    try:
                        value=r.layer.rear.decode(prog,twin,p[i,use],attr,mat,xy[i,use],offset,
                                                   r.layer._draws(int(use.sum()),8,41,2048))
                    except ValueError as exc:
                        raise ValueError(f"Could not score scene {name}") from exc
                    score[k][idx]=value
    np.savez_compressed(out/f"{split}_scores.npz",ids=ff.id.to_numpy(),**score)
    print(split,"views scored",int(mask.sum()),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","scores"))
    p.add_argument("--root",type=Path,default=OUT)
    p.add_argument("--split",choices=("val","test"),required=True)
    p.add_argument("--shard",type=int,choices=(0,1),default=0)
    p.add_argument("--threads",type=int,default=3)
    a=p.parse_args()
    _,device=f.v.setup(421)
    if device.type=="cpu":
        torch.set_num_threads(a.threads)
    if a.stage=="prepare":
        prepare(a.root,a.split,a.shard,device)
    else:
        scores(a.root,a.split,device)

