"""Compare spatial repairs with grouped holdouts and export distinct candidates."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from scripts.research import spatial_replay as spatial
from scripts.research.visible_fragment_eval import score_metrics

FOCUS=("58281.jpg","26268.jpg","78612.jpg","11105.jpg")
GEOMETRY=("geometry","geometry_delta")
CONTOUR_MODELS=(
    "r50_mixed/views/new/0.15","r50_mixed/views/existing/0.15",
    "r50_mixed/views/new/0.35","r50_mixed/one/new/0.35",
    "unet_boundary/views/new/0.55","agreement/views/new/0.35",
)


def load(path,frame):
    if not path.is_file():
        return None
    with np.load(path) as z:
        data=dict(z)
    if not np.array_equal(frame.id,data["ids"]):
        raise ValueError("Mismatched sample IDs")
    return data


def evidence(root,split):
    frame,old,base,selected=spatial.frames(root,split)
    path=root/"spatial"
    geo=load(path/f"{split}_geometry.npz",frame)
    contour=load(path/f"{split}_branches.npz",frame)
    g={}
    if geo is not None:
        g=dict(geometry=np.where(selected,geo["score"]-base,0.),geometry_delta=geo["delta"])
    c={}
    if contour is not None:
        c={str(n):delta for n,delta in zip(contour["keys"],contour["delta"]) if n in CONTOUR_MODELS}
    return frame,base,selected,g,c


def build(root,split):
    f,b,use,geom,contour=evidence(root,split)
    a=f.category.eq("spatial_counting").to_numpy()
    result={"shape_mat":b}
    scalar=[(0.5,2.),(0.5,3.),(1.,2.),(1.,3.)]
    def compose(delta,w,margin):
        keep=a&use&(np.abs(b)<=margin)
        return b+np.where(keep,w*np.clip(delta,-3,3),0.)
    g_options=[(k,d,w,m) for k,d in geom.items() for w,m in scalar]
    c_options=[(k,d,w,m) for k,d in contour.items() for w,m in scalar]
    for k,d,w,m in (g_options+c_options):
        result[f"{k}@{w}@{m}"]=compose(d,w,m)
    if geom and contour:
        g=geom["geometry_delta"]
        for k,d in contour.items():
            agree=(np.sign(g)==np.sign(d))&(g!=0)
            for w,m in scalar:
                result[f"agree/{k}@{w}@{m}"]=compose(np.where(agree,.5*(g+d),0.),w,m)
            result[f"stack/{k}@1.0@2.0"]=compose(g+np.where(agree,d,0.),1.,2.)
    return f,b,result


def inv(y,s):
    p=s[y==1]
    n=np.sort(s[y==0])
    return len(p)*len(n)-.5*(np.searchsorted(n,p,"left").sum()+np.searchsorted(n,p,"right").sum())


def family(key):
    if key=="shape_mat":
        return key
    if key.startswith(("agree/","stack/")):
        return "both"
    return "geo" if key.startswith("geometry") else "contour"


def validate(root):
    f,b,choices=build(root,"val")
    y=f.label.to_numpy()
    mask=~f.image.isin(FOCUS).to_numpy()
    scene,names=pd.factorize(f.image)
    base=score_metrics(f,b,b)
    def m(a):
        pairs=score_metrics(f,b,a)
        pairs["inversions_without_four"]=inv(y[mask],a[mask])
        pairs["affected_scenes"]=int(f.loc[a!=b,"image"].nunique())
        return pairs
    info={k:m(s) for k,s in choices.items()}
    seeds=(619,2401,4211)
    oof={}
    preferred={}
    for tag in ("geo","contour","both","all"):
        local={k:a for k,a in choices.items() if tag=="all" or family(k) in (tag,"shape_mat")}
        ranking=sorted(local,key=lambda k:(info[k]["inversions_without_four"],info[k]["regressed"],info[k]["inversions"]))
        useful=[k for k in ranking if k!="shape_mat" and info[k]["regressed"]<=info[k]["fixed"]
                and info[k]["inversions_without_four"]<info["shape_mat"]["inversions_without_four"]]
        if useful:
            preferred[tag]=useful[0]
        if len(ranking)==1:
            continue
        print(tag,[(k,info[k]) for k in ranking[:4]],flush=True)
        oof[tag]={}
        for seed in seeds:
            s=b.copy()
            selected=[]
            for tr,te in KFold(5,shuffle=True,random_state=seed).split(names):
                fit=np.isin(scene,tr)&mask
                hold=np.isin(scene,te)
                pick=min(local,key=lambda k:inv(y[fit],local[k][fit]))
                s[hold]=local[pick][hold]
                selected.append(pick)
            oof[tag][str(seed)]=dict(**m(s),selected=selected)
        print("oof",tag,[(k,a["inversions_without_four"],a["fixed"],a["regressed"])
                        for k,a in oof[tag].items()],flush=True)
    payload=dict(baseline=base,methods=info,folds=oof,preferred=preferred)
    (root/"spatial"/"validation.json").write_text(json.dumps(payload,indent=2),encoding="utf8")
    with (root/"spatial"/"val_scores.npz").open("wb") as out:
        np.savez_compressed(out,ids=f.id.to_numpy(),keys=list(choices),scores=np.stack(list(choices.values())))
    return payload


def export(root,selection):
    f,b,choices=build(root,"test")
    path=root/"spatial"/"validation.json"
    report=json.loads(path.read_text(encoding="utf8"))
    plan=selection if selection else report["preferred"]
    reference=pd.read_csv(root/"shape_mat.csv",float_precision="round_trip").set_index("id")
    p=reference.loc[f.id,"label"].to_numpy()
    sample=pd.read_csv(spatial.rim.v.DATA/"sample_submission.csv")
    if not (f.id.is_unique and sample.id.is_unique and set(f.id)==set(sample.id)):
        raise ValueError("IDs differ from sample submission")
    files={"geo":"shape_geo.csv","contour":"shape_contour.csv","both":"shape_spatial.csv"}
    seen=set()
    for method,name in files.items():
        key=plan.get(method)
        if key not in choices or key in seen:
            continue
        seen.add(key)
        score=choices[key]
        new=np.where(score==b,p,expit(score))
        if not np.isfinite(new).all() or np.any((new<0)|(new>1)):
            raise ValueError("Invalid prediction scores")
        table=sample.assign(label=pd.Series(new,index=f.id).loc[sample.id].to_numpy())
        destination=root/name
        table.to_csv(destination,index=False,float_format="%.17g")
        result=pd.read_csv(destination,float_precision="round_trip")
        assert result.id.equals(sample.id) and result.label.between(0,1).all()
        print(destination,key,int(np.count_nonzero(new!=p)),"updated rows",flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--root",type=Path,default=Path("artefacts/kaggle_suite/curriculum_run"))
    a.add_argument("--export",action="store_true")
    a.add_argument("--selection",type=json.loads)
    p=a.parse_args()
    export(p.root,p.selection) if p.export else validate(p.root)

