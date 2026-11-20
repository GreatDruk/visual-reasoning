"""Score independent detail experiments against the archived competition baselines."""
import json
import argparse
from pathlib import Path

from scipy.special import expit, logit
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.research.detail_counts import ARMS as COUNTS, OUT, SEEDS, EXTRA, v
from scripts.research.detail_union import ARMS as TABLES, mid
from scripts.research.visible_fragment_detector import pairwise_iou
from scripts.research.visible_fragment_eval import score_metrics, boxes_from, inside, COLORS
from scripts.research.layer_repair import canonical_form, parse_program, EXACT

FOCUS=("58281.jpg","26268.jpg","78612.jpg","11105.jpg")
ACTIVE=("counting","comparison")
CUTOFF=(.30,.50,.70,.85,.95)


def read(path,frame,key="score"):
    with np.load(path) as f:
        data=f[key]
        if not np.array_equal(f["ids"],frame.id) or not np.isfinite(data).all():
            raise ValueError(f"Invalid prediction IDs or values: {path}")
        return data


def detection(arm,seed,ref,output=OUT):
    file=output/f"{arm}_{seed}_objects.npz"
    masks=output/f"{arm}_{seed}_objects.json"
    if file.exists():
        with np.load(file) as a:
            detections={}
            for n,p,b,ok,c in zip(a["names"],a["p"],a["boxes"],a["valid"],a["classes"]):
                detections[n]=[(float(q),mid(box),box,COLORS[col.argmax()])
                               for q,box,col in zip(p[ok],b[ok],c[ok,2:10])]
    elif masks.exists():
        detections={}
        for row in json.loads(masks.read_text()):
            detections.setdefault(row["image"],[]).append((row["prob"],np.array(row["peak"]),np.array(row["box"]),COLORS[np.argmax(row.get("color",[1]+[0]*7))]))
    else:
        return None
    values=[]
    for row in ref.itertuples(index=False):
        target=boxes_from(row,"r")
        front=boxes_from(row,"f")
        pool=[]
        for p,point,box,color in detections.get(row.image,[]):
            if not inside(point,target,2 if row.rear else 0):
                continue
            if row.rear:
                if pairwise_iou(target,box)>=.15 and pairwise_iou(front,box)<.50:
                    pool.append(p)
            elif color==row.color:
                # A control has one object in its reviewed area. Count the peak
                # probability of a second box after excluding duplicate boxes.
                pool.append((p,box))
        if row.rear:
            values.append((row.image,True,max(pool,default=0.)))
        else:
            pool=sorted(pool,key=lambda e:e[0],reverse=True)
            second=0.
            if pool:
                second=max((p for p,b in pool[1:] if pairwise_iou(pool[0][1],b)<.45),default=0.)
            values.append((row.image,False,second))
    frame=pd.DataFrame(values,columns=["image","rear","score"])
    frame.to_csv(output/f"{arm}_{seed}_overlap.csv",index=False)
    return {str(q):dict(rear_hit=int((frame.rear&(frame.score>=q)).sum()),rear_total=int(frame.rear.sum()),
                        control_extra=int((~frame.rear&(frame.score>=q)).sum()),
                        control_total=int((~frame.rear).sum())) for q in CUTOFF}


def variants(arm,raw,frame,base,counter,occ):
    active=frame.category.isin(ACTIVE).to_numpy()
    if arm in TABLES:
        easy=~frame.category.isin(["counting","comparison","spatial_counting"])&(occ*raw>0)
        whole=(.75*occ+.25*raw)*np.where(easy,1.5,1.)
        preserve=np.where(active,whole,base)
        confident=active&(np.abs(raw)>=3.5)
        # Only a confident score from the new union can directly replace a count score.
        gated=np.where(confident,raw,preserve)
        return {"union":whole,"core":preserve,"gated":gated}
    return {"blend":np.where(active,.5*(base+raw),base),
            "replace":np.where(active,raw,base),
            "with_r50":np.where(active,.5*(counter+raw),base)}


def export():
    result=json.loads((OUT/"detail_metrics.json").read_text())
    route=result["winner"]
    if not route:
        return
    arm,method=route
    frame=pd.read_csv(v.DATA/"test.csv")
    probability=read(EXTRA/"anchor_test.npz", frame, "probability")
    base=logit(probability.clip(1e-15,1-1e-15))
    counter=read(EXTRA/"r50_count_test.npz",frame)
    occ=read(EXTRA/"occ_test.npz",frame)
    raw=np.mean([read(OUT/f"{arm}_{seed}_test.npz",frame) for seed in SEEDS],axis=0)
    score=variants(arm,raw,frame,base,counter,occ)[method]
    answer=np.where(score==base,probability,expit(score))
    sample=pd.read_csv(v.DATA/"sample_submission.csv")
    if not sample.id.is_unique or not frame.id.is_unique or set(sample.id)!=set(frame.id):
        raise ValueError("Sample and prediction IDs differ")
    sample["label"]=pd.Series(answer,index=frame.id).loc[sample.id].to_numpy()
    sample.to_csv(OUT/f"{arm}_{method}.csv",index=False,float_format="%.17g")


def main(smoke=False):
    frame=pd.read_csv(v.DATA/"val.csv")
    if smoke:
        with np.load(OUT/"pyramid_421_val.npz") as a:
            ids=a["ids"]
        frame=frame[frame.id.isin(ids)].reset_index(drop=True)
    else:
        ids=frame.id.to_numpy()
    baselines={}
    for tag in ("anchor","r50_count"):
        path=EXTRA/f"{tag}_val.npz"
        if not smoke:
            baselines[tag]=read(path,frame)
        else:
            with np.load(path) as z:
                baselines[tag]=pd.Series(z["score"],index=z["ids"]).loc[ids].to_numpy()
    base,counter=baselines.values()
    occ=read(EXTRA/"occ_val.npz",frame) if not smoke else base
    mask=frame.category.isin(ACTIVE).to_numpy()
    y=frame.label.to_numpy()
    index,scenes=pd.factorize(frame.image)
    rng=np.random.default_rng(394)
    weights=rng.multinomial(len(scenes),np.full(len(scenes),1/len(scenes)),size=400)[:,index] if not smoke else []
    base_boot=np.asarray([roc_auc_score(y,base,sample_weight=w) for w in weights])
    count_boot=np.asarray([roc_auc_score(y,counter,sample_weight=w) for w in weights])
    compare,operating,counts={},{},{}
    actual=np.full(len(frame),-1,dtype=int)
    for row in frame[frame.label.eq(1)&frame.category.eq("counting")].itertuples():
        prog=parse_program(canonical_form(row.caption,row.category))
        if prog is not None and prog.op==EXACT and not prog.relation:
            actual[row.Index]=prog.number
    scores=frame.assign(anchor=base,r50_count=counter)
    ref=pd.read_csv(EXTRA/"rear_review_val.csv").fillna("")
    for arm in (*COUNTS,*TABLES):
        seed_scores=[]
        for seed in SEEDS:
            path=OUT/f"{arm}_{seed}_val.npz"
            if not path.exists():
                continue
            raw=read(path,frame)
            seed_scores.append((str(seed),raw))
            if arm in COUNTS:
                with np.load(path) as z:
                    if "count_a" in z:
                        pred=z["count_a"]
                        use=(actual>=0)&(pred>=0)
                        diff=np.rint(pred[use]).astype(int)-actual[use]
                        counts[f"{arm}_{seed}"]=dict(n=len(diff),under=int((diff<0).sum()),
                                                     over=int((diff>0).sum()),exact=int((diff==0).sum()))
            overlap=detection(arm,seed,ref)
            if overlap is not None:
                operating[f"{arm}_{seed}"]=overlap
        if not seed_scores:
            continue
        if len(seed_scores)==2:
            seed_scores.append(("both",np.mean([p for _,p in seed_scores],axis=0)))
        for replica, raw in seed_scores:
            stem=f"{arm}_{replica}"
            tests=variants(arm,raw,frame,base,counter,occ)
            for name,s in tests.items():
                key=f"{stem}_{name}"
                metric=score_metrics(frame,base,s)
                metric["delta_vs_r50_count"]=metric["auc"]-roc_auc_score(y,counter)
                core=mask
                if len(np.unique(y[core]))==2:
                    metric["core_auc"]=float(roc_auc_score(y[core],s[core]))
                if replica=="both" and not smoke:
                    resampled=np.asarray([roc_auc_score(y,s,sample_weight=w) for w in weights])
                    metric["delta_vs_union_ci95"]=np.quantile(resampled-base_boot,[.025,.975]).tolist()
                    metric["delta_vs_r50_ci95"]=np.quantile(resampled-count_boot,[.025,.975]).tolist()
                compare[key]=metric
                scores[key]=s
    groups={"union_tv_control":score_metrics(frame,base,base),"r50_count":score_metrics(frame,base,counter)}
    eligible=[]
    for arm in (*COUNTS,*TABLES):
        for method in variants(arm,base,frame,base,counter,occ):
            row=compare.get(f"{arm}_both_{method}")
            if row is None or smoke:
                continue
            if (row["auc"]>groups["r50_count"]["auc"] and row["delta_vs_union_ci95"][0]>0
                    and row["errors"]<=groups["r50_count"]["errors"]
                    and all(compare[f"{arm}_{seed}_{method}"]["auc"]>groups["union_tv_control"]["auc"] for seed in SEEDS)):
                eligible.append((row["auc"],arm,method))
    winner=max(eligible)[1:] if eligible else None
    report=dict(baselines=groups,experiments=compare,overlaps=operating,exact_count_diagnostics=counts,winner=winner)
    (OUT/"detail_metrics.json").write_text(json.dumps(report,indent=2))
    scores.to_csv(OUT/"detail_val.csv",index=False)
    focused=scores[scores.image.isin(FOCUS)].copy()
    for arm in COUNTS:
        for seed in SEEDS:
            path=OUT/f"{arm}_{seed}_val.npz"
            if path.exists():
                with np.load(path) as rows:
                    if "count_a" in rows:
                        focused[f"{arm}_{seed}_count"]=rows["count_a"][focused.index]
    focused.to_csv(OUT/"detail_cases.csv",index=False)
    print(json.dumps({"baselines":groups,"experiments":len(compare)}),flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--smoke",action="store_true")
    cli.add_argument("--export",action="store_true")
    args=cli.parse_args()
    export() if args.export else main(args.smoke)
