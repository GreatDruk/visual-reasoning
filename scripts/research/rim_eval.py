"""Evaluate annotated pixel localization, conditional attributes, and count replay."""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit, softmax
from sklearn.metrics import roc_auc_score

from scripts.research import layer_repair as layer
from scripts.research.rim_fragments import ARMS, EXTRA, OUT, SEEDS, pairwise_iou, scene_targets, v
from scripts.research.visible_fragment_eval import score_metrics, boxes_from
from scripts.research.detail_eval import detection
from src.models.slot_program import EXCLUSIVE_GROUPS

CATEGORIES=("counting","comparison","spatial_counting")
GROUPS=EXCLUSIVE_GROUPS
SCALE=np.array([v.WIDTH,v.HEIGHT,v.WIDTH,v.HEIGHT])
MODES=(("logit",1.0),("logit",.5),("replace",.5),("replace",1.0))
THRESHOLDS=(.50,.70,.85,.92,.97)


def center(b):
    return b[..., :2]+b[...,2:]/2


def inside(p,b,pad=0):
    return bool(((p>=b[:2]-pad)&(p<b[2:]+pad)).all())


def read(path,frame,key):
    with np.load(path) as z:
        if not np.array_equal(z["ids"],frame.id) or not np.isfinite(z[key]).all():
            raise ValueError(path)
        return z[key]


def merge_seeds(first,second):
    for k in ("names","index","boxes","front","source"):
        if not np.array_equal(first[k],second[k]):
            raise ValueError(f"Mismatched candidates for {k}")
    common=dict(first)
    common["data"]=(first["data"]+second["data"])/2
    common["data"][:,0]=np.minimum(first["data"][:,0],second["data"][:,0])
    common["pixels"]=np.minimum(first["pixels"],second["pixels"])
    common["compactness"]=np.minimum(first["compactness"],second["compactness"])
    common["peak"]=(first["peak"]+second["peak"])/2
    agrees=np.linalg.norm(first["peak"]-second["peak"],axis=1)<=5
    common["pixels"]*=agrees
    return common


def load_arm(arm, seed, split):
    for shard in (0,1):
        def opening(s):
            with np.load(OUT/f"{arm}_{s}_{split}_{shard}.npz") as z:
                return dict(z)
        yield opening(seed) if seed!="both" else merge_seeds(*[opening(s) for s in SEEDS])


def by_image(arma):
    result={}
    for z in arma:
        for i,b,f,s,d,peak,pix,compact in zip(z["index"],z["boxes"],z["front"],z["source"],z["data"],
                                              z["peak"],z["pixels"],z["compactness"]):
            name=z["names"][i]
            result.setdefault(str(name),[]).append(dict(box=b,front=f,source=float(s),data=d,
                peak=peak,pixels=int(pix),compactness=float(compact),prob=float(expit(d[0]))))
    return result


def class_prob(raw, attrs):
    size,color,joint=[softmax(x) for x in np.split(raw[1:17],[2,10])]
    selected=[list(x).index(next((a for a in attrs if a in x),None))
              if set(x)&set(attrs) else None for x in GROUPS]
    m,s=selected[2:]
    classes=joint.reshape(2,3)
    rows=slice(None) if m is None else m
    cols=slice(None) if s is None else s
    return float(classes[rows,cols].sum()*(1 if selected[0] is None else size[selected[0]])*
                 (1 if selected[1] is None else color[selected[1]]))


def measure_local(proposals,focus,controls,complete):
    out={}
    for row in focus:
        b=np.asarray(row["xyxy"])
        near=[p for p in proposals.get(row["image"],[]) if inside(center(p["box"]),b,3)]
        identified=[dict(candidate=p["box"].tolist(),p=p["prob"],raw_R50=p["source"],
                         matching_attrs=class_prob(p["data"],row["attributes_user"]),
                         pixels=p["pixels"],compactness=p["compactness"],peak=p["peak"].tolist(),
                         peak_in_rect=inside(p["peak"],b)) for p in near]
        out[row["image"]]=identified
    reviewed={}
    windows={}
    for r in controls.itertuples():
        a=boxes_from(r,"r")
        target=np.r_[a[:2],a[:2]+a[2:]]
        front=boxes_from(r,"f")
        selected=[p for p in proposals.get(r.image,[]) if p["pixels"]>=3 and p["compactness"]>=.40
                  and inside(p["peak"],target) and pairwise_iou(p["front"],front)>=.20]
        if not r.rear:
            selected=[p for p in selected if GROUPS[1][p["data"][3:11].argmax()]==r.color]
        windows[(r.image,int(r.rear))]=max((p["prob"] for p in selected),default=0.)
    for q in THRESHOLDS:
        good=sum(any(k["p"]>=q and k["pixels"]>=3 and k["peak_in_rect"] and
                     k["compactness"]>=.40 for k in candidates) for candidates in out.values())
        attributes=sum(any(k["p"]>=q and k["pixels"]>=3 and k["peak_in_rect"] and
                           k["compactness"]>=.40 and k["matching_attrs"]>=.25
                           for k in candidates) for candidates in out.values())
        hits=sum(int(t and s>=q) for (_,t),s in windows.items())
        false=sum(int(not t and s>=q) for (_,t),s in windows.items())
        wrong_scenes=sum(any(k["prob"]>=q and k["pixels"]>=3 and k["compactness"]>=.40 and bad(k,boxes)
                           for k in proposals.get(name,[])) for name,boxes in complete.items())
        reviewed[str(q)]=dict(focus_hits=good,focus_hits_with_attributes=attributes,
                               reviewed_hits=hits,reviewed_rear_total=sum(t for _,t in windows),
                               reviewed_false_positives=false,reviewed_control_total=sum(not t for _,t in windows),
                               unmatched_in_exact_total_scenes=wrong_scenes,
                               exact_total_scenes=len(complete))
    return out,reviewed


def bad(proposal,boxes):
    return bool((pairwise_iou(np.asarray(proposal["box"]),boxes)<.32).all())


def count_controls(frame):
    """Use only val exact-total captions and complete independent teacher scenes."""
    path=EXTRA/"val_teacher_boxes.npz"
    if not path.exists():
        return {}
    with np.load(path) as z:
        names=z["names"]
        count=scene_targets("val",names)
        subset=set(frame.image)
        return {n:b[k] for n,b,k,total in zip(names,z["box"],z["keep"],count)
                if total>0 and int(k.sum())==total and n in subset}


def valid_choice(p,q):
    return p["prob"]>=q and p["pixels"]>=3 and p["compactness"]>=.40


def table_options(z,i,p,proposals,q):
    need=[]
    active=z["valid"][i]
    boxes=z["x"][i,:,v.BOX]*SCALE
    for o in proposals:
        if not valid_choice(o,q):
            continue
        b=o["box"]
        dist=np.linalg.norm(center(boxes)-center(b),axis=-1)
        same=active&(pairwise_iou(b,boxes)>.42)&(dist<7)
        # Strong boxes usually already cover this candidate.
        if (same & (p[i]>.60)).any():
            continue
        j=np.flatnonzero(same)
        slot=int(np.cumsum(active)[j[dist[j].argmin()]]-1) if len(j) else -1
        need.append((o,slot))
    return sorted(need,key=lambda entry:-entry[0]["prob"])


def bank(split):
    import torch
    return [(z,layer.rear.bank_parts(z,split,torch.device("cpu")))
            for z in (layer.rear.collect(split,0),layer.rear.collect(split,1))]


def branch(frame,base,proposals,q,tables):
    assigned=frame.groupby("image",sort=False).indices
    captions=[layer.canonical_form(r.caption,r.category) for r in frame.itertuples()]
    change=np.zeros(len(frame))
    corrected=np.zeros(len(frame))
    accepted=0
    for z,(p,xy,valid,attrs,offsets) in tables:
        for i,name in enumerate(z["names"]):
            options=table_options(z,i,p,proposals.get(name,[]),q)
            if not options:
                continue
            proposal,j=options[0]
            rows=assigned[name]
            cap=[captions[r] for r in rows]
            twins=layer.find_twins(frame.image.iloc[rows],cap,frame.category.iloc[rows])
            programs=[layer.parse_program(c) for c in cap]
            k=valid[i]
            presence=p[i,k]
            attributes=[a[i,k] for a in attrs]
            position=xy[i,k]
            box=proposal["box"]
            if j<0:
                j=len(presence)
                presence=np.r_[presence,0.]
                attributes=[np.vstack([a,np.full(a.shape[1],1/a.shape[1])]) for a in attributes]
                position=np.vstack([position,(box[:2]+box[2:]*[.5,.88])/SCALE[:2]*2-1])
            uniforms=layer._draws(len(presence),8,41,2048)
            joint=layer.material_priors(attributes)
            before=layer.rear.decode(programs,twins,presence,attributes,joint,position,offsets,uniforms)
            presence[j]=proposal["prob"]
            for a,r in zip([attributes[0],attributes[1],joint],np.split(proposal["data"][1:17],[2,10])):
                a[j]=softmax(r)
            after=layer.rear.decode(programs,twins,presence,attributes,joint,position,offsets,uniforms)
            ok=frame.category.iloc[rows].isin(CATEGORIES).to_numpy()
            change[rows]=np.where(ok,after-before,0.)
            corrected[rows]=np.where(ok,after,base[rows])
            accepted+=int(ok.any())
    return change,corrected,accepted


def compose(mode,weight,base,delta,corrected):
    if mode=="logit":
        return base+weight*delta
    return np.where(delta!=0,(1-weight)*base+weight*corrected,base)


def bootstrap_auc(y,score,weights):
    """Weighted rank sums, with half credit for ties at each distinct score."""
    order=np.argsort(score,kind="stable")
    boundaries=np.r_[0,1+np.flatnonzero(np.diff(score[order]))]
    w=weights[:,order]
    positive=np.add.reduceat(w*y[order],boundaries,axis=1)
    negative=np.add.reduceat(w*(1-y[order]),boundaries,axis=1)
    wins=(positive*(negative.cumsum(axis=1)-.5*negative)).sum(axis=1)
    return wins/(positive.sum(axis=1)*negative.sum(axis=1))


def evaluate_scene_unet(frame,base,focus,review):
    outputs=[]
    overlaps={}
    objects={}
    for seed in SEEDS:
        path=OUT/f"scene_unet_{seed}_val.npz"
        if path.exists():
            outputs.append((str(seed),read(path,frame,"score")))
            overlaps[str(seed)]=detection("scene_unet",seed,review,output=OUT)
            items=json.loads((OUT/f"scene_unet_{seed}_objects.json").read_text())
            keys={p["image"]:p for p in focus}
            objects[str(seed)]=[dict(o,matching_attrs=class_prob(np.asarray(o["attributes"]),keys[o["image"]]["attributes_user"]))
                                for o in items if o["image"] in keys and
                                inside(o["peak"],np.array(keys[o["image"]]["xyxy"]))]
    if len(outputs)==2:
        outputs.append(("both",np.mean([y for _,y in outputs],axis=0)))
    results={}
    core=frame.category.isin(["counting","comparison"]).to_numpy()
    y=frame.label.to_numpy()
    for seed,values in outputs:
        chosen=np.where(core,values,base)
        combined=np.where(core,.5*(base+values),base)
        results[seed]=dict(overlap=overlaps.get(seed),pixel_candidates=objects.get(seed),core_auc=float(roc_auc_score(y[core],values[core])),
                           replacement=score_metrics(frame,base,chosen),blend=score_metrics(frame,base,combined),
                           focus={r["image"]:frame.loc[frame.image.eq(r["image"])&frame.category.isin(["counting","comparison"]),
                                               ["caption","label"]].assign(score=values[frame.image.eq(r["image"])&
                                               frame.category.isin(["counting","comparison"])]).to_dict("records") for r in focus})
    return results


def main():
    frame=pd.read_csv(v.DATA/"val.csv")
    base=read(EXTRA/"anchor_val.npz",frame,"score")
    ref=read(EXTRA/"anchor_val.npz",frame,"confirmed")
    y=frame.label.to_numpy()
    has_tables=all((layer.rear.SCRATCH/f"val_{i}.npz").is_file() for i in (0,1))
    tables=bank("val") if has_tables else []
    focus=json.loads((EXTRA/"reviewed_fragments_val.json").read_text())
    review=pd.read_csv(EXTRA/"rear_review_val.csv").fillna("")
    checks=count_controls(frame)
    scene,names=pd.factorize(frame.image)
    rng=np.random.default_rng(901)
    weights=rng.multinomial(len(names),np.full(len(names),1/len(names)),size=600)[:,scene].astype(np.int32)
    baseline=bootstrap_auc(y,base,weights)
    report=dict(base=score_metrics(frame,base,base),r50_count=score_metrics(frame,base,ref),
                experiments={},scene_unet=evaluate_scene_unet(frame,base,focus,review))
    columns={}
    options=[]
    observed={}
    for arm in ARMS:
        for seed in (*SEEDS,"both"):
            needed=[OUT/f"{arm}_{s}_val_{k}.npz" for s in (SEEDS if seed=="both" else (seed,)) for k in (0,1)]
            if not all(p.exists() for p in needed):
                continue
            proposals=by_image(load_arm(arm,seed,"val"))
            sub,operating=measure_local(proposals,focus,review,checks)
            arm_report=dict(focused=sub,operating=operating,metrics={})
            for q in ((.85,.92,.97) if tables else ()):
                delta,corrected,accepted=branch(frame,base,proposals,q,tables)
                for mode,w in MODES:
                    name=f"{arm}_{seed}_{q}_{mode}_{w}"
                    score=compose(mode,w,base,delta,corrected)
                    m=score_metrics(frame,base,score)
                    if seed=="both":
                        diff=bootstrap_auc(y,score,weights)-baseline if accepted else np.zeros(len(weights))
                        m["delta_ci95"]=np.quantile(diff,[.025,.975]).tolist()
                    arm_report["metrics"][name]=dict(**m,accepted=accepted)
                    key=(arm,q,mode,w)
                    if seed=="both":
                        if (accepted and m["delta_ci95"][0]>0 and
                            m["fixed"]>=m["regressed"] and all(x>report["base"]["auc"] for x in observed[key])):
                            options.append((m["auc"],list(key)))
                    else:
                        observed.setdefault(key,[]).append(m["auc"])
                    columns[name]=score
            report["experiments"][f"{arm}_{seed}"]=arm_report
            print("validated",arm,seed,flush=True)
            (OUT/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
    if "both" in report["scene_unet"]:
        core=frame.category.isin(["counting","comparison"]).to_numpy()
        results=report["scene_unet"]
        branches=[read(OUT/f"scene_unet_{seed}_val.npz",frame,"score") for seed in SEEDS]
        for mode,weight in (("blend",.5),("replacement",1.0)):
            out=np.where(core,(1-weight)*base+weight*np.mean(branches,axis=0),base)
            delta=bootstrap_auc(y,out,weights)-baseline
            results["both"][mode]["delta_ci95"]=np.quantile(delta,[.025,.975]).tolist()
            m=results["both"][mode]
            if (m["delta_ci95"][0]>0 and m["fixed"]>=m["regressed"] and
                all(results[str(seed)][mode]["auc"]>report["base"]["auc"] for seed in SEEDS)):
                options.append((m["auc"],["scene_unet",0.,mode,weight]))
            columns[f"scene_unet_both_{mode}"]=out
    report["winner"]=max(options)[1] if options else None
    (OUT/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
    scores=pd.DataFrame(dict(base=base,r50_count=ref,**columns),index=frame.index)
    pd.concat([frame,scores],axis=1).to_csv(OUT/"rim_val.csv",index=False)


def export():
    result=json.loads((OUT/"metrics.json").read_text(encoding="utf8"))
    winner=result["winner"]
    if not winner:
        return
    arm,q,mode,w=winner
    frame=pd.read_csv(v.DATA/"test.csv")
    prob=read(EXTRA/"anchor_test.npz",frame,"probability")
    base=logit(prob.clip(1e-15,1-1e-15))
    if arm=="scene_unet":
        core=frame.category.isin(["counting","comparison"]).to_numpy()
        scores=np.mean([read(OUT/f"scene_unet_{s}_test.npz",frame,"score") for s in SEEDS],axis=0)
        values=np.where(core,(1-w)*base+w*scores,base)
    else:
        proposals=by_image(load_arm(arm,"both","test"))
        delta,corrected,_=branch(frame,base,proposals,q,bank("test"))
        values=compose(mode,w,base,delta,corrected)
    result=np.where(values==base,prob,expit(values))
    sample=pd.read_csv(v.DATA/"sample_submission.csv")
    if not sample.id.is_unique or len(sample)!=len(frame) or set(sample.id)!=set(frame.id):
        raise ValueError("Sample IDs do not match test.csv")
    sample["label"]=pd.Series(result,index=frame.id).loc[sample.id].to_numpy()
    file="unet_scene.csv" if arm=="scene_unet" else f"rim_{arm}.csv"
    sample.to_csv(OUT/file,index=False,float_format="%.17g")


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--export",action="store_true")
    export() if cli.parse_args().export else main()
