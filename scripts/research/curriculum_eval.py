"""Score checkpoints on real scenes and select score rules with scene-grouped folds."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit, expit, softmax
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from scripts.research import rim_eval as r
from scripts.research.curriculum_train import OUT,MODES,SEEDS,tag,write_json
from scripts.research.visible_fragment_eval import score_metrics

DECISIONS=OUT/"decisions.json"
THRESHOLDS=(.70,.85,.92)
OP_POINTS=(0,1,2,4)
PAIR_SEEDS=(619,2401,4211)
LOCAL=tuple(tag(*x) for x in MODES if x[0]!="scene")+tuple(
    f"{arm}_legacy{n}" for arm in ("unet","r50") for n in (240,1200))
SCENE=("scene_staged","scene_noexact")


def files(name,split):
    return all((OUT/f"{name}_{seed}_{split}_{k}.npz").is_file() for seed in SEEDS for k in (0,1))


def load(name,split):
    return list(r.load_arm(name,"both",split))


def agree(names,split):
    shards=[load(name,split) for name in names]
    for items in zip(*shards):
        reference=items[0]
        for o in items[1:]:
            if any(not np.array_equal(o[k],reference[k]) for k in ("names","index","boxes","front")):
                raise ValueError("Models must score identical weak candidates")
        values=np.stack([o["data"] for o in items])
        peak=np.stack([o["peak"] for o in items])
        keep=np.stack([((o["pixels"]>=3)&(o["compactness"]>=.40)) for o in items])
        shape=softmax(values[:,:,11:17],axis=2).reshape(*values.shape[:2],2,3).sum(2).argmax(2)
        sign=np.stack([((shape==j)&keep).sum(0) for j in range(3)])
        majority=sign.argmax(0)
        consensus=keep&(shape==majority[None])
        votes=consensus.sum(0)
        combined=dict(reference)
        if not votes.size:
            yield combined
            continue
        weight=consensus/votes.clip(1)[None]
        combined["data"]=(values*weight[...,None]).sum(0)
        combined["pixels"]=np.where(votes>=2,6,0)
        combined["compactness"]=np.where(votes>=2,.70,0.)
        combined["peak"]=(peak*weight[...,None]).sum(0)
        yield combined


def bins(rows,proposals):
    shape_names=("b","s","c")
    observations=[]
    for row in rows.itertuples():
        front=r.boxes_from(row,"f")
        target=r.boxes_from(row,"r")
        target=np.r_[target[:2],target[:2]+target[2:]]
        windows=[v for v in proposals.get(row.image,[]) if v["pixels"]>=3 and v["compactness"]>=.40
                 and r.inside(v["peak"],target,2 if row.rear else 0)
                 and r.pairwise_iou(v["front"],front)>=.20]
        v=max(windows,key=lambda v:v["prob"],default=None)
        joint=np.ones(6)/6 if v is None else softmax(v["data"][11:17])
        s=shape_names[joint.reshape(2,3).sum(0).argmax()]
        m=("rubber","metal")[joint.reshape(2,3).sum(1).argmax()]
        observations.append(dict(image=row.image,rear=bool(row.rear),p=v["prob"] if v else 0.,
            shape_ok=s==row.shape,material_known=row.material in ("metal","rubber"),
            material_ok=m==row.material,same_color=row.same_color==1,highlight=row.highlight==1))
    df=pd.DataFrame(observations)
    if df.empty:
        return {}
    pos=df[df.rear]
    neg=df[~df.rear]
    has_mat=pos[pos.material_known]
    report={}
    for n in OP_POINTS:
        nonzero=np.sort(neg.p[neg.p>0].to_numpy())[::-1]
        limit=float(nonzero[n]) if len(nonzero)>n else 0.
        match=pos.p>limit
        report[str(n)]=dict(threshold=limit,false_positives=int((neg.p>limit).sum()),
           negatives=len(neg),reviewed_rear=len(pos),hits=int(match.sum()),
           shape_hits=int((match&pos.shape_ok).sum()),known_material=len(has_mat),
           joint_hits=int((match&pos.shape_ok&pos.material_ok).sum()),
           same_color_total=int(pos.same_color.sum()),same_color_shape_hits=int((match&pos.same_color&pos.shape_ok).sum()),
           highlight_controls=int(neg.highlight.sum()),highlight_false_positives=int(((neg.p>limit)&neg.highlight).sum()))
    return report


def scene_info(frame,base,name,split):
    scores=[]
    for seed in SEEDS:
        file=OUT/f"{name}_{seed}_{split}.npz"
        if not file.is_file():
            return None,None
        scores.append(r.read(file,frame,"score"))
    score=np.mean(scores,0)
    if split=="test":
        return score,None
    mask=frame.category.isin(["counting","comparison"]).to_numpy()
    diagnostic={}
    ref=pd.read_csv(r.EXTRA/"curriculum_review.csv").fillna("")
    exact=r.count_controls(frame)
    for seed in SEEDS:
        path=OUT/f"{name}_{seed}_objects.json"
        if path.is_file():
            overlap=r.detection(name,seed,ref,output=OUT)
            v=json.loads(path.read_text(encoding="utf8"))
            objects=defaultdict(list)
            for o in v:
                objects[o["image"]].append(o)
            extra=0; under=0; exact_seen=0; unmatched=0
            for n,boxes in exact.items():
                items=[o for o in objects[n] if o["prob"]>=.5]
                extra+=max(0,len(items)-len(boxes))
                under+=max(0,len(boxes)-len(items))
                exact_seen+=len(items)==len(boxes)
                unmatched+=sum(r.bad(o,boxes) for o in items)
            diagnostic[str(seed)]=dict(overlap=overlap,exact_control_scenes=len(exact),
               exact_seen=exact_seen,excess_masks=extra,deficit_masks=under,unmatched_masks=unmatched)
    return score,dict(core_auc=float(roc_auc_score(frame.label[mask],score[mask])),
                      replacement=score_metrics(frame,base,np.where(mask,score,base)),
                      detections=diagnostic)


def keys(p):
    return json.dumps(p,separators=(",",":"))


def scoring(frame,base,plan,split,tables=None,branch_cache=None,scene_cache=None):
    kind,*param=plan
    if kind=="base":
        return base
    if kind=="joint":
        a,b=param
        first=scoring(frame,base,a,split,tables,branch_cache,scene_cache)
        second=scoring(frame,base,b,split,tables,branch_cache,scene_cache)
        return np.where(second!=base,second,first)
    if kind=="scene":
        name,scope,weight,margin=param
        if scene_cache is not None and name in scene_cache:
            s=scene_cache[name]
        else:
            s,_=scene_info(frame,base,name,split)
        mask=frame.category.isin(["counting","comparison"]).to_numpy()
        if scope=="colors":
            mask=frame.category.eq("comparison").to_numpy()&(frame.caption.str.count(
                r"\b(?:gray|red|blue|green|brown|purple|cyan|yellow) objects\b").to_numpy()==2)
        return np.where(mask&(np.abs(base)<=margin),(1-weight)*base+weight*s,base)
    models,q,weight=param
    k=(tuple(models),q)
    if branch_cache is not None and k in branch_cache:
        delta=branch_cache[k]
    else:
        z=load(models[0],split) if len(models)==1 else agree(models,split)
        delta,_,_=r.branch(frame,base,r.by_image(z),q,tables)
        if branch_cache is not None:
            branch_cache[k]=delta
    return base+weight*np.clip(delta,-2,2)


def supported_plans(model_list,scene_list):
    single=[["local",[a],q,.5] for a in model_list for q in THRESHOLDS]
    consensus=[]
    for candidates in (("unet_staged","r50_staged","efficient_staged","r18_staged","convnext_staged"),
                       ("unet_mixed","r50_mixed","efficient_mixed","r18_mixed","convnext_mixed"),
                       ("unet_boundary","r50_staged","efficient_staged")):
        models=[m for m in candidates if m in model_list]
        if len(models)>=2:
            consensus.extend([["local",models,q,.5] for q in THRESHOLDS])
    scene_plan=[["scene",a,scope,w,margin] for a in scene_list
                for scope,w,margin in (("core",.25,100),("core",.25,3),("colors",.5,3))]
    for a in (a for a in scene_plan if a[2]=="colors"):
        for b in consensus:
            yield ["joint",b,a]
    yield from single+consensus+scene_plan


def main(args):
    r.OUT=OUT
    if args.export:
        plan=json.loads(DECISIONS.read_text())["selected"]
        if not plan:
            return
        frame=pd.read_csv(r.v.DATA/"test.csv")
        pr=r.read(r.EXTRA/"anchor_test.npz",frame,"probability")
        base=logit(pr.clip(1e-15,1-1e-15))
        tb=r.bank("test") if plan[0] in ("local","joint") else None
        out=scoring(frame,base,plan,"test",tb)
        names=frame.id
        sample=pd.read_csv(r.v.DATA/"sample_submission.csv")
        if not names.is_unique or not sample.id.is_unique or set(names)!=set(sample.id):
            raise ValueError("Test IDs do not match sample_submission.csv")
        values=np.where(out==base,pr,expit(out))
        if not np.isfinite(values).all():
            raise ValueError("Non-finite test values")
        sample["label"]=pd.Series(values,index=names).loc[sample.id].to_numpy()
        sample.to_csv(OUT/f"{args.filename}.csv",index=False,float_format="%.17g")
        return
    frame=pd.read_csv(r.v.DATA/"val.csv")
    base=r.read(r.EXTRA/"anchor_val.npz",frame,"score")
    focus=json.loads((r.EXTRA/"reviewed_fragments_val.json").read_text())
    focus_mask=frame.image.isin([f["image"] for f in focus]).to_numpy()
    old=json.loads((r.EXTRA/"old_rim_metrics.json").read_text())
    review=pd.read_csv(r.EXTRA/"curriculum_review.csv").fillna("")
    checks=r.count_controls(frame)
    report=dict(base=score_metrics(frame,base,base),old_rim_base=old["base"],models={},scene={},folds={})
    report["old_rim_controls"]={k: max(v["metrics"].values(),key=lambda a:a["auc"])
                                for k,v in old["models"].items() if k.endswith("_both") and v["metrics"]}
    models=[a for a in LOCAL if files(a,"val")]
    local_proposals={a:r.by_image(load(a,"val")) for a in models}
    available=[a for a in SCENE if all((OUT/f"{a}_{s}_val.npz").exists() for s in SEEDS)]
    scene_scores={}
    for name in available:
        scene_scores[name],report["scene"][name]=scene_info(frame,base,name,"val")
    for name,proposals in local_proposals.items():
        report["models"][name]=dict(real=bins(review,proposals),
            operating=r.measure_local(proposals,[],review,checks)[1],
            inner={str(s):json.loads((OUT/f"{name.removesuffix('240').removesuffix('1200')}_{s}.json").read_text()).get("checkpoints",{})
                   for s in SEEDS})
    cache={}
    tables=r.bank("val") if all((r.layer.rear.SCRATCH/f"val_{k}.npz").exists() for k in (0,1)) else None
    if tables is None:
        models=[]
    plans=[["base"],*supported_plans(models,available)]
    score={"base":base}
    for p in plans[1:]:
        score[keys(p)]=scoring(frame,base,p,"val",tables,cache,scene_scores)
    report["scoring"]={k:score_metrics(frame,base,s) for k,s in score.items()}
    y=frame.label.to_numpy()
    scene_index,names=pd.factorize(frame.image)
    nonfocus=~focus_mask
    heldout=frame.loc[nonfocus].reset_index(drop=True)
    nids,nnames=pd.factorize(heldout.image)
    rng=np.random.default_rng(962)
    w=rng.multinomial(len(nnames),np.full(len(nnames),1/len(nnames)),size=400)[:,nids].astype(np.int32)
    bauc=r.bootstrap_auc(y[nonfocus],base[nonfocus],w)
    base_nonfocus=roc_auc_score(y[nonfocus],base[nonfocus])
    candidates=list(score)
    oof_rows={}
    for seed in PAIR_SEEDS:
        oof=base.copy()
        picks=[]
        for a,b in KFold(5,shuffle=True,random_state=seed).split(names):
            fit=np.isin(scene_index,a)&nonfocus
            held=np.isin(scene_index,b)
            candidate=max(candidates,key=lambda key:roc_auc_score(y[fit],score[key][fit]))
            oof[held]=score[candidate][held]
            picks.append(candidate)
        oof_rows[str(seed)]=oof
        m=score_metrics(frame,base,oof)
        m["picks"]=picks
        m["without_diagnostic"]=score_metrics(heldout,base[nonfocus],oof[nonfocus])
        m["auc_excluding_diagnostic"]=m["without_diagnostic"]["auc"]
        m["delta_ci95"]=np.quantile(r.bootstrap_auc(y[nonfocus],oof[nonfocus],w)-bauc,[.025,.975]).tolist()
        report["folds"][str(seed)]=m
    valid=all(v["auc_excluding_diagnostic"]>base_nonfocus and v["delta_ci95"][0]>0
              and v["without_diagnostic"]["fixed"]>=v["without_diagnostic"]["regressed"] for v in report["folds"].values())
    key=max(candidates,key=lambda k:roc_auc_score(y[nonfocus],score[k][nonfocus]))
    report["prospective"]=json.loads(key) if key!="base" else None
    report["selected"]=report["prospective"] if valid else None
    # The four manually reviewed cases are opened only for post-selection reporting.
    report["focus_diagnostic"]={a:r.measure_local(p,focus,review.iloc[:0],{})[0] for a,p in local_proposals.items()}
    report["focus_rows"]=frame.loc[focus_mask,["id","image","caption","label"]].assign(
        base=base[focus_mask],candidate=score[key][focus_mask]).to_dict("records")
    write_json(DECISIONS,report)
    np.savez_compressed(OUT/"validation_scores.npz",ids=frame.id.to_numpy(),
                        choices=np.asarray(list(score)),scores=np.stack(list(score.values())),**oof_rows)
    print("base",report["base"],"prospective",report["prospective"],"selected",report["selected"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export",action="store_true")
    parser.add_argument("--filename",default="curriculum")
    main(parser.parse_args())

