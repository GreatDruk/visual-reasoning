"""Check whether both trained occlusion arms agree on transferable corrections."""
import io
import json
from itertools import product
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

RUN=Path("artefacts/kaggle_suite/layer_run")
CORE=("counting","comparison","spatial_counting")


def architecture(archive,frame,arm):
    effects,contrasts,boxes=[],[],[]
    for seed in (421,853):
        proposals,predictions=[],[]
        for shard in (0,1):
            prefix=f"layer_result/{arm}_{seed}"
            table=pd.DataFrame(json.loads(archive.read(f"{prefix}_{shard}_candidates.json")))
            table=table[(table.probability>=.87)&(table.contrast>=.75)&(table.pixels>=5)]
            # Match the inference policy. Tied probabilities use the first box.
            proposals.append(table.sort_values("probability",ascending=False,kind="stable")
                             .drop_duplicates("image").set_index("image"))
            with np.load(io.BytesIO(archive.read(f"{prefix}_val_{shard}.npz"))) as saved:
                predictions.append(pd.Series(saved["delta"],index=saved["ids"]))
        delta=pd.concat(predictions)
        assert delta.index.is_unique and delta.index.isin(frame.id).all()
        dx=delta.reindex(frame.id,fill_value=0.).to_numpy()
        effects.append(np.where(frame.category.isin(CORE),dx,0.))
        evidence=pd.concat(proposals).reindex(frame.image)
        contrasts.append(evidence.contrast.fillna(0).to_numpy())
        boxes.append(np.stack([r if isinstance(r,list) else [0.,0.,0.,0.] for r in evidence.box]))
    return map(np.stack,(effects,contrasts,boxes))


def intersection_over_union(a,b):
    upper=np.minimum(a[:,:2]+a[:,2:],b[:,:2]+b[:,2:])
    lower=np.maximum(a[:,:2],b[:,:2])
    overlap=(upper-lower).clip(0).prod(1)
    return overlap/(a[:,2:].prod(1)+b[:,2:].prod(1)-overlap).clip(1)


def main():
    with zipfile.ZipFile(RUN/"layer_results.zip") as z:
        f=pd.read_csv(z.open("verify_result/val_rows.csv"))
        verify,vc,vb=architecture(z,f,"verify")
        split,sc,sb=architecture(z,f,"split")
    control=f.baseline.to_numpy()
    for arm,x in (("verify",verify),("split",split)):
        assert np.allclose(f[arm],control+.75*x.mean(0))
    with zipfile.ZipFile(RUN.parent/"rear_run/rear_results.zip") as z:
        a=pd.read_csv(z.open("verify_result/val_rows.csv"))
        assert a.id.equals(f.id)
        hidden=a.base.to_numpy()

    actual=pd.read_csv("data/val.csv")
    assert f[["id","image","label","category"]].equals(actual[["id","image","label","category"]])
    signs=(verify[0]*verify[1]>0)&(split[0]*split[1]>0)&(verify[0]*split[0]>0)
    same=(intersection_over_union(*vb)>.99)&(intersection_over_union(*sb)>=.50)
    nearby=(intersection_over_union(vb[0],sb[0])>.15)&(intersection_over_union(vb[1],sb[1])>.15)
    contrast=np.minimum(vc.min(0),sc.min(0))
    average=(verify.sum(0)+split.sum(0))/4

    def count(score):
        return dict(auc=float(roc_auc_score(y,score)),wrong=int(((y==1)&(score<=0)).sum()))

    results={}
    y=f.label.to_numpy()
    scene,names=pd.factorize(f.image)
    choices=[(0.,0.),*product((1.,2.,4.),(.10,.25,.5))]
    for name,base in (("union_tv",control),("count_hidden",hidden)):
        scores={k:base+np.where(signs&same&nearby&(contrast>=k[0]),average*k[1],0.) for k in choices}
        def best(mask):
            return max(choices,key=lambda k:roc_auc_score(y[mask],scores[k][mask]))
        key=best(np.ones(len(f),dtype=bool))
        chosen=scores[key]
        record=dict(baseline=count(base),best_grid=count(chosen),choice=key,
                    affected_scenes=int(f.image[chosen!=base].nunique()),cross_validation={})
        for seed in (619,2401,4211):
            oof=base.copy()
            selected=[]
            for tr,te in KFold(5,shuffle=True,random_state=seed).split(names):
                fit,out=np.isin(scene,tr),np.isin(scene,te)
                k=best(fit);selected.append(k);oof[out]=scores[k][out]
            record["cross_validation"][seed]=dict(**count(oof),picks=selected)
            if seed==619:
                rng=np.random.default_rng(439)
                d=[]
                for _ in range(1000):
                    w=rng.multinomial(len(names),np.full(len(names),1/len(names)))[scene]
                    d.append(roc_auc_score(y,oof,sample_weight=w)-roc_auc_score(y,base,sample_weight=w))
                record["delta_ci"]=np.quantile(d,[.025,.975]).tolist()
        use=~f.image.eq("26268.jpg").to_numpy()
        record["without_diagnostic_scene_delta"]=float(roc_auc_score(y[use],chosen[use])-roc_auc_score(y[use],base[use]))
        results[name]=record
    (RUN/"consensus_metrics.json").write_text(json.dumps(results,indent=2),encoding="utf8")
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
