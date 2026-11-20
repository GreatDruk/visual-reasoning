"""Compare verified union predictions with scene-grouped selection of the blend weight."""
from pathlib import Path
import io
import zipfile

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artefacts/kaggle_suite/verifier_run"


def scores(archive, key, ids):
    with zipfile.ZipFile(archive) as z, np.load(io.BytesIO(z.read(key))) as data:
        if not np.array_equal(ids, data["ids"]):
            raise ValueError(key)
        return data["score"]


def inputs(split, ids):
    anchor = scores(RUN.parent/"clevr_verifier.zip", f"priors/occ_{split}.npz", ids)
    union = np.mean([scores(RUN/"verifier_results.zip", f"control_{s}_{split}.npz", ids)
                     for s in (421,853)], axis=0)
    return anchor, union


def main():
    frame = pd.read_csv(ROOT/"data/val.csv")
    anchor, union = inputs("val",frame.id)
    label = frame.label.to_numpy()
    scene = pd.factorize(frame.image)[0]
    metric = roc_auc_score
    weights = np.array([.25,.375,.50,.625,.75])
    candidates = [(1-w)*anchor+w*union for w in weights]
    base = candidates[0]
    for w,x in zip(weights,candidates):
        print(w,metric(label,x),sum((label==1)&(x<=0)))
    oof = np.empty_like(base)
    picks = []
    for fit, hold in KFold(5,shuffle=True,random_state=1801).split(np.unique(scene)):
        train, val = np.isin(scene,fit), np.isin(scene,hold)
        k = np.argmax([metric(label[train],x[train]) for x in candidates])
        picks.append(weights[k])
        oof[val] = candidates[k][val]
    print("fold picks",picks,"oof",metric(label,oof))
    blend_oof_gain = metric(label,oof) > metric(label,base)
    winner = np.argmax([metric(label,x) for x in candidates])
    w = weights[winner]
    rng = np.random.default_rng(1801)
    deltas = []
    for _ in range(800):
        counts = rng.multinomial(1000,np.full(1000,.001))[scene]
        deltas.append(metric(label,oof,sample_weight=counts)-metric(label,base,sample_weight=counts))
    print("paired OOF delta interval (fixed fold predictions)",np.quantile(deltas,[.025,.975]))
    errors = frame[(label==1)&(base<0)].assign(margin=base[(label==1)&(base<0)])
    print(errors.sort_values("margin")[["image","caption","margin"]].head(5).to_string(index=False))
    test = pd.read_csv(ROOT/"data/test.csv")
    a,b = inputs("test",test.id)
    # Promote agreed, non-counting decisions in the global AUC ordering.
    def adjusted(rows, anchor, union, factor):
        reliable = ~rows.category.isin(["counting","spatial_counting","comparison"]) & (anchor*union > 0)
        return (.75*anchor+.25*union)*np.where(reliable,factor,1.)

    factors = [1.,1.125,1.25,1.5]
    adjusted_val = [adjusted(frame,anchor,union,t) for t in factors]
    picks = []
    for fit, hold in KFold(5,shuffle=True,random_state=1801).split(np.unique(scene)):
        train, val = np.isin(scene,fit), np.isin(scene,hold)
        k = np.argmax([metric(label[train],x[train]) for x in adjusted_val])
        picks.append(factors[k])
        oof[val] = adjusted_val[k][val]
    factor = factors[np.argmax([metric(label,x) for x in adjusted_val])]
    print("agreement factors",picks,"OOF AUC",metric(label,oof),"final factor",factor)
    agree = adjusted(test,a,b,factor)

    for name,frac in [("occ_union",.25),("union_mix",w),("union_cal",.25)]:
        if name=="union_mix" and (w==.25 or not blend_oof_gain):
            continue
        if name=="union_cal" and metric(label,oof)<=metric(label,base):
            continue
        sample = pd.read_csv(ROOT/"data/sample_submission.csv")
        assert test.id.is_unique and sample.id.is_unique and set(sample.id)==set(test.id)
        logits = agree if name=="union_cal" else (1-frac)*a+frac*b
        sample["label"] = pd.Series(expit(logits),index=test.id).loc[sample.id].to_numpy()
        path = RUN/f"{name}.csv"
        sample.to_csv(path,index=False,float_format="%.17g")
        assert pd.read_csv(path,float_precision="round_trip").equals(sample) and sample.label.between(0,1).all()
        print(path.name, len(sample), frac)

if __name__=="__main__":
    main()

