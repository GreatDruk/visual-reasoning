"""Group-held-out calibration of confidence from several visual models."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, SplineTransformer


ROOT = Path("artefacts/research")


def choices(frame, model_scores):
    scores = model_scores.mean(1)
    pairs = np.stack(list(frame.groupby("pair_id").indices.values()))
    assert np.allclose(scores[pairs[:,0]],-scores[pairs[:,1]])
    a,b = pairs.T
    left = np.where(scores[a]>=0,a,b)
    right = np.where(scores[a]>=0,b,a)
    group = frame.image.to_numpy()[left]
    y = frame.label.to_numpy()
    numeric = {
        "mean": scores[left], "spread": model_scores[left].std(1),
        "max":model_scores[left].max(1),"min":model_scores[left].min(1),
        **{f"m{i}":v[left] for i,v in enumerate(model_scores.T)},
    }
    x = pd.DataFrame({**numeric,"category":frame.category.to_numpy()[left],"scene":group})
    x["weakest"] = x.groupby("scene")["mean"].transform("min")
    cols = [*numeric,"weakest"]
    transformer = ColumnTransformer([
        ("category",OneHotEncoder(sparse_output=False,handle_unknown="ignore"),["category"]),
        ("confidence",SplineTransformer(n_knots=4,degree=1),cols),
    ])
    calibrated = np.empty(len(y))
    for train, held in GroupKFold(5).split(x,y[left],group):
        learner = LogisticRegression(C=.27,max_iter=300)
        learner.fit(transformer.fit_transform(x.iloc[train]),y[left[train]])
        p = learner.decision_function(transformer.transform(x.iloc[held]))
        calibrated[left[held]],calibrated[right[held]] = p,-p
    return scores,calibrated


if __name__ == "__main__":
    v = pd.read_csv("data/val.csv")
    parts = (("center_proposals8","permissive"),("region_count_supervision","joint"),
             ("region_learned_fusion","joint"),("native_epoch12","joint"))
    arrays = []
    for name,key in parts:
        with np.load(ROOT/f"{name}.npz") as f:
            assert np.array_equal(f["ids"],v.id)
            arrays.append(f[key])
    raw,confidence = choices(v,np.column_stack(arrays))
    for alpha in (0,.25,.5,.75,1.):
        score = (1-alpha)*raw + alpha*confidence
        print(alpha,roc_auc_score(v.label,score),flush=True)
    np.savez(ROOT/"rank_consistency.npz",ids=v.id,raw=raw,confidence=confidence)
