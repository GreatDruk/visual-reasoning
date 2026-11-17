"""Measure the sources of ranking loss and group-held-out calibration."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, SplineTransformer


root = Path("artefacts/research")
val = pd.read_csv("data/val.csv")
base = [np.load(root / f"region_{name}.npz") for name in ("crop_production", "primitives")]
assert all(np.array_equal(run["ids"], val.id) for run in base)
joint = np.stack([run["joint"] for run in base], axis=-1)
marginal = np.stack([run["scores"] for run in base], axis=-1)
score = joint.mean(axis=-1)
assert (val.groupby("pair_id").size() == 2).all()
a, b = np.stack(list(val.groupby("pair_id").indices.values())).T
assert np.allclose(score[a], -score[b])

# The target is the correctness of the chosen answer, not the row orientation.
first = np.where(score[a] >= 0, a, b)
second = np.where(score[a] >= 0, b, a)
categories = val.category.to_numpy()
group = val.image.to_numpy()[first]
y = val.label.to_numpy()
columns = ["category", "mean", "crop", "silhouette", "crop_marginal", "silhouette_marginal", "disagreement"]
features = pd.DataFrame(np.column_stack((categories[first], score[first], joint[first], marginal[first],
                                        np.abs(joint[first, 0] - joint[first, 1]))), columns=columns)
features[columns[1:]] = features[columns[1:]].astype(float)
transform = ColumnTransformer([
    ("category", OneHotEncoder(sparse_output=False), ["category"]),
    ("confidence", SplineTransformer(n_knots=4, degree=1), columns[1:]),
])
logits = np.empty(len(first))
for train, heldout in GroupKFold(5).split(features, y[first], group):
    x = transform.fit_transform(features.iloc[train])
    model = LogisticRegression(C=.3, max_iter=300)
    model.fit(x, y[first[train]])
    logits[heldout] = model.decision_function(transform.transform(features.iloc[heldout]))
calibrated = np.empty(len(y))
calibrated[first], calibrated[second] = logits, -logits
for fraction in (0, .25, .5, .75, 1):
    mixed = (1-fraction)*score + fraction*calibrated
    print(f"calibration share {fraction:.2f}", roc_auc_score(y, mixed), flush=True)
np.savez(root / "rank_calibration.npz", ids=val.id.to_numpy(), base=score, crossfit=calibrated)

true = val.label == 1
positive = val[true].copy()
positive["crop"], positive["silhouette"] = joint[true].T
positive["ensemble"] = score[true]
negative = np.sort(score[~true])
lo = np.searchsorted(negative, score[true], side="left")
hi = np.searchsorted(negative, score[true], side="right")
positive["ranking_loss"] = len(negative) - (lo + hi)/2
positive["wrong"] = positive.ensemble < 0
positive.to_csv(root / "validation_rank_audit.csv", index=False)
print(positive.groupby("category")[["wrong", "ranking_loss"]].sum().sort_values("ranking_loss", ascending=False))
