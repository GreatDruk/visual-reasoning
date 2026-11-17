"""Compare frozen full-validation predictions from the reconstruction experiments."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


root = Path("artefacts/research")
val = pd.read_csv("data/val.csv")
labels = val.label.to_numpy()
groups = np.stack(list(val.groupby("image").indices.values()))


def load(path, key="joint"):
    with np.load(root/f"{path}.npz") as data:
        assert np.array_equal(data["ids"],val.id)
        return data[key]


old = np.mean([load(f"region_{m}") for m in ("primitives","seams","count_supervision")],axis=0)
counts = load("region_count_supervision")
test = {"previous":old}
for mode in ("control","center_shape","split"):
    test["projection_"+mode] = np.mean(
        [load(f"projection_{m}",mode) for m in ("silhouette","seams","counts")],axis=0
    )
for c,f,d in itertools.product((8,16),(8,16),("epoch16","exact_epoch21","natural_epoch21")):
    name = f"native_{d}"
    if not (root/f"{name}.npz").is_file():
        continue
    fused = "region_learned_fusion" if f==8 else f"region_learned_fusion{f}"
    test[f"center{c}_fusion{f}_{d}"] = np.mean(
        [load(f"center_proposals{c}","permissive"),counts,load(fused),load(name)],axis=0
    )


def measure(x):
    auc = roc_auc_score(labels,x)
    wrong = np.sum((labels*2-1)*x<0)//2
    by_category = {}
    for category, frame in val.groupby("category"):
        by_category[category] = roc_auc_score(frame.label,x[frame.index])
    return {"auc":float(auc),"wrong_pairs":int(wrong),"category_auc":by_category}


measured = {name:measure(x) for name,x in test.items()}
best = max((name for name in test if name.startswith("center")),key=lambda n:measured[n]["auc"])
rng = np.random.default_rng(42)
delta = []
for _ in range(500):
    ix = groups[rng.integers(len(groups),size=len(groups))].ravel()
    y = labels[ix]
    delta.append(roc_auc_score(y,test[best][ix])-roc_auc_score(y,old[ix]))
summary = {"best":best,"auc_delta_bootstrap":np.quantile(delta,[.025,.975]).tolist(),"runs":measured}
(root/"compare_3d_experiments.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
for name,report in sorted(measured.items(),key=lambda p:-p[1]["auc"]):
    print(name,f"{report['auc']:.8f}",report["wrong_pairs"])
print("Scene-bootstrap interval",summary["auc_delta_bootstrap"])
