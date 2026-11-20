"""Cross-validate a category temperature for the retained count_hidden baseline."""
from itertools import product
from pathlib import Path
import json
import zipfile

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path("artefacts/kaggle_suite")
RUN = ROOT / "rear_run"
COUNT = ROOT / "count_run"
GRID = tuple(product((1., 1.5, 2., 3.), (1., 1.125, 1.25, 1.5)))


def adjusted(frame, score, other, spatial):
    global_counts = frame.category.isin(["counting", "comparison"])
    spatial_count = frame.category.eq("spatial_counting")
    return score * np.select([global_counts, spatial_count], [1., spatial], default=other)


def main():
    with zipfile.ZipFile(RUN / "rear_results.zip") as archive:
        frame = pd.read_csv(archive.open("verify_result/val_rows.csv"))
    reference = pd.read_csv("data/val.csv")
    assert frame[["id", "image", "label", "category"]].equals(reference[["id", "image", "label", "category"]])
    baseline = frame.base.to_numpy()
    label = frame.label.to_numpy()
    scene, names = pd.factorize(frame.image)
    candidates = {k: adjusted(frame, baseline, *k) for k in GRID}

    def best(mask):
        return max(GRID, key=lambda k: roc_auc_score(label[mask], candidates[k][mask]))

    report = {"baseline": float(roc_auc_score(label, baseline)), "folds": {}}
    oofs = []
    for seed in (374, 619, 2401):
        score = np.empty_like(baseline)
        chosen = []
        for training, evaluation in KFold(5, shuffle=True, random_state=seed).split(names):
            fit, out = np.isin(scene, training), np.isin(scene, evaluation)
            key = best(fit)
            chosen.append(key)
            score[out] = candidates[key][out]
        oofs.append(score)
        report["folds"][str(seed)] = {"auc": float(roc_auc_score(label, score)), "chosen": chosen}
    rng = np.random.default_rng(627)
    deltas = []
    for _ in range(1000):
        w = rng.multinomial(len(names), np.full(len(names), 1 / len(names)))[scene]
        deltas.append(roc_auc_score(label, oofs[0], sample_weight=w) -
                      roc_auc_score(label, baseline, sample_weight=w))
    report["interval"] = np.quantile(deltas, [.025, .975]).tolist()
    final = best(np.ones(len(frame), dtype=bool))
    report["final"] = {"weights": final, "auc": float(roc_auc_score(label, candidates[final]))}
    report["final"]["inversions"] = round((1 - report["final"]["auc"]) * sum(label) * sum(1-label))
    (RUN / "ranking_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))
    if not all(item["auc"] > report["baseline"] for item in report["folds"].values()):
        return

    frame = pd.read_csv("data/test.csv")
    original = pd.read_csv(COUNT / "count_hidden.csv", float_precision="round_trip")
    sample = pd.read_csv("data/sample_submission.csv")
    assert frame.id.is_unique and original.id.is_unique and original.id.equals(sample.id)
    assert set(original.id) == set(frame.id) and original.label.between(0, 1).all()
    frame = frame.set_index("id").loc[original.id].reset_index()
    weights = adjusted(frame, np.ones(len(frame)), *final)
    probabilities = original.label.to_numpy().copy()
    changed = (weights != 1.) & (probabilities > 0.) & (probabilities < 1.)
    probabilities[changed] = expit(weights[changed] * logit(probabilities[changed]))
    result = original.assign(label=probabilities)
    dest = RUN / "count_rank.csv"
    result.to_csv(dest, index=False, float_format="%.17g")
    loaded = pd.read_csv(dest, float_precision="round_trip")
    assert loaded.equals(result) and loaded.label.between(0, 1).all()
    assert np.array_equal(loaded.label.ge(.5), original.label.ge(.5))
    assert np.array_equal(loaded.label[weights == 1.], original.label[weights == 1.])
    print(dest, len(result), "rows", changed.sum(), "calibrated")


if __name__ == "__main__":
    main()
