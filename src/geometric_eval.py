"""Pair-bootstrap diagnostics for saved validation predictions; never reads test.

Example: python -m src.geometric_eval artefacts/hydra/debug_geometry_v1_s42/<run>
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .data import canonicalized, load_split, pair_groups, subset
from .models.count_distribution import parse_count_caption


def bootstrap_pairs(scores, labels, pair_ids, repeats=2000, seed=0):
    """Resample complete twins, retaining within-pair dependence and class balance."""
    scores, labels, pair_ids = map(np.asarray, (scores, labels, pair_ids))
    if repeats < 1 or not len(scores):
        raise ValueError("positive repeats and at least one complete pair are required")
    unique, counts = np.unique(pair_ids, return_counts=True)
    if not np.all(counts == 2):
        raise ValueError("expected complete pairs")
    order = np.lexsort((labels, pair_ids)).reshape(-1, 2)
    if not np.all(labels[order] == [0, 1]):
        raise ValueError("expected exactly one false and one true caption per pair")
    paired = scores[order]
    correct = (paired[:, 1] > paired[:, 0]) + .5 * (paired[:, 1] == paired[:, 0])
    rng = np.random.default_rng(seed)
    auc, acc = [], []
    for _ in range(repeats):
        chosen = rng.integers(len(unique), size=len(unique))
        auc.append(roc_auc_score(np.tile([0, 1], len(unique)), paired[chosen].flatten()))
        acc.append(correct[chosen].mean())
    return {"pairs": len(unique), "auc": float(roc_auc_score(labels, scores)),
            "auc_ci95": np.quantile(auc, [.025, .975]).tolist(),
            "pair_accuracy": float(correct.mean()),
            "pair_accuracy_ci95": np.quantile(acc, [.025, .975]).tolist(),
            "bootstrap_repeats": repeats, "bootstrap_seed": seed}


def summarize(run_dir, split, repeats=2000):
    run_dir = Path(run_dir)
    result = json.loads((run_dir / "results.json").read_text())
    ids = np.load(run_dir / "val_ids.npy")
    logits = np.load(run_dir / "val_logits.npy")
    if len(np.unique(ids)) != len(ids) or len(ids) != len(split) or set(ids) != set(split.ids):
        raise ValueError("saved predictions must contain exactly the validation ids")
    positions = {row_id: i for i, row_id in enumerate(ids)}
    logits = logits[[positions[i] for i in split.ids]]
    # Match Trainer.evaluate, including its float32 probability convention.
    with np.errstate(over="ignore"):
        scores = 1 / (1 + np.exp(-logits))
    mask = split.categories == "spatial_counting"
    spatial = subset(split, mask)
    statistics = bootstrap_pairs(scores[mask], spatial.labels, spatial.pair_ids, repeats)
    groups = pair_groups(spatial)
    parsed = [parse_count_caption(c) for c in spatial.captions]
    numeric = np.array([parsed[a].filter_text == parsed[b].filter_text
                        and parsed[a].reference_text == parsed[b].reference_text
                        and parsed[a].relation == parsed[b].relation for a, b in groups])
    statistics["twin_subsets"] = {}
    for name, selected in (("number", numeric), ("other_word", ~numeric)):
        rows = groups[selected].flatten()
        if len(rows):
            statistics["twin_subsets"][name] = bootstrap_pairs(scores[mask][rows], spatial.labels[rows],
                                                               spatial.pair_ids[rows], repeats)
    return {"run": str(run_dir), "seed": result["seed"], "best_epoch": result["best_epoch"],
            "total_min": result["total_min"], "val": result["val"], "spatial_counting": statistics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--val-csv", default="data/val.csv")
    parser.add_argument("--repeats", default=2000, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    split = canonicalized(load_split("val", args.val_csv))
    result = [summarize(run, split, args.repeats) for run in args.runs]
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
