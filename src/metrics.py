"""Ranking metrics distinguish tied pairs from incorrectly oriented pairs."""

import numpy as np
from sklearn.metrics import roc_auc_score


def metrics(frame, score):
    score = np.asarray(score, dtype=np.float64)
    if score.shape != (len(frame),) or not np.isfinite(score).all():
        raise ValueError("Invalid scores")
    y = frame.label.to_numpy(dtype=bool)
    negative = np.sort(score[~y])
    positive = score[y]
    inv = len(negative) * len(positive) - 0.5 * (
        np.searchsorted(negative, positive, "left").sum()
        + np.searchsorted(negative, positive, "right").sum()
    )
    pairs = (
        frame.reset_index(drop=True)
        .reset_index()
        .pivot(index=["image", "pair_id"], columns="label", values="index")[[1, 0]]
        .to_numpy()
    )
    gap = score[pairs[:, 0]] - score[pairs[:, 1]]
    result = dict(
        auc=1 - float(inv) / (len(negative) * len(positive)),
        inversions=float(inv),
        wrong_pairs=int((gap < 0).sum()),
        tied_pairs=int((gap == 0).sum()),
        pair_accuracy=float(((gap > 0) + 0.5 * (gap == 0)).mean()),
        accuracy=float(((score > 0) == y).mean()),
    )
    result["categories"] = {}
    pair_categories = frame.category.to_numpy()[pairs[:, 0]]
    for category, rows in frame.groupby("category").indices.items():
        local = gap[pair_categories == category]
        result["categories"][category] = dict(
            auc=float(roc_auc_score(y[rows], score[rows])),
            wrong_pairs=int((local < 0).sum()),
            tied_pairs=int((local == 0).sum()),
        )
    return result


def compare(frame, reference, candidate, repetitions=1000, seed=421):
    pairs = (
        frame.reset_index(drop=True)
        .reset_index()
        .pivot(index=["image", "pair_id"], columns="label", values="index")[[1, 0]]
        .to_numpy()
    )
    left, right = pairs.T
    old = reference[left] > reference[right]
    new = candidate[left] > candidate[right]
    result = dict(repaired=int((~old & new).sum()), introduced=int((old & ~new).sum()))
    rows = list(frame.groupby("image", sort=False).indices.values())
    rng = np.random.default_rng(seed)
    y = frame.label.to_numpy()
    deltas = []
    for _ in range(repetitions):
        idx = np.concatenate([rows[j] for j in rng.integers(len(rows), size=len(rows))])
        deltas.append(roc_auc_score(y[idx], candidate[idx]) - roc_auc_score(y[idx], reference[idx]))
    result["auc_delta_ci95"] = np.quantile(deltas, [0.025, 0.975]).tolist() if deltas else None
    return result
