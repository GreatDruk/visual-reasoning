import numpy as np
from sklearn.metrics import roc_auc_score


def pair_accuracy(scores, labels, pair_ids):
    """Share of twin pairs where the true caption scores higher than the false one (ties = 0.5)."""
    order = np.lexsort((labels, pair_ids))
    s, l, p = scores[order], labels[order], pair_ids[order]
    ok = (p[0::2] == p[1::2]) & (l[0::2] == 0) & (l[1::2] == 1)
    if not ok.all():
        raise ValueError("pair_accuracy expects exactly one true and one false caption per pair_id")
    diff = s[1::2] - s[0::2]
    return float(np.mean((diff > 0) + 0.5 * (diff == 0)))


def evaluate_scores(scores, labels, categories, pair_ids=None):
    """AUC, accuracy at 0.5 (on probabilities), pair accuracy and per-category AUC."""
    scores, labels, categories = np.asarray(scores), np.asarray(labels), np.asarray(categories)
    out = {
        "auc": float(roc_auc_score(labels, scores)),
        "accuracy": float(np.mean((scores > 0.5) == labels)),
    }
    if pair_ids is not None:
        out["pair_accuracy"] = pair_accuracy(scores, labels, np.asarray(pair_ids))
    for cat in sorted(set(categories)):
        m = categories == cat
        out[f"auc/{cat}"] = float(roc_auc_score(labels[m], scores[m]))
    return out
