"""Finding twin captions without pair_id and normalising scores inside a twin pair.

Twins describe the same image, share the caption template and differ in one detail
(1-3 tokens). Within each (image, category) group captions are matched greedily by
token-level edit distance. Rules allow using unlabelled test captions both to find
twins and for the twin normalisation; it is off by default and measured on val.
"""

from collections import defaultdict

import numpy as np


def _edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def find_twins(image_names_per_row, captions, categories, max_distance=3):
    """Return [P, 2] row pairs. Rows without a confident partner stay unmatched."""
    groups = defaultdict(list)
    for r, key in enumerate(zip(image_names_per_row, categories)):
        groups[key].append(r)
    pairs = []
    tokens = [c.split() for c in captions]
    for rows in groups.values():
        if len(rows) == 2:  # the data always has one twin pair per (image, category)
            pairs.append(tuple(rows))
            continue
        cand = sorted(
            (_edit_distance(tokens[a], tokens[b]), a, b)
            for i, a in enumerate(rows)
            for b in rows[i + 1 :]
        )
        used = set()
        for d, a, b in cand:
            if d > max_distance or a in used or b in used:
                continue
            used.update((a, b))
            pairs.append((a, b))
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def twin_normalize(logits, pairs):
    """p_a = sigmoid(s_a - s_b), p_b = 1 - p_a for matched pairs; sigmoid(s) otherwise."""
    logits = np.asarray(logits, dtype=np.float64)
    probs = 1 / (1 + np.exp(-logits))
    if len(pairs):
        a, b = pairs[:, 0], pairs[:, 1]
        pa = 1 / (1 + np.exp(-(logits[a] - logits[b])))
        probs[a], probs[b] = pa, 1 - pa
    return probs
