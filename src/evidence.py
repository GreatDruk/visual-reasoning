"""Train-derived hidden-object uncertainty and conservative paired evidence."""

import numpy as np
from scipy.special import expit, logit

from .counter import ACTIVE, TYPES
from .programs import EXACT, FEWER, MORE, SAME


def placement_prior(donors, background):
    rng = np.random.default_rng(607)
    with np.load(donors) as data:
        i, j = np.nonzero(data["keep"] & (data["labels"] >= 0).all(-1))
        take = rng.choice(len(i), min(4096, len(i)), replace=False)
        i, j = i[take], j[take]
        sz, col, joint = data["labels"][i, j].astype(int).T
        boxes, masks = data["box"][i, j], data["mask"][i, j]
        frequency = float(data["keep"].sum(1).mean())
    ys, xs = [], []
    for (x, y, w, h), mask in zip(boxes, masks):
        region = np.argwhere(mask > 127)
        if not len(region):
            region = np.array([[5, 5]])
        q = region[rng.integers(len(region), size=24)] + rng.random((24, 2))
        ys.append((y + h * q[:, 0] / 10).astype(int).clip(0, 159))
        xs.append((x + w * q[:, 1] / 10).astype(int).clip(0, 239))
    return (
        np.stack(ys),
        np.stack(xs),
        boxes[:, 1] + boxes[:, 3],
        (sz * 8 + col) * 6 + joint,
        frequency,
        background,
    )


def hidden_classes(image, prior):
    import cv2

    ys, xs, foot, kind, frequency, background = prior
    hue, saturation, value = np.moveaxis(cv2.cvtColor(image, cv2.COLOR_RGB2HSV), -1, 0)
    chroma = np.ptp(image.astype(np.int16), axis=-1)
    colored = (saturation > 55) & (chroma >= 17) & (value > 24)
    limits = ((0, 8), (8, 21), (21, 39), (39, 80), (80, 104), (104, 129), (129, 175))
    planes = [colored & (hue >= lo) & (hue < hi) for lo, hi in limits]
    planes[0] |= colored & (hue >= 175)
    delta = image.astype(np.float32).mean(-1) - background.mean(-1)
    delta -= np.median(np.concatenate((delta[:, :8], delta[:, -8:]), axis=1), axis=1)[:, None]
    gray = (np.abs(delta) >= 12) & (chroma <= 22)
    gray &= cv2.dilate(colored.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0
    depth = np.zeros(image.shape[:2], np.int32)
    for plane in [gray, *planes]:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(plane.astype(np.uint8))
        bottom = stats[:, 1] + stats[:, 3]
        bottom[stats[:, 4] < 45] = 0
        bottom[0] = 0
        np.maximum(depth, bottom[labels], out=depth)
    support = (depth[ys, xs] > foot[:, None] + 5).mean(1) >= 0.85
    hist = np.bincount(kind[support], minlength=96)
    return hist * (1 - np.exp(-frequency * support.mean())) / max(1, support.sum())


def transitions(a, b, hidden):
    if set(a.a) != set(b.a) or set(a.b) != set(b.b):
        return np.eye(2)
    if a.op == EXACT and b.op == EXACT:
        starts = [[(a.number, 1.0)], [(b.number, 1.0)]]
        increments = np.array([set(a.a) <= set(t) for t in TYPES], int)

        def ends(x):
            return [x == a.number, x == b.number]

    elif a.op in (MORE, FEWER, SAME) and b.op in (MORE, FEWER, SAME) and a.op != b.op:

        def diffs(op):
            if op == SAME:
                return [(0, 1.0)]
            sign = 1 if op == MORE else -1
            return [(sign, 0.5), (2 * sign, 0.5)]

        starts = [diffs(a.op), diffs(b.op)]
        increments = np.array([int(set(a.a) <= set(t)) - int(set(a.b) <= set(t)) for t in TYPES])

        def ends(x):
            return [
                x > 0 if op == MORE else x < 0 if op == FEWER else x == 0 for op in (a.op, b.op)
            ]

    else:
        return np.eye(2)
    out = np.eye(2) * (1 - hidden.sum())
    for i, options in enumerate(starts):
        for value, weight in options:
            for step in np.unique(increments):
                out[i] += weight * hidden[increments == step].sum() * np.array(ends(value + step))
    return out


def hidden_adjust(scenes, base, histograms):
    score = base.copy()
    for i, j in scenes.pairs:
        if scenes.frame.category.iat[i] not in ACTIVE:
            continue
        transition = transitions(
            scenes.programs[i], scenes.programs[j], histograms[scenes.scene[i]]
        )
        q = expit(base[i])
        mass = np.array([q, 1 - q]) @ transition
        score[i] = logit(np.clip(mass[0] / mass.sum(), 1e-8, 1 - 1e-8))
        score[j] = -score[i]
    return score


def guarded_swap(scenes, base, first, second):
    """Historical .85 agreement/1.0 uncertainty gate; preserve all score values."""
    a, b = scenes.pairs.T
    eligible = (
        scenes.frame.category.eq("counting").to_numpy()
        & (np.abs(base) <= 1.0)
        & (first * second > 0)
        & (np.minimum(np.abs(first), np.abs(second)) >= logit(0.85))
    )
    proposed = 0.75 * base + 0.25 * (first + second) / 2
    flip = eligible[a] & eligible[b] & ((base[a] - base[b]) * (proposed[a] - proposed[b]) < 0)
    result = base.copy()
    left, right = a[flip], b[flip]
    result[left], result[right] = base[right], base[left]
    return result, int(flip.sum())


def compose(scenes, parts, settings):
    c, u = parts["convnext"], parts["union421"]
    if settings["base"] == "compact":
        base = 0.5 * c + 0.5 * u
    else:
        base = 0.75 * (0.5 * c + 0.5 * parts["sigft"] + 0.25 * parts["rescue"])
        base += 0.25 * (u + parts["union853"]) / 2
    active = scenes.frame.category.isin(ACTIVE).to_numpy()
    weight = settings["counter_weight"]
    if weight:
        count = (parts["count54_421"] + parts["count54_853"]) / 2
        counted = np.where(active, (1 - weight) * base + weight * count, base)
    else:
        counted = base.copy()
    tta = counted + settings["tta_weight"] * (parts["r50_tta"] - parts["r50_plain"])
    hidden = (
        tta + settings["hidden_weight"] * (hidden_adjust(scenes, tta, parts["hidden"]) - tta)
        if settings["hidden_weight"]
        else tta.copy()
    )
    final, swapped = (
        guarded_swap(
            scenes,
            hidden,
            parts[settings.get("swap_source", "count53") + "_421"],
            parts[settings.get("swap_source", "count53") + "_853"],
        )
        if settings["swap"]
        else (hidden, 0)
    )
    return dict(base=base, counted=counted, tta=tta, hidden=hidden, final=final), swapped
