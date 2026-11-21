"""Caption supervision, incomplete pseudo targets, and pre-selection count learning."""

import numpy as np
import torch
from torch.nn import functional as F

from .heads import GROUPS
from .programs import EXACT, WORD_INDEX


def caption_loss(head, table, batch, simple=False):
    raw = head.score(table, batch["scenes"], batch["programs"])
    target = batch["label"]
    weight = 1 + 0.7 * torch.as_tensor(
        np.isin(batch["category"], ("counting", "comparison", "spatial_counting")),
        device=raw.device,
    )
    if simple:
        use = (batch["programs"]["relations"] == 0) & (batch["programs"]["ops"] <= 3)
        weight = weight * use
    loss = (
        (F.binary_cross_entropy_with_logits(raw, target, reduction="none") * weight).sum()
        / weight.sum().clamp_min(1)
        * weight.mean()
    )
    a, b = batch["pairs"].T
    pair_weight = weight[a]
    loss += (
        F.binary_cross_entropy_with_logits(raw[a] - raw[b], target[a], reduction="none")
        * pair_weight
    ).mean()
    if batch["aux"] is not None:
        loss += 0.3 * head.existence_loss(table, *batch["aux"])
    return loss


def dense_targets(parts, shape, stride, device, level=0, multiscale=False):
    h, w = shape
    centers = np.zeros((len(parts), h, w), np.float32)
    regression = np.zeros((len(parts), 4, h, w), np.float32)
    labels = np.full((len(parts), 4, h, w), -100, np.int64)
    uncertain = np.zeros_like(centers)
    for i, (boxes, p, valid, attrs) in enumerate(parts):
        for box, prob, ok, attr in zip(boxes, p, valid, attrs):
            if not ok:
                continue
            x, y, bw, bh = np.rint(box * np.array([240, 160, 240, 160]))
            if prob > 0.1:
                x0, y0 = max(0, int(x / stride)), max(0, int(y / stride))
                x1, y1 = (
                    min(w, int(np.ceil((x + bw) / stride))),
                    min(h, int(np.ceil((y + bh) / stride))),
                )
                uncertain[i, y0:y1, x0:x1] = np.maximum(uncertain[i, y0:y1, x0:x1], prob)
            if prob <= 0.84:
                continue
            if multiscale:
                assigned = 0 if max(bw, bh) < 32 else 1 if max(bw, bh) < 56 else 2
                if assigned != level:
                    continue
            u, v = (x + 0.5 * bw) / stride, (y + 0.5 * bh) / stride
            xi, yi = int(u), int(v)
            if 0 <= xi < w and 0 <= yi < h and min(bw, bh) > 0:
                centers[i, yi, xi] = 1
                regression[i, :, yi, xi] = (
                    u - xi,
                    v - yi,
                    np.log(bw / 24),
                    np.log(bh / 24),
                )
                labels[i, :, yi, xi] = attr
    return [torch.from_numpy(v).to(device) for v in (centers, regression, labels, uncertain)]


def detection_loss(outputs, parts, soft=False):
    total = outputs[0].new_zeros(())
    for level, raw in enumerate(outputs):
        raw = raw.float()
        center, reg, attr, uncertain = dense_targets(
            parts, raw.shape[-2:], 4 * 2**level, raw.device, level, len(outputs) > 1
        )
        present = center.bool()
        count = center.sum().clamp_min(1)
        p = raw[:, 0].sigmoid()
        around = F.max_pool2d(center[:, None], 3, 1, 1)[:, 0]
        negatives = p.square() * (1 - 0.85 * around)
        if soft:
            negatives = negatives * (1 - 0.9 * uncertain)
        weight = torch.where(present, (1 - p).square(), negatives)
        heat = (
            F.binary_cross_entropy_with_logits(raw[:, 0], center, reduction="none") * weight
        ).sum() / count
        location = (
            raw.sum() * 0
            if not present.any()
            else F.smooth_l1_loss(
                raw[:, 1:5].permute(0, 2, 3, 1)[present],
                reg.permute(0, 2, 3, 1)[present],
            )
        )
        classes = raw.sum() * 0
        for j, scores in enumerate(raw[:, 5:].split([len(g) for g in GROUPS], 1)):
            truth = attr[:, j]
            classes += F.cross_entropy(
                scores, truth, ignore_index=-100, reduction="sum"
            ) / truth.ge(0).sum().clamp_min(1)
        total += heat + 0.55 * location + 0.2 * classes
    return total / len(outputs)


def map_count_loss(raw, batch):
    """Send exact-count supervision to all local peaks before top-k and thresholding."""
    prog = batch["programs"]
    use = (prog["ops"] == EXACT) & (prog["relations"] == 0) & batch["label"].eq(1)
    if not use.any():
        return raw.sum() * 0
    source = raw.float()[batch["scenes"][use]]
    heat = source[:, 0]
    peaks = heat == F.max_pool2d(heat[:, None], 3, 1, 1)[:, 0]
    occupancy = peaks * (heat / 0.65).sigmoid()
    choices = torch.cat([x.softmax(1) for x in source[:, 5:].split([len(g) for g in GROUPS], 1)], 1)
    word_prob = choices.new_ones((len(source), len(WORD_INDEX) + 1, *heat.shape[-2:]))
    indices = torch.as_tensor(np.concatenate(GROUPS), device=raw.device)
    word_prob = word_prob.index_copy(1, indices, choices)
    matched = word_prob[torch.arange(len(source), device=raw.device)[:, None], prog["a"][use]].prod(
        1
    )
    return F.smooth_l1_loss((occupancy * matched).sum((1, 2)), prog["numbers"][use].float())
