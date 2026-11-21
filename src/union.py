"""One-to-one proposal fusion and small caption-trained recovery heads."""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.special import expit, logit
from torch import nn
from torch.nn import functional as F

from .heads import BOX, GROUPS, N_ATTRS, TableHead

BASE_DIM = 57
ROI_DIM = 256
MAX_SLOTS = 40
PHASES = ((0, 0), (2, 0), (0, 2), (2, 2))


def detector_rows(head, table):
    x, xy, valid = table
    enc = head.encode(x.float(), xy.float(), valid)
    obj, words = head.slot_logits(enc)
    attrs = torch.cat([words[..., g.tolist()].log_softmax(-1) for g in GROUPS], -1)
    raw = x[..., :1].float().masked_fill(~valid[..., None], -1e4)
    return torch.cat((raw, obj[..., None], attrs, x[..., BOX], enc["position"]), -1)


def undo_shift(rows, phase):
    """Box xy uses [0,1]; executor positions use [-1,1]."""
    shift = rows.new_tensor(phase) / rows.new_tensor([240, 160])
    rows = rows.clone()
    rows[..., -6:-4] -= shift
    rows[..., -2:] -= 2 * shift
    return rows


def local_features(feature_map, boxes):
    """Sample every proposal in one grid_sample without replicating feature maps."""
    b, k = boxes.shape[:2]
    axis = (torch.arange(4, device=boxes.device, dtype=torch.float32) + 0.5) / 4
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    grid = boxes[..., None, None, :2].float() + boxes[..., None, None, 2:].float() * torch.stack(
        (xx, yy), -1
    )
    grid = (grid * 2 - 1).reshape(b, k * 4, 4, 2)
    sampled = F.grid_sample(feature_map.float(), grid, align_corners=False, padding_mode="border")
    sampled = (
        sampled.reshape(b, feature_map.shape[1], k, 4, 4).transpose(1, 2).reshape(b * k, -1, 4, 4)
    )
    return F.adaptive_avg_pool2d(sampled, 2).reshape(b, k, -1)


def merge(left, right, left_roi, right_roi, settings):
    li = np.flatnonzero(left[:, 0] >= logit(settings["proposal_floor"]))
    ri = np.flatnonzero(right[:, 0] >= logit(settings["proposal_floor"]))
    a, b = left[li], right[ri]
    scale = np.array([240, 160])
    boxes = [v[:, -6:-2] for v in (a, b)]
    center = [(v[:, :2] + 0.5 * v[:, 2:]) * scale for v in boxes]
    distance = np.linalg.norm(center[0][:, None] - center[1], axis=-1)
    extent = [(v[:, 2:] * scale).max(-1) for v in boxes]
    radius = np.clip(
        settings["radius_factor"] * np.minimum(extent[0][:, None], extent[1]),
        settings["radius_min"],
        settings["radius_max"],
    )
    normalized = distance / radius
    costs = np.concatenate(
        (np.where(normalized < 1, normalized, 1000), np.full((len(a), len(a)), 2.0)), -1
    )
    rows, cols = linear_sum_assignment(costs)
    matches = {i: j for i, j in zip(rows, cols) if j < len(b)}
    used = set(matches.values())
    candidates = [(i, matches.get(i)) for i in range(len(a))]
    candidates += [(None, j) for j in range(len(b)) if j not in used]
    x = np.zeros((MAX_SLOTS, BASE_DIM), np.float32)
    xy = np.zeros((MAX_SLOTS, 2), np.float32)
    roi = np.zeros((MAX_SLOTS, ROI_DIM), np.float16)
    n = matched = 0
    for i, j in candidates:
        left_row, right_row = (None if i is None else a[i]), (None if j is None else b[j])
        take_left = right_row is None or (left_row is not None and left_row[1] >= right_row[1])
        ref = left_row if take_left else right_row
        if (left_row is None or right_row is None) and ref[0] < logit(settings["single_floor"]):
            continue
        if n >= MAX_SLOTS:
            raise ValueError("Union table exceeds its explicit slot budget")
        sources = [
            v[: 2 + N_ATTRS] if v is not None else np.zeros(2 + N_ATTRS)
            for v in (left_row, right_row)
        ]
        both = left_row is not None and right_row is not None
        x[n] = np.concatenate(
            (
                ref[1:-2],
                *sources,
                [left_row is not None, right_row is not None, normalized[i, j] if both else 1.0],
            )
        )
        xy[n] = ref[-2:]
        roi[n] = left_roi[li[i]] if take_left else right_roi[ri[j]]
        n += 1
        matched += both
    return (
        x,
        xy,
        np.arange(MAX_SLOTS) < n,
        roi,
        dict(candidates=n, matched=matched, singleton=n - matched),
    )


class UnionHead(TableHead):
    def __init__(self, relational=True, roi=False, teacher=None):
        super().__init__(BASE_DIM, teacher)
        self.relational, self.use_roi = relational, roi
        self.roi_projection = (
            nn.Sequential(nn.LayerNorm(ROI_DIM), nn.Linear(ROI_DIM, 32), nn.SiLU()) if roi else None
        )
        self.encoder = nn.Sequential(
            nn.Linear(BASE_DIM + (32 if roi else 0), 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Dropout(0.08),
        )
        self.query, self.key, self.value = nn.Linear(96, 32), nn.Linear(96, 32), nn.Linear(96, 48)
        self.geometry = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 1))
        self.residual = nn.Sequential(nn.Linear(144, 64), nn.SiLU(), nn.Linear(64, 3 + N_ATTRS))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def encode(self, features, positions, valid):
        base = features[..., :BASE_DIM].float()
        vector = (
            torch.cat((base, self.roi_projection(features[..., BASE_DIM:].float())), -1)
            if self.use_roi
            else base
        )
        feat = self.encoder(vector)
        context = torch.zeros((*feat.shape[:-1], 48), device=feat.device, dtype=feat.dtype)
        if self.relational:
            size = base[..., BOX][..., 2:].clamp_min(0.01)
            center = base[..., BOX][..., :2] + 0.5 * size
            delta = (center[:, :, None] - center[:, None]) / size[:, :, None]
            ratio = size.log()[:, :, None] - size.log()[:, None]
            geometry = self.geometry(torch.cat((delta.clamp(-5, 5), ratio), -1)).squeeze(-1)
            scores = self.query(feat) @ self.key(feat).transpose(1, 2) / 32**0.5 + geometry
            neighbors = valid[:, None] & ~torch.eye(
                valid.shape[1], device=valid.device, dtype=torch.bool
            )
            attention = scores.masked_fill(~neighbors, -1e4).softmax(-1) * neighbors
            context = attention @ self.value(feat)
        correction = self.residual(torch.cat((feat, context), -1))
        return dict(
            pooled=base,
            position=positions + 0.08 * correction[..., -2:].tanh(),
            residual=correction,
            valid=valid,
        )

    def slot_logits(self, enc):
        obj, attrs = super().slot_logits(enc)
        weak = enc["valid"] & (enc["pooled"][..., 0] < 0)
        return obj + weak * 3 * enc["residual"][..., 0].tanh(), attrs


def rescue_tables(left, right):
    """Historical agreement rule, separated from the learned union head."""
    p1, p2, q1, q2 = expit(left[:, 1]), expit(right[:, 1]), expit(left[:, 0]), expit(right[:, 0])
    valid = q1 >= expit(-2.442347)
    p1 = np.where(valid, p1, 0.0)
    p2 = np.where(q2 >= expit(-2.442347), p2, 0.0)

    def attributes(rows):
        return [np.exp(v) for v in np.split(rows[:, 2:17], [2, 10, 12], axis=-1)]

    attrs1, attrs2 = attributes(left), attributes(right)
    boxes1, boxes2 = left[:, -6:-2], right[:, -6:-2]
    extent = np.array([240, 160])
    mid1 = (boxes1[:, :2] + 0.5 * boxes1[:, 2:]) * extent
    mid2 = (boxes2[:, :2] + 0.5 * boxes2[:, 2:]) * extent
    size1, size2 = boxes1[:, 2:] * extent, boxes2[:, 2:] * extent
    d = np.linalg.norm(mid1[:, None] - mid2[None], axis=-1)
    limit = np.clip(0.57 * np.minimum(size1.max(-1)[:, None], size2.max(-1)), 7.0, 13.0)
    color, shape = attrs1[1] @ attrs2[1].T, attrs1[-1] @ attrs2[-1].T
    support = (d < limit) & ((color > 0.47) | (shape > 0.53)) & ((p2 > 0.75) & (q2 > 0.50))[None]
    get = np.argmax(np.where(support, p2[None] * (1 - 0.40 * d / limit), -1.0), axis=1)
    separated = np.linalg.norm(mid1[:, None] - mid1[None], axis=-1)
    strong = ((p1 > 0.65)[None] & (separated < np.clip(0.40 * size1.max(-1)[:, None], 6, 12))).any(
        -1
    )
    take = np.flatnonzero(support.any(-1) & ~strong & (q1 > 0.025) & (p1 < 0.35))
    presence, xy = p1.copy(), left[:, -2:].copy()
    attrs = [a.copy() for a in attrs1]
    used = set()
    for j in take:
        k = get[j]
        if k in used:
            continue
        used.add(k)
        valid[j] = True
        presence[j] = 0.85 * p2[k]
        xy[j] = right[k, -2:]
        for a, b in zip(attrs, attrs2):
            a[j] = 0.40 * a[j] + 0.60 * b[k]
    upper = boxes1[:, :2] + boxes1[:, 2:]
    overlap = (
        (np.minimum(upper[:, None], upper) - np.maximum(boxes1[:, None, :2], boxes1[:, :2]))
        .clip(0)
        .prod(-1)
    )
    covers = overlap / boxes1[:, 2:].prod(-1).clip(1e-6)[:, None]
    hidden = (
        (covers > 0.12) & (covers < 0.86) & (mid1[:, 1, None] < mid1[:, 1]) & (p1[None] > 0.85)
    ).any(-1)
    weak = (
        valid
        & (q1 >= 0.10)
        & (presence < 0.40)
        & ~strong
        & hidden
        & ((d < limit) & (q2[None] >= 0.11)).any(-1)
    )
    presence[weak] = np.maximum(presence[weak], 0.52)
    control = (p1, left[:, -2:], [a.copy() for a in attrs1], q1 >= expit(-2.442347))
    boosted = (presence, xy, attrs, valid)
    return control, boosted
