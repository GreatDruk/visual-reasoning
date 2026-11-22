"""Train-only silhouette teacher and historical initialization, restored for cold starts."""

import math
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .heads import FEATURES as TABLE_FEATURES
from .heads import TableHead
from .programs import AND, ATTRIBUTES, EXACT, EXCLUSIVE_GROUPS, EXIST, FEWER, MORE, NONE, WORD_INDEX
from .reasoning import ProgramExecutor, log1mexp
from .regions import BOUNDARY_FEATURES, COLORS, FEATURES, PRIMITIVE_FEATURES


class PixelReadout(ProgramExecutor):
    def __init__(
        self,
        slot_dim,
        feature_dim,
        dim=128,
        hidden=256,
        dropout=0.1,
        relation_sharpness=8.0,
        **unused,
    ):
        super().__init__()
        self.dim = dim
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.embed = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.words = nn.Embedding(len(ATTRIBUTES) + 1, dim)
        self.word_bias = nn.Embedding(len(ATTRIBUTES) + 1, 1)
        nn.init.zeros_(self.word_bias.weight)
        self.objectness = nn.Linear(dim, 1)
        self.log_sharpness = nn.Parameter(torch.full((7,), math.log(relation_sharpness)))

    def slot_logits(self, enc):
        h = self.embed(
            torch.cat((self.feature_norm(enc["pooled"]), enc["area"][..., None] * 10), -1)
        )
        words = torch.einsum("bkd,wd->bkw", h, self.words.weight) / math.sqrt(self.dim)
        return self.objectness(h).squeeze(-1), words + self.word_bias.weight[:, 0]


class HistoricalHead(TableHead):
    def __init__(self, teacher, features=TABLE_FEATURES):
        # These historical readouts were unused by Fusion, but their allocation
        # consumed RNG before the live residual. Preserve that initialization.
        PixelReadout(1, TABLE_FEATURES, dim=48, hidden=64, dropout=0.05)
        super().__init__(TABLE_FEATURES, teacher)
        if features != TABLE_FEATURES:
            self.residual[0] = nn.Linear(features, 48)


class Silhouette(PixelReadout):
    """Predict objectness and mutually exclusive attributes for each visible region."""

    def __init__(self, boundary=False, primitives=False):
        width = PRIMITIVE_FEATURES if primitives else BOUNDARY_FEATURES if boundary else FEATURES
        super().__init__(
            1,
            width + (4 if boundary else 0),
            dim=80,
            hidden=160,
            dropout=0.08,
            relation_sharpness=16.0,
            use_slot_vectors=False,
            reference_mode="marginal",
            exclusive_attributes=True,
            require_reference_exists=True,
            comparison_mode="coupled",
        )
        self.boundary = boundary
        self.primitives = primitives
        self.color_strength = nn.Parameter(torch.tensor(2.0))
        self.project_position = nn.Linear(width, 2)
        nn.init.zeros_(self.project_position.weight)
        nn.init.zeros_(self.project_position.bias)
        self.register_buffer("color_index", torch.tensor([WORD_INDEX[c] for c in COLORS]))
        if primitives:
            self.shape_strength = nn.Parameter(torch.tensor(4.0))
            self.register_buffer(
                "shape_index", torch.tensor([WORD_INDEX[w] for w in EXCLUSIVE_GROUPS[-1]])
            )

    def slot_logits(self, enc):
        if self.boundary:
            x = enc["pooled"]
            box = x[..., len(COLORS) : len(COLORS) + 4]
            lower, upper = box[..., :2], box[..., :2] + box[..., 2:]
            intersect = (
                (
                    torch.minimum(upper[:, :, None], upper[:, None, :])
                    - torch.maximum(lower[:, :, None], lower[:, None, :])
                )
                .clamp_min(0)
                .prod(-1)
            )
            area = box[..., 2:].prod(-1)
            union = area[:, :, None] + area[:, None, :] - intersect
            other = enc["valid"][:, None] & ~torch.eye(
                x.shape[1], device=x.device, dtype=torch.bool
            )
            overlap = (intersect / union.clamp_min(1e-6)) * other
            same_color = (x[..., : len(COLORS)] @ x[..., : len(COLORS)].transpose(1, 2)) > 0.5
            context = torch.stack(
                [
                    overlap.amax(-1),
                    (overlap * same_color).amax(-1),
                    (intersect / area[:, :, None].clamp_min(1e-6) * other).amax(-1),
                    (overlap > 0.12).sum(-1).float() / 4,
                ],
                -1,
            )
            enc = {**enc, "pooled": torch.cat((x, context), -1)}
        obj, words = super().slot_logits(enc)
        chroma = enc["pooled"][..., : len(COLORS)]
        prior = torch.zeros_like(words).index_copy(
            -1, self.color_index, chroma * self.color_strength
        )
        if self.primitives:
            shape = enc["pooled"][..., PRIMITIVE_FEATURES - 3 : PRIMITIVE_FEATURES]
            prior = prior.index_copy(-1, self.shape_index, shape * self.shape_strength)
        return obj.masked_fill(~enc["valid"], -30.0), words + prior

    def encode(self, features, positions, valid):
        area = features[..., len(COLORS) + 2] * features[..., len(COLORS) + 3]
        correction = 0.14 * self.project_position(features).tanh()
        return {
            "pooled": features,
            "position": positions + correction,
            "valid": valid,
            "area": area,
        }

    def score(self, features, positions, valid, image_index, program):
        enc = self.encode(features, positions, valid)
        truth, _, _ = self(enc, image_index, program)
        return truth - log1mexp(truth)

    def score_and_count_loss(self, features, positions, valid, image_index, program, target):
        enc = self.encode(features, positions, valid)
        truth, counts, _ = self(enc, image_index, program)
        support = torch.arange(counts.shape[-1], device=counts.device)
        expected = (counts.exp() * support).sum(-1)
        exact = (program["ops"] == EXACT) & (target > 0)
        deviation = F.smooth_l1_loss(expected, program["numbers"].float(), reduction="none")
        count_loss = deviation.masked_fill(~exact, 0).sum() / exact.sum().clamp_min(1)
        return truth - log1mexp(truth), count_loss

    def existence_loss(self, features, positions, valid, image_index, program):
        area = features[..., len(COLORS) + 2] * features[..., len(COLORS) + 3]
        obj, words = self.slot_logits({"pooled": features, "area": area, "valid": valid})
        per_region = self.membership(
            obj[image_index], words[image_index], program["a"], exclusive=True
        )
        return -log1mexp(log1mexp(per_region).sum(-1)).mean()

    def anchor_loss(self, features, positions, valid, anchors):
        obj, words = self.slot_logits(self.encode(features, positions, valid))
        active = anchors.ge(0).any(-1)
        obj_loss = F.softplus(-obj).masked_fill(~active, 0).sum() / active.sum().clamp_min(1)
        total = obj_loss
        for j, group in enumerate(EXCLUSIVE_GROUPS):
            indices = [WORD_INDEX[w] for w in group]
            scores = words[..., indices].reshape(-1, len(indices))
            targets = anchors[..., j].flatten()
            total = total + F.cross_entropy(
                scores, targets, ignore_index=-100, reduction="sum"
            ) / targets.ge(0).sum().clamp_min(1)
        return total / (1 + len(EXCLUSIVE_GROUPS))


def implied_filters(program, truth):
    """Project true training statements onto valid simpler existential supervision."""
    if program.relation:
        return (
            [(program.a, 1)]
            if truth and (program.op == EXIST or (program.op == EXACT and program.number > 0))
            else []
        )
    if program.op == EXIST:
        return [(program.a, truth)]
    if program.op == NONE:
        return [(program.a, 1 - truth)]
    if not truth:
        return []
    if program.op == EXACT:
        return [(program.a, int(program.number > 0))]
    if program.op == AND:
        return [(program.a, 1), (program.b, 1)]
    if program.op in (MORE, FEWER):
        return [(program.a if program.op == MORE else program.b, 1)]
    return []


def region_anchors(features, positions, valid, programs, images, targets):
    """Derive crop labels from train captions with a unique plausible color match."""
    families = {
        word: (j, k) for j, group in enumerate(EXCLUSIVE_GROUPS) for k, word in enumerate(group)
    }
    colors = features[..., : len(COLORS)].argmax(-1)
    width = features[..., len(COLORS) + 2]
    height = features[..., len(COLORS) + 3]
    plausible = valid & (width * height >= 52 / (240 * 160))
    votes = defaultdict(set)
    for program, scene, truth in zip(programs, images, targets):
        if not truth:
            continue
        facts = [(words, 1, 0) for words, y in implied_filters(program, truth) if words and y]
        if program.op == EXACT and program.number > 0 and not program.relation:
            facts = [(program.a, program.number, 0)]
        elif program.relation in (5, 6) and program.op == EXIST:
            facts = [(program.a, 1, program.relation)]
        elif 1 <= program.relation <= 4:
            facts.append((program.ref, 1, 0))
        for words, count, side in facts:
            hue = next((COLORS.index(w) for w in words if w in COLORS and w != "gray"), None)
            if hue is None:
                continue
            candidates = plausible[scene] & (colors[scene] == hue)
            if side:
                candidates &= positions[scene, :, 0] * (-1 if side == 5 else 1) > 0.03
            regions = np.flatnonzero(candidates)
            if len(regions) == count:
                for region in regions:
                    for word in words:
                        family, value = families[word]
                        votes[scene, region, family].add(value)
    result = np.full((*valid.shape, len(EXCLUSIVE_GROUPS)), -100, dtype=np.int64)
    for key, values in votes.items():
        if len(values) == 1:
            result[key] = next(iter(values))
    return torch.from_numpy(result)
