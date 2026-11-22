"""Dense counts bypass proposal selection; discrete scene execution stays unchanged."""

import itertools

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .programs import EXACT, EXCLUSIVE_GROUPS, EXIST, FEWER, MORE, NONE, SAME
from .runtime import amp

TYPES = tuple(itertools.product(*EXCLUSIVE_GROUPS))
ACTIVE = ("counting", "comparison")


class DenseCounter(nn.Module):
    def __init__(self, net, joint=False):
        super().__init__()
        self.net = net
        self.mass = nn.Sequential(nn.Conv2d(20, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.mass[-1].weight)
        nn.init.zeros_(self.mass[-1].bias)
        self.joint = nn.Conv2d(64, 96, 1) if joint else None
        if self.joint is not None:
            nn.init.zeros_(self.joint.weight)
            nn.init.zeros_(self.joint.bias)

    def forward(self, images, local_types=None):
        with amp(images.device):
            features = self.net.pyramid(images)[0]
            raw = self.net.head(features)
            correction = self.mass(raw)
            joint = None if self.joint is None else self.joint(features)
        hist, mass = self.integrate(
            raw.float(), correction.float(), None if joint is None else joint.float()
        )
        if local_types is None:
            return hist, mass
        if joint is not None:
            raise ValueError("Localized delta requires factorized density")
        return hist, mass, self.class_density(raw.float(), mass, local_types)

    @staticmethod
    def class_density(raw, mass, types):
        """One selected class per image, without allocating all 96 spatial maps."""
        selected = mass
        remaining = types.long()
        columns = (remaining // 48, remaining // 6 % 8, remaining // 3 % 2, remaining % 3)
        for logits, column in zip(raw[:, 5:].split((2, 8, 2, 3), 1), columns):
            index = column[:, None, None, None].expand(-1, 1, *mass.shape[-2:])
            selected = selected * logits.softmax(1).gather(1, index)[:, 0]
        return selected

    @staticmethod
    def integrate(raw, correction, joint=None):
        heat = raw[:, :1]
        evidence = (1.5 * heat.clamp(-12, 10)).exp()
        neighborhood = F.avg_pool2d(evidence, 5, 1, 2, divisor_override=1)
        mass = (
            heat.sigmoid() * evidence / neighborhood.clamp_min(1e-8) * (2 * correction.tanh()).exp()
        )
        sz, col, mat, shape = [a.softmax(1) for a in raw[:, 5:].split((2, 8, 2, 3), 1)]
        if joint is None:
            a = (sz[:, :, None] * col[:, None]).flatten(1, 2).flatten(2)
            b = (mat[:, :, None] * shape[:, None]).flatten(1, 2).flatten(2)
            hist = ((a * mass.flatten(2)) @ b.transpose(1, 2)).flatten(1)
        else:
            prior = (
                sz[:, :, None, None, None]
                * col[:, None, :, None, None]
                * mat[:, None, None, :, None]
                * shape[:, None, None, None, :]
            ).flatten(1, 4)
            probabilities = (prior.clamp_min(1e-12).log() + joint).softmax(1)
            hist = (mass * probabilities).sum((-2, -1))
        return hist, mass[:, 0]


def freeze_statistics(model):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


class CountQueries:
    """Padded per-scene query tensors avoid searching/parsing captions in train steps."""

    def __init__(self, scenes, device):
        supported = (EXACT, MORE, FEWER, SAME, EXIST, NONE)
        by_scene = []
        for rows, local in zip(scenes.rows, scenes._pairs_local):
            pairs = rows[local]
            by_scene.append(
                [
                    pair
                    for pair in pairs
                    if all(
                        not scenes.programs[j].relation and scenes.programs[j].op in supported
                        for j in pair
                    )
                ]
            )
        count = max(1, max(map(len, by_scene)))
        n = len(scenes.names)
        a = np.zeros((n, count, 2, 96), np.float32)
        b = np.zeros_like(a)
        ops = np.zeros((n, count, 2), np.int64)
        numbers = np.zeros((n, count, 2), np.float32)
        labels = np.zeros_like(numbers)
        valid = np.zeros((n, count), bool)
        ids = np.full((n, count, 2), -1, np.int64)
        for i, pairs in enumerate(by_scene):
            for k, pair in enumerate(pairs):
                valid[i, k] = True
                ids[i, k] = pair
                for t, j in enumerate(pair):
                    p = scenes.programs[j]
                    a[i, k, t] = [set(p.a) <= set(words) for words in TYPES]
                    b[i, k, t] = [set(p.b) <= set(words) for words in TYPES]
                    ops[i, k, t] = p.op
                    numbers[i, k, t] = p.number
                    if scenes.label is not None:
                        labels[i, k, t] = scenes.label[j]
        self.fields = {
            k: torch.as_tensor(v, device=device)
            for k, v in dict(a=a, b=b, ops=ops, numbers=numbers, labels=labels, valid=valid).items()
        }
        self.ids = ids
        self.valid = valid
        self.scenes = scenes
        expected = set(np.flatnonzero(scenes.frame.category.isin(ACTIVE)))
        if not expected <= set(ids[valid].ravel()):
            raise ValueError("Unsupported global count query in active categories")

    def values(self, hist, indices):
        q = {k: v[indices] for k, v in self.fields.items()}
        a = (hist[:, None, None, :] * q["a"]).sum(-1)
        b = (hist[:, None, None, :] * q["b"]).sum(-1)
        diff = a - b
        value = -1.125 * (a - q["numbers"]).square()
        for op, case in [
            (MORE, diff - 0.5),
            (FEWER, -diff - 0.5),
            (SAME, 0.5 - diff.abs()),
            (EXIST, a - 0.5),
            (NONE, 0.5 - a),
        ]:
            value = torch.where(q["ops"] == op, case, value)
        return value * 4, a, q

    def loss(self, hist, indices):
        values, counts, q = self.values(hist, indices)
        valid = q["valid"]
        logits = values[..., 0] - values[..., 1]
        factual = (
            F.binary_cross_entropy_with_logits(logits, q["labels"][..., 0], reduction="none")
            * valid
        ).sum() / valid.sum().clamp_min(1)
        exact = (q["ops"] == EXACT) & q["labels"].eq(1) & valid[..., None]
        absolute = (
            F.smooth_l1_loss(counts, q["numbers"], beta=0.25, reduction="none") * exact
        ).sum() / exact.sum().clamp_min(1)
        return factual, absolute

    def score(self, hist):
        indices = torch.arange(len(hist), device=hist.device)
        values, counts, q = self.values(hist, indices)
        logits = (values[..., 0] - values[..., 1]).clamp(-8, 8).cpu().numpy()
        score = np.zeros(len(self.scenes.frame), np.float64)
        pairs = self.ids[self.valid]
        v = logits[self.valid]
        score[pairs[:, 0]] = v
        score[pairs[:, 1]] = -v
        exact = (q["ops"] == EXACT) & q["labels"].eq(1) & q["valid"][..., None]
        error = (counts - q["numbers"]).abs()[exact].cpu().numpy()
        result = dict(
            exact_queries=len(error),
            mae=float(error.mean()) if len(error) else None,
            within_half=float((error < 0.5).mean()) if len(error) else None,
        )
        return score, result, counts.cpu().numpy()
