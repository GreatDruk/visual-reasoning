"""Train-only weak grounding and a shared geometric count density.

The frozen stage-2 scorer is retained for every non-spatial-count caption. All
extra inputs come from text; numbers only index the predicted count distribution.
"""

from dataclasses import dataclass
from functools import lru_cache
import math
import re

import torch
from torch import nn
import torch.nn.functional as F

from .count_distribution import PairCountChoiceHead, parse_count_caption


COUNT, POSITION, SPATIAL, ABSENCE = range(4)
_RELATION_IDS = {"to the left of": 1, "to the right of": 2, "in front of": 3, "behind": 4}
_RELATION_PATTERN = r"(to the left of|to the right of|in front of|behind)"


@dataclass(frozen=True)
class GeometricCaption:
    filter_text: str
    reference_text: str
    relation: int
    number: int = 0
    kind: int = COUNT


@lru_cache(maxsize=65536)
def parse_geometric_caption(caption):
    """Deterministic canonical grammar, independent of metadata and truth labels."""
    caption = " ".join(caption.lower().strip().rstrip(".").split())
    count = parse_count_caption(caption)
    if count is not None:
        return GeometricCaption("a " + count.filter_text.split(" ", 2)[2],
                                count.reference_text, count.relation, count.number)
    position = re.fullmatch(r"(?:there is a |the )(.+?)(?: is)? on the (left|right) side of the image", caption)
    if position:
        return GeometricCaption("a " + position[1], "a object",
                                1 if position[2] == "left" else 2, kind=POSITION)
    spatial = re.fullmatch(r"(?:there is )?(a |the |no )(.+?)(?: is)? " + _RELATION_PATTERN + r" the (.+)", caption)
    if spatial:
        return GeometricCaption("a " + spatial[2], "a " + spatial[4],
                                _RELATION_IDS[spatial[3]], kind=ABSENCE if spatial[1] == "no " else SPATIAL)
    return None


def encode_geometric_inputs(captions, vocab, device):
    parsed = [(i, p) for i, c in enumerate(captions) if (p := parse_geometric_caption(c)) is not None]
    if not parsed:
        return None
    rows, values = zip(*parsed)
    filters, filter_lengths = vocab.encode_batch([p.filter_text for p in values])
    references, reference_lengths = vocab.encode_batch([p.reference_text for p in values])
    return {"rows": torch.tensor(rows, device=device),
            "numbers": torch.tensor([p.number for p in values], device=device),
            "relations": torch.tensor([p.relation for p in values], device=device),
            "kinds": torch.tensor([p.kind for p in values], device=device),
            "filter_tokens": filters.to(device), "filter_lengths": filter_lengths,
            "reference_tokens": references.to(device), "reference_lengths": reference_lengths}


def coordinate_grid(h, w, device, dtype):
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=device, dtype=dtype),
                            torch.linspace(-1, 1, w, device=device, dtype=dtype), indexing="ij")
    return torch.stack([xs.flatten(), ys.flatten()], -1)


class GeometricCountDistribution(nn.Module):
    """Sigmoid membership, reference softmax, fixed half-plane, ordinal count bins.

    Behind means smaller image y. The sign was checked on true TRAIN spatial
    captions and remains a constructor option for train-only direction ablations.
    The numeric readout sees only masked density mass: it cannot bypass the gate
    through a relation embedding or global image features.
    """

    def __init__(self, in_channels, text_dim, dim=64, behind_sign=-1,
                 sharpness=12., exclusion_radius=.12):
        super().__init__()
        if dim < 1 or behind_sign not in (-1, 1) or sharpness <= 0 or exclusion_radius <= 0:
            raise ValueError("invalid geometric count dimensions or geometry")
        self.dim = dim
        self.exclusion_radius = exclusion_radius
        self.cell = nn.Sequential(nn.Conv2d(in_channels, dim, 1), nn.GroupNorm(1, dim), nn.GELU(),
                                  nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.GELU())
        self.query = nn.Linear(text_dim, dim)
        self.bias = nn.Linear(text_dim, 1)
        nn.init.zeros_(self.bias.weight)
        nn.init.constant_(self.bias.bias, -2.)
        self.log_gain = nn.Parameter(torch.tensor(math.log(40.)))
        self.log_scale = nn.Parameter(torch.tensor(-1.))
        self.register_buffer("directions", torch.tensor([[0., 0.], [-1., 0.], [1., 0.],
                                                       [0., -float(behind_sign)], [0., float(behind_sign)]]))
        self.register_buffer("bins", torch.arange(11, dtype=torch.float32))
        self.sharpness = sharpness

    def membership(self, cells, text):
        return torch.einsum("bd,bnd->bn", self.query(text), cells) / math.sqrt(self.dim) + self.bias(text)

    def relation_mask(self, xy, reference, relations):
        offsets = xy[None] - reference[:, None]
        distance = torch.einsum("bnc,bc->bn", offsets, self.directions[relations].to(xy.dtype))
        half_plane = (self.sharpness * distance).sigmoid()
        # A reference can match the filter itself. Suppress its immediate footprint.
        exclusion = 1 - torch.exp(-offsets.square().sum(-1) / (2 * self.exclusion_radius ** 2))
        gate = half_plane * exclusion
        return torch.where((relations == 0)[:, None], torch.ones_like(gate), gate)

    def forward(self, image, filters, references, relations, image_index):
        cells = self.cell(image).flatten(2).transpose(1, 2)[image_index]
        xy = coordinate_grid(*image.shape[-2:], cells.device, cells.dtype)
        scores = self.membership(cells, filters[0])
        reference_scores = self.membership(cells, references[0])
        reference = reference_scores.softmax(-1) @ xy
        subject = scores.softmax(-1) @ xy
        weights = scores.sigmoid() * self.relation_mask(xy, reference, relations)
        # expm1 in the absence readout overflows above ~11 under fp16. Keep
        # density integration and both numeric readouts in float32 under AMP.
        mean = self.log_gain.exp() * weights.float().mean(-1)
        scale = .25 + F.softplus(self.log_scale)
        # Same ordinal Gaussian machinery for ordinary and spatial counting.
        logits = -.5 * ((self.bins[None] - mean[:, None]) / scale).square()
        log_probs = logits.float().log_softmax(-1)
        relative = ((subject - reference) * self.directions[relations]).sum(-1) * self.sharpness
        position = subject[:, 0] * self.directions[relations, 0] * self.sharpness
        absence = -torch.log(torch.expm1(mean.clamp(1e-5, 30.)))
        return log_probs, position, relative, absence


class PairGeometricCountChoiceHead(PairCountChoiceHead):
    requires_geometry_inputs = True
    train_geometry_only = True
    # Unlike stage 2, this head requires a complete stage-2 checkpoint, including
    # its old count branch. Only the new geometry tensors may be absent.
    warm_start_optional_prefixes = ("fusion.geometry.",)

    def __init__(self, in_channels, text_dim, geometry_dim=64, geometry_weight=1.,
                 baseline_weight=1., grounding_weight=.5, behind_sign=-1,
                 geometry_sharpness=12., exclusion_radius=.12, **kwargs):
        super().__init__(in_channels, text_dim, **kwargs)
        if not self.freeze_base or self.train_count_only:
            raise ValueError("geometry head requires freeze_base=true and train_count_only=false")
        self.count_distribution.requires_grad_(False)
        self.geometry = GeometricCountDistribution(in_channels, text_dim,
            geometry_dim, behind_sign, geometry_sharpness, exclusion_radius)
        self.geometry_weight, self.baseline_weight = geometry_weight, baseline_weight
        self.grounding_weight = grounding_weight
        self._geometry_output = None

    def supports_training_caption(self, caption):
        parsed = parse_geometric_caption(caption)
        return parsed is not None and (parsed.kind == COUNT or self.grounding_weight > 0)

    def train(self, mode=True):
        super().train(False)
        self.geometry.train(mode)
        self.training = mode
        return self

    def forward(self, image, text, twin_index, difference_mask, count_text=None, geometry_text=None):
        # Training needs only the new branch. Avoid running the large frozen pair
        # head for grounding examples; evaluation always uses the full baseline.
        if self.training:
            base = image.new_zeros(text[0].shape[0])
        else:
            with torch.no_grad():
                base = super().forward(image, text, twin_index, difference_mask, count_text)
        self._geometry_output = None
        if geometry_text is None:
            return base
        rows, numbers, relations, kinds, filters, references = geometry_text
        pairs = twin_index.to(base.device)
        row_to_image = torch.empty_like(base, dtype=torch.long)
        row_to_image[pairs.flatten()] = torch.arange(len(pairs), device=base.device).repeat_interleave(2)
        outputs = self.geometry(image, filters, references, relations, row_to_image[rows])
        self._geometry_output = (rows, numbers, kinds, outputs)
        logp = outputs[0].gather(1, numbers[:, None]).squeeze(1)
        spatial_count = (kinds == COUNT) & (relations != 0)
        utility = base.new_zeros(base.shape).scatter(0, rows, logp.to(base.dtype))
        active_rows = torch.zeros_like(base, dtype=torch.bool).scatter(0, rows, spatial_count)
        active = active_rows[pairs].all(1)
        margin = self.geometry_weight * (utility[pairs[:, 0]] - utility[pairs[:, 1]])
        margin = margin + self.baseline_weight * (base[pairs[:, 0]] - base[pairs[:, 1]])
        routed = base.new_zeros(base.shape).scatter(0, pairs.flatten(), torch.stack([margin / 2, -margin / 2], 1).flatten())
        route = torch.zeros_like(base, dtype=torch.bool).scatter(0, pairs.flatten(), active[:, None].expand(-1, 2).flatten())
        return torch.where(route, routed, base)

    def auxiliary_loss(self, labels):
        if self._geometry_output is None:
            return labels.new_zeros((), dtype=torch.float32)
        rows, numbers, kinds, (log_probs, position, relative, absence) = self._geometry_output
        target = labels[rows].float()
        count = kinds == COUNT
        loss = log_probs.sum() * 0
        if count.any():
            lp = log_probs[count].gather(1, numbers[count, None]).squeeze(1)
            other = log_probs[count].masked_fill(F.one_hot(numbers[count], 11).bool(), -torch.inf)
            negative = -other.logsumexp(-1)
            loss = loss + self.auxiliary_weight * (target[count] * -lp + (1 - target[count]) * self.false_count_weight * negative).mean()
        for kind, logits in ((POSITION, position), (SPATIAL, relative), (ABSENCE, absence)):
            selected = kinds == kind
            if selected.any():
                loss = loss + self.grounding_weight * F.binary_cross_entropy_with_logits(logits[selected], target[selected])
        return loss
