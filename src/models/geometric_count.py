"""Train-only weak grounding and a shared geometric count density.

Optional existence supervision and inference routes leave the original preset
unchanged. All extra inputs and route decisions come from caption text alone.
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
EXISTENCE, NONEXISTENCE, BINDING = range(4, 7)
_RELATION_IDS = {"to the left of": 1, "to the right of": 2, "in front of": 3, "behind": 4}
_RELATION_PATTERN = r"(to the left of|to the right of|in front of|behind)"
_FILTER = r"(?:(?:small|large|gray|red|blue|green|brown|purple|cyan|yellow|metal|rubber) )*(?:object|cube|sphere|cylinder)"


@dataclass(frozen=True)
class ComparisonCaption:
    left: str
    right: str
    operator: str


@lru_cache(maxsize=65536)
def parse_comparison_caption(caption):
    """Count filters use exactly the stage-2 number-free encoding, including plurals."""
    caption = " ".join(caption.lower().strip().rstrip(".").split())
    match = re.fullmatch(r"(?:there are|the scene contains) (more|fewer) (.+?) than (.+)", caption)
    if match:
        operator, left, right = match.groups()
    else:
        match = re.fullmatch(r"(?:there are|the scene contains) as many (.+?) as (.+)", caption)
        if match is None:
            return None
        operator, (left, right) = "equal", match.groups()
    filters = [parse_count_caption("exactly one " + phrase) for phrase in (left, right)]
    if any(p is None or p.relation for p in filters):
        return None
    # Reject clauses that the permissive exactly-N parser would otherwise accept.
    if any(re.fullmatch(_FILTER, p.filter_text.split(" ", 2)[2]) is None for p in filters):
        return None
    return ComparisonCaption(filters[0].filter_text, filters[1].filter_text, operator)


def comparison_log_probability(left, right, operators):
    """Independent 0..10 distributions: sum joint mass over >, < or == bins.

    operators: 0=more, 1=fewer, 2=equal. Compute in log space under AMP too.
    """
    bins = torch.arange(11, device=left.device)
    masks = torch.stack([bins[:, None] > bins[None], bins[:, None] < bins[None],
                         bins[:, None] == bins[None]])[operators]
    joint = left.float()[:, :, None] + right.float()[:, None, :]
    return joint.masked_fill(~masks, -torch.inf).flatten(1).logsumexp(-1)


def encode_comparison_inputs(captions, vocab, device, source="stage2"):
    parsed = [(i, p) for i, c in enumerate(captions) if (p := parse_comparison_caption(c)) is not None]
    if not parsed:
        return None
    rows, values = zip(*parsed)
    texts = [p.left for p in values] + [p.right for p in values]
    if source == "geometry":
        texts = ["a " + text.split(" ", 2)[2] for text in texts]
    tokens, lengths = vocab.encode_batch(texts)
    references, reference_lengths = vocab.encode_batch(["a object"] * (2 * len(rows)))
    return {"rows": torch.tensor(rows, device=device),
            "operators": torch.tensor([("more", "fewer", "equal").index(p.operator) for p in values], device=device),
            "tokens": tokens.to(device), "lengths": lengths,
            "reference_tokens": references.to(device), "reference_lengths": reference_lengths}


def caption_pair_key(caption):
    """Canonical grammar family for twin grouping, without the category column."""
    caption = " ".join(caption.lower().strip().rstrip(".").split())
    if parse_comparison_caption(caption) is not None:
        return "comparison"
    parsed = parse_geometric_caption(caption, include_existence=True)
    if parsed is None:
        raise ValueError(f"unsupported canonical caption for text-only pairing: {caption!r}")
    if parsed.kind == COUNT:
        return "spatial_count" if parsed.relation else "count"
    if parsed.kind in (EXISTENCE, NONEXISTENCE):
        return "negation" if caption.startswith(("there is ", "no ", "the scene contains ")) else "attribute"
    return str(parsed.kind)


@dataclass(frozen=True)
class GeometricCaption:
    filter_text: str
    reference_text: str
    relation: int
    number: int = 0
    kind: int = COUNT


@lru_cache(maxsize=65536)
def parse_geometric_caption(caption, include_existence=False):
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
    if include_existence:
        body = re.sub(r"^(?:there is |the scene contains )", "", caption)
        body = re.sub(r" (?:is|are) in the scene$", "", body)
        absent = re.fullmatch(r"no (" + _FILTER + r")", body)
        if absent:
            return GeometricCaption("a " + absent[1], "a object", 0, kind=NONEXISTENCE)
        present = re.fullmatch(r"a (" + _FILTER + r")(?: and a (" + _FILTER + r"))?", body)
        if present:
            return GeometricCaption("a " + present[1], "a " + present[2] if present[2] else "a object",
                                    0, kind=BINDING if present[2] else EXISTENCE)
    return None


def encode_geometric_inputs(captions, vocab, device, include_existence=False):
    parsed = []
    for i, caption in enumerate(captions):
        value = parse_geometric_caption(caption, include_existence)
        if value is not None:
            if value.kind == BINDING:
                # Both components keep the SAME caption row and one conjunction label.
                for filter_text in (value.filter_text, value.reference_text):
                    parsed.append((i, GeometricCaption(filter_text, "a object", 0, kind=BINDING)))
            else:
                parsed.append((i, value))
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
                 geometry_sharpness=12., exclusion_radius=.12, existence_weight=0.,
                 route_spatial=False, comparison_route="off", comparison_weight=1.,
                 comparison_source="stage2", **kwargs):
        super().__init__(in_channels, text_dim, **kwargs)
        if not self.freeze_base or self.train_count_only:
            raise ValueError("geometry head requires freeze_base=true and train_count_only=false")
        self.count_distribution.requires_grad_(False)
        self.geometry = GeometricCountDistribution(in_channels, text_dim,
            geometry_dim, behind_sign, geometry_sharpness, exclusion_radius)
        self.geometry_weight, self.baseline_weight = geometry_weight, baseline_weight
        self.grounding_weight = grounding_weight
        if (existence_weight < 0 or comparison_route not in ("off", "equality", "all")
                or comparison_source not in ("stage2", "geometry")):
            raise ValueError("invalid existence weight or comparison route")
        self.existence_weight = existence_weight
        self.route_spatial = route_spatial
        self.comparison_route, self.comparison_weight = comparison_route, comparison_weight
        self.comparison_source = comparison_source
        self.caption_pairing = existence_weight > 0 or route_spatial or comparison_route != "off"
        self._geometry_output = None

    def supports_training_caption(self, caption):
        parsed = parse_geometric_caption(caption, include_existence=self.existence_weight > 0)
        if parsed is None:
            return False
        return (self.existence_weight > 0 if parsed.kind in (EXISTENCE, NONEXISTENCE, BINDING)
                else parsed.kind == COUNT or self.grounding_weight > 0)

    def train(self, mode=True):
        super().train(False)
        self.geometry.train(mode)
        self.training = mode
        return self

    def forward(self, image, text, twin_index, difference_mask, count_text=None, geometry_text=None,
                comparison_text=None):
        # Training needs only the new branch. Avoid running the large frozen pair
        # head for grounding examples; evaluation always uses the full baseline.
        if self.training:
            base = image.new_zeros(text[0].shape[0])
        else:
            with torch.no_grad():
                base = super().forward(image, text, twin_index, difference_mask, count_text)
        self._geometry_output = None
        if geometry_text is None:
            return self._route_comparison(base, image, twin_index, comparison_text)
        rows, numbers, relations, kinds, filters, references = geometry_text
        pairs = twin_index.to(base.device)
        row_to_image = torch.empty_like(base, dtype=torch.long)
        row_to_image[pairs.flatten()] = torch.arange(len(pairs), device=base.device).repeat_interleave(2)
        outputs = self.geometry(image, filters, references, relations, row_to_image[rows])
        self._geometry_output = (rows, numbers, kinds, outputs)
        logp = outputs[0].gather(1, numbers[:, None]).squeeze(1)
        spatial_count = (kinds == COUNT) & (relations != 0)
        if self.route_spatial and not self.training:
            # Existential truth is mass in the relation half-plane, not a centroid.
            logp = torch.where(kinds == SPATIAL, outputs[0][:, 1:].logsumexp(-1), logp)
            logp = torch.where(kinds == ABSENCE, outputs[0][:, 0], logp)
            spatial_count = spatial_count | (kinds == SPATIAL) | (kinds == ABSENCE)
        utility = base.new_zeros(base.shape).scatter(0, rows, logp.to(base.dtype))
        active_rows = torch.zeros_like(base, dtype=torch.bool).scatter(0, rows, spatial_count)
        active = active_rows[pairs].all(1)
        margin = self.geometry_weight * (utility[pairs[:, 0]] - utility[pairs[:, 1]])
        margin = margin + self.baseline_weight * (base[pairs[:, 0]] - base[pairs[:, 1]])
        routed = base.new_zeros(base.shape).scatter(0, pairs.flatten(), torch.stack([margin / 2, -margin / 2], 1).flatten())
        route = torch.zeros_like(base, dtype=torch.bool).scatter(0, pairs.flatten(), active[:, None].expand(-1, 2).flatten())
        return self._route_comparison(torch.where(route, routed, base), image, twin_index, comparison_text)

    def _route_comparison(self, base, image, twin_index, comparison_text):
        if self.training or self.comparison_route == "off" or comparison_text is None:
            return base
        rows, operators, filters = comparison_text
        pairs = twin_index.to(base.device)
        row_to_image = torch.empty_like(base, dtype=torch.long)
        row_to_image[pairs.flatten()] = torch.arange(len(pairs), device=base.device).repeat_interleave(2)
        doubled = torch.cat([row_to_image[rows], row_to_image[rows]])
        # relation=0 makes references irrelevant to the geometric gate, but the
        # learned count readout still needs its original "a object" encoding.
        filter_text, reference_text = filters
        with torch.no_grad():
            relations = torch.zeros(len(doubled), device=base.device, dtype=torch.long)
            if self.comparison_source == "geometry":
                distributions = self.geometry(image, filter_text, reference_text, relations, doubled)[0]
            else:
                distributions = self.count_distribution(image[doubled], filter_text, reference_text, relations)
        left, right = distributions.split(len(rows))
        logp = comparison_log_probability(left, right, operators)
        utility = base.new_zeros(base.shape).scatter(0, rows, logp.to(base.dtype))
        parsed = torch.zeros_like(base, dtype=torch.bool).scatter(0, rows, True)
        active = parsed[pairs].all(1)
        if self.comparison_route == "equality":
            equality = torch.zeros_like(base, dtype=torch.bool).scatter(0, rows, operators == 2)
            active = active & equality[pairs].any(1)
        margin = self.comparison_weight * (utility[pairs[:, 0]] - utility[pairs[:, 1]])
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
        if self.existence_weight > 0:
            present = log_probs[:, 1:].logsumexp(-1)
            for kind in (EXISTENCE, NONEXISTENCE, BINDING):
                selected = kinds == kind
                if not selected.any():
                    continue
                logp = log_probs[selected, 0] if kind == NONEXISTENCE else present[selected]
                log_not = present[selected] if kind == NONEXISTENCE else log_probs[selected, 0]
                targets = target[selected]
                if kind == BINDING:
                    unique = rows[selected].unique()
                    # P(X exists AND Y exists); a false conjunction does not
                    # imply either individual conjunct is false.
                    order = rows[selected].argsort(stable=True)
                    components = logp[order].reshape(-1, 2)
                    absent = log_not[order].reshape(-1, 2)
                    logp = components.sum(1)
                    # P(not X OR (X AND not Y)), stable even when P(both)
                    # rounds to one. Do not subtract that probability from one.
                    log_not = torch.logaddexp(absent[:, 0], components[:, 0] + absent[:, 1])
                    targets = labels[unique].float()
                loss = loss + self.existence_weight * F.binary_cross_entropy_with_logits(logp - log_not, targets)
        return loss
