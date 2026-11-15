"""Caption-only count parsing and a text-conditioned distribution over 0..10.

The number is a supervised target / distribution index, never an encoder input.
Singular/plural nouns and is/are are normalized too: otherwise they leak N=1.
No category, row identifiers, pair identifiers or labels are used by the parser.
"""

from dataclasses import dataclass
from functools import lru_cache
import math
import re

import torch
from torch import nn
import torch.nn.functional as F

from .pair_choice import PairChoiceHead


NUMBER_WORDS = {word: n for n, word in enumerate(
    "zero one two three four five six seven eight nine ten".split())}
NUMBER_WORDS["no"] = 0
COUNT_TOKEN = "<unk>"  # Existing reserved vocabulary entry; warm-start ids stay intact.
_SINGULAR = {w + "s": w for w in ("object", "cube", "sphere", "cylinder")}
_RELATIONS = {"to the left of": 1, "to the right of": 2, "in front of": 3, "behind": 4}
_RELATION = re.compile(r"\b(to the left of|to the right of|in front of|behind) the (.+)$")
_EXACT = re.compile(r"^(?:there (?:is|are) |the scene contains )?exactly (\w+) (.+)$")


@dataclass(frozen=True)
class CountCaption:
    number: int
    filter_text: str
    reference_text: str
    relation: int


@lru_cache(maxsize=65536)
def parse_count_caption(caption):
    """Parse canonical exactly-N captions; unsupported / non-count captions return None.

    Equivalent existential word orders are accepted for evaluation diagnostics.
    Negations without 'exactly' deliberately keep the original pair head route.
    """
    caption = " ".join(caption.lower().strip().rstrip(".").split())
    # Relation-first is a meaning-preserving surface form, not new supervision.
    first = re.fullmatch(r"(to the left of|to the right of|in front of|behind) the (.+?) there (?:is|are) (exactly .+)", caption)
    if first:
        caption = f"{first[3]} {first[1]} the {first[2]}"
    match = _EXACT.fullmatch(caption)
    if match is None or match[1] not in NUMBER_WORDS:
        return None
    body = re.sub(r" (?:is|are) in the scene$", "", match[2])
    relation, reference = 0, "a object"
    spatial = _RELATION.search(body)
    if spatial:
        relation, reference = _RELATIONS[spatial[1]], "a " + spatial[2]
        body = re.sub(r" (?:is|are)$", "", body[:spatial.start()].strip())
    words = [_SINGULAR.get(word, word) for word in body.split()]
    if not words or words[-1] not in ("object", "cube", "sphere", "cylinder"):
        return None
    return CountCaption(NUMBER_WORDS[match[1]], f"exactly {COUNT_TOKEN} {' '.join(words)}", reference, relation)


def encode_count_inputs(captions, vocab, device):
    """Build extra model inputs solely from caption text and the train vocabulary."""
    parsed = [(i, value) for i, c in enumerate(captions)
              if (value := parse_count_caption(c)) is not None]
    if not parsed:
        return None
    rows, values = zip(*parsed)
    filters, filter_lengths = vocab.encode_batch([value.filter_text for value in values])
    references, reference_lengths = vocab.encode_batch([value.reference_text for value in values])
    return {"rows": torch.tensor(rows, device=device),
            "numbers": torch.tensor([value.number for value in values], device=device),
            "relations": torch.tensor([value.relation for value in values], device=device),
            "filter_tokens": filters.to(device), "filter_lengths": filter_lengths,
            "reference_tokens": references.to(device), "reference_lengths": reference_lengths}


class CountDistribution(nn.Module):
    """Sigmoid set selection + reference geometry + categorical/ordinal readout.

    Multiple selectors preserve mass (softmax would discard object multiplicity).
    Their sums and selected feature moments feed an 11-way readout, with a learned
    ordinal Gaussian term. It is not a squared-error truth score against N.
    """

    def __init__(self, in_channels, text_dim, dim=64, selectors=4, hidden=128,
                 dropout=0.1, ordinal_weight=1.0):
        super().__init__()
        if min(dim, selectors, hidden) < 1:
            raise ValueError("count dimensions must be positive")
        self.dim, self.selectors = dim, selectors
        self.ordinal_weight = ordinal_weight
        self.cell = nn.Sequential(nn.Conv2d(in_channels, dim, 1), nn.GroupNorm(1, dim), nn.GELU(),
                                  nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.GELU())
        self.filter_query = nn.Linear(text_dim, selectors * dim)
        self.reference_query = nn.Linear(text_dim, dim)
        self.selection_bias = nn.Linear(text_dim, selectors)
        self.directions = nn.Embedding(5, 2)
        with torch.no_grad():
            self.directions.weight.copy_(torch.tensor([[0., 0.], [-1., 0.], [1., 0.], [0., 1.], [0., -1.]]))
        self.relation_sharpness = nn.Parameter(torch.tensor(2.0))
        self.relation_embedding = nn.Embedding(5, 8)
        # Set means, masses, maxima and coordinate moments, plus the query/global view.
        self.readout = nn.Sequential(nn.Linear(selectors * (dim + 6) + 2 * dim + 8, hidden),
                                     nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 13))
        self.register_buffer("bins", torch.arange(11, dtype=torch.float32))

    def forward(self, image, filter_text, reference_text, relations):
        cells = self.cell(image).flatten(2).transpose(1, 2)
        h, w = image.shape[-2:]
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=image.device, dtype=cells.dtype),
                                torch.linspace(-1, 1, w, device=image.device, dtype=cells.dtype), indexing="ij")
        xy = torch.stack([xs.flatten(), ys.flatten()], -1)
        query = self.filter_query(filter_text[0]).reshape(-1, self.selectors, self.dim)
        scores = torch.einsum("bkd,bnd->bkn", query, cells) / math.sqrt(self.dim)
        scores = scores + self.selection_bias(filter_text[0])[..., None]
        reference_query = self.reference_query(reference_text[0])
        reference_weights = (torch.einsum("bd,bnd->bn", reference_query, cells) / math.sqrt(self.dim)).softmax(-1)
        location = reference_weights @ xy
        offsets = xy[None] - location[:, None]
        signed_distance = torch.einsum("bnc,bc->bn", offsets, self.directions(relations))
        relation_gate = (signed_distance * (F.softplus(self.relation_sharpness) + 1)).sigmoid()
        relation_gate = torch.where((relations == 0)[:, None], torch.ones_like(relation_gate), relation_gate)
        weights = scores.sigmoid() * relation_gate[:, None]
        mass = weights.sum(-1).clamp_min(1e-5)
        selected = torch.einsum("bkn,bnd->bkd", weights, cells) / mass[..., None]
        loc = torch.einsum("bkn,nc->bkc", weights, xy) / mass[..., None]
        spread = torch.einsum("bkn,nc->bkc", weights, xy.square()) / mass[..., None] - loc.square()
        # Area-normalized mass is stable if the feature-map resolution changes.
        summary = torch.cat([selected, (mass / cells.shape[1])[..., None],
                             weights.amax(-1)[..., None], loc, spread], -1).flatten(1)
        out = self.readout(torch.cat([summary, cells.mean(1), query.mean(1), self.relation_embedding(relations)], -1))
        mean, scale = 10 * out[:, 11:12].sigmoid(), .5 + F.softplus(out[:, 12:13])
        logits = out[:, :11] - self.ordinal_weight * .5 * ((self.bins[None] - mean) / scale).square()
        return logits.log_softmax(-1)


class PairCountChoiceHead(PairChoiceHead):
    """The original pair head with an exactly-N-only distribution utility.

    A frozen warm-started base preserves all other caption routes. Labels enter only
    auxiliary_loss, called by the trainer on TRAIN batches after the forward pass.
    """

    requires_count_inputs = True

    def __init__(self, in_channels, text_dim, dim=256, queries=8, hidden=256,
                 dropout=.1, coord_channels=True, input_pool=None, difference_aware=True,
                 requires_spatial=True, count_dim=64, count_selectors=4, count_hidden=128,
                 count_dropout=.1, ordinal_weight=1., count_weight=2., base_count_weight=.25,
                 auxiliary_weight=.5, false_count_weight=.25, freeze_base=True,
                 train_count_only=True):
        super().__init__(in_channels, text_dim, dim, queries, hidden, dropout,
                         coord_channels, input_pool, difference_aware, requires_spatial)
        self.freeze_base = freeze_base
        self.train_count_only = train_count_only
        if train_count_only and not freeze_base:
            raise ValueError("train_count_only requires freeze_base=true")
        if freeze_base:
            self.requires_grad_(False)
        self.count_distribution = CountDistribution(in_channels, text_dim, count_dim, count_selectors,
                                                    count_hidden, count_dropout, ordinal_weight)
        self.count_weight, self.base_count_weight = count_weight, base_count_weight
        self.auxiliary_weight, self.false_count_weight = auxiliary_weight, false_count_weight
        self._count_output = None

    def train(self, mode=True):
        super().train(False if self.freeze_base else mode)
        self.count_distribution.train(mode)
        return self

    def forward(self, image, text, twin_index, difference_mask, count_text=None):
        base = super().forward(image, text, twin_index, difference_mask)
        self._count_output = None
        if count_text is None:
            return base
        rows, numbers, relations, filters, references = count_text
        pairs = twin_index.to(base.device)
        row_to_image = torch.empty(base.shape[0], dtype=torch.long, device=base.device)
        row_to_image[pairs.flatten()] = torch.arange(len(pairs), device=base.device).repeat_interleave(2)
        log_probs = self.count_distribution(image[row_to_image[rows]], filters, references, relations)
        self._count_output = (rows, numbers, log_probs)
        utility = base.new_zeros(base.shape[0]).scatter(0, rows, log_probs.gather(1, numbers[:, None]).squeeze(1).to(base.dtype))
        active = torch.zeros_like(base, dtype=torch.bool).scatter(0, rows, True)[pairs].all(1)
        margin = self.count_weight * (utility[pairs[:, 0]] - utility[pairs[:, 1]])
        margin = margin + self.base_count_weight * (base[pairs[:, 0]] - base[pairs[:, 1]])
        routed = margin.new_zeros(base.shape).scatter(0, pairs.flatten(), torch.stack([margin * .5, -margin * .5], 1).flatten())
        route = torch.zeros_like(base, dtype=torch.bool).scatter(0, pairs.flatten(), active[:, None].expand(-1, 2).flatten())
        return torch.where(route, routed, base)

    def auxiliary_loss(self, labels):
        if self._count_output is None:
            return labels.new_zeros((), dtype=torch.float32)
        rows, numbers, log_probs = self._count_output
        target = labels[rows].float()
        logp = log_probs.gather(1, numbers[:, None]).squeeze(1)
        # Sum the other bins in log space: log(1-p_N), without saturation at p_N=1.
        other = log_probs.masked_fill(F.one_hot(numbers, 11).bool(), -torch.inf)
        negative = -other.logsumexp(-1)
        return self.auxiliary_weight * (target * -logp + (1 - target) * self.false_count_weight * negative).mean()
