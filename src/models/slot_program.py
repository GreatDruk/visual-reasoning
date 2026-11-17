"""Caption programs executed over frozen object slots.

Every canonical caption compiles (text only, no category / pair id / label) into a
count program over at most three attribute filters:

    EXACT n  : count(A [rel R]) == n       counting, spatial_counting
    EXIST    : count(A [rel R]) >= 1       attribute, position, spatial, negation (a)
    NONE     : count(A [rel R]) == 0       negation (no), spatial_negation
    AND      : count(A) >= 1 and count(B) >= 1                binding
    MORE / FEWER / SAME : count(A) >, <, == count(B)          comparison

Each slot is an independent Bernoulli "matches the filter" with probability
objectness * prod_word sigmoid(slot . word) * relation gate, so count(filter) has an
exact Poisson-binomial distribution over 0..K. The number never enters an encoder;
it only indexes that distribution. Relations compare slot centroids with a soft
reference slot (softmax over the reference filter) and exclude that reference.
"""

from dataclasses import dataclass
from functools import lru_cache
import math
import re

import torch
from torch import nn
import torch.nn.functional as F

from .count_distribution import NUMBER_WORDS, PairCountChoiceHead
from .slots import SlotAutoencoder, load_slot_autoencoder
from ..utils import resolve_path


EXACT, EXIST, NONE, AND, MORE, FEWER, SAME = range(7)
OP_NAMES = ("exact", "exist", "none", "and", "more", "fewer", "same")
# 0: no relation; 1-4 relative to the reference slot; 5-6 image halves.
RELATIONS = {"to the left of": 1, "to the right of": 2, "in front of": 3, "behind": 4,
             "on the left side of the image": 5, "on the right side of the image": 6}
ATTRIBUTES = ("small", "large", "gray", "red", "blue", "green", "brown", "purple", "cyan",
              "yellow", "rubber", "metal", "cube", "sphere", "cylinder")
WORD_INDEX = {w: i + 1 for i, w in enumerate(ATTRIBUTES)}  # 0 = padding (no constraint)
MAX_WORDS = 4
# Every real object has exactly one attribute in each of these groups.
EXCLUSIVE_GROUPS = (
    ("small", "large"),
    ("gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"),
    ("rubber", "metal"),
    ("cube", "sphere", "cylinder"),
)
_PLURAL = {w + "s": w for w in ("object", "cube", "sphere", "cylinder")}
_REL = r"(to the left of|to the right of|in front of|behind)"


@dataclass(frozen=True)
class Program:
    op: int
    a: tuple
    b: tuple = ()
    ref: tuple = ()
    relation: int = 0
    number: int = 0


def _filter(phrase):
    """'small red cubes' -> ('small', 'red', 'cube'); None if not an attribute phrase."""
    words = [_PLURAL.get(w, w) for w in phrase.split()]
    if not words or words[-1] not in ("object", "cube", "sphere", "cylinder"):
        return None
    words = [w for w in words if w != "object"]
    if len(words) > MAX_WORDS or any(w not in WORD_INDEX for w in words):
        return None
    return tuple(words)


def _program(op, a, b=None, ref=None, relation=0, number=0):
    a, b, ref = _filter(a), () if b is None else _filter(b), () if ref is None else _filter(ref)
    if a is None or b is None or ref is None:
        return None
    return Program(op, a, b, ref, relation, number)


@lru_cache(maxsize=131072)
def parse_program(caption):
    """Canonical train grammar (32 templates, identical in train/val/test) -> Program."""
    c = " ".join(caption.lower().strip().rstrip(".").split())
    if m := re.fullmatch(r"(?:there (?:is|are) )?exactly (\w+) (.+?) " + _REL + r" the (.+)", c):
        if m[1] in NUMBER_WORDS:
            return _program(EXACT, m[2], ref=m[4], relation=RELATIONS[m[3]], number=NUMBER_WORDS[m[1]])
        return None
    if m := re.fullmatch(r"(?:there (?:is|are) )?exactly (\w+) (.+)", c):
        return _program(EXACT, m[2], number=NUMBER_WORDS[m[1]]) if m[1] in NUMBER_WORDS else None
    if m := re.fullmatch(r"there (?:is|are) (a|no) (.+?) " + _REL + r" the (.+)", c):
        return _program(EXIST if m[1] == "a" else NONE, m[2], ref=m[4], relation=RELATIONS[m[3]])
    if m := re.fullmatch(r"(?:the (.+?) is|there is a (.+?)) on the (left|right) side of the image", c):
        return _program(EXIST, m[1] or m[2], relation=5 if m[3] == "left" else 6)
    if m := re.fullmatch(r"(?:a (.+?)|the (.+?) is) " + _REL + r" the (.+)", c):
        return _program(EXIST, m[1] or m[2], ref=m[4], relation=RELATIONS[m[3]])
    if m := re.fullmatch(r"there (?:is|are) (more|fewer) (.+?) than (.+)", c):
        return _program(MORE if m[1] == "more" else FEWER, m[2], m[3])
    if m := re.fullmatch(r"there (?:is|are) as many (.+?) as (.+)", c):
        return _program(SAME, m[1], m[2])
    if m := re.fullmatch(r"(?:there is )?a (.+?) and a (.+)", c):
        return _program(AND, m[1], m[2])
    if m := re.fullmatch(r"there is (a|no) (.+)", c):
        return _program(EXIST if m[1] == "a" else NONE, m[2])
    if m := re.fullmatch(r"a (.+)", c):
        return _program(EXIST, m[1])
    return None


def _words(phrases):
    return torch.tensor([[WORD_INDEX[w] for w in phrase] + [0] * (MAX_WORDS - len(phrase))
                         for phrase in phrases], dtype=torch.long)


def encode_program_inputs(captions, device):
    parsed = [(i, p) for i, c in enumerate(captions) if (p := parse_program(c)) is not None]
    if not parsed:
        return None
    rows, programs = zip(*parsed)
    return {"rows": torch.tensor(rows, device=device),
            "ops": torch.tensor([p.op for p in programs], device=device),
            "numbers": torch.tensor([p.number for p in programs], device=device),
            "relations": torch.tensor([p.relation for p in programs], device=device),
            "a": _words([p.a for p in programs]).to(device),
            "b": _words([p.b for p in programs]).to(device),
            "ref": _words([p.ref for p in programs]).to(device)}


NEG = -1e4  # finite log(0): logaddexp(-inf, -inf) has a NaN gradient


def log1mexp(x):
    """log(1 - exp(x)) for x <= 0, stable on both ends and NaN-free in backward."""
    x = x.clamp(max=-1e-6)
    near = x > -math.log(2)
    a = torch.log(-torch.expm1(torch.where(near, x, torch.full_like(x, -1.))))
    b = torch.log1p(-torch.exp(torch.where(near, torch.full_like(x, -1.), x)))
    return torch.where(near, a, b)


def poisson_binomial(log_p):
    """Per-slot log success probabilities [R, K] -> log P(count = n), [R, K + 1]."""
    log_q = log1mexp(log_p)
    r, k = log_p.shape
    dist = log_p.new_full((r, k + 1), NEG)
    dist[:, 0] = 0
    pad = log_p.new_full((r, 1), NEG)
    for j in range(k):
        shifted = torch.cat([pad, dist[:, :-1]], 1)
        dist = torch.logaddexp(dist + log_q[:, j:j + 1], shifted + log_p[:, j:j + 1]).clamp_min(NEG)
    return dist


class SlotProgramExecutor(nn.Module):
    """Learned attribute / objectness readout on slots + fixed program semantics."""

    def __init__(self, slot_dim, feature_dim, dim=128, hidden=256, dropout=0.1,
                 relation_sharpness=8.0, use_slot_vectors=True,
                 reference_mode="barycenter", require_reference_exists=False,
                 exclusive_attributes=False, comparison_mode="independent"):
        super().__init__()
        self.dim, self.use_slot_vectors = dim, use_slot_vectors
        if reference_mode not in ("barycenter", "marginal"):
            raise ValueError("reference_mode must be 'barycenter' or 'marginal'")
        if comparison_mode not in ("independent", "coupled"):
            raise ValueError("comparison_mode must be 'independent' or 'coupled'")
        self.comparison_mode = comparison_mode
        self.reference_mode = reference_mode
        self.require_reference_exists = require_reference_exists
        self.exclusive_attributes = exclusive_attributes
        in_dim = feature_dim + 1 + (slot_dim if use_slot_vectors else 0)
        self.slot_norm = nn.LayerNorm(slot_dim) if use_slot_vectors else None
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.embed = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(hidden, dim))
        self.words = nn.Embedding(len(ATTRIBUTES) + 1, dim)
        self.word_bias = nn.Embedding(len(ATTRIBUTES) + 1, 1)
        nn.init.zeros_(self.word_bias.weight)
        self.objectness = nn.Linear(dim, 1)
        # Signed offsets: x for left/right, y (grows towards the camera) for front/behind.
        self.register_buffer("axes", torch.tensor([[0., 0.], [-1., 0.], [1., 0.], [0., 1.], [0., -1.],
                                                   [-1., 0.], [1., 0.]]))
        self.log_sharpness = nn.Parameter(torch.full((7,), math.log(relation_sharpness)))
        self.relation_offset = nn.Parameter(torch.zeros(7))
        self.log_reference_temperature = nn.Parameter(torch.zeros(()))

    def slot_logits(self, enc):
        parts = [self.feature_norm(enc["pooled"]), enc["area"][..., None] * 10]
        if self.use_slot_vectors:
            parts.insert(0, self.slot_norm(enc["slots"]))
        h = self.embed(torch.cat(parts, -1))
        words = torch.einsum("bkd,wd->bkw", h, self.words.weight) / math.sqrt(self.dim)
        words = words + self.word_bias.weight[:, 0]
        return self.objectness(h).squeeze(-1), words  # [B, K], [B, K, W]

    @staticmethod
    def membership(objectness, word_logits, words, exclusive=False):
        """log P(slot matches all filter words) [R, K]; words [R, MAX_WORDS] (0 = pad)."""
        logp = F.logsigmoid(word_logits)                                    # [R, K, W]
        if exclusive:
            for group in EXCLUSIVE_GROUPS:
                indices = torch.tensor([WORD_INDEX[w] for w in group], device=word_logits.device)
                logp = logp.index_copy(-1, indices,
                                       F.log_softmax(word_logits.index_select(-1, indices), dim=-1))
        picked = logp.gather(2, words[:, None].expand(-1, logp.shape[1], -1))  # [R, K, M]
        return F.logsigmoid(objectness) + (picked * (words > 0)[:, None]).sum(-1)

    def coupled_comparison(self, objectness, word_logits, words_a, words_b, log_a, log_b):
        """Exact sign of count(A)-count(B) under per-slot independent assignments.

        For every slot, the event (matches A, matches B) is JOINT, not two
        independent draws: if A and B are the same filter, count(A)==count(B)
        surely. We infer P(A ∩ B) from the union of their words, respecting the
        mutually exclusive CLEVR attribute groups. The remaining sum of signed
        Bernoulli increments is computed with a 1D dynamic program.
        """
        joined = torch.cat([words_a, words_b], dim=-1)  # [R, up to 8]
        identical_before = joined[:, :, None] == joined[:, None, :]
        prior_duplicate = identical_before & torch.tril(
            torch.ones_like(identical_before, dtype=torch.bool), diagonal=-1)
        union_words = joined.masked_fill(prior_duplicate.any(-1), 0)
        conflicting = torch.zeros(len(joined), dtype=torch.bool, device=joined.device)
        for group in EXCLUSIVE_GROUPS:
            group_idx = torch.tensor([WORD_INDEX[w] for w in group], device=joined.device)
            # Distinct attributes from the same family cannot belong to one object.
            conflicting |= torch.isin(union_words, group_idx).sum(-1) > 1
        p_a, p_b = log_a.exp(), log_b.exp()
        p_both = self.membership(objectness, word_logits, union_words,
                                 self.exclusive_attributes).exp()
        p_both = p_both.masked_fill(conflicting[:, None], 0.)
        p_both = torch.minimum(p_both, torch.minimum(p_a, p_b))
        plus = (p_a - p_both).clamp_min(0.)
        minus = (p_b - p_both).clamp_min(0.)
        same = (1 - plus - minus).clamp_min(0.)
        # p_both is a conservative approximation if within-group logits are not
        # calibrated, so renormalize any minor consistency violations.
        norm = (plus + minus + same).clamp_min(1e-8)
        plus, minus, same = plus / norm, minus / norm, same / norm
        batch, k = p_a.shape
        pmf = p_a.new_zeros((batch, 2 * k + 1))
        pmf[:, k] = 1
        for j in range(k):
            inc = F.pad(pmf[:, :-1], (1, 0))
            dec = F.pad(pmf[:, 1:], (0, 1))
            pmf = pmf * same[:, j:j+1] + inc * plus[:, j:j+1] + dec * minus[:, j:j+1]
        more = pmf[:, k+1:].sum(-1).clamp_min(1e-12).log()
        fewer = pmf[:, :k].sum(-1).clamp_min(1e-12).log()
        equal = pmf[:, k].clamp_min(1e-12).log()
        return more, fewer, equal

    def forward(self, enc, image_index, inputs):
        """-> log P(caption true) [R], plus per-row count distributions for inspection.

        Optional ``marginal`` mode computes a separate count distribution for each
        potential reference object, rather than treating the barycentre of reference
        slots as a real object. The original logic remains the default so previously
        trained checkpoints retain their exact behaviour.
        """
        objectness, word_logits = self.slot_logits(enc)
        # Count distributions and log-complements need FP32 even with AMP on T4.
        objectness = objectness[image_index].float()
        word_logits = word_logits[image_index].float()
        position = enc["position"][image_index].float()
        relations = inputs["relations"]
        log_a = self.membership(objectness, word_logits, inputs["a"], self.exclusive_attributes)
        log_b = self.membership(objectness, word_logits, inputs["b"], self.exclusive_attributes)
        log_ref = self.membership(objectness, word_logits, inputs["ref"], self.exclusive_attributes)

        # Original reference-centroid semantics; still used for image-side questions.
        reference_log_weights = (log_ref * self.log_reference_temperature.exp()).log_softmax(-1)
        reference = reference_log_weights.exp()
        anchor = torch.einsum("rk,rkc->rc", reference, position)
        anchor = torch.where((relations >= 5)[:, None], torch.zeros_like(anchor), anchor)
        offset = torch.einsum("rkc,rc->rk", position - anchor[:, None], self.axes[relations])
        gate = F.logsigmoid(self.log_sharpness.exp()[relations, None] *
                            (offset - self.relation_offset[relations, None]))
        # Centroid mode downweights a slot's probability of being its own reference.
        gate = gate + torch.log1p(-reference.clamp(max=1 - 1e-4)) * (
            (relations >= 1) & (relations <= 4))[:, None]
        gate = torch.where((relations == 0)[:, None], torch.zeros_like(gate), gate)
        dist_a = poisson_binomial(log_a + gate)
        dist_b = poisson_binomial(log_b)

        if self.reference_mode == "marginal":
            spatial = (relations >= 1) & (relations <= 4)
            if spatial.any():
                # [S, K_ref, K_target, 2]: target - reference position.
                pos = position[spatial]
                relation = relations[spatial]
                delta = pos[:, None, :, :] - pos[:, :, None, :]
                distance = torch.einsum("sjtc,sc->sjt", delta, self.axes[relation])
                rel_gate = F.logsigmoid(self.log_sharpness.exp()[relation, None, None] *
                                        (distance - self.relation_offset[relation, None, None]))
                n = pos.shape[1]
                self_mask = torch.eye(n, dtype=torch.bool, device=pos.device)[None]
                rel_gate = rel_gate.masked_fill(self_mask, NEG)
                log_match = log_a[spatial, None, :] + rel_gate
                conditional = poisson_binomial(log_match.reshape(-1, n))
                conditional = conditional.reshape(-1, n, n + 1)
                marginal = torch.logsumexp(conditional + reference_log_weights[spatial, :, None], 1)
                dist_a = dist_a.clone()
                dist_a[spatial] = marginal

        ops, numbers = inputs["ops"], inputs["numbers"]
        k = dist_a.shape[1] - 1
        exact = dist_a.gather(1, numbers.clamp(max=k)[:, None]).squeeze(1)
        exact = torch.where(numbers > k, torch.full_like(exact, -30.), exact)
        exist_a, exist_b = log1mexp(dist_a[:, 0]), log1mexp(dist_b[:, 0])
        joint = dist_a[:, :, None] + dist_b[:, None, :]
        i = torch.arange(k + 1, device=joint.device)
        more = joint.masked_fill(~(i[:, None] > i[None, :]), 2 * NEG).flatten(1).logsumexp(-1)
        fewer = joint.masked_fill(~(i[:, None] < i[None, :]), 2 * NEG).flatten(1).logsumexp(-1)
        same = joint.masked_fill(~(i[:, None] == i[None, :]), 2 * NEG).flatten(1).logsumexp(-1)
        if self.comparison_mode == "coupled":
            compare_mask = (ops == MORE) | (ops == FEWER) | (ops == SAME)
            if compare_mask.any():
                cm, cf, cs = self.coupled_comparison(
                    objectness[compare_mask], word_logits[compare_mask],
                    inputs["a"][compare_mask], inputs["b"][compare_mask],
                    log_a[compare_mask], log_b[compare_mask])
                more, fewer, same = more.clone(), fewer.clone(), same.clone()
                more[compare_mask], fewer[compare_mask], same[compare_mask] = cm, cf, cs
        table = torch.stack([exact, exist_a, dist_a[:, 0], exist_a + exist_b, more, fewer, same], 1)
        log_true = table.gather(1, ops[:, None]).squeeze(1)
        if self.require_reference_exists:
            # If the reference itself is absent, the spatial assertion cannot be true.
            log_no_ref = log1mexp(log_ref).sum(-1)
            log_ref_exists = log1mexp(log_no_ref)
            log_true = log_true + torch.where((relations >= 1) & (relations <= 4),
                                              log_ref_exists, torch.zeros_like(log_true))
        return log_true.clamp(-30., -1e-6), dist_a, dist_b


class PairSlotCountChoiceHead(PairCountChoiceHead):
    """Frozen stage-2 pair/count model + a train-only slot-program branch.

    During training only the executor runs (the frozen base is skipped); the trainer's
    main loss sees the antisymmetric slot margin and `auxiliary_loss` adds the exact
    row likelihood. At evaluation, pairs whose ops are all in `route_ops` use
    slot_weight * slot margin + base_weight * frozen stage-2 margin; others keep stage 2.
    """

    requires_program_inputs = True
    train_geometry_only = True  # train.py: restrict TRAIN rows to supported captions
    warm_start_optional_prefixes = ("fusion.slot_encoder.", "fusion.slot_executor.")

    def __init__(self, in_channels, text_dim, slot_checkpoint=None, slot_config=None,
                 executor_dim=128, executor_hidden=256, executor_dropout=.1,
                 relation_sharpness=8., use_slot_vectors=True, slot_weight=1., base_weight=1.,
                 program_weight=1., route_ops=("exact", "more", "fewer", "same"),
                 finetune_slots=False, slot_recon_weight=0.0,
                 reference_mode="barycenter", require_reference_exists=False,
                 exclusive_attributes=False, comparison_mode="independent", **kwargs):
        kwargs.setdefault("train_count_only", False)
        super().__init__(in_channels, text_dim, **kwargs)
        if not self.freeze_base or self.train_count_only:
            raise ValueError("slot head requires freeze_base=true and train_count_only=false")
        self.count_distribution.requires_grad_(False)
        if slot_checkpoint:
            self.slot_encoder = load_slot_autoencoder(resolve_path(slot_checkpoint))
        else:
            self.slot_encoder = SlotAutoencoder(in_channels, **(slot_config or {}))
        if slot_recon_weight < 0:
            raise ValueError("slot_recon_weight must be nonnegative")
        self.finetune_slots = finetune_slots
        self.slot_recon_weight = slot_recon_weight
        self._recon_loss = None
        self.slot_encoder.requires_grad_(finetune_slots)
        sa = self.slot_encoder.config
        self.slot_executor = SlotProgramExecutor(sa["slot_dim"], sa["in_channels"], executor_dim,
                                                 executor_hidden, executor_dropout, relation_sharpness,
                                                 use_slot_vectors, reference_mode,
                                                 require_reference_exists, exclusive_attributes,
                                                 comparison_mode)
        self.slot_weight, self.base_weight, self.program_weight = slot_weight, base_weight, program_weight
        unknown = set(route_ops) - set(OP_NAMES)
        if unknown:
            raise ValueError(f"unknown route_ops {sorted(unknown)}; choose from {OP_NAMES}")
        self.register_buffer("route_mask", torch.tensor([name in route_ops for name in OP_NAMES]), persistent=False)
        self._program_output = None

    def supports_training_caption(self, caption):
        return parse_program(caption) is not None

    def train(self, mode=True):
        super().train(False)
        self.slot_executor.train(mode)
        self.slot_encoder.train(False)
        self.training = mode
        return self

    def slot_logits(self, image, rows, pairs, size, inputs):
        row_to_image = torch.empty(size, dtype=torch.long, device=image.device)
        row_to_image[pairs.flatten()] = torch.arange(len(pairs), device=image.device).repeat_interleave(2)
        with torch.set_grad_enabled(self.finetune_slots and self.training):
            if self.finetune_slots and self.training and self.slot_recon_weight > 0:
                enc = self.slot_encoder(image)
                self._recon_loss = enc["loss"]
            else:
                enc = self.slot_encoder.encode(image)
        return self.slot_executor(enc, row_to_image[rows], inputs)

    def forward(self, image, text, twin_index, difference_mask, count_text=None, program_inputs=None):
        if self.training:
            base = image.new_zeros(twin_index.numel())
        else:
            with torch.no_grad():
                base = super().forward(image, text, twin_index, difference_mask, count_text)
        self._program_output = None
        self._recon_loss = None
        if program_inputs is None:
            return base
        rows, ops = program_inputs["rows"], program_inputs["ops"]
        pairs = twin_index.to(base.device)
        log_true, _, _ = self.slot_logits(image, rows, pairs, base.shape[0], program_inputs)
        self._program_output = (rows, log_true)
        logit = (log_true - log1mexp(log_true)).to(base.dtype)
        utility = base.new_zeros(base.shape).scatter(0, rows, logit)
        routed_op = self.route_mask[ops] if not self.training else torch.ones_like(ops, dtype=torch.bool)
        active = torch.zeros_like(base, dtype=torch.bool).scatter(0, rows, routed_op)[pairs].all(1)
        margin = self.slot_weight * (utility[pairs[:, 0]] - utility[pairs[:, 1]])
        margin = margin + self.base_weight * (base[pairs[:, 0]] - base[pairs[:, 1]])
        routed = base.new_zeros(base.shape).scatter(0, pairs.flatten(), torch.stack([margin / 2, -margin / 2], 1).flatten())
        route = torch.zeros_like(base, dtype=torch.bool).scatter(0, pairs.flatten(), active[:, None].expand(-1, 2).flatten())
        return torch.where(route, routed, base)

    def auxiliary_loss(self, labels):
        if self._program_output is None:
            return labels.new_zeros((), dtype=torch.float32)
        rows, log_true = self._program_output
        target = labels[rows].float()
        nll = -(target * log_true + (1 - target) * log1mexp(log_true))
        loss = self.program_weight * nll.mean()
        if self._recon_loss is not None:
            loss = loss + self.slot_recon_weight * self._recon_loss
        return loss
