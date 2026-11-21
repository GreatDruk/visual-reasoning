"""Train-time exact probability programs; no slot models or text encoders."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .programs import EXCLUSIVE_GROUPS, FEWER, MORE, SAME, WORD_INDEX

NEG = -1e4


def log1mexp(x):
    """log(1 - exp(x)) for x <= 0, stable on both ends and NaN-free in backward."""
    x = x.clamp(max=-1e-6)
    near = x > -math.log(2)
    a = torch.log(-torch.expm1(torch.where(near, x, torch.full_like(x, -1.0))))
    b = torch.log1p(-torch.exp(torch.where(near, torch.full_like(x, -1.0), x)))
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
        dist = torch.logaddexp(dist + log_q[:, j : j + 1], shifted + log_p[:, j : j + 1]).clamp_min(
            NEG
        )
    return dist


class ProgramExecutor(nn.Module):
    """Differentiable counts, coupled comparisons and marginal spatial references."""

    def __init__(self):
        super().__init__()
        self.exclusive_attributes = True
        self.comparison_mode = "coupled"
        self.reference_mode = "marginal"
        self.require_reference_exists = True
        self.register_buffer(
            "axes",
            torch.tensor(
                [
                    [0.0, 0.0],
                    [-1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, -1.0],
                    [-1.0, 0.0],
                    [1.0, 0.0],
                ]
            ),
        )
        self.log_sharpness = nn.Parameter(torch.full((7,), math.log(8.0)))
        self.relation_offset = nn.Parameter(torch.zeros(7))
        self.log_reference_temperature = nn.Parameter(torch.zeros(()))

    @staticmethod
    def membership(objectness, word_logits, words, exclusive=False):
        """log P(slot matches all filter words) [R, K]; words [R, MAX_WORDS] (0 = pad)."""
        logp = F.logsigmoid(word_logits)
        if exclusive:
            for group in EXCLUSIVE_GROUPS:
                indices = torch.tensor([WORD_INDEX[w] for w in group], device=word_logits.device)
                logp = logp.index_copy(
                    -1,
                    indices,
                    F.log_softmax(word_logits.index_select(-1, indices), dim=-1),
                )
        picked = logp.gather(2, words[:, None].expand(-1, logp.shape[1], -1))
        return F.logsigmoid(objectness) + (picked * (words > 0)[:, None]).sum(-1)

    def coupled_comparison(self, objectness, word_logits, words_a, words_b, log_a, log_b):
        """Exact sign of count(A)-count(B) under per-slot independent assignments.

        For every slot, the event (matches A, matches B) is JOINT, not two
        independent draws: if A and B are the same filter, count(A)==count(B)
        surely. We infer P(A ∩ B) from the union of their words, respecting the
        mutually exclusive CLEVR attribute groups. The remaining sum of signed
        Bernoulli increments is computed with a 1D dynamic program.
        """
        joined = torch.cat([words_a, words_b], dim=-1)
        identical_before = joined[:, :, None] == joined[:, None, :]
        prior_duplicate = identical_before & torch.tril(
            torch.ones_like(identical_before, dtype=torch.bool), diagonal=-1
        )
        union_words = joined.masked_fill(prior_duplicate.any(-1), 0)
        conflicting = torch.zeros(len(joined), dtype=torch.bool, device=joined.device)
        for group in EXCLUSIVE_GROUPS:
            group_idx = torch.tensor([WORD_INDEX[w] for w in group], device=joined.device)
            conflicting |= torch.isin(union_words, group_idx).sum(-1) > 1
        p_a, p_b = (log_a.exp(), log_b.exp())
        p_both = self.membership(
            objectness, word_logits, union_words, self.exclusive_attributes
        ).exp()
        p_both = p_both.masked_fill(conflicting[:, None], 0.0)
        p_both = torch.minimum(p_both, torch.minimum(p_a, p_b))
        plus = (p_a - p_both).clamp_min(0.0)
        minus = (p_b - p_both).clamp_min(0.0)
        same = (1 - plus - minus).clamp_min(0.0)
        norm = (plus + minus + same).clamp_min(1e-08)
        plus, minus, same = (plus / norm, minus / norm, same / norm)
        batch, k = p_a.shape
        pmf = p_a.new_zeros((batch, 2 * k + 1))
        pmf[:, k] = 1
        for j in range(k):
            inc = F.pad(pmf[:, :-1], (1, 0))
            dec = F.pad(pmf[:, 1:], (0, 1))
            pmf = pmf * same[:, j : j + 1] + inc * plus[:, j : j + 1] + dec * minus[:, j : j + 1]
        more = pmf[:, k + 1 :].sum(-1).clamp_min(1e-12).log()
        fewer = pmf[:, :k].sum(-1).clamp_min(1e-12).log()
        equal = pmf[:, k].clamp_min(1e-12).log()
        return (more, fewer, equal)

    def forward(self, enc, image_index, inputs):
        """-> log P(caption true) [R], plus per-row count distributions for inspection.

        Optional ``marginal`` mode computes a separate count distribution for each
        potential reference object, rather than treating the barycentre of reference
        slots as a real object. The original logic remains the default so previously
        trained checkpoints retain their exact behaviour.
        """
        objectness, word_logits = self.slot_logits(enc)
        objectness = objectness[image_index].float()
        word_logits = word_logits[image_index].float()
        position = enc["position"][image_index].float()
        relations = inputs["relations"]
        log_a = self.membership(objectness, word_logits, inputs["a"], self.exclusive_attributes)
        log_b = self.membership(objectness, word_logits, inputs["b"], self.exclusive_attributes)
        log_ref = self.membership(objectness, word_logits, inputs["ref"], self.exclusive_attributes)
        reference_log_weights = (log_ref * self.log_reference_temperature.exp()).log_softmax(-1)
        reference = reference_log_weights.exp()
        anchor = torch.einsum("rk,rkc->rc", reference, position)
        anchor = torch.where((relations >= 5)[:, None], torch.zeros_like(anchor), anchor)
        offset = torch.einsum("rkc,rc->rk", position - anchor[:, None], self.axes[relations])
        gate = F.logsigmoid(
            self.log_sharpness.exp()[relations, None]
            * (offset - self.relation_offset[relations, None])
        )
        gate = (
            gate
            + torch.log1p(-reference.clamp(max=1 - 0.0001))
            * ((relations >= 1) & (relations <= 4))[:, None]
        )
        gate = torch.where((relations == 0)[:, None], torch.zeros_like(gate), gate)
        dist_a = poisson_binomial(log_a + gate)
        dist_b = poisson_binomial(log_b)
        if self.reference_mode == "marginal":
            spatial = (relations >= 1) & (relations <= 4)
            if spatial.any():
                pos = position[spatial]
                relation = relations[spatial]
                delta = pos[:, None, :, :] - pos[:, :, None, :]
                distance = torch.einsum("sjtc,sc->sjt", delta, self.axes[relation])
                rel_gate = F.logsigmoid(
                    self.log_sharpness.exp()[relation, None, None]
                    * (distance - self.relation_offset[relation, None, None])
                )
                n = pos.shape[1]
                self_mask = torch.eye(n, dtype=torch.bool, device=pos.device)[None]
                rel_gate = rel_gate.masked_fill(self_mask, NEG)
                log_match = log_a[spatial, None, :] + rel_gate
                conditional = poisson_binomial(log_match.reshape(-1, n))
                conditional = conditional.reshape(-1, n, n + 1)
                marginal = torch.logsumexp(conditional + reference_log_weights[spatial, :, None], 1)
                dist_a = dist_a.clone()
                dist_a[spatial] = marginal
        ops, numbers = (inputs["ops"], inputs["numbers"])
        k = dist_a.shape[1] - 1
        exact = dist_a.gather(1, numbers.clamp(max=k)[:, None]).squeeze(1)
        exact = torch.where(numbers > k, torch.full_like(exact, -30.0), exact)
        exist_a, exist_b = (log1mexp(dist_a[:, 0]), log1mexp(dist_b[:, 0]))
        joint = dist_a[:, :, None] + dist_b[:, None, :]
        i = torch.arange(k + 1, device=joint.device)
        more = joint.masked_fill(~(i[:, None] > i[None, :]), 2 * NEG).flatten(1).logsumexp(-1)
        fewer = joint.masked_fill(~(i[:, None] < i[None, :]), 2 * NEG).flatten(1).logsumexp(-1)
        same = joint.masked_fill(~(i[:, None] == i[None, :]), 2 * NEG).flatten(1).logsumexp(-1)
        if self.comparison_mode == "coupled":
            compare_mask = (ops == MORE) | (ops == FEWER) | (ops == SAME)
            if compare_mask.any():
                cm, cf, cs = self.coupled_comparison(
                    objectness[compare_mask],
                    word_logits[compare_mask],
                    inputs["a"][compare_mask],
                    inputs["b"][compare_mask],
                    log_a[compare_mask],
                    log_b[compare_mask],
                )
                more, fewer, same = (more.clone(), fewer.clone(), same.clone())
                more[compare_mask], fewer[compare_mask], same[compare_mask] = (
                    cm,
                    cf,
                    cs,
                )
        table = torch.stack([exact, exist_a, dist_a[:, 0], exist_a + exist_b, more, fewer, same], 1)
        log_true = table.gather(1, ops[:, None]).squeeze(1)
        if self.require_reference_exists:
            log_no_ref = log1mexp(log_ref).sum(-1)
            log_ref_exists = log1mexp(log_no_ref)
            log_true = log_true + torch.where(
                (relations >= 1) & (relations <= 4),
                log_ref_exists,
                torch.zeros_like(log_true),
            )
        return (log_true.clamp(-30.0, -1e-06), dist_a, dist_b)
