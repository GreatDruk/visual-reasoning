"""Score captions jointly by sampling coherent object tables from the image."""

from functools import lru_cache

import numpy as np
from scipy.special import logit
from scipy.stats import qmc

from .programs import AND, EXACT, EXIST, FEWER, MORE, NONE, WORD_INDEX


def _word_mask(words):
    return sum(1 << WORD_INDEX[word] for word in set(words))


def _execute(programs, codes, present, xy, reference_quality, offsets, uniforms):
    results = []
    references = {}
    drawn = np.arange(len(codes))

    def matches(words):
        mask = _word_mask(words)
        return present & ((codes & mask) == mask)

    for p in programs:
        matched = matches(p.a)
        has_ref = True
        if 1 <= p.relation <= 4:
            if p.ref not in references:
                choices = matches(p.ref)
                # One sampled reference per filter is shared across all captions.
                rank = np.log(reference_quality.clip(1e-8)) - np.log(-np.log(uniforms))
                ref_idx = np.argmax(np.where(choices, rank, -np.inf), axis=1)
                references[p.ref] = (ref_idx, choices.any(1))
            ref_idx, has_ref = references[p.ref]
            dimension = int(p.relation >= 3)
            difference = xy[:, :, dimension] - xy[drawn, ref_idx, dimension, None]
            sign = -1 if p.relation in (1, 4) else 1
            matched = matched & (sign * difference > offsets[p.relation])
            matched[drawn, ref_idx] = False
        elif p.relation in (5, 6):
            sign = -1 if p.relation == 5 else 1
            matched = matched & (sign * xy[:, :, 0] > offsets[p.relation])
        a = matched.sum(-1)
        if p.op == EXACT:
            result = a == p.number
        elif p.op == EXIST:
            result = a > 0
        elif p.op == NONE:
            result = a == 0
        elif p.op == AND:
            result = (a > 0) & matches(p.b).any(-1)
        else:
            b = matches(p.b).sum(-1)
            result = a > b if p.op == MORE else a < b if p.op == FEWER else a == b
        results.append(result & has_ref)
    return np.stack(results, axis=1)


@lru_cache(maxsize=24)
def _draws(regions, dimension, seed, draws):
    u = qmc.Sobol(regions * dimension, scramble=True, seed=seed).random_base2(int(np.log2(draws)))
    return u.reshape(-1, regions, dimension).clip(1e-7, 1 - 1e-7)


def sample_scene_scores(programs, pairs, objectness, attributes, xy, offsets, seed, draws=1024):
    """Log-odds for each caption given image uncertainty and consistent twin choices."""
    if not len(objectness):
        return np.zeros(len(programs), dtype=np.float64)
    u = _draws(len(objectness), 4 + len(attributes), seed, draws)
    present = u[..., 0] < objectness
    codes = np.zeros_like(present, dtype=np.int32)
    for j, (indices, probs) in enumerate(attributes, start=1):
        picked = (u[..., j, None] > probs.cumsum(-1)).sum(-1).clip(max=len(indices) - 1)
        codes |= 1 << indices[picked]
    # Shared small coordinate noise preserves ordering consistency between captions.
    noise = logit(u[..., -3:-1]) * 0.026
    logical = _execute(programs, codes, present, xy[None] + noise, objectness, offsets, u[..., -1])
    a, b = pairs.T
    linked = logical[:, a] ^ logical[:, b]
    count = linked.sum(1)
    weights = np.exp(1.3 * (count - len(pairs)))
    weights = weights[:, None] * linked
    mass = weights.sum(axis=0)
    p_left = ((logical[:, a] * weights).sum(axis=0) + 0.75) / (mass + 1.5)
    logits = np.empty(len(programs), dtype=np.float64)
    logits[a] = logit(p_left)
    logits[b] = -logits[a]
    return logits
