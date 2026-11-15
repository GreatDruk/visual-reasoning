"""Twin-conditioned spatial grounding, using caption differences without labels."""

from functools import lru_cache
import math

import torch
from torch import nn
import torch.nn.functional as F


@lru_cache(maxsize=65536)
def _aligned_changes(a, b):
    """Minimum-edit alignment. Canonical argument order makes tie breaks symmetric."""
    if a > b:
        right, left = _aligned_changes(b, a)
        return left, right
    distance = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        distance[i][0] = i
    for j in range(len(b) + 1):
        distance[0][j] = j
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            distance[i][j] = min(distance[i - 1][j] + 1, distance[i][j - 1] + 1,
                                 distance[i - 1][j - 1] + (x != y))
    changed_a, changed_b = [False] * len(a), [False] * len(b)
    i, j = len(a), len(b)
    while i or j:
        if i and j and distance[i][j] == distance[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            changed_a[i - 1] = changed_b[j - 1] = a[i - 1] != b[j - 1]
            i, j = i - 1, j - 1
        elif i and distance[i][j] == distance[i - 1][j] + 1:
            changed_a[i - 1] = True
            i -= 1
        else:
            changed_b[j - 1] = True
            j -= 1
    return tuple(changed_a), tuple(changed_b)


def token_difference_mask(tokens, lengths, twin_index):
    """CPU token alignment, cached by token sequences; padding never counts as an edit.

    An insertion-only partner can have no changed token. Its head query falls back
    to valid context tokens. The returned mask itself still records only real edits.
    """
    tokens = tokens.detach().cpu()
    sequences = [tuple(row[:int(n)].tolist()) for row, n in zip(tokens, lengths)]
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    for a, b in twin_index.cpu().tolist():
        left, right = _aligned_changes(sequences[a], sequences[b])
        mask[a, :len(left)] = torch.tensor(left, dtype=torch.bool)
        mask[b, :len(right)] = torch.tensor(right, dtype=torch.bool)
    return mask


class PairChoiceHead(nn.Module):
    """Encode both twins, emphasize edited words, and ground queries in image cells.

    Shared candidate utilities are conditioned on both texts. Subtracting them makes
    the margin antisymmetric under exchanging candidates. Row logits are +/- half
    that margin: existing twin_normalize then applies sigmoid(margin) exactly once.
    No category, pair id, row id or label enters the head.
    """

    requires_pairs = True

    def __init__(self, in_channels, text_dim, dim=128, queries=4, hidden=128,
                 dropout=0.1, coord_channels=True, input_pool=None,
                 difference_aware=True, requires_spatial=True):
        super().__init__()
        if min(dim, queries, hidden) < 1:
            raise ValueError("dim, queries and hidden must be positive")
        self.dim, self.queries = dim, queries
        self.input_pool = input_pool
        self.difference_aware = difference_aware
        self.cell = nn.Sequential(nn.Conv2d(in_channels, dim, 1), nn.BatchNorm2d(dim))
        self.xy = nn.Linear(2, dim, bias=False) if coord_channels else None
        self.word_attention = nn.Linear(text_dim, queries)
        self.difference_bias = nn.Parameter(torch.full((queries,), 2.0))
        self.word_query = nn.Linear(text_dim, dim)
        self.sentence_query = nn.Linear(text_dim, queries * dim)
        self.difference_query = nn.Linear(2 * text_dim, queries * dim, bias=False)
        self.query_norm = nn.LayerNorm(dim)
        self.query_bias = nn.Linear(text_dim, queries)
        # Dot products and explicit spatial summaries give the scorer an interaction
        # gradient immediately, without quadratic cell-to-cell self-attention.
        extras = queries * 5 + (2 * queries * queries if coord_channels else 0)
        self.out = nn.Sequential(nn.Linear(queries * dim * 3 + extras, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, image, text, twin_index, difference_mask):
        """image [P,C,H,W], text for B=2P rows, unoriented twin_index [P,2]."""
        sentence, words, mask = text
        if image.ndim != 4:
            raise ValueError("PairChoiceHead requires spatial image feature maps")
        pairs = twin_index.to(sentence.device)
        if self.input_pool:
            image = F.avg_pool2d(image, self.input_pool)
        cells = self.cell(image).flatten(2).transpose(1, 2)
        h, w = image.shape[-2:]
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=image.device, dtype=cells.dtype),
                                torch.linspace(-1, 1, w, device=image.device, dtype=cells.dtype), indexing="ij")
        xy = torch.stack([xs.flatten(), ys.flatten()], dim=-1)
        if self.xy is not None:
            cells = cells + self.xy(xy)[None]

        changed = difference_mask[:, :words.shape[1]].to(mask.device) & mask
        if not self.difference_aware:
            changed = torch.zeros_like(changed)
        focus = torch.where(changed.any(1, keepdim=True), changed, mask)
        difference = (words * focus[..., None]).sum(1) / focus.sum(1, keepdim=True).clamp_min(1)
        paired_diff = difference[pairs]
        delta = paired_diff - paired_diff.flip(1)
        att = self.word_attention(words)
        if self.difference_aware:
            att = att + changed[..., None] * self.difference_bias
        att = att.masked_fill(~mask[..., None], -1e4).softmax(1)
        query = self.word_query(torch.einsum("blk,bld->bkd", att, words))
        query = query + self.sentence_query(sentence).view(-1, self.queries, self.dim)
        query = query[pairs]
        if self.difference_aware:
            query = query + self.difference_query(torch.cat([paired_diff, delta], -1)).view_as(query)
        query = self.query_norm(query)
        scores = torch.einsum("pskd,pnd->pskn", query, cells) / math.sqrt(self.dim)
        scores = scores + self.query_bias(sentence)[pairs][..., None]
        weights = scores.softmax(-1)
        attended = torch.einsum("pskn,pnd->pskd", weights, cells)
        location = torch.einsum("pskn,nc->pskc", weights, xy)
        if self.xy is None:
            location = torch.zeros_like(location)
        summaries = [torch.logsumexp(scores, -1) - math.log(cells.shape[1]),
                     scores.sigmoid().mean(-1), scores.amax(-1), location.flatten(2)]
        if self.xy is not None:
            summaries.append((location.unsqueeze(3) - location.unsqueeze(2)).flatten(2))
        features = torch.cat([attended.flatten(2), query.flatten(2),
                              (attended * query).flatten(2), *summaries], -1)
        utility = self.out(features).squeeze(-1)
        margin = utility[:, 0] - utility[:, 1]
        # Scatter back to the original row order, including non-adjacent twins.
        return margin.new_zeros(sentence.shape[0]).scatter(
            0, pairs.flatten(), torch.stack([margin * .5, -margin * .5], 1).flatten())
