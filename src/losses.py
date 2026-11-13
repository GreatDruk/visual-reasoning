import torch
import torch.nn.functional as F
from torch import nn


class TruthLoss(nn.Module):
    """BCE on truth logits + pair_weight * pairwise logistic loss over twins.

    For every twin pair (same pair_id, exactly one true) present in the batch the pair
    term is log(1 + exp(-(s_true - s_false))): it only cares about the ranking inside a
    pair, which is the hard part of the task (twins differ by one detail).
    """

    def __init__(self, pair_weight=0.0, label_smoothing=0.0):
        super().__init__()
        self.pair_weight = pair_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels, pair_index=None):
        """pair_index: LongTensor [P, 2] of (true_idx, false_idx) rows into the batch."""
        target = labels.float() * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        loss = F.binary_cross_entropy_with_logits(logits, target)
        if self.pair_weight > 0 and pair_index is not None and len(pair_index):
            margin = logits[pair_index[:, 0]] - logits[pair_index[:, 1]]
            loss = loss + self.pair_weight * F.softplus(-margin).mean()
        return loss
