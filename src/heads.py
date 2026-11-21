"""Object table refinement shared by both visual branches."""

import numpy as np
import torch
from torch import nn

from .programs import ATTRIBUTES, EXCLUSIVE_GROUPS, WORD_INDEX
from .reasoning import ProgramExecutor, log1mexp

GROUPS = [np.array([WORD_INDEX[w] for w in g]) for g in EXCLUSIVE_GROUPS]
N_ATTRS = sum(map(len, GROUPS))
FEATURES = 1 + N_ATTRS + 4 + 3 + N_ATTRS
BOX = slice(1 + N_ATTRS, 5 + N_ATTRS)


class TableHead(ProgramExecutor):
    def __init__(self, features=FEATURES, teacher=None):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Linear(features, 48),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(48, 3 + N_ATTRS),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.register_buffer("attribute_indices", torch.as_tensor(np.concatenate(GROUPS)))
        if teacher is not None:
            with torch.no_grad():
                for key in (
                    "log_reference_temperature",
                    "log_sharpness",
                    "relation_offset",
                ):
                    getattr(self, key).copy_(teacher[key])

    def encode(self, features, positions, valid):
        residual = self.residual(features)
        return dict(
            pooled=features,
            position=positions + 0.08 * residual[..., -2:].tanh(),
            residual=residual,
            valid=valid,
        )

    def slot_logits(self, encoded):
        x, change = encoded["pooled"], encoded["residual"]
        obj = (x[..., 0] + 3 * change[..., 0].tanh()).masked_fill(~encoded["valid"], -30.0)
        revised = x[..., 1 : 1 + N_ATTRS] + 2 * change[..., 1 : 1 + N_ATTRS].tanh()
        words = x.new_zeros((*x.shape[:-1], len(ATTRIBUTES) + 1)).index_copy(
            -1, self.attribute_indices, revised
        )
        return obj, words

    def score(self, table, scenes, programs):
        truth, _, _ = self(self.encode(*table), scenes, programs)
        return truth - log1mexp(truth)

    def existence_loss(self, table, scenes, words):
        obj, attrs = self.slot_logits(self.encode(*table))
        match = self.membership(obj[scenes], attrs[scenes], words, True)
        return -log1mexp(log1mexp(match).sum(-1)).mean()


def load_head(model, state):
    """Read historical heads while rejecting every unrecognised live parameter.

    The archived executor allocated six unused readouts which Fusion overrode.
    They are deliberately absent from the maintained implementation.
    """
    unused = ("feature_norm.", "embed.", "words.", "word_bias.", "objectness.")
    live = {k: v for k, v in state.items() if not k.startswith(unused)}
    model.load_state_dict(live, strict=True)


def probabilities(head, table):
    encoded = head.encode(*table)
    obj, attrs = head.slot_logits(encoded)
    return (
        obj.sigmoid(),
        encoded["position"],
        table[2],
        *(attrs[..., g.tolist()].softmax(-1) for g in GROUPS),
    )
