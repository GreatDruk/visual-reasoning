"""Independent caption verifier with separate target, comparator, and reference queries."""

import torch
from torch import nn
from torch.nn import functional as F

from src.models.slot_program import ATTRIBUTES, EXACT


def grid(shape, device):
    axes = [torch.linspace(-1, 1, n, device=device) for n in shape]
    y, x = torch.meshgrid(*axes, indexing="ij")
    return torch.stack((x, y), dim=-1).flatten(0, 1)


class EvidenceVerifier(nn.Module):
    def __init__(self, width=80):
        super().__init__()
        self.words = nn.Embedding(len(ATTRIBUTES) + 1, width, padding_idx=0)
        self.roles = nn.Parameter(torch.randn(3, width) * .08)
        self.detail = nn.Sequential(nn.LayerNorm(99), nn.Linear(99, width))
        self.semantic = nn.Sequential(nn.LayerNorm(144), nn.Linear(144, width))
        self.position = nn.Linear(2, width, bias=False)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.op = nn.Embedding(7, 12)
        self.relation = nn.Embedding(7, 12)
        self.number = nn.Embedding(20, 12)
        self.count = nn.Sequential(nn.LayerNorm(width + 5), nn.Linear(width + 5, width),
                                   nn.GELU(), nn.Linear(width, 20))
        feature_width = 3 * (width + 5 + 20) + 3 * 12
        self.logit = nn.Sequential(nn.LayerNorm(feature_width), nn.Linear(feature_width, 192),
                                   nn.GELU(), nn.Dropout(.10), nn.Linear(192, 1))
        self.register_buffer("axes", torch.tensor([[0., 0.], [-1., 0.], [1., 0.],
                                                  [0., 1.], [0., -1.], [-1., 0.], [1., 0.]]))

    def forward(self, detail, semantic, words, meta):
        xy = grid(detail.shape[1:3], detail.device)
        fine = self.detail(detail)
        context = self.semantic(semantic)
        context = F.interpolate(context.permute(0, 3, 1, 2), detail.shape[1:3],
                                mode="bilinear", align_corners=False)
        visual = (fine + context.permute(0, 2, 3, 1)).flatten(1, 2)
        keys = self.key(visual + self.position(xy))
        roles = self.roles + self.words(words).sum(dim=2)
        hit = (self.query(roles) @ keys.transpose(1, 2)).float() / keys.shape[-1] ** .5
        op, relation, number = meta.unbind(dim=-1)

        # Constrain the target to the requested half-space of the reference evidence.
        # Average the gate across plausible reference locations, not a coordinate average.
        reference_map = F.avg_pool2d(hit[:, 2].reshape(-1, 1, *detail.shape[1:3]), 2)
        ref_weights = reference_map.flatten(1).softmax(dim=-1).reshape_as(reference_map)[:, 0]
        axis = self.axes[relation]
        height, width = detail.shape[1:3]
        delta_x = xy[:width, 0, None] - torch.linspace(-1, 1, ref_weights.shape[2],
                                                       device=xy.device)
        delta_y = xy[::width, 1, None] - torch.linspace(-1, 1, ref_weights.shape[1],
                                                         device=xy.device)
        horizontal = (axis[:, 0, None, None] * 16 * delta_x).sigmoid()
        vertical = (axis[:, 1, None, None] * 16 * delta_y).sigmoid()
        gate_x = (horizontal @ ref_weights.sum(1)[..., None]).transpose(1, 2)
        gate_y = vertical @ ref_weights.sum(2)[..., None]
        relation_gate = torch.where((relation <= 2)[:, None, None], gate_x, gate_y)
        relation_gate = relation_gate.expand(-1, height, width).flatten(1)
        absolute_gate = torch.sigmoid(16 * (axis @ xy.T))
        gate = torch.where((relation >= 5)[:, None], absolute_gate, relation_gate)
        target = hit[:, 0] + torch.where((relation > 0)[:, None], gate.clamp_min(1e-4).log(), 0)
        evidence = torch.cat((target[:, None], hit[:, 1:]), dim=1)
        weights = evidence.softmax(dim=-1)
        content = weights.to(visual.dtype) @ visual
        location = weights @ xy
        summary = torch.cat((content.float(), location, evidence.amax(-1, keepdim=True),
                             evidence.topk(8, dim=-1).values.mean(-1, keepdim=True),
                             evidence.sigmoid().mean(-1, keepdim=True)), dim=-1)
        count = self.count(summary)
        text = torch.cat((self.op(op), self.relation(relation), self.number(number)), dim=-1)
        joint = torch.cat((summary, count.log_softmax(-1)), dim=-1).flatten(1)
        return self.logit(torch.cat((joint, text), dim=-1)).flatten().float(), count[:, 0].float()


def pair_loss(logits, count, meta, weight):
    truth = logits.new_tensor([1., 0.]).expand_as(logits)
    bce = F.binary_cross_entropy_with_logits(logits, truth, reduction="none").mean(dim=1)
    ranking = F.softplus(logits[:, 1] - logits[:, 0])
    exact = meta[:, 0, 0] == EXACT
    count_loss = F.cross_entropy(count[:, 0], meta[:, 0, 2], reduction="none")
    return ((bce + .5 * ranking + .25 * count_loss * exact) * weight).mean()
