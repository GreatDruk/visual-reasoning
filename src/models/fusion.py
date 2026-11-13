"""Fusion heads: combine image features and caption encoding into one truth logit.

Every head has the signature forward(image, text) -> logits [B], where `image` is a
feature map [B, C, H, W] (or a vector [B, C] for non-spatial heads) and `text` is the
tuple (sentence [B, D], token_states [B, L, D], mask [B, L]) from a text encoder.
`requires_spatial` tells the model whether the encoder must keep the spatial map.
"""

import torch
import torch.nn.functional as F
from torch import nn


def coord_map(x):
    """Two CoordConv channels (x, y in [-1, 1]) shaped like x[:, :2]."""
    b, _, h, w = x.shape
    ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    return torch.cat([xs, ys], dim=1)


def with_coords(x, enabled=True):
    return torch.cat([x, coord_map(x)], dim=1) if enabled else x


def mlp(dims, dropout=0.0, last_act=False):
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2 or last_act:
            layers.append(nn.ReLU(inplace=True))
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class CosineHead(nn.Module):
    """Lab baseline: scale * cos(proj(i), proj(t)) + bias."""

    def __init__(self, in_channels, text_dim, proj_dim=256, init_scale=10.0, init_bias=0.0, requires_spatial=False):
        super().__init__()
        self.img_proj = nn.Linear(in_channels, proj_dim)
        self.txt_proj = nn.Linear(text_dim, proj_dim)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, image, text):
        if image.dim() == 4:
            image = image.mean(dim=(2, 3))
        i = F.normalize(self.img_proj(image), dim=-1)
        t = F.normalize(self.txt_proj(text[0]), dim=-1)
        return self.scale * (i * t).sum(-1) + self.bias


class ConcatMLPHead(nn.Module):
    """MLP over [i, t, i*t, |i-t|] of projected pooled vectors."""

    def __init__(self, in_channels, text_dim, proj_dim=512, hidden_dims=(1024, 256), dropout=0.2, requires_spatial=False):
        super().__init__()
        self.img_proj = nn.Linear(in_channels, proj_dim)
        self.txt_proj = nn.Linear(text_dim, proj_dim)
        self.mlp = mlp([4 * proj_dim, *hidden_dims, 1], dropout)

    def forward(self, image, text):
        if image.dim() == 4:
            image = image.mean(dim=(2, 3))
        i, t = self.img_proj(image), self.txt_proj(text[0])
        return self.mlp(torch.cat([i, t, i * t, (i - t).abs()], dim=-1)).squeeze(-1)


class FiLMBlock(nn.Module):
    """FiLM residual block (Perez et al., 2018): conv1x1 -> conv3x3 -> BN -> FiLM -> ReLU + skip."""

    def __init__(self, channels, coord_channels=True):
        super().__init__()
        extra = 2 if coord_channels else 0
        self.coord_channels = coord_channels
        self.conv1 = nn.Conv2d(channels + extra, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels, affine=False)

    def forward(self, x, gamma, beta):
        h = F.relu(self.conv1(with_coords(x, self.coord_channels)))
        h = self.bn(self.conv2(h))
        h = F.relu(gamma[..., None, None] * h + beta[..., None, None])
        return x + h


class FiLMStack(nn.Module):
    """Stem + N FiLM blocks; gamma/beta are predicted from the sentence vector."""

    def __init__(self, in_channels, text_dim, num_blocks=4, hidden_channels=128, coord_channels=True):
        super().__init__()
        extra = 2 if coord_channels else 0
        self.coord_channels = coord_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels + extra, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(FiLMBlock(hidden_channels, coord_channels) for _ in range(num_blocks))
        self.film = nn.Linear(text_dim, num_blocks * 2 * hidden_channels)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.num_blocks, self.hidden = num_blocks, hidden_channels
        self.out_channels = hidden_channels

    def forward(self, image, sentence):
        x = self.stem(with_coords(image, self.coord_channels))
        params = self.film(sentence).view(-1, self.num_blocks, 2, self.hidden)
        for k, block in enumerate(self.blocks):
            # gamma = 1 + delta keeps the blocks near identity modulation at init.
            x = block(x, 1.0 + params[:, k, 0], params[:, k, 1])
        return x


class FiLMHead(nn.Module):
    """FiLM stack + conv1x1 classifier, global max pool, MLP."""

    def __init__(
        self,
        in_channels,
        text_dim,
        num_blocks=4,
        hidden_channels=128,
        coord_channels=True,
        classifier_channels=512,
        classifier_hidden=1024,
        dropout=0.1,
        requires_spatial=True,
    ):
        super().__init__()
        self.stack = FiLMStack(in_channels, text_dim, num_blocks, hidden_channels, coord_channels)
        extra = 2 if coord_channels else 0
        self.coord_channels = coord_channels
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_channels + extra, classifier_channels, 1),
            nn.BatchNorm2d(classifier_channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = mlp([classifier_channels, classifier_hidden, 1], dropout)

    def forward(self, image, text):
        x = self.stack(image, text[0])
        x = self.proj(with_coords(x, self.coord_channels))
        return self.classifier(x.amax(dim=(2, 3))).squeeze(-1)


class MACHead(nn.Module):
    """MAC network (Hudson & Manning, 2018), simplified: control, read and write units
    with shared weights across steps and step-specific question projections."""

    def __init__(
        self,
        in_channels,
        text_dim,
        dim=512,
        max_steps=8,
        self_attention=False,
        memory_gate=False,
        dropout=0.15,
        coord_channels=True,
        requires_spatial=True,
    ):
        super().__init__()
        extra = 2 if coord_channels else 0
        self.coord_channels = coord_channels
        self.steps = max_steps
        self.memory_gate = memory_gate
        self.kb = nn.Sequential(
            nn.Conv2d(in_channels + extra, dim, 1),
            nn.ELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ELU(),
        )
        self.q_proj = nn.Linear(text_dim, dim)
        self.w_proj = nn.Linear(text_dim, dim)
        self.q_step = nn.ModuleList(nn.Linear(dim, dim) for _ in range(max_steps))
        self.cq = nn.Linear(2 * dim, dim)
        self.c_attn = nn.Linear(dim, 1)
        self.r_mem = nn.Linear(dim, dim)
        self.r_kb = nn.Linear(dim, dim)
        self.r_cat = nn.Linear(2 * dim, dim)
        self.r_attn = nn.Linear(dim, 1)
        self.w_cat = nn.Linear(2 * dim, dim)
        self.gate = nn.Linear(dim, 1) if memory_gate else None
        self.c0 = nn.Parameter(torch.zeros(dim))
        self.m0 = nn.Parameter(torch.zeros(dim))
        self.drop = nn.Dropout(dropout)
        self.out = mlp([2 * dim, dim, 1], dropout)

    def forward(self, image, text):
        sentence, words, mask = text
        kb = self.kb(with_coords(image, self.coord_channels)).flatten(2).transpose(1, 2)  # [B, N, d]
        q = self.q_proj(sentence)
        words = self.w_proj(words)
        b = q.shape[0]
        c, m = self.c0.expand(b, -1), self.m0.expand(b, -1)
        kb_proj = self.r_kb(kb)
        for i in range(self.steps):
            cq = self.cq(torch.cat([c, self.q_step[i](torch.tanh(q))], dim=-1))
            logits = self.c_attn(cq.unsqueeze(1) * words).squeeze(-1).masked_fill(~mask, -1e4)
            c = (logits.softmax(-1).unsqueeze(-1) * words).sum(1)
            inter = self.r_mem(self.drop(m)).unsqueeze(1) * kb_proj
            inter = self.r_cat(torch.cat([inter, kb], dim=-1))
            r_logits = self.r_attn(F.elu(c.unsqueeze(1) * inter)).squeeze(-1)
            r = (r_logits.softmax(-1).unsqueeze(-1) * kb).sum(1)
            m_new = self.w_cat(torch.cat([r, m], dim=-1))
            if self.gate is not None:
                g = torch.sigmoid(self.gate(c))
                m_new = g * m + (1 - g) * m_new
            m = m_new
        return self.out(torch.cat([q, m], dim=-1)).squeeze(-1)


class RelationModule(nn.Module):
    """Relation Network over feature-map cells: f(sum_ij g(o_i, o_j, q)).

    The first g layer is linear in [o_i, o_j, q], so it is computed per cell and
    broadcast over pairs instead of materialising the concatenation.
    """

    def __init__(self, obj_dim, q_dim, g_hidden=256, g_layers=4, f_hidden=256, f_layers=2, dropout=0.1):
        super().__init__()
        self.wi = nn.Linear(obj_dim, g_hidden)
        self.wj = nn.Linear(obj_dim, g_hidden, bias=False)
        self.wq = nn.Linear(q_dim, g_hidden, bias=False)
        self.g = mlp([g_hidden] * g_layers, last_act=True)
        self.f = mlp([g_hidden] + [f_hidden] * (f_layers - 1) + [1], dropout)

    def forward(self, objects, q):
        # objects [B, N, d], q [B, Dq]
        a = self.wi(objects).unsqueeze(2)  # [B, N, 1, h]
        b = self.wj(objects).unsqueeze(1)  # [B, 1, N, h]
        c = self.wq(q)[:, None, None, :]
        pairs = F.relu(a + b + c)
        pairs = self.g(pairs)
        return self.f(pairs.sum(dim=(1, 2)) / objects.shape[1]).squeeze(-1)


class RelationHead(nn.Module):
    """Relation Network head on a pooled feature map with coordinate channels."""

    def __init__(
        self,
        in_channels,
        text_dim,
        pool_to=(8, 12),
        coord_channels=True,
        g_hidden=256,
        g_layers=4,
        f_hidden=256,
        f_layers=2,
        dropout=0.1,
        obj_channels=64,
        requires_spatial=True,
    ):
        super().__init__()
        self.pool_to = tuple(pool_to)
        self.coord_channels = coord_channels
        self.reduce = nn.Sequential(nn.Conv2d(in_channels, obj_channels, 1), nn.BatchNorm2d(obj_channels), nn.ReLU(inplace=True))
        obj_dim = obj_channels + (2 if coord_channels else 0)
        self.rn = RelationModule(obj_dim, text_dim, g_hidden, g_layers, f_hidden, f_layers, dropout)

    def forward(self, image, text):
        x = self.reduce(F.adaptive_avg_pool2d(image, self.pool_to))
        x = with_coords(x, self.coord_channels)
        return self.rn(x.flatten(2).transpose(1, 2), text[0])


class FiLMRelationHead(nn.Module):
    """FiLM stack followed by a Relation Network head instead of the max-pool classifier."""

    def __init__(self, in_channels, text_dim, film, rn, dropout=0.1, requires_spatial=True):
        super().__init__()
        self.stack = FiLMStack(in_channels, text_dim, **film)
        self.pool_to = tuple(rn.get("pool_to", (8, 12)))
        coord = film.get("coord_channels", True)
        self.coord_channels = coord
        obj_dim = self.stack.out_channels + (2 if coord else 0)
        self.rn = RelationModule(
            obj_dim,
            text_dim,
            g_hidden=rn.get("g_hidden", 256),
            g_layers=rn.get("g_layers", 3),
            f_hidden=rn.get("f_hidden", 256),
            dropout=dropout,
        )

    def forward(self, image, text):
        x = self.stack(image, text[0])
        x = F.adaptive_avg_pool2d(x, self.pool_to)
        x = with_coords(x, self.coord_channels)
        return self.rn(x.flatten(2).transpose(1, 2), text[0])
