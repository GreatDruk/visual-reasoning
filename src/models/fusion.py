"""Fusion heads: combine image features and caption encoding into one truth logit.

Every head has the signature forward(image, text) -> logits [B], where `image` is a
feature map [B, C, H, W] (or a vector [B, C] for non-spatial heads) and `text` is the
tuple (sentence [B, D], token_states [B, L, D], mask [B, L]) from a text encoder.
`requires_spatial` tells the model whether the encoder must keep the spatial map.
"""

import math

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


def adaptive_pool(x, size):
    """adaptive_avg_pool2d that also works on MPS for non-divisible sizes (via CPU)."""
    size = tuple(size)
    h, w = x.shape[-2:]
    if x.device.type == "mps" and (h % size[0] or w % size[1]):
        return F.adaptive_avg_pool2d(x.cpu(), size).to(x.device)
    return F.adaptive_avg_pool2d(x, size)


def mlp(dims, dropout=0.0, last_act=False, activation="relu"):
    if activation not in ("relu", "gelu"):
        raise ValueError(f"unsupported MLP activation {activation!r}; expected relu or gelu")
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2 or last_act:
            layers.append(nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True))
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


class BilinearReadout(nn.Module):
    """Final logit from a visual summary z and the sentence vector.

    mode="mlp":          MLP(z)
    mode="bilinear_mlp": MLP([z, z * W q]) (first-order interaction at init)
    mode="cosine":       scale * cos(W z, U q) + bias (bounded logit, like the lab
                         baseline; on 5k images unbounded MLP read-outs memorised the
                         training pairs and stayed at chance on val, see KaggleTemplate.md)
    """

    def __init__(self, dim, text_dim, hidden, dropout=0.1, mode="cosine", proj_dim=256, init_scale=10.0, activation="relu"):
        super().__init__()
        if mode is True:
            mode = "bilinear_mlp"
        elif mode is False:
            mode = "mlp"
        self.mode = mode
        if mode == "cosine":
            self.zp = nn.Linear(dim, proj_dim)
            self.qp = nn.Linear(text_dim, proj_dim)
            self.scale = nn.Parameter(torch.tensor(float(init_scale)))
            self.bias = nn.Parameter(torch.tensor(0.0))
            self.drop = nn.Dropout(dropout)
        else:
            self.q = nn.Linear(text_dim, dim) if mode == "bilinear_mlp" else None
            self.mlp = mlp([dim * (2 if self.q is not None else 1), hidden, 1], dropout, activation=activation)

    def forward(self, z, sentence):
        if self.mode == "cosine":
            a = F.normalize(self.zp(self.drop(z)), dim=-1)
            b = F.normalize(self.qp(sentence), dim=-1)
            return self.scale * (a * b).sum(-1) + self.bias
        if self.q is not None:
            z = torch.cat([z, z * self.q(sentence)], dim=-1)
        return self.mlp(z).squeeze(-1)


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
        # (default init: a zero init leaves no text-image interaction to start learning from)
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
        readout="cosine",
        input_pool=None,
        feature_dropout=0.0,
        requires_spatial=True,
        readout_activation="relu",
    ):
        super().__init__()
        self.input_pool = tuple(input_pool) if input_pool else None
        self.in_drop = nn.Dropout2d(feature_dropout)
        self.stack = FiLMStack(in_channels, text_dim, num_blocks, hidden_channels, coord_channels)
        extra = 2 if coord_channels else 0
        self.coord_channels = coord_channels
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_channels + extra, classifier_channels, 1),
            nn.BatchNorm2d(classifier_channels),
            nn.ReLU(inplace=True),
        )
        self.readout = BilinearReadout(
            classifier_channels, text_dim, classifier_hidden, dropout, readout, activation=readout_activation
        )

    def forward(self, image, text):
        if self.input_pool:
            image = adaptive_pool(image, self.input_pool)
        x = self.stack(self.in_drop(image), text[0])
        x = self.proj(with_coords(x, self.coord_channels))
        return self.readout(x.amax(dim=(2, 3)), text[0])


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
        x = self.reduce(adaptive_pool(image, self.pool_to))
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
        x = adaptive_pool(x, self.pool_to)
        x = with_coords(x, self.coord_channels)
        return self.rn(x.flatten(2).transpose(1, 2), text[0])


class TransformerHead(nn.Module):
    """Joint self-attention over [CLS] + caption tokens + feature-map cells.

    Cells get a 1x1 projection plus a learned 2D position embedding; tokens are the
    text encoder's states. Attention dot products between words and cells give a strong
    multiplicative text-image interaction from the first step, which FiLM's sentence-level
    modulation lacks on this small dataset. The [CLS] output feeds an MLP.
    """

    def __init__(
        self,
        in_channels,
        text_dim,
        dim=256,
        num_layers=4,
        num_heads=8,
        ff_dim=512,
        dropout=0.1,
        max_cells=(20, 30),
        readout="cosine",
        requires_spatial=True,
        input_pool=None,
    ):
        super().__init__()
        self.input_pool = input_pool  # avg-pool factor for the input map (attention cost ~ cells^2)
        self.cell_proj = nn.Sequential(nn.Conv2d(in_channels, dim, 1), nn.BatchNorm2d(dim))
        self.pos = nn.Parameter(torch.zeros(1, dim, *max_cells))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.tok_proj = nn.Linear(text_dim, dim)
        self.type_emb = nn.Parameter(torch.zeros(3, dim))  # cls, word, cell
        nn.init.trunc_normal_(self.type_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, num_heads, ff_dim, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.readout = BilinearReadout(dim, text_dim, dim, dropout, readout)

    def forward(self, image, text):
        _, words, mask = text
        if self.input_pool:
            image = F.avg_pool2d(image, self.input_pool)
        cells = self.cell_proj(image)
        h, w = cells.shape[-2:]
        pos = self.pos if self.pos.shape[-2:] == (h, w) else F.interpolate(self.pos, size=(h, w), mode="bilinear", align_corners=False)
        cells = (cells + pos).flatten(2).transpose(1, 2) + self.type_emb[2]
        words = self.tok_proj(words) + self.type_emb[1]
        cls = self.type_emb[0].expand(words.shape[0], 1, -1)
        x = torch.cat([cls, words, cells], dim=1)
        pad = torch.cat([torch.zeros_like(mask[:, :1]), ~mask, torch.zeros(mask.shape[0], cells.shape[1], dtype=torch.bool, device=mask.device)], dim=1)
        x = self.encoder(x, src_key_padding_mask=pad)
        return self.readout(self.norm(x[:, 0]), text[0])


class GroundingHead(nn.Module):
    """Word queries that attend directly to feature-map cells (no cell self-attention).

    Words are pooled into `queries` text queries (learned soft word selection plus a
    sentence term); each query takes a softmax over cells (1x1 projection + learned and
    coordinate position embeddings) by dot product. The readout sees the attended cell
    summary z, the query q and z * q, so a multiplicative text-image interaction reaches
    the logit from the first step. TransformerHead buries that interaction under joint
    self-attention and sat at chance for ~3800 steps; this head has no such plateau
    (CPU research on cached SigLIP2 maps, scripts/codex_cpu_research.py).
    """

    def __init__(self, in_channels, text_dim, dim=256, queries=4, hidden=128, dropout=0.1,
                 max_cells=(8, 12), input_pool=None, requires_spatial=True):
        super().__init__()
        self.dim, self.queries, self.input_pool = dim, queries, input_pool
        self.cell = nn.Sequential(nn.Conv2d(in_channels, dim, 1), nn.BatchNorm2d(dim))
        self.pos = nn.Parameter(torch.randn(1, dim, *max_cells) * 0.02)
        self.xy = nn.Linear(2, dim, bias=False)
        self.word_att = nn.Linear(text_dim, queries)
        self.word_q = nn.Linear(text_dim, dim)
        self.sent_q = nn.Linear(text_dim, queries * dim)
        self.q_norm = nn.LayerNorm(dim)
        self.bias = nn.Linear(text_dim, queries)
        self.out = nn.Sequential(nn.Linear(queries * dim * 3, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, image, text):
        sent, words, mask = text
        if self.input_pool:
            image = F.avg_pool2d(image, self.input_pool)
        cells = self.cell(image)
        h, w = cells.shape[-2:]
        pos = self.pos if self.pos.shape[-2:] == (h, w) else F.interpolate(self.pos, size=(h, w), mode="bilinear", align_corners=False)
        k = (cells + pos).flatten(2).transpose(1, 2) + self.xy(coord_map(cells[:1]).flatten(2).transpose(1, 2))
        att = self.word_att(words).masked_fill(~mask[..., None], -1e4).softmax(1)
        q = self.word_q(torch.einsum("blk,bld->bkd", att, words))
        q = self.q_norm(q + self.sent_q(sent).view(-1, self.queries, self.dim))
        scores = torch.einsum("bkd,bnd->bkn", q, k) / math.sqrt(self.dim) + self.bias(sent)[..., None]
        z = torch.einsum("bkn,bnd->bkd", scores.softmax(-1), k)
        return self.out(torch.cat([z.flatten(1), q.flatten(1), (z * q).flatten(1)], -1)).squeeze(-1)


class QueryHead(nn.Module):
    """Text-conditioned probes over feature-map cells, with relational read-outs.

    The caption produces K query vectors (attention over word states). Each query is
    scored against every cell embedding (cell features + coordinates):
        s[k, i] = <q_k, v_i> / sqrt(d) + b_k
    and summarised into features that are first-order in the text-image interaction:
      - existence: logsumexp_i s[k, i] (soft max),
      - count: sum_i sigmoid(s[k, i]) / cells_per_object,
      - location: softmax_i(s[k]) weighted (x, y),
      - relations: for every query pair (k, l), the soft count of cells matching l that
        lie left/right/behind/in front of query k's attended location.
    An MLP over these features (and the sentence) gives the logit. Unlike FiLM/MAC with an
    MLP read-out, the gradient for aligning words with visual directions exists from the
    first step, so training does not sit on the chance-level saddle.
    """

    def __init__(
        self,
        in_channels,
        text_dim,
        dim=256,
        num_queries=6,
        hidden=512,
        dropout=0.1,
        relation_temp=8.0,
        feature_dropout=0.0,
        sentence_in_readout=True,
        readout="mlp",
        requires_spatial=True,
    ):
        super().__init__()
        self.k = num_queries
        self.in_drop = nn.Dropout2d(feature_dropout)
        self.sentence_in_readout = sentence_in_readout
        self.cells = nn.Sequential(nn.Conv2d(in_channels + 2, dim, 1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True), nn.Conv2d(dim, dim, 1))
        self.word_attn = nn.Linear(text_dim, num_queries)
        self.query = nn.Linear(text_dim, dim)
        self.query_sent = nn.Linear(text_dim, num_queries * dim)
        self.bias = nn.Linear(text_dim, num_queries)
        # Unit-scale scores at init: with tiny ||q||, ||v|| the scores are near-constant and
        # the interaction gradient is negligible.
        self.v_norm = nn.GroupNorm(1, dim, affine=False)
        self.q_norm = nn.LayerNorm(dim)
        self.temp = relation_temp
        n_feat = num_queries * 5 + num_queries * num_queries * 4
        self.feat_norm = nn.LayerNorm(n_feat)
        self.readout_mode = readout
        if readout == "mlp":
            self.mlp = mlp([n_feat + (text_dim if sentence_in_readout else 0), hidden, hidden, 1], dropout)
        else:
            self.readout = BilinearReadout(n_feat, text_dim, hidden, dropout, readout)

    def forward(self, image, text):
        sentence, words, mask = text
        b = image.shape[0]
        image = self.in_drop(image)
        coords = coord_map(image)
        v = self.v_norm(self.cells(torch.cat([image, coords], dim=1))).flatten(2)  # [B, d, N]
        xy = coords.flatten(2)  # [B, 2, N]
        att = self.word_attn(words).masked_fill(~mask[..., None], -1e4).softmax(dim=1)  # [B, L, K]
        q = self.q_norm(self.query(torch.einsum("blk,bld->bkd", att, words)) + self.query_sent(sentence).view(b, self.k, -1))
        s = torch.einsum("bkd,bdn->bkn", q, v) / v.shape[1] ** 0.5 + self.bias(sentence)[..., None]  # [B, K, N]

        exist = torch.logsumexp(s, dim=-1) - math.log(s.shape[-1])
        p = torch.sigmoid(s)
        count = p.sum(-1)
        w = s.softmax(-1)
        loc = torch.einsum("bkn,bcn->bkc", w, xy)  # [B, K, 2] attended (x, y)
        # Directional soft counts: cells of query l relative to the location of query k.
        dx = xy[:, None, 0, :] - loc[..., 0:1]  # [B, K, N]
        dy = xy[:, None, 1, :] - loc[..., 1:2]
        dirs = torch.stack([torch.sigmoid(-self.temp * dx), torch.sigmoid(self.temp * dx), torch.sigmoid(-self.temp * dy), torch.sigmoid(self.temp * dy)], dim=2)  # [B, K, 4, N]
        rel = torch.einsum("bkrn,bln->bklr", dirs, p).flatten(1)  # [B, K*K*4]
        feats = torch.cat([exist, count, loc.flatten(1), p.amax(-1), rel], dim=-1)
        feats = self.feat_norm(feats)
        if self.readout_mode != "mlp":
            return self.readout(feats, sentence)
        if self.sentence_in_readout:
            feats = torch.cat([feats, sentence], dim=-1)
        return self.mlp(feats).squeeze(-1)
