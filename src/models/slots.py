"""DINOSAUR-style Slot Attention over frozen patch features (unsupervised).

Slot Attention (Locatello et al., 2020) groups the 16x24 SigLIP block-8 tokens into
K slots; an MLP spatial-broadcast decoder reconstructs every (standardized) token as
an alpha-weighted mixture of per-slot predictions (Seitzer et al., ICLR 2023). Only
the reconstruction loss trains it on the selected labeled image splits.

Slots start from K learned queries (BO-QSA, Jia et al., ICLR 2023): the output is
deterministic, and the straight-through final iteration keeps the queries trainable.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


def coordinate_grid(h, w, device=None, dtype=torch.float32):
    """[h*w, 2] (x, y) cell centres in [-1, 1]; y grows downwards (towards the camera)."""
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=device, dtype=dtype),
                            torch.linspace(-1, 1, w, device=device, dtype=dtype), indexing="ij")
    return torch.stack([xs.flatten(), ys.flatten()], -1)


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, input_dim, iterations=3, hidden=None, eps=1e-8):
        super().__init__()
        self.num_slots, self.dim, self.iterations, self.eps = num_slots, dim, iterations, eps
        self.queries = nn.Parameter(torch.randn(num_slots, dim) * dim ** -.5)
        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(input_dim, dim, bias=False)
        self.to_v = nn.Linear(input_dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        hidden = hidden or 2 * dim
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def step(self, slots, k, v):
        q = self.to_q(self.norm_slots(slots))
        logits = torch.einsum("bkd,bnd->bkn", q, k) / math.sqrt(self.dim)
        if logits.dtype == torch.float16:
            logits = logits.float()
        attn = logits.softmax(1) + self.eps            # slots compete for every token
        weights = attn / attn.sum(-1, keepdim=True)    # weighted mean over tokens
        updates = torch.einsum("bkn,bnd->bkd", weights, v)
        b, n_slots, d = slots.shape
        slots = self.gru(updates.reshape(-1, d), slots.reshape(-1, d)).reshape(b, n_slots, d)
        return slots + self.mlp(self.norm_mlp(slots)), attn

    def forward(self, inputs):
        """inputs [B, N, C] -> slots [B, K, D], attention [B, K, N] (sums to 1 over K)."""
        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        init = self.queries[None].expand(inputs.shape[0], -1, -1)
        slots = init
        for i in range(self.iterations):
            if i == self.iterations - 1:
                # Implicit differentiation through the fixed point (Chang et al., 2022)
                # with a straight-through path to the learned queries (BO-QSA).
                slots = slots.detach() + init - init.detach()
            slots, attn = self.step(slots, k, v)
        return slots, attn


class SlotAutoencoder(nn.Module):
    """Frozen-feature slot encoder + MLP broadcast decoder.

    `encode` returns per-slot vectors, attention masks over cells, mask-pooled input
    features and mask centroids: everything a reasoning head needs per object.
    """

    def __init__(self, in_channels=768, grid=(16, 24), num_slots=11, slot_dim=128,
                 input_dim=256, iterations=3, decoder_hidden=512, decoder_layers=3, target_dim=None,
                 background_slots=(), count_threshold=.5, count_sharpness=10.):
        super().__init__()
        self.config = dict(in_channels=in_channels, grid=tuple(grid), num_slots=num_slots,
                           slot_dim=slot_dim, input_dim=input_dim, iterations=iterations,
                           decoder_hidden=decoder_hidden, decoder_layers=decoder_layers,
                           target_dim=target_dim, background_slots=tuple(background_slots),
                           count_threshold=count_threshold, count_sharpness=count_sharpness)
        if any(not 0 <= k < num_slots for k in background_slots):
            raise ValueError("background slot index out of range")
        # Slots excluded from the object count (they absorb the floor / walls).
        self.register_buffer("object_slots", torch.tensor([k not in background_slots for k in range(num_slots)]),
                             persistent=False)
        self.count_threshold, self.count_sharpness = count_threshold, count_sharpness
        h, w = grid
        self.grid = (h, w)
        self.num_slots = num_slots
        # Per-channel standardisation of the frozen tokens (a few channels reach |58|).
        self.register_buffer("feature_mean", torch.zeros(in_channels))
        self.register_buffer("feature_std", torch.ones(in_channels))
        self.register_buffer("xy", coordinate_grid(h, w))
        # Optional PCA target: the decoder output [B, K, N, C] dominates the cost.
        target_dim = target_dim or in_channels
        self.register_buffer("target_proj", torch.eye(in_channels)[:, :target_dim].clone())
        self.input_mlp = nn.Sequential(nn.LayerNorm(in_channels), nn.Linear(in_channels, input_dim),
                                       nn.GELU(), nn.Linear(input_dim, input_dim))
        self.input_xy = nn.Linear(2, input_dim)
        self.slot_attention = SlotAttention(num_slots, slot_dim, input_dim, iterations)
        self.decoder_pos = nn.Parameter(torch.randn(h * w, slot_dim) * .02)
        layers, width = [], slot_dim
        for _ in range(decoder_layers):
            layers += [nn.Linear(width, decoder_hidden), nn.GELU()]
            width = decoder_hidden
        layers.append(nn.Linear(width, target_dim + 1))
        self.decoder = nn.Sequential(*layers)

    @torch.no_grad()
    def fit_normalization(self, features):
        """features [B, C, H, W] sample -> channel mean / std buffers."""
        tokens = features.float().flatten(2).transpose(1, 2).reshape(-1, features.shape[1])
        self.feature_mean.copy_(tokens.mean(0))
        self.feature_std.copy_(tokens.std(0).clamp_min(1e-3))
        if self.target_proj.shape[1] < self.target_proj.shape[0]:
            z = (tokens - self.feature_mean) / self.feature_std
            _, vectors = torch.linalg.eigh(torch.cov(z.T))
            self.target_proj.copy_(vectors.flip(1)[:, :self.target_proj.shape[1]])

    def tokens(self, features):
        """[B, C, H, W] -> standardized [B, N, C]."""
        if tuple(features.shape[-2:]) != self.grid:
            raise ValueError(f"slot encoder expects a {self.grid} grid, got {tuple(features.shape[-2:])}")
        tokens = features.float().flatten(2).transpose(1, 2)
        return (tokens - self.feature_mean) / self.feature_std

    def encode(self, features):
        tokens = self.tokens(features)
        x = self.input_mlp(tokens) + self.input_xy(self.xy)[None]
        slots, attn = self.slot_attention(x)
        mass = attn.sum(-1, keepdim=True)
        weights = attn / mass
        return {"slots": slots, "attn": attn, "tokens": tokens,
                "pooled": torch.einsum("bkn,bnc->bkc", weights, tokens),
                "position": weights @ self.xy, "area": mass.squeeze(-1) / attn.shape[-1]}

    def object_count(self, attn):
        """Soft number of occupied object slots, [B]: a slot is occupied when it wins
        some token outright (max over tokens of its slot-softmax attention)."""
        peak = attn[:, self.object_slots].amax(-1)
        return ((peak - self.count_threshold) * self.count_sharpness).sigmoid().sum(-1)

    def decode(self, slots):
        """slots [B, K, D] -> reconstruction [B, N, target_dim], alpha masks [B, K, N]."""
        out = self.decoder(slots[:, :, None] + self.decoder_pos[None, None])
        alpha = out[..., -1].softmax(1)
        return (alpha[..., None] * out[..., :-1]).sum(1), alpha

    def forward(self, features):
        enc = self.encode(features)
        recon, alpha = self.decode(enc["slots"])
        target = enc["tokens"] @ self.target_proj
        enc.update(reconstruction=recon, alpha=alpha,
                   loss=F.mse_loss(recon.float(), target) * target.shape[-1] / self.target_proj.shape[0])
        return enc


def expand_slot_state(state, num_slots, generator=None):
    """Checkpoint state for a model with more slots: copy the learned queries, add
    new ones drawn at the existing query scale (decoder and attention are K-agnostic)."""
    state = dict(state)
    queries = state["slot_attention.queries"]
    extra = num_slots - len(queries)
    if extra < 0:
        raise ValueError("cannot shrink the number of slots")
    if extra:
        noise = torch.randn(extra, queries.shape[1], generator=generator) * queries.std()
        state["slot_attention.queries"] = torch.cat([queries, noise.to(queries)])
    return state


def load_slot_autoencoder(path, map_location="cpu"):
    """Checkpoint written by src.slots_pretrain: {"config": ..., "state_dict": ...}."""
    payload = torch.load(path, map_location=map_location)
    model = SlotAutoencoder(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    return model
