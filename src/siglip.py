"""Crop geometry and explicit frozen, tail, or adapter SigLIP adaptation."""

import copy
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from .heads import N_ATTRS


def crop_images(images, boxes, size=256, mode="stretch", context=1.35):
    """Boxes are normalized xywh; letterbox pads only inside the selected crop."""
    axis = (torch.arange(size, device=images.device, dtype=torch.float32) + 0.5) / size
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    local = torch.stack((xx, yy), -1)[None].expand(len(boxes), -1, -1, -1)
    if mode == "context":
        local = 0.5 + context * (local - 0.5)
    mask = None
    if mode == "letterbox":
        wh = boxes[:, 2:] * boxes.new_tensor([images.shape[-1], images.shape[-2]])
        fraction = wh / wh.amax(-1, keepdim=True).clamp_min(1e-6)
        local = (local - 0.5) / fraction[:, None, None, :].clamp_min(1e-6) + 0.5
        mask = ((local >= 0) & (local <= 1)).all(-1)
    grid = boxes[:, None, None, :2] + local * boxes[:, None, None, 2:]
    pixels = F.grid_sample(images, 2 * grid - 1, align_corners=False, padding_mode="border")
    if mask is not None:
        pixels = torch.where(mask[:, None], pixels, torch.full_like(pixels, 0.5))
    return pixels


class Adapter(nn.Module):
    def __init__(self, width, bottleneck=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, width),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + 0.1 * self.net(x)


class Vision(nn.Module):
    def __init__(self, folder, settings, model=None):
        super().__init__()
        if model is None:
            from transformers import SiglipVisionModel

            model = SiglipVisionModel.from_pretrained(
                folder, local_files_only=True, attn_implementation="sdpa"
            )
        self.backbone = model
        self.core = getattr(model, "vision_model", model)
        self.settings = settings
        self.pool = copy.deepcopy(self.core.head).float()
        self.classifier = nn.Linear(model.config.hidden_size, N_ATTRS)
        self.adapters = (
            nn.ModuleList(
                [Adapter(model.config.hidden_size) for _ in range(settings["tail_blocks"])]
            )
            if settings["mode"] == "adapter"
            else nn.ModuleList()
        )
        self.backbone.requires_grad_(False)
        self.adapting = False
        if settings["mode"] == "tail":
            for layer in list(self.core.encoder.layers)[-settings["tail_blocks"] :]:
                layer.requires_grad_(True)
            self.core.post_layernorm.requires_grad_(True)

    @staticmethod
    def layer(layer, x):
        out = layer(x, attention_mask=None)
        return out[0] if isinstance(out, tuple) else out

    def tokens(self, pixels):
        layers = self.core.encoder.layers
        split = len(layers) - self.settings["tail_blocks"]
        hidden_intermediate = None
        with torch.no_grad():
            x = self.core.embeddings(2 * pixels - 1)
            for i, layer in enumerate(layers[:split]):
                x = self.layer(layer, x)
                if i + 1 == self.settings["intermediate"]:
                    hidden_intermediate = x
        # no_grad produces ordinary detached tensors, suitable for the trainable tail.
        context = (
            nullcontext()
            if self.adapting and self.settings["mode"] != "frozen"
            else torch.no_grad()
        )
        with context:
            for i, layer in enumerate(layers[split:]):
                x = self.layer(layer, x)
                if self.adapters:
                    x = self.adapters[i](x)
            x = self.core.post_layernorm(x)
        if hidden_intermediate is not None:
            hidden_intermediate = self.core.post_layernorm(hidden_intermediate)
        return x, hidden_intermediate

    def reduce(self, tokens):
        side = int(tokens.shape[1] ** 0.5)
        grid = self.settings["pool_grid"]
        if side == grid:
            return tokens
        return (
            F.adaptive_avg_pool2d(
                tokens.transpose(1, 2).reshape(len(tokens), -1, side, side),
                (grid, grid),
            )
            .flatten(2)
            .transpose(1, 2)
        )

    def embedding(self, pixels):
        tokens, intermediate = self.tokens(pixels)
        embedding = self.pool(self.reduce(tokens).float())
        if intermediate is not None:
            embedding = torch.cat((embedding, self.pool(self.reduce(intermediate).float())), -1)
        return embedding

    def adaptation_state(self):
        return {name: t.detach().cpu() for name, t in self.named_parameters() if t.requires_grad}


def attribute_loss(scores, labels):
    result = scores.sum() * 0
    count = labels.ge(0).sum().clamp_min(1)
    for i, group in enumerate(scores.float().split((2, 8, 2, 3), -1)):
        result += F.cross_entropy(group, labels[:, i], ignore_index=-100, reduction="sum")
    return result / count
