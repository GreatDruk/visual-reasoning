"""Classification backbones with task-trained center and attribute readouts."""

import torch
from torch import nn
from torch.nn import functional as F

from .heads import GROUPS, N_ATTRS


class Readout(nn.Module):
    def __init__(self, width, separate=False):
        super().__init__()

        def branch(n):
            return nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.SiLU(),
                nn.Conv2d(width, n, 1),
            )

        self.parts = nn.ModuleList(
            [branch(n) for n in ((1, 4, N_ATTRS) if separate else (5 + N_ATTRS,))]
        )
        nn.init.constant_(self.parts[0][-1].bias[0], -2.19)

    def forward(self, x):
        return torch.cat([b(x) for b in self.parts], 1)


class Detector(nn.Module):
    def __init__(self, settings, weights=None):
        super().__init__()
        from torchvision import models

        self.settings = settings
        self.neck = settings["neck"]
        self.is_resnet = settings["backbone"] == "resnet"
        width = settings["width"]
        if self.is_resnet:
            net = models.resnet50(weights=None)
            if weights is not None:
                net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = (
                net.layer1,
                net.layer2,
                net.layer3,
                net.layer4,
            )
            channels = (256, 512, 1024, 2048)
        else:
            net = models.convnext_tiny(weights=None)
            if weights is not None:
                net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            self.trunk = net.features
            channels = (96, 192, 384, 768)
        for i, c in enumerate(channels, 1):
            setattr(self, f"lateral{i}", nn.Conv2d(c, width, 1))
        # Keep the control's exact module layout for checkpoint replay.
        if not settings["separate"]:
            self.head = nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.SiLU(),
                nn.Conv2d(width, 5 + N_ATTRS, 1),
            )
            nn.init.constant_(self.head[-1].bias[0], -2.19)
        else:
            self.head = Readout(width, True)
        if self.neck == "bifpn":
            self.fusion_weights = nn.Parameter(torch.ones(2, 3, 2))
            self.smooth = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(width, width, 3, padding=1, groups=width),
                        nn.Conv2d(width, width, 1),
                        nn.SiLU(),
                    )
                    for _ in range(6)
                ]
            )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def pyramid(self, pixels):
        x = (pixels - self.mean) / self.std
        if self.is_resnet:
            a = self.layer1(self.stem(x))
            b = self.layer2(a)
            c = self.layer3(b)
            d = self.layer4(c)
            levels = [a, b, c, d]
        else:
            levels = []
            for i, layer in enumerate(self.trunk):
                x = layer(x)
                if i in (1, 3, 5, 7):
                    levels.append(x)
        levels = [getattr(self, f"lateral{i}")(x) for i, x in enumerate(levels, 1)]
        if self.neck == "bifpn":
            w = F.relu(self.fusion_weights)
            w = w / (w.sum(-1, keepdim=True) + 1e-4)
            for i in (2, 1, 0):
                up = F.interpolate(
                    levels[i + 1],
                    size=levels[i].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                levels[i] = self.smooth[i](w[0, i, 0] * levels[i] + w[0, i, 1] * up)
            for i in (1, 2, 3):
                down = F.adaptive_avg_pool2d(levels[i - 1], levels[i].shape[-2:])
                levels[i] = self.smooth[i + 2](w[1, i - 1, 0] * levels[i] + w[1, i - 1, 1] * down)
            return levels
        for i in (2, 1, 0):
            levels[i] = levels[i] + F.interpolate(
                levels[i + 1],
                size=levels[i].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return levels

    def forward(self, pixels):
        levels = self.pyramid(pixels)
        return [self.head(x) for x in (levels[:3] if self.neck == "multi" else levels[:1])]


def dense_table(raw, height=160, width=240, stride=4, topk=20, latent=None, min_logit=-2.442347):
    heat = raw[:, 0]
    peak = heat == F.max_pool2d(heat[:, None], 3, 1, 1)[:, 0]
    k = min(topk, heat.shape[-1] * heat.shape[-2])
    _, idx = heat.flatten(1).masked_fill(~peak.flatten(1), -torch.inf).topk(k)
    fields = raw.flatten(2).gather(2, idx[:, None].expand(-1, raw.shape[1], -1)).transpose(1, 2)
    ys, xs = idx.div(raw.shape[-1], rounding_mode="floor"), idx % raw.shape[-1]
    mid = (fields[..., 1:3].clamp(-0.5, 1.5) + torch.stack((xs, ys), -1)) * stride
    extent = fields[..., 3:5].clamp(-2, 1.6).exp() * 24
    norm = fields.new_tensor([width, height])
    box = torch.cat(((mid - 0.5 * extent) / norm, extent / norm), -1)
    xy = (mid + extent * fields.new_tensor([0.0, 0.38])) / norm * 2 - 1
    attrs = [t.log_softmax(-1) for t in fields[..., 5:].split([len(g) for g in GROUPS], -1)]
    extra = fields.new_zeros((*idx.shape, 3 + N_ATTRS))
    if latent is not None:
        extra = (
            latent.flatten(2)
            .gather(2, idx[:, None].expand(-1, latent.shape[1], -1))
            .transpose(1, 2)
        )
    x = torch.cat((fields[..., :1], *attrs, box, extra), -1)
    valid = (fields[..., 0] >= min_logit) & peak.flatten(1).gather(1, idx)
    return x, xy, valid


def tables(outputs, shape, topk=20, min_logit=-2.442347):
    parts = [
        dense_table(raw.float(), *shape, stride=4 * 2**i, topk=topk, min_logit=min_logit)
        for i, raw in enumerate(outputs)
    ]
    if len(parts) == 1:
        return parts[0]
    x, xy, valid = (torch.cat(v, dim=1) for v in zip(*parts))
    # Class-agnostic, center/scale based duplicate suppression across levels only.
    # A single-level control keeps its historical local-max decoder unchanged.
    boxes = x[..., 1 + N_ATTRS : 5 + N_ATTRS]
    scale = x.new_tensor([shape[1], shape[0]])
    centers = (boxes[..., :2] + 0.5 * boxes[..., 2:]) * scale
    extent = (boxes[..., 2:] * scale).amax(-1)
    radius = (0.25 * torch.minimum(extent[:, :, None], extent[:, None, :])).clamp_min(3)
    level = torch.cat(
        [torch.full((p[0].shape[1],), i, device=x.device) for i, p in enumerate(parts)]
    )
    duplicate = (torch.cdist(centers, centers) < radius) & (level[:, None] != level[None, :])
    available = valid.clone()
    batch = torch.arange(len(x), device=x.device)
    outx = []
    outxy = []
    outvalid = []
    # Batched greedy selection avoids a CPU/GPU synchronization per candidate.
    for _ in range(topk):
        present = available.any(-1)
        selected = x[..., 0].masked_fill(~available, -torch.inf).argmax(-1)
        outx.append(x[batch, selected])
        outxy.append(xy[batch, selected])
        outvalid.append(present)
        blocked = duplicate[batch, selected] | F.one_hot(selected, x.shape[1]).bool()
        available = available & ~blocked
    return torch.stack(outx, 1), torch.stack(outxy, 1), torch.stack(outvalid, 1)
