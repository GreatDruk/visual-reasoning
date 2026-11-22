"""Historical R50 initialization order; used only when retraining its source."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50


class HistoricalR50(nn.Module):
    def __init__(self, weights=None, *, full=False):
        super().__init__()
        backbone = resnet50(weights=None)
        if weights is not None:
            backbone.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.full = full
        if full:
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
            self.lateral3 = nn.Conv2d(1024, 64, 1)
            self.lateral4 = nn.Conv2d(2048, 64, 1)
        self.lateral1 = nn.Conv2d(256, 64, 1)
        self.lateral2 = nn.Conv2d(512, 64, 1)
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 20, 1),
        )
        nn.init.constant_(self.head[-1].bias[0], -2.19)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, pixels):
        high = self.layer1(self.stem((pixels - self.mean) / self.std))
        low = self.layer2(high)
        levels = [self.lateral1(high), self.lateral2(low)]
        if self.full:
            lower = self.layer3(low)
            levels.extend((self.lateral3(lower), self.lateral4(self.layer4(lower))))
        combined = levels[-1]
        for fine in reversed(levels[:-1]):
            combined = fine + F.interpolate(
                combined, size=fine.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.head(combined)
