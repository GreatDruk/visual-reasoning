"""Small team-trained proposal network used only by the historical crop control."""

from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                width,
                width,
                3,
                padding=dilation,
                dilation=dilation,
                groups=width,
                bias=False,
            ),
            nn.BatchNorm2d(width),
            nn.SiLU(),
            nn.Conv2d(width, width, 1, bias=False),
            nn.BatchNorm2d(width),
        )

    def forward(self, x):
        return F.silu(self.layers(x) + x)


class CenterDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 40, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.SiLU(),
            *[Block(40, dilation) for dilation in (1, 2, 3, 1, 2, 1)],
        )
        self.head = nn.Conv2d(40, 20, 1)
        nn.init.constant_(self.head.bias[0], -2.19)

    def forward(self, rgb):
        return self.head(self.features(rgb))
