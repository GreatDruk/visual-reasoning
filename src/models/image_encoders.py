"""Image encoders. Only ImageNet classification backbones from torchvision (and
CLIP image towers) are allowed by the rules; detection/segmentation weights are not."""

import torch
from torch import nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class TorchvisionEncoder(nn.Module):
    """A torchvision classifier truncated at `out_layer`.

    forward(x) returns a feature map [B, C, H, W] when `spatial` is true, otherwise the
    globally averaged vector [B, C]. With `freeze` the backbone is in eval mode and has
    no gradients, except the last `unfreeze_last_n_blocks` children of `out_layer`'s
    stage (ResNet bottlenecks / ConvNeXt blocks).
    """

    def __init__(self, arch, weights, out_layer, spatial=True, freeze=True, unfreeze_last_n_blocks=0):
        super().__init__()
        backbone = getattr(models, arch)(weights=weights)
        self.body = create_feature_extractor(backbone, {out_layer: "out"})
        self.spatial = spatial
        self.freeze = freeze
        self.arch, self.out_layer = arch, out_layer
        # Probe the output channels in eval mode: the dummy forward must not update
        # BatchNorm running stats of a backbone whose features will be cached.
        self.body.eval()
        with torch.no_grad():
            self.out_channels = self.body(torch.zeros(1, 3, 64, 64))["out"].shape[1]
        if freeze:
            for p in self.body.parameters():
                p.requires_grad_(False)
            if unfreeze_last_n_blocks:
                stage = self.body.get_submodule(out_layer)
                for block in list(stage.children())[-unfreeze_last_n_blocks:]:
                    for p in block.parameters():
                        p.requires_grad_(True)
        else:
            self.body.train()
        self.trainable = (not freeze) or unfreeze_last_n_blocks > 0

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            # Keep frozen BatchNorm statistics fixed even while the head trains.
            self.body.eval()
        return self

    def forward(self, x):
        fmap = self.body(x)["out"]
        return fmap if self.spatial else fmap.mean(dim=(2, 3))


class ClipImageEncoder(nn.Module):
    """CLIP image tower via open_clip. open_clip is not installed in this environment
    (no network access to PyPI), so this raises until it is added with `uv add open_clip_torch`."""

    def __init__(self, model_name, pretrained, spatial=False, freeze=True):
        super().__init__()
        try:
            import open_clip  # noqa: F401
        except ImportError as e:
            raise ImportError("ClipImageEncoder needs open_clip_torch: uv add open_clip_torch") from e
        raise NotImplementedError("CLIP encoders are not implemented yet")
