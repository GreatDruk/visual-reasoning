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
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)
NORMALIZATION = {"imagenet": (IMAGENET_MEAN, IMAGENET_STD), "clip": (CLIP_MEAN, CLIP_STD), "siglip": (SIGLIP_MEAN, SIGLIP_STD)}


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
    """Frozen image tower of an open_clip dual encoder (CLIP / SigLIP / SigLIP2).

    Only timm-backed towers (SigLIP family) are supported: their ViT trunk exposes
    `forward_intermediates` and `set_input_size`. The position embedding is resampled to
    `image_size` (H, W), so the 3:2 scenes are not squashed into a square. forward(x)
    returns the patch tokens of transformer block `out_layer` as a map [B, C, H/16, W/16]
    (`spatial=True`) or their mean [B, C]. The text tower is discarded.
    """

    def __init__(self, model_name, pretrained, image_size, out_layer=-1, norm=True, spatial=True, freeze=True, cache_dir=None):
        super().__init__()
        import open_clip

        if not freeze:
            raise NotImplementedError("ClipImageEncoder is frozen-only (features are cached)")
        model = open_clip.create_model(model_name, pretrained=pretrained, cache_dir=cache_dir)
        trunk = getattr(model.visual, "trunk", None)
        if trunk is None or not hasattr(trunk, "forward_intermediates"):
            raise ValueError(f"{model_name}: only timm-backed towers (SigLIP) are supported")
        trunk.set_input_size(img_size=tuple(image_size))
        self.trunk = trunk.eval()
        for p in self.trunk.parameters():
            p.requires_grad_(False)
        self.out_layer = out_layer % len(trunk.blocks)
        self.norm = norm
        self.spatial = spatial
        self.freeze = True
        self.trainable = False
        self.out_channels = trunk.embed_dim

    def train(self, mode=True):
        super().train(mode)
        self.trunk.eval()
        return self

    def forward(self, x):
        fmap = self.trunk.forward_intermediates(
            x, indices=[self.out_layer], norm=self.norm, stop_early=True, intermediates_only=True, output_fmt="NCHW"
        )[0]
        return fmap if self.spatial else fmap.mean(dim=(2, 3))
