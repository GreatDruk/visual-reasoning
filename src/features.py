"""Image loading and frozen-encoder feature caching.

With a frozen encoder the image features never change, so they are computed once per
(encoder, input size, normalisation, split, flip) and stored as float16 .npy files under
`artefacts/features/<key>/`. Training reads them through np.load(mmap_mode="r"), so the
cache does not have to fit in RAM (a 20x30 ResNet layer3 map is ~1.2 MB per image).
"""

import json
import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .models.image_encoders import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD
from .utils import resolve_path

log = logging.getLogger(__name__)


def load_images(image_dir, names):
    """Read JPEGs into one uint8 array [N, 3, H, W] (all images are 160x240)."""
    image_dir = resolve_path(image_dir)
    arrays = [np.asarray(Image.open(image_dir / n).convert("RGB")) for n in tqdm(names, desc=f"load {image_dir.name}")]
    return np.stack(arrays).transpose(0, 3, 1, 2).copy()


def preprocess(images_u8, size, normalize, flip=False):
    """uint8 [B, 3, H, W] tensor -> normalised float tensor resized to `size` (H, W)."""
    x = images_u8.float() / 255.0
    if flip:
        x = x.flip(-1)
    if tuple(x.shape[-2:]) != tuple(size):
        x = F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False)
    mean, std = (CLIP_MEAN, CLIP_STD) if normalize == "clip" else (IMAGENET_MEAN, IMAGENET_STD)
    mean = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def cache_key(encoder_cfg, data_cfg):
    h, w = data_cfg.image_size
    weights = "" if encoder_cfg.weights in ("IMAGENET1K_V1", None) or encoder_cfg.arch == "resnet18" else f"_{encoder_cfg.weights.lower()}"
    return f"{encoder_cfg.arch}{weights}_{encoder_cfg.out_layer}_{h}x{w}_{data_cfg.normalize}"


@torch.no_grad()
def cache_features(encoder, image_dir, names, out_path, size, normalize, device, flip=False, batch_size=64):
    """Run `encoder` over the images and write a float16 [N, C, h, w] .npy file.

    Returns a read-only memmap of the file. An existing file with a matching names list
    is reused. The encoder must return spatial maps.
    """
    out_path = resolve_path(out_path)
    names_path = out_path.with_suffix(".json")
    if out_path.exists() and names_path.exists() and json.loads(names_path.read_text()) == list(names):
        return np.load(out_path, mmap_mode="r")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    images = load_images(image_dir, names)
    encoder = encoder.to(device).eval()
    first = encoder(preprocess(torch.from_numpy(images[:1]).to(device), size, normalize, flip))
    shape = (len(names), *first.shape[1:])
    tmp_path = out_path.with_name(out_path.stem + ".tmp.npy")
    out = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float16, shape=shape)
    for start in tqdm(range(0, len(names), batch_size), desc=f"encode {out_path.name}"):
        x = torch.from_numpy(images[start : start + batch_size]).to(device)
        feats = encoder(preprocess(x, size, normalize, flip))
        out[start : start + len(x)] = feats.half().cpu().numpy()
    out.flush()
    del out
    tmp_path.rename(out_path)
    names_path.write_text(json.dumps(list(names)))
    log.info(f"cached {shape} -> {out_path}")
    return np.load(out_path, mmap_mode="r")
