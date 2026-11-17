"""Image loading and frozen-encoder feature caching.

With a frozen encoder the image features never change, so they are computed once per
(encoder, input size, normalisation, split, flip) and stored as float16 .npy files under
`artefacts/features/<key>/`. Training reads them through np.load(mmap_mode="r"), so the
cache does not have to fit in RAM (a 20x30 ResNet layer3 map is ~1.2 MB per image).
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .models.image_encoders import NORMALIZATION
from .utils import resolve_path

log = logging.getLogger(__name__)


def load_images(image_dir, names):
    """Read JPEGs into one uint8 array [N, 3, H, W] (all images are 160x240)."""
    image_dir = resolve_path(image_dir)
    arrays = []
    for name in names:
        with Image.open(image_dir / name) as image:
            arrays.append(np.asarray(image.convert("RGB")))
    return np.stack(arrays).transpose(0, 3, 1, 2).copy()


def preprocess(images_u8, size, normalize, flip=False):
    """uint8 [B, 3, H, W] tensor -> normalised float tensor resized to `size` (H, W)."""
    x = images_u8.float() / 255.0
    if flip:
        x = x.flip(-1)
    if tuple(x.shape[-2:]) != tuple(size):
        x = F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False)
    mean, std = NORMALIZATION[normalize]
    mean = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def cache_key(encoder_cfg, data_cfg):
    """Identity of a frozen-feature cache: arch, literal weights (None != V1 != V2),
    out layer, input size, normalisation. The `eval-v2_` prefix keeps these correctly
    extracted eval-mode features apart from older caches (whose keys lacked the prefix
    and collapsed the weights identity)."""
    h, w = data_cfg.image_size
    if "model_name" in encoder_cfg:
        norm = "norm" if encoder_cfg.get("norm", True) else "raw"
        return f"clip_{encoder_cfg.model_name}_{encoder_cfg.pretrained}_block{encoder_cfg.out_layer}{norm}_{h}x{w}_{data_cfg.normalize}"
    weights = "none" if encoder_cfg.weights is None else str(encoder_cfg.weights)
    return f"eval-v2_{encoder_cfg.arch}_{weights}_{encoder_cfg.out_layer}_{h}x{w}_{data_cfg.normalize}"


@torch.no_grad()
def cache_features(encoder, image_dir, names, out_path, size, normalize, device, flip=False, batch_size=64,
                   amp=False):
    """Run `encoder` over the images and write a float16 [N, C, h, w] .npy file.

    Returns a read-only memmap of the file. An existing file with a matching names list
    is reused. The encoder must return spatial maps.
    """
    out_path = resolve_path(out_path)
    names_path = out_path.with_suffix(".json")
    if out_path.exists() and names_path.exists() and json.loads(names_path.read_text()) == list(names):
        return np.load(out_path, mmap_mode="r")
    if not len(names) or batch_size < 1:
        raise ValueError("feature extraction requires images and a positive batch size")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoder = encoder.to(device).eval()
    tmp_path = out_path.with_name(out_path.stem + ".tmp.npy")
    out = None
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(load_images, image_dir, names[:batch_size])
        for start in tqdm(range(0, len(names), batch_size), desc=f"encode {out_path.name}"):
            images = pending.result()
            next_start = start + batch_size
            if next_start < len(names):
                pending = pool.submit(load_images, image_dir, names[next_start:next_start + batch_size])
            x = torch.from_numpy(images).to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                feats = encoder(preprocess(x, size, normalize, flip))
            if out is None:
                shape = (len(names), *feats.shape[1:])
                out = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float16, shape=shape)
            out[start : start + len(x)] = feats.half().cpu().numpy()
    out.flush()
    del out
    tmp_path.replace(out_path)
    names_path.write_text(json.dumps(list(names)))
    log.info(f"cached {shape} -> {out_path}")
    return np.load(out_path, mmap_mode="r")
