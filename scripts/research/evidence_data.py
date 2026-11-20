"""Scene-disjoint training pairs and fixed visual features for caption verification."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from src.models.slot_program import WORD_INDEX, parse_program
from src.pixel_regions import read_rgb
from src.text_aug import canonical_form


def read_captions(data, split):
    frame = pd.read_csv(data / f"{split}.csv")
    scene, names = pd.factorize(frame.image)
    caption = [canonical_form(r.caption, r.category) for r in frame.itertuples()]
    programs = [parse_program(c) for c in caption]
    if any(p is None for p in programs):
        examples = [c for c, p in zip(caption, programs) if p is None][:3]
        raise ValueError(f"Unsupported canonical captions in {split}: {examples}")
    words = np.zeros((len(frame), 3, 4), dtype=np.int64)
    meta = np.empty((len(frame), 3), dtype=np.int64)
    for i, p in enumerate(programs):
        for j, filt in enumerate((p.a, p.b, p.ref)):
            words[i, j, :len(filt)] = [WORD_INDEX[w] for w in filt]
        meta[i] = p.op, p.relation, p.number
    return frame, names.to_numpy(dtype=str), scene, caption, words, meta


def labeled_pairs(frame):
    pair = frame.reset_index().pivot(index=["image", "pair_id"], columns="label", values="index")
    if len(pair) * 2 != len(frame) or pair[[1, 0]].isna().any().any():
        raise ValueError("Every labeled pair must contain a true and a false caption")
    return pair[[1, 0]].to_numpy(dtype=np.int64)


def opposite_rows(captions, labels, scenes, allowed, seed):
    """Find the identical program with the opposite label on another allowed scene."""
    pools = defaultdict(list)
    for i in np.flatnonzero(allowed):
        pools[captions[i], labels[i]].append(i)
    rng = np.random.default_rng(seed)
    opposite = np.full(len(labels), -1, dtype=np.int64)
    for i in np.flatnonzero(allowed):
        pool = pools[captions[i], 1 - labels[i]]
        if not pool:
            continue
        start = rng.integers(len(pool))
        for k in range(len(pool)):
            j = pool[(start + k) % len(pool)]
            if scenes[i] != scenes[j]:
                opposite[i] = j
                break
    return opposite


def scene_folds(scene_count, count, seed):
    order = np.random.default_rng(seed).permutation(scene_count)
    folds = np.empty(scene_count, dtype=np.int64)
    folds[order] = np.arange(scene_count) % count
    return folds


def decode_images(data, split, names, workers=2):
    with ThreadPoolExecutor(workers) as pool:
        paths = (data / split / name for name in names)
        return np.stack(list(pool.map(read_rgb, paths)))


def backbone(weights, device):
    from torchvision.models import resnet50

    net = resnet50(weights=None)
    net.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    return net.to(device, memory_format=torch.channels_last).eval()


def reduce_channels(features, channels):
    batch, width, height, length = features.shape
    return features.reshape(batch, channels, width // channels, height, length).mean(2)


def maps(resnet, siglip, images):
    rgb = images.permute(0, 3, 1, 2).float().div(255)
    image_net = (rgb - rgb.new_tensor([.485, .456, .406])[None, :, None, None])
    image_net = image_net / rgb.new_tensor([.229, .224, .225])[None, :, None, None]
    early = resnet.relu(resnet.bn1(resnet.conv1(image_net.contiguous(memory_format=torch.channels_last))))
    detail = resnet.layer1(resnet.maxpool(early))
    context = resnet.layer2(detail)
    # Preserve the stride-four map with both mean appearance and max response.
    detail = reduce_channels(detail, 32)
    fine = torch.cat((F.avg_pool2d(detail, 2), F.max_pool2d(detail, 2),
                      reduce_channels(context, 32), F.avg_pool2d(rgb, 8)), dim=1)
    pixel_size = siglip.config.image_size
    pixel_values = F.interpolate(rgb, (pixel_size, pixel_size), mode="bilinear", align_corners=False)
    hidden = siglip(pixel_values=2 * pixel_values - 1).last_hidden_state
    side = int(hidden.shape[1] ** .5)
    patch = hidden.transpose(1, 2).reshape(len(rgb), -1, side, side)
    patch = F.adaptive_avg_pool2d(reduce_channels(patch, 144), (6, 8))
    return fine.permute(0, 2, 3, 1), patch.permute(0, 2, 3, 1)


def extract(data, split, target, init, vision, device, shard, choose_batch, limit=None):
    from transformers import SiglipVisionModel

    _, names, *_ = read_captions(data, split)
    if limit:
        names = names[:limit]
    names = names[shard::2]
    started = time.perf_counter()
    pixels = decode_images(data, split, names)
    decode_s = time.perf_counter() - started
    started = time.perf_counter()
    images = torch.from_numpy(pixels).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    upload_s = time.perf_counter() - started
    del pixels
    r50 = backbone(init, device)
    siglip = SiglipVisionModel.from_pretrained(vision, local_files_only=True).to(
        device, dtype=torch.float16 if device.type == "cuda" else torch.float32).eval()

    @torch.inference_mode()
    def compute(indices):
        with torch.autocast(device.type, enabled=device.type == "cuda", dtype=torch.float16):
            return maps(r50, siglip, images[indices])

    sizes = (32, 64, 128, 256, 384, 512)
    def transfer(indices):
        return [x.half().cpu().numpy() for x in compute(indices)]

    batch, probes = choose_batch(device, sizes, len(images), transfer)
    start = time.perf_counter()
    prefix = target / f"{split}_{shard}"
    np.save(prefix.with_name(prefix.name + "_names.npy"), names)
    cache = None
    for i in range(0, len(images), batch):
        fields = transfer(slice(i, i + batch))
        if cache is None:
            cache = [np.lib.format.open_memmap(prefix.with_name(prefix.name + f"_{field}.npy"),
                     mode="w+", dtype=np.float16, shape=(len(names), *array.shape[1:]))
                     for field, array in zip(("detail", "semantics"), fields)]
        for output, value in zip(cache, fields):
            output[i:i + len(value)] = value
    for output in cache:
        output.flush()
    return dict(decode_cpu_s=decode_s, upload_s=upload_s, encode_write_s=time.perf_counter() - start,
                scenes=len(names), batch=batch, probes=probes)


def load_maps(cache, split, names, device):
    result = []
    for field in ("detail", "semantics"):
        shards = []
        for shard in range(2):
            prefix = cache / f"{split}_{shard}"
            shard_names = np.load(prefix.with_name(prefix.name + "_names.npy"))
            np.testing.assert_array_equal(shard_names, names[shard::2])
            shards.append(np.load(prefix.with_name(prefix.name + f"_{field}.npy")))
        combined = np.empty((len(names), *shards[0].shape[1:]), dtype=np.float16)
        combined[::2], combined[1::2] = shards
        result.append(torch.from_numpy(combined).to(device))
    return result
