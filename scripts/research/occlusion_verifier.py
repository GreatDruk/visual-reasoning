"""Train a caption-supervised verifier on the union of two image detectors."""
import argparse
import copy
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import logit
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_suite_common import amp, caption_loss, joint, load, setup, table_predictions, tune_batch
from kaggle_suite_crops import crop_features, extract_head, vision_model
from kaggle_suite_detectors import native_forward, projection
from learned_center_fusion import Fusion, N_ATTRS
from native_centers import CenterDetector, HEIGHT, WIDTH, read_rgb
from resnet_centers_ablation import ResNetCenters
from scripts import pixel_reasoning as logic

DATA = Path(os.environ.get("CLEVR_DATA", "data"))
PRIORS = Path(os.environ.get("CLEVR_PRIORS", "priors"))
CACHE = Path("verify_cache")
OUT = Path("verify_result")
BOX = slice(1 + N_ATTRS, 5 + N_ATTRS)
BASE = 5 + N_ATTRS
INPUT_DIM = BASE + 2 * (2 + N_ATTRS) + 3
CROP_DIM = 2 * 128
MAX = 40


def model_head(key, device):
    head = Fusion(load(PRIORS / "scene_silhouette.pt"))
    head.load_state_dict(load(PRIORS / f"{key}_head.pt"))
    return head.to(device).eval()


def detector_outputs(detector, head, images, matrix=None):
    raw, xy, _ = native_forward(detector, images, matrix)
    keep = raw[..., 0] >= logit(.025)
    enc = head.encode(raw, xy, keep)
    obj, logits = head.slot_logits(enc)
    from native_centers import GROUPS
    attributes = torch.cat([logits[..., g.tolist()].log_softmax(-1) for g in GROUPS], -1)
    return torch.cat([raw[..., :1], obj[..., None], attributes, raw[..., BOX], enc["position"]], -1).cpu().numpy()


def merge(left, right):
    # First match all plausible peaks one-to-one. Keep corroborated weak peaks
    # and unmatched peaks above the old detector inclusion threshold.
    left, right = [a[a[:, 0] >= logit(.025)] for a in (left, right)]
    scale = np.array([WIDTH, HEIGHT])
    boxes = [a[:, -6:-2] for a in (left, right)]
    centers = [(b[:, :2] + .5 * b[:, 2:]) * scale for b in boxes]
    dist = np.linalg.norm(centers[0][:, None] - centers[1], axis=-1)
    extent = [(b[:, 2:] * scale).max(1) for b in boxes]
    radius = np.clip(.52 * np.minimum(extent[0][:, None], extent[1]), 6., 13.)
    normalized = dist / radius
    # Dummy columns allow genuine misses without displacing valid matches.
    costs = np.concatenate([np.where(normalized < 1, normalized, 1000),
                            np.full((len(left), len(left)), 2.)], axis=1)
    rows, cols = linear_sum_assignment(costs)
    matched = {i: j for i, j in zip(rows, cols) if j < len(right)}
    taken = set(matched.values())
    candidates = [(l, right[matched[i]], normalized[i, matched[i]]) if i in matched else (l, None, 1.)
                  for i, l in enumerate(left)]
    candidates += [(None, r, 1.) for j, r in enumerate(right) if j not in taken]
    table = []
    for l, r, distance in candidates:
        sources = [a for a in (l, r) if a is not None]
        ref = max(sources, key=lambda a: a[1])
        if len(sources) == 1 and ref[0] < logit(.08):
            continue
        source_features = [a[:2 + N_ATTRS] if a is not None else np.zeros(2 + N_ATTRS) for a in (l, r)]
        x = np.concatenate((ref[1:-2], *source_features, [l is not None, r is not None, distance]))
        table.append((x, ref[-2:]))
    x = np.zeros((MAX, INPUT_DIM), np.float32)
    xy = np.zeros((MAX, 2), np.float32)
    valid = np.arange(MAX) < len(table)
    if table:
        x[valid], xy[valid] = map(np.stack, zip(*table))
    return x, xy, valid


class CropEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.backbone = vision_model().to(device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
        self.pool = copy.deepcopy(extract_head(self.backbone, device))
        self.pool.load_state_dict(load(PRIORS / "siglip_pool.pt"))
        with np.load(PRIORS / "sigft_pca.npz") as data:
            for name in ("mean", "components", "scale"):
                self.register_buffer(name, torch.from_numpy(data[name]).to(device))
        axis = (torch.arange(256, device=device) + .5) / 256
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("grid", torch.stack((xx, yy), -1))

    def forward(self, images, boxes, factor):
        size = boxes[:, None, None, 2:]
        grid = boxes[:, None, None, :2] + size * (.5 + factor * (self.grid - .5))
        crop = F.grid_sample(images, grid * 2 - 1, align_corners=False, padding_mode="border")
        # extract_head moves the original pool to fp32, so run the full forward under AMP.
        with amp(images.device):
            _, tokens = crop_features(self.backbone, 2 * crop - 1)
            embedding = self.pool(tokens.float())
        return (((embedding.float() - self.mean) @ self.components.T) / self.scale).clamp(-7, 7)


@torch.inference_mode()
def extract(split, shard, device, geometry_only=False):
    names = pd.read_csv(DATA / f"{split}.csv", usecols=["image"]).image.unique()
    indices = np.arange(shard, len(names), 2)
    models = [ResNetCenters(full=True), CenterDetector()]
    heads = []
    for key, model in zip(("r50_img", "thaw"), models):
        model.load_state_dict(load(PRIORS / f"{key}.pt"))
        model.to(device, memory_format=torch.channels_last).eval()
        heads.append(model_head(key, device))
    transform = projection().to(device)
    cropper = None if geometry_only else CropEncoder(device).eval()
    batch = None
    parts = []
    started = time.monotonic()
    for start in range(0, len(indices), 32):
        selected = indices[start:start + 32]
        images = np.stack([read_rgb(DATA / split / name) for name in names[selected]])
        images = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float().div_(255)
        r50 = detector_outputs(models[0], heads[0], images)
        native = detector_outputs(models[1], heads[1], images, transform)
        x, xy, valid = map(np.stack, zip(*(merge(l, r) for l, r in zip(r50, native))))
        si, sj = np.nonzero(valid)
        boxes = torch.from_numpy(x[si, sj, BOX]).to(device)
        scene = torch.from_numpy(si).to(device)
        extra = np.zeros((*valid.shape, CROP_DIM), np.float16)
        def encode(i, scale):
            return cropper(images[scene[i]], boxes[i], scale)
        if cropper is not None and len(si) and batch is None:
            sizes = (96, 64, 48, 32, 16, 8) if device.type == "cuda" else (3,)
            batch = tune_batch("crop_verifier", sizes, lambda n: encode(torch.arange(n, device=device) % len(si), 1.), device)
        for begin in range(0, len(si) if cropper is not None else 0, batch or 1):
            sl = slice(begin, begin + batch)
            extra[si[sl], sj[sl]] = torch.cat([encode(sl, factor) for factor in (1., 1.55)], -1).half().cpu().numpy()
        parts.append((x, xy, valid, extra))
        if start % 320 == 0:
            print(split, shard, start + len(selected), "/", len(indices), round((time.monotonic()-started)/60, 1), "min", flush=True)
    dest = CACHE / f"{split}_{shard}.npz"
    with dest.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f, names=names[indices].astype(str), **dict(zip(("x","xy","valid","crops"),
                            (np.concatenate(v) for v in zip(*parts)))))
    dest.with_suffix(".tmp").replace(dest)


def collect(split):
    names = pd.read_csv(DATA / f"{split}.csv", usecols=["image"]).image.unique()
    arrays = {}
    for shard in range(2):
        with np.load(CACHE / f"{split}_{shard}.npz") as data:
            if not np.array_equal(data["names"], names[shard::2]):
                raise ValueError(f"{split}: mismatched image order")
            for key in ("x", "xy", "valid", "crops"):
                v = data[key]
                if key not in arrays:
                    arrays[key] = np.empty((len(names), *v.shape[1:]), dtype=v.dtype)
                arrays[key][shard::2] = v
    # SceneData already implements the caption parser, scene-group batches and
    # exact train supervision. Persist the merged table in its normal format.
    (CACHE / "tables").mkdir(exist_ok=True)
    with (CACHE / "tables" / f"{split}.tmp").open("wb") as f:
        np.savez(f, names=names.astype(str), x=np.concatenate([arrays["x"], arrays["crops"]], -1),
                 xy=arrays["xy"], valid=arrays["valid"])
    (CACHE / "tables" / f"{split}.tmp").replace(CACHE / "tables" / f"{split}.npz")


class Verifier(Fusion):
    def __init__(self, visual):
        super().__init__(load(PRIORS / "scene_silhouette.pt"))
        with torch.no_grad():
            self.relation_offset.zero_()
        self.visual = visual
        self.encoder = nn.Sequential(nn.Linear(INPUT_DIM + (CROP_DIM if visual else 0), 96),
                                     nn.LayerNorm(96), nn.SiLU(), nn.Dropout(.08))
        self.query = nn.Linear(96, 32)
        self.key = nn.Linear(96, 32)
        self.geometry = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 1))
        self.value = nn.Linear(96, 48)
        self.residual = nn.Sequential(nn.Linear(144, 64), nn.SiLU(), nn.Linear(64, 3 + N_ATTRS))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def encode(self, features, positions, valid):
        feat = self.encoder(features if self.visual else features[..., :INPUT_DIM])
        boxes = features[..., BOX]
        size = boxes[..., 2:].clamp_min(.01)
        center = boxes[..., :2] + .5 * size
        delta = (center[:, :, None] - center[:, None]) / size[:, :, None]
        ratio = size.log()[:, :, None] - size.log()[:, None]
        geometry = self.geometry(torch.cat((delta.clamp(-5, 5), ratio), -1)).squeeze(-1)
        scores = self.query(feat) @ self.key(feat).transpose(1, 2) / 32**.5 + geometry
        diag = torch.eye(valid.shape[1], device=valid.device, dtype=torch.bool)
        neighbors = valid[:, None] & ~diag
        attention = scores.masked_fill(~neighbors, -1e4).softmax(-1) * neighbors
        correction = self.residual(torch.cat((feat, attention @ self.value(feat)), -1))
        return {"pooled": features, "position": positions + .08 * correction[..., -2:].tanh(),
                "residual": correction, "valid": valid}

    def slot_logits(self, enc):
        obj, words = super().slot_logits(enc)
        # The old +/-3 limit caps a candidate starting at p=.036 below .43.
        # Give weak proposals room to recover when the train captions support it.
        weak = enc["valid"] & (enc["pooled"][..., 0] < 0)
        obj = obj + weak * 3 * enc["residual"][..., 0].tanh()
        return obj, words


def compact(table):
    # Valid candidates are a packed prefix. Keep count support through ten.
    n = max(10, int(table[2].sum(1).max().item()))
    return [a[:, :n] for a in table]


def train(kind, seed, epochs, device):
    from types import SimpleNamespace
    data = logic.SceneData(DATA, "train", None, device, cache_dir=CACHE / "tables", fitting=True)
    model = Verifier(kind == "visual").to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=.0008)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs, eta_min=.0001)
    rng = np.random.default_rng(seed)
    split = SimpleNamespace(data=data)
    effective = 96

    def probe(n):
        packed = next(data.batches(n, np.random.default_rng(seed)))
        try:
            caption_loss(model, compact(packed[1]), split, packed).backward()
        finally:
            optim.zero_grad(set_to_none=True)

    with torch.random.fork_rng(devices=[0] if device.type == "cuda" else []):
        sizes = tuple(dict.fromkeys(min(n, len(data.rows)) for n in (96, 48, 24, 12)))
        batch = tune_batch(kind, sizes, probe, device)
    for epoch in range(epochs):
        value = []
        seen = 0
        for packed in data.batches(batch, rng):
            count = len(packed[1][0])
            budget = min(effective, len(data.rows) - seen + seen % effective)
            loss = caption_loss(model, compact(packed[1]), split, packed)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite train loss")
            (loss * (count / budget)).backward()
            seen += count
            if seen % effective == 0 or seen == len(data.rows):
                nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optim.step()
                optim.zero_grad(set_to_none=True)
            value.append(loss.detach())
        schedule.step()
        print(kind, seed, epoch+1, float(torch.stack(value).mean()), flush=True)
    dest = OUT / f"{kind}_{seed}.pt"
    torch.save(model.state_dict(), dest.with_suffix(".tmp"))
    dest.with_suffix(".tmp").replace(dest)


def infer(split, kind, seed, device):
    from types import SimpleNamespace
    data = logic.SceneData(DATA, split, None, device, cache_dir=CACHE / "tables")
    model = Verifier(kind == "visual").to(device)
    model.load_state_dict(load(OUT / f"{kind}_{seed}.pt"))
    bank = table_predictions(model, [a.numpy() for a in data.table], device)
    # Map the actual caption-based global twin matches into each scene.
    local = []
    for rows in data.rows:
        partner = np.searchsorted(rows, data.partner[rows])
        first = np.flatnonzero(np.arange(len(rows)) < partner)
        local.append(np.stack((first, partner[first]), axis=1))
    score = joint(SimpleNamespace(data=data, scene_pairs=local), bank, model.relation_offset.detach().cpu().numpy())
    np.savez(OUT / f"{kind}_{seed}_{split}.npz", ids=data.ids, score=score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("extract", "collect", "train", "infer"))
    parser.add_argument("--split", default="train", choices=("train","val","test"))
    parser.add_argument("--shard", type=int, choices=(0,1), default=0)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--kind", choices=("visual","control"), default="visual")
    parser.add_argument("--seed", type=int, default=421)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    _, device = setup(args.seed)
    CACHE.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    if args.action == "extract":
        extract(args.split, args.shard, device, args.geometry_only)
    elif args.action == "collect":
        collect(args.split)
    elif args.action == "train":
        train(args.kind, args.seed, args.epochs, device)
    else:
        infer(args.split, args.kind, args.seed, device)
