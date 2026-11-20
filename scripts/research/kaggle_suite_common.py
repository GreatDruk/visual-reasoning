"""Common data, training and scoring for the complete Kaggle run."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

from scripts import pixel_reasoning as logic
from src.pixel_regions import HEIGHT, WIDTH, extract_split, fit_background, read_rgb
from src.region_shape_classifier import RegionShapeClassifier
from src.scene_posterior import sample_scene_scores
from src.twins import find_twins
from native_centers import GROUPS, STRIDE
from learned_center_fusion import FEATURES, N_ATTRS

DATA = Path(os.environ.get("CLEVR_DATA", "data"))
INPUT = Path(os.environ.get("CLEVR_WEIGHTS", "weights_in"))
CACHE = Path("cache")
OUTPUT = Path("result")
WEIGHTS = OUTPUT / "weights"
REPORT = OUTPUT / "reports"
SCORES = OUTPUT / "scores"
CSV = OUTPUT / "csv"
FIELDS = ("size", "color", "material", "shape")
COUNT_TYPES = ("counting", "comparison", "spatial_counting")


def say(task, message):
    print(f"[{task}] {message}", flush=True)


def tick(task, done, total, start):
    minutes = (time.monotonic() - start) / 60
    say(task, f"{done}/{total} | {minutes:.1f} min | ETA {minutes * (total / done - 1):.1f} min")


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def load(path):
    return torch.load(path, weights_only=True, map_location="cpu")


def save(model, name):
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, WEIGHTS / f"{name}.pt")


def setup(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    return np.random.default_rng(seed), torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def amp(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


def memory(device):
    if device.type == "cuda":
        peaks = [f"{torch.cuda.max_memory_allocated(i)/2**30:.2f}" for i in range(torch.cuda.device_count())]
        return f"GPU peak {', '.join(peaks)} GiB"
    return "CPU"


def reset_memory(device):
    if device.type == "cuda":
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def tune_batch(task, sizes, probe, device):
    """Probe a backward or inference pass before optimizer updates."""
    for batch in sizes:
        reset_memory(device)
        failed = False
        try:
            probe(batch)
            if device.type == "cuda":
                for i in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(i)
        except torch.cuda.OutOfMemoryError:
            failed = True
        near_limit = device.type == "cuda" and any(
            torch.cuda.max_memory_allocated(i) > .87 * torch.cuda.get_device_properties(i).total_memory
            for i in range(torch.cuda.device_count())
        )
        if not failed:
            if not near_limit or batch == sizes[-1]:
                say(task, f"batch {batch} | {memory(device)}")
                return batch
        torch.cuda.empty_cache()
        error = "OOM" if failed else "Insufficient memory headroom"
        say(task, f"{error} at batch {batch}; trying smaller batch")
    raise RuntimeError(f"{task}: smallest batch exhausted GPU memory")


class Split:
    def __init__(self, split, device):
        self.name = split
        self.data = logic.SceneData(DATA, split, None, device, cache_dir=CACHE / "sil")
        self.names = list(dict.fromkeys(row["image"] for row in logic.read_rows(DATA, split)))
        self.rgb = np.load(CACHE / f"{split}_rgb.npy", mmap_mode="r")
        self.scene_pairs = [
            find_twins(np.zeros(len(rows)), ["" for _ in rows], self.data.categories[rows])
            for rows in self.data.rows
        ]

    def images(self, indices, device):
        rgb = torch.from_numpy(np.asarray(self.rgb[indices]).copy()).to(device)
        return rgb.permute(0, 3, 1, 2).float().div_(255).contiguous(memory_format=torch.channels_last)

    def batches(self, n, rng=None):
        for packed in self.data.batches(n, rng):
            idx = packed[0]
            scenes = self.data.images[idx][np.r_[True, np.diff(self.data.images[idx]) != 0]]
            yield scenes, packed


def prepare(split, variant):
    """The stage scheduler calls this for test only after all fitting is over."""
    folder = CACHE / variant
    folder.mkdir(exist_ok=True)
    names = list(dict.fromkeys(row["image"] for row in logic.read_rows(DATA, split)))
    if variant == "sil":
        rgb_path = CACHE / f"{split}_rgb.npy"
        with ThreadPoolExecutor(2) as pool:
            rgb = list(pool.map(lambda name: read_rgb(DATA / split / name), names))
        np.save(rgb_path, np.stack(rgb))
        del rgb
    if split == "train":
        background = fit_background(DATA / split, names)
    else:
        background = np.load(CACHE / "background.npy")
    if split == "train" and variant == "sil":
        np.save(CACHE / "background.npy", background)
    x, xy, valid = extract_split(DATA / split, names, background, workers=2, boundary=True,
                                 primitives=variant == "sil", fragments=variant == "seam",
                                 seams=variant == "seam")
    if variant != "sil":
        classifier = RegionShapeClassifier()
        if split == "train":
            np.savez(folder / "train.npz", names=names, x=x, xy=xy, valid=valid)
            fitting = logic.SceneData(DATA, split, None, torch.device("cpu"), cache_dir=folder, fitting=True)
            anchors = logic.region_anchors(x, xy, valid, fitting.parsed, fitting.images, fitting.label)
            x, xy, valid = classifier.augment(DATA / split, names, (x, xy, valid), anchors)
            classifier.classifier.save_model(str(WEIGHTS / f"shape_{variant}.cbm"))
        else:
            classifier.classifier.load_model(str(WEIGHTS / f"shape_{variant}.cbm"))
            x, xy, valid = classifier.augment(DATA / split, names, (x, xy, valid))
    np.savez(folder / f"{split}.npz", names=names, x=x, xy=xy, valid=valid)
    say(f"prepare/{variant}", f"{split} | {len(names)} scenes | {int(valid.sum())} regions")


def fixed_table(split, variant, device):
    path = {"sil": "scene_silhouette", "seam": "scene_seams", "counts": "scene_counts"}[variant]
    model = logic.ImageTable(boundary=True, primitives=True).to(device)
    model.load_state_dict(load(INPUT / f"{path}.pt"))
    data = logic.SceneData(DATA, split, None, device, cache_dir=CACHE / variant)
    bank = table_predictions(model, tuple(t.numpy() for t in data.table), device)
    return bank, data.table[0].numpy(), model.relation_offset.detach().cpu().numpy()


@torch.inference_mode()
def table_predictions(model, data, device):
    model.eval()
    result = []
    for start in range(0, len(data[0]), 256):
        x, xy, valid = [torch.from_numpy(z[start:start + 256]).to(device) for z in data]
        encoded = model.encode(x, xy, valid)
        obj, attrs = model.slot_logits(encoded)
        groups = [attrs[..., g.tolist()].softmax(-1) for g in GROUPS]
        result.append([z.cpu().numpy() for z in (obj.sigmoid(), encoded["position"], valid, *groups)])
    return tuple(np.concatenate(items) for items in zip(*result))


def dense_bank(raw, latent=None):
    """Convert stride-four heatmaps to a fixed table, preserving gradients on readouts."""
    heat = raw[:, 0]
    peak = heat == F.max_pool2d(heat[:, None], 3, 1, 1)[:, 0]
    _, index = heat.flatten(1).masked_fill(~peak.flatten(1), -torch.inf).topk(20)
    fields = raw.flatten(2).gather(2, index[:, None].expand(-1, raw.shape[1], -1)).transpose(1, 2)
    ys, xs = index.div(raw.shape[-1], rounding_mode="floor"), index % raw.shape[-1]
    mid = (fields[..., 1:3].clamp(-.5, 1.5) + torch.stack((xs, ys), -1)) * STRIDE
    extent = fields[..., 3:5].clamp(-2, 1.6).exp() * 24
    norm = fields.new_tensor([WIDTH, HEIGHT])
    box = torch.cat(((mid - .5 * extent) / norm, extent / norm), -1)
    pos = (mid + extent * fields.new_tensor([0., .38])) / norm * 2 - 1
    attrs = [t.log_softmax(-1) for t in fields[..., 5:].split([len(g) for g in GROUPS], -1)]
    extra = fields.new_zeros((*index.shape, 3 + N_ATTRS))
    if latent is not None:
        extra = latent.flatten(2).gather(2, index[:, None].expand(-1, latent.shape[1], -1)).transpose(1, 2)
    x = torch.cat((fields[..., :1], *attrs, box, extra), -1)
    valid = (fields[..., 0] >= -2.442347) & peak.flatten(1).gather(1, index)
    return x, pos, valid


def dense_probabilities(table):
    x, xy, ok = table
    p = 1 / (1 + np.exp(-np.clip(x[..., 0], -40, 40)))
    attr = [np.exp(z) for z in np.split(x[..., 1:1 + N_ATTRS], np.cumsum([len(g) for g in GROUPS])[:-1], -1)]
    return p, xy, ok, *attr


def metrics(split, values):
    label = split.data.label
    categories = split.data.categories
    by_category = {}
    for category in np.unique(categories):
        use = categories == category
        by_category[category] = {
            "auc": float(roc_auc_score(label[use], values[use])),
            "accuracy": float(((values[use] > 0) == label[use]).mean()),
            "wrong_pairs": int(((values[use] < 0) & (label[use] == 1)).sum()),
        }
    auc = float(roc_auc_score(label, values))
    return {"auc": auc, "at_least_999": auc >= .999,
            "accuracy": float(((values > 0) == label).mean()),
            "wrong_pairs": int(((values < 0) & (label == 1)).sum()), "categories": by_category}


def record(name, split, values):
    if values.shape != split.data.ids.shape or not np.isfinite(values).all():
        raise ValueError(f"{name}/{split.name}: invalid scores")
    np.savez(SCORES / f"{name}_{split.name}.npz", ids=split.data.ids, score=values)
    if split.name == "val":
        result = metrics(split, values)
        write_json(REPORT / f"{name}.json", result)
        say(name, f"val AUC {result['auc']:.8f} | {result['wrong_pairs']} wrong pairs")


def joint(split, bank, offsets=None, draws=1024, seed=42):
    p, xy, valid, *attrs = bank
    offsets = np.zeros(7) if offsets is None else offsets
    values = np.zeros(len(split.data.ids), np.float64)
    for k, (rows, pairs) in enumerate(zip(split.data.rows, split.scene_pairs)):
        mask = valid[k]
        values[rows] = sample_scene_scores(
            [split.data.parsed[j] for j in rows], pairs, p[k, mask],
            [(g, prob[k, mask]) for g, prob in zip(GROUPS, attrs)], xy[k, mask], offsets, seed, draws,
        )
    return values


def caption_loss(head, table, split, packed):
    idx, _, scenes, program, pairs, aux, _ = packed
    target = torch.as_tensor(split.data.label[idx], device=scenes.device, dtype=torch.float32)
    weight = 1 + .7 * torch.as_tensor(np.isin(split.data.categories[idx], COUNT_TYPES), device=scenes.device)
    raw = head.score(*table, scenes, program)
    a, b = pairs.T
    loss = F.binary_cross_entropy_with_logits(raw, target, weight=weight)
    loss += F.binary_cross_entropy_with_logits(raw[a] - raw[b], target[a], weight=weight[a])
    if aux is not None:
        loss += .3 * head.existence_loss(*table, *aux)
    return loss


def configure_split(split, device, fitting=False):
    result = Split(split, device)
    if fitting:
        result.data = logic.SceneData(DATA, split, None, device, cache_dir=CACHE / "sil", fitting=True)
    return result
