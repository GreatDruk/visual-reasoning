"""Train-only donor visibility and GPU counterfactual insertion."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from PIL import Image
from torch.nn import functional as F

from .runtime import Progress


def donor_arrays(root, manifest, scenes):
    if scenes.split != "train":
        raise ValueError("Donors may only be read for train")
    name = manifest.get("assets", {}).get("donors")
    if name not in manifest["files"]:
        raise ValueError("Attach a registered train-donor asset")
    with np.load(root / name, allow_pickle=False) as z:
        names = z["names"].tolist()
        expected = pd.read_csv(scenes.root / "train.csv", usecols=["image"]).image.unique()
        if set(names) != set(expected):
            raise ValueError("Donor asset must contain exactly the train scenes")
        lookup = {name: i for i, name in enumerate(names)}
        if len(lookup) != len(names) or not set(scenes.names) <= lookup.keys():
            raise ValueError("Donor scene provenance mismatch")
        idx = np.array([lookup[name] for name in scenes.names])
        result = {k: z[k][idx].copy() for k in ["keep", "box", "mask", "labels"]}
    return result, names, manifest["files"][name]


def foreground_cache(arrays, scenes, digest, scratch):
    root = Path(scratch).resolve()
    if "artefacts" in root.parts:
        raise ValueError("Counter cache must be scratch")
    key = hashlib.sha256(
        (
            digest
            + "\n".join(scenes.names)
            + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ).encode()
    ).hexdigest()[:24]
    path = root / "counter" / f"{key}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        if not path.exists():
            pending = path.with_suffix(".tmp.npy")
            bank = np.lib.format.open_memmap(
                pending, mode="w+", dtype=bool, shape=(len(scenes.names), 160, 240)
            )
            bank[:] = False
            progress = Progress("train/foreground", len(scenes.names))
            for i in range(len(scenes.names)):
                for j in np.flatnonzero(arrays["keep"][i]):
                    x, y, w, h = map(int, arrays["box"][i, j])
                    if x < 0 or y < 0 or w < 1 or h < 1 or x + w > 240 or y + h > 160:
                        raise ValueError("Donor box outside image")
                    mask = (
                        np.asarray(
                            Image.fromarray(arrays["mask"][i, j]).resize(
                                (w, h), Image.Resampling.BILINEAR
                            )
                        )
                        > 127
                    )
                    bank[i, y : y + h, x : x + w] |= mask
                progress.update(i + 1)
            bank.flush()
            bank._mmap.close()
            pending.replace(path)
    return path


class CounterData:
    def __init__(self, root, manifest, scenes, scratch, device):
        arrays, self.all_names, digest = donor_arrays(root, manifest, scenes)
        path = foreground_cache(arrays, scenes, digest, scratch)
        scenes.prepare_images(Path(scratch) / "images", device)
        self.rgb = scenes._rgb.permute(0, 3, 1, 2)
        values = np.load(path, mmap_mode="r")
        self.foreground = torch.empty(values.shape, dtype=torch.bool, device=device)
        for first in range(0, len(values), 256):
            self.foreground[first : first + 256].copy_(
                torch.from_numpy(np.array(values[first : first + 256], copy=True)).to(device)
            )
        self.box = torch.as_tensor(arrays["box"].astype("float32"), device=device)
        self.mask = torch.as_tensor(arrays["mask"], device=device)
        self.labels = torch.as_tensor(arrays["labels"].astype("int64"), device=device)
        self.keep = torch.as_tensor(arrays["keep"], device=device)
        donors = np.column_stack(
            np.nonzero(
                arrays["keep"]
                & (arrays["labels"] >= 0).all(-1)
                & (arrays["mask"].sum((-2, -1)) > 0)
            )
        )
        if not len(donors):
            raise ValueError("No fully labelled train donors")
        self.donors = torch.as_tensor(donors, device=device)
        y, x = torch.meshgrid(
            torch.arange(160, device=device), torch.arange(240, device=device), indexing="ij"
        )
        self.grid = torch.stack((x, y), -1).float() + 0.5

    @torch.no_grad()
    def pair(self, indices, difficulty="mixed", attempts=1):
        image = self.rgb[indices].float() / 255
        n = len(indices)
        device = image.device
        occupied = self.foreground[indices]
        done = torch.zeros(n, dtype=torch.bool, device=device)
        after = image.clone()
        kind = torch.zeros(n, dtype=torch.long, device=device)
        ratios = image.new_zeros(n)
        pixels = image.new_zeros(n)
        for _ in range(attempts):
            front = (torch.rand(n, self.keep.shape[1], device=device) * self.keep[indices]).argmax(
                1
            )
            d, k = self.donors[torch.randint(len(self.donors), (n,), device=device)].T
            box = self.box[indices, front]
            source = self.box[d, k]
            extent = source[:, 2:] * (0.75 + 0.35 * torch.rand(n, 1, device=device))
            offset = (torch.rand(n, 2, device=device) - 0.5) * box[:, 2:] * 1.15
            offset[:, 1] = -offset[:, 1].abs()
            origin = box[:, :2] + 0.5 * box[:, 2:] + offset - 0.5 * extent
            uv = (self.grid - origin[:, None, None]) / extent[:, None, None]
            alpha = F.grid_sample(
                self.mask[d, k, None].float() / 255, uv * 2 - 1, align_corners=False
            )[:, 0]
            source_grid = (
                source[:, None, None, :2] + uv * source[:, None, None, 2:]
            ) / image.new_tensor([240, 160])
            patch = F.grid_sample(
                self.rgb[d].float() / 255, source_grid * 2 - 1, align_corners=False
            )
            visible = alpha * ~occupied
            fraction = visible.sum((1, 2)) / alpha.sum((1, 2)).clamp_min(1)
            count = (visible > 0.5).sum((1, 2))
            lo, hi = (0.035, 0.28) if difficulty == "tiny" else (0.12, 0.85)
            ok = (
                (fraction > lo)
                & (fraction < hi)
                & (count >= 4)
                & (d != indices)
                & self.keep[indices].any(1)
                & ~done
            )
            if difficulty == "tiny":
                ok &= count <= 60
                ok &= (F.max_pool2d(visible[:, None], 2)[:, 0] > 0.5).sum((1, 2)) >= 2
            else:
                ok &= (visible[:, 1::2, 1::2] > 0.5).sum((1, 2)) >= 4
            visible = visible * ok[:, None, None]
            after = after + (patch - image) * visible[:, None]
            sz, col, joint = self.labels[d, k].T
            kind = torch.where(ok, (sz * 8 + col) * 6 + joint, kind)
            ratios += fraction * ok
            pixels += count * ok
            done |= ok
        target = F.one_hot(kind, 96).float() * done[:, None]
        return image, after, target, torch.stack((done.sum(), ratios.sum(), pixels.sum())).float()
