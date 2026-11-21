"""Extract compact frozen-source tables once; cache only in explicit scratch."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from scipy.special import logit
from torch.nn import functional as F

from .detectors import Detector, dense_table
from .heads import N_ATTRS, TableHead, load_head
from .native import CenterDetector
from .runtime import Progress, amp, load_state, profile_batches
from .union import (
    BASE_DIM,
    MAX_SLOTS,
    PHASES,
    ROI_DIM,
    detector_rows,
    local_features,
    merge,
    undo_shift,
)


def cache_key(cfg, scenes, inputs_manifest, foundation_manifest):
    u = cfg["union"]
    item = foundation_manifest["models"].get(u["source"])
    source = (
        inputs_manifest["files"]["reference/r50_img.pt"]
        if u["source"] == "historical"
        else foundation_manifest["files"][item["checkpoint"]]
    )
    code = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    specification = dict(
        source=source,
        secondary={
            k: v
            for k, v in inputs_manifest["files"].items()
            if k in ("reference/thaw.pt", "reference/thaw_head.pt", "reference/r50_img_head.pt")
        },
        csv=hashlib.sha256((scenes.root / f"{scenes.split}.csv").read_bytes()).hexdigest(),
        root=str(scenes.root.resolve()),
        names=scenes.names.tolist(),
        code=code,
        matching={
            k: u[k]
            for k in ("proposal_floor", "single_floor", "radius_factor", "radius_min", "radius_max")
        },
    )
    return hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:24]


def load_sources(cfg, inputs, foundation_root, manifest, device):
    source = cfg["union"]["source"]
    if source == "historical":
        settings = dict(cfg["model"], backbone="resnet", neck="sum", separate=False, width=64)
        state = dict(
            model=load_state(inputs / "reference/r50_img.pt"),
            head=load_state(inputs / "reference/r50_img_head.pt"),
        )
    else:
        item = manifest["models"][source]
        settings = item["config"]["model"]
        state = load_state(foundation_root / item["checkpoint"])
        if state["config"] != item["config"]:
            raise ValueError("Foundation checkpoint configuration differs from manifest")
    if settings["neck"] != "sum" or settings["width"] != 64:
        raise ValueError("Recovery currently requires a 64-channel sum pyramid")
    primary = Detector(settings)
    primary.load_state_dict(state["model"])
    primary_head = TableHead()
    load_head(primary_head, state["head"])
    secondary = CenterDetector()
    secondary.load_state_dict(load_state(inputs / "reference/thaw.pt"))
    secondary_head = TableHead()
    load_head(secondary_head, load_state(inputs / "reference/thaw_head.pt"))
    primary = primary.to(device, memory_format=torch.channels_last).eval()
    secondary = secondary.to(device, memory_format=torch.channels_last).eval()
    models = [primary, primary_head.to(device).eval(), secondary, secondary_head.to(device).eval()]
    gen = torch.Generator().manual_seed(217)
    matrix = torch.linalg.qr(torch.randn(40, 3 + N_ATTRS, generator=gen))[0].to(device)
    return (*models, matrix)


@torch.inference_mode()
def prepare_bank(cfg, scenes, inputs, inputs_manifest, foundation_root, manifest, scratch, device):
    root = Path(scratch).resolve()
    if "artefacts" in root.parts:
        raise ValueError("Recovery cache must be disposable scratch")
    path = root / "recovery" / cache_key(cfg, scenes, inputs_manifest, manifest)
    path.mkdir(parents=True, exist_ok=True)
    wait = time.monotonic()
    with FileLock(str(path) + ".lock"):
        waited = time.monotonic() - wait
        if (path / "ready.json").exists():
            return path, dict(
                json.loads((path / "ready.json").read_text()), cache_hit=True, wait_seconds=waited
            )
        started = time.monotonic()
        if scenes._rgb is None:
            scenes.prepare_images(root / "images", device)
        primary, head, secondary, secondary_head, matrix = load_sources(
            cfg, inputs, foundation_root, manifest, device
        )
        floor = float(logit(cfg["union"]["proposal_floor"]))

        def native(images):
            with amp(device):
                latent = secondary.features(images)
                raw = secondary.head(latent)
                descriptor = torch.einsum("bdhw,ds->bshw", latent, matrix)
            return detector_rows(
                secondary_head, dense_table(raw.float(), latent=descriptor.float(), min_logit=floor)
            )

        def phase(images, native_rows, offset):
            dx, dy = offset
            shifted = (
                F.pad(images, (dx, 0, dy, 0), mode="replicate")[:, :, :160, :240]
                if dx or dy
                else images
            )
            with amp(device):
                pyramid = primary.pyramid(shifted)
                raw = primary.head(pyramid[0])
            rows = detector_rows(head, dense_table(raw.float(), min_logit=floor))
            other_boxes = native_rows[..., -6:-2].clone()
            other_boxes[..., :2] += other_boxes.new_tensor(offset) / other_boxes.new_tensor(
                [240, 160]
            )
            boxes = torch.cat((rows[..., -6:-2], other_boxes), 1)
            roi = local_features(pyramid[0], boxes).half()
            return undo_shift(rows, offset), roi

        def probe(n):
            images = scenes.images(np.arange(n) % len(scenes.names), device)
            phase(images, native(images), PHASES[-1])

        batch, profile = profile_batches(
            device, (32, 64, 128, cfg["union"]["extract_batch"]), len(scenes.names), probe
        )
        schemas = {
            "x": ((4, len(scenes.names), MAX_SLOTS, BASE_DIM), np.float32),
            "xy": ((4, len(scenes.names), MAX_SLOTS, 2), np.float32),
            "valid": ((4, len(scenes.names), MAX_SLOTS), bool),
            "roi": ((4, len(scenes.names), MAX_SLOTS, ROI_DIM), np.float16),
            "source": ((4, len(scenes.names), 20, 23), np.float32),
            "secondary": ((len(scenes.names), 20, 23), np.float32),
        }
        import shutil

        required = sum(
            np.prod(shape) * np.dtype(dtype).itemsize for shape, dtype in schemas.values()
        )
        if shutil.disk_usage(path).free < required + 512 * 2**20:
            raise OSError(
                f"Recovery tables need {required / 2**30:.2f} GiB plus 0.5 GiB scratch reserve"
            )
        arrays = {
            k: np.lib.format.open_memmap(path / f"{k}.tmp.npy", mode="w+", dtype=dtype, shape=shape)
            for k, (shape, dtype) in schemas.items()
        }
        progress = Progress(
            f"{scenes.split}/union/{cfg['union']['source']}", len(scenes.names) * 4, device
        )
        totals = dict(candidates=0, matched=0, singleton=0)
        for start in range(0, len(scenes.names), batch):
            end = min(start + batch, len(scenes.names))
            images = scenes.images(np.arange(start, end), device)
            native_rows = native(images)
            right = native_rows.cpu().numpy()
            arrays["secondary"][start:end] = right
            for ph, offset in enumerate(PHASES):
                rows, roi = phase(images, native_rows, offset)
                left = rows.cpu().numpy()
                vectors = roi.cpu().numpy()
                if not all(np.isfinite(a).all() for a in (left, right, vectors)):
                    raise FloatingPointError("Non-finite frozen source output or ROI features")
                arrays["source"][ph, start:end] = left
                for i in range(end - start):
                    x, xy, valid, extra, counts = merge(
                        left[i], right[i], vectors[i, :20], vectors[i, 20:], cfg["union"]
                    )
                    for k, v in [("x", x), ("xy", xy), ("valid", valid), ("roi", extra)]:
                        arrays[k][ph, start + i] = v
                    for k, v in counts.items():
                        totals[k] += int(v)
                progress.update(start * 4 + (ph + 1) * (end - start), batch=batch)
        for k, array in arrays.items():
            array.flush()
            array._mmap.close()
            (path / f"{k}.tmp.npy").replace(path / f"{k}.npy")
        np.save(path / "names.npy", scenes.names, allow_pickle=False)
        np.save(
            path / "offsets.npy", head.relation_offset.detach().cpu().numpy(), allow_pickle=False
        )
        report = dict(
            seconds=time.monotonic() - started,
            wait_seconds=waited,
            cache_hit=False,
            batch=batch,
            batch_profile=profile,
            source=cfg["union"]["source"],
            split=scenes.split,
            scenes=len(scenes.names),
            phases=PHASES,
            proposal_statistics=totals,
            bytes=int(required),
        )
        pending = path / "ready.tmp.json"
        pending.write_text(json.dumps(report, indent=2), encoding="utf8")
        pending.replace(path / "ready.json")
        progress.update(len(scenes.names) * 4, force=True)
        return path, report


class TableStore:
    def __init__(self, path, scenes, device, roi=False):
        path = Path(path)
        if not np.array_equal(np.load(path / "names.npy"), scenes.names):
            raise ValueError("Recovery scene order differs from CSV")
        valid = np.load(path / "valid.npy", mmap_mode="r")
        self.slots = max(10, int(valid.sum(-1).max()))
        self.fields = {}
        for key in ("x", "xy", "valid", *(("roi",) if roi else ())):
            values = np.load(path / f"{key}.npy", mmap_mode="r")[:, :, : self.slots]
            target = torch.empty(
                values.shape,
                dtype=torch.from_numpy(np.empty((), values.dtype)).dtype,
                device=device,
            )
            for i in range(4):
                for j in range(0, len(scenes.names), 256):
                    target[i, j : j + 256].copy_(
                        torch.from_numpy(np.array(values[i, j : j + 256], copy=True)).to(device)
                    )
            self.fields[key] = target

    def select(self, indices, phases):
        device = self.fields["x"].device
        idx = torch.as_tensor(indices, device=device)
        ph = torch.as_tensor(phases, device=device)
        x = self.fields["x"][ph, idx]
        if "roi" in self.fields:
            x = torch.cat((x, self.fields["roi"][ph, idx].float()), -1)
        return x, self.fields["xy"][ph, idx], self.fields["valid"][ph, idx]

    def phase(self, n):
        return self.select(np.arange(self.fields["x"].shape[1]), n)
