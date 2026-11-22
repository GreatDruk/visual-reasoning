"""Explicit immutable inputs, disposable work, and output-only artifact sinks."""

import hashlib
import io
import json
import os
import random
import subprocess
import threading
import time
import zipfile
from contextlib import nullcontext
from contextvars import ContextVar
from pathlib import Path

import numpy as np
import torch

_MONITOR = ContextVar("foundation_monitor", default=None)


def zero_loss(tensor):
    """Keep a zero gradient path without overflowing an FP16 reduction."""
    return tensor.sum(dtype=torch.float32) * 0.0


class Progress:
    """Low-frequency stage telemetry, relayed by the notebook process."""

    def __init__(self, stage, total, device=None, interval=30):
        self.stage, self.total, self.device = stage, total, device
        self.started = self.last = time.monotonic()
        self.interval = max(90, interval) if os.environ.get("FINAL_RUN") == "1" else interval
        self.update(0, force=True)

    def update(self, done, force=False, **values):
        now = time.monotonic()
        if not force and done < self.total and now - self.last < self.interval:
            return
        elapsed = now - self.started
        event = dict(
            stage=self.stage,
            done=done,
            total=self.total,
            seconds=round(elapsed, 1),
            rate=round(done / max(elapsed, 1e-6), 2),
            **values,
        )
        if self.device is not None and self.device.type == "cuda":
            event["vram_gib"] = round(torch.cuda.memory_allocated(self.device) / 2**30, 2)
            monitor = _MONITOR.get()
            event["gpu_pct"] = (
                monitor.values[-1] if monitor is not None and monitor.values else None
            )
        print("PROGRESS " + json.dumps(event), flush=True)
        self.last = now


def gradient_step(scaler, optimizer, parameters, *, loss=None, max_norm=5.0):
    """Recover from AMP overflow without clipping or applying invalid gradients."""
    scaler.unscale_(optimizer)
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        raise RuntimeError("Optimizer update has no gradients")
    norms = torch._foreach_norm(grads, 2.0)
    norm = torch.linalg.vector_norm(torch.stack(norms).float(), 2.0)
    values = [norm]
    if loss is not None:
        values.append(loss.detach().float())
    checked = torch.stack(values).detach().cpu().tolist()
    if loss is not None and not np.isfinite(checked[1]):
        raise FloatingPointError("Non-finite unscaled loss")
    finite = np.isfinite(checked[0])
    if finite:
        coefficient = (max_norm / (norm + 1e-6)).clamp(max=1.0)
        torch._foreach_mul_(grads, coefficient)
    elif not scaler.is_enabled():
        raise FloatingPointError("Non-finite full-precision gradients")
    elif all(torch.isfinite(g).all().item() for g in grads):
        # Norm overflow alone is not recorded by GradScaler's elementwise check.
        raise FloatingPointError("Gradient norm overflow with finite gradient entries")
    # unscale_ recorded inf/nan; GradScaler skips this optimizer update if needed.
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return bool(finite), checked[0]


def amp(device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )


def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1 if os.environ.get("FINAL_RUN") == "1" else 2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return np.random.default_rng(seed), torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )


def immutable_inputs(path, verify=True):
    path = Path(path).resolve()
    if "artefacts" in path.parts:
        raise ValueError("Artifacts are output-only; use immutable inputs")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if verify:
        for name, digest in manifest["files"].items():
            p = (path / name).resolve()
            if not p.is_relative_to(path):
                raise ValueError("Input path escapes its root")
            with p.open("rb") as f:
                actual = hashlib.file_digest(f, "sha256").hexdigest()
            if actual != digest:
                raise ValueError(f"Input checksum mismatch: {name}")
    return path, manifest


def load_state(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def profile_batches(device, sizes, count, step):
    """Measure completed passes; select throughput rather than the largest fitting batch."""
    if device.type != "cuda":
        return min(count, 2), []
    report = []
    capacity = torch.cuda.get_device_properties(device).total_memory
    for n in sorted({min(size, count) for size in sizes}):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            step(n)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            for _ in range(3):
                step(n)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            reserved = torch.cuda.max_memory_reserved(device)
            item = dict(
                batch=n,
                items_per_second=3 * n / elapsed,
                reserved_gib=reserved / 2**30,
                capacity_fraction=reserved / capacity,
            )
        except torch.cuda.OutOfMemoryError:
            item = dict(batch=n, oom=True)
        report.append(item)
        print("batch profile", json.dumps(item), flush=True)
        if item.get("oom") or item["capacity_fraction"] > 0.95:
            break
    eligible = [row for row in report if row.get("capacity_fraction", 1) <= 0.95]
    if not eligible:
        raise RuntimeError("No profiled batch left 5 percent VRAM headroom")
    fastest = max(row["items_per_second"] for row in eligible)
    selected = max(row["batch"] for row in eligible if row["items_per_second"] >= 0.97 * fastest)
    return selected, report


class Sink:
    def __init__(self, root, name, work):
        self.root = Path(root)
        self.name = name
        self.work = Path(work)
        self.stage = self.work / "delivery"
        self.stage.mkdir(parents=True, exist_ok=True)

    def write(self, kind, name, data):
        if kind not in ("results", "weights", "kaggle"):
            raise ValueError("Invalid output kind")
        stage = self.stage / kind / name
        stage.parent.mkdir(parents=True, exist_ok=True)
        stage.write_bytes(data)
        dest = self.root / kind / self.name / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def json(self, name, data):
        self.write(
            "results",
            name,
            json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        )

    def state(self, name, state):
        data = io.BytesIO()
        torch.save(state, data)
        self.write("weights", name, data.getvalue())

    def npz(self, name, **arrays):
        data = io.BytesIO()
        np.savez_compressed(data, **arrays)
        self.write("results", name, data.getvalue())

    def finish(self, include_weights=True):
        # Archive scratch copies, never reopen anything under artefacts.
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for p in self.stage.rglob("*"):
                if p.is_file() and (
                    include_weights or p.relative_to(self.stage).parts[0] != "weights"
                ):
                    z.write(p, p.relative_to(self.stage).as_posix())
        self.write("results", "result.zip", buffer.getvalue())


class Monitor:
    def __init__(self, device):
        self.device = device
        self.values = []
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        self.token = _MONITOR.set(self)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

            def collect():
                while not self.stop.wait(5):
                    try:
                        ordinal = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
                        s = subprocess.check_output(
                            [
                                "nvidia-smi",
                                "-i",
                                ordinal,
                                "--query-gpu=utilization.gpu",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                            timeout=2,
                        )
                        self.values.append(int(s.strip()))
                    except (OSError, ValueError, subprocess.SubprocessError):
                        return

            self.thread = threading.Thread(target=collect, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)
        _MONITOR.reset(self.token)

    def report(self):
        return dict(
            gpu_util_median=float(np.median(self.values)) if self.values else None,
            peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30
            if self.device.type == "cuda"
            else 0,
        )
