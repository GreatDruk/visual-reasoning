"""Measure device throughput before choosing a batch size."""

import json
import os
import subprocess
import threading
import time

import numpy as np
import torch


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Utilization:
    def __init__(self, device):
        self.device = device
        self.values = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        fallback = False
        while not self.stop.wait(.2):
            try:
                if not fallback:
                    self.values.append(int(torch.cuda.utilization(self.device)))
                    continue
            except (AttributeError, RuntimeError, ImportError, OSError):
                fallback = True
            try:
                ordinal = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
                percent = subprocess.check_output(
                    ["nvidia-smi", "-i", ordinal, "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"], text=True, timeout=2)
                self.values.append(int(percent.strip()))
            except (OSError, ValueError, subprocess.SubprocessError):
                break

    def __enter__(self):
        if self.device.type == "cuda":
            self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join()

    def summary(self):
        return dict(gpu_util_pct=float(np.median(self.values)) if self.values else None,
                    gpu_util_samples=len(self.values))


def choose_batch(device, sizes, count, step, headroom=.90):
    if device.type == "cpu":
        return min(count, 8), []
    proposals = []
    capacity = torch.cuda.get_device_properties(device).total_memory
    for n in sorted({min(x, count) for x in sizes}):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            step(slice(0, n))
            sync(device)
            with Utilization(device) as device_use:
                start, repetitions = time.perf_counter(), 0
                while repetitions < 3 or (repetitions < 10 and time.perf_counter() - start < 1.1):
                    step(slice(0, n))
                    sync(device)
                    repetitions += 1
                elapsed = time.perf_counter() - start
            used = torch.cuda.max_memory_reserved(device)
            stat = dict(batch=n, items_per_s=n * repetitions / elapsed, reserved_gib=used / 2**30,
                        allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                        capacity_fraction=used/capacity, **device_use.summary())
        except torch.cuda.OutOfMemoryError:
            stat = dict(batch=n, oom=True)
        print("batch probe", json.dumps(stat), flush=True)
        proposals.append(stat)
        if stat.get("oom") or stat["capacity_fraction"] > headroom:
            break
    eligible = [p for p in proposals if p.get("capacity_fraction", 1.) <= headroom]
    if not eligible:
        raise RuntimeError("No profiled batch kept the required VRAM headroom")
    best = max(p["items_per_s"] for p in eligible)
    return max(p["batch"] for p in eligible if p["items_per_s"] >= .97 * best), proposals
