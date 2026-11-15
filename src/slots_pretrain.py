"""Unsupervised slot-autoencoder pretraining on cached frozen features.

Uses every cached image of train, val and test (plus the cached horizontal flips of
train/val) and no captions or labels: the only objective is token reconstruction.

    uv run python -m src.slots_pretrain --features-dir <dir> --out artefacts/slots/v1
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch

from .models.slots import SlotAutoencoder
from .utils import get_device, resolve_path, set_seed

log = logging.getLogger(__name__)
DEFAULT_KEY = "clip_ViT-B-16-SigLIP2-256_webli_block8norm_256x384_siglip"
SPLITS = ("train", "val", "test", "train_flip", "val_flip")


def open_caches(base, splits=SPLITS):
    arrays = {}
    for name in splits:
        path = base / f"{name}.npy"
        if path.exists():
            arrays[name] = np.load(path, mmap_mode="r")
    if not arrays:
        raise FileNotFoundError(f"no feature caches under {base}")
    return arrays


class Sampler:
    """Uniform random images over the union of caches, read from memmaps in a thread."""

    def __init__(self, arrays, batch_size, seed):
        self.arrays = list(arrays.values())
        self.sizes = np.array([len(a) for a in self.arrays])
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.pool = ThreadPoolExecutor(1)
        self.pending = self.pool.submit(self._read)

    def _read(self):
        idx = np.sort(self.rng.choice(self.offsets[-1], self.batch_size, replace=False))
        parts = []
        for k, a in enumerate(self.arrays):
            local = idx[(idx >= self.offsets[k]) & (idx < self.offsets[k + 1])] - self.offsets[k]
            if len(local):
                parts.append(np.asarray(a[local]))
        return torch.from_numpy(np.concatenate(parts))

    def next(self):
        batch = self.pending.result()
        self.pending = self.pool.submit(self._read)
        return batch


def save_checkpoint(model, path, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config, "state_dict": model.state_dict(), **(extra or {})}, path)


def mask_overlays(model, features, image_paths, out_path, device):
    """Image | decoder-alpha argmax segmentation | one panel per slot (attention)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(np.array(features)).to(device))
    h, w = model.grid
    alpha = out["alpha"].reshape(len(features), -1, h, w).cpu()
    attn = out["attn"].reshape(len(features), -1, h, w).cpu()
    k = alpha.shape[1]
    colors = plt.get_cmap("tab20")(np.arange(k) % 20)[:, :3]
    fig, axes = plt.subplots(len(features), k + 2, figsize=(1.6 * (k + 2), 1.25 * len(features)))
    for row, path in enumerate(image_paths):
        image = np.asarray(Image.open(path).convert("RGB")) / 255.0
        H, W = image.shape[:2]
        up = torch.nn.functional.interpolate(alpha[row:row + 1], size=(H, W), mode="bilinear")[0]
        seg = colors[up.argmax(0).numpy()]
        axes[row, 0].imshow(image)
        axes[row, 1].imshow(.45 * image + .55 * seg)
        a_up = torch.nn.functional.interpolate(attn[row:row + 1], size=(H, W), mode="bilinear")[0]
        for s in range(k):
            m = a_up[s].numpy()[..., None]
            axes[row, s + 2].imshow(image * (.15 + .85 * m))
            axes[row, s + 2].set_title(f"{out['area'][row, s].item():.2f}", fontsize=6)
        for ax in axes[row]:
            ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout(pad=.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-dir", default="artefacts/features")
    p.add_argument("--key", default=DEFAULT_KEY)
    p.add_argument("--out", default="artefacts/slots/v1")
    p.add_argument("--slots", type=int, default=11)
    p.add_argument("--slot-dim", type=int, default=128)
    p.add_argument("--input-dim", type=int, default=256)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--decoder-hidden", type=int, default=512)
    p.add_argument("--decoder-layers", type=int, default=3)
    p.add_argument("--target-dim", type=int, default=256, help="PCA components to reconstruct (0: all)")
    p.add_argument("--amp", type=int, default=1, help="bf16 autocast on cuda/mps")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--splits", default=",".join(SPLITS))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    set_seed(args.seed)
    device = get_device(args.device)
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = resolve_path(args.features_dir) / args.key
    arrays = open_caches(base, args.splits.split(","))
    log.info("caches: " + ", ".join(f"{k} {v.shape}" for k, v in arrays.items()))
    first = next(iter(arrays.values()))
    model = SlotAutoencoder(first.shape[1], first.shape[2:], args.slots, args.slot_dim, args.input_dim,
                            args.iterations, args.decoder_hidden, args.decoder_layers,
                            args.target_dim or None)
    rng = np.random.default_rng(args.seed)
    sample = np.asarray(first[np.sort(rng.choice(len(first), 256, replace=False))])
    model.fit_normalization(torch.from_numpy(sample))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup)
                                              * .5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    sampler = Sampler(arrays, args.batch_size, args.seed)
    history, losses, t0 = [], [], time.time()
    for step in range(1, args.steps + 1):
        model.train()
        x = sampler.next().to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bool(args.amp) and device.type != "cpu"):
            loss = model(x)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % args.log_every == 0 or step == args.steps:
            row = {"step": step, "loss": float(np.mean(losses)), "lr": sched.get_last_lr()[0],
                   "min": round((time.time() - t0) / 60, 2)}
            history.append(row)
            losses = []
            log.info(json.dumps(row))
        if step % 4000 == 0:
            save_checkpoint(model, out / "slot_autoencoder.pt", {"args": vars(args), "history": history})
    save_checkpoint(model, out / "slot_autoencoder.pt", {"args": vars(args), "history": history})
    (out / "history.json").write_text(json.dumps({"args": vars(args), "history": history,
                                                  "train_min": (time.time() - t0) / 60}, indent=2))
    if "val" in arrays:
        names = json.loads((base / "val.json").read_text())[:8]
        mask_overlays(model, arrays["val"][:8], [resolve_path("data/val") / n for n in names],
                      out / "val_masks.png", device)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
