"""Slot-autoencoder pretraining on cached frozen features.

Uses train images (and optionally their flips). For final team fitting, train+val
may be selected explicitly, but test images must never fit representations.
Token reconstruction requires no captions. Optional count losses derive exact
counts and lower bounds from true train captions. Val labels are used for diagnostics.

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
import torch.nn.functional as F

from .data import canonicalized, load_split
from .models.slot_program import AND, EXACT, parse_program
from .models.slots import SlotAutoencoder, expand_slot_state
from .utils import get_device, resolve_path, set_seed

log = logging.getLogger(__name__)
DEFAULT_KEY = "clip_ViT-B-16-SigLIP2-256_webli_block8norm_256x384_siglip"
SPLITS = ("train", "train_flip")
ALLOWED_SPLITS = frozenset(("train", "train_flip", "val", "val_flip"))


def open_caches(base, splits=SPLITS):
    if not splits or len(set(splits)) != len(splits) or not set(splits) <= ALLOWED_SPLITS:
        raise ValueError("slot pretraining requires unique train/val splits; test is inference only")
    arrays = {}
    for name in splits:
        path = base / f"{name}.npy"
        if path.exists():
            arrays[name] = np.load(path, mmap_mode="r")
    if not arrays:
        raise FileNotFoundError(f"no feature caches under {base}")
    return arrays


def count_targets(split):
    """image name -> (exact object count or nan, lower bound), from true captions only.

    Exact: "exactly N objects" (no filter, no relation). Lower bounds: a true exactly-N
    caption with any filter implies >= N objects (+1 for its relation reference); a
    true binding caption names two different objects.
    """
    exact, bound = {}, {}
    for img, caption, label in zip(split.img_idx, split.captions, split.labels):
        p = parse_program(caption)
        if label != 1 or p is None:
            continue
        name = split.image_names[img]
        if p.op == EXACT:
            if not p.a and not p.relation:
                exact[name] = p.number
            bound[name] = max(bound.get(name, 0), p.number + (1 if p.relation else 0))
        elif p.op == AND:
            bound[name] = max(bound.get(name, 0), 2)
    return {n: (float(exact.get(n, np.nan)), float(b)) for n, b in bound.items() if b > 0 or n in exact}


class Sampler:
    """Random images over the union of caches, read from memmaps in a thread.

    With `labelled` = (global indices, targets [M, 2]), every batch starts with
    `count_batch` labelled images; the rest are uniform over all caches.
    """

    def __init__(self, arrays, batch_size, seed, labelled=None, count_batch=0):
        self.arrays = list(arrays.values())
        self.sizes = np.array([len(a) for a in self.arrays])
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.batch_size = min(batch_size, self.offsets[-1])
        self.labelled = labelled
        self.count_batch = min(count_batch, self.batch_size, len(labelled[0])) if labelled is not None else 0
        self.rng = np.random.default_rng(seed)
        self.pool = ThreadPoolExecutor(1)
        self.pending = self.pool.submit(self._read)

    def _gather(self, idx):
        out = np.empty((len(idx), *self.arrays[0].shape[1:]), dtype=self.arrays[0].dtype)
        for k, a in enumerate(self.arrays):
            m = (idx >= self.offsets[k]) & (idx < self.offsets[k + 1])
            if m.any():
                local = idx[m] - self.offsets[k]
                order = np.argsort(local)
                rows = np.nonzero(m)[0][order]
                out[rows] = np.asarray(a[local[order]])
        return out

    def _read(self):
        targets = None
        idx = self.rng.choice(self.offsets[-1], self.batch_size - self.count_batch, replace=False)
        if self.count_batch:
            # Exact totals are ~10% of labelled rows: draw them for half of the slots.
            exact = np.isfinite(self.labelled[1][:, 0])
            weight = np.where(exact, .5 / max(exact.sum(), 1), .5 / max((~exact).sum(), 1))
            chosen = self.rng.choice(len(self.labelled[0]), self.count_batch, replace=False, p=weight / weight.sum())
            idx = np.concatenate([self.labelled[0][chosen], idx])
            targets = torch.from_numpy(self.labelled[1][chosen])
        x = torch.from_numpy(self._gather(idx))
        return x if targets is None else (x, targets)

    def next(self):
        batch = self.pending.result()
        self.pending = self.pool.submit(self._read)
        return batch

    def close(self):
        self.pool.shutdown()


def save_checkpoint(model, path, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config, "state_dict": model.state_dict(), **(extra or {})}, path)


def checkpoint_splits(payload):
    """Recover the fitting splits from a checkpoint or a run trained from scratch."""
    splits = payload.get("pretraining_splits")
    if splits is None:
        args = payload.get("args", {})
        if args.get("init"):
            raise ValueError("pretraining splits are unknown for this resumed checkpoint")
        splits = [name.strip() for name in args.get("splits", "").split(",")]
    splits = set(splits)
    if "train" not in splits or not splits <= ALLOWED_SPLITS:
        raise ValueError("initial checkpoint must record train/val fitting splits; test is inference only")
    return splits


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


@torch.no_grad()
def count_diagnostic(model, features, truth, device):
    """Val-only evaluation of the slot count (labels never reach a loss)."""
    model.eval()
    enc = model.encode(features.to(device))
    soft = model.object_count(enc["attn"]).cpu().numpy()
    hard = (enc["attn"][:, model.object_slots].amax(-1) > model.count_threshold).sum(-1).cpu().numpy()
    metrics = {"val_count_mae": round(float(np.abs(soft - truth).mean()), 3),
               "val_count_exact": round(float((hard == truth).mean()), 3),
               "val_count_bias": round(float((hard - truth).mean()), 3)}
    if (truth >= 8).any():
        metrics["val_count_mae_8plus"] = round(float(np.abs(hard - truth)[truth >= 8].mean()), 3)
    return metrics


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
    p.add_argument("--amp", type=int, default=1, help="mixed precision on CUDA")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--splits", default=",".join(SPLITS))
    p.add_argument("--require-caches", action="store_true", help="fail if any selected cache is missing")
    p.add_argument("--init", default=None, help="slot_autoencoder.pt to continue from (K may grow)")
    p.add_argument("--background-slots", default="", help="comma-separated slot indices excluded from the count")
    p.add_argument("--count-weight", type=float, default=0., help="smooth-L1 weight on exact TRAIN object counts")
    p.add_argument("--bound-weight", type=float, default=0., help="squared-hinge weight on TRAIN lower bounds")
    p.add_argument("--count-batch", type=int, default=12, help="maximum labelled images per batch")
    p.add_argument("--count-warmup", type=int, default=0, help="steps of linear ramp-in for the count losses")
    p.add_argument("--count-threshold", type=float, default=.5)
    p.add_argument("--count-sharpness", type=float, default=10.)
    args = p.parse_args(argv)
    requested_splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    if not requested_splits or any(s not in ALLOWED_SPLITS for s in requested_splits):
        raise ValueError("slot pretraining may fit only train/val features, never test features")
    if "train" not in requested_splits:
        raise ValueError("slot pretraining must include train")
    counting = args.count_weight > 0 or args.bound_weight > 0
    if args.batch_size < 1 or (counting and args.count_batch < 1):
        p.error("training and supervised batch sizes must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    set_seed(args.seed)
    device = get_device(args.device)
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = resolve_path(args.features_dir) / args.key
    arrays = open_caches(base, requested_splits)
    if "train" not in arrays or (args.require_caches and set(arrays) != set(requested_splits)):
        raise FileNotFoundError(f"missing required feature caches in {base}: {set(requested_splits) - set(arrays)}")
    log.info("caches: " + ", ".join(f"{k} {v.shape}" for k, v in arrays.items()))
    first = arrays["train"]
    pretraining_splits = set(arrays)
    background = tuple(int(k) for k in args.background_slots.split(",") if k)
    if args.init:
        payload = torch.load(resolve_path(args.init), map_location="cpu")
        pretraining_splits.update(checkpoint_splits(payload))
        if not pretraining_splits <= set(requested_splits):
            raise ValueError("initial checkpoint used fitting splits outside the requested splits")
        config = {**payload["config"], "num_slots": args.slots, "background_slots": background,
                  "count_threshold": args.count_threshold, "count_sharpness": args.count_sharpness}
        model = SlotAutoencoder(**config)
        model.load_state_dict(expand_slot_state(payload["state_dict"], args.slots,
                                                torch.Generator().manual_seed(args.seed)))
        log.info(f"continued from {args.init}: {len(payload['state_dict']['slot_attention.queries'])} -> {args.slots} slots")
    else:
        model = SlotAutoencoder(first.shape[1], first.shape[2:], args.slots, args.slot_dim, args.input_dim,
                                args.iterations, args.decoder_hidden, args.decoder_layers,
                                args.target_dim or None, background, args.count_threshold, args.count_sharpness)
    model.to(device)
    if not args.init:
        rng = np.random.default_rng(args.seed)
        sample = np.asarray(first[np.sort(rng.choice(len(first), min(len(first), 256), replace=False))])
        model.fit_normalization(torch.from_numpy(sample).to(device))
    labelled, val_counts = None, None
    if counting:
        names = list(arrays)
        offsets = np.concatenate([[0], np.cumsum([len(arrays[n]) for n in names])])
        targets = count_targets(canonicalized(load_split("train", resolve_path("data/train.csv"))))
        idx, tgt = [], []
        for cache in ("train", "train_flip"):  # a flip keeps the object count
            if cache in arrays:
                train_names = json.loads((base / f"{cache}.json").read_text())
                if len(train_names) != len(arrays[cache]):
                    raise ValueError(f"{cache} feature cache and image names have different lengths")
                for i, n in enumerate(train_names):
                    if n in targets:
                        idx.append(offsets[names.index(cache)] + i)
                        tgt.append(targets[n])
        if not idx:
            raise ValueError("no count supervision matches the train feature caches")
        labelled = (np.array(idx), np.array(tgt, dtype=np.float32))
        log.info(f"count targets: {len(targets)} train images "
                 f"({int(np.isfinite(labelled[1][:, 0]).sum())} exact rows incl. flips), {len(idx)} rows")
        if "val" in arrays:
            from .slot_inspect import object_counts
            vc = object_counts(canonicalized(load_split("val", resolve_path("data/val.csv"))))
            val_names = json.loads((base / "val.json").read_text())
            rows = [i for i, n in enumerate(val_names) if n in vc]
            if rows:
                val_counts = (torch.from_numpy(np.array(arrays["val"][rows])), np.array([vc[val_names[i]] for i in rows]))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup)
                                              * .5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    sampler = Sampler(arrays, args.batch_size, args.seed, labelled, args.count_batch)
    amp = bool(args.amp) and device.type == "cuda"
    bf16 = amp and torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and not bf16)
    history, losses, count_losses, t0 = [], [], [], time.time()
    metadata = {"args": vars(args), "history": history, "pretraining_splits": sorted(pretraining_splits)}
    for step in range(1, args.steps + 1):
        model.train()
        batch = sampler.next()
        x, targets = batch if counting else (batch, None)
        x = x.to(device)
        with torch.autocast(device.type, dtype=dtype, enabled=amp):
            result = model(x)
        loss = result["loss"]
        if counting:
            targets = targets.to(device)
            count = model.object_count(result["attn"][:len(targets)].float())
            exact, bound = targets[:, 0], targets[:, 1]
            has = torch.isfinite(exact)
            ramp = min(1., step / args.count_warmup) if args.count_warmup else 1.
            count_loss = ramp * (args.count_weight * F.smooth_l1_loss(count[has], exact[has], reduction="sum")
                          + args.bound_weight * F.relu(bound - count).square().sum()) / len(targets)
            loss = loss + count_loss
            count_losses.append(count_loss.item())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        previous_scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() >= previous_scale:
            sched.step()
        losses.append(result["loss"].item())
        if step % args.log_every == 0 or step == args.steps:
            row = {"step": step, "loss": float(np.mean(losses)), "lr": sched.get_last_lr()[0],
                   "min": round((time.time() - t0) / 60, 2)}
            if counting:
                row["count_loss"] = float(np.mean(count_losses))
                if val_counts is not None:
                    row.update(count_diagnostic(model, *val_counts, device))
            history.append(row)
            losses.clear()
            count_losses.clear()
            log.info(json.dumps(row))
        if step % 4000 == 0:
            save_checkpoint(model, out / "slot_autoencoder.pt", metadata)
    sampler.close()
    save_checkpoint(model, out / "slot_autoencoder.pt", metadata)
    (out / "history.json").write_text(json.dumps({**metadata, "train_min": (time.time() - t0) / 60}, indent=2))
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
