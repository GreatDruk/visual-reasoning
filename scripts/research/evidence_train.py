"""Cache image maps, cross-fit hard examples, and train paired caption verifiers."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from scripts.research.evidence_data import (
    extract, labeled_pairs, load_maps, opposite_rows, read_captions, scene_folds,
)
from scripts.research.evidence_model import EvidenceVerifier, pair_loss
from scripts.research.evidence_runtime import Utilization, choose_batch, sync

SEED = 421
FOLDS = 4


class CaptionBank:
    def __init__(self, data, cache, split, device, limit=None):
        frame, names, scene, caption, words, meta = read_captions(data, split)
        if limit:
            use = scene < limit
            frame, names, scene = frame[use].reset_index(drop=True), names[:limit], scene[use]
            caption = [t for t, keep in zip(caption, use) if keep]
            words, meta = words[use], meta[use]
        self.frame, self.names, self.scene, self.caption = frame, names, scene, caption
        self.label = frame.label.to_numpy() if "label" in frame else None
        self.device = device
        self.fine, self.semantic = load_maps(cache, split, names, device)
        self.words, self.meta, self.image = (torch.as_tensor(x, device=device) for x in (words, meta, scene))

    def forward(self, model, rows, image_rows=None):
        scene = self.image[rows if image_rows is None else image_rows]
        fine, semantic = self.fine[scene], self.semantic[scene]
        if self.device.type == "cpu":
            fine, semantic = fine.float(), semantic.float()
        with torch.autocast(self.device.type, enabled=self.device.type == "cuda", dtype=torch.float16):
            return model(fine, semantic, self.words[rows], self.meta[rows])


@torch.inference_mode()
def predict(model, bank, rows, batch, image_rows=None):
    model.eval()
    values = []
    for i in range(0, len(rows), batch):
        sl = slice(i, i + batch)
        score, _ = bank.forward(model, rows[sl], None if image_rows is None else image_rows[sl])
        values.append(score.cpu().numpy())
    return np.concatenate(values)


def difficulty(out, bank, pairs):
    oof = np.full(len(bank.frame), np.nan)
    for fold in range(FOLDS):
        with np.load(out / f"fold_{fold}.npz") as data:
            rows, values = data["rows"], data["score"]
            np.testing.assert_array_equal(bank.frame.id.to_numpy()[rows], data["ids"])
            if np.isfinite(oof[rows]).any():
                raise ValueError("OOF predictions overlap")
            oof[rows] = values
    if not np.isfinite(oof).all():
        raise ValueError("OOF predictions must cover every training row")
    pair_gap = oof[pairs[:, 0]] - oof[pairs[:, 1]]
    weights = 1. + 2. * np.exp(-np.maximum(pair_gap, 0.))
    return weights / weights.mean()


def fit(bank, out, arm, epochs, fold, batch_override=None):
    device = bank.device
    torch.manual_seed(SEED)
    pairs = labeled_pairs(bank.frame)
    selected = np.ones(len(bank.frame), dtype=bool)
    if fold is not None:
        selected = scene_folds(len(bank.names), FOLDS, SEED)[bank.scene] != fold
    train = selected[pairs[:, 0]]
    assert np.array_equal(selected[pairs[:, 1]], train)
    pairs = pairs[train]
    hard = arm == "hard"
    weights = difficulty(out, bank, pairs) if hard else np.ones(len(pairs))
    pair_rows = torch.as_tensor(pairs, device=device)
    weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    partners = opposite_rows(bank.caption, bank.label, bank.scene, selected, SEED) if hard else None
    labels = torch.as_tensor(bank.label, dtype=torch.float32, device=device)
    if partners is not None:
        partners = torch.as_tensor(partners, device=device)

    def trainer():
        torch.manual_seed(SEED)
        model = EvidenceVerifier().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.015,
                                      fused=device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        return model, optimizer, scaler

    def update(model, optimizer, scaler, index):
        optimizer.zero_grad(set_to_none=True)
        original = pair_rows[index]
        flat = original.flatten()
        score, count = bank.forward(model, flat)
        logits = score.view(-1, 2)
        count = count.view(-1, 2, count.shape[-1])
        if hard:
            candidate = flat[::5]
            target = partners[candidate]
            available = target >= 0
            swap, _ = bank.forward(model, candidate, target.clamp_min(0))
            anchor, truth = score[::5], labels[candidate]
            sign = 2 * truth - 1
            image_loss = F.softplus(-sign * (anchor - swap))
            image_loss += F.binary_cross_entropy_with_logits(swap, 1 - truth, reduction="none")
            image_loss = .35 * (image_loss * available).sum() / available.sum().clamp_min(1)
        else:
            image_loss = 0
        loss = pair_loss(logits, count, bank.meta[original], weights[index]) + image_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        scaler.step(optimizer)
        scaler.update()
        return loss.detach()

    if batch_override is None:
        probe_model, probe_opt, probe_scaler = trainer()
        probe = lambda part: update(probe_model, probe_opt, probe_scaler, part)
        batch, probes = choose_batch(device, (64, 128, 256, 512, 768, 1024), len(pairs), probe)
        del probe, probe_model, probe_opt, probe_scaler
    else:
        batch, probes = min(batch_override, len(pairs)), []
    model, optimizer, scaler = trainer()
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=8e-5)
    history = []
    tag = f"fold_{fold}" if fold is not None else arm
    for epoch in range(epochs):
        order = torch.randperm(len(pairs), device=device)
        start = time.perf_counter()
        with Utilization(device) as utilization:
            loss = torch.stack([update(model, optimizer, scaler, index) for index in order.split(batch)]).mean().item()
        seconds = time.perf_counter() - start
        if not np.isfinite(loss):
            raise RuntimeError(f"{tag}: training produced nonfinite loss")
        schedule.step()
        row = dict(epoch=epoch+1, seconds=seconds, loss=loss, pairs_per_s=len(pairs)/seconds,
                   **utilization.summary())
        history.append(row)
        if epoch in (0, epochs-1) or (epoch+1) % 5 == 0:
            print(tag, json.dumps(row), flush=True)
    if fold is None:
        torch.save(model.state_dict(), out / f"{tag}.pt")
    else:
        heldout = np.flatnonzero(~selected)
        values = predict(model, bank, torch.as_tensor(heldout, device=device), min(2*batch, 512))
        np.savez_compressed(out / f"{tag}.npz", ids=bank.frame.id.to_numpy()[heldout], rows=heldout, score=values)
    return dict(batch_pairs=batch, pairs=len(pairs), epochs=history, probes=probes)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("extract", "fold", "fit", "predict"))
    cli.add_argument("--data", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--split", choices=("train", "val", "test"), default="train")
    cli.add_argument("--arm", choices=("control", "hard"), default="control")
    cli.add_argument("--fold", type=int, choices=range(FOLDS), default=0)
    cli.add_argument("--shard", type=int, choices=(0, 1), default=0)
    cli.add_argument("--epochs", type=int, default=28)
    cli.add_argument("--r50", type=Path)
    cli.add_argument("--vision", type=Path)
    cli.add_argument("--limit", type=int, help="Smoke test only")
    cli.add_argument("--batch", type=int, help="Override autotuning for a smoke test")
    args = cli.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = args.out / "maps"
    cache.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if args.stage == "extract":
        info = extract(args.data, args.split, cache, args.r50, args.vision, device, args.shard,
                       choose_batch, args.limit)
        tag = f"extract_{args.split}_{args.shard}"
    else:
        split = args.split if args.stage == "predict" else "train"
        bank = CaptionBank(args.data, cache, split, device, args.limit)
        sync(device)
        load_s = time.perf_counter() - start
        if args.stage == "predict":
            model = EvidenceVerifier().to(device)
            model.load_state_dict(torch.load(args.out / f"{args.arm}.pt", weights_only=True, map_location="cpu"))
            rows = torch.arange(len(bank.frame), device=device)
            score = predict(model, bank, rows, args.batch or (512 if device.type == "cuda" else 8))
            tag = f"{args.arm}_{args.split}"
            np.savez_compressed(args.out / f"{tag}.npz", ids=bank.frame.id.to_numpy(), score=score)
            info = dict(rows=len(rows))
        else:
            if args.stage == "fold":
                tag = f"fold_{args.fold}"
                info = fit(bank, args.out, "control", args.epochs, args.fold, args.batch)
            else:
                tag = args.arm
                info = fit(bank, args.out, args.arm, args.epochs, None, args.batch)
        info["load_maps_s"] = load_s
    info["elapsed_s"] = time.perf_counter() - start
    (args.out / f"{tag}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
