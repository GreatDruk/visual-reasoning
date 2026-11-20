"""Matched mask-set trials with synthetic occlusions and optional reviewed natural masks."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import expit, softmax
import torch
from torch.nn import functional as F

from scripts.research import detail_counts as detail
from scripts.research.evidence_data import decode_images, read_captions
from scripts.research.evidence_runtime import Utilization, choose_batch, sync
from src.models.slot_program import EXCLUSIVE_GROUPS, WORD_INDEX, parse_program
from src.scene_posterior import sample_scene_scores
from src.twins import find_twins


def natural_targets(path, names, device):
    records = pd.read_csv(path).fillna("")
    lookup = {name: i for i, name in enumerate(names)}
    batches = defaultdict(list)
    for row in records.itertuples(index=False):
        if row.image not in lookup or row.kind not in ("rear", "highlight", "duplicate", "background"):
            raise ValueError(f"Unexpected mask row: {row.image}, {row.kind}")
        file = path.parent / row.mask
        if not file.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError(f"Mask must be next to the review CSV: {file}")
        with Image.open(file) as image:
            if image.size != (240, 160):
                raise ValueError(f"Mask is not in scene coordinates: {file}")
            mask = np.asarray(image.convert("L")) > 127
        mask = mask.reshape(80, 2, 120, 2).any(axis=(1, 3))
        if mask.sum() < 2:
            raise ValueError(f"Mask has fewer than two cells: {file}")
        if row.kind == "rear":
            _, colors, materials, shapes = EXCLUSIVE_GROUPS
            target = colors.index(row.color), 3 * materials.index(row.material) + shapes.index(row.shape)
            batches["rear"].append((lookup[row.image], mask, target))
        else:
            batches["negative"].append((lookup[row.image], mask, (-1, -1)))
    result = {}
    for kind in ("rear", "negative"):
        if not batches[kind]:
            raise ValueError(f"No reviewed {kind} objects in {path}")
        indices, masks, labels = zip(*batches[kind])
        result[kind] = tuple(torch.as_tensor(np.asarray(a), device=device) for a in (indices, masks, labels))
    return result


def local_loss(raw, logits, mask, label, positives):
    prob = logits.float().sigmoid()
    truth = mask[:, None]
    covered = (prob * truth).sum((2, 3))
    mass = prob.sum((2, 3)).clamp_min(1e-4)
    dice = (2 * covered + 1) / (mass + truth.sum((2, 3)) + 1)
    presence = raw[:, :, 0]
    i = torch.arange(positives, device=raw.device)
    match = dice[:positives].detach().argmax(dim=1)
    positive_logits = logits[i, match]
    positive_mask = mask[:positives]
    mismatch = F.binary_cross_entropy_with_logits(positive_logits, positive_mask.float(), reduction="none")
    pixel = (mismatch * positive_mask).sum((1, 2)) / positive_mask.sum((1, 2))
    pixel = pixel + (mismatch * ~positive_mask).mean((1, 2))
    direct = (1 - dice[i, match] + pixel + F.softplus(-presence[i, match])).mean()
    color = F.cross_entropy(raw[i, match, 3:11], label[:positives, 0])
    joint = F.cross_entropy(raw[i, match, 11:], label[:positives, 1])
    # A foreground mask covering a large object may include a small highlight.
    # Penalize queries concentrated inside the marked false-instance region.
    false_instance = (covered[positives:] / mass[positives:]).detach()
    negatives = (F.softplus(presence[positives:]) * false_instance.square()).mean()
    return direct + .35 * (color + joint) + 2 * negatives


def fit(args, device):
    _, names, *_ = read_captions(args.data, "train")
    started = time.perf_counter()
    images = decode_images(args.data, "train", names)
    decode_s = time.perf_counter() - started
    started = time.perf_counter()
    with np.load(detail.EXTRA / "fragment_train.npz") as source:
        np.testing.assert_array_equal(source["names"], names)
    bank = detail.Scenes(torch.device("cpu"), images)
    prep_s = time.perf_counter() - started
    started = time.perf_counter()
    bank.device = device
    for name, tensor in vars(bank).items():
        if isinstance(tensor, torch.Tensor):
            setattr(bank, name, tensor.to(device))
    del images
    sync(device)
    upload_s = time.perf_counter() - started
    started = time.perf_counter()
    reviews = natural_targets(args.review, names, device) if args.review else None
    sync(device)
    review_s = time.perf_counter() - started
    eligible = torch.arange(len(names), device=device)[:args.limit]

    def trainer():
        torch.manual_seed(args.seed)
        model = detail.MaskSet().to(device, memory_format=torch.channels_last)
        for layer in model.net.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
        heads = [p for n, p in model.named_parameters() if not n.startswith("net.")]
        opt = torch.optim.AdamW([dict(params=model.net.parameters(), lr=6e-5),
                                 dict(params=heads, lr=6e-4)], fused=device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        return model, opt, scaler

    match = {}

    def update(model, optimizer, scaler, idx):
        optimizer.zero_grad(set_to_none=True)
        # Use half original and half synthetic scenes, preserving actual batch size.
        rgb, mask, _, attrs, _ = bank.pair(idx, tiny_fragments=True)
        select = torch.arange(len(idx), device=device) + len(idx) * (
            torch.rand(len(idx), device=device) < .50).long()
        rgb, mask, attrs = rgb[select], mask[select], attrs[select, 1:]
        with detail.amp(device):
            raw, pred = model(rgb)
            loss = detail.mask_loss(raw, pred, mask, attrs, timing=match)
            if reviews is not None:
                n = max(1, len(idx) // 8)
                selected = []
                for kind in ("rear", "negative"):
                    values = reviews[kind]
                    i = torch.randint(len(values[0]), (n,), device=device)
                    selected.append([a[i] for a in values])
                indices, truth, labels = (torch.cat(x) for x in zip(*selected))
                estimates = model(bank.rgb[indices].float().div(255))
                loss = loss + .55 * local_loss(*estimates, truth, labels, n)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        scaler.step(optimizer)
        scaler.update()
        return loss.detach()

    if args.batch is None:
        model, optimizer, scaler = trainer()
        batch, probes = choose_batch(device, (16, 32, 48, 64, 96, 128, 192), len(eligible),
                                      lambda part: update(model, optimizer, scaler, eligible[part]))
        del model, optimizer, scaler
    else:
        batch, probes = args.batch, []
    model, optimizer, scaler = trainer()
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=5e-6)
    history = []
    for epoch in range(args.epochs):
        start = time.perf_counter()
        order = eligible[torch.randperm(len(eligible), device=device)]
        loss, matching = [], []
        with Utilization(device) as use:
            for idx in order.split(batch):
                loss.append(update(model, optimizer, scaler, idx))
                matching.append(match["matching_cpu_ms"])
            mean = torch.stack(loss).mean().item()
        if not np.isfinite(mean):
            raise RuntimeError("Nonfinite mask loss")
        schedule.step()
        seconds = time.perf_counter() - start
        row = dict(epoch=epoch+1, loss=mean, seconds=seconds, scenes_per_s=len(eligible)/seconds,
                   matching_cpu_ms=float(np.mean(matching)), **use.summary())
        history.append(row)
        if epoch in (0, args.epochs-1) or (epoch+1) % 5 == 0:
            print("mask epoch", json.dumps(row), flush=True)
    torch.save(model.state_dict(), args.out / "mask.pt")
    return dict(decode_cpu_s=decode_s, rasterize_cpu_s=prep_s, upload_s=upload_s, review_s=review_s,
                batch=batch, probes=probes, history=history,
                reviewed={k: len(v[0]) for k, v in reviews.items()} if reviews else {})


@torch.inference_mode()
def infer(args, device):
    frame, names, scene, caption, *_ = read_captions(args.data, args.split)
    model = detail.MaskSet().to(device, memory_format=torch.channels_last).eval()
    model.load_state_dict(torch.load(args.out / "mask.pt", weights_only=True, map_location="cpu"))
    groups = frame.groupby("image", sort=False).indices
    started = time.perf_counter()
    pixels = decode_images(args.data, args.split, names)
    decode_s = time.perf_counter() - started
    sizes = ((16, 32, 64, 96, 128, 192), len(names))

    def forward(part):
        rgb = torch.from_numpy(pixels[part]).to(device).permute(0, 3, 1, 2).float().div_(255)
        with detail.amp(device):
            return model(rgb)

    batch, probes = choose_batch(device, *sizes, forward) if args.batch is None else (args.batch, [])
    scores, records, packed_masks = np.zeros(len(frame)), [], []
    for start in range(0, len(names), batch):
        raw, masks = (x.cpu().numpy() for x in forward(slice(start, start + batch)))
        for j, name in enumerate(names[start:start+len(raw)]):
            idx = groups[name]
            assert (scene[idx] == start+j).all()
            positions, rows, conf, boxes = [], [], [], []
            order = np.argsort(-raw[j, :, 0])
            kept = []
            for k in order:
                mask = masks[j, k] > 0
                y, x = np.where(mask)
                if len(x) < 2 or raw[j, k, 0] < -3:
                    continue
                if any((mask & m).sum() / (mask | m).sum() >= .65 for m in kept):
                    continue
                kept.append(mask)
                box = [int(v) for v in (2*x.min(), 2*y.min(), 2*x.max()+2, 2*y.max()+2)]
                rows.append(raw[j, k, 1:])
                boxes.append(box)
                positions.append((2*(x.mean()+.5)/120-1, 2*(y.max()+.5)/80-1))
                conf.append(float(expit(raw[j, k, 0])))
            if not rows:
                continue
            size, color, joint = np.split(np.asarray(rows), [2, 10], axis=1)
            joint = softmax(joint, axis=1).reshape(-1, 2, 3)
            probs = (softmax(size, axis=1), softmax(color, axis=1), joint.sum(2), joint.sum(1))
            attrs = [(np.array([WORD_INDEX[word] for word in g]), p) for g, p in zip(EXCLUSIVE_GROUPS, probs)]
            pairs = find_twins(frame.image.iloc[idx], [caption[i] for i in idx], frame.category.iloc[idx])
            if len(pairs)*2 != len(idx):
                raise ValueError(f"Cannot match mask executor rows for {name}")
            scores[idx] = sample_scene_scores([parse_program(caption[i]) for i in idx], pairs, np.array(conf),
                                             attrs, np.array(positions), np.zeros(7), seed=421, draws=256)
            if args.split == "val":
                for box, certainty, slot in zip(boxes, conf, zip(*(p.argmax(1) for p in probs))):
                    labels = [g[c] for g, c in zip(EXCLUSIVE_GROUPS, slot)]
                    records.append(dict(image=name, box=box, confidence=certainty,
                                        size=labels[0], color=labels[1], material=labels[2], shape=labels[3]))
                packed_masks.extend(np.packbits(np.stack(kept).reshape(len(kept), -1), axis=1))
    np.savez_compressed(args.out / f"mask_{args.split}.npz", ids=frame.id.to_numpy(), score=scores)
    if args.split == "val":
        (args.out / "objects.json").write_text(json.dumps(records), encoding="utf-8")
        packed = np.asarray(packed_masks, dtype=np.uint8).reshape(-1, 1200)
        np.savez_compressed(args.out / "visible_masks.npz", packed=packed, shape=[80, 120])
    return dict(decode_cpu_s=decode_s, rows=len(frame), batch=batch, probes=probes)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("fit", "predict"))
    cli.add_argument("--data", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--review", type=Path)
    cli.add_argument("--split", choices=("val", "test"), default="val")
    cli.add_argument("--epochs", type=int, default=12)
    cli.add_argument("--seed", type=int, default=421)
    cli.add_argument("--limit", type=int)
    cli.add_argument("--batch", type=int)
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    _, device = detail.v.setup(args.seed)
    info = fit(args, device) if args.stage == "fit" else infer(args, device)
    info["elapsed_s"] = time.perf_counter()-start
    (args.out / f"{args.stage}_{args.split}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
