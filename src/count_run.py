"""Train-only dense count ablations against fixed second-level research anchors."""

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .config import load_config
from .counter import ACTIVE, CountQueries, DenseCounter, freeze_statistics
from .counter_data import CounterData
from .data import Scenes
from .detectors import Detector
from .foundations import aligned_score, check_splits, read_bundle
from .metrics import compare, metrics
from .runtime import (
    Monitor,
    Progress,
    Sink,
    gradient_step,
    immutable_inputs,
    load_state,
    profile_batches,
    setup,
    zero_loss,
)


def make_counter(cfg, inputs, root, manifest, device):
    source = cfg["count"]["source"]
    if source == "historical":
        settings = cfg["model"]
        state = load_state(inputs / "reference/r50_img.pt")
    else:
        item = manifest["models"][source]
        settings = item["config"]["model"]
        checkpoint = load_state(root / item["checkpoint"])
        if checkpoint["config"] != item["config"]:
            raise ValueError("Counter source config differs from manifest")
        state = checkpoint["model"]
    if settings["neck"] != "sum" or settings["width"] != 64:
        raise ValueError("Counter requires stride-four sum pyramid")
    net = Detector(settings)
    net.load_state_dict(state)
    return DenseCounter(net, cfg["count"]["density"] == "joint").to(
        device, memory_format=torch.channels_last
    )


def objective(model, data, queries, idx, cfg):
    c = cfg["count"]
    stats = torch.zeros(3, device=idx.device)
    if c["paired"]:
        before, after, target, stats = data.pair(idx, c["difficulty"], c["attempts"])
        hist, _ = model(torch.cat((before, after)).contiguous(memory_format=torch.channels_last))
        original, changed = hist.chunk(2)
        difference = changed - original
        paired = (
            (F.smooth_l1_loss(difference, target, reduction="none", beta=0.15) * (1 + 10 * target))
            .sum(1)
            .mean()
        )
    else:
        original, _ = model(
            (data.rgb[idx].float() / 255).contiguous(memory_format=torch.channels_last)
        )
        paired = zero_loss(original)
    factual, absolute = queries.loss(original, idx)
    return factual + c["absolute_weight"] * absolute + c["pair_weight"] * paired, stats


def train(cfg, model, data, queries, device, steps=None):
    freeze_statistics(model)
    extra = [p for name, p in model.named_parameters() if not name.startswith("net.")]
    optim = torch.optim.AdamW(
        [
            dict(params=model.net.parameters(), lr=cfg["train"]["rate"]),
            dict(params=extra, lr=cfg["train"]["head_rate"]),
        ]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=128.0)
    n = len(data.rgb)
    effective = cfg["train"]["effective_batch"]

    def probe(size):
        try:
            scaler.scale(
                objective(model, data, queries, torch.arange(size, device=device) % n, cfg)[0]
            ).backward()
        finally:
            optim.zero_grad(set_to_none=True)

    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        batch, profile = profile_batches(
            device, tuple(b for b in (24, 48, effective) if effective % b == 0), n, probe
        )
    if effective % batch:
        batch = 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, cfg["train"]["epochs"])
    updates = skipped = consecutive = 0
    history = []
    started = time.monotonic()
    progress = Progress("count/train", cfg["train"]["epochs"] * n, device)
    for epoch in range(cfg["train"]["epochs"]):
        if time.monotonic() - started > 60 * cfg["train"]["max_minutes"]:
            raise TimeoutError("Counter exceeded training budget")
        order = torch.randperm(n, device=device)
        losses = []
        statistics = torch.zeros(3, device=device)
        pending_loss = torch.zeros((), device=device)
        seen = 0
        for first in range(0, n, batch):
            idx = order[first : first + batch]
            loss, stats = objective(model, data, queries, idx, cfg)
            budget = min(effective, n - first + first % effective)
            scaler.scale(loss * (len(idx) / budget)).backward()
            losses.append(loss.detach())
            pending_loss += loss.detach().float()
            statistics += stats
            seen += len(idx)
            if seen % effective == 0 or seen == n:
                finite, _ = gradient_step(
                    scaler, optim, list(model.parameters()), loss=pending_loss
                )
                pending_loss.zero_()
                skipped += not finite
                consecutive = 0 if finite else consecutive + 1
                updates += finite
                if not finite:
                    print(
                        json.dumps(
                            dict(stage="count/amp_skip", skipped=skipped, scale=scaler.get_scale())
                        ),
                        flush=True,
                    )
                if consecutive >= 8:
                    raise FloatingPointError("Repeated counter gradient overflows")
                if steps and updates >= steps:
                    break
            progress.update(epoch * n + seen, batch=batch)
        scheduler.step()
        accepted, ratios, pixels = statistics.cpu().tolist()
        row = dict(
            epoch=epoch + 1,
            loss=float(torch.stack(losses).mean().cpu()),
            updates=updates,
            accepted_fraction=accepted / seen,
            mean_visible_fraction=ratios / max(1, accepted),
            mean_visible_pixels=pixels / max(1, accepted),
        )
        history.append(row)
        print(json.dumps(dict(stage="count/epoch", **row)), flush=True)
        if steps and updates >= steps:
            break
    return dict(
        seconds=time.monotonic() - started,
        updates=updates,
        amp_skipped_updates=skipped,
        physical_batch=batch,
        effective_batch=effective,
        batch_profile=profile,
        history=history,
        bn_statistics="frozen",
    )


@torch.inference_mode()
def infer(model, scenes, queries, device, batch):
    model.eval()
    histograms = []
    maps = []
    progress = Progress("val/count", len(scenes.names), device)
    if device.type == "cpu":
        batch = min(batch, 4)
    for first in range(0, len(scenes.names), batch):
        images = scenes.images(np.arange(first, min(first + batch, len(scenes.names))), device)
        hist, mass = model(images)
        histograms.append(hist)
        maps.append(mass.half().cpu().numpy())
        progress.update(first + len(images))
    hist = torch.cat(histograms)
    score, absolute, counts = queries.score(hist)
    if not np.isfinite(score).all() or not torch.isfinite(hist).all():
        raise FloatingPointError("Non-finite counter predictions")
    return score, hist.cpu().numpy(), np.concatenate(maps), absolute, counts


def execute(
    config,
    data,
    inputs,
    foundations,
    work,
    output,
    seed=None,
    limit=None,
    steps=None,
    verified_inputs=False,
    prepare_only=False,
):
    cfg = load_config(config)
    if cfg["family"] != "count":
        raise ValueError("Count runner requires count family")
    if seed is not None:
        cfg["train"]["seed"] = seed
    _, device = setup(cfg["train"]["seed"])
    inputs, input_manifest = immutable_inputs(inputs, verify=not verified_inputs)
    root, manifest = read_bundle(foundations, verify=not verified_inputs)
    scratch = Path(work).resolve()
    if "artefacts" in scratch.parts:
        raise ValueError("Count scratch cannot be output artifacts")
    train_scenes, val = (Scenes(data, s, limit) for s in ("train", "val"))
    check_splits(manifest, train_scenes, val)
    if set(train_scenes.names) & set(val.names):
        raise ValueError("Train and val overlap")
    anchors = {}
    for name, file in manifest.get("anchors", {}).items():
        if file not in manifest["files"]:
            raise ValueError("Unregistered anchor file")
        anchors[name] = aligned_score(root, file, val.frame)
    if cfg["count"]["primary"] not in anchors:
        raise ValueError("Attach the fixed L2 anchors")
    if cfg["count"]["source"] != "historical" and cfg["count"]["source"] not in manifest["models"]:
        raise ValueError("Unknown counter source")
    started = time.monotonic()
    with Monitor(device) as monitor:
        prepared = time.monotonic()
        bank = CounterData(root, manifest, train_scenes, scratch, device)
        if set(bank.all_names) & set(val.names):
            raise ValueError("Train donor asset includes validation scenes")
        preparation = time.monotonic() - prepared
        if prepare_only:
            dest = Path(output) / "results/preparation/count.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                json.dumps(
                    dict(seconds=preparation, donors=len(bank.donors), scenes=len(bank.rgb)),
                    indent=2,
                ),
                encoding="utf8",
            )
            return dict(preparation_seconds=preparation)
        setup(cfg["train"]["seed"])
        model = make_counter(cfg, inputs, root, manifest, device)
        queries = CountQueries(train_scenes, device)
        training = train(cfg, model, bank, queries, device, steps)
        del bank, queries
        train_scenes._rgb = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        val.prepare_images(scratch / "images", device)
        queries = CountQueries(val, device)
        branch, hist, mass, absolute, counts = infer(
            model, val, queries, device, cfg["train"]["batch"]
        )
        active = val.frame.category.isin(ACTIVE).to_numpy()
        scores = {}
        blends = {}
        w = cfg["count"]["weight"]
        for name, base in anchors.items():
            candidate = np.where(active, (1 - w) * base + w * branch, base)
            scores[name] = candidate
            blends[name] = dict(
                base=metrics(val.frame, base), counter=metrics(val.frame, candidate)
            )
        primary = cfg["count"]["primary"]
        baseline = anchors[primary]
        candidate = scores[primary]
        report = dict(
            config=cfg,
            smoke=bool(limit or steps),
            device=str(device),
            inputs=input_manifest,
            foundations=manifest,
            fit_scene_names=train_scenes.names.tolist(),
            preparation_seconds=preparation,
            training=training,
            metrics=metrics(val.frame, candidate),
            reference_metrics=metrics(val.frame, baseline),
            blends=blends,
            branch_metrics=metrics(val.frame.loc[active].reset_index(drop=True), branch[active]),
            branch_scope=ACTIVE,
            absolute_count=absolute,
            comparison=compare(val.frame, baseline, candidate, 0),
            evaluation="train-only dense counter; repeatedly used validation; frozen L2 research anchors",
            versions={
                m: importlib.metadata.version(m)
                for m in ("torch", "torchvision", "numpy", "scipy", "pandas", "Pillow")
            },
            source_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob("*.py")
            },
            validation_csv_sha256=hashlib.sha256((Path(data) / "val.csv").read_bytes()).hexdigest(),
            **monitor.report(),
        )
    elapsed = time.monotonic() - started
    report.update(
        elapsed_seconds=elapsed,
        within_50_minutes=elapsed <= 3000,
        timing_scope="one counter with frozen L2 inputs; not final full fresh training and test",
    )
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1000000:06d}"
    name = f"{cfg['name']}/{cfg['train']['seed']}/{stamp}"
    run = scratch / name
    sink = Sink(output, name, run)
    sink.state(
        "model.pt",
        dict(
            config=cfg,
            source_model=model.net.settings,
            model={k: v.cpu() for k, v in model.state_dict().items()},
        ),
    )
    sink.json("report.json", report)
    sink.npz("branch.npz", ids=val.frame.id.to_numpy(), score=branch, active=active)
    sink.npz("val.npz", ids=val.frame.id.to_numpy(), score=candidate, reference=baseline)
    sink.npz("blends.npz", ids=val.frame.id.to_numpy(), **scores)
    sink.npz(
        "counts.npz",
        names=val.names,
        hist=hist,
        mass=mass,
        query_rows=queries.ids,
        query_counts=counts,
    )
    errors = val.frame.copy()
    errors["reference"] = baseline
    errors["candidate"] = candidate
    errors = errors[errors.image.isin(val.frame.image[(candidate <= 0) & val.frame.label.eq(1)])]
    sink.write("results", "errors.csv", errors.to_csv(index=False).encode("utf8"))
    pairs = (
        val.frame.reset_index()
        .pivot(index=["image", "pair_id"], columns="label", values="index")[[1, 0]]
        .to_numpy()
    )
    left, right = pairs.T
    before = baseline[left] - baseline[right]
    after = candidate[left] - candidate[right]
    selected = np.sign(before) != np.sign(after)
    changes = val.frame.iloc[left[selected]][["image", "category", "pair_id", "caption"]].copy()
    changes["opposite"] = val.frame.caption.iloc[right[selected]].to_numpy()
    changes["reference_margin"] = before[selected]
    changes["candidate_margin"] = after[selected]
    sink.write("results", "changes.csv", changes.to_csv(index=False).encode("utf8"))
    sink.finish(include_weights=False)
    receipt = dict(
        name=name, metrics=report["metrics"], elapsed_seconds=elapsed, smoke=report["smoke"]
    )
    (run / "receipt.json").write_text(json.dumps(receipt), encoding="utf8")
    print(json.dumps(receipt), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "inputs", "foundations", "work", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--limit", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--verified-inputs", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    execute(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
