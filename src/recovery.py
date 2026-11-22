"""Train and evaluate object recovery, reusing frozen research foundations."""

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from .config import load_config
from .data import Scenes
from .engine import predict
from .foundations import aligned_score, baselines, check_splits, read_bundle
from .heads import GROUPS
from .losses import caption_loss
from .metrics import compare, metrics
from .posterior import sample_scene_scores
from .recovery_bank import TableStore, prepare_bank
from .runtime import (
    Monitor,
    Progress,
    Sink,
    gradient_step,
    immutable_inputs,
    load_state,
    profile_batches,
    setup,
)
from .union import PHASES, UnionHead, rescue_tables


def training_phases(policy, count, seed=7):
    return (
        np.random.default_rng(seed).integers(4, size=count)
        if policy != "zero"
        else np.zeros(count, np.int64)
    )


def train_head(cfg, scenes, store, inputs, device, steps=None, fixed_batch=None):
    seed = cfg["train"]["seed"]
    rng, _ = setup(seed)  # Cache misses must not change initialization or dropout RNG.
    teacher = load_state(inputs / "reference/scene_silhouette.pt")
    teacher["relation_offset"].zero_()
    u = cfg["union"]
    head = UnionHead(u["relational"], u["roi"], teacher).to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=cfg["train"]["rate"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, cfg["train"]["epochs"], eta_min=0.0001
    )
    phases = training_phases(u["phase"], len(scenes.names))
    effective = cfg["train"]["effective_batch"]

    def probe(n):
        idx = np.arange(n) % len(scenes.names)
        try:
            caption_loss(head, store.select(idx, phases[idx]), scenes.batch(idx, device)).backward()
        finally:
            optim.zero_grad(set_to_none=True)

    if fixed_batch is not None:
        batch, profile = fixed_batch, []
    else:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            batch, profile = profile_batches(device, (24, 48, effective), len(scenes.names), probe)
    started = time.monotonic()
    updates = 0
    history = []
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    progress = Progress("union/train", cfg["train"]["epochs"] * len(scenes.names), device)
    for epoch in range(cfg["train"]["epochs"]):
        if time.monotonic() - started > cfg["train"]["max_minutes"] * 60:
            raise TimeoutError("Union training exceeded its configured budget")
        if u["phase"] == "epoch":
            phases = training_phases("scene", len(scenes.names), seed + epoch)
        order = rng.permutation(len(scenes.names))
        losses = []
        seen = 0
        for first in range(0, len(order), batch):
            idx = order[first : first + batch]
            packed = scenes.batch(idx, device)
            loss = caption_loss(head, store.select(idx, phases[idx]), packed)
            budget = min(effective, len(order) - first + first % effective)
            (loss * (len(idx) / budget)).backward()
            losses.append(loss.detach())
            seen += len(idx)
            if seen % effective == 0 or seen == len(order):
                gradient_step(scaler, optim, list(head.parameters()), loss=loss)
                updates += 1
                if steps and updates >= steps:
                    break
            progress.update(epoch * len(order) + seen, batch=batch)
        schedule.step()
        value = float(torch.stack(losses).mean().cpu())
        history.append(value)
        print(
            json.dumps(dict(stage="union/epoch", epoch=epoch + 1, loss=value, updates=updates)),
            flush=True,
        )
        if steps and updates >= steps:
            break
    return head, dict(
        seconds=time.monotonic() - started,
        updates=updates,
        history=history,
        physical_batch=batch,
        effective_batch=effective,
        batch_profile=profile,
        training_phase_policy=u["phase"],
        phase_assignment_seed=7,
        precision="FP32 head and executor",
    )


def rescue_scores(path, scenes, draws):
    dest = path / f"rescue_{draws}.npz"
    with FileLock(str(dest) + ".lock"):
        if dest.exists():
            with np.load(dest) as z:
                if not np.array_equal(z["ids"], scenes.frame.id):
                    raise ValueError("Rescue score order mismatch")
                return z["control"].copy(), z["delta"].copy(), int(z["affected"])
        left = np.load(path / "source.npy", mmap_mode="r")[0]
        right = np.load(path / "secondary.npy", mmap_mode="r")
        offsets = np.load(path / "offsets.npy")
        control = np.zeros(len(scenes.frame))
        delta = np.zeros_like(control)
        affected = 0
        progress = Progress("val/rescue", len(scenes.names))
        for i, rows in enumerate(scenes.rows):
            old, new = rescue_tables(left[i], right[i])

            def score(table):
                p, xy, attrs, valid = table
                return sample_scene_scores(
                    [scenes.programs[j] for j in rows],
                    scenes._pairs_local[i],
                    p[valid],
                    [(g, a[valid]) for g, a in zip(GROUPS, attrs)],
                    xy[valid],
                    offsets,
                    42,
                    draws,
                )

            control[rows] = score(old)
            if not (np.array_equal(old[0], new[0]) and np.array_equal(old[3], new[3])):
                delta[rows] = score(new) - control[rows]
                affected += 1
            progress.update(i + 1)
        with dest.with_suffix(".tmp").open("wb") as stream:
            np.savez(
                stream,
                ids=scenes.frame.id.to_numpy(),
                control=control,
                delta=delta,
                affected=affected,
            )
        dest.with_suffix(".tmp").replace(dest)
        return control, delta, affected


def evaluate(cfg, scenes, head, store, path, foundations):
    views = np.stack(
        [
            predict(
                scenes, head, store.phase(i), store.fields["x"].device, cfg["evaluation"]["draws"]
            )
            for i in range(4)
        ]
    )
    control, delta, affected = rescue_scores(path, scenes, cfg["evaluation"]["draws"])
    blends = {}
    scores = {}
    w = cfg["union"]["weight"]
    r = cfg["union"]["rescue_weight"]
    for name, base in foundations.items():
        variants = {"base": base, "rescue": base + r * delta}
        for view, u in [("plain", views[0]), ("phases", views.mean(0))]:
            variants[view] = (1 - w) * base + w * u
            variants[view + "_rescue"] = (1 - w) * (base + r * delta) + w * u
        scores[name] = variants
        blends[name] = {k: metrics(scenes.frame, v) for k, v in variants.items()}
    return views, delta, control, affected, blends, scores


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
    if cfg["family"] != "union":
        raise ValueError("Recovery runner requires a union configuration")
    if seed is not None:
        cfg["train"]["seed"] = seed
    _, device = setup(cfg["train"]["seed"])
    inputs, input_manifest = immutable_inputs(inputs, verify=not verified_inputs)
    foundation_root, manifest = read_bundle(foundations, verify=not verified_inputs)
    scratch = Path(work).resolve()
    if "artefacts" in scratch.parts:
        raise ValueError("Scratch cannot be artifact output")
    train, val = (Scenes(data, s, limit) for s in ("train", "val"))
    if set(train.names) & set(val.names):
        raise ValueError("Train and validation scenes overlap")
    check_splits(manifest, train, val)
    source = cfg["union"]["source"]
    if source != "historical" and source not in manifest["models"]:
        raise ValueError(f"Foundation source is not registered: {source}")
    if cfg["union"]["primary"] not in {"historical", *manifest["baselines"]}:
        raise ValueError("Primary baseline is not registered")
    start = time.monotonic()
    paths = {}
    preparation = {}
    with Monitor(device) as monitor:
        for scenes in (train, val):
            paths[scenes.split], preparation[scenes.split] = prepare_bank(
                cfg, scenes, inputs, input_manifest, foundation_root, manifest, scratch, device
            )
            scenes._rgb = None  # Images are no longer needed by this split or the head stage.
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if prepare_only:
            rescue_started = time.monotonic()
            _, _, affected = rescue_scores(paths["val"], val, cfg["evaluation"]["draws"])
            dest = Path(output) / "results" / "preparation" / (cfg["union"]["source"] + ".json")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                json.dumps(
                    dict(
                        config=cfg["name"],
                        elapsed_seconds=time.monotonic() - start,
                        splits=preparation,
                        rescue_seconds=time.monotonic() - rescue_started,
                        rescue_affected_scenes=affected,
                        **monitor.report(),
                    ),
                    indent=2,
                ),
                encoding="utf8",
            )
            return preparation
        store = TableStore(paths["train"], train, device, cfg["union"]["roi"])
        head, training = train_head(cfg, train, store, inputs, device, steps)
        del store
        if device.type == "cuda":
            torch.cuda.empty_cache()
        store = TableStore(paths["val"], val, device, cfg["union"]["roi"])
        historical = aligned_score(inputs / "reference", "r50_img_val.npz", val.frame)
        sigft = aligned_score(inputs / "reference", "sigft_val.npz", val.frame)
        bases = baselines(foundation_root, manifest, val.frame, historical, sigft)
        views, delta, control, affected, blends, scores = evaluate(
            cfg, val, head, store, paths["val"], bases
        )
        source = cfg["union"]["source"]
        expected = (
            historical
            if source == "historical"
            else aligned_score(foundation_root, manifest["models"][source]["scores"], val.frame)
        )
        replay = dict(
            max_absolute_difference=float(np.max(np.abs(control - expected))),
            mean_absolute_difference=float(np.mean(np.abs(control - expected))),
            comparison=compare(val.frame, expected, control, 0),
        )
        primary = cfg["union"]["primary"]
        candidate = scores[primary]["phases"]
        baseline = bases[primary]
        report = dict(
            config=cfg,
            smoke=bool(limit or steps),
            device=str(device),
            evaluation="repeatedly used historical val, no validation fitting; frozen train-only research foundations",
            fit_scene_names=train.names.tolist(),
            inputs=input_manifest,
            foundations=manifest,
            preparation=preparation,
            training=training,
            blends=blends,
            metrics=blends[primary]["phases"],
            reference_metrics=metrics(val.frame, baseline),
            branch_metrics=metrics(val.frame, views.mean(0)),
            source_control_metrics=metrics(val.frame, control),
            source_replay=replay,
            versions={
                m: importlib.metadata.version(m)
                for m in (
                    "torch",
                    "torchvision",
                    "numpy",
                    "scipy",
                    "pandas",
                    "scikit-learn",
                    "PyYAML",
                )
            },
            rescue_affected_scenes=affected,
            comparison=compare(
                val.frame,
                baseline,
                candidate,
                cfg["evaluation"]["bootstrap"] if not (limit or steps) else 0,
            ),
            phases=PHASES,
            source_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob("*.py")
            },
            validation_csv_sha256=hashlib.sha256((Path(data) / "val.csv").read_bytes()).hexdigest(),
            dependencies=dict(
                union_source=cfg["union"]["source"],
                secondary="thaw",
                roi_reuses_source_map=cfg["union"]["roi"],
                baseline_blends=manifest["baselines"],
            ),
            **monitor.report(),
        )
    elapsed = time.monotonic() - start
    report.update(
        elapsed_seconds=elapsed,
        within_50_minutes=elapsed <= 3000,
        timing_scope="head research with frozen foundations; preparation is separate and shared; not final fresh train+test",
    )
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1000000:06d}"
    name = f"{cfg['name']}/{cfg['train']['seed']}/{stamp}"
    run = scratch / name
    sink = Sink(output, name, run)
    sink.state(
        "model.pt", dict(config=cfg, head={k: v.cpu() for k, v in head.state_dict().items()})
    )
    sink.json("report.json", report)
    sink.npz("branch.npz", ids=val.frame.id.to_numpy(), score=views.mean(0))
    sink.npz(
        "views.npz", ids=val.frame.id.to_numpy(), scores=views, delta=delta, source_control=control
    )
    sink.npz("val.npz", ids=val.frame.id.to_numpy(), score=candidate, reference=baseline)
    sink.npz(
        "blends.npz",
        ids=val.frame.id.to_numpy(),
        **{n + "__" + k: v for n, variants in scores.items() for k, v in variants.items()},
    )
    bad = val.frame.copy()
    bad["reference"] = baseline
    bad["candidate"] = candidate
    bad = bad[bad.image.isin(val.frame.image[(candidate <= 0) & val.frame.label.eq(1)])]
    sink.write("results", "errors.csv", bad.to_csv(index=False).encode("utf8"))
    pairs = (
        val.frame.reset_index()
        .pivot(index=["image", "pair_id"], columns="label", values="index")[[1, 0]]
        .to_numpy()
    )
    a, b = pairs.T
    before = baseline[a] - baseline[b]
    after = candidate[a] - candidate[b]
    changed = np.sign(before) != np.sign(after)
    changes = val.frame.iloc[a[changed]][["image", "category", "pair_id", "caption"]].copy()
    changes["opposite"] = val.frame.caption.iloc[b[changed]].to_numpy()
    changes["reference_margin"] = before[changed]
    changes["candidate_margin"] = after[changed]
    sink.write("results", "changes.csv", changes.to_csv(index=False).encode("utf8"))
    # Compact inspectable object output; ROI is a shared cache, not another repeated output blob.
    table = store.phase(0)
    sink.npz(
        "objects.npz",
        names=val.names,
        features=table[0][..., :57].cpu().numpy(),
        positions=table[1].cpu().numpy(),
        valid=table[2].cpu().numpy(),
    )
    sink.finish()
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
    a = p.parse_args()
    execute(**vars(a))


if __name__ == "__main__":
    main()
