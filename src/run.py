"""Run one registered foundation experiment. Never fit on validation or test."""

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np

from .config import load_config
from .crops import train_siglip
from .data import Scenes, Teacher
from .engine import detector_bank, predict, proposal_tables, train_resnet
from .metrics import compare, metrics
from .runtime import Monitor, Sink, immutable_inputs, setup


def reference(inputs, frame, name):
    with np.load(inputs / "reference" / f"{name}_val.npz") as data:
        if not np.array_equal(data["ids"], frame.id):
            raise ValueError("Reference row order differs from validation CSV")
        return data["score"].copy()


def execute(
    config, data, inputs, work, output, seed=None, limit=None, steps=None, verified_inputs=False
):
    cfg = load_config(config)
    if seed is not None:
        cfg["train"]["seed"] = seed
    seed = cfg["train"]["seed"]
    rng, device = setup(seed)
    inputs, manifest = immutable_inputs(inputs, verify=not verified_inputs)
    work = Path(work).resolve()
    if "artefacts" in work.parts:
        raise ValueError("Scratch must be separate from output-only artifacts")
    scratch_root = work
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1000000:06d}"
    name = f"{cfg['name']}/{seed}/{stamp}"
    work = work / name
    work.mkdir(parents=True, exist_ok=False)
    sink = Sink(output, name, work)
    start = time.monotonic()
    val = Scenes(data, "val", limit)
    # Immutable references are row-aligned before optional smoke-test subsampling.
    import pandas as pd

    full = pd.read_csv(Path(data) / "val.csv")
    pos = full.set_index("id").index.get_indexer(val.frame.id)
    r50 = reference(inputs, full, "r50_img")[pos]
    sigft = reference(inputs, full, "sigft")[pos]
    baseline = 0.5 * (r50 + sigft)
    report = dict(
        config=cfg,
        inputs=manifest,
        smoke=bool(limit or steps),
        reference_metrics=metrics(val.frame, baseline),
        evaluation="official historical val; repeatedly used for development, not a new holdout",
        device=str(device),
        versions={
            m: importlib.metadata.version(m)
            for m in (
                "torch",
                "torchvision",
                "transformers",
                "safetensors",
                "numpy",
                "scipy",
                "pandas",
                "scikit-learn",
                "Pillow",
                "PyYAML",
            )
        },
    )
    report["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    report["validation_csv_sha256"] = hashlib.sha256(
        (Path(data) / "val.csv").read_bytes()
    ).hexdigest()
    with Monitor(device) as monitor:
        if cfg["family"] == "base":
            candidate = baseline
            report["branches"] = {
                key: metrics(val.frame, value)
                for key, value in [("resnet", r50), ("siglip", sigft)]
            }
        else:
            train = Scenes(data, "train", limit)
            if set(train.names) & set(val.names):
                raise ValueError("Training and validation scenes overlap")
            train.prepare_images(scratch_root / "images", device)
            val.prepare_images(scratch_root / "images", device)
            teacher = Teacher(inputs / "teacher/train.npz", train)
            report["fit_scene_names"] = train.names.tolist()
            if cfg["family"] == "resnet":
                model, head, training = train_resnet(
                    cfg, train, teacher, inputs, device, rng, steps
                )
                bank = detector_bank(
                    model, val, device, cfg["train"]["batch"], cfg["model"]["topk"]
                )
                branch = predict(val, head, bank, device, cfg["evaluation"]["draws"])
                candidate = (
                    cfg["evaluation"]["blend"] * branch + (1 - cfg["evaluation"]["blend"]) * sigft
                )
                sink.state(
                    "model.pt",
                    dict(
                        config=cfg,
                        model={k: v.cpu() for k, v in model.state_dict().items()},
                        head={k: v.cpu() for k, v in head.state_dict().items()},
                    ),
                )
            else:
                tick = time.monotonic()
                source = cfg["siglip"]["proposals"]
                train_table = proposal_tables(
                    train, inputs, cfg["model"], source, device, cfg["train"]["batch"]
                )
                val_table = proposal_tables(
                    val, inputs, cfg["model"], source, device, cfg["train"]["batch"]
                )
                vision, head, bank, projection, training = train_siglip(
                    cfg,
                    train,
                    val,
                    teacher,
                    train_table,
                    val_table,
                    inputs,
                    work,
                    device,
                    rng,
                    steps,
                    cache_lock=scratch_root / "pool.lock",
                )
                training["including_proposals_seconds"] = time.monotonic() - tick
                branch = predict(val, head, bank, device, cfg["evaluation"]["draws"])
                candidate = (
                    cfg["evaluation"]["blend"] * r50 + (1 - cfg["evaluation"]["blend"]) * branch
                )
                sink.state(
                    "model.pt",
                    dict(
                        config=cfg,
                        vision=vision.adaptation_state(),
                        projection=projection.state(),
                        head={k: v.cpu() for k, v in head.state_dict().items()},
                    ),
                )
            report["training"] = training
            report["cpu_image_read_seconds"] = dict(
                train=train.image_read_seconds, val=val.image_read_seconds
            )
            report["branch_metrics"] = metrics(val.frame, branch)
            report["comparison"] = compare(
                val.frame,
                baseline,
                candidate,
                0 if report["smoke"] else cfg["evaluation"]["bootstrap"],
            )
            sink.npz("branch.npz", ids=val.frame.id.to_numpy(), score=branch)
        report["metrics"] = metrics(val.frame, candidate)
    report.update(monitor.report())
    report["elapsed_seconds"] = time.monotonic() - start
    report["within_50_minutes"] = report["elapsed_seconds"] <= 3000
    sink.json("report.json", report)
    sink.npz("val.npz", ids=val.frame.id.to_numpy(), score=candidate, reference=baseline)
    failures = val.frame.copy()
    failures["reference"] = baseline
    failures["candidate"] = candidate
    failed = (candidate <= 0) & val.frame.label.eq(1)
    names = set(val.frame.image[failed])
    failures = failures[failures.image.isin(names)]
    sink.write("results", "errors.csv", failures.to_csv(index=False).encode("utf-8"))
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
    sink.write("results", "changes.csv", changes.to_csv(index=False).encode("utf-8"))
    sink.finish()
    # The scheduler reads this scratch receipt, never the artifact sink.
    (work / "receipt.json").write_text(
        json.dumps(
            dict(
                name=name,
                metrics=report["metrics"],
                elapsed_seconds=report["elapsed_seconds"],
                smoke=report["smoke"],
            )
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            dict(
                name=name,
                metrics=report["metrics"],
                elapsed_seconds=report["elapsed_seconds"],
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--inputs", type=Path, default=Path("inputs"))
    p.add_argument("--work", type=Path, default=Path(".work"))
    p.add_argument("--output", type=Path, default=Path("artefacts"))
    p.add_argument("--seed", type=int)
    p.add_argument("--limit", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument(
        "--verified-inputs",
        action="store_true",
        help="Parent already verified the immutable manifest",
    )
    a = p.parse_args()
    execute(
        a.config, a.data, a.inputs, a.work, a.output, a.seed, a.limit, a.steps, a.verified_inputs
    )


if __name__ == "__main__":
    main()
