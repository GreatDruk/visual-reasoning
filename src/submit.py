"""Shared, frozen train-only inference for the two explicit submission candidates."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch.nn import functional as F

from .config import load_config
from .counter import CountQueries, DenseCounter
from .data import Scenes
from .detectors import Detector, dense_table
from .engine import detector_bank, predict
from .evidence import compose, hidden_classes, placement_prior
from .foundations import aligned_score, check_splits, read_bundle
from .heads import GROUPS, TableHead, load_head
from .metrics import metrics
from .posterior import sample_scene_scores
from .recovery_bank import load_sources
from .runtime import Progress, amp, immutable_inputs, load_state, setup
from .union import UnionHead, detector_rows, merge, rescue_tables, undo_shift


def counter_dependencies(configs):
    roles = set()
    for cfg in configs:
        settings = cfg["blend"]
        if settings["counter_weight"]:
            roles.update(
                (
                    settings.get("count_primary", "count54_421"),
                    settings.get("count_secondary", "count54_853"),
                )
            )
        if settings["swap"]:
            source = settings.get("swap_source", "count53")
            roles.update((source + "_421", source + "_853"))
    return sorted(roles)


def component_path(work, split, job, counter_roles):
    suffix = "_" + "_".join(sorted(counter_roles)) if job == "count" else ""
    return work / f"{split}_{job}{suffix}.npz"


def restore_rows(rows, shifted, flipped):
    rows = rows.clone()
    if flipped:
        rows[..., -6] = 1 - rows[..., -6] - rows[..., -4]
        rows[..., -2] = -rows[..., -2]
    return undo_shift(rows, (2, 2) if shifted else (0, 0))


def row_scores(scenes, rows, offsets):
    score = np.zeros(len(scenes.frame))
    progress = Progress(scenes.split + "/paired_evidence", len(scenes.names))
    for i, indices in enumerate(scenes.rows):
        table = rows[i]
        keep = table[:, 0] >= -2.442347
        attrs = np.split(np.exp(table[keep, 2:17]), [2, 10, 12], axis=1)
        score[indices] = sample_scene_scores(
            [scenes.programs[j] for j in indices],
            scenes._pairs_local[i],
            expit(table[keep, 1]),
            list(zip(GROUPS, attrs)),
            table[keep, -2:],
            offsets,
            42,
            1024,
        )
        progress.update(i + 1)
    return score


@torch.inference_mode()
def visual(scenes, inputs, root, manifest, device, project):
    item = manifest["models"]["convnext"]
    checkpoint = load_state(root / item["checkpoint"])
    model = Detector(checkpoint["config"]["model"]).to(device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    head = TableHead().to(device)
    load_head(head, checkpoint["head"])
    bank = detector_bank(model, scenes, device)
    result = {"convnext": predict(scenes, head, bank, device)}
    del model, head, bank, checkpoint
    cfg = load_config(project / "configs/045_union.yaml")
    primary, head, secondary, secondary_head, matrix = load_sources(
        cfg, inputs, root, manifest, device
    )
    left, right = [], []
    progress = Progress(scenes.split + "/r50_tta4", len(scenes.names))
    batch = 96 if device.type == "cuda" else 2
    for first in range(0, len(scenes.names), batch):
        images = scenes.images(np.arange(first, min(first + batch, len(scenes.names))), device)
        with amp(device):
            latent = secondary.features(images)
            raw = secondary.head(latent)
            descriptor = torch.einsum("bdhw,ds->bshw", latent, matrix)
        native = detector_rows(
            secondary_head, dense_table(raw.float(), latent=descriptor.float(), min_logit=-3.663562)
        )
        right.append(native.cpu().numpy())
        views = []
        for shifted, flipped in [(False, False), (False, True), (True, False), (True, True)]:
            pixels = (
                F.pad(images, (2, 0, 2, 0), mode="replicate")[:, :, :160, :240]
                if shifted
                else images
            )
            if flipped:
                pixels = pixels.flip(-1)
            with amp(device):
                raw = primary.head(primary.pyramid(pixels)[0])
            rows = detector_rows(head, dense_table(raw.float(), min_logit=-3.663562))
            views.append(restore_rows(rows, shifted, flipped).cpu().numpy())
        left.append(np.stack(views))
        progress.update(first + len(images), batch=batch)
    left, right = np.concatenate(left, axis=1), np.concatenate(right)
    offsets = head.relation_offset.cpu().numpy()
    del primary, head, secondary, secondary_head
    result["r50_plain"] = row_scores(scenes, left[0], offsets)
    result["r50_tta"] = np.mean(
        [result["r50_plain"], *[row_scores(scenes, a, offsets) for a in left[1:]]], axis=0
    )
    zero = np.zeros((20, 256), np.float16)
    merged = [merge(a, b, zero, zero, cfg["union"])[:3] for a, b in zip(left[0], right)]
    bank = tuple(np.stack([a[j] for a in merged]) for j in range(3))
    for seed in (421, 853):
        checkpoint = load_state(root / manifest["assets"][f"union{seed}"])
        union = UnionHead(True, False).to(device)
        union.load_state_dict(checkpoint["head"], strict=True)
        result[f"union{seed}"] = predict(scenes, union, bank, device)
    rescue = np.zeros(len(scenes.frame))
    progress = Progress(scenes.split + "/rescue", len(scenes.names))
    for i, indices in enumerate(scenes.rows):
        old, new = rescue_tables(left[0, i], right[i])
        if np.array_equal(old[0], new[0]) and np.array_equal(old[3], new[3]):
            continue

        def score(table):
            p, xy, attrs, valid = table
            return sample_scene_scores(
                [scenes.programs[j] for j in indices],
                scenes._pairs_local[i],
                p[valid],
                [(g, a[valid]) for g, a in zip(GROUPS, attrs)],
                xy[valid],
                offsets,
                42,
                1024,
            )

        rescue[indices] = score(new) - score(old)
        progress.update(i + 1)
    result["rescue"] = rescue
    return result


@torch.inference_mode()
def counters(scenes, root, manifest, device, roles=None):
    queries = CountQueries(scenes, device)
    result = {}
    roles = (
        set(roles) if roles is not None else {f"count{n}_{s}" for n in (53, 54) for s in (421, 853)}
    )
    for role in sorted(roles):
        if role not in manifest.get("assets", {}):
            raise ValueError(f"Counter checkpoint role is not registered: {role}")
        checkpoint = load_state(root / manifest["assets"][role])
        model = DenseCounter(
            Detector(checkpoint["source_model"]),
            checkpoint["config"]["count"]["density"] == "joint",
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        model = model.to(device, memory_format=torch.channels_last).eval()
        histograms = []
        batch = 96 if device.type == "cuda" else 2
        progress = Progress(scenes.split + "/" + role, len(scenes.names), device)
        for first in range(0, len(scenes.names), batch):
            images = scenes.images(np.arange(first, min(first + batch, len(scenes.names))), device)
            histograms.append(model(images)[0])
            progress.update(first + len(images), batch=batch)
        hist = torch.cat(histograms)
        result[role] = queries.score(hist)[0]
        del model, checkpoint, histograms, hist
    return result


def write_npz(path, **values):
    pending = path.with_suffix(".tmp")
    with pending.open("wb") as stream:
        np.savez_compressed(stream, **values)
    pending.replace(path)


def worker(project, data, inputs, foundations, work, split, job, limit=None, counter_roles=None):
    _, device = setup(421)
    root, manifest = read_bundle(foundations, verify=False)
    scenes = Scenes(data, split, limit)
    scenes.prepare_images(work / "images", device)
    started = time.monotonic()
    counter_roles = counter_roles.split(",") if counter_roles else []
    allowed = {
        k
        for k in manifest.get("assets", {})
        if k.startswith("count") and k.endswith(("_421", "_853"))
    }
    if not set(counter_roles) <= allowed:
        raise ValueError("Unknown counter dependency")
    values = (
        visual(scenes, inputs, root, manifest, device, project)
        if job == "visual"
        else counters(scenes, root, manifest, device, counter_roles)
    )
    write_npz(
        component_path(work, split, job, counter_roles), ids=scenes.frame.id.to_numpy(), **values
    )
    print(json.dumps(dict(job=job, split=split, seconds=time.monotonic() - started)), flush=True)


def export_csv(frame, sample, score, destination):
    if not frame.id.is_unique or not sample.id.is_unique or set(frame.id) != set(sample.id):
        raise ValueError("Submission IDs must exactly match sample_submission.csv")
    if np.asarray(score).shape != (len(frame),) or not np.isfinite(score).all():
        raise ValueError("Invalid submission scores")
    values = pd.Series(expit(score), index=frame.id).loc[sample.id].to_numpy()
    table = sample.assign(label=values)
    pending = destination.with_suffix(".tmp")
    table.to_csv(pending, index=False, float_format="%.17g")
    pd.testing.assert_frame_equal(pd.read_csv(pending, float_precision="round_trip"), table)
    pending.replace(destination)


def run(
    project,
    data,
    inputs,
    foundations,
    work,
    output,
    selected,
    limit=None,
    splits=("val", "test"),
    gpus=(0, 1),
    submission_dir=None,
):
    project, data, work, output = map(lambda p: Path(p).resolve(), (project, data, work, output))
    if "artefacts" in work.parts:
        raise ValueError("Submission work must be scratch")
    root, manifest = read_bundle(foundations)
    inputs, input_manifest = immutable_inputs(inputs)
    configs = [load_config(project / "configs" / (name + ".yaml")) for name in selected]
    if not configs or any(c["family"] != "submission" for c in configs):
        raise ValueError("Select submission configurations")
    counter_roles = counter_dependencies(configs)
    missing = set(counter_roles) - manifest.get("assets", {}).keys()
    if missing:
        raise ValueError(
            f"Attach the required counter assets or use frozen replay: {sorted(missing)}"
        )
    train, val = Scenes(data, "train"), Scenes(data, "val")
    check_splits(manifest, train, val)
    provenance = json.loads((root / manifest["assets"]["deployment"]).read_text(encoding="utf8"))
    for role, item in provenance["checkpoints"].items():
        if role not in {"union421", "union853", *counter_roles}:
            continue
        if set(item["fit_scene_names"]) != set(train.names) or set(item["fit_scene_names"]) & set(
            val.names
        ):
            raise ValueError("Submission checkpoint provenance must be train-only")
        state = load_state(root / manifest["assets"][role])
        if state["config"] != item["config"]:
            raise ValueError("Deployment checkpoint/config mismatch")
        del state
    with np.load(root / manifest["assets"]["background"]) as z:
        if not set(z["names"]) <= set(train.names) or len(z["names"]) != 128:
            raise ValueError("Background must be estimated from train")
        background = z["background"].copy()
    prior = placement_prior(root / manifest["assets"]["donors"], background)
    digest = hashlib.sha256(
        json.dumps(
            dict(
                files=manifest["files"],
                inputs=input_manifest,
                code={
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (project / "src").glob("*.py")
                },
                data=str(data),
                limit=limit,
                gpus=gpus,
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()[:20]
    work = work / "submissions" / digest
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    submission_dir = Path(submission_dir) if submission_dir is not None else output
    submission_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    reports = {}
    for split in splits:
        scenes = Scenes(data, split, limit)
        scenes.prepare_images(work / "images", torch.device("cpu"))
        hidden = np.stack([hidden_classes(image.numpy(), prior) for image in scenes._rgb])
        scenes._rgb = None

        def launch(job, gpu):
            dest = component_path(work, split, job, counter_roles)
            if dest.exists():
                return
            env = dict(
                os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
            )
            command = [
                sys.executable,
                "-u",
                "-m",
                "src.submit",
                "--job",
                job,
                "--project",
                str(project),
                "--data",
                str(data),
                "--inputs",
                str(inputs),
                "--foundations",
                str(root),
                "--work",
                str(work),
                "--split",
                split,
                "--counter-roles",
                ",".join(counter_roles),
            ]
            if limit:
                command += ["--limit", str(limit)]
            with (work / f"{split}_{job}.log").open("w", encoding="utf8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=project,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(f"[{job} GPU{gpu}] {line.rstrip()}", flush=True)
                if process.wait():
                    raise RuntimeError(
                        f"Submission {job} failed; inspect {work / (split + '_' + job + '.log')}"
                    )

        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(launch, job, gpu) for job, gpu in zip(("visual", "count"), gpus)]
            for future in futures:
                future.result()
        parts = {
            "hidden": hidden,
            "sigft": aligned_score(root, manifest["assets"][f"sigft_{split}"], scenes.frame),
        }
        for job in ("visual", "count"):
            with np.load(component_path(work, split, job, counter_roles)) as z:
                if not np.array_equal(z["ids"], scenes.frame.id):
                    raise ValueError("Submission component IDs differ")
                parts.update({k: z[k].copy() for k in z.files if k != "ids"})
        write_npz(output / f"{split}_components.npz", ids=scenes.frame.id.to_numpy(), **parts)
        reports[split] = {}
        for cfg in configs:
            variants, swaps = compose(scenes, parts, cfg["blend"])
            if any(not np.isfinite(v).all() for v in variants.values()):
                raise FloatingPointError("Non-finite composed submission")
            write_npz(
                output / f"{cfg['name']}_{split}.npz", ids=scenes.frame.id.to_numpy(), **variants
            )
            report = dict(
                config=cfg,
                swapped_pairs=swaps,
                metrics={k: metrics(scenes.frame, v) for k, v in variants.items()}
                if scenes.label is not None
                else None,
            )
            if split == "test" and not limit:
                export_csv(
                    scenes.frame,
                    pd.read_csv(data / "sample_submission.csv"),
                    variants["final"],
                    submission_dir / (cfg["name"] + ".csv"),
                )
            reports[split][cfg["name"]] = report
    reports.update(
        seconds=time.monotonic() - started,
        scope="frozen train-only inference; cached historical SigLIP; not full fresh training",
        provenance=provenance,
        bundle_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
    )
    (output / "submission_report.json").write_text(json.dumps(reports, indent=2), encoding="utf8")
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "data", "inputs", "foundations", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--job", choices=("visual", "count"), required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--counter-roles")
    worker(**vars(parser.parse_args()))
