"""Complete competition train/validation/inference suite for two Kaggle GPUs."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import cv2
import numpy as np
import pandas as pd
from scipy.special import expit
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kaggle_suite_common import (
    CACHE, CSV, DATA, FEATURES, INPUT, N_ATTRS, OUTPUT, REPORT, SCORES, WEIGHTS,
    caption_loss, configure_split, dense_probabilities, fixed_table, joint, load,
    prepare, record, save, say, setup, table_predictions, tick, tune_batch, write_json,
)
from kaggle_suite_crops import check_pool, crop_extract, encode_tuned, features, table, train_pool
from kaggle_suite_detectors import DETECTORS, detector_eval, pasted_recall, train_detector
from learned_center_fusion import Fusion, region_features
from center_proposals import fuse
from mc_scene3d import reconstruct
from scene_ranking import evaluate, fit, hypotheses, pack
from native_centers import GROUPS, HEIGHT, WIDTH
from src.pixel_regions import COLORS, _color_planes
from src.scene_posterior import sample_scene_scores

HISTORICAL_HEADS = {"head_a": 42, "head_b": 73}
CROP_HEADS = {"hog": 721, "sig": 721, "sigft": 721}
OLD = ("centers", "counts", "fusion", "native")
FAMILIES = {
    "baseline": {n: .25 for n in OLD},
    "caption": {"centers": 1/6, "counts": 1/6, "fusion": 1/6, "head_a": .25, "head_b": .25},
    "caption_thaw": {"centers": 1/6, "counts": 1/6, "fusion": 1/6, "thaw": .5},
    "geometry": {"caption": .65, "mc3d": .2, "rank": .15},
    "crop_mix": {"caption": .55, "hog": .15, "sig": .15, "sigft": .15},
    "all": {"caption": .5, "thaw": .08, "mc3d": .10, "occ_paste": .07, "rank": .10,
            "r50_img": .06, "hog": .03, "sig": .03, "sigft": .03},
}
CONFIGS = (*OLD, *HISTORICAL_HEADS, *DETECTORS, *CROP_HEADS,
           "mc_control", "mc3d", "draws", "rank", *FAMILIES)


def fusion_model(name):
    teacher = load(INPUT / "scene_silhouette.pt")
    if name != "fusion":
        teacher["relation_offset"].zero_()
    model = Fusion(teacher)
    if name in CROP_HEADS:
        model.residual[0] = nn.Linear(FEATURES + 128, 48)
    return model


def fit_head(name):
    seed = (HISTORICAL_HEADS | CROP_HEADS)[name]
    rng, device = setup(seed)
    train = configure_split("train", device, fitting=True)
    source = features("train", name, True) if name in CROP_HEADS else table("native", "train")
    model = fusion_model(name).to(device)
    if name in CROP_HEADS:
        old = load(WEIGHTS / "thaw_head.pt")
        with torch.no_grad():
            model.residual[0].weight[:, :FEATURES].copy_(old.pop("residual.0.weight"))
            model.residual[0].weight[:, FEATURES:].zero_()
        model.load_state_dict(old, strict=False)
    epochs = 10 if name in CROP_HEADS else 6
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=.0003)
    val = configure_split("val", device) if name in HISTORICAL_HEADS else None
    val_table = table("native", "val") if val is not None else None
    from scripts.pixel_reasoning import measures, predict
    best = -np.inf
    start = time.monotonic()
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        for scenes, data in train.batches(64 if val is not None else 128, rng):
            tensor = [torch.from_numpy(z[scenes]).to(device) for z in source]
            loss = caption_loss(model, tensor, train, data)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            opt.step()
            losses.append(float(loss.detach()))
        sch.step()
        item = {"loss": float(np.mean(losses))}
        if val is not None:
            # The previous strong head used the best forward-pass val AUC for selection.
            val.data.table = tuple(torch.from_numpy(a) for a in val_table)
            auc = measures(predict(model, val.data, 128), val.data)["auc"]
            item["val_selection_auc"] = auc
            if auc > best:
                best = auc
                save(model, name)
        history.append(item)
        if epoch in (1, epochs) or epoch % 3 == 0:
            tick(name, epoch, epochs, start)
    if val is None:
        save(model, name)
    write_json(REPORT / f"{name}_train.json", {"seed": seed, "epochs": epochs, "history": history,
                                               "minutes": (time.monotonic()-start)/60})


def head_eval(split_name, name):
    _, device = setup(42)
    split = configure_split(split_name, device)
    source = features(split_name, name) if name in CROP_HEADS else table("native", split_name)
    model = fusion_model(name).to(device)
    model.load_state_dict(load(WEIGHTS / f"{name}.pt"))
    bank = table_predictions(model, source, device)
    # Retain the original Sobol seeds for the two previous configurations.
    record(name, split, joint(split, bank, model.relation_offset.detach().cpu().numpy(),
                              seed=HISTORICAL_HEADS.get(name, 42)))


def baseline(split_name):
    _, device = setup(42)
    split = configure_split(split_name, device)
    (p, xy, valid, *attrs), boxes, offsets = fixed_table(split_name, "seam", device)
    raw8 = table("center8", split_name)
    dense = dense_probabilities(raw8)
    values = np.empty(len(split.data.ids))
    for i, (rows, pairs) in enumerate(zip(split.data.rows, split.scene_pairs)):
        keep = valid[i]
        learned = evidence(raw8, dense, i)
        pp, position, attributes = fuse(boxes[i, keep, 8:12] * [WIDTH, HEIGHT, WIDTH, HEIGHT],
                                        p[i, keep], xy[i, keep], [z[i, keep] for z in attrs],
                                        learned, .57, False)
        values[rows] = sample_scene_scores([split.data.parsed[j] for j in rows], pairs, pp,
                                          list(zip(GROUPS, attributes)), position, offsets, 42)
    record("centers", split, values)
    bank, _, offsets = fixed_table(split_name, "counts", device)
    record("counts", split, joint(split, bank, offsets))
    model, source = fused_bank(split_name, "center16", device)
    record("fusion", split, joint(split, table_predictions(model, source, device),
                                  model.relation_offset.detach().cpu().numpy()))


def evidence(table, prob, i, threshold=.56):
    p, xy, valid, *attrs = prob
    sel = np.flatnonzero(valid[i] & (p[i] > threshold))[:15]
    boxes = table[0][i, sel, 1+N_ATTRS:5+N_ATTRS] * [WIDTH, HEIGHT, WIDTH, HEIGHT]
    return p[i, sel], boxes[:, :2] + .5 * boxes[:, 2:], boxes[:, 2:], [a[i, sel] for a in attrs]


def fused_bank(split_name, detector, device):
    raw = table(detector, split_name)
    dense = dense_probabilities(raw)
    (p, xy, valid, *attrs), boxes, _ = fixed_table(split_name, "sil", device)
    source = []
    for i, keep in enumerate(valid):
        source.append(region_features(boxes[i, keep, 8:12] * [WIDTH, HEIGHT, WIDTH, HEIGHT],
                                      xy[i, keep], p[i, keep], [a[i, keep] for a in attrs], evidence(raw, dense, i)))
    arrays = tuple(np.stack(z) for z in zip(*source))
    model = fusion_model("fusion").to(device)
    model.load_state_dict(load(INPUT / "fusion.pt"))
    return model, arrays


def historical_centers(split_name, variant):
    from kaggle_suite_detectors import infer_table
    from native_centers import CenterDetector

    _, device = setup(42)
    split = configure_split(split_name, device)
    model = CenterDetector()
    epoch = {"center8": 8, "center16": 16}[variant]
    model.load_state_dict(load(INPUT / f"native_epoch{epoch}.pt"))
    model = model.to(device, memory_format=torch.channels_last)
    x, xy, valid = infer_table(model, split, task=f"{variant}/{split_name}")
    np.savez(CACHE / f"{variant}_{split_name}.npz", x=x, xy=xy, valid=valid)


def expanded(split, device):
    """Permit weak native proposals to compete with the contour/dense union."""
    model, source = fused_bank(split.name, "center16", device)
    p, xy, valid, *attr = table_predictions(model, source, device)
    native = table("thaw", split.name)
    raw_p, raw_xy, raw_valid, *raw_attr = dense_probabilities(native)
    records = []
    out = np.zeros(len(split.data.ids))
    start = time.monotonic()
    offset = model.relation_offset.detach().cpu().numpy()
    for i, (rows, pairs) in enumerate(zip(split.data.rows, split.scene_pairs)):
        keep = valid[i]
        exist, coord, attrs, features = p[i, keep], xy[i, keep], [a[i, keep] for a in attr], source[0][i, keep]
        eligible = np.flatnonzero(raw_valid[i] & (raw_p[i] >= .14))
        if len(eligible):
            addxy = raw_xy[i, eligible]
            dist = np.linalg.norm((addxy[:, None] - coord[None]) * [120, 80], axis=-1)
            new = dist.min(1, initial=np.inf) > 7
            chosen = eligible[new][:max(0, 28-len(exist))]
            b = native[0][i, chosen].copy()
            b[:, 5+N_ATTRS] = raw_p[i, chosen]
            b[:, -3:] = raw_attr[-1][i, chosen]
            features = np.concatenate((features, b))
            exist = np.concatenate((exist, (.55 * raw_p[i, chosen]).clip(.06, .55)))
            coord = np.vstack((coord, raw_xy[i, chosen]))
            attrs = [np.vstack((a, n[i, chosen])) for a, n in zip(attrs, raw_attr)]
        programs = [split.data.parsed[j] for j in rows]
        h = hypotheses(programs, pairs, exist, coord, attrs, features, offset, 4096) if len(exist) else None
        records.append((rows, pairs, h))
        if (i+1) % 1250 == 0:
            tick(f"hypotheses/{split.name}", i+1, len(split.names), start)
    tick(f"hypotheses/{split.name}", len(split.names), len(split.names), start)
    return records


def ranking(split_name):
    _, device = setup(42)
    split = configure_split(split_name, device)
    scenes = expanded(split, device)
    coef_path = WEIGHTS / "rank.npy"
    if split_name == "train":
        training = pack(pd.DataFrame({"label": split.data.label}), scenes)
        del scenes
        np.save(coef_path, fit(training, 42, device))
    else:
        # evaluate only needs len(frame); labels are never read in inference.
        rows = np.empty(len(split.data.ids))
        coef = np.load(coef_path)
        record("draws", split, evaluate(rows, scenes, np.zeros_like(coef)))
        record("rank", split, evaluate(rows, scenes, coef))


def render_split(split_name, shard):
    _, device = setup(42)
    split = configure_split(split_name, device)
    source = table("thaw", split_name)
    head = fusion_model("thaw").to(device)
    head.load_state_dict(load(WEIGHTS / "thaw_head.pt"))
    p, _, valid, _, colors, _, shapes = table_predictions(head, source, device)
    color_group = ("gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow")
    order = np.array([COLORS.index(c) for c in color_group])
    rgb_bg = np.load(CACHE / "background.npy")
    indices = np.arange(shard, len(split.names), 2)
    # Reconstructed objects use the same classes and candidates as the control.
    box = source[0][..., 1+N_ATTRS:5+N_ATTRS] * np.array([WIDTH, HEIGHT, WIDTH, HEIGHT], np.float32)
    start = time.monotonic()

    def process(part, quick=False):
        maps = []
        for i in part:
            rgb = split.rgb[i]
            planes, _ = _color_planes(rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), rgb_bg)
            maps.append(np.stack(planes))
        arrays = [box[part], (p * valid)[part], shapes[part].argmax(-1), order[colors[part].argmax(-1)],
                  np.stack(maps)]
        return reconstruct(*(torch.from_numpy(t).to(device) for t in arrays), 167 + int(part[0]),
                           steps=(1, 1) if quick else (36, 20))

    batch = tune_batch(f"3d/{split_name}/{shard}", (96, 64, 32, 16, 8),
                       lambda n: process(indices[np.arange(n) % len(indices)], True), device)
    result = []
    for j in range(0, len(indices), batch):
        part = indices[j:j+batch]
        result.append([t.cpu().numpy() for t in process(part)])
        if j == 0 or (j//batch) % 8 == 0 or j+batch >= len(indices):
            tick(f"3d/{split_name}/{shard}", j+len(part), len(indices), start)
    np.savez(CACHE / f"3d_{split_name}_{shard}.npz", indices=indices,
             **dict(zip(("initial", "fitted", "before", "after"), (np.concatenate(a) for a in zip(*result)))))


def render_scores(split_name):
    _, device = setup(42)
    split = configure_split(split_name, device)
    source = table("thaw", split_name)
    head = fusion_model("thaw").to(device)
    head.load_state_dict(load(WEIGHTS / "thaw_head.pt"))
    p, _, valid, *attrs = table_predictions(head, source, device)
    coordinates = np.zeros((*p.shape, 2), np.float32)
    refined = coordinates.copy()
    before, after = [], []
    for shard in (0, 1):
        with np.load(CACHE / f"3d_{split_name}_{shard}.npz") as result:
            ids = result["indices"]
            coordinates[ids], refined[ids] = result["initial"], result["fitted"]
            before.extend(result["before"])
            after.extend(result["after"])
    # Scale of 3-D ground positions differs from the camera-plane training offsets.
    record("mc_control", split, joint(split, (p, coordinates, valid, *attrs)))
    record("mc3d", split, joint(split, (p, refined, valid, *attrs)))
    write_json(REPORT / f"3d_{split_name}_image_loss.json", {
        "initial": float(np.mean(before)), "refined": float(np.mean(after)),
        "fraction_improved": float((np.array(after) < np.array(before)).mean()),
        "steps": [36, 20], "resolution": [[40, 60], [64, 96]],
    })


def check_geometry():
    _, device = setup(42)
    box = torch.tensor([[[62., 40., 24., 28.], [112., 62., 38., 33.]]], device=device)
    probability = torch.tensor([[.95, .85]], device=device)
    shape = torch.tensor([[0, 2]], device=device)
    color = torch.tensor([[1, 4]], device=device)
    masks = torch.zeros((1, len(COLORS), HEIGHT, WIDTH), device=device)
    masks[:, 1, 40:68, 62:86] = 1
    masks[:, 4, 62:95, 112:150] = 1
    raw, fitted, before, after = reconstruct(box, probability, shape, color, masks, 42, steps=(1, 1))
    if not all(torch.isfinite(x).all().item() for x in (raw, fitted, before, after)) or (after > before).any():
        raise RuntimeError("3-D reconstruction preflight failed")
    say("preflight", f"3-D reconstruction passed on {device}")


def write_csv(sample, ids, logits, target):
    if not sample["id"].is_unique or len(ids) != len(set(ids)) or set(ids) != set(sample["id"]):
        raise ValueError("Sample and test IDs must be unique and match")
    p = pd.Series(expit(logits), index=ids).reindex(sample["id"])
    if not np.isfinite(p).all() or not p.between(0, 1).all():
        raise ValueError("Submission probabilities must be finite and in [0, 1]")
    result = sample.copy()
    result[target] = p.to_numpy()
    return result


def finalize():
    sample = pd.read_csv(DATA / "sample_submission.csv")
    target, = [k for k in sample if k != "id"]

    def results(split):
        reference = pd.read_csv(DATA / f"{split}.csv", usecols=["id"]).id.to_numpy()
        values = {}
        for name in CONFIGS:
            if name in FAMILIES:
                values[name] = sum(values[part]*weight for part, weight in FAMILIES[name].items())
                continue
            with np.load(SCORES / f"{name}_{split}.npz") as a:
                if not np.array_equal(a["ids"], reference):
                    raise ValueError(f"Mismatched ids in {name}/{split}")
                values[name] = a["score"]
        return reference, values

    config_split = configure_split("val", torch.device("cpu"))
    _, val = results("val")
    ids, test = results("test")
    for name in FAMILIES:
        record(name, config_split, val[name])
    for name, logits in test.items():
        path = CSV / f"{name}.csv"
        submission = write_csv(sample, ids, logits, target)
        submission.to_csv(path, index=False, float_format="%.16g")
        replay = pd.read_csv(path)
        if list(replay) != list(sample) or not replay.id.equals(sample.id) or not replay[target].between(0, 1).all():
            raise ValueError(f"CSV verification failed for {name}")
    rows = []
    for name in CONFIGS:
        result = json.loads((REPORT / f"{name}.json").read_text())
        rows.append({"config": name, **{k: result[k] for k in ("auc", "at_least_999", "wrong_pairs")}})
    summary = pd.DataFrame(rows).sort_values("auc", ascending=False)
    summary.to_csv(REPORT / "leaderboard.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def archives(folder):
    groups = {"weights": (WEIGHTS,), "reports": (REPORT,), "scores": (SCORES,), "csv": (CSV,)}
    for name, locations in groups.items():
        dest = folder / f"clevr_{name}.zip"
        with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            for location in locations:
                for path in sorted(location.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(location))
        say("zip", f"{dest.name} | {dest.stat().st_size / 2**20:.2f} MiB")


def dispatch(args):
    task, *p = args
    jobs = {
        "prep": prepare, "hist": historical_centers, "detect": detector_eval, "train": train_detector,
        "probe": lambda name: train_detector(name, probe_only=True),
        "head": fit_head, "head_eval": head_eval, "base": baseline, "recall": pasted_recall,
        "crop": crop_extract, "tune": train_pool, "encode": encode_tuned, "rank": ranking,
        "render": render_split, "render_scores": render_scores, "pool_check": check_pool,
        "geometry_check": check_geometry,
    }
    if task in {"crop", "encode", "render"}:
        p[-1] = int(p[-1])
    jobs[task](*p)


def stage_plan():
    initial = [("prep", "train", x) for x in ("sil", "counts", "seam")]
    stages = [
        ("train inputs", initial),
        ("validation inputs", [("prep", "val", x) for x in ("sil", "counts", "seam")]),
        ("matched detector batch calibration", [("probe", n) for n in DETECTORS if n != "thaw"]),
        ("matched detector training", [("train", "r50_img"), ("train", "r50_rand")]),
        ("overlap detector training", [("train", "occ_clean"), ("train", "occ_paste")]),
        ("native train and validation tables", [("detect", s, "native") for s in ("train", "val")]),
        ("frozen caption heads", [("head", n) for n in HISTORICAL_HEADS]),
        ("unfrozen caption head on both GPUs", [("both", "train", "thaw")]),
        ("train upstream representations", [("detect", "train", "thaw"), ("hist", "val", "center16"), ("recall",)]),
        ("crop features", [("crop", "train", str(i)) for i in (0, 1)]),
        ("learn crop pooling and generate historical training proposals", [("tune",), ("hist", "train", "center16")]),
        ("tuned crop features", [("encode", "train", str(i)) for i in (0, 1)]),
        ("fit the hypothesis ranker and crop heads",
         [("rank", "train"), *[("head", x) for x in CROP_HEADS]]),
    ]
    for split in ("val", "test"):
        if split == "test":
            stages.append(("inference inputs", [("prep", split, x) for x in ("sil", "counts", "seam")]))
        upstream = [("hist", split, "center8"), ("detect", split, "thaw")]
        if split == "test":
            upstream += [("hist", split, "center16"), ("detect", split, "native")]
        stages.extend([
            (f"{split} upstream predictions", upstream),
            (f"{split} crop features", [("crop", split, str(i)) for i in (0, 1)]),
            (f"{split} tuned crop features", [("encode", split, str(i)) for i in (0, 1)]),
            (f"{split} iterative 3-D reconstruction", [("render", split, str(i)) for i in (0, 1)]),
            (f"{split} scorers", [("base", split), ("render_scores", split), ("rank", split),
                *[("detect", split, n) for n in DETECTORS if n != "thaw"],
                *[("head_eval", split, n) for n in (*HISTORICAL_HEADS, *CROP_HEADS)]]),
        ])
    return stages


def orchestrate(destination):
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Select T4 x2 in Kaggle Settings before Run All")
    for path in (CACHE, OUTPUT, WEIGHTS, REPORT, SCORES, CSV, destination):
        path.mkdir(exist_ok=True, parents=True)
    for name in ("scene_silhouette", "scene_seams", "scene_counts", "native_epoch8",
                 "native_epoch16", "native_natural_epoch21", "fusion"):
        shutil.copy2(INPUT / f"{name}.pt", WEIGHTS / f"{name}.pt")
    for i in (0, 1):
        props = torch.cuda.get_device_properties(i)
        say("hardware", f"GPU {i}: {props.name}, {props.total_memory / 2**30:.2f} GiB")
    progress = OUTPUT / "progress"
    progress.mkdir(exist_ok=True)
    timing_path = REPORT / "timing.json"
    previous = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    stage_minutes = previous.get("stages", {})
    timings = [json.loads(path.read_text()) for path in sorted(progress.glob("*.json"))]
    full_start = time.monotonic()
    plan = stage_plan()
    for index, (name, tasks) in enumerate(plan, 1):
        def receipt(task):
            return progress / f"{index:02d}-{'-'.join(task)}.json"

        remaining = [task for task in tasks if not receipt(task).exists()]
        if not remaining and name in stage_minutes:
            say("resume", f"{index}/{len(plan)} | {name} already complete")
            continue
        if name == "inference inputs":
            elapsed = (time.monotonic() - full_start) / 60
            test_size = pd.read_csv(DATA / "test.csv", usecols=["image"]).image.nunique()
            val_size = pd.read_csv(DATA / "val.csv", usecols=["image"]).image.nunique()
            scale = test_size / val_size
            render_eta = stage_minutes["val iterative 3-D reconstruction"] * scale
            inference_eta = scale * (stage_minutes["validation inputs"]
                                     + sum(t for label, t in stage_minutes.items() if label.startswith("val ")))
            say("ETA", f"full run ~{elapsed+inference_eta:.0f} min | "
                       f"test 3-D ~{render_eta:.0f} min | rough extrapolation from this session")
        say("stage", f"{index}/{len(plan)} | {name}")
        stage_start = time.monotonic()
        pending = deque(remaining)

        def lane(gpu):
            while pending:
                try:
                    args = pending.popleft()
                except IndexError:
                    return
                marker = receipt(args)
                devices = "0,1" if args[0] == "both" else str(gpu)
                args = args[1:] if args[0] == "both" else args
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": devices, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                       "OPENBLAS_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
                stamp = time.monotonic()
                say("gpu", f"{devices} | {' '.join(args)}")
                subprocess.run([sys.executable, "-m", "scripts.research.kaggle_suite", "--task", *args],
                               env=env, check=True)
                item = {"task": " ".join(args), "gpu": devices, "minutes": (time.monotonic()-stamp)/60}
                write_json(marker, item)
                timings.append(item)

        with ThreadPoolExecutor(2) as pool:
            live = [pool.submit(lane, i) for i in (0, 1)]
            for future in live:
                future.result()
        elapsed = (time.monotonic()-full_start)/60
        stage_minutes[name] = (time.monotonic()-stage_start)/60
        say("stage", f"{name} | {stage_minutes[name]:.1f} min | elapsed {elapsed:.1f} min")
        write_json(REPORT / "timing.json", {"elapsed_minutes": elapsed, "stages": stage_minutes, "tasks": timings})
    finalize()
    write_json(REPORT / "timing.json", {"minutes_before_archiving": (time.monotonic()-full_start)/60,
                                        "stages": stage_minutes, "tasks": timings})
    archives(destination)
    say("complete", f"End-to-end wall time {(time.monotonic()-full_start)/60:.1f} min")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", nargs="+")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    options = parser.parse_args()
    if options.task:
        dispatch(options.task)
    else:
        orchestrate(options.output)


if __name__ == "__main__":
    main()
