"""Evaluate fixed blends and localization against the exact count_swap replay."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import expit, logit

from scripts.research.count_evidence import inversions, read_csv_scores
from scripts.research.evidence_data import labeled_pairs, opposite_rows, read_captions
from src.twins import find_twins


def summary(frame, pairs, scores, baseline=None):
    label = frame.label.to_numpy(dtype=bool)
    lost = inversions(label, scores)
    gap = scores[pairs[:, 0]] - scores[pairs[:, 1]]
    failed = gap <= 0
    result = dict(rows=len(frame), scenes=frame.image.nunique(), inversions=lost,
                  auc=1 - lost / (label.sum() * (~label).sum()), wrong=int((gap < 0).sum()),
                  tied=int((gap == 0).sum()), wrong_true_ids=frame.id.iloc[pairs[failed, 0]].tolist())
    if baseline is not None:
        base_wrong = (baseline[pairs[:, 0]] - baseline[pairs[:, 1]]) <= 0
        result.update(repaired=int((base_wrong & ~failed).sum()), regressed=int((~base_wrong & failed).sum()))
    return result


def apply_candidate(base, candidate, pairs):
    """Predeclared conservative residual on low-margin pairs."""
    left, right = pairs.T
    delta = np.clip((candidate[left] - candidate[right])/2, -6, 6)
    base_margin = (base[left] - base[right])/2
    weight = .30 * np.maximum(0, 1 - np.abs(base_margin)/4)
    merged = (1 - weight) * base_margin + weight * delta
    result = base.copy()
    use = weight > 0
    result[left[use]], result[right[use]] = merged[use], -merged[use]
    return result


def read_scores(path, frame):
    with np.load(path) as result:
        np.testing.assert_array_equal(result["ids"], frame.id.to_numpy())
        scores = result["score"]
    if scores.shape != (len(frame),) or not np.isfinite(scores).all():
        raise ValueError(f"Invalid scores: {path}")
    return scores


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    inter = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0).prod()
    return inter / max((a[2:]-a[:2]).prod() + (b[2:]-b[:2]).prod() - inter, 1)


def reviewed_regions(directory):
    reviewed = pd.read_csv(directory / "curriculum_review.csv").fillna("")
    positive = []
    negative = []
    shapes = dict(b="cube", c="cylinder", s="sphere")
    # These nine assignments were manually reviewed in package_curriculum.VAL_MATERIAL.
    known_material = {"80781.jpg", "20971.jpg", "90931.jpg", "46987.jpg", "42269.jpg",
                      "77083.jpg", "69401.jpg", "87292.jpg", "28868.jpg"}
    for r in reviewed.itertuples():
        if r.rear:
            attrs = [shapes[r.shape], r.color] + ([r.material] if r.image in known_material else [])
            positive.append(dict(image=r.image, box=[r.rx, r.ry, r.rx+r.rw, r.ry+r.rh],
                                 attrs=attrs, cohort="other_natural"))
        else:
            negative.append(dict(image=r.image, color=r.color, box=[r.fx, r.fy, r.fx+r.fw, r.fy+r.fh]))
    for row in json.loads((directory/"reviewed_fragments_val.json").read_text(encoding="utf-8")):
        positive.append(dict(image=row["image"], box=row["xyxy"], attrs=row["attributes_user"], cohort="four"))
    return positive, negative


def localization(path, positive, negative):
    objects = defaultdict(list)
    for row in json.loads(path.read_text(encoding="utf-8")):
        if row["confidence"] >= .45:
            objects[row["image"]].append(row)
    rears, negatives = [], []
    for p in positive:
        candidates = [r for r in objects[p["image"]] if iou(p["box"], r["box"]) >= .20]
        correct = [r for r in candidates if set(p["attrs"]) <= {r[key] for key in
                                                               ("size", "color", "material", "shape")}]
        rears.append(dict(image=p["image"], cohort=p["cohort"], localized=bool(candidates),
                          correct_attributes=bool(correct), candidates=len(candidates)))
    for n in negative:
        a = np.asarray(n["box"])
        selected = []
        for r in objects[n["image"]]:
            b = np.asarray(r["box"])
            center = (b[:2]+b[2:])/2
            if (center >= a[:2]).all() and (center <= a[2:]).all() and r["color"] == n["color"]:
                selected.append(r)
        # One object of that color is already annotated as foreground in this window.
        negatives.append(dict(image=n["image"], excess=max(0, len(selected)-1)))
    return dict(regions=len(rears), localized=sum(r["localized"] for r in rears),
                localized_with_known_attributes=sum(r["correct_attributes"] for r in rears),
                negative_windows=len(negatives), windows_with_extra=sum(n["excess"] > 0 for n in negatives),
                extra_detections=sum(n["excess"] for n in negatives), rears=rears, negatives=negatives)


def mask_review(path, review, frame):
    rows = json.loads(path.read_text(encoding="utf-8"))
    with np.load(path.with_name("visible_masks.npz")) as data:
        bits = data["packed"]
        height, width = data["shape"]
    predicted = defaultdict(list)
    for row, mask in zip(rows, bits, strict=True):
        if row["confidence"] >= .45:
            predicted[row["image"]].append((row, np.unpackbits(mask).reshape(height, width).astype(bool)))
    results = defaultdict(list)
    names = set(frame.image)
    for label in pd.read_csv(review).fillna("").itertuples(index=False):
        if label.image not in names or label.kind not in ("rear", "highlight", "duplicate", "background"):
            raise ValueError(f"Bad validation annotation: {label.image}, {label.kind}")
        file = review.parent / label.mask
        if not file.resolve().is_relative_to(review.parent.resolve()):
            raise ValueError(f"Mask must be next to the validation CSV: {file}")
        with Image.open(file) as image:
            if image.size != (240, 160):
                raise ValueError(f"Mask is not in scene coordinates: {file}")
            mask = (np.asarray(image.convert("L")) > 127).reshape(80, 2, 120, 2).any(axis=(1, 3))
        detections = []
        for row, candidate in predicted[label.image]:
            overlap = (mask & candidate).sum()
            total = (mask | candidate).sum() if label.kind == "rear" else candidate.sum()
            if overlap / max(1, total) >= (.20 if label.kind == "rear" else .50):
                detections.append(row)
        value = dict(image=label.image, hits=len(detections))
        if label.kind == "rear":
            value["correct_attributes"] = any(all(x[k] == getattr(label, k)
                                                   for k in ("shape", "color", "material"))
                                               for x in detections)
        results[label.kind].append(value)
    return dict(results)


def cohort(frame, scores, scenes):
    use = frame.image.isin(scenes).to_numpy()
    sub = frame[use].reset_index(drop=True)
    pairs = labeled_pairs(sub) if len(sub) else []
    return summary(sub, pairs, scores[use]) if len(sub) else None


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--inputs", type=Path, required=True)
    cli.add_argument("--review", type=Path, help="Optional natural_val.csv with confirmed masks")
    cli.add_argument("--test", action="store_true")
    args = cli.parse_args()
    split = "test" if args.test else "val"
    frame, _, images, captions, *_ = read_captions(args.data, split)
    base_prob = (read_csv_scores(args.inputs/"count_swap.csv", frame) if args.test
                 else read_scores(args.inputs/"leader_val.npz", frame))
    base = logit(base_prob.clip(1e-15, 1-1e-15))
    pairs = (find_twins(frame.image, captions, frame.category) if args.test else labeled_pairs(frame))
    methods = {n: args.out / f"{n}_{split}.npz" for n in ("control", "hard")}
    masks = {p.parent.name: p for p in args.out.glob(f"mask_*/mask_{split}.npz")}
    methods.update(masks)
    if args.test:
        output = {}
        for name, path in methods.items():
            if not path.exists():
                continue
            score = read_scores(path, frame)
            probability = expit(apply_candidate(base, score, pairs))
            result = frame[["id"]].assign(label=probability)
            name = {"control": "evid", "hard": "evid_hard"}.get(name, name)
            result.to_csv(args.out / f"{name}.csv", index=False)
            output[name] = dict(rows=len(result))
    else:
        positive, negative = reviewed_regions(args.inputs)
        known = defaultdict(set)
        for item in positive:
            known[item["cohort"]].add(item["image"])
        output = {"count_swap": {"full": summary(frame, pairs, base),
                                  "cohorts": {n: cohort(frame, base, ids) for n, ids in known.items()}}}
        truth = frame.label.to_numpy()
        partners = opposite_rows(captions, truth, images, np.ones(len(frame), bool), 421)
        left = np.flatnonzero((partners >= 0) & (truth == 1))
        for name, path in methods.items():
            if not path.exists():
                continue
            scores = read_scores(path, frame)
            mixed = apply_candidate(base, scores, pairs)
            right = partners[left]
            output[name] = dict(full=summary(frame, pairs, scores, base),
                                blended=summary(frame, pairs, mixed, base),
                                cohorts={k: cohort(frame, mixed, ids) for k, ids in known.items()},
                                image_swap=dict(total=len(left), ranks_correct=int((scores[left]>scores[right]).sum()),
                                                both_correct=int(((scores[left]>0)&(scores[right]<0)).sum())))
            boxes = path.with_name("objects.json")
            if boxes.exists():
                output[name]["localization"] = localization(boxes, positive, negative)
                if args.review:
                    output[name]["confirmed_localization"] = mask_review(boxes, args.review, frame)
    (args.out / f"evaluation_{split}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    for name, result in output.items():
        print(name, result.get("blended", result.get("full", result)), flush=True)


if __name__ == "__main__":
    main()
