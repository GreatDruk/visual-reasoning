"""Replay existing count and geometry evidence on uncertain twin captions."""

import argparse
import io
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.twins import find_twins


ROOT = Path("artefacts/kaggle_suite/curriculum_run")
GEO_KEY = "geometry_delta@1.0@2.0"
SPATIAL_KEY = "stack/unet_boundary/views/new/0.55@1.0@2.0"
MAX_BASE_LOGIT = 1.
MIN_COUNT_LOGIT = logit(.85)
COUNT_WEIGHT = .25


def read_npz(source, frame):
    with np.load(source) as data:
        if not np.array_equal(data["ids"], frame.id):
            raise ValueError(f"Prediction IDs do not match {source}")
        return dict(data)


def read_csv_scores(path, frame):
    table = pd.read_csv(path, float_precision="round_trip").set_index("id", verify_integrity=True)
    if len(table) != len(frame) or set(table.index) != set(frame.id):
        raise ValueError(f"Prediction IDs do not match {path}")
    scores = table.loc[frame.id, "label"].to_numpy()
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError(f"Invalid predictions in {path}")
    return scores


def load(root, frame, split):
    with ZipFile(root.parent/"count_run/count_results.zip") as archive:
        counters = [
            read_npz(io.BytesIO(archive.read(f"r50_{seed}/verify_result/control_{seed}_{split}.npz")), frame)["score"]
            for seed in (421, 853)
        ]
    geometry = read_npz(root/"spatial"/f"{split}_geometry.npz", frame)
    if split == "val":
        shapes = read_npz(root/"shape_mat_val.npz", frame)
        geom = read_npz(root/"spatial/val_scores.npz", frame)
        keys = list(geom["keys"])
        anchor, mat = shapes["base"], shapes["score"]
        geo, baseline = (geom["scores"][keys.index(key)] for key in (GEO_KEY, SPATIAL_KEY))
        baseline_prob = expit(baseline)
    else:
        anchor_prob = read_csv_scores(root.parent/"count_run/count_hidden.csv", frame)
        mat_prob = read_csv_scores(root/"shape_mat.csv", frame)
        geo_prob = read_csv_scores(root/"shape_geo.csv", frame)
        baseline_prob = read_csv_scores(root/"shape_spatial.csv", frame)
        anchor, mat, geo, baseline = (
            logit(prob.clip(1e-15, 1 - 1e-15))
            for prob in (anchor_prob, mat_prob, geo_prob, baseline_prob)
        )
    return anchor, mat, geo, baseline, baseline_prob, geometry, counters


def val_twins(frame):
    return frame.reset_index().pivot(index=["image", "pair_id"], columns="label", values="index")[[1, 0]].to_numpy()


def compose(frame, twins, inputs, max_base=MAX_BASE_LOGIT, min_count=MIN_COUNT_LOGIT):
    anchor, mat, geo, baseline, _, geometry, counters = inputs
    left, right = twins.T
    first, second = counters
    count = (
        frame.category.eq("counting").to_numpy()
        & (np.abs(mat-anchor) < 1e-10)
        & (np.abs(mat) <= max_base)
        & (first*second > 0)
        & (np.minimum(np.abs(first), np.abs(second)) >= min_count)
    )
    spatial = (
        frame.category.eq("spatial_counting").to_numpy()
        & geometry["selected"]
        & (np.abs(mat) <= max_base)
        & (geometry["score"]*geometry["delta"] > 0)
        & (np.abs(baseline-geo) < 1e-10)
    )
    masks = {}
    for family, selected in (("count", count), ("spatial", spatial)):
        matched = selected[left] & selected[right]
        mask = np.zeros(len(frame), dtype=bool)
        mask[left[matched]], mask[right[matched]] = True, True
        masks[family] = mask
    counted = (1-COUNT_WEIGHT)*baseline + COUNT_WEIGHT*(first+second)/2
    count_score = np.where(masks["count"], counted, baseline)
    spatial_score = np.where(masks["spatial"], geometry["score"], baseline)
    combined = np.where(masks["spatial"], geometry["score"], count_score)
    return {"baseline": baseline, "count": count_score, "geometry": spatial_score, "both": combined}, masks


def inversions(label, score):
    negative = np.sort(score[~label])
    positive = score[label]
    return float(len(negative)*len(positive) - .5*(
        np.searchsorted(negative, positive, "left").sum() +
        np.searchsorted(negative, positive, "right").sum()))


def swap_counts(baseline, counted, twins):
    """Use the count branch for orientation and the existing scores for ranking."""
    left, right = twins.T
    flipped = (baseline[left]-baseline[right])*(counted[left]-counted[right]) < 0
    a, b = left[flipped], right[flipped]
    result = baseline.copy()
    result[a], result[b] = baseline[b], baseline[a]
    return result


def evaluate(frame, twins, inputs, output):
    from sklearn.model_selection import KFold

    scores, masks = compose(frame, twins, inputs)
    truth = frame.label.to_numpy().astype(bool)
    l, r = twins.T
    scenes, names = pd.factorize(frame.image)
    evaluated = {}
    for name, score in scores.items():
        lost = inversions(truth, score)
        wrong = l[score[l] <= score[r]]
        evaluated[name] = {
            "inversions": lost,
            "auc": 1 - lost / (truth.sum()*(~truth).sum()),
            "wrong_pairs": len(wrong),
            "wrong_scenes": sorted(frame.image.iloc[wrong].unique()),
            "changed_rows": int(np.count_nonzero(score != scores["baseline"])),
        }
    oof = {}
    for seed in (619, 2401, 4211):
        selected = []
        values = scores["baseline"].copy()
        for train, holdout in KFold(5, shuffle=True, random_state=seed).split(names):
            fit = np.isin(scenes, train)
            apply = np.isin(scenes, holdout)
            family = min(scores, key=lambda k: inversions(truth[fit], scores[k][fit]))
            selected.append(family)
            values[apply] = scores[family][apply]
        oof[seed] = {
            "selected": selected,
            "inversions": inversions(truth, values),
            "wrong_pairs": int(np.count_nonzero(values[l] <= values[r])),
        }
    sensitivity = []
    for base in (.75, 1., 1.25):
        for threshold in (.80, .85, .90):
            s, _ = compose(frame, twins, inputs, max_base=base, min_count=logit(threshold))
            sensitivity.append(dict(base_limit=base, count_confidence=threshold,
                                    inversions=inversions(truth, s["both"])))
    swapped = swap_counts(scores["baseline"], scores["count"], twins)
    report = {"models": evaluated, "grouped_selection": oof, "sensitivity": sensitivity,
              "count_swap": {
                  "inversions": inversions(truth, swapped),
                  "wrong_pairs": int(np.count_nonzero(swapped[l] <= swapped[r])),
                  "changed_rows": int(np.count_nonzero(swapped != scores["baseline"])),
              }}
    (output/"count_guard_val.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output/"count_guard_val.npz", ids=frame.id.to_numpy(), **scores,
                        count_mask=masks["count"], spatial_mask=masks["spatial"])
    print(json.dumps(report, indent=2))


def export(root, data, frame, twins, inputs, swap):
    scores, _ = compose(frame, twins, inputs)
    unchanged = inputs[4]
    if swap:
        values = swap_counts(unchanged, scores["count"], twins)
        if not np.array_equal(np.sort(values), np.sort(unchanged)):
            raise ValueError("Count swaps must preserve the distribution of scores")
    else:
        score, baseline = scores["both"], scores["baseline"]
        values = np.where(score == baseline, unchanged, expit(score))
    sample = pd.read_csv(data/"sample_submission.csv")
    if not sample.id.is_unique or not frame.id.is_unique or set(sample.id) != set(frame.id):
        raise ValueError("Test IDs must match sample_submission.csv")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid export scores")
    table = sample.assign(label=pd.Series(values, index=frame.id).loc[sample.id].to_numpy())
    path = root/("count_swap.csv" if swap else "count_guard.csv")
    with NamedTemporaryFile(dir=root, suffix=".tmp", delete=False) as pending:
        tmp = Path(pending.name)
    try:
        table.to_csv(tmp, index=False, float_format="%.17g")
        pd.testing.assert_frame_equal(pd.read_csv(tmp, float_precision="round_trip"), table)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    changed = values != unchanged
    print(path, len(values), "rows", int(np.count_nonzero(changed)), "changed")
    print(frame[changed].groupby("category").image.agg(["size", "nunique"]).to_dict(orient="index"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("val", "test"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--swap", action="store_true", help="Swap original scores on count reversals only")
    args = parser.parse_args()
    frame = pd.read_csv(args.data/f"{args.split}.csv")
    inputs = load(args.root, frame, args.split)
    if args.split == "val":
        output = args.root/"error_audit"
        output.mkdir(exist_ok=True)
        evaluate(frame, val_twins(frame), inputs, output)
    else:
        twins = find_twins(frame.image, frame.caption, frame.category)
        export(args.root, args.data, frame, twins, inputs, args.swap)


if __name__ == "__main__":
    main()
