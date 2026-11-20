"""Inventory actual localization labels and mine review candidates in labeled splits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research import count_evidence as counts
from scripts.research.evidence_data import labeled_pairs, opposite_rows, read_captions
from src.models.slot_program import EXACT, EXCLUSIVE_GROUPS, parse_program


def leader_validation(root, frame):
    twins = labeled_pairs(frame)
    inputs = counts.load(root, frame, "val")
    baseline = inputs[4]
    scores, _ = counts.compose(frame, twins, inputs)
    return counts.swap_counts(baseline, scores["count"], twins)


def deficits(data, split, teacher, output):
    frame, names, scene, caption, *_ = read_captions(data, split)
    with np.load(teacher) as bank:
        if "names" in bank:
            np.testing.assert_array_equal(bank["names"], names)
        else:
            np.testing.assert_array_equal(bank["ids"], frame.id)
        present = bank["valid"] & (bank["p"] >= .85)
        attributes = [bank[group] for group in ("size", "color", "material", "shape")]
    records = []
    for i in np.flatnonzero(frame.label.to_numpy() == 1):
        p = parse_program(caption[i])
        if p.op != EXACT or p.relation:
            continue
        image = scene[i]
        possible = definite = present[image].copy()
        for word in p.a:
            j, k = next((j, g.index(word)) for j, g in enumerate(EXCLUSIVE_GROUPS) if word in g)
            attr = attributes[j][image]
            uncertain = attr.max(1) < .80
            matches = attr.argmax(1) == k
            possible = possible & (matches | uncertain)
            definite = definite & matches & ~uncertain
        lower, upper = int(definite.sum()), int(possible.sum())
        if p.number > upper:
            row = frame.iloc[i]
            records.append(dict(image=row.image, id=int(row.id), caption=row.caption,
                                expected=p.number, definite=lower, possible=upper,
                                deficit=p.number-upper, filter=" ".join(p.a) or "all"))
    report = pd.DataFrame.from_records(records, columns=[
        "image", "id", "caption", "expected", "definite", "possible", "deficit", "filter"])
    report.sort_values(["deficit", "image"], ascending=[False, True]).to_csv(output, index=False)
    return dict(captions=len(report), scenes=report.image.nunique(),
                one_missing_scenes=report[report.deficit.eq(1)].image.nunique())


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, default=Path("data"))
    cli.add_argument("--suite", type=Path, default=Path("artefacts/kaggle_suite"))
    cli.add_argument("--research", type=Path, default=Path("artefacts/research"))
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    curriculum = args.suite / "curriculum_run"
    val, *_ = read_captions(args.data, "val")
    train, _, images, caption, *_ = read_captions(args.data, "train")
    old = pd.read_csv(args.research / "overlap_train_controls/review_labels.csv")
    official = pd.read_csv(curriculum / "inputs/curriculum_review.csv")
    marked = json.loads((curriculum / "inputs/reviewed_fragments_val.json").read_text(encoding="utf-8"))
    assert set(old.image) <= set(train.image) and set(official.image) <= set(val.image)
    assert all(row["image"] in set(val.image) for row in marked)
    positives = official[official.rear.eq(1)]
    review = dict(train_windows=len(old), train_window_counts=int(old.number.notna().sum()),
                  train_confirmed_fragment_targets=0, train_confirmed_rear_masks=0,
                  val_native_boxes=len(positives), val_hard_negative_windows=int(official.rear.eq(0).sum()),
                  val_user_boxes=len(marked), val_native_scenes=len(set(positives.image) |
                          {r["image"] for r in marked}), val_masks=0)
    selection = {}
    for split, name in (("train", "teacher_train.npz"), ("val", "scene_silhouette.npz")):
        selection[split] = deficits(args.data, split, args.research / name,
                                    args.out / f"review_{split}.csv")
    reference = leader_validation(curriculum, val)
    np.savez_compressed(args.out / "leader_val.npz", ids=val.id.to_numpy(), score=reference)
    partner = opposite_rows(caption, train.label.to_numpy(), images, np.ones(len(train), bool), 421)
    report = dict(localization=review, count_discrepancies=selection,
                  train_same_caption_rows=int((partner >= 0).sum()))
    (args.out / "inventory.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
