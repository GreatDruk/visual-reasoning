"""Recheck aligned train/val experiment outputs and build comparison tables.

Usage: OMP_NUM_THREADS=4 uv run python scripts/codex_cpu_research_report.py
Only data/val.csv and completed codex_cpu_research results are read. No fitting.
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["POLARS_MAX_THREADS"] = "4"

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from src.metrics import pair_accuracy


CATEGORIES = ["counting", "position", "spatial", "spatial_counting", "spatial_negation"]


def score(df, logits):
    y, pairs = df["label"].to_numpy(), df["pair_id"].to_numpy()
    return {"auc": float(roc_auc_score(y, logits)),
            "pair_accuracy": pair_accuracy(logits, y, pairs),
            "accuracy": float(np.mean((logits > 0) == y))}


def axis_scores(df, logits):
    result = {}
    for cat in ["spatial", "spatial_counting", "spatial_negation"]:
        for axis, pattern in [("left_right", r"to the (?:left|right) of"),
                              ("front_behind", r"behind|in front of")]:
            mask = ((df["category"] == cat) & df["caption"].str.contains(pattern)).to_numpy()
            if mask.any():
                sub = df.filter(pl.Series(mask))
                result[f"{cat}/{axis}"] = {"rows": len(sub), **score(sub, logits[mask])}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("artefacts/codex_cpu_research"))
    p.add_argument("--baseline", type=Path,
                   default=Path("artefacts/hydra/debug_siglip_full/2026-11-15_11-33-28"))
    p.add_argument("--bootstrap", type=int, default=0)
    args = p.parse_args()
    df = pl.read_csv("data/val.csv")
    baseline_logits = np.load(args.baseline / "val_logits.npy")
    assert np.array_equal(np.load(args.baseline / "val_ids.npy"), df["id"].to_numpy())
    positions = {value: i for i, value in enumerate(df["id"])}
    cat_array = df["category"].to_numpy()
    result = {"baseline_path": str(args.baseline), "baseline": score(df, baseline_logits),
              "baseline_per_category": {}, "baseline_axes": axis_scores(df, baseline_logits), "runs": {}}
    for cat in sorted(set(cat_array)):
        m = cat_array == cat
        result["baseline_per_category"][cat] = score(df.filter(pl.Series(m)), baseline_logits[m])
    for path in sorted(args.root.glob("*/results.json")):
        if path.parent.name.startswith("smoke"):
            continue
        run = json.loads(path.read_text())
        if "total_sec" not in run or not run.get("cache_unchanged"):
            continue
        mixed = baseline_logits.copy()
        mixed_para = np.load(args.baseline / "val_para_logits.npy").copy()
        per_cat, train_cat, para_cat, shuffle_cat = {}, {}, {}, {}
        covered = []
        for name, fit in run["fits"].items():
            folder = path.parent / name
            ids = np.load(folder / "val_ids.npy")
            indices = np.array([positions[i] for i in ids])
            logits = np.load(folder / "val_logits.npy")
            para_logits = np.load(folder / "val_para_logits.npy")
            sub = df[indices]
            assert set(sub["category"]) == set(fit["categories"])
            assert np.array_equal(sub["id"].to_numpy(), ids)
            for cat in fit["categories"]:
                mask = sub["category"].to_numpy() == cat
                checked = score(sub.filter(pl.Series(mask)), logits[mask])
                saved = fit["val"]["per_category"][cat]
                for key in checked:
                    assert abs(checked[key] - saved[key]) < 1e-12, (path, cat, key)
                per_cat[cat] = checked
                para_cat[cat] = score(sub.filter(pl.Series(mask)), para_logits[mask])
                train_cat[cat] = fit["train"]["per_category"][cat]
                if "val_image_shuffle" in fit:
                    shuffle_cat[cat] = fit["val_image_shuffle"]["per_category"][cat]
            mixed[indices] = logits
            mixed_para[indices] = para_logits
            covered.extend(indices.tolist())
        assert len(covered) == len(set(covered))
        weak_mask = np.isin(cat_array, list(per_cat))
        assert sorted(covered) == np.flatnonzero(weak_mask).tolist()
        only_position = baseline_logits.copy()
        only_position[cat_array == "position"] = mixed[cat_array == "position"]
        record = {"command": run["command"], "args": run["args"],
                  "cpu_total_min": run["total_sec"] / 60,
                  "cpu_fit_min": sum(f["fit_sec"] for f in run["fits"].values()) / 60,
                  "params": {name: fit["params"] for name, fit in run["fits"].items()},
                  "best_epochs": {name: fit["best_epoch"] for name, fit in run["fits"].items()},
                  "steps": sum(f["steps"] for f in run["fits"].values()),
                  "val": per_cat, "val_para": para_cat, "train": train_cat,
                  "val_image_shuffle": shuffle_cat,
                  "last_val": {name: fit["last_val"] for name, fit in run["fits"].items()},
                  "val_macro_auc": float(np.mean([m["auc"] for m in per_cat.values()])),
                  "val_macro_pair": float(np.mean([m["pair_accuracy"] for m in per_cat.values()])),
                  "mixed_all9": score(df, mixed), "mixed_all9_para": score(df, mixed_para),
                  "position_only_mixed_all9": score(df, only_position),
                  "spatial_axes": axis_scores(df, mixed),
                  "cache_unchanged": run["cache_unchanged"]}
        if args.bootstrap:
            # Resample images, retaining all captions/pairs of a scene together.
            intervals = {}
            for cat in per_cat:
                m = cat_array == cat
                sub = df.filter(pl.Series(m))
                old, new = baseline_logits[m], mixed[m]
                names = sub["image"].to_numpy()
                unique = np.unique(names)
                groups = [np.flatnonzero(names == name) for name in unique]
                rng = np.random.default_rng(20261115)
                deltas = []
                for _ in range(args.bootstrap):
                    rows = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
                    y = sub["label"].to_numpy()[rows]
                    deltas.append(roc_auc_score(y, new[rows]) - roc_auc_score(y, old[rows]))
                intervals[cat] = {"delta_auc": per_cat[cat]["auc"] - result["baseline_per_category"][cat]["auc"],
                                  "image_bootstrap_95pct": np.quantile(deltas, [0.025, 0.975]).tolist(),
                                  "replicates": args.bootstrap,
                                  "limitation": "conditional on selected checkpoint; no multiple-selection correction"}
            record["baseline_delta_intervals"] = intervals
        result["runs"][path.parent.name] = record
    result["total_recorded_cpu_min"] = sum(r["cpu_total_min"] for r in result["runs"].values())
    (args.root / "comparison.json").write_text(json.dumps(result, indent=2))
    lines = ["В ячейках: val AUC / pair accuracy. Метрики заново пересчитаны из ids/logits.", "",
             "| Вариант | " + " | ".join(CATEGORIES) + " | CPU, мин |",
             "|---|" + "---|" * (len(CATEGORIES) + 1)]
    base = result["baseline_per_category"]
    lines.append("| Исходный SigLIP Transformer | " + " | ".join(f"{base[c]['auc']:.3f} / {base[c]['pair_accuracy']:.3f}" for c in CATEGORIES) + " | сохранённый запуск |")
    for name, r in result["runs"].items():
        cells = [f"{r['val'][c]['auc']:.3f} / {r['val'][c]['pair_accuracy']:.3f}" if c in r["val"] else "—" for c in CATEGORIES]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {r['cpu_total_min']:.2f} |")
    lines += ["", "В ячейках ниже: val_para AUC / pair accuracy; это переписывания общими правилами аугментации, не независимый OOD holdout.", "",
              "| Вариант | " + " | ".join(CATEGORIES) + " |", "|---|" + "---|" * len(CATEGORIES)]
    for name, r in result["runs"].items():
        cells = [f"{r['val_para'][c]['auc']:.3f} / {r['val_para'][c]['pair_accuracy']:.3f}" if c in r["val_para"] else "—" for c in CATEGORIES]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Точные команды:", ""]
    for name, r in result["runs"].items():
        lines += [f"`{name}`:", "", "```bash", r["command"], "```", ""]
    (args.root / "tables.md").write_text("\n".join(lines))
    print(json.dumps({"runs": list(result["runs"]), "recorded_cpu_min": result["total_recorded_cpu_min"],
                      "baseline": result["baseline"], "out": str(args.root / "comparison.json")}, indent=2))


if __name__ == "__main__":
    main()
