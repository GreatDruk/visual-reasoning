"""Replay registered frozen components; fitting never consumes these val/test assets."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data import Scenes
from .evidence import compose
from .foundations import aligned_score, read_bundle
from .metrics import compare, metrics
from .submit import export_csv, write_npz


def load_parts(root, manifest, scenes):
    metadata = json.loads(
        (root / manifest["assets"]["components_metadata"]).read_text(encoding="utf8")
    )
    entry = metadata["splits"][scenes.split]
    digest = hashlib.sha256((scenes.root / (scenes.split + ".csv")).read_bytes()).hexdigest()
    if digest != entry["csv_sha256"]:
        raise ValueError("Frozen component CSV fingerprint mismatch")
    with np.load(root / manifest["assets"]["components_" + scenes.split]) as z:
        ids = pd.Index(z["ids"])
        if not ids.is_unique:
            raise ValueError("Duplicate frozen component IDs")
        rows = ids.get_indexer(scenes.frame.id)
        names = pd.Index(entry["scene_names"])
        positions = names.get_indexer(scenes.names)
        if (rows < 0).any() or not names.is_unique or (positions < 0).any():
            raise ValueError("Missing frozen component rows/scenes")
        parts = {}
        for key in z.files:
            if key == "ids":
                continue
            value = z[key]
            if key == "hidden":
                if value.shape != (len(names), 96):
                    raise ValueError("Invalid hidden component layout")
                parts[key] = value[positions].copy()
            else:
                if value.shape != (len(ids),):
                    raise ValueError("Invalid frozen score layout")
                parts[key] = value[rows].copy()
            if not np.isfinite(parts[key]).all():
                raise ValueError("Non-finite frozen component")
    for key, split_roles in metadata.get("extra_scores", {}).items():
        parts[key] = aligned_score(
            root, manifest["assets"][split_roles[scenes.split]], scenes.frame
        )
    return parts, metadata


def replacement(scenes, branch, root, manifest, return_matched=False, matched_seed=421):
    parts, metadata = load_parts(root, manifest, scenes)
    settings = metadata["leader_settings"]
    control = compose(scenes, parts, settings)[0]["final"]
    if return_matched:
        source = parts.get(f"count54_{matched_seed}")
        matched = (
            compose(scenes, {**parts, "count54_421": source, "count54_853": source}, settings)[0][
                "final"
            ]
            if source is not None
            else None
        )
    # Replace the previous count expert, never add another count layer on top.
    parts["count54_421"] = parts["count54_853"] = branch
    candidate, swaps = compose(scenes, parts, settings)
    result = control, candidate["final"], swaps
    return (*result, matched) if return_matched else result


def run(project, data, foundations, selected, output, submission_dir, splits=("val", "test")):
    root, manifest = read_bundle(foundations)
    output, submission_dir = Path(output), Path(submission_dir)
    output.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    configs = [load_config(Path(project) / "configs" / (name + ".yaml")) for name in selected]
    if any(c["family"] != "submission" for c in configs):
        raise ValueError("Frozen replay requires submission configurations")
    report = {"scope": "frozen train-only component replay; no fitting; no new encoder inference"}
    for split in splits:
        scenes = Scenes(Path(data), split)
        parts, metadata = load_parts(root, manifest, scenes)
        control = compose(scenes, parts, metadata["leader_settings"])[0]["final"]
        report[split] = {}
        for cfg in configs:
            variants, swaps = compose(scenes, parts, cfg["blend"])
            write_npz(
                output / f"{cfg['name']}_{split}.npz", ids=scenes.frame.id.to_numpy(), **variants
            )
            result = dict(swapped_pairs=swaps, config=cfg)
            if scenes.label is not None:
                result.update(
                    metrics=metrics(scenes.frame, variants["final"]),
                    comparison=compare(scenes.frame, control, variants["final"], 0),
                )
            if split == "test":
                export_csv(
                    scenes.frame,
                    pd.read_csv(Path(data) / "sample_submission.csv"),
                    variants["final"],
                    submission_dir / (cfg["name"] + ".csv"),
                )
            report[split][cfg["name"]] = result
            print(json.dumps(dict(replay=cfg["name"], split=split, **result)), flush=True)
    (output / "replay_report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report
