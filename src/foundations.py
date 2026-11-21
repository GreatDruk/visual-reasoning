"""Explicit, checksummed train-only checkpoints for downstream research."""

import hashlib
import json
from pathlib import Path

import numpy as np


def read_bundle(root, verify=True):
    root = Path(root).resolve()
    if "artefacts" in root.parts:
        raise ValueError("Foundation inputs cannot originate in output-only artifacts")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or not manifest.get("models"):
        raise ValueError("Unsupported foundation input bundle")
    for name, digest in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Foundation input path escapes its root")
        if verify:
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError(f"Foundation checksum mismatch: {name}")
    for item in manifest["models"].values():
        if not {item["checkpoint"], item["scores"]} <= manifest["files"].keys():
            raise ValueError("Unregistered foundation file")
        names = item["fit_scene_names"]
        if not names or len(names) != len(set(names)):
            raise ValueError("Invalid foundation training scene provenance")
    return root, manifest


def check_splits(manifest, train, val):
    for item in manifest["models"].values():
        fit = set(item["fit_scene_names"])
        if set(val.names) & fit or not set(train.names) <= fit:
            raise ValueError("Foundation training scenes overlap val or miss train scenes")
        if (
            item["validation_csv_sha256"]
            != hashlib.sha256((val.root / "val.csv").read_bytes()).hexdigest()
        ):
            raise ValueError("Foundation validation CSV differs")


def aligned_score(root, name, frame):
    with np.load(Path(root) / name, allow_pickle=False) as data:
        import pandas as pd

        if len(set(data["ids"].tolist())) != len(data["ids"]):
            raise ValueError("Duplicate foundation score IDs")
        idx = pd.Index(data["ids"]).get_indexer(frame.id)
        if (idx < 0).any():
            raise ValueError("Missing foundation score IDs")
        score = data["score"][idx].copy()
    if not np.isfinite(score).all():
        raise ValueError("Non-finite foundation scores")
    return score


def baselines(root, manifest, frame, historical, sigft):
    scores = {"historical": historical, "sigft": sigft}
    scores.update(
        {
            key: aligned_score(root, item["scores"], frame)
            for key, item in manifest["models"].items()
        }
    )
    result = {"historical": 0.5 * (historical + sigft)}
    for name, weights in manifest["baselines"].items():
        if not weights or min(weights.values()) < 0 or not np.isclose(sum(weights.values()), 1):
            raise ValueError("Foundation blend weights must be nonnegative and sum to one")
        result[name] = sum(scores[key] * weight for key, weight in weights.items())
    return result
