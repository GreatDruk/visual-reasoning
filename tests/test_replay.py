"""Frozen component provenance and replacement of the existing count slot."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.programs import parse_program
from src.replay import load_parts, replacement


@pytest.fixture
def bundle(tmp_path):
    frame = pd.DataFrame({"id": [7, 2], "image": ["scene.jpg"] * 2, "category": ["counting"] * 2})
    frame.to_csv(tmp_path / "val.csv", index=False)
    settings = dict(
        base="quality", counter_weight=0.25, tta_weight=0.0, hidden_weight=0.0, swap=False
    )
    metadata = dict(
        leader_settings=settings,
        splits={
            "val": dict(
                csv_sha256=hashlib.sha256((tmp_path / "val.csv").read_bytes()).hexdigest(),
                scene_names=["scene.jpg"],
            )
        },
    )
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    v = np.array([4.0, -4.0])
    parts = {k: v for k in ["convnext", "sigft", "union421", "union853"]}
    parts.update(
        rescue=v * 0,
        r50_tta=v * 0,
        r50_plain=v * 0,
        count54_421=v * 2,
        count54_853=v * 2,
        hidden=np.zeros((1, 96)),
    )
    # Score rows have a different order from the CSV; hidden uses scene order.
    np.savez(
        tmp_path / "parts.npz",
        ids=np.array([2, 7]),
        **{k: a if k == "hidden" else a[::-1] for k, a in parts.items()},
    )
    manifest = dict(assets=dict(components_metadata="metadata.json", components_val="parts.npz"))
    scenes = SimpleNamespace(
        root=tmp_path,
        split="val",
        frame=frame,
        names=np.array(["scene.jpg"]),
        scene=np.array([0, 0]),
        pairs=np.array([[0, 1]]),
        programs=[parse_program("exactly two cubes"), parse_program("exactly one cube")],
    )
    return tmp_path, manifest, scenes


def test_component_alignment_and_counter_replacement_are_exact(bundle):
    root, manifest, scenes = bundle
    parts, _ = load_parts(root, manifest, scenes)
    np.testing.assert_array_equal(parts["convnext"], [4.0, -4.0])
    control, candidate, swaps = replacement(scenes, np.array([-4.0, 4.0]), root, manifest)
    np.testing.assert_array_equal(control, [5.0, -5.0])
    np.testing.assert_array_equal(candidate, [2.0, -2.0])
    assert swaps == 0  # A second blend of the old result would instead give +/-2.75.


def test_frozen_components_reject_changed_csv(bundle):
    root, manifest, scenes = bundle
    (root / "val.csv").write_text("changed")
    with pytest.raises(ValueError, match="fingerprint"):
        load_parts(root, manifest, scenes)


def test_frozen_components_reject_duplicate_ids(bundle):
    root, manifest, scenes = bundle
    np.savez(root / "parts.npz", ids=[7, 7], hidden=np.zeros((1, 96)))
    with pytest.raises(ValueError, match="Duplicate"):
        load_parts(root, manifest, scenes)
