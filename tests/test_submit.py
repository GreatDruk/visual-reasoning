"""Submission pairing, coordinate inversion, evidence scope and CSV integrity."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import load_config
from src.evidence import compose, guarded_swap, transitions
from src.programs import parse_program
from src.submit import export_csv, restore_rows


def scenes():
    return SimpleNamespace(
        frame=pd.DataFrame(
            {"id": [4, 1, 7, 3], "category": ["counting", "counting", "attribute", "attribute"]}
        ),
        pairs=np.array([[0, 1], [2, 3]]),
        scene=np.array([0, 0, 1, 1]),
        programs=[
            parse_program(c)
            for c in [
                "exactly two cubes",
                "exactly one cube",
                "there is a red cube",
                "there is a blue cube",
            ]
        ],
    )


def test_inverse_flip_and_shift_restore_boxes_and_positions():
    rows = torch.zeros(1, 2, 23)
    rows[..., -6:-2] = torch.tensor([0.2, 0.3, 0.1, 0.2])
    rows[..., -2:] = torch.tensor([-0.4, 0.1])
    changed = rows.clone()
    changed[..., -6:-4] += torch.tensor([2 / 240, 2 / 160])
    changed[..., -2:] += torch.tensor([4 / 240, 4 / 160])
    changed[..., -6] = 1 - changed[..., -6] - changed[..., -4]
    changed[..., -2] *= -1
    torch.testing.assert_close(restore_rows(changed, True, True), rows)


def test_guarded_swap_preserves_values_and_scope():
    s = scenes()
    base = np.array([0.2, -0.2, 0.1, -0.1])
    expert = np.array([-5.0, 5.0, -5.0, 5.0])
    swapped, count = guarded_swap(s, base, expert, expert)
    assert count == 1
    np.testing.assert_array_equal(np.sort(swapped), np.sort(base))
    np.testing.assert_array_equal(swapped[2:], base[2:])
    unchanged, count = guarded_swap(s, base, expert, -expert)
    assert count == 0
    np.testing.assert_array_equal(unchanged, base)


def test_zero_hidden_mass_is_identity():
    a, b = scenes().programs[:2]
    np.testing.assert_array_equal(transitions(a, b, np.zeros(96)), np.eye(2))


@pytest.mark.parametrize("name", ["056_compact", "057_quality"])
def test_composition_keeps_pairs_antisymmetric_and_labels_are_unused(name):
    s = scenes()
    vector = np.array([2.0, -2.0, 3.0, -3.0])
    keys = [
        "convnext",
        "union421",
        "union853",
        "rescue",
        "sigft",
        "r50_plain",
        "r50_tta",
        "count54_421",
        "count54_853",
        "count53_421",
        "count53_853",
    ]
    parts = {k: vector.copy() for k in keys}
    parts["hidden"] = np.zeros((2, 96))
    cfg = load_config("configs/" + name + ".yaml")
    variants, _ = compose(s, parts, cfg["blend"])
    for v in variants.values():
        np.testing.assert_allclose(v[s.pairs[:, 0]], -v[s.pairs[:, 1]])
    s.frame["label"] = [0, 1, 1, 0]
    again, _ = compose(s, parts, cfg["blend"])
    np.testing.assert_array_equal(variants["final"], again["final"])


def test_export_aligns_sample_ids_and_rejects_invalid_output(tmp_path):
    frame = scenes().frame
    sample = pd.DataFrame({"id": [7, 3, 4, 1], "label": [0.0, 0.0, 0.0, 0.0]})
    path = tmp_path / "submission.csv"
    export_csv(frame, sample, np.array([2.0, -2.0, 3.0, -3.0]), path)
    result = pd.read_csv(path)
    assert result.id.tolist() == sample.id.tolist()
    assert (result.label.between(0, 1)).all()
    with pytest.raises(ValueError):
        export_csv(frame, sample, np.array([np.nan] * 4), path)
    with pytest.raises(ValueError):
        export_csv(frame, sample.iloc[:2], np.zeros(4), path)
