"""Geometry, permutation, provenance and real training checks for recovery."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import logit

from src.config import load_config
from src.data import Scenes
from src.detectors import dense_table
from src.engine import predict
from src.foundations import aligned_score, check_splits, read_bundle
from src.heads import TableHead
from src.recovery import train_head, training_phases
from src.recovery_bank import TableStore, cache_key
from src.union import BASE_DIM, ROI_DIM, UnionHead, local_features, merge, rescue_tables, undo_shift


def row(x=0.2, y=0.2, p=0.7):
    r = np.zeros(23, np.float32)
    r[:2] = logit(p)
    r[2:17] = np.concatenate([np.full(n, -np.log(n)) for n in [2, 8, 2, 3]])
    r[-6:-2] = [x, y, 0.1, 0.15]
    r[-2:] = [2 * x - 1, 2 * y - 1]
    return r


def fuse(left, right):
    settings = load_config("configs/045_union.yaml")["union"]
    return merge(
        np.array(left, np.float32).reshape(-1, 23),
        np.array(right, np.float32).reshape(-1, 23),
        np.ones((len(left), ROI_DIM), np.float16),
        np.full((len(right), ROI_DIM), 2, np.float16),
        settings,
    )


def test_hungarian_does_not_match_two_primary_objects_to_one_secondary():
    x, _, valid, _, counts = fuse([row(), row(x=0.21)], [row()])
    assert valid.sum() == 2 and counts["matched"] == 1
    assert (x[valid, -3:-1].sum(-1) == 2).sum() == 1


def test_weak_singletons_are_removed_but_corroborated_peaks_survive():
    assert fuse([row(p=0.04)], [])[2].sum() == 0
    x, _, valid, _, _ = fuse([row(p=0.04)], [row(p=0.03)])
    assert valid.sum() == 1 and x[0, 0] < 0
    assert fuse([], [])[2].sum() == 0


def test_reference_roi_follows_the_selected_reference_box():
    x, _, valid, roi, _ = fuse([row(p=0.3)], [row(p=0.9)])
    assert valid.sum() == 1 and x[0, 0] == pytest.approx(logit(0.9))
    np.testing.assert_array_equal(roi[0], 2)


def test_lower_decode_floor_keeps_local_maximum_rule():
    raw = torch.zeros(1, 20, 4, 4)
    raw[:, 0] = -10
    raw[0, 0, 1, 1] = -3
    raw[0, 0, 1, 2] = -3.1
    assert not dense_table(raw, topk=4)[2].any()
    x, _, valid = dense_table(raw, topk=4, min_logit=logit(0.025))
    assert valid.sum() == 1 and x[0, valid[0], 0].item() == -3


def test_phase_unshift_preserves_box_sizes_and_uses_correct_coordinate_scales():
    original = torch.tensor(row())[None, None]
    shifted = original.clone()
    shifted[..., -6:-4] += torch.tensor([2 / 240, 2 / 160])
    shifted[..., -2:] += 2 * torch.tensor([2 / 240, 2 / 160])
    torch.testing.assert_close(undo_shift(shifted, (2, 2)), original)


def test_local_sampling_preserves_spatial_detail_without_map_replication():
    ramp = (torch.arange(60) + 0.5) / 60
    features = ramp[None, None, None, :].expand(1, 64, 40, 60)
    values = local_features(features, torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.2, 0.2, 0.4, 0.4]]]))
    assert values.shape == (1, 2, 256)
    torch.testing.assert_close(values[0, 0, :4], torch.tensor([0.25, 0.75, 0.25, 0.75]))
    torch.testing.assert_close(values[0, 1, :4], torch.tensor([0.3, 0.5, 0.3, 0.5]))


@pytest.mark.parametrize("relational,roi", [(False, False), (True, False), (True, True)])
def test_head_is_permutation_equivariant_and_padding_cannot_affect_valid_objects(relational, roi):
    torch.manual_seed(8)
    head = UnionHead(relational, roi).eval()
    with torch.no_grad():
        head.residual[-1].weight.normal_(0, 0.1)
    x = torch.randn(2, 5, BASE_DIM + (ROI_DIM if roi else 0))
    x[..., 16:20] = torch.tensor([0.2, 0.2, 0.1, 0.1])
    xy = torch.randn(2, 5, 2)
    valid = torch.tensor([[True, True, True, False, False]] * 2)
    enc = head.encode(x, xy, valid)
    obj, attrs = head.slot_logits(enc)
    order = torch.tensor([3, 1, 4, 0, 2])
    inverse = order.argsort()
    other = head.encode(x[:, order], xy[:, order], valid[:, order])
    o, a = head.slot_logits(other)
    torch.testing.assert_close(o[:, inverse], obj)
    torch.testing.assert_close(a[:, inverse], attrs)
    altered = x.clone()
    altered[:, 3:] = 500
    changed = head.encode(altered, xy, valid)
    o, a = head.slot_logits(changed)
    torch.testing.assert_close(o[:, :3], obj[:, :3])
    torch.testing.assert_close(a[:, :3], attrs[:, :3])


def test_isolated_and_empty_tables_have_finite_forward_and_backward():
    head = UnionHead()
    x = torch.zeros(2, 3, BASE_DIM)
    x[..., 16:20] = torch.tensor([0.2, 0.2, 0.1, 0.1])
    valid = torch.tensor([[True, False, False], [False, False, False]])
    enc = head.encode(x, torch.zeros(2, 3, 2), valid)
    o, a = head.slot_logits(enc)
    loss = o.sum() + a.sum()
    loss.backward()
    assert torch.isfinite(loss) and all(
        p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters()
    )


def test_rescue_keeps_an_unsupported_scene_unchanged_and_does_not_alias_inputs():
    left = np.stack([row(x=0.1), row(x=0.7)])
    right = np.stack([row(x=0.4)])
    old = left.copy()
    control, boost = rescue_tables(left, right)
    np.testing.assert_array_equal(left, old)
    np.testing.assert_array_equal(control[0], boost[0])
    np.testing.assert_array_equal(control[3], boost[3])


def test_phase_assignment_is_fixed_per_scene_and_independent_of_head_seed():
    a = training_phases("scene", 5000)
    b = training_phases("scene", 5000)
    np.testing.assert_array_equal(a, b)
    assert set(a) == {0, 1, 2, 3} and not training_phases("zero", 50).any()


def test_all_convnext_head_variants_share_the_same_extraction_cache(tmp_path):
    from types import SimpleNamespace

    (tmp_path / "train.csv").write_text("immutable data")
    scenes = SimpleNamespace(root=tmp_path, split="train", names=np.array(["a", "b"]))
    inputs = dict(files={"reference/r50_img.pt": "old", "reference/thaw.pt": "native"})
    manifest = dict(models={"convnext": dict(checkpoint="c.pt")}, files={"c.pt": "c"})
    keys = [
        cache_key(load_config(f"configs/{n:03d}_union.yaml"), scenes, inputs, manifest)
        for n in range(46, 50)
    ]
    assert len(set(keys)) == 1
    manifest["files"]["c.pt"] = "changed"
    assert cache_key(load_config("configs/047_union.yaml"), scenes, inputs, manifest) != keys[0]


def test_foundation_inputs_reject_outputs_paths_and_unregistered_files(tmp_path):
    with pytest.raises(ValueError, match="output-only"):
        read_bundle(tmp_path / "artefacts")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                version=1,
                files={},
                models={"c": dict(checkpoint="missing", scores="missing", fit_scene_names=["a"])},
            )
        )
    )
    with pytest.raises(ValueError, match="Unregistered"):
        read_bundle(tmp_path)


def test_foundation_score_ids_are_aligned_and_validation_overlap_is_rejected(tmp_path):
    np.savez(tmp_path / "score.npz", ids=[2, 1], score=[20.0, 10.0])
    np.testing.assert_array_equal(
        aligned_score(tmp_path, "score.npz", pd.DataFrame(dict(id=[1, 2]))), [10.0, 20.0]
    )
    from types import SimpleNamespace

    manifest = dict(models={"c": dict(fit_scene_names=["a", "val"])})
    with pytest.raises(ValueError, match="overlap"):
        check_splits(manifest, SimpleNamespace(names=["a"]), SimpleNamespace(names=["val"]))


@pytest.fixture
def training_store(tmp_path):
    frames = []
    for i in range(3):
        for category, yes, no in [
            ("attribute", "there is a red cube", "there is a blue cube"),
            ("counting", "exactly one red cube", "exactly two red cubes"),
            (
                "spatial",
                "a red cube to the left of the green sphere",
                "a red cube to the right of the green sphere",
            ),
        ]:
            for label, caption in [(1, yes), (0, no)]:
                frames.append(
                    dict(
                        id=len(frames),
                        image=str(i) + ".png",
                        category=category,
                        caption=caption,
                        label=label,
                        pair_id=len(frames) // 2,
                    )
                )
    pd.DataFrame(frames).to_csv(tmp_path / "train.csv", index=False)
    scenes = Scenes(tmp_path, "train")
    path = tmp_path / "bank"
    path.mkdir()
    rng = np.random.default_rng(1)
    x = np.zeros((4, 3, 40, BASE_DIM), np.float32)
    x[..., :20] = np.concatenate(
        (
            np.array([logit(0.6)], np.float32),
            row()[2:17],
            np.array([0.2, 0.2, 0.1, 0.1], np.float32),
        )
    )
    valid = np.zeros((4, 3, 40), bool)
    valid[..., :2] = True
    for name, value in [
        ("x", x),
        ("xy", rng.normal(size=(4, 3, 40, 2)).astype(np.float32)),
        ("valid", valid),
        ("roi", rng.normal(size=(4, 3, 40, ROI_DIM)).astype(np.float16)),
        ("names", scenes.names),
    ]:
        np.save(path / (name + ".npy"), value)
    inputs = tmp_path / "inputs"
    (inputs / "reference").mkdir(parents=True)
    torch.save(TableHead().state_dict(), inputs / "reference/scene_silhouette.pt")
    return path, scenes, inputs


@pytest.mark.parametrize("config", [f"{n:03d}_union" for n in range(45, 51)])
def test_every_recovery_configuration_trains_and_predicts(training_store, config):
    path, scenes, inputs = training_store
    cfg = load_config(Path("configs") / (config + ".yaml"))
    device = torch.device("cpu")
    store = TableStore(path, scenes, device, cfg["union"]["roi"])
    head, report = train_head(cfg, scenes, store, inputs, device, steps=1)
    assert report["updates"] == 1 and np.isfinite(report["history"]).all()
    score = predict(scenes, head, store.phase(0), device, draws=32)
    assert np.isfinite(score).all()
    np.testing.assert_allclose(score[scenes.pairs].sum(-1), 0, atol=1e-10)


def test_head_initialization_and_training_do_not_depend_on_cache_rng_consumption(training_store):
    path, scenes, inputs = training_store
    cfg = load_config("configs/048_union.yaml")
    device = torch.device("cpu")
    store = TableStore(path, scenes, device)
    first, _ = train_head(cfg, scenes, store, inputs, device, steps=1)
    torch.randn(10000)
    second, _ = train_head(cfg, scenes, store, inputs, device, steps=1)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
