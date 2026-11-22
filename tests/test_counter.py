"""Count conservation, visibility, query semantics and full-loss training paths."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from src.config import load_config
from src.count_run import train
from src.counter import TYPES, CountQueries, DenseCounter, freeze_statistics
from src.counter_data import CounterData, donor_arrays
from src.data import Scenes


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=4, padding=1), nn.BatchNorm2d(64), nn.SiLU()
        )
        self.head = nn.Conv2d(64, 20, 1)

    def pyramid(self, x):
        return [self.features(x)]


def test_zero_joint_correction_matches_factorized_histogram_and_preserves_mass():
    torch.manual_seed(1)
    raw = torch.randn(2, 20, 4, 6, requires_grad=True)
    correction = torch.randn(2, 1, 4, 6, requires_grad=True)
    hist, mass = DenseCounter.integrate(raw, correction)
    joint, _ = DenseCounter.integrate(raw, correction, torch.zeros(2, 96, 4, 6))
    torch.testing.assert_close(hist, joint, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(hist.sum(-1), mass.sum((-2, -1)))
    hist.square().mean().backward()
    assert torch.isfinite(raw.grad).all() and torch.isfinite(correction.grad).all()


def test_dense_normalization_is_finite_for_extreme_half_outputs():
    raw = torch.full((2, 20, 4, 6), 20.0, dtype=torch.float16, requires_grad=True)
    correction = torch.full((2, 1, 4, 6), 20.0, dtype=torch.float16, requires_grad=True)
    hist, mass = DenseCounter.integrate(raw.float(), correction.float())
    assert hist.dtype == torch.float32 and torch.isfinite(hist).all() and torch.isfinite(mass).all()
    hist.sum().backward()
    assert torch.isfinite(raw.grad).all()


def test_report_archive_does_not_duplicate_counter_weights(tmp_path):
    import zipfile

    from src.runtime import Sink

    sink = Sink(tmp_path / "artefacts", "run", tmp_path / "scratch")
    sink.state("model.pt", dict(weight=torch.ones(2)))
    sink.json("report.json", dict(ok=True))
    sink.finish(include_weights=False)
    assert (tmp_path / "artefacts/weights/run/model.pt").is_file()
    with zipfile.ZipFile(tmp_path / "artefacts/results/run/result.zip") as z:
        assert "results/report.json" in z.namelist()
        assert not any(n.startswith("weights/") for n in z.namelist())


@pytest.fixture
def counter_data(tmp_path):
    (tmp_path / "train").mkdir()
    rows = []
    for i in range(4):
        Image.fromarray(np.full((160, 240, 3), 40 + 30 * i, np.uint8)).save(
            tmp_path / "train" / f"{i}.png"
        )
        for category, yes, no in [
            ("counting", "exactly three small cubes", "exactly one small cube"),
            (
                "comparison",
                "there are more cubes than spheres",
                "there are fewer cubes than spheres",
            ),
            ("attribute", "there is a red cube", "there is a blue cube"),
        ]:
            for label, caption in [(1, yes), (0, no)]:
                rows.append(
                    dict(
                        id=len(rows),
                        image=f"{i}.png",
                        category=category,
                        caption=caption,
                        label=label,
                        pair_id=len(rows) // 2,
                    )
                )
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    scenes = Scenes(tmp_path, "train")
    np.savez(
        tmp_path / "donors.npz",
        names=scenes.names,
        keep=np.ones((4, 1), bool),
        box=np.tile([90, 70, 30, 30], (4, 1, 1)),
        mask=np.full((4, 1, 10, 10), 255, np.uint8),
        labels=np.tile([0, 1, 0], (4, 1, 1)),
    )
    manifest = dict(files={"donors.npz": "immutable"}, assets={"donors": "donors.npz"})
    bank = CounterData(tmp_path, manifest, scenes, tmp_path / "scratch", torch.device("cpu"))
    return scenes, bank, manifest


@pytest.mark.parametrize("difficulty,attempts", [("mixed", 1), ("tiny", 3)])
def test_insertions_do_not_touch_existing_foreground_and_targets_match_acceptance(
    counter_data, difficulty, attempts
):
    scenes, bank, _ = counter_data
    torch.manual_seed(9)
    idx = torch.arange(32) % len(scenes.names)
    before, after, target, stats = bank.pair(idx, difficulty, attempts)
    protected = bank.foreground[idx][:, None].expand_as(before)
    torch.testing.assert_close(before[protected], after[protected], rtol=0, atol=0)
    assert torch.all((target.sum(-1) == 0) | (target.sum(-1) == 1))
    rejected = target.sum(-1) == 0
    torch.testing.assert_close(before[rejected], after[rejected], rtol=0, atol=0)
    assert stats[0] > 0 and stats[0] == target.sum()
    assert stats[2] >= 4 * stats[0]
    if difficulty == "tiny":
        assert stats[2] <= 60 * stats[0]


def test_rejected_insertions_are_exact_noops(counter_data):
    _, bank, _ = counter_data
    bank.keep.zero_()
    before, after, target, stats = bank.pair(torch.arange(4), "mixed", 1)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert not target.any() and not stats.any()


def test_donor_loader_refuses_validation_and_output_provenance(counter_data, tmp_path):
    scenes, _, manifest = counter_data
    scenes.split = "val"
    with pytest.raises(ValueError, match="only.*train"):
        donor_arrays(tmp_path, manifest, scenes)


def test_donor_asset_cannot_include_extra_validation_scenes(counter_data, tmp_path):
    scenes, _, manifest = counter_data
    with np.load(tmp_path / "donors.npz") as z:
        arrays = {k: z[k].copy() for k in z.files}
    arrays = {k: np.concatenate((v, v[:1])) for k, v in arrays.items()}
    arrays["names"][-1] = "val"
    np.savez(tmp_path / "donors.npz", **arrays)
    with pytest.raises(ValueError, match="exactly the train"):
        donor_arrays(tmp_path, manifest, scenes)


def test_count_queries_use_known_absolute_counts_and_are_antisymmetric(counter_data):
    scenes, _, _ = counter_data
    queries = CountQueries(scenes, torch.device("cpu"))
    hist = torch.zeros(4, 96)
    hist[:, TYPES.index(("small", "red", "rubber", "cube"))] = 3
    hist[:, TYPES.index(("large", "green", "metal", "sphere"))] = 2
    score, report, _ = queries.score(hist)
    assert report["mae"] == 0 and report["within_half"] == 1
    assert (score[scenes.frame.label.eq(1)] > 0).all()
    np.testing.assert_allclose(score[scenes.pairs].sum(-1), 0)
    scenes.label = 1 - scenes.label
    other = CountQueries(scenes, torch.device("cpu"))
    np.testing.assert_array_equal(score, other.score(hist)[0])


@pytest.mark.parametrize("config", [p.stem for p in sorted(Path("configs").glob("*_count.yaml"))])
def test_every_counter_configuration_runs_full_objective_backward_and_update(counter_data, config):
    scenes, bank, _ = counter_data
    cfg = load_config(Path("configs") / (config + ".yaml"))
    device = torch.device("cpu")
    torch.manual_seed(5)
    model = DenseCounter(TinyDetector(), cfg["count"]["density"] == "joint")
    statistics = model.net.features[1].running_mean.clone()
    queries = CountQueries(scenes, device)
    report = train(cfg, model, bank, queries, device, steps=1)
    assert (
        report["updates"] == 1
        and report["amp_skipped_updates"] == 0
        and np.isfinite(report["history"][0]["loss"])
    )
    torch.testing.assert_close(model.net.features[1].running_mean, statistics, rtol=0, atol=0)
    freeze_statistics(model)
    assert not model.net.features[1].training


def test_selected_class_density_integrates_to_corresponding_histogram_entry():
    torch.manual_seed(57)
    raw = torch.randn(7, 20, 4, 6)
    hist, mass = DenseCounter.integrate(raw, torch.randn(7, 1, 4, 6))
    kinds = torch.tensor([0, 95, 48, 17, 23, 41, 82])
    density = DenseCounter.class_density(raw, mass, kinds)
    torch.testing.assert_close(density.sum((-2, -1)), hist[torch.arange(7), kinds])


def test_local_delta_penalizes_the_right_count_at_the_wrong_location():
    from src.count_run import local_delta_loss

    boxes = torch.tensor([[0.0, 0.0, 0.25, 0.25]])
    right = torch.zeros(1, 4, 4, requires_grad=True)
    wrong = torch.zeros(1, 4, 4, requires_grad=True)
    with torch.no_grad():
        right[0, 0, 0] = 1
        wrong[0, 3, 3] = 1
    assert right.sum() == wrong.sum()  # Same global +1, different spatial credit.
    accepted = torch.ones(1)
    assert local_delta_loss(right, boxes, accepted) == 0
    loss = local_delta_loss(wrong, boxes, accepted)
    assert loss > 1
    loss.backward()
    assert torch.isfinite(wrong.grad).all() and wrong.grad.abs().sum() > 0


def test_local_delta_rejected_insertion_has_no_gradient():
    from src.count_run import local_delta_loss

    delta = torch.randn(3, 4, 6, requires_grad=True)
    loss = local_delta_loss(delta, torch.zeros(3, 4), torch.zeros(3))
    loss.backward()
    assert loss == 0
    torch.testing.assert_close(delta.grad, torch.zeros_like(delta))


def test_returning_placement_boxes_preserves_sampling_and_rejected_boxes(counter_data):
    _, bank, _ = counter_data
    torch.manual_seed(88)
    first = bank.pair(torch.arange(4))
    torch.manual_seed(88)
    second = bank.pair(torch.arange(4), return_boxes=True)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    accepted = second[2].sum(1).bool()
    assert (second[4][accepted, 2:] > 0).all()
    assert not second[4][~accepted].any()


@pytest.mark.parametrize("scale", [1, 2])
def test_frozen_feature_training_never_calls_or_updates_the_encoder(counter_data, scale):
    from src.feature_bank import FrozenFeatures

    scenes, _, _ = counter_data
    model = DenseCounter(TinyDetector(), scale=scale)
    model.net.requires_grad_(False)
    model.net.head.requires_grad_(True)
    old = model.net.features[0].weight.detach().clone()
    bank = FrozenFeatures(model, scenes, torch.device("cpu"))
    assert bank.features.shape == (4, 64, 40 * scale, 60 * scale)
    assert not bank.features.is_inference()
    cfg = load_config(f"configs/{63 + scale:03d}_detail.yaml")

    def forbidden(_):
        raise AssertionError("CNN executed during cached readout training")

    model.net.pyramid = forbidden
    report = train(
        cfg, model, bank, CountQueries(scenes, torch.device("cpu")), torch.device("cpu"), steps=1
    )
    assert report["updates"] == 1
    torch.testing.assert_close(old, model.net.features[0].weight, rtol=0, atol=0)
