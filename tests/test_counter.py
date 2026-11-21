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


@pytest.mark.parametrize("config", [f"{n:03d}_count" for n in range(51, 56)])
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
