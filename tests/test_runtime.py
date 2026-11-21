"""Numerical recovery and cached data preserve the established objectives."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data import Scenes, Teacher, encode_programs, reflected
from src.losses import dense_targets
from src.runtime import gradient_step, zero_loss


def test_gradient_step_clips_finite_gradients():
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([6.0, 8.0])
    opt = torch.optim.SGD([parameter], lr=1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    done, norm = gradient_step(scaler, opt, [parameter], loss=torch.tensor(1.0))
    assert done and norm == 10
    torch.testing.assert_close(parameter.detach(), torch.tensor([-3.0, -4.0]))


def test_zero_loss_cannot_overflow_at_the_kaggle_detector_batch():
    raw = torch.zeros(128, 20, 40, 60, dtype=torch.float16)
    raw[:, 0] = -2.19
    raw.requires_grad_()
    assert torch.isfinite(raw).all() and not torch.isfinite(raw.sum() * 0)
    value = zero_loss(raw)
    assert value.dtype == torch.float32 and value.item() == 0
    value.backward()
    assert torch.isfinite(raw.grad).all() and not raw.grad.any()


def test_amp_overflow_skips_the_update_and_reduces_scale():
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    parameter.grad = torch.tensor([float("inf")])
    opt = torch.optim.SGD([parameter], lr=1.0)

    class SimulatedScaler:
        scale = 128.0

        def unscale_(self, optimizer):
            self.invalid = not torch.isfinite(parameter.grad).all()

        def is_enabled(self):
            return True

        def step(self, optimizer):
            if not self.invalid:
                optimizer.step()

        def update(self):
            if self.invalid:
                self.scale /= 2

    scaler = SimulatedScaler()
    done, _ = gradient_step(scaler, opt, [parameter], loss=torch.tensor(1.0))
    assert not done and scaler.scale == 64 and parameter.item() == 2 and parameter.grad is None


def test_full_precision_bad_gradients_are_not_ignored():
    p = torch.nn.Parameter(torch.tensor([2.0]))
    p.grad = torch.tensor([float("nan")])
    with pytest.raises(FloatingPointError):
        gradient_step(torch.amp.GradScaler("cuda", enabled=False), torch.optim.SGD([p], lr=1), [p])
    assert p.item() == 2


def test_real_scaler_recovers_after_scaled_gradient_overflow():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.SGD([p], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=128)
    loss = p.sum() * 1e38
    scaler.scale(loss).backward()
    done, _ = gradient_step(scaler, opt, [p], loss=loss)
    assert not done and scaler.get_scale() == 64 and p.item() == 1
    loss = p.sum()
    scaler.scale(loss).backward()
    done, _ = gradient_step(scaler, opt, [p], loss=loss)
    assert done
    torch.testing.assert_close(p.detach(), torch.tensor([0.9]))


def test_scene_cache_and_program_batches_preserve_values(tmp_path, monkeypatch):
    (tmp_path / "train").mkdir()
    rows = []
    for i in range(2):
        Image.fromarray(np.full((160, 240, 3), i * 70 + 30, np.uint8)).save(
            tmp_path / "train" / f"{i}.png"
        )
        for y, relation in ((1, "left"), (0, "right")):
            rows.append(
                dict(
                    id=2 * i + y,
                    image=f"{i}.png",
                    category="spatial",
                    pair_id=i,
                    label=y,
                    caption=f"a red cube to the {relation} of the green sphere",
                )
            )
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    data = Scenes(tmp_path, "train")
    indices = np.array([1, 0])
    device = torch.device("cpu")
    expected = data.images(indices, device)
    data.prepare_images(tmp_path / "cache", device)
    monkeypatch.setattr(Image, "open", lambda *_: pytest.fail("JPEG decoded again"))
    torch.testing.assert_close(data.images(indices, device), expected)
    batch = data.batch(indices, device, np.array([True, False]))
    programs = [
        reflected(p) if scene == 0 else p
        for p, scene in zip((data.programs[r] for r in batch["rows"]), batch["scenes"])
    ]
    for k, v in encode_programs(programs, device).items():
        torch.testing.assert_close(batch["programs"][k], v)


@pytest.mark.parametrize("levels", [1, 3])
@pytest.mark.parametrize("weak_only", [False, True])
def test_prepared_targets_match_original_rasterization(levels, weak_only):
    teacher = Teacher.__new__(Teacher)
    teacher.names = ["a", "b"]
    teacher._targets = None
    teacher.fields = dict(
        boxes=np.tile(
            np.array([[0.1, 0.2, 0.15, 0.2], [0.5, 0.5, 0.3, 0.4]], np.float32), (2, 1, 1)
        ),
        p=np.array([[0.95, 0.5], [0.92, 0.93]], np.float32),
        valid=np.ones((2, 2), bool),
    )
    for name, n in zip(("size", "color", "material", "shape"), (2, 8, 2, 3)):
        p = np.full((2, 2, n), 0.01, np.float32)
        p[..., 0] = 1 - 0.01 * (n - 1)
        teacher.fields[name] = p
    device = torch.device("cpu")
    teacher.prepare_targets(device, levels, True, weak_only=weak_only)
    idx = np.array([1, 0])
    flips = np.array([True, False])
    actual = teacher.targets(idx, flips)
    for level, targets in enumerate(actual):
        expected = dense_targets(
            teacher.batch(idx, flips),
            (40 // 2**level, 60 // 2**level),
            4 * 2**level,
            device,
            level,
            levels > 1,
            weak_only=weak_only,
        )
        for a, b in zip(targets, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
