from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import load_config
from src.data import encode_programs, reflected
from src.detectors import dense_table, tables
from src.heads import TableHead, load_head
from src.losses import dense_targets, detection_loss, map_count_loss
from src.metrics import metrics
from src.programs import parse_program
from src.runtime import Sink, immutable_inputs
from src.siglip import Vision, attribute_loss, crop_images


def test_registered_ids_are_unique_and_consecutive():
    paths = sorted(Path("configs").glob("*.yaml"))
    ids = [int(p.stem[:3]) for p in paths]
    assert ids == list(range(19, max(ids) + 1))
    for p in paths:
        assert load_config(p)["name"] == p.stem


@pytest.mark.parametrize(
    "caption",
    [
        "there is a red cube",
        "exactly two red cubes",
        "there are more cubes than spheres",
        "a red cube to the left of the green sphere",
        "there is no cube behind the blue sphere",
        "there is a red cube on the left side of the image",
    ],
)
def test_program_reflection_is_an_involution(caption):
    program = parse_program(caption)
    assert program is not None
    assert reflected(reflected(program)) == program
    assert reflected(program).a == program.a


def test_count_gradients_reach_peaks_excluded_from_the_object_table():
    raw = torch.zeros(1, 20, 8, 12)
    raw[:, 0] = -10
    raw.requires_grad_()
    assert not dense_table(raw, height=32, width=48)[2].any()
    batch = dict(
        programs=encode_programs([parse_program("exactly two red cubes")], "cpu"),
        label=torch.ones(1),
        scenes=torch.zeros(1, dtype=torch.long),
    )
    map_count_loss(raw, batch).backward()
    assert raw.grad[:, 0].abs().sum() > 0
    assert torch.isfinite(raw.grad).all()


def test_uncertain_teacher_boxes_reduce_only_their_background_penalty():
    raw = torch.zeros(1, 20, 40, 60, requires_grad=True)
    parts = [
        (
            np.array([[0.1, 0.1, 0.2, 0.2], [0.6, 0.6, 0.1, 0.1]]),
            np.array([0.5, 0.95]),
            np.array([True, True]),
            np.zeros((2, 4), dtype=np.int64),
        )
    ]
    plain = detection_loss([raw], parts, False)
    plain.backward()
    old = raw.grad.clone()
    raw.grad.zero_()
    detection_loss([raw], parts, True).backward()
    assert raw.grad[0, 0, 6, 9].abs() < old[0, 0, 6, 9].abs()
    torch.testing.assert_close(raw.grad[0, 0, 0, 0], old[0, 0, 0, 0])


def test_multiscale_decoding_does_not_count_one_object_twice():
    out = []
    for side in (40, 20, 10):
        raw = torch.zeros(1, 20, side, int(side * 1.5))
        raw[:, 0] = -10
        yy = 8 // (40 // side)
        xx = yy
        raw[0, 0, yy, xx] = 10
        out.append(raw)
    x, xy, valid = tables(out, (160, 240))
    assert valid.sum() == 1
    assert x.shape == (1, 20, 38)


def test_weak_only_background_preserves_all_foreground_targets():
    parts = [
        (
            np.array([[0.1, 0.1, 0.2, 0.2], [0.6, 0.6, 0.1, 0.1]]),
            np.array([0.5, 0.95]),
            np.array([True, True]),
            np.zeros((2, 4), dtype=np.int64),
        )
    ]
    all_boxes = dense_targets(parts, (40, 60), 4, torch.device("cpu"))
    weak = dense_targets(parts, (40, 60), 4, torch.device("cpu"), weak_only=True)
    for old, new in zip(all_boxes[:3], weak[:3]):
        torch.testing.assert_close(old, new, rtol=0, atol=0)
    torch.testing.assert_close(weak[3][0, 4:12, 6:18], all_boxes[3][0, 4:12, 6:18])
    assert all_boxes[3][0, 24:28, 36:42].sum() > 0
    assert weak[3][0, 24:28, 36:42].sum() == 0
    raw = torch.zeros(1, 20, 40, 60, requires_grad=True)
    detection_loss([raw], soft=True, targets=[all_boxes]).backward()
    old = raw.grad.clone()
    raw.grad.zero_()
    detection_loss([raw], soft=True, targets=[weak]).backward()
    torch.testing.assert_close(raw.grad[0, 0, 6, 9], old[0, 0, 6, 9])
    assert raw.grad[0, 0, 24, 36] > old[0, 0, 24, 36]


def test_head_migration_rejects_unknown_parameters_and_preserves_predictions():
    model = TableHead()
    state = model.state_dict()
    state["embed.0.weight"] = torch.zeros(64, 64)
    clone = TableHead()
    load_head(clone, state)
    for a, b in zip(model.parameters(), clone.parameters()):
        torch.testing.assert_close(a, b)
    state["surprise.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        load_head(clone, state)


def test_pair_ties_and_auc_ties_are_reported_separately():
    frame = pd.DataFrame(
        dict(
            id=[1, 2, 3, 4],
            image=["a", "a", "b", "b"],
            pair_id=[0, 0, 1, 1],
            label=[1, 0, 1, 0],
            category=["counting"] * 4,
        )
    )
    result = metrics(frame, np.array([0.0, 0.0, -1.0, 1.0]))
    assert result["wrong_pairs"] == 1 and result["tied_pairs"] == 1
    assert result["pair_accuracy"] == 0.25
    assert result["auc"] == 0.125


def test_artifact_sink_archives_scratch_copies_and_inputs_reject_outputs(tmp_path, monkeypatch):
    root = tmp_path / "artefacts"
    sink = Sink(root, "run", tmp_path / "work")
    original = Path.open

    def guarded(path, mode="r", *args, **kwargs):
        if path.is_relative_to(root) and "r" in mode:
            raise AssertionError("Artifact read")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    sink.json("report.json", dict(ok=True))
    sink.finish()
    with pytest.raises(ValueError):
        immutable_inputs(root)


@pytest.mark.parametrize("mode", ["tail", "adapter"])
def test_siglip_updates_only_the_requested_visual_parameters(mode):
    from transformers import SiglipVisionConfig, SiglipVisionModel

    settings = load_config("configs/032_siglip.yaml")["siglip"]
    settings = dict(settings, mode=mode, pool_grid=8)
    tiny = SiglipVisionModel(
        SiglipVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            image_size=64,
            patch_size=8,
        )
    )
    vision = Vision(None, settings, model=tiny).float()
    vision.adapting = True
    values = vision.embedding(torch.rand(2, 3, 64, 64))
    loss = attribute_loss(vision.classifier(values), torch.zeros(2, 4, dtype=torch.long))
    loss.backward()
    assert all(
        p.grad is None for layer in vision.core.encoder.layers[:2] for p in layer.parameters()
    )
    if mode == "tail":
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in vision.core.encoder.layers[-1].parameters()
        )
    else:
        assert all(p.grad is None for p in vision.backbone.parameters())
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in vision.adapters.parameters()
        )


def test_letterbox_preserves_selected_pixels_and_uses_neutral_padding():
    image = torch.ones(1, 3, 20, 40)
    crop = crop_images(image, torch.tensor([[0.0, 0.0, 1.0, 1.0]]), size=32, mode="letterbox")
    torch.testing.assert_close(crop[:, :, 0], torch.full_like(crop[:, :, 0], 0.5))
    torch.testing.assert_close(crop[:, :, 16], torch.ones_like(crop[:, :, 16]))
