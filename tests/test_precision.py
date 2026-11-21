"""FP16 model outputs must never force loss reductions or accumulators to FP16."""

from pathlib import Path

import pytest
import torch

from src.config import load_config
from src.data import encode_programs
from src.engine import resnet_loss
from src.heads import TableHead
from src.losses import detection_loss, map_count_loss
from src.programs import parse_program
from src.siglip import attribute_loss


def empty_targets(raw):
    n, _, h, w = raw.shape
    return [
        torch.zeros(n, h, w),
        torch.zeros(n, 4, h, w),
        torch.full((n, 4, h, w), -100, dtype=torch.long),
        torch.zeros(n, h, w),
    ]


@pytest.mark.parametrize("kind", ["detection", "count_without_targets", "attributes"])
def test_empty_or_zero_loss_paths_remain_fp32(kind):
    raw = torch.full((128, 20, 40, 60), -2.19, dtype=torch.float16, requires_grad=True)
    if kind == "detection":
        loss = detection_loss([raw], targets=[empty_targets(raw)])
    elif kind == "count_without_targets":
        batch = dict(
            programs=encode_programs([parse_program("there is a red cube")], "cpu"),
            label=torch.ones(1),
            scenes=torch.zeros(1, dtype=torch.long),
        )
        loss = map_count_loss(raw, batch)
    else:
        raw = torch.full((384, 15), 100.0, dtype=torch.float16, requires_grad=True)
        loss = attribute_loss(raw, torch.zeros(384, 4, dtype=torch.long))
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()


RESNET_STAGES = [
    (path.stem, stage)
    for path in sorted(Path("configs").glob("*.yaml"))
    if load_config(path)["family"] == "resnet"
    for stage, epochs in zip(("objects", "simple", "all"), load_config(path)["train"]["stages"])
    if epochs
]


@pytest.mark.parametrize("name,stage", RESNET_STAGES)
def test_every_resnet_configuration_stage_accepts_fp16_outputs(name, stage):
    cfg = load_config(f"configs/{name}.yaml")
    levels = 3 if cfg["model"]["neck"] == "multi" else 1
    outputs = []
    targets = []
    for level in range(levels):
        raw = torch.zeros(2, 20, 40 // 2**level, 60 // 2**level, dtype=torch.float16)
        raw[:, 0] = -2.19
        raw.requires_grad_()
        outputs.append(raw)
        targets.append(empty_targets(raw))
    packed = dict(
        scenes=torch.tensor([0, 0, 1, 1]),
        programs=encode_programs(
            [
                parse_program("exactly one red cube"),
                parse_program("exactly two red cubes"),
            ]
            * 2,
            "cpu",
        ),
        label=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        pairs=torch.tensor([[0, 1], [2, 3]]),
        category=["counting"] * 4,
        aux=None,
    )
    loss = resnet_loss(cfg, TableHead(), outputs, outputs, packed, targets, stage, (160, 240))
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert all(raw.grad is not None and torch.isfinite(raw.grad).all() for raw in outputs)
