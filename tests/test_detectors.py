import pytest
import torch

from src.config import load_config
from src.detectors import Detector


@pytest.mark.parametrize(
    "config", ["020_resnet", "024_resnet", "025_resnet", "026_resnet", "028_convnext"]
)
def test_each_detector_construction_backpropagates_through_its_backbone(config):
    cfg = load_config(f"configs/{config}.yaml")
    model = Detector(cfg["model"])
    outputs = model(torch.rand(2, 3, 64, 96))
    assert len(outputs) == (3 if cfg["model"]["neck"] == "multi" else 1)
    assert outputs[0].shape == (2, 20, 16, 24)
    sum(x.square().mean() for x in outputs).backward()
    first = model.stem[0].weight if model.is_resnet else model.trunk[0][0].weight
    assert first.grad is not None and first.grad.abs().sum() > 0
