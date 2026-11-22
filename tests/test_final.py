"""Cold-start assets, dependency ordering and inference compatibility."""

import hashlib
import json

import pytest
import torch

from src.config import load_config
from src.detectors import Detector
from src.final import graph, prepare_inputs
from src.historical import HistoricalR50


def test_final_graph_cannot_export_before_every_fit_and_inference_stage():
    dependencies, order = graph()
    assert len(order) == len(set(order)) == len(dependencies)
    done = set()
    while len(done) < len(dependencies):
        ready = {k for k, v in dependencies.items() if set(v) <= done} - done
        assert ready
        done |= ready

    def parents(name):
        result = set(dependencies[name])
        for dependency in dependencies[name]:
            result |= parents(dependency)
        return result

    assert parents("export") == set(dependencies) - {"export"}
    assert not any("val" in name for name in dependencies)


def test_final_recipe_keeps_057_and_complete_training_budgets():
    cfg = load_config("configs/066_final.yaml")["pipeline"]
    assert cfg["submission"] == "057_quality"
    assert cfg["convnext"]["batch"] == 64 and cfg["convnext"]["epochs"] == 21
    assert cfg["union"]["epochs"] == 12 and cfg["union"]["seeds"] == [421, 853]
    assert cfg["counter"]["epochs"] == 18 and cfg["counter"]["batch"] == 96
    assert load_config("configs/057_quality.yaml")["blend"]["tta_weight"] == 0.1


def test_locked_graph_trains_new_branches_and_has_no_old_training_jobs():
    from src.locked import graph as locked_graph

    jobs, order = locked_graph()
    assert set(order) == set(jobs)
    assert {
        "convnext",
        "union421",
        "union853",
        "count53_421",
        "count53_853",
        "count54_421",
        "count54_853",
    } <= set(jobs)
    assert not {"teacher", "native", "thaw", "resnet", "pool"} & set(jobs)
    assert jobs["count54_421"] == ["convnext"]
    assert jobs["union421"] == ["unionprep"]


def test_pretrained_registry_rejects_task_checkpoint_even_when_checksummed(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    (root / "pretrained.json").write_text(json.dumps({"files": {"teacher.pt": "abc"}}))
    with pytest.raises(ValueError, match="only the four"):
        prepare_inputs(root, tmp_path / "work")


def test_pretrained_registry_validates_content_and_uses_only_declared_files(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    names = ["resnet50.pth", "convnext.pth", "siglip/config.json", "siglip/model.safetensors"]
    hashes = {}
    for name in names:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(name.encode())
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (root / "pretrained.json").write_text(json.dumps({"files": hashes}))
    prepare_inputs(root, tmp_path / "work")
    assert (tmp_path / "work/inputs/pretrained/resnet50.pth").read_bytes() == b"resnet50.pth"
    (root / "convnext.pth").write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        prepare_inputs(root, tmp_path / "other")


def test_historical_detector_state_replays_in_current_inference_without_remapping():
    torch.manual_seed(812)
    old = HistoricalR50(full=True).eval()
    cfg = load_config("configs/019_base.yaml")["model"]
    current = Detector(cfg).eval()
    current.load_state_dict(old.state_dict(), strict=True)
    image = torch.rand(1, 3, 160, 240)
    with torch.no_grad():
        torch.testing.assert_close(old(image), current(image)[0], rtol=0, atol=0)
