"""Exercise complete training paths with local tiny weights and real image files."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.config import load_config
from src.crops import token_storage, train_siglip
from src.data import Scenes, Teacher
from src.engine import predict
from src.heads import TableHead
from src.siglip import Vision


def test_token_cache_is_removed_after_a_failed_step(tmp_path):
    path = tmp_path / "pool.npy"
    with pytest.raises(RuntimeError):
        with token_storage(path, (2, 4, 8)) as cache:
            cache[:] = 1
            raise RuntimeError("failed training step")
    assert not path.exists()


@pytest.mark.parametrize("config", ["029_siglip", "033_siglip", "034_siglip", "038_siglip"])
def test_crop_training_runs_pooling_projection_head_and_adaptation(tmp_path, monkeypatch, config):
    from transformers import SiglipVisionConfig, SiglipVisionModel

    from src import crops

    rng = np.random.default_rng(42)
    for split in ("train", "val"):
        (tmp_path / split).mkdir()
        rows = []
        for i in range(2):
            name = f"{split}{i}.png"
            Image.fromarray(rng.integers(0, 255, (160, 240, 3), dtype=np.uint8)).save(
                tmp_path / split / name
            )
            for label, color in [(1, "red"), (0, "blue")]:
                rows.append(
                    dict(
                        id=2 * i + label,
                        image=name,
                        caption=f"there is a {color} cube",
                        category="attribute",
                        pair_id=i,
                        label=label,
                    )
                )
        pd.DataFrame(rows).to_csv(tmp_path / f"{split}.csv", index=False)
    train, val = (Scenes(tmp_path, s) for s in ("train", "val"))
    boxes = np.tile(np.array([0.2, 0.2, 0.4, 0.4], np.float32), (2, 1, 1))
    groups = [np.full((2, 1, n), 1 / n, np.float32) for n in (2, 8, 2, 3)]
    for prob in groups:
        prob[:] = 0.01
        prob[..., 0] = 1 - 0.01 * (prob.shape[-1] - 1)
    np.savez(
        tmp_path / "teacher.npz",
        names=train.names,
        boxes=boxes,
        p=np.full((2, 1), 0.95),
        valid=np.ones((2, 1), bool),
        **dict(zip(("size", "color", "material", "shape"), groups)),
    )
    teacher = Teacher(tmp_path / "teacher.npz", train)
    x = np.zeros((2, 1, 38), np.float32)
    x[..., 0] = 3
    x[..., 16:20] = boxes
    begin = 1
    for prob in groups:
        x[..., begin : begin + prob.shape[-1]] = np.log(prob)
        begin += prob.shape[-1]
    table = (x, np.zeros((2, 1, 2), np.float32), np.ones((2, 1), bool))
    inputs = tmp_path / "inputs"
    (inputs / "reference").mkdir(parents=True)
    state = TableHead().state_dict()
    torch.save(state, inputs / "reference/scene_silhouette.pt")
    torch.save(state, inputs / "reference/thaw_head.pt")
    cfg = load_config(f"configs/{config}.yaml")
    cfg["siglip"]["pool_epochs"] = 1
    cfg["siglip"]["head_epochs"] = 1
    cfg["siglip"]["pool_batch"] = 2
    cfg["siglip"]["adapt_batch"] = 2
    if cfg["siglip"]["adapt_steps"]:
        cfg["siglip"]["adapt_steps"] = 1
    if config == "038_siglip":
        cfg["siglip"]["intermediate"] = 1

    def tiny(folder, settings):
        model = SiglipVisionModel(
            SiglipVisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=4,
                num_attention_heads=4,
                image_size=256,
                patch_size=32,
            )
        )
        return Vision(None, settings, model=model)

    monkeypatch.setattr(crops, "Vision", tiny)
    vision, head, bank, projection, report = train_siglip(
        cfg,
        train,
        val,
        teacher,
        table,
        table,
        inputs,
        tmp_path,
        torch.device("cpu"),
        rng,
        step_limit=1,
    )
    scores = predict(val, head, bank, torch.device("cpu"), draws=16)
    assert scores.shape == (4,) and np.isfinite(scores).all()
    assert report["matched_crops"] == 2
    assert not (tmp_path / "pool.npy").exists()
    assert bank[0].shape == (2, 1, 166)
    assert all(torch.is_tensor(x) for x in projection.state().values())
