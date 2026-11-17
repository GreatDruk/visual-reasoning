"""Reweight scene hypotheses using a count cue learned from reviewed train crops."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from crop_encoder_probe import load_regions, read_cases
from learned_center_fusion import Fusion
from overlap_review import select
from scene_ranking import ROOT, evaluate, tables


def main():
    torch.set_num_threads(3)
    train = read_cases().query("split == 'train'")
    columns = ("priority", "split", "image", "color", "x0", "y0", "x1", "y1", "wide_separation")
    val = pd.DataFrame(select("val", 1000), columns=columns)
    x_train, color_train, _ = load_regions(train)
    x_val, color_val, _ = load_regions(val)
    head = make_pipeline(StandardScaler(),
                         LogisticRegression(C=.1, solver="liblinear", class_weight="balanced"))
    head.fit(np.c_[x_train, color_train], train.number.to_numpy()==2)
    score = head.decision_function(np.c_[x_val, color_val])
    cues = {}
    for row, weight in zip(val.itertuples(index=False), score, strict=True):
        bounds = np.array([row.x0, row.y0, row.x1, row.y1]) / [240,160,240,160]
        cues[row.image] = row.color, bounds, weight
    teacher = torch.load(ROOT / "scene_silhouette.pt", weights_only=True, map_location="cpu")
    model = Fusion(teacher).eval()
    model.load_state_dict(torch.load(ROOT / "local_centers_final/fusion.pt", weights_only=True, map_location="cpu"))
    frame, scenes = tables("val", model, 1024, cues)
    others = []
    for path, key in (("center_proposals8","permissive"),("region_count_supervision","joint"),("native_natural_epoch21","joint")):
        with np.load(ROOT / f"{path}.npz") as f:
            assert np.array_equal(f["ids"], frame.id)
            others.append(f[key])
    coef = np.zeros(next(row[0].shape[-1] for _, _, row in scenes if row is not None), np.float32)
    values = {}
    for strength in (0, .25, .5, 1):
        coef[-1] = strength
        component = evaluate(frame, scenes, coef)
        output = (component + sum(others)) / 4
        values[f"scale_{strength:g}"] = output
        print(strength, "ensemble", roc_auc_score(frame.label, output),
              "wrong pairs", (((2*frame.label-1)*output)<0).sum()//2, flush=True)
    coef[:] = 0
    for strength in (.75, .95):
        _, promoted = tables("val", model, 1024, cues, promotion=strength)
        output = (evaluate(frame, promoted, coef) + sum(others)) / 4
        values[f"promote_{strength:g}"] = output
        print("promotion", strength, "ensemble", roc_auc_score(frame.label, output),
              "wrong pairs", (((2*frame.label-1)*output)<0).sum()//2, flush=True)
    np.savez(ROOT / "review_crop_scene_probe.npz", ids=frame.id, **values)


if __name__ == "__main__":
    main()
