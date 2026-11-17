"""Build train-only center supervision from a previously trained silhouette model."""

from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
from scripts.pixel_reasoning import ImageTable, SceneData, read_rows
from src.models.slot_program import EXCLUSIVE_GROUPS, WORD_INDEX
from src.pixel_regions import fit_background


@torch.no_grad()
def main():
    root = Path("data")
    out = Path("artefacts/research")
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    names = list(dict.fromkeys(row["image"] for row in read_rows(root, "train")))
    background = fit_background(root / "train", names)
    device = torch.device("cpu")
    train = SceneData(root, "train", background, device, boundary=True, primitives=True)
    model = ImageTable(boundary=True, primitives=True)
    model.load_state_dict(torch.load(out / "scene_silhouette.pt", weights_only=True, map_location="cpu"))
    model.eval()
    parts = []
    for _, regions, *_ in train.batches(128):
        encoded = model.encode(*regions)
        p, attr = model.slot_logits(encoded)
        groups = [attr[..., [WORD_INDEX[w] for w in group]].softmax(-1) for group in EXCLUSIVE_GROUPS]
        parts.append([p.sigmoid(), *groups])
    fields = {name: torch.cat(values).numpy() for name, values in zip(
        ("p", "size", "color", "material", "shape"), zip(*parts)
    )}
    features, xy, valid = [tensor.numpy() for tensor in train.table]
    np.savez_compressed(out / "teacher_train.npz", names=names, features=features, xy=xy, valid=valid,
                        background=background, **fields)
    print("Train-only scene teacher saved.", flush=True)


if __name__ == "__main__":
    main()
