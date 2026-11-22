"""Regenerate the train-only silhouette teacher and donors from competition JPEGs."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import Scenes, Teacher
from .heads import GROUPS
from .regions import extract_split, fit_background
from .runtime import gradient_step, setup
from .silhouette import Silhouette, region_anchors


def prepare(data, work, device, recipe, *, limit=None, steps=None):
    work = Path(work)
    reference = work / "inputs/reference"
    reference.mkdir(parents=True, exist_ok=True)
    scenes = Scenes(data, "train", limit)
    print("STAGE teacher: train JPEG regions", flush=True)
    background = fit_background(Path(data) / "train", scenes.names, seed=recipe["seed"])
    table = extract_split(
        Path(data) / "train", scenes.names, background, workers=2, boundary=True, primitives=True
    )
    anchors = region_anchors(*table, scenes.programs, scenes.scene, scenes.label).to(device)
    table = tuple(torch.as_tensor(a, device=device) for a in table)
    rng, _ = setup(recipe["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = Silhouette(boundary=True, primitives=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, recipe["schedule"], eta_min=0.0003
    )
    updates = 0
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    for epoch in range(recipe["epochs"]):
        model.train()
        order = rng.permutation(len(scenes.names))
        for start in range(0, len(order), recipe["batch"]):
            idx = order[start : start + recipe["batch"]]
            packed = scenes.batch(idx, device)
            selected = tuple(a[idx] for a in table)
            raw = model.score(*selected, packed["scenes"], packed["programs"])
            weight = 1 + 0.7 * torch.as_tensor(
                np.isin(packed["category"], ("counting", "comparison", "spatial_counting")),
                device=device,
            )
            a, b = packed["pairs"].T
            loss = F.binary_cross_entropy_with_logits(raw, packed["label"], weight=weight)
            loss += F.binary_cross_entropy_with_logits(
                raw[a] - raw[b], packed["label"][a], weight=weight[a]
            )
            if packed["aux"] is not None:
                indices, words = packed["aux"]
                loss += 0.3 * model.existence_loss(*selected, indices, {"a": words})
            loss += 0.35 * model.anchor_loss(*selected, anchors[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_step(scaler, optimizer, list(model.parameters()), loss=loss)
            updates += 1
            if steps and updates >= steps:
                break
        schedule.step()
        print(
            json.dumps(dict(stage="teacher", epoch=epoch + 1, loss=float(loss.detach()))),
            flush=True,
        )
        if steps and updates >= steps:
            break
    model.eval()
    torch.save(model.cpu().state_dict(), reference / "scene_silhouette.pt")
    model.to(device)
    fields = {key: [] for key in ["p", "size", "color", "material", "shape"]}
    with torch.no_grad():
        for start in range(0, len(scenes.names), 128):
            enc = model.encode(*(a[start : start + 128] for a in table))
            obj, words = model.slot_logits(enc)
            values = [obj.sigmoid(), *[words[..., group].softmax(-1) for group in GROUPS]]
            for key, value in zip(fields, values):
                fields[key].append(value.cpu().numpy())
    fields = {key: np.concatenate(value) for key, value in fields.items()}
    features, xy, valid = (a.cpu().numpy() for a in table)
    boxes = features[..., 8:12].copy()
    path = work / "inputs/teacher"
    path.mkdir(exist_ok=True)
    np.savez(path / "train.npz", names=scenes.names, boxes=boxes, valid=valid, **fields)
    np.savez(work / "teacher_features.npz", features=features, xy=xy, valid=valid, **fields)
    pixel_boxes = np.rint(boxes * [240, 160, 240, 160]).astype(np.int16)
    x, y, w, h = pixel_boxes.transpose(2, 0, 1)
    keep = valid & (fields["p"] > 0.84) & (w >= 8) & (h >= 8) & (x >= 0) & (y >= 0)
    keep &= (x + w <= 240) & (y + h <= 160)
    size, color, material, shape = [
        np.where(fields[key].max(-1) >= 0.85, fields[key].argmax(-1), -1)
        for key in ["size", "color", "material", "shape"]
    ]
    joint = np.where((material >= 0) & (shape >= 0), 3 * material + shape, -1)
    labels = np.stack((size, color, joint), -1).astype(np.int8)
    masks = (255 * features[..., 47:147].clip(0, 1)).round().astype(np.uint8)
    np.savez(
        work / "donors.npz",
        names=scenes.names,
        keep=keep,
        labels=labels,
        box=pixel_boxes,
        mask=masks.reshape(*keep.shape, 10, 10),
    )
    np.save(work / "background.npy", background)
    return dict(train_scenes=len(scenes.names), teacher_epochs=epoch + 1, updates=updates)


def natural_subset(scenes, work):
    """The historical natural detector filtered only train scenes by teacher consistency."""
    from .posterior import _execute

    with np.load(Path(work) / "teacher_features.npz") as z:
        f = {k: z[k] for k in z.files}
    result = []
    for i, indices in enumerate(scenes.rows):
        keep = f["valid"][i] & (f["p"][i] > 0.84)
        if not 3 <= keep.sum() <= 10:
            continue
        top = [f[key][i, keep].argmax(-1) for key in ["size", "color", "material", "shape"]]
        codes = np.zeros(keep.sum(), np.int32)
        for group, picked in zip(GROUPS, top):
            codes |= 1 << group[picked]
        box = f["features"][i, keep, 8:12]
        xy = (box[:, :2] + box[:, 2:] * [0.5, 0.88]) * 2 - 1
        valid = np.ones((1, len(codes)), bool)
        truth = _execute(
            [scenes.programs[j] for j in indices],
            codes[None],
            valid,
            xy[None],
            f["p"][i, keep],
            np.zeros(7),
            np.full_like(valid, 0.47, dtype=float),
        )
        if (truth[0] != scenes.label[indices]).sum() <= 1:
            result.append(i)
    return np.asarray(result, dtype=np.int64)


def train_native(data, work, device, recipe, *, limit=None, steps=None):
    from .losses import detection_loss
    from .native import CenterDetector

    rng, _ = setup(recipe["seed"])
    scenes = Scenes(data, "train", limit)
    scenes.prepare_images(Path(work) / "images", device)
    teacher = Teacher(Path(work) / "inputs/teacher/train.npz", scenes)
    teacher.prepare_targets(device)
    eligible = natural_subset(scenes, work)
    if not len(eligible):
        raise ValueError("No train scenes pass the historical natural-detector consistency filter")
    model = CenterDetector().to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["rate"])
    updates = 0
    # Natural control: no object insertion and no count supervision.
    for epoch in range(recipe["epochs"]):
        model.train()
        chosen = eligible[rng.permutation(len(eligible))[: recipe["scenes"]]]
        for idx in np.array_split(chosen, max(1, len(chosen) // recipe["batch"])):
            flip = rng.random(len(idx)) < 0.5
            image = scenes.images(idx, device)
            image = torch.where(
                torch.as_tensor(flip, device=device)[:, None, None, None], image.flip(-1), image
            )
            loss = detection_loss([model(image)], targets=teacher.targets(idx, flip))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
            if steps and updates >= steps:
                break
        print(
            json.dumps(dict(stage="native", epoch=epoch + 1, loss=float(loss.detach()))), flush=True
        )
        if steps and updates >= steps:
            break
    torch.save(model.cpu().state_dict(), Path(work) / "native.pt")
    return dict(epochs=epoch + 1, updates=updates, eligible_train_scenes=len(eligible))
