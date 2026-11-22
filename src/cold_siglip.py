"""Historical sigft pooling/PCA/readout, rebuilt exclusively from train crops."""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .cold_train import fit_head, projection, save, table
from .config import load_config
from .crops import Crops, Projection, feature_table
from .data import Scenes, Teacher
from .engine import predict
from .heads import FEATURES, GROUPS, TableHead
from .native import CenterDetector
from .runtime import Progress, amp, gradient_step, load_state, setup
from .siglip import Vision


def settings(project):
    return load_config(Path(project) / "configs/029_siglip.yaml")["siglip"]


def tables(data, work, device, *, limit=None):
    folder = Path(work) / "siglip"
    folder.mkdir(exist_ok=True)
    model = CenterDetector().to(device).eval()
    model.load_state_dict(load_state(Path(work) / "inputs/reference/thaw.pt"))
    matrix = projection(device)
    for split in ["train", "test"]:
        scenes = Scenes(data, split, limit)
        scenes.prepare_images(Path(work) / "images", device)
        bank = table(model, scenes, device, matrix)
        np.savez(folder / f"{split}_table.npz", x=bank[0], xy=bank[1], valid=bank[2])


def crops(data, work, split, limit=None):
    scenes = Scenes(data, split, limit)
    with np.load(Path(work) / f"siglip/{split}_table.npz") as z:
        bank = tuple(z[k] for k in ["x", "xy", "valid"])
    teacher = Teacher(Path(work) / "inputs/teacher/train.npz", scenes) if split == "train" else None
    return Crops(scenes, bank, teacher)


def vision(project, work, device):
    model = Vision(Path(work) / "inputs/pretrained/siglip", settings(project)).float().to(device)
    if device.type == "cuda":
        model.backbone.half()
    return model


@torch.no_grad()
def extract(project, data, work, device, recipe, split, shard, *, limit=None):
    setup(recipe["seed"])
    values = crops(data, work, split, limit)
    values.scenes.prepare_images(Path(work) / "images", device)
    rows = np.arange(shard, len(values.scene), 2)
    model = vision(project, work, device).eval()
    if split == "test":
        model.pool.load_state_dict(load_state(Path(work) / "inputs/reference/siglip_pool.pt"))
    width = model.backbone.config.hidden_size
    folder = Path(work) / "siglip"
    info = {"rows": rows}
    if values.labels is not None:
        info["labels"] = values.labels[rows]
    np.savez(folder / f"{split}_info{shard}.npz", **info)
    shape = (len(rows), 64, width) if split == "train" else (len(rows), width)
    path = folder / f"{split}_{'tokens' if split == 'train' else 'raw'}{shard}.npy"
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    batch = recipe["extract_batch"] if device.type == "cuda" else 2
    progress = Progress(f"siglip/{split}/shard{shard}", len(rows), device, interval=90)
    for first in range(0, len(rows), batch):
        selected = rows[first : first + batch]
        with amp(device):
            pixels = values.pixels(selected, device, settings(project))
            if split == "train":
                hidden, _ = model.tokens(pixels)
                value = model.reduce(hidden)
            else:
                value = model.embedding(pixels)
        out[first : first + len(selected)] = value.half().cpu().numpy()
        progress.update(first + len(selected))
    out.flush()
    del out


def pool(project, work, device, recipe, *, steps=None):
    rng, _ = setup(recipe["seed"])
    folder = Path(work) / "siglip"
    model = vision(project, work, device)
    head = model.pool.train()
    linear = model.classifier.train()
    del model
    tokens = [np.load(folder / f"train_tokens{s}.npy", mmap_mode="r") for s in (0, 1)]
    labels = [np.load(folder / f"train_info{s}.npz")["labels"] for s in (0, 1)]
    examples = np.array([(s, i) for s in (0, 1) for i in np.flatnonzero((labels[s] >= 0).any(1))])
    if not len(examples):
        raise ValueError("No confident train crop labels for historical sigft")
    optimizer = torch.optim.AdamW(
        [*head.parameters(), *linear.parameters()], lr=recipe["pool_rate"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    params = [*head.parameters(), *linear.parameters()]
    updates = 0
    for epoch in range(recipe["pool_epochs"]):
        for idx in np.array_split(
            rng.permutation(len(examples)),
            max(1, (len(examples) + recipe["pool_batch"] - 1) // recipe["pool_batch"]),
        ):
            chosen = examples[idx]
            x = torch.from_numpy(np.stack([tokens[s][i] for s, i in chosen])).to(
                device, dtype=torch.float32
            )
            target = torch.from_numpy(np.stack([labels[s][i] for s, i in chosen])).to(
                device, dtype=torch.long
            )
            with amp(device):
                scores = linear(head(x))
            loss = sum(
                F.cross_entropy(a, target[:, j], ignore_index=-100, reduction="sum")
                for j, a in enumerate(scores.float().split([len(g) for g in GROUPS], -1))
            )
            loss = loss / target.ge(0).sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            gradient_step(scaler, optimizer, params, loss=loss, max_norm=float("inf"))
            updates += 1
            if steps and updates >= steps:
                break
        print(
            f"STAGE siglip pooling epoch {epoch + 1}/{recipe['pool_epochs']}: {float(loss.detach()):.5f}",
            flush=True,
        )
        if steps and updates >= steps:
            break
    save(
        Path(work) / "inputs/reference/siglip_pool.pt",
        {k: v.cpu() for k, v in head.state_dict().items()},
    )
    head.eval()
    with torch.no_grad(), amp(device):
        for shard in (0, 1):
            output = np.lib.format.open_memmap(
                folder / f"train_raw{shard}.npy",
                mode="w+",
                dtype=np.float16,
                shape=(len(tokens[shard]), head.probe.shape[-1]),
            )
            for first in range(0, len(output), 256):
                x = torch.from_numpy(tokens[shard][first : first + 256].copy()).to(
                    device, dtype=torch.float32
                )
                output[first : first + len(x)] = head(x).half().cpu().numpy()
            output.flush()
            del output
    for shard in (0, 1):
        tokens[shard]._mmap.close()
        (folder / f"train_tokens{shard}.npy").unlink()


def assembled_raw(work, split):
    folder = Path(work) / "siglip"
    shards = [np.load(folder / f"{split}_raw{s}.npy", mmap_mode="r") for s in (0, 1)]
    result = np.empty((sum(len(x) for x in shards), shards[0].shape[-1]), np.float32)
    for shard in (0, 1):
        with np.load(folder / f"{split}_info{shard}.npz", allow_pickle=False) as info:
            result[info["rows"]] = shards[shard]
    return result


def scene_head(data, work, device, recipe, *, limit=None, steps=None):
    train = crops(data, work, "train", limit)
    raw = assembled_raw(work, "train")
    projector = Projection(raw, train, seed=recipe["pca_seed"])
    save(Path(work) / "inputs/reference/sigft_pca.pt", projector.state())
    bank = feature_table(train, raw, projector)
    policy = dict(
        seed=recipe["seed"],
        epochs=recipe["head_epochs"],
        batch=recipe["head_batch"],
        rate=recipe["head_rate"],
    )
    head = fit_head(
        train.scenes,
        bank,
        work,
        device,
        policy,
        warm=Path(work) / "inputs/reference/thaw_head.pt",
        steps=steps,
    )
    save(
        Path(work) / "inputs/reference/sigft.pt", {k: v.cpu() for k, v in head.state_dict().items()}
    )


def infer(data, work, device, *, limit=None):
    test = crops(data, work, "test", limit)
    state = load_state(Path(work) / "inputs/reference/sigft_pca.pt")
    projector = Projection.__new__(Projection)
    projector.mean = state["mean"].numpy()
    projector.components = state["components"].numpy()
    projector.scale = state["scale"].numpy()
    bank = feature_table(test, assembled_raw(work, "test"), projector)
    head = TableHead(FEATURES + 128).to(device)
    head.load_state_dict(load_state(Path(work) / "inputs/reference/sigft.pt"))
    score = predict(test.scenes, head, bank, device)
    np.savez(Path(work) / "sigft_test.npz", ids=test.scenes.frame.id.to_numpy(), score=score)
