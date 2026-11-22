"""Fixed cold-start training stages, without validation or batch calibration."""

import json
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .data import Scenes, Teacher
from .detectors import dense_table
from .engine import device_table, train_resnet
from .heads import N_ATTRS
from .historical import HistoricalR50
from .losses import caption_loss, detection_loss
from .native import CenterDetector
from .runtime import amp, gradient_step, load_state, setup
from .silhouette import HistoricalHead, Silhouette


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    torch.save(value, pending)
    pending.replace(path)


def teacher_state(work):
    result = load_state(Path(work) / "inputs/reference/scene_silhouette.pt")
    result["relation_offset"].zero_()
    return result


def projection(device):
    gen = torch.Generator().manual_seed(217)
    return torch.linalg.qr(torch.randn(40, 3 + N_ATTRS, generator=gen))[0].to(device)


def native_forward(model, image, matrix=None):
    with amp(image.device):
        if matrix is None:
            raw = model(image)
            latent = None
        else:
            latent = model.features(image)
            raw = model.head(latent)
            latent = torch.einsum("bdhw,ds->bshw", latent, matrix)
    return dense_table(raw.float(), latent=None if latent is None else latent.float())


@torch.inference_mode()
def table(model, scenes, device, matrix=None, batch=128):
    model.eval()
    values = []
    for start in range(0, len(scenes.names), batch if device.type == "cuda" else 2):
        idx = np.arange(
            start, min(start + (batch if device.type == "cuda" else 2), len(scenes.names))
        )
        values.append(
            tuple(
                x.cpu().numpy() for x in native_forward(model, scenes.images(idx, device), matrix)
            )
        )
    return tuple(np.concatenate(x) for x in zip(*values))


def fit_head(scenes, bank, work, device, recipe, *, warm=None, steps=None):
    rng, _ = setup(recipe["seed"])
    head = HistoricalHead(teacher_state(work), features=bank[0].shape[-1]).to(device)
    if warm is not None:
        state = load_state(warm)
        own = head.state_dict()
        for k, v in state.items():
            if k == "residual.0.weight":
                own[k].zero_()
                own[k][:, : v.shape[1]].copy_(v)
            else:
                own[k].copy_(v)
        head.load_state_dict(own)
    optimizer = torch.optim.AdamW(head.parameters(), lr=recipe["rate"], weight_decay=0.0001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, recipe["epochs"], eta_min=0.0003
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    bank = device_table(bank, device)
    updates = 0
    for epoch in range(recipe["epochs"]):
        head.train()
        order = rng.permutation(len(scenes.names))
        for start in range(0, len(order), recipe["batch"]):
            idx = order[start : start + recipe["batch"]]
            loss = caption_loss(head, tuple(x[idx] for x in bank), scenes.batch(idx, device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_step(scaler, optimizer, list(head.parameters()), loss=loss)
            updates += 1
            if steps and updates >= steps:
                break
        schedule.step()
        print(
            json.dumps(dict(stage="historical_head", epoch=epoch + 1, loss=float(loss.detach()))),
            flush=True,
        )
        if steps and updates >= steps:
            break
    return head.eval()


def historical(name, data, work, assets, recipe, device, *, limit=None, steps=None):
    seed = recipe["seed"]
    rng, _ = setup(seed)
    scenes = Scenes(data, "train", limit)
    scenes.prepare_images(Path(work) / "images", device)
    teacher = Teacher(Path(work) / "inputs/teacher/train.npz", scenes)
    teacher.prepare_targets(device)
    # Historical pseudo_examples constructed the silhouette model before the
    # source detector. Its parameter allocation changes the later RNG state.
    Silhouette(boundary=True, primitives=True)
    if name == "r50_img":
        model = HistoricalR50(Path(assets) / "resnet50.pth", full=True)
        matrix = None
    else:
        model = CenterDetector()
        model.load_state_dict(load_state(Path(work) / "native.pt"))
        matrix = projection(device)
    model = model.to(device, memory_format=torch.channels_last)
    head = HistoricalHead(teacher_state(work)).to(device)
    if name == "thaw":
        head.load_state_dict(load_state(Path(work) / "native_head.pt"))
    optimizer = torch.optim.AdamW(
        [
            dict(params=model.parameters(), lr=recipe["rate"]),
            dict(params=head.parameters(), lr=0.0015),
        ]
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, recipe["epochs"], eta_min=recipe["rate"] * 0.08
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    params = [*model.parameters(), *head.parameters()]
    updates = 0
    effective = 128
    batch = recipe["batch"] if device.type == "cuda" else min(2, recipe["batch"])
    for epoch in range(recipe["epochs"]):
        model.train()
        head.train()
        if name == "thaw":
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
        order = rng.permutation(len(scenes.names))
        for first in range(0, len(order), effective):
            block = order[first : first + effective]
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(block), batch):
                idx = block[start : start + batch]
                loss = caption_loss(
                    head,
                    native_forward(model, scenes.images(idx, device), matrix),
                    scenes.batch(idx, device),
                )
                if name == "r50_img":
                    selected = np.array(
                        [
                            (teacher.fields["valid"][i] & (teacher.fields["p"][i] > 0.84)).sum()
                            >= 2
                            for i in idx
                        ]
                    )
                    if selected.any():
                        chosen = idx[selected]
                        flip = rng.random(len(chosen)) < 0.5
                        image = scenes.images(chosen, device)
                        image = torch.where(
                            torch.as_tensor(flip, device=device)[:, None, None, None],
                            image.flip(-1),
                            image,
                        )
                        with amp(device):
                            raw = model(image)
                        loss += 0.65 * detection_loss(
                            [raw.float()], targets=teacher.targets(chosen, flip)
                        )
                scaler.scale(loss * len(idx) / len(block)).backward()
            gradient_step(scaler, optimizer, params, loss=loss)
            updates += 1
            if steps and updates >= steps:
                break
        schedule.step()
        print(json.dumps(dict(stage=name, epoch=epoch + 1, loss=float(loss.detach()))), flush=True)
        if steps and updates >= steps:
            break
    reference = Path(work) / "inputs/reference"
    save(reference / f"{name}.pt", {k: v.cpu() for k, v in model.state_dict().items()})
    save(reference / f"{name}_head.pt", {k: v.cpu() for k, v in head.state_dict().items()})
    return dict(epochs=epoch + 1, updates=updates)


def native_head(data, work, device, recipe, *, limit=None, steps=None):
    setup(42)
    scenes = Scenes(data, "train", limit)
    scenes.prepare_images(Path(work) / "images", device)
    model = CenterDetector().to(device).eval()
    model.load_state_dict(load_state(Path(work) / "native.pt"))
    bank = table(model, scenes, device, projection(device))
    del model
    head = fit_head(scenes, bank, work, device, recipe, steps=steps)
    save(Path(work) / "native_head.pt", {k: v.cpu() for k, v in head.state_dict().items()})


def convnext(project, data, work, device, recipe, *, limit=None, steps=None):
    cfg = load_config(Path(project) / "configs" / f"{recipe['config']}.yaml")
    cfg["train"].update(
        seed=recipe["seed"],
        epochs=recipe["epochs"],
        batch=128 if recipe.get("historical_warmup") else recipe["batch"],
        calibrate=bool(recipe.get("historical_warmup")),
    )
    scenes = Scenes(data, "train", limit)
    scenes.prepare_images(Path(work) / "images", device)
    teacher = Teacher(Path(work) / "inputs/teacher/train.npz", scenes)
    rng, _ = setup(recipe["seed"])
    model, head, report = train_resnet(
        cfg,
        scenes,
        teacher,
        Path(work) / "inputs",
        device,
        rng,
        steps,
        fixed_batch=recipe["batch"] if recipe.get("historical_warmup") else None,
        expected_first_epoch_loss=recipe.get("first_epoch_loss") if limit is None else None,
    )
    save(
        Path(work) / "convnext.pt",
        dict(
            config=cfg,
            model={k: v.cpu() for k, v in model.state_dict().items()},
            head={k: v.cpu() for k, v in head.state_dict().items()},
        ),
    )
    return report


class TableBank:
    def __init__(self, values, device):
        self.values = tuple(torch.as_tensor(x, device=device) for x in values)
        self.fields = {"x": self.values[0]}

    def select(self, indices, phases):
        if np.any(phases):
            raise ValueError("Final U045 trains on native phase only")
        return tuple(v[indices] for v in self.values)


def union(project, data, work, manifest, device, recipe, seed, *, limit=None, steps=None):
    from .recovery import train_head
    from .recovery_bank import load_sources
    from .union import detector_rows, merge

    cfg = load_config(Path(project) / "configs" / f"{recipe['config']}.yaml")
    cfg["train"]["seed"] = seed
    scenes = Scenes(data, "train", limit)
    scenes.prepare_images(Path(work) / "images", device)
    primary, head, secondary, other, matrix = load_sources(
        cfg, Path(work) / "inputs", Path(work), manifest, device
    )
    parts = []
    zero = np.zeros((20, 256), np.float16)
    with torch.no_grad():
        for first in range(0, len(scenes.names), 128 if device.type == "cuda" else 2):
            idx = np.arange(
                first, min(first + (128 if device.type == "cuda" else 2), len(scenes.names))
            )
            image = scenes.images(idx, device)
            with amp(device):
                raw = primary.head(primary.pyramid(image)[0])
                latent = secondary.features(image)
                right = secondary.head(latent)
                desc = torch.einsum("bdhw,ds->bshw", latent, matrix)
            left = detector_rows(head, dense_table(raw.float(), min_logit=-3.663562)).cpu().numpy()
            right = (
                detector_rows(
                    other, dense_table(right.float(), latent=desc.float(), min_logit=-3.663562)
                )
                .cpu()
                .numpy()
            )
            parts.extend(merge(a, b, zero, zero, cfg["union"])[:3] for a, b in zip(left, right))
    del primary, head, secondary, other
    values = tuple(np.stack([a[i] for a in parts]) for i in range(3))
    bank = TableBank(values, device)
    head, report = train_head(
        cfg,
        scenes,
        bank,
        Path(work) / "inputs",
        device,
        steps,
        fixed_batch=recipe["batch"] if device.type == "cuda" else 2,
    )
    save(
        Path(work) / f"union{seed}.pt",
        dict(head={k: v.cpu() for k, v in head.state_dict().items()}, config=cfg),
    )
    return report


def counter(
    project,
    data,
    work,
    manifest,
    device,
    config,
    seed,
    batch,
    *,
    limit=None,
    steps=None,
    historical_warmup=False,
):
    from .count_run import make_counter, train
    from .counter import CountQueries
    from .counter_data import CounterData

    cfg = load_config(Path(project) / "configs" / f"{config}.yaml")
    cfg["train"]["seed"] = seed
    scenes = Scenes(data, "train", limit)
    bank = CounterData(Path(work), manifest, scenes, Path(work), device)
    setup(seed)
    model = make_counter(cfg, Path(work) / "inputs", Path(work), manifest, device)
    report = train(
        cfg,
        model,
        bank,
        CountQueries(scenes, device),
        device,
        steps,
        fixed_batch=batch if device.type == "cuda" else 2,
        historical_warmup=historical_warmup,
    )
    role = f"count{53 if config == '053_count' else 54}_{seed}"
    save(
        Path(work) / f"{role}.pt",
        dict(
            config=cfg,
            source_model=model.net.settings,
            model={k: v.cpu() for k, v in model.state_dict().items()},
        ),
    )
    return report
