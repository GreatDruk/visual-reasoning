"""Train-only crop pooling and scene-supervised visual adaptation."""

import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from sklearn.decomposition import PCA

from .engine import device_table, train_head
from .heads import BOX
from .losses import caption_loss
from .runtime import Progress, amp, gradient_step, profile_batches
from .siglip import Vision, attribute_loss, crop_images


@contextmanager
def token_storage(path, shape):
    """Release the disk-backed cache on success and on failed training alike."""
    data = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    try:
        yield data
    finally:
        data.flush()
        data._mmap.close()
        path.unlink(missing_ok=True)


class Crops:
    def __init__(self, scenes, table, teacher=None):
        self.scenes = scenes
        self.table = table
        self.scene, self.slot = np.nonzero(table[2] & (table[0][..., 0] >= -1.6))
        self.box = table[0][self.scene, self.slot, BOX].copy()
        self.labels = teacher.crop_labels(self.scene, self.box) if teacher is not None else None
        self.by_scene = [np.flatnonzero(self.scene == i) for i in range(len(scenes.names))]
        if not len(self.scene):
            raise ValueError("No usable crop proposals")

    def pixels(self, rows, device, settings):
        unique, inverse = np.unique(self.scene[rows], return_inverse=True)
        images = self.scenes.images(unique, device)[torch.as_tensor(inverse, device=device)]
        boxes = torch.as_tensor(self.box[rows], device=device)
        return crop_images(images, boxes, mode=settings["crop"], context=settings["context"])


def warm_pool(vision, crops, cfg, work, device, rng, step_limit=None):
    settings = cfg["siglip"]
    path = Path(work) / "pool.npy"
    n = len(crops.scene)
    grid = settings["pool_grid"]
    width = vision.backbone.config.hidden_size
    required = n * grid * grid * width * 2
    if shutil.disk_usage(path.parent).free < required + 2 * 2**30:
        raise OSError(f"Pooling cache needs {required / 2**30:.1f} GiB plus 2 GiB of free scratch")
    with token_storage(path, (n, grid * grid, width)) as tokens:
        vision.eval()
        batch = settings.get("max_extract_batch", 512) if device.type == "cuda" else 2
        started = time.monotonic()
        progress = Progress("siglip/train_tokens", n, device)

        def probe(n):
            rows = np.arange(n)
            with torch.no_grad(), amp(device):
                hidden, _ = vision.tokens(crops.pixels(rows, device, settings))
                vision.reduce(hidden)

        batch, batch_report = profile_batches(
            device, [size for size in (16, 32, 64, 128, 256, 512) if size <= batch], n, probe
        )
        vision.extraction_batch = batch
        with torch.no_grad():
            for start in range(0, n, batch):
                rows = np.arange(start, min(start + batch, n))
                with amp(device):
                    hidden, _ = vision.tokens(crops.pixels(rows, device, settings))
                tokens[rows] = vision.reduce(hidden).half().cpu().numpy()
                progress.update(start + len(rows), batch=batch)
        tokens.flush()
        usable = np.flatnonzero((crops.labels >= 0).any(-1))
        if not len(usable):
            raise ValueError("No confident train-only crop labels")
        opt = torch.optim.AdamW(
            [*vision.pool.parameters(), *vision.classifier.parameters()], lr=settings["pool_rate"]
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=128.0)
        steps = 0
        skipped = 0
        consecutive = 0
        parameters = [*vision.pool.parameters(), *vision.classifier.parameters()]
        progress = Progress(
            "siglip/pooling",
            settings["pool_epochs"]
            * ((len(usable) + settings["pool_batch"] - 1) // settings["pool_batch"]),
            device,
        )
        vision.pool.train()
        vision.classifier.train()
        for epoch in range(settings["pool_epochs"]):
            order = rng.permutation(usable)
            for start in range(0, len(order), settings["pool_batch"]):
                rows = order[start : start + settings["pool_batch"]]
                x = torch.from_numpy(tokens[rows].astype(np.float32)).to(device)
                labels = torch.as_tensor(crops.labels[rows], device=device)
                with amp(device):
                    scores = vision.classifier(vision.pool(x))
                loss = attribute_loss(scores, labels)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                completed, _ = gradient_step(
                    scaler, opt, parameters, loss=loss, max_norm=float("inf")
                )
                skipped += not completed
                consecutive = 0 if completed else consecutive + 1
                if consecutive >= 8:
                    raise FloatingPointError("Persistent non-finite pooling gradients")
                steps += 1
                progress.update(steps, amp_skips=skipped)
                if step_limit and steps >= step_limit:
                    break
            if step_limit and steps >= step_limit:
                break
        # The trunk is frozen throughout this phase. Reuse its exact cached FP16
        # tokens instead of running all 27 transformer blocks on train a second time.
        raw = None
        if not settings["intermediate"]:
            raw = np.empty((n, width), np.float32)
            vision.pool.eval()
            progress = Progress("siglip/cached_train_embeddings", n, device)
            with torch.no_grad():
                for start in range(0, n, settings["pool_batch"]):
                    end = min(start + settings["pool_batch"], n)
                    x = torch.from_numpy(tokens[start:end].astype(np.float32)).to(device)
                    with amp(device):
                        value = vision.pool(x)
                    raw[start:end] = value.float().cpu().numpy()
                    progress.update(end)
    return dict(
        warmup_seconds=time.monotonic() - started,
        matched_crops=len(usable),
        warmup_steps=steps,
        peak_token_cache_gib=required / 2**30,
        extraction_batch=batch,
        batch_profile=batch_report,
        amp_skipped_pool_updates=skipped,
        reused_train_tokens=raw is not None,
    ), raw


@torch.inference_mode()
def embeddings(vision, crops, cfg, device):
    parts = []
    batch = (
        getattr(vision, "extraction_batch", cfg["siglip"]["extract_batch"])
        if device.type == "cuda"
        else 2
    )
    vision.eval()
    progress = Progress(f"siglip/{crops.scenes.split}_embeddings", len(crops.scene), device)
    for start in range(0, len(crops.scene), batch):
        rows = np.arange(start, min(start + batch, len(crops.scene)))
        with amp(device):
            value = vision.embedding(crops.pixels(rows, device, cfg["siglip"]))
        parts.append(value.float().cpu().numpy())
        progress.update(start + len(rows), batch=batch)
    return np.concatenate(parts)


class Projection:
    def __init__(self, raw, crops, seed=732):
        rng = np.random.default_rng(seed)
        selected = []
        # Match the historical two crop shards and cap of 8,000 examples per shard.
        for shard in (0, 1):
            rows = np.arange(shard, len(raw), 2)
            selected.append(rows[rng.choice(len(rows), min(8000, len(rows)), replace=False)])
        n = min(128, len(raw) - 1, raw.shape[1])
        if n < 1:
            raise ValueError("PCA requires at least two training crops")
        pca = PCA(n_components=n, whiten=True, svd_solver="randomized", random_state=seed).fit(
            raw[np.concatenate(selected)]
        )
        self.mean = pca.mean_.astype(np.float32)
        self.components = pca.components_.astype(np.float32)
        self.scale = np.sqrt(pca.explained_variance_).clip(1e-5).astype(np.float32)

    def apply(self, x):
        return ((x - self.mean) @ self.components.T / self.scale).clip(-7, 7)

    def tensor(self, x):
        mean = torch.as_tensor(self.mean, device=x.device)
        basis = torch.as_tensor(self.components, device=x.device)
        scale = torch.as_tensor(self.scale, device=x.device)
        return ((x.float() - mean) @ basis.T / scale).clamp(-7, 7)

    def state(self):
        return dict(
            mean=torch.as_tensor(self.mean),
            components=torch.as_tensor(self.components),
            scale=torch.as_tensor(self.scale),
        )


def feature_table(crops, raw, projection):
    x, xy, valid = crops.table
    out = np.zeros((*valid.shape, 128), np.float32)
    value = projection.apply(raw)
    out[crops.scene, crops.slot, : value.shape[-1]] = value
    return np.concatenate((x, out), -1), xy, valid


def adapt(vision, head, crops, projection, cfg, device, rng, step_limit=None):
    settings = cfg["siglip"]
    steps = settings["adapt_steps"]
    if not steps:
        return dict(adaptation_steps=0)
    if step_limit:
        steps = min(steps, step_limit)
    vision.adapting = True
    vision.train()
    head.train()
    params = [p for p in vision.parameters() if p.requires_grad] + list(head.parameters())
    tail_ids = {id(p) for p in vision.backbone.parameters() if p.requires_grad}
    opt = torch.optim.AdamW(
        [
            dict(
                params=[p for p in params if id(p) in tail_ids],
                lr=settings["tail_rate"],
            ),
            dict(
                params=[p for p in params if id(p) not in tail_ids],
                lr=settings["pool_rate"],
            ),
        ],
        weight_decay=0.0001,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=128.0)
    started = time.monotonic()
    last = 0.0
    skipped = consecutive = 0
    progress = Progress("siglip/adaptation", steps, device)
    for step in range(steps):
        indices = rng.choice(
            len(crops.scenes.names),
            min(settings["scene_batch"], len(crops.scenes.names)),
            replace=False,
        )
        rows = np.concatenate([crops.by_scene[i] for i in indices])
        if not len(rows):
            raise ValueError("Adaptation batch has no visual candidates")
        pixel = crops.pixels(rows, device, settings)
        if settings["augment"]:
            flips = torch.as_tensor(rng.random(len(pixel)) < 0.5, device=device)
            pixel = torch.where(flips[:, None, None, None], pixel.flip(-1), pixel)
            brightness = torch.as_tensor(
                rng.uniform(0.96, 1.04, (len(pixel), 1, 1, 1)),
                device=device,
                dtype=pixel.dtype,
            )
            pixel = (pixel * brightness).clamp(0, 1)
        values = []
        for start in range(0, len(pixel), settings["adapt_batch"]):
            with amp(device):
                values.append(vision.embedding(pixel[start : start + settings["adapt_batch"]]))
        encoded = torch.cat(values)
        labels = torch.as_tensor(crops.labels[rows], device=device)
        with amp(device):
            attr = vision.classifier(encoded)
        loss = attribute_loss(attr, labels)
        local = {n: i for i, n in enumerate(indices)}
        scene = torch.as_tensor([local[i] for i in crops.scene[rows]], device=device)
        slot = torch.as_tensor(crops.slot[rows], device=device)
        projected = projection.tensor(encoded)
        if settings["objective"] == "attributes":
            projected = projected.detach()
        extra = projected.new_zeros((len(indices), crops.table[0].shape[1], 128))
        extra[scene, slot, : projected.shape[-1]] = projected
        x, xy, valid = device_table(tuple(x[indices] for x in crops.table), device)
        bank = (torch.cat((x, extra), -1), xy, valid)
        packed = crops.scenes.batch(indices, device)
        loss += caption_loss(head, bank, packed)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        completed, _ = gradient_step(scaler, opt, params, loss=loss)
        if not completed:
            skipped += 1
            consecutive += 1
            print(f"AMP overflow in adaptation: scale={scaler.get_scale():g}", flush=True)
            if consecutive >= 8:
                raise FloatingPointError("Persistent non-finite adaptation gradients")
        else:
            consecutive = 0
        last = float(loss.detach())
        progress.update(step + 1, loss=last, amp_skips=skipped)
    vision.adapting = False
    return dict(
        adaptation_steps=steps,
        adaptation_seconds=time.monotonic() - started,
        last_adaptation_loss=last,
        amp_skipped_adaptation_updates=skipped,
    )


def train_siglip(
    cfg,
    train,
    val,
    teacher,
    train_table,
    val_table,
    inputs,
    work,
    device,
    rng,
    step_limit=None,
    cache_lock=None,
):
    settings = cfg["siglip"]
    vision = Vision(inputs / "pretrained/siglip", settings).float().to(device)
    train_crops = Crops(train, train_table, teacher)
    val_crops = Crops(val, val_table)
    waiting = time.monotonic()
    with FileLock(cache_lock or Path(work) / "pool.lock"):
        waited = time.monotonic() - waiting
        report, raw = warm_pool(vision, train_crops, cfg, work, device, rng, step_limit)
    report["cache_wait_seconds"] = waited
    if raw is None:
        raw = embeddings(vision, train_crops, cfg, device)
    progress = Progress("siglip/PCA_and_scene_head", 1, device)
    projection = Projection(raw, train_crops)
    train_bank = feature_table(train_crops, raw, projection)
    head = train_head(cfg, train, train_bank, inputs, device, rng, step_limit)
    progress.update(1, force=True)
    report.update(adapt(vision, head, train_crops, projection, cfg, device, rng, step_limit))
    val_raw = embeddings(vision, val_crops, cfg, device)
    return (
        vision,
        head,
        feature_table(val_crops, val_raw, projection),
        projection,
        report,
    )
