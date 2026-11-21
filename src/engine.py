"""Shared inference and train-only optimization for the visual foundation."""

import time

import numpy as np
import torch

from .detectors import Detector, dense_table, tables
from .heads import FEATURES, GROUPS, N_ATTRS, TableHead, load_head, probabilities
from .losses import caption_loss, detection_loss, map_count_loss
from .native import CenterDetector
from .posterior import sample_scene_scores
from .runtime import Progress, amp, gradient_step, load_state, profile_batches, zero_loss


def cpu_table(parts):
    return tuple(x.detach().cpu().numpy() for x in parts)


def device_table(parts, device):
    return tuple(torch.as_tensor(x, device=device) for x in parts)


@torch.inference_mode()
def proposal_tables(scenes, inputs, settings, source, device, batch=128):
    if source == "resnet":
        control = dict(settings, backbone="resnet", neck="sum", separate=False, width=64)
        model = Detector(control)
        model.load_state_dict(load_state(inputs / "reference/r50_img.pt"))
        matrix = None
    else:
        model = CenterDetector()
        model.load_state_dict(load_state(inputs / "reference/thaw.pt"))
        gen = torch.Generator().manual_seed(217)
        matrix = torch.linalg.qr(torch.randn(40, 3 + N_ATTRS, generator=gen))[0].to(device)
    model = model.to(device, memory_format=torch.channels_last).eval()
    parts = []
    if device.type == "cpu":
        batch = min(batch, 8)
    progress = Progress(f"{scenes.split}/proposals", len(scenes.names), device)
    for start in range(0, len(scenes.names), batch):
        images = scenes.images(np.arange(start, min(start + batch, len(scenes.names))), device)
        with amp(device):
            if matrix is None:
                outputs = tables(model(images), images.shape[-2:])
            else:
                latent = model.features(images)
                raw = model.head(latent)
                descriptor = torch.einsum("bdhw,ds->bshw", latent, matrix)
                outputs = dense_table(raw.float(), latent=descriptor.float())
        parts.append(cpu_table(outputs))
        progress.update(start + len(images), batch=batch)
    return tuple(np.concatenate(v) for v in zip(*parts))


@torch.inference_mode()
def predict(scenes, head, bank, device, draws=1024):
    head.eval()
    parts = []
    for start in range(0, len(bank[0]), 256):
        part = device_table(tuple(a[start : start + 256] for a in bank), device)
        parts.append(cpu_table(probabilities(head, part)))
    p, xy, valid, *attrs = (np.concatenate(v) for v in zip(*parts))
    score = np.zeros(len(scenes.frame), np.float64)
    offsets = head.relation_offset.detach().cpu().numpy()
    progress = Progress(f"{scenes.split}/posterior", len(scenes.names), device)
    for i, rows in enumerate(scenes.rows):
        local = scenes._pairs_local[i]
        keep = valid[i]
        score[rows] = sample_scene_scores(
            [scenes.programs[j] for j in rows],
            local,
            p[i, keep],
            [(g, a[i, keep]) for g, a in zip(GROUPS, attrs)],
            xy[i, keep],
            offsets,
            42,
            draws,
        )
        progress.update(i + 1)
    return score


def initial_head(inputs, features=FEATURES, source=None):
    teacher = load_state(inputs / "reference/scene_silhouette.pt")
    teacher["relation_offset"].zero_()
    head = TableHead(features, teacher)
    if source:
        load_head(head, load_state(inputs / "reference" / source))
    return head


def photometric(images, rng, amount):
    if not amount:
        return images
    brightness = torch.as_tensor(
        rng.uniform(1 - amount, 1 + amount, (len(images), 1, 1, 1)),
        device=images.device,
        dtype=images.dtype,
    )
    contrast = torch.as_tensor(
        rng.uniform(1 - amount, 1 + amount, (len(images), 1, 1, 1)),
        device=images.device,
        dtype=images.dtype,
    )
    mean = images.mean((1, 2, 3), keepdim=True)
    return ((images - mean) * contrast + mean).mul(brightness).clamp(0, 1)


def resnet_loss(cfg, head, outputs, det_outputs, packed, targets, stage, shape):
    """Use the same FP32 objective in batch profiling and every curriculum stage."""
    if stage == "objects":
        loss = zero_loss(outputs[0])
    else:
        bank = tables(outputs, shape, cfg["model"]["topk"])
        loss = caption_loss(head, bank, packed, simple=stage == "simple")
    loss = loss + cfg["train"]["detection_weight"] * detection_loss(
        det_outputs, soft=cfg["train"]["soft_background"], targets=targets
    )
    if cfg["train"]["count_weight"] and stage != "objects":
        loss = loss + cfg["train"]["count_weight"] * map_count_loss(outputs[0], packed)
    return loss


def train_resnet(cfg, scenes, teacher, inputs, device, rng, step_limit=None):
    weight_name = "resnet50" if cfg["model"]["backbone"] == "resnet" else "convnext"
    model = Detector(cfg["model"], inputs / "pretrained" / f"{weight_name}.pth").to(
        device, memory_format=torch.channels_last
    )
    head = initial_head(inputs).to(device)
    params = [*model.parameters(), *head.parameters()]
    opt = torch.optim.AdamW(
        [
            dict(params=model.parameters(), lr=cfg["train"]["rate"]),
            dict(params=head.parameters(), lr=cfg["train"]["head_rate"]),
        ]
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, cfg["train"]["epochs"], eta_min=cfg["train"]["rate"] * 0.08
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=128.0)
    teacher.prepare_targets(
        device,
        3 if cfg["model"]["neck"] == "multi" else 1,
        cfg["train"]["soft_background"],
        weak_only=cfg["train"].get("soft_background_weak_only", False),
    )
    batch = cfg["train"]["batch"] if device.type == "cuda" else min(2, cfg["train"]["batch"])
    batch_report = []
    active_stages = [
        stage
        for stage, epochs in zip(("objects", "simple", "all"), cfg["train"]["stages"])
        if epochs
    ]
    if device.type == "cuda" and cfg["train"].get("calibrate", True):
        saved_buffers = {name: t.clone() for name, t in model.named_buffers()}
        cpu_rng = torch.get_rng_state()
        gpu_rng = torch.cuda.get_rng_state(device)
        model.train()
        head.train()

        def probe(n):
            idx = np.arange(n)
            image = scenes.images(idx, device)
            packed = scenes.batch(idx, device)
            targets = teacher.targets(idx, np.zeros(n, bool))
            for stage in active_stages:
                try:
                    with amp(device):
                        out = model(image)
                        extra = model(image)
                    loss = resnet_loss(
                        cfg, head, out, extra, packed, targets, stage, image.shape[-2:]
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite {stage} loss in batch probe {n}")
                    loss.backward()
                finally:
                    model.zero_grad(set_to_none=True)
                    head.zero_grad(set_to_none=True)

        batch, batch_report = profile_batches(
            device, [n for n in (16, 32, 64, 128) if n <= batch], len(scenes.names), probe
        )
        with torch.no_grad():
            for name, t in model.named_buffers():
                t.copy_(saved_buffers[name])
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(gpu_rng, device)
    effective = (
        cfg["train"]["effective_batch"] if device.type == "cuda" and not step_limit else batch
    )
    if effective % batch:
        raise ValueError("Physical batch must divide the effective batch")
    history = []
    updates = 0
    overflows = 0
    consecutive = 0
    started = time.monotonic()
    for epoch in range(cfg["train"]["epochs"]):
        stage = (
            "objects"
            if epoch < cfg["train"]["stages"][0]
            else "simple"
            if epoch < sum(cfg["train"]["stages"][:2])
            else "all"
        )
        model.train()
        head.train()
        loss_sum = torch.zeros((), device=device)
        seen = 0
        order = rng.permutation(len(scenes.names))
        progress = Progress(f"resnet/epoch{epoch + 1}/{stage}", len(order), device)
        opt.zero_grad(set_to_none=True)
        for begin in range(0, len(order), effective):
            block = order[begin : begin + effective]
            block_loss = torch.zeros((), device=device)
            for start in range(0, len(block), batch):
                idx = block[start : start + batch]
                image = scenes.images(idx, device)
                flip = (
                    rng.random(len(idx)) < 0.5
                    if cfg["train"]["joint_flip"]
                    else np.zeros(len(idx), bool)
                )
                image = torch.where(
                    torch.as_tensor(flip, device=device)[:, None, None, None],
                    image.flip(-1),
                    image,
                )
                image = photometric(image, rng, cfg["train"]["photometric"])
                packed = scenes.batch(idx, device, flip)
                with amp(device):
                    out = model(image)
                # Control: a separate, mirrored pseudo-supervised pass; joint-flip arm
                # uses the same transformed image for caption and object supervision.
                det_flip = flip if cfg["train"]["joint_flip"] else rng.random(len(idx)) < 0.5
                det_image = (
                    image
                    if cfg["train"]["joint_flip"]
                    else torch.where(
                        torch.as_tensor(det_flip, device=device)[:, None, None, None],
                        image.flip(-1),
                        image,
                    )
                )
                # Keep two forwards in both arms, including BN update frequency.
                with amp(device):
                    det_out = model(det_image)
                loss = resnet_loss(
                    cfg,
                    head,
                    out,
                    det_out,
                    packed,
                    teacher.targets(idx, det_flip),
                    stage,
                    image.shape[-2:],
                )
                scaler.scale(loss * len(idx) / len(block)).backward()
                block_loss += loss.detach() * len(idx) / len(block)
                loss_sum += loss.detach() * len(idx)
                seen += len(idx)
            completed, norm = gradient_step(scaler, opt, params, loss=block_loss)
            if not completed:
                overflows += 1
                consecutive += 1
                print(
                    f"AMP overflow: skipped update, scale={scaler.get_scale():g}, consecutive={consecutive}",
                    flush=True,
                )
                if consecutive >= 8:
                    raise FloatingPointError(
                        "Persistent non-finite gradients after 8 AMP scale reductions"
                    )
            else:
                consecutive = 0
            updates += 1
            progress.update(seen, batch=batch, amp_skips=overflows)
            if step_limit and updates >= step_limit:
                break
            if (time.monotonic() - started) / 60 > cfg["train"]["max_minutes"]:
                raise TimeoutError("Configured training time exceeded; run is incomplete")
        sch.step()
        history.append(
            dict(
                epoch=epoch + 1, stage=stage, loss=float(loss_sum.cpu()) / max(seen, 1), scenes=seen
            )
        )
        progress.update(
            seen, force=True, loss=history[-1]["loss"], batch=batch, amp_skips=overflows
        )
        if step_limit and updates >= step_limit:
            break
    return (
        model.eval(),
        head.eval(),
        dict(
            history=history,
            updates=updates,
            training_seconds=time.monotonic() - started,
            scenes_per_second=sum(h["scenes"] for h in history) / (time.monotonic() - started),
            physical_batch=batch,
            batch_profile=batch_report,
            profiled_stages=active_stages if batch_report else [],
            amp_skipped_updates=overflows,
        ),
    )


@torch.inference_mode()
def detector_bank(model, scenes, device, batch=128, topk=20):
    parts = []
    if device.type == "cpu":
        batch = min(batch, 2)
    progress = Progress(f"{scenes.split}/detector", len(scenes.names), device)
    for start in range(0, len(scenes.names), batch):
        idx = np.arange(start, min(start + batch, len(scenes.names)))
        image = scenes.images(idx, device)
        with amp(device):
            out = model(image)
        parts.append(cpu_table(tables(out, image.shape[-2:], topk)))
        progress.update(start + len(idx), batch=batch)
    return tuple(np.concatenate(v) for v in zip(*parts))


def train_head(cfg, scenes, bank, inputs, device, rng, step_limit=None):
    head = initial_head(inputs, bank[0].shape[-1]).to(device)
    # Warm-start only the shared table columns; crop columns initially have zero effect.
    warm = initial_head(inputs, source="thaw_head.pt").state_dict()
    state = head.state_dict()
    for key, value in warm.items():
        if key == "residual.0.weight":
            state[key].zero_()
            state[key][:, :FEATURES].copy_(value)
        else:
            state[key].copy_(value)
    head.load_state_dict(state)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["siglip"]["head_rate"], weight_decay=0.0001)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, cfg["siglip"]["head_epochs"], eta_min=cfg["siglip"]["head_min_rate"]
    )
    steps = 0
    for epoch in range(cfg["siglip"]["head_epochs"]):
        head.train()
        order = rng.permutation(len(scenes.names))
        progress = Progress(f"siglip/head_epoch{epoch + 1}", len(order), device)
        for start in range(0, len(order), 128):
            idx = order[start : start + 128]
            packed = scenes.batch(idx, device)
            loss = caption_loss(head, device_table(tuple(x[idx] for x in bank), device), packed)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0, error_if_nonfinite=True)
            opt.step()
            steps += 1
            progress.update(start + len(idx))
            if step_limit and steps >= step_limit:
                break
        sch.step()
        if step_limit and steps >= step_limit:
            break
    return head.eval()
