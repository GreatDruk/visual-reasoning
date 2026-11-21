"""Full-resolution detector training and matched occlusion evaluation."""

import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from kaggle_suite_common import (
    CACHE, FIELDS, INPUT, N_ATTRS, REPORT, WEIGHTS, amp, caption_loss, configure_split,
    dense_bank, dense_probabilities, fixed_table, joint, load, memory,
    record, save, say, setup, tick, tune_batch, write_json,
)
from learned_center_fusion import Fusion
from native_centers import CenterDetector, GRID, GROUPS, HEIGHT, STRIDE, WIDTH, loss_for, mask_for, place_behind
from resnet_centers_ablation import ResNetCenters

DETECTORS = {
    "r50_img": (21, .001, True, False),
    "r50_rand": (21, .001, True, False),
    "occ_clean": (14, .00065, False, False),
    "occ_paste": (14, .00065, False, True),
    "thaw": (10, .00018, False, False),
}
PHYSICAL_BATCH = 128


def make_detector(name):
    if name.startswith("r50"):
        return ResNetCenters(INPUT / "resnet50-11ad3fa6.pth" if name == "r50_img" else None, full=True)
    model = CenterDetector()
    model.load_state_dict(load(INPUT / "native_natural_epoch21.pt"))
    return model


def projection():
    gen = torch.Generator().manual_seed(217)
    return torch.linalg.qr(torch.randn(40, 3 + N_ATTRS, generator=gen))[0]


def native_forward(model, images, matrix=None):
    if matrix is None:
        with amp(images.device):
            raw = model(images)
        return dense_bank(raw.float())
    with amp(images.device):
        latent = model.features(images)
        raw = model.head(latent)
        desc = torch.einsum("bdhw,ds->bshw", latent, matrix)
    return dense_bank(raw.float(), desc.float())


@torch.inference_mode()
def infer_table(model, split, matrix=None, task="detector"):
    device = split.data.device
    model.eval()
    if matrix is not None:
        matrix = matrix.to(device)
    indices = np.arange(len(split.names))
    start = time.monotonic()

    def build(part):
        return native_forward(model, split.images(part, device), matrix)

    batch = tune_batch(task, (192, 128, 64, 24), lambda n: build(indices[np.arange(n) % len(indices)]), device)
    parts = []
    for first in range(0, len(indices), batch):
        parts.append(tuple(x.cpu().numpy() for x in build(indices[first:first + batch])))
    tick(task, len(indices), len(indices), start)
    return tuple(np.concatenate(x) for x in zip(*parts))


def pseudo_examples(split, device):
    (p, _, valid, *prob), features, _ = fixed_table(split.name, "sil", device)
    result, donors = [], []
    for i in range(len(features)):
        keep = valid[i] & (p[i] > .84)
        attrs = [t[i, keep] for t in prob]
        labels = np.column_stack([g.argmax(-1) for g in attrs])
        for j, t in enumerate(attrs):
            labels[t.max(1) < .68, j] = -100
        descriptors = features[i, keep]
        boxes = np.rint(descriptors[:, 8:12] * [WIDTH, HEIGHT, WIDTH, HEIGHT]).astype(int)
        masks = [mask_for(f, box) for f, box in zip(descriptors, boxes)]
        for (x, y, w, h), mask, label, f in zip(boxes, masks, labels, descriptors):
            if min(w, h) >= 10 and max(w, h) <= 60 and f[14] > .83 and label[-1] >= 0:
                donors.append((split.rgb[i, y:y + h, x:x + w].copy(), mask, label))
        result.append((boxes, masks, labels))
    return result, donors


def dense_labels(box_targets, device):
    centers = np.zeros((len(box_targets), *GRID), np.float32)
    regression = np.zeros((len(box_targets), 4, *GRID), np.float32)
    labels = np.full((len(box_targets), 4, *GRID), -100, np.int64)
    for i, (boxes, attrs) in enumerate(box_targets):
        for (x, y, w, h), attr in zip(boxes, attrs):
            u, v = (x + .5 * w) / STRIDE, (y + .5 * h) / STRIDE
            xi, yi = int(u), int(v)
            if 0 <= xi < GRID[1] and 0 <= yi < GRID[0]:
                centers[i, yi, xi] = 1
                regression[i, :, yi, xi] = u - xi, v - yi, np.log(w / 24), np.log(h / 24)
                labels[i, :, yi, xi] = attr
    return tuple(torch.from_numpy(t).to(device) for t in (centers, regression, labels))


def augmented(split, indices, pseudo, donors, rng, paste, device):
    targets = []
    images = []
    for index in indices:
        rgb = split.rgb[index]
        boxes, masks, attrs = pseudo[index]
        if paste and len(boxes) and rng.random() < .75:
            rgb, boxes, attrs = place_behind(rgb, boxes, masks, attrs, donors, rng)
        # Only the detector branch sees flipped images and their flipped boxes.
        if rng.random() < .5:
            rgb = rgb[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0] = WIDTH - boxes[:, 0] - boxes[:, 2]
        images.append(rgb)
        targets.append((boxes, attrs))
    pixels = torch.from_numpy(np.stack(images)).to(device).permute(0, 3, 1, 2).float() / 255
    return pixels.contiguous(memory_format=torch.channels_last), dense_labels(targets, device)


def train_detector(name, probe_only=False, *, warm_start=False, epochs_override=None, rate_scale=1., batch_override=None):
    epochs, rate, resnet, paste = DETECTORS[name]
    epochs = epochs if epochs_override is None else epochs_override
    rate *= rate_scale
    rng, device = setup(812 if resnet else 42)
    cv = configure_split("train", device, fitting=True)
    pseudo, donors = pseudo_examples(cv, device)
    teacher = load(INPUT / "scene_silhouette.pt")
    teacher["relation_offset"].zero_()
    model = (load_detector(name, device) if warm_start else
             make_detector(name).to(device, memory_format=torch.channels_last))
    head = Fusion(teacher).to(device)
    matrix = projection().to(device) if name == "thaw" else None
    if warm_start:
        head.load_state_dict(load(WEIGHTS / f"{name}_head.pt"))
    elif name == "thaw":
        head.load_state_dict(load(WEIGHTS / "head_a.pt"))
    parallel = None
    if name == "thaw" and torch.cuda.device_count() == 2:
        class DetectorAndFeatures(torch.nn.Module):
            def __init__(self, backbone, transform):
                super().__init__()
                self.backbone = backbone
                self.register_buffer("transform", transform)

            def forward(self, image):
                latent = self.backbone.features(image)
                return self.backbone.head(latent), torch.einsum("bdhw,ds->bshw", latent, self.transform)

        parallel = torch.nn.DataParallel(DetectorAndFeatures(model, matrix), device_ids=[0, 1])
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW([{"params": model.parameters()}, {"params": head.parameters(), "lr": .0015*rate_scale}], lr=rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=rate * .08)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    def objective(indices, packed, allow_paste):
        image = cv.images(indices, device)
        if parallel is None:
            bank = native_forward(model, image, matrix)
        else:
            with amp(device):
                raw, descriptor = parallel(image)
            bank = dense_bank(raw.float(), descriptor.float())
        weak = caption_loss(head, bank, cv, packed)
        if name == "thaw":
            return weak
        keep = np.array([len(pseudo[i][0]) >= 2 for i in indices])
        if not keep.any():
            return weak
        extra, targets = augmented(cv, indices[keep], pseudo, donors, rng, allow_paste, device)
        with amp(device):
            raw = model(extra)
        return weak + (.65 if resnet else .85) * loss_for(raw.float(), *targets)

    start = time.monotonic()
    # Matched runs accumulate to the same effective batch even when microbatches differ.
    def probe(n):
        indices, packed = next(cv.batches(n))
        model.train(name != "thaw")
        head.train()
        opt.zero_grad(set_to_none=True)
        try:
            objective(indices, packed, paste).backward()
        finally:
            opt.zero_grad(set_to_none=True)

    if probe_only:
        batch = tune_batch(name, (128, 64, 32, 16), probe, device)
        write_json(REPORT / f"{name}_calibration.json", {"microbatch": batch})
        return
    if batch_override is not None:
        batch = batch_override
    elif name == "thaw":
        batch = tune_batch(name, (128, 64, 32, 16), probe, device)
    else:
        other = {"r50_img": "r50_rand", "r50_rand": "r50_img",
                 "occ_clean": "occ_paste", "occ_paste": "occ_clean"}[name]
        batch = min(json.loads((REPORT / f"{key}_calibration.json").read_text())["microbatch"]
                    for key in (name, other))
    rng = np.random.default_rng(812 if resnet else 42)
    write_json(REPORT / f"{name}_setup.json", {"epochs": epochs, "scenes": len(cv.names),
                                               "microbatch": batch, "effective_batch": PHYSICAL_BATCH})
    history = []
    total_steps = (len(cv.names) + batch - 1) // batch
    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        if name == "thaw":
            # Stabilize running statistics while gradients reach every convolution.
            model.apply(lambda m: m.eval() if isinstance(m, torch.nn.BatchNorm2d) else None)
        opt.zero_grad(set_to_none=True)
        aggregate = torch.zeros((), device=device)
        for step, (indices, packed) in enumerate(cv.batches(batch, rng), 1):
            loss = objective(indices, packed, paste and epoch > 1)
            begin = ((step - 1) * batch // PHYSICAL_BATCH) * PHYSICAL_BATCH
            budget = min(PHYSICAL_BATCH, len(cv.names) - begin)
            scaler.scale(loss * (len(indices) / budget)).backward()
            if step * batch % PHYSICAL_BATCH == 0 or step == total_steps:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 5.)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            aggregate += loss.detach()
        schedule.step()
        history.append(float(aggregate.cpu()) / total_steps)
        if epoch in (1, epochs) or epoch % 4 == 0:
            tick(name, epoch, epochs, start)
            say(name, f"loss {history[-1]:.5f} | {memory(device)}")
    save(model, name)
    save(head, f"{name}_head")
    write_json(REPORT / f"{name}_train.json", {"loss": history, "minutes": (time.monotonic()-start)/60})


def load_detector(name, device):
    if name.startswith("r50"):
        model = ResNetCenters(full=True)
    else:
        model = CenterDetector()
    path = INPUT / "native_natural_epoch21.pt" if name == "native" else WEIGHTS / f"{name}.pt"
    model.load_state_dict(load(path))
    return model.to(device, memory_format=torch.channels_last).eval()


def detector_eval(split_name, name):
    _, device = setup(42)
    split = configure_split(split_name, device)
    model = load_detector(name, device)
    matrix = projection() if name in ("native", "thaw") else None
    table = infer_table(model, split, matrix, f"{name}/{split_name}")
    np.savez(CACHE / f"{name}_{split_name}.npz", x=table[0], xy=table[1], valid=table[2])
    if split_name == "train":
        return
    if name == "native":
        record(name, split, joint(split, dense_probabilities(table)))
        return
    head = Fusion(load(INPUT / "scene_silhouette.pt")).to(device)
    head.load_state_dict(load(WEIGHTS / f"{name}_head.pt"))
    from kaggle_suite_common import table_predictions
    offset = head.relation_offset.detach().cpu().numpy()
    scores = joint(split, table_predictions(head, table, device), offset)
    if name == "r50_img":
        # Test-time views: detect on a mirrored and/or shifted image, map boxes back to the
        # original frame and average caption scores over all views (plain view included).
        views = [scores]
        for dx, dy, flip in TTA_VIEWS:
            seen = infer_table(model, ViewSplit(split, dx, dy, flip), None, f"{name}/{split_name}/{dx}{dy}{int(flip)}")
            views.append(joint(split, table_predictions(head, view_table(seen, dx, dy, flip), device), offset))
        scores = scores + TTA_WEIGHT * (np.mean(views, axis=0) - scores)
    record(name, split, scores)


# (dx, dy, mirrored) in pixels. A 2 px shift is half of the stride-4 detector grid.
# Val, r50_img fit on train: 1 view AUC 0.998744, + flip 0.999220, all four 0.999529.
TTA_VIEWS = ((0, 0, True), (2, 2, False), (2, 2, True))
# Only part of the view average is kept. Inside the downstream occ branch
# (.5*(r50+sigft)+.25*rescue) on val, the full average lowers AUC 0.999732 -> 0.999670,
# while plain + 0.25*(average - plain) keeps 0.999733 with 14 wrong pairs instead of 16.
TTA_WEIGHT = .25


class ViewSplit:
    """Split whose images are optionally mirrored, then shifted right/down by (dx, dy) pixels."""

    def __init__(self, split, dx=0, dy=0, flip=False):
        self._split, self.dx, self.dy, self.flip = split, dx, dy, flip

    def __getattr__(self, attr):
        return getattr(self._split, attr)

    def images(self, indices, device):
        x = self._split.images(indices, device)
        if self.flip:
            x = x.flip(-1)
        if self.dx or self.dy:
            x = F.pad(x, (self.dx, 0, self.dy, 0), mode="replicate")[..., :HEIGHT, :WIDTH]
        return x.contiguous(memory_format=torch.channels_last)


def view_table(table, dx, dy, flip):
    """Undo the shift, then the mirror, of a table detected on a ViewSplit image."""
    x, xy, valid = (a.copy() for a in table)
    box = 1 + sum(len(g) for g in GROUPS)
    x[..., box] -= dx / WIDTH
    x[..., box + 1] -= dy / HEIGHT
    xy[..., 0] -= 2 * dx / WIDTH
    xy[..., 1] -= 2 * dy / HEIGHT
    return mirror_table((x, xy, valid)) if flip else (x, xy, valid)


def mirror_table(table):
    """Map a table detected on mirrored images back to original coordinates."""
    x, xy, valid = (a.copy() for a in table)
    box = 1 + sum(len(g) for g in GROUPS)  # columns: x0, y0, w, h normalised by image size
    x[..., box] = 1 - x[..., box] - x[..., box + 2]
    xy[..., 0] = -xy[..., 0]
    return x, xy, valid


def pasted_recall():
    rng, device = setup(1534)
    val = configure_split("val", device)
    train = configure_split("train", device)
    _, donors = pseudo_examples(train, device)
    pseudo, _ = pseudo_examples(val, device)
    images, target = [], []
    for i, (boxes, masks, labels) in enumerate(pseudo):
        if not len(boxes):
            continue
        base = val.rgb[i]
        decorated, augmented_boxes, augmented_attr = place_behind(base, boxes, masks, labels, donors, rng)
        if len(augmented_boxes) == len(boxes):
            continue
        images.append((base, decorated))
        box = augmented_boxes[-1]
        target.append((box, augmented_attr[-1, -1]))
    shapes = np.array([label for _, label in target])
    boxes = np.stack([box for box, _ in target])
    reports = {}
    for name in ("native", "occ_clean", "occ_paste"):
        model = load_detector(name, device)
        entries = []
        with torch.inference_mode():
            for start in range(0, len(images), 48):
                part = np.stack(images[start:start+48])
                rgb = torch.from_numpy(part.reshape(-1, HEIGHT, WIDTH, 3)).to(device).permute(0, 3, 1, 2).float() / 255
                with amp(device):
                    raw = model(rgb.contiguous(memory_format=torch.channels_last)).float()
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(GRID[0], device=device), torch.arange(GRID[1], device=device), indexing="ij",
                )
                xy = (raw[:, 1:3].clamp(-.5, 1.5) + torch.stack((grid_x, grid_y))) * STRIDE
                box = torch.as_tensor(boxes[start:start+len(part)], device=device, dtype=torch.float32)
                box = box.repeat_interleave(2, 0)[..., None, None]
                center = box[:, :2] + .5 * box[:, 2:]
                near = (xy - center).square().sum(1) <= 8 ** 2
                extent = raw[:, 3:5].clamp(-2, 1.6).exp() * 24
                low = torch.maximum(xy - .5 * extent, box[:, :2])
                high = torch.minimum(xy + .5 * extent, box[:, :2] + box[:, 2:])
                inter = (high - low).clamp_min(0).prod(1)
                union = extent.prod(1) + box[:, 2:].prod(1) - inter
                overlap = inter >= .5 * union
                p = raw[:, 0].sigmoid()
                peak = p == F.max_pool2d(p[:, None], 3, 1, 1)[:, 0]
                kind = torch.from_numpy(shapes[start:start+len(part)]).to(device).repeat_interleave(2)
                same_shape = raw[:, -3:].argmax(1) == kind[:, None, None]
                entries.append(torch.stack((
                    p.masked_fill(~(near & peak), 0).flatten(1).amax(1),
                    p.masked_fill(~(near & peak & same_shape), 0).flatten(1).amax(1),
                    p.masked_fill(~(overlap & peak), 0).flatten(1).amax(1),
                    p.masked_fill(~(overlap & peak & same_shape), 0).flatten(1).amax(1),
                ), -1).reshape(-1, 2, 4).cpu().numpy())
        results = np.concatenate(entries)
        report = {}
        for j, key in enumerate(("center", "center_shape", "iou50", "iou50_shape")):
            control, positive = results[:, :, j].T
            for rate in (.01, .05, .10):
                limit = int(rate * len(control))
                threshold = max(float(np.nextafter(np.sort(control.astype("d"))[::-1][limit], np.inf)), 1e-6)
                control, positive = control.astype("d"), positive.astype("d")
                report[f"{key}_{rate}"] = {"hits": int((positive >= threshold).sum()),
                                          "false_hits": int((control >= threshold).sum()),
                                          "threshold": float(threshold)}
        reports[name] = report
    write_json(REPORT / "overlap_recall.json", {"paired_scenes": len(images), "detectors": reports,
               "reference": "Known full boxes and centers of pasted objects, contour-derived donor shapes. "
                            "False candidates are measured at the same location on the original val image."})
    say("overlap", f"{len(images)} paired validation backgrounds")
