"""Train a foreground-conditioned detector for the visible fragment of a rear object."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from scripts.research import layer_repair as layer
from scripts.research.fragment_votes import EXTRA, H, W, Scenes, images, v
from kaggle_suite_common import amp

ARMS = ("clean", "paste", "matched")
SIDE = 64
OUT = Path("fragment_detector")


def view(front):
    axis = (torch.arange(SIDE, device=front.device) + .5) / SIDE
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    width = front[:, 2:].amax(1).clamp_min(12) * 2.4
    origin = front[:, :2] + front[:, 2:] * front.new_tensor([.5, .26]) - width[:, None] / 2
    points = origin[:, None, None] + torch.stack([x, y], -1) * width[:, None, None, None]
    return points, origin, width


def sample(field, grid, mode="bilinear"):
    return F.grid_sample(field.float(), grid, mode=mode, align_corners=False, padding_mode="zeros")


def inputs(rgb, front):
    points, origin, width = view(front)
    grid = points * points.new_tensor([2 / W, 2 / H]) - 1
    pos = (points - front[:, None, None, :2]) / front[:, None, None, 2:].clamp_min(1)
    prompt = ((pos >= 0) & (pos <= 1)).all(-1)
    crop = F.grid_sample(rgb.float(), grid, align_corners=False, padding_mode="border") / 255
    return torch.cat([crop, prompt[:, None]], 1), points, grid, origin, width


class Bank:
    def __init__(self, device, limit=None):
        started = time.perf_counter()
        with np.load(EXTRA / "fragment_train.npz") as z:
            names = z["names"]
            keep = z["keep"]
        official = pd.read_csv(v.DATA / "train.csv").image.unique()
        if not np.array_equal(names, official):
            raise ValueError("The donor bank must come exclusively from the supplied train split")
        rgb = images("train", names)
        decoded = time.perf_counter()
        self.scene = Scenes(device, rgb)
        self.device = device
        self.idx = torch.as_tensor(np.flatnonzero(keep.any(1))[:limit], device=device)
        records = self.scene.donors
        colors = self.scene.labels[records[:, 0], records[:, 1], 1]
        by_color = [torch.where(colors == k)[0] for k in range(8)]
        self.color_counts = torch.tensor([len(k) for k in by_color], device=device)
        self.colors = torch.stack([F.pad(k, (0, max(map(len, by_color)) - len(k))) for k in by_color])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.load_report = dict(jpeg_decode_s=decoded-started, mask_rasterize_upload_s=time.perf_counter()-decoded,
                                scenes=len(self.idx), donor_instances=len(records))
        print(json.dumps(self.load_report), flush=True)

    def natural(self, idx, rng):
        s = self.scene
        available = s.anchors[idx]
        front_id = (torch.rand(available.shape, device=self.device, generator=rng) * available).argmax(1)
        front = s.box[idx, front_id]
        x, points, grid, origin, width = inputs(s.rgb[idx], front)
        instance = sample(s.instances[idx, None], grid, "nearest")[:, 0].long()
        boxes = s.box[idx]
        bottom = boxes[..., 1] + boxes[..., 3]
        front_bottom = front[:, 1] + front[:, 3]
        lo = torch.maximum(boxes[..., :2], front[:, None, :2] - 2)
        hi = torch.minimum(boxes[..., :2] + boxes[..., 2:], (front[:, :2] + front[:, 2:])[:, None] + 2)
        overlap = (hi - lo).clamp_min(0).prod(-1) / boxes[..., 2:].prod(-1).clamp_min(1)
        eligible = available & (bottom < front_bottom[:, None] - 3) & (overlap >= .10)
        rear_id = (overlap * eligible).argmax(1)
        visible = (instance == rear_id[:, None, None] + 1) & eligible.any(1)[:, None, None]
        has_rear = visible.sum((1, 2)) >= 4
        bundle = (x, visible & has_rear[:, None, None], s.box[idx, rear_id], s.labels[idx, rear_id],
                  origin, width, has_rear, torch.zeros_like(has_rear, dtype=torch.float32),
                  instance == front_id[:, None, None] + 1, x.new_zeros((len(idx),4)))
        return bundle, (front_id, front, points, instance)

    def paste(self, idx, natural, geometry, arm, rng):
        s = self.scene
        x, old_mask, old_box, old_label, origin, width, old_present, _, front_mask, _ = natural
        front_id, front, points, instance = geometry
        n = len(idx)

        def rand(*shape):
            return torch.rand(shape, device=self.device, generator=rng)

        picked = (rand(n) * len(s.donors)).long()
        front_color = s.labels[idx, front_id, 1]
        color = front_color.clamp_min(0)
        same = self.colors[color, (rand(n) * self.color_counts[color]).long()]
        if arm == "matched":
            picked = torch.where((rand(n) < .75) & (front_color >= 0) & (self.color_counts[color] > 0), same, picked)
        donor_img, donor_id = s.donors[picked].T
        donor = s.box[donor_img, donor_id]
        size = donor[:, 2:] * (.8 + .4 * rand(n, 1))
        if arm == "matched":
            # Match camera scale gently and move the rear object's support higher in the frame.
            scale = ((front[:, 1] + front[:, 3] + 80) / (donor[:, 1] + donor[:, 3] + 80)).clamp(.76, 1.24)
            size *= scale[:, None]
        center = front[:, :2] + front[:, 2:] / 2
        offset = (rand(n, 2) - .5) * torch.minimum(front[:, 2:], size) * 1.35
        offset[:, 1] -= .21 * front[:, 3]
        start = center + offset - size / 2
        uv = (points - start[:, None, None]) / size[:, None, None]
        alpha = sample(s.masks[donor_img, donor_id, None], 2 * uv - 1)[:, 0] / 255
        on_screen = ((points >= 0) & (points < points.new_tensor([W, H]))).all(-1)
        visible = alpha * (instance == 0) * on_screen
        total = alpha.sum((1, 2)).clamp_min(1)
        ratio = visible.sum((1, 2)) / total
        covered_by_front = (alpha * front_mask).sum((1, 2)) / total
        support = (visible > .5) & (alpha > .5)
        pixels = support.sum((1, 2)) * (width / SIDE).square()
        lowest = start[:, 1] + size[:, 1]
        valid = ((ratio >= .04) & (ratio <= (.36 if arm == "matched" else .70))
                 & (covered_by_front >= .78 * (1 - ratio)) & (pixels >= 4)
                 & (start >= 0).all(1) & ((start + size) <= points.new_tensor([W, H])).all(1)
                 & (lowest < front[:, 1] + front[:, 3] - 2) & (donor_img != idx) & ~old_present)
        donor_grid = (donor[:, None, None, :2] + uv * donor[:, None, None, 2:])
        donor_grid = donor_grid * points.new_tensor([2 / W, 2 / H]) - 1
        patch = sample(s.rgb[donor_img], donor_grid) / 255
        opacity = visible * valid[:, None, None]
        rgb = x[:, :3] + (patch - x[:, :3]) * opacity[:, None]
        return (torch.cat([rgb, x[:, 3:]], 1), torch.where(valid[:, None, None], support, old_mask),
                torch.where(valid[:, None], torch.cat([start, size], 1), old_box),
                torch.where(valid[:, None], s.labels[donor_img, donor_id], old_label),
                origin, width, valid | old_present, valid.float(), front_mask,
                torch.stack([valid & (ratio < .15), valid & (ratio >= .15) & (ratio <= .36),
                             valid & (ratio > .36), valid & (front_color == s.labels[donor_img, donor_id, 1])],1).float())

    def batch(self, idx, arm, natural_rng, paste_rng):
        natural, geometry = self.natural(idx, natural_rng)
        other = natural if arm == "clean" else self.paste(idx, natural, geometry, arm, paste_rng)
        return tuple(torch.cat([a, b]) for a, b in zip(natural, other))


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU())


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.full = block(4, 24)
        self.half_scale = block(24, 48, 2)
        self.context = nn.Sequential(block(48, 64, 2), block(64, 96, 2), block(96, 96))
        self.mid = nn.Conv2d(48, 24, 1)
        self.coarse = nn.Conv2d(96, 24, 1)
        self.fragments = nn.Conv2d(72, 2, 1)  # visible rear and front edge/mask
        self.readout = nn.Sequential(nn.Linear(96 * 4 + 72 + 3, 160), nn.SiLU(), nn.Dropout(.10), nn.Linear(160, 21))

    def forward(self, x):
        full = self.full(x)
        half = self.half_scale(full)
        global_view = self.context(half)
        pixels = torch.cat([full, F.interpolate(self.mid(half), size=full.shape[-2:], mode="bilinear", align_corners=False),
                            F.interpolate(self.coarse(global_view), size=full.shape[-2:], mode="bilinear", align_corners=False)], 1)
        mask = self.fragments(pixels).float()
        p = mask[:, :1].sigmoid() * (1 - mask[:, 1:].sigmoid())
        weight = p.square()
        position = torch.linspace(.5/SIDE, 1-.5/SIDE, SIDE, device=x.device)
        mass = weight.sum((2, 3)).clamp_min(1e-6)
        evidence = torch.cat([(pixels * weight).sum((2, 3)) / mass,
                              (weight * position).sum((2, 3)) / mass,
                              (weight * position[:, None]).sum((2, 3)) / mass,
                              mass.log1p()], 1)
        return self.readout(torch.cat([F.adaptive_avg_pool2d(global_view, 2).flatten(1), evidence], 1)).float(), mask


def objective(net, batch):
    x, visible, boxes, labels, origin, width, positive, _, front, _ = batch
    raw, mask = net(x.contiguous(memory_format=torch.channels_last))
    p = mask[:, 0].sigmoid()
    t = visible.float()
    positives = t.sum((1, 2)).clamp_min(1)
    focal = F.binary_cross_entropy_with_logits(mask[:, 0], t, reduction="none") * (p-t).square()
    pixel = (focal * t).sum((1, 2)) / positives + (focal * (1-t)).mean((1, 2))
    presence = F.binary_cross_entropy_with_logits(raw[:, 0], positive.float())
    cls = 0
    for pred, target in zip(raw[:, 1:17].split((2, 8, 6), 1), labels.T):
        use = positive & (target >= 0)
        # Ignore unavailable attributes without shrinking the detector's batch.
        cel = F.cross_entropy(pred, target.clamp_min(0), reduction="none")
        cls = cls + (cel * use).sum() / use.sum().clamp_min(1)
    encoded = torch.cat([(boxes[:, :2] + boxes[:, 2:] / 2 - origin) / width[:, None],
                         boxes[:, 2:] / width[:, None]], 1)
    box_loss = F.smooth_l1_loss(raw[:, 17:].sigmoid(), encoded.clamp(0, 1), reduction="none", beta=.10).sum(1)
    box_loss = (box_loss * positive).sum() / positive.sum().clamp_min(1)
    return presence + pixel.mean() + .25 * cls + 2 * box_loss + .15 * F.binary_cross_entropy_with_logits(mask[:, 1], front.float())



def seed_rng(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


def gpu_util(device):
    if device.type == "cuda":
        try:
            return torch.cuda.utilization(device)
        except (AttributeError, RuntimeError, OSError, ImportError):
            pass
    return None


def benchmark(bank, arm, candidates, seed):
    device = bank.device
    results = []
    for num in candidates:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        row = dict(batch=num * 2)
        try:
            net = Detector().to(device, memory_format=torch.channels_last).train()
            opt = torch.optim.AdamW(net.parameters(), lr=.001, fused=device.type == "cuda")
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
            natural_rng, paste_rng = seed_rng(device, seed + 1), seed_rng(device, seed + 2)
            idx = bank.idx[torch.arange(num, device=device) % len(bank.idx)]
            prep_ms, enqueue_ms, train_ms, elapsed = [], [], [], []
            started = time.perf_counter()
            for step in range(30 if device.type == "cuda" else 2):
                t = time.perf_counter()
                if device.type == "cuda":
                    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                    events[0].record()
                batch = bank.batch(idx, arm, natural_rng, paste_rng)
                prepared = time.perf_counter()
                if device.type == "cuda":
                    events[1].record()
                opt.zero_grad(set_to_none=True)
                with amp(device):
                    loss = objective(net, batch)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                if device.type == "cuda":
                    events[2].record()
                    events[2].synchronize()
                if step > 0:
                    prep_ms.append(events[0].elapsed_time(events[1]) if device.type == "cuda" else 1000*(prepared-t))
                    train_ms.append(events[1].elapsed_time(events[2]) if device.type == "cuda" else 1000*(time.perf_counter()-prepared))
                    elapsed.append(time.perf_counter()-t)
                    enqueue_ms.append(1000*(prepared-t))
                if step >= 5 and time.perf_counter()-started > 1.5:
                    break
            memory = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
            total = torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 1
            wall = sum(elapsed)
            row.update(crops_per_s=2*num*len(elapsed)/wall, aug_ms=float(np.mean(prep_ms)), aug_host_enqueue_ms=float(np.mean(enqueue_ms)),
                       forward_backward_step_ms=float(np.mean(train_ms)), wall_ms=1000*wall/len(elapsed),
                       peak_reserved_gib=memory/2**30,
                       peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type == "cuda" else 0,
                       vram_fraction=memory/total, sampled_gpu_util_pct=gpu_util(device))
        except torch.cuda.OutOfMemoryError:
            row["oom"] = True
        finally:
            # Larger probes may fail before all variables have been assigned.
            net = opt = scaler = batch = loss = None
        results.append(row)
        print("batch probe", json.dumps(row), flush=True)
    viable = [r for r in results if not r.get("oom") and r["vram_fraction"] <= .91]
    if not viable:
        raise RuntimeError("No batch leaves VRAM headroom")
    fastest = max(r["crops_per_s"] for r in viable)
    best = max((r for r in viable if r["crops_per_s"] >= .98*fastest), key=lambda r:r["batch"])
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best["batch"] // 2, results


def fit(arm, seed, device, epochs, limit, sizes):
    bank = Bank(device, limit)
    batch_size, probes = (sizes[0], []) if len(sizes) == 1 else benchmark(bank, arm, sizes, seed)
    torch.manual_seed(seed)
    natural_rng, paste_rng = seed_rng(device, seed+1), seed_rng(device, seed+2)
    order_rng = seed_rng(device, seed+3)
    net = Detector().to(device, memory_format=torch.channels_last).train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0012, weight_decay=.025, fused=device.type == "cuda")
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=.00006)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for ep in range(epochs):
        started = time.perf_counter()
        stats = torch.zeros(7, device=device)
        steps = 0
        for idx in bank.idx[torch.randperm(len(bank.idx), generator=order_rng, device=device)].split(batch_size):
            batch = bank.batch(idx, arm, natural_rng, paste_rng)
            optimizer.zero_grad(set_to_none=True)
            with amp(device):
                loss = objective(net, batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            stats[:3] += torch.stack([loss.detach(), batch[6].sum().float(), batch[7].sum()])
            stats[3:] += batch[9].sum(0)
            steps += 1
        loss, positive, augmented, low, mid, high, same = stats.tolist()
        if not np.isfinite(loss):
            raise RuntimeError(f"Invalid loss at epoch {ep+1}")
        if arm != "clean" and limit is None and augmented < len(bank.idx)*.02:
            raise RuntimeError(f"Only {augmented:.0f} accepted pastes at epoch {ep+1}")
        schedule.step()
        record = dict(epoch=ep+1, loss=loss/steps, positives=int(positive), pasted=int(augmented),
                      visible_04_15=int(low), visible_15_36=int(mid), visible_36_70=int(high), same_color=int(same),
                      seconds=time.perf_counter()-started, gpu_util_pct=gpu_util(device))
        history.append(record)
        print(arm, seed, json.dumps(record), flush=True)
    path = OUT / f"{arm}_{seed}"
    torch.save({k: t.detach().cpu() for k,t in net.state_dict().items()}, path.with_suffix(".pt"))
    path.with_suffix(".json").write_text(json.dumps(dict(startup=bank.load_report, probes=probes,
                                                         batch=2*batch_size, epochs=history), indent=2))


def evidence(mask, points):
    visible = (mask[:, 0].sigmoid() >= .50) & (mask[:, 1].sigmoid() < .50)
    inside = ((points >= 0) & (points < points.new_tensor([W, H]))).all(-1)
    visible &= inside
    pixels = visible.sum((1, 2)) * ((points[:, 0, 1, 0]-points[:, 0, 0, 0]).square())
    values = points.flatten(1, 2)
    keep = visible.flatten(1)[...,None]
    lo = values.masked_fill(~keep, 1e4).amin(1)
    hi = values.masked_fill(~keep, -1e4).amax(1)
    # Use the strongest visible point for reviewed box matching.
    peak = (mask[:, 0].sigmoid() * (1-mask[:, 1].sigmoid())).masked_fill(~visible, -1).flatten(1).argmax(1)
    peak = values[torch.arange(len(peak),device=points.device),peak]
    size = (hi-lo).clamp_min(0)
    return pixels, torch.cat([lo, size], 1), peak


def pairwise_iou(a, b):
    overlap = (np.minimum(a[:2]+a[2:], b[..., :2]+b[..., 2:])-np.maximum(a[:2], b[..., :2])).clip(0).prod(-1)
    return overlap/(a[2:].prod()+b[..., 2:].prod(-1)-overlap).clip(1e-6)


def baseline_bank(split, shard, table, device):
    path = layer.rear.SCRATCH / f"bank_{split}_{shard}.npz"
    if not path.exists():
        p, xy, valid, attrs, offsets = layer.rear.bank_parts(table, split, device)
        np.savez(path, p=p, xy=xy, valid=valid, offsets=offsets, **{f"attr_{i}":a for i,a in enumerate(attrs)})
    with np.load(path) as z:
        return z["p"], z["xy"], z["valid"], [z[f"attr_{i}"] for i in range(4)], z["offsets"]


@torch.inference_mode()
def predict(arm, seed, split, shard, device):
    table = layer.rear.collect(split, shard)
    p, xy, valid, attrs, offsets = baseline_bank(split, shard, table, device)
    pixels = np.load(layer.rear.SCRATCH / f"{split}_{shard}_rgb.npy", mmap_mode="r")
    boxes = (table["x"][...,v.BOX] * layer.rear.SCALE).astype("f")
    si, at = np.nonzero(valid & (table["x"][...,0] >= .85) & (boxes[...,2:].min(-1)>=7))
    net = Detector().to(device, memory_format=torch.channels_last).eval()
    net.load_state_dict(torch.load(OUT/f"{arm}_{seed}.pt", map_location="cpu", weights_only=True))
    values = []
    for start in range(0, len(si), 256):
        sl = slice(start, start+256)
        rgb = torch.from_numpy(pixels[si[sl]]).to(device).permute(0,3,1,2)
        front = torch.from_numpy(boxes[si[sl],at[sl]]).to(device)
        crop, points, _, origin, width = inputs(rgb, front)
        with amp(device):
            raw, mask = net(crop.contiguous(memory_format=torch.channels_last))
        support, visible_box, peak = evidence(mask, points)
        ctr, wh = raw[:,17:].sigmoid().split(2,1)
        wh *= width[:,None]
        rear = torch.cat([origin + ctr*width[:,None] - wh*.5, wh],1)
        values.append(torch.cat([raw, rear, support[:,None], visible_box, peak],1).cpu().numpy())
    data = np.concatenate(values) if values else np.zeros((0,32))
    np.savez(OUT/f"{arm}_{seed}_{split}_{shard}.npz", names=table["names"], si=si, at=at, data=data,
             boxes=boxes, probability=p, valid=valid, xy=xy, offsets=offsets,
             **{f"attr_{i}":a for i,a in enumerate(attrs)})
    print(arm, seed, split, shard, len(si), "foreground queries", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "probe", "fit", "predict"))
    parser.add_argument("--arm", choices=ARMS, default="clean")
    parser.add_argument("--seed", type=int, default=421)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--shard", type=int, choices=(0,1), default=0)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batches", type=int, nargs="+")
    args = parser.parse_args()
    _, device = v.setup(args.seed)
    OUT.mkdir(exist_ok=True)
    layer.rear.SCRATCH.mkdir(exist_ok=True)
    if args.stage in ("fit", "probe"):
        sizes = args.batches or ((64,128,192,256,384,512) if device.type == "cuda" else (4,8))
        if args.stage == "probe":
            bank = Bank(device, args.limit)
            _, probes = benchmark(bank, args.arm, sizes, args.seed)
            (OUT/f"probe_{args.arm}_{args.seed}.json").write_text(json.dumps(dict(startup=bank.load_report, candidates=probes), indent=2))
        else:
            fit(args.arm,args.seed,device,args.epochs,args.limit,sizes)
    elif args.stage == "prepare":
        layer.rear.prepare(args.split,args.shard,device)
        baseline_bank(args.split,args.shard,layer.rear.collect(args.split,args.shard),device)
    else:
        predict(args.arm,args.seed,args.split,args.shard,device)




