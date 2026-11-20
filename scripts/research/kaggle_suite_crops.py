"""Frozen and fine-tuned crop representations on identical detector proposals."""

import gc
import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F

from kaggle_suite_common import CACHE, FEATURES, INPUT, N_ATTRS, REPORT, WEIGHTS
from kaggle_suite_common import amp, configure_split, fixed_table, load, save, setup, tick, tune_batch, write_json
from native_centers import GROUPS


def table(name, split):
    with np.load(CACHE / f"{name}_{split}.npz") as data:
        return tuple(data[k] for k in ("x", "xy", "valid"))


def active_crops(x, valid):
    return valid & (x[..., 0] >= -1.6)


def extract_pixels(split, scenes, boxes, size, device):
    pixels = split.images(scenes, device)
    box = torch.from_numpy(boxes).to(device)
    axis = (torch.arange(size, device=device, dtype=torch.float32) + .5) / size
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    grid = box[:, None, None, :2] + torch.stack((xx, yy), dim=-1) * box[:, None, None, 2:]
    return F.grid_sample(pixels, grid.mul(2).sub(1), align_corners=False, padding_mode="border")


def hog(pixels):
    gray = (pixels * pixels.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1)
    gradient_x = F.pad(gray[:, :, 2:] - gray[:, :, :-2], (1, 1))
    gradient_y = F.pad(gray[:, 2:] - gray[:, :-2], (0, 0, 1, 1))
    magnitude = torch.hypot(gradient_x, gradient_y)
    theta = torch.atan2(gradient_y, gradient_x).mul(8 / torch.pi).floor().long() % 8
    cells = (torch.arange(24, device=pixels.device) // 6)
    cells = (cells[:, None] * 4 + cells).flatten()
    slot = theta.flatten(1) + 8 * cells
    hist = pixels.new_zeros((len(gray), 128)).scatter_add_(1, slot, magnitude.flatten(1))
    hist = hist.view(-1, 16, 8)
    return (hist / (hist.sum(-1, keepdim=True) + .04)).flatten(1)


def vision_model():
    from transformers import SiglipVisionModel

    return SiglipVisionModel.from_pretrained(INPUT / "siglip2_vision", local_files_only=True)


def crop_features(model, pixels):
    with amp(pixels.device):
        encoded = model(pixel_values=pixels)
    hidden = encoded.last_hidden_state
    side = int(hidden.shape[1] ** .5)
    reduced = F.adaptive_avg_pool2d(hidden.transpose(1, 2).reshape(len(pixels), -1, side, side), (8, 8))
    return encoded.pooler_output, reduced.flatten(2).transpose(1, 2)


def supervised_indices(split, x, valid, scenes, slots, device):
    if split.name != "train":
        return np.full((len(scenes), 4), -100, np.int16)
    (p, _, present, *attrs), features, _ = fixed_table("train", "sil", device)
    gt = features[..., 8:12]
    targets = np.full((len(scenes), 4), -100, np.int16)
    for start in range(0, len(scenes), 4096):
        sl = slice(start, start + 4096)
        i, j = scenes[sl], slots[sl]
        query = x[i, j, 1+N_ATTRS:5+N_ATTRS]
        ref = gt[i]
        low = np.maximum(query[:, None, :2], ref[..., :2])
        high = np.minimum(query[:, None, :2] + query[:, None, 2:], ref[..., :2] + ref[..., 2:])
        intersection = (high-low).clip(0).prod(-1)
        union = query[:, None, 2:].prod(-1) + ref[..., 2:].prod(-1) - intersection
        scores = intersection / union.clip(1e-8)
        scores = np.where(present[i] & (p[i] > .86), scores, 0)
        selected = scores.argmax(1)
        usable = scores.max(1) >= .55
        for g, attribute in enumerate(attrs):
            probs = attribute[i, selected]
            targets[sl, g] = np.where(usable & (probs.max(1) >= .8), probs.argmax(1), -100)
    return targets


def crop_extract(split_name, shard, tuned_only=False):
    _, device = setup(721)
    split = configure_split(split_name, device)
    x, _, valid = table("thaw", split_name)
    scenes, slots = np.nonzero(active_crops(x, valid))
    choose = np.arange(shard, len(scenes), 2)
    scenes, slots = scenes[choose], slots[choose]
    if not len(scenes):
        raise ValueError(f"{split_name}: no candidates in crop shard {shard}")
    boxes = x[scenes, slots, 1+N_ATTRS:5+N_ATTRS]
    labels = supervised_indices(split, x, valid, scenes, slots, device)
    prefix = CACHE / f"crops_{split_name}_{shard}"
    # Scratch files are disk-backed. Never retain all patch tokens in RAM or VRAM.
    model = vision_model().to(device, dtype=torch.float16 if device.type == "cuda" else torch.float32).eval()
    width = model.config.hidden_size
    if not tuned_only:
        pooled = np.lib.format.open_memmap(str(prefix)+"_sig.npy", mode="w+", dtype=np.float16, shape=(len(scenes), width))
        pixels = np.lib.format.open_memmap(str(prefix)+"_hog.npy", mode="w+", dtype=np.float16, shape=(len(scenes), 128))
    tokens = np.lib.format.open_memmap(str(prefix)+"_tokens.npy", mode="w+", dtype=np.float16,
                                      shape=(len(scenes), 64, width))
    np.savez(str(prefix)+".npz", scenes=scenes, slots=slots, labels=labels)
    started = time.monotonic()

    @torch.inference_mode()
    def encode(i):
        crop = extract_pixels(split, scenes[i], boxes[i], 256, device).mul(2).sub(1)
        return crop_features(model, crop)

    batches = tuple(dict.fromkeys(min(n, len(scenes)) for n in (128, 96, 64, 32, 16, 8)))
    batch = tune_batch(f"crops/{split_name}/{shard}", batches,
                       lambda n: encode(np.arange(n) % len(scenes)), device)
    total = (len(scenes) + batch - 1) // batch
    with torch.inference_mode():
        for step, start in enumerate(range(0, len(scenes), batch), 1):
            i = np.arange(start, min(start+batch, len(scenes)))
            a, b = encode(i)
            tokens[i] = b.cpu().numpy()
            if not tuned_only:
                pooled[i] = a.cpu().numpy()
                pixels[i] = hog(extract_pixels(split, scenes[i], boxes[i], 24, device)).cpu().numpy()
            if step in (1, total) or step % max(1, total // 5) == 0:
                tick(f"crops/{split_name}/{shard}", step, total, started)
    tokens.flush()
    if not tuned_only:
        pooled.flush()
        pixels.flush()


def extract_head(model, device):
    # Older Transformers wraps the vision encoder. Newer releases flatten it.
    return getattr(model, "vision_model", model).head.to(device=device, dtype=torch.float32)


def pool_head(device):
    head = extract_head(vision_model(), device)
    gc.collect()
    return head


def check_pool():
    _, device = setup(721)
    model = vision_model().to(device, dtype=torch.float16 if device.type == "cuda" else torch.float32).eval()
    with torch.no_grad():
        pooled, tokens = crop_features(model, torch.rand(2, 3, 256, 256, device=device).mul(2).sub(1))
    if pooled is None or not torch.isfinite(pooled).all() or not torch.isfinite(tokens).all():
        raise RuntimeError("SigLIP crop preflight produced invalid embeddings")
    head = extract_head(model, device).train()
    del model
    width = head.probe.shape[-1]
    linear = nn.Linear(width, N_ATTRS).to(device)
    opt = torch.optim.AdamW([*head.parameters(), *linear.parameters()], lr=.00012)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=128.)
    with amp(device):
        scores = linear(head(tokens.float()))
    loss = scores.float().square().mean()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    if not torch.isfinite(loss) or head.probe.grad is None or not torch.isfinite(head.probe.grad).all():
        raise RuntimeError("SigLIP pooling preflight produced an invalid loss or gradient")
    scaler.step(opt)
    scaler.update()
    print(f"[preflight] SigLIP crop extraction, pooling and optimizer step passed on {device}", flush=True)


def load_shards(split):
    for shard in (0, 1):
        prefix = str(CACHE / f"crops_{split}_{shard}")
        with np.load(prefix+".npz") as a:
            info = dict(a)
        yield prefix, info


def train_pool():
    rng, device = setup(721)
    head = pool_head(device).train()
    linear = nn.Linear(head.probe.shape[-1], N_ATTRS).to(device)
    opt = torch.optim.AdamW([*head.parameters(), *linear.parameters()], lr=.00012)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    cache = [(np.load(p+"_tokens.npy", mmap_mode="r"), info["labels"]) for p, info in load_shards("train")]
    examples = [(j, i) for j, (_, labels) in enumerate(cache) for i in np.flatnonzero((labels >= 0).any(1))]
    examples = np.asarray(examples)
    if not len(examples):
        raise ValueError("No usable train-only crop targets")
    start = time.monotonic()

    def loss_for(rows):
        # The small batch copies only the relevant slices from the mmap.
        pixels = np.stack([cache[j][0][i] for j, i in rows])
        truth = np.stack([cache[j][1][i] for j, i in rows])
        x = torch.from_numpy(pixels).to(device, dtype=torch.float32)
        target = torch.from_numpy(truth).to(device).long()
        with amp(device):
            scores = linear(head(x))
        loss = 0
        for k, logits in enumerate(scores.float().split([len(g) for g in GROUPS], -1)):
            loss += F.cross_entropy(logits, target[:, k], ignore_index=-100, reduction="sum")
        return loss / target.ge(0).sum().clamp_min(1)

    def probe(n):
        loss_for(examples[np.arange(n) % len(examples)]).backward()
        opt.zero_grad(set_to_none=True)

    batches = tuple(dict.fromkeys(min(n, len(examples)) for n in (384, 256, 128, 64)))
    batch = tune_batch("siglip_pool", batches, probe, device)
    for epoch in range(1, 5):
        for idx in np.array_split(rng.permutation(len(examples)), max(1, (len(examples)+batch-1)//batch)):
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss_for(examples[idx])).backward()
            scaler.step(opt)
            scaler.update()
        tick("siglip_pool", epoch, 4, start)
    save(head, "siglip_pool")
    save(linear, "siglip_attributes")
    write_json(REPORT / "siglip_pool_train.json", {"matched_train_crops": len(examples), "epochs": 4,
                                                  "minutes": (time.monotonic()-start)/60})


def encode_tuned(split, shard):
    _, device = setup(721)
    prefix = CACHE / f"crops_{split}_{shard}"
    token_path = prefix.with_name(prefix.name + "_tokens.npy")
    output_path = prefix.with_name(prefix.name + "_sigft.npy")
    if not token_path.exists() and output_path.is_file():
        return
    tokens = np.load(token_path, mmap_mode="r")
    head = pool_head(device).eval()
    head.load_state_dict(load(WEIGHTS / "siglip_pool.pt"))
    partial = output_path.with_suffix(".tmp")
    output = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16,
                                      shape=(len(tokens), tokens.shape[-1]))
    with torch.inference_mode(), amp(device):
        for start in range(0, len(tokens), 256):
            part = torch.from_numpy(tokens[start:start+256].copy()).to(device, dtype=torch.float32)
            output[start:start+len(part)] = head(part).half().cpu().numpy()
    output.flush()
    del output, tokens
    partial.replace(output_path)
    token_path.unlink()


def features(split, variant, fitting=False):
    x, xy, valid = table("thaw", split)
    source = [(p, info) for p, info in load_shards(split)]
    path = WEIGHTS / f"{variant}_pca.npz"
    if variant != "hog":
        if fitting:
            sample = []
            rng = np.random.default_rng(732)
            for prefix, _ in source:
                raw = np.load(prefix+f"_{variant}.npy", mmap_mode="r")
                sample.append(raw[rng.choice(len(raw), min(8000, len(raw)), replace=False)].astype("f"))
            train = np.concatenate(sample)
            dimensions = min(128, len(train) - 1, train.shape[1])
            if dimensions < 1:
                raise ValueError("At least two train crops are required for PCA")
            model = PCA(n_components=dimensions, whiten=True, svd_solver="randomized", random_state=732)
            model.fit(train)
            np.savez(path, mean=model.mean_, components=model.components_,
                     scale=np.sqrt(model.explained_variance_).clip(1e-5))
        with np.load(path) as archive:
            mean, components, scale = [archive[k] for k in ("mean", "components", "scale")]
    output = np.zeros((*valid.shape, 128), np.float32)
    for prefix, info in source:
        raw = np.load(prefix+f"_{variant}.npy", mmap_mode="r")
        for begin in range(0, len(raw), 8192):
            part = slice(begin, begin+8192)
            values = raw[part].astype("f")
            if variant != "hog":
                values = ((values-mean) @ components.T / scale).clip(-7, 7)
            else:
                values *= 4
            output[info["scenes"][part], info["slots"][part], :values.shape[-1]] = values
    return np.concatenate((x, output), -1), xy, valid
