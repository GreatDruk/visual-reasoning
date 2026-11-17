"""Train a native-resolution center detector on real scenes and occluded copy-paste."""

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path.cwd()))
from src.models.slot_program import EXACT, EXCLUSIVE_GROUPS, WORD_INDEX, parse_program
from src.pixel_regions import HEIGHT, WIDTH, read_rgb
from src.scene_posterior import _execute, sample_scene_scores
from src.text_aug import canonical_form
from src.twins import find_twins


ROOT = Path("data")
OUT = Path("artefacts/research")
GROUPS = [np.array([WORD_INDEX[w] for w in group]) for group in EXCLUSIVE_GROUPS]
STRIDE = 4
GRID = HEIGHT // STRIDE, WIDTH // STRIDE
FIELDS = ("size", "color", "material", "shape")


def read_rows(split):
    frame = pd.read_csv(ROOT / f"{split}.csv")
    groups = frame.groupby("image", sort=False).indices
    captions = [canonical_form(r.caption, r.category) for r in frame.itertuples()]
    return frame, groups, captions, [parse_program(c) for c in captions]


def mask_for(feature, box):
    return cv2.resize(feature[47:147].reshape(10, 10), tuple(box[2:]), interpolation=cv2.INTER_LINEAR)


def load_train():
    frame, groups, _, programs = read_rows("train")
    names = list(groups)
    with np.load(OUT / "teacher_train.npz") as cache:
        assert list(cache["names"]) == names
        features = cache["features"]
        valid = cache["valid"]
        probability = cache["p"]
        attributes = [cache[attr] for attr in FIELDS]
    ground = frame.label.to_numpy()
    examples = []
    donor = []
    for i, indices in enumerate(groups.values()):
        keep = valid[i] & (probability[i] > .84)
        if not 3 <= keep.sum() <= 10:
            continue
        raw_attrs = [group[i, keep] for group in attributes]
        top = np.stack([a.argmax(-1) for a in raw_attrs], -1)
        mode = np.zeros(keep.sum(), np.int32)
        for j, word in enumerate(GROUPS):
            mode |= 1 << word[top[:, j]]
        valid_scene = np.ones((1, keep.sum()), dtype=bool)
        xy = features[i, keep, 8:12].copy()
        xy = (xy[:, :2] + xy[:, 2:] * [.5, .88])*2-1
        truths = _execute([programs[k] for k in indices], mode[None], valid_scene,
                          xy[None], probability[i, keep], np.zeros(7), np.full_like(valid_scene, .47, dtype=float))
        if (truths[0] != ground[indices]).sum() > 1:
            continue
        labels = top.copy()
        for j, distribution in enumerate(raw_attrs):
            labels[distribution.max(1) < .68, j] = -100
        rgb = read_rgb(ROOT / "train" / names[i])
        descriptors = features[i, keep]
        boxes = np.rint(descriptors[:, 8:12]*[WIDTH, HEIGHT, WIDTH, HEIGHT]).astype(int)
        masks = [mask_for(feature, box) for feature, box in zip(descriptors, boxes)]
        for box, mask, label, f in zip(boxes, masks, labels, descriptors):
            x, y, w, h = box
            if min(w, h) >= 10 and max(w,h) <= 60 and f[14] > .83 and label[-1] >= 0:
                donor.append((rgb[y:y+h, x:x+w].copy(), mask, label))
        examples.append((rgb, boxes, masks, labels))
    return examples, donor


class Block(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation, groups=width, bias=False),
            nn.BatchNorm2d(width), nn.SiLU(),
            nn.Conv2d(width, width, 1, bias=False), nn.BatchNorm2d(width),
        )

    def forward(self, x):
        return F.silu(self.layers(x) + x)


class CenterDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(16), nn.SiLU(),
            nn.Conv2d(16, 40, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(40), nn.SiLU(),
            *[Block(40, dilation) for dilation in (1, 2, 3, 1, 2, 1)],
        )
        self.head = nn.Conv2d(40, 20, 1)
        nn.init.constant_(self.head.bias[0], -2.19)

    def forward(self, rgb):
        return self.head(self.features(rgb))


def place_behind(rgb, boxes, masks, labels, donors, rng):
    """Paste behind existing objects, keeping original forward pixels exactly."""
    occlusion = np.zeros((HEIGHT, WIDTH), np.float32)
    for (x, y, w, h), mask in zip(boxes, masks):
        view = occlusion[y:y+h, x:x+w]
        np.maximum(view, mask, out=view)
    for _ in range(9):
        patch, alpha, label = donors[rng.integers(len(donors))]
        j = rng.integers(len(boxes))
        cx, cy, cw, ch = boxes[j]
        scale = rng.uniform(.68, 1.08)
        h, w = [max(6, round(d*scale)) for d in patch.shape[:2]]
        x = int(np.clip(cx + (cw-w)/2 + rng.uniform(-.70, .70)*w, 0, WIDTH-w))
        y = int(np.clip(cy + (ch-h)/2 - rng.uniform(.35, .83)*h, 0, HEIGHT-h))
        alpha = cv2.resize(alpha, (w, h))
        visible = alpha * (1-occlusion[y:y+h, x:x+w])
        ratio = visible.sum()/max(1., alpha.sum())
        existing = boxes[:, :2] + boxes[:, 2:]/2
        if not .26 <= ratio <= .87 or np.linalg.norm(existing - [x+w/2, y+h/2], axis=1).min() < 9:
            continue
        decorated = rgb.copy()
        view = decorated[y:y+h, x:x+w]
        new = cv2.resize(patch, (w, h))
        view[:] = np.rint(view*(1-visible[..., None]) + new*visible[..., None]).astype(np.uint8)
        return decorated, np.vstack((boxes, [x,y,w,h])), np.vstack((labels, label))
    return rgb, boxes, labels


def batches(examples, donors, indices, rng, device, augment, paste_rng=None):
    rgb = []
    centers = np.zeros((len(indices), *GRID), np.float32)
    regression = np.zeros((len(indices), 4, *GRID), np.float32)
    categories = np.full((len(indices), 4, *GRID), -100, np.int64)
    paste_rng = paste_rng if paste_rng is not None else rng
    for n, i in enumerate(indices):
        im, boxes, masks, labels = examples[i]
        if augment and paste_rng.random() < .8:
            im, boxes, labels = place_behind(im, boxes, masks, labels, donors, paste_rng)
        if rng.random() < .5:
            im = im[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0] = WIDTH - boxes[:, 0] - boxes[:, 2]
        rgb.append(im)
        for (x, y, w, h), attrs in zip(boxes, labels):
            px, py = (x+w/2)/STRIDE, (y+h/2)/STRIDE
            xx, yy = int(px), int(py)
            centers[n, yy, xx] = 1
            regression[n, :, yy, xx] = px-xx, py-yy, np.log(w/24), np.log(h/24)
            categories[n, :, yy, xx] = attrs
    images = torch.tensor(np.stack(rgb), device=device).permute(0,3,1,2).float()/255
    return images.contiguous(memory_format=torch.channels_last), *[
        torch.as_tensor(x, device=device) for x in (centers, regression, categories)
    ]


def loss_for(output, centers, regression, attrs):
    head = output[:, 0]
    present = centers.bool()
    count = centers.sum().clamp_min(1)
    p = head.sigmoid()
    open_region = F.max_pool2d(centers[:, None], 3, stride=1, padding=1)[:, 0]
    weights = torch.where(present, (1-p).square(), p.square()*(1-.85*open_region))
    heat = (F.binary_cross_entropy_with_logits(head, centers, reduction="none")*weights).sum()/count
    locations = F.smooth_l1_loss(output[:, 1:5].permute(0,2,3,1)[present],
                                 regression.permute(0,2,3,1)[present])
    categories = 0
    for j, g in enumerate(output[:, 5:].split([len(group) for group in GROUPS], dim=1)):
        y = attrs[:, j]
        categories += F.cross_entropy(g, y, ignore_index=-100, reduction="sum") / y.ge(0).sum().clamp_min(1)
    return heat + .55*locations + .20*categories


def training_counts():
    frame, _, _, programs = read_rows("train")
    by_image = {}
    images, features, labels = [], [], []
    for row, p in zip(frame.itertuples(), programs):
        if row.label != 1 or p.op != EXACT or p.relation:
            continue
        if row.image not in by_image:
            by_image[row.image] = len(images)
            images.append(read_rgb(ROOT / "train" / row.image))
        words = [WORD_INDEX[w] for w in p.a]
        features.append([by_image[row.image], *words, *([0]*(4-len(words)))])
        labels.append(p.number)
    features = np.asarray(features)
    return np.stack(images), features[:,0], features[:,1:], np.asarray(labels,np.float32)


def count_loss(output, attributes, targets):
    heat = output[:,0]
    peaks = heat == F.max_pool2d(heat[:,None], 3, 1, 1)[:,0]
    occupancy = peaks * (heat/.65).sigmoid()
    choices = torch.cat([x.softmax(1) for x in output[:,5:].split([len(g) for g in GROUPS], dim=1)],1)
    indices = torch.as_tensor(np.concatenate(GROUPS), device=output.device)
    word_prob = torch.ones((len(output), len(WORD_INDEX)+1,*GRID), device=output.device)
    word_prob = word_prob.index_copy(1,indices,choices)
    matched = word_prob[torch.arange(len(output),device=output.device)[:,None],attributes].prod(1)
    return F.smooth_l1_loss((occupancy*matched).sum((1,2)),targets)


@torch.no_grad()
def evaluate(model, device, val, variant):
    model.eval()
    frame, groups, captions, programs, rgb = val
    output = []
    for chunk in np.array_split(rgb, max(1, len(rgb)//32)):
        t = torch.tensor(chunk, device=device).permute(0,3,1,2).float()/255
        output.append(model(t.contiguous(memory_format=torch.channels_last)).cpu())
    result = torch.cat(output)
    p = result[:,0].sigmoid()
    peaks = (p == F.max_pool2d(p[:, None], 3, 1, 1)[:,0]) & (p >= .08)
    scores = np.empty(len(frame))
    count = []
    for i, idx in enumerate(groups.values()):
        ys, xs = torch.where(peaks[i])
        keep = p[i, ys,xs].argsort(descending=True)[:20]
        ys, xs = ys[keep], xs[keep]
        center = (result[i,1:3,ys,xs].clip(-.5,1.5) + torch.stack((xs,ys)))*STRIDE
        box = result[i,3:5,ys,xs].clip(-2,1.6).exp()*24
        center[1] += .38*box[1]
        xy = center.T.numpy()/[WIDTH,HEIGHT]*2-1
        pp = p[i,ys,xs].numpy()
        attrs = [z.softmax(0).T.numpy() for z in result[i,5:,ys,xs].split([len(g) for g in GROUPS], dim=0)]
        pairs = find_twins(frame.image.iloc[idx], [captions[j] for j in idx], frame.category.iloc[idx])
        scores[idx] = sample_scene_scores([programs[j] for j in idx], pairs, pp, list(zip(GROUPS,attrs)), xy,
                                          np.zeros(7), 42)
        for j in idx:
            program = programs[j]
            if frame.label.iat[j] and program.op == EXACT and not program.a and not program.relation:
                count.append((int((pp>.5).sum()), program.number))
    n = np.array(count)
    print(variant, "AUC", roc_auc_score(frame.label, scores), "exact count scenes",
          int((n[:,0] == n[:,1]).sum()), len(n), flush=True)
    np.savez(OUT / f"native_{variant}.npz", ids=frame.id.to_numpy(), joint=scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--scenes-per-epoch", type=int, default=2304)
    parser.add_argument("--learning-rate", type=float, default=.002)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--without-occlusion", action="store_true")
    parser.add_argument("--count-supervision", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.manual_seed(148 + args.resume)
    rng = np.random.default_rng(148 + args.resume)
    paste_rng = np.random.default_rng(281 + args.resume) if args.run_tag else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.monotonic()
    examples, donors = load_train()
    print("Train scenes", len(examples), "candidate donor objects", len(donors), device, flush=True)
    frame, groups, captions, programs = read_rows("val")
    val = frame, groups, captions, programs, np.stack([read_rgb(ROOT / "val" / name) for name in groups])
    model = CenterDetector().to(device=device, memory_format=torch.channels_last)
    if args.resume:
        model.load_state_dict(torch.load(OUT/f"native_epoch{args.resume}.pt",weights_only=True,map_location="cpu"))
    weak_counts = training_counts() if args.count_supervision else None
    if weak_counts is not None:
        print("Train count labels",len(weak_counts[-1]),flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.resume+1, args.epochs+1):
        model.train()
        chosen = rng.permutation(len(examples))[:args.scenes_per_epoch]
        history = []
        for part in np.array_split(chosen, max(1, len(chosen)//32)):
            images, targets, boxes, labels = batches(
                examples, donors, part, rng, device, augment=epoch > 3 and not args.without_occlusion,
                paste_rng=paste_rng,
            )
            if weak_counts is None:
                loss = loss_for(model(images), targets, boxes, labels)
            else:
                rgb, scenes, filters, counts = weak_counts
                sample = rng.integers(len(counts), size=10)
                extra = torch.as_tensor(rgb[scenes[sample]], device=device).permute(0,3,1,2).float()/255
                combined = torch.cat((images,extra),dim=0).contiguous(memory_format=torch.channels_last)
                visual, count = model(combined).split((len(images),len(extra)),dim=0)
                partial = count_loss(count, torch.as_tensor(filters[sample],device=device),
                                     torch.as_tensor(counts[sample],device=device))
                loss = loss_for(visual, targets, boxes, labels) + .25*partial
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            history.append(float(loss.detach()))
        print(epoch, float(np.mean(history)), round((time.monotonic()-start)/60, 2), flush=True)
        if epoch in (3,5,8,12,args.epochs):
            key = f"{args.run_tag}_epoch{epoch}" if args.run_tag else f"epoch{epoch}"
            evaluate(model, device, val, key)
            torch.save(model.state_dict(), OUT / f"native_{key}.pt")


if __name__ == "__main__":
    main()
