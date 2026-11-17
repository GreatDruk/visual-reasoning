"""Select contour/center disagreements and render blinded real-scene review sheets."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("artefacts/research")
HEIGHT, WIDTH = 160, 240
COLORS = ("gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow")


def intersection(a, b):
    return np.maximum(0, np.minimum(a[:2] + a[2:], b[:2] + b[2:]) - np.maximum(a[:2], b[:2])).prod()


def select(split, limit):
    archive = ROOT / "learned_fusion16" / f"{split}.npz"
    with np.load(archive) as saved:
        x, valid, names = (saved[key] for key in ("x", "valid", "names"))
    possible = []
    for name, scene, active in zip(names, x, valid):
        scene = scene[active]
        box = scene[:, 16:20] * [WIDTH, HEIGHT, WIDTH, HEIGHT]
        p, support = scene[:, 0], scene[:, 20]
        new = (scene[:, 21] == 1) & (scene[:, 22] == 1) & (p < 0)
        geometric = np.flatnonzero(p > 1.35)
        best = None
        for i in geometric:
            for j in np.flatnonzero((support > .56) & (new | (p < .45))):
                a, b = box[i], box[j]
                area = np.minimum(a[2:].prod(), b[2:].prod())
                touching = intersection(a, b) / max(1, area)
                if touching < .19 or area < 30:
                    continue
                mid = np.linalg.norm(a[:2] + a[2:] / 2 - b[:2] - b[2:] / 2)
                if mid < 3.2:
                    continue
                hue_a = scene[i, 3:11].argmax()
                hue_b = scene[j, 3:11].argmax()
                if hue_a != hue_b:
                    continue
                low = np.maximum(0, np.floor(np.minimum(a[:2], b[:2]) - 4)).astype(int)
                high = np.minimum([WIDTH, HEIGHT], np.ceil(np.maximum(a[:2]+a[2:], b[:2]+b[2:]) + 4)).astype(int)
                crowded = mid > max(6.5, .22 * max(a[2:].max(), b[2:].max())) or touching < .55
                score = float(support[j] + .16*bool(new[j]) + .10*touching)
                case = (score, split, str(name), COLORS[hue_a], *map(int, low), *map(int, high), crowded)
                if best is None or case[0] > best[0]:
                    best = case
        if best is not None:
            possible.append(best)
    possible.sort(reverse=True)
    # A fixed round-robin over confidence bands retains both subtle and clear disagreements.
    bands = [group for crowded in (False, True)
             for group in np.array_split([idx for idx, row in enumerate(possible) if row[-1] == crowded], 3)]
    chosen = [possible[band[rank]] for rank in range(max(map(len, bands), default=0))
              for band in bands if rank < len(band)][:limit]
    print(split, "eligible distinct scenes", len(possible), "selected", len(chosen))
    return chosen


def sheets(frame, folder=None):
    folder = ROOT / "overlap_review" if folder is None else folder
    folder.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=16)
    for (split, page), chunk in frame.groupby(["split", "page"], sort=True):
        sheet = Image.new("RGB", (4*246, 4*238), "white")
        draw = ImageDraw.Draw(sheet)
        for lane, row in enumerate(chunk.itertuples(index=False)):
            ox, oy = lane % 4 * 246 + 4, lane // 4 * 238 + 2
            with Image.open(Path("data") / split / row.image) as source:
                rgb = source.convert("RGB")
            crop = rgb.crop((row.x0,row.y0,row.x1,row.y1))
            crop.thumbnail((190, 138), Image.Resampling.NEAREST)
            if crop.width < 120 and crop.height < 105:
                scale = min(190 / crop.width, 130 / crop.height)
                crop = crop.resize((round(scale*crop.width), round(scale*crop.height)), Image.Resampling.NEAREST)
            draw.text((ox, oy), f"{row.case}  {row.image}  {row.color}", font=font, fill="black")
            sheet.paste(crop, (ox+(240-crop.width)//2, oy+26))
            overview = rgb.resize((100, 67))
            small = ImageDraw.Draw(overview)
            small.rectangle((row.x0*100/240, row.y0*67/160, row.x1*100/240, row.y1*67/160),
                            outline="#ffef65", width=1)
            sheet.paste(overview, (ox+70, oy+167))
        sheet.save(folder / f"{split}-{page:02}.jpg", quality=92)
    frame.to_csv(folder / "queue.csv", index=False)


def single_controls(split, annotations, limit):
    with np.load(ROOT / "learned_fusion16" / f"{split}.npz") as saved:
        x, valid, names = (saved[key] for key in ("x", "valid", "names"))
    accepted = annotations[(annotations.number == 2) & (annotations.split == "train")]
    avoided = set(annotations.image)
    typical_extents = {}
    for color, same in accepted.groupby("color"):
        typical_extents[color] = np.median(np.c_[same.x1-same.x0, same.y1-same.y0], axis=0)
    rng = np.random.default_rng(928)
    queues = [[] for _ in COLORS]
    for scene, keep, name in zip(x, valid, names):
        if name in avoided:
            continue
        scene = scene[keep]
        boxes = scene[:, 16:20] * [WIDTH, HEIGHT, WIDTH, HEIGHT]
        positions = boxes[:, :2] + .5*boxes[:, 2:]
        colors = scene[:, 3:11].argmax(1)
        strong = scene[:, 0] > 1.8
        for i in np.flatnonzero(strong & (boxes[:, 2:].min(1) >= 8) & (scene[:,20] > .65)):
            color = colors[i]
            if color == 0:
                continue
            typical = typical_extents.get(COLORS[color])
            if typical is None:
                continue
            extent = np.maximum(boxes[i,2:]+8, typical * rng.uniform(.88, 1.12))
            center = positions[i] + rng.uniform(-4, 4, size=2)
            lo = (center-.5*extent).clip(0, [WIDTH,HEIGHT]).astype(int)
            hi = (center+.5*extent).clip(0, [WIDTH,HEIGHT]).astype(int)
            inside = ((positions >= lo) & (positions < hi)).all(-1)
            matching = (colors == color) & inside
            if (matching & strong).sum() != 1:
                continue
            risk = ((scene[:,20] > .52) & matching & ~strong).sum()
            if risk:
                continue
            fit = np.abs(np.log((hi-lo)/typical)).sum()
            score = float(-fit + rng.uniform(0,.16))
            queues[color].append((score,split,str(name),COLORS[color],*lo,*hi,False))
    for group in queues:
        group.sort(reverse=True)
    used, out = set(avoided), []
    for rank in range(max(map(len,queues),default=0)):
        for group in queues:
            if rank < len(group) and group[rank][2] not in used:
                out.append(group[rank])
                used.add(group[rank][2])
        if len(out) >= limit:
            break
    return out[:limit]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=144)
    parser.add_argument("--val", type=int, default=80)
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    columns = ("priority", "split", "image", "color", "x0", "y0", "x1", "y1", "wide_separation")
    if args.controls:
        annotations = pd.read_csv(ROOT / "overlap_review/review_labels.csv")
        frame = pd.DataFrame(single_controls("train", annotations, args.train) +
                             single_controls("val", annotations, args.val), columns=columns)
    else:
        frame = pd.DataFrame(select("train", args.train) + select("val", args.val), columns=columns)
    frame["case"] = frame.groupby("split").cumcount().add(1)
    frame["page"] = (frame.case - 1) // 16 + 1
    sheets(frame, args.output_dir or (ROOT / "overlap_controls" if args.controls else None))
