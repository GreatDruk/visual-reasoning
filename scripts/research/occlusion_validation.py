"""Measure recovery of known inserted objects on held-out JPEG backgrounds."""

from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path.cwd()))
from native_centers import CenterDetector, FIELDS, GRID, HEIGHT, STRIDE, WIDTH
from native_centers import load_train, mask_for, place_behind, read_rows
from src.pixel_regions import read_rgb


@torch.no_grad()
def main():
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    _, donors = load_train()
    _, scenes, _, _ = read_rows("val")
    rng = np.random.default_rng(9307)
    source = Path("artefacts/research")
    cache = np.load(source / "scene_silhouette.npz")
    features, p, valid = [cache[key] for key in ("features", "p", "valid")]
    top = np.stack([cache[key].argmax(-1) for key in FIELDS], axis=-1)
    names = list(scenes)
    scenes, markers, targets = [], [], []
    for i in rng.permutation(len(names))[:320]:
        keep = valid[i] & (p[i] > .88)
        useful = features[i, keep]
        if len(useful) < 3:
            continue
        boxes = np.rint(useful[:, 8:12]*[WIDTH,HEIGHT,WIDTH,HEIGHT]).astype(int)
        masks = [mask_for(f, b) for f, b in zip(useful, boxes)]
        rgb = read_rgb(Path("data/val") / names[i])
        decorated, augmented, label = place_behind(rgb, boxes, masks, top[i,keep], donors, rng)
        if len(augmented) == len(boxes) or label[-1, -1] < 0:
            continue
        x, y, w, h = augmented[-1]
        scenes.extend([rgb, decorated])
        markers.append((x+.5*w, y+.5*h))
        targets.append(label[-1,-1])
    batch = np.stack(scenes)
    markers = np.array(markers)
    y = torch.arange(GRID[0])[:, None].expand(GRID)
    x = torch.arange(GRID[1])[None, :].expand(GRID)
    targets = np.array(targets)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CenterDetector().to(device=device, memory_format=torch.channels_last)
    model.eval()
    for name in sys.argv[1:] or ("epoch3", "epoch5", "epoch8"):
        checkpoint = source / f"native_{name}.pt"
        if not checkpoint.exists():
            continue
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
        hits = np.zeros((len(markers),2), dtype=bool)
        shape = np.zeros_like(hits)
        data = []
        for images in np.array_split(batch, max(1,len(batch)//32)):
            rgb = torch.as_tensor(images,device=device).permute(0,3,1,2).float()/255
            data.append(model(rgb.contiguous(memory_format=torch.channels_last)).cpu())
        output = torch.cat(data)
        p = output[:,0].sigmoid()
        peaks = (p == F.max_pool2d(p[:,None],3,1,1)[:,0]) & (p > .5)
        dx = output[:,1].clip(-.5,1.5)
        dy = output[:,2].clip(-.5,1.5)
        for i, center in enumerate(markers):
            for version in (0,1):
                j = 2*i + version
                close = ((x+dx[j])*STRIDE-center[0]).square() + ((y+dy[j])*STRIDE-center[1]).square() <= 8**2
                candidate = peaks[j] & close
                hits[i,version] = candidate.any()
                if hits[i,version]:
                    selected = p[j].masked_fill(~candidate, -1).argmax()
                    shape[i,version] = output[j,-3:].reshape(3,-1)[:,selected].argmax() == targets[i]
        print(name, "paired backgrounds", len(markers), "hits before/after insert", hits.sum(0),
              "hits with matching shape", shape.sum(0), flush=True)


if __name__ == "__main__":
    main()
