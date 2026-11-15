"""Can classical image processing find the objects? Count probe against caption truth.

usage: uv run python scripts/classical_count_probe.py

Ground truth: true train captions "exactly N objects" give the total object count of an
image. Pipeline (no learning, no pretrained model): robust quadratic background model per
channel -> foreground = large residual or high saturation -> morphology -> connected
components, optionally split by distance-transform peaks (watershed-like). Thresholds
are grid-searched on the first 200 images and reported on the remaining ones.
"""

import itertools
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

from src.utils import resolve_path

NUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}
PATTERN = re.compile(r"^(?:there (?:are|is) |the scene contains )?exactly (\w+) objects?(?: (?:are|is) in the scene)?$")


def truth():
    d = pd.read_csv(resolve_path("data/train.csv"))
    d = d[d.label == 1]
    rows = [(img, NUM[m.group(1)]) for img, c in zip(d.image, d.caption) if (m := PATTERN.match(c))]
    return pd.DataFrame(rows, columns=["image", "n"]).drop_duplicates("image")


def background_residual(img):
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    basis = np.stack([np.ones_like(xx), xx, yy, xx * yy, xx**2, yy**2], -1).reshape(-1, 6)
    res = np.empty_like(img)
    for c in range(3):
        v = img[..., c].reshape(-1)
        keep = np.ones(len(v), bool)
        for _ in range(3):  # robust refit: drop object pixels
            coef, *_ = np.linalg.lstsq(basis[keep], v[keep], rcond=None)
            r = v - basis @ coef
            keep = np.abs(r) < 2.5 * np.median(np.abs(r[keep])) + 1e-3
        res[..., c] = r.reshape(h, w)
    return res


def count(img, t_res, t_sat, min_area, split):
    res = background_residual(img)
    sat = img.max(-1) - img.min(-1)
    fg = (np.abs(res).max(-1) > t_res) | (sat > t_sat)
    fg = ndi.binary_opening(fg, iterations=1)
    fg = ndi.binary_closing(fg, iterations=2)
    fg = ndi.binary_fill_holes(fg)
    lab, n = ndi.label(fg)
    if not n:
        return 0
    areas = ndi.sum(fg, lab, range(1, n + 1))
    if not split:
        return int((areas >= min_area).sum())
    total = 0
    dist = ndi.distance_transform_edt(fg)
    for k, a in enumerate(areas, 1):
        if a < min_area:
            continue
        comp = np.where(lab == k, dist, 0)
        peaks = (comp == ndi.maximum_filter(comp, size=split)) & (comp > 3)
        total += max(1, ndi.label(peaks)[1])
    return total


def main():
    t = truth()
    imgs = {n: np.asarray(Image.open(resolve_path(f"data/train/{n}")).convert("RGB"), np.float32) / 255 for n in t.image}
    tune, held = t.iloc[:200], t.iloc[200:]
    grid = list(itertools.product([0.04, 0.06, 0.08, 0.10], [0.12, 0.18, 0.25], [20, 40, 80], [0, 9, 15]))
    best = None
    for params in grid:
        acc = np.mean([count(imgs[i], *params) == n for i, n in zip(tune.image, tune.n)])
        if best is None or acc > best[0]:
            best = (acc, params)
    acc, params = best
    pred = np.array([count(imgs[i], *params) for i in held.image])
    err = pred - held.n.values
    print(f"truth images {len(t)}; counts {t.n.value_counts().sort_index().to_dict()}")
    print(f"best params (t_res, t_sat, min_area, split) {params}, tune exact {acc:.3f}")
    print(f"held-out {len(held)}: exact {np.mean(err == 0):.3f}, within 1 {np.mean(np.abs(err) <= 1):.3f}, "
          f"under {np.mean(err < 0):.3f}, over {np.mean(err > 0):.3f}")
    for n in sorted(held.n.unique()):
        m = held.n.values == n
        print(f"  n={n}: exact {np.mean(err[m] == 0):.2f} ({m.sum()} images)")


if __name__ == "__main__":
    main()
