"""Compare one-object and two-object renderings within high-confidence regions."""

from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path.cwd()))
from src.primitive_regions import SIZE, templates


def outline(mask):
    return mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3,3), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def fit_two(target, edges):
    """Use the visible masks of two depth-ordered silhouettes in a scene likelihood."""
    h, w = target.shape
    if min(h,w) < 10 or max(h,w) < 21 or target.sum() < 140:
        return None
    scale = min(1., 48/max(h,w))
    width, height = max(1, round(w*scale)), max(1, round(h*scale))
    mask = cv2.resize(target.astype(np.float32), (width,height)) > .52
    response = cv2.resize(edges, (width,height)) > 40
    near_edge = cv2.dilate(response.astype(np.uint8), np.ones((3,3), np.uint8)).astype(bool)
    foreground = mask.astype(np.float32)
    edges_float = near_edge.astype(np.float32)
    bank, _, ratios, types = templates()
    widths = np.unique(np.rint(np.geomspace(6.,max(width,height),10)).astype(int))
    candidates = []
    for idx in (0, 4, 9, 14, 19, 24, 27, 33, 36, 39):
        ar = np.exp(ratios[idx])
        for diameter in widths:
            pw, ph = round(diameter*min(1.,ar)), round(diameter/max(1.,ar))
            if min(pw,ph) < 5 or pw > width or ph > height:
                continue
            raster = cv2.resize(bank[idx].reshape(SIZE,SIZE), (pw,ph)) > .5
            tpl = raster.astype(np.float32)
            border = outline(raster).astype(np.float32)
            area = float(tpl.sum())
            mask_corr = cv2.matchTemplate(foreground,tpl,cv2.TM_CCORR)
            edge_corr = cv2.matchTemplate(edges_float,border,cv2.TM_CCORR)/border.sum()
            quality = mask_corr - 1.9*(area-mask_corr) + .11 * area * edge_corr
            for j in range(2):
                _, value, _, (x,y) = cv2.minMaxLoc(quality)
                if mask_corr[y,x] < .83*area or value < .57*area:
                    break
                rect = np.zeros(mask.shape,dtype=bool)
                rect[y:y+ph,x:x+pw] = raster
                candidates.append((value, rect, (x,y,pw,ph), types[idx]))
                quality[max(0,y-ph//3):y+ph//3+1,max(0,x-pw//3):x+pw//3+1] = -1e6
    pool = []
    for _, candidate, box, kind in sorted(candidates,key=lambda c: -c[0]):
        if any((candidate & other).sum() > .82 * (candidate | other).sum() for other, *_ in pool):
            continue
        pool.append((candidate, box, kind, outline(candidate), int(candidate.sum())))
        if len(pool) == 25:
            break
    if len(pool) < 2:
        return None

    def fidelity(rendered, boundary):
        visible_area = (rendered & mask).sum()
        leak = (rendered & ~mask).sum()
        agreement = visible_area/(mask.sum() + 1.3*leak)
        return agreement + .10*(boundary & near_edge).sum()/max(1,boundary.sum())

    single = max(fidelity(part, line) for part, _, _, line, _ in pool)
    best = single + .115
    found = None
    for i, (a, box_a, kind_a, line_a, size_a) in enumerate(pool):
        for b, box_b, kind_b, line_b, size_b in pool[i+1:]:
            ia = (a & b).sum()
            if ia > .79*min(size_a,size_b) or min(size_a,size_b)-ia < max(22,.09*mask.sum()):
                continue
            a_front = box_a[1]+.87*box_a[3] >= box_b[1]+.87*box_b[3]
            visible_line = line_a | (line_b & ~a) if a_front else line_b | (line_a & ~b)
            # A seam that survives foreground occlusion should leave a real image edge.
            internal = (line_a & b if a_front else line_b & a) & mask
            seam = (internal & near_edge).sum()/max(1,internal.sum())
            if 0 < internal.sum() and seam < .25:
                continue
            fit = fidelity(a | b, visible_line)
            if fit > max(.81,best):
                best = fit
                found = [(box_a,kind_a),(box_b,kind_b)]
    if found is None:
        return None
    return [(np.asarray(box) / scale, int(kind)) for box, kind in found], best-single
