"""Analytic projected silhouettes for separating adjacent geometric objects."""

import itertools
from functools import lru_cache

import cv2
import numpy as np


SIZE = 24


def _rasterize(points):
    low = points.min(0)
    extent = points.max(0) - low
    points = np.rint((points - low) / extent * (SIZE - 1)).astype(np.int32)
    canvas = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillConvexPoly(canvas, cv2.convexHull(points), 1)
    return canvas, extent[0] / extent[1]


@lru_cache(maxsize=1)
def templates():
    """Render only outlines of the three primitive classes from candidate viewpoints."""
    sphere = np.column_stack((np.cos(np.linspace(0, 2*np.pi, 80)),
                              np.sin(np.linspace(0, 2*np.pi, 80))))
    masks, ratios, types = [], [], []

    def add(points, shape):
        mask, aspect = _rasterize(points)
        masks.append(mask.flatten())
        ratios.append(aspect)
        types.append(shape)

    add(sphere, 1)
    for tilt in (.38, .53, .67):
        vertical = np.sqrt(1 - tilt**2)
        for yaw in np.linspace(0, np.pi/4, 10):
            x, y, z = np.array(list(itertools.product((-1., 1.), repeat=3))).T
            left = x*np.cos(yaw) - y*np.sin(yaw)
            depth = x*np.sin(yaw) + y*np.cos(yaw)
            add(np.column_stack((left, tilt*depth + vertical*z)), 0)
        for height in (.82, 1., 1.18):
            add(np.concatenate([sphere*[1., tilt] + [0, z*vertical*height] for z in (-1, 1)]), 2)
    masks = np.stack(masks).astype(np.float32)
    return masks, masks.sum(1), np.log(ratios), np.array(types)


def shape_scores(mask):
    """Silhouette IoU discounted by discrepancy in projected aspect ratio."""
    nonzero = cv2.findNonZero(mask)
    if nonzero is None:
        return np.zeros(3, np.float32)
    x, y, w, h = cv2.boundingRect(nonzero)
    target = cv2.resize(mask[y:y+h, x:x+w].astype(np.float32), (SIZE, SIZE),
                        interpolation=cv2.INTER_AREA).ravel()
    bank, area, ratio, family = templates()
    intersect = (bank * target).sum(1)
    iou = intersect / np.maximum(1, target.sum() + area - intersect)
    score = iou - .50 * np.abs(np.log(w/h) - ratio)
    return np.asarray([score[family == k].max() for k in range(3)], np.float32)


def contour_shape_scores(contour):
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [contour - np.array([x, y], np.int32)], -1, 1, cv2.FILLED)
    return shape_scores(mask)


def split_at_edges(contour, edges):
    """Split a contour when several edge-supported silhouettes explain the pixels better."""
    x, y, w, h = cv2.boundingRect(contour)
    if max(w, h) < 39 or min(w, h) < 14 or cv2.contourArea(contour) < 250:
        return [contour]
    origin = np.array([x, y], dtype=np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [contour - origin], -1, 1, cv2.FILLED)
    parent_fit = shape_scores(mask).max()
    if parent_fit < .79:
        parts = _fit_overlapping(mask, edges[y:y+h, x:x+w])
        if parts is not None:
            return [part + origin for part in parts]
    return [contour]


def split_at_luminance_seam(contour, luminance):
    """Split a same-hue blob when an interior edge separates plausible shapes."""
    x, y, w, h = cv2.boundingRect(contour)
    if w * h < 500 or min(w, h) < 10:
        return [contour]
    origin = np.array([x, y], np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [contour - origin], 0, 1, cv2.FILLED)
    original = shape_scores(mask).max()
    kernel = np.ones((3, 3), np.uint8)
    interior = cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    smooth = cv2.GaussianBlur(luminance[y:y+h, x:x+w], (3, 3), .65)
    for low, high in ((15, 37), (8, 21)):
        edges = (cv2.Canny(smooth, low, high) > 0) & interior
        cut = cv2.dilate(edges.astype(np.uint8), kernel) > 0
        separated = cv2.erode((interior & ~cut).astype(np.uint8), kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(separated)
        cores = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 22]
        if not 2 <= len(cores) <= 4:
            continue
        distances = np.stack([cv2.distanceTransform((labels != i).astype(np.uint8), cv2.DIST_L2, 5)
                              for i in cores])
        assignment = distances.argmin(axis=0)
        masks = [((assignment == i) & (mask > 0)).astype(np.uint8) for i in range(len(cores))]
        fit = np.array([shape_scores(part).max() for part in masks])
        areas = np.array([part.sum() for part in masks])
        if fit.min() < .64 or np.average(fit, weights=areas) <= original + .045:
            continue
        parts = [cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] for part in masks]
        return [max(regions, key=cv2.contourArea) + origin for regions in parts]
    return [contour]


def merge_reflected_faces(contours, planes):
    """Recover continuous warm-metal silhouettes cut by a brown/yellow hue boundary."""
    warm = (planes[2] | planes[3]).astype(np.uint8)
    warm = cv2.morphologyEx(warm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    components, _ = cv2.findContours(warm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    replaced = set()
    merged = []
    for component in components:
        area = cv2.contourArea(component)
        x, y, w, h = cv2.boundingRect(component)
        if area < 100 or not (14 <= min(w,h) and max(w,h) <= 75):
            continue
        parts = []
        for hue in (2, 3):
            for i, part in enumerate(contours[hue]):
                part_area = cv2.contourArea(part)
                if part_area < 18:
                    continue
                px, py, pw, ph = cv2.boundingRect(part)
                if px < x-1 or py < y-1 or px+pw > x+w+1 or py+ph > y+h+1:
                    continue
                inside = [cv2.pointPolygonTest(component, tuple(map(float, point[0])), False) >= 0
                          for point in part[::max(1,len(part)//9)]]
                if np.mean(inside) >= .72:
                    parts.append((hue, i, part_area, part))
        if not 2 <= len(parts) <= 6 or len({hue for hue, *_ in parts}) != 2:
            continue
        fit = contour_shape_scores(component)
        if fit[0] < .81 or sum(p[2] for p in parts) < .63 * area:
            continue
        previous = sum(p[2]*contour_shape_scores(p[3]).max() for p in parts) / sum(p[2] for p in parts)
        if fit[0] < previous + .07:
            continue
        hue = max(parts, key=lambda p: p[2])[0]
        replaced.update((p[0], p[1]) for p in parts)
        merged.append((hue, component))
    return [[part for i, part in enumerate(cs) if (hue, i) not in replaced]
            + [new for color, new in merged if color == hue] for hue, cs in enumerate(contours)]


def _fit_overlapping(mask, edge):
    """Fit a sparse union of overlapping projected objects to an irregular component."""
    h, w = mask.shape
    scale = min(1., 43 / max(h, w))
    size = max(1, round(w * scale)), max(1, round(h * scale))
    fg = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    support = cv2.resize(cv2.dilate(edge, np.ones((3, 3), np.uint8)), size,
                         interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    total = fg.sum()
    bank, _, ratios, _ = templates()
    candidates = []
    indices = (0, 4, 10, 23, 24, 25, 36, 37)
    lengths = np.unique(np.rint(np.geomspace(max(8, .23 * max(size)), max(size), 7)).astype(int))
    for i in indices:
        shape = bank[i].reshape(SIZE, SIZE)
        aspect = np.exp(ratios[i])
        for length in lengths:
            dw = round(length * min(aspect, 1.))
            dh = round(length / max(aspect, 1.))
            if min(dw, dh) < 6 or dw > size[0] or dh > size[1]:
                continue
            template = (cv2.resize(shape, (dw, dh), interpolation=cv2.INTER_LINEAR) >= .5).astype(np.float32)
            area = template.sum()
            outline = template - cv2.erode(template, np.ones((3, 3), np.uint8),
                                            borderType=cv2.BORDER_CONSTANT, borderValue=0)
            inside = cv2.matchTemplate(fg, template, cv2.TM_CCORR)
            sharp = cv2.matchTemplate(support, outline, cv2.TM_CCORR) / max(1., outline.sum())
            score = inside - 2.4 * (area-inside) + .38 * area * sharp
            # Two separated maxima keep a smaller object beside a larger match.
            for _ in range(2):
                _, quality, _, (x, y) = cv2.minMaxLoc(score)
                if quality < .51 * area or inside[y, x] < .77 * area:
                    break
                complete = np.zeros_like(fg)
                complete[y:y+dh, x:x+dw] = template
                candidates.append((quality, complete.astype(bool)))
                score[max(0,y-dh//3):y+dh//3+1, max(0,x-dw//3):x+dw//3+1] = -1e6
    pool = []
    for _, geom in sorted(candidates, key=lambda candidate: -candidate[0]):
        if all((geom & old).sum() < .72 * (geom | old).sum() for old in pool):
            pool.append(geom)
        if len(pool) == 25:
            break
    if len(pool) < 2:
        return None

    def fidelity(union):
        hit = fg[union].sum()
        return (hit - 1.45 * (union.sum()-hit)) / total

    best_single = max(fidelity(candidate) for candidate in pool)
    if best_single > .90:
        return None
    best = best_single + .17
    placed = None
    for i, a in enumerate(pool):
        for j, b in enumerate(pool[i+1:], i+1):
            if (a & b).sum() > .47 * min(a.sum(), b.sum()):
                continue
            joined = a | b
            value = fidelity(joined)
            if value > best:
                placed, best = [a, b], value
            if value < .69:
                continue
            for c in pool[j+1:]:
                if (c & joined).sum() > .43 * c.sum():
                    continue
                improved = fidelity(joined | c) - .12
                if improved > best:
                    placed, best = [a, b, c], improved
    if placed is None or best < .78:
        return None
    parts = []
    for region in placed:
        region = cv2.resize(region.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(region & mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        parts.append(max(contours, key=cv2.contourArea))
    return parts
