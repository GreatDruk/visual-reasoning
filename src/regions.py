"""Extract object proposals from native pixels using color and foreground contours."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

COLORS = ("gray", "red", "brown", "yellow", "green", "cyan", "blue", "purple")
MAX_REGIONS = 24
HEIGHT, WIDTH = 160, 240


def read_rgb(path):
    # imdecode also accepts local Windows paths containing non-ASCII characters.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape != (HEIGHT, WIDTH, 3):
        raise ValueError(f"Expected a {WIDTH}x{HEIGHT} JPEG: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def fit_background(folder, names, seed=42):
    """The position-wise training median estimates normal background illumination."""
    selected = np.random.default_rng(seed).choice(names, min(128, len(names)), replace=False)
    with ThreadPoolExecutor(4) as pool:
        images = list(pool.map(read_rgb, (Path(folder) / name for name in selected)))
    return np.median(images, axis=0).astype(np.float32)


def _color_planes(image, hsv, background):
    hue, sat, value = np.moveaxis(hsv, -1, 0)
    # The gray floor has little chroma. Hue is informative only above JPEG noise.
    chroma = np.ptp(image.astype(np.int16), axis=-1)
    colored = (sat > 55) & (chroma >= 17) & (value > 24)
    # OpenCV hue is measured in half-degrees. These intervals describe colors in
    # the training images; shape, material, size and color readouts are learned.
    limits = ((0, 8), (8, 21), (21, 39), (39, 80), (80, 104), (104, 129), (129, 175))
    planes = [colored & (hue >= lo) & (hue < hi) for lo, hi in limits]
    planes[0] |= colored & (hue >= 175)

    delta = image.astype(np.float32).mean(-1) - background.mean(-1)
    border = np.concatenate((delta[:, :8], delta[:, -8:]), axis=1)
    delta -= np.median(border, axis=1)[:, None]
    gray = (np.abs(delta) >= 12) & (chroma <= 22)
    gray &= cv2.dilate(colored.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0
    return [gray, *planes], delta


def _contours(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _separate_touching(contour):
    """Split clearly necked silhouettes at a gap between distance-transform cores."""
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    if area < 115 or max(w, h) < 18:
        return [contour]
    hull = cv2.contourArea(cv2.convexHull(contour))
    if area > 0.94 * hull:
        return [contour]
    origin = np.array([x - 1, y - 1], dtype=np.int32)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.drawContours(mask, [contour - origin], 0, 1, cv2.FILLED)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    for ratio in (0.48, 0.60):
        threshold = max(3.0, distance.max() * ratio)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(
            (distance >= threshold).astype(np.uint8)
        )
        cores = [
            j
            for j in range(1, count)
            if stats[j, cv2.CC_STAT_AREA] >= 8 and distance[labels == j].max() >= threshold + 0.8
        ]
        if not 2 <= len(cores) <= 4:
            continue
        yy, xx = np.nonzero(mask)
        pixels = np.column_stack((xx, yy))
        center = centers[cores]
        radii = np.array([distance[labels == j].max() for j in cores])
        assignment = (np.square(pixels[:, None] - center).sum(-1) / radii).argmin(-1)
        parts = []
        for j in range(len(cores)):
            part = np.zeros_like(mask)
            selected = assignment == j
            part[yy[selected], xx[selected]] = 1
            cs, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cs:
                largest = max(cs, key=cv2.contourArea)
                if cv2.contourArea(largest) >= 20:
                    parts.append(largest + origin)
        if len(parts) == len(cores):
            return parts
    return [contour]


def _edge_contours(gray, planes):
    smooth = cv2.GaussianBlur(gray, (3, 3), 0.7)
    edges = cv2.Canny(smooth, 12, 32)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    colored = np.logical_or.reduce(planes[1:])
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if not (5 <= w <= 69 and 5 <= h <= 77 and area >= max(23, 0.28 * w * h)):
            continue
        inside = np.zeros((h, w), np.uint8)
        cv2.drawContours(inside, [contour - np.array([x, y], dtype=np.int32)], 0, 1, cv2.FILLED)
        if colored[y : y + h, x : x + w][inside > 0].mean() <= 0.18:
            candidates.append(contour)
    return candidates


def _box_iou(left, right):
    x, y, w, h = left
    a, b, c, d = right
    intersection = max(0, min(x + w, a + c) - max(x, a)) * max(0, min(y + h, b + d) - max(y, b))
    return intersection / (w * h + c * d - intersection)


def _descriptor(rgb, hsv, delta, contour, color, gradient=None, *, min_area=19):
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    if area < min_area or min(w, h) < 4:
        return None
    contour = contour - np.array([x, y], dtype=np.int32)
    region = np.zeros((h, w), np.uint8)
    cv2.drawContours(region, [contour], 0, 1, cv2.FILLED)
    inside = region.astype(bool)
    local = np.s_[y : y + h, x : x + w]
    rgb_crop = rgb[local]
    color_crop = hsv[local]
    color_values = np.column_stack((rgb_crop[inside], color_crop[inside, 1:], delta[local][inside]))
    quantiles = np.percentile(color_values, (10, 30, 50, 70, 90), axis=0).ravel() / 255
    area_hull = cv2.contourArea(cv2.convexHull(contour))
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
    # An explicit normalized silhouette and a masked RGB thumbnail preserve
    # roundness, corners, occlusion and highlights for the learned readout.
    shape = cv2.resize(region.astype(np.float32), (10, 10), interpolation=cv2.INTER_AREA)
    thumbnail = (
        cv2.resize(
            (rgb_crop * region[..., None]).astype(np.float32), (8, 8), interpolation=cv2.INTER_AREA
        )
        / 255
    )
    location = np.array([(x + 0.5 * w) / WIDTH, (y + 0.88 * h) / HEIGHT], dtype=np.float32) * 2 - 1
    geometry = [
        x / WIDTH,
        y / HEIGHT,
        w / WIDTH,
        h / HEIGHT,
        w / h,
        area / (w * h),
        area / max(1, area_hull),
        4 * np.pi * area / perimeter**2,
        len(approx) / 10,
    ]
    values = np.concatenate(
        [np.eye(len(COLORS))[color], geometry, quantiles, shape.ravel(), thumbnail.ravel()]
    )
    if gradient is not None:
        boundary = region != cv2.erode(
            region, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0
        )
        response = gradient[local][boundary]
        # Sharp continuous boundaries are a useful cue for excluding diffuse shadows.
        edge = np.concatenate(
            (
                np.percentile(response, [25, 50, 75, 90]) / 128,
                [np.mean(response > t) for t in (12, 24, 40, 64)],
            )
        )
        values = np.concatenate((values, edge))
    return area, location, values


FEATURES = len(COLORS) + 9 + 30 + 100 + 192
BOUNDARY_FEATURES = FEATURES + 8
PRIMITIVE_FEATURES = BOUNDARY_FEATURES + 3


def _fragment_contours(rgb, delta, planes, gradient, boxes):
    """Propose small visible foreground fragments that morphology can erase."""
    chroma = np.ptp(rgb.astype(np.int16), axis=-1)
    masks = [np.abs(delta) >= 26, *planes[1:]]
    masks[0] &= (chroma <= 22) & ~np.logical_or.reduce(planes[1:])
    # Strong gray edges can isolate the dark cap of a specular object from a
    # large, similarly dark but smooth shadow.
    dark_edges = (delta <= -15) & (chroma <= 20) & (gradient >= 45)
    candidates = []
    for color, mask in [*enumerate(masks), (0, dark_edges)]:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if not (
                8 <= area <= 180 and min(w, h) >= 4 and max(w, h) <= 24 and area >= 0.23 * w * h
            ):
                continue
            if gradient[y : y + h, x : x + w].mean() < 35:
                continue
            candidates.append((area, color, contour, (x, y, w, h)))
    for _, color, contour, box in sorted(candidates, key=lambda item: -item[0]):
        x, y, w, h = box
        if any(
            max(0, min(x + w, a + ow) - max(x, a)) * max(0, min(y + h, b + oh) - max(y, b))
            >= 0.65 * w * h
            for a, b, ow, oh in boxes
        ):
            continue
        boxes.append(box)
        yield color, contour


def extract_image(path, background, boundary=False, primitives=False, fragments=False, seams=False):
    boundary = boundary or primitives or fragments or seams
    rgb = read_rgb(path)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    planes, delta = _color_planes(rgb, hsv, background)
    gradient = None
    if boundary:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.dilate(cv2.magnitude(gx, gy), np.ones((3, 3), np.uint8))
    if primitives:
        from .primitives import contour_shape_scores, merge_reflected_faces, split_at_edges

        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0.55), 10, 26)
    if seams:
        from .primitives import split_at_luminance_seam
    proposals = []
    boxes = []
    by_color = [_contours(plane) for plane in planes]
    if primitives:
        by_color = merge_reflected_faces(by_color, planes)
    for color, contours in enumerate(by_color):
        if boundary and color == 0:
            contours = [c for c in contours if max(cv2.boundingRect(c)[2:]) <= 82]
            for candidate in sorted(
                _edge_contours(gray, planes), key=cv2.contourArea, reverse=True
            ):
                box = cv2.boundingRect(candidate)
                if all(_box_iou(box, cv2.boundingRect(c)) < 0.65 for c in contours):
                    contours.append(candidate)
        if boundary and color != 0:
            separated = []
            for contour in contours:
                parts = _separate_touching(contour)
                if seams and len(parts) == 1:
                    parts = split_at_luminance_seam(contour, gray)
                if primitives and len(parts) == 1:
                    parts = split_at_edges(contour, edges)
                separated.extend(parts)
            contours = separated
        for contour in contours:
            desc = _descriptor(rgb, hsv, delta, contour, color, gradient)
            if desc is not None:
                if primitives:
                    area, location, values = desc
                    desc = (area, location, np.concatenate((values, contour_shape_scores(contour))))
                proposals.append(desc)
                if fragments:
                    boxes.append(cv2.boundingRect(contour))
    extra = []
    if fragments:
        for color, contour in _fragment_contours(rgb, delta, planes, gradient, boxes):
            area, location, values = _descriptor(
                rgb, hsv, delta, contour, color, gradient, min_area=8
            )
            if primitives:
                values = np.concatenate((values, contour_shape_scores(contour)))
            extra.append((area, location, values))
    proposals.sort(key=lambda proposal: -proposal[0])
    if extra:
        # Keep room for the strongest fragments even when shadow and specular
        # contours fill the regular proposal budget.
        extra = extra[:3]
        proposals = proposals[: MAX_REGIONS - len(extra)] + extra
    width = PRIMITIVE_FEATURES if primitives else BOUNDARY_FEATURES if boundary else FEATURES
    features = np.zeros((MAX_REGIONS, width), np.float32)
    locations = np.zeros((MAX_REGIONS, 2), np.float32)
    valid = np.zeros(MAX_REGIONS, bool)
    for i, (_, location, vector) in enumerate(proposals[:MAX_REGIONS]):
        features[i] = vector
        locations[i] = location
        valid[i] = True
    return features, locations, valid


def extract_split(
    folder,
    names,
    background,
    workers=4,
    boundary=False,
    primitives=False,
    fragments=False,
    seams=False,
):
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(workers) as pool:
        items = list(
            pool.map(
                lambda name: extract_image(
                    Path(folder) / name, background, boundary, primitives, fragments, seams
                ),
                names,
            )
        )
    return tuple(np.stack(column) for column in zip(*items))
