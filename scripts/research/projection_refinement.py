"""Fit visible projections of primitive objects with explicit foreground occlusion."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path.cwd()))
from src.pixel_regions import HEIGHT, WIDTH, read_rgb
from src.primitive_regions import SIZE, templates
from src.scene_posterior import sample_scene_scores
from src.models.slot_program import EXCLUSIVE_GROUPS, WORD_INDEX, parse_program
from src.text_aug import canonical_form
from src.twins import find_twins
from occlusion_splits import fit_two


ROOT = Path("artefacts/research")
VARIANTS = ("control", "center", "center_shape", "split")
VIEWS = (0, 3, 7, 10, 14, 17, 20, 24, 28, 30, 33, 35, 37, 39)
GROUPS = [np.array([WORD_INDEX[word] for word in group]) for group in EXCLUSIVE_GROUPS]


def patch_masks(features):
    boxes = np.rint(features[:, 8:12] * [WIDTH, HEIGHT, WIDTH, HEIGHT]).astype(int)
    masks = np.zeros((len(boxes), HEIGHT, WIDTH), np.float32)
    for i, (x, y, w, h) in enumerate(boxes):
        masks[i, y:y+h, x:x+w] = cv2.resize(features[i, 47:147].reshape(10, 10), (w, h))
    return boxes, masks


def fit_projection(target, front, box, shape_prior):
    """Optimize center, scale, primitive class and discrete camera-relative yaw."""
    x, y, w, h = box
    delta = max(6, round(.45 * max(w, h)))
    left, top = max(0, x-delta), max(0, y-delta)
    right, bottom = min(WIDTH, x+w+delta), min(HEIGHT, y+h+delta)
    view = np.s_[top:bottom, left:right]
    observed = target[view].copy()
    occluded = ((1-observed) * front[view]).copy()
    true_area = observed.sum()
    if true_area < 13:
        return np.zeros(2), np.zeros(3)
    bank, _, ratio, kind = templates()
    aspect = np.exp(ratio)
    score_by_shape = np.full(3, -.2)
    best = -.2
    shift = np.zeros(2)
    for i in VIEWS:
        if max(w / h / aspect[i], aspect[i] / (w / h)) > 2.7:
            continue
        extent = max(h, w/aspect[i])
        for scale in (.88, 1., 1.15, 1.34):
            ph, pw = round(extent * scale), round(extent * aspect[i] * scale)
            if min(ph,pw) < 4 or ph > len(observed) or pw > observed.shape[1]:
                continue
            proj = cv2.resize(bank[i].reshape(SIZE, SIZE), (pw, ph), interpolation=cv2.INTER_LINEAR)
            hit = cv2.matchTemplate(observed, proj, cv2.TM_CCORR).clip(0)
            hidden = cv2.matchTemplate(occluded, proj, cv2.TM_CCORR).clip(0)
            total = proj.sum()
            leak = (total - hit - hidden).clip(0)
            error = (true_area - hit).clip(0) + leak
            # Penalize speculative growth supported almost entirely by an occluder.
            faith = 1-error/(true_area + leak) - .09*hidden/total - .055*abs(np.log(scale))
            grid_y, grid_x = np.ogrid[:len(faith), :faith.shape[1]]
            move_x = left + grid_x + .5*pw - (x + .5*w)
            move_y = top + grid_y + .88*ph - (y + .88*h)
            faith -= .052 * ((move_x/w)**2 + (move_y/h)**2)
            _, value, _, (sx, sy) = cv2.minMaxLoc(faith)
            label = kind[i]
            score_by_shape[label] = max(score_by_shape[label], value)
            weighted = value + .10*np.log(max(.02, float(shape_prior[label])))
            if weighted > best:
                best = weighted
                shift = np.array([left + sx + .5*pw - (x + .5*w),
                                  top + sy + .88*ph - (y + .88*h)])
    if score_by_shape.max() < .48:
        return np.zeros(2), np.zeros(3)
    return shift, score_by_shape


def refine(features, p, xy, color, shape):
    boxes, masks = patch_masks(features)
    centers = xy.copy()
    shapes = shape.copy()
    bottom = boxes[:, 1] + .88*boxes[:, 3]
    hue = color.argmax(1)
    for i, box in enumerate(boxes):
        if p[i] < .28:
            continue
        occluders = (p > .76) & (bottom > bottom[i]+2) & (hue != hue[i])
        if not occluders.any():
            continue
        front = masks[occluders].max(0)
        x, y, w, h = box
        # Skip objects without an observed color boundary/contact where another
        # foreground object could hide a continuation.
        border = front[max(0,y-4):min(HEIGHT,y+h+5), max(0,x-4):min(WIDTH,x+w+5)]
        if border.max(initial=0) < .5:
            continue
        delta, fit = fit_projection(masks[i], front, box, shape[i])
        centers[i] += .7 * delta * [2/WIDTH, 2/HEIGHT]
        if fit.max() > .62:
            shapes[i] = shape[i] * np.exp(2 * (fit-fit.max()).clip(-.5, 0))
            shapes[i] /= shapes[i].sum()
    return centers, shapes


def split_regions(image, features, probability, xy, attributes):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.magnitude(dx, dy)
    boxes, _ = patch_masks(features)
    positions = list(xy)
    presence = list(probability)
    result = [list(attr) for attr in attributes]
    for i, (x, y, w, h) in enumerate(boxes):
        if probability[i] < .87 or attributes[1][i,0] > .5:
            continue
        visible = cv2.resize(features[i,47:147].reshape(10,10), (w,h)) > .5
        fit = fit_two(visible, edges[y:y+h,x:x+w])
        if fit is None:
            continue
        parts, gain = fit
        main, extra = sorted(parts, key=lambda item: -(
            np.log(.12 + attributes[-1][i, item[1]]) + .40*np.log(item[0][2]*item[0][3])
        ))

        def position(view):
            a, b, width, height = view
            return np.array([x+a+.5*width, y+b+.88*height]) / [WIDTH,HEIGHT]*2-1

        view, kind = main
        positions[i] = .75*position(view) + .25*xy[i]
        result[-1][i] = .7*attributes[-1][i] + .3*np.eye(3)[kind]
        view, kind = extra
        positions.append(position(view))
        presence.append(np.clip(.36 + 1.2*(gain-.115), .36, .68))
        large = attributes[0][i,1] * min(1., 1.6*view[2]*view[3]/(w*h))
        result[0].append(np.array([1-large,large]))
        result[1].append(attributes[1][i])
        result[2].append(.55*attributes[2][i] + .45*np.array([.5,.5]))
        result[3].append(.14*attributes[-1][i] + .86*np.eye(3)[kind])
    return np.array(presence), np.asarray(positions), [np.asarray(group) for group in result]


def run(variant):
    rows = pd.read_csv("data/val.csv")
    with np.load(ROOT / f"scene_{variant}.npz") as source:
        assert np.array_equal(source["ids"], rows.id)
        table = dict(source)
    indices = rows.groupby("image", sort=False).indices
    captions = [canonical_form(r.caption, r.category) for r in rows.itertuples()]
    programs = [parse_program(c) for c in captions]
    seeds = list(indices.values())
    output = {name: np.empty(len(rows)) for name in VARIANTS}

    def process(item):
        i, ids = item
        valid = table["valid"][i]
        positions, p = (table[field][i, valid] for field in ("xy", "p"))
        probabilities = [table[field][i, valid] for field in ("size", "color", "material", "shape")]
        subset = rows.iloc[ids]
        pairs = find_twins(subset.image, [captions[j] for j in ids], subset.category)
        parsed = [programs[j] for j in ids]
        features = table["features"][i, valid]
        xy, shape = refine(features, p, positions, probabilities[1], probabilities[3])
        image = read_rgb(Path("data/val") / subset.image.iat[0])
        split = split_regions(image, features, p, positions, probabilities)
        cases = ((p, positions, probabilities), (p, xy, probabilities), (p, xy, [*probabilities[:3], shape]), split)
        scores = [sample_scene_scores(parsed, pairs, q, list(zip(GROUPS, a)), c, table["offsets"], 42)
                  for q, c, a in cases]
        return ids, scores

    cv2.setNumThreads(1)
    with ThreadPoolExecutor(4) as pool:
        for ids, scores in pool.map(process, enumerate(seeds)):
            for key, value in zip(VARIANTS, scores):
                output[key][ids] = value
    for name, values in output.items():
        losses = {category: round(roc_auc_score(group.label, values[group.index]), 6)
                  for category, group in rows.groupby("category")}
        print(variant, name, roc_auc_score(rows.label, values), losses, flush=True)
    np.savez(ROOT / f"projection_{variant}.npz", ids=rows.id, **output)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "crop")
