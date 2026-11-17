"""Learn object shape from oriented edges in training crops."""

from catboost import CatBoostClassifier
import cv2
import numpy as np

from src.pixel_regions import COLORS, HEIGHT, WIDTH, read_rgb


PATCH_SIZE = 24
CELL_SIZE = 6
CELL = (np.arange(PATCH_SIZE)[:, None] // CELL_SIZE) * 4 + (np.arange(PATCH_SIZE)[None, :] // CELL_SIZE)


def oriented_edges(patch):
    dx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(dx, dy)
    bins = np.floor(angle * (8 / np.pi)).astype(int) % 8
    hist = np.bincount((8 * CELL + bins).ravel(), weights=magnitude.ravel(), minlength=128).reshape(16, 8)
    return (hist / (hist.sum(1, keepdims=True) + 18)).ravel()


def descriptors(folder, names, features, valid):
    rows = []
    for name, scene, active in zip(names, features, valid):
        gray = cv2.cvtColor(read_rgb(folder / name), cv2.COLOR_RGB2GRAY)
        for row in scene[active]:
            box = row[len(COLORS):len(COLORS) + 4]
            x, y, width, height = np.rint(box * [WIDTH, HEIGHT, WIDTH, HEIGHT]).astype(int)
            crop = gray[max(0, y - 1):min(HEIGHT, y + height + 1),
                        max(0, x - 1):min(WIDTH, x + width + 1)].astype(np.float32)
            image = np.clip((crop - crop.mean()) * 42 / max(8., crop.std()) + 128, 0, 255).astype(np.uint8)
            patch = cv2.resize(image, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_LINEAR)
            mask = row[47:147].reshape(10, 10)
            rows.append(np.concatenate((row[10:17], row[47:147], mask.mean(0), mask.mean(1),
                                        oriented_edges(patch))))
    return np.stack(rows)


class RegionShapeClassifier:
    def __init__(self, seed=420, trees=360):
        self.classifier = CatBoostClassifier(iterations=trees, depth=7, learning_rate=.1, thread_count=3,
                                             random_seed=seed, verbose=False, allow_writing_files=False)

    def augment(self, folder, names, arrays, anchors=None):
        features, positions, valid = arrays
        rich = descriptors(folder, names, features, valid)
        if anchors is not None:
            targets = anchors.numpy()[..., -1][valid]
            anchored = targets >= 0
            if np.unique(targets[anchored]).size != 3:
                raise ValueError("Training crops must cover cube, cylinder and sphere")
            print(f"Training CatBoost shape classifier on {anchored.sum()} caption-derived crop labels", flush=True)
            self.classifier.fit(rich[anchored], targets[anchored])
        shape = np.zeros((*valid.shape, 3), np.float32)
        shape[valid] = self.classifier.predict_proba(rich, thread_count=3)
        return np.concatenate((features, shape), axis=-1), positions, valid
