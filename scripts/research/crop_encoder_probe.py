"""Use the same crop-count head with HOG or frozen SigLIP2 features."""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path.cwd()))
from native_centers import read_rows
from overlap_review import COLORS
from src.pixel_regions import COLORS as REGION_COLORS
from src.pixel_regions import _color_planes, fit_background, read_rgb
from src.region_shape_classifier import oriented_edges

ROOT = Path("artefacts/research")
DATA = Path("data")
LABELS = ROOT / "overlap_review/review_labels.csv"


def read_cases():
    cases = pd.read_csv(LABELS)
    controls = ROOT / "overlap_controls/review_labels.csv"
    if controls.is_file():
        sampled = pd.read_csv(controls)
        train_controls = ROOT / "overlap_train_controls/review_labels.csv"
        if train_controls.is_file():
            sampled = pd.concat([pd.read_csv(train_controls), sampled[sampled.split=="val"]])
        cases = pd.concat([cases, sampled], ignore_index=True)
    return cases.dropna(subset=["number"]).reset_index(drop=True)


def load_regions(frame):
    background = fit_background(DATA / "train", list(read_rows("train")[1]))
    local, colors, pixels = [], [], []
    for row in frame.itertuples(index=False):
        rgb = read_rgb(DATA / row.split / row.image)
        planes, _ = _color_planes(rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), background)
        cut = np.s_[row.y0:row.y1, row.x0:row.x1]
        crop = rgb[cut]
        pixels.append(Image.fromarray(crop))
        mask = planes[REGION_COLORS.index(row.color)][cut].astype(np.uint8)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
        normalized = np.clip((gray-gray.mean())*42/max(8., gray.std())+128, 0, 255)
        hog = oriented_edges(cv2.resize(normalized.astype(np.uint8), (24, 24)))
        descriptor = cv2.resize(mask.astype(np.float32), (10, 10), interpolation=cv2.INTER_AREA)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        sizes = sorted(stats[1:, cv2.CC_STAT_AREA] / mask.size, reverse=True)[:3]
        sizes += [0.] * (3-len(sizes))
        local.append(np.r_[hog, descriptor.ravel(), descriptor.mean(0), descriptor.mean(1),
                          sizes, min(8, n-1), np.log(crop.shape[1]/crop.shape[0]), mask.mean()])
        colors.append(COLORS.index(row.color))
    return np.stack(local), np.eye(len(COLORS))[colors], pixels


def contour_counts(frame):
    values = np.zeros(len(frame), dtype=int)
    for split, rows in frame.groupby("split"):
        cache = ROOT / ("teacher_train.npz" if split=="train" else "scene_silhouette.npz")
        with np.load(cache) as data:
            names = data["names"] if split=="train" else list(read_rows("val")[1])
            img = {name:i for i,name in enumerate(names)}
            features, valid, p, color = (data[k] for k in ("features","valid","p","color"))
        for row in rows.itertuples():
            i = img[row.image]
            x, y = ((features[i, :, 8:10]+.5*features[i, :, 10:12])*[240,160]).T
            matching = valid[i] & (p[i]>.55) & (color[i].argmax(1)==COLORS.index(row.color))
            crop = (x>=row.x0) & (y>=row.y0) & (x<row.x1) & (y<row.y1)
            values[row.Index] = (matching & crop).sum()
    return values


def same_head(x, training, test, supervision, truth, hue=None):
    estimator = make_pipeline(StandardScaler(),
                              LogisticRegression(C=.1, solver="liblinear", class_weight="balanced"))
    estimator.fit(x[training], supervision[training])
    prediction = estimator.predict_proba(x[test])[:,1]
    result = {"auc": float(roc_auc_score(truth[test], prediction)),
            "accuracy": float(accuracy_score(truth[test], prediction>=.5)),
            "balanced_accuracy": float(balanced_accuracy_score(truth[test], prediction>=.5))}
    if hue is not None:
        result["within_color"] = {}
        for color in np.unique(hue[test]):
            include = hue[test] == color
            y = truth[test][include]
            if np.unique(y).size == 2:
                result["within_color"][color] = float(roc_auc_score(y, prediction[include]))
        result["within_color_macro_auc"] = float(np.mean(list(result["within_color"].values())))
    return result


def siglip_features(model_dir, images, keys, batch_size, cache):
    known = {}
    if cache.is_file():
        with np.load(cache) as saved:
            if list(saved["ids"]) == keys:
                return saved["embeddings"], None
            known = dict(zip(saved["ids"],saved["embeddings"], strict=True))
    pending = [i for i,key in enumerate(keys) if key not in known]
    if not pending:
        return np.stack([known[key] for key in keys]), None
    import torch
    from transformers import AutoImageProcessor, SiglipVisionModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    started = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    vision = SiglipVisionModel.from_pretrained(model_dir, local_files_only=True).eval().to(device)
    print("SigLIP2 vision load seconds", time.perf_counter()-started, device, flush=True)
    sizes, times = [], []
    with torch.inference_mode():
        for start in range(0, len(pending), batch_size):
            subset = pending[start:start+batch_size]
            batch = [images[i] for i in subset]
            pixels = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
            tick = time.perf_counter()
            pooled = vision(pixel_values=pixels).pooler_output.cpu().numpy()
            if len(pooled) != len(batch) or not np.isfinite(pooled).all():
                raise ValueError("Invalid vision embeddings")
            times.append(time.perf_counter()-tick)
            sizes.append(len(batch))
            known.update((keys[i],feature) for i,feature in zip(subset,pooled,strict=True))
            if len(times) in (1, 4) or len(times)%16==0 or start+len(batch)==len(pending):
                print("SigLIP2 crops", sum(sizes), "encoder crops/sec", sum(sizes)/sum(times), flush=True)
    output = np.stack([known[key] for key in keys])
    np.savez(cache, ids=keys, embeddings=output)
    return output, float(sum(sizes)/sum(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--benchmark", type=int, default=0)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    frame = read_cases()
    if frame.duplicated("image").any() or set(frame.split) != {"train", "val"}:
        raise ValueError("Each reviewed image must belong to a single split")
    x, color, images = load_regions(frame)
    truth = frame.number.to_numpy()==2
    training = (frame.split=="train").to_numpy()
    validation = ~training
    proxy_count = contour_counts(frame)
    proxy = proxy_count>=2
    hue = frame.color.to_numpy()
    features = np.column_stack((x,color))
    report = {
        "reviewed_train": int(training.sum()), "reviewed_val": int(validation.sum()),
        "contour_exact_count_val": float((proxy_count[validation]==frame.number[validation]).mean()),
        "contour_mean_count_val": float(proxy_count[validation].mean()),
        "color_only_reviewed": same_head(color, training, validation, truth, truth, hue),
        "hog_reviewed": same_head(features, training, validation, truth, truth, hue),
        "hog_proxy": same_head(features, training, validation, proxy, truth, hue),
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.model is not None:
        limit = args.benchmark or len(images)
        keys = [f"{r.split}/{r.image}:{r.x0},{r.y0},{r.x1},{r.y1}" for r in frame.itertuples()][:limit]
        cache = ROOT / (f"siglip2_crops_bench{limit}.npz" if args.benchmark else "siglip2_reviewed_crops.npz")
        embedding, rate = siglip_features(args.model, images[:limit], keys, args.batch_size, cache)
        report["siglip2_crops_per_second"] = rate
        if not args.benchmark:
            report["siglip2_reviewed"] = same_head(np.c_[embedding,color], training, validation, truth, truth, hue)
            print(json.dumps(report, indent=2), flush=True)
    path = ROOT / ("siglip2_crop_benchmark.json" if args.benchmark else "crop_encoder_probe.json")
    path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
