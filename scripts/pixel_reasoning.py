"""Fit a caption-supervised object table on native-image region proposals."""

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import scipy
from scipy.special import expit
import sklearn
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F

from src.models.slot_program import (
    AND, EXACT, EXIST, EXCLUSIVE_GROUPS, FEWER, MORE, NONE, WORD_INDEX,
    SlotProgramExecutor, encode_program_inputs, log1mexp, parse_program,
)
from src.pixel_regions import BOUNDARY_FEATURES, COLORS, FEATURES, PRIMITIVE_FEATURES, extract_split, fit_background
from src.scene_posterior import sample_scene_scores
from src.text_aug import canonical_form
from src.twins import find_twins


class ImageTable(SlotProgramExecutor):
    """Predict objectness and mutually exclusive attributes for each visible region."""

    def __init__(self, boundary=False, primitives=False):
        width = PRIMITIVE_FEATURES if primitives else BOUNDARY_FEATURES if boundary else FEATURES
        super().__init__(1, width + (4 if boundary else 0), dim=80, hidden=160, dropout=.08, relation_sharpness=16.,
                         use_slot_vectors=False, reference_mode="marginal", exclusive_attributes=True,
                         require_reference_exists=True, comparison_mode="coupled")
        self.boundary = boundary
        self.primitives = primitives
        self.color_strength = nn.Parameter(torch.tensor(2.))
        self.project_position = nn.Linear(width, 2)
        nn.init.zeros_(self.project_position.weight)
        nn.init.zeros_(self.project_position.bias)
        self.register_buffer("color_index", torch.tensor([WORD_INDEX[c] for c in COLORS]))
        if primitives:
            self.shape_strength = nn.Parameter(torch.tensor(4.))
            self.register_buffer("shape_index", torch.tensor([WORD_INDEX[w] for w in EXCLUSIVE_GROUPS[-1]]))

    def slot_logits(self, enc):
        if self.boundary:
            x = enc["pooled"]
            box = x[..., len(COLORS):len(COLORS)+4]
            lower, upper = box[..., :2], box[..., :2] + box[..., 2:]
            intersect = (torch.minimum(upper[:, :, None], upper[:, None, :])
                         - torch.maximum(lower[:, :, None], lower[:, None, :])).clamp_min(0).prod(-1)
            area = box[..., 2:].prod(-1)
            union = area[:, :, None] + area[:, None, :] - intersect
            other = enc["valid"][:, None] & ~torch.eye(x.shape[1], device=x.device, dtype=torch.bool)
            overlap = (intersect / union.clamp_min(1e-6)) * other
            same_color = (x[..., :len(COLORS)] @ x[..., :len(COLORS)].transpose(1, 2)) > .5
            context = torch.stack([overlap.amax(-1), (overlap * same_color).amax(-1),
                                   (intersect / area[:, :, None].clamp_min(1e-6) * other).amax(-1),
                                   (overlap > .12).sum(-1).float() / 4], -1)
            enc = {**enc, "pooled": torch.cat((x, context), -1)}
        obj, words = super().slot_logits(enc)
        chroma = enc["pooled"][..., :len(COLORS)]
        prior = torch.zeros_like(words).index_copy(-1, self.color_index, chroma * self.color_strength)
        if self.primitives:
            shape = enc["pooled"][..., PRIMITIVE_FEATURES-3:PRIMITIVE_FEATURES]
            prior = prior.index_copy(-1, self.shape_index, shape * self.shape_strength)
        return obj.masked_fill(~enc["valid"], -30.), words + prior

    def encode(self, features, positions, valid):
        area = features[..., len(COLORS) + 2] * features[..., len(COLORS) + 3]
        correction = .14 * self.project_position(features).tanh()
        return {"pooled": features, "position": positions + correction, "valid": valid, "area": area}

    def score(self, features, positions, valid, image_index, program):
        enc = self.encode(features, positions, valid)
        truth, _, _ = self(enc, image_index, program)
        return truth - log1mexp(truth)

    def score_and_count_loss(self, features, positions, valid, image_index, program, target):
        enc = self.encode(features, positions, valid)
        truth, counts, _ = self(enc, image_index, program)
        support = torch.arange(counts.shape[-1], device=counts.device)
        expected = (counts.exp() * support).sum(-1)
        exact = (program["ops"] == EXACT) & (target > 0)
        deviation = F.smooth_l1_loss(expected, program["numbers"].float(), reduction="none")
        count_loss = deviation.masked_fill(~exact, 0).sum() / exact.sum().clamp_min(1)
        return truth - log1mexp(truth), count_loss

    def existence_loss(self, features, positions, valid, image_index, program):
        area = features[..., len(COLORS) + 2] * features[..., len(COLORS) + 3]
        obj, words = self.slot_logits({"pooled": features, "area": area, "valid": valid})
        per_region = self.membership(obj[image_index], words[image_index], program["a"], exclusive=True)
        return -log1mexp(log1mexp(per_region).sum(-1)).mean()

    def anchor_loss(self, features, positions, valid, anchors):
        obj, words = self.slot_logits(self.encode(features, positions, valid))
        active = anchors.ge(0).any(-1)
        obj_loss = F.softplus(-obj).masked_fill(~active, 0).sum() / active.sum().clamp_min(1)
        total = obj_loss
        for j, group in enumerate(EXCLUSIVE_GROUPS):
            indices = [WORD_INDEX[w] for w in group]
            scores = words[..., indices].reshape(-1, len(indices))
            targets = anchors[..., j].flatten()
            total = total + F.cross_entropy(scores, targets, ignore_index=-100, reduction="sum") / targets.ge(0).sum().clamp_min(1)
        return total / (1 + len(EXCLUSIVE_GROUPS))


def read_rows(root, split):
    with (root / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def implied_filters(program, truth):
    """Project true training statements onto valid simpler existential supervision."""
    if program.relation:
        return [(program.a, 1)] if truth and (program.op == EXIST or
                (program.op == EXACT and program.number > 0)) else []
    if program.op == EXIST:
        return [(program.a, truth)]
    if program.op == NONE:
        return [(program.a, 1 - truth)]
    if not truth:
        return []
    if program.op == EXACT:
        return [(program.a, int(program.number > 0))]
    if program.op == AND:
        return [(program.a, 1), (program.b, 1)]
    if program.op in (MORE, FEWER):
        return [(program.a if program.op == MORE else program.b, 1)]
    return []


def region_anchors(features, positions, valid, programs, images, targets):
    """Derive crop labels from train captions with a unique plausible color match."""
    families = {word: (j, k) for j, group in enumerate(EXCLUSIVE_GROUPS) for k, word in enumerate(group)}
    colors = features[..., :len(COLORS)].argmax(-1)
    width = features[..., len(COLORS) + 2]
    height = features[..., len(COLORS) + 3]
    plausible = valid & (width * height >= 52 / (240 * 160))
    votes = defaultdict(set)
    for program, scene, truth in zip(programs, images, targets):
        if not truth:
            continue
        facts = [(words, 1, 0) for words, y in implied_filters(program, truth) if words and y]
        if program.op == EXACT and program.number > 0 and not program.relation:
            facts = [(program.a, program.number, 0)]
        elif program.relation in (5, 6) and program.op == EXIST:
            facts = [(program.a, 1, program.relation)]
        elif 1 <= program.relation <= 4:
            facts.append((program.ref, 1, 0))
        for words, count, side in facts:
            hue = next((COLORS.index(w) for w in words if w in COLORS and w != "gray"), None)
            if hue is None:
                continue
            candidates = plausible[scene] & (colors[scene] == hue)
            if side:
                candidates &= positions[scene, :, 0] * (-1 if side == 5 else 1) > .03
            regions = np.flatnonzero(candidates)
            if len(regions) == count:
                for region in regions:
                    for word in words:
                        family, value = families[word]
                        votes[scene, region, family].add(value)
    result = np.full((*valid.shape, len(EXCLUSIVE_GROUPS)), -100, dtype=np.int64)
    for key, values in votes.items():
        if len(values) == 1:
            result[key] = next(iter(values))
    return torch.from_numpy(result)


class SceneData:
    def __init__(self, root, split, background, device, *, fitting=False, cache_dir=None,
                 boundary=False, anchors=False, primitives=False, shape_classifier=None, fragments=False, seams=False):
        rows = read_rows(root, split)
        self.ids = np.array([int(r["id"]) for r in rows])
        names = list(dict.fromkeys(r["image"] for r in rows))
        positions = {name: index for index, name in enumerate(names)}
        caption = [canonical_form(r["caption"], r["category"]) for r in rows]
        self.parsed = [parse_program(c) for c in caption]
        self.categories = np.array([r["category"] for r in rows])
        self.images = np.array([positions[r["image"]] for r in rows])
        by_image = defaultdict(list)
        for i, scene in enumerate(self.images):
            by_image[scene].append(i)
        self.rows = [np.array(by_image[i]) for i in range(len(names))]
        self.pairs = find_twins([r["image"] for r in rows], caption, self.categories)
        if not np.array_equal(np.sort(self.pairs.ravel()), np.arange(len(rows))):
            raise ValueError(f"{split}: need exactly one twin per caption")
        self.partner = np.empty(len(rows), dtype=np.int64)
        a, b = self.pairs.T
        self.partner[a], self.partner[b] = b, a
        self.label = np.array([int(r["label"]) for r in rows]) if "label" in rows[0] else None
        self.device = device

        if cache_dir is None:
            print(f"Extracting regions from {len(names)} {split} JPEGs", flush=True)
            arrays = extract_split(root / split, names, background, boundary=boundary, primitives=primitives,
                                   fragments=fragments, seams=seams)
        else:
            # Optional local research cache of image features only. The Kaggle
            # notebook regenerates them from JPEGs when restarted.
            with np.load(cache_dir / f"{split}.npz") as loaded:
                if list(loaded["names"]) != names:
                    raise ValueError(f"{split}: cached image order does not match CSV")
                arrays = tuple(loaded[k] for k in ("x", "xy", "valid"))
        region_targets = (region_anchors(*arrays, self.parsed, self.images, self.label)
                          if fitting and (anchors or shape_classifier is not None) else None)
        if shape_classifier is not None:
            arrays = shape_classifier.augment(root / split, names, arrays, region_targets)
        self.table = tuple(torch.from_numpy(x) for x in arrays)

        self.programs = encode_program_inputs(caption, "cpu")
        self.programs.pop("rows")
        if len(self.programs["ops"]) != len(rows):
            raise ValueError(f"{split}: caption parser does not cover all rows")
        self.anchors = region_targets if anchors else None
        self.aux = None
        if fitting:
            aux_rows = defaultdict(set)
            for n, c, y in zip(self.images, caption, self.label):
                for words, truth in implied_filters(parse_program(c), y):
                    if words and truth:
                        aux_rows[n].update(((word,), 1) for word in words)
                        aux_rows[n].add((words, 1))
            # Single-attribute and full-filter entailments are useful even when
            # the head must choose between two visually similar objects.
            encoded = []
            aux_scene = []
            aux_idx = defaultdict(list)
            for n in range(len(names)):
                for words, _ in sorted(aux_rows[n]):
                    aux_idx[n].append(len(encoded))
                    aux_scene.append(n)
                    encoded.append([*[WORD_INDEX[w] for w in words], *([0] * (4 - len(words)))])
            a = torch.tensor(encoded, dtype=torch.long).reshape(-1, 4)
            scalar = torch.zeros(len(a), dtype=torch.long)
            self.aux = ({"a": a, "b": torch.zeros_like(a), "ref": torch.zeros_like(a),
                         "ops": torch.full_like(scalar, EXIST), "relations": scalar, "numbers": scalar},
                        np.array(aux_scene), aux_idx)

    def batches(self, batch_size, rng=None):
        order = np.arange(len(self.rows)) if rng is None else rng.permutation(len(self.rows))
        for start in range(0, len(order), batch_size):
            scenes = order[start:start + batch_size]
            idx = np.concatenate([self.rows[i] for i in scenes])
            mapping = {n: i for i, n in enumerate(scenes)}
            images = torch.tensor([mapping[n] for n in self.images[idx]], device=self.device)
            region = [x[scenes].to(self.device) for x in self.table]
            program = {k: v[idx].to(self.device) for k, v in self.programs.items()}
            # Train against the labelled orientation. Both captions are always
            # scored against the same hypothesized scene.
            pos = {row: j for j, row in enumerate(idx)}
            first = idx[idx < self.partner[idx]]
            paired_rows = torch.tensor([[pos[a], pos[self.partner[a]]] for a in first], device=self.device)
            aux = None
            if rng is not None and self.aux is not None:
                program_aux, scene_aux, rows_aux = self.aux
                ax = [j for n in scenes for j in rows_aux[n]]
                if ax:
                    aux = (
                        torch.tensor([mapping[n] for n in scene_aux[ax]], device=self.device),
                        {k: v[ax].to(self.device) for k, v in program_aux.items()},
                    )
            anchors = None if self.anchors is None else self.anchors[scenes].to(self.device)
            yield idx, region, images, program, paired_rows, aux, anchors


@torch.no_grad()
def predict(model, data, batch_size):
    model.eval()
    raw = np.empty(len(data.ids), np.float64)
    for idx, region, images, program, _, _, _ in data.batches(batch_size):
        raw[idx] = model.score(*region, images, program).cpu().numpy()
    a, b = data.pairs.T
    margin = raw[a] - raw[b]
    raw[a], raw[b] = margin, -margin
    return raw


def measures(scores, data):
    report = {"auc": float(roc_auc_score(data.label, scores)),
              "accuracy": float(((scores > 0) == data.label).mean())}
    for category in sorted(set(data.categories)):
        rows = data.categories == category
        report[category] = float(roc_auc_score(data.label[rows], scores[rows]))
    return report


@torch.no_grad()
def predict_joint(model, data, batch_size, seed):
    model.eval()
    raw = np.empty(len(data.ids), np.float64)
    groups = [np.array([WORD_INDEX[w] for w in group]) for group in EXCLUSIVE_GROUPS]
    offsets = model.relation_offset.cpu().numpy()
    for idx, region, images, _, pairs, _, _ in data.batches(batch_size):
        encoded = model.encode(*region)
        obj, attributes = model.slot_logits(encoded)
        presence = obj.sigmoid().cpu().numpy()
        positions = encoded["position"].cpu().numpy()
        valid = encoded["valid"].cpu().numpy()
        probs = [attributes[..., group.tolist()].softmax(-1).cpu().numpy() for group in groups]
        images, pairs = images.cpu().numpy(), pairs.cpu().numpy()
        for i, mask in enumerate(valid):
            rows = np.flatnonzero(images == i)
            within = pairs[np.isin(pairs[:, 0], rows)]
            programs = [data.parsed[j] for j in idx[rows]]
            raw[idx[rows]] = sample_scene_scores(
                programs, np.searchsorted(rows, within), presence[i, mask],
                [(group, p[i, mask]) for group, p in zip(groups, probs)],
                positions[i, mask], offsets, seed,
            )
    return raw


def run(root, epochs=30, batch_size=48, seed=42, *, cache_dir=None, final=False,
        output=Path("final_submission.csv"), report_path=None, device_name="auto",
        boundary=False, posterior=None, anchors=False, primitives=False, schedule_epochs=None,
        crop_shapes=False, fragments=False, count_supervision=False, seams=False, shape_trees=360):
    start = time.monotonic()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    np.random.seed(seed)
    random.seed(seed)
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(accelerator if device_name == "auto" else device_name)
    rng = np.random.default_rng(seed)
    names = list(dict.fromkeys(r["image"] for r in read_rows(root, "train")))
    background = None if cache_dir is not None else fit_background(root / "train", names, seed)
    shape_classifier = None
    if crop_shapes:
        from src.region_shape_classifier import RegionShapeClassifier
        shape_classifier = RegionShapeClassifier(seed * 10, trees=shape_trees)
    train = SceneData(root, "train", background, device, fitting=True, cache_dir=cache_dir,
                      boundary=boundary, anchors=anchors, primitives=primitives, shape_classifier=shape_classifier,
                      fragments=fragments, seams=seams)
    val = None if final else SceneData(root, "val", background, device, cache_dir=cache_dir,
                                     boundary=boundary, primitives=primitives, shape_classifier=shape_classifier,
                                     fragments=fragments, seams=seams)
    print(f"Features ready in {(time.monotonic() - start)/60:.2f} min on {device}", flush=True)
    model = ImageTable(boundary, primitives or crop_shapes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, schedule_epochs or epochs, eta_min=.0003)
    best = None
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for idx, region, images, program, pairs, aux, anchor in train.batches(batch_size, rng):
            target = torch.as_tensor(train.label[idx], device=device, dtype=torch.float32)
            if count_supervision:
                raw, count_loss = model.score_and_count_loss(*region, images, program, target)
            else:
                raw = model.score(*region, images, program)
            a, b = pairs.T
            if boundary:
                count = np.isin(train.categories[idx], ("counting", "comparison", "spatial_counting"))
                weight = 1 + torch.as_tensor(count, device=device, dtype=torch.float32) * .7
            else:
                weight = None
            loss = F.binary_cross_entropy_with_logits(raw, target, weight=weight)
            loss += F.binary_cross_entropy_with_logits(raw[a] - raw[b], target[a],
                                                       weight=None if weight is None else weight[a])
            if count_supervision:
                loss += .5 * count_loss
            if aux is not None:
                loss += .3 * model.existence_loss(*region, *aux)
            if anchor is not None:
                loss += .35 * model.anchor_loss(*region, anchor)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        schedule.step()
        row = {"epoch": epoch, "loss": float(np.mean(losses))}
        if val is not None:
            scores = predict(model, val, batch_size * 2)
            row.update(measures(scores, val))
            if best is None or row["auc"] > best["auc"]:
                best = row
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)
        print(json.dumps(row), flush=True)
    report = {"history": history, "best_validation": best, "seed": seed, "device": str(device),
              "configuration": {"epochs": epochs, "schedule_epochs": schedule_epochs or epochs,
                                "batch_size": batch_size, "boundary": boundary,
                                "anchors": anchors, "posterior": posterior, "primitives": primitives,
                                "crop_shapes": crop_shapes, "fragments": fragments,
                                "count_supervision": count_supervision, "seams": seams,
                                "shape_trees": shape_trees if crop_shapes else None},
              "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                           "numpy": np.__version__, "opencv": cv2.__version__,
                           "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}}
    if final:
        # Test contributes only images and captions at inference time.
        test = SceneData(root, "test", background, device, cache_dir=cache_dir,
                         boundary=boundary, primitives=primitives, shape_classifier=shape_classifier,
                         fragments=fragments, seams=seams)
        if posterior == "joint":
            scores = predict_joint(model, test, batch_size * 2, seed)
        else:
            scores = predict(model, test, batch_size * 2)
            if posterior == "blend":
                scores = .5 * (scores + predict_joint(model, test, batch_size * 2, seed))
        confidence = expit(scores)
        sample = read_rows(root, "sample_submission")
        column, = (k for k in sample[0] if k != "id")
        ids = [int(row["id"]) for row in sample]
        by_id = dict(zip(test.ids, confidence))
        if len(by_id) != len(test.ids) or len(set(ids)) != len(ids) or set(ids) != set(by_id):
            raise ValueError("test and sample IDs must match exactly once")
        if not np.isfinite(confidence).all():
            raise ValueError("nonfinite test predictions")
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", column])
            writer.writerows((i, by_id[i]) for i in ids)
        report["submission"] = str(output)
    elif best_state is not None and (report_path is not None or posterior):
        model.load_state_dict(best_state)
        base = predict(model, val, batch_size * 2)
        outputs = {"ids": val.ids, "scores": base}
        if posterior:
            joint = predict_joint(model, val, batch_size * 2, seed)
            blend = .5 * (base + joint)
            report["joint_validation"] = measures(joint, val)
            report["blend_validation"] = measures(blend, val)
            outputs.update(joint=joint, blend=blend)
        if report_path is not None:
            np.savez(report_path.with_suffix(".npz"), **outputs)
    report["minutes"] = round((time.monotonic() - start) / 60, 2)
    if report_path is not None:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--schedule-epochs", type=int, help="keep the original cosine learning-rate horizon")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--boundary", action="store_true")
    shape_parser = parser.add_mutually_exclusive_group()
    shape_parser.add_argument("--primitives", action="store_true", help="split merged silhouettes using JPEG edges")
    shape_parser.add_argument("--crop-shapes", action="store_true", help="train a shape classifier on cropped regions")
    parser.add_argument("--fragments", action="store_true", help="include small, partly occluded regions")
    parser.add_argument("--seams", action="store_true", help="split same-color blobs along interior luminance edges")
    parser.add_argument("--count-supervision", action="store_true", help="supervise expected counts from true captions")
    parser.add_argument("--shape-trees", type=int, default=360)
    parser.add_argument("--posterior", nargs="?", const="joint", choices=("joint", "blend"))
    parser.add_argument("--anchors", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("final_submission.csv"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    if args.shape_trees < 1 or (args.shape_trees != 360 and not args.crop_shapes):
        parser.error("shape-trees must be positive and requires crop-shapes when changed")
    if args.schedule_epochs is not None and args.schedule_epochs < args.epochs:
        parser.error("schedule-epochs must be at least epochs")
    if args.final and args.cache is not None:
        parser.error("final run regenerates proposals from images")
    run(args.data, args.epochs, args.batch_size, args.seed, cache_dir=args.cache, final=args.final,
        output=args.output, report_path=args.report, device_name=args.device,
        boundary=args.boundary or args.primitives or args.crop_shapes or args.fragments or args.seams,
        posterior=args.posterior, anchors=args.anchors, primitives=args.primitives,
        schedule_epochs=args.schedule_epochs, crop_shapes=args.crop_shapes, fragments=args.fragments,
        count_supervision=args.count_supervision, seams=args.seams, shape_trees=args.shape_trees)


if __name__ == "__main__":
    main()
