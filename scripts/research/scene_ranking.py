"""Fit a scene likelihood from train captions and evaluate on frozen val detections."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path.cwd()))
from learned_center_fusion import Fusion, N_ATTRS
from native_centers import GROUPS, read_rows
from src.models.slot_program import WORD_INDEX
from src.scene_posterior import _draws, _execute
from src.twins import find_twins


ROOT = Path("artefacts/research")


def promote_candidate(p, attributes, evidence, cue, gain):
    color, bounds, score = cue
    confidence = expit(score)
    if confidence < .80:
        return p
    boxes = evidence[:, 1 + N_ATTRS:5 + N_ATTRS]
    centers = boxes[:, :2] + .5 * boxes[:, 2:]
    inside = ((centers >= bounds[:2]) & (centers < bounds[2:])).all(-1)
    channel = int(np.flatnonzero(GROUPS[1] == WORD_INDEX[color])[0])
    matching = inside & (attributes[1][:, channel] >= .55)
    main = np.flatnonzero(matching & (p >= .65))
    if len(main) != 1:
        return p
    distance = np.linalg.norm((centers-centers[main[0]])*[240,160], axis=1)
    dense = evidence[:, 5+N_ATTRS]
    eligible = matching & (dense >= .56) & (p < .65) & (distance >= 5)
    if not eligible.any():
        return p
    index = np.argmax(np.where(eligible, .5*dense + .5*p, -1))
    output = p.copy()
    output[index] = max(p[index], gain * confidence)
    return output


def hypotheses(programs, pairs, p, xy, attributes, evidence, offsets, draws, crop_cue=None):
    u = _draws(len(p), 8, 42, draws)
    present = u[..., 0] < p
    codes = np.zeros_like(present, dtype=np.int32)
    log_probability = np.zeros_like(present, dtype=np.float32)
    chosen_shape = None
    for j, (indices, prob) in enumerate(zip(GROUPS, attributes), 1):
        index = (u[..., j, None] > prob.cumsum(-1)).sum(-1).clip(max=len(indices) - 1)
        codes |= 1 << indices[index]
        log_probability += np.log(prob[np.arange(len(p))[None, :], index].clip(1e-5))
        if j == len(attributes):
            chosen_shape = index
    position = xy[None] + logit(u[..., -3:-1]) * .026
    truth = _execute(programs, codes, present, position, p, offsets, u[..., -1])
    a, b = pairs.T
    linked = truth[:, a] ^ truth[:, b]
    count = present.sum(1)
    boxes = evidence[:, 1 + N_ATTRS:5 + N_ATTRS]
    origin, extent = boxes[:, :2], boxes[:, 2:]
    end = origin + extent
    overlap = (np.minimum(end[:, None], end) - np.maximum(origin[:, None], origin)).clip(0).prod(-1)
    area = extent.prod(-1)
    containment = overlap / np.minimum.outer(area, area).clip(1e-6)
    ratio = np.minimum.outer(area, area) / np.maximum.outer(area, area).clip(1e-6)
    aa, bb = np.where(np.triu((containment > .88) & (ratio > .35), 1))
    duplicates = (present[:, aa] & present[:, bb]).sum(-1)
    dense = evidence[:, 5 + N_ATTRS]
    dense_shapes = evidence[:, -3:]
    agreement = dense_shapes[np.arange(len(p))[None, :], chosen_shape]
    crop_score = np.zeros(len(count), np.float32)
    if crop_cue is not None:
        color, bounds, score = crop_cue
        center = origin + .5 * extent
        inside = ((center >= bounds[:2]) & (center < bounds[2:])).all(-1)
        hue = 1 << WORD_INDEX[color]
        multiple = ((present & inside) & ((codes & hue) != 0)).sum(1) >= 2
        crop_score = np.clip(score, -4., 4.) * multiple
    features = np.column_stack((
        count, abs(count - p.sum()), (present * p).sum(1), (present * (p < .4)).sum(1),
        (present * (p >= .8)).sum(1), (present * dense).sum(1), duplicates,
        (present * log_probability).sum(1), (present * agreement * dense).sum(1),
        (count < 3) | (count > 10), (present * np.sqrt(area.clip(0))).sum(1) * 10, crop_score,
    )).astype(np.float32)
    features -= features.mean(axis=0)
    base = (1.3 * (linked.sum(1) - len(pairs))).astype(np.float32)
    return features, base, linked, truth[:, a]


@torch.inference_mode()
def tables(split, model, draws, crop_cues=None, promotion=0.):
    frame, scenes, captions, programs = read_rows(split)
    path = ROOT / "learned_fusion16" / f"{split}.npz"
    with np.load(path) as archive:
        assert list(archive["names"]) == list(scenes)
        x, xy, valid = (torch.from_numpy(archive[key]) for key in ("x", "xy", "valid"))
    offsets = model.relation_offset.detach().numpy()
    result = []
    blocks = [list(range(i, min(i + 128, len(x)))) for i in range(0, len(x), 128)]
    names, rows = list(scenes), list(scenes.values())
    for block in blocks:
        enc = model.encode(x[block], xy[block], valid[block])
        p, attributes = model.slot_logits(enc)
        probabilities = [attributes[..., group.tolist()].softmax(-1).numpy() for group in GROUPS]
        p = p.sigmoid().numpy()
        for j, i in enumerate(block):
            idx = rows[i]
            pairs = find_twins(frame.image.iloc[idx], [captions[k] for k in idx], frame.category.iloc[idx])
            mask = valid[i].numpy()
            p_i = p[j, mask]
            if not len(p_i) or not len(pairs):
                result.append((idx, pairs, None))
                continue
            color_cue = crop_cues.get(names[i]) if crop_cues is not None else None
            attr = [arr[j, mask] for arr in probabilities]
            evidence = x[i, mask].numpy()
            if promotion and color_cue is not None:
                p_i = promote_candidate(p_i, attr, evidence, color_cue, promotion)
            record = hypotheses(
                [programs[k] for k in idx], pairs, p_i, enc["position"][j, mask].numpy(),
                attr, evidence, offsets, draws, color_cue,
            )
            result.append((idx, pairs, record))
    return frame, result


def pack(frame, scenes):
    max_pairs = max(len(pairs) for _, pairs, _ in scenes)
    available = [(frame.label.to_numpy()[idx[pairs[:, 0]]].astype(bool), record)
                 for idx, pairs, record in scenes if record is not None]
    features, base, keep, good = [], [], [], []
    for labels, (x, prior, linked, truth) in available:
        width = max_pairs - len(labels)
        features.append(x)
        base.append(prior)
        keep.append(np.pad(linked, ((0, 0), (0, width))))
        good.append(np.pad(linked & (truth == labels), ((0, 0), (0, width))))
    return tuple(torch.from_numpy(np.stack(rows)) for rows in (features, base, keep, good))


def fit(data, seed, device="cpu"):
    features, base, linked, correct = (tensor.to(device) for tensor in data)
    gen = torch.Generator().manual_seed(seed)
    coef = torch.nn.Parameter(torch.zeros(features.shape[-1], device=device))
    optimizer = torch.optim.AdamW([coef], lr=.014)
    for epoch in range(12):
        losses = []
        for idx in torch.randperm(len(features), generator=gen).split(96):
            idx = idx.to(device)
            shift = features[idx] @ (2.0 * coef.tanh())
            u = (base[idx] + shift.clamp(-9, 9)).exp()[..., None]
            mass = (u * linked[idx]).sum(1)
            good = (u * correct[idx]).sum(1)
            observed = linked[idx].any(1)
            loss = -(torch.log((good + .75) / (mass + 1.5)) * observed).sum() / observed.sum().clamp_min(1)
            joint = correct[idx].sum(-1) == observed.sum(-1, keepdim=True)
            compatible = joint.any(1)
            if compatible.any():
                joint_mass = (u.squeeze(-1) * joint).sum(1)
                all_mass = u.squeeze(-1).sum(1)
                loss -= .12 * ((joint_mass[compatible] + .75) / (all_mass[compatible] + 1.5)).log().mean()
            loss += .045 * coef.square().sum() + .012 * shift.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch in (0, 5, 11):
            print("train epoch", epoch + 1, "MML", round(float(np.mean(losses)), 5), flush=True)
    return (2 * coef.detach().tanh()).cpu().numpy()


def evaluate(frame, scenes, coef):
    predictions = np.zeros(len(frame))
    for idx, pairs, record in scenes:
        if record is None:
            continue
        x, base, linked, truth = record
        weights = np.exp(base + np.clip(x @ coef, -9, 9))[:, None] * linked
        left = logit(((truth * weights).sum(0) + .75) / (weights.sum(0) + 1.5))
        predictions[idx[pairs[:, 0]]] = left
        predictions[idx[pairs[:, 1]]] = -left
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=1024, choices=(512, 1024, 2048))
    args = parser.parse_args()
    torch.set_num_threads(3)
    teacher = torch.load(ROOT / "scene_silhouette.pt", map_location="cpu", weights_only=True)
    fusion = Fusion(teacher).eval()
    fusion.load_state_dict(torch.load(ROOT / "local_centers_final/fusion.pt", map_location="cpu", weights_only=True))
    train, training = tables("train", fusion, args.draws)
    print("built training hypotheses", len(training), args.draws, flush=True)
    coef = fit(pack(train, training), 42)
    del training
    frame, scenes = tables("val", fusion, args.draws)
    predictions = {}
    baseline = evaluate(frame, scenes, np.zeros_like(coef))
    print("fusion control", roc_auc_score(frame.label, baseline), flush=True)
    other = []
    for name, key in (("center_proposals8", "permissive"), ("region_count_supervision", "joint"),
                      ("native_natural_epoch21", "joint")):
        with np.load(ROOT / f"{name}.npz") as saved:
            assert np.array_equal(saved["ids"], frame.id)
            other.append(saved[key])
    for strength in (0, .25, .5, 1):
        values = evaluate(frame, scenes, strength * coef)
        ensemble = (values + sum(other)) / 4
        predictions[f"scale_{strength:g}"] = ensemble
        errors = int((((2 * frame.label.to_numpy() - 1) * ensemble) < 0).sum() // 2)
        print(strength, "fusion", roc_auc_score(frame.label, values),
              "ensemble", roc_auc_score(frame.label, ensemble), "wrong pairs", errors, flush=True)
    np.savez(ROOT / "train_scene_ranking.npz", ids=frame.id, coef=coef, **predictions)


if __name__ == "__main__":
    main()
