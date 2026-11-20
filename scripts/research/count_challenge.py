"""Evaluate count likelihoods against adjacent alternatives on labeled scenes."""

import argparse
from dataclasses import replace
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.special import logit
import torch

from scripts.research import dense_counts, spatial_geometry as geometry
from src.models.slot_program import EXACT


ROOT = Path("artefacts/kaggle_suite/curriculum_run")
ARMS = ("r50_421", "r50_853", "convnext_421")
RIM = geometry.rim


def query_rows(frame):
    fields = ["image", "category", "caption", "id", "pair_id"]
    for row in frame.loc[frame.category.isin(["counting", "spatial_counting"])
                         & frame.label.eq(1), fields].itertuples(index=False):
        program = RIM.layer.parse_program(RIM.layer.canonical_form(row.caption, row.category))
        if program is not None and program.op == EXACT:
            yield row, program


def histograms(root, split="val", names=ARMS):
    with ZipFile(root.parent/"count_run/count_results.zip") as archive:
        for arm in names:
            with np.load(io.BytesIO(archive.read(f"{arm}/verify_result/counts_{split}.npz"))) as data:
                yield arm, pd.DataFrame(data["hist"], index=data["names"])


def dense_count(hist, image, words):
    mask = np.array([set(words).issubset(kind) for kind in dense_counts.TYPES])
    return float(hist.loc[image].to_numpy()[mask].sum())


def pmf(weights):
    likelihood = np.array([1.])
    for weight in weights:
        likelihood = np.r_[likelihood*(1-weight), 0.] + np.r_[0., likelihood*weight]
    return likelihood


def masks(words):
    return np.concatenate([np.isin(group, words) if set(group) & set(words)
                           else np.ones(len(group), dtype=bool) for group in RIM.GROUPS])


def count_object(objs, words):
    weights = objs["p"].copy()
    groups = np.split(masks(words), np.cumsum([len(group) for group in RIM.GROUPS])[:-1])
    for key, allowed in zip(("size", "color", "material", "shape"), groups):
        weights *= objs[key][:, allowed].sum(-1)
    return pmf(weights)


def store_fields(cache, want):
    fields = {}
    for z, (p, xy, valid, attr, _) in cache:
        for i, name in enumerate(z["names"]):
            if name in want:
                fields[name] = dict(p=p[i]*valid[i], xy=xy[i], x=z["x"][i, :, RIM.v.BOX],
                                    **dict(zip(("size", "color", "material", "shape"),
                                               (data[i] for data in attr))))
    if set(fields) != want:
        raise ValueError("Missing object bank predictions")
    return fields


def canonical_objects(fields):
    """Keep one prediction for detections with the same box, attributes and anchor."""
    p = fields["p"].copy()
    origin = fields["x"][:, :2]
    extent = fields["x"][:, 2:]
    corner = origin+extent
    meet = np.minimum(corner[:, None], corner[None]) - np.maximum(origin[:, None], origin[None])
    area = np.maximum(extent, 0.).prod(-1)
    contained = np.maximum(meet, 0.).prod(-1) / np.maximum(np.minimum(area[:, None], area[None]), 1e-8)
    difference = np.abs(fields["xy"][:, None]-fields["xy"][None])*.5
    scale = np.minimum(extent[:, None], extent[None])
    nearby = np.all(difference <= scale*.25, axis=-1)
    same = (contained >= .8) & nearby
    for key in ("color", "shape", "size"):
        probs = fields[key]
        chosen = probs.argmax(-1)
        confident = probs.max(-1) >= .7
        same &= (chosen[:, None] == chosen[None]) & confident[:, None] & confident[None]
    kept = []
    for j in np.argsort(-p, kind="stable"):
        if p[j] <= 0:
            break
        if same[j, kept].any():
            p[j] = 0.
        else:
            kept.append(j)
    return dict(fields, p=p)


@torch.inference_mode()
def spatial_likelihood(fields, queries, model):
    samples, relations, numbers, subjects, references = [], [], [], [], []
    for name, program in queries:
        samples.append(fields[name])
        relations.append(program.relation)
        numbers.append(program.number)
        subjects.append(masks(program.a))
        references.append(masks(program.ref))
    arguments = [torch.as_tensor(np.asarray(v)) for v in (relations, numbers, subjects, references)]
    result = {k: [] for k in ("geometry", "center", "box", "unique")}
    unique = {name: canonical_objects(data)["p"] for name, data in fields.items()}
    blocks = torch.arange(len(queries)).split(128)
    for block in blocks:
        field = geometry.scene_tensors({key: np.stack([samples[i][key] for i in block])
                                       for key in samples[0]})
        args = [a[block] for a in arguments]
        result["geometry"].append(model(field, *args).numpy())
        result["center"].append(model(field, *args, learned=False).numpy())
        x, w = field["x"].split(2, dim=-1)
        box = dict(field, xy=(x + w*x.new_tensor([.5, .88]))*2-1)
        result["box"].append(model(box, *args, learned=False).numpy())
        separate = torch.as_tensor(np.stack([unique[queries[i][0]] for i in block]))
        result["unique"].append(model(dict(field, p=separate), *args).numpy())
    return {key: np.concatenate(parts) for key, parts in result.items()}


def preference(p, q):
    return float(logit(p.clip(1e-6, 1-1e-6)) - logit(q.clip(1e-6, 1-1e-6)))/2


def adjacent(program):
    return [replace(program, number=k) for k in (program.number-1, program.number+1) if 0 <= k <= 10]


def records(frame, root):
    queries = list(query_rows(frame))
    false = {
        (row.image, row.pair_id): RIM.layer.parse_program(RIM.layer.canonical_form(row.caption, row.category))
        for row in frame[frame.label.eq(0)].itertuples()
    }
    fields = store_fields(geometry.replay.tables_for(root, "val"), {row.image for row, _ in queries})
    dense = dict(histograms(root))
    model = geometry.Geometry()
    model.load_state_dict(RIM.v.load(root/"spatial_geometry.pt"))
    spatial_queries = [(row.image, p) for row, program in queries if program.relation
                       for p in (program, *adjacent(program))]
    spatial = spatial_likelihood(fields, spatial_queries, model)
    geometry_pos = 0
    for row, program in queries:
        other = adjacent(program)
        old = false[(row.image, row.pair_id)]
        def description(alt):
            return dict(image=row.image, id=row.id, caption=row.caption, alternative=alt.number,
                        category=row.category, relation=program.relation, novel=alt != old)
        if program.relation:
            selected = range(geometry_pos+1, geometry_pos+len(other)+1)
            for j, alt in zip(selected, other):
                yield dict(**description(alt),
                           **{key: preference(p[geometry_pos], p[j]) for key, p in spatial.items()})
            geometry_pos += len(other) + 1
        else:
            occ = count_object(fields[row.image], program.a)
            counts = {key: dense_count(hist, row.image, program.a) for key, hist in dense.items()}
            def evidence(count, alt):
                return float(np.clip(4.5*((count-alt.number)**2-(count-program.number)**2), -8, 8))
            for alt in other:
                yield dict(**description(alt),
                           objects=float(np.log((occ[program.number]+1e-5)/(occ[alt.number]+1e-5))),
                           **{key: evidence(n, alt) for key, n in counts.items()})


def compare(frame, models):
    def as_score(values):
        return np.where(values > 0, 1., .5*(values == 0))

    report = {}
    for name, prediction in models.items():
        report[name] = dict(accuracy=float(as_score(prediction).mean()) if len(prediction) else None,
                            wrong=int(np.count_nonzero(prediction < 0)),
                            ties=int(np.count_nonzero(prediction == 0)))
    return dict(rows=len(frame), scenes=int(frame.image.nunique()), models=report)


def evaluate(frame):
    count = frame[frame.category.eq("counting")]
    spatial = frame[frame.category.eq("spatial_counting")]
    dense = (count.r50_421.to_numpy()+count.r50_853.to_numpy())/2
    objects = count.objects.to_numpy()
    convnext = count.convnext_421.to_numpy()
    geom = spatial.geometry.to_numpy()
    base = spatial.center.to_numpy()
    box = spatial.box.to_numpy()
    unique = spatial.unique.to_numpy()
    split = (dense*objects < 0)
    agree = (count.r50_421*count.r50_853 > 0) & (np.minimum(abs(count.r50_421), abs(count.r50_853)) > logit(.85))
    # Match the evidence patterns that trigger direct overrides in the current ensemble.
    from scripts.research.count_evidence import MAX_BASE_LOGIT
    spatial_gate = (geom*(geom-base) > 0) & (np.abs(base) <= MAX_BASE_LOGIT)
    spatial_split = ((geom*base) < 0) & spatial_gate
    return {
        "count_all": compare(count, {"objects": objects, "r50": dense, "convnext": convnext,
                                    "r50_convnext": (dense+convnext)/2}),
        "count_conflict_confident_r50": compare(count[split&agree], {
            "objects": objects[split&agree], "r50": dense[split&agree], "convnext": convnext[split&agree]}),
        "count_conflict_independent_agreement": compare(count[split&agree&(convnext*dense>0)], {
            "r50": dense[split&agree&(convnext*dense>0)]}),
        "spatial_all": compare(spatial, {"center": base, "geometry": geom, "box": box, "unique": unique,
                                         "mean": (base+geom+box)/3}),
        "spatial_dedup_changed": compare(spatial[unique != geom], {
            "geometry": geom[unique != geom], "unique": unique[unique != geom]}),
        "spatial_conflict_current_gate": compare(spatial[spatial_split], {
            "center": base[spatial_split], "geometry": geom[spatial_split], "box": box[spatial_split]}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    frame = pd.read_csv(RIM.v.DATA/"val.csv")
    scores = pd.DataFrame(records(frame, args.root))
    report = dict(all_neighbors=evaluate(scores), new_neighbors=evaluate(scores[scores.novel]))
    output = args.root/"error_audit"
    output.mkdir(exist_ok=True)
    scores.to_json(output/"count_challenge_rows.json", orient="records", indent=2)
    (output/"count_challenge.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
