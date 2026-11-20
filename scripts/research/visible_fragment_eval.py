"""Evaluate real rear fragments and replay accepted proposals through the current executor."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit, softmax
from sklearn.metrics import roc_auc_score

from scripts.research import layer_repair as layer
from scripts.research.fragment_votes import EXTRA, H, W, v
from scripts.research.visible_fragment_detector import ARMS, OUT, pairwise_iou

SEEDS = (421, 853)
THRESHOLDS = (.50, .70, .85, .90, .95, .98)
PRIMARY = .90
WEIGHT = .50
SHAPES = ("b", "s", "c")
COLORS = ("gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow")


def midpoint(box):
    return box[..., :2] + box[..., 2:] / 2


def inside(point, box, margin=0):
    return bool(((point >= box[:2]-margin) & (point < box[:2]+box[2:]+margin)).all())


def candidates(z, values):
    for i, j, row in zip(z["si"], z["at"], values):
        b = row[21:25]
        if row[25] < 4 or b[2:].min() < 6 or not inside(midpoint(b), np.array([0,0,W,H])):
            continue
        yield str(z["names"][i]), int(i), int(j), row


def consensus(a,b):
    for key in ("si", "at", "names", "boxes", "probability", "valid", "xy", "offsets"):
        if not np.array_equal(a[key],b[key]):
            raise ValueError(f"Mismatched evaluation banks: {key}")
    left, right = a["data"], b["data"]
    avg = (left+right)/2
    agrees = ((np.linalg.norm(left[:,30:32]-right[:,30:32],axis=1) <= 7)
              & (np.linalg.norm(midpoint(left[:,21:25])-midpoint(right[:,21:25]),axis=1) <= 10))
    avg[:,0] = np.minimum(left[:,0],right[:,0])
    avg[:,25] = np.minimum(left[:,25],right[:,25]) * agrees
    return avg


def unresolved(z,i,j,box):
    bank = z["boxes"][i]
    p = z["probability"][i]
    overlap = pairwise_iou(box,bank)
    distance = np.linalg.norm(midpoint(box)-midpoint(bank),axis=1)
    same = ((overlap>.42) & (distance<np.clip(.29*max(box[2:]),4,10)) & z["valid"][i])
    if same[j] or (same & (p>=.60)).any():
        return None
    low = np.flatnonzero(same & (p<.60))
    return int(low[distance[low].argmin()]) if len(low) else -1


def rules(row, front):
    rear = row[21:25]
    # Require a localized fragment near (or within) the foreground box. A
    # pixel within the rectangular prompt can still lie outside the front mask.
    return (inside(row[30:32],front,6) and pairwise_iou(rear,front)>.06
            and rear[1]+rear[3] < front[1]+front[3]-2)


def boxes_from(row, prefix):
    return np.asarray([getattr(row, prefix+k) for k in ("x","y","w","h")])


def real_metrics(predictions, review, baseline_boxes):
    results = []
    for case in review.itertuples(index=False):
        front = boxes_from(case,"f")
        visible = boxes_from(case,"r")
        proposed = predictions.get(case.image,[])
        candidates = [(row, changed) for anchor,row,changed in proposed if pairwise_iou(front,anchor) >= .25]
        if not case.rear:
            candidates = [(r, changed) for r, changed in candidates if COLORS[r[3:11].argmax()] == case.color]
        old = any(pairwise_iou(visible,b) >= .25 and pairwise_iou(front,b) < .50
                  for b in baseline_boxes.get(case.image, []))
        found = [row for row, _ in candidates if case.rear and inside(row[30:32],visible,2)]
        if case.rear:
            # Do not treat a second prediction on an incompletely labeled scene as a confirmed false alarm.
            ordered = sorted(found, key=lambda row:row[0], reverse=True)
        else:
            ordered = sorted([row for row,_ in candidates if inside(row[30:32],visible)],key=lambda row:row[0],reverse=True)
        all_unresolved = [row for row, changed in candidates if changed and inside(row[30:32],visible,2 if case.rear else 0)]
        injected = max(all_unresolved,key=lambda row:row[0],default=None)
        top = ordered[0] if ordered else None
        pred = None if top is None else SHAPES[softmax(top[11:17]).reshape(2,3).sum(0).argmax()]
        results.append(dict(image=case.image, rear=bool(case.rear), shape=case.shape,
                            baseline_rear_match=bool(case.rear and old), localized_candidate=bool(candidates), score=0. if top is None else float(expit(top[0])),
                            shape_pred=pred, target_addition_score=0. if injected is None else float(expit(injected[0]))))
    df = pd.DataFrame(results)
    pos, neg = df[df.rear],df[~df.rear]
    operating = {}
    for q in THRESHOLDS:
        hit = pos.score>=q
        operating[str(q)] = dict(found=int(hit.sum()), rear_total=len(pos),
                                  baseline_rear_matches=int(pos.baseline_rear_match.sum()), novel_hits=int((hit & ~pos.baseline_rear_match).sum()),
                                  correct_shape=int((hit & pos["shape"].eq(pos.shape_pred)).sum()),
                                  false_control_crops=int((neg.score>=q).sum()), control_crops=len(neg),
                                  rear_additions=int((pos.target_addition_score>=q).sum()),
                                  false_addition_control_crops=int((neg.target_addition_score>=q).sum()))
    return df, operating


def score_metrics(frame, baseline, score):
    y = frame.label.to_numpy()
    rows = frame.assign(position=np.arange(len(frame))).sort_values("label",ascending=False)
    groups = rows.groupby(["image","pair_id"],sort=False).position.agg(list)
    pairs = np.asarray([r for r in groups if len(r)==2],dtype=np.int64)
    if len(pairs) != len(frame)//2 or not ((y[pairs[:,0]]==1)&(y[pairs[:,1]]==0)).all():
        raise ValueError("Expected one true and one false row in each val pair")
    def errors(values):
        return values[pairs[:,0]]<=values[pairs[:,1]]
    old,new = errors(baseline),errors(score)
    auc = roc_auc_score(y,score)
    per_category = {key:float(roc_auc_score(group.label,score[group.index])) for key,group in frame.groupby("category")}
    return dict(auc=float(auc), delta_auc=float(auc-roc_auc_score(y,baseline)), inversions=round(float((1-auc)*y.sum()*(1-y).sum()),2), errors=int(new.sum()),
                fixed=int((old&~new).sum()), regressed=int((~old&new).sum()), auc_by_category=per_category)


def run(split, arm, mode, frame, base, review):
    columns = [layer.canonical_form(r.caption,r.category) for r in frame.itertuples()]
    assigned = frame.groupby("image",sort=False).indices
    predictions, selected, baseline_boxes = {}, [], {}
    delta = np.zeros(len(frame))
    for shard in (0,1):
        seeds = SEEDS if mode == "both" else (int(mode),)
        data = []
        for seed in seeds:
            with np.load(OUT/f"{arm}_{seed}_{split}_{shard}.npz") as z:
                data.append(dict(z))
        z = data[0]
        values = consensus(*data) if len(data)==2 else z["data"]
        for name, b, p, ok in zip(z["names"], z["boxes"], z["probability"], z["valid"]):
            baseline_boxes[str(name)] = b[ok & (p >= .60)]
        by_image = {}
        for name,i,j,row in candidates(z,values):
            front = z["boxes"][i,j]
            if not rules(row,front):
                continue
            slot = unresolved(z,i,j,row[21:25])
            predictions.setdefault(name,[]).append((front,row,slot is not None))
            if slot is None or row[0]<logit(PRIMARY):
                continue
            previous = by_image.get(name)
            if previous is None or row[0]>previous[2][0]:
                by_image[name] = (i,j,row,slot)
        for name,(i,j,row,slot) in by_image.items():
            rows = assigned[name]
            use = z["valid"][i]
            if slot>=0:
                slot = np.cumsum(use)[slot]-1
            pairs = layer.find_twins(frame.image.iloc[rows],[columns[r] for r in rows],frame.category.iloc[rows])
            programs = [layer.parse_program(columns[r]) for r in rows]
            d = layer.change_score(programs,pairs,z["probability"][i,use].copy(),
                                   [z[f"attr_{k}"][i,use].copy() for k in range(4)],z["xy"][i,use],
                                   z["offsets"],slot,row,row[21:25])
            delta[rows] = np.where(frame.category.iloc[rows].isin(layer.rear.CORE),WEIGHT*d,0.)
            selected.append(dict(image=name, probability=float(expit(row[0])), visible_xy=row[30:32].tolist(),
                                 visible_pixels=float(row[25]), rear=row[21:25].tolist(), front=z["boxes"][i,j].tolist(),
                                 metal_cylinder_prob=float(softmax(row[11:17])[-1])))
    score = base+delta
    if split=="test":
        return score
    reviewed, operating = real_metrics(predictions,review,baseline_boxes)
    reviewed.to_csv(OUT/f"real_{arm}_{mode}.csv",index=False)
    (OUT/f"accepted_{arm}_{mode}.json").write_text(json.dumps(selected,indent=2))
    focused = {}
    for name in ("58281.jpg", "78612.jpg", "26268.jpg"):
        rows = sorted(predictions.get(name,[]),key=lambda entry:entry[1][0],reverse=True)
        focused[name] = [dict(probability=float(expit(row[0])), front=front.tolist(),
                              rear=row[21:25].tolist(), visible_xy=row[30:32].tolist(),
                              pixels=float(row[25]), unresolved=unresolved_box,
                              metal_cylinder=float(softmax(row[11:17])[-1]))
                         for front,row,unresolved_box in rows[:8]]
    (OUT/f"focus_{arm}_{mode}.json").write_text(json.dumps(focused,indent=2))
    return score, operating, len(selected)


def val(frame, base):
    review = pd.read_csv(EXTRA/"rear_review_val.csv").fillna("")
    if review.empty or not set(review.image)<=set(frame.image):
        raise ValueError("Reviewed boxes need to be drawn from held-out val scenes")
    report = {"baseline":score_metrics(frame,base,base), "threshold":PRIMARY, "weight":WEIGHT,"arms":{}}
    rows = frame.assign(base=base)
    y = frame.label.to_numpy()
    images, names = pd.factorize(frame.image)
    rng = np.random.default_rng(228)
    weights = [rng.multinomial(len(names),np.full(len(names),1/len(names)))[images] for _ in range(600)]
    baseline_boot = np.asarray([roc_auc_score(y,base,sample_weight=w) for w in weights])
    for arm in ARMS:
        report["arms"][arm] = {}
        for mode in (*map(str,SEEDS),"both"):
            score, detection, n = run("val",arm,mode,frame,base,review)
            metrics = score_metrics(frame,base,score)
            if mode=="both":
                boot = [roc_auc_score(y,score,sample_weight=w) for w in weights]
                metrics["delta_ci95"] = np.quantile(np.asarray(boot)-baseline_boot,[.025,.975]).tolist()
            rows[f"{arm}_{mode}"] = score
            report["arms"][arm][mode] = dict(ensemble=metrics,detection=detection,accepted_scenes=n)
    good = []
    clean = report["arms"]["clean"]["both"]["detection"][str(PRIMARY)]
    for arm in ARMS[1:]:
        results = report["arms"][arm]
        paired = results["both"]["ensemble"]
        detection = results["both"]["detection"][str(PRIMARY)]
        if (paired["delta_ci95"][0]>0 and paired["fixed"]>=paired["regressed"]
                and detection["found"]>clean["found"] and detection["false_control_crops"]<=clean["false_control_crops"]
                and all(results[str(s)]["ensemble"]["auc"]>report["baseline"]["auc"] for s in SEEDS)):
            good.append((paired["auc"],arm))
    report["winner"] = max(good)[1] if good else None
    rows.to_csv(OUT/"val_rows.csv",index=False)
    (OUT/"metrics.json").write_text(json.dumps(report,indent=2))
    return report


def main(split):
    frame = pd.read_csv(v.DATA/f"{split}.csv")
    with np.load(EXTRA/f"anchor_{split}.npz") as z:
        if not np.array_equal(z["ids"],frame.id):
            raise ValueError("Baseline IDs are out of order")
        values = z["score" if split=="val" else "probability"]
    if split=="val":
        report = val(frame,values)
        print("Selected", report["winner"], flush=True)
    else:
        arm = json.loads((OUT/"metrics.json").read_text())["winner"]
        if not arm:
            raise ValueError("No candidate passed validation")
        base = logit(values.clip(1e-12,1-1e-12))
        score = run("test",arm,"both",frame,base,None)
        prob = np.where(score==base,values,expit(score))
        sample = pd.read_csv(v.DATA/"sample_submission.csv")
        if not frame.id.is_unique or not sample.id.is_unique or set(frame.id)!=set(sample.id):
            raise ValueError("Submission IDs do not match")
        if not np.isfinite(prob).all():
            raise ValueError("Non-finite scores")
        sample["label"] = pd.Series(prob,index=frame.id).loc[sample.id].to_numpy()
        sample.to_csv(OUT/f"rear_{arm}.csv",index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split",choices=("val","test"))
    main(parser.parse_args().split)


