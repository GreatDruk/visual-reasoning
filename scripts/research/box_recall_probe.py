"""Measure diagnostic recall at fixed false-candidate budgets on reviewed val boxes."""

import json

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from native_centers import CenterDetector, ROOT, STRIDE, read_rgb, read_rows
from overlap_review import COLORS, ROOT as ART


def intersection(a, b):
    return np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()


def iou(a, b):
    overlap = intersection(a,b)
    union = np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-overlap
    return overlap/max(1.,union)


def select(rows, box, probability, color, shape):
    frame = rows.iloc[0]
    bounds = np.array([frame.x0,frame.y0,frame.x1,frame.y1])
    centers = .5*(box[:,:2]+box[:,2:])
    cropped = np.column_stack((np.maximum(box[:,:2],bounds[:2]),np.minimum(box[:,2:],bounds[2:])))
    valid = ((centers>=bounds[:2]) & (centers<bounds[2:])).all(1)
    valid &= (color==COLORS.index(frame.color)) & (probability>.025)
    return [(float(p),frame.image,b,int(s)) for b,p,s in zip(
        cropped[valid],probability[valid],shape[valid],strict=True)]


def contours(review):
    with np.load(ART/"scene_silhouette.npz") as data:
        table = dict(data)
    scenes = {n:i for i,n in enumerate(read_rows("val")[1])}
    out = []
    for name, rows in review.groupby("image"):
        i = scenes[name]
        mask = table["valid"][i]
        b = table["features"][i,mask,8:12]*[240,160,240,160]
        box = np.column_stack((b[:,:2],b[:,:2]+b[:,2:]))
        out.extend(select(rows,box,table["p"][i,mask],table["color"][i,mask].argmax(1),
                          table["shape"][i,mask].argmax(1)))
    return out


@torch.inference_mode()
def network(review, checkpoint):
    model = CenterDetector().to(memory_format=torch.channels_last).eval()
    model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location="cpu"))
    scenes = list(review.groupby("image"))
    rgb = np.stack([read_rgb(ROOT/"val"/name) for name,_ in scenes])
    tensor = torch.tensor(rgb).permute(0,3,1,2).float()/255
    raw = model(tensor.contiguous(memory_format=torch.channels_last))
    p = raw[:,0].sigmoid()
    peaks = (p == F.max_pool2d(p[:,None],3,1,1)[:,0]) & (p > .025)
    out = []
    for i, (_, rows) in enumerate(scenes):
        ys,xs = torch.where(peaks[i])
        center = (raw[i,1:3,ys,xs].clip(-.5,1.5)+torch.stack((xs,ys))).T.numpy()*STRIDE
        hw = .5*raw[i,3:5,ys,xs].clip(-2,1.6).exp().T.numpy()*24
        box = np.column_stack((center-hw,center+hw))
        out.extend(select(rows,box,p[i,ys,xs].numpy(),raw[i,7:15,ys,xs].argmax(0).numpy(),
                          raw[i,-3:,ys,xs].argmax(0).numpy()))
    return out


def matched_budget(predictions, gold, require_shape):
    targets = {name:[(np.array([v.x0,v.y0,v.x1,v.y1]),"bsc".index(v.shape))
                     for v in rows.itertuples(index=False)] for name,rows in gold.groupby("image")}
    budget = (0,2,4,8)
    assigned = set()
    false = 0
    points = {k:0 for k in budget}
    for _, name, box, shape in sorted(predictions, key=lambda row: -row[0]):
        available = [(iou(box,target),i) for i,(target,kind) in enumerate(targets[name])
                     if (name,i) not in assigned and (not require_shape or shape==kind)]
        quality, index = max(available, default=(0.,-1))
        if quality>=.5:
            assigned.add((name,index))
        else:
            false += 1
        for k in budget:
            if false<=k:
                points[k] = len(assigned)
    return {str(k):count for k,count in points.items()}


if __name__=="__main__":
    torch.set_num_threads(3)
    base = ART/"overlap_review"
    gold = pd.read_csv(base/"visible_boxes.csv")
    review = pd.read_csv(base/"review_labels.csv")
    review = review[(review.split=="val") & review.image.isin(gold.image)]
    designs = {"contours":contours(review)}
    for name in ("native_natural_epoch21","native_roi_pseudo_independent","native_roi_reviewed_independent"):
        if (ART/f"{name}.pt").is_file():
            designs[name] = network(review, ART/f"{name}.pt")
    report = {"scenes": int(gold.image.nunique()),"visible_boxes":len(gold),
              "recall_counts":{name:{"geometry":matched_budget(scores,gold,False),
                                      "geometry_and_shape":matched_budget(scores,gold,True)}
                               for name,scores in designs.items()}}
    print(json.dumps(report,indent=2))
    (ART/"review_box_recall.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
