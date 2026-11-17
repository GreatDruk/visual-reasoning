"""Compare visual confidence and duplicate-region hypotheses on frozen models."""

from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path.cwd()))
from native_centers import FIELDS, GROUPS, read_rows
from src.scene_posterior import _draws, _execute
from src.twins import find_twins


OPTIONS = {
    "control": (1., 1., 0.),
    "object_soft": (1.35, 1., 0.),
    "object_sharp": (.8, 1., 0.),
    "shape_soft": (1., 1.4, 0.),
    "deduplicate": (1., 1., 1.2),
    "soft_and_deduplicate": (1.35, 1.4, 1.2),
}


def duplicates(box, attr):
    low, high = box[:,:2], box[:,:2]+box[:,2:]
    intersection = (np.minimum(high[:,None],high) - np.maximum(low[:,None],low)).clip(0).prod(-1)
    area = box[:,2:].prod(1)
    min_area = np.minimum(area[:,None],area)
    max_area = np.maximum(area[:,None],area)
    agreement = attr @ attr.T
    warning = ((intersection > .74*(area[:,None] + area - intersection)) |
               ((intersection > .91*min_area)&(min_area > .36*max_area)&(agreement > .65)))
    return np.where(np.triu(warning,1))


def run_scene(programs, pairs, field, p, xy, box, offsets):
    if not len(p):
        return {name:np.zeros(len(programs)) for name in OPTIONS}
    u = _draws(len(p),8,42,1024)
    coords = xy[None]+logit(u[...,-3:-1])*.026
    source_p = logit(p.clip(1e-6,1-1e-6))
    duplicate_a, duplicate_b = duplicates(box,field[1])
    options = {}
    a,b = pairs.T
    for name,(object_temp,shape_temp,penalty) in OPTIONS.items():
        presence = u[...,0] < (p if object_temp==1 else expit(source_p/object_temp))
        code = np.zeros_like(presence,dtype=np.int32)
        for j,(indices,raw) in enumerate(zip(GROUPS,field),1):
            if j==4 and shape_temp!=1:
                raw = raw**(1/shape_temp)
                raw /= raw.sum(-1,keepdims=True)
            choice = (u[...,j,None]>raw.cumsum(-1)).sum(-1).clip(max=len(indices)-1)
            code |= 1 << indices[choice]
        truth = _execute(programs,code,presence,coords,p,offsets,u[...,-1])
        linked = truth[:,a]^truth[:,b]
        duplicated = (presence[:,duplicate_a]&presence[:,duplicate_b]).sum(1)
        mass = np.exp(1.3*(linked.sum(1)-len(a))-penalty*duplicated)[:,None]*linked
        left = logit(((truth[:,a]*mass).sum(0)+.75)/(mass.sum(0)+1.5))
        score = np.empty(len(programs))
        score[a],score[b] = left,-left
        options[name] = score
    return options


if __name__=="__main__":
    root = Path("artefacts/research")
    frame,scenes,captions,programs = read_rows("val")
    ensemble = {name:np.zeros(len(frame)) for name in OPTIONS}
    for variant in ("silhouette","seams","counts"):
        with np.load(root/f"scene_{variant}.npz") as f:
            assert np.array_equal(f["ids"],frame.id)
            table = dict(f)
        out = {name:np.zeros(len(frame)) for name in OPTIONS}
        for i,ix in enumerate(scenes.values()):
            keep = table["valid"][i]
            subset = frame.iloc[ix]
            pairs = find_twins(subset.image,[captions[j] for j in ix],subset.category)
            result = run_scene(
                [programs[j] for j in ix],pairs,[table[key][i,keep] for key in FIELDS],
                table["p"][i,keep],table["xy"][i,keep],table["features"][i,keep,8:12],table["offsets"]
            )
            for key,s in result.items():
                out[key][ix] = s
        print(variant,{key:roc_auc_score(frame.label,s) for key,s in out.items()},flush=True)
        for key,s in out.items():
            ensemble[key] += s/3
    print("ensemble",{key:roc_auc_score(frame.label,s) for key,s in ensemble.items()},flush=True)
    np.savez(root/"posterior_temperatures.npz",ids=frame.id,**ensemble)
