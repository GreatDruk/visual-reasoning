"""Learn how much to trust center detections using only labelled train captions."""

from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logit
import torch
from torch import nn

sys.path.insert(0, str(Path.cwd()))
from center_proposals import detect
from native_centers import CenterDetector, FIELDS, GROUPS, HEIGHT, WIDTH, read_rows
from scripts import pixel_reasoning as pipeline
from src.models.slot_program import ATTRIBUTES, SlotProgramExecutor, log1mexp


ROOT = Path("artefacts/research")
CACHE = ROOT / "learned_fusion"
ATTRS = np.concatenate(GROUPS)
N_ATTRS = len(ATTRS)
FEATURES = 1 + N_ATTRS + 4 + 3 + N_ATTRS
MAX_REGIONS = 24


def log_probs(probabilities):
    return np.log(np.concatenate(probabilities,axis=1).clip(1e-5,1))


def region_features(boxes, positions, p, attr, learned):
    xy = boxes[:,:2]+.5*boxes[:,2:]
    prob, coords, extent, groups = learned
    base = np.column_stack((logit(p.clip(1e-5,1-1e-5)),log_probs(attr),boxes/[WIDTH,HEIGHT,WIDTH,HEIGHT]))
    evidence = np.zeros((len(p),3+N_ATTRS))
    unmatched = np.ones(len(prob),dtype=bool)
    if len(p) and len(prob):
        distance = np.linalg.norm(xy[:,None]-coords[None],axis=-1)
        near_scale = np.maximum(6,np.minimum(boxes[:,None,2:],extent[None]).max(-1) * .33)
        color = attr[1] @ groups[1].T
        shape = attr[-1] @ groups[-1].T
        cost = distance / near_scale + .26*(1-color) + .10*(1-shape)
        a, b = linear_sum_assignment(cost)
        keep = (distance[a,b] <= near_scale[a,b]) & (color[a,b]>=.25)
        a,b = a[keep],b[keep]
        unmatched[b] = False
        evidence[a] = np.column_stack((prob[b], distance[a,b]/near_scale[a,b], color[a,b],
                                       log_probs([group[b] for group in groups])))
        # A learned candidate already explained by a strong region should not
        # enter as a separate instance even if assignment used another peak.
        duplicate = ((p[:,None]>.65)&(distance < near_scale)).any(axis=0)
        unmatched &= ~duplicate
    x = np.column_stack((base,evidence))
    take = np.flatnonzero(unmatched & (prob > .59))[:max(0,MAX_REGIONS-len(p))]
    if len(take):
        ext = extent[take]
        mid = coords[take]
        extra_attr = log_probs([group[take] for group in groups])
        extra_base = np.column_stack((logit(.30+.35*(prob[take]-.56)),extra_attr,
                                      np.column_stack((mid-.5*ext,ext))/[WIDTH,HEIGHT,WIDTH,HEIGHT]))
        extra_evidence = np.column_stack((prob[take], np.ones((len(take),2)), extra_attr))
        x = np.concatenate((x,np.column_stack((extra_base,extra_evidence))))
        positions = np.vstack((positions,(mid+ext*[0,.38])/[WIDTH,HEIGHT]*2-1))
    result = np.zeros((MAX_REGIONS,FEATURES),np.float32)
    ground = np.zeros((MAX_REGIONS,2),np.float32)
    valid = np.arange(MAX_REGIONS) < len(x)
    result[valid],ground[valid] = x,positions
    return result,ground,valid


def build_cache(split, detector, teacher):
    _, scene_rows, _, _ = read_rows(split)
    names = list(scene_rows)
    source = ROOT/("teacher_train.npz" if split=="train" else "scene_silhouette.npz")
    with np.load(source) as archive:
        if split=="train":
            assert list(archive["names"])==names
        table = {key:archive[key] for key in ("features","xy","p","valid",*FIELDS)}
    if split=="train":
        w = teacher["project_position.weight"].numpy()
        b = teacher["project_position.bias"].numpy()
        table["xy"] += .14*np.tanh(table["features"] @ w.T + b)
    out = []
    for i, evidence in enumerate(detect(detector,names,split)):
        valid = table["valid"][i]
        box = table["features"][i,valid,8:12]*[WIDTH,HEIGHT,WIDTH,HEIGHT]
        attr = [table[key][i,valid] for key in FIELDS]
        out.append(region_features(box,table["xy"][i,valid],table["p"][i,valid],attr,evidence))
    x,xy,valid = [np.stack(col) for col in zip(*out)]
    np.savez(CACHE/f"{split}.npz",names=names,x=x,xy=xy,valid=valid)
    print(split, len(names), "union proposals", int(valid.sum()),flush=True)


class Fusion(SlotProgramExecutor):
    def __init__(self, teacher):
        super().__init__(1,FEATURES,dim=48,hidden=64,dropout=.05,use_slot_vectors=False,
                         reference_mode="marginal",exclusive_attributes=True,require_reference_exists=True,
                         comparison_mode="coupled")
        self.residual = nn.Sequential(nn.Linear(FEATURES,48),nn.SiLU(),nn.Dropout(.08),nn.Linear(48,3+N_ATTRS))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.register_buffer("attribute_indices",torch.as_tensor(ATTRS.copy()))
        with torch.no_grad():
            for name in ("log_reference_temperature","log_sharpness","relation_offset"):
                getattr(self,name).copy_(teacher[name])

    def encode(self, features, positions, valid):
        residual = self.residual(features)
        return {"pooled":features,"position":positions+.08*residual[...,-2:].tanh(),
                "residual":residual,"valid":valid}

    def slot_logits(self, enc):
        x, change = enc["pooled"],enc["residual"]
        p = (x[...,0] + 3*change[...,0].tanh()).masked_fill(~enc["valid"],-30.)
        revised = x[...,1:1+N_ATTRS] + 2*change[...,1:1+N_ATTRS].tanh()
        logits = x.new_zeros((*x.shape[:-1],len(ATTRIBUTES)+1)).index_copy(-1,self.attribute_indices,revised)
        return p,logits

    def score(self, features, positions, valid, indices, programs):
        truth,_,_ = self(self.encode(features,positions,valid),indices,programs)
        return truth-log1mexp(truth)

    def existence_loss(self, features, positions, valid, indices, programs):
        enc = self.encode(features,positions,valid)
        obj,words = self.slot_logits(enc)
        match = self.membership(obj[indices],words[indices],programs["a"],exclusive=True)
        return -log1mexp(log1mexp(match).sum(-1)).mean()


if __name__=="__main__":
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    density = CenterDetector().to(memory_format=torch.channels_last).eval()
    epoch = int(sys.argv[1]) if len(sys.argv)>1 else 8
    tag = "" if epoch == 8 else str(epoch)
    if tag:
        CACHE = ROOT / f"learned_fusion{tag}"
    density.load_state_dict(torch.load(ROOT/f"native_epoch{epoch}.pt",weights_only=True,map_location="cpu"))
    silhouette = torch.load(ROOT/"scene_silhouette.pt",weights_only=True,map_location="cpu")
    CACHE.mkdir(exist_ok=True)
    if "--reuse-cache" not in sys.argv[2:]:
        for split in ("train","val"):
            build_cache(split,density,silhouette)
    pipeline.ImageTable = lambda *_: Fusion(silhouette)
    pipeline.run(Path("data"),epochs=8,batch_size=64,cache_dir=CACHE,posterior="joint",boundary=True,
                 device_name="cpu",report_path=ROOT/f"region_learned_fusion{tag}.json")
