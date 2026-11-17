"""Test native-center detections as independent evidence for unresolved objects."""

from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

sys.path.insert(0, str(Path.cwd()))
from native_centers import CenterDetector, FIELDS, GROUPS, HEIGHT, STRIDE, WIDTH, read_rows
from src.pixel_regions import read_rgb
from src.scene_posterior import sample_scene_scores
from src.twins import find_twins


ROOT = Path("artefacts/research")


@torch.no_grad()
def detect(model, names, split="val"):
    for start in range(0, len(names), 32):
        pixels = [read_rgb(Path("data") / split / name) for name in names[start:start+32]]
        images = torch.as_tensor(np.stack(pixels)).permute(0,3,1,2).float()/255
        raw = model(images.contiguous(memory_format=torch.channels_last))
        p = raw[:,0].sigmoid()
        peaks = (p == F.max_pool2d(p[:,None],3,1,1)[:,0]) & (p > .56)
        for i in range(len(raw)):
            ys,xs = torch.where(peaks[i])
            chosen = p[i,ys,xs].argsort(descending=True)[:15]
            ys,xs = ys[chosen],xs[chosen]
            pos = (raw[i,1:3,ys,xs].clip(-.5,1.5)+torch.stack((xs,ys)))*STRIDE
            extent = raw[i,3:5,ys,xs].clip(-2,1.6).exp()*24
            attr = [z.softmax(0).T.numpy() for z in raw[i,5:,ys,xs].split([len(g) for g in GROUPS])]
            yield p[i,ys,xs].numpy(), pos.T.numpy(), extent.T.numpy(), attr


def fuse(box, existence, xy, attr, evidence, level, new_only):
    if not len(box):
        return existence, xy, attr
    size = box[:,2:]
    middle = box[:,:2] + .5*size
    p, position, extent, groups = evidence
    out_p, out_xy = list(existence), list(xy)
    out_attrs = [list(a) for a in attr]
    for j in np.flatnonzero(p > level):
        center = position[j]
        distance = np.linalg.norm(middle - center, axis=1)
        near = distance <= np.maximum(6, .35*np.minimum(size,extent[j]).max(1))
        if (near & (existence > .65)).any():
            continue
        colour = attr[1][:,groups[1][j].argmax()]
        supported = near & (colour > .28)
        if supported.any() and not new_only:
            best = np.argmax(np.where(supported, colour+existence, -np.inf))
            out_p[best] = max(out_p[best], .50 + .29*(p[j]-.60))
        elif not near.any() or (not new_only and not (near & (existence > .22)).any()):
            out_p.append(.30 + .35*(p[j]-.56))
            ground = center + [0,.38*extent[j,1]]
            out_xy.append(ground/[WIDTH,HEIGHT]*2-1)
            for result, attributes in zip(out_attrs, groups):
                result.append(attributes[j])
    return np.asarray(out_p), np.asarray(out_xy), [np.asarray(group) for group in out_attrs]


def main(epoch):
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    model = CenterDetector().to(memory_format=torch.channels_last).eval()
    model.load_state_dict(torch.load(ROOT/f"native_epoch{epoch}.pt", weights_only=True,map_location="cpu"))
    frame, groups, captions, programs = read_rows("val")
    options = {"control": (1.,False), "strict": (.72,False), "permissive": (.57,False),
               "strict_new": (.72,True)}
    out = {k: np.empty(len(frame)) for k in options}
    with np.load(ROOT/"scene_seams.npz") as archive:
        table = dict(archive)
    assert np.array_equal(table["ids"], frame.id)
    for i, (rows, evidence) in enumerate(zip(groups.values(),detect(model,list(groups)))):
        keep = table["valid"][i]
        attr = [table[key][i,keep] for key in FIELDS]
        xy, p = table["xy"][i,keep], table["p"][i,keep]
        boxes = table["features"][i,keep,8:12]*[WIDTH,HEIGHT,WIDTH,HEIGHT]
        local = [programs[j] for j in rows]
        pairs = find_twins(frame.image.iloc[rows], [captions[j] for j in rows], frame.category.iloc[rows])
        for key, (level, new_only) in options.items():
            pp, zz, aa = fuse(boxes,p,xy,attr,evidence,level,new_only)
            out[key][rows] = sample_scene_scores(local,pairs,pp,list(zip(GROUPS,aa)),zz,table["offsets"],42)
    old_parts = [np.load(ROOT/f"region_{name}.npz")["joint"] for name in ("primitives","count_supervision")]
    for name, scores in out.items():
        print(epoch, name, roc_auc_score(frame.label,scores),
              "ensemble",roc_auc_score(frame.label,(scores+sum(old_parts))/3),flush=True)
    np.savez(ROOT/f"center_proposals{epoch}.npz",ids=frame.id,**out)


if __name__ == "__main__":
    main(int(sys.argv[1]))
