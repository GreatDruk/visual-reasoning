"""Compare center-detector fine-tuning with reviewed or contour-derived ROI counts."""

import argparse
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path.cwd()))
from crop_encoder_probe import read_cases
from native_centers import CenterDetector, GRID, ROOT, STRIDE, count_loss, evaluate, read_rgb, read_rows, training_counts
from overlap_review import COLORS

ART = Path("artefacts/research")
CHECKPOINT = ART / "native_natural_epoch21.pt"


def targets(frame, split):
    rows = frame[frame.split == split].reset_index(drop=True)
    samples, ranks, rects, real = [], [], [], []
    for row in rows.itertuples(index=False):
        samples.append(read_rgb(ROOT / split / row.image))
        ranks.append(COLORS.index(row.color))
        rects.append((row.x0,row.y0,row.x1,row.y1))
        real.append([row.shapes.count(c) for c in "bsc"])
    return (rows, torch.from_numpy(np.stack(samples)).permute(0,3,1,2).float()/255,
            torch.tensor(ranks), torch.tensor(rects), torch.tensor(real, dtype=torch.float32))


def proxies(rows, hue, bounds):
    with np.load(ART / "teacher_train.npz") as data:
        order = {n:i for i,n in enumerate(data["names"])}
        f, p, valid, color, shape = (data[k] for k in ("features", "p", "valid", "color", "shape"))
    out = []
    for name, column, box in zip(rows.image, hue.numpy(), bounds.numpy(), strict=True):
        i = order[name]
        center = (f[i,:,8:10]+.5*f[i,:,10:12])*[240,160]
        inside = ((center>=box[:2]) & (center<box[2:])).all(-1)
        use = valid[i] & (p[i]>.55) & (color[i].argmax(1)==column) & inside
        out.append(np.bincount(shape[i,use].argmax(1),minlength=3))
    return torch.as_tensor(np.asarray(out), dtype=torch.float32)


def counts(raw, hue, bounds, hard=False):
    heat = raw[:,0]
    peaks = heat == F.max_pool2d(heat[:,None], 3, 1, 1)[:,0]
    gx = (torch.arange(GRID[1])[None,None,:] + raw[:,1].clamp(-.5,1.5))*STRIDE
    gy = (torch.arange(GRID[0])[None,:,None] + raw[:,2].clamp(-.5,1.5))*STRIDE
    ax,ay,bx,by = bounds.T[:,:,None,None]
    area = (gx>=ax)&(gy>=ay)&(gx<bx)&(gy<by)
    occ = peaks & area
    color = raw[:,7:15].softmax(1)
    shape = raw[:,-3:].softmax(1)
    if hard:
        occ = occ & (heat>0) & (color.argmax(1)==hue[:,None,None])
        return occ & area, shape
    p = (heat/.65).sigmoid() * occ * color[torch.arange(len(raw)),hue]
    return (shape * p[:,None]).sum((2,3))


def load_model():
    net = CenterDetector().to(memory_format=torch.channels_last)
    net.load_state_dict(torch.load(CHECKPOINT, weights_only=True, map_location="cpu"))
    return net.eval()


class ShiftedObjectness(torch.nn.Module):
    def __init__(self, detector, shift):
        super().__init__()
        self.detector = detector
        self.shift = shift

    def forward(self, pixels):
        out = self.detector(pixels)
        out[:,0] = out[:,0] + self.shift
        return out


@torch.inference_mode()
def score_crops(net, frame, rgb, hue, bounds):
    predictions, hard_counts = [], []
    for idx in torch.arange(len(rgb)).split(16):
        raw = net(rgb[idx].contiguous(memory_format=torch.channels_last))
        expected = counts(raw, hue[idx], bounds[idx]).sum(-1)
        hard, _ = counts(raw, hue[idx], bounds[idx], hard=True)
        predictions.append(expected)
        hard_counts.append(hard.sum((1,2)))
    estimate = torch.cat(predictions).numpy()
    discrete = torch.cat(hard_counts).numpy()
    true = frame.number.to_numpy()
    return {"auc_one_two": roc_auc_score(true==2,estimate),
            "exact": float((discrete==true).mean()), "mae": float(np.abs(discrete-true).mean()),
            "under_count": int((discrete<true).sum()), "over_count": int((discrete>true).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-counts", action="store_true")
    parser.add_argument("--reviewed-only", action="store_true")
    parser.add_argument("--bias-only", action="store_true")
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    torch.set_num_threads(3)
    review = read_cases()
    review = review[review.number.isin((1,2))]
    train, inputs, hue, boxes, true = targets(review,"train")
    val, val_rgb, val_hue, val_boxes, _ = targets(review,"val")
    assert not set(train.image)&set(val.image)
    pseudo = proxies(train,hue,boxes)
    baseline = load_model()
    with torch.inference_mode():
        full_teacher = torch.cat([baseline(t.contiguous(memory_format=torch.channels_last)) for t in inputs.split(12)])
    full_teacher = full_teacher.clone()
    data = read_rows("val")
    full = (*data,np.stack([read_rgb(ROOT/"val"/name) for name in data[1]]))
    report = {"before":score_crops(baseline,val,val_rgb,val_hue,val_boxes),
              "reviewed_train":len(train),"reviewed_val":len(val)}
    if args.bias_only:
        @torch.inference_mode()
        def objective(shift):
            output = full_teacher.clone()
            output[:,0] += shift
            return float(F.smooth_l1_loss(counts(output,hue,boxes).sum(1),true.sum(1)))

        fit = minimize_scalar(objective, method="bounded", bounds=(-2.,2.))
        model = ShiftedObjectness(baseline,float(fit.x))
        report["train_fit_logit_shift"] = float(fit.x)
        report["bias_only"] = score_crops(model,val,val_rgb,val_hue,val_boxes)
        evaluate(model,torch.device("cpu"),full,"roi_bias")
        print(json.dumps(report,indent=2))
        (ART/"reviewed_center_probe_bias.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        return
    weak = training_counts() if args.exact_counts else None
    arms = (("reviewed",true),) if args.reviewed_only or weak is not None else (("pseudo",pseudo),("reviewed",true))
    for mode,labels in arms:
        suffix = "_exact" if weak is not None else ("" if args.seed == 71 else f"_repeat{args.seed}")
        if (ART / "overlap_train_controls/review_labels.csv").is_file():
            suffix += "_independent"
        model_name = f"{mode}{suffix}"
        model = load_model()
        model.features.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.head.parameters(),lr=.0004)
        generator = torch.Generator().manual_seed(args.seed)
        auxiliary = np.random.default_rng(841)
        for _ in range(8):
            for ix in torch.randperm(len(train),generator=generator).split(12):
                raw = model(inputs[ix].contiguous(memory_format=torch.channels_last))
                p = counts(raw,hue[ix],boxes[ix])
                loss = F.smooth_l1_loss(p.sum(1),labels[ix].sum(1)) + .5*F.smooth_l1_loss(p,labels[ix])
                loss += .08 * F.mse_loss(raw,full_teacher[ix])
                if weak is not None:
                    images, scene, filters, target = weak
                    selected = auxiliary.integers(len(target), size=10)
                    pixels = torch.tensor(images[scene[selected]]).permute(0,3,1,2).float()/255
                    weak_raw = model(pixels.contiguous(memory_format=torch.channels_last))
                    loss += .2*count_loss(weak_raw, torch.as_tensor(filters[selected]),
                                          torch.as_tensor(target[selected]))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        report[model_name] = score_crops(model,val,val_rgb,val_hue,val_boxes)
        torch.save(model.state_dict(), ART / f"native_roi_{model_name}.pt")
        evaluate(model,torch.device("cpu"), full, f"roi_{model_name}")
    print(json.dumps(report,indent=2))
    (ART/f"reviewed_center_probe{suffix}.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
