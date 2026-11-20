"""Recreate detector tables and audit validation errors in the Kaggle ensemble."""

import argparse
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_suite_common import dense_bank, N_ATTRS, table_predictions
from kaggle_suite_detectors import native_forward, projection
from learned_center_fusion import Fusion
from native_centers import CenterDetector, FIELDS, GROUPS, HEIGHT, WIDTH, read_rgb, read_rows
from resnet_centers_ablation import ResNetCenters
from src.models.slot_program import EXCLUSIVE_GROUPS
from src.scene_posterior import _draws, _execute
from src.twins import find_twins


RUN = Path("artefacts/kaggle_suite/run_2026-11-18")
DATA = Path("data")
PRIVATE = Path("artefacts/kaggle_suite/clevr_kaggle_inputs.zip")


def saved_scores(split, name, frame):
    with zipfile.ZipFile(RUN / "clevr_scores.zip") as archive:
        with np.load(io.BytesIO(archive.read(f"{name}_{split}.npz"))) as scores:
            if not np.array_equal(scores["ids"], frame.id):
                raise ValueError(f"{name}/{split} score IDs do not match")
            return scores["score"]


def weight(name):
    with zipfile.ZipFile(RUN / "clevr_weights.zip") as archive:
        return torch.load(io.BytesIO(archive.read(f"{name}.pt")), map_location="cpu", weights_only=True)


def inspect_candidates(parts, table, names):
    p, xy, valid, *attrs = parts
    boxes = table[0][..., 1+N_ATTRS:5+N_ATTRS] * [WIDTH, HEIGHT, WIDTH, HEIGHT]
    scene_result = {}
    for i, name in enumerate(names):
        entries = []
        for j in range(p.shape[1]):
            candidate = {}
            detector_p = float(expit(table[0][i,j,0]))
            for key, labels, arr in zip(FIELDS, EXCLUSIVE_GROUPS, attrs):
                indices = arr[i,j].argsort()[:-3:-1]
                candidate[key] = {labels[k]: round(float(arr[i,j,k]), 3) for k in indices}
            if max(float(p[i,j]), detector_p) >= .015:
                entries.append({"xywh": boxes[i,j].round(1).tolist(), "ground": xy[i,j].round(3).tolist(),
                                "p": round(float(p[i,j]), 3), "used": bool(valid[i,j]),
                                "detector_p": round(detector_p, 3), **candidate})
        scene_result[name] = entries
    return scene_result


def decode(programs, pairs, p, attrs, xy, offsets, *, draws=4096, seed=42):
    if not len(p):
        return np.zeros((3, len(programs)))
    u = _draws(len(p), 8, seed, draws)
    present = u[...,0] < p
    codes = np.zeros_like(present, dtype=np.int32)
    for j, (words, probs) in enumerate(zip(GROUPS, attrs), 1):
        choices = (u[...,j,None] > probs.cumsum(-1)).sum(-1).clip(max=len(words)-1)
        codes |= 1 << words[choices]
    xy = xy[None] + logit(u[...,-3:-1]) * .026
    truth = _execute(programs, codes, present, xy, p, offsets, u[...,-1])
    left, right = pairs.T
    link = truth[:,left] ^ truth[:,right]
    counts = link.sum(1)
    values = []
    for strength in (1.3, 2.6, 5.2):
        # Reuse the same sample to isolate the consistency weight.
        weights = np.exp(strength*(counts-len(pairs)))[:,None]*link
        l = logit(((truth[:,left]*weights).sum(0)+.75)/(weights.sum(0)+1.5))
        scores = np.zeros(len(programs))
        scores[left], scores[right] = l, -l
        values.append(scores)
    return np.stack(values)


def head(name, device, crops=False, *, weights=weight):
    teacher = weights("scene_silhouette")
    teacher["relation_offset"].zero_()
    model = Fusion(teacher)
    if crops:
        model.residual[0] = torch.nn.Linear(model.residual[0].in_features + 128, 48)
    model.load_state_dict(weights(name))
    return model.to(device).eval()


def pixels(split, names, device, data=DATA):
    rgb = np.stack([read_rgb(data / split / name) for name in names])
    return torch.from_numpy(rgb).to(device).permute(0,3,1,2).float().div_(255).contiguous(
        memory_format=torch.channels_last)


@torch.inference_mode()
def resnet(split, batch_size, device):
    data = read_rows(split)
    frame, groups, captions, parsed = data
    names = list(groups)
    model = ResNetCenters(full=True).to(device, memory_format=torch.channels_last).eval()
    model.load_state_dict(weight("r50_img"))
    predictor = head("r50_img_head", device)
    offsets = predictor.relation_offset.detach().cpu().numpy()
    control, wide, tight, strict, mirror = [np.zeros(len(frame)) for _ in range(5)]
    cards = {}
    r = saved_scores(split, "r50_img", frame)
    s = saved_scores(split, "sigft", frame)
    if split == "val":
        failed = frame.loc[((r+s)<0) & frame.label.eq(1), "image"]
        focus = set(failed)
    else:
        focus = set()

    from src.scene_posterior import sample_scene_scores
    for start in range(0, len(names), batch_size):
        seq = names[start:start+batch_size]
        batch = pixels(split, seq, device)
        original = dense_bank(model(batch).float())
        flipped = dense_bank(model(batch.flip(-1)).float())
        # Convert mirrored observations back to the original image's coordinates.
        xf, yf, keep = flipped
        xf[...,1+N_ATTRS] = 1 - xf[...,1+N_ATTRS] - xf[...,3+N_ATTRS]
        yf[...,0] *= -1
        original = tuple(a.cpu().numpy() for a in original)
        flipped = tuple(a.cpu().numpy() for a in flipped)
        banks = [table_predictions(predictor, source, device) for source in (original,flipped)]
        for name in set(seq) & focus:
            j = seq.index(name)
            sl = slice(j, j+1)
            cards[name] = inspect_candidates([x[sl] for x in banks[0]], [x[sl] for x in original], [name])[name]
        for j, name in enumerate(seq):
            idx = groups[name]
            programs = [parsed[k] for k in idx]
            pairs = find_twins(frame.image.iloc[idx], [captions[k] for k in idx], frame.category.iloc[idx])
            p, xy, valid, *attr = banks[0]
            mask = valid[j]
            control[idx] = sample_scene_scores(programs, pairs, p[j,mask],
                      [(word, a[j,mask]) for word,a in zip(GROUPS,attr)], xy[j,mask], offsets, 42)
            wide[idx], tight[idx], strict[idx] = decode(programs,pairs,p[j,mask],
                        [a[j,mask] for a in attr],xy[j,mask],offsets)
            p, xy, valid, *attr = banks[1]
            mask = valid[j]
            mirror[idx] = sample_scene_scores(programs, pairs, p[j,mask],
                      [(word, a[j,mask]) for word,a in zip(GROUPS,attr)], xy[j,mask], offsets, 42)
        if start % (batch_size * 10) == 0:
            print("resnet", split, f"{start+len(seq)}/{len(names)}", flush=True)
    values = dict(control=control, wide=wide, tight=tight, strict=strict, mirror=mirror)
    np.savez(RUN / f"audit_resnet_{split}.npz", ids=frame.id.to_numpy(), **values)
    if split=="val":
        for key,v in values.items():
            scores=(v+s)/2
            print(key, "AUC", roc_auc_score(frame.label,scores), "errors",
                  int(((scores<0)&frame.label.eq(1)).sum()), flush=True)
        (RUN/"audit_resnet_objects.json").write_text(json.dumps(cards, indent=2), encoding="utf-8")


@torch.inference_mode()
def crops(limit, device):
    from kaggle_suite_crops import active_crops, crop_features, extract_head
    from transformers import SiglipVisionModel

    frame, groups, captions, parsed = read_rows("val")
    original = saved_scores("val", "r50_img", frame)
    prior = saved_scores("val", "sigft", frame)
    bad = frame[(original+prior<0)&frame.label.eq(1)].copy()
    bad["margin"] = (original+prior)[bad.index]/2
    names = list(dict.fromkeys(bad.sort_values("margin").image))[:limit]

    with tempfile.TemporaryDirectory(dir=RUN) as directory, zipfile.ZipFile(PRIVATE) as archive:
        vision_path=Path(directory)
        for file in ("model.safetensors", "config.json"):
            with archive.open("weights_in/siglip2_vision/"+file) as source, (vision_path/file).open("wb") as target:
                shutil.copyfileobj(source,target)
        backbone=SiglipVisionModel.from_pretrained(vision_path, local_files_only=True).to(
            device=device, dtype=torch.float32).eval()
    pool=extract_head(backbone, device)
    # Keep the official pool on the trunk. Load the adapted pool into a separate module.
    import copy
    pool=copy.deepcopy(pool)
    pool.load_state_dict(weight("siglip_pool"))
    pool.eval()
    model=CenterDetector().to(device,memory_format=torch.channels_last).eval()
    model.load_state_dict(weight("thaw"))
    matrix=projection().to(device)
    images=pixels("val",names,device)
    table=[a.cpu().numpy() for a in native_forward(model,images,matrix)]
    x,xy,valid=table
    scene, slot=np.nonzero(active_crops(x,valid))
    boxes=x[scene,slot,1+N_ATTRS:5+N_ATTRS]
    with zipfile.ZipFile(RUN/"clevr_weights.zip") as z:
        with np.load(io.BytesIO(z.read("sigft_pca.npz"))) as a:
            mean,components,scale=[a[k] for k in ("mean","components","scale")]
    extra=np.zeros((*valid.shape,128),dtype="f")
    for start in range(0,len(scene),4):
        part=slice(start,start+4)
        box=torch.from_numpy(boxes[part]).to(device)
        axis=(torch.arange(256,device=device)+.5)/256
        yy,xx=torch.meshgrid(axis,axis,indexing="ij")
        grid=box[:,None,None,:2]+torch.stack((xx,yy),dim=-1)*box[:,None,None,2:]
        crop=torch.nn.functional.grid_sample(images[scene[part]],grid*2-1,align_corners=False,padding_mode="border")
        _, tokens=crop_features(backbone,crop*2-1)
        vec=pool(tokens.float()).half().float().cpu().numpy()
        extra[scene[part],slot[part]]=(((vec-mean)@components.T)/scale).clip(-7,7)
        print("sigft objects",min(start+4,len(scene)),"/",len(scene),flush=True)
    predictor=head("sigft",device,crops=True)
    bank=table_predictions(predictor,(np.concatenate((x,extra),axis=-1),xy,valid),device)
    records=inspect_candidates(bank,table,names)
    print("sigft inspected",len(names),"scenes",flush=True)
    (RUN/"audit_sigft_objects.json").write_text(json.dumps(records,indent=2),encoding="utf-8")


@torch.inference_mode()
def rescue(split, batch, device, *, weights=weight, scores_from=saved_scores, output_adjustment=False,
           data_path=DATA):
    """Check whether both image models support a weak object before changing its score."""
    from src.scene_posterior import sample_scene_scores

    if data_path==DATA:
        frame, groups, captions, parsed = read_rows(split)
    else:
        from native_centers import canonical_form, parse_program
        frame=pd.read_csv(data_path/f"{split}.csv")
        groups=frame.groupby("image",sort=False).indices
        captions=[canonical_form(row.caption,row.category) for row in frame.itertuples()]
        parsed=[parse_program(caption) for caption in captions]
    names = list(groups)
    primary = ResNetCenters(full=True).to(device, memory_format=torch.channels_last).eval()
    primary.load_state_dict(weights("r50_img"))
    secondary = CenterDetector().to(device, memory_format=torch.channels_last).eval()
    secondary.load_state_dict(weights("thaw"))
    readout = head("r50_img_head", device, weights=weights)
    secondary_head = head("thaw_head", device, weights=weights)
    transform = projection().to(device)
    offsets = readout.relation_offset.detach().cpu().numpy()
    archived = scores_from(split,"r50_img",frame) if split == "test" else np.zeros(len(frame))
    scores = np.tile(archived, (3,1))
    affected = []
    extent = np.array([WIDTH, HEIGHT])
    for start in range(0, len(names), batch):
        seq = names[start:start+batch]
        rgb = pixels(split,seq,device,data_path)
        t1 = tuple(t.cpu().numpy() for t in native_forward(primary,rgb))
        t2 = tuple(t.cpu().numpy() for t in native_forward(secondary,rgb,transform))
        p1, xy1, mask1, *attrs1 = table_predictions(readout,t1,device)
        p2, xy2, mask2, *attrs2 = table_predictions(secondary_head,t2,device)
        q1,q2 = expit(t1[0][...,0]),expit(t2[0][...,0])
        boxes1 = t1[0][...,1+N_ATTRS:5+N_ATTRS]
        boxes2 = t2[0][...,1+N_ATTRS:5+N_ATTRS]
        for i,name in enumerate(seq):
            idx = groups[name]
            programs = [parsed[k] for k in idx]
            pairs = find_twins(frame.image.iloc[idx], [captions[k] for k in idx], frame.category.iloc[idx])
            current = [(p1[i,mask1[i]],xy1[i,mask1[i]],[a[i,mask1[i]] for a in attrs1])]
            left, right = boxes1[i], boxes2[i]
            mid1,mid2 = (left[:,:2]+.5*left[:,2:])*extent, (right[:,:2]+.5*right[:,2:])*extent
            size1,size2 = left[:,2:]*extent,right[:,2:]*extent
            d = np.linalg.norm(mid1[:,None]-mid2[None],axis=-1)
            limit = np.clip(.57*np.minimum(size1.max(1)[:,None],size2.max(1)),7.,13.)
            color = attrs1[1][i]@attrs2[1][i].T
            shape = attrs1[-1][i]@attrs2[-1][i].T
            trusted = (p2[i]>.75)&(q2[i]>.50)
            supported = (d<limit)&((color>.47)|(shape>.53))&trusted[None]
            get = np.argmax(np.where(supported,p2[i][None] * (1-.40*d/limit),-1.),axis=1)
            supported = supported.any(1)
            separated = np.linalg.norm(mid1[:,None]-mid1[None],axis=-1)
            matches_strong = (p1[i]>.65)[None] & (separated<np.clip(.40*size1.max(1)[:,None],6,12))
            take = np.flatnonzero(supported & ~matches_strong.any(1) & (q1[i]>.025) & (p1[i]<.35))
            presence = p1[i].copy()
            xy = xy1[i].copy()
            attrs = [a[i].copy() for a in attrs1]
            valid = mask1[i].copy()
            used = set()
            for j in take:
                k = get[j]
                if k in used:
                    continue
                used.add(k)
                valid[j] = True
                presence[j] = .85 * p2[i,k]
                xy[j] = xy2[i,k]
                for a,b in zip(attrs,attrs2):
                    a[j] = .40 * a[j] + .60 * b[i,k]
            current.append((presence[valid],xy[valid],[a[valid] for a in attrs]))

            # Match weak proposals near a stronger foreground candidate. A corroborating
            # candidate can lift objectness while its original attributes stay uncertain.
            joint_support = (d < limit) & (q2[i][None] >= .11)
            upper = left[:,:2]+left[:,2:]
            overlap = (np.minimum(upper[:,None],upper)-np.maximum(left[:,None,:2],left[:,:2])).clip(0).prod(-1)
            covers = overlap / left[:,2:].prod(-1).clip(1e-6)[:,None]
            hidden = ((covers>.12)&(covers<.86)&(mid1[:,1,None]<mid1[:,1])&(p1[i][None]>.85)).any(1)
            weak = valid & (q1[i]>=.10) & (presence<.40) & ~matches_strong.any(1) & hidden & joint_support.any(1)
            deeper = presence.copy()
            deeper[weak] = np.maximum(deeper[weak],.52)
            current.append((deeper[valid],xy[valid],[a[valid] for a in attrs]))
            if used or weak.any():
                affected.append(name)
            elif split == "test":
                # All three variants agree. The eventual adjustment is exactly zero.
                continue
            for j,(presence,position,attr) in enumerate(current):
                scores[j,idx] = sample_scene_scores(programs,pairs,presence,list(zip(GROUPS,attr)),position,offsets,42)
        if start % (batch * 12) == 0:
            print("rescue", split, f"{start+len(seq)}/{len(names)}",flush=True)
    if output_adjustment:
        return scores[2]-scores[0]
    np.savez(RUN/f"audit_rescue_{split}.npz",ids=frame.id.to_numpy(),
             control=scores[0],borrow=scores[1],boost=scores[2],affected=affected)
    if split == "val":
        prior = saved_scores(split,"sigft",frame)
        for name,score in zip(("control","borrow","boost"),scores):
            mean = (score+prior)/2
            print(name,roc_auc_score(frame.label,mean),int(((mean<0)&frame.label.eq(1)).sum()),flush=True)
    else:
        from kaggle_suite import write_csv

        prior = saved_scores(split, "sigft", frame)
        sample = pd.read_csv(DATA / "sample_submission.csv")
        target, = [column for column in sample if column != "id"]
        # Keep the GPU scores as the anchor and apply half of the measured adjustment.
        prediction = .5 * (archived + prior) + .25 * (scores[2] - scores[0])
        csv = write_csv(sample, frame.id.to_numpy(), prediction, target)
        path = RUN / "r50_sigft_occ.csv"
        temporary = path.with_suffix(".tmp")
        csv.to_csv(temporary, index=False, float_format="%.17g")
        written = pd.read_csv(temporary, float_precision="round_trip")
        if not written.equals(csv) or not written.id.equals(sample.id):
            raise ValueError("Output CSV did not round-trip")
        temporary.replace(path)
        print(path, len(written), "verified rows", flush=True)
    print("affected scenes",len(affected),flush=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode",choices=("resnet","crops","rescue"))
    cli.add_argument("--split",default="val",choices=("val","test"))
    cli.add_argument("--batch",type=int,default=16)
    cli.add_argument("--limit",type=int,default=8)
    cli.add_argument("--threads",type=int)
    args=cli.parse_args()
    torch.set_num_threads(args.threads or (4 if args.mode=="crops" else 3))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode=="resnet":
        resnet(args.split,args.batch,device)
    elif args.mode=="crops":
        crops(args.limit,device)
    else:
        rescue(args.split,args.batch,device)


if __name__=="__main__":
    main()
