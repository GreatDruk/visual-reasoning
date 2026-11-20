"""Compare detector grid phases at inference and refit the union head on train + val."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.research import occlusion_verifier as v


@torch.inference_mode()
def extract(split, shard, device, single=False):
    names = pd.read_csv(v.DATA / f"{split}.csv").image.unique()[shard::2]
    models = [v.ResNetCenters(full=True), v.CenterDetector()]
    heads = []
    for model, key in zip(models, ("r50_img","thaw")):
        model.load_state_dict(v.load(v.PRIORS / f"{key}.pt"))
        model.to(device, memory_format=torch.channels_last).eval()
        heads.append(v.model_head(key, device))
    matrix = v.projection().to(device)
    phases = [(0,0)] if split == "train" or single else [(0,0),(2,0),(0,2),(2,2)]
    parts = {s:[] for s in phases}
    with ThreadPoolExecutor(3) as pool:
        for start in range(0,len(names),128):
            images = np.stack(list(pool.map(v.read_rgb,[v.DATA/split/n for n in names[start:start+128]])))
            rgb = torch.from_numpy(images).to(device).permute(0,3,1,2).float().div_(255)
            native = v.detector_outputs(models[1],heads[1],rgb,matrix)
            for dx,dy in phases:
                img = F.pad(rgb,(dx,0,dy,0),mode="replicate")[:,:,:v.HEIGHT,:v.WIDTH] if dx or dy else rgb
                r50 = v.detector_outputs(models[0],heads[0],img)
                shift = np.array([dx/v.WIDTH,dy/v.HEIGHT])
                r50[:,:,-6:-4] -= shift
                r50[:,:,-2:] -= 2*shift
                rows = [v.merge(a,b) for a,b in zip(r50,native)]
                parts[dx,dy].append(tuple(map(np.stack,zip(*rows))))
    for (dx,dy), stacks in parts.items():
        x,xy,valid = (np.concatenate(col) for col in zip(*stacks))
        np.savez(v.CACHE/f"{split}_{dx}{dy}_{shard}.npz",names=names.astype(str),x=x,xy=xy,valid=valid)


def assemble(split,phase):
    tables = v.CACHE/"tables"
    tables.mkdir(exist_ok=True)
    with np.load(v.CACHE/f"{split}_{phase}_0.npz") as a,np.load(v.CACHE/f"{split}_{phase}_1.npz") as b:
        merged = {}
        for key in a.files:
            x,y=a[key],b[key]
            dest=np.empty((len(x)+len(y),*x.shape[1:]),dtype=np.result_type(x.dtype,y.dtype))
            dest[::2],dest[1::2]=x,y
            merged[key]=dest
        np.savez(tables/f"{split}_{phase}.npz",**merged)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("extract","train","infer"))
    p.add_argument("split")
    p.add_argument("number",type=int)
    p.add_argument("phase",default="00",nargs="?")
    p.add_argument("fit",default="train",nargs="?")
    args=p.parse_args()
    _,device=v.setup(args.number)
    v.CACHE.mkdir(exist_ok=True)
    if args.action=="extract":
        extract(args.split,args.number,device,args.phase=='single')
    else:
        v.OUT=Path("verify_result")/args.fit
        v.OUT.mkdir(parents=True,exist_ok=True)
        if args.action=="train":
            # SceneData reads the train.csv and table under this isolated training root.
            v.DATA=Path("fit")/args.fit
            v.CACHE=v.DATA
            v.train("control",args.number,12,device)
        else:
            v.DATA=Path("phase")/args.phase
            v.CACHE=v.DATA
            v.infer(args.split,"control",args.number,device)
            path=v.OUT/f"control_{args.number}_{args.split}.npz"
            path.replace(v.OUT/f"{args.number}_{args.split}_{args.phase}.npz")
