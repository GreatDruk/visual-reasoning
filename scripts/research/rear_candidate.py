"""Verify weak rear-object proposals using paired prompts and real caption counts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
import torch
from torch import nn
from torch.nn import functional as F

from scripts.research.fragment_votes import EXTRA, Scenes, v
from kaggle_suite_common import amp, table_predictions, tune_batch
from native_centers import GROUPS, canonical_form, parse_program
from src.models.slot_program import EXACT, WORD_INDEX
from src.scene_posterior import _draws, _execute
from src.twins import find_twins

SIDE = 80
SCALE = np.array([v.WIDTH, v.HEIGHT, v.WIDTH, v.HEIGHT])
MAX_REAR = 4
SCRATCH = Path("rear_cache")
OUT = Path("rear_result")
CORE = ("counting", "comparison", "spatial_counting")


def prompts(x, xy, valid, raw):
    """Keep the old table and expose unmatched weak R50 proposals near a foreground box."""
    boxes = x[:, v.BOX] * SCALE
    center = boxes[:, :2] + .5*boxes[:,2:]
    strong_idx = np.flatnonzero(valid & (x[:,0]>logit(.75)))
    fronts = [(int(j),boxes[j]) for j in strong_idx]
    # Align a strong original R50 box to the same union slot, and retain both
    # views. Matching weak candidates against a shifted union box can erase
    # genuine contact at the edge of the occluder.
    if len(strong_idx):
        for row in raw[raw[:,0]>logit(.65)]:
            b = row[-6:-2]*SCALE
            d = np.linalg.norm(center[strong_idx]-b[:2]-.5*b[2:],axis=1)
            k = int(d.argmin())
            if d[k] < max(8.,.35*max(b[2:])):
                fronts.append((int(strong_idx[k]),b))
    pairs = np.full((MAX_REAR,2),-1,np.int16)
    geometry = np.zeros((MAX_REAR,2,4),np.float32)
    if not fronts:
        return x,xy,valid,pairs,geometry
    front_ids,front_boxes = zip(*fronts)
    front_boxes = np.asarray(front_boxes)
    options = []
    for j in np.flatnonzero(valid & (x[:, 0] < logit(.45))):
        options.append((j, boxes[j], float(x[j, 0]), None))
    for row in raw:
        q = float(expit(row[0]))
        if not .025 <= q < .45:
            continue
        b = row[-6:-2] * SCALE
        mid = b[:2] + .5*b[2:]
        distance = np.linalg.norm(center-mid,axis=1)
        near = valid & (distance < max(5., .25*max(b[2:])))
        if (near & (x[:,0]>=logit(.45))).any():
            continue
        j = int(np.argmin(np.where(near,distance,np.inf))) if near.any() else -1
        options.append((j,b,float(row[0]),row))
    chosen = []
    for j, b, confidence, row in options:
        c = b[:2] + .5 * b[2:]
        span = front_boxes[:,2:]
        midpoint = front_boxes[:,:2] + span/2
        distance = np.linalg.norm(midpoint - c, axis=1)
        lower = np.maximum(front_boxes[:,:2],b[:2])
        upper = np.minimum(front_boxes[:,:2]+span,b[:2]+b[2:])
        overlap = np.maximum(0., upper - lower).prod(1) / max(1., b[2:].prod())
        good = distance > .16*span.max(1)
        good &= (c[1] <= midpoint[:, 1] + .12 * span[:, 1])
        good &= (overlap >= .09) & (overlap <= .93)
        if not good.any():
            continue
        score = np.where(good, overlap - .15 * distance / span.max(1).clip(1), -1.)
        front = int(score.argmax())
        chosen.append((confidence + float(score[front]),j,front_ids[front],b,front_boxes[front],row))
    chosen.sort(key=lambda item: -item[0])
    accepted = []
    for _,j,front,b,foreground,row in chosen:
        if len(accepted)==MAX_REAR:
            break
        if any(j==previous or np.linalg.norm(b[:2]+.5*b[2:]-mid)<5.
               for previous,mid in accepted):
            continue
        if j < 0:
            empty = np.flatnonzero(~valid)
            if not len(empty):
                break
            j = int(empty[0])
            x[j, :v.BASE] = row[1:-2]
            xy[j] = row[-2:]
            valid[j] = True
            # Mark proposals absent from the original union. The reference decoder gives them zero mass.
            x[j, 0] = -30.
        k = len(accepted)
        pairs[k] = (j,front)
        geometry[k] = (b,foreground)
        accepted.append((j,b[:2]+.5*b[2:]))
    return x,xy,valid,pairs,geometry


@torch.inference_mode()
def prepare(split, shard, device):
    names = pd.read_csv(v.DATA / f"{split}.csv").image.unique()[shard::2]
    if not len(names):
        raise ValueError("Empty scene shard")
    models = [v.ResNetCenters(full=True), v.CenterDetector()]
    heads = []
    for key, model in zip(("r50_img", "thaw"), models):
        model.load_state_dict(v.load(v.PRIORS / f"{key}.pt"))
        model.to(device, memory_format=torch.channels_last).eval()
        heads.append(v.model_head(key, device))
    matrix = v.projection().to(device)
    def forward(rgb):
        return (v.detector_outputs(models[0], heads[0], rgb),
                v.detector_outputs(models[1], heads[1], rgb, matrix))
    batch = tune_batch("detector", (384, 256, 192, 128, 64, 32) if device.type == "cuda" else (12,),
                       lambda n: forward(torch.zeros(n, 3, v.HEIGHT, v.WIDTH, device=device)), device)
    arrays = []
    cache = np.lib.format.open_memmap(SCRATCH/f"{split}_{shard}_rgb.npy",mode="w+",dtype=np.uint8,
                                       shape=(len(names),v.HEIGHT,v.WIDTH,3))
    started = time.monotonic()
    with ThreadPoolExecutor(2) as pool:
        for i in range(0, len(names), batch):
            pixels = np.stack(list(pool.map(v.read_rgb,[v.DATA/split/n for n in names[i:i+batch]])))
            cache[i:i+len(pixels)] = pixels
            rgb = torch.from_numpy(pixels).to(device).permute(0,3,1,2).float()/255
            a,b = forward(rgb)
            tables = [prompts(*v.merge(aa,bb),aa) for aa,bb in zip(a,b)]
            arrays.append(tuple(map(np.stack,zip(*tables))))
    cache.flush()
    x,xy,valid,pairs,geometry = (np.concatenate(group) for group in zip(*arrays))
    path = SCRATCH / f"{split}_{shard}.npz"
    np.savez(path,names=names.astype(str),x=x,xy=xy,valid=valid,pairs=pairs,geometry=geometry)
    print(split, shard, len(names), int((pairs[..., 0]>=0).sum()), "candidates",
          round(time.monotonic()-started,1), "s", flush=True)


def collect(split, shard=None):
    if shard is not None:
        with np.load(SCRATCH / f"{split}_{shard}.npz") as z:
            return dict(z)
    total = {}
    for i in (0,1):
        part = collect(split, i)
        for key, value in part.items():
            if key not in total:
                n = len(pd.read_csv(v.DATA / f"{split}.csv").image.unique())
                total[key] = np.empty((n, *value.shape[1:]), dtype=value.dtype)
            total[key][i::2] = value
    return total


def view_grid(device):
    a = (torch.arange(SIDE, device=device, dtype=torch.float32)+.5)/SIDE
    y,x = torch.meshgrid(a,a,indexing="ij")
    return torch.stack((x,y),-1)


def relative_grid(grid, front, rear):
    low = torch.minimum(front[:,:2],rear[:,:2])
    high = torch.maximum(front[:,:2]+front[:,2:],rear[:,:2]+rear[:,2:])
    edge = (high-low).amax(1,keepdim=True)*1.16
    return ((low+high-edge)/2)[:,None,None]+grid*edge[:,None,None]


def box_prompt(grid, boxes):
    center = boxes[:,None,None,:2]+.5*boxes[:,None,None,2:]
    dist = (grid-center)/(.5*boxes[:,None,None,2:]).clamp_min(1.)
    return (-1.3*dist.square().sum(-1)).exp()


def view(rgb, grid, front, rear):
    screen = grid/grid.new_tensor([v.WIDTH,v.HEIGHT])
    crop = F.grid_sample(rgb.float()/255, screen*2-1, padding_mode="border", align_corners=False)
    marks = torch.stack((box_prompt(grid,front),box_prompt(grid,rear)),1)
    return torch.cat((crop,marks),1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        def block(cin, cout, stride):
            return nn.Sequential(nn.Conv2d(cin,cout,3,stride,1,bias=False),
                                 nn.GroupNorm(4,cout),nn.SiLU())
        self.edges=nn.Sequential(block(5,24,1),block(24,24,1))
        self.context=nn.Sequential(block(24,40,2),block(40,40,1),block(40,64,2),block(64,64,1))
        self.layers=nn.Conv2d(24,2,1)
        self.head=nn.Sequential(nn.Linear(64*2+2,96),nn.SiLU(),nn.Dropout(.10),nn.Linear(96,17))

    def forward(self,x):
        high = self.edges(x)
        coarse = self.context(high)
        masks = self.layers(high)
        mask = F.adaptive_avg_pool2d(x[:,4:5],coarse.shape[-2:])
        focused = (coarse*mask).sum((-1,-2))/mask.sum((-1,-2)).clamp_min(1e-3)
        features = torch.cat((focused,coarse.amax((-1,-2)),masks.sigmoid().mean((-1,-2))),1)
        return self.head(features).float(), masks.float()


class Paired:
    def __init__(self, device, rgb):
        self.scenes = Scenes(device,rgb=rgb)
        self.grid = view_grid(device)
        d,k = self.scenes.donors.T
        hue = self.scenes.labels[d,k,1]
        idx = [torch.where(hue==c)[0] for c in range(8)]
        width = max(map(len,idx))
        self.bank = torch.stack([F.pad(x,(0,width-len(x))) for x in idx])
        self.lengths = torch.tensor([len(x) for x in idx],device=device)

    def sample(self, idx):
        s = self.scenes
        device = idx.device
        n = len(idx)
        j = (torch.rand(n,s.box.shape[1],device=device)*s.anchors[idx]).argmax(1)
        front = s.box[idx,j]
        col = s.labels[idx,j,1].clamp_min(0)
        choice = (torch.rand(n,device=device)*self.lengths[col]).long()
        match = self.bank[col,choice]
        use = (torch.rand(n,device=device) < .72)&(self.lengths[col]>0)
        donor = torch.where(use,match,torch.randint(len(s.donors),(n,),device=device))
        d,k = s.donors[donor].T
        source = s.box[d,k]
        size = source[:,2:]*(.65+.38*torch.rand(n,1,device=device))
        jitter = torch.rand(n,2,device=device)-.5
        center = front[:,:2]+.5*front[:,2:] + jitter*front[:,2:]*1.4
        center[:,1] = front[:,1] + .5*front[:,3] - (.02+.50*jitter[:,1].abs())*front[:,3]
        rear = torch.cat((center-.5*size,size),1)
        weak = rear.clone()
        weak[:,:2] += (torch.rand(n,2,device=device)-.5)*size*.23
        grid = relative_grid(self.grid,front,weak)
        image = view(s.rgb[idx],grid,front,weak)
        screen = grid/grid.new_tensor([v.WIDTH,v.HEIGHT])
        raster = F.grid_sample(s.instances[idx,None].float(),screen*2-1,mode="nearest",align_corners=False)[:,0]
        u = (grid-rear[:,None,None,:2])/size[:,None,None]
        alpha = F.grid_sample(s.masks[d,k,None].float()/255,u*2-1,align_corners=False)[:,0]
        observed = alpha*(raster==0)
        ratio = observed.sum((-1,-2))/alpha.sum((-1,-2)).clamp_min(1)
        occupied = (alpha*(raster>0)*(raster!=(j[:,None,None]+1))).sum((-1,-2))
        usable = (ratio>.06)&(ratio<.76)&(observed.sum((-1,-2))>=8)
        usable &= (occupied<.15*alpha.sum((-1,-2)))&(d!=idx)&s.anchors[idx].any(1)
        source_grid = (source[:,None,None,:2]+u*source[:,None,None,2:])/grid.new_tensor([v.WIDTH,v.HEIGHT])
        donor_rgb = F.grid_sample(s.rgb[d].float()/255,source_grid*2-1,align_corners=False)
        composite = image.clone()
        composite[:,:3] += (donor_rgb-image[:,:3])*observed[:,None]
        foreground = (raster==j[:,None,None]+1).float()
        return image,composite,foreground,observed,usable,s.labels[d,k]


def bank_parts(table, split, device):
    reference = v.Verifier(False).to(device)
    weights = "union_trainval.pt" if split=="test" else "union_train.pt"
    reference.load_state_dict(v.load(v.PRIORS/weights))
    absent = table["x"][...,0] == -30.
    original = table["valid"] & ~absent
    p,xy,_,*attrs = table_predictions(reference,
        (table["x"],table["xy"],original),device)
    xy[absent] = table["xy"][absent]
    p[absent] = 0
    offset = 1
    for group, values in zip(GROUPS,attrs):
        values[absent] = np.exp(table["x"][...,offset:offset+len(group)][absent])
        offset += len(group)
    return p,xy,table["valid"],attrs,reference.relation_offset.detach().cpu().numpy()


def weak_supervision(frame, table, p, attrs):
    """Mine only constraints with a unique candidate and an unambiguous visible count."""
    lookup = {n:i for i,n in enumerate(table["names"])}
    groupmap = {int(word):(i,j) for i,group in enumerate(GROUPS) for j,word in enumerate(group)}
    accepted = {}
    for row in frame[frame.label.eq(1)&frame.category.eq("counting")].itertuples():
        if row.image not in lookup:
            continue
        prog = parse_program(canonical_form(row.caption,row.category))
        if prog is None or prog.op != EXACT or not prog.a or prog.relation:
            continue
        scene = lookup[row.image]
        applies = np.ones(p.shape[1])
        for word in prog.a:
            group,idx = groupmap[WORD_INDEX[word]]
            applies *= attrs[group][scene,:,idx]
        count = int(((p[scene]>=.85)&(applies>=.80)).sum())
        if ((p[scene]>.38)&(p[scene]<.85)&(applies>.45)).any():
            continue
        choices = [(k,int(j)) for k,(j,_) in enumerate(table["pairs"][scene])
                   if j>=0 and applies[j]>.42]
        if len(choices)!=1 or count not in (prog.number,prog.number-1):
            continue
        candidate,_ = choices[0]
        accepted.setdefault((scene,candidate),set()).add(int(count+1==prog.number))
    return np.asarray([(i,j,next(iter(label))) for (i,j),label in accepted.items() if len(label)==1],
                      dtype=np.int64).reshape(-1,3)


def crop_real(rgb, table, rows, slots, grid):
    objects = torch.from_numpy(table["geometry"][rows,slots]).to(rgb.device)
    rear,front = objects.unbind(1)
    return view(rgb,relative_grid(grid,front,rear),front,rear)


def fit(seed, device, epochs):
    names = pd.read_csv(v.DATA/"train.csv").image.unique()
    table = collect("train")
    if not np.array_equal(names,table["names"]):
        raise ValueError("Train image order differs")
    pixels = np.empty((len(names),v.HEIGHT,v.WIDTH,3),dtype=np.uint8)
    for shard in (0,1):
        pixels[shard::2] = np.load(SCRATCH/f"train_{shard}_rgb.npy",mmap_mode="r")
    with np.load(EXTRA/"fragment_train.npz") as data:
        if not np.array_equal(data["names"],names):
            raise ValueError("Training ROI teacher and image order differ")
    train = Paired(device,pixels)
    del pixels
    p,_,_,attrs,_ = bank_parts(table,"train",device)
    examples = weak_supervision(pd.read_csv(v.DATA/"train.csv"),table,p,attrs)
    positive = examples[examples[:,-1]==1]
    negative = examples[examples[:,-1]==0]
    print("train count-mined positives / negatives",len(positive),len(negative),flush=True)
    rng = np.random.default_rng(seed)
    model = Model().to(device,memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(),lr=.001)
    scaler = torch.amp.GradScaler("cuda",enabled=device.type=="cuda",init_scale=128.)
    def objective(idx, natural=True):
        base,composite,foreground,observed,valid,labels = train.sample(idx)
        mixed = torch.cat((base,composite),0)
        with amp(device):
            pred,masks = model(mixed.contiguous(memory_format=torch.channels_last))
        n = len(idx)
        use = torch.cat((valid,valid))
        y = torch.cat((torch.zeros_like(valid),torch.ones_like(valid))).float()
        losses = F.binary_cross_entropy_with_logits(pred[:,0],y,reduction="none")
        loss = (losses*use).sum()/use.sum().clamp_min(1)
        shape = torch.cat((torch.stack((foreground,torch.zeros_like(observed)),1),
                           torch.stack((foreground,observed),1)))
        mask_loss = F.binary_cross_entropy_with_logits(masks,shape,reduction="none")*(1+3*shape)
        loss += .3*(mask_loss.mean((1,2,3))*use).sum()/use.sum().clamp_min(1)
        for i,chunk in enumerate(pred[n:,1:].split((2,8,6),1)):
            ce = F.cross_entropy(chunk,labels[:,i],reduction="none")
            loss += .18*(ce*valid).sum()/valid.sum().clamp_min(1)
        if natural and len(examples):
            limit = min(max(2,n//16),32)
            controls = negative[rng.choice(len(negative),size=min(limit,len(negative)),replace=False)]
            recoveries = positive[rng.choice(len(positive),size=min(limit,len(positive)),replace=False)]
            rows = np.concatenate((controls,recoveries))
            scene,slot,target = rows.T
            crop = crop_real(train.scenes.rgb[scene],table,scene,slot,train.grid)
            with amp(device):
                output,_ = model(crop.contiguous(memory_format=torch.channels_last))
            target = torch.from_numpy(target).to(device,dtype=torch.float32)
            # Absolute support controls the contribution of rare mined cases.
            # One repeated positive should not outweigh all synthetic positives.
            terms=F.binary_cross_entropy_with_logits(output[:,0],.08+.84*target,reduction="none")
            loss += .20*terms.sum()/(2*limit)
        return loss, valid.sum()

    def benchmark():
        if device.type != "cuda":
            return 8
        results = []
        for n in (512,384,256,192,128,64,32):
            torch.cuda.reset_peak_memory_stats(device)
            idx = torch.arange(n,device=device)%len(names)
            try:
                for attempt in range(3):
                    torch.cuda.synchronize(device)
                    start = time.monotonic()
                    scaler.scale(objective(idx)[0]).backward()
                    torch.cuda.synchronize(device)
                    opt.zero_grad(set_to_none=True)
                    if attempt:
                        results.append((n, time.monotonic()-start,torch.cuda.max_memory_allocated(device)))
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        viable = []
        for n in sorted(set(x[0] for x in results)):
            times = [s for k,s,_ in results if k==n]
            peak = max(m for k,_,m in results if k==n)
            rate = n/np.median(times)
            print("benchmark",n,"pairs",round(rate,1),"pairs/s",round(peak/2**30,2),"GiB",flush=True)
            if len(times)==2 and peak<.88*torch.cuda.get_device_properties(device).total_memory:
                viable.append((rate,n))
        if not viable:
            raise RuntimeError("No training batch fits")
        return max(viable)[1]

    with torch.random.fork_rng(devices=[0] if device.type=="cuda" else []):
        batch = benchmark()
    print("selected batch",batch,"pairs",flush=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    for epoch in range(epochs):
        started = time.monotonic()
        log = []
        pairs_used = []
        for idx in torch.randperm(len(names),device=device).split(batch):
            opt.zero_grad(set_to_none=True)
            loss,used = objective(idx)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(opt)
            scaler.update()
            log.append(loss.detach())
            pairs_used.append(used)
        sched.step()
        print(epoch+1,float(torch.stack(log).mean()),int(torch.stack(pairs_used).sum()),
              "usable pairs",round(time.monotonic()-started,1),"s",flush=True)
    torch.save(model.state_dict(),OUT/f"rear_{seed}.pt")


def decode(programs, pairs, p, attrs, joint, xy, offsets, u):
    present = u[...,0] < p
    codes = np.zeros_like(present,dtype=np.int32)
    for i in range(2):
        picked = (u[...,i+1,None]>attrs[i].cumsum(-1)).sum(-1).clip(max=len(GROUPS[i])-1)
        codes |= 1 << GROUPS[i][picked]
    selected = (u[...,3,None]>joint.cumsum(-1)).sum(-1).clip(max=5)
    codes |= (1<<GROUPS[2][selected//3]) | (1<<GROUPS[3][selected%3])
    position = xy[None]+logit(u[...,-3:-1])*.026
    truth = _execute(programs,codes,present,position,p,offsets,u[...,-1])
    left,right = pairs.T
    linked = truth[:,left]^truth[:,right]
    w = np.exp(1.3*(linked.sum(1)-len(pairs)))[:,None]*linked
    l = logit(((truth[:,left]*w).sum(0)+.75)/(w.sum(0)+1.5))
    result = np.zeros(len(programs))
    result[left],result[right] = l,-l
    return result


@torch.inference_mode()
def predict(split, seed, shard, device):
    frame = pd.read_csv(v.DATA/f"{split}.csv")
    groups = frame.groupby("image",sort=False).indices
    table = collect(split,shard)
    p,pos,valid,attrs,offsets = bank_parts(table,split,device)
    model = Model().to(device,memory_format=torch.channels_last).eval()
    model.load_state_dict(v.load(OUT/f"rear_{seed}.pt"))
    captions = [canonical_form(row.caption,row.category) for row in frame.itertuples()]
    grid = view_grid(device)
    cache = np.load(SCRATCH/f"{split}_{shard}_rgb.npy",mmap_mode="r")
    out_ids,delta,slots_selected = [],[],[]
    candidates = []
    weak = {(i,j):int(y) for i,j,y in weak_supervision(frame,table,p,attrs)} if split=="val" else {}
    for start in range(0,len(table["names"]),192):
        names = table["names"][start:start+192]
        scene,slot = np.nonzero(table["pairs"][start:start+192,:,0]>=0)
        if not len(scene):
            continue
        rgb = torch.from_numpy(cache[scene+start]).to(device).permute(0,3,1,2)
        crops = crop_real(rgb,table,scene+start,slot,grid)
        outputs = []
        for batch in crops.split(768 if device.type=="cuda" else 64):
            with amp(device):
                a,_ = model(batch.contiguous(memory_format=torch.channels_last))
            outputs.append(a.cpu().numpy())
        outputs = np.concatenate(outputs)
        for i in np.unique(scene):
            j = start+i
            name = names[i]
            available = np.flatnonzero(scene==i)
            scores = expit(outputs[available,0])
            best = available[scores.argmax()]
            selected,front = table["pairs"][j,slot[best]]
            probability = float(expit(outputs[best,0]))
            if split=="val":
                for k in available:
                    slot_id = slot[k]
                    rear_id = table["pairs"][j,slot_id,0]
                    raw = outputs[k,1:]
                    joint = np.exp(raw[-6:]-raw[-6:].max())
                    joint /= joint.sum()
                    rear_box,front_box = table["geometry"][j,slot_id]
                    candidates.append(dict(image=str(name),object=float(expit(outputs[k,0])),
                        box=rear_box.round(2).tolist(),foreground=front_box.round(2).tolist(),
                        metal_cylinder=float(joint[-1]),prior_object=float(p[j,rear_id]),
                        weak_target=weak.get((j,slot_id)),
                        used=bool(k==best and probability>=.80 and probability*.95>p[j,rear_id])))
            if probability < .80 or probability*.95 <= p[j,selected]:
                continue
            rows = groups[name]
            twins = find_twins(frame.image.iloc[rows],[captions[k] for k in rows],frame.category.iloc[rows])
            programs = [parse_program(captions[k]) for k in rows]
            include = valid[j]
            index = np.cumsum(include)[selected]-1
            probs = p[j,include]
            local = [a[j,include] for a in attrs]
            independent = (local[-2][:,:,None]*local[-1][:,None,:]).reshape(-1,6)
            u = _draws(len(probs),8,42,2048)
            base = decode(programs,twins,probs,local,independent,pos[j,include],offsets,u)
            probs[index] = .95*probability
            raw = outputs[best,1:]
            for group,probabilities in zip(np.split(raw,[2,10]),[local[0],local[1],independent]):
                calibrated = np.exp(group-group.max())
                probabilities[index] = .5*probabilities[index]+.5*calibrated/calibrated.sum()
            corrected = decode(programs,twins,probs,local,independent,pos[j,include],offsets,u)
            out_ids.extend(frame.id.iloc[rows])
            delta.extend(corrected-base)
            slots_selected.extend([selected]*len(rows))
    np.savez(OUT/f"{split}_{seed}_{shard}.npz",ids=np.asarray(out_ids,dtype=np.int64),
             delta=np.asarray(delta),selected=np.asarray(slots_selected,dtype=np.int16))
    if split=="val":
        (OUT/f"candidates_{seed}_{shard}.json").write_text(json.dumps(candidates),encoding="utf8")
    print(split,seed,shard,"updated captions",len(delta),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","fit","predict"))
    parser.add_argument("--split",default="train",choices=("train","val","test"))
    parser.add_argument("--shard",type=int,default=0,choices=(0,1))
    parser.add_argument("--seed",type=int,default=421)
    parser.add_argument("--epochs",type=int,default=16)
    args = parser.parse_args()
    _,device = v.setup(args.seed)
    SCRATCH.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    if args.action=="prepare":
        prepare(args.split,args.shard,device)
    elif args.action=="fit":
        fit(args.seed,device,args.epochs)
    else:
        predict(args.split,args.seed,args.shard,device)
