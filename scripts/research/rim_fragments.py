"""Recognize tiny rear fragments at weak R50 peaks from native-resolution RGB."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit
import torch
from torch import nn
from torch.nn import functional as F

from scripts.research.fragment_votes import EXTRA, H, W, Scenes, images, v
from scripts.research.visible_fragment_detector import gpu_util, pairwise_iou
from native_centers import canonical_form, parse_program
from src.models.slot_program import EXACT, EXCLUSIVE_GROUPS
from kaggle_suite_common import amp, tune_batch

OUT = Path("rim")
SIZE = 40
SCALE = np.array([W, H, W, H])
ARMS = ("rgb", "edge", "unet", "r50", "r50native", "r50real", "r18", "convnext", "efficient")
SEEDS = (421, 853)


def coords(device, size=SIZE):
    axis = torch.arange(size, device=device) + .5 - size / 2
    iy, ix = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((ix, iy), -1)


def sample(pixels, points, mode="bilinear"):
    grid = points * points.new_tensor([2 / W, 2 / H]) - 1
    return F.grid_sample(pixels.float(), grid, mode=mode, align_corners=False, padding_mode="border")


def prompt(points, box):
    rel = (points - box[:, None, None, :2]) / box[:, None, None, 2:].clamp_min(1)
    # A soft box provides context even where the foreground silhouette bends.
    excess = torch.maximum(-rel, rel - 1).amax(-1)
    return (-5 * excess.clamp_min(0)).exp()


def paired_views(scenes, idx, grid, rng, modes, *, donors=None, difficulty=None, details=False):
    """Insert a real donor behind untouched foreground pixels, with matched controls."""
    b = scenes
    device = idx.device
    n = len(idx)
    random = lambda *shape: torch.rand(shape, device=device, generator=rng)
    ids = b.anchors[idx] & (b.labels[idx] >= 0).all(-1)
    slot = (random(*ids.shape) * ids).argmax(1)
    front = b.box[idx, slot]
    donors = b.donors if donors is None else donors
    source = donors[(random(n) * len(donors)).long()]
    # Same-colour examples deliberately oversample the subtle failure mode.
    col = b.labels[idx, slot, 1].clamp_min(0)
    match = modes[col, (random(n) * (modes[col] >= 0).sum(1)).long()]
    source = torch.where(((random(n) < .50) & (match >= 0))[:, None], donors[match.clamp_min(0)], source)
    di, dj = source.T
    box = b.box[di, dj]
    extent = box[:, 2:] * (.66 + .37 * random(n, 1))
    side = (random(n) < .5).float().mul(2).sub(1)
    c0 = front[:, :2] + front[:, 2:] * torch.stack((.5 + side * (.09 + .34 * random(n)), .10 + .34 * random(n)), 1)
    points = grid[None] + c0[:, None, None]
    base = sample(b.rgb[idx], points) / 255
    instance = sample(b.instances[idx, None], points, "nearest")[:, 0]
    occluder = instance == slot[:, None, None] + 1
    # Try several placements around the same boundary and keep the first valid one.
    initial = c0 + (random(n, 2) - .5) * front[:, 2:] * .3
    best = torch.full((n,), False, device=device)
    masks = torch.zeros_like(occluder, dtype=torch.float)
    grids = torch.zeros_like(points)
    shift = torch.zeros_like(c0)
    min_pixels,max_pixels,min_fraction,max_fraction,covered = {
        None:(4,100,.035,.27,.70), "middle":(21,180,.12,.65,.30), "tiny":(4,20,.02,.27,.60),
    }[difficulty]
    for _ in range(5 if difficulty is None else 10):
        location = initial + (random(n, 2)-.5) * extent * .9
        origin = location - extent / 2
        uv = (points - origin[:, None, None]) / extent[:, None, None]
        alpha = F.grid_sample(b.masks[di, dj, None].float() / 255, 2*uv-1, align_corners=False)[:, 0]
        on = ((points >= 0) & (points < points.new_tensor([W, H]))).all(-1)
        visible = alpha * (instance == 0) * on
        count = (visible >= .50).sum((1, 2))
        area = alpha.sum((1, 2)).clamp_min(1)
        behind = (alpha * occluder).sum((1, 2)) / area
        fraction = visible.sum((1, 2)) / area
        inside_crop = ((location - c0).abs() + extent / 2 <= SIZE / 2 - 1).all(1)
        ok = (~best & (count >= min_pixels) & (count <= max_pixels) & (behind >= covered) &
              (fraction >= min_fraction) & (fraction <= max_fraction) & (di != idx) & ids.any(1) &
              inside_crop & (origin[:, 1] + extent[:, 1] < front[:, 1] + front[:, 3] - 1))
        best |= ok
        masks = torch.where(ok[:, None, None], visible, masks)
        grids = torch.where(ok[:, None, None, None], uv, grids)
        loc = (points * (visible >= .50)[..., None]).sum((1, 2)) / count[:, None].clamp_min(1)
        shift = torch.where(ok[:, None], (loc-c0).clamp(-12,12), shift)
    sample_at = box[:, None, None, :2] + grids * box[:, None, None, 2:]
    patch = sample(b.rgb[di], sample_at) / 255
    # The pair shares the same perturbation and placement prompt.
    positive = base + (patch - base) * masks[:, None]
    # A detector peak can be offset from the few pixels that remain visible.
    uv = grid[None] - shift[:, None, None] - (random(n,2)-.5)[:,None,None]*8
    center = (-.5*(uv / 6).square().sum(-1)).exp()
    blob = (-.5*((uv + (random(n, 2)-.5)[:, None, None]*8)/2.5).square().sum(-1)).exp()
    flash = blob * occluder * (.20 + .5 * random(n))[:, None, None]
    highlight = (random(n) < .55)
    flash = flash[:,None] * highlight[:,None,None,None]
    negative = base + (1-base)*flash
    if difficulty is not None:
        positive = positive + (1-positive)*flash
    perturb = (random(n, 3, SIZE, SIZE)-.5)*(.02*random(n, 1, 1, 1))
    bias = (random(n, 1, 1, 1)-.5)*.05
    p = torch.cat([positive, negative]).add_(torch.cat([perturb+bias]*2)).clamp_(0,1)
    marks = torch.stack([prompt(points, front), center], 1).repeat(2, 1, 1, 1)
    target = torch.cat([masks >= .50, torch.zeros_like(occluder)])
    valid = best.repeat(2)
    labels = b.labels[di, dj]
    result = p, marks, target, occluder.repeat(2, 1, 1), valid, labels, best, torch.cat([ids.new_ones(n),ids.new_zeros(n)])
    if details:
        return result, source, b.labels[di,dj,1].eq(col), highlight
    return result


def block(cin, cout, dilation=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False),
                         nn.GroupNorm(8, cout), nn.SiLU())


class LocalUNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.high=nn.Sequential(block(channels,32),block(32,32))
        self.mid=nn.Sequential(block(32,48),block(48,48))
        self.low=nn.Sequential(block(48,64),block(64,64))
        self.up1=block(64+48,48)
        self.up0=block(48+32,40)

    def forward(self,x):
        a=self.high(x)
        b=self.mid(F.avg_pool2d(a,2))
        c=self.low(F.avg_pool2d(b,2))
        u=self.up1(torch.cat([b,F.interpolate(c,size=b.shape[-2:],mode="bilinear",align_corners=False)],1))
        return self.up0(torch.cat([a,F.interpolate(u,size=a.shape[-2:],mode="bilinear",align_corners=False)],1))


class Rim(nn.Module):
    def __init__(self, arm, pretrained=True):
        super().__init__()
        self.arm = arm
        self.early = None
        if arm in ("r50","r50native","r18","r50real","convnext","efficient"):
            if arm=="r18":
                from torchvision.models import resnet18
                model=resnet18(weights=None)
                if pretrained:
                    model.load_state_dict(v.load(EXTRA/"resnet18-f37072fd.pth"))
                self.early=nn.Sequential(model.conv1,model.bn1,model.relu,model.layer1)
                channels=64
            elif arm=="convnext":
                from torchvision.models import convnext_tiny
                model=convnext_tiny(weights=None)
                if pretrained:
                    model.load_state_dict(v.load(EXTRA/"convnext_tiny-983f1562.pth"))
                self.early=model.features[:2]
                channels=96
            elif arm=="efficient":
                from torchvision.models import efficientnet_v2_s
                model=efficientnet_v2_s(weights=None)
                if pretrained:
                    model.load_state_dict(v.load(EXTRA/"efficientnet_v2_s-dd5fe13b.pth"))
                self.early=model.features[:2]
                channels=24
            else:
                r50=v.ResNetCenters(full=True)
                if pretrained:
                    r50.load_state_dict(v.load(v.PRIORS/"r50_img.pt"))
                self.early=r50.stem[:3]
                if arm in ("r50native","r50real"):
                    self.early[0].stride=(1,1)
                channels=64
            self.early.requires_grad_(False)
            self.register_buffer("mean",torch.tensor([.485,.456,.406])[None,:,None,None])
            self.register_buffer("std",torch.tensor([.229,.224,.225])[None,:,None,None])
            self.project=nn.Conv2d(channels,16,1)
        # Every arm sees the same native-resolution pixels and target coordinates.
        n = 5 + (0 if arm == "rgb" else 7) + (16 if self.early is not None else 0)
        self.pixels = (LocalUNet(n) if arm=="unet" else
                       nn.Sequential(block(n,32),block(32,32,2),block(32,40,3),block(40,40,2)))
        self.mask = nn.Conv2d(40, 2, 1)
        self.read = nn.Sequential(nn.Linear(40*3+2, 96),nn.SiLU(),nn.Dropout(.08),nn.Linear(96,17))

    def train(self, mode=True):
        super().train(mode)
        if self.early is not None:
            self.early.eval()
        return self

    def forward(self, rgb, marks):
        fields = [rgb, marks]
        if self.arm != "rgb":
            dx = F.pad(rgb[..., 2:]-rgb[..., :-2], (1, 1, 0, 0), mode="replicate")
            dy = F.pad(rgb[..., 2:, :]-rgb[..., :-2, :], (0, 0, 1, 1), mode="replicate")
            local = F.avg_pool2d(rgb.mean(1, keepdim=True), 5, 1, 2)
            fields += [dx, dy, rgb.mean(1,keepdim=True)-local]
        if self.early is not None:
            with torch.set_grad_enabled(torch.is_grad_enabled() and any(p.requires_grad for p in self.early.parameters())):
                early = self.early((rgb - self.mean) / self.std)
            fields += [F.interpolate(self.project(early), size=rgb.shape[-2:], mode="bilinear", align_corners=False)]
        x = self.pixels(torch.cat(fields,1).contiguous(memory_format=torch.channels_last))
        mask = self.mask(x).float()
        candidate = marks[:, 1:2]
        w = mask[:, :1].sigmoid() * (1-mask[:, 1:2].sigmoid()) * candidate
        total = w.sum((2, 3)).clamp_min(1e-5)
        masked = (x*w).sum((2, 3))/total
        window = (x*candidate).sum((2, 3)) / candidate.sum((2, 3)).clamp_min(1e-5)
        record = torch.cat((masked, window, x.amax((2,3)), total.log1p(), w.amax((2,3))),1)
        return self.read(record).float(), mask


def objective(model,batch,outputs=None):
    rgb,marks,target,front,valid,labels,ok,truth = batch
    raw,mask = model(rgb,marks) if outputs is None else outputs
    n = len(ok)
    pos = truth.float()
    valid = valid.float()
    def bce(logit_value, truth, mask):
        loss = F.binary_cross_entropy_with_logits(logit_value,truth,reduction="none")
        return (loss*mask).sum()/mask.sum().clamp_min(1)
    loss = bce(raw[:,0],pos,valid)
    if n:
        loss += .6*bce(raw[:n,0]-raw[n:,0],torch.ones_like(pos[:n]),ok)
    certified=valid*pos
    rear = mask[:,0]
    score = F.binary_cross_entropy_with_logits(rear,target.float(),reduction="none")*(rear.sigmoid()-target.float()).square()
    fg = (score*target).sum((1,2))/target.sum((1,2)).clamp_min(1)
    bg = (score*~target).mean((1,2))
    loss += ((fg+bg)*valid).sum()/valid.sum().clamp_min(1)
    loss += .22 * bce(mask[:,1],front.float(),valid[:,None,None].expand_as(front))
    attrs=torch.cat([labels,labels]) if n else labels
    for z,y in zip(raw[:,1:].split((2,8,6),1),attrs.T):
        ce = F.cross_entropy(z,y.clamp_min(0),reduction="none")
        use=certified*(y>=0)
        loss += .20 * (ce*use).sum() / use.sum().clamp_min(1)
    return loss


def load_bank(device):
    with np.load(EXTRA/"fragment_train.npz") as z:
        names = z["names"]
    if not np.array_equal(names,pd.read_csv(v.DATA/"train.csv").image.unique()):
        raise ValueError("Train donor bank does not match the competition split")
    started = time.perf_counter()
    rgb = images("train",names)
    decoded = time.perf_counter()
    bank = Scenes(device,rgb)
    del rgb
    c = bank.labels[bank.donors[:,0],bank.donors[:,1],1]
    pools = [torch.where(c==k)[0] for k in range(8)]
    modes = torch.stack([F.pad(x,(0,max(map(len,pools))-len(x)),value=-1) for x in pools])
    sync(device)
    time_summary = dict(jpeg_s=decoded-started,rasterize_and_upload_s=time.perf_counter()-decoded)
    return bank,modes,time_summary


def sync(device):
    if device.type=="cuda":
        torch.cuda.synchronize(device)


def fit(arm,seed,device,epochs,limit=None):
    bank,modes,startup = load_bank(device)
    ids = torch.where((bank.anchors & (bank.labels>=0).all(-1)).any(1))[0][:limit]
    grid = coords(device)
    rng = torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed)
    net = Rim(arm).to(device,memory_format=torch.channels_last).train()
    opt = torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),lr=.0012,weight_decay=.02,
                            fused=device.type=="cuda")
    scale = torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    report = []
    t_real=time.perf_counter()
    real=RealPeaks(bank,device) if arm=="r50real" else None
    startup["real_mining_s"]=time.perf_counter()-t_real

    def probe(batch):
        idx = ids[torch.arange(batch,device=device)%len(ids)]
        if device.type=="cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            for i in range(4):
                sync(device)
                t = time.perf_counter()
                data = paired_views(bank,idx,grid,rng,modes)
                extra=real.draw(max(4,len(idx)//4),rng) if real is not None else None
                host = time.perf_counter()-t
                sync(device)
                augmented = time.perf_counter()-t
                with amp(device):
                    loss = objective(net,data)
                    if extra is not None:
                        loss+=.55*objective(net,extra)
                scale.scale(loss).backward()
                # Include optimizer memory and latency in the measured step.
                scale.step(opt)
                scale.update()
                sync(device)
                elapsed = time.perf_counter()-t
                opt.zero_grad(set_to_none=True)
                if i:
                    used = torch.cuda.max_memory_reserved(device) if device.type=="cuda" else 0
                    total = torch.cuda.get_device_properties(device).total_memory if device.type=="cuda" else 1
                    report.append(dict(batch=batch,crops_s=2*batch/elapsed,prepare_ms=1e3*augmented,
                                       host_ms=1e3*host,reserved_gib=used/2**30,vram_fraction=used/total,gpu=gpu_util(device)))
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            print(arm,batch,"OOM",flush=True)

    if device.type=="cuda":
        for batch in (32,64,128,192,256,384,512):
            probe(batch)
        usable = pd.DataFrame(report)
        if usable.empty:
            raise RuntimeError("No batch could be profiled")
        viable=usable[usable.vram_fraction<=.91].groupby("batch").crops_s.median()
        if viable.empty:
            raise RuntimeError("GPU capacity exceeded on all batch sizes")
        batch = int(viable[viable>=.97*viable.max()].index.max())
        torch.cuda.empty_cache()
    else:
        batch = min(8,len(ids))
    if device.type=="cuda":
        torch.manual_seed(seed)
        net=Rim(arm).to(device,memory_format=torch.channels_last).train()
        opt=torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),lr=.0012,weight_decay=.02,fused=True)
        scale=torch.amp.GradScaler("cuda")
    rng.manual_seed(seed+1)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=.0001)
    history = []
    for ep in range(epochs):
        start = time.perf_counter()
        progress = torch.zeros(2,device=device)
        steps = 0
        real_counts=torch.zeros(2,device=device)
        # Draw a fresh placement per source scene on every pass.
        for idx in ids[torch.randperm(len(ids),device=device,generator=rng)].split(batch):
            data = paired_views(bank,idx,grid,rng,modes)
            opt.zero_grad(set_to_none=True)
            with amp(device):
                loss = objective(net,data)
                if real is not None:
                    real_batch=real.draw(max(4,len(idx)//4),rng)
                    loss+=.55*objective(net,real_batch)
                    real_counts += torch.stack((real_batch[-1].sum(),real_batch[-1].numel()-real_batch[-1].sum()))
            scale.scale(loss).backward()
            scale.step(opt)
            scale.update()
            progress += torch.stack((loss.detach(),data[-2].sum()))
            steps += 1
        loss,n = progress.tolist()
        schedule.step()
        if not np.isfinite(loss):
            raise ValueError("Non-finite training objective")
        if n<1:
            raise ValueError("No valid occlusion examples this epoch")
        row = dict(epoch=ep+1,loss=loss/steps,pairs=int(n),elapsed_s=time.perf_counter()-start,gpu=gpu_util(device),real_positive=int(real_counts[0]),real_negative=int(real_counts[1]))
        history.append(row)
        print(arm,seed,row,flush=True)
    torch.save(net.state_dict(),OUT/f"{arm}_{seed}.pt")
    (OUT/f"{arm}_{seed}.json").write_text(json.dumps(dict(startup=startup,batch=batch,probes=report,epochs=history),indent=2))
def candidate_peaks(rows):
    """Find weak detector boxes near visible, stronger foreground objects."""
    p=expit(rows[:,0])
    b=rows[:,-6:-2]*SCALE
    center=b[:,:2]+.5*b[:,2:]
    fronts=np.flatnonzero((p>=.50)&(b[:,2:].min(1)>=7))
    weak=np.flatnonzero((p>=.025)&(p<=.42)&(b[:,2:].min(1)>=5))
    if not len(fronts):
        return []
    out=[]
    for j in weak:
        d=np.linalg.norm(center[fronts]-center[j],axis=1)
        inter=(np.minimum(b[fronts,:2]+b[fronts,2:],b[j,:2]+b[j,2:])-
               np.maximum(b[fronts,:2],b[j,:2])).clip(0).prod(1)
        cover=inter/np.prod(b[j,2:]).clip(1)
        dy=(b[fronts,1]+b[fronts,3])-(b[j,1]+b[j,3])
        gap=np.maximum(0,np.maximum(b[j,:2]-b[fronts,:2]-b[fronts,2:],
                                     b[fronts,:2]-b[j,:2]-b[j,2:])).max(1)
        near=(d>=4)&(d<=b[fronts,2:].max(1)*.90+8)&(dy>=-3)&((cover>.025)|(gap<4))
        if near.any():
            k=fronts[np.argmax(np.where(near,cover-.011*d,-100.))]
            out.append((b[j],b[k],float(p[j])))
    return out


@torch.inference_mode()
def prepare(split,shard,device,limit):
    t=time.perf_counter()
    names=pd.read_csv(v.DATA/f"{split}.csv").image.unique()[:limit][shard::2].astype(str)
    model=v.ResNetCenters(full=True).to(device,memory_format=torch.channels_last).eval()
    model.load_state_dict(v.load(v.PRIORS/"r50_img.pt"))
    head=v.model_head("r50_img",device)
    def forward(rgb):
        return v.detector_outputs(model,head,rgb)
    sizes=(256,192,128,64,32,16) if device.type=="cuda" else (16,)
    batch=tune_batch("weak peaks",sizes,lambda n:forward(torch.zeros(n,3,H,W,device=device)),device)
    index=[]; boxes=[]; front=[]; source=[]
    rgb_path=OUT/f"{split}_{shard}_rgb.npy"
    bank=np.lib.format.open_memmap(rgb_path,mode="w+",dtype=np.uint8,shape=(len(names),H,W,3))
    for start in range(0,len(names),batch):
        pix=images(split,names[start:start+batch])
        bank[start:start+len(pix)]=pix
        rgb=torch.from_numpy(pix).to(device).permute(0,3,1,2).float()/255
        for i,rows in enumerate(forward(rgb),start):
            found=candidate_peaks(rows)
            for a,b,score in found:
                index.append(i)
                boxes.append(a)
                front.append(b)
                source.append(score)
        if start%512==0:
            print(split,shard,start+len(pix),len(index),flush=True)
    bank.flush()
    np.savez_compressed(OUT/f"{split}_{shard}.npz",names=names,index=np.asarray(index,dtype=np.int64),
                        boxes=np.asarray(boxes,dtype=np.float32).reshape(-1,4),
                        front=np.asarray(front,dtype=np.float32).reshape(-1,4),source=np.asarray(source))
    print("prepared",split,shard,len(names),len(index),time.perf_counter()-t,flush=True)


def scene_targets(split, names):
    frame=pd.read_csv(v.DATA/f"{split}.csv")
    allowed=frame.label.eq(1)&frame.category.eq("counting")
    lookup={}
    for row in frame[allowed].itertuples(index=False):
        p=parse_program(canonical_form(row.caption,row.category))
        if p is not None and p.op==EXACT and not p.relation and not p.a:
            lookup[row.image]=p.number
    return np.asarray([lookup.get(str(n),-1) for n in names])


def geometry_inputs(rgb, boxes, fronts, grid):
    center=boxes[:,:2]+boxes[:,2:]/2
    points=grid[None]+center[:,None,None]
    px=sample(rgb,points)/255
    gaussian=(-.5*(grid/6).square().sum(-1)).exp().expand(len(center),-1,-1)
    marks=torch.stack([prompt(points,fronts),gaussian],1)
    return px,marks,points


class RealPeaks:
    """Mine positives from independently supported rear masks and conservative negatives."""
    def __init__(self,bank,device):
        self.bank=bank
        self.device=device
        self.grid=coords(device)
        with np.load(EXTRA/"fragment_train.npz") as z:
            names=z["names"]
            box=z["box"]
            labels=z["labels"]
            keep=z["keep"]
        counts=scene_targets("train",names)
        lookup={name:i for i,name in enumerate(names)}
        self.data=[]
        self.positive=[]
        self.negative=[]
        self.boxes=[]
        self.withheld=[]
        for shard in (0,1):
            with np.load(OUT/f"train_{shard}.npz") as z:
                self.mine(z,lookup,box,labels,keep,counts)
        natural=len(self.positive)
        if natural<32:
            self.supplement(box,labels,keep)
        if not self.positive or not self.negative:
            raise RuntimeError(f"Insufficient real candidates: {len(self.positive)}/{len(self.negative)}")
        self.idx,self.front,self.rear,self.truth=map(np.asarray,zip(*self.data))
        self.idx=torch.as_tensor(self.idx,device=device,dtype=torch.long)
        self.front=torch.as_tensor(self.front,device=device,dtype=torch.long)
        self.rear=torch.as_tensor(self.rear,device=device,dtype=torch.long)
        self.truth=torch.as_tensor(self.truth,device=device,dtype=torch.bool)
        self.boxes=torch.as_tensor(np.stack(self.boxes),device=device,dtype=torch.float32)
        self.positive=torch.as_tensor(self.positive,device=device,dtype=torch.long)
        self.negative=torch.as_tensor(self.negative,device=device,dtype=torch.long)
        print("train real crops",natural,"weak positives",len(self.withheld),"withheld positives",len(self.negative),"count-consistent negatives",flush=True)

    def mine(self,z,lookup,box,labels,keep,counts):
        names=z["names"]
        for scene,candidate,front in zip(z["index"],z["boxes"],z["front"]):
            i=lookup[names[scene]]
            confirmed=np.flatnonzero(keep[i]&(labels[i]>=0).all(1))
            if not len(confirmed):
                continue
            front_overlap=pairwise_iou(front,box[i,confirmed])
            k=confirmed[front_overlap.argmax()]
            if front_overlap.max()<.45:
                continue
            centers=box[i,:,:2]+box[i,:,2:]/2
            mid=candidate[:2]+candidate[2:]/2
            dist=np.linalg.norm(centers-mid,axis=1)
            cover=pairwise_iou(candidate,box[i])
            # A rear mask and the front mask must come from different teacher instances.
            candidates=confirmed[(confirmed!=k)&(cover[confirmed]>=.36)&(dist[confirmed]<7)&
                                 (box[i,confirmed,1]+box[i,confirmed,3] < box[i,k,1]+box[i,k,3]-2)]
            if len(candidates)==1:
                rear=int(candidates[0])
                if pairwise_iou(box[i,rear],box[i,k])<.035:
                    continue
                positive=True
            elif len(candidates)==0 and counts[i]==len(confirmed) and len(confirmed)==keep[i].sum():
                # A trusted total count excludes an additional object for this training scene.
                if ((dist[confirmed[confirmed!=k]]<10)&(cover[confirmed[confirmed!=k]]>.08)).any():
                    continue
                rear=k
                positive=False
            else:
                continue
            (self.positive if positive else self.negative).append(len(self.data))
            self.data.append((i,k,rear,positive))
            self.boxes.append(candidate)

    def supplement(self,box,labels,keep):
        # A known rear object from a real image can be withheld from the object
        # table. Its pixels are preserved and it is queried at the original location.
        rng=np.random.default_rng(731)
        for i in rng.permutation(len(box)):
            good=np.flatnonzero(keep[i]&(labels[i]>=0).all(1))
            for rear in good:
                b=box[i,rear]
                choices=[j for j in good if j!=rear and box[i,j,1]+box[i,j,3]>b[1]+b[3]+2]
                if not choices:
                    continue
                overlap=pairwise_iou(b.astype(float),box[i,choices])
                j=int(overlap.argmax())
                if overlap[j]<.08 or overlap[j]>.58:
                    continue
                f=choices[j]
                self.withheld.append(len(self.data))
                self.positive.append(len(self.data))
                self.data.append((i,f,rear,True))
                self.boxes.append(b)
                break
            if len(self.withheld)>=512:
                break

    def draw(self,n,rng):
        pos,neg=self.positive,self.negative
        take=torch.cat([pos[torch.randint(len(pos),(n//2,),device=self.device,generator=rng)],
                        neg[torch.randint(len(neg),(n-n//2,),device=self.device,generator=rng)]])
        src=self.idx[take]
        cf,cr=self.front[take],self.rear[take]
        rgb,marks,points=geometry_inputs(self.bank.rgb[src],self.boxes[take],self.bank.box[src,cf],self.grid)
        inst=sample(self.bank.instances[src,None],points,"nearest")[:,0]
        truth=self.truth[take]
        masks=(inst==cr[:,None,None]+1)&truth[:,None,None]
        return (rgb,marks,masks,inst==cf[:,None,None]+1,torch.ones_like(truth),self.bank.labels[src,cr],
                truth[:0],truth)


@torch.inference_mode()
def infer(arm,seed,split,shard,device,model=None,scene_names=None):
    data_path=OUT/f"{split}_{shard}.npz"
    with np.load(data_path) as z:
        info=dict(z)
    if scene_names is not None:
        keep=np.isin(info["names"][info["index"]],list(scene_names))
        for key in ("index","boxes","front","source"):
            info[key]=info[key][keep]
    pixels=np.load(OUT/f"{split}_{shard}_rgb.npy",mmap_mode="r")
    grid=coords(device)
    if model is None:
        model=Rim(arm,pretrained=False).to(device,memory_format=torch.channels_last).eval()
        model.load_state_dict(v.load(OUT/f"{arm}_{seed}.pt"))
    boxes=info["boxes"]; fs=info["front"]; idx=info["index"]
    values=[]; coordinates=[]; area=[]; relative=[]
    for start in range(0,len(idx),256):
        ids=slice(start,start+256)
        rgb=torch.from_numpy(pixels[idx[ids]]).to(device).permute(0,3,1,2)
        bb=torch.from_numpy(boxes[ids]).to(device)
        ff=torch.from_numpy(fs[ids]).to(device)
        patch,marks,points=geometry_inputs(rgb,bb,ff,grid)
        with amp(device):
            raw,logits=model(patch,marks)
        rear=logits[:,0].sigmoid()
        support=(rear>.50)&(logits[:,1].sigmoid()<.55)
        roi=marks[:,1]>=.08
        valid=(points>=0).all(-1)&(points<points.new_tensor([W,H])).all(-1)
        support &= valid
        score=rear.masked_fill(~(support&roi),-1)
        j=score.flatten(1).argmax(1)
        peak=points.flatten(1,2)[torch.arange(len(raw),device=device),j]
        total=support.sum((1,2))
        local=(support&roi).sum((1,2))
        values.append(raw.cpu().numpy())
        coordinates.append(peak.cpu().numpy())
        area.append(local.cpu().numpy())
        relative.append((local/total.clamp_min(1)).cpu().numpy())
    output=dict(data=np.concatenate(values) if values else np.zeros((0,17)),
                peak=np.concatenate(coordinates) if coordinates else np.zeros((0,2)),
                pixels=np.concatenate(area) if area else np.zeros(0),
                compactness=np.concatenate(relative) if relative else np.zeros(0))
    np.savez_compressed(OUT/f"{arm}_{seed}_{split}_{shard}.npz",**info,**output)
    print(arm,seed,split,shard,len(idx),"scored",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","fit","infer"))
    parser.add_argument("--split",choices=("train","val","test"),default="val")
    parser.add_argument("--shard",type=int,choices=(0,1),default=0)
    parser.add_argument("--seed",type=int,default=421)
    parser.add_argument("--arm",choices=ARMS,default="edge")
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    _,device=v.setup(args.seed)
    if args.stage=="prepare":
        prepare(args.split,args.shard,device,args.limit)
    elif args.stage=="fit":
        fit(args.arm,args.seed,device,args.epochs,args.limit)
    else:
        infer(args.arm,args.seed,args.split,args.shard,device)
