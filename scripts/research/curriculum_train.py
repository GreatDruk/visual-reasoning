"""Checkpointed local fragment training and full-frame U-Net ablations."""
import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.research import curriculum_data as data, rim_fragments as r, rim_scene_unet as scene
from scripts.research import detail_counts as segments
from kaggle_suite_common import amp

OUT = Path("curriculum")
MODES = (
    ("unet","legacy"),("r50","legacy"),
    ("unet","mixed"),("unet","staged"),("unet","noexact"),("unet","boundary"),
    ("r50","mixed"),("r50","staged"),("r50","noexact"),
    ("efficient","mixed"),("efficient","staged"),("r18","mixed"),("r18","staged"),
    ("convnext","mixed"),("convnext","staged"),("r101","staged"),
    ("scene","staged"),("scene","noexact"),
)
SEEDS = (421,853)
CHECKPOINTS = (400,800,1200)
WEIGHTS = {
    "r50":"resnet50-11ad3fa6.pth", "r101":"resnet101-cd907fc2.pth",
    "r18":"resnet18-f37072fd.pth", "efficient":"efficientnet_v2_s-dd5fe13b.pth",
    "convnext":"convnext_tiny-983f1562.pth",
}


def tag(arm,mode):
    return f"{arm}_{mode}"


def schedule(mode,limit):
    if mode=="legacy":
        return [(data.WEAK,j) for j in range(limit)]
    sizes=np.asarray([240,240,400,320])
    stops=np.rint(np.cumsum(sizes)/1200*limit).astype(int)
    begin=np.r_[0,stops[:-1]]
    keys=[(kind,j) for kind,a,b in zip(data.TAIL,begin,stops) for j in range(b-a)]
    if mode=="mixed":
        np.random.default_rng(4081).shuffle(keys)
    return keys


class Local(r.Rim):
    def __init__(self,arm,mode,pretrained):
        legacy=mode=="legacy"
        stem="r50native" if arm=="r50" and legacy else "unet" if arm=="unet" else "edge"
        super().__init__(stem,pretrained)
        if not legacy and arm!="unet":
            encoder,n=self.encoder(arm,pretrained)
            self.early=encoder
            self.project=nn.Conv2d(n,16,1)
            self.register_buffer("mean",torch.tensor([.485,.456,.406])[None,:,None,None])
            self.register_buffer("std",torch.tensor([.229,.224,.225])[None,:,None,None])
            self.pixels[0]=r.block(28,32)
        # Attach the same head on both sides of the ablation. Only its loss changes.
        if arm=="unet" and not legacy:
            self.mask=nn.Conv2d(40,4,1)

    @staticmethod
    def encoder(arm,pretrained):
        from torchvision import models
        architectures={"r50":models.resnet50,"r101":models.resnet101,
                       "r18":models.resnet18,"efficient":models.efficientnet_v2_s,
                       "convnext":models.convnext_tiny}
        kwargs={"replace_stride_with_dilation":[False,True,False]} if arm in ("r50","r101") else {}
        net=architectures[arm](weights=None,**kwargs)
        if pretrained:
            net.load_state_dict(r.v.load(data.EXTRA/WEIGHTS[arm]))
        if arm in ("r50","r101","r18"):
            stages=(net.conv1,net.bn1,net.relu,net.layer1,net.layer2)
            if arm in ("r50","r101"):
                stages=(*stages,net.layer3)
            return nn.Sequential(*stages),128 if arm=="r18" else 1024
        return (net.features[:3],48) if arm=="efficient" else (net.features[:2],96)

def seed_rng(device,seed,kind,occurrence):
    return torch.Generator(device=device).manual_seed(seed*1000003+kind*100003+occurrence)


def boundary_loss(logits,batch):
    mask,front,valid=batch[2:5]
    front=front.float()
    gx=F.pad(front[:,:,2:]-front[:,:,:-2],(1,1,0,0))
    gy=F.pad(front[:,2:,:]-front[:,:-2,:],(0,0,1,1))
    vec=torch.stack([gx,gy],1)
    overlap=F.max_pool2d(mask[:,None].float(),5,1,2)[:,0]>0
    edge=(vec.square().sum(1)>0)&overlap&valid[:,None,None]
    unit=F.normalize(vec,dim=1)
    error=(logits.tanh()-unit).square().sum(1)
    return (error*edge).sum()/edge.sum().clamp_min(1)


def local_batch(bank,kind,n,rng):
    if kind==data.CLEAR:
        return (bank.clear(n,rng,"fit"),)
    if kind==data.WEAK:
        sources,donors,modes,*_=bank.pools["fit"]
        idx=data.draw_index(sources,n,rng)
        return (r.paired_views(bank.bank,idx,bank.grid,rng,modes,donors=donors),)
    if kind==data.REAL:
        n=max(1,n//4)
        windows=bank.real.windows(n,rng,"fit")
        paired=bank.paired(n,rng,"fit","middle")
        return paired[0],data.intact(bank.bank,paired[1],bank.grid),windows,bank.real.bags(n,rng,"fit")
    paired=bank.paired(n,rng,"fit","tiny" if kind==data.TINY else "middle")
    return paired[0],data.intact(bank.bank,paired[1],bank.grid)


def local_loss(model,batches,mode):
    crop,*extras=batches
    raw,mask=model(*crop[:2])
    loss=r.objective(model,crop,(raw,mask[:,:2]))
    if mode=="boundary" and extras:
        loss=loss+.20*boundary_loss(mask[:,2:4],crop)
    if not extras:
        return loss
    intact,*real=extras
    raw_clear,mask_clear=model(*intact[:2])
    usable=crop[-2]
    loss=loss+.35*r.objective(model,intact,(raw_clear,mask_clear[:,:2]))
    p=raw[:len(usable),1:].split((2,8,6),1)
    q=raw_clear[:,1:].detach().split((2,8,6),1)
    kl=sum(F.kl_div(a.log_softmax(1),b.softmax(1),reduction="none").sum(1) for a,b in zip(p,q))
    loss=loss+.18*(kl*usable).sum()/usable.sum().clamp_min(1)
    if real:
        supervised,bag=real
        if supervised is not None:
            sr,sm=model(*supervised[:2])
            loss=loss+.40*r.objective(model,supervised,(sr,sm[:,:2]))
        if mode!="noexact":
            x,prompt,pad,filters,n,_=bag
            found,_=model(x,prompt)
            logits=found.new_zeros((*pad.shape,found.shape[1]))
            logits[pad]=found
            prob=data.membership(logits,filters)*pad
            loss=loss+.9*data.exact_count_loss(prob,n)
    return loss


def scene_batch(bank,kind,n,rng):
    # All stages include unaltered full frames, including scenes used for exact counts.
    sources=bank.pools["fit"][4]
    first=max(1,n//3)
    color_count=min(n-first,max(1,n//6)) if bank.color_scenes is not None else 0
    idx=[data.draw_index(bank.counted,first,rng)]
    if color_count:
        idx.append(data.draw_index(bank.color_scenes,color_count,rng))
    idx.append(data.draw_index(sources,n-first-color_count,rng))
    idx=torch.cat(idx)
    if kind==data.CLEAR:
        rgb=bank.bank.rgb[idx].float()/255
        mask=bank.bank.instances[idx,1::2,1::2].long()
        labels=bank.bank.labels[idx]
    else:
        donors=bank.pools["fit"][1]
        difficulty={data.MIDDLE:"middle",data.TINY:"tiny",data.REAL:"mixed"}[kind]
        rgb,mask,_,labels,_=bank.bank.pair(idx,tiny_fragments=True,donors=donors,difficulty=difficulty)
        labels=labels[:,1:]
        if kind==data.REAL:
            # Keep most of this stage as original scenes.
            take=torch.cat((torch.arange(n,device=idx.device),
                            n+torch.arange(max(1,n//4),device=idx.device)))
            rgb,mask,labels=rgb[take],mask[take],labels[take]
    selected=idx[:first]
    filters=[values[selected] for values in bank.scene_filters]
    number=bank.scene_target[selected].clamp_max(segments.K)
    color_query=None
    if color_count:
        rows=slice(first,first+color_count)
        color_query=rows,bank.color_filters[idx[rows]],bank.color_operators[idx[rows]]
    return rgb,mask,labels,filters,number,color_query


def scene_loss(model,batch,mode,timing):
    rgb,mask,labels,filters,number,color_query=batch
    with amp(rgb.device):
        raw,masks=model(rgb.contiguous(memory_format=torch.channels_last))
    loss=segments.mask_loss(raw,masks,mask,labels,timing)
    if mode!="noexact":
        n=len(number)
        p=data.membership(raw[:n],filters)
        loss=loss+.55*data.exact_count_loss(p,number)
    if color_query is not None:
        rows,allowed,operator=color_query
        colors=raw[rows,:,3:11].float().softmax(-1)
        members=raw[rows,:,:1].float().sigmoid()*colors
        probability=scene.compare_difference(members,allowed[:,0],allowed[:,1],operator)
        loss=loss-.35*probability.clamp_min(1e-6).log().mean()
    return loss


@dataclass
class Benchmark:
    batch:int
    examples_s:float
    prepare_ms:float
    host_ms:float
    matching_cpu_ms:float
    vram_gib:float
    vram_fraction:float
    gpu:int|None


def build(arm,mode,seed,device,pretrained=True):
    torch.manual_seed(seed)
    model=scene.SceneUNet() if arm=="scene" else Local(arm,mode,pretrained)
    return model.to(device,memory_format=torch.channels_last)


def optimizer(model,device,arm,mode):
    lr=.0007 if arm=="scene" else .0012
    early=getattr(model,"early",None)
    slow=[] if early is None else [p for p in early.parameters() if p.requires_grad]
    ids={id(p) for p in slow}
    main=[p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    groups=[{"params":main,"initial_lr":lr}]
    if slow:
        groups.append({"params":slow,"lr":lr*.15,"initial_lr":lr*.15})
    return torch.optim.AdamW(groups,lr=lr,weight_decay=.02,fused=device.type=="cuda")


def step(model,opt,scaler,bank,arm,mode,kind,n,rng,profile=False):
    tick=time.perf_counter()
    batch=scene_batch(bank,kind,n,rng) if arm=="scene" else local_batch(bank,kind,n,rng)
    host=time.perf_counter()-tick
    if profile:
        r.sync(bank.device)
    prep=time.perf_counter()-tick
    timing={} if profile else None
    opt.zero_grad(set_to_none=True)
    if arm=="scene":
        loss=scene_loss(model,batch,mode,timing)
        samples=len(batch[0])
        details=None
    else:
        with amp(bank.device):
            loss=local_loss(model,batch,mode)
        samples=sum(len(b[0]) for b in batch if b is not None)
        details=None if profile else (batch[0][4].sum().detach(),batch[0][-2].sum().detach(),
                 batch[-1][-1] if kind==data.REAL and mode!="noexact" else None)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    if profile:
        r.sync(bank.device)
        elapsed=time.perf_counter()-tick
        gpu=bank.device.type=="cuda"
        mem=torch.cuda.max_memory_reserved(bank.device) if gpu else 0
        capacity=torch.cuda.get_device_properties(bank.device).total_memory if gpu else 1
        return Benchmark(n,samples/elapsed,1000*prep,1000*host,(timing or {}).get("matching_cpu_ms",0),
                         mem/2**30,mem/capacity,r.gpu_util(bank.device))
    return loss.detach(),details


def probe(bank,arm,mode,seed):
    dev=bank.device
    model=build(arm,mode,seed,dev)
    opt=optimizer(model,dev,arm,mode)
    scaler=torch.amp.GradScaler("cuda",enabled=dev.type=="cuda",init_scale=128)
    options=(8,16,32,48,64,96,128) if arm=="scene" else (16,32,64,96,128,192,256,384,512)
    if dev.type!="cuda":
        options=(2,4)
    # Profile high-activation synthetic stages and the actual real-count forward/backward.
    kinds=[data.WEAK] if mode=="legacy" else [data.TINY,data.REAL]
    records=[]
    for n in options:
        if dev.type=="cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
        row=[]
        try:
            for kind in kinds:
                rng=seed_rng(dev,seed,kind,9)
                step(model,opt,scaler,bank,arm,mode,kind,n,rng,True)
                row.append(vars(step(model,opt,scaler,bank,arm,mode,kind,n,rng,True)))
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            print(arm,mode,n,"OOM",flush=True)
            break
        records.append(dict(batch=n,examples_s=min(j["examples_s"] for j in row),
                            vram_fraction=max(j["vram_fraction"] for j in row),measurements=row))
    good=[q for q in records if q["vram_fraction"]<=.91]
    speed=max(q["examples_s"] for q in good)
    batch=max(q["batch"] for q in good if q["examples_s"]>=.95*speed)
    print("probe",arm,mode,batch,flush=True)
    return dict(batch=batch,probes=records,startup=bank.startup,coverage=bank.real.coverage)


@torch.inference_mode()
def inner_local(model,bank):
    # Fixed independent source and donor sets. Match false positives by rank.
    model.eval()
    score=[]; correct=[]; group=[]; glare=[]
    for j in range(12):
        rng=seed_rng(bank.device,873,data.TINY,j)
        batch,_,same,highlight=bank.paired(96,rng,"inner","tiny")
        raw,mask=model(*batch[:2])
        ok=batch[-2]
        n=len(ok)
        labels=batch[5][ok,2]
        out=raw[:n][ok]
        mask_ok=((mask[:n][ok,0].sigmoid()>.5)&(mask[:n][ok,1].sigmoid()<.55)).sum((1,2))>=3
        score.append(torch.stack([out[:,0].masked_fill(~mask_ok,-40),raw[n:][ok,0]],1))
        correct.append(out[:,11:17].argmax(1).eq(labels))
        group.append(same[ok])
        glare.append(highlight[ok])
    s,ok,same,highlight=map(lambda x:torch.cat(x).cpu().numpy(),(score,correct,group,glare))
    if len(s)==0:
        raise ValueError("No held-out synthetic pairs")
    cut=float(np.quantile(s[:,1],.95))
    accepted=(s[:,0]>cut)&ok
    rate=lambda values:float(values.mean()) if len(values) else None
    real=[]
    for i in range(10):
        p,m,pad,filters,target,_=bank.real.bags(12,seed_rng(bank.device,901,data.REAL,i),"inner")
        raw,_=model(p,m)
        logits=raw.new_zeros((*pad.shape,raw.shape[1]))
        logits[pad]=raw
        q=data.membership(logits,filters)*pad
        real.append(data.exact_count_loss(q,target).item())
    model.train()
    return dict(selection=rate(accepted),matched_fpr=rate(s[:,1]>cut),count=len(s),
                same_color_recall=rate(accepted[same]),highlight_fpr=rate((s[:,1]>cut)[highlight]),
                threshold_logit=cut,real_count_nll=float(np.mean(real)))


@torch.inference_mode()
def inner_scene(model,bank):
    model.eval()
    probs=[]
    ids=bank.pools["inner"][4].tolist()
    for i in range(0,len(ids),32):
        group=[k for k in ids[i:i+32] if k in bank.scene_counts]
        if not group:
            continue
        rgb=bank.bank.rgb[group].float()/255
        raw,masks=model(rgb)
        target=[bank.scene_counts[k][0] for k in group]
        m=[torch.as_tensor(np.stack([g[0][j] for g in target])[:,None],device=bank.device) for j in range(3)]
        num=torch.as_tensor([g[1] for g in target],device=bank.device)
        q=data.membership(raw,m)*((masks>.0).sum((2,3))>=2)
        probs.extend(data.exact_probability(q,num).tolist())
    model.train()
    return dict(selection=float(np.mean(probs)),exact_captions=len(probs))


def write_json(path,result):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result,indent=2),encoding="utf8")
    tmp.replace(path)


def save(model,path):
    tmp=path.with_suffix(".tmp")
    state={k:(v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu())
           for k,v in model.state_dict().items()}
    torch.save(state,tmp)
    tmp.replace(path)


def fit(bank,arm,mode,seed,batch,steps,stop_after=None,fixed=False,mix_stages=False,log_interval=50,
        initial=None,rate_scale=1.):
    limit=steps if stop_after is None else stop_after
    if not 1<=limit<=steps:
        raise ValueError("Stop point must be in the original training schedule")
    if log_interval<1:
        raise ValueError("Log interval must be positive")
    dev=bank.device
    name=tag(arm,mode)
    result=OUT/f"{name}_{seed}.json"
    model=build(arm,mode,seed,dev,pretrained=initial is None)
    if initial is not None:
        model.load_state_dict(r.v.load(initial))
    opt=optimizer(model,dev,arm,mode)
    for group in opt.param_groups:
        group["initial_lr"]*=rate_scale
    scaler=torch.amp.GradScaler("cuda",enabled=dev.type=="cuda",init_scale=128)
    marks=[limit] if fixed else sorted(
        {n for n in CHECKPOINTS if n<=limit}|({240} if mode=="legacy" and limit>=240 else set())|{limit})
    epochs=[]
    history={}
    run=dict(batch=batch,checkpoints=history,training=epochs,startup=bank.startup,coverage=bank.real.coverage,
             scheduled_steps=steps,stop_after=limit,fixed_checkpoint=fixed,mixed_stages=mix_stages,
             color_comparison_scenes=0 if bank.color_scenes is None else len(bank.color_scenes))
    stepslog=defaultdict(int)
    bag_draws=torch.zeros(len(bank.real.groups["fit"][0]),dtype=torch.long,device=dev)
    scene_caption_draws=0
    start=time.perf_counter()
    step_losses=[]
    pool=schedule("mixed" if mix_stages else "staged" if arm=="scene" else mode,steps)
    for index,(kind,n) in enumerate(pool[:limit],1):
        if mode=="legacy":
            phase=(index-1)//10/24 if index<=240 else (index-241)/max(1,steps-240)
            lr_factor=(.083+.917*(1+math.cos(math.pi*phase))/2) if index<=240 else (.083+.167*(1+math.cos(math.pi*phase))/2)
        else:
            progress=(index-1)/max(1,steps-1)
            lr_factor=.12+.88*(1+math.cos(math.pi*progress))/2
        for g in opt.param_groups:
            g["lr"]=g["initial_lr"]*lr_factor
        rng=seed_rng(dev,seed,kind,n)
        torch.manual_seed(seed*1009+kind*13007+n)
        value,details=step(model,opt,scaler,bank,arm,mode,kind,batch,rng)
        if details is not None:
            step_losses.append(torch.stack((value,details[0].float(),details[1].float())))
            if details[2] is not None:
                bag_draws.scatter_add_(0,details[2],torch.ones_like(details[2]))
        else:
            step_losses.append(value[None])
            if arm=="scene" and mode!="noexact":
                scene_caption_draws+=max(1,batch//3)
        stepslog[kind]+=1
        if index%log_interval==0 or index in marks:
            avg=torch.stack(step_losses).mean(0).tolist()
            if not np.isfinite(avg[0]):
                raise ValueError("Non-finite training loss")
            row=dict(step=index,loss=avg[0],training_crops_mean=avg[1:],elapsed_s=time.perf_counter()-start,
                     updates=dict(stepslog),gpu=r.gpu_util(dev),lr=opt.param_groups[0]["lr"],
                     exact_bag_draws=int(bag_draws.sum())+scene_caption_draws,
                     unique_exact_bags=int((bag_draws>0).sum()) if arm!="scene" else None)
            epochs.append(row)
            step_losses.clear()
            print(arm,mode,seed,row,flush=True)
        if fixed and index==limit:
            save(model,OUT/f"{name}_{seed}.pt")
            run["selected_step"]=index
            run["elapsed_s"]=time.perf_counter()-start
            if dev.type=="cuda":
                run["peak_reserved_gib"]=torch.cuda.max_memory_reserved(dev)/2**30
            write_json(result,run)
        elif index in marks:
            selected=inner_scene(model,bank) if arm=="scene" else inner_local(model,bank)
            save(model,OUT/f"{name}_{seed}_{index}.pt")
            history[str(index)]=selected
            # The 240-step control is kept separate from model selection.
            eligible={k:v for k,v in history.items() if int(k)!=240} or history
            best=max(eligible,key=lambda k:(eligible[k]["selection"],-eligible[k].get("real_count_nll",0)))
            run["selected_step"]=int(best)
            shutil.copyfile(OUT/f"{name}_{seed}_{best}.pt",OUT/f"{name}_{seed}.pt")
            write_json(result,run)
            print("inner",name,seed,index,selected,"selected",best,flush=True)


def infer(arm,mode,seed,split,shard,device,step_number):
    name=tag(arm,mode)
    if arm=="scene":
        scene.infer(seed,device,split,None,tag=name,output=OUT)
        return
    stored=f"{name}_{seed}" if step_number is None else f"{name}_{seed}_{step_number}"
    model=build(arm,mode,seed,device,pretrained=False)
    model.load_state_dict(r.v.load(OUT/f"{stored}.pt"))
    model.eval()
    if step_number is not None:
        name=f"{name}{step_number}"
    r.OUT=OUT
    r.infer(name,seed,split,shard,device,model)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("probe","fit","infer"))
    p.add_argument("--arm",required=True)
    p.add_argument("--mode",required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--batch",type=int)
    p.add_argument("--steps",type=int,default=1200)
    p.add_argument("--stop-after",type=int)
    p.add_argument("--checkpoint",type=int)
    p.add_argument("--all-fit",action="store_true",help="Use every scene and save only the scheduled checkpoint")
    p.add_argument("--mix-stages",action="store_true",help="Interleave clear, occluded and real-scene updates")
    p.add_argument("--color-supervision",action="store_true",help="Train the scene branch on labeled color comparisons")
    p.add_argument("--split",choices=("val","test"),default="val")
    p.add_argument("--shard",type=int,default=0)
    a=p.parse_args()
    if (a.arm,a.mode) not in MODES:
        p.error("Unknown configuration")
    OUT.mkdir(exist_ok=True)
    _,dev=r.v.setup(a.seed)
    if a.stage=="infer":
        infer(a.arm,a.mode,a.seed,a.split,a.shard,dev,a.checkpoint)
        return
    if a.all_fit and a.stage!="fit":
        p.error("--all-fit applies to fit")
    if a.color_supervision and a.arm!="scene":
        p.error("--color-supervision requires the scene arm")
    bank=data.Data(dev,all_scenes=a.all_fit,color_supervision=a.color_supervision)
    if a.stage=="probe":
        write_json(OUT/f"probe_{tag(a.arm,a.mode)}_{a.seed}.json",probe(bank,a.arm,a.mode,a.seed))
    else:
        if not a.batch or a.batch<1:
            p.error("--batch is required for fit")
        fit(bank,a.arm,a.mode,a.seed,a.batch,a.steps,a.stop_after,fixed=a.all_fit,mix_stages=a.mix_stages)


if __name__=="__main__":
    main()

