"""Read the existing R50 pyramid before peak suppression and learn filtered counts."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F

from scripts.research.dense_counts import queries
from scripts.research.fragment_votes import EXTRA, H, W, Scenes, images, v
from scripts.research.visible_fragment_detector import gpu_util
from kaggle_suite_common import amp
from src.models.slot_program import EXACT, EXIST, NONE, MORE, FEWER, SAME

OUT = Path("detail")
ARMS = ("pyramid", "attention", "paired", "stride4", "maskset")
SEEDS = (421, 853)
K = 12


def resize(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class Pyramid(nn.Module):
    def __init__(self, step=2):
        super().__init__()
        self.net = v.ResNetCenters(full=True)
        self.net.load_state_dict(v.load(v.PRIORS/"r50_img.pt"))
        self.net.head.requires_grad_(False)
        self.step = step
        self.combine = nn.Sequential(nn.Conv2d(131, 48, 1), nn.GroupNorm(8,48), nn.SiLU(),
                                     nn.Conv2d(48,48,3,padding=1), nn.GroupNorm(8,48), nn.SiLU())

    def features(self, rgb):
        b = self.net
        z = b.stem[:3]((rgb-b.mean)/b.std)
        a = b.layer1(b.stem[3](z))
        c = b.layer2(a)
        d = b.layer3(c)
        deep = b.lateral4(b.layer4(d))
        for t in (b.lateral3(d), b.lateral2(c), b.lateral1(a)):
            deep = t + resize(deep,t.shape[-2:])
        size = (H//self.step,W//self.step)
        return self.combine(torch.cat([resize(z,size),resize(deep,size),resize(rgb,size)],1))


class Count(Pyramid):
    def __init__(self, arm):
        super().__init__(4 if arm=="stride4" else 2)
        self.attention = arm in ("attention","paired","stride4")
        self.mass = nn.Conv2d(48, 96, 1)
        nn.init.constant_(self.mass.bias,-4)
        self.embed = nn.Sequential(nn.Linear(96,48,bias=False),nn.SiLU(),nn.Linear(48,48,bias=False))
        self.key = nn.Conv2d(48,48,1)
        self.register_buffer("bins",torch.arange(K+1,dtype=torch.float32))

    def masses(self, rgb):
        f = self.features(rgb)
        return F.softplus(self.mass(f).float()) / (f.shape[-1]*f.shape[-2]/18), self.key(f)

    def counts(self, fields, i, mask):
        mass, key = fields
        pieces=[]
        if not self.attention:
            return (mass.sum((2,3))[i]*mask).sum(1)
        for start in range(0,len(i),48):
            ids=i[start:start+48]; q=mask[start:start+48].float()
            with torch.autocast(device_type=mass.device.type,enabled=False):
                density=torch.einsum("bchw,bc->bhw",mass[ids],q)
            norm=q.sum(1,keepdim=True).sqrt().clamp_min(1)
            query=self.embed(q/norm).float()
            attention=(key[ids].float()*query[:,:,None,None]).sum(1) / 48**.5
            density=density*(.15+1.7*attention.sigmoid())
            pieces.append(density.sum((1,2)))
        return torch.cat(pieces)

    def distributions(self,a):
        # Discrete bell-shaped count likelihood with shared scale for all requests.
        return (-(a[:,None]-self.bins).square()/(2*.32**2)).softmax(1)


def statements(pa,pb,op,num):
    bins = torch.arange(pa.shape[-1],device=pa.device)
    gt = bins[:,None]>bins[None,:]
    eq = (pa*pb).sum(-1)
    more = ((pa@gt.to(pa.dtype))*pb).sum(-1)
    fewer = ((pa@gt.T.to(pa.dtype))*pb).sum(-1)
    exact = pa.gather(1,num.clamp(0,K)[:,None])[:,0]
    p = torch.where(op==EXACT,exact,1-pa[:,0])
    for code, q in ((NONE,pa[:,0]),(MORE,more),(FEWER,fewer),(SAME,eq)):
        p=torch.where(op==code,q,p)
    return torch.logit(p.clamp(1e-5,1-1e-5)).clamp(-10,10)


class MaskSet(Pyramid):
    def __init__(self):
        super().__init__()
        self.tokens=nn.Parameter(torch.randn(K,48)*.09)
        self.context=nn.Sequential(nn.Conv2d(48,48,3,stride=2,padding=1),nn.GroupNorm(8,48),nn.SiLU())
        self.pos=nn.Conv2d(2,48,1)
        self.cross=nn.MultiheadAttention(48,4,batch_first=True)
        self.self_attention=nn.MultiheadAttention(48,4,batch_first=True)
        self.norm=nn.LayerNorm(48)
        self.classify=nn.Linear(48,17) # presence, size, color, material Р В Р’В Р Р†Р вЂљРЎС™Р В Р вЂ Р В РІР‚С™Р Р†Р вЂљРЎСљ shape
        self.mask=nn.Linear(48,48)
        nn.init.constant_(self.classify.bias[0],-2)

    def forward(self,rgb):
        f=self.features(rgb)
        y,x=torch.meshgrid(torch.linspace(-1,1,f.shape[-2],device=f.device),
                           torch.linspace(-1,1,f.shape[-1],device=f.device),indexing="ij")
        f=f+self.pos(torch.stack([x,y])[None])
        context=self.context(f).flatten(2).transpose(1,2)
        q=self.tokens[None].expand(len(rgb),-1,-1)
        q=self.norm(q+self.cross(q,context,context,need_weights=False)[0])
        q=self.norm(q+self.self_attention(q,q,q,need_weights=False)[0])
        return self.classify(q).float(),torch.einsum("bqd,bdhw->bqhw",self.mask(q),f).float()


def mask_loss(raw, pred, instance, labels, timing=None):
    pred=pred.flatten(2)
    ids=instance.flatten(1)
    target=(ids[:,None]==torch.arange(1,labels.shape[1]+1,device=ids.device)[None,:,None]).float()
    # Dice cost is computed before matching with all labelled masks in parallel.
    probs=pred.sigmoid()
    inter=probs@target.transpose(1,2)
    area=target.sum(2)
    cost=1-(2*inter+1)/(probs.sum(2)[:,:,None]+area[:,None]+1)
    for logits,kind in zip(raw[:,:,1:].split((2,8,6),2),labels.unbind(2)):
        logp=logits.log_softmax(2).gather(2,kind[:,None].expand(-1,K,-1).clamp_min(0))
        cost-=.18*logp*(kind[:,None]>=0)
    q,b,s=[],[],[]
    costs=cost.detach().float().cpu().numpy()
    areas=area.detach().cpu().numpy()
    match_start=time.perf_counter() if timing is not None else 0.
    for k,(matrix,observed) in enumerate(zip(costs,areas)):
        cols=np.flatnonzero(observed>=2)
        if len(cols):
            r,c=linear_sum_assignment(matrix[:,cols])
            q.extend(r); b.extend([k]*len(r)); s.extend(cols[c])
    if timing is not None:
        timing["matching_cpu_ms"]=1000*(time.perf_counter()-match_start)
    index=[torch.as_tensor(i,device=ids.device,dtype=torch.long) for i in (b,q,s)]
    bi,qi,si=index
    hit=torch.zeros_like(raw[:,:,0])
    hit[bi,qi]=1
    ce=F.binary_cross_entropy_with_logits(raw[:,:,0],hit,reduction="none")
    # Weak masks can omit tiny real objects.
    loss=(ce*torch.where(hit>0,1.,.25)).mean()
    if len(b):
        truth=target[bi,si]; logit=pred[bi,qi]; p=probs[bi,qi]
        loss+= (1-(2*(p*truth).sum(1)+1)/(p.sum(1)+truth.sum(1)+1)).mean()
        focal=F.binary_cross_entropy_with_logits(logit,truth,reduction="none")*(p-truth).square()
        loss+= ((focal*truth).sum(1)/truth.sum(1).clamp_min(1)+ (focal*(1-truth)).mean(1)).mean()
        for logits,kind in zip(raw[bi,qi,1:].split((2,8,6),1),labels[bi,si].T):
            use=kind>=0
            ce=F.cross_entropy(logits,kind.clamp_min(0),reduction="none")
            loss+=.2*(ce*use).sum()/use.sum().clamp_min(1)
    return loss


def masked_instance_rgb(bank, idx, rgb, anchor):
    """A detached highlight perturbation changes appearance without changing object count."""
    inst=bank.instances[idx]
    x=torch.arange(W,device=rgb.device)[None,None,:]
    y=torch.arange(H,device=rgb.device)[None,:,None]
    box=bank.box[idx,anchor]
    center=box[:,:2]+box[:,2:]*(.25+.5*torch.rand((len(idx),2),device=rgb.device))
    dist=((x-center[:,0,None,None])/3).square()+((y-center[:,1,None,None])/2).square()
    alpha=(.40*torch.exp(-.5*dist)*(inst==anchor[:,None,None]+1))[:,None]
    return rgb+(1-rgb)*alpha


def cover_existing(bank, idx, rgb):
    """Paste an opaque foreground over a known object, retaining an observable fragment."""
    s=bank
    n=len(idx); device=idx.device
    valid=s.anchors[idx]&(s.labels[idx]>=0).all(2)
    slot=(torch.rand(valid.shape,device=device)*valid).argmax(1)
    target=s.box[idx,slot]
    di,dj=s.donors[torch.randint(len(s.donors),(n,),device=device)].T
    donor=s.box[di,dj]
    extent=donor[:,2:] * (.8+.4*torch.rand((n,1),device=device))
    origin=target[:,:2]+target[:,2:]*(.3+.4*torch.rand((n,2),device=device))-extent/2
    uv=(s.grid-origin[:,None,None])/extent[:,None,None]
    alpha=F.grid_sample(s.masks[di,dj,None].float()/255,2*uv-1,align_corners=False)[:,0]
    grid=(donor[:,None,None,:2]+uv*donor[:,None,None,2:])/rgb.new_tensor([W,H])
    patch=F.grid_sample(s.rgb[di].float()/255,2*grid-1,align_corners=False)
    inst=s.instances[idx]
    target_mask=inst==slot[:,None,None]+1
    visible=target_mask&(alpha<.5)
    ratio=visible.sum((1,2))/target_mask.sum((1,2)).clamp_min(1)
    # Permit only foreground and the designated target to interact.
    other=(alpha*(inst!=0)*~target_mask).sum((1,2))
    ok=(ratio>=.04)&(ratio<=.45)&(visible.sum((1,2))>=5)&(other<2)&(di!=idx)
    ok &= valid.any(1)&(origin>=0).all(1)&((origin+extent)<=rgb.new_tensor([W,H])).all(1)
    ok &= (s.labels[di,dj] != s.labels[idx,slot]).any(1)
    composite=rgb+(patch-rgb)*alpha[:,None]*ok[:,None,None,None]
    return composite,s.labels[idx,slot],s.labels[di,dj],ok


def qmask(label):
    size,color,joint=label.clamp_min(0).unbind(1)
    return F.one_hot((size*8+color)*6+joint,96).bool()


def count_train(model, bundle, table, arm):
    rgb, idx, pairs, views, extra = bundle
    i, q, y= pairs
    ma,mb,op,num=table
    combined=torch.cat([rgb,*[v[0] for v in views]]) if views else rgb
    maps=model.masses(combined)
    matrix=torch.cat([ma[q],mb[q]])
    ends=torch.cat([i,i])
    ac,bc=model.counts(maps,ends,matrix).chunk(2)
    scores=statements(model.distributions(ac),model.distributions(bc),op[q],num[q])
    a,b=scores.reshape(-1,2).unbind(1)
    loss=F.binary_cross_entropy_with_logits(a-b,y[::2])
    ex=(op[q]==EXACT)&(y>0)
    error=F.smooth_l1_loss(ac,num[q].float(),beta=.20,reduction="none")
    loss+=.60*(error*ex).sum()/ex.sum().clamp_min(1)
    pos=(op[q]==MORE)|(op[q]==FEWER)
    gap=torch.where(op[q]==MORE,ac-bc,bc-ac)
    loss+=.25*(F.relu(.65-gap)*pos*y).sum()/(pos*y).sum().clamp_min(1)
    for j,(_,mask,usable,target) in enumerate(views,1):
        base_idx=torch.arange(len(idx),device=idx.device)
        left,right=model.counts(maps,torch.cat([base_idx,base_idx+j*len(idx)]),mask.repeat(2,1)).chunk(2)
        err=F.smooth_l1_loss(right-left,target,beta=.20,reduction="none")
        loss+=.20*(err*usable).sum()/usable.sum().clamp_min(1)
    if extra is not None:
        mask,usable=extra
        original=torch.arange(len(idx),device=idx.device)
        left,right=model.counts(maps,torch.cat([original,original+2*len(idx)]),mask.repeat(2,1)).chunk(2)
        err=F.smooth_l1_loss(right-left,torch.ones_like(right),beta=.2,reduction="none")
        loss+=.16*(err*usable).sum()/usable.sum().clamp_min(1)
    return loss


def run_epoch(model, arm, bank, indices, opt, scaler, qscene, table, label, timing=False, accumulation=1):
    device=bank.device
    totals=[]; parts=np.zeros(5)
    opt.zero_grad(set_to_none=True)
    for step,idx in enumerate(indices):
        start=time.perf_counter()
        rgb=bank.rgb[idx].float()/255
        if arm in ("paired","maskset"):
            rgb_pair, instance, _, attrs, added=bank.pair(idx)
        if arm=="maskset":
            packed=(rgb_pair,instance,attrs[:,1:])
        else:
            si,q=(idx[:,None]==qscene[None]).nonzero(as_tuple=True)
            rows=(2*q[:,None]+torch.arange(2,device=device)).flatten()
            views=[]
            extra=None
            if arm=="paired":
                use=added.flatten(1).sum(1)>=4
                donor=qmask(attrs[len(idx):,-1])
                # Keep highlight perturbations on an already observed class.
                slots=(bank.anchors[idx].float()*torch.rand_like(bank.anchors[idx],dtype=torch.float32)).argmax(1)
                label0=bank.labels[idx,slots]
                covered,target,foreground,cover_ok=cover_existing(bank,idx,rgb)
                extra=qmask(foreground),cover_ok
                views=[(rgb_pair[len(idx):],donor,use,use.float()),
                       (covered,qmask(target),cover_ok,torch.zeros_like(use,dtype=torch.float32)),
                       (masked_instance_rgb(bank,idx,rgb,slots),qmask(label0),(label0>=0).all(1),torch.zeros_like(use,dtype=torch.float32))]
            packed=(rgb,idx,(si.repeat_interleave(2),rows,label[rows]),views,extra)
        host_submit=time.perf_counter()-start
        if timing and device.type=="cuda":
            torch.cuda.synchronize()
        prepared=time.perf_counter()
        with amp(device):
            if arm=="maskset":
                raw,mask=model(packed[0])
            else:
                loss=count_train(model,packed,table,arm)
        if arm=="maskset":
            loss=mask_loss(raw,mask,*packed[1:])
        first=step//accumulation*accumulation
        end=min(first+accumulation,len(indices))
        batch_total=sum(len(ids) for ids in indices[first:end])
        scaler.scale(loss*(len(idx)/batch_total)).backward()
        if step+1==end:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        if timing and device.type=="cuda":
            torch.cuda.synchronize()
        parts += [len(idx),prepared-start,time.perf_counter()-prepared,1,host_submit]
        totals.append(loss.detach())
    if not totals:
        raise ValueError("No usable training batch")
    n,prep,train,steps,host=parts
    return torch.stack(totals).mean().item(),dict(scenes_per_s=n/max(prep+train,1e-5),prepare_ms=1000*prep/steps,
                                                 step_ms=1000*train/steps,host_aug_ms=1000*host/steps)


def fit(arm, seed, epochs, device, limit=None):
    names=pd.read_csv(v.DATA/"train.csv").image.unique()
    t=time.perf_counter()
    rgb=images("train",names)
    t_decode=time.perf_counter()-t
    t=time.perf_counter()
    bank=Scenes(device,rgb)
    if device.type=="cuda":
        torch.cuda.synchronize()
    del rgb
    startup=dict(jpeg_decode_s=t_decode,mask_rasterize_upload_s=time.perf_counter()-t)
    frame=pd.read_csv(v.DATA/"train.csv")
    names, rows, qscene, table=queries(frame,device)
    label=torch.from_numpy(frame.label.to_numpy()[rows.ravel()]).float().to(device)
    ids=torch.arange(len(names),device=device)[:limit]
    if arm!="maskset":
        ids=ids[torch.isin(ids,qscene)]
    if len(ids)==0:
        raise ValueError("No train scenes with supported captions")
    print("startup",startup,flush=True)
    batches=(8,16,32,64,128) if device.type=="cuda" else (min(2,len(ids)),)
    probes=[]

    def setup():
        torch.manual_seed(seed)
        m=(MaskSet() if arm=="maskset" else Count(arm)).to(device,memory_format=torch.channels_last).train()
        m.net.apply(lambda layer: layer.eval() if isinstance(layer,nn.BatchNorm2d) else None)
        opt=torch.optim.AdamW([dict(params=m.net.parameters(),lr=6e-5),
                               dict(params=[p for n,p in m.named_parameters() if not n.startswith("net.")],lr=6e-4)],
                              fused=device.type=="cuda")
        return m,opt,torch.amp.GradScaler("cuda",enabled=device.type=="cuda")

    if device.type=="cuda":
        for n in batches:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model=opt=scaler=None
            try:
                model,opt,scaler=setup()
                train_args=(model,arm,bank,[ids[:n]],opt,scaler,qscene,table,label,True)
                run_epoch(*train_args)
                train_args=(model,arm,bank,[ids[:n]]*3,opt,scaler,qscene,table,label,True)
                _,probe=run_epoch(*train_args)
                reserved=torch.cuda.max_memory_reserved()
                probe.update(batch=n,allocated_gib=torch.cuda.max_memory_allocated()/2**30,reserved_gib=reserved/2**30,
                             vram_fraction=reserved/torch.cuda.get_device_properties(device).total_memory,gpu_util=gpu_util(device))
                probes.append(probe)
            except torch.cuda.OutOfMemoryError:
                probes.append(dict(batch=n,oom=True))
            finally:
                model=opt=scaler=train_args=None
            print("probe",probes[-1],flush=True)
        valid=[p for p in probes if p.get("vram_fraction",1)<.91]
        if not valid:
            raise RuntimeError("No batch fits")
        best=max(p["scenes_per_s"] for p in valid)
        n=max(p["batch"] for p in valid if p["scenes_per_s"]>=.97*best)
        torch.cuda.empty_cache()
    else:
        n=batches[0]
    model,opt,scaler=setup()
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=5e-6)
    history=[]
    for ep in range(epochs):
        t=time.perf_counter()
        index=ids[torch.randperm(len(ids),device=device)].split(n)
        loss,profile=run_epoch(model,arm,bank,index,opt,scaler,qscene,table,label,accumulation=max(1,128//n) if device.type=="cuda" else 1)
        if not np.isfinite(loss):
            raise RuntimeError("Non-finite loss")
        report=dict(epoch=ep+1,loss=loss,seconds=time.perf_counter()-t,gpu_util=gpu_util(device))
        history.append(report)
        schedule.step()
        print(arm,seed,report,flush=True)
    path=OUT/f"{arm}_{seed}"
    kept={k:t.detach().cpu() for k,t in model.state_dict().items()}
    torch.save(kept,path.with_suffix(".pt"))
    path.with_suffix(".json").write_text(json.dumps(dict(startup=startup,probes=probes,batch=n,effective_batch=128 if device.type=="cuda" else n,epochs=history),indent=2))


def mask_members(raw,masks):
    a,b,c=[r.softmax(-1) for r in raw[:,:,1:].split((2,8,6),2)]
    classes=(a[...,None,None]*b[...,None,:,None]*c[...,None,None,:]).flatten(2)
    visible=(masks.sigmoid()>.5).sum((-2,-1))>=2
    return classes*(raw[:,:,0].sigmoid()*visible)[...,None]


def poisson(q):
    p=F.pad(torch.ones((len(q),1),device=q.device),(0,K))
    for x in q.clamp(0,1).unbind(1):
        p=p*(1-x[:,None])+F.pad(p[:,:-1],(1,0))*x[:,None]
    return p


def mask_boxes(names,raw,masks,include_attributes=False):
    ranks=raw[:,:,0].sigmoid().cpu().numpy()
    color=raw[:,:,3:11].softmax(-1).cpu().numpy()
    maps=masks.sigmoid().cpu().numpy()
    attributes=raw.cpu().numpy() if include_attributes else None
    result=[]
    for row,(name,score,inst,col) in enumerate(zip(names,ranks,maps,color)):
        for q in np.flatnonzero(score>=.05):
            yy,xx=np.nonzero(inst[q]>.5)
            if len(xx)<2:
                continue
            i,j=np.unravel_index(inst[q].argmax(),inst[q].shape)
            result.append(dict(image=str(name),prob=float(score[q]),color=col[q].tolist(),
                               peak=[int(2*j+1),int(2*i+1)],
                               box=[int(2*xx.min()),int(2*yy.min()),int(2*(xx.max()-xx.min()+1)),
                                    int(2*(yy.max()-yy.min()+1))]))
            if attributes is not None:
                result[-1].update(attributes=attributes[row,q].tolist(),pixels=4*len(xx))
    return result


@torch.inference_mode()
def infer_count(arm, seed, split, device, limit=None):
    model=(MaskSet() if arm=="maskset" else Count(arm)).to(device,memory_format=torch.channels_last)
    model.load_state_dict(v.load(OUT/f"{arm}_{seed}.pt"))
    model.eval()
    frame=pd.read_csv(v.DATA/f"{split}.csv")
    if limit:
        frame=frame[frame.image.isin(frame.image.unique()[:limit])].reset_index(drop=True)
    names,rows,si,table=queries(frame,device)
    ma,mb,op,num=table
    scores=np.zeros(len(frame))
    debug=[]
    count_a=np.full(len(frame),-1.)
    for start in range(0,len(names),48 if device.type=="cuda" else 4):
        batch=names[start:start+(48 if device.type=="cuda" else 4)]
        i=torch.where((si>=start)&(si<start+len(batch)))[0]
        if not len(i):
            continue
        q=(2*i[:,None]+torch.arange(2,device=device)).flatten()
        rgb=torch.from_numpy(images(split,batch)).to(device).permute(0,3,1,2).float()/255
        j=(si[i]-start).repeat_interleave(2)
        with amp(device):
            if arm=="maskset":
                raw, masks=model(rgb)
            else:
                fields=model.masses(rgb)
                ca,cb=model.counts(fields,torch.cat([j,j]),torch.cat([ma[q],mb[q]])).chunk(2)
        if arm=="maskset":
            members=mask_members(raw,masks)
            pa=poisson((members[j]*ma[q,None]).sum(-1))
            pb=poisson((members[j]*mb[q,None]).sum(-1))
            if split=="val":
                debug.extend(mask_boxes(batch,raw,masks))
        else:
            pa,pb=model.distributions(ca),model.distributions(cb)
        logits=statements(pa,pb,op[q],num[q]).reshape(-1,2).cpu().numpy()
        indices=rows[i.cpu().numpy()]
        count_a[indices.ravel()]=(pa*torch.arange(K+1,device=device)).sum(1).cpu().numpy()
        scores[indices[:,0]]=logits[:,0]-logits[:,1]
        scores[indices[:,1]]=-scores[indices[:,0]]
    np.savez_compressed(OUT/f"{arm}_{seed}_{split}.npz",ids=frame.id.to_numpy(),score=scores,count_a=count_a)
    if debug and split=="val":
        (OUT/f"maskset_{seed}_objects.json").write_text(json.dumps(debug))


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("action",choices=("fit","val","test"))
    cli.add_argument("arm",choices=ARMS)
    cli.add_argument("seed",type=int)
    cli.add_argument("--epochs",type=int,default=18)
    cli.add_argument("--limit",type=int)
    args=cli.parse_args()
    _,device=v.setup(args.seed)
    OUT.mkdir(exist_ok=True)
    if args.action=="fit":
        fit(args.arm,args.seed,args.epochs,device,args.limit)
    else:
        infer_count(args.arm,args.seed,args.action,device,args.limit)

