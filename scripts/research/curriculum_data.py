"""Train-only donors, real weak peaks and count supervision."""
from collections import defaultdict
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.research import rim_fragments as rim
from scripts.research.fragment_votes import EXTRA, v
from native_centers import canonical_form, parse_program
from src.models.slot_program import EXACT, EXCLUSIVE_GROUPS, FEWER, MORE, SAME

WEAK = 0
CLEAR = 1
MIDDLE = 2
TINY = 3
REAL = 4
TAIL = (CLEAR,MIDDLE,TINY,REAL)
KINDS = ("size","color","joint")
WORDS = [EXCLUSIVE_GROUPS[0],EXCLUSIVE_GROUPS[1],
         [f"{m} {s}" for m in EXCLUSIVE_GROUPS[2] for s in EXCLUSIVE_GROUPS[3]]]
VOCABULARY = [set(group) for group in EXCLUSIVE_GROUPS[:2]]+[set(EXCLUSIVE_GROUPS[2]+EXCLUSIVE_GROUPS[3])]


def exact_rows():
    frame=pd.read_csv(v.DATA/"train.csv")
    selected=frame[frame.category.eq("counting") & frame.label.eq(1)]
    for row in selected.itertuples(index=False):
        p=parse_program(canonical_form(row.caption,row.category))
        if p is not None and p.op==EXACT and not p.relation:
            yield row.image,p.a,p.number


def attribute_mask(words):
    return [np.array([v.intersection(words)<=set(label.split()) for label in group])
            for v,group in zip(VOCABULARY,WORDS)]


def membership(raw,mask):
    p=raw[...,0].sigmoid()
    for logits,m in zip(raw[...,1:].split((2,8,6),-1),mask):
        p=p*(logits.softmax(-1)*m).sum(-1)
    return p


def exact_probability(p,num):
    """Poisson binomial P(sum Bernoulli(p_i) = num) for all candidate slots."""
    coef=F.one_hot(torch.zeros(len(p),dtype=torch.long,device=p.device),p.shape[1]+1).float()
    for q in p.float().unbind(1):
        q=q[:,None]
        coef=coef*(1-q)+F.pad(coef[:,:-1],(1,0))*q
    return coef.gather(1,num[:,None]).squeeze(1)


def exact_count_loss(p,num):
    return -exact_probability(p,num).clamp_min(1e-8).log().mean()


def donor_pools(bank,split):
    donors=bank.donors[split[bank.donors[:,0]]]
    color=bank.labels[donors[:,0],donors[:,1],1]
    groups=[torch.where(color==c)[0] for c in range(8)]
    width=max(map(len,groups))
    return donors,torch.stack([F.pad(g,(0,width-len(g)),value=-1) for g in groups])


def intact(bank,source,grid):
    si,k=source.T
    box=bank.box[si,k]
    points=grid[None]+(box[:,:2]+box[:,2:]/2)[:,None,None]
    rgb=rim.sample(bank.rgb[si],points)/255
    instance=rim.sample(bank.instances[si,None],points,"nearest")[:,0]
    mask=instance.eq(k[:,None,None]+1)
    front=torch.zeros_like(mask)
    gaussian=(-.5*(grid/6).square().sum(-1)).exp().expand(len(si),-1,-1)
    marks=torch.stack([front.float(),gaussian],1)
    truth=torch.ones(len(si),dtype=torch.bool,device=si.device)
    return rgb,marks,mask,front,truth,bank.labels[si,k],truth[:0],truth


def draw_index(ids,n,rng):
    return ids[torch.randint(len(ids),(n,),device=ids.device,generator=rng)]


class Data:
    def __init__(self,device,all_scenes=False,color_supervision=False):
        self.bank,_,self.startup=rim.load_bank(device)
        bank=self.bank
        self.device=device
        self.grid=rim.coords(device)
        with np.load(EXTRA/"fragment_train.npz") as z:
            names,boxes,labels,keep,masks=(z[k] for k in ("names","box","labels","keep","mask"))
        self.names=names
        self.lookup={str(n):i for i,n in enumerate(names)}
        counts=list(exact_rows())
        self.scene_counts=defaultdict(list)
        for name,words,total in counts:
            self.scene_counts[self.lookup[name]].append((attribute_mask(words),total))
        rng=np.random.default_rng(90317)
        order=rng.permutation(len(names))
        inside=np.ones(len(names),bool)
        if not all_scenes:
            inside[order[:max(1,len(names)//6)]]=False
        self.pools={}
        for part,selected in (("fit",inside),("inner",~inside)):
            if not selected.any():
                continue
            flag=torch.as_tensor(selected,device=device)
            sources=torch.where(flag&(bank.anchors&(bank.labels>=0).all(-1)).any(1))[0]
            all_scenes=torch.where(flag)[0]
            donors,modes=donor_pools(bank,flag)
            mask_area=(masks>127).mean((2,3))
            clear=keep&(labels>=0).all(-1)&(mask_area>.30)&(boxes[:,:,2]<=36)&(boxes[:,:,3]<=36)
            ai,aj=np.nonzero(selected[:,None]&clear)
            self.pools[part]=sources,donors,modes,torch.as_tensor(np.stack([ai,aj],1),device=device),all_scenes
        counted=sorted(self.scene_counts)
        self.counted=torch.as_tensor([i for i in counted if inside[i]],dtype=torch.long,device=device)
        templates=attribute_mask(())
        targets=np.zeros(len(names),dtype=np.int64)
        filters=[np.zeros((len(names),1,len(template)),dtype=bool) for template in templates]
        for i in counted:
            masks,target=self.scene_counts[i][0]
            targets[i]=target
            for out,mask in zip(filters,masks):
                out[i,0]=mask
        self.scene_filters=[torch.as_tensor(values,device=device) for values in filters]
        self.scene_target=torch.as_tensor(targets,device=device)
        self.color_scenes=None
        if color_supervision:
            colors=EXCLUSIVE_GROUPS[1]
            comparison=pd.read_csv(v.DATA/"train.csv")
            positives=comparison[comparison.category.eq("comparison")&comparison.label.eq(1)]
            color_masks=np.zeros((len(names),2,len(colors)),dtype=bool)
            operators=np.zeros(len(names),dtype=np.int64)
            relevant=[]
            for row in positives.itertuples(index=False):
                p=parse_program(canonical_form(row.caption,row.category))
                if (p is None or p.op not in (FEWER,MORE,SAME) or len(p.a)!=1 or len(p.b)!=1
                        or p.a[0] not in colors or p.b[0] not in colors):
                    continue
                i=self.lookup[row.image]
                if inside[i]:
                    color_masks[i,0,colors.index(p.a[0])]=True
                    color_masks[i,1,colors.index(p.b[0])]=True
                    operators[i]=p.op
                    relevant.append(i)
            if not relevant:
                raise ValueError("No labeled color comparisons in the fitting split")
            self.color_scenes=torch.as_tensor(relevant,dtype=torch.long,device=device)
            self.color_filters=torch.as_tensor(color_masks,device=device)
            self.color_operators=torch.as_tensor(operators,device=device)
        began=time.perf_counter()
        self.real=RealData(self,counts,boxes,labels,keep,inside)
        rim.sync(device)
        self.startup["real_mining_and_upload_s"]=time.perf_counter()-began
        self.startup["fit_scenes"]=int(inside.sum())
        self.startup["inner_scenes"]=int((~inside).sum())

    def paired(self,n,rng,part,difficulty):
        sources,donors,modes,*_=self.pools[part]
        idx=draw_index(sources,n,rng)
        return rim.paired_views(self.bank,idx,self.grid,rng,modes,donors=donors,
                                difficulty=difficulty,details=True)

    def clear(self,n,rng,part):
        pool=self.pools[part][3]
        return intact(self.bank,pool[draw_index(torch.arange(len(pool),device=self.device),n,rng)],self.grid)


class RealData:
    def __init__(self,data,counts,boxes,labels,keep,inside):
        self.owner=data
        self.device=data.device
        bank=defaultdict(list)
        exact_total={data.lookup[name]:n for name,words,n in counts if not words}
        real={}
        paths=sorted(EXTRA.glob("old_train_*.npz"))
        if not paths:
            raise FileNotFoundError("Training proposal shards are missing")
        for path in paths:
            with np.load(path) as z:
                for j,candidate,front,score in zip(z["index"],z["boxes"],z["front"],z["source"]):
                    i=data.lookup[str(z["names"][j])]
                    bank[i].append((candidate,front,float(score)))
        unseen={}
        for i,entries in bank.items():
            certain=keep[i]&(labels[i]>=0).all(1)
            known=boxes[i,keep[i]]
            trusted=boxes[i,certain]
            if not len(trusted):
                continue
            trusted_ids=np.flatnonzero(certain)
            mids=known[:,:2]+known[:,2:]/2
            rejected=[]
            for candidate,front,_ in sorted(entries,key=lambda x:-x[2]):
                f_overlap=rim.pairwise_iou(front,trusted)
                k=trusted_ids[f_overlap.argmax()]
                if f_overlap.max()<.35:
                    continue
                iou=rim.pairwise_iou(candidate,known)
                pt=candidate[:2]+candidate[2:]/2
                dist=np.linalg.norm(mids-pt,axis=1)
                match=np.flatnonzero((iou>=.36)&(dist<7))
                rear_ids=np.flatnonzero(keep[i])[match]
                rear_ids=rear_ids[(rear_ids!=k)&certain[rear_ids]&
                                  (boxes[i,rear_ids,1]+boxes[i,rear_ids,3]<boxes[i,k,1]+boxes[i,k,3]-2)]
                if len(rear_ids)==1 and rim.pairwise_iou(boxes[i,rear_ids[0]],boxes[i,k])>=.035:
                    real.setdefault(i,[]).append((candidate,front,k,int(rear_ids[0]),True))
                if ((iou>=.33)&(dist<8)).any():
                    continue
                if exact_total.get(i)==int(keep[i].sum()):
                    real.setdefault(i,[]).append((candidate,front,k,k,False))
                # Merge adjacent proposals for the same physical candidate.
                if any(rim.pairwise_iou(candidate,other[0])>.42 and
                       np.linalg.norm(pt-(other[0][:2]+other[0][2:]/2))<6 for other in rejected):
                    continue
                rejected.append((candidate,front))
            if rejected:
                unseen[i]=rejected
        groups=defaultdict(list)
        coverage=defaultdict(int)
        for name,words,total in counts:
            i=data.lookup[name]
            part="fit" if inside[i] else "inner"
            m=attribute_mask(words)
            known=keep[i]
            inventory=labels[i,known]
            if any((q<0).any() and not a.all() for q,a in zip(inventory.T,m)):
                continue
            matches=np.logical_and.reduce([a[q.clip(0)] for q,a in zip(inventory.T,m)]).sum()
            missing=int(total-matches)
            kind="attribute" if words else "total"
            if missing==1:
                coverage[f"{part}_{kind}_one_missing"]+=1
            rows=unseen.get(i,[])
            if not rows or not 0<=missing<=len(rows):
                continue
            if missing==1:
                coverage[f"{part}_{kind}_one_missing_covered"]+=1
            if missing in (0,1,2):
                groups[part].append((i,rows,m,missing,kind))
        self.groups={}
        self.supervised={}
        for part,flag in (("fit",inside),("inner",~inside)):
            if not flag.any():
                continue
            candidates=groups[part]
            if not candidates:
                raise ValueError(f"No caption bags for {part}")
            coverage[f"{part}_bags"]=len(candidates)
            coverage[f"{part}_bag_scenes"]=len({r[0] for r in candidates})
            self.groups[part]=self.encode(candidates)
            samples=[(i,*r) for i,entries in real.items() if flag[i] for r in entries]
            coverage[f"{part}_natural_positive"]=sum(q[-1] for q in samples)
            coverage[f"{part}_natural_negative"]=sum(not q[-1] for q in samples)
            self.supervised[part]=samples
        self.coverage=dict(coverage)

    def encode(self,rows):
        n=max(len(r[1]) for r in rows)
        shape=(len(rows),n)
        valid=np.zeros(shape,bool)
        boxes=np.zeros((*shape,4),np.float32)
        fronts=np.zeros_like(boxes)
        for j,(_,candidates,*_) in enumerate(rows):
            k=len(candidates)
            valid[j,:k]=True
            boxes[j,:k],fronts[j,:k]=map(np.asarray,zip(*candidates))
        groups=[np.stack([row[2][g] for row in rows])[:,None] for g in range(3)]
        def tensor(a):
            return torch.as_tensor(a,device=self.device)
        scenes=tensor([r[0] for r in rows])
        target=tensor([r[3] for r in rows])
        subset=(target>0).nonzero().flatten(),(target==0).nonzero().flatten()
        return scenes,tensor(boxes),tensor(fronts),tensor(valid),[tensor(g) for g in groups],target,subset

    def bags(self,n,rng,part):
        scene,boxes,fronts,valid,groups,target,subset=self.groups[part]
        pools=[p for p in subset if len(p)]
        take=torch.cat([draw_index(p,n//len(pools)+(j<n%len(pools)),rng) for j,p in enumerate(pools)])
        mask=valid[take]
        i=scene[take,None].expand_as(mask)[mask]
        patch,prompt,_=rim.geometry_inputs(self.owner.bank.rgb[i],boxes[take][mask],fronts[take][mask],self.owner.grid)
        return patch,prompt,mask,[g[take] for g in groups],target[take],take

    def windows(self,n,rng,part):
        samples=self.supervised[part]
        if not samples:
            return None
        rng_cpu=np.random.default_rng(int(torch.randint(2**31,(1,),device=self.device,generator=rng)))
        p=[r for r in samples if r[-1]]
        q=[r for r in samples if not r[-1]]
        def sample(x,k):
            return [x[i] for i in rng_cpu.integers(len(x),size=k)] if len(x) else []
        k=min(max(1,n//3),len(p)*2) if q else n
        items=sample(p,k)+sample(q,n-k)
        i,b,f,k,j,t=zip(*items)
        def tensor(a):
            return torch.as_tensor(np.array(a),device=self.device)
        i,b,f,k,j,t=map(tensor,(i,b,f,k,j,t))
        bank=self.owner.bank
        rgb,marks,points=rim.geometry_inputs(bank.rgb[i],b,f,self.owner.grid)
        inst=rim.sample(bank.instances[i,None],points,"nearest")[:,0]
        return rgb,marks,(inst==j[:,None,None]+1)&t[:,None,None],inst==k[:,None,None]+1,torch.ones_like(t),bank.labels[i,j],t[:0],t

