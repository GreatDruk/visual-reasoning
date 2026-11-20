"""Two-layer ROI pilot for same-color objects on reviewed competition scenes."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

from src.pixel_regions import read_rgb

ROOT=Path("artefacts/research")
SIZE=64
SCALE=np.array([240,160,240,160])
COLORS=("gray","red","blue","green","brown","purple","cyan","yellow")


def teacher(split):
    path=ROOT/("teacher_train.npz" if split=="train" else "scene_silhouette.npz")
    with np.load(path) as a:
        names=a["names"] if split=="train" else pd.read_csv("data/val.csv").image.unique()
        features=a["features"]
        rgb_boxes=np.rint(features[...,8:12]*SCALE).astype(int)
        mask=features[...,47:147].reshape(*features.shape[:2],10,10).clip(0,1)
        p=a["p"]*a["valid"]
        colors=a["color"].argmax(-1)
    return names,rgb_boxes,mask,p,colors


def region(mask,box,image):
    x,y,w,h=box
    if x<0 or y<0 or w<1 or h<1 or x+w>240 or y+h>160:
        return
    a=cv2.resize(mask,(w,h),interpolation=cv2.INTER_LINEAR)
    return a,image[y:y+h,x:x+w]


def tensor(rgb,front,roi):
    x0,y0,x1,y1=roi
    crop=rgb[y0:y1,x0:x1]
    size=(SIZE,SIZE)
    rgb=cv2.resize(crop,size,interpolation=cv2.INTER_LINEAR) / 255
    # Give the network the candidate foreground object's location as a soft prompt.
    x,y,w,h=front
    xx=(np.arange(SIZE)+.5)*(x1-x0)/SIZE+x0
    yy=(np.arange(SIZE)+.5)*(y1-y0)/SIZE+y0
    dx=(xx-x-w/2)/max(w/2,1)
    dy=(yy-y-h/2)/max(h/2,1)
    prompt=np.exp(-(dx**2+dy[:,None]**2))
    return np.concatenate((rgb.transpose(2,0,1),prompt[None]),0).astype("f")


def roi_masks(front,back,roi):
    x,y,r,b=roi
    out=[]
    for mask in (front,back):
        out.append(cv2.resize(mask[y:b,x:r],(SIZE,SIZE),interpolation=cv2.INTER_AREA))
    return np.stack(out).astype("f")


def box_union(a,b,pad):
    lo=np.minimum(a[:2],b[:2])-pad
    hi=np.maximum(a[:2]+a[2:],b[:2]+b[2:])+pad
    return *lo.clip(0,[240,160]),*hi.clip(0,[240,160])


def synthetic(samples,seed):
    names,boxes,masks,prob,hue=teacher("train")
    rng=np.random.default_rng(seed)
    # Material and shape vary independently because the donors come from real train scenes.
    chosen=rng.choice(len(names),min(1000,len(names)),replace=False)
    cache={i:read_rgb(Path("data/train")/names[i]) for i in chosen}
    donors=[[] for _ in COLORS]
    objects=[]
    for i,rgb in cache.items():
        for j in np.flatnonzero(prob[i]>.92):
            box=boxes[i,j]
            if box[2:].min()<12:continue
            obj=region(masks[i,j],box,rgb)
            if obj is None:continue
            alpha,patch=obj
            if not .40<alpha.mean()<.95:continue
            donors[hue[i,j]].append((i,alpha,patch))
            objects.append((i,j,box,alpha,hue[i,j]))
    x=[];y=[];m=[]
    order=rng.integers(0,len(objects),samples*10)
    for index in order:
        i,j,front,alpha,color=objects[index]
        pool=donors[color]
        if not pool:continue
        k,back_alpha,back_rgb=pool[rng.integers(len(pool))]
        if k==i:continue
        w,h=np.array(back_alpha.shape[::-1])*rng.uniform(.68,1.10)
        w,h=max(8,int(w)),max(8,int(h))
        center=front[:2]+front[2:]*[.5,.5]+front[2:]*[rng.uniform(-.53,.53),rng.uniform(-.60,-.05)]
        origin=np.rint(center-[w/2,h/2]).astype(int)
        a,b=origin
        if a<0 or b<0 or a+w>240 or b+h>160:continue
        base=cache[i]
        outside=np.zeros((160,240),np.float32)
        foreground=outside.copy()
        fx,fy,fw,fh=front
        foreground[fy:fy+fh,fx:fx+fw]=alpha
        # Skip inserts that intersect a third confident object. The same RGB
        # context is used for the paired negative.
        for other in np.flatnonzero(prob[i]>.55):
            if other==j:continue
            obj=region(masks[i,other],boxes[i,other],base)
            if obj is None:continue
            ox,oy,ow,oh=boxes[i,other]
            view=outside[oy:oy+oh,ox:ox+ow]
            np.maximum(view,obj[0],out=view)
        amodal=cv2.resize(back_alpha,(w,h))
        obstructed=foreground[b:b+h,a:a+w]
        visible=amodal*(1-obstructed)
        if (visible*outside[b:b+h,a:a+w]).sum()>.06*visible.sum():continue
        visibility=visible.sum()/max(1.,amodal.sum())
        if not .10<visibility<.80 or visible.sum()<12:continue
        rgb=cv2.resize(back_rgb,(w,h))
        composed=base.copy()
        composed[b:b+h,a:a+w]=np.rint(rgb*visible[...,None]+base[b:b+h,a:a+w]*(1-visible[...,None])).astype("uint8")
        rear=np.zeros_like(foreground)
        rear[b:b+h,a:a+w]=visible
        roi=box_union(front,np.array([a,b,w,h]),rng.integers(3,12))
        # Equal compression on both members of every pair.
        for label,img in enumerate((base,composed)):
            _,encoded=cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,93])
            img=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
            x.append(tensor(img,front,roi))
            y.append(label)
            m.append(roi_masks(foreground,rear*label,roi))
        if len(y)>=2*samples:break
    return np.stack(x),np.asarray(y,dtype="f"),np.stack(m)


def reviewed():
    root=ROOT
    hard=pd.read_csv(root/"overlap_review/review_labels.csv").assign(hard=True)
    controls=pd.concat([
        pd.read_csv(root/"overlap_train_controls/review_labels.csv"),
        pd.read_csv(root/"overlap_controls/review_labels.csv").query("split == 'val'"),
    ]).assign(hard=False)
    frame=pd.concat([hard,controls]).dropna(subset=["number"])
    frame=frame[frame.number.isin((1,2))].drop_duplicates(["split","image"]).reset_index(drop=True)
    assert not set(frame.query("split=='train'").image)&set(frame.query("split=='val'").image)
    x=np.zeros((len(frame),4,SIZE,SIZE),np.float32)
    for split,group in frame.groupby("split"):
        names,box,_,p,hue=teacher(split)
        lookup={n:i for i,n in enumerate(names)}
        for row in group.itertuples():
            i=lookup[row.image]
            roi=np.array([row.x0,row.y0,row.x1,row.y1])
            center=box[i,:,:2]+.5*box[i,:,2:]
            inside=((center>=roi[:2]) & (center<roi[2:])).all(1)
            scores=p[i]*inside*(hue[i]==COLORS.index(row.color))
            if scores.max()>.1:
                front=box[i,scores.argmax()]
            else:
                front=np.r_[roi[:2],roi[2:]-roi[:2]]
            img=read_rgb(Path("data")/split/row.image)
            x[row.Index]=tensor(img,front,roi)
    return frame, x


class Layers(nn.Module):
    def __init__(self):
        super().__init__()
        def block(a,b,stride):
            return nn.Sequential(nn.Conv2d(a,b,3,stride,1),nn.GroupNorm(4,b),nn.SiLU(),
                                 nn.Conv2d(b,b,3,1,1),nn.GroupNorm(4,b),nn.SiLU())
        self.high=block(4,24,1)
        self.mid=block(24,32,2)
        self.low=block(32,48,2)
        self.mask=nn.Conv2d(56,2,1)
        self.pool=nn.AdaptiveAvgPool2d((4,4))
        self.classify=nn.Sequential(nn.Linear(48*4*4+2*4*4,96),nn.SiLU(),
                                    nn.Dropout(.2),nn.Linear(96,1))
    def forward(self,x):
        hi=self.high(x)
        mid=self.mid(hi)
        coarse=self.low(mid)
        masks=self.mask(torch.cat([hi,F.interpolate(mid,scale_factor=2,mode="bilinear",align_corners=False)],1))
        features=torch.cat([self.pool(coarse).flatten(1),self.pool(masks.sigmoid()).flatten(1)],1)
        return self.classify(features)[:,0],masks


def metrics(y,s):
    positive=s[y==1];negative=s[y==0]
    threshold=np.sort(negative)[-max(1,len(negative)//10)]
    return {"auc":float(roc_auc_score(y,s)), "detected_at_10pct_false_alarms":int((positive>threshold).sum()),
            "positives":len(positive),"negatives":len(negative)}


def fit(arm,synthetic,frame,real,epochs):
    torch.manual_seed(891)
    model=Layers()
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    sx,sy,sm=map(torch.from_numpy,synthetic)
    idx=np.flatnonzero(frame.split.eq("train"))
    x=torch.from_numpy(real[idx]);y=torch.as_tensor(frame.number.to_numpy()[idx]==2,dtype=torch.float)
    for epoch in range(epochs):
        for batch in torch.randperm(len(sy)).split(32):
            s=torch.randint(len(idx),(16,))
            xs=torch.cat([sx[batch],x[s]])
            flip=torch.rand(len(xs))<.5
            xs[flip]=xs[flip].flip(-1)
            logits,mask=model(xs)
            loss=F.binary_cross_entropy_with_logits(logits[:len(batch)],sy[batch])*.6
            loss+=F.binary_cross_entropy_with_logits(logits[len(batch):],y[s])
            if arm=="layers":
                truth=sm[batch].clone()
                truth[flip[:len(batch)]]=truth[flip[:len(batch)]].flip(-1)
                p=mask[:len(batch)]
                weight=1+3*truth
                loss+=.45*(F.binary_cross_entropy_with_logits(p,truth,reduction="none")*weight).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        print(arm,epoch+1,flush=True)
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(x)[0] for x in torch.from_numpy(real).split(32)]).numpy()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pairs",type=int,default=600)
    p.add_argument("--epochs",type=int,default=5)
    args=p.parse_args()
    torch.set_num_threads(5)
    cv2.setNumThreads(1)
    frame,real=reviewed()
    data=synthetic(args.pairs,893)
    print("pairs",len(data[1])//2,flush=True)
    result={}
    val=frame.split.eq("val")
    for arm in ("rgb","layers"):
        scores=fit(arm,data,frame,real,args.epochs)
        result[arm]={name:metrics(frame.number.to_numpy()[mask]==2,scores[mask])
                     for name,mask in (("all",val),("hard",val&frame.hard))}
        print(json.dumps(result[arm]),flush=True)
    (ROOT/"two_layer_roi.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":main()
