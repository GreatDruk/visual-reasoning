"""Ablate weak R50 proposals, contours and localized crop features in the union."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from scipy.special import expit, logit
import torch
from torch import nn
from concurrent.futures import ThreadPoolExecutor
from torch.nn import functional as F

from scripts.research.fragment_votes import EXTRA, H, W, images, v
from scripts.research.visible_fragment_detector import pairwise_iou
from src.pixel_regions import _color_planes, _fragment_contours


OUT=Path("detail")
ARMS=("union","r50_only","local","contour","geometry","siglip")
GEOM=10
LOCAL=32*3
VIS=128
SCALE=np.array([W,H,W,H],np.float32)
FEATURES=GEOM+LOCAL+VIS


def mid(box):
    return box[:2]+.5*box[2:]


def contour_boxes(rgb,bg):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    planes,delta=_color_planes(rgb,hsv,bg)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0)
    gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
    grad=cv2.dilate(cv2.magnitude(gx,gy),np.ones((3,3),np.uint8))
    ranked=[]
    for color,contour in _fragment_contours(rgb,delta,planes,grad,[]):
        box=np.asarray(cv2.boundingRect(contour),np.float32)
        ranked.append((color,contour,box))
    return ranked[:12],gray


def as_candidate(box,color):
    from src.pixel_regions import COLORS
    from src.models.slot_program import EXCLUSIVE_GROUPS
    row=np.zeros(2+v.N_ATTRS+6,np.float32)
    row[:2]=logit(.10)
    row[2:2+v.N_ATTRS]=np.log(np.repeat(1/np.array([2,8,2,3]),[2,8,2,3]))
    ci=EXCLUSIVE_GROUPS[1].index(COLORS[color])
    row[4:12]=np.log(.30/7)
    row[4+ci]=np.log(.70)
    row[-6:-2]=box/SCALE
    row[-2:]=(box[:2]+box[2:]*[.50,.88])/[W,H]*2-1
    return row


def edge_models(gray,front,candidate):
    """Estimate the extra observed edge support of a second shape model."""
    x0,y0=np.floor(np.minimum(front[:2],candidate[:2])-4).astype(int).clip(0)
    x1,y1=np.ceil(np.maximum(front[:2]+front[2:],candidate[:2]+candidate[2:])+4).astype(int)
    x1,y1=min(x1,W),min(y1,H)
    if x1-x0<5 or y1-y0<5:
        return np.zeros(GEOM,dtype=np.float32)
    img=gray[y0:y1,x0:x1]
    outer=cv2.Canny(img,19,48)>0
    offset=np.array([x0,y0])

    def templates(b):
        x,y,w,h=b
        origin=np.rint(b[:2]-offset).astype(int)
        wh=np.maximum(2,np.rint(b[2:])).astype(int)
        center=tuple(np.rint(mid(b)-offset).astype(int))
        box=np.zeros_like(img); ell=np.zeros_like(img); cyl=np.zeros_like(img)
        cv2.rectangle(box,tuple(origin),tuple(origin+wh),1,1)
        cv2.ellipse(ell,center,tuple(np.maximum(1,wh//2)),0,0,360,1,1)
        cap=max(1,wh[1]//6)
        for (cx,cy),start,end in ((origin+[wh[0]//2,cap],0,360),
                                  (origin+[wh[0]//2,wh[1]-cap],0,180)):
            cv2.ellipse(cyl,(int(cx),int(cy)),(max(1,wh[0]//2),cap),0,start,end,1,1)
        for col in (0,wh[0]):
            cv2.line(cyl,tuple(origin+[col,cap]),tuple(origin+[col,wh[1]-cap]),1,1)
        return np.stack([box,ell,cyl]).astype(bool)

    toward=cv2.dilate(outer.astype(np.uint8),np.ones((3,3),np.uint8))>0
    present=templates(front)
    choice=np.argmax((present&toward).sum((1,2))/present.sum((1,2)).clip(1))
    f=present[choice]
    explained=cv2.dilate(f.astype(np.uint8),np.ones((5,5),np.uint8))>0
    residual=outer & ~explained
    remaining=cv2.dilate(residual.astype(np.uint8),np.ones((3,3),np.uint8))>0
    tests=templates(candidate)
    # Remove the foreground interior conservatively, keeping the shared rim ambiguous.
    block=np.zeros_like(img)
    origin=np.rint(front[:2]-offset).astype(int)
    wh=np.rint(front[2:]).astype(int)
    cv2.ellipse(block,tuple(origin+wh//2),tuple(np.maximum(1,wh//2-2)),0,0,360,1,-1)
    best=np.full(3,-np.inf)
    support=np.zeros(3)
    denom=max(1,residual.sum())
    # Match the two-object alternative over nearby translations with the same
    # foreground. Penalize an unexplained rear perimeter and additional freedom.
    for dx,dy in ((0,0),(-2,0),(2,0),(0,-2),(0,2)):
        moved=np.roll(tests,(dy,dx),axis=(1,2))
        if dx<0: moved[:,:,dx:]=False
        if dx>0: moved[:,:,:dx]=False
        if dy<0: moved[:,dy:,:]=False
        if dy>0: moved[:,:dy,:]=False
        moved &= ~(block>0)
        new=(moved&remaining).sum((1,2))
        tested=moved.sum((1,2))
        fit=(new-.18*(tested-.8*new)-2-.15*(abs(dx)+abs(dy)))/denom
        support=np.where(fit>best,new/tested.clip(1),support)
        best=np.maximum(best,fit)
    evidence=best
    return np.asarray([*support,*evidence,pairwise_iou(candidate,front),
                       (mid(front)[1]-mid(candidate)[1])/H, f.sum()/img.size,
                       residual.sum()/img.size],np.float32)


def close_front(tab,ok,box):
    strong=np.flatnonzero(ok & (tab[:,0]>=logit(.55)))
    if not len(strong):
        return None
    geometry=tab[strong,v.BOX]*SCALE
    shifts=np.linalg.norm(geometry[:,:2]+geometry[:,2:]/2-mid(box),axis=1)
    overlap=pairwise_iou(box,geometry)
    slope=geometry[:,1]+geometry[:,3]-(box[1]+box[3])
    keep=(shifts>=4)&(shifts<=.92*geometry[:,2:].max(1)+8)&(slope>=0)
    if not keep.any():
        return None
    return geometry[np.where(keep, overlap-.009*shifts,-1e3).argmax()]


def table_for(left,right,rgb,bg,mode):
    x,xy,ok=v.merge(left,right if mode!="r50_only" else right[:0])
    if mode in ("union","r50_only"):
        return x,xy,ok,np.zeros((len(x),GEOM),np.float32)
    candidates,gray=contour_boxes(rgb,bg) if mode=="contour" else ([],None)
    for color,_,box in candidates:
        known=x[ok & (x[:,0]>=logit(.50)),v.BOX]*SCALE
        if not (pairwise_iou(box,known)<.35).all():
            continue
        front=close_front(x,ok,box)
        if front is None:
            continue
        spare=np.flatnonzero(~ok)
        if not len(spare):
            break
        row=as_candidate(box,color)
        slot=spare[0]
        x[slot,:v.BASE]=row[1:-2]
        xy[slot]=row[-2:]
        # Third-source flag lives in the feature block below. Keep R50/thaw flags unset.
        x[slot,-1]=1
        ok[slot]=True
    geom=np.zeros((len(x),GEOM),np.float32)
    if gray is not None:
        for j in np.flatnonzero(ok & (x[:,0]<logit(.58))):
            box=x[j,v.BOX]*SCALE
            front=close_front(x,ok,box)
            if front is not None:
                geom[j]=edge_models(gray,front,box)
    return x,xy,ok,geom


def roi_grids(boxes, size, factor):
    axis=(torch.arange(size,device=boxes.device)+.5)/size
    y,x=torch.meshgrid(axis,axis,indexing="ij")
    points=boxes[:,None,None,:2]+boxes[:,None,None,2:]*(.5+factor*(torch.stack([x,y],-1)-.5))
    return points*2-1


def local_features(rgb,stem,index,boxes):
    raw=F.grid_sample(rgb[index],roi_grids(boxes,24,1.35),align_corners=False,padding_mode="border")
    early=F.grid_sample(stem[index].float(),roi_grids(boxes,7,1.15),align_corners=False,padding_mode="border")
    gx=(raw[:,:,:,1:]-raw[:,:,:,:-1]).abs()
    gy=(raw[:,:,1:]-raw[:,:,:-1]).abs()
    # Keep weak spatial detail and a compact summary of the R50 pre-pool channels.
    colour=F.adaptive_avg_pool2d(raw,(3,3)).flatten(1)
    edge=torch.cat([gx.mean((2,3)),gy.mean((2,3)),raw.std((2,3))],1)
    embedded=early.flatten(2)
    stem=embedded.mean(2)
    return torch.cat([colour,edge,stem[:,:60]],1)


def fragment_crop(rgb,index,boxes,front):
    crop=F.grid_sample(rgb[index],roi_grids(boxes,48,1.10),align_corners=False,padding_mode="border")
    # Use only a candidate-local aperture and attenuate the foreground interior.
    grid=roi_grids(boxes,48,1.10)
    point=(grid+1)/2
    center=front[:,None,None,:2]+.5*front[:,None,None,2:]
    radius=(front[:,None,None,2:]*.38).clamp_min(1e-5)
    inside=(((point-center)/radius).square()).sum(-1)<1
    keep=(~inside).float()[:,None]
    mean=(crop*keep).sum((2,3),keepdim=True)/keep.sum((2,3),keepdim=True).clamp_min(1)
    mean=torch.where(keep.sum((2,3),keepdim=True)>0,mean,crop.mean((2,3),keepdim=True))
    return crop*keep+mean*(1-keep)



@torch.inference_mode()
def extract(split,shard,device,limit,visual=False):
    from kaggle_suite_crops import crop_features
    from kaggle_suite_common import amp, tune_batch
    t=time.perf_counter()
    names=pd.read_csv(v.DATA/f"{split}.csv",usecols=["image"]).image.unique()[:limit][shard::2]
    rgb=images(split,names)
    decoded=time.perf_counter()-t
    bg=np.load(EXTRA/"train_background.npy")
    models=[v.ResNetCenters(full=True),v.CenterDetector()]
    heads=[]
    for model,key in zip(models,("r50_img","thaw")):
        model.load_state_dict(v.load(v.PRIORS/f"{key}.pt"))
        model.to(device,memory_format=torch.channels_last).eval()
        heads.append(v.model_head(key,device))
    crop_model=v.CropEncoder(device).eval() if visual else None
    transform=v.projection().to(device)
    modes=("siglip",) if visual else ARMS[:-1]
    saved={mode:[] for mode in modes}
    batch=tune_batch("R50/table", (256,192,128,64,32) if device.type=="cuda" else (4,),
             lambda n:v.detector_outputs(models[0],heads[0],torch.zeros(n,3,H,W,device=device)),device)
    crop_batch=None
    cpu_table=0.
    roi_count=0
    pool=ThreadPoolExecutor(max_workers=2)
    for k in range(0,len(names),batch):
        pixels=rgb[k:k+batch]
        b=torch.from_numpy(pixels).to(device).permute(0,3,1,2).float()/255
        with amp(device):
            stem=models[0].stem[:3]((b-models[0].mean)/models[0].std)
        a=v.detector_outputs(models[0],heads[0],b)
        d=v.detector_outputs(models[1],heads[1],b,transform)
        tables={}
        then=time.perf_counter()
        unique=("contour",) if visual else ("union","r50_only","contour")
        for tag in unique:
            jobs=pool.map(lambda triplet:table_for(*triplet,bg,tag),zip(a,d,pixels))
            tables[tag]=tuple(map(np.stack,zip(*jobs)))
        if not visual:
            tables["local"]=tables["union"]
        cpu_table+=time.perf_counter()-then
        for tag in modes:
            key="contour" if tag in ("geometry","siglip") else tag
            x,xy,ok,geom=tables[key]
            extra=np.zeros((*ok.shape,FEATURES),np.float32)
            if tag in ("geometry","siglip"):
                extra[:,:,:GEOM]=geom
            i,j=np.nonzero(ok & (x[...,0]<logit(.58)))
            if len(i) and tag in ("local","contour","geometry","siglip"):
                index=torch.from_numpy(i).to(device)
                boxes=torch.from_numpy(x[i,j,v.BOX]).to(device)
                for k2 in range(0,len(i),512):
                    ss=slice(k2,k2+512)
                    extra[i[ss],j[ss],GEOM:GEOM+LOCAL]=local_features(b,stem,index[ss],boxes[ss]).cpu().numpy()
            if visual and len(i):
                roi_count+=len(i)
                fronts=[]
                for row,col in zip(i,j):
                    front=close_front(x[row],ok[row],x[row,col,v.BOX]*SCALE)
                    fronts.append(np.zeros(4,dtype=np.float32) if front is None else front/SCALE)
                foreground=torch.from_numpy(np.stack(fronts)).to(device)

                def encode(sel):
                    crop=fragment_crop(b,index[sel],boxes[sel],foreground[sel])
                    crop=F.interpolate(crop,size=(256,256),mode="bilinear",align_corners=False)
                    _,tokens=crop_features(crop_model.backbone,2*crop-1)
                    with amp(device):
                        pooled=crop_model.pool(tokens.float())
                    return (((pooled.float()-crop_model.mean)@crop_model.components.T)/crop_model.scale).clamp(-7,7)
                if crop_batch is None:
                    crop_batch=tune_batch("fragment SigLIP", (128,96,64,48,24,12) if device.type=="cuda" else (3,),
                          lambda n:encode(torch.arange(n,device=device)%len(i)),device)
                for k2 in range(0,len(i),crop_batch):
                    ss=slice(k2,k2+crop_batch)
                    extra[i[ss],j[ss],-VIS:]=encode(ss).cpu().numpy()
            saved[tag].append((np.concatenate([x,extra],-1),xy,ok))
    pool.shutdown()
    for tag in modes:
        np.savez_compressed(OUT/f"table_{tag}_{split}_{shard}.npz",names=names.astype(str),
                             **dict(zip(("x","xy","valid"),map(np.concatenate,zip(*saved[tag])))))
    report=dict(split=split,visual=visual,scenes=len(names),decode_s=decoded,geometry_cpu_s=cpu_table,
                elapsed_s=time.perf_counter()-t,detector_batch=batch,crop_batch=crop_batch,visual_crops=roi_count)
    (OUT/f"tables_{split}_{shard}_{visual}.json").write_text(json.dumps(report,indent=2))
    print(report,flush=True)


def join(tag,split,out):
    (out/"tables").mkdir(parents=True,exist_ok=True)
    with np.load(OUT/f"table_{tag}_{split}_0.npz") as a, np.load(OUT/f"table_{tag}_{split}_1.npz") as b:
        data={}
        for key in a.files:
            left,right=a[key],b[key]
            z=np.empty((len(left)+len(right),*left.shape[1:]),dtype=left.dtype)
            z[::2]=left; z[1::2]=right
            data[key]=z
    names=pd.read_csv(v.DATA/f"{split}.csv",usecols=["image"]).image.unique()
    if not np.array_equal(names,data["names"]):
        raise ValueError(f"{split}: mismatched scenes")
    np.savez(out/"tables"/f"{split}.npz",**data)


def branch(tag,seed,device,epochs,split="val",train=True):
    # Refit the caption-trained verifier for each proposal source.
    base=v.Verifier

    class Extended(base):
        def __init__(self,visual):
            super().__init__(False)
            self.visual=True
            self.encoder[0]=nn.Linear(v.INPUT_DIM+FEATURES,96)

    v.Verifier=Extended
    v.CACHE=OUT/f"{tag}_{seed}"
    v.OUT=v.CACHE
    join(tag,split,v.CACHE)
    if train:
        join(tag,"train",v.CACHE)
        v.train("control",seed,epochs,device)
    v.infer(split,"control",seed,device)
    with np.load(v.OUT/f"control_{seed}_{split}.npz") as z:
        np.savez(OUT/f"{tag}_{seed}_{split}.npz",**dict(z))
    if split!="val":
        return
    with np.load(v.CACHE/"tables/val.npz") as z:
        names=z["names"]
        tab=(z["x"],z["xy"],z["valid"])
    model=Extended(False).to(device)
    model.load_state_dict(v.load(v.OUT/f"control_{seed}.pt"))
    p,_,valid,*attr=v.table_predictions(model,tab,device)
    box=tab[0][...,v.BOX]*SCALE
    additions=tab[2] & (tab[0][...,-FEATURES-3:-FEATURES-1].sum(-1)==0)
    np.savez_compressed(OUT/f"{tag}_{seed}_objects.npz",names=names.astype(str),p=p,boxes=box,valid=valid,
                        classes=np.concatenate(attr,axis=-1),contour=additions)



if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("stage",choices=("extract","extract_siglip","fit","infer"))
    cli.add_argument("--arm",choices=ARMS,default="union")
    cli.add_argument("--seed",type=int,default=421)
    cli.add_argument("--split",choices=("train","val","test"),default="train")
    cli.add_argument("--shard",type=int,choices=(0,1),default=0)
    cli.add_argument("--epochs",type=int,default=16)
    cli.add_argument("--limit",type=int)
    args=cli.parse_args()
    OUT.mkdir(exist_ok=True)
    _,device=v.setup(args.seed)
    if args.stage in ("fit","infer"):
        branch(args.arm,args.seed,device,args.epochs,"val" if args.stage=="fit" else args.split,args.stage=="fit")
    else:
        extract(args.split,args.shard,device,args.limit,args.stage=="extract_siglip")

