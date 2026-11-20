"""Marginalize count comparisons over a possible instance hidden behind visible foreground."""
from pathlib import Path
import io
import zipfile
import cv2
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from scripts.research import dense_counts as dc
from src.pixel_regions import _color_planes,read_rgb

P=Path("artefacts/kaggle_suite")
OUT=P/"dense_count_pilot"


def placement_prior(donors=None,background=None):
    rng=np.random.default_rng(607)
    if donors is None:
        with zipfile.ZipFile(P/"clevr_fragments.zip") as z:
            donors=io.BytesIO(z.read("fragment_train.npz"))
    with np.load(donors) as a:
        keep=a["keep"] & (a["labels"]>=0).all(-1)
        i,j=np.nonzero(keep)
        take=rng.choice(len(i),min(4096,len(i)),replace=False)
        i,j=i[take],j[take]
        sz,col,joint=a["labels"][i,j].astype(int).T
        box=a["box"][i,j];masks=a["mask"][i,j]
        frequency=float(a["keep"].sum(1).mean())
    ys=[];xs=[]
    for (x,y,w,h),mask in zip(box,masks):
        region=np.argwhere(mask>127)
        if not len(region):region=np.array([[5,5]])
        q=region[rng.integers(len(region),size=24)]+rng.random((24,2))
        ys.append((y+h*q[:,0]/10).astype(int).clip(0,159))
        xs.append((x+w*q[:,1]/10).astype(int).clip(0,239))
    if background is None:
        with np.load("artefacts/research/teacher_train.npz") as a:
            background=a["background"]
    return np.stack(ys),np.stack(xs),box[:,1]+box[:,3],(sz*8+col)*6+joint,frequency,background


def hidden_classes(image,prior):
    ys,xs,foot,kind,frequency,bg=prior
    planes,_=_color_planes(image,cv2.cvtColor(image,cv2.COLOR_RGB2HSV),bg)
    depth=np.zeros(image.shape[:2],np.int32)
    for plane in planes:
        _,labels,stats,_=cv2.connectedComponentsWithStats(plane.astype("uint8"))
        bottom=stats[:,1]+stats[:,3]
        bottom[stats[:,4]<45]=0
        bottom[0]=0
        np.maximum(depth,bottom[labels],out=depth)
    covered=(depth[ys,xs]>foot[:,None]+5).mean(1)
    support=covered>=.85
    hist=np.bincount(kind[support],minlength=96)
    return hist*(1-np.exp(-frequency*support.mean()))/max(1,support.sum())


def transitions(a,b,h):
    # Rows are the twin orientation before adding a hidden object; columns after.
    if set(a.a)!=set(b.a) or set(a.b)!=set(b.b):
        return np.eye(2)
    if a.op==dc.EXACT and b.op==dc.EXACT:
        starts=[[(a.number,1.)],[(b.number,1.)]]
        ends=lambda x:[x==a.number,x==b.number]
        increments=np.array([set(a.a)<=set(t) for t in dc.TYPES],int)
    elif a.op in (dc.MORE,dc.FEWER,dc.SAME) and b.op in (dc.MORE,dc.FEWER,dc.SAME):
        if a.op==b.op:return np.eye(2)
        def diffs(op):
            if op==dc.SAME:return [(0,1.)]
            sign=1 if op==dc.MORE else -1
            return [(sign,.5),(2*sign,.5)]
        starts=[diffs(a.op),diffs(b.op)]
        ends=lambda x:[(x>0 if op==dc.MORE else x<0 if op==dc.FEWER else x==0) for op in (a.op,b.op)]
        increments=np.array([int(set(a.a)<=set(t))-int(set(a.b)<=set(t)) for t in dc.TYPES])
    else:return np.eye(2)
    out=np.eye(2)*(1-h.sum())
    for i,options in enumerate(starts):
        for value,w in options:
            for step in np.unique(increments):
                out[i]+=w*h[increments==step].sum()*np.array(ends(value+step))
    return out


def adjust(frame,base,prior,images):
    caps=[dc.canonical_form(r.caption,r.category) for r in frame.itertuples()]
    pairs=dc.find_twins(frame.image,caps,frame.category)
    score=base.copy()
    cache={}
    for i,j in pairs:
        if frame.category.iat[i] not in dc.ACTIVE:continue
        name=frame.image.iat[i]
        if name not in cache:
            cache[name]=hidden_classes(read_rgb(images/name),prior)
        t=transitions(dc.parse_program(caps[i]),dc.parse_program(caps[j]),cache[name])
        q=expit(base[i])
        mass=np.array([q,1-q])@t
        score[i]=logit(np.clip(mass[0]/mass.sum(),1e-8,1-1e-8))
        score[j]=-score[i]
    return score


def predict(split,prior):
    frame=pd.read_csv(f"data/{split}.csv")
    with zipfile.ZipFile(P/"clevr_fragments.zip") as z:
        with np.load(io.BytesIO(z.read(f"baseline_{split}.npz"))) as a:
            assert np.array_equal(frame.id,a["ids"])
            base=a["score"]
    return frame,base,adjust(frame,base,prior,Path("data")/split)


if __name__=="__main__":
    prior=placement_prior()
    f,b,s=predict("val",prior)
    auc=lambda x:roc_auc_score(f.label,x)
    print("hidden prior",auc(b),auc(s),int(((s<=0)&(f.label==1)).sum()))
    for n in ["58281.jpg","26268.jpg","78612.jpg"]:
        i=f.image.eq(n)&f.category.isin(dc.ACTIVE)&f.label.eq(1)
        print(n,b[i],s[i])
    idx=pd.factorize(f.image)[0];rng=np.random.default_rng(731)
    delta=[]
    for _ in range(800):
        w=rng.multinomial(1000,np.full(1000,.001))[idx]
        delta.append(roc_auc_score(f.label,s,sample_weight=w)-roc_auc_score(f.label,b,sample_weight=w))
    ci=np.quantile(delta,[.025,.975]);print(ci)
    np.savez(OUT/"hidden_val.npz",ids=f.id,score=s,interval=ci)
    if ci[0]>0 and sum((s<=0)&f.label.eq(1))<=sum((b<=0)&f.label.eq(1)):
        f,_,s=predict("test",prior)
        sample=pd.read_csv("data/sample_submission.csv")
        assert f.id.is_unique and sample.id.is_unique and set(f.id)==set(sample.id)
        sample["label"]=pd.Series(expit(s),index=f.id).loc[sample.id].to_numpy()
        path=OUT/"union_hidden.csv";sample.to_csv(path,index=False,float_format="%.17g")
        assert pd.read_csv(path,float_precision="round_trip").equals(sample)
        print(path)
