"""Test whether shadows add count evidence after accounting for visible silhouettes."""
import json
import re

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.research.fragment_attributes import OUT, ROOT, DATA, read_rgb

WORDS="zero one two three four five six seven eight nine ten".split()
EXACT=re.compile(r"(?:there (?:are|is) |the scene contains )?exactly ("+
                 "|".join(WORDS)+r") objects?(?:(?: are| is)? in the scene)?[.]?$")


def scenes(split,bg):
    frame=pd.read_csv(DATA/f"{split}.csv")
    targets={}
    for row in frame[frame.label.eq(1)].itertuples():
        match=EXACT.fullmatch(row.caption.strip().lower())
        if match:
            n=WORDS.index(match.group(1))
            if row.image in targets and n!=targets[row.image]:
                raise ValueError(f"Inconsistent visible-count target {row.image}")
            targets[row.image]=n
    path=ROOT/("teacher_train.npz" if split=="train" else "scene_silhouette.npz")
    with np.load(path) as z:
        names=z["names"] if split=="train" else frame.image.unique()
        x,p=z["features"],z["p"]*z["valid"]
    index={n:i for i,n in enumerate(names)}
    geometry=[];shadow=[];truth=[];baseline=[]
    for name,val in targets.items():
        i=index[name]
        rgb=read_rgb(DATA/split/name)
        rows=np.rint(x[i,:,8:12]*[240,160,240,160]).astype(int)
        matte=np.zeros((160,240),np.float32)
        for j in np.flatnonzero(p[i]>.35):
            a,b,w,h=rows[j]
            if not(0<=a<a+w<=240 and 0<=b<b+h<=160):continue
            patch=matte[b:b+h,a:a+w]
            body=cv2.resize(x[i,j,47:147].reshape(10,10),(w,h))
            np.maximum(patch,body,out=patch)
        luma=rgb.astype(np.float32).mean(-1)
        residual=bg.mean(-1)-luma
        edges=np.concatenate((residual[:,:6],residual[:,-6:]),axis=1)
        residual-=np.median(edges,axis=1)[:,None]
        excluded=cv2.dilate((matte>.35).astype(np.uint8),np.ones((5,5),np.uint8))>0
        chroma=np.ptp(rgb,axis=-1)
        floor=(~excluded)&(chroma<14)&(np.arange(160)[:,None]>38)
        shade=np.where(floor,residual,0)
        moments=[]
        for lo,hi in ((39,77),(77,116),(116,160)):
            v=shade[lo:hi]
            for threshold in (7,13,21):
                moments.append(float((v>threshold).mean()))
            moments.append(float(v.clip(0,38).mean()))
        # Characterize connected dark regions, excluding very tiny JPEG fluctuations.
        _,_,stats,_=cv2.connectedComponentsWithStats((shade>11).astype(np.uint8))
        areas=stats[1:,4]
        areas=areas[areas>=18]
        moments.extend([len(areas),float(areas.sum())/4000,float(max(areas,default=0))/4000])
        hard=p[i]>.65
        widths=rows[:,2].clip(0)
        heights=rows[:,3].clip(0)
        geometry.append([p[i].sum(),hard.sum(),(p[i]>.85).sum(),
                         (widths*heights*p[i]).sum()/1000,
                         (widths*p[i]).sum()/100])
        shadow.append(moments)
        truth.append(val)
        baseline.append(hard.sum())
    return np.asarray(geometry),np.asarray(shadow),np.asarray(truth),np.asarray(baseline)


def main():
    cv2.setNumThreads(1)
    with np.load(ROOT/"teacher_train.npz") as z:bg=z["background"]
    a,b,y,p=scenes("train",bg)
    c,d,t,q=scenes("val",bg)
    output={}
    for name,x,v in (("geometry",a,c),("geometry_and_shadows",np.c_[a,b],np.c_[c,d]),("shadows",b,d)):
        estimator=make_pipeline(StandardScaler(),RidgeCV(alphas=[.1,1,10,50,150]))
        estimator.fit(x,y-p if name!="shadows" else y)
        raw=estimator.predict(v)+(q if name!="shadows" else 0)
        output[name]={"mae":float(np.abs(raw-t).mean()),"exact":int((np.rint(raw)==t).sum()),
                      "bias":float((raw-t).mean())}
    result={"train_scenes":len(y),"val_scenes":len(t),"baseline":{
        "mae":float(np.abs(q-t).mean()),"exact":int((q==t).sum())},"models":output}
    (OUT/"shadow_count.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
