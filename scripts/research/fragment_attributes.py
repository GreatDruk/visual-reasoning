"""Assess which shape and material cues survive in small visible fragments."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score

from src.pixel_regions import COLORS as COLOR_PLANES, _color_planes, read_rgb

ROOT = Path("artefacts/research")
DATA = Path("data")
OUT = Path("artefacts/kaggle_suite/fragment_attributes")
SIDE = 32
RANGES = ((1., 1.), (.05, .15), (.15, .30), (.30, .55))
RANGE_NAMES = ("full", "5-15%", "15-30%", "30-55%")
GRID = np.stack(np.mgrid[:SIDE, :SIDE][::-1], -1).astype(np.float32)
KERNEL = np.ones((3, 3), np.uint8)
QUANTILES = (0., .10, .30, .50, .70, .90, .97, 1.)


def descriptors(image, support, aperture):
    """Ignore gradients along an artificial cut through a visible surface."""
    yy, xx = np.nonzero(support)
    if not len(xx):
        raise ValueError("Descriptor requires visible pixels")
    lo = np.maximum([xx.min()-2, yy.min()-2], 0)
    hi = np.minimum([xx.max()+3, yy.max()+3], image.shape[:2][::-1])
    x0, y0 = lo
    x1, y1 = hi
    crop = np.s_[y0:y1, x0:x1]
    resize = lambda x: cv2.resize(x[crop], (24, 24), interpolation=cv2.INTER_NEAREST)
    v = resize(support.astype(np.uint8)).astype(bool)
    known = resize(aperture.astype(np.uint8))
    shape = cv2.resize(v.astype(np.float32), (6,6), interpolation=cv2.INTER_AREA)
    # A predicted occlusion boundary gives no evidence for the intrinsic contour.
    outline = v & ~(cv2.erode(v.astype(np.uint8), KERNEL).astype(bool))
    outline &= cv2.erode(known, KERNEL).astype(bool)
    silhouette = cv2.resize(outline.astype(np.float32), (6,6), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = y1-y0, x1-x0
    interior = cv2.erode(support.astype(np.uint8), KERNEL,
                        borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    inside = interior if interior.sum()>=4 else support
    grad_x = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    magn,angles = cv2.cartToPolar(grad_x,grad_y)
    bins = (angles*(8/np.pi)).astype(np.int32)%8
    hist = np.bincount(bins[interior],weights=magn[interior],minlength=8)
    hist = hist/(hist.sum()+1.)
    lab = cv2.cvtColor(image,cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    q = np.quantile(lab[inside],QUANTILES,axis=0)
    saturation = np.quantile(hsv[...,1][inside],QUANTILES)
    sharpness = np.quantile(magn[interior],QUANTILES) if interior.any() else np.zeros(len(QUANTILES))
    laplacian = cv2.Laplacian(gray,cv2.CV_32F)
    details = np.quantile(np.abs(laplacian[interior]),QUANTILES) if interior.any() else np.zeros(len(QUANTILES))
    # Contrast and chroma statistics remain independent of the unknown pixels.
    return np.r_[h/w,support.sum()/(h*w),outline.mean(),shape.ravel(),silhouette.ravel(),
                 hist,q.ravel(),saturation,sharpness,details].astype(np.float32)


def aperture_view(mask,fraction,rng):
    if fraction==1.:
        return np.ones_like(mask)
    angle=rng.uniform(0,2*np.pi)
    normal=np.array([np.cos(angle),np.sin(angle)],dtype="f")
    along=GRID@normal
    cross=(GRID-GRID.mean((0,1)))@np.array([-normal[1],normal[0]],dtype="f")
    surface=along+(cross/32)**2*rng.uniform(-12,12)
    values=surface[mask]
    # Always keep at least four pixels for local colour statistics.
    rank=max(3, min(len(values)-1, round(fraction*len(values))-1))
    threshold=np.partition(values,rank)[rank]
    return surface<=threshold


def objects(split):
    path=ROOT/("teacher_train.npz" if split=="train" else "scene_silhouette.npz")
    with np.load(path) as z:
        features=z["features"]
        p=z["p"]*z["valid"]
        material,shape=z["material"],z["shape"]
        names=z["names"] if split=="train" else pd.read_csv(DATA/"val.csv").image.unique()
    qualified=(p>.93)&(material.max(-1)>.94)&(shape.max(-1)>.93)
    rng=np.random.default_rng(193 if split=="train" else 255)
    rows=[];target=[];bands=[]
    for i,name in enumerate(names):
        choices=np.flatnonzero(qualified[i])
        if not len(choices):continue
        im=read_rgb(DATA/split/name)
        for j in choices:
            x,y,w,h=np.rint(features[i,j,8:12]*[240,160,240,160]).astype(int)
            if min(w,h)<10 or x<0 or y<0 or x+w>240 or y+h>160:continue
            rgb=cv2.resize(im[y:y+h,x:x+w],(SIDE,SIDE))
            mask=cv2.resize(features[i,j,47:147].reshape(10,10),(SIDE,SIDE))>.5
            if mask.sum()<220:continue
            classes=(shape[i,j].argmax(),material[i,j].argmax())
            for band,(lo,hi) in enumerate(RANGES):
                visible=aperture_view(mask,rng.uniform(lo,hi),rng)
                rows.append(descriptors(rgb,mask&visible,visible))
                target.append(classes)
                bands.append(band)
    return np.stack(rows),np.asarray(target),np.asarray(bands)


def real_boxes(bg):
    frame=pd.read_csv(ROOT/"overlap_review/visible_boxes.csv")
    x=[];y=[];names=[]
    for row in frame.itertuples(index=False):
        im=read_rgb(DATA/"val"/row.image)
        mask=_color_planes(im,cv2.cvtColor(im,cv2.COLOR_RGB2HSV),bg)[0][COLOR_PLANES.index(row.color)]
        box=np.s_[row.y0:row.y1,row.x0:row.x1]
        rgb=cv2.resize(im[box],(SIDE,SIDE))
        support=cv2.resize(mask[box].astype(np.uint8),(SIDE,SIDE),interpolation=cv2.INTER_NEAREST)>0
        if support.sum()<4:continue
        x.append(descriptors(rgb,support,np.ones_like(support)))
        y.append("bsc".index(row.shape))
        names.append(row.image)
    return np.stack(x),np.asarray(y),np.asarray(names)


def main():
    cv2.setNumThreads(1)
    OUT.mkdir(exist_ok=True)
    x,y,portion=objects("train")
    t,z,bands=objects("val")
    print("train and val view counts",len(y),len(z),flush=True)
    with np.load(ROOT/"teacher_train.npz") as a:
        u,v,names=real_boxes(a["background"])
    results={}
    for arm in ("clean","partial"):
        chosen=(portion==0) if arm=="clean" else np.ones(len(y),bool)
        learners=[]
        for k,name in enumerate(("shape","material")):
            clf=CatBoostClassifier(iterations=225,depth=7,learning_rate=.10,thread_count=4,
                                   random_seed=319,verbose=False,allow_writing_files=False)
            clf.fit(x[chosen],y[chosen,k])
            clf.save_model(str(OUT/f"{arm}_{name}.cbm"))
            learners.append(clf)
        prediction=np.column_stack([c.predict(t).ravel() for c in learners])
        model={}
        for i,band in enumerate(RANGE_NAMES):
            seen=bands==i
            model[band]={k:float(balanced_accuracy_score(z[seen,j],prediction[seen,j]))
                         for j,k in enumerate(("shape","material"))}
            model[band]["joint_accuracy"]=float((z[seen]==prediction[seen]).all(1).mean())
        actual=learners[0].predict(u).ravel()
        model["reviewed_shape"]={"hits":int((actual==v).sum()),"total":len(v),
                                "balanced_accuracy":float(balanced_accuracy_score(v,actual))}
        np.savez(OUT/f"{arm}_review.npz",names=names,truth=v,pred=actual)
        results[arm]=model
        print(arm,model,flush=True)
    (OUT/"report.json").write_text(json.dumps(results,indent=2),encoding="utf8")


if __name__=="__main__":
    main()
