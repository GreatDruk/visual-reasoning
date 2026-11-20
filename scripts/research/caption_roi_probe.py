"""Add caption-supervised same-color crops to the reviewed HOG counter."""
import re
import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from scripts.research.two_layer_roi_probe import reviewed, metrics
from scripts.research.crop_encoder_probe import load_regions
from src.pixel_regions import COLORS, _color_planes, fit_background, read_rgb
from pathlib import Path


def caption_crops(split="train"):
    frame=pd.read_csv(f"data/{split}.csv")
    train=pd.read_csv("data/train.csv")
    bg=fit_background(Path("data/train"),train.image.unique())
    pattern=re.compile(r"exactly (one|two) ("+"|".join(COLORS)+r") objects?\b")
    cases=[]
    for row in frame[frame.label.eq(1)&frame.category.eq("counting")].itertuples():
        match=pattern.search(row.caption)
        if not match:continue
        color=match[2]
        image=read_rgb(Path("data")/split/row.image)
        planes,_=_color_planes(image,cv2.cvtColor(image,cv2.COLOR_RGB2HSV),bg)
        n,components,stats,_=cv2.connectedComponentsWithStats(planes[COLORS.index(color)].astype("uint8"))
        valid=np.flatnonzero(stats[:,4]>=18)
        valid=valid[valid!=0]
        if not len(valid):continue
        yy,xx=np.nonzero(np.isin(components,valid))
        low=np.maximum(0,[xx.min()-6,yy.min()-6])
        high=np.minimum([240,160],[xx.max()+7,yy.max()+7])
        if np.prod(high-low)>6000 or (high-low).max()>110:continue
        cases.append(dict(image=row.image,split=split,number=1 if match[1]=="one" else 2,
                          color=color,x0=low[0],y0=low[1],x1=high[0],y1=high[1],hard=False,
                          single_component=len(valid)==1))
    return pd.DataFrame(cases)


if __name__=="__main__":
    review,_=reviewed()
    source=caption_crops()
    used=set(review.image)
    source=source[~source.image.isin(used)]
    print("additional train crops",source.number.value_counts().to_dict(),flush=True)
    extra=caption_crops("val")
    extra=extra[~extra.image.isin(used)]
    frame=pd.concat([review,source,extra],ignore_index=True)
    features,color,_=load_regions(frame)
    x=np.concatenate((features,color),-1)
    val=frame.split.eq("val")
    reviewed_val=val&(frame.index<len(review))
    caption_val=val&(frame.index>=len(review))
    one_component=caption_val&frame.single_component.eq(True)
    i=np.flatnonzero(~val)
    human=i<len(review)
    for weight in (0,.25,.5,1.):
        train=i[human|(weight>0)]
        scale=np.where(train<len(review),1.,weight)
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,solver="liblinear",class_weight="balanced"))
        clf.fit(x[train],frame.number.iloc[train]==2,standardscaler__sample_weight=scale,
                logisticregression__sample_weight=scale)
        s=clf.decision_function(x)
        print(weight,[(n,metrics(frame.number.to_numpy()[k]==2,s[k]))
                     for n,k in (("reviewed",reviewed_val),("hard",reviewed_val&frame.hard),
                                 ("caption",caption_val),("one_component",one_component))],flush=True)
