"""Check fragment classifiers on material labels tied to positive official captions."""
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score

from scripts.research.fragment_attributes import (
    OUT, ROOT, DATA, SIDE, aperture_view, descriptors, read_rgb,
)
from src.models.slot_program import AND, EXACT, EXIST, EXCLUSIVE_GROUPS, parse_program
from src.text_aug import canonical_form


def material_crops():
    frame=pd.read_csv(DATA/"val.csv")
    names=frame.image.unique()
    index={name:i for i,name in enumerate(names)}
    with np.load(ROOT/"scene_silhouette.npz") as z:
        features,p=z["features"],z["p"]*z["valid"]
        probs=[z[k] for k in ("size","color","shape")]
        ids=z["ids"]
        assert np.array_equal(ids,frame.id)
    words={t:(j,k) for j,group in enumerate(
        (EXCLUSIVE_GROUPS[0],EXCLUSIVE_GROUPS[1],EXCLUSIVE_GROUPS[3]))
           for k,t in enumerate(group)}
    cases={}
    for row in frame[frame.label.eq(1)].itertuples():
        program=parse_program(canonical_form(row.caption,row.category))
        if program is None or program.relation or program.op not in (EXIST,AND,EXACT):
            continue
        if program.op==EXACT and program.number!=1:continue
        scene=index[row.image]
        for group in (program.a,program.b):
            material=next((t for t in group if t in ("metal","rubber")),None)
            if material is None:continue
            other=[words[t] for t in group if t in words]
            if len(other)<2:continue
            score=np.prod([probs[j][scene,:,k] for j,k in other],axis=0)
            match=(p[scene]>.10)&(score>.25)
            if match.sum()!=1:continue
            slot=int(np.flatnonzero(match)[0])
            if score[slot]<.82 or p[scene,slot]<.93:continue
            cases.setdefault((scene,slot),set()).add(int(material=="metal"))
    rows=[];target=[];rng=np.random.default_rng(133)
    for (scene,j),class_set in cases.items():
        if len(class_set)!=1:continue
        vec=features[scene,j]
        x,y,w,h=np.rint(vec[8:12]*[240,160,240,160]).astype(int)
        if min(w,h)<10 or not(0<=x<x+w<=240 and 0<=y<y+h<=160):continue
        rgb=cv2.resize(read_rgb(DATA/"val"/names[scene])[y:y+h,x:x+w],(SIDE,SIDE))
        support=cv2.resize(vec[47:147].reshape(10,10),(SIDE,SIDE))>.5
        if support.sum()<220:continue
        visibility=aperture_view(support,rng.uniform(.05,.15),rng)
        rows.append(np.stack([descriptors(rgb,support,np.ones_like(support)),
                              descriptors(rgb,support&visibility,visibility)]))
        target.append(next(iter(class_set)))
    return np.stack(rows),np.asarray(target)


def main():
    cv2.setNumThreads(1)
    x,y=material_crops()
    report=dict(objects=len(y),metal=int(y.sum()),rubber=int(len(y)-y.sum()),results={})
    for arm in ("clean","partial"):
        m=CatBoostClassifier()
        m.load_model(str(OUT/f"{arm}_material.cbm"))
        report["results"][arm]={k:float(balanced_accuracy_score(y,m.predict(x[:,i]).ravel()))
                                for i,k in enumerate(("full","5-15%"))}
    (OUT/"caption_material.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
