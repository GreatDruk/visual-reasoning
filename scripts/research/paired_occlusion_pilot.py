"""Paired train-only occlusion pilot, with donors separated by source scene."""
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SIDE = 32

def donors():
    with np.load(ROOT/"artefacts/research/teacher_train.npz") as z:
        feat = z["features"]
        material, shape = z["material"], z["shape"]
        keep = z["valid"] & (z["p"]>.94) & (material.max(-1)>.95) & (shape.max(-1)>.95)
        labels = 3*material.argmax(-1)+shape.argmax(-1)
        names = z["names"]
    out = [[],[]]
    for i in np.random.default_rng(831).permutation(len(names))[:1400]:
        rgb = Image.open(ROOT/"data/train"/names[i]).convert("RGB")
        for j in np.flatnonzero(keep[i]):
            x,y,w,h = np.rint(feat[i,j,8:12]*[240,160,240,160]).astype(int)
            if min(w,h)<12 or x<0 or y<0 or x+w>240 or y+h>160:
                continue
            mask = Image.fromarray(feat[i,j,47:147].reshape(10,10)).resize((SIDE,SIDE),Image.Resampling.BILINEAR)
            alpha = np.asarray(mask).clip(0,1)[...,None]
            if not .40<alpha.mean()<.95:
                continue
            patch = np.asarray(rgb.crop((x,y,x+w,y+h)).resize((SIDE,SIDE))).astype("f")
            out[int(i%5==0)].append((patch,alpha,int(labels[i,j])))
    return out

def make(pool,n,seed,occlude):
    rng=np.random.default_rng(seed)
    features,labels=[],[]
    for _ in range(n):
        target,mask,label=pool[rng.integers(len(pool))]
        front,alpha,_=pool[rng.integers(len(pool))]
        canvas=np.full_like(target,rng.uniform(90,155))
        dx=int(rng.choice([-1,1])*rng.integers(6,17))
        dy=int(rng.integers(3,13))
        front=np.roll(front,(dy,dx),(0,1))
        alpha=np.roll(alpha,(dy,dx),(0,1))
        alpha[:dy]=0
        if dx>0: alpha[:,:dx]=0
        else: alpha[:,dx:]=0
        for present in [True,False]:
            image=target*mask+canvas*(1-mask) if present else canvas
            if occlude or not present: image=front*alpha+image*(1-alpha)
            features.append(image.reshape(8,4,8,4,3).mean((1,3)).flatten())
            labels.append(label if present else 6)
    return np.stack(features), np.array(labels)

if __name__=="__main__":
    train,test=donors()
    print("train and held-out donors",len(train),len(test),flush=True)
    x,y=make(test,2000,833,True)
    for occlude in [False,True]:
        a,b=make(train,5000,832,occlude)
        clf=ExtraTreesClassifier(n_estimators=96,min_samples_leaf=2,n_jobs=4,random_state=834).fit(a,b)
        pred=clf.predict_proba(x)
        print("occluded training",occlude,"presence AUC",roc_auc_score(y!=6,1-pred[:,6]),
              "joint shape/material balanced accuracy, present targets",
              balanced_accuracy_score(y[y!=6],pred[y!=6,:6].argmax(1)),flush=True)
        scene=Image.open(ROOT/"data/val/58281.jpg").convert("RGB")
        for name,box in [("visible",(97,94,147,144)),("weak",(137,79,152,99))]:
            image=np.asarray(scene.crop(box).resize((SIDE,SIDE))).astype("f")
            prob=clf.predict_proba(image.reshape(8,4,8,4,3).mean((1,3)).reshape(1,-1))[0]
            print(name,"metal cylinder",round(prob[5],3),"absence",round(prob[6],3),flush=True)
