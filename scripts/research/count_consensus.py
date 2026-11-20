"""Fit a symmetric reliability model for count answers using scene-disjoint validation folds."""
from pathlib import Path
import io
import zipfile
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

P=Path("artefacts/kaggle_suite")
FAMILIES=(("caption","head_a","head_b"),("native","occ_clean","occ_paste"),
          ("r50_img","r50_rand"),("sig","sigft","hog"),("draws","rank","fusion"))

def read(z,k,ids):
    with np.load(io.BytesIO(z.read(k))) as a:
        assert np.array_equal(a["ids"],ids)
        return a["score"]

def inputs(split):
    frame=pd.read_csv(f"data/{split}.csv")
    ids=frame.id.to_numpy()
    with zipfile.ZipFile(P/"clevr_fragments.zip") as z:
        with np.load(io.BytesIO(z.read(f"baseline_{split}.npz"))) as a:
            assert np.array_equal(ids,a["ids"])
            base=a["score"]
    cols=[]
    with zipfile.ZipFile(P/"run_2026-11-18/clevr_scores.zip") as z:
        for group in FAMILIES:
            ss=np.stack([read(z,f"{name}_{split}.npz",ids) for name in group])
            # A family's uncertainty includes disagreement between its own members.
            cols.append(logit(expit(ss).mean(0).clip(.0001,.9999)))
    x=np.stack(cols,axis=1)
    consensus=logit(expit(x).mean(1))
    x=np.column_stack((x,consensus,base*np.std(expit(x),axis=1)))
    return frame,base,x

def fit(x,b,y):
    signed=2*y-1
    def objective(w):
        m=signed*(b+x@w)
        return np.logaddexp(0,-m).sum()+2*(w@w), x.T@(-signed*expit(-m))+4*w
    res=minimize(objective,np.zeros(x.shape[1]),jac=True,method="L-BFGS-B")
    if not res.success:raise ValueError(res.message)
    return res.x

def main():
    frame,b,x=inputs("val")
    core=frame.category.isin(["counting","comparison","spatial_counting"]).to_numpy()
    scene=pd.factorize(frame.image)[0];y=frame.label.to_numpy()
    oof=b.copy()
    for tr,te in KFold(5,shuffle=True,random_state=397).split(np.unique(scene)):
        train=np.isin(scene,tr)&core;test=np.isin(scene,te)&core
        w=fit(x[train],b[train],y[train])
        oof[test]+=x[test]@w
    print("stack",roc_auc_score(y,oof),int(((y==1)&(oof<=0)).sum()))
    for name in ("58281.jpg","26268.jpg","78612.jpg"):
        i=frame.image.eq(name)&core&(y==1);print(name,b[i],oof[i])
    rng=np.random.default_rng(396)
    diffs=[]
    for j in range(800):
        w=rng.multinomial(1000,np.full(1000,.001))[scene]
        diffs.append(roc_auc_score(y,oof,sample_weight=w)-roc_auc_score(y,b,sample_weight=w))
    ci=np.quantile(diffs,[.025,.975]);print(ci)
    np.savez(P/"dense_count_pilot/stack_oof.npz",ids=frame.id,score=oof,ci=ci)
    if ci[0]>0 and ((y==1)&(oof<=0)).sum()<=((y==1)&(b<=0)).sum():
        w=fit(x[core],b[core],y[core]);f,bt,xt=inputs("test")
        use=f.category.isin(["counting","comparison","spatial_counting"])
        bt[use]+=xt[use]@w
        sample=pd.read_csv("data/sample_submission.csv")
        assert f.id.is_unique and sample.id.is_unique and set(f.id)==set(sample.id)
        sample["label"]=pd.Series(expit(bt),index=f.id).loc[sample.id].to_numpy()
        path=P/"dense_count_pilot/union_consensus.csv"
        sample.to_csv(path,index=False,float_format="%.17g")
        assert pd.read_csv(path,float_precision="round_trip").equals(sample)
        print(path)

if __name__=="__main__":main()
