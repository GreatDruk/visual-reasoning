"""Scene-grouped selection of count ensemble and missing-object uncertainty."""
from pathlib import Path
import io,zipfile
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import KFold
P=Path("artefacts/kaggle_suite")
OUT=P/"count_run"

def inputs(split,need_old=True):
    f=pd.read_csv(f"data/{split}.csv")
    def read(z,k,col="score"):
        with np.load(io.BytesIO(z.read(k))) as a:
            assert np.array_equal(a["ids"],f.id)
            return a[col]
    with zipfile.ZipFile(P/"clevr_counters.zip") as z:b=read(z,f"baseline_{split}.npz")
    with zipfile.ZipFile(OUT/"count_results.zip") as z:
        r=np.mean([read(z,f"r50_{s}/verify_result/control_{s}_{split}.npz") for s in (421,853)],axis=0)
    with zipfile.ZipFile(P/"run_2026-11-18/clevr_scores.zip") as z:
        old=np.mean([read(z,f"{name}_{split}.npz") for name in ("caption","geometry","sigft")],axis=0) if need_old else np.zeros_like(b)
    return f,b,r,old

def combinations(f,b,r,old,hidden):
    active=f.category.isin(["counting","comparison"])
    difficult=active|f.category.eq("spatial_counting")
    for w in (.25,.5):
        for vote in ("logit","prob"):
            x=(1-w)*b+w*r if vote=="logit" else logit(((1-w)*expit(b)+w*expit(r)).clip(1e-10,1-1e-10))
            x=np.where(active,x,b)
            for o in (0,.125,.25):
                blend=np.where(difficult,(1-o)*x+o*old,x)
                for h in (0,.25,.5):
                    yield (w,vote,o,h),blend+h*(hidden-b)

def main():
    f,b,r,old=inputs("val"); y=f.label.to_numpy(); ids=pd.factorize(f.image)[0]
    with np.load(P/"dense_count_pilot/hidden_val.npz") as a:
        assert np.array_equal(a["ids"],f.id)
        hidden=a["score"]
    candidates=dict(combinations(f,b,r,old,hidden))
    ref=candidates[(.5,"logit",0,0)]
    keys=list(candidates)
    def best(mask):
        return max(keys,key=lambda k:auc(y[mask],candidates[k][mask]))
    full=best(np.ones(len(f),bool));print("full",full,auc(y,candidates[full]))
    for seed in [619,2401,4211]:
        oof=ref.copy();picks=[]
        for tr,te in KFold(5,shuffle=True,random_state=seed).split(np.unique(ids)):
            key=best(np.isin(ids,tr));use=np.isin(ids,te);oof[use]=candidates[key][use];picks.append(key)
        print("OOF",seed,auc(y,oof),int(((y==1)&(oof<=0)).sum()),picks)
        if seed==619:primary=oof
    rng=np.random.default_rng(231);delta=[]
    for _ in range(1000):
        sw=rng.multinomial(1000,np.full(1000,.001))[ids]
        delta.append(auc(y,primary,sample_weight=sw)-auc(y,ref,sample_weight=sw))
    ci=np.quantile(delta,[.025,.975]);print("CI",ci)
    out=candidates[full]
    print("full errors",int(((y==1)&(out<=0)).sum()))
    for n in ["58281.jpg","26268.jpg","78612.jpg","54087.jpg"]:
        i=f.image.eq(n)&f.label.eq(1)&(ref<0)
        print(n,ref[i],out[i])
    test,bt,rt,ot=inputs("test",need_old=bool(full[2]));active=test.category.isin(["counting","comparison"])
    logits=np.where(active,.5*(bt+rt),bt)
    sent=pd.read_csv(OUT/"r50_count.csv",float_precision="round_trip")
    sample=pd.read_csv("data/sample_submission.csv")
    assert sent.id.equals(sample.id) and test.id.is_unique and sent.id.is_unique and set(sent.id)==set(test.id)
    difference=np.max(np.abs(expit(pd.Series(logits,index=test.id).loc[sent.id]).to_numpy()-sent.label.to_numpy()))
    print("maximum CSV deviation",difference);assert difference<1e-12
    print("verified original",len(sent))
    if auc(y,out)>auc(y,ref):  # Export the exploratory candidate separately from the verified run.
        if full[-1]:
            from scripts.research.hidden_count_prior import predict,placement_prior
            frame,_,ht=predict("test",placement_prior());assert frame.id.equals(test.id)
        else:ht=bt
        values=dict(combinations(test,bt,rt,ot,ht))[full]
        sample.label=pd.Series(expit(values),index=test.id).loc[sample.id].to_numpy()
        path=OUT/"count_hidden.csv";sample.to_csv(path,index=False,float_format="%.17g")
        assert pd.read_csv(path,float_precision="round_trip").equals(sample) and sample.label.between(0,1).all()
        print(path)
if __name__=="__main__":main()


