"""Check whether supervised recalibration of a frozen scene histogram is sufficient."""
import io
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F

from scripts.research import dense_counts as dc

P=Path("artefacts/kaggle_suite/dense_count_pilot")

class Calibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.adjust=nn.Sequential(nn.Linear(96,64),nn.SiLU(),nn.Linear(64,96))
        nn.init.zeros_(self.adjust[-1].weight)
        nn.init.zeros_(self.adjust[-1].bias)

    def forward(self,x):
        return x*(1.5*self.adjust(x).tanh()).exp()


def dataset(split):
    frame=pd.read_csv(f"data/{split}.csv")
    names,pairs,scenes,table=dc.queries(frame,"cpu")
    with np.load(P/f"counts_{split}.npz",allow_pickle=True) as a:
        assert np.array_equal(names,a["names"])
        h=torch.from_numpy(a["hist"])
    return frame,pairs,scenes,table,h


if __name__=="__main__":
    torch.set_num_threads(3)
    torch.manual_seed(735)
    train,pairs,scenes,table,h=dataset("train")
    y=torch.from_numpy(train.label.to_numpy()[pairs[:,0]]).float()
    model=Calibrator()
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    for epoch in range(15):
        for idx in torch.randperm(len(y)).split(256):
            x=h[scenes[idx]]
            out=model(x)
            picks=(2*idx[:,None]+torch.arange(2)).flatten()
            score=dc.caption_scores(out,picks,table)
            true=idx*2+(1-y[idx]).long()
            exact=table[2][true]==dc.EXACT
            n=(out*table[0][true]).sum(-1)
            loss=F.binary_cross_entropy_with_logits(score,y[idx])+.03*(out-x).abs().sum(-1).mean()
            e=F.smooth_l1_loss(n,table[3][true].float(),reduction="none",beta=.25)
            loss+=.6*(e*exact).sum()/exact.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    val,pairs,scenes,table,h=dataset("val")
    with zipfile.ZipFile(P.parent/"phase_run/phase_results.zip") as z:
        with np.load(io.BytesIO(z.read("validation.npz"))) as a:
            assert np.array_equal(val.id,a["ids"])
            base=a["base"]
    with torch.no_grad():
        result={}
        for key,hist in (("raw",h),("calibrated",model(h))):
            score=dc.caption_scores(hist[scenes],torch.arange(2*len(scenes)),table).clamp(-8,8).numpy()
            scores=np.zeros(len(val))
            scores[pairs[:,0]],scores[pairs[:,1]]=score,-score
            blend=np.where(val.category.isin(dc.ACTIVE),(scores+base)/2,base)
            print(key,roc_auc_score(val.label,blend),int(((val.label==1)&(blend<=0)).sum()),
                  "hard case",blend[val.image.eq("58281.jpg")&val.category.eq("counting")&val.label.eq(1)])
            result[key]=blend
        np.savez(P/"frozen_val.npz",ids=val.id,**result)
