"""Evaluate image scale as a second view of the trained R50 on identical val scenes."""
import io
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

from scripts.research.fragment_votes import v,images
from kaggle_suite_common import dense_bank,table_predictions
from native_centers import read_rows,GROUPS
from src.scene_posterior import sample_scene_scores
from src.twins import find_twins

DEST=Path("artefacts/kaggle_suite/dense_count_pilot")


@torch.inference_mode()
def evaluate():
    torch.set_num_threads(5)
    frame,groups,caps,programs=read_rows("val")
    names=list(groups)
    model=v.ResNetCenters(full=True).eval().to(memory_format=torch.channels_last)
    model.load_state_dict(v.load(v.PRIORS/"r50_img.pt"))
    head=v.model_head("r50_img",torch.device("cpu"))
    offsets=head.relation_offset.detach().numpy()
    result={str(s):np.zeros(len(frame)) for s in (1.,1.5)}
    for k in range(0,len(names),12):
        seq=names[k:k+12]
        rgb=torch.from_numpy(images("val",seq)).permute(0,3,1,2).float()/255
        for scale in (1.,1.5):
            view=rgb if scale==1 else F.interpolate(rgb,scale_factor=scale,mode="bilinear",align_corners=False)
            x,xy,valid=dense_bank(model(view))
            x[...,v.BOX]/=scale
            xy=(xy+1)/scale-1
            bank=table_predictions(head,[z.numpy() for z in (x,xy,valid)],torch.device("cpu"))
            p,position,mask,*attrs=bank
            for j,name in enumerate(seq):
                rows=groups[name]
                twins=find_twins(frame.image.iloc[rows],[caps[r] for r in rows],frame.category.iloc[rows])
                keep=mask[j]
                result[str(scale)][rows]=sample_scene_scores([programs[r] for r in rows],twins,p[j,keep],
                    [(g,a[j,keep]) for g,a in zip(GROUPS,attrs)],position[j,keep],offsets,42)
        if k%120==0:print("scenes",k+len(seq),flush=True)
    np.savez(DEST/"scale_val.npz",ids=frame.id,**result)
    with zipfile.ZipFile(DEST.parent/"phase_run/phase_results.zip") as z:
        with np.load(io.BytesIO(z.read("validation.npz"))) as a:
            assert np.array_equal(frame.id,a["ids"])
            base=a["base"]
    eligible=frame.category.isin(["counting","comparison","spatial_counting"])
    scores=base+.25*(result["1.5"]-result["1.0"])*eligible
    print("scale control, enlarged",*[roc_auc_score(frame.label,x) for x in result.values()])
    print("anchored scale delta",roc_auc_score(frame.label,scores),int(((frame.label==1)&(scores<=0)).sum()))
    for name in ("58281.jpg","26268.jpg","78612.jpg"):
        use=frame.image.eq(name)&frame.label.eq(1)&eligible
        print(name,base[use],scores[use].to_numpy())

if __name__=="__main__":
    evaluate()
