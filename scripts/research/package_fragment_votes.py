"""Build the offline Kaggle fragment experiment from the team's existing train artefacts."""
import io
import json
from pathlib import Path
import zipfile
import hashlib

import numpy as np
import pandas as pd
from scipy.special import logit

from package_occlusion_verifier import SETUP as PREVIOUS_SETUP

ROOT=Path(__file__).resolve().parents[2]
DIR=ROOT/"artefacts/kaggle_suite"
PKG=DIR/"clevr_fragments.zip"

def npz(**values):
    stream=io.BytesIO()
    np.savez_compressed(stream,**values)
    return stream.getvalue()

def bundle():
    with np.load(ROOT/"artefacts/research/teacher_train.npz") as z:
        names=z["names"]
        assert np.array_equal(names,pd.read_csv(ROOT/"data/train.csv").image.unique())
        feat=z["features"]
        box=np.rint(feat[...,8:12]*[240,160,240,160]).astype(np.int16)
        x,y,w,h=box.transpose(2,0,1)
        keep=z["valid"]&(z["p"]>.84)&(w>=8)&(h>=8)&(x>=0)&(y>=0)&(x+w<=240)&(y+h<=160)
        labels=[np.where(z[k].max(-1)>=.85,z[k].argmax(-1),-1) for k in ("size","color","material","shape")]
        sz,col,mat,shape=labels
        labels=np.stack([sz,col,np.where((mat>=0)&(shape>=0),3*mat+shape,-1)],-1).astype(np.int8)
        masks=(255*feat[...,47:147].clip(0,1)).round().astype(np.uint8).reshape(*keep.shape,10,10)
        train=npz(names=names,keep=keep,labels=labels,box=box,mask=masks)
    with zipfile.ZipFile(PKG.with_suffix(".tmp"),"w") as out,zipfile.ZipFile(DIR/"clevr_verifier.zip") as old:
        manifest={}
        def write(name,content):
            out.writestr(name,content)
            manifest[name]=hashlib.sha256(content).hexdigest()
        for name in old.namelist():
            if name.endswith(".py") or name in ["priors/r50_img.pt","priors/r50_img_head.pt","priors/scene_silhouette.pt"]:
                write(name,old.read(name))
        write("scripts/research/fragment_votes.py",(ROOT/"scripts/research/fragment_votes.py").read_bytes())
        write("fragment_train.npz",train)
        with zipfile.ZipFile(DIR/"phase_run/phase_results.zip") as z:
            with np.load(io.BytesIO(z.read("validation.npz"))) as v:
                write("baseline_val.npz",npz(ids=v["ids"],score=v["base"]))
        test=pd.read_csv(ROOT/"data/test.csv")
        sent=pd.read_csv(DIR/"phase_run/union_tv.csv",float_precision="round_trip").set_index("id")
        write("baseline_test.npz",npz(ids=test.id.to_numpy(),score=logit(sent.loc[test.id,"label"].to_numpy())))
        out.writestr("fragments_manifest.json",json.dumps(manifest))
    with zipfile.ZipFile(PKG.with_suffix(".tmp")) as z:
        assert z.testzip() is None
    PKG.with_suffix(".tmp").replace(PKG)


SETUP=PREVIOUS_SETUP.split("WORKER =")[0].replace("clevr_verifier","clevr_fragments").replace("verifier_manifest","fragments_manifest")
RUN="""
env.update(FRAGMENT_INPUT=str(package))
seeds=(421,853)
def run(seed,gpu,stage):
    directory=ROOT/f"seed_{seed}"
    directory.mkdir(exist_ok=True)
    done=directory/f"{stage}.done"
    if done.exists():
        return
    command=[sys.executable,"-m","scripts.research.fragment_votes",stage,str(seed)]
    logfile=WORK/f"fragments_{seed}_{stage}.log"
    with logfile.open("w") as log, subprocess.Popen(command,cwd=directory,
            env={**env,"CUDA_VISIBLE_DEVICES":str(gpu)},stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
        for line in process.stdout:
            print(seed,line,end="",flush=True)
            log.write(line)
        if process.wait():
            raise RuntimeError(f"{stage} seed {seed}: {logfile}")
    done.touch()

def both(stage):
    with ThreadPoolExecutor(2) as pool:
        jobs=[pool.submit(run,s,g,stage) for g,s in enumerate(seeds)]
        for job in jobs:
            job.result()

both("fit")
both("val")
"""
EVAL="""
def outputs(split):
    frame=pd.read_csv(DATA/f"{split}.csv")
    def read(path):
        with np.load(path) as a:
            if not np.array_equal(a["ids"],frame.id) or not np.isfinite(a["score"]).all():
                raise ValueError(path)
            return a["score"]
    base=read(package/f"baseline_{split}.npz")
    core=frame.category.isin(["counting","spatial_counting","comparison"]).to_numpy()
    branches=[read(ROOT/f"seed_{s}/verify_result/control_{s}_{split}.npz") for s in seeds]
    def combine(branch):
        return np.where(core,.5*(base+branch),base)
    return frame,base,[combine(b) for b in branches],combine(np.mean(branches,axis=0))

frame,base,replicates,score=outputs("val")
y=frame.label.to_numpy()
def metrics(x):
    auc=roc_auc_score(y,x)
    return dict(auc=float(auc),wrong=int(((y==1)&(x<0)).sum()),tied=int(((y==1)&(x==0)).sum()),
                inversions=float(round((1-auc)*sum(y)*sum(1-y),1)))
summary={k:metrics(x) for k,x in zip(("baseline","seed421","seed853","fragments"),[base,*replicates,score])}
assert abs(summary["baseline"]["auc"]-.9997986666666666)<1e-8
display(pd.DataFrame(summary).T)
scene=pd.factorize(frame.image)[0]
n=len(np.unique(scene))
rng=np.random.default_rng(425)
delta=[]
for _ in range(1000):
    w=rng.multinomial(n,np.full(n,1/n))[scene]
    delta.append(roc_auc_score(y,score,sample_weight=w)-roc_auc_score(y,base,sample_weight=w))
ci=np.quantile(delta,[.025,.975])
passed=bool(ci[0]>0 and all(metrics(x)["auc"]>summary["baseline"]["auc"] for x in replicates)
            and summary["fragments"]["wrong"]+summary["fragments"]["tied"]<=summary["baseline"]["wrong"])
audit=frame.assign(base=base,fragments=score,corrected=(y==1)&(base<0)&(score>0),
                   regressed=(y==1)&(base>0)&(score<=0))
audit.to_csv(RESULT/"val_rows.csv",index=False)
display(audit[audit.label.eq(1)&((audit.base<=0)|(audit.fragments<=0))][
    ["image","caption","category","base","fragments","corrected","regressed"]].sort_values("base"))
# A wrong larger exact-number twin is an observable overcount on real scenes.
from scripts.research.native_centers import canonical_form,parse_program
overcounts={}
for key,values in (("base",base),("fragments",score)):
    over=under=0
    for _,g in frame[frame.category.isin(["counting","spatial_counting"])].groupby(["image","pair_id"]):
        pos=g[g.label==1].iloc[0]
        neg=g[g.label==0].iloc[0]
        if values[neg.name]<=values[pos.name]:
            continue
        number=lambda r:parse_program(canonical_form(r.caption,r.category)).number
        over+=int(number(neg)>number(pos))
        under+=int(number(neg)<number(pos))
    overcounts[key]=dict(over=over,under=under)
print("Точные количества на реальных сценах",overcounts)
report=dict(metrics=summary,delta_ci=ci.tolist(),eligible=passed,count_errors=overcounts)
(RESULT/"metrics.json").write_text(json.dumps(report,indent=2))
print("Интервал прироста",ci,"Переход к test",passed)
"""
END="""
if passed:
    both("test")
    frame,_,_,score=outputs("test")
    sample=pd.read_csv(DATA/"sample_submission.csv")
    assert frame.id.is_unique and sample.id.is_unique and set(frame.id)==set(sample.id)
    sample["label"]=pd.Series(expit(score),index=frame.id).loc[sample.id].to_numpy()
    path=WORK/"fragment_union.csv"
    sample.to_csv(path,index=False,float_format="%.17g")
    assert pd.read_csv(path,float_precision="round_trip").equals(sample) and sample.label.between(0,1).all()
with zipfile.ZipFile(WORK/"fragment_results.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
    for folder in [RESULT,*[ROOT/f"seed_{s}/verify_result" for s in seeds]]:
        for path in folder.glob("*"):
            archive.write(path,path.relative_to(ROOT))
    for path in WORK.glob("fragments_*.log"):
        archive.write(path,path.name)
os.chdir(WORK)
display(FileLink("fragment_results.zip"))
if passed:
    display(FileLink("fragment_union.csv"))
"""
def notebook():
    setup=SETUP+"\nsys.path.insert(0,str(package))\nsys.path.insert(0,str(package/'scripts/research'))\n"
    cells=[]
    for i,(typ,source) in enumerate([
        ("markdown","# Частично видимые объекты\n\nДанные соревнования и приватный clevr_fragments, две T4, интернет выключен. Каждый GPU обучает независимый seed. Метрики считаются на исходных JPEG официального val."),
        ("code",setup),("markdown","## Обучение на парных сценах и прогноз на val"),("code",RUN),
        ("markdown","## Ошибки подсчёта на реальных сценах"),("code",EVAL),
        ("markdown","## Архив и условный прогноз на test"),("code",END)]):
        cell=dict(cell_type=typ,id=f"fragment-{i}",metadata={},source=source.strip().splitlines(True))
        if typ=="code":
            compile(source,f"cell{i}","exec")
            cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    nb=dict(cells=cells,nbformat=4,nbformat_minor=5,metadata={"kernelspec":dict(display_name="Python 3",language="python",name="python3")})
    (ROOT/"notebooks/09_fragments.ipynb").write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding="utf8")

if __name__=="__main__":
    bundle()
    notebook()
