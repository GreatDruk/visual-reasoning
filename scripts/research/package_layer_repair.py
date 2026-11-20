"""Assemble a self-contained two-architecture Kaggle occlusion experiment."""
import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd

from package_fragment_votes import ROOT,DIR,SETUP,npz


def inputs():
    target=DIR/"clevr_layers.zip"
    with zipfile.ZipFile(DIR/"clevr_rear.zip") as source, \
         zipfile.ZipFile(DIR/"phase_run/phase_results.zip") as phase, \
         zipfile.ZipFile(target.with_suffix(".tmp"),"w") as archive:
        hashes={}
        def put(key,data):
            archive.writestr(key,data)
            hashes[key]=hashlib.sha256(data).hexdigest()
        skip={"anchor_val.npz","anchor_test.npz","rear_manifest.json"}
        for k in source.namelist():
            if k not in skip:put(k,source.read(k))
        put("scripts/research/layer_repair.py",(ROOT/"scripts/research/layer_repair.py").read_bytes())
        with np.load(ROOT/"artefacts/research/teacher_train.npz") as z:
            stream=io.BytesIO()
            np.save(stream,z["background"])
            put("train_background.npy",stream.getvalue())
        with np.load(io.BytesIO(phase.read("validation.npz"))) as z:
            assert np.array_equal(z["ids"],pd.read_csv(ROOT/"data/val.csv").id)
            put("anchor_val.npz",npz(ids=z["ids"],score=z["base"]))
        original=pd.read_csv(DIR/"phase_run/union_tv.csv",float_precision="round_trip")
        expected=pd.read_csv(ROOT/"data/sample_submission.csv")
        test=pd.read_csv(ROOT/"data/test.csv")
        assert original.id.equals(expected.id) and original.id.is_unique and test.id.is_unique
        probability=original.set_index("id").loc[test.id,"label"].to_numpy()
        assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
        put("anchor_test.npz",npz(ids=test.id.to_numpy(),probability=probability))
        archive.writestr("layers_manifest.json",json.dumps(hashes))
    with zipfile.ZipFile(target.with_suffix(".tmp")) as z:
        assert z.testzip() is None
    target.with_suffix(".tmp").replace(target)


TRAIN="""
env.update(FRAGMENT_INPUT=str(package),OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
arms=("verify","split")
seeds=(421,853)
scores=ROOT/"layer_result"

def run(gpu,stage,arm,split,seed,shard):
    step=f"{stage}_{arm}_{split}_{seed}_{shard}"
    done=ROOT/(step+".done")
    if done.exists():
        return
    cmd=[sys.executable,"-m","scripts.research.layer_repair",stage,"--arm",arm,
         "--split",split,"--seed",str(seed),"--shard",str(shard)]
    with (RESULT/(step+".log")).open("a") as log, subprocess.Popen(
            cmd,cwd=ROOT,env={**env,"CUDA_VISIBLE_DEVICES":str(gpu)},stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,bufsize=1) as job:
        for line in job.stdout:
            log.write(line)
            print(f"[GPU {gpu}] {line}",end="",flush=True)
        if job.wait():
            raise RuntimeError(f"{step}. Повторите ячейку в этой сессии.")
    done.touch()

def parallel(jobs):
    with ThreadPoolExecutor(2) as pool:
        for job in [pool.submit(run,gpu,*args) for gpu,args in enumerate(jobs)]:
            job.result()

def prepare(split):
    parallel([("prepare","verify",split,421,k) for k in (0,1)])
def predict(arm,seed,split):
    parallel([("predict",arm,split,seed,k) for k in (0,1)])

start=time.monotonic()
prepare("train")
for seed in seeds:
    parallel([("fit",a,"train",seed,0) for a in arms])
prepare("val")
for a in arms:
    for seed in seeds:
        predict(a,seed,"val")
print("Минут до получения val",round((time.monotonic()-start)/60,2))
"""
EVAL="""
CORE=("counting","comparison","spatial_counting")
def revision(arm,seed,split,data):
    parts=[]
    for shard in (0,1):
        with np.load(scores/f"{arm}_{seed}_{split}_{shard}.npz") as z:
            parts.append(pd.Series(z["delta"],index=z["ids"]))
    rows=pd.concat(parts)
    assert rows.index.is_unique and rows.index.isin(data.id).all() and np.isfinite(rows).all()
    change=rows.reindex(data.id,fill_value=0.).to_numpy()
    return np.where(data.category.isin(CORE),.75*change,0.)

def average(arm,split,data):
    return np.mean([revision(arm,seed,split,data) for seed in seeds],axis=0)

val=pd.read_csv(DATA/"val.csv")
y=val.label.to_numpy()
with np.load(package/"anchor_val.npz") as z:
    assert np.array_equal(val.id,z["ids"])
    baseline=z["score"]
base_auc=roc_auc_score(y,baseline)
assert abs(base_auc-.9997986666666666)<1e-8
factor=pd.factorize(val.image)[0]
n=len(np.unique(factor))
rng=np.random.default_rng(315)
weights=rng.multinomial(n,np.full(n,1/n),size=1000)[:,factor]
reference_auc=np.array([roc_auc_score(y,baseline,sample_weight=w) for w in weights])
def metrics(s,bootstrap=True):
    auc=roc_auc_score(y,s)
    false=int(((y==1)&(s<=0)&(baseline>0)).sum())
    fixed=int(((y==1)&(s>0)&(baseline<=0)).sum())
    if bootstrap:
        levels=np.array([roc_auc_score(y,s,sample_weight=w) for w in weights])-reference_auc
        confidence=np.quantile(levels,[.025,.975]).tolist()
    else:confidence=[0.,0.]
    return dict(auc=float(auc),errors=int(((y==1)&(s<=0)).sum()),fixed=fixed,regressions=false,
                delta_ci=confidence,
                inversions=round(float((1-auc)*sum(y)*sum(1-y)),2))
report={"union_tv_control":metrics(baseline,False)}
frame=val.assign(baseline=baseline)
candidates={}
for arm in arms:
    names=[]
    for seed in seeds:
        key=f"{arm}_{seed}"
        frame[key]=baseline+revision(arm,seed,"val",val)
        report[key]=metrics(frame[key].to_numpy(),False)
        names.append(key)
    frame[arm]=baseline+average(arm,"val",val)
    report[arm]=metrics(frame[arm].to_numpy())
    m=report[arm]
    # Require real improvements in choice as well as ranking.
    if m["auc"]>base_auc and m["fixed"]>m["regressions"] and all(report[k]["auc"]>base_auc for k in names):
        candidates[arm]=m["auc"]
winner=max(candidates,key=candidates.get,default=None)
frame.to_csv(RESULT/"val_rows.csv",index=False)
display(pd.DataFrame(report).T)
display(frame[frame.label.eq(1)&((frame.baseline<=0)|(frame.verify<=0)|(frame.split<=0))]
        [["image","caption","baseline","verify","split"]].sort_values("baseline"))
focused=[]
for path in scores.glob("*candidates.json"):
    focused.extend(dict(run=path.stem,**row) for row in json.loads(path.read_text())
                   if row["image"] in ("58281.jpg","78612.jpg","26268.jpg"))
if focused:display(pd.DataFrame(focused))
(RESULT/"metrics.json").write_text(json.dumps(dict(arms=report,winner=winner),indent=2))
print("Архитектура для test",winner)
"""
EXPORT="""
if winner:
    prepare("test")
    for seed in seeds:
        predict(winner,seed,"test")
    test=pd.read_csv(DATA/"test.csv")
    with np.load(package/"anchor_test.npz") as z:
        assert np.array_equal(test.id,z["ids"])
        probability=z["probability"]
    delta=average(winner,"test",test)
    use=delta!=0
    probability[use]=expit(logit(probability[use].clip(1e-15,1-1e-15))+delta[use])
    sample=pd.read_csv(DATA/"sample_submission.csv")
    assert sample.id.is_unique and test.id.is_unique and set(sample.id)==set(test.id)
    sample["label"]=pd.Series(probability,index=test.id).loc[sample.id].to_numpy()
    filename=f"occlude_{winner}.csv"
    sample.to_csv(WORK/filename,index=False,float_format="%.17g")
    assert pd.read_csv(WORK/filename,float_precision="round_trip").equals(sample) and sample.label.between(0,1).all()
with zipfile.ZipFile(WORK/"layer_results.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for folder in (RESULT,scores):
        for path in folder.iterdir():
            if path.is_file():z.write(path,path.relative_to(ROOT))
os.chdir(WORK)
display(FileLink("layer_results.zip"))
if winner:display(FileLink(filename))
"""

def notebook():
    setup=SETUP.replace("clevr_fragments","clevr_layers").replace("fragments_manifest","layers_manifest")
    setup+="\nfrom scipy.special import logit\n"
    content=[
        ("markdown","# Распознавание частично закрытых объектов\n\nПодключите данные конкурса и приватный clevr_layers. Выберите GPU T4 ×2. Исходный прогноз — union_tv. Первый модуль проверяет слабую рамку. Второй восстанавливает рамку заднего объекта по одному переднему. Обе модели обучаются на исходных train JPEG, искусственных перекрытиях и истинности train-подписей."),
        ("code",setup),("markdown","## Два seed каждой архитектуры, 64 эпохи на запуск"),("code",TRAIN),
        ("markdown","## Ошибки и оценки кандидатов на исходном val"),("code",EVAL),
        ("markdown","## Архив и предсказания победившей архитектуры"),("code",EXPORT)]
    cells=[]
    for i,(typ,src) in enumerate(content):
        cell=dict(cell_type=typ,id=f"layer-{i}",metadata={},source=src.strip().splitlines(True))
        if typ=="code":
            compile(src,f"cell{i}","exec")
            cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    result=dict(cells=cells,nbformat=4,nbformat_minor=5,
                metadata={"kernelspec":dict(display_name="Python 3",language="python",name="python3")})
    (ROOT/"notebooks/12_layer_repair.ipynb").write_text(json.dumps(result,ensure_ascii=False,indent=1),encoding="utf-8")

if __name__=="__main__":
    inputs()
    notebook()
