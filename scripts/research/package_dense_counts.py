"""Package two count models, including an independent torchvision ImageNet encoder."""
from pathlib import Path
import hashlib
import json
import zipfile

from package_fragment_votes import SETUP, ROOT, DIR


def build():
    dest=DIR/"clevr_counters.zip"
    with zipfile.ZipFile(DIR/"clevr_fragments.zip") as old,zipfile.ZipFile(dest.with_suffix(".tmp"),"w") as new:
        hashes={}
        def put(key,blob):
            hashes[key]=hashlib.sha256(blob).hexdigest()
            new.writestr(key,blob)
        for k in old.namelist():
            if not k.endswith("manifest.json"):
                put(k,old.read(k))
        name="convnext_tiny-983f1562.pth"
        put(name,(ROOT/"artefacts/research"/name).read_bytes())
        if hashes[name]!="983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d":
            raise ValueError(name)
        put("scripts/research/dense_counts.py",(ROOT/"scripts/research/dense_counts.py").read_bytes())
        new.writestr("counters_manifest.json",json.dumps(hashes))
    with zipfile.ZipFile(dest.with_suffix(".tmp")) as z:assert z.testzip() is None
    dest.with_suffix(".tmp").replace(dest)


TRAIN="""
env.update(FRAGMENT_INPUT=str(package))
def run(arch,seed,stage,gpu):
    folder=ROOT/f"{arch}_{seed}"
    folder.mkdir(exist_ok=True)
    done=folder/f"{stage}.done"
    if done.exists():
        return
    cmd=[sys.executable,"-m","scripts.research.dense_counts",stage,str(seed),"--architecture",arch]
    with (WORK/f"count_{arch}_{seed}_{stage}.log").open("a") as log, subprocess.Popen(cmd,cwd=folder,
            env={**env,"CUDA_VISIBLE_DEVICES":str(gpu)},stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,bufsize=1) as proc:
        for line in proc.stdout:
            log.write(line)
            print(arch,seed,line,end="",flush=True)
        if proc.wait():
            raise RuntimeError(f"Ошибка {arch}/{stage}. Повторите ячейку в той же сессии.")
    done.touch()
def parallel(jobs):
    with ThreadPoolExecutor(2) as pool:
        pending=[pool.submit(run,*args) for args in jobs]
        for job in pending:
            job.result()
arches=("r50","convnext")
for stage in ("fit","val"):
    parallel([(arch,421,stage,gpu) for gpu,arch in enumerate(arches)])
"""
EVAL="""
def read(path,ids):
    with np.load(path) as z:
        assert np.array_equal(z["ids"],ids) and np.isfinite(z["score"]).all()
        return z["score"]

def blend(split,models):
    df=pd.read_csv(DATA/f"{split}.csv")
    anchor=read(package/f"baseline_{split}.npz",df.id)
    predictions=[read(ROOT/f"{arch}_{seed}/verify_result/control_{seed}_{split}.npz",df.id) for arch,seed in models]
    active=df.category.isin(["counting","comparison"])
    return np.where(active,.5*(anchor+np.mean(predictions,axis=0)),anchor)

df=pd.read_csv(DATA/"val.csv")
y=df.label.to_numpy()
base=read(package/"baseline_val.npz",df.id)
assert abs(roc_auc_score(y,base)-.9997986666666666)<1e-9
scenes=pd.factorize(df.image)[0];num=len(np.unique(scenes))
rng=np.random.default_rng(316)
resamples=rng.multinomial(num,np.full(num,1/num),size=1000)[:,scenes]
def metrics(x):
    delta=[roc_auc_score(y,x,sample_weight=w)-roc_auc_score(y,base,sample_weight=w) for w in resamples]
    return dict(auc=float(roc_auc_score(y,x)),wrong=int(((y==1)&(x<0)).sum()),
                tied=int(((y==1)&(x==0)).sum()),lower=float(np.quantile(delta,.025)),
                upper=float(np.quantile(delta,.975)))
candidates={a:blend("val",[(a,421)]) for a in arches}
report={a:metrics(x) for a,x in candidates.items()}
display(pd.DataFrame(report).T)
# Two architecture comparisons: one-sided 2.5% bound per comparison.
eligible=[a for a in arches if report[a]["lower"]>0]
winner=max(eligible,key=lambda a:report[a]["auc"],default=None)
val_rows=df.assign(base=base,**candidates)
val_rows.to_csv(RESULT/"val_rows.csv",index=False)
print("Архитектура для независимого seed",winner)
"""
END="""
passed=False
if winner:
    run(winner,853,"fit",0)
    run(winner,853,"val",0)
    repeat=blend("val",[(winner,853)])
    result=blend("val",[(winner,421),(winner,853)])
    report["repeat"]=metrics(repeat)
    report["two_seeds"]=metrics(result)
    val_rows.assign(repeat=repeat,two_seeds=result).to_csv(RESULT/"val_rows.csv",index=False)
    passed=report["repeat"]["auc"]>roc_auc_score(y,base) and report["two_seeds"]["lower"]>0
    display(pd.DataFrame(report).T)
if passed:
    parallel([(winner,421,"test",0),(winner,853,"test",1)])
    frame=pd.read_csv(DATA/"test.csv")
    sample=pd.read_csv(DATA/"sample_submission.csv")
    assert frame.id.is_unique and sample.id.is_unique and set(frame.id)==set(sample.id)
    logits=blend("test",[(winner,421),(winner,853)])
    sample["label"]=pd.Series(expit(logits),index=frame.id).loc[sample.id].to_numpy()
    name=f"{winner}_count.csv"
    sample.to_csv(WORK/name,index=False,float_format="%.17g")
    assert sample.equals(pd.read_csv(WORK/name,float_precision="round_trip")) and sample.label.between(0,1).all()
(RESULT/"metrics.json").write_text(json.dumps(dict(report=report,winner=winner,eligible=bool(passed)),indent=2))
with zipfile.ZipFile(WORK/"count_results.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for folder in [RESULT,*ROOT.glob("*/verify_result")]:
        for path in folder.glob("*"):
            z.write(path,path.relative_to(ROOT))
    for path in WORK.glob("count_*.log"):
        z.write(path,path.name)
os.chdir(WORK)
display(FileLink("count_results.zip"))
if passed:
    display(FileLink(name))
"""
def notebook():
    setup=SETUP.replace("clevr_fragments","clevr_counters").replace("fragments_manifest","counters_manifest")
    parts=[
        ("markdown","# Подсчёт объектов по целому изображению\n\nДанные соревнования, приватный clevr_counters, две T4, интернет выключен. Контроль — union с фиксированной калибровкой. Первая GPU обучает ResNet, вторая — ConvNeXt-Tiny с подсчётом по областям."),
        ("code",setup),("markdown","## Обучение двух архитектур и прогноз на val"),("code",TRAIN),
        ("markdown","## Сравнение с текущей моделью"),("code",EVAL),
        ("markdown","## Повтор победителя и результаты"),("code",END)]
    cells=[]
    for i,(kind,text) in enumerate(parts):
        c=dict(cell_type=kind,id=f"count-{i}",metadata={},source=text.strip().splitlines(True))
        if kind=="code":
            compile(text,f"cell{i}","exec")
            c.update(outputs=[],execution_count=None)
        cells.append(c)
    nb=dict(cells=cells,nbformat=4,nbformat_minor=5,metadata={"kernelspec":dict(display_name="Python 3",language="python",name="python3")})
    (ROOT/"notebooks/10_dense_counts.ipynb").write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding="utf8")

if __name__=="__main__":
    build()
    notebook()
