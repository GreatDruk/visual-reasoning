"""Build the offline candidate verifier notebook and private inputs."""
import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd

from package_fragment_votes import ROOT, DIR, SETUP, npz


def bundle():
    dest=DIR/"clevr_rear.zip"
    with zipfile.ZipFile(DIR/"clevr_verifier.zip") as old, \
         zipfile.ZipFile(DIR/"clevr_fragments.zip") as pieces, \
         zipfile.ZipFile(DIR/"phase_run/phase_results.zip") as union, \
         zipfile.ZipFile(DIR/"count_run/count_results.zip") as count, \
         zipfile.ZipFile(dest.with_suffix(".tmp"),"w") as new:
        manifest={}
        def write(name,content):
            manifest[name]=hashlib.sha256(content).hexdigest()
            new.writestr(name,content)
        priors=("r50_img.pt","r50_img_head.pt","thaw.pt","thaw_head.pt","scene_silhouette.pt")
        for name in old.namelist():
            if name.endswith(".py") or name in {f"priors/{x}" for x in priors}:
                write(name,old.read(name))
        write("fragment_train.npz",pieces.read("fragment_train.npz"))
        write("scripts/research/fragment_votes.py",(ROOT/"scripts/research/fragment_votes.py").read_bytes())
        write("scripts/research/rear_candidate.py",(ROOT/"scripts/research/rear_candidate.py").read_bytes())
        for split in ("train","trainval"):
            write(f"priors/union_{split}.pt",union.read(f"{split}/control_421.pt"))
        val=pd.read_csv(ROOT/"data/val.csv")
        with np.load(io.BytesIO(union.read("validation.npz"))) as f:
            baseline=f["base"]
            assert np.array_equal(val.id,f["ids"])
        with np.load(DIR/"dense_count_pilot/hidden_val.npz") as f:
            hidden=f["score"]
            assert np.array_equal(val.id,f["ids"])
        counters=[]
        for seed in (421,853):
            with np.load(io.BytesIO(count.read(f"r50_{seed}/verify_result/control_{seed}_val.npz"))) as f:
                assert np.array_equal(f["ids"],val.id)
                counters.append(f["score"])
        r50=np.mean(counters,axis=0)
        active=val.category.isin(["counting","comparison"])
        confirmed=np.where(active,.5*(baseline+r50),baseline)
        candidate=np.where(active,.75*baseline+.25*r50,baseline)+.5*(hidden-baseline)
        write("anchor_val.npz",npz(ids=val.id.to_numpy(),score=candidate,confirmed=confirmed))
        test=pd.read_csv(ROOT/"data/test.csv")
        scores=pd.read_csv(DIR/"count_run/count_hidden.csv",float_precision="round_trip").set_index("id")
        assert scores.index.is_unique and set(scores.index)==set(test.id)
        write("anchor_test.npz",npz(ids=test.id.to_numpy(),probability=scores.loc[test.id,"label"].to_numpy()))
        new.writestr("rear_manifest.json",json.dumps(manifest))
    with zipfile.ZipFile(dest.with_suffix(".tmp")) as archive:
        assert archive.testzip() is None
    dest.with_suffix(".tmp").replace(dest)


TRAIN="""
env.update(FRAGMENT_INPUT=str(package),OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
seeds=(421,853)
root_out=ROOT/"rear_result"
def run(gpu,action,split,shard,seed):
    stage=f"{action}_{split}_{shard}_{seed}"
    done=ROOT/f"{stage}.done"
    if done.exists():return
    cmd=[sys.executable,"-m","scripts.research.rear_candidate",action,"--split",split,
         "--shard",str(shard),"--seed",str(seed)]
    with (RESULT/f"{stage}.log").open("a") as log, subprocess.Popen(cmd,cwd=ROOT,
            env={**env,"CUDA_VISIBLE_DEVICES":str(gpu)},stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,bufsize=1) as proc:
        for line in proc.stdout:
            log.write(line)
            print(f"[T4 {gpu}] {line}",end="",flush=True)
        if proc.wait():
            raise RuntimeError(f"Не завершился этап {stage}. Повторите эту ячейку.")
    done.touch()

def both(left,right):
    with ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(run,i,*args) for i,args in enumerate((left,right))]
        for future in futures:future.result()

def prepare(split):
    both(*[("prepare",split,i,421) for i in (0,1)])
def predict(split):
    prepare(split)
    for seed in seeds:
        both(*[("predict",split,i,seed) for i in (0,1)])

prepare("train")
both(*[("fit","train",0,s) for s in seeds])
predict("val")
"""
EVAL="""
frame=pd.read_csv(DATA/"val.csv")
CORE=("counting","comparison","spatial_counting")
def deltas(split,df):
    arms=[]
    accepted=[]
    for seed in seeds:
        chunks=[]
        for shard in (0,1):
            with np.load(root_out/f"{split}_{seed}_{shard}.npz") as a:
                chunks.append(pd.DataFrame(dict(delta=a["delta"],selected=a["selected"]),index=a["ids"]))
        correction=pd.concat(chunks)
        assert correction.index.is_unique and correction.index.isin(df.id).all()
        assert np.isfinite(correction.delta).all()
        change=correction.delta.reindex(df.id,fill_value=0.).to_numpy()
        arms.append(np.where(df.category.isin(CORE),change,0.))
        accepted.append(correction.selected.reindex(df.id,fill_value=-1).to_numpy())
    agree=(accepted[0]>=0)&(accepted[0]==accepted[1])
    return arms,np.where(agree,np.mean(arms,axis=0),0.)

with np.load(package/"anchor_val.npz") as a:
    assert np.array_equal(a["ids"],frame.id)
    base=a["score"]
    confirmed=a["confirmed"]
assert abs(roc_auc_score(frame.label,base)-.9998736666666667)<1e-10
arms,revision=deltas("val",frame)
score=base+revision
y=frame.label.to_numpy()
scenes=pd.factorize(frame.image)[0]
n=scenes.max()+1
resamples=np.random.default_rng(705).multinomial(n,np.full(n,1/n),size=1000)[:,scenes]
def measure(x):
    roc=roc_auc_score(y,x)
    errs=((x<=0)&(y==1)).sum()
    return dict(auc=float(roc),wrong_or_tied=int(errs),inversions=float(round((1-roc)*9_000_000,2)))
metrics={k:measure(x) for k,x in [
    ("r50_count",confirmed),("count_hidden",base),("seed421",base+arms[0]),
    ("seed853",base+arms[1]),("rear",score)]}
display(pd.DataFrame(metrics).T)
diff=[roc_auc_score(y,score,sample_weight=w)-roc_auc_score(y,base,sample_weight=w) for w in resamples]
ci=np.quantile(diff,[.025,.975])
joined=frame.assign(base=base,rear=score,change=score-base)
joined.to_csv(RESULT/"val_rows.csv",index=False)
changed=joined[joined.label.eq(1)&((joined.base<=0)|(joined.rear<=0))]
display(changed[["image","caption","base","rear","change"]].sort_values("base"))
records=[]
for path in sorted(root_out.glob("candidates_*.json")):
    for row in json.loads(path.read_text(encoding="utf8")):
        records.append(dict(run=path.stem,**row))
cards=pd.DataFrame(records)
weak_counts={}
if len(cards):
    display(cards[cards.image.isin(["58281.jpg","78612.jpg","26268.jpg"])])
    cards["seed"]=cards.run.str.split("_").str[1]
    for seed,group in cards.groupby("seed"):
        weak_counts[seed]={}
        for label in (0,1):
            known=group[group.weak_target.eq(label)]
            weak_counts[seed][str(label)]=dict(total=len(known),promoted=int(known.used.sum()))
    print("Слабая разметка по точным подписям val",weak_counts)
# A single candidate may independently change a twin sign or only its rank.
fixed=int(((y==1)&(base<=0)&(score>0)).sum())
new_errors=int(((y==1)&(base>0)&(score<=0)).sum())
passed=bool(metrics["rear"]["auc"]>metrics["count_hidden"]["auc"] and new_errors<=fixed)
(RESULT/"metrics.json").write_text(json.dumps(dict(metrics=metrics,ci=ci.tolist(),fixed=fixed,
    new_errors=new_errors,caption_constraints=weak_counts,test=passed),indent=2),encoding="utf8")
print("95% интервал изменения AUC",ci,"Исправлено",fixed,"Новых ошибок",new_errors,"Инференс test",passed)
"""
END="""
if passed:
    predict("test")
    test=pd.read_csv(DATA/"test.csv")
    with np.load(package/"anchor_test.npz") as a:
        assert np.array_equal(a["ids"],test.id)
        probability=a["probability"]
    _,change=deltas("test",test)
    update=change!=0
    probability[update]=expit(logit(probability[update].clip(1e-15,1-1e-15))+change[update])
    sample=pd.read_csv(DATA/"sample_submission.csv")
    assert sample.id.is_unique and test.id.is_unique and set(sample.id)==set(test.id)
    sample["label"]=pd.Series(probability,index=test.id).loc[sample.id].to_numpy()
    path=WORK/"count_rear.csv"
    sample.to_csv(path,index=False,float_format="%.17g")
    assert sample.equals(pd.read_csv(path,float_precision="round_trip")) and sample.label.between(0,1).all()
with zipfile.ZipFile(WORK/"rear_results.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for folder in (RESULT,root_out):
        for path in folder.iterdir():
            if path.is_file():
                z.write(path,path.relative_to(ROOT))
os.chdir(WORK)
display(FileLink("rear_results.zip"))
if passed:
    display(FileLink("count_rear.csv"))
"""
def notebook():
    setup=SETUP.replace("clevr_fragments","clevr_rear").replace("fragments_manifest","rear_manifest")
    setup+="\nfrom scipy.special import logit\n"
    content=[
        ("markdown","# Проверка слабых кандидатов при перекрытии\n\nПодключите конкурсные файлы и приватный clevr_rear. Выберите GPU T4 ×2. Исходная модель для сравнения — count_hidden. Каждый GPU обучает независимый seed."),
        ("code",setup),("markdown","## Извлечение кандидатов, обучение и валидация"),("code",TRAIN),
        ("markdown","## AUC, исправленные пары и кандидаты на сложных сценах"),("code",EVAL),
        ("markdown","## Архив и условный инференс на test"),("code",END)]
    cells=[]
    for i,(kind,text) in enumerate(content):
        cell=dict(cell_type=kind,id=f"rear-{i}",metadata={},source=text.strip().splitlines(True))
        if kind=="code":
            compile(text,f"cell{i}","exec")
            cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    nb=dict(cells=cells,nbformat=4,nbformat_minor=5,metadata={
        "kernelspec":dict(display_name="Python 3",language="python",name="python3")})
    (ROOT/"notebooks/11_rear_candidates.ipynb").write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding="utf8")


if __name__=="__main__":
    bundle()
    notebook()
