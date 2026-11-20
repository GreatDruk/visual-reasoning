"""Build the offline Kaggle notebook and immutable private inputs."""
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd

from package_fragment_votes import ROOT,DIR,SETUP,npz
from package_visible_fragments import RUN as RUNNER


def bundle():
    dst=DIR/"clevr_detail.zip"
    with zipfile.ZipFile(DIR/"clevr_visible.zip") as visible, \
         zipfile.ZipFile(DIR/"clevr_verifier.zip") as verifier, \
         zipfile.ZipFile(DIR/"clevr_layers.zip") as layers, \
         zipfile.ZipFile(DIR/"count_run/count_results.zip") as count, \
         zipfile.ZipFile(dst.with_suffix(".tmp"),"w",allowZip64=True) as out:
        hashes={}

        def put(key,blob):
            out.writestr(key,blob)
            hashes[key]=hashlib.sha256(blob).hexdigest()

        def transfer(old,name,key):
            sha=hashlib.sha256()
            with old.open(name) as inp,out.open(key,"w",force_zip64=True) as target:
                while chunk:=inp.read(8*1024*1024):
                    sha.update(chunk)
                    target.write(chunk)
            hashes[key]=sha.hexdigest()

        for key in visible.namelist():
            if key!="visible_manifest.json":
                if (ROOT/key).is_file() and key.endswith(".py"):
                    put(key,(ROOT/key).read_bytes())
                else:
                    transfer(visible,key,key)
        for name in ("detail_counts.py","detail_union.py","detail_eval.py","dense_counts.py"):
            key=f"scripts/research/{name}"
            put(key,(ROOT/key).read_bytes())
        transfer(layers,"train_background.npy","train_background.npy")
        for name in ("model.safetensors","config.json"):
            key=f"weights_in/siglip2_vision/{name}"
            transfer(verifier,key,key)
        for name in ("siglip_pool.pt","sigft_pca.npz"):
            key=f"priors/{name}"
            transfer(verifier,key,key)
        for split in ("val","test"):
            frame=pd.read_csv(ROOT/"data"/f"{split}.csv")
            values=[]
            for seed in (421,853):
                with np.load(count.open(f"r50_{seed}/verify_result/control_{seed}_{split}.npz")) as z:
                    assert np.array_equal(z["ids"],frame.id)
                    values.append(z["score"])
            with np.load(visible.open(f"anchor_{split}.npz")) as z:
                assert np.array_equal(z["ids"],frame.id)
                base=z["score"] if split=="val" else z["probability"]
            if split=="test":
                from scipy.special import logit
                base=logit(base.clip(1e-15,1-1e-15))
            score=np.where(frame.category.isin(["counting","comparison"]),.5*(base+np.mean(values,axis=0)),base)
            if split=="val":
                from sklearn.metrics import roc_auc_score
                assert abs(roc_auc_score(frame.label,score)-.9998391111111111)<1e-8
            else:
                from scipy.special import logit
                sent=pd.read_csv(DIR/"count_run/r50_count.csv",float_precision="round_trip").set_index("id")
                assert set(sent.index)==set(frame.id)
                score=logit(sent.loc[frame.id,"label"].to_numpy().clip(1e-15,1-1e-15))
            put(f"r50_count_{split}.npz",npz(ids=frame.id.to_numpy(),score=score))
            transfer(verifier,f"priors/occ_{split}.npz",f"occ_{split}.npz")
        out.writestr("detail_manifest.json",json.dumps(hashes,indent=2))
    with zipfile.ZipFile(dst.with_suffix(".tmp")) as out:
        assert out.testzip() is None
    dst.with_suffix(".tmp").replace(dst)
    print(dst,dst.stat().st_size)


MANAGE=RUNNER[:RUNNER.index("RESULT =")]+'''
RESULT = ROOT / "detail"
RESULT.mkdir(exist_ok=True)
COUNTS = ("pyramid","attention","paired","stride4","maskset")
TABLES = ("union","r50_only","local","contour","geometry","siglip")
SEEDS = (421,853)
COUNT_EPOCHS = 24
TABLE_EPOCHS = 16
C = "scripts.research.detail_counts"
T = "scripts.research.detail_union"
E = "scripts.research.detail_eval"
FAILURES = []
from threading import Lock
FAILURE_LOCK = Lock()
'''

# Reuse the two-GPU launcher, including stage logs and nvidia-smi telemetry.
start=RUNNER.index("def launch")
MANAGE+=RUNNER[start:].replace('step = "_".join(map(str,args)).replace("--","")',
                                'step = module.rsplit(".",1)[-1] + "_" + "_".join(map(str,args)).replace("--","")')
MANAGE=MANAGE[:MANAGE.index("def worker")] + MANAGE[MANAGE.index("def monitor"):]
MANAGE+='''
def safe(gpu,module,*args):
    try:
        launch(gpu,module,*args)
        return True
    except Exception as exc:
        failure = dict(module=module, args=list(args), error=repr(exc))
        with FAILURE_LOCK:
            FAILURES.append(failure)
            (RESULT/"failures.json").write_text(json.dumps(FAILURES,indent=2))
        print("Ошибка конфигурации",failure,flush=True)
        return False

def pair(module, arguments):
    with ThreadPoolExecutor(2) as pool:
        pending=[pool.submit(safe,gpu,module,*args) for gpu,args in enumerate(arguments)]
        return [job.result() for job in pending]

def fit_counts():
    for arm in COUNTS:
        status=pair(C,[("fit",arm,s,"--epochs",COUNT_EPOCHS) for s in SEEDS])
        with ThreadPoolExecutor(2) as pool:
            jobs=[pool.submit(safe,gpu,C,"val",arm,s) for gpu,s in enumerate(SEEDS) if status[gpu]]
            for job in jobs:
                job.result()

def extract(stage,splits):
    good=True
    for split in splits:
        status=pair(T,[(stage,"--split",split,"--shard",gpu) for gpu in (0,1)])
        good=all(status) and good
    return good

def fit_tables():
    if extract("extract",("train","val")):
        for arm in TABLES[:-1]:
            pair(T,[("fit","--arm",arm,"--seed",s,"--epochs",TABLE_EPOCHS) for s in SEEDS])
    if extract("extract_siglip",("train","val")):
        pair(T,[("fit","--arm","siglip","--seed",s,"--epochs",TABLE_EPOCHS) for s in SEEDS])
'''
REPORT='''
evaluation_ok = safe(0,E)
report = json.loads((RESULT/"detail_metrics.json").read_text()) if evaluation_ok else {}
if report:
    display(pd.DataFrame(report["baselines"]).T[["auc","inversions","errors"]])
    runs = pd.DataFrame(report["experiments"]).T
    if len(runs):
        display(runs.sort_values("auc",ascending=False).drop(columns="auc_by_category"))
    parts=[]
    for name,levels in report["overlaps"].items():
        parts.extend(dict(run=name, threshold=q, **values) for q,values in levels.items())
    if parts:
        display(pd.DataFrame(parts))
    if report.get("exact_count_diagnostics"):
        display(pd.DataFrame(report["exact_count_diagnostics"]).T)
    print("Выбрано для test",report.get("winner"),flush=True)
'''
EXPORT='''
winner = report.get("winner")
def test_selected():
    if not winner:
        return
    arm,method=winner
    if arm in COUNTS:
        done=pair(C,[("test",arm,s) for s in SEEDS])
    else:
        stage="extract_siglip" if arm=="siglip" else "extract"
        done=extract(stage,("test",)) and pair(T,[("infer","--arm",arm,"--seed",s,"--split","test") for s in SEEDS])
    if done and all(done) and safe(0,E,"--export"):
        filename=f"{arm}_{method}.csv"
        shutil.copy2(RESULT/filename,WORK/filename)
        display(FileLink(filename))
monitor(test_selected,"test")
with zipfile.ZipFile(WORK/"detail_results.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for path in RESULT.rglob("*"):
        if path.is_file() and path.suffix in {".json",".npz",".pt",".log",".csv"} and "tables" not in path.parts and not path.name.startswith("table_"):
            z.write(path,path.relative_to(RESULT))
os.chdir(WORK)
display(FileLink("detail_results.zip"))
'''


def notebook():
    setup=SETUP.replace("clevr_fragments","clevr_detail").replace("fragments_manifest","detail_manifest")
    cells=[]
    texts=[
        ("markdown","# Подсчёт мелких и перекрытых фигур\n\nПодключите данные соревнования и приватный `clevr_detail`. Включите GPU T4 ×2 и запустите Run All. Для оценки используется исходный val."),
        ("code",setup),("code",MANAGE),
        ("markdown","## Подсчёт до поиска центров и видимые маски\n\nСравниваются пирамида без внимания, внимание по фильтру, парное обучение и шаг 4 вместо 2. Модель набора масок обучается с сопоставлением один-к-одному. У весов R50 общая точка старта."),
        ("code",'monitor(fit_counts,"counts")'),
        ("markdown","## Слабые предложения и объединение объектов\n\nСравниваются переобученные union и R50 без thaw, локальные признаки, контурные предложения, геометрия одной или двух фигур и признаки SigLIP2 в локальном кропе."),
        ("code",'monitor(fit_tables,"tables")'),
        ("markdown","## Метрики и спорные сцены\n\nВ архиве сохраняются AUC, инверсии и ошибки по каждому запуску, таблица четырёх спорных сцен, контроль естественных перекрытий и телеметрия GPU."),
        ("code",REPORT),("code",EXPORT)]
    for i,(kind,source) in enumerate(texts):
        cell=dict(id=f"detail-{i}",cell_type=kind,source=source.strip().splitlines(keepends=True),metadata={})
        if kind=="code":
            compile(source,cell["id"],"exec")
            cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    payload=dict(cells=cells,nbformat=4,nbformat_minor=5,
                 metadata=dict(kernelspec=dict(display_name="Python 3",language="python",name="python3")))
    (ROOT/"notebooks/14_detail_lab.ipynb").write_text(json.dumps(payload,ensure_ascii=False,indent=1),encoding="utf8")


if __name__=="__main__":
    bundle()
    notebook()
