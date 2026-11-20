"""Assemble the private inputs and Kaggle notebook for near-occluded fragments."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from package_fragment_votes import DIR, ROOT, SETUP
from package_visible_fragments import RUN as RUNNER


def bundle():
    path=DIR/"clevr_rim.zip"
    code=("rim_fragments.py","rim_scene_unet.py","rim_eval.py","dense_counts.py","detail_counts.py","detail_eval.py","detail_union.py")
    weights={
        Path.home()/".cache/torch/hub/checkpoints/resnet18-f37072fd.pth":
            "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
        ROOT/"artefacts/research/convnext_tiny-983f1562.pth":
            "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d",
        DIR/"rim_inputs/efficientnet_v2_s-dd5fe13b.pth":
            "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74",
    }
    with zipfile.ZipFile(DIR/"clevr_rear.zip") as rear, \
         zipfile.ZipFile(DIR/"clevr_visible.zip") as visible, \
         zipfile.ZipFile(path.with_suffix(".tmp"),"w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as out:
        manifest={}
        def put(key,content):
            out.writestr(key,content)
            manifest[key]=hashlib.sha256(content).hexdigest()
        for name in visible.namelist():
            if name.endswith(".py") or name.startswith("priors/") or name in ("fragment_train.npz","rear_review_val.csv"):
                put(name,(ROOT/name).read_bytes() if name.endswith(".py") and (ROOT/name).is_file() else visible.read(name))
        for key in ("anchor_val.npz","anchor_test.npz"):
            put(key,rear.read(key))
        for name in code:
            key=f"scripts/research/{name}"
            put(key,(ROOT/key).read_bytes())
        for file,digest in weights.items():
            state=file.read_bytes()
            if hashlib.sha256(state).hexdigest()!=digest:
                raise ValueError(f"Weight checksum mismatch on {file}")
            put(file.name,state)
        focus=(DIR/"detail_run/reviewed_fragments_val.json").read_bytes()
        val=pd.read_csv(ROOT/"data/val.csv")
        if not {a["image"] for a in json.loads(focus)}<=set(val.image):
            raise ValueError("Reviewed regions must come from val")
        put("reviewed_fragments_val.json",focus)
        scale=np.array([240,160,240,160])
        with np.load(ROOT/"artefacts/research/scene_silhouette.npz") as z:
            assert np.array_equal(z["ids"],val.id)
            from package_fragment_votes import npz
            put("val_teacher_boxes.npz",npz(names=val.image.unique().astype(str),
                box=z["features"][...,8:12]*scale,keep=z["valid"]&(z["p"]>.90)))
        out.writestr("rim_manifest.json",json.dumps(manifest,indent=2))
    with zipfile.ZipFile(path.with_suffix(".tmp")) as z:
        assert z.testzip() is None
    path.with_suffix(".tmp").replace(path)
    print(path,path.stat().st_size)


MANAGE=RUNNER[:RUNNER.index("RESULT =")]+'''
RESULT = ROOT/"rim"
RESULT.mkdir(exist_ok=True)
ARMS = ("rgb","edge","unet","r50","r50native","r50real","r18","convnext","efficient")
SEEDS = (421,853)
EPOCHS=24
SCENE_EPOCHS=24
V="scripts.research.rim_fragments"
U="scripts.research.rim_scene_unet"
R="scripts.research.rear_candidate"
E="scripts.research.rim_eval"
from threading import Lock
failure_lock=Lock()
failures=[]
'''
start=RUNNER.index("def launch")
MANAGE+=RUNNER[start:].replace('step = "_".join(map(str,args)).replace("--","")',
                                'step = module.rsplit(".",1)[-1]+"_"+"_".join(map(str,args)).replace("--","")').replace('f"visible_{step}_failed.log"','f"rim_{step}_failed.log"')
MANAGE=MANAGE[:MANAGE.index("def worker")]+MANAGE[MANAGE.index("def monitor"):]
MANAGE+='''
def safe(gpu,module,*args):
    try:
        launch(gpu,module,*args)
        return True
    except Exception as exc:
        record=dict(job=module,args=list(args),error=str(exc))
        with failure_lock:
            failures.append(record)
            (RESULT/"failures.json").write_text(json.dumps(failures,indent=2))
        print("Сбой шага",record,flush=True)
        return False

def pair(module, tasks):
    with ThreadPoolExecutor(2) as pool:
        pending=[pool.submit(safe,gpu,module,*args) for gpu,args in enumerate(tasks)]
        return [job.result() for job in pending]

def extract(split):
    return all(pair(V,[("prepare","--split",split,"--shard",k) for k in (0,1)]))

def posterior(split):
    return all(pair(R,[("prepare","--split",split,"--shard",k) for k in (0,1)]))

def infer_local(arm,seed,split):
    return all(pair(V,[("infer","--arm",arm,"--seed",seed,"--split",split,"--shard",k) for k in (0,1)]))
'''

TRAIN='''
def train():
    prepared=extract("val")
    real_ready=extract("train")
    if prepared:
        for arm in ARMS:
            if arm=="r50real" and not real_ready:
                continue
            fit=pair(V,[("fit","--arm",arm,"--seed",s,"--epochs",EPOCHS) for s in SEEDS])
            for s,ok in zip(SEEDS,fit):
                if ok:
                    infer_local(arm,s,"val")
    fit=pair(U,[("fit","--seed",s,"--epochs",SCENE_EPOCHS) for s in SEEDS])
    if all(fit):
        pair(U,[("val","--seed",s) for s in SEEDS])
    else:
        for gpu,(s,ok) in enumerate(zip(SEEDS,fit)):
            if ok:
                safe(gpu,U,"val","--seed",s)
    posterior("val")

monitor(train,"train")
'''

EVAL='''
evaluation=safe(0,E)
report=json.loads((RESULT/"metrics.json").read_text()) if evaluation else {}
if report:
    print("ROC-AUC базы",report["base"]["auc"],"R50 count",report["r50_count"]["auc"])
    rows=[]
    diagnostics=[]
    for model,d in report["experiments"].items():
        rows.extend(dict(run=k,**value) for k,value in d["metrics"].items())
        if model.endswith("_both"):
            diagnostics.extend(dict(run=model,threshold=k,**value) for k,value in d["operating"].items())
    if rows:
        display(pd.DataFrame(rows).sort_values("auc",ascending=False).drop(columns="auc_by_category").head(30))
        display(pd.DataFrame(diagnostics))
    scene_rows=[]
    for seed,results in report["scene_unet"].items():
        scene_rows.append(dict(seed=seed,counts_and_comparison_auc=results["core_auc"],
                               replacement_auc=results["replacement"]["auc"],blend_auc=results["blend"]["auc"]))
    if scene_rows:
        display(pd.DataFrame(scene_rows))
print("Вариант для test",report.get("winner"))
'''

EXPORT='''
winner=report.get("winner")
def run_test():
    if not winner:
        return
    arm=winner[0]
    if arm=="scene_unet":
        if not all(pair(U,[("test","--seed",s) for s in SEEDS])):
            return
    else:
        if not extract("test") or not posterior("test"):
            return
        for seed in SEEDS:
            if not infer_local(arm,seed,"test"):
                return
    if safe(0,E,"--export"):
        name="unet_scene.csv" if arm=="scene_unet" else f"rim_{arm}.csv"
        shutil.copy2(RESULT/name,WORK/name)
        display(FileLink(name))

monitor(run_test,"test")
with zipfile.ZipFile(WORK/"rim_results.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for path in RESULT.glob("*"):
        if (path.is_file() and not path.name.endswith("_rgb.npy") and
            path.suffix in {".json",".npz",".pt",".log",".csv"}):
            z.write(path,path.name)
os.chdir(WORK)
display(FileLink("rim_results.zip"))
'''


def notebook():
    setup=SETUP.replace("clevr_fragments","clevr_rim").replace("fragments_manifest","rim_manifest")
    cells=[]
    content=[
        ("markdown","# Видимый фрагмент при сильном перекрытии\n\nПодключите данные конкурса и приватный clevr_rim. Выберите GPU T4 ×2. Обучение использует train, разметка val нужна для диагностики."),
        ("code",setup),("code",MANAGE),
        ("markdown","## Локальные модели и самостоятельный U-Net\n\nКаждая локальная модель видит кроп 40×40 в исходных пикселях и проверяет слабый пик R50. Кандидаты и границы передней фигуры задаются одинаково для всех вариантов. U-Net на полном кадре выделяет собственные маски, считает и сравнивает найденные фигуры. Каждый вариант обучается с двумя seed."),
        ("code",TRAIN),
        ("markdown","## Сравнение на исходных сценах\n\nСравниваем попадания по отмеченным фрагментам, совпадение атрибутов, ложные кандидаты и ROC-AUC поверх count_hidden. Валидируем подписи на всём val. Четыре пользовательские рамки не участвуют в обучении или выборе порога."),
        ("code",EVAL),
        ("markdown","## Сохранение результатов\n\nСкачайте rim_results.zip с метриками, весами, логами и прогнозами. Если выбран локальный вариант для test, ноутбук также сохраняет CSV."),
        ("code",EXPORT)]
    for i,(kind,source) in enumerate(content):
        cell=dict(id=f"rim-{i}",cell_type=kind,source=source.strip().splitlines(keepends=True),metadata={})
        if kind=="code":
            compile(source,cell["id"],"exec")
            cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    data=dict(cells=cells,nbformat=4,nbformat_minor=5,
              metadata=dict(kernelspec=dict(display_name="Python 3",language="python",name="python3")))
    (ROOT/"notebooks/15_rim_detection.ipynb").write_text(json.dumps(data,indent=1,ensure_ascii=False),encoding="utf8")


if __name__=="__main__":
    bundle()
    notebook()
