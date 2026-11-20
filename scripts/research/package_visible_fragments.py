"""Build the private offline Kaggle input and execution notebook."""
import hashlib
import io
import json
import zipfile

import pandas as pd

from package_fragment_votes import ROOT, DIR, SETUP


def review():
    root = ROOT/"artefacts/research"
    boxes = pd.read_csv(root/"overlap_review/visible_boxes.csv")
    # Rear row chosen from each val contact sheet after looking at the rendered scene.
    behind = {6:1, 13:1, 19:0, 23:0, 25:0, 31:0, 53:0, 56:0, 62:0, 72:1}
    fields = ["image","rear","shape","color","fx","fy","fw","fh","rx","ry","rw","rh"]

    def xywh(r):
        return [int(r.x0),int(r.y0),int(r.x1-r.x0),int(r.y1-r.y0)]

    items = []
    for key,k in behind.items():
        a,b = boxes[boxes.case.eq(key)].iloc[[k,1-k]].itertuples(index=False)
        items.append([a.image,1,a.shape,a.color,*xywh(b),*xywh(a)])
    checked = pd.read_csv(root/"overlap_controls/review_labels.csv")
    for r in checked[(checked.split=="val") & (checked.number==1)].itertuples(index=False):
        items.append([r.image,0,"",r.color,*xywh(r),*xywh(r)])
    result = pd.DataFrame(items,columns=fields).drop_duplicates(["image","fx","fy","rear"])
    val = pd.read_csv(ROOT/"data/val.csv")
    train = pd.read_csv(ROOT/"data/train.csv")
    assert set(result.image) <= set(val.image) and not set(result.image)&set(train.image)
    return result


def inputs():
    target = DIR/"clevr_visible.zip"
    with zipfile.ZipFile(DIR/"clevr_layers.zip") as old, zipfile.ZipFile(target.with_suffix(".tmp"),"w") as archive:
        manifest = {}

        def put(key, content):
            archive.writestr(key,content)
            manifest[key] = hashlib.sha256(content).hexdigest()

        keep = {"fragment_train.npz","anchor_val.npz","anchor_test.npz"}
        for key in old.namelist():
            if key.endswith(".py") or key.startswith("priors/") or key in keep:
                put(key,old.read(key))
        for name in ("visible_fragment_detector.py","visible_fragment_eval.py"):
            key = f"scripts/research/{name}"
            put(key,(ROOT/key).read_bytes())
        buf = io.StringIO()
        labeled = review()
        labeled.to_csv(buf,index=False)
        put("rear_review_val.csv",buf.getvalue().encode())
        archive.writestr("visible_manifest.json",json.dumps(manifest))
        print(labeled.rear.value_counts().to_dict(),"reviewed val regions")
    with zipfile.ZipFile(target.with_suffix(".tmp")) as archive:
        assert archive.testzip() is None
    target.with_suffix(".tmp").replace(target)


RUN = '''
assert all("T4" in torch.cuda.get_device_name(i) for i in range(2)), "Выберите GPU T4 ×2"
env.update(FRAGMENT_INPUT=str(package), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
RESULT = ROOT / "fragment_detector"
RESULT.mkdir(exist_ok=True)
ARMS = ("clean", "paste", "matched")
SEEDS = (421, 853)
EPOCHS = 48
DETECTOR = "scripts.research.visible_fragment_detector"
EVAL = "scripts.research.visible_fragment_eval"

def launch(gpu, module, *args):
    step = "_".join(map(str,args)).replace("--","")
    done = RESULT / f"{step}.done"
    if done.exists():
        return
    log = RESULT / f"{step}.log"
    cmd = [sys.executable, "-m", module, *map(str,args)]
    with log.open("a", buffering=1) as out, subprocess.Popen(cmd, cwd=ROOT, text=True,
            env={**env,"CUDA_VISIBLE_DEVICES":str(gpu)}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as job:
        for line in job.stdout:
            print(f"[GPU {gpu}] {line}", end="", flush=True)
            out.write(line)
        if job.wait():
            shutil.copy2(log, WORK/f"visible_{step}_failed.log")
            raise RuntimeError(f"{step} завершился с ошибкой. Лог сохранён в Output.")
    done.touch()

def parallel(jobs):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(fn,i,*args) for i,(fn,args) in enumerate(jobs)]
        for result in results:
            result.result()

def worker(gpu, *args):
    launch(gpu,DETECTOR,*args)

def monitor(job, stage):
    telemetry = RESULT / f"gpu_{stage}.csv"
    with telemetry.open("w",buffering=1) as out:
        query = ["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,temperature.gpu",
                 "--format=csv", "-l", "2"]
        try:
            process = subprocess.Popen(query,stdout=out,stderr=subprocess.STDOUT)
        except FileNotFoundError:
            process = None
        try:
            return job()
        finally:
            if process is not None:
                process.terminate()
                process.wait()
'''

PROBE = '''
# matched содержит самый дорогой путь подготовки. Меряем обе T4 одновременно.
monitor(lambda: parallel([(worker,("probe","--arm","matched","--seed",seed)) for seed in SEEDS]), "probe")
probes = [json.loads((RESULT/f"probe_matched_{seed}.json").read_text()) for seed in SEEDS]
results = pd.DataFrame([dict(gpu=gpu, **r) for gpu,p in enumerate(probes) for r in p["candidates"]])
display(results)
complete = results[results.vram_fraction.le(.91)].groupby("batch").filter(lambda g:len(g)==2)
throughput = complete.groupby("batch").crops_per_s.min()
if throughput.empty:
    raise RuntimeError("Нет размера батча, который поместился на обеих T4")
BATCH = int(throughput[throughput >= .98*throughput.max()].index.max())
print("Кропов за шаг на каждой T4",BATCH,"(нативный + парный кроп на каждую сцену)")
for gpu,p in enumerate(probes):
    print("Подготовка банка CPU и загрузка на GPU",gpu,p["startup"])
'''

TRAIN = '''
def stages():
    for arm in ARMS:
        parallel([(worker,("fit","--arm",arm,"--seed",seed,"--epochs",EPOCHS,"--batches",BATCH//2)) for seed in SEEDS])
monitor(stages,"fit")
'''

INFER = '''
def predict(split, arms):
    parallel([(worker,("prepare","--split",split,"--shard",shard)) for shard in (0,1)])
    for arm in arms:
        for seed in SEEDS:
            parallel([(worker,("predict","--arm",arm,"--seed",seed,"--split",split,"--shard",shard)) for shard in (0,1)])

monitor(lambda: predict("val",ARMS),"val")
launch(0,EVAL,"val")
report = json.loads((RESULT/"metrics.json").read_text())
cols = [dict(configuration="union_tv control",**report["baseline"])]
real = []
for arm, values in report["arms"].items():
    for seed, row in values.items():
        cols.append(dict(configuration=f"{arm} / {seed}", scenes=row["accepted_scenes"], **row["ensemble"]))
        if seed=="both":
            real.extend(dict(configuration=arm, threshold=q, **m) for q,m in row["detection"].items())
display(pd.DataFrame(cols).drop(columns="auc_by_category"))
display(pd.DataFrame(real))
print("Выбранная конфигурация",report["winner"])
'''

EXPORT = '''
winner = report["winner"]
if winner:
    monitor(lambda: predict("test",(winner,)),"test")
    launch(0,EVAL,"test")
    filename = f"rear_{winner}.csv"
    shutil.copy2(RESULT/filename,WORK/filename)
    display(FileLink(str(WORK/filename)))
with zipfile.ZipFile(WORK/"visible_results.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as out:
    for p in RESULT.iterdir():
        if p.suffix in {".csv",".json",".log",".pt",".npz"}:
            out.write(p,p.name)
display(FileLink(str(WORK/"visible_results.zip")))
'''


def notebook():
    setup = SETUP.replace("clevr_fragments","clevr_visible").replace("fragments_manifest","visible_manifest")
    items = [
        ("markdown","# Детектор видимого фрагмента задней фигуры\n\nПодключите данные конкурса и приватный `clevr_visible`, выберите GPU T4 ×2 и нажмите Run All. Обучение использует изображения train, оценка использует исходные сцены val."),
        ("code",setup), ("code",RUN),
        ("markdown","## Подбор общего батча\n\nОдновременно измеряем скорость обеих T4 на полной операции обучения с аугментацией. Учитываем выделенную и зарезервированную VRAM, время аугментации, время обучения и загрузку GPU."),
        ("code",PROBE),
        ("markdown","## Три конфигурации по два запуска\n\n`clean` видит исходные сцены. `paste` добавляет геометрические перекрытия из train. `matched` чаще ставит одноцветные фигуры за передними и оставляет видимыми от 4% до 36% маски. Архитектура, батч и число эпох общие."),
        ("code",TRAIN),
        ("markdown","## Исходные перекрытия и ROC-AUC\n\nПопадание означает, что пик маски оказался в размеченном фрагменте и запрос сделан по подходящей передней фигуре. Ложные срабатывания считаются в проверенных контрольных кропах. Delta ROC-AUC измеряется поверх train-only аналога `union_tv`."),
        ("code",INFER),("markdown","## Экспорт результатов\n\nCSV для Kaggle появится, если детектор улучшит результаты заданных проверок на val. Архив с метриками и логами сохраняется при любом исходе."),("code",EXPORT),
    ]
    cells = []
    for i,(kind,text) in enumerate(items):
        cell = dict(id=f"visible-{i}",cell_type=kind,source=text.strip().splitlines(keepends=True),metadata={})
        if kind=="code":
            compile(text,f"visible-{i}","exec")
            cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    nb = dict(cells=cells,nbformat=4,nbformat_minor=5,
              metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})
    (ROOT/"notebooks/13_visible_fragments.ipynb").write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding="utf8")


if __name__ == "__main__":
    inputs()
    notebook()

