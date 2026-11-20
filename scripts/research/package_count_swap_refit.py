"""Package a matched count_swap refit experiment with the existing union_tv anchor."""

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from scripts.research import package_compact_refit as compact


ROOT, SUITE = compact.ROOT, compact.SUITE
ARCHIVE = SUITE/"clevr_count_tv.zip"
NOTEBOOK = ROOT/"notebooks/18_count_swap_refit.ipynb"
PREVIOUS = SUITE/"curriculum_run/curriculum"
UPDATES = (*compact.MODULES, "count_swap_refit", "count_evidence", "dense_counts",
           "spatial_geometry", "spatial_replay", "compare_spatial", "hidden_count_prior",
           "swap_upstream_refit", "phase_refit", "postrun_visual_audit",
           "kaggle_suite_detectors", "kaggle_suite_crops")


def geometry_bank():
    parts = []
    for split, path in (("train", "teacher_train.npz"), ("val", "scene_silhouette.npz")):
        frame = pd.read_csv(ROOT/"data"/f"{split}.csv")
        names = frame.image.unique().astype(str)
        with np.load(ROOT/"artefacts/research"/path) as z:
            if split=="train":
                np.testing.assert_array_equal(z["names"],names)
                background=z["background"]
            else:
                np.testing.assert_array_equal(z["ids"],frame.id)
            parts.append(dict(names=names,x=z["features"][...,8:12].copy(),p=z["p"]*z["valid"],
                              **{key:z[key] for key in ("xy","size","color","material","shape")}))
    buffer=io.BytesIO()
    np.savez_compressed(buffer,background=background,**{
        key:np.concatenate([d[key] for d in parts]) for key in parts[0]})
    return buffer.getvalue()


def bundle():
    updates={f"scripts/research/{name}.py" for name in UPDATES}
    tmp=ARCHIVE.with_suffix(".tmp")
    hashes={}
    with ZipFile(compact.ARCHIVE) as old, \
         ZipFile(SUITE/"clevr_counters.zip") as baseline, \
         ZipFile(SUITE/"count_run/count_results.zip") as counts, \
         ZipFile(SUITE/"phase_run/phase_results.zip") as phase, \
         ZipFile(SUITE/"clevr_verifier.zip") as upstream, \
         ZipFile(tmp,"w",ZIP_DEFLATED,compresslevel=1) as new:
        def put(name,data):
            if name in hashes:
                raise ValueError(f"Duplicate entry {name}")
            hashes[name]=hashlib.sha256(data).hexdigest()
            new.writestr(name,data)

        for path in old.namelist():
            if path not in updates and path not in ("refit_manifest.json","anchor_test.npz"):
                put(path,old.read(path))
        for path in sorted(updates):
            put(path,(ROOT/path).read_bytes())
        put("union_tv_test.npz",baseline.read("baseline_test.npz"))
        put("geometry_trainval.npz",geometry_bank())
        upstream_files=json.loads(upstream.read("verifier_manifest.json"))
        required={key:value for key,value in upstream_files.items() if key.startswith("weights_in/siglip2_vision/")}
        if set(required)!=set(f"weights_in/siglip2_vision/{name}" for name in ("config.json","model.safetensors")):
            raise ValueError("The expected SigLIP model files were not found")
        put("siglip_inputs.json",json.dumps(required,indent=2).encode())
        for seed in (421,853):
            put(f"initial/union_{seed}.pt",phase.read(f"trainval/control_{seed}.pt"))
            for key in ("unet_noexact","r50_noexact","unet_boundary"):
                state=json.loads((PREVIOUS/f"{key}_{seed}.json").read_text())
                checkpoint=PREVIOUS/f"{key}_{seed}_{state['selected_step']}.pt"
                put(f"initial/{key}_{seed}.pt",checkpoint.read_bytes())
            put(f"initial/scene_unet_{seed}.pt",(SUITE/"rim_run"/f"scene_unet_{seed}.pt").read_bytes())
            put(f"initial/counter_{seed}.pt",counts.read(f"r50_{seed}/verify_result/counter.pt"))
        new.writestr("count_tv_manifest.json",json.dumps({"version":2,"files":hashes},indent=2))
    with ZipFile(tmp) as check:
        if check.testzip():
            raise ValueError("Archive CRC verification failed")
    tmp.replace(ARCHIVE)


SETUP=compact.SETUP.replace("refit_manifest.json","count_tv_manifest.json").replace(
    "clevr_refit.zip","clevr_count_tv.zip").replace("RUN_COLOR_EXPERIMENT = True","REFIT_UPSTREAM = True")
FIT=r'''
def tasks(stage, shard=False):
    return [(stage,"--root",ROOT,"--seed",seed,*(("--shard",gpu) if shard else ()))
            for gpu,seed in enumerate(seeds)]

if REFIT_UPSTREAM:
    tick=time.monotonic()
    paths=json.loads((package/"siglip_inputs.json").read_text())
    for relative,digest in paths.items():
        name=Path(relative).name
        hits=[p for p in INPUT.rglob(name) if p.parent.name=="siglip2_vision"]
        destination=ROOT/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        if len(hits)==1:
            destination.symlink_to(hits[0])
        else:
            holders=[path for pattern in ("clevr_verifier.zip","clevr_kaggle_inputs.zip")
                     for path in INPUT.rglob(pattern)]
            if len(holders)!=1:
                raise ValueError("Подключите clevr_verifier.zip или clevr_kaggle_inputs.zip для весов SigLIP")
            with ZipFile(holders[0]) as source, source.open(relative) as f, destination.open("wb") as out:
                shutil.copyfileobj(f,out,length=2**20)
        with destination.open("rb") as handle:
            if hashlib.file_digest(handle,"sha256").hexdigest()!=digest:
                raise ValueError("Неверная версия весов SigLIP")
    weights=ROOT/"weights_in"
    (weights/"scene_silhouette.pt").symlink_to(package/"priors/scene_silhouette.pt")
    for seed in seeds:
        (weights/f"union_{seed}.pt").symlink_to(package/"initial"/f"union_{seed}.pt")
    env["CLEVR_WEIGHTS"]=str(weights)
    times["upstream_inputs"]=time.monotonic()-tick

parallel("count_swap_refit",tasks("fit"),"fit",FIT)
'''
UPSTREAM=r'''
if REFIT_UPSTREAM:
    module="swap_upstream_refit"
    def stage(name,args,fit=False,gpu=0):
        tick=time.monotonic()
        run(module,args,gpu,name,FIT if fit else DATA)
        times[name]=time.monotonic()-tick
    stage("init_upstream",("setup",),True,"")
    stage("train_regions",("prepare","--split","train"),True,"")
    parallel(module,[("fit","--arm",name) for name in ("r50_img","thaw")],"upstream_fit",FIT)
    stage("test_regions",("prepare","--split","test"),False,"")
    tick=time.monotonic()
    with ThreadPoolExecutor(2) as pool:
        left=pool.submit(run,module,("detect","--split","train","--arm","thaw"),0,"train_native",FIT)
        right=pool.submit(run,module,("detect","--split","test","--arm","thaw"),1,"test_native",DATA)
        left.result()
        right.result()
    times["native_tables"]=time.monotonic()-tick
    for split,data_root in (("train",FIT),("test",DATA)):
        parallel(module,[("crops","--split",split,"--shard",gpu) for gpu in (0,1)],f"crops_{split}",data_root)
    stage("fit_pool",("pool",),True)
    for split,data_root in (("train",FIT),("test",DATA)):
        parallel(module,[("encode","--split",split,"--shard",gpu) for gpu in (0,1)],f"encode_{split}",data_root)
    stage("fit_caption_head",("fit_crop",),True)
    parallel(module,[("detect","--arm","r50_img","--split","test"),("score_crop",)],"base_predictions")
    parallel(module,[("union","--shard",gpu) for gpu in (0,1)],"union_tables")
    stage("union_predictions",("combine",))
    env["CLEVR_PRIORS"]=str(ROOT/"result/weights")
'''
PREDICT=r'''
MODULE = "count_swap_refit"
def cpu(stage,data_root=DATA,extra=()):
    tick=time.monotonic()
    run(MODULE,(stage,"--root",ROOT,*extra),"",stage,data_root)
    times[stage]=time.monotonic()-tick

cpu("geometry",FIT)
parallel(MODULE,tasks("count"),"count_predictions")
cpu("anchor")
parallel(MODULE,tasks("objects",shard=True),"local_and_color")
cpu("typed")
parallel(MODULE,tasks("spatial",shard=True),"spatial_predictions")
cpu("export",extra=("--output",WORK/"swap_tv.csv","--without-swap",WORK/"spatial_tv.csv"))
'''
REPORT=r'''
elapsed=time.monotonic()-started
report={"elapsed_minutes":elapsed/60,"within_50_minutes":elapsed<=50*60,
        "phase_seconds":times,"torch":torch.__version__,"cuda":torch.version.cuda,
        "gpu":[torch.cuda.get_device_name(i) for i in range(2)],
        "upstream_refit":REFIT_UPSTREAM}
(ROOT/"timing.json").write_text(json.dumps(report,indent=2))
with ZipFile(WORK/"swap_tv_results.zip","w",ZIP_DEFLATED,compresslevel=1) as archive:
    archive.write(WORK/"swap_tv.csv","swap_tv.csv")
    archive.write(WORK/"spatial_tv.csv","spatial_tv.csv")
    for path in (ROOT/"timing.json",ROOT/"spatial_geometry.pt",*LOGS.glob("*.log"),
                 *(ROOT/"curriculum").glob("*.json")):
        archive.write(path,path.relative_to(ROOT))
print(json.dumps(report,indent=2))
os.chdir(WORK)
for name in ("swap_tv.csv","spatial_tv.csv","swap_tv_results.zip"):
    display(FileLink(name))
'''


def notebook():
    overview="""# Переобучение веток count_swap

Подключите датасет соревнования, private dataset из clevr_count_tv.zip и ранее загруженный clevr_verifier.zip или clevr_kaggle_inputs.zip для весов SigLIP. Выберите GPU T4 x2 и выполните Run All.

Прогон продолжает обучение ResNet и U-Net на train + val, переобучает геометрию и рассчитывает заново прогноз базовых детекторов и SigLIP. Веса union сохраняются, поскольку уже обучены на train + val. Формула ансамбля прежняя.

swap_tv.csv включает перестановку ответов. spatial_tv.csv сохраняет оценки перед ней. REFIT_UPSTREAM = False запускает сокращённое сравнение на сохранённой базе."""
    cells=[]
    for i, (kind,text) in enumerate((("markdown",overview),("code",SETUP),("code",FIT),
                                    ("code",UPSTREAM),("code",PREDICT),("code",REPORT))):
        cell=dict(cell_type=kind,id=f"swap-tv-{i}",metadata={},source=text.strip().splitlines(keepends=True))
        if kind=="code":
            compile("".join(cell["source"]),cell["id"],"exec")
            cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    data=dict(nbformat=4,nbformat_minor=5,cells=cells,
              metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                        "language_info":{"name":"python"}})
    NOTEBOOK.write_text(json.dumps(data,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")


if __name__=="__main__":
    bundle()
    notebook()
    print(ARCHIVE,ARCHIVE.stat().st_size)
    print(NOTEBOOK)
