"""Build an offline two-T4 Kaggle run for the fragment curriculum."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from package_fragment_votes import DIR, ROOT, SETUP
from package_rim import MANAGE

ARCHIVE=DIR/"clevr_curriculum.zip"
RUN=DIR/"rim_run"
NOTEBOOK=ROOT/"notebooks/16_fragment_curriculum.ipynb"
CODE=("curriculum_data.py","curriculum_train.py","curriculum_eval.py",
      "rim_eval.py","rim_fragments.py","rim_scene_unet.py","fragment_votes.py")
CHECKPOINTS={
    "resnet50-11ad3fa6.pth":Path.home()/".cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth",
    "resnet101-cd907fc2.pth":DIR/"rim_inputs/resnet101-cd907fc2.pth",
}
# Visually reviewed contest val regions with clear matte or mirror-like appearance.
VAL_MATERIAL={
    "80781.jpg":"metal", "20971.jpg":"rubber", "90931.jpg":"metal",
    "46987.jpg":"metal", "42269.jpg":"metal", "77083.jpg":"rubber",
    "69401.jpg":"rubber", "87292.jpg":"rubber", "28868.jpg":"rubber",
}


def reviewed_areas(old):
    import io
    rows=pd.read_csv(io.BytesIO(old.read("rear_review_val.csv"))).fillna("")
    frame=pd.read_csv(ROOT/"data/val.csv")
    teacher=ROOT/"artefacts/research/scene_silhouette.npz"
    lookup={name:i for i,name in enumerate(frame.image.unique())}
    colors=("gray","red","blue","green","brown","purple","cyan","yellow")
    attrs=("rubber","metal")
    with np.load(teacher) as description:
        if not np.array_equal(description["ids"],frame.id):
            raise ValueError("Mismatch in validation teacher rows")
        boxes=description["features"][...,8:12]*[240,160,240,160]
        colors_out=description["color"]
        mats=description["material"]
        shapes=description["shape"]
        valid=description["valid"]&(description["p"]>.90)
    records=[]
    def match(i,box):
        bs=boxes[i]
        cover=np.maximum(0,np.minimum(bs[:,:2]+bs[:,2:],box[:2]+box[2:])-np.maximum(bs[:,:2],box[:2])).prod(1)
        intersection=cover/(bs[:,2:].prod(1)+box[2:].prod()-cover).clip(1)
        intersection=np.where(valid[i],intersection,0.)
        j=int(intersection.argmax())
        return j,float(intersection[j])
    for c in rows.itertuples():
        i=lookup[c.image]
        r,m=match(i,np.array([c.rx,c.ry,c.rw,c.rh]))
        f,q=match(i,np.array([c.fx,c.fy,c.fw,c.fh]))
        material=VAL_MATERIAL.get(c.image,"") if c.rear else ""
        if not material and c.rear and m>=.42 and mats[i,r].max()>=.85 and shapes[i,r].max()>=.85:
            if ("b","s","c")[shapes[i,r].argmax()]==c.shape:
                material=attrs[mats[i,r].argmax()]
        same=int(colors[colors_out[i,f].argmax()]==c.color) if q>=.45 and colors_out[i,f].max()>=.85 else -1
        records.append((material,same))
    rows[["material","same_color"]]=records
    rows["highlight"]=-1
    return rows.to_csv(index=False).encode()


def pack():
    updates={f"scripts/research/{f}":(ROOT/"scripts/research"/f).read_bytes() for f in CODE}
    temp=ARCHIVE.with_suffix(".tmp")
    with zipfile.ZipFile(DIR/"clevr_rim.zip") as source,zipfile.ZipFile(
            temp,"w",zipfile.ZIP_DEFLATED,compresslevel=1) as out:
        manifest={}
        def put(key,value):
            out.writestr(key,value)
            manifest[key]=hashlib.sha256(value).hexdigest()
        for key in source.namelist():
            if key!="rim_manifest.json" and key not in updates:
                put(key,source.read(key))
        for k,state in updates.items():
            put(k,state)
        for name,path in CHECKPOINTS.items():
            state=path.read_bytes()
            prefix=name.rsplit("-",1)[1].split(".")[0]
            if not hashlib.sha256(state).hexdigest().startswith(prefix):
                raise ValueError(f"Incorrect classification weights: {name}")
            put(name,state)
        m=json.loads((RUN/"metrics.json").read_text(encoding="utf8"))
        cols=dict(base=m["base"],models={k:{s:v[s] for s in ("metrics","operating")} for k,v in m["experiments"].items()})
        put("old_rim_metrics.json",json.dumps(cols).encode())
        for i in (0,1):
            put(f"old_train_{i}.npz",(RUN/f"train_{i}.npz").read_bytes())
        put("curriculum_review.csv",reviewed_areas(source))
        out.writestr("curriculum_manifest.json",json.dumps(manifest,indent=2))
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:
            raise ValueError("Package integrity check failed")
    temp.replace(ARCHIVE)
    print(ARCHIVE,ARCHIVE.stat().st_size,flush=True)


def literal_config():
    syntax=ast.parse((ROOT/"scripts/research/curriculum_train.py").read_text(encoding="utf8"))
    return next(ast.literal_eval(d.value) for d in syntax.body if isinstance(d,ast.Assign) and
                any(isinstance(target,ast.Name) and target.id=="MODES" for target in d.targets))


DRIVER=MANAGE[:MANAGE.index("def extract(")].replace('RESULT = ROOT/"rim"','RESULT = ROOT/"curriculum"')
DRIVER=DRIVER.replace('rim_{step}_failed.log','curriculum_{step}_failed.log')
DRIVER+='''
from collections import defaultdict
import torchvision
MODEL="scripts.research.curriculum_train"
REPORT="scripts.research.curriculum_eval"
OUTPUT=WORK/"fragment_checkpoints"
OUTPUT.mkdir(exist_ok=True)
RUN_STEPS=1200
(RESULT/"runtime.json").write_text(json.dumps(dict(torch=torch.__version__,torchvision=torchvision.__version__,
    cuda=torch.version.cuda,gpu=[torch.cuda.get_device_name(i) for i in range(2)],steps=RUN_STEPS)))

def preserve(arm,mode):
    tag=f"{arm}_{mode}"
    for p in RESULT.glob(f"{tag}_*"):
        if p.suffix in (".pt",".json",".npz") and not (OUTPUT/p.name).exists():
            shutil.copy2(p,OUTPUT/p.name)

def extraction(split):
    (ROOT/"rim").mkdir(exist_ok=True)
    if not all(pair(V,[("prepare","--split",split,"--shard",k) for k in (0,1)])):
        return False
    for k in (0,1):
        for extension in (".npz","_rgb.npy"):
            src=ROOT/"rim"/f"{split}_{k}{extension}"
            dst=RESULT/src.name
            if src.exists():
                src.replace(dst)
            elif not dst.exists():
                return False
    return True

def posterior(split):
    return all(pair(R,[("prepare","--split",split,"--shard",k) for k in (0,1)]))

def jobs(arm,mode,stage,*args):
    return [(stage,"--arm",arm,"--mode",mode,"--seed",s,*args) for s in SEEDS]

def inferred(arm,mode,split,ready=None,extra=()):
    if ready is None:
        ready=[True,True]
    if arm=="scene":
        tasks=jobs(arm,mode,"infer","--split",split)
        if all(ready):
            return all(pair(MODEL,tasks))
        for gpu,ok in enumerate(ready):
            if ok:
                safe(gpu,MODEL,*tasks[gpu])
        return False
    result=True
    for seed,ok in zip(SEEDS,ready):
        if ok:
            tasks=[("infer","--arm",arm,"--mode",mode,"--seed",seed,
                    "--split",split,"--shard",shard,*extra) for shard in (0,1)]
            result=all(pair(MODEL,tasks)) and result
    return result and all(ready)

def run_batch_probes():
    mapping={}
    for arm,mode in MODES:
        kind="legacy" if mode=="legacy" else "staged"
        if (arm,kind) in mapping:
            continue
        if all(pair(MODEL,jobs(arm,kind,"probe"))):
            profiles=[json.loads((RESULT/f"probe_{arm}_{kind}_{s}.json").read_text()) for s in SEEDS]
            mapping[arm,kind]=min(o["batch"] for o in profiles)
    # Keep the same microbatch and total number of training draws in the depth ablation.
    if ("r50","staged") in mapping and ("r101","staged") in mapping:
        common=min(mapping["r50","staged"],mapping["r101","staged"])
        mapping["r50","staged"]=mapping["r101","staged"]=common
    print("Measured batches",mapping,flush=True)
    return mapping

def train():
    prepared=extraction("val")
    mapping=run_batch_probes()
    for arm,mode in MODES:
        kind="legacy" if mode=="legacy" else "staged"
        batch=mapping.get((arm,kind))
        if not batch:
            continue
        trained=pair(MODEL,jobs(arm,mode,"fit","--batch",batch,"--steps",RUN_STEPS))
        # A failed late checkpoint may still have a usable earlier selected state.
        ready=[(RESULT/f"{arm}_{mode}_{s}.pt").exists() for s in SEEDS]
        preserve(arm,mode)
        if prepared or arm=="scene":
            inferred(arm,mode,"val",ready)
            if mode=="legacy":
                for checkpoint in (240,1200):
                    if checkpoint<=RUN_STEPS:
                        inner_ready=[(RESULT/f"{arm}_{mode}_{s}_{checkpoint}.pt").exists() for s in SEEDS]
                        inferred(arm,mode,"val",inner_ready,("--checkpoint",checkpoint))
                        preserve(arm,f"{mode}{checkpoint}")
        preserve(arm,mode)
    if prepared:
        posterior("val")

monitor(train,"train")
'''

EVALUATE='''
complete=safe(0,REPORT)
decisions=json.loads((RESULT/"decisions.json").read_text()) if complete else {}
if decisions:
    print("Base ROC-AUC",decisions["base"]["auc"])
    for seed,item in decisions["folds"].items():
        print("Grouped folds",seed,item["auc"],item["auc_excluding_diagnostic"],item["delta_ci95"])
    print("Prospective",decisions["prospective"],"Selected",decisions["selected"])
'''

EXPORT='''
def requirements(plan):
    kind,*args=plan
    if kind=="base":
        return set()
    if kind=="scene":
        return {args[0]}
    if kind=="joint":
        return requirements(args[0])|requirements(args[1])
    return set(args[0])

def test():
    plan=decisions.get("selected")
    if not plan:
        return
    required=requirements(plan)
    local=[n for n in sorted(required) if not n.startswith("scene_")]
    if local and (not extraction("test") or not posterior("test")):
        return
    for name in sorted(required):
        arm,mode=name.split("_",1)
        skip=("--checkpoint",int(mode.removeprefix("legacy"))) if mode in ("legacy240","legacy1200") else ()
        if skip:
            mode="legacy"
        if not inferred(arm,mode,"test",extra=skip):
            return
    name="visible_unet" if plan[0]=="scene" else "visible_mix" if len(required)>1 else next(iter(required))
    if safe(0,REPORT,"--export","--filename",name):
        shutil.copy2(RESULT/f"{name}.csv",WORK/f"{name}.csv")
        display(FileLink(str(WORK/f"{name}.csv")))

try:
    monitor(test,"test")
finally:
    archive=WORK/"curriculum_results.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for path in sorted(RESULT.iterdir()):
            if path.is_file() and path.suffix in (".json",".npz",".pt",".log",".csv"):
                # Selected states duplicate the versioned checkpoints and can be reconstructed from JSON.
                if path.suffix==".pt" and len(path.stem.rsplit("_",2))==3 and not path.stem.rsplit("_",2)[-2].isdigit():
                    continue
                z.write(path,path.name)
    display(FileLink(str(archive)))
'''


def notebook():
    setup=SETUP.replace("clevr_fragments","clevr_curriculum").replace("fragments_manifest","curriculum_manifest")
    driver="MODES = "+repr(literal_config())+"\n"+DRIVER
    cells=[
        ("markdown","# Форма и материал почти закрытой фигуры\n\nДобавьте Competition Data и приватный clevr_curriculum. Настройте GPU T4 ×2 и выключите Интернет. Прогон сравнивает 18 конфигураций по двум seed. Результаты сохраняются и при падении отдельной ветки."),
        ("code",setup),("code",driver),
        ("markdown","## Исходные сцены val\n\nВыбор контрольных точек происходит на отдельных изображениях train. Правила ансамбля сравниваются на фолдах из целых сцен. Ручные четыре области открываются для диагностики после выбора."),
        ("code",EVALUATE),("code",EXPORT),
    ]
    content=[]
    for i,(kind,text) in enumerate(cells):
        c=dict(id=f"curriculum-{i}",cell_type=kind,metadata={},source=text.strip().splitlines(keepends=True))
        if kind=="code":
            compile(text,c["id"],"exec")
            c.update(outputs=[],execution_count=None)
        content.append(c)
    NOTEBOOK.write_text(json.dumps(dict(cells=content,nbformat=4,nbformat_minor=5,
        metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}}),
        indent=1,ensure_ascii=False),encoding="utf8")
    print(NOTEBOOK)


if __name__=="__main__":
    pack()
    notebook()

