"""Package the occlusion verifier research run and its fixed public-score baseline."""
from hashlib import sha256
from pathlib import Path
import io
import json
import zipfile

import numpy as np
import pandas as pd
from scipy.special import expit

from package_kaggle_suite import ROOT, DEST, SOURCES

RUN = DEST / "run_2026-11-18"
ZIP = DEST / "clevr_verifier.zip"
NB = ROOT / "notebooks/07_occlusion_verifier.ipynb"


def bundle():
    with zipfile.ZipFile(DEST / "clevr_kaggle_inputs.zip") as inputs, \
         zipfile.ZipFile(RUN / "clevr_weights.zip") as weights, \
         zipfile.ZipFile(RUN / "clevr_scores.zip") as scores, \
         zipfile.ZipFile(ZIP.with_suffix(".tmp"), "w") as out:
        hashes = {}
        def put(key, blob):
            out.writestr(key, blob)
            hashes[key] = sha256(blob).hexdigest()

        for name in ("model.safetensors", "config.json"):
            key = f"weights_in/siglip2_vision/{name}"
            put(key, inputs.read(key))
        for key in ("scene_silhouette.pt", "r50_img.pt", "r50_img_head.pt", "thaw.pt",
                    "thaw_head.pt", "siglip_pool.pt", "sigft_pca.npz"):
            put(f"priors/{key}", weights.read(key))
        for name in (*SOURCES, "scripts/research/occlusion_verifier.py",
                     *(p.relative_to(ROOT).as_posix() for p in (ROOT/"src").rglob("*.py"))):
            put(name, (ROOT / name).read_bytes())
        for split in ("val", "test"):
            ids = pd.read_csv(ROOT / "data" / f"{split}.csv", usecols=["id"]).id.to_numpy()
            def load_score(name):
                with np.load(io.BytesIO(scores.read(f"{name}_{split}.npz"))) as a:
                    assert np.array_equal(ids, a["ids"])
                    return a["score"]
            with np.load(RUN / f"audit_rescue_{split}.npz") as audit:
                assert np.array_equal(ids, audit["ids"])
                score = .5 * (load_score("r50_img") + load_score("sigft")) + .25 * (audit["boost"] - audit["control"])
            if split == "test":
                sent = pd.read_csv(RUN / "r50_sigft_occ.csv", float_precision="round_trip").set_index("id")
                assert np.array_equal(expit(score), sent.loc[ids, "label"])
            stream = io.BytesIO()
            np.savez(stream, ids=ids, score=score)
            put(f"priors/occ_{split}.npz", stream.getvalue())
        out.writestr("verifier_manifest.json", json.dumps(hashes))
    with zipfile.ZipFile(ZIP.with_suffix(".tmp")) as built:
        assert built.testzip() is None
    ZIP.with_suffix(".tmp").replace(ZIP)


SETUP = '''
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from IPython.display import FileLink, display
from scipy.special import expit
from sklearn.metrics import roc_auc_score

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
assert torch.cuda.device_count() == 2, "Выберите GPU T4 x2"
ROOT = Path(tempfile.mkdtemp(prefix="clevr_verify_"))
RESULT = ROOT / "verify_result"
RESULT.mkdir()
print("Рабочая папка", ROOT, flush=True)

def unpack(path, dest):
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not (dest / name).resolve().is_relative_to(dest.resolve()):
                raise ValueError(name)
        z.extractall(dest)

def find_data(parent):
    return [p.parent for p in parent.rglob("train.csv")
            if all((p.parent / n).is_file() for n in ("val.csv","test.csv","sample_submission.csv"))
            and all((p.parent / n).is_dir() for n in ("train","val","test"))]

locations = list(INPUT.rglob("verifier_manifest.json"))
if not locations:
    archives = list(INPUT.rglob("clevr_verifier.zip"))
    if len(archives) != 1:
        raise ValueError("Подключите один Private Dataset из clevr_verifier.zip")
    unpack(archives[0], ROOT / "package")
    locations = [ROOT / "package/verifier_manifest.json"]
if len(locations) != 1:
    raise ValueError("Подключено несколько версий весов")
package = locations[0].parent
for key, digest in json.loads(locations[0].read_text()).items():
    with (package / key).open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != digest:
            raise ValueError(key)
datasets = find_data(INPUT)
if not datasets:
    for j, path in enumerate(INPUT.rglob("*.zip")):
        with zipfile.ZipFile(path) as z:
            names = {Path(k).name for k in z.namelist()}
        if {"train.csv","val.csv","test.csv","sample_submission.csv"} <= names:
            unpack(path, ROOT / "inputs" / str(j))
    datasets = find_data(ROOT / "inputs")
if len(datasets) != 1:
    raise ValueError("Подключите данные соревнования")
DATA = datasets[0]
env = dict(os.environ, CLEVR_DATA=str(DATA), CLEVR_PRIORS=str(package/"priors"),
           CLEVR_WEIGHTS=str(package/"weights_in"), PYTHONPATH=str(package),
           PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
WORKER = [sys.executable, "-m", "scripts.research.occlusion_verifier"]
print(DATA, *[torch.cuda.get_device_name(i) for i in range(2)], sep=" | ", flush=True)

def run(gpu, *args):
    stage = "_".join(args).replace("--", "")
    done = ROOT / (stage + ".done")
    if done.exists():
        return
    logfile = RESULT / (stage + ".log")
    with logfile.open("a") as f, subprocess.Popen([*WORKER, *args], cwd=ROOT,
            env={**env, "CUDA_VISIBLE_DEVICES":str(gpu)}, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as job:
        for line in job.stdout:
            f.write(line)
            print(f"[{gpu}] {line}", end="", flush=True)
        if job.wait():
            shutil.copy2(logfile, WORK / "verifier_failed.log")
            raise RuntimeError(f"{stage} завершился с ошибкой; повторите эту ячейку")
    done.touch()

def both(left, right):
    with ThreadPoolExecutor(2) as pool:
        jobs = [pool.submit(run, gpu, *args) for gpu, args in enumerate((left, right))]
        for job in jobs:
            job.result()

def prepare(split, geometry_only=False):
    options = ("--geometry-only",) if geometry_only else ()
    both(*[("extract","--split",split,"--shard",str(i),*options) for i in range(2)])
    run(0,"collect","--split",split)

SEEDS = ("421","853")
def predict(split):
    for seed in SEEDS:
        both(*[("infer","--split",split,"--kind",kind,"--seed",seed) for kind in ("control","visual")])
'''
FIT = '''
start = time.monotonic()
prepare("train")
for seed in SEEDS:
    both(*[("train","--kind",kind,"--seed",seed) for kind in ("control","visual")])
prepare("val")
predict("val")
print("Минут до оценки на val", round((time.monotonic()-start)/60, 1))
'''
EVAL = '''
def read_scores(path, ids):
    with np.load(path) as data:
        if not np.array_equal(data["ids"], ids) or not np.isfinite(data["score"]).all():
            raise ValueError(path)
        return data["score"]

def predictions(split, frame):
    ids = frame.id.to_numpy()
    anchor = read_scores(package / "priors" / f"occ_{split}.npz",ids)
    results = {"occ": anchor}
    for kind in ("control","visual"):
        scores = [read_scores(RESULT/f"{kind}_{seed}_{split}.npz",ids) for seed in SEEDS]
        results[kind] = np.mean(scores,axis=0)
        for seed, values in zip(SEEDS,scores):
            results[f"occ_{kind}_{seed}"] = .75 * anchor + .25 * values
        results[f"occ_{kind}"] = .75 * anchor + .25 * results[kind]
    return results

frame = pd.read_csv(DATA / "val.csv")
values = predictions("val",frame)
label = frame.label.to_numpy()
def metrics(score):
    pos = score[label == 1]
    auc = roc_auc_score(label,score)
    return {"auc":float(auc),"wrong_pairs":int((pos<0).sum()),"tied_pairs":int((pos==0).sum()),
            "inversions":round((1-auc)*len(pos)*(len(score)-len(pos)),1)}

summary = {k:metrics(v) for k,v in values.items()}
if abs(summary["occ"]["auc"] - .9997095555555555) > 1e-9:
    raise ValueError("Не воспроизведён исходный val")
display(pd.DataFrame(summary).T.sort_values("auc",ascending=False))
scenes = pd.factorize(frame.image)[0]
rng = np.random.default_rng(4081)
deltas = []
for _ in range(800):
    scene_weights = rng.multinomial(scenes.max()+1, np.full(scenes.max()+1, 1/(scenes.max()+1)))
    w = scene_weights[scenes]
    scores = [roc_auc_score(label, values[k], sample_weight=w) for k in ("occ","occ_control","occ_visual")]
    deltas.append([scores[2]-scores[0], scores[2]-scores[1], scores[1]-scores[0]])
intervals = dict(zip(("visual_minus_occ","visual_minus_control","control_minus_occ"),
                     np.quantile(deltas,[.025,.975],axis=0).T.tolist()))
print("95% интервалы разности AUC по целым сценам", intervals)
def passes(kind):
    return (all(summary[f"occ_{kind}_{s}"]["auc"] > summary["occ"]["auc"] for s in SEEDS) and
            summary[f"occ_{kind}"]["wrong_pairs"] <= summary["occ"]["wrong_pairs"] and
            intervals[f"{kind}_minus_occ"][0] > 0)

# Two predeclared baseline contrasts, each with a one-sided 2.5% lower bound.
eligible = [kind for kind in ("control","visual") if passes(kind)]
winner = max(eligible, key=lambda k:summary[f"occ_{k}"]["auc"], default=None)
changes = {}
for kind in ("control","visual"):
    pos = frame.label.eq(1)
    diff = {}
    for category, group in frame[pos].groupby("category"):
        idx = group.index.to_numpy()
        old, new = values["occ"][idx], values[f"occ_{kind}"][idx]
        diff[category] = {"fixed":int(((old<0)&(new>0)).sum()),"broken":int(((old>0)&(new<0)).sum())}
    changes[kind] = diff
report = {"metrics":summary,"intervals":intervals,"winner":winner,"pair_changes":changes,
          "visual_beats_control":summary["occ_visual"]["auc"] > summary["occ_control"]["auc"],
          "seeds":list(SEEDS),"anchor_weight":.75}
(RESULT/"report.json").write_text(json.dumps(report,indent=2))
frame.assign(occ=values["occ"],control=values["occ_control"],visual=values["occ_visual"]).to_csv(
    RESULT/"val_predictions.csv",index=False)
print("Вариант для test", winner)
'''
FINAL = '''
if winner:
    prepare("test", winner == "control")
    both(*[("infer","--split","test","--kind",winner,"--seed",seed) for seed in SEEDS])
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    if not sample.id.is_unique or not test.id.is_unique or set(sample.id) != set(test.id):
        raise ValueError("Некорректные ID")
    target, = sample.columns.drop("id")
    anchor = read_scores(package / "priors/occ_test.npz", test.id.to_numpy())
    extra = [read_scores(RESULT/f"{winner}_{seed}_test.npz", test.id.to_numpy()) for seed in SEEDS]
    prob = pd.Series(expit(.75*anchor+.25*np.mean(extra,axis=0)), index=test.id)
    sample[target] = prob.reindex(sample.id).to_numpy()
    file = WORK / ("occ_crops.csv" if winner == "visual" else "occ_union.csv")
    sample.to_csv(file,index=False,float_format="%.17g")
    check = pd.read_csv(file,float_precision="round_trip")
    assert check.equals(sample) and check[target].between(0,1).all()
else:
    print("Критерий улучшения не выполнен. Сохранены метрики, прогнозы val и веса.")
shutil.make_archive(str(WORK/"verifier_results"),"zip",RESULT)
os.chdir(WORK)
display(FileLink("verifier_results.zip"))
if winner:
    display(FileLink(file.name))
'''


def notebook():
    cells = []
    for i, (kind, body) in enumerate([
        ("markdown", "# Проверка частично закрытых объектов\n\nПодключите данные соревнования и приватный clevr_verifier. Выберите две T4. Признаки сцен распределяются между GPU. В обучении один GPU ведёт контроль, второй — модель с кропами. Обе обучаются на train с двумя одинаковыми для сравнения seed."),
        ("code", SETUP), ("markdown","## Обучение и прогноз на val"), ("code", FIT),
        ("markdown", "## Сравнение с r50_sigft_occ\n\nФиксированный ансамбль содержит 75% текущей модели и 25% нового прогноза по логитам. Критерий перехода к test проверяет оба seed, число ошибок и парный bootstrap по изображениям."),
        ("code", EVAL), ("markdown","## Результаты"), ("code", FINAL)]):
        cell = dict(cell_type=kind,id=f"verify-{i}",metadata={},source=body.strip().splitlines(True))
        if kind == "code":
            compile(body, f"cell{i}", "exec")
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    NB.write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,cells=cells,metadata={
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}}),
        indent=1,ensure_ascii=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    notebook()
    bundle()
