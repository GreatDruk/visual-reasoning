"""Create the offline two-T4 compact refit package and notebook."""

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT/"artefacts/kaggle_suite"
NOTEBOOK = ROOT/"notebooks/17_compact_refit.ipynb"
ARCHIVE = SUITE/"clevr_refit.zip"
MODULES = ("compact_refit", "compact_replay", "curriculum_replay", "curriculum_data", "curriculum_train")
MODULES += ("rim_fragments", "rim_scene_unet", "rim_eval")


def donor_bank(source):
    with np.load(io.BytesIO(source.read("fragment_train.npz"))) as z:
        train = dict(z)
    frame = pd.read_csv(ROOT/"data/val.csv")
    names = frame.image.unique().astype(str)
    with np.load(ROOT/"artefacts/research/scene_silhouette.npz") as z:
        if not np.array_equal(z["ids"], frame.id):
            raise ValueError("Validation scene teacher IDs do not match")
        box = np.rint(z["features"][..., 8:12]*[240, 160, 240, 160]).astype(np.int16)
        x, y, w, h = box.transpose(2, 0, 1)
        keep = z["valid"] & (z["p"]>.84) & (w>=8) & (h>=8) & (x>=0) & (y>=0) & (x+w<=240) & (y+h<=160)
        sz, col, mat, shape = (np.where(z[k].max(-1)>=.85, z[k].argmax(-1), -1)
                               for k in ("size", "color", "material", "shape"))
        joint = np.where((mat>=0)&(shape>=0), 3*mat+shape, -1)
        labels = np.stack((sz, col, joint), axis=-1).astype(np.int8)
        mask = (255*z["features"][..., 47:147].clip(0, 1)).round().astype(np.uint8)
        val = dict(names=names, keep=keep, labels=labels, box=box, mask=mask.reshape(*keep.shape, 10, 10))
    if not np.array_equal(train["names"], pd.read_csv(ROOT/"data/train.csv").image.unique()):
        raise ValueError("Training scene teacher names do not match")
    if set(train["names"]) & set(names):
        raise ValueError("Train and validation scene names overlap")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{key: np.concatenate((train[key], val[key])) for key in train})
    return buffer.getvalue()


def bundle():
    frozen = ("scene_silhouette", "r50_img", "r50_img_head", "thaw", "thaw_head", "union_trainval")
    updates = {f"scripts/research/{name}.py" for name in MODULES}
    pending = ARCHIVE.with_suffix(".tmp")
    hashes = {}
    with ZipFile(SUITE/"clevr_curriculum.zip") as old, ZipFile(pending, "w", ZIP_DEFLATED, compresslevel=1) as new:
        def put(name, data):
            hashes[name] = hashlib.sha256(data).hexdigest()
            new.writestr(name, data)

        for name in sorted(set(old.namelist())-updates):
            if name.endswith(".py"):
                put(name, old.read(name))
        for name in sorted(updates):
            put(name, (ROOT/name).read_bytes())
        for path in (*(f"priors/{name}.pt" for name in frozen), "anchor_test.npz",
                     "old_train_0.npz", "old_train_1.npz"):
            put(path, old.read(path))
        for k in (0, 1):
            put(f"old_train_{k+2}.npz", (SUITE/"rim_run"/f"val_{k}.npz").read_bytes())
        for seed in (421, 853):
            name = f"scene_unet_{seed}.pt"
            put(f"priors/{name}", (SUITE/"rim_run"/name).read_bytes())
        put("fragment_train.npz", donor_bank(old))
        new.writestr("refit_manifest.json", json.dumps({"version": 2, "files": hashes}, indent=2))
    with ZipFile(pending) as check:
        if check.testzip():
            raise ValueError("Refit archive CRC check failed")
    pending.replace(ARCHIVE)


SETUP = r'''
import time

started = time.monotonic()
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
import torch
from IPython.display import FileLink, display

RUN_COLOR_EXPERIMENT = True
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
assert torch.cuda.device_count() == 2, "Settings → Accelerator → GPU T4 x2"
scratch = Path("/kaggle/temp")
ROOT = Path(tempfile.mkdtemp(prefix="clevr_refit_", dir=scratch if scratch.is_dir() else None))
LOGS = ROOT / "logs"
LOGS.mkdir()

def unpack(path, dest):
    with ZipFile(path) as source:
        for member in source.infolist():
            if not (dest / member.filename).resolve().is_relative_to(dest.resolve()):
                raise ValueError(member.filename)
        source.extractall(dest)

manifests = list(INPUT.rglob("refit_manifest.json"))
if not manifests:
    archives = list(INPUT.rglob("clevr_refit.zip"))
    if len(archives) != 1:
        raise FileNotFoundError("Подключите private dataset из clevr_refit.zip")
    unpack(archives[0], ROOT / "bundle")
    manifests = [ROOT / "bundle/refit_manifest.json"]
if len(manifests) != 1:
    raise ValueError("Подключите ровно одну версию clevr_refit")
package = manifests[0].parent
manifest = json.loads(manifests[0].read_text())
if manifest.get("version") != 2:
    raise ValueError("Обновите private dataset из новой версии clevr_refit.zip")
for name, expected in manifest["files"].items():
    with (package / name).open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
            raise ValueError(f"Повреждён файл {name}")

required = ("train.csv", "val.csv", "test.csv", "sample_submission.csv")
def datasets(parent):
    return [p.parent for p in parent.rglob("train.csv")
            if all((p.parent / name).is_file() for name in required)
            and all((p.parent / name).is_dir() for name in ("train", "val", "test"))]

data = datasets(INPUT)
if not data:
    extracted = ROOT / "data_zip"
    for i, path in enumerate(INPUT.rglob("*.zip")):
        with ZipFile(path) as archive:
            if set(required) <= {Path(name).name for name in archive.namelist()}:
                unpack(path, extracted / str(i))
    data = datasets(extracted)
if len(data) != 1:
    raise ValueError("Подключите данные соревнования")
DATA = data[0]
FIT = ROOT / "fit"
(FIT / "train").mkdir(parents=True)
frames = [pd.read_csv(DATA / f"{split}.csv") for split in ("train", "val")]
merged = pd.concat(frames, ignore_index=True)
if not merged.id.is_unique or not merged.label.isin([0, 1]).all():
    raise ValueError("Невалидные train/val")
with np.load(package / "fragment_train.npz") as bank:
    if not np.array_equal(bank["names"], merged.image.unique()):
        raise ValueError("Датасет не совпадает с банком обучающих фигур")
merged.to_csv(FIT / "train.csv", index=False)
for split, frame in zip(("train", "val"), frames):
    for name in frame.image.unique():
        source = DATA / split / name
        if not source.is_file():
            raise FileNotFoundError(source)
        (FIT / "train" / name).symlink_to(source)

env = dict(os.environ, CLEVR_PRIORS=str(package / "priors"), FRAGMENT_INPUT=str(package),
           PYTHONPATH=os.pathsep.join((str(package), str(package / "scripts/research"))),
           PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
seeds = (421, 853)
times = {"prepare": time.monotonic() - started}
print("GPU", [torch.cuda.get_device_name(i) for i in range(2)], "fit scenes", merged.image.nunique())

def run(module, args, gpu, tag, data_root):
    command = [sys.executable, "-m", "scripts.research." + module, *map(str, args)]
    with (LOGS / f"{tag}.log").open("w") as log, subprocess.Popen(
            command, cwd=ROOT, env={**env, "CUDA_VISIBLE_DEVICES": str(gpu), "CLEVR_DATA": str(data_root)},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            log.write(line)
            print(f"[{tag}] {line}", end="", flush=True)
        if process.wait():
            log.flush()
            shutil.copyfile(LOGS / f"{tag}.log", WORK / "refit_failed.log")
            raise RuntimeError(f"Ошибка {tag}. См. refit_failed.log")

def parallel(module, jobs, stage, data_root=DATA):
    tick = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(run, module, args, gpu, f"{stage}_{gpu}", data_root)
                 for gpu, args in enumerate(jobs)]
        for task in tasks:
            task.result()
    times[stage] = time.monotonic() - tick
    print(stage, round(times[stage], 1), "сек.", flush=True)
'''

FIT = r'''
flags = () if RUN_COLOR_EXPERIMENT else ("--skip-color",)
fit_jobs = [("fit", "--root", ROOT, "--seed", seed, "--require-cuda", *flags) for seed in seeds]
parallel("compact_refit", fit_jobs, "fit", FIT)
'''

PREDICT = r'''
jobs = [("infer", "--root", ROOT, "--seed", seed, "--shard", gpu, "--require-cuda", *flags)
        for gpu, seed in enumerate(seeds)]
parallel("compact_refit", jobs, "infer")
tick = time.monotonic()
run("compact_refit", ("variants", "--root", ROOT, "--output", WORK, *flags), "", "export", DATA)
times["export"] = time.monotonic() - tick
'''

REPORT = r'''
elapsed = time.monotonic() - started
fits = {path.stem: json.loads(path.read_text()) for path in sorted((ROOT / "curriculum").glob("*.json"))}
report = {"elapsed_minutes": elapsed / 60, "within_50_minutes": elapsed <= 50*60,
          "phase_seconds": times, "torch": torch.__version__, "cuda": torch.version.cuda,
          "gpu": [torch.cuda.get_device_name(i) for i in range(2)],
          "per_model_seconds": {name: info["elapsed_s"] for name, info in fits.items()},
          "scene_experiment": RUN_COLOR_EXPERIMENT, "scope": "U-Net refits and inference with frozen upstream anchor"}
(ROOT / "timing.json").write_text(json.dumps(report, indent=2))
with ZipFile(WORK / "refit_results.zip", "w", ZIP_DEFLATED, compresslevel=1) as archive:
    for path in [ROOT / "timing.json", *LOGS.glob("*.log"), *(ROOT / "curriculum").glob("*.json"),
                 *(ROOT / "curriculum").glob("*.pt")]:
        archive.write(path, path.relative_to(ROOT))
print(json.dumps(report, indent=2))
os.chdir(WORK)
files = ["local_only.csv", "refit_results.zip"]
if RUN_COLOR_EXPERIMENT:
    files.extend(["local_tv.csv", "compact_tv.csv"])
for name in files:
    display(FileLink(name))
'''


def notebook():
    def cell(text, kind, key):
        block = {"cell_type": kind, "id": key, "metadata": {},
                 "source": text.strip().splitlines(keepends=True)}
        if kind=="code":
            block.update(execution_count=None, outputs=[])
            compile("".join(block["source"]), key, "exec")
        return block

    intro = """# Компактный ансамбль на train + val

Подключите данные соревнования и private dataset из clevr_refit.zip. Выберите GPU T4 x2 и запустите Run All. Интернет для прогона не нужен.

GPU 0 обучает seed 421, GPU 1 обучает seed 853. Каждый процесс загружает обучающие данные один раз и использует их для двух веток.

local_tv.csv использует новый локальный U-Net и прежний полноэкранный. compact_tv.csv использует обе обученные на train + val модели. В local_only.csv цветовые сравнения остаются за базовым ансамблем. Чтобы сразу измерить короткий вариант, установите RUN_COLOR_EXPERIMENT = False.

Эксперимент использует сохранённый прогноз базового ансамбля. Замер охватывает обучение двух U-Net и их инференс. Полный пересчёт базового прогноза измеряется отдельно."""
    result = {"nbformat": 4, "nbformat_minor": 5,
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                           "language_info": {"name": "python"}, "accelerator": "GPU"},
              "cells": [cell(intro, "markdown", "overview"), cell(SETUP, "code", "prepare"),
                        cell(FIT, "code", "fit"), cell(PREDICT, "code", "predict"), cell(REPORT, "code", "report")]}
    NOTEBOOK.write_text(json.dumps(result, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")


if __name__=="__main__":
    bundle()
    notebook()
    print(ARCHIVE, ARCHIVE.stat().st_size)
    print(NOTEBOOK)
