"""Build the notebook and its offline Kaggle Private Dataset archive."""

import base64
from hashlib import file_digest
from pathlib import Path
import json
import tempfile
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "artefacts/research"
DEST = ROOT / "artefacts/kaggle_suite"
NOTEBOOK = ROOT / "notebooks/06_full_suite.ipynb"
ARCHIVE = "clevr_kaggle_inputs.zip"
RESNET_SHA = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
SOURCES = (
    "scripts/pixel_reasoning.py",
    *("scripts/research/" + p + ".py" for p in (
        "kaggle_suite", "kaggle_suite_common", "kaggle_suite_detectors", "kaggle_suite_crops",
        "mc_scene3d", "native_centers", "resnet_centers_ablation", "learned_center_fusion",
        "center_proposals", "scene_ranking",
    )),
)
TEAM_WEIGHTS = ("scene_silhouette.pt", "scene_seams.pt", "scene_counts.pt",
                "native_epoch8.pt", "native_epoch16.pt", "native_natural_epoch21.pt")
COMPATIBILITY_FILES = ("scripts/research/kaggle_suite.py", "scripts/research/kaggle_suite_crops.py")


SETUP = '''
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

import torch

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
assert torch.cuda.device_count() == 2, "Settings → Accelerator → GPU T4 x2"
SCRATCH = Path("/kaggle/temp")
if not SCRATCH.is_dir() or not os.access(SCRATCH, os.W_OK):
    SCRATCH = Path("/tmp")
ROOT = Path(__import__("tempfile").mkdtemp(prefix="clevr_", dir=SCRATCH))
print(f"Scratch: {ROOT} ({shutil.disk_usage(ROOT).free / 2**30:.1f} GiB free)", flush=True)

for name in ("torch", "torchvision", "transformers", "safetensors", "catboost", "scipy", "pandas", "scikit-learn"):
    print(name, importlib.metadata.version(name))
from transformers import SiglipVisionModel

def extract_checked(archive, folder):
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (folder / member.filename).resolve()
            if not target.is_relative_to(folder.resolve()):
                raise ValueError(f"Unsafe ZIP path: {member.filename}")
        source.extractall(folder)

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

manifests = sorted(INPUT.rglob("clevr_manifest.json"))
if not manifests:
    zip_candidates = sorted(INPUT.rglob("clevr_kaggle_inputs.zip"))
    if len(zip_candidates) != 1:
        raise FileNotFoundError("Add the private clevr_kaggle_inputs dataset in Input")
    extract_checked(zip_candidates[0], ROOT)
    manifests = [ROOT / "clevr_manifest.json"]
if len(manifests) != 1:
    raise ValueError("Attach exactly one version of the private weights and source dataset")
package = manifests[0].parent
inventory = json.loads(manifests[0].read_text())
for relative, info in inventory["files"].items():
    source = package / relative
    if source.stat().st_size != info["size"] or digest(source) != info["sha256"]:
        raise ValueError(f"Input integrity check failed: {relative}")

dataset_files = ("train.csv", "val.csv", "test.csv", "sample_submission.csv")
def locate_dataset(directory):
    return [p.parent for p in sorted(directory.rglob("train.csv"))
            if all((p.parent / name).is_file() for name in dataset_files)
            and all((p.parent / name).is_dir() for name in ("train", "val", "test"))]

datasets = locate_dataset(INPUT)
if not datasets:
    scratch_data = ROOT / "_data_archives"
    scratch_data.mkdir()
    for k, candidate in enumerate(sorted(INPUT.rglob("*.zip"))):
        with zipfile.ZipFile(candidate) as source:
            names = {Path(n).name for n in source.namelist()}
        if set(dataset_files) <= names:
            folder = scratch_data / str(k)
            folder.mkdir()
            extract_checked(candidate, folder)
    datasets = locate_dataset(scratch_data)
if len(datasets) != 1:
    raise ValueError("Add one competition dataset containing train/val/test folders and four CSV files")
DATA = datasets[0]
print(f"Verified {len(inventory['files'])} source/weight files. Competition data: {DATA}", flush=True)
'''

PREFLIGHT = '''
import base64
import subprocess
import zlib

# Stage the checked sources in scratch. The attached dataset can be read-only.
worker_source = ROOT / "worker_source"
for relative in inventory["files"]:
    if relative.endswith(".py"):
        destination = worker_source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package / relative, destination)
for relative, encoded in compatibility_files.items():
    (worker_source / relative).write_bytes(zlib.decompress(base64.b64decode(encoded)))

env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
           MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1",
           CLEVR_DATA=str(DATA), CLEVR_WEIGHTS=str(package / "weights_in"))
env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(worker_source), env.get("PYTHONPATH"))))
worker = [sys.executable, "-m", "scripts.research.kaggle_suite"]
from concurrent.futures import ThreadPoolExecutor

def check(task, gpu):
    subprocess.run([*worker, "--task", task], cwd=ROOT,
                   env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)}, check=True)

with ThreadPoolExecutor(2) as pool:
    checks = [pool.submit(check, "pool_check", 0), pool.submit(check, "geometry_check", 1)]
    for result in checks:
        result.result()
print("[preflight] Both checks passed", flush=True)
'''

RUN = '''
started = time.monotonic()
cmd = [*worker, "--output", str(WORK)]
log = ROOT / "suite.log"
with log.open("a", encoding="utf-8") as stream:
    with subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as job:
        for line in job.stdout:
            stream.write(line)
            print(line, end="", flush=True)
        if job.wait() != 0:
            stream.flush()
            shutil.copy2(log, WORK / "clevr_failed.log")
            print(f"Scratch and completed tasks are in {ROOT}. Rerun this cell in the same session.")
            raise subprocess.CalledProcessError(job.returncode, cmd)
with zipfile.ZipFile(WORK / "clevr_reports.zip", "a", compression=zipfile.ZIP_DEFLATED) as zip_report:
    zip_report.write(log, "suite.log")
print(f"Wall time including ZIP creation: {(time.monotonic() - started)/60:.1f} min")
'''

RESULTS = '''
import pandas as pd
from IPython.display import FileLink, display

summary = pd.read_csv(ROOT / "result/reports/leaderboard.csv")
display(summary.style.format({"auc": "{:.8f}"}))
outputs = ["clevr_csv.zip", "clevr_reports.zip", "clevr_scores.zip", "clevr_weights.zip"]
os.chdir(WORK)
for name in outputs:
    path = WORK / name
    assert path.is_file()
    print(f"{name} ({path.stat().st_size / 2**20:.1f} MiB)")
    display(FileLink(name))
'''


def make_notebook():
    cells = []
    overrides = {name: base64.b64encode(zlib.compress((ROOT / name).read_bytes())).decode("ascii")
                 for name in COMPATIBILITY_FILES}
    content = [
        ("markdown", "# CLEVR · полный прогон на двух T4\n\n"
         "Подключите данные соревнования и приватный датасет из `clevr_kaggle_inputs.zip`. "
         "Выберите GPU T4 x2. В интерактивной сессии выполните ячейки подключения и проверки SigLIP и 3D. "
         "После успешной проверки запустите ячейку обучения. Если она упадёт, повторите только её "
         "в той же сессии Kaggle. Результаты и ссылки появятся в последней ячейке."),
        ("markdown", "## Подключение данных и весов"),
        ("code", SETUP),
        ("markdown", "## Проверка SigLIP и 3D перед обучением"),
        ("code", f"compatibility_files = {overrides!r}\n{PREFLIGHT}"),
        ("markdown", "## Обучение и инференс"),
        ("code", RUN),
        ("code", RESULTS),
    ]
    for i, (kind, body) in enumerate(content):
        cell = {"cell_type": kind, "id": f"clevr-suite-{i}", "metadata": {}, "source": body.strip().splitlines(True)}
        if kind == "code":
            compile(body.strip(), f"cell{i}", "exec")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}}
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def convert_vision(folder):
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch

    source = RESEARCH / "siglip2-so400m"
    physical = json.loads((source / "config.json").read_text())["vision_config"]
    physical.update(model_type="siglip_vision_model", architectures=["SiglipVisionModel"], dtype="float16",
                    torch_dtype="float16")
    (folder / "config.json").write_text(json.dumps(physical, indent=2), encoding="utf-8")
    with safe_open(source / "model.safetensors", framework="pt") as archive:
        vision = {name: archive.get_tensor(name).to(torch.float16) for name in archive.keys()
                  if name.startswith("vision_model.")}
    save_file(vision, folder / "model.safetensors", metadata={"format": "pt"})
    print(f"Packaged {len(vision)} SigLIP2 vision tensors")


def private_inputs():
    DEST.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=DEST) as temp:
        vision = Path(temp)
        convert_vision(vision)
        all_files = {path: ROOT / path for path in SOURCES}
        all_files.update({p.relative_to(ROOT).as_posix(): p for p in (ROOT / "src").rglob("*.py")})
        all_files.update({f"weights_in/{name}": RESEARCH / name for name in TEAM_WEIGHTS})
        all_files["weights_in/fusion.pt"] = RESEARCH / "local_centers_final/fusion.pt"
        resnet = RESEARCH / "resnet50-11ad3fa6.pth"
        if digest(resnet) != RESNET_SHA:
            raise ValueError("Official ImageNet ResNet-50 SHA256 mismatch")
        all_files["weights_in/resnet50-11ad3fa6.pth"] = resnet
        all_files.update({f"weights_in/siglip2_vision/{p.name}": p for p in vision.iterdir()})
        inventory = {"version": 1, "files": {}, "provenance": {
            "vision": "google/siglip2-so400m-patch16-256, vision-only FP16 conversion",
            "resnet": "torchvision ResNet50_Weights.IMAGENET1K_V2",
            "team": "Historical repository checkpoints fit from official competition training data",
        }}
        tmp_zip = DEST / "clevr_kaggle_inputs.building"
        with zipfile.ZipFile(tmp_zip, "w") as output:
            for name, path in sorted(all_files.items()):
                inventory["files"][name] = {"size": path.stat().st_size, "sha256": digest(path)}
                compression = zipfile.ZIP_DEFLATED if path.suffix in (".py", ".json") else zipfile.ZIP_STORED
                output.write(path, name, compress_type=compression)
            output.writestr("clevr_manifest.json", json.dumps(inventory, indent=2))
        with zipfile.ZipFile(tmp_zip) as archive:
            invalid = archive.testzip()
            if invalid or set(archive.namelist()) != set(all_files) | {"clevr_manifest.json"}:
                raise ValueError(f"Archive validation failed: {invalid}")
        tmp_zip.replace(DEST / ARCHIVE)
        (DEST / "input_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    make_notebook()
    private_inputs()
