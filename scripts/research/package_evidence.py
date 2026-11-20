"""Build the offline two-T4 natural-overlap and caption-evidence experiment."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "artefacts/kaggle_suite"
RESEARCH = ROOT / "artefacts/research"
NOTEBOOK = ROOT / "notebooks/19_caption_evidence.ipynb"


def bundle(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_suffix(".tmp")
    paths = [
        *(p for directory in ("src", "scripts") for p in (ROOT/directory).rglob("*.py")
          if "__pycache__" not in p.parts)
    ]
    review = SUITE/"curriculum_run/inputs"
    prior = SUITE/"rim_smoke/verified_package"
    inputs = {
        "leader_val.npz": SUITE/"evidence_run/leader_val.npz",
        "inventory.json": SUITE/"evidence_run/inventory.json",
        "count_swap.csv": SUITE/"curriculum_run/count_swap.csv",
        "curriculum_review.csv": review/"curriculum_review.csv",
        "reviewed_fragments_val.json": review/"reviewed_fragments_val.json",
        "fragment_train.npz": prior/"fragment_train.npz",
        "priors/r50_img.pt": prior/"priors/r50_img.pt",
        "resnet50-11ad3fa6.pth": RESEARCH/"resnet50-11ad3fa6.pth",
    }
    of = SUITE/"clevr_verifier.zip"
    with ZipFile(of) as source:
        vision = {k: v for k, v in json.loads(source.read("verifier_manifest.json")).items()
                  if k.startswith("weights_in/siglip2_vision/")}
    if set(vision) != {"weights_in/siglip2_vision/config.json", "weights_in/siglip2_vision/model.safetensors"}:
        raise ValueError("Expected the exact two SigLIP vision files from the verifier run")
    inputs.update({p.relative_to(ROOT).as_posix(): p for p in paths})
    with ZipFile(staging, "w", ZIP_DEFLATED, compresslevel=1) as archive:
        hashes = {}
        for name, path in sorted(inputs.items()):
            with path.open("rb") as handle:
                hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
            archive.write(path, name)
        if not hashes["resnet50-11ad3fa6.pth"].startswith("11ad3fa6"):
            raise ValueError("Invalid torchvision ImageNet weights")
        archive.writestr("evidence_manifest.json", json.dumps(dict(files=hashes, siglip=vision), indent=2))
    with ZipFile(staging) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("The evidence input ZIP failed its CRC check")
    staging.replace(destination)
    print(destination, destination.stat().st_size)


INTRO = """# Проверка утверждений и видимых фрагментов

Подключите данные соревнования, private dataset из clevr_evidence.zip и прежний private dataset
с clevr_verifier.zip либо распакованными весами SigLIP. Выберите GPU T4 ×2, Internet Off, Run All.

На выходе evidence_results.zip со сравнениями на официальных val-сценах. По умолчанию обучение
использует train. Для эксперимента с подтверждёнными естественными перекрытиями добавьте отдельный
private dataset с natural_train.csv и масками. Без него запускаются два прогона синтетического контроля.
"""

SETUP = r'''
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from zipfile import ZIP_DEFLATED, ZipFile

import torch

RUN_MASK = True
RUN_TEST = False
VERIFIER_EPOCHS = 28
FOLD_EPOCHS = 6
MASK_EPOCHS = 12

START = time.monotonic()
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
SCRATCH = Path(tempfile.mkdtemp(prefix="clevr_evidence_"))
PACKAGE = SCRATCH / "package"
RESULT = SCRATCH / "result"
LOGS = RESULT / "logs"
LOGS.mkdir(parents=True)
TIMES = {}
assert torch.cuda.device_count() == 2, "Select GPU T4 x2"
print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(2)])

def only(paths, label):
    matches = sorted(set(paths))
    if len(matches) != 1:
        raise ValueError(f"Expected one {label}, found {matches}")
    return matches[0]

def materialize(archive, name, target, digest):
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(name) as source, target.open("wb") as out:
        shutil.copyfileobj(source, out, length=2**20)
    with target.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
            raise ValueError(f"Invalid archive member: {name}")

DATA = only((p.parent for p in INPUT.rglob("train.csv") if
             all((p.parent/q).exists() for q in ("train","val","test","val.csv","test.csv"))),
            "competition data root")
manifest_paths = list(INPUT.rglob("evidence_manifest.json"))
if manifest_paths:
    PACKAGE = only(manifest_paths, "evidence manifest").parent
    manifest = json.loads((PACKAGE/"evidence_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        with (PACKAGE/name).open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                raise ValueError(f"Invalid private input: {name}")
else:
    source = only(INPUT.rglob("clevr_evidence.zip"), "clevr_evidence.zip")
    with ZipFile(source) as archive:
        manifest = json.loads(archive.read("evidence_manifest.json"))
        for name, digest in manifest["files"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"Invalid path: {name}")
            materialize(archive, name, PACKAGE/name, digest)
VISION = SCRATCH/"weights_in/siglip2_vision"
for name, digest in manifest["siglip"].items():
    target = SCRATCH/name
    cached = [p for p in INPUT.rglob(target.name) if p.parent.name == "siglip2_vision"]
    if len(cached) == 1:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(cached[0])
        with target.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                raise ValueError(f"Invalid SigLIP file: {target.name}")
    else:
        prior = only(INPUT.rglob("clevr_verifier.zip"), "clevr_verifier.zip")
        with ZipFile(prior) as archive:
            materialize(archive, name, target, digest)

review = list(INPUT.rglob("natural_train.csv"))
REVIEW = only(review, "natural_train.csv") if review else None
review_val = list(INPUT.rglob("natural_val.csv"))
REVIEW_VAL = only(review_val, "natural_val.csv") if review_val else None
ENV = os.environ.copy()
ENV.update(PYTHONPATH=os.pathsep.join((str(PACKAGE), str(PACKAGE/"scripts/research"))),
           CLEVR_DATA=str(DATA), CLEVR_PRIORS=str(PACKAGE/"priors"), FRAGMENT_INPUT=str(PACKAGE),
           HF_HUB_OFFLINE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
TIMES["prepare_inputs"] = time.monotonic() - START
'''

RUN = r'''
def worker(gpu, name, module, arguments, out=RESULT):
    log = LOGS/f"{name}.log"
    env = dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else "")
    cmd = [sys.executable, "-m", f"scripts.research.{module}", *map(str, arguments),
           "--data", str(DATA), "--out", str(out)]
    with log.open("w") as handle:
        status = subprocess.run(cmd, env=env, cwd=SCRATCH, stdout=handle, stderr=subprocess.STDOUT)
    if status.returncode:
        print(log.read_text(errors="replace")[-6000:])
        raise RuntimeError(f"{name} failed with status {status.returncode}")

def parallel(stage, *jobs):
    start = time.monotonic()
    with ThreadPoolExecutor(len(jobs)) as pool:
        futures = [pool.submit(worker, *job) for job in jobs]
        for future in futures:
            future.result()
    TIMES[stage] = time.monotonic() - start
    print(f"{stage} {TIMES[stage]/60:.2f} min", flush=True)

def extract(split):
    parallel(f"encode_{split}", *((gpu, f"encode_{split}_{gpu}", "evidence_train",
            ["extract","--split",split,"--shard",gpu,"--r50",PACKAGE/"resnet50-11ad3fa6.pth",
             "--vision",VISION]) for gpu in range(2)))

def verify(split):
    parallel(f"predict_{split}", *((gpu, f"{arm}_{split}", "evidence_train",
             ["predict","--split",split,"--arm",arm]) for gpu,arm in enumerate(("control","hard"))))

mask_names = ("mask_control", "mask_natural" if REVIEW else "mask_control_853")

def mask_jobs(stage, split):
    jobs = []
    for gpu, name in enumerate(mask_names):
        seed = 853 if name == "mask_control_853" else 421
        command = [stage,"--split",split,"--epochs",MASK_EPOCHS,"--seed",seed]
        if stage == "fit" and name == "mask_natural":
            command += ["--review",REVIEW]
        jobs.append((gpu,f"{name}_{stage}_{split}","evidence_masks",command,RESULT/name))
    parallel(f"mask_{stage}_{split}", *jobs)

def evaluate(test=False):
    options = ["--test"] if test else (["--review",REVIEW_VAL] if REVIEW_VAL else [])
    worker(None, "evaluate_test" if test else "evaluate_val", "evidence_eval",
           ["--inputs", PACKAGE, *options])

try:
    extract("train")
    for start in (0,2):
        parallel(f"oof_{start}", *((gpu,f"oof_{start+gpu}","evidence_train",
            ["fold","--fold",start+gpu,"--epochs",FOLD_EPOCHS]) for gpu in range(2)))
    parallel("fit_verifier", *((gpu,f"fit_{arm}","evidence_train",
        ["fit","--arm",arm,"--epochs",VERIFIER_EPOCHS]) for gpu,arm in enumerate(("control","hard"))))
    extract("val")
    verify("val")
    if RUN_MASK:
        mask_jobs("fit","val")
        mask_jobs("predict","val")
    evaluate()
    if RUN_TEST:
        extract("test")
        verify("test")
        if RUN_MASK:
            mask_jobs("predict","test")
        evaluate(test=True)
finally:
    elapsed = time.monotonic() - START
    timing = dict(elapsed_s=elapsed, phases_s=TIMES, budget_s=3000,
                  comparison_within_budget=elapsed<=3000, natural_train_review=bool(REVIEW),
                  gpu=[torch.cuda.get_device_name(i) for i in range(2)], torch=torch.__version__)
    (RESULT/"timing.json").write_text(json.dumps(timing, indent=2))
    with ZipFile(WORK/"evidence_results.zip", "w", ZIP_DEFLATED, compresslevel=1) as archive:
        for p in sorted(RESULT.rglob("*")):
            if p.is_file() and p.suffix in (".log",".json",".npz",".csv"):
                archive.write(p, p.relative_to(RESULT))
    with ZipFile(WORK/"evidence_weights.zip", "w", ZIP_DEFLATED, compresslevel=1) as archive:
        for p in sorted(RESULT.rglob("*.pt")):
            archive.write(p, p.relative_to(RESULT))
    for p in RESULT.glob("evid*.csv"):
        shutil.copy2(p, WORK/p.name)
    print(json.dumps(timing, indent=2), flush=True)
'''


def notebook():
    cells = []
    for i, (kind, code) in enumerate((("markdown", INTRO), ("code", SETUP), ("code", RUN))):
        content = code.strip() + "\n"
        cell = dict(id=f"caption-evidence-{i}", cell_type=kind, metadata={},
                    source=content.splitlines(keepends=True))
        if kind == "code":
            compile(content, cell["id"], "exec")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    data = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python",
                                                          name="python3")), nbformat=4, nbformat_minor=5)
    NOTEBOOK.write_text(json.dumps(data, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    print(NOTEBOOK)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=SUITE/"clevr_evidence.zip")
    cli.add_argument("--notebook-only", action="store_true")
    options = cli.parse_args()
    notebook()
    if not options.notebook_only:
        bundle(options.output)


if __name__ == "__main__":
    main()
