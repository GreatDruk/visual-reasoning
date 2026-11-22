"""Cold-start 057: two-GPU dependency scheduling, fixed recipes and test-only export."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config


def graph():
    """No job may read an upstream result until its producer has succeeded."""
    jobs = {
        "teacher": [],
        "native": ["teacher"],
        "nativehead": ["native"],
        "thaw": ["nativehead"],
        "resnet": ["teacher"],
        "convnext": ["teacher"],
        "crops": ["thaw"],
        "tokens0": ["crops"],
        "tokens1": ["crops"],
        "pool": ["tokens0", "tokens1"],
        "sigliphead": ["pool"],
        "testsiglip0": ["sigliphead"],
        "testsiglip1": ["sigliphead"],
        "siglipscore": ["testsiglip0", "testsiglip1"],
        "union421": ["resnet", "thaw"],
        "union853": ["union421"],
        "count53_421": ["resnet", "teacher"],
        "count53_853": ["resnet", "teacher"],
        "count54_421": ["convnext", "teacher"],
        "count54_853": ["convnext", "teacher"],
        "visual": ["convnext", "union421", "union853"],
        "counter0": ["count53_421", "count54_421"],
        "counter1": ["count53_853", "count54_853"],
        "export": ["visual", "counter0", "counter1", "siglipscore"],
    }
    # Unlock small prerequisites first; heavy ready jobs then fill both GPUs.
    priority = [
        "teacher",
        "native",
        "nativehead",
        "thaw",
        "resnet",
        "convnext",
        "crops",
        "tokens0",
        "tokens1",
        "pool",
        "sigliphead",
        "testsiglip0",
        "testsiglip1",
        "union421",
        "union853",
        "count54_421",
        "count54_853",
        "count53_421",
        "count53_853",
        "siglipscore",
        "visual",
        "counter0",
        "counter1",
        "export",
    ]
    return jobs, priority


def manifest(project, work):
    cfg = load_config(Path(project) / "configs/044_convnext.yaml")
    cfg["train"].update(seed=853, epochs=21, batch=64, calibrate=False)
    roles = {f"count{n}_{s}": f"count{n}_{s}.pt" for n in (53, 54) for s in (421, 853)}
    roles.update(union421="union421.pt", union853="union853.pt", donors="donors.npz")
    donor = Path(work) / "donors.npz"
    return dict(
        models=dict(convnext=dict(checkpoint="convnext.pt", config=cfg)),
        assets=roles,
        files={"donors.npz": hashlib.sha256(donor.read_bytes()).hexdigest()}
        if donor.exists()
        else {},
    )


def execute(stage, project, data, assets, work, *, limit=None, steps=None):
    from . import bootstrap, cold_siglip, cold_train
    from .runtime import setup

    policy = load_config(Path(project) / "configs/066_final.yaml")["pipeline"]
    _, device = setup(42)
    kwargs = dict(limit=limit, steps=steps)
    if stage == "teacher":
        return bootstrap.prepare(data, work, device, policy["teacher"], **kwargs)
    if stage == "native":
        return bootstrap.train_native(data, work, device, policy["native"], **kwargs)
    if stage == "nativehead":
        return cold_train.native_head(data, work, device, policy["nativehead"], **kwargs)
    if stage in ("resnet", "thaw"):
        return cold_train.historical(
            "r50_img" if stage == "resnet" else "thaw",
            data,
            work,
            assets,
            policy[stage],
            device,
            **kwargs,
        )
    if stage == "convnext":
        return cold_train.convnext(project, data, work, device, policy["convnext"], **kwargs)
    if stage == "crops":
        return cold_siglip.tables(data, work, device, limit=limit)
    if stage.startswith("tokens"):
        return cold_siglip.extract(
            project, data, work, device, policy["siglip"], "train", int(stage[-1]), limit=limit
        )
    if stage == "pool":
        return cold_siglip.pool(project, work, device, policy["siglip"], steps=steps)
    if stage == "sigliphead":
        return cold_siglip.scene_head(data, work, device, policy["siglip"], **kwargs)
    if stage.startswith("testsiglip"):
        return cold_siglip.extract(
            project, data, work, device, policy["siglip"], "test", int(stage[-1]), limit=limit
        )
    if stage == "siglipscore":
        return cold_siglip.infer(data, work, device, limit=limit)
    if stage.startswith("union"):
        return cold_train.union(
            project,
            data,
            work,
            manifest(project, work),
            device,
            policy["union"],
            int(stage[-3:]),
            **kwargs,
        )
    if stage.startswith("count5"):
        number, seed = stage.split("_")
        return cold_train.counter(
            project,
            data,
            work,
            manifest(project, work),
            device,
            f"0{number[5:]}_count",
            int(seed),
            policy["counter"]["batch"],
            **kwargs,
        )
    if stage in ("visual", "counter0", "counter1", "export"):
        return inference(stage, project, data, work, device, limit=limit)
    raise ValueError(f"Unknown final stage {stage}")


def inference(stage, project, data, work, device, *, limit=None):
    from .data import Scenes
    from .evidence import compose, hidden_classes, placement_prior
    from .foundations import aligned_score
    from .submit import counters, export_csv, visual, write_npz

    scenes = Scenes(data, "test", limit)
    scenes.prepare_images(
        Path(work) / "images", device if stage != "export" else torch.device("cpu")
    )
    m = manifest(project, work)
    if stage == "visual":
        parts = visual(scenes, Path(work) / "inputs", Path(work), m, device, Path(project))
    elif stage.startswith("counter"):
        seed = 421 if stage == "counter0" else 853
        parts = counters(
            scenes, Path(work), m, device, roles=[f"count53_{seed}", f"count54_{seed}"]
        )
    else:
        prior = placement_prior(Path(work) / "donors.npz", np.load(Path(work) / "background.npy"))
        parts = {"hidden": np.stack([hidden_classes(x.numpy(), prior) for x in scenes._rgb])}
        parts["sigft"] = aligned_score(work, "sigft_test.npz", scenes.frame)
        for job in ["visual", "counter0", "counter1"]:
            with np.load(Path(work) / f"test_{job}.npz") as z:
                if not np.array_equal(z["ids"], scenes.frame.id):
                    raise ValueError("Final component row IDs differ")
                parts.update({k: z[k].copy() for k in z.files if k != "ids"})
        cfg = load_config(Path(project) / "configs/057_quality.yaml")
        values, swaps = compose(scenes, parts, cfg["blend"])
        sample = pd.read_csv(Path(data) / "sample_submission.csv")
        if limit:
            sample = sample[sample.id.isin(scenes.frame.id)]
        export_csv(scenes.frame, sample, values["final"], Path(work) / "final_submission.csv")
        return dict(rows=len(scenes.frame), swapped_pairs=swaps, architecture="057_quality")
    write_npz(Path(work) / f"test_{stage}.npz", ids=scenes.frame.id.to_numpy(), **parts)


def prepare_inputs(assets, work):
    """Whitelist only external classification/dual-encoder pretrained weights."""
    assets = Path(assets).resolve()
    required = {"resnet50.pth", "convnext.pth", "siglip/config.json", "siglip/model.safetensors"}
    spec = json.loads((assets / "pretrained.json").read_text(encoding="utf-8"))
    if set(spec["files"]) != required:
        raise ValueError("Cold-start inputs must contain only the four declared pretrained files")
    for name in required:
        path = (assets / name).resolve()
        if not path.is_relative_to(assets) or not path.is_file():
            raise ValueError("Missing or unsafe pretrained input")
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != spec["files"][name]:
                raise ValueError(f"Pretrained checksum mismatch: {name}")
    root = Path(work) / "inputs"
    root.mkdir(parents=True, exist_ok=True)
    try:
        (root / "pretrained").symlink_to(assets, target_is_directory=True)
    except OSError:
        shutil.copytree(assets, root / "pretrained")
    return spec


def versions():
    result = {"python": sys.version.split()[0]}
    for package in [
        "torch",
        "torchvision",
        "transformers",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "opencv-python",
        "PyYAML",
    ]:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not installed under this distribution name"
    return result


def run(
    project,
    data,
    assets,
    work,
    output,
    *,
    limit=None,
    steps=None,
    started=None,
    runner="src.final",
    configuration="066_final",
    jobs=None,
    prepare=None,
):
    project, data, assets, work, output = [
        Path(x).resolve() for x in (project, data, assets, work, output)
    ]
    if "artefacts" in work.parts:
        raise ValueError("Final working directory must be disposable scratch")
    if (limit is None) != (steps is None):
        raise ValueError("Smoke requires both a scene limit and an update limit")
    if not limit and torch.cuda.device_count() != 2:
        raise ValueError("Final run requires two T4 GPUs")
    work.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="057_", dir=work))
    logs = scratch / "logs"
    logs.mkdir()
    started = time.monotonic() if started is None else started
    pretrained = (prepare or prepare_inputs)(assets, scratch)
    dependencies, priority = graph() if jobs is None else jobs
    pending = set(dependencies)
    done = set()
    active = {}
    records = []
    cancelled = threading.Event()
    processes = []
    workers = {}
    process_lock = threading.Lock()
    gpus = [0, 1] if torch.cuda.is_available() else [0]

    def launch(stage, gpu):
        begin = time.monotonic()
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(gpu),
            PYTHONHASHSEED="42",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            FINAL_RUN="1",
        )
        command = [
            sys.executable,
            "-u",
            "-m",
            runner,
            "--worker",
            "--project",
            str(project),
            "--data",
            str(data),
            "--assets",
            str(assets),
            "--work",
            str(scratch),
        ]
        if limit:
            command += ["--limit", str(limit), "--steps", str(steps)]
        print(f"STAGE {stage}: GPU {gpu}", flush=True)
        with (logs / f"{stage}.log").open("w", encoding="utf-8") as log:
            with process_lock:
                if gpu not in workers:
                    workers[gpu] = subprocess.Popen(
                        command,
                        cwd=project,
                        env=env,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        encoding="utf-8",
                        errors="replace",
                    )
                    processes.append(workers[gpu])
                process = workers[gpu]
                if cancelled.is_set():
                    process.terminate()
            process.stdin.write(json.dumps({"stage": stage}) + "\n")
            process.stdin.flush()
            completed = False
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith("STAGE_DONE "):
                    assert json.loads(line[len("STAGE_DONE ") :])["stage"] == stage
                    completed = True
                    break
                if line.startswith(("STAGE ", "EPOCH ", '{"stage":')):
                    print(f"[{stage}] {line.rstrip()}", flush=True)
                elif line.startswith("PROGRESS "):
                    event = json.loads(line[len("PROGRESS ") :])
                    if "loss" in event and event["done"] == event["total"]:
                        print(f"[{stage}] {event['stage']}: loss {event['loss']:.5f}", flush=True)
                    elif event["stage"].startswith("siglip/"):
                        print(
                            f"[{stage}] {event['stage']}: {event['done']}/{event['total']}",
                            flush=True,
                        )
            if not completed:
                raise RuntimeError(f"Final stage {stage} failed. Log: {logs / (stage + '.log')}")
        return dict(stage=stage, gpu=gpu, seconds=time.monotonic() - begin)

    with ThreadPoolExecutor(len(gpus)) as pool:
        while pending or active:
            used = {x[1] for x in active.values()}
            for gpu in gpus:
                if gpu in used:
                    continue
                ready = next(
                    (x for x in priority if x in pending and set(dependencies[x]) <= done), None
                )
                if ready is None:
                    continue
                pending.remove(ready)
                future = pool.submit(launch, ready, gpu)
                active[future] = (ready, gpu)
            if not active:
                raise RuntimeError("Final dependency graph is blocked")
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                stage, _ = active.pop(future)
                try:
                    records.append(future.result())
                except BaseException:
                    cancelled.set()
                    with process_lock:
                        for process in processes:
                            if process.poll() is None:
                                process.terminate()
                    raise
                done.add(stage)
                print(
                    f"FINISHED {stage}: {records[-1]['seconds']:.1f}s; total {time.monotonic() - started:.1f}s",
                    flush=True,
                )
    for process in processes:
        process.stdin.close()
        process.wait(timeout=30)
    seconds = time.monotonic() - started
    report = dict(
        architecture="057_quality",
        cold_start=runner == "src.final",
        smoke=bool(limit),
        seconds=seconds,
        within_50_minutes=seconds <= 3000,
        stages=records,
        recipe=load_config(project / "configs" / (configuration + ".yaml"))["pipeline"],
        pretrained=pretrained,
        training="official train only; no validation; "
        + (
            "original pretrained inputs"
            if runner == "src.final"
            else "frozen historical inputs; fresh ConvNeXt, union and count training"
        ),
        versions=versions(),
    )
    if not limit:
        output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scratch / "final_submission.csv", output / "final_submission.csv")
        (output / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    return report


def worker_main(executor=execute):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["project", "data", "assets", "work"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--stage")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--steps", type=int)
    args = vars(parser.parse_args())
    worker = args.pop("worker")
    stage = args.pop("stage")

    def perform(name):
        tick = time.monotonic()
        result = executor(stage=name, **args)
        reports = Path(args["work"]) / "reports"
        reports.mkdir(exist_ok=True)
        (reports / (name + ".json")).write_text(
            json.dumps(dict(stage=name, seconds=time.monotonic() - tick, result=result)),
            encoding="utf-8",
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("STAGE_DONE " + json.dumps(dict(stage=name)), flush=True)

    if worker:
        for line in sys.stdin:
            perform(json.loads(line)["stage"])
    else:
        if stage is None:
            parser.error("Select --worker or --stage")
        perform(stage)


if __name__ == "__main__":
    worker_main()
