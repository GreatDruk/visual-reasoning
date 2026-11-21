"""A bounded two-GPU queue shared by the single notebook and local launches."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import load_config


def run_suite(project, data, inputs, selected, work, output, seeds=None, gpus=(0, 1)):
    project = Path(project).resolve()
    work = Path(work).resolve()
    output = Path(output).resolve()
    work.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name in selected:
        path = project / "configs" / f"{name}.yaml"
        cfg = load_config(path)
        for seed in seeds or [cfg["train"]["seed"]]:
            jobs.append((path, seed))
    if not jobs:
        raise ValueError("Select at least one configuration")
    queues = [jobs[i :: len(gpus)] for i in range(len(gpus))]

    def worker(gpu, queue):
        results = []
        for config, seed in queue:
            log = work / f"{config.stem}_{seed}.log"
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu),
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false",
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2",
            )
            cmd = [
                sys.executable,
                "-m",
                "src.run",
                "--config",
                str(config),
                "--data",
                str(data),
                "--inputs",
                str(inputs),
                "--work",
                str(work),
                "--output",
                str(output),
                "--seed",
                str(seed),
            ]
            print("Starting", config.stem, seed, "GPU", gpu, flush=True)
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(
                    cmd, cwd=project, env=env, stdout=stream, stderr=subprocess.STDOUT
                )
            if result.returncode:
                tail = "\n".join(
                    log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
                )
                raise RuntimeError(f"{config.stem} failed; {log}\n{tail}")
            matches = sorted(
                (work / config.stem / str(seed)).glob("*/receipt.json"),
                key=lambda p: p.stat().st_mtime,
            )
            receipt = json.loads(matches[-1].read_text(encoding="utf-8"))
            results.append(receipt)
            print("Finished", config.stem, receipt["metrics"]["auc"], flush=True)
        return results

    with ThreadPoolExecutor(len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, queue) for gpu, queue in zip(gpus, queues)]
        rows = [row for f in futures for row in f.result()]
    dest = output / "results" / "suite.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows
