"""A bounded two-GPU queue shared by the single notebook and local launches."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import load_config
from .runtime import immutable_inputs


def run_suite(
    project, data, inputs, selected, work, output, seeds=None, gpus=(0, 1), verified_inputs=False
):
    project = Path(project).resolve()
    work = Path(work).resolve()
    output = Path(output).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if not verified_inputs:
        immutable_inputs(inputs)
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
                PYTHONUNBUFFERED="1",
                HF_HUB_DISABLE_PROGRESS_BARS="1",
                TQDM_DISABLE="1",
            )
            cmd = [
                sys.executable,
                "-u",
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
                "--verified-inputs",
            ]
            print("Starting", config.stem, seed, "GPU", gpu, flush=True)
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    cmd,
                    cwd=project,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(f"[{config.stem} GPU{gpu}] {line.rstrip()}", flush=True)
                returncode = process.wait()
            saved_log = output / "results" / "logs" / log.name
            saved_log.parent.mkdir(parents=True, exist_ok=True)
            saved_log.write_text(
                log.read_text(encoding="utf-8", errors="replace"), encoding="utf-8"
            )
            if returncode:
                tail = "\n".join(
                    log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
                )
                print(f"FAILED {config.stem}: full log saved to {saved_log}\n{tail}", flush=True)
                results.append(dict(name=config.stem, seed=seed, status="failed", error=tail))
                continue
            matches = sorted(
                (work / config.stem / str(seed)).glob("*/receipt.json"),
                key=lambda p: p.stat().st_mtime,
            )
            receipt = json.loads(matches[-1].read_text(encoding="utf-8"))
            receipt["status"] = "complete"
            results.append(receipt)
            print("Finished", config.stem, receipt["metrics"]["auc"], flush=True)
        return results

    with ThreadPoolExecutor(len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, queue) for gpu, queue in zip(gpus, queues)]
        rows = []
        for future in as_completed(futures):
            rows.extend(future.result())
            dest = output / "results" / "suite.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    dest = output / "results" / "suite.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    failed = [r["name"] for r in rows if r.get("status") == "failed"]
    if failed:
        raise RuntimeError(
            f"Failed experiments: {failed}. Completed results and full logs are saved under {output / 'results'}"
        )
    return rows
