"""Fixed 057: frozen historical sources, freshly fitted newer branches."""

import json
import shutil
from pathlib import Path

import numpy as np

from . import cold_train
from .config import load_config
from .data import Scenes
from .final import inference, manifest, worker_main
from .runtime import setup


def graph():
    jobs = {
        "convnext": [],
        "unionprep": [],
        "union421": ["unionprep"],
        "union853": ["unionprep"],
        "count53_421": [],
        "count53_853": [],
        "count54_421": ["convnext"],
        "count54_853": ["convnext"],
        "visual": ["convnext", "union421", "union853"],
        "counter0": ["count53_421", "count54_421"],
        "counter1": ["count53_853", "count54_853"],
        "export": ["visual", "counter0", "counter1"],
    }
    order = [
        "convnext",
        "unionprep",
        "union421",
        "union853",
        "count53_421",
        "count54_421",
        "count53_853",
        "count54_853",
        "visual",
        "counter0",
        "counter1",
        "export",
    ]
    return jobs, order


def prepare_inputs(assets, work):
    assets, work = Path(assets), Path(work)
    try:
        (work / "inputs").symlink_to(assets / "inputs", target_is_directory=True)
    except OSError:
        shutil.copytree(assets / "inputs", work / "inputs")
    source = assets / "foundations"
    for name, output in [
        ("assets_donors.npz", "donors.npz"),
        ("assets_sigft_test.npz", "sigft_test.npz"),
    ]:
        shutil.copyfile(source / name, work / output)
    with np.load(source / "assets_background.npz") as z:
        np.save(work / "background.npy", z["background"])
    return dict(
        mode="historical train-only inputs",
        manifest=json.loads((assets / "snapshot057.json").read_text()),
    )


def execute(stage, project, data, assets, work, *, limit=None, steps=None):
    from .recovery import train_head
    from .recovery_bank import TableStore, prepare_bank

    _, device = setup(421)
    p = load_config(Path(project) / "configs/067_locked.yaml")["pipeline"]
    m = manifest(project, work, historical_warmup=bool(p["convnext"].get("historical_warmup")))
    if stage == "convnext":
        return cold_train.convnext(
            project, data, work, device, p["convnext"], limit=limit, steps=steps
        )
    if stage.startswith("count5"):
        number, seed = stage.split("_")
        return cold_train.counter(
            project,
            data,
            work,
            m,
            device,
            "0" + number[5:] + "_count",
            int(seed),
            96,
            limit=limit,
            steps=steps,
            historical_warmup=bool(p["counter"].get("historical_warmup")),
        )
    if stage.startswith("union"):
        cfg = load_config(Path(project) / "configs/045_union.yaml")
        scenes = Scenes(data, "train", limit)
        if stage == "unionprep":
            spec = json.loads((Path(work) / "inputs/manifest.json").read_text())
            path, report = prepare_bank(
                cfg,
                scenes,
                Path(work) / "inputs",
                spec,
                Path(work),
                m,
                Path(work),
                device,
                fixed_batch=128 if device.type == "cuda" else 2,
            )
            (Path(work) / "union_path.txt").write_text(str(path))
            return report
        seed = int(stage[-3:])
        cfg["train"]["seed"] = seed
        store = TableStore(Path((Path(work) / "union_path.txt").read_text()), scenes, device)
        head, report = train_head(
            cfg,
            scenes,
            store,
            Path(work) / "inputs",
            device,
            steps,
            fixed_batch=96 if device.type == "cuda" else 2,
        )
        cold_train.save(
            Path(work) / f"union{seed}.pt",
            dict(config=cfg, head={k: v.cpu() for k, v in head.state_dict().items()}),
        )
        return report
    return inference(
        stage,
        project,
        data,
        work,
        device,
        limit=limit,
        historical_warmup=bool(p["convnext"].get("historical_warmup")),
    )


def run(project, data, work, output, *, started=None, limit=None, steps=None):
    from .final import run as schedule

    return schedule(
        project,
        data,
        project,
        work,
        output,
        started=started,
        limit=limit,
        steps=steps,
        runner="src.locked",
        configuration="067_locked",
        jobs=graph(),
        prepare=prepare_inputs,
    )


if __name__ == "__main__":
    worker_main(execute)
