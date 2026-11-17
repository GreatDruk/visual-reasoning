"""Train the SigLIP2 geometry/slot ensemble and evaluate it or export a submission."""

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version

import numpy as np
from scipy.special import expit

from src.data import ArraySource, canonicalized, load_split, save_submission, subset
from src.features import cache_features, cache_key
from src.metrics import evaluate_scores
from src.models.slot_program import parse_program
from src.slot_eval import build, image_bootstrap
from src.slots_pretrain import DEFAULT_KEY
from src.train import instantiate
from src.twins import find_twins
from src.utils import get_device, set_seed


EXPERIMENTS = {
    "pair": "pair_best_stage1",
    "count": "pair_best_stage2",
    "geometry": "pair_geometric_comparison",
    "slots": "pair_slots_all",
    "finetune": "pair_slots_all",
}


def cached_split(data, cache, name):
    split = canonicalized(load_split(name, data / f"{name}.csv"))
    names = json.loads((cache / f"{name}.json").read_text())
    positions = {image: i for i, image in enumerate(names)}
    split.source = ArraySource(np.load(cache / f"{name}.npy", mmap_mode="r"),
                               indices=np.array([positions[n] for n in split.image_names]))
    return split


def predict(run, split, *, slots=False, marginal=False):
    _, _, model, trainer = build(run, get_device())
    if slots:
        head = model.fusion
        head.route_mask[:] = True
        head.base_weight, head.slot_weight = 0., 1.
        if marginal:
            head.slot_executor.reference_mode = "marginal"
            head.slot_executor.require_reference_exists = True
    return trainer.predict(split)


def blend(split, geometry, slots=None, *, base_weight=1., slot_weight=1.):
    """Combine twin margins; retain the geometry score where slot parsing fails."""
    result = geometry.astype(np.float64, copy=True)
    names = [split.image_names[i] for i in split.img_idx]
    pairs = find_twins(names, split.captions, split.categories)
    if not np.array_equal(np.sort(pairs.ravel()), np.arange(len(split))):
        raise ValueError(f"{split.name}: expected complete twin pairs for ensembling")
    a, b = pairs.T
    margin = result[a] - result[b]
    if slots is not None:
        parsed = np.array([parse_program(c) is not None for c in split.captions])
        slot_margin = slots[a].astype(np.float64) - slots[b]
        margin = np.where(parsed[a] & parsed[b], base_weight * margin + slot_weight * slot_margin, margin)
    result[a], result[b] = margin, -margin
    return expit(result)


def write_submission(data, split, probs, output):
    import polars as pl

    sample = pl.read_csv(data / "sample_submission.csv")
    target, = (column for column in sample.columns if column != "id")
    sample_ids = sample["id"].to_numpy()
    by_id = {row_id: i for i, row_id in enumerate(split.ids)}
    if (len(by_id) != len(split) or len(set(sample_ids)) != len(sample_ids)
            or set(by_id) != set(sample_ids)):
        raise ValueError("test and sample IDs must match exactly once")
    if probs.shape != (len(split),) or not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError("invalid submission probabilities")
    save_submission(sample_ids, probs[[by_id[i] for i in sample_ids]], output, target)


def frozen_encoder(weights):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
        cfg = compose(config_name="config", overrides=["experiment=pair_best_stage1"])
    cfg.model.image_encoder.weights_file = str(weights)
    key = cache_key(cfg.model.image_encoder, cfg.data)
    if key != DEFAULT_KEY:
        raise ValueError(f"Unexpected feature key {key}")
    return instantiate(cfg.model.image_encoder), cfg.data


def extract(args):
    encoder, data_cfg = frozen_encoder(args.weights)
    device = get_device()
    for task in args.extract:
        split, _, flip = task.partition("_flip")
        names = load_split(split, args.data / f"{split}.csv").image_names
        cache_features(encoder, args.data / split, names, args.work / "features" / DEFAULT_KEY / f"{task}.npy",
                       data_cfg.image_size, data_cfg.normalize, device, flip=bool(flip),
                       batch_size=args.extract_batch, amp=True)


def score_test_shard(args):
    rank, count, geometry_run, slot_run = args.test_shard
    split = canonicalized(load_split("test", args.data / "test.csv"))
    # Keep every caption of an image together, independent of the CSV row ordering.
    split = subset(split, split.img_idx % int(count) == int(rank))
    encoder, cfg = frozen_encoder(args.weights)
    array = cache_features(encoder, args.data / "test", split.image_names,
                           args.work / "features" / DEFAULT_KEY / f"test_{rank}.npy",
                           cfg.image_size, cfg.normalize, get_device(), batch_size=args.extract_batch, amp=True)
    del encoder
    split.source = ArraySource(array)
    geometry = predict(Path(geometry_run), split)
    slots = None if slot_run == "-" else predict(Path(slot_run), split, slots=True,
                                                marginal=args.variant == "marginal")
    np.savez(args.work / f"test_{rank}.npz", ids=split.ids, probs=blend(split, geometry, slots))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("final_submission.csv"))
    p.add_argument("--mode", choices=("validate", "final"), default="validate")
    p.add_argument("--variant", choices=("geometry", "baseline", "marginal", "finetune"), default="baseline")
    p.add_argument("--refit", action="store_true", help="fit on train+val in final mode")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpus", type=int, choices=(1, 2), default=2)
    p.add_argument("--batch", type=int, default=256, help="caption rows per supervised batch")
    p.add_argument("--extract-batch", type=int, default=128)
    p.add_argument("--slot-batch", type=int, default=64, help="images per pretraining step")
    p.add_argument("--slots", type=int, default=16)
    p.add_argument("--slot-steps", type=int, default=1600)
    p.add_argument("--epochs", type=int, nargs=5, default=[18, 18, 18, 18, 6],
                   metavar=("PAIR", "COUNT", "GEOMETRY", "SLOTS", "FINETUNE"))
    p.add_argument("--schedule-epochs", type=int, nargs=5, help="original cosine horizons when replaying best epochs")
    p.add_argument("--extract", nargs="+", help=argparse.SUPPRESS)
    p.add_argument("--score", nargs=3, metavar=("RUN", "SPLIT", "KIND"), help=argparse.SUPPRESS)
    p.add_argument("--test-shard", nargs=4, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.refit and args.mode != "final":
        p.error("--refit is for final mode")
    if min(args.batch, args.extract_batch, args.slot_batch, args.slot_steps, args.slots, *args.epochs) < 1 or args.batch % 2:
        p.error("positive sizes and an even supervised batch are required")
    schedule_epochs = args.schedule_epochs or args.epochs
    if any(stop > horizon for stop, horizon in zip(args.epochs, schedule_epochs)):
        p.error("schedule epochs must cover the requested training epochs")
    args.data, args.weights, args.work, args.output = (
        value.resolve() for value in (args.data, args.weights, args.work, args.output))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    set_seed(args.seed)
    cache = args.work / "features" / DEFAULT_KEY
    if args.extract:
        extract(args)
        return
    if args.test_shard:
        score_test_shard(args)
        return
    if args.score:
        run_path, split_name, kind = args.score
        split = cached_split(args.data, cache, split_name)
        logits = predict(Path(run_path), split, slots=kind != "geometry", marginal=kind == "marginal")
        np.save(args.work / f"{split_name}_{kind}.npy", logits)
        return

    import torch

    if torch.cuda.device_count() < args.gpus:
        p.error(f"select an accelerator with {args.gpus} GPU(s), or change --gpus")
    if not args.weights.is_file():
        p.error(f"missing offline SigLIP2 weights: {args.weights}")
    args.work.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    prefix = args.work.name
    gpu1 = args.gpus - 1
    settings = [f"--data={args.data}", f"--weights={args.weights}", f"--work={args.work}",
                f"--seed={args.seed}", f"--extract-batch={args.extract_batch}"]
    worker = ["-m", "scripts.kaggle_slots", *settings]

    def run(tag, gpu, command):
        launch = time.monotonic()
        env = os.environ.copy()
        # The parent uses no CUDA allocations; each worker owns one card and two CPU threads.
        devices = env.get("CUDA_VISIBLE_DEVICES")
        env["CUDA_VISIBLE_DEVICES"] = devices.split(",")[gpu] if devices is not None else str(gpu)
        env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   POLARS_MAX_THREADS="2", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED=str(args.seed))
        log_path = args.work / f"{tag}.log"
        print(f"[GPU {gpu}] {tag} ({log_path.name})", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            ret = subprocess.run([sys.executable, "-u", *command], env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
        if ret.returncode:
            print("\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-50:]), flush=True)
            ret.check_returncode()
        print(f"[GPU {gpu}] {tag} finished in {(time.monotonic() - launch) / 60:.2f} min", flush=True)

    def run_path(stage):
        return Path("artefacts/hydra") / f"{prefix}_{stage}" / "run"

    def fit(stage, gpu, previous=None):
        epochs = dict(zip(EXPERIMENTS, args.epochs))[stage]
        horizon = dict(zip(EXPERIMENTS, schedule_epochs))[stage]
        command = ["-m", "src.train", f"experiment={EXPERIMENTS[stage]}", f"seed={args.seed}",
                   f"name_exp={prefix}_{stage}", f"paths.output_dir={run_path(stage).as_posix()}",
                   f"paths.data_dir={json.dumps(args.data.as_posix())}",
                   f"paths.features_dir={json.dumps(cache.parent.as_posix())}",
                   f"model.image_encoder.weights_file={json.dumps(args.weights.as_posix())}",
                   f"train.batch_size={args.batch}", f"train.epochs={epochs}", f"train.scheduler.total_epochs={horizon}",
                   "train.time_budget_min=null", "++train.eval.paraphrases=false"]
        if previous:
            command.append(f"model.init_from={prefix}_{previous}")
        if stage in ("slots", "finetune"):
            checkpoint = args.work / "slots" / "slot_autoencoder.pt"
            command.append(f"model.fusion.slot_checkpoint={json.dumps(checkpoint.as_posix())}")
        if stage == "finetune":
            command += ["model.fusion.finetune_slots=true", "model.fusion.slot_recon_weight=0.1",
                        "train.optimizer.slot_encoder_lr=1e-5", "train.optimizer.lr=1e-4"]
        if args.mode == "final":
            command += ["train.early_stopping=null", "train.eval.holdout=false"]
        if args.refit:
            command.append("data.train_on_val=true")
        run(stage, gpu, command)
        receipt = json.loads((run_path(stage) / "results.json").read_text())
        if receipt["stop_reason"].startswith("nonfinite"):
            raise RuntimeError(f"{stage}: {receipt['stop_reason']}; see {args.work / (stage + '.log')}")

    def parallel(*jobs):
        # Serial fallback also makes this usable on a single-GPU Kaggle session.
        with ThreadPoolExecutor(args.gpus) as pool:
            pending = [pool.submit(job) for job in jobs]
            for task in pending:
                task.result()

    extract_originals = ["train"] + (["val"] if args.mode == "validate" or args.refit else [])
    extra = ["val_flip"] if args.refit else []
    parallel(lambda: run("features_originals", 0, worker + ["--extract", *extract_originals]),
             lambda: run("features_flips", gpu1, worker + ["--extract", "train_flip", *extra]))

    def pair_and_count():
        fit("pair", 0)
        fit("count", 0, "pair")

    def pretrain():
        splits = "train,train_flip" + (",val,val_flip" if args.refit else "")
        run("pretrain", gpu1, ["-m", "src.slots_pretrain", f"--features-dir={cache.parent}",
            f"--out={args.work / 'slots'}", f"--steps={args.slot_steps}", f"--slots={args.slots}",
            f"--batch-size={args.slot_batch}", f"--seed={args.seed}", "--decoder-hidden=256",
            "--decoder-layers=2", "--target-dim=128", f"--splits={splits}", "--require-caches",
            f"--warmup={min(200, max(1, args.slot_steps // 8))}", "--log-every=100"])

    def slots():
        fit("slots", gpu1, "count")
        if args.mode == "validate" or args.variant == "finetune":
            fit("finetune", gpu1, "slots")

    if args.mode == "final" and args.variant == "geometry":
        pair_and_count()
        fit("geometry", 0, "count")
    else:
        parallel(pair_and_count, pretrain)
        parallel(lambda: fit("geometry", 0, "count"), slots)
    report = {"seed": args.seed, "training_splits": extract_originals if args.refit else ["train"],
              "settings": {k: getattr(args, k) for k in ("batch", "slot_batch", "slots", "slot_steps", "epochs")},
              "versions": {pkg: version(pkg) for pkg in ("torch", "torchvision", "numpy", "polars",
                           "scikit-learn", "hydra-core", "open-clip-torch", "timm")},
              "stages": {stage: json.loads((run_path(stage) / "results.json").read_text())
                         for stage in EXPERIMENTS if (run_path(stage) / "results.json").is_file()}}
    report["settings"]["schedule_epochs"] = schedule_epochs
    if args.mode == "validate":
        variants = ("baseline", "marginal", "finetune")
        parallel(lambda: run("score_val_geometry", 0,
                             worker + ["--score", str(run_path("geometry")), "val", "geometry"]),
                 lambda: score_slots(run, gpu1, worker, run_path, "val", variants))
        split = cached_split(args.data, cache, "val")
        geometry = np.load(args.work / "val_geometry.npy")
        results = {candidate: blend(split, geometry, np.load(args.work / f"val_{candidate}.npy"))
                   for candidate in variants}
        results["geometry"] = blend(split, geometry)
        report["holdout"] = {candidate: evaluate_scores(probs, split.labels, split.categories, split.pair_ids)
                             for candidate, probs in results.items()}
        report["image_bootstrap_vs_baseline"] = {
            candidate: image_bootstrap(probs, results["baseline"], split.labels, split.img_idx, seed=args.seed)
            for candidate, probs in results.items() if candidate != "baseline"}
        report["epochs_for_refit"] = [report["stages"][stage]["best_epoch"] for stage in EXPERIMENTS]
        print(json.dumps(report["holdout"], indent=2), flush=True)
    else:
        stage = "finetune" if args.variant == "finetune" else "slots"
        slot_run = "-" if args.variant == "geometry" else str(run_path(stage))

        def score_shard(rank):
            run(f"score_test_{rank}", rank, worker + [f"--variant={args.variant}", "--test-shard",
                str(rank), str(args.gpus), str(run_path("geometry")), slot_run])

        parallel(*(lambda rank=rank: score_shard(rank) for rank in range(args.gpus)))
        split = load_split("test", args.data / "test.csv")
        lookup = {}
        for rank in range(args.gpus):
            with np.load(args.work / f"test_{rank}.npz") as shard:
                lookup.update(zip(shard["ids"], shard["probs"]))
        if set(lookup) != set(split.ids):
            raise ValueError("test shards do not cover all ids")
        write_submission(args.data, split, np.array([lookup[i] for i in split.ids]), args.output)
        report["submission"] = {"path": str(args.output), "rows": len(split), "variant": args.variant}
    report["total_min"] = round((time.monotonic() - started) / 60, 2)
    (args.work / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Total: {report['total_min']} min. Report: {args.work / 'report.json'}", flush=True)


def score_slots(run, gpu, worker, run_path, name, variants):
    for variant in variants:
        stage = "finetune" if variant == "finetune" else "slots"
        run(f"score_{name}_{variant}", gpu, worker + ["--score", str(run_path(stage)), name, variant])


if __name__ == "__main__":
    main()
