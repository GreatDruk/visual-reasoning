"""Inference for the two U-Net branches refitted on train and validation scenes."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
import torch

from scripts.research import compact_replay as compact
from scripts.research import curriculum_replay as prior
from scripts.research import curriculum_train as train


RIM = prior.crops
BASE = prior.e.r
SEEDS = prior.e.SEEDS
LOCAL_TRAINING = (512, 800, 200)
SCENE_TRAINING = (128, 1200, 300)


def base_scores(frame):
    original = BASE.read(BASE.EXTRA/"anchor_test.npz", frame, "probability")
    return original, logit(original.clip(1e-15, 1-1e-15))


def color_mask(frame, anchor):
    return prior.scopes(frame)[0] & (np.abs(anchor) <= 3.)


def fit_models(root, seed, device, with_color):
    train.OUT = root/"curriculum"
    train.OUT.mkdir(parents=True, exist_ok=True)
    bank = train.data.Data(device, all_scenes=True, color_supervision=with_color)
    print("training bank", seed, bank.startup, flush=True)
    plan = [("unet", "boundary", LOCAL_TRAINING, True)]
    if with_color:
        plan.append(("scene", "staged", SCENE_TRAINING, False))
    for arm, mode, (batch, steps, interval), mixed in plan:
        np.random.seed(seed)
        if device.type=="cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        train.fit(bank, arm, mode, seed, batch, steps, fixed=True,
                  mix_stages=mixed, log_interval=interval)


@torch.inference_mode()
def predict_local(root, frame, shard, device):
    _, shape, material = prior.scopes(frame)
    names = set(frame.image[shape|material])
    prior.prepare(frame, root, shard, device, names)
    RIM.OUT = root/"replay"
    for seed in SEEDS:
        model = train.build("unet", "boundary", seed, device, pretrained=False)
        model.load_state_dict(RIM.v.load(root/"curriculum"/f"unet_boundary_{seed}.pt"))
        RIM.infer("unet_boundary", seed, "test", shard, device, model.eval(), scene_names=names)


@torch.inference_mode()
def predict_colors(root, frame, seed, device, legacy_options):
    _, anchor = base_scores(frame)
    required = color_mask(frame, anchor)
    if not required.any():
        return
    scenes = frame[frame.image.isin(frame.image[required])].reset_index(drop=True)
    names = scenes.image.unique()
    pixels = dict(zip(names, RIM.images("test", names)))
    for legacy in legacy_options:
        checkpoint = None
        if legacy:
            output, tag = root/"replay", "scene_unet"
            output.mkdir(parents=True, exist_ok=True)
            checkpoint = RIM.v.PRIORS/f"scene_unet_{seed}.pt"
        else:
            output, tag = root/"curriculum", "scene_staged"
        prior.whole.infer(seed, device, "test", None, tag=tag, output=output, frame=scenes,
                          checkpoint=checkpoint, rgb_cache=pixels)


def predict_color(root, frame, seed, device, legacy=False):
    predict_colors(root, frame, seed, device, (legacy,))


def color_scores(folder, tag, frame, required):
    output = []
    for seed in SEEDS:
        with np.load(folder/f"{tag}_{seed}_test.npz") as cache:
            values = pd.Series(cache["score"], index=cache["ids"])
            if not values.index.is_unique:
                raise ValueError(f"Duplicate score IDs for {tag} seed {seed}")
            output.append(values.reindex(frame.id).to_numpy())
    average = np.mean(output, axis=0)
    if not np.isfinite(average[required]).all():
        raise ValueError(f"Missing required color scores for {tag}")
    return average


def local_scores(root, frame, historical=False):
    output = root/"replay"
    BASE.OUT = output
    prior.e.OUT = output
    BASE.layer.rear.SCRATCH = output/"rear_cache"
    original, anchor = base_scores(frame)
    _, shape, material = prior.scopes(frame)
    color = color_mask(frame, anchor)
    tables = BASE.bank("test")
    one = BASE.by_image(prior.e.load("unet_boundary", "test"))
    shape_proposals = BASE.by_image(prior.e.load("unet_boundary_shape", "test")) if historical else one
    delta_shape, _, _ = BASE.branch(frame, anchor, shape_proposals, .92, tables)
    delta_material, _, _ = BASE.branch(frame, anchor, compact.recovered(one, tables), .65, tables)
    local = np.select((shape, material), (anchor+delta_shape, anchor+.65*delta_material), default=anchor)
    score = np.where(local == anchor, original, expit(local))
    return score, color, anchor


def add_color(score, color, anchor, detail):
    result = score.copy()
    result[color] = expit(.5*(anchor[color]+detail[color]))
    return result


def assemble(root, frame, historical=False, legacy_color=False):
    """Run only the branches used by shape_compact."""
    score, color, anchor = local_scores(root, frame, historical)
    if color.any():
        folder, tag = (root/"replay", "scene_unet") if historical or legacy_color else (root/"curriculum", "scene_staged")
        detail = color_scores(folder, tag, frame, color)
        score = add_color(score, color, anchor, detail)
    return score


def export(root, frame, output, historical=False, legacy_color=False):
    write_scores(frame, assemble(root, frame, historical, legacy_color), output)


def write_scores(frame, score, output):
    if not np.isfinite(score).all() or np.any((score<0)|(score>1)):
        raise ValueError("Invalid compact probabilities")
    index = pd.Index(frame.id)
    sample = pd.read_csv(RIM.v.DATA/"sample_submission.csv")
    if not index.is_unique or not sample.id.is_unique or set(index) != set(sample.id):
        raise ValueError("Unexpected submission IDs")
    sample["label"] = pd.Series(score, index=index).loc[sample.id].to_numpy()
    pending = output.with_suffix(".tmp")
    sample.to_csv(pending, index=False, float_format="%.17g")
    pd.testing.assert_frame_equal(sample, pd.read_csv(pending, float_precision="round_trip"))
    pending.replace(output)
    print(output, len(sample), "rows", flush=True)


def export_variants(root, frame, output, with_color):
    score, color, anchor = local_scores(root, frame)
    write_scores(frame, score, output/"local_only.csv")
    if not with_color:
        return
    variants = ((root/"replay", "scene_unet", "local_tv"), (root/"curriculum", "scene_staged", "compact_tv"))
    for folder, tag, filename in variants:
        detail = color_scores(folder, tag, frame, color) if color.any() else None
        values = add_color(score, color, anchor, detail) if color.any() else score
        write_scores(frame, values, output/f"{filename}.csv")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("fit", "infer", "local", "color", "export", "variants"))
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    cli.add_argument("--shard", type=int, choices=(0, 1), default=0)
    cli.add_argument("--historical", action="store_true")
    cli.add_argument("--legacy-color", action="store_true")
    cli.add_argument("--skip-color", action="store_true")
    cli.add_argument("--require-cuda", action="store_true")
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    _, device = RIM.v.setup(args.seed)
    if args.require_cuda and (device.type!="cuda" or torch.cuda.device_count()!=1):
        cli.error("Each GPU worker must see exactly one assigned CUDA device")
    if args.stage=="fit":
        fit_models(args.root, args.seed, device, not args.skip_color)
        return
    frame = pd.read_csv(RIM.v.DATA/"test.csv")
    if args.stage=="infer":
        predict_local(args.root, frame, args.shard, device)
        if not args.skip_color:
            predict_colors(args.root, frame, args.seed, device, (True, False))
    elif args.stage=="local":
        predict_local(args.root, frame, args.shard, device)
    elif args.stage=="color":
        predict_color(args.root, frame, args.seed, device, args.legacy_color)
    elif args.stage=="export":
        if args.output is None:
            cli.error("export requires --output")
        export(args.root, frame, args.output, args.historical, args.legacy_color)
    else:
        if args.output is None:
            cli.error("variants requires --output")
        export_variants(args.root, frame, args.output, not args.skip_color)


if __name__=="__main__":
    main()
