"""Union (control) heads trained on a random detector grid phase per scene.

At test time swap_upstream_refit averages union scores over the four stride-4 grid
phases. Heads fit on phase 00 only see one phase. Val, heads fit on train only, mean of
four phases: plain fit 421 0.999586, 853 0.999744, 1001 0.999597, 2002 0.999579; random
phase fit 0.999703, 0.999779, 0.999721, 0.999709; seeds 421+853 0.999700 -> 0.999756.

    python scripts/research/union_phase_aug.py --root union_aug --fit trainval --seeds 421 853
writes <root>/out/control_<seed>.pt, drop-in replacements for initial/union_<seed>.pt.
"""
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.research import occlusion_verifier as v

PHASES = ((0, 0), (2, 0), (0, 2), (2, 2))


@torch.inference_mode()
def extract(split, device):
    """Merged r50_img + thaw tables for every grid phase, in CSV image order."""
    names = np.asarray(pd.read_csv(v.DATA / f"{split}.csv").image.unique(), dtype=str)
    models = [v.ResNetCenters(full=True), v.CenterDetector()]
    heads = []
    for model, key in zip(models, ("r50_img", "thaw")):
        model.load_state_dict(v.load(v.PRIORS / f"{key}.pt"))
        model.to(device, memory_format=torch.channels_last).eval()
        heads.append(v.model_head(key, device))
    matrix = v.projection().to(device)
    parts = {phase: [] for phase in PHASES}
    with ThreadPoolExecutor(3) as pool:
        for start in range(0, len(names), 128):
            images = np.stack(list(pool.map(v.read_rgb, [v.DATA / split / n for n in names[start:start+128]])))
            rgb = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float().div_(255)
            native = v.detector_outputs(models[1], heads[1], rgb, matrix)
            for dx, dy in PHASES:
                img = F.pad(rgb, (dx, 0, dy, 0), mode="replicate")[:, :, :v.HEIGHT, :v.WIDTH] if dx or dy else rgb
                r50 = v.detector_outputs(models[0], heads[0], img)
                shift = np.array([dx / v.WIDTH, dy / v.HEIGHT])
                r50[:, :, -6:-4] -= shift
                r50[:, :, -2:] -= 2 * shift
                rows = [v.merge(a, b) for a, b in zip(r50, native)]
                parts[dx, dy].append(tuple(map(np.stack, zip(*rows))))
            print(split, start + len(images), "/", len(names), flush=True)
    return names, {phase: tuple(np.concatenate(c) for c in zip(*stacks)) for phase, stacks in parts.items()}


def mixed(names, tables, rng):
    """One random phase per scene. The merged table has a fixed slot count per scene."""
    pick = rng.integers(0, len(PHASES), len(names))
    x, xy, valid = (a.copy() for a in tables[PHASES[0]])
    for i, phase in enumerate(PHASES[1:], 1):
        rows = pick == i
        for dest, src in zip((x, xy, valid), tables[phase]):
            dest[rows] = src[rows]
    return dict(names=names, x=x, xy=xy, valid=valid)


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--fit", choices=("train", "trainval"), default="trainval")
    cli.add_argument("--seeds", type=int, nargs="+", default=[421, 853])
    cli.add_argument("--epochs", type=int, default=12)
    args = cli.parse_args()
    _, device = v.setup(args.seeds[0])
    splits = ("train", "val") if args.fit == "trainval" else ("train",)
    data = v.DATA
    fit = args.root / "fit"
    (fit / "tables").mkdir(parents=True, exist_ok=True)
    frames, tables = [], []
    for split in splits:
        names, phases = extract(split, device)
        frames.append(pd.read_csv(data / f"{split}.csv"))
        tables.append(mixed(names, phases, np.random.default_rng(7)))
    frame = pd.concat(frames, ignore_index=True)
    if not np.array_equal(frame.image.unique(), np.concatenate([t["names"] for t in tables])):
        raise ValueError("Merged table order does not match the merged CSV")
    frame.to_csv(fit / "train.csv", index=False)
    np.savez(fit / "tables" / "train.npz", **{k: np.concatenate([t[k] for t in tables]) for k in tables[0]})
    v.DATA = v.CACHE = fit
    v.OUT = args.root / "out"
    v.OUT.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        v.train("control", seed, args.epochs, device)
        print("saved", v.OUT / f"control_{seed}.pt", flush=True)


if __name__ == "__main__":
    main()
