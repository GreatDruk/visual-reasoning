"""Inspect a pretrained slot autoencoder on validation images.

Writes mask overlays and a binding diagnostic: for val images whose TRUE caption is
"exactly N objects" (labels used for evaluation only), how many slots own at least
`--min-cells` argmax cells, versus N. One slot per object plus one background slot
gives owned == N + 1.

    uv run python -m src.slot_inspect artefacts/slots/k11_v1
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch

from .data import canonicalized, load_split
from .models.slot_program import parse_program, EXACT
from .models.slots import load_slot_autoencoder
from .slots_pretrain import DEFAULT_KEY, mask_overlays
from .utils import get_device, resolve_path


def object_counts(split):
    """image name -> true total object count, from true 'exactly N objects' captions."""
    out = {}
    for img, caption, label in zip(split.img_idx, split.captions, split.labels):
        p = parse_program(caption)
        if label == 1 and p is not None and p.op == EXACT and not p.a and not p.relation:
            out[split.image_names[img]] = p.number
    return out


@torch.no_grad()
def owned_slots(model, features, device, min_cells, source="alpha", batch=64):
    owned = []
    for start in range(0, len(features), batch):
        x = torch.from_numpy(np.array(features[start:start + batch])).to(device)
        out = model(x)
        masks = out[source]
        winners = masks.argmax(1)                                    # [B, N]
        cells = torch.stack([(winners == k).sum(1) for k in range(masks.shape[1])], 1)
        owned.append((cells >= min_cells).sum(1).cpu())
    return torch.cat(owned).numpy()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--features-dir", default="artefacts/features")
    p.add_argument("--key", default=DEFAULT_KEY)
    p.add_argument("--images", type=int, default=10)
    p.add_argument("--min-cells", type=int, default=2)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    device = get_device(args.device)
    run = resolve_path(args.run)
    model = load_slot_autoencoder(run / "slot_autoencoder.pt").to(device).eval()
    base = resolve_path(args.features_dir) / args.key
    features = np.load(base / "val.npy", mmap_mode="r")
    names = json.loads((base / "val.json").read_text())
    split = canonicalized(load_split("val", "data/val.csv"))
    counts = object_counts(split)
    chosen = [i for i, n in enumerate(names) if n in counts][:args.images]
    mask_overlays(model, features[chosen], [resolve_path("data/val") / names[i] for i in chosen],
                  run / "val_masks_counted.png", device)
    report = {"labelled_images": len(counts), "min_cells": args.min_cells,
              "overlay_images": {names[i]: counts[names[i]] for i in chosen}}
    rows = np.array([i for i, n in enumerate(names) if n in counts])
    truth = np.array([counts[names[i]] for i in rows])
    for source in ("alpha", "attn"):
        owned = owned_slots(model, features[rows], device, args.min_cells, source)
        diff = owned - truth
        report[source] = {"mean_owned": float(owned.mean()), "corr_with_true_count": float(np.corrcoef(owned, truth)[0, 1]),
                          "owned_minus_true": dict(sorted(collections.Counter(diff.tolist()).items())),
                          "share_owned_eq_true_plus_1": float(np.mean(diff == 1)),
                          "share_within_1_of_true_plus_1": float(np.mean(np.abs(diff - 1) <= 1))}
    (run / "inspect.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
