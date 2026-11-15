"""Slot margin + best stage-3 margin per twin pair, on val (check) and test (submission).

usage (from the worktree root):
  uv run python scripts/slot_submission.py SLOT_RUN BASE_RUN OUT_CSV [--features-dir DIR]
SLOT_RUN: trained pair_slots_* run (config.yaml, model.pt, vocab.json).
BASE_RUN: run with val_/test_ ids+logits (sub_geometric_full).
Pairs come from (image, category) groups: no labels, no pair_id.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data import ArraySource, canonicalized, load_split
from src.features import cache_key
from src.models.slot_program import OP_NAMES, parse_program
from src.slot_eval import build
from src.twins import find_twins
from src.utils import get_device, resolve_path

p = argparse.ArgumentParser()
p.add_argument("slot_run", type=Path)
p.add_argument("base_run", type=Path)
p.add_argument("out_csv", type=Path)
p.add_argument("--features-dir", default="/Users/platon/Documents/kaggle-visual-reasoning/artefacts/features")
args = p.parse_args()

cfg, vocab, model, trainer = build(resolve_path(args.slot_run), get_device("auto"))
head = model.fusion
head.route_mask[:] = True
head.base_weight, head.slot_weight = 0.0, 1.0
feat_dir = Path(args.features_dir) / cache_key(cfg.model.image_encoder, cfg.data)


def run(name):
    split = canonicalized(load_split(name, resolve_path(f"data/{name}.csv")))
    names = json.loads((feat_dir / f"{name}.json").read_text())
    pos = {n: i for i, n in enumerate(names)}
    split.source = ArraySource(np.load(feat_dir / f"{name}.npy", mmap_mode="r"), None,
                               np.array([pos[n] for n in split.image_names]))
    slot = trainer.predict(split)
    base_ids = np.load(args.base_run / f"{name}_ids.npy")
    base_logits = np.load(args.base_run / f"{name}_logits.npy")
    bpos = {r: i for i, r in enumerate(base_ids)}
    base = base_logits[[bpos[i] for i in split.ids]]
    pairs = find_twins([split.image_names[i] for i in split.img_idx], split.captions, split.categories)
    assert sorted(pairs.flatten().tolist()) == list(range(len(split))), "every row needs a twin"
    ops = np.array([parse_program(c).op for c in split.captions])
    routed = np.array([all(OP_NAMES[o] in OP_NAMES for o in ops[r]) for r in pairs])
    a, b = pairs[:, 0], pairs[:, 1]
    margin = np.where(routed, (slot[a] - slot[b]) + (base[a] - base[b]), base[a] - base[b])
    mixed = np.empty(len(split))
    mixed[a], mixed[b] = margin / 2, -margin / 2
    print(f"{name}: rows {len(split)}, pairs {len(pairs)}, routed {routed.mean():.4f}")
    return split, base, mixed


val, val_base, val_mixed = run("val")
print("val AUC base", round(roc_auc_score(val.labels, val_base), 4), "| base+slot", round(roc_auc_score(val.labels, val_mixed), 4))
cats = pd.Series(val.categories)
print({c: round(roc_auc_score(val.labels[cats == c], val_mixed[cats == c]), 4) for c in sorted(set(val.categories))})

test, _, test_mixed = run("test")
lookup = dict(zip(test.ids, test_mixed))
sample = pd.read_csv(resolve_path("data/sample_submission.csv"))
col = [c for c in sample.columns if c != "id"][0]
# twin-normalized probability: p_a = sigmoid(s_a - s_b) = sigmoid(margin)
logit = np.array([lookup[i] for i in sample.id]) * 2
sample[col] = 1 / (1 + np.exp(-logit))
sample.to_csv(args.out_csv, index=False)
print("wrote", args.out_csv, len(sample))
