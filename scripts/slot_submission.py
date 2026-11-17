"""Slot margin + best stage-3 margin per twin pair, on val (check) and test (submission).

usage (from the worktree root):
  uv run python scripts/slot_submission.py SLOT_RUN BASE_RUN OUT_CSV [--features-dir DIR]
SLOT_RUN: trained pair_slots_* run (config.yaml, model.pt, vocab.json).
BASE_RUN: run with val_/test_ ids+logits (sub_geometric_full).
Pairs come from (image, category) groups: no labels, no pair_id.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.kaggle_slots import blend, cached_split, write_submission
from src.features import cache_key
from src.slot_eval import build
from src.utils import get_device, resolve_path

p = argparse.ArgumentParser()
p.add_argument("slot_run", type=Path)
p.add_argument("base_run", type=Path)
p.add_argument("out_csv", type=Path)
p.add_argument("--features-dir", default="artefacts/features")
p.add_argument("--skip-val", action="store_true", help="skip holdout diagnostics for a train+val refit")
p.add_argument("--reference-mode", choices=("barycenter", "marginal"), default="barycenter")
p.add_argument("--reference-exists", action="store_true")
p.add_argument("--exclusive-attributes", action="store_true")
p.add_argument("--comparison-mode", choices=("independent", "coupled"), default="independent")
p.add_argument("--base-weight", type=float, default=1.)
p.add_argument("--slot-weight", type=float, default=1.)
args = p.parse_args()

cfg, vocab, model, trainer = build(resolve_path(args.slot_run), get_device("auto"))
head = model.fusion
head.route_mask[:] = True
head.base_weight, head.slot_weight = 0.0, 1.0
head.slot_executor.reference_mode = args.reference_mode
head.slot_executor.require_reference_exists = args.reference_exists
head.slot_executor.exclusive_attributes = args.exclusive_attributes
head.slot_executor.comparison_mode = args.comparison_mode
feat_dir = resolve_path(args.features_dir) / cache_key(cfg.model.image_encoder, cfg.data)
data = resolve_path(cfg.paths.data_dir)


def run(name):
    split = cached_split(data, feat_dir, name)
    slot = trainer.predict(split)
    base_ids = np.load(args.base_run / f"{name}_ids.npy")
    base_logits = np.load(args.base_run / f"{name}_logits.npy")
    bpos = {r: i for i, r in enumerate(base_ids)}
    base = base_logits[[bpos[i] for i in split.ids]]
    return split, base, blend(split, base, slot, base_weight=args.base_weight, slot_weight=args.slot_weight)


if not args.skip_val:
    if cfg.data.train_on_val:
        p.error("use --skip-val for models fitted on train+val")
    val, val_base, val_mixed = run("val")
    print("val AUC base", round(roc_auc_score(val.labels, val_base), 4),
          "| base+slot", round(roc_auc_score(val.labels, val_mixed), 4))
    print({c: round(roc_auc_score(val.labels[val.categories == c], val_mixed[val.categories == c]), 4)
           for c in sorted(set(val.categories))})

test, _, test_mixed = run("test")
write_submission(data, test, test_mixed, args.out_csv)
print("wrote", args.out_csv, len(test))
