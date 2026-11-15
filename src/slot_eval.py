"""Validation variants of a trained slot-program run, with pair-bootstrap CIs.

Runs the model twice on val (frozen stage-2 only; slot margins only), then mixes
per twin pair in numpy: pairs whose ops are all in a route set get
slot_weight * slot margin + base_weight * base margin, others keep the base. A saved
run's val logits (e.g. stage 3) can replace the stage-2 base. Never reads test.

    uv run python -m src.slot_eval artefacts/hydra/debug_pair_slots_stage2/<run> \
        --features-dir <dir> --reference artefacts/hydra/debug_geometry_v1_s42/<run>
"""

import argparse
import json
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
import torch

from .data import ArraySource, canonicalized, load_split, pair_groups
from .features import cache_key
from .losses import TruthLoss
from .metrics import evaluate_scores
from .models.model import VisualReasoningModel
from .models.slot_program import OP_NAMES, parse_program
from .models.text_encoders import Vocab
from .training import Trainer
from .utils import get_device, resolve_path

FOCUS = ("counting", "comparison", "spatial_counting", "spatial", "spatial_negation")


def build(run, device):
    cfg = OmegaConf.load(run / "config.yaml")
    saved = json.loads((run / "vocab.json").read_text())
    vocab = Vocab([], lowercase=saved["lowercase"])
    vocab.itos, vocab.stoi = saved["itos"], {w: i for i, w in enumerate(saved["itos"])}
    text = instantiate(cfg.model.text_encoder, vocab_size=len(vocab))
    fusion = instantiate(cfg.model.fusion, in_channels=768, text_dim=text.out_dim, _convert_="all")
    model = VisualReasoningModel(text, fusion)
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    train_cfg = OmegaConf.create(OmegaConf.to_container(cfg.train, resolve=True))
    return cfg, vocab, model, Trainer(model, TruthLoss(), vocab, train_cfg, device)


def aligned_logits(run, ids):
    saved_ids, logits = np.load(run / "val_ids.npy"), np.load(run / "val_logits.npy")
    pos = {r: i for i, r in enumerate(saved_ids)}
    return logits[[pos[i] for i in ids]]


def paired_bootstrap(a, b, labels, pairs, repeats=2000, seed=0):
    """AUC of a, b and a - b over resampled twin pairs (pairs: [P, 2] row indices)."""
    rng = np.random.default_rng(seed)
    y = labels[pairs].reshape(-1)
    sa, sb = a[pairs].reshape(-1), b[pairs].reshape(-1)
    res = {"pairs": len(pairs), "auc": float(roc_auc_score(y, sa)), "ref_auc": float(roc_auc_score(y, sb))}
    da, db, dd = [], [], []
    for _ in range(repeats):
        c = rng.integers(len(pairs), size=len(pairs))
        rows = pairs[c].reshape(-1)
        x, z = roc_auc_score(labels[rows], a[rows]), roc_auc_score(labels[rows], b[rows])
        da.append(x), db.append(z), dd.append(x - z)
    res["auc_ci95"] = np.quantile(da, [.025, .975]).round(4).tolist()
    res["ref_auc_ci95"] = np.quantile(db, [.025, .975]).round(4).tolist()
    res["delta"] = res["auc"] - res["ref_auc"]
    res["delta_ci95"] = np.quantile(dd, [.025, .975]).round(4).tolist()
    return res


def mix(base, slot, pairs, pair_ops, route, base_weight, slot_weight=1.):
    out = base.copy()
    routed = np.array([all(OP_NAMES[o] in route for o in ops) for ops in pair_ops])
    a, b = pairs[routed, 0], pairs[routed, 1]
    margin = slot_weight * (slot[a] - slot[b]) + base_weight * (base[a] - base[b])
    out[a], out[b] = margin / 2, -margin / 2
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--features-dir", default="artefacts/features")
    p.add_argument("--reference", type=Path, action="append", default=[],
                   help="saved run whose val logits serve as an alternative base / reference")
    p.add_argument("--repeats", type=int, default=1000)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    run = resolve_path(args.run)
    device = get_device(args.device)
    cfg, vocab, model, trainer = build(run, device)
    split = canonicalized(load_split("val", resolve_path("data/val.csv")))
    base_dir = resolve_path(args.features_dir) / cache_key(cfg.model.image_encoder, cfg.data)
    names = json.loads((base_dir / "val.json").read_text())
    pos = {n: i for i, n in enumerate(names)}
    split.source = ArraySource(np.load(base_dir / "val.npy", mmap_mode="r"), None,
                               np.array([pos[n] for n in split.image_names]))
    head = model.fusion
    head.route_mask[:] = False
    stage2 = trainer.predict(split)
    head.route_mask[:] = True
    head.base_weight, head.slot_weight = 0., 1.
    slot = trainer.predict(split)
    pairs = pair_groups(split)
    ops = np.array([parse_program(c).op for c in split.captions])
    pair_ops = [tuple(ops[r]) for r in pairs]
    labels, cats = split.labels, split.categories

    bases = {"stage2": stage2}
    for ref in args.reference:
        bases[ref.parent.name] = aligned_logits(resolve_path(ref), split.ids)
    routes = {"count+cmp": ("exact", "more", "fewer", "same"), "exact": ("exact",),
              "cmp": ("more", "fewer", "same"), "all": OP_NAMES}
    variants = {}
    for bname, base in bases.items():
        variants[f"{bname}"] = base
        for rname, route in routes.items():
            for w in (0., 1.):
                variants[f"{bname}+slot[{rname}] base_w={w:g}"] = mix(base, slot, pairs, pair_ops, route, w)
    summary = {}
    for name, logits in variants.items():
        m = evaluate_scores(1 / (1 + np.exp(-logits.astype(np.float64))), labels, cats, split.pair_ids)
        summary[name] = {k: round(v, 4) for k, v in m.items()}
    focus_pairs = {c: pairs[cats[pairs[:, 0]] == c] for c in FOCUS}
    same = np.array([OP_NAMES[ops[a]] == "same" or OP_NAMES[ops[b]] == "same" for a, b in pairs])
    focus_pairs["comparison/as_many"] = pairs[(cats[pairs[:, 0]] == "comparison") & same]
    exact_n = np.array([parse_program(split.captions[a]).a == parse_program(split.captions[b]).a for a, b in pairs])
    focus_pairs["counting/number_twins"] = pairs[(cats[pairs[:, 0]] == "counting") & exact_n]
    focus_pairs["counting/filter_twins"] = pairs[(cats[pairs[:, 0]] == "counting") & ~exact_n]
    focus_pairs["overall"] = pairs
    bootstraps = {}
    for name in [n for n in variants if "[count+cmp]" in n or "[all]" in n]:
        bname = name.split("+")[0]
        bootstraps[name] = {c: paired_bootstrap(variants[name], bases[bname], labels, fp, args.repeats)
                            for c, fp in focus_pairs.items() if len(fp)}
    report = {"run": str(run), "metrics": summary, "paired_bootstrap_vs_base": bootstraps}
    (run / "slot_eval.json").write_text(json.dumps(report, indent=2))
    cols = ["auc", "pair_accuracy"] + [f"auc/{c}" for c in FOCUS]
    print("variant".ljust(48) + "".join(c.replace("auc/", "")[:12].rjust(13) for c in cols))
    for name, m in summary.items():
        print(name.ljust(48) + "".join(f"{m[c]:13.4f}" for c in cols))


if __name__ == "__main__":
    main()
