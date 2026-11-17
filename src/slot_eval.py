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
from .train import instantiate
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


def image_bootstrap(candidate, baseline, labels, img_idx, repeats=300, seed=42):
    """Bootstrap *images* rather than captions/twins (captions share scene content)."""
    rng = np.random.default_rng(seed)
    ids = np.unique(img_idx)
    blocks = [np.flatnonzero(img_idx == i) for i in ids]
    deltas = []
    for _ in range(repeats):
        chosen = rng.integers(len(ids), size=len(ids))
        rows = np.concatenate([blocks[j] for j in chosen])
        if len(np.unique(labels[rows])) != 2:
            continue
        deltas.append(roc_auc_score(labels[rows], candidate[rows]) -
                      roc_auc_score(labels[rows], baseline[rows]))
    return {"delta": float(roc_auc_score(labels, candidate) - roc_auc_score(labels, baseline)),
            "delta_ci95": np.quantile(deltas, [.025, .975]).round(6).tolist(),
            "sampled_images": len(ids), "iterations": len(deltas)}


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
    p.add_argument("--semantic-sweep", action="store_true",
                   help="test reference marginalisation, reference existence and exclusive attributes on val")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    run = resolve_path(args.run)
    device = get_device(args.device)
    cfg, vocab, model, trainer = build(run, device)
    split = canonicalized(load_split("val", cfg.data.val_csv))
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
    programs = [parse_program(c) for c in split.captions]
    if any(program is None for program in programs):
        raise ValueError("semantic sweep requires supported canonical validation captions")
    ops = np.array([program.op for program in programs])
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
    if args.semantic_sweep:
        # Keep the pretrained checkpoint fixed. Only the executor's interpretation
        # of the SAME per-slot predictions changes in this read-only diagnostic.
        semantic_modes = (
            ("ref_exists", "barycenter", True, False),
            ("marginal", "marginal", False, False),
            ("marginal+ref_exists", "marginal", True, False),
            ("marginal+ref_exists+exclusive", "marginal", True, True),
            ("marginal+ref_exists+coupled", "marginal", True, False),
            ("marginal+ref_exists+exclusive+coupled", "marginal", True, True),
        )
        head.route_mask[:] = True
        head.base_weight, head.slot_weight = 0., 1.
        for semantic, mode, require_ref, exclusive in semantic_modes:
            head.slot_executor.reference_mode = mode
            head.slot_executor.require_reference_exists = require_ref
            head.slot_executor.exclusive_attributes = exclusive
            head.slot_executor.comparison_mode = "coupled" if "coupled" in semantic else "independent"
            variant_slot = trainer.predict(split)
            for bname, base in bases.items():
                for rname, route in ("all", OP_NAMES), ("count+cmp", routes["count+cmp"]):
                    variants[f"{bname}+slot[{rname}] semantic={semantic}"] = mix(
                        base, variant_slot, pairs, pair_ops, route, base_weight=1., slot_weight=1.)
        # Restore the original model semantics.
        head.slot_executor.reference_mode = "barycenter"
        head.slot_executor.require_reference_exists = False
        head.slot_executor.exclusive_attributes = False
        head.slot_executor.comparison_mode = "independent"

    summary = {}
    for name, logits in variants.items():
        m = evaluate_scores(1 / (1 + np.exp(-logits.astype(np.float64))), labels, cats, split.pair_ids)
        summary[name] = {k: round(v, 4) for k, v in m.items()}
    focus_pairs = {c: pairs[cats[pairs[:, 0]] == c] for c in FOCUS}
    same = np.array([OP_NAMES[ops[a]] == "same" or OP_NAMES[ops[b]] == "same" for a, b in pairs])
    focus_pairs["comparison/as_many"] = pairs[(cats[pairs[:, 0]] == "comparison") & same]
    exact_n = np.array([programs[a].a == programs[b].a for a, b in pairs])
    focus_pairs["counting/number_twins"] = pairs[(cats[pairs[:, 0]] == "counting") & exact_n]
    focus_pairs["counting/filter_twins"] = pairs[(cats[pairs[:, 0]] == "counting") & ~exact_n]
    focus_pairs["overall"] = pairs
    bootstraps = {}
    for name in [n for n in variants if ("[count+cmp]" in n or "[all]" in n) and "semantic=" not in n]:
        bname = name.split("+")[0]
        bootstraps[name] = {c: paired_bootstrap(variants[name], bases[bname], labels, fp, args.repeats)
                            for c, fp in focus_pairs.items() if len(fp)}
    report = {"run": str(run), "metrics": summary, "paired_bootstrap_vs_base": bootstraps}
    if args.semantic_sweep:
        # Selection on the same validation set is exploratory, never an unbiased
        # estimate of the best-of-many improvement.
        semantic = [n for n in variants if "semantic=" in n]
        strongest = max(semantic, key=lambda n: summary[n]["auc"])
        parent = strongest.split("+slot[")[0]
        default = f"{parent}+slot[all] base_w=1"
        if default not in variants:
            default = parent
        report["semantic_sweep"] = {
            "best_on_this_val": strongest, "reference": default,
            "grouped_image_bootstrap": image_bootstrap(variants[strongest], variants[default],
                                                        labels, split.img_idx, min(args.repeats, 500)),
            "warning": "Variant chosen using the same val; CIs are exploratory, confirm on independent scenes",
        }
        print("semantic best", strongest, "vs", default,
              report["semantic_sweep"]["grouped_image_bootstrap"])
    (run / "slot_eval.json").write_text(json.dumps(report, indent=2))
    cols = ["auc", "pair_accuracy"] + [f"auc/{c}" for c in FOCUS]
    print("variant".ljust(48) + "".join(c.replace("auc/", "")[:12].rjust(13) for c in cols))
    for name, m in summary.items():
        print(name.ljust(48) + "".join(f"{m.get(c, float('nan')):13.4f}" for c in cols))


if __name__ == "__main__":
    main()
