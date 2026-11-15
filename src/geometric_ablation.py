"""Evaluate opt-in routes on one shared train-only geometry checkpoint.

No training, test data, category routing or threshold/weight search is performed.
Example: python -m src.geometric_ablation artefacts/hydra/<name>/<timestamp>
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from .data import canonicalized, load_split, pair_groups
from .geometric_eval import bootstrap_pairs
from .models.geometric_count import parse_comparison_caption
from .models.model import VisualReasoningModel
from .models.text_encoders import Vocab
from .train import _attach_sources, instantiate
from .training import Trainer
from .utils import get_device, set_seed


VARIANTS = {"baseline": (False, "off"), "existence": (False, "off"),
            "spatial": (True, "off"), "equality": (True, "equality"), "all": (True, "all")}


def prediction_statistics(logits, split, repeats=2000):
    """Same float32 probabilities as Trainer, complete-pair bootstrap throughout."""
    with np.errstate(over="ignore"):
        scores = 1 / (1 + np.exp(-logits))
    statistics = {"overall": bootstrap_pairs(scores, split.labels, split.pair_ids, repeats)}
    for category in sorted(set(split.categories)):
        selected = split.categories == category
        statistics[category] = bootstrap_pairs(scores[selected], split.labels[selected], split.pair_ids[selected], repeats)
    groups = pair_groups(split)
    comparison = np.array([parse_comparison_caption(c) for c in split.captions], dtype=object)
    for name in ("comparison_equality_pairs", "comparison_more_fewer_pairs",
                 "comparison_true_equal_pairs", "comparison_false_equal_pairs"):
        chosen = []
        for a, b in groups:
            if comparison[a] is not None and comparison[b] is not None:
                has_equal = comparison[a].operator == "equal" or comparison[b].operator == "equal"
                # Truth-conditioned subsets are diagnostics AFTER prediction.
                # Neither this label nor any subset key reaches the model.
                true_equal = comparison[a if split.labels[a] else b].operator == "equal"
                false_equal = comparison[b if split.labels[a] else a].operator == "equal"
                selected = {"comparison_equality_pairs": has_equal,
                            "comparison_more_fewer_pairs": not has_equal,
                            "comparison_true_equal_pairs": true_equal,
                            "comparison_false_equal_pairs": false_equal}[name]
                if selected:
                    chosen.extend((a, b))
        if chosen:
            statistics[name] = bootstrap_pairs(scores[chosen], split.labels[chosen], split.pair_ids[chosen], repeats)
    return statistics


def evaluate_variants(run_dir, variants, output, repeats=2000, device_name=None, comparison_source=None):
    start = time.perf_counter()
    run_dir, output = Path(run_dir), Path(output)
    cfg = OmegaConf.load(run_dir / "config.yaml")
    if comparison_source is not None:
        cfg.model.fusion.comparison_source = comparison_source
    training = json.loads((run_dir / "results.json").read_text())
    if cfg.data.train_on_val:
        raise ValueError("ablation validation must be held out from training")
    if any(v != "baseline" for v in variants) and cfg.model.fusion.get("existence_weight", 0) <= 0:
        raise ValueError("existence ablations need a checkpoint trained with existence_weight > 0")
    set_seed(cfg.seed)
    device = get_device(device_name or cfg.device)
    split = load_split("val", cfg.data.val_csv)
    if cfg.data.get("canonicalize", False):
        split = canonicalized(split)
    encoder, channels, mode = _attach_sources(cfg, {"val": split}, device, False)
    saved_vocab = json.loads((run_dir / "vocab.json").read_text())
    vocab = Vocab([], lowercase=saved_vocab["lowercase"])
    vocab.itos = saved_vocab["itos"]
    vocab.stoi = {word: i for i, word in enumerate(vocab.itos)}
    text = instantiate(cfg.model.text_encoder, vocab_size=len(vocab))
    fusion = instantiate(cfg.model.fusion, in_channels=channels, text_dim=text.out_dim, _convert_="all")
    model = VisualReasoningModel(text, fusion, image_encoder=encoder)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    trainer = Trainer(model, instantiate(cfg.loss), vocab, cfg.train, device, mode,
                      cfg.data.image_size, cfg.data.normalize, cfg.seed)
    output.mkdir(parents=True, exist_ok=True)
    results = {"checkpoint": str(run_dir), "seed": int(cfg.seed), "device": str(device),
               "training_fit_min": training["fit_min"], "training_total_min": training["total_min"],
               "best_epoch": training["best_epoch"], "epochs_run": training["epochs_run"],
               "training_stop_reason": training["stop_reason"], "variants": {}}
    for name in variants:
        fusion.route_spatial, fusion.comparison_route = VARIANTS[name]
        fusion.caption_pairing = name != "baseline" or cfg.model.fusion.get("existence_weight", 0) > 0
        predict_start = time.perf_counter()
        metrics, logits = trainer.evaluate(split)
        predict_sec = time.perf_counter() - predict_start
        variant_dir = output / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        np.save(variant_dir / "val_logits.npy", logits)
        np.save(variant_dir / "val_ids.npy", split.ids)
        # All bootstrap computation happens after inference; labels enter here only.
        statistics = prediction_statistics(logits, split, repeats)
        result = {"val": metrics, "predict_sec": predict_sec, "statistics": statistics,
                  "route_spatial": fusion.route_spatial, "comparison_route": fusion.comparison_route,
                  "comparison_source": fusion.comparison_source,
                  "geometry_weight": fusion.geometry_weight, "baseline_weight": fusion.baseline_weight,
                  "comparison_weight": fusion.comparison_weight}
        results["variants"][name] = result
        (variant_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"{name}: val AUC {metrics['auc']:.6f}; inference {predict_sec:.2f}s", flush=True)
    results["evaluation_total_sec"] = time.perf_counter() - start
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["existence", "spatial", "equality", "all"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--device")
    parser.add_argument("--comparison-source", choices=["stage2", "geometry"])
    args = parser.parse_args()
    output = args.output or args.run_dir / "route_ablations"
    evaluate_variants(args.run_dir, args.variants, output, args.repeats, args.device, args.comparison_source)


if __name__ == "__main__":
    main()
