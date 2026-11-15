import json
import logging
import platform
import time
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
import torchvision
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate as _instantiate
from omegaconf import OmegaConf

from .data import ArraySource, canonicalized, concat_splits, load_split, pair_groups, paraphrased, sample_pairs_by_category, save_submission, subset
from .features import cache_features, cache_key, load_images
from .metrics import evaluate_scores
from .models.model import VisualReasoningModel
from .models.count_distribution import parse_count_caption
from .models.geometric_count import caption_pair_key
from .models.text_encoders import Vocab
from .text_aug import paraphrases
from .training import Trainer
from .twins import find_twins, twin_normalize
from .utils import get_device, resolve_path, set_seed

log = logging.getLogger(__name__)

TRUSTED_TARGETS = ["src.*", "torch.optim.*"]


def instantiate(cfg, **kwargs):
    """hydra instantiate restricted to this package's classes (Hydra 1.4 execution whitelist)."""
    return _instantiate(cfg, _execution_whitelist_=TRUSTED_TARGETS, **kwargs)


def _archive_config(cfg):
    """Save the composed config to configs/archive/<name_exp>.yaml (append-only)."""
    path = resolve_path(f"configs/archive/{cfg.name_exp}.yaml")
    if path.exists():
        raise FileExistsError(f"{path} exists: archive is append-only, pick a new experiment id")
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path, resolve=True)


def _attach_sources(cfg, splits, device, need_flip):
    """Give each split an image source: cached feature maps or raw images."""
    enc_cfg = cfg.model.image_encoder
    # A complete cache needs no encoder construction or pretrained-weight download.
    # Name-indexed views also let data.limit reuse a full read-only cache safely.
    if cfg.train.cache_features and enc_cfg.freeze and not enc_cfg.get("unfreeze_last_n_blocks", 0):
        base = resolve_path(cfg.paths.features_dir) / cache_key(enc_cfg, cfg.data)
        sources = {}
        channels = set()
        for name, split in splits.items():
            path = base / f"{name}.npy"
            names_path = path.with_suffix(".json")
            flip_path = base / f"{name}_flip.npy"
            flip_names = flip_path.with_suffix(".json")
            use_flip = need_flip and name in ("train", "val")
            if not path.exists() or not names_path.exists():
                break
            cached_names = json.loads(names_path.read_text())
            if use_flip and (not flip_path.exists() or not flip_names.exists()
                             or json.loads(flip_names.read_text()) != cached_names):
                break
            positions = {n: i for i, n in enumerate(cached_names)}
            if not all(n in positions for n in split.image_names):
                break
            arr = np.load(path, mmap_mode="r")
            flipped = np.load(flip_path, mmap_mode="r") if use_flip else None
            if arr.ndim != 4 or len(arr) != len(cached_names) or (use_flip and flipped.shape != arr.shape):
                raise ValueError(f"invalid spatial feature cache: {path}")
            indices = None if cached_names == split.image_names else np.array([positions[n] for n in split.image_names])
            sources[name] = ArraySource(arr, flipped, indices)
            channels.add(arr.shape[1])
        if len(sources) == len(splits):
            if len(channels) != 1:
                raise ValueError("feature caches have inconsistent channel counts")
            for name, source in sources.items():
                splits[name].source = source
            log.info("reusing frozen feature caches without constructing the image encoder")
            return None, channels.pop(), "features"
    encoder = instantiate(enc_cfg)
    cached = cfg.train.cache_features and not encoder.trainable
    image_dirs = {"train": cfg.data.train_images, "val": cfg.data.val_images, "test": cfg.data.test_images}
    for name, split in splits.items():
        if cached:
            spatial_encoder = encoder
            spatial_encoder.spatial = True  # cache full maps; pooling happens in the model
            base = Path(cfg.paths.features_dir) / cache_key(enc_cfg, cfg.data)
            kw = dict(size=cfg.data.image_size, normalize=cfg.data.normalize, device=device)
            arr = cache_features(spatial_encoder, image_dirs[name], split.image_names, base / f"{name}.npy", **kw)
            flipped = None
            if need_flip and name in ("train", "val"):
                flipped = cache_features(spatial_encoder, image_dirs[name], split.image_names, base / f"{name}_flip.npy", flip=True, **kw)
            split.source = ArraySource(arr, flipped)
        else:
            imgs = load_images(image_dirs[name], split.image_names)
            split.source = ArraySource(imgs, imgs[..., ::-1].copy() if need_flip else None)
    encoder.spatial = enc_cfg.spatial
    if cached:
        return None, encoder.out_channels, "features"
    return encoder, encoder.out_channels, "images"


def _warm_start(model, name_exp):
    """Copy every parameter whose name and shape match from the latest run of `name_exp`.

    Spatial heads start at chance and take many epochs to find the text-image interaction;
    starting from a model trained on pooled features (same weights, 1x1 input) skips that.
    """
    runs = sorted(resolve_path(f"artefacts/hydra/{name_exp}").glob("*/model.pt"))
    if not runs:
        raise FileNotFoundError(f"no model.pt for init_from={name_exp}")
    src = torch.load(runs[-1], map_location="cpu")
    own = model.state_dict()
    copied = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
    if getattr(model.fusion, "freeze_base", False):
        optional = getattr(model.fusion, "warm_start_optional_prefixes", ("fusion.count_distribution.",))
        missing = [k for k in own if not k.startswith(optional) and k not in copied]
        if missing:
            raise ValueError(f"frozen count model needs a complete compatible baseline; missing tensors: {missing}")
    own.update(copied)
    model.load_state_dict(own)
    log.info(f"warm start from {runs[-1]}: {len(copied)}/{len(own)} tensors copied")


@hydra.main(config_path="../configs", config_name="config")
def main(cfg):
    """Train one configuration, report validation metrics, write a submission.

    Returns the validation AUC (used as the objective by multirun sweeps).
    """
    t_start = time.time()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    log.info(f"{cfg.name_exp} on {device}; output {out_dir}")
    if cfg.save_config:
        _archive_config(cfg)

    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if cfg.loss.pair_weight > 0 and not cfg.train.pair_sampler:
        raise ValueError("pair_weight > 0 requires train.pair_sampler=true")
    if cfg.train.pair_sampler and cfg.train.batch_size % 2:
        raise ValueError("pair sampler requires an even train.batch_size")
    pairs_per_category = cfg.data.train_pairs_per_category
    if pairs_per_category is not None and (cfg.data.limit is not None or cfg.data.train_on_val):
        raise ValueError("data.train_pairs_per_category cannot be combined with data.limit or train_on_val")
    names = ["train"]
    if cfg.train.eval.holdout or cfg.data.train_on_val:
        names.append("val")
    canonicalize = cfg.data.get("canonicalize", False)
    prep = canonicalized if canonicalize else (lambda split: split)
    splits = {s: prep(load_split(s, cfg.data[f"{s}_csv"])) for s in names}
    # Canonical captions make clause-reordering augmentation a no-op, so it is disabled.
    reorder_p = 0.0 if canonicalize else cfg.aug.text.reorder_clauses_p
    if pairs_per_category is not None:
        splits["train"] = sample_pairs_by_category(splits["train"], pairs_per_category, cfg.seed)
    if cfg.data.limit is not None:
        # Smoke tests: retain whole twin pairs in every loaded split.
        for s in splits:
            sp = splits[s]
            splits[s] = subset(sp, np.isin(sp.pair_ids, np.unique(sp.pair_ids)[: cfg.data.limit // 2]))
    for name, split in splits.items():
        if not len(split):
            raise ValueError(f"{name}: empty split after data.limit; at least one complete pair is required")

    need_flip = cfg.aug.image.hflip_swap_lr > 0
    cache_start = time.time()
    encoder, in_channels, image_mode = _attach_sources(cfg, splits, device, need_flip)
    cache_min = (time.time() - cache_start) / 60
    if cfg.data.get("categories"):
        # Category routing (category is a given column in every split, test included).
        # Filtered after attaching sources, so feature caches always cover the full split.
        keep = list(cfg.data.categories)
        splits = {s: subset(sp, np.isin(sp.categories, keep)) for s, sp in splits.items()}
        for name, split in splits.items():
            if not len(split):
                raise ValueError(f"{name}: empty split after data.categories={keep}")

    train_split = splits["train"]
    if cfg.data.train_on_val:
        train_split = concat_splits(splits["train"], splits["val"], "train_val")

    if cfg.model.text_encoder._target_ == "src.models.text_encoders.ClipTextEncoder":
        text_encoder = instantiate(cfg.model.text_encoder)
        vocab = text_encoder.tokenizer  # fixed pretrained vocabulary; no caption fitting
    else:
        # Vocabulary from the labelled training captions (and their rewrites) only.
        vocab_captions = list(train_split.captions)
        if reorder_p > 0:
            vocab_captions += [p for c, cat in zip(train_split.captions, train_split.categories) for p in paraphrases(c, cat)]
        vocab = Vocab(vocab_captions, lowercase=cfg.model.text_encoder.get("lowercase", True))
        if cfg.model.get("init_from"):
            # Reuse the source run's (train-only) vocabulary so word embeddings line up
            # for the warm start even when data.categories narrows the captions.
            runs = sorted(resolve_path(f"artefacts/hydra/{cfg.model.init_from}").glob("*/vocab.json"))
            if runs:
                saved = json.loads(runs[-1].read_text())
                vocab.itos = saved["itos"]
                vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}
                vocab.lowercase = saved["lowercase"]
                log.info(f"vocab from {runs[-1]} ({len(vocab)} entries)")

        text_encoder = instantiate(cfg.model.text_encoder, vocab_size=len(vocab))
    fusion = instantiate(cfg.model.fusion, in_channels=in_channels, text_dim=text_encoder.out_dim, _convert_="all")
    spatial = cfg.model.image_encoder.spatial or cfg.model.fusion.requires_spatial
    model = VisualReasoningModel(text_encoder, fusion, image_encoder=encoder, spatial=spatial)
    if cfg.model.get("init_from"):
        _warm_start(model, cfg.model.init_from)
    elif getattr(fusion, "freeze_base", False):
        raise ValueError("freeze_base=true requires model.init_from with a trained pair-choice checkpoint")
    if getattr(fusion, "train_count_only", False):
        count_rows = np.array([parse_count_caption(c) is not None for c in train_split.captions])
        if not count_rows.any() or not np.all(count_rows[pair_groups(train_split)].sum(1) % 2 == 0):
            raise ValueError("count-only training requires complete exactly-N twin pairs")
        train_split = subset(train_split, count_rows)
        log.info(f"frozen base; training count branch on {len(train_split)} exactly-N rows only")
    elif getattr(fusion, "train_geometry_only", False):
        geometry_rows = np.array([fusion.supports_training_caption(c) for c in train_split.captions])
        if not geometry_rows.any() or not np.all(geometry_rows[pair_groups(train_split)].sum(1) % 2 == 0):
            raise ValueError("geometry training requires complete supported twin pairs")
        train_split = subset(train_split, geometry_rows)
        log.info(f"frozen stage 2; training geometry on {len(train_split)} text-parsed rows only")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"trainable params: {n_params / 1e6:.2f}M, vocab {len(vocab)}, in_channels {in_channels}")

    loss_fn = instantiate(cfg.loss)
    train_cfg = OmegaConf.create(OmegaConf.to_container(cfg.train, resolve=True))
    preparation_min = (time.time() - t_start) / 60
    if cfg.train.time_budget_min:
        remaining = cfg.train.time_budget_min - preparation_min
        if remaining <= 0:
            raise RuntimeError("Training time budget exhausted during preparation")
        train_cfg.time_budget_min = remaining
    trainer = Trainer(model, loss_fn, vocab, train_cfg, device, image_mode, cfg.data.image_size, cfg.data.normalize,
                      cfg.seed, diagnostics_dir=out_dir)
    aug = {"hflip_swap_lr": cfg.aug.image.hflip_swap_lr, "reorder_clauses_p": reorder_p}

    eval_splits = {}
    if cfg.train.eval.train:
        eval_splits["train_fit"] = train_split
    if cfg.train.eval.holdout and not cfg.data.train_on_val:
        eval_splits["val"] = splits["val"]
        if cfg.train.eval.get("paraphrases", True):
            eval_splits["val_para"] = prep(paraphrased(splits["val"]))
    if cfg.train.early_stopping is not None:
        available = set()
        for name, split in eval_splits.items():
            metrics = ["auc", "accuracy"]
            if split.pair_ids is not None:
                metrics.append("pair_accuracy")
            metrics.extend(f"auc/{category}" for category in sorted(set(split.categories)))
            available.update(f"{name}_{metric}" for metric in metrics)
        monitor = cfg.train.early_stopping.monitor
        if monitor not in available:
            raise ValueError(f"early stopping monitor {monitor!r} unavailable; choose from {sorted(available)}")
    fit_start = time.time()
    history = trainer.fit(train_split, eval_splits, aug)
    fit_min = (time.time() - fit_start) / 60
    train_min = (time.time() - t_start) / 60

    results = {
        "name_exp": cfg.name_exp, "seed": cfg.seed,
        "architecture": f"{cfg.model.image_encoder.get('arch') or cfg.model.image_encoder.model_name}/{type(text_encoder).__name__}/{type(fusion).__name__}",
        "epochs_run": len(history), "best_epoch": trainer.best_epoch,
        "stop_reason": trainer.stop_reason,
        "device": str(device), "python_version": platform.python_version(),
        "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
        "cache_min": round(cache_min, 4), "preparation_min": round(preparation_min, 4),
        "fit_min": round(fit_min, 4), "train_min": round(train_min, 4),
        "trainable_params": n_params,
    }
    predict_start = time.time()
    for name, split in eval_splits.items():
        metrics, logits = trainer.evaluate(split)
        results[name] = metrics
        if name in ("val", "val_para") and cfg.train.postprocess.twin_normalize:
            names = [split.image_names[i] for i in split.img_idx]
            keys = [caption_pair_key(c) for c in split.captions] if getattr(fusion, "caption_pairing", False) else split.categories
            pairs = find_twins(names, split.captions, keys)
            tn = twin_normalize(logits, pairs)
            results[f"{name}_twin_norm"] = {k: v for k, v in evaluate_scores(tn, split.labels, split.categories, split.pair_ids).items() if "/" not in k}
        np.save(out_dir / f"{name}_logits.npy", logits)
        np.save(out_dir / f"{name}_ids.npy", split.ids)
        log.info(f"RESULT {cfg.name_exp} {name}: auc {metrics['auc']:.4f} acc {metrics['accuracy']:.4f} "
                 f"pair_acc {metrics['pair_accuracy']:.4f}")

    if cfg.train.predict_test:
        # Test data is first accessed only after fitting and restoring selected weights.
        test = prep(load_split("test", cfg.data.test_csv))
        _attach_sources(cfg, {"test": test}, device, False)
        test_logits = trainer.predict(test)
        np.save(out_dir / "test_logits.npy", test_logits)
        np.save(out_dir / "test_ids.npy", test.ids)
        probs = torch.sigmoid(torch.from_numpy(test_logits)).numpy()
        if cfg.train.postprocess.twin_normalize:
            names = [test.image_names[i] for i in test.img_idx]
            keys = [caption_pair_key(c) for c in test.captions] if getattr(fusion, "caption_pairing", False) else test.categories
            probs = twin_normalize(test_logits, find_twins(names, test.captions, keys))
        sample = pl.read_csv(resolve_path(cfg.data.sample_submission))
        if [c for c in sample.columns if c != "id"] != [cfg.data.submission_target_col]:
            raise ValueError("sample submission target column does not match data.submission_target_col")
        sample_ids = sample["id"].to_numpy()
        if (len(probs) != len(test.ids) or len(test.ids) != len(sample_ids)
                or len(np.unique(test.ids)) != len(test.ids)
                or len(np.unique(sample_ids)) != len(sample_ids)
                or set(test.ids) != set(sample_ids)):
            raise ValueError("test and sample must contain the same unique ids exactly once")
        if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
            raise ValueError("submission probabilities must be finite and in [0, 1]")
        positions = {row_id: i for i, row_id in enumerate(test.ids)}
        ordered_probs = probs[[positions[row_id] for row_id in sample_ids]]
        sub_path = save_submission(sample_ids, ordered_probs, cfg.paths.submission, cfg.data.submission_target_col)
        results["submission"] = str(cfg.paths.submission)
        log.info(f"submission {sub_path}")
    results["predict_min"] = round((time.time() - predict_start) / 60, 4)
    results["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    results["history"] = history
    (out_dir / "vocab.json").write_text(json.dumps({"itos": vocab.itos, "lowercase": vocab.lowercase}, indent=2))
    torch.save(model.state_dict(), out_dir / "model.pt")
    results["total_min"] = round((time.time() - t_start) / 60, 4)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    log.info(f"total {results['total_min']} min")
    return results["val"]["auc"] if "val" in results else 0.0


if __name__ == "__main__":
    main()
