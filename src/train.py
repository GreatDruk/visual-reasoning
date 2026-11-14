import json
import logging
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate as _instantiate
from omegaconf import OmegaConf

from .data import ArraySource, concat_splits, load_split, paraphrased, save_submission, subset
from .features import cache_features, cache_key, load_images
from .metrics import evaluate_scores
from .models.model import VisualReasoningModel
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

    splits = {s: load_split(s, cfg.data[f"{s}_csv"]) for s in ("train", "val", "test")}
    if cfg.data.get("limit"):
        # Smoke tests: keep the first `limit` captions (whole twin pairs) of train and val.
        for s in ("train", "val"):
            sp = splits[s]
            splits[s] = subset(sp, np.isin(sp.pair_ids, np.unique(sp.pair_ids)[: cfg.data.limit // 2]))

    need_flip = cfg.aug.image.hflip_swap_lr > 0
    encoder, in_channels, image_mode = _attach_sources(cfg, splits, device, need_flip)

    train_split = splits["train"]
    if cfg.data.train_on_val:
        train_split = concat_splits(splits["train"], splits["val"], "train_val")

    # Vocabulary from the labelled training captions (and their rewrites) only.
    vocab_captions = list(train_split.captions)
    if cfg.data.train_on_val is False:
        vocab_captions += splits["val"].captions
    vocab_captions += [p for c, cat in zip(train_split.captions, train_split.categories) for p in paraphrases(c, cat)]
    vocab = Vocab(vocab_captions, lowercase=cfg.model.text_encoder.get("lowercase", True))

    text_encoder = instantiate(cfg.model.text_encoder, vocab_size=len(vocab))
    fusion = instantiate(cfg.model.fusion, in_channels=in_channels, text_dim=text_encoder.out_dim, _convert_="all")
    spatial = cfg.model.image_encoder.spatial or cfg.model.fusion.requires_spatial
    model = VisualReasoningModel(text_encoder, fusion, image_encoder=encoder, spatial=spatial)
    if cfg.model.get("init_from"):
        _warm_start(model, cfg.model.init_from)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"trainable params: {n_params / 1e6:.2f}M, vocab {len(vocab)}, in_channels {in_channels}")

    loss_fn = instantiate(cfg.loss)
    trainer = Trainer(model, loss_fn, vocab, cfg.train, device, image_mode, cfg.data.image_size, cfg.data.normalize, cfg.seed)
    aug = {"hflip_swap_lr": cfg.aug.image.hflip_swap_lr, "reorder_clauses_p": cfg.aug.text.reorder_clauses_p}

    eval_splits = {}
    if not cfg.data.train_on_val:
        eval_splits = {"val": splits["val"], "val_para": paraphrased(splits["val"])}
    history = trainer.fit(train_split, eval_splits, aug)
    train_min = (time.time() - t_start) / 60

    results = {"name_exp": cfg.name_exp, "epochs_run": len(history), "train_min": round(train_min, 1), "trainable_params": n_params}
    if eval_splits:
        for name, split in eval_splits.items():
            metrics, logits = trainer.evaluate(split)
            results[name] = metrics
            names = [split.image_names[i] for i in split.img_idx]
            pairs = find_twins(names, split.captions, split.categories)
            tn = twin_normalize(logits, pairs)
            results[f"{name}_twin_norm"] = {k: v for k, v in evaluate_scores(tn, split.labels, split.categories, split.pair_ids).items() if "/" not in k}
            np.save(out_dir / f"{name}_logits.npy", logits)
        v = results["val"]
        log.info(
            f"RESULT {cfg.name_exp}: val auc {v['auc']:.4f} acc {v['accuracy']:.4f} pair_acc {v['pair_accuracy']:.4f} | "
            f"val_para auc {results['val_para']['auc']:.4f} | twin_norm auc {results['val_twin_norm']['auc']:.4f}"
        )
        log.info("per-category val auc: " + ", ".join(f"{k[4:]} {x:.3f}" for k, x in v.items() if k.startswith("auc/")))

    # Test predictions.
    test = splits["test"]
    test_logits = trainer.predict(test)
    np.save(out_dir / "test_logits.npy", test_logits)
    probs = 1 / (1 + np.exp(-test_logits))
    if cfg.train.postprocess.twin_normalize:
        names = [test.image_names[i] for i in test.img_idx]
        probs = twin_normalize(test_logits, find_twins(names, test.captions, test.categories))
    fusion_name = cfg.model.fusion._target_.rsplit(".", 1)[-1]
    sub_path = save_submission(
        test.ids, probs, f"{cfg.paths.submissions_dir}/submission_{fusion_name}_{cfg.model.image_encoder.arch}_{cfg.name_exp}.csv", cfg.data.submission_target_col
    )
    results["submission"] = str(sub_path.relative_to(resolve_path(".")))
    results["total_min"] = round((time.time() - t_start) / 60, 1)
    results["history"] = history
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    torch.save(model.state_dict(), out_dir / "model.pt")
    log.info(f"submission {sub_path}; total {results['total_min']} min")
    return results["val"]["auc"] if eval_splits else 0.0


if __name__ == "__main__":
    main()
