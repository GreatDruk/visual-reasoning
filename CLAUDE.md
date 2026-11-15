# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Course Kaggle competition (Intro-to-ML 2026): given a 240x160 CLEVR-like scene and a caption, predict P(caption is true). Metric is ROC-AUC. Full task, data and rules are in `docs/` (`data.md`, `rules.md`, `desc.md`); the working plan, open questions, experiment table and naming rules are in `KaggleTemplate.md` (Russian). Read `docs/rules.md` before adding any model, data source or augmentation: the rules are strict and violations disqualify.

Current state: configs are designed, but `src/` is still empty. Every `_target_` in `configs/` (`src.models.image_encoders.*`, `src.models.text_encoders.*`, `src.models.fusion.*`, `src.losses.TruthLoss`) and the `src.train` entrypoint are yet to be implemented. The configs are the spec.

## Commands

Package manager is uv, Python >= 3.14. `src` is installed as a package (`[tool.setuptools] packages = ["src"]`).

```bash
uv sync
uv run python -m src.train                                   # defaults: resnet101 + gru + film + bce_pair
uv run python -m src.train experiment=film_rn                # preset
uv run python -m src.train model/fusion=mac model/image_encoder=convnext_tiny
uv run python -m src.train -m model/fusion=film,mac loss=bce,bce_pair   # multirun
uv run python -m src.train -m tune=optuna experiment=film    # Optuna sweep
uv run marimo edit notebooks/01_edaa.py                      # notebooks are marimo .py files
```

No tests or linter are configured.

Dependencies referenced by configs but missing from `pyproject.toml`: `open_clip_torch` (CLIP encoders), `hydra-optuna-sweeper` (tune=optuna), `transformers` (bert_base), `scikit-learn` or equivalent for AUC. Add them with `uv add` when implementing.

## Data

`data/` (gitignored): `train.csv` ~30k captions / 5k images, `val.csv` 6k / 1k, `test.csv` ~36k / 6k, plus image folders per split. Columns: `id, image, caption, level, category, pair_id, label`; test has no `pair_id`/`label`. Each `pair_id` groups two captions for the same image differing by one detail, exactly one true. 9 categories over 5 levels, 50/50 balanced per category. Some test phrasings (word orders) never appear in train/val.

Submission column: `sample_submission.csv` uses `label`, `rules.md` says `target`. Currently `configs/data/default.yaml: submission_target_col: label`; unresolved question.

## Config architecture (Hydra)

`configs/config.yaml` composes groups: `data`, `model/image_encoder`, `model/text_encoder`, `model/fusion`, `loss`, `aug`, `train`, `tune`, then an optional `experiment` preset (`# @package _global_`, uses `override /group: ...`) applied last. `hydra.job.chdir: false` and all paths are relative to the repo root: rules require relative paths, keep it that way.

Contracts implied by the configs that implementation must respect:

- Image encoders expose `spatial` (feature map `[B, C, H, W]` vs pooled vector) and `freeze`. Fusion heads declare `requires_spatial`; FiLM/MAC/RN need spatial maps, cosine/mlp need pooled vectors (cosine averages a spatial map if given).
- `data.normalize` (`imagenet` | `clip`) and `data.image_size` must match the encoder. Default upsamples 160x240 to 320x480 so ResNet-101 `layer3` (stride 16) yields a 20x30 grid.
- `train.cache_features: true`: with a frozen encoder, features are computed once and reused.
- `loss=bce_pair` (BCE + pairwise logistic over `pair_id`) requires `train.pair_sampler: true` so both twins land in the same batch.
- `train.time_budget_min: 45`: the final notebook must train + predict in 50 min on Kaggle default GPU.
- `train.postprocess.twin_normalize`: softmax within twin pairs found in test (test has no `pair_id`; twins must be found from captions). Allowed; off by default, its effect is measured on val.
- `experiment=final` sets `data.train_on_val: true`, disables early stopping, fixed epoch count. `lab_baseline` is the individual-checkpoint model and must train on `train.csv` only.
- `tune=optuna` expects the train function to return val AUC.

## Rules that affect code (from docs/rules.md)

- Allowed: torchvision classification models with ImageNet weights (any layers), text-only encoders (GRU/LSTM/BERT-like), CLIP/SigLIP/OpenCLIP dual encoders, any custom layers.
- Forbidden: detection/segmentation models (even torchvision ones), joint vision-language models (BLIP, ViLT, OWL-ViT, ...), LLMs or text generators (including for making captions), external data or models trained on similar data (CLEVR, NLVR, ShapeWorld).
- Unlabelled `test.csv` images and captions may be used freely (analysis, twin finding, fitting preprocessing/vocabularies/representations). Only hidden test labels are off limits (no manual labelling, no attempts to recover them).
- Augmentations: hflip only with left/right swapped in the caption; no vflip, rotation, colour/hue jitter; crop/zoom risky; light brightness/contrast/blur/noise OK. Text: clause reordering (same label) and logical negatives (n->n+-1, left<->right) are allowed. See `configs/aug/default.yaml`.
- Final deliverable is one notebook that runs with Restart & Run All, offline, seeds set for Python/NumPy/torch, produces `final_submission.csv`, <= 20 MB of code.

## Conventions (from KaggleTemplate.md)

- Notebooks: `notebooks/0X_<name>`.
- Experiments: global three-digit id `00X_<name>`, never reused even for failed runs.
- Every main experiment run (one that goes into the Solution table) saves its full composed config to `configs/archive/00X_<name>.yaml`. The archive is append-only; never edit existing files there. Debug runs are not archived.
- Report both val AUC and val accuracy (Moodle table asks for accuracy, metric is AUC), plus pair accuracy and per-category AUC.
