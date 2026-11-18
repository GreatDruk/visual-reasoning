"""Read-only evaluation of a saved run on val (and optionally train rows not fitted).

usage: uv run python scripts/eval_checkpoint.py RUN_DIR [--state FILE] [--out FILE]
RUN_DIR must hold config.yaml and vocab.json; state defaults to RUN_DIR/model.pt.
No fitting: the model is loaded, put in eval mode and only predicts.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from omegaconf import OmegaConf
import torch

from src.data import load_split, subset
from src.models.model import VisualReasoningModel
from src.models.text_encoders import Vocab
from src.train import _attach_sources, instantiate
from src.training import Trainer
from src.utils import get_device, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--state", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    cfg = OmegaConf.load(args.run_dir / "config.yaml")
    set_seed(cfg.seed)
    device = get_device()
    splits = {"val": load_split("val", cfg.data.val_csv)}
    fitted = args.run_dir / "fitted_ids.npy"
    if fitted.exists():  # subset run: also score train rows the model never fitted
        train = load_split("train", cfg.data.train_csv)
        splits["train_unfitted"] = subset(train, ~np.isin(train.ids, np.load(fitted)))
        splits["train_fitted"] = subset(train, np.isin(train.ids, np.load(fitted)))

    names = {k: ("train" if k.startswith("train") else k) for k in splits}
    encoder = None
    for key, split in splits.items():
        encoder, channels, mode = _attach_sources(cfg, {names[key]: split}, device, False)

    saved = json.loads((args.run_dir / "vocab.json").read_text())
    vocab = Vocab([], lowercase=saved["lowercase"])
    vocab.itos = saved["itos"]
    vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}
    text = instantiate(cfg.model.text_encoder, vocab_size=len(vocab))
    fusion = instantiate(cfg.model.fusion, in_channels=channels, text_dim=text.out_dim, _convert_="all")
    model = VisualReasoningModel(text, fusion, encoder, spatial=cfg.model.image_encoder.spatial)

    state = torch.load(args.state or args.run_dir / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state and isinstance(state["model"], dict) else state)
    trainer = Trainer(model, instantiate(cfg.loss), vocab, cfg.train, device, mode,
                      cfg.data.image_size, cfg.data.normalize, cfg.seed)

    result = {"run_dir": str(args.run_dir), "state": str(args.state or "model.pt")}
    for key, split in splits.items():
        metrics, _ = trainer.evaluate(split)
        result[key] = {"rows": len(split), **metrics}
        print(key, len(split), {k: round(v, 4) for k, v in metrics.items()})
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
