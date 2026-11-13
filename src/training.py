"""Training loop: AdamW + warmup/cosine, grad clipping, early stopping, time budget."""

import copy
import logging
import math
import time

import numpy as np
import torch

from .data import eval_batches, train_batches
from .features import preprocess
from .metrics import evaluate_scores

log = logging.getLogger(__name__)


class Trainer:
    """Fit a VisualReasoningModel on a Split and score other splits.

    `image_mode` is "features" (cached maps from the split's source) or "images"
    (uint8 images, preprocessed on the device and passed through the encoder).
    """

    def __init__(self, model, loss_fn, vocab, train_cfg, device, image_mode="features", image_size=None, normalize="imagenet", seed=0):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.vocab = vocab
        self.cfg = train_cfg
        self.device = device
        self.image_mode = image_mode
        self.image_size = image_size
        self.normalize = normalize
        self.rng = np.random.default_rng(seed)
        self.amp_dtype = torch.bfloat16 if train_cfg.amp else None

    def _inputs(self, split, batch):
        x = split.source.get(batch["img_idx"], batch["flip"]).to(self.device, non_blocking=True)
        if self.image_mode == "images":
            x = preprocess(x, self.image_size, self.normalize)
        else:
            x = x.float()
        tokens, lengths = self.vocab.encode_batch(batch["captions"])
        return x, tokens.to(self.device), lengths

    def _forward(self, split, batch):
        x, tokens, lengths = self._inputs(split, batch)
        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
            return self.model(x, tokens, lengths).float()

    @torch.no_grad()
    def predict(self, split, batch_size=512):
        """Raw logits for every row of a split."""
        self.model.eval()
        out = [self._forward(split, b).cpu().numpy() for b in eval_batches(split, batch_size)]
        return np.concatenate(out)

    def evaluate(self, split):
        logits = self.predict(split)
        probs = 1 / (1 + np.exp(-logits))
        return evaluate_scores(probs, split.labels, split.categories, split.pair_ids), logits

    def fit(self, train_split, eval_splits, aug, monitor="val_auc"):
        """Train; after each epoch evaluate every split in eval_splits ({name: Split}).

        Keeps the weights of the best epoch by `monitor` (e.g. "val_auc"), or the last
        epoch when early stopping is disabled. Returns the per-epoch history.
        """
        cfg = self.cfg
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
        steps_per_epoch = math.ceil(len(train_split) / cfg.batch_size)
        total = steps_per_epoch * cfg.epochs
        warmup = steps_per_epoch * cfg.scheduler.get("warmup_epochs", 0)

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup
            if cfg.scheduler.name == "cosine":
                return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
            return 1.0

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        es = cfg.get("early_stopping")
        patience = es.patience if es else None
        monitor = es.monitor if es else monitor
        best, best_state, bad_epochs, history = -np.inf, None, 0, []
        start = time.time()

        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            t0, losses = time.time(), []
            for batch in train_batches(train_split, cfg.batch_size, self.rng, aug, cfg.pair_sampler):
                logits = self._forward(train_split, batch)
                pair_index = batch.get("pair_index")
                loss = self.loss_fn(
                    logits,
                    batch["labels"].to(self.device),
                    pair_index.to(self.device) if pair_index is not None else None,
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                sched.step()
                losses.append(loss.item())

            row = {"epoch": epoch, "loss": float(np.mean(losses)), "epoch_sec": round(time.time() - t0, 1)}
            for name, split in eval_splits.items():
                metrics, _ = self.evaluate(split)
                row.update({f"{name}_{k}": v for k, v in metrics.items() if "/" not in k})
            history.append(row)
            log.info(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))

            score = row.get(monitor, -row["loss"])
            if score > best:
                best, bad_epochs = score, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                bad_epochs += 1
            if es is None:
                best_state = copy.deepcopy(self.model.state_dict())
            elif bad_epochs >= patience:
                log.info(f"early stopping at epoch {epoch}, best {monitor} {best:.4f}")
                break
            elapsed_min = (time.time() - start) / 60
            if cfg.time_budget_min and elapsed_min * (epoch + 1) / epoch > cfg.time_budget_min:
                log.info(f"time budget reached after epoch {epoch} ({elapsed_min:.1f} min)")
                break

        self.model.load_state_dict(best_state)
        return history
