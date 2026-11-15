"""Training loop: AdamW + warmup/cosine, grad clipping, early stopping, time budget,
optional step budget and step snapshots (see ``Trainer.fit``)."""

import copy
from contextlib import nullcontext
import logging
import math
import time

import numpy as np
import torch

from .data import eval_batches, train_batches
from .diagnostics import TrainingDiagnostics
from .features import preprocess
from .metrics import evaluate_scores

log = logging.getLogger(__name__)


class Trainer:
    """Fit a VisualReasoningModel on a Split and score other splits.

    `image_mode` is "features" (cached maps from the split's source) or "images"
    (uint8 images, preprocessed on the device and passed through the encoder).
    """

    def __init__(self, model, loss_fn, vocab, train_cfg, device, image_mode="features", image_size=None, normalize="imagenet", seed=0, diagnostics_dir=None):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.vocab = vocab
        self.cfg = train_cfg
        self.device = device
        self.image_mode = image_mode
        self.image_size = image_size
        self.normalize = normalize
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.diagnostics_dir = diagnostics_dir
        # One AMP decision per Trainer: amp off or CPU/MPS -> float32; CUDA bf16 when
        # supported, otherwise fp16 with a GradScaler (its skips must not step the LR schedule).
        self.amp_dtype = None
        self.scaler = None
        if train_cfg.amp and device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16
                self.scaler = torch.amp.GradScaler("cuda")
        self.best_epoch = None
        self.stop_reason = "epochs"
        self.optimizer_steps = 0  # actual optimizer updates (a GradScaler skip is not a step)
        self.processed_rows = 0   # training rows consumed by the updates of one fit

    def _make_optimizer(self):
        """Partition trainable tensors by encoder identity, without duplicate ownership."""
        encoder = self.model.image_encoder
        image_ids = {id(p) for p in encoder.parameters()} if encoder is not None else set()
        image, task = [], []
        for p in self.model.parameters():
            if p.requires_grad:
                (image if id(p) in image_ids else task).append(p)
        cfg = self.cfg.optimizer
        image_lr = cfg.get("image_encoder_lr")
        groups = []
        if image:
            groups.append({"name": "image_encoder", "params": image, "lr": cfg.lr if image_lr is None else image_lr})
        if task:
            groups.append({"name": "task", "params": task, "lr": cfg.lr})
        return torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)

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
    def predict(self, split, batch_size=None):
        """Raw logits for every row of a split; `batch_size=None` uses `cfg.batch_size`."""
        if batch_size is None:
            batch_size = self.cfg.batch_size
        self.model.eval()
        out = [self._forward(split, b).cpu().numpy() for b in eval_batches(split, batch_size)]
        return np.concatenate(out)

    def evaluate(self, split):
        logits = self.predict(split)
        probs = 1 / (1 + np.exp(-logits))
        return evaluate_scores(probs, split.labels, split.categories, split.pair_ids), logits

    def fit(self, train_split, eval_splits, aug, monitor="val_auc", step_callback=None):
        """Train; after each epoch evaluate every split in eval_splits ({name: Split}).

        Keeps the weights of the best epoch by `monitor` (e.g. "val_auc"), or the last
        epoch when early stopping is disabled; the kept epoch is in `self.best_epoch`.
        Returns the per-epoch history. A time-limited `max_steps` diagnostic run
        returns its last state and fires the final callback; a fixed-epoch refit
        without `max_steps` still raises rather than silently returning a partial fit.

        Step control (optional keys, absent/None preserves legacy behavior):
        `cfg.max_steps` stops exactly after that many optimizer updates, even mid-epoch
        (`stop_reason` becomes "max_steps"); a skipped AMP update is not an update and
        does not count. The cosine horizon is `cfg.scheduler.total_steps`, or
        `steps_per_epoch * epochs` when null, and is never retargeted by truncation, so
        the LR at an absolute step is independent of split size and run length.

        `step_callback(trainer, optimizer, scheduler, epoch, step)` runs at most once
        per step (duplicates in `cfg.snapshot_steps` are ignored), after the update and
        its diagnostics, at: step 0 (before any update, with the optimizer built and the
        scheduler's initial LR set), every entry of `cfg.snapshot_steps`, and the
        `cfg.max_steps` target. If the run ends at another step (epoch cap, early
        stopping), it fires once more at that last step. `step` is the actual update
        count (`self.optimizer_steps`), `epoch` is 0 at the step-0 call and the 1-based
        epoch in progress afterwards. The callback may call predict() and leave the
        model in eval; the trainer restores the previous train/eval mode afterwards but
        does not save or restore RNG state. `self.processed_rows` counts the training
        rows consumed; every history row reports `step`, `processed_rows` and
        `completed_epoch` (False for an epoch cut short by `cfg.max_steps`).
        Bounded runs check a monotonic deadline after every update; optional
        `snapshot_reserve_sec` reserves time for the final callback. For mid-epoch
        continuation, callbacks can save `epoch_start_sampler_rng` and
        `batches_completed_in_epoch`, replay that epoch's batches to its cursor,
        and then restore the saved framework RNG before the next forward.
        """
        cfg = self.cfg
        self.optimizer_steps = 0
        self.processed_rows = 0
        self.stop_reason = "epochs"
        self.epoch_start_sampler_rng = None
        self.batches_completed_in_epoch = 0
        self.gradient_alert = False
        self.total_gradient_norm_before_clip = None
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = self._make_optimizer()
        for group in opt.param_groups:
            log.info(f"optimizer {group['name']}: {sum(p.numel() for p in group['params'])} parameters, lr {group['lr']}")
        steps_per_epoch = math.ceil(len(train_split) / cfg.batch_size)
        total = cfg.scheduler.get("total_steps")
        if total is None:
            total = steps_per_epoch * cfg.epochs
        warmup = cfg.scheduler.get("warmup_steps")
        if warmup is None:
            warmup = steps_per_epoch * cfg.scheduler.get("warmup_epochs", 0)

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup
            if cfg.scheduler.name == "cosine":
                return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
            return 1.0

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        max_steps = cfg.get("max_steps")
        if max_steps is not None and max_steps < 1:
            raise ValueError("train.max_steps must be a positive optimizer-step target")
        triggers = set(cfg.get("snapshot_steps") or [])
        if max_steps is not None:
            triggers.add(max_steps)
        fired = set()

        def fire(epoch):
            """Invoke step_callback once per step, restoring the previous train/eval mode."""
            if self.optimizer_steps in fired:
                return
            fired.add(self.optimizer_steps)
            was_training = self.model.training
            try:
                step_callback(self, opt, sched, epoch, self.optimizer_steps)
            finally:
                self.model.train(was_training)

        es = cfg.get("early_stopping")
        patience = es.patience if es else None
        monitor = es.monitor if es else monitor
        best, best_state, bad_epochs, history = -np.inf, None, 0, []
        start = time.time()
        bounded_start = time.monotonic()
        deadline = None
        if max_steps is not None and cfg.time_budget_min:
            deadline = bounded_start + cfg.time_budget_min * 60 - cfg.get("snapshot_reserve_sec", 0)
        diagnostics = None
        diagnostic_cfg = cfg.get("diagnostics")
        if diagnostic_cfg and diagnostic_cfg.enabled:
            if self.diagnostics_dir is None:
                raise ValueError("diagnostics_dir is required when training diagnostics are enabled")
            diagnostics = TrainingDiagnostics(self, train_split, diagnostic_cfg, self.diagnostics_dir, self.seed)
            diagnostics.probe(0, 0, "initial")
        if step_callback is not None:
            fire(0)  # before any update; optimizer and the scheduler's initial LR are set
        step = 0

        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            t0, losses = time.time(), []
            truncated = False
            self.epoch_start_sampler_rng = copy.deepcopy(self.rng.bit_generator.state)
            self.batches_completed_in_epoch = 0
            for batch_index, batch in enumerate(train_batches(train_split, cfg.batch_size, self.rng, aug, cfg.pair_sampler), 1):
                step += 1
                self.gradient_alert = False
                near_deadline = deadline is not None and time.monotonic() >= deadline - 1
                observe = diagnostics is not None and (diagnostics.should_capture(step) or near_deadline)
                with diagnostics.trace(train_split, batch, epoch, step, opt) if observe else nullcontext():
                    logits = self._forward(train_split, batch)
                    pair_index = batch.get("pair_index")
                    loss = self.loss_fn(
                        logits,
                        batch["labels"].to(self.device),
                        pair_index.to(self.device) if pair_index is not None else None,
                    )
                    opt.zero_grad(set_to_none=True)
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(opt)
                    else:
                        loss.backward()
                # Do not apply a corrupt update in a bounded diagnostic run.
                if max_steps is not None and not bool(torch.isfinite(loss)):
                    self.stop_reason = "nonfinite_loss"
                    truncated = True
                    break
                if cfg.grad_clip or max_steps is not None:
                    norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip or float("inf"))
                    self.total_gradient_norm_before_clip = float(norm)
                    alert_threshold = cfg.get("anomaly_snapshot_gradient_norm")
                    self.gradient_alert = alert_threshold is not None and float(norm) > alert_threshold
                    if max_steps is not None and not bool(torch.isfinite(norm)):
                        self.stop_reason = "nonfinite_gradients"
                        truncated = True
                        break
                if self.scaler is not None:
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(opt)
                    self.scaler.update()
                    stepped = self.scaler.get_scale() >= scale_before  # scale shrinks on an inf-skip
                else:
                    opt.step()
                    stepped = True
                if stepped:
                    sched.step()
                    self.optimizer_steps += 1
                if observe:
                    diagnostics.finish_step(logits, loss, batch, opt, stepped)
                    diagnostics.probe(epoch, step, "step")
                self.processed_rows += len(batch["captions"])
                self.batches_completed_in_epoch = batch_index
                if max_steps is not None and self.optimizer_steps >= max_steps:
                    self.stop_reason = "max_steps"
                if (deadline is not None and self.optimizer_steps != max_steps
                        and time.monotonic() >= deadline):
                    self.stop_reason = "time_budget"
                if step_callback is not None and (self.optimizer_steps in triggers or self.gradient_alert):
                    fire(epoch)
                losses.append(loss.item())
                if max_steps is not None and self.optimizer_steps >= max_steps:
                    truncated = batch_index < steps_per_epoch
                    break
                if self.stop_reason == "time_budget":
                    truncated = batch_index < steps_per_epoch
                    break

            if (diagnostics is not None and diagnostics.last_probe_step != step
                    and not self.stop_reason.startswith("nonfinite")):
                diagnostics.probe(epoch, step, "epoch_end")
            row = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else None,
                   "epoch_sec": round(time.time() - t0, 1),
                   "step": self.optimizer_steps, "processed_rows": self.processed_rows,
                   "completed_epoch": not truncated}
            row.update({f"lr/{g['name']}": g["lr"] for g in opt.param_groups})
            for name, split in eval_splits.items():
                metrics, _ = self.evaluate(split)
                row.update({f"{name}_{k}": v for k, v in metrics.items()})
            history.append(row)
            log.info(" | ".join(
                f"{k} {v:.3e}" if k.startswith("lr/") else
                f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                for k, v in row.items() if "_auc/" not in k
            ))

            score = row[monitor] if es is not None else (-row["loss"] if losses else -np.inf)
            improved = score > best
            if improved:
                best, bad_epochs = score, 0
            else:
                bad_epochs += 1
            if improved or es is None:
                # Fixed-epoch refit keeps the current last state (at max_steps when the run
                # is truncated); early stopping keeps the best monitored epoch instead.
                self.best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
            if self.stop_reason in ("time_budget", "nonfinite_loss", "nonfinite_gradients"):
                log.info(f"{self.stop_reason} at epoch {epoch}, step {self.optimizer_steps}")
                break
            if max_steps is not None and self.optimizer_steps >= max_steps:
                # The requested target is reached: stop exactly here, and a time budget
                # cannot call this fit incomplete.
                self.stop_reason = "max_steps"
                log.info(f"max_steps target reached after epoch {epoch}, step {self.optimizer_steps}")
                break
            if es is not None and bad_epochs >= patience:
                self.stop_reason = "early_stopping"
                log.info(f"early stopping at epoch {epoch}, best {monitor} {best:.4f}")
                break
            if epoch < cfg.epochs and max_steps is None:
                elapsed_min = (time.time() - start) / 60
                if cfg.time_budget_min and elapsed_min * (epoch + 1) / epoch > cfg.time_budget_min:
                    self.stop_reason = "time_budget"
                    if es is None:
                        raise RuntimeError(
                            f"time budget {cfg.time_budget_min} min would stop the fixed {cfg.epochs}-epoch "
                            f"refit after epoch {epoch} ({elapsed_min:.1f} min elapsed); fit is incomplete"
                        )
                    log.info(f"time budget reached after epoch {epoch} ({elapsed_min:.1f} min)")
                    break

        if step_callback is not None and self.optimizer_steps not in fired:
            # Final target: the run's last step when no earlier trigger covered it. Fired
            # before the restore so model/optimizer/scheduler all describe that step.
            fire(epoch)
        self.model.load_state_dict(best_state)
        if diagnostics is not None and not self.stop_reason.startswith("nonfinite"):
            diagnostics.probe(self.best_epoch, step, "restored")
        return history
