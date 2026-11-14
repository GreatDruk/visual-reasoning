"""Opt-in FiLM training observations; no changes to gradients, RNG or model state."""

import hashlib
import json
import logging
import math
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import pair_groups, sample_pairs_by_category
from .metrics import evaluate_scores
from .models.fusion import FiLMHead

log = logging.getLogger(__name__)


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _stats(tensor):
    x = tensor.detach().float()
    finite = torch.isfinite(x)
    out = {"shape": list(x.shape), "finite": bool(finite.all())}
    if not out["finite"]:
        out["finite_fraction"] = float(finite.float().mean())
    out.update(mean=_number(x.mean()), std=_number(x.std(unbiased=False)),
               min=_number(x.min()), max=_number(x.max()),
               zero_fraction=float((x == 0).float().mean()),
               negative_fraction=float((x < 0).float().mean()),
               abs_below_0_05_fraction=float((x.abs() < .05).float().mean()),
               abs_below_1e_6_fraction=float((x.abs() < 1e-6).float().mean()))
    if x.ndim == 2:
        out["all_zero_units"] = int((x == 0).all(dim=0).sum())
    if x.ndim == 4:
        out["spatial_std_mean"] = _number(x.std(dim=(2, 3), unbiased=False).mean())
    return out


def _channels(tensor):
    """Batch distribution per channel (and per FiLM block), never across channels."""
    x = tensor.detach().to(device="cpu", dtype=torch.float32)
    quantiles = torch.quantile(x, x.new_tensor([.05, .5, .95]), dim=0)
    return {
        "channel_shape": list(x.shape[1:]),
        "mean": x.mean(dim=0).tolist(), "std": x.std(dim=0, unbiased=False).tolist(),
        "p05": quantiles[0].tolist(), "p50": quantiles[1].tolist(), "p95": quantiles[2].tolist(),
        "negative_fraction": (x < 0).float().mean(dim=0).tolist(),
        "abs_below_0_05_fraction": (x.abs() < .05).float().mean(dim=0).tolist(),
    }


def _norm(tensors):
    norms = [t.detach().float().norm() for t in tensors]
    return float(torch.stack(norms).norm()) if norms else 0.0


class TrainingDiagnostics:
    """Observe selected optimizer steps and a fixed train-only intervention probe."""

    def __init__(self, trainer, split, cfg, output_dir, seed):
        if not isinstance(trainer.model.fusion, FiLMHead) or not hasattr(trainer.model.fusion.readout, "mlp"):
            raise ValueError("training diagnostics require FiLM with an MLP readout")
        if cfg.every_steps <= 0:
            raise ValueError("diagnostics.every_steps must be positive")
        self.trainer, self.cfg = trainer, cfg
        self.current: dict[str, Any] = {}
        self.before: dict[str, list[torch.Tensor]] = {}
        self.path = Path(output_dir) / "diagnostics.jsonl"
        self.path.touch(exist_ok=False)
        self.probe_split = sample_pairs_by_category(split, cfg.probe_pairs_per_category, seed)
        self.last_probe_step = None
        probe = self.probe_split
        indices = np.arange(len(probe))
        self.image_order = None
        for shift in range(1, len(probe)):
            order = np.roll(indices, shift)
            if np.all(probe.img_idx[order] != probe.img_idx):
                self.image_order = order
                break
        if self.image_order is None:
            raise ValueError("diagnostic probe needs enough distinct images for an image derangement")
        self.caption_order = indices.copy()
        groups = pair_groups(probe)
        self.caption_order[groups[:, 0]] = groups[:, 1]
        self.caption_order[groups[:, 1]] = groups[:, 0]
        self.stages = self._stages()
        self.groups = self._parameter_groups()
        digest = hashlib.sha256()
        for name, tensor in trainer.model.state_dict().items():
            digest.update(name.encode())
            value = tensor.detach().cpu().contiguous().numpy()
            digest.update(memoryview(value))
        manifest = {
            "schema_version": 1, "split": "fitted_train_probe", "seed": seed,
            "initial_state_sha256": digest.hexdigest(), "fitted_rows": len(split),
            "ids": probe.ids.tolist(), "labels": probe.labels.tolist(),
            "pair_ids": probe.pair_ids.tolist(), "categories": probe.categories.tolist(),
            "captions": probe.captions,
            "images": [probe.image_names[i] for i in probe.img_idx],
            "image_order": self.image_order.tolist(), "caption_order": self.caption_order.tolist(),
        }
        with (self.path.parent / "diagnostic_probe.json").open("x") as stream:
            json.dump(manifest, stream, indent=2)

    def _stages(self):
        model, fusion = self.trainer.model, self.trainer.model.fusion
        stages = {"sentence": model.text_encoder, "stem": fusion.stack.stem,
                  "film_params": fusion.stack.film,
                  **{f"film_block/{i}": block for i, block in enumerate(fusion.stack.blocks)},
                  "projection": fusion.proj, "readout_linear": fusion.readout.mlp[0],
                  "readout_activation": fusion.readout.mlp[1], "logits": fusion.readout.mlp[-1]}
        if model.image_encoder is not None:
            stages = {"image_encoder": model.image_encoder, **stages}
        return stages

    def _parameter_groups(self):
        model, fusion = self.trainer.model, self.trainer.model.fusion
        modules = {"text_encoder": model.text_encoder, "film_generator": fusion.stack.film,
                   "stem": fusion.stack.stem, "projection": fusion.proj,
                   **{f"film_block/{i}": block for i, block in enumerate(fusion.stack.blocks)},
                   "readout": fusion.readout}
        encoder = model.image_encoder
        if encoder is not None:
            stage = encoder.body.get_submodule(encoder.out_layer)
            modules.update({f"image_encoder/{encoder.out_layer}/{i}": block
                            for i, block in enumerate(stage.children())})
        groups = {name: [p for p in module.parameters() if p.requires_grad] for name, module in modules.items()}
        if encoder is not None:
            covered = {id(p) for params in groups.values() for p in params}
            groups["image_encoder/other"] = [p for p in encoder.parameters()
                                               if p.requires_grad and id(p) not in covered]
        return {name: params for name, params in groups.items() if params}

    def emit(self, row):
        with self.path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def should_capture(self, step):
        return step in (1, 10, 50) or step % self.cfg.every_steps == 0

    @contextmanager
    def trace(self, split, batch, epoch, step, optimizer):
        self.current = {
            "event": "optimizer_step", "epoch": epoch, "step": step,
            "ids": split.ids[batch["rows"]].tolist(),
            "images": [split.image_names[i] for i in batch["img_idx"]],
            "captions": batch["captions"], "labels": batch["labels"].tolist(),
            "lr_used": {g["name"]: g["lr"] for g in optimizer.param_groups},
            "activations": {}, "activation_gradients": {}, "channel_distributions": {},
        }
        self.before = {name: [p.detach().clone() for p in params] for name, params in self.groups.items()}
        handles = []
        loss_scale = self.trainer.scaler.get_scale() if self.trainer.scaler is not None else 1.
        started = time.perf_counter()

        def hook(name):
            def record(module, args, output):
                value = output[0] if isinstance(output, tuple) else output
                self.current["activations"][name] = _stats(value)
                if name == "film_params":
                    stack = self.trainer.model.fusion.stack
                    film = value.detach().view(-1, stack.num_blocks, 2, stack.hidden)
                    self.current["activations"]["gamma"] = _stats(1 + film[:, :, 0])
                    self.current["activations"]["beta"] = _stats(film[:, :, 1])
                    if self.cfg.get("channel_stats", False):
                        self.current["channel_distributions"]["gamma"] = _channels(1 + film[:, :, 0])
                        self.current["channel_distributions"]["beta"] = _channels(film[:, :, 1])
                if name == "readout_linear" and self.cfg.get("channel_stats", False):
                    self.current["channel_distributions"]["readout_linear"] = _channels(value)
                if name == "readout_linear" and value.requires_grad:
                    negative = value.detach() < 0
                    # Register before an in-place ReLU. retain_grad on its output instead
                    # observes the post-ReLU gradient and mislabels negative-input gradients.
                    def gradient_record(gradient):
                        g = gradient.detach()
                        if loss_scale != 1.:
                            g = g / loss_scale
                        self.current["activation_gradients"]["readout_linear"] = {
                            "norm": _number(g.norm()), "negative_input_norm": _number(g[negative].norm()),
                            "negative_input_nonzero_count": int(torch.count_nonzero(g[negative])),
                        }
                        if self.cfg.get("channel_stats", False):
                            self.current["activation_gradients"]["readout_linear"].update(
                                channel_norm=g.square().sum(dim=0).sqrt().cpu().tolist(),
                                negative_input_channel_norm=(g * negative).square().sum(dim=0).sqrt().cpu().tolist(),
                            )
                    handles.append(value.register_hook(gradient_record))
            return record

        try:
            for name, module in self.stages.items():
                handles.append(module.register_forward_hook(hook(name)))
            def input_record(module, args):
                self.current["activations"]["readout_input"] = _stats(args[0])
            handles.append(self.trainer.model.fusion.readout.register_forward_pre_hook(input_record))
            encoder = self.trainer.model.image_encoder
            if encoder is not None:
                def image_input_record(module, args):
                    self.current["activations"]["image_input"] = _stats(args[0])
                handles.append(encoder.register_forward_pre_hook(image_input_record))
            yield
            self.current["gradients_before_clip"] = self._gradient_stats()
            self.current["forward_backward_sec"] = round(time.perf_counter() - started, 4)
        finally:
            for handle in handles:
                handle.remove()

    def _gradient_stats(self):
        result = {}
        for name, params in self.groups.items():
            grads = [p.grad for p in params if p.grad is not None]
            count = sum(g.numel() for g in grads)
            norm = _norm(grads)
            result[name] = {"norm": _number(norm), "rms": _number(norm / math.sqrt(count)) if count else 0.,
                            "finite": all(bool(torch.isfinite(g).all()) for g in grads),
                            "nonzero_tensors": sum(bool(torch.count_nonzero(g)) for g in grads),
                            "tensors": len(grads), "elements": count}
        return result

    def finish_step(self, logits, loss, batch, optimizer, stepped):
        row = self.current
        row.update(loss=float(loss.detach()), logits=logits.detach().cpu().tolist(), stepped=stepped,
                   lr_after={g["name"]: g["lr"] for g in optimizer.param_groups},
                   gradients_after_clip=self._gradient_stats())
        row["updates"] = {}
        for name, params in self.groups.items():
            before = self.before[name]
            delta = _norm([p.detach() - old for p, old in zip(params, before)])
            weight = _norm(before)
            row["updates"][name] = {"norm": _number(delta), "weight_norm": _number(weight),
                                     "relative_norm": _number(delta / weight) if weight else None}
        self.before.clear()
        if "pair_index" in batch:
            pairs = batch["pair_index"].to(logits.device)
            margins = (logits[pairs[:, 0]] - logits[pairs[:, 1]]).detach()
            row["pair_margin"] = _stats(margins)
        row["batchnorm"] = {name: {"training": module.training,
                                   "running_mean": _stats(module.running_mean),
                                   "running_var": _stats(module.running_var),
                                   "batches_tracked": int(module.num_batches_tracked)}
                            for name, module in self.trainer.model.named_modules()
                            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}
        self.emit(row)

    def _capture_prediction(self, split):
        outputs = {name: [] for name in self.stages}
        outputs["readout_input"] = []
        handles = []

        def capture(name):
            def hook(module, args, output):
                value = output[0] if isinstance(output, tuple) else output
                outputs[name].append(value.detach().to(device="cpu", dtype=torch.float32, copy=True))
            return hook

        try:
            for name, module in self.stages.items():
                handles.append(module.register_forward_hook(capture(name)))
            def capture_input(module, args):
                outputs["readout_input"].append(args[0].detach().to(device="cpu", dtype=torch.float32, copy=True))
            handles.append(self.trainer.model.fusion.readout.register_forward_pre_hook(capture_input))
            logits = self.trainer.predict(split)
        finally:
            for handle in handles:
                handle.remove()
        return logits, {name: torch.cat(parts) for name, parts in outputs.items()}

    def probe(self, epoch, step, reason):
        trainer, split = self.trainer, self.probe_split
        was_training = trainer.model.training
        started = time.perf_counter()
        try:
            logits, stages = self._capture_prediction(split)
            swapped = replace(split, img_idx=split.img_idx[self.image_order])
            image_logits, image_stages = self._capture_prediction(swapped)
            captions = replace(split, captions=[split.captions[i] for i in self.caption_order])
            text_logits = trainer.predict(captions)
        finally:
            trainer.model.train(was_training)
        scores = 1 / (1 + np.exp(-logits))
        metrics = evaluate_scores(scores, split.labels, split.categories, split.pair_ids)
        statistics = {name: _stats(value) for name, value in stages.items()}
        channels = {}
        if self.cfg.get("channel_stats", False):
            stack = trainer.model.fusion.stack
            film = stages["film_params"].view(-1, stack.num_blocks, 2, stack.hidden)
            channels = {"gamma": _channels(1 + film[:, :, 0]), "beta": _channels(film[:, :, 1]),
                        "readout_linear": _channels(stages["readout_linear"])}
        sensitivity = {}
        for name, value in stages.items():
            delta = float((value - image_stages[name]).abs().mean())
            scale = float(value.std(unbiased=False))
            sensitivity[name] = {"mean_abs_delta": delta, "delta_over_std": delta / scale if scale else None}
        self.emit({"event": "probe", "epoch": epoch, "step": step, "reason": reason,
                   "metrics": metrics,
                   "logits": logits.tolist(), "image_shuffled_logits": image_logits.tolist(),
                   "caption_swapped_logits": text_logits.tolist(), "activations": statistics,
                   "channel_distributions": channels,
                   "image_sensitivity_by_stage": sensitivity,
                   "image_delta": float(np.abs(logits - image_logits).mean()),
                   "caption_delta": float(np.abs(logits - text_logits).mean()),
                   "probe_sec": round(time.perf_counter() - started, 4)})
        self.last_probe_step = step
        log.info(f"probe step {step}: auc {metrics['auc']:.4f}, "
                 f"image_delta {np.abs(logits - image_logits).mean():.6g}, "
                 f"MLP zero_fraction {statistics['readout_activation']['zero_fraction']:.3f}")
