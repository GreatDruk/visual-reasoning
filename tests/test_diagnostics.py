"""CPU regressions for observation-only training diagnostics and step-controlled training."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch

from test_pipeline import build_data_dir, run_train
from src.data import ArraySource, load_split, subset
from src.features import load_images
from src.losses import TruthLoss
from src.metrics import evaluate_scores
from src.models.fusion import FiLMHead
from src.models.image_encoders import TorchvisionEncoder
from src.models.model import VisualReasoningModel
from src.models.text_encoders import GRUEncoder, Vocab
from src.training import Trainer
from src.utils import set_seed


def build_case(tmp, activation="relu"):
    """Shared fixture: tiny data dir, a small FiLM model and a CPU-ready train config."""
    build_data_dir(tmp / "data", with_test=False)
    split = load_split("train", tmp / "data/train.csv")
    split.source = ArraySource(load_images(tmp / "data/train", split.image_names))
    vocab = Vocab(split.captions)
    set_seed(7)
    encoder = TorchvisionEncoder("resnet18", None, "layer3", freeze=True, unfreeze_last_n_blocks=1)
    text = GRUEncoder(len(vocab), embed_dim=8, hidden_dim=8, dropout=.2)
    fusion = FiLMHead(encoder.out_channels, text.out_dim, num_blocks=1, hidden_channels=8,
                      classifier_channels=16, classifier_hidden=16, readout="mlp", dropout=.2,
                      readout_activation=activation)
    model = VisualReasoningModel(text, fusion, encoder)
    cfg = OmegaConf.create({
        "amp": False, "epochs": 1, "batch_size": 2, "pair_sampler": True, "grad_clip": 1.,
        "early_stopping": None, "time_budget_min": None,
        "optimizer": {"lr": 3e-4, "image_encoder_lr": 3e-5, "weight_decay": 1e-4},
        "scheduler": {"name": "constant", "warmup_epochs": 0},
        "diagnostics": {"enabled": False, "every_steps": 1, "probe_pairs_per_category": 2,
                        "channel_stats": True},
    })
    return model, vocab, split, cfg


class TestTrainingDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_logging_preserves_weights_buffers_rng_and_alignment(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            model, vocab, split, cfg = build_case(tmp)
            outputs = []
            for enabled in (False, True):
                directory = tmp / str(enabled)
                directory.mkdir()
                current_cfg = copy.deepcopy(cfg)
                current_cfg.diagnostics.enabled = enabled
                trainer = Trainer(copy.deepcopy(model), TruthLoss(), vocab, current_cfg, torch.device("cpu"),
                                  "images", [64, 96], seed=7, diagnostics_dir=directory)
                set_seed(123)
                history = trainer.fit(split, {}, {})
                outputs.append(({k: v.clone() for k, v in trainer.model.state_dict().items()},
                                torch.get_rng_state().clone(), history[0]["loss"]))
            self.assertEqual(outputs[0][2], outputs[1][2])
            self.assertTrue(torch.equal(outputs[0][1], outputs[1][1]))
            for key in outputs[0][0]:
                self.assertTrue(torch.equal(outputs[0][0][key], outputs[1][0][key]), key)
            self.assertFalse((tmp / "False/diagnostics.jsonl").exists())
            manifest = json.loads((tmp / "True/diagnostic_probe.json").read_text())
            names = np.asarray(manifest["images"])
            self.assertTrue(np.all(names != names[manifest["image_order"]]))
            events = [json.loads(line) for line in (tmp / "True/diagnostics.jsonl").read_text().splitlines()]
            probes = [event for event in events if event["event"] == "probe"]
            self.assertEqual(probes[0]["step"], 0)
            self.assertEqual(probes[-1]["reason"], "restored")
            for event in probes:
                scores = 1 / (1 + np.exp(-np.asarray(event["logits"], dtype=np.float32)))
                expected = evaluate_scores(scores, manifest["labels"], manifest["categories"], manifest["pair_ids"])
                self.assertEqual(event["metrics"], expected)
            steps = [event for event in events if event["event"] == "optimizer_step"]
            self.assertEqual(sorted(i for event in steps for i in event["ids"]), sorted(split.ids.tolist()))
            for event in steps:
                self.assertEqual(event["activation_gradients"]["readout_linear"]["negative_input_norm"], 0.)
                self.assertLessEqual(sum(g["norm"] ** 2 for g in event["gradients_after_clip"].values()) ** .5, 1.00001)
                for name, state in event["batchnorm"].items():
                    self.assertEqual(state["training"], not name.startswith("image_encoder"))

    def test_negative_input_gradient_is_before_inplace_relu(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for activation in ("relu", "gelu"):
                root = tmp / activation
                root.mkdir()
                model, vocab, split, cfg = build_case(root, activation)
                cfg.diagnostics.enabled = True
                with torch.no_grad():
                    model.fusion.readout.mlp[0].weight.zero_()
                    model.fusion.readout.mlp[0].bias.copy_(torch.linspace(-.8, -.2, 16))
                    model.fusion.readout.mlp[-1].weight.fill_(.1)
                trainer = Trainer(model, TruthLoss(), vocab, cfg, torch.device("cpu"), "images", [64, 96],
                                  seed=7, diagnostics_dir=root)
                trainer.fit(split, {}, {})
                events = [json.loads(line) for line in (root / "diagnostics.jsonl").read_text().splitlines()]
                first = next(event for event in events if event["event"] == "optimizer_step")
                channels = first["channel_distributions"]["readout_linear"]
                self.assertEqual(channels["channel_shape"], [16])
                np.testing.assert_allclose(channels["p05"], np.linspace(-.8, -.2, 16), atol=1e-6)
                np.testing.assert_array_equal(channels["negative_fraction"], np.ones(16))
                grad_channels = first["activation_gradients"]["readout_linear"]["negative_input_channel_norm"]
                np.testing.assert_allclose(np.linalg.norm(grad_channels),
                                           first["activation_gradients"]["readout_linear"]["negative_input_norm"],
                                           rtol=1e-5, atol=1e-9)
                self.assertLess(first["activations"]["readout_linear"]["max"], 0.)
                gradient = first["activation_gradients"]["readout_linear"]["negative_input_norm"]
                if activation == "relu":
                    self.assertEqual(gradient, 0.)
                else:
                    self.assertGreater(gradient, 0.)

    def test_cli_trace_is_train_only_with_real_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            build_data_dir(tmp / "data", with_test=False)
            proc = run_train(tmp, "test_trace", [
                "experiment=film_trace", "model.image_encoder.arch=resnet18",
                "model.image_encoder.unfreeze_last_n_blocks=1", "data.image_size=[64,96]",
                "model.text_encoder.embed_dim=8", "model.text_encoder.hidden_dim=8",
                "model.fusion.num_blocks=1", "model.fusion.hidden_channels=8",
                "model.fusion.classifier_channels=16", "model.fusion.classifier_hidden=16",
            ])
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            result = json.loads((tmp / "out/results.json").read_text())
            self.assertFalse(any(key in result for key in ("val", "val_para", "submission")))
            manifest = json.loads((tmp / "out/diagnostic_probe.json").read_text())
            original = load_split("train", tmp / "data/train.csv")
            self.assertTrue(set(manifest["ids"]).issubset(set(original.ids)))
            events = [json.loads(line) for line in (tmp / "out/diagnostics.jsonl").read_text().splitlines()]
            restored = events[-1]
            self.assertEqual(restored["reason"], "restored")
            scores = 1 / (1 + np.exp(-np.asarray(restored["logits"], dtype=np.float32)))
            expected = evaluate_scores(scores, manifest["labels"], manifest["categories"], manifest["pair_ids"])
            self.assertEqual(restored["metrics"], expected)
            self.assertEqual(manifest["fitted_rows"], len(original))
            self.assertFalse(list((tmp / "out").glob("test_*")))
            self.assertFalse(list((tmp / "subs").glob("*.csv")))

    def test_absolute_warmup_overrides_epochs_and_preserves_legacy(self):
        with tempfile.TemporaryDirectory() as td:
            for setting, expected in ((None, [1/3, 2/3, 1]), (2, [.5, 1, 1]), (0, [1, 1, 1])):
                root = Path(td) / str(setting)
                root.mkdir()
                model, vocab, split, cfg = build_case(root)
                cfg.diagnostics.enabled = True
                cfg.scheduler.warmup_epochs = 1
                if setting is not None:
                    cfg.scheduler.warmup_steps = setting
                trainer = Trainer(model, TruthLoss(), vocab, cfg, torch.device("cpu"),
                                  "images", [64, 96], seed=7, diagnostics_dir=root)
                trainer.fit(split, {}, {})
                events = [json.loads(line) for line in (root / "diagnostics.jsonl").read_text().splitlines()]
                steps = [row for row in events if row["event"] == "optimizer_step"]
                np.testing.assert_allclose([row["lr_used"]["task"] / cfg.optimizer.lr for row in steps], expected)
                np.testing.assert_allclose([row["lr_used"]["image_encoder"] / cfg.optimizer.image_encoder_lr
                                           for row in steps], expected)

    def test_signed_margins_use_labels_not_row_order(self):
        from src.diagnostics import prediction_summary

        # Pairs are interleaved; one is reversed in row order, one is tied.
        summary = prediction_summary(np.array([0., 2., 1., 2., 1., -1.]),
                                     np.array([0, 0, 1, 1, 0, 1]), np.array([8, 9, 8, 9, 10, 10]))
        self.assertAlmostEqual(summary["signed_pair_margin"]["mean"], -1/3, places=6)
        self.assertEqual(summary["pair_accuracy"], .5)
        expected = torch.nn.functional.softplus(torch.tensor([-1., 0., 2.])).mean()
        self.assertAlmostEqual(summary["pair_logistic"], float(expected))


class TestStepControls(unittest.TestCase):
    """CPU regressions for max_steps truncation, absolute scheduler horizon and step_callback."""

    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def _train(self, root, model, vocab, split, cfg, *, epochs, max_steps=None,
               snapshot_steps=(), step_callback=None):
        root.mkdir(parents=True, exist_ok=True)
        cfg = copy.deepcopy(cfg)
        cfg.epochs = epochs
        cfg.max_steps = max_steps
        cfg.snapshot_steps = list(snapshot_steps)
        cloned = copy.deepcopy(model)
        cloned.train(model.training)  # FX GraphModule deepcopy resets child training flags.
        trainer = Trainer(cloned, TruthLoss(), vocab, cfg, torch.device("cpu"),
                          "images", [64, 96], seed=7)
        set_seed(123)
        history = trainer.fit(split, {}, {}, step_callback=step_callback)
        return trainer, history

    def test_max_steps_truncates_mid_epoch_and_returns_that_state(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            model, vocab, split, cfg = build_case(tmp)
            cfg.scheduler.name = "cosine"
            cfg.scheduler.warmup_steps = 0
            cfg.scheduler.total_steps = 6
            fired, at_target, lrs = [], [], []
            sampler_at_target = []

            def snapshot(trainer, opt, sched, epoch, step):
                fired.append(step)
                lrs.append({g["name"]: g["lr"] for g in opt.param_groups})
                if step == 5:
                    at_target.append({k: v.detach().clone() for k, v in trainer.model.state_dict().items()})
                    sampler_at_target.append(copy.deepcopy(trainer.rng.bit_generator.state))

            trainer, history = self._train(tmp / "a", model, vocab, split, cfg, epochs=2, max_steps=5,
                                           snapshot_steps=[2], step_callback=snapshot)
            # 3 steps per epoch: the target stops after two batches of epoch 2.
            self.assertEqual(fired, [0, 2, 5])
            self.assertEqual(trainer.optimizer_steps, 5)
            self.assertEqual(trainer.processed_rows, 10)
            self.assertEqual(trainer.stop_reason, "max_steps")
            self.assertEqual(trainer.rng.bit_generator.state, sampler_at_target[0])
            self.assertEqual([(r["epoch"], r["step"], r["processed_rows"], r["completed_epoch"]) for r in history],
                             [(1, 3, 6, True), (2, 5, 10, False)])
            # The cosine horizon (6) is untouched by truncation: step 5 sits at 5/6 of the
            # decay. A horizon retargeted to max_steps would report exactly 0 at step 5.
            for group in ("task", "image_encoder"):
                base = cfg.optimizer.lr if group == "task" else cfg.optimizer.image_encoder_lr
                np.testing.assert_allclose([lr[group] / base for lr in lrs], [1., .75, .06698729810778])
            # early_stopping=null returns exactly the state reached at the target step.
            self.assertEqual(len(at_target), 1)
            for key, tensor in trainer.model.state_dict().items():
                self.assertTrue(torch.equal(at_target[0][key], tensor), key)
            # The truncated fit is a prefix of an untruncated one at the same absolute step.

            def checkpoint(trainer, opt, sched, epoch, step):
                if step == 5:
                    at_target.append({k: v.detach().clone() for k, v in trainer.model.state_dict().items()})

            self._train(tmp / "b", model, vocab, split, cfg, epochs=2, snapshot_steps=[5],
                        step_callback=checkpoint)
            for key, tensor in at_target[1].items():
                self.assertTrue(torch.equal(at_target[0][key], tensor), key)

    def test_absolute_horizon_gives_equal_lr_across_split_sizes(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            model, vocab, split, cfg = build_case(tmp)
            cfg.scheduler.name = "cosine"
            cfg.scheduler.warmup_steps = 3
            cfg.scheduler.total_steps = 6
            small = subset(split, np.isin(split.pair_ids, [11, 12]))  # 2 pairs -> 2 steps/epoch
            trajectories = {}
            for name, data, epochs in (("large", split, 2), ("small", small, 4)):
                records = []

                def snapshot(trainer, opt, sched, epoch, step, records=records):
                    records.append((step, {g["name"]: g["lr"] for g in opt.param_groups}))

                trainer, _ = self._train(tmp / name, model, vocab, data, cfg, epochs=epochs, max_steps=5,
                                         snapshot_steps=[0, 1, 2, 3, 4, 5], step_callback=snapshot)
                self.assertEqual(trainer.stop_reason, "max_steps")
                trajectories[name] = records
            # Overlapping triggers (0 in snapshot_steps, target in snapshot_steps) fire once each.
            for name in ("large", "small"):
                self.assertEqual([step for step, _ in trajectories[name]], [0, 1, 2, 3, 4, 5])
            # Absolute warmup and horizon: the LR per absolute step is identical although
            # steps per epoch (3 vs 2) and the epoch cap (2 vs 4) differ. The legacy
            # epochs-derived totals (6 vs 8) would diverge from step 4 on.
            self.assertEqual(trajectories["large"], trajectories["small"])
            for group in ("task", "image_encoder"):
                base = cfg.optimizer.lr if group == "task" else cfg.optimizer.image_encoder_lr
                np.testing.assert_allclose([lr[group] / base for _, lr in trajectories["large"]],
                                           [1/3, 2/3, 1., 1., .75, .25])

    def test_callback_eval_leaves_training_state_intact(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            model, vocab, split, cfg = build_case(tmp)
            val = load_split("val", tmp / "data/val.csv")
            val.source = ArraySource(load_images(tmp / "data/val", val.image_names))
            reference = {name: module.training for name, module in model.named_modules()}
            modes = []

            def snapshot(trainer, opt, sched, epoch, step):
                before = {name: module.training for name, module in trainer.model.named_modules()}
                trainer.evaluate(val)  # predict() leaves the model in eval
                after = {name: module.training for name, module in trainer.model.named_modules()}
                modes.append((step, before, after))

            trainer, _ = self._train(tmp / "a", model, vocab, split, cfg, epochs=2, max_steps=4,
                                     snapshot_steps=[1, 2, 3], step_callback=snapshot)
            self.assertEqual([step for step, _, _ in modes], [0, 1, 2, 3, 4])
            for step, before, after in modes:
                self.assertEqual(before, reference, step)     # the trainer restored train mode
                self.assertFalse(any(after.values()), step)   # although the callback left eval
            # Training continued in train mode throughout: identical weights without callbacks.
            plain, _ = self._train(tmp / "b", model, vocab, split, cfg, epochs=2, max_steps=4,
                                   snapshot_steps=[1, 2, 3])
            for key, tensor in trainer.model.state_dict().items():
                self.assertTrue(torch.equal(plain.model.state_dict()[key], tensor), key)


if __name__ == "__main__":
    unittest.main()
