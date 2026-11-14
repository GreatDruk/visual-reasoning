"""CPU regressions for observation-only training diagnostics."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch

from test_pipeline import build_data_dir, run_train
from src.data import ArraySource, load_split
from src.features import load_images
from src.losses import TruthLoss
from src.metrics import evaluate_scores
from src.models.fusion import FiLMHead
from src.models.image_encoders import TorchvisionEncoder
from src.models.model import VisualReasoningModel
from src.models.text_encoders import GRUEncoder, Vocab
from src.training import Trainer
from src.utils import set_seed


class TestTrainingDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setup_case(self, tmp, activation="relu"):
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
            "diagnostics": {"enabled": False, "every_steps": 1, "probe_pairs_per_category": 2},
        })
        return model, vocab, split, cfg

    def test_logging_preserves_weights_buffers_rng_and_alignment(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            model, vocab, split, cfg = self.setup_case(tmp)
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
                model, vocab, split, cfg = self.setup_case(root, activation)
                cfg.diagnostics.enabled = True
                with torch.no_grad():
                    model.fusion.readout.mlp[0].weight.zero_()
                    model.fusion.readout.mlp[0].bias.fill_(-.2)
                    model.fusion.readout.mlp[-1].weight.fill_(.1)
                trainer = Trainer(model, TruthLoss(), vocab, cfg, torch.device("cpu"), "images", [64, 96],
                                  seed=7, diagnostics_dir=root)
                trainer.fit(split, {}, {})
                events = [json.loads(line) for line in (root / "diagnostics.jsonl").read_text().splitlines()]
                first = next(event for event in events if event["event"] == "optimizer_step")
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


if __name__ == "__main__":
    unittest.main()
