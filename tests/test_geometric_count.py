"""Geometry, shared numeric readout, frozen routing and train-only supervision."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.data import ArraySource, Split
from src.losses import TruthLoss
from src.geometric_eval import bootstrap_pairs
from src.models.count_distribution import PairCountChoiceHead
from src.models.geometric_count import (
    ABSENCE, COUNT, POSITION, SPATIAL, GeometricCountDistribution,
    PairGeometricCountChoiceHead, coordinate_grid, parse_geometric_caption,
)
from src.models.model import VisualReasoningModel
from src.models.text_encoders import GRUEncoder, Vocab
from src.train import _warm_start
from src.training import Trainer


def fixture():
    captions = ["exactly one red cube", "exactly two red cubes",
                "there is exactly one sphere behind the blue cube", "there are exactly two spheres behind the blue cube",
                "a red cube", "a blue cube",
                "the red cube is on the left side of the image", "there is a red cube on the right side of the image",
                "a red cube behind the blue sphere", "the red cube is in front of the blue sphere",
                "there is no sphere behind the blue cube", "there is no sphere in front of the blue cube"]
    torch.manual_seed(3)
    vocab = Vocab(captions)
    text = GRUEncoder(len(vocab), embed_dim=8, hidden_dim=8, dropout=.4)
    kw = dict(dim=8, queries=2, hidden=16, count_dim=8, count_selectors=2, count_hidden=16,
              train_count_only=False)
    head = PairGeometricCountChoiceHead(8, text.out_dim, geometry_dim=8, **kw)
    split = Split("train", np.arange(12), [f"{i}.jpg" for i in range(6)], np.repeat(np.arange(6), 2),
                  captions, np.repeat(["counting", "spatial_counting", "attribute", "position", "spatial", "spatial_negation"], 2),
                  np.tile([1, 0], 6), np.repeat(np.arange(6), 2),
                  ArraySource(np.random.default_rng(5).normal(size=(6, 8, 3, 5)).astype(np.float32)))
    cfg = OmegaConf.create({"amp": False, "batch_size": 12, "pair_sampler": True,
                           "optimizer": {"lr": .001, "weight_decay": .01}, "epochs": 1,
                           "scheduler": {"name": "cosine", "warmup_epochs": 0}, "early_stopping": None,
                           "grad_clip": 1., "time_budget_min": 0})
    model = VisualReasoningModel(text, head)
    return split, vocab, model, Trainer(model, TruthLoss(.5), vocab, cfg, torch.device("cpu"))


class TestGeometry(unittest.TestCase):
    def test_bootstrap_keeps_complete_pairs_and_counts_ties(self):
        perfect = bootstrap_pairs([.8, .2, .1, .9], [1, 0, 0, 1], [7, 7, 9, 9], repeats=20)
        self.assertEqual(perfect["pairs"], 2)
        self.assertEqual(perfect["auc_ci95"], [1., 1.])
        self.assertEqual(perfect["pair_accuracy_ci95"], [1., 1.])
        tied = bootstrap_pairs([0., 0.], [1, 0], [3, 3], repeats=20)
        self.assertEqual(tied["auc"], .5)
        self.assertEqual(tied["pair_accuracy_ci95"], [.5, .5])
        with self.assertRaisesRegex(ValueError, "complete pairs"):
            bootstrap_pairs([.5], [1], [0])

    def test_parser_removes_numbers_and_accepts_canonical_kinds(self):
        one = parse_geometric_caption("there is exactly one sphere behind the blue cube")
        two = parse_geometric_caption("exactly two spheres are behind the blue cube")
        self.assertEqual(one.filter_text, two.filter_text)
        self.assertEqual(one.reference_text, two.reference_text)
        self.assertEqual(one.filter_text, "a sphere")
        self.assertEqual(one.kind, COUNT)
        self.assertEqual(one.relation, 4)
        for caption, kind in (("the red cube is on the left side of the image", POSITION),
                              ("there is a red cube on the right side of the image", POSITION),
                              ("the red cube is behind the blue sphere", SPATIAL),
                              ("a red cube in front of the blue sphere", SPATIAL),
                              ("there is no cube to the left of the blue sphere", ABSENCE)):
            self.assertEqual(parse_geometric_caption(caption).kind, kind)
        self.assertIsNone(parse_geometric_caption("a red cube"))
        self.assertIsNone(parse_geometric_caption("there are more cubes than spheres"))

    def test_fixed_direction_exclusion_and_horizontal_flip(self):
        module = GeometricCountDistribution(8, 16, dim=8)
        xy = coordinate_grid(3, 5, torch.device("cpu"), torch.float32)
        ref = torch.zeros(4, 2)
        gates = module.relation_mask(xy, ref, torch.tensor([1, 2, 3, 4])).reshape(4, 3, 5)
        self.assertGreater(float(gates[0, 1, 0]), float(gates[0, 1, -1]))
        self.assertGreater(float(gates[1, 1, -1]), float(gates[1, 1, 0]))
        self.assertGreater(float(gates[2, -1, 2]), float(gates[2, 0, 2]))
        self.assertGreater(float(gates[3, 0, 2]), float(gates[3, -1, 2]))
        torch.testing.assert_close(gates[:, 1, 2], torch.zeros(4))
        torch.testing.assert_close(gates[0].flip(-1), gates[1])
        torch.testing.assert_close(module.relation_mask(xy, ref[:1], torch.tensor([0])), torch.ones(1, 15))

    def test_autocast_large_counts_have_finite_absence_loss_and_gradients(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                module = GeometricCountDistribution(8, 16, dim=8)
                with torch.no_grad():
                    module.log_gain.fill_(8.)
                image = torch.randn(2, 8, 3, 5)
                text = (torch.randn(2, 16), None, None)
                with torch.autocast("cpu", dtype=dtype):
                    outputs = module(image, text, text, torch.tensor([1, 4]), torch.arange(2))
                    loss = -outputs[0][:, 1].mean() + torch.nn.functional.binary_cross_entropy_with_logits(outputs[3], torch.ones(2))
                self.assertTrue(all(torch.isfinite(out).all() for out in outputs))
                loss.backward()
                self.assertTrue(torch.isfinite(module.query.weight.grad).all())

    def test_numbers_share_distribution_and_frozen_categories_are_identical(self):
        split, vocab, model, trainer = fixture()
        baseline = PairCountChoiceHead(8, 16, dim=8, queries=2, hidden=16, count_dim=8,
                                       count_selectors=2, count_hidden=16, train_count_only=False)
        baseline.load_state_dict({k: v for k, v in model.fusion.state_dict().items() if not k.startswith("geometry.")})
        base_model = VisualReasoningModel(model.text_encoder, baseline)
        base_trainer = Trainer(base_model, TruthLoss(.5), vocab, trainer.cfg, torch.device("cpu"))
        expected = base_trainer.predict(split)
        actual = trainer.predict(split)
        unchanged = np.ones(len(split), dtype=bool)
        unchanged[2:4] = False
        np.testing.assert_array_equal(actual[unchanged], expected[unchanged])
        rows, _, _, outputs = model.fusion._geometry_output
        for a, b in ((0, 1), (2, 3)):
            i, j = rows.tolist().index(a), rows.tolist().index(b)
            torch.testing.assert_close(outputs[0][i], outputs[0][j])
        torch.testing.assert_close(outputs[0].exp().sum(-1), torch.ones(len(rows)))

    def test_grounding_update_changes_only_geometry_and_uses_train_labels(self):
        split, _, model, trainer = fixture()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        history = trainer.fit(split, {}, {})
        self.assertTrue(np.isfinite(history[0]["loss"]))
        for key, value in before.items():
            if not key.startswith("fusion.geometry."):
                torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        self.assertFalse(torch.equal(before["fusion.geometry.query.weight"], model.fusion.geometry.query.weight))
        self.assertTrue(all(not p.requires_grad for p in model.fusion.count_distribution.parameters()))
        trainer.predict(split)
        loss = model.fusion.auxiliary_loss(torch.tensor(split.labels))
        flipped_loss = model.fusion.auxiliary_loss(1 - torch.tensor(split.labels))
        self.assertNotEqual(float(loss), float(flipped_loss))
        model.train()
        self.assertTrue(model.fusion.geometry.training)
        self.assertFalse(model.fusion.count_distribution.training)
        self.assertFalse(model.fusion.cell.training)

    def test_unlabelled_inference_and_metadata_and_row_order_invariance(self):
        split, _, _, trainer = fixture()
        original = trainer.predict(split)
        unlabelled = replace(split, labels=None, pair_ids=None, ids=split.ids + 100,
                             categories=np.array(["unused"] * len(split)))
        np.testing.assert_array_equal(trainer.predict(unlabelled), original)
        order = np.array([11, 0, 6, 2, 9, 7, 1, 10, 5, 3, 4, 8])
        permuted = replace(unlabelled, ids=unlabelled.ids[order], img_idx=unlabelled.img_idx[order],
                           captions=[unlabelled.captions[i] for i in order])
        np.testing.assert_allclose(trainer.predict(permuted), original[order], atol=1e-6)

    def test_warm_start_requires_stage_two_count_weights(self):
        _, _, model, _ = fixture()
        source = {k: v for k, v in model.state_dict().items() if not k.startswith("fusion.geometry.")}
        with patch("src.train.resolve_path") as root, patch("src.train.torch.load", return_value=source):
            root.return_value.glob.return_value = [Path("model.pt")]
            _warm_start(model, "stage2")
        stage1 = {k: v for k, v in source.items() if not k.startswith("fusion.count_distribution.")}
        with patch("src.train.resolve_path") as root, patch("src.train.torch.load", return_value=stage1):
            root.return_value.glob.return_value = [Path("model.pt")]
            with self.assertRaisesRegex(ValueError, "complete compatible baseline"):
                _warm_start(model, "stage1")

    def test_preset_is_isolated_and_never_reads_test(self):
        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
            baseline = compose(config_name="config", overrides=["experiment=pair_best_stage2"])
            new = compose(config_name="config", overrides=["experiment=pair_geometric_stage2"])
            ablation = compose(config_name="config", overrides=["experiment=pair_geometric_no_grounding"])
        self.assertEqual(baseline.model.fusion._target_, "src.models.count_distribution.PairCountChoiceHead")
        self.assertTrue(baseline.model.fusion.train_count_only)
        self.assertFalse(new.model.fusion.train_count_only)
        self.assertFalse(new.data.train_on_val)
        self.assertFalse(new.train.predict_test)
        self.assertLessEqual(new.train.time_budget_min, 14)
        self.assertEqual(ablation.model.fusion.grounding_weight, 0)

    def test_zero_grounding_weight_does_not_train_on_grounding_rows(self):
        _, _, model, _ = fixture()
        model.fusion.grounding_weight = 0
        self.assertTrue(model.fusion.supports_training_caption("exactly two spheres"))
        self.assertTrue(model.fusion.supports_training_caption("exactly two spheres behind the blue cube"))
        self.assertFalse(model.fusion.supports_training_caption("the red cube is on the left side of the image"))
        self.assertFalse(model.fusion.supports_training_caption("a red cube behind the blue sphere"))


if __name__ == "__main__":
    unittest.main()
