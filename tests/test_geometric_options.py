"""Opt-in existence supervision and metadata-free spatial/comparison inference."""

from dataclasses import replace
from pathlib import Path
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from src.data import ArraySource, eval_batches
from src.geometric_ablation import prediction_statistics
from src.models.count_distribution import parse_count_caption
from src.models.geometric_count import (
    ABSENCE, BINDING, EXISTENCE, NONEXISTENCE, SPATIAL, caption_pair_key, comparison_log_probability,
    encode_comparison_inputs, encode_geometric_inputs, parse_comparison_caption,
    parse_geometric_caption,
)
from test_geometric_count import fixture


class TestGeometricOptions(unittest.TestCase):
    def test_existence_parser_is_opt_in_and_binding_keeps_both_filters(self):
        for caption, kind in (("a small red metal cube", EXISTENCE),
                              ("there is a red cube", EXISTENCE),
                              ("there is no red cube", NONEXISTENCE),
                              ("a red cube and a blue sphere", BINDING)):
            self.assertIsNone(parse_geometric_caption(caption))
            self.assertEqual(parse_geometric_caption(caption, True).kind, kind)
        parsed = parse_geometric_caption("there is a red cube and a blue sphere", True)
        self.assertEqual((parsed.filter_text, parsed.reference_text), ("a red cube", "a blue sphere"))
        for caption in ("a red cube or a blue sphere", "there is no red cube and a blue sphere",
                        "a red cube beside the sphere", "more cubes than spheres"):
            self.assertIsNone(parse_geometric_caption(caption, True))
        _, vocab, _, _ = fixture()
        encoded = encode_geometric_inputs(["there is a red cube and a blue sphere"], vocab,
                                          torch.device("cpu"), include_existence=True)
        self.assertEqual(encoded["rows"].tolist(), [0, 0])
        self.assertEqual(encoded["kinds"].tolist(), [BINDING, BINDING])
        self.assertEqual(encoded["relations"].tolist(), [0, 0])

    def test_binding_false_label_supervises_conjunction_not_individual_absence(self):
        _, _, model, _ = fixture()
        head = model.fusion
        head.existence_weight = 1.
        # Each component has only 0/1 counts: P(exists) = .9 and .2.
        logits = torch.tensor([[math.log(.1), math.log(.9)] + [-100.] * 9,
                               [math.log(.8), math.log(.2)] + [-100.] * 9], requires_grad=True)
        logp = logits.log_softmax(-1)
        dummy = torch.zeros(2)
        head._geometry_output = (torch.tensor([0, 0]), torch.zeros(2, dtype=torch.long),
                                 torch.full((2,), BINDING), (logp, dummy, dummy, dummy))
        true_loss = head.auxiliary_loss(torch.tensor([1]))
        false_loss = head.auxiliary_loss(torch.tensor([0]))
        self.assertAlmostEqual(float(true_loss.detach()), -math.log(.9 * .2), places=5)
        self.assertAlmostEqual(float(false_loss.detach()), -math.log(1 - .9 * .2), places=5)
        false_loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_presence_and_absence_use_complementary_count_mass(self):
        _, _, model, _ = fixture()
        head = model.fusion
        head.existence_weight = 1.
        logp = torch.tensor([[.7, .2, .1] + [1e-20] * 8]).log()
        dummy = torch.zeros(1)
        for kind, expected in ((EXISTENCE, .3), (NONEXISTENCE, .7)):
            head._geometry_output = (torch.tensor([0]), torch.tensor([0]), torch.tensor([kind]),
                                     (logp, dummy, dummy, dummy))
            self.assertAlmostEqual(float(head.auxiliary_loss(torch.tensor([1]))), -math.log(expected), places=5)

    def test_saturated_existence_and_conjunction_still_have_finite_gradients(self):
        _, _, model, _ = fixture()
        model.fusion.existence_weight = 1.
        for kind, rows in ((EXISTENCE, [0]), (NONEXISTENCE, [0]), (BINDING, [0, 0])):
            logits = torch.tensor([[-100.] + [0.] * 10] * len(rows), requires_grad=True)
            dummy = torch.zeros(len(rows))
            model.fusion._geometry_output = (torch.tensor(rows), torch.zeros(len(rows), dtype=torch.long),
                torch.full((len(rows),), kind), (logits.log_softmax(-1), dummy, dummy, dummy))
            loss = model.fusion.auxiliary_loss(torch.tensor([1 if kind == NONEXISTENCE else 0]))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertGreater(float(logits.grad.abs().sum()), .1)

    def test_comparison_parser_reuses_exactly_n_filter_encoding(self):
        examples = [("there are more red objects than blue cubes", "more"),
                    ("there are fewer cubes than spheres", "fewer"),
                    ("there are as many red objects as brown objects", "equal")]
        for caption, operator in examples:
            self.assertEqual(parse_comparison_caption(caption).operator, operator)
        parsed = parse_comparison_caption(examples[0][0])
        self.assertEqual(parsed.left, parse_count_caption("exactly two red objects").filter_text)
        self.assertEqual(parsed.right, parse_count_caption("exactly one blue cube").filter_text)
        self.assertIsNone(parse_comparison_caption("there are more cubes than spheres behind the red cube"))
        self.assertIsNone(parse_comparison_caption("there are more unknown cubes than spheres"))
        _, vocab, _, _ = fixture()
        captions = ["there are more red cubes than blue spheres"]
        stage2 = encode_comparison_inputs(captions, vocab, torch.device("cpu"))
        geometry = encode_comparison_inputs(captions, vocab, torch.device("cpu"), source="geometry")
        expected, _ = vocab.encode_batch(["a red cube", "a blue sphere"])
        torch.testing.assert_close(geometry["tokens"], expected)
        self.assertNotEqual(stage2["lengths"].tolist(), geometry["lengths"].tolist())

    def test_joint_count_probabilities_match_enumeration_and_symmetry(self):
        left = torch.tensor([[.2, .3, .5] + [1e-20] * 8]).repeat(3, 1)
        right = torch.tensor([[.6, .1, .3] + [1e-20] * 8]).repeat(3, 1)
        probabilities = comparison_log_probability(left.log(), right.log(), torch.arange(3)).exp()
        torch.testing.assert_close(probabilities, torch.tensor([.53, .17, .30]))
        torch.testing.assert_close(probabilities.sum(), torch.tensor(1.))
        reverse = comparison_log_probability(right.log(), left.log(), torch.tensor([1, 0, 2])).exp()
        torch.testing.assert_close(probabilities, reverse)
        extreme = torch.nn.functional.log_softmax(torch.randn(3, 11) * 1000, -1).requires_grad_()
        comparison_log_probability(extreme, extreme.flip(-1), torch.arange(3)).sum().backward()
        self.assertTrue(torch.isfinite(extreme.grad).all())

    def test_comparison_diagnostics_keep_complete_pairs_and_separate_true_equality(self):
        split, _, _, _ = fixture()
        split = replace(split, categories=np.array(["comparison"] * 12), captions=[
            "there are more cubes than spheres", "there are fewer cubes than spheres",
            "there are as many red objects as blue objects", "there are more red objects than blue objects",
            "there are fewer cubes than spheres", "there are as many cubes as spheres"] * 2)
        stats = prediction_statistics(np.tile([2., -2.], 6), split, repeats=20)
        for key, pairs in (("overall", 6), ("comparison_equality_pairs", 4),
                           ("comparison_more_fewer_pairs", 2), ("comparison_true_equal_pairs", 2),
                           ("comparison_false_equal_pairs", 2)):
            self.assertEqual(stats[key]["pairs"], pairs)
            self.assertEqual(stats[key]["auc_ci95"], [1., 1.])

    def test_comparison_route_changes_only_selected_complete_pairs(self):
        _, _, model, _ = fixture()
        head = model.fusion.eval()
        base = torch.tensor([.5, -.5, .7, -.7, .2, -.2])
        pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])
        rows, ops = torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 0])
        distributions = torch.randn(8, 11).log_softmax(-1)
        comparison = (rows, ops, (None, None))
        with patch.object(head.count_distribution, "forward", return_value=distributions):
            head.comparison_route = "off"
            torch.testing.assert_close(head._route_comparison(base, torch.randn(3, 8, 3, 5), pairs, comparison), base)
            head.comparison_route = "equality"
            result = head._route_comparison(base, torch.randn(3, 8, 3, 5), pairs, comparison)
            torch.testing.assert_close(result[[0, 1, 4, 5]], base[[0, 1, 4, 5]], rtol=0, atol=0)
            self.assertNotEqual(float(result[2]), float(base[2]))
            torch.testing.assert_close(result[pairs].sum(1), torch.zeros(3))
            head.comparison_route = "all"
            result_all = head._route_comparison(base, torch.randn(3, 8, 3, 5), pairs, comparison)
            self.assertNotEqual(float(result_all[0]), float(base[0]))
            head.comparison_source = "geometry"
            dummy = torch.zeros(8)
            with patch.object(head.geometry, "forward", return_value=(distributions, dummy, dummy, dummy)) as geometric:
                result_geometry = head._route_comparison(base, torch.randn(3, 8, 3, 5), pairs, comparison)
            torch.testing.assert_close(result_geometry, result_all)
            torch.testing.assert_close(geometric.call_args.args[-1], torch.tensor([0, 0, 1, 1, 0, 0, 1, 1]))

    def test_spatial_route_uses_count_mass_and_preserves_base_blend(self):
        _, _, model, _ = fixture()
        head = model.fusion.eval()
        base = torch.tensor([.5, -.5])
        logp = torch.tensor([[.2, .8] + [1e-20] * 9, [.7, .3] + [1e-20] * 9]).log()
        dummy = torch.zeros(2)
        text = (torch.zeros(2, 16), None, None)
        for kind, event in ((SPATIAL, [.8, .3]), (ABSENCE, [.2, .7])):
            geometry = (torch.arange(2), torch.zeros(2, dtype=torch.long), torch.full((2,), 4),
                        torch.full((2,), kind), text, text)
            with patch("src.models.count_distribution.PairCountChoiceHead.forward", return_value=base), \
                 patch.object(head.geometry, "forward", return_value=(logp, dummy, dummy, dummy)):
                head.route_spatial = False
                unchanged = head(torch.zeros(1, 8, 3, 5), text, torch.tensor([[0, 1]]), None, geometry_text=geometry)
                torch.testing.assert_close(unchanged, base, rtol=0, atol=0)
                head.route_spatial = True
                actual = head(torch.zeros(1, 8, 3, 5), text, torch.tensor([[0, 1]]), None, geometry_text=geometry)
                expected = (math.log(event[0]) - math.log(event[1]) + 1) / 2
                torch.testing.assert_close(actual, torch.tensor([expected, -expected]))
                # Option 2 changes inference only, including when its preset is
                # used for a new fit rather than evaluating a saved checkpoint.
                head.train()
                training_result = head(torch.zeros(1, 8, 3, 5), text, torch.tensor([[0, 1]]), None,
                                       geometry_text=geometry)
                torch.testing.assert_close(training_result, torch.zeros(2))
                head.eval()

    def test_opt_in_routes_ignore_metadata_even_with_multiple_pairs_per_image(self):
        split, _, model, trainer = fixture()
        captions = ["a red cube", "a blue cube", "there is a red cube", "there is no red cube",
                    "a red cube and a blue sphere", "a blue cube and a red sphere",
                    "a red cube behind the blue sphere", "a red cube in front of the blue sphere",
                    "there is no cube behind the blue sphere", "there is no cube in front of the blue sphere",
                    "there are as many red cubes as blue spheres", "there are more red cubes than blue spheres"]
        model.fusion.existence_weight = .5
        model.fusion.caption_pairing = True
        model.fusion.route_spatial = True
        model.fusion.comparison_route = "equality"
        split = replace(split, captions=captions, img_idx=np.zeros(12, dtype=np.int64), image_names=["same.jpg"],
                        source=ArraySource(np.random.default_rng(3).normal(size=(1, 8, 3, 5)).astype(np.float32)))
        unlabelled = replace(split, labels=None, pair_ids=None, ids=split.ids + 99,
                             categories=np.array([f"wrong-{i}" for i in range(12)]))
        order = np.random.default_rng(5).permutation(12)
        permuted = replace(unlabelled, ids=unlabelled.ids[order], img_idx=unlabelled.img_idx[order],
                           categories=unlabelled.categories[order], captions=[captions[i] for i in order])
        for source in ("stage2", "geometry"):
            model.fusion.comparison_source = source
            actual = trainer.predict(split)
            np.testing.assert_array_equal(trainer.predict(unlabelled), actual)
            np.testing.assert_allclose(trainer.predict(permuted), actual[order], atol=1e-6)
        keys = [caption_pair_key(c) for c in captions]
        batch = next(eval_batches(unlabelled, 12, pairwise=True, pairing_keys=keys))
        self.assertNotIn("labels", batch)
        self.assertNotIn("pair_index", batch)

    def test_new_existence_rows_train_only_shared_geometry(self):
        split, _, model, trainer = fixture()
        split.captions = ["there is a red cube and a blue sphere", "there is a blue cube and a red sphere"] * 6
        model.fusion.existence_weight = .5
        model.fusion.caption_pairing = True
        before = {k: v.clone() for k, v in model.state_dict().items()}
        history = trainer.fit(split, {}, {})
        self.assertTrue(np.isfinite(history[0]["loss"]))
        for key, value in before.items():
            if not key.startswith("fusion.geometry."):
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        self.assertFalse(torch.equal(model.fusion.geometry.query.weight, before["fusion.geometry.query.weight"]))

    def test_presets_keep_old_defaults_and_options_independent(self):
        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
            configs = [compose(config_name="config", overrides=[f"experiment={name}"]) for name in
                       ["pair_geometric_stage2", "pair_geometric_existence", "pair_geometric_spatial", "pair_geometric_comparison"]]
        self.assertNotIn("existence_weight", configs[0].model.fusion)
        self.assertNotIn("route_spatial", configs[0].model.fusion)
        self.assertGreater(configs[1].model.fusion.existence_weight, 0)
        self.assertFalse(configs[1].model.fusion.route_spatial)
        self.assertTrue(configs[2].model.fusion.route_spatial)
        self.assertEqual(configs[3].model.fusion.comparison_route, "all")
        self.assertEqual(configs[3].model.fusion.comparison_source, "geometry")
        for cfg in configs:
            self.assertFalse(cfg.data.train_on_val)
            self.assertFalse(cfg.train.predict_test)
            self.assertLessEqual(cfg.train.time_budget_min, 14)


if __name__ == "__main__":
    unittest.main()
