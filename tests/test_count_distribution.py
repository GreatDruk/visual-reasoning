"""Numeric parsing, count supervision, routing and label-free pair contracts."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.data import ArraySource, Split
from src.losses import TruthLoss
from src.models.count_distribution import (
    CountDistribution, PairCountChoiceHead, encode_count_inputs, parse_count_caption,
)
from src.models.model import VisualReasoningModel
from src.models.pair_choice import PairChoiceHead, token_difference_mask
from src.models.text_encoders import GRUEncoder, Vocab
from src.training import Trainer
from src.train import _warm_start
from src.twins import twin_normalize


def fixture():
    captions = ["exactly one red cube", "there are exactly two spheres to the left of the blue cube",
                "a red cube", "exactly three red cubes",
                "there is exactly one sphere to the left of the blue cube", "a blue cube"]
    split = Split("train", np.arange(6), ["a.jpg", "b.jpg", "c.jpg"], np.array([0, 1, 2, 0, 1, 2]),
                  captions, np.array(["counting", "spatial_counting", "attribute"] * 2),
                  np.array([1, 0, 1, 0, 1, 0]), np.array([10, 20, 30, 10, 20, 30]),
                  ArraySource(np.random.default_rng(3).normal(size=(3, 8, 3, 5)).astype(np.float32)))
    vocab = Vocab(captions)
    torch.manual_seed(11)
    text = GRUEncoder(len(vocab), embed_dim=8, hidden_dim=8, dropout=.4)
    head = PairCountChoiceHead(8, text.out_dim, dim=8, queries=2, hidden=16,
                               count_dim=8, count_selectors=2, count_hidden=16, count_dropout=0)
    return split, vocab, VisualReasoningModel(text, head)


class TestCountParser(unittest.TestCase):
    def test_number_does_not_enter_filter_even_via_inflection(self):
        for word, n in zip("no zero one two three four five six seven eight nine ten".split(), [0, *range(11)]):
            value = parse_count_caption(f"there {'is' if n == 1 else 'are'} exactly {word} red {'cube' if n == 1 else 'cubes'}")
            self.assertEqual(value.number, n)
            self.assertEqual(value.filter_text, "exactly <unk> red cube")
            self.assertEqual(value.relation, 0)
        self.assertIsNone(parse_count_caption("there is no red cube"))
        self.assertIsNone(parse_count_caption("exactly eleven cubes"))
        self.assertIsNone(parse_count_caption("there are more red cubes than blue cubes"))

    def test_spatial_filter_reference_and_surface_forms(self):
        forms = ["there are exactly two spheres to the left of the blue cube",
                 "exactly two spheres are to the left of the blue cube",
                 "to the left of the blue cube there are exactly two spheres"]
        parsed = [parse_count_caption(c) for c in forms]
        self.assertTrue(all(p == parsed[0] for p in parsed))
        self.assertEqual(parsed[0].filter_text, "exactly <unk> sphere")
        self.assertEqual(parsed[0].reference_text, "a blue cube")
        self.assertEqual(parsed[0].relation, 1)
        self.assertEqual(parse_count_caption("exactly two spheres are in the scene").number, 2)
        other = parse_count_caption("there is exactly one sphere to the right of the red cube")
        self.assertNotEqual(parsed[0].relation, other.relation)
        self.assertNotEqual(parsed[0].reference_text, other.reference_text)


class TestCountHead(unittest.TestCase):
    def forward_fixture(self, model, split, vocab, pairs=None, padding=0):
        if pairs is None:
            pairs = torch.tensor([[0, 3], [1, 4], [2, 5]])
        tokens, lengths = vocab.encode_batch(split.captions)
        tokens = F.pad(tokens, (0, padding))
        diff = token_difference_mask(tokens, lengths, pairs)
        extra = encode_count_inputs(split.captions, vocab, torch.device("cpu"))
        logits = model(split.source.get(split.img_idx), tokens, lengths, pairs, diff, count_inputs=extra)
        return logits, pairs

    def test_distribution_is_shared_by_different_numbers(self):
        split, vocab, model = fixture()
        model.eval()
        with torch.no_grad():
            self.forward_fixture(model, split, vocab)
        rows, _, logp = model.fusion._count_output
        self.assertEqual(rows.tolist(), [0, 1, 3, 4])
        torch.testing.assert_close(logp[0], logp[2])
        torch.testing.assert_close(logp[1], logp[3])
        torch.testing.assert_close(logp.exp().sum(1), torch.ones(4))
        self.assertEqual(logp.shape, (4, 11))

    def test_centering_swaps_padding_and_twin_normalize(self):
        split, vocab, model = fixture()
        model.eval()
        with torch.no_grad():
            logits, pairs = self.forward_fixture(model, split, vocab)
            swapped, _ = self.forward_fixture(model, split, vocab, pairs.flip(1))
            padded, _ = self.forward_fixture(model, split, vocab, padding=4)
        torch.testing.assert_close(logits, swapped)
        torch.testing.assert_close(logits, padded)
        torch.testing.assert_close(logits[pairs].sum(1), torch.zeros(3))
        probs = twin_normalize(logits.numpy(), pairs.numpy())
        np.testing.assert_allclose(probs[pairs[:, 0]], (logits[pairs[:, 0]] - logits[pairs[:, 1]]).sigmoid().numpy(), rtol=1e-6)

    def test_frozen_base_route_and_state_stay_unchanged_after_update(self):
        split, vocab, model = fixture()
        base = PairChoiceHead(8, 16, dim=8, queries=2, hidden=16)
        base.load_state_dict({k: v for k, v in model.fusion.state_dict().items() if k in base.state_dict()})
        base_model = VisualReasoningModel(model.text_encoder, base).eval()
        tokens, lengths = vocab.encode_batch(split.captions)
        pairs = torch.tensor([[0, 3], [1, 4], [2, 5]])
        image = split.source.get(split.img_idx)
        diff = token_difference_mask(tokens, lengths, pairs)
        with torch.no_grad():
            expected = base_model(image, tokens, lengths, pairs, diff)
        original = {k: v.clone() for k, v in model.state_dict().items() if "count_distribution" not in k}
        model.train()
        self.assertFalse(model.text_encoder.training)
        self.assertFalse(model.fusion.cell.training)
        self.assertTrue(model.fusion.count_distribution.training)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        logits, _ = self.forward_fixture(model, split, vocab)
        loss = TruthLoss(.5)(logits, torch.tensor(split.labels), torch.tensor([[0, 3], [4, 1], [2, 5]]))
        loss = loss + model.fusion.auxiliary_loss(torch.tensor(split.labels))
        loss.backward()
        for name in ("cell.0.weight", "filter_query.weight", "reference_query.weight", "directions.weight", "readout.3.weight"):
            grad = dict(model.fusion.count_distribution.named_parameters())[name].grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(float(grad.abs().sum()), 0, name)
        opt.step()
        for key, value in original.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        model.eval()
        with torch.no_grad():
            actual, _ = self.forward_fixture(model, split, vocab)
        torch.testing.assert_close(actual[[2, 5]], expected[[2, 5]], rtol=0, atol=0)

    def test_train_auxiliary_matches_nll_and_false_count_penalty(self):
        split, vocab, model = fixture()
        self.forward_fixture(model, split, vocab)
        rows, numbers, logp = model.fusion._count_output
        labels = torch.tensor(split.labels)
        lp = logp[torch.arange(len(rows)), numbers]
        target = labels[rows].float()
        expected = .5 * (-target * lp - (1 - target) * .25 * torch.log1p(-lp.exp())).mean()
        torch.testing.assert_close(model.fusion.auxiliary_loss(labels), expected)
        self.assertNotEqual(float(expected.detach()), float(model.fusion.auxiliary_loss(1 - labels).detach()))

    def test_false_count_penalty_keeps_gradient_when_wrong_and_confident(self):
        _, _, model = fixture()
        logits = torch.tensor([[100.] + [0.] * 10], requires_grad=True)
        model.fusion._count_output = (torch.tensor([0]), torch.tensor([0]), logits.log_softmax(-1))
        loss = model.fusion.auxiliary_loss(torch.tensor([0]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad[0, 0]), 0)

    def test_label_free_prediction_row_order_and_real_trainer_update(self):
        split, vocab, model = fixture()
        cfg = OmegaConf.create({"amp": False, "batch_size": 6, "pair_sampler": True,
            "optimizer": {"lr": .001, "weight_decay": .01}, "epochs": 1,
            "scheduler": {"name": "cosine", "warmup_epochs": 0}, "early_stopping": None,
            "grad_clip": 1., "time_budget_min": 0})
        trainer = Trainer(model, TruthLoss(.5), vocab, cfg, torch.device("cpu"))
        prediction = trainer.predict(split)
        test = replace(split, labels=None, pair_ids=None)
        np.testing.assert_array_equal(trainer.predict(test), prediction)
        perm = np.array([5, 1, 0, 3, 2, 4])
        reordered = replace(test, ids=test.ids[perm], img_idx=test.img_idx[perm],
                            captions=[test.captions[i] for i in perm], categories=test.categories[perm])
        np.testing.assert_allclose(trainer.predict(reordered), prediction[perm], atol=1e-6)
        before = model.fusion.count_distribution.readout[-1].weight.detach().clone()
        history = trainer.fit(split, {}, {})
        self.assertTrue(np.isfinite(history[0]["loss"]))
        self.assertFalse(torch.equal(before, model.fusion.count_distribution.readout[-1].weight))

    def test_reference_and_relation_affect_distribution(self):
        torch.manual_seed(1)
        module = CountDistribution(8, 16, dim=8, selectors=2, hidden=16, dropout=0).eval()
        image = torch.randn(1, 8, 3, 5)
        text = (torch.randn(1, 16), None, None)
        ref = (torch.randn(1, 16), None, None)
        with torch.no_grad():
            a = module(image, text, ref, torch.tensor([1]))
            b = module(image, text, ref, torch.tensor([2]))
            c = module(image, text, (-ref[0], None, None), torch.tensor([1]))
        self.assertFalse(torch.allclose(a, b))
        self.assertFalse(torch.allclose(a, c))

    def test_autocast_count_margin_and_auxiliary_backward(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                split, vocab, model = fixture()
                with torch.autocast("cpu", dtype=dtype):
                    logits, _ = self.forward_fixture(model, split, vocab)
                    loss = TruthLoss(.5)(logits.float(), torch.tensor(split.labels),
                                          torch.tensor([[0, 3], [4, 1], [2, 5]]))
                    loss = loss + model.fusion.auxiliary_loss(torch.tensor(split.labels))
                loss.backward()
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.isfinite(model.fusion.count_distribution.readout[-1].weight.grad).all())

    def test_uncached_image_encoder_is_frozen_and_kept_in_eval(self):
        _, _, model = fixture()
        encoder = torch.nn.Sequential(torch.nn.Conv2d(8, 8, 1), torch.nn.BatchNorm2d(8))
        model = VisualReasoningModel(model.text_encoder, model.fusion, image_encoder=encoder)
        model.train()
        self.assertFalse(encoder.training)
        self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))

    def test_config_keeps_validation_and_test_out_of_training(self):
        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
            cfg = compose(config_name="config", overrides=["experiment=pair_choice_count_siglip"])
        self.assertFalse(cfg.data.train_on_val)
        self.assertFalse(cfg.train.predict_test)
        self.assertFalse(cfg.train.eval.paraphrases)
        self.assertFalse(cfg.save_config)
        self.assertTrue(cfg.model.fusion.freeze_base)
        self.assertEqual(cfg.model.image_encoder.out_layer, 8)
        self.assertIsNone(cfg.model.fusion.input_pool)

    def test_frozen_warm_start_rejects_partial_baseline(self):
        _, _, model = fixture()
        with patch("src.train.resolve_path") as root, patch("src.train.torch.load", return_value={}):
            root.return_value.glob.return_value = [Path("model.pt")]
            with self.assertRaisesRegex(ValueError, "complete compatible baseline"):
                _warm_start(model, "incomplete")


if __name__ == "__main__":
    unittest.main()
