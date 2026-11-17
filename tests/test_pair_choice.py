"""Pair-choice contracts: token alignment, equivariance, row alignment and cache reuse."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.data import ArraySource, Split, eval_batches, train_batches
from src.features import cache_key
from src.losses import TruthLoss
from src.models.model import VisualReasoningModel
from src.models.pair_choice import PairChoiceHead, token_difference_mask
from src.models.text_encoders import GRUEncoder, Vocab
from src.train import _attach_sources
from src.training import Trainer
from src.twins import twin_normalize


REPO = Path(__file__).resolve().parents[1]


def fixture():
    # Twins are deliberately non-adjacent; their label orders differ.
    captions = ["a red cube", "a blue sphere on the left", "a blue cube", "a blue sphere on the right"]
    split = Split("val", np.array([51, 20, 31, 70]), ["a.jpg", "b.jpg"],
                  np.array([0, 1, 0, 1]), captions, np.array(["attribute", "position"] * 2),
                  np.array([1, 0, 0, 1]), np.array([13, 2, 13, 2]),
                  ArraySource(np.random.default_rng(7).normal(size=(2, 8, 3, 5)).astype(np.float32)))
    vocab = Vocab(captions)
    return split, vocab


def model_and_cfg(vocab, coords=True):
    torch.manual_seed(17)
    text = GRUEncoder(len(vocab), embed_dim=8, hidden_dim=8, dropout=0)
    head = PairChoiceHead(8, text.out_dim, dim=8, queries=2, hidden=16,
                          dropout=0, coord_channels=coords)
    model = VisualReasoningModel(text, head)
    cfg = OmegaConf.create({"amp": False, "batch_size": 2, "pair_sampler": True})
    return model, cfg


class TestTokenDifferences(unittest.TestCase):
    def test_substitution_and_padding(self):
        captions = ["a red metal cube", "a blue metal cube", "a cube", "a small cube"]
        vocab = Vocab(captions)
        tokens, lengths = vocab.encode_batch(captions)
        mask = token_difference_mask(tokens, lengths, torch.tensor([[0, 1], [2, 3]]))
        self.assertEqual(mask.tolist(), [[False, True, False, False], [False, True, False, False],
                                         [False, False, False, False], [False, True, False, False]])

    def test_alignment_is_symmetric_for_insertions_repetitions_and_swaps(self):
        for a, b in [("a cube to the left of a sphere", "a cube to the right of a sphere"),
                     ("there is a cube", "there is no small cube"),
                     ("a red cube and a blue sphere", "a blue cube and a red sphere"),
                     ("a cube cube", "a cube")]:
            with self.subTest(a=a, b=b):
                vocab = Vocab([a, b])
                t, n = vocab.encode_batch([a, b])
                forward = token_difference_mask(t, n, torch.tensor([[0, 1]]))
                backward = token_difference_mask(t, n, torch.tensor([[1, 0]]))
                self.assertTrue(torch.equal(forward, backward))
                self.assertTrue(forward.any())
                self.assertFalse(forward[t == 0].any())


class TestPairHead(unittest.TestCase):
    def test_equivariance_centering_and_normalization(self):
        split, vocab = fixture()
        for coords in (True, False):
            with self.subTest(coords=coords):
                model, _ = model_and_cfg(vocab, coords)
                model.eval()
                tokens, lengths = vocab.encode_batch(split.captions)
                pairs = torch.tensor([[0, 2], [1, 3]])
                diff = token_difference_mask(tokens, lengths, pairs)
                image = split.source.get(split.img_idx)
                with torch.no_grad():
                    logits = model(image, tokens, lengths, pairs, diff)
                    swapped = model(image, tokens, lengths, pairs.flip(1), diff)
                    padded = model(image, torch.nn.functional.pad(tokens, (0, 3)), lengths, pairs,
                                   torch.nn.functional.pad(diff, (0, 3)))
                torch.testing.assert_close(logits, swapped)
                torch.testing.assert_close(logits, padded)
                torch.testing.assert_close(logits[pairs].sum(1), torch.zeros(2))
                margin = logits[pairs[:, 0]] - logits[pairs[:, 1]]
                probs = twin_normalize(logits.numpy(), pairs.numpy())
                np.testing.assert_allclose(probs[pairs[:, 0]], margin.sigmoid().numpy(), rtol=1e-6)
                self.assertEqual(logits.shape, (4,))

    def test_other_caption_and_spatial_cells_affect_scores_and_gradients(self):
        split, vocab = fixture()
        model, _ = model_and_cfg(vocab)
        model.eval()
        pairs = torch.tensor([[0, 2], [1, 3]])
        tokens, lengths = vocab.encode_batch(split.captions)
        image = split.source.get(split.img_idx).requires_grad_()
        diff = token_difference_mask(tokens, lengths, pairs)
        logits = model(image, tokens, lengths, pairs, diff)
        loss = TruthLoss(pair_weight=.5)(logits, torch.tensor(split.labels), torch.tensor([[0, 2], [3, 1]]))
        loss.backward()
        for name, parameter in [("embedding", model.text_encoder.embed.weight),
                                ("difference query", model.fusion.difference_query.weight),
                                ("cells", model.fusion.cell[0].weight), ("coordinates", model.fusion.xy.weight)]:
            with self.subTest(name=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertGreater(float(image.grad.abs().sum()), 0)
        changed = tokens.clone()
        changed[2, 1] = vocab.stoi["sphere"]
        other = model(image, changed, lengths, pairs, token_difference_mask(changed, lengths, pairs))
        self.assertNotEqual(float(logits[0].detach()), float(other[0].detach()))

    def test_identical_twins_are_finite_ties_and_metadata_is_required(self):
        vocab = Vocab(["a cube"])
        model, _ = model_and_cfg(vocab)
        model.eval()
        tokens, lengths = vocab.encode_batch(["a cube", "a cube"])
        image = torch.randn(1, 8, 2, 3).repeat(2, 1, 1, 1)
        pairs = torch.tensor([[0, 1]])
        with torch.no_grad():
            logits = model(image, tokens, lengths, pairs, token_difference_mask(tokens, lengths, pairs))
        torch.testing.assert_close(logits, torch.zeros(2))
        with self.assertRaisesRegex(ValueError, "requires twin_index"):
            model(image, tokens, lengths)

    def test_autocast_scatter_and_backward(self):
        split, vocab = fixture()
        pairs = torch.tensor([[0, 2], [1, 3]])
        tokens, lengths = vocab.encode_batch(split.captions)
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                model, _ = model_and_cfg(vocab)
                with torch.autocast("cpu", dtype=dtype):
                    logits = model(split.source.get(split.img_idx), tokens, lengths, pairs,
                                   token_difference_mask(tokens, lengths, pairs))
                    loss = TruthLoss(.5)(logits.float(), torch.tensor(split.labels), torch.tensor([[0, 2], [3, 1]]))
                loss.backward()
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.isfinite(model.fusion.out[-1].weight.grad).all())


class TestPairBatches(unittest.TestCase):
    def test_prediction_uses_test_metadata_and_restores_original_rows(self):
        split, vocab = fixture()
        model, cfg = model_and_cfg(vocab)
        trainer = Trainer(model, TruthLoss(), vocab, cfg, torch.device("cpu"))
        prediction = trainer.predict(split)
        # No labels or pair ids are required for finding inference partners.
        test = replace(split, labels=None, pair_ids=None)
        np.testing.assert_array_equal(prediction, trainer.predict(test))
        np.testing.assert_allclose(prediction, trainer.predict(test, batch_size=4), atol=1e-6)
        permutation = np.array([3, 0, 1, 2])
        reordered = replace(test, ids=test.ids[permutation], img_idx=test.img_idx[permutation],
                            captions=[test.captions[i] for i in permutation], categories=test.categories[permutation])
        np.testing.assert_allclose(trainer.predict(reordered), prediction[permutation], atol=1e-6)
        for batch in eval_batches(test, 2, pairwise=True):
            self.assertEqual(batch["img_idx"][0], batch["img_idx"][1])
            self.assertNotIn("pair_index", batch)

    def test_train_model_pair_order_does_not_use_labels(self):
        split, _ = fixture()
        kwargs = dict(batch_size=4, aug={}, pair_sampler=True)
        a = next(train_batches(split, rng=np.random.default_rng(3), **kwargs))
        b = next(train_batches(replace(split, labels=1 - split.labels), rng=np.random.default_rng(3), **kwargs))
        np.testing.assert_array_equal(a["rows"], b["rows"])
        torch.testing.assert_close(a["twin_index"], b["twin_index"])
        self.assertFalse(torch.equal(a["pair_index"], b["pair_index"]))

    def test_incomplete_pairs_odd_batches_and_disabled_sampler_fail(self):
        split, vocab = fixture()
        for batch_size in (1, 3):
            with self.assertRaisesRegex(ValueError, "even batch_size"):
                list(eval_batches(split, batch_size, pairwise=True))
        missing = replace(split, ids=split.ids[:3], img_idx=split.img_idx[:3],
                          captions=split.captions[:3], categories=split.categories[:3])
        with self.assertRaisesRegex(ValueError, "twin for every row"):
            list(eval_batches(missing, 4, pairwise=True))
        model, cfg = model_and_cfg(vocab)
        cfg.pair_sampler = False
        with self.assertRaisesRegex(ValueError, "pair_sampler=true"):
            Trainer(model, TruthLoss(), vocab, cfg, torch.device("cpu"))


class TestReadOnlyCache(unittest.TestCase):
    def test_subset_and_flip_reuse_cache_without_encoder(self):
        with tempfile.TemporaryDirectory() as td:
            with initialize_config_dir(config_dir=str(REPO / "configs")):
                cfg = compose(config_name="config", overrides=[f"paths.features_dir={td}"])
            root = Path(td) / cache_key(cfg.model.image_encoder, cfg.data)
            root.mkdir()
            names = ["unused.jpg", "b.jpg", "a.jpg"]
            features = np.arange(3 * 8 * 3 * 5, dtype=np.float16).reshape(3, 8, 3, 5)
            for suffix, array in [("train", features), ("train_flip", -features)]:
                np.save(root / f"{suffix}.npy", array)
                (root / f"{suffix}.json").write_text(json.dumps(names))
            split, _ = fixture()
            with patch("src.train.instantiate", side_effect=AssertionError("encoder must not be constructed")):
                encoder, channels, mode = _attach_sources(cfg, {"train": split}, torch.device("cpu"), True)
            self.assertIsNone(encoder)
            self.assertEqual((channels, mode), (8, "features"))
            self.assertEqual(len(split.source), 2)
            np.testing.assert_array_equal(split.source.get(np.array([0, 1]), np.array([False, True])).numpy(),
                                          np.stack([features[2], -features[1]]))
            self.assertFalse(split.source.array.flags.writeable)
            split.source.array._mmap.close()
            split.source.flipped._mmap.close()


class TestPairCLI(unittest.TestCase):
    def test_train_val_and_label_free_test_inference(self):
        from test_pipeline import build_data_dir, run_train

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            expected = build_data_dir(tmp / "data", with_test=True)
            proc = run_train(tmp, "pair_cli", ["experiment=pair_choice", "train.predict_test=true",
                "train.early_stopping=null", "model.text_encoder.embed_dim=8", "model.text_encoder.hidden_dim=8",
                "model.fusion.dim=8", "model.fusion.queries=2", "model.fusion.hidden=8", "data.image_size=[32,48]"])
            self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
            out = tmp / "out"
            results = json.loads((out / "results.json").read_text())
            self.assertEqual(results["architecture"].split("/")[-1], "PairChoiceHead")
            self.assertAlmostEqual(results["val"]["auc"], results["val_twin_norm"]["auc"])
            self.assertAlmostEqual(results["val"]["accuracy"], results["val"]["pair_accuracy"])
            np.testing.assert_array_equal(np.load(out / "val_ids.npy"), expected["val_ids"])
            self.assertEqual(len(np.load(out / "test_logits.npy")), 4)
            import csv
            with (tmp / "subs/pair_cli.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["id"]) for row in rows], expected["sample_ids"])


if __name__ == "__main__":
    unittest.main()
