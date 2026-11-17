"""Regression tests for the pipeline after plan steps 1-3 (stdlib unittest, no pytest).

Run from the repository root:
    uv run python -m unittest discover -s tests -p 'test_*.py'

Everything runs on CPU with random-initialized encoders (weights=null) and real
temporary CSV/JPEG fixtures; no network access is needed and no repository data is read.
CLI integration tests invoke the real entrypoint `python -m src.train`.
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data import ArraySource, load_split, pair_groups, sample_pairs_by_category, subset, train_batches  # noqa: E402
from src.features import cache_features, cache_key, load_images, preprocess  # noqa: E402
from src.losses import TruthLoss  # noqa: E402
from src.models.image_encoders import TorchvisionEncoder  # noqa: E402


# --------------------------------------------------------------------------- fixtures

def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_jpeg(path, seed):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)).save(path, format="JPEG")


LABELLED_HEADER = ["id", "image", "caption", "level", "category", "pair_id", "label"]
TEST_HEADER = ["id", "image", "caption", "level", "category"]


def build_data_dir(root, with_test):
    """Temporary fixture: 3 train pairs, 2 val pairs, optionally 4 test rows.

    "flurble" appears only in val captions, "zorblex" only in test captions; sample
    submission ids are deliberately shuffled against the test.csv row order.
    """
    root = Path(root)
    for split in ("train", "val") + (("test",) if with_test else ()):
        (root / split).mkdir(parents=True)
    for i in range(3):
        _write_jpeg(root / "train" / f"tr{i}.jpg", seed=i)
    for i in range(2):
        _write_jpeg(root / "val" / f"v{i}.jpg", seed=10 + i)

    train_rows = [
        (100, "tr0.jpg", "there is a red cube", 1, "counting", 10, 1),
        (101, "tr1.jpg", "a blue sphere is in the scene", 1, "counting", 11, 1),
        (102, "tr2.jpg", "there are exactly two cubes", 2, "counting", 12, 1),
        (103, "tr0.jpg", "there is a blue cube", 1, "counting", 10, 0),
        (104, "tr1.jpg", "a red sphere is in the scene", 1, "counting", 11, 0),
        (105, "tr2.jpg", "there are exactly three cubes", 2, "counting", 12, 0),
    ]
    val_rows = [
        (200, "v0.jpg", "a flurble cube is in the scene", 1, "counting", 20, 1),
        (201, "v1.jpg", "there is a red cube", 1, "counting", 21, 1),
        (202, "v0.jpg", "a flurble sphere is in the scene", 1, "counting", 20, 0),
        (203, "v1.jpg", "there is a blue cube", 1, "counting", 21, 0),
    ]
    _write_csv(root / "train.csv", LABELLED_HEADER, train_rows)
    _write_csv(root / "val.csv", LABELLED_HEADER, val_rows)

    fixture = {"val_ids": np.array([200, 201, 202, 203])}
    if with_test:
        for i in range(2):
            _write_jpeg(root / "test" / f"t{i}.jpg", seed=20 + i)
        test_rows = [
            (7001, "t0.jpg", "there is a zorblex cube", 1, "counting"),
            (7002, "t1.jpg", "there is a red cube", 1, "counting"),
            (7003, "t0.jpg", "a blue sphere is in the scene", 1, "counting"),
            (7004, "t1.jpg", "there are exactly two cubes", 2, "counting"),
        ]
        _write_csv(root / "test.csv", TEST_HEADER, test_rows)
        fixture["sample_ids"] = [7003, 7001, 7004, 7002]  # deliberately shuffled
        _write_csv(root / "sample_submission.csv", ["id", "label"], [(i, 0.0) for i in fixture["sample_ids"]])
    return fixture


def build_pair_fixture(path):
    """4 pairs over 4 images; pair twins are non-adjacent rows (image order 0,1,2,3,0,1,2,3)."""
    rows = []
    for k in range(2):
        for i in range(4):
            rows.append((101 + k * 4 + i, f"im{i}.jpg", f"caption number {k * 4 + i}", 1, "counting", 30 + i, 1 - k))
    _write_csv(path, LABELLED_HEADER, rows)
    return rows


def run_train(tmp, name_exp, extra):
    """Real CLI run on the temporary fixture; everything (data, output, cache) stays in tmp."""
    overrides = [
        "experiment=lab_baseline",
        "model.image_encoder.weights=null",   # no weight download
        "train.cache_features=false",
        "train.amp=false",
        "train.epochs=1",
        "train.batch_size=2",
        "save_config=false",
        "device=cpu",
        f"name_exp={name_exp}",
        f"paths.data_dir={tmp / 'data'}",
        f"paths.output_dir={tmp / 'out'}",
        f"paths.features_dir={tmp / 'features'}",
        f"paths.submissions_dir={tmp / 'subs'}",
    ] + list(extra)
    env = {**os.environ, "HYDRA_FULL_ERROR": "1"}
    return subprocess.run(
        [sys.executable, "-m", "src.train", *overrides],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=900,
    )


def _bn_modules(module):
    return [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


# --------------------------------------------------------------------------- CLI tests

class TestCLIOrdinaryBranch(unittest.TestCase):
    """train.predict_test=false (the default): no test read at all, val predictions saved."""

    def test_no_test_read_and_val_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fixture = build_data_dir(tmp / "data", with_test=False)  # no test.csv, no test dir
            proc = run_train(tmp, "test_ordinary", [])
            self.assertEqual(proc.returncode, 0, proc.stdout[-4000:] + proc.stderr[-4000:])

            out = tmp / "out"
            results = json.loads((out / "results.json").read_text())
            for key in ("auc", "accuracy", "pair_accuracy"):
                self.assertTrue(np.isfinite(results["val"][key]), f"val.{key} not finite")
            self.assertIn("val_para", results)

            self.assertEqual(len(np.load(out / "val_logits.npy")), len(fixture["val_ids"]))
            np.testing.assert_array_equal(np.load(out / "val_ids.npy"), fixture["val_ids"])
            self.assertTrue((out / "model.pt").exists())

            vocab = json.loads((out / "vocab.json").read_text())
            self.assertIn("cube", vocab["itos"])
            self.assertNotIn("flurble", vocab["itos"])   # val-only word must not reach the vocab
            self.assertNotIn("zorblex", vocab["itos"])   # test fixture is absent anyway

            # test was never read and no test artefacts were produced
            self.assertFalse((out / "test_logits.npy").exists())
            self.assertFalse((out / "test_ids.npy").exists())
            self.assertEqual(list((tmp / "subs").glob("*.csv")), [])

            # the fixture's train labels really form valid pairs
            split = load_split("train", tmp / "data" / "train.csv")
            groups = pair_groups(split)
            self.assertEqual(groups.shape, (3, 2))
            for g in groups:
                self.assertEqual(set(split.labels[g]), {0, 1})
                self.assertEqual(len(set(split.img_idx[g])), 1)


class TestCLIFinalBranch(unittest.TestCase):
    """data.train_on_val=true + train.predict_test=true: vocabulary and sample alignment."""

    def test_vocabulary_and_sample_order(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fixture = build_data_dir(tmp / "data", with_test=True)
            proc = run_train(tmp, "test_final", [
                "data.train_on_val=true",
                "train.early_stopping=null",
                "train.predict_test=true",
            ])
            self.assertEqual(proc.returncode, 0, proc.stdout[-4000:] + proc.stderr[-4000:])

            out = tmp / "out"
            vocab = json.loads((out / "vocab.json").read_text())
            self.assertIn("flurble", vocab["itos"])      # val word joins the vocabulary on the final refit
            self.assertNotIn("zorblex", vocab["itos"])   # test-only word never fits the vocabulary

            sub_path = tmp / "subs" / "test_final.csv"   # paths.submission = <submissions_dir>/<name_exp>.csv
            self.assertTrue(sub_path.exists())
            with open(sub_path, newline="") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, ["id", "label"])
                rows = list(reader)
            self.assertEqual([int(r["id"]) for r in rows], fixture["sample_ids"])  # sample order preserved
            for r in rows:
                p = float(r["label"])
                self.assertTrue(np.isfinite(p) and 0.0 <= p <= 1.0, f"bad probability {r['label']}")
            self.assertTrue((out / "results.json").exists())
            self.assertTrue((out / "test_logits.npy").exists())


# --------------------------------------------------------------------------- pair tests

class TestPairs(unittest.TestCase):
    def test_invalid_pairs_rejected_before_training(self):
        cases = [
            ("single_row", [(1, "a.jpg", "there is a cube", 1, "counting", 5, 1)], "5"),
            ("two_positives", [(1, "a.jpg", "c one", 1, "counting", 6, 1), (2, "a.jpg", "c two", 1, "counting", 6, 1)], "6"),
            ("two_images", [(1, "a.jpg", "c one", 1, "counting", 7, 1), (2, "b.jpg", "c two", 1, "counting", 7, 0)], "7"),
        ]
        for case, rows, pair_id in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as td:
                csv_path = Path(td) / "train.csv"
                _write_csv(csv_path, LABELLED_HEADER, rows)
                with self.assertRaises(ValueError) as cm:
                    load_split("train", csv_path)
                msg = str(cm.exception)
                self.assertIn("train", msg)   # error names the split
                self.assertIn(pair_id, msg)   # ... and the problem pair

    def test_sampler_keeps_pairs_and_pair_loss_ranks(self):
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "train.csv"
            build_pair_fixture(csv_path)
            split = load_split("train", csv_path)

            groups = pair_groups(split)
            self.assertEqual(groups.shape, (4, 2))
            for g in groups:
                self.assertEqual(set(split.labels[g]), {0, 1})
                self.assertEqual(len(set(split.img_idx[g])), 1)

            rng = np.random.default_rng(0)
            aug = {"hflip_swap_lr": 0.0, "reorder_clauses_p": 0.0}
            seen = []
            for batch in train_batches(split, 4, rng, aug, pair_sampler=True):
                rows = batch["rows"]
                pids = split.pair_ids[rows]
                self.assertEqual(2 * len(set(pids)), len(rows))  # both twins of every pair in the batch
                labels = batch["labels"].numpy()
                pi = batch["pair_index"].numpy()
                self.assertTrue((labels[pi[:, 0]] == 1).all())
                self.assertTrue((labels[pi[:, 1]] == 0).all())
                seen.extend(int(r) for r in rows)
            self.assertEqual(sorted(seen), list(range(len(split))))

        # pair loss penalises swapped true/false scores more than a correct ranking
        loss_fn = TruthLoss(pair_weight=0.5)
        labels = torch.tensor([1, 0])
        pair_index = torch.tensor([[0, 1]])
        good = float(loss_fn(torch.tensor([2.0, -2.0]), labels, pair_index))
        swapped = float(loss_fn(torch.tensor([-2.0, 2.0]), labels, pair_index))
        self.assertGreater(swapped, good)
        # Isolate the pair penalty so BCE alone cannot satisfy this regression.
        plain = TruthLoss(pair_weight=0.0)
        good_bce = float(plain(torch.tensor([2.0, -2.0]), labels, pair_index))
        swapped_bce = float(plain(torch.tensor([-2.0, 2.0]), labels, pair_index))
        self.assertGreater(swapped - swapped_bce, good - good_bce)


# --------------------------------------------------------------------------- subset tests

class TestSubset(unittest.TestCase):
    def test_subset_compacts_images_before_source(self):
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "train.csv"
            build_pair_fixture(csv_path)  # rows carry images 0,1,2,3,0,1,2,3
            split = load_split("train", csv_path)
            mask = np.isin(split.img_idx, [0, 2])  # rows 0,2,4,6: non-adjacent source images

            sub = subset(split, mask)
            self.assertIsNone(sub.source)
            self.assertEqual(sub.image_names, ["im0.jpg", "im2.jpg"])  # compacted image list
            np.testing.assert_array_equal(sub.img_idx, [0, 1, 0, 1])   # compact renumbering
            self.assertLess(int(sub.img_idx.max()), len(sub.image_names))
            np.testing.assert_array_equal(sub.labels, split.labels[mask])
            np.testing.assert_array_equal(sub.ids, split.ids[mask])
            self.assertEqual(sub.captions, [split.captions[r] for r in np.nonzero(mask)[0]])

            # once a source is attached, indexing must stay untouched (source indexing is fixed)
            split.source = ArraySource(np.zeros((4, 1), np.float32))
            sub2 = subset(split, mask)
            self.assertEqual(sub2.image_names, split.image_names)
            np.testing.assert_array_equal(sub2.img_idx, split.img_idx[mask])


# --------------------------------------------------------------------------- cache tests

class TestFeatureCache(unittest.TestCase):
    def test_cache_key_distinguishes_weights(self):
        from omegaconf import OmegaConf

        def key(weights, size=(160, 240), normalize="imagenet"):
            enc_cfg = OmegaConf.create({"arch": "resnet50", "weights": weights, "out_layer": "layer3"})
            data_cfg = OmegaConf.create({"image_size": list(size), "normalize": normalize})
            return cache_key(enc_cfg, data_cfg)

        keys = [key(None), key("IMAGENET1K_V1"), key("IMAGENET1K_V2")]
        self.assertEqual(len(set(keys)), 3)  # None != V1 != V2
        for k in keys:
            self.assertTrue(k.startswith("eval-v2_"), k)  # new caches never mix with old ones
        self.assertNotEqual(key(None), key(None, size=(32, 48)))
        self.assertNotEqual(key(None), key(None, normalize="clip"))

    def test_frozen_extraction_keeps_bn_and_matches_eval_forward(self):
        from torchvision import models

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            names = [f"im{i}.jpg" for i in range(4)]
            for i, n in enumerate(names):
                _write_jpeg(tmp / n, seed=i)

            # construction must not move BatchNorm statistics (init: mean 0, var 1, count 0)
            enc = TorchvisionEncoder("resnet18", None, "layer3", spatial=True, freeze=True)
            self.assertTrue(_bn_modules(enc.body))
            for m in _bn_modules(enc.body):
                self.assertEqual(int(m.num_batches_tracked), 0)
                self.assertTrue(torch.equal(m.running_mean, torch.zeros_like(m.running_mean)))
                self.assertTrue(torch.equal(m.running_var, torch.ones_like(m.running_var)))

            # a frozen backbone stays in eval mode even while the head trains
            enc.train()
            self.assertFalse(enc.body.training)
            self.assertFalse(any(m.training for m in _bn_modules(enc.body)))

            enc.eval()
            before = [
                (m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
                for m in _bn_modules(enc.body)
            ]
            out_path = tmp / "features" / "train.npy"
            arr = cache_features(enc, tmp, names, out_path, (64, 96), "imagenet", torch.device("cpu"), batch_size=2)
            for (rm0, rv0, nb0), (rm1, rv1, nb1) in zip(
                before, [(m.running_mean, m.running_var, m.num_batches_tracked) for m in _bn_modules(enc.body)]
            ):
                self.assertTrue(torch.equal(rm0, rm1) and torch.equal(rv0, rv1) and torch.equal(nb0, nb1))

            # cached float16 features match a direct eval forward pass
            self.assertEqual(arr.dtype, np.float16)
            images = load_images(tmp, names)
            x = preprocess(torch.from_numpy(images), (64, 96), "imagenet")
            with torch.no_grad():
                ref = enc(x).numpy()
            self.assertEqual(arr.shape, ref.shape)
            np.testing.assert_allclose(np.asarray(arr, dtype=np.float32), ref, rtol=2e-3, atol=2e-3)
            del arr  # release the memory map before removing the file on Windows


class TestRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_partial_update_and_two_lr(self):
        from omegaconf import OmegaConf
        from src.models.fusion import FiLMHead
        from src.models.model import VisualReasoningModel
        from src.models.text_encoders import GRUEncoder, Vocab
        from src.training import Trainer

        torch.manual_seed(7)
        encoder = TorchvisionEncoder("resnet18", None, "layer3", freeze=True, unfreeze_last_n_blocks=1)
        captions = ["red cube", "blue cube", "two spheres", "three spheres"]
        vocab = Vocab(captions)
        text = GRUEncoder(len(vocab), embed_dim=8, hidden_dim=8, dropout=0)
        fusion = FiLMHead(encoder.out_channels, text.out_dim, num_blocks=1, hidden_channels=8,
                          classifier_channels=16, classifier_hidden=16, readout="mlp", dropout=0)
        model = VisualReasoningModel(text, fusion, encoder)
        cfg = OmegaConf.create({"amp": False, "optimizer": {
            "lr": 3e-4, "image_encoder_lr": 3e-5, "weight_decay": 1e-4}})
        trainer = Trainer(model, TruthLoss(), vocab, cfg, torch.device("cpu"))
        opt = trainer._make_optimizer()
        ids = [id(p) for group in opt.param_groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})
        self.assertEqual([g["name"] for g in opt.param_groups], ["image_encoder", "task"])
        self.assertAlmostEqual(opt.param_groups[0]["lr"] / opt.param_groups[1]["lr"], .1)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: .5 ** step)
        parts = {"CNN": [p for p in encoder.parameters() if p.requires_grad],
                 "GRU": list(text.parameters()), "FiLM": list(fusion.stack.parameters()),
                 "readout": list(fusion.readout.parameters())}
        before = {name: [p.detach().clone() for p in ps] for name, ps in parts.items()}
        frozen = {n: p.detach().clone() for n, p in encoder.named_parameters() if not p.requires_grad}
        buffers = {n: b.clone() for n, b in encoder.named_buffers()}
        model.train()
        self.assertFalse(any(m.training for m in _bn_modules(encoder)))
        self.assertTrue(all(m.training for m in _bn_modules(fusion)))
        tokens, lengths = vocab.encode_batch(captions)
        loss = TruthLoss()(model(torch.randn(4, 3, 64, 96), tokens, lengths),
                           torch.tensor([1, 0, 1, 0]), torch.tensor([[0, 1], [2, 3]]))
        loss.backward()
        for name, ps in parts.items():
            norm = sum(float(p.grad.square().sum()) for p in ps if p.grad is not None) ** .5
            self.assertTrue(np.isfinite(norm) and norm > 0, name)
        opt.step()
        sched.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"] / opt.param_groups[1]["lr"], .1)
        for name, ps in parts.items():
            delta = sum(float((p.detach() - b).square().sum()) for p, b in zip(ps, before[name])) ** .5
            self.assertTrue(np.isfinite(delta) and delta > 0, name)
        for n, p in encoder.named_parameters():
            if n in frozen:
                self.assertTrue(torch.equal(p, frozen[n]), n)
        for n, b in encoder.named_buffers():
            self.assertTrue(torch.equal(b, buffers[n]), n)
        for p in encoder.parameters():
            p.requires_grad_(False)
        self.assertEqual([g["name"] for g in trainer._make_optimizer().param_groups], ["task"])
        model.image_encoder = None
        self.assertEqual([g["name"] for g in trainer._make_optimizer().param_groups], ["task"])

    def test_partial_encoder_disables_cache(self):
        from hydra import compose, initialize_config_dir
        from src.train import _attach_sources

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            build_data_dir(tmp / "data", with_test=False)
            with initialize_config_dir(config_dir=str(REPO / "configs")):
                cfg = compose(config_name="config", overrides=[
                    "model/image_encoder=resnet18", "model.image_encoder.weights=null",
                    "model.image_encoder.unfreeze_last_n_blocks=1", "train.cache_features=true",
                    f"paths.data_dir={tmp / 'data'}", f"paths.features_dir={tmp / 'features'}"])
            split = load_split("train", tmp / "data/train.csv")
            encoder, _, mode = _attach_sources(cfg, {"train": split}, torch.device("cpu"), False)
            self.assertEqual(mode, "images")
            self.assertIsNotNone(encoder)
            self.assertTrue(any(p.requires_grad for p in encoder.parameters()))
            self.assertEqual(split.source.array.dtype, np.uint8)
            self.assertEqual(list((tmp / "features").rglob("*")), [])

    def test_balanced_pairs(self):
        from dataclasses import replace

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "train.csv"
            rows = [(100 + k * 6 + i, f"im{i}.jpg", f"caption {i} label {k}", 1,
                     "counting" if i < 3 else "binding", 10 + i, 1 - k)
                    for k in range(2) for i in range(6)]
            _write_csv(path, LABELLED_HEADER, rows)
            split = load_split("train", path)
            selected = sample_pairs_by_category(split, 2, 42)
            np.testing.assert_array_equal(selected.ids, sample_pairs_by_category(split, 2, 42).ids)
            self.assertTrue(np.all(np.diff(selected.ids) > 0))
            self.assertEqual(set(selected.img_idx), set(range(len(selected.image_names))))
            for category in ("binding", "counting"):
                self.assertEqual(int(np.sum(selected.categories == category)), 4)
            for g in pair_groups(selected):
                self.assertEqual(set(selected.labels[g]), {0, 1})
                self.assertEqual(len(set(selected.pair_ids[g])), 1)
                self.assertEqual(len(set(selected.img_idx[g])), 1)
                self.assertEqual(len(set(selected.categories[g])), 1)
            for n in (0, -1, 1.5, True):
                with self.subTest(n=n), self.assertRaisesRegex(ValueError, "train"):
                    sample_pairs_by_category(split, n, 42)
            with self.assertRaisesRegex(ValueError, "binding.*requested 4.*available 3"):
                sample_pairs_by_category(split, 4, 42)
            categories = split.categories.copy()
            categories[6] = "binding"
            with self.assertRaisesRegex(ValueError, "pair_id 10"):
                sample_pairs_by_category(replace(split, categories=categories), 2, 42)
            for field in ("labels", "pair_ids"):
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "train"):
                    sample_pairs_by_category(replace(split, **{field: None}), 2, 42)

    def _small_cli_overrides(self):
        return ["experiment=film_memorize", "model.image_encoder.arch=resnet18",
                "model.image_encoder.unfreeze_last_n_blocks=1", "data.image_size=[64,96]",
                "data.train_pairs_per_category=1", "model.text_encoder.embed_dim=8",
                "model.text_encoder.hidden_dim=8", "model.fusion.num_blocks=1",
                "model.fusion.hidden_channels=8", "model.fusion.classifier_channels=16",
                "model.fusion.classifier_hidden=16"]

    def test_train_only_cli(self):
        import shutil
        from src.models.text_encoders import Vocab

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            build_data_dir(tmp / "data", with_test=False)
            (tmp / "data/val.csv").unlink()
            shutil.rmtree(tmp / "data/val")
            proc = run_train(tmp, "test_train_only", self._small_cli_overrides())
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            out = tmp / "out"
            result = json.loads((out / "results.json").read_text())
            fitted = sample_pairs_by_category(load_split("train", tmp / "data/train.csv"), 1, 42)
            np.testing.assert_array_equal(np.load(out / "train_fit_ids.npy"), fitted.ids)
            self.assertEqual(np.load(out / "train_fit_logits.npy").shape, (len(fitted),))
            self.assertTrue(np.isfinite(np.load(out / "train_fit_logits.npy")).all())
            for metric in ("auc", "accuracy", "pair_accuracy", "auc/counting"):
                self.assertTrue(np.isfinite(result["train_fit"][metric]))
                self.assertIn("train_fit_" + metric, result["history"][0])
            self.assertEqual(json.loads((out / "vocab.json").read_text())["itos"], Vocab(fitted.captions).itos)
            self.assertFalse(any(k in result for k in ("val", "val_para", "submission")))
            self.assertEqual(list(out.glob("val*")), [])
            self.assertEqual(list(out.glob("test*")), [])
            self.assertEqual(list((tmp / "subs").glob("*.csv")), [])

    def test_holdout_cli_and_missing_monitor(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fixture = build_data_dir(tmp / "data", with_test=False)
            overrides = self._small_cli_overrides() + [
                "data.train_pairs_per_category=null", "train.eval.holdout=true",
                "train.early_stopping.monitor=val_auc"]
            proc = run_train(tmp, "test_holdout", overrides)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            out = tmp / "out"
            result = json.loads((out / "results.json").read_text())
            for name in ("train_fit", "val", "val_para"):
                for metric in ("auc", "accuracy", "pair_accuracy", "auc/counting"):
                    self.assertTrue(np.isfinite(result[name][metric]))
                    self.assertIn(name + "_" + metric, result["history"][0])
            np.testing.assert_array_equal(np.load(out / "val_ids.npy"), fixture["val_ids"])
            train = load_split("train", tmp / "data/train.csv")
            np.testing.assert_array_equal(np.load(out / "train_fit_ids.npy"), train.ids)
            self.assertEqual(result["stop_reason"], "epochs")
            self.assertNotIn("flurble", json.loads((out / "vocab.json").read_text())["itos"])
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            build_data_dir(tmp / "data", with_test=False)
            proc = run_train(tmp, "test_missing_monitor", self._small_cli_overrides() + [
                "train.early_stopping.monitor=val_auc"])
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("early stopping monitor 'val_auc' unavailable", proc.stdout + proc.stderr)
            self.assertFalse((tmp / "out/model.pt").exists())
            self.assertFalse((tmp / "out/results.json").exists())


if __name__ == "__main__":
    unittest.main()
