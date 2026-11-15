"""Slot autoencoder, caption programs, exact count semantics and the frozen-base head."""

import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data import ArraySource, Split
from src.losses import TruthLoss
from src.models.model import VisualReasoningModel
from src.models.slot_program import (
    AND, EXACT, EXIST, FEWER, MORE, NONE, SAME, PairSlotCountChoiceHead, SlotProgramExecutor,
    encode_program_inputs, log1mexp, parse_program, poisson_binomial,
)
from src.models.slots import SlotAutoencoder, load_slot_autoencoder
from src.models.text_encoders import GRUEncoder, Vocab
from src.slots_pretrain import save_checkpoint
from src.training import Trainer
from omegaconf import OmegaConf

SLOT_CONFIG = dict(grid=(3, 5), num_slots=4, slot_dim=8, input_dim=8, decoder_hidden=16,
                   decoder_layers=2)


class TestSlotAutoencoder(unittest.TestCase):
    def test_attention_partitions_tokens_and_reconstruction_trains(self):
        torch.manual_seed(0)
        model = SlotAutoencoder(6, target_dim=4, **SLOT_CONFIG)
        x = torch.randn(5, 6, 3, 5)
        model.fit_normalization(x)
        self.assertTrue(torch.allclose(model.target_proj.T @ model.target_proj, torch.eye(4), atol=1e-4))
        out = model(x)
        self.assertEqual(out["slots"].shape, (5, 4, 8))
        self.assertEqual(out["reconstruction"].shape, (5, 15, 4))
        self.assertTrue(torch.allclose(out["attn"].sum(1), torch.ones(5, 15), atol=1e-4))
        self.assertTrue(torch.allclose(out["alpha"].sum(1), torch.ones(5, 15), atol=1e-5))
        self.assertTrue(((out["position"] >= -1) & (out["position"] <= 1)).all())
        out["loss"].backward()
        self.assertIsNotNone(model.slot_attention.queries.grad)  # straight-through init
        opt = torch.optim.Adam(model.parameters(), 1e-2)
        first = None
        for _ in range(30):
            loss = model(x)["loss"]
            first = first if first is not None else loss.item()
            opt.zero_grad(), loss.backward(), opt.step()
        self.assertLess(loss.item(), first)

    def test_encoding_is_deterministic_and_checkpoint_round_trips(self):
        model = SlotAutoencoder(6, **SLOT_CONFIG).eval()
        x = torch.randn(2, 6, 3, 5)
        with torch.no_grad():
            a, b = model.encode(x), model.encode(x)
        self.assertTrue(torch.equal(a["slots"], b["slots"]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slots.pt"
            save_checkpoint(model, path)
            loaded = load_slot_autoencoder(path).eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(loaded.encode(x)["slots"], a["slots"]))
        with self.assertRaisesRegex(ValueError, "grid"):
            model.encode(torch.randn(1, 6, 4, 5))


class TestPrograms(unittest.TestCase):
    def test_parser_covers_every_template_and_hides_numbers_from_filters(self):
        cases = {
            "exactly two small red spheres": (EXACT, ("small", "red", "sphere"), (), (), 0, 2),
            "exactly one object": (EXACT, (), (), (), 0, 1),
            "there are exactly three cubes to the left of the red sphere": (EXACT, ("cube",), (), ("red", "sphere"), 1, 3),
            "there is exactly one sphere behind the blue cube": (EXACT, ("sphere",), (), ("blue", "cube"), 4, 1),
            "there are more cubes than spheres": (MORE, ("cube",), ("sphere",), (), 0, 0),
            "there are fewer red objects than yellow objects": (FEWER, ("red",), ("yellow",), (), 0, 0),
            "there are as many red objects as cubes": (SAME, ("red",), ("cube",), (), 0, 0),
            "a purple sphere and a yellow cube": (AND, ("purple", "sphere"), ("yellow", "cube"), (), 0, 0),
            "there is no brown cube": (NONE, ("brown", "cube"), (), (), 0, 0),
            "there is a brown cube": (EXIST, ("brown", "cube"), (), (), 0, 0),
            "a red rubber cube": (EXIST, ("red", "rubber", "cube"), (), (), 0, 0),
            "the gray cylinder is on the left side of the image": (EXIST, ("gray", "cylinder"), (), (), 5, 0),
            "there is a gray sphere on the right side of the image": (EXIST, ("gray", "sphere"), (), (), 6, 0),
            "a purple cube behind the yellow cylinder": (EXIST, ("purple", "cube"), (), ("yellow", "cylinder"), 4, 0),
            "the green cube is in front of the yellow cube": (EXIST, ("green", "cube"), (), ("yellow", "cube"), 3, 0),
            "there is no sphere to the right of the yellow cylinder": (NONE, ("sphere",), (), ("yellow", "cylinder"), 2, 0),
        }
        for caption, expected in cases.items():
            p = parse_program(caption)
            self.assertEqual((p.op, p.a, p.b, p.ref, p.relation, p.number), expected, caption)
        for caption in ("the scene is empty", "exactly many cubes", "a red dragon"):
            self.assertIsNone(parse_program(caption))
        inputs = encode_program_inputs(["exactly two red cubes", "nonsense", "a cube"], "cpu")
        self.assertEqual(inputs["rows"].tolist(), [0, 2])
        self.assertEqual(inputs["a"].shape, (2, 4))

    def test_poisson_binomial_matches_enumeration(self):
        p = torch.tensor([[.9, .2, .5, .7], [1e-6, .999, .3, .01]], dtype=torch.float64)
        dist = poisson_binomial(p.log()).exp()
        self.assertTrue(torch.allclose(dist.sum(1), torch.ones(2, dtype=torch.float64)))
        for r in range(2):
            brute = torch.zeros(5, dtype=torch.float64)
            for bits in itertools.product([0, 1], repeat=4):
                bits_t = torch.tensor(bits, dtype=torch.float64)
                brute[sum(bits)] += torch.prod(p[r] * bits_t + (1 - p[r]) * (1 - bits_t))
            self.assertTrue(torch.allclose(dist[r], brute, atol=1e-9))
        x = torch.tensor([-1e-5, -.1, -5., -40.], dtype=torch.float64)
        self.assertTrue(torch.allclose(log1mexp(x), torch.log(1 - torch.exp(x)), rtol=1e-4))
        # Certain slots (p = 1 or 0) must not produce NaN gradients.
        logits = torch.tensor([[40., -40., 0., 40.]], requires_grad=True)
        poisson_binomial(torch.nn.functional.logsigmoid(logits)).logsumexp(-1).sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_executor_semantics_on_known_slots(self):
        """Slots: red cube (left), red cube (right), blue sphere (centre), background."""
        ex = SlotProgramExecutor(slot_dim=4, feature_dim=4, dim=4, relation_sharpness=50.)
        big = 30.
        objectness = torch.tensor([[big, big, big, -big]])
        names = ["red", "cube", "blue", "sphere"]
        table = {"red": [1, 1, 0, 0], "cube": [1, 1, 0, 0], "blue": [0, 0, 1, 0], "sphere": [0, 0, 1, 0]}
        from src.models.slot_program import WORD_INDEX
        words = torch.full((1, 4, len(WORD_INDEX) + 1), -big)
        for name in names:
            words[0, :, WORD_INDEX[name]] = torch.tensor(table[name], dtype=torch.float) * 2 * big - big
        ex.slot_logits = lambda enc: (objectness, words)
        enc = {"position": torch.tensor([[[-.8, 0.], [.8, 0.], [.3, 0.], [0., 0.]]])}

        def truth(caption):
            inputs = encode_program_inputs([caption], "cpu")
            log_true, _, _ = ex(enc, torch.zeros(1, dtype=torch.long), inputs)
            return log_true.exp().item()

        for caption, expected in (("exactly two red cubes", 1), ("exactly one red cube", 0),
                                  ("exactly three objects", 1), ("there is no blue sphere", 0),
                                  ("a blue sphere", 1), ("there are more cubes than spheres", 1),
                                  ("there are fewer cubes than spheres", 0),
                                  ("there are as many cubes as spheres", 0),
                                  ("a red cube and a blue sphere", 1),
                                  ("there is exactly one cube to the left of the blue sphere", 1),
                                  ("there are exactly two cubes to the left of the blue sphere", 0),
                                  ("there is no cube to the right of the blue sphere", 0),
                                  ("the blue sphere is on the left side of the image", 0)):
            self.assertAlmostEqual(truth(caption), expected, places=3, msg=caption)


def fixture(route_ops=("exact", "more", "fewer", "same")):
    captions = ["exactly one red cube", "exactly two red cubes",
                "there are more cubes than spheres", "there are fewer cubes than spheres",
                "a red cube", "a blue cube",
                "there is exactly one sphere behind the blue cube", "there are exactly two spheres behind the blue cube"]
    torch.manual_seed(3)
    vocab = Vocab(captions)
    text = GRUEncoder(len(vocab), embed_dim=8, hidden_dim=8, dropout=.0)
    head = PairSlotCountChoiceHead(6, text.out_dim, slot_config=SLOT_CONFIG, executor_dim=8,
                                   executor_hidden=16, route_ops=route_ops, dim=8, queries=2, hidden=16,
                                   count_dim=8, count_selectors=2, count_hidden=16)
    split = Split("train", np.arange(8), [f"{i}.jpg" for i in range(4)], np.repeat(np.arange(4), 2),
                  captions, np.repeat(["counting", "comparison", "attribute", "spatial_counting"], 2),
                  np.tile([1, 0], 4), np.repeat(np.arange(4), 2),
                  ArraySource(np.random.default_rng(5).normal(size=(4, 6, 3, 5)).astype(np.float32)))
    cfg = OmegaConf.create({"amp": False, "batch_size": 8, "pair_sampler": True,
                            "optimizer": {"lr": .01, "weight_decay": .0}, "epochs": 1,
                            "scheduler": {"name": "cosine", "warmup_epochs": 0}, "early_stopping": None,
                            "grad_clip": 1., "time_budget_min": 0})
    model = VisualReasoningModel(text, head)
    return split, vocab, model, Trainer(model, TruthLoss(.5), vocab, cfg, torch.device("cpu"))


class TestSlotHead(unittest.TestCase):
    def test_only_executor_trains_and_base_is_frozen(self):
        split, _, model, trainer = fixture()
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(all(n.startswith("fusion.slot_executor.") for n in trainable))
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        trainer.fit(split, {}, {"hflip_swap_lr": 0, "reorder_clauses_p": 0})
        for n, p in model.named_parameters():
            if n in frozen:
                self.assertTrue(torch.equal(p, frozen[n]), n)
        model.train()
        self.assertFalse(model.fusion.slot_encoder.training)
        self.assertFalse(model.fusion.cell.training)
        self.assertTrue(model.fusion.slot_executor.training)

    def test_routing_is_antisymmetric_and_respects_route_ops(self):
        split, _, model, trainer = fixture()
        logits = trainer.predict(split)
        self.assertTrue(np.allclose(logits[0::2], -logits[1::2], atol=1e-5))
        _, _, base_model, base_trainer = fixture(route_ops=())
        base_model.load_state_dict(model.state_dict())
        base_logits = base_trainer.predict(split)
        # Attribute pair (rows 4, 5) is never routed; routed pairs change.
        self.assertTrue(np.allclose(logits[4:6], base_logits[4:6], atol=1e-6))
        self.assertFalse(np.allclose(logits[:4], base_logits[:4]))
        reversed_split = Split("train", split.ids[::-1].copy(), split.image_names, split.img_idx[::-1].copy(),
                               split.captions[::-1], split.categories[::-1], split.labels[::-1].copy(),
                               split.pair_ids[::-1].copy(), split.source)
        self.assertTrue(np.allclose(trainer.predict(reversed_split)[::-1], logits, atol=1e-5))

    def test_unknown_route_op_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "route_ops"):
            fixture(route_ops=("counting",))


if __name__ == "__main__":
    unittest.main()


class TestCountSupervision(unittest.TestCase):
    def test_count_targets_use_true_captions_only(self):
        from src.slots_pretrain import count_targets
        captions = ["exactly four objects", "exactly five objects",          # img 0: exact 4
                    "exactly three cubes", "exactly two cubes",              # img 1: >= 3
                    "there are exactly two spheres behind the red cube", "there are exactly three spheres behind the red cube",
                    "a red cube and a blue sphere", "a blue cube and a red sphere",
                    "a red cube", "a blue cube"]                           # img 4: nothing
        split = Split("train", np.arange(10), [f"{i}.jpg" for i in range(5)], np.repeat(np.arange(5), 2),
                      captions, np.repeat(["counting"] * 3 + ["binding", "attribute"], 2),
                      np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0]), np.repeat(np.arange(5), 2))
        targets = count_targets(split)
        self.assertEqual(targets["0.jpg"], (4., 4.))
        self.assertTrue(np.isnan(targets["1.jpg"][0]) and targets["1.jpg"][1] == 3.)
        self.assertEqual(targets["2.jpg"][1], 4.)  # three spheres + their reference
        self.assertEqual(targets["3.jpg"][1], 2.)
        self.assertNotIn("4.jpg", targets)

    def test_object_count_ignores_background_and_empty_slots(self):
        model = SlotAutoencoder(6, background_slots=(0,), count_sharpness=50., **SLOT_CONFIG)
        attn = torch.full((1, 4, 15), .02)
        attn[0, 0, :8] = .97   # background owns tokens: never counted
        attn[0, 1, 8:10] = .97  # object
        attn[0, 2, 10] = .97    # object
        self.assertAlmostEqual(model.object_count(attn).item(), 2., places=3)
        with self.assertRaisesRegex(ValueError, "background"):
            SlotAutoencoder(6, background_slots=(9,), **SLOT_CONFIG)

    def test_expanding_slots_keeps_learned_queries(self):
        from src.models.slots import expand_slot_state
        small = SlotAutoencoder(6, **SLOT_CONFIG)
        state = expand_slot_state(small.state_dict(), 6, torch.Generator().manual_seed(0))
        big = SlotAutoencoder(6, **{**SLOT_CONFIG, "num_slots": 6})
        big.load_state_dict(state)
        self.assertTrue(torch.equal(big.slot_attention.queries[:4], small.slot_attention.queries))
        self.assertEqual(big.encode(torch.randn(2, 6, 3, 5))["slots"].shape, (2, 6, 8))
        with self.assertRaises(ValueError):
            expand_slot_state(big.state_dict(), 4)

    def test_sampler_puts_labelled_images_first_in_original_order(self):
        from src.slots_pretrain import Sampler
        a = np.arange(10, dtype=np.float32).reshape(10, 1, 1, 1)
        b = 100 + np.arange(5, dtype=np.float32).reshape(5, 1, 1, 1)
        labelled = (np.array([11, 3]), np.array([[np.nan, 2.], [5., 5.]], dtype=np.float32))
        sampler = Sampler({"a": a, "b": b}, 6, 0, labelled, count_batch=2)
        for _ in range(5):
            x, targets = sampler.next()
            self.assertEqual(x.shape[0], 6)
            first = x[:2].flatten().tolist()
            expected = {101.: 2., 3.: 5.}
            self.assertEqual(sorted(first), [3., 101.])
            for value, target in zip(first, targets[:, 1].tolist()):
                self.assertEqual(expected[value], target)
