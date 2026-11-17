"""Focused regression tests for optional slot-program inference semantics.

Does not require dataset artifacts; all object annotations here are synthetic.
"""
import unittest

import torch

from src.models.slot_program import SlotProgramExecutor, WORD_INDEX, encode_program_inputs
from src.slots_pretrain import open_caches


def executor_with_scene(object_words, xy, *, reference_mode="barycenter", require_reference_exists=False,
                        exclusive_attributes=False):
    """Inject known object labels for testing the program, independently of the encoder."""
    k = len(object_words)
    ex = SlotProgramExecutor(4, 4, dim=4, dropout=0., relation_sharpness=80.,
                             reference_mode=reference_mode,
                             require_reference_exists=require_reference_exists,
                             exclusive_attributes=exclusive_attributes)
    obj = torch.full((1, k), 25.)
    words = torch.full((1, k, len(WORD_INDEX) + 1), -25.)
    for i, names in enumerate(object_words):
        for w in names:
            words[0, i, WORD_INDEX[w]] = 25.
    ex.slot_logits = lambda enc: (obj, words)
    enc = {"position": torch.tensor([xy], dtype=torch.float32)}
    return ex, enc


def truth(ex, enc, caption):
    inp = encode_program_inputs([caption], "cpu")
    return ex(enc, torch.tensor([0]), inp)[0].exp().item()


class TestSemanticSlotModes(unittest.TestCase):
    def test_pretraining_rejects_test_before_opening_caches(self):
        with self.assertRaisesRegex(ValueError, "test is inference only"):
            open_caches(None, ("train", "test"))

    def test_marginalization_does_not_invent_an_average_reference(self):
        # Two red cubes in the middle, two blue spheres at opposite ends.
        # Relative to EACH existing sphere, the count of cubes to its left is 0 or 2.
        # Neither reference has exactly one cube to its left. However, the mean
        # reference position is x=0, and ONE cube lies to its left.
        objects = [("red", "cube"), ("red", "cube"),
                   ("blue", "sphere"), ("blue", "sphere")]
        xy = [[-.55, 0.], [.55, 0.], [-.9, 0.], [.9, 0.]]
        caption = "there is exactly one red cube to the left of the blue sphere"
        old, enc = executor_with_scene(objects, xy)
        new, _ = executor_with_scene(objects, xy, reference_mode="marginal")
        self.assertGreater(truth(old, enc, caption), .95)
        self.assertLess(truth(new, enc, caption), .05)

    def test_missing_reference_makes_spatial_statement_false(self):
        objects = [("red", "cube"), ("green", "sphere")]
        xy = [[-.7, 0.], [.7, 0.]]
        ex, enc = executor_with_scene(objects, xy, reference_mode="marginal",
                                      require_reference_exists=True)
        self.assertLess(truth(ex, enc, "a red cube to the left of the blue sphere"), 1e-3)
        self.assertLess(truth(ex, enc, "there is no cube to the right of the blue sphere"), 1e-3)
        self.assertGreater(truth(ex, enc, "a red cube"), .99)

    def test_attribute_groups_exclude_incompatible_labels(self):
        ex = SlotProgramExecutor(4, 4, dim=4, exclusive_attributes=True)
        objectness = torch.full((1, 1), 25.)
        word_logits = torch.zeros((1, 1, len(WORD_INDEX) + 1))
        word_logits[0, 0, WORD_INDEX["red"]] = 9
        word_logits[0, 0, WORD_INDEX["blue"]] = 7
        def p(word):
            idx = torch.tensor([[WORD_INDEX[word], 0, 0, 0]])
            return ex.membership(objectness, word_logits, idx, exclusive=True).exp().item()
        self.assertAlmostEqual(sum(p(c) for c in ("gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow")), 1, places=4)
        self.assertGreater(p("red"), p("blue"))

    def test_coupled_comparison_respects_shared_slot_counts(self):
        # Comparing any filtered set with itself is ALWAYS equality, including
        # completely uncertain slot matches. Independent two-count sampling is wrong.
        ex = SlotProgramExecutor(4, 4, dim=4, comparison_mode="coupled")
        k = 4
        objectness = torch.zeros((1, k), requires_grad=True)
        logits = torch.zeros((1, k, len(WORD_INDEX) + 1), requires_grad=True)
        ex.slot_logits = lambda enc: (objectness, logits)
        enc = {"position": torch.randn(1, k, 2)}
        self.assertGreater(truth(ex, enc, "there are as many cubes as cubes"), .999)
        self.assertLess(truth(ex, enc, "there are more cubes than cubes"), 1e-4)
        self.assertLess(truth(ex, enc, "there are fewer cubes than cubes"), 1e-4)

    def test_coupled_comparison_normalizes_and_backpropagates(self):
        torch.manual_seed(4)
        ex = SlotProgramExecutor(4, 4, dim=4, comparison_mode="coupled", exclusive_attributes=True)
        obj = torch.randn(1, 4, requires_grad=True)
        words = torch.randn(1, 4, len(WORD_INDEX) + 1, requires_grad=True)
        ex.slot_logits = lambda enc: (obj, words)
        enc = {"position": torch.randn(1, 4, 2)}
        inp = encode_program_inputs(["there are more red objects than blue objects",
                                     "there are fewer cubes than spheres",
                                     "there are as many red objects as cubes"], "cpu")
        logits = ex(enc, torch.zeros(3, dtype=torch.long), inp)[0]
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        self.assertTrue(torch.isfinite(obj.grad).all())
        self.assertTrue(torch.isfinite(words.grad).all())

    def test_marginal_backward_is_finite(self):
        torch.manual_seed(3)
        ex = SlotProgramExecutor(4, 4, dim=4, reference_mode="marginal",
                                  exclusive_attributes=True, require_reference_exists=True)
        obj = torch.randn(1, 4, requires_grad=True)
        words = torch.randn(1, 4, len(WORD_INDEX) + 1, requires_grad=True)
        positions = torch.randn(1, 4, 2, requires_grad=True)
        ex.slot_logits = lambda enc: (obj, words)
        captions = ["exactly two cubes to the left of the red sphere",
                    "a sphere to the right of the blue cube",
                    "there is no cylinder behind the red cube"]
        inputs = encode_program_inputs(captions, "cpu")
        assert inputs is not None
        result = ex({"position": positions}, torch.zeros(len(captions), dtype=torch.long), inputs)[0]
        result.sum().backward()
        for val in (obj, words, positions):
            self.assertIsNotNone(val.grad)
            self.assertTrue(torch.isfinite(val.grad).all())


if __name__ == "__main__":
    unittest.main()
