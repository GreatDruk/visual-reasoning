import unittest

from src.text_aug import canonical_form, novel_form, paraphrases
from src.twins import find_twins

# (caption, category, twin) triples written in train forms.
CASES = [
    ("there is a red cube", "attribute", "there is a blue cube"),
    ("there is a red cube and a blue sphere", "binding", "there is a red cube and a green sphere"),
    ("there is no red cube", "negation", "there is a red cube"),
    ("there are exactly three cubes", "counting", "there are exactly four cubes"),
    ("there are more cubes than spheres", "comparison", "there are fewer cubes than spheres"),
    ("there is a red cube on the left side of the image", "position", "there is a red cube on the right side of the image"),
    ("there is a red cube to the left of the blue sphere", "spatial", "there is a red cube to the right of the blue sphere"),
    ("there are exactly two cubes behind the red sphere", "spatial_counting", "there are exactly three cubes behind the red sphere"),
    ("there is no red cube in front of the blue sphere", "spatial_negation", "there is a red cube in front of the blue sphere"),
]


class CanonicalFormTest(unittest.TestCase):
    def test_every_paraphrase_maps_to_one_form(self):
        for caption, category, _ in CASES:
            canon = canonical_form(caption, category)
            for form in paraphrases(caption, category):
                self.assertEqual(canonical_form(form, category), canon, form)

    def test_novel_form_is_undone(self):
        for caption, category, _ in CASES:
            self.assertEqual(canonical_form(novel_form(caption, category), category), canonical_form(caption, category))

    def test_twins_stay_distinct(self):
        for caption, category, twin in CASES:
            self.assertNotEqual(canonical_form(caption, category), canonical_form(twin, category))


class FindTwinsTest(unittest.TestCase):
    def test_group_of_two_is_a_pair_regardless_of_distance(self):
        captions = ["a b c d e f", "u v w x y z", "there is a red cube", "there is a blue cube"]
        pairs = find_twins(["1.jpg", "1.jpg", "1.jpg", "1.jpg"], captions, ["counting", "counting", "attribute", "attribute"])
        self.assertEqual(sorted(map(tuple, pairs.tolist())), [(0, 1), (2, 3)])


if __name__ == "__main__":
    unittest.main()
