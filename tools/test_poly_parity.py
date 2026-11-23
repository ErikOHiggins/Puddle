"""Tests for tools/poly_parity.py's pieces that need no engine: strata-poly's --top-probs lines and the comparison.

    python -m unittest tools.test_poly_parity
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poly_parity as P  # noqa: E402


class Tops(unittest.TestCase):
    def test_parse(self):
        err = "strata-poly: loaded\ntop 12:0.9 7:0.05\nprompt 3 tokens\ntop 5:0.6 12:0.4\n"
        self.assertEqual(P.parse_tops(err), [[(12, 0.9), (7, 0.05)], [(5, 0.6), (12, 0.4)]])


class Compare(unittest.TestCase):
    def test_all_agree(self):
        tops = [[(1, 0.9), (3, 0.1)], [(2, 0.8), (4, 0.2)]]
        c = P.compare([1, 2], tops, tops)
        self.assertEqual((c["agree"], c["first"], c["ties"]), (2, None, 0))
        self.assertAlmostEqual(c["kl_mean"], 0.0, places=6)

    def test_picture_marks(self):
        """A picture's tokens by which the model has as single tokens: Qwen-VL's (Occamy), Nemotron 3.5's, or none."""
        qwen = {"<|vision_start|>": [11], "<|vision_end|>": [12], "<|image_pad|>": [13]}
        nemotron = {"<img>": [21], "</img>": [22], "<image>": [23]}
        split = lambda t: list(range(len(t)))                       # noqa: E731  every character a token
        self.assertEqual(P.picture_marks(lambda t: qwen.get(t) or split(t)), [[11], [12], [13]])
        self.assertEqual(P.picture_marks(lambda t: nemotron.get(t) or split(t)), [[21], [22], [23]])
        self.assertIsNone(P.picture_marks(split))
        both = dict(qwen, **nemotron)                                # Nemotron's first: its prompt as before
        self.assertEqual(P.picture_marks(lambda t: both.get(t) or split(t)), [[21], [22], [23]])

    def test_a_near_tie_and_a_real_difference(self):
        ref, tops = [1, 2], [[(1, 0.52), (9, 0.48)], [(2, 0.9), (4, 0.1)]]
        c = P.compare(ref, tops, [[(9, 0.51), (1, 0.49)], [(4, 0.7), (2, 0.3)]])
        self.assertEqual((c["agree"], c["first"], c["ties"]), (0, 0, 1))   # step 1 is not a near tie
        self.assertTrue(c["at_first"]["near_tie"])


if __name__ == "__main__":
    unittest.main()
