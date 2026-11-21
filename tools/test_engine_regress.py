"""Tests for tools/engine_regress.py's pieces that need no engine: the run's arguments, the ids it reads back and
where two runs part.

    python -m unittest tools.test_engine_regress
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_regress as R  # noqa: E402


class Args(unittest.TestCase):
    def test_profile_left_out_and_the_run_fixed(self):
        args = ["--model", "m.gguf", "--max-context", "4096", "--expert-profile", "p.txt", "--kv", "q8_0",
                "--expert-swaps", "64", "--serve"]
        got = R.regress_args(args, [1, 2, 3], 16)
        self.assertNotIn("--expert-profile", got)
        self.assertNotIn("p.txt", got)
        self.assertNotIn("--serve", got)
        self.assertEqual(got[:6], ["--model", "m.gguf", "--max-context", "4096", "--kv", "q8_0"])
        self.assertEqual(got[6:], ["--expert-swaps", "0", "--temperature", "0", "--tokens", "1,2,3", "--max-new", "16"])

    def test_other_arguments_stay_in_order(self):
        args = ["--model", "m.gguf", "--draft-model", "d.gguf", "--conversation-cache-mib", "auto", "--layer-split",
                "auto"]
        self.assertEqual(R.regress_args(args, [5], 1)[:8], args)


class Ids(unittest.TestCase):
    def test_the_last_line_of_ids(self):
        self.assertEqual(R.parse_ids("strata-poly: loaded\n12,13,14\n"), [12, 13, 14])
        self.assertEqual(R.parse_ids("7\n"), [7])
        self.assertEqual(R.parse_ids("no ids here\n"), [])

    def test_the_expert_cache_of_a_run(self):
        err = ("strata-poly: layers 0 to 20 on card 0: 2000 of 4000 routed experts in VRAM (9.0 GiB)\n"
               "strata-poly: loaded in 31.2 s; 5120 of 9472 routed experts in VRAM (21.7 GiB), 11 CPU threads\n")
        self.assertEqual(R.cache_of(err), "5120 of 9472 routed experts in VRAM (21.7 GiB)")
        self.assertEqual(R.cache_of("nothing\n"), "?")

    def test_first_difference(self):
        self.assertEqual(R.first_difference([1, 2, 3], [1, 2, 3]), -1)
        self.assertEqual(R.first_difference([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(R.first_difference([1, 2], [1, 2, 3]), 2)


if __name__ == "__main__":
    unittest.main()
