"""tools/strata_inspect.py's verdict, offline: the fingerprints of the files setup installs (data/gguf_fingerprints.json,
made from setup's pinned files) stand in for headers read from the network.

    python -m unittest tools.test_strata_inspect
"""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
import strata_inspect as SI  # noqa: E402


class Verdict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = SI.load_fingerprints()

    def fp(self, key):
        return copy.deepcopy({k: v for k, v in self.table[key].items() if k not in ("files", "url")})

    def test_the_table_is_setups_files(self):
        want = [f"{f}/{m}" for f, d in setup.FAMILIES.items() for m in d["models"]]
        self.assertEqual([k for k in self.table if not k.startswith("_")], want)

    def test_every_installed_file_is_known(self):
        for key in (k for k in self.table if not k.startswith("_")):
            kind, sentence = SI.verdict(self.fp(key), self.table)
            fam, model = key.split("/")
            self.assertEqual(kind, "known", key)
            self.assertIn(f"--family {fam} --model {model}", sentence)

    def test_same_layout_other_file(self):
        fp = self.fp("occamy/APEX-Quality")
        fp["bytes"] += 1000                          # another file, the experts stored the same
        kind, sentence = SI.verdict(fp, self.table)
        self.assertEqual(kind, "layout")
        self.assertIn("occamy/APEX-Quality", sentence)

    def test_another_file_of_a_family_s_architecture(self):
        for key, title in (("laguna-s/UD-Q3_K_XL", "Laguna S 2.1"), ("laguna-xs/Q4_K_M", "Laguna XS 2.1"),
                           ("occamy/APEX-Quality", "Occamy 1.0 35B-A3B")):
            with self.subTest(key=key):
                fp = self.fp(key)
                fp["expert_bytes"] = {"Q5_K": 1}       # experts stored as in no file setup offers
                kind, sentence = SI.verdict(fp, self.table)
                self.assertEqual(kind, "family")
                self.assertTrue(sentence.startswith(title), sentence)
                self.assertEqual(SI.family_title(fp), setup.FAMILIES[key.split("/")[0]]["title"])

    def test_laguna_by_its_layers(self):
        fp = self.fp("laguna-s/UD-Q3_K_XL")
        fp["expert_bytes"], fp["layers"] = {"Q5_K": 1}, 64
        kind, sentence = SI.verdict(fp, self.table)
        self.assertEqual(kind, "no")
        self.assertIn("64 layers", sentence)

    def test_other_architecture(self):
        # gemma4 and nemotron_h_moe: the engine runs them, but their families left setup (2026-11-23)
        for arch in ("qwen4exp", "glm5-next", "deepseek4", "qwen3moe", "gemma4", "nemotron_h_moe"):
            with self.subTest(arch=arch):
                fp = self.fp("occamy/APEX-Quality")
                fp["arch"] = arch
                kind, sentence = SI.verdict(fp, self.table)
                self.assertEqual(kind, "no")
                self.assertIn("not a model Puddle runs", sentence)

    def test_the_family_titles_are_setups(self):
        titles = {d["title"] for d in setup.FAMILIES.values()}
        for t in SI.POLY_ARCHS.values():
            for title in (t.values() if isinstance(t, dict) else [t]):
                self.assertIn(title, titles)
        self.assertEqual(set(SI.POLY_ARCHS), {a for v in setup.POLY_GGUF_ARCHS.values() for a in v})


if __name__ == "__main__":
    unittest.main()
