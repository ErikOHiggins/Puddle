"""Tests for how setup.py installs a strata-poly family (docs/BACKENDS.md), with Occamy 1.0: the estimates for a card
that holds part of the experts, the install by poly_setup (no shipped expert profile, no draft block to fetch), the
image encoder, the size taken without a name and the context menu.  The families' own pins are in
tools/test_setup_puddle_families.py.  Until 2026-11-23 these tests ran on Gemma 4, whose family left setup then; its
own (a draft model beside the file, a 256K line in its menu, a size from another repository) went with it.  Every
outside effect is mocked by tools/poly_setup_harness.py.

    python -m unittest tools.test_setup_gguf_models
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from tools.poly_setup_harness import PolyMain  # noqa: E402

# family: (size, file) of the size setup takes when the PC has the RAM for it
NEW = {"occamy": ("APEX-Quality", "occamy-1.0-APEX-Quality.gguf")}
CARD_16 = 15.93                                   # what an RTX 5060 Ti 16 GB reports


class Pins(unittest.TestCase):
    def test_tags_tell_the_families_apart(self):
        tags = [d["tag"] for d in setup.FAMILIES.values() if d["tag"]]
        for a in tags:
            self.assertEqual([b for b in tags if b.startswith(a)], [a])


class Estimates(unittest.TestCase):
    def test_a_32_gb_card_holds_every_expert(self):
        for family in NEW:
            for d in setup.family_models(family).values():
                self.assertEqual(setup.poly_held_gb(d, 31.84), d["arena_gb"], family)
                self.assertEqual(setup.poly_ram_need(d, 31.84), setup.POLY_RAM_BESIDE_GB)

    def test_a_16_gb_card(self):
        # what the family's menu line says of a 16 GB card, and its sizes' ram_gb
        for f in NEW:
            sizes = setup.family_models(f)
            n = setup.poly_ram_need(next(iter(sizes.values())), 16.0)
            self.assertIn(f"~{n:.0f} GB of RAM beside a 16 GB card", setup.FAMILIES[f]["about"])
            for m, d in sizes.items():
                self.assertEqual(d["ram_gb"], round(setup.poly_ram_need(d, 16.0)), m)

    def test_check_lines(self):
        d = setup.FAMILIES["occamy"]["models"]["APEX-Quality"]
        self.assertIn("fits by setup's estimate (~18 GB of RAM", setup.poly_verdict(d, 32, 16.0, False))
        self.assertIn("read from the SSD", setup.poly_verdict(d, 12, 16.0, False))
        self.assertIn("does not fit", setup.poly_verdict(d, 64, 6.0, False))
        with mock.patch.object(setup, "WIN", False):   # #11: AMD on Linux runs it
            self.assertNotIn("does not run", setup.poly_verdict(d, 64, 24.0, True))
        with mock.patch.object(setup, "WIN", True):     # a family that has run on Windows is not said to be untried
            self.assertIn("AMD", setup.poly_verdict(d, 64, 24.0, True))           # only AMD on Windows is left out
            self.assertIn("not tried on Windows", setup.poly_verdict(d, 32, 16.0, False))
            self.assertNotIn("not tried on Windows", setup.poly_verdict(d, 32, 16.0, False, True))


class Config(unittest.TestCase):
    def test_choices_from_config(self):
        with tempfile.TemporaryDirectory() as t:
            for family, (size, _file) in NEW.items():
                p = Path(t) / f"strata-{setup.FAMILIES[family]['tag']}{size.lower()}.json"
                p.write_text(json.dumps({"exe": "/x/engine-poly/" + setup.POLY_EXE, "args": ["--max-context", "8192"]}))
                ch = setup.choices_from_config(p)
                self.assertEqual((ch["family"], ch["model"], ch["context"]), (family, size, 8192))


class Main(unittest.TestCase):
    """setup.main() for the family, every outside effect mocked."""
    setUp, tearDown, main = PolyMain.setUp, PolyMain.tearDown, PolyMain.main

    def test_install(self):
        for family, (size, file) in NEW.items():
            with self.subTest(family):
                self.downloads, self.runs, self.verified = [], [], []
                code, out, cfg = self.main(family=family, ram=32.0, vram=CARD_16)
                self.assertEqual(code, 0, out)
                fam = setup.FAMILIES[family]
                self.assertEqual(self.downloads, [fam["hf"] + file])
                self.assertEqual(self.verified, [(file, *fam["sha256"][file])])
                tag = fam["tag"] + size
                pack = self.t / "data" / "packs" / tag.lower()
                # the tokenizer alone: no draft block is fetched
                self.assertEqual([Path(r[1]).name for r in self.runs], ["strata_tokenizer.py"])
                self.assertEqual(self.runs[0][self.runs[0].index("--out") + 1], str(pack))
                args = cfg["args"]
                self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / tag / file)
                self.assertEqual(Path(args[args.index("--expert-profile") + 1]), pack / "expert-profile.txt")
                self.assertFalse((pack / "expert-profile.txt").exists())   # none is shipped: the engine starts it
                self.assertNotIn("--draft-model", args)
                self.assertNotIn("--tail-skip", args)
                self.assertEqual(cfg["model_name"], f"{fam['name']}-{size.lower()}")
                self.assertEqual(cfg["display_name"], f"{fam['title']} {size}")
                self.assertEqual(Path(cfg["exe"]), self.t / setup.ENGINE_POLY_DIR / setup.POLY_EXE)
                self.assertIn("experts fit in VRAM (100% on a 32 GB card)", out)
                self.assertNotIn("has run on no card", out)   # it has run on 16 GB cards
                self.assertEqual((self.encoder, cfg.get("vision")), ("none", None))   # --yes: no images

    def test_a_card_smaller_than_it_has_run_on_is_said(self):
        code, out, cfg = self.main(family="occamy", ram=32.0, vram=8188 / 1024)   # what an 8 GB card reports
        self.assertEqual(code, 0, out)
        self.assertIn("has run on no card under 16 GB", out)

    def test_a_hybrid_cpu_gets_a_thread_count(self):
        """A hybrid CPU gets setup's count, the performance cores but one and half of the efficiency cores."""
        code, out, cfg = self.main(family="occamy", ram=32.0, vram=CARD_16, cores=(8, 12))
        self.assertEqual(code, 0, out)
        args = cfg["args"]
        self.assertEqual(args[args.index("--threads") + 1], "13")
        self.assertIn("hybrid CPU (8 performance + 12 efficiency cores): 13 CPU threads", out)
        for cores in (None, (8, 8)):
            code, out, cfg = self.main(family="occamy", ram=32.0, vram=CARD_16, cores=cores)
            self.assertEqual(code, 0, out)
            self.assertNotIn("--threads", cfg["args"])

    def test_images(self):
        for family, (size, file) in NEW.items():
            fam = setup.FAMILIES[family]
            folder = self.t / "models" / (fam["tag"] + size)
            # (the card's VRAM, --vision) -> where the encoder runs: a small card keeps its VRAM for the model
            for vram, asked, where in ((31.84, "yes", "gpu"), (8.0, "yes", "cpu"), (12.0, "gpu", "gpu")):
                with self.subTest(family=family, vram=vram, asked=asked):
                    self.downloads, self.verified = [], []
                    code, out, cfg = self.main(["--vision", asked], family=family, ram=64.0, vram=vram)
                    self.assertEqual(code, 0, out)
                    self.assertEqual(self.encoder, where)
                    if vram > 30:                      # the first run: the encoder is the last file fetched
                        self.assertEqual(self.downloads[-1], fam["mmproj_hf"] + fam["mmproj"])
                    self.assertEqual(self.verified[-1], (fam["mmproj"], *fam["sha256"][fam["mmproj"]]))
                    v = cfg["vision"]
                    self.assertEqual((Path(v["exe"]), Path(v["mmproj"]), Path(v["model"]), v["gpu"]),
                                     (self.t / setup.ENGINE_POLY_DIR / setup.VEXE, folder / fam["mmproj"],
                                      folder / file, where == "gpu"))
                    self.assertEqual(v["max_tokens"], setup.VISION[where]["max_tokens"])
                    self.assertEqual("threads" in v, where == "cpu")
                    self.assertIn("images: on" + (" (encoder on the CPU)" if where == "cpu" else ""), out)
            # the encoder was downloaded once, beside the model's file, and found there afterwards
            self.assertTrue((folder / fam["mmproj"]).exists())
            self.assertEqual(self.downloads, [])
        # the encoder on the GPU leaves less VRAM to the experts, and the estimate says so
        need = {}
        for asked in ("no", "gpu"):
            code, out, cfg = self.main(["--vision", asked], family="occamy", ram=64.0, vram=16.0, answers={})
            need[asked] = out
        self.assertIn("about 12 of", need["no"])
        self.assertIn("about 11 of", need["gpu"])
        # and on a card it leaves too little of, setup stops and says how to keep it off the card
        code, out, cfg = self.main(["--vision", "gpu"], family="occamy", ram=64.0, vram=7.9)
        self.assertNotEqual(code, 0)
        self.assertIn("and its image encoder 1.6 GB", out)
        self.assertIn("--vision cpu keeps the image encoder off the card", out)

    def test_images_are_asked_about(self):
        code, out, cfg = self.main(family="occamy", ram=64.0, vram=24.0, answers={"Do you want images": "y"})
        self.assertEqual(code, 0, out)
        self.assertIn("a 0.9 GB", out)
        self.assertEqual((self.encoder, cfg["vision"]["gpu"]), ("gpu", True))

    def test_a_pc_with_24_gb_of_ram(self):
        code, out, cfg = self.main(family="occamy", ram=24.0, vram=CARD_16)   # enough by the family's own estimate
        self.assertEqual(code, 0, out)
        self.assertIn("RAM: 24 GB", out)
        self.assertIsNotNone(cfg)

    def test_too_little_ram_for_the_model_stops(self):
        code, out, cfg = self.main(family="occamy", ram=7.0, vram=CARD_16)   # too little for its smallest size too
        self.assertNotEqual(code, 0)
        self.assertIn("wants about 12 GB", out)
        self.assertIsNone(cfg)
        code, out, cfg = self.main(family="occamy", ram=7.0, vram=CARD_16, model="APEX-Quality")   # named: a risk
        self.assertEqual(code, 0, out)
        self.assertIn("as you chose", out)

    def test_the_size_taken_without_a_name(self):
        """The first size where the PC has the RAM for it, the smaller one where not; the larger one only by name."""
        # a size is taken unasked from 4 GB below its estimate (poly_default): APEX-Quality ~18 GB, APEX-I-Compact ~12
        for ram, size in ((64.0, "APEX-Quality"), (15.0, "APEX-Quality"), (14.0, "APEX-I-Compact")):
            code, out, cfg = self.main(family="occamy", ram=ram, vram=CARD_16)
            self.assertEqual(code, 0, out)
            self.assertIn(f"size: {size}", out, ram)
        code, out, cfg = self.main(family="occamy", ram=64.0, vram=CARD_16, model="APEX-Balanced")
        self.assertEqual(code, 0, out)
        self.assertIn("size: APEX-Balanced", out)

    def test_the_context_note_is_the_familys(self):
        code, out, cfg = self.main(family="occamy", ram=32.0, vram=CARD_16, answers={"Context?": "3"})
        self.assertEqual(code, 0, out)
        self.assertIn("0.9 GB more VRAM", out)                           # 65536: 32768 tokens of 26624 bytes more
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "65536")

    def test_the_families_keep_four_lengths(self):
        for family in setup.FAMILIES:
            with self.subTest(family=family):
                self.assertNotIn("contexts", setup.FAMILIES[family])   # none has a menu of its own now
        code, out, cfg = self.main(family="laguna-xs", ram=64.0, vram=16.0)
        self.assertEqual(code, 0, out)
        self.assertIn("4) 128K tokens", out)
        self.assertNotIn("256K tokens", out)


if __name__ == "__main__":
    unittest.main()
