"""The owner's rule for setup (#403 #406 #364 #384): setup RECOMMENDS, it never forces - and the recommended answers
stay exactly what they were.  setup.main() with --yes (and with Enter for every question) on a few PCs writes the same
run config, byte for byte, as the recorded baseline (tools/test_setup_golden.json).  Every outside effect is mocked
(tools/poly_setup_harness.py): no GPU, no downloads, nothing written outside a temp folder.

The baseline was recorded again for Puddle's families (phase 1b-1, 2026-11-19: the Qwen engine's families are gone;
phase 1c: Laguna-S, Laguna-XS and Occamy added; T2, 2026-11-20: Occamy removed; B4, 2026-11-20: Nemotron 3.5 Bolt
added, the other entries unchanged; phase 2, 2026-11-21: Nemotron 3.5 Super added, the other entries unchanged;
2026-11-22: Super's MTP block runs, its entries without --no-draft, the other entries unchanged; 2026-11-22:
Super's VRAM figures measured on echo, so 47 GB of RAM beside two 16 GB cards no longer takes it unasked;
2026-11-22: Bolt's UD-Q5_K_XL added, the other entries unchanged);
record it again only when a default changes on purpose.

    python -m unittest tools.test_setup_golden
    python tools/test_setup_golden.py --write      (only to record a new baseline on purpose)
"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from tools.poly_setup_harness import PolyMain  # noqa: E402

GOLDEN = Path(__file__).with_name("test_setup_golden.json")


def card(i, name, vram, arch):
    return {"index": i, "name": name, "vram_gb": vram, "arch": arch, "driver": "580.97"}


PROFILES = {
    "64GB-1x32GB": (63.7, [card(0, "NVIDIA GeForce RTX 5090", 31.8, "120")]),                  # #406
    "32GB-2x24GB": (31.9, [card(i, "NVIDIA GeForce RTX 3090", 24.0, "86") for i in range(2)]),  # #364
    "47GB-2x16GB": (47.0, [card(i, "NVIDIA GeForce RTX 5060 Ti", 15.9, "120") for i in range(2)]),   # #384
    "96GB-1x16GB": (95.8, [card(0, "NVIDIA GeForce RTX 5080", 15.9, "120")]),
    "128GB-1x24GB": (127.8, [card(0, "NVIDIA GeForce RTX 4090", 24.0, "89")]),
}
# (family, model): None = the recommended size
MODELS = [x for f in setup.FAMILIES for x in [(f, None), *((f, m) for m in setup.family_models(f))]]


def normalize(v, t: Path):
    """The config with the temp folder and the platform's names taken out (it compares on Windows and Linux)."""
    if isinstance(v, dict):
        return {k: normalize(x, t) for k, x in v.items()}
    if isinstance(v, list):
        return [normalize(x, t) for x in v]
    if isinstance(v, str):
        v = v.replace(str(t), "<T>").replace("\\", "/")
        for exe in ("strata-poly", "strata-vision", "llama-server"):  # the whole executable name only
            v = re.sub(rf"(?<![\w.-]){exe}(?:\.exe)?(?![\w.-])", f"<{exe}>", v)
        return v
    return v


class _Run(PolyMain):
    """The harness outside a test case."""


def install(ram, found, argv, answers=None, extra=(), avx512=False, configs=()):
    """setup.main() on a mocked PC -> (exit code, printed text, the written config or None, the questions asked).
    answers: None = --yes (a question fails the run), "" = Enter for every question, or {words of a question: its
    answer} (Enter for the others).  extra: more patches (mock.patch objects).  configs: (name, dict) run configs
    already in the Puddle folder.  argv names the family and --no-start where it wants them."""
    r = _Run()
    r.setUp()
    try:
        code, out, cfg = r.main(list(argv), ram=ram, cards=found, family=None, no_start=False,
                                answers={} if answers == "" else answers, extra=extra, avx512=avx512,
                                configs=configs)
        return code, out, (normalize(cfg, r.t) if cfg is not None else None), r.asked
    finally:
        r.tearDown()


def argv_for(family, model):
    return ["--family", family, "--no-start"] + (["--model", model] if model else [])


def record() -> dict:
    """Every profile x model with --yes: the exit code and the config (the baseline this test compares against)."""
    res = {}
    for prof, (ram, found) in PROFILES.items():
        for family, model in MODELS:
            code, out, cfg, _ = install(ram, found, argv_for(family, model))
            res[f"{prof} {family} {model or 'recommended'}"] = {"code": code, "config": cfg}
    return res


class Golden(unittest.TestCase):
    """--yes and Enter: the same config as the baseline wherever the baseline wrote one."""

    @classmethod
    def setUpClass(cls):
        cls.golden = json.loads(GOLDEN.read_text(encoding="utf-8"))

    def cases(self):
        for prof, (ram, found) in PROFILES.items():
            for family, model in MODELS:
                key = f"{prof} {family} {model or 'recommended'}"
                if self.golden[key]["code"] == 0:
                    yield key, ram, found, family, model

    def test_the_baseline_covers_every_kind_of_pc(self):
        self.assertEqual(sorted(self.golden), sorted(f"{p} {f} {m or 'recommended'}" for p in PROFILES for f, m in MODELS))
        written = [k for k, v in self.golden.items() if v["code"] == 0]
        self.assertGreaterEqual(len(written), 50)
        for prof in PROFILES:                                        # every PC gets the recommended size
            self.assertEqual(self.golden[f"{prof} gemma4 recommended"]["code"], 0, prof)

    def test_yes(self):
        for key, ram, found, family, model in self.cases():
            with self.subTest(key):
                code, out, cfg, asked = install(ram, found, argv_for(family, model))
                self.assertEqual(code, 0, out[-3000:])
                self.assertEqual(cfg, self.golden[key]["config"], out[-3000:])
                self.assertEqual(asked, [])

    def test_enter_for_every_question(self):
        """Enter takes every recommended answer.  A size named with --model that setup expects to run badly is the
        one difference: --yes with the flag is consent, Enter at "Go on anyway?" (default n) is not."""
        for key, ram, found, family, model in self.cases():
            with self.subTest(key):
                code, out, cfg, asked = install(ram, found, argv_for(family, model), answers="")
                if code != 0 and model is not None and any("Go on anyway?" in q for q in asked):
                    self.assertIn("as you chose", install(ram, found, argv_for(family, model))[1])
                    continue
                self.assertEqual(code, 0, out[-3000:])
                self.assertEqual(cfg, self.golden[key]["config"], out[-3000:])

    def test_the_refusals_with_yes_alone_stay(self):
        """A size the baseline refused with --yes and no explicit --model still stops (--yes alone is no consent)."""
        for prof, (ram, found) in PROFILES.items():
            for family, model in MODELS:
                key = f"{prof} {family} {model or 'recommended'}"
                if self.golden[key]["code"] != 0 and model is None:
                    with self.subTest(key):
                        self.assertNotEqual(install(ram, found, argv_for(family, model))[0], 0)


if __name__ == "__main__":
    if "--write" in sys.argv:
        GOLDEN.write_text(json.dumps(record(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {GOLDEN}")
    else:
        unittest.main()
