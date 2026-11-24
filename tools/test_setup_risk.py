"""Tests for the owner's rule in setup.py - it recommends, it never forces: a choice bigger than the recommendation is
kept and its risk said (#406's 256K on 64 GB, #364/#384's two GPUs in the low-RAM mode, #403's RAM budget), and the
stops that stay are the ones a user can confirm (confirm_risk: --yes with an explicit flag, or y).  The defaults are
tools/test_setup_golden.py's (Occamy).  Mocked - no GPU, no downloads, nothing written outside a temp folder.

    python -m unittest tools.test_setup_risk
"""
from __future__ import annotations

import contextlib
import io
import itertools
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_golden import PROFILES, card, install  # noqa: E402


def arg(cfg, flag):
    a = cfg["args"]
    return a[a.index(flag) + 1] if flag in a else None


def run(fn, *args, stdin=None):
    """-> (exit code or None, printed text, questions asked); stdin answers every input() (None: never asked)."""
    out, asked, code = io.StringIO(), [], None

    def fake_input(prompt=""):
        asked.append(prompt)
        if stdin is None:
            raise AssertionError(f"asked {prompt!r} with --yes")
        return stdin

    with contextlib.redirect_stdout(out), mock.patch("builtins.input", fake_input):
        try:
            fn(*args)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), asked


class ConfirmRisk(unittest.TestCase):
    """S6: one helper for the stops a user may take on."""

    def go(self, explicit, yes, stdin=None):
        return run(setup.confirm_risk, "it may not fit", explicit, yes, "it does not fit", "a smaller size",
                   stdin=stdin)

    def test_yes_with_an_explicit_choice_is_consent(self):
        code, out, asked = self.go(True, True)
        self.assertIsNone(code)
        self.assertIn("it may not fit", out)
        self.assertEqual(asked, [])

    def test_yes_alone_keeps_the_stop(self):
        code, out, asked = self.go(False, True)
        self.assertEqual(code, 1)
        self.assertIn("it does not fit", out)
        self.assertIn("a smaller size", out)

    def test_interactive_default_is_no(self):
        for explicit in (True, False):
            code, out, asked = self.go(explicit, False, stdin="")
            self.assertEqual(code, 1)
            self.assertIn("[n]", asked[0])
            self.assertIsNone(self.go(explicit, False, stdin="y")[0])


class BrokenEarlierConfig(unittest.TestCase):
    """#459: a new copy of PolyStrata set up like an earlier install skips an earlier config that does not parse (an
    empty strata-*.json crashed START-HERE with JSONDecodeError) and goes on as a fresh install; configs are written
    whole (a temporary file moved over the old one)."""
    RAM64, GPU32 = PROFILES["64GB-1x32GB"]

    def setup_with(self, files):
        with tempfile.TemporaryDirectory() as d:
            for i, (name, text) in enumerate(files):                  # in order, oldest first
                (Path(d) / name).write_text(text, encoding="utf-8")
                os.utime(Path(d) / name, (1_700_000_000 + i, 1_700_000_000 + i))
            return install(self.RAM64, self.GPU32, [], extra=[
                mock.patch.object(setup, "other_installs", lambda settings: [Path(d)]),
                mock.patch.object(setup, "start", mock.Mock(return_value=0))])

    def test_an_empty_config_is_skipped(self):
        for text, why in (("", "the file is empty"), ("{\"args\": [", "not valid JSON"), ("[1]", "not a JSON object")):
            with self.subTest(text=text):
                code, out, cfg, _ = self.setup_with([("strata-occamy10-apex-quality.json", text)])
                self.assertEqual(code, 0, out)
                self.assertIn("skipped the earlier config ", out)
                self.assertIn(f"strata-occamy10-apex-quality.json ({why}): setting this copy up without it", out)
                self.assertNotIn("Found your earlier install", out)
                self.assertIsNotNone(cfg)                                # the fresh install's config

    def test_the_newest_readable_config_is_used(self):
        good = json.dumps({"args": ["--max-context", "65536"], "port": 8080})
        code, out, cfg, _ = self.setup_with([("strata-occamy10-apex-quality.json", good),
                                             ("strata-occamy10-apex-i-compact.json", "")])
        self.assertEqual(code, 0, out)
        self.assertIn("strata-occamy10-apex-i-compact.json (the file is empty)", out)
        self.assertIn("Found your earlier install", out)
        self.assertIn("(occamy10-apex-quality)", out)
        self.assertEqual(arg(cfg, "--max-context"), "65536")

    def test_write_config_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-iq3_s.json"
            p.write_text("old", encoding="utf-8")
            setup.write_config(p, {"args": ["--kv", "int8"]})
            self.assertEqual(p.read_text(encoding="utf-8"), json.dumps({"args": ["--kv", "int8"]}, indent=1))
            self.assertEqual([f.name for f in Path(d).iterdir()], ["strata-iq3_s.json"])   # no .tmp left
            with mock.patch.object(Path, "write_text", side_effect=OSError(28, "No space left on device")):
                with self.assertRaises(OSError):
                    setup.write_config(p, {"args": []})
            self.assertEqual(json.loads(p.read_text(encoding="utf-8")), {"args": ["--kv", "int8"]})   # kept whole


class StartOnSeveralGpus(unittest.TestCase):
    """--gpus on a start: saved in the config, nothing else changed."""

    def test_start_with_gpus(self):
        found = PROFILES["32GB-2x24GB"][1]
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / setup.POLY_EXE
            exe.write_bytes(b"")
            p = Path(d) / "strata-occamy10-apex-quality.json"
            p.write_text(json.dumps({"exe": str(exe), "args": ["--max-context", "32768"], "gpu": 0, "gpus_asked": True}))
            call = mock.Mock(return_value=0)
            with mock.patch.object(setup, "gpus", lambda: found), \
                    mock.patch.object(setup, "update_poly_engine", lambda *a, **k: None), \
                    mock.patch.object(setup, "poly_tail_skip", lambda *a, **k: None), \
                    mock.patch.object(setup, "poly_kv_whole", lambda *a, **k: None), \
                    mock.patch.object(setup.subprocess, "call", call):
                code, out, _ = run(setup.start, p, None, [0, 1], False, True)
            cfg = json.loads(p.read_text())
        self.assertIsNone(code, out)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertEqual(cfg["layer_split"], "auto")
        self.assertEqual(cfg["args"], ["--max-context", "32768"])   # nothing added on a start
        self.assertTrue(call.called)


class SmallCard(unittest.TestCase):
    """S9: a card under 8 GB named with --gpus is a risk to confirm, not a stop."""
    FOUND = [card(0, "NVIDIA GeForce RTX 3090", 24.0, "86"), card(1, "NVIDIA GeForce RTX 2060", 6.0, "75")]

    def test_named_with_yes(self):
        code, out, cfg, _ = install(63.7, self.FOUND, ["--family", "occamy", "--model", "APEX-Quality", "--no-start",
                                                       "--gpus", "0,1"])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertIn("GPU 1 (NVIDIA GeForce RTX 2060) has 6 GB of VRAM", out)
        self.assertIn("is used together with the others, as you chose", out)

    def test_named_and_enter_stops(self):
        code, out, cfg, asked = install(63.7, self.FOUND, ["--family", "occamy", "--model", "APEX-Quality",
                                                           "--no-start", "--gpus", "0,1"], answers="")
        self.assertEqual(code, 1)
        self.assertTrue(any("Use it anyway?" in q for q in asked), asked)

    def test_not_named_is_not_offered(self):
        code, out, cfg, _ = install(63.7, self.FOUND, ["--family", "occamy", "--model", "APEX-Quality", "--no-start"])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], 0)                                   # one card: as before


class SmallSecondCard(unittest.TestCase):
    """#448's rule (the bigger card alone when the second has under 12 GB) was measured on PolyStrata's Qwen engine
    and is gone: a pair that can share the model is recommended, one card alone is still there when named."""
    FOUND = [card(0, "NVIDIA RTX PRO 4500 Blackwell", 31.8, "120"), card(1, "NVIDIA GeForce RTX 3080", 10.0, "86")]

    def test_yes_takes_the_pair(self):
        code, out, cfg, _ = install(127.8, self.FOUND, ["--family", "occamy", "--model", "APEX-Quality", "--no-start"])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertNotIn("too small to lend a split", out)
        self.assertNotIn("20%", out)                                     # the Qwen engine's prompt figure is gone too

    def test_named_first_card(self):
        code, out, cfg, _ = install(127.8, self.FOUND, ["--family", "occamy", "--model", "APEX-Quality", "--no-start",
                                                        "--gpu", "0"])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], 0)


if __name__ == "__main__":
    unittest.main()
