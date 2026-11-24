"""Older NVIDIA GPUs (experimental): Pascal / Volta get strata-poly compiled with CUDA 12, chosen per model config by
its oldest card; --cuda 12|13 overrides it.  A PC whose cards CUDA 13 runs on gets exactly what it got before.  No
GPU, no downloads.

    python -m unittest tools.test_setup_older_gpus
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import setup  # noqa: E402
from test_setup_choices import quiet  # noqa: E402
from tools.poly_setup_harness import PolyMain  # noqa: E402


def card(i, name, vram, arch, driver="580.97"):
    return {"index": i, "name": name, "vram_gb": vram, "arch": arch, "driver": driver}


V100 = card(0, "Tesla V100-PCIE-32GB", 32.0, "70", "575.57")
P40 = card(0, "Tesla P40", 24.0, "61", "535.104")


class Choice(unittest.TestCase):
    def test_by_the_oldest_card(self):
        self.assertEqual(setup.cuda_choice(["120"]), (13, None))
        self.assertEqual(setup.cuda_choice([75, 86, 89]), (13, None))
        for archs in ([70], [61], [60], [70, 86]):
            tk, why = setup.cuda_choice(archs)
            self.assertEqual(tk, 12, archs)
            self.assertIn(f"sm_{min(archs)} is older than CUDA 13 supports", why)

    def test_the_override_is_honored_with_a_warning(self):
        tk, why = setup.cuda_choice([70], "13")
        self.assertEqual(tk, 13)                       # recommend, never force: kept, and said
        self.assertIn("will not run on that card", why)
        self.assertEqual(setup.cuda_choice([86], "12")[0], 12)
        self.assertIn("#220", setup.cuda_choice([86, 120], "12")[1])
        self.assertEqual(setup.cuda_choice([70], "auto")[0], 12)
        self.assertEqual(setup.cuda_choice([86], "auto"), (13, None))

    def test_configs(self):
        self.assertEqual(setup.config_toolkit({"exe": "/s/engine-poly/strata-poly"}), 13)
        self.assertEqual(setup.config_toolkit({"exe": "/s/engine-poly/strata-poly", "cuda": 12}), 12)


class OptIn(unittest.TestCase):
    """Pascal / Volta cards are used only when chosen; a PC with a newer card keeps recommending the newer one."""

    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"STRATA_EXPERIMENTAL_SM60": ""})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_reasons(self):
        rtx = card(0, "RTX 3090", 24.0, "86")
        v100 = {**V100, "index": 1}
        self.assertIsNone(setup.old_gpus_opt_in([rtx]))
        self.assertIsNone(setup.old_gpus_opt_in([rtx, v100]))                  # the default: unchanged
        self.assertIn("you chose GPU 1", setup.old_gpus_opt_in([rtx, v100], [1]))
        self.assertIsNone(setup.old_gpus_opt_in([rtx, v100], [0]))
        self.assertEqual(setup.old_gpus_opt_in([rtx, v100], cuda="12"), "--cuda 12")
        self.assertIn("only kind", setup.old_gpus_opt_in([V100]))
        self.assertIsNone(setup.old_gpus_opt_in([V100], other=True))            # an AMD card it can use instead
        self.assertIsNone(setup.old_gpus_opt_in([card(0, "GTX 980", 4.0, "52")]))
        with mock.patch.dict(os.environ, {"STRATA_EXPERIMENTAL_SM60": "1"}):
            self.assertEqual(setup.old_gpus_opt_in([rtx, v100]), "STRATA_EXPERIMENTAL_SM60=1")

    def test_named(self):
        self.assertEqual(setup.named_gpus(None, "0,2"), [0, 2])
        self.assertEqual(setup.named_gpus(1, None), [1])
        self.assertEqual(setup.named_gpus(None, "all"), [])
        self.assertEqual(setup.named_gpus(None, "x"), [])

    def test_the_gate_follows_the_opt_in(self):
        with mock.patch.object(setup, "OLD_GPUS", None):
            self.assertIsNotNone(setup.gpu_problem(V100))
        with mock.patch.object(setup, "OLD_GPUS", "you chose GPU 0"):
            self.assertIsNone(setup.gpu_problem(V100))
            self.assertIsNotNone(setup.gpu_problem(card(0, "GTX 980", 4.0, "52")))


class Libraries(unittest.TestCase):
    def test_cuda12_wheels_two_folders(self):
        with tempfile.TemporaryDirectory() as d:
            sp = Path(d) / "site-packages"
            names = {"cublas/bin/cublas64_12.dll", "cuda_runtime/bin/cudart64_12.dll", "cu13/bin/x86_64/cublas64_13.dll",
                     "cublas/lib/libcublas.so.12", "cuda_runtime/lib/libcudart.so.12", "cu13/lib/libcublas.so.13"}
            for n in names:
                f = sp / "nvidia" / n
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b"")
            with mock.patch.object(sys, "path", [str(sp)]):
                for win in (True, False):
                    with mock.patch.object(setup, "WIN", win):
                        got12 = sorted(Path(x).relative_to(sp / "nvidia").as_posix() for x in setup.cuda_lib_dirs(12))
                        got13 = [Path(x).relative_to(sp / "nvidia").as_posix() for x in setup.cuda_lib_dirs()]
                    if win:
                        self.assertEqual(got12, ["cublas/bin", "cuda_runtime/bin"])
                        self.assertEqual(got13, ["cu13/bin/x86_64"])
                    else:
                        self.assertEqual(got12, ["cublas/lib", "cuda_runtime/lib"])
                        self.assertEqual(got13, ["cu13/lib"])


class BuildTools(unittest.TestCase):
    """The compile path: a CUDA 12 engine looks for a 12.x toolkit (STRATA_NVCC picks one), sm_120 needs 12.8+."""

    def tools(self, gpu, nvcc):
        seen = []

        def find(below=None):
            seen.append(below)
            return nvcc(below)

        with mock.patch.object(setup, "find_nvcc", find), mock.patch.object(setup, "find_vcvars", lambda cuda_v=None: "vcvars"), \
                mock.patch.object(setup.shutil, "which", lambda n: "/usr/bin/" + n):
            got, text = quiet(setup.install_build_tools, gpu, True)
        return got, seen, text

    def test_cuda12_on_a_newer_card(self):
        got, seen, _ = self.tools({"arch": "86", "archs": [86], "toolkit": 12},
                                  lambda below: ("nvcc12", (12, 9)) if below else ("nvcc13", (13, 0)))
        self.assertEqual((got[0], seen), ("nvcc12", [(13, 0)]))

    def test_sm120_in_a_cuda12_engine_needs_12_8(self):
        got, _, text = self.tools({"arch": "70", "archs": [70, 120]},
                                  lambda below: ("nvcc12", (12, 9)) if below else ("nvcc13", (13, 0)))
        self.assertEqual(got[0], "nvcc12")
        self.assertNotIn("#220", text)                      # the Qwen engine's prompt-path warning is gone with it
        with self.assertRaises(SystemExit):
            self.tools({"arch": "70", "archs": [70, 120]}, lambda below: ("nvcc12", (12, 6)) if below else (None, None))


class Install(unittest.TestCase):
    """setup.main() on mocked PCs (tools/poly_setup_harness.py): the toolkit strata-poly is compiled with."""
    setUp, tearDown, main = PolyMain.setUp, PolyMain.tearDown, PolyMain.main

    def run_setup(self, ram, found, argv=(), env=None):
        self.toolkit = None
        code, out, cfg = self.main(list(argv), family="occamy", model="APEX-Quality", ram=ram, cards=found, env=env)
        return code, out, cfg

    def test_a_v100_alone_gets_cuda12(self):
        code, out, cfg = self.run_setup(95.0, [V100])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual((self.toolkit, cfg["cuda"]), (12, 12))
        self.assertIn("sm_70 is older than CUDA 13 supports", out)
        self.assertIn("experimental", out)

    def test_a_p40_with_an_older_driver(self):
        code, out, cfg = self.run_setup(95.0, [P40])
        self.assertEqual(code, 0, out[-3000:])                 # driver 535: enough for CUDA 12
        self.assertEqual(self.toolkit, 12)

    def test_a_newer_card_beside_a_v100_is_unchanged(self):
        """The RTX 3090 is recommended as before; the V100 is listed with how to choose it."""
        rtx = card(0, "NVIDIA GeForce RTX 3090", 24.0, "86")
        code, out, cfg = self.run_setup(95.0, [rtx, {**V100, "index": 1}])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(self.toolkit, 13)
        self.assertNotIn("cuda", cfg)
        self.assertIn("choose it with --gpu 1", out)

    def test_naming_the_v100_beside_a_newer_card(self):
        found = [card(0, "NVIDIA GeForce RTX 3090", 24.0, "86"), {**V100, "index": 1}]
        code, out, cfg = self.run_setup(95.0, found, ["--gpu", "1"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual((self.toolkit, cfg["cuda"], cfg["gpu"]), (12, 12, 1))
        self.assertIn("you chose GPU 1", out)

    def test_cuda12_asked_for_on_a_newer_card(self):
        found = [card(0, "NVIDIA GeForce RTX 4090", 24.0, "89")]
        for argv, env in ((["--cuda", "12"], None), ([], {"STRATA_CUDA": "12"})):
            with self.subTest(argv=argv, env=env):
                code, out, cfg = self.run_setup(128.0, found, argv, env)
                self.assertEqual(code, 0, out[-3000:])
                self.assertEqual((self.toolkit, cfg["cuda"]), (12, 12))
                self.assertIn("--cuda 12 (as you chose", out)

    def test_an_old_driver_is_told_about_cuda12(self):
        old = [card(0, "NVIDIA GeForce RTX 4090", 24.0, "89", "550.54")]
        code, out, _ = self.run_setup(127.8, old)
        self.assertEqual(code, 1)
        self.assertIn("580 or newer is needed", out)
        self.assertIn("--cuda 12", out)
        code, out, cfg = self.run_setup(127.8, old, ["--cuda", "12"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(self.toolkit, 12)

    def test_cuda13_forced_on_a_v100_is_said(self):
        code, out, cfg = self.run_setup(95.0, [V100], ["--cuda", "13"])
        self.assertIn("will not run on that card", out)


if __name__ == "__main__":
    unittest.main()
