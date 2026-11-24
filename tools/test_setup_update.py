"""Tests for `setup.py --update` (#475, UPDATE.bat / update.sh): it refreshes what a start would - the Python packages,
the strata-poly engine (compiled again when its source changed) and each model's config - and never starts the model.  Every outside effect is mocked: no
GPU, no downloads, nothing written outside a temp folder.

    python -m unittest tools.test_setup_update
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class Update(unittest.TestCase):
    def config(self, d: Path, **extra) -> Path:
        p = d / "strata-occamy10-apex-quality.json"
        p.write_text(json.dumps({"model_name": "Occamy 1.0 35B-A3B APEX-Quality",
                                 "exe": str(d / "engine-poly" / setup.POLY_EXE),
                                 "args": ["--model", "m.gguf"], **extra}), encoding="utf-8")
        return p

    def run_update(self, have, argv=()):
        a = mock.Mock(**dict(argv))
        out = io.StringIO()
        with mock.patch.object(setup, "pip_install") as pip, \
                mock.patch.object(setup, "update_poly_engine") as eng, \
                mock.patch.object(setup, "poly_tail_skip") as tail, \
                mock.patch.object(setup, "poly_kv_whole") as whole, \
                mock.patch.object(setup, "start") as start, \
                mock.patch("subprocess.call") as call, \
                contextlib.redirect_stdout(out):
            rc = setup.update_install(have, a)
        return rc, pip, eng, (tail, whole), start, call, out.getvalue()

    def test_refreshes_without_starting(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.config(Path(d))
            rc, pip, eng, (tail, whole), start, call, out = self.run_update([p])
        self.assertEqual(rc, 0)
        pip.assert_called_once()
        eng.assert_called_once()                                    # the model's engine, compiled again if stale
        tail.assert_called_once()
        whole.assert_called_once()
        start.assert_not_called()
        call.assert_not_called()
        self.assertIn("Puddle is updated", out)

    def test_a_config_of_the_qwen_engine_is_left_alone(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-iq3_s.json"
            p.write_text(json.dumps({"exe": str(Path(d) / "engine" / "strata"), "args": ["--native", "m.gguf"]}),
                         encoding="utf-8")
            before = p.read_text(encoding="utf-8")
            rc, pip, eng, (tail, whole), start, call, out = self.run_update([p])
            self.assertEqual(p.read_text(encoding="utf-8"), before)
        self.assertEqual(rc, 0)
        for m in (eng, tail, whole, start, call):
            m.assert_not_called()
        self.assertIn("strata-iq3_s.json: for an engine Puddle does not have (strata); left as it is", out)

    def test_start_refuses_a_config_of_the_qwen_engine(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-iq3_s.json"
            p.write_text(json.dumps({"exe": str(Path(d) / "strata"), "args": []}), encoding="utf-8")
            with mock.patch.object(setup, "update_poly_engine", side_effect=AssertionError("engine")), \
                    mock.patch("subprocess.call", side_effect=AssertionError("started")), \
                    contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
                setup.start(p, None, open_browser=False, yes=True)
        self.assertIn("for an engine Puddle does not have", out.getvalue())

    def test_nothing_installed(self):
        rc, pip, eng, (tail, whole), start, call, out = self.run_update([])
        self.assertEqual(rc, 0)
        for m in (pip, eng, tail, whole, start, call):
            m.assert_not_called()
        self.assertIn("START-HERE.bat", out)

    def test_a_json_that_is_no_model_config_is_skipped(self):
        """#549: a strata-*.json without "args" (not written by setup) stopped update.sh with KeyError: 'args'."""
        with tempfile.TemporaryDirectory() as d:
            p = self.config(Path(d))
            other = Path(d) / "strata-notes.json"
            other.write_text(json.dumps({"note": "mine"}), encoding="utf-8")
            broken = Path(d) / "strata-cut.json"
            broken.write_text("{\"args\": [", encoding="utf-8")
            rc, pip, eng, _, start, call, out = self.run_update([other, broken, p])
        self.assertEqual(rc, 0, out)
        self.assertIn('skipped strata-notes.json (no "exe" or "args"): it is not a Puddle model config', out)
        self.assertIn("skipped strata-cut.json (not valid JSON)", out)
        eng.assert_called_once()                                    # the real model is still refreshed
        self.assertIn("Occamy 1.0 35B-A3B APEX-Quality: up to date", out)
        self.assertIn("Puddle is updated", out)

    def test_installed_configs_lists_only_model_configs(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.config(Path(d))
            (Path(d) / "strata-notes.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(setup, "ROOT", Path(d)), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(setup.installed_configs(), [p])

    def test_main_update_never_starts(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.config(Path(d))
            with mock.patch.object(sys, "argv", ["setup.py", "--update"]), \
                    mock.patch.object(setup, "data_folder", return_value=(Path(d), [])), \
                    mock.patch.object(setup, "installed_configs", return_value=[p]), \
                    mock.patch.object(setup, "update_install", return_value=0) as up, \
                    mock.patch.object(setup, "start") as start, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(setup.main(), 0)
        up.assert_called_once()
        start.assert_not_called()


class EngineTests(unittest.TestCase):
    """./update.sh --tests [REGEX]: strata-poly with its tests in build-poly-tests/, run by ctest, after the update; the
    installed engine is not touched.  The build and ctest are mocked."""

    def run_tests(self, meta, pattern="", rc=0):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "engine-poly").mkdir()
            (root / "engine-poly" / "BUILD.json").write_text(json.dumps(meta), encoding="utf-8")
            bindir = root / "bin"
            bindir.mkdir()
            (bindir / ("ctest.exe" if setup.WIN else "ctest")).write_bytes(b"")
            cmake = str(bindir / ("cmake.exe" if setup.WIN else "cmake"))
            out = io.StringIO()
            with mock.patch.object(setup, "ROOT", root), \
                    mock.patch.object(setup, "install_build_tools", return_value=("/cuda/bin/nvcc", None)) as tools, \
                    mock.patch.object(setup, "get_llama_cpp", return_value=Path("/llama")), \
                    mock.patch.object(setup, "cmake_build") as build, \
                    mock.patch.object(setup, "find_tool", return_value=cmake), \
                    mock.patch.object(setup, "run", return_value=mock.Mock(returncode=rc)) as run, \
                    contextlib.redirect_stdout(out):
                got = setup.run_poly_tests(pattern)
            return got, tools, build, run, root, out.getvalue()

    META = {"source": "local", "archs": [120], "toolkit": 13, "cuda_dirs": ["/cuda/lib64"]}

    def test_builds_in_its_own_folder_and_runs_ctest(self):
        got, tools, build, run, root, out = self.run_tests(self.META, "nemotron")
        self.assertEqual(got, 0)
        self.assertEqual(tools.call_args[0][0]["archs"], [120])
        src, bdir, target, defs = build.call_args[0][:4]
        self.assertEqual((src, bdir, target), (root, root / "build-poly-tests", "all"))
        self.assertIn("-DSTRATA_BUILD_TESTS=ON", defs)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=120", defs)
        self.assertIn("-DCMAKE_CUDA_COMPILER=/cuda/bin/nvcc", defs)
        cmd = [str(x) for x in run.call_args[0][0]]
        self.assertTrue(Path(cmd[0]).name.startswith("ctest"))
        self.assertEqual(cmd[1:3], ["--test-dir", str(root / "build-poly-tests")])
        self.assertEqual(cmd[-2:], ["-R", "nemotron"])
        env = run.call_args[1]["env"]
        self.assertTrue(env["PATH" if setup.WIN else "LD_LIBRARY_PATH"].startswith("/cuda/lib64"))
        self.assertIn("tests passed", out)

    def test_every_test_without_a_pattern_and_a_failure_is_said(self):
        got, _, _, run, _, out = self.run_tests(self.META, "", rc=8)
        self.assertEqual(got, 8)
        self.assertNotIn("-R", [str(x) for x in run.call_args[0][0]])
        self.assertIn("did not all pass (ctest exit 8)", out)

    def test_cuda12_engine_gets_its_own_folder(self):
        _, _, build, _, root, _ = self.run_tests({**self.META, "archs": [61], "toolkit": 12})
        self.assertEqual(build.call_args[0][1], root / "build-poly-tests-cuda12")
        self.assertIn("-DSTRATA_EXPERIMENTAL_SM60=ON", build.call_args[0][3])

    def test_needs_an_engine_compiled_here(self):
        for meta in ({}, {"source": "release", "archs": [120]}, {"backend": "hip", "archs": ["gfx1100"]}):
            with self.subTest(meta=meta), self.assertRaises(SystemExit):
                self.run_tests(meta)

    def test_update_runs_them_only_when_asked(self):
        with tempfile.TemporaryDirectory() as d:
            p = Update.config(self, Path(d))
            for tests, rc, want in ((None, 0, 0), ("", 0, 0), ("mamba", 3, 3)):
                with self.subTest(tests=tests), \
                        mock.patch.object(setup, "run_poly_tests", return_value=rc) as rt, \
                        mock.patch.object(setup, "pip_install"), mock.patch.object(setup, "update_poly_engine"), \
                        mock.patch.object(setup, "poly_tail_skip"), mock.patch.object(setup, "poly_kv_whole"), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(setup.update_install([p], mock.Mock(tests=tests)), want)
                    if tests is None:
                        rt.assert_not_called()
                    else:
                        rt.assert_called_once_with(tests)

    def test_tests_without_update_is_refused(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(sys, "argv", ["setup.py", "--tests"]), \
                mock.patch.object(setup, "data_folder", return_value=(Path(d), [])), \
                mock.patch.object(setup, "installed_configs", return_value=[]), \
                mock.patch.object(setup, "run_poly_tests") as rt, \
                contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
            setup.main()
        rt.assert_not_called()
        self.assertIn("--tests goes with --update", out.getvalue())


class LlamaServerOption(unittest.TestCase):
    """./update.sh --llama-server: llama-server at the pin for every NVIDIA card, the reference tools/poly_parity.py
    checks strata-poly against (no family of setup builds it since Bolt's opt-in left)."""
    CARDS = [{"index": 0, "name": "NVIDIA GeForce RTX 5060 Ti", "vram_gb": 15.93, "arch": "120", "driver": "580.97"},
             {"index": 1, "name": "NVIDIA GeForce RTX 4090", "vram_gb": 23.99, "arch": "89", "driver": "580.97"}]

    def update(self, have, cards=CARDS, **a):
        with mock.patch.object(setup, "gpus", return_value=cards), \
                mock.patch.object(setup, "build_llama_server", return_value=Path("engine-llama")) as build, \
                mock.patch.object(setup, "pip_install"), mock.patch.object(setup, "update_poly_engine"), \
                mock.patch.object(setup, "poly_tail_skip"), mock.patch.object(setup, "poly_kv_whole"), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            rc = setup.update_install(have, mock.Mock(**{"tests": None, **a}))
        return rc, build, out.getvalue()

    def test_built_for_every_card_when_asked(self):
        with tempfile.TemporaryDirectory() as d:
            p = Update.config(self, Path(d))
            rc, build, _ = self.update([p], llama_server=True, yes=False)
            self.assertEqual(rc, 0)
            gpu, yes, toolkit = build.call_args.args
            self.assertEqual((gpu["archs"], yes, toolkit), ([89, 120], False, 13))
            p.write_text(json.dumps({**json.loads(p.read_text(encoding="utf-8")), "cuda": 12}), encoding="utf-8")
            _, build, _ = self.update([p], llama_server=True, yes=True)
            self.assertEqual(build.call_args.args[1:], (True, 12))      # the installed engine's CUDA 12

    def test_also_with_nothing_installed(self):
        rc, build, out = self.update([], llama_server=True)
        self.assertEqual(rc, 0)
        build.assert_called_once()
        self.assertIn("No model is installed", out)

    def test_not_built_unless_asked(self):
        with tempfile.TemporaryDirectory() as d:
            p = Update.config(self, Path(d))
            for a in ({"llama_server": False}, {}):               # a Mock's own attribute is not True
                with self.subTest(a=a):
                    _, build, _ = self.update([p], **a)
                    build.assert_not_called()

    def test_no_nvidia_card_is_said(self):
        with self.assertRaises(SystemExit):
            _, _, out = self.update([], cards=[], llama_server=True)

    def test_without_update_is_refused(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(sys, "argv", ["setup.py", "--llama-server"]), \
                mock.patch.object(setup, "data_folder", return_value=(Path(d), [])), \
                mock.patch.object(setup, "installed_configs", return_value=[]), \
                mock.patch.object(setup, "build_llama_server") as build, \
                contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
            setup.main()
        build.assert_not_called()
        self.assertIn("--llama-server goes with --update", out.getvalue())


class SettingsLine(unittest.TestCase):
    """#564: a start prints the settings it uses (the engine options without the model's paths, and the server's
    fields), so a change made by hand to strata-<model>.json shows without reading the log."""

    CFG = {"exe": "x", "host": "0.0.0.0", "port": 8081, "api_key": "secret", "gpu": [0, 1], "fit_max_tokens": True,
           "args": ["--native", "E:\\PolyStrata\\packs\\iq3_s", "--mtp", "/s/mtp/rt", "m.gguf", "--kv", "int8",
                    "--kv-resident", "32768", "--spec-min-p", "0.5", "--vram-reserve-mib", "2048", "--mmap-experts",
                    "--prefill", "auto"]}

    def test_summary(self):
        s = setup.settings_summary(self.CFG)
        self.assertEqual(s, "--kv int8 --kv-resident 32768 --spec-min-p 0.5 --vram-reserve-mib 2048 --mmap-experts "
                            "--prefill auto; server 0.0.0.0:8081, api key set, gpu 0,1, fit_max_tokens true")
        self.assertNotIn("secret", s)
        self.assertIn("127.0.0.1:9000", setup.settings_summary({"args": []}, 9000))

    def test_a_start_prints_it(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / setup.POLY_EXE
            exe.write_bytes(b"")
            p = Path(d) / "strata-iq3_s.json"
            args = [x for x in self.CFG["args"] if x != "m.gguf"]           # no model file here
            p.write_text(json.dumps({**self.CFG, "exe": str(exe), "gpu": 0, "args": args}), encoding="utf-8")
            out = io.StringIO()
            with mock.patch.object(setup, "gpus", lambda: []), \
                    mock.patch.object(setup, "update_poly_engine", lambda *a, **k: None), \
                    mock.patch.object(setup, "poly_tail_skip", lambda *a, **k: None), \
                    mock.patch.object(setup, "poly_kv_whole", lambda *a, **k: None), \
                    mock.patch.object(setup.subprocess, "call", return_value=0), \
                    contextlib.redirect_stdout(out):
                setup.start(p, None, open_browser=False, yes=True)
        text = " ".join(out.getvalue().split())
        self.assertIn("Settings (strata-iq3_s.json): --kv int8 --kv-resident 32768", text)
        self.assertIn("--vram-reserve-mib 2048", text)


class TailSkip(unittest.TestCase):
    """A config from before its family got "tail_skip" gets --tail-skip on update, once the engine knows it.  (Qwen3.6
    had it; a stand-in family keeps the rule tested.)"""

    def setUp(self):
        fams = mock.patch.dict(setup.FAMILIES, {"tsk": {"name": "a-model", "title": "A model", "tail_skip": 6}})
        fams.start()
        self.addCleanup(fams.stop)

    def upgraded(self, name, version, args=()):
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d) / "engine-poly"
            eng.mkdir()
            (eng / "BUILD.json").write_text(json.dumps({"version": version}), encoding="utf-8")
            p = Path(d) / "strata-x.json"
            cfg = {"model_name": name, "exe": str(eng / setup.POLY_EXE), "args": ["--model", "m.gguf", *args]}
            p.write_text(json.dumps(cfg), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                setup.poly_tail_skip(cfg, p)
            return json.loads(p.read_text(encoding="utf-8"))["args"]

    def test_its_family_gets_it(self):
        self.assertEqual(self.upgraded("a-model-ud-q2_k_xl", "0.1.42")[-2:], ["--tail-skip", "6"])

    def test_an_older_engine_or_another_family_does_not(self):
        for name, version in (("a-model-ud-q3_k_m", "0.1.41"), ("gemma-4-26b-a4b-ud-q3_k_m", "0.1.42")):
            with self.subTest(name=name, version=version):
                self.assertNotIn("--tail-skip", self.upgraded(name, version))

    def test_a_value_the_user_set_stays(self):
        args = self.upgraded("a-model-ud-q3_k_m", "0.1.42", ["--tail-skip", "0"])
        self.assertEqual(args.count("--tail-skip"), 1)
        self.assertEqual(args[-1], "0")


class KvWhole(unittest.TestCase):
    """A strata-poly config whose engine is the AMD build gets --kv-whole on update (issue #13); a CUDA one does not."""

    def upgraded(self, backend, args=()):
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d) / "engine-poly"
            eng.mkdir()
            meta = {"version": "0.1.42", **({"backend": backend} if backend else {})}
            (eng / "BUILD.json").write_text(json.dumps(meta), encoding="utf-8")
            p = Path(d) / "strata-x.json"
            cfg = {"model_name": "qwen3.6-35b-a3b-ud-q2_k_xl", "exe": str(eng / setup.POLY_EXE),
                   "args": ["--model", "m.gguf", *args]}
            p.write_text(json.dumps(cfg), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                setup.poly_kv_whole(cfg, p)
            return json.loads(p.read_text(encoding="utf-8"))["args"]

    def test_an_amd_engine_gets_it(self):
        self.assertEqual(self.upgraded("hip")[-1], "--kv-whole")

    def test_a_cuda_engine_does_not(self):
        self.assertNotIn("--kv-whole", self.upgraded(None))

    def test_once(self):
        self.assertEqual(self.upgraded("hip", ["--kv-whole"]).count("--kv-whole"), 1)

    def test_no_build_stamp_is_left_alone(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = {"exe": str(Path(d) / "engine-poly" / setup.POLY_EXE), "args": ["--model", "m.gguf"]}
            setup.poly_kv_whole(cfg)
        self.assertNotIn("--kv-whole", cfg["args"])


if __name__ == "__main__":
    unittest.main()
