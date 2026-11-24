"""Tests for setup.py's choices that depend on the PC (no GPU, no network, nothing installed): the image encoder of a
ready-made engine that has no code for the card (#331), --gguf-dir's shard count (#305), the experimental
Pascal/Volta build (#295), the CPU image encoder with the AMD backend (#304).

    python -m unittest tools.test_setup_choices
"""
from __future__ import annotations

import contextlib
import io
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


def quiet(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*args, **kw), out.getvalue()


class GgufDirShards(unittest.TestCase):
    """#305: --gguf-dir reads the shard count from the -of-N part of the name instead of assuming two.  A family of
    three shards per size, named as Unsloth's Laguna-S files are (a stand-in until that family is offered)."""
    FAM = {"file": "Laguna-S-2.1-{q}-{i:05d}-of-00003.gguf", "shards": 3,
           "models": {"UD-Q3_K_XL": {}, "UD-Q4_K_S": {}}}
    PUBLISHED = ["Laguna-S-2.1-UD-Q3_K_XL-%05d-of-00003.gguf" % i for i in range(1, 4)]

    def shards(self, names, model="UD-Q3_K_XL"):
        with tempfile.TemporaryDirectory() as d:
            for n in names:
                (Path(d) / n).write_bytes(b"")
            return [p.name for p in setup.gguf_dir_shards(Path(d), self.FAM, model)]

    def test_the_published_shards(self):
        self.assertEqual(self.shards(self.PUBLISHED), self.PUBLISHED)
        self.assertEqual(self.shards(self.PUBLISHED[:1]), self.PUBLISHED)   # the others named from the first
        self.assertEqual(self.shards(self.PUBLISHED + ["mmproj-F16.gguf"]), self.PUBLISHED)

    def test_another_split_of_a_setup_size(self):
        names = ["my-UD-Q3_K_XL-%05d-of-00004.gguf" % i for i in range(1, 5)]
        other = ["my-UD-Q4_K_S-%05d-of-00002.gguf" % i for i in range(1, 3)]
        self.assertEqual(self.shards(names + other), names)           # the one with the size in its name
        self.assertEqual(self.shards(other, model="UD-Q4_K_S"), other)

    def test_nothing_recognisable_keeps_the_published_names(self):
        self.assertEqual(self.shards([]), self.PUBLISHED)
        self.assertEqual(self.shards(["a-00001-of-00002.gguf", "b-00001-of-00002.gguf"]), self.PUBLISHED)   # ambiguous


class ReadOnlyGgufDir(unittest.TestCase):
    """--gguf-dir on a read-only folder (a share): whole shards without a finish mark still install, with a warning."""

    def test_mark_itself_never_stops_on_a_read_only_folder(self):
        """#570: every finish mark (the SHA-256 result of a verified shard too), not only the one for a whole shard."""
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(Path, "write_text",
                                                                     side_effect=PermissionError(13, "read-only")):
            setup.mark(Path(tmp) / "x.gguf", "sha256 abc")                   # no exception
            self.assertFalse(setup.done(Path(tmp) / "x.gguf"))


class VsRange(unittest.TestCase):
    """#985: Visual Studio 2026 (version 18) is looked for only with CUDA 13.3 or newer."""

    def asked(self, cuda_v):
        seen = []
        with mock.patch.object(setup.Path, "exists", lambda self: True), \
                mock.patch.object(setup, "out", lambda cmd, *a, **k: seen.append(cmd) or "C:/VS"):
            setup.find_vcvars(cuda_v)
        return seen[0][seen[0].index("-version") + 1] if "-version" in seen[0] else None

    def test_the_range_follows_the_toolkit(self):
        self.assertIsNone(self.asked(None))      # #881: without CUDA vswhere gets no version range
        self.assertEqual(self.asked((13, 0)), "[16.0,18.0)")
        self.assertEqual(self.asked((13, 2)), "[16.0,18.0)")
        self.assertEqual(self.asked((13, 3)), "[16.0,19.0)")
        self.assertEqual(self.asked((14, 0)), "[16.0,19.0)")


    def test_vcvars_by_hand(self):
        """#881: STRATA_VCVARS names a vcvars64.bat directly, for an install vswhere cannot use."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            bat = Path(d) / "vcvars64.bat"
            bat.write_text("rem")
            with mock.patch.dict(os.environ, {"STRATA_VCVARS": str(bat)}):
                self.assertEqual(setup.find_vcvars(), bat)
            with mock.patch.dict(os.environ, {"STRATA_VCVARS": str(Path(d) / "missing.bat")}):
                self.assertIsNone(setup.find_vcvars())

class ExperimentalSm60(unittest.TestCase):
    """#295: Pascal (6.x) and Volta (7.0) only with STRATA_EXPERIMENTAL_SM60=1, built with -DSTRATA_EXPERIMENTAL_SM60=ON
    and a CUDA 12.x toolkit; nothing changes without the variable."""

    def card(self, arch):
        return {"arch": arch, "vram_gb": 11.0, "index": 0, "name": "card"}

    def test_the_gate(self):
        with mock.patch.dict(os.environ, {"STRATA_EXPERIMENTAL_SM60": ""}):
            for arch in ("60", "61", "70"):
                p = setup.gpu_problem(self.card(arch))
                self.assertIn("not supported", p)
                self.assertIn("choose it with --gpu 0", p)              # the CUDA 12 engine (docs/OLDER_GPUS.md)
            self.assertNotIn("--gpu", setup.gpu_problem(self.card("52")))
            self.assertIsNone(setup.gpu_problem(self.card("75")))
        with mock.patch.dict(os.environ, {"STRATA_EXPERIMENTAL_SM60": "1"}):
            for arch in ("60", "61", "70", "75", "120"):
                self.assertIsNone(setup.gpu_problem(self.card(arch)), arch)
            for arch in ("52", "72"):
                self.assertIsNotNone(setup.gpu_problem(self.card(arch)), arch)


    def test_find_nvcc_below(self):
        with tempfile.TemporaryDirectory() as d:
            new, old = Path(d) / "13" / "nvcc", Path(d) / "12" / "bin" / "nvcc"
            for p in (new, old):
                p.parent.mkdir(parents=True)
                p.write_bytes(b"")
            versions = {str(new): "Cuda compilation tools, release 13.0, V13.0.88",
                        str(old): "Cuda compilation tools, release 12.9, V12.9.86"}
            with mock.patch.object(setup, "WIN", False), mock.patch.object(setup.shutil, "which", lambda n: str(new)), \
                    mock.patch.dict(os.environ, {"CUDA_PATH": str(Path(d) / "12")}), \
                    mock.patch.object(setup, "out", lambda cmd: versions.get(cmd[0], "")):  # #414
                self.assertEqual(setup.find_nvcc(), (str(new), (13, 0)))
                self.assertEqual(setup.find_nvcc(below=(13, 0)), (str(old), (12, 9)))
                # #601: STRATA_NVCC is the only one considered; CUDA_HOME is a candidate like CUDA_PATH
                with mock.patch.dict(os.environ, {"STRATA_NVCC": str(old)}):
                    self.assertEqual(setup.find_nvcc(), (str(old), (12, 9)))
                with mock.patch.dict(os.environ, {"STRATA_NVCC": str(new)}):
                    got, text = quiet(setup.find_nvcc, below=(13, 0))
                    self.assertEqual(got, (None, None))
                    self.assertIn("needs one older than 13.0", text)
                with mock.patch.dict(os.environ, {"STRATA_NVCC": str(Path(d) / "missing")}):
                    got, text = quiet(setup.find_nvcc)
                    self.assertEqual(got, (str(new), (13, 0)))
                    self.assertIn("no such file", text)
            with mock.patch.object(setup, "WIN", False), mock.patch.object(setup.shutil, "which", lambda n: None), \
                    mock.patch.dict(os.environ, {"CUDA_HOME": str(Path(d) / "12")}), \
                    mock.patch.object(setup, "out", lambda cmd: versions.get(cmd[0], "")):
                os.environ.pop("CUDA_PATH", None)
                got = setup.find_nvcc(below=(13, 0))
                self.assertEqual(got, (str(old), (12, 9)))

    def tools(self, archs, nvcc):
        seen = []

        def find(below=None):
            seen.append(below)
            return nvcc(below)

        with mock.patch.object(setup, "find_nvcc", find), mock.patch.object(setup, "find_vcvars", lambda cuda_v=None: "vcvars"), \
                mock.patch.object(setup.shutil, "which", lambda n: "/usr/bin/" + n):
            got, _ = quiet(setup.install_build_tools, {"arch": str(archs[0]), "archs": archs}, True)
        return got, seen

    def test_pascal_takes_cuda_12(self):
        got, seen = self.tools([61], lambda below: ("nvcc12", (12, 9)) if below == (13, 0) else ("nvcc13", (13, 0)))
        self.assertEqual(got[0], "nvcc12")
        self.assertEqual(seen, [(13, 0)])
        got, seen = self.tools([86, 120], lambda below: ("nvcc13", (13, 0)))
        self.assertEqual((got[0], seen), ("nvcc13", [None]))           # the default: unchanged

    def test_cuda_132_on_sm120(self):
        """#892 / #968: nvcc 13.2 for an RTX 50 card: an older 13.x is taken when there is one, else a warning; never an
        error, and no other card or toolkit is touched."""
        def nvcc(below):
            return ("nvcc130", (13, 0)) if below == (13, 2) else ("nvcc132", (13, 2))

        got, seen = self.tools([120], nvcc)
        self.assertEqual(got[0], "nvcc130")
        self.assertEqual(seen, [None, (13, 2)])
        with mock.patch.dict(os.environ, {"STRATA_NVCC": "x"}):          # the user's own pick is kept
            got, seen = self.tools([120], nvcc)
            self.assertEqual((got[0], seen), ("nvcc132", [None]))
        only132 = lambda below: (None, None) if below == (13, 2) else ("nvcc132", (13, 2))
        with mock.patch.object(setup, "warn") as warn:                   # nothing older: the one found, with a warning
            got, _ = self.tools([120], only132)
        self.assertEqual(got[0], "nvcc132")
        self.assertIn("#892", warn.call_args[0][0])
        for archs in ([86], [89, 86]):                                   # not an sm_120 card: as before
            got, seen = self.tools(archs, nvcc)
            self.assertEqual((got[0], seen), ("nvcc132", [None]))

    def test_pascal_without_cuda_12_stops(self):
        for archs in ([61], [70, 120]):
            with self.subTest(archs=archs), self.assertRaises(SystemExit):
                self.tools(archs, lambda below: (None, None) if below else ("nvcc13", (13, 0)))


class VramReserve(unittest.TestCase):
    """#493: --vram-reserve-mib N - VRAM the engine leaves free for other programs - goes into the config only when
    given (the default config unchanged), at setup and on a start.  Occamy through tools/poly_setup_harness.py."""
    from tools.poly_setup_harness import PolyMain
    found = [{"index": 0, "name": "NVIDIA GeForce RTX 5090", "vram_gb": 31.8, "arch": "120", "driver": "580.97"}]

    def setUp(self):
        self.PolyMain.setUp(self)

    def tearDown(self):
        self.PolyMain.tearDown(self)

    def install(self, *extra):
        code, out, cfg = self.PolyMain.main(self, list(extra), family="occamy", model="APEX-Quality", ram=64.0,
                                            cards=self.found)
        return code, out, cfg, None

    def test_given_at_setup(self):
        code, out, cfg, _ = self.install("--vram-reserve-mib", "2048")
        self.assertEqual(code, 0, out)
        a = cfg["args"]
        self.assertEqual(a.count("--vram-reserve-mib"), 1)
        self.assertEqual(a[a.index("--vram-reserve-mib") + 1], "2048")
        self.assertIn("VRAM kept free for other programs: 2048 MiB", out)
        code, out, cfg, _ = self.install()
        self.assertNotIn("--vram-reserve-mib", cfg["args"])          # not given: the engine's default, as before

    def test_with_images_one_value(self):
        code, out, cfg, _ = self.install("--vision", "cpu", "--vram-reserve-mib", "1500")
        self.assertEqual(code, 0, out)
        a = cfg["args"]
        self.assertEqual(a.count("--vram-reserve-mib"), 1)
        self.assertEqual(a[a.index("--vram-reserve-mib") + 1], "1500")

    def test_a_negative_value_is_refused(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            code, out, cfg, _ = self.install("--vram-reserve-mib", "-1")
        self.assertEqual(code, 2)
        self.assertIn("--vram-reserve-mib takes a number of MiB", err.getvalue())

    def test_kept_from_an_earlier_install(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-occamy10-apex-quality.json"
            p.write_text(json.dumps({"args": ["--max-context", "8192", "--vram-reserve-mib", "2048"]}))
            self.assertEqual(setup.choices_from_config(p)["vram_reserve_mib"], 2048)
            p.write_text(json.dumps({"args": ["--vision", "--vram-reserve-mib", "700"], "vision": {"gpu": True}}))
            self.assertIsNone(setup.choices_from_config(p)["vram_reserve_mib"])   # the images' own reserve
            p.write_text(json.dumps({"args": ["--max-context", "8192"]}))
            self.assertIsNone(setup.choices_from_config(p)["vram_reserve_mib"])

    def test_given_on_a_start(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / setup.POLY_EXE
            exe.write_bytes(b"")
            p = Path(d) / "strata-q2_0.json"
            p.write_text(json.dumps({"exe": str(exe), "args": ["--kv", "int8"], "gpu": 0, "gpus_asked": True}))
            call = mock.Mock(return_value=0)
            with mock.patch.object(setup, "gpus", lambda: self.found), \
                    mock.patch.object(setup, "update_poly_engine", lambda *a, **k: None), \
                    mock.patch.object(setup, "poly_tail_skip", lambda *a, **k: None), \
                    mock.patch.object(setup, "poly_kv_whole", lambda *a, **k: None), \
                    mock.patch.object(setup.subprocess, "call", call):
                for n in ("2048", "1024"):
                    _, out = quiet(setup.start, p, None, None, False, True, None, {"vram_reserve_mib": int(n)})
                    self.assertEqual(json.loads(p.read_text())["args"], ["--kv", "int8", "--vram-reserve-mib", n])
        self.assertIn("1024 MiB of VRAM kept free", out)
        self.assertTrue(call.called)


class HostCompilers(unittest.TestCase):
    """#1645: CC / CXX / CUDAHOSTCXX name the build's compilers - handed to CMake as -D, so a build folder that is
    already configured picks them up too (CXX alone reaches CMake only on the first configure: the reporter's
    `CXX=g++-14` run died at the last step).  CUDAHOSTCXX is nvcc's host compiler and defaults to CXX."""

    def test_env_names_the_compilers(self):
        with mock.patch.dict(os.environ, {"CC": "gcc-14", "CXX": "g++-14", "CUDAHOSTCXX": "g++-14-alt"}):
            self.assertEqual(setup.host_compiler_defs(True), ["-DCMAKE_C_COMPILER=gcc-14",
                                                              "-DCMAKE_CXX_COMPILER=g++-14",
                                                              "-DCMAKE_CUDA_HOST_COMPILER=g++-14-alt"])

    def test_cudahostcxx_defaults_to_cxx_and_needs_cuda(self):
        with mock.patch.dict(os.environ, {"CC": "", "CXX": "g++-14", "CUDAHOSTCXX": ""}):
            self.assertEqual(setup.host_compiler_defs(True), ["-DCMAKE_CXX_COMPILER=g++-14",
                                                              "-DCMAKE_CUDA_HOST_COMPILER=g++-14"])
            self.assertEqual(setup.host_compiler_defs(False), ["-DCMAKE_CXX_COMPILER=g++-14"])   # a C++-only project

    def test_unset_adds_nothing(self):
        with mock.patch.dict(os.environ, {"CC": "", "CXX": "", "CUDAHOSTCXX": ""}):
            self.assertEqual(setup.host_compiler_defs(True), [])


if __name__ == "__main__":
    unittest.main()
