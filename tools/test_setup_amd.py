"""Tests for setup.py's AMD card detection on a mocked KFD topology (/sys/class/kfd + /sys/class/drm): the arch
names from gfx_target_version, the CPU node skipped, HIP numbering, product names, which cards are supported and
which TheRock index each family installs from.  No GPU, no ROCm, no downloads.

    python -m unittest tools.test_setup_amd
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def fake_sysfs(root: Path, nodes: list) -> None:
    """nodes: (gfx_target_version, simd_count, render_minor, product_name or None, vram_bytes)"""
    for i, (ver, simd, minor, name, vram) in enumerate(nodes):
        n = root / "class/kfd/kfd/topology/nodes" / str(i)
        n.mkdir(parents=True)
        (n / "properties").write_text(f"cpu_cores_count {0 if simd else 12}\nsimd_count {simd}\n"
                                      f"gfx_target_version {ver}\ndrm_render_minor {minor}\n")
        if simd:
            d = root / f"class/drm/renderD{minor}/device"
            d.mkdir(parents=True)
            (d / "mem_info_vram_total").write_text(str(vram))
            if name is not None:
                (d / "product_name").write_text(name + "\n")


class KfdDetection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.win = setup.WIN
        setup.WIN = False

    def tearDown(self):
        setup.WIN = self.win
        self.tmp.cleanup()

    def test_every_family(self):
        fake_sysfs(self.root, [
            (0, 0, 0, None, 0),                                   # the CPU node: skipped
            (110001, 120, 128, "", 16 << 30),                     # gfx1101 without a product name
            (120000, 64, 129, None, 16 << 30),                    # gfx1200, no product_name file
            (120001, 128, 130, "AMD Radeon AI PRO R9700", 32 << 30),
            (110002, 64, 131, None, 8 << 30),                     # gfx1102: supported, unvalidated (#938)
            (100306, 4, 132, None, 512 << 20),                    # an integrated gfx1036: listed, not supported
            (110000, 192, 133, "Radeon RX 7900 XTX", 24 << 30),
        ])
        g = setup.amd_gpus(str(self.root))
        self.assertEqual([x["arch"] for x in g], ["gfx1101", "gfx1200", "gfx1201", "gfx1102", "gfx1036", "gfx1100"])
        self.assertEqual([x["index"] for x in g], [0, 1, 2, 3, 4, 5])          # HIP numbers: GPU nodes only
        self.assertEqual(g[0]["name"], setup.AMD_NAMES["gfx1101"])
        self.assertEqual(g[1]["name"], setup.AMD_NAMES["gfx1200"])
        self.assertEqual(g[2]["name"], "AMD Radeon AI PRO R9700")
        self.assertEqual(g[3]["name"], setup.AMD_NAMES["gfx1102"])
        self.assertAlmostEqual(g[2]["vram_gb"], 32.0)
        ok = [x["arch"] for x in g if setup.amd_problem(x) is None]
        self.assertEqual(ok, ["gfx1101", "gfx1200", "gfx1201", "gfx1102", "gfx1100"])
        self.assertIn("gfx1036", setup.amd_problem(g[4]))

    def test_no_kfd(self):
        self.assertEqual(setup.amd_gpus(str(self.root)), [])

    def test_rocm_index_per_family(self):
        for arch in setup.AMD_ARCHS:
            self.assertIn(arch, setup.ROCM_INDEXES)
        self.assertTrue(setup.ROCM_INDEXES["gfx1101"].endswith("/gfx110X-dgpu/"))
        self.assertTrue(setup.ROCM_INDEXES["gfx1200"].endswith("/gfx120X-all/"))
        self.assertEqual(setup.ROCM_INDEXES["gfx1101"], setup.ROCM_INDEXES["gfx1100"])
        self.assertEqual(setup.ROCM_INDEXES["gfx1200"], setup.ROCM_INDEXES["gfx1201"])
        # #524: the RX 6700 XT (gfx1031) takes the RDNA2 wheels, as the RX 6800 / 6900 (gfx1030)
        self.assertEqual(setup.ROCM_INDEXES["gfx1031"], setup.ROCM_INDEXES["gfx1030"])
        self.assertIsNone(setup.amd_problem({"arch": "gfx1031"}))
        # the RX 6600 / 6600 XT / 6650 XT (gfx1032) too: one owner built for it and ran Qwen3.6 on these wheels
        self.assertEqual(setup.ROCM_INDEXES["gfx1032"], setup.ROCM_INDEXES["gfx1030"])
        self.assertIsNone(setup.amd_problem({"arch": "gfx1032"}))
        # PR 102: an RX 5700 XT (gfx1010, RDNA1) ran four models on Windows; its Linux wheels are their own family
        self.assertTrue(setup.ROCM_INDEXES["gfx1010"].endswith("/gfx101X-dgpu/"))
        self.assertIsNone(setup.amd_problem({"arch": "gfx1010"}))
        self.assertIn("gfx1010", setup.AMD_CARDS)


class GpuLists(unittest.TestCase):
    """--gpus with AMD cards: every chosen card must be supported; the first is the main one."""
    AMD = [{"index": 0, "name": "AMD Radeon RX 9070 XT", "vram_gb": 16.0, "arch": "gfx1201"},
           {"index": 1, "name": "AMD Radeon AI PRO R9700", "vram_gb": 32.0, "arch": "gfx1201"},
           {"index": 2, "name": "AMD Radeon (gfx1036)", "vram_gb": 0.5, "arch": "gfx1036"},
           {"index": 3, "name": "AMD Radeon RX 7900 XTX", "vram_gb": 24.0, "arch": "gfx1100"}]

    def setUp(self):
        self.say = setup.say
        setup.say = lambda *a, **k: None

    def tearDown(self):
        setup.say = self.say

    def test_list_in_order(self):
        self.assertEqual([g["index"] for g in setup.amd_parse_gpus("1,0", self.AMD)], [1, 0])
        self.assertEqual([g["index"] for g in setup.amd_parse_gpus(" 0, 3 ", self.AMD)], [0, 3])

    def test_all_is_every_supported_card_most_vram_first(self):
        self.assertEqual([g["index"] for g in setup.amd_parse_gpus("all", self.AMD)], [1, 3, 0])

    def test_refusals(self):
        for text in ("1,2", "1,7", "1", "1,1", "x,y"):
            with self.assertRaises(SystemExit, msg=text):
                setup.amd_parse_gpus(text, self.AMD)

    def test_wheels_hold_one_family(self):
        with tempfile.TemporaryDirectory() as d:        # no system ROCm there: the wheels would be needed
            old = setup.os.environ.get("ROCM_PATH")
            setup.os.environ["ROCM_PATH"] = d
            try:
                with self.assertRaises(SystemExit):
                    setup.rocm_root(["gfx1100", "gfx1201"])
            finally:
                if old is None:
                    del setup.os.environ["ROCM_PATH"]
                else:
                    setup.os.environ["ROCM_PATH"] = old

    def test_runtime_only_system_rocm_falls_back_to_the_wheels(self):
        """#446: a system ROCm 7 with hipcc and libhipblas but no HIP development files (no hip-lang CMake package, no
        hip_runtime.h) is not used for the build: a warning says what is missing and the wheels path follows (here
        it stops at the two-family check, which only the wheels path makes); with the files it is used as before."""
        said = []
        setup.say = lambda msg="": said.append(msg)          # tearDown puts the real one back
        with tempfile.TemporaryDirectory() as d:
            sysroot = Path(d)
            for rel, text in (("bin/hipcc", ""), ("lib/libhipblas.so.3", ""),
                              ("include/rocm-core/rocm_version.h",
                               "#define ROCM_VERSION_MAJOR 7\n#define ROCM_VERSION_MINOR 14\n")):
                (sysroot / rel).parent.mkdir(parents=True, exist_ok=True)
                (sysroot / rel).write_text(text)
            with mock.patch.dict(setup.os.environ, {"ROCM_PATH": d}):
                with self.assertRaises(SystemExit):
                    setup.rocm_root(["gfx1100", "gfx1201"])
                self.assertIn(f"the ROCm in {sysroot} has no HIP development files (lib/cmake/hip-lang/hip-lang-"
                              "config.cmake, include/hip/hip_runtime.h): using AMD's wheels", "\n".join(said))
                self.assertIn("two GPU families", "\n".join(said))
                for lib in ("lib64", "lib"):                 # either place CMake looks
                    with self.subTest(lib=lib):
                        cfg = sysroot / lib / "cmake/hip-lang/hip-lang-config.cmake"
                        cfg.parent.mkdir(parents=True, exist_ok=True)
                        cfg.write_text("")
                        said.clear()
                        with self.assertRaises(SystemExit):
                            setup.rocm_root(["gfx1100", "gfx1201"])
                        self.assertIn("(include/hip/hip_runtime.h)", "\n".join(said))
                        (sysroot / "include/hip").mkdir(parents=True, exist_ok=True)
                        (sysroot / "include/hip/hip_runtime.h").write_text("")
                        self.assertEqual(setup.rocm_root(["gfx1100", "gfx1201"]), (sysroot, [str(sysroot / "lib")]))
                        cfg.unlink()
                        (sysroot / "include/hip/hip_runtime.h").unlink()


class WindowsDetection(unittest.TestCase):
    """Windows: the AMD cards from the display adapters (mocked Win32_VideoController rows and display-class registry
    values), and the card setup chose matched to its HIP ordinal (#325: an integrated Radeon is HIP's device 0)."""
    ADAPTERS = [  # Win32_VideoController: name, PNPDeviceID, AdapterRAM (32-bit: at most 4 GB)
        {"name": "NVIDIA GeForce RTX 5070", "pnp": r"PCI\VEN_10DE&DEV_2F04&SUBSYS_1234&REV_A1\4&1", "ram": 4293918720},
        {"name": "AMD Radeon RX 9070 XT", "pnp": r"PCI\VEN_1002&DEV_7550&SUBSYS_0E3B1002&REV_C0\6&2", "ram": 4293918720},
        {"name": "AMD Radeon(TM) Graphics", "pnp": r"PCI\VEN_1002&DEV_164E&SUBSYS_88771043&REV_C1\4&3", "ram": 536870912},
        {"name": "AMD Radeon RX 7800 XT", "pnp": r"PCI\VEN_1002&DEV_7499&SUBSYS_0000&REV_C8\6&4", "ram": 4293918720},
    ]
    REGISTRY = [  # the display class's driver instances: 64-bit VRAM size (as int, or as bytes)
        {"DriverDesc": "NVIDIA GeForce RTX 5070", "MatchingDeviceId": r"PCI\VEN_10DE&DEV_2F04",
         "HardwareInformation.qwMemorySize": 12 << 30},
        {"DriverDesc": "AMD Radeon RX 6600", "MatchingDeviceId": r"PCI\VEN_1002&DEV_73FF",     # a removed card
         "HardwareInformation.qwMemorySize": 8 << 30},
        {"DriverDesc": "AMD Radeon RX 9070 XT", "MatchingDeviceId": r"PCI\VEN_1002&DEV_7550&REV_C0",
         "HardwareInformation.qwMemorySize": (16 << 30).to_bytes(8, "little"), "DriverVersion": "32.0.21013.1000"},
        {"DriverDesc": "AMD Radeon(TM) Graphics", "MatchingDeviceId": r"PCI\VEN_1002&DEV_164E&REV_C1",
         "HardwareInformation.qwMemorySize": 512 << 20},
        {"DriverDesc": "AMD Radeon RX 7800 XT", "MatchingDeviceId": r"PCI\VEN_1002&DEV_7499",
         "HardwareInformation.qwMemorySize": 16 << 30},
    ]
    LIST = ("device 0: AMD Radeon(TM) Graphics\n  arch gfx1036, 28.1 GiB, wave32\n"
            "  cannot run: GPU 0 (AMD Radeon(TM) Graphics, gfx1036) is not an architecture this Puddle engine was "
            "compiled for (gfx1100,gfx1101,gfx1102,gfx1200,gfx1201,gfx1030); ...\n"
            "device 1: AMD Radeon RX 9070 XT\n  arch gfx1201, 15.9 GiB, wave32\n"
            "device 2: AMD Radeon RX 7800 XT\n  arch gfx1101, 16.0 GiB, wave32\n")

    def test_adapters(self):
        g = setup.amd_gpus_windows(self.ADAPTERS, self.REGISTRY)
        self.assertEqual([x["name"] for x in g], ["AMD Radeon RX 9070 XT", "AMD Radeon(TM) Graphics",
                                                  "AMD Radeon RX 7800 XT"])     # AMD only, present only, in order
        self.assertEqual([x["index"] for x in g], [0, 1, 2])
        self.assertEqual(g[0]["arch"], "gfx1201")                               # by PCI device id
        self.assertEqual(g[2]["arch"], "gfx1101")                               # an unlisted id: by its name
        self.assertTrue(g[1]["arch"].startswith("unknown"))                     # the iGPU: listed, not supported
        self.assertAlmostEqual(g[0]["vram_gb"], 16.0)                           # the registry's 64-bit size
        self.assertEqual(g[0]["driver"], "32.0.21013.1000")
        self.assertEqual([setup.amd_problem(x) is None for x in g], [True, False, True])

    def test_rx_6800m_is_gfx1031(self):
        """#881: PCI 73DF (RX 6700 XT / 6750 XT / 6800M) is gfx1031, not an unknown id."""
        self.assertEqual(setup.win_amd_arch(0x73DF, "AMD Radeon RX 6800M"), "gfx1031")
        self.assertIsNone(setup.amd_problem({"arch": "gfx1031"}))

    def test_rx_6650_xt_is_gfx1032(self):
        """Issue 121: PCI 73EF (RX 6650 XT) and 73FF (RX 6600 / 6600 XT) are gfx1032, and so is the name of one
        whose id is not listed."""
        self.assertEqual(setup.win_amd_arch(0x73EF, "AMD Radeon RX 6650 XT"), "gfx1032")
        self.assertEqual(setup.win_amd_arch(0x73FF, "AMD Radeon RX 6600"), "gfx1032")
        self.assertEqual(setup.win_amd_arch(0x1234, "AMD Radeon RX 6600 XT"), "gfx1032")
        self.assertIsNone(setup.amd_problem({"arch": "gfx1032"}))

    def test_rx_5700_xt_is_gfx1010(self):
        """PCI 731F (RX 5600 XT / 5700 / 5700 XT, Navi 10) is gfx1010, and so is the name of one whose id is not
        listed; the RX 5500 (gfx1012, Navi 14) is not taken for it."""
        self.assertEqual(setup.win_amd_arch(0x731F, "AMD Radeon RX 5700 XT"), "gfx1010")
        self.assertEqual(setup.win_amd_arch(0x1234, "AMD Radeon RX 5600 XT"), "gfx1010")
        self.assertEqual(setup.win_amd_arch(0x1234, "AMD Radeon RX 5700"), "gfx1010")
        self.assertEqual(setup.win_amd_arch(0x7340, "AMD Radeon RX 5500 XT"), "")
        self.assertIsNone(setup.amd_problem({"arch": "gfx1010"}))


    def test_registry_alone(self):
        """No WMI answer: the registry's own list (which can hold a removed card)."""
        g = setup.amd_gpus_windows([], self.REGISTRY)
        self.assertEqual([x["arch"] for x in g if not x["arch"].startswith("unknown")],
                         ["gfx1032", "gfx1201", "gfx1101"])   # the removed RX 6600 too

    def test_arch_names(self):
        for name, arch in (("AMD Radeon RX 9070 GRE", "gfx1201"), ("AMD Radeon AI PRO R9700", "gfx1201"),
                           ("AMD Radeon RX 9060 XT", "gfx1200"), ("AMD Radeon RX 7900 GRE", "gfx1100"),
                           ("AMD Radeon PRO W7800", "gfx1100"), ("AMD Radeon RX 7700 XT", "gfx1101"),
                           ("AMD Radeon RX 7600", "gfx1102"), ("AMD Radeon RX 6950 XT", "gfx1030"),
                           ("AMD Radeon RX 6800M", ""), ("AMD Radeon 780M Graphics", "gfx1103"), ("AMD Radeon RX 7700S", "")):
            self.assertEqual(setup.win_amd_arch(None, name), arch, name)
        self.assertEqual(setup.win_amd_arch(0x744C, "whatever"), "gfx1100")

    def test_windows_dispatch(self):
        """amd_gpus() on Windows: the display adapters, in their order."""
        with mock.patch.object(setup, "WIN", True), \
                mock.patch.object(setup, "amd_gpus_windows", lambda adapters=None, registry=None: ["display"]):
            self.assertEqual(setup.amd_gpus(), ["display"])


class HipRuntimeBesideExe(unittest.TestCase):
    """#468 #461: the bundled HIP runtime (and amd_comgr) goes next to strata.exe, so an AMD driver's System32 copy is
    not found first; rocBLAS and the rest stay in rocm/bin."""

    def test_copies_only_the_runtime(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d)
            rb = eng / "rocm" / "bin"
            rb.mkdir(parents=True)
            (eng / "BUILD.json").write_text(json.dumps({"backend": "hip", "lib_dirs": ["rocm/bin"]}))
            for n, data in (("amdhip64_7.dll", b"hip-3686"), ("amd_comgr.dll", b"comgr"), ("rocblas.dll", b"blas")):
                (rb / n).write_bytes(data)
            setup.hip_runtime_beside_exe(eng)
            self.assertEqual((eng / "amdhip64_7.dll").read_bytes(), b"hip-3686")
            self.assertEqual((eng / "amd_comgr.dll").read_bytes(), b"comgr")
            self.assertFalse((eng / "rocblas.dll").exists())          # finds its kernels relative to rocm/bin
            (rb / "amdhip64_7.dll").write_bytes(b"hip-3690-newer")    # a newer zip: replaced on the next start
            setup.hip_runtime_beside_exe(eng)
            self.assertEqual((eng / "amdhip64_7.dll").read_bytes(), b"hip-3690-newer")

    def test_no_rocm_bin_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as d:
            setup.hip_runtime_beside_exe(Path(d))                   # no BUILD.json (a CUDA or Linux engine)
            self.assertEqual(list(Path(d).iterdir()), [])


class DeviceAccess(unittest.TestCase):
    """Linux AMD: /dev/kfd and the render nodes must be openable by the user; setup warns, never refuses."""

    def dev(self, d, kfd=True, nodes=("renderD128",)):
        root = Path(d)
        (root / "dri").mkdir()
        if kfd:
            (root / "kfd").write_text("")
        for n in nodes:
            (root / "dri" / n).write_text("")
        return str(root)

    def test_accessible_is_silent(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(setup.amd_device_access_problem(self.dev(d), access=lambda p, m: True))

    def test_no_kfd_is_not_this_problem(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(setup.amd_device_access_problem(self.dev(d, kfd=False), access=lambda p, m: False))

    def test_kfd_denied_names_the_fix(self):
        with tempfile.TemporaryDirectory() as d:
            msg = setup.amd_device_access_problem(self.dev(d), access=lambda p, m: not p.endswith("kfd"))
            self.assertIn("kfd", msg)
            self.assertNotIn("renderD128", msg)
            self.assertIn("sudo usermod -aG render,video $USER", msg)
            self.assertIn("log out and in", msg)

    def test_render_node_denied(self):
        with tempfile.TemporaryDirectory() as d:
            msg = setup.amd_device_access_problem(self.dev(d), access=lambda p, m: "renderD" not in p)
            self.assertIn("renderD128", msg)
            self.assertNotIn("/kfd", msg)


class PolyHipWindows(unittest.TestCase):
    """strata-poly on an AMD card under Windows: setup compiles it itself (AMD's TheRock wheels in .rocm-win), the
    wheels' version coming from STRATA_ROCM_WIN_VERSION (or STRATA_ROCM_VERSION, PolyStrata's build script's name).
    No ROCm, no GPU, no downloads."""

    def test_an_installed_rocm_win_is_kept(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "rocm"
            (root / "lib" / "llvm" / "bin").mkdir(parents=True)
            (root / "lib" / "llvm" / "bin" / "clang++.exe").write_text("x")
            (root / ".kpack").mkdir()
            (root / ".kpack" / "blas_lib_gfx1201.kpack").write_text("x")
            (root / "bin").mkdir()
            venv = Path(t) / ".rocm-win" / "Scripts"
            venv.mkdir(parents=True)
            (venv / "python.exe").write_text("x")
            (venv / "rocm-sdk.exe").write_text("x")
            asked = []
            with mock.patch.object(setup, "ROOT", Path(t)), \
                    mock.patch.object(setup, "out", lambda cmd: asked.append(cmd) or str(root) + "\n"), \
                    mock.patch.object(setup, "run", mock.Mock(side_effect=AssertionError("installed"))):
                got, dirs = setup.rocm_win_root(["gfx1201"])
            self.assertEqual((got, dirs), (root, [str(root / "bin")]))
            # through the venv's python.exe: rocm-sdk.exe names the folder's old place once it is moved (issue 78)
            self.assertEqual(asked, [[str(venv / "python.exe"), "-m", "rocm_sdk", "path", "--root"]])

    @staticmethod
    def rocm_win_tree(t):
        """A .rocm-win holding every arch's kernels and a working root: the layout setup keeps as it is."""
        root = Path(t) / "rocm"
        (root / "lib" / "llvm" / "bin").mkdir(parents=True)
        (root / "lib" / "llvm" / "bin" / "clang++.exe").write_text("x")
        (root / ".kpack").mkdir()
        (root / ".kpack" / "blas_lib_gfx1201.kpack").write_text("x")
        venv = Path(t) / ".rocm-win"
        (venv / "Scripts").mkdir(parents=True)
        (venv / "Scripts" / "python.exe").write_text("x")
        (venv / "Scripts" / "rocm-sdk.exe").write_text("x")
        return root, venv

    def test_another_wheels_version_is_installed_in_its_place(self):
        with tempfile.TemporaryDirectory() as t:
            root, venv = self.rocm_win_tree(t)
            (venv / "Lib" / "site-packages" / "rocm-9.9.9.dist-info").mkdir(parents=True)
            ran = []
            with mock.patch.object(setup, "ROOT", Path(t)), \
                    mock.patch.object(setup, "out", lambda cmd: str(root) + "\n"), \
                    mock.patch.object(setup, "run", lambda cmd: ran.append(cmd)):
                got, dirs = setup.rocm_win_root(["gfx1201"])
            self.assertEqual((got, dirs), (root, [str(root / "bin")]))
            pip = [c for c in ran if "pip" in c]
            self.assertEqual(len(pip), 1, ran)                       # exactly one install, and it is the wanted version
            self.assertIn(f"rocm[libraries,devel,device-gfx1201]=={setup.ROCM_WIN_VERSION}", pip[0])
            self.assertIn([str(venv / "Scripts" / "python.exe"), "-m", "rocm_sdk", "init"], ran)

    def test_a_venv_that_does_not_say_its_version_is_kept(self):
        with tempfile.TemporaryDirectory() as t:
            root, venv = self.rocm_win_tree(t)                       # no rocm-*.dist-info: not pip's doing
            warned = []
            with mock.patch.object(setup, "ROOT", Path(t)), \
                    mock.patch.object(setup, "out", lambda cmd: str(root) + "\n"), \
                    mock.patch.object(setup, "ROCM_WIN_VERSION_OVERRIDE", "9.9.9"), \
                    mock.patch.object(setup, "ROCM_WIN_VERSION", "9.9.9"), \
                    mock.patch.object(setup, "warn", lambda msg: warned.append(msg)), \
                    mock.patch.object(setup, "run", mock.Mock(side_effect=AssertionError("installed"))):
                got, dirs = setup.rocm_win_root(["gfx1201"])
            self.assertEqual((got, dirs), (root, [str(root / "bin")]))
            self.assertTrue(any("does not say which ROCm" in m for m in warned), warned)
            self.assertTrue(any("9.9.9" in m for m in warned), warned)

    def test_the_env_var_picks_the_wheels_version(self):
        base = {k: v for k, v in os.environ.items()
                if k not in ("STRATA_ROCM_WIN_VERSION", "STRATA_ROCM_VERSION")}

        def constant(**env):
            r = subprocess.run([sys.executable, "-c", "import setup; print(setup.ROCM_WIN_VERSION)"],
                               cwd=Path(setup.__file__).resolve().parent, capture_output=True, text=True,
                               env={**base, **env})
            self.assertEqual(r.returncode, 0, r.stderr)
            return r.stdout.strip()

        self.assertEqual(constant(STRATA_ROCM_WIN_VERSION="9.9.9"), "9.9.9")
        self.assertEqual(constant(STRATA_ROCM_VERSION="8.8.8"), "8.8.8")   # PolyStrata's build script's name
        self.assertEqual(constant(STRATA_ROCM_WIN_VERSION="9.9.9", STRATA_ROCM_VERSION="8.8.8"), "9.9.9")
        self.assertEqual(constant(), setup.ROCM_WIN_VERSION_DEFAULT)

    def test_an_engine_of_this_source_is_kept(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            eng = root / setup.ENGINE_POLY_DIR
            eng.mkdir()
            (eng / setup.POLY_EXE).write_text("x")
            (eng / "BUILD.json").write_text(json.dumps({"source": "local-hip", "backend": "hip",
                                                        "archs": ["gfx1201"], "src": "abc"}), encoding="utf-8")
            compiled = mock.Mock(side_effect=AssertionError("compiled"))
            with mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "source_hash", lambda parts: "abc"), \
                    mock.patch.object(setup, "cmake_build", compiled):
                got = setup.build_engine_poly_hip_win({"arch": "gfx1201", "archs": ["gfx1201"]}, root / "llama.cpp")
            self.assertEqual(got, eng)
            compiled.assert_not_called()

    def test_the_windows_engine_is_dispatched_from_the_amd_build(self):
        seen = {}
        with mock.patch.object(setup, "WIN", True), \
                mock.patch.object(setup, "build_engine_poly_hip_win",
                                  lambda gpu, llama, portable=False: seen.update(gpu=gpu, portable=portable)):
            setup.build_engine_poly_hip({"arch": "gfx1201"}, Path("llama.cpp"), True)
        self.assertEqual(seen, {"gpu": {"arch": "gfx1201"}, "portable": True})

    def test_the_check_line_names_amd_on_windows(self):
        d = setup.FAMILIES["occamy"]["models"]["APEX-Quality"]
        with mock.patch.object(setup, "WIN", True):
            line = setup.poly_verdict(d, 63, 15.9, True)
        self.assertIn("measured on one AMD card on Windows", line)
        self.assertIn("fits by setup's estimate", line)

    def build(self, root, bundled, portable=False):
        """The Windows HIP build's compile with cmake mocked: returns (the cmake args, the BUILD.json)."""
        built = []

        def cmake(src_dir, bdir, target, defs, vcvars, bat):
            built.append((target, defs))
            bdir.mkdir(parents=True, exist_ok=True)
            (bdir / setup.POLY_EXE).write_text("x")

        with mock.patch.object(setup, "ROOT", root), \
                mock.patch.object(setup, "source_hash", lambda parts: "new"), \
                mock.patch.object(setup, "find_vcvars", lambda: "vcvars"), \
                mock.patch.object(setup, "rocm_win_root", lambda archs: (root / "rocm", [])), \
                mock.patch.object(setup, "cmake_build", cmake), \
                mock.patch.object(setup, "hipblaslt_version", lambda dirs: None), \
                mock.patch.object(setup, "source_version", lambda: "0.1.45"), \
                mock.patch.object(setup, "hip_runtime_beside_exe", lambda eng: None):
            got = setup.build_engine_poly_hip_win({"arch": "gfx1201"}, root / "llama.cpp", portable)
        self.assertEqual(got, root / setup.ENGINE_POLY_DIR)
        return built[0][1], json.loads((root / setup.ENGINE_POLY_DIR / "BUILD.json").read_text(encoding="utf-8"))

    def test_the_build_is_native_and_loads_rocm_win(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            rocm = root / "rocm"
            defs, meta = self.build(root, rocm)
            self.assertIn("-DSTRATA_PORTABLE=OFF", defs)           # native for the PC setup builds on (as on Linux)
            self.assertNotIn("-DSTRATA_NATIVE_EXPERTS=ON", defs)   # ON by default
            self.assertEqual(meta["lib_dirs"], [str(rocm / "bin")])   # .rocm-win's ROCm: not on the disk twice
            self.assertFalse((root / setup.ENGINE_POLY_DIR / "rocm").exists())

    def test_max_compatibility_uses_the_portable_build_and_changes_rebuild(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            defs, meta = self.build(root, root / "engine" / "rocm" / "bin", portable=True)
            self.assertIn("-DSTRATA_PORTABLE=ON", defs)
            self.assertTrue(meta["portable"])
            defs, _ = self.build(root, root / "engine" / "rocm" / "bin")
            self.assertIn("-DSTRATA_PORTABLE=OFF", defs)


class TdrPointer(unittest.TestCase):
    def test_setup_points_windows_gfx12_to_the_entry(self):
        src = Path(setup.__file__).read_text(encoding="utf-8")
        self.assertIn('WIN and str(gpu.get("arch") or "").startswith("gfx12")', src)
        doc = (Path(setup.__file__).resolve().parent / "docs/TROUBLESHOOTING.md").read_text(encoding="utf-8")
        self.assertIn("Windows AMD: the driver resets", doc)


if __name__ == "__main__":
    unittest.main()
