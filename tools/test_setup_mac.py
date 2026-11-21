"""Tests for setup on a Mac (metal/setup_mac.py and setup.py's hand-over): the Mac check, the model menu, the
strata-poly install of every family and size, the image encoder and the config. Runs on any PC: sysctl / sw_vers /
xcode-select are mocked. No GPU, no network.

    python -m unittest tools.test_setup_mac
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "metal"))
import setup  # noqa: E402
import setup_mac as M  # noqa: E402


def fake_run(values: dict, clt_rc=0):
    def run(cmd, **kw):
        if cmd[0] == "sysctl":
            return subprocess.CompletedProcess(cmd, 0, values.get(cmd[2], ""), "")
        if cmd[0] == "sw_vers":
            return subprocess.CompletedProcess(cmd, 0, values.get("macos", ""), "")
        if cmd[0] == "xcode-select":
            return subprocess.CompletedProcess(cmd, clt_rc, "/Library/Developer/CommandLineTools" if clt_rc == 0 else "",
                                               "")
        raise AssertionError(cmd)
    return run


M5 = {"machdep.cpu.brand_string": "Apple M5 Max", "hw.memsize": str(128 * 2 ** 30), "macos": "27.2",
      "hw.perflevel0.physicalcpu": "12"}


class MacCheck(unittest.TestCase):
    def test_reads_chip_memory_macos_and_tools(self):
        with mock.patch.object(M.platform, "machine", return_value="arm64"):
            info = M.mac_info(fake_run(M5))
        self.assertEqual(info["chip"], "Apple M5 Max")
        self.assertAlmostEqual(info["ram_gb"], 128.0)
        self.assertEqual(info["macos"], "27.2")
        self.assertTrue(info["arm64"])
        self.assertTrue(info["clt"])
        self.assertEqual(M.mac_problems(info), [])

    def test_intel_old_macos_and_missing_tools_are_named(self):
        with mock.patch.object(M.platform, "machine", return_value="x86_64"):
            info = M.mac_info(fake_run({**M5, "macos": "14.6", "machdep.cpu.brand_string": "Intel Core i9"}, clt_rc=2))
        p = " | ".join(M.mac_problems(info))
        self.assertIn("Apple Silicon", p)
        self.assertIn("macOS 15", p)
        self.assertIn("xcode-select --install", p)


class Config(unittest.TestCase):
    def test_config_is_a_model_config_with_the_metal_backend(self):
        cfg = M.mac_config(Path("/r/build-metal/strata-poly"), ["--model", "x"], Path("/p"), "gemma-4-26b-a4b-ud-q3_k_m",
                           "gemma4-UD-Q3_K_M", 8080)
        self.assertEqual(cfg["backend"], "metal")
        self.assertEqual(cfg["tokenizer"], str(Path("/p") / "tokenizer"))
        self.assertTrue(cfg["exe"] and isinstance(cfg["args"], list))


class Vision(unittest.TestCase):
    def test_the_encoder_builds_with_metal(self):
        src = (ROOT / "metal" / "setup_mac.py").read_text(encoding="utf-8")
        self.assertIn('"-DSTRATA_VISION_CUDA=OFF"', src)
        self.assertIn('"-DGGML_METAL=ON"', src)
        self.assertIn('cfg["vision"] = {', src)


class HandOver(unittest.TestCase):
    def test_darwin_hands_the_run_to_setup_mac(self):
        with mock.patch.object(setup.sys, "platform", "darwin"), \
                mock.patch.object(setup.subprocess, "call", return_value=0) as call, \
                mock.patch.object(setup.sys, "argv", ["setup.py", "--model", "UD-Q3_K_M", "--yes"]):
            self.assertEqual(setup.main(), 0)
        cmd = call.call_args[0][0]
        self.assertTrue(cmd[1].endswith(str(Path("metal") / "setup_mac.py")))
        self.assertEqual(cmd[2:], ["--model", "UD-Q3_K_M", "--yes"])

    def test_other_platforms_never_reach_it(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn('if sys.platform == "darwin":', src)
        with mock.patch.object(setup, "mac_setup", side_effect=AssertionError("called")), \
                mock.patch.object(setup.sys, "platform", "linux"), \
                mock.patch.object(setup.sys, "argv", ["setup.py", "--inspect", "x"]), \
                mock.patch.object(setup.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            with self.assertRaises(SystemExit):
                setup.main()


class Source(unittest.TestCase):
    """setup.py hands a Mac over before its own parser runs, so the Mac parser has to know --source:
    source_hint() recommends it on every platform (#2363)."""

    def test_the_source_flag_reaches_strata_source(self):
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("STRATA_SOURCE", None)
        warnings = []
        # setup.data_folder() is mocked: the real one moves the model files of this checkout and of the data folder in
        # the user's settings.json into --data-dir, and saves that dir as the new data folder; with a temp dir that
        # deletes the models on exit and leaves settings.json pointing at a deleted directory
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(setup, "data_folder", lambda requested: (Path(tmp), [])), \
                mock.patch("subprocess.run", fake_run(M5)), \
                mock.patch.object(setup, "warn", lambda *a, **k: warnings.append(" ".join(str(x) for x in a))), \
                mock.patch.object(setup, "say", lambda *a, **k: None), \
                mock.patch.object(setup, "ok", lambda *a, **k: None), \
                mock.patch.object(setup, "step", lambda *a, **k: None):
            M.main(["--source", "modelscope", "--yes", "--check", "--data-dir", tmp])
        self.assertEqual(os.environ.get("STRATA_SOURCE"), "modelscope")
        self.assertFalse([w for w in warnings if "--source" in w],
                         f"--source was not used: {warnings}")


class SmallerModelCatalog(unittest.TestCase):
    def test_uses_shared_family_sizes_and_filenames(self):
        catalog = M.smaller_model_catalog()
        self.assertEqual(set(catalog), {f for f,d in setup.FAMILIES.items()
                                        if d.get("engine")=="poly" and d.get("metal", True)})
        for family, entry in catalog.items():
            self.assertTrue(entry["metal_ready"])
            self.assertFalse(entry["model_validated"])
            self.assertEqual(set(entry["models"]), set(setup.family_models(family)))
            for model, spec in entry["models"].items():
                self.assertEqual(spec["files"][0], setup.model_file(setup.FAMILIES[family], model, 1))
                self.assertGreater(spec["download_gb"], 0)
        self.assertIn("UD-Q3_K_M", catalog["gemma4"]["models"])

    def test_listing_needs_no_hardware_or_downloads(self):
        with mock.patch.object(M.S, "say") as say, \
                mock.patch.object(M.S, "data_folder", side_effect=AssertionError("writes")), \
                mock.patch.object(M, "mac_info", side_effect=AssertionError("hardware")), \
                mock.patch.object(M.S, "download", side_effect=AssertionError("download")):
            self.assertEqual(M.main(["--list-models"]), 0)
        self.assertTrue(any("gemma4:" in c.args[0] for c in say.call_args_list))


class SmallerModelSetup(unittest.TestCase):
    def test_existing_files_keep_shared_server_features(self):
        from contextlib import ExitStack
        for family, model in [(f,m) for f in M.MAC_POLY_FAMILIES for m in setup.family_models(f)]:
            with self.subTest(family=family, model=model), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                fam = setup.FAMILIES[family]
                spec = setup.family_models(family)[model]
                names = [setup.model_file(fam, model, i) for i in range(1, setup.model_shards(fam, model)+1)]
                if spec.get("draft_model"):
                    names.append(Path(spec["draft_model"]).name)
                if fam.get("draft") and spec.get("draft") != "own":
                    draft = root / fam["draft"]
                    draft.parent.mkdir(parents=True, exist_ok=True)
                    draft.write_bytes(b"fixture draft")
                for name in names:
                    (root / name).parent.mkdir(parents=True, exist_ok=True)   # ornith122 keeps its shards in a folder
                    (root / name).write_bytes(b"fixture")
                def patch(obj, name, **kw):
                    return stack.enter_context(mock.patch.object(obj, name, **kw))
                patch(M, "ROOT", new=root)
                patch(M, "mac_configs", return_value=[])
                patch(M, "mac_info", return_value={"chip":"M1 Pro", "ram_gb":32, "macos":"27.0", "arm64":True, "clt":"/tools", "perf_cores":8})
                patch(setup, "data_folder", return_value=(root, []))
                patch(setup, "free_gb", return_value=1000)
                patch(setup, "pip_install")
                patch(setup, "get_llama_cpp", return_value=root / "llama")
                build = patch(M, "build_engine", return_value=root / "strata-poly")
                patch(setup, "download", side_effect=AssertionError("model download"))
                patch(setup, "done", return_value=True)
                patch(setup, "verify_sha256")
                patch(setup, "check_shards")
                patch(setup, "run")
                patch(setup, "find_in", return_value=None)
                patch(M.shutil, "copyfile")
                write = patch(setup, "write_setup_config")
                patch(setup, "write_run_script", return_value=root / "run.sh")
                patch(M, "start", side_effect=AssertionError("server start"))
                for fn in ("say", "ok", "step", "warn"):
                    patch(setup, fn)
                rc = M.main(["--family",family,"--model",model,"--gguf-dir",str(root),"--yes","--no-start",
                             "--no-download-models","--vision","no","--parallel","3","--conversation-cache-mib","128"])
                self.assertEqual(rc, 0)
                build.assert_called_once_with(root / "llama")
                cfg = write.call_args.args[1]
                self.assertEqual(cfg["backend"], "metal")
                saved = write.call_args.args[0]
                import json
                saved.write_text(json.dumps(cfg))
                choice = setup.choices_from_config(saved)
                self.assertEqual((choice["family"], choice["model"]), (family, model))
                self.assertEqual(Path(cfg["exe"]).name, "strata-poly")
                self.assertEqual(cfg["parallel"], 3)
                self.assertIn("--conversation-cache-mib", cfg["args"])
                self.assertEqual(cfg["args"][cfg["args"].index("--model")+1], str(root/names[0]))
                if spec.get("draft_model"):
                    self.assertIn("--draft-model", cfg["args"])
                self.assertNotIn("--pack", cfg["args"])

    def test_no_download_models_also_covers_missing_draft(self):
        import argparse
        family = "gemma4"
        model = next(iter(setup.family_models(family)))
        spec = setup.family_models(family)[model]
        self.assertIn("draft_model", spec)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / setup.model_file(setup.FAMILIES[family], model, 1)).write_bytes(b"fixture")
            a = argparse.Namespace(family=family, model=model, yes=True, context=8192, vision="no", models_dir=None,
                                   gguf_dir=tmp, update=False, build=False, no_download_models=True)
            with mock.patch.object(setup, "pip_install"), mock.patch.object(setup, "get_llama_cpp", return_value=root), \
                    mock.patch.object(M, "build_engine", return_value=root / "strata-poly"), \
                    mock.patch.object(setup, "done", return_value=True), mock.patch.object(setup, "check_shards"), \
                    mock.patch.object(setup, "verify_sha256"), mock.patch.object(setup, "download") as download:
                with self.assertRaises(SystemExit):
                    M.setup_poly(a, {"ram_gb":32}, root, [])
                download.assert_not_called()

    def test_missing_local_file_stops_before_build(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(M, "build_engine") as build, \
                mock.patch.object(setup, "pip_install") as pip, mock.patch.object(setup, "download") as download:
            import argparse
            a = argparse.Namespace(family="gemma4",model="UD-Q3_K_M",yes=True,context=8192,vision="no",models_dir=None,
                                   gguf_dir=tmp,update=False)
            with self.assertRaises(SystemExit):
                M.setup_poly(a,{"ram_gb":32},Path(tmp),[])
            build.assert_not_called()
            pip.assert_not_called()
            download.assert_not_called()


class FullCatalog(unittest.TestCase):
    def test_every_family_and_quant_uses_the_shared_catalog(self):
        catalog = M.model_catalog()
        # not llama-server's (Bolt), nor one the Metal build has no kernels for (Super: "metal": False)
        poly = {f for f, d in setup.FAMILIES.items() if d.get("engine") == "poly" and d.get("metal", True)}
        self.assertEqual(set(catalog), poly)
        self.assertEqual(set(M.MAC_POLY_FAMILIES), poly)
        self.assertNotIn("nemotron-bolt", catalog)
        self.assertNotIn("nemotron-super", catalog)
        for family, entry in catalog.items():
            self.assertEqual(entry["models"], setup.family_models(family))
            self.assertEqual(entry["metal_engine"], "strata-poly")


class Menu(unittest.TestCase):
    """No --family: step 2 lists the families and takes the first with --yes (or Enter)."""

    def test_the_menu_offers_every_family_and_hands_the_choice_on(self):
        said, chosen = [], []
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(M, "mac_configs", return_value=[]), \
                mock.patch.object(M, "mac_info", return_value={"chip": "M1 Pro", "ram_gb": 32, "macos": "27.0",
                                                               "arm64": True, "clt": "/tools", "perf_cores": 8}), \
                mock.patch.object(setup, "data_folder", return_value=(Path(tmp), [])), \
                mock.patch.object(setup, "free_gb", return_value=1000), \
                mock.patch.object(M, "setup_poly", side_effect=lambda a, *_: chosen.append(a.family) or 0), \
                mock.patch.object(setup, "say", side_effect=lambda *a, **k: said.append(" ".join(map(str, a)))), \
                mock.patch.object(setup, "ok"), mock.patch.object(setup, "step"), mock.patch.object(setup, "warn"):
            self.assertEqual(M.main(["--yes", "--no-start"]), 0)
        self.assertEqual(chosen, [M.MAC_POLY_FAMILIES[0]])
        for family in M.MAC_POLY_FAMILIES:
            self.assertTrue(any(setup.FAMILIES[family]["title"] in line for line in said), family)


if __name__ == "__main__":
    unittest.main()
