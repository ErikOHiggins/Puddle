"""Puddle's own families (phase 1c): Laguna-S 2.1 and Laguna-XS 2.1, beside Gemma 4; and Nemotron 3.5 Bolt on
llama-server (B4).  Each size's files are pinned (revision, size, SHA-256), the run config a --yes install writes has
the names, the KV format and the image encoder the live configs on echo have, and Laguna-S is the menu's default.
Mocked (tools/poly_setup_harness.py): no GPU, no downloads.

    python -m unittest tools.test_setup_puddle_families
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from tools.poly_setup_harness import PolyMain  # noqa: E402

SHA = re.compile(r"/resolve/[0-9a-f]{40}/$")
# echo's run configs (local/liveConfigs, 2026-11-19): the names clients ask the server for
LIVE_NAMES = {("laguna-s", "UD-Q3_K_XL"): ("laguna-s-2.1-ud-q3_k_xl", "Laguna S 2.1 UD-Q3_K_XL"),
              ("laguna-xs", "Q4_K_M"): ("laguna-xs-2.1-q4_k_m", "Laguna XS 2.1 Q4_K_M")}


def arg(cfg, flag):
    a = cfg["args"]
    return a[a.index(flag) + 1] if flag in a else None


class Catalog(unittest.TestCase):
    def test_the_families_and_their_order(self):
        self.assertEqual(list(setup.FAMILIES), ["laguna-s", "laguna-xs", "nemotron-bolt", "nemotron-bolt-poly",
                                                "nemotron-super", "gemma4"])
        self.assertEqual(list(setup.family_models("nemotron-bolt")), ["24GB"])
        self.assertEqual(list(setup.family_models("laguna-s")), ["UD-Q3_K_XL", "UD-Q4_K_XL", "Q4_K_M"])
        self.assertEqual(list(setup.family_models("laguna-xs")), ["Q4_K_M"])

    def test_every_file_is_pinned_with_its_size_and_sha256(self):
        for family, fam in setup.FAMILIES.items():
            for model, d in setup.family_models(family).items():
                sha = {**fam["sha256"], **d.get("sha256", {})}
                with self.subTest(family=family, model=model):
                    self.assertRegex(d.get("hf", fam["hf"]), SHA)
                    files = [setup.model_file(fam, model, i) for i in range(1, setup.model_shards(fam, model) + 1)]
                    if d.get("draft_model"):                      # a size's download has its draft model too
                        files.append(d["draft_model"])
                    total = 0
                    for f in files:
                        size, digest = sha[Path(f).name]
                        self.assertRegex(digest, r"^[0-9a-f]{64}$")
                        total += size
                    self.assertAlmostEqual(total / 1e9, d["download_gb"], delta=0.06)
                    if fam.get("engine") != "llama-server":   # llama-server's sizes have no experts in RAM
                        self.assertLess(d["arena_gb"], d["download_gb"])
                    if fam.get("vision"):
                        self.assertRegex(d.get("mmproj_hf", fam["mmproj_hf"]), SHA)
                        self.assertIn(fam["mmproj"], sha)

    def test_the_architectures(self):
        self.assertEqual(setup.POLY_GGUF_ARCHS["laguna-s"], ("laguna",))
        self.assertEqual(setup.POLY_GGUF_ARCHS["laguna-xs"], ("laguna",))
        self.assertEqual(setup.POLY_GGUF_ARCHS["gemma4"], ("gemma4",))

    def test_laguna_s_shards_and_poolside_s_own_file(self):
        fam = setup.FAMILIES["laguna-s"]
        self.assertEqual([setup.model_file(fam, "UD-Q3_K_XL", i) for i in (1, 2, 3)],
                         [f"UD-Q3_K_XL/Laguna-S-2.1-UD-Q3_K_XL-0000{i}-of-00003.gguf" for i in (1, 2, 3)])
        d = setup.family_models("laguna-s")["Q4_K_M"]
        self.assertEqual(setup.model_shards(fam, "Q4_K_M"), 1)
        self.assertEqual(setup.model_file(fam, "Q4_K_M", 1), "laguna-s-2.1-Q4_K_M.gguf")
        self.assertIn("/poolside/Laguna-S-2.1-GGUF/", d["hf"])
        self.assertIn("BF16", d["about"])                           # what it costs is said in the menu


class Install(unittest.TestCase):
    setUp, tearDown, main = PolyMain.setUp, PolyMain.tearDown, PolyMain.main

    def test_each_family_writes_its_sampling(self):
        # Laguna: echo's testing (min_p 0.1); Gemma 4: Google's model card
        want = {"laguna-s": {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "min_p": 0.1},
                "laguna-xs": {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "min_p": 0.1},
                "nemotron-bolt": {"temperature": 1.0, "top_p": 0.95, "top_k": 40, "min_p": 0.05},
                "nemotron-bolt-poly": {"temperature": 1.0, "top_p": 0.95, "top_k": 40, "min_p": 0.05},
                "nemotron-super": {"temperature": 1.0, "top_p": 0.95, "top_k": 40, "min_p": 0.05},
                "gemma4": {"temperature": 1.0, "top_p": 0.95, "top_k": 64}}
        self.assertEqual({f: fam.get("sampling") for f, fam in setup.FAMILIES.items()}, want)
        for family, sampling in want.items():
            with self.subTest(family=family):
                self.tearDown()
                self.setUp()
                code, out, cfg = self.main(family=family, ram=128.0, vram=15.9, n_gpus=2)
                self.assertEqual(code, 0, out[-3000:])
                self.assertEqual(cfg["sampling"], sampling)
                self.assertIn("sampling for requests that set none", out)
                # the server takes the block as it is (a bad value would stop it at its start)
                sys.path.insert(0, str(ROOT / "serve"))
                from serve import server
                self.assertEqual(server.sampling_defaults_from_config(cfg), sampling)

    def test_a_sampling_block_of_the_user_s_is_kept(self):
        mine = {"temperature": 0.6, "top_p": 0.9}
        code, out, cfg = self.main(family="laguna-s", model="UD-Q3_K_XL", ram=128.0, vram=15.9, n_gpus=2,
                                   configs=[("strata-laguna-s21-ud-q3_k_xl.json",
                                             {"exe": "x", "args": [], "sampling": mine})])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["sampling"], mine)
        self.assertNotIn("sampling for requests that set none", out)

    def test_the_menu_s_default_is_laguna_s(self):
        code, out, cfg = self.main(family=None, ram=128.0, vram=15.9, n_gpus=2)
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["model_name"], "laguna-s-2.1-ud-q3_k_xl")

    def test_the_names_are_the_live_configs(self):
        for (family, model), (name, display) in LIVE_NAMES.items():
            with self.subTest(family=family):
                self.tearDown()
                self.setUp()
                code, out, cfg = self.main(family=family, model=model, ram=128.0, vram=15.9, n_gpus=2)
                self.assertEqual(code, 0, out[-3000:])
                self.assertEqual((cfg["model_name"], cfg["display_name"]), (name, display))

    def test_laguna_s_q3(self):
        code, out, cfg = self.main(family="laguna-s", model="UD-Q3_K_XL", ram=128.0, vram=15.9, n_gpus=2)
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(arg(cfg, "--kv"), "q8_0")                  # as it runs on echo
        model = Path(arg(cfg, "--model"))
        self.assertEqual(model.parts[-3:], ("laguna-s21-UD-Q3_K_XL", "UD-Q3_K_XL",
                                            "Laguna-S-2.1-UD-Q3_K_XL-00001-of-00003.gguf"))
        self.assertEqual(len(self.downloads), 3)
        self.assertTrue(all("/unsloth/Laguna-S-2.1-GGUF/resolve/750f92f9" in u and "/UD-Q3_K_XL/" in u
                            for u in self.downloads), self.downloads)
        self.assertNotIn("vision", cfg)
        self.assertIn("OpenMDW-1.1", out)

    def test_poolside_s_q4_k_m_comes_from_poolside(self):
        code, out, cfg = self.main(family="laguna-s", model="Q4_K_M", ram=128.0, vram=15.9, n_gpus=2)
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(self.downloads, ["https://huggingface.co/poolside/Laguna-S-2.1-GGUF/resolve/"
                                          "19036076775e9ba6758595f78af99cb976ef8ff0/laguna-s-2.1-Q4_K_M.gguf"])
        self.assertEqual(arg(cfg, "--kv"), "q8_0")

    def test_laguna_xs(self):
        code, out, cfg = self.main(family="laguna-xs", ram=64.0, vram=15.9)
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(arg(cfg, "--kv"), "q8_0")
        self.assertEqual(Path(arg(cfg, "--model")).parts[-2:], ("laguna-xs21-Q4_K_M", "Laguna-XS-2.1-Q4_K_M.gguf"))

    def test_a_small_pc_gets_the_smaller_size(self):
        code, out, cfg = self.main(family="gemma4", ram=14.0, vram=7.9)   # UD-Q3_K_M wants more
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["model_name"], "gemma-4-26b-a4b-ud-q2_k_xl")


class BoltOnStrataPoly(unittest.TestCase):
    """Nemotron 3.5 Bolt on strata-poly (nemotron-bolt-poly), opt-in beside the llama-server family: Bolt's files, from
    that family's folder (downloaded once), a config and model name of its own."""
    setUp, tearDown, main = PolyMain.setUp, PolyMain.tearDown, PolyMain.main

    def two_16(self, *argv, family="nemotron-bolt-poly", **kw):
        cards = [{"index": i, "name": "NVIDIA GeForce RTX 5060 Ti", "vram_gb": 15.93, "arch": "120", "driver": "580.97"}
                 for i in range(2)]
        return self.main(list(argv), family=family, ram=128.0, cards=cards, **kw)

    def test_the_family_is_bolts_files_on_strata_poly(self):
        bolt, poly = setup.FAMILIES["nemotron-bolt"], setup.FAMILIES["nemotron-bolt-poly"]
        for k in ("hf", "file", "shards", "mmproj_hf", "mmproj", "vision_gb", "sha256", "license"):
            self.assertEqual(poly[k], bolt[k], k)
        self.assertEqual((poly["engine"], poly["args"]), ("poly", ["--no-draft"]))
        self.assertEqual(setup.POLY_GGUF_ARCHS["nemotron-bolt-poly"], ("nemotron_h_moe",))
        self.assertEqual(list(poly["models"]), list(bolt["models"]))
        # its size is the other family's: that folder; a size of its own would have its own
        self.assertEqual(setup.files_tag(poly, "nemotron-poly-bolt-24GB", "24GB"), "nemotron-bolt-24GB")
        self.assertEqual(setup.files_tag(poly, "nemotron-poly-bolt-UD-Q4_K_XL", "UD-Q4_K_XL"),
                         "nemotron-poly-bolt-UD-Q4_K_XL")

    def test_a_config_of_its_own_on_strata_poly(self):
        code, out, cfg = self.two_16("--vision", "gpu")             # pictures are asked for, as Super's are
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(Path(cfg["exe"]).name, setup.POLY_EXE)
        self.assertEqual(cfg["model_name"], "nemotron-3.5-bolt-poly-24gb")
        self.assertIn("--no-draft", cfg["args"])
        self.assertEqual(Path(arg(cfg, "--model")).parent.name, "nemotron-bolt-24GB")
        self.assertEqual(Path(cfg["vision"]["mmproj"]).name, setup.FAMILIES["nemotron-bolt"]["mmproj"])
        self.assertFalse(cfg["vision"].get("max_tokens"))           # the encoder file's own limits
        self.assertNotIn("computed on the CPU", out)                # every expert fits the two cards
        self.assertEqual(setup.choices_from_config(next(self.t.glob("strata-nemotron-poly-bolt-*.json")))["family"],
                         "nemotron-bolt-poly")

    def test_one_card_says_part_runs_on_the_cpu(self):
        cards = [{"index": 0, "name": "NVIDIA GeForce RTX 5060 Ti", "vram_gb": 15.93, "arch": "120",
                  "driver": "580.97"}]
        code, out, cfg = self.main(family="nemotron-bolt-poly", ram=128.0, cards=cards)
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("computed on the CPU", out)

    def test_the_llama_server_familys_download_is_used(self):
        code, out, cfg = self.two_16(family="nemotron-bolt")
        self.assertEqual(code, 0, out[-3000:])
        first = list(self.downloads)
        code, out, cfg = self.two_16()
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(self.downloads, first)                   # nothing downloaded again
        configs = sorted(x.name for x in self.t.glob("strata-nemotron-*.json"))
        self.assertEqual(configs, ["strata-nemotron-bolt-24gb.json", "strata-nemotron-poly-bolt-24gb.json"])


class Bolt(unittest.TestCase):
    """Nemotron 3.5 Bolt: llama-server (compiled here for NVIDIA), NVIDIA's own file and image encoder, echo's args."""
    setUp, tearDown, main = PolyMain.setUp, PolyMain.tearDown, PolyMain.main

    def two_16(self, *argv, **kw):
        cards = [{"index": i, "name": "NVIDIA GeForce RTX 5060 Ti", "vram_gb": 15.93, "arch": "120", "driver": "580.97"}
                 for i in range(2)]
        return self.main(list(argv), family="nemotron-bolt", ram=128.0, cards=cards, **kw)

    def test_echos_cards_get_echos_command(self):
        code, out, cfg = self.two_16()
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["engine"], "llama-server")
        self.assertEqual(Path(cfg["exe"]).parts[-2:], (setup.ENGINE_LLAMA_DIR, setup.LLAMA_SERVER_EXE))
        model, mmproj = Path(arg(cfg, "--model")), Path(arg(cfg, "--mmproj"))
        self.assertEqual(model.name, "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf")
        self.assertEqual(mmproj.name, "mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf")
        self.assertEqual(cfg["args"][4:], ["--ctx-size", "168000", "--parallel", "1", "--flash-attn", "on",
                                           "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--batch-size", "2048",
                                           "--ubatch-size", "2048", "--load-mode", "none", "--n-gpu-layers", "all",
                                           "--tensor-split", "1,1"])
        self.assertEqual((cfg["gpu"], cfg["model_name"], cfg["display_name"]),
                         ([0, 1], "nemotron-3.5-bolt-24gb", "Nemotron 3.5 Bolt 24GB"))
        self.assertEqual(cfg["sampling"], {"temperature": 1.0, "top_p": 0.95, "top_k": 40, "min_p": 0.05})
        self.assertNotIn("vision", cfg)                  # llama-server reads pictures with its --mmproj
        self.assertEqual(self.built, [[120]])
        pin = "/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/resolve/e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246/"
        self.assertEqual(self.downloads, ["https://huggingface.co" + pin + model.name,
                                          "https://huggingface.co" + pin + mmproj.name])
        self.assertEqual([v[0] for v in self.verified], [model.name, mmproj.name])
        self.assertTrue(any("strata_tokenizer.py" in " ".join(r) for r in self.runs), self.runs)
        self.assertIn("OpenMDW-1.1", out)
        from serve import server                         # the server takes its args
        self.assertIsNone(server.llama_server_args_problem(cfg["args"]))

    def test_other_cards_let_llama_server_fit_the_context(self):
        code, out, cfg = self.main(family="nemotron-bolt", ram=64.0, vram=31.84)      # one RTX 5090
        self.assertEqual(code, 0, out[-3000:])
        self.assertNotIn("--ctx-size", cfg["args"])
        self.assertNotIn("--tensor-split", cfg["args"])
        self.assertIn("chosen by llama-server each time it loads", out)
        self.tearDown()
        self.setUp()
        code, out, cfg = self.two_16("--context", "65536")
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(arg(cfg, "--ctx-size"), "65536")

    def test_too_little_vram_is_refused(self):
        code, out, cfg = self.main(family="nemotron-bolt", ram=128.0, vram=23.99)     # one 24 GB card
        self.assertNotEqual(code, 0)
        self.assertIn("30 GB of VRAM or more", out)
        self.assertIsNone(cfg)
        self.tearDown()
        self.setUp()
        code, out, cfg = self.main(["--model", "24GB"], family="nemotron-bolt", ram=128.0, vram=15.93)
        self.assertNotEqual(code, 0)                     # --model and --yes are no way round it
        self.assertIn("30 GB of VRAM or more", out)

    def test_images_off_leaves_the_encoder_out(self):
        code, out, cfg = self.two_16("--vision", "no")
        self.assertEqual(code, 0, out[-3000:])
        self.assertNotIn("--mmproj", cfg["args"])
        self.assertEqual(len(self.downloads), 1)

    def test_files_you_have_are_used_where_they_are(self):
        own = self.t / "own"
        own.mkdir()
        for name in setup.FAMILIES["nemotron-bolt"]["sha256"]:
            (own / name).write_bytes(b"")
        code, out, cfg = self.two_16("--gguf-dir", str(own))
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(self.downloads, [])
        self.assertEqual(Path(arg(cfg, "--model")).parent, own)

    def test_the_check_says_it_by_the_cards_together(self):
        code, out, _ = self.two_16("--check")
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("Nemotron 3.5 Bolt 24GB: EXPERIMENTAL, fits: 32 GB of VRAM on the chosen cards", out)

    def test_a_later_start_compiles_it_again_only_when_the_commit_moved(self):
        import json
        from unittest import mock
        code, out, cfg = self.two_16()
        self.assertEqual(code, 0, out[-3000:])
        path = next(self.t.glob("strata-nemotron-bolt-*.json"))
        eng = Path(cfg["exe"]).parent
        (eng / setup.LLAMA_SERVER_EXE).write_bytes(b"")
        built = []
        cards = [{"index": i, "name": "NVIDIA GeForce RTX 5060 Ti", "vram_gb": 15.93, "arch": "120",
                  "driver": "580.97"} for i in range(2)]
        with mock.patch.object(setup, "gpus", lambda: cards), \
                mock.patch.object(setup, "build_llama_server", lambda g, y, t=13: built.append(g["archs"]) or eng):
            (eng / "BUILD.json").write_text(json.dumps({"commit": setup.LLAMA_SERVER_COMMIT, "archs": [120],
                                                        "cuda_dirs": ["/c"]}))
            setup.update_llama_server(cfg, cfg_path=path)
            self.assertEqual(built, [])
            (eng / "BUILD.json").write_text(json.dumps({"commit": "0" * 40, "archs": [120], "cuda_dirs": ["/c"]}))
            setup.update_llama_server(cfg, cfg_path=path)
            self.assertEqual(built, [[120]])
        ch = setup.choices_from_config(path)
        self.assertEqual((ch["family"], ch["model"], ch["context"], ch["vision"]),
                         ("nemotron-bolt", "24GB", 168000, "gpu"))


if __name__ == "__main__":
    unittest.main()
