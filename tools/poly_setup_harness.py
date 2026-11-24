"""The harness the setup tests use to run setup.main() for a strata-poly family with every outside effect mocked:
the GPU, RAM and CPU it reports, the downloads, the engine's build, the tokenizer and profile steps, the start.  It was
tools/test_setup_glm.py's Main class (GLM is no longer offered); a test class takes its methods:

    class Main(unittest.TestCase):
        setUp, tearDown, main = PolyMain.setUp, PolyMain.tearDown, PolyMain.main

main() returns (exit code, printed text, the run config written or None).  Not a test module itself.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def quiet(fn, *args):
    """-> (exit code or None, printed text)."""
    out = io.StringIO()
    code = None
    with contextlib.redirect_stdout(out):
        try:
            code = fn(*args)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


class PolyMain:
    """setup.main() for --family FAMILY (default occamy, a family that reads pictures), every outside effect mocked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        (self.t / "data").mkdir()
        self.downloads, self.runs, self.verified, self.built = [], [], [], []
        self.portable = None

    def tearDown(self):
        self.tmp.cleanup()

    def main(self, argv=(), ram=128.0, vram=31.84, model=False, answers=None, win=False, n_gpus=1, family="occamy",
             cores=None, configs=(), cards=None, env=None, no_start=True, extra=(), avx512=True):
        """answers: None = --yes; else {words of a question: its answer} (Enter for the others); the questions asked
        are kept in self.asked.  model: False or a size's name.  cores: (performance,
        efficiency) of a hybrid CPU.  configs: (file name, dict) of run configs already in the folder (an earlier
        install that setup runs again over).  cards: the NVIDIA cards nvidia-smi lists (default: n_gpus RTX 5090s
        of `vram`).  env: environment variables for the run (STRATA_CUDA and STRATA_EXPERIMENTAL_SM60 are cleared).
        family=None: no --family (argv may name one); no_start=False: no --no-start; extra: more patches (they win
        over the harness's own); avx512: what the mocked CPU has."""
        self.asked = []
        for name, cfg in configs:
            (self.t / name).write_text(json.dumps(cfg), encoding="utf-8")

        def fake_input(prompt=""):
            if answers is None:
                raise AssertionError(f"asked {prompt!r}")
            self.asked.append(prompt)
            return next((v for k, v in answers.items() if k in prompt), "")

        found = cards if cards is not None else \
            [{"index": i, "name": "NVIDIA GeForce RTX 5090", "vram_gb": vram, "arch": "120", "driver": "580.97"}
             for i in range(n_gpus)]

        def fake_download(url, dst, what=None):
            self.downloads.append(url)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)

        def fake_build(gpu, yes, llama, vision="none", toolkit=13, portable=False):
            self.built.append(gpu["archs"])
            self.encoder, self.toolkit, self.portable = vision, toolkit, portable
            eng = self.t / setup.ENGINE_POLY_DIR
            eng.mkdir(exist_ok=True)
            (eng / "BUILD.json").write_text(json.dumps({"source": "local", "cuda_dirs": ["/cuda/lib64"],
                                                        **({"portable": True} if portable else {})}))
            return eng

        def fake_llama_build(gpu, yes, toolkit=13):
            self.built.append(gpu["archs"])
            self.toolkit = toolkit
            eng = self.t / setup.ENGINE_LLAMA_DIR
            eng.mkdir(exist_ok=True)
            (eng / "BUILD.json").write_text(json.dumps({"source": "local", "engine": "llama-server",
                                                        "commit": setup.LLAMA_SERVER_COMMIT,
                                                        "cuda_dirs": ["/cuda/lib64"]}))
            return eng

        patches = [
            mock.patch.object(setup, "ROOT", self.t),
            mock.patch.object(setup, "WIN", win),
            mock.patch.object(setup, "GPU_PICK", None),
            mock.patch.object(setup, "OLD_GPUS", None),         # main() sets this global
            mock.patch.dict(os.environ, {"STRATA_CUDA": "", "STRATA_EXPERIMENTAL_SM60": "", **(env or {})}),
            mock.patch.object(setup, "data_folder", lambda d: (self.t / "data", [])),
            mock.patch.object(setup, "installed_configs", lambda: []),
            mock.patch.object(setup, "gpus", lambda: found),
            mock.patch.object(setup, "amd_gpus", lambda *a: []),
            mock.patch.object(setup, "ram_gb", lambda: ram),
            mock.patch.object(setup, "cpu_info", lambda: ("Test CPU", True, avx512)),
            mock.patch.object(setup, "cpu_cores", lambda: cores),
            mock.patch.object(setup, "page_file_gb", lambda: 16.0),
            mock.patch.object(setup, "free_gb", lambda p: 500.0),
            mock.patch.object(setup, "pip_install", lambda *a, **k: None),
            mock.patch.object(setup, "get_llama_cpp", lambda: self.t / "llama.cpp"),
            mock.patch.object(setup, "build_engine_poly", fake_build),
            mock.patch.object(setup, "build_llama_server", fake_llama_build),
            mock.patch.object(setup, "download", fake_download),
            mock.patch.object(setup, "whole_shard", lambda s: False),
            mock.patch.object(setup, "check_shards", lambda shards: None),
            mock.patch.object(setup, "verify_sha256", lambda s, size, sha, own=False: self.verified.append((s.name, size, sha))),
            mock.patch.object(setup, "verify_poly_own", lambda s, size, sha, fam: self.verified.append((s.name, size, sha))),
            mock.patch.object(setup, "run", lambda cmd, *a, **k: self.runs.append([str(x) for x in cmd])),
            mock.patch.object(setup, "write_run_script", lambda tag, cfg, port, *_: self.t / f"start-{tag}.sh"),
            mock.patch.object(setup, "start", mock.Mock(side_effect=AssertionError("started"))),
            mock.patch.object(sys, "argv", ["setup.py", *(["--family", family] if family else []),
                                            *(["--model", model] if model else []),
                                            *(["--yes"] if answers is None else []),
                                            *(["--no-start"] if no_start else []),
                                            "--models-dir", str(self.t / "models"), *argv]),
            mock.patch("builtins.input", fake_input),
            *extra,
        ]
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            code, out = quiet(setup.main)
        pattern = f"strata-{setup.FAMILIES[family]['tag']}*.json" if family else "strata-*.json"
        written = sorted(self.t.glob(pattern), key=lambda p: p.stat().st_mtime)
        cfg = written[-1] if written and code == 0 else None
        return code, out, (json.loads(cfg.read_text()) if cfg else None)
