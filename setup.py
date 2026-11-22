#!/usr/bin/env python3
"""Puddle one-click setup and start (Windows and Linux, NVIDIA or AMD GPUs; a Mac: metal/setup_mac.py).

    START-HERE.bat  (Windows)   /   ./setup.sh  (Linux)      - they install Python if needed and run this file

The first time it asks which model, which size, how much context and whether the model should also read images,
then installs everything and starts the model on http://127.0.0.1:8080 (OpenAI- and Anthropic-compatible API and a
web app).  Every later start skips straight to running the model: nothing that is already downloaded, installed or
prepared is done again.

Every model runs on strata-poly, read from its GGUF in place (docs/BACKENDS.md); Nemotron 3.5 Bolt also on llama.cpp's
llama-server when asked (nemotron-bolt-llama, compiled here as well, NVIDIA cards only for now).  What the first run
does (each step is skipped when it is already done):

  1. checks your PC: NVIDIA or AMD GPU and driver, RAM, CPU, free disk space
  2. asks the questions
  3. installs the Python packages it needs into .venv
  4. compiles strata-poly for your GPU from this checkout's source (it installs the build tools first, and asks);
     AMD (--backend hip, chosen by itself on a PC with no usable NVIDIA card): with AMD's ROCm
  5. downloads the model from Hugging Face (resumable, a pinned revision), its draft model, and the image encoder if
     you want images
  6. prepares the model's tokenizer and expert profile
  7. writes run-<model>.bat / run-<model>.sh and starts the model

Options: --family F, --model SIZE, --context 32768, --vision yes|no|gpu|cpu, --port 8080, --yes (recommended
answers, no questions), --setup (install another model / change settings instead of starting), --no-start,
--host 0.0.0.0 --api-key KEY (reach it from other devices on your network), --models-dir DIR, --gguf-dir DIR (use
GGUF files you already have), --max-compatibility (Windows: an engine that also runs on other PCs), --cuda 12|13,
--check (only check this PC).

Setup recommends, it never forces: the recommended answers are the defaults (--yes, or Enter), and a bigger choice
than it recommends - a longer context, more GPUs, a size it thinks will not fit - is kept, with what it risks.  With
--yes, an explicit flag (--model, --gpus, ...) is the consent to a risk setup would otherwise stop at; --yes alone
is not.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WIN = os.name == "nt"
# #214: every Hugging Face file comes from a fixed commit of its repository (the `sha` of
# https://huggingface.co/api/models/<repo> when this was pinned), so a checkout installs the same files on any
# day.  A revision the repository no longer has falls back to its current files, with a message (download()).
HF_REVISIONS = {
    "unsloth/Laguna-S-2.1-GGUF": "750f92f90cf54159c4d7a610cb7b3e74498e75c6",                       # 2026-07-27
    "poolside/Laguna-S-2.1-GGUF": "19036076775e9ba6758595f78af99cb976ef8ff0",                      # 2026-08-06
    "poolside/Laguna-XS-2.1-GGUF": "1a37c0a5fb8c7a18e6106decb6be6327d1b63fa6",                     # 2026-07-18
    "unsloth/gemma-4-26B-A4B-it-GGUF": "c099eb48e663fd284577b04978a94ffccb261841",                     # 2026-10-05
    "unsloth/gemma-4-26B-A4B-it-qat-GGUF": "7b92b5b28818151e8669af2e45e88d6086f490dd",                 # 2026-10-26
    "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF": "e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246",  # 2026-10-15
    "unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF": "34841e360524d421c564cc3ed247aa04a865473b",      # 2026-10-16
    "unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF": "75a5829055f69a0359085ae6bbd0a81b60bc05c6",  # 2026-10-15
}


HF_DEFAULT = "https://huggingface.co"


def hf_endpoint() -> str:
    """#495: the Hugging Face host - HF_ENDPOINT as huggingface_hub reads it (a mirror, e.g. https://hf-mirror.com),
    else huggingface.co.  The pinned revisions and the SHA-256 checks are the same whichever host serves the files."""
    return (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/") or HF_DEFAULT


def hf(repo: str) -> str:
    """The download folder of a Hugging Face repository at its pinned revision."""
    return f"{hf_endpoint()}/{repo}/resolve/{HF_REVISIONS[repo]}/"


def hf_unpinned(url: str) -> str:
    """The same file at the repository's current revision (main)."""
    return re.sub(r"^(https?://[^/]+/.+?/resolve/)[0-9a-f]{40}/", r"\1main/", url, count=1)


# ModelScope (www.modelscope.cn) hosts every repository above under the same name, with the same file paths and
# sizes, and is reachable from mainland China where huggingface.co often is not.  It serves a repository's current
# files (no pinned revision), so a file from it is checked against the SHA-256 ModelScope publishes for it, and the
# MTP tensors against the pinned checkpoint's own hashes (tools/mtp_fetch.py).
# --source / STRATA_SOURCE: huggingface (what auto means), or modelscope: only when asked for (a different host
# serving current files, not our pinned revision: setup never switches to it by itself, it recommends it, #908).
MS_DEFAULT = "https://www.modelscope.cn"
SOURCES = ("auto", "modelscope", "huggingface")
HF_FILE = re.compile(r"^https?://[^/]+/(?P<repo>[^/]+/[^/]+)/resolve/[^/]+/(?P<path>.+)$")
_sources = {}                                      # model_source()'s answer per (STRATA_SOURCE, HF_ENDPOINT, host)
_ms_files = {}


def ms_endpoint() -> str:
    return (os.environ.get("MODELSCOPE_ENDPOINT") or "").strip().rstrip("/") or MS_DEFAULT


def reachable(url: str, timeout: float = 5.0) -> bool:
    """Whether `url` answers a HEAD request with success (2xx, after redirects) within `timeout` seconds."""
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers={"User-Agent": "strata-setup"}),
                               timeout=timeout).close()
        return True
    except OSError:                                    # HTTPError too: an error page is not the file
        return False


def model_source() -> str:
    """"modelscope" or "huggingface".  ModelScope only when --source modelscope / STRATA_SOURCE=modelscope asks for it;
    everything else (auto, a HF_ENDPOINT mirror chosen on purpose, #495) is Hugging Face.  Setup does not switch hosts
    by itself (#908): ModelScope serves a repository's current files, not our pinned revision."""
    want = (os.environ.get("STRATA_SOURCE") or "auto").strip().lower()
    key = (want, os.environ.get("HF_ENDPOINT") or "", ms_endpoint())
    if key not in _sources:
        _sources[key] = "modelscope" if want in ("ms", "modelscope") else "huggingface"
    return _sources[key]


def source_hint(url: str) -> str:
    """The recommendation that goes with a failed download from Hugging Face: ModelScope, by an explicit choice."""
    if "huggingface" in url or "hf-mirror" in url:
        return ("; if huggingface.co does not reach you (mainland China), the same files are on ModelScope: run setup "
                "with --source modelscope (docs/INSTALL.md)")
    return ""


def ms_file(url: str):
    """(repo, path) when `url` is a Hugging Face file of a repository setup knows (HF_REVISIONS), else None."""
    m = HF_FILE.match(url)
    if m is None or m.group("repo") not in HF_REVISIONS:
        return None
    return m.group("repo"), m.group("path")


def ms_url(repo: str, path: str) -> str:
    return f"{ms_endpoint()}/models/{repo}/resolve/master/{path}"


def ms_meta(repo: str, path: str):
    """(size, sha256) ModelScope publishes for a file, or None when it cannot be asked."""
    if repo not in _ms_files:
        try:
            api = f"{ms_endpoint()}/api/v1/models/{repo}/repo/files?Recursive=true"
            with urllib.request.urlopen(urllib.request.Request(api, headers={"User-Agent": "strata-setup"}),
                                        timeout=60) as r:
                files = json.loads(r.read())["Data"]["Files"]
            _ms_files[repo] = {f["Path"]: (int(f.get("Size") or 0), (f.get("Sha256") or "").lower()) for f in files}
        except (OSError, ValueError, KeyError, TypeError):
            return None
    return _ms_files[repo].get(path)


LLAMA_CPP_COMMIT = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
LLAMA_CPP_ZIP = f"https://github.com/ggml-org/llama.cpp/archive/{LLAMA_CPP_COMMIT}.zip"
# The image encoder's own llama.cpp, newer than the engines' pin; only strata-vision builds from it (a second checkout,
# like #14 suggests).  llama-server's commit (LLAMA_SERVER_COMMIT): its mtmd has the Nemotrons' encoder (nemotron_omni),
# and the two share one checkout.  Before phase 2 it was 69f201a2 (MiniCPM-V 4.7's).
VISION_LLAMA_CPP_COMMIT = "08c252e221759e0f69e80f0450b295c92d2344e1"
VISION_LLAMA_CPP_ZIP = f"https://github.com/ggml-org/llama.cpp/archive/{VISION_LLAMA_CPP_COMMIT}.zip"
# llama.cpp's llama-server, the engine of a family with "engine": "llama-server" (Nemotron 3.5 Bolt's opt-in; it was
# Bolt's engine until strata-poly ran the Mamba-2 hybrids): its own newer checkout (nemotron_h_moe, NVFP4, mtmd's
# nemotron_omni), the commit echo ran Bolt with (2026-11-20), compiled here into engine-llama/ (serve/server.py's
# LlamaServerEngine talks to it)
LLAMA_SERVER_COMMIT = "08c252e221759e0f69e80f0450b295c92d2344e1"
LLAMA_SERVER_ZIP = f"https://github.com/ggml-org/llama.cpp/archive/{LLAMA_SERVER_COMMIT}.zip"
LLAMA_SERVER_EXE = "llama-server.exe" if WIN else "llama-server"
ENGINE_LLAMA_DIR = "engine-llama"

MIN_DRIVER = 580                       # CUDA 13.0
# Older NVIDIA GPUs (experimental): CUDA 13 dropped Pascal (sm_60/61) and Volta (sm_70), so a model whose GPUs include
# one runs a second engine, built with CUDA 12.9 (-DSTRATA_EXPERIMENTAL_SM60=ON) and kept in its own folder: the
# ready-made one is CUDA12_ASSET (Windows; on Linux it is compiled here with a CUDA 12.x toolkit).  One engine runs per
# model, so the choice is per model config, by its oldest GPU; --cuda 12|13 overrides it (docs/OLDER_GPUS.md).
CUDA13_MIN_ARCH = 75                   # the oldest compute capability CUDA 13 compiles for (sm_75, RTX 20)
# CUDA 12.x minor-version compatibility (NVIDIA's table: Linux 525.60.13, Windows 527.41); the wheels match the 12.9.1
# toolkit the CUDA 12 zip is built with (cuBLAS 12.9.1.4, runtime 12.9.79).  Not tested on such an old driver here.
CUDA12_MIN_DRIVER = 528 if WIN else 525
PY_PACKAGES = ["numpy", "jinja2", "regex", "pyyaml", "tqdm", "requests", "cmake", "ninja", "pillow", "psutil"]
REQUIREMENTS = ROOT / "requirements.txt"   # the same packages and their dependencies, pinned (#214)


# The model families Puddle offers: each runs on strata-poly and has sizes of its own ("models").
# Each family's sampling: setup writes it as the run config's "sampling" block (a block of the user's stays; a
# request's own values win).  Laguna: the setting echo's testing settled on (2026-11-18: min_p 0.1; poolside's
# generation_config has min_p 0.0).  Gemma 4: Google's model card ("Sampling
# Parameters") and generation_config.
LAGUNA_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "min_p": 0.1}
FAMILIES = {
    # Laguna-S 2.1 (poolside): 48 layers, 12 of them full attention (8 KV heads of 128), 36 sliding (a 512-token
    # window), 256 experts of which 10 are used.  Its KV is q8_0 (setup writes --kv q8_0), as it runs every day on
    # 2x RTX 5060 Ti 16 GB + 128 GB of RAM (echo), 262K context.  row_bytes: the q8_0 KV of the 12 full layers per
    # token (24,576 values).  vram_beside_gb: from that run's log (21.7 GiB of UD-Q3_K_XL's experts in VRAM at load on
    # 2x 15.48 GiB, less setup's 2.0 GiB for the second card) plus the q8_0 KV of 32K; the other sizes have the same
    # dense weights within 0.1 GB.  ram_gb: setup's estimate beside a 16 GB card (poly_ram_need).
    "laguna-s": {"title": "Laguna S 2.1", "by": "poolside; quantized by Unsloth", "sampling": LAGUNA_SAMPLING,
                 "about": "a 54 GB file, ~51 GB of RAM beside a 16 GB card; runs on 2x RTX 5060 Ti 16 GB at 262K; "
                          "two larger sizes too",
                 "hf": hf("unsloth/Laguna-S-2.1-GGUF"), "file": "{q}/Laguna-S-2.1-{q}-0000{i}-of-00003.gguf",
                 "shards": 3, "tag": "laguna-s21-", "name": "laguna-s-2.1", "engine": "poly", "experimental": True,
                 "ran_on_gb": 16, "kv": "q8_0", "row_bytes": 26112,
                 "license": "OpenMDW-1.1 (poolside's): https://huggingface.co/poolside/Laguna-S-2.1",
                 "models": {"UD-Q3_K_XL": {"about": "IQ3_XXS gate/up and IQ4_XS down experts (Unsloth Dynamic); the one "
                                                     "run every day",
                                           "download_gb": 54.1, "ram_gb": 51, "arena_gb": 49.8, "vram_beside_gb": 8.0},
                            "UD-Q4_K_XL": {"about": "Q4_K gate/up and Q5_K down experts (Unsloth Dynamic): higher "
                                                     "quality; not run yet",
                                           "download_gb": 73.4, "ram_gb": 71, "arena_gb": 69.0, "vram_beside_gb": 8.1},
                            # poolside's own file: 8 of its 47 expert layers are BF16, which the VRAM expert cache
                            # does not hold (native_mmvq has no BF16), so they run on the CPU; and a 1M-token rope
                            # (factor 128 over 8,192) where Unsloth's files have 262,144 (factor 32)
                            "Q4_K_M": {"about": "poolside's own file: Q4_K experts, 8 layers of them BF16 (those run on "
                                                 "the CPU), a 1M-token rope; not run yet",
                                       "hf": hf("poolside/Laguna-S-2.1-GGUF"), "file": "laguna-s-2.1-Q4_K_M.gguf",
                                       "shards": 1, "download_gb": 96.0, "ram_gb": 93, "arena_gb": 91.7,
                                       "vram_beside_gb": 8.1}},
                 "sha256": {"Laguna-S-2.1-UD-Q3_K_XL-00001-of-00003.gguf":
                          (3683648, "39de6a5ff3b9813b923e1bf8684d8ff6dc786d6337f75fa245f21d1e6f2c586b"),
                          "Laguna-S-2.1-UD-Q3_K_XL-00002-of-00003.gguf":
                          (49699878368, "0e8bc5a30c755c5faac6e6cdef3cc8c8c2425bc7468e31ecb282bbbbf6558c19"),
                          "Laguna-S-2.1-UD-Q3_K_XL-00003-of-00003.gguf":
                          (4390258592, "25c8d2c8e9e1c6c059b1b7dfbeaa02e29caeadc85974c4971dea5e7878665d2d"),
                          "Laguna-S-2.1-UD-Q4_K_XL-00001-of-00003.gguf":
                          (3683648, "0cfaf46917260d253773e5e2fab64329fa5c9c60fdf0db0f59f31205b5f5dd32"),
                          "Laguna-S-2.1-UD-Q4_K_XL-00002-of-00003.gguf":
                          (49971821312, "2296102462b02edca70163121ac62bacf7a82078c0eafc91625c8822850769bf"),
                          "Laguna-S-2.1-UD-Q4_K_XL-00003-of-00003.gguf":
                          (23419667040, "9150e2338f7690af29685b6a2ca621a8fda7ecf9724678266c4b04b7c6dd0ef3"),
                          "laguna-s-2.1-Q4_K_M.gguf":
                          (96031829760, "a8b55c75714ea73fd90ec85de5defdc0b8d88ca0ad2108343cdd8fc22f7583e4")}},
    # Laguna-XS 2.1 (poolside): 40 layers, 10 of them full attention, 256 experts of which 8 are used; q8_0 KV as
    # Laguna-S (row_bytes: 20,480 values a token).  Run on the same PC at 262K, every expert in VRAM (17.7 GiB on
    # the two cards).  vram_beside_gb: Laguna-S's less the 2.8 GiB of dense weights XS does not have (an estimate).
    "laguna-xs": {"title": "Laguna XS 2.1", "by": "poolside", "sampling": LAGUNA_SAMPLING,
                  "about": "the small Laguna: a 20 GB file, ~17 GB of RAM beside a 16 GB card; runs on 2x RTX 5060 Ti "
                           "16 GB at 262K",
                  "hf": hf("poolside/Laguna-XS-2.1-GGUF"), "file": "Laguna-XS-2.1-{q}.gguf", "shards": 1,
                  "tag": "laguna-xs21-", "name": "laguna-xs-2.1", "engine": "poly", "experimental": True,
                  "ran_on_gb": 16, "kv": "q8_0", "row_bytes": 21760,
                  "license": "OpenMDW-1.1 (poolside's): https://huggingface.co/poolside/Laguna-XS-2.1",
                  "models": {"Q4_K_M": {"about": "poolside's own file: Q4_K gate/up, Q4_K and Q6_K down experts",
                                        "download_gb": 20.3, "ram_gb": 17, "arena_gb": 19.0,
                                        "vram_beside_gb": 5.0}},
                  "sha256": {"Laguna-XS-2.1-Q4_K_M.gguf":
                          (20274300032, "1ac7079101fca5a6df8c5a7523a3c30ea7d1c0e4b1258090e7d6d4039287f6cb")}},
    # Nemotron 3.5 Bolt 74B-A9B 24GB (NVIDIA): a Mamba-2 / attention hybrid (nemotron_h_moe, 88 blocks) with 496 routed
    # experts of which 9 are used, quantized by NVIDIA with quantization-aware training (IQ2_XXS experts, NVFP4 Mamba
    # layers, Q8_0 embedding and output), and a C-RADIO v4-H image encoder (BF16).  It runs on strata-poly
    # (src/poly/nemotron_h.cpp): on echo its first choices were llama.cpp's at 198 of 199 steps, and 64 of 64 on a
    # picture (tools/poly_parity.py, 2026-11-21).  Same day, same requests through Puddle on 2x RTX 5060 Ti 16 GB,
    # against the llama-server family below: writing 60.4 tokens/s after a 17,613-token prompt (llama-server 56.5)
    # and 69.5 after a 35-token one (61.1); reading that prompt 1,766.8 tokens/s (llama-server 2,347.6).  arena_gb:
    # the file's routed experts, 17,052,794,880 bytes.  vram_beside_gb: echo, one RTX 5060 Ti 16 GB (15.93 GiB) at a
    # 32K context, 7,913 of 13,824 experts in VRAM (8.9 GiB), the estimate's way; on both cards all 13,824 (15.9
    # GiB), with the image encoder too.  vision_vram_gb: Super's measurement of the same encoder (nemotron_omni, the
    # same settings).  --no-draft: Bolt has no MTP block.  Sampling: NVIDIA's recipe
    # as llama.cpp runs it (the card's "Recommended Sampling" temperature 1.0 and top_p 0.95, with llama.cpp's top_k
    # 40 and min_p 0.05 that its command leaves on).  "vision_default_yes": the pictures question defaults to yes, as
    # it did on llama-server (reading pictures is what Bolt is used for).  UD-Q5_K_XL: Unsloth's quantization (Q5_K /
    # Q6_K up and Q6_K / Q8_0 down experts, the other weights at more bits than NVIDIA's file) from Unsloth's
    # repository, with NVIDIA's image encoder: the one the 24GB size's folder has, or its own download of it.  On echo
    # (2026-11-22) its first choices were llama.cpp's at 157 of 160 steps, the other three near ties
    # (tools/poly_parity.py: short, code, chat, a 2,190-token text, a picture).  Its vram_beside_gb: one RTX 5060 Ti 16
    # GB at 32K, 1,617 of 13,824 experts in VRAM (5.4 GiB); arena_gb: its routed experts, 50,288,885,760 bytes; ram_gb
    # by setup's estimate.  vram_card_gb: on both cards at 32K it held 4,620 (15.5 GiB), so the two took 16.4 GiB
    # beside the experts together (Super's second card takes 5.7 too).
    "nemotron-bolt": {"title": "Nemotron 3.5 Bolt", "by": "NVIDIA", "engine": "poly",
                      "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 40, "min_p": 0.05},
                      "about": "74B-A9B, reads pictures; a 22 GB file, ~17 GB of RAM beside a 16 GB card, every expert "
                               "on two; a larger size too",
                      "hf": hf("nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF"),
                      "file": "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-{q}.gguf", "shards": 1,
                      "tag": "nemotron-bolt-", "name": "nemotron-3.5-bolt", "vision": True, "experimental": True,
                      "metal": False, "ran_on_gb": 16, "row_bytes": 8192,
                      "mmproj_hf": hf("nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF"),
                      "mmproj": "mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf", "vision_gb": 1.7,
                      "vision_vram_gb": 2.5, "vision_max_tokens": 0, "vision_default_yes": True,
                      "vram_card_gb": 5.8, "contexts": [8192, 32768, 65536, 131072, 262144],
                      "args": ["--no-draft"],
                      "license": "OpenMDW-1.1 (NVIDIA's): "
                                 "https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF",
                      "models": {"24GB": {"about": "NVIDIA's own 2-bit file (quantization-aware trained)",
                                          "download_gb": 22.0, "ram_gb": 17, "arena_gb": 17.1,
                                          "vram_beside_gb": 7.0},
                                 "UD-Q5_K_XL": {"about": "Unsloth Dynamic: Q5_K / Q6_K up and Q6_K / Q8_0 down "
                                                         "experts, 6.08 bits a weight; most on the CPU beside 16 GB "
                                                         "cards",
                                                "hf": hf("unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF"),
                                                "file": "{q}/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-{q}-0000{i}"
                                                        "-of-00003.gguf",
                                                "shards": 3, "download_gb": 58.9, "ram_gb": 54, "arena_gb": 50.3,
                                                "vram_beside_gb": 10.5}},
                      "sha256": {"NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf":
                                 (22018326528, "91d63363e782c8a3d5f4be42816eba8e0d42dd2530bd07a7d0d2dc440599af75"),
                                 "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-UD-Q5_K_XL-00001-of-00003.gguf":
                                 (7876064, "5bb0b06d303ccc6d4d0c8c230d5afd70600bf557059a4ba40c338e9451cd3033"),
                                 "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-UD-Q5_K_XL-00002-of-00003.gguf":
                                 (49594839136, "f634851a11898c88a6e3a7f162e7dbb6f018def08220e6ada7180a835ca3ec81"),
                                 "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-UD-Q5_K_XL-00003-of-00003.gguf":
                                 (9309916864, "a847a83a82152143b09dc66a3e397a372bc2d15cbd633cc0ce74efc566068213"),
                                 "mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf":
                                 (1733692736, "0ee891eadc7bf0cae6289b249c74eb1ef420534351f6ef8b64d49f0ca6a926b0")}},
    # Nemotron 3.5 Bolt on llama.cpp's llama-server, opt-in beside the family above ("engine": "llama-server"; the args
    # are llama-server's): the same files, kept in that family's folder ("files_of": downloaded once), with a config and
    # model name of its own, so the two run side by side and a request switches between them by name.  It was Bolt's
    # only engine before strata-poly ran it.  On echo, 2x RTX 5060 Ti 16 GB, everything on the GPUs with --ctx-size
    # 168000 and these args: 57.6 tokens/s writing, 1,402 tokens/s reading an 11,331-token prompt with pictures, 95% and
    # 93% of the two cards' VRAM in use (2026-11-20).  vram_need_gb: what setup asks of the chosen cards together (a 32
    # GB card, or two 16 GB ones); ram_gb: setup's estimate (the file goes to VRAM; llama-server's prompt cache takes up
    # to 8 GiB of RAM).  Its tag does not begin with "nemotron-bolt-": a config's family is the one whose tag its name
    # begins with (choices_from_config()).
    "nemotron-bolt-llama": {"title": "Nemotron 3.5 Bolt (llama-server)", "by": "NVIDIA", "engine": "llama-server",
                            "files_of": "nemotron-bolt",
                            "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 40, "min_p": 0.05},
                            "about": "74B-A9B, reads pictures; Bolt's file on llama.cpp's llama-server instead of "
                                     "Puddle's engine: NVIDIA cards with 32 GB of VRAM together",
                            "tag": "nemotron-llama-bolt-", "name": "nemotron-3.5-bolt-llama", "vision": True,
                            "experimental": True, "echo_context": 168000,
                            "models": {"24GB": {"about": "NVIDIA's own 2-bit file (quantization-aware trained)",
                                                "download_gb": 22.0, "ram_gb": 16, "vram_need_gb": 30}}},
    # Nemotron 3.5 Super VL 120B-A12B (NVIDIA; quantized by Unsloth): Bolt's architecture (nemotron_h_moe: Mamba-2,
    # attention and LatentMoE layers) at its full size, 512 experts a layer of which 22 are used, and an MTP block.  It
    # runs on strata-poly (src/poly/nemotron_h.cpp); on echo its first choices were llama.cpp's at 177 of 177 steps
    # (tools/poly_parity.py, 2026-11-21).  Its MTP block guesses ahead (NemotronH::draft_step(), llama.cpp's graph_mtp):
    # through Puddle on echo, 2x RTX 5060 Ti 16 GB, 2026-11-22, Super wrote 22.2 / 20.9 / 23.6 / 21.8 tokens/s after
    # prompts of 27 / 2,520 / 26,460 / 96,159 tokens with it and 18.3 / 17.5 / 19.5 / 17.3 the same day without
    # (--no-draft), 85 to 92% of its guesses right, and read those prompts 0.6% slower (the block's rows); greedy, the
    # CLI wrote the same 256 tokens either way.  arena_gb: the file's routed experts with the block's,
    # 58,562,445,312 bytes (without them 55.6); there the engine's mapped file took 3.0 GiB more RAM with the block
    # (/proc RssFile after loading, 48.06 against 45.05 GiB).  row_bytes: the 8 attention layers' rows and the block's.
    # vram_beside_gb: echo, one RTX 5060 Ti 16 GB (15.93 GiB) at a 32K context with the block, 2,316 of 20,992 experts
    # in VRAM (6.0 GiB), the estimate's way (Bolt's).  vram_card_gb: what each further card takes beside its share of
    # the experts, where setup's 2.0 a card is too little: on both cards at 32K 6,170 experts (16.3 GiB), so the two
    # took 15.6 GiB beside the experts together (each card's engine has its own scratch, reserve and buffers).  With
    # these the estimate is echo's: 6.5 GB of experts on one card (6.4 measured), 17.5 on two (17.5).  Pictures:
    # strata-vision (llama.cpp 08c252e22's mtmd, nemotron_omni) at the encoder file's own limits (vision_max_tokens 0:
    # up to 3,407,872 pixels, as llama-server reads Bolt's).  vision_vram_gb: measured on echo at a 256K context, the
    # encoder warmed up at its largest picture before the engine started: 6,451 -> 5,735 experts in VRAM (16.6 -> 14.1
    # GiB).  Its first choices on a picture were llama.cpp's at 63 of 64 steps, the other a near tie (2026-11-21).
    # vram_beside_gb: from echo's log, 17.1 GiB of experts on 2x RTX 5060 Ti 16 GB at an 8K context, the estimate's way.
    "nemotron-super": {"title": "Nemotron 3.5 Super", "by": "NVIDIA; quantized by Unsloth",
                       "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 40, "min_p": 0.05},
                       "about": "120B-A12B, reads pictures; a 66 GB file, ~62 GB of RAM beside a 16 GB card; runs on 2x "
                                "RTX 5060 Ti 16 GB on the strata-poly engine",
                       "hf": hf("unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF"),
                       "file": "{q}/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-{q}-0000{i}-of-00003.gguf", "shards": 3,
                       "tag": "nemotron-super-", "name": "nemotron-3.5-super", "engine": "poly", "vision": True,
                       "experimental": True, "metal": False, "ran_on_gb": 16, "row_bytes": 9216,
                       "mmproj_hf": hf("unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF"), "mmproj": "mmproj-BF16.gguf",
                       "vision_gb": 1.7, "vision_vram_gb": 2.5, "vision_max_tokens": 0, "vram_card_gb": 5.7,
                       "contexts": [8192, 32768, 65536, 131072, 262144],
                       "license": "OpenMDW-1.1 (NVIDIA's): "
                                  "https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
                       "models": {"UD-Q3_K_XL": {"about": "Unsloth Dynamic: IQ4_XS / IQ3_XXS / IQ2_XS up and IQ4_NL / "
                                                          "Q2_0 down experts, 4.05 bits a weight",
                                                 "download_gb": 65.8, "ram_gb": 62, "arena_gb": 58.6,
                                                 "vram_beside_gb": 9.9}},
                       "sha256": {"NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q3_K_XL-00001-of-00003.gguf":
                                  (7874568, "b2a570b4fc2e2e0ef3657937d6d4689c5d9647125e5fe22e31426789cf4e02fa"),
                                  "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q3_K_XL-00002-of-00003.gguf":
                                  (49225290592, "94827cf22f98854a68127fd4a18cca136951d1181bba8c21a7d9dcd6d3c68674"),
                                  "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q3_K_XL-00003-of-00003.gguf":
                                  (16528375872, "c2245a2faa51dc1aabcb6b58b7c311bc06ebada10345057e106ac94fcfcef8ce"),
                                  "mmproj-BF16.gguf":
                                  (1733695840, "9c09381a5744d0ea9a30048c94922f6f25ee0d63842de9c45779eca4fb5924d9")}},
    "gemma4": {"title": "Gemma 4 26B-A4B", "by": "Google; quantized by Unsloth",
               "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
               "about": "on the strata-poly engine too; a 13 GB file, ~18 GB of RAM beside an 8 GB card; a smaller and a "
                        "larger size too",
               "hf": hf("unsloth/gemma-4-26B-A4B-it-GGUF"), "file": "gemma-4-26B-A4B-it-{q}.gguf", "shards": 1,
               "tag": "gemma4-", "name": "gemma-4-26b-a4b", "engine": "poly", "vision": True, "experimental": True,
               "windows": True, "ran_on_gb": 8,
               # the context menu's lengths (else POLY_CONTEXTS): its files' own length, gemma4.context_length =
               # 262144 (both repositories' headers, read 2026-11-21), as one more; not run at that length yet
               "contexts": [8192, 32768, 65536, 131072, 262144],
               "row_bytes": 28672, "mmproj_hf": hf("unsloth/gemma-4-26B-A4B-it-GGUF"), "mmproj": "mmproj-F16.gguf",
               "vision_gb": 1.2, "vision_vram_gb": 2.4, "vision_most": 1024,
               "models": {"UD-Q3_K_M": {"about": "3-bit experts (Unsloth Dynamic), and Google's draft model for it",
                                        "download_gb": 13.2, "ram_gb": 18, "arena_gb": 10.3,
                                        "vram_beside_gb": 5.8,
                                        "draft_model": "MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf"},
                          "UD-Q2_K_XL": {"about": "2-bit i-quant experts (Unsloth Dynamic): the least RAM; the same "
                                                  "draft model",
                                         "download_gb": 11.0, "ram_gb": 15, "arena_gb": 8.7,
                                         "vram_beside_gb": 4.6,
                                         "draft_model": "MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf"},
                          "UD-Q4_K_M": {"about": "4- and 5-bit experts (Unsloth Dynamic); the same draft model",
                                        "download_gb": 17.4, "ram_gb": 21, "arena_gb": 14.3,
                                        "vram_beside_gb": 5.3,
                                        "draft_model": "MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf"},
                          # Google's quantization-aware trained checkpoint, quantized by Unsloth; its own repository,
                          # with a draft model and an image encoder of the same names as the others but not the same
                          "QAT-UD-Q4_K_XL": {"about": "Google's QAT weights, 4-bit (Unsloth Dynamic), with their own "
                                                      "draft model",
                                             "hf": hf("unsloth/gemma-4-26B-A4B-it-qat-GGUF"),
                                             "file": "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
                                             "mmproj_hf": hf("unsloth/gemma-4-26B-A4B-it-qat-GGUF"),
                                             "download_gb": 14.7, "ram_gb": 19, "arena_gb": 12.8,
                                             "vram_beside_gb": 4.1,
                                             "draft_model": "MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf",
                                             "sha256": {"mtp-gemma-4-26B-A4B-it-Q8_0.gguf":
                                                        (461785600, "75f58e20c32281d1adc086c0569d53f147e8d81a7b8493cf4a73a028415f67af"),
                                                        "mmproj-F16.gguf":
                                                        (1193058784, "d00f211a7d4f7fb19bd9b75d8e9342eccffb5920b08fd9562167560fdfcc5dd1")}}},
               "sha256": {"gemma-4-26B-A4B-it-UD-Q3_K_M.gguf":
                          (12728497888, "811882f6df81c8ea204a5d382f196d6bd601245091786912d4c70df41cdfa052"),
                          "gemma-4-26B-A4B-it-UD-Q2_K_XL.gguf":
                          (10546934240, "2a1d26dfe6ea00a467940a5728316af6edb366bbdba950d65b85d232392fb658"),
                          "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf":
                          (16947541728, "f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f"),
                          "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf":
                          (14249047104, "a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891"),
                          "mtp-gemma-4-26B-A4B-it-Q8_0.gguf":
                          (461766816, "6326fb9f5e487aa8dcdd313a091e3c67724cb2a666ec3b7d2895b5b26d93ed1b"),
                          "mmproj-F16.gguf":
                          (1193058784, "418a6d8723067cd712235facbbc5cba6c8fbbd413fc1292d2aace5a027d5a42f")}},
}
# A family on another family's files ("files_of"): the same downloads at the same revision, checked against the same
# SHA-256, and kept in that family's folder (files_tag()), so they are downloaded once for both.  A size of its own
# (one the other family does not have) gives its own "hf", "file", "shards" and "sha256" and has its own folder.
for _d in FAMILIES.values():
    if _d.get("files_of"):
        for _k in ("hf", "file", "shards", "mmproj_hf", "mmproj", "vision_gb", "sha256", "license"):
            if _k in FAMILIES[_d["files_of"]]:
                _d.setdefault(_k, FAMILIES[_d["files_of"]][_k])
# the image encoder's defaults: the most image tokens a picture becomes, and the VRAM kept free beside it
VISION = {"gpu": {"max_tokens": 1024, "reserve_mib": 700},
          "cpu": {"max_tokens": 300, "reserve_mib": 700}}
VEXE = "strata-vision.exe" if WIN else "strata-vision"
# strata-poly, the engine of every model here, read from one GGUF (docs/BACKENDS.md), in engine-poly/.  It is always
# compiled here from this checkout's source (Puddle has no ready-made engines: the Strata and PolyStrata releases have
# no Laguna).  It was called strata-glm, after its first model, and an install from then has that name and folder in
# its run config (poly_config(), update_poly_engine()).
POLY_EXE = "strata-poly.exe" if WIN else "strata-poly"
# A model of strata-poly that reads pictures has its image encoder on the GPU from this much VRAM (GiB), and on the
# CPU below it: the encoder holds 1.6 to 2.4 GB there, which a smaller card needs for the model
POLY_VISION_GPU_GB = 15.5
ENGINE_POLY_DIR = "engine-poly"
OLD_POLY_EXE = "strata-glm.exe" if WIN else "strata-glm"
OLD_ENGINE_POLY_DIR = "engine-glm"
POLY_CONTEXTS = [8192, 32768, 65536, 131072]
GLM_ROW_BYTES = 28160           # VRAM per context token: a row in each of the 11 sparse attention layers' caches
GLM_VRAM_BESIDE_GB = 12         # what is not expert cache on a 32 GB card at 32K (weights, caches, buffers, reserve):
                                # the 3.0BIT file's; a size with other dense weights has its own "vram_beside_gb"
POLY_RAM_BESIDE_GB = 10          # RAM beside the experts in the file cache: the OS, the server, the engine's buffers
POLY_CONV_CACHE_MIB = 8192       # the most the engine's "auto" conversation cache takes (engine: auto_conversation_cache_mib)


# ------------------------------------------------------------------------------------------------ output
def say(msg=""):
    print(msg, flush=True)


def step(n, title):
    say()
    say(f"=== Step {n}: {title} ===")


def ok(msg):
    say(f"  [ok] {msg}")


def warn(msg):
    say(f"  [!]  {msg}")


def fail(msg, hint=None):
    say(f"\n  [X]  {msg}")
    if hint:
        say(f"       {hint}")
    say("\nSetup stopped. Fix the item above and run it again - everything already done is kept and skipped.")
    sys.exit(1)


def flush_typed_ahead() -> None:
    """#841: keys pressed while setup was busy (a download, a build) are not answers to the next question: Enter pressed
    to "wake" a slow step answered "Go on anyway?" with its default.  Dropped before a question is asked, on a terminal
    only (a pipe or a test keeps its input)."""
    try:
        if not sys.stdin.isatty():
            return
        if WIN:
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getwch()
        else:
            import termios
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except (ImportError, OSError, ValueError, AttributeError):
        pass


def ask(question, choices, default, yes):
    if yes:
        return default
    flush_typed_ahead()
    while True:
        try:
            a = input(f"{question} [{default}]: ").strip()
        except EOFError:
            fail("input ended before a setup answer was received",
                 "run setup in a terminal, or pass --yes to accept the recommended answers")
        if not a:
            return default
        if a.lower() in [c.lower() for c in choices]:
            return next(c for c in choices if c.lower() == a.lower())
        say(f"  please answer one of: {', '.join(choices)}")


def run(cmd, cwd=None, env=None, check=True, quiet=False):
    say("  > " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env,
                       stdout=subprocess.PIPE if quiet else None, stderr=subprocess.STDOUT if quiet else None,
                       text=True)
    if check and r.returncode != 0:
        if quiet and r.stdout:
            say(r.stdout[-4000:])
        fail(f"command failed (exit {r.returncode}): {Path(str(cmd[0])).name}")
    return r


def out(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def done(path: Path) -> bool:
    """A step's finish mark: <path>.done exists (written only after the step completed)."""
    return path.with_name(path.name + ".done").exists()


def mark(path: Path, text=""):
    """Write the finish mark.  A folder that cannot be written (--gguf-dir on a read-only share, #570) only costs the
    mark: setup says so and goes on (the step is repeated on the next run), it does not stop."""
    try:
        path.with_name(path.name + ".done").write_text(text or time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
    except OSError as e:
        warn(f"the finish mark of {path.name} cannot be written ({e}): the next run does this step again")


# ------------------------------------------------------------------------------------------------ the PC
def _memory_status():
    """Windows' GlobalMemoryStatusEx: RAM, and the commit limit (ullTotalPageFile = RAM + page file)."""
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m


def container_ram_gb():
    """A container's memory limit in GB, None without one (Linux cgroup v2)."""
    try:
        v = open("/sys/fs/cgroup/memory.max").read().strip()
    except OSError:
        return None
    return int(v) / 2**30 if v.isdigit() else None


def ram_gb():
    if WIN:
        return _memory_status().ullTotalPhys / 2**30
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal"):
            # inside a container this is the host's RAM: a pod with 120 GB of a 1 TB machine reads 1 TB
            return min(int(line.split()[1]) * 1024 / 2**30, container_ram_gb() or float("inf"))
    return 0.0


_MM_KEY = r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management"


def _mm_multi_sz(name):
    """A REG_MULTI_SZ value of Memory Management, None when it cannot be read.  PagingFiles is what the Virtual memory
    dialog set; ExistingPageFiles is where the page files are now."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _MM_KEY) as k:
            v, _t = winreg.QueryValueEx(k, name)
    except (ImportError, OSError):
        return None
    return [x for x in v if x] if isinstance(v, list) else None


def parse_paging_file(line: str):
    """One PagingFiles entry ("C:\\pagefile.sys 16000 64000", "C:\\pagefile.sys 0 0", "?:\\pagefile.sys"): (drive,
    initial MB, maximum MB), both -1 when Windows manages the size and the drive '?' when it manages every drive;
    None for anything else."""
    s = line.strip()
    if len(s) < 2 or s[1] != ":":
        return None
    parts = s.split()
    try:
        initial, maximum = (int(parts[1]), int(parts[2])) if len(parts) >= 3 else (-1, -1)
    except ValueError:
        initial, maximum = -1, -1
    if maximum <= 0:
        initial, maximum = -1, -1
    return s[0].upper(), initial, maximum


PAGE_FILE_FIXED_MB = 65536   # the advice: a fixed size (initial = maximum) of 64 GB, the size #60's PC started with


def _drive_mb(drive: str):
    """(free MB, the page file's size now in MB) of a drive, None when it cannot be read.  The page file is locked:
    its size comes from the directory entry."""
    root = f"{drive}:\\"
    try:
        du = shutil.disk_usage(root)
    except OSError:
        return None
    current = 0
    try:
        with os.scandir(root) as it:
            for e in it:
                if e.name.lower() == "pagefile.sys":
                    current = e.stat(follow_symlinks=False).st_size
                    break
    except OSError:
        pass
    return du.free / 2**20, current / 2**20


def page_file_setting():
    """The page files Windows is set to use (PagingFiles, what the Virtual memory dialog writes): [(drive, initial MB,
    maximum MB, MB now, the drive's free MB)], the sizes -1 for a system-managed file; "?:" (every drive automatic) is
    the drive ExistingPageFiles names.  None off Windows or when the setting cannot be read."""
    if not WIN:
        return None
    lines = _mm_multi_sz("PagingFiles")
    if lines is None:
        return None
    auto = []                                          # "?:\pagefile.sys": where Windows keeps it now, else C:
    for s in _mm_multi_sz("ExistingPageFiles") or []:
        i = s.find(":")
        if i > 0:
            auto.append(s[i - 1].upper())
    auto = auto or [(os.environ.get("SystemDrive") or "C:")[0].upper()]
    files = []
    for line in lines:
        e = parse_paging_file(line)
        if e is None:
            continue
        drive, initial, maximum = e
        for d in (auto if drive == "?" else [drive]):
            free, now = _drive_mb(d) or (0.0, 0.0)
            files.append((d, initial, maximum, now, free))
    return files


def page_file_grows(files) -> bool:
    """Whether Windows grows one of these page files on demand: system-managed, or an initial size below the maximum."""
    return any(maximum <= 0 or initial < maximum for _d, initial, maximum, _now, _free in files or [])


def page_file_gb():
    """The page files' size (GB) that is there for sure on Windows, None elsewhere.  The graphics card's memory needs
    room there too: under Windows' driver model every allocation on the card is also charged to the commit (RAM + page
    file), so with the page file off or tiny the engine cannot use the free VRAM (issue #60).  What counts is the size
    the files have now (the commit limit minus RAM), or a file's configured initial size when that is larger (within
    its drive's free space) - never what a file may grow to: a page file Windows grows on demand may not grow in time
    while the card's memory is charged (#60: "System managed" and 4096-32768 MB still failed, a fixed 64 GB worked)."""
    if not WIN:
        return None
    m = _memory_status()
    mb = max(0.0, (m.ullTotalPageFile - m.ullTotalPhys) / 2**20)
    files = page_file_setting()
    if files:
        mb = max(mb, sum(max(now, min(max(initial, 0), now + free)) for _d, initial, _m, now, free in files))
    return mb / 1024


def page_file_advice(pf: float, files) -> str:
    """Step 1's warning for a page file below 4 GB (issue #60): a fixed size, and why when the setting (`files`,
    page_file_setting) has Windows grow it on demand."""
    grows = ("It is set to grow on demand (\"System managed\" or an initial size below the maximum), and a growing "
             "page file may not grow in time for the card's memory (#60: \"System managed\" and 4096-32768 MB still "
             "failed, a fixed 64 GB worked). ") if page_file_grows(files) else ""
    return (f"Windows' page file is {pf:.1f} GB: the graphics card's memory needs room there too (issue #60), so the "
            f"model may not start or may use less VRAM. {grows}Set a fixed size: System > About > Advanced system "
            "settings > Performance > Advanced > Virtual memory > Change, untick \"Automatically manage\", Custom size "
            f"with initial and maximum size {PAGE_FILE_FIXED_MB} MB, Set, then restart Windows")


def cpu_cores():
    """#642: (performance cores, efficiency cores) of a hybrid CPU (Intel 12th gen+, AMD Zen 5 + Zen 5c), counted
    as the engine's pool counts them (detect_cpu_topology: physical cores, by Windows' EfficiencyClass or Linux's
    cpu_capacity); None on a CPU whose cores are all alike, or when the OS does not say."""
    classes = []                                       # one entry per physical core: its efficiency/capacity class
    try:
        if WIN:
            k32 = ctypes.windll.kernel32
            n = ctypes.c_ulong(0)
            k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(n))   # RelationProcessorCore: the size
            if not n.value:
                return None
            buf = ctypes.create_string_buffer(n.value)
            if not k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(n)):
                return None
            raw, at = buf.raw[:n.value], 0
            while at + 10 <= len(raw):   # SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: Relationship, Size, then
                rel, size = struct.unpack_from("<II", raw, at)               # PROCESSOR_RELATIONSHIP (Flags,
                if size <= 0:                                                # EfficiencyClass, ...)
                    break
                if rel == 0:
                    classes.append(raw[at + 9])
                at += size
        else:
            classes = linux_core_classes()
            if classes is None:
                return None
    except (OSError, ValueError, AttributeError):
        return None
    if not classes or max(classes) == min(classes):
        return None
    p = sum(1 for c in classes if c == max(classes))
    return p, len(classes) - p


def linux_core_classes(sys_root: str = "/sys"):
    """One class per physical core on Linux: 1 for a P-core, 0 for an E-core; None when the kernel does not say.
    #798: Intel's hybrid PMU lists (cpu_core/cpus, cpu_atom/cpus) when they exist; else cpu_capacity, where an E-core is
    one under 90% of the largest - Turbo Boost Max 3.0 gives the "favored" P-cores a slightly higher capacity (1024
    against 1012), so "below the maximum" counted 2 of a Core Ultra 7 270K Plus's 8 P-cores."""
    pmu = linux_hybrid_pmu(sys_root + "/devices")
    seen = {}
    for cpu in sorted((Path(sys_root) / "devices" / "system" / "cpu").glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        cap = cpu / "cpu_capacity"
        pkg, core = cpu / "topology" / "physical_package_id", cpu / "topology" / "core_id"
        if not cap.exists() and pmu is None:
            return None
        key = (pkg.read_text(encoding="utf-8").strip(), core.read_text(encoding="utf-8").strip()) \
            if pkg.exists() and core.exists() else cpu.name
        if pmu is not None:
            seen.setdefault(key, 0 if int(cpu.name[3:]) in pmu[1] else 1)
        else:
            seen.setdefault(key, int(cap.read_text(encoding="utf-8").strip()))
    classes = list(seen.values())
    if pmu is None:
        top = max(classes, default=0)
        classes = [1 if c * 10 >= top * 9 else 0 for c in classes]
    return classes


def cpulist(text: str) -> set:
    """A kernel cpulist ("0-7,16") as a set of CPU numbers."""
    out = set()
    for part in text.strip().split(","):
        lo, _, hi = part.strip().partition("-")
        if lo.isdigit():
            out.update(range(int(lo), int(hi or lo) + 1))
    return out


def linux_hybrid_pmu(root: str = "/sys/devices"):
    """#798: (P-core CPUs, E-core CPUs) from Intel's hybrid PMU lists (cpu_core/cpus, cpu_atom/cpus), or None on a CPU
    that has no such lists."""
    try:
        p = cpulist((Path(root) / "cpu_core" / "cpus").read_text(encoding="utf-8"))
        e = cpulist((Path(root) / "cpu_atom" / "cpus").read_text(encoding="utf-8"))
    except OSError:
        return None
    return (p, e) if p and e else None


def hybrid_pool_workers(cores) -> int | None:
    """#642 (Hardin22's measurement): on a hybrid CPU the expert pool runs best on the P-cores but the host loop's one
    plus HALF of the E-cores - an E-core runs the expert kernels ~2.2x slower and each layer waits for its slowest
    part (i9-14900KF, 8P + 16E: 15 workers decoded 165 / 116 tok/s against 106 / 84 with all 23).  Only on a CPU with
    more E-cores than P-cores: on an i7-13700KF (8P + 8E, docs/AMD_HIP.md's gfx1030 report) all 15 workers decoded
    38-42 tok/s against 36 with 8, so there the engine's own count stays.  None: the engine's own default (one worker
    per physical core but the host's) stays."""
    if not cores:
        return None
    p, e = cores
    if e <= p:
        return None
    return max(1, p - 1 + e // 2)


def cpu_info():
    """(name, avx2, avx512): avx512 means everything Puddle's fast AVX-512 kernels use (F, BW, VL, VNNI, VBMI),
    the same test the engine makes (cpu_avx512_ok), not just AVX-512F."""
    name, avx2, avx512 = platform.processor() or "unknown CPU", False, False
    if WIN:
        pf = ctypes.windll.kernel32.IsProcessorFeaturePresent
        avx2 = bool(pf(40)) or _cpuid_avx2()      # PF_AVX2_INSTRUCTIONS_AVAILABLE, else the CPU itself (#159)
        n = out(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]).strip()
        name = n or name
        avx512 = bool(pf(41)) and _cpuid_avx512_full()
    else:
        try:
            txt = open("/proc/cpuinfo").read()
            flags = set(re.search(r"^flags\s*:\s*(.*)$", txt, re.M).group(1).split())
            avx2 = "avx2" in flags
            avx512 = {"avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512vbmi"} <= flags
            m = re.search(r"^model name\s*:\s*(.*)$", txt, re.M)
            name = m.group(1) if m else name
        except OSError:
            pass
    return name, avx2, avx512


def _cpuid_floor() -> str:
    """Below AVX2 (Windows): "avx" when the CPU has AVX and the OS saves the YMM registers, "sse4.2" with SSE4.2 and
    POPCNT, else ""."""
    try:
        regs = (ctypes.c_uint32 * 4)()
        _run_stub(bytes([0x53, 0x49, 0x89, 0xC8, 0x89, 0xD0, 0x31, 0xC9, 0x0F, 0xA2,      # push rbx; r8=rcx; eax=edx; ecx=0; cpuid
                         0x41, 0x89, 0x00, 0x41, 0x89, 0x58, 0x04, 0x41, 0x89, 0x48, 0x08,  # [r8]=eax, [r8+4]=ebx, [r8+8]=ecx
                         0x41, 0x89, 0x50, 0x0C, 0x5B, 0xC3]),                             # [r8+12]=edx; pop rbx
                  ctypes.addressof(regs), 1)
        ecx1 = regs[2]
        if (ecx1 >> 27) & 1 and (ecx1 >> 28) & 1:                     # OSXSAVE, AVX
            xcr0 = (ctypes.c_uint32 * 2)()
            _run_stub(bytes([0x49, 0x89, 0xC8, 0x31, 0xC9, 0x0F, 0x01, 0xD0,                    # r8=rcx; ecx=0; xgetbv
                             0x41, 0x89, 0x00, 0x41, 0x89, 0x50, 0x04, 0xC3]), ctypes.addressof(xcr0))
            if xcr0[0] & 6 == 6:
                return "avx"
        return "sse4.2" if (ecx1 >> 20) & 1 and (ecx1 >> 23) & 1 else ""
    except Exception:
        return ""


def cpu_floor(avx2: bool) -> str:
    """The experimental older-CPU build this PC needs (#394 #595 #623): "" with AVX2 (the normal engine), "avx" (Sandy /
    Ivy Bridge, AMD Bulldozer), "none" (SSE4.2 + POPCNT: Nehalem, Westmere), or "unsupported".  STRATA_ISA_FLOOR=avx|
    none asks for that build on any PC (testing it on a newer one)."""
    forced = os.environ.get("STRATA_ISA_FLOOR", "").strip().lower()
    if forced in ("avx", "none"):
        return forced
    if avx2:
        return ""
    if WIN:
        f = _cpuid_floor()
    else:
        try:
            txt = open("/proc/cpuinfo").read()
            flags = set(re.search(r"^flags\s*:\s*(.*)$", txt, re.M).group(1).split())
        except (OSError, AttributeError):
            flags = set()
        f = "avx" if "avx" in flags else "sse4.2" if {"sse4_2", "popcnt"} <= flags else ""
    return {"avx": "avx", "sse4.2": "none"}.get(f, "unsupported")


def _cpuid_avx512_full() -> bool:
    """Windows has no feature bit for VNNI / VBMI: ask the CPU (CPUID leaf 7) through a tiny machine-code stub."""
    try:
        code = bytes([0x53, 0x49, 0x89, 0xC8, 0xB8, 0x07, 0x00, 0x00, 0x00, 0x31, 0xC9, 0x0F, 0xA2,   # push rbx; r8=rcx; cpuid(7,0)
                      0x41, 0x89, 0x18, 0x41, 0x89, 0x48, 0x04, 0x5B, 0xC3])                   # [r8]=ebx,[r8+4]=ecx; pop rbx
        k32 = ctypes.windll.kernel32
        k32.VirtualAlloc.restype = ctypes.c_void_p
        buf = k32.VirtualAlloc(None, len(code), 0x3000, 0x40)
        if not buf:
            return False
        ctypes.memmove(buf, code, len(code))
        regs = (ctypes.c_uint32 * 2)()
        ctypes.CFUNCTYPE(None, ctypes.c_void_p)(buf)(ctypes.addressof(regs))
        ebx, ecx = regs[0], regs[1]
        need_ebx = (1 << 16) | (1 << 30) | (1 << 31)                   # F, BW, VL
        need_ecx = (1 << 1) | (1 << 11)                                # VBMI, VNNI
        return (ebx & need_ebx) == need_ebx and (ecx & need_ecx) == need_ecx
    except Exception:
        return False


def _run_stub(code: bytes, *args) -> None:
    """Runs a few bytes of x64 machine code (Windows calling convention: the arguments in rcx, rdx)."""
    k32 = ctypes.windll.kernel32
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualFree.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32)
    buf = k32.VirtualAlloc(None, len(code), 0x3000, 0x40)
    if not buf:
        raise OSError("VirtualAlloc failed")
    try:
        ctypes.memmove(buf, code, len(code))
        ctypes.CFUNCTYPE(None, *[ctypes.c_void_p] * len(args))(buf)(*args)
    finally:
        k32.VirtualFree(buf, 0, 0x8000)


def _cpuid_avx2() -> bool:
    """AVX2 asked from the CPU (CPUID leaf 7 EBX bit 5), with the OS saving the YMM registers (OSXSAVE + XCR0):
    Windows' IsProcessorFeaturePresent(PF_AVX2) says no on some PCs whose CPU has it (a Ryzen 9 3950X, #159)."""
    try:
        def cpuid(leaf):
            regs = (ctypes.c_uint32 * 4)()
            _run_stub(bytes([0x53, 0x49, 0x89, 0xC8, 0x89, 0xD0, 0x31, 0xC9, 0x0F, 0xA2,      # push rbx; r8=rcx; eax=edx; ecx=0; cpuid
                             0x41, 0x89, 0x00, 0x41, 0x89, 0x58, 0x04, 0x41, 0x89, 0x48, 0x08,  # [r8]=eax, [r8+4]=ebx, [r8+8]=ecx
                             0x41, 0x89, 0x50, 0x0C, 0x5B, 0xC3]),                             # [r8+12]=edx; pop rbx
                      ctypes.addressof(regs), leaf)
            return list(regs)
        if cpuid(0)[0] < 7:
            return False
        ecx1 = cpuid(1)[2]
        if not (ecx1 >> 27) & 1 or not (ecx1 >> 28) & 1:             # OSXSAVE, AVX
            return False
        xcr0 = (ctypes.c_uint32 * 2)()
        _run_stub(bytes([0x49, 0x89, 0xC8, 0x31, 0xC9, 0x0F, 0x01, 0xD0,                        # r8=rcx; ecx=0; xgetbv
                         0x41, 0x89, 0x00, 0x41, 0x89, 0x50, 0x04, 0xC3]), ctypes.addressof(xcr0))
        if xcr0[0] & 6 != 6:                                           # the OS saves XMM and YMM
            return False
        return bool((cpuid(7)[1] >> 5) & 1)
    except Exception:
        return False


def gpus():
    """Every NVIDIA GPU, numbered as nvidia-smi numbers them (by PCI bus, the order the engine is told to use)."""
    s = out(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap,driver_version",
             "--format=csv,noheader,nounits"])
    found = []
    for line in s.strip().splitlines():
        try:
            idx, name, mem, cc, drv = [x.strip() for x in line.split(",")]
            found.append({"index": int(idx), "name": name, "vram_gb": float(mem) / 1024.0, "arch": cc.replace(".", ""),
                          "driver": drv})
        except ValueError:
            continue
    return found


def pcie_link(index: int) -> dict | None:
    """The NVIDIA card's PCIe link: {"gen": the generation card and board both run (an idle card drops to a lower
    one, so the current generation is not asked), "gpu_gen", "host_gen", "width", "max_width"}; None when nvidia-smi
    does not say."""
    s = out(["nvidia-smi", "-i", str(index), "--query-gpu=pcie.link.gen.max,pcie.link.gen.gpumax,pcie.link.gen.hostmax,"
             "pcie.link.width.current,pcie.link.width.max", "--format=csv,noheader,nounits"])
    try:
        gen, gpu_gen, host_gen, width, max_width = (int(x.strip()) for x in s.strip().splitlines()[0].split(","))
    except (ValueError, IndexError):
        return None
    return {"gen": gen, "gpu_gen": gpu_gen, "host_gen": host_gen, "width": width, "max_width": max_width}


PCIE_GBPS = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.97, 5: 3.94}    # GB/s per lane, each direction


def pcie_lines(link: dict) -> tuple[str, str | None]:
    """(the ok line, a warning or None) for step 1.  The copy rate the engine measures is about 75% of the link's."""
    lim = []
    if link["gpu_gen"] > link["gen"]:
        lim.append(f"the card supports {link['gpu_gen']}.0, the board {link['host_gen']}.0")
    line = f"PCIe: {link['gen']}.0 x{link['width']}" + (f" ({'; '.join(lim)})" if lim else "")
    rate = PCIE_GBPS.get(link["gen"], 0) * link["width"]
    if 0 < rate < 12:
        line += (f" - up to ~{rate:.0f} GB/s to the GPU: long prompts are read slower than on a PCIe 4.0 x16 PC (their "
                 "experts are copied over it); the engine measures the link at start for the output speed")
    warn_line = None
    if link["width"] < link["max_width"]:
        # (#912) the width is read now, and some cards and laptops narrow the link while idle: a hint, not a verdict
        warn_line = (f"the GPU's PCIe link reads {link['width']} of its {link['max_width']} lanes right now (some cards "
                     "narrow it when idle); if it stays narrow under load: is it in the right slot (the one wired "
                     "x16), fully seated, and not sharing lanes with an M.2 drive? (the BIOS can say)")
    return line, warn_line


GPU_PICK = None                                         # --gpu N (issue #51); None: the card with the most VRAM
SPLIT_MIN_VRAM_GB = 8                                   # a card sharing a model holds the dense weights and its own
                                                        # prompt buffers too (docs/MULTI_GPU.md)


def cc(g) -> str:
    return f"{g['arch'][:-1]}.{g['arch'][-1]}"


OLD_GPUS = None       # why Pascal / Volta cards are admitted in this run (old_gpus_opt_in), None: they are not


def experimental_sm60() -> bool:
    """#295: STRATA_EXPERIMENTAL_SM60=1 admits Pascal (6.x) and Volta (7.0) cards, run by the experimental CUDA 12
    engine (-DSTRATA_EXPERIMENTAL_SM60=ON).  So does naming such a card (--gpu N / --gpus), --cuda 12, or a PC that
    has no newer card (old_gpus_opt_in)."""
    return os.environ.get("STRATA_EXPERIMENTAL_SM60", "").strip() == "1" or OLD_GPUS is not None


def sm60_card(arch) -> bool:
    return 60 <= int(arch) <= 70


def old_gpus_opt_in(found, named=(), cuda=None, other=False):
    """Why this run may use Pascal / Volta cards (the experimental CUDA 12 engine), or None.  The cards are an opt-in:
    the user named one (`named`: --gpu / --gpus), asked for --cuda 12, set STRATA_EXPERIMENTAL_SM60=1, or the PC has
    no card the ready-made engine runs on and no supported AMD card (`other`; it used to stop there).  A PC with a
    newer card keeps recommending it."""
    old = [g for g in found if sm60_card(g["arch"])]
    if not old:
        return None
    if os.environ.get("STRATA_EXPERIMENTAL_SM60", "").strip() == "1":
        return "STRATA_EXPERIMENTAL_SM60=1"
    if str(cuda) == "12":
        return "--cuda 12"
    picked = [g for g in old if g["index"] in set(named)]
    if picked:
        return "you chose " + ", ".join(f"GPU {g['index']} ({g['name']})" for g in picked)
    if not other and not any(int(g["arch"]) >= CUDA13_MIN_ARCH for g in found):
        return "it is the only kind of NVIDIA GPU in this PC"
    return None


def named_gpus(gpu, gpus) -> list:
    """The card numbers --gpu / --gpus name (an unreadable value: none; parse_gpus says what is wrong later)."""
    try:
        if gpus and str(gpus).strip().lower() != "all":
            return [int(x) for x in str(gpus).split(",") if x.strip()]
        return [int(gpu)] if gpu is not None else []
    except ValueError:
        return []


def cuda_choice(archs, cuda=None, poly=False):
    """The CUDA toolkit of one model's engine: (12 or 13, why).  13 (the ready-made engine) unless a card is older
    than CUDA 13 supports (Pascal / Volta: CUDA 13 cannot compile for them) - one engine runs per model, so its oldest
    card decides.  `cuda` (--cuda 12|13) overrides it; setup recommends, it does not refuse (the caller warns)."""
    archs = sorted({int(x) for x in archs})
    old = [a for a in archs if a < CUDA13_MIN_ARCH]
    if str(cuda) == "13":
        return 13, ("--cuda 13 (as you chose)" + (f"; CUDA 13 has no code for sm_{old[0]}: the engine will not run "
                                                   "on that card" if old else ""))
    if str(cuda) == "12":                              # #220 is the Qwen engine's prompt path, not strata-poly's
        crashed = archs and archs[-1] >= 120 and not poly
        return 12, "--cuda 12 (as you chose" + ("; RTX 50 (sm_120) engines built with CUDA 12.8 crashed on long "
                                                  "prompts, #220" if crashed else "") + ")"
    if old:
        return 12, (f"sm_{old[0]} is older than CUDA 13 supports (it dropped Pascal and Volta): this model runs the "
                    "experimental CUDA 12 engine")
    return 13, None


def config_toolkit(cfg: dict) -> int:
    """12 when a model config runs strata-poly built with CUDA 12 ("cuda": 12), else 13."""
    return 12 if cfg.get("cuda") == 12 else 13


def gpu_problem(g, together=False):
    """Why Puddle cannot use this card, in plain words (None: it can)."""
    if int(g["arch"]) < 75 and not (sm60_card(g["arch"]) and experimental_sm60()):
        return (f"not supported - older than the RTX 20 series (compute capability {cc(g)}; Puddle needs 7.5 or "
                "newer" + ("; experimental: choose it with --gpu " + str(g["index"]) + " (the CUDA 12 engine, "
                          "docs/OLDER_GPUS.md)" if sm60_card(g["arch"]) else "") + ")")
    if together and g["vram_gb"] < SPLIT_MIN_VRAM_GB - 0.5:
        return (f"not supported together with other GPUs - {g['vram_gb']:.0f} GB of VRAM (a card sharing the model "
                f"needs {SPLIT_MIN_VRAM_GB} GB or more)")
    return None


def gpu_rank(g):
    """The order cards share a model in: the newest generation first (it gets the first layers and most of the
    work), then the most VRAM."""
    return (-int(g["arch"]), -round(g["vram_gb"]), g["index"])


def gpu_name(g) -> str:
    return f"GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB)"


def gpu_table(found) -> None:
    say("  Your NVIDIA GPUs:")
    for g in found:
        p = gpu_problem(g)
        say(f"    GPU {g['index']}: {g['name']}, {g['vram_gb']:.0f} GB VRAM - " + ("can be used" if p is None else p))


def together_ok(found) -> list:
    """The cards that can share one model, in the order they would (empty if fewer than two)."""
    ok_ = sorted([g for g in found if gpu_problem(g, together=True) is None], key=gpu_rank)
    return ok_ if len(ok_) >= 2 else []


def parse_gpus(text, found) -> list:
    """--gpus / --gpu with several: "0,2" or "all" (every card that can share the model)."""
    if str(text).strip().lower() == "all":
        sel = [g["index"] for g in together_ok(found)]
        if not sel:
            gpu_table(found)
            fail("--gpus all: this PC does not have two GPUs Puddle can use together")
        return sel
    try:
        sel = [int(x) for x in str(text).split(",") if x.strip()]
    except ValueError:
        fail(f"--gpus takes GPU numbers as nvidia-smi numbers them, e.g. --gpus 0,2 (or --gpus all), not {text!r}")
    if len(sel) < 2 or len(set(sel)) != len(sel):
        fail("--gpus takes two or more different GPUs, e.g. --gpus 0,2 (one GPU: --gpu 0)")
    return sel


def check_gpus(sel, found, what="", yes=False, named=False) -> None:
    """Stops with a plain message when a chosen card is missing or cannot be used, and says what can.  named: the user
    named these cards (--gpus 0,1, or a config that has them): a card that is only short of VRAM for sharing the model
    is then a risk to confirm, not a stop (the owner's rule; --yes with the named cards is the consent)."""
    together = len(sel) > 1
    for i in sel:
        g = next((x for x in found if x["index"] == i), None)
        p = "not found on this PC" if g is None else gpu_problem(g, together)
        if p is None:
            continue
        if named and g is not None and gpu_problem(g) is None:     # it runs Puddle; only its VRAM is small
            confirm_risk(f"GPU {i} ({g['name']}) has {g['vram_gb']:.0f} GB of VRAM: a card sharing the model needs "
                         f"{SPLIT_MIN_VRAM_GB} GB or more (it holds the dense weights of its layers and its own prompt "
                         "buffers), so the model may not start, or run slower than without it", True, yes,
                         f"GPU {i} ({g['name']}) {what}is not used together with other GPUs: {p}",
                         "leave it out of --gpus, or answer y to use it anyway", "  Use it anyway?")
            warn(f"GPU {i} ({g['name']}) is used together with the others, as you chose")
            continue
        say()
        gpu_table(found)
        can = together_ok(found)
        single = [x for x in found if gpu_problem(x) is None]
        ones = " or ".join(f"--gpu {x['index']}" for x in single)
        both = "--gpus " + ",".join(str(x["index"]) for x in can) if can else ""
        hint = ((f"use these together: {both}" + (f" (or one card: {ones})" if not together else "")) if can else
                f"use one card: {ones}" if single else "Puddle needs an NVIDIA RTX 20 series or newer card")
        fail(f"GPU {i}{'' if g is None else ' (' + g['name'] + ')'} {what}cannot be used: {p}", hint)


def start_gpus(text):
    """--gpus when starting an installed model: NVIDIA cards as nvidia-smi numbers them, or on a PC whose AMD cards
    are the ones Puddle can use, AMD cards as setup lists them ("all": every supported AMD card)."""
    if not text:
        return None
    found = gpus()
    if str(text).strip().lower() == "all" and not WIN and not together_ok(found):
        amd = amd_gpus()
        if len([g for g in amd if amd_problem(g) is None]) >= 2:
            return [g["index"] for g in amd_parse_gpus("all", amd)]
    if not found and not WIN:                          # #1594: no NVIDIA card, AMD ones: "1,0" names those, as setup lists them
        amd = amd_gpus()
        if amd:
            return [g["index"] for g in amd_parse_gpus(text, amd)]
    return parse_gpus(text, found)


def choose_gpus(a, found) -> list:
    """Which cards this install uses: --gpus / --gpu, or asked when two or more can share the model (the two best
    together recommended), else the supported card with the most VRAM.  Returns their numbers, the main one first."""
    if a.gpus:
        sel = parse_gpus(a.gpus, found)
        check_gpus(sel, found, yes=a.yes, named=str(a.gpus).strip().lower() != "all")
        return sel
    if a.gpu is not None:
        check_gpus([a.gpu], found)
        return [a.gpu]
    single = sorted([g for g in found if gpu_problem(g) is None], key=lambda x: (-round(x["vram_gb"]), x["index"]))
    if not single:
        gpu_table(found)
        fail("none of your GPUs can run Puddle", "it needs an NVIDIA RTX 20 series or newer (compute capability 7.5+)")
    can = together_ok(found)
    if not can:
        return [single[0]["index"]]
    say()
    say(f"  Puddle can run the model on one GPU, or share it across {'these' if len(can) > 2 else 'both'}: then each"
        " card holds the")
    say("  experts of its own layers, so together they hold about twice as many (details: docs/MULTI_GPU.md).")
    say("  A much slower extra card can also make it slower.")
    opts = [can[:2]] + ([can] if len(can) > 2 else []) + [[g] for g in single]
    rec = 1
    for i, o in enumerate(opts, 1):
        label = (" + ".join(gpu_name(g) for g in o) + " together") if len(o) > 1 else gpu_name(o[0]) + " only"
        say(f"  {i}) {label}" + ("   (recommended)" if i == rec else ""))
    for g in found:
        if gpu_problem(g, together=True) is not None:
            say(f"     (GPU {g['index']}, {g['name']}: {gpu_problem(g, together=True)})")
    pick = opts[int(ask("Which GPUs?", [str(i) for i in range(1, len(opts) + 1)], str(rec), a.yes or a.check)) - 1]
    return [g["index"] for g in pick]


def family_models(family: str) -> dict:
    """A family's sizes ("models"), each name with what the family says of it."""
    return FAMILIES[family]["models"]


def model_file(fam: dict, model: str, i: int) -> str:
    """Shard i's file name: the family's pattern, or the model's own (#621: UD-IQ4_XS has three shards, not four)."""
    return fam["models"].get(model, {}).get("file", fam["file"]).format(q=model, i=i)


def model_shards(fam: dict, model: str) -> int:
    return fam["models"].get(model, {}).get("shards", fam.get("shards", 2))


def files_tag(fam: dict, tag: str, model: str, custom=None) -> str:
    """The folder under models/ a size's files are kept in: its own tag's, or for a size of a family on another
    family's files ("files_of") that the other family has too, that family's, so both find the one download.  A file
    the user gave (`custom`) keeps its own tag."""
    other = FAMILIES.get(fam.get("files_of") or "", {})
    return other["tag"] + model if model in other.get("models", {}) and not custom else tag


def gpu_info(pick=None):
    """The GPU Puddle runs on: `pick` (nvidia-smi's number) if given, else the one with the most VRAM (ties: the
    lower number).  None when there is no NVIDIA GPU.  The dict also says how many there are ("count")."""
    found = gpus()
    if not found:
        return None
    pick = GPU_PICK if pick is None else pick
    if pick is not None:
        g = next((x for x in found if x["index"] == pick), None)
        if g is None:
            fail(f"there is no GPU {pick}: " + ", ".join(f"{x['index']} = {x['name']}" for x in found))
    else:
        g = max(found, key=lambda x: (round(x["vram_gb"]), -x["index"]))
    return {**g, "count": len(found)}


def find_nvcc(below=None):
    """The newest CUDA toolkit's nvcc and its (major, minor); with `below`, the newest older than that version.
    #601: STRATA_NVCC=<path to nvcc> is the only one considered (a newer toolkit beside it that cannot build on this
    PC - CUDA 12.9 with glibc 2.43 - is not taken instead)."""
    pick = os.environ.get("STRATA_NVCC")
    if pick:
        if not Path(pick).exists():
            warn(f"STRATA_NVCC={pick}: no such file; looking for a CUDA toolkit as usual")
        else:
            v = re.search(r"release (\d+)\.(\d+)", out([pick, "--version"]))
            ver = (int(v.group(1)), int(v.group(2))) if v else None
            if ver and below is not None and ver >= below:
                warn(f"STRATA_NVCC={pick} is CUDA {ver[0]}.{ver[1]}; this build needs one older than "
                     f"{below[0]}.{below[1]}")
                return (None, None)
            return (pick, ver) if ver else (None, None)
    cands = [shutil.which("nvcc")]
    for var in ("CUDA_PATH", "CUDA_HOME"):           # CUDA_HOME: Linux's usual name (#601)
        if os.environ.get(var):
            cands.append(str(Path(os.environ[var]) / "bin" / ("nvcc.exe" if WIN else "nvcc")))
    if WIN:
        base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if base.exists():
            cands += [str(p / "bin" / "nvcc.exe") for p in sorted(base.iterdir(), reverse=True)]
    else:
        cands += [str(p / "bin" / "nvcc") for p in sorted(Path("/usr/local").glob("cuda*"), reverse=True)]
        cands += [str(p / "bin" / "nvcc") for p in sorted(Path("/opt").glob("cuda*"), reverse=True)]   # Arch (#46)
    best = (None, None)
    for c in dict.fromkeys(cands):                     # every toolkit found; the newest wins
        if c and Path(c).exists():
            v = re.search(r"release (\d+)\.(\d+)", out([c, "--version"]))
            ver = (int(v.group(1)), int(v.group(2))) if v else None
            if ver and (below is None or ver < below) and (best[1] is None or ver > best[1]):
                best = (c, ver)
    return best


def find_vcvars(cuda_v=None):
    """Visual Studio's vcvars64.bat.  #985: CUDA 13.0-13.2 accept Visual Studio 2019 and 2022 only, so a newer one
    (2026 = version 18) is taken only with CUDA 13.3 or newer (`cuda_v`, the toolkit's (major, minor)), or when it
    has the 2022 compiler in it.

    The version range is CUDA's requirement alone, so a caller with no CUDA (`cuda_v is None`) asks vswhere for no
    range at all: the CPU image encoder is built by MSVC, which is happy with any version.  With the range applied to every caller, a machine whose only C++ tools are
    Visual Studio 2026 gets None back and nothing can be built there (#881).  STRATA_VCVARS=<vcvars64.bat> picks one
    by hand, for an install vswhere cannot use."""
    override = os.environ.get("STRATA_VCVARS")
    if override:
        v = Path(override)
        return v if v.exists() else None
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.exists():
        return None

    def newest(*versions):
        p = out([str(vswhere), "-latest", "-products", "*", *versions, "-requires",
                 "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"]).strip()
        v = Path(p) / "VC/Auxiliary/Build/vcvars64.bat" if p else None
        return v if v and v.exists() else None

    if cuda_v is None:
        return newest()
    # A Visual Studio the toolkit does not accept, installed next to one it does, must not be picked ("unsupported
    # Microsoft Visual Studio version").  A newer one alone will do when it has the 2022 compiler in it (the v143
    # toolset), which vcvars_toolset() asks it for.
    upper = "19.0" if tuple(cuda_v) >= (13, 3) else "18.0"
    v = newest("-version", f"[16.0,{upper})")
    if v is None:
        v = newest()
        if v is not None and not vcvars_toolset(v):
            return None
    return v


def vcvars_toolset(vcvars) -> str:
    """What vcvars64.bat of a Visual Studio newer than 2022 is called with to set up its 2022 compiler
    (" -vcvars_ver=14.44.35207"); "" for 2019 and 2022, whose own compiler is the one, and for a newer one that
    was installed without that toolset."""
    def version(name):
        try:
            return (Path(vcvars).parent / name).read_text().strip()
        except OSError:
            return ""

    own, v143 = version("Microsoft.VCToolsVersion.default.txt"), version("Microsoft.VCToolsVersion.v143.default.txt")
    m = re.match(r"14\.(\d+)\.", own)
    return f" -vcvars_ver={v143}" if m and int(m.group(1)) >= 50 and re.fullmatch(r"14\.[34]\d\.\d+", v143) else ""


def find_tool(name):
    """cmake or ninja: the one pip installed next to this Python, else one on PATH.  The system's own can be older
    than CMakeLists.txt takes: Ubuntu 22.04 has cmake 3.22, and the engine asks for 3.24.  pip's launcher names the
    Python it was installed with, so in a folder moved after setup it does not start ("Fatal error in launcher",
    issue 78); the same version installed again writes it for this folder."""
    file = name + (".exe" if WIN else "")
    for d in (Path(sys.executable).parent / "Scripts", Path(sys.executable).parent):
        if (d / file).exists():
            if not out([str(d / file), "--version"]).strip():
                say(f"  {file} does not start (was this folder moved?): installing it again")
                try:
                    import importlib.metadata as md
                    spec = f"{name}=={md.version(name)}"
                except Exception:
                    spec = name
                run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                     "--force-reinstall", "--no-deps", spec])
            return str(d / file)
    p = shutil.which(name)
    if p:
        return p
    c = Path.home() / ".local" / "bin" / file
    return str(c) if c.exists() else None


def free_gb(path):
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


# ------------------------------------------------------------------------------------------------ downloads
def drop_archive(z: Path) -> None:
    """An unpacked or refused engine archive and its .done mark go: a refused one kept them, and every later run
    reused it ("already downloaded") instead of the published one (PR #324)."""
    z.unlink(missing_ok=True)
    z.with_name(z.name + ".done").unlink(missing_ok=True)


def download(url, dst: Path, what=None):
    """Resumable HTTP(S) download with a progress line; `file://` and plain paths are copied (tests, mirrors).
    A finished file gets a <name>.done mark, so a later run skips it without asking the server."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and done(dst):
        ok(f"{what or dst.name} already downloaded")
        return
    if not url.startswith(("http://", "https://")):
        src = Path(url[7:] if url.startswith("file://") else url)
        if not src.exists():
            fail(f"not found: {src}")
        shutil.copyfile(src, dst)
        mark(dst)
        ok(f"{what or dst.name} copied")
        return
    ms = ms_file(url) if model_source() == "modelscope" else None
    if ms is not None:
        if reachable(ms_url(*ms), timeout=30):
            url = ms_url(*ms)
        else:
            warn(f"{what or dst.name}: ModelScope does not answer; downloading it from {hf_endpoint()}")
            ms = None
    part = dst.with_name(dst.name + ".part")
    total = 0
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "strata-setup"})
            total = int(urllib.request.urlopen(req, timeout=60).headers.get("Content-Length", 0))
            break
        except urllib.error.HTTPError as e:
            if e.code == 404 and hf_unpinned(url) != url:  # #214: the pinned revision is gone from the repository
                warn(f"{what or dst.name}: not at the pinned revision any more; downloading the repository's "
                     "current file")
                url = hf_unpinned(url)
                continue
            if attempt == 4:
                fail(f"cannot reach {url.split('/')[2]} ({e})",
                     "check your internet connection and run it again" + source_hint(url))
            time.sleep(5)
        except OSError as e:
            if attempt == 4:
                fail(f"cannot reach {url.split('/')[2]} ({e})",
                     "check your internet connection and run it again" + source_hint(url))
            time.sleep(5)
    if dst.exists() and total and dst.stat().st_size == total:    # finished by an older setup (no mark yet)
        mark(dst)
        ok(f"{what or dst.name} already downloaded")
        return
    have = part.stat().st_size if part.exists() else 0
    for attempt in range(30):
        if total and have >= total:                    # stopped after the last byte, before the rename: nothing to
            break                                      # ask for (a range past the end is a 416, retried 30 times)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "strata-setup", "Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:                     # the server ignored the range: start over
                    f.seek(0)
                    f.truncate()
                    have = 0
                last = 0.0
                while True:
                    b = r.read(8 << 20)
                    if not b:
                        break
                    f.write(b)
                    have += len(b)
                    if time.time() - last > 2:
                        last = time.time()
                        size = f"{have / 1e9:6.2f} / {total / 1e9:.2f} GB ({100 * have / total:.0f}%)" if total \
                            else f"{have / 1e6:7.1f} MB"
                        print(f"\r  {what or dst.name}: {size}   ", end="", flush=True)
            print()
            if not total or have >= total:
                break
        except OSError as e:
            print()
            warn(f"download interrupted ({e}); retrying in 10 s ...")
            time.sleep(10)
    if total and part.stat().st_size != total:
        fail(f"could not finish downloading {dst.name}: {part.stat().st_size:,} bytes on disk, the server says {total:,}",
             "check your internet connection and run it again (the download resumes where it stopped)")
    part.replace(dst)
    meta = ms_meta(*ms) if ms is not None else None
    if meta is not None and meta[1]:
        verify_sha256(dst, meta[0], meta[1])           # ModelScope's published hash; kept in the finish mark
    else:
        if ms is not None:
            warn(f"{what or dst.name}: ModelScope publishes no SHA-256 for it: the file is not verified")
        mark(dst)
    ok(f"{what or dst.name} downloaded")


def whole_shard(s: Path) -> bool:
    """A shard as long as its own tensor directory says (check_shards' test, without stopping setup)."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile
    try:
        g = GGUFFile(s)
        return s.stat().st_size >= g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors),
                                                     default=0)
    except (OSError, ValueError, struct.error):
        return False


SHARD_NAME = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def custom_gguf_tag(shards: list[Path] | None, model: str) -> str:
    """Name for the config, start script and data folders: with --custom-gguf the GGUF's base name (shard suffix and
    extension stripped) so it won't collide with existing installs; otherwise the size name."""
    return Path(SHARD_NAME.sub("", shards[0].name)).stem if shards else model


def gguf_dir_shards(folder: Path, fam: dict, model: str) -> list[Path]:
    """--gguf-dir's shards (#305): every -0000i-of-0000N file of the model, N read from the first shard's name as the
    engine does.  The published name first; else the one first shard in the folder whose name has
    the size in it (an upload split or named differently: -00001-of-00003, Unsloth's ...-00001-of-00004.gguf).  A
    missing shard is check_shards' error later, as before."""
    first = folder / model_file(fam, model, 1)
    if not first.exists():
        found = sorted(p for p in folder.glob("*-00001-of-*.gguf") if SHARD_NAME.search(p.name))
        mine = [p for p in found if model.lower() in p.name.lower()]
        pick = mine if mine else found
        if len(pick) == 1:
            first = pick[0]
    m = SHARD_NAME.search(first.name)
    total = int(m.group(2)) if m else 1
    if not m or total < 1:
        return [first]
    stem = first.name[:m.start()]
    return [first.with_name("%s-%05d-of-%05d.gguf" % (stem, i, total)) for i in range(1, total + 1)]


# #444: the quantization in a GGUF's name (Unsloth's UD-IQ3_XXS, a K-quant's Q2_K_XL, a GSQ-RCO IQ3_S, ...)
GGUF_QUANT = re.compile(r"(?<![A-Za-z0-9])((?:UD-)?(?:I?Q\d+(?:_[A-Za-z0-9]+)*|BF16|F16|F32))"
                        r"(?=-\d{5}-of-\d{5}\.gguf$|\.gguf$)", re.I)


def sha256_matches(s: Path, size: int, sha: str) -> bool:
    """Whether a shard is the pinned file: its size, then its SHA-256 (the ~5 minutes for 111 GB are spent once: a
    match is kept in its finish mark)."""
    m = s.with_name(s.name + ".done")
    if m.exists() and f"sha256 {sha}" in m.read_text(encoding="utf-8", errors="replace"):
        return True
    if (s.stat().st_size if s.exists() else -1) != size:
        return False
    say(f"  checking {s.name} (SHA-256, {size / 1e9:.1f} GB) ...")
    h = hashlib.sha256()
    with open(s, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
    if h.hexdigest() != sha:
        return False
    mark(s, f"sha256 {sha}")
    return True


def verify_sha256(s: Path, size: int, sha: str, own: bool = False) -> None:
    """A shard's size and SHA-256 against the pinned values (the Unsloth file).  A wrong file setup downloaded is
    deleted, so the next run downloads it again; one in the user's own folder (`own`: --gguf-dir) is left alone."""
    if sha256_matches(s, size, sha):
        return
    have = s.stat().st_size if s.exists() else -1
    if have != size:
        fail(f"{s.name} is {have:,} bytes, not {size:,}",
             "give --gguf-dir the file setup offers" if own else "delete it and run setup again (the download restarts)")
    if own:
        fail(f"{s} has the wrong SHA-256 (expected {sha}): not the file setup offers",
             "give --gguf-dir the file setup offers, or leave out --gguf-dir and setup downloads it")
    s.unlink(missing_ok=True)
    s.with_name(s.name + ".done").unlink(missing_ok=True)
    fail(f"{s.name} has the wrong SHA-256 (expected {sha}): deleted", "run setup again to download it again")


# general.architecture of each strata-poly family's GGUFs, the name Engine::open() (src/poly/engine.cpp) picks its
# code by; the families not listed are qwen35moe
POLY_GGUF_ARCHS = {"laguna-s": ("laguna",), "laguna-xs": ("laguna",), "nemotron-bolt": ("nemotron_h_moe",),
                   "nemotron-super": ("nemotron_h_moe",), "gemma4": ("gemma4",)}


def gguf_arch(s: Path) -> str:
    """A GGUF's general.architecture, from its header; "" when it cannot be read."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile
    try:
        return str(GGUFFile(s).metadata.get("general.architecture") or "")
    except (OSError, ValueError, struct.error):
        return ""


def mixed_gate_up(s: Path) -> str:
    """The first layer whose routed experts' gate and up are stored with different types, as "blk.16: IQ2_S and
    IQ3_XXS", which Engine::load() refuses (it computes the two as one); "" for none or a file it cannot read."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile
    try:
        types = {t.name: t.type_name for t in GGUFFile(s).tensors}
    except (OSError, ValueError, struct.error):
        return ""
    for name, gate in types.items():
        up = types.get(name.replace(".ffn_gate_exps.", ".ffn_up_exps.")) if ".ffn_gate_exps." in name else None
        if up and up != gate:
            return f"{name.split('.ffn_')[0]}: {gate} and {up}"
    return ""


def verify_poly_own(s: Path, size: int, sha: str, family: str) -> None:
    """--gguf-dir's file for a strata-poly family: the pinned file, or another GGUF of the family's architecture (a
    fine-tune, another quantization: docs/FAQ.md), which the engine reads as it is.  Nothing in that folder is
    deleted."""
    m = s.with_name(s.name + ".done")
    if m.exists() and "own file" in m.read_text(encoding="utf-8", errors="replace"):
        return
    if sha256_matches(s, size, sha):
        return
    want = POLY_GGUF_ARCHS.get(family, ("qwen35moe",))
    arch = gguf_arch(s)
    if arch not in want:
        fail(f"{s} is not the file setup offers, and it is a {arch or 'unreadable'!r} model, not {want[0]!r}",
             f"give --gguf-dir a GGUF of {FAMILIES[family]['title']}'s architecture")
    if not whole_shard(s):
        fail(f"{s} is shorter than its own tensor directory says: an incomplete copy", "copy the whole file again")
    mixed = mixed_gate_up(s)
    if mixed:
        fail(f"{s} stores its experts' gate and up with different types ({mixed}), which the strata-poly engine "
             "does not run: it computes the two as one", "use a quantization that keeps them alike, or one setup offers")
    warn(f"{s.name} is not the file setup offers ({s.stat().st_size:,} bytes, not {size:,}): your own {arch} GGUF, "
         "run as it is; setup's RAM and VRAM estimates are the offered file's")
    mark(s, "own file")


def verify_poly_download(s: Path, size: int, sha: str, family: str) -> None:
    """A strata-poly family's file in setup's own models folder: the pinned one (verify_sha256).  A whole GGUF of the
    family's architecture of another size there is the user's own file put where setup keeps its downloads (issue 54):
    setup says how to run it instead of telling them to delete it."""
    have = s.stat().st_size if s.exists() else -1
    if have != size and have > 0 and whole_shard(s) and gguf_arch(s) in POLY_GGUF_ARCHS.get(family, ("qwen35moe",)):
        fail(f"{s} is a whole {gguf_arch(s)} GGUF but not the file setup offers ({have:,} bytes, not {size:,})",
             "to run your own file, move it to a folder of its own and run setup with --gguf-dir <that folder>; "
             "to download setup's file instead, delete it and run setup again")
    verify_sha256(s, size, sha)


def check_shards(shards):
    """Every shard present and whole, or setup stops naming the file and the numbers.  Whole means as long as
    its own tensor directory says (the header is read, the data is not): a truncated copy (--gguf-dir, a .part
    renamed by hand, a download finished by an older setup) otherwise passes as a model file and the engine
    fails much later, at the first tensor that runs past the end."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile
    for s in shards:
        if not s.exists():
            fail(f"missing {s}")
        try:
            g = GGUFFile(s)
        except (ValueError, struct.error) as e:
            fail(f"{s.name} is not a whole GGUF shard ({e})", "delete it and run setup again")
        need = g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors), default=0)
        have = s.stat().st_size
        if not g.tensors:
            # #2346: a metadata-only shard (Unsloth's UD-Q5_K_XL 00001, 10,946,618 bytes) ends at its header: the
            # alignment padding before an empty data section is not in the file, so data_start can be past its end
            need = min(need, have)
        if have < need:
            fail(f"{s.name} is short: {have:,} of {need:,} bytes ({need - have:,} missing)",
                 "delete it and run setup again (or copy the whole file into --gguf-dir)")
        if len(g.tensors) == 1 and have - need >= 64:       # (a few bytes may be the last tensor's alignment padding)
            # #657: a shard that holds one tensor (the PLE table) is exactly as long as that tensor; extra bytes are a
            # damaged or wrong file that the engine would refuse at start ("PLE table size mismatch")
            fail(f"{s.name} is longer than its tensor: {have:,} bytes, {need:,} expected ({have - need:,} extra)",
                 f"delete {s.name} and its .done mark and run setup again")


def get_llama_cpp(vision=False, server=False):
    """llama.cpp at the pinned commit (ggml for the build, gguf-py for the tools, mtmd for images), as a zip: no git.
    `vision`: the image encoder's own newer checkout (VISION_LLAMA_CPP_COMMIT); the engines never use it.
    `server`: llama-server's own checkout (LLAMA_SERVER_COMMIT), for build_llama_server."""
    if vision and VISION_LLAMA_CPP_COMMIT == LLAMA_CPP_COMMIT:
        vision = False
    commit = LLAMA_SERVER_COMMIT if server else VISION_LLAMA_CPP_COMMIT if vision else LLAMA_CPP_COMMIT
    llama = ROOT / "third_party" / (f"llama.cpp-{commit[:7]}" if vision or server else "llama.cpp")
    # llama-server's build reads the UI's templates (an unpack from before they were kept has none: it is unpacked again)
    if (llama / "ggml" / "CMakeLists.txt").exists() and (llama / "gguf-py").is_dir() and \
            (not server or (llama / "ui" / "apps" / "chat" / "ui.h.in").is_file()):
        return llama
    z = ROOT / "third_party" / f"llama.cpp-{commit[:7]}.zip"
    download(LLAMA_SERVER_ZIP if server else VISION_LLAMA_CPP_ZIP if vision else LLAMA_CPP_ZIP, z, "llama.cpp source")
    tmp = ROOT / "third_party" / "_unpack"
    shutil.rmtree(tmp, ignore_errors=True)

    def wanted(m: str) -> bool:
        # llama.cpp's own web UI (tools/ui) is not used, and its deep paths passed Windows' 260-character limit in a
        # folder like Downloads\Puddle-main\Puddle-main (#206).  llama-server's checkout has it in ui/: only the files
        # of ui/apps/chat itself are kept, not its folders.  Built without the UI (LLAMA_BUILD_UI and
        # LLAMA_USE_PREBUILT_UI off), the build still reads that folder's CMakeLists.txt and the two templates of the
        # empty table it writes (ui.h.in, ui.cpp.in: scripts/ui-assets.cmake's emit_files)
        if "/tools/ui/" in m:
            return False
        rel = m.split("/", 1)[1] if "/" in m else ""     # the path below the zip's top folder
        chat = "ui/apps/chat/"
        return not server or not rel.startswith("ui/") or (rel.startswith(chat) and "/" not in rel[len(chat):])
    try:
        with zipfile.ZipFile(z) as f:
            f.extractall(tmp, [m for m in f.namelist() if wanted(m)])
    except zipfile.BadZipFile:
        # #1797: do not let the .done mark make every later run reuse a damaged source archive.
        try:
            drop_archive(z)
        except OSError as e:
            warn(f"cannot remove damaged {z.name} ({e}): delete it and its .done mark before running setup again")
        raise
    top = next(tmp.iterdir())
    shutil.rmtree(llama, ignore_errors=True)
    # PR #63: on Windows a rename can fail with PermissionError while an antivirus scanner still holds a file of the
    # fresh unpack; shutil.move falls back to copy-and-delete, and a few retries let the scanner finish.  The target
    # is `llama` itself - moving into its parent would keep the zip's `llama.cpp-<sha>` folder name.
    for attempt in range(5):
        try:
            shutil.move(str(top), str(llama))
            break
        except PermissionError:
            if attempt == 4:
                raise
            shutil.rmtree(llama, ignore_errors=True)   # a partial copy from the failed attempt
            time.sleep(2)
    shutil.rmtree(tmp, ignore_errors=True)
    z.unlink(missing_ok=True)
    z.with_name(z.name + ".done").unlink(missing_ok=True)
    return llama


def req_name(line: str) -> str:
    """The distribution name of a requirement line ("numpy==2.5.3; python_version >= '3.12'" -> "numpy")."""
    return re.split(r"[\s<>=!~;\[]", line.strip(), maxsplit=1)[0].lower().replace("_", "-")


def requirement_lines(path: Path | None = None) -> list[str]:
    """requirements.txt's requirements, comments and blank lines left out."""
    lines = []
    for raw in (path or REQUIREMENTS).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def _installed(name: str) -> bool:
    try:
        import importlib.metadata as md
        md.distribution(name)
        return True
    except Exception:
        return False


def pip_install(packages, what):
    """pip install into .venv, skipped when the same list was installed before.  An install from before the pinned
    requirements (#214) recorded bare names: those packages are kept as they are (nothing is reinstalled), and the
    pinned dependencies it already has count as installed."""
    stamp = Path(sys.prefix) / ".strata-pip.json"
    have = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else []
    bare = {p.lower() for p in have if req_name(p) == p.lower()}
    need = [p for p in packages if p not in have and req_name(p) not in bare
            and not (bare and "==" in p and _installed(req_name(p)))]
    if not need:
        ok(f"{what} already installed")
        return
    say(f"  Installing {what} ...")
    run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *need])
    stamp.write_text(json.dumps(sorted(set(have) | set(need)), indent=0))
    ok(f"{what} installed")


def cuda_lib_dirs(toolkit=13):
    """Where pip put NVIDIA's CUDA libraries (nvidia/cu13/bin/x86_64 on Windows, nvidia/cu13/lib on Linux).
    toolkit 12: the CUDA 12 wheels' cuBLAS and runtime, in two folders (nvidia/cublas/bin, nvidia/cuda_runtime/bin)."""
    if int(toolkit) == 12:
        patterns = ("cublas64_12.dll", "cudart64_12.dll") if WIN else ("libcublas.so.12*", "libcudart.so.12*")
    else:
        patterns = ("cublas64_13.dll",) if WIN else ("libcublas.so.13*",)
    dirs = []
    for sp in {Path(p) for p in sys.path if p.endswith("site-packages")}:
        for pattern in patterns:
            for hit in (sp / "nvidia").rglob(pattern) if (sp / "nvidia").is_dir() else []:
                if hit.parent not in dirs:
                    dirs.append(hit.parent)
    return [str(d) for d in dirs]


# ------------------------------------------------------------------------------------------------ AMD
# The RX 7900 XT / XTX (gfx1100) and the RX 9070 series / Radeon AI PRO R9700 (gfx1201) on Linux, through the HIP
# backend (docs/AMD_HIP.md); the RX 7800 XT / 7700 XT (gfx1101, #254) and the RX 9060 XT (gfx1200, #256) were run by
# their owners; the RX 6800 / 6900 series (gfx1030, #311) and the RX 6700 XT (gfx1031, #524) and the RX 6600 / 6600 XT / 6650 XT (gfx1032, one run reported) run but are unvalidated; the RX 5700 XT (gfx1010) has run on Windows only (PR 102).  There is no ready-made AMD engine: ROCm comes from AMD's TheRock Python wheels into .venv (no sudo;
# a system ROCm 7 in /opt/rocm is used when it has hipcc and hipBLAS) and the engine is compiled here for the cards.
# No images yet.
ROCM_INDEXES = {"gfx1100": "https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/",   # TheRock's wheels per GPU family
                "gfx1101": "https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/",
                "gfx1102": "https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/",
                "gfx1200": "https://rocm.nightlies.amd.com/v2/gfx120X-all/",
                "gfx1201": "https://rocm.nightlies.amd.com/v2/gfx120X-all/",
                "gfx1030": "https://rocm.nightlies.amd.com/v2/gfx103X-all/",
                "gfx1031": "https://rocm.nightlies.amd.com/v2/gfx103X-all/",
                "gfx1032": "https://rocm.nightlies.amd.com/v2/gfx103X-all/",
                "gfx1010": "https://rocm.nightlies.amd.com/v2/gfx101X-dgpu/",  # RDNA1: run on Windows only so far
                "gfx1151": "https://rocm.nightlies.amd.com/v2/gfx1151/",       # Strix Halo
                "gfx1150": "https://rocm.nightlies.amd.com/v2/gfx1150/",       # Strix Point: only with STRATA_EXPERIMENTAL_GFX1150=1
                "gfx1103": "https://rocm.nightlies.amd.com/v2/gfx110X-all/"}   # Radeon 780M: only with STRATA_EXPERIMENTAL_GFX1103=1
# The TheRock nightly indexes are pruned and move on, and each GPU family's index holds its own range (#1103: gfx103X-all
# starts at 7.13.0a20260422 and has no 7.10; #1267: the 7.10 wheel segfaults on a Strix Halo with kernel 7.2.8), so one
# version for every card cannot work.  The wheel is chosen per index (the family): a preferred version, tried first; when
# the index no longer offers it, the newest one of the same 7.x line, else of the same major, with a warning (rocm_pick).
# STRATA_ROCM_VERSION still forces one exact version (no lookup).
ROCM_VERSION_DEFAULT = "7.10.0a20251120"                 # what PolyStrata's HIP build was tested with (gfx120X, gfx110X)
ROCM_FAMILY_PINS = {"gfx103X-all": "7.13.0a20260515",    # #1103: 7.13.0a20260515 runs; the 7.14 nightlies time out
                    "gfx110X-all": "7.10.0a20251121",    # the experimental gfx1103: this index has no ...20251120
                    "gfx1151": "7.14.0a20260608"}        # #1267: 7.14.0a20260529 to 20260608 run on kernel 7.2.8; the 7.10 wheel segfaults
ROCM_VERSION_OVERRIDE = os.environ.get("STRATA_ROCM_VERSION") or None
ROCM_SYSTEM_MIN = (7, 0)       # an older system ROCm is passed over for the wheels (gfx1201 needs ROCm 6.4 or newer)
# Windows: the same ROCm from AMD's TheRock Windows wheels, into .rocm-win (a private venv; a package per card family,
# so the device wheels are asked for by arch), and strata-poly is compiled here for its card (build_engine_poly_hip_win).
# STRATA_ROCM_WIN_VERSION forces one exact version of those wheels, and so does STRATA_ROCM_VERSION (the name PolyStrata's
# Windows build script read); the index stays ROCM_WIN_INDEX.
ROCM_WIN_VERSION_DEFAULT = "10.2.0a20260930"               # what PolyStrata's Windows HIP build is tested with
ROCM_WIN_VERSION_OVERRIDE = os.environ.get("STRATA_ROCM_WIN_VERSION") or \
    os.environ.get("STRATA_ROCM_VERSION") or None
ROCM_WIN_VERSION = ROCM_WIN_VERSION_OVERRIDE or ROCM_WIN_VERSION_DEFAULT
ROCM_WIN_INDEX = "https://nightly.repo.amd.com/rocm/whl-next/"
ROCM_WIN_VENV = ".rocm-win"
# STRATA_EXPERIMENTAL_GFX1103=1 (opt-in, unsupported): the Radeon 780M / 760M / 740M iGPU (Ryzen 7040 / 8040, gfx1103) is taken
# as a unified-memory AMD card like Strix Halo; setup compiles strata-poly for it as for any AMD card.  Unset: unchanged.
GFX1103_OPT_IN = os.environ.get("STRATA_EXPERIMENTAL_GFX1103") == "1"
# STRATA_EXPERIMENTAL_GFX1150=1 (opt-in, unvalidated): the Radeon 890M / 880M iGPU (Ryzen AI 300, Strix Point, gfx1150)
# is taken as a unified-memory AMD card like Strix Halo; setup compiles strata-poly for it as for any AMD card.
# Unset: unchanged.
GFX1150_OPT_IN = os.environ.get("STRATA_EXPERIMENTAL_GFX1150") == "1"
AMD_ARCHS = ("gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201", "gfx1030", "gfx1031", "gfx1032", "gfx1010",
             "gfx1151") \
            + (("gfx1103",) if GFX1103_OPT_IN else ()) + (("gfx1150",) if GFX1150_OPT_IN else ())
# The Qwen engine's speed switches PolyStrata's setup wrote into a gfx1103 / gfx1150 install's "env".  Setup's own
# entries, so a setup run again for such an install leaves them out (SETUP_ENV).
OLD_GFX_SWITCHES = frozenset({
    "STRATA_GDN_HEAD", "STRATA_GDN_PP", "STRATA_GDN_CONVL2", "STRATA_GDN_NOY", "STRATA_CVEC_FUSE", "STRATA_PF_PAD",
    "STRATA_Q8_PACKED", "STRATA_Q6_PACKED", "STRATA_MMVF_ROWS", "STRATA_ATTN_LANECELL", "STRATA_EXPERT_V2",
    "STRATA_TSUM", "STRATA_LFUSE", "STRATA_GDN_SPLIT", "STRATA_QFUSE", "STRATA_PLE_BATCH", "STRATA_SH_STREAM"})
AMD_NAMES = {"gfx1100": "AMD Radeon RX 7900 series (gfx1100)",   # when sysfs has no product name
             "gfx1101": "AMD Radeon RX 7800 XT / 7700 XT (gfx1101)",
             "gfx1102": "AMD Radeon RX 7600 / 7600 XT (gfx1102)",
             "gfx1200": "AMD Radeon RX 9060 series (gfx1200)",
             "gfx1201": "AMD Radeon RX 9070 series / AI PRO R9700 (gfx1201)",
             "gfx1030": "AMD Radeon RX 6800 / 6900 series (gfx1030)",
             "gfx1031": "AMD Radeon RX 6700 XT series (gfx1031)",
             "gfx1032": "AMD Radeon RX 6600 / 6600 XT / 6650 XT (gfx1032)",
             "gfx1010": "AMD Radeon RX 5600 XT / 5700 / 5700 XT (gfx1010)",
             "gfx1151": "AMD Radeon 8060S / 8050S / 8040S (Ryzen AI Max, Strix Halo, gfx1151)",
             # integrated Radeons of other Ryzen families: named so they are not taken for Strix Halo (not supported)
             "gfx1150": "AMD Radeon 890M / 880M (Ryzen AI 300, Strix Point, gfx1150)",
             "gfx1152": "AMD Radeon 860M / 840M (Ryzen AI 300, Krackan Point, gfx1152)",
             "gfx1103": "AMD Radeon 780M / 760M / 740M (Ryzen 7040 / 8040, Phoenix / Hawk Point, gfx1103)"}
AMD_CARDS = ("the RX 7900 XT / XTX (gfx1100), RX 7800 XT / 7700 XT (gfx1101), RX 9060 XT (gfx1200) and "
             "RX 9070 / 9070 XT / Radeon AI PRO R9700 (gfx1201), and the RX 6800 / 6900 series (gfx1030) and RX 6700 XT "
             "(gfx1031, #524), the RX 6600 / 6600 XT / 6650 XT (gfx1032, one run reported), the RX 5600 XT / 5700 / 5700 XT (gfx1010, run on Windows only), and the RX 7600 / 7600 XT (gfx1102, one run reported, #942), all unvalidated, and the Ryzen AI Max \"Strix Halo\" APU (Radeon 8060S / 8050S / 8040S, "
             "gfx1151: experimental)")


def rocm_index(arch):
    return os.environ.get("STRATA_ROCM_INDEX") or ROCM_INDEXES[arch]


def rocm_family(index):
    """The family name of a TheRock index url (.../v2/gfx103X-all/ -> gfx103X-all)."""
    return index.rstrip("/").rsplit("/", 1)[-1]


def rocm_vkey(version):
    """Sort key of a TheRock version: 7.13.0a20260515 -> (7, 13, 0, 20260515); None when it is not one."""
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[a-z]+(\d+))?", version)
    return tuple(int(g or 0) for g in m.groups()) if m else None


def rocm_index_versions(html):
    """The versions of the `rocm` package that a TheRock index page (<index>/rocm/) lists, oldest first."""
    return sorted({m for m in re.findall(r"rocm-(\d+\.\d+\.\d+(?:[a-z]+\d+)?)\.(?:tar\.gz|whl)", html or "")
                   if rocm_vkey(m)}, key=rocm_vkey)


def rocm_pick(available, preferred):
    """(version, note) for an index that lists `available`: `preferred` when it is there, else the newest of the same
    7.x line, else the newest of the same major (the note says so); `preferred` unchanged when the list is empty (the
    index could not be read: pip then reports its own error) or nothing of that major exists (the note says so)."""
    if not available or preferred in available:
        return preferred, None
    want = rocm_vkey(preferred)
    for same, what in ((lambda k: k[:2] == want[:2], "line"), (lambda k: k[0] == want[0], "major")):
        hits = [v for v in available if same(rocm_vkey(v))]
        if hits:
            got = max(hits, key=rocm_vkey)
            return got, (f"ROCm {preferred} is not on this index any more: using {got}, the newest of the same {what} "
                         f"(set STRATA_ROCM_VERSION to choose another)")
    return preferred, (f"ROCm {preferred} is not on this index and nothing of its major version is: "
                       f"{available[0]} to {available[-1]} are (set STRATA_ROCM_VERSION to one of them)")


def rocm_wanted(index, listing=None):
    """(ROCm wheel version, warning or None) for a TheRock index (see ROCM_FAMILY_PINS).  `listing` is the index
    page's html (None: fetched from <index>/rocm/; a failed fetch keeps the preferred version)."""
    if ROCM_VERSION_OVERRIDE:
        version, note = ROCM_VERSION_OVERRIDE, None
    else:
        preferred = ROCM_FAMILY_PINS.get(rocm_family(index), ROCM_VERSION_DEFAULT)
        if listing is None:
            try:
                req = urllib.request.Request(index.rstrip("/") + "/rocm/", headers={"User-Agent": "strata-setup"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    listing = r.read().decode("utf-8", "replace")
            except (OSError, ValueError):
                listing = ""
        version, note = rocm_pick(rocm_index_versions(listing), preferred)
    if rocm_family(index) == "gfx1151" and (rocm_vkey(version) or (9, 0))[:2] < (7, 11):
        note = ((note + "; ") if note else "") + (
            "the ROCm 7.10 wheels segfault in the HSA runtime on a Strix Halo with a new kernel (7.2.8, #1267): "
            "the 7.14 line is the one that works")
    return version, note


# Strix Halo (Ryzen AI Max 380 / 385 / 390 / 395 + PRO: Radeon 8040S / 8050S / 8060S, RDNA 3.5, gfx1151) and the other
# integrated Radeons it must never be taken for.  pci.ids lists ONE device id for the whole Strix Halo family
# ("1586  Strix Halo [Radeon Graphics / Radeon 8050S Graphics / Radeon 8060S Graphics]"; the 8040S shares it: the
# three differ by their CU count, 40 / 32 / 16, not by id).  Aurora's real answers (tools/test_setup_strix_halo.py):
# KFD gfx_target_version 110501, vendor_id 4098, device_id 5510 (0x1586), simd_count 80.
STRIX_HALO_ARCH = "gfx1151"
STRIX_HALO_PCI_IDS = frozenset({0x1586})
AMD_IGPU_PCI = {0x1586: "gfx1151",                                    # Strix Halo
                0x150E: "gfx1150",                                    # Strix Point: Radeon 880M / 890M
                0x1114: "gfx1152", 0x1902: "gfx1152",                 # Krackan Point (Krackan2): Radeon 840M / 860M
                0x15BF: "gfx1103", 0x15C8: "gfx1103", 0x164F: "gfx1103",   # Phoenix: Radeon 740M / 760M / 780M
                0x1900: "gfx1103", 0x1901: "gfx1103"}                 # Hawk Point
STRIX_HALO_BY_SIMDS = {80: "8060S", 64: "8050S", 32: "8040S"}   # KFD simd_count (2 per CU): 40 / 32 / 16 CUs
UMA_OS_LEFT_GB = 6             # the share of the unified memory kept for the OS: what the engine's device_free_bytes() leaves


def gfx_arch_is(arch, name) -> bool:
    """include/strata/kernels/gfx_arch.hpp's gfx_arch_is: `arch` (gcnArchName, "gfx1151" or "gfx1151:sramecc-:xnack-")
    is exactly `name`, never a prefix: gfx1150 and gfx1152 are not gfx1151 (and gfx1151 is not gfx115)."""
    arch = str(arch or "")
    return arch == name or arch.startswith(name + ":")


def is_strix_halo(g) -> bool:
    return isinstance(g, dict) and gfx_arch_is(g.get("arch"), STRIX_HALO_ARCH)


def is_uma_apu(g) -> bool:
    """A unified-memory AMD APU whose "VRAM" is a BIOS carve-out plus shared system memory: Strix Halo (gfx1151), and
    the opt-in gfx1150 (Strix Point) / gfx1103 (Phoenix / Hawk Point) iGPUs.  Everything else is a discrete card."""
    arch = g.get("arch") if isinstance(g, dict) else None
    return (is_strix_halo(g)
            or (GFX1150_OPT_IN and gfx_arch_is(arch, "gfx1150"))
            or (GFX1103_OPT_IN and gfx_arch_is(arch, "gfx1103")))


def amd_apply_uma(g: dict, gtt_gb: float = 0.0, ram: float | None = None) -> dict:
    """Strix Halo is an APU with unified memory.  Its "dedicated VRAM" is a BIOS carve-out (512 MB to 96 GB; 2 GB on the
    maintainers' box) and the GPU also reaches shared system memory (Linux: the GTT pool, `mem_info_gtt_total`, half the
    RAM unless the kernel was told otherwise; Windows: "shared GPU memory", half the RAM).  The shared part IS the RAM
    the model's experts live in, so it is not extra room beside the RAM.  g gets: uma, dedicated_gb (the carve-out: the
    only part that adds to the RAM, which the low-RAM mode counts) and vram_gb = the memory the GPU can use in all
    (the carve-out plus the shared pool less what the OS keeps), which sizes the context and the parallel slots."""
    if g.get("uma") or not is_uma_apu(g):
        return g
    ram = ram_gb() if ram is None else ram
    carve = max(0.0, float(g.get("vram_gb") or 0.0))
    shared = gtt_gb if gtt_gb > 0 else ram / 2
    shared = max(0.0, min(shared, ram - UMA_OS_LEFT_GB))
    g.update({"uma": True, "dedicated_gb": carve, "shared_gb": shared, "vram_gb": carve + shared})
    return g


def amd_mem_text(g) -> str:
    """How setup says a card's memory: "24 GB VRAM", or for an APU its unified memory."""
    if g.get("uma"):
        return (f"unified memory: {g['dedicated_gb']:.1f} GB BIOS carve-out + {g['shared_gb']:.0f} GB shared with the "
                f"RAM, {g['vram_gb']:.0f} GB usable by the GPU")
    return f"{g['vram_gb']:.0f} GB VRAM"


def amd_device_access_problem(dev="/dev", access=os.access, listing=glob.glob) -> str | None:
    """Linux AMD: /dev/kfd and the render nodes must be readable and writable by this user (the render / video groups).
    None when they are, else a sentence for the user.  A missing /dev/kfd is not this problem (no driver / no ROCm)."""
    kfd = os.path.join(dev, "kfd")
    if not os.path.exists(kfd):
        return None
    bad = [kfd] if not access(kfd, os.R_OK | os.W_OK) else []
    bad += [n for n in sorted(listing(os.path.join(dev, "dri", "renderD*"))) if not access(n, os.R_OK | os.W_OK)]
    if not bad:
        return None
    return (f"this user cannot open {', '.join(bad)}: ROCm will find no GPU and the engine will fail with 'no ROCm-capable "
            "device' although nothing else holds the GPU. Add your user to the render and video groups: "
            "sudo usermod -aG render,video $USER, then log out and in again")


def gfx1103_notes(gpu, ram) -> list[str]:
    """What setup tells the owner of a Radeon 780M / 760M / 740M (Phoenix / Hawk Point, gfx1103, unified memory): never
    the Strix Halo text, which is about another chip (gfx1151).  Opt-in only (STRATA_EXPERIMENTAL_GFX1103=1)."""
    notes = [f"  Radeon 780M / 760M class (gfx1103, Ryzen 7040 / 8040): the CPU and the GPU share one memory pool ({ram:.0f} GB "
             f"seen by the OS + a {gpu.get('dedicated_gb', 0.0):.1f} GB BIOS carve-out), so the model's experts live in "
             "that pool and the expert cache is sized from the memory the OS can give back."]
    notes.append("  The engine is compiled here for gfx1103 (experimental opt-in, STRATA_EXPERIMENTAL_GFX1103=1, "
                 "unvalidated): see docs/AMD_HIP.md. This is not a Strix Halo.")
    if not WIN and gpu.get("shared_gb", 0) < 0.75 * ram - 1:
        notes.append(f"!the GPU can reach {gpu.get('shared_gb', 0):.0f} GB of shared memory (the GTT pool; the kernel's "
                     "default is about half of the RAM). The kernel option ttm.pages_limit raises it; setup changes no "
                     "host setting - a bigger model needs the room, a smaller one runs as it is")
    return notes


def gfx1150_notes(gpu, ram) -> list[str]:
    """What setup tells the owner of a Radeon 890M / 880M (Ryzen AI 300, Strix Point, gfx1150, unified memory): never
    the Strix Halo text, which is about another chip (gfx1151).  Opt-in only (STRATA_EXPERIMENTAL_GFX1150=1)."""
    notes = [f"  Radeon 890M / 880M class (gfx1150, Strix Point): the CPU and the GPU share one memory pool ({ram:.0f} GB "
             f"seen by the OS + a {gpu.get('dedicated_gb', 0.0):.1f} GB BIOS carve-out), so the model's experts live in "
             "that pool and the expert cache is sized from the memory the OS can give back."]
    notes.append("  The engine is compiled here for gfx1150 (experimental opt-in, STRATA_EXPERIMENTAL_GFX1150=1, "
                 "unvalidated: docs/AMD_HIP.md).")
    if not WIN and gpu.get("shared_gb", 0) < 0.75 * ram - 1:
        notes.append(f"!the GPU can reach {gpu.get('shared_gb', 0):.0f} GB of shared memory (the GTT pool; the kernel's "
                     "default is about half of the RAM). The kernel option ttm.pages_limit raises it; setup changes no "
                     "host setting - a bigger model needs the room, a smaller one runs as it is")
    return notes


def igpu_notes(gpu, ram) -> list[str]:
    """The unified-memory notes for exactly this chip: Strix Halo (gfx1151) or the gfx1150 / gfx1103 opt-in, nothing else."""
    if is_strix_halo(gpu):
        return strix_halo_notes(gpu, ram)
    if GFX1150_OPT_IN and gfx_arch_is(gpu.get("arch"), "gfx1150"):
        return gfx1150_notes(gpu, ram)
    if gfx_arch_is(gpu.get("arch"), "gfx1103"):
        return gfx1103_notes(gpu, ram)
    return []


def strix_halo_notes(gpu, ram) -> list[str]:
    """What setup tells a Strix Halo owner (a leading "!" = a warning): the unified memory, the build, the guide, and the
    memory settings that give the GPU room.  Nothing is changed or refused: recommend, never force."""
    notes = [f"  Strix Halo (gfx1151, Ryzen AI Max): the CPU and the GPU share one memory pool ({ram:.0f} GB seen by the "
             f"OS + a {gpu.get('dedicated_gb', 0.0):.1f} GB BIOS carve-out), so the model's experts live in that pool "
             "and the expert cache is sized from the memory the OS can give back, not from the carve-out."]
    notes.append("  The engine is compiled here for gfx1151 (experimental): see docs/AMD_HIP.md.")
    if not WIN and gpu.get("shared_gb", 0) < 0.75 * ram - 1:
        notes.append(f"!the GPU can reach {gpu.get('shared_gb', 0):.0f} GB of shared memory (the GTT pool; the kernel's "
                     "default is about half of the RAM). The kernel options ttm.pages_limit and ttm.page_pool_size "
                     "raise it; setup changes no host setting - a bigger model needs the room, a smaller one runs as it is")
    if gpu.get("dedicated_gb", 0.0) > 16:
        notes.append(f"!the BIOS carve-out is {gpu['dedicated_gb']:.0f} GB: that memory is taken from the OS, while the GPU "
                     "reaches the shared memory anyway. A small carve-out (the BIOS's default or 512 MB - 2 GB) leaves "
                     "the RAM to the model")
    return notes


def amd_rank(g):
    """The order setup picks a card in, best first: a discrete card before an APU (its own GDDR is faster than the
    shared LPDDR5X), then the most memory.  A Strix Halo beside a Radeon RX / PRO card is therefore left to --gpu N."""
    return (bool(g.get("uma")), -round(g["vram_gb"]), g["index"])


def amd_pci_devices(sysfs="/sys") -> list[dict]:
    """The AMD (vendor 0x1002) display devices the kernel lists in /sys/class/drm/card*/device, as [{"pci_id",
    "vram_gb", "gtt_gb", "arch"}] with the arch AMD_IGPU_PCI knows ("" when it does not).  The KFD topology answers the
    same without ROCm; this one only names a card the topology does not list (a kernel without /dev/kfd)."""
    found = []
    base = Path(sysfs) / "class/drm"
    if not base.is_dir():
        return found
    for card in sorted(base.glob("card[0-9]*")):
        if not card.name[4:].isdigit():
            continue
        try:
            dev = card / "device"
            if int((dev / "vendor").read_text().strip(), 16) != 0x1002:
                continue
            did = int((dev / "device").read_text().strip(), 16)
        except (OSError, ValueError):
            continue
        row = {"pci_id": did, "arch": AMD_IGPU_PCI.get(did, ""), "vram_gb": 0.0, "gtt_gb": 0.0}
        for key, f in (("vram_gb", "mem_info_vram_total"), ("gtt_gb", "mem_info_gtt_total")):
            try:
                row[key] = int((dev / f).read_text()) / 2 ** 30
            except (OSError, ValueError):
                pass
        found.append(row)
    return found


def amd_gpus(sysfs="/sys"):
    """AMD GPUs from the kernel's KFD topology (the amdgpu driver; no ROCm needed), numbered as HIP numbers them:
    the GPU nodes in order, the CPU nodes skipped.  Integrated GPUs are listed too (not supported).
    sysfs: the tree to read (tools/test_setup_amd.py passes a mocked one).  Windows: amd_gpus_windows."""
    if WIN:
        return amd_gpus_windows()
    base = Path(sysfs) / "class/kfd/kfd/topology/nodes"
    found = []
    if not base.is_dir():
        return found
    for node in sorted((p for p in base.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)):
        try:
            props = {}
            for line in (node / "properties").read_text().splitlines():
                k, _, v = line.partition(" ")
                props[k] = v.strip()
            ver = int(props.get("gfx_target_version") or 0)
            if ver == 0 or int(props.get("simd_count") or 0) == 0:
                continue
        except (OSError, ValueError):
            continue
        arch = f"gfx{ver // 10000}{(ver // 100) % 100:x}{ver % 100:x}"
        dev = Path(sysfs) / f"class/drm/renderD{props.get('drm_render_minor', '')}/device"
        try:
            vram = int((dev / "mem_info_vram_total").read_text()) / 2 ** 30
        except (OSError, ValueError):
            vram = 0.0
        try:
            name = (dev / "product_name").read_text().strip() or f"AMD Radeon ({arch})"
        except OSError:
            name = f"AMD Radeon ({arch})"
        if name == f"AMD Radeon ({arch})" and arch in AMD_NAMES:
            name = AMD_NAMES[arch]
            if gfx_arch_is(arch, STRIX_HALO_ARCH) and int(props.get("simd_count") or 0) in STRIX_HALO_BY_SIMDS:
                name = (f"AMD Radeon {STRIX_HALO_BY_SIMDS[int(props['simd_count'])]} "
                        "(Ryzen AI Max, Strix Halo, gfx1151)")   # the CU count tells the 8060S / 8050S / 8040S apart
        try:
            gtt = int((dev / "mem_info_gtt_total").read_text()) / 2 ** 30
        except (OSError, ValueError):
            gtt = 0.0
        g = {"index": len(found), "name": name, "vram_gb": vram, "arch": arch, "driver": "amdgpu", "vendor": "amd"}
        try:
            g["pci_id"] = int((dev / "device").read_text().strip(), 16)
        except (OSError, ValueError):
            pass
        found.append(amd_apply_uma(g, gtt))            # Strix Halo: unified memory (carve-out + shared)
    return found


def amd_problem(g):
    if g["arch"] not in AMD_ARCHS:
        return (f"not supported - Puddle's AMD backend runs on {AMD_CARDS} only, this is {g['arch']}"
                + (" (an integrated Radeon, not Strix Halo)" if g["arch"] in ("gfx1150", "gfx1152", "gfx1103") else ""))
    return None


def amd_parse_gpus(text, amd) -> list:
    """--gpus with AMD cards, numbered as HIP numbers them (setup's list): "1,0", or "all" (every supported card, the
    most VRAM first).  Every chosen card must be one Puddle supports (AMD_ARCHS; they may be of different
    architectures: the engine is compiled for each).  Returns the cards, the main one first."""
    usable = [g for g in amd if amd_problem(g) is None]
    if str(text).strip().lower() == "all":
        sel = [g["index"] for g in sorted(usable, key=amd_rank)]
    else:
        try:
            sel = [int(x) for x in str(text).split(",") if x.strip()]
        except ValueError:
            fail(f"--gpus takes AMD GPU numbers as setup lists them, e.g. --gpus 1,0 (or --gpus all), not {text!r}")
    if len(sel) < 2 or len(set(sel)) != len(sel):
        fail("--gpus takes two or more different GPUs, e.g. --gpus 1,0 (one GPU: --gpu 1)",
             "this PC has " + (f"{len(usable)} AMD card{'s' if len(usable) != 1 else ''} Puddle can use"
                               + (": " + ", ".join(f"GPU {g['index']} ({g['name']})" for g in usable) if usable else "")))
    byid = {g["index"]: g for g in amd}
    for i in sel:
        g = byid.get(i)
        p = "not found on this PC" if g is None else amd_problem(g)
        if p is not None:
            fail(f"AMD GPU {i}{'' if g is None else ' (' + g['name'] + ')'} cannot be used: {p}",
                 ("use these together: --gpus " + ",".join(str(x["index"]) for x in usable)) if len(usable) >= 2 else
                 ("use one card: --gpu " + str(usable[0]["index"])) if usable else f"the AMD backend runs on {AMD_CARDS}")
    return [byid[i] for i in sel]


# ------------------------------------------------------------------------------------------------ AMD on Windows
# PCI device ids (VEN_1002) of the cards the AMD backend knows; the names below cover a card whose id is not listed
_WIN_AMD_DID = {0x744C: "gfx1100", 0x7448: "gfx1100", 0x745E: "gfx1100",            # RX 7900 XTX/XT/GRE, W7900, W7800
                0x747E: "gfx1101",                                                  # RX 7800 XT / 7700 XT
                0x7480: "gfx1102",                                                  # RX 7600 / 7600 XT
                0x7590: "gfx1200",                                                  # RX 9060 XT
                0x7550: "gfx1201", 0x7551: "gfx1201",                               # RX 9070 / 9070 XT, AI PRO R9700
                0x73BF: "gfx1030", 0x73AF: "gfx1030", 0x73A5: "gfx1030",            # RX 6800 / 6800 XT / 6900 XT / 6950 XT
                0x73DF: "gfx1031",                                                  # RX 6700 XT / 6750 XT / 6800M (#881)
                0x73FF: "gfx1032", 0x73EF: "gfx1032",                               # RX 6600 / 6600 XT, RX 6650 XT (issue 121)
                0x731F: "gfx1010",                                                  # RX 5600 XT / 5700 / 5700 XT (Navi 10)
                **AMD_IGPU_PCI}                                                     # Ryzen APUs: 0x1586 = Strix Halo (gfx1151)
_WIN_AMD_NAME = ((re.compile(r"\b80[456]0S\b", re.I), "gfx1151"),                   # Radeon 8060S / 8050S / 8040S
                 (re.compile(r"\b8[89]0M\b", re.I), "gfx1150"),                     # Radeon 890M / 880M
                 (re.compile(r"\b8[46]0M\b", re.I), "gfx1152"),                     # Radeon 860M / 840M
                 (re.compile(r"\b7[468]0M\b", re.I), "gfx1103"),                    # Radeon 780M / 760M / 740M
                 (re.compile(r"\b9070\b|R9700", re.I), "gfx1201"),
                 (re.compile(r"\b9060\b", re.I), "gfx1200"),
                 (re.compile(r"RX\s*7900|W7900|W7800", re.I), "gfx1100"),
                 (re.compile(r"RX\s*7800|RX\s*7700(?!\s*S)|W7700", re.I), "gfx1101"),
                 (re.compile(r"RX\s*7600|W7600|W7500", re.I), "gfx1102"),
                 (re.compile(r"RX\s*6800(?!\s*[MS])|RX\s*6900|RX\s*6950|W6800", re.I), "gfx1030"),
                 (re.compile(r"RX\s*66[05]0", re.I), "gfx1032"),
                 (re.compile(r"RX\s*5[67]00|W5700", re.I), "gfx1010"))             # RDNA1 Navi 10; the RX 5500 is gfx1012
_DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"


def _win_display_adapters() -> list[dict]:
    """The display adapters present ({"name", "pnp"}), from Win32_VideoController."""
    ps = ("Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.PNPDeviceID + '|' + "
          "$_.AdapterRAM }")
    text = out(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps])
    found = []
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 2 and parts[1]:
            ram = parts[2] if len(parts) > 2 else ""
            found.append({"name": parts[0].strip(), "pnp": parts[1].strip(),
                          "ram": int(ram) if ram.strip().isdigit() else 0})
    return found


def _win_display_registry() -> list[dict]:
    """Each display driver instance's description, matching PCI id and VRAM size (the 64-bit value; the WMI one stops
    at 4 GB), from the display-class registry key - readable without admin."""
    found = []
    try:
        import winreg
        cls = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS)
    except (ImportError, OSError):
        return found
    i = 0
    while True:
        try:
            sub = winreg.EnumKey(cls, i)
        except OSError:
            break
        i += 1
        if not sub.isdigit():
            continue
        try:
            key = winreg.OpenKey(cls, sub)
        except OSError:
            continue
        vals = {}
        for name in ("DriverDesc", "MatchingDeviceId", "HardwareInformation.qwMemorySize", "DriverVersion"):
            try:
                vals[name] = winreg.QueryValueEx(key, name)[0]
            except OSError:
                pass
        found.append(vals)
    return found


def _pci_device_id(text: str) -> int | None:
    m = re.search(r"VEN_1002&DEV_([0-9A-F]{4})", str(text or ""), re.I)
    return int(m.group(1), 16) if m else None


def win_amd_arch(device_id: int | None, name: str) -> str:
    """A Windows AMD adapter's architecture from its PCI device id, else its name; "" when it is none Puddle knows
    (an integrated Radeon, an older card)."""
    if device_id in _WIN_AMD_DID:
        return _WIN_AMD_DID[device_id]
    return next((a for rx, a in _WIN_AMD_NAME if rx.search(name or "")), "")


def amd_gpus_windows(adapters=None, registry=None) -> list[dict]:
    """The AMD display adapters present, in display-adapter order (setup's numbering until the HIP engine is
    installed), with the arch Puddle would run them as ("" = unknown: listed, not supported).  adapters / registry:
    tools/test_setup_amd.py passes mocked ones."""
    adapters = _win_display_adapters() if adapters is None else adapters
    registry = _win_display_registry() if registry is None else registry
    if not adapters:                                   # no WMI answer: the registry alone (it can list removed cards)
        adapters = [{"name": r.get("DriverDesc", ""), "pnp": r.get("MatchingDeviceId", ""), "ram": 0} for r in registry]
    used = set()
    found = []
    for ad in adapters:
        did = _pci_device_id(ad.get("pnp"))
        if did is None:
            continue                                   # not an AMD (VEN_1002) PCI device
        vram, driver = 0.0, ""
        for k, r in enumerate(registry):               # the same card's driver instance: its 64-bit VRAM size
            if k in used or _pci_device_id(r.get("MatchingDeviceId")) != did:
                continue
            if r.get("DriverDesc") and ad.get("name") and r["DriverDesc"].strip() != ad["name"].strip():
                continue
            used.add(k)
            mem = r.get("HardwareInformation.qwMemorySize")
            if isinstance(mem, bytes):
                mem = int.from_bytes(mem[:8], "little")
            vram = int(mem) / 2 ** 30 if isinstance(mem, int) and mem > 0 else 0.0
            driver = str(r.get("DriverVersion") or "")
            break
        if vram == 0.0 and ad.get("ram", 0) > 0:
            vram = ad["ram"] / 2 ** 30                 # WMI's 32-bit figure (at most 4 GB)
        arch = win_amd_arch(did, ad.get("name", ""))
        name = ad.get("name") or AMD_NAMES.get(arch, f"AMD Radeon (device {did:04X})")
        g = {"index": len(found), "name": name, "vram_gb": vram, "arch": arch or f"unknown (PCI {did:04X})",
             "driver": driver or "amd", "vendor": "amd", "pci_id": did}
        found.append(amd_apply_uma(g))                 # Strix Halo: the registry's figure is the BIOS carve-out only
    return found


def hip_lib_dirs(eng: Path) -> list[Path]:
    """Where the ready-made Windows HIP engine's ROCm DLLs are (its BUILD.json "lib_dirs", relative to engine/)."""
    try:
        rel = json.loads((eng / "BUILD.json").read_text(encoding="utf-8")).get("lib_dirs") or []
    except (OSError, ValueError):
        rel = []
    return [eng / d for d in rel if (eng / d).is_dir()]


# #468 #461: the HIP runtime the ready-made engine was built with, next to strata.exe.  Windows looks for an imported
# DLL in the exe's folder, then System32, and only then on PATH (where rocm/bin is): an AMD driver that installs its own
# amdhip64_7.dll in System32 won, and the bundled rocBLAS/hipBLAS ran on that runtime - an access violation (W7900) or
# hipErrorInvalidDeviceFunction (7900 XTX) on the first prompt.  Only the runtime and the compiler it loads by name:
# rocBLAS/hipBLASLt stay in rocm/bin, where they find their kernel libraries and ../.kpack.
HIP_RUNTIME_DLLS = ("amdhip64_*.dll", "amd_comgr*.dll")


def hip_runtime_closure(d: Path) -> list[str]:
    """The DLLs of `d` (the engine's rocm/bin) that amdhip64_7.dll and amd_comgr.dll need, the two included, read from
    their PE import tables and followed recursively (tools/hip/dll_closure.py).  #461: amdhip64_7.dll imports
    rocm_kpack.dll, which imports the MSVC runtime; with only the two DLLs beside strata.exe the full-path load fails
    (error 126), Windows falls back to System32's copy of the runtime, and the first big prompt dies with
    hipErrorInvalidDeviceFunction.  Falls back to the names above when the helper cannot be read."""
    roots = sorted({p.name for pat in HIP_RUNTIME_DLLS for p in d.glob(pat)})
    try:
        sys.path.insert(0, str(ROOT / "tools" / "hip"))
        from dll_closure import dll_closure
        return dll_closure(d, roots)
    except Exception:                                  # noqa: BLE001 - a missing helper must not stop setup
        return roots
    finally:
        if str(ROOT / "tools" / "hip") in sys.path:
            sys.path.remove(str(ROOT / "tools" / "hip"))


def hip_runtime_beside_exe(eng: Path) -> None:
    """Copy the bundled HIP runtime DLLs and everything they import from rocm/bin next to the engine's exes when
    missing or different (a 0.1.34 install, whose zip had them in rocm/bin only, is fixed on its next start; a
    0.1.40.2 install, which got only the two runtime DLLs, gets rocm_kpack.dll and the C++ runtime now, #461)."""
    for d in hip_lib_dirs(eng):
        for name in hip_runtime_closure(d):
            src = d / name
            dst = eng / src.name
            try:
                if dst.exists() and dst.stat().st_size == src.stat().st_size and \
                        dst.stat().st_mtime >= src.stat().st_mtime:
                    continue
                shutil.copy2(src, dst)
            except OSError as e:                   # e.g. the engine is running and holds the old copy
                warn(f"could not put {src.name} next to the AMD engine ({e}); if the engine stops on its first "
                     "request, close Puddle and run START-HERE.bat again")


def rocm_version(root):
    """(major, minor) of a ROCm install, from rocm-core's header; None when it has none."""
    try:
        text = (Path(root) / "include" / "rocm-core" / "rocm_version.h").read_text(encoding="utf-8")
        return tuple(int(re.search(rf"#define\s+ROCM_VERSION_{k}\s+(\d+)", text).group(1)) for k in ("MAJOR", "MINOR"))
    except (OSError, AttributeError, ValueError):
        return None


def rocm_dev_missing(sysroot: Path) -> list:
    """#446: the HIP development files the engine's build needs that a system ROCm lacks (a runtime-only install has
    hipcc and libhipblas but not these, and cmake's enable_language(HIP) then fails on the hip-lang package)."""
    lang = "cmake/hip-lang/hip-lang-config.cmake"     # where CMake's HIP support looks for it
    need = {"lib/" + lang: [sysroot / d / lang for d in ("lib", "lib64", "lib/x86_64-unknown-linux-gnu")],
            "include/hip/hip_runtime.h": [sysroot / "include" / "hip" / "hip_runtime.h"]}
    return [name for name, paths in need.items() if not any(p.is_file() for p in paths)]


def rocm_root(archs):
    """ROCm for compiling and running the HIP engine for `archs` (one arch or a list: the cards of a layer split):
    (root, library folders).  A system ROCm 7 with hipcc, hipBLAS and the HIP development files (#446), else AMD's
    TheRock wheels (rocm_wanted: a version per card family, from the family's index) installed into .venv."""
    archs = [archs] if isinstance(archs, str) else list(archs)
    sysroot = Path(os.environ.get("ROCM_PATH") or "/opt/rocm")
    if (sysroot / "bin" / "hipcc").exists() and list((sysroot / "lib").glob("libhipblas.so*")):
        ver = rocm_version(sysroot)
        missing = rocm_dev_missing(sysroot)
        if (ver is None or ver >= ROCM_SYSTEM_MIN) and not missing:
            return sysroot, [str(sysroot / "lib")]
        if ver is not None and ver < ROCM_SYSTEM_MIN:
            warn(f"the ROCm in {sysroot} is {ver[0]}.{ver[1]}; Puddle needs {ROCM_SYSTEM_MIN[0]}.{ROCM_SYSTEM_MIN[1]} "
                 "or newer: using AMD's wheels in .venv instead")
        else:                                          # #446: a runtime-only ROCm (no -dev packages): cmake would fail
            warn(f"the ROCm in {sysroot} has no HIP development files ({', '.join(missing)}): using AMD's wheels in "
                 ".venv instead (or install them, e.g. AMD's amdrocm-core-dev package for your ROCm and card)")
    indexes = list(dict.fromkeys(rocm_index(a) for a in archs))
    if len(indexes) > 1:                               # TheRock's wheels hold one GPU family's libraries
        fail(f"cards of two GPU families ({', '.join(archs)}) need a system ROCm 7 (in /opt/rocm): AMD's Python "
             "wheels come per family", "install ROCm 7 system-wide, or use cards of one family (--gpu N for one card)")
    index = indexes[0]
    stamp = Path(sys.prefix) / ".strata-rocm.json"
    have = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else {}
    wheel = None
    if have.get("index") == index and have.get("version") and not ROCM_VERSION_OVERRIDE \
            and (Path(sys.executable).parent / "rocm-sdk").exists() and have["version"] == ROCM_FAMILY_PINS.get(rocm_family(index), ROCM_VERSION_DEFAULT):
        wheel = have["version"]                         # the family's pin is installed: no lookup
    if wheel is None:
        wheel, note = rocm_wanted(index)
        if note:
            warn(note)
    if have.get("version") != wheel or have.get("index") != index:
        say(f"  Installing ROCm {wheel} for AMD GPUs into .venv (AMD's TheRock wheels, ~10 GB, no sudo) ...")
        pip = [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--index-url", index]
        if have.get("version") == wheel:               # the same version for another GPU family: its own libraries
            run(pip + ["--force-reinstall", "--no-deps", f"rocm=={wheel}"])
        run(pip + [f"rocm[libraries,devel]=={wheel}"])
        stamp.write_text(json.dumps({"version": wheel, "index": index}))
    # the module, not .venv/bin/rocm-sdk: pip's script names this venv's Python by its path, and stops working when the
    # Puddle folder is moved (issue 78's cmake); the venv's Python itself still runs
    root = Path(out([sys.executable, "-m", "rocm_sdk", "path", "--root"]).strip())
    if not (root / "llvm" / "bin" / "clang++").exists():
        fail(f"ROCm was installed but its compiler is missing ({root})",
             f"remove {stamp} and run this again; or install ROCm 7 system-wide")
    # the card family's libraries only (gfx120X-all -> _rocm_sdk_libraries_gfx120X_all): another family's
    # libhipblaslt.so first on the path would have no kernels for this card
    family = "_rocm_sdk_libraries_" + index.rstrip("/").rsplit("/", 1)[-1].replace("-", "_")
    dirs = [str(root / "lib")]
    for sp in {Path(p) for p in sys.path if p.endswith("site-packages")}:
        libs = sorted(sp.glob("_rocm_sdk_libraries_*"))
        libs = [d for d in libs if d.name.lower() == family.lower()] or libs
        dirs += [str(d / "lib") for d in libs if (d / "lib").is_dir()]
    ok(f"ROCm: {root}")
    return root, list(dict.fromkeys(dirs))


def rocm_win_version(venv):
    """The version of AMD's ROCm wheels installed in a Windows venv, out of their pip metadata; None when it holds
    none (a venv made by hand, or one whose wheels were installed some other way)."""
    for info in sorted((Path(venv) / "Lib" / "site-packages").glob("rocm-*.dist-info")):
        name = info.name[len("rocm-"):-len(".dist-info")]
        if name:
            return name
    return None


def rocm_win_root(archs):
    """Windows: ROCm for compiling the HIP engine for `archs` (one arch, or the cards of a layer split): the TheRock
    wheels in .rocm-win (a private venv: no admin rights, no AMD GPU), with
    the device package of every arch.  An installation there holding the wanted version and all of them is kept as it
    is; another version (STRATA_ROCM_WIN_VERSION) is installed in its place.  Returns (root, the library
    folders)."""
    archs = [archs] if isinstance(archs, str) else list(archs)
    venv = ROOT / ROCM_WIN_VENV
    py, sdk = venv / "Scripts" / "python.exe", venv / "Scripts" / "rocm-sdk.exe"
    # rocm-sdk.exe is pip's launcher, which names the venv's python.exe by its path and does not start once the folder
    # is moved (issue 78's cmake.exe); the venv's python.exe does, so the module is run through it
    rocm_sdk = [str(py), "-m", "rocm_sdk"]
    if not py.exists():
        say(f"  Installing ROCm into {ROCM_WIN_VENV} (AMD's TheRock wheels for Windows, ~7 GB) ...")
        run([sys.executable, "-m", "venv", str(venv)])
    root = None
    if sdk.exists():
        got = out(rocm_sdk + ["path", "--root"]).strip()
        root = Path(got) if got else None
    have = rocm_win_version(venv)
    other = have is not None and have != ROCM_WIN_VERSION
    if other:
        warn(f"{ROCM_WIN_VENV} holds ROCm {have}; {ROCM_WIN_VERSION} is installed in its place "
             "(STRATA_ROCM_WIN_VERSION)")
    elif have is None and ROCM_WIN_VERSION_OVERRIDE and sdk.exists():
        warn(f"{ROCM_WIN_VENV} does not say which ROCm it holds: keeping it, so STRATA_ROCM_WIN_VERSION "
             f"{ROCM_WIN_VERSION} is not taken - remove the folder and run this again to install it")
    missing = root is None or not (root / "lib" / "llvm" / "bin" / "clang++.exe").exists() or \
        [a for a in archs if not (root / ".kpack" / f"blas_lib_{a}.kpack").exists()] or other
    if missing:
        extras = "libraries,devel" + "".join(f",device-{a}" for a in archs)
        say(f"  Installing ROCm {ROCM_WIN_VERSION} for AMD GPUs into {ROCM_WIN_VENV} (AMD's TheRock wheels, ~7 GB) ...")
        run([str(py), "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--index-url", ROCM_WIN_INDEX,
             f"rocm[{extras}]=={ROCM_WIN_VERSION}"])
        run(rocm_sdk + ["init"])
        got = out(rocm_sdk + ["path", "--root"]).strip()
        root = Path(got) if got else None
    if root is None or not (root / "lib" / "llvm" / "bin" / "clang++.exe").exists():
        fail(f"ROCm was installed but its compiler is missing ({root})",
             f"remove the folder {venv} and run this again (AMD's wheels from {ROCM_WIN_INDEX})")
    ok(f"ROCm: {root}")
    return root, [str(root / "bin")]


def hipblaslt_version(lib_dirs):
    """The installed hipBLASLt's version as the engine reads it (hipblasLtGetVersion: 1.4.1 -> 100401), from the
    header of the ROCm whose libraries the engine loads; None when not found."""
    for d in lib_dirs:
        try:
            text = (Path(d).parent / "include" / "hipblaslt" / "hipblaslt-version.h").read_text(encoding="utf-8")
            v = [int(re.search(rf"#define\s+HIPBLASLT_VERSION_{k}\s+(\d+)", text).group(1))
                 for k in ("MAJOR", "MINOR", "PATCH")]
        except (OSError, AttributeError, ValueError):
            continue
        return v[0] * 100000 + v[1] * 100 + v[2]
    return None


# ------------------------------------------------------------------------------------------------ the engine
def driver_major(gpu):
    try:
        return int(gpu["driver"].split(".")[0])
    except (ValueError, KeyError):
        return 0


CUDA_SM120_SUSPECT = (13, 2)   # #892 #968: nvcc 13.2.0 / 13.2.1 (build 13.2.51) for sm_120 made garbage answers (IQ1_S / IQ2_S / IQ3_S) and prompts
                               # (K-quant MMQ) that the same source compiled with 13.0.88 answers correctly (llama.cpp hit it too)
CUDA_SM120_FIXED_BUILD = 86    # CUDA 13.2.2 (nvcc build 13.2.86) fixes it (ggml-org/llama.cpp#28581, confirmed on a 5090 in #968)


def nvcc_build(nvcc):
    """The build number of an nvcc ("Build cuda_13.2.r13.2/compiler...", "V13.2.86" -> 86); None when it cannot be read."""
    m = re.search(r"\bV\d+\.\d+\.(\d+)", out([nvcc, "--version"]) if nvcc else "")
    return int(m.group(1)) if m else None


def sm120_nvcc(nvcc, cuda_v, archs):
    """#892 / #968: CUDA 13.2.0 / 13.2.1's nvcc for an RTX 50 card (sm_120).  Returns (nvcc, cuda_v): an older 13.x toolkit
    when one is installed and the user did not name a compiler (STRATA_NVCC), else the one found, with a warning that says
    what to do.  CUDA 13.2.2 (build 86 or newer) is fine.  Recommends, never forces: a user who picked 13.2 keeps it."""
    if not nvcc or cuda_v != CUDA_SM120_SUSPECT or max(archs) < 120:
        return nvcc, cuda_v
    build = nvcc_build(nvcc)
    if build is not None and build >= CUDA_SM120_FIXED_BUILD:
        return nvcc, cuda_v
    if not os.environ.get("STRATA_NVCC"):
        alt, alt_v = find_nvcc(below=CUDA_SM120_SUSPECT)
        if alt and alt_v and alt_v >= (13, 0):
            ok(f"CUDA {alt_v[0]}.{alt_v[1]} is used for the RTX 50 card (sm_120): CUDA 13.2.0 / 13.2.1's compiler made wrong "
               "answers there (#892, #968); STRATA_NVCC=<nvcc> picks another")
            return alt, alt_v
    warn("CUDA 13.2.0 and 13.2.1's compiler (nvcc) made garbage prompts and answers for RTX 50 cards (sm_120) in two "
         "reports (#892, #968): the same source compiled with CUDA 13.0, 13.1 or 13.2.2 is right. If the engine answers "
         "with nonsense, update to CUDA 13.2.2 or install CUDA 13.0 next to it (https://developer.nvidia.com/cuda-toolkit-archive) "
         "and set STRATA_NVCC to its nvcc; the ready-made engine is built with 13.0")
    return nvcc, cuda_v


def cuda_apt_repo(osr):
    """The NVIDIA apt repo for this distribution ("ubuntu2404", "debian13") from /etc/os-release, or None when it is
    not one setup adds the CUDA keyring for.  Both families publish cuda-keyring and CUDA 13.x."""
    ver = osr.get("VERSION_ID", "").strip('"').replace(".", "")
    repo = {"ubuntu": "ubuntu", "debian": "debian"}.get(osr.get("ID", ""), "") + ver
    return repo if repo in ("ubuntu2204", "ubuntu2404", "debian12", "debian13") else None


def install_build_tools(gpu, yes):
    """The compiler and the CUDA toolkit, installed for the user (asks once).  Returns (nvcc, vcvars)."""
    archs = [int(x) for x in gpu.get("archs", [gpu["arch"]])]
    # #295: Pascal / Volta need a CUDA 12.x toolkit - CUDA 13 cannot build sm_60/sm_70; gpu["toolkit"] = 12: the
    # experimental CUDA 12 engine for any cards (--cuda 12, docs/OLDER_GPUS.md)
    old = int(gpu.get("toolkit") or (12 if min(archs) < CUDA13_MIN_ARCH else 13)) == 12
    need12 = (12, 8) if max(archs) >= 120 else (12, 0)     # sm_120 needs CUDA 12.8 or newer
    nvcc, cuda_v = find_nvcc(below=(13, 0)) if old else find_nvcc()
    if old and (nvcc is None or cuda_v < need12):
        fail("the experimental CUDA 12 engine (Pascal / Volta, or --cuda 12) is compiled here with the NVIDIA CUDA "
             f"Toolkit {need12[0]}.{need12[1]} or a newer 12.x (CUDA 13 cannot compile for these cards)" +
             (f"; found CUDA {cuda_v[0]}.{cuda_v[1]}" if nvcc else ""),
             "install CUDA 12.9 (it can sit next to a newer one) from https://developer.nvidia.com/cuda-toolkit-archive "
             "and run it again (STRATA_NVCC=<its nvcc> picks one toolkit)")
    # RTX 50 (sm_120): CUDA 13.0 - an engine built with 12.8 crashed in the prompt path on Linux (#220).
    # gpu["need_cuda"]: another engine's own floor (strata-poly, which is another prompt path: build_engine_poly)
    need_cuda = need12 if old else gpu.get("need_cuda") or ((13, 0) if max(archs) >= 120 else (12, 0))
    vcvars = find_vcvars(cuda_v) if WIN else None
    have_cc = vcvars is not None if WIN else shutil.which("g++") is not None
    missing = []
    if not have_cc:
        missing.append("Visual Studio 2022 Build Tools (C++)" if WIN else "the C++ compiler (build-essential)")
    if nvcc is None or cuda_v < need_cuda:
        missing.append("the NVIDIA CUDA Toolkit 13.0")
    if not missing:
        nvcc, cuda_v = sm120_nvcc(nvcc, cuda_v, archs)
        ok(f"build tools present (CUDA {cuda_v[0]}.{cuda_v[1]})")
        return nvcc, vcvars
    say("  The engine has to be compiled for your PC, which needs: " + " and ".join(missing) + ".")
    say("  They can be installed now (about 8-10 GB, 15-40 minutes" + (", Windows will ask for permission" if WIN else
                                                                        ", sudo will ask for your password") + ").")
    if ask("  Install them now?", ["y", "n"], "y", yes) != "y":
        fail("the build tools are needed", "install them yourself (see README.md) and run it again")
    if WIN:
        if shutil.which("winget") is None:
            fail("winget (Windows package manager) is not available",
                 "install 'App Installer' from the Microsoft Store, or install the tools by hand (README.md)")
        wg = ["winget", "install", "-e", "--source", "winget", "--accept-package-agreements",
              "--accept-source-agreements", "--disable-interactivity"]
        if not have_cc:
            run([*wg, "--id", "Microsoft.VisualStudio.2022.BuildTools", "--override",
                 "--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"],
                check=False)
        if nvcc is None or cuda_v < need_cuda:
            run([*wg, "--id", "Nvidia.CUDA", "--version", "13.0"], check=False)
        vcvars = find_vcvars(cuda_v)
    else:
        apt = shutil.which("apt-get")
        if apt is None:
            fail("missing: " + " and ".join(missing) + " (the automatic install is only done on Ubuntu/Debian)",
                 "install them with your distribution's packages (Arch: pacman -S base-devel cuda; nvcc is found on "
                 "PATH, in /usr/local/cuda* and in /opt/cuda*), then run it again")
        if not have_cc:
            run(["sudo", "apt-get", "install", "-y", "build-essential"])
        if nvcc is None or cuda_v < need_cuda:
            osr = dict(line.split("=", 1) for line in open("/etc/os-release").read().splitlines() if "=" in line)
            repo = cuda_apt_repo(osr)
            if repo is None:
                fail("the CUDA Toolkit can be installed automatically on Ubuntu 22.04 / 24.04 and Debian 12 / 13 only",
                     "install it from https://developer.nvidia.com/cuda-downloads and run it again")
            deb = Path("/tmp/cuda-keyring.deb")
            download(f"https://developer.download.nvidia.com/compute/cuda/repos/{repo}/x86_64/cuda-keyring_1.1-1_all.deb",
                     deb, "CUDA repository key")
            run(["sudo", "dpkg", "-i", str(deb)])
            run(["sudo", "apt-get", "update"])
            if repo.startswith("ubuntu"):                  # as before: 13.0, and a failed install stops setup here
                run(["sudo", "apt-get", "install", "-y", "cuda-toolkit-13-0"])
            else:
                # Debian 13's CUDA repo carries no 13.0 package: take the oldest 13.x minor it has
                for minor in range(10):
                    if run(["sudo", "apt-get", "install", "-y", f"cuda-toolkit-13-{minor}"], check=False).returncode == 0:
                        break
                else:
                    fail(f"no cuda-toolkit-13-x package could be installed from NVIDIA's {repo} repository",
                         "install the CUDA Toolkit 13 from https://developer.nvidia.com/cuda-downloads and run it again")
    nvcc, cuda_v = find_nvcc(below=(13, 0)) if old else find_nvcc()
    if (WIN and find_vcvars(cuda_v) is None) or (not WIN and shutil.which("g++") is None):
        fail("the C++ build tools did not install", "install them by hand (README.md) and run it again")
    if nvcc is None or cuda_v < need_cuda:
        fail("the CUDA Toolkit did not install", "install it from https://developer.nvidia.com/cuda-downloads, then run it again")
    nvcc, cuda_v = sm120_nvcc(nvcc, cuda_v, archs)
    ok(f"build tools installed (CUDA {cuda_v[0]}.{cuda_v[1]})")
    return nvcc, find_vcvars(cuda_v) if WIN else None


def cached_compiler(bdir, defs):
    """defs with -DCMAKE_CUDA_COMPILER spelled as bdir's CMake cache has it, when the two name the same file.  CMake
    compares the strings: nvcc.exe against nvcc.EXE (find_nvcc() takes either, by PATH or CUDA_PATH) deletes the cache
    and configures again without this command's other -D options, so STRATA_ENABLE_CUDA is off and there is no
    strata-poly target to build."""
    cache = Path(bdir) / "CMakeCache.txt"
    m = re.search(r"^CMAKE_CUDA_COMPILER:\w+=(.+)$", cache.read_text(errors="replace"), re.M) if cache.is_file() else None
    if not m:
        return defs
    flag, old, out = "-DCMAKE_CUDA_COMPILER=", m.group(1).strip(), []
    for d in defs:
        new = d[len(flag):] if d.startswith(flag) else None
        try:
            if new and new != old and os.path.samefile(new, old):
                d = flag + old
        except OSError:
            pass
        out.append(d)
    return out


def cmake_build(src, bdir, target, defs, vcvars, bat_name):
    """`target`: one CMake target, or a list of them (one `--target` each)."""
    cmake, ninja = find_tool("cmake"), find_tool("ninja")
    if cmake is None or ninja is None:
        fail("cmake / ninja not found after installing them", "run: .venv python -m pip install cmake ninja")
    targets = [target] if isinstance(target, str) else list(target)
    conf = [cmake, "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}", "-S", str(src), "-B", str(bdir),
            "-DCMAKE_BUILD_TYPE=Release", *cached_compiler(bdir, defs)]
    build = [cmake, "--build", str(bdir), *[x for t in targets for x in ("--target", t)],
             "-j", str(max(2, (os.cpu_count() or 4) // 2))]
    # A failed build is tried once more: CUDA 13.0's ptxas now and then fails to parse a PTX file it just wrote, and
    # the same command then gets past it (issue #45); a second attempt only compiles what is still missing.
    if WIN:
        if vcvars is None:                             # #881: the build needs MSVC's environment
            fail("the Visual Studio C++ build tools were not found",
                 "install them (Visual Studio 2019 or 2022, workload 'Desktop development with C++') and run setup again")
        bat = ROOT / bat_name
        q = lambda c: " ".join(f'"{x}"' if " " in str(x) else str(x) for x in c)  # noqa: E731
        bat.write_text(f'@echo off\r\ncall "{vcvars}"{vcvars_toolset(vcvars)} >nul\r\n'
                       f'{q(conf)} || exit /b 1\r\n{q(build)} && exit /b 0\r\n'
                       f'echo   (the build stopped - trying it once more)\r\n{q(build)} || exit /b 1\r\n',
                       encoding="utf-8")
        run(["cmd", "/c", str(bat)])
    else:
        run(conf)
        if run(build, check=False).returncode != 0:
            say("  (the build stopped - trying it once more)")
            run(build)


ENGINE_SOURCES = ("CMakeLists.txt", "cmake", "src", "include", "third_party/ggml")
VISION_SOURCES = ("tools/vision",)


def source_hash(parts) -> str:
    """A fingerprint of the files a compiled engine is built from, kept in engine/BUILD.json: when a `git pull`
    changes them, the engine is compiled again (issue #31)."""
    h = hashlib.sha256(LLAMA_CPP_COMMIT.encode())
    for part in parts:
        base = ROOT / part
        for f in [base] if base.is_file() else sorted(x for x in base.rglob("*") if x.is_file()):
            h.update(f.relative_to(ROOT).as_posix().encode() + b"\0" + f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def vision_hash() -> str:
    """The image encoder's fingerprint in engine/BUILD.json: its source and the llama.cpp it is built from
    (VISION_LLAMA_CPP_COMMIT), so a new pin compiles it again."""
    return hashlib.sha256((VISION_LLAMA_CPP_COMMIT + source_hash(VISION_SOURCES)).encode()).hexdigest()[:16]


def host_compiler_defs(cuda: bool) -> list:
    """CC / CXX / CUDAHOSTCXX pick the host compilers of a build - as -D definitions, so an existing build folder
    takes them too (the environment reaches CMake only on a folder's first configure, which is why CXX=g++-14 alone
    could die at the last step, #1645).  CUDAHOSTCXX is nvcc's host compiler (-ccbin); it defaults to CXX, as
    CMake's does.  `cuda`: the project compiles CUDA (the host-compiler definition means nothing to a C++-only
    one)."""
    cc = os.environ.get("CC", "").strip()
    cxx = os.environ.get("CXX", "").strip()
    hostcxx = os.environ.get("CUDAHOSTCXX", "").strip() or cxx
    return ([f"-DCMAKE_C_COMPILER={cc}"] if cc else []) + \
           ([f"-DCMAKE_CXX_COMPILER={cxx}"] if cxx else []) + \
           ([f"-DCMAKE_CUDA_HOST_COMPILER={hostcxx}"] if cuda and hostcxx else [])


# ------------------------------------------------------------------------------------------------ the data folder
# The model files - the GGUFs, the prepared packs and the MTP layer, 70-120 GB - live in a data folder NEXT TO the
# Puddle folder (`Puddle-data`), not inside it: updating Puddle by unzipping a new copy used to give a new, empty
# folder and a full download again.  Where it is, and which Puddle folders this user ran, is kept in a small
# per-user file of Puddle's own (not PolyStrata's), so every Puddle folder on the PC finds the same files.  Another
# program may share the data folder (a PolyStrata install pointed at it through a symlink): Puddle reads the files
# there and never moves files out of a PolyStrata folder or rewrites its configs.
DATA_ITEMS = ("models", "packs", "mtp")
DATA_FOLDER_NAME = "Puddle-data"
INSTALL_PREFIX = "puddle"                  # the folders other_installs() looks at: Puddle's, never PolyStrata's


def confirm_risk(msg, explicit, yes, stop, hint=None, question="  Go on anyway?", default="n") -> None:
    """The owner's rule: setup recommends, it never forces.  A choice setup expects to fail or run badly is said
    plainly (msg), then asked (default n; `default` keeps an older question's own default), or with --yes taken as
    consent when it was asked for explicitly (a flag such as --model or --gpus): --yes alone keeps the stop
    (stop, hint).  Returns when it goes on; the caller says what it does."""
    warn(msg)
    if yes and explicit:
        return
    if ask(question, ["y", "n"], default, yes) != "y":
        fail(stop, hint)


def settings_path() -> Path:
    """Puddle's own per-user settings (the data folder, the Puddle folders this user ran, calibrations): not
    PolyStrata's %APPDATA%\\PolyStrata / ~/.config/polystrata, so a PolyStrata install on the same PC is left alone."""
    if WIN:
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "Puddle" / "settings.json"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "puddle" / "settings.json"


def load_settings() -> dict:
    try:
        return json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(s: dict) -> None:
    try:
        settings_path().parent.mkdir(parents=True, exist_ok=True)
        settings_path().write_text(json.dumps(s, indent=1), encoding="utf-8")
    except OSError as e:
        warn(f"could not save {settings_path()} ({e})")


def has_data(folder: Path) -> bool:
    for d in DATA_ITEMS:
        try:
            if (folder / d).is_dir() and any((folder / d).iterdir()):
                return True
        except OSError:
            pass
    return False


def other_installs(settings: dict) -> list:
    """Puddle folders besides this one that may hold model files: the ones this user ran before, and Puddle* folders
    next to this one (a zip unpacked again lands in e.g. `Puddle-main (1)\\Puddle-main`).  PolyStrata folders are not
    among them: Puddle never moves their files or rewrites their configs."""
    cands = [Path(p) for p in settings.get("installs", [])]
    for base in dict.fromkeys((ROOT.parent, ROOT.parent.parent)):
        try:
            for d in base.iterdir():
                if d.is_dir() and d.name.lower().startswith(INSTALL_PREFIX):
                    cands.append(d)
                    cands += [c for c in d.iterdir() if c.is_dir() and c.name.lower().startswith(INSTALL_PREFIX)]
        except OSError:
            pass
    found = []
    for d in cands:
        try:
            d = d.resolve()
            if d != ROOT and d not in found and (d / "setup.py").is_file():
                found.append(d)
        except OSError:
            pass
    return found


def same_drive(a: Path, b: Path) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def same_folder(a, b) -> bool:
    """`a` and `b` are one folder: the same path, or two names of it (a symlink such as PolyStrata-data -> Puddle-data,
    a junction, other letter case on Windows).  A folder that does not exist is compared by its resolved path."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        pass
    try:
        return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))
    except OSError:
        return os.path.normcase(str(a)) == os.path.normcase(str(b))


def move_into(src: Path, dst: Path) -> None:
    """A rename into the data folder (same drive: instant); a folder merges into one already there, keeping what the
    destination has.  Whatever cannot be moved (a file in use) stays where it is."""
    if not dst.exists():
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst)
            return
        except OSError:
            if not src.is_dir():
                return
            dst.mkdir(parents=True, exist_ok=True)
    if src.is_dir() and dst.is_dir():
        for c in list(src.iterdir()):
            move_into(c, dst / c.name)
        try:
            src.rmdir()
        except OSError:
            pass


def repoint_config(cfg_file: Path, old: Path, new: Path) -> None:
    """A config whose model files moved from `old` to `new` points at them there (each path only if its file is
    now there and no longer at the old place)."""
    try:
        cfg = json.loads(cfg_file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return

    def fix(v):
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, str):
            for d in DATA_ITEMS:
                o = str(old / d)
                nv, no = os.path.normcase(v), os.path.normcase(o)   # Windows: C:\ and c:\ are the same place
                if nv == no or nv.startswith(no + os.sep):
                    n = str(new / d) + v[len(o):]
                    if Path(n).exists() and not Path(v).exists():
                        return n
        return v

    new_cfg = fix(cfg)
    if new_cfg != cfg:
        write_config(cfg_file, new_cfg)


def data_folder(requested: str | None) -> tuple:
    """(the data folder, folders on other drives that still hold model files).  Moves the model files of this folder
    and of earlier Puddle folders on the same drive into the data folder, and points their configs there.  A source
    that is the data folder under another name (a symlink to it) is the data folder: nothing moves onto itself."""
    settings = load_settings()
    dest = Path(requested).expanduser().resolve() if requested else \
        Path(settings["data_dir"]) if settings.get("data_dir") else ROOT.parent / DATA_FOLDER_NAME
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as e:                                # e.g. no write access next to the Puddle folder
        warn(f"cannot use {dest} for the model files ({e}): keeping them in {ROOT}")
        dest = ROOT
    elsewhere = []
    # #198: the data folder remembered before (a --data-dir to a new place) is a source too, and so is a Puddle-data
    # folder nested in any of them (an install that kept its models one level down)
    sources = [ROOT, *other_installs(settings)]
    if settings.get("data_dir") and not same_folder(settings["data_dir"], dest):
        sources.append(Path(settings["data_dir"]))
    sources += [f / DATA_FOLDER_NAME for f in list(sources) if not same_folder(f / DATA_FOLDER_NAME, dest)]
    seen = set()
    for folder in sources:
        try:
            key = os.path.normcase(str(folder.resolve()))   # two names of one folder are one source
        except OSError:
            key = os.path.normcase(str(folder))
        if key in seen:
            continue
        seen.add(key)
        if same_folder(folder, dest) or not has_data(folder):
            continue
        if not same_drive(folder, dest):
            elsewhere.append(folder)                    # another drive: used where it is (no 70 GB copy)
            continue
        # the downloads merge file by file (the same file wherever it came from); a prepared pack or MTP layer moves
        # whole or not at all, so two copies are never mixed
        if (folder / "models").is_dir():
            move_into(folder / "models", dest / "models")
        for item in [*((folder / "packs").glob("*") if (folder / "packs").is_dir() else []), folder / "mtp"]:
            rel = item.relative_to(folder)
            if item.exists() and not (dest / rel).exists():
                move_into(item, dest / rel)
        for d in ("packs",):
            try:
                (folder / d).rmdir()                    # empty now
            except OSError:
                pass
        for c in model_config_files(folder):
            repoint_config(c, folder, dest)
        if has_data(folder):
            elsewhere.append(folder)                    # in use, or a copy the data folder already has
            warn(f"some model files are still in {folder} (in use, or already in {dest})")
        else:
            ok(f"model files from {folder} moved to {dest} (a new copy of Puddle finds them there)")
    installs = [str(ROOT)] + [p for p in settings.get("installs", []) if p != str(ROOT) and Path(p).is_dir()]
    save_settings({**settings, "data_dir": str(dest), "installs": installs[:20]})
    return dest, elsewhere


def write_config(path: Path, cfg: dict):
    """A run config, written whole or not at all (#459): to a temporary file first, then moved over the old one, so
    a setup stopped half-way (a closed window, a full disk) never leaves an empty strata-*.json behind."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# #629: the run config's keys setup writes itself (and rewrites on every setup run); any other key is the user's - a
# "sampling" or "mcp_servers" block, "allowed_hosts", "cors_origins", "open_browser" - and is kept when setup runs again
SETUP_KEYS = frozenset({"exe", "args", "cwd", "tokenizer", "model_name", "log", "lib_dirs", "port", "backend", "env",
                        "gpu", "gpus_asked", "layer_split", "host", "api_key", "vision",
                        "display_name", "cuda", "engine"})
SETUP_ENV = frozenset({"STRATA_HIPBLASLT_TUNING", "STRATA_ROCBLAS_TUNING", "STRATA_RESIDENT_PIN", "STRATA_NO_ARENA_THP", "STRATA_MMQ_KQ", "STRATA_HC_Q8",
                       "STRATA_ROUTE_TAIL_SKIP"}) | OLD_GFX_SWITCHES   # the "env" entries setup writes (the gfx1103 / gfx1150 switches are setup's too: they do not linger)
# the Qwen engine wrote STRATA_MMQ_KQ, STRATA_HC_Q8 and STRATA_ROUTE_TAIL_SKIP for Unsloth's files; in a strata-poly
# config they are the user's own (STRATA_ROUTE_TAIL_SKIP=0 turns the tail skip off ...): a setup run again keeps them
SETUP_ENV_USER = frozenset({"STRATA_MMQ_KQ", "STRATA_HC_Q8", "STRATA_ROUTE_TAIL_SKIP"})
SETUP_VISION = frozenset({"exe", "mmproj", "model", "gpu", "max_tokens", "threads"})


def carry_over(old: dict, cfg: dict) -> list[str]:
    """#629: setup run again for an installed model keeps what the user added to its run config: every key setup does
    not write (`SETUP_KEYS`), the "env" entries setup does not write, and in "vision" the keys setup does not write
    plus an mmproj of their own (a Q8_0 one, #625) that still exists.  `cfg` (the new config) is updated in place;
    the names of what was kept are returned.  Engine options added by hand to "args" are not merged (setup chooses
    those): `args_dropped` names them."""
    kept = []
    for k, v in old.items():
        if k not in SETUP_KEYS and k not in cfg:
            cfg[k] = v
            kept.append(k)
    mine = SETUP_ENV - SETUP_ENV_USER
    env = {k: v for k, v in (old.get("env") or {}).items() if k not in mine and k not in (cfg.get("env") or {})} \
        if isinstance(old.get("env"), dict) else {}
    if env:
        cfg["env"] = {**(cfg.get("env") or {}), **env}
        kept += [f"env {k}" for k in env]
    ov, nv = old.get("vision"), cfg.get("vision")
    if isinstance(ov, dict) and isinstance(nv, dict):
        for k, v in ov.items():
            if k not in SETUP_VISION and k not in nv:
                nv[k] = v
                kept.append(f"vision {k}")
        mm = ov.get("mmproj")                          # a file of the user's own: not the one setup downloads
        if isinstance(mm, str) and Path(mm).name != Path(str(nv.get("mmproj"))).name and Path(mm).is_file():
            nv["mmproj"] = mm
            kept.append("vision mmproj")
    kept += carry_profile_args(old, cfg)
    return kept


def flag_value(args: list, flag: str):
    """The value after `flag` in an argument list, or None."""
    for i, a in enumerate(args[:-1]):
        if a == flag:
            return str(args[i + 1])
    return None


def carry_profile_args(old: dict, cfg: dict) -> list[str]:
    """#775: a learned expert profile the user wired in by hand (--expert-profile <their file>, --expert-profile-save
    <file>) survives a setup run: setup writes the shipped profile, which a measured one of their own beats (2.5-3.1x at
    256K on a 16 GB card in the report).  Their --expert-profile replaces the shipped one only when it is another file
    that still exists; --expert-profile-save is kept when the new config has none.  The names kept are returned."""
    oargs, nargs = old.get("args"), cfg.get("args")
    if not isinstance(oargs, list) or not isinstance(nargs, list):
        return []
    kept = []
    mine, shipped = flag_value(oargs, "--expert-profile"), flag_value(nargs, "--expert-profile")
    if mine and shipped and Path(mine).name != Path(shipped).name and Path(mine).is_file():
        nargs[nargs.index("--expert-profile") + 1] = mine
        kept.append("args --expert-profile")
    save = flag_value(oargs, "--expert-profile-save")
    if save and flag_value(nargs, "--expert-profile-save") is None:
        nargs += ["--expert-profile-save", save]
        kept.append("args --expert-profile-save")
    return kept


def args_dropped(old: dict, cfg: dict) -> list[str]:
    """#629: the engine options of the earlier run config that the new one has no more (by flag name): options added
    by hand, which a setup run does not carry over - the start of the line that names them."""
    def flags(c):
        a = c.get("args") if isinstance(c.get("args"), list) else []
        return [str(x) for x in a if str(x).startswith("--")]
    new = set(flags(cfg))
    return list(dict.fromkeys(f for f in flags(old) if f not in new))


def write_setup_config(cfg_path: Path, cfg: dict, source: Path | None = None, sampling: dict | None = None) -> None:
    """#629: setup's run config, written over an earlier one for the same model without losing what the user added
    to it: the keys setup does not write are carried over (carry_over), and the earlier file is kept as
    strata-<model>.json.bak when it changes.  `source`: an earlier install's config to carry the keys over from when
    this folder has none yet (a copy set up like the last one).  A line says what was kept, one what was not.
    `sampling`: the family's sampling, written as the "sampling" block when the config has none of the user's."""
    old_path = cfg_path if cfg_path.is_file() else source
    old = None
    if old_path is not None and old_path.is_file():
        try:
            old = json.loads(old_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            pass
        if not isinstance(old, dict):
            old = None
    kept = carry_over(old, cfg) if old is not None else []
    if sampling and old is not None and old.get("sampling") == sampling:   # setup's own block from a run before:
        kept = [k for k in kept if k != "sampling"]                        # kept, but not news
    if sampling and "sampling" not in cfg:             # a block of the user's (carried over) stays as it is
        cfg["sampling"] = dict(sampling)
        ok("sampling for requests that set none: " + ", ".join(f"{k} {v}" for k, v in sampling.items())
           + " (\"sampling\" in the config)")
    bak = None
    if cfg_path.is_file() and old != cfg:
        bak = cfg_path.with_name(cfg_path.name + ".bak")
        try:
            shutil.copyfile(cfg_path, bak)
        except OSError as e:
            warn(f"could not keep a copy of the earlier {cfg_path.name} ({e.strerror or e})")
            bak = None
    write_config(cfg_path, cfg)
    if kept:
        ok(f"kept from your earlier {old_path.name}: " + ", ".join(kept))
    if bak is not None:
        dropped = args_dropped(old, cfg) if old is not None else []
        say(f"  the earlier run config is kept as {bak.name}" + (
            f"; engine options it had that this one has not (setup chooses those): {' '.join(dropped)}"
            if dropped else ""))


def readable_config(path: Path) -> bool:
    """#459: a config that parses as a JSON object; any other gets a one-line warning naming it."""
    text = None
    try:
        text = path.read_text(encoding="utf-8-sig")
        if isinstance(json.loads(text), dict):
            return True
        why = "not a JSON object"
    except OSError as e:
        why = e.strerror or str(e)
    except ValueError:                                 # JSONDecodeError, or bytes that are not UTF-8
        why = "the file is empty" if text is not None and not text.strip() else "not valid JSON"
    warn(f"skipped the earlier config {path} ({why}): setting this copy up without it")
    return False


def model_config_files(folder: Path) -> list:
    """#346: the strata-*.json files of a folder that can be model configs. The server keeps a config's Chat settings
    next to it as strata-<model>.shared-settings.json: it matches the pattern but is no config (no exe, no args)."""
    return [p for p in folder.glob("strata-*.json") if not p.name.endswith(".shared-settings.json")]


def previous_config(elsewhere_first: list, settings: dict):
    """The most recently used model config of another Puddle folder on this PC, for a folder that has none yet.  One
    that does not parse (an empty or cut-off file, #459) is skipped with a warning: the newest readable one is used,
    and with none this copy is set up as a fresh install."""
    cands = []
    for folder in [*elsewhere_first, *other_installs(settings)]:
        cands += model_config_files(folder)
    cands = [c for c in dict.fromkeys(cands) if c.is_file()]
    return next((c for c in sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True) if readable_config(c)), None)


def choices_from_config(cfg_path: Path) -> dict:
    """The setup answers a config was written with (family, size, context, images, network).  A config of a family
    Puddle does not have (a Qwen engine's) has no family and no size."""
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    tag = cfg_path.stem[len("strata-"):]
    family = next((f for f, d in FAMILIES.items() if d["tag"] and tag.startswith(d["tag"])), None)
    if family is None:
        return {"family": None, "model": None}
    model = tag[len(FAMILIES[family]["tag"]):].upper()
    sizes = family_models(family)
    # Some published quant names are lowercase (MiniCPM-V's mix_q3gud_im).
    # Match the filename case-insensitively, then return the catalog's canonical key.
    model = next((m for m in sizes if m.casefold() == model.casefold()), model)
    a = cfg.get("args", [])
    val = lambda k: a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else None   # noqa: E731
    vis = cfg.get("vision")
    if llama_config(cfg):                              # llama-server reads pictures with its --mmproj
        vis = {"gpu": True} if val("--mmproj") else None
    return {"family": family, "model": model if model in sizes else None,
            "context": int(val("--max-context") or val("--ctx-size")) if val("--max-context") or val("--ctx-size")
            else None,
            "vision": ("gpu" if vis.get("gpu") else "cpu") if isinstance(vis, dict) else "none",
            "host": cfg.get("host"), "api_key": cfg.get("api_key"), "port": cfg.get("port"), "gpu": cfg.get("gpu"),
            "layer_split": cfg.get("layer_split"), "cuda": 12 if config_toolkit(cfg) == 12 else None,
            # #493: --vram-reserve-mib given at setup (images write the default 700 themselves)
            "vram_reserve_mib": int(val("--vram-reserve-mib")) if (val("--vram-reserve-mib") or "").isdigit() and (
                vis is None or int(val("--vram-reserve-mib")) != VISION["gpu"]["reserve_mib"]) else None}


def find_in(roots: list, rel: str):
    """The first of roots/rel that exists."""
    for r in roots:
        if (r / rel).exists():
            return r / rel
    return None


# ------------------------------------------------------------------------------------------------ start
def model_config(path: Path) -> bool:
    """#549: a model's run config (a JSON object with "exe" and "args"). Any other strata-*.json in the folder (a
    file of the user's own, a cut-off one) is skipped with a warning naming it instead of stopping setup."""
    try:
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(cfg, dict) and cfg.get("exe") and isinstance(cfg.get("args"), list):
            return True
        why = 'no "exe" or "args"'
    except OSError as e:
        why = e.strerror or str(e)
    except ValueError:
        why = "not valid JSON"
    warn(f"skipped {path.name} ({why}): it is not a Puddle model config")
    return False


def installed_configs():
    return [p for p in sorted(model_config_files(ROOT), key=lambda p: p.stat().st_mtime, reverse=True)
            if model_config(p)]


def source_version() -> str:
    """The engine version the source tree builds (CMakeLists.txt's project version)."""
    m = re.search(r"project\(strata VERSION ([\d.]+)", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8"))
    return m.group(1) if m else "0"


def engine_version(exe: Path) -> tuple:
    """The version in the engine folder's BUILD.json, or else the one compiled into the binary.  A locally compiled
    engine is not necessarily the source's version: when compiling a `git pull` fails, the previous engine is kept
    (issue #49)."""
    try:
        meta = json.loads((Path(exe).parent / "BUILD.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    v = str(meta.get("version") or "")
    if not v:                                          # the version compiled into the binary: 0.1.13 and newer
        try:                                           # carry it, so a binary without it is older
            m = re.search(rb"engine=(\d+\.\d+\.\d+(?:\.\d+)?)\n", Path(exe).read_bytes())
            v = m.group(1).decode() if m else "0.1.12"
        except OSError:
            v = "0"
    return tuple(int(x) for x in v.split(".")[:4] if x.isdigit())


def update_install(have: list, a) -> int:
    """#475: `setup.py --update` (UPDATE.bat / update.sh, after their git pull): what a plain START-HERE.bat does to
    an install before it starts the model, without starting it - the Python packages, the engine compiled again when
    its source changed, and each installed model's config upgrades.  No question is asked and the model files are not
    touched."""
    have = [p for p in have if model_config(p)]        # #549: a strata-*.json that is no model config is skipped
    if not have:
        say("  No model is installed in this Puddle folder yet: run START-HERE.bat (Linux: ./setup.sh) to set it up -")
        say("  it finds an earlier install's model files next to it and reuses them.")
        return 0
    pip_install(requirement_lines() if REQUIREMENTS.exists() else PY_PACKAGES,
                "numpy, jinja2, regex, pyyaml, tqdm, requests, cmake, ninja, pillow, psutil")
    for cfg_path in have:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        if llama_config(cfg):                          # Bolt on llama-server: compiled again when its commit moved
            update_llama_server(cfg, cfg_path=cfg_path)
            ok(f"{cfg.get('model_name', cfg_path.stem)}: up to date")
            continue
        if not poly_config(cfg):                       # a Qwen engine's config: Puddle has no engine for it
            warn(f"{cfg_path.name}: for an engine Puddle does not have ({Path(str(cfg.get('exe', '?'))).name}); "
                 "left as it is")
            continue
        update_poly_engine(cfg, cfg_path=cfg_path)
        poly_tail_skip(cfg, cfg_path)
        poly_kv_whole(cfg, cfg_path)
        if cfg.get("backend") == "hip" and WIN:
            hip_runtime_beside_exe(Path(cfg["exe"]).parent)   # #468 #461
        ok(f"{cfg.get('model_name', cfg_path.stem)}: up to date")
    tests = getattr(a, "tests", None)
    if isinstance(tests, str):                         # --tests [REGEX]: the engine's own tests, compiled and run
        rc = run_poly_tests(tests)
        if rc != 0:
            return rc
    ver = engine_version(Path(json.loads(have[0].read_text(encoding="utf-8-sig"))["exe"]))
    say()
    ok("Puddle is updated" + (f" (engine {'.'.join(map(str, ver))})" if any(ver) else "") +
       ". Start the model with " + ("START-HERE.bat" if WIN else "./setup.sh") + " when you want it.")
    return 0


def settings_summary(cfg: dict, port=None) -> str:
    """#564: the settings a start uses, in one line: the config's engine options (the model's file paths left out)
    and the server's own fields, so a change made by hand to strata-<model>.json can be checked without the log."""
    a, out, i = [str(x) for x in cfg.get("args") or []], [], 0
    while i < len(a):
        flag = a[i]
        val = a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else None
        i += 1 if val is None else 2
        if not flag.startswith("--"):
            continue                                   # a positional: the model file
        if val is not None and ("/" in val or "\\" in val or val.lower().endswith((".gguf", ".bin"))):
            continue                                   # a path: --native, --mtp, --profile ...
        out.append(flag if val is None else f"{flag} {val}")
    srv = [f"{cfg.get('host', '127.0.0.1')}:{port or cfg.get('port', 8080)}"]
    if cfg.get("api_key"):
        srv.append("api key set")
    if cfg.get("open_browser") is False:               # #609
        srv.append("no browser")
    for k in ("gpu", "layer_split", "draft_vocab", "fit_max_tokens", "reasoning_budget_tokens", "anthropic_thinking"):
        if cfg.get(k) is not None:
            v = cfg[k]
            srv.append(f"{k} {','.join(map(str, v)) if isinstance(v, list) else str(v).lower() if isinstance(v, bool) else v}")
    return " ".join(out) + ("; " if out else "") + "server " + ", ".join(srv)


def start(cfg_path: Path, port: int | None, gpu: int | list | None = None, open_browser=True, yes=False,
          layer_split=None, keep=None) -> int:
    """keep: settings given on this start that the model keeps from now on (--host, --api-key, --vram-reserve-mib)."""
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    llama = llama_config(cfg)                          # Nemotron 3.5 Bolt on llama-server
    if not poly_config(cfg) and not llama:             # strata-poly or llama-server (the Qwen engine is gone)
        fail(f"{cfg_path.name} is for an engine Puddle does not have ({Path(str(cfg.get('exe', '?'))).name})",
             "set the model up again: ./setup.sh --setup (Windows: START-HERE.bat --setup)")
    missing = [p for p in [cfg["exe"], *[a for a in cfg["args"] if a.endswith(".gguf")]] if not Path(p).exists()]
    if missing:
        fail(f"{cfg_path.name} refers to missing files: {missing[0]}", "run it again with --setup to repair")
    if isinstance(gpu, list):                          # update_poly_engine() wants code for these cards
        cfg["gpu"] = gpu
    if llama:
        update_llama_server(cfg, yes, cfg_path)
    else:
        update_poly_engine(cfg, yes, cfg_path)
        poly_tail_skip(cfg, cfg_path)
        poly_kv_whole(cfg, cfg_path)
    keep = {k: v for k, v in (keep or {}).items() if v is not None}
    reserve = keep.pop("vram_reserve_mib", None)       # #493: an engine argument, kept in the config's args
    if reserve is not None and llama:
        warn("--vram-reserve-mib is strata-poly's setting; llama-server has none: not saved")
        reserve = None
    if reserve is not None:
        args = cfg["args"]
        if "--vram-reserve-mib" in args[:-1]:
            args[args.index("--vram-reserve-mib") + 1] = str(reserve)
        else:
            args += ["--vram-reserve-mib", str(reserve)]
        write_config(cfg_path, cfg)
        ok(f"saved for this model: {reserve} MiB of VRAM kept free for other programs (--vram-reserve-mib)")
    if keep and any(cfg.get(k) != v for k, v in keep.items()):   # #179: a --host/--api-key on a start was ignored
        cfg.update(keep)
        write_config(cfg_path, cfg)
        ok("saved for this model: " + ", ".join(
            "api key" if k == "api_key" else ("the browser opens" if v else "no browser") if k == "open_browser"
            else f"{k.replace('_', ' ')} {v}" for k, v in keep.items()))
    cfg_path.touch()                                     # the most recently used model
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(port or cfg.get("port", 8080))]
    if cfg.get("backend") == "hip":                    # AMD, numbered as HIP numbers them (setup's KFD order)
        if WIN:
            hip_runtime_beside_exe(Path(cfg["exe"]).parent)   # #468 #461: also fixes a 0.1.34 install
        amd = amd_gpus()
        if isinstance(gpu, list):                      # --gpus: saved, this model runs on these cards from now on
            amd_parse_gpus(",".join(str(i) for i in gpu), amd)   # (update_poly_engine() compiled for them)
            cfg["gpu"], cfg["gpus_asked"] = gpu, True
            cfg["layer_split"] = layer_split or cfg.get("layer_split") or "auto"
            write_config(cfg_path, cfg)
            gpu = None
        elif gpu is not None:
            cmd += ["--gpu", str(gpu)]
        found = []
        use = gpu if gpu is not None else cfg.get("gpu")
        if isinstance(use, list):
            byid = {x["index"]: x for x in amd}
            ok("GPUs: " + " + ".join(gpu_name(byid[i]) if i in byid else f"GPU {i} (not found)" for i in use)
               + f" together, AMD (layers split {cfg.get('layer_split') or 'auto'})")
        else:
            g = next((x for x in amd if x["index"] == (use if use is not None else x["index"])
                      and amd_problem(x) is None), None)
            if g is not None:
                ok(f"GPU: {g['name']} ({g['vram_gb']:.0f} GB, AMD)")
    else:
        found = gpus()
        global OLD_GPUS                                # Pascal / Volta: named on this start, or this model's CUDA 12 engine
        OLD_GPUS = OLD_GPUS or old_gpus_opt_in(found, gpu if isinstance(gpu, list) else [gpu] if gpu is not None else
                                               cfg.get("gpu") if isinstance(cfg.get("gpu"), list) else [cfg.get("gpu")],
                                               12 if config_toolkit(cfg) == 12 else None)
    if cfg.get("backend") == "hip":
        pass
    elif isinstance(gpu, list):                        # --gpus: saved, this model runs on these cards from now on
        check_gpus(gpu, found, yes=yes, named=True)
        cfg["gpu"], cfg["gpus_asked"] = gpu, True
        cfg["layer_split"] = layer_split or cfg.get("layer_split") or "auto"
        write_config(cfg_path, cfg)
        gpu = None
    elif gpu is not None:                              # --gpu N: this start only, on that card
        check_gpus([gpu], found)
        cmd += ["--gpu", str(gpu)]
    use = gpu if gpu is not None else cfg.get("gpu")
    if cfg.get("backend") == "hip":
        pass
    elif isinstance(use, list):
        check_gpus(use, found, "(chosen for this model) ", yes=True, named=True)
        byid = {g["index"]: g for g in found}
        ok("GPUs: " + " + ".join(gpu_name(byid[i]) for i in use) + f" together (layers split {cfg.get('layer_split') or 'auto'})")
    elif found:
        g = next((x for x in found if x["index"] == use), None) if use is not None else max(
            found, key=lambda x: (round(x["vram_gb"]), -x["index"]))
        if g is not None:
            ok("GPU: " + gpu_name(g))
    browser = open_browser and cfg.get("open_browser") is not False   # #609: "open_browser": false, --no-browser
    if browser:
        cmd.append("--open")
    gb = 0.0
    for flag in ("--native", "--model"):
        if flag in cfg["args"]:
            try:
                gb = Path(cfg["args"][cfg["args"].index(flag) + 1]).stat().st_size / 1e9
            except (OSError, IndexError):
                pass
    say()
    say("  " + "-" * 100)
    size = f'about {gb:.0f} GB' if gb >= 1 else 'the model file'
    if llama:
        say(f"  Starting {cfg.get('model_name', 'the model')} on llama-server: it reads the model file ({size}) onto "
            "the GPUs.")
    else:
        say(f"  Starting {cfg.get('model_name', 'the model')}: it reads the model file ({size}) in place: the weights "
            "go to the GPU,")
        say("  and the OS file cache keeps the experts the GPU does not hold.")
    say("  While it does, YOUR PC CAN BE SLOW OR STOP RESPONDING FOR 1-3 MINUTES (longer the first time after a")
    say("  restart). That is normal: please wait and don't close this window - " + (
        "the browser opens when it is ready." if browser else "the server says when it is ready."))
    say("  Later, closing this window stops the model.")
    say("  " + "-" * 100)
    for n, line in enumerate(textwrap.wrap(f"Settings ({cfg_path.name}): {settings_summary(cfg, port)}", 100,
                                           break_on_hyphens=False)):   # #564: what this start uses
        say(("  " if n == 0 else "    ") + line)
    return subprocess.call(cmd)


def vision_tokens(asked: int | None, vision: str, earlier: Path | None, own: int | None = None) -> int:
    """#625: the most image tokens a picture becomes (the config's vision.max_tokens): --vision-tokens N, else what an
    earlier config of this model chose for the same encoder device (a setup run again keeps it), else the family's own
    default (`own`; 0: the encoder file's own limits, no --max-tokens) or the device's (VISION).  More is allowed with
    a note on the time it takes: setup recommends, it does not cap."""
    default = VISION[vision]["max_tokens"] if own is None else own
    if asked is None and earlier is not None:
        try:
            v = json.loads(earlier.read_text(encoding="utf-8-sig")).get("vision")
        except (OSError, ValueError, AttributeError):
            v = None
        mt = v.get("max_tokens") if isinstance(v, dict) and bool(v.get("gpu")) == (vision == "gpu") else None
        if isinstance(mt, int) and mt > 0 and mt != default:
            asked = mt
    if asked is None:
        return default
    note = ""
    if vision == "cpu" and asked > default:
        note = (" - on the CPU a picture takes longer to encode the more tokens it gets (several seconds more at "
                "1,024 than at 300)")
    elif asked > VISION["gpu"]["max_tokens"]:
        note = " - more than the encoder's default needs more VRAM and context per picture"
    ok(f"images: up to {asked} image tokens per picture (--vision-tokens; default {default}){note}")
    return asked


PARALLEL_MAX = 8               # #465: the engine's batch window holds at most 8 requests


def engine_lib_dirs(eng: Path, toolkit=13) -> list:
    """The library folders a CUDA engine loads from: its own (a compiled one: the toolkit's), else pip's wheels."""
    meta = json.loads((eng / "BUILD.json").read_text(encoding="utf-8"))
    return meta.get("lib_dirs") or meta.get("cuda_dirs") or cuda_lib_dirs(toolkit)


def write_run_script(model, cfg_path, port, open_browser=True):
    """run-<model>.bat / .sh: the server with this config; `open_browser` False (#609: --no-browser) leaves --open out."""
    serve = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
             "--port", str(port)] + (["--open"] if open_browser else [])
    if WIN:
        script = ROOT / f"run-{model.lower()}.bat"
        script.write_text("@echo off\r\ntitle Puddle " + model + "\r\ncd /d \"" + str(ROOT) + "\"\r\n" +
                          " ".join(f'"{x}"' for x in serve) + "\r\nif errorlevel 1 pause\r\n", encoding="utf-8")
    else:
        script = ROOT / f"run-{model.lower()}.sh"
        script.write_text("#!/bin/sh\ncd \"" + str(ROOT) + "\"\nexec " + " ".join(f'"{x}"' for x in serve) + "\n",
                          encoding="utf-8")
        script.chmod(0o755)
    return script


# ------------------------------------------------------------------------------------------------ strata-poly
def poly_config(cfg: dict) -> bool:
    """A run config of strata-poly, under this name or the one it had."""
    return Path(str(cfg.get("exe", ""))).name in (POLY_EXE, OLD_POLY_EXE)


# The estimates below take a size's entry: what its family's "models" says of it (family_models()).
def poly_beside_gb(d) -> float:
    """The GiB of VRAM a size takes before any expert, at a 32K context."""
    return d.get("vram_beside_gb", GLM_VRAM_BESIDE_GB)


def poly_held_gb(d, vram_gb) -> float:
    """About the GB of a size's experts a card with this much VRAM (GiB) holds at a 32K context (a llama-server size,
    which has "vram_need_gb" and no experts in RAM: none)."""
    if "arena_gb" not in d:
        return 0.0
    return min(d["arena_gb"], max(0.0, vram_gb - poly_beside_gb(d)) * 1.073741824)


def poly_parallel_note(asked: int | None, d, vram_gb) -> list[str]:
    """What setup says of "parallel" for a strata-poly size: answers written together pay where the card holds every
    expert of the size.  Where most of them run on the CPU, different requests route to different experts and the
    CPU does each one's: on the 8 GB laptop three Qwen3.6 UD-Q3_K_M chats at once wrote 105 tokens/s together and
    79 one after the other, so there "parallel" gains less than it does on a large card."""
    whole = poly_held_gb(d, vram_gb) >= d["arena_gb"]
    turns = ("on this card most experts run on the CPU and do each request's own, so requests at once gain less: on "
             "an 8 GB laptop three chats at once wrote about a third faster than one after the other (docs/BATCHING.md)")
    if asked is None or asked <= 1:
        if whole:
            return ["Several requests at once (opt-in): --parallel 2 to 4 writes their answers together, each in a "
                    "slot whose attention rows take VRAM from the expert cache as its conversation grows "
                    "(docs/BATCHING.md)."]
        return [f"Several requests at once (opt-in, --parallel N): {turns}."]
    lines = [f"parallel requests: {asked} at once (each slot's attention rows take VRAM from the expert cache as its "
             "conversation grows)"]
    if asked > PARALLEL_MAX:
        lines.append(f"the engine runs at most {PARALLEL_MAX} at once; it will use {PARALLEL_MAX}")
    if not whole:
        lines.append(turns)
    return lines


def poly_batch(fam, d, vram_gb, ctx) -> bool:
    """Whether setup's config reads a prompt 2,048 tokens at a time instead of the engine's 1,024: on a card that holds
    every expert of a size beside the larger batch and the context's rows past 32K, with a GiB to spare, llama.cpp's
    matrix kernels get twice the rows of each expert and a long prompt is read 14% (Qwen3.6) to 30% (LFM2.5) faster;
    on a smaller card the experts outside VRAM are copied to it half as often.  A family without "batch_gb" keeps
    1,024, one with "batch_from_gb" takes 2,048 on any card from that size on."""
    if fam.get("batch_from_gb") and vram_gb >= fam["batch_from_gb"]:
        return True
    if not fam.get("batch_gb"):
        return False
    rows = max(0, ctx - 32768) * fam.get("row_bytes", GLM_ROW_BYTES) / 2**30
    return d["arena_gb"] <= poly_held_gb(d, vram_gb - fam["batch_gb"] - rows - 1)


def poly_default(sizes, ram, vram_gb) -> str:
    """The size setup takes when none is named: the first whose RAM estimate this PC meets, else the one that wants
    the least.  A size's "pick_extra_gb" is RAM it wants beyond that before it is taken unasked."""
    return next((m for m, d in sizes.items() if ram >= poly_ram_need(d, vram_gb) - 4 + d.get("pick_extra_gb", 0)),
                min(sizes, key=lambda m: poly_ram_need(sizes[m], vram_gb)))


def poly_ram_need(d, vram_gb) -> float:
    """About the GB of RAM a size wants on that card: the experts the card does not hold, in the OS file cache,
    and room for the rest.  An estimate from one machine (32 GB of VRAM, 120 GB of RAM).  A llama-server size (no
    experts in RAM): its own ram_gb."""
    if "arena_gb" not in d:
        return float(d["ram_gb"])
    return d["arena_gb"] - poly_held_gb(d, vram_gb) + POLY_RAM_BESIDE_GB


def poly_fits(d, ram, vram_gb, hip) -> bool:
    """Whether a size of strata-poly's models runs on this PC with its experts in RAM, by setup's estimate.  A
    llama-server size ("vram_need_gb"): NVIDIA cards with that much VRAM (`vram_gb`: the chosen cards together)."""
    if "vram_need_gb" in d:
        return not hip and vram_gb >= d["vram_need_gb"] and ram >= d["ram_gb"] - 4
    return vram_gb >= poly_beside_gb(d) + 2 and ram >= poly_ram_need(d, vram_gb) - 4


def poly_verdict(d, ram, vram_gb, hip, windows=False) -> str:
    """--check's line for a size of strata-poly's models.  `windows`: its family has run there (FAMILIES).  A
    llama-server size ("vram_need_gb"): by the VRAM of the chosen cards together."""
    if "vram_need_gb" in d:
        if hip:
            return "not on AMD cards yet (llama-server is compiled here for NVIDIA only)"
        if vram_gb < d["vram_need_gb"]:
            return (f"does not fit: it runs with {d['vram_need_gb']} GB of VRAM or more on the chosen cards together "
                    f"({vram_gb:.0f} here)")
        return ("EXPERIMENTAL, " + ("not tried on Windows, " if WIN else "") +
                f"fits: {vram_gb:.0f} GB of VRAM on the chosen cards, everything on the GPUs")
    if vram_gb < poly_beside_gb(d) + 2:
        return f"does not fit: it takes about {poly_beside_gb(d)} GB of VRAM before any expert"
    need = poly_ram_need(d, vram_gb)
    # an AMD card on Windows: setup compiles the HIP engine there (AMD's ROCm wheels); one RX 9070 XT has run one
    head = ("EXPERIMENTAL, measured on one AMD card on Windows, " if hip and WIN else
            "EXPERIMENTAL, " + ("not tried on Windows, " if WIN and not windows else ""))
    return head + (
        f"fits by setup's estimate (~{need:.0f} GB of RAM on this card)" if poly_fits(d, ram, vram_gb, hip) else
        f"wants ~{need:.0f} GB of RAM on this card: with {ram:.0f} its experts are read from the SSD")


def poly_saved_conv(paths) -> int | str | None:
    """--conversation-cache-mib of an earlier run config of this model (the first of `paths` there is): MiB, or "auto";
    0 when that config has none; None without an earlier config: setup then asks."""
    for p in paths:
        if p is None or not p.is_file():
            continue
        try:
            args = json.loads(p.read_text(encoding="utf-8-sig")).get("args")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(args, list):
            v = str(args[args.index("--conversation-cache-mib") + 1]) if "--conversation-cache-mib" in args[:-1] else ""
            return int(v) if v.isdigit() else "auto" if v == "auto" else 0
    return None


def build_engine_poly(gpu, yes, llama, vision="none", toolkit=13, portable=False) -> Path:
    """Compile strata-poly for this GPU and, for images, the encoder beside it ("gpu": with CUDA, "cpu": without);
    the results go to engine-poly/.  Each is compiled again when its source changed (a `git pull`); the engine also
    when the card is one it has no code for, and the encoder when it is wanted on the GPU and was built without.
    toolkit 12 (--cuda 12): with a CUDA 12.x toolkit, for a driver older than CUDA 13's or a Pascal / Volta card;
    both are compiled again when the ones there were built with the other toolkit.  portable (Windows,
    --max-compatibility): STRATA_PORTABLE, the release build (AVX2 baseline, static MSVC runtime), for other PCs."""
    eng = ROOT / ENGINE_POLY_DIR
    eng.mkdir(exist_ok=True)
    stamp = eng / "BUILD.json"
    meta = json.loads(stamp.read_text()) if stamp.exists() else {}
    src, vsrc = source_hash(ENGINE_SOURCES), vision_hash()
    cards = {int(x) for x in gpu.get("archs", [gpu["arch"]])}
    built = {int(x) for x in meta.get("archs", [])}
    same = int(meta.get("toolkit", 13)) == toolkit
    # one compiled here keeps the cards it was built for; one in place of a ready-made engine is for these cards
    release = meta.get("source") == "release"
    archs = sorted(cards if release or not same else built | cards)
    have = (eng / POLY_EXE).exists()
    portable_same = release or bool(meta.get("portable")) == bool(portable)
    engine_ok = have and same and meta.get("src") == src and (cards <= built if release else set(archs) == built) and \
        portable_same
    # an encoder built with CUDA also runs on the CPU, so the one there is kept unless the GPU is asked of a CPU one
    was = meta.get("vision") or "none"
    vision_ok = vision == "none" or ((eng / VEXE).exists() and meta.get("vision_src") == vsrc and
                                     (was == "gpu" and same or was == "cpu" and vision == "cpu") and
                                     portable_same)
    if engine_ok and vision_ok:
        ok(f"strata-poly already {'downloaded' if release else 'built'} for this PC")
        return eng
    # CUDA 12.8 is the toolkit it was written and measured with, on an RTX 5090: the Qwen engine's reason to want 13.0
    # there (#220, its prompt path) is not this engine's
    nvcc, vcvars = install_build_tools({**gpu, "archs": archs, "toolkit": toolkit,
                                        "need_cuda": (12, 8) if max(archs) >= 120 else (12, 0)}, yes)
    cuda_archs = ";".join(str(x) for x in archs)
    if not engine_ok:
        say("  strata-poly's source changed: compiling it again (only what changed, a few minutes) ..."
            if have and same and set(archs) == built else
            "  Compiling the strata-poly engine for your GPU (some minutes, once) ...")
        # a folder per toolkit: CMake does not take another compiler into a folder configured with one
        bdir = ROOT / ("build-poly" if toolkit == 13 else "build-poly-cuda12")
        defs = ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}",
                f"-DCMAKE_CUDA_COMPILER={nvcc}", f"-DSTRATA_GGML_DIR={llama}",
                f"-DSTRATA_PORTABLE={'ON' if portable else 'OFF'}"]
        if min(archs) < CUDA13_MIN_ARCH:
            defs.append("-DSTRATA_EXPERIMENTAL_SM60=ON")
        cmake_build(ROOT, bdir, "strata-poly", defs, vcvars, "build-strata-poly.bat")
        shutil.copy2(bdir / POLY_EXE, eng / POLY_EXE)
    if not vision_ok:
        say("  Compiling the image encoder" + (" with CUDA (10-20 minutes, once) ..." if vision == "gpu" else " ..."))
        defs = [f"-DLLAMA_DIR={get_llama_cpp(vision=True)}", f"-DSTRATA_VISION_CUDA={'ON' if vision == 'gpu' else 'OFF'}",
                f"-DSTRATA_PORTABLE={'ON' if portable else 'OFF'}"]
        if vision == "gpu":
            defs += [f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}", f"-DCMAKE_CUDA_COMPILER={nvcc}"]
        # its own build folder: the Qwen engine's encoder may be built for another toolkit in build-vision/
        vdir = ROOT / ("build-vision-poly" if toolkit == 13 else "build-vision-poly-cuda12")
        cmake_build(ROOT / "tools" / "vision", vdir, "strata-vision", defs, vcvars, "build-vision-poly.bat")
        shutil.copy2(vdir / "bin" / VEXE, eng / VEXE)
        was, meta = vision, {**meta, "vision_src": vsrc}
    elif vision == "none" and not portable_same:       # the encoder there is of another portability setting: not ours
        was, meta = "none", {**meta, "vision_src": None}
    bindir = Path(nvcc).parent                            # the toolkit's own libraries (bin, bin/x64, lib64)
    dirs = [str(x) for x in (bindir, bindir / "x64", bindir.parent / "lib64") if x.is_dir()]
    if toolkit == 13:                                     # pip's, for a ready-made encoder kept beside it
        dirs += [d for d in cuda_lib_dirs(13) if d not in dirs]
    stamp.write_text(json.dumps({"source": "local", "version": source_version(), "archs": archs, "cuda_dirs": dirs,
                                 "toolkit": toolkit, "src": src, "vision": was, "vision_src": meta.get("vision_src"),
                                 **({"portable": True} if portable else {})},
                                indent=1))
    ok(f"strata-poly compiled: {eng / POLY_EXE}" + (f", image encoder: {eng / VEXE}" if vision != "none" else ""))
    return eng


POLY_TESTS_DIR = "build-poly-tests"


def run_poly_tests(pattern: str = "") -> int:
    """`setup.py --update --tests [REGEX]` (./update.sh --tests): strata-poly and its own tests (STRATA_BUILD_TESTS=ON)
    compiled in a build folder of their own, for the cards and with the toolkit the installed engine was built for,
    then run by ctest: every registered test, or those whose names match REGEX.  The engine in engine-poly/ is not
    touched.  A test that needs a card and finds none is skipped (exit code 77).  Returns ctest's exit code."""
    eng = ROOT / ENGINE_POLY_DIR
    stamp = eng / "BUILD.json"
    meta = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else {}
    if meta.get("backend") == "hip" or meta.get("source") == "release" or not meta.get("archs"):
        fail("--tests: there is no strata-poly compiled here for NVIDIA cards to take the build's settings from",
             "run ./setup.sh (START-HERE.bat) first")
    archs = sorted(int(x) for x in meta["archs"])
    toolkit = int(meta.get("toolkit", 13))
    nvcc, vcvars = install_build_tools({"arch": archs[0], "archs": archs, "toolkit": toolkit,
                                        "need_cuda": (12, 8) if max(archs) >= 120 else (12, 0)}, True)
    bdir = ROOT / (POLY_TESTS_DIR if toolkit == 13 else POLY_TESTS_DIR + "-cuda12")
    defs = ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=ON", f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(map(str, archs))}",
            f"-DCMAKE_CUDA_COMPILER={nvcc}", f"-DSTRATA_GGML_DIR={get_llama_cpp()}",
            f"-DSTRATA_PORTABLE={'ON' if meta.get('portable') else 'OFF'}"]
    if min(archs) < CUDA13_MIN_ARCH:
        defs.append("-DSTRATA_EXPERIMENTAL_SM60=ON")
    say(f"  Compiling strata-poly and its tests in {bdir.name}/ (the engine in {ENGINE_POLY_DIR}/ stays as it is) ...")
    cmake_build(ROOT, bdir, "all", defs, vcvars, "build-poly-tests.bat")
    cmake = find_tool("cmake")                         # ctest comes with it (pip's cmake has both)
    ctest = Path(cmake).parent / ("ctest.exe" if WIN else "ctest") if cmake else None
    if ctest is None or not ctest.exists():
        fail("ctest not found next to cmake", "run: .venv python -m pip install cmake")
    env = dict(os.environ)
    dirs = [str(d) for d in (meta.get("cuda_dirs") or [])]
    var = "PATH" if WIN else "LD_LIBRARY_PATH"
    env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    r = run([ctest, "--test-dir", str(bdir), "--output-on-failure", *(["-R", pattern] if pattern else [])],
            env=env, check=False)
    (ok if r.returncode == 0 else warn)("strata-poly's tests " + ("passed" if r.returncode == 0 else
                                       f"did not all pass (ctest exit {r.returncode}): the failures are above"))
    return r.returncode


def build_engine_poly_hip(gpu, llama, portable=False) -> Path:
    """Compile strata-poly for this AMD GPU into engine-poly/ (again only when its source changed: a `git pull`).
    Windows: AMD's TheRock Windows wheels in .rocm-win (build_engine_poly_hip_win; portable: see that function).
    Linux: ROCm from rocm_root: a system ROCm 7 or AMD's TheRock wheels in .venv, as for the main engine's HIP
    build.  There is no ready-made AMD engine and no HIP image encoder: images stay off."""
    if WIN:
        return build_engine_poly_hip_win(gpu, llama, portable)
    eng = ROOT / ENGINE_POLY_DIR
    eng.mkdir(exist_ok=True)
    stamp = eng / "BUILD.json"
    meta = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else {}
    src = source_hash(ENGINE_SOURCES)
    archs = sorted(set(gpu.get("archs") or [gpu["arch"]]))
    engine_ok = meta.get("backend") == "hip" and (eng / POLY_EXE).exists() and meta.get("src") == src and \
        set(archs) <= set(meta.get("archs", [])) and bool(meta.get("portable")) == bool(portable)
    if engine_ok:
        ok("strata-poly already built for this PC")
        return eng
    if not (shutil.which("c++") or shutil.which("g++")) or not shutil.which("git"):
        fail("a C++ compiler and git are needed to compile the AMD strata-poly",
             "Ubuntu/Debian: sudo apt install build-essential git   Fedora: sudo dnf install gcc-c++ git")
    root, dirs = rocm_root(archs)
    libs = [str(Path(d).parent) for d in dirs[1:]]
    bitcode = next((p for p in (root / "lib" / "llvm" / "amdgcn" / "bitcode", root / "amdgcn" / "bitcode") if p.is_dir()),
                   root / "amdgcn" / "bitcode")
    os.environ.update({"HIP_PLATFORM": "amd", "HIP_COMPILER": "clang", "HIP_RUNTIME": "rocclr", "ROCM_PATH": str(root),
                       "HIP_PATH": str(root)})
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + [os.environ.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    os.environ["PATH"] = os.pathsep.join([str(root / "bin"), str(root / "llvm" / "bin"), os.environ.get("PATH", "")])
    say("  strata-poly's source changed: compiling it again (only what changed, a few minutes) ..."
        if meta.get("backend") == "hip" and (eng / POLY_EXE).exists()
        else f"  Compiling the strata-poly engine for your AMD GPU ({', '.join(archs)}; some minutes, once) ...")
    cmake_build(ROOT, ROOT / "build-poly-hip", "strata-poly",
                ["-DSTRATA_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=OFF", "-DSTRATA_BUILD_TESTS=OFF",
                 f"-DSTRATA_PORTABLE={'ON' if portable else 'OFF'}",
                 "-DCMAKE_HIP_ARCHITECTURES=" + ";".join(archs),
                 f"-DCMAKE_HIP_COMPILER={root / 'llvm' / 'bin' / 'clang++'}", f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={root}",
                 "-DCMAKE_PREFIX_PATH=" + ";".join([str(root), *libs]),
                 f"-DCMAKE_HIP_FLAGS=--rocm-path={root} --rocm-device-lib-path={bitcode}",
                 f"-DSTRATA_GGML_DIR={llama}"], None, "")
    shutil.copy2(ROOT / "build-poly-hip" / POLY_EXE, eng / POLY_EXE)
    stamp.write_text(json.dumps({"source": "local-hip", "backend": "hip", "version": source_version(),
                                 "archs": archs, "vision": "none", "lib_dirs": dirs, "src": src,
                                 **({"portable": True} if portable else {})}, indent=1))
    ok(f"strata-poly compiled: {eng / POLY_EXE}")
    return eng


def build_engine_poly_hip_win(gpu, llama, portable=False) -> Path:
    """Compile strata-poly for this AMD GPU on Windows into engine-poly/ (again only when its source changed: a
    `git pull`).  ROCm comes from AMD's TheRock Windows wheels in .rocm-win (rocm_win_root), MSVC's headers, linker
    and runtime from the Visual Studio C++ build tools, and ROCm's clang compiles the host code too (CMake refuses to
    mix it with cl.exe).  The DLLs and kernels it loads are .rocm-win's own: the
    config's lib_dirs points there, so the libraries are not on the disk twice.
    portable (--max-compatibility): STRATA_PORTABLE, the release build (AVX2 baseline, static MSVC runtime), for other
    PCs.  There is no HIP image encoder: images stay off."""
    eng = ROOT / ENGINE_POLY_DIR
    eng.mkdir(exist_ok=True)
    stamp = eng / "BUILD.json"
    meta = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else {}
    src = source_hash(ENGINE_SOURCES)
    archs = sorted(set(gpu.get("archs") or [gpu["arch"]]))
    engine_ok = meta.get("backend") == "hip" and (eng / POLY_EXE).exists() and meta.get("src") == src and \
        set(archs) <= set(meta.get("archs", [])) and bool(meta.get("portable")) == bool(portable)
    if engine_ok:
        ok("strata-poly already built for this PC")
        return eng
    vcvars = find_vcvars()
    if vcvars is None:
        fail("the Visual Studio C++ build tools were not found",
             "install them (Visual Studio 2019 or 2022, workload 'Desktop development with C++') and run setup again")
    root, _ = rocm_win_root(archs)
    rocm = str(root).replace("\\", "/")
    bitcode = f"{rocm}/lib/llvm/amdgcn/bitcode"
    os.environ.update({"HIP_PLATFORM": "amd", "HIP_PATH": str(root), "ROCM_PATH": str(root)})
    os.environ["PATH"] = os.pathsep.join([str(root / "bin"), str(root / "lib" / "llvm" / "bin"),
                                          os.environ.get("PATH", "")])
    say("  strata-poly's source changed: compiling it again (only what changed, a few minutes) ..."
        if meta.get("backend") == "hip" and (eng / POLY_EXE).exists()
        else f"  Compiling the strata-poly engine for your AMD GPU ({', '.join(archs)}; 10-20 minutes, once) ...")
    bdir = ROOT / "build-poly-hip-win"
    cmake_build(ROOT, bdir, "strata-poly",
                ["-DSTRATA_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=OFF", "-DSTRATA_BUILD_TESTS=OFF",
                 f"-DSTRATA_PORTABLE={'ON' if portable else 'OFF'}",
                 "-DCMAKE_HIP_ARCHITECTURES=" + ";".join(archs),
                 f"-DCMAKE_C_COMPILER={rocm}/lib/llvm/bin/clang.exe",
                 f"-DCMAKE_CXX_COMPILER={rocm}/lib/llvm/bin/clang++.exe",
                 f"-DCMAKE_HIP_COMPILER={rocm}/lib/llvm/bin/clang++.exe",
                 f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={rocm}", f"-DCMAKE_PREFIX_PATH={rocm}",
                 f"-DCMAKE_HIP_FLAGS=--rocm-path={rocm} --rocm-device-lib-path={bitcode}",
                 f"-DSTRATA_GGML_DIR={llama}"], vcvars, "build-strata-poly-hip.bat")
    shutil.copy2(bdir / POLY_EXE, eng / POLY_EXE)
    # the DLLs, the GEMM kernels and the .kpack files rocBLAS finds beside its DLL: .rocm-win's (rocm/bin)
    dirs = [str(root / "bin")]
    stamp.write_text(json.dumps({"source": "local-hip", "backend": "hip", "platform": "windows-x64",
                                 "version": source_version(), "archs": archs, "rocm": ROCM_WIN_VERSION,
                                 "hipblaslt_version": hipblaslt_version([str(root / "bin")]), "vision": "none",
                                 "lib_dirs": dirs, "src": src,
                                 **({"portable": True} if portable else {})}, indent=1))
    hip_runtime_beside_exe(eng)                        # #468 #461: before System32's amdhip64_7.dll
    ok(f"strata-poly compiled: {eng / POLY_EXE}")
    return eng


def update_poly_engine(cfg: dict, yes=False, cfg_path: Path | None = None) -> None:
    """Before a model of strata-poly starts: the engine setup installed is replaced when its source changed (a `git
    pull`) or the card is another generation, a ready-made one by the release's for the new source, one compiled here
    by a compile.  A ready-made one stays while no release has the new source: a compile would first install the
    build tools.  A failure must not stop the start: the installed one runs.  An install from before the engine was
    renamed has the old engine in its config: the engine is compiled under its new name and, with cfg_path, the
    config is moved to it."""
    eng = Path(cfg["exe"]).parent
    renamed = eng == ROOT / OLD_ENGINE_POLY_DIR and Path(cfg["exe"]).name == OLD_POLY_EXE
    if renamed:
        eng = ROOT / ENGINE_POLY_DIR
    stamp = eng / "BUILD.json"
    if eng != ROOT / ENGINE_POLY_DIR or not (stamp.exists() or renamed):   # an engine of the user's own
        return
    try:
        meta = json.loads(stamp.read_text()) if stamp.exists() else {}
        if meta.get("backend") == "hip":   # the AMD engine compiled here: the same way again when the source moved
            if not (meta.get("src") == source_hash(ENGINE_SOURCES) and (eng / POLY_EXE).exists()):
                build_engine_poly_hip({"arch": (meta.get("archs") or [""])[0], "archs": meta.get("archs")},
                                      get_llama_cpp(), bool(meta.get("portable")))
                cfg["lib_dirs"] = engine_lib_dirs(eng)
            if renamed:
                cfg["exe"] = str(eng / POLY_EXE)
                if cfg_path is not None:
                    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
            return
        picks = cfg["gpu"] if isinstance(cfg.get("gpu"), list) else [cfg["gpu"] if isinstance(cfg.get("gpu"), int) else None]
        gpu = gpu_info(picks[0])
        if gpu is None:
            raise RuntimeError("no NVIDIA GPU found")
        gpu["archs"] = sorted({int(gpu_info(i)["arch"]) for i in picks})   # a layer split's cards
        # the encoder too, when this model has one: on the device it was built for
        vis = cfg.get("vision")
        vision = ("gpu" if meta.get("vision") == "gpu" or vis.get("gpu") else "cpu") if isinstance(vis, dict) else "none"
        changed = renamed
        fits = set(gpu["archs"]) <= {int(x) for x in meta.get("archs", [])}
        if not (meta.get("src") == source_hash(ENGINE_SOURCES) and (eng / POLY_EXE).exists() and fits
                and (vision == "none" or meta.get("vision_src") == vision_hash())):
            build_engine_poly(gpu, yes, get_llama_cpp(), vision, int(meta.get("toolkit", 13)),
                              bool(meta.get("portable")))
            # a ready-made engine loads CUDA from pip, one compiled here from the toolkit it was compiled with
            cfg["lib_dirs"], changed = engine_lib_dirs(eng), True
        if changed:
            cfg["exe"] = str(eng / POLY_EXE)
            if cfg_path is not None:
                cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        if renamed:
            ok(f"this install's engine is {POLY_EXE} now (it was {OLD_POLY_EXE}); {OLD_ENGINE_POLY_DIR}/ can be deleted")
    except (Exception, SystemExit) as e:
        warn(f"could not update strata-poly{'' if isinstance(e, SystemExit) else f' ({e})'}: starting the installed one")


def poly_tail_skip(cfg: dict, cfg_path: Path | None = None) -> None:
    """A config written before its family got "tail_skip" gets the option, once the installed engine knows it (0.1.42).
    A config that has --tail-skip, with any value, keeps it."""
    a = cfg["args"]
    fam = next((f for f in FAMILIES.values() if f.get("tail_skip")
                and str(cfg.get("model_name", "")).startswith(f["name"] + "-")), None)
    if fam is None or "--tail-skip" in a or engine_version(Path(cfg["exe"])) < (0, 1, 42):
        return
    a += ["--tail-skip", str(fam["tail_skip"])]
    ok(f"{fam['title']}: --tail-skip {fam['tail_skip']}, a few experts outside VRAM are left out (faster on a small "
       "card, the answers move a little; STRATA_ROUTE_TAIL_SKIP=0 turns it off)")
    if cfg_path is not None:
        write_config(cfg_path, cfg)


def poly_kv_whole(cfg: dict, cfg_path: Path | None = None) -> None:
    """A config of strata-poly on an AMD card gets --kv-whole: ROCm keeps the VRAM that cuMemUnmap() and
    cuMemRelease() gave back (issue #13), so the attention caches take theirs at load instead of as the
    conversation grows, where the expert cache's VRAM would be lost to them.  A config that has it keeps it."""
    a = cfg["args"]
    try:
        meta = json.loads((Path(cfg["exe"]).parent / "BUILD.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if meta.get("backend") != "hip" or "--kv-whole" in a:
        return
    a.append("--kv-whole")
    ok("AMD card: --kv-whole, the attention caches take their VRAM at load (ROCm keeps what the expert cache gives "
       "back, so a long prompt was refused VRAM later)")
    if cfg_path is not None:
        write_config(cfg_path, cfg)


# ------------------------------------------------------------------------------------------------ llama-server
# A family with "engine": "llama-server" (nemotron-bolt-llama) runs on llama.cpp's llama-server at LLAMA_SERVER_COMMIT,
# compiled here like strata-poly (always from source) into engine-llama/.  Its run config has "engine": "llama-server"
# and llama-server's own arguments; the server runs it on 127.0.0.1 with a port and a key of its own
# (serve/server.py's LlamaServerEngine).  NVIDIA cards only for now: it has been run on CUDA only.
def llama_config(cfg: dict) -> bool:
    """A run config of llama-server (Nemotron 3.5 Bolt on llama-server)."""
    return cfg.get("engine") == "llama-server"


def build_llama_server(gpu, yes, toolkit=13) -> Path:
    """Compile llama-server at LLAMA_SERVER_COMMIT for these NVIDIA cards into engine-llama/: once, and again when the
    commit changes (a `git pull` that moves it) or a card needs code it was not built with.  One file, the CUDA
    backend linked in (BUILD_SHARED_LIBS off), without llama.cpp's own chat app (Puddle has its own), without
    OpenSSL (it downloads nothing).  The CUDA libraries come from the toolkit it was compiled with (BUILD.json)."""
    eng = ROOT / ENGINE_LLAMA_DIR
    eng.mkdir(exist_ok=True)
    stamp = eng / "BUILD.json"
    meta = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else {}
    cards = {int(x) for x in gpu.get("archs", [gpu["arch"]])}
    built = {int(x) for x in meta.get("archs", [])}
    same = meta.get("commit") == LLAMA_SERVER_COMMIT and int(meta.get("toolkit", 13)) == toolkit
    if (eng / LLAMA_SERVER_EXE).exists() and same and cards <= built:
        ok(f"llama-server already built for this PC (llama.cpp {LLAMA_SERVER_COMMIT[:9]})")
        return eng
    archs = sorted(cards | built if same else cards)
    nvcc, vcvars = install_build_tools({**gpu, "archs": archs, "toolkit": toolkit,
                                        "need_cuda": (12, 8) if max(archs) >= 120 else (12, 0)}, yes)
    src = get_llama_cpp(server=True)
    ok(f"llama.cpp {LLAMA_SERVER_COMMIT[:9]} (llama-server)")
    say("  Compiling llama-server for your GPU (llama.cpp's CUDA kernels, once; it takes a while) ..."
        if not (eng / LLAMA_SERVER_EXE).exists() else
        "  llama-server's llama.cpp changed, or a card needs code it has not: compiling it again ...")
    bdir = ROOT / ("build-llama-server" if toolkit == 13 else "build-llama-server-cuda12")
    defs = ["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(str(x) for x in archs)}",
            f"-DCMAKE_CUDA_COMPILER={nvcc}", "-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=ON", "-DLLAMA_BUILD_UI=OFF",
            "-DLLAMA_USE_PREBUILT_UI=OFF", "-DLLAMA_OPENSSL=OFF", *host_compiler_defs(True)]
    cmake_build(src, bdir, "llama-server", defs, vcvars, "build-llama-server.bat")
    shutil.copy2(bdir / "bin" / LLAMA_SERVER_EXE, eng / LLAMA_SERVER_EXE)
    bindir = Path(nvcc).parent                            # the toolkit's own libraries (bin, bin/x64, lib64)
    dirs = [str(x) for x in (bindir, bindir / "x64", bindir.parent / "lib64") if x.is_dir()]
    stamp.write_text(json.dumps({"source": "local", "engine": "llama-server", "commit": LLAMA_SERVER_COMMIT,
                                 "version": f"llama.cpp {LLAMA_SERVER_COMMIT[:9]}", "archs": archs, "cuda_dirs": dirs,
                                 "toolkit": toolkit}, indent=1))
    ok(f"llama-server compiled: {eng / LLAMA_SERVER_EXE}")
    return eng


def update_llama_server(cfg: dict, yes=False, cfg_path: Path | None = None) -> None:
    """Before a llama-server model starts (and on --update): llama-server is compiled again when LLAMA_SERVER_COMMIT
    moved or the model's cards need code it was not built with; nothing happens otherwise."""
    eng = Path(cfg["exe"]).parent
    try:
        meta = json.loads((eng / "BUILD.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    found = gpus()
    use = cfg.get("gpu")
    sel = use if isinstance(use, list) else [use] if isinstance(use, int) else \
        [max(found, key=lambda x: (round(x["vram_gb"]), -x["index"]))["index"]] if found else []
    archs = sorted({int(g["arch"]) for g in found if g["index"] in sel}) or [int(x) for x in meta.get("archs", [])]
    if (eng / LLAMA_SERVER_EXE).exists() and meta.get("commit") == LLAMA_SERVER_COMMIT and \
            set(archs) <= {int(x) for x in meta.get("archs", [])}:
        return
    if not found:
        warn("llama-server should be compiled again, but no NVIDIA GPU answered: left as it is")
        return
    gpu = {**gpu_info(sel[0] if sel else None), "archs": archs}
    built = build_llama_server(gpu, yes, 12 if config_toolkit(cfg) == 12 else 13)
    dirs = engine_lib_dirs(built)
    if cfg_path is not None and (cfg.get("lib_dirs") != dirs or Path(cfg["exe"]).parent != built):
        cfg["lib_dirs"], cfg["exe"] = dirs, str(built / LLAMA_SERVER_EXE)
        write_config(cfg_path, cfg)


def llama_server_setup(a, family, gpu, hip, cuda_tk, multi, ram, data, elsewhere, port, adopted) -> int:
    """Steps 2 to 7 for a model of llama-server (Nemotron 3.5 Bolt on llama-server): the choices (cards, context,
    pictures), the Python packages, llama-server (compiled here), the model file and its image encoder, its
    tokenizer, the run config and the start script."""
    fam = FAMILIES[family]
    title = fam["title"]
    if hip:
        fail(f"{title} runs on NVIDIA cards for now: llama-server is compiled here with CUDA only, and it has not been "
             "run on an AMD card", "use an NVIDIA card (--backend cuda)")
    if WIN:
        warn(f"{title} on Windows: llama-server is compiled here with MSVC and CUDA as strata-poly is; NVIDIA lists "
             "Windows as supported, but it has not been run there by anyone here (untested)")
    unused = [flag for flag, given in (
        ("--vision-tokens", a.vision_tokens is not None), ("--layer-split", bool(a.layer_split)),
        ("--vram-reserve-mib", a.vram_reserve_mib is not None), ("--parallel", a.parallel is not None),
        ("--replicas", a.replicas is not None), ("--max-compatibility", bool(a.max_compatibility))) if given]
    if unused:
        warn(f"not used with {title} (llama-server has no such setting here): " + ", ".join(unused))
    sizes = family_models(family)
    model = a.model or next(iter(sizes))
    if model not in sizes:
        fail(f"{title} has no {model} model file", "choose one of: " + ", ".join(sizes))
    d = sizes[model]
    sel = multi or [gpu["index"]]
    cards = {g["index"]: g for g in gpus()}
    vram = sum(cards[i]["vram_gb"] for i in sel if i in cards) or gpu["vram_gb"]
    need = d["vram_need_gb"]
    if vram < need:
        fail(f"{title} runs with {need} GB of VRAM or more on the chosen cards together (a 32 GB card, or two 16 GB "
             f"ones); {' + '.join(gpu_name(cards[i]) for i in sel if i in cards) or gpu_name(gpu)} have "
             f"{vram:.0f} GB", "choose more cards with --gpus, or another model")
    ok(f"size: {model} ({d['about']}), on {vram:.0f} GB of VRAM" + (f" across GPUs {sel}" if multi else ""))
    tag = fam["tag"] + model

    # the context: echo's measured one on two 16 GB cards; elsewhere llama-server's --fit takes the largest that fits
    echo_cards = len(sel) == 2 and all(i in cards and round(cards[i]["vram_gb"]) == 16 for i in sel)
    if a.context:
        ctx = a.context
        if not 4096 <= ctx <= 262144:
            fail(f"--context {ctx}: {title} takes 4096 to 262144 tokens")
        ok(f"context: {ctx} tokens (--context)")
    elif echo_cards:
        ctx = fam["echo_context"]
        ok(f"context: {ctx} tokens (as it runs on 2x RTX 5060 Ti 16 GB, where 95% and 93% of the two cards' VRAM is "
           "then in use)")
    else:
        ctx = None
        ok("context: the largest that fits these cards, chosen by llama-server each time it loads (its --fit); "
           "--context N sets one")

    # pictures: the encoder (1.7 GB, BF16) sits on the GPUs beside the model, as on echo
    if a.vision:
        vision = a.vision not in ("no", "none")
    else:
        say()
        say(f"  Images: {title} can read pictures (screenshots, photos, scanned pages). This adds a "
            f"{fam['vision_gb']} GB download, and its image encoder takes VRAM beside the model (as on echo).")
        vision = ask("Do you want images?", ["y", "n"], "y", a.yes) == "y"
    ok("images: " + ("on" if vision else "off"))

    ftag = files_tag(fam, tag, model)                  # Bolt's files: the strata-poly family's folder
    models_dir = Path(a.gguf_dir) if a.gguf_dir else Path(a.models_dir) / ftag
    gguf = models_dir / model_file(fam, model, 1)
    mmproj = models_dir / fam["mmproj"] if vision else None
    if a.gguf_dir:
        for f in (gguf, mmproj):
            if f is not None and not f.exists():
                fail(f"{models_dir} has no {f.name}",
                     "put the file there under that name, or leave out --gguf-dir and setup downloads it")
    elif not (gguf.exists() and done(gguf)):
        for r in elsewhere:                            # already downloaded in a Puddle folder on another drive
            cand = r / "models" / ftag / gguf.name
            if cand.exists() and done(cand):
                models_dir, gguf = cand.parent, cand
                mmproj = models_dir / fam["mmproj"] if vision else None
                ok(f"model file found in {models_dir}")
                break
    have_model = gguf.exists() and (done(gguf) or bool(a.gguf_dir))
    have_mmproj = mmproj is None or mmproj.exists() and (done(mmproj) or bool(a.gguf_dir))
    need_gb = (0 if have_model else d["download_gb"]) + (0 if have_mmproj else fam["vision_gb"]) + 2
    if free_gb(models_dir) < need_gb:
        fail(f"not enough free disk space in {models_dir}: need ~{need_gb:.0f} GB",
             "use --models-dir on a bigger drive")

    step(3, "Python packages")
    pip_install(requirement_lines() if REQUIREMENTS.exists() else PY_PACKAGES,
                "numpy, jinja2, regex, pyyaml, tqdm, requests, cmake, ninja, pillow, psutil")

    step(4, "llama-server")
    eng = build_llama_server({**gpu, "archs": sorted({int(cards[i]["arch"]) for i in sel if i in cards}
                                                     or {int(gpu["arch"])})}, a.yes, cuda_tk)
    lib_dirs = engine_lib_dirs(eng)
    ok(f"engine: {eng / LLAMA_SERVER_EXE}")

    step(5, f"downloading {title} {model}")
    if not have_model:
        say(f"  The model file goes in {models_dir}")
        say(f"  A file you already have: put it there with its original name ({gguf.name}), or use --gguf-dir "
            "<its folder>.")
        download(d.get("hf", fam["hf"]) + gguf.name, gguf)
    check_shards([gguf])
    verify_sha256(gguf, *fam["sha256"][gguf.name], own=bool(a.gguf_dir))
    if mmproj is not None:
        if not have_mmproj:
            download(fam["mmproj_hf"] + fam["mmproj"], mmproj, "image encoder")
        verify_sha256(mmproj, *fam["sha256"][mmproj.name], own=bool(a.gguf_dir))
        ok(f"image encoder: {mmproj}")
    ok("model file present")

    step(6, "preparing the model for Puddle")
    pack = find_in([data, *elsewhere], f"packs/{tag.lower()}") or data / "packs" / tag.lower()
    if not (pack / "tokenizer" / "vocab.json").exists():
        run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"), "--gguf", str(gguf), "--out", str(pack)])
    ok(f"model prepared: {pack} (its tokenizer and chat template)")

    step(7, "writing the start script")
    # echo's command (2026-11-20), in llama-server's long spellings: one slot, flash attention, a q8_0 KV cache,
    # 2,048-token batches, the file read in whole (--load-mode none), every layer on the GPUs, split evenly
    args = ["--model", str(gguf)] + (["--mmproj", str(mmproj)] if mmproj is not None else []) + \
        (["--ctx-size", str(ctx)] if ctx else []) + \
        ["--parallel", "1", "--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
         "--batch-size", "2048", "--ubatch-size", "2048", "--load-mode", "none", "--n-gpu-layers", "all"]
    if multi:
        args += ["--tensor-split", ",".join("1" for _ in multi)]
    cfg_path = ROOT / f"strata-{tag.lower()}.json"
    cfg = {"engine": "llama-server", "exe": str(eng / LLAMA_SERVER_EXE), "args": args, "cwd": str(ROOT),
           "tokenizer": str(pack / "tokenizer"), "model_name": f"{fam['name']}-{model.lower()}",
           "display_name": f"{title} {model}", "log": str(ROOT / f"strata-{tag.lower()}.log"), "lib_dirs": lib_dirs,
           "port": port}
    if multi:                                          # the server shows these cards, in this order, to llama-server
        cfg["gpu"], cfg["gpus_asked"] = multi, True
        ok(f"split evenly across GPUs {multi} (--tensor-split)")
    elif gpu["count"] > 1 or a.gpu is not None:
        cfg["gpu"], cfg["gpus_asked"] = gpu["index"], True
    if cuda_tk == 12:
        cfg["cuda"] = 12
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if a.browser is not None:
        cfg["open_browser"] = a.browser
    earlier = adopted if adopted is not None and adopted.name == cfg_path.name else None
    write_setup_config(cfg_path, cfg, earlier, sampling=fam.get("sampling"))
    script = write_run_script(tag, cfg_path, port, cfg.get("open_browser") is not False)
    ok(f"start script: {script.name}")

    say()
    say("All set.")
    say(f"  API (OpenAI):     http://127.0.0.1:{port}/v1   (any API key; model name: anything)")
    say(f"  API (Anthropic):  http://127.0.0.1:{port}/v1/messages")
    if vision:
        say("  Images:           send them in the chat page, in chat.py (/image <path>) or over the API")
    if a.host and a.host not in ("127.0.0.1", "localhost"):
        say(f"  Other devices:    the server window prints this PC's address (http://<IP>:{port}/)"
            + ("" if a.api_key else " - no API key set: anyone on your network can use it"))
    say(f"  Next time:        just run {'START-HERE.bat' if WIN else './setup.sh'} (or {script.name}) - it starts right away")
    if a.no_start:
        return 0
    return start(cfg_path, port)


def poly_setup(a, family, gpu, hip, cuda_tk, multi, ram, data, elsewhere, port, adopted) -> int:
    """Steps 2 to 7 for a model of strata-poly (docs/BACKENDS.md): its questions, the Python packages, the engine
    (compiled here), the model file, its tokenizer and expert profile, the run config and the start script."""
    _sg = getattr(a, "_custom_gguf_shards", None)
    fam = FAMILIES[family]
    title = fam["title"]
    if hip and WIN:   # the engine is compiled here from AMD's ROCm Windows wheels (rocm_win_root)
        warn(f"{title} on AMD cards on Windows: EXPERIMENTAL.  Setup compiles the HIP engine here (AMD's ROCm "
             "wheels); upstream ran Gemma 4 with it on an RX 5700 XT; the log says where it stops if it does")
    if cuda_tk == 12:
        warn(f"{title}'s engine is compiled here with CUDA 12" +
             (": on Pascal / Volta it is untested" if int(gpu["arch"]) < CUDA13_MIN_ARCH else ""))
    if WIN and not fam.get("windows"):
        confirm_risk(f"{title} has been run on Linux only: its engine compiles on Windows, the model was not tried "
                     "there", bool(a.family), a.yes, f"{title} has not been tried on Windows",
                     f"use Linux, or --family {family} --yes to try it here")
    sees = fam.get("vision") is True
    if sees and hip:   # the image encoder is a CUDA program: no HIP build of it yet (as for the main engine)
        say("  Images: the image encoder is a CUDA program: on the AMD card images are off (text answers).")
        sees = False
    unused = [flag for flag, given in (
        ("--vision", not sees and a.vision not in (None, "no", "none")),
        ("--vision-tokens", not sees and a.vision_tokens is not None),
        ("--layer-split", a.layer_split and not multi)) if given]
    if unused:
        warn(f"not used with {title} (its engine has no such setting): " + ", ".join(unused))

    sizes = family_models(family)
    names = list(sizes)
    if a.model and a.model not in names:
        fail(f"{title} has no {a.model} model file", "choose one of: " + ", ".join(names))
    vram = gpu["vram_gb"]
    if multi:   # each card has its layers' dense weights; its own scratch and staging buffer cost ~2 GB a card more
        cards = {g["index"]: g for g in (amd_gpus() if hip else gpus())}   # (a family's "vram_card_gb": measured)
        vram = sum(cards[i]["vram_gb"] for i in multi) - fam.get("vram_card_gb", 2.0) * (len(multi) - 1)
    model = a.model or poly_default(sizes, ram, vram)
    if not a.model and len(names) > 1:
        say()
        say("  Which size?  The RAM is setup's estimate for this card: the experts it does not hold stay in RAM.")
        for i, m in enumerate(names, 1):
            say(f"  {i}) {m:14s} {sizes[m]['about']}; download {sizes[m]['download_gb']:.0f} GB, "
                f"~{poly_ram_need(sizes[m], vram):.0f} GB of RAM")
        model = names[int(ask("Size?", [str(i) for i in range(1, len(names) + 1)], str(names.index(model) + 1),
                              a.yes)) - 1]
    d = sizes[model]
    # images: the encoder on the GPU takes VRAM the experts would have, so a small card gets it on the CPU
    vision = "none"
    if sees:
        auto = "gpu" if vram >= POLY_VISION_GPU_GB else "cpu"
        if a.vision:
            vision = {"yes": auto, "no": "none"}.get(a.vision, a.vision)
        else:
            say()
            say(f"  Images: the model can also read pictures (screenshots, photos, scanned pages). This adds a "
                f"{fam['vision_gb']} GB")
            say("  download and a compile of the image encoder, which " +
                (f"keeps ~{fam['vision_vram_gb']} GB of VRAM (--vision cpu: none, a few seconds a picture)."
                 if auto == "gpu" else
                 f"runs on the CPU on this card (a few seconds a picture; --vision gpu: ~{fam['vision_vram_gb']} "
                 "GB of VRAM)."))
            # a family whose pictures are its main use answers yes by default (Bolt: "vision_default_yes")
            first = "y" if fam.get("vision_default_yes") else "n"
            vision = auto if ask("Do you want images?", ["y", "n"], first, a.yes) == "y" else "none"
        ok("images: " + {"none": "off", "gpu": "on", "cpu": "on (encoder on the CPU)"}[vision])
    room = max(0.0, vram - (fam["vision_vram_gb"] if vision == "gpu" else 0))   # the VRAM left for the model
    beside = poly_beside_gb(d)
    held, need_ram = poly_held_gb(d, room), poly_ram_need(d, room)
    say()
    say(f"  {model}: {d['about']}; download {d['download_gb']:.0f} GB. The file is read in place: about {held:.0f} of")
    say(f"  its {d['arena_gb']:.0f} GB of experts fit {'these cards' if multi else 'this card'}, the rest stay in RAM "
        "(the OS file cache).")
    if room < beside + 2:
        enc = vision == "gpu"                          # the encoder's share may be what does not fit
        confirm_risk(f"{title} takes about {beside} GB of VRAM before any expert (measured on a 32 GB "
                     f"card at a 32K context)" + (f" and its image encoder {fam['vision_vram_gb']} GB" if enc else "") +
                     f"; {gpu_name(gpu)} has less than it takes to start", bool(a.model), a.yes,
                     f"{title} needs more VRAM than {vram:.0f} GB",
                     ("--vision cpu keeps the image encoder off the card; " if enc else "") +
                     f"--family {family} --model {model} --yes installs it anyway")
        warn(f"going on with {vram:.0f} GB of VRAM, as you chose")
    elif vram < 30:
        ran = fam.get("ran_on_gb", 32)                 # a card of 8 GB reports 8,188 MiB
        untested = f"{title} has run on no card under {ran} GB. " if round(vram) < ran else ""
        if held < d["arena_gb"]:                       # some experts stay in RAM (Bolt's all fit two 16 GB cards)
            warn(untested + f"On {vram:.0f} GB about {100 * held / d['arena_gb']:.0f}% of {title}'s experts fit in "
                 f"VRAM ({100 * poly_held_gb(d, 31.84) / d['arena_gb']:.0f}% on a 32 GB card): more of every answer "
                 "is computed on the CPU, so it is slower")
        elif untested:
            warn(untested.rstrip())
    if ram < need_ram - 4:
        confirm_risk(f"RAM: {ram:.0f} GB. {title} wants about {need_ram:.0f} GB on this card by setup's estimate: the "
                     f"{d['arena_gb'] - held:.0f} GB of experts the GPU does not hold are read through the OS file "
                     "cache. With less RAM they come from the SSD while it answers: expect it to be much slower (not "
                     "measured)", bool(a.model), a.yes,
                     f"{title} wants about {need_ram:.0f} GB of RAM; this PC has {ram:.0f}",
                     f"--family {family} --model {model} --yes installs it anyway")
        warn(f"going on with {ram:.0f} GB of RAM, as you chose")
    ok(f"size: {model}")
    tag = fam["tag"] + custom_gguf_tag(_sg, model)

    if a.context:
        ctx = a.context
        if not 16 <= ctx <= 1048576:
            fail(f"--context {ctx}: {title} takes 16 to 1048576 tokens")
    else:
        say()
        say("  Context length = how much text the model can see at once (your chat, files, tool output).")
        say("  Longer takes VRAM from the expert cache:")
        row = fam.get("row_bytes", GLM_ROW_BYTES)
        # writing after a prompt that fills the context, tokens/s, on the 8 GB laptop (docs/BACKENDS.md)
        laptop = fam.get("laptop_tps") or ({32768: "52-58", 65536: "42-46", 131072: "28-30"}
                                           if fam.get("ran_on_gb") == 8 else {})
        contexts = fam.get("contexts", POLY_CONTEXTS)
        for i, c in enumerate(contexts, 1):
            on8 = f"; 8 GB card: {laptop[c]} tokens/s after a full prompt" if c in laptop else ""
            note = f"   (recommended{on8})" if c == 32768 else \
                f"   ({(c - 32768) * row / 1e9:.1f} GB more VRAM{on8 or '; no long prompt run'})" if c > 32768 else ""
            say(f"  {i}) {c // 1024}K tokens{note}")
        ctx = contexts[int(ask("Context?", [str(i) for i in range(1, len(contexts) + 1)],
                               str(contexts.index(32768) + 1), a.yes)) - 1]
    ok(f"context: {ctx} tokens")

    # the conversation cache: asked once, then kept as the model's run config has it
    cfg_path = ROOT / f"strata-{tag.lower()}.json"
    conv = poly_saved_conv([cfg_path, adopted if adopted is not None and adopted.name == cfg_path.name else None])
    if conv is None:
        say()
        say("  Conversation cache: a coding agent sends several conversations in turn (the main one, its subagents,")
        say("  short side requests). With the cache, one that makes room for another is kept in RAM and comes back")
        say(f"  without being read again. It takes what it holds, up to {POLY_CONV_CACHE_MIB // 1024} GB of RAM, and")
        say("  less when little RAM is left free once the model is loaded.")
        room = ram >= need_ram + POLY_CONV_CACHE_MIB / 1024
        conv = "auto" if ask("Keep several conversations in RAM?", ["y", "n"], "y" if room else "n",
                             a.yes) == "y" else 0
    if conv == "auto":
        ok(f"conversation cache: 40% of the RAM free once the model is loaded, up to {POLY_CONV_CACHE_MIB} MiB")
    else:
        ok("conversation cache: " + (f"up to {conv} MiB of RAM" if conv else "off (the last conversation is kept)"))

    # one file, or the numbered files of a split GGUF: the engine is given the first and finds the others by name
    ftag = files_tag(fam, tag, model, _sg)
    models_dir = Path(a.gguf_dir) if a.gguf_dir else Path(a.models_dir) / ftag
    shards = (_sg if _sg
              else gguf_dir_shards(models_dir, fam, model) if a.gguf_dir else
              [models_dir / model_file(fam, model, i) for i in range(1, model_shards(fam, model) + 1)])
    absent = [s.name for s in shards if not s.exists()]
    if a.gguf_dir and absent:
        fail(f"{models_dir} has no {absent[0]}",
             "put the file there under that name, or leave out --gguf-dir and setup downloads it")
    if not a.gguf_dir and not all(s.exists() and done(s) for s in shards):
        for r in elsewhere:                            # already downloaded in a Puddle folder on another drive
            cand = [r / "models" / ftag / s.name for s in shards]
            if all(c.exists() and done(c) for c in cand):
                models_dir, shards = cand[0].parent, cand
                ok(f"model file found in {models_dir}")
                break
    for s in shards:                                   # #173: a whole file copied in by hand
        if s.exists() and not done(s) and whole_shard(s):
            mark(s, "whole (checked against its own tensor directory)")
    gguf = shards[0]
    have_model = all(s.exists() and (done(s) or bool(a.gguf_dir)) for s in shards)
    # the size's draft model: beside the model's file, or, when --gguf-dir has none, where setup keeps its own
    guesser = models_dir / Path(d["draft_model"]).name if d.get("draft_model") else None
    if guesser is not None and a.gguf_dir and not guesser.exists():
        guesser = Path(a.models_dir) / ftag / guesser.name
    have_guesser = guesser is None or guesser.exists() and (done(guesser) or bool(a.gguf_dir) or whole_shard(guesser))
    # the image encoder, in the same places
    mmproj = models_dir / fam["mmproj"] if vision != "none" else None
    if mmproj is not None and a.gguf_dir and not mmproj.exists():
        mmproj = Path(a.models_dir) / ftag / mmproj.name
    # a size from another repository that has no image encoder of its own (Unsloth's quants of Bolt) uses the family's,
    # where another of its sizes keeps it (NVIDIA's, beside Bolt's 24GB file): not downloaded twice
    if (mmproj is not None and not a.gguf_dir and not _sg and d.get("hf") and not d.get("mmproj_hf")
            and not (mmproj.exists() and done(mmproj))):
        for other in fam["models"]:
            cand = Path(a.models_dir) / files_tag(fam, fam["tag"] + other, other) / fam["mmproj"]
            if other != model and cand.exists() and done(cand):
                mmproj = cand
                break
    # user-supplied mmproj: accept any file with "mmproj" in the name
    if (mmproj is not None and not mmproj.is_file() and _sg):
        cands = list(models_dir.glob("*mmproj*"))
        if len(cands) == 1:
            mmproj = cands[0]
        elif len(cands) > 1:
            names = ", ".join(f.name for f in sorted(cands))
            warn(f"found several files with 'mmproj' in {models_dir}: {names}\n"
                 f"renaming one to exactly {fam['mmproj']}, or deleting the others")
    have_mmproj = mmproj is None or mmproj.exists() and (done(mmproj) or bool(a.gguf_dir) or whole_shard(mmproj))
    on_disk = sum(f.stat().st_size for s in shards for f in (s, s.with_name(s.name + ".part")) if f.is_file()) / 1e9
    need = (0 if have_model and have_guesser else max(d["download_gb"] - on_disk, 0)) + \
        (0 if have_mmproj else fam["vision_gb"]) + 2
    if free_gb(models_dir) < need:
        fail(f"not enough free disk space in {models_dir}: need ~{need:.0f} GB" +
             (f" ({on_disk:.0f} GB of the model is already there)" if on_disk >= 1 and not have_model else ""),
             "use --models-dir on a bigger drive")

    step(3, "Python packages")
    pip_install(requirement_lines() if REQUIREMENTS.exists() else PY_PACKAGES,
                "numpy, jinja2, regex, pyyaml, tqdm, requests, cmake, ninja, pillow, psutil")

    step(4, "the strata-poly engine")
    llama = get_llama_cpp()
    ok(f"llama.cpp {LLAMA_CPP_COMMIT[:7]} (gguf-py, ggml)")
    # always compiled here from Puddle's source (no ready-made engine: upstream's have no Laguna); --max-compatibility:
    # the portable build, kept as the installed engine records it
    prev = ROOT / ENGINE_POLY_DIR / "BUILD.json"
    prev_meta = json.loads(prev.read_text(encoding="utf-8")) if prev.exists() else {}
    portable = bool(WIN and (getattr(a, "max_compatibility", False) or prev_meta.get("portable")))
    eng = build_engine_poly_hip(gpu, llama, portable) if hip else \
        build_engine_poly(gpu, a.yes, llama, vision, cuda_tk, portable)
    lib_dirs = engine_lib_dirs(eng)
    ok(f"engine: {eng / POLY_EXE}")

    step(5, f"downloading {title} {model}")
    if not have_model:
        say(f"  The model file goes in {models_dir}")
        say(f"  A file you already have: put it there with its original name ({', '.join(s.name for s in shards)}), "
            "or use --gguf-dir <its folder>.")
        if hf_endpoint() != HF_DEFAULT:
            say(f"  Downloading from {hf_endpoint()} (HF_ENDPOINT)")
        for s in shards:
            if s.exists() and done(s):
                ok(f"{s.name} already downloaded")
                continue
            rel = s.relative_to(models_dir).as_posix() if s.is_relative_to(models_dir) else s.name
            download(d.get("hf", fam["hf"]) + rel, s)
    check_shards(shards)
    # a size from another repository can ship a draft model or an image encoder of the family's file names
    sha = {**fam["sha256"], **d.get("sha256", {})}

    def _skip_or_verify(path, sha_key):
        """Verify a downloaded file's SHA-256; skip for user-supplied files (header-matched, no pinned hash)."""
        if _sg:                       # header-matched: trust it, the user pointed at this file
            return
        verify_sha256(path, *sha[sha_key], own=bool(a.gguf_dir) and path.parent == models_dir)

    for s in shards:
        if _sg:
            continue                    # user-supplied file: no sha256 to match
        elif a.gguf_dir:
            verify_poly_own(s, *sha[s.name], family)
        else:
            verify_poly_download(s, *sha[s.name], family)
    if guesser is not None:
        if not have_guesser:
            say("  Its draft model lets it guess a few words ahead and check them at once (faster answers).")
            download(d.get("hf", fam["hf"]) + d["draft_model"], guesser)
        _skip_or_verify(guesser, guesser.name)
    if mmproj is not None:
        if not have_mmproj:
            download(d.get("mmproj_hf", fam["mmproj_hf"]) + fam["mmproj"], mmproj, "vision encoder")
        _skip_or_verify(mmproj, mmproj.name)
        ok(f"vision encoder: {mmproj}")
    ok("model file present")

    step(6, "preparing the model for Puddle")
    pack = find_in([data, *elsewhere], f"packs/{tag.lower()}") or data / "packs" / tag.lower()
    if not (pack / "tokenizer" / "vocab.json").exists():
        run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"), "--gguf", str(gguf), "--out", str(pack)],
            env=dict(os.environ, STRATA_GGUF_PY=str(llama / "gguf-py")))   # writes <pack>/tokenizer/
    # a file without a chat template gets the model's own from the model's repository, pinned like the files
    tpl = pack / "tokenizer" / "chat_template.jinja"
    if fam.get("template") and not tpl.exists():
        name, size, sha = fam["template"]
        download(fam["template_hf"] + name, tpl, "chat template")
        verify_sha256(tpl, size, sha)
    # The engine counts which experts are used and writes the counts back when it stops, so its profile is a file of
    # this install's own: it starts as the counts the repository ships, or with none where it ships none.
    profile = pack / "expert-profile.txt"
    shipped = d.get("profile", fam.get("profile"))
    if shipped and not profile.exists():
        pack.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "data" / shipped, profile)
    # GLM's draft block is the model's, one for every size.  A size whose file leaves it out gets it here: the block
    # alone is downloaded, not the file it comes from.
    draft = None
    if d.get("draft") != "own" and fam.get("draft"):
        draft = find_in([data, *elsewhere], fam["draft"]) or data / fam["draft"]
        if not draft.exists():
            say("  The draft block lets the model guess a few words ahead and check them at once (faster answers).")
            say(f"  {model}'s file has none, so the block alone is downloaded (~{fam['draft_gb']} GB).")
            run([sys.executable, str(ROOT / "tools" / "glm_draft_fetch.py"), "fetch", "--out", str(draft)])
        ok(f"draft block: {draft}")
    if guesser is not None:
        draft = guesser
        ok(f"draft model: {draft}")
    ok(f"model prepared: {pack}")

    step(7, "writing the start script")
    args = ["--model", str(gguf), "--max-context", str(ctx), "--expert-profile", str(profile)]
    if draft:
        args += ["--draft-model", str(draft)]
    if conv:
        args += ["--conversation-cache-mib", str(conv)]
    if fam.get("kv"):                                  # the family's KV format (Laguna: q8_0, as it runs)
        args += ["--kv", fam["kv"]]
    args += list(fam.get("args", []))                 # the family's own (Nemotron 3.5 Bolt: --no-draft)
    if a.vram_reserve_mib is not None:
        args += ["--vram-reserve-mib", str(a.vram_reserve_mib)]
        ok(f"VRAM kept free for other programs: {a.vram_reserve_mib} MiB (--vram-reserve-mib; the expert cache takes "
           "that much less)")
    if fam.get("tail_skip"):
        args += ["--tail-skip", str(fam["tail_skip"])]
    if not multi and poly_batch(fam, d, vram, ctx):
        args += ["--prompt-batch", "2048"]
        ok("prompts are read 2,048 tokens at a time (--prompt-batch in the config)")
    # The engine's own count is 70% of the logical CPUs.  A hybrid CPU gets one of setup's: an efficiency core runs
    # an expert at half the speed, and a layer waits for its slowest part (hybrid_pool_workers()).
    cores = cpu_cores()
    threads = hybrid_pool_workers(cores)
    if threads is not None:
        args += ["--threads", str(threads)]
        ok(f"hybrid CPU ({cores[0]} performance + {cores[1]} efficiency cores): {threads} CPU threads for the experts "
           "outside VRAM (--threads in the config)")
    base = custom_gguf_tag(_sg, model) if _sg else None
    cfg = {"exe": str(eng / POLY_EXE), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
           "model_name": base.lower() if base else f"{fam['name']}-{model.lower()}",
           "display_name": base if base else f"{title} {model}",
           "log": str(ROOT / f"strata-{tag.lower()}.log"), "lib_dirs": lib_dirs, "port": port}
    if fam.get("tool_call_recovery"):
        cfg["tool_call_recovery"] = True
        ok("its tool calls are JSON inside the wrapper: they are read with the recovery (tool_call_recovery in the "
           "config)")
    if multi:                                          # the server passes --layer-split and orders the cards
        cfg["gpu"], cfg["gpus_asked"] = multi, True
        cfg["layer_split"] = a.layer_split or "auto"
        ok(f"layer split across GPUs {multi} ({cfg['layer_split']})")
    elif gpu["count"] > 1 or a.gpu is not None:
        cfg["gpu"] = gpu["index"]                      # the engine is told this card (issue #51)
        cfg["gpus_asked"] = True
    if hip:                                            # AMD: HIP_VISIBLE_DEVICES, hip_env_guard, sysfs GPU readings
        cfg["backend"] = "hip"
        if WIN and not multi and gpu.get("driver") == "hip":   # Windows: setup's list is HIP's own (#325: an iGPU
            cfg["hip_ordinal"] = gpu["index"]          # may be HIP's device 0); the ordinal wins for one card
    if cuda_tk == 12:
        cfg["cuda"] = 12                               # a copy set up from this install takes it too
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if a.browser is not None:
        cfg["open_browser"] = a.browser
    earlier = adopted if adopted is not None and adopted.name == cfg_path.name else None
    if vision != "none":
        # the most rows a picture becomes: one of this engine's prompt batches (1,024) holds a picture of Gemma 4's
        vt = vision_tokens(a.vision_tokens, vision, cfg_path if cfg_path.is_file() else earlier,
                           fam.get("vision_max_tokens"))
        if vt > fam.get("vision_most", vt):
            warn(f"--vision-tokens {vt}: {title} reads a picture in one prompt batch, {fam['vision_most']} tokens "
                 "unless the engine is started with a larger --prompt-batch; a picture with more is refused")
        cfg["vision"] = {"exe": str(eng / VEXE), "mmproj": str(mmproj), "model": str(gguf), "gpu": vision == "gpu",
                         "max_tokens": vt}
        if vision == "cpu":
            cfg["vision"]["threads"] = max(1, (os.cpu_count() or 8) // 2)
    # requests at once: written only when given, else an earlier "parallel" is carried over (write_setup_config)
    if a.parallel is not None:
        cfg["parallel"] = a.parallel if a.parallel >= 2 else 1
    for line in poly_parallel_note(a.parallel, d, vram):
        (ok if a.parallel is not None else say)(line if a.parallel is not None else "  " + line)
    poly_kv_whole(cfg)
    write_setup_config(cfg_path, cfg, earlier, sampling=fam.get("sampling"))
    script = write_run_script(tag, cfg_path, port, cfg.get("open_browser") is not False)
    ok(f"start script: {script.name}")

    say()
    say("All set.")
    say(f"  API (OpenAI):     http://127.0.0.1:{port}/v1   (any API key; model name: anything)")
    say(f"  API (Anthropic):  http://127.0.0.1:{port}/v1/messages")
    if vision != "none":
        say("  Images:           send them in the chat page, in chat.py (/image <path>) or over the API")
    if a.host and a.host not in ("127.0.0.1", "localhost"):
        say(f"  Other devices:    the server window prints this PC's address (http://<IP>:{port}/)"
            + ("" if a.api_key else " - no API key set: anyone on your network can use it"))
    say(f"  Next time:        just run {'START-HERE.bat' if WIN else './setup.sh'} (or {script.name}) - it starts right away")
    if a.no_start:
        return 0
    return start(cfg_path, port)


# ------------------------------------------------------------------------------------------------ main
def mac_setup(argv) -> int:
    """A Mac (Apple Silicon): strata-poly for Metal, built from source on the Mac by metal/setup_mac.py (its own
    steps: the Mac check, the model, the engine build, the Metal defaults). Nothing of the CUDA / HIP paths is used or
    changed."""
    return subprocess.call([sys.executable, str(ROOT / "metal" / "setup_mac.py"), *argv])


def resolve_custom_gguf(path_str: str) -> tuple[str, str, list[Path]]:
    """--custom-gguf PATH: inspect the GGUF header, match it to a family/model,
    and return (family, model, [shard paths])."""
    _tools = str(ROOT / "tools")
    if _tools not in sys.path:
        sys.path.insert(0, _tools)
    from strata_inspect import read_headers, fingerprint, verdict, load_fingerprints

    p = Path(path_str)
    files: list[Path]
    if p.is_file() and p.suffix.lower() == ".gguf":
        files = [p]
    elif p.is_dir():
        # group by shard stem so two models in one folder don't corrupt the fingerprint
        def _shard_idx(f):
            m = SHARD_NAME.search(f.name)
            return int(m.group(1)) if m else 0

        stems: dict[str, list[Path]] = {}
        for f in sorted(p.iterdir(), key=lambda x: (_shard_idx(x), x.name)):
            if f.suffix.lower() != ".gguf" or "mmproj" in f.name.lower():
                continue
            base = SHARD_NAME.search(f.name)
            key = SHARD_NAME.sub("", f.name) if base else f.stem
            stems.setdefault(key, []).append(f)
        if not stems:
            fail(f"{path_str} has no .gguf files", "point at a .gguf file or a folder containing them")
        if len(stems) > 1:
            keys = ", ".join(sorted(stems))
            fail(f"{path_str} contains shards of several models ({keys})",
                 "put one model's shards in their own folder, or point at the single .gguf file directly")
        files = stems[list(stems)[0]]   # exactly one model: use all its shards
    else:
        fail(f"{path_str} is neither a .gguf file nor a folder of GGUF shards",
             "give the path to one .gguf file, or a folder holding all its shards")

    from quantscope import Source
    sources = [Source(name=f.name, size=f.stat().st_size, path=str(f)) for f in files]
    arch, tensors = read_headers(sources)
    fp = fingerprint(arch, tensors)
    kind, sentence = verdict(fp, load_fingerprints())

    if kind in ("known", "layout"):
        # parse family/model from the matching key(s) in the table
        table = load_fingerprints()
        known = {k: v for k, v in table.items() if not k.startswith("_")}
        if kind == "known":
            matches = [k for k, v in known.items()
                       if v["expert_bytes"] == fp["expert_bytes"] and v["bytes"] == fp["bytes"]
                       and v["tensors"] == fp["tensors"]]
        else:  # layout
            matches = [k for k, v in known.items() if v["expert_bytes"] == fp["expert_bytes"]]
        if len(matches) > 1:
            say(f"Puddle: {sentence}")
            fail(f"this GGUF's header matches several entries ({', '.join(sorted(matches))})",
                 "use --inspect to read the full explanation, then pick explicitly with --family X --model Y")
        fam_model = matches[0].split("/")
        return fam_model[0], fam_model[1], files

    if kind == "family":
        from strata_inspect import family_title
        title = family_title(fp)
        fam, models_dict = next(((k, v["models"]) for k, v in FAMILIES.items() if v.get("title") == title),
                                (None, None))
        if not fam:
            say(f"Puddle: {sentence}")
            fail("cannot install this GGUF with --custom-gguf",
                 "use --inspect to read the full explanation, or give it to --gguf-dir")
        # derive the size from the filename's quant token (e.g. "Q4_K_M" in "...-Q4_K_M.gguf")
        m_q = GGUF_QUANT.search(files[0].name)
        quant = m_q.group(1) if m_q else Path(SHARD_NAME.sub("", files[0].name)).stem.rsplit("-", 1)[-1]
        model = next((m for m in models_dict if quant.lower().replace("_", "") in m.replace("_", "").lower()
                      or m.lower().endswith(quant.lower())), None)
        if not model:
            model = sorted(models_dict)[0]
            warn(f"Puddle: {sentence}\n  your file's quant '{quant}' is not one setup offers "
                 f"{', '.join(sorted(models_dict))}: it will run as-is; the RAM and VRAM estimates are for "
                 f"{FAMILIES[fam]['title']}'s --model {model}")
        return fam, model, files

    say(f"Puddle: {sentence}")
    fail("cannot install this GGUF with --custom-gguf",
         "use --inspect to read the full explanation, or give it to --gguf-dir")


def main() -> int:
    if sys.platform == "darwin":                       # a Mac: metal/setup_mac.py (strata-poly on Metal)
        return mac_setup(sys.argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=list(FAMILIES),
                    help="gemma4 = Gemma 4 26B-A4B (docs/BACKENDS.md)")
    ap.add_argument("--model", choices=list(dict.fromkeys(m for f in FAMILIES.values() for m in f["models"])))
    ap.add_argument("--context", type=int)
    ap.add_argument("--vision", choices=["yes", "no", "none", "gpu", "cpu"],
                    help="let the model read images (yes = the encoder on the GPU)")
    ap.add_argument("--vision-tokens", type=int, metavar="N",
                    help="the most image tokens a picture becomes (default 1024 with the encoder on the GPU, 300 on "
                         "the CPU): more reads small text and charts better, and takes longer to encode; remembered "
                         "for this model")
    ap.add_argument("--port", type=int, help="the server's port (default: the one the install was set up with, 8080 for a new one)")
    ap.add_argument("--gpu", help="one GPU, numbered as nvidia-smi numbers them (AMD: as setup lists them) (default: asked when several can be "
                                  "used; with --setup it is saved, when starting it is for that start only)")
    ap.add_argument("--gpus", help="several GPUs sharing one model, as nvidia-smi numbers them (AMD: as setup lists "
                                   "them): \"0,2\", or \"all\" (every card that can); the first is the main one. "
                                   "Saved, also when starting (see docs/MULTI_GPU.md)")
    ap.add_argument("--layer-split", help="with --gpus: where each later GPU's layers start (\"18\", \"16,32\"), one "
                                          "rising number per GPU after the first - not layers per card; default "
                                          "auto, placed from each GPU's free VRAM")
    ap.add_argument("--no-remote-expert-opt", action="store_true",
                    help="with two or more GPUs: leave out --remote-expert-opt, which setup adds there (#578)")
    ap.add_argument("--host", help="where the server listens: 127.0.0.1 = this PC only (default), 0.0.0.0 = also other "
                                   "devices on your network (issue #26; set --api-key too)")
    ap.add_argument("--api-key", help="require this key from clients (recommended with --host 0.0.0.0); several keys: key1,key2")
    ap.add_argument("--no-browser", dest="browser", action="store_false", default=None,
                    help="do not open the chat page in the browser when the model is ready (for a harness or an app "
                         "that uses the API; remembered for this model, also in run-<model>.bat/.sh)")
    ap.add_argument("--browser", dest="browser", action="store_true",
                    help="open the chat page again when the model is ready (the default; undoes --no-browser)")
    ap.add_argument("--data-dir", help="where the model files go (~70-120 GB): default Puddle-data next to this folder, "
                                       "remembered for every Puddle folder on this PC")
    ap.add_argument("--models-dir", help="where the GGUF files go (default: <data folder>/models)")
    ap.add_argument("--gguf-dir", help="use GGUF files you already have (a folder with every shard: "
                                       "<name>-00001-of-0000N.gguf ... -0000N-of-0000N.gguf)")
    ap.add_argument("--yes", action="store_true", help="accept the recommended answers")
    ap.add_argument("--setup", action="store_true", help="install another model or change settings")
    ap.add_argument("--no-start", action="store_true", help="install only, do not start the model")
    ap.add_argument("--update", action="store_true",
                    help="update the installed engine (compiled again when its source changed), Python packages and "
                         "model settings as a start would, without starting the model (UPDATE.bat / update.sh run it "
                         "after a git pull)")
    ap.add_argument("--tests", nargs="?", const="", default=None, metavar="REGEX",
                    help="with --update (./update.sh --tests): also compile strata-poly with its own tests in "
                         f"{POLY_TESTS_DIR}/ and run them (ctest); REGEX: only the tests whose names match. The "
                         "installed engine is not touched")
    ap.add_argument("--max-compatibility", action="store_true",
                    help="on Windows: compile the engine for other PCs too (STRATA_PORTABLE: AVX2 baseline, static "
                         "MSVC runtime); the default build uses this PC's own CPU instructions")
    ap.add_argument("--source", choices=SOURCES, default=None,
                    help="where the model files come from: auto (default: Hugging Face), huggingface or modelscope "
                         "(mainland China: the same files, checked against ModelScope's published SHA-256; "
                         "STRATA_SOURCE)")
    ap.add_argument("--inspect", nargs="+", metavar=("SOURCE", "VARIANT"),
                    help="what a GGUF is and whether Puddle runs it, from its headers only (no download): a file, a "
                         "folder, a URL, ms:owner/repo (ModelScope) or hf:owner/repo, and optionally a variant name")
    ap.add_argument("--custom-gguf", metavar="PATH",
                    help="a .gguf file or folder of shards you already have: setup inspects it, matches it to a family "
                         "and size, and installs without downloading the model")
    ap.add_argument("--cuda", choices=["12", "13", "auto"], default=os.environ.get("STRATA_CUDA") or None,
                    help="NVIDIA: the CUDA toolkit the engine is compiled with. auto (default): CUDA 13; CUDA 12 "
                         "(experimental) when a chosen card is older than CUDA 13 supports (Pascal, Volta). 12 also "
                         "runs with an older driver (Windows 528+, Linux 525+). docs/OLDER_GPUS.md")
    ap.add_argument("--check", action="store_true", help="only check this PC and exit")
    ap.add_argument("--vram-reserve-mib", type=int, metavar="N",
                    help="VRAM in MiB the engine leaves free for other programs (a game, another model; the engine's "
                         "default: 700); the expert cache takes that much less")
    ap.add_argument("--parallel", type=int, metavar="N",
                    help="up to N requests decode together (batch slots, opt-in; default: one at a time, the others "
                         "wait). Each slot takes VRAM from the expert cache; setup says what it recommends")
    ap.add_argument("--replicas", type=int, metavar="N",
                    help="several GPUs: run N independent engines, each on its own equal group of the chosen cards (2 "
                         "replicas on 4 cards = two 2-card engines), requests spread over them (opt-in; more total "
                         "throughput for a model that fits fewer cards; each replica holds its own copy of the model)")
    ap.add_argument("--backend", choices=["cuda", "hip"],
                    help="cuda = NVIDIA (default), hip = AMD RX 7900 / 7800 / 7700 XT, RX 9060 XT / 9070 / AI PRO R9700 on "
                         "Linux or Windows (chosen by itself when the PC has no NVIDIA card Puddle can use)")
    ap.add_argument("--skip-build", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    a._custom_gguf_shards = None                      # set by --custom-gguf; None means "not using your own file"
    if a.source:
        os.environ["STRATA_SOURCE"] = a.source
    if a.inspect:                                      # headers only: nothing is installed
        sys.exit(subprocess.run([sys.executable, str(ROOT / "tools" / "strata_inspect.py"), *a.inspect[:2]]).returncode)
    if a.custom_gguf:                                   # your own GGUF: inspect it, match it, skip download
        _fam, _model, a._custom_gguf_shards = resolve_custom_gguf(a.custom_gguf)
        p = Path(a.custom_gguf)
        if (a.family and a.family != _fam) or (a.model and a.model != _model):
            fail(f"--custom-gguf resolves to --family {_fam} --model {_model}, "
                 f"but you chose --family {a.family} --model {a.model}",
                 f"--custom-gguf files pick their own family and model: drop the conflicting flags, or use a GGUF of your chosen size")
        if a.gguf_dir:                                  # explicit --gguf-dir conflicts with custom-gguf (no download)
            warn(f"--gguf-dir {a.gguf_dir} is ignored: --custom-gguf files are not downloaded")
        a.gguf_dir = str(p) if p.is_dir() else str(p.parent)
        a.family, a.model = _fam, _model
        # validate the resolved pair: the family must be one Puddle offers, with that size
        ok_models = FAMILIES[_fam]["models"] if _fam in FAMILIES else {}
        if a.model not in ok_models:
            avail = sorted(ok_models)
            fail(f"--custom-gguf matched --family {_fam} --model {_model}, but {a.family}'s engine path cannot install it",
                 f"available sizes for {a.family}: {', '.join(avail)}")
    if a.vision_tokens is not None and a.vision_tokens < 1:
        ap.error("--vision-tokens takes a number of image tokens, 1 or more, e.g. --vision-tokens 768")
    if a.vram_reserve_mib is not None and a.vram_reserve_mib < 0:
        ap.error("--vram-reserve-mib takes a number of MiB, 0 or more, e.g. --vram-reserve-mib 2048")
    if a.gpu is not None:                             # --gpu 0,2 means --gpus 0,2 (a user tried it: issue report)
        if "," in a.gpu:
            a.gpus, a.gpu = a.gpus or a.gpu, None
        elif a.gpu.strip().isdigit():
            a.gpu = int(a.gpu)
        else:
            ap.error(f"--gpu takes a GPU number as nvidia-smi numbers them, e.g. --gpu 1 (or --gpus 0,2), not {a.gpu!r}")
    say("Puddle - Laguna, Nemotron 3.5 Bolt and Gemma 4 on a normal PC (a GPU + system RAM + CPU)")
    data, elsewhere = data_folder(a.data_dir)          # the model files: in the data folder, found from any copy
    roots = [data, *elsewhere]
    if a.models_dir is None:
        a.models_dir = str(data / "models")

    # ---- 0. already installed: just start it
    have = installed_configs()
    if a.tests is not None and not a.update:
        fail("--tests goes with --update", "run ./update.sh --tests (UPDATE.bat --tests)")
    if a.update:                                       # #475: UPDATE.bat / update.sh - never starts the model
        return update_install(have, a)
    explicit = a.setup or a.model or a.family or a.check or a.no_start
    adopted = None                                     # #629: the earlier install this copy is set up like
    if not have and not explicit:                      # a new copy of Puddle (an update unzipped elsewhere): set it
        prev = previous_config(elsewhere, load_settings())   # up like the last one, from the files already here
        if prev is not None:
            ch = choices_from_config(prev)
            if ch["model"]:
                say(f"  Found your earlier install in {prev.parent} ({prev.stem[len('strata-'):]}): setting up this "
                    "copy the same way - the model files are reused, nothing big is downloaded.")
                a.family, a.model, a.context = ch["family"], ch["model"], a.context or ch["context"]
                adopted = prev
                a.vision = a.vision or ch["vision"]
                a.host, a.api_key = a.host or ch["host"], a.api_key or ch["api_key"]
                a.port = a.port or ch["port"]
                if a.vram_reserve_mib is None:          # #493: an explicit reserve set up before
                    a.vram_reserve_mib = ch.get("vram_reserve_mib")
                if a.cuda is None and ch.get("cuda") == 12:   # the experimental CUDA 12 engine, as before
                    a.cuda = "12"
                if isinstance(ch.get("gpu"), list):     # a layer split: set up across the same cards again
                    a.gpus = a.gpus or ",".join(str(g) for g in ch["gpu"])
                    a.layer_split = a.layer_split or ch.get("layer_split")
                else:
                    a.gpu = a.gpu if a.gpu is not None else ch.get("gpu")
                a.yes = True
    global GPU_PICK, OLD_GPUS
    # starting an installed model: --gpus 0,2 (or all) saves those cards for it and starts on them (it used to start
    # on the first one alone unless given with --setup), --gpu N runs this start on one card; neither: the saved
    # choice, and asked once when the PC has cards that could share the model
    run_gpu = start_gpus(a.gpus) or a.gpu
    port = a.port or 8080                              # a new install's port (issue #32: --port for an existing one)
    if have and not (a.setup or a.model or a.family or a.check or a.no_start):
        if len(have) == 1:
            return start(have[0], a.port, run_gpu, yes=a.yes, layer_split=a.layer_split,
                     keep={"host": a.host, "api_key": a.api_key,
                           "vram_reserve_mib": a.vram_reserve_mib, "open_browser": a.browser})
        say()
        for i, c in enumerate(have, 1):
            say(f"  {i}) {json.loads(c.read_text(encoding='utf-8-sig')).get('model_name', c.stem)}")
        say(f"  {len(have) + 1}) install another model / change settings")
        pick = int(ask("Which one?", [str(i) for i in range(1, len(have) + 2)], "1", a.yes))
        if pick <= len(have):
            return start(have[pick - 1], a.port, run_gpu, yes=a.yes, layer_split=a.layer_split,
                     keep={"host": a.host, "api_key": a.api_key,
                           "vram_reserve_mib": a.vram_reserve_mib, "open_browser": a.browser})

    # ---- 1. the PC
    step(1, "checking your PC")
    found = gpus()
    amd = amd_gpus()
    amd_ok = [g for g in amd if amd_problem(g) is None]
    # older NVIDIA GPUs (Pascal / Volta): the experimental CUDA 12 engine, when chosen (docs/OLDER_GPUS.md)
    OLD_GPUS = OLD_GPUS or old_gpus_opt_in(found, named_gpus(a.gpu, a.gpus), a.cuda,
                                           other=bool(amd_ok) or a.backend == "hip")
    if OLD_GPUS and a.backend != "hip":
        warn(f"older NVIDIA GPUs (Pascal / Volta) can be used ({OLD_GPUS}): experimental, through a second engine "
             "built with CUDA 12 (docs/OLDER_GPUS.md)")
    nv_ok = any(gpu_problem(g) is None for g in found)
    hip = a.backend == "hip" or (a.backend is None and not nv_ok and bool(amd_ok))
    if a.backend is None and nv_ok and amd_ok:
        # both kinds of card: asked (a first run on such a PC used to take NVIDIA without mentioning the Radeon)
        say()
        say("  This PC has NVIDIA and AMD cards Puddle can use:")
        say("  1) NVIDIA: " + ", ".join(f"{g['name']} ({g['vram_gb']:.0f} GB)" for g in found if gpu_problem(g) is None)
            + "   (recommended)")
        say("  2) AMD: " + ", ".join(f"{g['name']} ({g['vram_gb']:.0f} GB{', unified memory' if g.get('uma') else ''})"
                               for g in amd_ok)
            + f"   ({'compiled here, no images' if WIN else 'compiled here, images on the CPU'}"
              " - docs/AMD_HIP.md)")
        hip = ask("Which cards?", ["1", "2"], "1", a.yes or a.check) == "2"
        if a.check and not hip:
            say(f"  (the AMD card: {'START-HERE.bat' if WIN else './setup.sh'} --backend hip)")
    cuda_tk = 13                                       # NVIDIA: the toolkit of this model's engine (cuda_choice)
    if hip:                                            # AMD: compiled here; Windows: ready-made
        if WIN and a.gpus:
            fail("several AMD cards sharing one model (--gpus) is Linux-only for now", "use one card: --gpu N")
        say("  Your AMD GPUs:" if amd else "  No AMD GPU found (" + ("Windows lists no AMD display adapter)." if WIN
                                                                   else "the amdgpu driver's KFD topology is empty)."))
        for g in amd:
            say(f"    GPU {g['index']}: {g['name']}, {amd_mem_text(g)} - " + (amd_problem(g) or "can be used"))
        usable = [g for g in amd if amd_problem(g) is None]
        if not WIN and (amd or amd_pci_devices()):     # a warning only: recommend, never force
            acc = amd_device_access_problem()
            if acc:
                warn(acc)
        if not amd and not WIN:                        # the KFD topology is empty: name a Strix Halo the kernel sees
            for d in amd_pci_devices():
                if d["pci_id"] in STRIX_HALO_PCI_IDS:
                    say(f"    The kernel lists an AMD Strix Halo (PCI 1002:{d['pci_id']:04x}) but /dev/kfd's topology is "
                        "empty: ROCm cannot use it yet (it needs the kernel's amdgpu driver and /dev/kfd access: "
                        "your user in the render and video groups)")
        if not usable:
            fail("no AMD GPU Puddle can use", f"the AMD backend runs on {AMD_CARDS}")
        if a.gpus:                                     # a layer split across these cards, the first one the main
            chosen = amd_parse_gpus(a.gpus, amd)
            gpu = chosen[0]
        elif a.gpu is not None:
            gpu = next((g for g in usable if g["index"] == a.gpu), None)
            if gpu is None:
                fail(f"AMD GPU {a.gpu} cannot be used", "use one of: " + ", ".join(f"--gpu {g['index']}" for g in usable))
            chosen = [gpu]
        else:
            gpu = min(usable, key=amd_rank)            # a discrete card before an APU, then the most memory
            chosen = [gpu]
            if len(usable) > 1 and any(g.get("uma") for g in usable) != all(g.get("uma") for g in usable):
                others = [g for g in usable if g is not gpu]
                say(f"  Using GPU {gpu['index']} ({gpu['name']}); setup recommends the card with its own memory over "
                    "the APU's shared memory. To use another: "
                    + ", ".join(f"--gpu {g['index']} ({g['name']})" for g in others))
        # the engine is compiled for every chosen card's architecture
        gpu = {**gpu, "count": len(amd), "archs": sorted({g["arch"] for g in chosen})}
        chosen = [gpu] + chosen[1:]
        sel = [g["index"] for g in chosen]
        multi = sel if len(sel) > 1 else []
        a.gpu = gpu["index"] if len(amd) > 1 else a.gpu
        if multi:
            ok("GPUs: " + " + ".join(gpu_name(x) for x in chosen) + " together (the model's layers are split across them)")
        ok(f"GPU: {gpu['name']}, {amd_mem_text(gpu) if gpu.get('uma') else format(gpu['vram_gb'], '.1f') + ' GB VRAM'}, "
           f"{gpu['arch']} (AMD: docs/AMD_HIP.md)")
        if WIN and str(gpu.get("arch") or "").startswith("gfx12"):   # only a pointer; no default changes
            say("  If Windows resets the AMD driver (VIDEO_ENGINE_TIMEOUT_DETECTED, flicker, the engine dies mid-answer): "
                "docs/TROUBLESHOOTING.md, \"Windows AMD: the driver resets\"")
        if gpu.get("uma"):
            for line in igpu_notes(gpu, ram_gb()):
                (warn if line.startswith("!") else say)(line.lstrip("!"))
    else:
        if not found:
            fail("no NVIDIA GPU found (nvidia-smi did not answer)",
                 "install the NVIDIA driver from https://www.nvidia.com/drivers and restart the PC"
                 + (f"; AMD ({', '.join(AMD_ARCHS)}): --backend hip" if amd else ""))
        if len(found) > 1 or gpu_problem(found[0]) is not None:
            gpu_table(found)
        sel = choose_gpus(a, found)                    # asked when two or more cards can share the model
        multi = sel if len(sel) > 1 else []
        a.gpu = sel[0]                                 # the main GPU: the checks and the sizing below are its
        GPU_PICK = a.gpu
        gpu = gpu_info(a.gpu)
        chosen = [gpu_info(i) for i in sel]
        gpu["archs"] = sorted({x["arch"] for x in chosen})  # the engine needs code for every one of them
        if multi:
            ok("GPUs: " + " + ".join(gpu_name(x) for x in chosen) + " together (the model's layers are split across them)")
        ok(f"GPU: {gpu['name']}, {gpu['vram_gb']:.1f} GB VRAM, compute capability {cc(gpu)}, driver {gpu['driver']}")
        # one engine per model: its oldest card decides
        cuda_tk, why = cuda_choice(gpu["archs"], a.cuda, True)   # every family runs on strata-poly
        if why:
            (warn if cuda_tk == 13 or str(a.cuda) == "12" else ok)(f"CUDA {cuda_tk}: {why}")
        min_driver = CUDA12_MIN_DRIVER if cuda_tk == 12 else MIN_DRIVER
        if driver_major(gpu) < min_driver:
            fail(f"the NVIDIA driver is too old ({gpu['driver']}; {min_driver} or newer is needed)",
                 "update it with the NVIDIA App or from https://www.nvidia.com/drivers, restart, and run this again" +
                 ("" if cuda_tk == 12 else f" (or --cuda 12: the experimental CUDA 12 engine runs with driver "
                                           f"{CUDA12_MIN_DRIVER} or newer, docs/OLDER_GPUS.md)"))
    if gpu["vram_gb"] < 11:
        warn("less than 12 GB of " + ("GPU memory (this APU's carve-out + shared memory)" if gpu.get("uma") else "VRAM")
             + ": Puddle will run, but most experts stay on the CPU and it will be slow")
    ram = ram_gb()
    cpu, avx2, avx512 = cpu_info()
    # each family has its own estimate of the RAM it wants, by size and card (poly_setup); here only the floor
    need = min(d["ram_gb"] for f in FAMILIES.values() for d in f["models"].values())
    (ok if ram >= need else warn)(f"RAM: {ram:.0f} GB" + ("" if ram >= need else
                                                         f" (less than the {need} GB the smallest model wants)"))
    pf = page_file_gb()
    if pf is not None and pf < 4:
        warn(page_file_advice(pf, page_file_setting()))
    ok(f"CPU: {cpu} ({'AVX-512' if avx512 else 'AVX2' if avx2 else 'no AVX2'})")
    link = None if hip else pcie_link(int(gpu.get("index", 0)))
    if link is not None:
        line, problem = pcie_lines(link)
        ok(line)
        if problem:
            warn(problem)
    floor = cpu_floor(avx2)
    if floor == "unsupported":
        fail("this CPU has neither AVX2 nor SSE4.2; Puddle needs at least SSE4.2 (Intel Nehalem, 2008, or newer)")
    if not avx2:
        # #394 #595 #623: an older CPU: the CPU's share of the experts runs on ggml-cpu's kernels for this CPU.
        # Experimental: measured only on newer CPUs with the older path forced, and by users on a few Xeons.  A
        # warning, not a stop.
        warn("this CPU has no AVX2: Puddle support for it is EXPERIMENTAL and slow - the CPU's share of the "
             "experts runs on ggml-cpu's kernels, a few times slower than on an AVX2 CPU")
    if a.check:
        say()
        any_fits = False
        together = sum(x["vram_gb"] for x in (amd if hip else found) if x["index"] in sel) or gpu["vram_gb"]
        for f, fd in FAMILIES.items():                  # every family has sizes of its own
            for m, d in fd.get("models", {}).items():
                v = together if fd.get("engine") == "llama-server" else gpu["vram_gb"]   # by the chosen cards together
                any_fits = any_fits or poly_fits(d, ram, v, hip)
                say(f"  {fd['title']} {m}: {poly_verdict(d, ram, v, hip, bool(fd.get('windows')))}")
        if not any_fits:
            # #977: every size says "does not fit", so the verdict says so too (and the exit code, for scripts). It only
            # reports: --model NAME --yes still installs one anyway.
            say(f"\nThis PC cannot run Puddle yet: no model size fits {ram:.0f} GB of RAM (the smallest needs about "
                f"{need} GB). --model NAME --yes installs one anyway, slowly.")
            return 1
        say("\nThis PC can run Puddle. Run it again without --check to install.")
        return 0

    # ---- 2. the questions
    step(2, "your choices")
    fams = list(FAMILIES)
    if a.family:
        family = a.family
    else:
        for i, f in enumerate(fams, 1):
            d = FAMILIES[f]
            say(f"  {i}) {d['title']:20s} {d['by']} - {d['about']}" + ("   [experimental]" if d.get("experimental") else ""))
        family = fams[int(ask("Which model?", [str(i) for i in range(1, len(fams) + 1)], "1", a.yes)) - 1]
    fam = FAMILIES[family]
    ok(f"model: {fam['title']}")
    if fam.get("license"):
        say(f"  Its license: {fam['license']}")
    if fam.get("engine") == "llama-server":
        return llama_server_setup(a, family, gpu, hip, cuda_tk, multi, ram, data, elsewhere, port, adopted)
    return poly_setup(a, family, gpu, hip, cuda_tk, multi, ram, data, elsewhere, port, adopted)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\nstopped.")
        sys.exit(1)
