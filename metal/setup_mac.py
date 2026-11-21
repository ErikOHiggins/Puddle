"""setup.py on a Mac: Apple Silicon, strata-poly's Metal engine (src/poly with metal/src/poly's kernels), built from
source on the Mac, for Puddle's families (setup.py's FAMILIES).

    ./setup.sh [options]          (setup.py hands a macOS run to this file; the PC paths are not used)

What it does, each step skipped when it is already done:

  1. checks the Mac: Apple Silicon, the chip, unified memory, macOS version, the Command Line Tools, free disk
  2. the model: its family and size (recommend, never force: an explicit --model is kept, with what it risks)
  3. installs the Python packages into .venv (cmake and ninja come from pip: no Homebrew, no sudo)
  4. builds strata-poly for Metal from source: metal-cpp (Apple's headers) and llama.cpp's ggml at the pinned commit
  5. downloads the model (setup.py's downloader: resumable, the pinned revision), and its image encoder if asked
  6. prepares its tokenizer
  7. writes strata-<model>.json and run-<model>.sh and starts the server (serve/server.py), the same as on a PC

Everything shared - the model table, the downloader, the config and run-script writers, the server - is setup.py's.
(The Qwen3.8-Flash-Next Metal engine this file also installed is gone with the Qwen families, phase 1b-3.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import setup as S  # noqa: E402
from tools import metal_poly_setup as P  # noqa: E402

# Apple's metal-cpp headers for the macOS 26 SDK (https://developer.apple.com/metal/cpp/), pinned by SHA-256
METAL_CPP_URL = "https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip"
METAL_CPP_SHA256 = "4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4"
BUILD_DIR = ROOT / "build-metal"
EXE = BUILD_DIR / "strata-poly"
MIN_MACOS = (15, 0)        # MTLResidencySet (the engine keeps every buffer resident through one)

# Every family runs on strata-poly; its files, revisions, sizes and checksums stay in setup.py's single catalog.
# (a family the Metal build has no kernels for says "metal": False: the Nemotrons' Mamba-2 layers)
MAC_POLY_FAMILIES = tuple(f for f, d in S.FAMILIES.items() if d.get("engine") == "poly" and d.get("metal", True))


def smaller_model_catalog() -> dict:
    """The strata-poly families of the shared installer, with their files."""
    catalog = {}
    for family in MAC_POLY_FAMILIES:
        fam = S.FAMILIES[family]
        catalog[family] = {
            "title": fam["title"],
            "engine": "strata-poly",
            "metal_ready": True,
            "model_validated": False,
            "models": {
                model: {"download_gb": spec["download_gb"],
                        "files": [S.model_file(fam, model, i)
                                  for i in range(1, S.model_shards(fam, model) + 1)]}
                for model, spec in S.family_models(family).items()
            },
        }
    return catalog


def model_catalog() -> dict:
    """All installer choices, keyed by family and then quantization (never quant alone): strata-poly's families
    (not the Nemotrons: the Metal build has no Mamba-2 kernels, and llama-server is built for NVIDIA cards only)."""
    return {family: dict(S.FAMILIES[family], models=S.family_models(family), metal_engine="strata-poly")
            for family in MAC_POLY_FAMILIES}


# beside the mapped file: the KV cache, the engine's buffers, the draft model and the macOS desktop
OVERHEAD_GB = 8.0
GIB_GB = 1.073741824    # a GiB in GB: ram_gb is binary, the model sizes are decimal


# ------------------------------------------------------------------------------------------------ the Mac
def _sysctl(name: str, run=subprocess.run) -> str:
    try:
        return run(["sysctl", "-n", name], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def mac_info(run=subprocess.run) -> dict:
    """The chip, its unified memory, macOS, the architecture and the Command Line Tools."""
    chip = _sysctl("machdep.cpu.brand_string", run) or "unknown chip"
    try:
        ram = int(_sysctl("hw.memsize", run) or 0) / 2 ** 30   # GB as Apple sells them (128 GB = 2^37 bytes)
    except ValueError:
        ram = 0.0
    try:
        ver = run(["sw_vers", "-productVersion"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        ver = ""
    try:
        clt = run(["xcode-select", "-p"], capture_output=True, text=True, timeout=10)
        clt_path = clt.stdout.strip() if clt.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        clt_path = ""
    return {"chip": chip, "ram_gb": ram, "macos": ver, "arm64": platform.machine() == "arm64",
            "clt": clt_path, "perf_cores": int(_sysctl("hw.perflevel0.physicalcpu", run) or 0)}


def macos_tuple(ver: str) -> tuple:
    try:
        return tuple(int(x) for x in ver.split(".")[:2])
    except ValueError:
        return (0, 0)


def mac_problems(info: dict) -> list[str]:
    """What stops the setup on this Mac (each with what to do)."""
    p = []
    if not info["arm64"]:
        p.append(f"this Mac has an Intel CPU ({info['chip']}): the Metal engine needs Apple Silicon (M1 or newer)")
    if macos_tuple(info["macos"]) < MIN_MACOS:
        p.append(f"macOS {info['macos'] or '?'}: the Metal engine needs macOS {MIN_MACOS[0]} or newer (System Settings "
                 "> General > Software Update)")
    if not info["clt"]:
        p.append("the Command Line Tools are missing (the compiler the engine is built with): run "
                 "`xcode-select --install`, then this setup again")
    return p


# ------------------------------------------------------------------------------------------------ the engine
def metal_cpp() -> Path:
    """Apple's metal-cpp headers in third_party/metal-cpp (one zip, pinned)."""
    dst = ROOT / "third_party" / "metal-cpp"
    if (dst / "Metal" / "Metal.hpp").exists():
        return dst
    z = ROOT / "third_party" / "metal-cpp_26.zip"
    S.download(METAL_CPP_URL, z, "metal-cpp (Apple's Metal headers)")
    S.verify_sha256(z, z.stat().st_size, METAL_CPP_SHA256)
    tmp = ROOT / "third_party" / "_metal_cpp"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(tmp)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.move(str(tmp / "metal-cpp"), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)
    return dst


def tool(name: str) -> str:
    """cmake / ninja from the venv's pip packages (no Homebrew)."""
    p = Path(sys.executable).parent / name
    return str(p) if p.exists() else (shutil.which(name) or name)


def engine_current() -> bool:
    exe, stamp = BUILD_DIR / "strata-poly", "BUILD-poly.json"
    try:
        meta = json.loads((BUILD_DIR / stamp).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return exe.exists() and meta.get("version") == S.source_version() and meta.get("source") == source_digest()


def source_digest() -> str:
    """What the engine is built from: every source file's path, size and modification time (src/, include/, metal/,
    third_party/ggml).  A git pull, an unzipped update or a local edit changes it; nothing else does."""
    h = hashlib.sha256()
    for d in ("src", "include", "metal", "third_party/ggml"):
        for f in sorted((ROOT / d).rglob("*")):
            if f.is_file() and f.suffix in (".cpp", ".cu", ".h", ".hpp", ".mm", ".metal", ".inc", ".txt", ".cuh", ".py"):
                st = f.stat()
                h.update(f"{f.relative_to(ROOT).as_posix()}|{st.st_size}|{int(st.st_mtime)}\n".encode())
    return h.hexdigest()


def build_engine(llama: Path, jobs: int | None = None) -> Path:
    """strata-poly for Metal from source: cmake + ninja (pip's), Apple's clang from the Command Line Tools."""
    exe, stamp = BUILD_DIR / "strata-poly", "BUILD-poly.json"
    if engine_current():
        S.ok(f"Metal engine: {exe} (built for this source)")
        return exe
    mcpp = metal_cpp()
    S.say("  Building the Metal engine from source (5-15 minutes the first time) ...")
    BUILD_DIR.mkdir(exist_ok=True)
    log = BUILD_DIR / "build.log"
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    cfg = [tool("cmake"), "-S", str(ROOT / "metal"), "-B", str(BUILD_DIR), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
           f"-DCMAKE_MAKE_PROGRAM={tool('ninja')}", f"-DMETAL_CPP_DIR={mcpp}", f"-DSTRATA_GGML_DIR={llama}"]
    cfg += ["-DSTRATA_METAL_POLY=ON"]
    build = [tool("ninja"), "-C", str(BUILD_DIR), exe.name] + (["-j", str(jobs)] if jobs else [])
    with open(log, "w", encoding="utf-8") as f:
        for cmd in (cfg, build):
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            if rc != 0:
                S.fail(f"the Metal engine did not build (see {log})",
                       "the last lines of the log name the problem; the Command Line Tools must be up to date "
                       "(softwareupdate --list)")
    (BUILD_DIR / stamp).write_text(json.dumps({"version": S.source_version(), "source": source_digest(),
                                                      "backend": "metal"}, indent=1), encoding="utf-8")
    S.ok(f"Metal engine built: {exe}")
    return exe


# ------------------------------------------------------------------------------------------------ the config
VISION_EXE = ROOT / "build-vision" / "bin" / "strata-vision"


def build_vision(llama: Path) -> Path:
    """strata-vision (tools/vision: llama.cpp's mtmd at the pinned commit), with ggml's Metal backend: the encoder
    runs on the GPU."""
    stamp = ROOT / "build-vision" / "BUILD.json"
    src = source_digest()
    try:
        if VISION_EXE.exists() and json.loads(stamp.read_text(encoding="utf-8")).get("source") == src:
            S.ok(f"image encoder: {VISION_EXE}")
            return VISION_EXE
    except (OSError, ValueError):
        pass
    S.say("  Building the image encoder (1-3 minutes) ...")
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    log = ROOT / "build-vision.log"
    with open(log, "w", encoding="utf-8") as f:
        for cmd in ([tool("cmake"), "-S", str(ROOT / "tools" / "vision"), "-B", str(ROOT / "build-vision"), "-G", "Ninja",
                     "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_MAKE_PROGRAM={tool('ninja')}", f"-DLLAMA_DIR={llama}",
                     "-DSTRATA_VISION_CUDA=OFF", "-DSTRATA_PORTABLE=OFF",
                     "-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON"],
                    [tool("ninja"), "-C", str(ROOT / "build-vision"), "strata-vision"]):
            if subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env) != 0:
                S.fail(f"the image encoder did not build (see {log})")
    stamp.write_text(json.dumps({"source": src}), encoding="utf-8")
    S.ok(f"image encoder built: {VISION_EXE}")
    return VISION_EXE


def mac_config(exe: Path, args: list, pack: Path, model_name: str, tag: str, port: int) -> dict:
    return {"exe": str(exe), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
            "model_name": model_name, "log": str(ROOT / f"strata-{tag.lower()}.log"), "lib_dirs": [], "port": port,
            "backend": "metal"}


def mac_configs() -> list[Path]:
    out = []
    for p in S.installed_configs():
        try:
            if json.loads(p.read_text(encoding="utf-8-sig")).get("backend") == "metal":
                out.append(p)
        except (OSError, ValueError):
            pass
    return out


def start(cfg_path: Path, port: int | None, open_browser: bool = True) -> int:
    """serve/server.py with this config (setup.py's start without the NVIDIA / AMD card checks)."""
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    missing = [p for p in [cfg["exe"], *[a for a in cfg["args"] if a.endswith(".gguf")]] if not Path(p).exists()]
    if missing:
        S.fail(f"{cfg_path.name} refers to missing files: {missing[0]}", "run ./setup.sh --setup to repair it")
    cfg_path.touch()
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(port or cfg.get("port", 8080))]
    if open_browser and cfg.get("open_browser") is not False:
        cmd.append("--open")
    S.say()
    S.say(f"  Starting {cfg.get('model_name', 'the model')} on Metal: the engine maps the model file and is ready in about "
          "half a minute (longer the first time after a restart, while macOS reads the file).")
    S.say("  Closing this window stops the model.")
    return subprocess.call(cmd)


def setup_poly(a, info: dict, data: Path, elsewhere: list) -> int:
    """Shared GGUF catalog and server, with native strata-poly Metal kernels."""
    fam = S.FAMILIES[a.family]
    sizes = S.family_models(a.family)
    if a.model and a.model not in sizes:
        S.fail(f"{fam['title']} has no {a.model} file", "choose one of: " + ", ".join(sizes))
    # Unified memory is shared by the mapped weights, GPU buffers and macOS.
    # Prefer a smaller download; do not apply the discrete-VRAM sizing rule.
    model = a.model or min(sizes, key=lambda m: sizes[m]["download_gb"])
    if not a.model and not a.yes:
        names = list(sizes)
        for i, m in enumerate(names, 1):
            S.say(f"  {i}) {m}: {sizes[m]['download_gb']:.1f} GB")
        model = names[int(S.ask("Which size?", [str(i) for i in range(1, len(names)+1)],
                                str(names.index(model)+1), a.yes))-1]
    d = sizes[model]
    S.warn("strata-poly on Metal is experimental: kernel parity checks pass; broader full-model validation is pending (docs/MAC.md)")
    if d["download_gb"] + OVERHEAD_GB > info["ram_gb"] * GIB_GB:
        S.warn("the model plus working memory may exceed this Mac's unified memory")
    tag = fam["tag"] + model
    ctx = a.context or 8192
    if ctx <= 0:
        S.fail("--context must be positive")
    vision = a.vision == "yes"
    if vision and not fam.get("vision"):
        S.fail(f"{fam['title']} has no image encoder")
    models_root = Path(a.models_dir) if a.models_dir else data / "models"
    folder = Path(a.gguf_dir) if a.gguf_dir else models_root / tag
    files = [folder / S.model_file(fam, model, i) for i in range(1, S.model_shards(fam, model)+1)]
    if a.gguf_dir and not a.update:
        for file in files:
            if not file.is_file():
                S.fail(f"missing local model: {file}")
    S.step(3, "Python packages")
    S.pip_install(S.requirement_lines() if S.REQUIREMENTS.exists() else S.PY_PACKAGES,
                  "numpy, jinja2, regex, cmake, ninja and server packages")
    S.step(4, "the strata-poly engine (Metal)")
    llama = S.get_llama_cpp()
    if a.build:
        (BUILD_DIR / "BUILD-poly.json").unlink(missing_ok=True)
    exe = build_engine(llama)
    if a.update:
        return 0

    def asset(path: Path, url: str) -> Path:
        if path.is_file() and (S.done(path) or S.whole_shard(path)):
            S.ok(f"using {path}")
        else:
            if a.no_download_models:
                S.fail(f"model downloads disabled; missing complete file: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            S.download(url, path)
        S.verify_sha256(path, *fam["sha256"][path.name])
        return path

    S.step(5, f"{fam['title']} {model}")
    base = d.get("hf", fam["hf"]).format(q=model)
    for file in files:
        asset(file, base + file.name)
    S.check_shards(files)
    draft = None
    if d.get("draft_model"):
        draft = folder / Path(d["draft_model"]).name
        draft = asset(draft, base + d["draft_model"])
    if d.get("draft") != "own" and fam.get("draft"):
        draft = S.find_in([data, *elsewhere], fam["draft"]) or data / fam["draft"]
        if not draft.is_file():
            if a.no_download_models:
                S.fail(f"model downloads disabled; missing draft: {draft}")
            S.run([sys.executable, str(ROOT / "tools" / "glm_draft_fetch.py"), "fetch", "--out", str(draft)])
        S.run([sys.executable, str(ROOT / "tools" / "glm_draft_fetch.py"), "verify", "--out", str(draft)])
    mmproj = vexe = None
    if vision:
        mmproj = asset(folder / fam["mmproj"], fam["mmproj_hf"] + fam["mmproj"])
        vexe = build_vision(llama)

    S.step(6, "preparing the tokenizer")
    pack = S.find_in([data, *elsewhere], f"packs/{tag.lower()}") or data / "packs" / tag.lower()
    if not (pack / "tokenizer" / "vocab.json").exists():
        S.run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"), "--gguf", str(files[0]), "--out", str(pack)],
              env=dict(os.environ, STRATA_GGUF_PY=str(llama / "gguf-py")))
    profile = pack / "expert-profile.txt"
    shipped = d.get("profile", fam.get("profile"))
    if shipped and not profile.exists():
        pack.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "data" / shipped, profile)
    args = ["--model", str(files[0]), "--max-context", str(ctx), "--expert-profile", str(profile)]
    if draft:
        args += ["--draft-model", str(draft)]
    if fam.get("tail_skip"):
        args += ["--tail-skip", str(fam["tail_skip"])]
    if info.get("perf_cores"):
        args += ["--threads", str(info["perf_cores"])]
    for flag, value in (("--conversation-cache-mib", a.conversation_cache_mib),
                        ("--vram-reserve-mib", a.vram_reserve_mib)):
        if value is not None:
            if value < 0:
                S.fail(f"{flag} cannot be negative")
            args += [flag, str(value)]
    cfg = mac_config(exe, args, pack, f"{fam['name']}-{model.lower()}", tag, a.port or 8080)
    cfg["display_name"] = f"{fam['title']} {model}"
    for key in ("host", "api_key"):
        if getattr(a, key):
            cfg[key] = getattr(a, key)
    if a.browser is not None:
        cfg["open_browser"] = a.browser
    if a.parallel is not None:
        cfg["parallel"] = max(1, a.parallel)
    if vision:
        cfg["vision"] = {"exe": str(vexe), "mmproj": str(mmproj), "model": str(files[0]),
                         "gpu": True, "max_tokens": a.vision_tokens or fam.get("vision_most", 1024)}
    S.poly_kv_whole(cfg)
    cfg_path = ROOT / f"strata-{tag.lower()}.json"
    previous = P.read_config(cfg_path)
    P.apply_options(cfg["args"], a, previous)
    headers, why = P.read_headers([*files, *([draft] if draft else [])])
    if getattr(a, "poly_kv", None) == "q8_0" and (
            not headers or headers[0].metadata.get("general.architecture") != "qwen35moe"):
        S.fail("--poly-kv q8_0 requires a verified Qwen35MoE GGUF header")
    estimate = P.estimate(headers, ctx, kv=P.flag_value(cfg["args"], "--kv", "auto"),
                          memory=P.flag_value(cfg["args"], "--metal-memory", "balanced"),
                          parallel=cfg.get("parallel", previous.get("parallel", 1)))
    if why:
        estimate["reason"] = why
    P.report(estimate, S.say)
    if P.flag_value(cfg["args"], "--metal-memory") == "balanced":
        S.say("  balanced KV grows in 4096-token steps with 1024 tokens ahead, up to the chosen context")
    S.write_setup_config(cfg_path, cfg, sampling=fam.get("sampling"))
    script = S.write_run_script(tag, cfg_path, cfg["port"], cfg.get("open_browser") is not False)
    S.ok(f"start script: {script.name}")
    return 0 if a.no_start else start(cfg_path, a.port, a.browser is not False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list-models", action="store_true",
                    help="list all model families and quantizations from the shared installer")
    ap.add_argument("--family", choices=list(model_catalog()), default=None,
                    help="model family (quantization names are scoped to this family)")
    ap.add_argument("--model", help="quantization from the chosen family (see --list-models)")
    ap.add_argument("--no-download-models", action="store_true", help="use existing GGUF, draft and projector files only")
    ap.add_argument("--parallel", type=int)
    ap.add_argument("--vision-tokens", type=int)
    ap.add_argument("--conversation-cache-mib", type=int, default=None)
    ap.add_argument("--vram-reserve-mib", type=int, default=None)
    ap.add_argument("--context", type=int)
    ap.add_argument("--port", type=int)
    ap.add_argument("--host")
    ap.add_argument("--api-key")
    ap.add_argument("--no-browser", dest="browser", action="store_false", default=None)
    ap.add_argument("--data-dir")
    ap.add_argument("--models-dir")
    ap.add_argument("--gguf-dir")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--source", choices=S.SOURCES, default=None,
                    help="where model files come from: auto (huggingface), huggingface, modelscope; "
                         "same as STRATA_SOURCE")
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--no-start", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--build", action="store_true", help="build the engine again even when it is current")
    ap.add_argument("--vision", choices=["yes", "no"],
                    help="let the model read images (the image encoder on the GPU, built here; default: asked, no)")
    P.add_arguments(ap)
    a, rest = ap.parse_known_args(argv)
    if a.list_models:
        for family, entry in model_catalog().items():
            S.say(f"{family}: {entry['title']} — {entry['metal_engine']} Metal")
            for model, spec in entry["models"].items():
                size = f" ({spec['download_gb']:.1f} GB download)" if "download_gb" in spec else ""
                S.say(f"  {model}{size}")
        return 0
    try:
        P.validate(a, MAC_POLY_FAMILIES)
    except ValueError as e:
        ap.error(str(e))
    if a.host and a.host not in ("127.0.0.1", "localhost", "::1") and not a.api_key:
        S.fail("a non-loopback --host requires --api-key")
    if rest:
        S.warn("not used on a Mac: " + " ".join(rest))
    # setup.py hands a Mac over before its own parser runs, so --source has to be honoured here too:
    # source_hint() recommends it on every platform, and a Mac that dropped it failed the download
    # with the same unreachable host (#2363).
    if a.source:
        os.environ["STRATA_SOURCE"] = a.source
    S.say("Puddle on a Mac (Apple Silicon, Metal)")
    data, elsewhere = S.data_folder(a.data_dir)

    have = mac_configs()
    if have and not (a.setup or a.family or a.model or a.check or a.no_start or a.update or a.build or P.requested(a)):
        return start(have[0], a.port, a.browser is not False)

    # ---- 1. the Mac
    S.step(1, "checking your Mac")
    info = mac_info()
    S.ok(f"{info['chip']}, {info['ram_gb']:.0f} GB unified memory, macOS {info['macos'] or '?'}")
    problems = mac_problems(info)
    for p in problems:
        S.warn(p)
    free = S.free_gb(data if data.exists() else ROOT)
    S.ok(f"free disk: {free:.0f} GB at {data}")
    if a.check:
        return 1 if problems else 0
    if problems:
        S.fail(problems[0])

    # ---- 2. the model
    S.step(2, "your choices")
    family = a.family
    if family is None:
        fams = list(MAC_POLY_FAMILIES)
        for i, f in enumerate(fams, 1):
            d = S.FAMILIES[f]
            S.say(f"  {i}) {d['title']:20s} {d['by']} - {d['about']}")
        family = fams[int(S.ask("Which model?", [str(i) for i in range(1, len(fams) + 1)], "1", a.yes)) - 1]
    a.family = family
    S.ok(f"model: {S.FAMILIES[family]['title']}")
    return setup_poly(a, info, data, elsewhere)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        S.say("\nstopped.")
        sys.exit(1)
