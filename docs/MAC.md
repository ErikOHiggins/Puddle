# Puddle on a Mac (Apple Silicon, Metal)

strata-poly has a Metal port (`metal/`), from PolyStrata: the host code and the server are the PC's, and the GPU
kernels are Metal ones. `./setup.sh` on a Mac runs `metal/setup_mac.py`, which builds strata-poly from source on your
Mac (5-15 minutes the first time) and sets up a model the way `START-HERE.bat` does on Windows.

**Status: untested in Puddle.** Nobody has run Puddle on a Mac yet.

- **Laguna-S and Laguna-XS do not run on a Mac yet.** The Metal build does not compile the Laguna code. Adding it is
  planned, and it will be untested until someone with a Mac runs it.
- **The Nemotrons are not offered on a Mac.** The Metal build has no kernels for their Mamba-2 layers, and setup
  compiles llama-server for NVIDIA cards only ([NEMOTRON.md](NEMOTRON.md)).
- **Gemma 4** uses an architecture the Metal port has (gemma4). Upstream ran models of the Metal port's architectures
  on Metal, but not these files.

Reports from a Mac are welcome: [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md).

## What you need

- **An Apple Silicon Mac (M1 or newer).** Intel Macs cannot run it.
- **macOS 15 or newer.** The engine keeps its buffers resident through a residency set, which macOS 15 introduced.
- **The Command Line Tools:** `xcode-select --install`. That is the compiler. Xcode itself is not needed.
- **Unified memory for the model:** the GPU maps the model file. Metal lets one program keep about 3/4 of the memory
  resident (107.5 GB of 128 GB, measured upstream on an M5 Max).

## Install

```
git clone https://github.com/ErikOHiggins/Puddle && cd Puddle
./setup.sh                 # asks the model and size; --yes takes the recommended answers
```

What the first run does (each step is skipped when it is already done):

1. **Python.** It needs Python 3.10 or newer, but macOS ships 3.9. If that is all your Mac has, `setup.sh` downloads
   a standalone Python 3.12 (25 MB, pinned by SHA-256) into `.python`. No sudo, no Homebrew.
2. **Checks your Mac:** the chip, the memory, the macOS version, the Command Line Tools and the free disk.
3. **Python packages** into `.venv`. CMake and Ninja come from pip.
4. **Builds the engine** (`build-metal/strata-poly`), with Apple's metal-cpp headers (pinned by SHA-256) and llama.cpp's
   ggml at the commit the PC build pins.
5. **Downloads the model.** The download resumes if interrupted. `--gguf-dir` uses files you already have.
6. **Writes `strata-<model>.json` and `run-<model>.sh`, then starts the server:** the OpenAI-compatible API on
   `http://127.0.0.1:8080/v1`, the Anthropic API on `/v1/messages`, the chat page on `http://127.0.0.1:8080/`.

Options: `--family NAME --model SIZE` (all choices: `./setup.sh --list-models`), `--vision yes|no`, `--context N`,
`--port`, `--host` / `--api-key`, `--no-browser`, `--data-dir`, `--models-dir`, `--gguf-dir`, `--no-download-models`
(use files you already have, download nothing), `--setup` (another model or new settings), `--no-start`, `--check`
(only check the Mac), `--build` (build the engine again).

## Metal settings

- Optimized dispatch (`--metal-optimize`, on by default) submits cached GPU experts while CPU experts run.
  `--no-metal-optimize` turns it off.
- `--metal-memory balanced` (the default) grows the KV buffers in 4,096-token steps up to the selected context and
  budgets expert copies against the available unified memory. `--metal-memory legacy` allocates everything at load.
- The KV cache is fp16, or q8_0 for qwen35moe models with compatible heads (setup offers none). `--poly-kv q8_0|fp16` chooses it
  at setup; other architectures refuse q8_0.

## Build and regression checks

The checks need metal-cpp and llama.cpp at setup's pinned commits; they generate their own inputs and need no model
files:

```sh
cmake -S metal -B build-metal-check -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_METAL_POLY=ON \
      -DMETAL_CPP_DIR=/path/to/metal-cpp -DSTRATA_GGML_DIR=/path/to/llama.cpp
cmake --build build-metal-check
ctest --test-dir build-metal-check --output-on-failure
```

They compare strata-poly's Metal kernels with CPU references and run small generated GGUF files; they check execution
and state handling, not a model's quality. Without `-DSTRATA_METAL_POLY=ON` only the runtime check (`runtime_hello`)
is built.

## How it works

The code is in `metal/`. The PC host code compiles for macOS: `metal/include/metal_compat` maps the CUDA runtime calls
it makes onto a small Metal runtime (`metal/src/gpu`), and strata-poly's Metal kernels are in `metal/src/poly/`. The
experts are read in place from the mapped model file. Changes to shared host code are behind `STRATA_GPU_METAL` or
`__APPLE__`; the Metal build has its own CMake project.
