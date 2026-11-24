# AGENTS.md

Puddle runs three mixture-of-experts models - Laguna-S 2.1, Laguna-XS 2.1 and Occamy 1.0 - on a normal PC: one or
two NVIDIA or AMD graphics cards plus system RAM, on Windows or Linux (a Mac as a beta). Its engine is `strata-poly`
(`src/poly/`), with a Python server that has an OpenAI- and
Anthropic-compatible API and a web app (`serve/`), and a one-click installer (`setup.py`, started by `START-HERE.bat` /
`setup.sh`) that compiles the engine on the PC. strata-poly also runs the Nemotrons' architecture (`nemotron_h_moe`,
`src/poly/nemotron_h.cpp`) and Gemma 4's, and the server can run llama.cpp's `llama-server` as an engine
(`LlamaServerEngine`); setup has offered none of them since 2026-11-23 ([docs/NEMOTRON.md](docs/NEMOTRON.md)).

Puddle is a fork of [PolyStrata](https://github.com/simplexai-labs/PolyStrata), which is built on
[Strata](https://github.com/Niko1221/Strata). Strata's Qwen3.8 engine and PolyStrata's other models have been removed:
`src/` and `include/` hold strata-poly and the kernels it uses. strata-poly's `qwen35moe` class (`src/poly/qwen35.cpp`)
runs Occamy 1.0, a fine-tune of Qwen3.6-35B-A3B. Setup's family for it is `occamy`, with the tag `occamy10-`: its
folders in a data folder shared with PolyStrata are then not PolyStrata's (`occamy-`), whose pack the engine would
write its expert counts into.

## Which models belong

Open-weight models whose license puts no restriction on their use: **OpenMDW first** (Laguna-S 2.1 and Laguna-XS 2.1
are OpenMDW-1.1), **with Apache 2.0 accepted to fill a gap when required**: Occamy 1.0 35B-A3B, a general model that
reads pictures and documents beside Laguna, which reads none. A model is checked against this rule before it is
added. The OpenMDW index,
[models.openmdw.ai](https://models.openmdw.ai/), lists the candidates.

## Installing Puddle for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, Puddle's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

- How strata-poly works, its backends and what was measured: [docs/BACKENDS.md](docs/BACKENDS.md). The Nemotrons'
  engine code and its record (not offered by setup): [docs/NEMOTRON.md](docs/NEMOTRON.md). The server's API and every
  setting: [docs/API.md](docs/API.md).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); several GPUs: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Laguna's bring-up and measurement tools: [laguna/README.md](laguna/README.md).
- Tests that run without a GPU or downloads: setup's, `python -m unittest tools.test_setup_<name>`; the server's,
  `python -m unittest serve.test_<name>`; the tools', `python -m unittest tools.test_<name>`.
- strata-poly's own tests (C++/CUDA; those that need a card skip without one): `./update.sh --tests` compiles the
  engine with them in `build-poly-tests/` and runs them with ctest, `./update.sh --tests REGEX` the matching ones.
  The installed engine is not touched.
- A change to strata-poly that should leave the models' answers as they were: `tools/engine_regress.py --old <the
  engine before it> strata-<model>.json ...` runs both engines greedy on each installed model and compares the ids.
- strata-poly against llama.cpp on one GGUF: `tools/poly_parity.py --gguf <file>` (llama.cpp writes greedy answers,
  strata-poly follows them, and each step's first choice and probabilities are compared). It runs llama.cpp's
  llama-server from `engine-llama/`, which `./update.sh --llama-server` compiles.
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Contributing a change or report

- Search the open issues and pull requests first, and add to a thread that already covers your point.
- Open an issue with the form that fits (bug report, feature request or question).
- One change per pull request. Say what it changes and what it leaves alone.
- A new feature is opt-in, and the default path stays as it was. Say how you checked.
- Build every backend a file touches (CUDA, HIP, Metal) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).
