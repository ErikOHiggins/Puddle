# Questions

The numbers are measured on the machines named next to them; how is in [BACKENDS.md](BACKENDS.md).

### What is Puddle, and how does it relate to PolyStrata and Strata?

Puddle is a fork of [PolyStrata](https://github.com/simplexai-labs/PolyStrata), which is built on
[Strata](https://github.com/Niko1221/Strata). Strata runs Qwen3.8-Flash-Next on its own engine; PolyStrata added a
second engine, strata-poly, for other mixture-of-experts models, built on Strata's code (its quantized matvec kernels,
GGUF reader, draft policy, conversation memory and the llama.cpp MMQ kernels). Puddle keeps strata-poly with
PolyStrata's server, web app and installer, offers five models (Laguna-S 2.1, Laguna-XS 2.1, Nemotron 3.5 Bolt,
Nemotron 3.5 Super and Gemma 4 26B-A4B), and added Laguna's and the Nemotrons' architectures. Setup runs Nemotron 3.5
Super on strata-poly and Nemotron 3.5 Bolt on llama.cpp's llama-server ([NEMOTRON.md](NEMOTRON.md)). Strata's Qwen3.8 engine and PolyStrata's other models
have been removed.

### How do I know the output is right?

It reads the same GGUF files llama.cpp reads, and its output is compared with llama.cpp's on the same file. For
Laguna-S UD-Q3_K_XL on echo (two RTX 5060 Ti), with llama.cpp choosing each token: the same first choice at 95.3% of
384 steps over six prompts, and every first difference a near tie ([BACKENDS.md](BACKENDS.md#laguna-on-two-16-gb-cards)).
`laguna/parity.py` runs that comparison. PolyStrata's checks of Gemma 4 and the qwen35moe models are in
[BACKENDS.md](BACKENDS.md#how-each-backend-was-checked). Guessing tokens ahead does not change the answer: the guesses
are checked like any token.

### How fast will it be on my card?

What was measured is in the [README](../README.md#how-fast-is-it): Laguna-S on two 16 GB cards, Nemotron 3.5 Bolt on the
same cards with llama-server alone, Gemma 4 on PolyStrata's machines. Laguna-XS has not been measured yet. `./bench.sh` (`BENCH.bat`) measures your PC, and
[COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md) says what to send.

### How can I compare it with llama.cpp myself?

With `bench/bench_vs_llama.py`. It runs Puddle and llama.cpp's `llama-server` in turns on the same GGUF and the same
prompts and prints the medians of both ([how to run it](COMMUNITY_BENCHMARKS.md#comparing-with-llamacpp)). On echo,
llama.cpp with every expert on the CPU wrote Laguna-S at 11.3 to 11.5 tokens/s and read prompts at 51 to 73.

### Does it run on Radeon / AMD?

strata-poly compiles for AMD cards (setup compiles it on your PC: ROCm from your system or AMD's wheels on Linux,
AMD's wheels on Windows), as a beta. Puddle's models have not been run on an AMD card yet; PolyStrata's contributors
ran other models of the same engine on an RX 7600 and an RX 9070 XT ([AMD_HIP.md](AMD_HIP.md)). No pictures on AMD.

### Will it run on a smaller card?

Setup's figures for the VRAM a model takes beside its experts: Laguna-S about 8 GB (from echo), Laguna-XS about 5 GB
(an estimate), Gemma 4 about 4.1 to 5.8 GB by size. Setup wants 2 GB more than that on the card. Laguna-XS and Gemma 4
are the ones to try on an 8 GB card. Nemotron 3.5 Bolt keeps every layer on the cards and wants 30 GB of VRAM
together: setup refuses it on less.

### What about older cards like a GTX 1050?

Pascal and Volta cards have an experimental CUDA 12 path ([OLDER_GPUS.md](OLDER_GPUS.md)); nobody has run Puddle's
models on one.

### Can I run a fine-tune, or another model with the same architecture?

The engine reads the architecture and the weights from the GGUF, so a fine-tune that keeps the architecture of Laguna
(`laguna`) or Gemma 4 (`gemma4`) should run. `./setup.sh --custom-gguf <file>` reads its header,
picks the family and installs it as it is; `./setup.sh --inspect <file>` says what it holds first. Setup's RAM and
VRAM estimates are then those of the nearest size. For Nemotron 3.5 Bolt setup installs NVIDIA's own file only.

### How do I remove a model?

There is no command for it. A model's files are these, where `<model>` is the name in its run config's file name
(`laguna-s21-ud-q3_k_xl` for `strata-laguna-s21-ud-q3_k_xl.json`):

- its folder in `models/`: the GGUF files, and `mmproj` for pictures
- `packs/<model>/`: its tokenizer and expert profile
- `strata-<model>.json` and `strata-<model>.json.bak`: its run config
- `strata-<model>.log`
- `run-<model>.bat` or `run-<model>.sh`

`models/` and `packs/` are in the `Puddle-data` folder. The engine (`engine-poly/`) is shared by every model: keep it.
`engine-llama/` is Nemotron 3.5 Bolt's llama-server alone.

### Which size should I take?

Setup takes the default size when your PC has the RAM for it: Laguna-S's UD-Q3_K_XL, Gemma 4's UD-Q3_K_M. The larger sizes keep more bits in the experts and want more RAM; poolside's Laguna-S Q4_K_M (96 GB) has 8
of its expert layers in BF16, which run on the processor, and has not been run yet.

### Does it work on Windows?

Setup compiles strata-poly on Windows with Visual Studio's build tools and the CUDA toolkit, and offers to install them.
PolyStrata ran Gemma 4 there; Laguna has not been run on Windows, and setup says so before it installs it.

### Can I use it with Claude Code, Codex or other apps?

Yes. It serves OpenAI- and Anthropic-compatible APIs on localhost, with tool calls. Claude Code:
`ANTHROPIC_BASE_URL=http://127.0.0.1:8080`, and add `"anthropic_thinking": "on_request"` and `"fit_max_tokens": true`
to `strata-<model>.json`. Codex CLI uses the Responses API at `/v1/responses`. poolside's pool agent needs
`"keepalive_events": true`. Details: [API.md](API.md#apps-and-agents).

### A long chat stops with "request exceeds the context window". Can it drop old messages instead?

Yes, if you ask for it: `"drop_oldest": true` in `strata-<model>.json` leaves out the oldest turns until the prompt
leaves room for the answer (the system prompt and the last message stay), and `"truncate_middle": true` trims the
middle of one very long text prompt. Both are off by default ([API.md](API.md#context)).

### Can the model see what it thought in earlier turns?

With Laguna, if you ask for it: `"preserve_thinking": true` in `strata-<model>.json`, or
`"chat_template_kwargs": {"preserve_thinking": true}` with a request. The server also remembers the thinking of the
answers it wrote, so an app that sends an answer back without it still gets it, as long as the answer's text comes back
unchanged.

### Why are the answers the same every time?

Setup writes each model's sampling into `strata-<model>.json` (`"sampling"`), so this happens only with a config
that has no such block (one made by hand): a request that sends no temperature is then answered greedy, the engine's
default. Add a block, or run setup again for the model, or set the sampling in your app
([API.md](API.md#sampling-and-limits)).

### Can several people use one server at once?

Yes. By default the server answers one request at a time and the others wait their turn. With `"parallel": N` in
`strata-<model>.json` (or `--parallel N` when you run setup) it answers up to N together ([BATCHING.md](BATCHING.md)).

### Who maintains it?

It is a personal fork, [ErikOHiggins/Puddle](https://github.com/ErikOHiggins/Puddle), MIT licensed like PolyStrata and
Strata.
