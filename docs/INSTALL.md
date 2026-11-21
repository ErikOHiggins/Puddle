# Installing Puddle

Everything about installing, starting, updating and configuring Puddle, on Windows and Linux, with an NVIDIA or an
AMD graphics card. The short version is in the [README](../README.md#install); an AI coding assistant can do all of
this for you with [AI_SETUP.md](AI_SETUP.md). A Mac: [MAC.md](MAC.md).

> **On this page:** [What you need](#what-you-need) · [Windows](#windows) · [Linux](#linux) ·
> [AMD cards](#amd-cards) · [Several cards](#two-or-three-cards) · [Older CPUs](#older-cpus-experimental) ·
> [Updating](#updating) · [Where things are stored](#where-things-are-stored) · [Setup's questions](#setups-questions) ·
> [All options](#options-without-questions)

## What you need

| | |
| --- | --- |
| **GPU** | **NVIDIA** RTX 20 series or newer, 8 GB of VRAM or more; Laguna-S wants 16 GB or two cards, Nemotron 3.5 Bolt 30 GB of VRAM together on llama-server (one 32 GB card or two 16 GB, NVIDIA only) or 16 GB on strata-poly (`nemotron-bolt-poly`), Nemotron 3.5 Super 16 GB or more. Laguna and both Nemotrons have run on two RTX 5060 Ti 16 GB. **AMD** cards are experimental: [AMD cards](#amd-cards). Pascal / Volta cards: [OLDER_GPUS.md](OLDER_GPUS.md). |
| **RAM** | By model and size, setup's estimate beside the card: Laguna-S about 51 to 93 GB beside a 16 GB card, Laguna-XS about 17 GB beside a 16 GB card, Nemotron 3.5 Bolt about 16 GB on llama-server (it keeps every layer on the cards; not measured) and about 17 GB beside a 16 GB card on strata-poly, Nemotron 3.5 Super about 47 GB beside two 16 GB cards and 62 beside one, Gemma 4 15 to 21 GB beside an 8 GB card. `--check` prints it for your PC. |
| **CPU** | x86-64 with AVX2 (any Intel/AMD desktop CPU from the last ~8 years). Without AVX2 it is experimental and slow: [Older CPUs](#older-cpus-experimental). |
| **Disk** | The model: 11 to 96 GB by size (Laguna-S UD-Q3_K_XL 54 GB, Laguna-XS 20 GB, Nemotron 3.5 Bolt 22 GB, Nemotron 3.5 Super 66 GB, Gemma 4 11 to 17 GB), plus 1.2 to 1.7 GB for pictures, a few GB for the build, and on Linux with an AMD card ~10 GB for ROCm. An NVMe SSD makes starts faster. |
| **OS** | Linux or Windows 10/11. Laguna and the Nemotrons have run on Linux only so far: on Windows setup says so (and asks first for Laguna and Nemotron 3.5 Super). |
| **Driver** | **NVIDIA:** version 580 or newer ([nvidia.com/drivers](https://www.nvidia.com/drivers) or the NVIDIA App); older ones with `--cuda 12`. **AMD:** on Linux the kernel's amdgpu driver; on Windows a current AMD Software: Adrenalin Edition driver. |

The driver is the only thing you install yourself. Everything else is set up for you the first time: Python 3.12 if
you have none (for your user account, no admin), a private Python environment in `.venv/`, llama.cpp's source at a
pinned commit in `third_party/`, the engine, and the model. Setup always compiles the engine, strata-poly, for your
card: it needs a CUDA toolkit (and on Windows Visual Studio's C++ build tools), which it offers to install, and the
first compile takes 10 to 20 minutes. For Nemotron 3.5 Bolt it compiles llama.cpp's llama-server instead
(into `engine-llama/`, once): [NEMOTRON.md](NEMOTRON.md).

## Windows

1. [Download Puddle](https://github.com/ErikOHiggins/Puddle/archive/refs/heads/main.zip) and unzip it (or
   `git clone` it).
2. Double-click **`START-HERE.bat`**.
3. Answer a few questions, or press Enter each time for the recommended choice ([the questions](#setups-questions)).

Then it compiles the engine, downloads the model (you can stop and it picks up where it left off) and **starts the
model**. Your browser opens the app at `http://127.0.0.1:8080`.

**Next time**, double-click `START-HERE.bat` again: it starts right away, nothing is downloaded twice. Close its window
to stop the model. `SETUP.bat` (the same as `START-HERE.bat --setup`) installs another model or changes the settings.

## Linux

```bash
./setup.sh
```

The same questions, the same automatic install (it uses `sudo apt` for Python if it has to, and asks before it
installs a CUDA toolkit), and the same start: `http://127.0.0.1:8080`. Later runs of `./setup.sh` (or
`./run-<model>.sh`) start the model directly; `./setup.sh --setup` installs another model or changes the settings.

On Nix/NixOS, you can use the flake:

```bash
nix develop -c ./setup.sh
```

## AMD cards

The steps are the same as with NVIDIA. Setup finds the Radeon card and chooses the AMD (HIP) engine by itself on a PC
with no NVIDIA card it can use; `--backend hip` chooses it on a PC that has both. On Linux setup uses a system ROCm 7
when there is one, or installs ROCm into `.venv` from AMD's wheels (~10 GB, no sudo); on Windows it installs AMD's
ROCm wheels into `.rocm-win`. Either way it compiles the engine for the card (10-20 minutes, once). There are no
pictures on an AMD card for now (the image encoder is a CUDA program). Puddle's models have not been run on an AMD card
yet. Cards, ROCm versions and what upstream measured: [AMD_HIP.md](AMD_HIP.md).

## Two or three cards

Setup lists your cards, says which ones it can use, and asks whether to share the model across them (recommended when
two can). Or choose yourself: `--gpus 0,1` (both, remembered), `--gpus all`, or `--gpu 0` (one card, this start
only). Each card keeps the layers of its share and their experts; prompts flow through the cards in turn. Laguna-S
and Laguna-XS run this way on two RTX 5060 Ti 16 GB. See [MULTI_GPU.md](MULTI_GPU.md).

## Older CPUs (experimental)

On a CPU without AVX2 setup warns that it is experimental and slow, and goes on: it compiles the engine on that PC
anyway. The share of the experts the CPU computes runs slower there. Not measured with Puddle's models.

## Updating

**`UPDATE.bat`** (Linux: `./update.sh`) updates Puddle without starting the model. In a `git clone` it runs
`git pull`, then does what `START-HERE.bat` does before a start: the engine is compiled again when its source changed,
the Python packages, and each installed model's settings. The model files are not touched and no question is asked.
Close the model's window first (a running engine cannot be replaced). In a copy that was downloaded as a zip it says to
download the new zip: it cannot fetch new files itself.

Or by hand: download the new version and unzip it anywhere (or `git pull`), then run `START-HERE.bat` (Linux:
`./setup.sh`) in it. The model files are kept in a `Puddle-data` folder next to your Puddle folder, so a new copy finds
them and sets itself up the same way: nothing big is downloaded again.

## Where things are stored

- **Your chats: only in your browser.** The Chat tab keeps the conversation, its settings and the API key you typed
  in the browser's local storage (`strata.*` keys), not on the server and not in the Puddle folder. Pictures are not
  kept, only their names.
- **How the model starts:** `strata-<model>.json` in the Puddle folder (context, GPUs, host, API key, ...), written by
  setup; next to it `run-<model>.bat` / `.sh`, the log `strata-<model>.log` and, when you use "Use for other apps too",
  `strata-<model>.shared-settings.json`. Running setup again for the same model rewrites the keys setup writes (`exe`,
  `args`, `port`, `gpu`, `host`, `api_key`, `vision`, ...) and keeps the ones you added (`sampling`, `mcp_servers`,
  `cors_origins`, ...); the earlier file is kept as `strata-<model>.json.bak`. Engine options you added to `"args"` by
  hand are not carried over: setup names them, and you add them again. Setup writes the model's sampling (`"sampling"`,
  [API.md](API.md#sampling-and-limits)) when the config has no block of yours.
- **The model files** (`models/`, `packs/`): in **`Puddle-data` next to the Puddle folder**, or wherever `--data-dir`
  put them. A `packs/<model>/` folder holds the tokenizer setup made from the GGUF and the expert profile the engine
  keeps.
- **Where that data folder is:** `%APPDATA%\Puddle\settings.json` on Windows, `~/.config/puddle/settings.json` on
  Linux. Puddle never moves files out of a PolyStrata install, so a PolyStrata data folder can be shared through a link.
- **The program itself:** `.venv/`, `engine-poly/`, `build-poly/` and `third_party/` in the Puddle folder; for
  Nemotron 3.5 Bolt also `engine-llama/` and `build-llama-server/`.
- **Where github.com cannot be reached:** setup downloads llama.cpp's source from GitHub at the commit
  `LLAMA_CPP_COMMIT` in `setup.py` names. Put that commit's tree in `third_party/llama.cpp` yourself and setup uses it.

## Setup's questions

- **Which model and size?** Laguna-S, Laguna-XS, Nemotron 3.5 Bolt (on llama-server, or on strata-poly), Nemotron 3.5 Super or Gemma 4, and a size of it; each line says its
  download and the RAM setup expects on your card.
- **Which cards?** One, or several sharing the model.
- **How much context?** How much text it can keep in mind at once: 8K, 32K (recommended), 64K or 128K, each with the
  VRAM it takes from the expert cache. Gemma 4 also offers 256K, its files' own length (not yet run at that length). `--context N` takes any length up to 1048576. Nemotron 3.5 Bolt on llama-server is not asked:
  168K on two 16 GB cards (as it runs on echo), else the largest that fits, chosen by llama-server each time it loads.
- **Images?** For Gemma 4 and the Nemotrons: whether the model should also read pictures. On strata-poly the image
  encoder runs on the GPU from 15.5 GB of VRAM, else on the CPU (Gemma 4's: a few seconds a picture; the Nemotrons'
  has not been run on the CPU); Bolt's on llama-server is on the cards with the model (recommended: yes).
- **Several conversations in RAM?** The conversation cache keeps conversations a coding agent switches between in RAM,
  so they are not read again (up to 8 GB of RAM). Not asked for Bolt on llama-server: llama-server keeps earlier prompts
  in RAM itself (up to 8 GiB).

With `--yes` setup takes the recommended answer to every question.

## Options without questions

```
./setup.sh --setup                               install another model, or change context / images
./setup.sh --family laguna-s --model UD-Q3_K_XL --context 262144 --yes      no questions
./setup.sh --gguf-dir /models/laguna             use GGUF files you already have
./setup.sh --custom-gguf /models/my.gguf         install one GGUF; setup reads its header and picks the family
./setup.sh --inspect /models/my.gguf             what a GGUF holds and whether Puddle runs it (no download)
./setup.sh --data-dir /mnt/big/Puddle-data       keep the model files somewhere else
./setup.sh --port 8081                           another port
./setup.sh --gpu 1                               another GPU (setup picks the one with the most VRAM)
./setup.sh --gpus 0,1                            several GPUs sharing the model
./setup.sh --vram-reserve-mib 2048               leave 2 GB of VRAM free for other programs (remembered)
./setup.sh --no-browser                          do not open the chat page when the model is ready (remembered)
./setup.sh --setup --backend hip                 the AMD engine on a PC that also has an NVIDIA card
./setup.sh --setup --cuda 12                     compile with CUDA 12 (Pascal / Volta, or a driver older than 580)
./setup.sh --setup --host 0.0.0.0 --api-key <secret>     reachable from other devices, with a key
./setup.sh --check                               only check this PC
```

On Windows the same options go to `START-HERE.bat`. `./setup.sh --help` lists them all.

**Leaving VRAM for other programs:** the engine fills the card's free VRAM with experts (the expert cache) and leaves
`--vram-reserve-mib` MiB free. For a game, a 3D program or another model beside it, leave more:
`./setup.sh --vram-reserve-mib 2048` writes it into the model's `strata-<model>.json` and starts it. By hand: add
`"--vram-reserve-mib", "2048"` to the config's `"args"` list and restart. The expert cache is then that much smaller,
so answers can be a little slower.

**Model files downloaded by hand:** setup's step 5 prints the folder it expects them in
(`Puddle-data/models/<family-tag><SIZE>/`, e.g. `Puddle-data/models/laguna-s21-UD-Q3_K_XL/`): put them there with their
original names, or point setup at them with `--gguf-dir`. `--inspect` reads only a file's headers (a few MB, also over
the network: a file, a folder, a URL, or `hf:owner/repo`) and says whether it is one of the files setup installs, one
stored like them, another file of a family's architecture, or not one Puddle runs. To let setup download from a
Hugging Face mirror, set `HF_ENDPOINT` first (`export HF_ENDPOINT=https://hf-mirror.com`): the same pinned revisions
and checks apply. `--source modelscope` downloads from ModelScope where a repository is mirrored there; that was not
checked for Puddle's models.

The server's own settings (sharing the GPU with other programs, MCP tools, CORS, API keys, the API itself):
[API.md](API.md).
