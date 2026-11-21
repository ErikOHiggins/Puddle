# Installing Puddle for a user (instructions for AI agents)

This page is written for an AI coding assistant (Claude Code, Cursor, Codex, GitHub Copilot, ...) that a user asked to
set up Puddle on their PC. Puddle runs Laguna-S 2.1, Laguna-XS 2.1, Nemotron 3.5 Bolt, Nemotron 3.5 Super or Gemma 4 26B-A4B locally on
one or two NVIDIA or AMD graphics cards plus system RAM, on its engine strata-poly (Nemotron 3.5 Bolt also on llama.cpp's
llama-server, opt-in, which setup compiles too), and serves an OpenAI- and Anthropic-compatible API on
`http://127.0.0.1:8080`. The human-oriented version of everything below is [INSTALL.md](INSTALL.md).

Work through the steps in order. Tell the user what you are doing in plain words; they may not be technical.

## 0. Ground rules

- **Never expose the server without an API key.** Keep the default `--host 127.0.0.1`. Only if the user asks for
  access from other devices, use `--host 0.0.0.0` **together with** `--api-key <a long random secret>`, and give the
  user the key. Never put a tunnel or port forward in front of a server without a key.
- Do not change the user's system beyond what setup does (setup installs Python for the user account if needed, the
  build tools it asks for, and everything else inside the Puddle folder and its `Puddle-data` folder). Ask before
  installing drivers.
- The model download is 11 to 96 GB, by model and size (step 3). Confirm the user is fine with that before you start,
  especially on a metered connection.
- Setup is long-running: it compiles the engine (10-20 minutes the first time) and downloads the model. Run it in the
  background or with a long timeout and poll its output; do not kill it because it is quiet for a while. It is
  resumable: running the same command again continues where it stopped.

## 1. Check the PC

Setup's own check prints the GPU(s), driver, RAM, CPU, PCIe link and which model sizes fit by its estimate, and
installs nothing except Python and the `.venv` it needs to run (step 2 gets the repository first):

```
Windows:  START-HERE.bat --check
Linux:    ./setup.sh --check
```

To check by hand:

| What | Windows (PowerShell) | Linux |
| --- | --- | --- |
| OS | `[Environment]::OSVersion` | `cat /etc/os-release` |
| NVIDIA GPU, VRAM, driver | `nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv` | the same |
| AMD GPU | `Get-CimInstance Win32_VideoController \| Select-Object Name, AdapterRAM` (AdapterRAM is capped at 4 GB; trust the model name) | `lspci \| grep -i -E 'vga\|display'`; VRAM: `cat /sys/class/drm/card*/device/mem_info_vram_total` (bytes) |
| RAM | `(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB` | `free -g` |
| Free disk | `Get-PSDrive -PSProvider FileSystem` | `df -h .` |

Requirements (details: [INSTALL.md](INSTALL.md#what-you-need)):

- **GPU:** NVIDIA RTX 20 series or newer, 8 GB of VRAM or more (Laguna-S wants 16 GB or two cards; Nemotron 3.5 Bolt
  wants 30 GB of VRAM together, one 32 GB card or two 16 GB, NVIDIA only). AMD cards are
  experimental ([AMD_HIP.md](AMD_HIP.md)); Pascal / Volta cards (P40, V100) have an experimental CUDA 12 path the user
  opts into ([OLDER_GPUS.md](OLDER_GPUS.md)). Integrated GPUs are not supported.
- **Driver:** NVIDIA 580 or newer. AMD on Linux: the kernel's amdgpu driver; on Windows: a current AMD Adrenalin
  driver. If the driver is missing or too old, tell the user to update it and restart; do not install drivers
  yourself unless they ask.
- **RAM and disk:** by model, in step 3. An NVMe SSD is best. **CPU:** x86-64 with AVX2.
- **OS:** Linux or Windows 10/11. Laguna has run on Linux only so far: on Windows setup says so and asks
  first (with `--yes`, an explicit `--family` is the consent).

If the PC does not meet them, say which part is missing and stop.

## 2. Get Puddle

```
git clone https://github.com/ErikOHiggins/Puddle.git
```

Without git: download https://github.com/ErikOHiggins/Puddle/archive/refs/heads/main.zip and unzip it. Pick a drive
with enough free space: the model files go to a `Puddle-data` folder **next to** the Puddle folder (or
`--data-dir <path>` on another drive).

## 3. Pick the model

RAM is setup's estimate beside the card named (a larger card or a second one needs less; `--check` prints it for this
PC):

| Model | `--family` | Sizes (`--model`), download | RAM | Pictures |
| --- | --- | --- | --- | --- |
| Laguna-S 2.1 | `laguna-s` | UD-Q3_K_XL 54 GB (default), UD-Q4_K_XL 73 GB, Q4_K_M 96 GB | about 51 / 71 / 93 GB beside a 16 GB card | no |
| Laguna-XS 2.1 | `laguna-xs` | Q4_K_M 20 GB | about 17 GB beside a 16 GB card | no |
| Nemotron 3.5 Bolt | `nemotron-bolt` | 24GB (NVIDIA's own 2-bit file) 22 GB, 1.7 GB more with pictures | about 17 GB beside a 16 GB card; every expert on two | yes |
| Nemotron 3.5 Bolt on llama-server (opt-in) | `nemotron-bolt-llama` | the same 24GB file (setup uses the one `nemotron-bolt` downloads) | about 16 GB (not measured); needs 30 GB of VRAM together | yes |
| Nemotron 3.5 Super | `nemotron-super` | UD-Q3_K_XL 66 GB | about 47 GB beside two 16 GB cards, 62 GB beside one | yes |
| Gemma 4 26B-A4B | `gemma4` | UD-Q3_K_M 13 GB (default), UD-Q2_K_XL, UD-Q4_K_M, QAT-UD-Q4_K_XL | about 15 to 21 GB beside an 8 GB card | yes |

Ask the user what they want it for. Laguna-S is the largest and the default; Laguna-XS and Gemma 4 fit a PC with
less RAM. Nemotron 3.5 Super is the largest model (120B-A12B) and wants the most RAM; it has run on Linux only. The
Nemotrons and Gemma 4 read pictures; Bolt on llama-server (opt-in) needs 30 GB of VRAM on NVIDIA cards together and
setup refuses it on less ([NEMOTRON.md](NEMOTRON.md)). With `--yes` and no `--model`, setup takes the first size whose RAM
estimate this PC meets. poolside's Laguna-S Q4_K_M keeps 8 of its expert layers in BF16 on the processor and has not
been run yet: take it only when the user asks for it.

## 4. Run setup without questions

From the Puddle folder (Windows: in `cmd`, or `cmd /c START-HERE.bat ...` from PowerShell):

```
Windows:  START-HERE.bat --yes --family laguna-xs --no-start
Linux:    ./setup.sh --yes --family laguna-s --no-start
```

The flags (all of them: `./setup.sh --help`):

| Flag | Meaning |
| --- | --- |
| `--yes` | take the recommended answer to every question (no prompts) |
| `--family laguna-s\|laguna-xs\|nemotron-bolt\|nemotron-bolt-llama\|nemotron-super\|gemma4` | the model |
| `--model SIZE` | the size (step 3) |
| `--context N` | context in tokens (default asked; `--yes`: 32768); up to 1048576. Nemotron 3.5 Bolt on llama-server: 168000 on two 16 GB cards, else the largest that fits (llama-server's `--fit`); 4096 to 262144 |
| `--vision yes\|no\|gpu\|cpu` | read pictures (Gemma 4; `--yes` leaves them off) or not (Nemotron 3.5 Bolt; `--yes` turns them on); none on AMD cards |
| `--gpu N` / `--gpus 0,1` / `--gpus all` | one card, or several sharing the model |
| `--backend cuda\|hip` | NVIDIA or AMD; chosen by itself on a PC with only one kind of card |
| `--cuda 12\|13` | the CUDA toolkit the engine is compiled with (12 for Pascal / Volta or a driver older than 580) |
| `--max-compatibility` | Windows: compile the release build (AVX2 baseline, static runtime), so it also runs on other PCs |
| `--data-dir PATH` | where the model files go |
| `--gguf-dir PATH` | use GGUF files you already have (setup's names, or another GGUF of the same architecture) |
| `--custom-gguf PATH` | install one GGUF; setup reads its header and picks the family and size |
| `--inspect PATH` | print what setup reads from a GGUF's header; no install |
| `--port N` | the server port (default 8080) |
| `--host H --api-key K` | listen beyond this PC; **only together with a key** |
| `--no-start` | install only, do not start the server |
| `--setup` | install another model or change the settings of an installed one |
| `--check` | only check the PC |

`--no-start` is recommended for agents: the server runs in the foreground until its window is closed, which would
block your shell. Start it separately in step 6.

Notes:
- **Linux:** setup uses `sudo apt` (or dnf/pacman) to install Python with venv if it is missing. You cannot type the
  user's password: if a `sudo` step is needed, ask the user to run it (e.g. `sudo apt install python3-venv`) and then
  rerun setup.
- **Build tools:** setup compiles strata-poly on the PC. Without a CUDA toolkit (or, on Windows, Visual Studio's C++
  build tools) it offers to install them; `--yes` accepts.
- **AMD:** on Linux setup installs ROCm into `.venv` (~10 GB, no sudo) unless a system ROCm 7 exists; on Windows it
  installs AMD's ROCm wheels into `.rocm-win` and compiles with them (Visual Studio's C++ build tools needed). The
  engine is compiled for the card either way (10-20 minutes, once). No images on AMD cards.

## 5. While it works, tell the user

- It compiles the engine for their card (10-20 minutes the first time), then downloads the model from Hugging Face:
  how long depends on their connection (54 GB at 100 Mbit/s is about 1.5 hours). It can be stopped and continues
  where it left off.
- Nothing leaves their PC: the model runs locally.

## 6. Start the server

Setup prints the start script it wrote (`start script: run-<model>.sh`). The name is the family's tag plus the size,
lower case: `run-laguna-s21-ud-q3_k_xl`, `run-laguna-xs21-q4_k_m`, `run-nemotron-bolt-24gb`, `run-gemma4-ud-q3_k_m`.

```
Windows (PowerShell):  Start-Process -FilePath ".\run-laguna-xs21-q4_k_m.bat"     (opens its own window)
Linux:                 nohup ./run-laguna-s21-ud-q3_k_xl.sh > strata-server.out 2>&1 &
```

Or `START-HERE.bat` / `./setup.sh` without flags, which starts the installed model (it asks which one when several
are installed; add `--yes` to take the first, the most recently used). The
server opens the user's browser on `http://127.0.0.1:8080` when it is ready. Closing its window (or stopping the
process) stops the model.

## 7. Verify

The HTTP server answers once the model is loaded. Poll:

```
curl http://127.0.0.1:8080/health
```

It returns `{"status": "ok", "model": ..., "max_context": ..., "images": ..., "api_key": ..., "loaded": true, ...}`.
Then:

```
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"strata\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in five words.\"}], \"max_tokens\": 64, \"reasoning_effort\": \"none\"}"
```

With an API key, add `-H "Authorization: Bearer <key>"`. The server window prints a `ready: http://127.0.0.1:8080/v1`
line when it is ready; the engine log is `strata-<model>.log` in the Puddle folder.

## 8. Connect the user's apps

- **Browser:** `http://127.0.0.1:8080` - Chat, a live Monitor, and About (settings and addresses).
- **Any OpenAI-compatible app or agent:** base URL `http://127.0.0.1:8080/v1`, any API key (or the configured one),
  any model name.
- **Anthropic-compatible apps:** `http://127.0.0.1:8080/v1/messages`.
- **Codex CLI and other Responses API apps:** `http://127.0.0.1:8080/v1/responses` (stateless; Codex's
  `config.toml`: [API.md](API.md#the-responses-api-and-codex-cli)).
- **Claude Code:** `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`, `ANTHROPIC_MODEL` set to a Claude model name it knows
  (Puddle ignores the name), and any `ANTHROPIC_AUTH_TOKEN` (or the configured key). Add `"fit_max_tokens": true` to
  `strata-<model>.json`: Claude Code asks for 32,000 tokens of answer in every request, and the server refuses a
  request whose prompt leaves less room than that in the context.
- **Thinking level:** `"reasoning_effort": "none" | "low" | "medium" | "high"`.
- Puddle answers one request at a time; `"parallel": N` in the model's config (or setup `--parallel N`) decodes up to
  N together ([BATCHING.md](BATCHING.md)). API details: [API.md](API.md).

## 9. When something fails

| Symptom | What to do |
| --- | --- |
| `the NVIDIA driver is too old` | The user updates the driver and restarts; rerun setup (or `--cuda 12` for a driver older than 580). |
| Download, compile or install stopped | Rerun the same setup command; it continues. |
| `port 8080 is already in use` | Puddle is already running (check `/health`), or use `--port 8081`. |
| Very slow, disk busy, or "the engine stopped unexpectedly" | Not enough free RAM: close programs, or set up a smaller size or model. |
| `prompt ... exceeds the context` | Rerun setup with `--setup --context <bigger>`. |
| Python or build tools could not be installed | Install what the message names (links are printed), then rerun. |

More: [TROUBLESHOOTING.md](TROUBLESHOOTING.md). If it still fails, collect `strata-<model>.log` and the setup output,
and suggest an issue at https://github.com/ErikOHiggins/Puddle/issues.

## Alternative: the MCP server

Puddle also has an MCP server, so an AI tool can check, install, start and stop Puddle through tool calls
(`strata_status`, `strata_models`, `strata_install`, `strata_start`, `strata_stop`, `strata_logs`) instead of the shell
commands above. See [MCP_SERVER.md](MCP_SERVER.md).
