# Puddle's MCP server: let your AI assistant install and run Puddle

`tools/strata_mcp.py` is an [MCP](https://modelcontextprotocol.io) server for the AI assistant you already use
(Claude Code, Claude Desktop, Cursor, VS Code, Codex ...). Add it once, then ask in plain words:

- "Install Puddle for this PC."
- "Start Puddle." / "Stop Puddle."
- "Is Puddle running? How much VRAM does it use?"
- "Which Puddle model fits my PC?"
- "Puddle didn't start - what does the log say?"
- "How fast is Puddle on this PC?" (a short speed test)
- "How do I connect Open WebUI / the OpenAI SDK / Claude Code to Puddle?"

The assistant calls the server's tools. The tools run Puddle's own setup and start the model the way the start
scripts do. It is one Python file that uses only the standard library, so it works before setup has created
`.venv`. It needs Python 3.10 or newer, 64-bit on Windows.

It is not the same thing as [Tools from MCP servers](API.md#tools-from-mcp-servers). That feature
(`serve/mcp.py`) lets the *Puddle model* call tools from your MCP servers in its chat page. This page is the other
direction: *your assistant* manages Puddle.

## Add it

You need a copy of Puddle: the zip from GitHub, or `git clone https://github.com/ErikOHiggins/Puddle`. Below,
`C:\Users\you\Puddle` (Windows) and `/home/you/Puddle` (Linux) stand for that folder. Use `python3` on Linux. On
Windows, use `py` if `python` is not on your PATH.

### Claude Code

```bash
claude mcp add strata -- python C:\Users\you\Puddle\tools\strata_mcp.py         # Windows
claude mcp add strata -- python3 /home/you/Puddle/tools/strata_mcp.py           # Linux
```

Add `--scope user` to have it in every project. `/mcp` in Claude Code shows that it is connected.

### Claude Desktop

Open Settings > Developer > Edit Config (`claude_desktop_config.json`) and add:

```json
{
  "mcpServers": {
    "strata": {
      "command": "python",
      "args": ["C:\\Users\\you\\Puddle\\tools\\strata_mcp.py"]
    }
  }
}
```

On Linux, use `"command": "python3"` and `"args": ["/home/you/Puddle/tools/strata_mcp.py"]`. Restart Claude
Desktop after the change.

### Cursor

Use `~/.cursor/mcp.json` for every project, or `.cursor/mcp.json` in one project. The format is the same as
Claude Desktop's:

```json
{
  "mcpServers": {
    "strata": {
      "command": "python",
      "args": ["C:\\Users\\you\\Puddle\\tools\\strata_mcp.py"]
    }
  }
}
```

### VS Code (GitHub Copilot agent mode)

Use `.vscode/mcp.json` in a workspace, or run "MCP: Open User Configuration" from the command palette:

```json
{
  "servers": {
    "strata": {
      "type": "stdio",
      "command": "python",
      "args": ["C:\\Users\\you\\Puddle\\tools\\strata_mcp.py"]
    }
  }
}
```

On Linux, use `"command": "python3"` and `"args": ["/home/you/Puddle/tools/strata_mcp.py"]`.

### Codex CLI

In `~/.codex/config.toml`:

```toml
[mcp_servers.strata]
command = "python"
args = ['C:\Users\you\Puddle\tools\strata_mcp.py']    # Linux: command = "python3", args = ["/home/you/Puddle/tools/strata_mcp.py"]
tool_timeout_sec = 180                                # strata_stop / strata_benchmark can take longer than 60 s
```

### Any other MCP client

The command is `python <Puddle folder>/tools/strata_mcp.py`. The transport is stdio. It speaks MCP revision
2025-06-18, and also 2025-03-26 and 2024-11-05.

`--root <folder>` manages a Puddle folder other than the one the file is in. For example:
`python D:\tools\strata_mcp.py --root D:\Puddle`.

## The tools

| Tool | What it does | Arguments (all optional) |
| --- | --- | --- |
| `strata_status` | Whether a model is running: the model, engine version, context, busy or idle, VRAM and RAM in use. What is installed: models, start scripts, engine version, data folder, downloads. An install in progress. This PC's hardware: OS, CPU, RAM, GPUs, page file, free disk. The model and context setup recommends for this PC. | `port`, `hardware` (true/false) |
| `strata_models` | Every family and size setup offers, with download size, RAM needed, whether it fits this PC, and whether it is installed. Also the context lengths and the recommendation. | - |
| `strata_install` | Runs setup without questions, in the background (see below). Without `confirm: true` it only returns the plan. A second call shows the progress. `cancel: true` stops the install. | `family` (laguna-s, laguna-xs, occamy), `model` (a size of it, see `strata_models`), `context` (8192, 32768, 65536, 131072), `vision` (no, yes, gpu, cpu), `gpu` (a number) or `gpus` ("0,2", "all"), `backend` (auto, cuda, hip), `port`, `data_dir`, `confirm`, `cancel` |
| `strata_start` | Starts an installed model in the background and waits until it answers. If it is not ready after `wait_seconds` (default 50), it returns "loading". | `model` (an id from `strata_status`, e.g. `laguna-s21-ud-q3_k_xl`, `occamy10-apex-quality`; default: the most recently used), `port`, `gpu`, `wait_seconds` (0-600) |
| `strata_stop` | Unloads the model through the server's own API, then ends the server it started. | `port`, `force` |
| `strata_logs` | The last lines of the server log, the engine log or the setup log. | `source` (server, engine, setup), `model`, `lines` (1-500) |
| `strata_benchmark` | Sends a short fixed request (greedy, thinking off) and reports output and prompt tokens/s. | `port`, `max_tokens` (16-512) |
| `strata_connect_info` | The OpenAI and Anthropic base URLs, the model id, whether a key is needed, and settings for Claude Code, Codex, the OpenAI SDK, curl, Cursor, Continue and Open WebUI. | `port` |

Every result is JSON with a one-line `summary`. It is sent both as text and as MCP structured content.

## How it works

**Install.** `strata_install` uses setup's own logic and tables (it imports `setup.py`'s `FAMILIES`,
`POLY_CONTEXTS` and its RAM estimates) to fill in what you left out:

- the model and size setup itself would pick: the first family of its menu (Laguna-S) whose preselected size fits
  this PC by setup's estimate for the card, and none when no size fits (the answer then names the one that wants the
  least RAM);
- the context setup preselects (32K);
- images off.

The first call returns the plan: the download size (11-96 GB), the free disk space, the RAM check, and the exact
setup command. The assistant should show you the plan and call again with `confirm: true` once you agree.

It then runs the same steps `START-HERE.bat` / `setup.sh` run:

1. It creates `.venv` with the Python it runs on, if `.venv` is missing.
2. It runs `setup.py --yes --no-start --family ... --model ... --context ... --vision ...` as a background process.
   The output goes to `.strata-mcp/setup.log` in the Puddle folder.

The tool returns at once. Call it again to see the step (1-7), the download percentage, or setup's error message.
Everything setup has finished is skipped on the next run, and downloads resume. If the AI app closes, the install
keeps running. The install is refused while a model this server started is running, because setup may replace the
engine files that model uses.

**Start.** `strata_start` runs the same command as the `run-<model>` script, `serve/server.py --engine strata
--config strata-<model>.json --port N`, with the venv's Python, without `--open` (no browser) and without a console
window. The server's output goes to `.strata-mcp/server.log`. The tool then polls `GET /health` until the model
answers. Loading takes 1-3 minutes. If the server ends while loading, the tool returns the end of its log. The model
keeps running after the AI app closes, until `strata_stop`. If the AI app runs its tools inside a Windows job that
does not allow processes to leave it, the model ends together with the app.

Two cases are refused:

- a model is already running: the tool returns it and starts nothing;
- another program uses the port: the tool says so.

**Stop.** `strata_stop` first sends `POST /unload`. The server sends the engine `QUIT`, and the engine frees its
VRAM and pinned RAM itself, as when you close the window. Then the server ends:

- Linux: SIGTERM, which runs the server's own Ctrl+C path.
- Windows: the process ends. Its job object ends the vision encoder and the MCP servers it started.

While a request is running, the tool refuses unless you pass `force: true`.

**Status.** The tool reads `GET /health`, `/v1/status`, `/status` and, on older servers, `/metrics` on 127.0.0.1.
If a config of the folder sets an `api_key`, the tool sends that key with its own requests. It never shows the key.

## Safety

- **No shell, no free paths.** Every argument is checked against a fixed schema and against setup's own choices. An
  unknown argument, a model or family setup does not offer, or a `gpus` value other than digits and commas is
  refused before anything runs. Commands are passed as argument lists; nothing goes through a shell.
- **Paths.** The only path argument is `data_dir`. It must be one of:
  - the current Puddle data folder;
  - a folder inside the Puddle folder;
  - a new absolute folder whose name starts with `Puddle` (e.g. `D:\Puddle-data`), whose parent exists, and which is
    not in a system folder.

  `..`, relative paths and network (UNC) paths are refused. Logs are read only from `.strata-mcp/` and from the
  engine log a config names, if that log is inside the Puddle folder or the data folder.
- **Only its own processes.** The server records the process id *and start time* of every process it starts, in
  `.strata-mcp/*.json`. It only ends a process when both match, so a process id the OS has given to another program
  is never touched. A Puddle started some other way (its window, a run script, a service) is only asked over its
  API to unload the model. To end that server, close its window.
- **Big downloads need a yes.** `strata_install` does nothing until it is called with `confirm: true`. The plan
  shows the download size and the free space first, and the install is refused if the disk is too small.
- **Local only.** The server checks `127.0.0.1` and never changes the network settings: it does not install with
  `--host 0.0.0.0` or an API key. To share Puddle on your network, run setup yourself (see
  [Network and security](API.md#network-and-security)). Run setup by hand for anything the install tool does not
  take.

## Troubleshooting

- **The assistant does not see the tools.** Run the command by hand:
  `python C:\Users\you\Puddle\tools\strata_mcp.py`. It should print `[strata-mcp] ready (Puddle folder ...)` to
  stderr and wait for input (Ctrl+C ends it). If it reports `not a Puddle folder`, the path is wrong or `--root` is
  needed. Python older than 3.10 is refused.
- **The install failed.** Ask the assistant for the setup log (`strata_logs` with `source: setup`), or open
  `.strata-mcp/setup.log`. Setup's `[X]` line says what to fix. Then install again: every finished step is skipped.
- **The model does not start.** Get the server log with `strata_logs`, and the engine log with `source: engine`.
  [TROUBLESHOOTING.md](TROUBLESHOOTING.md) has the usual causes.
- **Tests.** `python -m unittest tools/test_strata_mcp.py` runs the tests with a fake setup and server. They do not
  download anything or use the GPU.
