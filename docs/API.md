# The server's API and settings

Puddle's server (`serve/server.py`) runs the engine, strata-poly, and speaks OpenAI's and Anthropic's APIs on
`http://127.0.0.1:8080`. This page lists every endpoint and every setting of the model's `strata-<model>.json`.
Installing: [INSTALL.md](INSTALL.md). Several requests at once: [BATCHING.md](BATCHING.md).

> **On this page:** [Endpoints](#endpoints) · [Monitoring](#monitoring) · [Thinking](#thinking) ·
> [Answers and streams](#answers-and-streams) · [Apps and agents](#apps-and-agents) · [Context](#context) ·
> [Models](#model-names-and-switching) · [Network and security](#network-and-security) ·
> [Conversation cache](#conversation-cache) · [Sampling](#sampling-and-limits) ·
> [Responses API](#the-responses-api-and-codex-cli) · [MCP tools](#tools-from-mcp-servers) · [Pictures](#pictures) ·
> [Sharing the GPU](#sharing-the-gpu-with-other-programs) · [Lazy start](#start-the-api-without-loading-the-model) ·
> [JSON formats](#json-response-formats) · [Request monitor](#api-request-monitor)

## Endpoints

| API | Endpoint |
| --- | --- |
| OpenAI Chat Completions (stream and non-stream, tools) | `POST /v1/chat/completions` |
| Anthropic Messages (stream and non-stream, tools) | `POST /v1/messages`, `POST /v1/messages/count_tokens` |
| OpenAI Responses (stream and non-stream, tools; stateless, [below](#the-responses-api-and-codex-cli)) | `POST /v1/responses` |
| Model list / health | `GET /v1/models`, `GET /models`, `GET /health` |
| Model properties | `GET /props` (also accepts `?model=<loaded-model-id>`) |
| What the model is doing right now | `GET /status`, `GET /slots` (busy or idle, with `n_prompt_tokens`; one entry per slot with `parallel`) |
| Everything the Monitor tab shows (engine, live state, last requests, hardware) | `GET /metrics` |
| The same for Prometheus, with vLLM's metric names (`Accept: text/plain` or `?format=prometheus`) | `GET /metrics` |
| Request totals and what is running, under llama-server's metric names | `GET /metrics/prometheus` |
| The MCP servers, their state and tools ([below](#tools-from-mcp-servers)) | `GET /mcp` |
| Load / unload the model ([below](#sharing-the-gpu-with-other-programs)) | `POST /load`, `POST /unload`, `POST /v1/load`, `POST /v1/unload` |
| Give VRAM back to other programs | `POST /v1/vram` |

`POST /slots/0?action=save|restore` (llama-server's session files) answers **501**: strata-poly has no session files.
`logit_bias` in a request answers **400**: strata-poly does not support it.

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "strata", "messages": [{"role": "user", "content": "Write a haiku about GPUs."}], "max_tokens": 512 }'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(model="strata", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.content)
```

`/models` and `/v1/models` list the loaded model, with its context limit and input modalities (with
`switch_on_request`, every installed model that can run, the loaded one first). `/props` has the original chat
template, the context limit, the configured generation defaults, the model path and the engine version. `n_predict: -1`
means no fixed output cap. `/slots` uses llama.cpp's names: `n_ctx` and `n_prompt_tokens` (the running request's
prompt size, kept after it ends: what a front-end's context meter divides by `n_ctx`). `total_slots` in `/props` is
the number of `parallel` slots (1 without). An unknown `model` returns 404. These endpoints require the API key when
one is configured, and do not load, unload or restart models.

`chat_template_caps` in `/props` uses llama.cpp's field names. The server checks them when it loads the template, by
rendering small requests; a feature the template rejects is `false`, and a template that fails a probe only turns that
feature off. `supports_tools`, `supports_tool_calls`, `supports_parallel_tool_calls`, `supports_system_role` and
`supports_preserve_reasoning` (older assistant turns keep their `reasoning_content` in the prompt) are hints for
clients, not a promise that the model follows a request. Zed's llama.cpp provider reads them to offer tools.

## Monitoring

`GET /metrics` answers a Prometheus scrape (`Accept: text/plain` or `application/openmetrics-text`) in the text format
with vLLM's names: `vllm:num_requests_running` / `_waiting`, `vllm:kv_cache_usage_perc`, the token and request counters,
`vllm:prefix_cache_queries_total` / `_hits_total` (prompt tokens read / reused), `vllm:spec_decode_num_draft_tokens_total`
/ `_accepted_tokens_total`, and the histograms `vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds` and
`vllm:e2e_request_latency_seconds`, so dashboards written for a vLLM server read this one. With an API key, the scraper
sends it as a bearer token. Any other request gets JSON.

The JSON's own facts come in the same scrape under `strata:`, named after their JSON key: `strata:live_state{state="..."}`,
`strata:live_tok_s`, `strata:live_prefill_tok_s_mean`, `strata:engine_max_context`, `strata:last_decode_tok_s`, and the
hardware (`strata:gpu_util`, `strata:gpu_mem_used_bytes`, `strata:gpu_temp_celsius`, `strata:gpu_power_watts` per card,
`strata:cpu`, `strata:ram_used_bytes`, ...). A value the server does not have has no sample rather than a zero.

Ready to use, in [monitoring/](monitoring/): `prometheus.yml` (a scrape config), `servicemonitor.yaml` (the same for the
Prometheus Operator) and `grafana-strata.json`, a Grafana dashboard. Its first row uses only vLLM's names.

## Thinking

The model thinks before it answers (streamed as `reasoning_content`; Anthropic: `thinking` blocks). Choose how much
per request, in the chat page's "Thinking" menu or over the API:

| API | how |
| --- | --- |
| OpenAI | `"reasoning_effort": "none" \| "low" \| "medium" \| "high"` (also `"reasoning": {"effort": ...}`, or `"chat_template_kwargs": {"enable_thinking": false}`) |
| Anthropic | `"output_config": {"effort": "low" \| "medium" \| "high"}`, `"thinking": {"type": "disabled"}`, or `"thinking": {"type": "enabled", "budget_tokens": N}` (under 2K = low, under 8K = medium, more = high) |

How a level reaches the model depends on its chat template. `"reasoning_effort": "low"` (or another level) in
`strata-<model>.json`, outside `"args"`, is the level for every request that names none; the web app's shared Chat
settings ("Use for other apps too") and a request's own level win over it.

- **A hard thinking budget.** `"reasoning_budget_tokens": N` in a request (OpenAI or Anthropic) caps the thinking at
  N tokens: there the server ends it with a short wrap-up line and the end of the thinking, and the model answers from
  there (nothing is read again). The wrap-up counts as output. The same key in `strata-<model>.json` sets it for every
  request; a request's own value wins, and `0` means no budget. Anthropic's `"thinking": {"budget_tokens": N}` is also
  the budget.
- **max_tokens counts the thinking, so the answer keeps its room.** A request with thinking on, no budget of its own
  and `max_tokens` of 1,024 or more gets one: the thinking is wrapped up when a quarter of the cap (at least 512
  tokens) is left, and the model answers in that room. Requests without `max_tokens` get a budget of three quarters
  of the context left. `"answer_room": false` in `strata-<model>.json` turns it off.
- **A reply that ends inside its thinking (opt-in).** `"reasoning_close_retry": true` closes the thinking once and
  continues, for a reply that ended in its thinking with no answer and no tool call.
- **Anthropic requests that don't ask for thinking (opt-in).** By default a `/v1/messages` request with no
  `"thinking"`, effort or budget thinks as the model's template does. `"anthropic_thinking": "on_request"` renders such
  a request without thinking, as Anthropic does; Claude Code's short helper calls (a session title with 512 tokens of
  room) need it.
- **Repeated reasoning (opt-in).** `"reasoning_loop_recovery": "stop"` measures the reasoning every 512 tokens and,
  when at least 25% of its last 2,000 words belong to 12-word passages seen three times (or its tail repeats with a
  period of at most 64 characters), ends the reply there as `"length"`. `"recover"` goes on once with a low-effort
  sentence in place of Qwen3.8's "xhigh" one. It was written for that template: where the prompt does not have the
  sentence it does nothing, and the reply goes on. Use `"stop"` with Puddle's models.
- **Thinking of earlier turns.** `"preserve_thinking": true` keeps the thinking of earlier assistant turns in the
  prompt, for templates that support it (Laguna, Occamy).

## Answers and streams

- **Streaming.** With `"stream": true` everything arrives as it is made: the thinking, the answer, and tool calls
  (the tool's name first, then its arguments piece by piece). While the model reads a long prompt the stream sends
  keep-alives, so agents do not time out; the server window prints progress every 15 s, and `GET /status` says what it
  is doing. Closing the connection or pressing stop really stops the model.
- **Prefill progress in the stream (opt-in per request).** `"return_progress": true` adds llama.cpp's
  `prompt_progress` (`total`, `cache`, `processed`, `time_ms`) to an empty chunk for each chunk of the prompt the engine
  reads, so a client that draws a prefill bar for llama.cpp draws one here.
- **Keep-alives as events (opt-in).** The keep-alive is an SSE comment, which an older openai-go takes as an empty
  event and fails on, so poolside's pool agent disconnects while a long prompt is read. `"keepalive_events": true` in
  `strata-<model>.json` sends the API's own event instead: on `/v1/chat/completions` a chunk with an empty delta, on
  `/v1/messages` Anthropic's `event: ping`. `/v1/responses` keeps the comment (Codex gets `response.in_progress`).
- **Stop strings.** OpenAI `stop` (a string or up to 4) and Anthropic `stop_sequences`: the answer ends before the first
  one, which is not sent (`finish_reason` "stop"; `stop_reason` "stop_sequence").
- **A reply stuck on one token is ended.** When a reply repeats the same token 256 times in a row, the server ends it
  with `finish_reason` `"length"` and says so. `"repeat_stop_tokens": N` sets the run length; `0` turns it off.
- **Tool calls in other forms (opt-in, the XML `<function=NAME>` format).** `"tool_call_recovery": true` also reads a call of a declared
  tool written as `<parameter=NAME>` instead of `<function=NAME>`, as JSON inside `<tool_call>`, as a `<function=NAME>`
  block without `<tool_call>`, or as a second call inside the same `<tool_call>`.
- **opencode's environment block (opt-in).** `"env_to_user": true` moves the `<env>` block opencode writes into its
  system message (working directory, date) to the first user message, so the system turn is the same for every session
  and its cached start is reused.
- **Reading an agent's prompt start when the server starts (opt-in).** `"warm_start": true` keeps the first turn of
  the last request with tools (system prompt and tools, when 2,048 tokens or more) in `strata-<model>.warm-start` and
  reads it when the server starts, so an agent's first request after a restart reuses it. A request that comes in
  during that read goes first.
- **Keep Windows awake while it works (opt-in).** `"prevent_sleep": true` keeps the PC from sleeping while a request is
  running or waiting. It does nothing on Linux and macOS.

## Apps and agents

- **Chat apps.** Any app with an "OpenAI-compatible" provider: base URL `http://127.0.0.1:8080/v1`, any API key. Zed's
  generic OpenAI-compatible provider needs `"capabilities": {"tools": true, "chat_completions": true}` on the model.
- **Claude Code:** `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`, `ANTHROPIC_MODEL` set to a Claude model name it knows
  (Puddle ignores the name), and any `ANTHROPIC_AUTH_TOKEN` (or your `api_key`). It asks for 32,000 tokens of answer in
  every request, which the server shortens to the room the context has left (`fit_max_tokens`, on by default); add
  `"anthropic_thinking": "on_request"` to `strata-<model>.json`.
- **Codex CLI:** [the Responses API](#the-responses-api-and-codex-cli) below.
- **OpenCode.** A starting point for `opencode.jsonc` (field names are OpenCode's):

  ```jsonc
  {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
      "strata": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Puddle (local)",
        "options": { "baseURL": "http://127.0.0.1:8080/v1", "apiKey": "none" },  // or your api_key
        "models": {
          "strata": {
            "name": "Laguna-S 2.1 (Puddle)",
            "limit": { "context": 131072, "output": 32768 },           // context: what you chose in setup
            "options": { "reasoningEffort": "high" }
          }
        }
      }
    },
    "model": "strata/strata"
  }
  ```

  Set `limit.context` to the context you chose in setup, and keep `limit.output` well under it (`fit_max_tokens`,
  on by default, shortens an answer that would not fit).

## Context

Chosen in setup. Requests longer than that are refused (400), never silently cut. A request whose `max_tokens` would
run past the context has its `max_tokens` shortened to the room left (`fit_max_tokens`, on by default since
2026-11-23, as in PolyStrata 0.2.19-poly.1); `"fit_max_tokens": false` (or `--no-fit-max-tokens` to
`serve/server.py`) refuses it with 400 instead, as llama.cpp does. A prompt that does not fit is refused unless:

- `"drop_oldest": true`: a conversation's oldest turns are left out when it no longer fits (the system prompt and the
  last turn stay).
- `"truncate_middle": true`: the middle of an oversized text prompt is trimmed, keeping its beginning and end.

A prompt that leaves no room at all is still refused.

## Model names and switching

- **Aliases.** `"aliases": ["laguna", "local-model"]` lists the model under those names too in `/v1/models`, like
  llama-server's `--alias`. Any other name is still served.
- **Switching models on request (opt-in).** `"switch_on_request": true` in the config the server starts with (or
  `--switch-on-request`) makes the API work like llama-server's router: `/v1/models` lists every installed model that
  can run (status `loaded`, `unloaded` or `loading`), and a request that names another one loads it: the loaded model is
  stopped first, then the named one starts with its own settings. The name must be a model's name, an alias or its
  config file's name, whole (`laguna-s-2.1`, not `laguna`); any other name is answered by the loaded model. Requests
  already running finish first. `POST /v1/load {"model": "<name>"}` switches the same way. A model that cannot run
  answers 400, one that does not start 503 (the earlier one is loaded again).
- **Switching in the web page.** With more than one model installed, the chat page's model list switches the server
  without a restart. Each model runs with its own `strata-<model>.json` and its own shared Chat settings. The server's
  own keys come from the config it started with: `host`, `port`, `api_key`, `allowed_hosts`, `cors_origins`,
  `trusted_origins`, `api_monitor`, `mcp_servers`, `prevent_sleep`, `env_to_user`, `open_browser`, `images` and
  `lazy_load`.
- **Model settings in the web page.** The About tab's Model settings card shows and changes the `sampling` defaults,
  `reasoning_budget_tokens`, `fit_max_tokens`, `drop_oldest`, `truncate_middle`, `preserve_thinking`,
  `anthropic_thinking`, `aliases`, `idle_unload_s`, `lazy_load`, `engine_silence_s`, `api_monitor`, `switch_on_request`,
  `open_browser` and `--vram-reserve-mib`. An empty field removes the key. The earlier file is kept as
  `strata-<model>.json.bak`, and the change applies from the next start. Only Puddle's own page can save.

## Network and security

- **From other devices on your network.** The server listens on your PC only (`127.0.0.1`) unless you say otherwise:
  `./setup.sh --setup --host 0.0.0.0 --api-key some-long-secret` (or `"host"` and `"api_key"` in
  `strata-<model>.json`). The server window prints this PC's addresses. On Windows, allow it through the firewall
  (private networks).
- **From the internet.** Put a tunnel in front of it (for example `cloudflared tunnel --url http://127.0.0.1:8080`).
  **Set a key first**, or anyone with the link can use your PC: `"api_key": "some-long-secret"` (or the
  `STRATA_API_KEY` environment variable). Several keys: `"api_key": "key1,key2"` or `["key1", "key2"]`. Streamed answers
  carry `X-Accel-Buffering: no` for nginx-style proxies. The web app's settings and MCP tools only answer Puddle's own
  page: add a proxy's address with `"trusted_origins": ["https://puddle.example.com"]`.
- **From web apps in a browser (CORS).** Off by default. `"cors_origins": ["https://chat.example.com"]` lets those
  pages call `/v1/*`; `["*"]` lets any page, which is only sensible with an API key. It never opens `/settings`,
  `/unload` or the MCP tools.
- **Host names (DNS rebinding).** Without an API key the server answers only requests whose `Host` it knows:
  `localhost`, any IP address, and, when it listens beyond this PC, this PC's name and `host.docker.internal`. Others
  get **403**. Add names with `"allowed_hosts": ["puddle.example.com"]` (or `STRATA_ALLOWED_HOSTS`); `".example.com"`
  allows that name and the names below it, `["*"]` turns the check off, and so does an API key.
- **Web pages without an API key.** A `POST` to `/v1/*` with an `Origin` header is answered only for Puddle's own page,
  `localhost` pages, allowed host names, `trusted_origins` / `cors_origins`, and browser extensions and desktop apps, and
  only with a JSON body; any other page gets **403**. Clients that send no `Origin` (curl, the SDKs) are not affected.

## Conversation cache

A request that continues a chat reads only the part after what the engine already holds: the live session, or one of
the checkpoints it keeps in RAM (`--prompt-cache N`, default 6, about 150 MB each: one every `--prompt-cache-every`
tokens, default 2,048, and one at a prompt's end). A checkpoint is used only when the prompt starts with exactly its tokens and pictures. Claude Code stamps its
system prompt with a header that changes on every request; the server pins it, so the system prompt and the tools in
front of it stay the same prompt on every turn.

**Several conversations (opt-in).** setup's "Several conversations in RAM?" question, or `--conversation-cache-mib 8192
--conversation-cache-slots 4` in the engine's `"args"`, parks up to four conversations in a bounded 8 GiB RAM cache, so
a coding agent's main conversation and its helpers do not read each other's prompts again when their requests
alternate. It does not run requests at the same time. Only exact token and picture prefixes are reused, and the
oldest parked conversation leaves first. `--conversation-cache-min-tokens N` parks only conversations of at least N
tokens (default 1,024), so short side requests do not push a long conversation out. `--conversation-cache-min-free-mib N` (default 2,560)
keeps that much RAM free: a park that would go under it is skipped. The Monitor tab's **Conversation cache** card and `/metrics`
show the parked conversations and how many prompt tokens the cache gave back.

## Sampling and limits

One request is answered at a time unless `"parallel": N` is set (up to N decoded together: [BATCHING.md](BATCHING.md)).

**Temperature / top_p / top_k / min_p / seed** are honored per request (OpenAI and Anthropic fields). The config's
`sampling` block sets the defaults for requests that leave them out (`"sampling": {"temperature": 1.0, "top_p": 0.95,
"top_k": 20}`); the web page's shared Chat settings win over it, and a request's own fields win over both.
Setup writes each family's sampling as the block: Laguna-S and Laguna-XS temperature 1.0, top_p 1.0, top_k 20,
min_p 0.1 (what testing on echo settled on; poolside's own config has min_p 0); Occamy 1.0 / 0.95 / 20 with
presence_penalty 1.5 (the config it ran with on echo). A block you write yourself is kept when setup
runs again. A config without a block (one made by hand, or before setup wrote them) answers a request with no sampling
greedy (strata-poly's default, temperature 0). `top_k` takes at most 64.
The penalties (`presence_penalty`, `frequency_penalty`, `repetition_penalty`, with `penalty_last_n`, default 64;
llama.cpp's `repeat_penalty` and `repeat_last_n` work too) count the tokens the request has consumed.

## The Responses API and Codex CLI

`POST /v1/responses` speaks OpenAI's Responses API, which Codex CLI uses. It runs on the same path as
`/v1/chat/completions`, so the thinking levels, the budget, the conversation cache and the security checks apply.

It is **stateless**: nothing is stored, so the client sends the whole conversation in `input` every time (Codex does,
with `store: false`). `previous_response_id`, `conversation`, `background` and the retrieve/delete/cancel endpoints are
refused with an error that says so.

| Request | What Puddle does |
| --- | --- |
| `input` as a string, or as items | `message` items (`user`, `assistant`, `system`, `developer`; text and images), `reasoning`, `function_call`, `function_call_output`, `custom_tool_call(_output)` |
| `instructions` | The system message |
| `tools` | `function` tools, `namespace` tools (the model sees `namespace.name`), `custom` tools (one free-form `input` string). Hosted tools (`web_search`, ...) are left out |
| `tool_choice` | `"none"` hides the tools; anything else lets the model choose |
| `reasoning.effort` | `none`/`minimal`, `low`, `medium`, `high`/`xhigh`; without it the config's `reasoning_effort` |
| `max_output_tokens` | The output cap (thinking included). Running out ends the response `incomplete` |
| `text.format` | `json_schema` and `json_object` use the [JSON response formats](#json-response-formats) |
| Codex's thread-title request | Opt-in, `"codex_thread_titles": true`: answered without the engine with the user's line cut to 36 characters, so the conversation keeps its cache |

The thinking comes back as a `reasoning` output item with `reasoning_text` content. With
`"include": ["reasoning.encrypted_content"]` the item also carries `encrypted_content` (base64, not encrypted). Send the
reasoning items back with the conversation, as Codex does, so the cache is reused. With `"stream": true` the events
are the official ones, in order, and a `response.in_progress` event goes out every 15 s while a long prompt is read.

**Codex CLI.** In `~/.codex/config.toml` (Windows: `%USERPROFILE%\.codex\config.toml`):

```toml
model = "strata"                        # any name; Puddle answers with its model
model_provider = "strata"
model_context_window = 131072           # the context you chose in setup: Codex compacts before it gets there
show_raw_agent_reasoning = true         # show the model's thinking

[model_providers.strata]
name = "Puddle (local)"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
stream_idle_timeout_ms = 600000         # a first, long prompt can take minutes to read
# env_key = "STRATA_API_KEY"            # only if the server has an api_key: the variable holding it
```

Codex warns `Model metadata for ... not found` for a local model name; that is expected.

**Codex's compaction (opt-in: `"codex_compaction_cache": true`).** When the context fills up, Codex sends the
conversation with a request to summarize it and with `tools: []`, so the prompt would differ from the cached one right
after the start. With the option on, a compaction request from the same Codex conversation is rendered with the tools
of its last prompt, and the conversation's cache is reused.

## Tools from MCP servers

The chat page can give the model tools from [MCP](https://modelcontextprotocol.io) servers, as LM Studio and Claude
Desktop do. List the servers in `strata-<model>.json` under `"mcp_servers"` (the shape of Claude Desktop's `mcpServers`
block, which you can paste as it is):

```json
"mcp_servers": {
  "files": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/me/notes"]},
  "search": {"url": "http://127.0.0.1:3000/mcp", "headers": {"Authorization": "Bearer ..."}}
},
"mcp": {"timeout_s": 60, "max_result_chars": 20000, "max_rounds": 8}
```

Or start the server with `--mcp-config path/to/claude_desktop_config.json`. Restart Puddle after a change.

- **A program** (`command`, `args`, optional `env` and `cwd`) is started by Puddle and spoken to over stdin/stdout.
  **An address** (`url`, optional `headers`) uses MCP's Streamable HTTP transport. `"disabled": true` leaves one out.
- The servers start with Puddle, in the background; the server window says what each one offers, or why it did not
  start. The Monitor tab lists them, and the Sampling drawer has **Use tools from MCP servers**.
- In the chat each call shows as a small block; the model reads the result and goes on, up to `max_rounds` calls per
  answer. A tool that fails or takes longer than `timeout_s` gives the model an `error: ...` result.
- Only the chat page uses them. API clients keep their own tools; a request to `/v1/chat/completions` opts in with
  `"strata_mcp": true`.

**Security.** MCP tools run with your user's rights, and **the model decides when to call them**, also because of what
it reads (a web page or a file can contain instructions). Give a filesystem server only the folders it needs, and set
an API key if Puddle is reachable from other devices.

The other direction, your AI assistant managing Puddle, is [MCP_SERVER.md](MCP_SERVER.md).

## Pictures

Occamy reads pictures with its mmproj file and a small helper, `strata-vision` (from llama.cpp's `mtmd`). Say yes to
"Images?" in setup, or run `./setup.sh --setup --vision yes`. The encoder runs on the GPU from 15.5 GB of VRAM (it
keeps about 1.6 GB of VRAM, by setup's figures), else on the CPU (a few seconds
a picture). No pictures on AMD cards for now.

The `"vision"` section of `strata-<model>.json`:

- `"max_tokens"`: the most tokens a picture becomes (setup's `--vision-tokens N`). More keeps more detail and takes
  longer to encode.
- `"min_tokens"`: a small picture is scaled up to at least this many tokens.
- `"mmproj"`: another mmproj file of the same model (a Q8_0 one is smaller and uses less RAM on the CPU).
- `"cuda_device"`: run the encoder on a card the engine doesn't use (numbered like `nvidia-smi`).
- `"elastic"` (the default on the CPU): the encoder starts with the first picture and stops after `"elastic_idle_s"`
  (default 120) without one.

**Sending a picture.** OpenAI: an `image_url` part (a `data:` URL, an `http(s)://` URL or a local file path):

```python
import base64
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
img = base64.b64encode(open("photo.jpg", "rb").read()).decode()
r = client.chat.completions.create(model="strata", messages=[{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
    {"type": "text", "text": "What is in this picture?"}]}])
print(r.choices[0].message.content)
```

Anthropic: an `image` block with a `base64` (or `url`) source. JPEG, PNG, BMP, GIF, WebP, TIFF and AVIF work. The same
picture sent again, as chat apps do every turn, is encoded once.

## Sharing the GPU with other programs

By default the model stays loaded until you close Puddle. On a PC that also games, renders or runs another model
server, these options (off by default; also as keys in `strata-<model>.json`) give the VRAM back:

| Option | Config key | What it does |
| --- | --- | --- |
| `--idle-unload 600` | `"idle_unload_s": 600` | unload the model after 600 s without requests; the next request loads it again |
| `--min-free-vram-mib 11000` | `"min_free_vram_mib": 11000` | load an unloaded model only when that much VRAM is free (it waits up to 15 s), else answer **503** "the GPU is in use by another program" |
| `--before-load "cmd"` | `"before_load": "cmd"` or `["cmd", "arg"]` | a command run before the model is loaded again, e.g. one that unloads another server's model |
| | `"before_load_timeout_s": 120` | how long that command may run. When it fails, the free VRAM decides; `"before_load_strict": true` always refuses, `"before_load_ignore_failure": true` always loads |

`POST /unload` unloads it now (`409` while a request is running; `?wait=1` waits for the running turn first, up to
300 s) and `POST /load` loads it ahead of a request (both with `Content-Type: application/json`). `/health` says
`"loaded"`, `/v1/models` lists it as `unloaded`, and the Monitor shows the state. Unloading ends the engine process,
and the image encoder, so their VRAM and RAM go straight back; the model files stay in the OS file cache, so loading
again is quicker while that RAM is not needed elsewhere.

**Giving part of the VRAM back while it keeps serving.** `POST /v1/vram` with `{"reserve_mib": 8000}` makes strata-poly
empty its expert cache between requests and fill it again with the experts used most so far, leaving that much VRAM
free; the experts it gives up are computed on the CPU, so answers keep coming, slower. `{"reserve_mib": null}` is the
reserve the engine started with. Measured upstream with Gemma 4 on a laptop's 8 GB card: `{"reserve_mib": 2500}` left
2.0 GiB free and the next answer came at 55 tokens/s where the one before came at 66; `{"reserve_mib": null}` put 592
of the 635 experts back. strata-poly also looks before each request on its own: from the second on it keeps half its
reserve free, giving up its least used experts when another program takes VRAM.

## Start the API without loading the model

`"lazy_load": true` in the config (or `serve/server.py ... --lazy`) starts the HTTP API without the engine or the image
encoder. The first generation request, or `POST /load`, loads them, including `before_load` and `min_free_vram_mib`.

`POST /v1/load` and `/v1/unload` are JSON aliases for integrations, taking `{}` or `{"model": "<configured model>"}`.
They require the API key, `application/json`, and no foreign browser Origin; they return **409** while a request is
active or queued and **404** for an unknown model. `/api/health` aliases `/health`; `/v1/status` has `loaded` and
`auto_load`.

## JSON response formats

`POST /v1/chat/completions` accepts `response_format: {"type": "json_object"}` or
`{"type": "json_schema", "json_schema": {"name": "answer", "strict": true, "schema": {...}}}`. The schema's root must
accept only objects. `json_schema` is checked with the Python package `jsonschema` when it is installed (setup does not
add it); without it the answer is only checked to be one JSON object.

This is **schema prompting followed by validation**, not grammar-constrained decoding, with no hidden retry. A valid
answer is returned; malformed JSON, schema violations and incomplete answers return **502** with
`error.code: structured_output_failed`; invalid request schemas return **400**. JSON formats with tools are refused.
A streamed structured answer is held until it is checked, with keep-alives meanwhile.

## API request monitor

Off by default, since it keeps prompts and answers in memory: `"api_monitor": true` (or `--api-monitor`). Then
`/api-monitor`, or the web app's Requests tab, shows the model state, active and queued requests, request bodies,
output and reasoning, and the timings. `GET /api/requests` returns summaries, `GET /api/requests?id=<id>` one request.
It keeps the newest 100 requests until a restart, up to 262,144 characters per field. Treat that history as sensitive
on a network: set an API key.

## Other server settings

- `"engine": "llama-server"`: the model runs on llama.cpp's llama-server, and `"args"` are llama-server's own
  (setup wrote it for Nemotron 3.5 Bolt on llama-server until 2026-11-23, and writes it for no model now; without the
  key the engine is strata-poly). The args must not set `--host`, `--port` or `--api-key`: the server runs llama-server
  on 127.0.0.1
  with a port and a key of its own ([NEMOTRON.md](NEMOTRON.md#llama-servers-run-config)).
- `"engine_silence_s": 300`: end a request when the engine says nothing for this long (0 = wait forever); the next
  request starts the engine again.
- `"open_browser": false` (setup's `--no-browser`): do not open the chat page when the model is ready.
- `"parallel": N`: decode up to N requests together ([BATCHING.md](BATCHING.md)).
