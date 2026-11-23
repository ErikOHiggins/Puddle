# Nemotron 3.5 Bolt and Super

NVIDIA's [Nemotron 3.5 Bolt 74B-A9B 24GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF)
reads text and pictures. It is under the OpenMDW-1.1 license. Setup runs it on Puddle's own engine, strata-poly
(`src/poly/nemotron_h.cpp`), as it runs Nemotron 3.5 Super ([below](#nemotron-35-super-on-strata-poly)): what was
measured is [below](#nemotron-35-bolt-on-strata-poly). llama.cpp's `llama-server`, which ran Bolt before strata-poly
could, is an opt-in beside it (`--family nemotron-bolt-llama`), which setup compiles on your PC. Puddle's own server
stays in front of either: the chat template, the tool calls, the thinking, every endpoint and setting are the same as
for the other models.

**Status: experimental.** It has run on one machine, echo: on strata-poly, with llama-server alone, and through Puddle
on both ([llama-server](#measured-on-llama-server), [strata-poly](#nemotron-35-bolt-on-strata-poly)).

## The model

What its files say (their headers, read with `tools/gguf_reader.py`):

| | |
| --- | --- |
| Architecture | `nemotron_h_moe`: Mamba-2 layers, attention layers and mixture-of-experts layers, 88 blocks |
| Experts | 496 routed experts, 9 used for each token |
| The file | NVIDIA's own, 22,018,326,528 bytes: the experts in IQ2_XXS (2-bit, trained for it), the Mamba layers in NVFP4, the embedding and output in Q8_0 |
| The image encoder | C-RADIO v4-H, 1,733,692,736 bytes, in BF16 and F32 (not quantized) |
| Context | trained to 262,144 tokens |
| Tokenizer | 131,072 tokens, Mistral's Tekken pre-tokenizer (`pixtral` in the GGUF) |

Its tokenizer was checked twice on 2026-11-20. Puddle's tokenizer gives the same token ids as Hugging Face's
`tokenizers` 0.23.3, with the model's own `tokenizer.json`, on 239 strings. It also gives the same ids as llama-server's
`/tokenize` on echo, on 242 strings ([`tools/tokenizer_check.py`](../tools/tokenizer_check.py)).

## What you need

- **An NVIDIA card with 16 GB of VRAM or more.** On strata-poly the experts that do not fit the cards stay in RAM:
  setup estimates about 17 GB of RAM beside one 16 GB card; on two 16 GB cards every expert fits (measured, below).
  Unsloth's UD-Q5_K_XL (`--model UD-Q5_K_XL`, [below](#unsloths-ud-q5_k_xl)) has three times the bits in its experts:
  about 54 GB of RAM beside one 16 GB card and 44 GB beside two by setup's estimate. Its UD-Q4_K_XL
  ([below](#unsloths-ud-q4_k_xl)) wants about 46 GB beside one and 35 GB beside two.
- **On llama-server (`nemotron-bolt-llama`): NVIDIA cards with 30 GB of VRAM or more together**, one 32 GB card or two
  16 GB ones. Setup refuses less. It keeps every layer on the GPUs. Setup's RAM estimate there is 16 GB: the file goes
  to the GPUs, and llama-server keeps up to 8 GiB of earlier prompts in RAM (its prompt cache). Not measured.
- **Not on AMD cards or a Mac yet.** It has not been run on an AMD card; the Mac build has no Mamba-2 kernels; and
  setup compiles llama-server for NVIDIA only.
- **Windows:** setup compiles both engines there (MSVC and CUDA), and says that the model is untested. NVIDIA lists
  Windows as supported; nobody here has run it.
- **Disk:** 22 GB for the model (50 GB for UD-Q4_K_XL, 59 GB for UD-Q5_K_XL), 1.7 GB more for pictures, and a few
  GB for the build.

## Install

```bash
./setup.sh --family nemotron-bolt
```

Setup then:

1. compiles strata-poly, as for every model, and with pictures its image encoder (strata-vision).
2. downloads the model and its image encoder from NVIDIA's repository at a pinned revision, and checks their SHA-256
   (UD-Q4_K_XL and UD-Q5_K_XL: from Unsloth's repository, each into a folder of its own such as
   `nemotron-bolt-UD-Q5_K_XL/`, with NVIDIA's image encoder, the one in `nemotron-bolt-24GB/` when it is there).
   It keeps them in `Puddle-data/models/nemotron-bolt-24GB/`. Files you already have: move them there, and setup checks
   them instead of downloading them. Or give their folder with `--gguf-dir` to use them where they are.
3. writes the model's tokenizer and chat template from the GGUF to `Puddle-data/packs/nemotron-bolt-24gb/`.
4. writes `strata-nemotron-bolt-24gb.json` and `run-nemotron-bolt-24gb.sh` (`.bat`), and starts the model. Clients ask
   for it as `nemotron-3.5-bolt-24gb`.

It asks whether the model should read pictures, and the default is yes; `--vision no` leaves them out, and the image
encoder is not downloaded.

On llama-server instead, beside it:

```bash
./setup.sh --family nemotron-bolt-llama
```

This compiles llama-server from llama.cpp at commit `08c252e22` (the one echo ran it with) into `engine-llama/`, for
your cards: llama.cpp's CUDA kernels, once (about 10 minutes on echo's i9-9900K, by the owner's watch). It builds
without llama.cpp's own chat app, so nothing is downloaded during the build, and compiles again only when that commit
changes or a card needs code it does not have. It uses the same files from the same folder (downloaded once), and
writes `strata-nemotron-llama-bolt-24gb.json`: clients ask for it as `nemotron-3.5-bolt-llama-24gb`, so the two run
side by side and a request switches between them by name. Its default for pictures is yes.

A Bolt set up before 2026-11-21 ran on llama-server under the plain name: `./setup.sh --setup --family nemotron-bolt`
writes strata-poly's config over it, keeps the keys you added, and keeps the old file as
`strata-nemotron-bolt-24gb.json.bak`.

## llama-server's run config

`strata-nemotron-llama-bolt-24gb.json` has `"engine": "llama-server"`, and its `"args"` are llama-server's own, the
command echo ran:

```
--model <the file> --mmproj <the image encoder> --ctx-size 168000 --parallel 1 --flash-attn on
--cache-type-k q8_0 --cache-type-v q8_0 --batch-size 2048 --ubatch-size 2048 --load-mode none --n-gpu-layers all
--tensor-split 1,1
```

- **The context:** `--ctx-size 168000` on two 16 GB cards, as on echo. On other cards setup writes none, and
  llama-server's `--fit` takes the largest context that fits each time it loads. `./setup.sh --setup --family
  nemotron-bolt-llama --context N` sets one.
- **Several cards:** `--tensor-split` splits the model evenly across them (`1,1` is the same split as `30,30`). The
  config's `"gpu"` says which cards, in order, as for the other models.
- **`--load-mode none`** reads the file into memory instead of mapping it (llama.cpp's default is to map it).
- **Sampling:** setup writes `"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 40, "min_p": 0.05}`. This is
  NVIDIA's recipe as llama.cpp runs it: the model card's temperature and top_p, plus the top_k and min_p that
  NVIDIA's llama-server command leaves at llama.cpp's defaults. Puddle sends every sampling value to llama-server, so
  llama-server's own defaults never apply. Neither do the ones the GGUF carries (`general.sampling.temp` 1.0 and
  `top_p` 0.95).

The args must not set `--host`, `--port`, `--api-key` or `--api-key-file`. Puddle starts llama-server on `127.0.0.1`
at a port it picks each time, with a key it makes each time and passes in llama-server's environment, so only Puddle
can reach it. Your apps reach Puddle, at Puddle's address and with Puddle's key. A `"vision"` section, `"replicas"`
and `"parallel"` are refused: llama-server reads pictures with its `--mmproj`, and Puddle sends it one request at a
time.

llama-server's own output goes to `strata-nemotron-llama-bolt-24gb.log`, where the other models' engine logs go.

## Measured on llama-server

On echo (two RTX 5060 Ti 16 GB, Linux), 2026-11-20, with llama-server alone: llama.cpp `08c252e22`, these args with
`--tensor-split 30,30`, one request with pictures:

| | |
| --- | --- |
| Read the prompt (11,331 tokens) | 1,401.85 tokens/s |
| Wrote 589 tokens | 57.61 tokens/s |
| VRAM in use | 95% and 93% of the two cards |
| Context | 168,192 tokens (llama-server's size for `--ctx-size 168000`) |

Through Puddle on the same machine, the same day, its first two requests: a short text question (answered, with its
thinking apart), and llama.cpp's own test picture, a 640 x 488 scan of the New York Times of July 21, 1969: it named
the paper, the date and both headlines correctly, and wrote its 369 tokens at 61.2 tokens/s.

A tool call through Puddle (2026-11-20): asked for the weather with a `get_weather` tool, it called it with
`{"city": "Paris"}` (`finish_reason` `tool_calls`). Given the tool's answer in a second request, it answered from it.
That second request read only its 60 new tokens: llama-server reused 276 of the 336 (`cache_n`), up to its checkpoint
4 tokens before the first prompt's end.

Not measured yet: a long conversation through Puddle, any other card, and Windows.

`tools/llama_server_check.py --config strata-nemotron-llama-bolt-24gb.json --image <picture>` asks the same question,
greedy, through Puddle and then of llama-server alone (the model must not be running). It compares the prompt-token
counts, the prompts as llama-server read them, and the answers. It also replays Puddle's own request on
llama-server alone's `/completion`.

On echo, 2026-11-20, with llama.cpp's test picture, run three times:

- the prompts were the same: 327 tokens on both paths (Puddle's count of the picture, 302 tokens, was llama-server's),
  and the same text;
- Puddle's request on llama-server alone wrote Puddle's thinking and answer exactly;
- llama-server's own chat endpoint wrote the same first 401 characters of thinking, then took another word at a near
  tie ("other text" where Puddle's has "other elements"), and its answer followed from there. It reads the same
  prompt in other batches: it ends a batch where a user message starts, for its checkpoints of the Mamba state
  (`message_spans` in llama.cpp's `server-context.cpp`), and `/completion` has no such breaks. The rounding of the
  two parts a little, the same way on every run. Neither answer is the wrong one.

## Pictures

Puddle reads a picture as it does for the other models: a `data:` URL, an `http(s)://` URL or a local file. It turns
WebP, TIFF and the like into PNG, and turns an EXIF-rotated phone photo upright. Bolt's chat template writes `<image>`
where the picture goes. On strata-poly, Puddle's image encoder (strata-vision, built from llama.cpp's image code
`mtmd`) turns the picture into rows, and the server puts `<img>` and `</img>` around them, as `mtmd` does. On
llama-server, Puddle sends the prompt as text, with llama-server's marker in that place and the picture's file beside
it, and llama.cpp's `mtmd` puts `<img>` and `</img>` around the picture itself.

By llama.cpp's code at `08c252e22`, a picture is resized to at most 3,407,872 pixels and at least 262,144 (the numbers
in the image encoder's file), each side a multiple of 32. It then becomes one token per 32 x 32 pixels, plus the two
around it. A 1920 x 1080 screenshot is read at 1920 x 1088, 2,042 tokens; a 2560 x 1440 one at 2432 x 1376, 3,270
tokens. On llama-server, Puddle plans the request with that count; llama-server counts the prompt itself, and the
server window says when the two differ. That has not been seen on echo yet.

On llama-server, a message that quotes a special token's text (`<|im_end|>`, `<think>` ...) next to a picture gets a word joiner
(U+2060) after that text's first character. It looks the same, but llama-server reads it as text. llama-server reads
every special token's text in a prompt with pictures as the token itself, while Puddle keeps such text inside a
message as text. Prompts without pictures go to llama-server as token ids and are not changed.

## How Puddle talks to llama-server

For developers. `LlamaServerEngine` in `serve/server.py`:

- starts llama-server, waits for `/health`, and reads the context and the media marker from `/props`;
- sends each request to `POST /completion`. A prompt without pictures goes as Puddle's token ids. A prompt with
  pictures goes as text plus the pictures' files. The reply streams back with each token's id and the prompt's
  progress (the heartbeats, the Monitor and `keepalive_events` read it);
- stops a request by closing its connection;
- takes its figures from llama-server's last event, which has its own clock.

llama-server sends a token's id only once the text ends on a whole character. A character split across several
tokens comes with the last of them, and Puddle rebuilds the ids from the text. With one token missing the rebuild is
exact; with two or more the text is the same, and the ids can differ from the ones the model wrote.

llama.cpp also ends a reply at Bolt's `</s>`, which is not the model's end of turn (`<|im_end|>`). Such a reply ends
as a stop, and the token is not shown.

The tests run without a GPU, against a stand-in for llama-server: `python -m unittest serve.test_llama_server
serve.test_llama_server_images`.

## Nemotron 3.5 Super on strata-poly

NVIDIA's [Nemotron 3.5 Super VL 120B-A12B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16),
under the OpenMDW-1.1 license, in [Unsloth](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF)'s
UD-Q3_K_XL quantization: Bolt's architecture at its full size. **Status: experimental**, run on echo only.

| | |
| --- | --- |
| Layers | 40 Mamba-2, 8 attention and 40 MoE, and an MTP block (89 blocks in the file) |
| Experts | 512 routed experts a layer, 22 used for each token, in a latent of 1,024; a shared expert |
| The file | 65,761,541,032 bytes in 3 parts: the routed experts 55.6 GB (IQ4_XS / IQ3_XXS / IQ2_XS up, IQ4_NL / Q2_0 down, 4.05 bits a weight), the rest 7.1 GB, the MTP block 3.1 GB |
| Context | trained to 1,048,576 tokens; setup offers up to 256K (9,216 bytes of VRAM a token, the MTP block's rows among them) |
| Tokenizer | the same Tekken pre-tokenizer as Bolt's (`tekken` in this GGUF) |

`./setup.sh --setup --family nemotron-super` installs it like the other strata-poly models: the experts that do not
fit the cards stay in RAM, about 51 GB beside two 16 GB cards and 62 GB beside one by setup's estimate (about 3
GB more with pictures on the GPU), the MTP block's experts among them ([below](#the-mtp-block)). On echo at a 32K
context one RTX 5060 Ti 16 GB held 6.0 GiB of experts and the two 16.3 GiB: each card's engine takes VRAM of its
own beside its share of the experts (setup counts 5.7 GiB for Super's second card). Pictures go through
Puddle's image encoder (strata-vision), built from llama.cpp `08c252e22`'s mtmd, which has this encoder
(`nemotron_omni`, the 1.7 GB `mmproj-BF16.gguf`, at its own limits: up to 3,407,872 pixels, as llama-server reads
Bolt's). The server puts `<img>` and `</img>` around the picture's rows, as mtmd does; strata-poly reads them as text
(causal, no rotation), so a picture may take more than one prompt batch. `--vision gpu` in setup turns them on. The
encoder warms up at its largest picture before the engine starts, so the engine sizes its expert cache around it:
on echo at a 256K context it took the room of 716 experts, 6,451 to 5,735 of 20,480 in VRAM (16.6 to 14.1 GiB).

**Measured** on echo (two RTX 5060 Ti 16 GB, i9-9900K, 128 GB of DDR4-3200, Linux), 2026-11-21, with
[`tools/poly_parity.py`](../tools/poly_parity.py): llama.cpp `08c252e22` (llama-server) answers each prompt greedily,
strata-poly is made to write the same answer, and each step's first choice is compared.

| Prompt | Super: first choices the same | Bolt: first choices the same |
| --- | ---: | ---: |
| short (5 tokens) | 32 of 32 | 63 of 64 (the other a near tie) |
| code (22 tokens) | 32 of 32 | 64 of 64 |
| chat (11 tokens) | 49 of 49 | 48 of 48 |
| doc (2,109 tokens) | 64 of 64 | 23 of 23 |
| picture (318 tokens, 300 of them the picture) | 63 of 64 (the other a near tie) | 64 of 64 |

The picture is llama.cpp's `tools/mtmd/test-1.jpeg` (640x488, a newspaper's front page), read by each program's own
encoder from the same encoder file (`--image <picture> --mmproj <file>`): llama-server read 318 prompt tokens,
as many as strata-poly's prompt has. Through Puddle's server, asked for the headline, Super gave it and the line
under it word for word.

A first look at Super's speed in those runs, not a benchmark: with 6,647 of its 20,480 experts in VRAM (17.1 GiB),
it wrote 13.7 tokens/s after a short prompt, with the expert cache held still, no guesses and every launch made anew.
The CPU's experts took 50 to 54 ms of each token.

Since 2026-11-22 the four layers whose down matrices are Q2_0 read a prompt in batches too
([BACKENDS.md](BACKENDS.md#q2_0-experts-in-mmq-tiles-on-by-default-strata_mmq_q2_00-turns-it-off)). Through Puddle
on echo, Super then read 192.6 tokens/s at 2,520 tokens, 618.4 at 26,460 and 597.9 at 96,137, instead of 110.4, 202.6
and 215.5; writing stayed at 17 to 20 tokens/s.

### The MTP block

Super's file has a block after its 88 layers that guesses the token after the next: the token's embedding and the
model's output, an attention sublayer, 512 routed experts (Q8_0) and a shared expert, a LayerNorm and the model's
output matrix. strata-poly runs it as llama.cpp `08c252e22`'s `graph_mtp` does (`NemotronH::draft_step()`), and checks
its guesses in windows, so the answer is the model's: greedy, it wrote the same 256 tokens with the block and without
it (`--no-draft`). The block's experts take 3.0 GB more RAM (measured: the engine's mapped file after loading, 48.06
against 45.05 GiB) and its weights 0.2 GiB of VRAM (75 fewer experts in the cache).

**Measured** on echo (two RTX 5060 Ti 16 GB), 2026-11-22, through Puddle (`puddle.py benchmark`, 256 tokens written):

| Prompt | writing without | writing with | its guesses right | reading without | reading with |
| --- | ---: | ---: | ---: | ---: | ---: |
| 27 tokens | 18.3 tokens/s | 22.2 | 92% | | |
| 2,520 tokens | 17.5 | 20.9 | 91% | 192.6 tokens/s | 191.4 |
| 26,460 tokens | 19.5 | 23.6 | 85% | 618.4 | 614.6 |
| about 96K tokens | 17.3 | 21.8 | 87% | 597.9 | 594.6 |

(The longest prompt was 96,137 tokens without and 96,159 with: its text changed between the runs.) The block's
rows for a prompt cost its reading 0.6%. The same day llama-server (llama.cpp `08c252e22`, its default fit of the
two cards) wrote 256 tokens after a 32-token prompt at 18.41 tokens/s with its MTP head (`--spec-type draft-mtp`, 178
of 230 guesses right) and 14.15 without; strata-poly, run alone on the same tokens, 24.73 and 21.04.

## Nemotron 3.5 Bolt on strata-poly

Bolt's default since 2026-11-21 ([Install](#install)). The engine starts with `--no-draft` (Bolt has no MTP block).
`--vision gpu` adds pictures through the image encoder Super uses (strata-vision with NVIDIA's
`mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf`, which has the same settings as Super's).

strata-poly does not need every layer on the cards: the experts that do not fit stay in RAM. **Measured** on echo
(2026-11-21, strata-poly started from the shell, a 32K context, no image encoder):

| Cards | Routed experts in VRAM |
| --- | ---: |
| one RTX 5060 Ti 16 GB | 7,913 of 13,824 (8.9 GiB) |
| two RTX 5060 Ti 16 GB | all 13,824 (15.9 GiB) |

From that, setup estimates about 17 GB of RAM beside one 16 GB card. Its first choices matched llama.cpp's at 198 of
199 steps, and at 64 of 64 on a picture ([the table above](#nemotron-35-super-on-strata-poly)). Through Puddle's server
on two cards with the image encoder (all 13,824 experts still in VRAM: 7.6 and 8.3 GiB), asked for the headline of
`test-1.jpeg`, it gave it and the line under it, and wrote 824 tokens at 69.8 tokens/s after a 335-token prompt.

**Against llama-server, the same day** (echo, 2026-11-21): the same two requests, greedy, through Puddle to each
engine by name. strata-poly's config at a 256K context, llama-server's at 168,000 with `--flash-attn on` and a q8_0
cache (its run config above); both with the image encoder loaded. The long request is the first 45,000 characters of
[BACKENDS.md](BACKENDS.md) and "Summarize the text above in five bullet points." (17,613 tokens); the short one asks
for a Python function that checks for primes (35 tokens). Each wrote up to 1,000 tokens.

| | strata-poly | llama-server |
| --- | ---: | ---: |
| Read the long prompt (17,613 tokens) | 1,766.8 tokens/s | 2,347.6 tokens/s |
| Wrote 1,000 tokens after it | 60.4 tokens/s | 56.5 tokens/s |
| Wrote after the short prompt | 69.5 tokens/s (647 tokens) | 61.1 tokens/s (691 tokens) |

strata-poly wrote 7% faster after the long prompt and 14% faster after the short one, and read the long prompt 25%
slower. In all, the long request took about 26.5 s on strata-poly and 25.2 s on llama-server.

Since then, a prompt's attention goes through llama.cpp's flash-attention kernel
([BACKENDS.md](BACKENDS.md#flash-attention-for-prompt-reading-on-by-default-strata_fattn0-turns-it-off)): Bolt read
a 96,851-token prompt at 2,021 tokens/s instead of 1,290 (llama-server 1,997), and a 26,460-token one at 2,090
instead of 1,807 (llama-server 2,387). At 2,520 tokens it read 1,473 tokens/s, as before (llama-server 1,696).
And a prompt's Mamba-2 scan runs as chunked products on the tensor cores, as llama.cpp's does
([BACKENDS.md](BACKENDS.md#mamba-2-scan-in-chunked-products-on-by-default-strata_mamba2_ssd0-turns-it-off)): Bolt
then read 1,548 tokens/s at 2,520 tokens, 2,364 at 26,460 and 2,285 at 96,548 (2026-11-22). From 4,096
positions of context an answer's attention goes through the tensor-core decode kernel
([BACKENDS.md](BACKENDS.md#tensor-core-decode-attention-on-by-default-from-4096-tokens-of-context-strata_fdec0-turns-it-off)):
Bolt wrote 58.4 tokens/s after 26,460 tokens and 53.1 after 96,165 with it, instead of 53.7 and 40.4 (llama-server
54.6 and 42.3). With its layers' launches recorded once and replayed
([BACKENDS.md](BACKENDS.md#recorded-launches-for-the-nemotrons-on-without-the-mtp-block-strata_nemotron_graphs0-turns-it-off)),
79.5 after 27 tokens, 65.6 after 2,520, 62.9 after 26,460 and 58.0 after 96,199.

Most of what is left is a choice of llama-server's: on an RTX 50 card it rounds the activations of NVFP4 products
(Bolt's Mamba-2 projections) to FP4, strata-poly to 8 bits. Like for like, with llama-server made to round them to
8 bits as well (`GGML_CUDA_MMQ_PREC=q8`), strata-poly read a 2,175-token prompt at 1,486 tokens/s and llama-server at
1,408 (2026-11-22, a kernel profile of each: the NVFP4 products 357 ms against 369, the experts about 540 against
731, the Mamba-2 scan 334 against about 130), and their first choices were the same at all 64 steps of the text
and of a picture. strata-poly's FP4 rounding is a setting
([BACKENDS.md](BACKENDS.md#fp4-activations-for-nvfp4-prompt-products-strata_prefill_nvfp4w4a4)).

### Unsloth's UD-Q5_K_XL

[Unsloth](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF)'s quantization of Bolt, 58,912,632,064
bytes in 3 parts: the routed experts 50.3 GB (Q5_K / Q6_K up, Q6_K / Q8_0 down, 6.08 bits a weight), the other
weights at more bits than NVIDIA's file (one layer's shared expert and latent in BF16). `--model UD-Q5_K_XL` installs
it; NVIDIA's 24GB file stays the default. **Measured** on echo, 2026-11-22:

| Prompt (tools/poly_parity.py, llama.cpp `08c252e22`) | first choices the same |
| --- | ---: |
| short (5 tokens) | 32 of 32 |
| code (22 tokens) | 32 of 32 |
| chat (11 tokens) | 32 of 32 |
| doc (2,190 tokens) | 31 of 32 (the other a near tie) |
| picture (318 tokens, NVIDIA's image encoder) | 30 of 32 (both others near ties) |

At a 32K context one RTX 5060 Ti 16 GB held 1,617 of its 13,824 experts (5.4 GiB), and two held 4,620 (15.5 GiB),
so most of its experts run on the processor. From the shell, 32 tokens then 256 written on the two cards, it wrote
29.5 tokens/s (30.2 with its layers' launches recorded, the default since). Through Puddle (`puddle.py benchmark`,
setup's config: both cards, a 256K context, pictures on, 256 tokens written):

| Prompt | read | first token after | writing |
| --- | ---: | ---: | ---: |
| 27 tokens | 31.6 tokens/s | 0.86 s | 29.2 tokens/s |
| 2,520 tokens | 214.5 | 11.9 s | 28.1 |
| 26,460 tokens | 768.0 | 31.9 s | 30.6 |
| 95,812 tokens | 736.1 | 100.0 s | 29.6 |

### Unsloth's UD-Q4_K_XL

The same repository and revision, 50,219,478,368 bytes in 3 parts: the routed experts 42.6 GB, up and down each a
mix of IQ4_XS, Q4_K, Q5_K and Q6_K by layer (Q5_K the most, 25.4 GB), 5.16 bits a weight. `--model UD-Q4_K_XL`
installs it. **Measured** on echo, 2026-11-23:

| Prompt (tools/poly_parity.py, llama.cpp `08c252e22`) | first choices the same |
| --- | ---: |
| short (5 tokens) | 32 of 32 |
| code (22 tokens) | 32 of 32 |
| chat (11 tokens) | 32 of 32 |
| doc (2,289 tokens) | 31 of 32 (the other a near tie) |
| picture (318 tokens, NVIDIA's image encoder) | 31 of 32 (the other a near tie) |

At a 32K context one RTX 5060 Ti 16 GB held 2,248 of its 13,824 experts (6.5 GiB), and two held 5,886 (16.9 GiB).

## Not there yet

- strata-poly's speed on both: guesses, and Super's recorded launches (with its MTP block its guesses fell); Super's
  writing, with most of its experts on the processor.
- NVIDIA's DFlash draft model (a separate file, `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF`). Setup does not
  download or use it.
- On llama-server: `--vram-reserve-mib`, several requests at once, and replicas: llama-server is run with one slot.
