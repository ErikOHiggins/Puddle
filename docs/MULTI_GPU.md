# Two or more GPUs (layer split)

One model can run across several cards in one PC. This is pipeline (layer) parallelism, not tensor parallelism: each
card runs a range of the model's layers, and a token's rows cross from one card to the next through pinned RAM. No
NVLink or peer-to-peer access is needed; cards on x4 or x1 slots work. Each card keeps the routed experts of its own
layers, so two cards hold about twice the experts one card holds.

Laguna-S 2.1 and Laguna-XS 2.1 run this way on echo, on two RTX 5060 Ti 16 GB (PCIe 3.0 x8
and x4) under Linux, with `"layer_split": "auto"`. Laguna-S's speeds there: [BACKENDS.md](BACKENDS.md#laguna-on-two-16-gb-cards).
AMD cards split the same way on Linux ([AMD_HIP.md](AMD_HIP.md)).

Nemotron 3.5 Bolt on llama-server (`nemotron-bolt-llama`, opt-in) is split by llama-server instead, not by strata-poly: its run config's `--tensor-split` puts an
even share of the model's layers on each card in `"gpu"`, every layer on a card. It ran this way on the same two cards
([NEMOTRON.md](NEMOTRON.md)). The rest of this page is about strata-poly.

## How it works

strata-poly splits a model's layers across two or more cards in one PC. Each card runs a range of layers and keeps the routed experts of those layers in its VRAM, as many
as fit. The last card also runs the output and the draft block, or Gemma 4's draft model. After its layers a card
copies the rows of the tokens it read into pinned RAM, and the next card reads them from there: one row of the
model's width a token (8 KB for a model 2,048 wide). No NVLink or peer-to-peer access is needed.

A prompt read in several calls keeps the cards busy together when every routed expert is in VRAM, or when the
prompt is shorter than `--stage-ahead` tokens: the next card runs its layers for a call on a thread of its own while
this card runs the call after it. A longer prompt with experts outside VRAM is read a layer at a time for all its
calls, so that each layer's experts are copied in once, and there the cards take turns. Having them overlap would
mean reading the prompt in parts, which copies every layer's experts once a part, and the copy is most of the time
such a read takes. An answer is written a token at a time, and the cards take turns on it too. So with no expert in
VRAM two cards read a long prompt no faster than one. What a split adds there is VRAM: two cards hold more of the
experts, and an expert on a card is not computed by the processor.

## Using it

Setup lists the cards and asks which to use (`--gpus 0,1`, or the list it shows), and writes
`"gpu": [0, 1]` and `"layer_split": "auto"` into the config. With `"auto"` each card gets a share of the layers'
bytes by its free VRAM less 1.5 GiB, and the last card's share is smaller by the output and the draft block. A list
of numbers is the first layer of each later card, as above, and the last card keeps two layers at least. The server
puts the faster card last. The engine's flag is `--layer-split auto|N[,N...]`; it
splits across the cards `CUDA_VISIBLE_DEVICES` shows it, in that order. The log names each card's layers:

```
strata-poly: layers 0 to 21 on card 0: 5632 of 5632 routed experts in VRAM (7.2 GiB)
strata-poly: layers 22 to 39 on card 1: 4864 of 4864 routed experts in VRAM (6.5 GiB)
```

## Measured

Measured by PolyStrata on two RTX 3080 (10 GB each, 320 W) with a Xeon Platinum 8255C under Linux, the engine built with CUDA
12.4: a prompt of 2,727 tokens read with nothing cached, and 256 tokens written after a short prompt, in three
starts of each. The engine of 2026-10-11 ran it again, three starts on one card and three on both in turn, and read
at 1,567-1,582 and 3,425-3,562 tokens/s and wrote at 165-177 and 217-221. A cell has the higher of the two passes.

| Qwen3.6 UD-Q3_K_M | Experts in VRAM | Reads the prompt | Writes |
| --- | ---: | ---: | ---: |
| one RTX 3080 | 4,606 of 10,496 | 1,570-1,700 tokens/s | 165-177 tokens/s |
| both | 10,496 of 10,496 | 3,420-3,560 tokens/s | 217-221 tokens/s |

With the cards taking turns on the prompt, as they did before, both read it at 3,070-3,090 tokens/s in two starts
of the same test, and wrote as fast.

Gemma 4 UD-Q3_K_M: a prompt of 3,527 tokens at 2,190 tokens/s on one card in one start, and at 3,600 on both in
two starts (3,110-3,160 with the cards taking turns); writing at 178 and 189 tokens/s in one start each. The engine
of 2026-10-11, two starts of each in turn, read it at 2,170 and 3,540-3,590 and wrote at 182-187 and 191-193. One
card already held 2,042 of its 3,840 experts, which were most of what an answer used.

With no expert in VRAM (`--vram-reserve-mib 20000`) and `--cpu-share 0`, and with every expert on the processor
(`--no-gpu-experts`), both cards gave the same greedy tokens as one card, for Qwen3.6 and for Gemma 4, through a
prompt read in several calls, the prompt cache, a checkpoint, the conversation cache and a request stopped in the
middle. With no expert in VRAM and the CPU's share at its default, Gemma 4 still did, and two of Qwen3.6's eight
answers came out other: the share follows how long the calls before took, so which experts the processor computes
changes from one run to the next. With experts in VRAM the answers can differ a little, as they do between two
starts on one card: which experts run on a card and which on the processor changes the rounding.

GLM 5.3 Flash REAP50-Q3_K_M (79 GB) ran on the same two cards with 80 GB of RAM. One 10 GB card cannot hold the
weights outside the routed experts, about 12 GB, so there is no one-card row. With `"auto"` the cards took layers 0
to 27 and 28 to 44 and held 602 of the 6,192 routed experts (6.7 GiB). A prompt of 6,087 tokens read at 397-398
tokens/s and 256 tokens were written at 20-21 tokens/s, in two starts; with no expert in VRAM, 367-384 and 16-16.5.
The engine of 2026-10-11 read it at 398-399 and wrote at 20-22 in two starts, and with no expert in VRAM at 386
and 14-16 in four.
With no expert in VRAM and `--cpu-share 0` the split at layer 28 and the split at layer 30 gave the same greedy
tokens for a short prompt, a long one read in several calls, the prompt cache and a checkpoint; with the share at
its default, six of the eight answers differed. The conversation cache was not compared:
with the RAM nearly full it parked nothing in the runs split at layer 28. Of eight starts of that test, one failed
its long prompt with `pinned memory: invalid argument` and one stopped without a message while reading it. Both
come from the container's memory: its limit is 80 GiB and the file is 73.4 GiB. With the whole file read into
the page cache before the start, so that the container was within 2% of its limit, the engine was ended by
SIGKILL in 3 of 6 starts of the speed test, each 17 s into the first long prompt, with no OOM kill counted by
the kernel for the container or for the host; the 6 starts in turn with no expert in VRAM, which copy nothing
to the cards, all ran through. The starts that lived wrote at 6 tokens/s where 20, the experts read from the
disk again. With the cache empty at the start, the same test ran through at about the speeds above. The five
`pinned memory: invalid argument` all came at load within five minutes, while a 78 GB download was filling the
same container's cache; the engine now asks four times, a quarter of a second apart, before it gives up, and
says at load when the container has less than the model's files and 8 GiB. This pod has less than setup asks
for REAP50 beside 6.7 GiB of experts in VRAM, about 82 GB.

## Not done yet

- Only two cards under Linux have run it. Three cards and Windows were not tried.
- A long prompt with experts outside VRAM, and an answer, have the cards take turns (above).
