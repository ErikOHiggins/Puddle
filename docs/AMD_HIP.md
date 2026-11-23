# AMD Radeon: strata-poly through HIP

strata-poly is CUDA-shaped code that also compiles for AMD cards through the `hip_compat` headers (the CUDA runtime,
BLAS and virtual-memory calls mapped onto HIP's). Setup compiles it for the card; there is no ready-made AMD engine.

**Status: experimental, and untested with Puddle's models.** Laguna, Occamy and Gemma 4 have not been run on an AMD
card. What is below was measured by PolyStrata's contributors on other models of the same engine. Reports are welcome
([COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md)).

Setup accepts the RX 7900 XT / XTX (gfx1100), RX 7800 XT / 7700 XT (gfx1101), RX 7600 / 7600 XT (gfx1102), RX 9060 XT
(gfx1200), RX 9070 / 9070 XT / Radeon AI PRO R9700 (gfx1201), the RX 6800 / 6900 / 6700 XT / 6600 series (gfx1030,
gfx1031, gfx1032), the RX 5600 XT / 5700 / 5700 XT (gfx1010) and the Ryzen AI Max "Strix Halo" APU (gfx1151,
experimental). The Radeon 780M / 760M (gfx1103) is opt-in (`STRATA_EXPERIMENTAL_GFX1103=1`, unvalidated). Other
AMD architectures and AMD and NVIDIA cards in one run are not supported.

## Install with setup

The steps are the same as with NVIDIA: `./setup.sh` on Linux, `START-HERE.bat` on Windows. On a PC with no NVIDIA card
Puddle can use, setup chooses the AMD engine by itself; `--backend hip` chooses it on a PC that has both.

- **Linux:** setup finds the card through the kernel's KFD topology (the amdgpu driver; no ROCm install needed). It
  uses a system ROCm 7 in `/opt/rocm` (or `$ROCM_PATH`) when there is one, else it installs ROCm into `.venv` from
  AMD's TheRock wheels (~10 GB, no sudo), from the card family's index (`gfx110X-dgpu`, `gfx120X-all`, `gfx103X-all`,
  `gfx101X-dgpu`, `gfx1151`), at a version pinned per family. When the index no longer offers it, setup warns and takes
  the newest of the same line. `STRATA_ROCM_VERSION` / `STRATA_ROCM_INDEX` override them. The engine is compiled for the
  card into `engine-poly/` (10-20 minutes, once; again after a `git pull` that changes it); this needs a C++ compiler and
  git (`sudo apt install build-essential git`).
- **Windows:** you need Windows 10 or 11, one of the cards above and a current AMD driver (AMD Software: Adrenalin
  Edition). Setup installs AMD's TheRock Windows wheels into `.rocm-win` (about 7 GB) and compiles with Visual Studio
  2022's C++ build tools, which it offers to install. `STRATA_ROCM_WIN_VERSION` picks the wheels' version; a version
  other than the one `.rocm-win` holds is installed in its place.
- **Several cards (Linux):** setup takes one card (the one with the most VRAM, or `--gpu N`) unless you name more:
  `--gpus 1,0` splits the model's layers across them, the first one the main card. Cards of two families (gfx1100 +
  gfx1201) need a system ROCm 7: AMD's wheels hold one family.
- **No pictures:** the image encoder is a CUDA program.
- **The attention caches take their VRAM at load** (`--kv-whole`; setup writes it): on ROCm the VRAM that
  `hipMemUnmap` and `hipMemRelease` give back never returns to the free pool (an RX 7600 showed 48 MiB free after
  releasing 1 GiB, [PolyStrata #13](https://github.com/simplexai-labs/PolyStrata/issues/13)), so caches that grow with
  the conversation could not get it back.

## What was measured

- Opt-in `--vram-pool` (strata-poly, one card, VMM): the engine claims the card's VRAM once at load, as 2 MiB
  handles, and keeps it for the whole run. The attention caches and the expert cache trade handles of that pool as the
  context grows and shrinks, and nothing is released to the card, so the "released VRAM never returns" above does not
  arise (not tried on Linux ROCm). `--vram-reserve-mib` is then the margin left outside the pool (256 MiB at least),
  not VRAM kept free and idle. Under Windows the claim is sized by the budget Windows gives the process, read again
  every 256 MiB while the pool grows, because that budget falls as the process does (on an RX 5700 XT, 8 GB: 7,383
  to 6,956 MiB while 3.4 GiB were claimed), and memory past it is moved out to system RAM: sized from the card's free
  VRAM, a margin of 256 MiB left the process 150 MiB over its budget and 2.4 GB were moved out (decode 6 tokens/s).
  The pool is capped at what the model can use (every expert slot, the caches' whole address range and 512 MiB), so a
  card that holds the model keeps the rest. Measured on that RX 5700 XT (Ryzen 5 3600X, 32 GB, Windows 11), Tiel
  Coder 35B-A3B UD-Q4_K_XL, `--max-context 262144 --kv q8_0`, a 34K-token prompt between short chats:
  `--vram-reserve-mib 2048` without the pool held 1,172 experts and wrote 40 to 46 tokens/s after the prompt;
  `--vram-pool` with a margin of 256 MiB held 1,882 and wrote 50 to 53, with the same 648 MiB of shared GPU memory and
  the prompt read at 312 against 314 tokens/s. A long read grows the pinned host buffers, which count as shared GPU
  memory (+460 MiB after a 53K prompt); that is not a spill. `STRATA_FREEZE_EXPERTS` cannot grow the caches past the
  pool.
- Measured by the port's author on an RX 7600 (gfx1102, 8 GB; Ryzen 5 5600X, 30 GB of RAM) with PolyStrata's
  benchmark, `--kv-whole` on ([PolyStrata #13](https://github.com/simplexai-labs/PolyStrata/issues/13)):

  | Model | Short chat, writes | After a 32K prompt, reads / writes |
  | --- | --- | --- |
  | Qwen3.6-35B-A3B UD-Q2_K_XL | 74 tokens/s | 481 / 53 tokens/s |
  | Ornith 1.5 APEX-I-Mini | 40 tokens/s | 444 / 43 tokens/s |

  The first tokens after a load come slower while the expert cache fills (the port's own steady-state figures
  were 64 and 35 tokens/s, with 5,125 and 4,517 experts in VRAM).
- Measured under Windows 11 on an RX 9070 XT (gfx1201, 16 GB; Ryzen 7 7800X3D, 64 GB of RAM), engine 0.1.45
  compiled by setup from AMD's TheRock wheels (ROCm 10.2.0a20260930), Tiel Coder 35B-A3B UD-Q3_K_XL, 32K of
  context, `--kv-whole`, one request at a time: the engine loaded in 8.5 s (15.5 s the first time) with 8,031 of
  10,496 experts in VRAM (10.5 GiB); a 7,376-token prompt was read at 2,750 tokens/s, and the next request on
  that same prompt came from the conversation cache (`--conversation-cache-mib 8192`: 7,375 of its tokens
  reused, 25 ms); answers were written at 55 to 75 tokens/s (13 to 18 ms/token), with 72 to 93% of the guessed
  tokens right. An answer right after a load wrote at 30 to 34 tokens/s while the experts the cache did not hold
  were read in. This is one model on one card.
- The same greedy tokens as on CUDA are not promised across backends, and the card's experts and the
  CPU's do not give the same ones on either: with every expert on the CPU (`--no-gpu-experts`) a greedy answer
  parts from the card's after 15 to 50 tokens on the RX 7600, and after 11 to 70 on an RTX 5090 (Qwen3.6
  UD-Q2_K_XL and UD-Q3_K_M, a 24-token and a 1.2K-token prompt, 256 tokens, mean KL of the top 40 before the
  parting 0.0001 to 0.005; each side gives itself the same tokens in a second round). The two sum an expert's
  products in a different order, and a near tie goes the other way. On CUDA the port changes nothing: the same
  tokens and speed as before on an RTX 5090.

## Opt-in split value reduction

`STRATA_HIP_VALUE_SPLIT=1` splits the weighted-value reduction into 1,024-key
parts, then sums those parts. It applies only to `attend()` with head dimension
256, eight query heads per KV head, at least 12,288 keys, and no sliding window,
ring buffer or whole-context attention. Both FP16 and Q8_0 value caches are
supported. The existing score and softmax kernels still run.

The option reuses the score scratch space after softmax; it allocates no extra
VRAM. It falls back to the existing reduction if transposed weights or scratch
space are unavailable. Only the literal value `1` enables it; unset, `0`, and
other values retain the default path. Set it before starting the engine: a
captured HIP graph retains the path selected at capture time.

This targets the D256/GQA8 full-attention path of qwen35moe models such as Qwen3.6-35B-A3B (and Occamy's
architecture):
its published configuration has 16 query heads, two KV heads and 10
full-attention layers among 40. Variants with the same geometry are eligible
under the runtime gates above; linear-attention layers are unaffected.
The engine logs availability once the model geometry is known.

On the tested RX 9070 XT configuration, three restarted runs per flag measured
58.5 -> 59.8 tokens/s at 12K, 48.5 -> 52.3 at 32K and 36.5 -> 44.6 at 98K.
The 12K ranges overlap; the repeatable gains are +7.8% at 32K and +22.2% at 98K.
Prompt processing stayed within 3.8%, and peak sampled VRAM was unchanged.

Benchmark representative: Cyber-Tiel-Coder-35B-A3B UD-Q4_K_XL, a community
`qwen35moe` derivative with this geometry, using Q8_0 KV. These numbers are
measured on that artifact. End-to-end gains on other compatible variants
depend on quantization and CPU/GPU expert placement. The separate results
submission records exact artifacts, hashes, raw runs and reproduction commands.

The enabled reduction changes FP32 addition order, so its output need not be
bitwise identical to the default. The option targets long-context decode;
it leaves prefill unchanged.

The project's existing `STRATA_FDEC` path (PR #51) also splits decode
attention into 1,024-key parts and merges them, but uses CUDA tensor cores and
keeps the old path on HIP. This new option is a HIP value-reduction kernel; it
does not replace `STRATA_FDEC` and makes no claim to the general split-key
algorithm itself.

## Opt-in split-key decode attention (wave32 HIP)

`STRATA_HIP_SPLIT_ATTEND=1` (or `=2`) replaces the three decode attention
kernels (scores, softmax, values), which walk the whole context in 16 to 32
blocks, with one kernel that splits the keys into blocks of 512, keeps a running
softmax per block and joins the blocks afterwards. It is the fuller version of
the idea behind `STRATA_HIP_VALUE_SPLIT`: the keys are split in the score phase
too, so the card has enough blocks to read the cache at its memory speed.

It applies only to `attend()` with head dimension 256, eight query heads per KV
head, a Q8_0 key and value cache, at least 4,096 keys, no sliding window, ring
buffer, sinks or whole-context attention, and a card with 32-lane waves. Up to
four tokens go in one launch (a decode step with MTP drafts). It needs room for
its partial sums in the existing score scratch (the engine passes its size as
`part_floats`); where that is short, or any condition
fails, the old kernels run unchanged. Only the values `1` and `2` enable it;
unset, `0` and other values keep the default path, byte-identical to before.
Set it before starting the engine: a captured HIP graph keeps the path chosen
at capture time. It allocates no VRAM.

* `=1` reads the queries as FP32, as the old kernels do. Against an FP64
  reference its error is about 1e-8 on outputs of 0.005 to 0.03, the same size
  as the old kernels' (8e-9).
* `=2` also quantizes each query to int8 in blocks of 32 values and computes
  the scores with integer dot products (`dp4a`; on gfx1010 and gfx1012 the
  same SDWA sequence the other kernels use). Its error against FP64 is
  1.2e-5 to 5.7e-5 on the same outputs, about 0.2% of the largest value. That
  is a different rounding of the query, not a reordering: judge it as you
  would a lower-precision attention. `=1` is the choice when exactness matters.

Measured on an RX 5700 XT (gfx1010, 8 GB), Windows, `strata-poly` with
Cyber-Tiel-Coder-35B-A3B UD-Q4_K_XL, Q8_0 KV, MTP on. A microbenchmark of
`attend()` alone at 110K keys: one token 2.34 ms with the old kernels, 1.62 with
`STRATA_HIP_VALUE_SPLIT=1`, 1.05 with `=1` and 0.91 with `=2`; four tokens 8.54,
8.56, 3.42 and 2.06 ms (2.5x and 4.1x faster than the old kernels). At 8K keys
the gain is 1.3x to 1.9x, below 4,096 keys the old kernels run. On a replay of a
recorded agentic session (eight requests at 90K to 112K tokens of context, the
same each time), decode went from 24.5 tokens/s with the old kernels to 27.5
with the value split, 32.2 with `=1` (graphs off) and 37.4 with `=2` (graphs
on), with 76% to 88% of drafts accepted in each run. At 100K tokens of context
attention was about 57% of a decode step before. Prompt processing is unchanged (`attend_batch` is another path).

A needle test (a code word hidden in 54K tokens, asked for twice) found the
word both times with `=1`.

What was not tested: any card but gfx1010, the CUDA build (the kernel
is HIP only and compiled out elsewhere), other head geometries (they use the
old kernels), and the text of long generations beyond the checks below.
`poly_hip_split_attend_test` (79 checks, about 12 s, needs a card; exits 77
without one) compares both modes with an FP64 reference and the old kernels
at sizes around the 4,096-key start and the 512-key blocks, with the position
read on the card, the short-scratch fallback and a graph replay at three
positions.

## Opt-in FP32 prompt attention

`STRATA_HIP_PREFILL_FP32=1` or `=2` runs the two products of the prompt's attention (`attend_batch()`: the
scores, then the weighted values, for every block of keys) on FP32 inputs instead of FP16 ones with an FP32
sum. On an RX 5700 XT (gfx1010) hipBLAS's FP32 products reach 4.5 to 5 TFLOPS, its FP16-input ones 3 to 4: the
card has no packed FP16 math with an FP32 sum, so the halves only add the work of rounding to them. The
queries, the keys and values and the softmax's weights stay FP32. It is also closer to the exact attention:
against an FP64 reference, on outputs of 0.009 to 0.04, the default path's largest error is 2e-6 to 1.5e-5
(about 3e-4 of the largest output, from rounding to halves), these paths' 6e-9 to 9e-8.

- `=1`: hipBLAS SGEMM on an FP32 copy of each block of the cache. Needs a q8 workspace of at least 256 FP32
  rows (the engine's holds 1,024); a block of keys is shortened by 16 where it would be a multiple of 1,024,
  at which hipBLAS's FP32 values product runs at half its speed.
- `=2`: our own kernels (`src/poly/hip_prefill_gemm.cuh`) read the q8_0 codes and fp16 scales as they lie,
  so there is no FP32 copy, no dequantizing kernel and no workspace. 128 x 128 x 16 register tiles of
  8 x 8 outputs a thread, double-buffered in LDS, a loop without bounds checks for whole tiles. On the
  engine's shapes (4,096 columns, 900 to 1,400 keys a block, two KV heads of 256) they reach 5.4 TFLOPS for the
  scores and 6.3 to 6.6 for the values, against hipBLAS SGEMM's 4.5 and 5.0 (the card's peak is 9.75).

Both apply to a q8_0 cache for K and V and a score scratch four times the grouped queries' room (`=2` also to
heads of a multiple of 32 values); anything else runs the default path unchanged. Unset, `0` and other values
keep the default path, byte-identical to before. No extra VRAM: the scores, the weights and the grouped
queries share the score scratch. Decode, `STRATA_FDEC` and the CUDA build are unaffected (the code
is compiled only under HIP). Set it before starting the engine.

Measured on an RX 5700 XT 8 GB, Windows, with the engine's buffers (512 tokens a chunk, 16 query heads, two KV
heads of 256, q8_0 cache): `attend_batch()` for 512 tokens, one layer (the model has ten), default / `=1` /
`=2`: at 30,000 keys 83.4 / 61.5 / 50.9 ms, at 100,000 keys 277 / 204 / 169 ms (1.36x and 1.64x). On a replay
of a recorded agentic session (8 requests at 90K to 112K tokens of context, Cyber-Tiel-Coder-35B-A3B
UD-Q4_K_XL, MTP on) the engine read new prompt tokens at 122 / 145 / 159 tokens/s and a cold 90K prompt at
216 / 247 / 266; decode speed was unchanged (37.4 / 37.5 / 38.6 tokens/s, within the noise of the runs). A
needle test (a code word in 54K tokens, three rounds) found it every time with both.

What was not tested: any card but gfx1010, the CUDA build (the new kernels are compiled only under HIP),
other cache formats (an FP16 cache runs the default path) and text quality beyond the FP64 comparison above.
`poly_hip_prefill_fp32_test` (32 checks, about 10 s, needs a card; exits 77 without one) compares both levels
with an FP64 reference and the default path at several lengths and chunk sizes, for heads of 64, 128 and 256,
and checks the fallbacks.
