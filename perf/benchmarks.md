# Benchmarks

The results below were measured on the RK3588 NPU through the `rocket` stack, per model and per
frontend. They cover prefill and decode throughput against the CPU, interactive latency, and
faithfulness to the CPU result. Each project README quotes a standalone excerpt of the figures for
its frontend.

## At a glance

The NPU is a prefill and batched-GEMM engine: it accelerates prompt processing (prefill) and the
Whisper and vision encoders. Decode (M=1 GEMV) stays on the CPU on both backends
([decode-gemv.md](decode-gemv.md)), and its rate depends on the quantization rather than on the
backend. The two effects are independent: the NPU shortens time-to-first-token, and quantization
raises the decode rate and determines whether a model fits in RAM. Each model's section below gives
its prefill, decode, interactive and faithfulness results.

**Conditions:** RK3588 on a Turing RK1 (4xA76 + 4xA55, shared LPDDR). Figures are warm, and each
comparison runs the same GGUF or model on the NPU and on the CPU [HW sweep]. Each table states its
NPU clock: 928 MHz is the 900 MHz request and ~816 MHz the 600 MHz request, both at the 0.80 V
rail ([clock.md](clock.md)).

> [!NOTE]
> The NPU clock returns to its 200 MHz idle rate between runs, so a cold run measures ~15% low.
> Run a discarded warm-up first, and compare warm runs.

### LLM and multimodal (llama.cpp via ggml-rocket)

The table gives each model's prefill at pp2048 against the CPU, its decode rate and its
faithfulness. Rows are sorted by size. In the table:

- Prefill runs the F16 GGUF where it fits a 31 GiB board. Otherwise it runs the top precision
  that does, named in the GGUF column: Q4_K_M, or MXFP4, the gpt-oss native block-float.
- Decode and the GGUF size are Q4_K_M's, the fastest decode and the smallest fit, and MXFP4's on
  gpt-oss-20b.
- PPL Δ is NPU−CPU on the same GGUF.

| Model | Params | GGUF | NPU prefill, pp2048, t/s | x CPU, pp2048 | Decode Q4, t/s | Q4 GGUF, GiB | NPU−CPU PPL Δ |
|---|---:|---|---:|---:|---:|---:|---:|
| SmolVLM2-2.2B (llama+SigLIP) | 1.8B | F16 | 117.8 | 4.2x | 20.7 | 1.0 | −0.07% |
| Llama-3.2-3B | 3.2B | F16 | 85.1 | 5.0x | 11.1 | 1.9 | +0.02% |
| Ministral-3-3B | 3.4B | F16 | 75.0 | 4.7x | 10.5 | 2.0 | −0.04% |
| Phi-4-mini | 3.8B | F16 | 76.1 | 5.0x | 8.8 | 2.3 | −0.12% |
| Ministral-3-8B | 8.5B | F16 | 39.6 | 5.8x | 4.7 | 4.8 | −0.01% |
| Qwen3.5-9B | 9.0B | F16 | 36.1 | 4.9x | 4.0 | 5.3 | +0.05% |
| Gemma-4-12B | 11.9B | F16 | 24.6 | 5.2x | 3.1 | 6.9 | −0.81% |
| Phi-4 (14B) | 14.7B | Q4_K_M | 21.9 | 4.0x | 2.7 | 8.3 | −0.31% |
| DeepSeek-V2-Lite (MoE+MLA) | 15.7B | Q4_K_M | 43.4 | 1.83x | 13.5 | 9.7 | −0.42% |
| gpt-oss-20b (MoE) | 20.9B | MXFP4 | 37.8 | 2.29x | 9.1 | 11.3 | cosine 0.9998 † |
| Qwen3.6-27B (hybrid) | 27.3B | Q4_K_M | 11.5 | 3.8x | 1.3 | 15.9 | −0.26% |

**Conditions:** NPU clock 928 MHz, CPU governor `performance`, llama.cpp b11539 [HW sweep
2026-11-01]. Each speedup pairs the NPU with the 8-thread CPU backend, the same binary with the NPU
backend unloaded, on the same GGUF. The F16 rows run at llama.cpp's default `-b` and `-ub`. The
four quantized rows run at `-b 2048 -ub 2048`, with llama.cpp's weight repack on in the CPU
backend. Decode is the CPU backend's `tg64` at `-t 4` on the four A76 cores (`taskset 0xf0`). The
PPL Δ column comes from each model's section, measured on 2026-07 builds and, for DeepSeek-V2-Lite,
on 2026-08-27.

† gpt-oss is a reasoning model, so wikitext PPL is not a valid metric for it. Its faithfulness
figure is the per-matmul cosine against an fp64 reference.

Notes on the table:

- Phi-4 (14B) also runs Q8_0, at essentially the Q4_K_M prefill on the ~816 MHz sweep of its
  section.
- Qwen3.5-9B also runs IQ4_XS (4.80 GiB) on the same NPU path. Its section gives the IQ4_XS
  figures.
- SmolVLM2's 1.0 GiB is its 1.81 B LLM half, plus ~0.8 GB fp16 vision mmproj.

> [!NOTE]
> The quantized rows' x CPU column compares against the CPU backend with llama.cpp's weight
> repack on. The NPU configuration runs without the repack. The repack speeds the CPU
> configuration 1.26-1.71x at pp2048 on these four models [HW sweep 2026-09-29, ~816 MHz,
> llama.cpp b11242].
> [cpu-repack-baseline.md](cpu-repack-baseline.md) has the method. The F16 rows are unaffected.

At pp2048 the NPU prefills a dense F16 model at 4.2-5.8x the CPU. It prefills a dense quantized
model at 3.8-4.0x the repacked CPU, and a MoE model at 1.83-2.29x. Held resident under
`ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, a Q4_K_M GGUF of 1.8-9.0 B prefills at
3.2-4.2x the repacked CPU [HW sweep 2026-11-02]. F16 prefill falls slightly with
prompt length: from pp512 to pp2048, Ministral-3-8B measures 44.0 -> 39.6 t/s and Gemma-4-12B 25.6
-> 24.6 t/s on the same series. A streamed quantized GGUF dequantizes its weights to fp16 again on
every micro-batch, which `-b 2048 -ub 2048` amortizes
([quant-prefill-microbatch.md](quant-prefill-microbatch.md)).

Decode at `taskset 0xf0 -t 4` measures 1.62-4.22x the 8-thread rate on the table's GGUFs [HW sweep
2026-11-01].

The MoE expert FFNs of gpt-oss-20b and DeepSeek-V2-Lite run through `MUL_MAT_ID`, whose NPU path
holds each quantized expert resident as native int8. Residency removes the per-micro-batch host
dequantization that makes the naive expert offload a loss. The path is on by default. On the
2026-08-27 build at ~816 MHz, against an unrepacked CPU, it took gpt-oss to 2.38x at pp2048, where
the experts on the CPU measured 1.15x.

The default claims an expert op only where every one of these holds [source-confirmed,
`ggml-rocket.cpp` `supports_op`]:

- The micro-batch has at least `ROCKET_MOE_MIN_TOKENS` (512) tokens.
- The expert stack is GGUF-quantized, with `K` and `N` multiples of 32.
- The per-expert row count `M_e` = `n_tokens · n_used / n_expert` reaches the 64-row granule,
  `ROCKET_MOE_M_BUCKET`.
- The per-expert GEMM contains enough work to repay its own dispatch cost: `M_e · K · N` of at
  least `ROCKET_MOE_MIN_WORK`, 340 mega-MACs.
- A residency pre-flight reserves the whole expert stack before ingesting it.

DeepSeek's experts (6-of-64 routing, 2048×1408 each) therefore stay on the CPU at short prefill,
and move onto the NPU past ~1250 tokens in a micro-batch
([moe-placement.md](moe-placement.md)).

DeepSeek adds MLA attention. The flash-attention (FA) offload accepts DK≠DV and is bit-faithful.
On DeepSeek-V2-Lite at `-b 2048 -ub 2048`, `-fa on` prefills 1.04-1.05x as fast as `-fa off` at
pp2048 and 1.10-1.11x at pp4096 [HW sweep 2026-11-02]. Its output under `-fa` was not compared
against the CPU's. Qwen3.6-27B's Gated-DeltaNet
hybrid keeps its linear-attention layers on the CPU, a small share of its prefill FLOPs. The two
MoE models decode faster than their parameter counts suggest (gpt-oss 9.1 t/s, DeepSeek 13.5 t/s),
because only ~3.6 B (gpt-oss) and ~2.4 B (DeepSeek) params are active per token.

Every NPU−CPU PPL Δ sits within its per-run stderr, so the fp16-accumulation prefill is faithful.
The absolute PPL of an instruct or reasoning model is not meaningful, and only the delta is.

SmolVLM2's vision half (SigLIP-SO400M, 729 tokens) also offloads, through clip.cpp. It runs in
7.90 -> 6.66 s warm (1.19x) at ~816 MHz, with a projected-embedding cosine of 0.99998 against the
CPU [HW sweep 2026-07-04]. The speedup is modest, limited by the graph breaking into 272 CPU↔NPU
splits. The figure is for the generic drop-in, not the resident `rocket_siglip_encoder`. The
offload requires `MTMD_BACKEND_DEVICE=ROCKET`.

### Tuning gain per model (stock -> recommended, pp2048)

The table gives the gain from each model's recommended settings over stock, with both
configurations on the NPU [HW sweep 2026-08-28..31]. Stock is llama-bench's own defaults with the
backend loaded: `-ub 512`, and every `ROCKET_*` option at its default, including MoE placement on
AUTO. Recommended is the model's row in `MODEL-NOTES.md` and `TUNING.md`. The recommended
configuration is per-model, not a universal recipe, and on two models it is the defaults
themselves. They are Qwen3-30B-A3B, the table's last row, and the Qwen3.5-0.8B F16 GGUF, which the
table omits (below).

On every quantized model that fits resident, wholly or in part, the recommended configuration is
**residency without the `-ub` raise**. Seven of seven models measured both ways, from 1.8 to
11.9 B (`llama-bench` parameter counts), are 3.5-34% faster with residency alone than with
residency combined with `-b 2048 -ub 2048` (the combined form).
Below 4 B, `-ub 2048` also has a loss that does not come from dequantization, and the combined form
incurs it. With the dequantization cost removed, `-ub 2048` measures 0.79-0.86x of `-ub 512` on
those models.

| Model | Recommended config | Stock t/s | Recommended t/s | Multiple |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B Q4_K_M | `-ub 2048` + `QUANT_RESIDENT` | 99.1 | 116.9 | **1.18x** |
| SmolVLM2-2.2B Q4_K_M | `QUANT_RESIDENT` at stock `-ub` | 51.5 | 69.3 | **1.35x** |
| Llama-3.2-3B Q4_K_M | `QUANT_RESIDENT` at stock `-ub` | 39.2 | 57.7 | **1.47x** |
| Llama-3.2-3B F16 | `F16_RESIDENT` | 56.0 | 60.5 | **1.08x** |
| Ministral-3-3B Q4_K_M | `QUANT_RESIDENT` at stock `-ub` | 34.6 | 50.0 | **1.45x** |
| Ministral-3-3B F16 | `F16_RESIDENT` | 48.3 | 52.0 | **1.08x** |
| Phi-4-mini Q4_K_M | `QUANT_RESIDENT` at stock `-ub` | 35.9 | 54.2 | **1.51x** |
| Phi-4-mini F16 | `F16_RESIDENT` | 50.2 | 54.4 | **1.08x** |
| Ministral-3-8B Q4_K_M | `QUANT_RESIDENT` at stock `-ub` | 17.8 | 29.4 | **1.65x** |
| Qwen3.5-9B Q4_K_M | `QUANT_RESIDENT` at stock `-ub` | 19.1 | 33.5 | **1.75x** |
| Gemma-4-12B Q4_K_M | `QUANT_RESIDENT` at the default `-ub` (83-87% resident) | 12.8 | 17.5 | **1.37x** |
| Phi-4-14B Q4_K_M | `-ub 2048` | 10.9 | 14.3 | **1.31x** |
| gpt-oss-20b MXFP4 (MoE) | `-ub 2048` (experts default-on) | 22.8 | 28.6 | **1.25x** |
| DeepSeek-V2-Lite Q4_K_M (MoE) | `-ub 2048` (experts default-on) | 21.3 | 28.2 | **1.32x** |
| Qwen3.6-27B Q4_K_M | `-ub 2048` | 6.1 | 9.3 | **1.53x** |
| Qwen3-30B-A3B Q4_K_M (MoE) | defaults, every tuned configuration loses | 14.7 | 14.7 | **1.00x** |

**Conditions:** pp2048, NPU clock ~816 MHz, governor pinned, both configurations on the NPU.

The table abbreviates the options. In it, `-ub 2048` means `-b 2048 -ub 2048`. The entry
`QUANT_RESIDENT` is `ROCKET_QUANT_RESIDENT=auto`, which dequantizes each weight once and holds it
resident as fp16. It needs the fp16 image in RAM. The entry `F16_RESIDENT` is
`ROCKET_F16_RESIDENT=auto`.

The Qwen3.5-0.8B row uses the combined form. At 928 MHz residency alone is bimodal by process
on that model, 132 t/s in three processes and 151-155 t/s in two. The combined form measures
140-142 t/s in all five [HW sweep 2026-11-02].

Every other resident row is residency alone, `Gemma-4-12B` included. Gemma-4-12B places only
part of its weights, and residency alone is still faster there. It also places more of them:
83-87% resident, against 73-74% for the combined form. The combined form, `-b 2048 -ub 2048`, uses that RAM for
compute buffers before the memory reserve stops placement.

The table omits the Qwen3.5-0.8B F16 GGUF, because its residency option reaches only 24 weights at
the K this model uses.

### ASR (Whisper encoder, whisper.cpp via ggml-rocket)

In ASR, the NPU runs the encoder. The decoder is autoregressive (M=1 GEMV) and runs on the CPU on
both backends. Transcripts are byte-identical between the CPU and the NPU (WER 0), and the
encoder-output cosine is 0.9998.

| Model | enc d_model / layers | CPU -> NPU | speedup |
|---|---|---:|---:|
| tiny.en | 384 / 4 | 697.7 -> 591.9 ms | 1.18x |
| base.en | 512 / 6 | 1513.5 -> 1015.8 ms | 1.49x |
| small.en | 768 / 12 | 5711.5 -> 3718.9 ms | 1.54x |
| medium.en | 1024 / 24 | 19268.2 -> 10444.9 ms | 1.84x |
| large-v3 | 1280 / 32 | 36399.9 -> 17013.5 ms | 2.14x |
| large-v3-turbo | 1280 / 32 | 32957.2 -> 15544.1 ms | 2.12x |

**Conditions:** NPU clock ~816 MHz, encoder latency from `whisper-bench` encode-only, warm
[HW sweep 2026-07-04]. The base.en row was measured 2026-09-26, with the CPU governor at
`performance` and both configurations under `taskset -c 4-7 -t 4`, flash attention on.

The CPU encoder's thread count does not move these ratios. At 8 unpinned threads it measures
within 2% of 4 threads on the A76 cores [HW sweep 2026-11-02, whisper.cpp v1.9.5]. Base measures
1510-1565 against 1522-1556 ms, and small 5419-5481 against 5530-5548 ms.

The speedup grows monotonically with the encoder (matmul work ~d_model², host packing ~linear).
The large-v3-turbo model pairs the full 32-layer encoder with a 4-layer decoder (~6x cheaper per
step). On large-v3-turbo, the NPU-accelerated encoder is the largest share of a real
transcription.

### ASR beyond Whisper (transcribe.cpp via ggml-rocket)

The same `.so` drops into transcribe.cpp, a ggml-based host for several ASR
models. It offloads a range of speech models beyond OpenAI Whisper [HW sweep]. The NPU offloads encoders, not autoregressive
decode, so the speedup depends on how encode-heavy the model is. On long audio it also depends on
how much of decode is offloadable prefill rather than per-token M=1 steps.

| Model | encoder + decoder | NPU x (120 s) | rt (NPU) | best at |
|---|---|---:|---:|---|
| SenseVoice-small 234M | SAN-M + CTC (single-pass) | 1.26x | 6.2x | lowest latency |
| Fun-ASR-nano 800M | SAN-M + Qwen3-0.6B AR | 1.15x | 1.75x | 31-language coverage |
| Granite-Speech-2B (base) | conformer + LLM cross-attn | 1.68x | 1.76x | fast clean transcript |
| Voxtral-mini 3B | Whisper-lg-v3 + Ministral-3B AR | 1.57x | 0.55x | best transcript + translation |
| MOSS 0.9B | Whisper-Med + Qwen3-0.6B AR | 1.08x | 0.42x | best diarization (33 seg + timestamps) |

**Conditions:** Q8_0, warm, pinned to the A76 cores, a hard 120 s two-speaker clip, one fresh
process per figure [HW sweep 2026-07-21].

**Not characterized:** the NPU clock of these runs.

The cross-attention decoder (Granite) offloads decode best, because its per-step cross-attn over
the encoder is a large-K matmul (~1.9x). A large decoder-only model over long audio (Voxtral 3B)
offloads its 1500-token audio prefill (decode 1.46x). A small decoder-only model (MOSS and FunASR,
0.6B) keeps generation on the CPU (M=1), so its total changes little despite a 1.6x encode.

### Detection (SSD-MobileDet, tflite-rocket)

A single inference is bound by host work, so its latency depends on the host path and the CPU
governor. The NPU's larger gain is throughput under a multi-camera pool, the load Frigate runs.
Accuracy is COCO mAP.

| Metric | Value |
|---|---|
| COCO mAP@[.5:.95] | 0.3325 NPU vs 0.3301 CPU over 500 images (2026-10-29) |
| Single-stream latency, warm | 51.4 ms with the governor pinned, 73.5 ms on `ondemand`, NPU 928 MHz (2026-10-29) |
| Multi-camera pool, P=1->4 | 3.33-3.51x at P=4 in the delegate's own pool test (2026-10-29); live in Frigate, 3.20 -> 9.55 detection_fps (2.98x) on a 2026-07-01 build at ~816 MHz, ~336 ms a single inference |

## LLM (llama.cpp via ggml-rocket)

Each model reports prefill, which the NPU runs, and decode, which stays on the CPU on both
backends. An interactive table decomposes a turn into time-to-first-token (TTFT), set by prefill,
plus a streaming rate, set by decode. Faithfulness is a differential perplexity. The same GGUF runs
through the NPU prefill and through the CPU, and the NPU−CPU delta isolates the fp16-accumulation
fidelity of the NPU matmul.

**Conditions:** RK3588 on the RK1. Each table states its NPU clock and date. A 2026-11-01 figure
belongs to the series of the table in §"At a glance": NPU clock 928 MHz, governor `performance`,
llama.cpp b11539. That series' prefill baseline is the 8-thread CPU backend, and its decode runs at
`taskset 0xf0 -t 4`. The earlier per-model sweeps ran at ~816 MHz with the flags and 8-thread CPU
baseline of §"Method".

### Ministral-3-8B-Instruct-2512

Ministral-3-8B is Mistral's December-2025 8B (arch `mistral3`, GQA, interleaved sliding-window
attention), with 8.49 B params. The GGUFs come from `unsloth/Ministral-3-8B-Instruct-2512-GGUF`,
and F16 is derived from the published BF16.

**Prefill (prompt processing), t/s (the NPU's job).**

| test | F16 CPU->NPU | Q8_0 CPU->NPU | Q4_K_M CPU->NPU |
|---|---|---|---|
| pp512  | 7.2 -> **44.0** (6.1x) | 12.8 -> 50.2 (3.9x) | 10.3 -> 50.5 (4.9x) |
| pp1024 | 7.1 -> **41.5** (5.8x) | 12.4 -> 48.4 (3.9x) | 10.1 -> 47.9 (4.7x) |
| pp2048 | 6.8 -> **39.6** (5.8x) | 11.7 -> 45.0 (3.8x) | 9.6 -> 45.1 (4.7x) |

**Conditions:** NPU clock 928 MHz, the default `-b` and `-ub` [HW sweep 2026-11-01]. The quantized
GGUFs run under `ROCKET_QUANT_RESIDENT=auto`, against a CPU backend with llama.cpp's weight repack
on.

The quantized GGUFs prefill faster on the NPU than F16 does. Quantized residency holds all 235 of
their weights resident as fp16. The F16 GGUF streams its weights, because the default F16
residency admits only a weight with `K` <= 2048, and this model has none. The quantized speedups
are lower than F16's because the repacked CPU backend runs those GGUFs faster.

A board without RAM for the fp16 copy streams the quantized GGUF. Each weight is then dequantized
to fp16 again on every micro-batch, which `-b 2048 -ub 2048` amortizes
([quant-prefill-microbatch.md](quant-prefill-microbatch.md)):

| test | Q8_0 CPU->NPU, streamed | Q4_K_M CPU->NPU, streamed |
|---|---|---|
| pp512  | 6.9 -> 17.4 (2.5x) | 6.5 -> 17.0 (2.6x) |
| pp1024 | 6.7 -> 19.3 (2.9x) | 6.4 -> 19.1 (3.0x) |
| pp2048 | 6.4 -> 17.7 (2.8x) | 6.0 -> 17.4 (2.9x) |

**Conditions:** NPU clock ~816 MHz, `-b 2048 -ub 2048`, against a CPU backend built without the
weight repack [HW sweep 2026-07-01].

**Decode (generation), t/s (CPU-bound, LPDDR-bandwidth-limited).**

| | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| tg64, CPU backend | 1.60 | 3.01 | 4.72 |

**Conditions:** `taskset 0xf0 -t 4`, llama.cpp b11539 [HW sweep 2026-11-01].

Decode runs on the CPU in both configurations ([decode-gemv.md](decode-gemv.md)). Its rate
follows the bytes each token reads: F16 1.6 -> Q8_0 3.0 -> Q4_K_M 4.7 t/s (2.9x).

**Interactive.** TTFT and streaming are computed from the rates above [host-computed]. The TTFT is
the prompt over the pp2048 rate, and the total adds 200 tokens at the decode rate.

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | F16 CPU | 301 s | 1.6 t/s | 426 s |
| | F16 + NPU | 52 s | 1.6 t/s | 177 s |
| | Q4_K_M + NPU, `ROCKET_QUANT_RESIDENT=auto` | **45 s** | 4.7 t/s | **88 s** |

On this prompt the NPU cuts the time to first token 5.8x and the turn 2.4x (F16). A short chat
turn is mostly decode, so the quantization sets its length: 400 tokens take 250 s at F16 and 85 s
at Q4_K_M.

**Faithfulness: differential perplexity, wikitext test, 12 chunks (same GGUF, NPU prefill vs
CPU).**

| | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| F16    | 6.8621 | 6.8615 | −0.01% |
| Q4_K_M | 6.9626 | 6.9597 | −0.04% |

**Conditions:** NPU clock ~816 MHz [HW sweep 2026-07-01].

The per-chunk stderr is ±0.43, so the NPU−CPU delta is ~100x smaller than the noise. The NPU
prefill matches the CPU to within 0.05% on both GGUFs, so the fp16-accumulation path has no
measurable accuracy cost. The Q4_K_M−F16 gap (6.96 vs 6.86) is the expected quantization cost,
present equally on both backends.

**Verdict.** Run this model as Q4_K_M with the NPU. Under `ROCKET_QUANT_RESIDENT=auto` it
prefills faster than F16 (45.1 against 39.6 t/s at pp2048) and decodes 2.9x as fast, at the same
faithfulness. Residency holds an fp16 copy of the weights, 13.8 GB, which needs a board with that
RAM free. Without it, the 4.83 GiB GGUF fits an 8 GB board and prefills at the streamed rate.
Q8_0 (8.40 GiB) sits between them in size and decode rate.

### gpt-oss-20b (MXFP4, MoE)

gpt-oss-20b is OpenAI's `gpt-oss` 20B, arch `gpt-oss`:

- 24 layers
- 32 experts with 4 active per token, ~3.6 B active of 20.9 B
- GQA with 64/8 heads
- Alternating sliding-window (128) and full-attention layers
- A learned attention sink per head

The model is distributed only in native MXFP4, the 4-bit block-float it was trained to emit. The
GGUF comes from `ggml-org/gpt-oss-20b-GGUF` (11.27 GiB), and the raw data is in
[data/gpt-oss-20b.md](data/gpt-oss-20b.md).

In llama.cpp the expert FFNs run through `GGML_OP_MUL_MAT_ID`, and they are the bulk of prefill
FLOPs (~75%). The dense attention projections and `lm_head` are the rest.

**Prefill (prompt processing), t/s (MXFP4, `-b 2048 -ub 2048`).** The expert path is on by
default, behind a residency pre-flight. With the default placement, gpt-oss-20b prefills at
37.8 t/s at pp2048, 2.29x the CPU backend with llama.cpp's weight repack on. The default places
62 of its 72 expert stacks, every placed expert resident [HW sweep 2026-11-01, NPU clock 928 MHz,
the series of §"At a glance"].

Setting `ROCKET_MOE=1` also claims what the pre-flight declines. On one build, the configurations
compare as follows [HW sweep 2026-08-27, ~816 MHz, governor `performance`, llama.cpp 171974745]:

| test | CPU, unrepacked | NPU, `ROCKET_MOE=0` | **NPU, default** | NPU, `ROCKET_MOE=1` |
|---|---:|---:|---:|---:|
| pp512  | 12.17 | 14.08 (1.16x) | **22.0 (1.81x)** | 24.81 (2.04x) |
| pp2048 | 11.80 | 13.62 (1.15x) | **28.1 (2.38x)** | 34.01 (2.88x) |

> [!NOTE]
> The default configuration varies from run to run. On the 2026-08-27 build it measured
> 20.5-22.8 t/s at pp512 and 26.9-28.7 at pp2048, at identical placement (63 of 72 stacks, 1656
> experts, 0 streamed). Average several runs before comparing it with another configuration.

The expert path accounts for this model's result. On the 2026-08-27 build at pp2048, holding the
routed experts on the NPU as native int8 gives 2.06x the experts-on-CPU configuration. It gives
2.38x the unrepacked CPU. It is faster at every prefill length. CPU prefill is itself fast on this
model (~12 t/s, against ~7 for a dense 8B), because only ~3.6 B params are active per token. The
2.38x is therefore against a strong baseline.

By design, the default is slower than `ROCKET_MOE=1`. Over two sessions on 2026-08-27,
`ROCKET_MOE=1` measures 6-13% faster at pp512 and 18-21% faster at pp2048 [HW sweep]. The table's
own figures are the upper ends of those ranges. The pre-flight reserves a whole expert stack before
claiming its op. Reserving charges all `n_expert` experts, where the lazy admission it runs ahead of
charges only the ~82% the router exercises. The default therefore takes 63 of 72 stacks (100%
resident, 0 streamed), where `ROCKET_MOE=1` takes all 72.

On a board that does not fit the stack, the default stays faster than the experts-on-CPU baseline,
and `ROCKET_MOE=1` falls below it. At an induced 12 GB budget, `ROCKET_MOE=1` falls to 52% resident
and measures 13.63 t/s at pp512 against 14.08. The default's 30 fully-resident stacks measure 17.02
(1.21x) there. On a board with RAM to spare, raise `ROCKET_MOE_CACHE_MB` to recover the peak. It
keeps zero experts streamed, and `ROCKET_MOE=1` does not.

The pre-flight is designed on the premise that any streaming loses. Streaming does not lose at every
residency: the fastest configuration measured is 91% resident. The crossover lies between 52% and
82% resident, and has not been bracketed more tightly.

#### Micro-batch size

The headline table runs at `-b 2048 -ub 2048`, the setting for the expert path. At `-ub 512` the
expert path measures 22.2 t/s (21.7-22.7) at pp512 and 22.5 (22.4-22.8) at pp2048 [HW sweep
2026-08-27, ~816 MHz]. With the experts on the CPU it measures 14.13 and 13.53. The expert path is
therefore about 1.6x and 1.7x the experts-on-CPU configuration at the llama.cpp default
micro-batch. Against the `-ub 2048`
figures of 22.0 and 28.1, the smaller micro-batch costs ~20% at pp2048 and nothing at pp512.

Two mechanisms cause that cost. Neither is the per-expert dequantization, because the
native-quantization path ingests each expert once:

- The dense MXFP4 weights are not in the expert cache, and still re-dequantize per micro-batch. At
  `-ub 512` they are decoded four times as often.
- Each expert receives only `n_tokens · n_used / n_expert` rows, 64 at `-ub 512` against 256 at
  `-ub 2048`. The per-expert dispatch, gather, scatter and bucket padding around the GEMM stay
  constant.

The ~0.42x figure for MoE prefill at `-ub 512` belongs to the fp16 streaming path, whose
per-micro-batch dequantization a smaller `-ub` multiplies. The native path does not dequantize the
experts per micro-batch.

> [!NOTE]
> Compare gpt-oss configurations at one `-ub`. On this model `-ub 2048` slows the dense graph, and
> the CPU configuration does not depend on `-ub`, so the best `-ub` differs per configuration.

### gpt-oss-20b native-quant expert path and results

The native-quant expert path ingests each expert once into resident int8 codes, and its speedup
comes from that residency. The fp16 expert path (`ROCKET_MOE=1 ROCKET_MOE_NATIVE=0`) dequantizes
each quantized expert to fp16 on the host every micro-batch. It measures 4.59 at pp512 and 10.18 at
pp2048 [HW sweep 2026-07-14], worse than leaving the experts on the CPU. The CPU's fused quantized
kernel performs no dequantization. MXFP4 dequantizes correctly.

The cost of the per-micro-batch dequantization, and why it does not shrink with the row count, are
in [not-mac-bound.md](not-mac-bound.md) §"Quantized MoE experts and residency". That section also
shows that the int8 GEMM's own traffic is not the source of the gain.

**Conditions:** RK1 (RK3588), NPU clock ~816 MHz, gpt-oss-20b MXFP4.

The native-quant path requires residency, and its gain depends on nearly all the experts fitting.
At 99% resident it measures 17.57 at pp512 and 26.78 at pp2048 [HW sweep 2026-07-14]. At 82%
resident it measures 12.19 at pp512,
below the 14.11 of leaving the experts on the CPU [HW sweep 2026-07-14]. A streamed expert keeps
its per-micro-batch dequantization, while a resident one costs a GEMM that shrinks with `M`.

On this 31 GiB board gpt-oss reaches 99%, which is ~14 GB of int8 experts beside its 11.3 GiB GGUF.
The GGUF must stay mapped for CPU decode. A smaller board will not reach 99%, and the backend warns
when it falls short.

The ingest is a one-time cost per `llama_context`. It is lazy: an expert is ingested the first
time the router sends it a row. On gpt-oss it is 20.6-21.7 s for ~1575 experts, 13 ms each:
11.2-11.8 s of MXFP4->int8 decode and 9.4-9.9 s of NPU-BO pack. Measured 2026-10-09 on the RK1 at
~816 MHz, governor `ondemand`, `-p 512 -b 2048 -ub 2048` [HW sweep]. The pack runs ~1.3 GB/s and
grows with the bytes held resident, not with the expert count.

DeepSeek-V2-Lite packs 3.0x as many experts, holding ~14.9 GB, in 11.7-11.8 s. Its Q4_K->int8
decode beside that is 6.4-6.6 s, at `-p 2048 -ub 2048`. The benchmark tool `llama-bench` builds a
fresh context per test row, so it incurs the ingest per row, and a long-running host incurs it
once.

> [!NOTE]
> The `llama-bench` warm-up does not absorb all of the ingest. An expert the warm-up never routed
> to is ingested inside the first timed repetition that routes to it. A MoE prefill figure from
> `llama-bench` therefore includes part of the ingest.

The resident group-wise int8 readback runs as NEON. The 2026-08-27 figures in this section and in
§"gpt-oss-20b (MXFP4, MoE)" predate that readback and the faster pack. The 2026-11-01 figures run on a build that
has both.

Attention stays on the CPU on this model, as a correctness requirement. The gpt-oss model has an
attention sink, a learned per-head logit that joins the softmax denominator. The NPU FLASH_ATTN
handler has no sink term, so the offload is declined for such ops. Declining it is also +26% at
pp2048, because the accepted offload takes real time to compute the wrong answer (see
[attention-offload-crossover.md](attention-offload-crossover.md)).

> [!CAUTION]
> An accepted FLASH_ATTN offload on a model with attention sinks silently computes a sink-less
> softmax past the `n_kv` threshold of 1024. The handler drops the sinks (`src[4]`), and the model
> still produces fluent text and a plausible perplexity.

**Decode (generation), t/s (CPU-bound).**

| | MXFP4 |
|---|---|
| tg64, CPU backend | 9.11 |

**Conditions:** `taskset 0xf0 -t 4`, llama.cpp b11539 [HW sweep 2026-11-01].

Decode stays on the CPU, as for every model here. The MoE's small active-parameter count makes it
comparatively fast: 9.1 t/s, against Ministral-3-8B Q4_K_M's 4.7.

**Interactive.** On a 2048-token prompt the default configuration reaches the first token in
~54 s, from the 2026-11-01 pp2048 rate of 37.8 t/s [host-computed]. The repacked CPU backend
takes 2.29x as long. There is no quant option to raise the 9.1 t/s decode, since MXFP4 is the only
distribution.

**Faithfulness: per-matmul cosine and paired differential perplexity.**

`gpt-oss` is a harmony reasoning model, so absolute wikitext PPL is meaningless (its own F16
reference is ~385). Faithfulness is checked the way [MODEL-NOTES.md](../MODEL-NOTES.md) prescribes
for a reasoning model, on a real ~1400-token passage. The passage clears the FA `n_kv` threshold of
1024, so the check reaches the regime where a sink-less attention takes effect.

> [!NOTE]
> Differential perplexity, NPU prefill against CPU on the same text, does not detect a wrong
> attention function on this model. With the NPU FLASH_ATTN offload on the full-attention layers,
> it measures +1.0% against a ±5.7% per-run standard error, inside the noise. Validate with a
> per-matmul cosine and a paired differential perplexity, on a prompt past the FA `n_kv` threshold.
> A short prompt passes with such a defect present.

| test | result |
|---|---|
| per-matmul cosine vs an fp64 CPU reference, on real weights and real activations (one expert per `MUL_MAT_ID` op, rotating across every layer, projection and expert) | **mean 0.999821, min 0.998980** over 50 expert GEMMs under `ROCKET_MOE=1`, and **mean 0.999815, min 0.998976** over 55 on the default placement [2026-08-27]. The two placements are numerically the same |
| greedy match vs the CPU (`--temp 0`, same seed, 48 tokens) | No resolution (below) |
| synthetic expert-path test, outlier-channel activations | MXFP4 0.9999, Q4_K 0.9999, Q8_0 0.9999 |
| differential PPL on the default placement, DeepSeek-V2-Lite, `-c 2048` (`M_e` = 192, expert path active) | **−0.121% against the experts-on-CPU configuration**, 8 paired chunks, 1.20 se, indistinguishable from zero |

The per-matmul cosine is above the synthetic prediction (0.9994), and far above the 0.98 at which
int4 with a Hadamard rotation was token-identical. The int8 activations therefore survive the real
outlier-channel distribution with no Hadamard rotation.

Greedy token match against the CPU cannot rank configurations by faithfulness. A 0.9998 cosine still
flips an argmax wherever the top two logits are close, at a position nothing controls. Which
configuration diverges first is therefore a weighted coin, not a faithfulness measure. The greedy
comparison can still show whether the text stays coherent (it does, on every configuration). It can
also show whether divergence is early and large rather than early and small.

The cosine and the differential perplexity decide this test, and the differential perplexity must
be read paired. At 8 chunks the absolute error bar is ±2.5% and resolves nothing. Pairing on the
same chunks cancels the variance both configurations share and gives ~0.10% in PPL terms. To
isolate the expert path, difference the two NPU configurations, not the default against the CPU.
Both NPU configurations run the same dense fp16 offload. The dense path is itself −0.296% (3.74 se,
all eight chunks negative), which would otherwise be credited to the experts.

**Verdict.** The `MUL_MAT_ID` handler closes the op-coverage gap. With the experts held resident on
the NPU as native int8, it prefills at 2.29x the repacked CPU at pp2048 (2026-11-01). On the
2026-08-27 build it measured 2.38x at pp2048 and 1.81x at pp512 against the unrepacked CPU. It wins
at every length on this model, and eligibility on another model is per architecture
([moe-placement.md](moe-placement.md) §"The accept boundary"). The speedup comes from residency: each expert is converted to int8
once and held resident, which removes the per-micro-batch dequantization the fp16 path incurs.

`ROCKET_MOE` is **on by default** on builds from 2026-08-27. Without a pre-flight, the expert
path's sign depends on the host's memory, and the residency pre-flight removes that dependence. An
expert stack's resident cost is knowable from its tensor alone. The `supports_op` callback
therefore reserves the whole stack before the first ingest and leaves on the CPU what it cannot
reserve. The result is bounded below by the experts-on-CPU baseline at every budget.

The default has two costs [HW sweep]. The reservation is more conservative than the lazy admission
it precedes. On a board that nearly fits, the default therefore gives up 6-13% at pp512 and 18-21%
at pp2048 against `ROCKET_MOE=1`. Setting `ROCKET_MOE_CACHE_MB` recovers most of that without
giving up the sign guarantee. The default also needs the per-expert size thresholds: without them
it regresses DeepSeek-V2-Lite 27% at pp512 at 100% residency.

The recommended invocation on a 31 GiB board is the default, with `-b 2048 -ub 2048`. Expect the
one-time expert ingest, 20.6-21.7 s on gpt-oss, at the first prefill of each `llama_context`.

### MoE expert placement

The default places an expert stack on the NPU only where one expert dispatch performs at least
340 mega-MACs (`M_e · K · N`). It also requires the residency pre-flight to reserve the whole stack.
The threshold separates gpt-oss-20b's gains from DeepSeek-V2-Lite's loss at a similar per-expert row
count, and declines Qwen3-30B-A3B's experts at `-ub` 2048 and below. Forcing them onto the NPU there
measures 0.800x the experts-on-CPU configuration [HW sweep 2026-08-30, ~816 MHz]. On an
expert-dominated model the automatic budget places more stacks than the board can hold, and a
smaller `ROCKET_MOE_CACHE_MB` is faster. The accept-boundary map, the cost model and the budget
series are in [moe-placement.md](moe-placement.md).

### Qwen3.5-9B

Qwen3.5-9B is Alibaba's Qwen3.5 9B (arch `qwen35`, dense, GQA), with 8.95 B params. The GGUFs
come from `unsloth/Qwen3.5-9B-GGUF` (base model `Qwen/Qwen3.5-9B`), and the F16 is derived from the
published BF16. Three GGUFs are measured: F16, Q4_K_M and IQ4_XS, an importance-matrix 4-bit quant.

**Prefill (prompt processing), t/s (the NPU's job).** At pp2048 the F16 GGUF prefills at
36.1 t/s, 4.9x the CPU [HW sweep 2026-11-01, NPU clock 928 MHz, the default `-b` and `-ub`]. Held resident under `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, the Q4_K_M
GGUF prefills at 41.3 t/s, 3.9x the repacked CPU. At pp128 the F16 GGUF measures 3.3-3.4x the
CPU [HW sweep 2026-11-02].
The sweep below gives the rest of the prompt-length range and the quantized GGUFs:

| test | F16 CPU->NPU | Q4_K_M CPU->NPU | IQ4_XS CPU->NPU |
|---|---|---|---|
| pp512  | 7.25 -> **25.86** (3.6x) | 6.25 -> 16.68 (2.7x) | 7.44 -> 16.33 (2.2x) |
| pp1024 | 7.14 -> **25.14** (3.5x) | 6.21 -> 21.31 (3.4x) | 7.37 -> 20.87 (2.8x) |
| pp2048 | the 2026-11-01 figure above | 6.15 -> 22.88 (3.7x) | 7.29 -> 22.49 (3.1x) |

**Conditions:** NPU clock ~816 MHz, the flags and 8-thread CPU baseline of §"Method" [HW sweep
2026-07-02]. The quantized GGUFs stream at `-b 2048 -ub 2048`, against a CPU backend built
without the weight repack.

On that sweep the F16 rate was flat across pp512 and pp1024, where Ministral-3-8B's declined. The
quants' NPU rates rise with M (Q4_K_M 16.7->22.9, IQ4_XS 16.3->22.5). A quantized GGUF is
dequantized to fp16 again on every micro-batch. At pp512 (one 512-row batch) the dequantization is
a large fixed fraction of prefill time, and by pp2048 it amortizes.

The same mechanism sets the tuning for a quantized GGUF. Where its fp16 image fits,
`ROCKET_QUANT_RESIDENT=auto` at the default `-ub` removes the per-micro-batch dequantization. Where
it does not, `-b 2048 -ub 2048` amortizes it (the tuning table of §"At a glance",
[quant-prefill-microbatch.md](quant-prefill-microbatch.md)).

IQ4_XS prefill offloads on the same path as Q4_K_M, at 3.1x the CPU at pp2048. An importance matrix
is used only at quantization time. At inference IQ4_XS is an ordinary `ggml_is_quantized` type with
a `to_float` trait and 256-element super-blocks, so `K%32` holds. Its prefill GEMMs therefore
dequantize to fp16 and offload exactly as Q4_K_M's do, and the importance-matrix quants use the
full quantized-prefill path.

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).** Q4_K_M decodes at
4.00 t/s at `taskset 0xf0 -t 4` on llama.cpp b11539 [HW sweep 2026-11-01]. F16 decodes at 1.56 t/s and IQ4_XS at 4.49 t/s there [HW sweep 2026-11-02]. The July build's
8-thread pairing gives the other GGUFs:

| | F16 | Q4_K_M | IQ4_XS |
|---|---|---|---|
| tg64 CPU | 1.35 | 3.60 | 4.08 |
| tg64 NPU | 1.34 | 3.59 | 4.09 |

**Conditions:** 8 threads, unpinned [HW sweep 2026-07-02].

Decode with the NPU backend loaded matches the CPU, and runs on the CPU
([decode-gemv.md](decode-gemv.md)). Its rate scales with the bytes each token reads, so the smallest
quant decodes fastest. At 8 threads it measures F16 1.4 -> Q4_K_M 3.6 (5.28 GiB) -> IQ4_XS
4.1 t/s (4.80 GiB).

**Interactive.** TTFT and stream rate are derived from the 2026-07-02 pp2048 and tg64 rates
[host-computed].

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | F16 CPU | 289 s | 1.4 t/s | 437 s |
| | F16 + NPU | **82 s** | 1.4 t/s | 231 s |
| | Q4_K_M + NPU | 90 s | 3.6 t/s | 146 s |
| | IQ4_XS + NPU | 91 s | 4.1 t/s | **140 s** |
| chat, 128-tok prompt -> 400 out | F16 + NPU | 5 s | 1.4 t/s | 304 s |
| | Q4_K_M + NPU | 8 s | 3.6 t/s | 119 s |
| | IQ4_XS + NPU | 8 s | 4.1 t/s | **106 s** |

On a 2048-token prompt the NPU shortens the first-token wait ~3.5x (F16 289->82 s) and nearly
halves the turn. On a chat with a short prompt and a long reply, the turn is decode-bound. There the
NPU barely shortens it, and quantization does (F16 304 -> IQ4_XS 106 s). IQ4_XS gives the shortest
turn in both scenarios, with a TTFT on par with the other quants and the fastest stream.

**Faithfulness: differential perplexity, wikitext test, 12 chunks (same GGUF, NPU prefill vs
CPU).**

| | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| F16    | 9.0637 | 9.0683 | +0.05% |
| Q4_K_M | 9.2358 | 9.2294 | −0.07% |
| IQ4_XS | 9.3502 | 9.3411 | −0.10% |

The per-chunk stderr is ±0.43, so the NPU−CPU delta is 49-93x smaller than the noise
[host-computed]. The NPU prefill matches the CPU to within 0.10% on all three, so the
fp16-accumulation path is faithful for the importance-matrix quant as well. The PPL rise from F16 to
Q4_K_M to IQ4_XS (9.06 -> 9.24 -> 9.35) is the expected quantization cost, present equally on both
backends.

**Verdict.** For batch or RAG workloads on a board with 16 GB or more, F16 + NPU prefills at 36.1
t/s at pp2048, 4.9x the CPU. For most boards, use IQ4_XS + NPU. It fits an 8 GB board (4.80 GiB) and
had the fastest stream on the July build (4.1 t/s at 8 threads). It keeps ~3x NPU prefill on long
prompts at ~816 MHz, and is faithful (Δ −0.10%). Q4_K_M has marginally higher fidelity (PPL 9.23
against 9.35), at 5.28 GiB and a 3.6 t/s stream on the same July build. On b11539 Q4_K_M streams at
4.00 t/s at 4 threads, and IQ4_XS is not re-measured there.

### Phi-4-mini-instruct

Microsoft's Phi-4-mini-instruct (arch `phi3`, dense, GQA) has 3.84 B params. The GGUFs come from
`unsloth/Phi-4-mini-instruct-GGUF` (base model `microsoft/Phi-4-mini-instruct`), and the F16 is
derived from the published BF16.

**Prefill (prompt processing), t/s (the NPU's job).** At pp2048 the F16 GGUF prefills at
76.1 t/s, 5.0x the CPU [HW sweep 2026-11-01, NPU clock 928 MHz, the default `-b` and `-ub`]. Held resident under `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, the Q4_K_M
GGUF prefills at 87.4 t/s, 4.2x the repacked CPU. `ROCKET_F16_RESIDENT=auto` lifts the F16 GGUF
to 88.4 t/s, and at pp128 it measures 3.2x the CPU [HW sweep 2026-11-02].
The sweep below gives the rest of the prompt-length range and the quantized GGUFs:

| test | F16 CPU->NPU | Q8_0 CPU->NPU | Q4_K_M CPU->NPU |
|---|---|---|---|
| pp512  | 16.55 -> **55.44** (3.4x) | 15.20 -> 37.26 (2.5x) | 13.85 -> 36.70 (2.6x) |
| pp1024 | 15.96 -> **47.86** (3.0x) | 14.81 -> 38.77 (2.6x) | 13.55 -> 37.29 (2.8x) |
| pp2048 | the 2026-11-01 figure above | 13.90 -> 34.91 (2.5x) | 12.89 -> 33.42 (2.6x) |

**Conditions:** NPU clock ~816 MHz, the flags and 8-thread CPU baseline of §"Method" [HW sweep
2026-07-02]. The quantized GGUFs stream at `-b 2048 -ub 2048`, against a CPU backend built
without the weight repack.

On that sweep F16 NPU prefill reaches 55 t/s at pp512, and its speedup falls with prompt length
(3.4x->3.0x by pp1024) as the NPU rate declines 55->48. On a model this small the per-op readback
and dispatch are a larger fixed fraction, and they grow with M (more output tiles to read back).
The quants are flatter (~33-39 t/s), as their per-micro-batch dequantization amortizes.

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).** Q4_K_M decodes at
8.77 t/s at `taskset 0xf0 -t 4` on llama.cpp b11539 [HW sweep 2026-11-01]. F16 decodes at 3.34 t/s and Q8_0 at 6.19 t/s there [HW sweep 2026-11-02]. The July build's
8-thread pairing gives the other GGUFs:

| | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| tg64 CPU | 2.86 | 4.86 | 7.04 |
| tg64 NPU | 2.84 | 4.82 | 6.95 |

**Conditions:** 8 threads, unpinned [HW sweep 2026-07-02].

Decode with the NPU backend loaded matches the CPU, and runs on the CPU
([decode-gemv.md](decode-gemv.md)). At 3.84 B it decodes at F16 2.9 -> Q8_0 4.9 -> Q4_K_M 7.0 t/s
at 8 threads, an interactive stream from a 2.31 GiB file.

**Interactive.** TTFT and stream rate are derived from the 2026-07-02 pp2048 and tg64 rates
[host-computed].

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | F16 CPU | 136 s | 2.9 t/s | 206 s |
| | F16 + NPU | **51 s** | 2.8 t/s | 121 s |
| | Q8_0 + NPU | 59 s | 4.8 t/s | 100 s |
| | Q4_K_M + NPU | 61 s | 7.0 t/s | **90 s** |
| chat, 128-tok prompt -> 400 out | F16 + NPU | 2 s | 2.8 t/s | 143 s |
| | Q4_K_M + NPU | 3 s | 7.0 t/s | **61 s** |

The NPU shortens the long-prompt first-token wait ~2.6x (F16 136->51 s). On the short-prompt chat
the turn is decode-bound, so quantization shortens it (F16 143 -> Q4_K_M 61 s).

The `total turn` figures are optimistic by ~15-20%, uniformly across configurations. The 128 tokens
decoded after a 2048-token prompt read a full KV cache, and they stream slower than tg64 from an
empty one. The penalty is relatively larger on a small, fast model.

**Faithfulness: differential perplexity, wikitext test, 12 chunks (same GGUF, NPU prefill vs
CPU).**

| | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| F16    | 10.9284 | 10.9152 | −0.12% |
| Q4_K_M | 11.6977 | 11.5905 | −0.92% |

Both deltas are inside the reported ±0.5-0.6 PPL uncertainty. The F16 path matches the CPU to
−0.12%. The Q4_K_M's larger −0.92% reflects the NPU consuming the Q4_K weights through
dequantization to fp16, rather than through the CPU's native Q4_K kernel. The delta is in the NPU's
favor (lower PPL), so no accuracy is lost. The absolute PPL is high, ~11, because Phi-4 is trained
largely on curated and synthetic data and models raw wikitext poorly. As with the other instruct
models, only the NPU−CPU delta is informative, not the absolute value.

**Verdict.** Phi-4-mini suits a small or edge board. With 7 GB of RAM available, F16 + NPU
prefills at 76.1 t/s at pp2048, 5.0x the CPU. Q4_K_M + NPU is the practical choice: 2.31 GiB (fits
a 4 GB board), ~2.6x NPU prefill at ~816 MHz, and an interactive 8.8 t/s stream at 4 threads.

### Gemma-4-12B-it

Google's Gemma-4-12B-it (arch `gemma4`, dense, GQA) has 11.91 B params. The GGUFs come from
`unsloth/gemma-4-12b-it-GGUF` (base model `google/gemma-4-12b-it`), and the F16 is derived from the
published BF16.

**Prefill (prompt processing), t/s (the NPU's job).**

| test | F16 CPU->NPU | Q8_0 CPU->NPU | Q4_K_M CPU->NPU |
|---|---|---|---|
| pp512  | 4.98 -> **25.62** (5.1x) | 8.44 -> 26.45 (3.1x) | 6.84 -> 26.25 (3.8x) |
| pp1024 | 4.87 -> **25.31** (5.2x) | 8.14 -> 26.58 (3.3x) | 6.67 -> 25.38 (3.8x) |
| pp2048 | 4.72 -> **24.64** (5.2x) | 7.78 -> 24.75 (3.2x) | 6.42 -> 24.51 (3.8x) |

**Conditions:** NPU clock 928 MHz, the default `-b` and `-ub` [HW sweep 2026-11-01]. The quantized
GGUFs run under `ROCKET_QUANT_RESIDENT=auto`, against a CPU backend with llama.cpp's weight repack
on.

NPU prefill runs at 5.1-5.2x the CPU rate on F16 from pp512 to pp2048, against a 4.7-5.0 t/s CPU
baseline for this 12 B model. Under quantized residency the Q8_0 and Q4_K_M GGUFs prefill at about
the F16 rate at every length, with 92-95% of their weights resident (298-310 of 325). The rest
stream, because the RAM reserve or the NPU's IOVA window stops the admission. The F16 GGUF streams
its weights, because the default F16 residency admits only a weight with `K` <= 2048.

A board without RAM for the fp16 copy streams the quantized GGUF, at `-b 2048 -ub 2048`:

| test | Q8_0 CPU->NPU, streamed | Q4_K_M CPU->NPU, streamed |
|---|---|---|
| pp512  | 4.52 -> 11.41 (2.5x) | 4.23 -> 11.20 (2.6x) |
| pp1024 | 4.35 -> 13.53 (3.1x) | 4.15 -> 13.19 (3.2x) |
| pp2048 | 4.16 -> 13.43 (3.2x) | 4.02 -> 13.28 (3.3x) |

**Conditions:** NPU clock ~816 MHz, the flags and 8-thread CPU baseline of §"Method", against a
CPU backend built without the weight repack [HW sweep 2026-07-02].

Streamed, the quantized rates rise with M (Q4_K_M 11.2->13.3, 2.6x->3.3x). A streamed quantized
GGUF is dequantized to fp16 again on every micro-batch. At pp512 that dequantization is a large
fixed fraction, and by pp2048 it amortizes (the mechanism behind `-b 2048 -ub 2048`,
[quant-prefill-microbatch.md](quant-prefill-microbatch.md)).

**Decode (generation), t/s (CPU-bound, LPDDR-bandwidth-limited).**

| | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| tg64, CPU backend | 1.07 | 1.97 | 3.05 |

**Conditions:** `taskset 0xf0 -t 4`, llama.cpp b11539 [HW sweep 2026-11-01].

Decode runs on the CPU in both configurations ([decode-gemv.md](decode-gemv.md)). At 12 B, F16
decode runs at 1.07 t/s, because every token streams 22 GiB of weights from LPDDR. Decode scales
with the bytes each token reads: F16 1.07 -> Q8_0 1.97 -> Q4_K_M 3.05 t/s (2.9x). For interactive
use of this model, use a quantized GGUF.

**Interactive.** TTFT and stream rate are computed from the rates above [host-computed]. The TTFT
is the prompt over the pp2048 rate, and the total adds 200 tokens at the decode rate.

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | F16 CPU | 434 s | 1.07 t/s | 621 s |
| | F16 + NPU | **83 s** | 1.07 t/s | 270 s |
| | Q4_K_M + NPU, `ROCKET_QUANT_RESIDENT=auto` | 84 s | 3.05 t/s | **149 s** |

On a 2048-token prompt the NPU shortens the first-token wait 5.2x (F16 434->83 s). The stream
dominates the turn on this model: with NPU prefill, F16 still needs 270 s for a 200-token reply. A
400-token chat answer takes 374 s at F16 and 131 s at Q4_K_M. Q4_K_M + NPU is the best interactive
configuration, with the NPU's short TTFT and Q4_K_M's fastest stream.

**Faithfulness: differential perplexity, wikitext test, 12 chunks (same GGUF, NPU prefill vs CPU).**

| | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| F16    | 630.9975 | 625.8672 | −0.81% |
| Q8_0   | 639.9257 | 652.1588 | +1.91% |
| Q4_K_M | 670.6969 | 675.4451 | +0.71% |

**Conditions:** NPU clock ~816 MHz [HW sweep 2026-07-02].

The per-run stderr is ±~65 (~±10%). All three NPU−CPU deltas (−0.8%, +1.9%, +0.7%) are well inside
it, so the fp16-accumulation prefill path is faithful across F16 and both quants. The absolute PPL
(~630+) is not meaningful. Gemma-4-12B-it is instruction- and reasoning-tuned, and scores raw
wikitext very poorly. As with the other instruct models, only the NPU−CPU delta is informative.

**Verdict.** Q4_K_M + NPU is the practical choice. It decodes at 2.9x the F16 rate, and under
`ROCKET_QUANT_RESIDENT=auto` it prefills at about the F16 rate (24.5 t/s at pp2048), which gives
the shortest turn. Its 6.86 GiB GGUF fits an 8 GB board, where it streams its weights and prefills
at the streamed rate. For batch or RAG work on a 32 GB board (the F16 GGUF is 22 GiB), F16 + NPU
prefills at 24.6 t/s at pp2048, 5.2x the CPU. Its 1.07 t/s decode makes long replies slow on
either backend. Q8_0 (11.78 GiB) sits between them in size and decode rate.

### Qwen3.6-27B (hybrid Gated-DeltaNet, Q4_K_M)

Alibaba's Qwen3.6 27B (arch `qwen35`, 65 blocks, 27.32 B params) is a hybrid model. It interleaves
Gated-DeltaNet linear-attention / SSM-scan layers with dense GQA attention (GGUF SSM metadata: state
128, conv kernel 4, 16 groups, inner size 6144). The GGUF comes from `unsloth/Qwen3.6-27B-MTP-GGUF`.
Only Q4_K_M (15.92 GiB) is measured. At 27 B the F16 GGUF (~54 GB) and Q8_0 (~29 GB) do not fit the
31 GB board.

**Prefill (prompt processing), t/s (the NPU's job).** At pp2048, Q4_K_M at `-b 2048 -ub 2048`
prefills at 11.5 t/s [HW sweep 2026-11-01, NPU clock 928 MHz]. A CPU backend with llama.cpp's weight
repack on runs at 3.0 t/s, and the NPU measures 3.8x it. The sweep below gives the shorter prompts:

| test | Q4_K_M CPU->NPU |
|---|---|
| pp512  | 1.79 -> 5.38 (3.0x) |
| pp1024 | 1.79 -> 6.78 (3.8x) |

**Conditions:** NPU clock ~816 MHz, the flags and 8-thread CPU baseline of §"Method", against a CPU
backend built without the weight repack [HW sweep 2026-07-03].

On that sweep the NPU rate climbs with M as the per-micro-batch dequantization amortizes (the
`-b 2048 -ub 2048` mechanism, [quant-prefill-microbatch.md](quant-prefill-microbatch.md)). The
unrepacked CPU baseline is ~1.8 t/s, a 27 B model streamed through 8 CPU cores.

The Gated-DeltaNet and SSM-scan layers have no NPU handler, and run on the CPU. They are a small
share of the prefill FLOPs. The FFN and attention projections dominate and offload as ordinary
`MUL_MAT`, so the hybrid layers do not reduce the prefill speedup. A mixture-of-experts model
differs: its expert FFNs go through `MUL_MAT_ID`, whose NPU path holds each quantized expert
resident as native int8. The path is on by default, and places an expert op only where the
conditions listed in §"LLM and multimodal (llama.cpp via ggml-rocket)" all hold.

**Decode (generation), t/s (CPU-bound, LPDDR-bandwidth-limited).**

| | Q4_K_M |
|---|---|
| tg64, CPU backend | 1.30 |

**Conditions:** `taskset 0xf0 -t 4`, llama.cpp b11539 [HW sweep 2026-11-01].

Decode runs on the CPU in both configurations ([decode-gemv.md](decode-gemv.md)). At 27 B it runs
at 1.3 t/s, because every token streams the 16 GiB of Q4_K_M weights from LPDDR. There is no
smaller quant here to raise the stream rate, so this model suits prefill and batch work rather
than chat.

**Interactive.** TTFT and stream rate are derived from the 2026-07-03 pp2048 and 8-thread tg64
rates at ~816 MHz [host-computed].

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | Q4_K_M CPU | 1151 s | 1.1 t/s | 1333 s |
| | Q4_K_M + NPU | **263 s** | 1.1 t/s | **443 s** |
| chat, 128-tok prompt -> 400 out | Q4_K_M CPU | 72 s | 1.1 t/s | 436 s |
| | Q4_K_M + NPU | 24 s | 1.1 t/s | 384 s |

On a 2048-token prompt the NPU shortens the first-token wait 4.4x (1151->263 s, ~19 min -> ~4.5 min)
and the whole RAG turn ~3x (1333->443 s). On the short-prompt chat the turn is decode-bound: 400
tokens at 1.1 t/s take ~6 min on either backend. With no smaller quant to raise the stream rate, the
NPU shortens only the small TTFT (436->384 s). On this model the NPU's gain is in prefill, for a long
prompt with a short answer.

**Faithfulness: differential perplexity, wikitext test, 12 chunks (same GGUF, NPU prefill vs CPU).**

| | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| Q4_K_M | 7.2188 | 7.1997 | −0.26% |

The per-chunk stderr is ±0.32, so the NPU−CPU delta is ~17x smaller than the noise. Qwen3.6-27B
scores a normal wikitext PPL (~7.2), unlike the instruction- and reasoning-tuned models in this
record, whose raw-wikitext PPL is inflated and not meaningful. The NPU−CPU delta (−0.26%, in the
NPU's favor) is well inside the per-chunk stderr. The fp16-accumulation prefill path is therefore
faithful on the hybrid model too.

**Verdict.** On Qwen3.6-27B the NPU prefills at 3.8x the repacked CPU at pp2048. At the 2026-11-01
rates a 2048-token prompt reaches its first token in ~3 min on the NPU, against ~11 min on the CPU
[host-computed]. On a 31 GB board the model suits prefill and RAG work rather than interactive
chat. Only Q4_K_M fits, decode is ~1.3 t/s (a 400-token reply takes ~5 min), and there is no
smaller quant to raise the stream rate.

Use it for long-prompt, short-answer work (summarize, extract, RAG), where the first-token speedup
applies. Where replies must stream at conversational speed, use a smaller model (Gemma-4-12B or
Qwen3.5-9B at Q4_K_M). Linear-attention and SSM hybrids offload their prefill, and the DeltaNet
layers stay on the CPU.

### Llama-3.2-3B-Instruct

Meta's Llama-3.2-3B-Instruct (arch `llama`, dense, GQA) has 3.21 B params, and is the field's
standard reference point. The GGUFs come from `unsloth/Llama-3.2-3B-Instruct-GGUF`, and the F16 is
unsloth's published F16 (derived from the BF16).

**Prefill (prompt processing), t/s (the NPU's job).** At pp2048 the F16 GGUF prefills at
85.1 t/s, 5.0x the CPU [HW sweep 2026-11-01, NPU clock 928 MHz, the default `-b` and `-ub`]. Held resident under `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, the Q4_K_M
GGUF prefills at 92.2 t/s, 3.9x the repacked CPU. `ROCKET_F16_RESIDENT=auto` lifts the F16 GGUF
to 97.8 t/s, and at pp128 it measures 3.1-3.3x the CPU [HW sweep 2026-11-02].
The sweep below gives the rest of the prompt-length range and the quantized GGUFs:

| test | F16 CPU->NPU | Q8_0 CPU->NPU | Q4_K_M CPU->NPU |
|---|---|---|---|
| pp512  | 18.23 -> **61.79** (3.4x) | 17.05 -> 34.19 (2.0x) | 16.71 -> 33.60 (2.0x) |
| pp1024 | 17.74 -> **52.91** (3.0x) | 16.48 -> 36.31 (2.2x) | 16.34 -> 36.66 (2.2x) |
| pp2048 | the 2026-11-01 figure above | 15.72 -> 34.11 (2.2x) | 15.49 -> 34.46 (2.2x) |

**Conditions:** NPU clock ~816 MHz, the flags and 8-thread CPU baseline of §"Method" [HW sweep
2026-07-03]. The quantized GGUFs stream at `-b 2048 -ub 2048`, against a CPU backend built
without the weight repack.

On that sweep F16 NPU prefill reaches 61.79 t/s at pp512, above Phi-4-mini's 55.44, because this
smaller model has fewer FLOPs per matmul. As with Phi-4-mini, the F16 speedup falls with prompt
length (3.4x->3.0x by pp1024). On a small model the per-op readback and dispatch are a larger fixed
fraction, and they grow with M.

The quants measure a lower speedup (~2.0-2.2x) than Phi-4-mini's ~2.5-2.6x because of the CPU
denominator, rather than a weaker NPU. Llama-3.2-3B's CPU quant prefill is ~16-17 t/s, close to its
own F16 CPU rate of 17-18. Its NPU quant rate (~34-36 t/s) is in line with Phi-4-mini's. The quants'
rates rise from pp512 to pp1024 and are flat at pp2048, as the per-micro-batch dequantization
amortizes (the `-b 2048 -ub 2048` mechanism,
[quant-prefill-microbatch.md](quant-prefill-microbatch.md)).

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).** Q4_K_M decodes at
11.09 t/s at `taskset 0xf0 -t 4` on llama.cpp b11539 [HW sweep 2026-11-01]. F16 decodes at 3.98 t/s and Q8_0 at 7.37 t/s there [HW sweep 2026-11-02]. The July build's
8-thread pairing gives the other GGUFs:

| | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| tg64 CPU | 3.15 | 5.62 | 7.88 |
| tg64 NPU | 3.27 | 5.47 | 7.93 |

**Conditions:** 8 threads, unpinned [HW sweep 2026-07-03].

Decode with the NPU backend loaded matches the CPU, and runs on the CPU
([decode-gemv.md](decode-gemv.md)). At 3.21 B it decodes at F16 3.3 -> Q8_0 5.5 -> Q4_K_M 7.9 t/s
at 8 threads, an interactive stream from a 1.87 GiB file.

**Interactive.** TTFT and stream rate are derived from the 2026-07-03 pp2048 and tg64 rates
[host-computed].

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | F16 CPU | 118 s | 3.2 t/s | 182 s |
| | F16 + NPU | **45 s** | 3.3 t/s | 106 s |
| | Q8_0 + NPU | 60 s | 5.5 t/s | 97 s |
| | Q4_K_M + NPU | 59 s | 7.9 t/s | **84 s** |
| chat, 128-tok prompt -> 400 out | F16 + NPU | 3 s | 3.3 t/s | 125 s |
| | Q4_K_M + NPU | 4 s | 7.9 t/s | **54 s** |

The NPU shortens the long-prompt first-token wait ~2.6x (F16 118->45 s). On the short-prompt chat
the turn is decode-bound, so quantization shortens it (F16 125 -> Q4_K_M 54 s).

The `total turn` figures are optimistic, uniformly across configurations. The 128 tokens decoded
after a 2048-token prompt read a full KV cache, and they stream slower than tg64 from an empty one.
The penalty is relatively larger on a small, fast model.

**Faithfulness: differential perplexity, wikitext test, 12 chunks (same GGUF, NPU prefill vs
CPU).**

| | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| F16    | 12.6656 | 12.6682 | +0.02% |
| Q4_K_M | 12.9554 | 12.8978 | −0.44% |

Both deltas are well inside the reported ±0.65 PPL uncertainty, and the F16 NPU−CPU delta is ~250x
smaller than the per-chunk stderr [host-computed]. The F16 path matches the CPU to +0.02%. The
Q4_K_M delta (−0.44%, in the NPU's favor) reflects the NPU consuming the Q4_K weights through
dequantization to fp16, rather than through the CPU's native Q4_K kernel. No accuracy is lost. The
absolute PPL is high, ~12.7, because Llama-3.2-3B-Instruct is instruction-tuned and models raw
wikitext poorly. As with the other instruct models, only the NPU−CPU delta is informative, not the
absolute value.

**Verdict.** Llama-3.2-3B's F16 NPU prefill is 85.1 t/s at pp2048, 5.0x the CPU, and needs ~6 GB of
RAM free. Q4_K_M + NPU is the practical choice: 1.87 GiB (fits a 4 GB board with headroom), ~2.2x
NPU prefill streamed at ~816 MHz, and an 11.1 t/s stream at 4 threads. It gives the shorter
effective turn in both scenarios.

On the ~816 MHz sweep the quantized NPU/CPU prefill ratio is lower because the CPU's quantized
baseline is comparatively fast, not because the NPU is weaker.

### Ministral-3-3B-Instruct-2512

Mistral's Ministral-3-3B-Instruct-2512 (arch `mistral3`, dense, GQA) has 3.43 B params. It is the 3B
sibling of Ministral-3-8B (§"Ministral-3-8B-Instruct-2512"). The GGUFs come from
`unsloth/Ministral-3-3B-Instruct-2512-GGUF`, and the F16 is derived from the published BF16.

**Prefill (prompt processing), t/s (the NPU's job).** At pp2048 the F16 GGUF prefills at
75.0 t/s, 4.7x the CPU [HW sweep 2026-11-01, NPU clock 928 MHz, the default `-b` and `-ub`]. Held resident under `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, the Q4_K_M
GGUF prefills at 81.1 t/s, 3.7-3.8x the repacked CPU. `ROCKET_F16_RESIDENT=auto` lifts the F16
GGUF to 86.1 t/s, and at pp128 it measures 3.2x the CPU [HW sweep 2026-11-02].
The sweep below gives the rest of the prompt-length range and the quantized GGUFs:

| test | F16 CPU->NPU | Q8_0 CPU->NPU | Q4_K_M CPU->NPU |
|---|---|---|---|
| pp512  | 18.02 -> **57.62** (3.2x) | 16.16 -> 32.94 (2.0x) | 15.55 -> 32.96 (2.1x) |
| pp1024 | 17.39 -> **48.57** (2.8x) | 15.54 -> 35.14 (2.3x) | 15.10 -> 34.67 (2.3x) |
| pp2048 | the 2026-11-01 figure above | 14.50 -> 30.78 (2.1x) | 14.32 -> 30.54 (2.1x) |

**Conditions:** NPU clock ~816 MHz, the flags and 8-thread CPU baseline of §"Method" [HW sweep
2026-07-03]. The quantized GGUFs stream at `-b 2048 -ub 2048`, against a CPU backend built
without the weight repack.

On that sweep F16 NPU prefill reaches 57.6 t/s at pp512, between Llama-3.2-3B (61.8) and
Phi-4-mini (55.4). The three ~3-4 B dense models run fewer FLOPs per matmul than the larger
models. As on the other two, the F16 speedup falls with prompt length (3.2x->2.8x by pp1024). A
small model's per-op readback is a larger fixed fraction and grows with M.

The quants measure ~2.0-2.3x, at an NPU rate of ~31-35 t/s. Their rates rise and then flatten with
M as the per-micro-batch dequantization amortizes (the `-b 2048 -ub 2048` mechanism,
[quant-prefill-microbatch.md](quant-prefill-microbatch.md)). The lower ratio comes from the strong
CPU quant baseline (~14-16 t/s) rather than a weaker NPU, as on Llama-3.2-3B.

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).** Q4_K_M decodes at
10.47 t/s at `taskset 0xf0 -t 4` on llama.cpp b11539 [HW sweep 2026-11-01]. F16 decodes at 3.70 t/s and Q8_0 at 6.86 t/s there [HW sweep 2026-11-02]. The July build's
8-thread pairing gives the other GGUFs:

| | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| tg64 CPU | 3.01 | 5.39 | 7.59 |
| tg64 NPU | 3.00 | 5.53 | 7.66 |

**Conditions:** 8 threads, unpinned [HW sweep 2026-07-03].

Decode with the NPU backend loaded matches the CPU, and runs on the CPU
([decode-gemv.md](decode-gemv.md)). At 3.43 B it decodes at F16 3.0 -> Q8_0 5.5 -> Q4_K_M 7.7 t/s
at 8 threads, from a 1.99 GiB file, against Llama-3.2-3B's 7.9 t/s from 1.87 GiB.

**Interactive.** TTFT and stream rate are derived from the 2026-07-03 pp2048 and tg64 rates
[host-computed].

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | F16 CPU | 126 s | 3.0 t/s | 192 s |
| | F16 + NPU | **51 s** | 3.0 t/s | 118 s |
| | Q8_0 + NPU | 67 s | 5.5 t/s | 103 s |
| | Q4_K_M + NPU | 67 s | 7.7 t/s | **93 s** |
| chat, 128-tok prompt -> 400 out | F16 + NPU | 3 s | 3.0 t/s | 136 s |
| | Q4_K_M + NPU | 4 s | 7.7 t/s | **56 s** |

The NPU shortens the long-prompt first-token wait ~2.5x (F16 126->51 s). On the short-prompt chat
the turn is decode-bound, so quantization shortens it (F16 136 -> Q4_K_M 56 s).

The `total turn` figures are optimistic, uniformly across configurations. The 128 tokens decoded
after a 2048-token prompt read a full KV cache, and they stream slower than tg64 from an empty one.

**Faithfulness: differential perplexity, wikitext test, 12 chunks (same GGUF, NPU prefill vs
CPU).**

| | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| F16    | 9.9014 | 9.8979 | −0.04% |
| Q4_K_M | 10.1022 | 10.0918 | −0.10% |

Both deltas are well inside the ±0.46 per-chunk stderr, and the F16 NPU−CPU delta is ~130x smaller
than it.
The NPU prefill matches the CPU to −0.04% on F16 and −0.10% on Q4_K_M, both in the NPU's favor, so
the fp16-accumulation path is faithful. The absolute PPL (~9.9) is higher than the 8B sibling's
~6.9, which is the small model's quality cost rather than an NPU effect. Because Ministral-3-3B is an
instruct model, only the NPU−CPU delta is informative.

**Verdict.** Ministral-3-3B suits a small or edge board, with an F16 NPU prefill of 75.0 t/s at
pp2048 (4.7x), which needs ~7 GB of RAM free. Q4_K_M + NPU is the practical choice: 1.99 GiB (fits
a 4 GB board), ~2.1x NPU prefill streamed at ~816 MHz, and a 10.5 t/s stream at 4 threads. It
gives the shorter effective turn in both scenarios.

Among the small dense models at pp2048, Ministral-3-3B's F16 prefill (75.0 t/s) sits below
Llama-3.2-3B's (85.1) and level with Phi-4-mini's (76.1). Its 4-thread Q4_K_M stream (10.5 t/s)
sits between Llama-3.2-3B's (11.1) and Phi-4-mini's (8.8).

### DeepSeek-V2-Lite (MLA + MoE, Q4_K_M)

DeepSeek's V2-Lite (arch `deepseek2`, 27 layers, 15.71 B params, ~2.4 B active per token) is a
mixture-of-experts model: 64 routed experts and 2 shared experts, with 6 routed experts active per
token and block 0 dense. Its attention is Multi-head Latent Attention (MLA). The KV is compressed to
a `kv_lora_rank=512` latent, and the per-head key/value dims are asymmetric: key_length 192 (128
nope + 64 rope), value_length 128. The GGUF comes from `mradermacher/DeepSeek-V2-Lite-GGUF` (the
base model `deepseek-ai/DeepSeek-V2-Lite`, 9.65 GiB). Only Q4_K_M is measured, because at 15.71 B
the F16 GGUF (~31 GB) does not fit the 31 GB board.

With the default placement and every routed expert resident, DeepSeek-V2-Lite prefills at 43.4 t/s
at pp2048 [HW sweep 2026-11-01, NPU clock 928 MHz, `-b 2048 -ub 2048`]. Against the CPU backend with
llama.cpp's weight repack on, in the series of §"At a glance", it measures 1.83x. The tables below
are earlier builds and state their own conditions.

**Conditions:** RK3588, NPU clock ~816 MHz, Q4_K_M, the flags and 8-thread CPU baseline of
§"Method", measured 2026-07-03, with the MoE offload re-measured 2026-07-04 [HW sweep]. A table
measured under other conditions states them.

MLA attention and the routed experts each have an op-level handler. The FLASH_ATTN handler accepts
MLA's asymmetric key and value dims (DK=192 ≠ DV=128) rather than requiring `DK==DV`. The MLA FA
primitive is therefore ready, and bit-faithful (cos=1.000000). In the 2026-07-03 runs the backend's
FA path did not engage on this model under `-fa auto` or `-fa 1`, so attention ran on the CPU. The
path is dispatch-bound (large DK, few KV heads) and gains only at long context.

The routed experts go through `GGML_OP_MUL_MAT_ID`. The backend places them on the NPU by default
where the conditions of §"LLM and multimodal (llama.cpp via ggml-rocket)" hold. Where they run
therefore depends on the expert path and the prefill length (below).

With the routed experts on the CPU, the NPU runs the large MLA projections (q and kv down/up), the 2
always-on shared experts' gate/up/down, and `lm_head`. All are ordinary static-weight `MUL_MAT`.
These dense GEMMs are much larger than gpt-oss's GQA projections, and gpt-oss has no shared expert.
The NPU's prefill speedup is therefore modest, and larger than gpt-oss's.

**Prefill (prompt processing), t/s (Q4_K_M, the only precision).**

| test | CPU -> NPU |
|---|---|
| pp512  | 20.37 -> 24.04 (1.18x) |
| pp1024 | 19.92 -> 24.85 (1.25x) |
| pp2048 | 19.01 -> **23.87 (1.26x)** |

**Conditions:** routed experts on the CPU, llama.cpp b9568, 2026-07-03.

The NPU prefill holds a flat ~24 t/s across the curve while the CPU baseline declines with M
(20.4->19.0). The speedup therefore rises from 1.18x to 1.26x.

DeepSeek and gpt-oss were also measured in one session, with their experts on the CPU, at
llama.cpp b10558 and `-ub 2048` on 2026-08-27. DeepSeek's speedup there is 1.33x at pp512 and
1.17x at pp2048, and gpt-oss's 1.16x at both [host-computed from the per-model records]. The margin at pp2048 is about 1%.
The difference is the MLA projections and the two shared experts, substantial dense GEMMs that
offload.

It is far below the dense models' 3x+ on the same builds. Attention (MLA, on the CPU here) and the
routed experts are the bulk of the graph, and in this configuration they stay on the CPU. CPU
prefill is ~19-20 t/s for a 16 B model, because only ~2.4 B params are active per token.

On the fp16 streaming path, `ROCKET_MOE=1 ROCKET_MOE_NATIVE=0`, the `MUL_MAT_ID` handler lowers
prefill below the CPU rate. The loss is larger than gpt-oss's. DeepSeek's CPU is faster and its
experts-on-CPU NPU configuration already runs faster than the CPU, so there is more to give up.

| test | CPU | NPU (fp16 expert path) |
|---|---|---|
| pp512  | 20.40 | 5.05 (0.25x) |
| pp1024 | 19.93 | 7.82 (0.39x) |
| pp2048 | 18.99 | 11.30 (0.59x) |

**Conditions:** `-ub 2048`, the CPU column measured in the same session, 2026-07-04.

The loss is dequant-bound, the same cause as on gpt-oss. The host dequantizes each of the 64 routed
experts' Q4_K weights to fp16 on every micro-batch, amortized over only `M_e ~ n_tokens·6/64` rows.
DeepSeek's routed experts are smaller (K=2048, N=1408) and more numerous than gpt-oss's. The
per-expert dequantization and dispatch overhead is therefore a larger share, and the loss is larger.

On the native-quant path, the default, the expert offload's effect depends on prefill length
[HW sweep 2026-08-27].

| test | CPU | NPU, `ROCKET_MOE=0` | NPU, experts on the NPU |
|---|---:|---:|---:|
| pp512  | 19.54 | **26.04** | 19.09 (**0.73x the experts-on-CPU configuration**) |
| pp2048 | 18.50 | 21.63 | **27.9 (1.29x)** |

**Conditions:** RK3588, NPU clock ~816 MHz, governor `performance`, llama.cpp b10558, 2026-08-27.
The CPU column is unrepacked.

**Not characterized:** what moves the experts-on-CPU configuration between the 2026-07-03 table and
this one: 24.04 to 26.04 t/s at pp512, and 23.87 to 21.63 at pp2048.

Across the session's runs the pp2048 ratio measured 1.28-1.31x, the accept-boundary map's 1.31x
among them.

At pp512 the expert offload is a 27% regression at 100% residency, with 0 experts streamed. Each
expert receives `n_tokens · n_used / n_expert` rows, and the row count is an architecture property
rather than a prompt one. DeepSeek selects 6 of its 64 experts per token, so 512 tokens give each
expert ~48 rows. The count is below the 64-row granule the GEMM is padded up to, and the padded rows
are computed and read back in full.

The gpt-oss-20b model selects 4 of its 32 experts per token, so the same token count gives each
expert 64 rows, at the granule. Its offload gains. At pp2048 each DeepSeek expert receives
`M_e` ~192 rows, above the granule, and the offload gains 1.29x (the table above).

The backend's default placement therefore requires two size thresholds, beside the residency
pre-flight and the 512-token minimum micro-batch. One is `M_e >= ROCKET_MOE_M_BUCKET`, the 64-row
granule. The other is a work per dispatch `M_e · K · N` of at least `ROCKET_MOE_MIN_WORK`,
340 mega-MACs ([moe-placement.md](moe-placement.md) §"The accept boundary").

On DeepSeek the work threshold admits the experts from `M_e` = 118, past ~1250 tokens in a
micro-batch. The default keeps DeepSeek's experts on the CPU at short prefill and places them on
the NPU at long prefill. One model is therefore placed differently by prompt length. At pp512
(`M_e` = 48) the row threshold declines the experts, and DeepSeek measures 25.9 on the device, the
experts-on-CPU rate.

> [!NOTE]
> A residency check alone does not detect this failure. `MUL_MAT_ID` placement has two independent
> failure modes, and a residency check sees only one of them. A residency check therefore does not
> show that the gpt-oss-20b result transfers to another MoE model.

**Decode (generation), t/s (CPU-bound, LPDDR-bandwidth-limited).**

| | Q4_K_M |
|---|---|
| tg64, CPU backend | 13.52 |

**Conditions:** `taskset 0xf0 -t 4`, llama.cpp b11539 [HW sweep 2026-11-01].

Decode runs on the CPU in both configurations ([decode-gemv.md](decode-gemv.md)). The MoE's small
active-param count keeps decode at 13.5 t/s from a 9.65 GiB file, above every dense model here
but SmolVLM2's 1.8 B.

**Interactive.** TTFT and stream rate are derived from the 2026-07-03 experts-on-CPU pp2048 rate
and the 8-thread tg64 rate of that build, 7.7 t/s [host-computed].

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | Q4_K_M CPU | 108 s | 7.7 t/s | 134 s |
| | Q4_K_M + NPU | **86 s** | 7.7 t/s | **112 s** |
| chat, 128-tok prompt -> 400 out | Q4_K_M CPU | 6 s | 7.7 t/s | 58 s |
| | Q4_K_M + NPU | 5 s | 7.7 t/s | 57 s |

On a 2048-token prompt the NPU shortens the first-token wait ~1.26x (108->86 s). TTFT is the only
term the NPU shortens, because there is no smaller quant and decode is CPU-bound on both backends.
On the short-prompt chat the turn is decode-bound, and the NPU barely shortens it (58->57 s).

The RAG `total turn` figures are optimistic, and the chat turns (short prompt) are accurate. The 128
tokens decoded after a 2048-token prompt stream at only ~3.7 t/s on both backends [host-computed],
about half the tg64 rate from an empty cache. MLA decode cost grows with the filled latent KV cache.

**Faithfulness: differential perplexity, wikitext test, 12 chunks (same GGUF, NPU prefill vs CPU).**

| config | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| experts on the CPU (the default then) | 8.2444 | 8.2232 | −0.26% |
| `ROCKET_MOE=1` (routed experts -> NPU, fp16 expert path) | 8.2444 | 8.2205 | −0.29% |

**Conditions:** `-c 512`, the July 2026 builds. Their default kept the routed experts on the CPU,
and `ROCKET_MOE=1` ran the experts on the fp16 expert path.

Both NPU−CPU deltas are well inside the per-chunk stderr of ±0.38. The dense offload and the
fp16 `MUL_MAT_ID` expert offload (−0.29%) are therefore both faithful. The handler's cost is in
speed, not accuracy, which matches its unit-test cosine (cos = 1.000000). DeepSeek-V2-Lite is a base
model, unlike the instruct and reasoning models in this record, and scores a normal wikitext PPL
(~8.2).

The 2026-08-27 default was also measured at `-c 2048`, where `M_e` = 192 and the native-quant
expert path is active. Its paired differential perplexity over 8 chunks is −0.42% against the CPU
(4.01 se) [HW sweep 2026-08-27]. The table in §"At a glance" quotes that figure. Against the
experts-on-CPU configuration the same run is −0.12%, indistinguishable from zero (the faithfulness
table of §"gpt-oss-20b native-quant expert path and results").

**Verdict.** DeepSeek-V2-Lite combines MLA attention and MoE in one model. Both have op-level
handlers, and the MoE expert offload is a default gain **at long prefill only**. The FLASH_ATTN
handler is bit-faithful for MLA's head dims. Its DeepSeek case has not been validated end to end
on the device.

The routed experts take the native-quant path by default at pp2048: 43.4 t/s, 1.83x the repacked CPU
(2026-11-01). On the 2026-08-27 build they measured 1.29x the experts-on-CPU configuration. Below
the per-dispatch work threshold they stay on the CPU. The fp16 expert path loses more here than on
gpt-oss (0.25x->0.59x the CPU) and is never taken by default.

Below the threshold, the MLA projections, the 2 shared experts and `lm_head` offload for a modest,
faithful 1.18-1.26x prefill (b9568, 2026-07-03). Measured on one build with gpt-oss, the speedup
is larger than gpt-oss's at pp512 and about the same at pp2048. It is below the dense models' 3x+.

### Phi-4 (14B)

Phi-4 is Microsoft's 14.66 B-parameter dense model, the 14 B sibling of the Phi-4-mini above, with a
different architecture. In llama.cpp the 14 B Phi-4 maps to the `llama` arch (a Llama-style dense
GQA transformer, shown as "llama 13B" in the rows), where Phi-4-mini is `phi3`. At 14.66 B the F16
GGUF (29.3 GB) does not fit the board. Q8_0 (14.51 GiB) is therefore the highest precision that
runs, as for the 27 B Qwen3.6 above, one size class down. In size the model lies between Gemma-4-12B
and Qwen3.6-27B [HW sweep 2026-07-04].

**Conditions:** RK3588 on the RK1 (31 GB, 29 GB free, no swap), NPU clock ~816 MHz, the flags in
§"Method", 2026-07-04. Q8_0 and Q4_K_M GGUFs from `unsloth/phi-4-GGUF` (base model
`microsoft/phi-4`, arch `llama`).

**Prefill (prompt processing), t/s (the NPU's job).** At pp2048, Q4_K_M at `-b 2048 -ub 2048`
prefills at 21.9 t/s [HW sweep 2026-11-01, NPU clock 928 MHz]. A CPU backend with llama.cpp's weight
repack on runs at 5.47 t/s, and the NPU measures 4.0x it. The sweep below gives Q8_0 and the shorter
prompts:

| test | Q8_0 CPU->NPU | Q4_K_M CPU->NPU |
|---|---|---|
| pp512  | 3.83 -> 10.61 (2.8x) | 3.53 -> 10.31 (2.9x) |
| pp1024 | 3.70 -> 12.13 (3.3x) | 3.49 -> 11.93 (3.4x) |
| pp2048 | 3.60 -> **12.20 (3.4x)** | the 2026-11-01 figure above |

**Conditions:** NPU clock ~816 MHz, `-b 2048 -ub 2048`, against a CPU backend built without the
weight repack [HW sweep 2026-07-04].

On that sweep NPU prefill runs at up to 3.4x the CPU rate, against a slow (~3.5 t/s) 14 B CPU
baseline. With no F16 that fits, both GGUFs are quantized, and both show the quantized-GGUF
pattern. The speedup rises with prompt length (Q8_0 2.8x->3.4x by pp2048, Q4_K_M 2.9x->3.4x by
pp1024) as
the per-micro-batch dequantization amortizes across the `-b 2048 -ub 2048` window
([quant-prefill-microbatch.md](quant-prefill-microbatch.md)). At pp1024 the two measure ~12 t/s,
so neither quantized GGUF prefills faster than the other ([not-mac-bound.md](not-mac-bound.md)).

Because its F16 GGUF does not fit, this model cannot run the F16 configuration that gives the dense
rows of §"At a glance" their 4.2-5.8x. On this board a quantized GGUF is the only option.

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).** Q4_K_M decodes at
2.74 t/s at `taskset 0xf0 -t 4` on llama.cpp b11539 [HW sweep 2026-11-01]. Q8_0 decodes at 1.70 t/s there [HW sweep 2026-11-02]. The July build's
8-thread pairing gives Q8_0:

| | Q8_0 | Q4_K_M |
|---|---|---|
| tg64 CPU | 1.43 | 2.17 |
| tg64 NPU | 1.42 | 2.19 |

**Conditions:** 8 threads, unpinned [HW sweep 2026-07-04].

Decode runs on the CPU in both configurations ([decode-gemv.md](decode-gemv.md)), so the two rates
match. At 14.66 B and Q8_0 it is slow (1.4 t/s at 8 threads), with 14.5 GiB streamed from LPDDR
per token. Q4_K_M nearly doubles it at 8.28 GiB a token. Interactive use of this model needs a
quantized GGUF in practice.

**Interactive.** TTFT and streaming are derived from the 2026-07-04 pp2048 and tg64 rates
[host-computed].

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | Q8_0 CPU | 569 s | 1.4 t/s | 709 s |
| | Q8_0 + NPU | **168 s** | 1.4 t/s | 309 s |
| | Q4_K_M + NPU | 174 s | 2.2 t/s | **265 s** |
| chat, 128-tok prompt -> 400 out | Q8_0 + NPU | 12 s | 1.4 t/s | 294 s |
| | Q4_K_M + NPU | 12 s | 2.2 t/s | **195 s** |

On a long prompt the NPU cuts the first-token wait 3.4x (Q8_0 569->168 s, ~9.5 min -> ~2.8 min).
This model's decode is slow enough that the stream dominates a long reply: even with NPU prefill,
Q8_0 needs ~309 s for a 200-token answer. Q4_K_M shortens that turn to 265 s, and the chat turn
from 294 to 195 s.

The `total turn` figures are optimistic by ~10-15%. Tokens decoded after a 2048-token prompt read
a full KV cache, so they stream slower than tg64 from an empty cache. The best interactive
configuration is NPU prefill (short TTFT) with Q4_K_M (fastest stream).

**Faithfulness: differential perplexity, wikitext test, 12 chunks (same GGUF, NPU prefill vs CPU).**

| | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| Q8_0   | 6.6391 | 6.6330 | −0.09% |
| Q4_K_M | 6.7688 | 6.7479 | −0.31% |

Both NPU−CPU deltas (−0.09%, −0.31%) are far inside the per-run stderr (±0.29 PPL, ~±4%), and both
are lower on the NPU. The fp16-accumulation prefill path is therefore faithful for both Q8_0 and
Q4_K_M, with no accuracy lost to the NPU matmul. Phi-4 (14B) scores raw wikitext at a normal
absolute PPL (~6.6-6.8), as a near-base model does. Here the absolute value is therefore informative
as well as the delta. The instruct and reasoning models in this record score it higher (Gemma-4-12B
~630, Phi-4-mini ~11).

**Verdict.** Phi-4 (14B) is a mid-large model for an edge board with ~9 GB of RAM free, and a
quality step up from Phi-4-mini. F16 does not fit a 31 GB board, so use Q4_K_M + NPU. At 8.28 GiB
it prefills at 21.9 t/s at pp2048 (4.0x the repacked CPU) and streams at 2.7 t/s at 4 threads.
Where accuracy matters more than stream speed (1.4 against 2.2 t/s at 8 threads on the July
build), Q8_0 + NPU (14.51 GiB) is the near-lossless option, at essentially the same prefill on the
~816 MHz sweep. On this model the NPU's gain is in prefill.

### SmolVLM2-2.2B-Instruct (vision-language, mtmd)

SmolVLM2-2.2B is a vision-language model: a SigLIP-SO400M vision encoder feeds a SmolLM2-class
language model, run end to end through llama.cpp's `mtmd` path. The language model (arch `llama`)
has 1.81 B parameters, and "2.2B" is the combined vision and language count. The model offloads
both a vision encoder and an LLM prefill, through two different paths that are measured
separately [HW sweep 2026-07-04].

**Conditions:** RK3588 on the RK1, NPU clock ~816 MHz, 2026-07-04. GGUFs from
`ggml-org/SmolVLM2-2.2B-Instruct-GGUF`: the language model as F16, Q8_0 and Q4_K_M, and the vision
tower as an fp16 `mmproj` GGUF.

**Vision encoder (SigLIP-SO400M), clip encode latency, warm.** The vision tower runs through
`clip.cpp`, whose ggml graph is scheduled over a `[backend, CPU]` pair.

> [!NOTE]
> Set `MTMD_BACKEND_DEVICE=ROCKET` to run the vision encoder on the NPU. The automatic device
> selection in clip probes only GPU and IGPU device types, and the NPU is an ACCEL device. Without
> the selector the vision encoder silently stays on the CPU.

| encoder (hidden / layers / tokens) | graph splits | CPU | NPU | speedup |
|---|---|---|---|---|
| SigLIP-SO400M (1152 / 27 / 729) | 1 -> **272** | 7.90 s | **6.66 s** | **1.19x** |

The NPU encodes ~1.19x faster warm (6.66 against 7.90 s), at the small-encoder end of the Whisper
crossover (the ASR section's tiny.en is 1.18x). The speedup is **warm-only**: a cold single encode
measures 8.12 s (~1.03x slower than the CPU), because the NPU clock parks at idle.

The graph splits into 272 CPU↔NPU segments (against 1 on the CPU alone), and the handoffs limit
the speedup. The ggml-rocket backend offloads only static-weight GEMMs that meet the K%32 and N%16
alignment. Per SigLIP layer, that admits only the q/k/v/o projections and fc1. Three kinds of op
stay on the CPU, and each offloaded GEMM has a handoff on either side:

- The fc2 down-projection (K=4304, 4304%32=16) fails the alignment rule.
- The attention QK^T/score·V matmuls are excluded by design, because their src0 is a computed
  tensor rather than a weight.
- Every norm, softmax, GELU and patch-embed op is a type ggml-rocket does not implement.

The handoffs are the cost of the generic ggml drop-in, which offloads op by op through the stock
clip graph. The `rocket_siglip_encoder` native driver instead runs the whole SigLIP-B/16 encoder
resident on the NPU, at far higher efficiency.

**Prefill (prompt processing), t/s (the LLM half, the NPU's job).** At pp2048 the F16 language
model prefills at 117.8 t/s, 4.2x the CPU [HW sweep 2026-11-01, NPU clock 928 MHz, the default
`-b` and `-ub`]. The default F16 residency holds 130 of its 142 weights resident. Held resident
under `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, the Q4_K_M GGUF prefills at 125.1 t/s,
3.2x the repacked CPU. At pp128 the F16 GGUF measures 2.7x the CPU [HW sweep 2026-11-02]. The
sweep below gives the rest of the prompt-length range and the quantized GGUFs:

| test | F16 CPU->NPU | Q8_0 CPU->NPU | Q4_K_M CPU->NPU |
|---|---|---|---|
| pp512  | 27.97 -> **98.98** (3.5x) | 26.78 -> 47.93 (1.8x) | 27.01 -> 48.57 (1.8x) |
| pp1024 | 29.46 -> **71.99** (2.4x) | 27.29 -> 47.11 (1.7x) | 26.25 -> 47.43 (1.8x) |
| pp2048 | the 2026-11-01 figure above | 23.81 -> 39.90 (1.7x) | 24.61 -> 40.56 (1.6x) |

**Conditions:** NPU clock ~816 MHz, the quantized GGUFs streamed at `-b 2048 -ub 2048` against a CPU
backend built without the weight repack [HW sweep 2026-07-04].

On that sweep F16 measures 98.98 t/s at pp512, against Llama-3.2-3B's 61.79. At 1.81 B this is the
smallest LLM here, with the fewest FLOPs per matmul. The F16 speedup falls steeply with prompt
length (3.5x->2.4x by pp1024). The small models show this pattern (Llama-3.2-3B, Phi-4-mini), and
this one most strongly: per-op readback and dispatch are a large fixed fraction, and grow with M.

The quantized GGUFs measure a lower multiple (~1.6-1.8x) than the ~2x of the other small dense
models. The cause is the CPU baseline rather than a slower NPU. This model's CPU quantized prefill
is fast (~24-27 t/s, near its F16 CPU rate), while the NPU's quantized rate (~40-48 t/s) is in line
with the others.

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).** Q4_K_M decodes at
20.70 t/s at `taskset 0xf0 -t 4` on llama.cpp b11539 [HW sweep 2026-11-01]. F16 decodes at 7.32 t/s and Q8_0 at 13.44 t/s there [HW sweep 2026-11-02]. The July build's
8-thread pairing gives the other GGUFs:

| | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| tg64 CPU | 5.11 | 9.60 | 14.66 |
| tg64 NPU | 5.19 | 9.83 | 14.54 |

**Conditions:** 8 threads, unpinned [HW sweep 2026-07-04].

Decode runs on the CPU in both configurations ([decode-gemv.md](decode-gemv.md)), so the two rates
match. At 1.81 B it measures F16 5.1 -> Q8_0 9.7 -> Q4_K_M 14.5 t/s at 8 threads, the last from a
1.03 GiB file. The Q4_K_M tg64 spread is ±3.3 t/s, run-to-run variation at this decode rate on a
small model, and NPU 14.54 against CPU 14.66 is inside it.

**Interactive.** TTFT and streaming are derived from the 2026-07-04 pp2048 and tg64 rates
[host-computed].

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | F16 CPU | 75 s | 5.1 t/s | 114 s |
| | F16 + NPU | **39 s** | 5.2 t/s | 78 s |
| | Q8_0 + NPU | 51 s | 9.8 t/s | 72 s |
| | Q4_K_M + NPU | 50 s | 14.5 t/s | **64 s** |
| chat, 128-tok prompt -> 400 out | F16 + NPU | 1 s | 5.2 t/s | 78 s |
| | Q4_K_M + NPU | 3 s | 14.5 t/s | **30 s** |

The NPU nearly halves the long-prompt first-token wait (F16 75->39 s). The short-prompt chat turn is
decode-bound, and quantization shortens it (F16 78 -> Q4_K_M 30 s).

**Faithfulness.** The language and vision halves are checked separately.

*LLM, differential perplexity (wikitext test, 12 chunks, same GGUF, NPU prefill vs CPU):*

| | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| F16    | 12.1622 | 12.1540 | −0.07% |
| Q4_K_M | 12.7622 | 12.6711 | −0.71% |

Both deltas are far inside the ±0.60-0.63 PPL stderr, and the F16 delta is ~75x smaller than its
stderr. The language half is faithful on the NPU. The absolute ~12.2 is inflated, because an
instruction-tuned model scores raw wikitext, so only the delta is informative.

*Vision, projected image-embedding cosine (vision on NPU vs CPU):* the 81 projected image tokens
that clip passes to the language model match at cosine 0.99998 (per-token minimum 0.99967). The
comparable figures are 0.999998 for the native SigLIP encoder and 0.9998 for Whisper. The greedy
captions differ in wording, and both read the same NYT moon-landing front page correctly.
Autoregressive decode amplifies the ~0.5% embedding perturbation into a different but equivalent
token path. As with Whisper, the encoder's metric is the cosine, not token identity.

**Verdict.** SmolVLM2-2.2B runs end to end on the FOSS NPU stack, vision encoder and language model
both. As the smallest model here it measures 117.8 t/s F16 prefill at pp2048 (4.2x) and a
20.7 t/s Q4_K_M stream at 4 threads. Use Q4_K_M: a 1.03 GiB language model plus a ~0.8 GB fp16
mmproj fits any board, with ~1.7x NPU prefill streamed at ~816 MHz and 20.7 t/s decode.

On the vision half the SigLIP encoder offload works and is faithful (cosine 0.99998), at ~1.19x.
The SO400M dimensions (fc2 K=4304, off the %32 grid) and the static-weight-only offload rule leave
the attention core, fc2 and every norm and activation op on the CPU. The graph takes 272
handoffs. The mtmd frontend has no resident vision-encoder path of the `rocket_siglip_encoder`
kind. The drop-in clip path is faithful and modestly faster than the CPU.

## ASR (whisper.cpp via ggml-rocket)

Whisper runs through the `ggml-rocket` drop-in `.so` on stock whisper.cpp, with no fork. The NPU
runs the encoder, once per 30 s audio window: a stack of self-attention and FFN blocks, all large-M
`MUL_MAT`. The decoder is autoregressive (M=1 GEMV per token) and runs on the CPU in both
configurations, as LLM decode does ([decode-gemv.md](decode-gemv.md)). The figures below are
encoder latency, CPU against NPU, across the model sizes. Faithfulness is the transcript (WER) and
the encoder-output cosine [HW sweep 2026-07-04].

**Conditions:** RK3588 on the RK1, NPU clock ~816 MHz, stock whisper.cpp with ggml-rocket, the
encoder timed alone by `whisper-bench -w 0`, 2026-07-04. The base.en row was measured
2026-09-26, with the CPU governor at `performance` and both configurations under
`taskset -c 4-7 -t 4`, flash attention on.

**Encoder latency, ms (the NPU's job).**

| model | encoder d_model / layers | CPU -> NPU |
|---|---|---|
| tiny.en        | 384 / 4  |   697.7 ->   591.9 (1.18x) |
| base.en        | 512 / 6  |  1513.5 ->  1015.8 (1.49x) |
| small.en       | 768 / 12 |  5711.5 ->  3718.9 (1.54x) |
| medium.en      | 1024 / 24 | 19268.2 -> 10444.9 (1.84x) |
| large-v3       | 1280 / 32 | 36399.9 -> 17013.5 (**2.14x**) |
| large-v3-turbo | 1280 / 32 | 32957.2 -> 15544.1 (**2.12x**) |

The NPU is faster at every size, and the speedup grows monotonically with the encoder: 1.18x at tiny
to 2.1x at large. The matmul work grows as d_model², while the CPU-side host packing and readback
grow ~linearly, so the NPU gains as the encoder grows
([encodings/whisper-encoder.md](../encodings/whisper-encoder.md)). Fixed per-op host cost,
interleaved with the CPU-only conv/LayerNorm/softmax front end, dominates a small encoder
(tiny). A large encoder is dominated by matmuls, where the NPU gains most.

At the 20 s window that `whisper-cli -ac 1000` sets, base.en encodes in 619.0 ms on the NPU
against 883.6 ms on the CPU, 1.43x [HW sweep 2026-09-26].

Without KACC, and with an fd opened per call, base.en is slower on the NPU than on the CPU.

The large-v3-turbo model keeps large-v3's full 32-layer encoder and replaces its 32-layer decoder
with a 4-layer one. Its NPU-accelerated encoder is therefore a much larger share of a
transcription. In the same NPU run, decode costs 19 ms/step for turbo against 116 ms/step for
large-v3 (~6x lower), while the encoders take nearly the same time (15.5 s against 17.0 s on the
NPU). The encoder's ~2.1x NPU speedup covers proportionally more of turbo's whole-pipeline time
than of large-v3's. Turbo is the model where NPU encoder offload matters most.

**Faithfulness: transcript agreement and encoder cosine.**

| | jfk.wav transcript (greedy) |
|---|---|
| base.en CPU vs NPU | **byte-identical** (WER 0) |
| large-v3-turbo CPU vs NPU | **byte-identical** (WER 0) |

The NPU encoder's output matches whisper.cpp's base.en encoder at cosine 0.9998
([encodings/whisper-encoder.md](../encodings/whisper-encoder.md)), and that fp16-accumulation
fidelity produces the identical transcript. In a separate validation, the full encoder (LN /
attention / softmax / GELU / residual) is byte-identical on the NPU across tiny/base/small/medium.

> [!CAUTION]
> In whisper.cpp the weight tensors have no names, so ggml names them `leaf_%d` by graph position.
> One `leaf_N` string then names different weights across whisper's separate
> conv/encode/cross/decode graphs. A resident-weight cache keyed on the tensor name serves one
> weight's packed tiles for another's matmul, and the encode is fast and its output garbage. The
> whisper-bench tool times the encode without checking it, so check the transcript.

In ggml-rocket, `rocket_weight_key` rejects the ggml-default `leaf_`/`node_` names, so those weights
stream per call instead of being cached. For a single-pass encoder that is correct and
timing-neutral: the figures above are the same with or without the check. The weights in llama.cpp
have real names (`blk.N.*`) and never match, so LLM resident caching and prefill speed are
unchanged. Llama-3.2-3B F16 pp512 stays at ~61 t/s, within run-to-run noise of its section's
61.79 at ~816 MHz.

**Verdict.** Whisper encoder offload is faithful to the CPU, and its speedup grows with model size.
It is ~1.2x at tiny, ~1.5x at base and small, and ~2.1x at large-v3 and large-v3-turbo. Its
transcripts are byte-identical to the CPU's. Stock `whisper-bench` and `whisper-cli` load the
`.so` through `GGML_BACKEND_PATH`, with no fork. The raw data and full per-rep timings are in
[data/whisper-encoder.md](data/whisper-encoder.md).

## ASR, multi-model (transcribe.cpp via ggml-rocket)

The `ggml-rocket` `.so` loads with no fork into transcribe.cpp (handy-computer, MIT), a ggml-based
host for 16+ ASR families (Whisper, Parakeet, Canary, Granite-Speech, MOSS diarize, Voxtral,
SenseVoice, FunASR). The ggml that transcribe.cpp vendors has a `GGML_BACKEND_API_VERSION` that
matches this backend, and the NPU registers as an ACCEL device. Each model's runner logs
`using accel backend: ROCKET` and offloads the encoder. The results cover two clips: `jfk.wav`
(11 s, clean) and a hard 120 s two-speaker conversational recording [HW sweep 2026-07-21].

**Conditions:** the process on the A76 cores (`taskset -c 4-7 --threads 4`), Q8_0, `ROCKET_KACC=1`,
warm, a fresh single process per run (no session reuse), 2026-07-21.

**Not characterized:** the NPU clock of these runs.

### The offload taxonomy

Encoders offload, and autoregressive decode mostly does not. Two rules set every ASR model's NPU
speedup:

- **Encode always offloads**, and its speedup scales with encoder size × audio length: 1.25-1.82x
  on the 120 s clip, and 1.00-1.80x on jfk. A large encoder (Voxtral's Whisper-large-v3, MOSS's
  Whisper-Medium) gains more than the small SAN-M encoder (SenseVoice/FunASR), which gains
  nothing on jfk.
- **Autoregressive decode is M=1 GEMV and mostly stays on the CPU** ([decode-gemv.md](decode-gemv.md)).
  The exception is the batched decode-prefill over the injected audio context on long audio, a
  matmul that offloads once it is large enough.

Decode offload varies with the decoder:

| Decoder | Decode offload | Why |
|---|---|---|
| Cross-attention decoder (Granite-Speech) | Best, ~1.9x on the long clip | Every step's cross-attn over the encoder output is a large-K matmul |
| Big decoder-only, long audio (Voxtral 3B) | 1.46x on the long clip, 1.00x on jfk | The 1500-token audio prefill through 3B is a large offloadable chunk. On jfk the prefill is too short |
| Small decoder-only (MOSS/FunASR 0.6B) | 1.00x (MOSS) to 1.12x (FunASR) | The prefill is a small, dequant-bound slice |
| Single-pass CTC (SenseVoice) | No decode at all | 48 ms even on 120 s |

The NPU speedup is therefore ~ (encoder fraction × encoder offload) + (prefill fraction × prefill
offload). Decode-bound models with a small decoder barely move, and encode-heavy or large-prefill
models gain more.

### Speed: jfk (11 s, clean)

| Model | arch | CPU | NPU | NPU speedup | encode C->N | decode C->N |
|---|---|--:|--:|--:|--|--|
| SenseVoice-small 234M | SAN-M + CTC (single-pass) | 1367 ms | 1369 ms | 1.00x | 1324->1326 (1.00x) | 5->5 ms |
| Fun-ASR-nano 800M | SAN-M enc + Qwen3-0.6B AR | 4772 ms | 4739 ms | 1.01x | 1237->1239 (1.00x) | 3497->3462 (1.01x) |
| MOSS 0.9B | Whisper-Med + Qwen3-0.6B AR | 21198 ms | 15514 ms | 1.37x | 14876->9166 (1.62x) | 6273->6300 (1.00x) |
| Voxtral-mini 3B | Whisper-lg-v3 + Ministral-3B AR | 62133 ms | 49193 ms | 1.26x | 29350->16306 (1.80x) | 32726->32832 (1.00x) |

On jfk no decode offloads, because the audio context is too short to make a large prefill. The NPU
speedup is the encoder's alone, and it follows the encoder's share of runtime. SenseVoice and FunASR
have a tiny encoder or are decode-dominated, and measure ~1.0x. MOSS and Voxtral have large encoders
and measure ~1.3x.

### Speed: 120 s conversational clip

| Model | CPU | NPU | NPU speedup | rt (NPU) | encode C->N | decode C->N |
|---|--:|--:|--:|--:|--|--|
| SenseVoice-small 234M | 24.37 s | 19.41 s | 1.26x | 6.2x | 23904->18939 (1.26x) | 48->48 ms |
| Fun-ASR-nano 800M | 78.75 s | 68.42 s | 1.15x | 1.75x | 22843->18350 (1.25x) | 55490->49656 (1.12x) |
| MOSS 0.9B (diarize) | 306.21 s | 284.18 s | 1.08x | 0.42x | 60626->37029 (1.64x) | 245395->246969 (1.00x) |
| Voxtral-mini 3B | 340.48 s | 217.55 s | 1.57x | 0.55x | 117329->64396 (1.82x) | 222944->152947 (1.46x) |
| Granite-Speech-2B (base) | 114.5 s | 68.2 s | 1.68x | 1.76x | n/a | ~1.9x (cross-attn) |

Longer audio raises every speedup: the encoder matmuls are larger, and for Voxtral and Granite the
decode-prefill also offloads. Voxtral's 1.57x on the long clip is the largest among the models
without cross-attention, because its 3B decode-prefill over 1500 audio tokens offloads. Parakeet
(FastConformer CTC-0.6B, F16, A76-pinned) also runs, at ~4.6x realtime on the long clip and 1.37x
the CPU. Its convolution modules stay on the CPU, because ggml-rocket has no conv path, and they
limit its speedup.

### Quality and capability: the 120 s hard clip

| Model | transcript accuracy | diarization | timestamps | translation | CPU==NPU |
|---|---|---|---|---|---|
| SenseVoice-small | worst: garbled run-on (fed past its 30 s window) | no | no | no | ~ (tiny CTC drift) |
| Fun-ASR-nano | semi-fluent, garbles hard names | no | no | no | ~ (tiny AR drift) |
| MOSS | good, coherent | **yes: 33 seg, clean two-speaker** | **yes, fine** | no | no (32 vs 33 seg, both good) |
| Voxtral-mini 3B | **best: cleanest, correct names, complete** | no | no | **yes** | **yes (char-identical)** |
| Granite-Speech-2B base | clean but drops ~40 s of the middle | plus variant: 8 coarse turns | no | no | base F16: yes (char-identical). Base Q8: one word. Plus: no (loops on the CPU) |

- **Voxtral** is the best transcriber. It alone recovers every proper name and transcribes the full
  120 s without dropping spans, but emits flat text (no speaker turns, no timestamps). It is the
  only model here that translates (de->en and ja->en, both accurate and char-identical CPU vs NPU).
- **MOSS** is the best diarizer by a wide margin: 33 fine segments with timestamps, cleanly
  splitting the two speakers. Its output is coherent and does not degenerate.
- **Granite-Speech** comes in three variants, of which base is the fastest, with the cleanest
  text. At F16 it is char-identical CPU==NPU on both clips [HW sweep 2026-10-12]. At Q8 it is
  char-identical on jfk and differs by one word on the hard clip. There the CPU transcript departs
  from the 2026-07-21 run's, and the NPU's matches it. The NAR variant is a non-autoregressive editor
  that runs iterative refinement, so its decode is *larger* than base's: ~1.4x slower and rougher.
  The plus variant adds diarization: 8 coarse speaker turns, no timestamps, coarser than MOSS.

On the hard clip, offloading the encoder does not by itself keep the output CPU==NPU (the table's
last column). Voxtral stays char-identical CPU vs NPU throughout. SenseVoice and FunASR drift
slightly, and MOSS emits 32 segments on one backend and 33 on the other, both good. A model whose
decoder cross-attention offloads (Granite) is not guaranteed CPU==NPU either: the fp16
cross-attention accumulation perturbs the logits, and greedy decoding amplifies the perturbation.
On the hard clip Granite-plus falls into a repetition loop on the CPU and stays coherent on the NPU.
The divergence follows the model's own decode instability rather than an NPU fault.

### MOSS on the NPU

MOSS-Transcribe-Diarize (0.9B: Whisper-Medium encoder -> 4x temporal merge -> VQ adaptor ->
Qwen3-0.6B decoder, audio injected as KV tokens) is decode-bound, and the NPU barely changes its
runtime. On the CPU, the long clip at Q8_0 takes 60.5 s to encode and 245.5 s to decode (80% of
runtime). Decode splits into prefill, 23.4 s (1638-token batched audio and prompt, 9.5%), and the
713 AR steps × 311 ms = 221.6 s (90.5%).

On the NPU, encode falls 60.5->36.9 s (1.64x, offloads) and prefill 23.4->18.8 s (1.25x, partial,
bound by dequantization). The 713 M=1 steps do not offload (313 ms NPU ~311 ms CPU), and decode as
a whole measures 245.4 against 247.0 s, 1.00x. MOSS injects audio as KV tokens, unlike Granite's
cross-attention, so generation is M=1 whatever the length of the 1500-token context. Its prefill
offloads, and the one-token-at-a-time generation (90% of decode) cannot.

The options measured give little. A sweep across quantizations moves decode ~1.4%, and Q8_0 stays
the default because it keeps the best WER at half F16's RAM. The `--spec-k-drafts` flag has no
effect, because MOSS does not advertise `supports_spec_decode`. The threads are already on the A76
cores. MOSS has no low-cost NPU optimization, because the M=1 autoregressive decode cannot be
reduced on the NPU.

The NPU's remaining headroom on MOSS is the prefill's dequantization. Raising the prefill from
~1.25x to ~2x would save ~7 s of the NPU's 284 s [expected]. That takes the pipeline from 1.08x
to ~1.10x the CPU, ~1.03x the current NPU configuration. The fixes that remain are
algorithmic (speculative decode, a smaller decoder, fewer emitted tokens). Against an A76-pinned
CPU baseline (306 s), MOSS on the NPU is 1.08x, and an untuned CPU baseline overstates that ratio.

### Build and run

transcribe.cpp needs its CPU backend and the rocket `.so` both discoverable, and one upstream
one-liner for shared/DL builds:

```sh
# build transcribe.cpp shared + DL-capable, tuned for the A76 (armv8.2 + dotprod + fp16)
cmake -B build -DTRANSCRIBE_BUILD_SHARED=ON -DTRANSCRIBE_GGML_BACKEND_DL=ON \
  -DTRANSCRIBE_VULKAN=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16
cmake --build build -j
cp build/bin/libggml-cpu.so build/src/   # backend scan globs build/src; the ARM cpu module lands in build/bin
# run with the rocket backend loaded
LD_LIBRARY_PATH=build/ggml/src:build/bin GGML_BACKEND_PATH=/path/to/libggml-rocket.so \
  sudo -n -E ./build/bin/transcribe-cli -m /path/model.gguf -f audio.wav
```

Two traps apply to this build:

- A `TRANSCRIBE_GGML_BACKEND_DL` build forces `GGML_NATIVE=OFF`. Without
  `-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`, the CPU baseline has no dotprod/fp16 kernels. It
  then runs ~1.3x too slow, which inflates the NPU speedup.
- `transcribe-cli` initializes the default backends only for `--list-devices`, and not for
  transcription. A one-line `main.cpp` patch that calls `transcribe_init_backends_default()` before
  model load fixes the shared/DL build. The default static build compiles the CPU backend in and
  does not need it.

### Verdict

The `ggml-rocket` drop-in generalizes from Whisper to the wider set of ASR models. Ordered from
fastest to most capable, they are SenseVoice (fastest, lowest quality) -> FunASR -> Granite-Speech
-> Voxtral (best transcript, only translator) / MOSS (best diarizer, slowest).

Q8_0 is the default for every model. For ASR it ties or beats F16 on both CPU and NPU, at half the
RAM. These models are less dominated by NPU matmuls than LLM prefill, so the per-micro-batch
dequantization costs little. The CPU-side decode is memory-bound, and Q8 moves half the bytes.

The NPU accelerates the encoder and, on long audio, the offloadable decode-prefill. The
autoregressive decode that dominates the audio LLMs stays on the CPU, as LLM decode does.

## Detection (tflite-rocket)

Detection runs quantized TFLite detectors (SSD-MobileDet, SSD MobileNet v2, EfficientDet-Lite0)
through the tflite-rocket external delegate on `librocketnpu`. A single inference is bound by the
host's cube scatter and gather rather than by compute or submits, so its latency follows the host
path. The NPU's larger gain is throughput in a multi-camera pool, the configuration Frigate runs.
Accuracy is COCO mAP rather than perplexity.

**Conditions:** RK3588, `native_int8=1`, batch 1. Each table states its NPU clock.

### Single-stream latency

Single-stream latency is set by host work. The measured host path gives these warm latencies, with
COCO mAP at parity with the CPU [HW sweep 2026-10-29]:

| Model | Warm latency | COCO mAP, NPU | COCO mAP, CPU |
|---|---:|---:|---:|
| MobileDet | 51.4 ms | 0.3325 | 0.3301 |
| SSD MobileNet v2 | 47.1 ms | 0.2634 | 0.2631 |
| EfficientDet-Lite0 | 94.5 ms | 0.2810 | 0.2823 |

**Conditions:** RK1, NPU clock 928 MHz (the 900 MHz request), governor `performance`, the process
on the A76 cores, 2026-10-29. Each latency is the median of 30 warm invokes on one image. mAP is
over 500 COCO val2017 images. The change table in the Evidence section's
[Detection (tflite-rocket)](#detection-tflite-rocket-1) records how the host path came down.

On `ondemand` with a 1.2 GHz minimum frequency and no `taskset`, the same build measures 73.5, 64.3
and 127.8 ms. The governor lowers the clock of the cores that an offloading process leaves idle
([cpu-governor-and-offload.md](cpu-governor-and-offload.md)).

Three host costs account for most of the difference between a per-element host path (205 ms on
MobileDet at ~816 MHz) and the measured one:

- Per-element cube packs. Where the conv entries pack their cubes one element at a time, that loop
  is 83% of the int8 depthwise call ([../depthwise-conv.md](../depthwise-conv.md)).
- Buffers sized for the largest call and synced whole on every call
  ([bo-sync-cost.md](bo-sync-cost.md)).
- The int32 readback and the host requant. On MobileDet's direct convs the `read` phase is the
  int32 copy, 262 ms over 21 invokes against 19 ms of `FINI_BO`. The int8-out writer removes it
  ([../encodings/out-cvt-converter.md](../encodings/out-cvt-converter.md)).

**Faithfulness: COCO-val mAP, the detection correctness check.** On the host path of §"Single-stream
latency", COCO mAP over 500 val images is at CPU parity for all three detectors [HW sweep
2026-10-29]. The table there gives them.

**Throughput: multi-camera pool, aggregate detection_fps.** The pool runs one detector process per
camera (`num_threads: 1`), live on the RK1 through Frigate's `rocket.py` detector, with four
SSD-MobileDet cameras [HW sweep 2026-07-01]. The `ROCKET_CPU_AFFINITY` variable pins each process
to a distinct A76 core. The P contexts then spread across the four big cores instead of colliding
on one.

**Conditions:** RK1, NPU clock ~816 MHz, the NPU IRQs pinned to the big cores
(`npu_set_irq_affinity.sh throughput`), the 2026-07-01 delegate build.

**Not characterized:** the live pool on the host path of §"Single-stream latency".

| pool size P | detection_fps | scaling |
|---|---|---|
| 1 | 3.20 | 1.00x |
| 2 | 6.00 | 1.88x |
| 3 | 8.00 | 2.50x |
| 4 | 9.55 | **2.98x** |

Aggregate throughput scales ~3x at P=4, with each of the four A76 cores running one detector (~73%
busy) while the A55 cores handle video decode. Scaling is below linear (2.98x, not 4x) because a
MobileDet inference on that build is DRAM-bandwidth-bound in its host scatter/gather phase.
Per-inference latency rises 338->424 ms as the four contexts contend for bandwidth. The live pool
therefore stays below the submit-bound limit of the delegate alone, which scales 1.00, 2.17, 3.11
and 3.56x at P=1 to 4 (~816 MHz, 2026-06-28).
The pool needs no delegate or driver change, and it is how Frigate runs cameras (one process each).

On the host path of §"Single-stream latency", the delegate's own pool test scales 3.33-3.51x at P=4,
52.9-53.6 inferences a second [HW sweep 2026-10-29]. Every process's output hash equals the single
process's. The test is `tools/pool_hash.py`, one SSD-MobileDet process per A76 core at `nthreads=1`,
on `performance` at NPU 928 MHz.

**Verdict.** For detection the FOSS-NPU delegate is an accelerator with CPU-parity accuracy, and
its gain is throughput. One pinned process per camera serves ~3x the aggregate detection rate of a
single stream. The NPU takes the conv/matmul work of four cameras. Each A76 core runs one
detector's host work at ~73% busy, and the A55 cores handle video decode. Single-stream latency is
bound by host work, so it falls with host-path changes rather than with changes to the NPU submit
path.

## ONNX Runtime (ort-rocket)

`ort-rocket` is an ONNX Runtime execution provider (EP) on `librocketnpu`. It claims whole encoders,
and single ConvTranspose and Conv nodes (Conv behind a cost rule), and leaves the rest of a graph on
the CPU EP. A vision family's encoder runs one image per run, and RF-DETR runs at its nano and base
geometries only. The ort-rocket README has the full per-model table. The table below gives each
model's single-stream speedup and faithfulness against the CPU EP [HW sweep 2026-10-27, rocket-userspace
`88cae6d` without its Conv estimate refit, ort-rocket `fa7810f`]. The COCO mAP and mask IoU figures are from July
2026 builds.

**Conditions:** for the vision rows, RK1, NPU clock 928 MHz (the 900 MHz request), governor
`performance`, one image per run, warm medians. The EP and the CPU EP both run under
`taskset -c 4-7` at 4 intra-op threads. The CPU EP runs at `ORT_ENABLE_ALL`, and the EP session at
`ORT_DISABLE_ALL`, which a family claim needs, with spinning off. The pix2pix and SAM mask decoder
rows run both sessions at `ORT_ENABLE_ALL` with spinning off. The mask decoder row runs with the
NPU cores held on (`power/control=on`). The Laya rows use ONNX Runtime's default thread pool and the
default session (§"ONNX Runtime thread spinning").

**Not characterized:** the Depth Anything rows under the `Conv` cost rule's current estimate.

| Model | Offloaded | Single stream, x the CPU EP | Faithfulness against the CPU EP |
|---|---|---:|---|
| SigLIP-B/16 | plain ViT, d=768, 196 tokens | 3.05-3.09x | hidden-state cosine 0.9999868 |
| CLIP ViT-B/16 | plain ViT, d=768, 197 tokens | 3.01-3.04x | hidden-state cosine 0.9999890 |
| Depth Anything v2 Base | DINOv2 ViT-B, global attention over 1370 tokens | 2.25-2.28x | depth cosine 0.9999999 |
| RF-DETR base | windowed DINOv2 ViT-S, d=384 | 1.83-1.88x | COCO mAP 0.564 against 0.564, 200 images |
| RF-DETR nano | windowed DINOv2 ViT-S, d=384 | 1.70-1.73x | COCO mAP 0.5049 against 0.5051, 500 images |
| Depth Anything v2 Small | DINOv2 ViT-S, global attention over 1370 tokens | 1.63-1.65x | depth cosine 0.9999998 |
| SAM ViT-B | ViT-Det, d=768, 4096 tokens | 1.08-1.10x | encoder cosine 0.9999996, mask IoU 0.9998 |
| Laya English | ModernBERT-large, d=1024, 48-512 tokens | 2.34-5.05x | 4 of 4 answers agree |
| Laya typed decisions | ModernBERT-large, d=1024, 58-639 tokens | 2.68-5.36x | 3 of 3 answers agree, option probabilities within 3.3e-4 |
| Laya multilingual | mmBERT-base, d=768, 56-726 tokens | 1.50-3.57x | 6 of 6 answers agree, option probabilities within 3.3e-4 |
| pix2pix generator | ConvTranspose and Conv nodes | 3.09x at 4 threads, 1.92x with the ConvTranspose claim alone | PSNR 77.3 dB, and 80.8 dB with the ConvTranspose claim alone |
| SAM ViT-B mask decoder | ConvTranspose nodes | 1.06x at 1 point, 1.11x at 64 points | Mask IoU 0.9975 or better at 1 point, and 0.9931 the worst of 192 masks at 64 points |

A model's single-stream ratio falls as its attention-to-projection ratio, `ntok / 4d`, rises. The
ratio is averaged over the layers, with a windowed layer counted at its window length:

| Model | `ntok / 4d` |
|---|---:|
| SigLIP-B/16, CLIP ViT-B/16 | 0.06 |
| RF-DETR nano | 0.17 |
| RF-DETR base | 0.33 |
| Depth Anything v2 Base | 0.45 |
| SAM ViT-B | 0.49 |
| Depth Anything v2 Small | 0.89 |

The ratio orders the global-attention encoders: SigLIP and CLIP lead, then Depth Anything Base,
then Small. The two Depth Anything variants run the same 1370 tokens, and Base's d=768 halves the
ratio and raises the speedup from 1.63-1.65x to 2.25-2.28x. The windowed encoders measure below a
global encoder at a similar ratio: RF-DETR (0.17, 0.33) measures below Depth Anything Base (0.45).
RF-DETR base measures slightly above nano, so the ratio does not separate values that close. SAM
measures lowest of all.

SAM's four global layers attend over 4096 tokens, 16.8 million scores per head, the largest single
attention in the set. They are the likely cause of its low ratio [hypothesis]. Its attention also
runs on its own per-window path rather than on the flash attention head range. The head range's
shorter host passes therefore do not apply to it
([attention-offload-crossover.md](attention-offload-crossover.md) §"The head range's host passes at
an encoder shape").

ONNX Runtime's CPU kernels run the Laya graph at ~77 GOP/s, which places Laya's ratio above what the
vision models predict.

### Per-node claims

A `ConvTranspose` or `Conv` claim gains in proportion to the claimed op's share of the model. On
pix2pix the CPU EP takes 146.6 ms, the `ConvTranspose` claim 76.4 ms and both claims 47.4 ms
[HW sweep 2026-10-27]. On Depth Anything v2 Small, the `Conv` cost rule claims 16 of the DPT head's
30 convs. The model then runs in 0.931-0.948x the time it takes with the `Conv` claim off
[HW sweep 2026-10-29].

**Conditions:** RK1, NPU clock 928 MHz (the 900 MHz request), governor `performance`, 4 intra-op
threads on the A76 cores, spinning off. pix2pix runs both sessions at `ORT_ENABLE_ALL`. Depth
Anything runs at the `ORT_DISABLE_ALL` that its family claim needs.

### Multi-stream throughput

One process per stream adds throughput, because a single stream is partly host-bound. The NPU
pool's best is ~1.5x the single-stream rate on RF-DETR nano and ~1.6x on SAM's image encoder
[HW sweep 2026-10-27/29]. The CPU EP's own pool runs at 0.98-1.10x its single stream.

**Conditions:** RK1, NPU clock 928 MHz (the 900 MHz request), governor `performance`, spinning
off. Each EP process runs 4 intra-op threads at `ORT_DISABLE_ALL`. RF-DETR nano's EP processes run
unpinned, and SAM's each on its own A76 core. The CPU EP runs at `ORT_ENABLE_ALL` as 2 processes of
4 threads, 4 of 2 or 8 of 1. The table gives its best layout.

**Not characterized:** RF-DETR base, RF-DETR nano above five processes, and SAM's CPU EP at 8
processes of 1 thread, which does not fit in 32 GB.

| Model | NPU pool, best | NPU processes | CPU EP pool, best | CPU EP processes x threads | NPU pool over the CPU EP's best pool |
|---|---:|---:|---:|---|---:|
| RF-DETR nano | 7.57-7.61 img/s | 5 | 2.85-3.20 img/s | 8 x 1 | 2.36-2.67x |
| SAM ViT-B image encoder | 0.157-0.164 img/s | 4 | 0.090-0.095 img/s | 4 x 2 | 1.72-1.75x |

SAM's pool peaks at four processes, one per A76 core, and falls to 0.126-0.129 img/s at five.
RF-DETR nano's pool still rises 1.04-1.06x from four processes to five.

### The CPU EP baseline on the RK3588

The CPU EP's thread setting changes its time by up to 1.75x [HW sweep 2026-10-09]:

| Model | 8 threads | ONNX Runtime's default pool | 4 threads, `taskset -c 4-7` |
|---|---:|---:|---:|
| SigLIP-B/16 | 340-537 ms | 331-573 ms | 345-346 ms |
| CLIP ViT-B/16 | 340-504 ms | 306-567 ms | 326 ms |
| Depth Anything v2 Small | 1604-1643 ms | 1410-1417 ms | 1349-1351 ms |
| RF-DETR nano | 432-535 ms | 374-375 ms | 345-346 ms |
| RF-DETR base | 1080-1240 ms | 1010-1017 ms | 951-953 ms |

**Conditions:** RK1, governor `performance`, each figure the range over four processes, 2026-10-09.

Within one process each setting is steady. Across processes, an explicit 8 threads falls into modes
up to 1.75x apart, because the threads span both clusters. The default pool, which
`intra_op_num_threads` 0 builds with its own thread affinity, is bimodal on SigLIP and CLIP. Four
threads on the A76 cores give the fastest median on four of the five models, and repeat within
0.2%. The ort-rocket table above uses that setting.

With its spinning off, the EP measures the same in either process setting.

### ONNX Runtime thread spinning

ONNX Runtime's intra-op threads spin while idle by default, and the spinning slows the EP. With
`session.intra_op.allow_spinning` set to `0`, the EP gains and the CPU EP does not change
[HW sweep 2026-09-27]. An offloaded node leaves the intra-op threads idle, so they spin on the A76
cores that the EP's host threads and NPU workers need [hypothesis].

**Conditions:** mainline RK1, `taskset -c 4-7`, governor `performance`, 2026-09-27.

| Case | Default session | Spinning off |
|---|---:|---:|
| Laya, one 58-token request, EP | 137-139 ms | 118-121 ms |
| pix2pix, 4 threads, EP | 99.9 ms, 1.46x the CPU EP | 82.9 ms, 1.76x |
| pix2pix, 4 threads, CPU EP | 145.5 ms | 146.6 ms |
| SAM mask decoder, EP, x the CPU EP | 1.017x | 1.049x |

Four of the seven vision encoders gain as well, and three move by 4% or less [HW sweep 2026-10-27]:

| Encoder | Time with spinning off, x the default session |
|---|---:|
| SigLIP-B/16 | 0.81-0.84x |
| Depth Anything v2 Small | 0.84-0.91x |
| RF-DETR nano | 0.90-0.91x |
| Depth Anything v2 Base | 0.91-0.95x |
| RF-DETR base | 0.96-0.99x |
| SAM ViT-B | 0.96-1.00x |
| CLIP ViT-B/16 | 0.99-1.01x |

**Conditions:** RK1, NPU clock 928 MHz (the 900 MHz request), governor `performance`,
`taskset -c 4-7`, 4 threads, the build of the ort-rocket table above, 2026-10-27.

At 2 threads pix2pix shows no difference. A 726-token Laya request varies +-10% between passes
either way, so its effect is not characterized. An execution provider cannot change a session's
options, so the application must set this one:

```python
so.add_session_config_entry("session.intra_op.allow_spinning", "0")
```

Spinning biases a comparison in the same way as a load-sampling CPU governor
([cpu-governor-and-offload.md](cpu-governor-and-offload.md)). Both are host-side defaults that slow
the offloading configuration and leave the CPU-only configuration unchanged.

## Evidence

### At a glance

#### Method

- **Board and operating point.** The NPU clock boots at 200 MHz and rises under load. The clock
  patch's request is set at module load with `rocket_npu_clk_hz`. The 900 MHz request runs at
  928 MHz and the 600 MHz request at ~816 MHz at the 0.80 V rail ([clock.md](clock.md)). The
  2026-11-01 series ran at the 900 MHz request, and the per-model sweeps of 2026-07 and 2026-08 at
  the 600 MHz request. On the per-model sweeps the request is confirmed during the run, because a
  sample taken while idle between repetitions measures 200.
- **CPU threads.** The CPU prefill baseline runs at 8 threads, unpinned. On b11539 that prefills
  4-10% faster than 4 threads on the A76 cores, on the two models compared. The 2026-11-01 decode
  rates run at `taskset 0xf0 -t 4`. That decodes 1.32-4.22x faster than unpinned `-t 8` on every
  one of the fifteen GGUFs measured, F16 included [HW sweep 2026-11-01]. The per-model sweeps of
  2026-07 ran decode at 8 threads, unpinned.
- **Warm discipline.** The per-model sweeps' figures are warm medians (`llama-bench -r 2` plus a
  discarded warmup).
- **Flags.** On the per-model sweeps, quantized GGUFs run at `-b 2048 -ub 2048`, because a
  quantized GGUF re-dequantizes to fp16 per micro-batch. On the 2026-06-28 builds that preceded the
  per-model tables, `-ub 2048` measured 2.1-2.25x the default `-ub 512` on Qwen3.5-9B and
  Qwen3.6-27B. On the 2026-08-29/30 builds it measures 0.91-1.53x over eleven models,
  model-dependent and negative on two ([quant-prefill-microbatch.md](quant-prefill-microbatch.md)).
  Where the fp16 form fits RAM, `ROCKET_QUANT_RESIDENT=auto` at the default `-ub` is faster than
  either on seven of seven models (§"Tuning gain per model").
- **Three tables.** One tokens/s figure misleads, because decode (M=1 GEMV) stays on the CPU on
  both backends ([decode-gemv.md](decode-gemv.md)). Every model therefore reports prefill (the
  NPU's job), decode (the CPU-bound equalizer), and the interactive decomposition.
- **Faithfulness.** Every speed number is paired with a correctness check. Detection uses COCO mAP,
  and the Whisper and SigLIP encoders use output cosine.
- **Reproducibility.** The generator and the raw `llama-bench` output for most models are under
  [data/](data/). A few runs are summarized inline rather than archived as separate raw files:
  gpt-oss-20b, the DeepSeek `ROCKET_MOE=1` re-bench, and the Ministral-3-8B perplexity. Models are
  stock GGUFs, with provenance per model. F16 is `llama-quantize <bf16> <f16> F16` from the
  published BF16.

#### LLM and multimodal

gpt-oss's faithfulness also rests on `test-rocket-moe`, not on a greedy match. The faithfulness
table of §"gpt-oss-20b native-quant expert path and results" shows why the greedy comparison has no
resolution here.

The table is one measurement series on the RK1 [HW sweep 2026-11-01]. It ran llama.cpp b11539,
the backend built from its 2026-10-29 state, NPU clock 928 MHz and CPU governor `performance`. Each
configuration of a GGUF ran in its own process from a memory reset, and an NPU configuration first
ran a discarded warm-up process. `llama-bench` ran three repetitions on the NPU and one on the CPU.
The series made two passes, and the second reversed both the GGUF order and the configuration
order.

Each speedup is the mean of the two per-pass ratios. The two ratios of an entry agree within 2% on
24 of 27 entries, and within 5.6% on all of them. On the MoE rows the mean of the three
repetitions agrees with the last repetition within 0.3%. The one-time expert ingest therefore does
not reach the timed repetitions.

The 3-12 B F16 configurations stream their weights, because the default F16 residency admits only
a weight with `K` <= 2048. SmolVLM2's F16 configuration holds 130 of its 142 weights resident under
the default 2048 MB budget. The series does not decompose the NPU's rise over the earlier figures
among the clock, CBUF weight streaming, the host-cost cuts and the governor. The CPU rates the
earlier ratios imply are within about 4% of the series'.

On b11539 the 8-thread CPU backend decodes the table's GGUFs at:

| GGUF | 8 threads, t/s | 4 threads on the A76 cores, t/s | Ratio |
|---|---:|---:|---:|
| SmolVLM2-2.2B Q4_K_M | 5.90 | 20.70 | 3.51x |
| Llama-3.2-3B Q4_K_M | 4.51 | 11.09 | 2.46x |
| Ministral-3-3B Q4_K_M | 5.02 | 10.47 | 2.08x |
| Phi-4-mini Q4_K_M | 4.15 | 8.77 | 2.11x |
| Ministral-3-8B F16 | 1.21 | 1.60 | 1.32x |
| Ministral-3-8B Q8_0 | 1.97 | 3.01 | 1.53x |
| Ministral-3-8B Q4_K_M | 2.39 | 4.72 | 1.98x |
| Qwen3.5-9B Q4_K_M | 1.90 | 4.00 | 2.11x |
| Gemma-4-12B F16 | 0.80 | 1.07 | 1.33x |
| Gemma-4-12B Q8_0 | 1.27 | 1.97 | 1.55x |
| Gemma-4-12B Q4_K_M | 1.50 | 3.05 | 2.03x |
| Phi-4 Q4_K_M | 1.70 | 2.74 | 1.62x |
| DeepSeek-V2-Lite Q4_K_M | 3.20 | 13.52 | 4.22x |
| gpt-oss-20b MXFP4 | 3.14 | 9.11 | 2.90x |
| Qwen3.6-27B Q4_K_M | 0.79 | 1.30 | 1.65x |

**Conditions:** `tg64`, three repetitions, the series' first pass, each configuration from a
memory reset [HW sweep 2026-11-01].

Why eight threads lose 1.32-4.22x is not established. The likely cause is that each decode step
waits on its A55 threads [hypothesis].

##### Revision history

The table before the 2026-11-01 series:

| Model | Prefill NPU t/s, pp512->2048 | x CPU, pp512->2048 | Decode Q4 t/s, 8 threads |
|---|---:|---:|---:|
| SmolVLM2-2.2B | 99.0 -> 52.6 | 3.5x -> 1.9x | 14.5 |
| Llama-3.2-3B | 61.8 -> 45.5 | 3.4x -> 2.6x | 7.9 |
| Ministral-3-3B | 57.6 -> 39.8 | 3.2x -> 2.4x | 7.7 |
| Phi-4-mini | 55.4 -> 39.8 | 3.4x -> 2.6x | 7.0 |
| Ministral-3-8B | 27.4 -> 21.4 | 3.8x -> 3.1x | 3.8 |
| Qwen3.5-9B | 25.9 -> 24.9 | 3.6x -> 3.5x | 3.6 |
| Gemma-4-12B | 17.4 -> 15.0 | 3.6x -> 3.2x | 2.5 |
| Phi-4 (14B), Q4_K_M | 10.3 -> 11.8 | 2.9x -> 3.5x | 2.2 |
| DeepSeek-V2-Lite, Q4_K_M | 26.0 -> 28.4 | 1.33x -> 1.53x | 7.8 |
| gpt-oss-20b, MXFP4 | 22.0 -> 28.1 | 1.81x -> 2.38x | 7.2 |
| Qwen3.6-27B, Q4_K_M | 5.4 -> 7.8 | 3.0x -> 4.4x | 1.1 |

**Conditions:** NPU clock ~816 MHz (the 600 MHz request). Every row but the two MoE rows is a
per-model sweep of 2026-07-01 to 2026-07-04, F16 at `-ub 512`. The quantized rows ran at
`-b 2048 -ub 2048` against a CPU built without the weight repack. The two MoE rows are 2026-08-27
builds on llama.cpp b10558. Decode is the CPU at 8 threads, unpinned.

The gpt-oss-20b row's 22.0 and 28.1 are six-run means of the default. The DeepSeek-V2-Lite row's
26.0 and 28.4 are the default configuration of the accept-boundary map at `M_e` 48 and 192.

Against a CPU built with the weight repack, at pp2048, the quantized rows measured Phi-4 3.06x
and DeepSeek-V2-Lite 1.44x. They measured gpt-oss-20b 1.90x and Qwen3.6-27B 3.21x [HW sweep
2026-09-29, ~816 MHz, llama.cpp b11242, governor pinned,
[cpu-repack-baseline.md](cpu-repack-baseline.md)]. A
2026-10-09 `librocketnpu` build measured gpt-oss pp512 at 24.90-25.23 t/s and DeepSeek-V2-Lite
pp2048 at 32.69-32.83 t/s under governor `ondemand`, NPU only.

The DeepSeek default measured 27.62-28.37 t/s at pp2048 across the 2026-08-27 runs on llama.cpp
b10558, 1.28-1.31x the experts-on-CPU configuration of each run's session. The native-quant table
of §"DeepSeek-V2-Lite (MLA + MoE, Q4_K_M)" gives 27.9 [HW sweep 2026-08-27].

#### Tuning per model

Each figure is a mean over three rotated interleaved passes. The memory is reset before every
configuration, and the governor pinned [HW sweep 2026-08-28..31, RK1, ~816 MHz].

The below-4 B residues of 0.79-0.86 are measured under the dqc mechanism control, the same `-ub`
pair with the dequantization cache on.

The class of `Qwen3.5-0.8B` cannot resolve a ratio of this size at any affordable pass count, so
its residency-only test point was not run. The Qwen3.5-0.8B F16 GGUF's spread across processes is
~8-11%, unique to the 0.8B class on this board. The spread exceeds the effect, so its ratio does not
resolve at an affordable pass count. The raw rows and the per-pass ratios are in
[data/tuning-matrix.md](data/tuning-matrix.md).

#### ASR beyond Whisper

Each figure is a single fresh-process A/B run.

### LLM (llama.cpp via ggml-rocket)

#### Ministral-3-8B

The combined `pp2048+tg128` llama-bench point (F16 NPU 10.3 t/s, Q4_K_M NPU 12.3 t/s) validates
the TTFT and streaming decomposition to ~10% [HW sweep 2026-07-01, ~816 MHz].

The 2026-11-01 Q8_0 and Q4_K_M configurations hold all 235 weights resident.

##### Revision history

| Quantity | Earlier figure | Its conditions | Current figure |
|---|---|---|---|
| F16 prefill, CPU -> NPU | pp512 7.2 -> 27.4 (3.8x), pp1024 7.0 -> 24.4 (3.5x), pp2048 6.8 -> 21.4 (3.1x) | ~816 MHz, `-ub 512`, 2026-07-01 | 44.0, 41.5 and 39.6 t/s (6.1x, 5.8x, 5.8x), 2026-11-01 |
| Decode, tg64 | CPU F16 1.37, Q8_0 2.46, Q4_K_M 3.76; NPU-loaded 1.43, 2.47, 3.83 t/s | 8 threads, unpinned, 2026-07-01 | 1.60, 3.01 and 4.72 t/s at `taskset 0xf0 -t 4`, 2026-11-01 |
| RAG turn, 2048-token prompt -> 200 tokens | TTFT F16 CPU 299 s, F16 + NPU 96 s, Q4_K_M + NPU 118 s; totals 445, 236 and 170 s | The 2026-07-01 rates, Q4_K_M streamed | 301, 52 and 45 s; totals 426, 177 and 88 s |
| Chat turn, 128-token prompt -> 400 tokens | F16 + NPU 285 s, Q4_K_M + NPU 112 s (TTFT 5 and 8 s) | The 2026-07-01 rates | 400 decoded tokens take 250 s at F16 and 85 s at Q4_K_M. No pp128 rate in the 2026-11-01 series |

#### gpt-oss-20b

The 2026-07-14 measurements come from one board in one session, clock pinned, with every baseline
re-measured alongside the configuration under test [HW sweep, ~816 MHz, 2026-07-14].

The 2026-08-27 prefill table comes from one board and one `.so`. Its default column is a mean of
six runs. One run directly after a rebuild fell at the bottom of that spread, and looked like a 10%
regression. The `ROCKET_MOE=0` control returned 13.61-13.63 at pp2048 on every binary, 13.62 in
the run the prefill table quotes ([data/gpt-oss-20b.md](data/gpt-oss-20b.md)). The board had
therefore not drifted, and the spread belongs to the default configuration.

The July 2026 builds' default kept the routed experts on the CPU. On them, a `-ub 512` default
figure of 13.11 t/s at pp2048 looks like a regression against the like-for-like `-ub 2048` figure
of ~11. The regression does not exist. The ~11 is 10.99 on 2026-07-14, before the attention-sink
fix. The expert path, the default from 2026-08-27, is a different configuration
(§"Micro-batch size").

The 2026-07-14 matrix comes from a different llama.cpp and a different driver build [HW sweep
2026-07-14, llama.cpp a646006f0]. It is not row-comparable with the 2026-08-27 table. Each matrix is
internally consistent, and the ratios agree in direction.

| Configuration, 2026-07-14 | pp512 | pp2048 |
|---|---:|---:|
| CPU | 13.09 | 12.40 |
| Experts on CPU | 14.11 | 13.89 |
| Experts on NPU | 17.57 | 26.78 |

`ROCKET_MOE=1`'s margin over the default is 12.8% at pp512 and 21.0% at pp2048 against the
six-run default mean of the prefill table (22.0 -> 24.81, 28.1 -> 34.01). In a second session the
same day, behind a `drop_caches`, it is 6.3% and 18.0% (21.96 -> 23.34, 26.51 -> 31.27).

##### Revision history

| Quantity | Earlier figure | Its conditions | Current figure |
|---|---|---|---|
| The default's deficit against `ROCKET_MOE=1` | 8% at pp512, 16% at pp2048 | One default run (22.82, 28.66) against `ROCKET_MOE=1` (24.81, 34.01), 2026-08-27 | `ROCKET_MOE=1` 6-13% and 18-21% faster, two sessions |
| Experts-on-CPU speedup over the CPU | ~1.04x at pp2048 (12.62 -> 13.11 t/s) | 2026-07-02, the NPU at the default `-ub 512`, a default that kept the routed experts on the CPU | 1.15x at pp2048, `-ub 2048`, 2026-08-27 |
| Time to first token, 2048-token prompt | CPU 162 s, NPU 156 s (~4%), and a combined `pp2048+tg128` point of CPU 11.61, NPU 12.03 t/s | 2026-07-02, the same default | ~54 s on the NPU, 2.29x the repacked CPU, 2026-11-01. The combined point is not re-measured on the expert path |
| Time to first token, 2048-token prompt | ~73 s on the NPU against ~174 s on the CPU | The 2026-08-27 pp2048 rates, 28.1 and 11.80 t/s, ~816 MHz, unrepacked CPU | ~54 s on the NPU, 2.29x the repacked CPU, 2026-11-01 |
| Decode, tg64 | CPU 7.15, NPU-loaded 7.16 t/s | 8 threads, unpinned, July 2026 build | 9.11 t/s at `taskset 0xf0 -t 4`, 2026-11-01 |
| Default prefill, pp2048 | 28.1 t/s, 2.38x the CPU | ~816 MHz, unrepacked CPU, 2026-08-27, six-run mean | 37.8 t/s, 2.29x the repacked CPU, 62 of 72 stacks placed, 2026-11-01. The 2026-08-27 table stays in the reading path as the one-build comparison of the four configurations |

### gpt-oss-20b native-quant expert path and results

The 2026-10-09 `librocketnpu` build introduced the NEON group-wise int8 readback. The int8
activation pack's group copy also stopped compiling to a byte loop. End to end, on llama.cpp
a7fb71f with `--repack 0` and `-p 512 -n 0 -b 2048 -ub 2048`, gpt-oss measured pp512 24.90-25.23
t/s [HW sweep 2026-10-09, ~816 MHz, governor `ondemand`]. The gain over the build before it is
1.093-1.098x, with the ingest unchanged at 20.7-21.1 s. DeepSeek-V2-Lite at `-p 2048 -ub 2048`
measured pp2048 32.69-32.83 t/s, with an ingest of 18.2-18.4 s.

The `llama-bench` warm-up: three timed pp512 repetitions take 22.60, 22.51 and 21.27 s each, in
order [HW sweep 2026-10-09, one run]. The run exercised 1629 experts, where a one-repetition run
exercises 1575. Two builds differ only in the pack's speed, with ingests of ~41 s and ~21 s. Over
three rotated passes each, they measure pp512 at 21.18 and 22.67 t/s.

At gpt-oss's expert (K=N=2880, group 576, three workers), `moe_gw_bench` measures the group-wise
call across the 2026-10-09 build. It moves 4.16-4.19 -> 3.40-3.52 ms at M 64 and 12.05-12.09 ->
8.92-10.26 ms at M 256. Each is two rotated runs a build [HW sweep 2026-10-09, RK1, ~816 MHz,
governor `ondemand`].

End to end, the build before it measured gpt-oss pp512 at 22.78-22.99 t/s. The 1.093-1.098x is
paired within a pass, over three rotated passes. On the build before it, DeepSeek-V2-Lite measured
pp2048 28.82-29.99 t/s with a 38.3-38.8 s ingest, over two rotated passes. That build also lacked
the faster pack, and the timed repetitions of `llama-bench` include part of the ingest.

The synthetic expert-path test in the faithfulness table is `test-rocket-moe`, 7/7.

The greedy row: on one prompt the native-quant configuration diverges at token ~35, no earlier and
no worse than the experts-on-CPU path, which diverges identically. That equality reads as evidence
that the expert path adds nothing. On the default placement the two configurations diverge at word
16 and word 10. They do not track each other, and one prompt cannot show that they do.

### Qwen3.5-9B

IQ4_XS was measured as a gap-finder for the quantized-prefill path: it tests whether the path covers
importance-matrix quants. It finds no gap. A CPU fallback would give ~1x at pp2048 [expected],
against the 3.1x measured.

The combined `pp2048+tg128` llama-bench point (F16 NPU 11.96, Q4_K_M NPU 16.77, IQ4_XS NPU 17.01
t/s) validates the TTFT-plus-stream decomposition to ~5%.

The raw `llama-bench` and `llama-perplexity` output is in [data/qwen35-9b.md](data/qwen35-9b.md).

#### Revision history

| Quantity | Earlier figure | Its conditions | Current figure |
|---|---|---|---|
| F16 prefill, pp2048, CPU -> NPU | 7.08 -> 24.85 t/s (3.5x) | ~816 MHz, `-ub 512`, 2026-07-02 | 36.1 t/s, 4.9x, 2026-11-01 |

### Phi-4-mini-instruct

The combined `pp2048+tg128` llama-bench point (F16 NPU 18.35, Q8_0 20.89, Q4_K_M 21.43 t/s) runs
~15-20% below the TTFT+stream sum. That margin is the one by which the `total turn` figures are
optimistic.

The raw output is in [data/phi4-mini.md](data/phi4-mini.md).

#### Revision history

| Quantity | Earlier figure | Its conditions | Current figure |
|---|---|---|---|
| F16 prefill, pp2048, CPU -> NPU | 15.07 -> 39.79 t/s (2.6x) | ~816 MHz, `-ub 512`, 2026-07-02 | 76.1 t/s, 5.0x, 2026-11-01 |

### Gemma-4-12B-it

The fp16 prefill path was first brought up on this model.

The combined `pp2048+tg128` llama-bench point (F16 NPU 7.37, Q8_0 8.72, Q4_K_M 9.61 t/s) validates
the TTFT-plus-stream decomposition.

The raw output is in [data/gemma4-12b.md](data/gemma4-12b.md).

The 2026-11-01 Q8_0 and Q4_K_M configurations hold 298-310 of 325 weights resident.

#### Revision history

| Quantity | Earlier figure | Its conditions | Current figure |
|---|---|---|---|
| F16 prefill, CPU -> NPU | pp512 4.82 -> 17.42 (3.6x), pp1024 4.78 -> 16.09 (3.4x), pp2048 4.63 -> 14.98 (3.2x) | ~816 MHz, `-ub 512`, 2026-07-02 | 25.62, 25.31 and 24.64 t/s (5.1x, 5.2x, 5.2x), 2026-11-01 |
| Decode, tg64 | CPU F16 0.94, Q8_0 1.75, Q4_K_M 2.44; NPU-loaded 0.94, 1.59, 2.50 t/s | 8 threads, unpinned, 2026-07-02 | 1.07, 1.97 and 3.05 t/s at `taskset 0xf0 -t 4`, 2026-11-01 |
| RAG turn, 2048-token prompt -> 200 tokens | TTFT F16 CPU 442 s, F16 + NPU 137 s, Q8_0 + NPU 152 s, Q4_K_M + NPU 154 s; totals 655, 350, 278 and 234 s | The 2026-07-02 rates, the quantized GGUFs streamed | 434, 83 and 84 s; totals 621, 270 and 149 s |
| Chat turn, 128-token prompt -> 400 tokens | F16 + NPU 433 s, Q4_K_M + NPU 171 s (TTFT 7 and 11 s) | The 2026-07-02 rates | 400 decoded tokens take 374 s at F16 and 131 s at Q4_K_M. No pp128 rate in the 2026-11-01 series |

### Qwen3.6-27B (hybrid Gated-DeltaNet, Q4_K_M)

The combined `pp2048+tg128` point (CPU 1.70, NPU 5.57 t/s) validates the TTFT-plus-stream
decomposition.

The raw output is in [data/qwen36-27b.md](data/qwen36-27b.md).

#### Revision history

| Quantity | Earlier figure | Its conditions | Current figure |
|---|---|---|---|
| Q4_K_M prefill, pp2048, CPU -> NPU | 1.78 -> 7.78 t/s (4.4x) | ~816 MHz, `-b 2048 -ub 2048`, CPU without the weight repack, 2026-07-03 | 11.5 t/s, 3.8x the repacked CPU at 3.0 t/s, 2026-11-01 |
| Decode, tg64 | CPU 1.10, NPU-loaded 1.11 t/s | 8 threads, unpinned, 2026-07-03 | 1.30 t/s at `taskset 0xf0 -t 4`, 2026-11-01 |
| First-token wait, 2048-token prompt | ~19 min on the CPU, ~4.5 min on the NPU (4.4x) | The 2026-07-03 rates | ~11 min and ~3 min, from the 2026-11-01 rates |

### Llama-3.2-3B-Instruct

The combined `pp2048+tg128` llama-bench point (F16 NPU 21.06, Q8_0 21.89, Q4_K_M 22.34 t/s) runs
below the TTFT+stream sum, which sets the margin by which the `total turn` figures are optimistic.

The raw output is in [data/llama32-3b.md](data/llama32-3b.md).

#### Revision history

| Quantity | Earlier figure | Its conditions | Current figure |
|---|---|---|---|
| F16 prefill, pp2048, CPU -> NPU | 17.29 -> 45.54 t/s (2.6x) | ~816 MHz, `-ub 512`, 2026-07-03 | 85.1 t/s, 5.0x, 2026-11-01 |

### Ministral-3-3B-Instruct-2512

The per-model benchmark format of this record was first validated on Ministral-3-8B.

The combined `pp2048+tg128` llama-bench point (F16 NPU 19.10, Q8_0 20.06, Q4_K_M 20.86 t/s) runs
below the TTFT+stream sum, which sets the margin by which the `total turn` figures are optimistic.

The raw output is in [data/ministral3-3b.md](data/ministral3-3b.md).

#### Revision history

| Quantity | Earlier figure | Its conditions | Current figure |
|---|---|---|---|
| F16 prefill, pp2048, CPU -> NPU | 16.26 -> 39.78 t/s (2.4x) | ~816 MHz, `-ub 512`, 2026-07-03 | 75.0 t/s, 4.7x, 2026-11-01 |

### DeepSeek-V2-Lite (MLA + MoE, Q4_K_M)

This model was measured in July 2026 as a combined gap-finder. It has two offload gaps in one
model, MLA attention and the routed experts, both addressed at the op level. Neither was a default
win then. Attention stayed on the CPU. The fp16 expert path is a net loss (the fp16-path table), so
that default kept the experts on the CPU. The native-quant expert path, on by default from
2026-08-27, gains at long prefill (the native-quant table).

The fp16-path table's CPU column (20.40, 19.93 and 18.99 t/s) was measured in the same session as
its NPU column. Its flags are given in their current spelling. The native-quant table is from one
board and one `.so`.

The combined `pp2048+tg128` point (CPU 15.25, NPU 18.12 t/s) validates the TTFT-plus-stream
decomposition. It also implies the ~3.7 t/s decode rate after a 2048-token prompt.

The unit-test cosine is from `test-rocket-moe`.

The raw output is in [data/deepseek-v2-lite.md](data/deepseek-v2-lite.md).

#### Excluded causes

The pp512 regression with the experts on the NPU is not the partial-residency mechanism. It occurs
at 100% residency, with 0 experts streamed.

#### Revision history

| Quantity | Earlier figure | Its conditions | Current figure |
|---|---|---|---|
| Decode, tg64 | CPU 7.73, NPU-loaded 7.77 t/s | 8 threads, unpinned, 2026-07-03 | 13.52 t/s at `taskset 0xf0 -t 4`, 2026-11-01 |

### Phi-4 (14B)

The combined `pp2048+tg128` llama-bench point (Q8_0 NPU 7.48, Q4_K_M NPU 8.29 t/s) validates the
TTFT and stream decomposition. It runs ~10-15% below the naive TTFT+stream sum, because the 128
tokens that `pp2048+tg128` decodes read a full KV cache.

#### Revision history

| Quantity | Earlier figure | Its conditions | Current figure |
|---|---|---|---|
| Q4_K_M prefill, pp2048, CPU -> NPU | 3.40 -> 11.79 t/s (3.5x) | ~816 MHz, `-b 2048 -ub 2048`, CPU without the weight repack, 2026-07-04 | 21.9 t/s, 4.0x the repacked CPU at 5.47 t/s, 2026-11-01 |

### SmolVLM2-2.2B-Instruct (vision-language, mtmd)

The combined `pp2048+tg128` llama-bench point (F16 NPU 27.06, Q8_0 26.11, Q4_K_M 27.68 t/s)
validates the TTFT and stream decomposition.

#### Revision history

| Quantity | Earlier figure | Its conditions | Current figure |
|---|---|---|---|
| F16 prefill, pp2048, CPU -> NPU | 27.45 -> 52.59 t/s (1.9x) | ~816 MHz, `-ub 512`, 2026-07-04 | 117.8 t/s, 4.2x, 2026-11-01 |

### ASR (whisper.cpp via ggml-rocket)

The size trend is the crossover that the encoder analysis in
[encodings/whisper-encoder.md](../encodings/whisper-encoder.md) predicted. An integration without
KACC and with a per-call fd measured base.en below break-even. K-accumulation and the reductions in
per-submit overhead take every model size above 1x.

The per-step decoder costs of large-v3 and large-v3-turbo are from whisper-bench's full timing, in
the same NPU run as the encoder figures.

The 2026-07-04 figures are the warm mean of repetitions 2-4 of four. The base.en figures of
2026-09-26 are five interleaved repetitions a configuration, the order alternating, with the page
cache dropped and memory compacted before each. They ran whisper.cpp `eacbd82` and `rocket` 1.3.0,
with `ROCKET_KACC=1`. The 30 s window is `whisper-bench`, and the 20 s window `whisper-cli -ac 1000`
on `jfk.wav`, whose transcript was identical across all ten runs. The per-repetition timings are in
[data/whisper-encoder.md](data/whisper-encoder.md).

#### Revision history

| Measured | Quantity | Value | Conditions |
|---|---|---|---|
| 2026-07-04 | base.en encoder, 30 s window, CPU -> NPU | 1580.0 -> 1223.4 ms, 1.29x | ~816 MHz, `whisper-bench -t 4`, governor and pinning not recorded |

### ASR, multi-model (transcribe.cpp via ggml-rocket)

Each speed figure is a CPU-against-NPU A/B.

The MOSS CPU profile was taken with `TRANSCRIBE_PERF_DEBUG`. `TRANSCRIBE_PERF_DEBUG=1` alone prints
the per-step timings, the step count and `T_enc`/prefill from lightweight `ggml_time_us()` markers.
The per-op ENCODE/DECODE tables install a per-node sched-eval callback that syncs after every op.
The callback inflates long-audio decode ~10x, which makes it unusable past a few seconds. A local
RK1 patch to `src/arch/moss/model.cpp` puts just that callback behind a second flag. With the
patch, `TRANSCRIBE_PERF_DEBUG=1` gives per-step timing at full speed, and the op tables need
`TRANSCRIBE_PERF_DEBUG=1 TRANSCRIBE_OP_PROF=1`.

The Granite-Speech-2B base agreement was taken on 2026-07-21 on the hard clip only, at
transcribe.cpp `8c7ae67`. It ran `taskset -c 4-7 --threads 4` with `ROCKET_KACC=1`, a fresh
process per run and two runs a configuration. Q8 and F16 were char-identical CPU==NPU, and Q8==F16. The plus variant looped on
the CPU, and NAR diverged slightly.

The 2026-10-12 re-run used the same frontend commit, rebuilt with GCC 16.2 on kernel 7.2.9, with
ggml-rocket `6bf55cc` and rocket-userspace `ffd4ccc`, `rocket` 1.3.0 at ~816 MHz. It ran both clips,
both precisions, CPU and NPU, two fresh processes each, and both runs of every configuration
matched. jfk is char-identical everywhere. On the hard clip, F16 is char-identical CPU==NPU, and Q8
differs by one word (word 13, 246 against 249 characters). The Q8 NPU and both F16 transcripts are
byte-identical to the 2026-07-21 ones. The Q8 CPU transcript is the one that changed [HW sweep, RK1,
2026-10-12].

### Detection (tflite-rocket)

Host-path changes, and moving the requant on chip, took MobileDet from 205 to 56 ms. Each row of the
change table is the warm latency once that change is in. All rows ran at ~816 MHz (the 600 MHz
request), governor `performance`, the process on the A76 cores [HW sweep, 2026-09-27 and
2026-09-28]:

| Host path | MobileDet | SSD MobileNet v2 | EfficientDet-Lite0 |
|---|---:|---:|---:|
| Per-element cube packs, fp16-approximated depthwise | 205 ms | 201 ms | 309 ms |
| Blocked packs, uint8 depthwise on chip through CPEND | 151 | 131 | 256 |
| Matmul output and regcmd BOs sized to the call's tiles | 130 | 120 | 229 |
| Table-driven unary and concat, a one-K-tile gather straight into C, `lrintf` inlined, transposed conv packs | 105.5 | 94 | 150 |
| Transposed delegate conversions, BO size classes, int8 depthwise row bands | ~93 | ~78 | ~133 |
| Per-axis depthwise on the per-channel multiplier | unchanged | unchanged | 121 |
| uint8 direct convs on the int8-out writer | 75 | ~66 | unchanged |
| Large-plane 1×1s on the int8-out writer, per-axis direct convs on it | 55.7 | 52.1 | 101.3 |

Rows 3 to 5 leave every output byte identical, apart from SSD MobileNet v2's two 150×150 depthwise
layers, which move to the int8 path. Rows 2 and 6 to 8 change which program computes a layer, so
outputs move there. COCO mAP over 100 images does not fall at any of them. The last row is one A/B
on one build: 76.6 -> 55.7 ms, 65.6 -> 52.1 and 123.3 -> 101.3. The `tflite-rocket` README lists
the per-change ratios.

The 500-image COCO mAP is from tflite-rocket's `tools/coco_map.py`. The submit-bound scaling of the
delegate alone is from `tools/pool_throughput.py`.

The 2026-10-29 latencies are medians over four passes, one process per configuration, with the
governor order and the model order rotated. Memory was reset before every process. On 2026-09-28, at ~816 MHz, the single-stream table
measured 55.7, 52.1 and 101.3 ms on `performance` and 75.8, 70.5 and 135.0 ms on `ondemand`. Its mAP
over 100 images was 0.4040 against the CPU's 0.3980 (MobileDet), 0.3229 against 0.3215 (SSD
MobileNet v2) and 0.3023 against 0.2996 (EfficientDet-Lite0).

The delegate build of 2026-06-21 ran fp16-approximated depthwise, per-tensor depthwise int8 and
the host requant on every conv. It measured SSD-MobileDet mAP@[.5:.95] over 500 COCO val images at
0.3321, against the CPU int8 reference's 0.3318. The Δ is +0.0002, so on that build the two
depthwise approximations cost ~0 mAP.

A persistent worker pool for the delegate's host transforms saves at most 0.83% of an inference
[host-computed]. On a build of 2026-07-01 at ~816 MHz, SSDLite-MobileDet took ~380 ms an
inference, with 39 parallel host regions [HW sweep 2026-07-01]. Creating and joining their
threads takes ~3.1 ms, which is that bound. The serial input conversion writes at most 6.5 MB, in
~2-3 ms.

### ONNX Runtime (ort-rocket)

Each vision figure in the table is a warm median over four passes with the configuration order
rotated, and the pix2pix row over five.

The 2026-10-10 build shortened the head range's host passes. Against the 2026-10-09 build, SAM moved
1.02-1.03x -> 1.03-1.05x, while the ratio of every family on the head range rose 5-17%. The
2026-10-11 build hands the scores and P between attention's two matmuls as rows. On the 2026-10-10
build the table's ratios were SigLIP 2.44-2.52x, CLIP 2.39-2.48x, RF-DETR nano 1.57-1.59x and base
1.61-1.65x. Depth Anything was 1.33-1.35x and Laya English 1.99-3.95x. The CPU EP's times did not
change, and the outputs are bit-identical.

On the 2026-10-11 build (rocket-userspace `ffd4ccc`, ort-rocket `3c50635`) at ~816 MHz (the 600 MHz
request), the table's ratios were SigLIP 2.64-2.67x and CLIP 2.58-2.64x. They were RF-DETR nano
1.65-1.70x and base 1.77-1.79x, SAM 1.03-1.04x, Depth Anything Small 1.58-1.61x and Laya English
2.03-4.16x. Laya's typed decisions measured 2.29-4.18x and multilingual 1.32-2.59x. The hidden-state
cosines were 0.9999864 (SigLIP) and 0.9999955 (CLIP), and the depth cosine 0.9999999. On the
2026-10-10 build at ~816 MHz the pix2pix row measured 2.91x and 1.86x.

The 2026-10-27 table is a newer library (CBUF weight streaming as the fp16 default and
the K-accumulation stride, both in rocket-userspace `88cae6d`) at the 900 MHz request, and the two changes are not separated.
The CPU EP's times are the same on both builds, so the change is the EP's. The NPU's PVTPLL counter
read 927.8 MHz before the 2026-10-27 run.

The Depth Anything rows ran under the `Conv` cost rule's earlier NPU estimate, fitted at ~816 MHz.
The current estimate, refitted at 928 MHz, admits 16 of Small's 30 head convs where the earlier one
admits 10. It runs Small in 0.975-1.008x the earlier one's time [HW sweep 2026-10-29].

Each CPU EP process in §"The CPU EP baseline on the RK3588" started after a memory reset.

The Laya rows are two passes with the configuration order rotated, a process per configuration.
The multilingual checkpoint's EP time differed between the passes by up to 1.27x, so its range is
wide. The SAM mask decoder figures are medians over six passes, four at 64 points, with paired
ratios of 1.01-1.08x and 1.11-1.12x.

The per-node pix2pix figures are medians over five passes with the configuration order and the
governor order rotated. The Depth Anything ratio is one five-pass A/B of the claim off against the
cost rule's current estimate.

An fp32-input entry for the per-node claims removes only their input narrowing. On pix2pix
that is 0.59 ms for the `Conv` claims and 1.34 ms for the `ConvTranspose` claims [host-computed].
Against the 52.6 ms pinned time, that is 1.1% for the `Conv` claims alone and 3.7% for both. The
profile ran at 4 threads under `performance` on a build of 2026-10-08, before the per-call worker
placement change.

The pool figures are two passes, each pool size and CPU EP layout in its own process, after a page
cache drop and memory compaction. The order of the two pools is reversed between the passes, and
each ratio pairs the pools of one pass. Pass 0 ran on 2026-10-27 and pass 1 on 2026-10-29. On July
2026 builds, SAM's NPU pool measured 1.44x the CPU EP's best pool.

### ONNX Runtime thread spinning

Each vision-encoder ratio is paired within one of four passes. On the 2026-10-10 build at ~816 MHz
(the 600 MHz request), the split measured SigLIP 0.84-0.88x, Depth Anything Small 0.89-0.92x and
RF-DETR nano 0.90-0.93x. CLIP, RF-DETR base and SAM measured 0.98-1.02x. On the 2026-10-09 build,
with 8 unconfined threads, it measured SigLIP 0.75-0.79x and RF-DETR nano 0.83-0.85x.
