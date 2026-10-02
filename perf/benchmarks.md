# Benchmarks

The consolidated benchmark record for the rocket NPU stack: the canonical numbers, the method
behind them, and the per-model results. Each project README carries a curated, standalone
excerpt of the numbers relevant to that frontend; this file is the superset and states the
method once. New models are appended here first.

## At a glance

The cross-model summary; each model's full prefill / decode / interactive / faithfulness
breakdown is in its section below. All figures are warm medians, RK3588 at 600 MHz, the same
GGUF / model run NPU-vs-CPU (8-thread) [HW sweep]. The NPU is a prefill / batched-GEMM engine: it
accelerates **prompt processing (prefill)** and the **Whisper / vision encoders**; **decode**
(M=1 GEMV) stays on the CPU on both backends ([decode-gemv.md](decode-gemv.md)), scaling with
quantization rather than the backend. The two levers are orthogonal: **the NPU shortens
time-to-first-token; quantization speeds the stream and sets the RAM fit.**

### LLM and multimodal (llama.cpp via ggml-rocket)

Sorted by size. Prefill t/s and ×CPU are shown **pp512 -> pp2048** to expose the length trend: F16
peaks on short prompts and eases as output-tile readback grows with M, while a quantized GGUF
*rises* with length as its per-micro-batch dequant amortizes (`-b 2048 -ub 2048`,
[quant-prefill-microbatch.md](quant-prefill-microbatch.md)). Prefill is **F16** (its fastest
prefill) where F16 fits a board, else the top precision that runs (tagged). Decode and Fits are
**Q4_K_M** (the fastest stream / smallest fit); PPL Δ is NPU−CPU on the same GGUF.

| Model | Params | Prefill NPU t/s, pp512->2048 | ×CPU, pp512->2048 | Decode Q4 t/s | Fits Q4, GB | NPU−CPU PPL Δ |
|---|---:|---:|---:|---:|---:|---:|
| SmolVLM2-2.2B (llama+SigLIP) | 1.8B | 99.0 -> 52.6 | 3.5x -> 1.9x | 14.5 | 1.0 | −0.07% |
| Llama-3.2-3B | 3.2B | 61.8 -> 45.5 | 3.4x -> 2.6x | 7.9 | 1.9 | +0.02% |
| Ministral-3-3B | 3.4B | 57.6 -> 39.8 | 3.2x -> 2.4x | 7.7 | 2.0 | −0.04% |
| Phi-4-mini | 3.8B | 55.4 -> 39.8 | 3.4x -> 2.6x | 7.0 | 2.3 | −0.12% |
| Ministral-3-8B | 8.5B | 27.4 -> 21.4 | 3.8x -> 3.1x | 3.8 | 4.8 | −0.01% |
| Qwen3.5-9B | 9.0B | 25.9 -> 24.9 | 3.6x -> 3.5x | 3.6 | 5.3 | +0.05% |
| Gemma-4-12B | 11.9B | 17.4 -> 15.0 | 3.6x -> 3.2x | 2.5 | 6.9 | −0.81% |
| Phi-4 (14B) | 14.7B | 10.3 -> 11.8 (Q4) | 2.9x -> 3.5x | 2.2 | 8.3 | −0.31% |
| DeepSeek-V2-Lite (MoE+MLA) | 15.7B | 26.0 -> 28.4 (Q4) | 1.33x -> 1.53x | 7.8 | 9.7 | −0.42% |
| gpt-oss-20b (MoE) | 20.9B | 22.0 -> 28.1 (MXFP4) | 1.81x -> 2.38x | 7.2 | 11.3 | cosine 0.9998 † |
| Qwen3.6-27B (hybrid) | 27.3B | 5.4 -> 7.8 (Q4) | 3.0x -> 4.4x | 1.1 | 15.9 | −0.26% |

† gpt-oss is a reasoning model, so wikitext PPL is not a valid metric for it; its faithfulness is
carried by the per-matmul cosine against an fp64 reference and by `test-rocket-moe`, not by a greedy
match; see "The accept boundary" and the MoE gate table below for why the greedy comparison has no
resolution here.

**The quantized rows' ×CPU is against an unrepacked CPU.** The NPU arm has to run without
llama.cpp's weight repack, and the repack speeds the CPU arm 1.26-1.71x at pp2048 on these four
models. Against a repacked CPU at pp2048 the NPU leads Phi-4 **3.06x** and DeepSeek-V2-Lite
**1.44x**. It leads gpt-oss-20b **1.90x** and Qwen3.6-27B **3.21x** [HW sweep 2026-09-29, llama.cpp
b11242, governor pinned].
[cpu-repack-baseline.md](cpu-repack-baseline.md) has the method. The F16 rows are unaffected.

The prefill multiple grows with model size, to Qwen3.6-27B's **4.4x at pp2048, the largest here**,
because the CPU baseline degrades faster than the NPU as the matmuls grow. The exceptions are
architectural: the **MoE expert FFNs** of gpt-oss-20b and DeepSeek-V2-Lite route through
`MUL_MAT_ID`, which holds each quantized expert resident as native int8 and deletes the
per-micro-batch host dequant that makes the naive expert offload a loss. It is **default-on** since
2026-08-27 and takes gpt-oss to **2.38x** at pp2048 from 1.15x with the experts on the CPU. The
default claims an expert op only where a residency **pre-flight** can reserve the whole stack before
ingesting it *and* the per-expert GEMM carries enough work to pay for its own dispatch
(`M_e · K · N`, where `M_e` = `n_tokens · n_used / n_expert`), so DeepSeek's 6-of-64 routing over a
2048×1408 expert keeps its experts on the CPU at short prefill and moves them onto the NPU past
~1250 tokens. DeepSeek adds
**MLA** attention (the FA gate accepts DK≠DV and is bit-faithful, but its DeepSeek FA path is not
yet exercised on-device, so attention stays on the CPU here). Qwen3.6-27B's
**Gated-DeltaNet** hybrid keeps its linear-attention layers on the CPU but they do not gate the
win. The two MoE models decode briskly for their size (gpt-oss 7.2, DeepSeek 7.8 t/s) because only
~3.6 B / ~2.4 B params are active per token. Every NPU−CPU PPL Δ sits within its per-run stderr,
the fp16-accumulation prefill is faithful; the absolute PPL of an instruct / reasoning model is not
meaningful (only the delta is). Notes on the table: prefill is F16 unless tagged (Q4 = Q4_K_M,
MXFP4 = the gpt-oss native block-float, all F16-untenable on a 31 GB board); Phi-4 (14B) also runs
Q8_0 at essentially the same prefill; Qwen3.5-9B also ships **IQ4_XS** (decode 4.1 t/s, 4.8 GB,
same NPU path); SmolVLM2's 1.0 GB is its 1.81 B LLM half, plus ~0.8 GB fp16 vision mmproj.

SmolVLM2's **vision half** (SigLIP-SO400M, 729 tokens) also offloads, through clip.cpp: **7.90 ->
6.66 s warm (1.19x)**, projected-embedding cosine **0.99998** vs CPU, a modest win capped by the
graph shattering into 272 CPU↔NPU splits (the generic drop-in, not the resident
`rocket_siglip_encoder`). Needs `MTMD_BACKEND_DEVICE=ROCKET`.

### What tuning buys per model (stock -> recommended, pp2048)

The delta a user captures by following the guide, both arms on the NPU. Stock is llama-bench's
own defaults with the backend loaded: `-ub 512`, and every `ROCKET_*` knob at its default,
including MoE placement on AUTO. Recommended is the model's row in `MODEL-NOTES.md` and
`TUNING.md`. Means over three rotated interleaved passes, memory reset before every arm,
governor pinned [HW sweep 2026-08-28..31, RK1, 600 MHz]. **The recommended config is
per-model, not a universal recipe**, and on two models it is the defaults themselves.

And on **every quant model that fits resident it is residency *without* the `-ub` raise**. Six
of six measured both ways prefer the unstacked form, from 2.2 to 9.0 B, by 5.5-34%. Below 4 B
the model also carries a real non-dequant `-ub 2048` loss, residues 0.79-0.86 under the dqc
mechanism control, that stacking pays before residency arrives.

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
| Qwen3-30B-A3B Q4_K_M (MoE) | defaults, every tuned arm loses | 14.7 | 14.7 | **1.00x** |

One row still shows the stacked form. `Qwen3.5-0.8B`'s class cannot resolve a ratio of this
size at any affordable pass count, so its unstacked cell was not run. Every other resident row
is the unstacked recipe, `Gemma-4-12B` included. That model places only part of its weights,
which was the case expected to break the rule, and it does not. The unstacked form there also
buys residency, 83-87% against the stacked recipe's 73-74%. `-b 2048 -ub 2048` spends that RAM
on compute buffers before the reserve floor stops placement.

The knobs, in full. Here `-ub 2048` means `-b 2048 -ub 2048`. The `QUANT_RESIDENT` column is
`ROCKET_QUANT_RESIDENT=auto`, which dequantizes each weight once and holds it resident as fp16,
and needs the fp16 image in RAM. The `F16_RESIDENT` column is `ROCKET_F16_RESIDENT=auto`.

The Qwen3.5-0.8B **F16** unit is omitted, because its knob reaches only 24 weights at the
K this model uses. That unit's spread across processes is ~8-11%, unique to the 0.8B class on this
board. It exceeds the effect, so the ratio does not resolve at affordable pass count. Raw rows
and per-pass ratios: [data/tuning-matrix.md](data/tuning-matrix.md).

### ASR (Whisper encoder, whisper.cpp via ggml-rocket)

The NPU's job in ASR is the **encoder** (the decoder is autoregressive, M=1 GEMV, CPU on both).
Encoder latency, `whisper-bench` encode-only, warm; transcripts byte-identical CPU-vs-NPU (WER 0),
encoder-output cosine 0.9998.

| Model | enc d_model / layers | CPU -> NPU | speedup |
|---|---|---:|---:|
| tiny.en | 384 / 4 | 697.7 -> 591.9 ms | 1.18x |
| base.en | 512 / 6 | 1580.0 -> 1223.4 ms | 1.29x |
| small.en | 768 / 12 | 5711.5 -> 3718.9 ms | 1.54x |
| medium.en | 1024 / 24 | 19268.2 -> 10444.9 ms | 1.84x |
| large-v3 | 1280 / 32 | 36399.9 -> 17013.5 ms | 2.14x |
| large-v3-turbo | 1280 / 32 | 32957.2 -> 15544.1 ms | 2.12x |

The win grows monotonically with the encoder (matmul work ~d_model² vs ~linear host packing).
**large-v3-turbo**, the full 32-layer encoder with a 4-layer decoder (~6x cheaper per step), is
where the NPU-accelerated encoder carries the most of a real transcription.

### STT beyond Whisper (transcribe.cpp via ggml-rocket)

Beyond OpenAI Whisper, the same `.so` drops into **transcribe.cpp** (a ggml-based multi-STT host)
and offloads a range of speech models. The unifying result: **the NPU offloads encoders, not
autoregressive decode**, so the win tracks how encode-heavy the model is, and (on long audio) how
much of decode is offloadable prefill vs per-token M=1 steps. Single fresh-process runs, warm,
A76-pinned, Q8_0, on a hard 120 s two-speaker clip. [HW sweep, A/B]

| Model | encoder + decoder | NPU × (120 s) | rt (NPU) | best at |
|---|---|---:|---:|---|
| SenseVoice-small 234M | SAN-M + CTC (single-pass) | 1.26x | 6.2x | lowest latency |
| Fun-ASR-nano 800M | SAN-M + Qwen3-0.6B AR | 1.15x | 1.75x | 31-language coverage |
| Granite-Speech-2B (base) | conformer + LLM cross-attn | 1.68x | 1.76x | fast clean transcript |
| Voxtral-mini 3B | Whisper-lg-v3 + Ministral-3B AR | 1.57x | 0.55x | best transcript + translation |
| MOSS 0.9B | Whisper-Med + Qwen3-0.6B AR | 1.08x | 0.42x | best diarization (33 seg + timestamps) |

The **cross-attention decoder (Granite)** offloads decode best (its per-step cross-attn over the
encoder is a large-K matmul, ~1.9x); a **big decoder-only over long audio (Voxtral 3B)** offloads
its 1500-token audio prefill (decode 1.46x); a **small decoder-only (MOSS/FunASR 0.6B)** keeps
generation on the CPU (M=1), so it barely moves despite a 1.6x encode.

### Detection (SSD-MobileDet, tflite-rocket)

A single inference is bound by host work, so its latency follows the host path and the CPU
governor. The NPU's larger value is throughput under a multi-camera pool, Frigate's regime.
Accuracy is COCO mAP.

| Metric | Value |
|---|---|
| COCO mAP@[.5:.95] | 0.3321 NPU vs 0.3318 CPU (parity) |
| Single-stream latency, warm | 56 ms with the governor pinned, 76 ms on `ondemand` (2026-09-28) |
| Multi-camera pool, P=1->4 | 3.20 -> 9.55 detection_fps (2.98x at P=4), measured at a ~336 ms single stream |

## Method

- **Board / operating point.** RK3588 (Turing RK1: 4xA76 + 4xA55, shared LPDDR). NPU at
  **600 MHz** (the clock patch, [clock.md](clock.md)); it boots at 200 MHz and rides up under
  load, so the clock is loaded via `rocket_npu_clk_hz=600000000` and confirmed at 600 MHz
  *during* the run (an idle sample between reps reads 200). The CPU baseline is the same
  llama.cpp binary with the backend unloaded, 8 threads, unpinned, F16 decode is
  LPDDR-bandwidth-bound, so the A55 cores add bandwidth and confining the process to the A76
  cluster starves it ([decode-gemv.md](decode-gemv.md)).
- **Warm discipline.** Discard the first (cold) run, the clock parks at idle and a cold read
  is ~15% low. Figures are warm medians (`llama-bench -r 2` plus a discarded warmup).
- **Flags.** F16 at llama.cpp defaults (`-ub 512`); quantized GGUFs at **`-b 2048 -ub 2048`**
  (a quantized GGUF re-dequantizes to fp16 per micro-batch, so the default `-ub 512` ~halves
  prefill, [quant-prefill-microbatch.md](quant-prefill-microbatch.md)).
- **Three tables, because one "tokens/s" misleads.** The NPU is a prefill / batched-GEMM
  engine; decode (M=1 GEMV) stays on the CPU on both backends ([decode-gemv.md](decode-gemv.md)).
  So every model reports **prefill** (the NPU's job), **decode** (the CPU-bound equalizer), and
  an **interactive** decomposition into time-to-first-token (set by prefill) + streaming rate
  (set by decode). Two orthogonal levers: **the NPU shortens the first-token wait; quantization
  speeds the stream.**
- **Faithfulness.** Every speed number is paired with a correctness check. For LLMs it is a
  **differential perplexity**: the same GGUF run through the NPU prefill vs the CPU, so the
  NPU−CPU delta isolates the fp16-accumulation fidelity of the NPU matmul (the absolute PPL of
  an instruct model is not meaningful; the delta is). Detection uses COCO mAP; the Whisper /
  SigLIP encoders use output cosine.
- **Reproducibility.** Generator and raw `llama-bench` output under [data/](data/) for most
  models; a few runs (gpt-oss-20b, the DeepSeek `ROCKET_MOE=1` re-bench, and the Ministral-3-8B
  perplexity) are summarized inline in this doc rather than archived as separate raw files. Models
  are stock GGUFs (provenance per model); F16 is `llama-quantize <bf16> <f16> F16` from the
  published BF16.

## LLM (llama.cpp via ggml-rocket)

### Ministral-3-8B-Instruct-2512

Mistral's December-2025 8B (arch `mistral3`; GQA, interleaved sliding-window attention),
8.49 B params. GGUFs from `unsloth/Ministral-3-8B-Instruct-2512-GGUF`; F16 derived from the
published BF16. [HW sweep, 600 MHz, 2026-07-01].

**Prefill (prompt processing), t/s (the NPU's job).**

| test | F16 CPU->NPU | Q8_0 CPU->NPU | Q4_K_M CPU->NPU |
|---|---|---|---|
| pp512  | 7.2 -> **27.4** (3.8x) | 6.9 -> 17.4 (2.5x) | 6.5 -> 17.0 (2.6x) |
| pp1024 | 7.0 -> **24.4** (3.5x) | 6.7 -> 19.3 (2.9x) | 6.4 -> 19.1 (3.0x) |
| pp2048 | 6.8 -> **21.4** (3.1x) | 6.4 -> 17.7 (2.8x) | 6.0 -> 17.4 (2.9x) |

The NPU carries prefill, ~3x on F16, ~2.5-3x on quants. F16 prefills fastest (27 t/s): a
quantized GGUF re-dequantizes to fp16, so quantization does not buy prefill speed at this
operating point (the datatype-independent dispatch floor, [not-mac-bound.md](not-mac-bound.md)).

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).**

| | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| tg64 CPU | 1.37 | 2.46 | 3.76 |
| tg64 NPU | 1.43 | 2.47 | 3.83 |

NPU decode matches the CPU, and stays off the NPU ([decode-gemv.md](decode-gemv.md)). It scales with the
quant, not the backend: F16 1.4 -> Q4_K_M 3.8 t/s (2.8x). Quantization is the only decode lever.

**Interactive.** TTFT and streaming derived from the measured pp/tg rates; the
combined `pp2048+tg128` llama-bench point (F16 NPU 10.3, Q4_K_M NPU 12.3 t/s) validates the
decomposition to ~10%.

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | F16 CPU | 299 s | 1.4 t/s | 445 s |
| | F16 + NPU | **96 s** | 1.4 t/s | 236 s |
| | Q4_K_M + NPU | 118 s | 3.8 t/s | **170 s** |
| chat, 128-tok prompt -> 400 out | F16 + NPU | 5 s | 1.4 t/s | 285 s |
| | Q4_K_M + NPU | 8 s | 3.8 t/s | **112 s** |

On a long prompt the NPU cuts first-token wait ~3x (F16 299->96 s) and nearly halves the turn;
on a short-prompt/long-reply chat the turn is decode-bound, so the NPU barely moves it and
*quantization* is what helps (F16 285 -> Q4_K_M 112 s).

**Faithfulness: differential perplexity, wikitext test, 12 chunks (same GGUF, NPU prefill vs
CPU).**

| | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| F16    | 6.8621 | 6.8615 | −0.01% |
| Q4_K_M | 6.9626 | 6.9597 | −0.04% |

(±0.43 per-chunk stderr, the NPU−CPU delta is ~100x smaller than the noise.) The NPU prefill
matches the CPU to within 0.05% on both models, so the fp16-accumulation path trades no
measurable accuracy for the prefill speedup. The Q4_K_M−F16 gap (6.96 vs 6.86) is the expected
quantization cost, present equally on both backends.

**Verdict.** For a 16 GB+ board on batch / RAG workloads, **F16 + NPU** (fastest prefill). For
most boards and mixed interactive use, **Q4_K_M + NPU**; it keeps ~2.9x prefill *and* a
2.8x-faster stream in 4.83 GB (fits an 8 GB board), and is the faster *effective* experience on
a mixed turn (combined pp2048+tg128: Q4_K_M 12.3 > F16 10.3 t/s). Q8_0 (9 GB) is the
near-lossless middle.

### gpt-oss-20b (MXFP4, MoE)

OpenAI's `gpt-oss` 20B (arch `gpt-oss`; 24 layers, 32 experts with 4 active per token, ~3.6 B
active of 20.9 B, GQA 64/8 heads, **alternating** sliding-window (128) and full-attention layers,
and a learned **attention sink** per head). Distributed **only** in native **MXFP4** (the 4-bit
block-float the model was trained to emit); GGUF from `ggml-org/gpt-oss-20b-GGUF` (11.27 GiB).
Raw data: [data/gpt-oss-20b.md](data/gpt-oss-20b.md). [HW sweep, 600 MHz, 2026-07-14; one board,
one session, clock pinned; every baseline below re-measured alongside the thing under test.]

This is the MoE model, and the whole question it asks is what to do with the expert FFNs. llama.cpp
routes them through `GGML_OP_MUL_MAT_ID`, and they are the bulk of prefill FLOPs (~75%); the dense
attention projections and `lm_head` are the rest.

> **`-ub` costs this model something, but it does not collapse it, and mixing `-ub` between arms
> is the classic trap.** The headline table is `-b 2048 -ub 2048`, which is the setting to run the
> expert route at. Two mechanisms make a smaller micro-batch cost it, and **neither is the
> per-expert dequant**, native-quant ingests each expert once and deletes that. (a) The **dense**
> MXFP4 weights are not in the expert cache and still re-dequantize per micro-batch, so `-ub 512`
> decodes them four times over. (b) Each expert receives only `n_tokens · n_used / n_expert` rows,
> 64 at `-ub 512` vs 256 at `-ub 2048`, while the per-expert dispatch, gather, scatter and bucket
> padding around the GEMM stay flat.
>
> **Now measured at `-ub 512`** [HW sweep 2026-08-27, 600 MHz]: **22.2 (21.7-22.7) / 22.5 (22.4-22.8)**
> at pp512 / pp2048 against **14.13 / 13.53** with the experts on the CPU, about **1.6x / 1.7x**,
> against 22.0 / 28.1 at `-ub 2048`. So the two mechanisms together cost ~20% at pp2048 and nothing at pp512, and the
> route is a clear win at the llama.cpp default micro-batch. **Do not quote the "`-ub 512` collapses
> MoE" figure at it**; that ~0.42x belongs to the fp16 streaming route, whose per-micro-batch
> dequant is exactly what a smaller `-ub` multiplies and which the native route does not pay.
> Meanwhile `-ub 2048` makes the *dense* graph slower here, and the CPU does not care either way,
> so there is no single best `-ub`, only a per-configuration one. An earlier `-ub 512` NPU-default
> figure of 13.11 t/s at pp2048 circulated as a baseline and caused a nonexistent "regression" to be
> chased: the like-for-like `-ub 2048` figure was always ~11.

**Prefill (prompt processing), t/s (MXFP4, `-b 2048 -ub 2048`).** The expert route is
**default-on** as of 2026-08-27, behind a residency pre-flight; `ROCKET_MOE=1` additionally claims
what the pre-flight declines. [HW sweep 2026-08-27, 600 MHz, governor `performance`, one board, one
`.so`, llama.cpp 171974745.]

| test | CPU | NPU, `ROCKET_MOE=0` | **NPU, default** | NPU, `ROCKET_MOE=1` |
|---|---:|---:|---:|---:|
| pp512  | 12.17 | 14.08 (1.16x) | **22.0 (1.81x)** | 24.81 (2.04x) |
| pp2048 | 11.80 | 13.62 (1.15x) | **28.1 (2.38x)** | 34.01 (2.88x) |

**The default column is a mean of six runs, and that is deliberate.** Its spread is 20.5-22.8 at
pp512 and 26.9-28.7 at pp2048, at *identical* placement every time (63 of 72 stacks, 1656 experts,
0 streamed). One run landed at the bottom of it directly after a rebuild and read as a 10% regression;
the discriminator was the `ROCKET_MOE=0` control, which returned **13.63 at pp2048 on every binary and
every repetition**, so the board had not drifted and the spread belongs to this arm. A single sample
of the offloaded arm is not a measurement of it; the control is what says so.

The earlier session's matrix, CPU 13.09 / 12.40, experts-on-CPU 14.11 / 13.89, experts-on-NPU
17.57 / 26.78 at pp512 / pp2048 [HW sweep 2026-07-14, llama.cpp a646006f0], is a **different
llama.cpp and a different driver build**, so the two are not row-comparable; each is internally
consistent and the ratios agree in direction.

**The expert route is the model's whole story.** Holding the routed experts on the NPU as native int8
is **2.06x the experts-on-CPU arm** and **2.38x the CPU** at pp2048, and it wins at every prefill
length. CPU prefill is itself unusually fast here (~12 t/s, vs ~7 for a dense 8B) because only ~3.6 B
params are active per token, so that is against a strong baseline.

**The default is deliberately not the peak.** The pre-flight reserves a whole expert stack before
claiming its op, and reserving charges all `n_expert` experts where the lazy admission it front-runs
charges only the ~82% the router exercises, so the default takes 63 of 72 stacks (100% resident, **0
streamed**) where `ROCKET_MOE=1` takes all 72. That costs 8% at pp512 and 16% at pp2048. What it buys
is the **sign**: on a board that does not fit the stack the default is still above the experts-on-CPU
baseline, where `ROCKET_MOE=1` is **below** it, at an induced 12 GB budget the forced arm falls to
52% resident and reads **13.63 at pp512 against 14.08**, while the default's 30 fully-resident stacks
read **17.02 (1.21x)**. A user with RAM to spare buys the peak back with `ROCKET_MOE_CACHE_MB`, which
keeps the zero-streamed property, rather than with `ROCKET_MOE=1`, which does not.

**"Any streaming is a cliff" is too strong**, and it was the premise the pre-flight was designed
against. **91% resident was the fastest arm measured.** The crossover sits between **52% and 82%**;
no run has bracketed it more tightly.

### The accept boundary, and why the per-expert row count is the wrong handle

The gate that keeps a small expert GEMM on the CPU first sat at the tile granule, 64 rows, the
smallest tile the resident path can run, placed there **by mechanism rather than measured as the
edge**. Mapping `M_e` = `n_tokens · n_used / n_expert` against `default ÷ ROCKET_MOE=0` over
`-p 512,768,1024,1536,2048` on both MoE models says the granule is not where the sign changes, and
says something stronger: **`M_e` is not a quantity a single threshold can be put on.**
[HW sweep 2026-08-27, RK1, 600 MHz pinned, governor `performance`, `-b 2048 -ub 2048`, `-r 3`,
one board and one `.so`; 100% resident and 0 streamed in every offloaded cell, so nothing here is
a residency effect.]

| | `M_e` | work per dispatch | `ROCKET_MOE=0` | default | ratio |
|---|---:|---:|---:|---:|---:|
| **gpt-oss-20b**, 4-of-32, expert GEMM 2880×2880 | 64 | 5.31e8 | 14.08 | 23.08 | **1.64x** |
| | 96 | 7.96e8 | 14.22 | 25.13 | **1.77x** |
| | 128 | 1.06e9 | 14.10 | 26.92 | **1.91x** |
| | 192 | 1.59e9 | 13.88 | 28.38 | **2.05x** |
| | 256 | 2.12e9 | 13.64 | 28.30 | **2.08x** |
| **DeepSeek-V2-Lite**, 6-of-64, expert GEMM 2048×1408 | 48 | 1.38e8 | 25.94 | 26.03 | 1.00 (declined) |
| | 72 | 2.08e8 | 26.35 | 24.86 | **0.94x** |
| | 96 | 2.77e8 | 23.06 | 24.18 | 1.049x *(marginal)* |
| | 144 | 4.15e8 | 22.33 | 27.20 | **1.22x** |
| | 192 | 5.54e8 | 21.60 | 28.37 | **1.31x** |

**How many pairs each cell got is set by its distance from 1.00, not by the cell.** The offloaded
arm varies ~15% run to run on DeepSeek, so a cell a few percent from parity needs repeats and one
at 1.6-2.1x does not. `M_e` = 72 and 144 are means of **four** adjacent pairs; `M_e` = 96 is
**eight**, because it is the cell that decides where the threshold goes; `M_e` = 192 is two, taken
either side of the rebuild; every gpt-oss cell is a single pair, its margin an order of magnitude
outside the same spread.

**The `M_e` = 96 cell is why that budget was spent.** One of its eight pairs read **0.936** and the
other seven read **1.019-1.076**; all eight placement-identical (4767 resident, 0 streamed, the
same ingest profile), so nothing but variance separates them and there is no mechanistic ground to
drop the low one. Pooled it is **1.049**; over the seven repeats alone, 1.065. Either way it is
marginal, and marginal is the verdict that matters.

The `M_e` = 48 row is the null cell: the shipped floor declines it, so both arms are the same
placement, and they agree to 0.3%, which is what says the harness is measuring the gate and not
the weather.

**gpt-oss wins 1.64x at `M_e` = 64 while DeepSeek loses 5-6% at `M_e` = 72**: four pairs, every one
under 1.00, none above 0.970, so any row floor low enough to admit the first admits the second. No
value of a row floor separates them. And the loss at `M_e` = 72 belongs to the **route**, not the
gate: Auto admits every DeepSeek stack at every prefill length, so `ROCKET_MOE=1` is the same
placement there, and it reads 25.55 against the same 26.35 control.

**What the floor is actually amortising is a per-dispatch cost.** One gather, one per-`(row, K-group)`
activation quantize, one submit, one fence, one scatter, per `(op, expert-with-rows)`. The work a
dispatch carries is `M_e · K · N`, so the NPU beats the CPU on that expert when
`fixed_dispatch < M_e · K · N · (1/rate_cpu − 1/rate_npu)`, a threshold on the **product**, for
which `M_e` is a proxy only while `K · N` is held constant. It is not: gpt-oss's expert GEMM is
**2.88x larger** than DeepSeek's at the same row count, which is the whole of the discrepancy above.
Against that quantity the cells sort with no overlap on two architectures at once: the one clear
loser sits at **2.08e8** MACs, the boundary cell at **2.77e8** is a marginal **1.049x** (eight pairs,
0.94-1.08) that does not repay its own ingest, and every materially winning cell is at or over
**4.15e8**. The shipped threshold is the geometric midpoint of that last gap, **340 mega-MACs**.

### The third architecture, and the threshold declines it correctly

The number above is fitted on two models, and gpt-oss contributed nothing to where it sits; its
smallest reachable cell is 5.31e8, an order of magnitude clear of 3.4e8, so the bracket is
DeepSeek's alone. **Qwen3-30B-A3B** is the case that tests it from outside: 128 experts, 8 active,
`n_embd` 2048 and `moe_intermediate` 768, so `K · N` = **1.573e6**, 5.3x smaller than gpt-oss's
and 2.7x smaller than DeepSeek's. At `-ub` 2048 it reaches `M_e` = 128 and **201 mega-MACs**, under
the floor, and clearing 340 would need `M_e` >= 217, i.e. more than 3472 tokens in one micro-batch.
So the shipped default declines its experts at `-ub` 2048 and below, and **accepts at `-ub` 4096**,
which this board runs. The accept side is measured below.

[HW sweep 2026-08-27, RK1, 600 MHz pinned, governor `performance`, `-b/-ub` as shown, `-r 3`,
three process-pairs with arms alternated, warm-up process discarded, page cache dropped first.]

| cell | work/dispatch | arm | t/s | ratio |
|---|---:|---|---:|---:|
| `ub` 2048, `M_e` 128 | 201 MMAC | `ROCKET_MOE=0` | 13.38 / 13.32 / 13.31 | n/a |
| | | **shipped default** | 13.32 / 13.34 | **1.000x** |
| | | `ROCKET_MOE=1` | 11.31 / 10.98 / 11.08 | **0.834x** |
| `ub` 1024, `M_e` 64 | 101 MMAC | `ROCKET_MOE=0` | 14.62 | n/a |
| | | `ROCKET_MOE=1` | 9.66 | **0.661x** |

The default is **1.000x** its own `ROCKET_MOE=0` control and prints no MoE line at all, not a
small ingest, zero, which is the null cell that says the gate is what is being measured. The
control arm's spread across three processes is **0.5%**, the forced arm's 3.0%, so the 17% deficit
is far outside both and the per-pair ratios barely move: 0.845 / 0.824 / 0.832.

**The confound, and why it does not carry the conclusion.** 29 of this model's 30.5 B parameters
are experts, so it cannot be held resident on a 31 GB board at any budget: the int8 ingest stops at
9539 of 15277 experts, **62.4%**, and the rest streams. A partial-residency arm is strictly worse
than a resident one, so a forced arm reading *above* 1.05 would have falsified the threshold
outright, while one reading below it cannot by itself separate "the work per dispatch is too small"
from "it cannot be held". Two comparisons separate them.

**The residency-matched pair is the direct one.** The `ub` 1024 cell holds residency fixed at
**62.5%** against the `ub` 2048 cell's 62.4%, 9548 experts against 9539, and halves the work per
dispatch. The ratio falls from 0.834 to **0.661**. With residency constant to a tenth of a percent,
work per dispatch alone moves the result by 0.17, which is the term the floor is placed on behaving
exactly as the cost model says it should.

**And the residency axis itself is too shallow to explain the deficit.** `ROCKET_MOE_CACHE_MB`
raises residency 62.4% -> 65.7% -> **70.8%** and the ratio moves 0.834 -> 0.856 -> **0.863**: 8.3
points of residency buy 0.029 of ratio. Extrapolated linearly, full residency would read **0.965x**
and reaching parity would need **110%** residency **[extrapolated from two points, 29 points beyond
the data, indicative, not measured]**. The forced arm loses at every residency this board can
reach, and on that trend at every residency at all.

So the threshold's decision is right on a third architecture, and right for the reason it was
derived from. What this does *not* establish is the number: 201 MMAC is far enough below 340 that
it tests the floor's **sign**, not its **position**, and the position is still fitted on
DeepSeek's bracket alone. A model landing between 277 and 415 MMAC is what would move it.

**The rival the map sets aside.** The bucket ladder rounds `M_e` up (64, 96, 128, 192, 256…), so
DeepSeek's points alternate 33% padding (72 -> 96, 144 -> 192) with none (96 -> 96, 192 -> 192), and
padding predicts a **sawtooth**, each step onto an unpadded rung steeper than the step off one. The
measured curve is monotone and **saturating** instead: per-row slopes of **0.0042, 0.0037, 0.0018**
across `M_e` 72 -> 96 -> 144 -> 192, so one step agrees with the sawtooth and the next contradicts it,
while a falling slope is exactly what amortising a fixed per-dispatch cost looks like. Padding is
capped at ~33% by construction and is not the term that flips the sign.

**The floor is a materiality bar, not `> 1.00`, and the ingest is why.** The one-time expert ingest
(~32 s per `llama_context` on DeepSeek, ~36 s on gpt-oss) is charged **only if the gate accepts**, so
the gate is the one place it can be avoided. An offload at ratio `r` saves `1 − 1/r` of prefill wall,
so the ingest breaks even after:

| cell | ratio | saves | breaks even after |
|---|---:|---:|---:|
| DeepSeek `M_e` = 96 | 1.05x | 4.8% | ~16 300 tokens of prefill at that shape |
| DeepSeek `M_e` = 144 | 1.22x | 18.0% | ~4 800 tokens |
| DeepSeek `M_e` = 192 | 1.31x | 23.7% | ~3 800 tokens |
| gpt-oss `M_e` = 64 | 1.64x | 39.0% | ~2 100 tokens |

A cell a few percent above parity therefore costs a short session more than it saves, which is what
makes a bar above 1.00 the correct default rather than a cautious one.

**Scope.** The form is derived; the threshold is **fitted on two architectures** and should be read as
a measured boundary rather than a bound. What the form does not carry is `rate_cpu`, gpt-oss at
5.31e8 wins 1.64x where DeepSeek at 5.54e8 wins 1.31x, same work per dispatch and a different margin,
because their CPU kernels differ (MXFP4 against Q4_K). A third MoE architecture is where the number
gets tested, and it will move placement there: a 128-expert/8-used model with a 2048×768 expert sits
at 201 MMAC at `M_e` = 128, which a row floor accepts and this one declines.

**Why it works, and why the obvious alternative does not.** A quantized expert on the *streaming*
route is dequantized to fp16 on the host **every micro-batch**, and that decode is **independent of
the row count**; it decodes the whole `[2880,2880]` weight whatever the router gave that expert.
Measured: **75 ms per expert, per micro-batch**, and a prefill touches ~1580 of them, so streaming
burns **~119 s per prefill** before any arithmetic. That is why the fp16 expert route (`ROCKET_MOE=1
ROCKET_MOE_NATIVE=0`) measures **4.59 / 10.18** at pp512 / pp2048, *worse than leaving the experts on
the CPU*. The native-quant route ingests each expert **once** into resident int8 codes and deletes
that tax entirely. **Quantization here buys residency; residency buys the speed**: the int8 GEMM
itself moves *more* bytes than fp16 would (see [not-mac-bound.md](not-mac-bound.md)).

**Residency is the route, not a nice-to-have.** The win is conditional on nearly all the experts
fitting: 99% resident gives the numbers above, but at **82%** resident the same route reads **12.19**
at pp512, *below* the 14.11 you get by leaving the experts on the CPU. The cliff is the cost
structure, not a threshold effect: a streamed expert keeps paying that M-independent 75 ms while a
resident one pays a GEMM that shrinks with M, so the streamed remainder's share of the wall clock
**grows as the prefill shortens** (12% of pp2048, 43% of pp512). On this 31 GiB board gpt-oss reaches
99% (~14 GB of int8 experts alongside its 11.3 GiB GGUF, which must stay mapped for CPU decode);
a smaller board will not, and the backend warns when it lands short.

**The one-time cost.** The ingest is lazy and lands inside the first prefill: **~70 s** for ~1750
experts (42 ms each, 21 s of MXFP4->int8 decode, 50 s of NPU-BO scatter). It is paid **per
`llama_context`**, so `llama-bench`, which builds a fresh one per test row, pays it per row; a
long-running host pays it once. It does **not** contaminate the numbers: llama-bench's warmup is a
full prompt run, so the ingest lands there.

**Attention stays on the CPU here, and that is a correctness requirement.** gpt-oss carries an
**attention sink**, a learned per-head logit that joins the softmax denominator, and the NPU
FLASH_ATTN handler has no sink term. The offload is declined for such ops. (It had been silently
*accepted*, computing a sink-less softmax past the `n_kv` floor of 1024; declining it is both correct
and **+26%** at pp2048, since it was spending real time on the wrong answer. See
[attention-offload-crossover.md](attention-offload-crossover.md).)

**Decode (generation), t/s (CPU-bound on both).**

| | MXFP4 |
|---|---|
| tg64 CPU | 7.15 |
| tg64 NPU | 7.16 |

Decode is off the NPU as always; the MoE's small active-parameter count makes it comparatively
brisk (7.2 t/s, roughly Ministral's Q4_K_M rate at ~half the bytes touched).

**Interactive.** Both levers are flat for this model. The NPU shortens time-to-first-token only ~4%
(2048-tok prompt: CPU 162 s -> NPU 156 s), because prefill itself is barely accelerated; and there is
no quant lever, since MXFP4 is the only distribution. The combined `pp2048+tg128` point (CPU 11.61,
NPU 12.03 t/s) confirms the ~1.04x ceiling end to end.

**Faithfulness, and the trap in measuring it.**

`gpt-oss` is a harmony **reasoning** model, so absolute wikitext PPL is meaningless (its own F16
reference is ~385). Differential PPL, NPU prefill against CPU on the same text, cannot carry the
gate alone. It reads **+1.0%** against a ±5.7% per-run stderr on a configuration whose attention was
outright wrong: comfortably "within noise".

> **That +1.0% was a bug, not noise.** The configuration ran the NPU FLASH_ATTN offload on
> gpt-oss's full-attention layers, and the handler dropped the model's **attention sinks**
> (`src[4]`) silently, computing a sink-less softmax: *a different attention*. A wrong-but-plausible
> attention still produces fluent text and a plausible perplexity, so the metric cannot tell a
> **wrong function** from run-to-run variance. **A faithfulness metric that cannot distinguish those
> is not a faithfulness metric.** The gate declines such ops, which is also +26% at pp2048: the
> offload was paying to compute the wrong answer.

Faithfulness is gated the way [MODEL-NOTES.md](../MODEL-NOTES.md) prescribes for a reasoning
model, and the prompt is a real **~1400-token** passage chosen to clear the FA `n_kv` floor of 1024,
so the gate reaches the regime where that class of bug lives. A short prompt passes with such a bug
live.

| gate | result |
|---|---|
| **per-matmul cosine** vs an fp64 CPU reference, on **real weights and real activations** (one expert per `MUL_MAT_ID` op, rotating across every layer / projection / expert) | **mean 0.999821, min 0.998980** over 50 expert GEMMs under `ROCKET_MOE=1`; **mean 0.999815, min 0.998976** over 55 on the **shipped** placement [2026-08-27], the two placements are numerically the same |
| **greedy match** vs the CPU (`--temp 0`, same seed, 48 tokens) | *see below; this gate has no resolution* |
| synthetic route gate (`test-rocket-moe`, 7/7, outlier-channel activations) | MXFP4 0.9999 / Q4_K 0.9999 / Q8_0 0.9999 |
| **differential PPL on the shipped placement**, DeepSeek-V2-Lite, `-c 2048` (`M_e` = 192, route active) | **−0.121% against the experts-on-CPU arm**, 8 paired chunks, 1.20 se, indistinguishable from zero |

The **cosine** is the number that matters for the int8 question: it is above the synthetic
prediction (0.9994) and far above the **0.98** that already proved token-identical for
int4+Hadamard, so int8 **activations** survive the real outlier-channel distribution with no
Hadamard rotation.

**The greedy row has no resolution, and why is worth keeping.** On one prompt it reads "the
native-quant arm diverges at token ~35, no earlier and no worse than the experts-on-CPU path, which
diverges identically", and that equality reads as evidence the expert route adds nothing. On the
shipped placement the two arms diverge at **word 16 and word 10**: they do not track each other, and
one prompt could never have shown that they do. The
mechanism says why, a 0.9998 cosine is excellent and is still enough to flip an argmax wherever the
top two logits are close, at a position nothing controls. **"Which arm flips first" is a weighted
coin, not a faithfulness measure.** What the greedy comparison can still see is whether the text
stays coherent (it does, on every arm) and whether divergence is *early and large* rather than early
and small. The quantitative legs, the cosine and the differential PPL, are what carry this.

**The differential PPL is the leg that gates composition, and it must be read paired.** The absolute
error bar at 8 chunks is ±2.5% and resolves nothing; pairing on the same chunks cancels the variance
both arms share and gives ~0.10% in PPL terms. Isolating the expert route means differencing the two
**NPU** arms, not the default against the CPU; both carry the same dense fp16 offload, and the
dense path is itself −0.296% (3.74 se, all eight chunks negative), which would otherwise be credited
to the experts.

**Verdict.** The `MUL_MAT_ID` handler closes the op-coverage gap, and with the experts held resident on
the NPU as **native int8** it is a **2.38x prefill win** at pp2048 (1.81x at pp512; it wins at every
length on this model; eligibility on another is per architecture, see "The accept boundary"). The **fp16 streaming** expert route still loses (4.59 / 10.18), and that is the finding the
earlier verdict recorded: the datatype was never the blocker (MXFP4 dequants fine); the blocker is that
streaming per-expert dequant costs more than the small-`M_e` GEMM saves, which the CPU's fused quant
kernel sidesteps. **The native-quant route removes that dequant entirely, which is the whole win**,
quantization here buys residency, and residency buys the speed.

`ROCKET_MOE` is **default-on** as of 2026-08-27. What made it opt-in was that its sign depended on the
host's memory, and the **residency pre-flight** removes that dependence rather than documenting it: an
expert stack's resident cost is knowable from its tensor alone, so `supports_op` reserves the whole
stack before the first ingest and leaves on the CPU what it cannot reserve. The result is bounded
below by the experts-on-CPU baseline at every budget. Two things the flip cost, both measured rather
than assumed: the reservation is more conservative than the lazy admission it front-runs, so the
default gives up **6-13% at pp512 and 18-21% at pp2048** against `ROCKET_MOE=1` on a board that
nearly fits, most of which `ROCKET_MOE_CACHE_MB` buys back without giving up the sign guarantee; and the flip as first
written **regressed DeepSeek-V2-Lite 27% at pp512** at 100% residency, which is what added the
per-expert size floors. Recommended invocation on a 31 GiB board: the default, with `-b 2048 -ub 2048`,
and expect a one-time expert ingest at the first prefill, per `llama_context`: **~36 s** here (9.5-10.4 s MXFP4->int8 decode, 26.0-27.1 s NPU-BO pack, five samples). The pack half is **bytes-bound at ~505 MB/s**, not per-expert, DeepSeek-V2-Lite pays the same ~27 s for 2.9x as many experts holding the same ~14 GB.

### The accept side of the same floor, and the budget above it

`-ub` 4096 gives `M_e` = 256 and **403 mega-MACs**, the first cell this model reaches above the
340 floor, and only **1.19x** it, which makes it a test of the threshold's position rather than a
comfortable one. `M_e` is set by the **micro**-batch, so `-ub` is the only knob that moves the cell
and `-p` need only be large enough to fill it.

It is reachable: `-b 4096 -ub 4096` runs on a 31 GiB board with the 17.28 GiB model loaded, probed
with a single cell before any harness was built.

[HW sweep 2026-08-28, RK1, 600 MHz pinned, governor `performance`, `-p 4096 -b 4096 -ub 4096`,
`-r 2`, three process-triples with the arm order a balanced Latin square, warm-up process
discarded, page cache dropped **before every cell**.]

| arm | t/s, three processes | spread | paired ratios | mean | placement |
|---|---|---:|---|---:|---|
| `ROCKET_MOE=0` | 10.29 / 10.32 / 10.31 | 0.3% | n/a | n/a | n/a |
| **shipped default** | 10.40 / 10.50 / 10.50 | 1.0% | 1.011 / 1.017 / 1.018 | **1.016x** | 79 stacks, 0 streamed |
| `ROCKET_MOE=1` | 9.80 / 9.93 / 10.09 | 2.9% | 0.952 / 0.962 / 0.979 | **0.964x** | 62.6% resident, 5703 streamed |

**The default does not lose, and the two arms are in different regimes.** The default reserves whole
stacks and streams **nothing**: it holds 79 and leaves the rest on the CPU, while the forced arm
claims everything and streams 37% of the experts. Published numbers elsewhere on this model are
forced-regime numbers, and do not describe the default. The default's placement line is **identical across all three cells**: 8818
experts, 14877 MB, 0 streamed, so the three readings are of one placement, which is what dropping
the page cache per cell buys.

**The cost model predicted the forced arm out of sample to 0.4%.** Fitting
`ratio = A / (B + (P/ub)·k)`; one term fixed per prefill, one proportional to the **dispatch
count**, on the two lower cells gives `k` = 0.314A and `B` = 0.885A, and predicts **0.960** here
against **0.964** measured. The three forced points are residency-matched throughout (62.6% against
62.4%), so they are a clean curve on work per dispatch alone: **101 MMAC -> 0.661, 201 -> 0.834,
403 -> 0.964**, saturating rather than linear. The `P` term is not optional: `-ub` 4096 needs
`-p` >= 4096, so the prefill work doubles while the per-prefill constants do not, and a `P`-fixed
extrapolation of the same two points reads ~1.007, wrong by 5%.

**Where the ingest is charged moves this cell by 12%.** `llama-bench`'s warm-up pays the one-time
expert ingest, so the timed reps do not. Turn the warm-up off and the same cell reads **0.934x**
instead of 1.016. Two numbers from one configuration, differing only in where a one-time cost is
charged: state which.

#### The auto budget over-places, and on this model it costs the whole offload

`ROCKET_MOE_CACHE_MB` sets the pre-flight's budget directly, so on the **default** arm it moves the
number of admitted stacks with **zero streaming at every rung**; one regime, one varying quantity.
Ratios against the pooled `ROCKET_MOE=0` mean, which read 10.29 / 10.32 / 10.31 / 10.35 / 10.39 /
10.27 across six processes spanning seven hours, 1.2%, non-monotone, so the arm's own spread rather
than drift:

| budget | stacks | resident | IOVA | ingest | t/s | n | ratio | rep sd | ingest repaid after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 0 | n/a | n/a | n/a | 10.32 | 6 | 1.000x | ±0.01 | n/a |
| 12000 MB | 38 | 7813 MB | 7296 | 20.2 s | 10.65 | 1 | 1.032x | ±0.00 | ~6 700 tokens |
| **18000 MB** | **58** | 11430 MB | 11136 | 30.9 s | **10.76** | 3 | **1.043x** | ±0.03-0.08 | ~7 400 |
| **21000 MB** | **67** | 13167 MB | 12864 | 36.7 s | **10.83** | 4 | **1.049x** | ±0.04-0.08 | ~7 700 |
| auto (24425) | 79 | 14877 MB | 15168 | 43.2 s | 10.47 | 3 | 1.014x | ±0.6 | ~34 000 |
| 28000 MB | 90 | 16716 MB | 17280 | 48.5 s | 10.31 | 1 | 0.999x | ±1.02 | never |

**The shape is a plateau and then a cliff, and the shipped budget is over the edge of it.** The two
middle rungs are not separable, 10.76 against 10.83, 0.65% apart against arms whose own spreads are
1.1% and 1.6%, so 58 and 67 stacks are one plateau, pooled **1.046x over seven processes**. Between
67 and 79 the ratio falls to 1.014.

**The two ends of the ladder are one process each, and the `n` column is load-bearing.** The plateau
is seven processes over two rungs and the auto rung is three, but 12000 and 28000 are **n=1**. A
single process on this board can sit ~10% off the level its own configuration repeats at [HW sweep
2026-08-28], so 28000's 0.999x supports "one process read parity", not "at 90 stacks the offload
buys nothing"; the cliff's existence rests on the plateau-to-auto step, which has the depth, and its
*depth* past 79 stacks does not. Per-rep spread grows monotonically with placement across the whole
ladder (±0.00 -> ±1.02), which is a memory-pressure signature rather than anything about the
offload, and it is also why the two single-process rungs are the least trustworthy: they are the
ones taken where the spread is widest.

Set against the materiality table above, the shipped budget's admission repays after ~34 000
tokens, **twice the 1.05x cell the floor was fitted to exclude**, where the plateau repays after
~7 500, between the 1.22x and 1.31x rows. So the practical recommendation is a **range,
18000-21000 MB**, not a value that has to be hit exactly.

**The budget's units are code-relative, so the number does not transfer to another quant or
another board.** What the knob bounds is not RAM: the admission charge is the int8 codes plus the
per-(channel, K-group) scales plus that expert's GGUF source bytes, so a budget buys strictly less
residency than it names. Across the ladder:

| budget | resident int8 | budget / resident |
|---:|---:|---:|
| 18000 MB | 11430 MB | 1.58 |
| 21000 MB | 13167 MB | 1.60 |
| 24425 MB | 14877 MB | 1.64 |
| 28000 MB | 16716 MB | 1.68 |

**The charge's own form predicts the ratio, which is what makes it usable rather than a curve fit.**
Per expert the charge is `N·K` code bytes + `N·(K/group)·4` scale bytes + the source stride `nb[2]`,
so against the codes alone

```
charge / codes  =  1 + 4/group + source_bits_per_weight/8
```

This GGUF is 18556686912 bytes for 30.5 B parameters = **4.87 bits/weight**, and `K`=2048 auto-picks
group 512, giving **1.617**. The scales term is the small one (0.8% here) and the source term
carries it.

**The route measures the factor directly, and that is the number to compare against** -- the
budget-over-resident column above is only an estimator of it, since a budget is what was *requested*.
The pre-flight prints both halves of the charge:

```
[moe-int8] resident budget reached after 79 expert stacks (24425MB RAM, 15168MB IOVA)
```

RAM is the charge and IOVA is the int8 codes, so their ratio **is** `1 + 4/group + bits/8` with
nothing inferred. Measured that way, on two models and two quant formats:

| model | quant | charged / codes | derived | agreement |
|---|---|---:|---:|---:|
| Qwen3-30B-A3B | Q4_K_M, group 512 | 24425 / 15168 = **1.610** | 1.617 | 0.4% |
| gpt-oss-20b | MXFP4, group 576 | 21433 / 14092 = **1.521** | 1.538 | 1.1% |

[HW sweep 2026-08-28 for the first, RK1; the second from four archived runs
([data/gpt-oss-20b.md](data/gpt-oss-20b.md)) whose six such lines read 1.5209-1.5211.] The derived
value is ~1% high in both cases, so the form is a good sizing rule and not an exact one. Two caveats
on its input explain the direction: 4.87 bits/weight is the whole file's average standing in for the
expert tensors' own rate, and a Q4_K_M GGUF mixes types.

So the same physical placement needs a different number typed in for a different quant of the same
model -- about **2.07** for Q8_0 (8.5 bits/weight) and **1.54** for IQ4_XS (4.25)
**[expected -- derived from bits/weight, not measured]**.

**The factor also decides WHICH budget binds, and on this route it is never the IOVA window.** The
pre-flight holds two budgets: RAM, and `ROCKET_N_THREADS x 3840 MB` of NPU IOVA. It charges each
stack `codes + scales + source` against the first. It charges `codes` alone against the second. So
the window binds first only where

    RAM budget / IOVA budget  >  1 + 4/group + source_bits_per_weight/8

which is the same factor. On `gpt-oss-20b` MXFP4 the right side is **1.538** and the left is
**1.28** at the default five workers. RAM binds, and by a margin no worker count can close, because
raising `ROCKET_N_THREADS` grows only the denominator.

Three runs say so directly. At 5 workers and at 8 the window doubles from 19200 to 30720 MB. The
admitted set is **byte-identical** across all three, at 62 stacks, 24140 MB RAM, 15693 MB IOVA. All
three announce `Bound by RAM` [HW sweep 2026-09-02, RK1, `-b 2048 -ub 2048 -p 2048`, raw in
[data/ro-session/trackd17-moe-preflight.md](data/ro-session/trackd17-moe-preflight.md)]. Every
pre-flight decline recorded for this model says the same. That is 14 occurrences over five distinct
budget and placement points, from an induced 11.7 GB budget to a pinned 25.7 GB one, none naming
the window.

**So the window is not reachable on this route at or above the default worker count.** The
threshold is a property of the source rather than of the board. At five workers it would need
`MemAvailable` above **35.7 GB**, each further worker adding about 5.9 GB to that. On a 32 GiB
board the inequality needs a source under about **2.2 bits per weight**, which no quantization this
route ingests approaches. MXFP4, the densest measured, is 4.25. The exit `Bound by RAM` names,
`ROCKET_MOE_CACHE_MB`, is the only one this route's line ever asks for.

**A model can print no pre-flight line at all, and that is not the pre-flight passing.** Auto
placement checks the tile granule and the per-dispatch work floor **before** it calls the
pre-flight [source-confirmed, `ggml-rocket.cpp`]. `Qwen3-30B-A3B` Q4_K_M at `-b 2048 -ub 2048`
carries 128 experts with 8 used, so a 2048-token micro-batch gives `M_e` = 128 and the work floor
declines it first. Two runs at that shape, at 5 workers and at 8, emit **no `[moe-int8]` line** at
all [HW sweep 2026-09-02, RK1]. The same model reaches the pre-flight at `-ub 4096`, where `M_e`
is 256. So read a missing line as a gate upstream of the budget, and raise `-ub` before concluding
anything about RAM or the window.

**So read the factor off the log when an exact number is wanted, and derive it when sizing a budget
before a run.** The derivation is what makes it transferable across quants and boards; the log is
what pins it for one model. (One archived gpt-oss line reads 1.3926 at 10.70 MB/expert against the
others' 8.07 -- a different expert geometry, not a disagreement. And a run at a *pinned*
`ROCKET_MOE_CACHE_MB` stops at the pin rather than at the natural budget, so take the ratio from a
line whose stack count is limited by RAM, which is what "budget reached ... Bound by RAM" says.)

**It is also why the rule and the recommendation look contradictory.** `MemTotal` minus the whole
expert GGUF minus the reserve reads 31.7 − 17.3 = 14.4 GB on this board, nothing like 21000 MB.
Converted through the measured factor, with a runtime-headroom term, it closes:
`1.610 × (31.7 − 17.3 − 1.2)` = 21.3 GB. **That is a consistent FORM and not an independent prediction** -- the headroom term is
the one free parameter and it was chosen to land on the measured plateau.

**The mechanism is an omission in the charge [hypothesis].** The pre-flight charges each *placed*
expert as int8 codes plus that expert's GGUF source bytes. What it does not charge is the rest of the
expert GGUF, the unplaced experts, which the CPU reads from the same mmap every micro-batch, and
the placed experts' own source pages, which were faulted during the ingest moments earlier. Charging
the **whole** expert GGUF against the anonymous int8 set:

| stacks | anonymous int8 | + whole expert GGUF (16.3-17.8 GB) | vs 31744 MB total |
|---:|---:|---:|---|
| 38 | 7813 | 24.1-25.6 GB | under by 6-7.6 |
| 58 | 11430 | 27.7-29.2 GB | under by 2.5-4 |
| 67 | 13167 | 29.5-31.0 GB | under by 0.8-2.3 |
| **79** | 14877 | **31.2-32.7 GB** | **crosses 31.7** |
| 90 | 16716 | 33.0-34.5 GB | over by 1.3-2.8 |

The crossing lands at 79 stacks, which is where the cliff is. The expert GGUF's own size is bounded
rather than known, 16.3 GB from the pre-flight's per-stack charge at low placement, 17.8 GB at high
placement, because the stacks are not equal-sized, so this is an ordering-and-magnitude fit, not a
coincidence to 0.1%. The 6 GiB reserve exists to absorb exactly this, and the uncharged remainder
here is 7-12 GB.

**The curve is itself the evidence for the load-bearing half.** If the placed experts' source pages
were promptly reclaimable after ingest; they are read once, and this is a prefill-only run, the hot
set would be `anonymous + unplaced GGUF only`, which computes to 20.6 / 21.7 / 22.4 / 22.2 / 22.4 GB
across the five rungs: **flat, crossing nothing, explaining nothing**. A curve exists, so those pages
are not promptly reclaimed. What remains inferred is which pages the kernel actually retains and for
how long; that is not measured here.

It predicts its own scope, and the prediction holds where it can be checked: the defect needs the
experts to dominate the model *and* the GGUF to be large against RAM. gpt-oss-20b is neither
(11.27 GiB), and `ROCKET_MOE_CACHE_MB=28000` there measured **+8.3% / +14.1%** with no degradation
-- but that ladder is **one `-r 3` process per arm**, so those two deltas carry the same n=1 caveat
as this ladder's ends and support a direction rather than a magnitude. What the two ladders jointly
establish is the **sign flip between model classes**, which is a comparison of directions and
survives it. So a `ROCKET_MOE_CACHE_MB` recommendation is scoped to the model class, not to the
board size.

**What this does not settle.** One architecture at one operating point, with the clock, the governor,
`rate_npu` and the model held fixed. Because residency cannot reach 100% on this model on this board,
it still cannot separate the floor's **position** from a residency term, only a model that both
lands near 340 MMAC and fits resident could do that. And 403 MMAC is 1.19x the floor; the tightest
accept-side cell this architecture reaches is `-ub` 3584 -> `M_e` 224 -> **352 MMAC**, 1.035x the
floor, which is what would test the position rather than the sign.

### The induced-scarcity ladder: the budget mechanism does not produce an inverted U

The MoE budget hypothesis is that the per-expert admission charge covers only the *placed*
experts' GGUF source bytes, leaving the rest of a still-mapped expert GGUF charged to nobody, so
the route over-places once the true hot set approaches RAM. It had an arithmetic fit to five rungs
on **one** model, and no second model on this board reaches the regime unaided. The way to test it
on a second one is to induce the scarcity: hold `B` GiB of touched anonymous memory beside the run
and walk `B` up. The account said the turn would come once `25.3 + B` exceeded ~31.7, i.e. at
`B` ~ 6-7 GB and **not** at `B` = 4.

[HW sweep 2026-08-28, RK1, 600 MHz pinned, governor `performance`, gpt-oss-20b MXFP4,
`-p 2048 -n 0 -r 1 -b 2048 -ub 2048`, page cache dropped and `compact_memory` before every rung,
one process per rung; `perf/data/moe-ballast-ladder.sh`.]

| `B` | MemAvailable | pre-flight budget | stacks placed | resident | streamed | pp2048 t/s | vs `B`=0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30.2 GB | 24598 MB | 63 | 13454 MB | **0** | 28.93 | 1.000 |
| 4 | 26.2 | 20488 | 52 | 11317 | **0** | 24.04 | 0.831 |
| 8 | 22.2 | 16401 | 42 | 9486 | **0** | 20.89 | 0.722 |
| 12 | 18.2 | 12290 | 31 | 7348 | **0** | 18.16 | 0.628 |
| 16 | 14.2 | 8225 | 21 | 5372 | **0** | 15.36 | 0.531 |

**There is no turn, at any rung.** Throughput falls monotonically and very nearly linearly in the
number of placed stacks (`t/s = 8.32 + 0.315 x stacks`, residuals within ±0.8 t/s over a 15-29
range), and every rung reports **100% resident, 0 streamed, and no OOM line in `dmesg`**. The
predicted collapse into the 0.42-0.97x partial-residency regime does not happen; neither does the
"any memory pressure at all" rival's earlier turn. What the ladder shows instead is the route
degrading exactly as its own code says it should: fewer stacks placed, the rest left on the CPU,
"a partial offload and not a loss".

**The mechanism is that the ladder's knob and the route's input are the same quantity.** The
pre-flight budget is `MemAvailable - 6 GiB`, and the budget column falls by exactly 4096 MB per
4 GiB rung. So ballast makes the route **more** conservative, never less: it cannot reach a state
where the route holds more than the board can honour, because the memory it removes is removed
from the budget first. **An induced-scarcity ladder of this shape cannot test the over-placement
hypothesis at all**. That is a fact about the instrument, and it is the useful output here.

**And the account that predicted `B` ~ 6-7 over-counts a file-backed GGUF.** At `B` = 16 the
accounted hot set is 5372 MB of int8 codes + 11.27 GiB of GGUF + 16 GiB of ballast = **32.5 GB
against 31.7 GB of RAM**, and nothing streamed, nothing was killed, and the linear fit did not
bend. So at least some of what that account calls hot is being reclaimed under pressure, and the
GGUF, file-backed unlike the ballast, is the only candidate [hypothesis: not separated from a
smaller-than-assumed activation footprint]. The charge model treats each placed expert's source
bytes as unreclaimable because **decode** reads them from the mmap every token; during a `-n 0`
prefill they are not touched after ingest.

**What this does not settle.** It does not refute the charge omission itself: the uncharged
remainder is still real and still largest at low placement, and the ladder never reached a state
where it could bite. What it refutes is that **this** experiment can reach that state. The
remaining route to it is memory that disappears **after** the budget is frozen, which the
pre-flight cannot see by construction, and that is the runtime-floor safety item rather than the
charge item.

**And that route reaches it on the first try.** Same model, same flags, but the 16 GiB of ballast
is allocated **after** the pre-flight line prints rather than before [HW sweep 2026-08-28, RK1]:

| | budget frozen at | MemAvailable when the ingest ran | stacks reserved | outcome |
|---|---:|---:|---|---|
| ballast **before** (`B`=16 rung above) | 8225 MB | 14.2 GB | 21 | 100% resident, 0 streamed, no OOM |
| ballast **after** the freeze | **24608 MB** | **11.1 GB** | **63** | kernel **OOM-killer fired**; 91% resident, 144 streamed, pp2048 25.53 |

The budget was resolved at 24608 MB against an idle board and then **kept** while 18 GB of the
board disappeared underneath it, so the route went on reserving 63 stacks and 24529 MB against
11.1 GB of available RAM. The board recovered only because the OOM killer picked the **ballast**,
16.8 GB of anonymous RSS, the largest badness score on the box. A competing allocation smaller
than the inference process would have made `llama-bench` the victim instead. The run itself
returned **rc=0 and a plausible number**; nothing in its output says a process was killed.

Note the same two rows are a controlled pair: at essentially the same free RAM (14.2 against
11.1 GB) the route reserved 8225 MB in one and 24608 MB in the other, and the only difference is
**when** the memory went away. That is the runtime `MemAvailable` floor gap tracked as an open
safety item, reproduced rather than argued.

### Qwen3.5-9B

Alibaba's Qwen3.5 9B (arch `qwen35`; dense, GQA), 8.95 B params. GGUFs from
`unsloth/Qwen3.5-9B-GGUF` (base model `Qwen/Qwen3.5-9B`); F16 derived from the published BF16.
This block carries **IQ4_XS**, an importance-matrix 4-bit quant, alongside F16 and Q4_K_M as
a gap-finder for the quantized-prefill path. [HW sweep, 600 MHz, 2026-07-02].

**Prefill (prompt processing), t/s (the NPU's job).**

| test | F16 CPU->NPU | Q4_K_M CPU->NPU | IQ4_XS CPU->NPU |
|---|---|---|---|
| pp512  | 7.25 -> **25.86** (3.6x) | 6.25 -> 16.68 (2.7x) | 7.44 -> 16.33 (2.2x) |
| pp1024 | 7.14 -> **25.14** (3.5x) | 6.21 -> 21.31 (3.4x) | 7.37 -> 20.87 (2.8x) |
| pp2048 | 7.08 -> **24.85** (3.5x) | 6.15 -> 22.88 (3.7x) | 7.29 -> 22.49 (3.1x) |

The NPU carries prefill, ~3.5x on F16, which here holds a **flat ~25 t/s across the whole
curve** (pp512->pp2048), unlike Ministral's declining F16. The quants instead *rise* with M
(Q4_K_M 16.7->22.9, IQ4_XS 16.3->22.5): a quantized GGUF re-dequantizes to fp16 **per
micro-batch**, so at pp512 (one 512-row batch) that dequant is a large fixed fraction, and by
pp2048 it amortizes and the quants close on F16, the mechanism behind the `-b 2048 -ub 2048`
guidance ([quant-prefill-microbatch.md](quant-prefill-microbatch.md)).

**IQ4_XS is the gap-finder result: there is no gap.** An importance matrix is a *quantize-time*
construct; at inference IQ4_XS is an ordinary `ggml_is_quantized` type with a `to_float` trait
and 256-element super-blocks (so `K%32` holds), so its prefill GEMMs dequantize to fp16 and
offload exactly like Q4_K_M (3.1x at pp2048, not the ~1x a CPU fallback would give). The
importance-matrix quants inherit the full quantized-prefill path.

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).**

| | F16 | Q4_K_M | IQ4_XS |
|---|---|---|---|
| tg64 CPU | 1.35 | 3.60 | 4.08 |
| tg64 NPU | 1.34 | 3.59 | 4.09 |

NPU decode matches the CPU, and stays off the NPU ([decode-gemv.md](decode-gemv.md)). It scales with bytes
touched, so the smallest quant streams fastest: F16 1.4 -> Q4_K_M 3.6 -> **IQ4_XS 4.1 t/s** (the
4.80 GB IQ4_XS edges out the 5.28 GB Q4_K_M).

**Interactive.** TTFT and streaming derived from the measured pp/tg rates; the
combined `pp2048+tg128` llama-bench point (F16 NPU 11.96, Q4_K_M NPU 16.77, IQ4_XS NPU 17.01 t/s)
validates the decomposition to ~5%.

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | F16 CPU | 289 s | 1.4 t/s | 437 s |
| | F16 + NPU | **82 s** | 1.4 t/s | 231 s |
| | Q4_K_M + NPU | 90 s | 3.6 t/s | 146 s |
| | IQ4_XS + NPU | 91 s | 4.1 t/s | **140 s** |
| chat, 128-tok prompt -> 400 out | F16 + NPU | 5 s | 1.4 t/s | 304 s |
| | Q4_K_M + NPU | 8 s | 3.6 t/s | 119 s |
| | IQ4_XS + NPU | 8 s | 4.1 t/s | **106 s** |

On a long prompt the NPU cuts first-token wait ~3.5x (F16 289->82 s) and nearly halves the turn;
on a short-prompt/long-reply chat the turn is decode-bound, so the NPU barely moves it and
*quantization* is what helps (F16 304 -> IQ4_XS 106 s). IQ4_XS wins the interactive turn on both
scenarios, TTFT on par with the other quants and the fastest stream.

**Faithfulness: differential perplexity, wikitext test, 12 chunks (same GGUF, NPU prefill vs
CPU).**

| | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| F16    | 9.0637 | 9.0683 | +0.05% |
| Q4_K_M | 9.2358 | 9.2294 | −0.07% |
| IQ4_XS | 9.3502 | 9.3411 | −0.10% |

(±0.43 per-chunk stderr, the NPU−CPU delta is ~100x smaller than the noise.) The NPU prefill
matches the CPU to within 0.10% on all three, so the fp16-accumulation path is faithful for the
importance-matrix quant as well. The F16->Q4_K_M->IQ4_XS ladder (9.06 -> 9.24 -> 9.35) is the
expected quantization cost, present equally on both backends.

**Verdict.** For a 16 GB+ board on batch / RAG workloads, **F16 + NPU**, fastest prefill and,
unusually, flat at ~25 t/s across the prompt curve. For most boards, **IQ4_XS + NPU** is the
sweet spot: it fits an 8 GB board (4.80 GB) with the fastest stream (4.1 t/s), keeps ~3x NPU
prefill at length, and is faithful (Δ −0.10%). Q4_K_M is the marginally-higher-fidelity
alternative (PPL 9.23 vs 9.35) at a little more RAM and a slightly slower stream. The
finding that outlives this model: importance-matrix quants are **not** a backend-offload gap,
IQ4_XS takes the same NPU quantized-prefill path as Q4_K_M.

### Phi-4-mini-instruct

Microsoft's Phi-4-mini-instruct (arch `phi3`; dense, GQA), 3.84 B params. GGUFs from
`unsloth/Phi-4-mini-instruct-GGUF` (base model `microsoft/Phi-4-mini-instruct`); F16 derived from
the published BF16. The small-end reference point next to the 8-9 B models above. [HW sweep,
600 MHz, 2026-07-02].

**Prefill (prompt processing), t/s (the NPU's job).**

| test | F16 CPU->NPU | Q8_0 CPU->NPU | Q4_K_M CPU->NPU |
|---|---|---|---|
| pp512  | 16.55 -> **55.44** (3.4x) | 15.20 -> 37.26 (2.5x) | 13.85 -> 36.70 (2.6x) |
| pp1024 | 15.96 -> **47.86** (3.0x) | 14.81 -> 38.77 (2.6x) | 13.55 -> 37.29 (2.8x) |
| pp2048 | 15.07 -> **39.79** (2.6x) | 13.90 -> 34.91 (2.5x) | 12.89 -> 33.42 (2.6x) |

F16 pp512 hits **55 t/s**, but the win *falls* with prompt
length (3.4x->2.6x) as the NPU rate declines 55->40. On a model this small the per-op readback and
dispatch are a larger fixed fraction and grow with M (more output tiles to read back), so the F16
curve slopes down, the opposite of the flat Qwen3.5-9B F16. The quants are flatter (~33-39 t/s)
and, at pp2048, close much of the gap to F16 (33-35 vs 40) as their per-micro-batch dequant
amortizes while F16's readback grows.

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).**

| | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| tg64 CPU | 2.86 | 4.86 | 7.04 |
| tg64 NPU | 2.84 | 4.82 | 6.95 |

NPU decode matches the CPU, and stays off the NPU ([decode-gemv.md](decode-gemv.md)), and at 3.84 B it is brisk:
F16 2.9 -> Q8_0 4.9 -> **Q4_K_M 7.0 t/s**, a genuinely interactive stream from a 2.31 GB file.

**Interactive.** TTFT and streaming derived from the measured pp/tg rates.

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | F16 CPU | 136 s | 2.9 t/s | 206 s |
| | F16 + NPU | **51 s** | 2.8 t/s | 121 s |
| | Q8_0 + NPU | 59 s | 4.8 t/s | 100 s |
| | Q4_K_M + NPU | 61 s | 7.0 t/s | **90 s** |
| chat, 128-tok prompt -> 400 out | F16 + NPU | 2 s | 2.8 t/s | 143 s |
| | Q4_K_M + NPU | 3 s | 7.0 t/s | **61 s** |

The NPU cuts the long-prompt first-token wait ~2.6x (F16 136->51 s); on the short-prompt chat the
turn is decode-bound, so quantization is the lever (F16 143 -> Q4_K_M 61 s). The combined
`pp2048+tg128` llama-bench point (F16 NPU 18.35, Q8_0 20.89, Q4_K_M 21.43 t/s) runs ~15-20% below
the naive TTFT+stream sum: the 128 tokens decoded *after* a 2048-token prompt read a full KV cache
and stream slower than tg64-from-empty, a larger relative penalty on a small, fast model. So the
`total turn` figures are optimistic by that margin, uniformly across configs.

**Faithfulness: differential perplexity, wikitext test, 12 chunks (same GGUF, NPU prefill vs
CPU).**

| | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| F16    | 10.9284 | 10.9152 | −0.12% |
| Q4_K_M | 11.6977 | 11.5905 | −0.92% |

Both deltas sit inside the reported ±0.5-0.6 PPL uncertainty. The F16 path matches the CPU to
−0.12%; the Q4_K_M's larger −0.92% reflects the NPU consuming the Q4_K weights via dequant-to-fp16
rather than the CPU's native Q4_K kernel, and it lands in the NPU's favor (lower PPL), so no
accuracy is lost. (The **absolute** PPL is high, ~11, because Phi-4 is trained largely on curated
/ synthetic data and models raw wikitext poorly; as with the other instruct models only the
NPU−CPU delta is informative, not the absolute value.)

**Verdict.** Phi-4-mini is a strong small / edge-board fit. **F16 + NPU** gives the fastest prefill
in this record (55 t/s at pp512) when 7 GB of RAM is available. **Q4_K_M + NPU** is the practical
pick: **2.31 GB** (fits a 4 GB board), ~2.6x NPU prefill, and a genuinely interactive 7 t/s stream.
The NPU prefill win is real but narrows with prompt length (F16 3.4x->2.6x) as this small model's
per-op readback grows with M, so the largest wins are on short-to-medium prompts.

### Gemma-4-12B-it

Google's Gemma-4-12B-it (arch `gemma4`; dense, GQA), 11.91 B params. GGUFs from
`unsloth/gemma-4-12b-it-GGUF` (base model `google/gemma-4-12b-it`); F16 derived from the published
BF16. A large-model point in this record (the 27B Qwen3.6 below is the largest), and the model
this stack's fp16 prefill was first brought up on. [HW sweep, 600 MHz, 2026-07-02].

**Prefill (prompt processing), t/s (the NPU's job).**

| test | F16 CPU->NPU | Q8_0 CPU->NPU | Q4_K_M CPU->NPU |
|---|---|---|---|
| pp512  | 4.82 -> **17.42** (3.6x) | 4.52 -> 11.41 (2.5x) | 4.23 -> 11.20 (2.6x) |
| pp1024 | 4.78 -> **16.09** (3.4x) | 4.35 -> 13.53 (3.1x) | 4.15 -> 13.19 (3.2x) |
| pp2048 | 4.63 -> **14.98** (3.2x) | 4.16 -> 13.43 (3.2x) | 4.02 -> 13.28 (3.3x) |

The NPU carries prefill, up to 3.6x on F16 over a slow (~4.6 t/s) 12 B CPU baseline. F16 prefills
fastest in absolute terms (17.4 t/s at pp512), but its win *narrows* with prompt length (3.6x->3.2x)
as the NPU rate falls 17.4->15.0 (more output tiles to read back per matmul as M grows). The quants
move the opposite way, *rising* with M (Q4_K_M 11.2->13.3, 2.6x->3.3x): a quantized GGUF
re-dequantizes to fp16 per micro-batch, so at pp512 that dequant is a large fixed fraction and by
pp2048 it amortizes (the mechanism behind `-b 2048 -ub 2048`,
[quant-prefill-microbatch.md](quant-prefill-microbatch.md)). By pp2048 all three land at ~13-15 t/s:
a quantized GGUF does not prefill faster than F16 at this operating point, it only fits smaller
([not-mac-bound.md](not-mac-bound.md)).

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).**

| | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| tg64 CPU | 0.94 | 1.75 | 2.44 |
| tg64 NPU | 0.94 | 1.59 | 2.50 |

NPU decode matches the CPU, and stays off the NPU ([decode-gemv.md](decode-gemv.md)). At 12 B and F16 it is
**very slow (0.94 t/s)**: 22 GiB of weights are streamed from LPDDR every token. Decode scales with
bytes touched, so quantization is the only lever and a large one, F16 0.94 -> Q8_0 1.7 -> Q4_K_M
2.5 t/s (2.7x). On this model a quant is effectively mandatory for interactive use.

**Interactive.** TTFT and streaming derived from the measured pp/tg rates; the
combined `pp2048+tg128` llama-bench point (F16 NPU 7.37, Q8_0 8.72, Q4_K_M 9.61 t/s) validates the
decomposition.

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | F16 CPU | 442 s | 0.9 t/s | 655 s |
| | F16 + NPU | **137 s** | 0.9 t/s | 350 s |
| | Q8_0 + NPU | 152 s | 1.6 t/s | 278 s |
| | Q4_K_M + NPU | 154 s | 2.5 t/s | **234 s** |
| chat, 128-tok prompt -> 400 out | F16 + NPU | 7 s | 0.9 t/s | 433 s |
| | Q4_K_M + NPU | 11 s | 2.5 t/s | **171 s** |

On a long prompt the NPU cuts first-token wait ~3.2x (F16 442->137 s). But this model's decode is so
slow that the *stream* dominates the turn: even with NPU prefill, F16 needs ~350 s for a 200-token
reply, so *quantization*, not the backend, is what makes the turn usable (Q4_K_M 234 s; the chat
turn 433->171 s). Best interactive = NPU (short TTFT) + Q4_K_M (fastest stream).

**Faithfulness: differential perplexity, wikitext test, 12 chunks (same GGUF, NPU prefill vs CPU).**

| | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| F16    | 630.9975 | 625.8672 | −0.81% |
| Q8_0   | 639.9257 | 652.1588 | +1.91% |
| Q4_K_M | 670.6969 | 675.4451 | +0.71% |

The **absolute** PPL (~630+) is not meaningful, Gemma-4-12B-it is an instruction/reasoning-tuned
model and scores raw wikitext very poorly (as with the other instruct models, only the NPU−CPU delta
is informative, not the absolute value). The per-run stderr is ±~65 (~±10%), and all three NPU−CPU
deltas (−0.8%, +1.9%, +0.7%) sit well inside it, so the fp16-accumulation prefill path is faithful
across F16 and both quants.

**Verdict.** Gemma-4-12B is the large-end fit. **F16 + NPU** gives the fastest prefill (17.4 t/s at
pp512, 3.6x the CPU) for a 32 GB board (the F16 GGUF is 22 GiB) doing batch / RAG work, but its
0.94 t/s decode makes long *replies* slow regardless of backend. **Q4_K_M + NPU** is the practical
pick: 6.86 GB (fits an 8 GB board), ~3.3x NPU prefill at length, and a 2.7x-faster stream, the
fastest *effective* turn on both the RAG and chat scenarios. Q8_0 (11.78 GB) is the near-lossless
middle.

### Qwen3.6-27B (hybrid Gated-DeltaNet, Q4_K_M)

Alibaba's Qwen3.6 27B (arch `qwen35`; 65 blocks, 27.32 B params), a **hybrid** model interleaving
Gated-DeltaNet linear-attention / SSM-scan layers with dense GQA attention (GGUF SSM metadata:
state 128, conv kernel 4, 16 groups, inner size 6144). GGUF from `unsloth/Qwen3.6-27B-MTP-GGUF`. The
**largest model in this record**, reported in **Q4_K_M only**: at 27 B the F16 GGUF (~54 GB) and
Q8_0 (~29 GB) do not fit the 31 GB board, so Q4_K_M (15.92 GiB) is the precision that runs, itself
the honest large-model story on an edge board. [HW sweep, 600 MHz, 2026-07-03].

**Prefill (prompt processing), t/s (the NPU's job).**

| test | Q4_K_M CPU->NPU |
|---|---|
| pp512  | 1.79 -> 5.38 (3.0x) |
| pp1024 | 1.79 -> 6.78 (3.8x) |
| pp2048 | 1.78 -> **7.78 (4.4x)** |

The NPU carries prefill, and here the win **rises with prompt length** (3.0x->4.4x) to **4.4x at
pp2048, the largest NPU prefill win in this record**. Two effects compound: a quantized GGUF
re-dequantizes to fp16 per micro-batch, so its NPU rate climbs as that dequant amortizes (5.4->7.8
t/s, the `-b 2048 -ub 2048` mechanism, [quant-prefill-microbatch.md](quant-prefill-microbatch.md));
and the CPU baseline is unusually slow (~1.8 t/s, a 27 B streamed through 8 CPU cores), so the NPU
laps it more decisively than on any smaller model. The relative win grows with model size because the
CPU degrades faster than the NPU as the matmuls grow.

**The hybrid architecture does not block the prefill offload.** Qwen3.6-27B's Gated-DeltaNet /
SSM-scan layers are CPU-only ops (no NPU handler) and stay on the CPU, but they are a small share of
the prefill FLOPs, the FFN and attention projections dominate and offload as ordinary `MUL_MAT`, so
the prefill win holds in full. Linear attention is therefore **not** a prefill-offload blocker, unlike
a mixture-of-experts model whose expert FFNs route through `MUL_MAT_ID`, which has a handler, but
offloading the quantized experts is dequant-bound and loses, so they stay on the CPU by default (the
gpt-oss-20b block).

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).**

| | Q4_K_M |
|---|---|
| tg64 CPU | 1.10 |
| tg64 NPU | 1.11 |

NPU decode matches the CPU, and stays off the NPU ([decode-gemv.md](decode-gemv.md)), and at 27 B it is **very
slow (1.1 t/s)**: the 16 GiB of Q4_K_M weights are a large per-token stream from LPDDR. There is no
smaller quant here to speed the stream, so this model is a prefill / batch engine, not a chatbot.

**Interactive.** TTFT and streaming derived from the measured pp/tg rates; the
combined `pp2048+tg128` point (CPU 1.70, NPU 5.57 t/s) validates the decomposition.

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | Q4_K_M CPU | 1151 s | 1.1 t/s | 1333 s |
| | Q4_K_M + NPU | **263 s** | 1.1 t/s | **443 s** |
| chat, 128-tok prompt -> 400 out | Q4_K_M CPU | 72 s | 1.1 t/s | 436 s |
| | Q4_K_M + NPU | 24 s | 1.1 t/s | 384 s |

On a long prompt the NPU cuts first-token wait 4.4x (1151->263 s, ~19 min -> ~4.5 min) and the whole
RAG turn ~3x (1333->443 s). On the short-prompt chat the turn is decode-bound, 400 tokens at 1.1 t/s
is ~6 min regardless of backend, and with no quant lever the NPU only trims the small TTFT (436->384
s). The NPU's value on this model is unambiguously prefill: long prompt in, short answer out.

**Faithfulness: differential perplexity, wikitext test, 12 chunks (same GGUF, NPU prefill vs CPU).**

| | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| Q4_K_M | 7.2188 | 7.1997 | −0.26% |

(±0.32 per-chunk stderr, the NPU−CPU delta is ~17x smaller than the noise.) Unlike the
instruction/reasoning models in this record (whose raw-wikitext PPL is inflated and not meaningful),
Qwen3.6-27B scores a **normal** wikitext PPL (~7.2), and the NPU−CPU delta (−0.26%, in the NPU's
favor) sits well inside the per-chunk stderr, the fp16-accumulation prefill path is faithful on the
hybrid model too.

**Verdict.** Qwen3.6-27B is the largest-model fit and posts the record's **largest NPU prefill win**
(4.4x at pp2048), the NPU turns a ~19-minute CPU first-token wait on a 2048-token prompt into ~4.5
minutes. But at 27 B on a 31 GB board it is a **prefill / RAG engine, not an interactive chatbot**:
only Q4_K_M fits, decode is ~1.1 t/s (a 400-token reply is ~6 min), and there is no smaller quant to
speed the stream. Run it for long-prompt / short-answer work (summarize, extract, RAG) where the
4.4x first-token win lands; reach for a smaller model (Gemma-4-12B or Qwen3.5-9B at Q4_K_M) when
replies must stream at conversational speed. The finding that outlives this model: **linear-attention
/ SSM hybrids offload their prefill fine**, the DeltaNet layers stay on the CPU but do not gate the
win.

### Llama-3.2-3B-Instruct

Meta's Llama-3.2-3B-Instruct (arch `llama`; dense, GQA), 3.21 B params, the **smallest model
in this record** and the field's standard reference point. GGUFs from
`unsloth/Llama-3.2-3B-Instruct-GGUF`; F16 is unsloth's published F16 (derived from the BF16).
[HW sweep, 600 MHz, 2026-07-03].

**Prefill (prompt processing), t/s (the NPU's job).**

| test | F16 CPU->NPU | Q8_0 CPU->NPU | Q4_K_M CPU->NPU |
|---|---|---|---|
| pp512  | 18.23 -> **61.79** (3.4x) | 17.05 -> 34.19 (2.0x) | 16.71 -> 33.60 (2.0x) |
| pp1024 | 17.74 -> **52.91** (3.0x) | 16.48 -> 36.31 (2.2x) | 16.34 -> 36.66 (2.2x) |
| pp2048 | 17.29 -> **45.54** (2.6x) | 15.72 -> 34.11 (2.2x) | 15.49 -> 34.46 (2.2x) |

F16 pp512 hits **61.79 t/s** (past Phi-4-mini's 55), because
this is the smallest model here: fewer FLOPs per matmul. As with Phi-4-mini the F16 win *falls*
with prompt length (3.4x->2.6x, the NPU rate declining 62->46), on a small model the per-op readback
and dispatch are a larger fixed fraction and grow with M.

The quants read as a *lower* multiplier (~2.0-2.2x) than Phi-4-mini's ~2.5-2.6x, but that is the
CPU denominator, not a weaker NPU: Llama-3.2-3B's **CPU** quant prefill is unusually strong (~16-17
t/s, barely under its own F16 CPU 17-18), while its NPU quant absolute (~34-36 t/s) sits right in
line with Phi-4-mini. The quants also *rise* then flatten with M (pp512->pp1024 up, pp2048 flat), the
per-micro-batch dequant amortizing, the `-b 2048 -ub 2048` mechanism
([quant-prefill-microbatch.md](quant-prefill-microbatch.md)). By pp2048 the two quants (~34 t/s)
trail F16 (46 t/s): at this operating point a quantized GGUF does not prefill faster than F16, it
only fits smaller ([not-mac-bound.md](not-mac-bound.md)).

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).**

| | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| tg64 CPU | 3.15 | 5.62 | 7.88 |
| tg64 NPU | 3.27 | 5.47 | 7.93 |

NPU decode matches the CPU, and stays off the NPU ([decode-gemv.md](decode-gemv.md)), and at 3.21 B it is the
briskest stream in the record: F16 3.3 -> Q8_0 5.5 -> **Q4_K_M 7.9 t/s**, a fully interactive stream
from a **1.87 GB** file.

**Interactive.** TTFT and streaming derived from the measured pp/tg rates.

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | F16 CPU | 118 s | 3.2 t/s | 182 s |
| | F16 + NPU | **45 s** | 3.3 t/s | 106 s |
| | Q8_0 + NPU | 60 s | 5.5 t/s | 97 s |
| | Q4_K_M + NPU | 59 s | 7.9 t/s | **84 s** |
| chat, 128-tok prompt -> 400 out | F16 + NPU | 3 s | 3.3 t/s | 125 s |
| | Q4_K_M + NPU | 4 s | 7.9 t/s | **54 s** |

The NPU cuts the long-prompt first-token wait ~2.6x (F16 118->45 s); on the short-prompt chat the
turn is decode-bound, so quantization is the lever (F16 125 -> Q4_K_M 54 s). The combined
`pp2048+tg128` llama-bench point (F16 NPU 21.06, Q8_0 21.89, Q4_K_M 22.34 t/s) runs below the naive
TTFT+stream sum: the 128 tokens decoded *after* a 2048-token prompt read a full KV cache and stream
slower than tg64-from-empty, a larger relative penalty on a small, fast model, so the `total turn`
figures are optimistic by that margin, uniformly across configs.

**Faithfulness: differential perplexity, wikitext test, 12 chunks (same GGUF, NPU prefill vs
CPU).**

| | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| F16    | 12.6656 | 12.6682 | +0.02% |
| Q4_K_M | 12.9554 | 12.8978 | −0.44% |

Both deltas sit well inside the reported ±0.65 PPL uncertainty (the F16 NPU−CPU delta is ~300x
smaller than the per-chunk stderr). The F16 path matches the CPU to +0.02%; the Q4_K_M delta
(−0.44%, in the NPU's favor) reflects the NPU consuming the Q4_K weights via dequant-to-fp16 rather
than the CPU's native Q4_K kernel, no accuracy lost. (The **absolute** PPL is high, ~12.7, because
Llama-3.2-3B-Instruct is instruction-tuned and models raw wikitext poorly; as with the other
instruct models only the NPU−CPU delta is informative, not the absolute value.)

**Verdict.** Llama-3.2-3B is the smallest, most portable LLM in this record and posts its **fastest
prefill** (61.8 t/s at pp512, 3.4x). **F16 + NPU** is the max-prefill pick when ~6 GB of RAM is free;
the win narrows with prompt length (3.4x->2.6x) as this small model's per-op readback grows with M, so
it lands hardest on short-to-medium prompts. **Q4_K_M + NPU** is the practical pick: **1.87 GB** (fits
a 4 GB board with headroom), ~2.2x NPU prefill, and the record's fastest stream (7.9 t/s), the
faster *effective* turn on both scenarios. The finding that outlives this model: the NPU/CPU prefill
*ratio* narrows on small dense models not because the NPU weakens but because their CPU baseline is
comparatively fast, the NPU's absolute prefill rate stays the honest headline.

### Ministral-3-3B-Instruct-2512

Mistral's Ministral-3-3B-Instruct-2512 (arch `mistral3`; dense, GQA), 3.43 B params, the 3B
sibling of the Ministral-3-8B this benchmark format was first validated on, and a small-end point
alongside Llama-3.2-3B and Phi-4-mini. GGUFs from `unsloth/Ministral-3-3B-Instruct-2512-GGUF`; F16
derived from the published BF16. [HW sweep, 600 MHz, 2026-07-03].

**Prefill (prompt processing), t/s (the NPU's job).**

| test | F16 CPU->NPU | Q8_0 CPU->NPU | Q4_K_M CPU->NPU |
|---|---|---|---|
| pp512  | 18.02 -> **57.62** (3.2x) | 16.16 -> 32.94 (2.0x) | 15.55 -> 32.96 (2.1x) |
| pp1024 | 17.39 -> **48.57** (2.8x) | 15.54 -> 35.14 (2.3x) | 15.10 -> 34.67 (2.3x) |
| pp2048 | 16.26 -> **39.78** (2.4x) | 14.50 -> 30.78 (2.1x) | 14.32 -> 30.54 (2.1x) |

F16 pp512 hits **57.6 t/s**, between Llama-3.2-3B (61.8) and Phi-4-mini (55.4): the three ~3-4 B
dense models cluster near the top of the absolute-prefill table
because they run the fewest FLOPs per matmul. As on the other two the F16 win *falls* with prompt
length (3.2x->2.4x, the NPU rate declining 58->40), a small model's per-op readback is a larger fixed
fraction and grows with M. The quants read ~2.0-2.3x (NPU absolute ~31-35 t/s) and *rise* then
flatten with M as their per-micro-batch dequant amortizes (the `-b 2048 -ub 2048` mechanism,
[quant-prefill-microbatch.md](quant-prefill-microbatch.md)); the modest ratio is the strong CPU quant
baseline (~14-16 t/s), not a weaker NPU, the same reading as Llama-3.2-3B. By pp2048 the quants
(~31 t/s) trail F16 (40): a quantized GGUF does not prefill faster than F16 here, it only fits
smaller ([not-mac-bound.md](not-mac-bound.md)).

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).**

| | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| tg64 CPU | 3.01 | 5.39 | 7.59 |
| tg64 NPU | 3.00 | 5.53 | 7.66 |

NPU decode matches the CPU, and stays off the NPU ([decode-gemv.md](decode-gemv.md)), and at 3.43 B it streams
briskly: F16 3.0 -> Q8_0 5.5 -> **Q4_K_M 7.7 t/s** from a **1.99 GB** file (a hair under Llama-3.2-3B's
7.9 from 1.87 GB).

**Interactive.** TTFT and streaming derived from the measured pp/tg rates.

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | F16 CPU | 126 s | 3.0 t/s | 192 s |
| | F16 + NPU | **51 s** | 3.0 t/s | 118 s |
| | Q8_0 + NPU | 67 s | 5.5 t/s | 103 s |
| | Q4_K_M + NPU | 67 s | 7.7 t/s | **93 s** |
| chat, 128-tok prompt -> 400 out | F16 + NPU | 3 s | 3.0 t/s | 136 s |
| | Q4_K_M + NPU | 4 s | 7.7 t/s | **56 s** |

The NPU cuts the long-prompt first-token wait ~2.5x (F16 126->51 s); on the short-prompt chat the
turn is decode-bound, so quantization is the lever (F16 136 -> Q4_K_M 56 s). The combined
`pp2048+tg128` llama-bench point (F16 NPU 19.10, Q8_0 20.06, Q4_K_M 20.86 t/s) runs below the naive
TTFT+stream sum: the 128 tokens decoded *after* a 2048-token prompt read a full KV cache and stream
slower than tg64-from-empty, so the `total turn` figures are optimistic by that margin, uniformly
across configs.

**Faithfulness: differential perplexity, wikitext test, 12 chunks (same GGUF, NPU prefill vs
CPU).**

| | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| F16    | 9.9014 | 9.8979 | −0.04% |
| Q4_K_M | 10.1022 | 10.0918 | −0.10% |

Both deltas sit well inside the ±0.46 per-chunk stderr (the F16 NPU−CPU delta is ~130x smaller). The
NPU prefill matches the CPU to −0.04% on F16 and −0.10% on Q4_K_M (both in the NPU's favor), so the
fp16-accumulation path is faithful. The **absolute** PPL (~9.9) is higher than the 8B sibling's ~6.9,
the small-model quality cost rather than an NPU effect; as an instruct model, only the NPU−CPU
delta is informative.

**Verdict.** Ministral-3-3B is a strong small / edge-board fit and posts the record's second-fastest
prefill (57.6 t/s at pp512, 3.2x). **F16 + NPU** is the max-prefill pick when ~7 GB of RAM is free,
with the win largest on short-to-medium prompts (3.2x->2.4x as M grows). **Q4_K_M + NPU** is the
practical pick: **1.99 GB** (fits a 4 GB board), ~2.1x NPU prefill, and a 7.7 t/s stream, the faster
*effective* turn on both scenarios. It sits between Llama-3.2-3B (marginally faster prefill and
stream) and Phi-4-mini in the small-model cohort; all three confirm the same reading, on small dense
models the NPU/CPU *ratio* is modest because the CPU baseline is fast, while the NPU's absolute
prefill rate (55-62 t/s F16) tops the record.

### DeepSeek-V2-Lite (MLA + MoE, Q4_K_M)

DeepSeek's V2-Lite (arch `deepseek2`; 27 layers, 15.71 B params, ~2.4 B active per token), a
**mixture-of-experts** model (64 routed + 2 shared experts, 6 routed active per token; block 0 dense)
whose attention is **Multi-head Latent Attention (MLA)**: the KV is compressed to a `kv_lora_rank=512`
latent and the per-head key/value dims are **asymmetric**, key_length 192 (128 nope + 64 rope),
value_length 128. GGUF from `mradermacher/DeepSeek-V2-Lite-GGUF` (the base model
`deepseek-ai/DeepSeek-V2-Lite`, 9.65 GiB). **Q4_K_M only**: at 15.71 B the F16 GGUF (~31 GB) does not
fit the 31 GB board. [HW sweep, 600 MHz, 2026-07-03; MoE offload re-bench 2026-07-04].

This is a **combined gap-finder**; it stacks two offload gaps in one model, both addressed at the
op level, neither a default win. (1) Its **MLA attention** has asymmetric key/value dims (DK=192 ≠ DV=128);
the FLASH_ATTN handler **accepts DK≠DV** (bit-faithful primitive, cos=1.000000), so the MLA FA
primitive is ready, though the DeepSeek DL-backend FA path is not yet wired on-device, and it is
dispatch-bound anyway (large-DK, few-KV-head, pays only at long context), so attention stays on the
CPU here (pp-neutral). (2) Its **routed experts** go through
`GGML_OP_MUL_MAT_ID`, which has a handler (`ROCKET_MOE=1`), but offloading the quantized experts is a
**net loss** (below), so they too stay on the CPU by default.

By default, then, what reaches the NPU is the **large MLA projections** (q and kv down/up), the **2
always-on shared experts'** gate/up/down, and `lm_head`; all ordinary static-weight `MUL_MAT`. Those
dense GEMMs are big enough (much larger than gpt-oss's GQA projections, and gpt-oss has no shared
expert) that the NPU posts a **modest but real** prefill win, larger than gpt-oss's.

**Prefill (prompt processing), t/s (Q4_K_M, the only precision).**

| test | CPU -> NPU |
|---|---|
| pp512  | 20.37 -> 24.04 (1.18x) |
| pp1024 | 19.92 -> 24.85 (1.25x) |
| pp2048 | 19.01 -> **23.87 (1.26x)** |

The NPU prefill holds a flat ~24 t/s across the curve while the CPU baseline declines with M
(20.4->19.0), so the win *rises* modestly (1.18x->1.26x). It is **bigger than gpt-oss's ~1.04x**: the MLA
projections and the two shared experts are substantial dense GEMM that offload, but far short of the
dense models' 3x+, because attention (MLA, on the CPU here) and the routed experts (below), the bulk of
the graph, stay on the CPU. CPU prefill is itself brisk (~19-20 t/s for a 16 B model, same-session CPU
20.40 / 19.93 / 18.99) because only ~2.4 B params are active per token.

**Offloading the routed experts on the fp16 streaming route: the negative that closed.** On
that route the `MUL_MAT_ID` handler *drops* prefill, and here the loss is **larger** than gpt-oss's,
DeepSeek's CPU is faster and its default NPU already wins, so there is more to give up (`-ub 2048`,
same-session, `ROCKET_MOE=1 ROCKET_MOE_NATIVE=0` in today's spelling):

| test | CPU | NPU (fp16 expert route) |
|---|---|---|
| pp512  | 20.40 | 5.05 (0.25x) |
| pp1024 | 19.93 | 7.82 (0.39x) |
| pp2048 | 18.99 | 11.30 (0.59x) |

Same dequant-bound cause as gpt-oss (each of the 64 routed experts' Q4_K weights dequantized to fp16 on
the host every micro-batch, amortized over only `M_e ~ n_tokens·6/64` rows), and worse here because the
routed experts are smaller (K=2048, N=1408) and more numerous, so the per-expert dequant + dispatch
overhead is a larger share.

**The native-quant route on DeepSeek is prefill-length-dependent.** This is the
model that showed a residency-independent failure mode, so read it before assuming the gpt-oss result
ports. [HW sweep 2026-08-27, 600 MHz, governor `performance`, one board, one `.so`.]

| test | CPU | NPU, `ROCKET_MOE=0` | NPU, experts on the NPU |
|---|---:|---:|---:|
| pp512  | 19.54 | **26.04** | 19.09 (**0.73x the experts-on-CPU arm**) |
| pp2048 | 18.50 | 21.63 | **27.9 (1.29x)** |

The pp512 row is a **27% regression at 100% residency, 0 streamed**: so it is not the partial-residency
mechanism at all. The rows an expert receives are `n_tokens · n_used / n_expert`, which is an
**architecture** property and not a prompt one: DeepSeek routes **6 of 64**, so 512 tokens give each
expert ~**48** rows, *below* the 64-row granule its GEMM is then padded up to, and the padded rows are
computed and read back in full. gpt-oss routes 4 of 32, so the same token count gives it 64 rows, at
the granule, and it wins. At pp2048 DeepSeek gets `M_e` ~192, clears the floor, and wins 1.28x.

The backend's default gate therefore requires `M_e >= ROCKET_MOE_M_BUCKET`, which declines DeepSeek's
experts at short prefill and takes them at long, the same model, placed differently by length.
Confirmed on device: with the floor in place DeepSeek pp512 reads **25.9**, i.e. the experts-on-CPU
number restored. **A residency check alone would not have caught this**, which is the transferable
lesson: `MUL_MAT_ID` placement has two independent failure modes, and one of them is invisible to the
residency instrument.

On the fp16 route the routed experts run faithfully on
the CPU.

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).**

| | Q4_K_M |
|---|---|
| tg64 CPU | 7.73 |
| tg64 NPU | 7.77 |

NPU decode matches the CPU, and stays off the NPU ([decode-gemv.md](decode-gemv.md)); the MoE's small active-param
count keeps it brisk (~7.7 t/s from a 9.65 GB file, roughly a dense 3 B's rate).

**Interactive.** TTFT and streaming from the measured pp/tg rates; the combined
`pp2048+tg128` point (CPU 15.25, NPU 18.12 t/s) validates the decomposition.

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | Q4_K_M CPU | 108 s | 7.7 t/s | 134 s |
| | Q4_K_M + NPU | **86 s** | 7.7 t/s | **112 s** |
| chat, 128-tok prompt -> 400 out | Q4_K_M CPU | 6 s | 7.7 t/s | 58 s |
| | Q4_K_M + NPU | 5 s | 7.7 t/s | 57 s |

On a long prompt the NPU trims first-token wait ~1.26x (108->86 s), the only lever, since there is no
smaller quant and decode is CPU-bound either way; on the short-prompt chat the turn is decode-bound and
the NPU barely moves it (58->57 s). One caveat: the combined point implies the 128 tokens decoded
*after* a 2048-token prompt stream at only ~3.7 t/s on both backends (about half the tg64-from-empty
rate, MLA decode cost grows with the filled latent KV cache), so the **RAG** `total turn` figures
(which decode after the long prompt) are optimistic; the chat turns (short prompt) are accurate.

**Faithfulness: differential perplexity, wikitext test, 12 chunks (same GGUF, NPU prefill vs CPU).**

| config | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| default (experts -> CPU) | 8.2444 | 8.2232 | −0.26% |
| `ROCKET_MOE=1` (routed experts -> NPU) | 8.2444 | 8.2205 | −0.29% |

(±0.38 per-chunk stderr.) Unlike the instruct/reasoning models in this record, DeepSeek-V2-Lite is a
**base** model and scores a normal wikitext PPL (~8.2), and both NPU−CPU deltas sit well inside the
per-chunk stderr, the default offloads and the `MUL_MAT_ID` **expert** offload (−0.29%) are both
faithful. The handler's problem is purely speed, not accuracy (matching `test-rocket-moe` cos = 1.000000).

**Verdict.** DeepSeek-V2-Lite stacks **MLA attention** and **MoE** in one model; both gaps now have
op-level handlers, and the MoE one is a default win **at long prefill only**. MLA's asymmetric head dims (DK=192 ≠ DV=128) once failed the
FLASH_ATTN `DK==DV` contract; the gate is now **relaxed to accept DK≠DV** (bit-faithful primitive), so
the MLA FA primitive is ready, but the DeepSeek DL-backend FA path is not yet exercised on-device, and
it is dispatch-bound and pp-neutral at these lengths, so attention stays on the CPU. The routed experts
take the **native-quant** route by default at pp2048 (1.31x over the experts-on-CPU arm) and stay on the
CPU below the per-dispatch work floor; the old **fp16** expert route loses harder here than on gpt-oss
(0.25x->0.59x the CPU) and is never taken by default. Below the floor, then, the MLA projections +
2 shared experts + `lm_head` offload for a **modest, real 1.18-1.26x** prefill, larger than gpt-oss's ~1.04x (bigger dense
projections), below the dense models' 3x+, faithful. The remaining moves are a **default** MoE win
(native-quant experts / resident caching) and engaging MLA at the long context where it pays.

### Phi-4 (14B)

Microsoft's Phi-4 (arch `llama`, 14.66 B params), the 14 B sibling of the Phi-4-mini above, though a
different architecture: llama.cpp maps the 14 B Phi-4 to the `llama` arch (a Llama-style dense GQA
transformer, shown as "llama 13B" in the rows), where Phi-4-mini is `phi3`. GGUFs from
`unsloth/phi-4-GGUF` (base model `microsoft/phi-4`). Reported in **Q8_0 + Q4_K_M**: at 14.66 B the
F16 GGUF (29.3 GB) does not fit the 31 GB board (29 GB free, no swap), so Q8_0 (14.51 GiB) is the
highest precision that runs, the same forced-quant situation as the 27 B Qwen3.6 above, one size
class down. A mid-large point between Gemma-4-12B and Qwen3.6-27B. [HW sweep, 600 MHz, 2026-07-04].

**Prefill (prompt processing), t/s (the NPU's job).**

| test | Q8_0 CPU->NPU | Q4_K_M CPU->NPU |
|---|---|---|
| pp512  | 3.83 -> 10.61 (2.8x) | 3.53 -> 10.31 (2.9x) |
| pp1024 | 3.70 -> 12.13 (3.3x) | 3.49 -> 11.93 (3.4x) |
| pp2048 | 3.60 -> **12.20 (3.4x)** | 3.40 -> **11.79 (3.5x)** |

The NPU carries prefill, up to 3.5x over a slow (~3.5 t/s) 14 B CPU baseline. With no F16 that fits,
both variants are quantized GGUFs, and both show the quant signature: the win **rises with prompt
length** (2.8x->3.4x Q8_0, 2.9x->3.5x Q4_K_M) as the per-micro-batch dequant amortizes across the
`-b 2048 -ub 2048` window ([quant-prefill-microbatch.md](quant-prefill-microbatch.md)). By pp2048
both land at ~12 t/s, a quantized GGUF does not prefill faster than the other at this operating
point ([not-mac-bound.md](not-mac-bound.md)); Q4_K_M's slightly higher *ratio* (3.5x vs 3.4x) is its
slightly slower CPU denominator, not a faster NPU. The absolute ~12 t/s sits between Gemma-4-12B's
Q4_K_M (13.3) and Qwen3.6-27B's Q4_K_M (7.8), tracking model size. Because F16 does not fit, this
model does not reach the higher absolute prefill an unconstrained board would give it (Gemma-4-12B's
F16 hits ~15-17 t/s), the quant is the ceiling here, not a choice.

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).**

| | Q8_0 | Q4_K_M |
|---|---|---|
| tg64 CPU | 1.43 | 2.17 |
| tg64 NPU | 1.42 | 2.19 |

NPU decode matches the CPU, and stays off the NPU ([decode-gemv.md](decode-gemv.md)). At 14.66 B and Q8_0 it is
**slow (1.4 t/s)**, 14.5 GiB streamed from LPDDR per token, and quantization is the only lever:
Q4_K_M nearly doubles it to 2.2 t/s (8.28 GiB/token). A quant is effectively mandatory for
interactive use on this model.

**Interactive.** TTFT and streaming derived from the measured pp/tg rates; the
combined `pp2048+tg128` llama-bench point (Q8_0 NPU 7.48, Q4_K_M NPU 8.29 t/s) validates the
decomposition.

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | Q8_0 CPU | 569 s | 1.4 t/s | 709 s |
| | Q8_0 + NPU | **168 s** | 1.4 t/s | 309 s |
| | Q4_K_M + NPU | 174 s | 2.2 t/s | **265 s** |
| chat, 128-tok prompt -> 400 out | Q8_0 + NPU | 12 s | 1.4 t/s | 294 s |
| | Q4_K_M + NPU | 12 s | 2.2 t/s | **195 s** |

On a long prompt the NPU cuts the first-token wait 3.4x (Q8_0 569->168 s, ~9.5 min -> ~2.8 min). But
this model's decode is slow enough that the *stream* dominates a long reply: even with NPU prefill
Q8_0 needs ~309 s for a 200-token answer, so *quantization* is what makes the turn usable (Q4_K_M
265 s; the chat turn 294->195 s). The combined `pp2048+tg128` point runs ~10-15% below the naive
TTFT+stream sum, the 128 tokens decoded after a 2048-token prompt read a full KV cache and stream
slower than tg64-from-empty, so the `total turn` figures are uniformly optimistic by that margin.
Best interactive = NPU (short TTFT) + Q4_K_M (fastest stream).

**Faithfulness: differential perplexity, wikitext test, 12 chunks (same GGUF, NPU prefill vs CPU).**

| | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| Q8_0   | 6.6391 | 6.6330 | −0.09% |
| Q4_K_M | 6.7688 | 6.7479 | −0.31% |

Unlike the instruct/reasoning models in this record (Gemma-4-12B ~630, Phi-4-mini ~11), Phi-4 (14B)
scores raw wikitext at a **normal absolute PPL (~6.6-6.8)**; it models the text like a near-base
model, so here the absolute value is informative, not just the delta. Both NPU−CPU deltas (−0.09%,
−0.31%) land far inside the per-run stderr (±0.29 PPL, ~±4%) and in the NPU's favor (lower PPL), so
the fp16-accumulation prefill path is faithful for both Q8_0 and Q4_K_M, no accuracy lost to the NPU
matmul.

**Verdict.** Phi-4 (14B) is a mid-large edge-board fit, the quality step up from Phi-4-mini when
~9 GB of RAM is free. With F16 out of reach on a 31 GB board, **Q4_K_M + NPU** is the pick: 8.28 GiB,
~3.5x NPU prefill at length (~12 t/s), and a 2.2 t/s stream, the fastest effective turn on both
scenarios. **Q8_0 + NPU** (14.51 GiB) is the near-lossless option when accuracy matters more than
stream speed (1.4 vs 2.2 t/s), at essentially the same prefill. The NPU's value here is unambiguously
prefill: it turns a ~9.5-minute first-token wait on a 2048-token RAG prompt (Q8_0 CPU) into
~2.8 minutes.

### SmolVLM2-2.2B-Instruct (vision-language, mtmd)

The first **multimodal** model in this record, and the one that stretches the stack across both
its pillars: a **SigLIP-SO400M vision encoder** (the SigLIP pillar) feeding a **SmolLM2-class
language model** (the LLM prefill pillar), run end-to-end through llama.cpp's `mtmd` path. GGUFs
from `ggml-org/SmolVLM2-2.2B-Instruct-GGUF`: the language model (arch `llama`, **1.81 B**, the
"2.2B" is the combined vision+LLM count) as F16 / Q8_0 / Q4_K_M, and the vision tower as an fp16
`mmproj` GGUF. The two halves take two different NPU routes, measured separately.
[HW sweep, 600 MHz, 2026-07-04].

**Vision encoder (SigLIP-SO400M): clip encode latency, warm, the mtmd / SigLIP-pillar half.**
The vision tower runs through `clip.cpp`, whose ggml graph is scheduled over a `[backend, CPU]`
pair. The NPU is attached with **`MTMD_BACKEND_DEVICE=ROCKET`**: clip's auto-path only probes
GPU/IGPU *device types*, and rocket is an ACCEL device, so without the explicit selector the
vision encoder silently stays on the CPU.

| encoder (hidden / layers / tokens) | graph splits | CPU | NPU | speedup |
|---|---|---|---|---|
| SigLIP-SO400M (1152 / 27 / 729) | 1 -> **272** | 7.90 s | **6.66 s** | **1.19x** |

The NPU wins the encode ~1.19x warm (6.66 vs 7.90 s), right at the small-encoder end of the
Whisper crossover (the ASR section's tiny.en is 1.18x), a modest but real win, and it is
**warm-only**: the cold single-encode reads 8.12 s (~1.03x *slower* than CPU)
because the NPU clock parks at idle, so warm discipline is load-bearing here. The win is *capped*
by the graph shattering into **272 CPU↔NPU splits** (vs 1 pure-CPU): ggml-rocket offloads only
static-weight GEMMs clearing K%32 / N%16, which per SigLIP layer admits just the q/k/v/o
projections and fc1. The **fc2** down-projection (K=4304, 4304%32=16) misses the alignment gate;
the **attention** QK^T/score·V matmuls are excluded by design (their src0 is a computed tensor,
not a weight); and every norm / softmax / GELU / patch-embed op is a type ggml-rocket does not
implement; all of which stay on the CPU and bracket each offloaded GEMM with a handoff. The
`rocket_siglip_encoder` native driver runs the *whole* SigLIP-B/16 encoder resident on the NPU at
far higher efficiency; this is the cost of the generic ggml drop-in, which offloads op-by-op
through the stock clip graph rather than as one resident encoder.

**Prefill (prompt processing), t/s (the LLM half; the NPU's job).**

| test | F16 CPU->NPU | Q8_0 CPU->NPU | Q4_K_M CPU->NPU |
|---|---|---|---|
| pp512  | 27.97 -> **98.98** (3.5x) | 26.78 -> 47.93 (1.8x) | 27.01 -> 48.57 (1.8x) |
| pp1024 | 29.46 -> **71.99** (2.4x) | 27.29 -> 47.11 (1.7x) | 26.25 -> 47.43 (1.8x) |
| pp2048 | 27.45 -> **52.59** (1.9x) | 23.81 -> 39.90 (1.7x) | 24.61 -> 40.56 (1.6x) |

F16 pp512 hits **98.98 t/s, the fastest prefill in this record** (past Llama-3.2-3B's 61.8),
because at 1.81 B this is the smallest LLM here: fewest FLOPs per matmul. The F16 win *falls*
steeply with prompt length (3.5x->1.9x, the NPU rate 99->53), the small-model pattern (Llama-3.2-3B,
Phi-4-mini) at its extreme: per-op readback and dispatch are a large fixed fraction and grow with
M. The quants read a *lower* multiplier (~1.6-1.8x) than the ~2x small-dense cohort, again the CPU
denominator not a weaker NPU; this model's CPU quant prefill is strong (~24-27 t/s, near its F16
CPU), while the NPU quant absolute (~40-48 t/s) is in line. By pp2048 F16 (53) still leads the
quants (~40): at this operating point a quantized GGUF does not prefill faster than F16, it only
fits smaller ([not-mac-bound.md](not-mac-bound.md)).

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).**

| | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| tg64 CPU | 5.11 | 9.60 | 14.66 |
| tg64 NPU | 5.19 | 9.83 | 14.54 |

NPU decode matches the CPU, and stays off the NPU ([decode-gemv.md](decode-gemv.md)), and at 1.81 B it is the
**briskest stream in the record**: F16 5.1 -> Q8_0 9.7 -> **Q4_K_M 14.5 t/s** from a **1.03 GB** file.
(The Q4_K_M tg64 spread is ±3.3 t/s, real run-to-run jitter at this fast decode rate on a small
model; NPU 14.54 vs CPU 14.66 sits inside it.)

**Interactive.** TTFT and streaming derived from the measured pp/tg rates.

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | F16 CPU | 75 s | 5.1 t/s | 114 s |
| | F16 + NPU | **39 s** | 5.2 t/s | 78 s |
| | Q8_0 + NPU | 51 s | 9.8 t/s | 72 s |
| | Q4_K_M + NPU | 50 s | 14.5 t/s | **64 s** |
| chat, 128-tok prompt -> 400 out | F16 + NPU | 1 s | 5.2 t/s | 78 s |
| | Q4_K_M + NPU | 3 s | 14.5 t/s | **30 s** |

The NPU nearly halves the long-prompt first-token wait (F16 75->39 s); on the short-prompt chat the
turn is decode-bound, so quantization is the lever (F16 78 -> Q4_K_M 30 s). The combined
`pp2048+tg128` llama-bench point (F16 NPU 27.06, Q8_0 26.11, Q4_K_M 27.68 t/s) validates the
decomposition.

**Faithfulness.** Two checks, because two backends carry the model.

*LLM, differential perplexity (wikitext test, 12 chunks, same GGUF, NPU prefill vs CPU):*

| | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| F16    | 12.1622 | 12.1540 | −0.07% |
| Q4_K_M | 12.7622 | 12.6711 | −0.71% |

Both deltas sit far inside the ±0.60-0.63 PPL stderr (the F16 delta is ~75x smaller than its
stderr). The absolute ~12.2 is inflated, an instruction-tuned model on raw wikitext, so only the
delta is informative. NPU-faithful.

*Vision, projected image-embedding cosine (vision on NPU vs CPU):* the 81 projected image tokens
clip hands to the LLM match at **cosine 0.99998** (per-token min 0.99967), the SigLIP-pillar bar
(0.999998 native, 0.9998 Whisper). The greedy *captions* diverge in wording (both correctly read
the same NYT moon-landing front page) because autoregressive decode amplifies the ~0.5%
embedding perturbation into a different-but-equivalent token path, as with Whisper, the encoder
metric is the cosine, not token-identity.

**Verdict.** SmolVLM2-2.2B runs end-to-end on the FOSS NPU stack, vision encoder *and* LLM, and
posts the record's **fastest LLM prefill (99 t/s F16 pp512, 3.5x)** and **fastest stream (14.5 t/s
Q4_K_M)** by being the smallest model here. **Q4_K_M is the pick**: a **1.03 GB** LLM + ~0.8 GB
fp16 mmproj fits any board, with ~1.7x NPU prefill and 14.5 t/s decode. The gap-finder result on
the vision half: the SigLIP encoder offload *works and is faithful* (cosine 0.99998) but only wins
~1.19x, the SO400M dims (fc2 K=4304 off the %32 grid) and the static-weight-only offload gate
leave the attention core, fc2, and all norm/act ops on the CPU, so the graph pays 272 handoffs.
The unlock is not a new dtype but a **resident vision-encoder path** (the `rocket_siglip_encoder`
pattern) wired into the mtmd frontend, deferred; the drop-in clip route is faithful and modestly
positive as-is.

## ASR (whisper.cpp via ggml-rocket)

Whisper through the `ggml-rocket` drop-in `.so` on stock whisper.cpp, no fork. The NPU's job in
an ASR pipeline is the **encoder**: a stack of self-attention + MLP blocks, all large-M `MUL_MAT`,
run once per 30 s audio window. The **decoder** is autoregressive (M=1 GEMV per token) and stays on
the CPU on both backends ([decode-gemv.md](decode-gemv.md)), like LLM decode. So the benchmark is
**encoder latency**: `whisper-bench -w 0`, which times the encoder in isolation, CPU vs FOSS-NPU,
across the model-size ladder. Faithfulness here is the transcript (WER) and the encoder-output
cosine, not perplexity. [HW sweep, 600 MHz, 2026-07-04].

**Encoder latency: whisper-bench encode, warm mean of reps 2-4, ms, the NPU's job.**

| model | encoder d_model / layers | CPU -> NPU |
|---|---|---|
| tiny.en        | 384 / 4  |   697.7 ->   591.9 (1.18x) |
| base.en        | 512 / 6  |  1580.0 ->  1223.4 (1.29x) |
| small.en       | 768 / 12 |  5711.5 ->  3718.9 (1.54x) |
| medium.en      | 1024 / 24 | 19268.2 -> 10444.9 (1.84x) |
| large-v3       | 1280 / 32 | 36399.9 -> 17013.5 (**2.14x**) |
| large-v3-turbo | 1280 / 32 | 32957.2 -> 15544.1 (**2.12x**) |

The NPU wins at every size, and the win **grows monotonically with the encoder**: 1.18x at tiny
to 2.1x at large. Same mechanism as the crossover the encoder RE work predicted
([encodings/whisper-encoder.md](../encodings/whisper-encoder.md)): the matmul work grows as
d_model², while the CPU-side host packing/readback grows ~linearly, so the NPU gains as the encoder
scales. A small encoder (tiny/base) is dominated by fixed per-op host cost interleaved with the
CPU-only conv/LayerNorm/softmax front-end; a large encoder is matmul-heavy, where the NPU pays off.
(This supersedes the earlier "base.en below break-even" reading from the pre-KACC, per-call-fd
integration, the shipped dispatch-floor and K-accumulation work moved the whole ladder above 1x.)

**Featured: large-v3-turbo.** turbo keeps large-v3's full **32-layer encoder** but replaces the
32-layer decoder with a **4-layer** one, so the (NPU-accelerated) encoder becomes a much larger
share of a real transcription. The per-step decoder cost bears this out (whisper-bench full timing,
same NPU run): decode **19 ms/step** for turbo vs **116 ms/step** for large-v3 (~6x lower), while
the encoders time nearly the same (15.5 s vs 17.0 s NPU). So the encoder's ~2.1x NPU speedup drives
proportionally more of turbo's whole-pipeline time than full large-v3's, turbo is the model where
NPU encoder offload matters most.

**Faithfulness: transcript agreement and encoder cosine.**

| | jfk.wav transcript (greedy) |
|---|---|
| base.en CPU vs NPU | **byte-identical** (WER 0) |
| large-v3-turbo CPU vs NPU | **byte-identical** (WER 0) |

The FOSS-NPU encoder output matches whisper.cpp's real base.en encoder at **cosine 0.9998**
([encodings/whisper-encoder.md](../encodings/whisper-encoder.md)), the fp16-accumulation fidelity
that produces the identical transcript. The full encoder (LN / attention / softmax / GELU /
residual) was separately validated byte-identical on the NPU across tiny/base/small/medium.

**Resident-weight cache and whisper's tensor names (a fixed correctness trap).** whisper.cpp leaves
its weight tensors unnamed, so ggml auto-names them `leaf_%d` by graph position, and the same
`leaf_N` string names *different* weights across whisper's separate conv/encode/cross/decode graphs.
ggml-rocket's resident-weight cache keys on the weight name, so it would serve one weight's packed
tiles for another's matmul: a fast but **garbage** encode (whisper-bench times the encode without
checking it, so the broken config still benchmarks, the transcript is what catches it). Fixed in
`rocket_weight_key` by rejecting ggml-default `leaf_`/`node_` names (they stream per call instead of
caching, correct, and timing-neutral for a single-pass encoder, so the numbers above are the same
with or without the fix). llama.cpp weights carry real names (`blk.N.*`) and never match, so LLM
resident caching and prefill speed are unchanged (Llama-3.2-3B F16 pp512 stays ~61 t/s, within
run-to-run noise of the 61.79 headline).

**Verdict.** Whisper encoder offload is a CPU-faithful accelerator whose win scales with model size:
~1.2x at tiny/base, ~2.1x at large-v3 / large-v3-turbo, transcripts byte-identical to the CPU. It is
a true drop-in, stock `whisper-bench` / `whisper-cli` load the `.so` via `GGML_BACKEND_PATH`, no
fork. Raw data + full per-rep timings in [data/whisper-encoder.md](data/whisper-encoder.md).

## Speech-to-text, multi-model (transcribe.cpp via ggml-rocket)

The `ggml-rocket` `.so` is not whisper-specific. It drops into **transcribe.cpp** (handy-computer,
MIT, a ggml-based host for 16+ STT families: Whisper, Parakeet, Canary, Granite-Speech, MOSS
diarize, Voxtral, SenseVoice, FunASR) with no fork: transcribe.cpp vendors a ggml whose
`GGML_BACKEND_API_VERSION` matches this backend, the NPU registers as an ACCEL device, and each
model's runner logs `using accel backend: ROCKET` and offloads the encoder. This section is the
cross-model result; it uses two clips, `jfk.wav` (11 s, clean) and a hard 120 s two-speaker
conversational recording, A76-pinned (`taskset -c 4-7 --threads 4`), Q8_0, warm, fresh
single-process runs (no session reuse), `ROCKET_KACC=1`. [HW sweep, A/B, 2026-07-21].

### The offload taxonomy: encoders, not autoregressive decode

One law decides every STT model's NPU win:

- **Encode always offloads**, 1.2x-2.7x, scaling with encoder size × audio length. A big encoder
  (Voxtral's Whisper-large-v3, MOSS's Whisper-Medium) offloads harder than the small SAN-M encoder
  (SenseVoice/FunASR).
- **Autoregressive decode is M=1 GEMV and mostly stays on the CPU** ([decode-gemv.md](decode-gemv.md)).
  The exception is the **batched decode-prefill over the injected audio context** on long audio: a
  real matmul that offloads when it is large enough. So decode offload is a spectrum:
  - **Cross-attention decoder (Granite-Speech):** every step's cross-attn over the encoder output is
    a large-K matmul -> decode offloads best (~1.9x on the long clip).
  - **Big decoder-only + long audio (Voxtral 3B):** the 1500-token audio prefill through 3B is a
    large offloadable chunk -> decode 1.46x on the long clip, but 1.00x on jfk (prefill too short).
  - **Small decoder-only (MOSS/FunASR 0.6B):** prefill is a small, dequant-bound slice -> decode
    1.00x (MOSS) to 1.12x (FunASR).
  - **Single-pass CTC (SenseVoice):** no decode at all (48 ms even on 120 s).

So the NPU speedup ~ (encoder fraction × encoder offload) + (prefill fraction × prefill offload).
Decode-bound small-decoder models barely move; encode-heavy or big-prefill models win more.

### Speed: jfk (11 s, clean)

| Model | arch | CPU | NPU | NPU × | encode C->N | decode C->N |
|---|---|--:|--:|--:|--|--|
| SenseVoice-small 234M | SAN-M + CTC (single-pass) | 1367 ms | 1369 ms | 1.00x | 1324->1326 (1.00x) | 5->5 ms |
| Fun-ASR-nano 800M | SAN-M enc + Qwen3-0.6B AR | 4772 ms | 4739 ms | 1.01x | 1237->1239 (1.00x) | 3497->3462 (1.01x) |
| MOSS 0.9B | Whisper-Med + Qwen3-0.6B AR | 21198 ms | 15514 ms | 1.37x | 14876->9166 (1.62x) | 6273->6300 (1.00x) |
| Voxtral-mini 3B | Whisper-lg-v3 + Ministral-3B AR | 62133 ms | 49193 ms | 1.26x | 29350->16306 (1.80x) | 32726->32832 (1.00x) |

On jfk no decode offloads (the audio context is too short to make a large prefill), so the NPU win
is purely the encoder; it tracks the encoder's share of runtime. SenseVoice/FunASR are
encoder-tiny or decode-dominated -> ~1.0x; MOSS/Voxtral carry big encoders -> ~1.3x.

### Speed: 120 s conversational clip

| Model | CPU | NPU | NPU × | rt (NPU) | encode C->N | decode C->N |
|---|--:|--:|--:|--:|--|--|
| SenseVoice-small 234M | 24.37 s | 19.41 s | 1.26x | 6.2x | 23904->18939 (1.26x) | 48->48 ms |
| Fun-ASR-nano 800M | 78.75 s | 68.42 s | 1.15x | 1.75x | 22843->18350 (1.25x) | 55490->49656 (1.12x) |
| MOSS 0.9B (diarize) | 306.21 s | 284.18 s | 1.08x | 0.42x | 60626->37029 (1.64x) | 245395->246969 (1.00x) |
| Voxtral-mini 3B | 340.48 s | 217.55 s | 1.57x | 0.55x | 117329->64396 (1.82x) | 222944->152947 (1.46x) |
| Granite-Speech-2B (base) | 114.5 s | 68.2 s | 1.68x | 1.76x | n/a | ~1.9x (cross-attn) |

Longer audio widens every win (bigger encoder matmuls; for Voxtral and Granite the decode-prefill
also offloads). Voxtral on the long clip is the standout non-cross-attn win (1.57x) because its 3B
decode-prefill over 1500 audio tokens offloads. **Parakeet** (FastConformer CTC-0.6B, F16,
A76-pinned) also runs, ~4.6x realtime on the long clip, 1.37x over CPU, but the win is
conv-capped: its convolution modules do not offload (ggml-rocket has no conv path) and stay on the
CPU.

### Quality and capability: the 120 s hard clip

| Model | transcript accuracy | diarization | timestamps | translation | CPU==NPU |
|---|---|---|---|---|---|
| SenseVoice-small | worst: garbled run-on (fed past its 30 s window) | no | no | no | ~ (tiny CTC drift) |
| Fun-ASR-nano | semi-fluent, garbles hard names | no | no | no | ~ (tiny AR drift) |
| MOSS | good, coherent | **yes: 33 seg, clean two-speaker** | **yes, fine** | no | no (32 vs 33 seg, both good) |
| Voxtral-mini 3B | **best: cleanest, correct names, complete** | no | no | **yes** | **yes (char-identical)** |
| Granite-Speech-2B base | clean but drops ~40 s of the middle | plus variant: 8 coarse turns | no | no | no (AR decode diverges) |

- **Voxtral** is the best transcriber; it alone recovers every proper name and transcribes the full
  120 s without dropping spans, but emits flat text (no speaker turns, no timestamps). It is the
  only model here that **translates** (de->en and ja->en, both accurate and char-identical CPU vs NPU).
- **MOSS** is the best diarizer by a wide margin: 33 fine segments with timestamps, cleanly splitting
  the two speakers, coherent, never degenerates.
- **Granite-Speech** ships three variants, **base** (fastest, cleanest text, CPU==NPU and Q8==F16
  char-identical), **NAR** (a non-autoregressive editor that runs iterative refinement, so its decode
  is *bigger* than base's, ~1.4x slower and rougher, not the NPU speedster the name suggests), and
  **plus** (the diarization variant: 8 coarse speaker turns, no timestamps, coarser than MOSS).

**AR-decode divergence.** Encoder-only offload is bit-faithful (SenseVoice, MOSS, Voxtral CPU==NPU),
but a model whose decoder cross-attn offloads (Granite) is **not** guaranteed CPU==NPU: the fp16
cross-attn accumulation perturbs logits and greedy decoding amplifies it. On the hard clip
Granite-plus even degenerates into a repetition loop on the CPU while staying coherent on the NPU,
divergence tracks the model's own decode instability, not an NPU fault. Voxtral, by contrast, stays
char-identical CPU vs NPU throughout.

### MOSS: why the best diarizer barely moves on the NPU

MOSS-Transcribe-Diarize (0.9B: Whisper-Medium encoder -> 4x temporal merge -> VQ adaptor -> Qwen3-0.6B
decoder, audio injected as KV tokens) is the best diarizer but **decode-bound**. Profiled with a
`TRANSCRIBE_PERF_DEBUG` patch (below), long-clip Q8_0 on the CPU: encode 60.5 s, decode 245.5 s
(**80% of runtime**) = prefill 23.4 s (1638-token batched audio+prompt, 9.5%) + **713 AR steps ×
311 ms = 221.6 s (90.5%)**. On the NPU: encode 60.5->36.9 s (**1.64x, offloads**), prefill 23.4->18.8 s
(**1.25x, partial, dequant-bound**), and the 713 M=1 steps do **not** offload (313 ms NPU ~311 ms
CPU). Net decode 245.4 vs 247.0 s = **1.00x**. Unlike Granite's cross-attention, MOSS injects audio
as KV tokens, so generation is inherently M=1 regardless of the long 1500-token context: the merged
context is not "too short," its prefill offloads, but one-token-at-a-time generation (90% of decode)
cannot. Measured levers all cap out: a quant re-sweep moves decode ~1.4% (Q8_0 keeps the best WER at
half F16's RAM -> stays default), `--spec-k-drafts` is a no-op (MOSS does not advertise
`supports_spec_decode`), threads are already A76-pinned. **MOSS has no cheap NPU lever**: the M=1 AR
wall is irreducible; the only NPU-side headroom is un-dequant-binding the prefill (~1.25x->~2x on 9%
of decode, i.e. a ~1.10x ceiling). Real fixes are algorithmic (speculative decode, a smaller decoder,
fewer emitted tokens). **The honest number: against a fair A76-pinned CPU baseline (306 s), MOSS on
the NPU is 1.08x** (an earlier untuned-CPU comparison overstated it).

**Profiling method.** `TRANSCRIBE_PERF_DEBUG=1` alone prints the per-step timings, step count, and
`T_enc`/prefill from lightweight `ggml_time_us()` markers; the rich per-op ENCODE/DECODE tables
install a per-node sched-eval callback that syncs after every op and inflates long-audio decode ~10x
(unusable past a few seconds). A local RK1 patch to `src/arch/moss/model.cpp` gates just that callback
behind a second flag, so `TRANSCRIBE_PERF_DEBUG=1` gives real per-step timing at full speed and the
op tables need `TRANSCRIBE_PERF_DEBUG=1 TRANSCRIBE_OP_PROF=1` (upstream-worthy).

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

Traps worth knowing: a `TRANSCRIBE_GGML_BACKEND_DL` build forces `GGML_NATIVE=OFF`, so without
`-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16` the CPU baseline ships zero dotprod/fp16 kernels (~1.3x
too slow, inflating the NPU speedup); and `transcribe-cli` only initializes the default backends for
`--list-devices`, not for transcription, a one-line `main.cpp` patch to call
`transcribe_init_backends_default()` before model load fixes the shared/DL build (upstream-worthy;
the default static build compiles the CPU backend in and does not need it).

### Verdict

The `ggml-rocket` drop-in generalizes cleanly from Whisper to the wider STT landscape. The frontier
it exposes: **SenseVoice** (fastest, lowest quality) -> **FunASR** -> **Granite-Speech** ->
**Voxtral** (best transcript, only translator) / **MOSS** (best diarizer, slowest). Q8_0 is the
default across the board, for STT it ties or beats F16 on both CPU and NPU (the models are less
NPU-matmul-dominated than LLM prefill, so the per-micro-batch dequant tax is small while the CPU-side
decode is memory-bound and Q8 moves half the bytes) at half the RAM. The NPU's contribution is the
encoder (and, on long audio, the offloadable decode-prefill); the autoregressive decode that
dominates the audio-LLMs stays a CPU problem, exactly as for LLM decode.

## Detection (tflite-rocket)

SSD-MobileDet (uint8) through the tflite-rocket external delegate on `librocketnpu`, RK3588 at
600 MHz, `native_int8=1`. Detection differs from LLM prefill: a single inference is bound by the
host's cube scatter and gather, and by neither compute nor submits. So its latency follows the
host path, and the NPU's larger value is throughput under a multi-camera pool, the regime
Frigate runs. Faithfulness here is COCO mAP rather than perplexity.

### Single-stream latency

Host-path changes, and moving the requant on chip, take MobileDet from 205 to 56 ms. Each row
is the warm latency once that change is in, on one operating point: RK1, 600 MHz, governor `performance`, the process on
the A76 cores, four rotated passes [HW sweep, 2026-09-27 and 2026-09-28].

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

Rows 3 to 5 leave every output byte identical, apart from SSD MobileNet v2's two 150×150
depthwise layers, which move to the int8 route. Rows 2 and 6 to 8 change which program computes
a layer, so outputs move there. COCO mAP over 100 images does not fall at any of them. At the
last row it is 0.4040, 0.3229 and 0.3023, against the CPU's 0.3980, 0.3215 and 0.2996. The last
row is one A/B on one build: 76.6 -> 55.7 ms, 65.6 -> 52.1 and 123.3 -> 101.3.

The same build unpinned, on `ondemand` with a 1.2 GHz floor and no `taskset`, reads 75.8, 70.5
and 135.0 ms. The governor parks the cores an offloading process leaves idle
([cpu-governor-and-offload.md](cpu-governor-and-offload.md)).

Three host costs carried most of the gap:

- The conv entries packed their cubes one element at a time. That loop was 83% of the int8
  depthwise call ([../depthwise-conv.md](../depthwise-conv.md)).
- Buffers were sized for the largest call and synced whole on every call
  ([bo-sync-cost.md](bo-sync-cost.md)).
- The int32 readback and the host requant. On MobileDet's direct convs the `read` phase is the
  int32 copy, 262 ms over 21 invokes against 19 ms of `FINI_BO`. The int8-out writer removes it
  ([../encodings/out-cvt-converter.md](../encodings/out-cvt-converter.md)).

The `tflite-rocket` README carries the per-change ratios.

**Faithfulness: COCO-val mAP, the detection correctness check.**

| | mAP@[.5:.95] |
|---|---|
| CPU (int8 reference) | 0.3318 |
| FOSS-NPU delegate | **0.3321** |

500 COCO val images, [tflite-rocket `tools/coco_map.py`]; Δ +0.0002, CPU parity. The delegate's
fp16-approx depthwise and per-tensor (not per-channel) DW int8 cost ~0 mAP, so the accelerated
detector is numerically as good as the CPU int8 reference.

**Throughput: multi-camera pool, aggregate detection_fps, the NPU's regime.** One detector
process per camera (`num_threads: 1`), each pinned to a distinct A76 core via
`ROCKET_CPU_AFFINITY` so the P contexts spread across the four big cores instead of colliding on
one; NPU IRQs pinned to the big cores (`npu_set_irq_affinity.sh throughput`). Live on the RK1
through Frigate's `rocket.py` detector, four SSD-MobileDet cameras. [HW sweep].

| pool size P | detection_fps | scaling |
|---|---|---|
| 1 | 3.20 | 1.00x |
| 2 | 6.00 | 1.88x |
| 3 | 8.00 | 2.50x |
| 4 | 9.55 | **2.98x** |

Aggregate throughput scales ~3x at P=4, the four A76 cores carrying one detector each (~73%
busy) while the A55s handle video decode. It tapers below linear (2.98x, not 4x) because a real
MobileDet inference is DRAM-bandwidth-bound in its host scatter/gather phase: per-inference
latency rises 338->424 ms as the four contexts contend for bandwidth, so the live pool tracks
below the submit-bound delegate ceiling (`tools/pool_throughput.py`: 1.00 / 2.17 / 3.11 / 3.56x).
The pool needs no delegate or driver change; it is exactly how Frigate runs cameras (one process
each).

The pool table was measured when a single inference took ~336 ms. It is not re-measured on the
host path above, so its absolute rates and its scaling are that build's.

**Verdict.** For detection the FOSS-NPU delegate is a CPU-parity-accuracy accelerator whose value
is throughput: run one pinned process per camera and it serves ~3x the aggregate detection rate of
a single stream, offloading the conv/matmul work of four cameras to the NPU so the A76 cluster is
free and the A55s handle decode. Single-stream latency is bound by host work, so its levers are
the host path's, as the table above shows, and not the NPU submit path.

## ONNX Runtime (ort-rocket)

`ort-rocket` is an ONNX Runtime execution provider on `librocketnpu`. It claims whole encoders
and single ConvTranspose nodes, and leaves the rest of a graph on the CPU EP. Its README owns the
full per-model table. The figures here are warm, on the RK1 at 600 MHz, against the CPU EP at
`ORT_ENABLE_ALL` [HW sweep].

| Model | Offloaded | Single stream, x the CPU EP | Faithfulness against the CPU EP |
|---|---|---:|---|
| SigLIP-B/16 | plain ViT, d=768, 196 tokens | 1.27x | hidden-state cosine 0.9999864 |
| RF-DETR base | windowed DINOv2 ViT-S, d=384 | 1.05x | COCO mAP 0.564 against 0.564 |
| SAM ViT-B | ViT-Det, d=768, 4096 tokens | 1.03-1.11x | mask IoU 0.9998 |
| Depth Anything v2 Small | DINOv2 ViT-S, global attention over 1370 tokens | 0.78x | depth cosine 0.9999999 |
| Laya English | ModernBERT-large, d=1024, 48-512 tokens | 1.66-3.11x | 4 of 4 answers agree |
| pix2pix generator | ConvTranspose nodes, 65% of the CPU wall | 1.76x at 4 threads, 2.15x at 2 | PSNR 80.8 dB |

A model's single-stream ratio follows its attention-to-projection ratio, `ntok / 4d`, with a
windowed layer counted at its window length. The NPU runs the large-K projection GEMMs about 2.7x
more efficiently per MAC than attention. So the one global-attention encoder loses, and the wide
text encoder gains most. ONNX Runtime's CPU kernels run the Laya graph at ~77 GOP/s, which is
why its ratio sits above what the vision models predict.

### ONNX Runtime's spinning threads cost the EP

ONNX Runtime's intra-op threads spin while idle by default. An offloaded node leaves them idle,
so they spin on the A76 cores the EP's host threads and NPU workers need [hypothesis]. With
`session.intra_op.allow_spinning` set to `0` the EP arm gains and the CPU arm does not move
[HW sweep, mainline RK1, `taskset -c 4-7`, governor `performance`, 2026-09-27]:

| Case | Default session | Spinning off |
|---|---:|---:|
| Laya, one 58-token request, EP | 137-139 ms | 118-121 ms |
| pix2pix, 4 threads, EP | 99.9 ms, 1.46x the CPU EP | 82.9 ms, 1.76x |
| pix2pix, 4 threads, CPU EP | 145.5 ms | 146.6 ms |
| SAM mask decoder, EP, x the CPU EP | 1.017x | 1.049x |

At 2 threads pix2pix shows no difference. A 726-token Laya request varies +-10% between passes
either way, so that cell is unresolved. An execution provider cannot change a session's options,
so the application sets this one. The ViT and Laya rows in the table above were taken under the
default session, and the pix2pix row with spinning off.

```python
so.add_session_config_entry("session.intra_op.allow_spinning", "0")
```

This is the same class of bias as a load-sampling CPU governor
([cpu-governor-and-offload.md](cpu-governor-and-offload.md)): a host-side default that costs
the offloading arm and leaves the CPU arm alone.
