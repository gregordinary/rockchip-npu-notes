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

**Conditions:** RK3588 on a Turing RK1 (4xA76 + 4xA55, shared LPDDR), NPU clock 600 MHz under the
clock patch ([clock.md](clock.md)). Figures are warm medians, and each comparison runs the same
GGUF or model on the NPU and on the CPU (8 threads) [HW sweep].

> [!NOTE]
> The NPU clock returns to its 200 MHz idle rate between runs, so a cold run measures ~15% low.
> Run a discarded warm-up first, and compare warm runs.

### LLM and multimodal (llama.cpp via ggml-rocket)

F16 prefill is fastest on short prompts, and slows as output-tile readback grows with M at longer
prompts. A quantized GGUF's prefill instead rises with prompt length, as its per-micro-batch
dequantization amortizes (`-b 2048 -ub 2048`,
[quant-prefill-microbatch.md](quant-prefill-microbatch.md)). Rows are sorted by size. In the table:

- Prefill is F16, a model's fastest prefill, where F16 fits a board. Otherwise it is the top
  precision that runs, tagged with that precision. Q4 is Q4_K_M, MXFP4 is the gpt-oss native
  block-float, and the tagged models are all F16-untenable on a 31 GB board.
- The two prefill columns show pp512 -> pp2048, to show the trend with prompt length.
- Decode and Fits are Q4_K_M, the fastest decode and the smallest fit.
- PPL Δ is NPU−CPU on the same GGUF.

| Model | Params | Prefill NPU t/s, pp512->2048 | x CPU, pp512->2048 | Decode Q4 t/s | Fits Q4, GB | NPU−CPU PPL Δ |
|---|---:|---:|---:|---:|---:|---:|
| SmolVLM2-2.2B (llama+SigLIP) | 1.8B | 99.0 -> 52.6 | 3.5x -> 1.9x | 14.5 | 1.0 | −0.07% |
| Llama-3.2-3B | 3.2B | 61.8 -> 45.5 | 3.4x -> 2.6x | 7.9 | 1.9 | +0.02% |
| Ministral-3-3B | 3.4B | 57.6 -> 39.8 | 3.2x -> 2.4x | 7.7 | 2.0 | −0.04% |
| Phi-4-mini | 3.8B | 55.4 -> 39.8 | 3.4x -> 2.6x | 7.0 | 2.3 | −0.12% |
| Ministral-3-8B | 8.5B | 27.4 -> 21.4 | 3.8x -> 3.1x | 3.8 | 4.8 | −0.01% |
| Qwen3.5-9B | 9.0B | 25.9 -> 24.9 | 3.6x -> 3.5x | 3.6 | 5.3 | +0.05% |
| Gemma-4-12B | 11.9B | 17.4 -> 15.0 | 3.6x -> 3.2x | 2.5 | 6.9 | −0.81% |
| Phi-4 (14B) | 14.7B | 10.3 -> 11.8 (Q4) | 2.9x -> 3.5x | 2.2 | 8.3 | −0.31% |
| DeepSeek-V2-Lite (MoE+MLA) | 15.7B | 26.0 -> 28.4 (Q4) ‡ | 1.33x -> 1.53x | 7.8 | 9.7 | −0.42% |
| gpt-oss-20b (MoE) | 20.9B | 22.0 -> 28.1 (MXFP4) ‡ | 1.81x -> 2.38x | 7.2 | 11.3 | cosine 0.9998 † |
| Qwen3.6-27B (hybrid) | 27.3B | 5.4 -> 7.8 (Q4) | 3.0x -> 4.4x | 1.1 | 15.9 | −0.26% |

**Conditions:** the CPU column is the same llama.cpp binary with the backend unloaded, unpinned.
F16 runs at the llama.cpp default `-ub 512`, and a quantized GGUF at `-b 2048 -ub 2048`. The two
MoE prefill rows are 2026-08-27 builds (‡).

† gpt-oss is a reasoning model, so wikitext PPL is not a valid metric for it. Its faithfulness
figure is the per-matmul cosine against an fp64 reference.

‡ A 2026-10-09 `librocketnpu` build packs the experts faster and runs the group-wise int8 readback
as NEON. Under governor `ondemand` it measures gpt-oss pp512 at 24.90-25.23 t/s and
DeepSeek-V2-Lite pp2048 at 32.69-32.83 t/s (§"gpt-oss-20b (MXFP4, MoE)"). Its x CPU ratio is not
characterized.

Notes on the table:

- Phi-4 (14B) also runs Q8_0 at essentially the same prefill.
- Qwen3.5-9B also runs IQ4_XS (decode 4.1 t/s, 4.8 GB, same NPU path).
- SmolVLM2's 1.0 GB is its 1.81 B LLM half, plus ~0.8 GB fp16 vision mmproj.

> [!NOTE]
> The quantized rows' x CPU column compares against an unrepacked CPU. The NPU configuration must
> run without llama.cpp's weight repack, and the repack speeds the CPU configuration 1.26-1.71x at
> pp2048 on these four models. Against a repacked CPU at pp2048 the NPU leads Phi-4 3.06x and
> DeepSeek-V2-Lite 1.44x. It leads gpt-oss-20b 1.90x and Qwen3.6-27B 3.21x [HW sweep 2026-09-29,
> llama.cpp b11242, governor pinned]. [cpu-repack-baseline.md](cpu-repack-baseline.md) has the
> method. The F16 rows are unaffected.

The prefill multiple grows with model size, up to Qwen3.6-27B's 4.4x at pp2048, the largest here.
It grows because the CPU baseline degrades faster than the NPU as the matmuls grow. The exceptions
are architectural.

The MoE expert FFNs of gpt-oss-20b and DeepSeek-V2-Lite run through `MUL_MAT_ID`, whose NPU path
holds each quantized expert resident as native int8. Residency removes the per-micro-batch host
dequantization that makes the naive expert offload a loss. The path is on by default from the
2026-08-27 build. On that build it takes gpt-oss to 2.38x at pp2048, against 1.15x with the experts
on the CPU.

The default claims an expert op only where two conditions hold. A residency pre-flight must be
able to reserve the whole expert stack before ingesting it. The per-expert GEMM must also contain
enough work to repay its own dispatch cost (`M_e · K · N`, where `M_e` =
`n_tokens · n_used / n_expert`). DeepSeek's experts (6-of-64 routing, 2048×1408 each) therefore
stay on the CPU at short prefill, and move onto the NPU past ~1250 tokens.

DeepSeek adds MLA attention. The flash-attention (FA) offload accepts DK≠DV and is bit-faithful.
Its DeepSeek case is not exercised on the device, so attention stays on the CPU here. Qwen3.6-27B's
Gated-DeltaNet hybrid keeps its linear-attention layers on the CPU, and they do not limit the
speedup. The two MoE models decode faster than their parameter counts suggest (gpt-oss 7.2 t/s,
DeepSeek 7.8 t/s), because only ~3.6 B (gpt-oss) and ~2.4 B (DeepSeek) params are active per token.

Every NPU−CPU PPL Δ sits within its per-run stderr, so the fp16-accumulation prefill is faithful.
The absolute PPL of an instruct or reasoning model is not meaningful, and only the delta is.

SmolVLM2's vision half (SigLIP-SO400M, 729 tokens) also offloads, through clip.cpp. It runs in
7.90 -> 6.66 s warm (1.19x), with a projected-embedding cosine of 0.99998 against the CPU. The
speedup is modest, limited by the graph breaking into 272 CPU↔NPU splits. The figure is for the
generic drop-in, not the resident `rocket_siglip_encoder`. The offload requires
`MTMD_BACKEND_DEVICE=ROCKET`.

### What tuning buys per model (stock -> recommended, pp2048)

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

**Conditions:** pp2048, governor pinned, both configurations on the NPU.

The table abbreviates the options. In it, `-ub 2048` means `-b 2048 -ub 2048`. The entry
`QUANT_RESIDENT` is `ROCKET_QUANT_RESIDENT=auto`, which dequantizes each weight once and holds it
resident as fp16. It needs the fp16 image in RAM. The entry `F16_RESIDENT` is
`ROCKET_F16_RESIDENT=auto`.

The Qwen3.5-0.8B row uses the combined form, and its residency-only form was not measured. Every
other resident row is residency alone, `Gemma-4-12B` included. Gemma-4-12B places only part of its
weights, and residency alone is still faster there. It also places more of them: 83-87% resident,
against 73-74% for the combined form. The combined form, `-b 2048 -ub 2048`, uses that RAM for
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
| base.en | 512 / 6 | 1580.0 -> 1223.4 ms | 1.29x |
| small.en | 768 / 12 | 5711.5 -> 3718.9 ms | 1.54x |
| medium.en | 1024 / 24 | 19268.2 -> 10444.9 ms | 1.84x |
| large-v3 | 1280 / 32 | 36399.9 -> 17013.5 ms | 2.14x |
| large-v3-turbo | 1280 / 32 | 32957.2 -> 15544.1 ms | 2.12x |

**Conditions:** encoder latency from `whisper-bench` encode-only, warm.

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
process per figure.

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
| COCO mAP@[.5:.95] | 0.4040 NPU vs 0.3980 CPU over 100 images (2026-09-28) |
| Single-stream latency, warm | 56 ms with the governor pinned, 76 ms on `ondemand` (2026-09-28) |
| Multi-camera pool, P=1->4 | 3.20 -> 9.55 detection_fps (2.98x at P=4), measured at a ~336 ms single stream |

## LLM (llama.cpp via ggml-rocket)

Each model reports prefill, which the NPU runs, and decode, which stays on the CPU on both
backends. An interactive table decomposes a turn into time-to-first-token (TTFT), set by prefill,
plus a streaming rate, set by decode. Faithfulness is a differential perplexity. The same GGUF runs
through the NPU prefill and through the CPU, and the NPU−CPU delta isolates the fp16-accumulation
fidelity of the NPU matmul.

**Conditions:** as for the LLM table in §"At a glance", unless a model's section states
otherwise.

### Ministral-3-8B-Instruct-2512

Ministral-3-8B is Mistral's December-2025 8B (arch `mistral3`, GQA, interleaved sliding-window
attention), with 8.49 B params. The GGUFs come from `unsloth/Ministral-3-8B-Instruct-2512-GGUF`,
and F16 is derived from the published BF16. The results below were measured at 600 MHz on
2026-07-01 [HW sweep].

**Prefill (prompt processing), t/s (the NPU's job).**

| test | F16 CPU->NPU | Q8_0 CPU->NPU | Q4_K_M CPU->NPU |
|---|---|---|---|
| pp512  | 7.2 -> **27.4** (3.8x) | 6.9 -> 17.4 (2.5x) | 6.5 -> 17.0 (2.6x) |
| pp1024 | 7.0 -> **24.4** (3.5x) | 6.7 -> 19.3 (2.9x) | 6.4 -> 19.1 (3.0x) |
| pp2048 | 6.8 -> **21.4** (3.1x) | 6.4 -> 17.7 (2.8x) | 6.0 -> 17.4 (2.9x) |

NPU prefill runs at ~3x the CPU rate on F16, and ~2.5-3x on the quantized GGUFs. F16 has the
fastest prefill (27 t/s), because a quantized GGUF re-dequantizes to fp16. Quantization therefore
does not raise prefill speed at this operating point, where the datatype-independent dispatch
overhead is the limit ([not-mac-bound.md](not-mac-bound.md)).

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).**

| | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| tg64 CPU | 1.37 | 2.46 | 3.76 |
| tg64 NPU | 1.43 | 2.47 | 3.83 |

Decode in the NPU configuration matches the CPU, and decode stays off the NPU
([decode-gemv.md](decode-gemv.md)). Its rate depends on the quantization, not the backend: F16 1.4
-> Q4_K_M 3.8 t/s (2.8x). Quantization is the only setting that raises decode speed.

**Interactive.** TTFT and streaming are derived from the measured pp and tg rates.

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | F16 CPU | 299 s | 1.4 t/s | 445 s |
| | F16 + NPU | **96 s** | 1.4 t/s | 236 s |
| | Q4_K_M + NPU | 118 s | 3.8 t/s | **170 s** |
| chat, 128-tok prompt -> 400 out | F16 + NPU | 5 s | 1.4 t/s | 285 s |
| | Q4_K_M + NPU | 8 s | 3.8 t/s | **112 s** |

With a long prompt, the NPU cuts the time to first token ~3x (F16 299->96 s) and nearly halves the
turn. In a short-prompt, long-reply chat the turn is decode-bound: the NPU changes it little, and
quantization shortens it (F16 285 -> Q4_K_M 112 s).

**Faithfulness: differential perplexity, wikitext test, 12 chunks (same GGUF, NPU prefill vs
CPU).**

| | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| F16    | 6.8621 | 6.8615 | −0.01% |
| Q4_K_M | 6.9626 | 6.9597 | −0.04% |

The per-chunk stderr is ±0.43, so the NPU−CPU delta is ~100x smaller than the noise. The NPU
prefill matches the CPU to within 0.05% on both GGUFs, so the fp16-accumulation path has no
measurable accuracy cost. The Q4_K_M−F16 gap (6.96 vs 6.86) is the expected quantization cost,
present equally on both backends.

**Verdict.** For a 16 GB+ board on batch or RAG workloads, use F16 + NPU (fastest prefill). For
most boards and mixed interactive use, use Q4_K_M + NPU. It keeps ~2.9x prefill and a 2.8x-faster
stream in 4.83 GB, which fits an 8 GB board. It is also faster on a mixed turn (combined
pp2048+tg128: Q4_K_M 12.3 t/s, F16 10.3 t/s). Q8_0 (9 GB) is the near-lossless option between
them.

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

**Prefill (prompt processing), t/s (MXFP4, `-b 2048 -ub 2048`).** The expert path is on by default
from the 2026-08-27 build, behind a residency pre-flight. Setting `ROCKET_MOE=1` also claims what
the pre-flight declines [HW sweep 2026-08-27, 600 MHz, governor `performance`, llama.cpp
171974745].

| test | CPU | NPU, `ROCKET_MOE=0` | **NPU, default** | NPU, `ROCKET_MOE=1` |
|---|---:|---:|---:|---:|
| pp512  | 12.17 | 14.08 (1.16x) | **22.0 (1.81x)** | 24.81 (2.04x) |
| pp2048 | 11.80 | 13.62 (1.15x) | **28.1 (2.38x)** | 34.01 (2.88x) |

> [!NOTE]
> The default configuration varies from run to run: 20.5-22.8 t/s at pp512 and 26.9-28.7 at
> pp2048, at identical placement (63 of 72 stacks, 1656 experts, 0 streamed). Average several runs
> before comparing it with another configuration.

The expert path accounts for this model's result. Holding the routed experts on the NPU as native
int8 gives 2.06x the experts-on-CPU configuration and 2.38x the CPU at pp2048 on the 2026-08-27
build. It is faster at every prefill length. CPU prefill is itself fast on this model (~12 t/s,
against ~7 for a dense 8B), because only ~3.6 B params are active per token. The 2.38x is therefore
against a strong baseline.

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
2026-08-27, 600 MHz]. With the experts on the CPU it measures 14.13 and 13.53. The expert path is
therefore about 1.6x and 1.7x faster at the llama.cpp default micro-batch. Against the `-ub 2048`
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

### The accept boundary, and why the per-expert row count is the wrong handle

The MoE placement threshold keeps a small expert GEMM on the CPU. It is a threshold on the work per
dispatch, `M_e · K · N`. A threshold on the per-expert row count, `M_e` =
`n_tokens · n_used / n_expert`, does not separate wins from losses across architectures [HW sweep
2026-08-27]. One lower bound on `M_e` is the tile granule, 64 rows, the smallest tile the resident
path can run. That bound comes from the mechanism, and the measured sign change is not at it.

The map below plots `M_e` against `default ÷ ROCKET_MOE=0` over `-p 512,768,1024,1536,2048` on both
MoE models.

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

**Conditions:** RK1, NPU clock 600 MHz pinned, governor `performance`, `-b 2048 -ub 2048`. Every
offloaded test point is 100% resident with 0 streamed, so no ratio here is a residency effect.

**Not characterized:** the `M_e` = 72 and 96 test points on the 2026-10-09 build. Its readback and
activation-pack changes reduce the expert call's cost on gpt-oss's shape
(§"gpt-oss-20b (MXFP4, MoE)"). The threshold is fitted on the 2026-08-27 measurements.

The gpt-oss model is 1.64x faster at `M_e` = 64, while DeepSeek is 5-6% slower at `M_e` = 72. Any
row threshold low enough to admit the first admits the second, and no row threshold separates
them. The loss at `M_e` = 72 belongs to the expert path, not to the threshold. Auto placement admits
every DeepSeek stack at every prefill length, so `ROCKET_MOE=1` is the same placement there. It
measures 25.55 against the same 26.35 control.

The threshold amortizes a per-dispatch cost, incurred once per `(op, expert-with-rows)`:

- One gather
- One per-`(row, K-group)` activation quantize
- One submit
- One fence
- One scatter

One dispatch performs `M_e · K · N` MACs. For the NPU to beat the CPU on that expert,
`fixed_dispatch < M_e · K · N · (1/rate_cpu − 1/rate_npu)` must hold. The condition is a threshold
on the product, and `M_e` stands in for it only while `K · N` is held constant. Across these two
models `K · N` is not constant. The gpt-oss expert GEMM is 2.88x larger than DeepSeek's at the same
row count, which accounts for the whole of the discrepancy above.

Ordered by that product, the test points of both architectures separate with no overlap:

- The one clear loss is at 2.08e8 MACs.
- The boundary test point at 2.77e8 is a marginal 1.049x (0.94-1.08 over its repeats), which does
  not repay its own ingest.
- Every test point with a material gain is at or over 4.15e8.

The default threshold is the geometric midpoint of that last gap, 340 mega-MACs.

### The third architecture, and the threshold declines it correctly

On Qwen3-30B-A3B, a third MoE architecture, the default declines the experts at `-ub` 2048 and
below. Forcing them onto the NPU with `ROCKET_MOE=1` measures 0.834x at `-ub` 2048 and 0.661x at
`-ub` 1024 [HW sweep 2026-08-27]. The decision is right on this model, and right for the reason
the threshold was derived from: the work per dispatch.

**Conditions:** RK1 (RK3588), NPU clock 600 MHz pinned, governor `performance`, Qwen3-30B-A3B,
`-b/-ub` as shown, 2026-08-27.

**Not characterized:** the threshold's position. At 201 MMAC the test point is far enough below
340 that it tests the threshold's sign, not its position. A model between 277 and 415 MMAC is what
would move it.

Qwen3-30B-A3B has 128 experts, 8 active, `n_embd` 2048 and `moe_intermediate` 768. Its `K · N`
is therefore 1.573e6, 5.3x smaller than gpt-oss's and 2.7x smaller than DeepSeek's. At `-ub` 2048
it reaches `M_e` = 128 and 201 mega-MACs, under the threshold. Clearing 340 needs `M_e` >= 217,
which is more than 3472 tokens in one micro-batch. The default therefore declines its experts at
`-ub` 2048 and below, and accepts them at `-ub` 4096, which the RK1 runs. The accept side is
measured below.

| test point | work/dispatch | configuration | t/s | ratio |
|---|---:|---|---:|---:|
| `ub` 2048, `M_e` 128 | 201 MMAC | `ROCKET_MOE=0` | 13.38 / 13.32 / 13.31 | n/a |
| | | **default** | 13.32 / 13.34 | **1.000x** |
| | | `ROCKET_MOE=1` | 11.31 / 10.98 / 11.08 | **0.834x** |
| `ub` 1024, `M_e` 64 | 101 MMAC | `ROCKET_MOE=0` | 14.62 | n/a |
| | | `ROCKET_MOE=1` | 9.66 | **0.661x** |

The default measures 1.000x the `ROCKET_MOE=0` configuration and prints no MoE line at all. Its
ingest is zero.

The conclusion does not depend on residency. Experts make up 29 of this model's 30.5 B parameters,
so it cannot be held resident on a 31 GB board at any budget. Under `ROCKET_MOE=1` the int8 ingest
stops at 9539 of 15277 experts, 62.4%, and the rest streams. Two results separate the work per
dispatch from residency:

- At matched residency, halving the work per dispatch lowers the ratio from 0.834 to 0.661. The
  `ub` 1024 test point holds residency at 62.5% against the `ub` 2048 test point's 62.4%, 9548
  experts against 9539. Work per dispatch alone moves the result by 0.17, and it behaves exactly as
  the cost model predicts.
- Raising residency to 70.8% with `ROCKET_MOE_CACHE_MB` raises the ratio only to 0.863. The
  forced configuration loses at every residency this board can reach.

The threshold's position rests on DeepSeek-V2-Lite's bracket alone. The gpt-oss model contributes
nothing to where it sits. Its smallest reachable test point, 5.31e8, is 1.56x the 3.4e8
threshold and above the bracket's upper end of 4.15e8 [host-computed]. The threshold's form is
derived and its value is fitted on two architectures, so it is a measured boundary rather than a
bound. The form does not contain `rate_cpu`, and the two models'
CPU kernels differ (MXFP4 against Q4_K). The gpt-oss model at 5.31e8 therefore wins 1.64x where
DeepSeek at 5.54e8 wins 1.31x, with the same work per dispatch and a different margin.

On a third architecture the threshold changes placement. A 128-expert, 8-used model with a
2048×768 expert is at 201 MMAC at `M_e` = 128, which a row threshold accepts and this one
declines.

The threshold demands a material gain rather than `> 1.00`, because of the one-time expert ingest.
The ingest takes ~18 s per `llama_context` on DeepSeek and ~21 s on gpt-oss [HW sweep 2026-10-09,
RK1, 600 MHz, governor `ondemand`]. The ingest is charged only on an accepted offload, so the
threshold is the one place where it can be avoided. An offload at ratio `r` saves `1 − 1/r` of
prefill wall-clock time, so the ingest breaks even after:

| test point | ratio | saves | breaks even after |
|---|---:|---:|---:|
| DeepSeek `M_e` = 96 | 1.05x | 4.8% | ~7 700 tokens of prefill at that shape |
| DeepSeek `M_e` = 144 | 1.22x | 18.0% | ~2 300 tokens |
| DeepSeek `M_e` = 192 | 1.31x | 23.7% | ~1 800 tokens |
| gpt-oss `M_e` = 64 | 1.64x | 39.0% | ~1 100 tokens |

**Conditions:** the ratios are the 2026-08 test points. Each break-even count is that test point's
count at the 2026-08 ingest, scaled by the model's ingest ratio [host-computed]. The ingest ratio,
current pack against earlier pack, is 0.47x on DeepSeek and 0.51x on gpt-oss.

A test point a few percent above parity therefore costs a short-lived context more than it saves.
The threshold is placed above 1.00 for that reason.

### gpt-oss-20b native-quant expert path and results

The native-quant expert path ingests each expert once into resident int8 codes, and its speedup
comes from that residency. The fp16 expert path (`ROCKET_MOE=1 ROCKET_MOE_NATIVE=0`) dequantizes
each quantized expert to fp16 on the host every micro-batch. It measures 4.59 at pp512 and 10.18 at
pp2048 [HW sweep 2026-07-14], worse than leaving the experts on the CPU. The CPU's fused quantized
kernel performs no dequantization. MXFP4 dequantizes correctly.

The cost of the per-micro-batch dequantization, and why it does not shrink with the row count, are
in [not-mac-bound.md](not-mac-bound.md) §"Quantized MoE experts and residency". That section also
shows that the int8 GEMM's own traffic is not the source of the gain.

**Conditions:** RK1 (RK3588), NPU clock 600 MHz, gpt-oss-20b MXFP4.

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
600 MHz, governor `ondemand`, `-p 512 -b 2048 -ub 2048` [HW sweep]. The pack runs ~1.3 GB/s and
grows with the bytes held resident, not with the expert count.

DeepSeek-V2-Lite packs 3.0x as many experts, holding ~14.9 GB, in 11.7-11.8 s. Its Q4_K->int8
decode beside that is 6.4-6.6 s, at `-p 2048 -ub 2048`. The benchmark tool `llama-bench` builds a
fresh context per test row, so it incurs the ingest per row, and a long-running host incurs it
once.

> [!NOTE]
> The `llama-bench` warm-up does not absorb all of the ingest. An expert the warm-up never routed
> to is ingested inside the first timed repetition that routes to it. A MoE prefill figure from
> `llama-bench` therefore includes part of the ingest.

The 2026-10-09 `librocketnpu` build runs the resident group-wise int8 readback as NEON. The int8
activation pack's group copy also stops compiling to a byte loop. End to end, on llama.cpp
a7fb71f with `--repack 0` and `-p 512 -n 0 -b 2048 -ub 2048`, gpt-oss measures pp512 24.90-25.23
t/s [HW sweep 2026-10-09]. The gain over the build before it is 1.093-1.098x, with the ingest
unchanged at 20.7-21.1 s. DeepSeek-V2-Lite at `-p 2048 -ub 2048` measures pp2048 32.69-32.83 t/s,
with an ingest of 18.2-18.4 s. The 2026-08-27 tables in this section and the previous one predate
both changes.

Attention stays on the CPU on this model, as a correctness requirement. The gpt-oss model has an
attention sink, a learned per-head logit that joins the softmax denominator. The NPU FLASH_ATTN
handler has no sink term, so the offload is declined for such ops. Declining it is also +26% at
pp2048, because the accepted offload takes real time to compute the wrong answer (see
[attention-offload-crossover.md](attention-offload-crossover.md)).

> [!CAUTION]
> An accepted FLASH_ATTN offload on a model with attention sinks silently computes a sink-less
> softmax past the `n_kv` threshold of 1024. The handler drops the sinks (`src[4]`), and the model
> still produces fluent text and a plausible perplexity.

**Decode (generation), t/s (CPU-bound on both).**

| | MXFP4 |
|---|---|
| tg64 CPU | 7.15 |
| tg64 NPU | 7.16 |

Decode stays on the CPU, as for every model here. The MoE's small active-parameter count makes it
comparatively fast (7.2 t/s, roughly Ministral's Q4_K_M rate at ~half the bytes touched).

**Interactive.** On a 2048-token prompt the default configuration reaches the first token in ~73 s,
against ~174 s on the CPU [host-computed]. Both follow from the pp2048 rates of the prefill table,
28.1 and 11.80 t/s. Decode runs at 7.2 t/s on both backends, and there is no quant option to raise
it, since MXFP4 is the only distribution.

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
the NPU as native int8, it is a 2.38x prefill win at pp2048 (1.81x at pp512) on the 2026-08-27
build. It wins at every length on this model, and eligibility on another model is per architecture
(see "The accept boundary"). The speedup comes from residency: each expert is converted to int8
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

### The accept side of the same floor, and the budget above it

At `-ub` 4096, Qwen3-30B-A3B reaches `M_e` = 256 and 403 mega-MACs, and the default accepts its
experts. It is the first test point this model reaches above the 340 MMAC threshold. The default
measures 1.016x the `ROCKET_MOE=0` configuration, and `ROCKET_MOE=1` measures 0.964x [HW sweep
2026-08-28]. At only 1.19x the threshold, the test point tests its position rather than clearing it
comfortably.

**Conditions:** RK1 (RK3588), NPU clock 600 MHz pinned, governor `performance`, Qwen3-30B-A3B,
`-p 4096 -b 4096 -ub 4096`, 2026-08-28.

**Not characterized:** the threshold's position apart from a residency term, and any other
architecture or operating point.

The result covers one architecture at one operating point. The clock, the governor and `rate_npu`
are held fixed, on one model. Residency cannot reach 100% on this model on this board. Only a model
that both falls near 340 MMAC and fits resident can separate the position from residency.

At 403 MMAC this test point is 1.19x the threshold. The tightest accept-side test point this
architecture reaches is `-ub` 3584 -> `M_e` 224 -> 352 MMAC, 1.035x the threshold. A run there
would test the position rather than the sign.

`M_e` is set by the micro-batch, so `-ub` is the only option that moves the test point. The
prompt length `-p` needs only to be large enough to fill it. A 31 GiB board runs
`-b 4096 -ub 4096` with the 17.28 GiB model loaded.

| configuration | t/s, three processes | spread | paired ratios | mean | placement |
|---|---|---:|---|---:|---|
| `ROCKET_MOE=0` | 10.29 / 10.32 / 10.31 | 0.3% | n/a | n/a | n/a |
| **default** | 10.40 / 10.50 / 10.50 | 1.0% | 1.011 / 1.017 / 1.018 | **1.016x** | 79 stacks, 0 streamed |
| `ROCKET_MOE=1` | 9.80 / 9.93 / 10.09 | 2.9% | 0.952 / 0.962 / 0.979 | **0.964x** | 62.6% resident, 5703 streamed |

The two configurations place the experts differently. The default reserves whole stacks and streams
nothing: it holds 79 and leaves the rest on the CPU. Its placement is the same in all three
processes: 8818 experts, 14877 MB, 0 streamed. With `ROCKET_MOE=1`, every expert is claimed and
37% of them stream. Published figures for this model elsewhere are `ROCKET_MOE=1` figures, and do
not describe the default.

The three `ROCKET_MOE=1` test points are residency-matched (62.6% against 62.4%). They trace the
ratio against work per dispatch alone: 101 MMAC -> 0.661, 201 -> 0.834, 403 -> 0.964. The curve is
saturating rather than linear. A cost model predicts the third point out of sample to 0.4%. The
model is `ratio = A / (B + (P/ub)·k)`, with one term fixed per prefill and one proportional to the
dispatch count. Fitted on the two lower test points, it gives `k` = 0.314A and `B` = 0.885A, and
predicts 0.960 here against 0.964 measured.

> [!NOTE]
> Where the ingest is charged moves this test point by 12%. The `llama-bench` warm-up incurs most of
> the one-time expert ingest, and the timed repetitions incur only the experts it never routed to.
> With the warm-up off, the same test point measures 0.934x instead of 1.016. A figure quoted from
> this test point must state which charging it used.

#### The auto budget and `ROCKET_MOE_CACHE_MB`

The automatic budget over-places, and on this model the over-placement costs the whole offload.
The option `ROCKET_MOE_CACHE_MB` sets the pre-flight's budget directly. On the default
configuration it changes only the number of admitted stacks, with zero streaming at every budget.
The ladder below therefore has one regime and one varying quantity [HW sweep 2026-08-28].

**Conditions:** as above. Ratios are against the pooled `ROCKET_MOE=0` mean of six processes. The
`n` column is the number of processes, and the rep sd column the spread between repetitions.

The ingest and repaid-after columns assume an NPU-BO pack of ~505 MB/s [HW sweep 2026-08-28]. The
pack measured on gpt-oss runs ~1.3 GB/s [HW sweep 2026-10-09]. At that rate the pack term of each
ingest is ~0.35x as large, so both columns overstate the cost [host-computed].

| budget | stacks | resident | IOVA | ingest | t/s | n | ratio | rep sd | ingest repaid after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 0 | n/a | n/a | n/a | 10.32 | 6 | 1.000x | ±0.01 | n/a |
| 12000 MB | 38 | 7813 MB | 7296 | 20.2 s | 10.65 | 1 | 1.032x | ±0.00 | ~6 700 tokens |
| **18000 MB** | **58** | 11430 MB | 11136 | 30.9 s | **10.76** | 3 | **1.043x** | ±0.03-0.08 | ~7 400 |
| **21000 MB** | **67** | 13167 MB | 12864 | 36.7 s | **10.83** | 4 | **1.049x** | ±0.04-0.08 | ~7 700 |
| auto (24425) | 79 | 14877 MB | 15168 | 43.2 s | 10.47 | 3 | 1.014x | ±0.6 | ~34 000 |
| 28000 MB | 90 | 16716 MB | 17280 | 48.5 s | 10.31 | 1 | 0.999x | ±1.02 | never |

The ratio rises to a plateau and then falls, and the automatic budget is past the fall. The
18000 MB and 21000 MB budgets, 58 and 67 stacks, are one plateau, pooled at 1.046x over seven
processes. Between 67 and 79 stacks the ratio falls to 1.014.

The 12000 MB and 28000 MB rows are one process each. A single process on this board can sit ~10%
off the level its own configuration repeats at [HW sweep 2026-08-28]. The 28000 MB row's 0.999x
therefore shows one process at parity, not that the offload gives nothing at 90 stacks.

**Not characterized:** the depth of the fall past 79 stacks.

The ladder's repaid-after column uses the 2026-08 pack rate. At that rate the materiality table's
test points break even after ~16 300 tokens (1.05x), ~4 800 (1.22x) and ~3 800 (1.31x). Against
those counts, the automatic budget's admission repays after ~34 000 tokens, about twice the 1.05x
test point the threshold was fitted to exclude. The plateau repays after ~7 500, between the 1.05x
and 1.22x test points. The recommended budget is therefore a range, 18000-21000 MB, rather than a
value that has to be hit exactly.

#### Budget units and the charge factor

The `ROCKET_MOE_CACHE_MB` budget is in code-relative units, so a value does not transfer to another
quant or another board. The budget bounds the admission charge rather than RAM. The charge is the
int8 codes plus the per-(channel, K-group) scales plus that expert's GGUF source bytes. A budget
therefore gives strictly less residency than it names. Across the ladder:

| budget | resident int8 | budget / resident |
|---:|---:|---:|
| 18000 MB | 11430 MB | 1.58 |
| 21000 MB | 13167 MB | 1.60 |
| 24425 MB | 14877 MB | 1.64 |
| 28000 MB | 16716 MB | 1.68 |

The charge's own form predicts the ratio. Per expert, the charge is `N·K` code bytes +
`N·(K/group)·4` scale bytes + the source stride `nb[2]`. Against the codes alone:

```
charge / codes  =  1 + 4/group + source_bits_per_weight/8
```

This GGUF is 18556686912 bytes for 30.5 B parameters, which is 4.87 bits/weight. At `K`=2048
the group auto-picks to 512, giving 1.617. The scales term is the small one (0.8% here), and the
source term sets the ratio.

The pre-flight's log gives the factor directly, and that is the number to compare against. The
budget-over-resident column above only estimates it, since a budget is what was requested. The
pre-flight prints both halves of the charge:

```
[moe-int8] resident budget reached after 79 expert stacks (24425MB RAM, 15168MB IOVA)
```

RAM is the charge and IOVA is the int8 codes, so their ratio is `1 + 4/group + bits/8` with nothing
inferred. Measured that way, on two models and two quant formats:

| model | quant | charged / codes | derived | agreement |
|---|---|---:|---:|---:|
| Qwen3-30B-A3B | Q4_K_M, group 512 | 24425 / 15168 = **1.610** | 1.617 | 0.4% |
| gpt-oss-20b | MXFP4, group 576 | 21433 / 14092 = **1.521** | 1.538 | 1.1% |

The first row is [HW sweep 2026-08-28, RK1]. The second row comes from four archived runs
([data/gpt-oss-20b.md](data/gpt-oss-20b.md)), whose six such lines measure 1.5209-1.5211. The
derived value is ~1% high in both cases, so the form is a good sizing rule and not an exact one.
Two caveats on its input explain the direction. The whole file's average, 4.87 bits/weight, stands
in for the expert tensors' own rate, and a Q4_K_M GGUF mixes types.

The same physical placement therefore needs a different budget for a different quant of the same
model. The factor is about 2.07 for Q8_0 (8.5 bits/weight) and 1.54 for IQ4_XS (4.25)
[expected: derived from bits/weight, not measured].

To size a budget before a run, derive the factor. For an exact value, read the factor off the
pre-flight line. The derivation transfers across quants and boards, and the log fixes the factor
for one model. Take the ratio from a line whose stack count is limited by RAM, which says "budget
reached ... Bound by RAM". A run at a pinned `ROCKET_MOE_CACHE_MB` stops at the pin rather than at
the natural budget.

#### RAM and IOVA budgets

The factor also decides which budget binds, and on this path it is never the IOVA window. The
pre-flight holds two budgets: RAM, and `ROCKET_N_THREADS x 3840 MB` of NPU IOVA. It charges each
stack `codes + scales + source` against the first. It charges `codes` alone against the second.
The window therefore binds first only where

    RAM budget / IOVA budget  >  1 + 4/group + source_bits_per_weight/8

which is the same factor. On `gpt-oss-20b` MXFP4 the right side is 1.538, and the left is 1.28 at
the default five workers. RAM binds, by a margin no worker count can close, because raising
`ROCKET_N_THREADS` grows only the denominator. At 5 and at 8 workers the admitted set is the same,
and the pre-flight reports `Bound by RAM` [HW sweep 2026-09-02].

The window cannot bind on this path at or above the default worker count. The threshold is set by
the source rather than by the board. At five workers it would need `MemAvailable` above 35.7 GB,
and each further worker adds about 5.9 GB to that. On a 32 GiB board the inequality needs a source
under about 2.2 bits per weight, which no quantization this path ingests approaches. MXFP4, the
densest measured, is 4.25. The option that `Bound by RAM` names, `ROCKET_MOE_CACHE_MB`, is the only
one this path's line ever asks for.

> [!NOTE]
> A model can print no pre-flight line at all, and that does not mean the pre-flight passed. Auto
> placement checks the tile granule and the per-dispatch work threshold before it calls the
> pre-flight [source-confirmed, `ggml-rocket.cpp`]. If the line is missing, raise `-ub` before
> concluding anything about RAM or the window.

`Qwen3-30B-A3B` Q4_K_M at `-b 2048 -ub 2048` has 128 experts with 8 used. A 2048-token micro-batch
therefore gives `M_e` = 128, and the work threshold declines it first. Two runs at that shape, at 5
workers and at 8, emit no `[moe-int8]` line at all [HW sweep 2026-09-02, RK1]. The same model
reaches the pre-flight at `-ub 4096`, where `M_e` is 256.

#### Over-placement mechanism

The automatic budget over-places because of an omission in the charge [hypothesis]. The pre-flight
charges each placed expert as int8 codes plus that expert's GGUF source bytes. It does not charge
two things. The first is the rest of the expert GGUF, the unplaced experts, which the CPU reads
from the same mmap every micro-batch. The second is the placed experts' own source pages, which the
ingest faulted moments earlier. Charging the whole expert GGUF against the anonymous int8 set gives:

| stacks | anonymous int8 | + whole expert GGUF (16.3-17.8 GB) | vs 31744 MB total |
|---:|---:|---:|---|
| 38 | 7813 | 24.1-25.6 GB | under by 6-7.6 |
| 58 | 11430 | 27.7-29.2 GB | under by 2.5-4 |
| 67 | 13167 | 29.5-31.0 GB | under by 0.8-2.3 |
| **79** | 14877 | **31.2-32.7 GB** | **crosses 31.7** |
| 90 | 16716 | 33.0-34.5 GB | over by 1.3-2.8 |

The crossing is at 79 stacks, which is where the ratio falls. The expert GGUF's own size is bounded
rather than known, because the stacks are not equal-sized. The pre-flight's per-stack charge puts
it at 16.3 GB at low placement and 17.8 GB at high placement. The fit is therefore one of ordering
and magnitude, not a coincidence to 0.1%. The 6 GiB reserve exists to absorb exactly this, and the
uncharged remainder here is 7-12 GB.

The fit needs the placed experts' source pages to stay resident after ingest. Without them the hot
set is flat across the five budgets and crosses nothing.

**Not characterized:** which pages the kernel retains, and for how long.

The factor also reconciles the RAM arithmetic with the recommended budget. On this board,
`MemTotal` minus the whole expert GGUF minus the reserve is 31.7 − 17.3 = 14.4 GB, far from
21000 MB. Converted through the measured factor, with a runtime-headroom term, it gives
`1.610 × (31.7 − 17.3 − 1.2)` = 21.3 GB. The form is consistent and is not an independent
prediction: the headroom term is the one free parameter, chosen to match the measured plateau.

The mechanism predicts its own scope, and the prediction holds where it can be checked. The defect
needs the experts to dominate the model and the GGUF to be large against RAM. The gpt-oss-20b model
is neither (11.27 GiB). There, `ROCKET_MOE_CACHE_MB=28000` measured +8.3% / +14.1% with no
degradation. The gpt-oss ladder is one `-r 3` process per configuration, so the two deltas have the
same n=1 caveat as this ladder's ends. They show a direction rather than a magnitude.

The two ladders jointly establish the sign flip between model classes, a comparison of directions
that the single-process caveat does not affect. A `ROCKET_MOE_CACHE_MB` recommendation is therefore
scoped to the model class, not to the board size.

### The induced-scarcity ladder: the budget mechanism does not produce an inverted U

On gpt-oss-20b, a ballast of `B` GiB of touched anonymous memory is held beside the run. Prefill
throughput falls as `B` rises, with no turn at any step [HW sweep 2026-08-28]. Every step
reports 100% resident, 0 streamed, and no OOM line in `dmesg`. Fewer stacks are placed, and the
rest stay on the CPU, which the code calls "a partial offload and not a loss".

**Conditions:** RK1 (RK3588), NPU clock 600 MHz pinned, governor `performance`, gpt-oss-20b MXFP4,
`-p 2048 -n 0 -r 1 -b 2048 -ub 2048`, one process per step, 2026-08-28. The page cache is dropped
and `compact_memory` run before every step.

| `B` | MemAvailable | pre-flight budget | stacks placed | resident | streamed | pp2048 t/s | vs `B`=0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30.2 GB | 24598 MB | 63 | 13454 MB | **0** | 28.93 | 1.000 |
| 4 | 26.2 | 20488 | 52 | 11317 | **0** | 24.04 | 0.831 |
| 8 | 22.2 | 16401 | 42 | 9486 | **0** | 20.89 | 0.722 |
| 12 | 18.2 | 12290 | 31 | 7348 | **0** | 18.16 | 0.628 |
| 16 | 14.2 | 8225 | 21 | 5372 | **0** | 15.36 | 0.531 |

Throughput falls monotonically and very nearly linearly in the number of placed stacks
(`t/s = 8.32 + 0.315 x stacks`, residuals within ±0.8 t/s over a 15-29 range).

The ladder's variable and the pre-flight's input are the same quantity. The pre-flight budget is
`MemAvailable - 6 GiB`, and the budget column falls by exactly 4096 MB per 4 GiB step. Ballast
therefore makes the pre-flight more conservative, never less. The memory the ballast removes is
removed from the budget first.

> [!NOTE]
> An induced-scarcity ladder of this shape cannot test whether the pre-flight over-places. The
> pre-flight subtracts the ballast from its budget first, so the expert path never holds more than
> the board can honor. Only memory that disappears after the budget is frozen reaches that state.

The hot-set account of int8 codes, the whole GGUF and the ballast over-counts a file-backed GGUF.
At `B` = 16 the accounted hot set is 5372 MB of int8 codes + 11.27 GiB of GGUF + 16 GiB of
ballast. The total is 32.5 GB against 31.7 GB of RAM, and yet nothing streamed, nothing was killed,
and the linear fit did not bend. At least some of what that account calls hot is therefore
reclaimed under pressure. The GGUF, file-backed unlike the ballast, is the only candidate
[hypothesis: not separated from a smaller-than-assumed activation footprint].

The charge model treats each placed expert's source bytes as unreclaimable because decode reads
them from the mmap every token. During a `-n 0` prefill they are not touched after ingest.

The ladder does not refute the charge omission itself. The uncharged remainder is still real and
still largest at low placement, and the ladder never reaches a state where it takes effect. Memory
that disappears after the budget is frozen reaches that state, and the pre-flight cannot see it by
construction.

> [!CAUTION]
> The pre-flight resolves its budget once and keeps it. Memory that another process takes after
> that is still counted as available, so the expert path reserves stacks against RAM that is gone.
> In the run below the kernel OOM killer fired, and the run still exited with status 0 and a
> plausible throughput. Nothing in its output says a process was killed.

In the second row below, the 16 GiB of ballast is allocated after the pre-flight line prints, with
the same model and flags. The run reaches the over-placed state [HW sweep 2026-08-28, RK1]:

| | budget frozen at | MemAvailable when the ingest ran | stacks reserved | outcome |
|---|---:|---:|---|---|
| ballast **before** (`B`=16 step above) | 8225 MB | 14.2 GB | 21 | 100% resident, 0 streamed, no OOM |
| ballast **after** the freeze | **24608 MB** | **11.1 GB** | **63** | kernel **OOM-killer fired**, 91% resident, 144 streamed, pp2048 25.53 |

The pre-flight resolved the budget at 24608 MB against an idle board, and kept it while 18 GB of
the board disappeared. It therefore went on reserving 63 stacks and 24529 MB against 11.1 GB of
available RAM. The board recovered only because the OOM killer picked the ballast, 16.8 GB of
anonymous RSS, the largest badness score on the board. A competing allocation smaller than the
inference process would have made `llama-bench` the victim instead.

The two rows are a controlled pair. At essentially the same free RAM (14.2 against 11.1 GB), the
pre-flight reserved 8225 MB in one and 24608 MB in the other. The only difference is when the
memory went away: the pre-flight has no run-time `MemAvailable` check.

### Qwen3.5-9B

Qwen3.5-9B is Alibaba's Qwen3.5 9B (arch `qwen35`, dense, GQA), with 8.95 B params. The GGUFs
come from `unsloth/Qwen3.5-9B-GGUF` (base model `Qwen/Qwen3.5-9B`), and the F16 is derived from the
published BF16. Three GGUFs are measured: F16, Q4_K_M and IQ4_XS, an importance-matrix 4-bit quant.

**Conditions:** RK3588, NPU clock 600 MHz, the flags and 8-thread CPU baseline of §"Method",
measured 2026-07-02 [HW sweep].

**Prefill (prompt processing), t/s (the NPU's job).**

| test | F16 CPU->NPU | Q4_K_M CPU->NPU | IQ4_XS CPU->NPU |
|---|---|---|---|
| pp512  | 7.25 -> **25.86** (3.6x) | 6.25 -> 16.68 (2.7x) | 7.44 -> 16.33 (2.2x) |
| pp1024 | 7.14 -> **25.14** (3.5x) | 6.21 -> 21.31 (3.4x) | 7.37 -> 20.87 (2.8x) |
| pp2048 | 7.08 -> **24.85** (3.5x) | 6.15 -> 22.88 (3.7x) | 7.29 -> 22.49 (3.1x) |

NPU prefill runs at ~3.5x the CPU rate on F16, flat at ~25 t/s from pp512 to pp2048, where
Ministral-3-8B's F16 rate declines. The quants' NPU rates rise with M (Q4_K_M 16.7->22.9, IQ4_XS
16.3->22.5). A quantized GGUF is dequantized to fp16 again on every micro-batch. At pp512 (one
512-row batch) the dequantization is a large fixed fraction of prefill time. By pp2048 it amortizes
and the quants close on F16.

The same mechanism sets the tuning for a quantized GGUF. Where its fp16 image fits,
`ROCKET_QUANT_RESIDENT=auto` at the default `-ub` removes the per-micro-batch dequantization. Where
it does not, `-b 2048 -ub 2048` amortizes it (the tuning table of §"At a glance",
[quant-prefill-microbatch.md](quant-prefill-microbatch.md)).

IQ4_XS prefill offloads on the same path as Q4_K_M, at 3.1x the CPU at pp2048. An importance matrix
is used only at quantization time. At inference IQ4_XS is an ordinary `ggml_is_quantized` type with
a `to_float` trait and 256-element super-blocks, so `K%32` holds. Its prefill GEMMs therefore
dequantize to fp16 and offload exactly as Q4_K_M's do, and the importance-matrix quants use the
full quantized-prefill path.

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).**

| | F16 | Q4_K_M | IQ4_XS |
|---|---|---|---|
| tg64 CPU | 1.35 | 3.60 | 4.08 |
| tg64 NPU | 1.34 | 3.59 | 4.09 |

Decode with the NPU backend loaded matches the CPU, and runs on the CPU
([decode-gemv.md](decode-gemv.md)). Its rate scales with the bytes each token reads, so the smallest
quant decodes fastest: F16 1.4 -> Q4_K_M 3.6 (5.28 GB) -> IQ4_XS 4.1 t/s (4.80 GB).

**Interactive.** TTFT and stream rate are derived from the measured pp and tg rates
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

The per-chunk stderr is ±0.43, so the NPU−CPU delta is ~100x smaller than the noise. The NPU prefill
matches the CPU to within 0.10% on all three, so the fp16-accumulation path is faithful for the
importance-matrix quant as well. The PPL rise from F16 to Q4_K_M to IQ4_XS (9.06 -> 9.24 -> 9.35)
is the expected quantization cost, present equally on both backends.

**Verdict.** For batch or RAG workloads on a board with 16 GB or more, F16 + NPU gives this model's
fastest prefill. It holds a flat ~25 t/s across the prompt curve. For most boards, use IQ4_XS + NPU.
It fits an 8 GB board (4.80 GB), has the fastest stream (4.1 t/s), keeps ~3x NPU prefill on long
prompts, and is faithful (Δ −0.10%). Q4_K_M has marginally higher fidelity (PPL 9.23 against 9.35),
at 5.28 GB and a 3.6 t/s stream.

### Phi-4-mini-instruct

Microsoft's Phi-4-mini-instruct (arch `phi3`, dense, GQA) has 3.84 B params. The GGUFs come from
`unsloth/Phi-4-mini-instruct-GGUF` (base model `microsoft/Phi-4-mini-instruct`), and the F16 is
derived from the published BF16.

**Conditions:** RK3588, NPU clock 600 MHz, the flags and 8-thread CPU baseline of §"Method",
measured 2026-07-02 [HW sweep].

**Prefill (prompt processing), t/s (the NPU's job).**

| test | F16 CPU->NPU | Q8_0 CPU->NPU | Q4_K_M CPU->NPU |
|---|---|---|---|
| pp512  | 16.55 -> **55.44** (3.4x) | 15.20 -> 37.26 (2.5x) | 13.85 -> 36.70 (2.6x) |
| pp1024 | 15.96 -> **47.86** (3.0x) | 14.81 -> 38.77 (2.6x) | 13.55 -> 37.29 (2.8x) |
| pp2048 | 15.07 -> **39.79** (2.6x) | 13.90 -> 34.91 (2.5x) | 12.89 -> 33.42 (2.6x) |

F16 NPU prefill reaches 55 t/s at pp512, and its speedup falls with prompt length (3.4x->2.6x) as
the NPU rate declines 55->40. On a model this small the per-op readback and dispatch are a larger
fixed fraction, and they grow with M (more output tiles to read back). The F16 rate therefore falls
with M, where the Qwen3.5-9B F16 rate is flat. The quants are flatter (~33-39 t/s). At pp2048 they
close much of the gap to F16 (33-35 against 40), as their per-micro-batch dequantization amortizes
while F16's readback grows.

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).**

| | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| tg64 CPU | 2.86 | 4.86 | 7.04 |
| tg64 NPU | 2.84 | 4.82 | 6.95 |

Decode with the NPU backend loaded matches the CPU, and runs on the CPU
([decode-gemv.md](decode-gemv.md)). At 3.84 B it decodes at F16 2.9 -> Q8_0 4.9 -> Q4_K_M 7.0 t/s,
an interactive stream from a 2.31 GB file.

**Interactive.** TTFT and stream rate are derived from the measured pp and tg rates
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

**Verdict.** Phi-4-mini suits a small or edge board. With 7 GB of RAM available, F16 + NPU gives
this model's fastest prefill, 55 t/s at pp512. Q4_K_M + NPU is the practical choice: 2.31 GB (fits
a 4 GB board), ~2.6x NPU prefill, and an interactive 7 t/s stream. As this small model's per-op
readback grows with M, the NPU's prefill speedup narrows with prompt length (F16 3.4x->2.6x). The
largest speedups are therefore on short-to-medium prompts.

### Gemma-4-12B-it

Google's Gemma-4-12B-it (arch `gemma4`, dense, GQA) has 11.91 B params. The GGUFs come from
`unsloth/gemma-4-12b-it-GGUF` (base model `google/gemma-4-12b-it`), and the F16 is derived from the
published BF16.

**Conditions:** RK3588, NPU clock 600 MHz, the flags and 8-thread CPU baseline of §"Method",
measured 2026-07-02 [HW sweep].

**Prefill (prompt processing), t/s (the NPU's job).**

| test | F16 CPU->NPU | Q8_0 CPU->NPU | Q4_K_M CPU->NPU |
|---|---|---|---|
| pp512  | 4.82 -> **17.42** (3.6x) | 4.52 -> 11.41 (2.5x) | 4.23 -> 11.20 (2.6x) |
| pp1024 | 4.78 -> **16.09** (3.4x) | 4.35 -> 13.53 (3.1x) | 4.15 -> 13.19 (3.2x) |
| pp2048 | 4.63 -> **14.98** (3.2x) | 4.16 -> 13.43 (3.2x) | 4.02 -> 13.28 (3.3x) |

NPU prefill runs at up to 3.6x the CPU rate on F16, against a ~4.6 t/s CPU baseline for this 12 B
model. F16 has the highest NPU rate, 17.4 t/s at pp512. Its speedup narrows with prompt length
(3.6x->3.2x) as the NPU rate falls 17.4->15.0. The NPU rate falls because each matmul has more
output tiles to read back as M grows.

The quants move the opposite way, rising with M (Q4_K_M 11.2->13.3, 2.6x->3.3x). A quantized GGUF
is dequantized to fp16 again on every micro-batch. At pp512 that dequantization is a large fixed
fraction, and by pp2048 it amortizes (the mechanism behind `-b 2048 -ub 2048`,
[quant-prefill-microbatch.md](quant-prefill-microbatch.md)).

By pp2048 all three measure ~13-15 t/s. A quantized GGUF does not prefill faster than F16 at this
operating point, and only reduces memory use ([not-mac-bound.md](not-mac-bound.md)).

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).**

| | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| tg64 CPU | 0.94 | 1.75 | 2.44 |
| tg64 NPU | 0.94 | 1.59 | 2.50 |

Decode with the NPU backend loaded matches the CPU, and runs on the CPU
([decode-gemv.md](decode-gemv.md)). At 12 B, F16 decode runs at 0.94 t/s, because every token
streams 22 GiB of weights from LPDDR. Decode scales with the bytes each token reads, so quantization
is the only option that raises it. Quantization raises it 2.7x: F16 0.94 -> Q8_0 1.7 -> Q4_K_M 2.5
t/s. For interactive use of this model, use a quantized GGUF.

**Interactive.** TTFT and stream rate are derived from the measured pp and tg rates
[host-computed].

| scenario | config | TTFT | stream | total turn |
|---|---|---|---|---|
| RAG / summarize, 2048-tok prompt -> 200 out | F16 CPU | 442 s | 0.9 t/s | 655 s |
| | F16 + NPU | **137 s** | 0.9 t/s | 350 s |
| | Q8_0 + NPU | 152 s | 1.6 t/s | 278 s |
| | Q4_K_M + NPU | 154 s | 2.5 t/s | **234 s** |
| chat, 128-tok prompt -> 400 out | F16 + NPU | 7 s | 0.9 t/s | 433 s |
| | Q4_K_M + NPU | 11 s | 2.5 t/s | **171 s** |

On a 2048-token prompt the NPU shortens the first-token wait ~3.2x (F16 442->137 s). The stream
dominates the turn on this model: even with NPU prefill, F16 needs ~350 s for a 200-token reply.
Quantization shortens the turn more than the backend does (Q4_K_M 234 s, and the chat turn
433->171 s). Q4_K_M + NPU is the best interactive configuration, with the NPU's short TTFT and
Q4_K_M's fastest stream.

**Faithfulness: differential perplexity, wikitext test, 12 chunks (same GGUF, NPU prefill vs CPU).**

| | CPU PPL | NPU PPL | Δ |
|---|---|---|---|
| F16    | 630.9975 | 625.8672 | −0.81% |
| Q8_0   | 639.9257 | 652.1588 | +1.91% |
| Q4_K_M | 670.6969 | 675.4451 | +0.71% |

The per-run stderr is ±~65 (~±10%). All three NPU−CPU deltas (−0.8%, +1.9%, +0.7%) are well inside
it, so the fp16-accumulation prefill path is faithful across F16 and both quants. The absolute PPL
(~630+) is not meaningful. Gemma-4-12B-it is instruction- and reasoning-tuned, and scores raw
wikitext very poorly. As with the other instruct models, only the NPU−CPU delta is informative.

**Verdict.** For batch or RAG work on a 32 GB board (the F16 GGUF is 22 GiB), F16 + NPU gives this
model's fastest prefill (17.4 t/s at pp512, 3.6x the CPU). Its 0.94 t/s decode makes long replies
slow on either backend. Q4_K_M + NPU is the practical choice: 6.86 GB (fits an 8 GB board), ~3.3x
NPU prefill on long prompts, and a 2.7x-faster stream. It gives the shortest effective turn in both
the RAG and chat scenarios. Q8_0 (11.78 GB) is the near-lossless middle option.

### Qwen3.6-27B (hybrid Gated-DeltaNet, Q4_K_M)

Alibaba's Qwen3.6 27B (arch `qwen35`, 65 blocks, 27.32 B params) is a hybrid model. It interleaves
Gated-DeltaNet linear-attention / SSM-scan layers with dense GQA attention (GGUF SSM metadata: state
128, conv kernel 4, 16 groups, inner size 6144). The GGUF comes from `unsloth/Qwen3.6-27B-MTP-GGUF`.
Only Q4_K_M (15.92 GiB) is measured. At 27 B the F16 GGUF (~54 GB) and Q8_0 (~29 GB) do not fit the
31 GB board.

**Conditions:** RK3588, NPU clock 600 MHz, the flags and 8-thread CPU baseline of §"Method",
measured 2026-07-03 [HW sweep].

**Prefill (prompt processing), t/s (the NPU's job).**

| test | Q4_K_M CPU->NPU |
|---|---|
| pp512  | 1.79 -> 5.38 (3.0x) |
| pp1024 | 1.79 -> 6.78 (3.8x) |
| pp2048 | 1.78 -> **7.78 (4.4x)** |

NPU prefill runs at 3.0x the CPU rate at pp512, rising to 4.4x at pp2048. Two effects combine. The
NPU rate climbs from 5.4 to 7.8 t/s as the per-micro-batch dequantization amortizes (the
`-b 2048 -ub 2048` mechanism, [quant-prefill-microbatch.md](quant-prefill-microbatch.md)). The CPU
baseline is ~1.8 t/s, a 27 B model streamed through 8 CPU cores. The NPU's speedup over it is larger
than on any smaller model. The speedup grows with model size because the CPU rate falls
faster than the NPU rate as the matmuls grow.

The Gated-DeltaNet and SSM-scan layers have no NPU handler, and run on the CPU. They are a small
share of the prefill FLOPs. The FFN and attention projections dominate and offload as ordinary
`MUL_MAT`, so the hybrid layers do not reduce the prefill speedup. A mixture-of-experts model
differs: its expert FFNs go through `MUL_MAT_ID`, whose NPU path holds each quantized expert
resident as native int8. The path is on by default, and places an expert op only where a residency
pre-flight and a per-dispatch work threshold both admit it (§"gpt-oss-20b (MXFP4, MoE)").

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).**

| | Q4_K_M |
|---|---|
| tg64 CPU | 1.10 |
| tg64 NPU | 1.11 |

Decode with the NPU backend loaded matches the CPU, and runs on the CPU
([decode-gemv.md](decode-gemv.md)). At 27 B it runs at 1.1 t/s, because every token streams the
16 GiB of Q4_K_M weights from LPDDR. There is no smaller quant here to raise the stream rate, so
this model suits prefill and batch work rather than chat.

**Interactive.** TTFT and stream rate are derived from the measured pp and tg rates
[host-computed].

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

**Verdict.** On Qwen3.6-27B the NPU's prefill speedup is 4.4x at pp2048. It turns a ~19-minute CPU
first-token wait on a 2048-token prompt into ~4.5 minutes. On a 31 GB board the model suits prefill
and RAG work rather than interactive chat. Only Q4_K_M fits, decode is ~1.1 t/s (a 400-token reply
takes ~6 min), and there is no smaller quant to raise the stream rate.

Use it for long-prompt, short-answer work (summarize, extract, RAG), where the 4.4x first-token
speedup applies. Where replies must stream at conversational speed, use a smaller model (Gemma-4-12B
or Qwen3.5-9B at Q4_K_M). Linear-attention and SSM hybrids offload their prefill. The DeltaNet layers
stay on the CPU and do not limit the speedup.

### Llama-3.2-3B-Instruct

Meta's Llama-3.2-3B-Instruct (arch `llama`, dense, GQA) has 3.21 B params, and is the field's
standard reference point. The GGUFs come from `unsloth/Llama-3.2-3B-Instruct-GGUF`, and the F16 is
unsloth's published F16 (derived from the BF16).

**Conditions:** RK3588, NPU clock 600 MHz, the flags and 8-thread CPU baseline of §"Method",
measured 2026-07-03 [HW sweep].

**Prefill (prompt processing), t/s (the NPU's job).**

| test | F16 CPU->NPU | Q8_0 CPU->NPU | Q4_K_M CPU->NPU |
|---|---|---|---|
| pp512  | 18.23 -> **61.79** (3.4x) | 17.05 -> 34.19 (2.0x) | 16.71 -> 33.60 (2.0x) |
| pp1024 | 17.74 -> **52.91** (3.0x) | 16.48 -> 36.31 (2.2x) | 16.34 -> 36.66 (2.2x) |
| pp2048 | 17.29 -> **45.54** (2.6x) | 15.72 -> 34.11 (2.2x) | 15.49 -> 34.46 (2.2x) |

F16 NPU prefill reaches 61.79 t/s at pp512, above Phi-4-mini's 55, because this smaller model has
fewer FLOPs per matmul. As with Phi-4-mini, the F16 speedup falls with prompt length (3.4x->2.6x, the
NPU rate declining 62->46). On a small model the per-op readback and dispatch are a larger fixed
fraction, and they grow with M.

The quants measure a lower speedup (~2.0-2.2x) than Phi-4-mini's ~2.5-2.6x because of the CPU
denominator, rather than a weaker NPU. Llama-3.2-3B's CPU quant prefill is ~16-17 t/s, close to its
own F16 CPU rate of 17-18. Its NPU quant rate (~34-36 t/s) is in line with Phi-4-mini's. The quants'
rates rise from pp512 to pp1024 and are flat at pp2048, as the per-micro-batch dequantization
amortizes (the `-b 2048 -ub 2048` mechanism,
[quant-prefill-microbatch.md](quant-prefill-microbatch.md)).

By pp2048 the two quants (~34 t/s) trail F16 (46 t/s). At this operating point a quantized GGUF does
not prefill faster than F16, and only reduces memory use ([not-mac-bound.md](not-mac-bound.md)).

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).**

| | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| tg64 CPU | 3.15 | 5.62 | 7.88 |
| tg64 NPU | 3.27 | 5.47 | 7.93 |

Decode with the NPU backend loaded matches the CPU, and runs on the CPU
([decode-gemv.md](decode-gemv.md)). At 3.21 B it decodes at F16 3.3 -> Q8_0 5.5 -> Q4_K_M 7.9 t/s,
an interactive stream from a 1.87 GB file.

**Interactive.** TTFT and stream rate are derived from the measured pp and tg rates
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

Both deltas are well inside the reported ±0.65 PPL uncertainty, and the F16 NPU−CPU delta is ~300x
smaller than the per-chunk stderr. The F16 path matches the CPU to +0.02%. The Q4_K_M delta (−0.44%,
in the NPU's favor) reflects the NPU consuming the Q4_K weights through dequantization to fp16,
rather than through the CPU's native Q4_K kernel. No accuracy is lost. The absolute PPL is high,
~12.7, because Llama-3.2-3B-Instruct is instruction-tuned and models raw wikitext poorly. As with the
other instruct models, only the NPU−CPU delta is informative, not the absolute value.

**Verdict.** Llama-3.2-3B's F16 NPU prefill is 61.8 t/s at pp512, 3.4x the CPU. With ~6 GB of RAM
free, F16 + NPU gives this model's highest prefill rate. Its speedup narrows with prompt length
(3.4x->2.6x) as this small model's per-op readback grows with M, so it is largest on
short-to-medium prompts. Q4_K_M + NPU is the practical choice: 1.87 GB (fits a 4 GB board with
headroom), ~2.2x NPU prefill, and a 7.9 t/s stream. It gives the shorter effective turn in both
scenarios.

On small dense models the NPU/CPU prefill ratio is lower because their CPU baseline is
comparatively fast, not because the NPU is weaker. The NPU's absolute prefill rate remains the
headline figure for these models.

### Ministral-3-3B-Instruct-2512

Mistral's Ministral-3-3B-Instruct-2512 (arch `mistral3`, dense, GQA) has 3.43 B params. It is the 3B
sibling of Ministral-3-8B (§"Ministral-3-8B-Instruct-2512"). The GGUFs come from
`unsloth/Ministral-3-3B-Instruct-2512-GGUF`, and the F16 is derived from the published BF16.

**Conditions:** RK3588, NPU clock 600 MHz, the flags and 8-thread CPU baseline of §"Method",
measured 2026-07-03 [HW sweep].

**Prefill (prompt processing), t/s (the NPU's job).**

| test | F16 CPU->NPU | Q8_0 CPU->NPU | Q4_K_M CPU->NPU |
|---|---|---|---|
| pp512  | 18.02 -> **57.62** (3.2x) | 16.16 -> 32.94 (2.0x) | 15.55 -> 32.96 (2.1x) |
| pp1024 | 17.39 -> **48.57** (2.8x) | 15.54 -> 35.14 (2.3x) | 15.10 -> 34.67 (2.3x) |
| pp2048 | 16.26 -> **39.78** (2.4x) | 14.50 -> 30.78 (2.1x) | 14.32 -> 30.54 (2.1x) |

F16 NPU prefill reaches 57.6 t/s at pp512, between Llama-3.2-3B (61.8) and Phi-4-mini (55.4). The
three ~3-4 B dense models cluster near the top of the absolute-prefill table, because they run fewer
FLOPs per matmul than the larger models. As on the other two, the F16 speedup falls with prompt
length (3.2x->2.4x, the NPU rate declining 58->40). A small model's per-op readback is a larger fixed
fraction and grows with M.

The quants measure ~2.0-2.3x, at an NPU rate of ~31-35 t/s. Their rates rise and then flatten with
M as the per-micro-batch dequantization amortizes (the `-b 2048 -ub 2048` mechanism,
[quant-prefill-microbatch.md](quant-prefill-microbatch.md)). The lower ratio comes from the strong
CPU quant baseline (~14-16 t/s) rather than a weaker NPU, as on Llama-3.2-3B. By pp2048 the quants
(~31 t/s) trail F16 (40). A quantized GGUF does not prefill faster than F16 here, and only reduces
memory use ([not-mac-bound.md](not-mac-bound.md)).

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).**

| | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| tg64 CPU | 3.01 | 5.39 | 7.59 |
| tg64 NPU | 3.00 | 5.53 | 7.66 |

Decode with the NPU backend loaded matches the CPU, and runs on the CPU
([decode-gemv.md](decode-gemv.md)). At 3.43 B it decodes at F16 3.0 -> Q8_0 5.5 -> Q4_K_M 7.7 t/s
from a 1.99 GB file, against Llama-3.2-3B's 7.9 t/s from 1.87 GB.

**Interactive.** TTFT and stream rate are derived from the measured pp and tg rates
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

Both deltas are well inside the ±0.46 per-chunk stderr, and the F16 NPU−CPU delta is ~130x smaller.
The NPU prefill matches the CPU to −0.04% on F16 and −0.10% on Q4_K_M, both in the NPU's favor, so
the fp16-accumulation path is faithful. The absolute PPL (~9.9) is higher than the 8B sibling's
~6.9, which is the small model's quality cost rather than an NPU effect. Because Ministral-3-3B is an
instruct model, only the NPU−CPU delta is informative.

**Verdict.** Ministral-3-3B suits a small or edge board, with an F16 NPU prefill of 57.6 t/s at
pp512 (3.2x). With ~7 GB of RAM free, F16 + NPU gives this model's highest prefill rate, with the
speedup largest on short-to-medium prompts (3.2x->2.4x as M grows). Q4_K_M + NPU is the practical
choice: 1.99 GB (fits a 4 GB board), ~2.1x NPU prefill, and a 7.7 t/s stream. It gives the shorter
effective turn in both scenarios.

Among the small dense models, Ministral-3-3B sits between Llama-3.2-3B (61.8 against 57.6 t/s
prefill, 7.9 against 7.7 t/s stream) and Phi-4-mini. The three show the same pattern. On small dense
models the NPU/CPU ratio is modest because the CPU baseline is fast, while the NPU's absolute F16
prefill rate is 55-62 t/s.

### DeepSeek-V2-Lite (MLA + MoE, Q4_K_M)

DeepSeek's V2-Lite (arch `deepseek2`, 27 layers, 15.71 B params, ~2.4 B active per token) is a
mixture-of-experts model: 64 routed experts and 2 shared experts, with 6 routed experts active per
token and block 0 dense. Its attention is Multi-head Latent Attention (MLA). The KV is compressed to
a `kv_lora_rank=512` latent, and the per-head key/value dims are asymmetric: key_length 192 (128
nope + 64 rope), value_length 128. The GGUF comes from `mradermacher/DeepSeek-V2-Lite-GGUF` (the
base model `deepseek-ai/DeepSeek-V2-Lite`, 9.65 GiB). Only Q4_K_M is measured, because at 15.71 B
the F16 GGUF (~31 GB) does not fit the 31 GB board.

**Conditions:** RK3588, NPU clock 600 MHz, Q4_K_M, the flags and 8-thread CPU baseline of
§"Method", measured 2026-07-03, with the MoE offload re-measured 2026-07-04 [HW sweep]. A table
measured under other conditions states them.

MLA attention and the routed experts each have an op-level handler. The FLASH_ATTN handler accepts
MLA's asymmetric key and value dims (DK=192 ≠ DV=128) rather than requiring `DK==DV`. The MLA FA
primitive is therefore ready, and bit-faithful (cos=1.000000). The DeepSeek FA path in the
dynamically loaded backend is not wired on the device. The path is dispatch-bound (large DK, few KV
heads) and gains only at long context, so attention stays on the CPU here, with no effect on
prefill.

The routed experts go through `GGML_OP_MUL_MAT_ID`, which has a handler (`ROCKET_MOE=1`). Where the
experts run depends on the expert path and the prefill length (below).

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

It is far below the dense models' 3x+. Attention (MLA, on the CPU here) and the routed experts are
the bulk of the graph, and they stay on the CPU. CPU prefill is ~19-20 t/s for a 16 B model,
because only ~2.4 B params are active per token.

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

**Conditions:** RK3588, NPU clock 600 MHz, governor `performance`, llama.cpp b10558, 2026-08-27.

**Not characterized:** what moves the experts-on-CPU configuration between the 2026-07-03 table and
this one: 24.04 to 26.04 t/s at pp512, and 23.87 to 21.63 at pp2048.

At pp512 the expert offload is a 27% regression at 100% residency, with 0 experts streamed. Each
expert receives `n_tokens · n_used / n_expert` rows, and the row count is an architecture property
rather than a prompt one. DeepSeek selects 6 of its 64 experts per token, so 512 tokens give each
expert ~48 rows. The count is below the 64-row granule the GEMM is padded up to, and the padded rows
are computed and read back in full.

The gpt-oss-20b model selects 4 of its 32 experts per token, so the same token count gives each
expert 64 rows, at the granule. Its offload gains. At pp2048 each DeepSeek expert receives
`M_e` ~192 rows, above the granule, and the offload gains 1.29x (the table above).

The backend's default placement therefore requires two thresholds: `M_e >= ROCKET_MOE_M_BUCKET`,
the 64-row granule, and a work per dispatch `M_e · K · N` of at least `ROCKET_MOE_MIN_WORK`,
340 mega-MACs (§"The accept boundary, and why the per-expert row count is the wrong handle"). On
DeepSeek the work threshold admits the experts from `M_e` = 118, past ~1250 tokens in a
micro-batch. The default keeps DeepSeek's experts on the CPU at short prefill and places them on
the NPU at long prefill. One model is therefore placed differently by prompt length. At pp512
(`M_e` = 48) the row threshold declines the experts, and DeepSeek measures 25.9 on the device, the
experts-on-CPU rate.

> [!NOTE]
> A residency check alone does not detect this failure. `MUL_MAT_ID` placement has two independent
> failure modes, and a residency check sees only one of them. A residency check therefore does not
> show that the gpt-oss-20b result transfers to another MoE model.

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).**

| | Q4_K_M |
|---|---|
| tg64 CPU | 7.73 |
| tg64 NPU | 7.77 |

Decode with the NPU backend loaded matches the CPU, and runs on the CPU
([decode-gemv.md](decode-gemv.md)). The MoE's small active-param count keeps decode at ~7.7 t/s
from a 9.65 GB file, roughly a dense 3 B model's rate.

**Interactive.** TTFT and stream rate are derived from the measured pp and tg rates
[host-computed].

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
table of §"gpt-oss-20b (MXFP4, MoE)").

**Verdict.** DeepSeek-V2-Lite combines MLA attention and MoE in one model. Both have op-level
handlers, and the MoE expert offload is a default gain **at long prefill only**. Attention stays on
the CPU: the FLASH_ATTN handler is bit-faithful for MLA's head dims, and the DeepSeek FA path is not
exercised on the device. It is also dispatch-bound with no effect on prefill at these lengths.

The routed experts take the native-quant path by default at pp2048 (1.29x over the experts-on-CPU
configuration in the native-quant table). Below the per-dispatch work threshold they stay on the
CPU. The fp16 expert path loses more here than on gpt-oss (0.25x->0.59x the CPU) and is never taken
by default.

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

**Conditions:** RK3588 on the RK1 (31 GB, 29 GB free, no swap), NPU clock 600 MHz, the flags in
§"Method", 2026-07-04. Q8_0 and Q4_K_M GGUFs from `unsloth/phi-4-GGUF` (base model
`microsoft/phi-4`, arch `llama`).

**Prefill (prompt processing), t/s (the NPU's job).**

| test | Q8_0 CPU->NPU | Q4_K_M CPU->NPU |
|---|---|---|
| pp512  | 3.83 -> 10.61 (2.8x) | 3.53 -> 10.31 (2.9x) |
| pp1024 | 3.70 -> 12.13 (3.3x) | 3.49 -> 11.93 (3.4x) |
| pp2048 | 3.60 -> **12.20 (3.4x)** | 3.40 -> **11.79 (3.5x)** |

NPU prefill runs at up to 3.5x the CPU rate, against a slow (~3.5 t/s) 14 B CPU baseline. With no
F16 that fits, both GGUFs are quantized, and both show the quantized-GGUF pattern. The speedup rises
with prompt length (2.8x->3.4x Q8_0, 2.9x->3.5x Q4_K_M) as the per-micro-batch dequantization
amortizes across the `-b 2048 -ub 2048` window
([quant-prefill-microbatch.md](quant-prefill-microbatch.md)). At pp2048 both measure ~12 t/s, so
neither quantized GGUF prefills faster than the other ([not-mac-bound.md](not-mac-bound.md)).
Q4_K_M's slightly higher ratio (3.5x against 3.4x) comes from its slightly slower CPU baseline, and
its NPU rate is the lower of the two. The ~12 t/s lies between Gemma-4-12B's Q4_K_M (13.3) and
Qwen3.6-27B's Q4_K_M (7.8), in order of model size.

Because its F16 GGUF does not fit, this model cannot reach the higher prefill of an F16 GGUF
(Gemma-4-12B's F16 measures ~15-17 t/s). On this board a quantized GGUF is the only option.

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).**

| | Q8_0 | Q4_K_M |
|---|---|---|
| tg64 CPU | 1.43 | 2.17 |
| tg64 NPU | 1.42 | 2.19 |

Decode runs on the CPU in both configurations ([decode-gemv.md](decode-gemv.md)), so the two rates
match. At 14.66 B and Q8_0 it is slow (1.4 t/s), with 14.5 GiB streamed from LPDDR per token.
Quantization is the only setting that raises it: Q4_K_M nearly doubles it to 2.2 t/s
(8.28 GiB/token). Interactive use of this model needs a quantized GGUF in practice.

**Interactive.** TTFT and streaming are derived from the measured pp and tg rates.

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
it gives ~3.5x NPU prefill at length (~12 t/s) and a 2.2 t/s stream, the fastest effective turn in
both scenarios. Where accuracy matters more than stream speed (1.4 against 2.2 t/s), Q8_0 + NPU
(14.51 GiB) is the near-lossless option, at essentially the same prefill. On this model the NPU's
gain is in prefill: it shortens a ~9.5-minute first-token wait on a 2048-token RAG prompt (Q8_0
CPU) to ~2.8 minutes.

### SmolVLM2-2.2B-Instruct (vision-language, mtmd)

SmolVLM2-2.2B is a vision-language model: a SigLIP-SO400M vision encoder feeds a SmolLM2-class
language model, run end to end through llama.cpp's `mtmd` path. The language model (arch `llama`)
has 1.81 B parameters, and "2.2B" is the combined vision and language count. The model offloads
both a vision encoder and an LLM prefill, through two different paths that are measured
separately [HW sweep 2026-07-04].

**Conditions:** RK3588 on the RK1, NPU clock 600 MHz, 2026-07-04. GGUFs from
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

**Prefill (prompt processing), t/s (the LLM half, the NPU's job).**

| test | F16 CPU->NPU | Q8_0 CPU->NPU | Q4_K_M CPU->NPU |
|---|---|---|---|
| pp512  | 27.97 -> **98.98** (3.5x) | 26.78 -> 47.93 (1.8x) | 27.01 -> 48.57 (1.8x) |
| pp1024 | 29.46 -> **71.99** (2.4x) | 27.29 -> 47.11 (1.7x) | 26.25 -> 47.43 (1.8x) |
| pp2048 | 27.45 -> **52.59** (1.9x) | 23.81 -> 39.90 (1.7x) | 24.61 -> 40.56 (1.6x) |

F16 measures 98.98 t/s at pp512, against Llama-3.2-3B's 61.8. At 1.81 B this is the smallest LLM
here, with the fewest FLOPs per matmul. The F16 speedup falls steeply with prompt length
(3.5x->1.9x, the NPU rate 99->53). The small models show this pattern (Llama-3.2-3B, Phi-4-mini),
and this one most strongly: per-op readback and dispatch are a large fixed fraction, and grow with
M.

The quantized GGUFs measure a lower multiple (~1.6-1.8x) than the ~2x of the other small dense
models. The cause is the CPU baseline rather than a slower NPU. This model's CPU quantized prefill
is fast (~24-27 t/s, near its F16 CPU rate), while the NPU's quantized rate (~40-48 t/s) is in line
with the others. At pp2048 F16 (53) still leads the quantized GGUFs (~40). A quantized GGUF does not
prefill faster than F16, and only fits in less memory ([not-mac-bound.md](not-mac-bound.md)).

**Decode (generation), t/s (CPU-bound on both, LPDDR-bandwidth-limited).**

| | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| tg64 CPU | 5.11 | 9.60 | 14.66 |
| tg64 NPU | 5.19 | 9.83 | 14.54 |

Decode runs on the CPU in both configurations ([decode-gemv.md](decode-gemv.md)), so the two rates
match. At 1.81 B it measures F16 5.1 -> Q8_0 9.7 -> Q4_K_M 14.5 t/s, the last from a 1.03 GB file.
The Q4_K_M tg64 spread is ±3.3 t/s, run-to-run variation at this decode rate on a small model, and
NPU 14.54 against CPU 14.66 is inside it.

**Interactive.** TTFT and streaming are derived from the measured pp and tg rates.

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
both. As the smallest model here it measures 99 t/s F16 prefill at pp512 (3.5x) and a 14.5 t/s
Q4_K_M stream. Use Q4_K_M: a 1.03 GB language model plus a ~0.8 GB fp16 mmproj fits any board, with
~1.7x NPU prefill and 14.5 t/s decode.

On the vision half the SigLIP encoder offload works and is faithful (cosine 0.99998), at ~1.19x.
The SO400M dimensions (fc2 K=4304, off the %32 grid) and the static-weight-only offload rule leave
the attention core, fc2 and every norm and activation op on the CPU. The graph takes 272
handoffs. The remaining optimization is a resident vision-encoder path (the `rocket_siglip_encoder`
pattern) wired into the mtmd frontend, rather than a new datatype, and it is not implemented. As it
stands, the drop-in clip path is faithful and modestly faster than the CPU.

## ASR (whisper.cpp via ggml-rocket)

Whisper runs through the `ggml-rocket` drop-in `.so` on stock whisper.cpp, with no fork. The NPU
runs the encoder, once per 30 s audio window: a stack of self-attention and FFN blocks, all large-M
`MUL_MAT`. The decoder is autoregressive (M=1 GEMV per token) and runs on the CPU in both
configurations, as LLM decode does ([decode-gemv.md](decode-gemv.md)). The figures below are
encoder latency, CPU against NPU, across the model sizes. Faithfulness is the transcript (WER) and
the encoder-output cosine [HW sweep 2026-07-04].

**Conditions:** RK3588 on the RK1, NPU clock 600 MHz, stock whisper.cpp with ggml-rocket, the
encoder timed alone by `whisper-bench -w 0`, the warm mean of reps 2-4, 2026-07-04.

**Encoder latency, ms (the NPU's job).**

| model | encoder d_model / layers | CPU -> NPU |
|---|---|---|
| tiny.en        | 384 / 4  |   697.7 ->   591.9 (1.18x) |
| base.en        | 512 / 6  |  1580.0 ->  1223.4 (1.29x) |
| small.en       | 768 / 12 |  5711.5 ->  3718.9 (1.54x) |
| medium.en      | 1024 / 24 | 19268.2 -> 10444.9 (1.84x) |
| large-v3       | 1280 / 32 | 36399.9 -> 17013.5 (**2.14x**) |
| large-v3-turbo | 1280 / 32 | 32957.2 -> 15544.1 (**2.12x**) |

The NPU is faster at every size, and the speedup grows monotonically with the encoder: 1.18x at tiny
to 2.1x at large. The matmul work grows as d_model², while the CPU-side host packing and readback
grow ~linearly, so the NPU gains as the encoder grows
([encodings/whisper-encoder.md](../encodings/whisper-encoder.md)). Fixed per-op host cost,
interleaved with the CPU-only conv/LayerNorm/softmax front end, dominates a small encoder
(tiny/base). A large encoder is dominated by matmuls, where the NPU gains most.

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
unchanged. Llama-3.2-3B F16 pp512 stays at ~61 t/s, within run-to-run noise of the 61.79 headline.

**Verdict.** Whisper encoder offload is faithful to the CPU, and its speedup grows with model size:
~1.2x at tiny/base and ~2.1x at large-v3 and large-v3-turbo. Its transcripts are byte-identical to
the CPU's. Stock `whisper-bench` and `whisper-cli` load the `.so` through `GGML_BACKEND_PATH`, with
no fork. The raw data and full per-rep timings are in
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

### The offload taxonomy

Encoders offload, and autoregressive decode mostly does not. Two rules set every ASR model's NPU
speedup:

- **Encode always offloads**, at 1.2x-2.7x, scaling with encoder size × audio length. A large
  encoder (Voxtral's Whisper-large-v3, MOSS's Whisper-Medium) gains more than the small SAN-M
  encoder (SenseVoice/FunASR).
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
~1.25x to ~2x on 9% of decode caps the gain at ~1.10x [expected]. The fixes that remain are
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

Detection runs SSD-MobileDet (uint8) through the tflite-rocket external delegate on
`librocketnpu`. A single inference is bound by the host's cube scatter and gather rather than by
compute or submits, so its latency follows the host path. The NPU's larger gain is throughput in a
multi-camera pool, the configuration Frigate runs. Accuracy is COCO mAP rather than perplexity.

**Conditions:** RK3588, NPU clock 600 MHz, `native_int8=1`.

### Single-stream latency

Single-stream latency is set by host work. The measured host path gives these warm latencies, with
COCO mAP at parity with the CPU [HW sweep 2026-09-28]:

| Model | Warm latency | COCO mAP, NPU | COCO mAP, CPU |
|---|---:|---:|---:|
| MobileDet | 55.7 ms | 0.4040 | 0.3980 |
| SSD MobileNet v2 | 52.1 ms | 0.3229 | 0.3215 |
| EfficientDet-Lite0 | 101.3 ms | 0.3023 | 0.2996 |

**Conditions:** RK1, NPU clock 600 MHz, governor `performance`, the process on the A76 cores, four
rotated passes, 2026-09-28. The host path is the last row of the change table in §"Evidence:
Detection (tflite-rocket)". mAP is over 100 COCO images.

On `ondemand` with a 1.2 GHz minimum frequency and no `taskset`, the same build measures 75.8, 70.5
and 135.0 ms. The governor lowers the clock of the cores that an offloading process leaves idle
([cpu-governor-and-offload.md](cpu-governor-and-offload.md)).

Three host costs account for most of the difference between a per-element host path (205 ms on
MobileDet) and the measured one:

- Per-element cube packs. Where the conv entries pack their cubes one element at a time, that loop
  is 83% of the int8 depthwise call ([../depthwise-conv.md](../depthwise-conv.md)).
- Buffers sized for the largest call and synced whole on every call
  ([bo-sync-cost.md](bo-sync-cost.md)).
- The int32 readback and the host requant. On MobileDet's direct convs the `read` phase is the
  int32 copy, 262 ms over 21 invokes against 19 ms of `FINI_BO`. The int8-out writer removes it
  ([../encodings/out-cvt-converter.md](../encodings/out-cvt-converter.md)).

**Faithfulness: COCO-val mAP, the detection correctness check.** On the host path of
§"Single-stream latency", COCO mAP over 100 val images is at parity with the CPU [HW sweep
2026-09-28]. The table there gives all three detectors. Over 500 val images, an earlier delegate
build also measures parity:

| | mAP@[.5:.95] |
|---|---|
| CPU (int8 reference) | 0.3318 |
| FOSS-NPU delegate | **0.3321** |

**Conditions:** SSD-MobileDet, 500 COCO val images, the delegate build of 2026-06-21, with
fp16-approximated depthwise, per-tensor depthwise int8 and the host requant on every conv.

**Not characterized:** the 500-image mAP on the host path of §"Single-stream latency".

The Δ is +0.0002, CPU parity. On that build the two depthwise approximations cost ~0 mAP, so the
accelerated detector is numerically as good as the CPU int8 reference.

**Throughput: multi-camera pool, aggregate detection_fps.** The pool runs one detector process per
camera (`num_threads: 1`), live on the RK1 through Frigate's `rocket.py` detector, with four
SSD-MobileDet cameras [HW sweep]. The `ROCKET_CPU_AFFINITY` variable pins each process to a
distinct A76 core. The P contexts then spread across the four big cores instead of colliding on
one.

**Conditions:** RK1, the NPU IRQs pinned to the big cores (`npu_set_irq_affinity.sh throughput`),
an earlier host path on which a single inference took ~336 ms.

**Not characterized:** the pool on the host path of §"Single-stream latency".

| pool size P | detection_fps | scaling |
|---|---|---|
| 1 | 3.20 | 1.00x |
| 2 | 6.00 | 1.88x |
| 3 | 8.00 | 2.50x |
| 4 | 9.55 | **2.98x** |

Aggregate throughput scales ~3x at P=4, with each of the four A76 cores running one detector (~73%
busy) while the A55 cores handle video decode. Scaling is below linear (2.98x, not 4x) because a
MobileDet inference is DRAM-bandwidth-bound in its host scatter/gather phase. Per-inference latency
rises 338->424 ms as the four contexts contend for bandwidth. The live pool therefore stays below
the submit-bound limit of the delegate alone, which scales 1.00, 2.17, 3.11 and 3.56x at P=1 to 4.
The pool needs no delegate or driver change, and it is how Frigate runs cameras (one process each).

**Verdict.** For detection the FOSS-NPU delegate is an accelerator with CPU-parity accuracy, and
its gain is throughput. One pinned process per camera serves ~3x the aggregate detection rate of a
single stream. The NPU takes the conv/matmul work of four cameras. Each A76 core runs one
detector's host work at ~73% busy, and the A55 cores handle video decode. Single-stream latency is
bound by host work, so it falls with host-path changes rather than with changes to the NPU submit
path.

## ONNX Runtime (ort-rocket)

`ort-rocket` is an ONNX Runtime execution provider (EP) on `librocketnpu`. It claims whole encoders,
and single ConvTranspose and Conv nodes (Conv behind a cost rule), and leaves the rest of a graph on
the CPU EP. The ort-rocket README has the full per-model table. The table below gives each model's
single-stream speedup and faithfulness against the CPU EP [HW sweep 2026-10-10, rocket-userspace
`1401c46`, ort-rocket `3c50635`].

**Conditions:** for the vision rows, RK1, NPU clock 600 MHz, governor `performance`, warm medians
over four passes with the configuration order rotated. The EP and the CPU EP both run under
`taskset -c 4-7` at 4 intra-op threads. The CPU EP runs at `ORT_ENABLE_ALL`, and the EP session
with spinning off. The pix2pix row also has spinning off. The Laya rows use ONNX Runtime's default
thread pool and the default session (§"ONNX Runtime thread spinning").

| Model | Offloaded | Single stream, x the CPU EP | Faithfulness against the CPU EP |
|---|---|---:|---|
| SigLIP-B/16 | plain ViT, d=768, 196 tokens | 2.44-2.52x | hidden-state cosine 0.9999864 |
| CLIP ViT-B/16 | plain ViT, d=768, 197 tokens | 2.39-2.48x | hidden-state cosine 0.9999955 |
| RF-DETR nano | windowed DINOv2 ViT-S, d=384 | 1.57-1.59x | COCO mAP 0.5049 against 0.5051 |
| RF-DETR base | windowed DINOv2 ViT-S, d=384 | 1.61-1.65x | COCO mAP 0.564 against 0.564 |
| SAM ViT-B | ViT-Det, d=768, 4096 tokens | 1.03-1.05x | mask IoU 0.9998 |
| Depth Anything v2 Small | DINOv2 ViT-S, global attention over 1370 tokens | 1.33-1.35x | depth cosine 0.9999999 |
| Laya English | ModernBERT-large, d=1024, 48-512 tokens | 1.99-3.95x | 4 of 4 answers agree |
| pix2pix generator | ConvTranspose and Conv nodes | 2.91x at 4 threads, 1.86x with the ConvTranspose claim alone | PSNR 77.3 dB |

A model's single-stream ratio falls as its attention-to-projection ratio, `ntok / 4d`, rises. The
ratio is averaged over the layers, with a windowed layer counted at its window length:

| Model | `ntok / 4d` |
|---|---:|
| SigLIP-B/16, CLIP ViT-B/16 | 0.06 |
| RF-DETR nano | 0.17 |
| RF-DETR base | 0.33 |
| SAM ViT-B | 0.49 |
| Depth Anything v2 Small | 0.89 |

The ratio orders the vision models coarsely, and SigLIP and CLIP lead. RF-DETR base measures
slightly above nano on this build, so the ratio does not separate values that close. SAM measures
lowest of all.

SAM's four global layers attend over 4096 tokens, 16.8 million scores per head, the largest single
attention in the set. They are the likely cause of its low ratio [hypothesis]. Its attention also
runs on its own per-window path rather than on the flash attention head range. The head range's
shorter host passes therefore do not apply to it
([attention-offload-crossover.md](attention-offload-crossover.md) §"The head range's host passes at
an encoder shape").

ONNX Runtime's CPU kernels run the Laya graph at ~77 GOP/s, which places Laya's ratio above what the
vision models predict.

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

Three of the six vision encoders gain as well, and three do not move [HW sweep 2026-10-10]:

| Encoder | Time with spinning off, x the default session |
|---|---:|
| SigLIP-B/16 | 0.84-0.88x |
| Depth Anything v2 Small | 0.89-0.92x |
| RF-DETR nano | 0.90-0.93x |
| CLIP ViT-B/16 | 0.98-1.02x |
| RF-DETR base | 0.98-1.01x |
| SAM ViT-B | 0.98-1.02x |

**Conditions:** RK1, governor `performance`, `taskset -c 4-7`, 4 threads, the build of the
ort-rocket table above, 2026-10-10.

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

- **Board and operating point.** The NPU clock boots at 200 MHz and rises under load. The 600 MHz
  rate is set at module load with `rocket_npu_clk_hz=600000000` and confirmed during the run,
  because a sample taken while idle between repetitions measures 200. The CPU baseline runs
  unpinned. F16 decode is LPDDR-bandwidth-bound, so the A55 cores add bandwidth, and confining the
  process to the A76 cluster starves it ([decode-gemv.md](decode-gemv.md)).
- **Warm discipline.** Figures are warm medians (`llama-bench -r 2` plus a discarded warmup).
- **Flags.** Quantized GGUFs run at `-b 2048 -ub 2048`, because a quantized GGUF re-dequantizes to
  fp16 per micro-batch. On the 2026-06-28 builds that preceded the per-model tables, `-ub 2048`
  measured 2.1-2.25x the default `-ub 512` on Qwen3.5-9B and Qwen3.6-27B. On the 2026-08-29/30
  builds it measures 0.91-1.53x over eleven models, model-dependent and negative on two
  ([quant-prefill-microbatch.md](quant-prefill-microbatch.md)).
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

gpt-oss's faithfulness also rests on `test-rocket-moe`, not on a greedy match. The section "The
accept boundary" and its MoE threshold table show why the greedy comparison has no resolution
here.

The gpt-oss-20b row's 22.0 and 28.1 are six-run means of the default. The DeepSeek-V2-Lite row's
26.0 and 28.4 are the default configuration of the accept-boundary map at `M_e` 48 and 192
(§"The accept boundary, and why the per-expert row count is the wrong handle"). The DeepSeek default
measures 27.62-28.37 t/s at pp2048 across the 2026-08-27 runs on llama.cpp b10558, 1.28-1.31x the
experts-on-CPU configuration of each run's session. The native-quant table of
§"DeepSeek-V2-Lite (MLA + MoE, Q4_K_M)" gives 27.9 [HW sweep 2026-08-27].

#### Tuning per model

Each figure is a mean over three rotated interleaved passes. The memory is reset before every
configuration, and the governor pinned [HW sweep 2026-08-28..31, RK1, 600 MHz].

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
the TTFT and streaming decomposition to ~10%.

#### gpt-oss-20b

The 2026-07-14 measurements come from one board in one session, clock pinned, with every baseline
re-measured alongside the configuration under test [HW sweep, 600 MHz, 2026-07-14].

The 2026-08-27 prefill table comes from one board and one `.so`. Its default column is a mean of
six runs. One run directly after a rebuild fell at the bottom of that spread, and looked like a 10%
regression. The `ROCKET_MOE=0` control returned 13.63 at pp2048 on every binary and every
repetition. The board had therefore not drifted, and the spread belongs to the default
configuration.

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
| Time to first token, 2048-token prompt | CPU 162 s, NPU 156 s (~4%), and a combined `pp2048+tg128` point of CPU 11.61, NPU 12.03 t/s | 2026-07-02, the same default | ~174 s against ~73 s from the 2026-08-27 pp2048 rates. The combined point is not re-measured on the expert path |

#### MoE placement threshold

The map's test points are `-r 3` each, on one board and one `.so` [HW sweep 2026-08-27, RK1].

Each test point's pair count follows from its distance from 1.00. The offloaded configuration
varies ~15% run to run on DeepSeek. A test point a few percent from parity therefore needs
repeats, and one at 1.6-2.1x does not. The pair counts:

- `M_e` = 72 and 144 are means of four adjacent pairs.
- `M_e` = 96 is a mean of eight, because it is the test point that decides where the threshold
  goes.
- `M_e` = 192 is a mean of two, taken either side of the rebuild.
- Every gpt-oss test point is a single pair. Its margin from parity, 0.6-1.1, is 4-7x the same
  spread [host-computed].

At `M_e` = 96, one of the eight pairs measures 0.936 and the other seven measure 1.019-1.076. All
eight are placement-identical (4767 resident, 0 streamed, the same ingest profile). Nothing but
variance separates them, and there is no mechanistic ground to drop the low one. Pooled, the test
point is 1.049, and over the seven repeats alone it is 1.065. Either way it is marginal.

The `M_e` = 48 row is the null test point. The default threshold declines it, so both
configurations are the same placement, and they agree to 0.3%. The agreement shows that the map
measures the threshold and not run-to-run drift.

The DeepSeek `M_e` = 72 test point has four pairs, every one under 1.00 and none above 0.970.

### The third architecture, and the threshold declines it correctly

The decline-side test points are [HW sweep 2026-08-27, RK1, 600 MHz pinned, governor
`performance`]. Each is `-r 3`, over three process-pairs with the configurations alternated. The
warm-up process is discarded and the page cache dropped first.

The default's 1.000x is the null test point, and shows that the threshold is what is being
measured. The `ROCKET_MOE=0` control's spread across three processes is 0.5%, and `ROCKET_MOE=1`'s
is 3.0%. The 17% deficit is far outside both, and the per-pair ratios barely move: 0.845 / 0.824 /
0.832.

Residency confounds the decline-side result. A partially resident configuration is strictly worse
than a resident one. A `ROCKET_MOE=1` result above 1.05 would therefore have falsified the
threshold outright. One result below 1.05 cannot by itself separate "the work per dispatch is too
small" from "it cannot be held". The residency-matched pair and the residency axis separate them.

The residency axis in full: `ROCKET_MOE_CACHE_MB` raises residency 62.4% -> 65.7% -> 70.8%, and the
ratio moves 0.834 -> 0.856 -> 0.863. The 8.3 points of residency give 0.029 of ratio. Extrapolated
linearly, full residency would measure 0.965x. Reaching parity would need 110% residency
[extrapolated from two points, 29 points beyond the data, indicative, not measured]. On that trend
`ROCKET_MOE=1` loses at every residency at all.

The ingest ratio in the break-even table compares the earlier pack with the current one in one
session. At the 2026-08 ingest (~32 s on DeepSeek and ~36 s on gpt-oss, a pack of ~505-545 MB/s)
the four test points break even after ~16 300, ~4 800, ~3 800 and ~2 100 tokens. The table scales
those counts by 0.47x and 0.51x.

#### Excluded causes

Bucket padding is the alternative explanation that the map sets aside. The bucket ladder rounds
`M_e` up (64, 96, 128, 192, 256…). DeepSeek's points therefore alternate 33% padding (72 -> 96,
144 -> 192) with none (96 -> 96, 192 -> 192). Padding predicts a sawtooth, in which each step onto
an unpadded bucket is steeper than the step off one.

The measured curve is monotone and saturating instead. Its per-row slopes across `M_e` 72 -> 96 ->
144 -> 192 are 0.0042, 0.0037 and 0.0018. One step agrees with the sawtooth and the next
contradicts it, while a falling slope is exactly what amortizing a fixed per-dispatch cost looks
like. Padding is capped at ~33% by construction and is not the term that flips the sign.

### gpt-oss-20b native-quant expert path and results

The `llama-bench` warm-up: three timed pp512 repetitions take 22.60, 22.51 and 21.27 s each, in
order [HW sweep 2026-10-09, one run]. The run exercised 1629 experts, where a one-repetition run
exercises 1575. Two builds differ only in the pack's speed, with ingests of ~41 s and ~21 s. Over
three rotated passes each, they measure pp512 at 21.18 and 22.67 t/s.

At gpt-oss's expert (K=N=2880, group 576, three workers), `moe_gw_bench` measures the group-wise
call across the 2026-10-09 build. It moves 4.16-4.19 -> 3.40-3.52 ms at M 64 and 12.05-12.09 ->
8.92-10.26 ms at M 256. Each is two rotated runs a build [HW sweep 2026-10-09, RK1, 600 MHz,
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

### The accept side of the same floor, and the budget above it

The accept-side test point is [HW sweep 2026-08-28, RK1, 600 MHz pinned, governor `performance`].
It is `-r 2`, over three process-triples (configuration order a balanced Latin square). The warm-up
process is discarded and the page cache dropped before every test point. A single-test-point run
first showed the configuration is reachable. The default's placement line is identical in all three processes,
which dropping the page cache before every test point gives.

The cost model's `P` term is required: `-ub` 4096 needs `-p` >= 4096, so the prefill work doubles
while the per-prefill constants do not. A `P`-fixed extrapolation of the same two points gives
~1.007, wrong by 5%.

#### The auto budget and `ROCKET_MOE_CACHE_MB`

The pooled `ROCKET_MOE=0` control measures 10.29 / 10.32 / 10.31 / 10.35 / 10.39 / 10.27 across six
processes spanning seven hours. Its 1.2% spread is non-monotone, so it is the configuration's own
spread rather than drift.

The two middle budgets are not separable: 10.76 against 10.83 is 0.65% apart, against
configurations whose own spreads are 1.1% and 1.6%. The plateau is seven processes over two
budgets and the automatic budget is three, but 12000 and 28000 are n=1.

The fall's existence rests on the plateau-to-automatic step, which has the depth. Its depth past 79
stacks does not. Per-repetition spread grows monotonically with placement across the whole ladder
(±0.00 -> ±1.02), a memory-pressure signature rather than anything about the offload. The two
single-process budgets are the least trustworthy for the same reason: they were taken where the
spread is widest.

#### RAM and IOVA budgets

Three runs show the RAM budget binding on gpt-oss-20b, two at the default 5 workers and one at 8.
The window grows 1.6x between them, from 19200 to 30720 MB. The admitted set is byte-identical
across all three, at 62 stacks, 24140 MB RAM and 15693 MB IOVA. All three announce `Bound by RAM`
[HW sweep 2026-09-02, RK1, `-b 2048 -ub 2048 -p 2048`, raw in
[data/ro-session/trackd17-moe-preflight.md](data/ro-session/trackd17-moe-preflight.md)].

Every pre-flight decline recorded for this model says the same. That is 14 occurrences over five
distinct budget and placement points, from an induced 11.7 GB budget to a pinned 25.7 GB one. None
names the window.

One archived gpt-oss line measures 1.3926 at 10.70 MB/expert against the others' 8.07. That is a
different expert geometry, not a disagreement.

#### Over-placement mechanism

The curve is itself the evidence that the placed experts' source pages stay resident. Suppose they
were promptly reclaimable after ingest, since they are read once and this is a prefill-only run.
The hot set would then be `anonymous + unplaced GGUF only`, which computes to 20.6 / 21.7 / 22.4 /
22.2 / 22.4 GB across the five budgets. That curve is flat, crosses nothing and explains nothing. A
curve exists, so those pages are not promptly reclaimed.

### The induced-scarcity ladder: the budget mechanism does not produce an inverted U

The ladder tests the MoE budget hypothesis: that the per-expert admission charge covers only the
placed experts' GGUF source bytes. The rest of a still-mapped expert GGUF is then charged to nobody,
so the pre-flight over-places once the true hot set approaches RAM. The hypothesis has an arithmetic
fit to five budgets on one model, and no second model on this board reaches the regime unaided.

The ladder induces the scarcity on a second model. It holds `B` GiB of touched anonymous memory
beside the run and walks `B` up. The account predicts the turn once `25.3 + B` exceeds ~31.7, which
is at `B` ~ 6-7 GB and not at `B` = 4. The predicted collapse into the 0.42-0.97x partial-residency
regime does not happen. Neither does the earlier turn that the "any memory pressure at all"
alternative predicts.

The ladder ran one process per step (`perf/data/moe-ballast-ladder.sh`).

### Qwen3.5-9B

IQ4_XS was measured as a gap-finder for the quantized-prefill path: it tests whether the path covers
importance-matrix quants. It finds no gap. A CPU fallback would give ~1x at pp2048 [expected],
against the 3.1x measured.

The combined `pp2048+tg128` llama-bench point (F16 NPU 11.96, Q4_K_M NPU 16.77, IQ4_XS NPU 17.01
t/s) validates the TTFT-plus-stream decomposition to ~5%.

The raw `llama-bench` and `llama-perplexity` output is in [data/qwen35-9b.md](data/qwen35-9b.md).

### Phi-4-mini-instruct

The combined `pp2048+tg128` llama-bench point (F16 NPU 18.35, Q8_0 20.89, Q4_K_M 21.43 t/s) runs
~15-20% below the TTFT+stream sum. That margin is the one by which the `total turn` figures are
optimistic.

The raw output is in [data/phi4-mini.md](data/phi4-mini.md).

### Gemma-4-12B-it

The fp16 prefill path was first brought up on this model.

The combined `pp2048+tg128` llama-bench point (F16 NPU 7.37, Q8_0 8.72, Q4_K_M 9.61 t/s) validates
the TTFT-plus-stream decomposition.

The raw output is in [data/gemma4-12b.md](data/gemma4-12b.md).

### Qwen3.6-27B (hybrid Gated-DeltaNet, Q4_K_M)

The combined `pp2048+tg128` point (CPU 1.70, NPU 5.57 t/s) validates the TTFT-plus-stream
decomposition.

The raw output is in [data/qwen36-27b.md](data/qwen36-27b.md).

### Llama-3.2-3B-Instruct

The combined `pp2048+tg128` llama-bench point (F16 NPU 21.06, Q8_0 21.89, Q4_K_M 22.34 t/s) runs
below the TTFT+stream sum, which sets the margin by which the `total turn` figures are optimistic.

The raw output is in [data/llama32-3b.md](data/llama32-3b.md).

### Ministral-3-3B-Instruct-2512

The per-model benchmark format of this record was first validated on Ministral-3-8B.

The combined `pp2048+tg128` llama-bench point (F16 NPU 19.10, Q8_0 20.06, Q4_K_M 20.86 t/s) runs
below the TTFT+stream sum, which sets the margin by which the `total turn` figures are optimistic.

The raw output is in [data/ministral3-3b.md](data/ministral3-3b.md).

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

### Phi-4 (14B)

The combined `pp2048+tg128` llama-bench point (Q8_0 NPU 7.48, Q4_K_M NPU 8.29 t/s) validates the
TTFT and stream decomposition. It runs ~10-15% below the naive TTFT+stream sum, because the 128
tokens that `pp2048+tg128` decodes read a full KV cache.

### SmolVLM2-2.2B-Instruct (vision-language, mtmd)

The combined `pp2048+tg128` llama-bench point (F16 NPU 27.06, Q8_0 26.11, Q4_K_M 27.68 t/s)
validates the TTFT and stream decomposition.

### ASR (whisper.cpp via ggml-rocket)

The size trend is the crossover that the encoder analysis in
[encodings/whisper-encoder.md](../encodings/whisper-encoder.md) predicted. An integration without
KACC and with a per-call fd measured base.en below break-even. K-accumulation and the reductions in
per-submit overhead take every model size above 1x.

The per-step decoder costs of large-v3 and large-v3-turbo are from whisper-bench's full timing, in
the same NPU run as the encoder figures.

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
ggml-rocket `6bf55cc` and rocket-userspace `ffd4ccc`, `rocket` 1.3.0 at 600 MHz. It ran both clips,
both precisions, CPU and NPU, two fresh processes each, and both runs of every configuration
matched. jfk is char-identical everywhere. On the hard clip, F16 is char-identical CPU==NPU, and Q8
differs by one word (word 13, 246 against 249 characters). The Q8 NPU and both F16 transcripts are
byte-identical to the 2026-07-21 ones. The Q8 CPU transcript is the one that changed [HW sweep, RK1,
2026-10-12].

### Detection (tflite-rocket)

Host-path changes, and moving the requant on chip, took MobileDet from 205 to 56 ms. Each row of the
change table is the warm latency once that change is in. All rows share the operating point of
§"Single-stream latency" [HW sweep, 2026-09-27 and 2026-09-28]:

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

### ONNX Runtime (ort-rocket)

The 2026-10-10 build shortened the head range's host passes. Against the 2026-10-09 build, SAM moved
1.02-1.03x -> 1.03-1.05x, while the ratio of every family on the head range rose 5-17%.

Each CPU EP process in §"The CPU EP baseline on the RK3588" started after a memory reset.

### ONNX Runtime thread spinning

Each vision-encoder ratio is paired within one of four passes. On the 2026-10-09 build, with 8
unconfined threads, the same split measured SigLIP 0.75-0.79x and RF-DETR nano 0.83-0.85x.
