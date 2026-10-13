# Tuning the rocket NPU stack

This guide lists the options to set for each workload on the RK3588, with the measured gain of
each. The performance options are on by default, and the default settings provide most of the
gain. The choices that remain are a few opt-in options. Their value depends on the model, the
workload and the free RAM.

Each project's `API.md` is the per-flag reference, with a table of each flag's meaning.
[perf/](perf/) holds the evidence behind each figure. For per-model behavior and sampling, see
[MODEL-NOTES.md](MODEL-NOTES.md). For the full benchmarks, see
[perf/benchmarks.md](perf/benchmarks.md).

**Conditions:** RK3588, NPU clock 600 MHz, warm medians [HW sweep]. The baseline is the same
model on the CPU at 8 threads. A section states any other conditions. pp*N* is
`llama-bench`'s [prefill](GLOSSARY.md) test over *N* tokens, and `-ub` is llama.cpp's
micro-batch size.

**Not characterized:** the per-model gain of most options on most models. The option defaults
do not depend on the model. A gain measured on one model names that model, and
[What is not yet measured](#what-is-not-yet-measured) lists the coverage.

## Quick start

1. Complete the [prerequisites](#prerequisites), which include raising the NPU clock to
   600 MHz.
2. Leave the [default options](#default-settings) on.
3. For a quantized GGUF whose fp16 size fits in free RAM, set `ROCKET_QUANT_RESIDENT=auto`.
   Keep the default `-ub`, and do not add `-b 2048 -ub 2048`.
4. For a quantized GGUF whose fp16 size does not fit, run it at `-b 2048 -ub 2048`.
5. For an F16 model used for repeated prefill, with ~2x its size free, set
   `ROCKET_F16_RESIDENT=auto`.
6. For a mixture-of-experts (MoE) model, the expert offload is on by default and needs no
   option. Whether `-b 2048 -ub 2048` helps depends on the model: measure it
   ([Options by workload](#options-by-workload)).
7. Choose the precision for model fit and decode speed, not for prefill speed
   ([Precision](#precision)).

## Prerequisites

Do these once. Every figure in this guide assumes the following steps:

1. Raise the NPU clock to 600 MHz with the `patches/rocket` clock patch. Load the module with
   `rocket_npu_clk_hz=600000000`.
2. Run under `sudo -E`. Plain `sudo` strips the environment, which drops the `/dev/accel`
   privilege context and every `ROCKET_*` and `GGML_BACKEND_PATH` setting. `-E` keeps them.
3. Set `GGML_BACKEND_PATH` to the absolute path of `libggml-rocket.so`.
4. Compare warm runs only. Discard the cold run. `llama-bench` discards it for you. Use `-r 3`.
5. Confirm that each option took effect. Run once with `ROCKET_LOG_STDERR=1`, and look for the
   mode and budget line. Add `ROCKET_MM_PROFILE=1` for the phase breakdown.

The NPU boots at 200 MHz (one-fifth speed). Raising the clock to 600 MHz gives ~1.43x:
Gemma-4-12B pp2048 goes 7.98 -> 11.40 t/s [HW sweep]. A 900 MHz clock gives no further gain
([perf/clock.md](perf/clock.md)).

> [!CAUTION]
> With `power/control=on` pinned, an `rmmod` at 900 MHz hard-locks the board
> ([perf/clock.md](perf/clock.md)).

> [!NOTE]
> With a wrong `GGML_BACKEND_PATH`, the run falls back to the CPU silently. It prints one
> `failed to load` line, and then nothing runs on the NPU.

The NPU clock drops at idle and rises under load, so a cold run measures ~15% low. Without
`ROCKET_LOG_STDERR=1`, `llama-bench` hides every rocket line, and a mis-set option is
invisible.

> [!NOTE]
> Read the phase breakdown as an upper bound. It shows which phases ran, not what each costs
> in wall-clock time. Its buckets are thread intervals that include contention, and inside a
> model they sum to several times the wall-clock time. For one term, the value derived from
> wall-clock time was 33x lower [HW sweep 2026-09-01, RK1].

## Default settings

These options are on by default. Leave them on, and set one to `0` only to measure its effect.
Each gain is over a baseline with that option off.

| Option | Default | Function | Gain | Measured on |
|---|---|---|---|---|
| `ROCKET_KACC` | `ROCKET_KACC=1` | fp16 [K-accumulation](GLOSSARY.md) on the NPU | +19% | Gemma-4-12B [HW sweep] |
| `ROCKET_REUSE` | `ROCKET_REUSE=2` | CBUF DATA_REUSE | +7% more | Gemma-4-12B [HW sweep] |
| `ROCKET_MM_ASYM` | `ROCKET_MM_ASYM=1` | Asymmetric tiling | +6-9% on F16 | Qwen3.5-9B, Gemma-4-12B [HW sweep] |
| `ROCKET_FLASH_ATTN` | On | Attention offload | Gains above ~2K context (measured 2026-06-26) | Gemma-4-12B F16 ([Attention offload](#attention-offload)) |
| `ROCKET_DEQUANT_THREADS` | On | Threaded dequantization of quantized weights | +19-33% | [perf/quant-prefill-microbatch.md](perf/quant-prefill-microbatch.md) |
| MoE expert offload | On | Mixture-of-experts FFNs on the NPU | [MoE routed experts on the NPU](#moe-routed-experts-on-the-npu) | gpt-oss-20b [HW sweep] |

For a dense F16 model the tuned configuration is mostly the default, and the F16 figures in
[perf/benchmarks.md](perf/benchmarks.md) already use it. The large gains from tuning are on the
quantized path, where the llama.cpp and rocket defaults leave speed unused.

## Workload variables

Four variables select the options, and each has a section below.

### Model class

Each class offloads different work to the NPU:

- Dense LLM
- Mixture-of-experts (MoE)
- Multimodal (a vision encoder plus an LLM)
- ASR (Whisper)
- Detection

### Precision

The precision is F16, a quantized GGUF (`Q4_K_M`, `Q8_0`, `IQ4_XS` or MXFP4), or a native
integer path inside the model. It sets both the RAM footprint and which residency option
applies.

Precision (`ROCKET_INT4`, `ROCKET_INT8` or a `Q4_K_M` GGUF) reduces RAM and makes a model fit.
It is not a speed option. Quantization does not speed prefill at this operating point
([perf/not-mac-bound.md](perf/not-mac-bound.md)). Choose it to make a model fit or to speed
[decode](GLOSSARY.md), never to speed prefill.

### Prompt profile

The prompt profile is how many tokens reach prefill per turn. It decides whether the NPU is on
the critical path at all:

| Profile | Prefill per turn | What runs |
|---|---|---|
| Short | A chatty turn, tens of tokens | An F16 prefill of 32 tokens or more offloads (`ROCKET_MIN_M`), and a quantized one stays on the CPU below 512 (`ROCKET_MIN_M_QUANT`). Either way the turn is decode-bound, so the NPU does little for it |
| Medium | 512-2048: a RAG chunk, a document paragraph, an agentic tool result | NPU prefill engages and gains |
| Long or batched | >2048, or a large system pre-prompt re-processed every turn | The NPU's best case, and where residency and attention offload gain the most |

A large system pre-prompt turns a chat workload into a prefill-bound one. An agent with a
several-thousand-token tool or persona preamble loads the NPU heavily every turn, even where
the user's message is short.

### Resource budget

The resource budget is free RAM and disk. Every residency option trades RAM for speed. The
native integer paths trade disk for runtime RAM, because they require a full-precision GGUF as
their source. Each entry in the [option reference](#option-reference) gives its RAM cost.

## Options by workload

Prefill is fastest on F16. A quantized GGUF reduces memory use and increases decode speed. In
the table, "fits" compares the fp16 footprint that the residency options need against free
RAM.

| Workload | Model | Set | Reason |
|---|---|---|---|
| Interactive chat, short prompts | Any | A `Q4_K_M` GGUF for decode. No NPU options | Prefill is below the offload threshold, and the turn is decode-bound |
| Medium or long prefill | F16, fits | Defaults only (KACC, REUSE, ASYM and FA on) | Already the fastest prefill |
| Agentic or RAG, repeated prefill | F16, fits ~2x | `ROCKET_F16_RESIDENT=auto` | Packs the weights once |
| Medium or long prefill | Quantized GGUF, fp16 does not fit | `-b 2048 -ub 2048` | Amortizes the per-micro-batch dequantization. Two of eleven models lose, so measure it |
| Agentic or RAG | Quantized GGUF >= ~8 B, fp16 fits | `ROCKET_QUANT_RESIDENT=auto` at the default `-ub` | Costs the full fp16 footprint |
| Agentic or RAG | Quantized GGUF < 4 B, fp16 fits | `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`. Do not combine it with `-ub 2048` | The class has a non-dequantization loss at `-ub 2048` |
| Medium or long prefill | MoE (gpt-oss, DeepSeek-V2-Lite) | `-b 2048 -ub 2048`. The expert offload needs no option | Default-on, and it decides its own placement. Eligibility is per architecture: gpt-oss offloads from `-ub 512` up, DeepSeek-V2-Lite only past ~1250 tokens in a micro-batch. `-ub 2048` is 0.908x on Qwen3-30B-A3B, so measure it on another MoE |
| Any | Too large for RAM at F16 | A `Q4_K_M` GGUF, or `ROCKET_INT4=1` from an F16 GGUF | Footprint, not speed ([Native int8 / int4 / bf16](#native-int8--int4--bf16)) |
| Batch or multi-stream | Any | One process per stream, each with `ROCKET_CPU_AFFINITY` | More concurrency raises aggregate throughput |
| ASR | Whisper | `ROCKET_CPU_AFFINITY`. Keep `ROCKET_MIN_M` at 8 or above | The encoder offloads, and beam decode stays on the CPU |
| ASR, short utterances | CTC or transducer encoder | `ROCKET_MIN_M=16` | A 1-5 s utterance is below the default threshold |
| Detection | `tflite-rocket` | `native_int8=1`. One process per camera, each with `ROCKET_CPU_AFFINITY` | Exact int8 convolutions, and throughput from a process pool |

### Interactive chat (short prompts, decode-bound)

Almost all of a short user turn's wall-clock time is decode. Decode is CPU-bound and
bandwidth-limited on both backends ([perf/decode-gemv.md](perf/decode-gemv.md)), and the NPU
barely touches the turn. A quantized GGUF's prefill is below its offload threshold and runs on
the CPU. An F16 prefill of 32 tokens or more offloads, but the turn stays decode-bound. Three
facts follow:

- Quantization is the option that helps, through decode speed. `Q4_K_M` decodes ~2.8x faster
  than F16 (Ministral-3-8B: F16 1.4 -> Q4_K_M 3.8 t/s decode [HW sweep]) and fits a smaller
  board. Choose the quantization on decode speed and RAM, not on the NPU.
- Residency and `-ub` have no effect here, because there is no large prefill to amortize them
  over.
- A long system pre-prompt is the exception. A chat with a large preamble (an agent persona,
  tool schemas, retrieved context) has a large prefill every turn, and is the agentic case
  below.

### Agentic / tool-use (large, repeated prefill)

An agent re-processes a growing context (system prompt, tool schemas, prior steps, tool
outputs) every turn. Every step therefore has a large prefill, which is the NPU's work. The
fixed setup cost of residency amortizes across turns, so residency repays it best here: the
same weights serve many turns.

Attention offload needs no setting. It is automatic, and gains once the context passes ~2K
([perf/attention-offload-crossover.md](perf/attention-offload-crossover.md)). The MoE expert
offload also needs no setting, and declines itself where it would not gain.

### RAG / long-context / document processing (large one-shot prefill)

This workload is a single large prefill: a retrieved passage, a pasted document, a filled
context window. The largest NPU prefill gains are here. The speedup over the CPU grows with
prefill length, up to Qwen3.6-27B's 4.4x at pp2048 [HW sweep].

The settings are those of the decision table above. For an F16 model, add
`ROCKET_F16_RESIDENT=auto` as RAM allows. [Attention offload](#attention-offload) matters at
these lengths, and is on by default. Leave it on.

### Batch / multi-stream throughput (offline, or many detectors)

Optimize aggregate throughput, not single-request latency. Run one process per stream, each
pinned to a distinct A76 core with `ROCKET_CPU_AFFINITY`. Single-stream latency is bound by
the host cube gather, not by the NPU. More concurrency therefore raises throughput, and a
faster submit does not. The detection pool below is the measured case.

### ASR: Whisper (whisper.cpp)

The NPU accelerates the encoder. The decoder is autoregressive (M=1 GEMV) and stays on the CPU.
The gain grows with the model, from tiny.en at 1.18x to large-v3 at 2.14x [HW sweep]. Larger
models gain more, because encoder matmul work grows ~d_model² and host packing ~linearly. No
options are needed beyond the prerequisites and `ROCKET_CPU_AFFINITY`. Three points need
attention:

- **Keep `ROCKET_MIN_M` at 8 or above.** The default beam-5 search in whisper.cpp presents M=5
  per decode step. A threshold of 4 offloads that small GEMV, at a net 1.40x loss end to end.
  The default 32 keeps beam decode on the CPU, and so does 16. A threshold of 128 measures the
  same as 32 for whisper-small F16: 0.995-1.006x, transcript byte-identical
  [HW sweep, 2026-10-09].
- **Short utterances on a CTC or transducer encoder want `ROCKET_MIN_M=16`.** Parakeet runs
  12.5 encoder rows a second of audio, so a 1-5 s voice command sits under the default
  threshold and nothing offloads. At 16, Parakeet TDT 0.6B through whisper.cpp's
  `parakeet-cli` ties the CPU at 1.3-1.8 s and runs 1.18-1.73x faster at 2-6 s [HW sweep]. It
  frees 34-61% of the CPU. When another process shares the cores, add
  `OMP_WAIT_POLICY=PASSIVE`. See
  [perf/asr-cpu-relief.md](perf/asr-cpu-relief.md#short-utterances-below-the-f16-floor).
- **The best transcription target is large-v3-turbo.** It pairs the full 32-layer encoder
  (NPU-accelerated, 2.12x) with a 4-layer decoder (~6x cheaper per step). The accelerated
  encoder is then most of the transcription's work.

### Detection: Frigate / TFLite (tflite-rocket)

The delegate's main setting is a delegate option, not an environment variable. Pass it through
`load_delegate` or `--option`. `native_int8=1` selects the exact-int8 convolution path, and is
the default in the Frigate `rocket.py` plugin. MobileDet COCO mAP is at CPU parity: 0.4040
against the CPU's 0.3980 over 100 images [HW sweep 2026-09-28].

Throughput comes from a process pool, one process per camera, each pinned with
`ROCKET_CPU_AFFINITY`. The pool goes from 3.20 to 9.55 detection_fps from one process to four
(P=1->4), 2.98x. A single MobileDet stream runs 56 ms warm with the governor pinned, and 76 ms
on `ondemand`. Pin the governor for latency ([perf/benchmarks.md](perf/benchmarks.md)
§"Single-stream latency").

## Option reference

Each entry gives:

- What the option does
- When to use it
- The RAM or disk it costs
- The measured gain
- How to confirm that it took effect

### `-b 2048 -ub 2048` (llama.cpp)

- **What it does:** a quantized GGUF re-dequantizes to fp16 per micro-batch. The llama.cpp
  default `-ub 512` re-decodes the whole model every 512 rows, and `-ub 2048` spreads that
  fixed cost. It is a llama.cpp flag, not a `ROCKET_*` one.
- **When to use it:** a quantized prefill of >= ~512 rows, where `ROCKET_QUANT_RESIDENT=auto`
  is unavailable because the fp16 image does not fit resident. It does not apply to F16 (no
  dequantization) or to short prompts (one micro-batch).
- **RAM cost:** the larger micro-batch grows the activation and compute buffers ~4x
  (512->2048). The growth is negligible against the weights, but not zero. On a RAM-tight
  board with no swap, like the one measured, confirm that it still fits.
- **Gain:** 0.91-1.53x over `-ub 512`, model-dependent, on eleven models of 0.75-30.53 B
  [HW sweep 2026-08-29/30]. Two of the eleven lose.

The gain splits at ~4 B. Dense models above 8 B gain 1.18-1.53x, and within that group
parameter count does not order the gain. Models under 4 B measure 0.94-1.13x, with `smolvlm2`
(1.81 B) a loss at 0.941x. The flag is nearly free. Where the fp16 image fits, residency at the
default `-ub` gains more than this flag on each of the seven models with both measured. On the
three of 8 B or more it is 1.37-1.75x against this flag's 1.18-1.42x [host-computed]
([Residency in place of `-ub 2048`](#residency-in-place-of--ub-2048)).

The split holds for dense models only. MoE models measure 0.91-1.32x. `qwen3-30b-a3b` is
30.53 B and measures 0.908x, the second loss of the eleven, because only ~3 B are active per
token. Read a MoE model from its own row, not from its parameter count.

The quantization type does not change NPU prefill throughput: `Q4_K`, `IQ4_XS` and `Q8_0`
converge within noise.

> [!NOTE]
> Measure the flag on the target model instead of assuming its class's gain. Its value on the
> 9B and the 27B fell by the same 0.68 factor between two builds
> ([Revision history](#revision-history)).

> [!NOTE]
> Never compare a `-ub 512` figure against a `-ub 2048` one. The difference appears as a
> regression that does not exist ([perf/quant-prefill-microbatch.md](perf/quant-prefill-microbatch.md)).

> [!NOTE]
> A `-ub` gain measured on a loaded host overstates the gain on an idle one. Behind a single
> busy core, the flag measures 1.48x, 1.53x and 1.12x on three of the models. The busy core
> takes A76 capacity from the dequantization pool, which the `-ub 512` configuration uses 4x
> as often. The load turns `smolvlm2`'s loss into a gain.

### `ROCKET_QUANT_RESIDENT=auto`: quant weights held resident

- **What it does:** dequantizes each quantized weight to fp16 once and holds it in
  [resident](GLOSSARY.md) NPU BOs. It removes both the per-micro-batch dequantization and the
  per-call pack, and lifts quantized prefill to fp16 parity or past it. Perplexity is
  bit-identical to streaming [HW sweep 2026-06-28].
- **When to use it:** a quantized GGUF used for repeated prefill (agentic, RAG), where the
  model's fp16 size also fits RAM. It trades the quantization's RAM saving back for the full
  fp16 footprint. Run it at the default `-ub`.
- **RAM cost:** roughly the fp16 model size free (Qwen3.5-9B ~18 GB), plus the NPU
  [IOVA window](GLOSSARY.md). There is no disk cost beyond the quantized GGUF. Use `auto`,
  which sizes the budget from free RAM, rather than a blanket `=1`. On a model larger than the
  default 2 GB budget, `=1` places only part of it, which is a net loss against streaming.
- **Gain:** 1.35-1.75x over streaming at the default `-ub`, against 1.00-1.66x combined with
  `-ub 2048`, on seven models of 1.8 to 11.9 B [HW sweep 2026-08-29/31]. On a model that does
  not fit, it falls back to streaming, correctly.
- **Confirm:** `ROCKET_LOG_STDERR=1` prints the one-shot budget decision.

#### Residency in place of `-ub 2048`

Run residency at the default `-ub`, and do not combine it with `-b 2048 -ub 2048`. Every model
measured both ways gains more alone, seven of seven across 1.8-11.9 B [HW sweep 2026-08-29/31].
No measured model gains more from the combination.

| Model size | Models measured | Alone, at the default `-ub` | Combined with `-ub 2048` | Alone over combined |
|---|---:|---:|---:|---:|
| < 4 B | 4 | 1.35-1.51x | 1.00-1.21x | +21-34% |
| >= ~8 B | 3 | 1.37-1.75x | 1.32-1.66x | +3.5-25% |

Three causes add up, and the third applies only to a partly placed model:

- Below 4 B, `-ub 2048` has a real non-dequantization loss that the dequantization gain masks.
  The remainder once dequantization is removed measures 0.789-0.864. The combination incurs
  that loss before residency applies.
- On every model, residency also removes the per-call pack and upload, and the default `-ub`
  has four times as many calls to remove them from. This cause is why the 9B still gains 5.5%
  alone, although its own `-ub` remainder is a gain at 1.038x.
- On a partly placed model, `-ub 2048` uses RAM for buffers that would otherwise hold weights.

`gemma4-12b` places only part of its weights. Both configurations stop at the same 9535 MB
`MemAvailable` reserve, rather than at the resident-weight budget. The configuration at the
default `-ub` reaches that reserve 2.6-2.8 GB later. `-b 2048 -ub 2048` grows the
activation and compute buffers about 4x, and uses that RAM for the micro-batch instead of for
weights. The measured placement is 83-87% resident alone against 73-74% combined.

On a partly placed model, raising `-ub` costs residency directly. To compare
configurations on such a model, read the `[f16-resident]` outcome line of each.

**Not characterized:** the `qwen35-08b` class, which cannot resolve a ratio this size at an
affordable number of runs.

### `ROCKET_F16_RESIDENT=auto`: F16 weights held resident

- **What it does:** the F16 counterpart of `ROCKET_QUANT_RESIDENT=auto`. It packs the all-K F16
  weights once and reuses them across micro-batches and turns.
- **When to use it:** an F16 model that fits ~2x in RAM (the resident tiles plus the source),
  used for repeated prefill.
- **RAM cost:** ~2x the fp16 model size.
- **Gain:** 1.078-1.084x at pp2048 on three 3-4 B F16 models [HW sweep 2026-08-29]. A
  fusable projection group (Q\|K\|V, gate\|up) is placed as one combined weight. The gain
  therefore includes fusion's shared packA, which added ~+5.7% to residency alone on the build
  of 2026-07-15.
  [perf/weight-residency-fusion.md](perf/weight-residency-fusion.md) has the mechanism and the
  comparison.

> [!CAUTION]
> `ROCKET_PREPACK_MADVISE` reclaims the source weights for a prefill-only run, and breaks CPU
> decode. Do not use it for an interactive or serving run.

### Host CPU pinning

For a prefill-heavy run, pin the host process to the big cluster with `taskset 0xf0`. The
prefill gains:

| Model | Prefill gain, pinned |
|---|---:|
| `Qwen3.5-0.8B` F16 | 1.10-1.13x |
| `Qwen3.5-9B` held quant-resident | 1.06x |
| `Gemma-4-12B` F16 streaming every weight | 1.05-1.06x |

**Conditions:** RK1, 600 MHz, governor `performance` [HW sweep 2026-09-01].

The NPU time is identical with and without pinning, so the gain is on the host, and it does
not depend on residency. Adding `-t 4` gains a little more on the smaller models and nothing on
the larger. The gain tracks the share of instructions the unpinned run retires on the little
cluster, at roughly a third to a half of that share.

> [!NOTE]
> Leave a decode-heavy run unpinned. F16 decode is LPDDR-bandwidth-bound, and pinning lowers it
> 34% on a 12B F16 model.

The gains of pinning and of residency overlap, so do not multiply their published figures. Both
remove A76 pack work, and the second option applied finds less of it left. On `Qwen3.5-9B`,
residency combined with `-b 2048 -ub 2048` is worth 1.67x unpinned and 1.61x pinned. Both
options from stock measure 1.77x, not the 1.83x the two multiply to [HW sweep 2026-09-01/02,
RK1].

### `ROCKET_MIN_M` and `ROCKET_MIN_M_QUANT`

- **What it does:** sets the smallest prefill, in rows, that offloads to the NPU.
  `ROCKET_MIN_M` applies to an F16 model and defaults to 32. `ROCKET_MIN_M_QUANT` applies to a
  quantized model and defaults to 512. A smaller prefill runs on the CPU.
- **When to change it:** for ASR, as [ASR: Whisper](#asr-whisper-whispercpp) describes.
- **Gain:** the F16 prompt-size crossover sits at 21-25 rows on the 0.8B, 3B and 8B models
  measured. The default `ROCKET_MIN_M` of 32 takes every gain above it. It measures 1.20-1.46x
  at pp32 and 1.96-3.90x at pp128 against the faster CPU configuration [HW sweep, 2026-10-09].
  The CPU governor was `performance`.

### MoE routed experts on the NPU

- **What it does:** runs the mixture-of-experts FFNs (`MUL_MAT_ID`) on the NPU. It places
  experts per expert stack: the weight tensor of one `MUL_MAT_ID` op, which holds every
  expert's matrix. A quantized expert takes the native-int8 resident path
  (`ROCKET_MOE_NATIVE`), which ingests each expert once into int8 codes. That removes the
  per-micro-batch host dequantization that makes the fp16 expert path a loss. The fp16 path
  (`ROCKET_MOE_NATIVE=0`) re-dequantizes every expert every micro-batch, and is a net loss on
  both models measured. The default never takes it.
- **When to use it:** nothing to set. The offload has been on by default since 2026-08-27, and
  decides per expert stack. It claims a stack only where it can reserve the whole stack up
  front, and where the per-expert GEMM repays its own dispatch. `ROCKET_MOE=0` leaves the
  experts on the CPU.
- **RAM cost:** gpt-oss-20b holds ~13.4 GB of int8 codes on the NPU. The GGUF source must
  coexist, because MoE decode reads the active experts from it every token. The offload
  therefore charges ~21 GB against a budget of `MemAvailable` − 6 GiB. On a 31 GB board that
  admits 63 of its 72 expert stacks. The remaining 9 stay on the CPU, which is a partial
  offload and not a loss.
- **Time:** a one-time expert ingest inside the first prefill, per `llama_context`. It takes
  ~21 s on gpt-oss-20b and ~18 s on DeepSeek-V2-Lite [HW sweep 2026-10-09, 600 MHz]. Its NPU-BO
  pack runs ~1.3 GB/s, so it grows with the bytes held resident rather than with the expert
  count.
- **Gain:** gpt-oss-20b MXFP4 at `-b 2048 -ub 2048` measures 1.81x the CPU at pp512 and 2.38x
  at pp2048. It measures 1.64x -> 2.08x over the experts-on-CPU configuration across
  pp512-pp2048 [HW sweep 2026-08-27, 600 MHz pinned]. At the llama.cpp default `-ub 512` it is
  ~1.6x/1.7x over experts-on-CPU, with no collapse: the `-ub 512` loss belongs to the fp16
  path.
- **Confirm:** `ROCKET_LOG_STDERR=1` prints the resident and streamed expert split at teardown.
  Under the default the split is 100% resident. A nonzero streamed count means a limit the
  pre-flight could not see ahead of the ingest, most likely an exhausted NPU IOVA window.

The gain figures were measured on a build without the NEON group-wise readback. A 2026-10-09
build with that readback measures pp512 1.093-1.098x the build before it [HW sweep 2026-10-09,
`ondemand`]. The gain on a build with the readback is not characterized.

> [!NOTE]
> An expert that `llama-bench`'s warm-up never selected ingests inside a timed repetition. A
> MoE prefill figure from `llama-bench` therefore includes part of the ingest.

#### Residency pre-flight

A residency pre-flight makes the sign of the offload independent of the host's RAM. A stack's
resident cost is known from its tensor alone. The placement check therefore reserves the whole
stack before the first ingest, and leaves on the CPU what it cannot reserve. A declined stack
costs nothing at the backend boundary, because the backend's buffer type is the CPU buffer
type. The default is therefore bounded below by the experts-on-CPU baseline at any RAM size.

#### Eligibility per architecture

Eligibility is per architecture, not universal. Two size thresholds decide each op. One is the
minimum tile height of the per-expert rows, 64 by default (`ROCKET_MOE_M_BUCKET`). The other is the work in one dispatch
(`ROCKET_MOE_MIN_WORK`), in mega-MACs of `M_e · K · N`, where `M_e` =
`n_tokens · n_used / n_expert`.

gpt-oss-20b selects 4 of 32 experts, each 2880×2880, and clears both thresholds from `-ub 512`
up. DeepSeek-V2-Lite selects 6 of 64 experts, each 2048×1408, which is 2.88x less work per
dispatch at the same row count. It clears the thresholds only past ~1250 tokens in a
micro-batch. The threshold is on work rather than on a per-expert row count, because no row
count separates the two. At `M_e` = 96, gpt-oss measures 1.77x and DeepSeek ~parity.

> [!NOTE]
> One MoE model's ratio does not predict another's.

`ROCKET_MOE_MIN_WORK=240` admits DeepSeek's marginal test point (1.05x at `M_e` = 96). That
test point repays its ~18 s expert ingest only after ~7 700 tokens of prefill at that shape
[host-computed]. The count takes the 2026-08 ratio at the 2026-10-09 ingest
([perf/benchmarks.md](perf/benchmarks.md) §"The third architecture, and the threshold declines
it correctly"). Lower the threshold only for a workload that prefills that much.

#### `ROCKET_MOE=1`

`ROCKET_MOE=1` is for comparison, not a recommended setting. It claims every op the handler
can compute, reserves nothing, and ignores both size thresholds. Where the stacks nearly fit,
it is faster than the default: 6-13% at pp512 and 18-21% at pp2048. Where they do not fit, it
is unsafe, and measures below the experts-on-CPU baseline (0.97x at pp512 on an induced 12 GB
budget). Raise `ROCKET_MOE_CACHE_MB` instead. It recovers most of that gain, and keeps both
the zero-streamed property and the guarantee on the sign.

#### `ROCKET_MOE_CACHE_MB` on gpt-oss-20b

On gpt-oss-20b on a 31 GiB board, more budget places more stacks, and on this model class
that raises throughput ([measured budget series](#budget-series-on-gpt-oss-20b)). A budget of
28000 places 71 of 72 stacks. It recovers 79% of the pp2048 gain from the default to
`ROCKET_MOE=1`, and measures above `ROCKET_MOE=1` at pp512. The pre-flight still guarantees
the sign. The budget leaves only ~2.8 GB of the headroom the 6 GiB auto reserve exists for (KV
cache, activations). It is a setting for a known working set, not a new default.

#### Expert-dominated models

On an expert-dominated model the direction inverts, and a larger `ROCKET_MOE_CACHE_MB` lowers
the gain. Qwen3-30B-A3B holds 29 of its 30.5 B parameters in the experts (17.28 GiB GGUF). At
`-ub` 4096 on the same board, its gain over experts-on-CPU is a plateau, then a fall to parity
[HW sweep 2026-08-28, RK1, 600 MHz]. The plateau is 18000-21000 MB
([measured budget series](#budget-series-on-qwen3-30b-a3b)).

The cause is in the admission charge. It counts the GGUF source bytes of the experts it
places, but not of the ones it leaves on the CPU. The CPU reads those from the same mmap every
micro-batch, and they are equally unreclaimable.

#### `ROCKET_MOE_CHARGE_ALL_SOURCE=1`

`ROCKET_MOE_CHARGE_ALL_SOURCE=1` is the charge that counts both, and it is opt-in. It charges
the whole mmapped weight buffer once, up front, and leaves only the int8 codes and scales per
stack. On this board it places Qwen3-30B-A3B near 46 stacks against the default's 79.

It is not the default for two reasons. The 3.2% it targets cannot be confirmed on this board at
any affordable number of runs. The same charge also moves gpt-oss-20b from 63 stacks to about
58 [expected: derived from that model's 1.521 charge factor, not measured]. That model gains
from a larger budget.

Use it on an expert-dominated model whose GGUF is large against RAM. Pinning
`ROCKET_MOE_CACHE_MB` to the plateau does the same job with a number you chose.

#### Charge factor

The units of `ROCKET_MOE_CACHE_MB` are not bytes of RAM, so a budget does not transfer by
arithmetic on RAM alone. The charge per expert is `N·K` int8 code bytes, plus `N·(K/group)·4`
scale bytes, plus that expert's GGUF source stride. A budget therefore places less than it
names, by a charge factor of `1 + 4/group + source_bits_per_weight/8`. The offload measures
the factor for you.

The pre-flight's `resident budget reached after N expert stacks (X MB RAM, Y MB IOVA)` line has
the charge and the codes side by side. `X/Y` is 1.610 on Qwen3-30B-A3B Q4_K_M against 1.617
derived, and 1.521 on gpt-oss MXFP4 against 1.538 derived. The formula sizes a budget to ~1%
before a run, and the log gives the exact factor after it. A Q8_0 MoE would need about 2.07
[expected: derived from bits/weight, not measured]. Convert a budget, do not copy it:

```
budget_MB  ~=  factor x (MemTotal - the whole expert GGUF - ~1.2 GiB runtime headroom)
```

#### Board size

The table gives what fits on each board size, and where `auto` over-places. Its rows are
computed from the charge-factor formula plus a stack's code size (`n_expert · K · N` bytes).

**Conditions:** every MoE figure here is from one 31 GiB board, the only RK3588 size measured.
The rows below 32 GB are arithmetic from that board's measurements, and are tagged [expected].
The 32 GB row's `auto` figures are estimates for a board of that size [expected]. The runs
reported budgets of 24425 MB on Qwen3-30B-A3B and 24672 MB on gpt-oss-20b. From those, the same
arithmetic gives 79.0 and 64.1 stacks, against 79 and 63 placed
([Board-size arithmetic](#board-size-arithmetic)).

| board (MemTotal) | Qwen3-30B-A3B Q4_K_M, 17.3 GiB | gpt-oss-20b MXFP4, 11.3 GiB | DeepSeek-V2-Lite Q4_K_M, 9.7 GiB |
|---|---|---|---|
| 32 GB (31.0 GiB) | ~67 of 144 stacks fit. Auto asks ~81. Pin 18000-21000 | whole stack fits. Auto asks ~65 of 72. Raise to 28000 | whole stack fits with room. Auto is right |
| | *measured: plateau 58-67, auto took 79* | *measured: auto took 63, 28000 took 71* | *measured: all 78 admitted* |
| 16 GB (15.4 GiB) | not viable: the GGUF alone is 17.3 GiB | ~12 of 72 fit. Auto asks ~24, 2x too many. Pin ~4600 [expected] | ~26 of 78 fit. Auto asks ~32. Pin ~7500, and needs `-ub` past ~1250 to clear the work threshold [expected] |
| 8 GB (7.6 GiB) | no | no: GGUF > RAM | no: GGUF > RAM |
| 4 GB | no | no | no |

Prefill touches every expert every micro-batch, so the working set is the whole expert GGUF,
however little of it is placed. A GGUF larger than RAM therefore thrashes, and no budget
setting changes that. At 8 GB and below, all three models are refused on GGUF size alone.

16 GB is the only size where the answer depends on the budget. There the GGUF fits for two of
the three models, and the question becomes how much int8 fits beside it. On a 16 GB board, pin
the budget, because `auto`'s uncharged remainder is largest at low placement. Placing 12 of
gpt-oss's 72 stacks leaves 60 stacks' worth of source, ~9.4 GiB, charged to nobody on a board
with 15.4.

`auto` is the default because it fails in the safe direction. Its flat 6 GiB reserve is 19% of
a 32 GB board, 37.5% of 16 GB and 75% of 8 GB. `auto` therefore withholds proportionally more
as the board gets smaller, and effectively turns the offload off at 8 GB. A stack the budget
cannot reserve is never claimed, so its layers run wholly on the CPU. Those layers match the
experts-on-CPU baseline, rather than a streamed partial-residency loss. Partial residency is a
`ROCKET_MOE=1` failure mode (52% resident, 0.97x), and the default cannot reach it.

### Native int8 / int4 / bf16

- **What it does:** `ROCKET_INT8=1` (with `ROCKET_INT8_HADAMARD=1`), `ROCKET_INT4=1` and
  `ROCKET_BF16=1` select the native int8, int4 and bf16 paths. All are numerically faithful:
  int4 and int8 are char-identical to fp16 greedy output, and bf16 is token-identical. All
  measure the same ~460 GOP/s throughput limit, because the NPU is DMA/dispatch-bound. Fewer
  bits therefore do not raise prefill speed.
- **When to use it:** only to make a model fit that would not at F16, or for bf16's fp32 range.
  Resident int8 in-model prefill is 0.60x fp16, and int4 is ~0.53x. No on-chip K-accumulation
  is implemented for the int32 partials, so each K-tile reads back. The result is slower, but
  a quarter to a half of the footprint.
- **Disk cost:** the native int4 and int8 paths quantize from a full-precision (F16) GGUF and
  require Hadamard rotation. They do not take a pre-quantized `Q4` file, so they use the disk
  of the larger F16 GGUF to save runtime RAM. With only a `Q4_K` GGUF, use the GGUF-quant
  streaming path (`-ub 2048` or `ROCKET_QUANT_RESIDENT`) instead. That path is a different
  mechanism, and the one with a speed gain.
- **Recommendation:** for RAM alone, a `Q4_K_M` GGUF at `-b 2048 -ub 2048` is simpler and also
  speeds decode. Native int4 is for the ¼ footprint from an F16 source.

### Attention offload

- **What it does:** `ROCKET_FLASH_ATTN` (on) offloads prefill attention at `n_kv >= 1024`. It is
  bit-faithful for the attention it implements (`softmax(scale·QKᵀ+mask)·V`). The handler
  declines an op that has attention sinks (gpt-oss, some others). It has no sink term, so
  accepting the op would silently compute a different softmax.
- **When it matters:** long contexts. It measures parity at <=1K, 1.50x at 8K and 1.25x at 16K
  on Gemma-4-12B F16 [HW sweep 2026-06]. Below ~2K it neither gains nor loses, and the default
  threshold handles that.
- **Diagnosis:** it rarely needs changing. If a model's NPU curve collapses past ~1K context
  but is normal below it, suspect an attention with sinks. The handler must decline such an
  op, and it does.
- **Related settings:** `ROCKET_FLASH_ATTN_MIN_KV` (1024) is the context threshold.
  `ROCKET_FA_CHAIN` (on) batches a worker's per-head submits into one job.
  `ROCKET_FA_ROWMAJOR` (on) hands the scores and P between its two jobs as rows, with no
  layout pass on the host. `ROCKET_FLASH_ATTN_NO_CTX=1` forces the per-call path, for
  comparison against the persistent context.

The gain figures are from a build without the flash attention's host-pass changes of
2026-09-27, 2026-10-10 and 2026-10-11. The gain on a build with them is not characterized.

## Appendix: measurements

### What is not yet measured

The option defaults do not depend on the model, so the recommendations hold. The per-model
default-against-tuned comparison is measured on a subset of models. Where a recommendation
implies a per-model figure that [perf/benchmarks.md](perf/benchmarks.md) does
not list, treat it as a projection, not a measurement.

| Option | Measured on | Not characterized |
|---|---|---|
| NPU clock at 600 MHz | Gemma-4-12B [HW sweep] | |
| `ROCKET_MM_ASYM`, `ROCKET_KACC` and DATA_REUSE | A few models, mostly Gemma-4-12B and Qwen3.5 | Every other model, which takes the default without a measurement |
| `-b 2048 -ub 2048` | Eleven models, 0.75-30.53 B [HW sweep 2026-08-29/30] | |
| `ROCKET_QUANT_RESIDENT` | Combined with `-ub 2048`, nine quantized models of 0.75-11.91 B [HW sweep 2026-08-28..31]. Alone and combined, seven models of 1.8 to 11.9 B | How the gain falls with the placed fraction. `gemma4-12b` is the only partly placed model on the board, at 83-87% alone against 73-74% combined, a two-point contrast on one model rather than a curve |
| `ROCKET_F16_RESIDENT` | A 3B F16 model [HW sweep] | |
| MoE expert offload | gpt-oss-20b [HW sweep] | |
| `ROCKET_MOE` size thresholds | Fitted on gpt-oss-20b and DeepSeek-V2-Lite, which disagree about which shapes are eligible. Qwen3-30B-A3B, a third architecture, confirms the threshold's sign but not its position, at 201 MMAC against the 340 default | The threshold's position on another architecture |
| `ROCKET_MOE_CACHE_MB` | Two budget series, gpt-oss-20b and Qwen3-30B-A3B, which invert. Both on the same 31 GiB board | The board-size axis. The small-board table is arithmetic |
| SmolVLM2 resident `rocket_siglip_encoder` vision path | No end-to-end benchmark. The generic clip drop-in is the only measured multimodal-vision figure (1.19x) | The resident path end to end |
| `ROCKET_MIN_M` | F16 models of 0.8B, 3B and 8B | A model smaller than 0.8B, or one with much narrower matmuls |

### Prerequisites

The 33x comparison of the phase breakdown against wall-clock time is a per-process readout
measurement [HW sweep 2026-09-01, RK1].

### Host CPU pinning

Each pinning gain is from rotated interleaved passes. The overlap with residency on
`Qwen3.5-9B` is from six rotated passes [HW sweep 2026-09-01/02, RK1].

### `-b 2048 -ub 2048` (llama.cpp)

Each figure is from rotated passes [HW sweep 2026-08-29/30]. Within the group above 8 B,
parameter count does not order the gain:

| Parameters | `-ub 2048` over `-ub 512` |
|---:|---:|
| 8.49 B | 1.176 |
| 8.95 B | 1.424 |
| 11.91 B | 1.257 |
| 14.66 B | 1.308 |
| 27.32 B | 1.532 |

The gain splits into dequant amortization and a non-dequant remainder, and the split is per
class [HW sweep 2026-08-31, six models]. The method is the dequant-cache control of
[perf/quant-prefill-microbatch.md](perf/quant-prefill-microbatch.md). In that control the 9B's
full `-ub` gain measures 1.430x, against 1.424x in the 2026-08-29/30 sweep. Of the 1.430, 1.378
is dequant amortization, and the non-dequant remainder is 1.038x. The documented
mechanism is ~90% of the gain there. On every measured sub-4B model the remainder is a real
loss that the dequant gain only masks:

| Model | Non-dequant remainder |
|---|---:|
| `smolvlm2` | 0.789x |
| `ministral3-3b` | 0.835x |
| `llama32-3b` | 0.863x |
| `phi4mini` | 0.864x |

The 0.8B straddles parity, unresolvable within its class's per-process spread. About half of
`smolvlm2`'s loss is the output de-tile at [2048,N]: the driver `read` bucket grows 64% at
identical output bytes. The rest is in `wait` [hypothesis, not isolated].

### `ROCKET_QUANT_RESIDENT=auto`

Each row is three rotated passes, with ratios paired within a pass. Every model is 100%
resident on every pass except `gemma4-12b`, which places part of its weights
[HW sweep 2026-08-29/31].

| model | alone, at the default `-ub` | combined with `-ub 2048` | alone over combined |
|---|---:|---:|---:|
| `smolvlm2` (1.81 B) | 1.346x | 1.002x | +34% |
| `llama32-3b` | 1.472x | 1.212x | +21% |
| `ministral3-3b` | 1.445x | 1.132x | +28% |
| `phi4mini` | 1.508x | 1.211x | +25% |
| `ministral3-8b` | 1.651x | 1.325x | +25% |
| `qwen35-9b` (8.95 B) | 1.752x | 1.661x | +5.5% |
| `gemma4-12b` (11.9 B) | 1.368x | 1.322x | +3.5% |

`gemma4-12b`'s +3.5% is the narrowest margin in the set, and its per-pass spread is the widest,
at 6.3%. Its sign is resolved, and its size is not.

### MoE routed experts on the NPU

#### Budget series on gpt-oss-20b

Every step is n=1: one `-r 3` process per configuration, behind a `drop_caches`, with 0
streamed in every configuration [HW sweep 2026-08-27, 600 MHz].

| setting | budget | stacks | pp512 | pp2048 | vs default |
|---|---|---:|---:|---:|---|
| default (auto) | 24672 MB | 63 of 72 | 21.96 | 26.51 | n/a |
| `ROCKET_MOE_CACHE_MB=26000` | 26000 MB | 66 | 22.90 | 27.85 | +4.3% / +5.1% |
| `ROCKET_MOE_CACHE_MB=28000` | 28000 MB | 71 | 23.79 | 30.25 | +8.3% / +14.1% |
| `ROCKET_MOE=1` (every stack placed) | none | 72 | 23.34 | 31.27 | +6.3% / +18.0% |

The stack counts are exact, and the percentages are not. One process on this board can sit
~10% off the level its own configuration repeats at. The series supports the direction (more
budget places more stacks, and on this model class raises throughput) rather than the four
deltas. The pp512 column also sits inside a ±1.4-2.1 within-process spread, so it is not
resolved finely at all.

The end steps of both `ROCKET_MOE_CACHE_MB` series are n=1 (every step of the gpt-oss series,
and Qwen3's 12000 and 28000), against a per-process spread on this board of ~10%. Only the Qwen3 plateau (n=7)
against auto (n=3) is deep enough to quote as a size.

#### Budget series on Qwen3-30B-A3B

At `-ub` 4096 [HW sweep 2026-08-28, RK1, 600 MHz]:

| budget | stacks | vs experts-on-CPU baseline | processes |
|---|---:|---:|---|
| 12000 MB | 38 | 1.032x | n=1 |
| 18000-21000 MB | 58-67 | 1.046x | pooled over seven processes |
| auto (24425 MB) | 79 | 1.014x | n=3 |
| 28000 MB | 90 | 0.999x | n=1 |

The 12000 MB and 28000 MB steps are one process each. The 28000 MB step therefore shows only that
one process measured parity.

#### Board-size arithmetic

The board-size arithmetic is checked against the 32 GB board, where it is out of sample for
two of the three models. One half of that check is independent evidence. The `auto` column is
mostly arithmetic the pre-flight itself performs. Reproducing it gives 79.0 predicted against
79 measured on Qwen3, and 64.1 against 63 on gpt-oss. Each figure comes from the run's own
reported budget and measured charge factor.

That agreement confirms the inputs, the per-stack code size `n_expert · K · N` and the charge
factor, rather than the model. The confirmation matters because the small-board rows are
computed from those inputs, but it is not independent.

The `fits` column is the independent half. It is `MemTotal − GGUF − headroom`, and it has one
free parameter. Its prediction for Qwen3-30B is ~67 stacks, which is the top of the measured
58-67 plateau, a boundary the arithmetic never saw. The same column says the gpt-oss and
DeepSeek stacks fit whole on this board, which is what was measured. The inputs are therefore
confirmed exactly, and the one prediction that could have been wrong falls on the measured
edge.

### ASR

The `ROCKET_MIN_M` comparison of 128 against 32 on whisper-small F16 is over three passes
[HW sweep, 2026-10-09]. 128 is the earlier default of `ROCKET_MIN_M`.

### Revision history

Readings taken on earlier builds:

| Measured | Quantity | Value | Conditions |
|---|---|---|---|
| 2026-06-28 | `-b 2048 -ub 2048` on the 9B | ~2.1x (2.098): 8.2 -> 17.2 t/s | The 2026-08-29/30 sweep measures 1.424x |
| 2026-06-28 | `-b 2048 -ub 2048` on the 27B | 2.25x (2.250) | The 2026-08-29/30 sweep measures 1.532x |
| 2026-06-28 | `ROCKET_QUANT_RESIDENT`, Qwen3.5-9B `Q4_K` pp2048 | Resident 24.6 vs streaming 15.8 = 1.56x (~0.92x the 26.8 F16) | Streaming at `-ub 2048` with serial dequant [HW sweep 2026-06-28] |

Both `-ub` readings are lower in the 2026-08-29/30 sweep by the same 0.68 factor
(2.098 -> 1.424, 2.250 -> 1.532). Between the two sweeps, three default-on host-cost cuts
raised the `-ub 512` baseline ~1.47x more than the `-ub 2048` configuration on each model. On
the 9B, the cuts raised its `-ub 512` baseline 2.33x against the `-ub 2048` configuration's
1.59x.
