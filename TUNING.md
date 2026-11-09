# Tuning the rocket NPU stack

This guide lists the options to set for each workload on the RK3588, with the measured gain of
each. The performance options are on by default, and the default settings provide most of the
gain. The choices that remain are a few opt-in options. Their value depends on the model, the
workload and the free RAM.

Each project's `API.md` is the per-flag reference, with a table of each flag's meaning.
[perf/](perf/) holds the evidence behind each figure. For per-model behavior and sampling, see
[MODEL-NOTES.md](MODEL-NOTES.md). For the full benchmarks, see
[perf/benchmarks.md](perf/benchmarks.md).

**Conditions:** RK3588, NPU clock ~816 MHz (the 600 MHz request) unless a figure states
928 MHz (the 900 MHz request), warm runs [HW sweep]. A prefill speedup's baseline is the same
GGUF on the CPU backend at 8 threads, its faster prefill configuration. A CPU decode rate is at
`taskset 0xf0 -t 4`, its faster decode configuration on llama.cpp b11539
([Host CPU pinning](#host-cpu-pinning)). A section states any other conditions. pp*N* is
`llama-bench`'s [prefill](GLOSSARY.md) test over *N* tokens, and `-ub` is llama.cpp's
micro-batch size.

**Not characterized:** the per-model gain of most options on most models. The option defaults
do not depend on the model. A gain measured on one model names that model, and
[Measurement coverage](#measurement-coverage) lists the coverage.

## Quick start

1. Complete the [prerequisites](#prerequisites), which include raising the NPU clock to
   928 MHz (the 900 MHz request).
2. Leave the [default options](#default-settings) on.
3. Run the process on the A76 cores with `taskset 0xf0 -t 4`
   ([Host CPU pinning](#host-cpu-pinning)).
4. For a quantized GGUF whose fp16 size fits in free RAM, set `ROCKET_QUANT_RESIDENT=auto`.
   Keep the default `-ub`, and do not add `-b 2048 -ub 2048`.
5. For a quantized GGUF whose fp16 size does not fit, run it at `-b 2048 -ub 2048`, and compare
   it with the default `-ub` on that model.
6. For an F16 model used for repeated prefill, with ~2x its size free, set
   `ROCKET_F16_RESIDENT=auto`.
7. For a mixture-of-experts (MoE) model, the expert offload is on by default and needs no
   option. Where the experts offload, run at `-b 2048 -ub 2048`. Where the backend leaves them
   on the CPU, keep the default `-ub` ([MoE routed experts](#moe-routed-experts-on-the-npu)).
8. Choose the precision for model fit and decode speed, not for prefill speed
   ([Precision](#precision)).

## Prerequisites

Do these once. Every figure in this guide assumes the following steps, and its Conditions
block states the NPU clock it was measured at:

1. Raise the NPU clock with the `patches/rocket` clock patch. Load the module with
   `rocket_npu_clk_hz=900000000`. At the 0.80 V rail this 900 MHz request runs at 928 MHz,
   because the clock follows the rail voltage ([perf/clock.md](perf/clock.md)). Most figures
   here were measured at the 600 MHz request, ~816 MHz.
2. Run under `sudo -E`. Plain `sudo` strips the environment, which drops the `/dev/accel`
   privilege context and every `ROCKET_*` and `GGML_BACKEND_PATH` setting. `-E` keeps them.
3. Set `GGML_BACKEND_PATH` to the absolute path of `libggml-rocket.so`.
4. Compare warm runs only. Discard the cold run. `llama-bench` discards it for you. Use `-r 3`.
5. Confirm that each option took effect. Run once with `ROCKET_LOG_STDERR=1`, and look for the
   mode and budget line. Add `ROCKET_MM_PROFILE=1` for the phase breakdown.

The NPU boots at 198 MHz (one-fifth speed). The 900 MHz request runs at 928 MHz and gives 1.66x:
Gemma-4-12B F16 pp2048 goes 14.8 -> 24.6 t/s [HW sweep 2026-11-02]. Over ~816 MHz, the 600 MHz
request, 928 MHz is a 1.13x clock step that gives 1.044x on matmul device time
([perf/clock.md](perf/clock.md) §"Rates from ~816 to 1019 MHz").

> [!CAUTION]
> With `power/control=on` pinned, an `rmmod` at the 900 MHz request (928 MHz) hard-locks the
> board ([perf/clock.md](perf/clock.md)).

> [!NOTE]
> With a wrong `GGML_BACKEND_PATH`, the run falls back to the CPU silently. It prints one
> `failed to load` line, and then nothing runs on the NPU.

The NPU clock drops at idle and rises under load, so a cold run measures ~15% low. Without
`ROCKET_LOG_STDERR=1`, `llama-bench` hides every rocket line, and a mis-set option is
invisible.

> [!NOTE]
> Read the phase breakdown as an upper bound. It shows which phases ran, not what each costs
> in wall-clock time. Its buckets are thread intervals that include contention, and inside a
> model they sum to several times the wall-clock time. For one term, the breakdown measured 33x
> the value derived from wall-clock time [HW sweep 2026-09-01, RK1].

## Default settings

These options are on by default. Leave them on, and set one to `0` only to measure its effect.
Each gain is over a baseline with that option off.

| Option | Default | Function | Gain | Measured on |
|---|---|---|---|---|
| `ROCKET_KACC` | `ROCKET_KACC=1` | fp16 [K-accumulation](GLOSSARY.md) on the NPU | +19% | Gemma-4-12B [HW sweep] |
| `ROCKET_REUSE` | Auto, with `ROCKET_KACC` on | CBUF operand reuse. Auto picks DATA_REUSE or WEIGHT_REUSE per shape, DATA_REUSE on a tie | DATA_REUSE +7% | Gemma-4-12B [HW sweep] |
| `ROCKET_MM_STREAM` | `ROCKET_MM_STREAM=1` | fp16 matmul plans that stream the weights through the CBUF, with far fewer tasks. `=0` selects the whole-tile plan | 1.07-1.17x at pp512 (928 MHz) | Llama-3.2-3B F16 and Q4_K_M, Ministral-3-3B F16, Qwen3.5-9B Q4_K_M, each with resident weights [HW sweep 2026-10-21] |
| `ROCKET_MM_ASYM` | `ROCKET_MM_ASYM=1` | Asymmetric tiling of the whole-tile plan, which runs only under `ROCKET_MM_STREAM=0` or a tile override | +6-9% on F16, on the whole-tile plan | Qwen3.5-9B, Gemma-4-12B [HW sweep] |
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

Precision (`ROCKET_INT4`, `ROCKET_INT8` or a `Q4_K_M` GGUF) reduces RAM, so a larger model fits,
and a quantized GGUF speeds [decode](GLOSSARY.md). Quantization does not speed prefill at this
operating point ([perf/not-mac-bound.md](perf/not-mac-bound.md)).

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

A quantized GGUF prefills on the same fp16 path as F16. Held resident, it prefills at
0.95-1.00x the same model's resident F16 GGUF on the three 3-4 B models measured both ways
[HW sweep 2026-08-29/31]. Its precision therefore sets memory use and decode speed, not prefill
speed. In the table, "fits" compares the fp16 footprint that the residency options need against
free RAM.

| Workload | Model | Set | Reason |
|---|---|---|---|
| Interactive chat, short prompts | Any | A `Q4_K_M` GGUF for decode, and `taskset 0xf0 -t 4`. No NPU options | Prefill is below the offload threshold, and the turn is decode-bound |
| Medium or long prefill | F16, fits | Defaults only (K-accumulation, weight streaming, operand reuse and attention offload on) | Already the fastest prefill |
| Agentic or RAG, repeated prefill | F16, fits ~2x | `ROCKET_F16_RESIDENT=auto` | Packs the weights once |
| Agentic or RAG, repeated prefill | F16, fits ~1.5x but not ~2x | `ROCKET_INT8=1 ROCKET_INT8_RESIDENT=1`, and `ROCKET_INT8_CACHE_MB=0` at 8 B and above | Holds the weights resident in half the bytes of resident fp16. Perplexity rises 0.7-2.8% ([Native int8 / int4 / bf16](#native-int8--int4--bf16)) |
| Medium or long prefill | Quantized GGUF, fp16 does not fit | `-b 2048 -ub 2048` | Amortizes the per-micro-batch dequantization. Two of eleven models lose, so measure it |
| Agentic, RAG or long prefill | Quantized GGUF, fp16 fits | `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, without `-b 2048 -ub 2048` | Faster than the combination with `-ub 2048` on all seven models measured both ways. Costs the full fp16 footprint |
| Medium or long prefill | MoE whose experts offload (gpt-oss, DeepSeek-V2-Lite) | `-b 2048 -ub 2048`. The expert offload needs no option | Default-on, and it decides its own placement ([MoE routed experts](#moe-routed-experts-on-the-npu)) |
| Any | MoE whose experts stay on the CPU (Qwen3-30B-A3B) | Defaults | `-ub 2048` is 0.908x on Qwen3-30B-A3B |
| Any | Too large for RAM at F16 | A `Q4_K_M` GGUF, or `ROCKET_INT4=1` from an F16 GGUF | Footprint, not speed ([Native int8 / int4 / bf16](#native-int8--int4--bf16)) |
| Batch or multi-stream | Any | One process per stream, each with `ROCKET_CPU_AFFINITY` | More concurrency raises aggregate throughput |
| ASR | Whisper | Keep `ROCKET_MIN_M` at 8 or above | The encoder offloads, and beam decode stays on the CPU |
| ASR, short utterances | CTC or transducer encoder | `ROCKET_MIN_M=16` | Below 2.6 s of audio, the default threshold leaves most encoder GEMMs on the CPU |
| Detection | `tflite-rocket` | `native_int8=1`. One process per camera, each with `ROCKET_CPU_AFFINITY` | Exact int8 convolutions, and throughput from a process pool |

### Interactive chat (short prompts, decode-bound)

Almost all of a short user turn's wall-clock time is decode. Decode is CPU-bound and
bandwidth-limited on both backends ([perf/decode-gemv.md](perf/decode-gemv.md)), and the NPU
barely touches the turn. A quantized GGUF's prefill is below its offload threshold and runs on
the CPU. An F16 prefill of 32 tokens or more offloads, but the turn stays decode-bound. Four
facts follow:

- Quantization is the option that helps, through decode speed. `Q4_K_M` decodes ~3x as fast
  as F16 (Ministral-3-8B: F16 1.60 -> Q4_K_M 4.72 t/s decode at `taskset 0xf0 -t 4`,
  llama.cpp b11539 [HW sweep 2026-11-01]) and fits a smaller board. Choose the quantization on
  decode speed and RAM, not on the NPU.
- Pinning the process to the A76 cores at four threads decodes 1.32-4.22x as fast as unpinned
  `-t 8` ([Host CPU pinning](#host-cpu-pinning)).
- Residency and `-ub` have no effect here, because there is no large prefill to amortize them
  over.
- A long system pre-prompt is the exception. A chat with a large preamble (an agent persona,
  tool schemas, retrieved context) has a large prefill every turn, and follows
  [Agentic / tool-use](#agentic--tool-use-large-repeated-prefill).

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
context window. The largest NPU prefill gains are here. At pp2048 the NPU prefills a dense F16
model at 4.2-5.8x the 8-thread CPU backend [HW sweep 2026-11-01, 928 MHz, llama.cpp b11539].
It prefills a dense `Q4_K_M` model at 3.2-4.7x held resident and at 3.8-4.0x at
`-b 2048 -ub 2048`, and a MoE model at 1.8-2.3x ([perf/benchmarks.md](perf/benchmarks.md)).

The settings are those of the decision table above. For an F16 model that serves repeated
prefills, add `ROCKET_F16_RESIDENT=auto` as RAM allows. [Attention offload](#attention-offload)
matters at these lengths, and is on by default. Leave it on.

### Batch / multi-stream throughput (offline, or many detectors)

Optimize aggregate throughput, not single-request latency. Run one process per stream, each
pinned to a distinct A76 core with `ROCKET_CPU_AFFINITY`. Single-stream latency is bound by
the host cube gather, not by the NPU. More concurrency therefore raises throughput, but a
faster submit does not. [Detection](#detection-frigate--tflite-tflite-rocket) gives the measured
pool.

### ASR: Whisper (whisper.cpp)

The NPU accelerates the encoder. The decoder is autoregressive (M=1 GEMV) and stays on the CPU.
The gain grows with the model, from tiny.en at 1.18x to large-v3 at 2.14x [HW sweep]. Larger
models gain more, because encoder matmul work grows ~d_model² and host packing ~linearly. No
options are needed beyond the prerequisites. Three points need attention:

- **Keep `ROCKET_MIN_M` at 8 or above.** The default beam-5 search in whisper.cpp presents M=5
  per decode step. A threshold of 4 offloads that small GEMV, and the transcription takes 40%
  longer end to end. The default 32 keeps beam decode on the CPU, and so does 16. A threshold of
  128 measures the same as 32 for whisper-small F16: 0.995-1.006x, transcript byte-identical
  [HW sweep, 2026-10-09].
- **Short utterances on a CTC or transducer encoder want `ROCKET_MIN_M=16`.** Parakeet TDT 0.6B
  through whisper.cpp's `parakeet-cli` runs 12.5 encoder rows a second of audio, so a 1.3-6 s
  voice command is 16-75 rows. The default threshold of 32 offloads every encoder GEMM from
  2.6 s up, and 24 of 265 below that. At 16, every encoder GEMM offloads from 1.3 s up, which
  runs 1.16-1.39x as fast as 32 below 2.6 s [HW sweep 2026-11-02, 928 MHz]. The
  encoder then ties the CPU at 1.3-1.8 s and runs 1.18-1.73x the CPU's speed at 2-6 s, freeing
  34-61% of the CPU [HW sweep 2026-09-30]. When another process shares the cores, add
  `OMP_WAIT_POLICY=PASSIVE`
  ([perf/asr-cpu-relief.md](perf/asr-cpu-relief.md#short-utterances-below-the-f16-threshold)).
- **The best transcription target is large-v3-turbo.** It pairs the full 32-layer encoder
  (NPU-accelerated, 2.12x) with a 4-layer decoder (~6x cheaper per step). The accelerated
  encoder is then most of the transcription's work.

### Detection: Frigate / TFLite (tflite-rocket)

The delegate's main setting is a delegate option, not an environment variable. Pass it through
`load_delegate` or `--option`. `native_int8=1` selects the exact-int8 convolution path, and is
the default in the Frigate `rocket.py` plugin. SSDLite-MobileDet COCO mAP is at CPU parity:
0.3325 against the CPU interpreter's 0.3301 over 500 images [HW sweep 2026-10-29, 928 MHz].

Throughput comes from a process pool, one process per camera, each pinned with
`ROCKET_CPU_AFFINITY`. Four processes run SSDLite-MobileDet at 3.33-3.51x one process's rate
[HW sweep 2026-10-29, 928 MHz]. A single stream runs 51.4 ms warm with the governor pinned, and
73.5 ms on `ondemand` [HW sweep 2026-10-29, 928 MHz]. Pin the governor for latency
([perf/benchmarks.md](perf/benchmarks.md) §"Single-stream latency").

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
  default 2 GB budget, `=1` places only part of it, which is a net loss against streaming. On
  Qwen3.5-9B `Q4_K_M` at pp2048 and `-ub 2048`, `=1` measured 19.3 t/s against 22.1 streamed
  [HW sweep 2026-07-01].
- **Gain:** 1.35-1.75x over streaming at the default `-ub`, against 1.00-1.66x combined with
  `-ub 2048`, on seven models of 1.8 to 11.9 B [HW sweep 2026-08-29/31]. At pp2048 a resident
  Q4_K_M GGUF of 1.8-9.0 B prefills at 3.2-4.2x the repacked 8-thread CPU backend
  [HW sweep 2026-11-02, 928 MHz]. On a model that does not fit, it falls back to streaming,
  correctly.
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

`gemma4-12b` is the only partly placed model measured. It places 83-87% of its weights alone
against 73-74% combined, because `-b 2048 -ub 2048` grows the compute buffers about 4x
([MODEL-NOTES.md](MODEL-NOTES.md#gemma-4-12b-it)).

On a partly placed model, raising `-ub` costs residency directly. To compare
configurations on such a model, read the `[f16-resident]` outcome line of each.

On `qwen35-08b` the two forms do not order. Over five passes at 928 MHz, residency alone
measures 132 t/s in three processes and 151-155 t/s in two. The combined form measures
140-142 t/s in all five [HW sweep 2026-11-02]. Either form serves that model.

### `ROCKET_F16_RESIDENT=auto`: F16 weights held resident

- **What it does:** the F16 counterpart of `ROCKET_QUANT_RESIDENT=auto`. It packs the all-K F16
  weights once and reuses them across micro-batches and turns.
- **When to use it:** an F16 model that fits ~2x in RAM (the resident tiles plus the source),
  used for repeated prefill.
- **RAM cost:** ~2x the fp16 model size.
- **Gain:** 1.14-1.16x at pp2048 on three 3-4 B F16 models with the streamed default
  [HW sweep 2026-11-02, 928 MHz]. Gemma-4-12B F16, 94% resident, gains 1.00-1.12x. A fusable
  projection group (Q\|K\|V, gate\|up) is placed as one combined weight. The gain
  therefore includes fusion's shared packA, which added ~+5.7% to residency alone on the build
  of 2026-07-15.
  [perf/weight-residency-fusion.md](perf/weight-residency-fusion.md) has the mechanism and the
  comparison.
- **Workers:** the worker count (`ROCKET_N_THREADS`, default 5) changes how many weights this
  option places, and not monotonically. On Gemma-4-12B F16 at pp512, six workers place 293
  weights, against 286 at five and 274 at seven [HW sweep 2026-09-02]. At pp2048 the memory
  reserve binds first, and six workers place 284 against five's 286 [HW sweep 2026-09-03].

> [!CAUTION]
> `ROCKET_PREPACK_MADVISE` reclaims the source weights for a prefill-only run, and breaks CPU
> decode. Do not use it for an interactive or serving run.

Prefetch hints do not shorten a cold GGUF load. The kernel's default readahead already streams a
cold mapping as minor faults, at ~780 MB/s, the NVMe's read rate. `MADV_WILLNEED` is neutral, and
`MADV_SEQUENTIAL` with `MADV_WILLNEED` takes 33% longer, with 16383 major faults [HW sweep
2026-06-30]. The sequential hint's drop-behind evicts pages that the weight scatter reads again
[hypothesis].

### Host CPU pinning

Run the host process on the four A76 cores with `taskset 0xf0 -t 4`. NPU prefill gains over
llama.cpp's unpinned default of 8 threads:

| Model | Prefill gain, `taskset 0xf0 -t 4` |
|---|---:|
| `Qwen3.5-0.8B` F16 | 1.13x |
| `Qwen3.5-9B` held quant-resident | 1.06x |
| `Gemma-4-12B` F16 streaming every weight | 1.06x |

**Conditions:** RK1, NPU clock ~816 MHz, governor `performance` [HW sweep 2026-09-01].

The NPU time is identical with and without pinning, so the gain is on the host. Residency does
not decide it: the streamed 12B gains as much as the resident 9B. `taskset 0xf0` at 8 threads
gains 1.05-1.10x on the same three. The 8-thread gain is roughly a third to a half of the share
of instructions the unpinned run retires on the little cluster.

> [!NOTE]
> Pin with `taskset`, not with `llama-bench`'s own affinity flags. Over the same eight cores,
> `-C 0xff --cpu-strict 1` measured 59.94 t/s on Qwen3.5-0.8B F16, against 125.97 t/s with no
> affinity set [HW sweep 2026-08-28/29].

No runtime flag replaces the pin. llama.cpp's `--poll 0` measures 1.013x the unpinned prefill on
Qwen3.5-0.8B F16 [HW sweep 2026-09-01]. In an OpenMP build of llama.cpp the flag reaches no
compiled code. libgomp's `OMP_WAIT_POLICY` and `GOMP_SPINCOUNT` do reach the thread barrier. But on
Gemma-4-12B F16 all of libgomp's work is at most 0.24% of prefill time [host-computed]
([perf/data/tuning-matrix.md](perf/data/tuning-matrix.md) §"Pinning to the A76s is worth
1.05-1.13x on prefill, and the gain tracks the little cluster's instruction share").

CPU decode gains too. At `taskset 0xf0 -t 4` it runs 1.32-4.22x the unpinned `-t 8` rate over
fifteen GGUFs, F16 and quantized, dense and MoE [HW sweep 2026-11-01, llama.cpp b11539,
`tg64`]. On Gemma-4-12B F16 it is 1.33x. A CPU-only prefill is the exception: it runs 4-10%
faster at 8 threads on the two models measured both ways [HW sweep 2026-11-01]. A CPU prefill
baseline therefore uses 8 threads.

The gains of pinning and of residency overlap, so do not multiply their published figures. Both
remove A76 pack work, and the second option applied finds less of it left. On `Qwen3.5-9B`,
residency combined with `-b 2048 -ub 2048` is worth 1.67x unpinned and 1.61x pinned. Both
options from stock measure 1.77x, not the 1.83x the two multiply to [HW sweep 2026-09-01/02,
RK1].

### `ROCKET_MIN_M` and `ROCKET_MIN_M_QUANT`

- **What it does:** sets the smallest prefill, in rows, that offloads to the NPU.
  `ROCKET_MIN_M` applies to an F16 model and defaults to 32. `ROCKET_MIN_M_QUANT` applies to a
  quantized model and defaults to 512. A smaller prefill runs on the CPU.
- **When to change it:** for ASR, as [ASR: Whisper](#asr-whisper-whispercpp) describes. Leave
  an LLM process's `ROCKET_MIN_M` at the default: Llama-3.2-3B F16 prefills at 0.78x the CPU at
  pp16 [HW sweep 2026-10-09]. `ROCKET_MIN_M_QUANT=256` gains 1.46-1.80x on Qwen3.5-9B Q4_K_M
  prompts of 256-384 rows, whose crossover sits between 128 and 256
  [HW sweep 2026-11-02, 928 MHz]
  ([perf/quant-prefill-microbatch.md](perf/quant-prefill-microbatch.md#routing-threshold-rocket_min_m_quant)).
- **Gain:** the F16 prompt-size crossover sits at 21-25 rows on the 0.8B, 3B and 8B models
  measured. The default `ROCKET_MIN_M` of 32 takes every gain above it. It measures 1.20-1.46x
  at pp32 against the faster CPU configuration [HW sweep, 2026-10-09]. At pp128 five F16 models
  of 1.8-9.0 B measure 2.7-3.4x the 8-thread CPU [HW sweep 2026-11-02, 928 MHz]. The CPU
  governor was `performance`.

### MoE routed experts on the NPU

- **What it does:** runs the mixture-of-experts FFNs (`MUL_MAT_ID`) on the NPU. It places
  experts per expert stack: the weight tensor of one `MUL_MAT_ID` op, which holds every
  expert's matrix. A quantized expert takes the native-int8 resident path
  (`ROCKET_MOE_NATIVE`), which ingests each expert once into int8 codes. That removes the
  per-micro-batch host dequantization that makes the fp16 expert path a loss. The fp16 path
  (`ROCKET_MOE=1` with `ROCKET_MOE_NATIVE=0`) re-dequantizes every expert every micro-batch,
  and is a net loss on both models measured. The default never takes it.
- **When to use it:** nothing to set. The offload is on by default, and decides per expert
  stack ([Eligibility per architecture](#eligibility-per-architecture)). `ROCKET_MOE=0` leaves
  the experts on the CPU.
- **RAM cost:** the GGUF source must coexist with the int8 codes, because MoE decode reads the
  active experts from it every token. The pre-flight therefore charges each whole stack its
  codes, scales and source against a budget of `MemAvailable` − 6 GiB. On a 31 GiB board that
  reserves 24529 MB for 63 of gpt-oss-20b's 72 stacks [HW sweep 2026-08-27]. The experts a
  prefill exercises then hold ~13.4 GB of int8 codes. The remaining 9 stacks stay on the CPU,
  which is a partial offload and not a loss.
- **Time:** a one-time expert ingest inside the first prefill, per `llama_context`. It takes
  ~21 s on gpt-oss-20b and ~18 s on DeepSeek-V2-Lite [HW sweep 2026-10-09, ~816 MHz]. Its NPU-BO
  pack runs ~1.3 GB/s, so it grows with the bytes held resident rather than with the expert
  count.
- **Gain:** gpt-oss-20b MXFP4 at `-b 2048 -ub 2048` measures 2.29x the CPU backend with its
  weight repack on at pp2048 [HW sweep 2026-11-01, 928 MHz, llama.cpp b11539]. At pp512 and the
  default `-ub 512` it measures 1.97-2.00x that CPU backend, and 2.17-2.21x the experts-on-CPU
  configuration [HW sweep 2026-11-02, 928 MHz]. That configuration reads 0.90-0.91x the CPU
  backend at pp512.
- **Workers:** set `ROCKET_N_THREADS=6`. On gpt-oss-20b at pp2048, six workers measure 1.022x
  the default five, and seven and eight measure no more [HW sweep 2026-09-02]. Every count
  places the same stacks.
- **Micro-batch:** where the experts offload, `-b 2048 -ub 2048` measures 1.25x the default
  `-ub` on gpt-oss-20b and 1.32x on DeepSeek-V2-Lite at pp2048 [HW sweep 2026-08-29]. Where the
  backend leaves the experts on the CPU, keep the default `-ub`: on Qwen3-30B-A3B `-ub 2048` is
  0.908x. The teardown split that **Confirm** names is printed only where the experts offload.
- **Confirm:** `ROCKET_LOG_STDERR=1` prints the resident and streamed expert split at teardown.
  Under the default the split is 100% resident. A nonzero streamed count means a limit the
  pre-flight could not see ahead of the ingest, most likely an exhausted NPU IOVA window.

DeepSeek-V2-Lite keeps its experts on the CPU at pp512, below the work gate. The dense
offload alone then measures 1.11x the repacked CPU backend, and `ROCKET_MOE=0` reads the same
[HW sweep 2026-11-02, 928 MHz].

> [!NOTE]
> An expert that `llama-bench`'s warm-up never selected ingests inside a timed repetition. A
> MoE prefill figure from `llama-bench` can therefore include part of the ingest, as it did at
> pp512 on gpt-oss-20b [HW sweep 2026-10-09, 2026-11-02]. Its repetitions rise from 31.3 to
> 34.1 t/s within one run, so its pp512 ratio is a lower bound.

#### Residency pre-flight

A residency pre-flight makes the sign of the offload independent of the host's RAM. A stack's
resident cost is known from its tensor alone. The placement check therefore reserves the whole
stack before the first ingest, and leaves on the CPU what it cannot reserve. A declined stack
costs nothing at the backend boundary, because the backend's buffer type is the CPU buffer
type. The default is therefore bounded below by the experts-on-CPU baseline at any RAM size.

#### Eligibility per architecture

Eligibility is per architecture, not universal. The default claims an expert op only where four
conditions hold:

- The stack is GGUF-quantized.
- The micro-batch has at least `ROCKET_MOE_MIN_TOKENS` tokens (512), a minimum that also holds
  under `ROCKET_MOE=1`.
- Each expert's GEMM clears two size thresholds.
- The pre-flight then reserves the whole stack.

One size threshold is the minimum tile height of the per-expert rows, 64 by default
(`ROCKET_MOE_M_BUCKET`). The other is the work in one dispatch (`ROCKET_MOE_MIN_WORK`), in
mega-MACs of `M_e · K · N`, where `M_e` = `n_tokens · n_used / n_expert`.

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
([perf/moe-placement.md](perf/moe-placement.md) §"The accept boundary"). Lower the threshold only for a workload that prefills that much.

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
[HW sweep 2026-08-28, RK1, ~816 MHz]. The plateau is 18000-21000 MB
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
setting changes that. At 8 GB and below, each of the three models' GGUF alone exceeds RAM.

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
  int4 and int8 are char-identical to fp16 greedy output, and bf16 is token-identical. At
  `512×3840×4096`, int8 with streamed weights and int4 run 1.14-1.20x fp16's streamed matmul.
  Their MAC rates are 2x and 4x, but the matmul is DMA/dispatch-bound
  ([perf/not-mac-bound.md](perf/not-mac-bound.md)). In a model, at equal residency, fewer bits
  do not raise prefill speed.
- **When to use it:** to make a model fit that would not at F16, or for bf16's fp32 range.
  The int8 path also holds a model's weights resident in half the memory resident fp16 takes.
  Resident int8 in-model prefill runs at 0.95-0.99x resident fp16 and 1.11-1.16x the default
  fp16 path, on two 3 B models at 928 MHz. On Ministral-3-8B it reaches resident fp16's rate [HW sweep 2026-11-07]
  ([perf/not-mac-bound.md](perf/not-mac-bound.md) §"In-model int8 and int4 prefill"). int4
  measured ~0.53x fp16 in the whole-tile program, at ~816 MHz.
- **Budget:** at 8 B and above, set `ROCKET_INT8_CACHE_MB=0`, which removes the budget. The
  default 4096 MB holds 58-62% of an 8-9 B model's int8 weights, and the rest take the fp16
  path. The partial placement costs 3-5% of prefill on Ministral-3-8B and 1-3% on Qwen3.5-9B
  [HW sweep 2026-11-07, 928 MHz].
- **Accuracy:** resident int8 raises wikitext perplexity 0.7-2.8% over fp16 on three models of
  3-9 B [HW sweep 2026-11-06/07]. The Hadamard rotation lowers Qwen3.5-9B's, but not
  Ministral-3-8B's ([MODEL-NOTES.md](MODEL-NOTES.md#qwen35-9b-q4_k_m)).
- **Disk cost:** the native int4 and int8 paths quantize from a full-precision (F16) GGUF and
  require Hadamard rotation. They do not take a pre-quantized `Q4` file, so they use the disk
  of the larger F16 GGUF to save runtime RAM. With only a `Q4_K` GGUF, use the GGUF-quant path
  instead: `ROCKET_QUANT_RESIDENT=auto` where the fp16 form fits RAM, `-ub 2048` where it does
  not. That path is a different mechanism, and the one with a speed gain.
- **Recommendation:** for RAM alone, a `Q4_K_M` GGUF is simpler and also speeds decode. Run it
  as [Options by workload](#options-by-workload) gives. Native int4 is for the ¼ footprint from
  an F16 source.

A bf16 GGUF offloads without `ROCKET_BF16=1`, which selects the exact bf16 path that writes an
fp32 result. Without it, each micro-batch converts the bf16 weights to fp16 and runs the fp16
path. That conversion is faithful: Qwen3.5-0.8B's perplexity is 20.4427 from the BF16 GGUF and
20.4425 from the F16 one. It is also repeated on every micro-batch. At pp2048 the BF16 GGUF
therefore prefills at 0.56-0.58x the F16 GGUF on Qwen3.5-0.8B and 9B [HW sweep 2026-07-02,
~816 MHz, whole-tile fp16 plan]. Convert a bf16 GGUF to F16 once, with
`llama-quantize in.gguf out.gguf f16`.

### Attention offload

- **What it does:** `ROCKET_FLASH_ATTN` (on) offloads masked prefill attention at `n_kv >= 1024`
  keys and 16 or more query tokens. It is bit-faithful for the attention it implements
  (`softmax(scale·QKᵀ+mask)·V`). The handler declines an op that has attention sinks (gpt-oss,
  some others). It has no sink term, so accepting the op would silently compute a different
  softmax. An unmasked op, an encoder's self-attention, stays on the CPU unless
  `ROCKET_FLASH_ATTN_UNMASKED=1`.
- **When it matters:** long contexts. It measures parity at <=1K, 1.50x at 8K and 1.25x at 16K
  on Gemma-4-12B F16 [HW sweep 2026-06]. Below ~2K it neither gains nor loses, and the default
  threshold handles that.
- **Diagnosis:** it rarely needs changing. If a model's NPU curve collapses past ~1K context
  but is normal below it, suspect an attention with sinks. The handler must decline such an
  op, and it does.
- **Related settings:** `ROCKET_FLASH_ATTN_MIN_KV` (1024) sets the fewest keys an op needs to
  offload, and `ROCKET_FLASH_ATTN_MIN_T` (16) the fewest query tokens.
  `ROCKET_FA_CHAIN` (on) batches a worker's per-head score matmuls into one submit and its value matmuls into a second.
  `ROCKET_FA_ROWMAJOR` (on) hands the scores and P between its two jobs as rows, with no
  layout pass on the host. `ROCKET_FLASH_ATTN_NO_CTX=1` forces the per-call path, for
  comparison against the persistent context.

The gain figures are from a build without the flash attention's host-pass changes of
2026-09-27, 2026-10-10 and 2026-10-11. The gain on a build with them is not characterized.

## Appendix: measurements

### Measurement coverage

The option defaults do not depend on the model, so the recommendations hold. The per-model
default-against-tuned comparison is measured on a subset of models. Where a recommendation
implies a per-model figure that [perf/benchmarks.md](perf/benchmarks.md) does
not list, treat it as a projection, not a measurement.

| Option | Measured on | Not characterized |
|---|---|---|
| NPU clock at the 600 MHz request (~816 MHz) | Gemma-4-12B [HW sweep] | |
| NPU clock at the 900 MHz request (928 MHz) | The fp16 matmul [HW sweep 2026-10-19] | The clock's own effect on prefill on the current library |
| `ROCKET_MM_STREAM` | Four models with resident weights at pp512, one also at pp2048 [HW sweep 2026-10-21] | Non-resident weights, re-packed on every call. The int8, int4 and bf16 planners, which do not stream |
| `ROCKET_MM_ASYM`, `ROCKET_KACC` and DATA_REUSE | A few models, mostly Gemma-4-12B and Qwen3.5 | Every other model, which takes the default without a measurement |
| `-b 2048 -ub 2048` | Eleven models, 0.75-30.53 B [HW sweep 2026-08-29/30] | |
| `ROCKET_QUANT_RESIDENT` | Combined with `-ub 2048`, nine quantized models of 0.75-11.91 B [HW sweep 2026-08-28..31]. Alone and combined, seven models of 1.8 to 11.9 B | How the gain falls with the placed fraction. `gemma4-12b` is the only partly placed model on the board, at 83-87% alone against 73-74% combined, a two-point contrast on one model rather than a curve |
| `ROCKET_F16_RESIDENT` | Three 3-4 B F16 models with CBUF weight streaming on [HW sweep 2026-11-02], and Gemma-4-12B F16 | Models above 12 B |
| MoE expert offload | gpt-oss-20b and DeepSeek-V2-Lite, and Qwen3-30B-A3B, which it declines [HW sweep] | |
| `ROCKET_MOE` size thresholds | Fitted on gpt-oss-20b and DeepSeek-V2-Lite, which disagree about which shapes are eligible. Qwen3-30B-A3B, a third architecture, confirms the threshold's sign but not its position, at 201 MMAC against the 340 default | The threshold's position on another architecture |
| `ROCKET_MOE_CACHE_MB` | Two budget series, gpt-oss-20b and Qwen3-30B-A3B, which invert. Both on the same 31 GiB board | The board-size axis. The small-board table is arithmetic |
| SmolVLM2 resident `rocket_siglip_encoder` vision path | No end-to-end benchmark. The generic clip drop-in is the only measured multimodal-vision figure (1.19x) | The resident path end to end |
| `ROCKET_MIN_M` | F16 models of 0.8B, 3B and 8B | A model smaller than 0.8B, or one with much narrower matmuls |

### Prerequisites

The 33x comparison of the phase breakdown against wall-clock time is a per-process readout
measurement [HW sweep 2026-09-01, RK1].

### Host CPU pinning

Each prefill pinning gain is from rotated interleaved passes. The overlap with residency on
`Qwen3.5-9B` is from six rotated passes [HW sweep 2026-09-01/02, RK1]. The decode figures are
one pass of `tg64 -r 3` per configuration, each from a memory reset, at 928 MHz
([perf/data/tuning-matrix.md](perf/data/tuning-matrix.md)). The CPU-only prefill comparison is
one process per thread count at pp512 [HW sweep 2026-11-01]. It measured 1.04x on
SmolVLM2-2.2B F16 (30.94 against 29.63 t/s) and 1.10x on Ministral-3-8B `Q4_K_M` (10.29 against
9.34 t/s), from `ggml-rocket`'s `MEASUREMENTS.md`.

The `--poll 0` figure is six rotated passes, with ratios paired within a pass, 0.44 standard
errors above 1.00. In the same passes `taskset 0xf0 -t 4` measured 1.165x. The A55 instruction
count moved 0.35%, and context switches 0.05% [HW sweep 2026-09-01, RK1]. The
`-C 0xff --cpu-strict 1` reading is one process per configuration.

### `ROCKET_F16_RESIDENT=auto`

The placement counts at pp512 are byte-identical across independent processes [HW sweep
2026-09-02, RK1]. Five readings are at five workers, three at six and two at seven. The pp2048
counts are two processes per worker count, `-p 2048 -n 0 -r 1` [HW sweep 2026-09-03]. At pp2048,
eight workers measured 0.987x five's prefill over three rotated passes, and placed 281 weights
against 286 [HW sweep 2026-09-01].

The default of five workers comes from a dense fp16 sweep on Gemma-4-12B F16 at pp1024 and
`-ub 1024`. Three to six workers measured 7.15, 8.06, 8.24 and 8.27 t/s, with the NPU at its
198 MHz boot clock [HW sweep 2026-06-14]. That operating point is superseded, and the sweep is not
characterized at 928 MHz.

### Cold GGUF load

The prefetch figures are a microbenchmark on the RK1's NVMe [HW sweep 2026-06-30]. It reads a
2 GB file through a cold mapping in the weight scatter's tiled order, after a `drop_caches`:

| Advice | Wall-clock time | Major faults |
|---|---:|---:|
| None | 2.72-2.75 s | 1 |
| `MADV_WILLNEED` | 2.78-2.82 s | 1 |
| `MADV_SEQUENTIAL` | 2.73 s | 1 |
| A sequential warm-up pass | 2.77-2.79 s | 1 |
| `MADV_SEQUENTIAL` and `MADV_WILLNEED` | 3.64 s | 16383 |

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

The blanket `=1` comparison, 19.3 against 22.1 t/s, is one process per configuration
[HW sweep 2026-07-01, RK1].

### MoE routed experts on the NPU

#### Worker count

The worker-count ratios on gpt-oss-20b are paired within each of three rotated passes, at
`-b 2048 -ub 2048` [HW sweep 2026-09-02, RK1]. An A/B with the two configurations adjacent
measured eight workers at 1.027x five.

#### Budget series on gpt-oss-20b

Every step is n=1: one `-r 3` process per configuration, behind a `drop_caches`, with 0
streamed in every configuration [HW sweep 2026-08-27, ~816 MHz].

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

At `-ub` 4096 [HW sweep 2026-08-28, RK1, ~816 MHz]:

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
| 2026-08-29 | `ROCKET_F16_RESIDENT=auto`, three 3-4 B F16 models, pp2048 | 1.078-1.084x | ~816 MHz, the whole-tile fp16 plan. The 2026-11-02 series at 928 MHz with CBUF weight streaming measures 1.14-1.16x |

Both `-ub` readings are lower in the 2026-08-29/30 sweep by the same 0.68 factor
(2.098 -> 1.424, 2.250 -> 1.532). Between the two sweeps, three default-on host-cost cuts
raised the `-ub 512` baseline ~1.47x more than the `-ub 2048` configuration on each model. On
the 9B, the cuts raised its `-ub 512` baseline 2.33x against the `-ub 2048` configuration's
1.59x.
