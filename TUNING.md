# Tuning the rocket NPU stack

This guide says which knobs to set for a given workload, and why. The stack ships its
performance defaults on, so most of the win needs no configuration. The choices that
remain are a handful of opt-ins. Their value depends on the model, the workload and the
RAM you have.

Each project's `API.md` is the per-flag reference (the flag -> meaning tables), and
[perf/](perf/) holds the evidence behind each number. This guide is the decision layer for
both. For per-model behavior and sampling, see [MODEL-NOTES.md](MODEL-NOTES.md). For the
raw benchmarks, see [perf/benchmarks.md](perf/benchmarks.md).

All figures are warm medians on the RK3588 at 600 MHz, the same model NPU-vs-CPU
(8-thread) [HW sweep]. Where a recipe inherits a delta that was measured on one model, the
text says so. The flag defaults are model-independent by construction. The per-model
number is not always separately measured (see
[What is not yet measured](#what-is-not-yet-measured)).

## Summary

The defaults already carry most of the win. On a correctly set-up board you set almost
nothing.

These levers are on by default. Leave them on, and set one to `0` only to A/B it:

- `ROCKET_KACC`: fp16 K-accumulation, +19%
- `ROCKET_REUSE=2`: CBUF DATA_REUSE, +7%
- `ROCKET_MM_ASYM`: asymmetric tiling, +6-9% on F16
- `ROCKET_FLASH_ATTN`: attention offload, wins above ~2K context
- `ROCKET_DEQUANT_THREADS`: threaded quant dequant, +19-33%

Four opt-ins change the outcome, each gated on your workload and RAM:

| Opt-in | Set it for | What it buys |
|---|---|---|
| `-b 2048 -ub 2048` (a llama.cpp flag, not a `ROCKET_*` one) | A quantized GGUF whose fp16 image does not fit resident | 0.91-1.53x over the llama.cpp `-ub 512` default, model-dependent |
| `ROCKET_QUANT_RESIDENT=auto` | A quantized GGUF, if the model's fp16 size fits RAM, wholly or partly | Lifts quant prefill to fp16 parity or past it |
| `ROCKET_F16_RESIDENT=auto` | An F16 model that fits ~2x in RAM | Single-digit-percent gain |
| Nothing | A mixture-of-experts model | ~2.4x the CPU at pp2048 on gpt-oss-20b |

Eleven models are measured for `-b 2048 -ub 2048`. The gain is larger on the big dense
models but not predictable from parameter count, and negative on two (`smolvlm2` 0.941x,
`qwen3-30b-a3b` 0.908x). It is still the biggest of the flag levers on a large quant
model, and nearly free. The larger micro-batch grows the activation and compute buffers
~4x (512->2048). That is negligible against the weights but not zero, so check it on a
RAM-tight, swap-less board.

**`-b 2048 -ub 2048` is for the case where `ROCKET_QUANT_RESIDENT=auto` is unavailable.**
On every one of seven models measured both ways, residency at the default `-ub` beats
residency stacked on `-ub 2048`. So the flag is for a model whose fp16 image does not fit
resident.

**Run `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, and do not stack
`-b 2048 -ub 2048` on it.** Unstacked reads 1.35-1.75x on seven models from 1.8 to
11.9 B, against 1.00-1.66x for the stacked form. No measured model prefers stacking. On a
model that places only part of its weights, the unstacked form also buys more residency.
`-ub 2048` spends that RAM on compute buffers instead.

A mixture-of-experts model needs no flag. The routed-expert offload is on by default since
2026-08-27 and gates itself. It claims a stack only where it can reserve the whole stack
up front and the per-expert GEMM pays for its own dispatch. `ROCKET_MOE=1` overrides both
checks. It is faster where the stack nearly fits and **unsafe where it does not**, so it
is an A/B arm, not a recommendation.

Precision (`ROCKET_INT4`, `ROCKET_INT8` or a `Q4_K_M` GGUF) is a RAM and model-fit lever,
not a speed lever. Quantization does not speed prefill at this operating point
([perf/not-mac-bound.md](perf/not-mac-bound.md)). Choose it to make a model fit or to
speed decode, never to speed prefill.

If you read nothing else, raise the clock. Then run a quantized GGUF with
`ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, where the RAM math (below) says it fits.
Where it does not fit, run it at `-b 2048 -ub 2048`.

## The operating-point floor

Do these once. Every number in this stack assumes them:

- **Raise the clock to 600 MHz.** The NPU boots pinned at 200 MHz (one-fifth speed). The
  `patches/rocket` clock patch takes it to 600 MHz, which is ~1.43x: Gemma-4-12B pp2048
  goes 7.98 -> 11.40 t/s [HW sweep]. Load the module with `rocket_npu_clk_hz=600000000`.
  A 900 MHz clock gives no gain here, and **pinning it hard-locks the box**. See
  [perf/clock.md](perf/clock.md).
- **Run under `sudo -E`.** Plain `sudo` strips the environment. That drops both the
  `/dev/accel` privilege context and every `ROCKET_*` and `GGML_BACKEND_PATH` knob. `-E`
  keeps them.
- **Set `GGML_BACKEND_PATH` to the absolute path** of `libggml-rocket.so`. A wrong path
  **silently falls back to CPU-only**: one `failed to load` line, then no NPU.
- **Compare warm runs.** The clock parks at idle and rides up under load, so a cold run
  reads ~15% low. Discard the first run and compare warm ones. `llama-bench` discards it
  for you. Use `-r 3`.
- **Confirm the config engaged.** Run once with `ROCKET_LOG_STDERR=1` and look for the
  mode/budget line. `llama-bench` hides all rocket lines without it. Add
  `ROCKET_MM_PROFILE=1` for the phase breakdown. A mis-set knob is otherwise invisible.
- **Read the phase breakdown as an upper bound.** It says which phases ran, not what they
  cost in wall. Its buckets are thread intervals that count contention, and inside a model
  they sum to several times the wall. The wall-derived value for one term came out 33x
  lower [HW sweep, readout, 2026-09-01, RK1].

### Host CPU pinning

For a prefill-heavy run, pin the host process to the big cluster with `taskset 0xf0`.
Prefill gains [HW sweep 2026-09-01, RK1, 600 MHz, governor `performance`, rotated
interleaved passes]:

| Model | Prefill gain, pinned |
|---|---:|
| `Qwen3.5-0.8B` F16 | 1.10-1.13x |
| `Qwen3.5-9B` held quant-resident | 1.06x |
| `Gemma-4-12B` F16 streaming every weight | 1.05-1.06x |

The NPU half is identical in every arm, so the gain is host-side, and residency does not
gate it. Adding `-t 4` gains a little more on the smaller models and nothing on the
larger. The gain tracks the share of instructions the unpinned run retires on the little
cluster, at roughly a third to a half of that share. **Leave a decode-heavy run
unpinned.** F16 decode is LPDDR-bandwidth-bound, and pinning costs it 34% on a 12B F16
model.

Pinning and residency overlap, so **do not multiply their published numbers**. Both remove
A76 pack work, and the second lever applied finds less of it left. On `Qwen3.5-9B` the
residency knob stacked on `-b 2048 -ub 2048` is worth 1.67x unpinned and 1.61x once the
run is pinned. Taking both from stock reads 1.77x, not the 1.83x the two multiply to
[HW sweep 2026-09-01/02, RK1, six rotated passes].

## Workload variables

The recipe is a function of four things: model class, precision, prompt profile and
resource budget. Read your workload against each before you pick flags.

### Model class

Each class routes different work to the NPU:

- Dense LLM
- Mixture-of-experts (MoE)
- Multimodal (a vision encoder plus an LLM)
- ASR (Whisper)
- Detection

### Precision

The precision is F16, a quantized GGUF (`Q4_K_M`, `Q8_0`, `IQ4_XS` or MXFP4), or a
native-int in-model path. It sets both the RAM footprint and which residency lever
applies.

### Prompt profile

The prompt profile is how many tokens hit prefill per turn. It decides whether the NPU is
on the critical path at all:

| Profile | Prefill per turn | What runs |
|---|---|---|
| Short | A chatty turn, tens of tokens | An F16 prefill of 32 tokens or more offloads (`ROCKET_MIN_M`), and a quantized one stays on the CPU below 512 (`ROCKET_MIN_M_QUANT`). Either way the turn is decode-bound, so the NPU does little for it |
| Medium | 512-2048: a RAG chunk, a document paragraph, an agentic tool result | NPU prefill engages and wins |
| Long or batched | >2048, or a large system pre-prompt re-processed every turn | The NPU's best case, and where residency and attention offload pay the most |

A large system pre-prompt turns a chat workload into a prefill-bound one. An agent with a
several-thousand-token tool or persona preamble hits the NPU hard every turn, even where
the user's message is short.

### Resource budget

The budget is free RAM and disk. Every residency opt-in trades RAM for speed. The
native-int paths trade disk for runtime RAM, because they require a full-precision GGUF as
their source. The opt-in entries give the RAM math per lever.

## Use-case recipes

### Interactive chat (short prompts, decode-bound)

A short user turn spends almost all its wall time in decode. Decode is CPU-bound and
bandwidth-limited on both backends ([perf/decode-gemv.md](perf/decode-gemv.md)). The NPU
barely touches the turn, because the prefill is below the offload floor and runs on the
CPU. Three facts follow:

- The lever is quantization, for the stream. `Q4_K_M` decodes ~2.8x faster than F16
  (Ministral-3-8B: F16 1.4 -> Q4_K_M 3.8 t/s decode [HW sweep]) and fits a smaller board.
  Pick the quant on decode speed and RAM, not on the NPU.
- Residency and `-ub` do nothing here, because there is no large prefill to amortize them
  over.
- The exception is a long system pre-prompt. If the chat carries a big preamble (agent
  persona, tool schemas, retrieved context), the per-turn prefill is large. The workload
  is then the agentic case in the next section.

### Agentic / tool-use (large, repeated prefill)

An agent re-processes a growing context (system prompt, tool schemas, prior steps, tool
outputs) every turn. That is a large prefill on every step, which is the NPU's job. The
fixed setup cost of residency amortizes across turns. The flags by model:

- **Quantized, fp16 footprint fits RAM:** `ROCKET_QUANT_RESIDENT=auto` alone, at the
  default `-ub` (see the RAM math). Residency pays back best here, because the same weights
  serve many turns. Stacking `-ub 2048` on it loses 21-34% of the win under 4 B, and 3.5-25%
  from 8 B up.
- **Quantized, fp16 footprint does not fit:** `-b 2048 -ub 2048`.
- **F16 with RAM to spare:** `ROCKET_F16_RESIDENT=auto`.
- **Attention offload:** nothing to set. It is automatic and pays once the context passes
  ~2K ([perf/attention-offload-crossover.md](perf/attention-offload-crossover.md)).
- **MoE model:** nothing to set. The expert offload is on by default and declines itself
  where it would not pay (below).

### RAG / long-context / document processing (large one-shot prefill)

This workload is a single big prefill (a retrieved passage, a pasted document, a filled
context window). The largest NPU prefill wins land here. The CPU multiple grows with
prefill length, up to Qwen3.6-27B's 4.4x at pp2048 [HW sweep].

For a quantized GGUF whose fp16 footprint fits RAM, use `ROCKET_QUANT_RESIDENT=auto` at the
default `-ub`. Where it does not fit, use `-b 2048 -ub 2048`. For an F16 model, add
`ROCKET_F16_RESIDENT=auto` as RAM allows.

Attention offload matters at these lengths. Gemma-4-12B F16 reads 1.50x at 8K context and
1.25x at 16K [HW sweep]. It is on by default, so leave it on.

### Batch / multi-stream throughput (offline, or many detectors)

Optimize aggregate throughput, not single-request latency. Run multiple processes, one
per stream, each pinned to a distinct A76 core via `ROCKET_CPU_AFFINITY`. The detection
pool reaches 2.98x at four streams (the tflite-rocket delegate). Single-stream latency is
host cube-gather-bound, not NPU-bound. So the lever is more concurrency, not a faster
submit.

### ASR: Whisper (whisper.cpp)

The NPU accelerates the encoder. The decoder is autoregressive (M=1 GEMV) and stays on
the CPU. The win grows with the model, from tiny.en at 1.18x to large-v3 at 2.14x
[HW sweep]. Larger models benefit more (encoder matmul work ~d_model², host packing
~linear). No opt-in flags are needed beyond the operating-point floor and
`ROCKET_CPU_AFFINITY`. Three points need attention:

- **Keep `ROCKET_MIN_M` at 8 or above.** The default beam-5 search in whisper.cpp presents
  M=5 per decode step. A floor of 4 offloads that tiny GEMV, at a net 1.40x loss
  end-to-end. The default 32 keeps beam decode on the CPU, and so does 16. Against the old
  default of 128 it changes nothing for whisper-small F16: 0.995-1.006x over three passes,
  transcript byte-identical [HW sweep, 2026-10-09].
- **Short utterances on a CTC or transducer encoder want `ROCKET_MIN_M=16`.** Parakeet
  runs 12.5 encoder rows a second of audio, so a 1-5 s voice command sits under the
  default floor and nothing offloads. At 16, Parakeet TDT 0.6B through whisper.cpp's
  `parakeet-cli` ties the CPU at 1.3-1.8 s and runs 1.18-1.73x faster at 2-6 s
  [HW sweep]. It frees 34-61% of the CPU. When another process shares the cores, add
  `OMP_WAIT_POLICY=PASSIVE`. See
  [perf/asr-cpu-relief.md](perf/asr-cpu-relief.md#short-utterances-below-the-f16-floor).
- **Best real-transcription target: large-v3-turbo.** It pairs the full 32-layer encoder
  (NPU-accelerated, 2.12x) with a 4-layer decoder (~6x cheaper per step). So the
  accelerated encoder carries most of the transcription.

### Detection: Frigate / TFLite (tflite-rocket)

The delegate's key knob is a delegate option, not an env var. Pass it via `load_delegate`
or `--option`. `native_int8=1` selects the exact-int8 conv path, which is the default in
the Frigate `rocket.py` plugin. MobileDet COCO mAP is CPU-parity (0.3321 vs 0.3318
[HW sweep]).

Throughput comes from a process pool, one process per camera, each pinned with
`ROCKET_CPU_AFFINITY` (3.20 -> 9.55 detection_fps, P=1->4). A single MobileDet stream
runs 56 ms warm with the governor pinned and 76 ms on `ondemand`. So pin the governor for
latency ([perf/benchmarks.md](perf/benchmarks.md) §"Single-stream latency").

## LLM configuration by scenario

This table is the dense-LLM recipe. Prefill is fastest on F16. A quantized GGUF is a RAM
play that also speeds decode. Sizes are the fp16 footprint the residency levers need.

| You are running | Prompt profile | Recommended flags | Why |
|---|---|---|---|
| F16, fits RAM | medium / long | defaults only (KACC/REUSE/ASYM/FA on) | Already the fastest prefill. Nothing to add |
| F16, fits ~2x RAM | agentic / RAG (repeated) | `+ ROCKET_F16_RESIDENT=auto` | Pack the weights once: ~+6% pp2048, +9% pp512 on a 3B F16 model [HW sweep] |
| Quantized GGUF, fp16 does not fit RAM | medium / long | `-b 2048 -ub 2048` | 0.91-1.53x over the `-ub 512` default. Dense: 1.18-1.53x above 8 B, 0.94-1.13x under 4 B. MoE: read its own row, 0.91-1.32x. Measure it: two of eleven lose |
| Quantized GGUF >= ~8 B, fp16 fits RAM | agentic / RAG | `ROCKET_QUANT_RESIDENT=auto` at the default `-ub` | 1.37-1.75x on the three measured models, against 1.32-1.66x stacked. Costs the full fp16 footprint |
| Quantized GGUF < 4 B, fp16 fits RAM | agentic / RAG | `ROCKET_QUANT_RESIDENT=auto` at the default `-ub` | 1.35-1.51x on all four measured models, against 1.00-1.21x stacked. The class carries a real non-dequant `-ub 2048` loss, so **do not stack** |
| Any, short prompts | interactive chat | pick `Q4_K_M` for decode, no NPU flags | Prefill is below the offload floor, and the turn is decode-bound |
| MoE (gpt-oss, DeepSeek, …) | medium / long | `-b 2048 -ub 2048`. The expert offload needs no flag | ~2.4x the CPU at pp2048 on gpt-oss-20b, default-on and self-gating. Eligibility is per architecture, not universal: gpt-oss offloads from `-ub 512` up, DeepSeek-V2-Lite only past ~1250 tokens in a micro-batch |
| Model too big for RAM at F16 | any | a `Q4_K_M` GGUF, or `ROCKET_INT4=1` from an F16 GGUF | Footprint, not speed (see below) |

## The opt-ins in detail

Each entry gives:

- What the opt-in does
- When it helps
- The RAM or disk it costs
- The measured delta
- How to confirm that it engaged

### `-b 2048 -ub 2048` (llama.cpp)

- **What it does:** a quantized GGUF re-dequantizes to fp16 per micro-batch. The llama.cpp
  default `-ub 512` re-decodes the whole model every 512 rows, and `-ub 2048` spreads that
  fixed cost.
- **When:** any quantized prefill of >= ~512 rows. It is irrelevant to F16 (no dequant)
  and to short prompts (one micro-batch).
- **Cost:** the larger micro-batch grows the activation and compute buffers ~4x
  (512->2048). That is negligible against the weights, but not zero. On a RAM-tight board
  (no swap on this one), confirm that it still fits.
- **Trap:** never compare a `-ub 512` number against a `-ub 2048` one. A whole class of
  phantom "regressions" is this mistake. See
  [perf/quant-prefill-microbatch.md](perf/quant-prefill-microbatch.md).

#### Measured gain

The flag reads 0.91-1.53x, model-dependent, over eleven models 0.75-30.53 B
[HW sweep 2026-08-29/30, rotated passes]. The split is at ~4 B: models under 4 B read
0.94-1.13x, with `smolvlm2` (1.81 B) a loss at 0.941x. Models above 8 B read 1.18-1.53x,
and within that group parameter count does not order the gain:

| Parameters | `-ub 2048` over `-ub 512` |
|---:|---:|
| 8.49 B | 1.176 |
| 8.95 B | 1.424 |
| 11.91 B | 1.257 |
| 14.66 B | 1.308 |
| 27.32 B | 1.532 |

That split holds for dense models only. `qwen3-30b-a3b` is 30.53 B and reads 0.908x, the
matrix's second loss, because only ~3 B are active per token. Read a MoE model from its
own row, not from its parameter count.

The 2026-06-28 readings of the 9B and the 27B are ~2.1x and 2.25x. Both read lower in
this sweep, by the same 0.68 factor (2.098 -> 1.424, 2.250 -> 1.532). Between the two
readings, three default-on host-cost cuts raised the `-ub 512` baseline ~1.47x more than
the `-ub 2048` arm on each model. On the 9B, the 2026-06-28 reading of ~2.1x is
8.2 -> 17.2 t/s. The cuts raised its `-ub 512` baseline 2.33x against the `-ub 2048`
arm's 1.59x. So measure the flag on your model instead of assuming the class.

`Q4_K`, `IQ4_XS` and `Q8_0` converge within noise, so quant type does not change NPU
prefill throughput.

#### Dequant amortization and the non-dequant residue

The gain splits into dequant amortization and a non-dequant residue, and the split is per
class [HW sweep 2026-08-31, dqc mechanism control, six units]. On the 9B, 1.378 of the
1.430 is dequant amortization and the non-dequant residue is 1.038x. The documented
mechanism is ~90% of the lever there. On every measured sub-4B model the residue is a
real loss that the dequant win only masks:

| Model | Non-dequant residue |
|---|---:|
| `smolvlm2` | 0.789x |
| `ministral3-3b` | 0.835x |
| `llama32-3b` | 0.863x |
| `phi4mini` | 0.864x |

The 0.8B straddles, unresolvable in its class's per-process spread. About half of
`smolvlm2`'s loss is the output de-tile at [2048,N]: the driver `read` bucket grows 64% at
identical output bytes. The rest sits in `wait` [hypothesis, not isolated]. So for the
sub-4B class the right lever is removing the dequant at the default `-ub` (residency,
unstacked), not raising `-ub`.

#### Host load

The flag is load-sensitive, in its favor. Measured behind a single busy core, the lever
reads 1.48x, 1.53x and 1.12x on the same three units. The tenant steals the dequant pool's
A76 capacity, which the `-ub 512` arm uses 4x as often, and that flips `smolvlm2`'s loss
to a win. A `-ub` recommendation measured on a loaded host overstates the flag for an idle
one.

### `ROCKET_QUANT_RESIDENT=auto`: quant weights held resident

- **What it does:** dequantizes each quantized weight to fp16 once and holds it in
  resident NPU BOs. That removes both the per-micro-batch dequant and the per-call pack,
  and lifts quant prefill to fp16 parity.
- **When:** a quantized GGUF used for repeated prefill (agentic, RAG), where the model's
  fp16 size also fits RAM. It trades the quant's RAM saving back for the full fp16
  footprint.
- **RAM math:** you need roughly the fp16 model size free (Qwen3.5-9B ~18 GB), plus the
  NPU IOVA window. There is no disk cost beyond the quant GGUF. Use `auto`, which sizes
  the budget from free RAM, not a blanket `=1`. On a model larger than the default 2 GB
  budget, `=1` residents only part of it, and that is a **net loss against streaming**.
- **Delta:** Qwen3.5-9B `Q4_K` pp2048 reads resident 24.6 vs streaming 15.8 = 1.56x
  (~0.92x the 26.8 F16), with bit-identical PPL [HW sweep 2026-06-28, streaming at
  `-ub 2048` with serial dequant]. On a model that does not fit, it falls back to
  streaming, correctly.
- **Confirm:** `ROCKET_LOG_STDERR=1` prints the one-shot budget decision.

#### Residency in place of `-ub 2048`

**Run residency in place of `-ub 2048`, not stacked on it.** Every model measured both
ways shows it, seven of seven, across a 1.8-11.9 B span. Each row is three rotated passes
with ratios paired within a pass. Every model is 100% resident on every pass except
`gemma4-12b`, which places part of its weights [HW sweep 2026-08-29/31].

| model | unstacked, at the default `-ub` | stacked on `-ub 2048` | unstacked over stacked |
|---|---:|---:|---:|
| `smolvlm2` (1.81 B) | 1.346x | 1.002x | +34% |
| `llama32-3b` | 1.472x | 1.212x | +21% |
| `ministral3-3b` | 1.445x | 1.132x | +28% |
| `phi4mini` | 1.508x | 1.211x | +25% |
| `ministral3-8b` | 1.651x | 1.325x | +25% |
| `qwen35-9b` (8.95 B) | 1.752x | 1.661x | +5.5% |
| `gemma4-12b` (11.9 B) | 1.368x | 1.322x | +3.5% |

Three reasons stack, and the third applies only to a partly-placed model. Below 4 B the
model carries a real non-dequant `-ub 2048` loss that the dequant win masks, at residues
0.789-0.864 under the dqc mechanism control. Stacking pays that loss before residency
arrives. On every model, residency also removes the per-call pack and upload, and the
default `-ub` has four times as many calls to remove them from. That second reason is why
the 9B still gains 5.5% unstacked, even though its own `-ub` residue is a win at 1.038x.

The third is residency itself. `gemma4-12b` places only part of its weights, and both
arms stop at the same 9535 MB `MemAvailable` reserve floor rather than at the
resident-weight budget. The unstacked arm reaches that floor 2.6-2.8 GB later.
`-b 2048 -ub 2048` grows the activation and compute buffers about 4x, and spends that RAM
on the micro-batch instead of on weights.

The measured placement is 83-87% resident unstacked against 73-74% stacked. So on a
partly-placed model, raising `-ub` costs residency directly. A comparison there must read
the `[f16-resident]` outcome line per arm.

One model class is not covered by that table: the `qwen35-08b` class cannot resolve a
ratio this size at an affordable pass count. `gemma4-12b`'s +3.5% is the narrowest margin
in the set, and its per-pass spread is the widest at 6.3%. Read its sign, not its size.

### `ROCKET_F16_RESIDENT=auto`: F16 weights held resident

- **What it does:** the F16 sibling of `ROCKET_QUANT_RESIDENT=auto`. It packs the all-K
  F16 weights once and reuses them across micro-batches and turns.
- **When:** an F16 model that fits ~2x in RAM (the resident tiles plus the source), used
  for repeated prefill.
- **RAM math:** ~2x the fp16 model size. `ROCKET_PREPACK_MADVISE` reclaims the source for
  a prefill-only run, but it **breaks CPU decode**. Do not use it for an interactive or
  serving run.
- **Delta:** single-digit percent, ~+6% pp2048 and +9% pp512 on a 3B F16 model
  [HW sweep]. A fusable projection group (Q\|K\|V, gate\|up) goes resident as one combined
  weight. That stacks pack-once with a shared packA, for a further ~+5.7%. See
  [perf/weight-residency-fusion.md](perf/weight-residency-fusion.md) for the mechanism and
  the A/B.

### MoE routed experts on the NPU

- **What it does:** routes the mixture-of-experts FFNs (`MUL_MAT_ID`) to the NPU. A
  quantized expert takes the native-int8 resident route (`ROCKET_MOE_NATIVE`), which
  ingests each expert once into int8 codes. That removes the per-micro-batch host dequant
  that makes the naive fp16 expert route a loss.
- **When:** nothing to set. The offload is on by default since 2026-08-27 and decides per
  expert stack.
- **RAM math:** gpt-oss-20b holds ~13.4 GB of int8 codes on the NPU. The GGUF source must
  coexist, because MoE decode reads the active experts from it every token. So ~21 GB is
  charged against a budget of `MemAvailable` − 6 GiB. On a 31 GB board that admits 63 of
  its 72 expert stacks. The remaining 9 stay on the CPU, which is a partial offload and
  not a loss.
- **Time:** a one-time expert ingest inside the first prefill, per `llama_context`. It
  takes ~21 s on gpt-oss-20b and ~18 s on DeepSeek-V2-Lite [HW sweep 2026-10-09, 600 MHz].
  Its NPU-BO pack runs ~1.3 GB/s, so it grows with the bytes held resident rather than
  with the expert count. An expert that `llama-bench`'s warm-up never routed to ingests
  inside a timed rep, so a MoE prefill figure from it carries part of the ingest.
- **Delta:** gpt-oss-20b MXFP4 at `-b 2048 -ub 2048` reads 1.81x the CPU at pp512 and
  2.38x at pp2048. It reads 1.64x -> 2.08x over the experts-on-CPU arm across pp512-pp2048
  [HW sweep 2026-08-27, 600 MHz pinned]. At the llama.cpp default `-ub 512` it is
  ~1.6x/1.7x over experts-on-CPU, with no collapse. That tax belongs to the fp16 route.
- **Confirm:** `ROCKET_LOG_STDERR=1` prints the resident/streamed expert split at teardown.
  Under the default, that split reads 100% resident. A nonzero streamed count means a
  limit the pre-flight could not see ahead of the ingest, most likely an exhausted NPU
  IOVA window.

#### Residency pre-flight

A residency pre-flight keeps the sign of the offload independent of the host's RAM. The
stack's resident cost is knowable from its tensor alone. So the placement gate reserves
the whole stack before the first ingest, and leaves on the CPU what it cannot reserve. A
declined stack costs nothing at the backend boundary, because this backend's buffer type
is the CPU buffer type. The default is therefore bounded below by the experts-on-CPU
baseline at any RAM size.

#### Eligibility per architecture

Eligibility is per architecture, not universal. Two size floors gate each op. One is the
tile granule (`ROCKET_MOE_M_BUCKET`). The other is the work one dispatch carries
(`ROCKET_MOE_MIN_WORK`), in mega-MACs of `M_e · K · N` where `M_e` =
`n_tokens · n_used / n_expert`.

gpt-oss-20b routes 4-of-32 over a 2880×2880 expert and clears both floors from `-ub 512`
up. DeepSeek-V2-Lite routes 6-of-64 over a 2048×1408 expert, which is 2.88x less work per
dispatch at the same row count. It clears the floors only past ~1250 tokens in a
micro-batch. **Do not read one MoE's ratio across to another.**

`ROCKET_MOE_MIN_WORK=240` admits DeepSeek's marginal cell (1.05x at `M_e` = 96). That
cell repays its expert ingest only after ~16 000 tokens of prefill. So lower the floor
only for a workload that prefills that much.

#### `ROCKET_MOE=1`

`ROCKET_MOE=1` is the A/B arm, not a recommendation. It claims every op the handler can
compute, reserving nothing and ignoring both size floors. Where the stack nearly fits,
that is faster than the default: 6-13% at pp512 and 18-21% at pp2048. **Where the stack
does not fit, it reads below the experts-on-CPU baseline** (0.97x at pp512 on an induced
12 GB budget). Raise `ROCKET_MOE_CACHE_MB` instead. It buys most of that back while
keeping both the zero-streamed property and the sign guarantee.

#### `ROCKET_MOE_CACHE_MB` on gpt-oss-20b

The table is the measured budget ladder for gpt-oss-20b on a 31 GiB board. **Every rung is
n=1**: one `-r 3` process per arm, behind a `drop_caches`, every arm 0 streamed
[HW sweep 2026-08-27, 600 MHz].

| setting | budget | stacks | pp512 | pp2048 | vs default |
|---|---|---:|---:|---:|---|
| default (auto) | 24672 MB | 63 of 72 | 21.96 | 26.51 | n/a |
| `ROCKET_MOE_CACHE_MB=26000` | 26000 MB | 66 | 22.90 | 27.85 | +4.3% / +5.1% |
| `ROCKET_MOE_CACHE_MB=28000` | 28000 MB | 71 | 23.79 | 30.25 | +8.3% / +14.1% |
| `ROCKET_MOE=1` (the ceiling) | none | 72 | 23.34 | 31.27 | +6.3% / +18.0% |

A budget of 28000 is the setting worth knowing. It places 71 of 72 stacks, recovers 79% of
the default-to-ceiling gain at pp2048, and reads above the forced arm at pp512. The
pre-flight still guarantees the sign. But it leaves only ~2.8 GB of the headroom the 6 GiB
auto reserve exists for (KV cache, activations). So it is a knob for a known working set, not a new
default.

The stack counts are exact and the percentages are not. One process on this board can sit
~10% off the level its own configuration repeats at. So the ladder supports "more budget
places more stacks, and on this model class that pays" rather than those four deltas. The
pp512 column also sits inside a ±1.4-2.1 within-process spread, so do not read it finely
at all. `ROCKET_MOE=0` leaves the experts on the CPU.

#### Expert-dominated models

**On an expert-dominated model the direction inverts**, so `ROCKET_MOE_CACHE_MB` is not a
"raise it if you have RAM" knob. Qwen3-30B-A3B holds 29 of its 30.5 B parameters in the
experts (17.28 GiB GGUF). At `-ub` 4096 on the same board its curve is a plateau, then a
cliff [HW sweep 2026-08-28, RK1, 600 MHz]:

| budget | stacks | vs experts-on-CPU baseline | processes |
|---|---:|---:|---|
| 18000-21000 MB | 58-67 | 1.046x | pooled over seven processes |
| auto (24425 MB) | 79 | 1.014x | n=3 |
| 28000 MB | 90 | 0.999x | n=1 |

The last rung is one process. So it supports "one process read parity", not "at 90 stacks
the offload buys nothing".

The cause is in the admission charge. It counts the GGUF source bytes of the experts it
places, but not of the ones it leaves on the CPU. The CPU reads those from the same mmap
every micro-batch, and they are equally unreclaimable.

#### `ROCKET_MOE_CHARGE_ALL_SOURCE=1`

`ROCKET_MOE_CHARGE_ALL_SOURCE=1` is the charge that counts both, and it is opt-in. It
charges the whole mmapped weight buffer once, up front, and leaves only the int8 codes and
scales per stack. On this board that lands Qwen3-30B-A3B near 46 stacks against the
default's 79.

It is not the default for two reasons. The 3.2% it targets cannot be confirmed on this
board at any affordable pass count. And the same charge moves gpt-oss-20b from 63 stacks
to about 58 [expected: derived from that model's 1.521 charge factor, not measured]. That
model's own ladder pays for going the other way.

Use it on an expert-dominated model whose GGUF is large against RAM. Pinning
`ROCKET_MOE_CACHE_MB` to the plateau does the same job with a number you chose.

#### Charge factor

The units of `ROCKET_MOE_CACHE_MB` are not bytes of RAM, so a recommendation does not
transfer by arithmetic on RAM alone. The charge per expert is `N·K` int8 code bytes, plus
`N·(K/group)·4` scale bytes, plus that expert's GGUF source stride. So a budget buys less
residency than it names, by a charge factor of `1 + 4/group + source_bits_per_weight/8`.
The route measures the factor for you.

The pre-flight's `resident budget reached after N expert stacks (X MB RAM, Y MB IOVA)`
line has the charge and the codes side by side. `X/Y` reads 1.610 on Qwen3-30B-A3B Q4_K_M
against 1.617 derived, and 1.521 on gpt-oss MXFP4 against 1.538 derived. So the formula
sizes a budget to ~1% before a run, and the log pins it after. A Q8_0 MoE would need
about 2.07 [expected: derived from bits/weight, not measured]. Convert a budget, do not
copy it:

```
budget_MB  ~=  factor x (MemTotal - the whole expert GGUF - ~1.2 GiB runtime headroom)
```

#### Board size

This section says what fits on each board size, and where `auto` over-places. Every MoE
number here was taken on a 31 GiB board, which is the only RK3588 size in hand. **The rows
below 32 GB are arithmetic from that board's measurements, not measurements, and are
tagged [expected].** The arithmetic is the charge-factor formula plus a stack's code size
(`n_expert · K · N` bytes). It is checked against the 32 GB board first, where it is out
of sample for two of the three models.

| board (MemTotal) | Qwen3-30B-A3B Q4_K_M, 17.3 GiB | gpt-oss-20b MXFP4, 11.3 GiB | DeepSeek-V2-Lite Q4_K_M, 9.7 GiB |
|---|---|---|---|
| 32 GB (31.0 GiB) | ~67 of 144 stacks fit. Auto asks ~81. Pin 18000-21000 | whole stack fits. Auto asks ~65 of 72. Raise to 28000 | whole stack fits with room. Auto is right |
| | *measured: plateau 58-67, auto took 79* | *measured: auto took 63, 28000 took 71* | *measured: all 78 admitted* |
| 16 GB (15.4 GiB) | not viable: the GGUF alone is 17.3 GiB | ~12 of 72 fit. Auto asks ~24, 2x too many. Pin ~4600 [expected] | ~26 of 78 fit. Auto asks ~32. Pin ~7500, and needs `-ub` past ~1250 to clear the work floor [expected] |
| 8 GB (7.6 GiB) | no | no: GGUF > RAM | no: GGUF > RAM |
| 4 GB | no | no | no |

The 32 GB row is the check, and one half of it is independent evidence. The `auto` column
is mostly arithmetic the pre-flight itself performs. Reproducing it gives 79.0 predicted
against 79 measured on Qwen3 and 64.1 against 63 on gpt-oss. Each figure comes from the
run's own reported budget and measured charge factor.

That confirms the inputs, which are the per-stack code size `n_expert · K · N` and the
charge factor, rather than the model. The confirmation is worth having, because the
small-board rows are computed from those inputs. It is not independent.

The `fits` column is the independent half. It is `MemTotal − GGUF − headroom`, and it has
the one free parameter. Its prediction for Qwen3-30B is ~67 stacks, which is the top of
the measured 58-67 plateau. The arithmetic never saw that performance boundary. The same
column says the gpt-oss and DeepSeek stacks fit whole on this board, which is what was
measured. So the inputs are confirmed exactly, and the one prediction that could have been
wrong landed on the measured edge.

16 GB is the only tier where the answer is "it depends". Prefill touches every expert
every micro-batch, so the working set is the whole expert GGUF, however little of it is
placed. A GGUF larger than RAM therefore thrashes, and no budget setting changes that. At
8 GB and below, all three models are refused on GGUF size alone.

At 16 GB the GGUF fits for two of the three, and the question becomes how much int8 fits
beside it. **That is the case where the budget must be pinned**, because `auto`'s
uncharged remainder is largest at low placement. Placing 12 of gpt-oss's 72 stacks leaves
60 stacks' worth of source, ~9.4 GiB, charged to nobody on a board with 15.4.

The default fails in the safe direction, which is why `auto` is the shipping value. The
flat 6 GiB reserve is 19% of a 32 GB board, 37.5% of 16 GB and 75% of 8 GB. So `auto`
withholds proportionally more as the board gets smaller, and effectively turns the route
off at 8 GB. A stack the budget cannot reserve is never claimed, so those layers run
wholly on the CPU. That is the experts-on-CPU baseline, not a streamed partial-residency
loss. Partial residency is a `ROCKET_MOE=1` failure mode (52% resident, 0.97x), and the
default cannot reach it.

#### Negative results

The fp16 expert route (`ROCKET_MOE_NATIVE=0`) is a net loss on both models, because it
re-dequantizes every expert every micro-batch. The default never takes it. A per-expert
row count is the wrong handle for the floor. `M_e` = 96 is 1.77x on gpt-oss and ~parity on
DeepSeek, so no row threshold separates them.

### Native int8 / int4 / bf16

- **What it does:** the flags are `ROCKET_INT8=1` (+`ROCKET_INT8_HADAMARD=1`),
  `ROCKET_INT4=1` and `ROCKET_BF16=1`. All are numerically faithful (int4/int8
  char-identical to fp16 greedy, bf16 token-identical). All tie the ~460 GOP/s floor,
  because the NPU is DMA/dispatch-bound. So fewer bits do not buy prefill speed.
- **When:** only to make a model fit that would not at F16, or for bf16's fp32 range.
  Resident int8 in-model prefill is 0.60x fp16, and int4 is ~0.53x. No on-chip
  K-accumulation ships for the int32 partials, so each K-tile reads back. The result is
  slower, but a quarter to a half the footprint.
- **Disk cost:** the native int4/int8 paths quantize from a full-precision (F16) GGUF and
  require Hadamard rotation. They are not fed a pre-quantized `Q4` file. So you spend the
  disk of the larger F16 GGUF to save runtime RAM. If you only have a `Q4_K` GGUF, use the
  GGUF-quant streaming path (`-ub 2048` or `ROCKET_QUANT_RESIDENT`) instead. That is a
  different mechanism, and the one that has a speed story.
- **Rule of thumb:** if the goal is RAM, a `Q4_K_M` GGUF at `-b 2048 -ub 2048` is simpler
  and also speeds decode. Native int4 is for the case where you want the ¼ footprint from
  an F16 source.

### Attention offload

- **What it does:** `ROCKET_FLASH_ATTN` (on) offloads prefill attention at
  `n_kv >= 1024`. It is bit-faithful for the attention it implements
  (`softmax(scale·QKᵀ+mask)·V`). **The handler declines an op carrying attention sinks**
  (gpt-oss, some others). It has no sink term, so accepting the op would silently compute
  a different softmax.
- **When it matters:** long contexts. It reads parity at <=1K, 1.50x at 8K and 1.25x at
  16K [HW sweep]. Below ~2K it is a wash, and the default gate handles that.
- **Diagnosis:** you rarely touch it. If a model's NPU curve collapses past ~1K context
  but is fine below it, suspect a sink-bearing attention. The handler must decline such
  an op, and it does.
- **The knobs behind it:** `ROCKET_FLASH_ATTN_MIN_KV` (1024) is the context floor.
  `ROCKET_FA_CHAIN` (on) batches a worker's per-head submits into one job.
  `ROCKET_FLASH_ATTN_NO_CTX=1` forces the per-call path, for an A/B against the persistent
  context.

## Measured default-vs-tuned deltas

The datapath levers are default-on, so the stock-vs-tuned gap for a dense F16 model is
small. The tuned config is mostly the default, and the F16 numbers in
[perf/benchmarks.md](perf/benchmarks.md) are already at it. The large default-vs-tuned
gaps are on the quantized path, where the llama.cpp and stack defaults leave speed
unclaimed:

| Lever | Default | Tuned | Gain | Measured on |
|---|---|---|---|---|
| Clock | 200 MHz | 600 MHz (`patches/rocket`) | 1.43x | Gemma-4-12B [HW sweep] |
| Quant micro-batch | `-ub 512` | `-b 2048 -ub 2048` | 0.91-1.53x. Dense 1.18-1.53x above 8 B | eleven models 0.75-30.53 B [HW sweep 2026-08-29/30, rotated passes]. The 2026-06-28 figures are ~2.1x/2.25x. This sweep reads the 9B at 1.424x and the 27B at 1.532x, both lower by the same 0.68 factor. Total parameter count is the wrong axis for a MoE model |
| Quant residency | streaming | `ROCKET_QUANT_RESIDENT=auto` | 1.35-1.75x at the default `-ub`, which is how to run it. Stacked on `-ub 2048` it reads 1.00-1.66x | nine quant models 0.75-11.91 B [HW sweep 2026-08-28..31, rotated passes]. Unstacked measured on seven: smolvlm2 1.346x, llama32-3b 1.472x, ministral3-3b 1.445x, phi4mini 1.508x, ministral3-8b 1.651x, qwen35-9b 1.752x, gemma4-12b 1.368x (partial residency) |
| F16 residency | re-pack per turn | `ROCKET_F16_RESIDENT=auto` | ~+6-9% | 3B F16 [HW sweep] |
| MoE experts | (default-on) | n/a | 1.64x -> 2.08x over experts-on-CPU, pp512->pp2048 | gpt-oss-20b [HW sweep] |
| Asymmetric tiling | (default-on) | `ROCKET_MM_ASYM=1` | +6-9% F16 | Qwen3.5-9B, Gemma-4-12B [HW sweep] |
| fp16 K-accumulation | (default-on) | `ROCKET_KACC=1` | +19% (+7% more from DATA_REUSE) | Gemma-4-12B [HW sweep] |

The last three rows are deltas over a hypothetical no-lever baseline, shown to size the
win. You do not set them, because they are on. The actionable rows are the first four.

## What is not yet measured

The flag defaults are model-independent, so the recipes hold. The per-model paired
default-vs-tuned A/B has been run only on a subset. Where a recipe implies a per-model
number that is not in [perf/benchmarks.md](perf/benchmarks.md), treat it as a projection,
not a datum. The gaps, and the plan to close them into a full model × use-case × flag
matrix, are tracked outside this repository. The
largest ones:

- `ROCKET_MM_ASYM`, `ROCKET_KACC` and DATA_REUSE isolation exists only on a few models, mostly
  Gemma-4-12B and Qwen3.5. Every other model inherits the default silently.
- `ROCKET_QUANT_RESIDENT` is measured across the matrix, 0.75-11.91 B stacked and
  unstacked on seven models from 1.8 to 11.9 B in size. The lever itself has no
  model-coverage gap left. What is open is the shape of the partial-residency case.
  `gemma4-12b` is the only partly-placed model on the board, at 83-87% unstacked against
  73-74% stacked. How the ratio falls with placed fraction is therefore a two-point
  contrast on one model, not a curve. A second partly-placed model, or a placement sweep
  on this one via `ROCKET_QUANT_RESIDENT_RESERVE_MB`, is what would turn it into one.
- `ROCKET_MOE`'s size floors are fitted on gpt-oss-20b and DeepSeek-V2-Lite, which disagree
  about which shapes are eligible. Qwen3-30B-A3B is a third architecture, and it confirms the
  floor's sign but not its position, at 201 MMAC against the 340 default.
  `ROCKET_MOE_CACHE_MB` has two ladders, gpt-oss and Qwen3-30B-A3B, and they invert. Both
  are on the same 31 GiB board, so the board-size axis is unmeasured and the small-board
  table is arithmetic.
- The `ROCKET_MOE_CACHE_MB` ladders' end rungs are n=1 (both of gpt-oss's three, and Qwen3's
  12000 and 28000), against a per-process spread on this board of ~10%. Only the Qwen3 plateau
  (n=7) against auto (n=3) is deep enough to quote as a size.
- The SmolVLM2 resident `rocket_siglip_encoder` vision path is described but has no
  end-to-end benchmark. The generic clip drop-in is the only measured multimodal-vision
  number (1.19x).
- The F16 prompt-size crossover sits at 21-25 rows on the 0.8B, 3B and 8B models measured. The
  default `ROCKET_MIN_M` of 32 takes every win above it: 1.20-1.46x at pp32 and 1.96-3.90x at
  pp128 against the faster CPU arm. That is on the governor `performance` [HW sweep, 2026-10-09]. A model
  smaller than 0.8B, or one with much narrower matmuls, is not measured.
