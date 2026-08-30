# Tuning the rocket NPU stack

Which knobs to set for a given workload, and why. The stack ships its performance
defaults **on**, so most of the win needs no configuration; the choices that remain are a
handful of opt-ins whose value is conditional on the model, the workload, and the RAM you
have. This guide is the decision layer over the per-flag reference in each project's
`API.md` (the flag -> meaning tables) and the evidence in [perf/](perf/) (the why behind
each number). For per-model behavior and sampling, see [MODEL-NOTES.md](MODEL-NOTES.md);
for the raw benchmarks, [perf/benchmarks.md](perf/benchmarks.md).

All figures are warm medians, RK3588 at 600 MHz, the same model NPU-vs-CPU (8-thread)
[HW sweep]. Where a delta was measured on one model and inherited by the recipe for
others, that is stated. The flag defaults are model-independent by construction, but the
per-model number is not always separately measured (see
[What is not yet measured](#what-is-not-yet-measured)).

## The short version

The defaults already carry most of the win. On a correctly set-up board you need to set
almost nothing:

- **On by default, leave them:** `ROCKET_KACC` (fp16 K-accumulation, +19%), `ROCKET_REUSE=2`
  (CBUF DATA_REUSE, +7%), `ROCKET_MM_ASYM` (asymmetric tiling, +6-9% on F16), `ROCKET_FLASH_ATTN`
  (attention offload, wins above ~2K context), `ROCKET_DEQUANT_THREADS` (threaded quant
  dequant, +19-33%). You do not set these; you would only ever set one to `0` to A/B it.
- **The four opt-ins that actually change the outcome**, each gated on your workload and RAM:
  1. **`-b 2048 -ub 2048`** (a llama.cpp flag, not a `ROCKET_*` one), for any **quantized
     GGUF**. **0.94-1.53x over the llama.cpp `-ub 512` default, model-dependent** — ten models
     measured, larger on the big DENSE models but not predictable from parameter count, and
     negative on two (`smolvlm2` 0.941x, `qwen3-30b-a3b` 0.908x). Still the
     biggest of the flag levers on a large quant model, and nearly free:
     the larger micro-batch grows the activation/compute buffers ~4x (512->2048), negligible against the
     weights but not zero, so glance at it on a RAM-tight, swap-less board.
  2. **`ROCKET_QUANT_RESIDENT=auto`**, for a quantized GGUF **if the model's fp16 size fits RAM**.
     Lifts quant prefill to fp16 parity (~1.5x).
  3. **`ROCKET_F16_RESIDENT=auto`**, for an F16 model that **fits ~2x in RAM**. Single-digit-percent gain.
  4. **Nothing** for a mixture-of-experts model: the routed-expert offload is **on by default**
     since 2026-08-27 and gates itself (it claims a stack only where it can reserve the whole thing
     up front and the per-expert GEMM pays for its own dispatch). Worth ~2.4x the CPU at pp2048 on
     gpt-oss-20b. `ROCKET_MOE=1` overrides both checks and is **faster where the stack nearly fits,
     unsafe where it does not**: an A/B arm, not a recommendation.
- **Precision (`ROCKET_INT4` / `ROCKET_INT8` / a `Q4_K_M` GGUF)** is a **RAM / model-fit** lever,
  not a speed lever; quantization does not speed prefill at this operating point
  ([perf/not-mac-bound.md](perf/not-mac-bound.md)). Choose it to make a model fit or to speed
  **decode**, never to speed prefill.

If you read nothing else: raise the clock, run quantized GGUFs at `-b 2048 -ub 2048`, and
turn on residency only when the RAM math (below) says it fits.

## The operating-point floor

Do these once; every number in this stack assumes them.

- **Raise the clock to 600 MHz.** The NPU boots pinned at 200 MHz (one-fifth speed). The
  `patches/rocket` clock patch takes it to 600 MHz (~1.43x: Gemma-4-12B pp2048 7.98 -> 11.40 t/s
  [HW sweep]). Load with `rocket_npu_clk_hz=600000000`; 900 MHz gives no gain here and pinning it
  hard-locks the box. See [perf/clock.md](perf/clock.md).
- **`sudo -E`.** Plain `sudo` strips the environment, dropping both `/dev/accel` privilege context
  and every `ROCKET_*` / `GGML_BACKEND_PATH` knob. `-E` keeps them.
- **`GGML_BACKEND_PATH` = the absolute path** to `libggml-rocket.so`. A wrong path silently falls
  back to CPU-only (one `failed to load` line, then no NPU).
- **Warm-run discipline.** The clock parks at idle and rides up under load, so a cold run reads
  ~15% low. Discard the first run; compare warm (`llama-bench` does this; use `-r 3`).
- **Confirm the config engaged.** Run once with `ROCKET_LOG_STDERR=1` (llama-bench hides all
  rocket lines without it) and look for the mode/budget line, and `ROCKET_MM_PROFILE=1` for the
  phase breakdown. A mis-set knob is otherwise invisible.

## The four workload variables

The recipe is a function of four things. Read your workload against each before picking flags.

1. **Model class**: dense LLM, mixture-of-experts (MoE), multimodal (a vision encoder plus an
   LLM), ASR (Whisper), or detection. Each routes different work to the NPU.
2. **Precision**: F16, a quantized GGUF (`Q4_K_M` / `Q8_0` / `IQ4_XS` / MXFP4), or a native-int
   in-model path. Sets both the RAM footprint and which residency lever applies.
3. **Prompt profile**: how many tokens hit prefill per turn, which decides whether the NPU is
   even on the critical path:
   - **Short** (a chatty turn, tens of tokens): prefill is below the offload floor
     (`ROCKET_MIN_M=128` for F16, `ROCKET_MIN_M_QUANT=512` for quant), so it runs on the **CPU**
     regardless of backend. The NPU does nothing for you here; the turn is decode-bound.
   - **Medium** (512-2048, a RAG chunk, a document paragraph, an agentic tool result): NPU prefill
     engages and wins.
   - **Long / batched** (>2048, or a large system pre-prompt re-processed every turn): the NPU's
     best case, and where residency and attention offload pay the most.
   A **large system pre-prompt turns a "chat" workload into a prefill-bound one**: an agent with a
   several-thousand-token tool/persona preamble hits the NPU hard every turn even if the user's
   message is short.
4. **Resource budget**: free RAM and disk. Every residency opt-in trades RAM for speed, and the
   native-int paths trade **disk** (they require a full-precision GGUF as their source) for runtime
   RAM. The RAM math is per-lever below.

## Use-case recipes

### Interactive chat (short prompts, decode-bound)

A short user turn spends almost all its wall time in **decode**, which is CPU-bound and
bandwidth-limited on both backends ([perf/decode-gemv.md](perf/decode-gemv.md)). The NPU barely
touches it: the prefill is below the offload floor and runs on the CPU anyway.

- **Lever is quantization, for the stream.** `Q4_K_M` decodes ~2.8x faster than F16 (Ministral-3-8B:
  F16 1.4 -> Q4_K_M 3.8 t/s decode [HW sweep]) and fits a smaller board. Pick the quant on decode
  speed and RAM, not on the NPU.
- **Residency and `-ub` do nothing here**: there is no large prefill to amortize them over.
- **Exception: a long system pre-prompt.** If the chat carries a big preamble (agent persona, tool
  schemas, retrieved context), the per-turn prefill is large and this becomes the agentic case below.

### Agentic / tool-use (large, repeated prefill)

An agent re-processes a growing context (system prompt, tool schemas, prior steps, tool
outputs) every turn. That is a large prefill on every step, which is exactly the NPU's job, and
the fixed setup cost of residency amortizes across turns.

- **Quantized:** `-b 2048 -ub 2048`, and **`ROCKET_QUANT_RESIDENT=auto` if the fp16 footprint fits
  RAM** (see the RAM math). Residency pays back best here because the same weights serve many turns.
- **F16 with RAM to spare:** `ROCKET_F16_RESIDENT=auto`.
- **Attention offload is automatic** and pays once the context passes ~2K
  ([perf/attention-offload-crossover.md](perf/attention-offload-crossover.md)); nothing to set.
- **MoE model:** nothing to set; the expert offload is on by default and declines itself where it
  would not pay (below).

### RAG / long-context / document processing (large one-shot prefill)

A single big prefill (a retrieved passage, a pasted document, a filled context window). The
largest NPU prefill wins land here, and the CPU multiple grows with prefill length up to Qwen3.6-27B's 4.4x at
pp2048 [HW sweep].

- `-b 2048 -ub 2048` for any quantized GGUF; `ROCKET_QUANT_RESIDENT=auto` / `ROCKET_F16_RESIDENT=auto`
  as RAM allows.
- **Attention offload matters at these lengths.** Gemma-4-12B F16: 1.50x at 8K context, 1.25x at
  16K [HW sweep]. On by default; leave it.

### Batch / multi-stream throughput (offline, or many detectors)

Optimize aggregate throughput, not single-request latency.

- **Multiple processes, one per stream**, each pinned to a distinct A76 core via `ROCKET_CPU_AFFINITY`;
  the detection pool reaches 2.98x at four streams (the tflite-rocket delegate).
- Single-stream latency is host cube-gather-bound, not NPU-bound, so more concurrency (not a faster
  submit) is the lever.

### ASR: Whisper (whisper.cpp)

The NPU accelerates the **encoder**; the decoder is autoregressive (M=1 GEMV) and stays on the CPU.
The win grows with the model: tiny.en 1.18x -> large-v3 2.14x [HW sweep].

- **No opt-in flags needed** beyond the operating-point floor and `ROCKET_CPU_AFFINITY`. Larger models
  benefit more (encoder matmul work ~d_model²; host packing ~linear).
- **Do not lower `ROCKET_MIN_M`.** whisper.cpp's default beam-5 search presents M=5 per decode step;
  a floor of 4 wrongly offloads that tiny GEMV and costs a net 1.40x end-to-end. The default 128
  keeps beam decode on the CPU where it belongs.
- **Best real-transcription target: large-v3-turbo**, the full 32-layer encoder (NPU-accelerated,
  2.12x) with a 4-layer decoder (~6x cheaper per step), so the accelerated encoder carries most of the
  transcription.

### Detection: Frigate / TFLite (tflite-rocket)

The delegate's key knob is a **delegate option**, not an env var (pass it via `load_delegate` /
`--option`):

- **`native_int8=1`**: the exact-int8 conv path (the default in the Frigate `rocket.py` plugin).
  MobileDet COCO mAP is CPU-parity (0.3321 vs 0.3318 [HW sweep]).
- **Throughput = a process pool**, one per camera, each pinned with `ROCKET_CPU_AFFINITY`
  (3.20 -> 9.55 detection_fps, P=1->4). Single-stream ~336 ms is host gather-bound.

## LLM configuration by scenario

The dense-LLM recipe as a table. "Prefill" is F16's fastest; a quantized GGUF is a RAM play that
also speeds decode. Sizes are the fp16 footprint the residency levers need.

| You are running | Prompt profile | Recommended flags | Why |
|---|---|---|---|
| **F16, fits RAM** | medium / long | defaults only (KACC/REUSE/ASYM/FA on) | Already the fastest prefill; nothing to add |
| **F16, fits ~2x RAM** | agentic / RAG (repeated) | `+ ROCKET_F16_RESIDENT=auto` | Pack the weights once; ~+6% pp2048, +9% pp512 on a 3B F16 model [HW sweep] |
| **Quantized GGUF** | medium / long | `-b 2048 -ub 2048` | **0.91-1.53x** over the `-ub 512` default. **Dense**: 1.18-1.53x above 8 B, 0.94-1.13x under 4 B. **MoE**: read its own row, 0.91-1.32x. Measure it: two of eleven lose |
| **Quantized GGUF, fp16 fits RAM** | agentic / RAG | `-b 2048 -ub 2048` + `ROCKET_QUANT_RESIDENT=auto` | Dequant once -> fp16 parity (~1.5x); costs the full fp16 footprint |
| **Any, short prompts** | interactive chat | pick `Q4_K_M` for decode; no NPU flags | Prefill is below the offload floor; the turn is decode-bound |
| **MoE (gpt-oss, DeepSeek, …)** | medium / long | `-b 2048 -ub 2048`; the expert offload needs no flag | ~2.4x the CPU at pp2048 on gpt-oss-20b, default-on and self-gating. Eligibility is per architecture, not universal: gpt-oss offloads from `-ub 512` up, DeepSeek-V2-Lite only past ~1250 tokens in a micro-batch |
| **Model too big for RAM at F16** | any | a `Q4_K_M` GGUF, or `ROCKET_INT4=1` from an F16 GGUF | Footprint, not speed; see below |

## The opt-ins in detail

Each entry: what it does, when it helps, the RAM/disk it costs, the measured delta, and how to
confirm it engaged.

### `-b 2048 -ub 2048` (llama.cpp) for every quantized GGUF

A quantized GGUF re-dequantizes to fp16 **per micro-batch**. The llama.cpp default `-ub 512`
re-decodes the whole model every 512 rows; `-ub 2048` spreads that fixed cost.

- **When:** any quantized prefill of >= ~512 rows. Irrelevant to F16 (no dequant) and to short prompts
  (one micro-batch).
- **Cost:** the larger micro-batch grows the activation/compute buffers ~4x (512->2048), negligible
  against the weights, but not zero; on a RAM-tight board (no swap on this one) confirm it still fits.
- **Delta:** **0.94-1.53x, model-dependent**, over ten models 0.75-27.32 B
  [HW sweep 2026-08-29, rotated passes]. **The split is at ~4 B**: models above 8 B read
  **1.18-1.53x** (1.176 / 1.424 / 1.257 / 1.308 / 1.532 at 8.49 / 8.95 / 11.91 / 14.66 / 27.32 B)
  and models under 4 B read **0.94-1.13x**, with `smolvlm2` (1.81 B) a **loss** at 0.941x. Within
  the large group parameter count does not order it. **That split is DENSE models only**:
  `qwen3-30b-a3b` is 30.53 B and reads **0.908x**, the matrix's second loss, because only ~3 B are
  active per token — read a MoE model from its own row, not from its parameter count. The superseded ~2.1x / 2.25x are 2026-06-28
  readings of the 9B and 27B; **both have fallen by the same 0.68 factor** (2.098 -> 1.424,
  2.250 -> 1.532), because three default-on host-cost cuts raised the `-ub 512` baseline ~1.47x
  more than the `-ub 2048` arm on each — measure it rather than assuming the class. The superseded ~2.1x
  (8.2 -> 17.2 t/s) is a 2026-06-28 reading of the same 9B; three default-on host-cost cuts have
  since raised the `-ub 512` baseline 2.33x against the `-ub 2048` arm's 1.59x. `Q4_K` / `IQ4_XS` /
  `Q8_0` converge within noise; quant type does not change NPU prefill throughput.
- **Trap:** never compare a `-ub 512` number against a `-ub 2048` one; a whole class of phantom
  "regressions" is this mistake. See [perf/quant-prefill-microbatch.md](perf/quant-prefill-microbatch.md).

### `ROCKET_QUANT_RESIDENT=auto`: quant weights held resident

Dequantizes each quantized weight to fp16 **once** and holds it in resident NPU BOs, removing both
the per-µbatch dequant and the per-call pack. Lifts quant prefill to fp16 parity.

- **When:** a quantized GGUF used for **repeated** prefill (agentic, RAG), **and** the model's **fp16**
  size fits RAM. It trades the quant's RAM saving back for the full fp16 footprint.
- **RAM math:** you need roughly the **fp16** model size free (Qwen3.5-9B ~18 GB), plus the NPU IOVA
  window. Disk cost: none beyond the quant GGUF. Use **`auto`** (budget sized from free RAM), not a
  blanket `=1`: on a model larger than the default 2 GB budget, `=1` residents only part of it and is a
  **net loss vs streaming**.
- **Delta:** Qwen3.5-9B `Q4_K` pp2048 **resident 24.6 vs streaming 15.8 = 1.56x** (~0.92x the 26.8 F16),
  bit-identical PPL [HW sweep]. On a model that does not fit, it falls back to streaming, correctly.
- **Confirm:** `ROCKET_LOG_STDERR=1` prints the one-shot budget decision.

### `ROCKET_F16_RESIDENT=auto`: F16 weights held resident

The F16 sibling of the above: pack the all-K F16 weights once and reuse across micro-batches and turns.

- **When:** an F16 model that fits **~2x** in RAM (the resident tiles plus the source), used for
  repeated prefill.
- **RAM math:** ~2x the fp16 model size. Prefill-only reclaim of the source is available
  (`ROCKET_PREPACK_MADVISE`) but **breaks CPU decode**; do not use it for an interactive or serving run.
- **Delta:** single-digit percent, ~+6% pp2048 and +9% pp512 on a 3B F16 model [HW sweep]. A fusable
  projection group (Q\|K\|V, gate\|up) goes resident as one combined weight, stacking pack-once with a
  shared packA for another ~+5.7% on top; see
  [perf/weight-residency-fusion.md](perf/weight-residency-fusion.md) for the mechanism and the A/B.

### MoE routed experts on the NPU

Routes the mixture-of-experts FFNs (`MUL_MAT_ID`) to the NPU. A quantized expert takes the native-int8
resident route (`ROCKET_MOE_NATIVE`), ingesting each expert **once** into int8 codes, which is what
removes the per-µbatch host dequant that makes the naive fp16 expert route a loss.

- **When: nothing to set.** It is on by default since 2026-08-27 and decides per expert stack. What
  made it opt-in was that its sign depended on the host's RAM; a **residency pre-flight** now removes
  that dependence instead of documenting it: the stack's resident cost is knowable from its tensor
  alone, so the placement gate reserves the whole stack *before* the first ingest and leaves on the CPU
  what it cannot reserve. A declined stack costs nothing at the seam (this backend's buffer type *is*
  the CPU buffer type), so the default is bounded below by the experts-on-CPU baseline at any RAM size.
- **Eligibility is per architecture, not universal.** Two size floors gate each op: the tile granule
  (`ROCKET_MOE_M_BUCKET`) and the work one dispatch carries (`ROCKET_MOE_MIN_WORK`, in mega-MACs of
  `M_e · K · N` where `M_e` = `n_tokens · n_used / n_expert`). gpt-oss-20b routes 4-of-32 over a
  2880×2880 expert and clears both from `-ub 512` up; DeepSeek-V2-Lite routes 6-of-64 over a 2048×1408
  expert, **2.88x less work per dispatch at the same row count**, and clears them only past ~1250
  tokens in a micro-batch. Do not read one MoE's ratio across to another.
- **RAM math:** gpt-oss-20b holds ~13.4 GB of int8 codes on the NPU, and the GGUF source must coexist
  (MoE decode reads the active experts from it every token), so ~21 GB is charged against a budget of
  `MemAvailable` − 6 GiB. On a 31 GB board that admits 63 of its 72 expert stacks; the remaining 9 stay
  on the CPU, which is a partial offload and not a loss. Time: a one-time expert ingest inside the
  first prefill, per `llama_context`: **~36 s** on gpt-oss-20b, ~32 s on DeepSeek-V2-Lite, dominated
  by an NPU-BO pack that is bytes-bound at ~500-545 MB/s rather than per-expert.
- **Delta:** gpt-oss-20b MXFP4 at `-b 2048 -ub 2048`: **1.81x / 2.38x the CPU** at pp512 / pp2048, and
  1.64x -> 2.08x over the experts-on-CPU arm across pp512-pp2048 [HW sweep 2026-08-27, 600 MHz pinned].
  At the llama.cpp default `-ub 512` it is ~1.6x/1.7x over experts-on-CPU, with **no collapse**; that tax
  belonged to the fp16 route.
- **`ROCKET_MOE=1` is the A/B arm, not a recommendation.** It claims every op the handler can compute,
  reserving nothing and ignoring both size floors. Where the stack nearly fits that is faster than the
  default, **6-13% at pp512 and 18-21% at pp2048**, and where it does not it reads **below** the
  experts-on-CPU baseline (0.97x at pp512 on an induced 12 GB budget). **Raise
  `ROCKET_MOE_CACHE_MB` instead**: it buys most of that back while keeping both the zero-streamed
  property and the sign guarantee.
- **The measured budget ladder, gpt-oss-20b on a 31 GiB board** [HW sweep 2026-08-27, 600 MHz,
  behind a `drop_caches`, every arm 0 streamed, **one `-r 3` process per arm, so every rung is
  n=1**]:

  | setting | budget | stacks | pp512 | pp2048 | vs default |
  |---|---|---:|---:|---:|---|
  | default (auto) | 24672 MB | 63 of 72 | 21.96 | 26.51 | n/a |
  | `ROCKET_MOE_CACHE_MB=26000` | 26000 MB | 66 | 22.90 | 27.85 | +4.3% / +5.1% |
  | **`ROCKET_MOE_CACHE_MB=28000`** | 28000 MB | **71** | 23.79 | 30.25 | **+8.3% / +14.1%** |
  | `ROCKET_MOE=1` (the ceiling) | none | 72 | 23.34 | 31.27 | +6.3% / +18.0% |

  **28000 is the setting worth knowing**: 71 of 72 stacks, 79% of the pp2048 ceiling, and above the
  forced arm at pp512, with the pre-flight still guaranteeing the sign. But it leaves only ~2.8 GB
  of the headroom the 6 GiB auto reserve exists for (KV cache, activations), so it is a knob for a
  **known working set**, not a new default. **The stack counts are exact and the percentages are
  not**: one process on this board can sit ~10% off the level its own configuration repeats at, so
  the ladder supports "more budget places more stacks, and on this model class that pays" rather
  than those four deltas. The pp512 column is inside a ±1.4-2.1 within-process spread on top of
  that and should not be read finely at all. `ROCKET_MOE=0` leaves the experts on the CPU.
- **On an expert-dominated model the direction INVERTS**, so this is not a "raise it if you have
  RAM" knob. Qwen3-30B-A3B holds 29 of its 30.5 B parameters in the experts (17.28 GiB GGUF), and
  at `-ub` 4096 on the same board the curve is a plateau then a cliff: **18000-21000 MB** takes
  58-67 stacks and reads 1.046x the experts-on-CPU baseline (pooled over seven processes), auto
  (24425 MB) takes 79 and reads 1.014x (n=3), and 28000 MB takes 90 and reads 0.999x (**n=1**, so
  it supports "one process read parity", not "at 90 stacks the offload buys nothing")
  [HW sweep 2026-08-28, RK1, 600 MHz]. The cause is that the admission charge counts the GGUF
  source bytes of the experts it *places* but not of the ones it leaves on the CPU, which the CPU
  reads from the same mmap every micro-batch and which are equally unreclaimable.
- **The knob's units are not bytes of RAM, so a recommendation does not transfer by arithmetic on
  RAM alone.** The charge per expert is `N·K` int8 code bytes + `N·(K/group)·4` scale bytes + that
  expert's GGUF source stride, so budget buys less residency than it names by a **charge factor**
  `1 + 4/group + source_bits_per_weight/8` -- and the route measures it for you. The pre-flight's
  `resident budget reached after N expert stacks (X MB RAM, Y MB IOVA)` line has the charge and the
  codes side by side, and `X/Y` reads **1.610** on Qwen3-30B-A3B Q4_K_M and **1.521** on gpt-oss
  MXFP4 against **1.617** and **1.538** derived, so the formula sizes a budget to ~1% before a run
  and the log pins it after. A Q8_0 MoE would need about **2.07** **[expected -- derived from
  bits/weight, not measured]**. Convert, do not copy:

  ```
  budget_MB  ~=  factor x (MemTotal - the whole expert GGUF - ~1.2 GiB runtime headroom)
  ```
- **Board size: what fits, and where `auto` over-places.** Every MoE number here was taken on a
  31 GiB board, which is the only RK3588 size in hand, so **the rows below 32 GB are arithmetic
  from that board's measurements and are tagged [expected]** -- not measurements. The arithmetic is
  the formula above plus a stack's code size (`n_expert · K · N` bytes), and it is checked against
  the 32 GB board first, where it is out of sample for two of the three models:

  | board (MemTotal) | Qwen3-30B-A3B Q4_K_M, 17.3 GiB | gpt-oss-20b MXFP4, 11.3 GiB | DeepSeek-V2-Lite Q4_K_M, 9.7 GiB |
  |---|---|---|---|
  | **32 GB** (31.0 GiB) | ~67 of 144 stacks fit; auto asks ~81. **Pin 18000-21000** | whole stack fits; auto asks ~65 of 72. **Raise to 28000** | whole stack fits with room; auto is right |
  | | *measured: plateau 58-67, auto took 79* | *measured: auto took 63, 28000 took 71* | *measured: all 78 admitted* |
  | **16 GB** (15.4 GiB) | **not viable** -- the GGUF alone is 17.3 GiB | ~12 of 72 fit; auto asks ~24, **2x too many**. Pin ~4600 [expected] | ~26 of 78 fit; auto asks ~32. Pin ~7500, and needs `-ub` past ~1250 to clear the work floor [expected] |
  | **8 GB** (7.6 GiB) | no | no -- GGUF > RAM | no -- GGUF > RAM |
  | **4 GB** | no | no | no |

  **The 32 GB row is the check, and it is worth being exact about which half of it is evidence.**
  The `auto` column is mostly arithmetic the pre-flight itself performs, so reproducing it
  (**79.0 predicted against 79 measured** on Qwen3, **64.1 against 63** on gpt-oss, from each run's
  own reported budget and measured charge factor) confirms the *inputs* -- the per-stack code size
  `n_expert · K · N` and the charge factor -- rather than the model. That is worth having, because
  those inputs are what the small-board rows are computed from, but it is not independent.

  **The `fits` column is the independent half.** It is `MemTotal − GGUF − headroom`, it has the one
  free parameter, and its prediction for Qwen3-30B is **~67 stacks** -- which is the top of the
  **measured 58-67 plateau**, a performance boundary the arithmetic never saw. The same column says
  the gpt-oss and DeepSeek stacks fit whole on this board, which is what was measured. So: the
  inputs are confirmed exactly, and the one prediction that could have been wrong landed on the
  measured edge.

  **Why 16 GB is the only tier where the answer is "it depends".** Prefill touches **every** expert
  **every** micro-batch, so the working set is the whole expert GGUF no matter how little of it is
  placed. A GGUF larger than RAM therefore thrashes, and no budget setting changes that: at 8 GB
  and below all three models are refused on GGUF size alone. At 16 GB the GGUF fits for two of the
  three and the question becomes how much int8 fits beside it -- which is the case where the budget
  actually has to be pinned, because `auto`'s uncharged remainder is largest at LOW placement.
  Placing 12 of gpt-oss's 72 stacks leaves 60 stacks' worth of source, ~9.4 GiB, charged to nobody
  on a board with 15.4.

  **The default's own failure direction is safe, which is why `auto` is still the shipping value.**
  The flat 6 GiB reserve is 19% of a 32 GB board, 37.5% of 16 GB and 75% of 8 GB, so `auto`
  withholds proportionally more the smaller the board is and effectively turns the route off at
  8 GB. And a stack the budget cannot reserve is never claimed, so those layers run wholly on the
  CPU: the experts-on-CPU baseline, not a streamed partial-residency loss. Partial residency is a
  `ROCKET_MOE=1` failure mode (52% resident, 0.97x), and the default cannot reach it.
- **Negative results, not worth chasing:** the **fp16** expert route (`ROCKET_MOE_NATIVE=0`) is a net
  loss on both models, re-dequantizing every expert every µbatch, and the default never takes it. And a
  per-expert row count is the wrong handle for the floor: `M_e` = 96 is 1.77x on gpt-oss and ~parity on
  DeepSeek, so no row threshold separates them.
- **Confirm:** `ROCKET_LOG_STDERR=1` prints the resident/streamed expert split at teardown. Under the
  default that split should read **100% resident**; a nonzero streamed count means a limit the
  pre-flight could not see ahead of the ingest, most likely an exhausted NPU IOVA window.

### Native int8 / int4 / bf16

`ROCKET_INT8=1` (+`ROCKET_INT8_HADAMARD=1`), `ROCKET_INT4=1`, `ROCKET_BF16=1`. All are numerically
faithful (int4/int8 char-identical to fp16 greedy; bf16 token-identical), and all tie the ~460 GOP/s
floor, because the NPU is DMA/dispatch-bound, so fewer bits do **not** buy prefill speed.

- **When:** only to make a model **fit** that would not at F16, or for bf16's fp32 range. Resident int8
  in-model prefill is **0.60x fp16**, int4 ~0.53x (the int32 partials can't be K-accumulated on-chip, so
  each K-tile reads back): slower, but a quarter to a half the footprint.
- **Disk cost, the one people miss:** the native int4/int8 paths quantize from a **full-precision (F16)
  GGUF** and require Hadamard rotation; they are not fed a pre-quantized `Q4` file. So you spend the disk
  of the larger F16 GGUF to save runtime RAM. If you only have a `Q4_K` GGUF, use the GGUF-quant streaming
  path (`-ub 2048` / `ROCKET_QUANT_RESIDENT`) instead: a different mechanism, and the one that has a speed story.
- **Rule of thumb:** if the goal is RAM, a `Q4_K_M` GGUF at `-b 2048 -ub 2048` is simpler and also speeds
  decode; reach for native int4 only when you specifically want the ¼ footprint from an F16 source.

### Attention offload

`ROCKET_FLASH_ATTN` (on) offloads prefill attention when `n_kv >= 1024`. Bit-faithful for the attention it
implements (`softmax(scale·QKᵀ+mask)·V`); an op carrying **attention sinks** (gpt-oss, some others) is
declined, because the handler has no sink term and accepting it would silently compute a different softmax.

- **When it matters:** long contexts. Parity at <=1K, 1.50x at 8K, 1.25x at 16K [HW sweep]. Below ~2K it
  is a wash; the default gate handles that.
- **You rarely touch it.** If a model's NPU curve *collapses past ~1K context but is fine below it*, suspect
  a sink-bearing attention that should be (and now is) declined.

## Measured default-vs-tuned deltas

Because the datapath levers are default-on, the "stock vs tuned" gap for a **dense F16** model is small:
the tuned config *is* mostly the default, and the F16 numbers in [perf/benchmarks.md](perf/benchmarks.md)
are already at it. The large default-vs-tuned gaps are on the **quantized** and **MoE** paths, where the
llama.cpp/stack defaults leave real speed on the table:

| Lever | Default | Tuned | Gain | Measured on |
|---|---|---|---|---|
| Clock | 200 MHz | 600 MHz (`patches/rocket`) | 1.43x | Gemma-4-12B [HW sweep] |
| Quant micro-batch | `-ub 512` | `-b 2048 -ub 2048` | **0.91-1.53x**; dense 1.18-1.53x above 8 B | eleven models 0.75-30.53 B [HW sweep 2026-08-29/30, rotated passes]. The superseded ~2.1x/2.25x are 2026-06-28 figures; the 9B now reads 1.424x and the 27B 1.532x, both down by the same 0.68 factor. Total parameter count is the wrong axis for a MoE model |
| Quant residency | streaming | `ROCKET_QUANT_RESIDENT=auto` | ~1.5x (-> fp16 parity) | Qwen3.5-0.8B, 9B [HW sweep] |
| F16 residency | re-pack per turn | `ROCKET_F16_RESIDENT=auto` | ~+6-9% | 3B F16 [HW sweep] |
| MoE experts | (now default-on) | n/a | 1.64x -> 2.08x over experts-on-CPU, pp512->pp2048 | gpt-oss-20b [HW sweep] |
| Asymmetric tiling | (now default-on) | `ROCKET_MM_ASYM=1` | +6-9% F16 | Qwen3.5-9B, Gemma-4-12B [HW sweep] |
| fp16 K-accumulation | (now default-on) | `ROCKET_KACC=1` | +19% (+7% more from DATA_REUSE) | Gemma-4-12B [HW sweep] |

The last three are shown as deltas over a hypothetical no-lever baseline to size the win; you do not
set them (they are on). The actionable rows are the first four.

## What is not yet measured

The flag *defaults* are model-independent, so the recipes above hold, but the **per-model paired
default-vs-tuned A/B** has only been run on a subset. Treat a per-model number the recipe implies but that
is not in [perf/benchmarks.md](perf/benchmarks.md) as a projection, not a datum. The gaps, and the plan to
close them into a full model × use-case × flag matrix, are tracked in
the project's own open-work tracker, which is not part of this repo. The largest ones:

- `ROCKET_MM_ASYM` / `ROCKET_KACC` / DATA_REUSE isolation exists only on a few models (mostly Gemma-4-12B,
  Qwen3.5); every other model inherits the default silently.
- `ROCKET_QUANT_RESIDENT` is measured only on Qwen3.5-0.8B/9B; it is untested whether a 12B+ fp16 resident even
  fits, or its delta.
- `ROCKET_MOE`'s size floors are fitted on gpt-oss-20b and DeepSeek-V2-Lite, which disagree about
  which shapes are eligible; Qwen3-30B-A3B is a third architecture and confirms the floor's **sign**
  but not its position (201 MMAC is far below the 340 default). `ROCKET_MOE_CACHE_MB` now has two
  ladders, gpt-oss and Qwen3-30B-A3B, and they invert -- but both are on the same 31 GiB board, so
  the board-size axis is unmeasured and the small-board table above is arithmetic.
- The `ROCKET_MOE_CACHE_MB` ladders' end rungs are **n=1** (both of gpt-oss's three, and Qwen3's
  12000 and 28000), against a per-process spread on this board of ~10%. Only the Qwen3 plateau
  (n=7) against auto (n=3) is deep enough to quote as a size.
- The SmolVLM2 resident `rocket_siglip_encoder` vision path is described but has no end-to-end benchmark;
  the generic clip drop-in is the only measured multimodal-vision number (1.19x).
- Prompt-size crossover is characterized on a few models (`ROCKET_MIN_M` sweep on 0.8B/3B/8B); the exact
  short/medium/long boundary per model is not swept.
