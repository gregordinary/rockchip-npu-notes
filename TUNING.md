# Tuning the rocket NPU stack — what to enable, when

Which knobs to set for a given workload, and why. The stack ships its performance
defaults **on**, so most of the win needs no configuration; the choices that remain are a
handful of opt-ins whose value is conditional on the model, the workload, and the RAM you
have. This guide is the decision layer over the per-flag reference in each project's
`API.md` (the flag → meaning tables) and the evidence in [perf/](perf/) (the why behind
each number). For per-model behavior and sampling, see [MODEL-NOTES.md](MODEL-NOTES.md);
for the raw benchmarks, [perf/benchmarks.md](perf/benchmarks.md).

All figures are warm medians, RK3588 at 600 MHz, the same model NPU-vs-CPU (8-thread)
[HW sweep]. Where a delta was measured on one model and inherited by the recipe for
others, that is stated — the flag defaults are model-independent by construction, but the
per-model number is not always separately measured (see
[What is not yet measured](#what-is-not-yet-measured)).

## The short version

The defaults already carry most of the win. On a correctly set-up board you need to set
almost nothing:

- **On by default, leave them:** `ROCKET_KACC` (fp16 K-accumulation, +19%), `ROCKET_REUSE=2`
  (CBUF DATA_REUSE, +7%), `ROCKET_MM_ASYM` (asymmetric tiling, +6–9% on F16), `ROCKET_FLASH_ATTN`
  (attention offload, wins above ~2K context), `ROCKET_DEQUANT_THREADS` (threaded quant
  dequant, +19–33%). You do not set these; you would only ever set one to `0` to A/B it.
- **The four opt-ins that actually change the outcome**, each gated on your workload and RAM:
  1. **`-b 2048 -ub 2048`** (a llama.cpp flag, not a `ROCKET_*` one) — for any **quantized
     GGUF**. ~2.1× over the llama.cpp `-ub 512` default. The single biggest lever, and nearly free —
     the larger micro-batch grows the activation/compute buffers ~4× (512→2048), negligible against the
     weights but not zero, so glance at it on a RAM-tight, swap-less board.
  2. **`ROCKET_QUANT_RESIDENT=auto`** — a quantized GGUF, **if the model's fp16 size fits RAM**.
     Lifts quant prefill to fp16 parity (~1.5×).
  3. **`ROCKET_F16_RESIDENT=auto`** — an F16 model that **fits ~2× in RAM**. Single-digit-percent gain.
  4. **`ROCKET_MOE=1`** — a mixture-of-experts model, **if nearly the whole expert stack fits RAM**.
     Up to 2.16× on gpt-oss-20b; a net loss if it does not fit.
- **Precision (`ROCKET_INT4` / `ROCKET_INT8` / a `Q4_K_M` GGUF)** is a **RAM / model-fit** lever,
  not a speed lever — quantization does not speed prefill at this operating point
  ([perf/not-mac-bound.md](perf/not-mac-bound.md)). Choose it to make a model fit or to speed
  **decode**, never to speed prefill.

If you read nothing else: raise the clock, run quantized GGUFs at `-b 2048 -ub 2048`, and
turn on residency only when the RAM math (below) says it fits.

## The operating-point floor

Do these once; every number in this stack assumes them.

- **Raise the clock to 600 MHz.** The NPU boots pinned at 200 MHz (one-fifth speed). The
  `patches/rocket` clock patch takes it to 600 MHz (~1.43×: Gemma-4-12B pp2048 7.98 → 11.40 t/s
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

1. **Model class** — dense LLM, mixture-of-experts (MoE), multimodal (a vision encoder plus an
   LLM), ASR (Whisper), or detection. Each routes different work to the NPU.
2. **Precision** — F16, a quantized GGUF (`Q4_K_M` / `Q8_0` / `IQ4_XS` / MXFP4), or a native-int
   in-model path. Sets both the RAM footprint and which residency lever applies.
3. **Prompt profile** — how many tokens hit prefill per turn, which decides whether the NPU is
   even on the critical path:
   - **Short** (a chatty turn, tens of tokens): prefill is below the offload floor
     (`ROCKET_MIN_M=128` for F16, `ROCKET_MIN_M_QUANT=512` for quant), so it runs on the **CPU**
     regardless of backend. The NPU does nothing for you here; the turn is decode-bound.
   - **Medium** (512–2048, a RAG chunk, a document paragraph, an agentic tool result): NPU prefill
     engages and wins.
   - **Long / batched** (>2048, or a large system pre-prompt re-processed every turn): the NPU's
     best case, and where residency and attention offload pay the most.
   A **large system pre-prompt turns a "chat" workload into a prefill-bound one** — an agent with a
   several-thousand-token tool/persona preamble hits the NPU hard every turn even if the user's
   message is short.
4. **Resource budget** — free RAM and disk. Every residency opt-in trades RAM for speed, and the
   native-int paths trade **disk** (they require a full-precision GGUF as their source) for runtime
   RAM. The RAM math is per-lever below.

## Use-case recipes

### Interactive chat (short prompts, decode-bound)

A short user turn spends almost all its wall time in **decode**, which is CPU-bound and
bandwidth-limited on both backends ([perf/decode-gemv.md](perf/decode-gemv.md)). The NPU barely
touches it: the prefill is below the offload floor and runs on the CPU anyway.

- **Lever is quantization, for the stream.** `Q4_K_M` decodes ~2.8× faster than F16 (Ministral-3-8B:
  F16 1.4 → Q4_K_M 3.8 t/s decode [HW sweep]) and fits a smaller board. Pick the quant on decode
  speed and RAM, not on the NPU.
- **Residency and `-ub` do nothing here** — there is no large prefill to amortize them over.
- **Exception — a long system pre-prompt.** If the chat carries a big preamble (agent persona, tool
  schemas, retrieved context), the per-turn prefill is large and this becomes the agentic case below.

### Agentic / tool-use (large, repeated prefill)

An agent re-processes a growing context — system prompt, tool schemas, prior steps, tool
outputs — every turn. That is a large prefill on every step, which is exactly the NPU's job, and
the fixed setup cost of residency amortizes across turns.

- **Quantized:** `-b 2048 -ub 2048`, and **`ROCKET_QUANT_RESIDENT=auto` if the fp16 footprint fits
  RAM** (see the RAM math). Residency pays back best here because the same weights serve many turns.
- **F16 with RAM to spare:** `ROCKET_F16_RESIDENT=auto`.
- **Attention offload is automatic** and pays once the context passes ~2K
  ([perf/attention-offload-crossover.md](perf/attention-offload-crossover.md)); nothing to set.
- **MoE model:** add `ROCKET_MOE=1` only if the expert stack fits (below).

### RAG / long-context / document processing (large one-shot prefill)

A single big prefill (a retrieved passage, a pasted document, a filled context window). The
largest NPU prefill wins land here — the ×CPU grows with prefill length up to Qwen3.6-27B's 4.4× at
pp2048 [HW sweep].

- `-b 2048 -ub 2048` for any quantized GGUF; `ROCKET_QUANT_RESIDENT=auto` / `ROCKET_F16_RESIDENT=auto`
  as RAM allows.
- **Attention offload matters at these lengths** — Gemma-4-12B F16: 1.50× at 8K context, 1.25× at
  16K [HW sweep]. On by default; leave it.

### Batch / multi-stream throughput (offline, or many detectors)

Optimize aggregate throughput, not single-request latency.

- **Multiple processes, one per stream**, each pinned to a distinct A76 core via `ROCKET_CPU_AFFINITY`
  — the detection pool reaches 2.98× at four streams (the tflite-rocket delegate).
- Single-stream latency is host cube-gather-bound, not NPU-bound, so more concurrency (not a faster
  submit) is the lever.

### ASR — Whisper (whisper.cpp)

The NPU accelerates the **encoder**; the decoder is autoregressive (M=1 GEMV) and stays on the CPU.
The win grows with the model — tiny.en 1.18× → large-v3 2.14× [HW sweep].

- **No opt-in flags needed** beyond the operating-point floor and `ROCKET_CPU_AFFINITY`. Larger models
  benefit more (encoder matmul work ~d_model²; host packing ~linear).
- **Do not lower `ROCKET_MIN_M`.** whisper.cpp's default beam-5 search presents M=5 per decode step;
  a floor of 4 wrongly offloads that tiny GEMV and costs a net 1.40× end-to-end. The default 128
  keeps beam decode on the CPU where it belongs.
- **Best real-transcription target: large-v3-turbo** — the full 32-layer encoder (NPU-accelerated,
  2.12×) with a 4-layer decoder (~6× cheaper per step), so the accelerated encoder carries most of the
  transcription.

### Detection — Frigate / TFLite (tflite-rocket)

The delegate's key knob is a **delegate option**, not an env var (pass it via `load_delegate` /
`--option`):

- **`native_int8=1`** — the exact-int8 conv path (the default in the Frigate `rocket.py` plugin).
  MobileDet COCO mAP is CPU-parity (0.3321 vs 0.3318 [HW sweep]).
- **Throughput = a process pool**, one per camera, each pinned with `ROCKET_CPU_AFFINITY`
  (3.20 → 9.55 detection_fps, P=1→4). Single-stream ~336 ms is host gather-bound.

## LLM configuration by scenario

The dense-LLM recipe as a table. "Prefill" is F16's fastest; a quantized GGUF is a RAM play that
also speeds decode. Sizes are the fp16 footprint the residency levers need.

| You are running | Prompt profile | Recommended flags | Why |
|---|---|---|---|
| **F16, fits RAM** | medium / long | defaults only (KACC/REUSE/ASYM/FA on) | Already the fastest prefill; nothing to add |
| **F16, fits ~2× RAM** | agentic / RAG (repeated) | `+ ROCKET_F16_RESIDENT=auto` | Pack the weights once; ≈+6% pp2048, +9% pp512 on a 3B F16 model [HW sweep] |
| **Quantized GGUF** | medium / long | `-b 2048 -ub 2048` | ~2.1× over the `-ub 512` default; per-µbatch dequant amortized |
| **Quantized GGUF, fp16 fits RAM** | agentic / RAG | `-b 2048 -ub 2048` + `ROCKET_QUANT_RESIDENT=auto` | Dequant once → fp16 parity (~1.5×); costs the full fp16 footprint |
| **Any, short prompts** | interactive chat | pick `Q4_K_M` for decode; no NPU flags | Prefill is below the offload floor; the turn is decode-bound |
| **MoE (gpt-oss, DeepSeek, …)** | medium / long | `-b 2048 -ub 2048` `+ ROCKET_MOE=1` **iff experts fit** | 2.16× if ~99% resident; a loss below ~82% |
| **Model too big for RAM at F16** | any | a `Q4_K_M` GGUF, or `ROCKET_INT4=1` from an F16 GGUF | Footprint, not speed — see below |

## The opt-ins in detail

Each entry: what it does, when it helps, the RAM/disk it costs, the measured delta, and how to
confirm it engaged.

### `-b 2048 -ub 2048` (llama.cpp) — for every quantized GGUF

A quantized GGUF re-dequantizes to fp16 **per micro-batch**. The llama.cpp default `-ub 512`
re-decodes the whole model every 512 rows; `-ub 2048` spreads that fixed cost.

- **When:** any quantized prefill of ≥ ~512 rows. Irrelevant to F16 (no dequant) and to short prompts
  (one micro-batch).
- **Cost:** the larger micro-batch grows the activation/compute buffers ~4× (512→2048) — negligible
  against the weights, but not zero; on a RAM-tight board (no swap on this one) confirm it still fits.
- **Delta:** ~2.1× — 9B `Q4_K` pp2048 **8.2 → 17.2 t/s** [HW sweep]. `Q4_K` / `IQ4_XS` / `Q8_0`
  converge within noise; quant type does not change NPU prefill throughput.
- **Trap:** never compare a `-ub 512` number against a `-ub 2048` one — a whole class of phantom
  "regressions" is this mistake. See [perf/quant-prefill-microbatch.md](perf/quant-prefill-microbatch.md).

### `ROCKET_QUANT_RESIDENT=auto` — quant weights held resident

Dequantizes each quantized weight to fp16 **once** and holds it in resident NPU BOs, removing both
the per-µbatch dequant and the per-call pack. Lifts quant prefill to fp16 parity.

- **When:** a quantized GGUF used for **repeated** prefill (agentic, RAG), **and** the model's **fp16**
  size fits RAM. It trades the quant's RAM saving back for the full fp16 footprint.
- **RAM math:** you need roughly the **fp16** model size free (Qwen3.5-9B ≈ 18 GB), plus the NPU IOVA
  window. Disk cost: none beyond the quant GGUF. Use **`auto`** (budget sized from free RAM), not a
  blanket `=1`: on a model larger than the default 2 GB budget, `=1` residents only part of it and is a
  **net loss vs streaming**.
- **Delta:** Qwen3.5-9B `Q4_K` pp2048 **resident 24.6 vs streaming 15.8 = 1.56×** (≈0.92× the 26.8 F16),
  bit-identical PPL [HW sweep]. On a model that does not fit, it falls back to streaming, correctly.
- **Confirm:** `ROCKET_LOG_STDERR=1` prints the one-shot budget decision.

### `ROCKET_F16_RESIDENT=auto` — F16 weights held resident

The F16 sibling of the above: pack the all-K F16 weights once and reuse across micro-batches and turns.

- **When:** an F16 model that fits **~2×** in RAM (the resident tiles plus the source), used for
  repeated prefill.
- **RAM math:** ~2× the fp16 model size. Prefill-only reclaim of the source is available
  (`ROCKET_PREPACK_MADVISE`) but **breaks CPU decode** — do not use it for an interactive/serving run.
- **Delta:** single-digit percent — ≈+6% pp2048, +9% pp512 on a 3B F16 model [HW sweep]. A fusable
  projection group (Q\|K\|V, gate\|up) goes resident as one combined weight, stacking pack-once with a
  shared packA for another ≈+5.7% on top — see
  [perf/weight-residency-fusion.md](perf/weight-residency-fusion.md) for the mechanism and the A/B.

### `ROCKET_MOE=1` — MoE routed experts on the NPU

Routes the mixture-of-experts FFNs (`MUL_MAT_ID`) to the NPU. A quantized expert takes the native-int8
resident route by default (`ROCKET_MOE_NATIVE`), ingesting each expert **once** into int8 codes — this
is what removes the per-µbatch host dequant that makes the naive fp16 expert route a loss.

- **When:** a MoE model where **nearly the whole expert stack fits RAM**. This is the one opt-in whose
  sign flips on the machine, which is why it is opt-in.
- **RAM math:** the experts must be ~99% resident. gpt-oss-20b holds ~14 GB of int8 codes on the NPU,
  and the GGUF source must coexist (MoE decode reads the active experts from it every token), so ~21 GB
  is charged — it fits a 31 GB board. Below ~82% resident it **loses** (pp512 12.19 < the 14.11 you get
  leaving experts on the CPU). Disk: the GGUF. Time: a one-time ~70 s expert ingest inside the first
  prefill. Raise `ROCKET_MOE_CACHE_MB` if the RAM is there.
- **Delta:** gpt-oss-20b MXFP4 at `-b 2048 -ub 2048`: **1.34× / 1.88× / 2.16× the CPU** at pp512 /
  pp1024 / pp2048 (and 1.93× the NPU-default at pp2048) [HW sweep]. Wins at every prefill length **when
  resident**.
- **Negative results — do not chase these:** the **fp16** expert route (`ROCKET_MOE_NATIVE=0`) is a net
  loss (re-dequantizes every expert every µbatch). On **DeepSeek-V2-Lite** the whole `ROCKET_MOE=1` path
  is a loss (0.25–0.59× CPU) — leave its experts on the CPU (the default). `ROCKET_MOE=1` helps gpt-oss
  and only where residency holds.
- **Confirm:** `ROCKET_LOG_STDERR=1` prints the resident/streamed expert split at teardown.

### Native int8 / int4 / bf16 — RAM and model-fit, not speed

`ROCKET_INT8=1` (+`ROCKET_INT8_HADAMARD=1`), `ROCKET_INT4=1`, `ROCKET_BF16=1`. All are numerically
faithful (int4/int8 char-identical to fp16 greedy; bf16 token-identical), and all tie the ~460 GOP/s
floor — the NPU is DMA/dispatch-bound, so fewer bits do **not** buy prefill speed.

- **When:** only to make a model **fit** that would not at F16, or for bf16's fp32 range. Resident int8
  in-model prefill is **0.60× fp16**, int4 ~0.53× (the int32 partials can't be K-accumulated on-chip, so
  each K-tile reads back) — slower, but a quarter to a half the footprint.
- **Disk cost — the one people miss:** the native int4/int8 paths quantize from a **full-precision (F16)
  GGUF** and require Hadamard rotation; they are not fed a pre-quantized `Q4` file. So you spend the disk
  of the larger F16 GGUF to save runtime RAM. If you only have a `Q4_K` GGUF, use the GGUF-quant streaming
  path (`-ub 2048` / `ROCKET_QUANT_RESIDENT`) instead — different mechanism, and the one that has a speed story.
- **Rule of thumb:** if the goal is RAM, a `Q4_K_M` GGUF at `-b 2048 -ub 2048` is simpler and also speeds
  decode; reach for native int4 only when you specifically want the ¼ footprint from an F16 source.

### Attention offload — automatic, occasionally worth knowing

`ROCKET_FLASH_ATTN` (on) offloads prefill attention when `n_kv ≥ 1024`. Bit-faithful for the attention it
implements (`softmax(scale·QKᵀ+mask)·V`); an op carrying **attention sinks** (gpt-oss, some others) is
declined, because the handler has no sink term and accepting it would silently compute a different softmax.

- **When it matters:** long contexts — parity at ≤1K, 1.50× at 8K, 1.25× at 16K [HW sweep]. Below ~2K it
  is a wash; the default gate handles that.
- **You rarely touch it.** If a model's NPU curve *collapses past ~1K context but is fine below it*, suspect
  a sink-bearing attention that should be (and now is) declined.

## Default vs tuned — measured deltas

Because the datapath levers are default-on, the "stock vs tuned" gap for a **dense F16** model is small:
the tuned config *is* mostly the default, and the F16 numbers in [perf/benchmarks.md](perf/benchmarks.md)
are already at it. The large default-vs-tuned gaps are on the **quantized** and **MoE** paths, where the
llama.cpp/stack defaults leave real speed on the table:

| Lever | Default | Tuned | Gain | Measured on |
|---|---|---|---|---|
| Clock | 200 MHz | 600 MHz (`patches/rocket`) | 1.43× | Gemma-4-12B [HW sweep] |
| Quant micro-batch | `-ub 512` | `-b 2048 -ub 2048` | ~2.1× | Qwen3.5-9B `Q4_K`, 27B `Q4_K` [HW sweep] |
| Quant residency | streaming | `ROCKET_QUANT_RESIDENT=auto` | ~1.5× (→ fp16 parity) | Qwen3.5-0.8B, 9B [HW sweep] |
| F16 residency | re-pack per turn | `ROCKET_F16_RESIDENT=auto` | ≈+6–9% | 3B F16 [HW sweep] |
| MoE experts | on CPU | `ROCKET_MOE=1` (resident) | up to 2.16× / 1.93× the NPU-default | gpt-oss-20b [HW sweep] |
| Asymmetric tiling | (now default-on) | `ROCKET_MM_ASYM=1` | +6–9% F16 | Qwen3.5-9B, Gemma-4-12B [HW sweep] |
| fp16 K-accumulation | (now default-on) | `ROCKET_KACC=1` | +19% (+7% more from DATA_REUSE) | Gemma-4-12B [HW sweep] |

The last two are shown as deltas over a hypothetical no-lever baseline to size the win; you do not set
them (they are on). The actionable rows are the first five.

## What is not yet measured

The flag *defaults* are model-independent, so the recipes above hold, but the **per-model paired
default-vs-tuned A/B** has only been run on a subset. Treat a per-model number the recipe implies but that
is not in [perf/benchmarks.md](perf/benchmarks.md) as a projection, not a datum. The gaps — and the plan to
close them into a full model × use-case × flag matrix — are tracked in
[../NPU_TODO.md](../NPU_TODO.md) under the tuning-guide effort. The largest ones:

- `ROCKET_MM_ASYM` / `ROCKET_KACC` / DATA_REUSE isolation exists only on a few models (mostly Gemma-4-12B,
  Qwen3.5); every other model inherits the default silently.
- `ROCKET_QUANT_RESIDENT` is measured only on Qwen3.5-0.8B/9B — untested whether a 12B+ fp16 resident even
  fits, or its delta.
- `ROCKET_MOE` is measured on gpt-oss-20b (win) and DeepSeek-V2-Lite (loss) only; no `ROCKET_MOE_CACHE_MB`
  residency sweep beyond the gpt-oss observation.
- The SmolVLM2 resident `rocket_siglip_encoder` vision path is described but has no end-to-end benchmark;
  the generic clip drop-in is the only measured multimodal-vision number (1.19×).
- Prompt-size crossover is characterized on a few models (`ROCKET_MIN_M` sweep on 0.8B/3B/8B); the exact
  short/medium/long boundary per model is not swept.
