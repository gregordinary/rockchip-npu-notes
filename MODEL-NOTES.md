# Model notes: LLMs and ASR models on the rocket NPU stack

This page lists the models validated on the FOSS NPU stack, with their recommended settings and
known behavior. LLMs run through stock llama.cpp, and ASR models run through
transcribe.cpp, each with the `ggml-rocket` backend loaded. Each model's entry gives:

- Whether its prefill runs faithfully on the NPU
- The options and sampling settings it needs
- Its known behavior
- The use it suits

[TUNING.md](TUNING.md) is the decision guide for which options to set for a workload. It covers
the use case, the precision, the prompt size and the RAM or disk each opt-in needs.

## Supported models

### LLMs

| Model | Precision | Status | Recommended options | Gain from the options | Prefill against the CPU | Conditions |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | F16 | Supported. Prefill is faithful: greedy output is token-identical to the CPU [HW sweep 2026-06-28] | Defaults | | ~1.44x at pp512 [HW sweep] | Warm |
| Qwen3.5-0.8B | Q4_K_M | Supported | `-b 2048 -ub 2048` with `ROCKET_QUANT_RESIDENT=auto` | 1.179x | | pp2048 [HW sweep 2026-08-28] |
| Llama-3.2-3B-Instruct | Q4_K_M | Supported | `ROCKET_QUANT_RESIDENT=auto` at the default `-ub` | 1.472x | | [HW sweep 2026-08-29/31] |
| Llama-3.2-3B-Instruct | F16 | Supported | `ROCKET_F16_RESIDENT=auto` | 1.081x | | |
| Ministral-3-3B | Q4_K_M | Supported | `ROCKET_QUANT_RESIDENT=auto` at the default `-ub` | 1.445x | | [HW sweep 2026-08-29/31] |
| Ministral-3-3B | F16 | Supported | `ROCKET_F16_RESIDENT=auto` | 1.078x | | |
| Phi-4-mini | Q4_K_M | Supported | `ROCKET_QUANT_RESIDENT=auto` at the default `-ub` | 1.508x | | [HW sweep 2026-08-29/31] |
| Phi-4-mini | F16 | Supported | `ROCKET_F16_RESIDENT=auto` | 1.084x | | |
| SmolVLM2-2.2B-Instruct, text path | Q4_K_M | Supported | `ROCKET_QUANT_RESIDENT=auto` at the default `-ub 512` | 1.346x | | [HW sweep 2026-08-31] |
| Qwen3.5-9B | Q4_K_M | Supported | `ROCKET_QUANT_RESIDENT=auto` at the default `-ub` | 1.752x | | [HW sweep 2026-08-29/31] |
| Ministral-3-8B | Q4_K_M | Supported | `ROCKET_QUANT_RESIDENT=auto` at the default `-ub` | 1.651x | | [HW sweep 2026-08-29/31] |
| Gemma-4-12B-it | Q4_K_M | Supported | `ROCKET_QUANT_RESIDENT=auto` at the default `-ub` | 1.368x | | [HW sweep 2026-08-29/31] |
| Gemma-4-12B-it | F16 | Supported | `ROCKET_F16_RESIDENT=auto` | 1.0614x | | |
| Phi-4-14B | Q4_K_M | Supported | `-b 2048 -ub 2048` | 1.308x | | pp2048 [HW sweep 2026-08-29] |
| Qwen3.6-27B | Q4_K_M | Supported | `-b 2048 -ub 2048` | 1.532x | | pp2048 [HW sweep 2026-08-29] |
| gpt-oss-20b, MoE | MXFP4 | Supported. Greedy output differs from the CPU ([stack status](#gpt-oss-20b-stack-status)) | `-b 2048 -ub 2048`. The expert offload is on by default | | 1.81x at pp512, 2.38x at pp2048 | [HW sweep 2026-08-27], governor `performance` |
| DeepSeek-V2-Lite, MoE and MLA | Q4_K_M | Supported | `-b 2048 -ub 2048`. The expert offload is on by default | | 1.33x at pp512, 1.53x at pp2048 | [HW sweep 2026-08-27] |
| Qwen3-30B-A3B, MoE | Q4_K_M | Supported | Defaults | Every tuned configuration is slower | | [HW sweep 2026-08-30] |

**Conditions:** RK3588 on a Turing RK1 (31 GiB), NPU clock ~816 MHz. Each tuning figure dated
2026-08-28 to 2026-08-31 is from three passes per model, with the configuration order rotated.

The gain is prefill with the recommended options against prefill with the defaults, both on the
NPU. [perf/benchmarks.md](perf/benchmarks.md) has the per-model results against the CPU.

### ASR models

| Model | Best use |
|---|---|
| Voxtral-mini 3B | Highest-accuracy transcription and translation, where latency is not the concern |
| MOSS 0.9B | Diarization where quality matters more than speed (~0.42x realtime) |
| Granite-Speech-2B | Fast clean transcription (base), and coarse diarization (plus) |
| SenseVoice-small 234M | Low-latency short-clip transcription where quality is secondary |
| Fun-ASR-nano 800M | Multilingual coverage |
| Parakeet CTC/TDT-0.6B | Short utterances such as voice commands ([Parakeet on short utterances](#parakeet-on-short-utterances)) |

The settings every ASR model needs are under
[Flags for every ASR model](#flags-for-every-asr-model).

## NPU role and operating point

**Conditions:** RK3588 on a Turing RK1, NPU clock ~816 MHz, governor `performance`.

- Prefill (prompt eval, batched GEMM) runs on the NPU, and decode (token generation, M=1
  GEMV) stays on the CPU. Generation throughput is therefore CPU-bound, and is ~identical with
  or without the backend loaded. The NPU is a prefill engine.
- An F16 prefill is faster than the CPU from ~32 tokens. On the 0.8B, 3B and 8B models
  measured, the crossover is at 21-25 rows. At pp32 they run 1.20-1.46x the faster CPU
  configuration [HW sweep 2026-10-09]. A quantized prefill stays on the CPU below 512 rows
  (`ROCKET_MIN_M_QUANT`).
- Below the crossover, the fixed per-call cost (host scatter and pack, submit, readback, the
  per-submit dispatch overhead) dominates, and the CPU is faster. Do not judge prefill speed
  from a short interactive prompt, whose `Prompt: N t/s` figure mostly measures overhead. See
  [perf/asr-cpu-relief.md](perf/asr-cpu-relief.md) §"Where the floor comes from" and
  [perf/not-mac-bound.md](perf/not-mac-bound.md).
- Discard the first, cold-clock run. The NPU idles at 198 MHz and ramps under load, so a
  cold run measures ~15% low. A short job finishes before the clock reaches ~816 MHz. Compare
  warm runs. `llama-bench` discards the cold run, so use `-r 3`.

## Baseline invocation

**Conditions:** RK3588 on a Turing RK1.

Every NPU run needs these settings:

- Set `GGML_BACKEND_PATH` to the absolute path of `ggml-rocket/build-dl/libggml-rocket.so`.
- Build the backend against the host application's own ggml.
- Run under `sudo -E`. Plain `sudo` strips the environment, and `-E` keeps
  `GGML_BACKEND_PATH` and the `ROCKET_*` settings alongside the `/dev/accel` privilege.
- Pin to the A76 big cores with `taskset 0xf0` (cpus 4-7 on the RK3588). Pinning gives
  1.05-1.13x on NPU prefill over three models, most on the smallest, with or without resident
  weights [HW sweep 2026-09-01]. Leave a decode-heavy run unpinned, because F16 decode uses
  all eight cores for their bandwidth.
- For a quantized GGUF, consider `-b 2048 -ub 2048`. A quantized GGUF re-dequantizes to fp16
  per micro-batch, and the larger micro-batch amortizes that. Its gain depends on the model:
  0.91-1.53x over eleven models, and a loss on two [HW sweep 2026-08-29/30, ~816 MHz]. Use the
  model's own row. Where the fp16 image fits in RAM, `ROCKET_QUANT_RESIDENT=auto` gives a
  larger gain.
- Confirm that the NPU ran the prefill. Prepend `ROCKET_MM_PROFILE=1`, and look for a
  `ROCKET profile total(ms):` line on stderr. Check also that startup prints no
  `failed to load` line.
- Stage models on external storage. On the RK1, the eMMC root and `/tmp` are small.

> [!NOTE]
> With a wrong `GGML_BACKEND_PATH`, the run falls back to the CPU silently. The only sign is a
> `failed to load` line at startup.

> [!NOTE]
> A backend built against another ggml can stop offloading attention with no error.
> [ggml op ordinals](#ggml-op-ordinals) gives the cause.

These settings are on by default:

- fp16 K-accumulation (`ROCKET_KACC`)
- DATA_REUSE (`ROCKET_REUSE=2`)
- Asymmetric tiling (`ROCKET_MM_ASYM`)
- Attention offload (`ROCKET_FLASH_ATTN`)

The workload-specific opt-ins are in [TUNING.md](TUNING.md).

### ggml op ordinals

The op enum in ggml shifts between versions without moving `GGML_BACKEND_API_VERSION`.
Version 0.15 of ggml inserted an op at ordinal 55, so `FLASH_ATTN_EXT` moved from 73 to 74
while `MUL_MAT` (29) held. A backend built on whisper.cpp 1.8.6 (ggml 0.14.0) and loaded
into 1.8.7 or later passes the version check and still offloads matmuls. **It silently
drops attention** [source-confirmed, ggml header diff, 2026-09-28]. The llama.cpp builds
b10558 (ggml 0.20.2), b11242 (0.25.3) and b11401 share ordinals [host-computed, header diff
of all 102 ops, 2026-10-04]. Current `ggml-rocket` checks each
ordinal it uses against the host's op names at load, and refuses a mismatch.

## Per-model notes

**Conditions:** RK3588 on a Turing RK1 (31 GiB), NPU clock ~816 MHz, unless an entry states
otherwise.

In these notes `-ub 2048` stands for `-b 2048 -ub 2048`. The combined recipe is
`ROCKET_QUANT_RESIDENT=auto` with `-ub 2048`. The residue is the part of the `-ub 2048` effect
outside dequantization amortization.

### Qwen3.5-0.8B (F16 / Q4_K_M)

- **Stack status:** prefill runs with the defaults. The `qwen35` architecture needs
  llama.cpp >= b9568.
- **Recommended flags:** the defaults. F16 fits any board, so no opt-in applies. At 0.8B the
  NPU is faster at prefill from between pp16 and pp32, where it measures 0.75x and 1.20x the
  CPU [HW sweep 2026-10-09].
- **Recommended sampling:** `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0`, the Qwen
  thinking-mode defaults. Add `--presence-penalty 1.0` or `--dry-multiplier 0.8` to curb
  repetition.
- **Behavior:** a reasoning model, emitting a `[Start thinking]` block before answering.
  At 0.8B it confabulates niche facts (mislabels the RK3588, swaps digits like 3588->3580)
  and falls into semantic reasoning loops. Token-level anti-repetition (dry,
  presence-penalty) does not stop a semantic loop: the model varies surface wording while
  repeating the meaning. For clean Q&A, disable thinking with a `/no_think` suffix on the
  message, or cap the context.
- **Best use:** a first model to confirm that a new installation runs and performs, and
  text-transform tasks (summarize, rewrite) over text you supply. At this size it is not a
  factual-recall chat model.
- **Q4_K_M variant:** the combined recipe takes prefill from 99.1 to 116.9 t/s at pp2048, and
  `-ub 2048` alone gains 1.130x [HW sweep 2026-08-28].

> [!NOTE]
> On this board the 0.8B model, in either precision, varies 8-11% between processes, where
> every larger model varies 0.4-2%. A comparison from one process per configuration gives no
> conclusion on this model.

### Llama-3.2-3B-Instruct (Q4_K_M / F16)

- **Recommended flags (Q4_K_M):** `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, without
  `-ub 2048`. Residency takes prefill from 39.2 to 57.7 t/s, with 193 weights resident
  (5232 MB) and 0 streamed. The combined recipe gains 1.212x, and `-ub 2048` alone 1.089x
  [HW sweep 2026-08-29/31]. Here `-ub 2048` is a 1.258x dequantization gain over a 0.863x
  residue, the pattern of the sub-4B models.
- **Recommended flags (F16):** `ROCKET_F16_RESIDENT=auto` takes prefill from 56.0 to 60.5
  t/s. The default admits resident weights only at K<=2048, and this model has K=3072, so
  without the option no weights are resident.
- **F16 against Q4_K_M:** the two GGUFs place identically (the same 193 weights, 5232 MB).
  At that placement the F16 GGUF is still faster, at 60.5 t/s. The quantized GGUF measures
  57.7 at the default `-ub` and 47.2 with `-ub 2048`. Residency removes the per-micro-batch
  dequantization, not the one-time decode. If the 6.4 GiB F16 file fits your disk and RAM
  budget, run it.

### Ministral-3-3B (Q4_K_M / F16)

- **Recommended flags (Q4_K_M):** `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, without
  `-ub 2048`. Residency takes prefill from 34.6 to 50.0 t/s, with 100% of the weights
  resident. The combined recipe gains 1.132x, and `-ub 2048` alone 1.038x
  [HW sweep 2026-08-29/31]. This model has the deepest loss outside dequantization among
  the sub-4B models, a 0.835x residue. Its combined-recipe figures are the smallest of the 3B models for that reason.
- **Recommended flags (F16):** `ROCKET_F16_RESIDENT=auto` takes prefill from 48.3 to 52.0
  t/s. This model has K=3072, as Llama-3.2-3B does, so without the option no weights are
  resident.

### Phi-4-mini (Q4_K_M / F16)

- **Recommended flags (Q4_K_M):** `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, without
  `-ub 2048`. Residency takes prefill from 35.9 to 54.2 t/s, with 100% of the weights
  resident. The combined recipe gains 1.211x, and `-ub 2048` alone 1.093x
  [HW sweep 2026-08-29/31]. The `-ub 2048` residue here is 0.864x.
- **Recommended flags (F16):** `ROCKET_F16_RESIDENT=auto` takes prefill from 50.2 to 54.4
  t/s. This model has K=3072, the same as the other 3B-class models.

### SmolVLM2-2.2B-Instruct (Q4_K_M, text path)

It is the only dense model measured on which the combined recipe gives no gain, and on which
`-ub 2048` alone is a loss:

- **Recommended flags:** `ROCKET_QUANT_RESIDENT=auto` at the default `-ub 512`, without
  `-ub 2048`. Residency takes prefill from 51.5 to 69.3 t/s, with 165 weights resident
  (2976 MB) and 0 streamed. The combined recipe measures 1.002x [HW sweep 2026-08-31].
- **Cause:** `-ub 2048` is a loss here, 0.941x [HW sweep 2026-08-29]. It is a 1.199x
  dequantization gain over a 0.789x residue, a cost of the larger micro-batch whose mechanism
  is not characterized. Residency removes the dequantization without that cost, so on this
  model it replaces `-ub 2048` rather than adding to it.
- **Host load:** these figures are for an idle host. With one core loaded by another
  process, `-ub 2048` measures 1.120x, so on a loaded host the combined recipe does not
  lose. Residency alone has no such condition.
- **Vision path:** separate, with its own resident SigLIP encoder. These figures are for the
  text prefill only.

### Qwen3.5-9B (Q4_K_M)

- **Recommended flags:** `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, without
  `-ub 2048`. Residency takes prefill from 19.1 to 33.5 t/s, with 200 weights resident
  (13184 MB) and 0 streamed. The combined recipe gains 1.661x, and `-ub 2048` alone
  1.42-1.43x [HW sweep 2026-08-29/31]. Its gain from residency is the largest from tuning
  among the models measured.
- **`-ub 2048` alone:** ~90% of its gain here is dequantization amortization, a 1.378x
  dequantization gain with a 1.038x residue. Residency removes the dequantization outright,
  so it gives more than raising `-ub`. On this model the residue is a gain, and the combined
  recipe still gives less than residency alone. Residency also removes the per-call pack and
  upload, and the default `-ub` presents four times as many calls to remove them from. The
  saving outweighs the 3.8% the residue gives back.
- **Footprint:** residency holds ~13.2 GB of fp16 weight cubes beside the 5.3 GiB GGUF. They
  fit a 31 GiB board with room to spare, and do not fit a 16 GiB one.

### Ministral-3-8B (Q4_K_M)

- **Recommended flags:** `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, without
  `-ub 2048`. Residency takes prefill from 17.8 to 29.4 t/s, with 100% of the weights
  resident. The combined recipe gains 1.325x, and `-ub 2048` alone 1.176x
  [HW sweep 2026-08-29/31].

### Gemma-4-12B-it

- **Stack status:** fp16 prefill runs coherently on the NPU, and greedy output is
  char-identical to CPU fp16, so prefill is faithful at 12B scale [HW sweep]. Native int8
  (`ROCKET_INT8=1 ROCKET_INT8_HADAMARD=1`) and int4 (`ROCKET_INT4=1`) from an F16 GGUF also run
  coherently, with greedy output char-identical to fp16. They reduce RAM use, so a larger
  model fits, and do not raise prefill throughput. At this operating point the NPU is
  dispatch-bound ([perf/not-mac-bound.md](perf/not-mac-bound.md)).
- **Recommended flags (Q4_K_M):** `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, without
  `-ub 2048`. Residency takes prefill from 12.8 to 17.5 t/s. The combined recipe gains
  1.322x, and `-ub 2048` alone 1.257x [HW sweep 2026-08-29/31].
- **F16 GGUF:** 22.2 GiB. With the defaults, zero weights are resident, and
  `ROCKET_F16_RESIDENT=auto` places 286 of 328.
- **Full residency:** possible, and not recommended. `ROCKET_N_THREADS=8` with
  `ROCKET_QUANT_RESIDENT_RESERVE_MB=6144` places all 328 weights, and measures 1.009x, which
  does not resolve from 1.00. The worker count it needs costs 0.987x, which cancels the gain
  from the last 42 weights. It also holds 2712 MB more on a board without swap.
- **Partial placement:** it is the only model in the tuning measurements that places part of
  its weights. On a 31 GiB board, residency places 83-87% at the default `-ub` (272-284 of
  328 weights, 17.1-17.9 GB), against 73-74% with `-ub 2048` (239-243, ~15 GB). Both
  configurations stop at the same 9535 MB `MemAvailable` reserve rather than at the 21129 MB
  resident-weight budget. At the default `-ub`, placement reaches that reserve 2.6-2.8 GB
  later, because `-ub 2048` uses that RAM for compute buffers. Raising `-ub` on this model
  therefore costs residency directly.
- **Comparing configurations:** placement varies from run to run on this model, so read each
  run's `[f16-resident]` outcome line. The default `-ub`'s +3.5% over the combined recipe
  gives the sign of the difference, not its size.
- **Evaluation:** it is a reasoning model, so wikitext perplexity is not a valid quality
  metric (the F16 reference PPL is itself ~545). Evaluate it with a greedy match against the
  CPU reference and a cosine probe, not PPL.
- **Recommended invocation:** the standard Gemma chat template, applied automatically from
  the GGUF. The resident, prepacked path gives the warm prefill figures.
- **Best use:** general Q&A, and prefill benchmarking. It does not loop or confabulate as the
  0.8B does.

### Phi-4-14B (Q4_K_M)

- **Recommended flags:** `-ub 2048`, which takes prefill from 10.9 to 14.3 t/s at pp2048
  [HW sweep 2026-08-29]. Residency was not measured: the fp16 resident image is ~29 GiB
  beside the GGUF, and does not fit a 31 GiB board.

### Qwen3.6-27B (Q4_K_M)

- **Recommended flags:** `-ub 2048`, which takes prefill from 6.1 to 9.3 t/s at pp2048
  [HW sweep 2026-08-29]. It is the largest `-ub 2048` gain measured, on the largest dense
  model. Residency was not measured, because the fp16 resident image is ~54 GiB. At ~6-9 t/s
  prefill, this model suits batch work on this board rather than interactive use.

### gpt-oss-20b (MXFP4, MoE)

gpt-oss-20b is a mixture-of-experts model whose settings interact. It has 24 layers and 32
experts with 4 active. Attention alternates windowed (128) and full, so half its layers are
full-attention. Its settings:

- **Recommended flags:** `-ub 2048`, and nothing for the experts. The routed-expert offload
  is on by default since 2026-08-27, and decides for itself which expert stacks to offload.
  It reserves a whole expert stack before it claims that stack's op, and leaves on the CPU
  what it cannot reserve. On this board it admits 63 of 72 stacks at 100% residency
  [HW sweep 2026-08-27]. The [supported-models table](#llms) gives the resulting figures
  against the CPU. The sections below give the `-ub` and residency detail.
- **Attention:** stays on the CPU, and must. The model has a learned attention sink per
  head, and the NPU FLASH_ATTN handler has no sink term, so the handler declines the
  offload. A handler that accepts it computes an incorrect, sink-less softmax past the `n_kv`
  threshold of 1024. Declining is both correct and +26% at pp2048. The signature is an NPU
  curve that collapses past ~1K context but is fine below it. If a benchmark shows that
  curve, suspect this class of defect.
- **Best use:** prefill on a mixture-of-experts model, with the experts on the NPU. Its
  expert stack only just fits a 31 GiB board alongside its own GGUF.

#### gpt-oss-20b stack status

Prefill runs coherently. Greedy output does not match the CPU reference on any NPU
configuration, including the one with the experts left on the CPU. On a ~1400-token passage at
`--temp 0 --seed 1`, the experts-on-CPU configuration first diverges at word 16 of the
generation. Every configuration with the expert offload first diverges at word 10
[HW sweep 2026-08-27].

All continuations stay coherent and on-topic, and that is the criterion for this model. Only
an early and large divergence points at the stack ([Greedy diff](#greedy-diff)). An fp16
prefill against an fp32 CPU reference flips greedy boundaries as a matter of course. The
expert offload moves the first divergence about six words earlier. A single 48-token
generation cannot separate that from one more flip in the same cascade.

It is a reasoning model (harmony format), so wikitext PPL is not a valid quality metric. Its
quantitative checks are the per-matmul cosine and the backend's MoE test, not the greedy
match. The cosine check is `ROCKET_MOE_COSINE=1`: real weights and real activations against an
fp64 CPU reference. It measures mean 0.999815 and min 0.998976 over 55 expert GEMMs on the
default placement. Decode stays on the CPU (~7.2 t/s, brisk for 20B because only ~3.6 B
params are active per token).

#### gpt-oss-20b `-ub` setting

On this model `-ub` matters only with the experts on the NPU.

Run the expert offload at `-ub 2048`. The figures for this model are at that setting unless
they name another. Two mechanisms make a smaller micro-batch cost the offload:

- The dense MXFP4 weights (attention projections, `lm_head`) are not in the expert cache,
  and still re-dequantize to fp16 per micro-batch. On a 2048-token prompt, `-ub 512`
  therefore runs that decode four times over.
- The router gives each expert only `n_tokens · n_used / n_expert` rows: 64 at `-ub 512`
  against 256 at `-ub 2048`. The per-expert overhead around the GEMM (dispatch, row
  gather, scatter, M-bucket padding) does not shrink with the row count. A quarter of the
  rows therefore costs close to the same overhead.

The micro-batch threshold `ROCKET_MOE_MIN_TOKENS` (default 512), which `ROCKET_MOE=1` does not
lift, is also at `-ub 512`, so the offload barely qualifies there.

| Experts | `-ub` | pp512 t/s | pp2048 t/s |
|---|---|---:|---:|
| On the NPU | 512 | 22.2 | 22.5 |
| On the CPU | 512 | 14.13 | 13.53 |
| On the NPU | 2048 | 22.0 | 28.1 |
| On the CPU | 2048 | 14.08 | 13.63 |

**Conditions:** [HW sweep 2026-08-27], llama.cpp b10558, governor `performance`.

At `-ub 512` the expert offload is about 1.6x the experts-on-CPU configuration at pp512, and
1.7x at pp2048. The two mechanisms together cost it ~20% at pp2048 and nothing at pp512, and
it does not collapse.

The fp16 streaming expert path falls to ~0.42x at `-ub 512`, because a smaller `-ub`
multiplies its per-expert dequantization. The native-quant path ingests each expert once and
removes that cost, so the fp16 path's figure does not apply to it.

With the experts on the CPU, the two `-ub` settings measure within 1% of each other: 13.53 t/s
at `-ub 512` and 13.63 at `-ub 2048`, at pp2048. The CPU-only configuration is unaffected by
`-ub`.

> [!NOTE]
> Compare a `-ub 512` figure only with another `-ub 512` figure. Set against a `-ub 2048`
> figure, it shows a regression that does not exist.

#### gpt-oss-20b routed experts

The routed-expert offload is on by default, and holds the experts resident on the NPU as
native int8. It gives the `-ub 2048` prefill in the table above and the ratios against the CPU
in the supported-models table. Across pp512-pp2048 it is 1.64x -> 2.08x the experts-on-CPU
configuration [HW sweep 2026-08-27, governor `performance`]. The fp16 expert path is a net loss,
4.59 t/s at pp512 and 10.18 at pp2048. It re-dequantizes every expert every micro-batch, ~75 ms
each, independent of the row count.

The native-quant path ingests each expert once and removes that cost. The one-time ingest
takes ~21 s inside the first prefill, per `llama_context` [HW sweep 2026-10-09].

A 2026-10-09 `librocketnpu` build runs the group-wise int8 readback as NEON. Against the build
before it, pp512 measures 22.78-22.99 -> 24.90-25.23 t/s, 1.093-1.098x within a pass
[HW sweep 2026-10-09, governor `ondemand`]. The ratios against the CPU were measured on the
earlier build.

A residency pre-flight keeps the offload a gain whatever the host's RAM: a stack it cannot
reserve stays on the CPU whole. A half-ingested stack takes the partial-residency loss
instead. An 82% resident stack measures 12.19 t/s at pp512, below the 14.11 with the experts
left on the CPU. Both are from one session at `-ub 2048` [HW sweep 2026-07-14]. Check the
split with `ROCKET_LOG_STDERR=1`. Under the default it reports 100% resident.

`ROCKET_MOE=1` overrides the pre-flight and both size thresholds. It is faster where the stack
nearly fits, and slower than the experts-on-CPU baseline where it does not. Use it to compare
configurations, not as a setting.

With RAM to spare, raise `ROCKET_MOE_CACHE_MB` instead. Raising it helps on this model only.
On an expert-dominated MoE, where the experts are most of the GGUF, raising it is a loss.
Before you apply it to another model, see [TUNING.md](TUNING.md) §"MoE routed experts on the
NPU".

This model clears the size thresholds at every prefill length, and DeepSeek-V2-Lite does not
(see its entry below). Its ratio does not transfer to another MoE.

### DeepSeek-V2-Lite (Q4_K_M, MoE + MLA)

DeepSeek-V2-Lite is a mixture-of-experts model whose expert-offload eligibility follows from
its architecture, not from a flag. It has 27 blocks (block 0 dense), 64 routed experts with 6
active plus 2 always-on shared, and MLA attention. Per token, ~2.4 B of its 15.7 B params are
active. It is a base model, so unlike gpt-oss its wikitext PPL is a valid quality metric. Its
settings:

- **Recommended flags:** `-ub 2048`, nothing else. The expert offload is on by default and
  decides per micro-batch.
- **Expert offload:** engages only past ~1250 tokens in a micro-batch. Below that, the
  offload is no faster than leaving the experts on the CPU (measured below). Two size
  thresholds decide each op. The one that decides here is the work per dispatch,
  `M_e · K · N`, where `M_e` = `n_tokens · n_used / n_expert`. DeepSeek's 6-of-64 routing
  over a 2048×1408 expert gives 2.88x less work per dispatch than gpt-oss's 4-of-32 over a
  2880×2880 one, at the same row count.
- **Expert offload, measured:** at `M_e` = 72 the offload measures 0.94x the experts-on-CPU
  configuration over four pairs, a loss, where gpt-oss at `M_e` = 64 measures 1.64x. Past the threshold it
  gains: 1.22x at `M_e` = 144 and 1.31x at `M_e` = 192 [HW sweep 2026-08-27, ~816 MHz
  pinned]. Whole-model prefill under the default is 26.0 t/s at pp512 and 28.4 at pp2048, in
  the same measurement series.
  A 2026-10-09 build measures pp2048 at 32.69-32.83 t/s, against 28.82-29.99 for the build
  before it [HW sweep 2026-10-09, governor `ondemand`]. Both its faster expert pack and its
  NEON readback contribute, because `llama-bench`'s timed repetitions include part of the
  ingest. The `M_e` = 72 and 96 results were measured on the earlier build.
- **Other MoE models:** gpt-oss's MoE ratio does not transfer to this model, nor this one's
  to a third. A per-expert row count does not separate them: `M_e` = 96 is 1.77x on gpt-oss
  and roughly parity here.
- **Attention (MLA):** stays on the CPU. The FLASH_ATTN handler accepts DK≠DV, and the
  primitive measures cosine 1.000000 at MLA's head dims. The DeepSeek attention path has not
  been run on the device. Below the expert threshold, three things still run on the NPU: the
  large MLA projections, the 2 shared experts, and `lm_head`. They give a modest gain on their
  own.
- **Best use:** prompts long enough for the expert offload to engage.

#### DeepSeek-V2-Lite faithfulness

Faithfulness is measured as differential wikitext PPL at `-c 2048`. A context of 2048 gives
`M_e` = 192, so the expert offload is active (4800 experts resident, 0 streamed).

| Configuration | PPL |
|---|---:|
| CPU | 5.3063 |
| NPU, `ROCKET_MOE=0` | 5.2906 |
| NPU, default | 5.2842 |

The expert offload's own effect, the difference between the two NPU configurations, is
−0.121%, indistinguishable from zero. The default against the CPU, −0.416%, includes the
dense fp16 path's own −0.296%.

> [!NOTE]
> Isolate the expert offload by differencing the two NPU configurations, not the default
> against the CPU. Compare the paired per-chunk difference, never the final PPLs. The
> absolute error bar is ±2.5% and resolves nothing.

### Qwen3-30B-A3B (Q4_K_M, MoE)

Qwen3-30B-A3B is a mixture-of-experts model on which every tuned configuration is slower than
the defaults [HW sweep 2026-08-30]:

- **Recommended flags:** none. The defaults (14.7 t/s at pp2048) are faster than every tuned
  configuration: `-ub 2048` is 0.908x, `ROCKET_MOE=1` is 0.726x and `ROCKET_MOE=0` is 0.908x.
- **Expert offload:** declines this model correctly. Every expert dispatch at `-ub` 2048 and
  below has at most 201 MMAC of work, under the 340 MMAC threshold
  (`ROCKET_MOE_MIN_WORK`). The automatic placement places nothing, so the default and
  `ROCKET_MOE=0` are the same configuration reached two ways (0.908x against 0.908x). Routed
  experts are 29 of its 30.5 B parameters, and they cannot be held resident on a 31 GiB
  board.
- **`ROCKET_MOE=1`:** forcing it places 62%, streams the rest, and costs 0.800x against the
  experts-on-CPU baseline. The residency pre-flight exists to avoid that loss, here measured
  end to end. Do not set it on this model.
- **Sizing:** do not size this model by its total parameter count. Only ~3 B parameters are
  active per token. The dense >8 B band (1.18-1.53x from `-ub 2048`) does not apply, and its
  own result is below even the 3B dense band.
- **Best use:** chat on a board with plenty of RAM, where its 14.7 t/s prefill and small
  active set suit the board. Tuning gives no gain on it.

### ASR models (transcribe.cpp)

These models run through transcribe.cpp (a ggml multi-model ASR host), not llama.cpp. They load
the same `ggml-rocket` `.so` (`GGML_BACKEND_PATH`), and the NPU does the same job: the
encoder and, on long audio, the batched decode-prefill. Autoregressive decode stays on the
CPU.

Faithfulness differs from the LLM case. Offloading only the encoder does not by itself keep the
transcript identical to the CPU's. On a 120 s hard conversational clip, SenseVoice and Fun-ASR
drift slightly and MOSS emits 32 segments against 33, while Voxtral-mini stays char-identical
[HW sweep]. A model whose decoder cross-attention offloads can also diverge late, as fp16
accumulation perturbs greedy decoding. A late divergence of that kind is expected, and is not
an NPU defect. Cross-model speed and quality tables are in
[perf/benchmarks.md](perf/benchmarks.md#asr-multi-model-transcribecpp-via-ggml-rocket).

#### Flags for every ASR model

Pin to the A76 cluster (`taskset -c 4-7 --threads 4`), because the A55 cluster is a
straggler, and set `ROCKET_KACC=1`. Use Q8_0 by default: for ASR it ties or beats F16 on both
the CPU and the NPU, at half the RAM. These models are less matmul-dominated than LLM
prefill, so the per-micro-batch dequantization cost is small. CPU-side decode is
memory-bound, and Q8 moves half the bytes.

The build and run pitfalls are in [perf/benchmarks.md](perf/benchmarks.md) §"Build and run".
The shared/DL build needs `-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`, and `main.cpp` needs a
one-line backend-init patch.

#### Voxtral-mini 3B

Voxtral-mini pairs a Whisper-large-v3 encoder with a Ministral-3B AR decoder. It gives the
best transcript and is the only translator here (de->en, ja->en). CPU and NPU output is
char-identical throughout (CPU==NPU). The NPU gain is largest on long audio, at 1.57x: the
encoder runs 1.82x and the 1500-token audio prefill offloads at 1.46x. On short clips only
the encoder offloads (jfk 1.26x). It is slow (0.55x realtime, a 3B decode), and it has no
diarization and no timestamps.

#### MOSS 0.9B

MOSS pairs a Whisper-Medium encoder with a Qwen3-0.6B AR decoder, and injects audio as KV
tokens. It is the best diarizer by a wide margin (33 fine segments + timestamps, clean
two-speaker split, never degenerates). It is decode-bound: the encoder offloads at 1.64x,
but the M=1 AR decode (90% of runtime) cannot offload. The whole pipeline on the NPU is
therefore only 1.08x over a fair A76-pinned CPU baseline. No cheap NPU optimization remains:
speculative decoding is unsupported, and quantization and thread count are already tuned.

#### Granite-Speech-2B

Granite-Speech-2B pairs a conformer encoder with an LLM cross-attention decoder. It comes
in three variants:

| Variant | Notes |
|---|---|
| base | Fastest, cleanest text. Char-identical CPU==NPU at F16; at Q8, one word differs on a hard 120 s clip. 1.68x on long audio, since the cross-attention decode offloads ~1.9x, unusually for an audio-LLM |
| plus | Diarization: 8 coarse speaker turns and no timestamps, coarser than MOSS |
| NAR | An iterative non-autoregressive editor, so its decode is larger than base's, ~1.4x slower and rougher |

The base variant can drop spans on hard conversational audio. Cross-attention decode is not
guaranteed to match the CPU. Greedy can diverge, and plus even loops on the CPU while staying
coherent on the NPU.

#### SenseVoice-small 234M

SenseVoice-small pairs a SAN-M encoder with one CTC head. It is the only true single-pass
model: decode is 48 ms even on 120 s of audio, so the run is 100% encoder-offloaded. It is
the fastest (6.2x realtime) and the lowest in quality. It has a 30 s window, garbles
longer audio into a run-on, and has no diarization or timestamps. The encoder is
dispatch-bound at short audio (jfk NPU==CPU 1.00x) and wins 1.26x at 120 s.

#### Fun-ASR-nano 800M

Fun-ASR-nano pairs the same SAN-M encoder with a bundled Qwen3-0.6B AR decoder, and covers
31 languages. Despite the shared encoder it is autoregressive, so the decoder limits the
gain: 1.15x at 120 s against SenseVoice's 1.26x, for only marginally better text. The AR
decoder is pure overhead that the NPU cannot recover.

#### Parakeet CTC/TDT-0.6B

Parakeet is a FastConformer. At F16, A76-pinned, it runs ~4.6x realtime and 1.37x over the
CPU. Its convolutions limit the gain: its convolution modules do not offload (no conv path in
ggml-rocket) and stay on the CPU. `ROCKET_F16_RESIDENT` slows it (single encoder pass).
Quality is rougher on hard conversational audio (LibriSpeech clean-speech bias). It has no
diarization.

#### Parakeet on short utterances

For short utterances such as voice commands, Parakeet needs `ROCKET_MIN_M=16`. Through
whisper.cpp's `parakeet-cli` the encoder runs 12.5 rows a second of audio, so a 1-5 s command
is under the default threshold, and nothing offloads. At 16, every encoder GEMM offloads from
1.3 s up. It ties the CPU at 1.3-1.8 s and runs 1.18-1.73x faster at 2-6 s
[HW sweep 2026-09-30, mainline, TDT 0.6B v3 F16]. It frees 34-61% of the CPU, and the
transcript stays byte-identical.

Under CPU contention the offloaded stream slows more than the CPU one. Set
`OMP_WAIT_POLICY=PASSIVE` with it
([perf/asr-cpu-relief.md](perf/asr-cpu-relief.md#short-utterances-below-the-f16-floor)).

## Appendix

### Validating a new model

Two questions about a model on the NPU are independent:

1. **Does the NPU stack run it faithfully?** First establish that the NPU configuration
   offloaded anything, then measure what the offload cost. The three checks below do both.
2. **Is the model good at the task?** The answer depends on sampling and model choice, and is
   independent of question 1.

#### Offload check

The NPU configuration must offload before it can be compared. A quantized GGUF sends a
prefill shorter than `ROCKET_MIN_M_QUANT` (default 512 tokens) to the CPU, so a short prompt
compares the CPU with itself. Use a prompt at least that long, or set
`ROCKET_MIN_M_QUANT=0` for the check. Run the NPU configuration with
`ROCKET_MM_PROFILE=1 ROCKET_LOG_STDERR=1`. Its exit line counts the jobs it submitted. A
count of zero, or no line, means the configuration computed on the CPU.

#### Differential perplexity

Differential perplexity is the measure. Run `llama-perplexity` over the same text with and
without `GGML_BACKEND_PATH`, and compare the ratio. The defects found in the stack cost
perplexity. One per-tensor int8 output scale costs 1.11-2.26x over two models, simulated
on the host from their own activations ([chips/rk3576.md](chips/rk3576.md)). Perplexity
measures that loss, and fluent output does not rule it out.

#### Greedy diff

A greedy diff is a quick screen, not the verdict. Run `--temp 0 --seed 1` on the same
prompt with and without the backend:

```bash
P="Summarize the following in one sentence: <... a few hundred tokens ...>"
# NPU
sudo -E GGML_BACKEND_PATH=/path/to/ggml-rocket/build-dl/libggml-rocket.so ROCKET_KACC=1 \
  taskset 0xf0 /path/to/llama-cli -m /path/to/models/<model>.gguf -p "$P" -n 200 \
  --temp 0 --seed 1 -no-cnv > npu.txt 2>/dev/null
# CPU (drop GGML_BACKEND_PATH)
taskset 0xf0 /path/to/llama-cli -m /path/to/models/<model>.gguf -p "$P" -n 200 \
  --temp 0 --seed 1 -no-cnv > cpu.txt 2>/dev/null
diff npu.txt cpu.txt && echo IDENTICAL || echo DIVERGED
```

Use a few-hundred-token prompt, so the prefill is large enough to offload to the NPU. A
prompt below the offload threshold runs prefill on the CPU regardless, and proves nothing.

fp16 prefill against fp32 eventually flips one greedy boundary, so a late divergence is
expected, and an early, large one points at the stack. Identical output does not show
faithfulness: a configuration that offloaded nothing produces it too.

A coherent-but-wrong answer, or a repetition loop, is usually a model or sampling issue.
Token salad (broken grammar, non-words) points at the stack. Neither reading clears the
numerics, which is what the perplexity ratio is for.

### Template for a new row

```
### <model> (<quant>)

- Stack status: <prefill faithful? greedy NPU-vs-CPU result + provenance>; <warm pp512/pp2048 vs CPU>; <llama.cpp build / arch caveats>.
- Recommended flags: <the workload-conditional opt-ins for this model, e.g. ROCKET_QUANT_RESIDENT=auto at the default -ub for a quant GGUF whose fp16 fits RAM, and -b 2048 -ub 2048 only where residency is unavailable; no MoE flag, since the expert offload is default-on and gates itself. Defaults (KACC/REUSE/ASYM/FA) are on; do not restate them. See TUNING.md>.
- Recommended sampling: <temp/top-p/top-k/min-p + any anti-repetition>.
- Behavior: <reasoning vs not; loops/confabulation; thinking on/off>.
- Best use: <canary / transform / chat / bench>.
```

## Evidence

### Supported models

Per-pass readings of residency at the default `-ub`, against the defaults:

| Model | Pass 1 | Pass 2 | Pass 3 |
|---|---:|---:|---:|
| Qwen3.5-9B Q4_K_M | 1.747 | 1.737 | 1.771 |
| Ministral-3-8B Q4_K_M | 1.630 | 1.650 | 1.674 |
| Gemma-4-12B-it Q4_K_M | 1.402 | 1.382 | 1.319 |

The Gemma-4-12B-it per-pass spread is 6.3%, the widest of the seven models measured with and
without `-ub 2048`. Its full-residency configuration's three passes straddle 1.00.

On Qwen3-30B-A3B the automatic placement placed nothing in 3 of 3 passes.

### Baseline invocation

The pinning figures are from rotated interleaved passes on the RK1.

### Per-model notes

The split of the `-ub 2048` effect into a dequantization gain and a residue is from a
mechanism control. The control is described in
[perf/quant-prefill-microbatch.md](perf/quant-prefill-microbatch.md) §"Dequant component of the
`-ub` lever". In that control the full `-ub 2048` effect on Qwen3.5-9B is 1.430x.

The SmolVLM2 `-ub 2048` loss resolves in two runs: 0.941x in the matrix run and 0.946x in the
mechanism-control run [HW sweep 2026-08-31]. The 1.120x with one core loaded compares with
the 0.946x idle reading.

The gpt-oss-20b MoE test named under its stack status is `test-rocket-moe`. The experts-on-CPU
rows of its `-ub` table are one process each, in one session.

The DeepSeek-V2-Lite expert-offload difference of −0.121% is over 8 paired chunks, at 1.20 se.
Its absolute PPL is not comparable to an archived 8.24, which was measured at `-c 512`.

DeepSeek-V2-Lite's whole-model prefill figures are from the accept-boundary series of
2026-08-27, three repetitions with the configurations interleaved. A single process on the same
day and llama.cpp build measured 25.9 t/s at pp512 and 28.1 at pp2048, 1.52x the CPU
[HW sweep 2026-08-27]. It ran the binary with the per-expert row threshold.

### Revision history

Readings that later measurements replace:

| Measured | Quantity | Value | Conditions |
|---|---|---|---|
| 2026-07-14 | Qwen3.5-0.8B F16 prefill against the CPU | 0.36x at pp16, 0.83x at pp64, 1.04x at pp96, 1.15x at pp128, crossing at ~86 rows | Warm, ~816 MHz, the build of that date. The 2026-10-09 measurement in §"NPU role and operating point" crosses at 21-25 rows |
| 2026-07-03 | gpt-oss-20b, experts on the CPU, pp2048 | 13.11 t/s at `-ub 512`, 11.31 at `-ub 2048` | A build that offloaded gpt-oss's attention with no sink term, which the handler declines from 2026-07-14. The 2026-08-27 table in §"gpt-oss-20b `-ub` setting" measures 13.53 and 13.63 |
