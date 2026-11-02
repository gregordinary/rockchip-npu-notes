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
| Qwen3.5-0.8B | F16 | Supported. Prefill is faithful: greedy output is token-identical to the CPU [HW sweep 2026-06-28] | Defaults | | ~1.44x at pp512 | Warm [HW sweep 2026-06-27/28] |
| Qwen3.5-0.8B | Q4_K_M | Supported | `-b 2048 -ub 2048` with `ROCKET_QUANT_RESIDENT=auto` | 1.179x | | pp2048 [HW sweep 2026-08-28] |
| Llama-3.2-3B-Instruct | Q4_K_M | Supported | `ROCKET_QUANT_RESIDENT=auto` at the default `-ub` | 1.472x | | [HW sweep 2026-08-29/31] |
| Llama-3.2-3B-Instruct | F16 | Supported | `ROCKET_F16_RESIDENT=auto` | 1.081x | | pp2048 [HW sweep 2026-08-29] |
| Ministral-3-3B | Q4_K_M | Supported | `ROCKET_QUANT_RESIDENT=auto` at the default `-ub` | 1.445x | | [HW sweep 2026-08-29/31] |
| Ministral-3-3B | F16 | Supported | `ROCKET_F16_RESIDENT=auto` | 1.078x | | pp2048 [HW sweep 2026-08-29] |
| Phi-4-mini | Q4_K_M | Supported | `ROCKET_QUANT_RESIDENT=auto` at the default `-ub` | 1.508x | | [HW sweep 2026-08-29/31] |
| Phi-4-mini | F16 | Supported | `ROCKET_F16_RESIDENT=auto` | 1.084x | | pp2048 [HW sweep 2026-08-29] |
| SmolVLM2-2.2B-Instruct, text path | Q4_K_M | Supported | `ROCKET_QUANT_RESIDENT=auto` at the default `-ub 512` | 1.346x | | [HW sweep 2026-08-31] |
| Qwen3.5-9B | Q4_K_M | Supported | `ROCKET_QUANT_RESIDENT=auto` at the default `-ub` | 1.752x | | [HW sweep 2026-08-29/31] |
| Ministral-3-8B | Q4_K_M | Supported | `ROCKET_QUANT_RESIDENT=auto` at the default `-ub` | 1.651x | | [HW sweep 2026-08-29/31] |
| Gemma-4-12B-it | Q4_K_M | Supported | `ROCKET_QUANT_RESIDENT=auto` at the default `-ub` | 1.368x | | [HW sweep 2026-08-29/31] |
| Gemma-4-12B-it | F16 | Supported | `ROCKET_F16_RESIDENT=auto` | 1.0614x | | pp2048 [HW sweep 2026-09-01] |
| Phi-4-14B | Q4_K_M | Supported | `-b 2048 -ub 2048` | 1.308x | | pp2048 [HW sweep 2026-08-29] |
| Qwen3.6-27B | Q4_K_M | Supported | `-b 2048 -ub 2048` | 1.532x | | pp2048 [HW sweep 2026-08-29] |
| gpt-oss-20b, MoE | MXFP4 | Supported. Greedy output differs from the CPU ([stack status](#gpt-oss-20b-stack-status)) | `-b 2048 -ub 2048`. The expert offload is on by default | | 2.29x at pp2048 | [HW sweep 2026-11-01] |
| DeepSeek-V2-Lite, MoE and MLA | Q4_K_M | Supported | `-b 2048 -ub 2048`. The expert offload is on by default | | 1.83x at pp2048 | [HW sweep 2026-11-01] |
| Qwen3-30B-A3B, MoE | Q4_K_M | Supported | Defaults | Every tuned configuration is slower | | [HW sweep 2026-08-30] |

**Conditions:** RK3588 on a Turing RK1 (31 GiB), NPU clock ~816 MHz (the 600 MHz request). The
rows dated 2026-11-01 ran at 928 MHz (the 900 MHz request) on llama.cpp b11539, with the CPU
governor at `performance` and the CPU backend's weight repack on.

The gain is prefill with the recommended options against prefill with the defaults, both on the
NPU. A ratio against the CPU takes the 8-thread CPU backend on the same GGUF as its baseline.
[perf/benchmarks.md](perf/benchmarks.md) has the per-model results against the CPU.

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
- An F16 prefill offloads from 32 rows (`ROCKET_MIN_M`), just above the 21-25-row crossover at
  which the NPU overtakes the CPU. A quantized prefill offloads from 512 rows
  (`ROCKET_MIN_M_QUANT`) ([TUNING.md](TUNING.md#rocket_min_m-and-rocket_min_m_quant)).
- Below the crossover, the fixed per-call cost (host scatter and pack, submit, readback, the
  per-submit dispatch overhead) dominates, and the CPU is faster. Do not judge prefill speed
  from a short interactive prompt, whose `Prompt: N t/s` figure mostly measures overhead. See
  [perf/asr-cpu-relief.md](perf/asr-cpu-relief.md) §"The default `ROCKET_MIN_M`" and
  [perf/not-mac-bound.md](perf/not-mac-bound.md).
- Discard the first, cold-clock run. The NPU idles at 198 MHz and ramps under load, so a
  cold run measures ~15% low. A short job finishes before the clock reaches its operating rate.
  Compare warm runs. `llama-bench` discards the cold run, so use `-r 3`.

## Baseline invocation

**Conditions:** RK3588 on a Turing RK1.

Every NPU run needs these settings:

- Set `GGML_BACKEND_PATH` to the absolute path of `ggml-rocket/build-dl/libggml-rocket.so`.
- Build the backend against the host application's own ggml.
- Run under `sudo -E`. Plain `sudo` strips the environment, and `-E` keeps
  `GGML_BACKEND_PATH` and the `ROCKET_*` settings alongside the `/dev/accel` privilege.
- Pin to the A76 big cores with `taskset 0xf0 -t 4` (cpus 4-7 on the RK3588). Pinning gives
  1.06-1.13x on NPU prefill over three models, most on the smallest, with or without resident
  weights [HW sweep 2026-09-01]. It also speeds up CPU decode, 1.32-4.22x the `-t 8` rate over
  fifteen GGUFs on llama.cpp b11539 [HW sweep 2026-11-01].
- For a quantized GGUF whose fp16 form fits in RAM, set `ROCKET_QUANT_RESIDENT=auto` and keep
  the default `-ub`. Where the fp16 form does not fit, `-b 2048 -ub 2048` amortizes the
  per-micro-batch dequantization to fp16. Its gain depends on the model: 0.91-1.53x over eleven
  models, and a loss on two [HW sweep 2026-08-29/30, ~816 MHz]. Use the model's own row.
- Confirm that the NPU ran the prefill. Prepend `ROCKET_MM_PROFILE=1 ROCKET_LOG_STDERR=1`, and
  look for a `ROCKET profile total(ms):` line on stderr. `llama-bench` silences the backend's
  lines without `ROCKET_LOG_STDERR=1`. A `failed to load` line at startup means a wrong
  `GGML_BACKEND_PATH`, and the run falls back to the CPU with no other sign.
- Stage models on external storage. On the RK1, the eMMC root and `/tmp` are small.

> [!NOTE]
> A backend built against another ggml can stop offloading attention with no error.
> [ggml op ordinals](#ggml-op-ordinals) gives the cause.

These settings are on by default:

- fp16 K-accumulation (`ROCKET_KACC`)
- CBUF weight streaming (`ROCKET_MM_STREAM`), the plan the fp16 matmul takes by default
- CBUF operand reuse (`ROCKET_REUSE`), which picks DATA_REUSE or WEIGHT_REUSE per shape
- Asymmetric tiling (`ROCKET_MM_ASYM`), which applies only to the whole-tile plan that
  `ROCKET_MM_STREAM=0` or a tile override selects
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
otherwise. A figure dated 2026-11-01 ran at 928 MHz (the 900 MHz request) on llama.cpp b11539.
Its ratio against the CPU takes the 8-thread CPU backend with its weight repack on, and its
decode rate is at `taskset 0xf0 -t 4`.

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
  `-ub 2048`. All 193 weights are resident (5232 MB). The configuration prefills at 92.2 t/s
  at pp2048, 3.9x the repacked CPU [HW sweep 2026-11-02]. On the ~816 MHz series residency
  took prefill from 39.2 to 57.7 t/s. The combined recipe gains 1.212x, and `-ub 2048` alone 1.089x
  [HW sweep 2026-08-29/31]. Here `-ub 2048` is a 1.258x dequantization gain over a 0.863x
  residue, the pattern of the sub-4B models.
- **Recommended flags (F16):** `ROCKET_F16_RESIDENT=auto` takes prefill from 85.0 to 97.8 t/s
  at pp2048, 1.14-1.16x [HW sweep 2026-11-02]. The default admits resident weights only at
  K<=2048, and this model has K=3072, so without the option no weights are resident.
- **F16 against Q4_K_M:** the two GGUFs place identically (the same 193 weights, 5232 MB).
  At that placement the F16 GGUF is still faster, 97.8 t/s against 92.2
  [HW sweep 2026-11-02]. Residency removes the per-micro-batch
  dequantization, not the one-time decode. If the 6.4 GiB F16 file fits your disk and RAM
  budget, run it.

### Ministral-3-3B (Q4_K_M / F16)

- **Recommended flags (Q4_K_M):** `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, without
  `-ub 2048`. All the weights are resident, and the configuration prefills at 81.1 t/s at
  pp2048, 3.7-3.8x the repacked CPU [HW sweep 2026-11-02]. On the ~816 MHz series residency
  took prefill from 34.6 to 50.0 t/s. The combined recipe gains 1.132x, and `-ub 2048` alone 1.038x
  [HW sweep 2026-08-29/31]. This model has the deepest loss outside dequantization among
  the sub-4B models, a 0.835x residue. Its combined-recipe figures are the smallest of the 3B models for that reason.
- **Recommended flags (F16):** `ROCKET_F16_RESIDENT=auto` takes prefill from 75.0 to 86.1 t/s
  at pp2048, 1.15x [HW sweep 2026-11-02]. This model has K=3072, as Llama-3.2-3B does, so
  without the option no weights are resident.

### Phi-4-mini (Q4_K_M / F16)

- **Recommended flags (Q4_K_M):** `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, without
  `-ub 2048`. All the weights are resident, and the configuration prefills at 87.4 t/s at
  pp2048, 4.2x the repacked CPU [HW sweep 2026-11-02]. On the ~816 MHz series residency took
  prefill from 35.9 to 54.2 t/s. The combined recipe gains 1.211x, and `-ub 2048` alone 1.093x
  [HW sweep 2026-08-29/31]. The `-ub 2048` residue here is 0.864x.
- **Recommended flags (F16):** `ROCKET_F16_RESIDENT=auto` takes prefill from 76.0 to 88.4 t/s
  at pp2048, 1.16x [HW sweep 2026-11-02]. This model has K=3072, the same as the other
  3B-class models.

### SmolVLM2-2.2B-Instruct (Q4_K_M, text path)

It is the only dense model measured on which the combined recipe gives no gain, and on which
`-ub 2048` alone is a loss:

- **Recommended flags:** `ROCKET_QUANT_RESIDENT=auto` at the default `-ub 512`, without
  `-ub 2048`. All 165 weights are resident (2976 MB), and the configuration prefills at
  125.1 t/s at pp2048, 3.2x the repacked CPU [HW sweep 2026-11-02]. On the ~816 MHz series
  residency took prefill from 51.5 to 69.3 t/s, and the combined recipe measured 1.002x
  [HW sweep 2026-08-31].
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
  `-ub 2048`. All 197 weights are resident (12896 MB), and the configuration prefills at
  41.3 t/s at pp2048, 3.9x the repacked CPU [HW sweep 2026-11-02]. On the ~816 MHz series
  residency took prefill from 19.1 to 33.5 t/s. The combined recipe gains 1.661x, and `-ub 2048` alone
  1.42-1.43x [HW sweep 2026-08-29/31]. Its gain from residency is the largest from tuning
  among the models measured.
- **`-ub 2048` alone:** ~90% of its gain here is dequantization amortization, a 1.378x
  dequantization gain with a 1.038x residue. Residency removes the dequantization outright,
  so it gives more than raising `-ub`. On this model the residue is a gain, and the combined
  recipe still gives less than residency alone
  ([TUNING.md](TUNING.md#residency-in-place-of--ub-2048) gives the cause).
- **Footprint:** residency holds ~13 GB of fp16 weight cubes beside the 5.3 GiB GGUF. They
  fit a 31 GiB board with room to spare, but do not fit a 16 GiB one.

### Ministral-3-8B (Q4_K_M)

- **Recommended flags:** `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, without
  `-ub 2048`. All 235 weights are resident. The configuration prefills at 45.1 t/s at pp2048,
  4.7x the CPU [HW sweep 2026-11-01]. The F16 GGUF, whose weights stream at the defaults,
  prefills at 39.6 t/s.
- **Other configurations:** against the streamed default, the combined recipe gains 1.325x,
  and `-ub 2048` alone 1.176x [HW sweep 2026-08-29/31].

### Gemma-4-12B-it

- **Stack status:** fp16 prefill runs coherently on the NPU, and greedy output is
  char-identical to CPU fp16, so prefill is faithful at 12B scale [HW sweep]. Native int8
  (`ROCKET_INT8=1 ROCKET_INT8_HADAMARD=1`) and int4 (`ROCKET_INT4=1`) from an F16 GGUF also run
  coherently, with greedy output char-identical to fp16. They reduce RAM use, so a larger
  model fits, but do not raise prefill throughput. At this operating point the NPU is
  dispatch-bound ([perf/not-mac-bound.md](perf/not-mac-bound.md)).
- **Recommended flags (Q4_K_M):** `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, without
  `-ub 2048`. With 92-95% of its weights resident, the configuration prefills at 24.5 t/s at
  pp2048, 3.8x the CPU and about the F16 GGUF's 24.6 t/s [HW sweep 2026-11-01]. Against the
  streamed default, the combined recipe gains 1.322x, and `-ub 2048` alone 1.257x
  [HW sweep 2026-08-29/31].
- **F16 GGUF:** 22.2 GiB. With the defaults, zero weights are resident, and
  `ROCKET_F16_RESIDENT=auto` places 286 of 328.
- **Full residency:** possible, and not recommended. `ROCKET_N_THREADS=8` with
  `ROCKET_QUANT_RESIDENT_RESERVE_MB=6144` places all 328 weights, and measures 1.009x, which
  does not resolve from 1.00. The worker count it needs costs 0.987x, which cancels the gain
  from the last 42 weights. It also holds 2712 MB more on a board without swap.
- **Partial placement:** it is the only model in the tuning measurements that places part of
  its weights. On a 31 GiB board, residency places 83-87% at the default `-ub` (272-284 of
  328 weights, 17.1-17.9 GB), against 73-74% with `-ub 2048` (239-243, ~15 GB)
  [HW sweep 2026-08-29/31]. Both
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

- **Recommended flags:** `-ub 2048`, which gains 1.308x over the default `-ub` at pp2048
  [HW sweep 2026-08-29]. Residency was not measured: the fp16 resident image is ~29 GiB
  beside the GGUF, and does not fit a 31 GiB board.
- **Prefill:** 21.9 t/s at pp2048 with `-ub 2048`, 4.0x the CPU [HW sweep 2026-11-01].

### Qwen3.6-27B (Q4_K_M)

- **Recommended flags:** `-ub 2048`, which gains 1.532x over the default `-ub` at pp2048
  [HW sweep 2026-08-29]. It is the largest `-ub 2048` gain measured, on the largest dense
  model. Residency was not measured, because the fp16 resident image is ~54 GiB.
- **Prefill and decode:** 11.5 t/s prefill at pp2048 with `-ub 2048`, 3.8x the CPU, and 1.3 t/s
  decode [HW sweep 2026-11-01]. At those rates this model suits batch work on this board rather
  than interactive use.

### gpt-oss-20b (MXFP4, MoE)

gpt-oss-20b is a mixture-of-experts model whose settings interact. It has 24 layers and 32
experts with 4 active. Attention alternates windowed (128) and full, so half its layers are
full-attention. Its settings:

- **Recommended flags:** `-ub 2048`, and nothing for the experts. The routed-expert offload
  is on by default, and decides for itself which expert stacks to offload ([TUNING.md](TUNING.md)
  §"MoE routed experts on the NPU"). On this board it admits 62-63 of 72 stacks, with every
  placed expert resident [HW sweep 2026-08-27, 2026-11-01]. The [supported-models table](#llms)
  gives the resulting figure against the CPU. The sections below give the `-ub` and residency
  detail.
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
default placement. Decode stays on the CPU, at 9.1 t/s [HW sweep 2026-11-01], brisk for 20B
because only ~3.6 B params are active per token.

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
1.7x at pp2048. The two mechanisms together cost it ~20% at pp2048 and nothing at pp512, but
it does not collapse.

The fp16 streaming expert path falls to ~0.42x at `-ub 512`, because a smaller `-ub`
multiplies its per-expert dequantization. The native-quant path ingests each expert once and
removes that cost, so the fp16 path's figure does not apply to it.

> [!NOTE]
> Compare a `-ub 512` figure only with another `-ub 512` figure. Set against a `-ub 2048`
> figure, it shows a regression that does not exist.

#### gpt-oss-20b routed experts

The routed-expert offload is on by default, and holds the experts resident on the NPU as
native int8. It gives the `-ub 2048` prefill in the table above and the ratio against the CPU
in the supported-models table. Across pp512-pp2048 it is 1.64x -> 2.08x the experts-on-CPU
configuration [HW sweep 2026-08-27, governor `performance`]. The fp16 expert path is a net loss,
4.59 t/s at pp512 and 10.18 at pp2048. It re-dequantizes every expert every micro-batch, ~75 ms
each, independent of the row count.

The native-quant path's one-time ingest takes ~21 s inside the first prefill, per
`llama_context` [HW sweep 2026-10-09].

A residency pre-flight keeps the offload a gain whatever the host's RAM: a stack it cannot
reserve stays on the CPU whole. Without it, a half-ingested stack takes the partial-residency
loss. At 82% of its experts resident, gpt-oss measures 12.19 t/s at pp512, below the 14.11 with
the experts left on the CPU. Both are from one session at `-ub 2048` [HW sweep 2026-07-14].
Check the split with `ROCKET_LOG_STDERR=1`. Under the default it reports 100% resident.

`ROCKET_MOE=1` overrides the pre-flight and both size thresholds. It is faster where the stack
nearly fits, and slower than the experts-on-CPU baseline where it does not. Use it to compare
configurations, not as a setting.

With RAM to spare, raise `ROCKET_MOE_CACHE_MB` instead. Of the two models measured, raising it
helps on this one. On an expert-dominated MoE, where the experts are most of the GGUF, raising
it is a loss.
Before you apply it to another model, see [TUNING.md](TUNING.md) §"MoE routed experts on the
NPU".

This model clears the size thresholds from a 512-token micro-batch up, and DeepSeek-V2-Lite
only past ~1250 tokens (see its entry below). Its ratio does not transfer to another MoE.

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
  pinned].
- **Prefill:** 43.4 t/s at pp2048 under the default, 1.83x the CPU, with every routed expert
  resident [HW sweep 2026-11-01].
- **Other MoE models:** gpt-oss's MoE ratio does not transfer to this model, nor this one's
  to a third. A per-expert row count does not separate them: `M_e` = 96 is 1.77x on gpt-oss
  and roughly parity here.
- **Attention (MLA):** the FLASH_ATTN handler accepts DK≠DV, and the primitive measures cosine
  1.000000 at MLA's head dims. The backend's `supports_op` admits an op whose DK and DV differ
  [source-confirmed: `ggml-rocket.cpp`]. MLA attention can therefore offload with flash
  attention on (`-fa`) at `n_kv` >= 1024, and stays on the CPU without `-fa`. At
  `-b 2048 -ub 2048`, `-fa on` prefills 1.04-1.05x as fast as `-fa off` at pp2048 and
  1.10-1.11x at pp4096 [HW sweep 2026-11-02, 928 MHz]. The output under `-fa` was not compared
  against the CPU's on this model.
- **Below the expert threshold:** three things still run on the NPU: the large MLA
  projections, the 2 shared experts, and `lm_head`. They give a modest gain on their own.
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
straggler. Use Q8_0 by default: for ASR it ties or beats F16 on both
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

For short utterances such as voice commands, set `ROCKET_MIN_M=16` in the ASR process. Below
2.6 s of audio, the default threshold leaves most of the encoder on the CPU. At 16, every
encoder GEMM offloads from 1.3 s up, with the transcript byte-identical [HW sweep 2026-09-30,
TDT 0.6B v3 F16]. Under CPU contention the offloaded stream slows more than the CPU one, so set
`OMP_WAIT_POLICY=PASSIVE` with it. [TUNING.md](TUNING.md#asr-whisper-whispercpp) gives the
measured gain.

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
perplexity. One per-tensor int8 output scale raises perplexity to 1.11-2.26x the fp32 run's
over two models, simulated on the host from their own activations
([perf/rk3576-w8a8-llm.md](perf/rk3576-w8a8-llm.md#output-scale-of-the-int8-entry)).
Perplexity measures that loss, and fluent output does not rule it out.

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
- Recommended flags: <the workload-conditional opt-ins for this model, e.g. ROCKET_QUANT_RESIDENT=auto at the default -ub for a quant GGUF whose fp16 fits RAM, and -b 2048 -ub 2048 only where residency is unavailable; no MoE flag, since the expert offload is default-on and gates itself. Defaults (KACC/STREAM/REUSE/FA) are on; do not restate them. See TUNING.md>.
- Recommended sampling: <temp/top-p/top-k/min-p + any anti-repetition>.
- Behavior: <reasoning vs not; loops/confabulation; thinking on/off>.
- Best use: <canary / transform / chat / bench>.
```

## Evidence

### Supported models

Each tuning figure dated 2026-08-28 to 2026-08-31 is from three passes per model, with the
configuration order rotated. The figures dated 2026-11-01 are one series of two passes, the
second with the GGUF order and the configuration order reversed. Each ratio against the CPU is
the mean of its two per-pass ratios. `ggml-rocket`'s `MEASUREMENTS.md` gives the series' method.

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

The prefill pinning figures are from rotated interleaved passes on the RK1. The decode figures
are one pass of `tg64 -r 3` per configuration, each from a memory reset
([perf/data/tuning-matrix.md](perf/data/tuning-matrix.md)).

### Per-model notes

The split of the `-ub 2048` effect into a dequantization gain and a residue is from a
mechanism control. The control is described in
[perf/quant-prefill-microbatch.md](perf/quant-prefill-microbatch.md) §"Dequant component of
`-ub`". In that control the full `-ub 2048` effect on Qwen3.5-9B is 1.430x.

The SmolVLM2 `-ub 2048` loss resolves in two runs: 0.941x in the matrix run and 0.946x in the
mechanism-control run [HW sweep 2026-08-31]. The 1.120x with one core loaded compares with
the 0.946x idle reading.

The gpt-oss-20b MoE test named under its stack status is `test-rocket-moe`. The experts-on-CPU
rows of its `-ub` table are one process each, in one session.

The DeepSeek-V2-Lite expert-offload difference of −0.121% is over 8 paired chunks, at 1.20 se.
Its absolute PPL is not comparable to an archived 8.24, which was measured at `-c 512`.

The 2026-11-02 residency figures ran at NPU 928 MHz, governor `performance`, llama.cpp
b11539 at `-t 8`. The NPU arms ran `--repack 0`, and the CPU arms the repack's default.
Each model ran three passes with the arm order rotated, each process after a memory reset and a
discarded warm-up process, `llama-bench -r 2` at pp2048. Each ratio pairs the arms within one
pass.

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
