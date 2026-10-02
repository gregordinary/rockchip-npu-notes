# Model notes: running LLMs on the rocket NPU stack

This file is a living, per-model record of how each model behaves on the FOSS NPU stack,
meaning stock llama.cpp plus the `ggml-rocket` backend. Each row covers four things:

- Whether that model's prefill runs faithfully on the NPU
- The sampling settings it wants
- Behavioral quirks worth knowing
- What it is good for

When you test a new model, add a row. Speech-to-text models run through transcribe.cpp on
the same backend. They get their own subsection at the end of the per-model notes.

For which flags to enable for a given workload, see [TUNING.md](TUNING.md). It covers
use-case, precision, prompt size and the RAM or disk each opt-in needs. This file records
per-model behavior, and that one is the decision guide.

## Stack faithfulness and model quality

Before you judge any model on the NPU, separate two independent questions:

1. **Does the NPU stack run it faithfully?** First establish that the NPU arm offloaded
   anything, then measure what the offload cost. The three steps follow this list.
2. **Is the model any good at the task?** That is a sampling and model-choice question,
   fully independent of question 1.

### Offload check

The NPU arm must offload before it can be compared. A quantized GGUF sends a prefill
shorter than `ROCKET_MIN_M_QUANT` (default 512 tokens) to the CPU, so a short prompt
compares the CPU with itself. Use a prompt at least that long, or set
`ROCKET_MIN_M_QUANT=0` for the check. Run the NPU arm with
`ROCKET_MM_PROFILE=1 ROCKET_LOG_STDERR=1`. Its exit line counts the jobs it submitted. A
count of zero, or no line, means the arm computed on the CPU.

### Differential perplexity

Differential perplexity is the measure. Run `llama-perplexity` over the same text with and
without `GGML_BACKEND_PATH`, and read the ratio. The defects this stack has met cost
perplexity. One per-tensor int8 output scale costs 1.11-2.26x over two models, simulated
on the host from their own activations ([chips/rk3576.md](chips/rk3576.md)). Perplexity
scores that loss, and fluent output does not rule it out.

### Greedy diff

A greedy diff is a quick screen, not the verdict. Run `--temp 0 --seed 1` on the same
prompt with and without the backend. The command pair is under
[Baseline invocation](#baseline-invocation). fp16 prefill against fp32 eventually flips
one greedy boundary, so a late divergence is expected, and an early, large one points at
the stack. **Identical output is a warning rather than a pass**: it is also what an arm
that offloaded nothing produces.

A coherent-but-wrong answer, or a repetition loop, is usually a model or sampling issue.
Token salad (broken grammar, non-words) points at the stack. Neither reading clears the
numerics, which is what the perplexity ratio is for.

## NPU role and operating point

- Prefill (prompt eval, batched GEMM) runs on the NPU, and decode (token generation, M=1
  GEMV) stays on the CPU. Generation throughput is therefore CPU-bound and reads
  ~identical with or without the backend loaded. The NPU is a prefill engine.
- NPU prefill wins only at scale (pp512-pp2048). Short prompts of tens of tokens are
  below break-even. With few MACs to amortize it over, the fixed per-call cost (host
  scatter and pack, submit, readback, the dispatch floor) dominates, and the CPU is
  faster. Do not judge prefill speed from a short interactive prompt, whose
  `Prompt: N t/s` figure is mostly measuring overhead. See
  [perf/not-mac-bound.md](perf/not-mac-bound.md).
- Discard the first, cold-clock run. The NPU idles at 200 MHz and ramps under load, so a
  cold run reads ~15% low. A short job finishes before the clock reaches 600 MHz. Compare
  warm runs. `llama-bench` discards the cold run, so use `-r 3`.

## Baseline invocation

The knobs every NPU run wants:

- Set `GGML_BACKEND_PATH` to the absolute path of
  `ggml-rocket/build-dl/libggml-rocket.so`. A wrong path **silently falls back to
  CPU-only**, with no error beyond a `failed to load` line.
- Build the backend against the host application's own ggml. A mismatched build can
  **silently drop attention**. The reason is under "ggml op ordinals" at the end of this
  section.
- `sudo -E`: plain `sudo` strips the environment. The `-E` flag keeps `GGML_BACKEND_PATH`
  and the `ROCKET_*` knobs alive alongside `/dev/accel` privilege.
- `taskset 0xf0`: pin to the A76 big cores (cpus 4-7 on the RK3588). Pinning is worth
  1.05-1.13x on NPU prefill over three models, largest on the smallest
  [HW sweep 2026-09-01, RK1, rotated interleaved passes]. Residency does not gate it.
  **Leave a decode-heavy run unpinned**, because F16 decode wants all eight cores for
  their bandwidth.
- **Quantized GGUF:** consider `-b 2048 -ub 2048`. A quantized GGUF re-dequantizes to fp16
  per micro-batch, and the larger micro-batch amortizes that. What it buys is
  model-dependent: 0.91-1.53x measured over eleven models, and a loss on two
  [HW sweep 2026-08-29/30, RK1, 600 MHz]. Read the model's own row.
  `ROCKET_QUANT_RESIDENT=auto` is the stronger lever where the fp16 image fits RAM.
- **Confirm that the NPU ran the prefill:** prepend `ROCKET_MM_PROFILE=1` and look for a
  `ROCKET profile total(ms):` line on stderr. Check also that startup prints no
  `failed to load` line.
- Stage models on external storage: an eMMC root and a `/tmp` tmpfs are usually too small.

The operating-mode knobs are on by default, so you need not set them:

- fp16 K-accumulation (`ROCKET_KACC`)
- DATA_REUSE (`ROCKET_REUSE=2`)
- Asymmetric tiling (`ROCKET_MM_ASYM`)
- Attention offload (`ROCKET_FLASH_ATTN`)

The workload-specific opt-ins are in [TUNING.md](TUNING.md).

To validate a model against question 1:

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

Use a few-hundred-token prompt so the prefill is large enough to route to the NPU. A prompt
below the routing floor runs prefill on the CPU regardless, and proves nothing.

### ggml op ordinals

The op enum in ggml shifts between versions, while `GGML_BACKEND_API_VERSION` stays at 2.
Version 0.15 of ggml inserted an op at ordinal 55, so `FLASH_ATTN_EXT` moved from 73 to 74
while `MUL_MAT` (29) held. A backend built on whisper.cpp 1.8.6 (ggml 0.14.0) and loaded
into 1.8.7 or later passes the version check and still offloads matmuls. **It silently
drops attention** [source-confirmed, ggml header diff, 2026-09-28]. The llama.cpp builds
b10558 (ggml 0.20.2) and b11242 (0.25.3) share ordinals. Current `ggml-rocket` checks each
ordinal it uses against the host's op names at load, and refuses a mismatch.

## Per-model notes

### Qwen3.5-0.8B (F16)

- **Stack status:** prefill runs out of the box, and greedy output is token-identical to
  CPU [HW sweep, 2026-06-28, RK1, 600 MHz]. So prefill is faithful for this arch. Warm
  prefill is ~1.44x CPU at pp512 [HW sweep]. The `qwen35` arch needs llama.cpp >= b9568.
- **Recommended flags:** defaults only. F16 fits any board trivially, and at 0.8B the NPU
  wins prefill only past ~pp96. There is nothing to opt into. Its role is a
  throughput/bring-up canary.
- **Recommended sampling:** `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0`, the Qwen
  thinking-mode defaults. Add `--presence-penalty 1.0` or `--dry-multiplier 0.8` to curb
  repetition.
- **Behavior:** a reasoning model, emitting a `[Start thinking]` block before answering.
  At 0.8B it confabulates niche facts (mislabels the RK3588, swaps digits like 3588->3580)
  and falls into semantic reasoning loops. Token-level anti-repetition (dry,
  presence-penalty) does not stop a semantic loop: the model varies surface wording while
  repeating the meaning. For clean Q&A, disable thinking with a `/no_think` suffix on the
  message, or cap the context.
- **Best use:** a bring-up / throughput canary, and text-transform tasks (summarize,
  rewrite) over text you supply. It is not a factual-recall chat model at this size.
- **Q4_K_M variant:** `-b 2048 -ub 2048` measures 1.130x and
  `+ROCKET_QUANT_RESIDENT=auto` 1.179x over stock at pp2048 (99.1 -> 116.9 t/s)
  [HW sweep 2026-08-28, RK1, 600 MHz, three rotated passes]. This model class, 0.8B in
  both precisions, is also the one that carries this board's per-process spread. Single
  arms have spanned 8-11% across processes where every larger model resolves to 0.4-2%,
  so a one-process A/B on it settles nothing.

### Llama-3.2-3B-Instruct (Q4_K_M / F16)

- **Recommended flags (Q4_K_M):** `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`. **Do
  not stack `-b 2048 -ub 2048`.** Unstacked residency reads 1.472x (39.2 -> 57.7 t/s, 193
  weights resident, 5232 MB, 0 streamed). The stacked recipe reads 1.212x there and the
  `-ub` lever alone 1.089x [HW sweep 2026-08-29/31, RK1, 600 MHz, three rotated passes
  each]. The mechanism control decomposes the `-ub` flag here as a 1.258x dequant win over
  a real 0.863x non-dequant loss, which is the sub-4B class pattern.
- **Recommended flags (F16):** `ROCKET_F16_RESIDENT=auto`, worth 1.081x (56.0 -> 60.5
  t/s). The default admits resident weights only at K<=2048 and this model has K=3072, so
  without the knob the resident route gets nothing.
- **Route comparison:** the two routes place identically (the same 193 weights, 5232 MB),
  and the F16 GGUF is still faster: 60.5 t/s against the quant GGUF's 47.2 at the same
  placement. Residency removes the per-micro-batch dequant, not the one-time decode. If
  the 6.4 GiB fits your disk and RAM budget, run the F16 file.

### Ministral-3-3B (Q4_K_M / F16)

- **Recommended flags (Q4_K_M):** `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`. **Do
  not stack `-b 2048 -ub 2048`.** Unstacked residency reads 1.445x (34.6 -> 50.0 t/s,
  100% resident). The stacked recipe reads 1.132x and the `-ub` arm alone 1.038x
  [HW sweep 2026-08-29/31, RK1, 600 MHz, three rotated passes each]. This model has the
  class's deepest non-dequant `-ub 2048` loss, a 0.835x residue under the mechanism
  control. That is why its stacked numbers are the smallest in the 3B group.
- **Recommended flags (F16):** `ROCKET_F16_RESIDENT=auto`, worth 1.078x (48.3 -> 52.0
  t/s), at the same K=3072 gate as Llama-3.2-3B, so the knob turns the whole route on.

### Phi-4-mini (Q4_K_M / F16)

- **Recommended flags (Q4_K_M):** `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`. **Do
  not stack `-b 2048 -ub 2048`.** Unstacked residency reads 1.508x (35.9 -> 54.2 t/s,
  100% resident). The stacked recipe reads 1.211x and the `-ub` arm alone 1.093x
  [HW sweep 2026-08-29/31, RK1, 600 MHz, three rotated passes each]. The `-ub` flag's
  non-dequant residue here is 0.864x.
- **Recommended flags (F16):** `ROCKET_F16_RESIDENT=auto`, worth 1.084x (50.2 -> 54.4
  t/s), at K=3072, the same gate as the other 3B-class models.

### SmolVLM2-2.2B-Instruct (Q4_K_M, text path)

This is the one dense model measured where the standard quant recipe is a net zero, and
half of that recipe is a loss. Its rows are why the guide's recipes are per-model:

- **Recommended flags:** `ROCKET_QUANT_RESIDENT=auto` at the default `-ub 512`, and **do
  not set `-b 2048 -ub 2048`**. Unstacked residency reads 1.346x (51.5 -> 69.3 t/s). The
  stacked recipe reads 1.002x [HW sweep 2026-08-31, RK1, 600 MHz, three rotated passes,
  165 weights resident (2976 MB), 0 streamed].
- **Why:** `-b 2048 -ub 2048` is a resolved loss here, 0.941x in the matrix run and 0.946x
  in the mechanism-control run [HW sweep 2026-08-29, 2026-08-31]. The mechanism control
  decomposes it as a 1.199x dequant-amortization win masking a real 0.789x non-dequant
  cost of the larger micro-batch, whose mechanism is open. Residency removes the dequant
  without paying that cost, so it replaces the `-ub` lever on this model instead of
  stacking on it.
- **Scope:** an idle host. One busy core flips the `-ub` flag's sign back to a win
  (0.946x -> 1.120x measured under an accidental one-core load, a leaked spinner), so on a
  loaded box the stacked recipe stops losing. The unstacked one needs no such caveat.
- **Vision path:** separate, with its own resident SigLIP encoder work. These rows are
  the text prefill only.

### Qwen3.5-9B (Q4_K_M)

- **Recommended flags:** `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`. **Do not
  stack `-b 2048 -ub 2048`.** Unstacked residency reads 1.752x (19.1 -> 33.5 t/s, 200
  weights resident, 13184 MB, 0 streamed). The stacked recipe reads 1.661x there and the
  `-ub` lever alone 1.42-1.43x [HW sweep 2026-08-29/31, RK1, 600 MHz, three rotated
  passes each]. The per-pass readings are 1.747 / 1.737 / 1.771. This is the largest
  tuning win in the matrix.
- **`-ub` lever:** ~90% dequant amortization here (mechanism control: 1.430 = 1.378
  dequant x 1.038 residue). That is why residency beats raising `-ub`: it removes the
  dequant outright. This model is also the one whose non-dequant residue is a win, which
  is the case where the residue argument predicts stacking holds. Stacking does not hold.
  Residency removes the per-call pack and upload as well, and the default `-ub` presents
  four times as many calls to remove them from. That is worth more than the 3.8% the
  residue gives back.
- **Footprint:** the resident arm holds ~13.2 GB of fp16 cubes beside the 5.3 GiB GGUF.
  It fits a 31 GiB board with room, not a 16 GiB one.

### Ministral-3-8B (Q4_K_M)

- **Recommended flags:** `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`. **Do not
  stack `-b 2048 -ub 2048`.** Unstacked residency reads 1.651x (17.8 -> 29.4 t/s, 100%
  resident). The stacked recipe reads 1.325x and the `-ub` arm alone 1.176x
  [HW sweep 2026-08-29/31, RK1, 600 MHz, three rotated passes each]. The per-pass
  readings are 1.630 / 1.650 / 1.674.

### Gemma-4-12B-it

- **Stack status:** fp16 prefill runs coherently on the NPU, and greedy output is
  char-identical to CPU fp16, so prefill is faithful at 12B scale [HW sweep]. Native int8
  (`ROCKET_INT8=1 ROCKET_INT8_HADAMARD=1`) and int4 (`ROCKET_INT4=1`) also run coherently
  and greedy char-identical to fp16. Those are RAM and model-fit levers, not prefill-speed
  levers. At this operating point the NPU is dispatch-bound, so quantization buys
  footprint rather than speed.
- **Recommended flags (Q4_K_M):** `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`. **Do
  not stack `-b 2048 -ub 2048`.** Unstacked residency reads 1.368x (12.8 -> 17.5 t/s).
  The stacked recipe reads 1.322x, and the `-ub` lever alone 1.257x. The per-pass
  readings are 1.402 / 1.382 / 1.319 [HW sweep 2026-08-29/31, RK1, 600 MHz, three rotated
  passes each].
- **F16 GGUF:** 22.2 GiB. Stock places zero weights, and `ROCKET_F16_RESIDENT=auto`
  places 286 of 328 for 1.0614x.
- **Full residency:** reachable, and not recommended. `ROCKET_N_THREADS=8` with
  `ROCKET_QUANT_RESIDENT_RESERVE_MB=6144` places 328 of 328, and reads 1.009x straddling
  1.00 over three passes. The worker count it needs costs 0.987x, cancelling what the last
  42 weights buy, and it holds 2712 MB more on a swapless board.
- **Partial placement:** this is the matrix's only partly placed model, and the recipe is
  why that matters. On a 31 GiB board the resident route places 83-87% unstacked (272-284
  of 328 weights, 17.1-17.9 GB) against 73-74% stacked (239-243, ~15 GB). Both arms stop
  at the same 9535 MB `MemAvailable` reserve floor rather than at the 21129 MB
  resident-weight budget. The unstacked arm reaches that floor 2.6-2.8 GB later, because
  `-b 2048 -ub 2048` spends that RAM on compute buffers. Raising `-ub` here costs
  residency directly.
- **Reading a comparison:** read the `[f16-resident]` outcome line per arm on this model,
  because placement varies run to run. Treat its +3.5% over stacked as a sign and not a
  size. The per-pass spread is 6.3%, the widest of the seven models measured both ways.
- **Native `ROCKET_INT8` and `ROCKET_INT4`:** both run coherently, but they are RAM-fit
  levers from an F16 GGUF, not prefill-speed levers.
- **Eval note:** it is a reasoning model, so **wikitext perplexity is not a valid quality
  metric** (the F16 reference PPL is itself ~545). Evaluate with greedy-match against the
  CPU reference and a cosine probe, not PPL.
- **Recommended invocation:** the standard Gemma chat template, applied automatically from
  the GGUF. The resident and prepacked path is what gives the warm prefill numbers.
- **Best use:** the primary LLM-pillar target, for real Q&A and prefill benchmarking. It
  has no loop or confabulation pathology like the 0.8B.

### Phi-4-14B (Q4_K_M)

- **Recommended flags:** `-b 2048 -ub 2048`, worth 1.308x at pp2048 (10.9 -> 14.3 t/s)
  [HW sweep 2026-08-29, RK1, 600 MHz, three rotated passes]. There is no residency arm:
  the fp16 resident image is ~29 GiB beside the GGUF and does not fit a 31 GiB board.

### Qwen3.6-27B (Q4_K_M)

- **Recommended flags:** `-b 2048 -ub 2048`, worth 1.532x at pp2048 (6.1 -> 9.3 t/s).
  That is the largest `-ub` lever measured, on the largest dense model
  [HW sweep 2026-08-29, RK1, 600 MHz, three rotated passes]. There is no residency arm,
  because the fp16 resident image is ~54 GiB. At ~6-9 t/s prefill this is a batch model on
  this board, not an interactive one.

### gpt-oss-20b (MXFP4, MoE)

This is the mixture-of-experts model, and the one whose settings pull against each other.
It has 24 layers and 32 experts with 4 active. Attention alternates windowed (128) and
full, so half its layers are full-attention. Its settings:

- **Recommended flags:** `-b 2048 -ub 2048`, and nothing for the experts. The
  routed-expert offload is on by default since 2026-08-27 and gates itself. It reserves a
  whole expert stack before it claims that stack's op, and leaves on the CPU what it
  cannot reserve. On this board that admits 63 of 72 stacks at 100% residency, which is
  2.38x the CPU at pp2048. The `-ub` and residency detail follows this list.
- **Attention:** stays on the CPU, and must. The model carries a learned attention sink
  per head, and the NPU FLASH_ATTN handler has no sink term, so the handler declines the
  offload. A handler that accepts it computes a sink-less (wrong) softmax past the `n_kv`
  floor of 1024. Declining is both correct and +26% at pp2048. The signature is an NPU
  curve that collapses past ~1K context but is fine below it. If a benchmark shows that
  curve, suspect this class of bug.
- **Best use:** the MoE showcase, and the residency stress case. Its expert stack only
  just fits a 31 GiB board alongside its own GGUF, so it is where partial residency gets
  exercised.

#### gpt-oss-20b stack status

Prefill runs coherently. **Greedy output does not match the CPU reference on any NPU
arm**, including the one with the experts left on the CPU. On a ~1400-token passage at
`--temp 0 --seed 1`, the experts-on-CPU arm first diverges at word 16 of the generation.
Every arm carrying the expert offload first diverges at word 10 [HW sweep 2026-08-27].

All continuations stay coherent and on-topic, which is the criterion that matters here.
This file's own rule is that only early and large divergence points at the stack. An fp16
prefill against an fp32 CPU reference flips greedy boundaries as a matter of course. What
the expert route adds is about six words earlier. A single 48-token generation cannot
separate that from one more flip in the same cascade.

It is a reasoning model (harmony format), so **wikitext PPL is not a valid quality
metric**. The quantitative gates for this model are the per-matmul cosine probe and
`test-rocket-moe`, not the greedy match. The probe is `ROCKET_MOE_COSINE=1`: real weights
and real activations against an fp64 CPU reference. It reads mean 0.999815 and min
0.998976 over 55 expert GEMMs on the shipped placement. Decode stays on the CPU as always
(~7.2 t/s, brisk for 20B because only ~3.6 B params are active per token).

#### gpt-oss-20b `-ub` setting

The `-ub` setting pulls two ways, and you must choose deliberately.

Run the MoE expert route at `-b 2048 -ub 2048`. Every number here was measured there, and
two mechanisms say a smaller micro-batch costs it:

- The dense MXFP4 weights (attention projections, `lm_head`) are not in the expert cache,
  and still re-dequantize to fp16 per micro-batch. So `-ub 512` runs that decode four
  times over on a 2048-token prompt.
- The router gives each expert only `n_tokens · n_used / n_expert` rows: 64 at `-ub 512`
  against 256 at `-ub 2048`. The per-expert overhead around the GEMM (dispatch, row
  gather, scatter, M-bucket padding) does not shrink with the row count. So a quarter of
  the rows buys close to the same overhead.

The micro-batch gate `ROCKET_MOE_MIN_TOKENS` (default 512), which `ROCKET_MOE=1` does not
lift, also sits right at `-ub 512`, so offload barely qualifies.

Measured at `-ub 512`, the expert route reads 22.2 t/s at pp512 and 22.5 at pp2048. With
the experts on the CPU it reads 14.13 and 13.53, so the route is about 1.6x and 1.7x. At
`-ub 2048` it reads 22.0 and 28.1. So the two mechanisms together cost ~20% at pp2048 and
nothing at pp512, and there is no collapse.

The often-quoted "`-ub 512` collapses MoE to ~0.42x" is the fp16 streaming route, whose
per-expert dequant is what `-ub` multiplies. Native-quant ingests each expert once and
deletes exactly that cost, so that reasoning does not transfer to it.

But **`-ub 2048` makes the dense graph slower on this model**. NPU-default reads 13.11
t/s at `-ub 512` against 11.31 at `-ub 2048` (pp2048). The CPU does not care either way.

So there is no single best `-ub` here: it depends on whether the experts are on the NPU.
**Never compare a `-ub 512` number against a `-ub 2048` one.** That mistake reads as a
regression that does not exist.

#### gpt-oss-20b routed experts

The routed experts are worth 2.38x the CPU, and the offload is on by default. Prefill
reads 22.0 t/s at pp512 and 28.1 at pp2048, with the experts held resident on the NPU as
native int8. That is 1.81x and 2.38x the CPU, and 1.64x -> 2.08x over the experts-on-CPU
arm across pp512-pp2048. The fp16 expert route is a net loss (4.59 / 10.18). It
re-dequantizes every expert every micro-batch, ~75 ms each, independent of the row count.
The native route ingests each expert once and deletes that tax, at a one-time ~36 s
ingest inside the first prefill, per `llama_context`.

A residency pre-flight keeps the sign of the offload independent of the host's RAM. A
stack it cannot reserve stays on the CPU whole. The alternative is a half-ingested stack
and the partial-residency loss. An 82% resident stack reads 12.19 at pp512, below the
14.11 you get leaving the experts on the CPU. Check the split with `ROCKET_LOG_STDERR=1`.
Under the default it reads 100% resident.

`ROCKET_MOE=1` overrides the pre-flight and both size floors. It is faster where the
stack nearly fits, and **below the experts-on-CPU baseline where it does not**. So it is
the A/B arm, not a setting. With RAM to spare, raise `ROCKET_MOE_CACHE_MB` instead, **on
this model**. Raising it is a loss on an expert-dominated MoE, where the experts are most
of the GGUF. Before you carry the direction to another model, see [TUNING.md](TUNING.md)
§"MoE routed experts on the NPU".

This model clears the size floors at every prefill length, and DeepSeek-V2-Lite does not.
See its section, and do not read this ratio across to another MoE.

### DeepSeek-V2-Lite (Q4_K_M, MoE + MLA)

This is the second MoE, and the one that shows expert-offload eligibility is a property of
the architecture, not of the flag. It has 27 blocks (block 0 dense), 64 routed experts
with 6 active plus 2 always-on shared, and MLA attention. Per token, ~2.4 B of its 15.7 B
params are active. It is a base model, so unlike gpt-oss its wikitext PPL is a valid
quality metric. Its settings:

- **Recommended flags:** `-b 2048 -ub 2048`, nothing else. The expert offload is on by
  default and decides per micro-batch.
- **Expert offload:** only past ~1250 tokens in a micro-batch, and that is correct. Two
  size floors gate each op. The binding one here is the work a dispatch carries,
  `M_e · K · N` where `M_e` = `n_tokens · n_used / n_expert`. DeepSeek's 6-of-64 routing
  over a 2048×1408 expert carries 2.88x less work per dispatch than gpt-oss's 4-of-32 over
  a 2880×2880 one, at the same row count.
- **Expert offload, measured:** at `M_e` = 72 the offload measures 0.95x the
  experts-on-CPU arm, a real loss, while gpt-oss at `M_e` = 64 measures 1.64x. Past the
  floor it wins: 1.22x at `M_e` = 144 and 1.31x at `M_e` = 192
  [HW sweep 2026-08-27, 600 MHz pinned]. Whole-model prefill under the default is
  25.9 -> 28.1 t/s at pp512 -> pp2048, which is 1.33x -> 1.52x the CPU.
- **Do not read gpt-oss's MoE ratio across to this model, or this one's across to a
  third.** A per-expert row count cannot separate them: `M_e` = 96 is 1.77x on gpt-oss and
  roughly parity here.
- **Attention (MLA):** stays on the CPU. The FLASH_ATTN gate accepts DK≠DV, and the
  primitive is bit-faithful for MLA. The DeepSeek path is not yet exercised on-device, so
  attention is CPU-side here. Below the expert floor, three things still reach the NPU:
  the large MLA projections, the 2 shared experts, and `lm_head`. That is a modest but
  real win on its own.
- **Best use:** the second MoE architecture, and the one to re-run whenever the placement
  floors move.

#### DeepSeek-V2-Lite faithfulness on the shipped placement

The measure is differential wikitext PPL at `-c 2048`. That context puts `M_e` = 192, so
the expert route is active (4800 experts resident, 0 streamed). The arms read 5.3063 on
the CPU, 5.2906 with `ROCKET_MOE=0` and 5.2842 on the default.

**Isolate the expert route by differencing the two NPU arms**, not the default against
the CPU. That difference reads −0.121% (8 paired chunks, 1.20 se), indistinguishable from
zero. Default-vs-CPU (−0.416%) would credit the experts with the dense fp16 path's own
−0.296%. Read the paired per-chunk difference, never the finals. The absolute error bar
is ±2.5% and resolves nothing. Absolute PPL is not comparable to the archived 8.24, which
was `-c 512`.

### Qwen3-30B-A3B (Q4_K_M, MoE)

This is the third MoE, and the model where every tuned arm loses. Its rows are the
measured case for leaving the defaults alone [HW sweep 2026-08-30, RK1, 600 MHz, three
rotated passes]:

- **Recommended flags:** none. Stock (14.7 t/s at pp2048) beats every tuned arm:
  `-b 2048 -ub 2048` is 0.908x, `ROCKET_MOE=1` is 0.726x and `ROCKET_MOE=0` is 0.908x.
- **Expert route:** correctly declines this model, because every expert dispatch at `-ub`
  2048 and below carries at most 201 MMAC, under the 340 MMAC floor
  (`ROCKET_MOE_MIN_WORK`). The auto placement places nothing (3 of 3 passes), so the
  default and `ROCKET_MOE=0` are the same configuration reached two ways (0.908x against
  0.908x). Routed experts are 29 of its 30.5 B parameters, and they cannot be held resident
  on a 31 GiB board. **Forcing `ROCKET_MOE=1` places
  62%, streams the rest, and costs 0.800x against the experts-on-CPU baseline.** That is
  the loss the residency pre-flight exists to avoid, measured end to end. Do not set it
  here.
- **Do not size this model by total parameter count.** Only ~3 B are active per token.
  The dense >8 B band (1.18-1.53x on the `-ub` lever) does not apply, and its own row is
  below even the 3B dense band.
- **Best use:** the guard-rail regression model, which tests that the auto placement
  declines what it must decline. It is also for bulk-RAM chat, where its 14.7 t/s prefill
  and small active set suit the board. Expect nothing from tuning.

### Speech-to-text models (transcribe.cpp)

These models run through transcribe.cpp (a ggml multi-STT host), not llama.cpp. They load
the same `ggml-rocket` `.so` (`GGML_BACKEND_PATH`), and the NPU does the same job: the
encoder and, on long audio, the batched decode-prefill. Autoregressive decode stays on the
CPU.

The faithfulness question differs from the LLM one. Encoder-only offload is bit-faithful
(CPU==NPU). A model whose decoder cross-attn offloads can diverge late, as fp16
accumulation perturbs greedy decoding. That is expected, and is not an NPU bug. Cross-model
speed and quality tables are in
[perf/benchmarks.md](perf/benchmarks.md#speech-to-text-multi-model-transcribecpp-via-ggml-rocket).

#### Flags for every STT model

Pin to the A76 cluster (`taskset -c 4-7 --threads 4`), because the A55 cluster is a
straggler, and set `ROCKET_KACC=1`. Q8_0 is the default. For STT it ties or beats F16 on
both CPU and NPU at half the RAM. These models are less matmul-dominated than LLM prefill,
so the per-micro-batch dequant tax is small. CPU-side decode is memory-bound, and Q8 moves
half the bytes.

The build and run traps are in the benchmarks doc. The shared/DL build needs
`-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`, and `main.cpp` needs a one-line backend-init
patch.

#### Voxtral-mini 3B

Voxtral-mini pairs a Whisper-large-v3 encoder with a Ministral-3B AR decoder. It gives the
best transcript and is the only translator here (de->en, ja->en). CPU and NPU output is
char-identical throughout (CPU==NPU). The NPU win is big on long audio, at 1.57x: the
encoder runs 1.82x and the 1500-token audio prefill offloads at 1.46x. On short clips only
the encoder offloads (jfk 1.26x). It is slow (0.55x realtime, a 3B decode), and it has no
diarization and no timestamps.

Best use: highest-accuracy transcription and translation, where latency is not the
concern.

#### MOSS 0.9B

MOSS pairs a Whisper-Medium encoder with a Qwen3-0.6B AR decoder, and injects audio as KV
tokens. It is the best diarizer by a wide margin (33 fine segments + timestamps, clean
two-speaker split, never degenerates). But it is decode-bound. The encoder offloads at
1.64x, yet the M=1 AR decode (90% of runtime) cannot offload. So the whole pipeline on the
NPU is only 1.08x over a fair A76-pinned CPU baseline. No cheap NPU lever exists
(spec-decode unsupported, quant and threads already tuned).

Best use: diarization where quality matters more than speed (~0.42x realtime).

#### Granite-Speech-2B

Granite-Speech-2B pairs a conformer encoder with an LLM cross-attention decoder. It comes
in three variants:

| Variant | Notes |
|---|---|
| base | Fastest, cleanest text, CPU==NPU and Q8==F16 char-identical. 1.68x on long audio, since the cross-attention decode offloads ~1.9x, unusually for an audio-LLM |
| plus | Diarization: 8 coarse speaker turns and no timestamps, coarser than MOSS |
| NAR | An iterative non-autoregressive editor, so its decode is larger than base's, ~1.4x slower and rougher. Not the NPU speedster the name implies |

The base variant can drop spans on hard conversational audio. Cross-attn decode is not
guaranteed bit-faithful CPU-vs-NPU: greedy can diverge, and plus even loops on CPU while
staying coherent on NPU.

Best use: fast clean transcription (base), and coarse diarization (plus).

#### SenseVoice-small 234M

SenseVoice-small pairs a SAN-M encoder with one CTC head. It is the only true single-pass
model: decode is 48 ms even on 120 s of audio, so the run is 100% encoder-offloaded. It is
the fastest (6.2x realtime) and the lowest in quality. It has a 30 s window, garbles
longer audio into a run-on, and has no diarization or timestamps. The encoder is
dispatch-bound at short audio (jfk NPU==CPU 1.00x) and wins 1.26x at 120 s.

Best use: low-latency short-clip transcription where quality is secondary.

#### Fun-ASR-nano 800M

Fun-ASR-nano pairs the same SAN-M encoder with a bundled Qwen3-0.6B AR decoder, and covers
31 languages. Despite the shared encoder it is autoregressive, so the decoder limits the
gain: 1.15x at 120 s against SenseVoice's 1.26x, for only marginally better text. The AR
decoder is pure overhead that the NPU cannot recover.

Best use: multilingual coverage.

#### Parakeet CTC/TDT-0.6B

Parakeet is a FastConformer. At F16, A76-pinned, it runs ~4.6x realtime and 1.37x over
CPU. It is conv-capped: its convolution modules do not offload (no conv path in
ggml-rocket) and stay on the CPU. `ROCKET_F16_RESIDENT` hurts here (single encoder pass).
Quality is rougher on hard conversational audio (LibriSpeech clean-speech bias). It has no
diarization.

#### Parakeet on short utterances

Parakeet on short utterances (voice commands) wants `ROCKET_MIN_M=16`. Through
whisper.cpp's `parakeet-cli` the encoder runs 12.5 rows a second of audio. So a 1-5 s
command sits under the default floor, and nothing offloads. At 16 every encoder GEMM
offloads from 1.3 s up. It ties the CPU at 1.3-1.8 s and runs 1.18-1.73x faster at 2-6 s
[HW sweep 2026-09-30, mainline, TDT 0.6B v3 F16]. It frees 34-61% of the CPU, and the
transcript stays byte-identical.

Under CPU contention the offloaded stream slows more than the CPU one. So pair it with
`OMP_WAIT_POLICY=PASSIVE`
([perf/asr-cpu-relief.md](perf/asr-cpu-relief.md#short-utterances-below-the-f16-floor)).

## Template for a new row

```
### <model> (<quant>)

- Stack status: <prefill faithful? greedy NPU-vs-CPU result + provenance>; <warm pp512/pp2048 vs CPU>; <llama.cpp build / arch caveats>.
- Recommended flags: <the workload-conditional opt-ins for this model, e.g. ROCKET_QUANT_RESIDENT=auto at the default -ub for a quant GGUF whose fp16 fits RAM, and -b 2048 -ub 2048 only where residency is unavailable; no MoE flag, since the expert offload is default-on and gates itself. Defaults (KACC/REUSE/ASYM/FA) are on; do not restate them. See TUNING.md>.
- Recommended sampling: <temp/top-p/top-k/min-p + any anti-repetition>.
- Behavior: <reasoning vs not; loops/confabulation; thinking on/off>.
- Best use: <canary / transform / chat / bench>.
```
