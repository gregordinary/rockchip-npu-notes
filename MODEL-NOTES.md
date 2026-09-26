# Model notes: running LLMs on the rocket NPU stack

A living, per-model record of how each model behaves on the FOSS NPU stack, meaning stock
llama.cpp plus the `ggml-rocket` backend. Each row covers whether that model's prefill runs
faithfully on the NPU, the sampling settings it wants, behavioral quirks worth knowing, and
what it is good for. Add a row when you test a new model. Speech-to-text models, run through
transcribe.cpp on the same backend, get their own subsection at the end of the per-model notes.

For **which flags to enable for a given workload**, meaning use-case, precision, prompt size
and the RAM or disk each opt-in needs, see [TUNING.md](TUNING.md). This file records per-model
behavior, and that one is the decision guide.

## Read two questions separately

Before judging any model on the NPU, separate two independent things:

1. **Does our NPU stack run it faithfully?** First establish that the NPU arm offloaded
   anything, then measure what it cost. The three steps are below.
2. **Is the model any good at the task?** A sampling + model-choice question, fully
   independent of (1).

**The NPU arm must offload before it can be compared.** A quantized GGUF sends a prefill
shorter than `ROCKET_MIN_M_QUANT` (default 512 tokens) to the CPU, so a short prompt compares the
CPU with itself. Use a prompt at least that long, or set `ROCKET_MIN_M_QUANT=0` for the check.
Run the NPU arm with `ROCKET_MM_PROFILE=1 ROCKET_LOG_STDERR=1`: its exit line counts the jobs it
submitted. A count of zero, or no line, means the arm computed on the CPU.

**Differential perplexity is the measure.** Run `llama-perplexity` over the same text with and
without `GGML_BACKEND_PATH`, and read the ratio. The defects this stack has met cost perplexity.
One per-tensor int8 output scale costs 1.11-2.26x over two models, simulated on the host from
their own activations ([chips/rk3576.md](chips/rk3576.md)). Perplexity scores that loss, and
fluent output does not rule it out.

**A greedy diff is a quick screen, not the verdict.** Run `--temp 0 --seed 1` on the same prompt
with and without the backend. fp16 prefill against fp32 eventually flips one greedy boundary, so
a late divergence is expected, and an early, large one points at the stack. Identical output is
a warning rather than a pass: it is also what an arm that offloaded nothing produces.

A coherent-but-wrong answer, or a repetition loop, is usually a model or sampling issue. Token
salad (broken grammar, non-words) points at the stack. Neither reading clears the numerics,
which is what the perplexity ratio is for.

## How the NPU is used (the operating-point context)

- **Prefill (prompt eval, batched GEMM) runs on the NPU, and decode (token generation, M=1
  GEMV) stays on the CPU.** Generation throughput is therefore CPU-bound and reads
  ~identical with or without the backend loaded. The NPU is a prefill engine.
- **NPU prefill wins only at scale (pp512-pp2048).** Short prompts of tens of tokens are
  below break-even. The fixed per-call cost (host scatter and pack, submit, readback, the
  dispatch floor) dominates when there are few MACs to amortize it over, and the CPU is
  faster. Do not judge prefill speed from a short interactive prompt, whose `Prompt: N t/s`
  figure is mostly measuring overhead. See
  [perf/not-mac-bound.md](perf/not-mac-bound.md).
- **Discard the first, cold-clock run.** The NPU idles at 200 MHz and ramps under load, so a
  cold run reads ~15% low and a short job finishes before the clock reaches 600 MHz. Compare
  warm runs. `llama-bench` discards the cold run, so use `-r 3`.

## Baseline invocation

The knobs every NPU run wants:

- `GGML_BACKEND_PATH` = **absolute** path to `ggml-rocket/build-dl/libggml-rocket.so`. A
  wrong path silently falls back to CPU-only (no error beyond a `failed to load` line).
- The operating-mode knobs, fp16 K-accumulation (`ROCKET_KACC`), DATA_REUSE
  (`ROCKET_REUSE=2`), asymmetric tiling (`ROCKET_MM_ASYM`), attention offload
  (`ROCKET_FLASH_ATTN`), are **on by default**; you need not set them. The workload-specific
  opt-ins are in [TUNING.md](TUNING.md).
- `sudo -E`: plain `sudo` strips the environment. The `-E` flag keeps `GGML_BACKEND_PATH`
  and the `ROCKET_*` knobs alive alongside `/dev/accel` privilege.
- `taskset 0xf0`: pin to the A76 big cores (cpus 4-7 on RK3588). Worth **1.05-1.13x** on NPU
  prefill over three models, largest on the smallest [HW sweep 2026-09-01, RK1, rotated
  interleaved passes]. Residency does not gate it. Leave a decode-heavy run unpinned, because F16
  decode wants all eight cores for their bandwidth.
- **Quantized GGUF:** consider `-b 2048 -ub 2048`. A quantized GGUF re-dequantizes to fp16
  per micro-batch and the larger micro-batch amortizes that, but what it buys is
  model-dependent — **0.91-1.53x measured over eleven models, and a loss on two**
  [HW sweep 2026-08-29/30, RK1, 600 MHz]. Read the model's own row below;
  `ROCKET_QUANT_RESIDENT=auto` is the stronger lever where the fp16 image fits RAM.
- **Confirm the NPU actually ran the prefill:** prepend `ROCKET_MM_PROFILE=1` and look for
  a `ROCKET profile total(ms):` line on stderr, plus no `failed to load` at startup.
- Stage models on external storage: an eMMC root and a `/tmp` tmpfs are usually too small.

To validate a model against question (1) above:

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
below the routing floor runs prefill on the CPU regardless, proving nothing.

## Per-model notes

### Qwen3.5-0.8B (F16)

- **Stack status:** prefill runs out of the box, and greedy output is **token-identical to
  CPU** [HW sweep, 2026-06-28, RK1, 600 MHz], so prefill is faithful for this arch. Warm
  prefill is ~ **1.44x CPU at pp512** [HW sweep]. The `qwen35` arch needs llama.cpp >= b9568.
- **Recommended flags:** defaults only. F16 fits any board trivially, and at 0.8B the NPU
  wins prefill only past ~pp96. Nothing to opt into. Its role is a throughput/bring-up canary.
- **Recommended sampling:** `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0`, the Qwen
  thinking-mode defaults. Add `--presence-penalty 1.0` or `--dry-multiplier 0.8` to curb
  repetition.
- **Behavior:** a reasoning model, emitting a `[Start thinking]` block before answering.
  At 0.8B it confabulates niche facts (mislabels the RK3588, swaps digits like 3588->3580)
  and falls into **semantic** reasoning loops. Token-level anti-repetition (dry,
  presence-penalty) does **not** stop a semantic loop: the model varies surface wording
  while repeating the meaning. For clean Q&A, disable thinking with a `/no_think` suffix on
  the message, or cap the context.
- **Best use:** a bring-up / throughput canary, and text-transform tasks (summarize,
  rewrite) over text you supply. Not a factual-recall chat model at this size.
- **The Q4_K_M variant** measures `-b 2048 -ub 2048` at **1.130x** and
  `+ROCKET_QUANT_RESIDENT=auto` at **1.179x** over stock at pp2048 (99.1 -> 116.9 t/s)
  [HW sweep 2026-08-28, RK1, 600 MHz, three rotated passes]. This model class, 0.8B in both
  precisions, is also the one that carries this board's per-process spread. Single arms have
  spanned 8-11% across processes where every larger model resolves to 0.4-2%, so a one-process
  A/B on it settles nothing.

### Llama-3.2-3B-Instruct (Q4_K_M / F16)

- **Recommended flags (Q4_K_M):** `ROCKET_QUANT_RESIDENT=auto` at the DEFAULT `-ub`. Do
  **not** stack `-b 2048 -ub 2048`. Unstacked residency reads **1.472x** (39.2 -> 57.7 t/s,
  193 weights resident, 5232 MB, 0 streamed). The stacked recipe reads 1.212x there and the
  `-ub` lever alone 1.089x [HW sweep 2026-08-29/31, RK1, 600 MHz, three rotated passes each].
  The mechanism control decomposes the `-ub` flag here as a 1.258x dequant win over a real
  **0.863x** non-dequant loss, which is the sub-4B class pattern.
- **Recommended flags (F16):** `ROCKET_F16_RESIDENT=auto`, worth **1.081x** (56.0 -> 60.5
  t/s). The default admits resident weights only at K<=2048 and this model's K=3072, so
  without the knob nothing is offered to the resident route at all.
- **The two routes place identically** (the same 193 weights, 5232 MB) and the F16 GGUF is
  still the faster ridge: 60.5 t/s against the quant GGUF's 47.2 at the same placement.
  Residency removes the per-micro-batch dequant, not the one-time decode. Run the F16 file
  if the 6.4 GiB fits your disk and RAM budget.

### Ministral-3-3B (Q4_K_M / F16)

- **Recommended flags (Q4_K_M):** `ROCKET_QUANT_RESIDENT=auto` at the DEFAULT `-ub`. Do
  **not** stack `-b 2048 -ub 2048`. Unstacked residency reads **1.445x** (34.6 -> 50.0 t/s,
  100% resident), where the stacked recipe reads 1.132x and the `-ub` arm alone 1.038x
  [HW sweep 2026-08-29/31, RK1, 600 MHz, three rotated passes each]. This model has the
  class's deepest non-dequant `-ub 2048` loss, a **0.835x** residue under the mechanism
  control. That is why its stacked numbers were the smallest in the 3B group.
- **Recommended flags (F16):** `ROCKET_F16_RESIDENT=auto`, worth **1.078x** (48.3 -> 52.0
  t/s), at the same K=3072 gate as Llama-3.2-3B, so the knob turns the whole route on.

### Phi-4-mini (Q4_K_M / F16)

- **Recommended flags (Q4_K_M):** `ROCKET_QUANT_RESIDENT=auto` at the DEFAULT `-ub`. Do
  **not** stack `-b 2048 -ub 2048`. Unstacked residency reads **1.508x** (35.9 -> 54.2 t/s,
  100% resident), where the stacked recipe reads 1.211x and the `-ub` arm alone 1.093x
  [HW sweep 2026-08-29/31, RK1, 600 MHz, three rotated passes each]. The `-ub` flag's
  non-dequant residue here is **0.864x**.
- **Recommended flags (F16):** `ROCKET_F16_RESIDENT=auto`, worth **1.084x** (50.2 -> 54.4
  t/s), at K=3072, the same gate as the other 3B-class models.

### SmolVLM2-2.2B-Instruct (Q4_K_M, text path)

The one dense model measured where the standard quant recipe is a net zero and half of it
is a loss. Its rows are why the guide's recipes are per-model:

- **Recommended flags:** `ROCKET_QUANT_RESIDENT=auto` at the DEFAULT `-ub 512`, and do
  **not** set `-b 2048 -ub 2048`. Unstacked residency reads **1.346x** (51.5 -> 69.3 t/s),
  where the stacked recipe reads 1.002x [HW sweep 2026-08-31, RK1, 600 MHz, three rotated
  passes, 165 weights resident (2976 MB), 0 streamed].
- **Why:** `-b 2048 -ub 2048` is a resolved **0.946x loss** here. The mechanism control
  decomposes it as a 1.199x dequant-amortization win masking a real **0.789x non-dequant
  cost** of the larger micro-batch, whose mechanism is open. Residency removes the dequant
  without paying that cost, so it replaces the `-ub` lever on this model instead of stacking
  on it.
- **Scope:** an idle host. One busy core flips the `-ub` flag's sign back to a win
  (0.946x -> 1.120x measured under a deliberate one-core load), so on a loaded box the
  stacked recipe stops losing. The unstacked one needs no such caveat.
- **The vision path is separate** and has its own resident SigLIP encoder work. These rows
  are the text prefill only.

### Qwen3.5-9B (Q4_K_M)

- **Recommended flags:** `ROCKET_QUANT_RESIDENT=auto` at the DEFAULT `-ub`. Do **not**
  stack `-b 2048 -ub 2048`. Unstacked residency reads **1.752x** (19.1 -> 33.5 t/s, 200
  weights resident, 13184 MB, 0 streamed). The stacked recipe reads 1.661x there and the
  `-ub` lever alone 1.42-1.43x [HW sweep 2026-08-29/31, RK1, 600 MHz, three rotated
  passes each, per-pass 1.747 / 1.737 / 1.771]. The largest tuning win in the matrix.
- **The `-ub` lever here is ~90% dequant amortization** (mechanism control: 1.430 = 1.378
  dequant x 1.038 residue), which is why residency beats raising `-ub`: it removes the
  dequant outright. This model is also the one whose non-dequant residue is a WIN, which is
  the case where the residue argument predicts stacking holds. It does not. Residency removes
  the per-call pack and upload as well, and the default `-ub` presents four times as many
  calls to remove them from. That is worth more than the 3.8% the residue gives back.
- **Footprint:** the resident arm holds ~13.2 GB of fp16 cubes beside the 5.3 GiB GGUF.
  It fits a 31 GiB board with room, not a 16 GiB one.

### Ministral-3-8B (Q4_K_M)

- **Recommended flags:** `ROCKET_QUANT_RESIDENT=auto` at the DEFAULT `-ub`. Do **not**
  stack `-b 2048 -ub 2048`. Unstacked residency reads **1.651x** (17.8 -> 29.4 t/s, 100%
  resident), where the stacked recipe reads 1.325x and the `-ub` arm alone 1.176x
  [HW sweep 2026-08-29/31, RK1, 600 MHz, three rotated passes each, per-pass
  1.630 / 1.650 / 1.674].

### Gemma-4-12B-it

- **Stack status:** fp16 prefill runs coherently on the NPU, and greedy output is
  **char-identical to CPU fp16**, so prefill is faithful at 12B scale [HW sweep]. Native
  **int8** (`ROCKET_INT8=1 ROCKET_INT8_HADAMARD=1`) and **int4** (`ROCKET_INT4=1`) also run
  coherently and greedy char-identical to fp16. Those are RAM and model-fit levers, **not**
  prefill-speed levers: at this operating point the NPU is dispatch-bound, so quantization
  buys footprint rather than speed.
- **Recommended flags (Q4_K_M):** `ROCKET_QUANT_RESIDENT=auto` at the DEFAULT `-ub`. Do
  **not** stack `-b 2048 -ub 2048`. Unstacked residency reads **1.368x** (12.8 -> 17.5 t/s).
  The stacked recipe reads 1.322x, and the `-ub` lever alone 1.257x. Per-pass
  1.402 / 1.382 / 1.319 [HW sweep 2026-08-29/31, RK1, 600 MHz, three rotated passes each].
  The F16 GGUF is 22.2 GiB: stock places zero weights, and `ROCKET_F16_RESIDENT=auto` places
  286 of 328 for 1.0614x.
- **Full residency is reachable and is not recommended.** `ROCKET_N_THREADS=8` with
  `ROCKET_QUANT_RESIDENT_RESERVE_MB=6144` places 328 of 328, and reads 1.009x straddling 1.00
  over three passes. The worker count it needs costs 0.987x, cancelling what the last 42 weights
  buy, and it holds 2712 MB more on a swapless board.
- **This is the matrix's only PARTLY placed model, and the recipe is why it matters.** On a
  31 GiB board the resident route places **83-87%** unstacked (272-284 of 328 weights,
  17.1-17.9 GB) against **73-74%** stacked (239-243, ~15 GB). Both arms stop at the same
  9535 MB `MemAvailable` reserve floor rather than at the 21129 MB resident-weight budget. The
  unstacked arm reaches that floor 2.6-2.8 GB later, because `-b 2048 -ub 2048` spends that RAM
  on compute buffers. Raising `-ub` here costs residency directly.
- **Read the `[f16-resident]` outcome line per arm on this model**, because placement varies
  run to run. Treat its +3.5% over stacked as a sign and not a size: the per-pass spread is
  6.3%, the widest of the seven models measured both ways.
- **Native `ROCKET_INT8` and `ROCKET_INT4`** run coherently but are RAM-fit levers from an F16
  GGUF, not prefill-speed levers.
- **Eval note:** it is a reasoning model, so **wikitext perplexity is not a valid quality
  metric** (the F16 reference PPL is itself ~545). Evaluate with greedy-match against the
  CPU reference and a cosine probe, not PPL.
- **Recommended invocation:** the standard Gemma chat template, applied automatically from
  the GGUF. The resident and prepacked path is what gives the warm prefill numbers.
- **Best use:** the primary LLM-pillar target, for real Q&A and prefill benchmarking. No
  loop/confabulation pathology like the 0.8B.

### Phi-4-14B (Q4_K_M)

- **Recommended flags:** `-b 2048 -ub 2048`, worth **1.308x** at pp2048 (10.9 -> 14.3
  t/s) [HW sweep 2026-08-29, RK1, 600 MHz, three rotated passes]. No residency arm: the
  fp16 resident image is ~29 GiB beside the GGUF and does not fit a 31 GiB board.

### Qwen3.6-27B (Q4_K_M)

- **Recommended flags:** `-b 2048 -ub 2048`, worth **1.532x** at pp2048 (6.1 -> 9.3 t/s).
  That is the largest `-ub` lever measured, on the largest dense model
  [HW sweep 2026-08-29, RK1, 600 MHz, three rotated passes]. There is no residency arm,
  because the fp16 resident image is ~54 GiB. At ~6-9 t/s prefill this is a batch model on
  this board, not an interactive one.

### gpt-oss-20b (MXFP4, MoE)

The mixture-of-experts model, and the one whose settings pull against each other. It has 24
layers and 32 experts with 4 active. Attention alternates windowed (128) and full, so **half
its layers are full-attention**. Its settings:

- **Recommended flags:** `-b 2048 -ub 2048`, and **nothing for the experts**. The routed-expert
  offload is on by default since 2026-08-27 and gates itself. It reserves a whole expert stack
  before it claims that stack's op, and leaves on the CPU what it cannot reserve. On this board
  that admits 63 of 72 stacks at 100% residency, which is **2.38x the CPU** at pp2048. The `-ub`
  and residency detail is below.
- **Stack status: prefill runs coherently, and greedy output does not match the CPU reference on
  any NPU arm, including the one with the experts left on the CPU.** On a ~1400-token passage at
  `--temp 0 --seed 1`, the experts-on-CPU arm first diverges at **word 16** of the generation and
  every arm carrying the expert offload at **word 10** [HW sweep 2026-08-27]. All continuations stay
  coherent and on-topic, which is the criterion that matters here; this file's own rule is that
  only *early **and large*** divergence points at our stack, and an fp16 prefill against an fp32 CPU
  reference flips greedy boundaries as a matter of course. What the expert route adds is **about six
  words earlier**, and a single 48-token generation cannot separate that from one more flip in the
  same cascade. It is a **reasoning** model (harmony format), so **wikitext PPL is not a valid
  quality metric**; the quantitative gates for this model are the per-matmul cosine probe
  (`ROCKET_MOE_COSINE=1`, real weights and real activations against an fp64 CPU reference,
  **mean 0.999815, min 0.998976 over 55 expert GEMMs** on the shipped placement) and
  `test-rocket-moe`, not the greedy match. Decode stays on the CPU as always (~7.2 t/s, brisk for
  20B because only ~3.6 B params are active per token).
- **The `-ub` setting pulls two ways, and you must choose deliberately.**
  - **Run the MoE expert route at `-b 2048 -ub 2048`**: every number here was measured there,
    and two mechanisms say a smaller micro-batch costs it. First, the **dense** MXFP4 weights
    (attention projections, `lm_head`) are *not* in the expert cache and still re-dequantize to
    fp16 **per micro-batch**, so `-ub 512` runs that decode four times over on a 2048-token
    prompt. Second, the router gives each expert only `n_tokens · n_used / n_expert` rows,
    **64 at `-ub 512` against 256 at `-ub 2048`**, and the per-expert overhead around the GEMM
    (dispatch, row gather, scatter, M-bucket padding) does not shrink with the row count, so a
    quarter of the rows buys close to the same overhead. `ROCKET_MOE_MIN_TOKENS` (default 512)
    also sits right at `-ub 512`, so offload barely qualifies.
    **Now measured at `-ub 512`**: 22.2 / 22.5 t/s at pp512 / pp2048 against 14.13 / 13.53 with the
    experts on the CPU, about 1.6x / 1.7x, against 22.0 / 28.1 at `-ub 2048`. So the two mechanisms
    together cost ~20% at pp2048 and nothing at pp512, and there is **no collapse**. (The
    often-quoted "`-ub 512` collapses MoE to ~0.42x" is the **fp16 streaming** route, whose
    per-expert dequant *is* what `-ub` multiplies. Native-quant ingests each expert once and deletes
    exactly that cost, so the old reasoning never transferred to it.)
  - But **`-ub 2048` makes the *dense* graph slower on this model**: NPU-default reads 13.11
    t/s at `-ub 512` against 11.31 at `-ub 2048` (pp2048). The CPU does not care either way.
  - So there is no single best `-ub` here: it depends on whether the experts are on the NPU.
    **Never compare a `-ub 512` number against a `-ub 2048` one**; that mistake is what made
    an earlier session chase a nonexistent regression.
- **Routed experts: worth 2.38x the CPU, and on by default.** Prefill **22.0 / 28.1 t/s** at
  pp512 / pp2048 (**1.81x / 2.38x** the CPU; 1.64x -> 2.08x over the experts-on-CPU arm across
  pp512-pp2048) with the experts held resident on the NPU as native int8. The *fp16* expert route is
  a net **loss** (4.59 / 10.18): it re-dequantizes every expert every micro-batch, ~75 ms each,
  *independent of the row count*. The native route ingests each expert **once** and deletes that tax,
  at a one-time **~36 s** ingest inside the first prefill, per `llama_context`.
  **What made it opt-in was that its sign depended on the host's RAM**, and a residency pre-flight now
  removes that dependence: a stack it cannot reserve is left on the CPU whole rather than
  half-ingested into the partial-residency loss (82% resident read 12.19 at pp512, below the 14.11 you
  get leaving the experts on the CPU). Check the split with `ROCKET_LOG_STDERR=1`; under the default
  it should read **100% resident**. `ROCKET_MOE=1` overrides the pre-flight and both size floors: it
  is *faster* where the stack nearly fits and **below the experts-on-CPU baseline where it does not**,
  so it is the A/B arm, not a setting. With RAM to spare, raise `ROCKET_MOE_CACHE_MB` instead --
  **on this model.** Raising it is a loss on an expert-dominated MoE, where the experts are most of
  the GGUF; see [TUNING.md](TUNING.md) §"MoE routed experts on the NPU" before carrying the
  direction to another model.
  **This model clears the size floors at every prefill length; DeepSeek-V2-Lite does not**; see its
  section, and do not read this ratio across to another MoE.
- **Its attention stays on the CPU, and must.** gpt-oss carries a learned **attention sink** per
  head, and the NPU FLASH_ATTN handler has no sink term, so the offload is declined for it. It had
  been silently *accepted*, computing a sink-less (wrong) softmax past the `n_kv` floor of 1024.
  Declining is both correct and **+26%** at pp2048. If you benchmark a model and the NPU curve
  *collapses past ~1K context but is fine below it*, suspect this class of bug.
- **Best use:** the MoE showcase, and the residency stress case: its expert stack only just fits a
  31 GiB board alongside its own GGUF, so it is where partial residency gets exercised.

### DeepSeek-V2-Lite (Q4_K_M, MoE + MLA)

The second MoE, and the one that shows expert-offload eligibility is a property of the
**architecture**, not of the flag. 27 blocks (block 0 dense), **64 routed experts with 6 active**
plus 2 always-on shared, MLA attention, ~2.4 B of 15.7 B params active per token. A **base** model,
so unlike gpt-oss its wikitext PPL is a valid quality metric.

- **Recommended flags:** `-b 2048 -ub 2048`, nothing else. The expert offload is on by default and
  decides per micro-batch.
- **Its experts offload only past ~1250 tokens in a micro-batch, and that is correct.** Two size
  floors gate each op, and the binding one here is the work a dispatch carries, `M_e · K · N` where
  `M_e` = `n_tokens · n_used / n_expert`. DeepSeek's 6-of-64 routing over a **2048×1408** expert
  carries **2.88x less work per dispatch** than gpt-oss's 4-of-32 over a 2880×2880 one at the same row
  count, so at `M_e` = 72 the offload measures **0.95x the experts-on-CPU arm**, a real loss, while
  gpt-oss at `M_e` = 64 measures **1.64x**. Past the floor it wins: **1.22x at `M_e` = 144 and 1.31x
  at `M_e` = 192** [HW sweep 2026-08-27, 600 MHz pinned]. Whole-model prefill under the default:
  **25.9 -> 28.1 t/s** at pp512 -> pp2048, i.e. 1.33x -> 1.52x the CPU.
- **Do not read gpt-oss's MoE ratio across to this model, or this one's across to a third.** A
  per-expert row count cannot separate them: `M_e` = 96 is 1.77x on gpt-oss and roughly parity here.
- **Its attention (MLA) stays on the CPU.** The FLASH_ATTN gate accepts DK≠DV and the primitive is
  bit-faithful for MLA, but the DeepSeek path is not yet exercised on-device, so attention is CPU-side
  here. What does reach the NPU below the expert floor: the large MLA projections, the 2 shared
  experts, and `lm_head`, a modest but real win on its own.
- **Faithfulness on the shipped placement:** differential wikitext PPL at `-c 2048` (which puts
  `M_e` = 192, so the expert route is **active**, 4800 experts resident, 0 streamed). CPU 5.3063,
  `ROCKET_MOE=0` 5.2906, default 5.2842. **Isolate the expert route by differencing the two NPU
  arms**, not the default against the CPU: that reads **−0.121%, 8 paired chunks, 1.20 se,
  indistinguishable from zero**, where default-vs-CPU (−0.416%) would credit the experts with the
  dense fp16 path's own −0.296%. Read the paired per-chunk difference, never the finals: the
  absolute error bar is ±2.5% and resolves nothing. Absolute PPL is not comparable to the archived
  8.24, which was `-c 512`.
- **Best use:** the second MoE architecture, and the one to re-run whenever the placement floors move.

### Qwen3-30B-A3B (Q4_K_M, MoE)

The third MoE, and the model where every tuned arm loses — its rows are the measured case
for leaving the defaults alone [HW sweep 2026-08-30, RK1, 600 MHz, three rotated passes].

- **Recommended flags: none.** Stock (14.7 t/s at pp2048) beats every tuned arm:
  `-b 2048 -ub 2048` is **0.908x**, `ROCKET_MOE=1` is **0.726x**, `ROCKET_MOE=0` is
  0.908x.
- **The expert route correctly declines this model.** 29 of its 30.5 B parameters are
  routed experts, which cannot be held resident on a 31 GiB board; AUTO places nothing
  (3 of 3 passes), so the default and `ROCKET_MOE=0` are the same configuration reached
  two ways (0.908x against 0.908x). **Forcing `ROCKET_MOE=1` places 62%, streams the
  rest, and costs 0.800x against the experts-on-CPU baseline** — the loss the residency
  pre-flight exists to avoid, measured end to end. Do not set it here.
- **Do not size this model by total parameter count.** Only ~3 B are active per token;
  the dense >8 B band (1.18-1.53x on the `-ub` lever) does not apply, and its own row is
  below even the 3B dense band.
- **Best use:** the guard-rail regression model — the one that tests that AUTO declines
  what it should — and bulk-RAM chat where its 14.7 t/s prefill and small active set suit
  the board. Expect nothing from tuning.

### Speech-to-text models (transcribe.cpp, not llama.cpp)

These run through **transcribe.cpp** (a ggml multi-STT host), not llama.cpp, but they load the same
`ggml-rocket` `.so` (`GGML_BACKEND_PATH`) and the NPU does the same job: the **encoder** (and, on long
audio, the batched decode-prefill). Autoregressive decode stays on the CPU. The faithfulness question
splits from the LLM one above: **encoder-only offload is bit-faithful** (CPU==NPU), but a model whose
decoder cross-attn offloads can diverge late as fp16 accumulation perturbs greedy decoding; that is
expected, not an NPU bug. Cross-model speed/quality tables are in
[perf/benchmarks.md](perf/benchmarks.md#speech-to-text-multi-model-transcribecpp-via-ggml-rocket).

- **Recommended flags (all STT):** A76-pin (`taskset -c 4-7 --threads 4`, since the A55 cluster is a
  straggler), `ROCKET_KACC=1`. **Q8_0 is the default**: for STT it ties or beats F16 on both CPU and
  NPU at half the RAM (these models are less matmul-dominated than LLM prefill, so the per-micro-batch
  dequant tax is small while CPU-side decode is memory-bound and Q8 moves half the bytes). Build/run
  traps (shared/DL build needs `-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`, and a one-line
  `main.cpp` backend-init patch) are in the benchmarks doc.

- **Voxtral-mini 3B** (Whisper-large-v3 encoder + Ministral-3B AR decoder): **best transcript** and
  the only **translator** here (de->en, ja->en). CPU==NPU **char-identical** throughout. Big NPU win on
  long audio (1.57x: encoder 1.82x *and* the 1500-token audio prefill offloads at 1.46x); on short
  clips only the encoder offloads (jfk 1.26x). Slow (0.55x realtime, a 3B decode). No diarization, no
  timestamps. Best use: highest-accuracy transcription and translation when latency is not the concern.

- **MOSS 0.9B** (Whisper-Medium encoder + Qwen3-0.6B AR decoder, audio injected as KV tokens): **best
  diarizer** by a wide margin (33 fine segments + timestamps, clean two-speaker split, never
  degenerates). But **decode-bound**: the encoder offloads 1.64x yet the M=1 AR decode (90% of runtime)
  cannot, so whole-pipeline NPU is only **1.08x** over a fair A76-pinned CPU baseline. No cheap NPU
  lever exists (spec-decode unsupported; quant/threads already tuned). Best use: diarization where
  quality matters more than speed (~0.42x realtime).

- **Granite-Speech-2B** (conformer encoder + LLM cross-attention decoder), in three variants: **base**
  (fastest, cleanest text, CPU==NPU and Q8==F16 char-identical, 1.68x on long audio, since the
  cross-attention decode offloads ~1.9x, unusually for an audio-LLM), **plus** (diarization: 8 coarse
  speaker turns and no timestamps, coarser than MOSS), **NAR** (iterative non-autoregressive editor ->
  decode is *larger* than base's, ~1.4x slower and rougher; not the NPU speedster the name implies).
  base can drop spans on hard conversational audio. Cross-attn decode is **not** bit-faithful CPU-vs-NPU
  (greedy can diverge; plus even loops on CPU while staying coherent on NPU). Best use: fast clean
  transcription (base); coarse diarization (plus).

- **SenseVoice-small 234M** (SAN-M encoder + one CTC head): the only **true single-pass** model
  (decode 48 ms even on 120 s -> 100% encoder-offloaded). **Fastest** (6.2x realtime), lowest quality
  (30 s window; garbles longer audio into a run-on; no diarization/timestamps). Encoder is
  dispatch-bound at short audio (jfk NPU==CPU 1.00x), wins 1.26x at 120 s. Best use: low-latency
  short-clip transcription where quality is secondary.

- **Fun-ASR-nano 800M** (same SAN-M encoder + a bundled Qwen3-0.6B AR decoder; 31 languages): despite
  the shared encoder it is **autoregressive**, so the decoder is the wall (1.15x at 120 s vs
  SenseVoice's 1.26x) for only marginally better text. The AR decoder is pure overhead the NPU cannot
  recover. Best use: multilingual coverage.

- **Parakeet CTC/TDT-0.6B** (FastConformer): F16, A76-pinned, ~4.6x realtime, 1.37x over CPU, but
  **conv-capped**: its convolution modules do not offload (no conv path in ggml-rocket) and stay on the
  CPU. `ROCKET_F16_RESIDENT` hurts here (single encoder pass). Quality is rougher on hard
  conversational audio (LibriSpeech clean-speech bias). No diarization.

## Template for a new row

```
### <model> (<quant>)

- Stack status: <prefill faithful? greedy NPU-vs-CPU result + provenance>; <warm pp512/pp2048 vs CPU>; <llama.cpp build / arch caveats>.
- Recommended flags: <the workload-conditional opt-ins for this model, e.g. -b 2048 -ub 2048 for a quant GGUF; ROCKET_QUANT_RESIDENT=auto if its fp16 fits RAM; ROCKET_MOE=1 iff the expert stack fits. Defaults (KACC/REUSE/ASYM/FA) are on; do not restate them. See TUNING.md>.
- Recommended sampling: <temp/top-p/top-k/min-p + any anti-repetition>.
- Behavior: <reasoning vs not; loops/confabulation; thinking on/off>.
- Best use: <canary / transform / chat / bench>.
```
