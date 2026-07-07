# Running LLMs on the rocket NPU stack — model notes

A living, per-model record of how each model behaves on the FOSS NPU stack (stock
llama.cpp + the `ggml-rocket` backend): whether its prefill runs faithfully on the NPU,
the sampling settings it wants, behavioral quirks worth knowing, and what it's good for.
Add a row when you test a new model.

## Read two questions separately

Before judging any model on the NPU, separate two independent things:

1. **Does our NPU stack run it faithfully?** Answered only by a **greedy NPU-vs-CPU diff**
   (`--temp 0 --seed 1`, same prompt, once with `GGML_BACKEND_PATH` set and once without).
   Identical output — or output that tracks closely and diverges only late — means prefill
   is numerically faithful. fp16 NPU prefill vs fp32 CPU can eventually flip one greedy
   boundary and diverge *late*; that is expected. *Early, large* divergence is the only
   thing that points at our stack.
2. **Is the model any good at the task?** A sampling + model-choice question, fully
   independent of (1).

A coherent-but-wrong answer, or a repetition loop, is always a model/sampling issue —
**never** a sign of an NPU numerics bug. A real precision bug produces token salad (broken
grammar, non-words), not fluent prose. So fluency itself is evidence the numerics are fine.

## How the NPU is used (the operating-point context)

- **Prefill (prompt eval, batched GEMM) runs on the NPU; decode (token generation, M=1
  GEMV) stays on the CPU.** Generation throughput is therefore CPU-bound and reads
  ~identical with or without the backend loaded. The NPU is a prefill engine.
- **NPU prefill wins only at scale (pp512–pp2048).** Short prompts (tens of tokens) are
  below break-even: the fixed per-call cost — host scatter/pack, submit, readback, the
  dispatch floor — dominates when there are few MACs to amortize it over, and the CPU is
  faster. Don't judge prefill speed from a short interactive prompt; its `Prompt: N t/s`
  figure is mostly measuring overhead. See
  [perf/not-mac-bound.md](perf/not-mac-bound.md).
- **Discard the first (cold-clock) run.** The NPU idles at 200 MHz and ramps under load; a
  cold run reads ~15% low and a short job may never spin up to 600 MHz. Compare warm runs
  (`llama-bench` discards the cold run; use `-r 3`).

## Baseline invocation

The knobs every NPU run wants:

- `GGML_BACKEND_PATH` = **absolute** path to `ggml-rocket/build-dl/libggml-rocket.so`. A
  wrong path silently falls back to CPU-only (no error beyond a `failed to load` line).
- `ROCKET_KACC=1` — the best operating mode (DATA_REUSE follows automatically).
- `sudo -E` — `sudo` strips the environment; `-E` keeps `GGML_BACKEND_PATH` and the
  `ROCKET_*` knobs alive alongside `/dev/accel` privilege.
- `taskset 0xf0` — pin to the A76 big cores (cpus 4–7 on RK3588).
- **Quantized GGUF:** add `-b 2048 -ub 2048`. A quantized GGUF re-dequantizes to fp16 per
  micro-batch, so the llama.cpp default `-ub 512` roughly halves NPU prefill.
- **Confirm the NPU actually ran the prefill:** prepend `ROCKET_MM_PROFILE=1` and look for
  a `ROCKET profile total(ms):` line on stderr, plus no `failed to load` at startup.
- Models are staged on the RK1 under `/mnt/nvdata/` (the eMMC root and `/tmp` are small).

To validate a model against question (1) above:

```bash
P="Summarize the following in one sentence: <... a few hundred tokens ...>"
# NPU
sudo -E GGML_BACKEND_PATH=/path/to/ggml-rocket/build-dl/libggml-rocket.so ROCKET_KACC=1 \
  taskset 0xf0 /path/to/llama-cli -m /mnt/nvdata/<model>.gguf -p "$P" -n 200 \
  --temp 0 --seed 1 -no-cnv > npu.txt 2>/dev/null
# CPU (drop GGML_BACKEND_PATH)
taskset 0xf0 /path/to/llama-cli -m /mnt/nvdata/<model>.gguf -p "$P" -n 200 \
  --temp 0 --seed 1 -no-cnv > cpu.txt 2>/dev/null
diff npu.txt cpu.txt && echo IDENTICAL || echo DIVERGED
```

Use a few-hundred-token prompt so the prefill is large enough to actually route to the NPU
(a tiny prompt may run prefill on the CPU regardless, proving nothing).

## Per-model notes

### Qwen3.5-0.8B (F16)

- **Stack status:** prefill runs out of the box; greedy output is **token-identical to CPU**
  [HW sweep, 2026-06-28, RK1, 600 MHz] → prefill is faithful for this arch. Warm prefill
  ≈ **1.44× CPU at pp512** [HW sweep]. The `qwen35` arch needs llama.cpp ≥ b9568.
- **Recommended sampling:** `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0` (Qwen
  thinking-mode defaults); add `--presence-penalty 1.0` or `--dry-multiplier 0.8` to curb
  repetition.
- **Behavior:** a reasoning model — it emits a `[Start thinking]` block before answering.
  At 0.8B it confabulates niche facts (mislabels the RK3588, swaps digits like 3588→3580)
  and falls into **semantic** reasoning loops. Token-level anti-repetition (DRY,
  presence-penalty) does **not** stop a semantic loop: the model varies surface wording
  while repeating the meaning. For clean Q&A, disable thinking with a `/no_think` suffix on
  the message, or cap the context.
- **Best use:** a bring-up / throughput canary, and text-transform tasks (summarize,
  rewrite) over text you supply. Not a factual-recall chat model at this size.

### Gemma-4-12B-it

- **Stack status:** fp16 prefill runs coherently on the NPU; greedy output is **char-identical
  to CPU fp16** → prefill is faithful at 12B scale [HW sweep]. Native **int8**
  (`ROCKET_INT8=1 ROCKET_INT8_HADAMARD=1`) and **int4** (`ROCKET_INT4=1`) also run
  coherently and greedy char-identical to fp16 — these are RAM / model-fit levers, **not**
  prefill-speed levers (the NPU is dispatch-bound, so quantization buys footprint, not
  speed).
- **Eval note:** it is a reasoning model, so **wikitext perplexity is not a valid quality
  metric** (the F16 reference PPL is itself ~545). Evaluate with greedy-match against the
  CPU reference and a cosine probe, not PPL.
- **Recommended invocation:** the standard Gemma chat template (applied automatically from
  the GGUF). A quantized GGUF wants `-b 2048 -ub 2048`; the resident/prepacked path gives
  the warm prefill numbers.
- **Best use:** the primary LLM-pillar target — real Q&A and prefill benchmarking. No
  loop/confabulation pathology like the 0.8B.

## Template for a new row

```
### <model> (<quant>)

- Stack status: <prefill faithful? greedy NPU-vs-CPU result + provenance>; <warm pp512/pp2048 vs CPU>; <llama.cpp build / arch caveats>.
- Recommended sampling: <temp/top-p/top-k/min-p + any anti-repetition>; <quant → -b/-ub>.
- Behavior: <reasoning vs not; loops/confabulation; thinking on/off>.
- Best use: <canary / transform / chat / bench>.
```
