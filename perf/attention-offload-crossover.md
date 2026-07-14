# Prefill attention offload crosses the CPU at ~2K context

Offloading the fused prefill attention op (`FLASH_ATTN_EXT`: per-head QK → mask → softmax →
P·V) from the CPU to the NPU is a **wash at short context and a net win from ~2K tokens up**,
because the two backends scale differently with context length. This is the lever that moves
long-context prefill on reasoning models (Gemma-4 and the like). The crossover sits at ~2K with
per-worker QK/AV submit chaining on (the default); without it the NPU per-head dispatch floor
pushes the crossover out to ~6K.

> **Scope it honestly.** Measured at the current operating point — fp16, 600 MHz, the
> multicore + host-softmax attention handler (`librocketnpu` `rocket_flash_attn_fp16_mt`),
> heads fanned across the worker fds, per-head QK/AV submits collapsed into one job each via a
> resident batched-matmul context. The crossover *context* depends on that handler's host
> overhead and on the CPU's flash-attention speed; it is not a fixed property of the silicon.
> Within the attention **shape** the handler implements, the offload is bit-faithful
> (differential perplexity FA-NPU == FA-CPU; per-head cosine 1.0) — only the speed crosses over.

> **The handler implements one attention, and it must decline anything else.** It computes
> `softmax(scale·QKᵀ + mask) · V`. An **attention sink** — a learned per-head logit that joins the
> softmax denominator, passed as the op's `src[4]` by `ggml_flash_attn_ext_add_sinks` — is a
> *different* attention, and the handler has no term for it. Accepting such an op does not lose a
> little accuracy; it computes the wrong function, silently, and the graph cannot tell.
>
> This was live: `supports_op` validated `src[0..3]` and never looked at `src[4]`, so **gpt-oss**
> (which carries sinks on every layer; so do `mimo2` and `deepseek4`) took the offload and got a
> sink-less softmax. It hid well — it fires only past the `n_kv` floor of 1024, so short-prompt
> tests never reached it, and a wrong-but-plausible attention still produces fluent text, so the
> differential-PPL check read it as "+1.0%, within noise". **A faithfulness metric that cannot
> distinguish a wrong function from run-to-run noise is not a faithfulness metric.** The gate now
> declines on `src[4]`. Implementing the sink is easy in principle — the softmax is host-side, so
> it is one more term in the denominator — but it is only worth doing where the offload *wins*,
> and on gpt-oss's head geometry it does not (below).

## What the offload targets

Under llama.cpp's default `-fa auto`, attention is **one fused `GGML_OP_FLASH_ATTN_EXT` op per
layer**, not separate QK/AV `mul_mat` nodes (confirmed by a `supports_op` graph dump) — so the
offload is a backend op handler, not a matmul interception. The handler gathers the op's permuted
F32 Q, the strided F16 K/V cache views, and the F16 causal/sliding-window mask (supplied as an
input, applied additively — the handler does not synthesise it), runs per-head
`scale·QKᵀ → +mask → softmax → P·V` on the NPU with 2:1 GQA broadcast, and scatters the F32 result.

The Gemma-4-12B shape sets the gate: 48 layers in a **5 local : 1 global** sliding-window pattern =
**40 windowed-local layers (window 1024) + 8 global layers**; head_dim 256, 16 query heads, GQA 8
KV heads on the local layers / 1 (MQA) on the global. So local-layer attention cost is flat past
the 1024 window and only the 8 global layers grow with context — which is why attention is a large
*wall* share at long context despite a modest FLOP share, and why the gate sits at the window length.

## The headline

Gemma-4-12B, F16, 600 MHz, performance governor. Prefill throughput (`llama-bench`, t/s),
chaining on, the NPU offloading every prefill attention op; ≤4K rows `-r2`, 8K `-r1`
[HW sweep 2026-06-26]:

| context | CPU flash-attn | NPU offload (chained) | ratio | faster |
|---|---:|---:|---|---|
| 512   | 16.80 | 16.78 | 1.00× | tie |
| 1024  | 15.95 | 15.49 | 0.97× | CPU |
| 2048  | 14.43 | 14.65 | **1.02×** | **NPU** |
| 4096  | 12.84 | 13.76 | **1.07×** | **NPU** |
| 8192  | 8.68  | **12.54** | **1.45×** | **NPU** |

Submit chaining is what flips the short/mid context from a loss to a win. The same handler with
chaining OFF (the prior baseline) lost everywhere below ~6K — 0.80×@512, 0.83×@2K, 0.92×@4K,
crossing only at ~8K (1.32×). Collapsing each worker's per-head QK (and AV) matmuls into one
NPU job — via a per-worker resident batched-matmul context, prezeroed once — roughly doubled the
FA-op throughput and pulled the crossover in to ~2K.

The table is the offload-all ceiling (gate 0). The shipped gate is `n_kv ≥ 1024` (below), a touch
more conservative: the early-ubatch ops whose local-layer `n_kv` is still below the window stay on
the CPU.

### Multi-rep confirmation

Three timed reps per point, CPU and NPU back-to-back per depth (shared thermals), the shipped
`n_kv ≥ 1024` gate [HW sweep 2026-06-28, F16, 600 MHz, performance governor, `-r3`]:

| context | CPU flash-attn | NPU offload (gated) | ratio |
|---|---:|---:|---|
| 8192  | 8.40 ± 0.02 | **12.63 ± 0.02** | **1.50×** |
| 16384 | 8.97 ± 0.01 | **11.21 ± 0.00** | **1.25×** |

This settles the earlier single-rep 8K volatility: across three reps the CPU baseline is tight
(8.40 ± 0.02, not the 8.17–11.16 cross-session swing), the 8K win is a clean **1.50×**, and 16K is
**1.25×** — well above the prior chain-off single-rep 1.09×, so the win does not decay to parity at
depth. The NPU runs even started warmer than the CPU runs (60–61 °C vs 43–54 °C under 120 s
cooldowns) yet still won, so the ratios are conservative. On this evidence the offload is **on by
default** (context-gated; `ROCKET_FLASH_ATTN=0` disables). 32K is unmeasured (the flat-CPU /
slow-NPU-growth trend through 16K projects a continued win, and the per-op `n_kv` gate makes a
deeper regression self-limiting and overridable). At 32K the bigger lever is **chaining**, not
tiling: re-engaging head-chaining at long context (the default head-group budget, below) wins,
while an online/tiled handler loses — both measured (see "Long-context: chain the heads").

## Why they cross

The weight GEMMs run on the NPU in **both** configs — only the attention moves. CPU
flash-attention cost grows super-linearly with context (the global layers are O(L²); the
sliding-window layers cap at the window), so the CPU prefill curve falls steeply (12.8 → 8.7
t/s from 4K → 8K). The NPU attention is dispatch-bound — hundreds of small per-head GEMMs, the
chip's weak regime (see [not-mac-bound.md](not-mac-bound.md)) — but with the heads fanned across
the worker fds *and* each worker's per-head submits collapsed into one job, that cost is roughly
**flat** in context (13.8 → 12.5 over the same span). A steep line and a flat line cross: with
chaining, at ~2K. Below it the NPU's per-head dispatch floor dominates and CPU wins; above it the
CPU's super-linear attention dominates and the NPU wins.

## The levers — provenance and targeted A/B

The flat NPU curve is the product of three stacked, independently-measured levers, all bit-faithful
(per-head cosine 1.0; differential perplexity FA-NPU == FA-CPU held after each) [HW, F16, 600 MHz,
performance governor]:

1. **Multicore the heads.** Fanning the 16 heads across the 5 worker fds (one DRM scheduling entity
   per fd → the NPU cores run head ranges in parallel; each worker also gathers + softmaxes its own
   heads) lifts pp512 / pp2048 from a single-fd **7.01 / 5.73** t/s to **13.05 / 11.04** — the
   single-fd path serialises ~2300 NPU submits per forward, the dispatch-bound floor.
2. **Host softmax.** The additive mask already brings the scores host-side, so the on-NPU softmax
   was a pure round-trip; dropping it adds +5% / +10% → **13.73 / 12.19** t/s.
3. **Submit chaining** — the lever that pulls the crossover from ~6K to ~2K. Each worker's per-head
   QK matmuls share one `(Tp, dh, Kn)` shape (its AV matmuls one `(Tp, Kn, dh)`), so each set
   batches into a single NPU job — one submit + one fence for the whole head range instead of one
   per head.

Targeted A/B for the chaining lever — per-head submits vs the shipped persistent batched context
[`fabench`, `_ctx` path, T=512, nthreads=5, warm; FA-op head-range time]:

| n_kv | per-head | persistent batched | speedup |
|---|---:|---:|---|
| 512  | 186 ms | **50 ms**  | **3.7×** |
| 2048 | 363 ms | **183 ms** | **2.0×** |

The win is biggest where the per-head GEMMs are tiniest (most dispatch-bound) and narrows with
depth — but does **not** vanish there: with a large-enough head-group budget chaining still pays
1.10–1.47× out to 32K (see "Long-context: chain the heads"); the narrowing in this short-context
table is partly the 4M-elem default budget capping the group, not the GEMMs outgrowing the batch. A per-call batch — without the
resident BOs/scratch — gives the intermediate curve 1.69×@512 → 1.38×@1024 → 1.25×@2048 →
1.04×@4096; the **persistent** context (resident in/wt/out BOs + score scratch, the full-BO zero
skipped when the `(M,K,N,nbatch)` layout repeats) roughly doubles it by removing the per-call BO
alloc + zero. `ROCKET_MM_PROFILE` confirms the mechanism: NPU job-batches 896 → 280 (3.2× fewer),
`sync` 430 → 131 ms. The **userspace one-job batching is the whole win** — re-running with the
kernel one-IRQ chaining (`ROCKET_BATCH_SUBMIT=1` + `rocket_batch_submit=1`) reproduced the table
within noise, so the FA dispatch floor is the per-head submit+fence, not the IRQ count, and this
lever needs no kernel patch.

Holding the worker fds open + the per-worker score scratch resident across layers (`rocket_fa_ctx`,
removing the per-call fd-open and the 8–16 MB score-matrix mmap) is, on its own, **perf-neutral**
(pp8192 11.84 vs 11.87 t/s; pp16384 10.87 vs 10.77, within single-rep noise) — the per-call
syscall/alloc overhead is <1% of long-context wall. It is the default anyway (never worse, removes
real churn, and is the substrate the chained batching builds on), but the throughput came from the
chaining, not the fd/mmap persistence.

## Consequence for the offload gate

Gate the offload on **`n_kv` (context length), not `n_tokens`.** Under llama.cpp's 512-token
ubatching, every `FLASH_ATTN_EXT` op sees `n_tokens ≈ 512` regardless of total prompt length —
so `n_tokens` cannot tell a short prompt from a deep ubatch in a long one. `n_kv` is the
position that drives the crossover: gate `n_kv ≥ ~1024` and a mid-prefill ubatch deep in a long
context offloads while a short prompt stays on the CPU. Each ubatch independently picks the
faster backend; mixing CPU and NPU attention across a single prefill is correct (the ops are
independent).

The `~1024` gate is the model's **sliding-window length**, and that is deliberate: Gemma-4's 40
local layers cap their `n_kv` at the 1024 window, so a gate at 1024 admits them — and they are
where chaining pays most (their per-head GEMMs are the smallest, most dispatch-bound). At 8K the
offload is 1.50× *with* the local layers offloaded vs 1.15× offloading only the 8 global layers
(a `n_kv ≥ 6144` gate). The cost of the low gate is a ~3% loss at a 1024-token prompt (its ops
sit right at the gate) — trivial against the depth wins. A gate above the window would dodge that
3% but forfeit the local-layer win, so it is not worth it.

## Host split at depth — compute-bound, not gather-bound

The handler's outer gather (the strided ggml Q/K/V/mask views → dense fp16 tiles) and the F32
scatter are single-threaded host loops that the driver's `ROCKET_MM_PROFILE` does not see. A
dedicated probe (`ROCKET_FA_TIMING`) splits the FA op into gather / on-NPU compute / scatter. At
16K (n_kv 1024..16384) the aggregate is **gather 15% / compute 82% / scatter 3%**, and the gather
*share shrinks* with depth (25% at 2K → 15% at 16K) because the on-NPU per-head GEMMs grow faster
than the O(n_kv) gather. So where the offload wins, attention is **compute-bound, not
gather-bound** — threading the outer gather would touch ~6% of prefill wall and falling, not worth
the complexity. The win narrowing 1.50× → 1.25× from 8K → 16K is the on-NPU compute and the host
softmax growing with n_kv, not the gather; the online/tiled-attention follow-on (never materialize
the full `[Tp, n_kv]` score matrix) **does not pay** — see below.

## Long-context: chain the heads (wins), don't tile the softmax (loses)

Two ways to attack long-context FA, measured against the materialized per-head path (one QK
matmul over the full key axis → host mask+softmax → one AV matmul). Both keep the op
bit-faithful (cos = 1.000000 vs the fp64 oracle); only one is faster. The split is the
dispatch-bound story everywhere on this chip — **fewer, bigger submits win; more, smaller
submits lose** [HW sweep 2026-06-29, fp16, 600 MHz, `fabench` / `flash_attn_rocket`, T=512,
nthreads=5].

**Online/tiled softmax loses.** A FlashAttention-2 handler (`ROCKET_FA_TILE_KV`) that walks the
key axis in tiles carrying the running max / denom / output (fp32), so the working score tile is
`[Tp, tile]` and the full `[Tp, n_kv]` matrix (32 MB/head at 32K) is never materialized, is
**slower** than materializing the whole score matrix — and converges to it *from below* as the
tile grows back toward the full axis:

| n_kv 32768, tile width | 2048 | 4096 | 8192 | 16384 |
|---|---:|---:|---:|---:|
| tiled / materialized | 0.58× | 0.72× | 0.81× | 0.93× |

The host score-matrix bandwidth tiling saves is **not** the long-context bottleneck — dispatch
is. Each KV subdivision multiplies the per-head submit count (16 tiles × 2 matmuls vs 2),
trading a non-bottleneck (host score traffic / cache locality) for more of the actual one (NPU
submits). The online softmax is correct and *more* numerically stable (fp32 running accumulation,
cos = 1.0 incl. the per-row fully-masked-tile skip that causal / sliding-window needs), so it is
kept **opt-in, default off**: its only value is bounding the FA scratch to `[Tp, tile]` at
extreme context (a memory escape hatch, not a speed lever). It reaches the ggml backend with no
code change — the env knob flows through `rocket_flash_attn_fp16_ctx`/`_mt`.

**Re-engaging head-chaining at long context wins.** The chaining lever was scoped to short
context by a 4M-elem head-group budget (`ROCKET_FA_CHAIN_ELEMS`), on a since-disproven premise
that each head's GEMM fills a submit batch on its own at depth so there is nothing left to
collapse. Measured, collapsing a worker's whole head range into one QK + one AV job keeps paying
well past the short-context regime:

| n_kv | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|
| chained / per-head | 1.10× | 1.47× | 1.32× | 1.16× |

So the default head-group budget is now **32M** elems (was 4M): it batches a worker's ~3-head
range up to ~20K context, bounding the batched score scratch to ~150–200 MB/worker, with no
short-context change (already batched ≤2K) and no regression (chaining is bit-identical to
per-head). End-to-end this is **+3% pp8192** on Qwen3.5-0.8B-F16 — FA is a small share of a 0.8B
prefill, and the share (so the win) grows with model size and depth. The bump targets the
512-token-ubatch F16 prefill path: at a 2048-token ubatch each head's score alone exceeds a sane
budget and the win shrinks to ~1.05×, so deeper chaining is a knob (raise `ROCKET_FA_CHAIN_ELEMS`,
at resident scratch ∝ the group size), not a default.
