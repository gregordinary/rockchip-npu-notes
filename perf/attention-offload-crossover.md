# Prefill attention offload crossover

Offloading the fused prefill attention op from the CPU to the NPU is a wash at short context.
It is a net win from ~2K tokens up. The op is `FLASH_ATTN_EXT`: per-head QK -> mask -> softmax
-> P·V. The two backends scale differently with context length. This is the lever that moves
long-context prefill on reasoning models (Gemma-4 and the like). The crossover is at ~2K with
per-worker QK/AV submit chaining on, which is the default. Without chaining, the NPU per-head
dispatch floor moves the crossover out to ~6K.

> **Scope.** The numbers are measured at the current operating point: fp16, 600 MHz, and the
> multicore + host-softmax attention handler (`librocketnpu` `rocket_flash_attn_fp16_mt`).
> That handler fans the heads across the worker fds and collapses the per-head QK/AV submits
> into one job each through a resident batched-matmul context. The crossover *context*
> depends on that handler's host overhead and on the CPU's flash-attention speed. It is not a
> fixed property of the silicon. Within the attention shape the handler implements, the
> offload is bit-faithful (differential perplexity FA-NPU == FA-CPU, per-head cosine 1.0).
> Only the speed crosses over.

> **The handler implements one attention, and it must decline anything else.** It computes
> `softmax(scale·QKᵀ + mask) · V`. An attention sink is a learned per-head logit that joins
> the softmax denominator, passed as the op's `src[4]` by `ggml_flash_attn_ext_add_sinks`. It
> is a *different* attention, and the handler has no term for it. Accepting such an op does
> not lose a little accuracy. It computes the wrong function, silently, and the graph cannot
> tell.
>
> A `supports_op` that validates `src[0..3]` and never reads `src[4]` accepts such an op. The
> gpt-oss model carries sinks on every layer, and so do `mimo2` and `deepseek4`. Under that
> gate gpt-oss takes the offload and gets a sink-less softmax. The defect hides well. It
> fires only past the `n_kv` floor of 1024, so short-prompt tests never reach it. A
> wrong-but-plausible attention still produces fluent text, and on gpt-oss the
> differential-PPL check read it as "+1.0%, within noise".
>
> **A faithfulness metric that cannot distinguish a wrong function from run-to-run noise is
> not a faithfulness metric.** The gate declines on `src[4]`. Implementing the sink is easy
> in principle: the softmax is host-side, so the sink is one more term in the denominator. It
> is only worth doing where the offload *wins*, and on gpt-oss's head geometry it does not.
> The gpt-oss geometry is 64 heads of dimension 64, where Gemma-4 has 16 of dimension 256. So
> gpt-oss issues more per-head NPU matmuls, each 4x thinner.

## Offload target

Under llama.cpp's default `-fa auto`, attention is one fused `GGML_OP_FLASH_ATTN_EXT` op per
layer, not separate QK/AV `mul_mat` nodes (confirmed by a `supports_op` graph dump). So the
offload is a backend op handler, not a matmul interception. The handler gathers three inputs:
the op's permuted F32 Q, the strided F16 K/V cache views, and the F16 causal/sliding-window
mask. The mask is supplied as an input and applied additively. The handler does not
synthesize it. The handler then runs per-head `scale·QKᵀ -> +mask -> softmax -> P·V` on the
NPU with 2:1 GQA broadcast, and scatters the F32 result.

The Gemma-4-12B shape sets the gate. Its 48 layers follow a 5 local : 1 global sliding-window
pattern, which is 40 windowed-local layers (window 1024) + 8 global layers. The head_dim is
256, with 16 query heads. GQA uses 8 KV heads on the local layers and 1 (MQA) on the global
layers.

So local-layer attention cost is flat past the 1024 window, and only the 8 global layers grow
with context. That is why attention is a large *wall* share at long context despite a modest
flop share. It is also why the gate is at the window length.

## Prefill throughput by context

The operating point is Gemma-4-12B, F16, 600 MHz, performance governor. The table is prefill
throughput (`llama-bench`, t/s) with chaining on and the NPU offloading every prefill
attention op. The <=4K rows are `-r2` and the 8K row is `-r1` [HW sweep 2026-06-26]:

| context | CPU flash-attn | NPU offload (chained) | ratio | faster |
|---|---:|---:|---|---|
| 512   | 16.80 | 16.78 | 1.00x | tie |
| 1024  | 15.95 | 15.49 | 0.97x | CPU |
| 2048  | 14.43 | 14.65 | 1.02x | NPU |
| 4096  | 12.84 | 13.76 | 1.07x | NPU |
| 8192  | 8.68  | 12.54 | 1.45x | NPU |

Submit chaining turns the short and mid context from a loss into a win. The same handler with
chaining off loses everywhere below ~6K: 0.80x@512, 0.83x@2K, 0.92x@4K. It crosses only at
~8K (1.32x). Chaining collapses each worker's per-head QK (and AV) matmuls into one NPU job,
through a per-worker resident batched-matmul context prezeroed once. That roughly doubles the
FA-op throughput and moves the crossover in to ~2K.

The table is the offload-all ceiling (gate 0). The shipped gate is `n_kv >= 1024` (below),
which is slightly more conservative. The early-micro-batch ops whose local-layer `n_kv` is
still below the window stay on the CPU.

### Multi-rep confirmation

Each point is three timed reps, with CPU and NPU back-to-back per depth (shared thermals).
The runs use the shipped `n_kv >= 1024` gate
[HW sweep 2026-06-28, F16, 600 MHz, performance governor, `-r3`]:

| context | CPU flash-attn | NPU offload (gated) | ratio |
|---|---:|---:|---|
| 8192  | 8.40 ± 0.02 | 12.63 ± 0.02 | 1.50x |
| 16384 | 8.97 ± 0.01 | 11.21 ± 0.00 | 1.25x |

Across three reps the CPU baseline is tight: 8.40 ± 0.02, against a single-rep cross-session
swing of 8.17-11.16 at 8K. The 8K win is 1.50x and the 16K win is 1.25x. The 16K figure is
well above the chain-off single-rep 1.09x, so the win does not decay to parity at depth. The
NPU runs started warmer than the CPU runs (60-61 °C vs 43-54 °C under 120 s cooldowns) and
still won, so the ratios are conservative.

On this evidence the offload is on by default. It is context-gated, and `ROCKET_FLASH_ATTN=0`
disables it. The 32K point is unmeasured. The flat-CPU / slow-NPU-growth trend through 16K
projects a continued win, and the per-op `n_kv` gate makes a deeper regression self-limiting
and overridable.

At 32K the bigger lever is chaining, not tiling. Re-engaging head-chaining at long context
(the default head-group budget, below) wins, and an online/tiled handler loses. Both are
measured (see "Long-context: chain the heads").

## Scaling with context

The weight GEMMs run on the NPU in both configs. Only the attention moves. CPU
flash-attention cost grows super-linearly with context: the global layers are O(L²), and the
sliding-window layers cap at the window. So the CPU prefill curve falls steeply (12.8 -> 8.7
t/s from 4K -> 8K).

The NPU attention is dispatch-bound. It runs hundreds of small per-head GEMMs, which is the
chip's weak regime (see [not-mac-bound.md](not-mac-bound.md)). With the heads fanned across
the worker fds *and* each worker's per-head submits collapsed into one job, that cost is
roughly flat in context (13.8 -> 12.5 over the same span). A steep line and a flat line
cross, and with chaining they cross at ~2K. Below the crossover the NPU's per-head dispatch
floor dominates and the CPU wins. Above it the CPU's super-linear attention dominates and the
NPU wins.

## Levers behind the flat NPU curve

The flat NPU curve is the product of three stacked, independently measured levers. All are
bit-faithful: per-head cosine is 1.0, and differential perplexity FA-NPU == FA-CPU held after
each [HW sweep, F16, 600 MHz, performance governor]:

1. **Multicore the heads.** Fanning the 16 heads across the 5 worker fds lifts pp512 from a
   single-fd 7.01 t/s to 13.05, and pp2048 from 5.73 to 11.04. Each fd is one DRM scheduling
   entity, so the NPU cores run head ranges in parallel. Each worker also gathers and
   softmaxes its own heads. The single-fd path serializes ~2300 NPU submits per forward,
   which is the dispatch-bound floor.
2. **Host softmax.** The additive mask already brings the scores host-side, so an on-NPU
   softmax is a pure round-trip. Dropping it adds +5% at pp512 and +10% at pp2048, to 13.73
   and 12.19 t/s.
3. **Submit chaining.** This lever moves the crossover from ~6K to ~2K. Each worker's
   per-head QK matmuls share one `(Tp, dh, Kn)` shape, and its AV matmuls share one
   `(Tp, Kn, dh)` shape. So each set batches into a single NPU job: one submit + one fence
   for the whole head range instead of one per head.

A targeted A/B for the chaining lever compares per-head submits against the shipped
persistent batched context
[`fabench`, `_ctx` path, T=512, nthreads=5, warm, FA-op head-range time]:

| n_kv | per-head | persistent batched | speedup |
|---|---:|---:|---|
| 512  | 186 ms | 50 ms  | 3.7x |
| 2048 | 363 ms | 183 ms | 2.0x |

The win is biggest where the per-head GEMMs are tiniest (most dispatch-bound). It narrows
with depth and does not vanish there. With a large-enough head-group budget, chaining still
pays 1.10-1.47x out to 32K (see "Long-context: chain the heads"). The narrowing in this
short-context table is partly the 4M-elem budget of that run capping the group, not the GEMMs
outgrowing the batch.

A per-call batch, without the resident BOs and scratch, gives the intermediate curve
1.69x@512 -> 1.38x@1024 -> 1.25x@2048 -> 1.04x@4096. The persistent context roughly doubles
it by removing the per-call BO alloc + zero. That context holds resident in/wt/out BOs +
score scratch, and it skips the full-BO zero when the `(M,K,N,nbatch)` layout repeats. The
`ROCKET_MM_PROFILE` readout confirms the mechanism: NPU job-batches 896 -> 280 (3.2x fewer),
`sync` 430 -> 131 ms.

The userspace one-job batching is the whole win. A re-run with the kernel one-IRQ chaining
(`ROCKET_BATCH_SUBMIT=1` + `rocket_batch_submit=1`) reproduces the table within noise. So the
FA dispatch floor is the per-head submit+fence, not the IRQ count, and this lever needs no
kernel patch.

Holding the worker fds open and the per-worker score scratch resident across layers is, on
its own, perf-neutral. That is `rocket_fa_ctx`, which removes the per-call fd-open and the
8-16 MB score-matrix mmap. The readings are pp8192 11.84 vs 11.87 t/s and pp16384 10.87 vs
10.77, within single-rep noise. The per-call syscall/alloc overhead is <1% of long-context
wall. The persistence is the default anyway: it is never worse, it removes churn, and the
chained batching builds on it. The throughput comes from the chaining, not the fd/mmap
persistence.

## The offload gate

**Gate the offload on `n_kv` (context length), not `n_tokens`.** Under llama.cpp's 512-token
micro-batching, every `FLASH_ATTN_EXT` op sees `n_tokens ~512` regardless of total prompt
length. So `n_tokens` cannot tell a short prompt from a deep micro-batch in a long one. The
position `n_kv` drives the crossover. With a gate of `n_kv >= ~1024`, a mid-prefill
micro-batch deep in a long context offloads, and a short prompt stays on the CPU. Each
micro-batch independently picks the faster backend, and mixing CPU and NPU attention across a
single prefill is correct (the ops are independent).

The `~1024` gate is the model's sliding-window length, by design. Gemma-4's 40 local layers
cap their `n_kv` at the 1024 window, so a gate at 1024 admits them. They are where chaining
pays most, because their per-head GEMMs are the smallest and most dispatch-bound. At 8K the
offload is 1.50x *with* the local layers offloaded vs 1.15x offloading only the 8 global
layers (a `n_kv >= 6144` gate). The cost of the low gate is a ~3% loss at a 1024-token prompt
(its ops sit right at the gate), which is trivial against the depth wins. A gate above the
window would avoid that 3% and forfeit the local-layer win, so it is not worth it.

## Host split at depth

The handler's outer gather (the strided ggml Q/K/V/mask views -> dense fp16 tiles) and the F32
scatter are single-threaded host loops that the driver's `ROCKET_MM_PROFILE` does not see. A
dedicated probe (`ROCKET_FA_TIMING`) splits the FA op into gather, on-NPU compute and scatter.
At 16K (n_kv 1024..16384) the aggregate is gather 15%, compute 82% and scatter 3%. The gather
share shrinks with depth (25% at 2K -> 15% at 16K), because the on-NPU per-head GEMMs grow
faster than the O(n_kv) gather. So where the offload wins, attention is compute-bound, not
gather-bound. Threading the outer gather touches ~6% of prefill wall, and that share falls
with depth.

The knob `ROCKET_FA_THREADS=k` splits all five host walks over the process-wide pool, each on
its own outer index. It is worth 1.0389x of the pinned prefill wall at `k`=4 on `gemma4-12b`
F16 at pp2048 [HW sweep 2026-09-07].

The 6% above is also the right size. The walks are 6.01% of that wall, and four workers
recover 3.75% of it. A fit of `G_k` = `A`/`k` + `B` puts the fixed residue at 0.57%, so more
workers buy almost nothing.

The win narrowing 1.50x -> 1.25x from 8K -> 16K is the on-NPU compute and the host softmax
growing with n_kv, not the gather. The online/tiled-attention follow-on (never materialize the
full `[Tp, n_kv]` score matrix) does not pay (see below).

## Long-context: chain the heads (wins), don't tile the softmax (loses)

Two approaches to long-context FA are measured against the materialized per-head path (one
QK matmul over the full key axis -> host mask+softmax -> one AV matmul). Both keep the op
bit-faithful (cos = 1.000000 vs the fp64 oracle), and only one is faster. The split follows
the dispatch-bound pattern seen everywhere on this chip. Fewer, bigger submits win, and more,
smaller submits lose
[HW sweep 2026-06-29, fp16, 600 MHz, `fabench` / `flash_attn_rocket`, T=512, nthreads=5].

### Online/tiled softmax

Online/tiled softmax loses. A FlashAttention-2 handler (`ROCKET_FA_TILE_KV`) walks the key
axis in tiles, carrying the running max, denom and output in fp32. The working score tile is
`[Tp, tile]`, and the full `[Tp, n_kv]` matrix (32 MB/head at 32K) is never materialized.
That handler is slower than materializing the whole score matrix. It converges to the
materialized speed *from below* as the tile grows back toward the full axis:

| n_kv 32768, tile width | 2048 | 4096 | 8192 | 16384 |
|---|---:|---:|---:|---:|
| tiled / materialized | 0.58x | 0.72x | 0.81x | 0.93x |

The host score-matrix bandwidth that tiling saves is not the long-context bottleneck.
Dispatch is. Each KV subdivision multiplies the per-head submit count (16 tiles × 2 matmuls
vs 2). That trades a non-bottleneck (host score traffic / cache locality) for more of the
bottleneck (NPU submits).

The online softmax is correct and *more* numerically stable (fp32 running accumulation,
cos = 1.0 incl. the per-row fully-masked-tile skip that causal / sliding-window needs). So it
is kept opt-in, default off. Its only value is bounding the FA scratch to `[Tp, tile]` at
extreme context (a memory escape hatch, not a speed lever). It reaches the ggml backend with
no code change: the env knob flows through `rocket_flash_attn_fp16_ctx`/`_mt`.

The masked-tile skip belongs to that tiled path alone, which engages from 8192 keys
(`ROCKET_FA_TILE_MIN_KV`). The default materialized path computes the dense score matrix on a
sliding-window layer too. The ModernBERT encoder in `librocketnpu` bands its windowed layers
itself, with query tiles of 128 rows against their own keys plus the window. That reads -13%
at 639 tokens and -14% at 726 [HW sweep, RK1, 600 MHz, 2026-09-27].

### Head-chaining at long context

Re-engaging head-chaining at long context wins. A 4M-elem head-group budget
(`ROCKET_FA_CHAIN_ELEMS`) scopes the chaining lever to short context. The premise behind that
budget is that each head's GEMM fills a submit batch on its own at depth, so nothing is left
to collapse. The measurement refutes that premise. Collapsing a worker's whole head range into
one QK + one AV job keeps paying well past the short-context regime:

| n_kv | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|
| chained / per-head | 1.10x | 1.47x | 1.32x | 1.16x |

So the default head-group budget is 32M elems, not 4M. It batches a worker's ~3-head range up
to ~20K context, which bounds the batched score scratch to ~150-200 MB/worker. It changes
nothing at short context (already batched <=2K) and regresses nothing (chaining is
bit-identical to per-head). End-to-end this is +3% pp8192 on Qwen3.5-0.8B-F16, since FA is a
small share of a 0.8B prefill. The share, and so the win, grows with model size and depth.

The 32M budget targets the 512-token-micro-batch F16 prefill path. At a 2048-token
micro-batch each head's score alone exceeds a sane budget, and the win shrinks to ~1.05x. So
deeper chaining is a knob (raise `ROCKET_FA_CHAIN_ELEMS`, at resident scratch ∝ the group
size), not a default.
