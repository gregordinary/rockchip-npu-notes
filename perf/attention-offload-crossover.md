# Prefill attention offload crossover

Offloading llama.cpp's fused prefill flash-attention (FA) op, `FLASH_ATTN_EXT` (per-head QK ->
mask -> softmax -> P·V), from the CPU to the NPU ties with the CPU at short context. It is a net
gain from ~2K tokens up on Gemma-4-12B F16 [HW sweep 2026-06-26]. The two backends scale differently
with context length, and the offload raises long-context prefill throughput on reasoning models
(Gemma-4 and the like). The crossover is at ~2K with per-worker QK/AV submit chaining on, which
is the default. Without chaining, the NPU's fixed per-head dispatch overhead moves the crossover
out to between 4K and 8K. Within the attention the handler implements, the offload is
bit-faithful: per-head cosine is 1.0, and differential perplexity FA-NPU == FA-CPU.

**Conditions:** RK3588, fp16, NPU clock 600 MHz, `performance` governor, Gemma-4-12B F16,
`llama-bench` prefill throughput (t/s), chaining on, the NPU offloading every prefill attention
op, 2026-06-26. The handler is `librocketnpu`'s multicore, host-softmax
`rocket_flash_attn_fp16_mt` (§"Offload target"), on the library before the FA call's host-pass
changes of 2026-09-27 to 2026-10-11 (§"Revision history"). The 8K row is a single repetition.
§"8K and 16K at the default threshold" measures the 8K point over three repetitions on
2026-06-28.

**Not characterized:** the crossover context on the library with those host-pass changes, and
end-to-end prefill at 32K.

| context | CPU flash-attn | NPU offload (chained) | ratio | faster |
|---|---:|---:|---|---|
| 512   | 16.80 | 16.78 | 1.00x | tie |
| 1024  | 15.95 | 15.49 | 0.97x | CPU |
| 2048  | 14.43 | 14.65 | 1.02x | NPU |
| 4096  | 12.84 | 13.76 | 1.07x | NPU |
| 8192  | 8.68  | 12.54 | 1.45x | NPU |

## Offload target

Under llama.cpp's default `-fa auto`, attention is one fused `GGML_OP_FLASH_ATTN_EXT` op per
layer, not separate QK/AV `mul_mat` nodes. The offload is therefore a backend op handler, not a
matmul interception. The handler gathers three inputs: the op's permuted F32 Q, the strided F16
K/V cache views, and the F16 causal/sliding-window mask. The mask is supplied as an input and
applied additively. The handler does not synthesize it. The handler then runs per-head
`scale·QKᵀ -> +mask -> softmax -> P·V` on the NPU with 2:1 GQA broadcast, and scatters the F32
result.

The handler is `librocketnpu`'s `rocket_flash_attn_fp16_mt`. It fans the heads across the
worker fds and runs the softmax on the host. Through a resident batched-matmul context, it
collapses each worker's per-head QK/AV submits into one job each.

The Gemma-4-12B shape sets the offload threshold. Its 48 layers follow a 5 local : 1 global
sliding-window pattern, which is 40 windowed-local layers (window 1024) + 8 global layers. The
head_dim is 256, with 16 query heads. GQA uses 8 KV heads on the local layers and 1 (MQA) on
the global layers.

Local-layer attention cost is therefore flat past the 1024 window, and only the 8 global layers
grow with context. Attention is a large share of wall-clock time at long context, despite a
modest share of the FLOPs, for that reason. The threshold sits at the window length for the
same reason (§"The `n_kv` offload threshold").

## Cause

### Scaling with context

The weight GEMMs run on the NPU in both configurations. Only the attention moves. CPU
flash-attention cost grows super-linearly with context: the global layers are O(L²), and the
sliding-window layers cap at the window. The CPU prefill curve therefore falls steeply (12.8 ->
8.7 t/s from 4K -> 8K).

The NPU attention is dispatch-bound. It runs hundreds of small per-head GEMMs, which is the
chip's weak regime (see [not-mac-bound.md](not-mac-bound.md)). With the heads fanned across
the worker fds *and* each worker's per-head submits collapsed into one job, that cost is
roughly flat in context (13.8 -> 12.5 over the same span). A steep line and a flat line
cross, and with chaining they cross at ~2K. Below the crossover the NPU's fixed per-head
dispatch overhead dominates and the CPU is faster. Above it the CPU's super-linear attention
dominates and the NPU is faster.

### Optimizations behind the flat NPU curve

The flat NPU curve is the product of three combined, independently measured optimizations. All
are bit-faithful: per-head cosine is 1.0, and differential perplexity FA-NPU == FA-CPU held
after each [HW sweep, F16, 600 MHz, performance governor]:

1. **Multicore the heads.** Fanning the 16 heads across the 5 worker fds lifts pp512 from a
   single-fd 7.01 t/s to 13.05, and pp2048 from 5.73 to 11.04. Each fd is one DRM scheduling
   entity, so the NPU cores run head ranges in parallel. Each worker also gathers and
   softmaxes its own heads. The single-fd path serializes ~2300 NPU submits per forward,
   which is the dispatch-bound limit.
2. **Host softmax.** The additive mask already brings the scores host-side, so an on-NPU
   softmax is a pure round-trip. Dropping it adds +5% at pp512 and +10% at pp2048, to 13.73
   and 12.19 t/s.
3. **Submit chaining.** Chaining moves the crossover from between 4K and 8K to ~2K. Each
   worker's per-head QK matmuls share one `(Tp, dh, Kn)` shape, and its AV matmuls share one
   `(Tp, Kn, dh)` shape. Here `Tp` is the query count padded to a multiple of 4. Each set
   therefore batches into a single NPU job: one submit + one fence for the whole head range
   instead of one per head.

The persistent batched context against per-head submits, in FA-op head-range time
[HW sweep]:

| n_kv | per-head | persistent batched | speedup |
|---|---:|---:|---|
| 512  | 186 ms | 50 ms  | 3.7x |
| 2048 | 363 ms | 183 ms | 2.0x |

**Conditions:** 512 queries, 5 worker threads, warm, a 4M-element head-group budget.

The gain is biggest where the per-head GEMMs are tiniest (most dispatch-bound). It narrows
with depth and does not vanish there. With a large-enough head-group budget, chaining still
gives 1.10-1.47x out to 32K (see "Long-context: chain the heads"). The narrowing in this
short-context table is partly the 4M-elem budget of that run capping the group, not the GEMMs
outgrowing the batch.

A per-call batch, without the resident BOs and scratch, gives the intermediate curve
1.69x@512 -> 1.38x@1024 -> 1.25x@2048 -> 1.04x@4096. The persistent context roughly doubles
it by removing the per-call BO alloc + zero. That context holds resident in/wt/out BOs +
score scratch, and it skips the full-BO zero when the `(M,K,N,nbatch)` layout repeats. In the
`ROCKET_MM_PROFILE` readout, NPU job-batches fall 896 -> 280 (3.2x fewer), and `sync`
430 -> 131 ms.

The userspace one-job batching is the whole gain. With the kernel one-IRQ chaining added
(`ROCKET_BATCH_SUBMIT=1` + `rocket_batch_submit=1`), the table is the same within noise. The
FA fixed overhead is therefore the per-head submit+fence, not the IRQ count, and the
optimization needs no kernel patch.

Holding the worker fds open and the per-worker score scratch resident across layers is, on
its own, perf-neutral. `rocket_fa_ctx` does both, which removes the per-call fd-open and the
8-16 MB score-matrix mmap. Throughput is pp8192 11.84 vs 11.87 t/s and pp16384 10.87 vs
10.77, within single-rep noise. The per-call syscall/alloc overhead is <1% of long-context
wall-clock time. The persistence is the default: it is never worse, it removes churn, and the
chained batching builds on it. The throughput comes from the chaining, not the fd/mmap
persistence.

## Prefill throughput by context

Submit chaining turns the short and mid context from a loss into a gain. The same handler with
chaining off loses at every measured point through 4K: 0.80x@512, 0.83x@2K, 0.92x@4K. It is
faster at 8K (1.32x). Chaining collapses each worker's per-head QK (and AV) matmuls into one
NPU job, through a per-worker resident batched-matmul context prezeroed once. Chaining roughly
doubles the FA-op throughput and moves the crossover in to ~2K.

The first table is the offload-all ceiling (threshold 0). The default threshold is
`n_kv >= 1024` (below), which is slightly more conservative. The early-micro-batch ops whose
local-layer `n_kv` is still below the window stay on the CPU.

### 8K and 16K at the default threshold

With the default `n_kv >= 1024` threshold [HW sweep 2026-06-28, F16, 600 MHz, performance
governor]:

| context | CPU flash-attn | NPU offload (gated) | ratio |
|---|---:|---:|---|
| 8192  | 8.40 ± 0.02 | 12.63 ± 0.02 | 1.50x |
| 16384 | 8.97 ± 0.01 | 11.21 ± 0.00 | 1.25x |

The 16K ratio is well above the chain-off 1.09x, so the gain does not decay to parity at depth.

At 32K, head chaining raises the FA op's throughput and an online/tiled handler lowers it.
Both are measured (see "Long-context: chain the heads").

## The `n_kv` offload threshold

Software must select the offload by `n_kv` (context length), not by `n_tokens`. Under
llama.cpp's 512-token micro-batching, every `FLASH_ATTN_EXT` op sees `n_tokens ~512` regardless
of total prompt length. `n_tokens` therefore cannot tell a short prompt from a deep micro-batch
in a long one. The position `n_kv` drives the crossover. With a threshold of `n_kv >= ~1024`, a
mid-prefill micro-batch deep in a long context offloads, and a short prompt stays on the CPU.
Each micro-batch independently picks the faster backend, and mixing CPU and NPU attention across
a single prefill is correct (the ops are independent).

The `~1024` threshold is the model's sliding-window length, by design. Gemma-4's 40 local layers
cap their `n_kv` at the 1024 window, so a threshold at 1024 admits them. They gain most from
chaining, because their per-head GEMMs are the smallest and most dispatch-bound. At 8K the
offload is 1.50x *with* the local layers offloaded vs 1.15x offloading only the 8 global
layers (a `n_kv >= 6144` threshold). The cost of the low threshold is a ~3% loss at a
1024-token prompt (its ops sit right at the threshold), which is small against the gains at
depth. A threshold above the window avoids that 3% and loses the local-layer gain.

## Host split at depth

The handler's outer gather (the strided ggml Q/K/V/mask views -> dense fp16 tiles, one walk
per view) and the F32 scatter are single-threaded host loops that the driver's
`ROCKET_MM_PROFILE` does not see. The `ROCKET_FA_TIMING` readout splits the FA op into gather,
the library call and scatter. The library call is the device waits plus the library's own host
passes: the score de-tile, the softmax and the packing. A 16K profile (n_kv 1024..16384),
from 2026-06-28 or earlier, is gather 15%, library call 82% and scatter 3%.

The gather share shrinks with depth (25% at 2K -> 15% at 16K), because the library call grows
faster than the O(n_kv) gather. Where the offload is faster, attention is therefore bound by
the library call, not by the gather. That call's host passes are a large share of it at an
encoder shape (§"The head range's host passes at an encoder shape"). Threading the outer gather
touches ~6% of prefill wall-clock time, and that share falls with depth.

The knob `ROCKET_FA_THREADS=k` (default 1, the serial walks) splits all five host walks over
the process-wide pool, each on its own outer index. It is worth 1.0389x of the pinned prefill
wall-clock time at `k`=4 on `gemma4-12b` F16 at pp2048 [HW sweep 2026-09-07]. The walks are
6.01% of that time, and four workers recover 3.75% of it. A fit of `G_k` = `A`/`k` + `B` puts
the fixed remainder at 0.57%, so more workers give almost nothing.

The gain narrows 1.50x -> 1.25x from 8K -> 16K. The library call grows with n_kv, its device
work and its host softmax alike, and the gather does not. The online/tiled attention, which
never materializes the full `[Tp, n_kv]` score matrix, is slower (see below).

## The head range's host passes at an encoder shape

At a ViT encoder's shape the FA call is set by each worker's host passes over the score
matrix, not by the device. Each column below adds one library change to the column before it
[HW sweep 2026-10-10 and 2026-10-11]:

| ms per head range | fp32 accumulator | direct store | NEON softmax | scores and P as rows |
|---|---:|---:|---:|---:|
| device wait, QK and AV | 10.5-12.1 | 10.8-11.0 | 10.5-11.3 | 9.0-9.5 |
| de-tile, QK's scores and AV's output (`read`) | 9.5-10.0 | 6.3-6.6 | 6.3-6.6 | 0.4 |
| accumulator zero and narrow (`acc`) | 5.1-6.0 | 0.12-0.14 | 0.13 | 0.11-0.12 |
| packing Q and P (`packA`) | 6.1-6.9 | 6.5-6.7 | 6.5-6.8 | 0.3 |
| host softmax | 20.8-21.2 | 22.3-22.6 | 15.1-15.6 | 15.1-15.8 |
| the FA call | 56.7-58.5 | 48.9-49.6 | 41.4-42.7 | 27.6-28.7 |

**Conditions:** Depth Anything v2-small's layer: 1370 tokens, 6 heads of 64, unmasked, on three
workers of two heads each. RK3588, 600 MHz, `ondemand` governor, A76-confined. Each column comes
from its own comparison against the column before it. The last column's comparison is one build
switched by `ROCKET_FA_ROWMAJOR`. Its cube handoff (the scores and P passed in the cube layout)
measured 9.7-11.0 ms of device waits a range,
where the NEON softmax column's own comparison measured 10.5-11.3.

The first three columns pass the scores and P between the two matmuls in the cube layout, the
cube handoff. With it, the device waits are about a quarter of the range, and the rest is host
movement of the 1372×1376 score matrix. Every change keeps the
output bit-identical.

### Direct store for a one-tile K

QK's K is the head dim, which plans as one K tile at every head dim to 256 [host-computed,
`rocket_matmul_plan`]. One tile therefore writes each score, and the NPU writes it as fp16. The
batched matmul then de-tiles straight into the caller's fp16 output. Through an fp32
accumulator, the same scores cost a 15 MB zero, a read-modify-write de-tile and a narrow, about
11 ms of a 57 ms call.

The store de-tile keeps two thirds of the accumulate's time, where the byte count says 0.4x.
Its reads of the output BO likely set its rate [hypothesis]. The softmax that follows ran 7%
slower in the same passes [hypothesis: the scores' cache state after a tile-order store].

### The softmax's own instruction rate

On one A76, built at `-O2`, the fused softmax ran at 4.9-5.1 ns a score [HW sweep]. Three
copies side by side ran at the same rate. The softmax is therefore compute-bound in its own
loops, at about 1.1 IPC.

With one vector's Horner chain per iteration, its exp pass has ~60 cycles of dependent latency
an iteration. Four independent chains a step, added into the one sum in the original order,
keep the bits and run the loop at 0.64x. The lower clamp becomes one max and the floor one
`vcvtmq`. The argument needs no upper clamp, because it is a score minus its row maximum.

### Scores and P as rows

QK writes its scores as plain rows into one BO a worker, through the stride-only row-major
form ([../matmul-as-conv.md](../matmul-as-conv.md) §"Layouts"). A QK row here is 2752 bytes,
inside the 4 KiB the form writes for free. The fused softmax rewrites each row as P in place,
and AV reads its K slices of P as rows out of the same BO. Both layout passes go.

One sync of the BO each way replaces them, and the `sync` bucket falls, 1.3 to 0.5 ms a range.
The cube handoff syncs its rolling 8 MB output BO around every job, and the packed P around
its pack. The cube handoff's device waits measured 9.7-11.0 ms a range in this comparison. The
row handoff's fall to 9.0-9.5 [hypothesis: less host DRAM traffic beside the NPU's own].

In that comparison, the call at Depth Anything's layer is 0.636-0.690x the cube handoff's. At
the Gemma-4 prefill shape it is 0.727-0.763x (88.0-90.2 to 64.8-67.2 ms). That shape is 512
queries against 2048 keys, head dim 256, 16 query heads over 8 KV heads, causal, five workers.
The output is bit-identical between the two handoffs at each shape [HW sweep 2026-10-11].

### Rows past 4 KiB

The device writes a row-major C at a cost that grows with its pitch past a 4 KiB row. At the
Gemma-4 shape the plain-row handoff's device waits double at 8192 keys, 67-69 to 134-136 ms a
range. By 16384 keys its call is within 4% of the cube handoff's.

A block of rows at one tile's own width costs the device nothing at any key count. Past a
4 KiB row, QK therefore writes each N tile as its own `[Tp][Nt]` block. The softmax reads a row
across the blocks and writes P as plain rows into a second BO. AV reads P at the full pitch.

The call against the cube handoff [HW sweep 2026-10-11]:

| Keys | Row of scores | Plain rows | Blocks |
|---|---:|---:|---:|
| 1370, Depth Anything's layer | 2.7 KiB | 0.670-0.677x | 0.667-0.673x |
| 1500, Whisper-small's encoder layer | 2.9 KiB | 0.646-0.657x | 0.646-0.673x |
| 2048, Gemma-4 shape | 4 KiB | 0.738-0.743x | 0.749-0.759x |
| 4096 | 8 KiB | 0.880-0.891x | 0.756-0.758x |
| 8192 | 16 KiB | 0.963-0.987x | 0.807-0.822x |
| 16384 | 32 KiB | 0.959-0.991x | 0.794-0.818x |
| 32768 | 64 KiB | 0.958-0.991x | 0.765-0.772x |

**Conditions:** five workers, RK3588, 600 MHz, `ondemand` governor, A76-confined. The table is its own
measurement series. The comparison in §"Scores and P as rows" measures the plain rows at
0.636-0.690x at 1370 keys and 0.727-0.763x at 2048.

The library writes plain rows to a 4 KiB row and blocks past it (`ROCKET_FA_RM_MAX_ROW`). Under
blocks the device waits stay within 5% of the cube handoff's at every width. AV's row-major
read of P therefore costs nothing measurable at a 64 KiB pitch. The softmax takes the cost of
the read across blocks instead: 66 to 76 ms a range at 8192 keys, 263-274 to 316-318 at 32768.

Each width's output is bit-identical across the three configurations, except at 32768 keys.
There the chain budget leaves some workers a last group of one head. The cube handoff runs a
lone head as a one-shot matmul, which accumulates AV's K tiles in fp16 on the device. The row
handoff keeps the batch's fp32 host accumulation, and the two differ by a few fp16 ulps on
those heads (cosine 0.9999998). With `ROCKET_KACC=0` the cube configuration matches the row
configuration bit for bit.

### End to end

Through ort-rocket:

| Model | direct store, against the accumulator | NEON softmax, against the direct store | rows, against the cube handoff |
|---|---:|---:|---:|
| Depth Anything v2-small, 1370 tokens | 0.919-0.948x | 0.919-0.949x | 0.840-0.855x |
| SigLIP, 196 tokens | 0.959-0.976x | 0.975-0.996x | 0.944-0.948x |
| CLIP, 197 tokens | 0.955-0.988x | 0.965-0.974x | 0.933-0.945x |

**Conditions:** 4 intra-op threads, spinning off, RK3588, 600 MHz, `ondemand` governor,
A76-confined.

Every model's cosine against the CPU is unchanged [HW sweep, 2026-10-10 and 2026-10-11].
RF-DETR's backbone attention changes too: nano 0.939-0.945x and base 0.906-0.947x.

Through llama.cpp, the row handoff lifts a whole prefill. Llama-3.2-3B-Instruct F16 under
`llama-bench -fa 1` at the default `-b`/`-ub` measures 1.058-1.068x the tokens/s of the cube
handoff at pp2048, and 1.080-1.116x at pp4096. Every attention op past 1024 keys offloads, and
the micro-batches past 2048 keys run blocks [HW sweep 2026-10-11, RK3588, 600 MHz, governor
`performance`].

### Remaining host terms

Two host terms grow with the key count under blocks. The softmax's read across blocks is
15-20% of the softmax at 8192 keys and past it. AV's output is de-tiled and accumulated on the
host once per K tile, 48-49 ms a range at 32768 keys. On-chip K-accumulation for AV would read
that output once **[expected]**.

## Long-context: chain the heads (wins), don't tile the softmax (loses)

Two approaches to long-context FA are measured against the materialized per-head path (one
QK matmul over the full key axis -> host mask+softmax -> one AV matmul). Both keep the op
bit-faithful (cos = 1.000000 vs the fp64 oracle), and only one is faster. The split follows
the dispatch-bound pattern of this chip. Fewer, bigger submits are faster, and more, smaller
submits are slower [HW sweep 2026-06-29].

**Conditions:** fp16, 600 MHz, 512 queries, 5 worker threads.

### Online/tiled softmax

Online/tiled softmax is slower. A FlashAttention-2 handler (`ROCKET_FA_TILE_KV`) walks the key
axis in tiles, keeping the running max, denom and output in fp32. The working score tile is
`[Tp, tile]`, and the full `[Tp, n_kv]` matrix (32 MB/head at 32K) is never materialized.
That handler is slower than materializing the whole score matrix. It converges to the
materialized speed *from below* as the tile grows back toward the full axis:

| n_kv 32768, tile width | 2048 | 4096 | 8192 | 16384 |
|---|---:|---:|---:|---:|
| tiled / materialized | 0.58x | 0.72x | 0.81x | 0.93x |

Tiling loses on dispatch. Each KV subdivision multiplies the per-head submit count (16 tiles
× 2 matmuls vs 2). At this shape and date, what tiling saves in host score traffic and cache
locality is smaller than what the extra NPU submits cost.

The host passes are still large. At the Gemma-4 shape and 32768 keys, the cube handoff's
score de-tile and P pack take 290-296 ms a head range [HW sweep 2026-10-11]. The range's device
waits are 282 ms [HW sweep 2026-10-11]. The row handoff removes both (§"Rows past 4 KiB").

The online softmax is correct and *more* numerically stable (fp32 running accumulation,
cos = 1.0 incl. the per-row fully-masked-tile skip that causal / sliding-window needs). It is
opt-in, default off. Its only value is bounding the FA scratch to `[Tp, tile]` at extreme
context, as a memory limit rather than a speed optimization. It reaches the ggml backend with
no code change: the environment variable passes through `rocket_flash_attn_fp16_ctx`/`_mt`.

The masked-tile skip belongs to that tiled path alone, which engages from 8192 keys
(`ROCKET_FA_TILE_MIN_KV`). The default materialized path computes the dense score matrix on a
sliding-window layer too. The ModernBERT encoder in `librocketnpu` bands its windowed layers
itself, with query tiles of 128 rows against their own keys plus the window. The banding
measures -13% at 639 tokens and -14% at 726 [HW sweep, RK1, 600 MHz, 2026-09-27].

### Head chaining at long context

Head chaining also raises throughput at long context. Collapsing a worker's whole head range
into one QK and one AV job keeps raising it well past the short-context regime, against per-head
submits:

| n_kv | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|
| chained / per-head | 1.10x | 1.47x | 1.32x | 1.16x |

The default head-group budget (`ROCKET_FA_CHAIN_ELEMS`) is therefore 32M elems, not 4M. It
batches a worker's ~3-head range up to ~20K context, which bounds the batched score scratch
to ~150-200 MB/worker. It changes nothing at short context (already batched <=2K) and
regresses nothing (chaining is bit-identical to per-head). End-to-end this is +3% pp8192 on
Qwen3.5-0.8B-F16, since FA is a small share of a 0.8B prefill. The share, and with it the
gain, grows with model size and depth [expected].

The 32M budget targets the 512-token-micro-batch F16 prefill path. At a 2048-token
micro-batch each head's score alone exceeds a practical budget, and the gain shrinks to
~1.05x. Deeper chaining is therefore an option, not a default: raise `ROCKET_FA_CHAIN_ELEMS`,
at resident scratch ∝ the group size.

## Software support

| Feature | Control | Status |
|---|---|---|
| Prefill attention offload, at `n_kv >= 1024` | `ROCKET_FLASH_ATTN=0` disables it | Supported (default) |
| Attention with a sink, `src[4]` | None | Not supported. The op stays on the CPU |
| Head chaining, 32M-element head-group budget | `ROCKET_FA_CHAIN_ELEMS` | Supported (default) |
| Worker fds and score scratch held across layers | `rocket_fa_ctx` | Supported (default) |
| Scores and P as rows to a 4 KiB row, blocks past it | `ROCKET_FA_ROWMAJOR`, `ROCKET_FA_RM_MAX_ROW` | Supported (default) |
| Online/tiled softmax, from 8192 keys | `ROCKET_FA_TILE_KV`, `ROCKET_FA_TILE_MIN_KV` | Supported (opt-in) |
| Threaded gather and scatter walks | `ROCKET_FA_THREADS=k`, default 1 | Supported (opt-in) |

The handler implements one attention, `softmax(scale·QKᵀ + mask) · V`, and it declines anything
else. An attention sink is a learned per-head logit that joins the softmax denominator, passed
as the op's `src[4]` by `ggml_flash_attn_ext_add_sinks`. It is a *different* attention, and the
handler has no term for it, so the offload declines an op with `src[4]` set. The gpt-oss model
has sinks on every layer, and so do `mimo2` and `deepseek4`.

> [!CAUTION]
> A `supports_op` that validates `src[0..3]` and never reads `src[4]` accepts a sink op. The
> NPU then computes a sink-less softmax, the wrong function, with no error. On gpt-oss it fires
> only past the `n_kv` threshold of 1024, so short-prompt tests never reach it. A wrong but
> plausible attention still produces fluent text.

> [!NOTE]
> Differential perplexity does not distinguish a wrong attention function from run-to-run
> noise. On gpt-oss, the sink-less softmax measured "+1.0%, within noise".

Implementing the sink is easy in principle: the softmax is host-side, so the sink is one more
term in the denominator. It is only worth doing where the offload gains, and on gpt-oss's head
geometry it does not. The gpt-oss geometry is 64 heads of dimension 64, where Gemma-4 has 16 of
dimension 256. On gpt-oss the offload therefore issues more per-head NPU matmuls, each 4x
thinner.

## Applicability

The crossover context depends on the handler's host overhead and on the CPU's flash-attention
speed, at the operating point in the Conditions block. The flat CPU and slowly falling NPU
throughput through 16K project a continued gain at 32K [expected]. The per-op `n_kv` threshold
makes a deeper regression self-limiting and overridable.

## Evidence

### Revision history

The crossover figures were measured 2026-06-26 and 2026-06-28. The FA call's host passes have
changed since, each change bit-identical:

| Date | Library change | FA call at the Gemma-4 prefill shape, against the build before |
|---|---|---|
| 2026-09-27 | A fused softmax pass | n/a |
| 2026-10-10 | A direct fp16 de-tile (the store) | 0.875-0.896x |
| 2026-10-10 | A NEON softmax | 0.910-0.912x |
| 2026-10-11 | Scores and P handed between the two matmuls as rows | 0.73-0.82x, from 2048 to 32768 keys |

### Offload target

A `supports_op` graph dump confirms that `-fa auto` presents one fused op per layer.

### Prefill throughput by context

The first table's <=4K rows are `-r2` and the 8K row is `-r1`. The chain-off figures and the
16K chain-off 1.09x are single-rep.

The 8K and 16K table is three timed reps a point, `-r3`, with CPU and NPU back-to-back
per depth (shared thermals). Across three reps the CPU baseline is tight: 8.40 ± 0.02, against
a single-rep cross-session swing of 8.17-11.16 at 8K. The NPU runs started warmer than the CPU
runs (60-61 °C vs 43-54 °C under 120 s cooldowns) and still won, so the ratios are
conservative.

### Optimizations behind the flat NPU curve

The chaining table is `fabench`, `_ctx` path, FA-op head-range time. The `rocket_fa_ctx`
readings are single-rep.

### The head range's host passes at an encoder shape

Each value in the encoder table is a bucket's sum over 114 ranges, divided by 114. The run is
`fabench 1370 1370 3 30 64 6 6 0` under `ROCKET_FA_PROFILE` and `ROCKET_MM_PROFILE`, four
processes per configuration, configuration order rotated. Each column is compared against the
one before it, and bit-identity is one md5 over every run of each pair.

The row handoff's call ratios are paired within each of four rotated passes. Both
configurations are one build, switched by `ROCKET_FA_ROWMAJOR`, and the output's md5 is one
value over all sixteen runs at each shape. The past-4 KiB table is `fabench`, paired within
each of two rotated passes, and bit-identity is one md5 across the three configurations.

Through ort-rocket, Depth Anything is four passes, SigLIP and CLIP two. The first two columns
compare library builds, and the last switches one build by `ROCKET_FA_ROWMAJOR`. RF-DETR is
two passes each. Through llama.cpp the configurations are one build, three passes with the
order rotated, memory reset before every process.

### Long-context: chain the heads (wins), don't tile the softmax (loses)

The long-context tables are `fabench` / `flash_attn_rocket`. The 32768-key host-pass figures
are `fabench` at the Gemma-4 shape.

### Excluded causes

A 4M-element head-group budget scoped chaining to short context. Its premise was that each
head's GEMM fills a submit batch on its own at depth, so nothing is left to collapse. The
long-context chaining table refutes that premise.
