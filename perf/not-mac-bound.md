# Matmul throughput and its limits

On the `rocket` path, the whole-tile matmul program measures ~460 GOP/s resident and multicore
at fp16, int8 and int4 alike [HW sweep]. In that program each task holds both operand tiles in
the CBUF. The rate is ~15% of the fp16 MAC peak and ~4% of the int4 peak. That program is
bound by a datatype-independent limit, not by the MAC array. The limit is the DMA that loads
operands into the CBUF, plus the per-job dispatch latency. int8's 2x MAC rate and int4's 4x MAC
rate therefore do not raise its throughput, and quantization reduces memory use rather than
prefill time.

**Conditions:** RK3588, NPU clock ~816 MHz, resident weights, three cores reached through 5
worker fds, `512×3840×4096` unless a table states another shape.

**Not characterized:** an operating point where DMA and dispatch do not bind
(§"Applicability"), and the datatype comparison with streamed weights.

| datatype | GOP/s | MAC advantage | Advantage realized |
|---|---:|---|---|
| fp16 (+K-accumulation +DATA_REUSE) | 461 | 1x | n/a |
| int8 | 386 | 2x | no |
| int4 | 413 | 4x | no |

Other shapes measure in the same band (GOP/s):

| shape | fp16 | int8 | int4 |
|---|---:|---:|---:|
| small-tile | 453 | 478 | 439 (int4 single-pass K) |
| ffn-down | 486 | 452 | 463 |

Every datatype is within ~20% of the others.

A task that streams its weights through the CBUF removes most of the whole-tile program's
tasks. On one core it runs `160×1024×4096` at 3.4x the best whole-tile tiling. As the library
default, it gives 1.07-1.17x on prefill across four models (§"Streamed weights").

## Cause

### MAC array and readback

The matmul is not bound by the MAC array. If it were, int8's 2x and int4's 4x MAC advantage
would raise throughput, and they do not.

K-tile readback does not determine the common limit either. The int4 single-pass-K run (`nKt=1`,
*zero* K-tile readback) is no faster than the rest (439). Readback only lowers a datatype below
the limit. The un-K-accumulated int32 readback of int8, ∝ `M·N·nKt`, is extra traffic that does
so (§"In-model int8 and int4 prefill"). Removing readback from a datatype already at the limit
(fp16 with K-accumulation, int4 single-pass) does not raise it.

What remains is the DMA and dispatch limit: loading tiles into the CBUF, and the per-job launch
and fence overhead. Both are datatype-independent.

### In-model int8 and int4 prefill

In the live model, int8 prefill is slower than fp16. Resident int8 prefill is ~9.1 t/s against
resident fp16's ~15.1 t/s, 0.60x [HW sweep, ~816 MHz].

The fp16 path has on-NPU K-accumulation (`read ∝ M·N`), and int8 does not. No integer EW path is
supported, and fp32 cannot hold int8's int32 partials exactly (see
[../encodings/k-accumulation.md](../encodings/k-accumulation.md)). The un-K-accumulated int32
readback of int8 (`∝ M·N·nKt`, 2x the bytes of fp16) costs more than its MAC advantage gains.
Making int8 *resident* removes the per-call requant and packB overhead (+25-58% over the
non-resident int8 path). The readback cost is unchanged, and it holds int8 below fp16.

A same-weights precision sweep shows the same in a model. It runs the Gemma-4-12B F16 GGUF
through the *one-shot* paths, which re-quantize on each prefill. The sweep runs at pp512 under
the `performance` governor [HW sweep, 2026-06-24, RK1 7.1.0-1, ~816 MHz]:

| path | t/s | vs fp16 |
|---|---:|---:|
| fp16 + KACC | 14.4 | 1.00x |
| int8 + Hadamard | 6.9 | 0.48x |
| int4 + Hadamard | 4.4 | 0.30x |
| CPU (8 threads) | 4.6 | 0.32x |

Prefill throughput falls with each step down in precision, because the one-shot quantization
and the integer readback add to the datatype-independent limit. These are the per-call-requant
paths. The *resident* int8 path above amortizes the requant and measures higher (0.60x), still
below fp16.

A quantized `Q8_0` GGUF, dequantized to fp16 on the fly and at half the RAM (11.8 GB), runs at
5.7 t/s. The Q8_0 model is slower than native F16, because of the per-call dequantization, and
faster than the CPU, at half the memory. The dequantization runs once per micro-batch and amortizes with
`-ub`: raising the micro-batch gives 0.91-1.53x on quantized prefill, depending on the model. The
`ROCKET_MIN_M_QUANT` threshold keeps quantized prefills below the crossover on the CPU (see
[quant-prefill-microbatch.md](quant-prefill-microbatch.md)).

### Resident group-wise int4

Resident group-wise int4 shows the resident-int8 result more strongly. Holding the group-wise
Hadamard int4 weights resident (`ROCKET_INT4_RESIDENT`) removes the per-call weight scatter, as
resident int8 does. The in-model gain is ~2x over the one-shot int4 path: 6.94 t/s against 3.56,
with fp16 at 13.18, at pp512 and ~816 MHz [HW sweep, 2026-06-25]. Resident int4 is still 0.53x
fp16.

A group-wise int4 matmul reads back one int16 partial *per K-group* (`read ∝ M·N·nKt`,
`nKt = K/group ~120` on the deep K=15360 FFN). The un-K-accumulated readback that holds int8 back
therefore holds int4 back further, because finer groups mean more readback.

The resident int4 weights are ¼ the NPU-BO bytes of fp16 (2634 MB against ~10.5 GB for the
offloaded set). The resident path keeps that memory saving and raises int4 to half fp16's speed.

### Vendor-stack int8 prefill

The proprietary rknpu2 / rk-llama.cpp stack runs int8 LLM prefill faster than its *own* fp16,
and the `rocket` path does not. The difference is operand bandwidth at a lower dispatch limit,
not a hardware feature the open path cannot reach.

RKLLM on the RK3588 quantizes to W8A8, 8-bit weights *and* activations, the only LLM
quantization the chip's toolkit offers. int8 there halves the bytes moved for both operands. The
vendor's own Gemma int8 prefill runs at 40-58% NPU utilization, so that stack is not bound by
the MAC array either. int8 reduces its bytes rather than its MAC time. On a path bound by
dispatch and DMA, halving the operand bytes helps where the 2x MAC rate cannot. [W8A8-only is
the vendor LLM path, and the utilization is from rk-llama.cpp forum benchmarks, a single
external source.]

The rest of the vendor's advantage is dispatch efficiency, in three parts:

- A batched whole-graph submit, with ~10x fewer kernel transitions (see
  [iova-and-multicore.md](iova-and-multicore.md) §batched submit)
- 3-core dispatch, which the `rocket` path matches
- The 1 GHz default clock, against ~816 MHz under the `rocket` clock patch

The "+200-400%" quoted for the vendor's int8 prefill is the 6-vs-3 TOPS spec-sheet ratio
(6 TOPS int8 / 3 TOPS fp16), not a measured fp16-to-int8 comparison. The measured 40-58%
utilization is the better guide, and it independently matches the result at the top of this
note.

Two other mechanisms do not explain the vendor's advantage:

- **On-device int32 K-accumulation.** The convolution accumulates K on-chip only *within one
  CBUF-resident K-tile* [source-confirmed: Mesa `rkt_task.c`]. Beyond one tile, the partials
  are summed through DRAM, through the DPU-EW path that `ROCKET_KACC` uses for fp16 [HW sweep,
  source-confirmed]. A K-accumulation on that path's integer ALU is not characterized, and no
  stack is known to accumulate int32 across tiles on the device. The mechanism is in
  [../encodings/k-accumulation.md](../encodings/k-accumulation.md).
- **On-chip SRAM.** The 956 KB NPU SRAM stages whole *weight* or *internal (activation)* tensors
  to relieve DDR bandwidth, not partial sums. The vendor's own documentation says it can "have a
  certain impact on inference time", and its per-layer example is a vision CNN. The SRAM's
  956 KB cannot hold an LLM's weights or a prefill activation matrix [source-confirmed: see
  [sram-nbuf.md](sram-nbuf.md)].

The bandwidth account rests on the capability facts (hardware and source-confirmed) and the
vendor's utilization figures [hypothesis].

### Quantization's benefits

Quantization on this hardware reduces memory use, not prefill time:

- **Model size.** A model that does not fit in fp16 can run. The int4 Gemma weights are ~¼ the
  bytes, and the int4 matmul is complete and bit-exact.
- **IOVA residency.** A quantized model fits the per-fd IOVA windows whole (see
  [iova-and-multicore.md](iova-and-multicore.md)).
- **Decode alongside prefill.** A quantized GGUF serves CPU decode alongside NPU prefill.

### Quantized MoE experts and residency

A MoE routed expert is the one place where quantization raises speed, because it makes the
expert resident.

On the streaming path, the host dequantizes a quantized expert to fp16 on every micro-batch.
That dequantization does not depend on `M`: it decodes the whole `[N,K]` weight whatever row
count the router gives that expert. On gpt-oss-20b's `[2880,2880]` MXFP4 experts it measures
75 ms per expert, per micro-batch, and a prefill touches ~1580 of them. Dequantizing weights
therefore takes the host ~119 s per prefill, 59% of a pp2048 prefill, before any arithmetic runs.
Offloading quantized MoE experts through the fp16 path is slower than leaving them on the CPU
for that reason [HW sweep, ~816 MHz, 2026-07-14].

Ingesting each expert once into resident int8 codes removes that cost. The speedup is not from
int8 arithmetic: the int8 GEMM's int32 output reads back at 8 B/element and moves *more* bytes
than the fp16 one would. It is from the weight staying on the device instead of crossing the host
on every micro-batch.

Partial residency does not remove a proportional share of the cost. A streamed expert costs the
full, M-independent 75 ms, while a resident one costs a small GEMM that *shrinks with M*. The
streamed remainder's share of wall-clock time therefore grows as the prefill gets shorter. With
85% of the experts resident:

| prefill | share of the prefill spent on the streamed 15% |
|---|---:|
| pp2048 | 12% |
| pp512 | 43% |

The native-quantization path wins at long prefill and loses at short for that reason. The
resident weights' memory efficiency is a first-order throughput factor. Every byte of tile
padding is an expert that has to stream instead, and a streamed expert costs many times what a
resident one does. The N-tile padding fix is in
[iova-and-multicore.md](iova-and-multicore.md).

## Streamed weights

[CBUF weight streaming](../GLOSSARY.md) keeps the input tile in the CBUF while the weight groups
cycle through the weight banks. One task therefore covers a whole N slice
([../matmul-as-conv.md](../matmul-as-conv.md) §"CBUF weight streaming"). The whole-tile form
shares the CBUF between both tiles, which caps Nt and multiplies the task count. The streamed
form's gain at a shape follows the task count that the whole-tile form needs there [HW sweep
2026-10-21].

**Conditions:** RK3588 (Turing RK1), kernel 7.2.9, `rocket` 1.3.0, NPU clock 928 MHz (the
900 MHz request), CPU governor `performance`, fp16. One core, one job a configuration, tasks
submitted gapped. Median submit-to-fence time. Measured 2026-10-21.

| Shape | Streamed against the best whole-tile tiling with DATA_REUSE | Speedup |
|---|---|---:|
| `160×1024×4096` | 1.75 ms (766 GFLOP/s) against 6.01 ms (223 GFLOP/s) | 3.4x |
| `128×2560×4096` | Streamed at 32-row M tiles | 1.30x |
| `128×512×4096` | The whole-tile form already runs 128×256 tiles at K 512, so few tasks are removed | 0.90x |

An external userspace reports fp16 device time of ~2.6 TFLOPS over three cores at K 1024 and
N 4096 [source-confirmed: `rknpu2_matmul_open`, [../SOURCES.md](../SOURCES.md)]. It runs on the
vendor `rknpu` driver at 1 GHz. The reported rate is ~870 GFLOP/s a core [host-computed]. The
streamed device rate over three cores on `rocket` is not characterized.

### Weight bandwidth a core

At M 4, one core streams resident weights at ~5.4 GB/s, fp16 and int8 alike [HW sweep
2026-10-21]. The rates are 5.49 GB/s at fp16 and 5.38 GB/s at int8, and 5.39 GB/s at int8 with
K 8192 and N 3072. Three cores, each driven by its own process on its own fd, keep their solo
rate: ~16.4 GB/s at fp16 and ~15.4 GB/s at int8 in aggregate. What this bandwidth means for
decode is in [decode-gemv.md](decode-gemv.md).

**Conditions:** as above, at M 4, fp16 and int8. Each rate is the weight bytes over the median
submit-to-fence time, one submit and fence a run.

> [!NOTE]
> Several jobs in one submit run one after another on `rocket` 1.3.0: three jobs take 2.96x
> the time of one [HW sweep 2026-10-21]. Software reaches the three cores through separate fds.

### Through the resident matmul

The streamed plan is faster than the whole-tile plan at every LLM projection shape measured,
for fp16 and for int8 [HW sweep 2026-10-21].

**Conditions:** RK3588, kernel 7.2.9, `rocket` 1.3.0, NPU clock 928 MHz, CPU governor
`performance`. M 512, resident weights split over five worker fds on the three cores,
fp16 K-accumulation on. The streamed plans are those of a planner that counts DRAM bytes,
before the K-tile choice below. Each speedup is the whole-tile call's time over the streamed
call's, per pass, over three passes with the order rotated. Measured 2026-10-21.

| Shapes, K × N | fp16 | int8 |
|---|---:|---:|
| K 1024-4096, N 1024-15360: attention and FFN up projections | 1.10-1.51x | 0.99-2.43x |
| K 8192-15360, N 3072-4096: FFN down projections | 1.22-1.39x | 1.72-2.09x |
| K 128-512, N 2048-4096 | 1.01-1.17x | Not measured |

The int8 gain follows the K passes that the streamed plan removes, because the host reads and
sums every pass's int32 partial. A single K pass at K 1024 gives 2.20-2.43x. A plan that keeps
Kt at 1024 against a slice of N 2464-3072 gives 0.99-1.06x.

Within the streamed plans, the K-tile and the M-tile it leaves set the time [HW sweep
2026-10-21]:

- **fp16.** Kt 1024 at Mt 160 is the fastest of 4-5 candidates at all six shapes swept. Kt
  608-640 at Mt 256 costs 1.09-1.14x more, because it adds K passes. An M tile under about 128
  rows cannot hide the weight stream it re-reads, and Kt 1536-3072 at Mt 32-96 costs
  1.11-2.99x more.
- **int8.** The deepest Kt that keeps Mt at 72 rows or more is fastest, Kt 1376-3072. Each K
  pass is an int32 partial that the host reads and sums, and Kt 1024 at Mt 256 costs
  1.31-2.03x more.

One candidate departs from both trends. At K 3840, fp16 Kt 960 at Mt 168 costs 1.25-1.6x more
than Kt 640 at Mt 256 [HW sweep 2026-10-21]. It takes fewer K passes. Its cause is not
characterized.

### In a model

`librocketnpu` plans fp16 matmuls with streamed weights by default. Its planner trades Kt
against Mt, because each M tile reads the whole weight slice again.

**Conditions:** RK3588 (Turing RK1), kernel 7.2.9, `rocket` 1.3.0, NPU clock 928 MHz, CPU
governor `performance`. The runs use llama.cpp through `ggml-rocket` on three cores, with
`llama-bench -n 0 -r 3` at the default `-b` and `-ub`, pp512 unless the row states another.
Each range is over three passes with the configuration order rotated. Measured 2026-10-21.

| Model | Weights | Whole-tile t/s | Streamed t/s | Speedup |
|---|---|---:|---:|---:|
| Llama-3.2-3B-Instruct F16 | `ROCKET_F16_RESIDENT=auto` | 87.8-89.3 | 100.4-101.3 | 1.134-1.143x |
| Llama-3.2-3B-Instruct F16, pp2048 | `ROCKET_F16_RESIDENT=auto` | 79.2-80.0 | 91.2-91.7 | 1.140-1.157x |
| Llama-3.2-3B Q4_K_M | `ROCKET_QUANT_RESIDENT=auto` | 81.6-82.9 | 95.3-96.6 | 1.163-1.167x |
| Ministral-3-3B F16 | Resident F16 | 82.0-83.7 | 93.7-94.9 | 1.120-1.153x |
| Qwen3.5-9B Q4_K_M | `ROCKET_QUANT_RESIDENT=auto` | 34.9-35.7 | 38.2-38.7 | 1.069-1.110x |

Llama-3.2-3B F16's wikitext perplexity (8 chunks of 512) is 10.9135 whole-tile and 10.9120
streamed [HW sweep 2026-10-21]. The flash attention gains 1.04-1.06x at Depth Anything's layer
and 1.12-1.19x at Gemma-4's shape (T 512, 2048 keys, head dimension 256). The one-shot
single-fd call at `512×3840×4096` goes from 150-153 to 163-167 GFLOP/s, 1.08-1.10x.

### The planner's K-tile cost

The table above uses a planner that counts DRAM bytes. The current planner charges a streamed
task the larger of two costs: its weight bytes, or the MACs of 128 rows per weight byte. An M tile
of 128 rows or more therefore hides its weight re-read. With K accumulated on the NPU, a K pass
costs its partial, written and read back once. When the host reads and sums every pass, a pass
costs 8x its partial's bytes.

No fixed charge applies per task or per pass. The 128-row crossover and the 8x factor are fitted
to the K-tile sweep above.

At the resident call at M 512, the current planner is 1.061x faster at fp16 than the byte-count
planner. At int8 with int8 streaming on it is 1.228x faster. Both are geometric means, over 24
and 20 projection shapes [HW sweep 2026-10-21]. One shape is slower: K 3584 N 1024 at 0.87x. The
planner charges that slice's 76-row last M tile as a whole exposed weight read. In a model it is
faster on the 3B units and neutral on Qwen3.5-0.8B, whose FFN down projection is that shape [HW
sweep 2026-10-22]:

| Model | Weights | Byte-count planner t/s | Current planner t/s | Speedup |
|---|---|---:|---:|---:|
| Llama-3.2-3B-Instruct F16 | `ROCKET_F16_RESIDENT=auto` | 98.7-102.3 | 107.4-108.5 | 1.060-1.089x |
| Llama-3.2-3B Q4_K_M | `ROCKET_QUANT_RESIDENT=auto` | 93.6-97.6 | 100.3-101.0 | 1.027-1.080x |
| Qwen3.5-0.8B F16 | `ROCKET_F16_RESIDENT=auto` | 132.0-149.2 | 133.0-147.0 | 0.891-1.102x, geometric mean 0.998x |

**Conditions:** as the table above, pp512. `ggml-rocket` built twice from one `librocketnpu`
tree, the two builds differing only in the planner's cost. Each speedup is a ratio within one
pass, the system memory reset before every process. Three passes with the order rotated, eight
for Qwen3.5-0.8B. Measured 2026-10-22.

> [!NOTE]
> Qwen3.5-0.8B F16 runs at either ~133 or ~147 t/s in a process, under both planners. On that
> model one pass resolves no difference smaller than ~10%.

<a name="scope"></a>
## Applicability

The result holds while DMA and dispatch bind. The tie between datatypes is reproducible at the
conditions above. Whether quantization raises prefill throughput once that limit is lowered is
not characterized.

### Conditions that would change the result

The remaining prefill optimization is fewer, larger NPU jobs, which lowers the per-job dispatch
limit for every datatype at once. The matmul runs at ~15% of the fp16 MAC peak, so a long
datatype-independent range remains before the MAC array binds. Only after DMA and dispatch stop
binding, and the NPU approaches its MAC limit, would int8's 2x and int4's 4x MAC rates raise
throughput [expected].

The result can also stay as it is. In the whole-tile form the CBUF's 384 KB caps the MAC work
per task, and weight streaming lifts that cap for the weights (§"Streamed weights"). If jobs
cannot be made large enough for MAC time to exceed DMA and dispatch latency, quantization never
raises throughput [expected].

int4 reads ¼ the weight bytes, *and* has 4x the MAC rate, and it still does not win in the
whole-tile form. That form's limit is therefore not weight-DMA
bandwidth either. It behaves like latency (dispatch, fence, CBUF fill), which large tiles
amortize and cannot remove [hypothesis]. The streamed form's gain follows the task count it
removes, which agrees with that reading.

### Host-side part of the limit

Part of the limit is host overhead: cache-sync on an over-allocated output BO that is synced
repeatedly. The KACC path issues one job *per K-tile* and syncs the output BO each time. With a
`BATCH`-sized BO and only `nMt·nNt` tiles live, `PREP_BO`/`FINI_BO` cache-sync ~8x too much,
`nKt` times (sync cost is ∝ BO size, see [bo-sync-cost.md](bo-sync-cost.md)). Right-sizing the
BO cuts `sync` 127->15 ms and raises resident fp16 throughput ~+11% (up to +17% best case),
bit-exact. Its effect has two limits:

- The right-sized BO does not change the order of the datatypes. It lowers the *common* limit,
  and quantization still does not pull ahead.
- It has no measurable effect on Gemma F16 prefill in a model. That model's K>2048 shapes
  *stream* (re-pack B on every call), so the weight BO dominates their `sync`, not the output
  BO. The gain is at the resident operating point.

The remaining `wait` term, the CPU blocked on the fence, is the NPU-bound part. Fewer, larger
jobs for that term are not characterized, because the KACC K-dependency forces `nKt` sequential
fences.

## Datatype-independent prefill optimizations

The limit is DMA, dispatch and the clock, so the prefill optimizations are
datatype-independent. They compound:

1. **The clock.** 198 -> ~816 MHz gives 1.43x. For 928 MHz (the 900 MHz request) and
   1019 MHz, see [clock.md](clock.md) §"Rates from ~816 to 1019 MHz".
2. **Fewer, larger NPU jobs.** Fusion and a larger micro-batch, and batching independent tasks
   into one hardware kick instead of one kick and completion IRQ per task. The vendor stack
   uses fewer submits than the open one in this way (see
   [iova-and-multicore.md](iova-and-multicore.md) §batched submit).
3. **Right-sized cache-synced BOs.** ~+11% on *resident* fp16, by cutting host
   `PREP_BO`/`FINI_BO` maintenance on the KACC output BO (see [bo-sync-cost.md](bo-sync-cost.md)).
   It has no effect on streaming prefill in a model.
4. **CBUF DATA_REUSE.** +7%, by removing a DMA (see
   [../encodings/cbuf-reuse.md](../encodings/cbuf-reuse.md)).
5. **fp16 on-NPU K-accumulation.** +19%, by removing host readback, fp16 only.

All four datatypes, int4, int8, int16 and fp16, are complete and correct, and the lower
precisions do not raise throughput.

### Per-submit overhead reductions

Two optimizations, IRQ affinity and IOMMU keep-attached, reduce the per-submit overhead rather
than the per-tile compute. **Both are ~flat on the large tiled prefill matmul.** Prefill is a few
large submits, where the per-submit overhead is negligible against ms-scale tile compute.

They raise throughput on paths with many small submits: decode GEMV, multi-fd contention, and
the detection throughput pool *under contention*. They do not raise prefill throughput, or
single-stream detection, which is bound by the host gather (below). The two optimizations, and
the CPU governor that the per-submit overhead depends on, are:

- **IRQ affinity.** The default IRQ mask services the NPU completion IRQ on an A55 little core.
  Binding the 3 NPU IRQs to an A76 and placing the waiter on the same core about halves the
  per-submit overhead ([iova-and-multicore.md](iova-and-multicore.md) §"IRQ affinity"). It is a
  runtime system setting, with no code change.
- **IOMMU keep-attached.** Stock `rocket` re-attaches the IOMMU domain on every job. Keeping the
  per-context domain attached across jobs of one fd removes ~15-20 µs (~38%) per submit. It is a
  driver patch.
- **CPU governor and frequency.** The per-submit overhead is CPU-side work (the submit `ioctl`,
  the blocking wait on the completion IRQ), so it scales with the CPU clock, not the NPU clock.
  On an idle board an `ondemand`/`interactive` governor lowers the cores' clock between submits,
  which inflates and varies any submit-bound measurement.

  An external RKNN-path writeup sizes the effect. One YOLOv8s `rknn_run` went from 59 to 35 ms
  (−41%) on the CPU governor alone. The governor changed from `ondemand` (cores at 408 MHz, idle
  board) to `performance` (1.8 GHz), with the NPU clock at 1 GHz throughout. Pinning the NPU governor
  *alone* changed nothing, because the NPU's `ondemand` already reaches its maximum under load.
  Locking both CPU and NPU to `performance` also cut the run-to-run variation from ~6 ms to
  0.64 ms. The on-demand frequency ramp was the source of the variation
  [external, proprietary path, corroborates the CPU-side submit overhead].

  On the `rocket` path the SigLIP-B/16 encoder behaves the same way [HW sweep]. It has many
  small attention submits, and a host softmax and de-tile per layer. Its resident warm median
  goes from 5.44 s under `schedutil` to 2.71 s under `performance` (−50%) [HW sweep]. Its
  variation goes from ±1.5 s to ±0.02 s, with the NPU at ~816 MHz throughout. The readback-bound matmuls are the most
  governor-sensitive, because readback is host work. For the same reason a NEON KACC de-tile
  gather (`detile_store_f16`, shared by the prepacked, stream and multicore paths) saved a
  further ~0.35 s ([encodings/siglip-encoder.md](../encodings/siglip-encoder.md),
  `rocket-userspace/tools/npu_perf_governor.sh`).

> [!NOTE]
> Pin the A76 cores to `performance` before timing any per-submit overhead
> (`submit_overhead_rocket`, decode GEMV, the detection convolutions). The governor's frequency
> ramp otherwise changes the µs-scale result, as it did for the external `rknn_run` above. The
> CPU governor is separate from the NPU clock's rise from its 200 MHz idle rate, for which the
> first measurement is discarded ([clock.md](clock.md)).

IRQ affinity and IOMMU keep-attached both measure flat on `matmul_tiled_rocket 512 3840 4096`
(320 tiles in 5 jobs of 64, at the symmetric tiling of 2026-06-22) and large on
`submit_overhead_rocket` (tiny 1-task jobs). See
[iova-and-multicore.md](iova-and-multicore.md) §IRQ affinity / §per-job IOMMU cost.

#### Detection single-stream

Single-stream detection behaves the same way [HW sweep 2026-06-29]. Coalescing a native-int8 or
uint8 convolution's per-tile submits into one job (`ROCKET_CONV_BATCH`, gapped) is flat on warm
MobileDet (~250 ms, 227 -> 215 submits), and flat across 4 parallel MobileDet processes. A tiled
convolution's wall-clock time is the host cube scatter and de-tile, not the per-submit
overhead, and most native-u8 convolutions are single-tile. The 227 submits are the matmul's
multicore worker fan-out, not convolution tiling. Coalescing submits helps only on a
convolution-tile-heavy model under multi-process contention (+7.6% aggregate at P=4, where the
contended submit and IOMMU path is the shared limit).

The single-stream detection optimization is therefore a smaller host gather, as for prefill
readback, not batched submits. NEON code for the requant epilogue and for the cube scatter
provides it. The M-major 1×1 requant vectorizes 8 channels per step, with OC contiguous on both
the int32 read and the NHWC write. It is bit-exact and gains +5.5% on MobileDet and +9.3% on
EfficientDet-Lite0. The detection profile matches the prefill one below. The host pack and
de-tile dominate the matmul and convolution time, and the per-submit overhead is a small
fraction.

Host-path changes and moving the requant on chip take warm MobileDet from 205 to 56 ms
[HW sweep, RK1, ~816 MHz, governor pinned, 2026-09-27 and 2026-09-28]. The submit path is
unchanged ([benchmarks.md](benchmarks.md) §"Single-stream latency").

## CPU-side profile

The wall-clock breakdown depends on the operating point. At ~816 MHz with on-NPU K-accumulation
and DATA_REUSE, it is `wait` ~60-68%, packB ~22%, and a small read. At 198 MHz with host
K-accumulation and no reuse, it is roughly 67% layout pack, scatter and readback, and only ~25%
NPU FLOPs. The CPU scatter, the readback and the per-call setup are therefore where
optimizations gain, and those gains compound at every model size and precision.

### Memory-bound and gather-bound CPU work

Memory traffic and index arithmetic, not instruction throughput, bound the fp16 matmul's
remaining CPU work, its `packB` and `read` loops:

- Compiler flags have no effect. The builds `-O3 -mcpu=native -DNDEBUG`, `+-flto`, and
  `+-fno-math-errno -fno-trapping-math` move the matmul within ±3% of the `-O2` baseline (noise,
  with LTO marginally worse). The `ROCKET_MM_PROFILE` readout shows `packB` (124 ms) and `read`
  (214 ms) byte-identical across builds on `512×15360×3840`. Those two loops scatter and gather
  over DRAM and are not ALU-bound. The branchy *llama.cpp decode and sampling* path is a separate
  case, where flags or PGO could still help, and it is not characterized.
- The readback de-tile vectorizes ~3x with NEON. The output-cube de-tile, the largest CPU
  component, reads 8 contiguous fp16 per column group for a fixed row. Those go into 8
  contiguous fp32 accumulators: one `vld1q_f16` -> 2x `vcvt_f32_f16` -> 2x `vaddq_f32`. The NEON
  form cuts `read` 214 -> ~76 ms (−64%) and the single-fd matmul wall-clock time 739 -> 604 ms
  (−18%). Multicore gains ~+7%, because the fan-out already overlaps readback across the 3
  cores. The result is bit-exact.

The scalar de-tile is therefore bound by its index arithmetic (restructuring the gather helps),
not by instruction issue (flags do not help), which matches the datatype-independent DMA and
dispatch limit above.

### Aliasing at `-O2`

The flag sweep covers the fp16 `packB` and `read` loops, and no other host loop. GCC 16 at `-O2`
does not version a loop for aliasing. A loop between two pointers it cannot prove distinct
compiles to a byte or scalar loop, and a block-scope `restrict` does not settle it. In
`librocketnpu`, `MM_NO_OVERLAP` (GCC `ivdep`, clang `assume_safety`) marks such a loop, and the
mark vectorizes it [HW sweep 2026-10-09, RK1, ~816 MHz, governor `ondemand`]:

| Loop | Measured with the mark |
|---|---|
| The int8, bf16 and tf32 weight scatters' 32-byte group copy | gpt-oss's expert-ingest pack 0.69-0.71x, 0.97-0.98x of it a row-order change |
| The int8, bf16 and tf32 activation packs' group copy | `packA` 83 -> 53-56 ms in `moe_gw_bench` at gpt-oss's expert |
| The SigLIP-family encoder's bias add | 93-95 -> 46-48 ms an image on Depth Anything v2-small |
| The same encoder's residual add | 19.9-20.5 ms an image unmarked, against 9.7-11.2 vectorized |

> [!NOTE]
> A flat compiler-flag sweep does not show that a host loop is memory-bound. Read the loop's
> disassembly first. Not every host pass is memory-bound. The flash attention's fused softmax
> ran compute-bound at about 1.1 IPC, and four independent exp chains ran it at 0.64x
> ([attention-offload-crossover.md](attention-offload-crossover.md) §"The softmax's own
> instruction rate").

Moving the de-tile to a DMA engine does not help, because the de-tile is not bandwidth-bound.
The RGA 2D blitter's throughput (~5.8 GB/s, best case) is below a single CPU core's `memcpy`
(~13 GB/s), and the NEON path already runs on 3 cores. The NEON de-tile is the supported path
([rga-detile.md](rga-detile.md), and by the same argument the PL330 DMAC).

## Bytes-moved model

The NPU's own DMA byte counters are unusable on the RK3588. Reading the `0x2xxx` page hard-locks
the SoC, and `0x80xx` is configuration only (see [hw-byte-counters.md](hw-byte-counters.md)).
RKNN's "Total Memory R/W per frame" is itself *computed from the graph*, not read from hardware.
A host-side model therefore computes the DRAM bytes each phase moves, analytically. Its inputs
are the shape, the real tiling from the datatype's own planner (`rocket_matmul_plan` for fp16,
`rocket_matmul_plan_int8` for int8, and so on: pure, no hardware), the datatype and the reuse
mode. It maps each term to a `ROCKET_MM_PROFILE` bucket:

| phase | bucket | formula |
|---|---|---|
| packB (host weight scatter) | `pack` | `N·K·ein` |
| packA (host input scatter) | `pack` | `M·K·ein` |
| weight DMA (DRAM->CBUF) | `wait` | `nMt·N·K·ein` (no weight reuse) |
| feature DMA (DRAM->CBUF) | `wait` | `M·K·ein` (data_reuse) \| `nNt·M·K·ein` (no reuse) |
| output WDMA (CBUF->DRAM) | `wait` | `M·N·eout` (KACC) \| `nKt·M·N·eout` (no KACC) |
| readback (host de-tile) | `read` | `M·N·eout` (KACC) \| `nKt·M·N·eout` (no KACC) |

The model is validated against a warm `512×3840×4096` fp16 profile on one fd, with the weight
packed on every call and KACC and DATA_REUSE on. It measures
`pack=36 packA=4 packB=32, wait=84, read=10` ms [HW sweep, RK1, 2026-06-16 to 2026-06-28].
That build tiled the shape symmetrically (Kt 384, Nt 256, §"Revision history") and packed with
scalar loops. The pack and readback byte counts below do not depend on the tiling, and the
times are that build's:

- The pack byte model is exact. The model gives `packB:packA = 30 MB : 3.75 MB = 8:1`, and the
  measurement gives `32 ms : 4 ms = 8:1`.
- Pack and read are not bandwidth-bound. The achieved pack rate is
  ~`33.75 MB / 36 ms ~0.9 GB/s`, and readback is ~`4 MB / 10 ms ~0.4 GB/s`, both ~5 % of LPDDR
  streaming (~17 GB/s). The host scatter and gather are bound by latency and index arithmetic
  (random-stride writes and the fp16↔fp32 convert). Reducing pack and read *bytes*, for example
  by quantization, gives far less than changing their *gather pattern* (the NEON de-tile).
- The model sizes DATA_REUSE. The planner's whole-tile tiling for this shape
  (`ROCKET_MM_STREAM=0`) is Mt 256, Kt 512, Nt 128 (`nMt`=2, `nKt`=8, `nNt`=32). There, turning
  data_reuse off changes only `fDMA`, 3.75 -> 120 MB (`×nNt=32`), and moves the total
  133.50 -> 249.75 MB [host-computed]. The
  `ROCKET_REUSE` CBUF-reuse gain removes that DMA ([cbuf-reuse.md](../encodings/cbuf-reuse.md)).

The DDR controller's PMU shows the model right on DATA_REUSE and 3.57x low in absolute terms
(§"Evidence"). The host scatter moves a whole cache line per sub-line chunk, twice. On the bus
the scatter runs at ~3.4 GB/s, against a ~17 GB/s ceiling, so pack is still not
bandwidth-bound.

### The int8 readback limit

The model makes the int8 readback cost concrete. Take the same shape in int8, with no on-NPU
K-accumulation and int32 out. The int8 planner tiles it at Kt 640 (`nKt`=6), so `oWDMA` +
`readback` = `nKt·M·N·4` *each* = 48 MB + 48 MB = 96 MB of output traffic. The fp16-KACC path
moves `4 MB + 4 MB`. The int8 total is 144.75 MB, against fp16's 133.50 MB, although the int8
weights are half the bytes [host-computed]. The un-K-accumulated int32 readback (`∝ M·N·nKt`) is
why int8 prefill is slower (§"In-model int8 and int4 prefill").

The DDR PMU shows the int8 figure as a lower bound (§"Evidence").

The model shows this for any shape before anything runs. It is pure, and runs anywhere.

## Evidence

### Applicability

Two earlier readings on this hardware reversed once the bottleneck moved. At the 200 MHz boot
clock, CPU-side optimizations measured as having no effect, and they gained once the clock was
raised. The fp16 EW K-accumulation measured as unworkable until a configuration fix. The
bottleneck-conditional scope above rests on that pattern and on the CBUF argument.

The right-sized BO's +11% is a drift-controlled comparison, and +17% is its best case.

### Streamed weights

The resident-matmul table comes from `tests/matmul_dtype_perf_rocket`, one process a
configuration with the memory reset before each. A process makes eight calls, drops the first
and keeps the fastest of the rest. The shapes are five models' projections (Llama-3.2-3B,
Ministral-3-3B, Qwen3.5-9B, Gemma-4-12B and Qwen3.5-0.8B) and four short-K shapes. It counts 24 fp16 shapes and 20 int8 shapes. int8
streams with one weight bank beyond the fp16 minimum (Erratum E11). The K-tile sweep forced each candidate
with `ROCKET_MM_STREAM_KT` at six shapes a datatype, 52 configurations, three passes.

The single-core comparison is `tests/weight_stream_probe`, 15 interleaved runs a configuration
with the order rotated. The whole-tile configuration is the best tiling found with DATA_REUSE
set, which moves the same DRAM bytes as the streamed one. The bandwidth figures are
`tests/gemv_bw_probe` at M 4, and include one submit and fence a run. The three-core figure ran
three processes at once, and all three cores raised interrupts. One submit of three jobs is
`multicore_probe`.

The model figures are `llama-bench` with memory reset before every process. Every run was
bit-exact against its reference. The RK3588 CTest suite (126 tests) passes under both settings,
with three model tests skipped for lack of model paths. The flash-attention figures
are 30 iterations, five runs a configuration. The one-shot call's gain includes the output BO
that the K-accumulating path no longer allocates up front, which alone made the whole-tile
call ~4% faster.

### Vendor-stack int8 prefill

An isolated fixed-clock fp16-against-W8A8 prefill sweep on the vendor stack, with a DDR or NOC
PMU read, would test the bandwidth account directly. It has not been run.

### Bytes-moved model

The DDR PMU check [HW sweep 2026-08-28, RK1, ~816 MHz] uses `rockchip_ddr`, calibrated to 0.4%
against known bytes ([hw-byte-counters.md](hw-byte-counters.md) §5). Differencing 45 against 5
repetitions of the same `512×3840×4096` test point isolates 40 matmuls:

- **The `data_reuse` term is right.** The model predicts that `ROCKET_REUSE=0` adds `+116.25 MB`
  of reads and nothing to writes. The counter measures +120.98 MiB read and -1.93 MiB write, 4%
  high. A counter therefore confirms the tiling and the loop order the model assumes.
- **The absolute total is 3.57x the model**: 475.95 MiB measured against 133.50 MB. The excess is
  the host scatter. Hoisting `packB` out of the loop (the prepacked path, same arithmetic)
  removes 249.34 MiB per call against a 30.00 MB term: 8.3x, in both directions. A scatter moves
  useful chunks smaller than a cache line, so the bus moves a whole line per chunk and moves it
  twice.
- **The `KACC` sensitivity is wrong on the read side.** The model predicts that `ROCKET_KACC=0`
  adds `+28 MB` of writes and no reads. The counter measures +20.4 MiB write and +189.0 MiB read.
  The int8 readback figure in the reading path is therefore a lower bound.

The 3.57x excess does not change the conclusion that pack is not bandwidth-bound. The 249 MiB
over the ~78 ms of wall-clock time the scatter adds per call is ~3.4 GB/s against a ~17 GB/s
ceiling. The margin is 8x smaller than the 0.9 GB/s figure in the model's validation implies.
The 78 ms is a three-repetition timing with the first repetition included, so it is the less
precise of the two figures.

The model's test is the CTest `bytes_moved_rocket`, which cross-checks its tile count against
the planner's `njobs`.

The validation profile has no date of its own. It ran with DATA_REUSE, first validated on
2026-06-16, and these notes first recorded it on 2026-06-28. It therefore predates the
asymmetric default of 2026-07-01, and the `librocketnpu` build of 2026-07-15 that vectorized
the packA and packB scatters with NEON. Its NPU clock was not recorded with it.

### Revision history

The model's figures follow the planner's tiling. The asymmetric-Nt plan became the planner's
default on 2026-07-01 ([asymmetric-tile.md](asymmetric-tile.md)). Before it, the symmetric plan
tiled `512×3840×4096` at Mt 256, Kt 384, Nt 256 (`nMt`=2, `nKt`=10, `nNt`=16: 320 tiles). The
model then gave these figures. `ROCKET_MM_ASYM=0` reproduces the fp16 ones [host-computed]:

- The fp16 total, KACC and data_reuse on: 141.5 MB.
- With data_reuse off: `fDMA` 3.75 -> 60 MB (`×nNt=16`), and the total 141.5 -> 197.8 MB.
- The int8 `oWDMA` + `readback`: 80 MB + 80 MB = 160 MB, and the int8 total 208.8 MB. These
  tiled int8 with the fp16 planner's symmetric plan, which no current option reproduces.

Until 2026-10-12 the model tiled every datatype with the fp16 planner. At the asymmetric default
it gave the int8 `oWDMA` + `readback` as 64 MB + 64 MB = 128 MB, and the int8 total as 176.75 MB
[host-computed]. The int8 planner's own tiling gives the figures in §"The int8 readback limit".

The DDR PMU measurement of 2026-08-28 ran at the asymmetric default, and its model column uses
that tiling.
