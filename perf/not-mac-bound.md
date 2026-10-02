# The matmul is not MAC-bound

This is the most important, and most counterintuitive, performance finding from the whole
project. It is a negative result, and it is load-bearing for anyone who assumes that
int8/int4 is faster.

> **Read [Scope](#scope) first.** This is a measurement *at the current operating point*
> (resident, multicore, 600 MHz), where the binding constraint is a DMA/dispatch floor. It
> is not a proven permanent property of the silicon. Quantization's MAC advantage is gated
> *behind* the datatype-independent dispatch floor, so it is a later-stage lever rather than
> a dead one. This project has already seen "X doesn't help" flip once the underlying
> bottleneck moved. The 200 MHz clock made "CPU-side levers are dead" true, and raising the
> clock revived them.

## Throughput across precisions

On the FOSS `rocket` path at 600 MHz, resident multicore matmul throughput is ~460 GOP/s
across precisions. The table is measured on the same `512×3840×4096` shape, with all three
datatypes resident across 5 worker fds:

| datatype | GOP/s | MAC advantage | expressed? |
|---|---:|---|---|
| fp16 (+K-accumulation +DATA_REUSE) | 461 | 1x | n/a |
| int8 | 386 | 2x | no |
| int4 | 413 | 4x | no |

Other shapes land in the same band (GOP/s):

| shape | fp16 | int8 | int4 |
|---|---:|---:|---:|
| small-tile | 453 | 478 | 439 (int4 single-pass K) |
| ffn-down | 486 | 452 | 463 |

Everything ties within ~20%. The 2x MAC of int8 and the 4x MAC of int4 do not express as
speed.

That ~460 GOP/s is ~15% of the fp16 MAC peak and ~4% of the int4 peak. The MAC array is
mostly idle. The NPU is bound by a datatype-independent floor: the DMA to load operands into
CBUF plus per-job dispatch latency. It is not bound by compute.

## MAC and readback checks

Two independent measurements confirm that reading.

The matmul is not MAC-bound. If it were, the 2x/4x quant advantages would show up, and they
do not.

K-tile readback does not dominate the common floor. The int4 single-pass-K run (`nKt=1`,
*zero* K-tile readback) is no faster (439 ~ the rest). If readback set the floor,
eliminating it would win, and it does not. That result let this project descope the int4 EW
K-accumulation rung before building it (see
[../encodings/k-accumulation.md](../encodings/k-accumulation.md)).

The precise claim is narrower than "readback never matters". Readback does not *set the
floor*, but it still decides datatype ordering around the floor. Readback can only push a
datatype *below* the floor: the un-K-accumulated int32 readback of int8, ∝ `M·N·nKt`, is
extra traffic (see §"In-model int8 and int4 prefill"). Readback cannot lift a datatype that
is already at the floor (fp16-KACC, int4 single-pass) above it. "Not readback-bound" means
that removing readback from a floor-sitting datatype does not raise the floor.

What remains is the DMA/dispatch floor: loading tiles into the CBUF and the per-NPU-job
launch/fence overhead. Both are datatype-independent.

## In-model int8 and int4 prefill

In the live model int8 prefill is slower than fp16. Resident int8 prefill is ~9.1 t/s vs
resident fp16 ~15.1 t/s = 0.60x [HW sweep, 600 MHz].

The fp16 path has on-NPU K-accumulation (`read ∝ M·N`), and int8 does not. No integer EW path
ships, and fp32 cannot hold its int32 partials exactly (see
[../encodings/k-accumulation.md](../encodings/k-accumulation.md)). So the un-K-accumulated
int32 readback of int8 (`∝ M·N·nKt`, 2x the bytes of fp16) costs more than its MAC advantage
gains. Making int8 *resident* removes the per-call requant + packB overhead (+25-58% over the
naive int8 path). The readback wall is untouched, and it caps int8 below fp16.

A same-weights precision sweep confirms it in-model. It runs the Gemma-4-12B F16 GGUF through
the *one-shot* paths (re-quantize each prefill) at pp512, under the `performance` governor
[HW sweep, 2026-06-24, RK1 7.1.0-1, 600 MHz]:

| path | t/s | vs fp16 |
|---|---:|---:|
| fp16 + KACC | 14.4 | 1.00x |
| int8 + Hadamard | 6.9 | 0.48x |
| int4 + Hadamard | 4.4 | 0.30x |
| CPU (8 threads) | 4.6 | 0.32x |

Lower precision lowers prefill monotonically: the one-shot quant + int-readback cost adds to
the datatype-independent floor. These are the per-call-requant paths. The *resident* int8
above amortizes the requant and lands higher (0.60x), still below fp16.

A quantized `Q8_0` GGUF (on-the-fly dequant->fp16, half the RAM at 11.8 GB) ran 5.7 t/s. That
is slower than native F16, because of the per-call dequant. It is still above the CPU and at
half the footprint, which is the model-fit payoff quantization buys here. That per-call
dequant is per micro-batch and amortizes with `-ub`: raising the micro-batch is worth
0.91-1.53x on quantized prefill, by model. A routing floor (`ROCKET_MIN_M_QUANT`) keeps sub-crossover quant prefills on the CPU
(see [quant-prefill-microbatch.md](quant-prefill-microbatch.md)).

### Resident group-wise int4

Resident int4 (group-wise) repeats the resident-int8 result, more sharply. Holding the
group-wise + Hadamard int4 weights resident (`ROCKET_INT4_RESIDENT`) removes the per-call
weight scatter, exactly as resident int8 does. The in-model gain is ~2x over the one-shot int4
path: 6.94 t/s vs 3.56 (resident vs non-resident int4), fp16 13.18, pp512 @600 MHz
[HW sweep, 2026-06-25]. That is still 0.53x fp16. A group-wise int4 matmul reads back one
int16 partial *per K-group* (`read ∝ M·N·nKt`, `nKt = K/group ~120` on the deep K=15360 FFN).
So the same un-K-accumulated-readback wall that caps int8 caps int4 harder, because finer
groups mean more readback.

The payoff is footprint. The resident int4 weights are ¼ the NPU-BO bytes of fp16 (2634 MB vs
~10.5 GB for the offloaded set). So the resident path turns int4 from "RAM win at a steep
speed cost" into "RAM win at half fp16 speed". Quantization still buys footprint, not
throughput. The lever that would change that is the dispatch/readback floor, not the
datatype.

## The proprietary stack's int8 win is bandwidth, not a capability the open path lacks

The proprietary rknpu2 / rk-llama.cpp stack runs int8 LLM prefill faster than its *own*
fp16. The FOSS `rocket` path does not reproduce that ordering. The cause is not a hardware
feature that the open path cannot reach. Three mechanisms were proposed for it, and all
three are settled.

### On-device int32 K-accumulation

On-device int32 K-accumulation is real, and bounded the same way for everyone. The conv
accumulates K on-chip only *within one CBUF-resident K-tile*. No register field accumulates
the CACC across CBUF passes. The CORE block has no accumulate-vs-reset control, and the conv
task splitter splits spatial height, never the channel (K) axis
[source-confirmed: Mesa `rkt_task.c`].

Beyond one tile the partials are summed through DRAM, through the DPU-EW path that the
`ROCKET_KACC` mechanism uses for fp16 [HW sweep, source-confirmed]. That path's ALU adds int32
exactly under the whole precision bundle, and a K-accumulation on it is not measured. So no
stack is known to accumulate int32 across tiles on the device
(see [../encodings/k-accumulation.md](../encodings/k-accumulation.md)).

### On-chip SRAM

On-chip SRAM stages whole tensors, not partials. The 956 KB NPU SRAM holds *weight* or
*internal (activation)* tensors to relieve DDR bandwidth. The vendor's own doc notes that it can
"have a certain impact on inference time", and its per-layer example is a vision CNN. The SRAM
is not a partial-sum accumulator, and 956 KB cannot hold an LLM's weights or a prefill
activation matrix [source-confirmed: see [sram-nbuf.md](sram-nbuf.md)].

### Operand bandwidth

The win is operand bandwidth at a lower dispatch floor. RKLLM on the RK3588 quantizes to
W8A8: 8-bit weights *and* activations, the only LLM quant the chip's toolkit offers. So int8
halves the bytes moved for both operands. The vendor's own Gemma int8 prefill runs at 40-58%
NPU utilization. Their stack is *also* not MAC-bound, so int8 buys them bytes, not MACs. On
a path bound by dispatch/DMA, halving the operand bytes helps where the 2x MAC cannot.

[W8A8-only is the vendor LLM path, and the utilization is from rk-llama.cpp forum benchmarks,
a single external source.]

The remaining vendor edge is dispatch efficiency. It has three parts: a batched whole-graph
submit (~10x fewer kernel transitions, see [iova-and-multicore.md](iova-and-multicore.md)
§batched submit), 3-core dispatch (matched on the `rocket` path), and the 1 GHz default clock
(this project's clock patch reaches 600 MHz).

The "+200-400%" sometimes quoted for the vendor's int8 prefill is the 6-vs-3 TOPS spec-sheet
ratio (6 TOPS int8 / 3 TOPS fp16), not a measured fp16->int8 A/B result. The realized 40-58%
utilization is the better guide. That the vendor is also sub-60% utilized independently
corroborates the not-MAC-bound result above.

The causal account is bandwidth at a lower floor rather than a secret accumulator. It rests
on the capability facts (HW + source-confirmed) plus the vendor's utilization numbers. An
isolated fixed-clock fp16-vs-W8A8 prefill sweep on the vendor stack, with a DDR/NOC PMU read,
would convert it from well-supported to proven.

## What quantization buys

Quantization buys RAM, not speed. Its payoff on this hardware is:

- **Model size and fitting in memory.** Run a model that does not fit in fp16. The int4
  Gemma weights are ~¼ the bytes, and the int4 matmul is fully working and bit-exact.
- **IOVA residency.** A quantized model fits the per-fd IOVA windows whole (see
  [iova-and-multicore.md](iova-and-multicore.md)).
- **Decode coexistence.** A quantized GGUF serves CPU decode alongside NPU prefill.

At the current operating point it is not a prefill throughput win. The DMA/dispatch floor
hides its MAC advantage. State that plainly, and state it with its scope (below), not as a
permanent law.

### Quantized MoE experts and residency

A MoE routed expert is the exception: the one place where quantization does buy speed. It
buys residency, and residency buys speed. The arithmetic is not the one people reach for.

The host dequantizes a quantized expert on the streaming path to fp16 every micro-batch.
That decode is independent of `M`: it decodes the whole `[N,K]` weight whatever row count the
router gives that expert. On gpt-oss-20b's `[2880,2880]` MXFP4 experts it measures 75 ms per
expert, per micro-batch. A prefill touches ~1580 of them. So the host spends ~119 s per
prefill dequantizing weights, 59% of a pp2048 prefill, before any arithmetic happens. That is
why offloading quantized MoE experts through the fp16 route loses to leaving them on the CPU
[HW sweep, 600 MHz, 2026-07-14].

Ingesting the expert once into resident int8 codes removes that cost. The speed does not come
from int8 being int8. The int8 GEMM's int32 output reads back at 8 B/element and moves *more*
bytes than the fp16 one would. The speed comes from the weight not crossing the host every
micro-batch. Quantization here buys residency, and residency buys the speed.

**Partial residency is not linear.** Reading "85% of the experts are resident" as "85% of the
tax is gone, 15% remains" is wrong. The two routes have different *cost structures*.
A streamed expert keeps paying the full, M-independent 75 ms, while a resident one pays a
small GEMM that *shrinks with M*. So the non-resident remainder's share of the wall clock
grows as the prefill gets shorter:

| prefill | share of the prefill spent on the streamed 15% |
|---|---:|
| pp2048 | 12% |
| pp512 | 43% |

That is why the native-quant route wins at long prefill and loses at short. A residency
percentage is not a cost percentage. So the resident weight's *memory efficiency* is a
first-order throughput lever, not a footprint nicety. Every byte of tile padding is an expert
that has to stream instead, and a streamed expert costs many times what a resident one does.
That makes the N-tile padding fix in [iova-and-multicore.md](iova-and-multicore.md) a
performance change rather than a housekeeping one.

<a name="scope"></a>
## Scope

Treat "quantization doesn't speed up prefill" as **bottleneck-conditional, not a
hardware fact.** The measured tie is solid and reproducible. The *generalization* to
"quant can never help prefill here" is stronger than the evidence.

### Conditions that would change the result

The explicitly named remaining prefill lever is the per-job dispatch floor (fewer, bigger NPU
jobs). That lever is datatype-independent, so it lifts all precisions first. The matmul runs
at only ~15% of fp16 MAC peak, so there is a long datatype-independent ramp before MAC binds
at all. *Only after* dispatch/DMA stop binding and the NPU approaches its MAC ceiling would
int8's 2x and int4's 4x have room to express. So quantization is plausibly a later-stage
lever, gated behind the dispatch floor, not eliminated.

This project has precedent. "CPU-side levers are dead" and the fp16 EW K-accumulation were
both "dead" until the clock or a config fix moved the bottleneck. Then they worked.

### The structural counter-argument

The result can also stay flat. The CBUF is only 384 KB, which caps the MAC-work-per-job. If
jobs cannot be made big enough for MAC time to dominate DMA/dispatch latency, the floor is
structural and quant never gets its opening. One data point supports that reading: int4
reads ¼ the weight bytes *and* has 4x MAC, and it still does not win. So the floor is not
weight-DMA bandwidth either. It is latency-like (dispatch / fence / CBUF-fill), which big
tiles amortize but cannot remove.

### Reducible part of the floor

The floor is not entirely irreducible. One slice of it is reducible: host-side cache-sync on
an over-allocated, repeatedly-synced output BO. The KACC path issues one job *per K-tile* and
syncs the output BO each time. With a `BATCH`-sized BO and only `nMt·nNt` tiles live,
`PREP_BO`/`FINI_BO` cache-sync ~8x too much, `nKt` times (sync cost is ∝ BO size, see
[bo-sync-cost.md](bo-sync-cost.md)). Right-sizing the BO cuts `sync` 127->15 ms and lifts
resident fp16 ~+11% (drift-controlled A/B, up to +17% best-case), bit-exact.

Two caveats apply:

- The right-size does not reorder the datatype spread. Quant still does not pull ahead: the
  lever lowers the *common* floor and does not give MAC its opening.
- It is invisible in-model for Gemma F16 prefill. That model's K>2048 shapes *stream* (re-pack
  B every call), so the weight BO dominates their `sync`, not the output BO. So the win is at
  the resident operating point.

The residual `wait` term (CPU blocked on the fence) is the NPU-bound part. "Fewer/bigger jobs"
for that term is still open, because the KACC K-dependency forces `nKt` sequential fences. In
sum, part of the "dispatch floor" is reducible host overhead (~+11% resident fp16), and the
residual NPU-compute floor still ties the precisions. Quant's *known* payoff remains RAM,
model size and decode coexistence.

## Datatype-independent prefill levers

The floor is DMA + dispatch + clock, so the prefill levers are datatype-independent:

1. **The clock** (200 -> 600 MHz = 1.43x, with 900 MHz/1 GHz gated, see
   [clock.md](clock.md)).
2. **The per-job dispatch floor**: fewer, bigger NPU jobs (fusion, larger micro-batch), and
   batching independent tasks into one HW kick instead of one submit + completion IRQ per
   task. That is the open-vs-vendor submit-count gap (see
   [iova-and-multicore.md](iova-and-multicore.md) §batched submit).
3. **Right-sized cache-synced BOs** (~+11% *resident* fp16). This cuts host
   `PREP_BO`/`FINI_BO` maintenance on the repeatedly-synced KACC output BO (see
   [bo-sync-cost.md](bo-sync-cost.md)). It is invisible to streaming/in-model prefill.
4. **CBUF DATA_REUSE** (+7%, cuts a DMA, see
   [../encodings/cbuf-reuse.md](../encodings/cbuf-reuse.md)).
5. **fp16 on-NPU K-accumulation** (+19%, cuts host readback, fp16 only).

These compound and are precision-independent. The deliverable from the datatype work is the
completeness of the matrix (int4/int8/int16/fp16 all working and correct), not a speedup.

### Dispatch-floor reducers

Two levers, IRQ affinity and IOMMU keep-attached, cut the per-submit floor (not the per-tile
compute), so they belong here for completeness. **Both are ~flat on the big tiled prefill
matmul this note is about.** Prefill is a few large submits, where the per-submit overhead is
negligible against ms-scale tile compute.

They pay on many-small-submit paths (decode GEMV, multi-fd contention, the detection
throughput pool *under contention*). They do not pay on prefill, or on single-stream
detection, which is host-gather-bound (see below). Do not read them as prefill speedups. The
two levers, and the CPU governor that the same floor depends on, are:

- **IRQ affinity.** The default IRQ mask services the NPU completion IRQ on an A55 little
  core. Binding the 3 NPU IRQs to an A76 and co-locating the waiter cuts the submit floor
  51 -> ~27 µs (−47%). This is runtime/system config, with no code change.
- **IOMMU keep-attached.** Stock `rocket` re-attaches the IOMMU domain on every job. Keeping
  the per-context domain attached across same-fd jobs removes ~15-20 µs (~38%) per submit.
  This is a driver patch.
- **CPU governor and frequency.** The per-submit floor is CPU-side work (the submit `ioctl`,
  the blocking wait on the completion IRQ), so it scales with the CPU clock, not the NPU
  clock. On an idle box an `ondemand`/`interactive` governor parks the cores low between
  submits, which inflates and jitters any submit-bound measurement.

  An external RKNN-path writeup sizes the effect. One YOLOv8s `rknn_run` swung 59 -> 35 ms
  (−41%) purely on the CPU governor. The governor moved from `ondemand` (cores at 408 MHz,
  idle box) to `performance` (1.8 GHz), with the NPU clock untouched at 1 GHz throughout.
  Pinning the NPU governor *alone* changed nothing, because the NPU `ondemand` already ramps
  to its ceiling under load. Locking CPU and NPU to `performance` also collapsed the
  run-to-run jitter (~6 ms -> 0.64 ms). The on-demand ramp was the jitter source
  [external, proprietary path, corroborates the CPU-side submit floor].

  On the `rocket` path the SigLIP-B/16 encoder (many small attention submits + host
  softmax/de-tile per layer) shows the same shape [HW sweep]. Its resident warm median goes
  5.44 s `schedutil` -> 2.71 s `performance` (−50%) and its jitter ±1.5 s -> ±0.02 s, with the
  NPU held at 600 MHz throughout. The readback-bound matmuls are the most governor-sensitive,
  because readback is host work. For the same reason a NEON KACC de-tile gather
  (`detile_store_f16`, shared by the prepacked/stream/multicore path) saved a further ~0.35 s.
  Pin the CPU governor before any submit-bound bench
  ([encodings/siglip-encoder.md](../encodings/siglip-encoder.md),
  `rocket-userspace/tools/npu_perf_governor.sh`).

IRQ affinity and IOMMU keep-attached both measure flat on `matmul_tiled_rocket 512 3840 4096`
(320 tiles in 5 jobs of 64) and large on `submit_overhead_rocket` (tiny 1-task jobs). See
[iova-and-multicore.md](iova-and-multicore.md) §IRQ affinity / §per-job IOMMU cost.

#### Detection single-stream

Detection single-stream has the same shape [HW sweep 2026-06-29]. Coalescing a
native-int8/uint8 conv's per-tile submits into one job (`ROCKET_CONV_BATCH`, the gapped
lever-1) is flat on warm MobileDet (~250 ms, 227 -> 215 submits). It is also flat across 4
parallel MobileDet processes. A tiled conv's wall is the host cube scatter/descatter, not the
submit floor, and most native-u8 convs are single-tile anyway. The 227 submits are the matmul
multicore worker fan-out, not conv tiling. Submit-coalescing pays only on a conv-tile-heavy
unit under multi-process contention (+7.6% aggregate at P=4, the contended submit/IOMMU path
being the shared bottleneck).

So the detection single-stream lever is host-gather reduction, exactly as for prefill
readback, and not submit-batching. The levers are NEON for the requant epilogue and NEON for
the cube scatter. The M-major 1×1 requant vectorizes 8 channels/step, with OC contiguous on
both the int32 read and the NHWC write. It is bit-exact and gains +5.5% on MobileDet and +9.3%
on EfficientDet-Lite0. The detection profile mirrors the prefill one below: host pack/de-tile
dominates the mm + conv buckets, and the dispatch floor is a small fraction.

Host-path changes and moving the requant on chip take warm MobileDet from 205 to 56 ms. The
submit path is unchanged
[HW sweep, RK1, 600 MHz, governor pinned, 2026-09-27 and 2026-09-28]
([benchmarks.md](benchmarks.md) §"Single-stream latency").

#### Measurement hygiene

Before timing any submit/dispatch floor (`submit_overhead_rocket`, decode GEMV, the detection
convs), pin the A76 cores to `performance`. Otherwise the governor ramp confounds the µs-scale
number, the way it confounded the external `rknn_run` above. This is separate from the NPU
cold-clock throwaway in [clock.md](clock.md). One is the CPU governor between submits. The
other is the NPU clock ramping from its 200 MHz idle park.

## CPU-side profile

The wall-time breakdown depends on the operating point. At the unoptimized baseline (cold
clock, host K-accumulation, no reuse) it is roughly layout pack/scatter + readback ~ 67% and
NPU FLOPs only ~25%. That is why the optimization ladder targets CPU scatter, readback, and
per-call setup, and why those wins compound at every model size and precision. At the
current operating point (600 MHz, on-NPU K-accumulation, DATA_REUSE) it is `wait` ~60-68%,
packB ~22%, and a small read.

### Memory-bound and gather-bound CPU work

Two measurements show that memory traffic and index math bound the remaining CPU cost, not
instruction throughput:

- Compiler flags are flat. The builds `-O3 -mcpu=native -DNDEBUG`, `+-flto`, and
  `+-fno-math-errno -fno-trapping-math` move the matmul within ±3% of the `-O2` baseline
  (noise, with LTO marginally worse). The `ROCKET_MM_PROFILE` readout shows `packB` (124 ms)
  and `read` (214 ms) byte-identical across builds on `512×15360×3840`. The compiler has
  nothing to optimize: these loops scatter and gather over DRAM and are not ALU-bound. The
  CPU-bound, branchy *llama.cpp decode/sampling* path is a separate case where flags/PGO could
  still help, untested here.
- The readback de-tile NEON-vectorizes ~3x. The output-cube de-tile, the single largest CPU
  component, reads 8 contiguous fp16 per column-group for a fixed row. Those land in 8
  contiguous fp32 accumulators: one `vld1q_f16` -> 2x `vcvt_f32_f16` -> 2x `vaddq_f32`. The
  NEON form cuts `read` 214 -> ~76 ms (−64%) and the single-fd matmul wall 739 -> 604 ms
  (−18%). Multicore gains ~+7%, because the fan-out already overlaps readback across the 3
  cores. The result is bit-exact.

So the scalar de-tile is index-math-bound (restructuring the gather helps) and not
instruction-issue-bound (flags do not help). That is consistent with the
datatype-independent DMA/dispatch floor above.

Offloading the de-tile to a DMA engine does not help. The move is not bandwidth-bound. The
RGA 2D blitter's throughput ceiling (~5.8 GB/s, best case) is below a single CPU core's
`memcpy` (~13 GB/s), and the NEON path already fans across 3 cores. So the NEON de-tile stays
the path ([rga-detile.md](rga-detile.md), and by the same argument the PL330 DMAC).

## Analytical bytes-moved model

The NPU's own DMA byte counters are dead on the RK3588. Reading the `0x2xxx` page hard-locks
the SoC, and `0x80xx` is config-only (see [hw-byte-counters.md](hw-byte-counters.md)). RKNN's
"Total Memory R/W per frame" is itself *computed from the graph*, not read from HW. So
`tests/bytes_moved_rocket.c` computes the DRAM bytes each phase moves analytically. Its
inputs are the shape and the real tiling (`rocket_matmul_plan`, pure/no-HW), plus the
datatype and the reuse mode. It maps each term to a `ROCKET_MM_PROFILE` bucket:

| phase | bucket | formula |
|---|---|---|
| packB (host weight scatter) | `pack` | `N·K·ein` |
| packA (host input scatter) | `pack` | `M·K·ein` |
| weight DMA (DRAM->CBUF) | `wait` | `nMt·N·K·ein` (no weight reuse) |
| feature DMA (DRAM->CBUF) | `wait` | `M·K·ein` (data_reuse) \| `nNt·M·K·ein` (no reuse) |
| output WDMA (CBUF->DRAM) | `wait` | `M·N·eout` (KACC) \| `nKt·M·N·eout` (no KACC) |
| readback (host de-tile) | `read` | `M·N·eout` (KACC) \| `nKt·M·N·eout` (no KACC) |

The model is validated against a warm `512×3840×4096` fp16 profile (KACC+DATA_REUSE,
single-fd: `pack=36 packA=4 packB=32, wait=84, read=10` ms):

- The pack byte-model is exact. The model gives `packB:packA = 30 MB : 3.75 MB = 8:1`, and
  the measurement gives `32 ms : 4 ms = 8:1`. The scatter time tracks the scattered bytes
  precisely.
- Pack and read are not bandwidth-bound. The achieved pack rate is
  ~`33.75 MB / 36 ms ~0.9 GB/s` and readback is ~`4 MB / 10 ms ~0.4 GB/s`, both ~5 % of
  LPDDR streaming (~17 GB/s). The host scatter/gather is latency/index-math-bound
  (random-stride writes + fp16↔fp32 convert). That is the "memory/gather-bound, not
  instruction-bound" conclusion above, with a number. Cutting pack/read *bytes* (e.g.
  through quant) buys far less than cutting their *gather pattern* (the NEON de-tile).
- The model quantifies DATA_REUSE. Turning data_reuse off changes only `fDMA`, 3.75 -> 60 MB
  (`×nNt=16`), and moves the total 141.5 -> 197.8 MB. That is the DMA the `ROCKET_REUSE`
  CBUF-reuse win removes ([cbuf-reuse.md](../encodings/cbuf-reuse.md)).

### DDR PMU check

The system-level DDR PMU checks the model, and the result is split
[HW sweep 2026-08-28, RK1, 600 MHz]. The counter is `rockchip_ddr`, calibrated to 0.4%
against known bytes ([hw-byte-counters.md](hw-byte-counters.md) §5). Differencing 45 against
5 reps of this same `512×3840×4096` cell isolates 40 matmuls:

- The `data_reuse` term is right. The model predicts that `ROCKET_REUSE=0` adds `+116.25 MB`
  of reads and nothing to writes. The counter measures +120.98 MiB read and -1.93 MiB write,
  4% high. So a counter confirms the tiling and the loop order the model assumes.
- **The absolute total is 3.57x the model**: 475.95 MiB measured against 133.50 MB. The
  excess is the host scatter. Hoisting `packB` out of the loop (the prepacked path, same
  arithmetic) removes 249.34 MiB per call against a 30.00 MB term: 8.3x, in both directions.
  A scatter moves useful chunks smaller than a cache line, so the bus moves a whole line per
  chunk and moves it twice.
- **The `KACC` sensitivity is wrong on the read side.** The model predicts that
  `ROCKET_KACC=0` adds `+28 MB` of writes and no reads. The counter measures +20.4 MiB write
  and +189.0 MiB read. So the int8 readback floor just below is a lower bound, not the
  figure.

The 3.57x excess does not overturn "pack is not bandwidth-bound". The 249 MiB over the ~78 ms
of wall the scatter adds per call is ~3.4 GB/s against a ~17 GB/s ceiling. The headroom is 8x
smaller than the 0.9 GB/s figure in the validation list implies. So quote the achieved rate as
a rate on the *bus*, not on the operand. The 78 ms is a three-rep timing with the first rep
included, so it is the loose end of the two.

### The int8 readback floor

The model makes the int8 readback floor concrete. Take the same shape in int8 (no on-NPU
K-accumulation, int32 out). There `oWDMA` + `readback` = `nKt·M·N·4` *each* = 80 MB + 80 MB =
160 MB of output traffic. The fp16-KACC path moves `4 MB + 4 MB`. The int8 total is 208.8 MB >
fp16's 141.5 MB, despite int8 weights being half the bytes. The un-K-accumulated int32
readback (`∝ M·N·nKt`) is why int8 prefill loses (§"In-model int8 and int4 prefill").

The model shows this for any shape before anything runs. It is pure and runs anywhere (CTest
`bytes_moved_rocket`, which cross-checks its tile count against the planner's `njobs`).
