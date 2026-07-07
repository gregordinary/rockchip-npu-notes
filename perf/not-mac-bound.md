# The matmul is not MAC-bound — quantization does not speed up prefill

This is the most important — and most counterintuitive — performance finding from
the whole project. It is a **negative result**, and it is load-bearing for anyone
who, like us, assumed int8/int4 would be faster.

> **Scope it honestly (read [#scope](#scope--is-this-permanent) first).** This is a
> measurement *at the current operating point* (resident, multicore, 600 MHz), where
> the binding constraint is a DMA/dispatch floor. It is **not** a proven permanent
> property of the silicon. Quantization's MAC advantage is gated *behind* the
> dtype-independent dispatch floor — it is a later-stage lever, not a dead one. This
> project has already seen "X doesn't help" flip once the underlying bottleneck moved
> (the 200 MHz clock made "CPU-side levers are dead" true; raising it revived them).

## The headline

On the FOSS `rocket` path at 600 MHz, **resident multicore matmul throughput is
~460 GOP/s across precisions.** Measured on the same `512×3840×4096` shape, all
three datatypes resident across 5 worker fds:

| dtype | GOP/s | MAC advantage | expressed? |
|---|---:|---|---|
| fp16 (+K-accum +DATA_REUSE) | 461 | 1× | — |
| int8 | 386 | 2× | **no** |
| int4 | 413 | 4× | **no** |

Other shapes land in the same band: small-tile fp16 453 / int8 478 / int4 439 (int4
single-pass K); ffn-down fp16 486 / int8 452 / int4 463. **Everything ties within
~20%.** int8's 2× MAC and int4's 4× MAC simply **do not express as speed.**

~460 GOP/s is **~15% of the fp16 MAC peak** and **~4% of the int4 peak**. The MAC
array is mostly idle. The NPU is bound by a **dtype-independent floor** — the DMA to
load operands into CBUF plus per-job dispatch latency — not by compute.

## Two independent confirmations: not MAC-bound, not readback-bound

1. **It's not MAC-bound** — the 2×/4× quant advantages would show up if it were; they
   don't.
2. **The common floor is not dominated by K-tile readback** — int4 single-pass-K
   (`nKt=1`, *zero* K-tile readback) was **no faster** (439 ≈ the rest). If readback set
   the floor, eliminating it would win. It didn't. This is what let us *descope* the
   int4 EW K-accum rung before building it (see
   [../encodings/k-accumulation.md](../encodings/k-accumulation.md)). **Note the precise
   claim:** readback doesn't *set the floor*, but it still decides dtype ordering around
   it — readback can only push a dtype *below* the floor (int8's un-K-accumulated int32
   readback ∝ `M·N·nKt` is extra traffic — see §"int8 is slower than fp16"), it cannot
   lift one already sitting at the floor (fp16-KACC, int4 single-pass) above it. "Not
   readback-bound" means *removing readback from a floor-sitting dtype doesn't raise the
   floor* — not that readback never matters.

What's left is the DMA/dispatch floor: loading tiles into the CBUF and the
per-NPU-job launch/fence overhead. Those are dtype-independent.

## The in-model consequence: int8 prefill is slower than fp16

In the live model int8 loses, decisively:

- **Resident int8 prefill ≈ 9.1 t/s vs resident fp16 ≈ 15.1 t/s = 0.60×.** [HW sweep, 600 MHz]

fp16 has on-NPU K-accum (`read ∝ M·N`); int8 cannot (its int32 partials exceed the EW's
≤16-bit operand — see [../encodings/k-accumulation.md](../encodings/k-accumulation.md)). So
int8's un-K-accumulated int32 readback (`∝ M·N·nKt`, 2× the bytes of fp16) eats its MAC
advantage and then some. Making int8 *resident* removes the per-call requant + packB
overhead (+25–58% over the naive int8 path), but the readback wall is untouched and caps it
below fp16.

A same-weights precision sweep confirms it in-model — Gemma-4-12B F16 GGUF through the *one-shot*
paths (re-quantize each prefill), pp512, `performance` governor:

| path | t/s | vs fp16 |
|---|---:|---:|
| fp16 + KACC | 14.4 | 1.00× |
| int8 + Hadamard | 6.9 | 0.48× |
| int4 + Hadamard | 4.4 | 0.30× |
| CPU (8 threads) | 4.6 | 0.32× |

[HW sweep, 2026-06-24, RK1 7.1.0-1, 600 MHz] Lower precision lowers prefill monotonically: the
one-shot quant + int-readback cost stacks onto the dtype-independent floor. These are the
per-call-requant paths; the *resident* int8 above amortizes the requant and lands higher (0.60×),
still below fp16. A quantized `Q8_0` GGUF (on-the-fly dequant→fp16, half the RAM at 11.8 GB) ran
5.7 t/s — slower than native F16 from the per-call dequant, but still above the CPU and at half
the footprint, which is the model-fit payoff quantization actually buys here. That per-call
dequant is **per micro-batch** and amortizes with `-ub`: raising the micro-batch ~2×'s
quantized prefill, and a routing floor (`ROCKET_MIN_M_QUANT`) keeps sub-crossover quant
prefills on the CPU — see [quant-prefill-microbatch.md](quant-prefill-microbatch.md).

**Resident int4 (group-wise) repeats the resident-int8 lesson, more sharply.** Holding the
group-wise + Hadamard int4 weights resident (`ROCKET_INT4_RESIDENT`) removes the per-call
weight scatter exactly like resident int8 does, and the in-model gain is **~2×** over the
one-shot int4 path — **6.94 t/s vs 3.56 (resident vs non-resident int4), fp16 13.18, pp512
@600 MHz** [HW sweep, 2026-06-25]. But it is still **0.53× fp16**: a group-wise int4 matmul
reads back one int16 partial *per K-group* (`read ∝ M·N·nKt`, `nKt = K/group ≈ 120` on the
deep K=15360 FFN), so the same un-K-accumulated-readback wall that caps int8 caps int4 harder
(finer groups ⇒ more readback). The payoff is footprint: the resident int4 weights are **¼**
the NPU-BO bytes of fp16 (2634 MB vs ~10.5 GB for the offloaded set). So the resident path
turns int4 from "RAM win at a steep speed cost" into "RAM win at half fp16 speed" — quantization
still buys footprint, not throughput, and the lever that would change that is the dispatch/
readback floor, not the datatype.

## The proprietary stack's int8 win is bandwidth, not a capability the open path lacks

The proprietary rknpu2 / rk-llama.cpp stack runs int8 LLM prefill faster than its *own*
fp16; the FOSS `rocket` path does not reproduce that ordering. The cause is **not** a
hardware feature the open path can't reach. Three mechanisms were proposed for it; all three
are settled:

- **On-device int32 K-accumulation is real, and bounded the same way for everyone.** The
  conv accumulates K on-chip only *within one CBUF-resident K-tile*. There is **no register
  field that accumulates the CACC across CBUF passes**: the CORE block has no
  accumulate-vs-reset control, and the conv task splitter splits spatial height, never the
  channel (K) axis [source-confirmed: Mesa `rkt_task.c`]. Beyond one tile the partials are
  summed through DRAM via the same DPU-EW path the `ROCKET_KACC` mechanism uses, capped at the
  ≤16-bit operand — so there is **no** on-device cross-tile int32 K-accum on any stack.
  [HW sweep + source-confirmed; see [../encodings/k-accumulation.md](../encodings/k-accumulation.md)]
- **On-chip SRAM stages whole tensors, not partials.** The 956 KB NPU SRAM holds *weight* or
  *internal (activation)* tensors to relieve DDR bandwidth; the vendor's own doc notes it "may
  have a certain impact on inference time" and its per-layer example is a vision CNN. It is not
  a partial-sum accumulator, and 956 KB cannot hold an LLM's weights or a prefill activation
  matrix. [source-confirmed: see [sram-nbuf.md](sram-nbuf.md)]
- **The win is operand bandwidth at a lower dispatch floor.** RK3588's RKLLM quantizes to
  **W8A8** — 8-bit weights *and* activations, the only LLM quant the chip's toolkit offers — so
  int8 halves the bytes moved for both operands. The vendor's own Gemma int8 prefill runs at
  **40–58% NPU utilization**: their stack is *also* not MAC-bound, so int8 is buying them bytes,
  not MACs. On a path bound by dispatch/DMA, halving the operand bytes helps where the 2× MAC
  cannot. [W8A8-only is the vendor LLM path; utilization from rk-llama.cpp forum benchmarks, a
  single external source.] The remaining vendor edge is dispatch efficiency — a batched
  whole-graph submit (≈10× fewer kernel transitions, see
  [iova-and-multicore.md](iova-and-multicore.md) §batched submit), 3-core dispatch (matched on
  our path), and the 1 GHz default clock (our clock patch reaches 600 MHz).

The "+200–400%" sometimes quoted for the vendor's int8 prefill is the **6-vs-3 TOPS spec-sheet
ratio** (6 TOPS int8 / 3 TOPS fp16), not a measured fp16→int8 A/B; the realized 40–58%
utilization is the better guide. That the vendor is also sub-60% utilized independently
corroborates the not-MAC-bound result above. The causal account — bandwidth at a lower floor,
not a secret accumulator — rests on the capability facts (HW + source-confirmed) plus their
utilization numbers; an isolated fixed-clock fp16-vs-W8A8 prefill sweep on their stack, with a
DDR/NOC PMU read, would convert it from well-supported to proven.

## So what is quantization good for here?

**RAM, not speed.** Quantization's payoff on this hardware is:

- **Model size / fitting in memory** — run a model that doesn't fit in fp16 (int4
  Gemma weights are ~¼ the bytes; the int4 matmul is fully working and bit-exact).
- **IOVA residency** — a quantized model fits the per-fd IOVA windows whole (see
  [iova-and-multicore.md](iova-and-multicore.md)).
- **Decode coexistence** — a quantized GGUF for CPU decode alongside NPU prefill.

At the current operating point it is **not** a prefill throughput win — its MAC
advantage is buried under the DMA/dispatch floor. Say that plainly, but say it with
its scope (below), not as a permanent law.

<a name="scope"></a>
## Scope — is this permanent?

Treat "quantization doesn't speed up prefill" as **bottleneck-conditional, not a
hardware fact.** The measured tie is solid and reproducible; the *generalization* to
"quant can never help prefill here" is stronger than the evidence.

**How it could flip.** The explicitly-named remaining prefill lever is the per-job
**dispatch floor** (fewer, bigger NPU jobs). That is dtype-independent, so it lifts
all precisions first — but we are at only ~15% of fp16 MAC peak, so there is a long
dtype-independent ramp before MAC binds at all. *Only after* dispatch/DMA stop binding
and the NPU approaches its MAC ceiling would int8's 2× and int4's 4× have room to
express. So quantization is plausibly a **later-stage lever, gated behind the dispatch
floor** — not eliminated. (Precedent: in this project "CPU-side levers are dead" and
the fp16 EW K-accum were both "dead" until the clock / a config fix moved the
bottleneck, then they worked.)

**Why it might genuinely stay flat (the structural counter-argument).** The CBUF is
only 384 KB, which caps the MAC-work-per-job — you may simply be unable to make jobs
big enough for MAC time to dominate DMA/dispatch latency, in which case the floor is
structural and quant never gets its opening. A supporting data point: int4 reads ¼ the
weight bytes *and* has 4× MAC and still didn't win, so the floor is not weight-DMA
bandwidth either — it is latency-like (dispatch / fence / CBUF-fill), which big tiles
amortize but cannot remove.

**The floor is not entirely irreducible.** One slice of it — host-side **cache-sync on an
over-allocated, repeatedly-synced output BO** — is reducible: the KACC path issues one job
*per K-tile* and syncs the output BO each time, but the BO was `BATCH`-sized while only
`nMt·nNt` tiles are live, so `PREP_BO`/`FINI_BO` cache-synced ~8× too much, `nKt` times
(sync cost is ∝ BO size, see [bo-sync-cost.md](bo-sync-cost.md)). Right-sizing it cuts
`sync` 127→15 ms and lifts **resident** fp16 **~+11%** (drift-controlled A/B; up to +17%
best-case), bit-exact.
**Two honest caveats:** (1) it did **not** reorder the dtype spread (quant still doesn't
pull ahead — the lever lowered the *common* floor, it didn't give MAC its opening);
(2) it is **invisible in-model** for Gemma F16 prefill, whose K>2048 shapes *stream*
(re-pack B every call) so their `sync` is dominated by the **weight** BO, not the output
BO — the win lives at the **resident** operating point. The residual `wait` term (CPU
blocked on the fence) is the genuinely NPU-bound part, and "fewer/bigger jobs" for *it*
is still open (the KACC K-dependency forces `nKt` sequential fences). **Net:** part of the
"dispatch floor" is reducible host overhead (~+11% resident fp16); the residual NPU-compute
floor still ties the precisions. Quant's *known* payoff remains RAM / model-size /
decode-coexistence.

## The only real prefill levers are dtype-independent

Since the floor is DMA + dispatch + clock:

1. **The clock** (200 → 600 MHz = 1.43×; 900 MHz/1 GHz gated, see
   [clock.md](clock.md)).
2. **The per-job dispatch floor** — fewer, bigger NPU jobs (fusion, larger ubatch), and
   **batching independent tasks into one HW kick** instead of one submit + completion IRQ
   per task (the open-vs-vendor submit-count gap; see
   [iova-and-multicore.md](iova-and-multicore.md) §batched submit).
3. **Right-sized cache-synced BOs** (~+11% *resident* fp16, cuts host `PREP_BO`/`FINI_BO`
   maintenance on the repeatedly-synced KACC output BO — see
   [bo-sync-cost.md](bo-sync-cost.md); invisible to streaming/in-model prefill).
4. **CBUF DATA_REUSE** (+7%, cuts a real DMA — see
   [../encodings/cbuf-reuse.md](../encodings/cbuf-reuse.md)).
5. **fp16 on-NPU K-accum** (+19%, cuts host readback — fp16 only).

These compound and are precision-independent. The deliverable from the datatype work
is the **completeness of the matrix** (int4/int8/int16/fp16 all working and correct),
not a speedup.

### Dispatch-floor reducers — small-job paths only, *flat on prefill*

Two later-measured levers cut the **per-submit** floor (not the per-tile compute), so
they belong here for completeness but with a sharp caveat: **both are ~flat on the big
tiled prefill matmul this doc is about**, because prefill is a few large submits where
the per-submit overhead vanishes under ms-scale tile compute. They pay on
**many-small-submit** paths — decode GEMV, multi-fd contention, the detection
throughput pool *under contention* — *not* on prefill, and *not* on single-stream
detection (host-gather-bound, see below). Do not read them as prefill speedups.

- **IRQ affinity** — the default IRQ mask services the NPU completion IRQ on an A55
  little core; binding the 3 NPU IRQs to an A76 + co-locating the waiter cuts the
  submit floor **51 → ~27 µs (−47%)**. Runtime/system config, no code change.
- **IOMMU keep-attached** — stock `rocket` re-attaches the IOMMU domain on every job;
  keeping the per-context domain attached across same-fd jobs removes **~15–20 µs
  (~38%)** per submit. (Driver patch.)
- **CPU governor / frequency** — the per-submit floor is CPU-side work (the submit
  `ioctl`, the blocking wait on the completion IRQ), so it scales with the **CPU** clock,
  not the NPU clock. On an idle box an `ondemand`/`interactive` governor parks the cores
  low between submits, inflating and jittering any submit-bound measurement. An external
  RKNN-path writeup pins the size of this: one YOLOv8s `rknn_run` swung **59 → 35 ms
  (−41%)** purely by moving the CPU governor from `ondemand` (cores at 408 MHz, idle box)
  to `performance` (1.8 GHz), with the **NPU** clock untouched at 1 GHz throughout;
  pinning the NPU governor *alone* changed nothing (the NPU `ondemand` already ramps to
  its ceiling under load). Locking CPU **and** NPU to `performance` also collapsed the
  run-to-run jitter (≈6 ms → 0.64 ms), the on-demand ramp being the jitter source.
  [external, proprietary path — corroborates the CPU-side submit floor]
  On **our** path the SigLIP-B/16 encoder (many small attention submits + host
  softmax/de-tile per layer) shows the same shape **[HW sweep]**: resident warm median
  **5.44 s `schedutil` → 2.71 s `performance` (−50%)**, jitter ±1.5 s → ±0.02 s, NPU held at
  600 MHz throughout. The readback-bound matmuls are the most governor-sensitive (readback is
  host work) — the same reason a NEON KACC de-tile gather (`detile_store_f16`, shared by the
  prepacked/stream/multicore path) shaved a further ~0.35 s. Pin the CPU governor before any
  submit-bound bench ([encodings/siglip-encoder.md](../encodings/siglip-encoder.md),
  `rocket-userspace/tools/npu_perf_governor.sh`).

Both measured flat on `matmul_tiled_rocket 512 3840 4096` (one big job) and large on
`submit_overhead_rocket` (tiny 1-task jobs). See
[iova-and-multicore.md](iova-and-multicore.md) §IRQ affinity / §per-job IOMMU cost.

**Detection single-stream is the same shape [HW sweep 2026-06-29].** Coalescing a native-int8/uint8
conv's per-tile submits into one job (`ROCKET_CONV_BATCH`, the gapped lever-1 — int8-safe because the
CACC clears per kick, unlike fp16 chaining) is **flat on warm MobileDet** (~250 ms, 227 → 215 submits)
and flat across 4 parallel MobileDet processes too. A tiled conv's wall is the host cube
scatter/descatter, not the submit floor, and most native-u8 convs are single-tile anyway — the 227
submits are the **matmul multicore worker fan-out**, not conv tiling. Submit-coalescing pays only on a
**conv-tile-heavy** unit under **multi-process contention** (+7.6% aggregate at P=4, the contended
submit/IOMMU path being the shared bottleneck). So the detection single-stream lever is **host-gather
reduction**, exactly as for prefill readback: NEON the requant epilogue (the M-major 1×1 requant
vectorizes 8 channels/step — OC contiguous on both the int32 read and the NHWC write — bit-exact, **+5.5%
MobileDet / +9.3% EfficientDet-Lite0**) and the cube scatter, not submit-batching. The detection profile
mirrors the prefill one below: mm + conv buckets dominated by host pack/de-tile, the dispatch floor a
small fraction.

**Measurement hygiene that follows:** before timing any submit/dispatch floor
(`submit_overhead_rocket`, decode GEMV, the detection convs), pin the A76 cores to
`performance` — otherwise the governor ramp confounds the µs-scale number the way it
confounded the external `rknn_run` above. This is separate from the NPU cold-clock
throwaway in [clock.md](clock.md): one is the CPU governor between submits, the other is
the NPU clock ramping from its 200 MHz idle park.

## The CPU-side profile

The wall-time breakdown depends on the operating point. At the unoptimized baseline (cold
clock, host K-accum, no reuse) it is roughly: layout pack/scatter + readback ≈ **67%**, NPU
FLOPs only ~25% — which is why the optimization ladder targets CPU scatter, readback, and
per-call setup, and why those wins compound at every model size and precision. At the
current operating point (600 MHz, on-NPU K-accum, DATA_REUSE) it is `wait` ~60–68%, packB
~22%, read small.

### The CPU work is memory/gather-bound, not instruction-bound

Two measurements pin down the *nature* of the remaining CPU cost — it is bound by
memory traffic and index math, not by instruction throughput:

- **Compiler flags are flat.** `-O3 -mcpu=native -DNDEBUG`, `+-flto`, and
  `+-fno-math-errno -fno-trapping-math` over the `-O2` baseline move the matmul
  within ±3% (noise; LTO marginally worse). `ROCKET_MM_PROFILE` shows `packB`
  (124 ms) and `read` (214 ms) **byte-identical** across builds on `512×15360×3840`
  — the compiler has nothing to optimize because these loops are scatter/gather over
  DRAM, not ALU-bound. (The genuinely CPU-bound, branchy *llama.cpp decode/sampling*
  path is a separate story where flags/PGO could still help — untested here.)
- **The readback de-tile NEON-vectorizes ~3×.** The output-cube de-tile (the single
  largest CPU component) reads, for a fixed row, 8 contiguous fp16 per column-group
  that land in 8 contiguous fp32 accumulators — one `vld1q_f16` → 2× `vcvt_f32_f16`
  → 2× `vaddq_f32`. That cut `read` 214 → ~76 ms (−64%) and the single-fd matmul wall
  739 → 604 ms (−18%); ~+7% multicore (the fan-out already overlaps readback across
  the 3 cores). Bit-exact. So the de-tile *was* index-math-bound (restructuring the
  gather helps) but not instruction-issue-bound (flags don't) — consistent with the
  dtype-independent DMA/dispatch floor above. **Offloading the de-tile to a DMA engine
  does not help:** the move isn't bandwidth-bound, and the RGA 2D blitter's throughput
  ceiling (~5.8 GB/s, best case) is below a single CPU core's `memcpy` (~13 GB/s), while
  the NEON path already fans across 3 — so the NEON de-tile stays the path
  ([rga-detile.md](rga-detile.md), and by the same argument the PL330 DMAC).

## An analytical bytes-moved model (no HW counter)

The real DDR/DMA byte counters are dead on rk3588 (reading the `0x2xxx` page
hard-locks the SoC; `0x80xx` is config-only — [hw-byte-counters.md](hw-byte-counters.md)),
and RKNN's "Total Memory R/W per frame" is itself *computed from the graph*, not read
from HW. So **`tests/bytes_moved_rocket.c`** computes the DRAM bytes each phase moves
analytically from the shape + the real tiling (`rocket_matmul_plan`, pure/no-HW) + the
dtype + the reuse mode. It maps each term to a `ROCKET_MM_PROFILE` bucket:

| phase | bucket | formula |
|---|---|---|
| packB (host weight scatter) | `pack` | `N·K·ein` |
| packA (host input scatter) | `pack` | `M·K·ein` |
| weight DMA (DRAM→CBUF) | `wait` | `nMt·N·K·ein` (no weight reuse) |
| feature DMA (DRAM→CBUF) | `wait` | `M·K·ein` (data_reuse) \| `nNt·M·K·ein` (no reuse) |
| output WDMA (CBUF→DRAM) | `wait` | `M·N·eout` (KACC) \| `nKt·M·N·eout` (no KACC) |
| readback (host de-tile) | `read` | `M·N·eout` (KACC) \| `nKt·M·N·eout` (no KACC) |

**Validated against a warm `512×3840×4096` fp16 profile** (KACC+DATA_REUSE, single-fd:
`pack=36 packA=4 packB=32, wait=84, read=10` ms):

- **The pack byte-model is exact.** Model `packB:packA = 30 MB : 3.75 MB = 8:1`;
  measured `32 ms : 4 ms = 8:1`. The scatter time tracks the scattered bytes precisely.
- **…but pack/read are NOT bandwidth-bound.** Achieved pack rate ≈ `33.75 MB / 36 ms
  ≈ 0.9 GB/s`, readback ≈ `4 MB / 10 ms ≈ 0.4 GB/s` — both **~5 % of LPDDR streaming
  (~17 GB/s)**. The host scatter/gather is latency/index-math-bound (random-stride
  writes + fp16↔fp32 convert), exactly the "memory/gather-bound, not instruction-bound"
  conclusion above, now with a number. Cutting pack/read *bytes* (e.g. via quant) buys
  far less than cutting their *gather pattern* (the NEON de-tile).
- **DATA_REUSE quantified.** Turning data_reuse off changes only `fDMA` 3.75 → 60 MB
  (`×nNt=16`), total 141.5 → 197.8 MB — that is the DMA the `ROCKET_REUSE` CBUF-reuse
  win removes ([cbuf-reuse.md](../encodings/cbuf-reuse.md)).

**It makes the int8 readback floor concrete.** Same shape, int8 (no on-NPU K-accum,
int32 out): `oWDMA` + `readback` = `nKt·M·N·4` *each* = **80 MB + 80 MB = 160 MB** of
output traffic, vs fp16-KACC's `4 MB + 4 MB`. int8's **total 208.8 MB > fp16's
141.5 MB despite int8 weights being half the bytes** — the un-K-accumulated int32
readback (`∝ M·N·nKt`) is why int8 prefill loses (§"int8 is slower than fp16"). The
model lets you see this for any shape before running anything. Pure, runs anywhere
(CTest `bytes_moved_rocket`, cross-checks its tile count against the planner's `njobs`).
