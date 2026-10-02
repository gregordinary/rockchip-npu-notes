# K-accumulation via the DPU eltwise unit (DPU-EW / DPU-RDMA)

When K is too large to contract in one CBUF-resident tile, the matmul splits into
`nKt` K-tiles whose partial products must be summed. One option reads each partial back
and sums on the host in fp32, which reads every output tile `nKt` times
(`read ∝ M·N·nKt`). The other accumulates the K-partials on the NPU through the DPU
eltwise (EW) add path, and reads each output tile once (`read ∝ M·N`). For fp16 the
on-NPU EW path is the shipping default (+19% on Gemma-4-12B prefill, the operating mode).
The host fp32 sum is the byte-exact fallback and oracle (`ROCKET_KACC=0`).

There is no on-chip third option: the conv accumulator cannot span tiles. The conv's CACC
reduces K only within one CBUF-resident tile. The CORE register block has no
accumulate-vs-reset control, so nothing makes a later CBUF pass add into the prior CACC
contents. The conv task splitter splits spatial height, never the channel (K) axis
[source-confirmed: Mesa `rkt_task.c`].

So a K larger than one CBUF tile is always `nKt` separate tasks whose partials leave the chip.
The only choice is where the sum runs: on the host (fp32 for fp16, int64 for the integer
types) or in the DPU-EW add below.

The EW add works for fp16 and ships. No integer path ships. The EW ALU adds int32 exactly once
the whole precision bundle says integer [HW sweep]
([sdp-stage-precision.md](sdp-stage-precision.md) §"The EW stage in integer mode"). An integer
K-accumulation built on it would not move a prefill, because the wall is not readback.

## fp16 EW K-accumulation

The mechanism mirrors Mesa's working `add_tensor` residual-add geometry exactly
[HW sweep, source-confirmed]. The conv result is the DPU main input (MRDMA fed), and the
ERDMA reads the running partial from DRAM. The EW ALU adds them, and the WDMA writes back.
The loop is ki-outer, with a ping-pong between two output BOs. An in-place add corrupts,
because the ERDMA would read the buffer the WDMA is writing.

The configuration that works (all fields are reg bits [4:31], i.e. the geometric
value `<< 4`):

- `DPU_EW_CFG = 0x108202C0`: per-pixel EW mode (bit28), `EDATA_SIZE=2` (16-bit),
  `EW_ALU_ALGO=2` (add), RELU/LUT bypass, `EW_OP_SRC=1`.
- `DPU_RDMA_ERDMA_CFG = 0x40000008`: ERDMA per-pixel mode (bit30), `DATA_SIZE=2`
  (16-bit fp16).
- `DPU_RDMA_FEATURE_MODE_CFG = 0x17D40`: `COMB_USE(5)` combines the conv main-data
  with the ERDMA operand (MRDMA enabled here, unlike the plain path).
- `SURF_NOTCH = EW_SURF_STRIDE = MAX(out_w·out_h, 12) << 4`: the planar
  pointer-advance. **Leaving it 0 is a trap**: the ERDMA never advances, reads the
  offset-0 atom, and broadcasts it to every position and surface. The `MAX(.,12)` floor
  over-states the stride for M<12, so test with M>=12.
- ERDMA `EW_BASE = add_dma + out_w·out_h·16` (one surface offset, 16 B/position =
  8 fp16 channels = the atomic K block), while MRDMA `SRC_BASE = add_dma` (no offset).

The fp16 EW-add geometry above is HW-verified on this datapath. Mesa's `add_tensor`
residual-add is int8 with `EDATA_SIZE=1`, so its exact values do not transfer to the 16-bit
fp16 path. The allbilly EW encodings are another reference, listed in
[SOURCES.md](../SOURCES.md).

### Single-knob sweeps

Three registers must be right at once: per-pixel (not per-channel) mode, a nonzero
`SURF_NOTCH` and `EW_SURF_STRIDE`, and the right ERDMA base. If any one is wrong,
`EW_SURF_STRIDE` looks inert and the path looks impossible, so no single-knob sweep can
converge. Copy a known-good geometry (Mesa's `add_tensor`) wholesale rather than sweeping
one field at a time. With all three matching, `N=16…384` (2 to 48 surfaces) and `M` up to
512 all pass.

### Precision cost

The EW running sum accumulates in fp16, and each add rounds. The host path sums fp16
partials in fp32. On a standalone matmul the difference is ~0.2-0.4%
(max_abs ~8 on a 512×3840×4096 whose outputs reach ~4000, worst ~56 on the deepest FFN).

In practice it does not flip greedy LLM tokens. Gemma-4-12B output stays coherent,
and the in-model per-op verify at M=512 is `nonfinite=0`, worst `max_abs~0.38`. So the fp16
EW K-accumulation is good enough here, and it is not bit-exact. Measured: +19% Gemma-4-12B
prefill (pp2048, 600 MHz, 2026-06-15).

## Integer EW K-accumulation

Per op, int8 without K-accumulation reads its int32 partials back at 2x fp16's bytes, so it
is strictly worse per op than fp16 with K-accumulation. EW is the only per-element SDP
stage. BS and BN broadcast a per-channel `[C]` vector, which is useless for per-pixel
K-partials (see [sdp-stage-precision.md](sdp-stage-precision.md)). Three approaches through
the EW stage fail as programmed here:

1. **int32 integer-add via `EW_ALU_ALGO`** (`0x10C202C0`) returns garbage: the EW adds the
   int32 bit patterns as float.
2. **int32 via the `EW_OP_TYPE=1` integer path** (`0x10C203C4`) returns garbage: a true add
   of conv(226)+op(1000000) gives 1000226, and the EW returns `0x3A7C80`, an fp16 inf/NaN
   pattern in the low 16 bits. The probe used a constant operand, so the addressing is not
   the cause.
3. **fp32 EW-add** (cast int8-conv's int32 -> fp32, then add via the float path). The pieces
   exist. The int8-conv -> fp32 output cast is bit-exact (still needs `size_e=7`). A 32-bit
   fp32 EW operand read works too (`EDATA_SIZE(3)`=32-bit, the same read the fp32-output matmul
   uses).

   But fp32 cannot hold the sum. A Kt=768 tile reaches ~12M and the full K-sum ~248M,
   past fp32's exact-integer range (2²⁴ ~16.7M). So accumulating the int32 partials as fp32
   drops the low bits. int16 EW-add cannot hold them either.

Probes 1 and 2 fail on a partial register bundle, not on the ALU. Both move `DPU_EW_CFG`
and leave the precision fields at fp16. DPU `0x4010` (`RKNN_dpu_data_format`) enumerates
`out_precision`, `in_precision` and `proc_precision` identically and explicitly [TRM,
`Rockchip RK3588 TRM V1.0-Part1`]:

| code | precision |
|---|---|
| `3'd0` | Integer 8bit |
| `3'd1` | Integer 16bit |
| `3'd2` | Float point 16bit |
| `3'd3` | Bfloat 16bit |
| **`3'd4`** | **Integer 32bit** |
| `3'd5` | Float point 32bit |
| `3'd6` | Integer 4bit |

Under the whole bundle the EW ALU computes ADD, MINUS, MAX and MIN exactly at int8, int16 and
int32 [HW sweep, both drivers, `tests/ew_int_probe.c`]. ADD and MINUS saturate. With the
triple left at fp16 the int32 element sizes match no model of an add. allbilly/rk3588
measures the same [source-confirmed]. The bundle and the per-op table are in
[sdp-stage-precision.md](sdp-stage-precision.md) §"The EW stage in integer mode".

An int32 K-accumulation on that ALU is not measured. It needs the conv's CACC result as the
EW main feed at an integer `proc_precision`, with the running partial on the ERDMA. The measured
program fed both operands from memory. Two facts stand regardless. fp32 cannot represent
an int32 K-sum exactly (a Kt=768 tile reaches ~12M and the full K-sum ~248M, past 2^24), so the
float route is not the answer. And the int8 readback lever, measured, does not move the wall,
because the wall is not readback ([../perf/not-mac-bound.md](../perf/not-mac-bound.md)).

**Set the whole precision bundle, not the ALU algorithm.** A probe that moves `DPU_EW_CFG` alone
reads a working integer ALU as garbage, as probes 1 and 2 did. Float mode forces the same rule:
three registers that move together.

## int4 EW K-accumulation

int4's int16 output has K-sums that stay within fp32's exact-integer range, so int4 EW
K-accumulation via the float path could be bit-exact. Two facts make that moot. int4 can
reach single-pass K (`nKt=1`, zero K-accumulation) by shrinking the output tile. The
matmul is also not readback-bound (single-pass int4 is no faster, see
[../perf/not-mac-bound.md](../perf/not-mac-bound.md)). So int4 EW K-accumulation is not
worth building: eliminating readback does not raise throughput here.

## Shipping paths

The path that ships for each datatype:

- **fp16:** the EW K-accumulation (the `ROCKET_KACC` path) is the default-on operating
  mode. It gains +5-20% over the host-sum fallback across pp512-2048 on 0.8B and 9B F16
  (peak +20% at 9B pp512) [HW sweep 2026-06-28], with coherent greedy output. CBUF
  DATA_REUSE follows automatically (~+7% more). Opt out with `ROCKET_KACC=0` for the
  byte-exact host sum.
- **int8, int16 and int32:** no on-NPU K-accumulation ships. The EW adds int32 exactly under
  the whole bundle, and the conv-fed composition a K-accumulation needs is not measured. Sum
  partials on the host (int64, bit-exact). To cut readback, grow Kt through the conv's native
  K-reduction (shrink Mt and Nt). That will not move the wall, because the wall is not
  readback.

### ki-fence chaining (`ROCKET_KACC_CHAIN`)

The KACC path fences once per ki-step (each ki>0 reads the prior partial). Chaining the
whole `[ki][tile]` sequence into one self-chained kick collapses nKt fences to one. The
hardware honors the in-kick read-after-write, so the chained path is byte-exact to the
per-ki path. That result is the proof of the in-kick-dependency property (see
[regcmd-task-model.md](regcmd-task-model.md) §"in-kick data dependency"). The ki-steps
are serially dependent, so a chained kick pipelines only the independent tiles within
each ki-block. The net effect tracks `gcap = BATCH/nKt` (BATCH=64)
[HW sweep 2026-06-30, 600 MHz]:

| gcap = BATCH/nKt | example | vs per-ki |
|---|---|---|
| >=3 (nKt <= 21) | nKt=20 | ~0.95 (fence savings win) |
| 2 (nKt 22-32) | nKt=24,32 | ~1.01 (slight loss) |
| 1 (nKt >= 33) | nKt=40 | ~1.08 (`wait` +18%, serial stalls) |

So the chained path ships as an adaptive opt-in, default-off. `=1` engages only at gcap>=3
and otherwise falls back, so it never regresses. `=2` forces any fitting nKt (the byte-exact
gate's strict gcap=1 test). Gemma FFN-down (nKt=40) falls back, so the end-to-end LLM gain
is ~0. The value is the in-kick-dependency finding and a regression gate, not a speed lever.
The gate is `tests/matmul_kacc_chain_rocket.c`, and the A/B bench is
`matmul_kacc_chain_bench.c`.

The diagnostic for the EW path is a standalone classifier (`matmul_accum_rocket.c` and
`matmul_accum_int8_rocket.c`) with a constant operand and a sentinel output. It separates
"can the EW add this type at all" from "is the addressing right" in a single run. Keep it:
it is the gate for any future EW attempt.
