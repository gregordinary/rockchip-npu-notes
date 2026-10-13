# K-accumulation on the DPU EW stage (DPU-EW / DPU_RDMA)

K-accumulation sums a matmul's partial results over K. When K is too large to contract in one
CBUF-resident tile, the matmul splits into `nKt` K-tiles. Each tile's partial is added to a
running sum. The DPU's eltwise (EW) stage can do that add on the NPU. The convolution result is
the DPU main input, and the ERDMA reads the running partial from DRAM. The EW ALU adds them, and
the write DMA (WDMA) writes the sum back. `DPU_EW_CFG`, `DPU_RDMA_ERDMA_CFG` and
`DPU_RDMA_FEATURE_MODE_CFG` select it.

The two places the sum can run differ in readback. A host sum reads every output tile `nKt`
times (`read ∝ M·N·nKt`), in fp32 for fp16 and in int64 for the integer types. The EW path reads
each output tile once (`read ∝ M·N`).

The convolution's own accumulator cannot span tiles, so these are the only two places. The
CORE's accumulator, CACC, reduces K only within one CBUF-resident tile. The CORE register block
has no accumulate-or-reset control, so nothing makes a later CBUF pass add into the prior CACC
contents. The convolution task splitter splits the spatial height, never the channel (K) axis
[source-confirmed: Mesa `rkt_task.c`]. A K larger than one CBUF tile is therefore `nKt`
separate tasks whose partials leave the chip.

EW is the only per-element SDP stage. BS and BN broadcast a per-channel `[C]` vector, which
cannot hold per-pixel K-partials (see [sdp-stage-precision.md](sdp-stage-precision.md)).

## Support status

| Datatype | On-NPU K-accumulation | Status |
|---|---|---|
| fp16 | The EW add, `ROCKET_KACC` | Supported (default). `ROCKET_KACC=0` sums on the host in fp32, byte-exact |
| int8, int16, int32 | None | Not supported. Software sums the partials on the host in int64, bit-exact |
| int4 | None | Not supported. int4 reaches single-pass K (`nKt=1`) by shrinking the output tile |

The EW ALU adds int8, int16 and int32 exactly when the whole precision bundle is integer
[HW sweep] ([sdp-stage-precision.md](sdp-stage-precision.md) §"The EW stage in integer mode").
An integer K-accumulation built on it is not characterized.

## Registers

**Conditions:** fp16, validated at `N=16…384` (2 to 48 surfaces) and `M` up to 512.

The fp16 geometry follows Mesa's working `add_tensor` residual-add geometry
[HW sweep, source-confirmed]. All the fields below are register bits [4:31], that is, the
geometric value `<< 4`:

| Register | Value | Fields |
|---|---|---|
| `DPU_EW_CFG` | `0x108202C0` | Per-pixel EW mode (bit28), `EDATA_SIZE=2` (16-bit), `EW_ALU_ALGO=2` (add), RELU and LUT bypassed, `EW_OP_SRC=1` |
| `DPU_RDMA_ERDMA_CFG` | `0x40000008` | ERDMA per-pixel mode (bit30), `DATA_SIZE=2` (16-bit fp16) |
| `DPU_RDMA_FEATURE_MODE_CFG` | `0x17D40` | `COMB_USE(5)` combines the convolution's main data with the ERDMA operand. The MRDMA is enabled here, unlike on the plain path |
| `SURF_NOTCH`, `EW_SURF_STRIDE` | `MAX(out_w·out_h, 12) << 4` | The planar pointer advance |
| ERDMA `EW_BASE` | `add_dma + out_w·out_h·16` | One surface offset: 16 B per position, which is 8 fp16 channels, the atomic K block |
| MRDMA `SRC_BASE` | `add_dma` | No offset |

Mesa's `add_tensor` residual-add is int8 with `EDATA_SIZE=1`, so its exact values do not
transfer to the 16-bit fp16 path. The allbilly EW encodings are another reference, listed in
[SOURCES.md](../SOURCES.md).

DPU `0x4010` (`RKNN_dpu_data_format`) enumerates `out_precision`, `in_precision` and
`proc_precision` identically and explicitly [TRM, `Rockchip RK3588 TRM V1.0-Part1`]:

| code | precision |
|---|---|
| `3'd0` | Integer 8bit |
| `3'd1` | Integer 16bit |
| `3'd2` | Float point 16bit |
| `3'd3` | Bfloat 16bit |
| **`3'd4`** | **Integer 32bit** |
| `3'd5` | Float point 32bit |
| `3'd6` | Integer 4bit |

## Programming requirements

Software must program three things together: per-pixel (not per-channel) mode, a nonzero
`SURF_NOTCH` and `EW_SURF_STRIDE`, and the ERDMA base above.

> [!CAUTION]
> With `SURF_NOTCH` left at 0, the ERDMA never advances. It reads the offset-0 atom and
> broadcasts it to every position and surface.

> [!CAUTION]
> An in-place add corrupts the sum, because the ERDMA reads the buffer the WDMA is writing.
> Software must alternate the running sum between two output BOs, with the K-steps as the
> outer loop.

The `MAX(.,12)` minimum over-states the stride for M<12, so validate this path with M>=12.

For an integer EW add, software must set the whole precision bundle to one integer code. The
bundle is the three precision fields in DPU `0x4010`, the DPU_RDMA `FEATURE_MODE_CFG`, and the
EW and ERDMA element sizes ([sdp-stage-precision.md](sdp-stage-precision.md) §"The EW stage in integer
mode").

> [!CAUTION]
> Changing `DPU_EW_CFG` alone, with the precision fields left at fp16, returns garbage from a
> working integer ALU. Float mode follows the same rule: three registers move together.

## Accuracy

The EW running sum accumulates in fp16, and each add rounds. The host path sums fp16 partials
in fp32. On a standalone matmul the difference is ~0.2-0.4%: max_abs ~8 on a 512×3840×4096
whose outputs reach ~4000, and worst ~56 on the deepest FFN.

The fp16 EW K-accumulation is not bit-exact, and it does not change greedy LLM tokens.
Gemma-4-12B output stays coherent, and the in-model per-op verify at M=512 is `nonfinite=0`,
worst `max_abs~0.38`.

## Integer and float paths

Under the whole precision bundle, the EW ALU computes ADD, MINUS, MAX and MIN exactly at int8,
int16 and int32 [HW sweep, both drivers]. ADD and MINUS saturate. allbilly/rk3588 measures the
same [source-confirmed]. The bundle and the per-op table are in
[sdp-stage-precision.md](sdp-stage-precision.md) §"The EW stage in integer mode".

**Not characterized:** an int32 K-accumulation on that ALU. It needs the convolution's CACC
result as the EW main feed at an integer `proc_precision`, with the running partial on the
ERDMA.

An fp32 EW add has the pieces it needs. The int8-convolution -> fp32 output cast is bit-exact
(it still needs `size_e=7`). A 32-bit fp32 EW operand read works too (`EDATA_SIZE(3)`=32-bit,
the same read the fp32-output matmul uses). fp32 cannot hold an int32 K-sum exactly, though. A
Kt=768 tile reaches ~12M and the full K-sum ~248M, past fp32's exact-integer range
(2²⁴ ~16.7M). Accumulating the int32 partials as fp32 drops the low bits, and an int16 EW add
cannot hold them either.

int4's int16-output K-sums stay within fp32's exact-integer range, so an int4 EW
K-accumulation through the float path is expected to be bit-exact [expected].

Per op, int8 without K-accumulation reads its int32 partials back at 2x fp16's bytes, so it is
slower per op than fp16 with K-accumulation. Readback does not determine prefill wall-clock time
([../perf/not-mac-bound.md](../perf/not-mac-bound.md)), and single-pass int4 is no faster. An
integer or int4 K-accumulation therefore does not raise prefill throughput. To reduce readback
on the integer types, grow Kt through the convolution's own K-reduction, by shrinking Mt and Nt.

## Software support

**Conditions:** RK3588, 600 MHz.

`ROCKET_KACC=1`, the EW path, is the default for fp16. It gains +19% on Gemma-4-12B prefill
(pp2048, 2026-06-15). Across pp512-2048 on 0.8B and 9B F16 models it gains +5-20% over the host
sum [HW sweep 2026-06-28]. The peak is +20%, at 9B pp512 [HW sweep 2026-06-28]. Greedy output
stays coherent.

CBUF DATA_REUSE follows automatically, for ~+7% more. `ROCKET_KACC=0` selects the byte-exact
host sum.

### ki-fence chaining (`ROCKET_KACC_CHAIN`)

The KACC path fences once per K-step, because each step after the first reads the prior
partial. Chaining the whole `[ki][tile]` sequence into one self-chained kick collapses `nKt`
fences to one. The hardware honors the in-kick read-after-write, so the chained path is
byte-exact to the per-K-step path (see [regcmd-task-model.md](regcmd-task-model.md) §"in-kick
data dependency").

The K-steps are serially dependent, so a chained kick pipelines only the independent tiles
within each K-step. The net effect tracks `gcap = BATCH/nKt` (BATCH=64)
[HW sweep 2026-06-30, 600 MHz]:

| gcap = BATCH/nKt | example | vs per-ki |
|---|---|---|
| >=3 (nKt <= 21) | nKt=20 | ~0.95 (fence savings win) |
| 2 (nKt 22-32) | nKt=24,32 | ~1.01 (slight loss) |
| 1 (nKt >= 33) | nKt=40 | ~1.08 (`wait` +18%, serial stalls) |

`ROCKET_KACC_CHAIN` is opt-in and off by default. `=1` engages only at gcap>=3 and otherwise
falls back, so it never slows a matmul down. `=2` forces any `nKt` that fits. Gemma's FFN-down
(nKt=40) falls back, so the end-to-end LLM gain is ~0.

## Evidence

### Registers

The three fp16 requirements interact. If any one is wrong, `EW_SURF_STRIDE` looks inert and the
path looks impossible, so no sweep of one field at a time converges. The working geometry came
from copying Mesa's `add_tensor` geometry wholesale. With all three matching, `N=16…384` and `M`
up to 512 all pass.

### Integer and float paths

Two earlier probes of an int32 add moved `DPU_EW_CFG` and left the precision fields at fp16.
Both returned garbage, and both failed on the partial register bundle, not on the ALU:

1. **int32 integer-add via `EW_ALU_ALGO`** (`0x10C202C0`) returns garbage: the EW adds the
   int32 bit patterns as float.
2. **int32 via the `EW_OP_TYPE=1` integer path** (`0x10C203C4`) returns garbage: a true add
   of conv(226)+op(1000000) gives 1000226, and the EW returns `0x3A7C80`, an fp16 inf/NaN
   pattern in the low 16 bits. The probe used a constant operand, so the addressing is not
   the cause.

With the triple left at fp16, the int32 element sizes match no model of an add. The
whole-bundle measurement is `tests/ew_int_probe.c`, on both drivers. It fed both operands from
memory, which is why the convolution-fed K-accumulation remains not characterized.

### ki-fence chaining

`=2` is the byte-exact test's strict gcap=1 case. The test is
`tests/matmul_kacc_chain_rocket.c`, and the comparison bench is `matmul_kacc_chain_bench.c`.

The EW path's diagnostic is a standalone classifier (`matmul_accum_rocket.c` and
`matmul_accum_int8_rocket.c`) with a constant operand and a sentinel output. It separates "can
the EW add this type at all" from "is the addressing right" in a single run.
