# SDP stage precision and int32 K-accumulation

The DPU's post-CACC datapath, NVDLA's SDP, has three stages: BS (X1), BN (X2) and EW (Y). Each
stage can read an operand from memory through its own read DMA in the
[DPU_RDMA](../GLOSSARY.md). The stage combines that operand with the accumulator or the result
before write-back. The read DMA's config register sets the operand's shape, and only the EW
stage's, `RDMA_ERDMA_CFG`, addresses the operand per element. The precision fields in DPU
`0x4010` and DPU_RDMA `FEATURE_MODE_CFG` (`0x5044`) select the number format.

With every precision field at one integer code, the EW ALU computes integer ops exactly
[HW sweep, RK3588]. The CORE's accumulator does not persist across conv ops, so a
K-accumulation past one CBUF tile must use an SDP stage.

## Registers

Each SDP stage has a config register in the DPU, and reads its operand through a read DMA with
its own config register:

| Stage | NVDLA | Stage config | Operand read DMA | Operand registers |
|---|---|---|---|---|
| BS | X1 | `DPU_BS_CFG` 0x4040 (`BS_ALU_ALGO`/`BS_ALU_SRC`/bypass bits) | BRDMA | `RDMA_BRDMA_CFG` 0x501C (`BRDMA_DATA_USE[4:1]`), `RDMA_BS_BASE_ADDR` 0x5020 |
| BN | X2 | `DPU_BN_*` 0x4060 | NRDMA | `RDMA_NRDMA_CFG` 0x5028 (`NRDMA_DATA_USE[4:1]`) |
| EW | Y | `DPU_EW_CFG` 0x4070 | ERDMA | `RDMA_ERDMA_CFG` 0x5034, `EW_BASE_ADDR` 0x5038 |

## Operand shape per stage

The fields of each stage's read-DMA config register decide the operand's shape:

- **BS (X1).** `RDMA_BRDMA_CFG` contains only `BRDMA_DATA_USE[4:1]`, with no `DATA_MODE`,
  `DATA_SIZE` or `SURF_MODE` field. Data use 1 reads a per-channel `[C]` int32 bias vector and
  broadcasts it over all pixels, as the bit-exact native int8-out depthwise does. Data use 7
  reads a 64-byte group per 8 channels [HW sweep, RK1]. The group contains the int32 bias, an
  int16 per-channel CPEND operand and the int16 per-channel multiplier
  ([out-cvt-converter.md](out-cvt-converter.md) §"Per-channel (per-axis) requant on the BS
  multiplier"). The stage addresses its operand per channel only, with no per-element
  addressing.
- **BN (X2).** `RDMA_NRDMA_CFG` contains only `NRDMA_DATA_USE[4:1]`. Its operand has the BS
  shape: a per-channel broadcast, with no per-element addressing.
- **EW (Y).** `RDMA_ERDMA_CFG` is the only per-element operand register
  (`ERDMA_DATA_MODE[31:30]`, `ERDMA_SURF_MODE`, `ERDMA_DATA_SIZE[3:2]`). `DATA_SIZE` reaches
  3, a 32-bit element, wide enough for an int32 operand. The precision-safe variant of the
  fp16 K-accumulation reads that element as fp32. The EW ALU computes in integer once the whole
  precision bundle is integer (§"The EW stage in integer mode").

EW is the only per-element stage, and it adds int32 exactly. BS and BN, the two other
integer-capable stages, broadcast a `[C]` vector and cannot hold per-pixel K-partials.

## The EW stage in integer mode

In integer mode, every precision field is set to the same integer code. The EW ALU then
computes ADD, MINUS, MAX, MIN and MUL on int8, int16 and int32 elements [HW sweep, RK3588]. The
allbilly/rk3588 project measures the same on the same silicon [source-confirmed].

**Conditions:** RK3588, under `rocket` 1.3.0 and under the vendor `rknpu` 0.9.8. A DPU-only
program: the main operand on a flying MRDMA feed (`FEATURE_MODE_CFG` bit 0) and the second
operand on the ERDMA, both from memory. A 4096-element cube.

**Not characterized:** the conv's CACC result as the main feed at an integer
`proc_precision`, which an int32 K-accumulation needs. Also ABS and NEG, the broadcast operand
modes, a cube taller than one row, the RK3576, and the cost of any op.

The bundle that selects integer mode is five settings:

- `out_precision`, `in_precision` and `proc_precision` in DPU `0x4010` are all at the code
  (`0` int8, `1` int16, `4` int32).
- DPU_RDMA `FEATURE_MODE_CFG` (`0x5044`) `IN_PRECISION[17:15]` and `PROC_PRECISION[7:5]` are
  at the same code.
- `DPU_EW_CFG` `EDATA_SIZE[23:22]` and `RDMA_ERDMA_CFG` `ERDMA_DATA_SIZE[3:2]` are at the
  element width (`1` 8-bit, `2` 16-bit, `3` 32-bit).
- The EW input converter is bypassed (`DPU_EW_CFG[8]`) at int32, with
  `EW_CVT_SCALE_VALUE` 1.
- OUT_CVT is at scale 1, shift 0, with `FP32TOFP16_EN` clear.

| Op | int8 | int16 | int32 |
|---|---|---|---|
| ADD, MINUS | exact, saturating | exact, saturating | exact, saturating |
| MAX, MIN (`EW_EQUAL_EN` set) | exact | exact | exact |
| MUL (`EW_OP_TYPE` 1) | exact, saturating | exact, saturating | second operand read as signed int16 |

> [!CAUTION]
> With the int32 element sizes and the precision triple at fp16 (`2`), the output matches none
> of six models of the op [HW sweep]. A working integer ALU then looks broken.
> Software must program the whole bundle together, not `DPU_EW_CFG` alone.

## A two-buffer EW op needs no conv main feed

The same DPU-only program computes fp16 ADD, MAX, MIN, MINUS and MUL exactly on every element
[HW sweep, RK3588, both drivers]. Software must program three fields for it
[HW sweep, RK3588, vendor driver]:

| Register | Field | Requirement | With the field wrong |
|---|---|---|---|
| `DPU_DATA_CUBE_CHANNEL` (`0x403C`) | `ORIG_CHANNEL[28:16]` | Equal to `CHANNEL[12:0]` | At `ORIG_CHANNEL` 0, only lane 0 of each atom is correct: one element in 8 at fp16, one in 4 at int32 |
| `FEATURE_MODE_CFG` | `COMB_USE[10:8]` | Bit 0 clear. At `4` the op is exact | At `1` or `5`, the program never completes |
| `DPU_EW_CFG` | `[8]`, the EW input converter bypass | Op-dependent at fp16: the converter live for ADD, bypassed for MUL | ADD with the converter bypassed passes the main operand through on about half the elements. MUL with it live times out |

## Accumulator lifetime across conv ops

The CORE's accumulator, CACC, does not persist across conv ops. It accumulates
int8×int8->int32, and the fp16 and bf16 MACs, within one conv op over its full Kt depth, and it
is cleared per op. NVDLA-family hardware spills partial sums to memory through the SDP between
"hardware layers", and does not keep them in the accumulator across ops.

CORE has no cross-op "no-clear / first-continue" bit. `S_POINTER` (0x3004) is register
ping-pong banking (`POINTER_PP_EN/MODE/CLEAR`, `EXECUTER_PP_*`), and `MISC_CFG` (0x3010)
contains `PROC_PRECISION`/`DW_EN`/`QD_EN`. Software therefore cannot raise Kt past the CBUF by
streaming several ops into one accumulator. The CBUF caps Kt
(`banks_for(Mt,Kt)+banks_for(Nt,Kt) <= 12`), and a K-accumulation past that cap must use an SDP
read-modify-write stage.

An int32 K-accumulation on the EW stage in integer mode removes int8's per-K-pass readback of its
int32 partials (∝ M·N·nKt). Kt already reaches the CBUF cap (~384). Readback does not determine
prefill wall-clock time, so an int32 K-accumulation does not raise prefill throughput
([../perf/not-mac-bound.md](../perf/not-mac-bound.md)).

## The CNA DCOMP block

The CNA's weight-decompression block, DCOMP, reduces weight-DRAM bytes and zero MACs, and
applies only to pruned weights. Neither of those determines matmul throughput
([../perf/not-mac-bound.md](../perf/not-mac-bound.md)).
[cna-dcomp-weight-decompression.md](cna-dcomp-weight-decompression.md) has its registers, its
dense programming and its compressed format.

## Software support

The library's fp16 K-accumulation, `ROCKET_KACC`, programs the EW stage through its
`ew_accumulate` path. [k-accumulation.md](k-accumulation.md) has its registers and its measured
gain.

## Evidence

### Operand shape per stage

The fp16 K-accumulation corroborates that EW is the per-element stage. It reads its operand
through the ERDMA and adds on the EW stage, and it would have used BS if BS could add per
element.

### The EW stage in integer mode

The measurement is `tests/ew_int_probe.c`. It ran under `rocket` 1.3.0 and under the vendor
`rknpu` 0.9.8, with identical counts. It scores each element of a 4096-element cube against six
models:

- The saturating op, and the wrapping op
- The op through fp32 values, and the int32 bits read as fp32
- The main operand passed through, and unwritten

The int32 operands sit past 2^24 with odd low bits, so an fp32 datapath is distinguishable, and
it explains only 1286 of 4096 ADD results. int32 MUL matches the int16-operand model on all 4096
elements.

The int32 element sizes with the precision triple left at fp16 (`2`) match none of the six
models. Leaving BRDMA and NRDMA enabled changed no result.

### A two-buffer EW op needs no conv main feed

The fp16 results come from the same `tests/ew_int_probe.c`, on both drivers. Each of the three
fields was moved alone on the working program, on the vendor driver.
