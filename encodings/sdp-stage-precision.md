# SDP stage precision and int32 K-accumulation

The int8 matmul's dominant cost is reading the int32 partial back per K-pass (∝ M·N·nKt).
More precisely, it is the A76 NEON de-tile gather that runs once per readback. fp16 avoids
nKt gathers through on-device K-accumulation (`ROCKET_KACC`, +19%). This note covers
whether int8 can do the same: accumulate int32 partials on-device so the gather happens
once.

## Accumulator lifetime across conv ops

The CORE/CACC accumulator does not carry across conv ops. It accumulates int8×int8->int32
(and the fp16/bf16 MACs) within one conv op over its full Kt depth, and it is cleared per
op. NVDLA-family hardware spills partial sums to memory between "hardware layers" through
the SDP, and does not keep them in the accumulator across ops.

CORE has no cross-op "no-clear / first-continue" bit. `S_POINTER` (0x3004) is register
ping-pong banking (`POINTER_PP_EN/MODE/CLEAR`, `EXECUTER_PP_*`), and `MISC_CFG` (0x3010) is
`PROC_PRECISION`/`DW_EN`/`QD_EN`. So "raise Kt past CBUF by streaming into one accumulator"
is not expressible. CBUF hard-caps Kt (`banks_for(Mt,Kt)+banks_for(Nt,Kt) <= 12`).

## SDP read-modify-write stages

K-accumulation must use an SDP read-modify-write stage. The post-CACC SDP (== NVDLA SDP)
has three stages. Each can read an operand from memory and combine it with the accumulator
or the result before write-back:

- **BS / X1**: `DPU_BS_CFG` 0x4040 (`BS_ALU_ALGO`/`BS_ALU_SRC`/bypass bits), operand through
  BRDMA (`RDMA_BRDMA_CFG` 0x501C `BRDMA_DATA_USE[4:1]`, `RDMA_BS_BASE_ADDR` 0x5020).
- **BN / X2**: `DPU_BN_*` 0x4060.
- **EW / Y**: `DPU_EW_CFG` 0x4070, operand through ERDMA (`RDMA_ERDMA_CFG` 0x5034,
  `EW_BASE_ADDR` 0x5038). This is the existing `ew_accumulate` path.

## Per-stage precision

Each SDP stage has a separate RDMA-config register, and its fields decide the operand
shape:

- **BS/X1, `RDMA_BRDMA_CFG` (0x501C)** has only `BRDMA_DATA_USE[4:1]`: no `DATA_MODE`, no
  `DATA_SIZE`, no `SURF_MODE`. Data use 1 reads a per-channel `[C]` int32 bias vector and
  broadcasts it over all pixels, the bit-exact native int8-out depthwise. Data use 7 reads a
  64-byte group per 8 channels [HW sweep, RK1]. The group holds the int32 bias, an int16
  per-channel CPEND operand and the int16 per-channel multiplier
  ([out-cvt-converter.md](out-cvt-converter.md) §"Per-channel (per-axis) requant on the BS
  multiplier"). The stage is per-channel only, with no per-element addressing.
- **BN/X2, `RDMA_NRDMA_CFG` (0x5028)** has only `NRDMA_DATA_USE[4:1]`. Its shape is
  identical to BS: per-channel broadcast, with no per-element addressing.
- **EW/Y, `RDMA_ERDMA_CFG` (0x5034)** is the only per-element stage
  (`ERDMA_DATA_MODE[31:30]`, `ERDMA_SURF_MODE`, `ERDMA_DATA_SIZE[3:2]`). `DATA_SIZE` even
  reaches 3 = 32-bit (used as fp32 for the precision-safe fp16 K-accumulation variant), so
  the bit width is not the limit. Its ALU computes in integer once the whole precision
  bundle says integer (§"The EW stage in integer mode" below).

The EW stage is the only per-element stage, and it adds int32 exactly. The two other
integer-capable stages (BS, BN) can only broadcast a `[C]` vector, which is useless for
per-pixel K-partials. The fp16 K-accumulation win corroborates that EW is the per-element
stage. It uses the ERDMA/EW path, and it would have used BS if BS could do per-element
adds.

Reducing int8's int32 readback floor (∝ M·N·nKt, gather-bound per K-pass) on the device
goes through the EW stage in integer mode, and Kt is already CBUF-maxed (~384). It would
not raise prefill speed, because prefill is not readback-bound
([../perf/not-mac-bound.md](../perf/not-mac-bound.md)). So int8 and int4 on the rocket path
buy RAM (resident weights, measured 0.60x fp16), not prefill speed.

## The EW stage in integer mode

With every precision field set to the same integer code, the EW ALU computes integer ops
exactly. That was measured under `rocket` 1.3.0 and under the vendor `rknpu` 0.9.8, with
identical counts [HW sweep, RK3588, `tests/ew_int_probe.c`]. This agrees with allbilly/rk3588's
measurement on the same silicon [source-confirmed]. The bundle is five settings:

- `out_precision`, `in_precision` and `proc_precision` in DPU `0x4010` are all at the code
  (`0` int8, `1` int16, `4` int32).
- DPU_RDMA `FEATURE_MODE_CFG` (`0x5044`) `IN_PRECISION[17:15]` and `PROC_PRECISION[7:5]` are
  at the same code.
- `DPU_EW_CFG` `EDATA_SIZE[23:22]` and `RDMA_ERDMA_CFG` `ERDMA_DATA_SIZE[3:2]` are at the
  element width (`1` 8-bit, `2` 16-bit, `3` 32-bit).
- The EW input converter is bypassed (`DPU_EW_CFG[8]`) at int32, with `EW_CVT_SCALE_VALUE`
  1.
- OUT_CVT is at scale 1, shift 0, with `FP32TOFP16_EN` clear.

| Op | int8 | int16 | int32 |
|---|---|---|---|
| ADD, MINUS | exact, saturating | exact, saturating | exact, saturating |
| MAX, MIN (`EW_EQUAL_EN` set) | exact | exact | exact |
| MUL (`EW_OP_TYPE` 1) | exact, saturating | exact, saturating | second operand read as signed int16 |

The probe scores each element of a 4096-element cube against six models:

- The saturating op, and the wrapping op
- The op through fp32 values, and the int32 bits read as fp32
- The main operand passed through, and unwritten

The int32 operands sit past 2^24 with odd low bits, so an fp32 datapath is
distinguishable, and it explains only 1286 of 4096 ADD results. int32 MUL matches the
int16-operand model on all 4096 elements.

The int32 element sizes with the precision triple left at fp16 (`2`) match none of the six
models. So a probe that moves `DPU_EW_CFG` alone reads an integer ALU as broken.

The program was DPU-only: a flying MRDMA main feed (`FEATURE_MODE_CFG` bit 0) and the operand on
the ERDMA, both from memory. Leaving BRDMA and NRDMA enabled changed no result.

The measurement does not show the conv's CACC result as the main feed at an integer
`proc_precision`, which an int32 K-accumulation needs. It also does not show:

- ABS and NEG
- The broadcast operand modes
- A cube taller than one row
- The RK3576
- Any cost

## A two-buffer EW op needs no conv main feed

The same DPU-only program computes fp16 ADD, MAX, MIN, MINUS and MUL exactly on every element
[HW sweep, RK3588, both drivers, `tests/ew_int_probe.c`]. Three fields separate it from a
program that fails, each moved alone on this program [HW sweep, RK3588, vendor driver]:

- `DPU_DATA_CUBE_CHANNEL` (`0x403C`) needs `ORIG_CHANNEL[28:16]` equal to `CHANNEL[12:0]`. At
  `ORIG_CHANNEL` 0 only lane 0 of each atom is right: one element in 8 at fp16, one in 4 at
  int32.
- `FEATURE_MODE_CFG` `COMB_USE[10:8]` needs bit 0 clear. At `1` or `5` the program never
  completes, and at `4` it is exact.
- `DPU_EW_CFG[8]`, the EW input converter bypass, is op-dependent at fp16. ADD needs the
  converter live, and with it bypassed passes the main operand through on about half the
  elements. MUL needs it bypassed, and times out with it live.

## The CNA DCOMP block

The CNA DCOMP block exists on the RK3588. Mesa `registers.xml` shows it: `DCOMP_CTRL`
0x1100, `DCOMP_REGNUM` 0x1104, `DCOMP_ADDR0` 0x1110, `DCOMP_AMOUNT0..15` 0x1140-0x117C. It
is decoded for the dense mode: the compressed format is NVDLA CWT/WMB/WGS [source-confirmed],
and Teflon's dense programming is `DECOMP_CONTROL=0` pass-through. It is deprioritized. It
reduces weight-DRAM bytes and zero MACs, neither of which binds the matmul
([../perf/not-mac-bound.md](../perf/not-mac-bound.md)), and it applies only to pruned weights.
The full decode and the reasoning are in
[cna-dcomp-weight-decompression.md](cna-dcomp-weight-decompression.md).
