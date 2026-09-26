# SDP stage precision and int32 K-accumulation

**Context:** int8 matmul's dominant cost is reading the int32 partial back per K-pass
(∝ M·N·nKt). More precisely, it is the **A76 NEON de-tile *gather*** that runs once per
readback. fp16 avoids nKt gathers via on-device K-accumulation (`ROCKET_KACC`, +19%). The
question is whether int8 can do the same: accumulate int32 partials on-device so the
gather happens once.

## The accumulator does not carry across conv ops
The CORE/CACC accumulates int8×int8->int32 (and the fp16/bf16 MACs) **within one conv op** over
its full Kt depth. It is **cleared per op**. NVDLA-family hardware spills partial sums to memory
between "hardware layers" through the SDP, and does not keep them in the accumulator across ops.
There is **no cross-op "no-clear / first-continue" bit** in CORE: `S_POINTER` (0x3004) is
register **ping-pong banking** (`POINTER_PP_EN/MODE/CLEAR`, `EXECUTER_PP_*`), and `MISC_CFG`
(0x3010) is `PROC_PRECISION`/`DW_EN`/`QD_EN`. So "raise Kt past CBUF by streaming into one
accumulator" is not expressible. Kt is hard-capped by CBUF
(`banks_for(Mt,Kt)+banks_for(Nt,Kt) <= 12`).

## K-accumulation must use an SDP read-modify-write stage
The post-CACC SDP (== NVDLA SDP) has three stages, each able to read an operand from memory and
combine it with the accumulator/result before write-back:
- **BS / X1**: `DPU_BS_CFG` 0x4040 (`BS_ALU_ALGO`/`BS_ALU_SRC`/bypass bits), operand via
  **BRDMA** (`RDMA_BRDMA_CFG` 0x501C `BRDMA_DATA_USE[1:4]`, `RDMA_BS_BASE_ADDR` 0x5020).
- **BN / X2**: `DPU_BN_*` 0x4060.
- **EW / Y**: `DPU_EW_CFG` 0x4070, operand via **ERDMA** (`RDMA_ERDMA_CFG` 0x5034,
  `EW_BASE_ADDR` 0x5038). This is the existing `ew_accumulate` path.

## Per-stage precision
Each SDP stage has a separate RDMA-config register, and their *fields* decide the operand shape:
- **BS/X1, `RDMA_BRDMA_CFG` (0x501C): only `BRDMA_DATA_USE[1:4]`.** No `DATA_MODE`, no
  `DATA_SIZE`, no `SURF_MODE`. It reads a **per-channel `[C]` int32 bias vector** and broadcasts
  it over all pixels, confirmed by `rocket_conv.c:1440` (`bs_bo` filled `dst[c]` for `c<C`) and
  the bit-exact native int8-out conv (`npu_regcmd.c:2879`, `bias_en`). **int32, but per-channel
  only, with no per-element addressing.**
- **BN/X2, `RDMA_NRDMA_CFG` (0x5028): only `NRDMA_DATA_USE[1:4]`.** Identical shape to BS:
  per-channel broadcast, no per-element.
- **EW/Y, `RDMA_ERDMA_CFG` (0x5034): the only per-element stage** (`ERDMA_DATA_MODE[30:31]`,
  `ERDMA_SURF_MODE`, `ERDMA_DATA_SIZE[2:3]`). `DATA_SIZE` even reaches **3 = 32-bit** (used as
  **fp32** for the precision-safe fp16 KACC variant), so the bit width is not the limit. Whether
  its ALU adds integers is unestablished. Feeding int32 bit patterns with the precision still at
  fp16 adds them *as float*, which is garbage (`EW_ALU_ALGO` int32-add `0x10C202C0`, HW-tested).
  That is what an integer operand gives in a stage at `proc_precision = 2`, and DPU `0x4010`
  enumerates Integer 32bit as `3'd4` [TRM]. The whole-bundle test is in `k-accumulation.md`
  §"Integer EW K-accumulation".

**Conclusion: only the EW stage could add a per-element int32 tensor, and whether it can is
unestablished.** The two integer-capable stages (BS, BN) can only *broadcast a `[C]` vector*,
useless for per-pixel K-partials. The only *per-element* stage (EW) is untested with the whole
precision bundle. The fp16 KACC win corroborates that EW is *the* per-element stage. It uses the
**ERDMA/EW** path, and it would have used BS if BS could do per-element adds.

**Consequence:** reducing int8's int32 readback floor (∝ M·N·nKt, gather-bound per K-pass) on
the device needs the EW stage in integer mode, and Kt is already CBUF-maxed (~384). It would not
raise prefill speed either way, because prefill is not readback-bound
([../perf/not-mac-bound.md](../perf/not-mac-bound.md)). So **int8/int4 on the rocket path is a RAM
play (resident weights, measured 0.60x fp16), not a prefill-speed play.** The register fields
settle BS and BN, since neither has a per-element mode. They do not settle EW, whose integer mode
needs the precision triple set with it.

## Cross-finding: the CNA DCOMP block exists on RK3588
While here: Mesa `registers.xml` shows a real **CNA DCOMP block**: `DCOMP_CTRL` 0x1100,
`DCOMP_REGNUM` 0x1104, `DCOMP_ADDR0` 0x1110, `DCOMP_AMOUNT0..15` 0x1140-0x117C. It is fully
decoded: the compressed format **is** NVDLA CWT/WMB/WGS [source-confirmed], and Teflon's dense
programming is `DECOMP_CONTROL=0` pass-through. It is **deprioritized**. It reduces weight-DRAM
bytes and zero MACs, neither of which binds the matmul
([../perf/not-mac-bound.md](../perf/not-mac-bound.md)), and it only applies to pruned weights. Full decode + the why:
[cna-dcomp-weight-decompression.md](cna-dcomp-weight-decompression.md).
