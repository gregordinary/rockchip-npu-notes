# SDP stage precision — why on-device int32 K-accumulation is impossible

**Context:** int8 matmul's dominant cost is reading the int32 partial back per K-pass
(∝ M·N·nKt) and, more precisely, the **A76 NEON de-tile *gather*** that runs once per
readback. fp16 avoids nKt gathers via on-device K-accumulation (`ROCKET_KACC`, +19%). The
question is whether int8 can do the same: accumulate int32 partials on-device so the
gather happens once.

## The accumulator does NOT carry across conv ops
The CORE/CACC accumulates int8×int8→int32 (and the fp16/bf16 MACs) **within one conv op** over
its full Kt depth. It is **cleared per op** — NVDLA-family hardware spills partial sums to memory
between "hardware layers" through the SDP, it does not keep them in the accumulator across ops.
There is **no cross-op "no-clear / first-continue" bit** in CORE: `S_POINTER` (0x3004) is
register **ping-pong banking** (`POINTER_PP_EN/MODE/CLEAR`, `EXECUTER_PP_*`), and `MISC_CFG`
(0x3010) is `PROC_PRECISION`/`DW_EN`/`QD_EN`. So "raise Kt past CBUF by streaming into one
accumulator" is not expressible — Kt is hard-capped by CBUF
(`banks_for(Mt,Kt)+banks_for(Nt,Kt) ≤ 12`).

## K-accumulation must use an SDP read-modify-write stage
The post-CACC SDP (== NVDLA SDP) has three stages, each able to read an operand from memory and
combine it with the accumulator/result before write-back:
- **BS / X1** — `DPU_BS_CFG` 0x4040 (`BS_ALU_ALGO`/`BS_ALU_SRC`/bypass bits), operand via
  **BRDMA** (`RDMA_BRDMA_CFG` 0x501C `BRDMA_DATA_USE[1:4]`, `RDMA_BS_BASE_ADDR` 0x5020).
- **BN / X2** — `DPU_BN_*` 0x4060.
- **EW / Y** — `DPU_EW_CFG` 0x4070, operand via **ERDMA** (`RDMA_ERDMA_CFG` 0x5034,
  `EW_BASE_ADDR` 0x5038). This is the existing `ew_accumulate` path.

## Per-stage precision — no per-element int32 add exists
Each SDP stage has a separate RDMA-config register; their *fields* decide the operand shape:
- **BS/X1 — `RDMA_BRDMA_CFG` (0x501C): only `BRDMA_DATA_USE[1:4]`.** No `DATA_MODE`, no
  `DATA_SIZE`, no `SURF_MODE`. It reads a **per-channel `[C]` int32 bias vector** and broadcasts
  it over all pixels — confirmed by `rocket_conv.c:1440` (`bs_bo` filled `dst[c]` for `c<C`) and
  the bit-exact native int8-out conv (`npu_regcmd.c:2879`, `bias_en`). **int32, but per-channel
  only — no per-element addressing.**
- **BN/X2 — `RDMA_NRDMA_CFG` (0x5028): only `NRDMA_DATA_USE[1:4]`.** Identical shape to BS —
  per-channel broadcast, no per-element.
- **EW/Y — `RDMA_ERDMA_CFG` (0x5034): the ONLY per-element stage** (`ERDMA_DATA_MODE[30:31]`,
  `ERDMA_SURF_MODE`, `ERDMA_DATA_SIZE[2:3]`). `DATA_SIZE` even reaches **3 = 32-bit** (used as
  **fp32** for the precision-safe fp16 KACC variant) — so the bit-WIDTH isn't the limit. The
  limit is that the **EW ALU is FLOAT-only**: it does fp16/fp32 per-element adds but never an
  integer add — feeding int32 bit-patterns adds them *as float* = garbage (`k-accumulation.md`
  §"Integer EW K-accumulation — DEAD"; `EW_ALU_ALGO` int32-add `0x10C202C0` → garbage, HW-tested).

**Conclusion — on-device int32 per-element K-accumulation is a HARD HARDWARE CEILING.** The two
integer-capable stages (BS, BN) can only *broadcast a `[C]` vector* (useless for per-pixel
K-partials); the only *per-element* stage (EW) has a **float-only ALU** (fp16/fp32 per-element adds
only, never integer). There is **no** SDP path that adds a per-element *integer* tensor. Decisive
corroboration: the fp16 KACC win uses the **ERDMA/EW** path (because EW is *the* per-element stage)
— it would have used BS if BS could do per-element adds.

**Consequence:** int8's int32 readback floor (∝ M·N·nKt, gather-bound per K-pass) cannot
be reduced by on-device accumulation, and Kt is already CBUF-maxed (~384). So **int8/int4 on the
rocket path is a RAM play (resident weights — measured 0.60× fp16), NOT a prefill-speed
play.** No probe was needed: the register architecture is conclusive (and a "negative" per-element
probe would be weak evidence vs. the structural fact that the per-element CFG field simply does not
exist on the int32-capable stages).

## Cross-finding: the CNA DCOMP block exists on RK3588
While here: Mesa `registers.xml` shows a real **CNA DCOMP block** — `DCOMP_CTRL` 0x1100,
`DCOMP_REGNUM` 0x1104, `DCOMP_ADDR0` 0x1110, `DCOMP_AMOUNT0..15` 0x1140–0x117C. Now fully
decoded: the compressed format **is** NVDLA CWT/WMB/WGS [source-confirmed], Teflon's dense
programming is `DECOMP_CONTROL=0` pass-through, and it is **deprioritized** — it reduces
weight-DRAM bytes + zero MACs, neither of which binds the matmul ([../perf/not-mac-bound.md](../perf/not-mac-bound.md)),
and it only applies to pruned weights. Full decode + the why:
[cna-dcomp-weight-decompression.md](cna-dcomp-weight-decompression.md).
