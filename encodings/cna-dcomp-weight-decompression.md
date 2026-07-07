# CNA DCOMP — the weight-decompression block (decoded; a non-bottleneck here)

The RK3588 CNA has a real **weight-decompression block** (`DCOMP`, the NVDLA **CC/CDMA**
analog) — it expands a *sparse* compressed weight stream into the MAC feed, skipping zero
weights. The register map is fully decoded and the compressed format is the NVDLA
CWT/WMB/WGS format [source-confirmed]. It is **deprioritized**: it reduces weight-DRAM bytes
and zero MACs, and [the matmul is bound by neither](../perf/not-mac-bound.md) — so it cannot
speed up prefill at the current operating point, and it only applies to *pruned* models we
don't build. Decoded here so the next person doesn't re-RE it or chase it as a speed lever.

## Register map [source-confirmed: Mesa `rocket/registers.xml`, our `npu_hw.h`]

| reg | offset | fields |
|---|---|---|
| `DCOMP_CTRL` | 0x1100 | `WT_DEC_BYPASS` (bit 3), `DECOMP_CONTROL` (bits 2:0) |
| `DCOMP_REGNUM` | 0x1104 | number of compressed regions/groups in use |
| `DCOMP_ADDR0` | 0x1110 | base address of the (compressed) weight stream — doubles as the weight base |
| `DCOMP_AMOUNT0..15` | 0x1140..0x117C | per-group compressed byte counts (16 slots) |

There is a register gap 0x1114–0x113C (between `ADDR0` and `AMOUNT0`) that Mesa does not
name; the WMB/WGS surface addresses (see format below) may live there. Mesa documents only
what Teflon emits, and **Teflon never compresses**, so those fields are unmapped.

## Dense (pass-through) programming [source-confirmed: teflon int8-conv capture]

A real Teflon int8 conv emits, every tile:

```
DCOMP_CTRL    = 0x0          # WT_DEC_BYPASS=0, DECOMP_CONTROL=0  -> dense pass-through
DCOMP_REGNUM  = 0x0          # 0 compressed groups
DCOMP_ADDR0   = <weight base IOVA>
DCOMP_AMOUNT0..N = 0x0       # no compressed sizes
```

So **`DECOMP_CONTROL=0` is the dense mode** the whole existing stack runs in (our bit-exact
matmul/conv gates implicitly validate it). A nonzero `DECOMP_CONTROL` selects a decompression
mode; the exact enabling value is not in any capture we have (would need a vendor compressed
capture to confirm — see below). Our `gen_*` set the same dense values; nothing in the FOSS
stack drives the decompressor.

## The compressed format = NVDLA CWT/WMB/WGS [source-confirmed: nvdla.org/hw/format.html]

Three 128-byte-aligned surfaces, per **kernel group** (int8 = **32** kernels/group, fp16/int16
= **16**/group — exactly our weight-tile group sizes):

- **CWT** (compressed weight): the non-zero weight bytes only, packed compactly, zeros removed.
- **WMB** (weight mask block): **1 bit per element** (int8: 1 bit ↔ 1 byte; fp16/int16: 1 bit ↔
  2 bytes), `1`=kept, **little-endian**, one WMB per kernel group.
- **WGS** (weight group size): one **uint32 per group** = the remaining byte count after zero
  removal. Lets the CDMA navigate variable-length compressed groups.

The CDMA reads WMB+WGS, streams only the CWT non-zeros into the MAC feed, and the mask drives
zero-skip at the MACs. The RK3588 `DCOMP_AMOUNT0..15` very likely **are** the per-group WGS
values inlined into registers (16 groups), with `DCOMP_REGNUM` = the group count and
`DCOMP_ADDR0` the CWT/blob base — but the WMB placement and the `DECOMP_CONTROL` enable value
are **inferred, not confirmed** (no compressed capture to check against).

## Why it cannot pay here (the decisive point)

DCOMP buys two things, and [not-mac-bound.md](../perf/not-mac-bound.md) shows the matmul is
bound by **neither** at the current operating point (resident, multicore, 600 MHz):

1. **Fewer weight-DRAM bytes** (sparsity). But the floor is **not** weight-DMA bandwidth:
   int4 already reads **¼** the weight bytes of fp16 and got **no** speedup — the floor is
   latency-like (dispatch / fence / CBUF-fill), dtype- and weight-size-independent.
2. **Skipped zero MACs.** But the NPU runs at **~15%** of fp16 MAC peak — the MAC array is
   already mostly idle; removing MACs removes idle work.

So a *working* DCOMP would land on the same dtype-independent floor as int4/int8 — a footprint
play, not a speed play — and it is doubly gated: it pays **only on pruned weights** (dense
quantized weights have ~no zeros → nothing to compress), and only at a future operating point
where weight bandwidth or MAC count actually binds. The one indirect angle —
sparse weights occupy **less CBUF**, which *could* allow a bigger K-tile and thus fewer
dispatches (the real floor) — still requires sparse models and a pruning pipeline this stack
does not have.

## What a bring-up would require (bounded; not pursued)

- A **vendor RKNN capture with `compress_weight=True`** (sparse inference) of a pruned
  model, then RE its weight BO to confirm the `DCOMP_ADDR0`/`AMOUNT`/`REGNUM`/`DECOMP_CONTROL`
  programming + the WMB surface placement (the dense Teflon capture can't show this).
- A **sparsification / pruning pipeline** to produce weights with enough zeros to matter
  (RKNN's own lossless-pruning is sparsity-gated; many models don't prune losslessly).
- A **weight-bandwidth-bound operating point** for it to convert to speed (absent today; the
  dispatch floor binds first).

All three are absent, so DCOMP stays decoded-but-unbuilt. Guessing the compressed register
programming on HW risks an IOMMU-fault wedge (recoverable via `rmmod rocket`) for a lever that
cannot move the current bottleneck even if it worked.
