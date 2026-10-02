# CNA DCOMP, the weight-decompression block

The RK3588 CNA has a weight-decompression block (`DCOMP`, the NVDLA CC/CDMA analog). It
expands a sparse compressed weight stream into the MAC feed, skipping zero weights. The
register map is fully decoded, and the compressed format is the NVDLA CWT/WMB/WGS format
[source-confirmed].

The block is deprioritized. It reduces weight-DRAM bytes and zero MACs, and the matmul is
bound by neither ([not-mac-bound.md](../perf/not-mac-bound.md)). So it cannot speed up
prefill at the current operating point. It also applies only to pruned models, which this
project does not build. The decode is recorded here so that the next person does not
reverse-engineer it again or pursue it as a speed lever.

## Register map

Mesa `rocket/registers.xml` and this project's `npu_hw.h` give the map [source-confirmed]:

| reg | offset | fields |
|---|---|---|
| `DCOMP_CTRL` | 0x1100 | `WT_DEC_BYPASS` (bit 3), `DECOMP_CONTROL` (bits 2:0) |
| `DCOMP_REGNUM` | 0x1104 | number of compressed regions/groups in use |
| `DCOMP_ADDR0` | 0x1110 | base address of the (compressed) weight stream. Doubles as the weight base |
| `DCOMP_AMOUNT0..15` | 0x1140..0x117C | per-group compressed byte counts (16 slots) |

Mesa does not name the register gap 0x1114-0x113C (between `ADDR0` and `AMOUNT0`). That
gap is a candidate location for the WMB/WGS surface addresses (see the format below),
unconfirmed. Mesa documents only what Teflon emits, and Teflon never compresses, so those
fields are unmapped.

## Dense (pass-through) programming

A Teflon int8 conv emits, on every tile [source-confirmed: teflon int8-conv capture]:

```
DCOMP_CTRL    = 0x0          # WT_DEC_BYPASS=0, DECOMP_CONTROL=0  -> dense pass-through
DCOMP_REGNUM  = 0x0          # 0 compressed groups
DCOMP_ADDR0   = <weight base IOVA>
DCOMP_AMOUNT0..N = 0x0       # no compressed sizes
```

So `DECOMP_CONTROL=0` is the dense mode, which the whole existing stack runs in. This
project's bit-exact matmul and conv gates validate it implicitly. A nonzero
`DECOMP_CONTROL` selects a decompression mode. The exact enabling value is in no capture
this project holds, and confirming it needs a vendor compressed capture (see below). This
project's `gen_*` generators set the same dense values. Nothing in the FOSS stack drives
the decompressor.

## Compressed format

The compressed format is NVDLA CWT/WMB/WGS [source-confirmed: nvdla.org/hw/format.html].
It has three 128-byte-aligned surfaces per kernel group. A group is 32 kernels for int8 and
16 for fp16 and int16, exactly the weight-tile group sizes this project uses:

- **CWT** (compressed weight): the non-zero weight bytes only, packed compactly, zeros removed.
- **WMB** (weight mask block): 1 bit per element (1 bit per 1 byte for int8, 1 bit per
  2 bytes for fp16/int16), `1`=kept, little-endian, one WMB per kernel group.
- **WGS** (weight group size): one uint32 per group, the remaining byte count after zero
  removal. It lets the CDMA navigate variable-length compressed groups.

The CDMA reads WMB+WGS and streams only the CWT non-zeros into the MAC feed, and the mask
drives zero-skip at the MACs. The RK3588 `DCOMP_AMOUNT0..15` are very likely the per-group
WGS values inlined into registers (16 groups). `DCOMP_REGNUM` is then the group count and
`DCOMP_ADDR0` the CWT/blob base. The WMB placement and the `DECOMP_CONTROL` enable value
are inferred, not confirmed: there is no compressed capture to check against.

## Payoff at the current operating point

DCOMP cannot pay at the current operating point (resident, multicore, 600 MHz). It buys two
things, and [not-mac-bound.md](../perf/not-mac-bound.md) shows the matmul is bound by
neither:

1. **Fewer weight-DRAM bytes** (sparsity). The floor is not weight-DMA bandwidth: int4
   already reads ¼ the weight bytes of fp16 and got no speedup. The floor is latency-like
   (dispatch, fence, CBUF-fill), and independent of datatype and weight size.
2. **Skipped zero MACs.** The NPU runs at ~15% of fp16 MAC peak, so the MAC array is
   already mostly idle. Removing MACs removes idle work.

So a working DCOMP would land on the same datatype-independent floor as int4 and int8. It
would buy footprint rather than speed. It is also doubly gated. It pays only on pruned
weights, because dense quantized weights have ~no zeros and so nothing to compress. And it
pays only at a future operating point where weight bandwidth or MAC count binds.

One indirect angle remains: sparse weights occupy less CBUF, which could allow a bigger
K-tile and thus fewer dispatches (the real floor). That angle still requires sparse models
and a pruning pipeline that this stack does not have.

## Bring-up requirements

A bring-up is bounded, and this project has not pursued it. It would require:

- **A vendor RKNN capture with `compress_weight=True`** (sparse inference) of a pruned
  model. Reverse-engineering its weight BO would confirm the
  `DCOMP_ADDR0`/`AMOUNT`/`REGNUM`/`DECOMP_CONTROL` programming and the WMB surface
  placement. The dense Teflon capture cannot show either.
- **A sparsification or pruning pipeline** that produces weights with enough zeros to
  matter. RKNN's own lossless pruning is sparsity-gated, and many models do not prune
  losslessly.
- **A weight-bandwidth-bound operating point**, for the saving to convert to speed. None
  exists today: the dispatch floor binds first.

All three are absent, so DCOMP stays decoded but unbuilt. Guessing the compressed register
programming on hardware risks an IOMMU-fault wedge (recoverable with `rmmod rocket`), for
a lever that cannot move the current bottleneck even if it worked.
