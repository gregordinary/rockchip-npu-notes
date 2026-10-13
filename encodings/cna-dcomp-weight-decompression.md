# CNA DCOMP, the weight-decompression block

The RK3588 CNA has a weight-decompression block (`DCOMP`, the NVDLA CC/CDMA analog). It
expands a sparse compressed weight stream into the MAC feed, skipping zero weights, and applies
only to pruned models. `DECOMP_CONTROL` in `DCOMP_CTRL` (0x1100) selects the mode, and 0 is the
dense pass-through. The register map is decoded for the dense mode only. The compressed
format is expected to be NVDLA's CWT/WMB/WGS format [expected], and no compressed RK3588
capture exists to confirm it.

## Support status

| Mode | Status |
|---|---|
| Dense pass-through, `DECOMP_CONTROL=0` | Supported (default). Every `librocketnpu` program runs in it |
| Compressed weights, nonzero `DECOMP_CONTROL` | Not supported. `librocketnpu` does not drive the decompressor. The enabling value and the WMB placement are not characterized |

## Registers

Mesa `rocket/registers.xml` and `librocketnpu`'s `npu_hw.h` give the map [source-confirmed]:

| Offset | Register | Fields |
|---|---|---|
| 0x1100 | `DCOMP_CTRL` | `WT_DEC_BYPASS` (bit 3), `DECOMP_CONTROL` (bits 2:0) |
| 0x1104 | `DCOMP_REGNUM` | number of compressed regions/groups in use |
| 0x1110 | `DCOMP_ADDR0` | base address of the (compressed) weight stream. Doubles as the weight base |
| 0x1140..0x117C | `DCOMP_AMOUNT0..15` | per-group compressed byte counts (16 slots) |

Mesa does not name the register gap 0x1114-0x113C, between `ADDR0` and `AMOUNT0`. That gap is
a candidate location for the WMB/WGS surface addresses (§"Compressed format"), unconfirmed.
Mesa documents only what Teflon emits, and Teflon never compresses, so those fields are
unmapped.

## Dense (pass-through) programming

A Teflon int8 conv emits, on every tile [source-confirmed: teflon int8-conv capture]:

```
DCOMP_CTRL    = 0x0          # WT_DEC_BYPASS=0, DECOMP_CONTROL=0  -> dense pass-through
DCOMP_REGNUM  = 0x0          # 0 compressed groups
DCOMP_ADDR0   = <weight base IOVA>
DCOMP_AMOUNT0..N = 0x0       # no compressed sizes
```

`DECOMP_CONTROL=0` is the dense mode. A nonzero `DECOMP_CONTROL` selects a decompression mode.
The exact enabling value appears in no available capture, and confirming it needs a vendor
compressed capture (§"Bring-up requirements").

## Compressed format

NVDLA's compressed format is CWT/WMB/WGS [source-confirmed: nvdla.org/hw/format.html], and the
RK3588's DCOMP is expected to use it [expected].
It has three 128-byte-aligned surfaces per kernel group. A group is 32 kernels for int8 and
16 for fp16 and int16, the same sizes as the weight tile's kernel groups:

- **CWT** (compressed weight): the non-zero weight bytes only, packed compactly, zeros removed.
- **WMB** (weight mask block): 1 bit per element (1 bit per 1 byte for int8, 1 bit per
  2 bytes for fp16/int16), `1`=kept, little-endian, one WMB per kernel group.
- **WGS** (weight group size): one uint32 per group, the remaining byte count after zero
  removal. It lets the CDMA navigate variable-length compressed groups.

The CDMA reads WMB+WGS and streams only the CWT non-zeros into the MAC feed, and the mask
drives zero-skip at the MACs. The RK3588 `DCOMP_AMOUNT0..15` are very likely the per-group
WGS values inlined into registers (16 groups). `DCOMP_REGNUM` is then the group count and
`DCOMP_ADDR0` the CWT/blob base.

**Not characterized:** the WMB placement and the `DECOMP_CONTROL` enable value, which are
inferred. No compressed capture exists to check them against.

## Performance

**Conditions:** RK3588, resident weights, multicore, NPU clock 600 MHz.

DCOMP reduces two costs: the weight-DRAM bytes, through sparsity, and the zero MACs, which it
skips. The matmul is bound by neither ([not-mac-bound.md](../perf/not-mac-bound.md)), so DCOMP
cannot speed up prefill at this operating point. A working DCOMP reduces footprint rather than
raising throughput.

DCOMP gives a gain only on pruned weights, because dense quantized weights have ~no zeros to
compress. It gives a speed gain only at an operating point where weight bandwidth or MAC count
binds.

Sparse weights also occupy less CBUF, which can allow a bigger K-tile and so fewer dispatches
[expected]. Dispatch is the term that binds. That gain also requires sparse models and a
pruning pipeline.

## Bring-up requirements

A compressed-weight bring-up requires:

- **A vendor RKNN capture with `compress_weight=True`** (sparse inference) of a pruned
  model. Reverse-engineering its weight BO can confirm the
  `DCOMP_ADDR0`/`AMOUNT`/`REGNUM`/`DECOMP_CONTROL` programming and the WMB surface
  placement. The dense Teflon capture cannot show either.
- **A sparsification or pruning pipeline** that produces weights with enough zeros to
  matter. RKNN's own lossless pruning is sparsity-gated, and many models do not prune
  losslessly.
- **A weight-bandwidth-bound operating point**, for the saving to convert to speed. None
  exists: the dispatch limit binds first.

All three are absent, so DCOMP is decoded and not built.

> [!CAUTION]
> Guessing the compressed register programming on hardware risks an IOMMU-fault wedge,
> recoverable with `rmmod rocket`.

## Software support

`librocketnpu`'s `gen_*` generators set the dense values above. Nothing in the FOSS stack
drives the decompressor, and the stack has no pruning pipeline.

## Evidence

### Dense (pass-through) programming

Every program in the stack runs in the dense mode, and `librocketnpu`'s bit-exact matmul and
conv tests validate it implicitly.
