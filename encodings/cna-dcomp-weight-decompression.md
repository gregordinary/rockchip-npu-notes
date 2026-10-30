# CNA DCOMP, the weight-decompression block

The RK3588 CNA has a weight-decompression block (`DCOMP`, the NVDLA CC/CDMA analog). It expands
a compressed weight stream as the CNA loads the weights. `DECOMP_CONTROL` in `DCOMP_CTRL`
(0x1100) selects the mode, and 0 is the dense pass-through. An external userspace drives DCOMP
as a lossless entropy codec over dense int8 weights [source-confirmed: `rknpu2_matmul_open`],
which is the leading description of the compressed format. NVDLA's sparse CWT/WMB/WGS format is
the alternative [expected]. Neither format is characterized on hardware in these notes.

## Support status

| Mode | Status |
|---|---|
| Dense pass-through, `DECOMP_CONTROL=0` | Supported (default). Every `librocketnpu` program runs in it |
| Compressed weights, `DCOMP_CTRL=1` | Not characterized. An external userspace drives it on the vendor `rknpu` driver [source-confirmed]. `librocketnpu` does not drive the decompressor |

## Registers

Mesa `rocket/registers.xml` and `librocketnpu`'s `npu_hw.h` give the map [source-confirmed]. The
last column gives what the external codec programs [source-confirmed: `rknpu2_matmul_open`]:

| Offset | Register | Fields | External codec |
|---|---|---|---|
| 0x1100 | `DCOMP_CTRL` | `WT_DEC_BYPASS` (bit 3), `DECOMP_CONTROL` (bits 2:0) | 1 |
| 0x1104 | `DCOMP_REGNUM` | number of compressed regions/groups in use | 0x55555555 |
| 0x1110 | `DCOMP_ADDR0` | base address of the (compressed) weight stream. Doubles as the weight base | The stream base |
| 0x1140..0x117C | `DCOMP_AMOUNT0..15` | per-group compressed byte counts (16 slots) | Each lane's stream length in 16-byte blocks, minus 1 |

Mesa does not name the register gap 0x1114-0x113C, between `ADDR0` and `AMOUNT0`. That gap is
a candidate location for the WMB/WGS surface addresses of the NVDLA format (§"NVDLA sparse
format"), unconfirmed. Mesa documents only what Teflon emits, and Teflon never compresses, so
those fields are unmapped.

## Dense (pass-through) programming

A Teflon int8 conv emits, on every tile [source-confirmed: teflon int8-conv capture]:

```
DCOMP_CTRL    = 0x0          # WT_DEC_BYPASS=0, DECOMP_CONTROL=0  -> dense pass-through
DCOMP_REGNUM  = 0x0          # 0 compressed groups
DCOMP_ADDR0   = <weight base IOVA>
DCOMP_AMOUNT0..N = 0x0       # no compressed sizes
```

`DECOMP_CONTROL=0` is the dense mode. A nonzero `DECOMP_CONTROL` selects a decompression mode,
and the external codec enables it with `DCOMP_CTRL` = 1 [source-confirmed:
`rknpu2_matmul_open`]. No compressed program has run on `rocket`.

## Compressed format

### Dense lossless codec

The external userspace compresses dense int8 weights losslessly [source-confirmed: read from
`rknpu2_matmul_open`'s source, not run on hardware here]. Its stream has this form:

- Each 1024-byte weight block, N32 × K32 in the weight tile layout, splits into 16 lanes over
  its 4-byte groups.
- Each lane's stream is 64-byte aligned and at least 128 bytes long.
- A lane's stream opens with a 16-byte header. Byte 1 is the modal value, and bytes 4-7 hold
  eight 4-bit width codes.
- Each symbol is a prefix code, then a sign bit, then the magnitude bits of its residual from
  the modal value.
- A block that does not shrink stays raw.

The codec needs no zero weights, so it applies to a dense quantized model rather than only to
a pruned one.

### NVDLA sparse format

NVDLA's compressed format is CWT/WMB/WGS [source-confirmed: nvdla.org/hw/format.html]. It is the
alternative description of the RK3588's DCOMP [expected]. It has three 128-byte-aligned
surfaces per kernel group. A group is 32 kernels for int8 and 16 for fp16 and int16, the same
sizes as the weight tile's kernel groups:

- **CWT** (compressed weight): the non-zero weight bytes only, packed compactly, zeros removed.
- **WMB** (weight mask block): 1 bit per element (1 bit per 1 byte for int8, 1 bit per
  2 bytes for fp16/int16), `1`=kept, little-endian, one WMB per kernel group.
- **WGS** (weight group size): one uint32 per group, the remaining byte count after zero
  removal. It lets the CDMA navigate variable-length compressed groups.

The CDMA reads WMB+WGS and streams only the CWT non-zeros into the MAC feed, and the mask
drives zero-skip at the MACs. Under this format, `DCOMP_AMOUNT0..15` would hold per-group WGS
values, `DCOMP_REGNUM` the group count and `DCOMP_ADDR0` the CWT base [expected]. The external
codec's per-lane lengths and its `DCOMP_REGNUM` value do not fit that reading.

**Not characterized:** either format on hardware, the meaning of `DCOMP_REGNUM`'s value, and the
codec's effect on the weight fetch rate.

## Performance

**Conditions:** RK3588, kernel 7.2.9, `rocket` 1.3.0, NPU clock 928 MHz, weights streamed
through the CBUF.

A streamed weight cube is fetched at ~5.4 GB/s a core, fp16 and int8 alike, and three cores each
keep that rate [HW sweep 2026-10-21] ([../perf/not-mac-bound.md](../perf/not-mac-bound.md)
§"Weight bandwidth a core"). The fetch rate limits decode on the NPU
([../perf/decode-gemv.md](../perf/decode-gemv.md)). A working int8 codec that shrinks the weight
bytes is therefore expected to raise the weights fetched per second by its compression ratio
[expected].

The dense codec compresses int8 language-model weights quantized per output channel by
1.11-1.12x [host-computed]. At that ratio a three-core weight stream of ~16.4 GB/s carries
~18.4 GB/s of int8 weights [expected]. Native int4 weights are half the bytes of int8.

**Conditions:** the codec's stream size, computed on the host for every matmul weight tensor of
three decoder layers spread over the model's depth. Each tensor is quantized per output channel,
symmetric, round-to-nearest, with the scale at the channel's largest magnitude over the levels
given. The Q8_0 row sizes a GGUF file's own codes, which carry one scale per 32 weights.

| Weights | Levels a side | Compression | Bits per weight |
|---|---:|---:|---:|
| Llama-3.2-3B | 127 (int8) | 1.12x | 7.15 |
| Qwen3.5-9B | 127 (int8) | 1.12x | 7.15 |
| Phi-4-mini | 127 (int8) | 1.11x | 7.20 |
| Llama-3.2-3B, Q8_0 codes | 127 | 0.93x | 8.60 |
| Llama-3.2-3B | 63 | 1.30x | 6.18 |
| Llama-3.2-3B | 31 | 1.55x | 5.18 |
| Llama-3.2-3B | 15 | 1.85x | 4.33 |

A Q8_0 block's scale puts its largest weight at the end of the range, so its codes expand. The
coarser rows are lossy quantizations of the same weights, and the codec only stores them
losslessly.

The whole-tile matmul program is bound by dispatch rather than weight bandwidth, so DCOMP is not
expected to speed it [expected].

Compressed weights are also expected to occupy less CBUF, allowing a bigger K-tile and so fewer
dispatches [expected].

## Bring-up requirements

A compressed-weight bring-up requires:

- **A reimplementation of the dense codec** from the register programming above, checked
  bit-exact against the same matmul with uncompressed weights.
- **A weight-bandwidth-bound operating point**, for the saving to convert to speed. Streamed
  weights at decode M are one.
- **For the NVDLA format only, a vendor RKNN capture with `compress_weight=True`** of a pruned
  model, and a pruning pipeline that produces weights with enough zeros to matter. RKNN's own
  lossless pruning is sparsity-gated, and many models do not prune losslessly. The dense Teflon
  capture cannot show the format's register programming or the WMB surface placement.

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

### Compressed format

The dense codec's register values and stream layout are read from `rknpu2_matmul_open`
(AGPL-3.0), which drives the vendor `rknpu` driver with its own register programs. They were
not reproduced on `rocket`. The NVDLA format is from its public documentation, and no
compressed RK3588 capture exists to compare it with.

### Performance

The compression table comes from `rocket-userspace/tools/dcomp_ratio.py`, which computes each
lane's stream size from the format above without writing the stream. Its sizes equal the
external encoder's output byte for byte on five shapes of synthetic int8 weights, K 64-4096 and
N 32-512. The table's tensors are each decoder layer's attention and FFN projections, from the
models' F16 and Q8_0 GGUF files. Over 14 FFN tensors of Llama-3.2-3B the per-tensor ratio at 127
levels is 1.10-1.13x. No compressed weight was submitted to the NPU.
