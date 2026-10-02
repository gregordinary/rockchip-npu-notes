# Integer-factor resize / upsample (nearest, bilinear) on rocket

This is the FPN and decoder neck operator: it upsamples a feature map by an integer factor.
The frameworks name it TFLite `RESIZE_NEAREST_NEIGHBOR` / `RESIZE_BILINEAR`, ONNX `Resize`
and PyTorch `F.interpolate`. `librocketnpu` implements it as `rocket_upsample_nearest_fp16`
and `rocket_upsample_bilinear_fp16` (`src/rocket_resize.c`, `include/rocket_resize.h`). The
hardware gate is `tests/resize_rocket.c` (CTest `resize_rocket`).

**Established by:** a hardware run on the Turing RK1 (kernel 7.1.0-1, 600 MHz), 2026-06-22.
Nearest is bit-exact vs block-replication. Bilinear is within fp16 tolerance vs an
independent 2-tap gather. The partition-of-unity and linear-exactness properties both hold.

## Lowering to a depthwise transposed conv

The RK3588 NPU has no resize hardware (no gather or sampler block). An integer-factor
upsample is exactly a depthwise [ConvTranspose2d](conv-transpose.md) with a fixed
per-channel kernel. It scatters each input pixel onto a stride-`scale` lattice and convolves
with a small kernel. Resize therefore reuses the transposed-conv lowering (-> the forward
conv) bit-for-bit. Only the kernel differs:

| mode | 1D kernel | size | what it does |
|------|-----------|------|--------------|
| nearest | `1,1,…,1` (box) | `scale` | replicate each pixel into a `scale×scale` block |
| bilinear | triangle `1−\|i−c\|/scale`, `c=(k−1)/2` | `k = 2·scale − scale%2` | 2-tap linear interpolation |

Both modes use `pad = (k − scale)/2` and `opad = 0`, which makes the output exactly
`IH·scale × IW·scale`. The kernel size cancels in
`OH = (IH−1)·scale − 2·pad + (k−1) + 1 = IH·scale`. The 2D kernels are the separable outer
product `tri_y ⊗ tri_x`.

## The triangle kernel as bilinear interpolation

The triangle has width `2·scale`, but its stride-`scale` subsample is a partition of unity.
At every output phase, the taps landing on input-lattice positions sum to exactly 1. A
constant input therefore upsamples to that constant (interior). Only two taps are nonzero at
any output position: the two nearest input samples, weighted `(1−d)` and `d`. The geometry
gives the half-pixel map as the source coordinate:

```
src = (o + 0.5)/scale − 0.5         (== F.interpolate(..., align_corners=False))
```

The map has a zero boundary: out-of-lattice taps contribute 0, because the dilated input is
zero-padded. For `scale=2` the kernel is `[0.25, 0.75, 0.75, 0.25]`, which is exact in fp16,
so the upsample is bit-exact. For `scale=3` the kernel is `[⅓,⅔,1,⅔,⅓]`, where ⅓ rounds in
fp16, so the error is ~1e-3. These are the FCN/segmentation "bilinear-deconv" init kernels.

Framework-exact coordinate modes are a delegate-wiring concern: `align_corners=True`, clamp
vs zero boundary, and the TFLite `half_pixel_centers` flag. This primitive fixes the
half-pixel, zero-boundary convention. A clamp boundary would need the boundary output rows
patched on the host after readback, like the LeakyReLU x~0 repair.

## Constraints and cost

- **`C % 32 == 0`** is required, inherited from the depthwise forward conv's channel group
  (G=32). A feature map whose channel count is not a multiple of 32 needs host fallback (or
  a pad-channels-then-slice wrapper). FPN necks are typically 64/128/256.
- Cost scales with the upsampled size (the materialized stride dilation spends MACs on
  zeros). The sub-pixel / `scale²` decomposition (one small dense conv per phase, no
  zero-MACs) removes that cost in principle. Priced for ConvTranspose, it costs more at every
  k4 layer measured ([conv-transpose.md](conv-transpose.md)).
- Standalone, a host upsample is cheaper than the NPU round-trip. The value is keeping the
  feature cube-resident between two NPU ops (the inter-op goal).

## Validation (`tests/resize_rocket.c`)

The references are independent of the NPU path. The NPU runs a transposed-conv *scatter*
and the references run a *gather*, so a kernel or coordinate bug cannot hide. The gate
checks:

- Nearest vs block-replication (bit-exact)
- Bilinear vs the 2-tap half-pixel gather (<=0.05)
- A constant->constant partition-of-unity check (<=0.05)
- A y-ramp->half-pixel-ramp linear-exactness check (<=0.05)

All sweep cases pass.

See also: [conv-transpose.md](conv-transpose.md), [depthwise-conv.md](../depthwise-conv.md).
