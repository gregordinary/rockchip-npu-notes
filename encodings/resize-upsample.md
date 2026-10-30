# Integer-factor resize / upsample (nearest, bilinear) on rocket

Resize is the FPN and decoder neck operator: it upsamples a feature map by an integer factor.
The frameworks name it TFLite `RESIZE_NEAREST_NEIGHBOR` / `RESIZE_BILINEAR`, ONNX `Resize`
and PyTorch `F.interpolate`. `librocketnpu` implements it as `rocket_upsample_nearest_fp16`
and `rocket_upsample_bilinear_fp16` (`src/rocket_resize.c`, `include/rocket_resize.h`).

## Constraints

- **`C % 32 == 0`** is required, inherited from the depthwise forward conv's channel group
  (G=32). A feature map whose channel count is not a multiple of 32 needs host fallback (or
  a pad-channels-then-slice wrapper). FPN necks are typically 64/128/256.
- **The scale factor is an integer.**
- **The coordinate convention is half-pixel with a zero boundary.** Framework-exact
  coordinate modes are left to the frontend that calls the primitive: `align_corners=True`,
  clamp vs zero boundary, and the TFLite `half_pixel_centers` flag. A clamp boundary would
  need the boundary output rows patched on the host after readback, like the LeakyReLU x~0
  repair ([dpu-lut-activation.md](dpu-lut-activation.md)).

## Lowering

### Depthwise transposed convolution

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

### The triangle kernel as bilinear interpolation

The triangle has width `2·scale`, but its stride-`scale` subsample is a partition of unity.
At every output phase, the taps that fall on input-lattice positions sum to exactly 1. A
constant input therefore upsamples to that constant (interior). Only two taps are nonzero at
any output position: the two nearest input samples, weighted `(1−d)` and `d`. The geometry
gives the half-pixel map as the source coordinate:

```
src = (o + 0.5)/scale − 0.5         (== F.interpolate(..., align_corners=False))
```

The map has a zero boundary: out-of-lattice taps contribute 0, because the dilated input is
zero-padded. These are the FCN/segmentation "bilinear-deconv" init kernels.

## Accuracy

**Conditions:** RK3588 (Turing RK1), kernel 7.1.0-1, NPU clock ~816 MHz, 2026-06-22.

Nearest is bit-exact against block replication, and bilinear is within fp16 tolerance of an
independent 2-tap gather [HW sweep]. The partition-of-unity and linear-exactness properties
both hold [HW sweep]:

| Check | Reference | Result |
|---|---|---|
| Nearest | Block replication | Bit-exact |
| Bilinear | The 2-tap half-pixel gather | <=0.05 |
| Partition of unity | A constant input upsamples to the same constant | <=0.05 |
| Linear exactness | A y-ramp upsamples to the half-pixel ramp | <=0.05 |

For `scale=2` the kernel is `[0.25, 0.75, 0.75, 0.25]`, which is exact in fp16, so the
upsample is bit-exact. For `scale=3` the kernel is `[⅓,⅔,1,⅔,⅓]`, where ⅓ rounds in fp16, so
the error is ~1e-3.

## Performance

- Cost scales with the upsampled size (the materialized stride dilation performs MACs on
  zeros). The sub-pixel / `scale²` decomposition (one small dense conv per phase, no
  zero-MACs) removes that cost in principle. For ConvTranspose, it costs more than the
  lowering at every k4 layer measured ([conv-transpose.md](conv-transpose.md)).
- Standalone, a host upsample is cheaper than the NPU round-trip. The NPU path's value is
  keeping the feature cube-resident between two NPU ops
  ([cross-op-chaining.md](cross-op-chaining.md)).

See also: [conv-transpose.md](conv-transpose.md), [depthwise-conv.md](../depthwise-conv.md).

## Evidence

### Accuracy

The test is `tests/resize_rocket.c` (CTest `resize_rocket`). Its references are independent
of the NPU path. The NPU runs a transposed-conv *scatter* and the references run a *gather*,
so a kernel or coordinate bug cannot hide. All sweep cases pass.
