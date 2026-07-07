# Integer-factor resize / upsample (nearest, bilinear) on rocket

The FPN / decoder **neck** operator — upsample a feature map by an integer factor (TFLite
`RESIZE_NEAREST_NEIGHBOR` / `RESIZE_BILINEAR`, ONNX `Resize`, PyTorch `F.interpolate`).
Implemented: `rocket_upsample_nearest_fp16` / `rocket_upsample_bilinear_fp16`
(`src/rocket_resize.c`, `include/rocket_resize.h`), HW gate `tests/resize_rocket.c`
(CTest `resize_rocket`).

**Established by:** HW run on the Turing RK1 (kernel 7.1.0-1, 600 MHz), 2026-06-22 — nearest
bit-exact vs block-replication, bilinear within fp16 tolerance vs an independent 2-tap
gather, plus partition-of-unity + linear-exactness properties.

## An upsample IS a depthwise transposed conv

There is no resize hardware on the RK3588 NPU (no gather/sampler block). But an
integer-factor upsample is exactly a **depthwise [ConvTranspose2d](conv-transpose.md)** with
a fixed per-channel kernel: scatter each input pixel onto a stride-`scale` lattice and
convolve with a small kernel. So resize reuses the transposed-conv lowering (→ the forward
conv) bit-for-bit; only the kernel differs:

| mode | 1D kernel | size | what it does |
|------|-----------|------|--------------|
| nearest | `1,1,…,1` (box) | `scale` | replicate each pixel into a `scale×scale` block |
| bilinear | triangle `1−\|i−c\|/scale`, `c=(k−1)/2` | `k = 2·scale − scale%2` | 2-tap linear interpolation |

Both use **`pad = (k − scale)/2`, `opad = 0`**, which makes the output exactly `IH·scale ×
IW·scale` (the kernel size cancels in `OH = (IH−1)·scale − 2·pad + (k−1) + 1 = IH·scale`).
2D kernels are the separable outer product `tri_y ⊗ tri_x`.

## Why the triangle kernel is true bilinear

The triangle has width `2·scale`, but its **stride-`scale` subsample is a partition of
unity**: at every output phase the taps landing on input-lattice positions sum to exactly 1.
So a constant input upsamples to that constant (interior), and only **two** taps are nonzero
at any output position — the two nearest input samples, weighted `(1−d)` and `d`. Working out
the geometry, the source coordinate is the **half-pixel** map

```
src = (o + 0.5)/scale − 0.5         (== F.interpolate(..., align_corners=False))
```

with a **zero boundary** (out-of-lattice taps contribute 0, because the dilated input is
zero-padded). For `scale=2` the kernel is `[0.25, 0.75, 0.75, 0.25]` (exact in fp16 ⇒
bit-exact upsample); for `scale=3` it is `[⅓,⅔,1,⅔,⅓]` (⅓ rounds in fp16 ⇒ ~1e-3 error).
These are the FCN/segmentation "bilinear-deconv" init kernels.

Framework-exact coordinate modes (`align_corners=True`, clamp vs zero boundary, the TFLite
`half_pixel_centers` flag) are a **delegate-wiring** concern — this primitive fixes the
half-pixel / zero-boundary convention. (A clamp boundary would need the boundary output rows
patched on the host after readback, like the LeakyReLU x≈0 repair.)

## Constraints & cost

- **`C % 32 == 0`** — inherited from the depthwise forward conv's channel group (G=32). A
  feature map whose channel count isn't a multiple of 32 needs host fallback (or a
  pad-channels-then-slice wrapper); FPN necks are typically 64/128/256.
- Cost scales with the **upsampled** size (the materialised stride dilation MACs zeros). The
  sub-pixel / `scale²` decomposition (one small dense conv per phase, no zero-MACs) is the
  shared perf follow-on with [conv-transpose.md](conv-transpose.md).
- Standalone, a host upsample is cheaper than the NPU round-trip — the value is keeping the
  feature **cube-resident** between two NPU ops (the inter-op goal).

## Validation (`tests/resize_rocket.c`)

Independent references (the NPU runs a transposed-conv *scatter*; the refs run a *gather*, so
a kernel/coordinate bug cannot hide): nearest vs block-replication (bit-exact); bilinear vs
the 2-tap half-pixel gather (≤0.05); plus a constant→constant partition-of-unity check and a
y-ramp→half-pixel-ramp linear-exactness check (both ≤0.05). All sweep cases pass.

See also: [conv-transpose.md](conv-transpose.md), [depthwise-conv.md](../depthwise-conv.md).
