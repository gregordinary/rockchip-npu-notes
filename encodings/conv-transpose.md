# Transposed convolution (ConvTranspose2d / "deconvolution") on rocket

The transpose of a strided convolution — every input pixel **scatter-adds** a
kernel-weighted copy into a *larger* output. It is the learned-upsampling primitive of
segmentation heads, decoder / super-resolution / GAN-generator blocks, and FPN
learned-upsample (ONNX `ConvTranspose`, PyTorch `nn.ConvTranspose2d`, TFLite
`TRANSPOSE_CONV`). Implemented: `rocket_conv_transpose2d_fp16` (`src/rocket_conv_transpose.c`,
`include/rocket_conv.h`), HW gate `tests/conv_transpose_rocket.c` (CTest `conv_transpose_rocket`).

**Established by:** HW run on the Turing RK1 (kernel 7.1.0-1, 600 MHz), bit-exact vs a
direct scatter-add reference across stride 1/2/3, pad, output_padding, dilation>1,
asymmetric kernels, multi-group IC/OC, and a tiled 64×64 output (2026-06-22).

## There is no transpose-conv hardware — it lowers onto the forward conv

The CNA is a forward convolution engine; the RK3588 NPU has **no** dedicated
transposed-conv / deconv mode (and no on-chip layout/scatter engine to build the dilated
input — consistent with [no on-chip layout conversion](../perf/ppu-pooling-not-detile.md)).
So a transposed conv is realised by the **standard lowering identity**:

```
ConvTranspose(X; W, stride s, pad p, dil d, opad)
  ==  Conv( dilate_and_pad(X), rot180(Wᵀ);  stride 1, pad 0, dil d )
```

i.e. it reuses the **HW-validated forward `rocket_conv2d_fp16`** (auto-tiled over
OC / OH-rows / OW-cols) bit-for-bit. Only the host packing is new:

1. **Interior-dilate the input** — insert `s−1` zero rows/cols *between* input pixels, so
   `X[ih]` lands at output-of-dilation row `lead + ih·s`.
2. **Border-pad** — leading border `lead = d·(K−1) − p`, trailing border `lead + opad`.
   The dilated+padded height is `(IH−1)·s + 1 + 2·d·(K−1) − 2p + opad`.
3. **Rotate + transpose the kernel** — `wf[oc][ic][kh][kw] = W[ic][oc][K−1−kh][K−1−kw]`
   (180° spatial flip **and** in/out-channel swap, because ConvTranspose weights are
   stored `[IC][OC][KH][KW]` — in-channels first).
4. **Forward stride-1 conv** with `wf` over the dilated input.

### Why the 180° flip (one-axis derivation)

ConvTranspose places `in[ih]` at output position `ph = ih·s − p + kh·d`. In the lowered
input `xd`, `in[ih]` sits at row `r = lead + ih·s` with `lead = d·(K−1) − p`. The forward
conv reads `xd[ph + kf·d]` for forward-kernel index `kf`; that equals `lead + ih·s` exactly
when **`kf = K−1−kh`** — so the forward kernel slot `K−1−kh` must carry ConvTranspose weight
`kh`. `xd` is zero off the stride lattice and in the border, so only integral, in-range `ih`
contribute (zero-padding handles the boundary). Output size checks out:
`IHd − d·(K−1) = (IH−1)·s − 2p + d·(K−1) + opad + 1 = OH`.

## Output size

```
OH = (IH−1)·stride_y − 2·pad_top  + dil_y·(KH−1) + opad_y + 1
OW = (IW−1)·stride_x − 2·pad_left + dil_x·(KW−1) + opad_x + 1
```

`opad` (output_padding) is an extra **trailing-only** border that disambiguates the size
when `s>1` (must be `< stride`); it appears only in the trailing pad, never the leading one.

## Constraints & cost

- **`pad ≤ dil·(K−1)`** on each axis, else the leading border `d·(K−1)−p` goes negative —
  that case is an output *crop*, not implemented; `rocket_conv_transpose2d_plan` returns `−2`
  (a clean decline, never a miscompute). The lowered forward conv's CBUF-fit is propagated
  too (`−4` from `rocket_conv2d_plan`).
- **OC/IC** follow the forward conv: any OC is zero-padded to the 16-channel oc-group, any IC
  to the 32-channel K-group (so an RGB-width `IC=3` transpose works).
- **Cost scales with the *upsampled* size.** The materialised dilation means the inserted
  zeros are still MAC'd by the CNA (a stride-`s` transpose does ~`s²` redundant zero-MACs).
  Correctness-first. The perf follow-on is the **sub-pixel / stride² decomposition**: run `s²`
  small *dense* forward convs (one per `(kh mod s, kw mod s)` phase) and interleave their
  outputs — no zero-MACs — which is how efficient deconv is normally done. Not yet built.

## Validation

`tests/conv_transpose_rocket.c` is a two-layer gate:
- **Lowering self-check** (runs anywhere, no NPU): an *independent* in-test re-derivation of
  the dilate+flip lowering, fed through the forward-conv CPU oracle, compared to the direct
  scatter-add definition `rocket_conv_transpose2d_ref_fp16`. Proves the geometry math.
- **HW end-to-end**: `rocket_conv_transpose2d_fp16` on the NPU vs the scatter reference.

Small integer inputs keep every result exact in fp16 (`|sum| < 2048`) → the bar is
`max_abs == 0` (lowering) and `≤ 1.0` (HW fp16 narrowing). All sweep shapes pass `max_abs = 0`.

See also: [matmul-as-conv.md](../matmul-as-conv.md) (the forward conv = CNA primitive),
[depthwise-conv.md](../depthwise-conv.md), [tile-layouts.md](tile-layouts.md).
