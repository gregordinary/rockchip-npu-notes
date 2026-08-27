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

## The shipping path lowers onto the forward conv

The CNA is a forward convolution engine and there is no on-chip layout/scatter engine to
build the dilated input (consistent with
[no on-chip layout conversion](../perf/ppu-pooling-not-detile.md)), so the transposed conv
is realised by the **standard lowering identity**:

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

## There IS a hardware deconvolution mode, and it is live

The CNA register map carries a transposed-convolution mode nothing in this stack drives:
**`CNA_CONV_CON1[16]` `DECONV`**, plus **`CNA_CONV_CON3[13:11]` `DECONV_Y_STRIDE`** and
**`[10:8]` `DECONV_X_STRIDE`** [Mesa `registers.xml`; allbilly `rkt_registers.h` — both name
all three]. NVDLA has no deconvolution engine at any revision, so this is Rockchip's own
addition and its ancestor's documentation says nothing about it
([nvdla-lineage.md](../nvdla-lineage.md)).

It is **live on the RK3588** [HW sweep, Turing RK1, `tests/deconv_mode_probe.c`]. A 32×8×8
k3 stride-1 fp16 conv, run once plain and once per cell with the mode bit and the two stride
fields set, every cell run twice and reported only when the two agree:

| `DECONV` | stride field | result |
|---|---|---|
| 0 | 0–7 | byte-identical to the forward conv at every value |
| 1 | 0, 2, 4, 5, 6 | byte-identical to the forward conv |
| 1 | 1 | 1145 of 1152 elements differ, 49 zero |
| 1 | 3 | 1118 of 1152 differ, 640 zero |
| 1 | 7 | 1152 of 1152 differ, 1120 zero — exactly one non-zero per output channel |

Three things are settled by that table. The stride fields are **gated by bit 16** — with the
bit clear they do nothing at any value, which is the control that makes the rest readable.
Only field values **1, 3 and 7** are live, i.e. `2^k − 1`, so the field is stride-encoded and
the mode covers **power-of-two strides only** (the common learned-upsample case). And the
surfaces get monotonically **sparser** with the field value, ending at one live element per
channel — the structure a scatter into a stride-dilated grid leaves, and the strongest
evidence that this is a transposed convolution rather than a corrupted fetch.

### The geometry, decoded

**The field is `s−1`, and the mode is a hardware INTERIOR DILATION of the input**
[HW sweep, Turing RK1 at 600 MHz, `tests/deconv_geometry_probe.c` +
`tests/deconv_arith_probe.c`].

The instrument is an **impulse**: one non-zero input element at `(r,c)`, one non-zero filter
`(oc0,ic0)` holding 1..9, everything else zero. Output channel 0 then holds exactly one copy
of the kernel and its ADDRESS is the measurement — a count of differing elements, which is
all the table above has, cannot say where the mode puts its result.

| field | box start, impulse row 2 | impulse row 3 | step |
|---|---|---|---|
| 0 | 0 | 1 | 1 |
| 1 | 2 | 4 | 2 |
| 3 | 6 | 10 | 4 |
| 7 | 14 | (off the 22-row canvas) | 8 |

So the box starts at **`s·r − (k−1)` with `s = field + 1`**, and the kernel arrives
**FLIPPED** — the same orientation the plain forward correlation gives. Placement at `s·r`
with a flipped kernel and a `−(k−1)` offset is exactly a forward correlation over an input
interior-dilated by `s`, which is the textbook transposed-convolution identity
`ConvTranspose(x,W,s,p) == Conv(dilate_s(x), flip(W), 1, k−1−p)`.

**Confirmed on dense data against the independent oracle**: driving the undilated input, a
180°-flipped kernel and the `[OC][IC] ← [IC][OC]` channel transpose gives a result
**bit-exact** against `rocket_conv_transpose2d_ref_fp16` — 1568 elements at `s=2` and 5408 at
`s=4`, zero mismatches. An impulse alone could not have shown this: it never exercises the
accumulation where two scattered copies overlap.

**The caller still owes both host-side transforms.** The spatial 180° flip and the in/out
channel transpose are NOT done by the mode; only the dilation is. Getting either wrong
computes a full, correctly-sized, entirely plausible surface, which is this datapath's
signature failure.

### The pad reaches the dilated surface, and the pad field is 4 bits

`CNA_PAD_CON0`'s `PAD_TOP`/`PAD_LEFT` are 4-bit fields, so the usable pad is **0–15** and
**pad 16 behaves exactly as pad 0, 17 as 1, 18 as 2** — measured, not read off the map, by
sweeping past the claimed ceiling [HW sweep, `tests/deconv_pad_probe.c`].

Within that range the pad lands on the **dilated** surface: the result matches the oracle at
offset `(k−1) − P`, identically at `s=2` and `s=4`. Had the pad been applied before dilation
the offset would have scaled with `s`; it does not.

**The output extent is NOT enlarged by the mode.** The CORE/DPU geometry registers are still
driven from the UNDILATED forward arithmetic, so the part writes `ih + 2P − k + 1` rows and
truncates the rest of the transposed result. A complete result therefore needs
**`P ≥ toh − ih`** (measured: `s=2` needs `P≥5`, `s=4` needs `P≥11`, both for a 4×4 input),
and with `P ≤ 15` that bounds the single-pass input to

    ih ≤ (15 + s − k) / (s − 1)      →   ih ≤ 14 at s=2, ih ≤ 5 at s=4, ih ≤ 2 at s=8

**which is the finding that bounds the payoff.** A 32×32 stride-2 decoder layer would need
`P = 33` and the field cannot express it, so the mode is not reachable single-pass at the
resolutions segmentation heads, depth decoders and FPN upsamples actually run at. An encoder
would have to tile the output. That is a real constraint on the item, not a detail.

**One narrow anomaly is open**: at `s=2`, pads **12–15** match the oracle at no offset at all,
while every pad 0–11 matches and 16+ wraps cleanly. At `s=4` the same pads are fine. It sits
immediately below the wrap and is undecoded; do not assume the `(k−1)−P` rule holds there.

**What is still NOT measured: whether any of this is FASTER.** Nothing here times the mode
against the shipping lowering. The host no longer materialises the `s²`-larger dilated input,
which is a real and certain saving; but whether the hardware SKIPS the `s²` zero-MACs or
merely expands internally is **not established**, and the `s²`-zero-MAC figure the section
above quotes as the payoff remains unverified. Measure before building an encoder.

**How to drive it**: `ROCKET_CNA_DECONV=1` sets the bit, `ROCKET_CNA_DECONV_X` /
`ROCKET_CNA_DECONV_Y` write the two 3-bit fields raw (the value is `s−1`). Still env-only and
absent from `npu_cna_desc`: the semantics are now decoded, but the single-pass input bound
above means a shipping entry would refuse most real decoder layers, so what it needs next is
a timing measurement and a tiling decision rather than an API.

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
