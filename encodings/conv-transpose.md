# Transposed convolution (ConvTranspose2d / "deconvolution") on rocket

A transposed convolution is the transpose of a strided convolution: every input pixel
scatter-adds a kernel-weighted copy into a larger output. It is the learned-upsampling
primitive of segmentation heads, of decoder, super-resolution and GAN-generator blocks, and
of the FPN learned-upsample. The frameworks name it ONNX `ConvTranspose`, PyTorch
`nn.ConvTranspose2d` and TFLite `TRANSPOSE_CONV`. `librocketnpu` implements it as
`rocket_conv_transpose2d_fp16` (`src/rocket_conv_transpose.c`, `include/rocket_conv.h`). The
hardware gate is `tests/conv_transpose_rocket.c` (CTest `conv_transpose_rocket`).

The entry has two routes. On the RK3588, a direct transposed conv at power-of-two strides
whose compact input fits one CBUF pass runs on the hardware deconvolution mode described
below. `librocketnpu` lowers everything else onto the forward conv.

A third entry keeps its weight resident and runs the transpose as one matmul and a host
scatter-add. At model shapes it is the fast one (§"The resident route: col2im over the matmul").

**Established by:** a hardware run on the Turing RK1 (kernel 7.1.0-1, 600 MHz), 2026-06-22.
The result is bit-exact vs a direct scatter-add reference across these axes:

- Stride 1/2/3
- Pad and output_padding
- Dilation>1
- Asymmetric kernels
- Multi-group IC/OC
- A tiled 64×64 output

## The lowering onto the forward conv

The lowering covers what the hardware mode does not:

- Odd strides
- Dilation
- The depthwise form
- Inputs past one CBUF pass

It needs no layout or scatter engine on the chip (consistent with
[no on-chip layout conversion](../perf/ppu-pooling-not-detile.md)), because it uses the
standard lowering identity:

```
ConvTranspose(X; W, stride s, pad p, dil d, opad)
  ==  Conv( dilate_and_pad(X), rot180(Wᵀ);  stride 1, pad 0, dil d )
```

The lowering reuses the hardware-validated forward `rocket_conv2d_fp16` (auto-tiled over
OC / oh-rows / ow-cols) bit-for-bit. Only the host packing is new:

1. **Interior-dilate the input.** Insert `s−1` zero rows and columns between input pixels,
   so `X[ih]` lands at output-of-dilation row `lead + ih·s`.
2. **Border-pad.** The leading border is `lead = d·(K−1) − p`, trailing border `lead + opad`.
   The dilated+padded height is `(IH−1)·s + 1 + 2·d·(K−1) − 2p + opad`.
3. **Rotate + transpose the kernel.** `wf[oc][ic][kh][kw] = W[ic][oc][K−1−kh][K−1−kw]`.
   This is a 180° spatial flip and an in/out-channel swap, because ConvTranspose weights
   are stored `[IC][OC][KH][KW]`, in-channels first. The depthwise form stores
   `[C][1][KH][KW]` with `OC == IC`, so it takes the spatial flip alone and no channel swap.
4. **Forward stride-1 conv** with `wf` over the dilated input.

### The 180° flip (one-axis derivation)

ConvTranspose places `in[ih]` at output position `ph = ih·s − p + kh·d`. In the lowered
input `xd`, `in[ih]` sits at row `r = lead + ih·s` with `lead = d·(K−1) − p`. The forward
conv reads `xd[ph + kf·d]` for forward-kernel index `kf`. That row equals `lead + ih·s`
exactly when `kf = K−1−kh`, so the forward kernel slot `K−1−kh` must carry ConvTranspose
weight `kh`. The lowered input `xd` is zero off the stride lattice and in the border, so
only integral, in-range `ih` contribute (zero-padding handles the boundary). The output size
checks out: `IHd − d·(K−1) = (IH−1)·s − 2p + d·(K−1) + opad + 1 = OH`.

## Output size

```
OH = (IH−1)·stride_y − 2·pad_top  + dil_y·(KH−1) + opad_y + 1
OW = (IW−1)·stride_x − 2·pad_left + dil_x·(KW−1) + opad_x + 1
```

`opad` (output_padding) is an extra trailing-only border that disambiguates the size when
`s>1`. It must be `< stride`. It appears only in the trailing pad, never the leading one.

## Constraints and cost

- **`pad <= dil·(K−1)`** on each axis, or the leading border `d·(K−1)−p` goes negative.
  That case is an output crop, which is not implemented. `rocket_conv_transpose2d_plan`
  returns `−2` (a clean decline, never a miscompute). It also propagates the lowered forward
  conv's CBUF fit (`−4` from `rocket_conv2d_plan`).
- **OC/IC** follow the forward conv: any OC is zero-padded to the 16-channel oc-group, any IC
  to the 32-channel K-group (so an RGB-width `IC=3` transpose works).
- **Cost scales with the upsampled size.** The materialized dilation means the host packs
  and the CNA fetches an input about `s²` larger, and the inserted zeros are MAC'd. The
  hardware mode below is 0.35-0.54x this path's wall at decoder shapes. The resident route
  below avoids both costs. The sub-pixel decomposition, `s²` dense forward convs interleaved,
  was priced against it and costs more at every k4 layer measured.

## The hardware deconvolution mode

The CNA register map carries a transposed-convolution mode that no reference driver programs.
Its fields are `CNA_CONV_CON1[16]` `DECONV`, `CNA_CONV_CON3[13:11]` `DECONV_Y_STRIDE` and
`[10:8]` `DECONV_X_STRIDE` [Mesa `registers.xml` and allbilly `rkt_registers.h`, both naming
all three]. NVDLA has no deconvolution engine at any revision, so the mode is Rockchip's own
addition, and the NVDLA documentation says nothing about it
([nvdla-lineage.md](../nvdla-lineage.md)).

The mode is live on the RK3588 [HW sweep, Turing RK1, `tests/deconv_mode_probe.c`]. The
probe runs a 32×8×8 k3 stride-1 fp16 conv once plain, and once per cell with the mode bit
and the two stride fields set. It runs every cell twice and reports a cell only when the two
runs agree:

| `DECONV` | stride field | result |
|---|---|---|
| 0 | 0-7 | byte-identical to the forward conv at every value |
| 1 | 0, 2, 4, 5, 6 | byte-identical to the forward conv |
| 1 | 1 | 1145 of 1152 elements differ, 49 zero |
| 1 | 3 | 1118 of 1152 differ, 640 zero |
| 1 | 7 | 1152 of 1152 differ, 1120 zero: exactly one non-zero per output channel |

The table settles three things:

- **Bit 16 gates the stride fields.** With the bit clear, they do nothing at any value. That
  is the control that makes the rest readable.
- **Only field values 1, 3 and 7 are live**, i.e. `2^k − 1`. The field is therefore
  stride-encoded, and the mode covers power-of-two strides only (the common learned-upsample
  case).
- **The surfaces get monotonically sparser with the field value**, ending at one live
  element per channel. A scatter into a stride-dilated grid leaves that structure. It is the
  strongest evidence that the mode is a transposed convolution rather than a corrupted fetch.

### The decoded geometry

The field is `s−1`, and the mode is a hardware interior dilation of the input
[HW sweep, Turing RK1 at 600 MHz, `tests/deconv_geometry_probe.c` +
`tests/deconv_arith_probe.c`].

The instrument is an impulse: one non-zero input element at `(r,c)`, one non-zero filter
`(oc0,ic0)` holding 1..9, everything else zero. Output channel 0 then holds exactly one copy
of the kernel, and its address is the measurement. A count of differing elements, which is
all the table above has, cannot say where the mode puts its result.

| field | box start, impulse row 2 | impulse row 3 | step |
|---|---|---|---|
| 0 | 0 | 1 | 1 |
| 1 | 2 | 4 | 2 |
| 3 | 6 | 10 | 4 |
| 7 | 14 | (off the 22-row canvas) | 8 |

The box therefore starts at `s·r − (k−1)` with `s = field + 1`, and the kernel arrives
flipped, the same orientation the plain forward correlation gives. Placement at `s·r` with
a flipped kernel and a `−(k−1)` offset is exactly a forward correlation over an input
interior-dilated by `s`. That is the textbook transposed-convolution identity
`ConvTranspose(x,W,s,p) == Conv(dilate_s(x), flip(W), 1, k−1−p)`.

Dense data confirms the decode against the independent oracle. Driving the undilated input,
a 180°-flipped kernel and the `[OC][IC] ← [IC][OC]` channel transpose gives a result
bit-exact against `rocket_conv_transpose2d_ref_fp16`. The match covers 1568 elements at
`s=2` and 5408 at `s=4`, with zero mismatches. An impulse alone could not have shown this:
it never exercises the accumulation where two scattered copies overlap.

**The caller owes both host-side transforms.** The mode does the dilation only. It does not
do the spatial 180° flip or the in/out channel transpose. Getting either wrong computes a
full, correctly-sized, entirely plausible surface, which is this datapath's signature
failure.

### The programmed output extent

The mode computes the whole transposed convolution in one task when the output geometry is
programmed as the transposed extent [HW sweep, 2026-09-23]. The probe is
`tests/deconv_extent_probe.c`, run on the Turing RK1 at 600 MHz with rocket 1.3.0. The CNA
walks whatever extent the geometry registers give it, so the program sets two extents that
a forward conv derives from one:

| Register | Value |
|---|---|
| `CNA_CONV_CON1[16]` `DECONV` | 1 |
| `CNA_CONV_CON3` `DECONV_Y/X_STRIDE` | `s-1` per axis |
| `CNA_CONV_CON3` conv stride | 1 |
| `CNA_DATA_SIZE0` and the feature DMA geometry | the undilated input |
| `CNA_PAD_CON0` | `k-1-p` per axis |
| `CNA_DATA_SIZE2/3`, `CORE_DATAOUT_SIZE_0`, the DPU cube and WDMA sizes | the transposed extent |

The caller flips and channel-transposes the kernel as in the lowering above. Against
`rocket_conv_transpose2d_ref_fp16` ten cells are bit-exact over the whole surface:

| Input | Channels | Kernel, stride, pad | Output |
|---|---|---|---|
| 4×4 | 32 -> 32 | k3, s2, p0 | 9×9 |
| 32×32 | 32 -> 32 | k3, s2, p0 | 65×65 |
| 16×16 | 32 -> 32 | k3, s4, p0 | 63×63 |
| 32×32 | 32 -> 32 | k4, s2, p1 | 64×64 |
| 32×32 | 32 -> 32 | k2, s2, p0 | 64×64 |
| 8×16 | 32 -> 32 | k3, s2, p1 | 15×31 |
| 8×8 | 32 -> 32 | k3, sy2 sx1, p0 | 17×10 |
| 16×16 | 64 -> 32 | k3, s2, p1 | 31×31 |
| 16×16 | 32 -> 32 | k3, s2, p1, output_padding 1 | 32×32 |
| 64×64 | 32 -> 32 | k4, s2, p1 | 128×128 |

The default feature-grain count (`IH+1`) suffices. Every cell ran twice with identical output,
and the device logged no job timeout. open-rknpu reports the same program byte-exact on the
RV1106 NPU, across about 90 of its own models. The RV1106's CNA geometry words sit at the
RK3588's offsets, and its stride fields are live without `CONV_CON1[16]` (see
[../SOURCES.md](../SOURCES.md)).

**An extent derived from the forward arithmetic truncates the result, silently.** Programmed
the forward conv's way, from the undilated input, the part writes `ih + 2P - k + 1` rows and
drops the rest. The result is a full, correctly sized surface of the wrong size. A complete
result that way needs `P >= toh - ih`, and the pad field caps that at
`ih <= (15 + s - k) / (s - 1)`, which is 14 at `s=2`. The bound belongs to the derived
extent. With the extent programmed, the pad is `k-1-p` and the field never binds.

### The pad on the dilated surface, and the 4-bit pad field

`CNA_PAD_CON0`'s `PAD_TOP`/`PAD_LEFT` are 4-bit fields, so the usable pad is 0-15. **Pad 16
behaves exactly as pad 0, 17 as 1, 18 as 2** [HW sweep, `tests/deconv_pad_probe.c`]. The
wrap is measured by sweeping past the claimed ceiling, rather than read off the map.

Within that range the pad lands on the dilated surface: the result matches the oracle at
offset `(k-1) - P`, identically at `s=2` and `s=4`. A pad applied before dilation would give
an offset that scales with `s`, and the measured offset does not.

**One narrow anomaly is open.** At `s=2`, pads 12-15 match the oracle at no offset at all,
while every pad 0-11 matches and 16+ wraps cleanly. At `s=4` the same pads are fine. The
anomaly sits immediately below the wrap and is undecoded. Do not assume the `(k-1)-P` rule
holds there. A transposed conv's own pad, `k-1-p`, stays below it for every kernel up to 12.

### Speed against the lowering

One hardware job is 0.35-0.54x the lowering's wall per call [HW sweep, Turing RK1 at
600 MHz, rocket 1.3.0, 2026-09-23]. The arms ran under the `performance` governor and
`taskset 0xf0`, from `deconv_extent_probe bench`. Each figure is a median of 15 interleaved
reps per pass, and two passes agree within 1%:

| Shape | Lowering | Hardware job | Ratio |
|---|---|---|---|
| 64 -> 64 channels, 32×32 -> 64×64, k4 s2 p1 | 7.00 ms | 3.67 ms | 0.525x |
| 32 -> 32, 64×64 -> 128×128, k4 s2 p1 | 11.35 ms | 5.62 ms | 0.495x |
| 128 -> 64, 16×16 -> 32×32, k4 s2 p1 | 6.01 ms | 2.08 ms | 0.345x |
| 64 -> 64, 32×32 -> 64×64, k2 s2 p0 | 5.45 ms | 2.92 ms | 0.535x |

Both arms are per call with no resident context, and the hardware arm includes the host's
kernel flip. Their outputs are identical and bit-exact against the oracle. The measurement
cannot say whether the MAC array skips the inserted zeros, because the host packing and the
device time move together here.

### Driving the mode

`rocket_conv_transpose2d_fp16` takes the mode on its own route
(`rocket_conv_transpose2d_route()` says which). It sets `npu_cna_desc.deconv` and the two
stride fields from `conv_params_t.deconv_sy/sx`. `ROCKET_CONV_TRANSPOSE_HW=0` forces the
lowering. For probes, `ROCKET_CNA_DECONV=1` and `ROCKET_CNA_DECONV_X` / `_Y` write the three
fields raw (the stride value is `s-1`). `rocket_conv2d_fp16_job_extent()`
(`src/rocket_conv_internal.h`) runs one fp16 conv job with the output extent given rather
than derived.

## The resident route: col2im over the matmul

A transposed convolution is one matmul followed by a scatter. Read the input as `IC` planes of
`IH·IW` pixels and the weight as stored, `[IC][OC·KH·KW]`. Their product holds every tap of every
input pixel. Each tap's plane then adds into the output at the offset its tap gives it. Nothing
is dilated and no zero is multiplied. rocket-userspace's `rocket_conv_transpose2d_fp16_prepacked`
runs the product on the resident fp16 matmul, its weight packed once, and the add on the host.

The layer is worth a route of its own in two models. ConvTranspose is 65.5% of pix2pix's CPU
wall and 17.8% of SAM's automatic mask generation. In the other models profiled it is 0.1-6.6%
[ONNX Runtime CPU profiles, 2026-09-27].

The lowering loses at model shapes through its job count. At pix2pix's up6 (512 channels,
32×32 -> 128 at k4 s2) it runs 704 jobs a call: 8 OC tiles, 22 row bands and 4 column bands. A
dilated input row is 68.6 KB, so a band holds only the tiler's 6-row floor. The 16-kernel weight
tile takes 8 of the 12 banks. The per-job weight scatter is 47% of the 272 ms call [HW sweep,
RK1, `ROCKET_CONV_PROFILE`, 2026-09-27].

The table compares each call with ONNX Runtime 1.27.0's CPU time for the same layer at two
threads, on the same board. The NPU arm ran on cores 4-7 and the CPU arm on 6-7 [HW sweep, RK1
at 600 MHz, `rocket` 1.3.0, governor `performance`, `tests/ct_model_bench.c`, 2026-09-27]:

| Layer | Resident | CPU, 2 threads | Resident / CPU | Lowering or deconv mode |
|---|---:|---:|---:|---:|
| pix2pix up1, 512×1×1 -> 512 | 0.93 ms | 1.68 ms | 0.56 | 40.7 ms |
| pix2pix up2, 1024×2×2 -> 512 | 1.77 | 3.61 | 0.49 | refused |
| pix2pix up3, 1024×4×4 -> 512 | 1.92 | 5.13 | 0.38 | refused |
| pix2pix up4, 1024×8×8 -> 512 | 2.50 | 20.5 | 0.12 | refused |
| pix2pix up5, 1024×16×16 -> 256 | 4.23 | 40.2 | 0.11 | refused |
| pix2pix up6, 512×32×32 -> 128 | 5.28 | 43.8 | 0.12 | 293 |
| pix2pix up7, 256×64×64 -> 64 | 9.19 | 45.1 | 0.20 | 129 |
| pix2pix up8, 128×128×128 -> 3 | 3.18 | 8.78 | 0.36 | 57.4 |
| SAM upscaler 1, 256×64×64 -> 64 | 2.94 | 13.3 | 0.22 | 24.1 |
| SAM upscaler 2, 64×128×128 -> 32 | 4.87 | 10.3 | 0.47 | 16.1 |

The table is two rotated passes with ratios paired inside each pass. Five passes read 0.11-0.58.
The matmul is 69-87% of a call from up4 on, and the single-threaded scatter-add is the rest. The
CPU times are each node's share of a whole-model profile. Every weight is packed outside the
timing.

### Accuracy

Integer data whose partials stay below 2048 comes back bit-exact against the direct
scatter-add reference. With real data and a bias, the worst error against an fp64 reference
is 2^-13.9 to 2^-11.1 of each output's magnitude sum. That sum is `|x·w|` over the output's
taps. The gate bounds it at 2^-9 [HW sweep, RK1, `tests/conv_transpose_resident.c`,
2026-09-27].

### Scope of the measurements

A frontend's layout and precision conversion, a whole model's wall and accuracy, and the
memory of every layer resident at once (about 100 MB for pix2pix) are unmeasured. The CPU
arm ran at two threads only. The route refuses depthwise and the RK3576.

## Validation

`tests/conv_transpose_rocket.c` is a two-layer gate:

- **Lowering self-check** (runs anywhere, no NPU): an independent in-test re-derivation of
  the dilate+flip lowering, fed through the forward-conv CPU oracle, compared to the direct
  scatter-add definition `rocket_conv_transpose2d_ref_fp16`. It proves the geometry math.
- **HW end-to-end**: `rocket_conv_transpose2d_fp16` on the NPU vs the scatter reference.

Small integer inputs keep every result exact in fp16 (`|sum| < 2048`), so the bar is
`max_abs == 0` (lowering) and `<= 1.0` (HW fp16 narrowing). All sweep shapes pass `max_abs = 0`.

See also: [matmul-as-conv.md](../matmul-as-conv.md) (the forward conv = CNA primitive),
[depthwise-conv.md](../depthwise-conv.md), [tile-layouts.md](tile-layouts.md).
