# Transposed convolution (ConvTranspose2d / "deconvolution") on rocket

A transposed convolution is the transpose of a strided convolution: every input pixel
scatter-adds a kernel-weighted copy into a larger output. It is the learned-upsampling
primitive of segmentation heads, of decoder, super-resolution and GAN-generator blocks, and
of the FPN learned-upsample. The frameworks name it ONNX `ConvTranspose`, PyTorch
`nn.ConvTranspose2d` and TFLite `TRANSPOSE_CONV`. The CNA has a hardware deconvolution mode
that computes a transposed convolution at a power-of-two stride in one task.
`CNA_CONV_CON1[16]` `DECONV` enables it, and `CNA_CONV_CON3` `DECONV_Y_STRIDE` and
`DECONV_X_STRIDE` set the stride. A transposed convolution the mode does not cover lowers
onto a stride-1 forward convolution, or onto one matmul and a host scatter-add.

## Support status

| Transposed convolution | Path | Status |
|---|---|---|
| RK3588, direct, power-of-two strides, compact input fits one CBUF pass | CNA deconvolution mode, one task | Supported (default) |
| Odd strides, dilation, the depthwise form, or an input past one CBUF pass | Lowering onto a stride-1 forward convolution | Supported |
| `pad > dil·(K−1)` on an axis, an output crop | Lowering | Not supported |
| Weight held resident | One matmul on the resident fp16 matmul, and a host scatter-add | Supported |
| Weight held resident, depthwise or on the RK3576 | Resident path | Not supported |

## Output size

```
OH = (IH−1)·stride_y − 2·pad_top  + dil_y·(KH−1) + opad_y + 1
OW = (IW−1)·stride_x − 2·pad_left + dil_x·(KW−1) + opad_x + 1
```

`opad` (output_padding) is an extra trailing-only border. When `s>1`, it disambiguates the
size. It must be `< stride`. It appears only in the trailing pad, never the leading one.

## Registers

**Conditions:** RK3588 (Turing RK1). The geometry decode ran at NPU clock ~816 MHz.

The CNA register map contains a transposed-convolution mode that no reference driver
programs [source-confirmed: Mesa `registers.xml`, allbilly `rkt_registers.h`, both naming all
three fields]. NVDLA has no deconvolution engine at any revision, so the mode is a Rockchip
addition, and the NVDLA documentation does not describe it
([nvdla-lineage.md](../nvdla-lineage.md)). The RK3588 implements the mode [HW sweep]:

| Register | Field | Bits | Function |
|---|---|---|---|
| `CNA_CONV_CON1` | `DECONV` | 16 | 1 enables the mode. At 0, the two stride fields have no effect at any value |
| `CNA_CONV_CON3` | `DECONV_Y_STRIDE` | 13:11 | The vertical stride `s`, programmed as `s−1` |
| `CNA_CONV_CON3` | `DECONV_X_STRIDE` | 10:8 | The horizontal stride `s`, programmed as `s−1` |

Only the field values 1, 3 and 7 are live, i.e. `2^k − 1`. The field is therefore
stride-encoded, and the mode covers power-of-two strides only (the common learned-upsample
case). With `DECONV` set, the field values 0, 2, 4, 5 and 6 leave the output byte-identical to
the forward convolution's [HW sweep].

The mode is a hardware interior dilation of the input [HW sweep]. An input element at row `r`
places its copy of the kernel starting at output row `s·r − (k−1)`, with `s = field + 1`. The
kernel arrives flipped, the same orientation the plain forward correlation gives. Placement at
`s·r` with a flipped kernel and a `−(k−1)` offset is exactly a forward correlation over an
input interior-dilated by `s`. The mode therefore implements the textbook transposed-convolution
identity `ConvTranspose(x,W,s,p) == Conv(dilate_s(x), flip(W), 1, k−1−p)`.

## Programming requirements

Software must flip the kernel 180 degrees and transpose its channels, `[OC][IC] ← [IC][OC]`.
The mode performs the dilation only. With the undilated input, the 180°-flipped kernel and the
channel transpose, the output is bit-exact against a host scatter-add reference [HW sweep].

> [!CAUTION]
> A kernel with either transform missing or wrong gives output of the correct size with
> incorrect values, and no error is reported.

### The programmed output extent

**Conditions:** RK3588 (Turing RK1), NPU clock ~816 MHz, `rocket` 1.3.0, 2026-09-23.

When software programs the output geometry as the transposed extent, the mode computes the
whole transposed convolution in one task [HW sweep, 2026-09-23]. The CNA walks whatever extent
the geometry registers give it. Software must therefore program two extents that a forward
convolution derives from one:

| Register | Value |
|---|---|
| `CNA_CONV_CON1[16]` `DECONV` | 1 |
| `CNA_CONV_CON3` `DECONV_Y/X_STRIDE` | `s-1` per axis |
| `CNA_CONV_CON3` conv stride | 1 |
| `CNA_DATA_SIZE0` and the feature DMA geometry | the undilated input |
| `CNA_PAD_CON0` | `k-1-p` per axis |
| `CNA_DATA_SIZE2/3`, `CORE_DATAOUT_SIZE_0`, the DPU cube and WDMA sizes | the transposed extent |

Software flips and channel-transposes the kernel as above. Against the host scatter-add
reference, ten test points are bit-exact over the whole output [HW sweep]:

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

The default feature-grain count (`IH+1`) is sufficient.

> [!NOTE]
> open-rknpu reports the same program byte-exact on the RV1106 NPU, across about 90 of its own
> models. The RV1106's CNA geometry words sit at the RK3588's offsets, and its stride fields are
> live without `CONV_CON1[16]` (see [../SOURCES.md](../SOURCES.md)).

> [!CAUTION]
> An extent derived from the forward arithmetic truncates the result, and no error is reported.
> Programmed the forward convolution's way, from the undilated input, the NPU writes
> `ih + 2P - k + 1` rows and drops the rest. The rows it writes form a complete output of the
> wrong size.

A complete result from a derived extent needs `P >= toh - ih`, where `toh` is the transposed
output height. The pad field caps that at `ih <= (15 + s - k) / (s - 1)`, which is 14 at `s=2`.
The bound applies to the derived extent only. With the extent programmed, the pad is `k-1-p`,
and the pad field never limits the input height.

### Pad field

`CNA_PAD_CON0`'s `PAD_TOP`/`PAD_LEFT` are 4-bit fields, so the usable pad is 0-15 [HW sweep].

> [!CAUTION]
> The pad field wraps with no error. Pad 16 behaves exactly as pad 0, 17 as 1, and 18 as 2.

Within that range, the pad applies to the dilated input: the output matches the reference at
offset `(k-1) - P`, identically at `s=2` and `s=4` [HW sweep].

> [!CAUTION]
> At `s=2`, pads 12-15 do not follow the `(k-1)-P` rule: the output matches the reference at
> no offset at all. Every pad 0-11 follows it, and 16 and above wrap. At `s=4` the same pads
> follow it. The range sits immediately below the wrap, and its cause is not known. A
> transposed convolution's own pad, `k-1-p`, stays below that range for every kernel up to 12.

## Software support

`librocketnpu` implements a transposed convolution as `rocket_conv_transpose2d_fp16`
(`src/rocket_conv_transpose.c`, `include/rocket_conv.h`). On the RK3588, it takes the
hardware mode for a direct transposed convolution at power-of-two strides whose compact
input fits one CBUF pass. It lowers everything else onto the forward convolution.
`rocket_conv_transpose2d_route()` reports which path a call takes. On the mode's path,
`rocket_conv_transpose2d_fp16` sets `npu_cna_desc.deconv` and the two stride fields from
`conv_params_t.deconv_sy/sx`. `rocket_conv2d_fp16_job_extent()` (`src/rocket_conv_internal.h`)
runs one fp16 convolution job with the output extent given rather than derived.

`ROCKET_CONV_TRANSPOSE_HW=0` forces the lowering. For register-level testing,
`ROCKET_CNA_DECONV=1` and `ROCKET_CNA_DECONV_X` / `_Y` write the three fields raw (the stride
value is `s-1`).

A third `librocketnpu` entry keeps its weight resident and runs the transpose as one matmul
and a host scatter-add. At model shapes it is the fastest of the three
(§"The resident route: col2im over the matmul").

### Speed against the lowering

**Conditions:** RK3588 (Turing RK1), NPU clock ~816 MHz, `rocket` 1.3.0, governor
`performance`, `taskset 0xf0`, 2026-09-23. Both paths run per call with no resident context,
and the hardware path includes the host's kernel flip.

**Not characterized:** whether the MAC array skips the inserted zeros. The host packing and the
device time change together in this comparison.

One hardware job takes 0.35-0.54x the lowering's wall-clock time per call [HW sweep]. The two
paths' outputs are identical, and bit-exact against the reference.

| Shape | Lowering | Hardware job | Ratio |
|---|---|---|---|
| 64 -> 64 channels, 32×32 -> 64×64, k4 s2 p1 | 7.00 ms | 3.67 ms | 0.525x |
| 32 -> 32, 64×64 -> 128×128, k4 s2 p1 | 11.35 ms | 5.62 ms | 0.495x |
| 128 -> 64, 16×16 -> 32×32, k4 s2 p1 | 6.01 ms | 2.08 ms | 0.345x |
| 64 -> 64, 32×32 -> 64×64, k2 s2 p0 | 5.45 ms | 2.92 ms | 0.535x |

### The lowering onto the forward conv

**Conditions:** RK3588 (Turing RK1), kernel 7.1.0-1, NPU clock ~816 MHz, 2026-06-22.

The lowering covers what the hardware mode does not:

- Odd strides
- Dilation
- The depthwise form
- Inputs past one CBUF pass

The lowering is bit-exact against a direct scatter-add reference across these axes
[HW sweep]:

- Stride 1/2/3
- Pad and output_padding
- Dilation>1
- Asymmetric kernels
- Multi-group IC/OC
- A tiled 64×64 output

It uses the standard lowering identity, so it needs no layout or scatter engine on the chip
([no on-chip layout conversion](../perf/ppu-pooling-not-detile.md)):

```
ConvTranspose(X; W, stride s, pad p, dil d, opad)
  ==  Conv( dilate_and_pad(X), rot180(Wᵀ);  stride 1, pad 0, dil d )
```

The lowering reuses the hardware-validated forward `rocket_conv2d_fp16` (auto-tiled over
OC / oh-rows / ow-cols) unchanged. Only the host packing is added:

1. **Interior-dilate the input.** Insert `s−1` zero rows and columns between input pixels,
   so `X[ih]` sits at output-of-dilation row `lead + ih·s`.
2. **Border-pad.** The leading border is `lead = d·(K−1) − p`, trailing border `lead + opad`.
   The dilated+padded height is `(IH−1)·s + 1 + 2·d·(K−1) − 2p + opad`.
3. **Rotate + transpose the kernel.** `wf[oc][ic][kh][kw] = W[ic][oc][K−1−kh][K−1−kw]`.
   The transform is a 180° spatial flip and an in/out-channel swap, because ConvTranspose
   weights are stored `[IC][OC][KH][KW]`, in-channels first. The depthwise form stores
   `[C][1][KH][KW]` with `OC == IC`, so it takes the spatial flip alone and no channel swap.
4. **Forward stride-1 conv** with `wf` over the dilated input.

#### The 180° flip (one-axis derivation)

ConvTranspose places `in[ih]` at output position `ph = ih·s − p + kh·d`. In the lowered
input `xd`, `in[ih]` sits at row `r = lead + ih·s` with `lead = d·(K−1) − p`. The forward
conv reads `xd[ph + kf·d]` for forward-kernel index `kf`. That row equals `lead + ih·s`
exactly when `kf = K−1−kh`, so the forward kernel slot `K−1−kh` must hold ConvTranspose
weight `kh`. The lowered input `xd` is zero off the stride lattice and in the border, so
only integral, in-range `ih` contribute (zero-padding handles the boundary). The output size
agrees: `IHd − d·(K−1) = (IH−1)·s − 2p + d·(K−1) + opad + 1 = OH`.

#### Constraints and cost

- **`pad <= dil·(K−1)`** on each axis, or the leading border `d·(K−1)−p` goes negative.
  That case is an output crop, which is not implemented. `rocket_conv_transpose2d_plan`
  refuses it with `−2` rather than computing an incorrect result. It also propagates the
  lowered forward conv's CBUF fit (`−4` from `rocket_conv2d_plan`).
- **OC/IC** follow the forward conv: any OC is zero-padded to the 16-channel oc-group, any IC
  to the 32-channel K-group (so an RGB-width `IC=3` transpose works).
- **Cost scales with the upsampled size.** The materialized dilation means the host packs
  and the CNA fetches an input about `s²` larger, and the inserted zeros are MAC'd. The
  hardware mode is faster at decoder shapes (§"Speed against the lowering"), and the resident
  path avoids both costs. The sub-pixel decomposition, `s²` dense forward convs interleaved,
  costs more than the lowering at every k4 layer measured.

### The resident route: col2im over the matmul

A transposed convolution is one matmul followed by a scatter. Read the input as `IC` planes of
`IH·IW` pixels and the weight as stored, `[IC][OC·KH·KW]`. Their product holds every tap of every
input pixel. Each tap's plane then adds into the output at the offset its tap gives it. Nothing
is dilated and no zero is multiplied. rocket-userspace's `rocket_conv_transpose2d_fp16_prepacked`
runs the product on the resident fp16 matmul, its weight packed once, and the add on the host.

ConvTranspose takes a large share of two models' CPU time: 65.5% of pix2pix's CPU
wall-clock time, and 17.8% of SAM's automatic mask generation. In the other models profiled it
is 0.1-6.6% [ONNX Runtime CPU profiles, 2026-09-27].

At model shapes, the lowering is slow because of its job count. At pix2pix's up6 (512
channels, 32×32 -> 128 at k4 s2), it runs 704 jobs a call: 8 OC tiles, 22 row bands and 4
column bands. A dilated input row is 68.6 KB, so a band holds only the tiler's 6-row minimum.
The 16-kernel weight tile takes 8 of the 12 banks. The per-job weight scatter is 47% of a
profiled call [HW sweep, 2026-09-27]. The table below gives the call's time.

The table compares each resident call with the CPU time for the same layer [HW sweep]. A whole
model's wall-clock time and accuracy through ONNX Runtime are in
[perf/benchmarks.md](../perf/benchmarks.md) §"ONNX Runtime (ort-rocket)".

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

**Conditions:** RK3588 (RK1), NPU clock ~816 MHz, `rocket` 1.3.0, governor `performance`,
2026-09-27. The NPU configuration ran on cores 4-7. The CPU configuration is ONNX Runtime
1.27.0's CPU time for the same layer at two threads, on cores 6-7 of the same board. Each CPU
time is that node's share of a whole-model profile. Every weight is packed outside the timing.

**Not characterized:** a frontend's layout and precision conversion, the memory of every layer
resident at once (about 100 MB for pix2pix), and the CPU at other thread counts.

The matmul is 69-87% of a call from up4 on, and the scatter-add, here on one thread, is the rest.

The scatter splits by output channel, since each channel accumulates its own phase planes. On
three threads it takes 0.45-0.47x its one-thread time at up6 and up7, and 0.56-0.57x at SAM's
upscalers. The output pass is memory-bound, so three threads do not give 3x [hypothesis].
Inside ONNX Runtime, the three-thread scatter changes pix2pix's wall-clock time by
0.946-0.972x [HW sweep, 2026-10-08, governor `ondemand`].

#### Accuracy

Integer data whose partials stay below 2048 comes back bit-exact against the direct
scatter-add reference. With real data and a bias, the worst error against an fp64 reference
is 2^-13.9 to 2^-11.1 of each output's magnitude sum. The magnitude sum is `|x·w|` over the
output's taps. The library's test threshold is 2^-9 [HW sweep, 2026-09-27].

See also: [matmul-as-conv.md](../matmul-as-conv.md) (the forward conv = CNA primitive),
[depthwise-conv.md](../depthwise-conv.md), [tile-layouts.md](tile-layouts.md).

## Evidence

### Registers

The mode test is `tests/deconv_mode_probe.c`, on the Turing RK1. It runs a 32×8×8 k3
stride-1 fp16 conv once plain, and once per test point with the mode bit and the two stride
fields set. It runs every test point twice and reports a test point only when the two runs
agree:

| `DECONV` | stride field | result |
|---|---|---|
| 0 | 0-7 | byte-identical to the forward conv at every value |
| 1 | 0, 2, 4, 5, 6 | byte-identical to the forward conv |
| 1 | 1 | 1145 of 1152 elements differ, 49 zero |
| 1 | 3 | 1118 of 1152 differ, 640 zero |
| 1 | 7 | 1152 of 1152 differ, 1120 zero: exactly one non-zero per output channel |

These results show:

- **Bit 16 enables the stride fields.** With the bit clear, they do nothing at any value. The
  `DECONV` = 0 row is the control for the rest of the table.
- **Only field values 1, 3 and 7 are live.**
- **The outputs get monotonically sparser with the field value**, ending at one live element
  per channel. A scatter into a stride-dilated grid leaves that structure. It is the strongest
  evidence that the mode is a transposed convolution rather than a corrupted fetch.

The geometry decode is `tests/deconv_geometry_probe.c` and `tests/deconv_arith_probe.c`, on
the Turing RK1 at ~816 MHz. The instrument is an impulse: one non-zero input element at
`(r,c)`, one non-zero filter `(oc0,ic0)` holding 1..9, everything else zero. Output channel 0
then holds exactly one copy of the kernel, and its address is the measurement. A count of
differing elements, which is all the table above has, cannot say where the mode puts its
result.

| field | box start, impulse row 2 | impulse row 3 | step |
|---|---|---|---|
| 0 | 0 | 1 | 1 |
| 1 | 2 | 4 | 2 |
| 3 | 6 | 10 | 4 |
| 7 | 14 | (off the 22-row canvas) | 8 |

Dense data confirms the decode against the independent reference,
`rocket_conv_transpose2d_ref_fp16`. The match covers 1568 elements at `s=2` and 5408 at
`s=4`, with zero mismatches. An impulse alone could not have shown this: it never exercises
the accumulation where two scattered copies overlap.

### Programming requirements

The output extent is `tests/deconv_extent_probe.c`, run on the Turing RK1 at ~816 MHz with
rocket 1.3.0, against `rocket_conv_transpose2d_ref_fp16`. Every test point ran twice with
identical output, and the device logged no job timeout.

The pad field is `tests/deconv_pad_probe.c`. The wrap is measured by sweeping past the claimed
ceiling, rather than read off the map. A pad applied before dilation would give an offset that
scales with `s`, and the measured offset does not.

### Software support

The speed comparison is `deconv_extent_probe bench`. Each figure is a median of 15
interleaved reps per pass, and two passes agree within 1%.

The lowering's test is `tests/conv_transpose_rocket.c` (CTest `conv_transpose_rocket`), in two
layers:

- **Lowering self-check** (runs anywhere, no NPU): an independent in-test re-derivation of
  the dilate+flip lowering, fed through the forward-conv CPU oracle, compared to the direct
  scatter-add definition `rocket_conv_transpose2d_ref_fp16`. It proves the geometry math.
- **HW end-to-end**: `rocket_conv_transpose2d_fp16` on the NPU vs the scatter reference.

Small integer inputs keep every result exact in fp16 (`|sum| < 2048`), so the threshold is
`max_abs == 0` (lowering) and `<= 1.0` (HW fp16 narrowing). All sweep shapes pass
`max_abs = 0`.

The lowering's job count and weight-scatter share at pix2pix's up6 are from
`ROCKET_CONV_PROFILE` on the RK1. That profiled call took 272 ms. The table's 293 ms for the
same call is from the table's benchmark, measured the same day (2026-09-27). The benchmark's
five-pass run below measures the call separately. The benchmark reports the median of warm
calls after a discarded warm-up.

The resident path's table is `tests/ct_model_bench.c`, two rotated passes with ratios paired
inside each pass. A five-pass run of the same benchmark ran on the same board and day, with the
page cache dropped before every process. It measures the resident-to-CPU ratio at 0.11-0.58.
Its lowering or deconvolution mode column is 40.3 ms at up1, 305.4 at up6, 134.7 at up7 and
54.5 at up8. At SAM's two upscalers it is 25.8 and 16.8 ms.

The three-thread scatter in ONNX Runtime is six rotated passes, on the RK1 at ~816 MHz, with
`ROCKET_CONV_PROFILE`. The resident accuracy test is `tests/conv_transpose_resident.c`, on the
RK1.
