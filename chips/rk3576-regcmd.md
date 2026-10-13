# RK3576 register encoding

This sheet describes the register encoding of the RK3576 NPU, as programmed through the
mainline `rocket` driver. The RK3576 NPU is an [RKNPU](../GLOSSARY.md) of the same family as
the RK3588's, with its blocks at the same bases. Its CNA, CORE and DPU use a different
geometry-register encoding. The SoC parameter sheet, [rk3576.md](rk3576.md), covers identity,
integration, clocks and power. The validation record,
[rk3576-validation.md](rk3576-validation.md), lists the captures and tests behind each section
here. The regcmd encoder in `librocketnpu` is
[npu_regcmd_rk3576.c](https://github.com/gregordinary/rocket-userspace/blob/main/src/npu_regcmd_rk3576.c).

The sheet covers:

- The int8 convolution, with its CBUF budget, weight slice and coefficient buffer
- The row window and the widths of the geometry fields
- The first-conv ARGB sub-encoding
- CBUF row reuse, and the stages a DPU epilogue appends
- Depthwise convolution
- The fp16 datapath
- The DPU shift word and the `PC_TASK_CON` task-number width
- Task completion, the IOMMU and BO lifetime under `rocket`
- The matmul as a 1×1 convolution, and its int32 output
- The DPU LUT, PPU pooling and the DPU's elementwise stage

The register formulas are transcribed from RKNN-Toolkit2 register programs for known conv
geometries.

> [!NOTE]
> A vendor capture can be built to order. RKNN-Toolkit2 emits the register program for
> whatever geometry an ONNX model compiled for `rk3576` contains. Setting `group=C` gives the
> depthwise path at any channel count, kernel, stride and plane. Setting
> `do_quantization=False` on a float ONNX gives the float datapath. A diff between two such
> captures identifies a field faster, and less ambiguously, than a bit sweep.

> [!NOTE]
> A capture contains a register program and nothing about the memory it addresses. A program
> that matches a capture register for register can still produce incorrect output, because
> the buffer layouts are not in the capture. The depthwise path's two buffer layouts are
> measured on the SoC, in "Depthwise convolution". A program decoded from a `.rknn` file is
> also the *stored* register set. A value the vendor runtime patches in at load time does not
> appear in it.

## Corrections to the published map

The published map is Ga Hing Woo's `RK3576_CNA_MAP.md` (github.com/gahingwoo/linux-rk3576-npu).
It differs from the hardware in four places. A regcmd encoder that takes the published reading of
any of them programs the SoC incorrectly, and no error is reported:

| Offset | Field | Published | Correct | Evidence |
|---|---|---|---|---|
| CORE `0x301c` hi | Output height | Halved | `oh_task-1`, the output rows the task writes, minus one | The `conv2d` capture: `0x27` = 39 for a 40-row output |
| `0x1088`, `0x1070` | Feature-data address | `0x1070` on the ARGB path | `0x1088` on both datapaths | The `conv0` capture's fourth task: `0x1088` is `0x12360`, and `0x1070` is zero |
| `0x1080` | Pad word | Open. A lookup keyed on stride and depthwise | The closed form below | The `conv2d` capture: `0x02020101` |
| `0x103c` hi, `0x1044` lo | CBUF entries | `max(iw/4, ic/2)` | `ceil(iw*ic/64)` | The depthwise capture: 56 at `iw`=112, `ic`=32 |

The output-height field is the number of output rows *this task* writes, minus one. Software
must program CORE `0x301c` as `(oh_task-1)<<16 | (ow-1)`, and the same `oh_task-1` at DPU
`0x4024` / `0x4034` and RDMA `0x5010`.

The feature-data address is `0x1088` on both datapaths, the ARGB first-conv path included.
Register `0x1070` is zero in every capture. The only address register at the same offset as
on the RK3588 is the weight base (`0x1110`).

The pad word `0x1080` has a closed form, four byte fields:

```
0x1080 = pad_right<<24 | pad_bottom<<16 | pad_left<<8 | pad_top
```

The leading pads are the configured padding. The trailing two are the pad consumed by this
task's last window, `max(0, (out-1)*stride + k - (pad_lead + in_extent))`.

The pad word has no configured trailing pad, and an asymmetric pad is programmed through the
output extent. The trailing fields are a function of the output extent and the leading pad.
For TFLite's SAME at an even plane and stride two (`pad_before = 0`, `pad_after = 1`), software
programs a leading pad of zero. It also programs an output extent one larger than the
symmetric formula gives, and nothing has to be materialized in the feature buffer. In
`librocketnpu`, the fields `rocket_conv2d_desc.oh/.ow` name that extent, with zero meaning the
symmetric derivation.

The kernel bounds the trailing pad: a trailing pad of `k` or more is an output row whose
whole window is pad. A zero leading pad against a consumed trailing one is bit-exact on the
int8 direct and depthwise paths [HW sweep, H96 MAX M9]. The float path is not characterized
at that geometry, and `librocketnpu` refuses it there.

The CBUF entries word is `ceil(iw*ic/64)`, the count of 64-byte CBUF granules in one feature
row. The published `max(iw/4, ic/2)` agrees with it only where `ic` is 16 or `iw` is 32
[host-computed]. That covers the captures the published map was fitted to. The depthwise
capture separates the two. At `iw` 112 and `ic` 32 the granule count is 56, and the max form
gives 28. A narrow, deep tile, such as a 1×1 pointwise convolution over a small plane, is in the
region where the two differ.

## Unnamed registers and their RK3588 equivalents

The offsets below are RK3588 registers at a new offset. The published map records several of
them as RK3576-only, with no RK3588 counterpart:

| RK3576 | Function | RK3588 |
|---|---|---|
| `0x1048` | CVT control word, `data_sign<<3 \| cvt_type<<1 \| cvt_bypass`, identical packing | `CNA_CVT_CON0` `0x104C` |
| `0x104C` / `0x1050` | the four CVT scales, two per register (`scale1<<16 \| scale0`, `scale3<<16 \| scale2`) | `CNA_CVT_CON1..4`, one each |
| `0x1054`-`0x105C` | the CVT offsets (non-zero only on the int8 ARGB path, where the captures set them to -128 per channel, and inert there: §"The int8 ARGB input domain") | folded into the same CON1..4 |
| `0x108C` | burst lengths, `weight_burst<<16 \| data_burst`, `0x000F000F` | `CNA_DMA_CON0` `0x1078` |
| `0x103C` hi, `0x1044` lo | CBUF data entries, `ceil(iw*ic/64)`, written twice | `CNA_CBUF_CON1` `0x1044` |
| `0x1084` | the CNA border pad constant | `CNA_PAD_CON1` `0x1184` |
| `0x5024` | base of the DPU shift word, a per-task operand rather than a spare (see below) | the register fields `BS_MUL_CFG.BS_MUL_SHIFT_VALUE` `0x4048` and `DATA_FORMAT.BS_MUL_SHIFT_VALUE_NEG` `0x4010` |

The kernel word has a closed form. Register `0x1024` hi is `((kw-1)<<8) | (kh-1)`, the
width in bits [31:24] and the height in [23:16]: `0x0202` at k=3, `0x0404` at k=5 and
`0x0000` at k=1. The published map gives it as a two-entry lookup.

> [!CAUTION]
> Every vendor capture has a square kernel, so no capture shows which byte of `0x1024`
> contains the width. With the two bytes swapped, the non-square kernels 1×7, 7×1, 1×3 and 3×1
> produce incorrect output in every element region [HW sweep, H96 MAX M9]. Software must
> program the kernel width in bits [31:24] and the height in [23:16].

The OUT_CVT triple is `0x40AC` / `0x40B0` / `0x40B4`: offset, scale and shift, in three
consecutive registers. It is the same requant the RK3588 programs at `0x4080`-`0x4088`
[HW sweep, H96 MAX M9].

## int8 convolution

**Conditions:** H96 MAX M9 (RK3576), mainline 7.1.3, the regcmd encoder driven through
`librocketnpu`, symmetric quantization [HW sweep, H96].

With the RK3576 program, an int8 1×1 convolution, IC=OC=32, 8×8, writes the whole,
correctly sized output: 2048 bytes for a 32-channel 8×8 int8 output. It is the same NC1HWC2
cube the RK3588 uses (C2=16, surface stride `ow*oh` in 16-byte atoms, which is why `0x401c` is
`ow*oh` and not `ow*oh*16`).

> [!CAUTION]
> With the RK3588 register program, the RK3576 writes nothing at all, and the output BO comes
> back untouched. Software must program the RK3576 with the RK3576 encoding.

A k=3 SAME 32×32 int8 convolution, IC=OC=32, is bit-exact against a CPU model across the whole
output, and so is the same convolution unpadded. Its weights are a dense random set, all 9216
non-zero, and its feature tensor spans the full signed int8 range. This result confirms that
each of these is programmed correctly:

- The CNA feature and weight DMA
- The CBUF staging
- The CSC
- The CMAC
- The BS bias add
- The OUT_CVT requant
- The output geometry
- The writer

At IC=OC=32 the RK3576 takes the same weight cube as the RK3588 (`weight_conv_int8`,
oc-group 32 / ic-group 32).

The int8 convolution is bit-exact at every geometry whose channel counts are programmed as
multiples of 32 and whose plane fits the CBUF budget. Over the measured range, the regcmd encoder
reproduces the CPU model bit-exactly over the whole output on every run. The range covers
these axes:

- IC and OC from 8 to 128
- Planes from 8×8 to 128×64
- Kernels k1, k3 and k5
- Stride 1 and 2
- VALID and SAME padding
- Feature tensors that vary on all three axes

Both limits below concern how the operands are described. `librocketnpu` meets both by
construction: the channel counts come from `rocket_rk3576_pad_ic()` / `_pad_oc()`, and the CBUF
budget is the `rocket_hw_rk3576` machine-parameter profile.

## Channel counts as multiples of 32

Software must program the register `ic` and `oc` each as a multiple of 32, the group that
`weight_conv_int8` pads its two channel axes to.

> [!CAUTION]
> A convolution whose programmed channel counts are not multiples of 32 produces incorrect
> output at every geometry [HW sweep, H96 MAX M9]. No capture shows this, because every
> captured geometry already satisfies it. Software must size the feature cube, the output BO
> and the coefficient buffer to the padded counts.

On the input axis, ic of 8, 16, 17 and 48 is incorrect at every kernel and plane size.
Values of 32, 64 and 96 are exact. An ic=17 conv fails although it spans two whole C2=16
surfaces, so the unit is the 32-channel MAC group and not the surface. Padding the feature
cube alone does not help, because every read past `ic` already reads mapped zeros. The
register value is what matters: the padded count, with the cube zero-filled to match, is
bit-exact.

On the output axis, a partial group fails by a different mechanism at k=1 and at k>1:

- At k=1, the DPU writes only output row 0 of the trailing group and leaves the rest of the
  output untouched. The oc values 8, 16, 40 and 48 fail this way, while 24, 32 and 64 are
  exact. The rule that fits k=1 alone is surface parity (`ceil(oc/16)` must be even), which is
  why oc=24 passes there.
- At k>1, the weights are read incorrectly instead, and oc=24 and oc=56 fail although their
  surface count is even. The register `WEIGHT_BYTES` (`0x101C`) is `ic*oc*kh*kw`, which
  describes a smaller cube than the padded one software supplies. In the
  `[kh][kw][oc2=32][ic2=32]` cube, that truncation drops whole `(kh,kw)` planes rather than
  trimming `oc` inside each. At k=1 there is only one such plane, so the shortfall falls on
  the unused `oc2` slots, which is why k=1 is the tolerant case.

Rounding `oc` up to 32 satisfies both.

> [!NOTE]
> At k=1, a partial output-channel group writes row 0 of the trailing group and leaves the
> rest unwritten. The result looks like an output-geometry defect rather than a channel-count
> one. A sweep at one kernel size sees only one of the two mechanisms.

The vendor compiler does not pad. Its `conv2d` capture is ic=16 with weight bytes
`ic*oc*kh*kw`, the unpadded size, so the vendor packs a 16-channel weight group where
`weight_conv_int8` pads to 32. The regcmd encoder matches that program register for register, so the
difference is in the weight cube and not in the registers. Padding is the RK3588 cube's form of
the same conv. The vendor's partial-group packing is the other form, at half the weight bytes
for such a layer.

## The CBUF budget and `0x1040`

The CBUF budget is programmable, through `0x1040`.

> [!CAUTION]
> A conv whose feature plane does not fit the budget produces incorrect output. The DPU still
> writes the whole output, with no IOMMU fault and nothing in dmesg. Software must keep each
> task's feature plane within the budget below.

The corruption is graded. One row past the budget, about half the output is still
bit-exact. The exact share falls with the depth of the overflow until almost none is. No other
RK3576 failure presents this way, so a graded corruption identifies a CBUF overflow.

The budget is one scalar in the CNA's granule unit. The register that the published map
records as the constant `0x10000000` sets it:

```
ceil(iw * ic / 64) granules per feature row  x  the task's input rows  <=  4096 + F
      where F = bits[16:27] of CNA_CBUF_CON0 (0x1040)
```

Every boundary is exact. Each ceiling below is the last granule count that is bit-exact over
the whole output:

| F | ceiling | = 4096 + F? |
|---|---|---|
| 0 | 4096 | yes |
| 63 | 4152-4159 (519 whole rows of 8) | yes |
| 256 | 4352 | yes |
| 512 | 4608 | yes |
| 1024 | 5120 | yes |
| 2048 | 6144 | yes |
| 1280 (256\|1024) | 5152 | **no**, 5376 expected |
| 1536 (512\|1024) | 5152 | **no**, 5632 expected |

The limit is the granule count, independent of rows, width or channels. Widths of 16, 32 and
64 all break at the same granule totals, 4096 at F=0 and 6144 at F=2048. The data budget does
not exceed 6144 granules (384 KiB): F=3056 still measures 6144. [HW sweep, H96 MAX M9]

### Single-bit F values and conditional delivery

Only single-bit F values give their face value. Each power of two measures exactly 4096+F, but
a combination gives less than the sum of its bits. The two combinations tested both measure
5152, whichever second bit is set, so the field is not an integer the CNA adds.
[HW sweep, H96 MAX M9]

> [!CAUTION]
> An F with more than one bit set programs a budget the hardware does not provide. A plane over
> the provided allowance still writes the whole output, with incorrect values and no error.
> Software must use the measured single-bit settings (0, 256, 512, 1024, 2048), rounding a
> deficit up to one of them.

Rounding up costs a little weight headroom the plane did not need.

F=256 and F=512 give their budget only under a condition, and otherwise each gives 4096
granules, the F=0 budget. F=0, 1024 and 2048 are unaffected at every footprint tried on either
path. The vendor's windowed depthwise capture is a k=3 program at F=1024.

> [!CAUTION]
> Where F=256 or F=512 falls back to 4096 granules, a task planned for the larger budget
> overruns its allowance. Its output has the correct size and an incorrect tail: every output
> row past `4096 / entries`. Software must use F=256 and F=512 only where the condition below
> holds.

The failure is a band of window heights, not a ceiling. A planner takes the smallest F that
covers the window. As the window grows, the output is exact while F=0 still covers it. It is
then incorrect across the band of windows that select a fallen-back F=256 or F=512. It is exact
again where the next setting up gives its budget.

One task per height at 160×160 ic = oc = 32 k3 depthwise, 80 granules a row, shows the band.
It is exact at 45-51, incorrect at 52-57 and exact at 58-76.

> [!NOTE]
> A bisection over the window height reports whichever edge of the band it reaches. A
> bisection assumes that a smaller window is never worse, which holds for a capacity bound and
> not for this band. Measure every height.

The condition differs by path. On the depthwise path, no footprint threshold fits, and the
condition is the tap count. A single-tap kernel gets F=256 and F=512 at three channel counts
spanning 32x. A multi-tap kernel never does [HW sweep, H96 MAX M9]. Each row below forces
F=256 and F=512 under one fixed window that F=0 does not cover:

| path | kernel | oc | weight footprint held in the CBUF | F=256 / F=512 |
|---|---|---|---|---|
| direct | 1×1 | 32 | 1024 B = 16 granules | deliver |
| direct | 1×1 | 64 | 2048 B = 32 granules | fall back |
| depthwise | 1×1 | 32 | 64 B = 1 granule | deliver |
| depthwise | 1×1 | 256 | 512 B = 8 granules | deliver |
| depthwise | 1×1 | 1024 | 2048 B = 32 granules | deliver |
| depthwise | 2×2 | 32 | 256 B = 4 granules | fall back |
| depthwise | 3×3 | 32 | 576 B = 9 granules | fall back |
| depthwise | 5×5 | 32 | 1600 B = 25 granules | fall back |

`librocketnpu`'s regcmd encoder does not use F=256 or F=512 on the depthwise path. The fallback
setting is strictly larger and always gives its budget, so declining them costs nothing.

### The direct path's weight-footprint condition

On the direct path, the condition is the weight footprint held in the CBUF, at most 1 KiB
(16 granules). The footprint is the weight slice the CBUF holds for one output-channel group,
`32*ic*kh*kw` bytes, whichever of `ic` and the kernel sets it.
F=256 and F=512 give their budget at 16 granules (`ic` 32, k=1). They fall back at 32
granules (`ic` 64), 64 (`ic` 128), 144 (`ic` 32, k=3) and 400 (`ic` 32, k=5).
[HW sweep, H96 MAX M9]

Where they fall back, the last correct output row is, in every case, the one fed by input row
`4096 / entries`. The fallback budget is therefore 4096 granules.

The quantity is one output-channel group's slice, not the whole resident cube. With the slice
held at the measured 16 granules, F=256 and F=512 both still give their budget at `oc` 64
and 96. The cubes there are 32 and 48 granules. [HW sweep, H96 MAX M9]

`librocketnpu`'s regcmd encoder charges the whole cube, the smaller limit, so its rule is
conservative. Where it declines F=256 or F=512, the next setting up also gives its budget, at
the same CBUF and with no extra submit. Relaxing the rule therefore gives nothing.

Past the threshold, the regcmd encoder rounds up to F=1024, at the same CBUF and with no extra
submits. Where the weight path leaves no room for 1024, it plans a shorter row window: one
more task, with correct output. The regcmd encoder programs a direct F=256 or F=512 only where the
plane is 4097-4608 granules and the weights are at most 1 KiB. The matmul's own row planner is
past that at every K it runs.

**Not characterized:** the threshold between 16 and 32 granules, and F=256 and F=512 on the
packed-image path.

On the direct path, `ic` is padded to a multiple of 32. The slice `32*ic*kh*kw` then moves in
1024-byte steps at every shape that a square kernel and a whole `ic` can build. Only a
non-square kernel falls between them. The packed-image path keeps the direct rule. Inception
V3's 299×299, the widest stem measured, stages 5681 granules and uses F=2048, and a 224×224
stem sits at F=0. [HW sweep, H96 MAX M9]

### The captured words and the shared pool

The two `0x1040` values in the captures are two points on the F scale. The word `0x10000000`
(F=0) gives 4096 granules, and `0x14000000` (F=1024) gives 5120. The second is what the
vendor's windowed depthwise program needs. At 112 wide and ic=32 a row is 56 granules, and its
91-row window is 5096 granules. The window exceeds 4096 and fits 5120 with 24 to spare.

`0x1040` keeps the RK3588 field layout in its low bits, and adds a granule allowance in
bits[16:27], which is reserved on the RK3588 [HW sweep]:

| Bits | Field | Function |
|---|---|---|
| 0-13 | `DATA_BANK[0:3]`, `WEIGHT_BANK[4:7]`, `FC_DATA_BANK[8:10]`, `DATA_REUSE[12]`, `WEIGHT_REUSE[13]` | The RK3588 bank fields, per Mesa's `registers.xml`. Live. The vendor leaves them zero |
| 14, 15 | | Don't-care |
| 16:27 | F | The granule allowance |
| 28 | | Required. A `0x1040` of zero corrupts the conv |
| 29 | | Setting it corrupts the conv |
| 30, 31 | | Don't-care |

> [!CAUTION]
> Setting any single bit of 0-13 on top of a working program corrupts the conv [HW sweep].
> Setting bit 29 or clearing bit 28 corrupts it too. Software must set bit 28 and leave bits
> 0-13 and 29 clear.

The data and weight sides share one pool, and F trades between them. The weight path stages
per output-channel group, so its resident slice is `32 * ic * kh * kw` bytes rather than the
whole cube. A 441 KiB weight cube is therefore bit-exact. A 150 KiB slice is bit-exact at F=0
and breaks at F=2048, and the slice ceiling falls as the data side grows:

| F | data allowance | weight slice ceiling | sum |
|---|---|---|---|
| 0 | 4096 gr = 256 KiB | 175 KiB exact, 200 KiB breaks | ~445 KiB |
| 1024 | 5120 gr = 320 KiB | 125 KiB exact, 150 KiB breaks | ~455 KiB |
| 2048 | 6144 gr = 384 KiB | below 75 KiB | ~450 KiB |

Each +1024 granules of data costs the weight path about the 64 KiB the data side gained. All
three pairs sum to roughly 448 KiB, 14 banks of the RK3588's 32 KiB, of which the captures'
default program gives 8 to data. The bank *count* is an inference from that bank size, and
the granule budget, the trade and the total are measured. [HW sweep]

> [!CAUTION]
> At F of ~3060 or more, the SoC writes no output at all, at any plane size. On the 14-bank
> reading, F there leaves the weight path nothing. The symptom is the same as a wrong
> geometry encoding. Software must keep F below ~3060.

The 8 x 32 KiB in `rocket_hw_rk3576` is therefore the **default data allocation**, and the
physical CBUF is larger. Raising F gives up to 1.5x the feature capacity, at the cost of
weight-slice headroom that software must then respect. The published map records `0x1040` as a
constant, and it is a per-task setting. A regcmd encoder that copies the constant uses the vendor's
choice for a full-plane task.

The pool boundary is sharp enough to plan against. A slice of exactly 192 KiB (6 banks, what
the 14-bank model leaves beside F=0) is bit-exact at ic=1536 k=2, and a 196 KiB slice at
ic=1568 breaks. [HW sweep]

### Per-task planning of F

`librocketnpu`'s regcmd encoder plans F per task, in `rocket_rk3576_cbuf_f()`. It picks the lowest
setting that gives its budget and covers the plane. On the direct path, F=256 and F=512 count
only where the resident weight cube is at most 16 granules. On the depthwise path they never
count. Where the plane needs more than the data cap, or the setting would leave the weight path
too little, the regcmd encoder refuses the task. The recourse, a shorter row window or an ic split, is
the caller's to choose.

The tiler-facing half is `rocket_rk3576_max_task_rows()`: the tallest window one task can
take at the highest setting the weight slice leaves room for. With per-task F, every plane
between 4096 and 6144 granules is bit-exact rather than silently corrupted. The flag
`ROCKET_RK3576_CBUF_F` forces F and bypasses both checks.

## The weight slice and the output-channel group count

The resident weight slice limits how many output-channel groups produce correct output. The
pool limits above are measured at one output-channel group.

> [!CAUTION]
> A conv driving several output-channel groups loses the trailing ones well before the slice
> reaches what the pool leaves it. It loses them one at a time as the slice grows. The leading
> groups come back bit-exact and the rest incorrect, with the whole output written, no fault
> and nothing in dmesg. The symptom looks like an output-channel defect rather than a capacity one.
> Software must keep the slice within the limit for its group count, below.

The governing quantity is the resident weight slice `32 * ic * kh * kw`, one output-channel
group. The same slice behaves identically whichever `(ic, kh, kw)` produces it. Both
`ic=512 k=3` and `ic=4608 k=1` give 144 KiB and are exact on all four groups, while
`ic=192 k=5` and `ic=544 k=3` sit near 150 KiB and lose two.

The number of groups that are bit-exact at each slice:

| slice | 144 KiB | 145 | 146 | 147 | 148 | 152 | 156 | 162 |
|---|---|---|---|---|---|---|---|---|
| groups exact | 4 | 3 | 3 | 3 | 3 | 2 | 2 | 1 |

**Conditions:** F=0, a 4×2 plane at oc=128 (four groups), `ic` swept at k=1, cross-checked at
k=3 and k=5. H96 MAX M9, measured 2026-07-25 [HW sweep].

Within one program, every output-channel count is exact at **`ic*kh*kw <= 4608`** (a 144 KiB
slice). The boundary falls as the group count rises, roughly as though each additional group
costs a small fixed staging allowance on top of the slice. A weight path that stages the next
group while computing the current one fits that shape [hypothesis].

Past that limit, `rocket_rk3576_cbuf_f()` refuses the program rather than running it. It checks
against a small measured table keyed on the group count: 144 KiB at four groups or more, 148 at
three, 156 at two. The recourse is an ic split, which is the caller's to choose. A single group
is left to the pool check alone, which already falls on the measured single-group boundary
(175 KiB is exact, 200 KiB is not, and the pool reaches its limit at 192 KiB).

### Output-channel tiling under the group cap

The group count is software's choice, so the limit is one too. The table above bounds a
program, not a shape: its group count is what one program drives, not what the convolution
needs. Splitting the output channels across submits lowers the group count per submit, which
raises the slice the SoC takes. The table then maps the slice a shape needs to the most output
channels one submit can take:

```
slice <= 144 KiB   ->  no constraint      (four groups or more)
slice <= 148 KiB   ->  oc tile 96         (three groups)
slice <= 156 KiB   ->  oc tile 64         (two groups)
otherwise          ->  oc tile 32         (one group; the pool check governs)
```

The four-group limit puts `ic=576 k=3` (162 KiB) and `ic=5184 k=1` (162 KiB) well out of
range. Split into four tiles of 32 channels, both are bit-exact
[HW sweep, H96 MAX M9, 2026-07-27]. `ic*kh*kw <= 4608` is therefore the single-program limit.
With output-channel tiling, the limit is the CBUF pool at one group:
**`32*ic*kh*kw <= 175 KiB`**, that is `ic*kh*kw <= 5600`.

`rocket_conv2d_int8_rk3576()` plans exactly that split, at one submit per tile. One shape is
still refused: `ic=256 k=5` needs 200 KiB, which is past the pool at a single group. The only
recourse there is an input-channel split, which the on-chip requant forecloses: the int8
partials cannot be summed without quantizing each one. That shape needs the int32-output
writer.

The limit is the slice, and neither the cube size nor the group count on its own. The shape
`ic=448 k=3 oc=128` has a 126 KiB slice under a 504 KiB cube and is exact on all four groups.
The shape `ic=256 k=5 oc=32` has a 200 KiB slice under a 200 KiB cube and does not fit at all.
Eight groups, `oc=256` at `ic=64`, are exact.

## The DPU epilogue and the operand DMAs

The DPU stages after the MAC array produce correct output, and the bias path is bit-exact with
the coefficient-buffer layout below. With features and weights zero and a per-channel bias, the
output matches the CPU model exactly across all 32 output channels. The OUT_CVT offset, scale
and shift move that output exactly as predicted.

The OUT_CVT requant rounds to nearest rather than truncating. A border whose exact value is
-10.0006 comes back as -10, where an arithmetic shift gives -11. Adding half an LSB before the
shift, as `(acc*scale + (1<<(shift-1))) >> shift`, reproduces every pixel of the bias and
OUT_CVT measurements above exactly. A CPU model without a rounding term is off by one wherever
the fraction crosses a half. [HW sweep, H96]

That half-up form differs from the hardware only at an exact tie, and the measurements above
contain none. Every vendor program and `librocketnpu` leave `0x40B4` bit 30 (`CVT_ROUND`)
clear. A tie then rounds half to even [HW sweep, H96 MAX M9, 2026-09-23]. The
tie rule is in
[../encodings/out-cvt-converter.md](../encodings/out-cvt-converter.md).

All four operand DMAs are active and reach their programmed addresses. Each base below,
pointed at an unmapped IOVA, makes the job fault with its own IOMMU status word, while an
unchanged run is clean:

- The weight base, `0x1110`
- The feature base, `0x1088`
- The output base, `0x4018`
- The bias base, `0x5020`

The fault appears as two `rk_iommu ... Enable stall request timed out` lines. [HW sweep, H96]

> [!CAUTION]
> The job immediately after an unmapped-IOVA fault produces correct output. But a session that
> ends with these faults can leave the SoC in a state where the weight loader never starts
> again. Every later job then completes cleanly, with no timeout and no new dmesg line, and
> writes a bias-only output. The state survived 7 hours of idle and a full `rmmod rocket` /
> `modprobe rocket` cycle [HW sweep, H96 MAX M9, 2026-07-25]. Only a reboot cleared it.
> Software must not trust a negative result taken after these faults until the board is
> rebooted and a known-good conv produces correct output again.

That state was measured on a `rocket` from before interface version 1.0, with no settle
recorded between `rmmod` and `modprobe`.

**Not characterized:** this state on `rocket` 1.6.0, and whether the reload with an
eight-second settle that clears the IOMMU wedge (§"The IOMMU wedge") clears it.

The feature strides are bit-exact along rows, along columns and across channel *groups*. Each
axis is varied with the other two held constant. The three axes confirm, in turn, the line
stride, the C2=16 channel atom inside a row, and the CBUF surface/group stride. [HW sweep, H96]

| Register | Value | Unit | Function |
|---|---|---|---|
| `0x1090` | `iw*4` | 4-byte | Line stride |
| `0x1094`, `0x1098` | `iw*ih` | 16-byte | CBUF surface/group stride |

### The coefficient buffer

The bias/coefficient buffer at `BS_BASE_ADDR` (`0x5020`) has the structure of the vendor's own
coefficient buffer. It is made of groups of 64 bytes covering 8 output channels each, with
these fields:

| field | type | offset in group |
|---|---|---|
| `A[oc]`, per-channel bias term | int32 | `(oc%8)*4` |
| `B[oc]`, weight-zero-point correction | int16 | `32 + (oc%8)*2` |
| `C[oc]`, per-channel multiplier | int16 | `48 + (oc%8)*2` |

Output channel `oc` is in group `oc/8`.

> [!CAUTION]
> The coefficient buffer is not a flat per-OC int32 array. Read as this structure, a flat
> int32 array gives a term to only the first 16 channels, on alternating channels, at 1024x the
> intended magnitude. That pattern is an artifact of the layout, not of a 2-byte operand DMA.
> Software must pack the coefficient buffer in the grouped layout above.

The BS stage adds `A` and then multiplies by `C`: the output is
`(acc + A[oc] + B[oc]*sum(x)) * C[oc]` [HW sweep, H96].

> [!CAUTION]
> A bias quantized in the accumulator domain is multiplied by the per-channel gain, and
> software must not pre-divide it by that gain. With the order backwards, the BS stage scales
> the bias by the wrong channel gain, and the output has incorrect values rather than a fault.

The BS result saturates at int32, after the shift word, not before it. With the shift word at
zero, the product saturates and does not wrap [HW sweep, H96]. With no shift,
`C[oc] <= INT32_MAX / max(abs(acc + A))`, which falls as the layer's fan-in grows.

Under a nonzero shift the multiply is held wide. At a shift of 14 and `C = 16384`, products from
2.6e9 to 3.3e13 are exact [HW sweep, H96 MAX M9, 2026-09-23]. The bound is therefore
`((acc + A)*C) >> s <= INT32_MAX`, and with a shift C can use its int16 field at any fan-in.

`C` is a live linear per-channel multiplier. Every one of 32 channels reads its own `C` at 32
distinct values. At `C=1` the datapath is bit-exact, and `C=4` scales by exactly 4.
[HW sweep, H96]

> [!CAUTION]
> At `C=0` the DPU writes an empty output of the correct size, whatever the CNA and the MAC
> array computed. It does so even for a conv with no bias at all. An all-zero coefficient
> buffer produces this output, and it is indistinguishable by inspection from a wrong geometry
> encoding [HW sweep, H96]. Software must program a non-zero `C` for every output channel.

`librocketnpu` packs the buffer with `rocket_rk3576_pack_coeff()`. It pins the `B` term to 0,
which is what symmetric quantization needs.

> [!CAUTION]
> How `C` is read depends on the precision, so `C = 1` is not portable across it. A float
> program reads the field as fp16, where the integer `1` is the denormal 6e-8. It underflows the
> output to empty, the same symptom as `C=0`. Software must program `C` in a float program as
> fp16, where `1.0` is `0x3C00`. See
> [the fp16 datapath](#fp16-direct-convolution).

### The B term is added: `acc + B*sum(x)`

The BS stage adds `B*sum(x)` to the accumulator [HW sweep, H96 MAX M9, measured 2026-07-27].
Here `sum(x)` is the sum of the input elements the output contracted. The model
`acc + B*sum(x)` explains every output of every direct shape in the validation set, at
`B` = 1, 40, 127, -37 and -128.

> [!CAUTION]
> A weight zero point is programmed negated. An asymmetric weight is
> `w_true = w_stored - wzp`, and its correction is `-wzp*sum(x)`. Software must program
> `B = -wzp`. A reversed sign gives a plausible output with a bias-shaped error and nothing
> that faults.

In `librocketnpu`, `rocket_rk3576_pack_coeff_asym()` names its parameter `b_term` for that
reason: the caller passes `B = -wzp`.

## The feature domain and the padded border

The feature domain is signed int8 and needs no centering. A k=1 conv whose output plane sweeps
every int8 value once, from -128 to 127, reproduces the CPU model over the whole range. Its
input is raw signed bytes, with no `+0x80` applied anywhere. [HW sweep, H96]

The DPU epilogue is independently exact for negative accumulators. Negative per-channel biases,
which the MAC never touches, are bit-exact on all 32 channels. An all-zero coefficient buffer
empties the output whatever the features are, so an empty output on negative features comes
from the coefficient buffer. [HW sweep, H96]

The padded border is exact. The k=3 SAME 32×32 conv with dense random weights reproduces the CPU
model over the whole output, border included. A border driven with a -128 pad tap against zero
features is exact too. The border pad constant `0x1084`, the pad word `0x1080` and the window
geometry are all correct. A border defect on this path is the same coefficient-buffer artifact.
[HW sweep, H96]

## The row window and its three registers

A row window splits a convolution by input rows, so that a feature plane larger than its CBUF
allowance can run. A plane over its allowance produces incorrect output, not a slower one.
Each task reads a window of the plane's input rows and writes the output rows that window
supports. The task's register program describes the full plane alongside its window. Three
registers take a second value only on a windowed task: `0x40B8`, `0x1018` and `0x1038`.

**Conditions:** H96 MAX M9, int8 direct convolution. Each task refetches its whole window
against a CBUF base of zero.

### Window addressing

Software offsets each task's two base addresses by a plain row stride: the feature base by
`iy0*iw*16` bytes and the output base by `oy0*ow*16`. Here `iy0` is the window's first input
row and `oy0` its first output row. The cubes are NC1HWC2 with a 16-byte channel atom. The CNA
takes the DDR group stride from the full plane (`0x1094` = `iw*ih_full` by default). One base
plus a row offset therefore addresses that row of every channel group. The vendor's sliced
capture programs the feature base `0x1088` the same way [source-confirmed: RKNN-Toolkit2
capture].

The CNA uses `0x1094` as programmed, and honors any value at or above the plane
[HW sweep, H96 MAX M9]. A task can therefore read another task's output whose channel groups
are further apart than its own plane would put them. With a padded group stride, `0x1098`
stays at `round4(iw*fetch_rows)`, and the output is bit-exact. The CNA does not read `0x1098`
as a second DDR stride [HW sweep, H96 MAX M9].

### The windowed registers

The three registers that take a second value on a windowed task [HW sweep, H96 MAX M9]:

| Offset | Bits | Function | Measured values |
|---|---|---|---|
| `0x40B8` | all | DPU channel-group jump, `ow * (2*oh_full - oh_task)` | The only form bit-exact at 32-, 16- and 8-row windows of a 64-row plane |
| `0x1018` | 30 | Live, and required. Software must set it | With bit 30 clear, the whole output is incorrect |
| `0x1018` | 15:0 | Ignored on the direct int8 path | `0x0404`, `0x0505`, `0x040b` and `0x0000` are all bit-exact, windowed and un-windowed alike |
| `0x1018` | 31 | The reuse form's retain flag (§"CBUF row reuse across a windowed sequence"). Clear on a refetching task | Not characterized |
| `0x1038` | 31 | Live. Clear on a refetching task, and set on every continuation task of the reuse form | On a refetching task, `0x80000010` corrupts the output |
| `0x1038` | other bits | Ignored | `0x07`, `0x010e` and `0x10` are all bit-exact |

The channel-group jump is one full destination plane plus the rows of it the task does not
write. The DPU writer walks the task's rows, then adds the jump to reach the same rows of the
next channel group. On a task that covers the whole plane, the jump reduces to `ow*oh_full`,
which is what both full-plane vendor captures contain. A full-plane capture therefore shows
the register as a constant.

The jump is bit-exact over this range, measured 2026-07-25 [HW sweep, H96 MAX M9]:

- Output-channel groups 2, 4, 8 and 16
- Kernels k1/k3/k5, stride 1 and 2, VALID and SAME
- Wide, tall and ragged planes, at ic 32/64/128
- Windows from 3 to 16

> [!CAUTION]
> A windowed task whose `0x40B8` leaves out the rows it does not write places every later
> channel group short of its rows. With one group (`oc=32`) the output is exact, and with two
> (`oc=64`) half of it is wrong. The defect therefore looks
> like an output-channel fault rather than a windowing one. Software must program `0x40B8` =
> `ow * (2*oh_full - oh_task)` on every windowed task.

> [!NOTE]
> The windowed `0x1018` value does not stop the DPU on an un-windowed task. A single task that
> writes nothing there is the cold-start defect (§"The cold-start wall"). Re-run the task
> before attributing the failure to `0x1018`.

A windowed convolution is bit-exact against the CPU model, over the whole output, at every
shape measured [HW sweep, H96 MAX M9]. The shapes include 112×112 and 224×224 at ic=32, the
geometries with no single-task plan at all (6272 and 25088 granules against a 6144 cap). They
also include forced-cap splits from 2 to 16 windows across ic/oc of 32/64/128, k1/k3/k5,
stride 1 and 2, VALID and SAME.

### Cost per window

**Conditions:** H96 MAX M9, measured 2026-07-25, on a `rocket` with the cold-start defect fixed
and without patch 0012. The output is 64×64 ic64 oc64 k3 SAME, cut into progressively more
windows and submitted with no inter-task gap.

| windows | 2 | 4 | 5 | 11 | 32 |
|---|---|---|---|---|---|
| wall-clock time | 2.5 ms | 4.8 ms | 6.0 ms | 13.0 ms | 38.7 ms |

The cost is flat at ~1.2 ms per window [HW sweep, H96 MAX M9]. A split is therefore linear in
windows, and the fewest windows that fit cost least. Almost all of the 1.2 ms is
submit-and-wait.
Emitting a window's regcmd on the host takes 0.02-0.05 ms, and a single-task conv costs 1.2 ms
whether its output is 2 KB or 1.6 MB.

The fixed per-task cost is the driver's completion poll rather than NPU execution. The RK3576
has no maskable completion IRQ, so `rocket` polls on an hrtimer, and the per-task cost follows
the poll interval almost exactly. The DPU's writes drain after `PC_DONE` fires. Retiring on
the DPU's own completion instead of `PC_DONE` is what makes a shorter interval safe. The patch
`patches/rk3576/npu/0012` retires on the DPU's completion and takes the period to 50 us, and
0011 removes the IOMMU half of the hazard. With 0012 the fixed per-task cost falls to the
per-submit time in §"The per-submit floor is the driver's completion poll", below.

### Software support

In `librocketnpu`, the planner `rocket_rk3576_plan_rows()` lays the window sequence out, and
the regcmd encoder takes the window in `conv_params_t` `ih`/`oh` against `ih_full`/`oh_full`. The
regcmd encoder emits the vendor's values for all three registers, so its programs stay
byte-identical to the captures.

The planner spreads the output rows evenly over the fewest tasks that fit, rather than taking
greedy maximum windows. A greedy pass leaves a ragged tail: a 112-row plane at a 109-row cap
comes out 109+2+1 instead of two windows of 56. Every extra task costs a submit.

## The group stride on a single-task plan

Every task needs the DDR channel-group stride `0x1094` set from the full plane, including a
task whose row window is shorter than the plane. A single-task plan can have such a window.
Any stride greater than 1 whose output does not consume the plane leaves trailing input rows
unread. A 32×32 k1 s2 VALID conv therefore runs as one task over 31 of its 32 rows. With the
stride taken from the window, `0x1094` comes out `iw*31` instead of `iw*32`. Every channel
group past the first is then read at the wrong offset [HW sweep, H96 MAX M9].

> [!CAUTION]
> A `0x1094` taken from the task's window gives an output of the correct size that bears no
> relation to the input. There is no fault and no dmesg line, and the output looks identical
> to that of a regcmd encoder with broken geometry fields. Software must set `0x1094` from the
> full plane on every task, including the only task of a single-task plan.

In `librocketnpu`, a single-task plan needs `ih_full` and `oh_full` set from the plane, the
same as a split plan. The planner `rocket_rk3576_plan_rows()` can return one task whose row
window is shorter than the plane. With `ih_full` left at 0, the regcmd encoder takes the group
stride from the window. A caller that sets `ih_full`/`oh_full` only when the plan splits
therefore mis-programs the stride. The defect appears only once a caller sends every conv
through the planner, which is what a tiler does. Software must set `ih_full`/`oh_full` from
the plane on every task.

## The geometry fields are narrower than their register halves

Each geometry extent is programmed into a 16-bit register half or a whole register. The
RK3576 computes with fewer bits [HW sweep, H96 MAX M9, 2026-09-26]. The field widths, and what
a task past each one writes:

| Extent | Field | Past it |
|---|---|---|
| Width, input or output | 13 bits, exact at 8192 | `W & 0x1FFF` columns written, the rest wrong |
| Output channels | 13 bits, depthwise exact at 8192 | 8224 retires at the backstop, unwritten |
| Kernel height and width, `0x1024` | 5 bits, exact at 32 | 33 wrong from the first element |
| Stride, `0x1014` | 3 bits, exact at 7 | 9 runs as 1 and 14 as 6 |
| Task rows | at least 13 bits | the CBUF caps a task at 6144 rows first |
| Input channels | at least 15 bits | the matmul form contracts K 16384 through it |
| Plane strides | whole words | exact at 409600 elements |
| PPU rows and channels | 13 bits | 8300 rows wrong from row 107, 8208 channels from channel 16 |

**Not characterized:** which of the input and output width registers holds the 13-bit field,
and strides 8 and 16, which program 0.

> [!CAUTION]
> A width, kernel, stride or PPU extent past its field wraps, and the task writes an output of
> the correct size with incorrect values. The task completes normally. A depthwise
> output-channel count past its field leaves the output unwritten, and the task retires at the
> driver's 125 ms backstop on `rocket` 1.6.0. Neither reports an error. Software must keep
> every extent within the field widths in the table above.

> [!CAUTION]
> A wrapped width also breaks the next valid job in the same process. Software must refuse
> these extents before the device sees one. Recovery after the job does not protect the next
> one.

After an 8300-wide task, an 8192-wide job returned 259665 of 262144 elements wrong, with no
error reported [HW sweep, H96 MAX M9]. A 2048-wide one returned 64961 of 65536 wrong. The
mechanism is not decoded.

`librocketnpu` refuses every extent past its field, at claim time and in the regcmd encoder. Nothing
in `librocketnpu` tiles a convolution's width on the RK3576.

### The fields the vendor compiler checks

RKNN-Toolkit2 checks each field it emits against its width, and prints each one that
overflows, as `REGTASK: ... target: f2, offset: 0x1090, shift = 0, limit: 0x3fff`
[source-confirmed: RKNN-Toolkit2 2.2.0 and 2.3.0]. For the RK3576 it prints target `f2` and
these fields:

| Offset | Field | Printed by |
|---|---|---|
| CNA `0x1090`, the line stride | [13:0], limit `0x3FFF` | a stride-2 1×1 conv 16400 wide |
| CORE `0x3020` | [12:0] | a depthwise conv of 8400 channels |
| DPU `0x402C` | [12:0] | the same |
| DPU_RDMA `0x5014` | [12:0] | the same |

The three channel fields agree with the 13 bits in the table above. The CNA line stride does
not. The regcmd encoder writes `iw*4` there, and the row planner gives 2-row tasks at `iw` 4096
and 6000. Those tasks program `0x4000` and `0x5DC0`, past the 14 bits, and are exact
[HW sweep, H96 MAX M9, 2026-09-27].

> [!NOTE]
> A field width RKNN-Toolkit2 prints is the compiler's limit. Treat a printed width as a bound
> to test on the device, not as one measured.

**Not characterized:** fields the compiler does not overflow. An axis it tiles, splits across
cores or sends to the CPU prints nothing. On both SoCs that was width, height, input and
output channels, and a matmul's K and M.

## The first-conv ARGB sub-encoding

A convolution whose input is a packed image runs on its own CNA datapath. The image has 3 or 4
interleaved bytes per pixel rather than an NC1HWC2 cube. About sixteen registers are packed
differently from the normal program. A vision model's stem runs on the RK3576 only on this
datapath. The normal path needs `ic` a multiple of 32 (§"Channel counts as multiples of 32"),
and an image has 3. The encoding is transcribed from vendor captures
[source-confirmed: RKNN-Toolkit2 captures].

The RK3588 has the same datapath. Mesa drives it as `CNA_CONV_CON1`
`NONALIGN_DMA | GROUP_LINE_OFF | ARGB_IN(8)` for a 1-channel input. Register `0x100C` is one
of the few geometry registers the RK3576 does not re-pack. On both SoCs, `GROUP_LINE_OFF` is
bit 29 and `ARGB_IN` is bits[15:12]. The captured `ARGB_IN` of `0xA` sits exactly where
Mesa's 1-channel `0x8` does, one step per extra image channel, so the field is
`8 | (image_channels - 1)`. The `CONV_MODE` field is 6, a third value beside direct (0) and
depthwise (1).

The low field of `0x100C` depends on the precision: 6 at int8 and 2 at fp16. The found int8
ARGB programs contain `0x2000a006` and the float ones `0x2020a122`. Bit 2 clears on the float
path, while `GROUP_LINE_OFF` and `ARGB_IN` stay where they are. The `ARGB_IN` field follows
the channel count as `8 | (ic-1)`: `0x8` at 1 channel, `0xA` at 3 and `0xB` at 4. Mesa's
1-channel `0x8` is therefore the same field.

The CNA reads the packed row straight out of DDR. The CVT, bypassed on every other layer,
expands each pixel to four int8 lanes. The captures program it with a per-channel Q14 unity
scale and an offset, and the offset is inert on the RK3576 (§"The int8 ARGB input domain"). The
kernel's horizontal extent is then folded into the channel axis: the conv the MAC sees is
`kh x 1` over `4*kw` channels. An ARGB program therefore contains two disagreeing channel
counts, the sub-encoding's clearest signature in a capture.

| Register | Normal path | int8 ARGB path |
|---|---|---|
| `0x100C` | `proc<<7 \| in<<4 \| conv_mode` | `GROUP_LINE_OFF \| ARGB_IN(8\|(ic-1))<<12 \| 6` |
| `0x1020` weight elems | `ic*kh*kw` | `kh * round16(4*kw)`, a 16-byte row per kernel row |
| `0x101C` weight bytes | `ic*oc*kh*kw` | `oc * kh * round16(4*kw)` |
| `0x1030` hi | `ic*kh*kw*2` | `weight_elems * 2` (same rule, folded count) |
| `0x1028` lo | `ic-1` | `4*kw - 1`, the folded count |
| `0x107C` | `ic-1` | `image_channels - 1`, the DMA's real count |
| `0x1044` lo, `0x103C` hi | `ceil(iw*ic/64)` | `iw*4/64 + 1` |
| `0x1048` CVT | `0x0B`, bypassed | truncate 14 per live channel, bypass clear |
| `0x104C`/`0x1050` | scales = 1 | `0x4000` (Q14 unity) per live channel |
| `0x1054`-`0x105C` | 0 | -128 per live channel in the captures. Inert |
| `0x1084` pad const | `zp - 0x80`, already centered | the stored zero point, `zp + 0x80` for a uint8 `zp`, one byte per channel |
| `0x1090` line stride | `iw*4` (a cube row in 4-byte words) | `iw*ic/16` (the packed row in 16-byte atoms) |
| `0x1078` hi | `iw-1` | `iw*ic/16 - 1` |
| `0x1094`, `0x1098` | plane / task surface strides | both `line_stride * ih` |
| `0x118C` | `(iw-1)<<16 \| (ih_full-1)` | `(entries-1)<<16 \| (entries-2)` |
| CORE `0x3018` | `0x10000001` | `0x10000081` |

Everything past the CNA, including `0x40B8`'s channel-group jump, is programmed as on the
direct path. The ARGB datapath is a CNA sub-encoding, not a second pipeline.

On this datapath, software must:

- Offset a row window's feature base in packed-image units. The feature buffer is a packed
  image, `iw*ic` bytes per row.
- Write the image in two's complement and program the pad constant `0x1084` as the stored
  zero point, one byte per image channel. The CNA reads each packed byte as a signed int8,
  and the CVT offset does not center it (§"The int8 ARGB input domain").
- Program an `iw` that is a multiple of 16. The DDR row stride is counted in 16-byte atoms,
  and the CBUF row in 64-byte granules. Both are whole only at such an `iw`.

The `+1` on the CBUF row is the fold's lookahead. The row is the image row expanded to 4 lanes
per pixel (`iw*4` bytes, `iw/16` granules), plus one granule. The extra granule is there
because the last output column reads `kw` pixels and reaches past the granule its own pixel
sits in. Both captured widths have exactly `+1`.

On this datapath `0x118C` is not a plane extent. An ARGB program contains no plane height
anywhere, because the packed image is a single surface. `0x1094` and `0x1098` both hold the
task's own rows, and there is no channel-group stride to describe.

> [!NOTE]
> A normal-path decode of an ARGB capture reads a meaningless full-plane height out of
> `0x118C`.

Every found ARGB capture is `kw=3`, where `4*kw` and a constant 12 fit the programmed channel
count equally. Programmed as `4*kw`, the int8 first conv is bit-exact at `kw` 3, 5 and 7
[HW sweep, H96 MAX M9], which separates the two. The float program does not fold, and its
manufactured captures cover `k` 1, 3, 5 and 7 (§"The fp16 first-conv program").

### The ARGB weight cube

The int8 register program declares the ARGB weight cube as `oc * kh * round16(4*kw)` bytes,
one 16-byte row per kernel row per output channel. A capture contains the register program and
not the weight BO, so it shows the cube's size and stride and nothing about its byte order.
The fp16 and int8 cubes order their bytes differently.

The fp16 cube's layout [source-confirmed, RKNN-Toolkit2 rk3576 float build]:

```
slot(oc, c, kh, kw) = (oc/16) * (KH*KW*64)
                    + kh * (KW*64)
                    + kw * 64
                    + (oc%16) * 4
                    + c
```

In this layout:

- Each weight sits in a sixteen-bit slot.
- Each (output channel, tap) has four lanes. Lane `c` holds image channel `c`, and any lane
  past `ic` is don't-care.
- Output channels are interleaved in groups of sixteen inside one tap.
- The tap axis is kh-outer.

The 64 is 16 output channels times 4 lanes. The term `4*kw` is not rounded up to 16 here,
where the int8 cube's row is. The fp16 program declares `oc * 4*kh*kw * 2` bytes, which is
this layout's size. At `oc=16 k=3` the cube is 576 halfwords, and the program declares 1152
bytes [source-confirmed, RKNN-Toolkit2 rk3576 float build]. The layout holds
at `ic` 1/3/4, `k` 1/3/5/7 and `oc` 16/32/48/64, at four widths, three heights and both
strides [source-confirmed, RKNN-Toolkit2 rk3576 float build].

The int8 cube is a different object from the fp16 cube in every axis but the lanes
[HW sweep, H96 MAX M9]:

```
byte(oc, c, kh, kw) = (oc/32) * (KH * R * 32)
                    + kh * (32 * R)
                    + (oc%32) * R
                    + kw * 4
                    + c              R = round16(4*KW)
```

In this layout:

- Each weight sits in one byte.
- Output channels are grouped by thirty-two, not sixteen.
- The tap row sits outside that group, where the float cube puts the whole tap axis
  outermost.
- The tap column is folded into the same `R`-byte row as the four lanes. Of that row,
  `4*KW` is live, and the rest is padding the DMA still fetches.
- Lane `c` holds image channel `c`.

The map is exact over every live byte at `oc` 32 and 64, `k` 3/5/7 and `ic` 3 and 4
[HW sweep, H96 MAX M9]. It is also translation-invariant.

### The int8 ARGB input domain

On the int8 path the CVT's offset is inert, and the packed byte is a plain signed int8
[HW sweep, H96 MAX M9]. Driven on the RK3576, `0x1054`-`0x105C` do nothing. An image written
at `raw = s + (zp + 0x80)` comes back with the raw byte read as a signed int8 and no
subtraction at all. The byte `0x80` reads -128, `0xC0` reads -64 and `0xFF` reads -1. Every
int8 ARGB capture programs the same offset, -128 on each live channel, so no capture shows
whether the CVT applies it.

> [!CAUTION]
> The CVT does not subtract the zero point in `0x1054`-`0x105C` [HW sweep, H96 MAX M9]. The
> CNA reads each packed byte as a signed int8. Software must write the image in two's
> complement and fold the input zero point into the coefficient group's `A` term, as on the
> direct path. Software must still program the border constant as the stored zero point, so
> that the pad rows and corners feed a true value of zero. The pad columns do not read the
> constant and feed 0 whatever it holds [HW sweep, H96 MAX M9, 2026-08-02]. At a non-zero
> input zero point, software must materialize them ([rk3576.md](rk3576.md) §"Pad columns on
> the packed path").

### Geometry bounds of the int8 first conv

The int8 first conv has four geometry bounds. None of them appears in any capture, because
every captured first conv is a 3×3 stride-2 SAME convolution that satisfies all four. None
raises an error when violated. Two give an output of the correct size with incorrect values,
and two leave the output untouched.

> [!CAUTION]
> At `pad_left = 0` the DPU writes nothing at all, and the output buffer is left untouched.
> Software must program a non-zero left pad in bits[15:8] of `CNA_PAD_CON0` (`0x1080`).

The left pad decides it alone, at every plane, stride, kernel and channel count measured.
Forcing `0x0100` into a zero-pad program makes that same program write, and forcing `0x0000`
into a working one stops it. The `pad_top` field in bits[7:0] does neither. A TFLite-style
SAME 3×3 stride-2 stem therefore does not run, and an ONNX-style symmetric one does. TFLite
puts the odd pad byte on the trailing edge and leaves `pad_left` at 0.

> [!CAUTION]
> An output width other than `iw/stride` writes a full, correctly sized output that is
> sheared. Software must program the output width as `iw/stride`.

The tap a weight byte falls on drifts one output column per output row, exactly as a
row-stride mismatch does. The output shears for a narrow `ow` and an over-padded wide one
alike.

> [!CAUTION]
> An output width that is not a multiple of 16 writes output row 0 exactly and every row after
> it wrong. Software must program the output width as a multiple of 16. An `iw` that is a
> multiple of 16 does not imply it.

At `ow` 24 and 56, output row 0 is exact and every row after it is wrong. Both come from an
`iw` that is a multiple of 16. The widths `ow` 16, 32, 48, 64, 80, 96 and 112 are all exact.
Taken with the previous bound, `iw` must be a multiple of `16*stride`. The direct path has no
such rule, and this one comes with the channel fold.

> [!CAUTION]
> At `ic=1` the DPU writes nothing at all, and the output buffer is left untouched. Software
> must program one image channel as two, with a second interleaved channel of zero samples
> against zero weights.

The feature DMA's row width forces the second channel, not the mode word. Bits[31:16] of
`0x1078` contain `line_stride - 1`, which is right from `ic=2` up. Raising that field alone
revives the write at every plane, but nothing is exact, because the DMA then reads past the
packed row. At `iw=64` (4 granules) the field must reach 4 for the write, and at `iw=128`
(8 granules) 8. The best exactness any value reaches is about 40% of the output
[HW sweep, H96 MAX M9, measured 2026-07-28].

With the second channel, `ic=1` is bit-exact at 64×64, 128×128 and 224×224, over the same
shapes as `ic=2` [HW sweep, H96 MAX M9, measured 2026-07-28]. The pad leaves the arithmetic
untouched. The MAC term is zero because the weight is, and `sum_w` is unchanged, so the
coefficient `A` is too. The asymmetric `B` multiplies a sum of raw samples that gains only
zeros, so the zero-point fold keeps the caller's tap count. The cost is one byte per pixel of
host packing and a doubled feature read.

The mode word's `ARGB_IN` nibble is `8 | (ic-1)`, so `ic=1` is also the one value leaving its
low bits clear. The nibble does not control the write [HW sweep, H96 MAX M9, measured
2026-07-28]. Forcing it to any of the working values leaves the output untouched.

The output-channel count follows the direct path's multiple-of-32 rule (`oc=16` writes
nothing). One int8 program delivers 64 output channels, where the float first conv has a
32-channel per-program cap. Above that, software splits the output channels, as on the float
path.

### The fp16 first-conv program

The fp16 first conv is a separate program from the int8 one. The seven fields below move with
the precision. The first two change the shape of the datapath rather than a width
[source-confirmed, RKNN-Toolkit2 rk3576 float build]:

| | int8 | fp16 |
|---|---|---|
| mode nibble (`0x100C` low) | 6 | 2 |
| programmed channels (`0x1028` low) | `4*kw`, the kernel width folded into the channel axis | `ic`, the image's own count |
| CBUF granules per row (`0x103C` hi) | `iw*4/64 + 1` | `iw*4*2/64`, no `+1` |
| `0x1044` low | the granule count | `iw` |
| `0x118C` | `(entries-1, entries-2)` | `(entries-1, entries-1)` |
| DDR row stride (`0x1090`) | `iw*ic/16` | `iw*ic*2/16` |
| CVT | runs: Q14 unity, with a per-channel offset that is inert | bypassed, offsets zero |

**Conditions:** manufactured captures with `iw` at 16, 32 and 64, `ih` at 16, 32 and 48, both
strides, `k` 1/3/5/7 and `oc` 16/32/48/64.

The fp16 program has no channel fold. The int8 path folds the kernel's width into the channel
axis (4 lanes per pixel column, `kw` columns), so it programs 12 at `kw=3` against a feature
DMA of 3. The float path does not fold: the programmed count is the image's, and the taps
stay on the kernel axis. The `+1` granule goes with the fold, because it is the fold's
lookahead (the last output column of a row reads `kw` pixels, which reach past its own
granule). A float program at `k` 1, 3, 5 and 7 has the same entry count at every kernel size.

The decoded fp16 weight cube agrees independently. It has explicit `kh` and `kw` axes with four
lanes inside a tap, where a folded cube would put the columns inside the lanes.

The input is an fp16 packed image: 3 or 4 interleaved halfwords per pixel, which is why the
row stride takes the element size. The CVT is bypassed rather than configured, since a float
image is already in the value domain the MAC wants. The per-channel truncates and Q14 scales
stay in `0x1048` under the bypass, where they are inert.

The weight-byte registers are the same two on both paths. Register `0x1020` is the bytes one
output channel occupies, the element count scaled by the element size. The high half of
`0x1030` is twice the raw element count. The two coincide at fp16 and differ by the factor of
two at int8, so one can be mistaken for the other.

The element count per output channel is `ic*kh*kw` on the direct path, `4*kh*kw` on the float
first conv (dense, no round-up of the lane group to 16) and `kh*round16(4*kw)` on the int8
one. Software must not scale the depthwise count. A depthwise weight occupies a 16-bit slot
whatever the precision, so its count is already the float one.

### Output channels per fp16 first-conv program

One fp16 first-conv program delivers 32 output channels and no more [HW sweep, H96 MAX M9].
At `oc` 24 and 32 the output is complete. The written part is a contiguous prefix, not the
interleave the int32 writer's byte budget gives.

> [!CAUTION]
> At `oc` 48, 64 and 96, exactly 32 whole channels come back bit-exact, and the rest of the
> output is never written. Software must split the output channels into tiles of 32, one
> submit per tile.

The vendor's compiler emits single programs that the RK3576 does not fully execute. A program
and weight cube that match the vendor's own `oc=48` and `oc=64` captures register for register
write only 32 channels. The split is the one the direct path uses for its weight slice, and
each tile is an independent convolution over its own channels. The output cube's channel
groups are contiguous planes, so a tile is the same BO at a plane offset.

The row window composes with the output-channel split on the other axis. At fp16 the CBUF
entry count per row and the row cap the allowance gives both change. On this path the feature
offsets also change, because a float packed image is `ic` interleaved halfwords per pixel,
where an int8 one is bytes.

A 224×224 plane is 6272 granules against a 6144 ceiling and has no single-task plan. With the
window it is bit-exact at `k` 3 and 7 and at `oc` 32 and 64, the two splits running together.

### The vendor's float BS arrangement

The vendor's float programs use a BS arrangement that differs from the integer path's in
three DPU_RDMA registers and in `0x4044`. Every vendor float program contains the values below
[source-confirmed: RKNN-Toolkit2 captures]. The last column is measured on the RK3576
[HW sweep, H96 MAX M9]:

| Register | Integer path | Vendor float programs | Vendor value on the RK3576 |
|---|---|---|---|
| `0x5034` | `0x41` | `0x1` | bit-exact |
| `0x5044` | `0x40000010` | `0x40010050` | bit-exact |
| `0x501C` (BRDMA_CFG), the BS operand reader | `0x710` | `0x100` | the DPU writes nothing |
| `0x4044`, the BS ALU config | 1 | 2 | bit-exact with `0x501C` at `0x710` |

> [!CAUTION]
> The vendor's `0x501C` value `0x100` makes the DPU write nothing at all: an untouched output,
> no fault, nothing in dmesg. Paired with the vendor's `0x4044 = 2`, the writer runs again,
> and the arithmetic is wrong on about a tenth of the output. Of 2048 elements, 1866 are within
> 1% of the CPU model, and the worst relative error is 0.016. Software must keep `0x501C` at
> `0x710` with a coefficient group packed for the integer arrangement [HW sweep, H96 MAX M9].

On its own, with `0x501C` at `0x710`, `0x4044 = 2` is bit-exact. The vendor's float epilogue
is therefore a different BS *arrangement*, not a different constant, and it is not decoded as
a whole.

Two more float-path values are bit-exact on the RK3576. CORE `0x3018` bit 0 is an int8
marker that every float program clears. DPU `0x4010`'s middle field (the DPU's input width)
stays zero on the float path, where only the output width and the processing precision move.

### The vendor compiler's packed-path choice

The vendor compiler takes the packed path at 1, 3 and 4 channels, and not at 2. It compiles a
2-channel float conv down the direct path, with no `GROUP_LINE_OFF` and no `ARGB_IN`. That
choice is a toolkit preference: the packed path is bit-exact at two channels on the RK3576
[HW sweep, H96 MAX M9]. In a capture set, a channel count with no ARGB capture does not show
that the RK3576 refuses that count.

### Software support

In `librocketnpu`, `rocket_rk3576_weight_argb_fp16()` packs the fp16 ARGB cube and
`rocket_rk3576_weight_argb_int8()` the int8 one.

`rocket_conv2d_int8_rk3576()` implements the int8 first conv, its path for `ic <= 4`:

- The packed image
- The cube
- The row window
- The output-channel split
- The zero-point fold
- The de-tile

It is bit-exact against a CPU model at twenty shapes [HW sweep, H96 MAX M9]. They include the
224×224 stride-2 stem at `oc` 32 and 64 and at `k` 3 and 7. It refuses the three silent
geometry bounds rather than leaving them to compute. It widens a 1-channel image's row to two
channels instead of raising the `0x1078` field. At `ic=1` it refuses `k=1`, `k=5` and `oc=16`
exactly where two channels do.

`rocket_conv2d_fp16_rk3576()` does the fp16 output-channel split. The planner at a precision
is `rocket_rk3576_plan_rows_prec()`, which sets the CBUF entry count per row, the row cap and
the feature offsets for the precision.

On the float path `librocketnpu` reproduces the vendor's `0x5034` and `0x5044` values. It
packs its coefficient group (`A`, `B`, `C`) for the integer BS arrangement, and keeps
`0x501C` at `0x710`. It takes the packed path for every channel count at or below four.

## CBUF row reuse across a windowed sequence

Consecutive windows share `kh - stride_y` input rows. The vendor's programs keep those rows
resident and move the CBUF write pointer instead of refetching them [source-confirmed:
RKNN-Toolkit2 captures]. A task can instead fetch its whole window against a CBUF base of
zero, which is self-consistent and bit-exact on the RK3576. Seven registers program a
continuation task:

| Register | On a continuation task |
|---|---|
| `0x1028` hi | `entries * (ih - retained)`, the granules this task fills, not its window |
| `0x1078` lo | `ih - retained - 1` |
| `0x1098` | `iw * (ih - retained)` |
| `0x103C` lo | CBUF granule where the task's window begins, where the retained rows sit |
| `0x1040` lo | CBUF granule where fetching resumes = the window base plus the retained rows |
| `0x1018` bit 31 | set when the task retains rows |
| `0x1038` bit 31 | set on every task past the first, retaining or not |

The feature address (`0x1088`) also points at the first new row rather than at the window
start. In the vendor's `dw` capture, the second task reads input rows 89..111 of a 112-row
plane, retains 2, and has `0x1088` = row 91.

`retained` is the window arithmetic, the previous window's end minus this window's start,
which at these geometries equals `kh - stride_y`. At zero, the vendor resets the base to the
sequence's origin instead of continuing it, which is what a k=1 continuation contains.

Reuse saves `entries * retained` granules of feature DMA per continuation task, under a tenth
of the window's traffic at the captured geometries. A window costs ~1.2 ms, and that cost is
the driver's completion poll, not the fetch. Reuse is therefore expected not to shorten a
windowed conv measurably [expected].

> [!CAUTION]
> A continuation task with a wrong CBUF base reads resident rows that are not there, and
> corrupts the output silently. Software must derive each continuation task's base and resume
> point from the window arithmetic above.

**Not characterized:** the reuse path on silicon, and sequences of three or more tasks. Every
captured split is two tasks. Over a longer sequence, `cbuf_resident` is expected to accumulate
by the same arithmetic and nothing more [expected].

In `librocketnpu`, `rocket_rk3576_plan_rows()` fills `retained` and `cbuf_resident` on every
task, and `gen_conv2d_int8_rk3576_reuse()` emits the reuse form. The reuse form is opt-in, and
the refetching path is the default. Emitting it makes the vendor's split programs a complete
reference for seven registers the regcmd encoder's register comparison cannot otherwise reach.

## The same graph, compiled three times

Each `.rknn` capture holds its graph compiled three times, and the three compiles never
differ in geometry. Two of them are the same per-layer task split at different CBUF origins,
0 and 7168 granules. The origin is added to every base in the table above. The third splits
the whole chain instead of each layer. It runs all five layers over the top half of the image,
then all five over the bottom half, from a CBUF origin of 6144. It is the one that sets
`0x1014` bit 28 and drops the early duplicate `0x1038` preamble write (138 register writes
rather than 139).

On one reading, the third compile is the two-core compile [hypothesis]. Under that reading,
7168 granules are one core's whole CBUF pool, and an image-plane split is the standard way to
use two cores. Bit 28 of `0x1014` sits where the RK3588 puts `NN_MODE`, which is consistent
with a mode bit and says nothing about which mode. All of it is decodable off-device, and none
of it is decided by these captures.

A regcmd encoder does not need to reproduce the three compiles. The origins, the format low bits
(`0x1018`, `0x1038`) and the allowance field are the vendor allocator's per-compile choices.
On the refetching int8 path the live bits measured are `0x1018` bit 30 and `0x1038` bit 31
(§"The row window and its three registers"). `0x1018` bit 31, the reuse form's retain flag, is
not characterized on the RK3576.

## The appended DPU epilogue stage

One DPU epilogue op appended to a convolution changes three registers [source-confirmed:
RKNN-Toolkit2 captures]. A bias add, a scale and an eltwise sum each change the same three, to
the same values:

| Register | plain conv | conv + an epilogue stage |
|---|---|---|
| `0x4044` BS ALU operand | `1` | `0` |
| `0x4050` BS config | `0x80011111` | `0x80021111` (bits[19:16] 1 -> 2) |
| `0x501C` BRDMA config | `0x710` | `0x114` |

These registers do not encode which epilogue op runs. A per-output-channel bias, a
per-channel scale and a per-channel sum all lower onto one BS-stage configuration. What
separates them is the coefficient buffer's contents and the requant triple
(`0x40AC`-`0x40B4`), the only other thing that differs between the three captures.

Neither `0x4050` nor `0x501C` decodes under the RK3588's field map at these offsets. The value
`0x80011111` sets bit 12, which is reserved in the RK3588's `BS_CFG`. The RK3576's
`BRDMA_CFG` is wider than the RK3588's 5-bit field. Both values are therefore transcribed
constants rather than decomposed fields. The BS stage appears to take its second operand from
`DPU_RDMA` rather than from the inline path [hypothesis]. On that reading it is the same stage
and the same buffer as the asymmetric-weight `B` term.

### The BS operand source

`0x501C` is the BS operand-source register, and `0x4044` and `0x4050` are inert without it
[HW sweep, H96 MAX M9]. Driven one at a time and together, against a conv with a known
non-zero `B`, the three registers give these outputs:

| driven | outputs `+B*sum(x)` explains | outputs no-B explains |
|---|---|---|
| nothing (plain conv) | **32768/32768** | 537 |
| `0x4044 = 0` | 32768/32768 | 537 |
| `0x4050 = 0x80021111` | 32768/32768 | 537 |
| `0x4044` + `0x4050` | 32768/32768 | 537 |
| `0x501C = 0x114` | *the job writes nothing* | n/a |
| `0x4044` + `0x501C` | *the job writes nothing* | n/a |
| `0x4050` + `0x501C` | 1423/32768 | **8192/32768** |
| all three | 1423/32768 | 8192/32768 |

On the plain path, `0x4044` and `0x4050` change nothing at all. Not one of the 32768 outputs
moves, down to the incidental 537 that the no-B model always coincides on.

> [!CAUTION]
> Driven alone, `0x501C` stops the DPU writing, with no fault and nothing in dmesg. Software
> must program `0x501C` with its matching BS config, which depends on the value. The
> epilogue's `0x114` is re-enabled by `0x4050` bits[19:16] = 2, and the float path's `0x100`
> by `0x4044` = 2.

Under the epilogue arrangement, the BS stage does not read the coefficient group's `B`. The
count that `+B*sum(x)` explains falls from every output to 1423. Exactly one output in four
matches the model with no B term at all. The counts are 8192/32768, 7200/28800 and 512/2048 on
three different shapes, a quarter each time.

These registers select where the BS stage takes its second operand from, not an epilogue
*op*. The quarter is what a buffer packed for the inline positions looks like when read as
the other kind of stream [hypothesis].

The float path's refusal to write is the same mechanism, and it makes a third BS arrangement
visible:

| arrangement | `0x4044` | `0x4050` | `0x501C` |
|---|---|---|---|
| plain int8 conv | 1 | `0x80011111` | `0x710` |
| int8 conv + an epilogue stage | 0 | `0x80021111` | `0x114` |
| vendor float conv | 2 | `0x00021111` | `0x100` |

The first arrangement is bit-exact at both precisions with a coefficient group (`A`, `B`, `C`)
packed for it. `librocketnpu` packs its coefficient group for that arrangement.

**Not characterized:** the fields of the other two arrangements. Neither is needed to
compute.

## Depthwise convolution

The int8 depthwise convolution is bit-exact on the RK3576 [HW sweep, H96 MAX M9]. A register
program that matches the vendor's does not on its own give a correct result. Two of the path's
inputs are buffers rather than registers: the coefficient group and the weight cube. A `.rknn`
capture contains a register program and says nothing about the memory the registers point
at. Both buffers differ from the direct path's, and a wrong one gives incorrect output
with no fault. Both layouts are therefore measured on the RK3576.

In `librocketnpu`, `gen_conv2d_dw_int8_rk3576()` emits the depthwise program. It reproduces
the vendor's across 88 captured task programs with no field left open.

### Validated depthwise shapes

The depthwise path is bit-exact against a CPU model over the whole output, at every shape
below [HW sweep, H96 MAX M9].

| axis | covered |
|---|---|
| channels | 8, 16, 24, 32, 48, 64, 72, 80, 96, 112, 128, 144, 176, 256 |
| kernel | 1, 3, 5, 7 |
| stride | 1 and 2 |
| padding | SAME and VALID |
| plane | 16×16 to 112×112, plus 15×18, 17×19 and 19×19 |
| row window | 1, 2, 3 and 6 tasks |

A depthwise task costs about 1.1-1.2 ms of NPU wall-clock time, whatever the channel count,
the kernel or the plane. The cost is the same fixed per-submit dispatch cost the direct path
has, with no compute term visible at these shapes. A row window costs that per task.

Channel counts that are not multiples of 32 are where this path's granule counts stop
agreeing:

- The weight cube rounds to 16.
- The CBUF allocation takes a 16-group count of 3 mod 4 one group further.
- The 2-bit group field in `0x4050` wraps.

Planes whose `ow*oh` is not a multiple of four are where the padded output surface stride
below applies.

> [!NOTE]
> A test that rounds every depthwise channel count up to 32 exercises none of those
> quantities. Channel counts in multiples of 32 and planes in multiples of four hide both.

### The depthwise coefficient group

The depthwise coefficient group covers 8 output channels in 48 bytes and has no `B`. The
direct path's group is 64 bytes for the same 8 channels: `A` (int32 bias) at `(oc%8)*4`, `B`
(weight zero point) at `+32`, `C` (int16 multiplier) at `+48`. The depthwise layout is:

```
A[oc]  int32  at (oc%8)*4
C[oc]  int16  at 32 + (oc%8)*2
```

Channel `oc` sits in group `oc/8`, and the group stride is 48 [HW sweep, H96 MAX M9].

> [!CAUTION]
> Handed the direct path's 64-byte group, the depthwise path writes a correctly sized, mostly
> empty output with no fault. Software must pack the depthwise coefficient group as 48 bytes
> per 8 channels.

With the 64-byte group, the depthwise path reads `A` and `C` at indices that drift apart at
different rates. There, `A` strides 4 bytes per channel and `C` strides 2. Past the first
eight channels, each is read out of a different group. The BS stage computes `(acc + A) * C`,
so the output comes back as follows:

- Most channels multiply a bias by a zero and never reach DDR at all.
- A few channels, whose `C` falls in the `A` region, square their own bias.
- One group's `C` multipliers are read as the next group's biases.

The result has the same signature a wrong geometry register gives, so it reads as a datapath
defect rather than a packing one.

In `librocketnpu`, `rocket_rk3576_pack_coeff_dw()` and `rocket_rk3576_coeff_bytes_dw()`
implement the depthwise layout.

The depthwise group has no slot for a weight zero point. Its correction is not a per-channel
constant, so the bias cannot apply it either. Software must apply it in the weight cube, as
`w - w_zp` across each (channel, tap)'s two live bytes ([rk3576.md](rk3576.md) §"The depthwise
weight zero point").

### The weight cube

The depthwise weight cube has two layouts, one per precision. They share a block structure
and nothing else.

The float cube places each weight in a 16-bit slot:

```
slot(c, kh, kw) = (c/32)*32*KH*KW + (kh*KW + kw)*held + (c%32)
held            = min(32, C - (c/32)*32)
```

The layout is decoded from vendor captures with a unique weight value per (channel, tap).
The captures cover C = 24, 32, 48, 64 and 128, at k = 3 and 5.

The layout has four properties:

- A weight occupies a 16-bit slot, so one geometry's float and int8 cubes are the same size.
  The slot width is also why `WEIGHT_BYTES` contains a factor of 2 that is not the element
  size.
- Channels group by 32, each group a contiguous block.
- Inside a group the order is tap-major, kh outer.
- A trailing partial group is dense. Its tap stride is the number of channels it holds, not
  32, so C=48 is 432 slots and not 576.

The int8 cube is neither the float cube's low byte nor the RK3588's group-of-64 single-byte
packing. Each byte of a C=32 k=3 cube, 576 in all, reaches exactly one (channel, tap)
[HW sweep, H96 MAX M9]:

```
byte(c, kh, kw) = (c/64)*64*KH*KW*2 + (kh*KW + kw)*held*2 + 4*((c%64)/2) + (c%2)
held            = min(64, round16(C - (c/64)*64))
```

Against the float cube, the int8 cube differs in three ways:

- The channel group is 64, where the float cube's is 32.
- Inside a tap block, the byte at offset `b` belongs to channel `2*(b/4) + (b%2)`. Channel
  `c` has two bytes per tap, at `4*(c/2) + (c%2)` and two further on. Both bytes are read,
  and a weight written into both is added twice. Software must write the weight into the
  first byte and leave the second at zero.
- A trailing partial group strides by what it holds rounded up to 16, where the float cube's
  is dense in the raw count. The whole cube is exactly `round16(C)*KH*KW*2` bytes.

> [!CAUTION]
> An int8 weight written as a 16-bit value into the float slot puts its sign extension into
> the next channel's byte. The sign extension is a silent -1 weight on the neighbor, not
> padding. Software must write an int8 weight as one byte, at the int8 cube's offset.

> [!NOTE]
> Two shapes do not distinguish the group sizes. At a single channel group the two layouts
> coincide. At `k=1` there is one tap, so the group base and the tap stride cannot be told
> apart. C=64 k=3 separates a group of 64 from a group of 32, and C=24 and C=72 separate the
> round16 partial group from the raw one.

In `librocketnpu`, `rocket_rk3576_weight_dw()` computes the float slot, and
`rocket_rk3576_weight_dw_int8()` computes the int8 cube's byte offset.

### The depthwise output surface stride

Register `0x401C` is the output surface stride: the distance in elements from one
16-channel output group to the next. The depthwise path programs `round4(ow*oh_full)` there,
where the direct path programs `ow*oh_full`. The stride applies in DDR.

The padding has no effect at a plane whose `ow*oh_full` is already a multiple of four, which
is most planes. Planes of 16×16 and 112×112 are unaffected, and 15×18, 17×19 and 19×19 are
padded. For `ow*oh_full` of 270, 323 and 361 the vendor programs 272, 324 and 364
[source-confirmed, HW sweep, H96 MAX M9].

On the direct path the stride is not rounded. A VALID-padded oc=128 conv at
`ow*oh_full = 105` is bit-exact with 105 programmed and wrong with 108
[source-confirmed, HW sweep, H96 MAX M9].

> [!CAUTION]
> Software must size the depthwise output BO, and de-tile it, with the rounded stride. A
> de-tile that assumes the unrounded plane places every group past the first up to four
> elements early. Only the first 16 channels are then exact: half the output at C=32, and a
> quarter at C=64.

In `librocketnpu`, `rocket_rk3576_out_surf_elems(ow, oh_full, dw)` returns the stride.

A depthwise program is given the raw output channel count, and the DPU writes no channel group
past it. At `oc=144` the output has 9 written groups against the 10 a direct consumer of 144
channels reads [HW sweep, H96 MAX M9, 2026-07-31].

### Depthwise geometry registers

The depthwise path's geometry registers follow the formulas below
[source-confirmed, RKNN-Toolkit2 rk3576 depthwise builds].

**Conditions:** vendor register programs at 22 channel counts from 8 to 256, strides 1 and 2,
kernels 1, 3, 5 and 7, and rectangular planes.

The path has two channel granules, and they are different numbers:

- The weight granule is `C` rounded up to 16. `WEIGHT_BYTES` (`0x101C`) is
  `round16(C)*kh*kw*2`, and `WEIGHT_ELEMS` (`0x1020`) is `round16(C)*kh*kw`. The
  weight-bytes-per-kernel term (`0x1030` high) is `kh*kw*round16(C)/8`, at k=1, 3, 5 and 7.
- The feature granule is `C` rounded up to 16. If the resulting 16-channel group count is 3
  mod 4, the granule rounds up one group further. For example, 48 rounds to 64 and 112 to
  128, while 80 and 144 are unchanged. The granule sizes the CBUF entry counts (`0x1028`
  high, `0x103C` high, `0x1044` low). The rule is consistent with a fetch in blocks of 64
  channels, where a trailing partial block holds one, two or four sixteens, never three.

The depthwise field in `0x4050`, bits[9:8], is the 16-channel group count minus one, modulo
4, computed on the *unrounded* count. C=48 has 2 there, while its feature allocation is
64's. The field is independent of the stride, and steps with C, wrapping at 80, 144 and 208.

> [!CAUTION]
> Software must program the depthwise `0x4050` field from the channel count. Above C=32,
> the C=32 word hands the BS stage a word the vendor never uses. The output BO is then left
> wholly untouched, with no fault and no dmesg line.

On the direct path, `0x4050` is the even-count word at every count `librocketnpu`'s regcmd encoder
programs. A partial 32-channel output group produces incorrect output, so the regcmd encoder pads a
direct conv's output channels to whole 32-channel groups (`rocket_rk3576_pad_oc`). The
programmed 16-channel atom count is always even. The `gahingwoo/mesa-rk3576` fork clears bit
8 only when `DIV_ROUND_UP(oc, 16)` is odd, and programs `oc` unpadded. At every padded count,
its rule gives the same `0x80011111`.

Real `oc` 8-112, padded, are exact through the regcmd encoder and the tiling entry [HW sweep, H96
MAX M9, rocket 1.6.0, 2026-09-30]. Written alone into those even-count programs, the
odd-count word `0x80011011` produces incorrect output at 19 of 21 shapes, consistent with the
fork's rule.

Four of the five other `0x4050` fields change the result. In a single-field sweep at `oc` 128
and `oc` 88, each was moved alone to upstream Mesa's value. `RGP_CNTER` gives incorrect output
with no hang, and `RESERVED_0` a correct output with a job timeout. `SIZE_E_0` and `OW_SRC`
give incorrect output and a timeout. `SIZE_E_1` changes nothing at either shape.

Across 94 vendor programs, `SIZE_E_1` is 0 in all 81 regular convolutions. It is 1 in all 13
depthwise ones [external: gahingwoo/mesa-rk3576 at `1f0de93`, their round 260, ROCK 4D, not
reproduced]. A program that changes `0x4050` writes the whole word for its path.

**Not characterized:** whether the odd-count word lets an unpadded partial group compute.

The plane term in `0x40B8` is a multiple of the output surface stride, and its multiplier is
independent of the kernel size. The register is `4*surface - ow*oh_task` on the depthwise path
and `2*surface - ow*oh_task` on the direct one, where `surface` is `0x401C`. The multiplier
does not move at k=1, 5 and 7.

Register `0x118C` is `iw-1` in both halves, on the depthwise path and the direct path alike.
On a square plane the low half also matches the full plane height, so only a non-square
plane separates the two.

The low half of `0x1024` is the same raw `1` at every depthwise channel count from 8 to 256.
It is not a channel count.

## fp16 direct convolution

An fp16 direct convolution is bit-faithful on the RK3576 and writes every output channel. One
task contracts sixteen input channels. A convolution with more input channels runs as one
task per 16-channel slice, with the partial outputs summed on the host.

Each fact in this section is
[source-confirmed, RKNN-Toolkit2 rk3576 float build, HW sweep, H96 MAX M9, measured 2026-07-26]
unless tagged otherwise.

The DPU's output element stride is `16/ic` words. At `ic = 8` it is two words per element, so
the writer uses four bytes for a two-byte element and reaches only `oc/2` channels. At
`ic = 16` it is exactly one word, and the output is the plain native cube.

> [!NOTE]
> An incorrect fp16 program gives one of three results, each from a different stage. An
> untouched output buffer is a DMA that did not run. An all-zero output is an underflowed or
> gated epilogue. Only an output of the correct size with incorrect values is an arithmetic
> or layout error. Only a comparison against a CPU model reveals it.

In `librocketnpu`, `gen_conv2d_fp16_rk3576()` emits one task, and `rocket_rk3576_plan_ic()`
plans an arbitrary input-channel count.

### The vendor float program

`librocketnpu`'s float program is a transcription of the vendor's. An int8 program has every
precision field at zero. A zero field does not show where a float field is, how wide it is,
or whether the RK3576 has one there.

RKNN-Toolkit2 emits a float program for the RK3576. With `do_quantization=False` on a float
ONNX, the toolkit leaves the weights float and the vendor compiler selects the float
datapath.

> [!NOTE]
> RKNN-Toolkit2 imports `onnx.mapping`, which onnx 1.16 removed, and `pkg_resources`, which
> setuptools 81 removed. Neither import failure names its cause.

> [!NOTE]
> For a model whose plane size is not a power of two, the compiler appends a 1×1×1 program
> beside the conv. The helper program has a DPU_RDMA configuration that no conv program has,
> and a conv regcmd encoder never produces it.

Against the fp16 program emitted from the RK3588 encoding, the vendor's float program differs
in two ways:

- Only seven registers move with `ic` across the whole 139-word program: `0x101C`,
  `0x1020`, `0x1028`, `0x1030`, `0x103C`, `0x1044` and `0x107C`. All seven scale linearly,
  and the fp16 program emitted from the RK3588 encoding matches six of them.
- Fourteen registers differ from the RK3588-encoding program, the same fourteen at every
  geometry. Three of them are required, the float-mode registers below. Each of the other
  eleven can be dropped with the datapath still exact.

### The float-mode registers

Three registers select the float mode. Each is constant across every geometry the vendor
emits (`ic` 8-64, `oc` 32/64, `k` 1/3, planes 16 and 32), and each is required on its own:

| register | integer value | float value | required |
|---|---|---|---|
| CNA `0x100C` | `0x00000000` | `0x00200120` | all of `bit21 \| proc<<7 \| in<<4` |
| DPU `0x4038` | `0x00120080` | `0x00100092` | the low half, `0x0092` |
| DPU `0x4050` | `0x80011111` | `0x00021111` | bit 31 clear |

> [!CAUTION]
> Software must program all three registers to their float values. With any one of them at
> its integer value, the contraction reads every feature surface twice and skips the odd
> ones.

The three registers behave as follows:

- CNA `0x100C` bit 21 is a float enable that the integer programs never set. The two
  precision fields do not imply it. With both precisions right and bit 21 clear, the
  contraction still doubles, exactly as it does with bit 21 set and a precision wrong.
- Software must program DPU `0x4038`'s high half non-zero, or the DPU writes nothing at all.
  Its value is otherwise free: `0x0012` and `0x0010` both work.
- In DPU `0x4050`, bit 31 is the float bit, and the low nibbles have no effect on the float
  mode. The values `0x00011111`, `0x00021111` and `0x00020000` all work. The value
  `0x80021111` returns `+inf`, because the accumulator has summed the doubled reads. The
  nibble is the write-extent field described below.

### Precision fields

PROC_PRECISION, the `proc_precision` field of CNA `0x100C`, is the operand width.

> [!CAUTION]
> Software must program `proc_precision` to fp16 (`2`), not fp32. The RK3588's field
> semantics suggest the field names the multiply-accumulate datapath's width, so that a
> convolution accumulating in fp32 wants `5`. Programming `5` with every other float field
> right reads each feature surface twice. A test input that is uniform in the channel axis
> does not detect it, because there a doubled read is indistinguishable from a scale.

CORE `0x3018` does not follow the CNA word, and the two divide the work:

- CNA `0x100C`'s `in_precision` selects the operand width class, not its type. Values `1`,
  `2` and `3`, the three 2-byte codes, all work, so an fp16 and a bf16 program are
  indistinguishable in this field. Value `0` (a 1-byte element) writes an entirely zero
  output.
- CORE `0x3018` selects the operand type. Value `2` is fp16 and `3` is bf16, and each
  returns incorrect output on the other's bytes. Fed bf16 bytes, `3` contracts bit-exactly
  in the matmul-form program, the case that [rk3576.md](rk3576.md) §"The precision fork"
  records.

### The float narrowing and the output width

The DPU epilogue is a float path, and its result is an fp32 word. The output is fp16 only
with the float narrowing enabled. Two fields select what reaches DDR:

- DPU `0x40B0` bit 16 is `fp32tofp16_en`, at the RK3588's bit position. Without it, the
  DPU writes the fp32 word as-is.
- DPU `0x4010` bits [31:29] select a width, not a conversion. Value `5` writes the whole
  32-bit word, and `2` and `3` write its low and high halves.

The two combine as follows:

| narrowing | width | what reaches DDR |
|---|---|---|
| on | `2` | true fp16 (`0x5400` for 64.0) |
| on | `3` | the top half of the fp32 word, which is bfloat16 |
| off | `5` | fp32 |
| off | `2` | the fp32 word's low mantissa bits, zero for every value a small integer test pattern can produce |

> [!CAUTION]
> An fp16 program emitted from the RK3588 encoding programs the last combination. It writes
> an all-zero output of the correct size from a running datapath, which looks like a dead
> MAC array. Software must set `fp32tofp16_en` and width `2` for an fp16 output.

> [!NOTE]
> Every value that a small integer convolution produces has zero low mantissa bits in fp32.
> Read through a 16-bit output width, an fp32 result is then uniformly zero, the same result
> as a gated epilogue or a dead MAC array. Read at width `5`, or drive values whose low
> mantissa bits are non-zero.

### The direct-path float weight cube

The float weight cube groups both channel axes by 16, where the int8 weight cube groups both
by 32. Otherwise the reorder is Mesa's `oc1, ic1, kh, kw, oc2, ic2`. The feature cube is
C2 = 8 fp16 lanes in the same 16-byte atom (C2 = 4 and 16 both fail).

Each weight group is pinned at a shape where it is observable:

- The ic group is 16. It is exact where 4, 8 and 32 are not.
- The oc group is 16. It is exact at `ic = 16` with `k` 3 and 5.

> [!NOTE]
> The oc group cannot be pinned at `k = 1`, because the kernel index sits between the group
> and the `(oc2, ic2)` pair. At `k = 1` the index collapses to `(oc/G)*G + oc%G = oc`, and
> every group size is byte-identical. A kernel that ramps over `ic`, or any score summed over
> the channel axis, is invariant under every permutation of that axis. Neither can pin the ic
> group. Use realistic weights at a shape where the group is live.

> [!CAUTION]
> A partial input-channel group still occupies a whole one. A convolution with 8 real input
> channels runs as an `ic = 16` task, zero-padded, and needs a cube for 16. Software must size
> the weight buffer from `ceil(ic/group)*group`. At `k = 1` an allocation taken from the real
> channel count overruns inside the BO's page, and the output is correct. At a kernel large enough to run past the page, the overrun looks as if the
> RK3576 cannot run `k = 3`.

In `librocketnpu`, `rocket_rk3576_weight_conv_fp16()` is the index.

### The fp16 output map

The fp16 output map is the plain native cube. At the 16-channel contraction the writer
places 8 channels in a 16-byte atom, one atom per pixel, with channel groups as contiguous
planes. Every programmed channel is present exactly once. For output channel `c` at pixel
`p`, in 16-bit words:

```
word = (ow*oh*8)*(c/8) + 8*p + (c%8)
```

The map changes with `ic`. At the same `oc = 16` and a 4×4 plane:

| `ic` | slot of channel `c` within its atom | channels reaching DDR |
|---|---|---|
| 8 | `2*(c%4)`, odd words zero | `oc/2` |
| 16 | `c%8` | all of them |
| 32 | `(c/2)%8`, odd channels lost | `oc/2` |

In `librocketnpu`, `rocket_rk3576_fp16_out_index()` is that map. Its companion
`rocket_rk3576_fp16_accumulate()` de-tiles one output through it into a row-major fp32
accumulator, which an `ic` split needs.

### The contraction width

The fp16 contraction is sixteen input channels wide.

> [!CAUTION]
> Software must program each fp16 conv task with `ic = 16`. A task with any other count
> writes an output of the correct size with incorrect values, and nothing faults.

Without the float-mode fields, at `ic = 32`, input channel 0 is read twice and input channel
16 is not read. With them, the read count is one at `ic` 16, 32, 64 and 128 and at every
channel, so the reads do not bound `ic`. What bounds a task is the writer's element stride,
`16/ic` words.

In `librocketnpu`, `gen_conv2d_fp16_rk3576()` refuses any `ic` other than 16. The unchecked
bring-up entry beside it is `gen_conv2d_rk3576_prec()`.

### Single-task fp16 coverage

A single fp16 task is bit-faithful against the CPU model (`worst relative error 0`) at
`ic = 16`, and every programmed output channel is written. The coverage is:

| axis | covered |
|---|---|
| `k` | 1, 3 and 5 |
| plane | 8×8 through 56×56 |
| stride | 1 and 2 |
| `oc` | 16 through 64 |

### An arbitrary `ic` via a 16-channel split

Splitting `ic` into slices of 16 and summing the partial outputs on the host reproduces the
model exactly at any input channel count. It is bit-exact, `worst relative error 0`, at `ic`
16, 32, 64 and 128 crossed with `k` 1, 3 and 5.

The slices cost nothing to address. At C2 = 8 the feature cube's channel groups are
contiguous planes of `iw*ih_full` 16-byte atoms. Slice `k` is therefore the same BO at a
base offset. An atom stays 16 bytes when C2 halves, so the stride is independent of the
element size. Only the weight cube is rebuilt per slice.

> [!CAUTION]
> Software must build each slice's weight cube as a convolution of its own, with its own
> group count, and size it by the groups. A slice's cube is not a sub-cube of the whole
> conv's. Slicing the full cube by byte range gives a correctly sized cube with the kernel
> positions shuffled.

> [!CAUTION]
> Software must give the bias to one slice only. Every slice runs the whole epilogue and adds
> the whole `A` term, so a bias in every slice's coefficient buffer is added `ic/16` times.
> The error is a per-channel constant on an otherwise exact output. A wrong `B` sign
> convention gives the same shape.

> [!CAUTION]
> Software must plan the CBUF allowance at the precision the conv is emitted at. The feature
> plane's granule cost is per byte, so a 2-byte element doubles it. An allowance planned at
> the int8 rate is sized for half an fp16 plane. That plane then gives incorrect values in a
> full-size output.

In `librocketnpu`, `rocket_rk3576_plan_ic()` is the split. It sits beside
`rocket_rk3576_plan_rows()` and follows the same pattern: it lays the slices out, and the
caller emits one conv per entry. The slice is fixed at 16 rather than as wide as the CBUF
allows, because a wider slice is a wrong task. The planner splits the `ic` axis only. It
refuses a plane whose 16-channel slice still overflows the CBUF, with a pointer at the row
planner. `rocket_rk3576_fp16_pack_slice_weights()` builds a slice's weight cube.

Both CBUF planners take the precision the conv is emitted at: `rocket_rk3576_cbuf_f_prec`
and `rocket_rk3576_max_task_rows_prec`. The shorter names are the same planners at int8. The
weight side stays at the int8 model on purpose, as an upper bound on the float slice. An
over-estimate refuses early instead of corrupting.

### Cost of an fp16 conv

The fp16 convolution runs as the `ic/16` split, and its cost follows the submits the split
issues [HW sweep, H96 MAX M9, measured 2026-07-26].

**Conditions:** `oc` 64.

| shape | submits | NPU | host pack + de-tile | total |
|---|---|---|---|---|
| 32×32 ic16 k3 | 1 | 1.51 ms | 2.52 ms | 4.04 ms |
| 32×32 ic64 k3 | 4 | 5.57 ms | 3.36 ms | 8.92 ms |
| 56×56 ic64 k1 | 4 | 6.00 ms | 24.20 ms | 30.20 ms |
| 56×56 ic64 k3 | 4 | 6.21 ms | 8.62 ms | 14.83 ms |
| 56×56 ic128 k3 | 8 | 11.93 ms | 15.93 ms | 27.86 ms |

The NPU time is flat in the work and linear in the submits, at about 1.4-1.5 ms per submit
whatever the plane and the kernel. At 56×56, `k = 3` does nine times the multiply-accumulates
of `k = 1` and costs the same 6 ms. No compute term is visible at these shapes. The NPU time
is entirely the per-submit dispatch overhead, the same 1.1-1.3 ms that the int8 path costs
once for its whole `ic`.

Only the NPU column is a measurement. The host column is noisy on this board, where the same
shape varies by a factor of two run to run. The 56×56 `k = 1` row
shows it most clearly. The NPU column is stable to within 6% across pacing gaps of 0, 20 and
50 ms.

#### Readback share

On the RK3588 the DPU eltwise stage sums partials on-chip (`ROCKET_KACC`). Here it would
remove `ic/16` readbacks [expected]. Nothing in the datapath blocks it: at this contraction
width, the partial it would read back is a plain, dense, complete fp16 cube.

On-chip accumulation is not the largest saving on this path. Readback is 7-13% of the
entry's wall-clock time [HW sweep, H96 MAX M9, measured 2026-07-28]:

**Conditions:** `oc` 32, 28×28 k3, `ic` 16-128.

| phase | share of the entry's wall-clock time |
|---|---|
| submit | 77-87% |
| readback (what on-chip accumulation removes) | 7-13% |
| weight repack | 4-7% |
| output stamp | 2-4% |

The sixteen-channel contraction sets the slice count, whatever sums the partials. On-chip
accumulation therefore removes the readback and leaves the submits. The EW operand DMA that
the DPU would issue instead is not free, so the net saving is below the readback share
[expected].

#### DPU grace

**Conditions:** `ic` 128, 8 slices. A `rocket` with patch `0012`, from before interface version
1.0, measured 2026-07-28.

On that module a slice costs the driver's per-submit completion-poll overhead plus
`dpu_grace_us`, in full: the driver retires it on `PC_DONE` plus the grace. Halving the grace
reduces the wall-clock time by very nearly the grace, once per slice [HW sweep]:

| `dpu_grace_us` | 8-slice wall-clock time | per slice |
|---|---|---|
| 500 (default) | 15.39-16.16 ms, median 15.89 | ~1.99 ms |
| 250 | 12.95-13.97 ms, median 13.29 | ~1.66 ms |

Each slice saves 300 us for the 250 us cut, 1.20x over the eight slices [host-computed]. The
convolution stays bit-exact at every grace value tried down to 150.

On that module the default grace cannot be lowered for this path alone. The same number is the
deadline on the DPU-completion wait for every job that raises one. A narrow-output matmul
fails 10 runs of 10 at 200 us (§"Tasks that raise no DPU completion").

On `rocket` 1.6.0 the fp16 programs raise the DPU completion, and the grace does not set a
slice's cost ([rk3576.md](rk3576.md) §"Retirement on the DPU completion").

### The write-extent field and the byte-placement words

The low nibbles of `0x4050` encode the written extent. The field encodes 1, 2 and 4 bytes per
element as `{0, 1, 3}`: the direct int8 word contains `1` and the depthwise word `3`. The
encoding is the same `size_e = bytes-1` relation that the RK3588 pairs with its
surface-advance multiplier. The RK3576's surface advance follows the same multiplier
(`ow*(2*oh_full - oh)` direct, `ow*(4*oh_full - oh)` depthwise). The field moves the written
*extent* only, never the atom. The float mode depends on bit 31 of this register, not on this
field.

Two DPU words move the output within the atom. Neither selects the float mode:

| register | effect |
|---|---|
| `0x40D0` low 16 | a per-byte write enable for the 16-byte atom: clearing bit *n* leaves byte *n* untouched |
| `0x40D0` bits [19:16], `0x40CC` bits [3:0] | shift the written data by whole bytes within the atom |

### Coefficient formats in float mode

In float mode the DPU reads `A` (bias) as fp32 and `C` (multiplier) as fp16.

> [!CAUTION]
> Software must program `C` as an fp16 value in float mode. The int8 default `C = 1` is the
> denormal 6e-8, and underflows the whole output to empty. The fp16 value `1.0` is `0x3C00`.

Asymmetric weights are a packing question, not an encoding one. The `B` field in the
coefficient group is a per-output-channel int16 beside the bias `A` and the multiplier `C`.
Driving a weight zero point through it needs no register change. The DPU adds the term, so
software must program a weight zero point negated. The measurement is under "The B term is
added" above. In `librocketnpu`, `rocket_rk3576_pack_coeff_asym()` takes the zero point.

## DPU shift operand (`0x5024`)

`BS_BASE_ADDR1` (`0x5024`) is the base address of a per-task operand, the DPU shift word. The
DPU reads one 32-bit word through it per task and right-shifts the BS product by two
independent 6-bit fields, one per sign:

```
bits[5:0]   right shift applied where the result is NON-NEGATIVE
bits[13:8]  right shift applied where the result is NEGATIVE
```

Both are exact powers of two over the whole range. A word of `0x0202` takes an accumulator of
+64 to +16 and one of -64 to -16. A field of 8 or more flushes that sign to zero. The two
sides are independent. [HW sweep, H96]

The sign of the BS product selects the field, not the sign of the weight. With `C` positive,
as in every measurement, that is the sign of `acc + A`. Swapping the feature sign swaps which
output channels move. [HW sweep, H96]

The shift applies to the BS product, and it rounds half to even [HW sweep, H96 MAX M9,
2026-09-23]. The stage computes `((acc + A)*C) >> s`, multiplying before it shifts, with the
product held wide as the coefficient buffer section above records. A per-channel `C` is
therefore a fixed-point gain, with the shift as its binary point.

Every capture stores 0 here, for the same reason every capture stores 0 for the feature,
weight, output and bias bases. It is an address that the vendor runtime patches at load
time, and a `.rknn`-derived program is the *stored* register set.

> [!CAUTION]
> Software must point `0x5024` at a word that holds the intended shifts, such as 64 zeroed
> bytes. IOVA 0 is a real buffer on a mainline `rocket` stack. The per-fd address space
> bump-starts at 0, so whichever BO is allocated first is addressable as 0. With `0x5024` at
> zero, the DPU takes both shift amounts out of that buffer's first four bytes, normally the
> head of the feature cube.

The fault presents as data-dependent arithmetic, not as addressing, because the shift depends
on two bytes of *feature data*. A conv is bit-exact for one feature tensor and attenuated by
an arbitrary power of two for another. It does so reproducibly, with no fault and no dmesg.

Common test inputs do not show it:

- A uniform feature fill puts the same byte in both fields, so the result is either right or
  uniformly wrong.
- An identity weight set reads one input channel per output channel, and cannot show a
  contribution that the model expects to be zero.
- Any feature whose first two bytes have zero low-6-bits, such as a fill of 64 or of 0, is
  exact.

The symptom follows the *feature cube's channel 1* only because that channel's first byte is
byte 1 of the buffer. The result is exact for some feature tensors and a clean power-of-two
too small for others. A factor that does not change with the programmed requant scale places
the fault upstream of the OUT_CVT and downstream of the MAC.

In `librocketnpu`, `rocket_rk3576_coeff_bytes()` reserves 64 zeroed bytes past the A, B and
C groups, and the RK3576 regcmd encoder points `0x5024` at them.

## The cold-start wall

On an unpatched `rocket`, only the first job of each NPU power session writes its output.
The same job, submitted repeatedly, behaves as follows [HW sweep, H96]:

| Gap between jobs | Result |
|---|---|
| 0 ms (back to back, one fd) | job 1 writes, and jobs 2-8 leave the output untouched |
| 100 ms | all 8 write, byte-identical |
| 300 ms | all 8 write, byte-identical |
| separate processes | every one writes |

The NPU runtime-suspends 50 ms after its last job. The unit is the NPU power session, not the
fd, not the process, and not chained-versus-independent submits. Independent per-op submits
inside one power session fail exactly as a chained task does, and any gap past the
autosuspend delay re-arms the PC.

> [!CAUTION]
> On an unpatched `rocket`, every job after the first in an NPU power session completes and
> leaves its output untouched. Software must leave a gap longer than the 50 ms autosuspend
> delay between jobs, or run a `rocket` that programs the RK3576's `PC_TASK_CON` field width
> (below).

An external report on a different userspace (mesa/rocket + Teflon) describes the same
behavior, and found that independent per-op submits cannot bypass it. The same behavior
under `librocketnpu`'s own regcmd encoder and submit path places the cause in the driver.

### The `PC_TASK_CON` field width

`PC_TASK_CON` holds the task count of a kick, `TASK_NUMBER`, in its low bits, with three
control bits directly above it. The field is 12 bits wide on the RK3588 and **16 on the
RK3576**, and the difference causes the cold-start defect. All three control bits move up by
four:

| | `TASK_NUMBER` | `TASK_PP_EN` | `TASK_COUNT_CLEAR` | `TASK_LAST_LAYER_CLEAR` |
|---|---|---|---|---|
| RK3588 | `BIT[11:0]` | `BIT(12)` | `BIT(13)` | `BIT(14)` |
| RK3576 | `BIT[15:0]` | `BIT(16)` | `BIT(17)` | `BIT(18)` |

The RK3588 register description marks the top control bit reserved. On the RK3576 it is
`task_last_layer_clear`, and software should set it on every submit alongside the count
clear. Chaoyi Chen of Rockchip gave the layout on the `linux-rockchip` list
([message](https://lore.kernel.org/all/4f300b78-d96d-4d98-8819-dc292b0c9b97@rock-chips.com/)).
Shifting the RK3588's control triple up by four gives the same word, and both derivations
write `0x70001` [source-confirmed].

Mainline `rocket` builds the word from the RK3588 field accessors on every SoC, giving
`0x7001`. On the RK3576 that word asks the PC for a task count of 28673, with all three
control bits above the register's defined fields. The PC starts that 28673-task program,
runs the one task it was handed, and is left mid-stream. No later job in the power session
starts. A power cycle resets the PC, and that reset is the only reason a gap between jobs
re-arms it.

Software must build the word from a 16-bit `TASK_NUMBER` field on the RK3576. Building the
word from a per-SoC width removes the defect, and leaves the RK3588 word bit-identical at 12
bits. Back-to-back submits with no gap then all write, byte-identical to
each other and bit-exact against the CPU model. A row-windowed 112×112 k3 convolution is
bit-exact over its whole output with no gap between tasks [HW sweep, H96 MAX M9, measured
2026-07-25]. The driver change is patch `rk3576/npu/0008` in the `rk3576-npu` profile.

`TASK_NUMBER` alone decides whether later jobs write. With the field at its 16-bit position,
every job writes whatever the three control bits hold, including with every control bit
clear. With the field at the RK3588 position, no control-bit value tried makes later jobs
write [HW sweep, H96 MAX M9].

No regcmd write removes the defect, so the fix is a driver patch [HW sweep, H96 MAX M9].

The `POINTER` field of `S_POINTER` is live, and selects a producer register group:

> [!CAUTION]
> `POINTER=1` in the CNA, CORE, DPU or RDMA `S_POINTER` regcmd makes every job write
> nothing [HW sweep, H96 MAX M9]. Software must leave `POINTER` at 0 in the register program.

Under `PP_MODE`, the `POINTER` field is hardware-owned [linux-rockchip RFC v4 4/6 cover and
code comments, Jiaxing Hu, 2026-08-03, their measurement, not reproduced here]. Writes to it,
from the driver or from the regcmd, change neither its readback nor the result. A second party
measured both on a Radxa ROCK 4D. The report agrees with the `S_POINTER` sweep on the
RK3576, which finds no value that changes the outcome of a second submit. It disagrees on
`POINTER=1` written from the regcmd, which stops every job on the H96 MAX M9 and changes
nothing in the report.

**Not characterized:** the cause of that disagreement.

The vendor `rknpu` config has three per-SoC PC parameters, and mainline `rocket` hardcodes
each at the RK3588 value:

| | RK3588 | RK3576 |
|---|---|---|
| `pc_task_number_bits` | 12 | **16** |
| `pc_task_status_offset` | `0x3c` | `0x48` |
| `pc_dma_ctrl` | 0 | **1** (writes `PC_DATA_ADDR` under the IRQ lock) |

Only `pc_task_number_bits` is required for back-to-back submits to write.
`pc_task_status_offset` is required to read the completion of a chained stream: the register
at `0x48` is the live task counter (§"Per-task `PC_DONE` and the `PC_TASK_STATUS` counter"
below). `pc_dma_ctrl`, the IRQ-locked `PC_DATA_ADDR` write, is not needed.

The RK3576 also has a per-SoC `state_init` hook that no other Rockchip NPU in that driver
has. It runs at every power-on with the PC in slave mode. It seeds CNA `0x1024` with bit 31
in both ping-pong register groups, then clears the pointer (`S_POINTER = 0x1e`). The `rocket`
driver has no equivalent, and a regcmd the PC is fetching cannot replay the slave-mode entry.

Neither the hook nor `pc_dma_ctrl` is needed. Back-to-back submits write without them, because
the task-number width alone causes the cold-start defect. Two jobs in flight on the two cores
are the workload that can reach the second pointer group. Neither the hook nor `pc_dma_ctrl`
causes their incorrect output [HW sweep, H96 MAX M9]. The mechanism that output is narrowed to
is in [rk3576.md](rk3576.md).

### Working against an unpatched driver

**Conditions:** a `rocket` module without patch `0008`, which includes the stock module.

`autosuspend_delay_ms` is the only setting userspace can change, and it does not remove the
defect. Writing 0 to `/sys/.../27700000.npu/power/autosuspend_delay_ms` makes the driver drop
the power domains straight after each job. A back-to-back loop then rises from 1 job in N
writing to about half. The setting races the autosuspend work. A job that arrives before that
work runs gets no power cycle and does not execute.

> [!CAUTION]
> With `autosuspend_delay_ms` at 0, a job that arrives before the autosuspend work runs leaves
> its output untouched. Software must check, per task, that its own output was written. If it
> was not, software must resubmit the task.

A gap of ~1 s between jobs makes results stable and repeatable. At ~0.4 s an output BO still
occasionally comes back wholly untouched.

> [!CAUTION]
> The cold-start defect has two signatures. Either the output BO comes back wholly untouched,
> or the DPU writes a full-size output that holds only the bias. In the second case the MAC
> contribution is missing and the output reads all zero. Both match a wrong register program.
> The second is the harder to recognize: it also matches a `C=0` coefficient buffer, so three
> unrelated causes give one result.
>
> Software must re-run a job that wrote nothing before reading it as a register result.
> Software must treat an all-zero output as the defect only where the CPU model predicts a
> non-zero output.

> [!NOTE]
> On this driver the second submit of every power session writes nothing, whatever register is
> under test. A sweep that varies a register between job 1 and job 2 therefore shows no change
> at any value. Before attributing a result to a configuration difference between two jobs,
> confirm that the second job can write at all.

> [!CAUTION]
> A job that writes nothing leaves the previous job's output in the BO, so a repeated
> configuration reads byte-exact. One configuration then appears byte-exact on every run, and
> any other appears to write nothing. A fresh BO's zeros cannot separate a job that never ran
> from one that wrote zeros (see the write guard in [rk3576.md](rk3576.md)). Software must stamp
> the output BO with a sentinel before each job, through a `PREP_BO`/`FINI_BO` bracket. A bare
> `memset` leaves dirty cache lines that race the DMA.

## The per-submit floor is the driver's completion poll

**Conditions:** H96 MAX M9, `rocket` with patches `0006`, `0009` and `0012`, from before
interface version 1.0, measured 2026-07-28. The per-submit time on later modules is in
[rk3576.md](rk3576.md) §"Per-submit overhead and the completion poll".

On the RK3576 the per-submit cost is set by the driver's completion poll, not by the NPU.
`PC_DONE` is read-only in `INTERRUPT_MASK`, so it cannot be routed to the GIC. The RK3588 takes
an interrupt, and on the RK3576 `patches/rk3576/npu/0006` polls the bit on an hrtimer instead.
The poll period sets the minimum cost of a submit, so the cost of RK3576 work follows its
submit count.

`patches/rk3576/npu/0009` makes the period a module parameter. Patch `0012` sets it to 50 us
and changes what the poll retires on. At that period a submit costs about a fifth of its time
at the stock 1 ms period [host-computed]. The shape is an 8×64×32 int8 matmul, at which
dispatch dominates compute [HW sweep, H96 MAX M9, 2026-07-31]. The per-submit time on each
module is in [rk3576.md](rk3576.md) §"Per-submit overhead and the completion poll".

> [!NOTE]
> A test suite's total wall-clock time does not show the per-submit cost. Host packing and the
> suite's CPU model dominate it, so it stays flat across poll periods. Time a single small shape
> instead.

### Chained row tasks

A row-windowed convolution is one submit per window, and its row tasks are independent. Each
writes its own rows of the same output from its own window of the same feature cube. The
weight and coefficient buffers stay unchanged across them. Chaining them into one kick saves
one completion poll per row task removed.

> [!CAUTION]
> Concatenating the row tasks' programs into one regcmd stream is arithmetically sound, but one
> drm task runs one PC program, whatever the stream holds. The driver programs `PC_TASK_CON`
> with `TASK_NUMBER = 1` per drm task descriptor, so the PC executes the first task in the
> stream and stops, deterministically. The rows of every later task are not written
> [HW sweep, H96 MAX M9, 2026-07-28]. Software must submit each program as its own task.

Saving the poll needs a kernel patch and a layout that userspace and the driver agree on.
Submitting n drm task descriptors is not enough on its own. The driver runs them back to back
(`rocket_job_hw_submit()` re-arms on `next_task_idx` from the completion path), but each
through its own completion poll. That form saves the ioctl and the fence round trip, and each
task still waits for a poll. The RK3588 series' `086` arrangement, ported as
`patches/rk3576/npu/0015` and `0016`, runs all n from one kick:

- Userspace lays the n programs out contiguously at one even-word stride. It rewrites each
  trailer's inert `OP_NONE` filler into a `PC_BASE_ADDRESS` write pointing at the next
  (`src/rocket_chain.c`, shared with the RK3588). The RK3576 emitter ends every program with
  the same `[OP_NONE, PC_REGISTER_AMOUNTS, OP_40, OP_ENABLE]` trailer that the rewrite expects.
- Userspace submits them as n drm tasks with `DRM_ROCKET_JOB_BATCHED`.
- The driver programs task 0 only, sets `TASK_NUMBER = n`, and advances `next_task_idx`
  straight to the end. The PC streams all n from one kick, and the job retires on the single
  completion that follows the last of the `TASK_NUMBER` tasks.

The `TASK_NUMBER` bound is per SoC, so `0016` takes it from `rocket_soc_data` (16 bits on the
RK3576, 12 on the RK3588) rather than from the RK3588 field mask.

Both trailer fields describe the next program. The driver programs `PC_BASE_ADDRESS` and
`PC_REGISTER_AMOUNTS` from task 0 alone. In a chain of programs that differ in length, software
must therefore write the successor's register count into each trailer, not the program's own.
A uniform row-task chain cannot distinguish the two.

> [!CAUTION]
> A kernel that does not honor `DRM_ROCKET_JOB_BATCHED` runs a chained layout down the
> per-task path. It runs task 0 and stalls until the job timeout. Software must confirm
> support before chaining. It reads `/sys/module/rocket/parameters/rocket_batch_submit`, or
> else the advertised DRM interface version, which must be >= 1.1.

Chained, every shape of the convolution library test gives the same result as one submit per
task [HW sweep, H96 MAX M9, 2026-07-29]:

| Convolution | One submit per task | Chained |
|---|---|---|
| 32×32 k3 plane, forced into 8 row windows | 2.8 ms | 1.0 ms |
| 224×224 k3 s1 | 21.3 ms | 19.1 ms |

Chaining saves one submit's cost per row task removed and nothing else. It shortens a call on
a deep plane and leaves a shallow one unchanged. Over a whole test suite the saving is small, because most
shapes plan into one or two row tasks. Host packing and the per-call power-cycle guard
dominate such a suite, not submits.

`librocketnpu` chains by default in its int8 convolution path, and
`ROCKET_RK3576_BATCH_TASKS=0` turns chaining off. `rocket_batched_submit_supported()` makes
the support check above, and the library does not self-chain where it fails. The same layout
runs a sequence of cube-joined layers as one kick (see the cross-layer kick in
[rk3576.md](rk3576.md)). The fp16 `ic` split does not chain this way. Its slices repack the
same weight BO, then read back and accumulate into the host's buffer between submits, so they
are not one stream. Chaining them needs per-slice weight buffers, and either per-slice output
regions or on-chip accumulation.

A chained stream honors read-after-write between its programs, although `PC_DONE` sets before
the writes are visible. Two 1×1 int8 convolutions were chained, the second reading the first's
output. One kick gives the same output as two submits, at planes of 2 KB to 387 KB [HW sweep,
H96 MAX M9].

### `PC_DONE` and the DPU completion bits

**Conditions:** H96 MAX M9, `rocket` with patch `0012`, from before interface version 1.0,
measured 2026-07-28. The `0017` results are on interface 1.2, measured 2026-07-29. On `rocket`
1.6.0 the wait for the DPU bit has no deadline by default (§"Tasks that raise no DPU
completion").

`PC_DONE` sets when the program counter finishes issuing, before the DPU's write DMA has
drained. The DPU's own completion bits set once the writes have reached memory, and they are
the retire signal on the RK3576. A driver that retires on `PC_DONE` tears the IOMMU mapping
down and signals the fence early. A long poll period hides both actions behind its own
detection lag, and with them two independent failures:

- **IOMMU detach.** The per-job IOMMU detach races the write drain. Detaching on completion
  faults writes that are still in flight, which leaves a *write* page fault active:
  `RK_MMU_STATUS` `0x2b` = `PAGING_ENABLED|PAGE_FAULT_ACTIVE|IDLE|PAGE_FAULT_IS_WRITE`.
  The fault blocks the next `enable_stall`, which the IOMMU reports as
  `Enable stall request timed out`. The companion status `0x1d` contains `STALL_ACTIVE`,
  so the stall did arrive, just after the poll gave up. The NPU then raises
  `DMA_READ_ERROR|DMA_WRITE_ERROR` and the job comes back short. Patch
  `patches/rk3576/npu/0011` removes this failure by keeping the domain attached across jobs.
- **Early fence.** The fence is signaled before the writes are visible. Patch `0012` closes
  it.

> [!CAUTION]
> A job retired on `PC_DONE` can return before its last bytes are written, and its output is
> then missing a contiguous tail. Software must retire a job on the DPU completion bit, with a
> bounded wait for the tasks that raise none.

`RAW_STATUS` contains a signal that sets once the writes have reached memory. The RK3588 takes
`DPU_0` (`0x100`) and `DPU_1` (`0x200`) as its completion interrupt. On the RK3576 they set
tens to hundreds of microseconds after `PC_DONE`. A log line taken at the moment the poll
retires reads:

```
raw@retire=0x20000001 -> 0x30000155 after 82us
```

The word first shows `PC_DONE_1` with the CNA feature bit and no DPU bit, then `DPU_0` 82 us
later. Retiring on the DPU bit keeps every result correct at short poll periods, which lets
the period drop:

| poll period | 500 us | 250 us | 125 us | 50 us | 25 us |
|---|---|---|---|---|---|
| us/submit, retire on `PC_DONE` | 1065 | 731* | 535* | 433* | n/a |
| us/submit, retire on the DPU bit | 1073 | 739 | 558 | 439 | 398 |
| int8 matmul test, retire on `PC_DONE` | 43/43 | 42/43* | n/a | 36/43* | n/a |
| int8 matmul test, retire on the DPU bit | 43/43 | 43/43 | 43/43 | 43/43 | 43/43 |

`*` marks an incorrect result. [HW sweep, H96 MAX M9, measured 2026-07-28]

The data of a failed job is not lost: the missing bytes are a contiguous tail that drains
after the fence. The drain reaches 343 us for a 32×2048×3072 output, and ~82-98 us for the
convolution test's smaller ones. It scales with bytes in flight [HW sweep, H96 MAX M9,
measured 2026-07-28].

#### Tasks that raise no DPU completion

Not every task raises a DPU completion, so the wait must be bounded. The narrow and wide int32
writers complete CNA and CORE and never set a DPU bit. Nor does a pool, which enables no DPU
stage (§"Pool completion"). The fp16 programs, the dense int32 writer and the one-byte int8
programs raise the DPU completion. [HW sweep, H96 MAX M9, rocket 1.6.0, 2026-09-27 and
2026-09-30]

The status word of an int32 writer's task holds at `RAW_STATUS = 0x30000055`
(`PC_DONE_0|PC_DONE_1|CORE_0|CNA_CSC_0|CNA_WEIGHT_0|CNA_FEATURE_0`) for milliseconds, with
nothing further arriving. Patch `0012` gives these tasks `rocket.dpu_grace_us` and then retires
them on `PC_DONE`, as the unpatched driver does. The grace is required, because those tasks
still need a settle. Its threshold is the same ~250 us that the settle measurement found:

| `dpu_grace_us` | 100 | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|
| int8 matmul test | 40/43 | 43/43 | 43/43 | 43/43 | 43/43 |
| through the wide int32 writer | 41/43 | 43/43 | 43/43 | 43/43 | 43/43 |

On modules before 1.6.0 the grace is also the deadline on the wait for the DPU bit that most
jobs raise. That role sets its default. It covers the drain of every ordinary narrow-output job
too. Lowering it retires those jobs on `PC_DONE`, exactly as the unpatched driver does. The two roles pull in
opposite directions, and both are required.

With the poll retiring on the DPU bit, the 32×2048×3072 output fails on every run at a grace
of 200 us. It fails on no run at 250 us and above. The failure point is sharp and
deterministic, so a grace of 250 us is 1.25x the measured failure point, with no margin above
it.

The longest drain measured is 745 us, for a 392 KiB output: `fc-rgb-224-k7`, the int8 stem at
224×224 k7 s2, reproducible to within 3 us. The 500 us default is therefore below the worst
drain measured on the RK3576, and the stem still passes at a grace of 200 us. Only the drain
of the last submit before the host reads must be covered. The submits that follow
a row-split job's earlier tasks cover those tasks' drains. That exposed drain is what the
failure point measures, and it is not separately observable. The raw drain, up to 745 us, is
therefore the only available bound on it.

On those modules the default is 500 us, 2.5x the measured failure point. The drain scales
with bytes in flight, and `librocketnpu` can emit outputs larger than any tested shape.

On `rocket` 1.6.0, `dpu_grace_us` defaults to 0, which waits for the writing block's own
completion with no deadline, and a nonzero value restores the deadline. An unflagged task that
raises no completion then waits out the driver's 125 ms backstop [HW sweep, H96 MAX M9, rocket
1.6.0, 2026-09-27]. The measurement is in [rk3576.md](rk3576.md) §"`dpu_grace_us` on
`rocket` 1.6.0".

Patch `patches/rk3576/npu/0017` (interface 1.2) separates the two classes at submit time. It
adds `DRM_ROCKET_JOB_NO_DPU_DONE` to `drm_rocket_job.flags`. A job with the flag waits
`dpu_blind_us` (default 250) past `PC_DONE` instead of `dpu_grace_us`. A job without it is
bit-for-bit unaffected. The poll still retires the moment a DPU completion arrives, whatever
the flag says.

A wrong flag still costs correctness. A flagged program that raises its own completion can
still be writing when the blind settle ends. Its fence then comes at the settle,
and the output's tail is unwritten ([rk3576.md](rk3576.md) §"Retirement on the DPU
completion").

`librocketnpu` sets the flag on the narrow and wide int32 writers only. The fp16 programs and
the dense int32 writer go out without it.

> [!NOTE]
> A build in which the flag does not reach the poll shows no effect at any `dpu_blind_us`.
> Before concluding that the flag does nothing, confirm that the job reaches it: an
> out-of-range `dpu_blind_us` must move the job's time. If `dpu_grace_us` moves it instead, the
> job is still waiting out the grace.

#### Completion and the wide-output hazard

The narrow and wide int32 writers raise no DPU completion and also trigger the wide-output
hazard (erratum RK3576-E2, [rk3576.md](rk3576.md) §"The poisoning is one hazard, and its cost
is a system setting"). The two are separate. The fp16 programs trigger the hazard and raise
their completion. With the DPU bit as the retire condition, the hazard's reach is unchanged
[HW sweep, H96 MAX M9, measured 2026-07-28].

On the module before interface 1.0, a submit that the hazard left unwritten raised its DPU
completion. A missing DPU bit did not detect the hazard there [HW sweep, H96 MAX M9, measured
2026-07-28]. On `rocket` 1.6.0 it raises none, and the driver retires it at the 125 ms
backstop [HW sweep, H96 MAX M9, 2026-09-30]. On either module, a sentinel scan of the
output detects the hazard.

> [!NOTE]
> At a long poll period the DPU bits appear redundant: `DPU_0` is already set on the same poll
> tick as `PC_DONE`. At 500 us of detection lag the drain has always finished by the time the
> poll looks, so the two signals cannot be told apart. At 50 us they separate cleanly.

### Per-task `PC_DONE` and the `PC_TASK_STATUS` counter

`PC_DONE` is per task, and `PC_TASK_STATUS` is the live task counter. The bits
`PC_DONE_0` and `PC_DONE_1` are two alternating per-task pulses, not one whole-kick
completion. On a `TASK_NUMBER = n` kick the two bits swap as the program counter retires
each program. The first of them appears a few tens of microseconds into a stream that
can run for milliseconds. It does not mean the stream is over.

The bits are also not latched across tasks. Mid-stream the whole `RAW_STATUS` word cycles
through `0x10000000`, `0x20000000`, `0x30000000` and `0x00000000`. The per-task block
completion bits (`0x155` for one ping-pong group, `0x2aa` for the other) appear only
briefly and are gone again by the next task. Only the last task's block bits survive,
because no task follows to clear them. [HW sweep, H96 MAX M9]

`PC_TASK_STATUS` gives the position in the kick. The register is at `0x48` on the RK3576 and
at `0x3c` on the RK3588. The vendor `rknpu` config has the difference as
`pc_task_status_offset` and reads the register as its "task counter". It holds two 16-bit
counters, **both modulo the programmed `TASK_NUMBER`**:

| bits | field |
|---|---|
| 15:0 | tasks started |
| 31:16 | tasks completed |

While task `k` (0-based) of `n` is in flight, it reads `(k << 16) | (k + 1)`. While the
last one runs, `started` has wrapped and it reads `(n - 1) << 16`. Once every task has
retired, both counters wrap and it reads 0, which is also what it reads before the first
task starts. The model holds on kicks of 2, 3, 5, 8, 9, 35 and 90 tasks. The RK3588's
reported `0x0000f000` at IRQ time fits the same model: `& 0xfff == 0` is the completed
count having wrapped, meaning the whole job is done. [HW sweep, H96 MAX M9]

Mid-stream at least one of the two halves is non-zero. `PC_DONE` set together with a zero
`PC_TASK_STATUS` therefore means the whole kick is over, and no other state does. Software must
place the completion wait of a chained stream on that combination. A trace of the 35-program
cross-layer kick of a MobileNetV1-224, at a 10 us poll:

```
t=   12 us   PC_DONE_0,  ts=0x00000001   task 1 of 35 in flight
t=   31 us   PC_DONE_1,  ts=0x00000001
 ...         the two bits alternating, once per task
t= 1911 us   PC_DONE_1,  ts=0x00220000   task 35 of 35 in flight
t= 1921 us   0x300002aa, ts=0x00000000   the DPU's own completion; the kick is over
```

`PC_OPERATION_ENABLE` is a self-clearing go bit. It reads back 0 at every poll of a kick that
is still running [HW sweep]. The vendor driver writes `1` and then `0` to it back to back
(`rknpu_job.c`) [source-confirmed].

> [!CAUTION]
> A completion test of the form `OPERATION_ENABLE == 0 || PC_DONE` is true on the RK3576 from
> the first poll onward. Any wait built on that test ends at the first poll. Software must not
> use `PC_OPERATION_ENABLE` as a busy flag.

### The half-started job runtime-PM pin

**Conditions:** mainline `rocket` without patch `0010`. The defect is in mainline code, so a
kernel without `0010` has it on the RK3588 as well. Measured on H96 MAX M9, 2026-07-27. In the
RK3576 series `0010` precedes `0015`, the patch that declares interface version 1.0, so a
module that reports an interface version contains the fix.

A half-started job pins the NPU runtime-active, and then only the int32 path breaks. Patch
`patches/rk3576/npu/0010` fixes it.

`rocket_job_run()` takes a runtime-PM reference and then attaches the IOMMU group. Both
of its early returns leave the job half-started. Each of two independent leaks is enough
on its own to pin the device runtime-`active`, with `power/runtime_suspended_time`
frozen. The pin holds even with nothing running and `power/control` at `auto`:

- **Usage counter.** The usage counter only climbs. `pm_runtime_get_sync()` keeps its
  reference when the resume fails, and the IOMMU path has already resumed successfully. At
  that point `core->in_flight_job` is still `NULL`. The completion path and the timeout path
  put the reference only when a job is in flight, so neither balances it.
- **Scheduler credit.** Nothing will ever signal the fence handed to the scheduler, so the
  job stays pending and the scheduler never gets its credit back. While a credit is
  outstanding, `rocket_device_runtime_suspend()` returns `-EBUSY`.

The leaked usage-counter reference outlives the driver. The reference is on the platform
device, which belongs to the OF core, so `rmmod`/`insmod` does not clear it. A freshly probed
device comes back `active` and stays there until a reboot.

> [!CAUTION]
> On a device pinned runtime-active, every int32-output shape writes nothing, and every other
> path still works. The power domain's cycling is what clears the int32 output writer's
> wide-output hazard, and on a device that cannot suspend the hazard never clears. The symptom
> matches a defect in the int32 path. A matmul measured in this state fails at every poll
> period, whatever the period. Software must check `power/runtime_status` before trusting any
> int32 result.

Without `0010` the state also sustains itself. One failed attach times out at 500 ms, and the
timeout path calls `rocket_core_reset()`. That reset takes the IOMMU down, so the next attach
fails for real. With `0010` the driver retires the failed job with its error, the caller's
next submit attaches and computes normally, and the device suspends. [HW sweep, H96 MAX M9,
measured 2026-07-27]

### A per-job core reset is not the way to clear the poisoning

**Conditions:** H96 MAX M9, a `rocket` from before interface version 1.0, measured
2026-07-27.

`rocket_core_reset()` is a `reset_control` bulk assert, `udelay(10)`, then a deassert, and the
driver runs it on a job timeout. Called at job start, it does not replace the power-cycle idle
that an int32-output job forces on the next submit. Asserted before every job, it takes the
IOMMU down with it:

```
rk_iommu 27702000.iommu: Error during raw reset. MMU_DTE_ADDR is not functioning
rocket 27700000.npu: NPU job timed out: RAW_STATUS=0x30000000 MASK=0x800fffff OP_EN=0x00000000
```

`RAW_STATUS` contains both `PC_DONE` bits, so the job ran, but nothing reached DDR because the
page-table pointer was gone. The int8 matmul test falls from 43 shapes passing to 2, and the
convolution test to 0. The damage outlives the setting: the NPU stays wedged until the module
is reloaded. The driver's own timeout path resets inside `drm_sched_stop()`, an IOMMU detach
and a re-attach, and the bare call lacks that surrounding re-initialization. The vendor
`rknpu` driver never resets per job either. [HW sweep, H96 MAX M9, measured 2026-07-27]

The vendor's recovery reset covers the CBUF, and `rocket`'s does not. `rknpu_soft_reset()`
runs on a job timeout or on request, never per job. It sleeps 100 ms, then asserts all four
of the node's resets for 10 us: `SRST_A_RKNN0`, `SRST_A_RKNN1`, `SRST_A_RKNN_CBUF` and
`SRST_H_RKNN_CBUF`. It then detaches and re-attaches the IOMMU and re-runs its `state_init`
[source-confirmed: BSP `rknpu_reset.c` and `rk3576.dtsi`, unchanged in `develop-6.12`].

`rocket`'s binding names the core's `srst_a` alone, so under `rocket` only a power cycle
resets the CBUF.

**Not characterized:** whether a CBUF reset inside the timeout path's detach and re-attach
clears the wide-output hazard. If the latched state is in the CBUF, it clears without cycling
the domain [expected].

## The IOMMU wedge

**Conditions:** H96 MAX M9, a `rocket` from before interface version 1.0, measured 2026-07-27.
The two wedges below were induced independently, one by a 100 us poll period and one by a
per-job core reset.

A run can leave the NPU wedged, and the NPU does not recover on its own. The signature in
`dmesg` is:

```
rk_iommu 27702000.iommu: Enable stall request timed out, status: 0x0000..
rocket 27700000.npu: NPU job timed out: RAW_STATUS=0x30000000 MASK=0x800fffff OP_EN=0x00000000
rk_iommu 27702000.iommu: Error during raw reset. MMU_DTE_ADDR is not functioning
```

> [!CAUTION]
> After that sequence, every submit times out and every output BO comes back untouched. The
> result matches a regcmd encoder that writes nothing. The rail, the clock and the driver binding all
> still look healthy in this state, so they do not distinguish it. Software must check `dmesg`
> for the stall timeout before trusting any negative result on the RK3576. [HW sweep, H96]

The `Enable stall request timed out` and `MMU_DTE_ADDR is not functioning` sequence above is
the per-job IOMMU detach racing the DPU's write drain. With patches 0010 and 0011 the wedge is
much harder to reach. Patch 0011 stops detaching per job. Patch 0010 stops one failed attach
from cascading through the timeout path's core reset into the next attach.

Reloading the module clears the compute path, so try it before a reboot. The reload
`rmmod rocket; sleep 8; insmod rocket.ko` re-runs the IOMMU attach, and the convolution tests
pass again immediately. The settle is required: a reload with a two-second gap left the NPU
still failing, where an eight-second one cleared it. The reload cleared both induced wedges.

On a kernel without `patches/rk3576/npu/0010`, a reload does not clear everything. A leaked
runtime-PM reference is on the platform device, so a reload leaves it in place. The NPU then
computes convolutions correctly while every int32 matmul writes nothing, as the half-started
job section above describes. Check `power/runtime_status` after a reload. If it does not
return to `suspended`, reboot. [HW sweep, H96 MAX M9, measured 2026-07-27]

## Kernel oops on a rejected submit

**Conditions:** mainline `rocket`. The code is identical in a pristine v7.1 tree and present
since the submit ioctl was added, and the RK3576 series does not introduce it. Mainline
declares no interface version. Measured on H96 MAX M9, kernel 7.1.3.

Any submit that `rocket_ioctl_submit_job()` rejects before it assigns the job's IOMMU domain
oopses the kernel. The function allocates its `rocket_job` with `kzalloc`. It assigns
`rjob->domain` only after the task copy and both BO lookups have succeeded, while
`rocket_job_cleanup()` puts that domain unconditionally. Every rejection ahead of the
assignment therefore unwinds into `rocket_iommu_domain_put(NULL)`, which dereferences the NULL
to reach the kref:

```
Unable to handle kernel NULL pointer dereference at virtual address 0000000000000008
Internal error: Oops: 0000000096000004 [#1]  SMP
pc : rocket_iommu_domain_put+0x50/0xac [rocket]
lr : rocket_job_cleanup+0x20/0x23c [rocket]
Call trace:
 rocket_iommu_domain_put+0x50/0xac [rocket] (P)
 rocket_job_cleanup+0x20/0x23c [rocket]
 rocket_ioctl_submit+0x3e0/0x7cc [rocket]
```

Five rejection sites are reachable, all of them ahead of the assignment:

- `task_struct_size` below `sizeof(struct drm_rocket_task)`
- An unreadable `drm_rocket_job.tasks` pointer
- `drm_rocket_task.regcmd_count == 0`
- Either BO handle not resolving

The device node `/dev/accel/accel0` is group `render`, so the trigger is unprivileged. Patch
`patches/rk3576/npu/0013` fixes it by taking the domain reference at construction, so a job
holds one from the moment it exists. With the fix, every site returns to userspace
[HW sweep, H96 MAX M9, 7.1.3]. Patch `patches/rocket/085` adds the complementary NULL guard in
the put. The two touch different files and compose.

A generator that refuses leaves its output parameters untouched, and a caller that submits
anyway hands the kernel a malformed job. `gen_matmul_fp16()` emits the RK3588 geometry encoding
and refuses on the RK3576 by construction. The refusal leaves `matmul_params_t.task_count` at 0,
and task descriptors built from those unset parameters submit `regcmd_count == 0`. Software
must check the generate step before building a submit from it.

A single oops does not disable the NPU: an ordinary client and the full 46-shape matmul test
both still run afterwards. Several oopses do. After three, the file-close path faults again
inside the IOMMU layer (`iommu_map_nosync`, `iommu_unmap`, `iommu_domain_free` from
`rocket_postclose`). The process then cannot be killed, and the kernel prints
`Fixing recursive fault but reboot is needed!`.

> [!CAUTION]
> Results measured on a kernel that has taken an oops are not about the NPU. Software must not
> run tests on such a kernel. Reboot it first.

### The DMA write error warning

The warning at `rocket_job.c:639` in `rocket_job_irq_handler()` is a different thing, and it
is not a driver defect. It is mainline's
`WARN_ON(raw_status & PC_INTERRUPT_RAW_STATUS_DMA_WRITE_ERROR)`, the driver reporting that
the NPU raised a DMA write error. Seen on core 1's IRQ during a two-core run, it is the
kernel-log sign of the concurrent-job corruption, not a separate defect.

### Client-supplied addresses

A client-supplied bad address does not reach that `WARN_ON` on the RK3576. The field
`drm_rocket_task.regcmd` is a raw IOVA. The driver's `rocket_job_hw_submit()` writes it into
the PC block's `BASE_ADDRESS`, with no check that the job's BOs mapped it. A client can
therefore aim the instruction fetch at unmapped memory. Through a real program with a
rewritten output address, it can aim the DPU's write DMA there too.

On the RK3576 both cases surface as `rk_iommu: Enable stall request timed out` and nothing
else. No WARNING is printed, the taint is unchanged, and the matmul test passes all 46 shapes
immediately afterwards. The RK3576's IOMMU absorbs a bad address as a stall, rather than the
NPU's PC raising a DMA-error bit [HW sweep, H96 MAX M9, 2026-07-28].

A `WARN_ON` on a hardware error condition taints the kernel, and panics it under
`panic_on_warn`. The `WARN_ON` is therefore an unprivileged denial-of-service concern upstream
in principle. No client-triggerable path to it is demonstrated on the RK3576.

> [!CAUTION]
> A job whose program faults still retires cleanly. The submit returns 0, `PREP_BO` returns 0,
> the output BO is untouched, and the driver tells userspace nothing. The result matches a regcmd
> encoder that writes nothing, arriving from the driver rather than from the regcmd encoder. Software
> must not treat an untouched output BO as evidence about an encoding.

On the RK3588 the same two cases reach a client-triggerable warning other than that `WARN_ON`.
A client can therefore taint the kernel through the NPU there. Both cases end in the job
timeout, and the reset behind it walks into the IOMMU core:

```
rk_iommu fdaca000.iommu: Enable stall request timed out, status: 0x2b
WARNING: drivers/iommu/iommu.c:157 at __iommu_group_set_core_domain
 iommu_detach_group / rocket_reset.part.0 [rocket] / rocket_job_timedout [rocket]
```

The NPU is still faulting when `rocket_reset()` detaches the group, so the IOMMU's stall and
disable-paging requests time out and the detach WARNs. Patch `patches/rocket/083`
(keep-the-domain-attached-across-jobs) is applied on that board and does not cover it, because
the reset detaches explicitly. On both SoCs the path a client reaches is the reset, not the
IRQ handler, and the two differ only in whether the detach WARNs. [HW sweep, Turing RK1,
2026-07-28]

## A BO in an in-flight job must survive the file that made it

**Conditions:** mainline `rocket`, in code that the RK3576 enablement does not touch, and which
declares no interface version. Measured on H96 MAX M9, 2026-07-28.

The per-context IOVA allocator (`drm_mm` + `mm_lock`) is in `struct rocket_file_priv`, and
`rocket_postclose()` tears it down and frees it. Only `rocket_gem_bo_free()` removes a BO's
IOVA node. The drm_sched free worker drops a job's BO references asynchronously, and it can run
after the owning file has closed. A client that submits and closes without waiting therefore
leaves `rocket_gem_bo_free()` taking `bo->driver_priv->mm_lock` and calling
`drm_mm_remove_node()` on memory `rocket_postclose()` already freed. The device node
`/dev/accel/accel0` is group `render`, so the trigger is unprivileged.

The defect reproduces on demand. A client that opens a fresh fd, submits and closes it
immediately triggers the warning on every iteration. Left running longer, the same client
reaches the dereference:

```
Internal error: Oops: 0000000096000004 [#1]  SMP
Workqueue: 27700000.npu drm_sched_free_job_work
pc : add_hole+0x34/0x15c
lr : drm_mm_remove_node+0x1e8/0x380
 rocket_gem_bo_free [rocket] / drm_gem_object_free / rocket_job_cleanup [rocket] /
 rocket_job_free [rocket] / drm_sched_free_job_work
```

The defect is rare in ordinary use only because an ordinary caller waits for its job, and so
never sets the race up.

Patch `patches/rk3576/npu/0014` anchors the allocator to `struct rocket_iommu_domain`, which is
refcounted and outlives the file, every mapped BO and the attached core. With it, no iteration
warns. It is the RK3588 series' `084` ported, and it applies to the RK3576 series with offsets
only. [HW sweep, H96 MAX M9, 2026-07-28]

> [!CAUTION]
> This defect fails no syscall, so a run can pass every ioctl while the defect fires on every
> iteration. A 7.1 kernel prints
> `warning: drivers/gpu/drm/drm_mm.c:965 at drm_mm_takedown+0x28/0x38`, and the older
> "allocator still has nodes" wording does not appear at all. A detector keyed on that message
> therefore reads clean. Software must detect the defect in the kernel log, and match the
> symbol, not drm_mm's message text.

## The matmul as a 1×1 convolution

A matmul is a 1×1 convolution over these blocks, exactly as it is on the RK3588. The A rows
are the convolution's spatial pixels, K is the input-channel axis, and N is the output-channel
axis. The register program is the convolution's, unchanged: `ic=4608`, `oc=128` on a 4×2 VALID
plane is M=8, K=4608, N=128 in one submit. The matmul work is therefore all in userspace:

- The operand scatter
- The de-tile
- A tiling planner
- A dispatch point

The `librocketnpu` entry is `rocket_matmul_int8_rk3576()`.

The mapping, for software that drives the emitter directly:

| operand | layout |
|---|---|
| A[M,K] | the feature cube, channel `k`, pixel `m` at `(m/iw, m%iw)`, C2 = 16 int8 lanes |
| B[N,K] | `weight_conv_int8(N, K, 1, 1, n+1, k+1, 1, 1)` |
| C[M,N] | the int8 output cube, `(n/16)*surf_elems*16 + 16*(y*iw + x) + n%16` |

The plane is free. Any `(iw, ih)` with `iw*ih == M` holds the M rows, and every such plane
gives the correct result, so the choice is a tiling one. The plane costs granules: a feature
row takes `ceil(iw*K/64)` of them against the task's 4096. One task therefore holds
`262144/K` pixels however the plane is cut, provided `iw*K` is a whole number of granules.
When it is not, every row rounds up and the waste is real. Take the widest divisor of M that
divides evenly.

### The M axis

The RK3576 places no constraint on M. Every M tried (1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 16, 17,
31, 32 and 64) is bit-exact at every factorization into a plane, from `1xM` through `Mx1`.
`M=1` needs no padding here, the opposite of the RK3588. There, rows are the convolution's
spatial height, a height under 4 produces incorrect output, `M%4` must be 0, and software pads
`M==1` to 4. The per-submit cost, not correctness, decides whether a matmul runs at M=1.
[HW sweep, H96 MAX M9]

### Matmul precision

**Conditions:** H96 MAX M9, measured 2026-07-27. Both precisions run as 1×1 convolutions
through the convolution programs.

Through the convolution programs, int8 is the faster matmul precision by a large margin. The
two programs contract at very different rates. One int8 task takes `ic*kh*kw <= 4608`, so
K = 4608 fits in one submit. One task of the fp16 convolution program contracts exactly
sixteen input channels, so the same K costs 288 submits [HW sweep]:

| shape | int8 submits | int8 ms | fp16 submits | fp16 ms | ratio |
|---|---|---|---|---|---|
| 16×512×64 | 1 | 1.4 | 32 | 42.9 | 30x |
| 32×1024×128 | 1 | 1.3 | 64 | 85.2 | 64x |
| 128×2048×128 | 1 | 1.2 | 128 | 171.3 | 142x |
| 56×4608×128 | 1 | 1.3 | 288 | 386.1 | 299x |

On the RK3588, fp16 is the faster precision and int8 reduces only memory use. On the RK3576
the order is reversed.

The ratios above come from the contraction width, not the arithmetic.
The fp16 convolution program contracts 16 input channels a task, and the int8 program 4608
[HW sweep]. The 16 belongs to that program, not to the SoC. The matmul-form fp16 program
contracts the whole of K in one task. Its cost against the int8 matmul is in
[rk3576.md](rk3576.md) §"The precision fork: what the op library should be founded on here".

### Throughput and the N axis

**Conditions:** H96 MAX M9, `M*K = 262144`, measured 2026-07-27.

Matmul throughput on the RK3576 is set by the MACs per submit. A submit costs the same
whatever work it contains (§"The per-submit floor is the driver's completion poll"). The feature budget caps `M*K`, and the resident weight slice caps K,
so N is the only free axis. At the feature-budget cap, throughput scales almost linearly with N
[HW sweep]:

| N | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 2560 |
|---|---|---|---|---|---|---|---|---|
| GOP/s | 12 | 24 | 47 | 93 | 190 | 367 | 719 | 1000 |

Throughput does not depend on how `M*K` is split: 1024×256, 256×1024 and 64×4096 measure
within a few percent of each other. [HW sweep]

### Output-channel bounds

A matmul task on the RK3576 has two output-channel bounds [HW sweep, H96 MAX M9]:

| Bound | Limit | Past the limit |
|---|---|---|
| Output channels | 2944 | Intermittent at 3072, and never correct at 4096 |
| Whole weight cube | 6 MiB, as at `ic=4096, oc=1536` | Never correct at `ic=4608, oc=1536` (6.75 MiB). About a quarter of the channels lost at `ic=4096, oc=2048` (8 MiB) |

> [!CAUTION]
> Past either bound, the trailing output channels do not reach DDR. The output is full and
> correctly sized, its leading channels are bit-exact, and nothing faults. Software must keep
> every matmul task within both bounds.

The weight-cube bound was measured at `k=1`, the matmul's own kernel. It is stated over the
whole cube rather than per kernel tap. A `k>1` convolution is therefore refused earlier than
any failure measured there, which is the safe direction. No convolution shape in the convolution
test suite comes near either bound: the largest weight cube is 516 KiB, at oc=128.

In `librocketnpu`, the planner `r76_plan_cbuf` refuses a shape past either bound.

> [!NOTE]
> A single run per point does not locate these bounds, because failures near them are
> intermittent. Repeat each point three times and classify each failure. An all-sentinel
> output is a dead submit, a clean prefix is a capacity bound, and wrong values in a fully
> written output are arithmetic.

### The int32 output

With DPU `0x4010`'s output-width field at `precision_int32` and OUT_CVT at exact unity, the DPU
writes the raw 32-bit accumulators of an int8 program. The words are the accumulators
themselves, laid out on the RK3588's int32 cube map in 32-bit words:

```
word = (c/4) * ow*oh_full * 4 + 4 * (y*ow + x) + (c%4)
```

> [!CAUTION]
> With the int32 output width, only the first eight output channels of every thirty-two reach
> DDR. The other channels are not written. Software must place every real output channel in a
> delivered slot through the weight cube, as described below.

The map is decoded at `oc` 4, 8, 16, 32, 64 and 96, `K` 16 through 256, and `M` 4 and 8. Every
position fits it with no misses.

The writer keeps the int8 output's byte budget whatever the element width. It writes
`2*ceil(oc/32)` contiguous blocks of `ow*oh_full` 16-byte atoms, exactly the int8 output of
`oc` rounded up to 32 channels. `oc` 8 and `oc` 16 both write two blocks [HW sweep, H96 MAX
M9]. At four bytes an element, each block holds four channels where an int8 block holds sixteen. Block `j`
holds channels `32*(j/2) + 4*(j%2) .. +3`.

The first eight output channels of every thirty-two therefore reach DDR, and nothing else is
written. At `oc <= 8` every channel is delivered and the output is complete. At a multiple of
32, the number of channels delivered is `oc/4`.

The weight cube works around the rule, and no register changes it (§"Output-channel
registers"). Software must program a multiple of the output channels and put real channel `n`
in a slot the writer delivers, with the other slots zero. Every real channel is then placed in
a delivered slot, and the output reads back as a plain cube. On this writer, the narrow
writer, the multiple is four and the slot is `32*(n/8) + n%8`. That placement reduces the map
above to the plain int32 cube.

The workaround costs output channels, not bytes: the byte budget comes out at exactly the
int32 output. But the resident weight slice, the N tile and the 2944-channel bound are all
functions of the programmed `oc`. The multiple therefore costs MACs per submit. The cost
follows from the measured budget rather than from the encoding, so no register removes it.
Setting `PROC_PRECISION` halves the multiple (§"The wide writer").

The regcmd encoder's `gen_conv2d_int8_rk3576_i32out()` emits this program. The entry
`rocket_matmul_int8_rk3576_i32()` places the channels this way when its dense writer is off
(`ROCKET_RK3576_I32_DENSE=0`, or a writer forced with `ROCKET_RK3576_I32_OC_MULT`), and it is
bit-exact against a CPU int32 model. By default the entry writes through the dense writer
(§"The dense writer: every accumulator as an int32, and no poisoning").

#### Output-channel registers

No output-channel register on the RK3576 changes the delivered-channel rule. Scaling the five
output-channel registers in the table together increases the address extent and nothing else.
The rule is therefore in the datapath rather than in a geometry field. [HW sweep, H96 MAX M9, measured 2026-07-27]

At `M=8, N=32` on an 8×1 plane the baseline is 64 words:

| driven | extent | what it did |
|---|---|---|
| nothing (widths 0, 4, 5) | 64 words, last at 63 | the baseline, identical at all three widths |
| `0x402C` and `0x4030` high half, oc x4 | 64, last at 63 | nothing at all |
| `0x401C` (DST_SURF) doubled | 64, last at **95** | spaced the blocks apart. It is the block stride |
| `0x40B8` (SURFACE_ADD) doubled | 64, last at 63 | inert at `ih=1`, where the task window is the plane |
| `0x401C` and `0x40B8` doubled | **0 words** | the DPU wrote nothing, so the two are coupled and `0x40B8` is live |
| `0x4030` low half 0x710 -> 0x310 | 64, last at 63 | nothing (the 7-vs-3 the RK3588 calls `size_e`) |
| `0x4030` low half -> 0xF10 | 64, last at 63 | corrupted the values without moving the extent |
| CNA `0x1024` (kernel word \| oc-1) x4 | 64, last at 63 | nothing |
| DPU_RDMA `0x5014` (CUBE_CHANNEL) x4 | 64, last at 63 | nothing |
| CORE `0x3020` (dataout_channel) x4 alone | **0 words** | the DPU wrote nothing. It is coupled to the others, not inert |
| all five oc registers x4 together | **256**, the full output | the byte budget scales, the delivered channels do not, still 8 |

### The wide writer

The wide writer delivers the first eight output channels of every sixteen, so a real channel
costs two programmed channels rather than four. `PROC_PRECISION`, DPU `0x4010`'s low field
(`[2:0]`), is the DPU's own operand width. Set from int8 to int32, it doubles the byte budget:
the writer emits two 16-byte atoms per (16-channel block, pixel) instead of one. Nothing else
changes. The operands stay int8 everywhere, the arithmetic is bit-identical, and the CNA and
CORE programs are byte-unchanged.

The regcmd encoder's `gen_conv2d_int8_rk3576_i32out_wide()` emits the wide writer.

Two atoms is the maximum. The field's eight values at `M=8, K=32, N=32` give:

| `0x4010[2:0]` | extent | what came back |
|---|---|---|
| 0 int8, 6 int4 | 64 words | one atom per (block, pixel), the 4x rule |
| 1 int16, 4 int32 | 128 words | two atoms, the 2x rule, values correct |
| 2 fp16, 3 bf16, 5 | 128 words | two atoms, and the operands reinterpreted: no accumulator survives |
| 7 tf32 | n/a | a dead submit, nothing written |

The map works in 32-channel super-groups, each `4*A` atoms long, where `A` is the output's
pixel count `ow*oh_full`. Inside one super-group the writer emits a single linear stream
indexed by `s = 2*p + j`. In that index `p` is the pixel and `j` the 16-channel block. The
writer cuts the stream into segments of `A` atoms. The segment index also contains the lane
group `L = (c%16)/4`, which is the slower axis:

```
atom = 4*A*(c/32) + A*(2*(s/A) + L) + s%A       s = 2*p + j
word = 4*atom + c%4                             delivered iff c%16 < 8
```

Because the map is a stream, nothing rounds: an odd `A` cuts it at an odd place, and `A = 1`
works as readily as `A = 16`. The map is decoded at `oc` 8, 16, 24, 32, 40, 48, 64, 96, 128
and 192. The pixel counts decoded are 4, 5, 6, 7, 8, 12, 13, 16, 21 and 33.

#### Wide-writer bounds

The wide writer has two bounds, and past either one its output is incorrect with no error.

> [!CAUTION]
> On the wide writer, an `oc` that is not a multiple of 32 produces incorrect output. A partial
> super-group is not a truncated one. At `oc = 16 (mod 32)` the trailing group holds
> one 16-channel block instead of two, and packs at one atom per pixel with no stream. At an
> `oc` that is not a multiple of 16, the delivered channel set rotates with the pixel and has no
> block form. The case read off is 24. Software must program `oc` as a multiple of 32 and leave
> the padding channels zero.

`librocketnpu`'s `rocket_rk3576_pad_oc()` gives the padded count.

> [!CAUTION]
> A wide-writer task whose output exceeds 8 KiB produces incorrect output, and the bound is over
> the whole plane. Stated in the programmed channel count, a row task is correct while
> `iw * oh_full * oc_prog < 4096`. Software must keep every task within that bound, including
> its `iw` term.

A wide task writes `oc_prog/8` atoms a pixel at 16 bytes each. The product
`iw * oh_full * oc_prog` is therefore the output in two-byte units, and the bound is 8192
bytes, or 512 atoms, per task [host-computed].

A row split can shorten `oh_full` but cannot narrow `iw`, and the plane chooser takes the
widest divisor of M it can. On a shape whose plane is `Mx1`, a cap that divides only by
`oc_prog` passes every task. The output is then still M times over the bound, and incorrect. At
`iw = 1`, where the two forms coincide, the same shapes are exact. A plane too wide for even a
one-row task cannot be split into one, so such a shape cannot use the wide writer.
[HW sweep, H96 MAX M9]

On the wide writer, the entry `rocket_matmul_int8_rk3576_i32()` splits the row plan until
every task fits the bound. It refuses a plane too wide for a one-row task. A row task here is a
standalone 1×1 convolution with its own output, so the bound costs submits and nothing else.

A longer contraction moves the bound outward, and never inward. The same `oc = 128` that is
wrong from `oh_full = 16` at `K <= 128` is wrong at only 31, 32 and 40 at `K = 512`. It is exact
to `oh_full = 40` at `K = 1024` and `K = 2048`. The 8 KiB bound is therefore the worst case,
reached once the contraction is short enough. Software must enforce it at every K.

The bound holds for a weight cube whose undelivered channels are zero. With all sixteen
channels of a group driven, the map stops fitting well before the bound, at `oc = 192` from
`A = 48` on a flat plane. With the undelivered half of each sixteen zero, the same geometries
are bit-exact at `A` 48, 64 and 128.

#### Zero atoms and their detection

Both int32 writers intermittently write a few atoms as zero inside the bound, while the rest
of the output is exact. On the wide writer the zeroed block is contiguous and one super-group
wide. On the narrow writer it is a handful of atoms. Both writers show the same defect.

> [!CAUTION]
> A zero atom cannot be told from a zero accumulator by its bytes. The defect also repeats often
> enough to pass a "same result twice, so it is data" rule. Only the arithmetic separates the
> two. Software must check every atom of an int32 writer's output against a sentinel, and must
> verify each zero atom against the operands.

A whole-task check, "did this task write anything at all", answers "no" on every legitimately
zero output, so it cannot serve as the detector.

The rate tracks host DDR traffic rather than CPU load. At `M=128 K=256 N=2048`, 5.8% of row
tasks return a zero atom on a quiet host, and 51.7% under a streaming host `memcpy` to DDR. Host
load that keeps the CPUs busy and the caches hot without reaching DDR does not raise the rate
measurably. Across the same contrast, the count of atoms never emitted at all barely moves. That
count is the wide-output hazard (§"The wide-output hazard after an int32-output job"), so the
two hazards are separate. [HW sweep, H96 MAX M9, 2026-07-28]

The defect does not reach the conv entries or the narrow int8 matmul: under the same DDR load,
neither fails a case. The conv entries' own write check is per row task, so it cannot see a
task that wrote all but a few atoms. [HW sweep, H96 MAX M9, 2026-07-28]

Every wrong element comes back zero, past the bound and inside it alike. No wrong element holds
another element's correct value, or anything else. The writer emits empty atoms rather than
misplacing full ones.

The writer writes the zero atoms. With the output BO stamped with a sentinel before the tasks
run, the bad atoms come back zero rather than holding the sentinel. On a fresh BO, which arrives
zeroed, an unwritten atom and a zeroed one are indistinguishable.

The zeroed block reads most clearly in the wide writer's own coordinates. The writer emits two
atoms per stream position `s`, one per lane group, so its emission order is `(s, L)` ascending.
The zeroed block is contiguous in *that* order and in no other. In address order it shows up as
two separate holes, and it lies inside a single super-group.

The block's extent varies by shape, so a detector cannot key on it. On one shape it always ends
two emissions short of the group's last. On others it is three emissions in mid-stream, at a
fixed address. [HW sweep, H96 MAX M9]

A redo for a zero atom needs no idle in front of it. An atom that still holds the sentinel is
the wide-output hazard instead, and that redo keeps its clearing idle.

`librocketnpu` detects the defect and verifies it exactly rather than avoiding it. The signal is
two adjacent zero emissions on the wide writer, or any zero atom on the narrow one. The library
stamps a sentinel, so "never emitted" is known per atom, and redoes the task once, which clears
most zero atoms. From the second attempt it computes the zero atoms' accumulators on the CPU
over that K slice. It keeps the output only if they really are zero. The functions are
`r76_i32_wide_suspect()` and `r76_i32_zeros_are_data()` in `rocket_matmul_rk3576.c`.

Sparse data does not cost either writer its retry budget: with every row of `A` zero, both
writers pass.

#### Cost and the writer choice

The N tile is bounded by the programmed output channels, so halving the multiplier doubles the
tile and halves the submits. On this path a submit costs a clearing idle rather than 1.4 ms. Below the
tile cap the two writers cost the same, so a shape whose N fits one tile gains nothing.

With the bound stated over the plane, the wide writer saves no submits. The bound costs it
exactly the submits its wider N tile saves, on the N-heavy shapes where the tile would save
them. Over 315 natural shapes, M 4-256, K 64-2048 and N 32-2048, the chooser takes the narrow
writer for every one. [HW sweep, H96 MAX M9]

With the dense writer off (`ROCKET_RK3576_I32_DENSE=0`), the `librocketnpu` planner picks
the narrow or the wide writer per shape by submit count. The count is known before anything is
submitted. The function `r76_i32_submit_count()` counts K slices x N tiles x row
tasks for each writer, and the output cap splits the wide writer's tasks further. The cheaper
writer is selected, and a tie goes to the narrow one. `ROCKET_RK3576_I32_OC_MULT` forces either.

On that path the wide writer is not chosen for any shape in the sweep, and a shape outside it
can still reach it. [HW sweep, H96 MAX M9]

### The dense writer: every accumulator as an int32, and no poisoning

**Conditions:** H96 MAX M9, `rocket` 1.6.0, measured 2026-09-30.

Eight DPU words make the int8 direct program write every accumulator as a raw int32. Emitted
together over it, one task delivers all `M*N` accumulators exactly, in `M*N` words, with no bias
added. Seven are the `charsiu` project's wide output stage, read from vendor streams
(`src/job.c`), and its `0x40B8` is theirs too, fitted there at one row:

| register | int8 direct | dense int32 |
|---|---|---|
| `0x4010` | `0` | `0xa0000002`, `out` 5 and `proc` 2 |
| `0x4030` low half | `0x0710` | `0x0310` |
| `0x4038` | `0x00120080` | `0x00000053` |
| `0x4044` | `1` | `2` |
| `0x4050` | `0x80011111` | `0x00023333` |
| `0x40AC` / `0x40B0` / `0x40B4` | the requant | `0` / `1` / `0` |
| `0x40B8` | `2*S - ow*oh_task` | `3*S` |

`S` is the task's pixel count `ow*oh`, and `0x401C` stays at `S`. [HW sweep]

The map is the wide writer's stream with all four lane groups delivered. With `A` the pixel
count, `g = c/32`, `j = (c%32)/16`, `L = (c%16)/4` and `s = 2p + j`:

```
atom = 8A*g + 4A*(s/A) + A*L + s%A
word = 4*atom + c%4
```

It decodes as a bijection, every accumulator distinct, at `N` 32, 64, 96 and 128. The planes
run from 3×2 to 12×10, `M` 4-120, and `K` 32-4096 on random operands. One task writes a 38 KiB
output exactly at `K` 32, so the wide writer's 8 KiB bound does not hold for this writer there.

> [!CAUTION]
> The programmed `oc` must be a whole number of 32-channel groups, the int8 program's own rule.
> A 16-channel trailing group produces incorrect output past `K` 32. Software must pad `oc` to
> a multiple of 32.

> [!CAUTION]
> Doubling `0x401C` and `0x40B8`, instead of the values in the table, is exact at 32 channels
> and overlaps past them. The four quad planes then sit twice as far apart. One 32-channel group is placed whole, and the
> next group's start falls one plane in. The overlapping writes race, and the result differs
> from run to run. Software must program `0x401C` at `S` and `0x40B8` at `3*S`.

**Not characterized:** `0x40B8` for a row window.

The dense writer does not leave the next submit unwritten: see [rk3576.md](rk3576.md) §"The
poisoning is one hazard".

### The wide-output hazard after an int32-output job

**Conditions:** H96 MAX M9, measured 2026-07-27 and 2026-07-28, on a `rocket` from before
interface version 1.0.

An int32-output job through the narrow or the wide writer leaves the next submit unwritten
until the NPU power domain cycles. This is erratum RK3576-E2, the wide-output hazard. The
hazard, the idle that clears it and the workaround are in [rk3576.md](rk3576.md) §"The
poisoning is one hazard, and its cost is a system setting". The dense writer does not
trigger it (§"The dense writer: every accumulator as an int32, and no poisoning"). This
section gives the register conditions under which the int32 programs trigger it.

On the narrow and wide writers, the entry `rocket_matmul_int8_rk3576_i32()` sizes its idle
from the driver's autosuspend delay, at twice the delay plus 20 ms. It idles between its own
submits and once more on the way out, and retries any submit whose own region came back
untouched.

#### Register conditions

DPU `0x4010` (the DATA_FORMAT word, `out<<29 | in<<26 | proc`) triggers the hazard on the int32
path, and no single register triggers it in general. The int32-output program differs from the
int8 direct conv in only four registers: `0x4010`, `0x40AC`, `0x40B0` and `0x40B4`. Zeroing
`0x4010` alone stops the hazard with the program still writing, and zeroing any of the other
three does not. The sufficient condition is joint, not one register:

| program | `0x4010` forced to | triggers the hazard? |
|---|---|---|
| int8 direct | `0` (its own) | no |
| int8 direct | `out`=1, 2 or 3, any `proc` | no |
| int8 direct | `out`=4 or 5 | yes |
| int32 output | `0` | no |
| int32 output | any non-zero, including `out`=1..3 and `proc` alone | yes |
| fp16 first conv | each of its 19-register delta, one at a time | yes, all 19 |

The same `out` value is therefore clean on one program and triggers the hazard on another. A
leave-one-out over the whole fp16 first-conv delta clears nothing, because either of
`0x4038[4]` and `0x4050[17]` triggers it on the fp16 programs. The dense writer's whole output
stage sets both bits and does not trigger it ([rk3576.md](rk3576.md) §"The register
conditions").

The mechanism is undecoded. The next job programs `0x4010` back to all-int8 and still comes back empty. The state is
therefore latched in the datapath rather than held in a stale register. Changing the buffers or
their addresses does not clear it.

#### Driver completion poll

`patches/rk3576/npu/0012` closes the completion-visibility race. It retires the poll on the
DPU's own completion bit rather than on `PC_DONE`. The reproducer and the fix are under "The
per-submit floor is the driver's completion poll" above. The module parameter
`poll_interval_us` is live, so the poll period can be swept from sysfs with no rebuild.

The per-submit cost on the RK3576 is set by the driver's poll, not by the silicon. PC_DONE is
not routable to the GIC on the RK3576, so `patches/rk3576/npu/0006` polls it where the RK3588
takes an interrupt. Every RK3576 cost table is therefore a submit-count table.

#### Related hazards

The wide-output hazard also reaches the plain int8 matmul, whose first submit can then come back
untouched. The entry `rocket_matmul_int8_rk3576()` stamps its output and redoes any row task
whose region is still the stamp. [HW sweep, H96 MAX M9, measured 2026-07-27]

> [!CAUTION]
> A bare `memset` of an output buffer from the CPU before a submit leaves dirty cache lines.
> They race the DPU's DMA, and the writeback overwrites the result. The output then comes back
> all zeros, intermittently, on the first submit as readily as a later one. Software must bracket
> any CPU fill of an output BO with `PREP_BO` and `FINI_BO`.

A fill bracketed by `PREP_BO` and `FINI_BO` is written back before the submit and leaves nothing
dirty. Such a stamp makes "this was never written" a property of the output instead of an
inference from its value. A freshly allocated BO arrives zeroed, which cannot distinguish an
unwritten word from a legitimate zero. A guard band past the output needs 64-byte alignment, so
that no line it dirties is shared with an output byte. Both matmul entries use a bracketed
stamp.

An int32 sweep wedged the IOMMU once, with a stall timeout and a `rocket_job_irq_handler` warn
at `rocket_job.c:528`. A reboot cleared it. Whether the module reload that clears the IOMMU
wedge (§"The IOMMU wedge") clears it is not characterized. The measured extent sits well inside the BO that was
allocated, so the wedge is not a buffer-size problem, and its mechanism is not established.
[HW sweep, H96 MAX M9, measured 2026-07-27]

### Timing a matmul

> [!NOTE]
> `rocket_submit_matmul()` returns before the job does. Timing it without a fence on the output
> BO reports 0.0 ms.

> [!CAUTION]
> An unpaced job completes without writing at all, which reads as an arithmetic failure and not
> as a scheduling one. Software that measures this path must pace between runs as well as
> between tasks, and keep the pacing outside the measurement.

## The DPU LUT

The DPU's LUT evaluates a nonlinear activation per element, from two tables held in the LUT
RAM. On the RK3576 its registers are the bank `0x4100`-`0x4194`, and `EW_CFG` (`0x407C`)
selects it as the EW stage. A DPU-only program loads the tables, and the convolution that
follows reads them.

### Table load program

A nonlinear activation is a separate program, not a fused epilogue. The vendor compiler emits
two programs for each convolution plus activation [Manufactured capture]. The first is DPU + DPU_RDMA only, with no
CNA and no CORE. It drives a 1×1×16 dummy cube (`0x401C = 1`, `0x4030 = 0x000f0f00`,
`0x40B0 = 0x00010001`), and its work is loading the table, not computing. The second is an
ordinary convolution, configured to read the table.

The table is written through a two-register window that auto-increments. The
`LUT_ACCESS_CFG` register (`0x4100`) selects the table and the start offset. Every
subsequent write to `LUT_ACCESS_DATA` (`0x4104`) stores one entry and advances. The load
program writes `0x4108 = 0x00ff0000` first, then:

| `0x4100` | Entries | Table |
|---|---|---|
| `0x00020000` | 513 | LE |
| `0x00030000` | 513 | LO |

Each table holds 513 = 2^9 + 1 entries: 512 intervals with both endpoints, the layout a
linearly interpolating LUT needs. The two tables are the same LE/LO pair the NVDLA SDP
documents.

The vendor's load program is 1121 words:

- The 1×1×16 dummy cube.
- The `0x4100 = 0` / `0x4104 = 0` / `0x4108 = 0x00ff0000` preamble.
- The two 513-entry bursts.
- The four-word trailer.

The trailer's `PC_OPERATION_ENABLE` is `0x18`, DPU and DPU_RDMA, neither the convolution's
`0x1D` nor the pool's `0x60`.

> [!NOTE]
> The load program's dummy cube writes nothing at any address, so the load program cannot
> show that the table loaded. A convolution that reads the table shows it: its output follows
> the table.

> [!CAUTION]
> The table is held in the LUT RAM, and a separate submit can take a runtime-PM cycle between
> the load and the use. Software must put the table load and the program that reads it in one
> job.

### Table contents

The two tables are the lower and upper halves of one monotone curve, meeting at the value
the function takes at the split. The entries are Q15 of the function's output. For
`Sigmoid` the entries run `0x3b -> 0x4000` in the `0x00020000` table and `0x4000 -> 0x7fc4`
in the `0x00030000` one, and `0x4000/0x8000` is exactly 0.5. For `Tanh` they run
`-0x7f63 -> 0` and `0 -> 0x7f63`, symmetric about zero.

The table is uniform in the input, and each function's span in its own units is the
compiler's choice. The compiler puts the table where each function saturates
[Manufactured capture]:

| Function | Step | Half-span |
|---|---|---|
| sigmoid | 0.0122 | 6.25 |
| tanh | 0.00589 | 3.02 |

The compiler sets the span by scaling the value the datapath hands the
LUT, and leaves the LUT's own window fixed.

### Registers

The registers that enable the LUT, against a bare convolution from the same capture set
[Manufactured capture]:

| Register | Bare convolution | LUT program | Function |
|---|---|---|---|
| `0x407C` | `0x010041C1` | `0x01004140` | EW_CFG: `EW_LUT_BYPASS` (bit 7) and `EW_BYPASS` (bit 0) clear, so the LUT is the EW stage |
| `0x4108` | 0 | `0x02000006` | LUT_CFG |
| `0x410C` | 0 | `0x00050500` | LUT_INFO |
| `0x4110`-`0x411C` | 0 | `0xffffc000` | LE_START |
| `0x4140`-`0x414C` | 0 | `0x00004000` | LO_END |
| `0x4188`-`0x4194` | 0 | the tails | the two clamps |
| `0x5028` | 0 | `0x1A` | DPU_RDMA NRDMA_CFG |
| `0x1004`/`0x3004`/`0x4004`/`0x5004` | `0x0E` | `0x30` | every block's S_POINTER |
| `0x4060` | `0x903` | `0x20` | BN_CFG. Do not transcribe the vendor's value (Caution below) |

The `EW_CFG` field layout is the RK3588's and transfers unchanged (`EW_LUT_BYPASS` at bit 7,
`EW_BYPASS` at bit 0), and the two values above decode by it. The register offset does not
transfer: `0x4070` on the RK3588, `0x407C` on the RK3576.

The register pairs `0x4188`/`0x418C` and `0x4190`/`0x4194` are the output clamps: the value
used below the table and the value used above it. Each equals the table's first or last
entry. In the vendor's packing, the first register of each pair holds the value alone in its
high half. The second holds it doubled across both halves. For the low clamp, `0x4188` =
`lo << 16` and `0x418C` = `(lo << 16) | lo`, and the high clamp takes the same form at `0x4190`/`0x4194`. In four
captured functions, each clamp matches its own table's endpoint exactly
[Manufactured capture]:

| Function | Below the table | Above the table |
|---|---|---|
| sigmoid | `0x003b` | `0x7fc4` |
| tanh | `0x809d` | `0x7f63` |
| softplus | `0x0094` | `0x7fff` |
| swish | `0xff9d` | `0x7fff` |

Two register groups, `0x4150`/`0x4154` with `0x4160` and `0x4170`/`0x4174` with `0x4184`,
are the underflow and overflow linear-extrapolation slopes (a scale and a shift each). In
the four captured functions, the slopes are the pair of tails each function has
[Manufactured capture]:

- `tanh` saturates both ends and has zero in all four.
- `sigmoid` has a small lower slope and zero above.
- `swish` has zero below and about 1.0 above.
- `softplus` has a small lower slope and about 1.0 above.

In a LUT program, `0x40AC`/`0x40B0`/`0x40B4` are the LUT's output requant: the DPU's
ordinary output converter, which the piecewise activations and the bare conv write too. On a
fused nonlinear activation they convert Q15 to the op's own quantization exactly. They do not
select which interval an input falls in [Manufactured capture]. Sigmoid's output is
`[0, 1]` at zero point -128, so its gain is `255/32768 = 7.7820e-3`, and the registers hold
`0x7f81 >> 22 = 7.7815e-3`. Tanh's output is `[-1, 1]` at zero point 0, so its gain is
`127.5/32768 = 3.8910e-3`, against `0x7f81 >> 23 = 3.8907e-3`. The multiplier is the same
with one more shift, and `0x40AC` is -128 for sigmoid and 0 for tanh.

### Input map

The input map is `index = (value - LE_START) / 2^sel` [HW sweep, H96 MAX M9]. Three
registers set it:

| register | field | value the vendor writes | what it does |
|---|---|---|---|
| `0x4110`-`0x411C` | `LE_START`, four lanes | `0xffffc000` = -16384 | where the LE table starts |
| `0x4140`-`0x414C` | `LO_END`, four lanes | `0x00004000` = +16384 | where the LO table ends |
| `0x410C` | `LUT_INFO`, two index selects | `0x00050500` | the index step, `2^5 = 32` |

The `0x00020000` table (LE) covers `[LE_START, 0]` and the `0x00030000` table (LO)
covers `[0, LO_END)`, 512 intervals each. All three registers are live. The map is
bit-explained at `sel` 4, 5 and 6, and at an asymmetric `LE_START = -8192` with
`LO_END = +16384`. The maximum disagreement is one output count [HW sweep, H96 MAX M9].

The vendor never moves these registers. It places the value inside the fixed window with the
BS stage's own per-channel `C` instead. At fp16 the scale is in the coefficient buffer rather
than in a register. Sigmoid and tanh therefore compile to identical fp16 programs, with tables
spanning 6.25 and 3.02.

The hardware interpolates linearly between entries, at the full resolution of the step
[HW sweep, H96 MAX M9].

The domain is half-open at the top. An input of `value == LO_END` takes the overflow
clamp, not the last table entry. An input of `value == LE_START` is in the domain and
reads `LE[0]` [HW sweep, H96 MAX M9].

> [!CAUTION]
> Every vendor activation program writes `BN_CFG` (`0x4060`) = `0x20`, which makes the BN
> stage active. Transcribed, that value makes the LUT return the value at the table boundary for
> every input from -2^21 to +2^21. The output is constant and tracks the table as the table
> changes. It looks like a working LUT whose input map is nowhere in the registers.
> Software must program `BN_CFG` to `0x903`, the bare-convolution value, in a LUT program.

With `0x20`, the BN stage multiplies by an operand register that no vendor program writes,
and the register file is not cleared between jobs. At `0x903` the BN stage is bypassed, the
LUT reads the BS output directly, and the input map takes effect. A LUT program does not
need the BN operand. The BS stage's per-output-channel `C` is a per-channel scale, where a BN
multiply would be one global one.

**Not characterized:** where the RK3576 keeps `BN_MUL_OPERAND`.

> [!CAUTION]
> The DPU register file is not cleared between jobs. A program that leaves the LUT bank
> unwritten can run with the previous job's LUT window. Software must write the whole bank,
> `0x4100`-`0x4194`, in every program, with zeros where no LUT is used.

### Software support

`gen_conv2d_int8_rk3576()` emits the enabling registers above from `conv_params_t.lut`
(`lut_rk3576_t`: `le_start`, `lo_end`, `sel`, and the two clamps). The generator writes the
whole bank either way. When the field is NULL, it writes zeros. It packs the two clamps as the
vendor packs them, and leaves `0x4184` at zero. `gen_lut_load_rk3576()` emits the table-load
program, 1121 words as the vendor's is.

## The PPU pooling program

On the RK3576, pooling is its own NPU program: an independent 31-write program, with 23 PPU
writes and 8 PPU_RDMA writes. A `MaxPool` or `AveragePool` behind a convolution emits that
program [Manufactured capture]. A bare pool emits exactly the same program, with no
convolution beside it. It works in the same NC1HWC2 cube the convolution path already packs,
so it needs no new host packing.

The vendor compiler lowers `GlobalAveragePool` to convolutions, not to pooling. Every
program in that capture is CNA + CORE + DPU, the same lowering the RK3588 uses for a
reduce. The compiler lowers `GlobalMaxPool` to pooling, in two cascaded PPU passes (k7 s7,
then k3 s3 over what is left).

### PPU registers

The 23 PPU writes [Manufactured capture]:

| Offset | Function |
|---|---|
| `0x6004` | `0x0e`, constant (the same enable word CNA/CORE/DPU take at their `+4`) |
| `0x600C` | input width consumed, minus 1 |
| `0x6010` | input height consumed, minus 1 |
| `0x6014` | input channels minus 1, rounded up to 16 |
| `0x6018` | output width minus 1 |
| `0x601C` | output height minus 1 |
| `0x6020` | output channels minus 1, same rounding |
| `0x6024` | mode: `0x11` max, `0x10` average, `0x18` average with the pad excluded from the divisor |
| `0x6034` | `(sy-1)<<20 \| (sx-1)<<16 \| (kh-1)<<8 \| (kw-1)` |
| `0x6038` / `0x603C` | `1/kw`, `1/kh` in Q16, `0x8000` at k2, `0x5555` at k3, zero for max |
| `0x6040` | four pad nibbles: left [3:0], top [7:4], trailing pair [15:8] |
| `0x6044`-`0x6050` | pad values: `-128` (as `0x0007ff80`, sign-extended in a 19-bit field) for max, and the input zero point times 1, 2, 3, 4 for average |
| `0x607C`, `0x6084` | `round4(ow*oh) * 16`, the destination surface stride per 16-channel group, the same round-to-four the convolution's `0x401C` takes |
| `0x6054`, `0x6058`, `0x605C`, `0x6070`, `0x60DC` | zero in every capture. `0x6070` is the destination base address (below) |

The input extent is what the windows consume, not the plane. The registers
`0x600C`/`0x6010` hold `(ow-1)*sx + kw` and `(oh-1)*sy + kh`, clamped to the plane. A
15×18 plane pooled k2 s2 therefore programs 18×14 and not 18×15: the last row, which no
window reaches, is not described.

The consumed extent excludes the leading pad, so it is
`min(plane, (ow-1)*stride + k - pad_start)` and not the windows' raw span. A VALID k3 s2
over 16 programs 15. A SAME k3 s2 over the same 16 programs 16, where its windows span 17.

The leading pads sit in the low nibbles of `0x6040`: left [3:0], top [7:4] [HW sweep, H96 MAX
M9]. A symmetric pad cannot tell this map from its transpose. Every capture's pool declares a
symmetric pad. The SAME k3 s2 capture over 16 declares one on every edge, and its last window
consumes no trailing pad. It programs `0x0011`, the leading pads alone, which agrees with this
map [Manufactured capture].

**Not characterized:** which of the high pair is right and which is bottom.

The destination base address is `0x6070` [HW sweep, H96 MAX M9]. No capture can show it.
Like every other base on the RK3576, the vendor runtime patches it at load time, and the
stored program holds zero.

### PPU_RDMA registers

The eight PPU_RDMA writes are the source side [Manufactured capture]:

| Offset | Function |
|---|---|
| `0x7004` | `0x0e`, the same enable word |
| `0x700C` / `0x7010` | input width / height consumed, minus 1, the PPU's own `0x600C`/`0x6010` again |
| `0x7014` | input channels minus 1, rounded up to 16 |
| `0x701C` | source base address |
| `0x7024` | DDR line stride in bytes, `iw*16`, the full plane, not the consumed extent |
| `0x7028` | channel-group surface stride, `round4(iw*ih)*16` |
| `0x7030` | `0x40` in every capture |

The strides are the only fields that hold the full plane. The line stride `0x7024` is what
separates them from the consumed extent: a 19-wide plane pooled k2 s2 consumes 18 and
strides 19.

PPU_RDMA honors its source surface stride as programmed, including a value that is not a
multiple of four [HW sweep, H96 MAX M9, 2026-07-31]. Every vendor pooling program stores
`round4(iw*ih) * 16` in `R76_PPUR_SRC_SURF` (`0x7028`). A direct convolution's output surface
stride is `ow*oh` exactly, 49 against 52 at a 7×7 plane. A pool can therefore read a
convolution's output as its feature cube without a copy. The PPU takes the unpadded stride
bit-exactly at 49, 25 and 9 elements per channel group. The result holds over max and
average, at zero points either side of zero.

> [!CAUTION]
> A source surface stride that does not match the input cube's layout raises no fault. It can
> produce output of the correct size with incorrect values. Software must program the stride
> of the cube the PPU reads.

### Operation enable

`PC_OPERATION_ENABLE` is a per-block bitmap. A convolution enables its blocks with `0x1D`.
The vendor's pooling program enables PPU and PPU_RDMA with `0x60`, and the two bit sets are
disjoint.

> [!CAUTION]
> A pool program with the convolution's trailer is a fully configured PPU that is never
> started. The job completes in the usual per-submit time, faults nothing and writes nothing.
> Its result cannot be told apart from a wrong destination address. Software must
> transcribe the whole program, trailer included.

The bitmap does not partition a chained stream. A `conv -> pool -> conv` kick gives the same
outputs as three submits [HW sweep, H96 MAX M9, 2026-08-02]. The planes measured are 16 to
110, at 32 and 64 channels.

### Average rounding

The average rounds half to even, the same rule as the DPU's `OUT_CVT`. Against
round-half-away-from-zero, a k2 average disagrees on one output in eight. The divisor is the
window (`0x6038`/`0x603C` are `1/kw` and `1/kh`), not the tap count. Mode `0x18` excludes
the pad from the divisor.

With those fields, the pooling program is bit-exact against a CPU model over 11 shapes
[HW sweep, manufactured capture, H96 MAX M9, 2026-07-29]. The shapes cover:

- Max and average
- Kernel 2/3/5 and stride 1/2/3
- Non-square and odd planes
- 8/32/64 channels
- Padded and unpadded input

Whether the average is exactly rounded is a function of the window size. The reciprocals are
truncated, `0x10000/kw` and `0x10000/kh`, so the computed quotient is low by
`|sum| * (1/n - rw*rh)`. The closest a quotient of an integer by `n` comes to a half is
`1/(2n)`. The error is far under that boundary at k2/k3/k5, and is not guaranteed to be
under it at larger windows.

A global 7×7 average over 1024 channels, a MobileNet's pooling layer, is exact even so. It
also matches TFLite's `AveragePool` exactly there [HW sweep, H96 MAX M9]. An odd window has
no tie, so half-to-even and TFLite's half-up cannot differ. The int8 rebase is exact because
`128*49` is an integer multiple of the divisor.

### Pool completion

A pool enables PPU and PPU_RDMA and no DPU stage at all, so it raises no DPU completion. The
`DPU_0`/`DPU_1` bits that `patches/rk3576/npu/0012` retires a job on can never set for it.
On a module before 1.6.0, the job falls through to the `dpu_grace_us` deadline. It waits that
out in full on every submit [HW sweep, H96 MAX M9, 2026-07-31]. The wait tracks the parameter
count for count: 641 us at 500, 389 at 250 and 213 at 100. On `rocket` 1.6.0, where
`dpu_grace_us` 0 waits for the writing block, a pool submitted without the PPU flag below waits
the 125 ms backstop instead ([rk3576.md](rk3576.md) §"Pool completion").

The PPU's own bits are two positions over in the same register: `PPU_0` at `0x400` and
`PPU_1` at `0x800`, against the DPU's `0x100`/`0x200`. The patch `patches/rk3576/npu/0021`
adds `DRM_ROCKET_JOB_PPU_DONE` to select them per job, and the wait is then 71 us
[HW sweep, H96 MAX M9, 2026-07-31]. The selection is per job rather than per driver. A
job's class is invisible at `PC_DONE` time, and known to the userspace that wrote
`PC_OPERATION_ENABLE`.

Software must set `DRM_ROCKET_JOB_PPU_DONE` only on an interface version >= 1.3. The submit
ioctl rejects a flag word it does not recognize, so an older kernel fails the submit rather
than ignoring the bit.

> [!CAUTION]
> Below interface 1.4, in a stream that mixes DPU and PPU programs, an interior program's PPU
> bit can retire the job early. A later DPU write can then still be draining.
> Software must not set `DRM_ROCKET_JOB_PPU_DONE` on such a stream there.

From interface 1.4 the driver reads the completion bits only once `PC_TASK_STATUS` says the
whole kick has retired, which closes that window [source-confirmed: patch
`patches/rk3576/npu/0022`].

**Not characterized:** a mixed stream under `DRM_ROCKET_JOB_PPU_DONE` on interface 1.4 or later.

### Software support

`rocket_pool_int8_rk3576()` is the library entry over the pooling program. It takes
row-major `[C][IH][IW]` in and out, and handles the cube, the sentinel, the submit and the
de-tile, the same shape the convolution entries have. It is a separate entry rather than a
dispatch from the RK3588's `rocket_pool_int8()`. The RK3588 entry truncates the average
toward zero and this one rounds half to even, and two roundings are two functions. It takes
an input zero point, which the RK3588 entry does not, because the average path pads with it.

The function `rocket_pool_int8_rk3576_exact()` reports whether a descriptor's average is
exactly rounded, without running it. The field `pool_params_rk3576_t.src_surf_elems` holds
the source surface stride, and 0 keeps the vendor's `round4`.

## The unprivileged double free, and the crashes it was mistaken for

**Conditions:** mainline `rocket` at v7.1, unchanged by the RK3576 series. The RK3588 reaches
the same path. Measured on H96 MAX M9 and Turing RK1, 2026-07-29. In the RK3576 series the fix,
`0020`, precedes `0021`, the patch that declares interface version 1.3.

`rocket_copy_tasks()` frees the submit job's task array twice. It allocates
`rjob->tasks`, and on any failure inside its copy loop takes a `fail:` path that
`kvfree()`s it and returns, without clearing the pointer. Its only caller,
`rocket_ioctl_submit_job()`, then unwinds through `rocket_job_put()` ->
`rocket_job_cleanup()`, which frees `job->tasks` unconditionally. The same allocation
goes back to the allocator twice. [source-confirmed:
`drivers/accel/rocket/rocket_job.c` at v7.1]

Two failures reach it, and both are ordinary userspace input:

| Failure | Trigger |
|---|---|
| `copy_struct_from_user()` fails | a bad task pointer, or a trailing field this kernel does not know (`-E2BIG`) |
| `task.regcmd_count == 0` | a task that asks for no registers |

The second is the one that matters in practice. A generator that refuses leaves its output
parameters untouched. A caller that submits anyway hands the kernel a task with
`regcmd_count` still at its zeroed value.

`/dev/accel/accel0` is group `render`, so this is an unprivileged double free of a
kmalloc-16 object (a `struct rocket_task` is 16 bytes), one of the hottest caches in the
kernel. The freelist then hands one object to two owners.

> [!CAUTION]
> On a `rocket` without the fix, any submit the kernel rejects for its task array corrupts
> the kmalloc-16 freelist. The damage surfaces far from the double free, as two faults that
> look like defects in their own subsystems (below). Software must not submit a task the
> kernel rejects. Check a generator's result before building a submit from it.

The two faults are this corruption reaching other code:

```
dma_direct_unmap_sg <- dma_unmap_sg_attrs <- drm_gem_shmem_release
  <- drm_gem_shmem_free <- rocket_gem_bo_free        (GEM_CLOSE, file release,
                                                      and the drm_sched free worker)

__pi_memcpy_generic <- swiotlb_bounce <- swiotlb_tbl_map_single <- swiotlb_map
  <- dma_direct_map_sg <- dma_map_sgtable
  <- drm_gem_shmem_get_pages_sgt <- rocket_ioctl_create_bo
```

The first walks a scatter-gather list whose entries are garbage. The second bounces
through swiotlb only because a corrupt `sg_phys()` exceeds the DMA mask, and then
`phys_to_virt()`s a wild address. A corrupt sg table on the free side and a corrupt one on
the map side are the same event seen twice. The free-side fault looks like an sgt torn down
while still published. Two live BOs naming one `sg_table` identify it as two owners of one
slab object, not a lifetime race.

The crash surfaces roughly once per few minutes of heavy BO churn, whichever workload is
running, and the churn does not cause it. One rejected submit plants the corruption, and
whichever workload runs next collects it.

`slub_debug=FZPU slab_nomerge` on the kernel command line detects the double free in one
run, and the result is deterministic [HW sweep, H96 MAX M9, 2026-07-29]. The allocator
reports "BUG kmalloc-16: Object already free". The report names `rocket_ioctl_submit()` as
the allocation site and `rocket_job_cleanup()` as the second free.

The fix is `patches/rk3576/npu/0020`, which gives the array a single owner and deletes the
`fail:` free. The `fail:` free can go because `rocket_job_cleanup()` already runs on every
path out of the ioctl. [HW sweep, H96 MAX M9, 2026-07-29]

The RK3588 reaches the defect identically [HW sweep, Turing RK1, 2026-07-29]. On a Turing
RK1 at 7.1.1 the same detector reports it at the same allocation and second-free sites, and
`087` removes it. Patch `087` is `0020`'s one-line deletion against the
RK3588 series.

> [!CAUTION]
> A rejected submit returns its errno normally while it corrupts the slab, so a test of the
> rejection paths exits 0. An exit status cannot see the corruption. Software that tests
> rejection paths must read the kernel log, and run under `slub_debug` at least once.

### The scheduler-array leak

A second, smaller mainline defect in the same source is a leak rather than a crash. The
function `rocket_job_open()` allocates the `drm_gpu_scheduler` array, and
`rocket_job_close()` frees `entity->sched_list`. But `drm_sched_entity_init()` stores that
pointer only for `num_sched_list > 1`, and `drm_sched_entity_select_rq()` clears it again
once the entity has chosen a runqueue. On a single-core device, which is the supported
RK3576 configuration, the close is `kfree(NULL)`. The array then leaks on every
open/close, unprivileged and unbounded. [source-confirmed, HW sweep, H96 MAX M9]

The patch `patches/rk3576/npu/0019` keeps the driver's own pointer and frees it after
`drm_sched_entity_destroy()` rather than before. It also checks the allocation, which
mainline leaves unchecked, and frees it on the init-failure path too.
[source-confirmed, HW sweep, H96 MAX M9]

## The DPU's elementwise stage

The DPU's elementwise stage computes on one operand and requantizes the result. An
elementwise op on the RK3576 is a program of its own: DPU + DPU_RDMA only, no CNA and no
CORE, 89 writes, `PC_OPERATION_ENABLE` `0x18`. It has the same shape as the LUT table
load, but it computes a result. The vendor compiler does not fuse it into the producing
convolution's epilogue, so a vendor residual block costs one more program for its add.
`librocketnpu` lowers the add onto the convolution datapath instead ("The residual add on the
convolution datapath", below).

What it computes [HW sweep, H96 MAX M9]:

```
out = sat8( ( ((ew + EW_CVT_OFFSET) * EW_CVT_SCALE >> EW_CVT_SHIFT)
                                    * OUT_CVT_SCALE >> OUT_CVT_SHIFT ) + OUT_CVT_OFFSET )
```

The hardware adds `OUT_CVT_OFFSET` after the final shift, in output counts
[HW sweep, H96 MAX M9, 2026-09-26]. The vendor's own Add program has an offset of -1. With
the offset added before the scale, every element of it is off by one.

The one-operand model is bit-exact over channel counts 16-320 and planes 1×1 to 28×28
[HW sweep, H96 MAX M9, 2026-07-31]. The operand is the NC1HWC2 cube the convolution path
already packs. The final shift rounds half to even, the same rule the direct path's OUT_CVT
uses.

The registers that differ from a convolution's DPU half:

| Register | Add | Convolution | Function |
|---|---|---|---|
| DPU `0x400C` | `0x00000005` | `0x40000004` | FEATURE_MODE |
| DPU `0x4030` | `(c-1)<<16 \| 0x0F00` | `… \| 0x0710` | WDMA_SIZE0 |
| DPU `0x4038` | `0x00100012` | `0x00120080` | NOTCH_CFG |
| DPU `0x4044` / `0x4050` | 0 / 0 | `0x1` / `0x80011111` | BS stage, off |
| DPU `0x407C` | `0x8002C0C0` | `0x010041C1` | EW_CFG: bit 0 clear, so the stage is on |
| DPU `0x40C0` | `0x04440000` | `0x04440100` | SURFACE_ADD in the RK3588's map |
| DPU_RDMA `0x501C` | `0x1A` | `0x710` | the operand-source register |
| DPU_RDMA `0x5034` | `0x40000044` | `0x41` | ERDMA_CFG |
| DPU_RDMA `0x5038` | the operand base | 0 | EW_BASE_ADDR, the one base that is read |
| DPU_RDMA `0x5040` | `w*h` | 0 | the operand's channel-group stride |
| DPU_RDMA `0x5044` | `0x9` | `0x40000010` | FEATURE_MODE |

DPU `0x400C` bit 0 is the NVDLA feature-mode flying bit, and the add sets it. DPU_RDMA
`0x5044` bit 4 is `MRDMA_DISABLE` in the same lineage's map, and the add clears it. The
program is therefore configured to read a main DMA feed [HW sweep, H96 MAX M9]. That feed
contributes exactly zero at every output gain from 2^14 down to 2^-31.

`MUL` is the same program, differing in four registers:

| Register | `MUL` |
|---|---|
| `0x407C` | `0x810F4094` |
| `0x4044` | `0x2` |
| `0x4050` | `0x00020000` |
| `0x501C` | `0x2` |

**Not characterized:** the arithmetic of `MUL`.

### Operands

The elementwise stage takes exactly one operand through this register set
[HW sweep, H96 MAX M9, 2026-09-26]. The search covered these:

- Every register the program leaves at zero, and every register it never writes
- The main feed and the mode fields
- Destination accumulation, and the addresses the program reads

The search also found these register facts:

- The program reads exactly the cube its operand base addresses, and nothing outside it
  affects the output. The two operands are therefore not one allocation at a fixed offset,
  which is how the RK3588's K-accumulation feeds its pair.
- DPU `0x40C0`, `SURFACE_ADD` in the RK3588's map, is not an accumulate mode on the RK3576.
  The captured word and its 28 single-bit variants always overwrite the destination. Two bits,
  `0x00010000` and `0x00020000`, saturate the output.
- The vendor's `.rknn` relocation table sends the first operand to DPU_RDMA `0x5018`, the
  second to `0x5038` and the output to DPU `0x4018`. Replayed with both bases patched, the
  vendor's program writes the one-operand result and ignores the operand at `0x5018`.

> [!CAUTION]
> DPU_RDMA `0x5068`, the RK3588's `RDMA_WEIGHT`, holds the four read DMAs' arbiter weights.
> At any value that is not a weight, the job writes nothing and retires at the driver's
> 125 ms backstop. The value then persists, and later jobs stall until the NPU's power domains
> have been off for minutes. Software must write `0x5068` only with arbiter weights.

The vendor's program has one operand converter. A `Sub` compiles to this same program with
the operand converter's scale negated, and a `Sub` with its operands swapped negates it too.
The subtrahend is therefore always the elementwise cube. The other operand is on a feed
whose weight is fixed at +1, since no register holds a second scale [Manufactured capture].

Both operands also share one quantization scale. Two graph inputs calibrated 64x apart
still compile to one converter. Its OUT_CVT gain comes out as the ratio of a single shared
input scale to the output's.

**Not characterized:** anything else the vendor runtime patches or submits for an Add.

### The residual add on the convolution datapath

The elementwise stage has one operand, so a residual add runs on the convolution datapath
instead. `librocketnpu` lowers it as a direct 1×1 convolution over the two operands,
concatenated along the channel axis. The kernel is two diagonal blocks, `W[o][o] = w1` and
`W[o][C+o] = w2`, zero elsewhere:

```
out[o] = requant( w1 * (a[o] - a_zp) + w2 * (b[o] - b_zp) )
```

The register program is the direct int8 convolution's, at `ic = 2C` and `oc = C`. The
datapath has one input zero point, so the bias holds the per-output-channel constant
`w2 * (a_zp - b_zp)`. The weight-slice rule, `ic*kh*kw <= 4608`, caps a standalone add at
`C = 2304`.

A block's last convolution can also absorb its own skip. It takes `C` more input channels and
an identity block at the center tap of its kernel. The datapath computes that form bit-exactly
[HW sweep, H96 MAX M9, 2026-07-31], and the vendor compiler folds `Add(Conv(x), x)` into
exactly this shape. In every real residual block measured, the quantization needs an
identity weight that does not fit int8, so a network uses the standalone add.

[rk3576.md](rk3576.md) §"The residual add on the convolution datapath" gives the lowering's
accuracy and its advantages over the vendor's program. It also gives the fusion's weight limit
and a standalone add's cost.

### Traps in the elementwise program

> [!CAUTION]
> Every base left at the capture's stored zero reads IOVA 0, which is a real buffer. Per-fd
> IOVA starts at zero under `rocket`, so the first BO the fd allocates is what those bases
> read. With the operands allocated first, the output comes back a constant and the
> arithmetic looks broken. A zero base is not a disabled one. Software must allocate a guard
> BO first, so that no operand or output buffer is at IOVA 0.

> [!CAUTION]
> Read as a clamp pair and written as `0x00407F80`, DPU `0x40D0` leaves exactly half of every
> 16-channel group unwritten, with no error. It is a write-coverage failure, not a wrong
> value, and it looks like a channel-budget limit of the writer. Software must write DPU
> `0x40D0` as the captured `0x0040FFFF` verbatim.

The clamps are `OUT_CLAMP_MIN`/`MAX` (`0x40A4`/`0x40A8`), and the no-clamp pair is
`INT32_MIN`/`INT32_MAX`.

> [!CAUTION]
> `OUT_CVT_SCALE` is a signed 16-bit field on both SoCs: `32768` reads back as `-32768` and
> flips the output's sign. Software must keep the scale at or below the usable maximum, 32767
> ([../encodings/out-cvt-converter.md](../encodings/out-cvt-converter.md)).

### Vendor lowering of an elementwise add

The vendor compiler emits the elementwise program only for an operand it cannot fold into a
convolution [Manufactured capture]. Besides `Add(Conv(x), x)`, the graph
`Add(Conv_a(x), Conv_b(x))` folds too, and so does a per-channel broadcast operand. A graph
`Add(Conv(x), k)` with `k` of shape `[1, C, 1, 1]` compiles to a plain convolution with `k`
in its bias. Each of these three emits the elementwise program, identically:

- A second graph input.
- A constant tensor of the same shape.
- A real MobileNetV2 bottleneck whose skip crosses three convolutions.

## Uncharacterized registers and functions

These registers and functions are not characterized on the RK3576:

- On-chip accumulation across the fp16 `ic` split is not characterized. The fp16 section's "An
  arbitrary `ic` via a 16-channel split" states what it would remove.
- The vendor's epilogue and float BS arrangements are not decoded. Register `0x501C` is the BS
  operand-source register, and with `0x4044` it selects a BS operand arrangement that
  `librocketnpu`'s coefficient group is not packed for. The library's arrangement is bit-exact
  and the vendor's is not, so the regcmd encoder programs the library's. What the vendor's two
  arrangements read, and what they give its epilogue, is not known. Under the epilogue
  arrangement, one output in four survives. "The vendor's float BS arrangement" and "The BS
  operand source" have the measurements.
- Register `0x1060` is probably `CVT_OFFSET3`. Registers `0x1054`-`0x105C` are the first three
  CVT offsets and there are four CVT scales, so the fourth offset almost certainly sits here.
  Every capture leaves it zero, including the ARGB ones, which use only three channels, and
  nothing confirms it.
- Register `0x1064` is the offset the RK3588 calls `CNA_FC_CON1`. A live vendor dump, an
  `rknpu` kernel capture taken at submit, has `0x777` here. Every stored program has 0, so
  the vendor runtime patches this register at load time, the way it patches the addresses.
  Forcing it to `0x777` on `librocketnpu`'s path changes nothing (nor does forcing `0x1060` or
  `0x1074`).
  The RK3588 runs its own matmul with this register at 0, so it is not required for the
  direct-conv datapath. Its function in a fully-connected mode is not characterized.
