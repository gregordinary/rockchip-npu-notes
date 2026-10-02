# RK3576 int8 conv register encoding, as an emitter

The RK3576 NPU runs the same IP family as the RK3588 through the same mainline
`rocket` driver. Its CNA/CORE/DPU blocks use a different geometry-register encoding at
the same block bases. This sheet writes that encoding out as something you can emit,
and records what running it on real silicon settles and what it does not. It is the
register layer. The SoC-level parameter sheet (identity, integration, clocks, power) is
[rk3576.md](rk3576.md).

Provenance: RKNN-Toolkit2 register programs for known conv geometries, from two
capture sets under
[rocket-userspace `tests/data/rk3576-vendor-capture/`](https://github.com/gregordinary/rocket-userspace/blob/main/tests/data/rk3576-vendor-capture/).

Found captures, from Ga Hing Woo's bring-up repo, real convolutions, including a
MobileNet-shaped stem:

| Capture | Geometry | What only it can show |
|---|---|---|
| `conv2d_rk3576.rknn` | ic 16, oc 128, k5 s2, 80×80 -> 40×40 | the normal `in_ch>4` path at k != 3, and a single task over the whole plane |
| `dw_rk3576.rknn` | depthwise c32, k3 s1, 112×112 | the depthwise words, and a task windowed to 91 input rows |
| `conv64_rk3576.rknn` / `conv0_rk3576.rknn` | ic 3, oc 32, k3 s2 | the first-conv ARGB sub-encoding, at two image sizes |
| `iso_bias` / `iso_scale` / `iso_sum` | the `conv2d` geometry plus one appended op | what a DPU epilogue stage moves |

Manufactured captures, in `dw/` and `dw/named/`. **A vendor capture can be built to
order**, and that is the single most useful thing to know about this part. An ONNX
compiled for `rk3576` emits the register program for whatever geometry it asks for. A
diff then answers a question that a bit sweep answers slowly and ambiguously. Setting
`group=C` gives the depthwise path at any channel count, kernel, stride and plane, and
`do_quantization=False` on a float ONNX gives the float datapath. The scripts
`dw/mkdw.py` and `dw/named/mknamed.py` rebuild both sets and carry the dependency pins.

The found depthwise captures confound almost everything. In them, every C=32 program
is stride 1 and every C=64 one is stride 2. Every one is k=3, every one is a multiple
of 32 channels, and every one is a square plane. Manufactured captures separate all of
it, at 22 channel counts from 8 to 256, both strides, kernels 1/3/5/7 and rectangular
planes. The "Depthwise" section below transcribes four register formulas from them.

The emitter built from them is
[npu_regcmd_rk3576.c](https://github.com/gregordinary/rocket-userspace/blob/main/src/npu_regcmd_rk3576.c).
The gate that diffs it against the captures register-for-register is
[regcmd_rk3576_gate.c](https://github.com/gregordinary/rocket-userspace/blob/main/tests/regcmd_rk3576_gate.c).
The emitter reproduces every checked register of every captured program, 110 conv
programs, 88 of them depthwise, with no open field left.

Register fidelity is not correctness, and depthwise is where the two part company. An
emitter matching the vendor register for register across 88 depthwise programs still
computed nothing. What it had wrong were the buffers the registers point at. A capture
carries a register program and says nothing about the memory it addresses. Both of
that path's buffer layouts are read off the part (see "Depthwise" below), and that
lesson is general rather than a depthwise one.

The correctness gate is
[rk3576_conv_gate.c](https://github.com/gregordinary/rocket-userspace/blob/main/tests/rk3576_conv_gate.c),
a shape table swept in one process. It compares each entry against a CPU model
bit-exactly over the whole surface. The table covers the envelope, the row window, the
channel-group jump and the weight-slice boundary, with the depthwise envelope beside
them. The gate runs gap-free on a `rocket` carrying the per-SoC `PC_TASK_CON` width. It
reports every cold-start-wall retry it takes, so a run that prints none is a run the
wall did not touch. All 102 shapes pass in 4.6 s.

## Corrections to the published map

The published map is wrong in four places. Each correction is load-bearing: an emitter
that takes the published reading silently mis-programs the part.

**The output-height field is not halved.** It is the number of output rows *this
task* writes, minus one. The published halving comes from a capture whose task covers
half its plane. The same conv captured as a single task writes the full `oh-1`.

In `conv2d`, CORE `0x301c` hi is `0x27` = 39 for a 40-row output. The ARGB `conv0`
full-plane task writes `0x6f` = 111 for 112 rows, and only its half-height slices
write 55. Emit `(oh_task-1)<<16 | (ow-1)`, and the same `oh_task-1` at DPU `0x4024` /
`0x4034` and RDMA `0x5010`.

**The feature-data address is `0x1088` on both datapaths**, not `0x1070` on the
ARGB path. Register `0x1070` reads zero in every capture, including the one program
that provably needs a non-zero feature address. The `conv0` capture's fourth task reads
input rows 111.. of a 224-wide 3-channel image, and `0x1088` carries exactly
`111 * 224 * 3 = 0x12360`. Register `0x1070` is zero in that same program. The only
address register that coincides with the RK3588 is the weight base (`0x1110`).

**The pad word `0x1080` is closed form**, not open. It carries four byte fields:

```
0x1080 = pad_right<<24 | pad_bottom<<16 | pad_left<<8 | pad_top
```

where the leading pads are the configured padding and the trailing two are the
pad *actually consumed* by this task's last window,
`max(0, (out-1)*stride + k - (pad_lead + in_extent))`. The formula reproduces all four
captures exactly, including the two that make it unambiguous. One is `conv2d`'s
asymmetric SAME padding (l1 r2 t1 b2 -> `0x02020101`). The other is the depthwise task
whose row window stops short of the image bottom. Its bottom pad is 0 while its right
pad is 1 (`0x01000101`). The published rule, a lookup keyed on stride and depthwise,
predicts `0x00000101` for `conv2d` and is wrong.

So there is no configured trailing pad, and that is how an asymmetric pad is
expressed. The trailing fields are a function of the output extent and the leading
pad. A caller who wants TFLite's SAME at an even plane and stride two
(`pad_before = 0`, `pad_after = 1`) asks for a leading pad of zero. It also asks for
an output extent one larger than the symmetric formula gives, and nothing has to be
materialized in the feature buffer. The fields `rocket_conv2d_desc.oh/.ow` name that
extent, with zero meaning the symmetric derivation.

The bound is the kernel: a trailing pad of `k` or more is an output row whose whole
window is pad. Six shapes of the int8 correctness envelope carry a zero leading pad
against a consumed trailing one, direct and depthwise, and all are bit-exact. The float
path has never been run through that geometry and refuses it. [HW sweep, H96 MAX M9]

**The CBUF entries word is `ceil(iw*ic/64)`, not `max(iw/4, ic/2)`.** It is the count
of 64-byte CBUF granules in one feature row. The two forms agree on every capture the
published map was fitted to, which is why the map carries the second one. They part
company as soon as `ic/2` exceeds `iw/4`, and the depthwise capture is exactly that
case.

At `iw`=112 and `ic`=32, the granule count gives 56 and the max form gives 28. The
capture holds 56 at both `0x103c` hi and `0x1044` lo. The distinction is invisible at
the shapes the map was derived from and load-bearing for any narrow-and-deep tile (a
1×1 pointwise over a small plane sits squarely in the diverging region).

## Registers the map left unnamed, and their RK3588 equivalents

Four of the offsets recorded as "RK3576-only, no RK3588 counterpart" are the same
registers the RK3588 has, moved:

| RK3576 | Function | RK3588 |
|---|---|---|
| `0x1048` | CVT control word, `data_sign<<3 \| cvt_type<<1 \| cvt_bypass`, identical packing | `CNA_CVT_CON0` `0x104C` |
| `0x104C` / `0x1050` | the four CVT scales, two per register (`scale1<<16 \| scale0`, `scale3<<16 \| scale2`) | `CNA_CVT_CON1..4`, one each |
| `0x1054`-`0x105C` | the CVT offsets (non-zero only on the ARGB path, where they carry the per-channel -128) | folded into the same CON1..4 |
| `0x108C` | burst lengths, `weight_burst<<16 \| data_burst`, `0x000F000F` | `CNA_DMA_CON0` `0x1078` |
| `0x103C` hi, `0x1044` lo | CBUF data entries, `ceil(iw*ic/64)`, written twice | `CNA_CBUF_CON1` `0x1044` |
| `0x1084` | the CNA border pad constant | `CNA_PAD_CON1` `0x1184` |
| `0x5024` | base of the DPU shift word, a per-task operand, not a spare (see below) | the register fields `BS_MUL_CFG.BS_MUL_SHIFT_VALUE` `0x4048` and `DATA_FORMAT.BS_MUL_SHIFT_VALUE_NEG` `0x4010` |

The identification is by value, not by analogy. The int8 conv's CVT word is `0x0b` in
both encodings, and the burst word is `0x000F000F` in both, for the same reason.

The kernel word has a closed form. Register `0x1024` hi is `((kw-1)<<8) | (kh-1)`, the
width in bits [31:24] and the height in [23:16]: `0x0202` at k=3, `0x0404` at k=5 and
`0x0000` at k=1. The `conv2d` capture is the only one that shows a second k. The published
map carries it as a two-entry lookup, and it is one expression.

Every capture is square, so the captures cannot tell the two halves apart. Inception V3's non-square kernels can. Its 34
layers at 1×7, 7×1, 1×3 and 3×1 are bit-exact with the width in the high byte. With the
halves swapped they are wrong in every element region
[HW sweep, H96 MAX M9, `tests/rk3576_net_gate.c --net iv3`].

The OUT_CVT triple is `0x40AC` / `0x40B0` / `0x40B4`: offset, scale and shift, in three
consecutive registers. It is the same requant the RK3588 puts at `0x4080`-`0x4088`. The
reading comes from position (a small signed value ahead of a scale and a shift), and a
register sweep on hardware confirms it. Forcing the offset to 16 adds exactly 16 to
every output channel, and forcing the scale to 0 zeroes the surface. Raising the shift
attenuates a saturated result into range. [HW sweep, H96 MAX M9]

## Open questions

- **On-chip accumulation across the fp16 `ic` split.** The DPU eltwise stage does this
  job on the RK3588 (`ROCKET_KACC`) and would remove `ic/16` readbacks here. At the
  16-channel contraction, the partial it would read back is a plain, dense, complete
  fp16 cube, so nothing blocks it, the output writer included.
- **The ARGB first conv's int8 weight cube.** The float cube is decoded, and the fp16
  first conv computes on it. The int8 cube is not decoded, and it is a different
  object: the depthwise path shows that this part's int8 and float cubes differ. A
  quantized ARGB capture does take the path, but its weights do not survive
  quantization as a findable value set. This cube likely has to be read off the part,
  the way the depthwise int8 cube was. The entry `rocket_conv2d_int8_rk3576()` owns
  `ic <= 4` through its own packed-image path.
- **The vendor's float BS arrangement.** Registers `0x501C` and `0x4044` together
  select a BS operand arrangement that this library's coefficient group is not packed
  for. The library's arrangement is bit-exact and the vendor's is not, so the emitter
  programs the library's. Decoding the vendor's would say what its epilogue buys. See
  the section above for the leave-one-out result.
- **What the vendor's other two BS arrangements buy.** Register `0x501C` is decoded as
  the BS operand-source register (see the epilogue section). This library packs for
  the arrangement that is bit-exact. What the epilogue and float arrangements read
  instead is not decoded. The one-in-four survival rate under the epilogue arrangement
  is the lead.
- **`0x1060` is probably `CVT_OFFSET3`.** Registers `0x1054`-`0x105C` are the first
  three CVT offsets and there are four CVT scales, so the fourth offset almost
  certainly sits here. Every capture leaves it zero, including the ARGB ones, which use
  only three channels. Nothing pins it, so it stays in the address-placement shotgun
  list rather than being claimed.
- **`0x1064`.** The one live vendor dump (an `rknpu` kernel capture taken at submit,
  not one of the ten programs decoded out of `.rknn` files) carries `0x777` here. Every
  stored program carries 0, so the vendor runtime patches this register at load time,
  the way it patches the addresses. It is the offset the RK3588 calls `CNA_FC_CON1`.
  Forcing it to `0x777` on this library's path changes nothing (nor does forcing
  `0x1060` or `0x1074`). The RK3588 runs its own matmul with this register at 0, so it
  is not required for the direct-conv datapath.

  If a fully-connected mode is ever driven, re-test this register there. The wider
  point is methodological: a `.rknn`-derived program is the *stored* register set, and
  any value the runtime patches in is invisible in it.

## Results on silicon

Measured on an H96 MAX M9 (RK3576, mainline 7.1.3), driving the emitter above
through `librocketnpu` with an int8 1×1 conv, IC=OC=32, 8×8, symmetric
quantization. [HW sweep, H96]

The encoder is the difference between no output and an output surface. With the
RK3588 register program this part writes nothing at all, and the output BO comes back
untouched. With the RK3576 program the DPU writes the full, correctly-sized surface:
2048 bytes for a 32-channel 8×8 int8 output. It is the same NC1HWC2 cube the RK3588
uses (C2=16, surface stride `ow*oh` in 16-byte atoms, which is why `0x401c` holds
`ow*oh` and not `ow*oh*16`).

The part computes convolutions, and the emitter is what makes it do so. The test
drives a k=3 SAME 32×32 int8 conv, IC=OC=32, with a dense random weight set (all 9216
weights non-zero). Its feature tensor spans the full signed int8 range. The conv
reproduces the CPU model bit-exactly across the whole surface, 32768/32768, max |diff|
0, and the same conv unpadded is 28800/28800.

That single result carries most of the encoding. It shows that each of these is
programmed correctly:

- The CNA feature and weight DMA
- The CBUF staging
- The CSC
- The CMAC
- The BS bias add
- The OUT_CVT requant
- The output geometry
- The writer

The same result also settles the weight cube independently. A dense random set of 9216
weights cannot match bit-exactly under a permuted layout. So at IC=OC=32 the RK3576
takes the same weight cube as the RK3588 (`weight_conv_int8`, oc-group 32 / ic-group
32).

The envelope is every geometry whose channel counts are programmed as multiples of 32
and whose plane fits the CBUF budget. Across the sweep below, the emitter reproduces
the CPU model bit-exactly over the whole surface, every time, repeatably across
separate power sessions. The sweep covers these axes:

- IC and OC from 8 to 128
- Planes from 8×8 to 128×64
- Kernels k1, k3 and k5
- Stride 1 and 2
- VALID and SAME padding
- Feature tensors that vary on all three axes

Both boundaries below are properties of how the operands are described, not of the
datapath. Both are closed by construction: the channel counts come from
`rocket_rk3576_pad_ic()` / `_pad_oc()`, and the CBUF budget is the `rocket_hw_rk3576`
machine-parameter profile.

## Channel counts as multiples of 32

The register `ic` and `oc` must each be a multiple of 32, the group that
`weight_conv_int8` pads its two channel axes to. Size the feature cube, the output BO
and the coefficient buffer to the padded counts. Without that padding, a convolution
whose channel counts are not multiples of 32 computes wrong, at every geometry. No
capture shows this, because every captured geometry already satisfies it.
[HW sweep, H96 MAX M9]

On the input axis, ic of 8, 16, 17 and 48 is wrong at every kernel and plane size.
Values of 32, 64 and 96 are exact. An ic=17 conv fails despite spanning two whole C2=16
surfaces, so the unit is the 32-channel MAC group and not the surface. Padding the
feature cube alone does not help, because every read past `ic` already lands in mapped
zeros. The register value is what matters, and passing the padded count with the cube
zero-filled to match is bit-exact.

On the output axis, a partial group trips two different mechanisms. That is why it
presents two ways, and why a sweep at one kernel size reads as "oc is unconstrained":

- At k=1, the DPU writes only output row 0 of the trailing group and leaves the rest
  of the surface untouched. The oc values 8, 16, 40 and 48 fail this way, while 24, 32
  and 64 are exact. The rule that fits k=1 alone is surface parity (`ceil(oc/16)` must
  be even), which is why oc=24 passes there.
- At k>1, the weights come out wrong instead, and oc=24 and oc=56 fail even though
  their surface count is even. The register `WEIGHT_BYTES` (`0x101C`) is
  `ic*oc*kh*kw`, which describes a cube tighter than the padded one the caller
  supplies. In the `[kh][kw][oc2=32][ic2=32]` cube, that truncation drops whole
  `(kh,kw)` planes rather than trimming `oc` inside each. At k=1 there is only one such
  plane, so the same shortfall lands harmlessly on the unused `oc2` slots. That is
  exactly why k=1 is the tolerant case.

Rounding `oc` up to 32 satisfies both. The trap to recognize is the first mechanism: a
partially-written surface reads as an output-geometry defect, not a channel one.

The vendor does not pad. Its `conv2d` capture is ic=16 with weight bytes
`ic*oc*kh*kw`, the tight size. So the vendor packs a 16-channel weight group where
`weight_conv_int8` pads to 32. This project's emitter matches that program 131/131, so
the difference is in the cube, not the registers. Padding is how the RK3588 cube
expresses the same conv. A partial-group weight packing would be the other way, at half
the weight bytes for such a layer.

## The CBUF budget and `0x1040`

The CBUF budget is programmable, and `0x1040` is the register that programs it.

A conv whose feature plane does not fit computes wrong, with the DPU still writing a
full surface, no IOMMU fault and nothing in dmesg. The corruption is graded. One row
past the budget, about half the surface is still bit-exact. It degrades with the depth
of the overflow until almost nothing is (at ic=32, iw=16: 4096 granules is exact, 4104
gives 132001/262656, 4160 gives 1090/266240). Nothing else on this part presents that
way, which makes it the instrument: grow a known-good conv until it breaks.

The budget is one scalar in the CNA's granule unit. The register that the published
map carries as the constant `0x10000000` sets it:

```
ceil(iw * ic / 64) granules per feature row  x  the task's input rows  <=  4096 + F
      where F = bits[16:27] of CNA_CBUF_CON0 (0x1040)
```

Every boundary is exact. Measured ceilings, each the last granule count that computes
bit-exactly over the whole surface:

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

Widths of 16, 32 and 64 all break at the same granule totals: 4096 at F=0 and 6144 at
F=2048. So the limit is the granule count, not rows, width or channels individually.
The data budget does not exceed 6144 granules (384 KiB): F=3056 still measures 6144.
[HW sweep, H96 MAX M9]

### Single-bit rungs and conditional delivery

**Only single-bit F values deliver their face value.** Each power of two measures
exactly 4096+F, but a combination delivers less than the sum of its bits. The two
combinations tested both land on 5152, whichever second bit is set, so the field is
not simply an integer the CNA adds. An emitter that computes an arbitrary F programs a
budget the hardware does not honor. That corrupts silently, since a plane over its
allowance still writes a full surface. [HW sweep, H96 MAX M9]

Use the measured rungs (0, 256, 512, 1024, 2048) and round a deficit up to one. The
cost is a little weight headroom the plane did not need.

**The 256 and 512 rungs deliver conditionally.** Where they do not, each delivers 4096
granules, the F=0 budget. A task the planner puts on one of them then overruns its
allowance. It writes a full, correctly sized surface with a wrong tail: every output
row past `4096 / entries`. The rungs 0, 1024 and 2048 are unaffected at every
footprint tried on either path. The vendor's own windowed depthwise capture is a k=3
program at F=1024.

**The failure is a band, not a ceiling, and that is what makes it hard to see.** The
allowance is a ladder, and the planner takes the smallest rung that covers the window.
As the window grows, the surface is exact while F=0 still covers it. It is then wrong
across the band of windows that select an unhonored rung, and exact again where the
next rung up delivers. A walk of one task per height at 160×160 ic = oc = 32 k3
depthwise, 80 granules a row, shows the band. It reads exact at 45-51, wrong at 52-57
and exact at 58-76.

**A probe that bisects the window reports whichever edge it walks into.** A bisection
rests on "a smaller window is never worse", which is what a capacity bound means, and
this is not one. Read the map.

The condition differs by path, and on the depthwise path no footprint threshold fits.
The table forces each rung under one fixed window that F=0 does not buy:

| path | kernel | oc | resident footprint | F=256 / F=512 |
|---|---|---|---|---|
| direct | 1×1 | 32 | 1024 B = 16 granules | deliver |
| direct | 1×1 | 64 | 2048 B = 32 granules | fall back |
| depthwise | 1×1 | 32 | 64 B = 1 granule | deliver |
| depthwise | 1×1 | 256 | 512 B = 8 granules | deliver |
| depthwise | 1×1 | 1024 | 2048 B = 32 granules | deliver |
| depthwise | 2×2 | 32 | 256 B = 4 granules | fall back |
| depthwise | 3×3 | 32 | 576 B = 9 granules | fall back |
| depthwise | 5×5 | 32 | 1600 B = 25 granules | fall back |

On the depthwise path, 4 granules is dead where 32 is live, which refutes a threshold
in both directions. What survives the eight cells is the tap count: single-tap
delivers at three channel counts spanning 32x, and multi-tap never does. A
64-channel-group footprint (`R76_DW_W_GROUP_INT8`) against the direct path's
16-granule threshold fits k1, k3 and k5. The 2×2 cell refutes it, which is why that
cell is in the probe. [HW sweep, H96 MAX M9, `tests/rk3576_conv_lib_gate.c rowmap`]

So the emitter declines the low rungs on the depthwise path rather than gating them on
a fitted quantity. That costs nothing, since the fallback rung is strictly larger and
always live.

### The direct path's weight-footprint condition

The direct path's own condition is the resident weight footprint, under 1 KiB, and the
rest of this subsection is that measurement.

**The kernel is not the axis, and a square-kernel sweep cannot say so.** A
characterization that holds `ic` at 32 and moves the kernel uses one granule total of
4352 and holds it constant across five plane widths. It reads k=1 exact everywhere and
k=3 and k=5 wrong everywhere:

| plane | entries/row | rows | k=1 | k=3 | k=5 |
|---|---|---|---|---|---|
| 16×544 | 8 | 544 | exact | wrong | n/a |
| 32×272 | 16 | 272 | exact | wrong | n/a |
| 64×136 | 32 | 136 | exact | wrong | wrong |
| 128×68 | 64 | 68 | exact | wrong | n/a |
| 272×32 | 136 | 32 | exact | wrong | n/a |

That table reads as "the rung needs `kh == 1`". Crossing the axes refutes that
reading. At the same 4352-granule total and the same 1×1 kernel, the same (row size,
row count) is bit-exact at `ic` 32 and wrong at `ic` 64 and 128. Both wrong cells are
68 rows of 64 granules, and both are wrong from output row 64. The F=0 control at 64
rows is exact, and so is the same plane forced under the boundary.

| 4352 granules, 1×1 kernel | entries/row | rows | result |
|---|---|---|---|
| iw 16, ic 32 | 8 | 544 | exact |
| iw 32, ic 32 | 16 | 272 | exact |
| iw 64, ic 32 | 32 | 136 | exact |
| iw 128, ic 32 | 64 | 68 | exact |
| iw 64, **ic 64** | 64 | 68 | **wrong from row 64** |
| iw 32, **ic 128** | 64 | 68 | **wrong from row 64** |

So `ic` at a fixed kernel and the kernel at a fixed `ic` move the same quantity,
`32*ic*kh*kw`, the resident weight slice. That is the quantity
`r76_weight_slice_cap()` is already stated over. The rung is live at 16 granules
(`ic` 32, k=1). It is dead at 32 granules (`ic` 64), 64 (`ic` 128), 144 (`ic` 32, k=3)
and 400 (`ic` 32, k=5).

What the rung delivers backs out to the row from the surviving prefix. In every case,
the last correct output row is the one fed by input row `4096 / entries`. That makes
4096 a measurement rather than a reading of a graded corruption.

The quantity is one output-channel group's slice, not the whole resident cube. Every
cell above that reaches a rung carries `oc` 32, one group, where the two are the same
number. So the emitter charges the cube, the smaller envelope.
[HW sweep, H96 MAX M9, `rk3576_conv_lib_gate rowmap`]

Holding the slice at the measured-live 16 granules and raising the group count
separates the two. At `oc` 64 and 96, cubes of 32 and 48 granules, F=256 and F=512
both still deliver. So the shipped rule is conservative rather than wrong. It stays
that way because relaxing it buys nothing. The rung it declines to use is replaced by a
strictly larger one that also delivers, at the same CBUF and with no extra submit.
[HW sweep, H96 MAX M9, `rk3576_conv_lib_gate rowmap`]

The threshold between 16 and 32 granules is still bracketed, not measured, and the
harness cannot narrow it. On the direct path, `ic` is padded to a multiple of 32. So
`32*ic*kh*kw` moves in 1024-byte steps at every shape that a square kernel and a whole
`ic` can build. A non-square kernel is what would land between them.

Everything past the threshold rounds up to 1024. The cost is the next rung, at the
same CBUF and with no extra submits. Where the weight path leaves no room for 1024, the
cost is a shorter row window: one more task, and not a wrong answer.

The correctness envelope does not reach this on its own, and a whole network does.
Every direct shape in the table sits at F=0. So does every depthwise one, except a
MobileNet's 3×3 over a 112-row plane. Nothing reaches the ic axis either. A direct rung
is programmed only where the plane is 4097-4608 granules and the weights are under
1 KiB. The matmul's own row planner is past that at every K it runs.

**The packed-image path keeps the direct rule and has never been driven at a rung.**
The widest stem in the corpus, Inception V3's 299×299, stages 5681 granules and lands
on F=2048. A 224×224 stem sits at F=0. So that axis is unverified rather than verified.
[HW sweep, H96 MAX M9, `tests/rk3576_conv_sym.c rung` and `rk3576_conv_lib_gate rowmap`,
with `tests/rk3576_conv_lib_gate.c` groups `rung256` and `dwbig` for the original kernel
reading]

### The captured words and the shared pool

The two values the captures carry are two points on the F scale, not a constant and a
variant. The word `0x10000000` (F=0) buys 4096 granules, and `0x14000000` (F=1024)
buys 5120. The second is what the vendor's windowed depthwise program needs. At 112
wide and ic=32 a row is 56 granules, and its 91-row window is 5096 granules. That
overflows 4096 and fits 5120 with 24 to spare.

The low bits of `0x1040` are the RK3588 field layout, and the vendor leaves them zero.
Bits 0-13 are live, and per Mesa's `registers.xml` they are `DATA_BANK[0:3]`,
`WEIGHT_BANK[4:7]`, `FC_DATA_BANK[8:10]`, `DATA_REUSE[12]` and `WEIGHT_REUSE[13]`.
Setting any single one of them on top of a working program corrupts the conv. So the
bank *fields* did not move in the re-pack. What moved is that this part also takes a
granule allowance in bits[16:27], which is reserved on the RK3588. [HW sweep]

Bit 28 is required (a `0x1040` of zero corrupts), bit 29 corrupts, and bits 14, 15,
30 and 31 are don't-care. [HW sweep]

The data and weight sides share one pool, and F trades between them. The weight path
stages per output-channel group, so its resident slice is `32 * ic * kh * kw` bytes
rather than the whole cube. That is why a 441 KiB weight cube computes fine. Sizing
that slice against F shows the trade directly. A 150 KiB slice is bit-exact at F=0 and
breaks at F=2048, and the ceiling falls as the data side grows.

| F | data allowance | weight slice ceiling | sum |
|---|---|---|---|
| 0 | 4096 gr = 256 KiB | 175 KiB exact, 200 KiB breaks | ~445 KiB |
| 1024 | 5120 gr = 320 KiB | 125 KiB exact, 150 KiB breaks | ~455 KiB |
| 2048 | 6144 gr = 384 KiB | below 75 KiB | ~450 KiB |

Each +1024 granules of data costs the weight path about the 64 KiB the data side
gained. All three pairs sum to roughly 448 KiB, 14 banks of the RK3588's 32 KiB, of
which the captures' default program takes 8 for data. The bank *count* is an inference
from that bank size, and what is measured is the granule budget, the trade and the
total. Reading the pool as 14 banks also explains the wedge: F at or past ~3060 leaves
the weight path nothing. The part then writes no output at all at any plane size,
which looks exactly like a wrong geometry encoder. [HW sweep]

Two consequences follow. First, the 8 x 32 KiB in `rocket_hw_rk3576` is the **default
data allocation**, not the physical CBUF. Raising F buys up to 1.5x the feature
capacity, at the cost of weight-slice headroom that must then be respected. Second, a
register the published map records as a constant is a tuning knob. An emitter that
copies the constant inherits the vendor's choice for a full-plane task rather than
making its own.

The boundary is sharp enough to plan against, and the pool figure survives a direct
test. A slice of exactly 192 KiB (6 banks, what the model leaves beside F=0) computes
bit-exactly at ic=1536 k=2. A 196 KiB slice at ic=1568 breaks. [HW sweep]

### Per-task planning of F

The emitter plans F per task, in `rocket_rk3576_cbuf_f()`. It picks the lowest live
rung whose budget covers the plane. On the direct path, the 256 and 512 rungs count as
live only where the resident weight cube is at most 16 granules. On the depthwise path
they never count as live, per above. Where the plane needs more than the data cap, or
the rung would starve the weight path, the emitter refuses the task. It refuses because
the recourse (a shorter row window, an ic split) is the caller's to choose.

The tiler-facing half is `rocket_rk3576_max_task_rows()`: the tallest window one task
can carry at the highest rung the weight slice leaves room for. Planning reproduces
both captured words, so the register-fidelity gate stays byte-identical. It turns every
plane between 4096 and 6144 granules from silent corruption into an exact result. That
is validated on the part at 4160, 4800, 5600 and 6144 granules, and at 6144 across
three widths. The flag `ROCKET_RK3576_CBUF_F` forces F and bypasses both checks, and it
is how the allowance was characterized.

## The weight slice and the output-channel group count

The resident weight slice caps how many output-channel groups compute. The pool
arithmetic above is characterized at one output-channel group, and it is not
sufficient on its own. A conv driving several groups loses the trailing ones well
before the slice reaches what the pool leaves it. It loses them one at a time as the
slice grows. The leading groups come back bit-exact and the rest wrong, with a full
surface written, no fault and nothing in dmesg. It reads as an output-channel defect
rather than a capacity one.

The governing quantity is the resident weight slice `32 * ic * kh * kw`, one
output-channel group. The same slice behaves identically whichever `(ic, kh, kw)`
produces it. Both `ic=512 k=3` and `ic=4608 k=1` give 144 KiB and compute all four
groups, while `ic=192 k=5` and `ic=544 k=3` sit near 150 KiB and lose two. Sweeping
`ic` at k=1 moves the slice in 1 KiB steps, which is what resolves the boundary at all.
At k=3 the step is 9 KiB and at k=5 it is 25. That coarse sampling makes the loss look
like a jump from four groups straight to two.

Measured at F=0 on a 4×2 plane at oc=128 (four groups), and cross-checked at k=3 and
k=5. The table gives the number of groups that come back bit-exact at each slice:

| slice | 144 KiB | 145 | 146 | 147 | 148 | 152 | 156 | 162 |
|---|---|---|---|---|---|---|---|---|
| groups exact | 4 | 3 | 3 | 3 | 3 | 2 | 2 | 1 |

So the usable rule is **`ic*kh*kw <= 4608`** (a 144 KiB slice), at which every output
channel count computes. Past it, the emitter refuses rather than running, in
`rocket_rk3576_cbuf_f()`, against a small measured table keyed on the group count
(144 KiB at four groups or more, 148 at three, 156 at two). The recourse is an ic
split, which is the caller's to choose. A single group is left to the pool check
alone. That check already lands on the measured single-group boundary (175 KiB
computes, 200 KiB does not, and the pool reaches its limit at 192 KiB). A graded
multi-group loss is by definition not a single-group effect.

The shape of the loss suggests a mechanism that the points do not settle. The boundary
falls as the group count rises, roughly as though each additional group costs a small
fixed staging allowance on top of the slice. That fits a weight path that stages the
next group while computing the current one. Two free parameters against six points is
not a mechanism, so the emitter carries the measured table rather than a formula.
[HW sweep, H96 MAX M9, measured 2026-07-25]

### Output-channel tiling under the group cap

The group count is the caller's choice, so the cap is one too. The table above bounds a
program, not a shape: its group count is what one program drives, not what the
convolution needs. Splitting the output channels across submits lowers the group count
per submit, which raises the slice the part takes. Read that way, the table is a
planner. It maps the slice a shape needs to the most output channels one submit can
carry.

```
slice <= 144 KiB   ->  no constraint      (four groups or more)
slice <= 148 KiB   ->  oc tile 96         (three groups)
slice <= 156 KiB   ->  oc tile 64         (two groups)
otherwise          ->  oc tile 32         (one group; the pool check governs)
```

`rocket_conv2d_int8_rk3576()` plans exactly that and costs one submit per tile. It
lifts fourteen of the emitter's refusals. The two that reach furthest, `ic=576 k=3`
(162 KiB) and `ic=5184 k=1` (162 KiB), are ones the four-group rule puts well out of
range. Both are bit-exact at four tiles of 32 channels. So `ic*kh*kw <= 4608` is the
single-program rule, not the part's. Through the library the bound is the CBUF pool at
one group, and the rule is **`32*ic*kh*kw <= 175 KiB`**, that is `ic*kh*kw <= 5600`.

One refusal survives the split: `ic=256 k=5` needs 200 KiB, which is past the pool at a
single group. The only recourse there is an input-channel split, which the on-chip
requant forecloses. The int8 partials cannot be summed without quantizing each one. That
shape belongs on the int32-output writer.

The gate `tests/rk3576_conv_lib_gate.c` holds this result. It drives the emitter gate's
own shape table through the library entries. It passes 111 shapes, asserts the one
refusal in both directions, and finds every computed shape bit-exact against a CPU
model. [HW sweep, H96 MAX M9, measured 2026-07-27]

The same sweep confirms that the cube size is not the constraint. The shape
`ic=448 k=3 oc=128` has a 126 KiB slice under a 504 KiB cube and is exact on all four
groups. The shape `ic=256 k=5 oc=32` has a 200 KiB slice under a 200 KiB cube and does
not fit at all. Group count on its own is not the constraint either: `oc=256` at
`ic=64` is eight groups and exact.

### The DPU epilogue and the operand DMAs

Everything downstream of the MAC works, and the bias path is bit-exact. A bias-only
probe (features and weights zero, a per-channel bias) reproduces the CPU model exactly
across all 32 output channels (2048/2048 bytes, max |diff| 0). The OUT_CVT sweep moves
that surface exactly as its offset, scale and shift predict. The bias path computes
this only with the coefficient-buffer layout below.

The OUT_CVT requant rounds to nearest rather than truncating. A border whose exact
value is -10.0006 comes back as -10, where an arithmetic shift gives -11. Adding half an
LSB before the shift, as `(acc*scale + (1<<(shift-1))) >> shift`, reproduces every
pixel of the probes above exactly. Without it, every otherwise-correct surface is off
by one wherever the fraction crosses a half. [HW sweep, H96]

All four operand DMAs fire and reach their programmed addresses. Each base below,
pointed at an unmapped IOVA, makes the job fault with its own IOMMU status word, while
an untouched run is clean:

- The weight base, `0x1110`
- The feature base, `0x1088`
- The output base, `0x4018`
- The bias base, `0x5020`

The output and bias bases are the positive controls, because they are known to be read
and written. So a "no fault" reading elsewhere would have been evidence, and a fault
everywhere confirms that no operand loader is silently idle. The fault signature is two
`rk_iommu ... Enable stall request timed out` lines. [HW sweep, H96]

**That fault is not as recoverable as the next job makes it look.** The job
immediately after it computes normally. But a session that ends with these faults can
leave the part in a state where the weight loader never arms again. Every later job
then completes cleanly, with no timeout and no new dmesg line, and writes a bias-only
surface. The state survived 7 hours of idle and a full `rmmod rocket` /
`modprobe rocket` cycle, and only a reboot cleared it [HW sweep, H96]. Budget a reboot
after any unmapped-IOVA probing, and re-run a known-good conv before trusting a
negative result taken afterward.

The feature strides are confirmed one axis at a time. A uniform feature fill proves
nothing about addressing, because every read that lands inside the buffer returns the
same byte. So the sweep varies each axis with the other two held flat. Varying along
rows, along columns and across channel *groups* each reproduces the CPU model
bit-exactly. The three axes confirm, in turn, the line stride (`0x1090` = `iw*4`, in
4-byte units), the C2=16 channel atom inside a row, and the CBUF surface/group stride
(`0x1094`/`0x1098` = `iw*ih`, in 16-byte units). [HW sweep, H96]

### The coefficient buffer

**The bias/coefficient buffer at `BS_BASE_ADDR` (`0x5020`) is not a flat per-OC
int32 array.** It is the structure the vendor's own coefficient buffer uses: groups of
64 bytes covering 8 output channels each, holding these fields:

| field | type | offset in group |
|---|---|---|
| `A[oc]`, per-channel bias term | int32 | `(oc%8)*4` |
| `B[oc]`, weight-zero-point correction | int16 | `32 + (oc%8)*2` |
| `C[oc]`, per-channel multiplier | int16 | `48 + (oc%8)*2` |

Output channel `oc` lives at group `oc/8`. The part reads a flat int32 array handed to
it *as* this structure. Only the first 16 channels then get a term at all, on
alternating channels, at 1024x the intended magnitude. That pattern is an artifact of
the layout, not a 2-byte operand DMA.

**The BS stage adds `A` and then multiplies by `C`**: the surface is
`(acc + A[oc] + B[oc]*sum(x)) * C[oc]`. The probe reads this off the part against a
known accumulator and a known bias. It sets `C` to 1 on the even channels and 2 on the
odd. There, `(acc + A)*C` explains 32 of 32 channels, and `acc*C + A` explains only the
16 where `C` is 1 [HW sweep, H96, `tests/rk3576_coeff_c.c`]. So a bias quantized in the
accumulator domain rides the per-channel gain for free, and must not be pre-divided by
it. With the order backwards, the BS stage scales the bias by the wrong channel gain, a
plausible surface rather than a fault.

The BS result saturates at int32, after the shift word, not before it. A sweep with the
shift word at zero walks `(acc + A)*C` across `2^31` at a fixed accumulator. Every
inexact cell implies the same ceiling, 2.147e9 to 2.158e9 against `2^31` = 2.1475e9,
and none of them wraps [HW sweep, H96]. So with no shift,
`C[oc] <= INT32_MAX / max(abs(acc + A))`, which falls as the layer's fan-in grows.

Under a nonzero shift the multiply is held wide. At a shift of 14 and `C = 16384`, products from
2.6e9 to 3.3e13 read back exactly [HW sweep, H96 MAX M9, `tests/rk3576_coeff_c.c shift`,
2026-09-23]. A saturate-first product reads 131071 there. The bound is therefore
`((acc + A)*C) >> s <= INT32_MAX`, and with a shift C can use its int16 field at any fan-in.

**`C` is per channel, and it gates the whole BS stage.** Every one of 32 channels
reads its own `C` at 32 distinct values. At `C=1` the datapath is bit-exact, and `C=4`
scales by exactly 4, so `C` is a live linear per-channel multiplier. At `C=0` the DPU
writes a full, correctly sized, entirely empty surface, whatever the CNA and the MAC
did. That holds even for a conv with no bias at all. [HW sweep, H96]

An all-zero coefficient buffer is what any caller unaware of this layout hands over.
The resulting empty surface is indistinguishable by inspection from a wrong geometry
encoder, and it is the single most expensive trap on this part. Pack the buffer with
`rocket_rk3576_pack_coeff()` rather than by hand. The `B` term is pinned to 0, which is
what symmetric quantization wants. It is unvalidated against a non-zero weight zero
point, and nothing observed so far needs it. [HW sweep, H96]

**How `C` is read depends on the precision, so `C = 1` is not portable across it.** The
integer multiplier above is the int8 program's reading. A float program reads the same
field as fp16, where the integer `1` is the denormal 6e-8. It underflows the surface to
empty, the identical signature, from arithmetic rather than from a gate. In fp16, `1.0`
is `0x3C00`. See [the fp16 datapath](#fp16-the-datapath-and-the-contraction-width-that-bounds-it).

### The feature domain and the padded border

The feature domain is signed int8 and needs no centering. A k=1 conv whose output plane
sweeps every int8 value once reproduces the CPU model over the whole range. That is one
flat run from -128 to 127, 8192/8192, max |diff| 0, with the probe feeding raw signed
bytes and no `+0x80` applied anywhere. [HW sweep, H96]

The DPU epilogue is independently exact for negative accumulators. A bias-only probe
driven with negative per-channel biases, which the MAC never touches, is bit-exact on
all 32 channels. The datapath is not uint8-centered, and a feature tensor carrying
negatives is not wrong for it. A run that says otherwise was made against an all-zero
coefficient buffer, where the surface is empty whatever the features are. [HW sweep, H96]

The padded border is exact. A k=3 SAME 32×32 conv with dense random weights reproduces
the CPU model over the whole surface, ring included, 32768/32768, max |diff| 0. So does
the pad probe, which drives the border with a -128 pad tap against zero features. The
border pad constant `0x1084`, the pad word `0x1080` and the window geometry are all
right. A border defect on this path is the same coefficient-buffer artifact.
[HW sweep, H96]

## The row window and its three registers

A plane over its allowance computes a wrong conv, not a slow one, so the recourse the
allowance planner names has to exist. That recourse is a split by input rows. Each task
reads a row window of the full plane and writes the output rows that window supports.
The task's program still describes the full plane alongside the window. The planner
`rocket_rk3576_plan_rows()` lays the sequence out, and the emitter takes the window in
`conv_params_t` `ih`/`oh` against `ih_full`/`oh_full`.

The caller's whole job per task is two byte offsets. Both are plain row strides:
`iy0*iw*16` into the feature cube and `oy0*ow*16` into the output. The cubes are NC1HWC2
with a 16-byte channel atom, and the CNA takes the DDR group stride from the full plane
(`0x1094` = `iw*ih_full` by default). So one base plus a row offset addresses that row
of every channel group. The vendor's sliced capture is the direct evidence on the
feature side. Its fourth task reads input rows 111.. and `0x1088` carries exactly that
row offset.

`0x1094` is a quantity the emitter fills, not a derivation the hardware repeats. The
part honors any value at or above the plane. That lets a consumer read a producer's
surface whose groups are further apart than its own plane would put them. It is
bit-exact at `+3`, `+16` and `+64` elements over five geometries, with one row task and
with ten. A control that lays the same padded buffer out without setting the register
differs every time [HW sweep, H96 MAX M9, `tests/rk3576_surf_stride.c`].

The CNA does not read `0x1098` as a second DDR stride. The padded cases are bit-exact
with it left at `round4(iw*fetch_rows)` [HW sweep, H96 MAX M9, `tests/rk3576_surf_stride.c`].

Driving the window settles the three registers that take a second value only on a
windowed program. The sweep measures all three with an `oc=64` conv cut into 32-, 16-
and 8-row windows of the same 64-row plane. Its two output-channel groups expose
group 1. [HW sweep, H96 MAX M9]

`0x40B8` is the channel-group jump, `ow * (2*oh_full - oh_task)`, and the form is swept,
not fitted. It holds for a conv driving many groups at a wide kernel. The form is
bit-exact across 2, 4, 8 and 16 output-channel groups, k1/k3/k5, stride 1 and 2, and
VALID and SAME. It is also exact on wide, tall and ragged planes, at ic 32/64/128, and
at 3 to 16 windows. That sweep is the `surface` group of the conv gate, and the
description below is what it confirms. [HW sweep, H96 MAX M9, measured 2026-07-25]

The jump is a full destination surface plus the rows of it this task does not write.
The writer walks the task's rows and then adds this jump to reach the same rows of the
next group. So a windowed task must be told about the rows it skipped, or every group
past the first lands short. That is why an `oc=32` split is exact while `oc=64` is half
wrong. It is also why the defect reads as an output-channel fault rather than a
windowing one.

Only this form is bit-exact at all three window sizes. The forms `ow*oh_full`,
`ow*oh_task`, `2*ow*oh_full` and 0 each corrupt the whole surface. When the task is the
whole plane, the form reduces to `ow*oh_full`, which is what both full-plane captures
carry. So the register reads like a constant until a window separates the two terms.

`0x1018` carries one live bit, and it is not the byte pair. Bit 30 is required:
clearing it corrupts the whole surface. The low 16 bits are don't-care on the direct
int8 path. The values `0x0404`, `0x0505`, `0x040b` and `0x0000` are all bit-exact,
windowed and un-windowed alike.

The windowed value does not stop the DPU on an un-windowed task. A surface that
suggests otherwise is the cold-start wall, the trap this sheet warns about. Read a
single "the DPU did not write" as the wall, and re-run before believing it.

`0x1038` likewise carries one live bit, bit 31. The values `0x07`, `0x010e` and `0x10`
are all bit-exact, and `0x80000010` corrupts the surface.

The emitter still reproduces the vendor's values for all three, so the
register-fidelity gate stays byte-identical to the captures.

The split computes bit-exactly against the CPU model, over the whole surface, at every
shape tried. Those include 112×112 and 224×224 at ic=32, the geometries with no
single-task plan at all (6272 and 25088 granules against a 6144 cap). They also include
forced-cap splits from 2 to 16 windows across ic/oc of 32/64/128, k1/k3/k5, stride 1
and 2, VALID and SAME.

The planner spreads the output rows evenly over the fewest tasks that fit, rather than
taking greedy maximum windows. A greedy pass leaves a ragged tail: a 112-row plane at a
109-row cap comes out 109+2+1 instead of two windows of 56. Every extra task costs a
submit.

A second sweep measures what a window costs, with the cold-start wall closed. It cuts
the same 64×64 ic64 oc64 k3 SAME output into progressively more windows and submits
them with no inter-task gap. [HW sweep, H96 MAX M9, measured 2026-07-25]

| windows | 2 | 4 | 5 | 11 | 32 |
|---|---|---|---|---|---|
| wall time | 2.5 ms | 4.8 ms | 6.0 ms | 13.0 ms | 38.7 ms |

The cost is flat at ~1.2 ms per window, so a split is linear in windows and the
fewest-that-fit rule is the right one. Almost all of that is submit-and-wait. Emitting a
window's regcmd on the host takes 0.02-0.05 ms, and a single-task conv costs 1.2 ms
whether its output is 2 KB or 1.6 MB.

The floor is the driver's completion poll, not the silicon. This part has no maskable
completion IRQ, so `rocket` polls on an hrtimer. Shortening that interval moves the
per-task cost with it almost exactly. Retiring on the DPU's own completion, rather than
on `PC_DONE`, is what makes it safe to shorten. The `PC_DONE` signal fires before the
DPU's writes have drained. The patch `patches/rk3576/npu/0012` retires on the DPU's
completion and takes the period to 50 us, and 0011 removes the IOMMU half of the hazard.

The per-task figures in this section were measured without 0012, at ~1.2 ms. With 0012
the floor is ~0.44 ms. See "The per-submit floor is the driver's completion poll" below.

## `ih_full` on a single-task plan

A single-task plan needs `ih_full` and `oh_full` set from the plane, the same as a split
plan. The planner `rocket_rk3576_plan_rows()` can return one task whose row window is
shorter than the plane. A caller that sets `ih_full`/`oh_full` only when the plan splits
then mis-programs the DDR channel-group stride.

This is ordinary geometry, not an edge case. Any stride greater than 1 whose output does
not consume the plane leaves trailing input rows unread. So a 32×32 k1 s2 VALID conv
plans one task over 31 of its 32 rows. With `ih_full` left at 0, the emitter takes the
group stride from the window. Register `0x1094` then comes out `iw*31` instead of
`iw*32`, and every channel group past the first reads at the wrong offset.

The symptom is a full, correctly sized surface bearing no relation to the input. There
is no fault and no dmesg line, and the surface looks identical to a broken geometry
encoder's. The un-windowed path in `rk3576_first_light` sets the window directly and
never trips it. So the defect appears only once a caller routes every conv through the
planner, which is what a tiler does. Set `ih_full`/`oh_full` from the plane on every
task. [HW sweep, H96 MAX M9]

## The geometry fields are narrower than their register halves

The encoder writes each extent into a 16-bit half or a whole register. The part computes
with fewer bits. Past each field a task writes a full, plausible, wrong surface and
completes normally. Measured through the public entries, every element scored [HW sweep,
H96 MAX M9, 2026-09-26, `tests/rk3576_conv_width_probe`]:

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

**A wrapped width also breaks the next legal job in the same process.** After an
8300-wide task, an 8192-wide one returned 259665 of 262144 elements wrong and a 2048-wide
one 64961 of 65536, both with rc 0. A third job retired, and the entry's guard redid it
exactly. The mechanism is not decoded. So a driver must refuse these extents before the
device sees one, not recover after.

The library refuses every extent past its field, at claim time and in the encoder. Nothing
on this part tiles a convolution's width. The probe could not say which width register
holds the 13-bit field, since every arm had the input and output widths equal. Strides 8
and 16 program 0 and did not run.

### The fields the vendor compiler checks

RKNN-Toolkit2 checks each field it emits against its width and prints the ones that
overflow, as `REGTASK: ... target: f2, offset: 0x1090, shift = 0, limit: 0x3fff`. One-op
graphs sized past a field along one axis at a time make it name fields
[`rocket-userspace/tests/data/rk3576-vendor-capture/regtask/mkregtask.py`, toolkits 2.2.0
and 2.3.0, 2026-09-27]. On the RK3588 every field it printed matches Mesa's `registers.xml`:
DPU_RDMA `0x500C` [12:0], DPU `0x4038` [12:0] and DPU `0x403C` [12:0] and [28:16]
[source-confirmed]. So the method reads the compiler's widths correctly. The RK3576 prints
target `f2` and these:

| Offset | Field | Printed by |
|---|---|---|
| CNA `0x1090`, the line stride | [13:0], limit `0x3FFF` | a stride-2 1×1 conv 16400 wide |
| CORE `0x3020` | [12:0] | a depthwise conv of 8400 channels |
| DPU `0x402C` | [12:0] | the same |
| DPU_RDMA `0x5014` | [12:0] | the same |

A compiler width is the compiler's, and the line stride shows it. The encoder writes
`iw*4` there, and the row planner gives 2-row tasks at `iw` 4096 and 6000. Those tasks carry
`0x4000` and `0x5DC0`, past the 14 bits, and compute exactly [HW sweep, H96 MAX M9,
2026-09-27, `tests/rk3576_conv_width_probe r4`]. So read a printed width as a bound to test,
not as one measured. The three channel fields agree with the 13 bits in the table above.

The sweep sees only fields the compiler overflows. An axis it tiles, splits across cores or
sends to the CPU prints nothing. On both targets that was width, height, input and output
channels, and a matmul's K and M.

## The first-conv ARGB sub-encoding

A convolution whose input is a packed image runs on its own CNA datapath. The image
carries 3 or 4 interleaved bytes per pixel rather than an NC1HWC2 cube. This datapath
is not the normal program at a small channel count: about sixteen registers are packed
differently.

It is the only way a vision model's stem runs on this part at all. The normal path
needs `ic` a multiple of 32 (see below), and an image is 3. Twelve captured programs
pin it, at two image sizes (224×224 and 64×64) and two row splits.

The RK3588 has the same datapath. Mesa drives it as `CNA_CONV_CON1`
`NONALIGN_DMA | GROUP_LINE_OFF | ARGB_IN(8)` for a 1-channel input. Register `0x100C`
is one of the few geometry registers the RK3576 does not re-pack. On both parts,
`GROUP_LINE_OFF` is bit 29 and `ARGB_IN` is bits[15:12]. The captured `ARGB_IN` of
`0xA` sits exactly where Mesa's 1-channel `0x8` does, one step per extra image channel,
so the field is `8 | (image_channels - 1)`. The `CONV_MODE` field is 6, a third value
beside direct (0) and depthwise (1).

The CNA reads the packed row straight out of DDR. The CVT, bypassed on every other
layer, expands each pixel to four int8 lanes while applying a per-channel scale and
offset. The kernel's horizontal extent is then folded into the channel axis: the conv
the MAC sees is `kh x 1` over `4*kw` channels. That is why these programs carry two
disagreeing channel counts, the sub-encoding's clearest signature in a capture.

| Register | Normal path | ARGB path |
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
| `0x1054`-`0x105C` | 0 | the uint8 zero point, negated, per live channel |
| `0x1084` pad const | `zp - 0x80`, already centered | raw `zp + 0x80`, one byte per channel |
| `0x1090` line stride | `iw*4` (a cube row in 4-byte words) | `iw*ic/16` (the packed row in granules) |
| `0x1078` hi | `iw-1` | `iw*ic/16 - 1` |
| `0x1094`, `0x1098` | plane / task surface strides | both `line_stride * ih` |
| `0x118C` | `(iw-1)<<16 \| (ih_full-1)` | `(entries-1)<<16 \| (entries-2)` |
| CORE `0x3018` | `0x10000001` | `0x10000081` |

Everything past the CNA is the direct path's, including `0x40B8`'s channel-group
jump, so this is a CNA sub-encoding and not a second pipeline.

Three things follow that a caller must act on. The feature buffer is a packed image,
`iw*ic` bytes per row, so a row-window plan's feature offsets are in those units. The
pixels are raw uint8 and the converter does the centering. That is why `0x1084` pads in
the raw byte domain: the pad is inserted before the CVT runs. And `iw` must be a
multiple of 16, because both the DDR row stride and the CBUF row are counted in 16-byte
granules.

The `+1` on the CBUF row is the fold's lookahead. The row is the image row expanded to
4 lanes per pixel (`iw*4` bytes, `iw/16` granules), plus one granule. The extra granule
is there because the last output column reads `kw` pixels and reaches past the granule
its own pixel sits in. Both captured widths carry exactly `+1`.

**`0x118C` is not a plane extent here.** A normal-path decode of an ARGB capture reads
a nonsense full-plane height out of it. An ARGB program carries no plane height
anywhere, because the packed image is a single surface. So `0x1094` and `0x1098` both
hold the task's own rows, and there is no channel-group stride to describe.

One geometry reading is an inference. Every ARGB capture is `kw=3`, so `4*kw` and a
constant 12 fit the programmed channel count equally. Several readings of the
48-element weight kernel fit equally too. The mechanism above (4 lanes per pixel
column, `kw` columns) is what makes `4*kw` the reading rather than a curve fit. A first
conv at another kernel width is unvalidated.

### The ARGB weight cube

The captures carry register programs, not weight BOs. So the found programs pin only
the cube's size and stride, and nothing about its byte order. The size is
`oc * kh * round16(4*kw)` bytes, one 16-byte row per kernel row per output channel. The
float cube's layout is decoded from captures manufactured with weights unique per
position (`tests/data/rk3576-vendor-capture/argb/mkargb.py`):

```
slot(oc, c, kh, kw) = (oc/16) * (KH*KW*64)
                    + kh * (KW*64)
                    + kw * 64
                    + (oc%16) * 4
                    + c
```

In this layout:

- Each weight sits in a sixteen-bit slot.
- Each (output channel, tap) has four lanes. Lane `c` carries image channel `c`, and
  any lane past `ic` is don't-care.
- Output channels are interleaved in groups of sixteen inside one tap.
- The tap axis is kh-outer.

The 64 is 16 output channels times 4 lanes. The term `4*kw` is not rounded up to 16
here, so this layout's element count is not the byte size the register program
declares. At `oc=16 k=3` the cube is 576 halfwords, where the declared size is 768
bytes. The layout holds at `ic` 1/3/4, `k` 1/3/5/7 and `oc` 16/32/48/64, at four
widths, three heights and both strides. The packer is
`rocket_rk3576_weight_argb_fp16()`. [source-confirmed, RKNN-Toolkit2 rk3576 float build]

**The int8 cube is not this one.** A quantized ARGB capture does take the path:
`0x100C` reads `0x2000a006`, exactly the found value. But its weights do not survive
quantization as a findable value set. So the probe reads this cube off the part
instead, with an impulse image and a one-byte cube (`rk3576_conv_gate fcmap`). Every
live byte then names its output channel, both taps and its lane at once. The answer is
a different object from the float cube in every axis but the lanes:

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
- Lane `c` carries image channel `c`.

The map is a bijection over every live byte at `oc` 32 and 64, `k` 3/5/7 and `ic` 3
and 4. That covers 1152, 2304, 3200, 6272 and 864 live bytes. Each lands on exactly one
output position of one channel, with none left over and none doubled. The map is also
translation-invariant: the impulse moved three and six pixels reads the same.
[HW sweep, H96 MAX M9]

The 32-channel group is observable only above one group. At `oc=32` a flat `oc*R` fits
equally, and `oc=64` is what separates them. The `k=5` and `k=7` cubes separate `R`
from a constant 16, because `R` is 32 there. The packer is
`rocket_rk3576_weight_argb_int8()`. [HW sweep, H96 MAX M9]

**The converter's offset is inert, and the packed byte is a plain signed int8.** The
table above lists `0x1054`-`0x105C` as the uint8 zero point the CVT subtracts, as the
datapath's description says. But every ARGB capture is zero point 0, so no capture
exercises those registers. Driven on the part, they do nothing. An image written at
`raw = s + (zp + 0x80)` comes back with the raw byte read as a signed int8 and no
subtraction at all. The byte `0x80` reads -128, `0xC0` reads -64 and `0xFF` reads -1
[HW sweep, H96 MAX M9].

So a caller writes two's complement and folds the input zero point into the coefficient
group's `A` term, exactly as the direct path does. The border constant must still be the
stored zero point, so that a pad tap's true value is zero. [HW sweep, H96 MAX M9]

### Geometry bounds of the int8 first conv

The int8 first conv adds four geometry bounds. None of them appears in any capture,
because every captured first conv is a 3×3 stride-2 SAME convolution that satisfies all
four by construction. Three are silent when violated, so the library entry refuses them
rather than leaving them to compute.

**The left pad must be non-zero.** At `pad_left = 0` the DPU writes nothing at all, an
untouched surface, not a wrong one, at every plane, stride, kernel and channel count
tried. Bits[15:8] of `CNA_PAD_CON0` (`0x1080`) decide it alone. Forcing `0x0100` into a
zero-pad program makes that same program write, and forcing `0x0000` into a working one
stops it. The `pad_top` field in bits[7:0] does neither.

This bound is why a TFLite-style SAME 3×3 stride-2 stem does not run and an ONNX-style
symmetric one does. TFLite puts the odd pad byte on the trailing edge and leaves
`pad_left` at 0.

**The output width must be `iw/stride`.** Any other width writes a full, correctly sized
surface that is sheared. The tap a weight byte lands on drifts one output column per
output row, exactly as a row-stride mismatch does. It shears for a narrow `ow` and an
over-padded wide one alike.

**The output width must also be a multiple of 16**, and `iw` being one does not imply
it. At `ow` 24 and 56, output row 0 is exact and every row after it is wrong. Both come
from an `iw` that is a multiple of 16. The widths `ow` 16, 32, 48, 64, 80, 96 and 112 are all
exact. Taken with the rule above, this means `iw` must be a multiple of `16*stride`. The
direct path carries no such rule, and this one comes with the channel fold.

**One image channel is programmed as two**, and the feature DMA's row width forces
that, not the mode word. Bits[31:16] of `0x1078` carry `line_stride - 1`, which is right
from `ic=2` up. At `ic=1` the DPU writes nothing at all, an untouched surface, not a
wrong one. Raising that field alone revives the write at every plane, but nothing is
exact, because the DMA then reads past the packed row. At `iw=64` (4 granules) the
field must reach 4, and at `iw=128` (8 granules) it must reach 8. The best exactness any
value reaches is about 40% of the surface [HW sweep, H96 MAX M9, measured 2026-07-28].

So the library widens the row instead of the register, with a second interleaved
channel of zero samples against zero weights. Then `ic=1` is bit-exact at 64×64,
128×128 and 224×224, with the same envelope as `ic=2`. It refuses `k=1`, `k=5` and
`oc=16` exactly where two channels do [HW sweep, H96 MAX M9, measured 2026-07-28]. The
pad leaves the arithmetic untouched. The MAC term is zero because the weight is, and
`sum_w` is unchanged, so the coefficient `A` is too. The asymmetric `B` multiplies a sum
of raw samples that gains only zeros, so the zero-point fold keeps the caller's tap
count.

The cost is one byte per pixel of host packing and a doubled feature read. The mode
word's `ARGB_IN` nibble is `8 | (ic-1)`, so `ic=1` is also the one value leaving its low
bits clear. But forcing that nibble to any of the working values leaves the surface
untouched, so the nibble does not gate the write. The test `tests/rk3576_argb_ic1.c`
holds this result, with the register sweep that found it.
[HW sweep, H96 MAX M9, measured 2026-07-28]

The output-channel count follows the direct path's multiple-of-32 rule (`oc=16` writes
nothing). It does not follow the float first conv's 32-channel per-program cap: one
int8 program delivers 64 output channels. Above that the caller splits, as on the float
path.

`rocket_conv2d_int8_rk3576()` owns all of it:

- The packed image
- The cube
- The row window
- The output-channel split
- The zero-point fold
- The de-scatter

It is bit-exact against a CPU model at twenty shapes. Those include the 224×224
stride-2 stem at `oc` 32 and 64 and at `k` 3 and 7. The `fq` group of
`rk3576_conv_lib_gate` holds this, and also asserts all seven refusals in both
directions. [HW sweep, H96 MAX M9]

A capture is an oracle for a whole program, and running one verbatim separates a dead
mode from a wrong extrapolation. This library's emitter reproduced twelve int8 ARGB
captures register for register and still wrote nothing at every geometry tried. That
reads as a dead datapath.

The test `tests/rk3576_fc_vendor.c` takes the captured op stream out of the golden
table, patches only the five address registers and submits it. The vendor's own
program then writes its whole surface. So the mode works, and the extrapolation away
from the captured geometry is what fails. Every bound above comes from walking from
that geometry back toward a small one, one axis at a time.

The low field of `0x100C` is not precision-independent. The found int8 ARGB programs
carry `0x2000a006` and the float ones `0x2020a122`, so the field is 6 at int8 and 2 at
fp16. Bit 2 clears on the float path, while `GROUP_LINE_OFF` and `ARGB_IN` stay where
they are. The `ARGB_IN` field tracks the channel count as `8 | (ic-1)`: `0x8` at 1
channel, `0xA` at 3 and `0xB` at 4. That confirms the reading that Mesa's 1-channel
`0x8` is the same field.

### The fp16 first-conv program

The fp16 first conv is a different program, not the int8 one at another precision. Six
fields move with the precision, and the first two are the shape of the datapath rather
than a width. Manufactured captures separate every one of them, each varying one axis
against the rest. They vary `iw` at 16, 32 and 64, `ih` at 16, 32 and 48, both strides,
`k` 1/3/5/7 and `oc` 16/32/48/64. [source-confirmed, RKNN-Toolkit2 rk3576 float build]

| | int8 | fp16 |
|---|---|---|
| mode nibble (`0x100C` low) | 6 | 2 |
| programmed channels (`0x1028` low) | `4*kw`, the kernel width folded into the channel axis | `ic`, the image's own count |
| CBUF granules per row (`0x103C` hi) | `iw*4/64 + 1` | `iw*4*2/64`, no `+1` |
| `0x1044` low | the granule count | `iw` |
| `0x118C` | `(entries-1, entries-2)` | `(entries-1, entries-1)` |
| DDR row stride (`0x1090`) | `iw*ic/16` | `iw*ic*2/16` |
| CVT | runs: Q14 unity, the uint8 zero point as a per-channel offset | bypassed, offsets zero |

There is no channel fold at fp16. The int8 path folds the kernel's width into the
channel axis (4 lanes per pixel column, `kw` columns), so it programs 12 at `kw=3`
against a feature DMA of 3. The float path does not fold: the programmed count is the
image's, and the taps stay on the kernel axis. The `+1` granule goes with the fold,
because it is the fold's lookahead (the last output column of a row reads `kw` pixels,
which reach past its own granule). A float program at `k` 1, 3, 5 and 7 carries the
same entry count at every kernel size.

The decoded weight cube says the same thing independently. It carries explicit `kh`
and `kw` axes with four lanes inside a tap, where a folded cube would carry the columns
inside the lanes.

The input is an fp16 packed image: 3 or 4 interleaved halfwords per pixel, which is why
the row stride takes the element size. The CVT is bypassed rather than configured,
since a float image is already in the value domain the MAC wants. The per-channel
truncates and Q14 scales stay in `0x1048` under the bypass, where they are inert.

Read correctly, the weight-byte registers are the same two on both paths. Register
`0x1020` is the bytes one output channel occupies, the element count scaled by the
element size. The high half of `0x1030` is twice the raw element count. The two
coincide at fp16 and differ by the factor of two at int8, which is why one can be
mistaken for the other.

The element count per output channel is `ic*kh*kw` on the direct path, `4*kh*kw` on the
float first conv (dense, no round-up of the lane group to 16) and `kh*round16(4*kw)` on
the int8 one. Depthwise is its own case and must not be scaled. A depthwise weight
occupies a 16-bit slot whatever the precision, so its count is already the float one.

### Output channels per first-conv program

One first-conv program delivers **32 output channels and no more**. At `oc` 48, 64 and
96, exactly 32 whole channels come back bit-exact, and the rest of the surface is never
written. The written part is a contiguous prefix, not the interleave the int32 writer's
byte budget gives. At `oc` 24 and 32 the surface is complete. [HW sweep, H96 MAX M9]

This is not a register the emitter gets wrong: the program matches the vendor's own
`oc=48` and `oc=64` captures register for register. The weight cube reproduces those
captures too, so the vendor's compiler emits single programs the part does not fully
execute. The recourse is the one the direct path already uses for its weight slice:
split the output channels, one submit per tile of 32. Each tile is an independent
convolution over its own channels.

The entry `rocket_conv2d_fp16_rk3576()` does the split. The output cube's channel groups
are contiguous planes, so a tile is the same BO at a plane offset.

The row window composes with it on the other axis. The planner at a precision is
`rocket_rk3576_plan_rows_prec()`. It changes the CBUF entry count per row and the row
cap the allowance affords. On this path it also changes the feature offsets. A float
packed image is `ic` interleaved halfwords per pixel, where an int8 one is bytes.

A 224×224 plane is 6272 granules against a 6144 ceiling and has no single-task plan.
With the window it is bit-exact at `k` 3 and 7 and at `oc` 32 and 64, the two splits
running together.

### The vendor's float BS arrangement

The vendor's float programs use a BS arrangement that this library does not. Three
DPU_RDMA words carry a different constant in every vendor float program. Two of them are
`0x5034` at `0x1` and `0x5044` at `0x40010050`, against the integer path's `0x41` and
`0x40000010`. This library reproduces both, and both are bit-exact on the part.

The third shows that register fidelity is not the goal. **`0x501C` (BRDMA_CFG), the BS
operand reader, reads `0x100` in every vendor float program, against the integer path's
`0x710`. Emitting the vendor's value makes the DPU write nothing at all**: an untouched
surface, no fault, nothing in dmesg. It does not act alone: paired with the BS ALU config
those same programs carry (`0x4044 = 2`, where the integer path uses 1), the writer runs
again. The arithmetic is then wrong on about a tenth of the surface
(1866/2048, worst relative error 0.016).
[HW sweep, H96 MAX M9, leave-one-out over the six float-path registers]

On its own, with `0x501C` at `0x710`, `0x4044 = 2` is bit-exact. So the vendor's float
epilogue is a different BS *arrangement*, not a different constant. This library packs
its A/B/C coefficient group for the integer one. Keep `0x710` until the arrangement is
decoded as a whole.
[HW sweep, H96 MAX M9, leave-one-out over the six float-path registers]

Two more float-path corrections are bit-exact on the part. CORE `0x3018` bit 0 is an int8
marker that every float program clears. DPU `0x4010`'s middle field (the DPU's input
width) stays zero on the float path, where only the output width and the processing
precision move.

### The vendor compiler's packed-path choice

The vendor compiler takes the packed path at 1, 3 and 4 channels, and not at 2. It
compiles a 2-channel float conv down the direct path, with no `GROUP_LINE_OFF` and no
`ARGB_IN`. That is a toolkit preference and not a hardware bound. The packed path
computes bit-exactly at two channels on the part (`rk3576_conv_lib_gate fc-2ch-k3`), so
the library takes it for every count at or below four. In a capture set, a channel count
with no ARGB capture is not evidence that the part refuses that count.

## CBUF row reuse across a windowed sequence

The emitter's own windows refetch: each task fetches its whole window against a CBUF
base of zero, which is self-consistent and bit-exact on the part. The vendor's programs
keep rows resident instead, and twelve captured continuation tasks pin the scheme
exactly.

Consecutive windows share `kh - stride_y` input rows. The vendor keeps those resident
and moves the CBUF write pointer instead of refetching them. Seven registers carry it:

| Register | On a continuation task |
|---|---|
| `0x1028` hi | `entries * (ih - retained)`, the granules this task fills, not its window |
| `0x1078` lo | `ih - retained - 1` |
| `0x1098` | `iw * (ih - retained)` |
| `0x103C` lo | CBUF granule where the task's window begins, where the retained rows sit |
| `0x1040` lo | CBUF granule where fetching resumes = the window base plus the retained rows |
| `0x1018` bit 31 | set when the task retains rows |
| `0x1038` bit 31 | set on every task past the first, retaining or not |

The feature address (`0x1088`) also points at the first new row rather than at the
window start. The vendor's `dw` capture is the clean case. Its second task reads input
rows 89..111 of a 112-row plane, retains 2, and carries `0x1088` = row 91.

`retained` is the window arithmetic, the previous window's end minus this window's
start, which at these geometries equals `kh - stride_y`. When it is zero, the vendor
resets the base to the sequence's origin instead of continuing it, which is what a k=1
continuation carries.

`rocket_rk3576_plan_rows()` fills `retained` and `cbuf_resident` on every task, and
`gen_conv2d_int8_rk3576_reuse()` emits the reuse form. The reuse is opt-in. It saves
`entries * retained` granules of feature DMA per continuation task, under a tenth of the
window's traffic at the captured geometries. A window costs ~1.2 ms, and that cost is
the driver's completion poll, not the fetch. So the reuse does not shorten a windowed
conv measurably.

What the reuse buys is that the vendor's split programs become a complete oracle. Seven
registers that the gate's diff cannot otherwise reach are checked exactly across
thirteen continuation tasks. The tasks span three kernel sizes, both strides, both
channel counts and all three CBUF origins.

**The reuse path is unvalidated on silicon.** A wrong base reads resident rows that are
not there and corrupts silently, so the refetching path stays the default.

Three-task sequences are not pinned. Every captured split is two tasks. Accumulation of
`cbuf_resident` over a longer sequence is the natural extension of the same arithmetic
and nothing more. [expected]

## The same graph, compiled three times

Each `.rknn` capture holds its graph compiled three times, and the three compiles never
differ in geometry. Two of them are the same per-layer task split at different CBUF
origins, 0 and 7168 granules. The origin is added to every base in the table above. The
third splits the whole chain instead of each layer. It runs all five layers over the top
half of the image, then all five over the bottom half, from a CBUF origin of 6144. It is
the one that sets `0x1014` bit 28 and drops the early duplicate `0x1038` preamble write
(138 register writes rather than 139).

One reading fits: the third is the two-core compile. Under that reading, 7168 granules
are one core's whole CBUF pool, and an image-plane split is the standard way to use two
cores. The fit is not evidence. Bit 28 of `0x1014` sits where the RK3588 puts
`NN_MODE`, which is consistent with a mode bit and says nothing about which mode. All of
it is decodable off-device, and none of it is decided by these captures.

None of this matters to an emitter. The origins, the format low bits (`0x1018`,
`0x1038`) and the allowance field are the vendor allocator's per-compile choices. A
hardware sweep found only `0x1018` bit 30 and `0x1038` bit 31 live on the int8 path.

## The DPU epilogue: what an appended stage moves

Three captures are the same conv as `conv2d_rk3576.rknn` with one extra op appended, one
each of a bias add, a scale and an eltwise sum. That makes them a controlled experiment
run by the vendor's own compiler. The delta against the plain conv is three registers,
and it is the same three, with the same values, for all three ops:

| Register | plain conv | conv + an epilogue stage |
|---|---|---|
| `0x4044` BS ALU operand | `1` | `0` |
| `0x4050` BS config | `0x80011111` | `0x80021111` (bits[19:16] 1 -> 2) |
| `0x501C` BRDMA config | `0x710` | `0x114` |

The finding is that the three ops are indistinguishable here. These registers do not
encode which epilogue op runs. A per-output-channel bias, a per-channel scale and a
per-channel sum all lower onto one BS-stage configuration. What separates them is the
coefficient buffer's contents and the requant triple (`0x40AC`-`0x40B4`), which is the
only other thing that differs between the three captures.

Neither `0x4050` nor `0x501C` decodes under the RK3588's field map at these offsets. The
value `0x80011111` sets bit 12, which is reserved in the RK3588's `BS_CFG`. The RK3576's
`BRDMA_CFG` is wider than the RK3588's 5-bit field. So this note carries both as
transcribed constants rather than decomposed fields. The reading that fits is that the
BS stage takes its second operand from `DPU_RDMA` rather than from the inline path. That
is the same stage and the same buffer as the asymmetric-weight `B` term.

### The BS operand source

Driven one register at a time, the bundle separates cleanly, and `0x501C` moves the BS
operand source. The drive runs against a conv carrying a known non-zero `B`
(`rk3576_conv_gate` with `ROCKET_G_WZP=40`, which reports how many outputs each
candidate sign convention explains). [HW sweep, H96 MAX M9]

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

`0x501C` is the operand-source register, and the other two are inert without it. On the
plain path, `0x4044` and `0x4050` change nothing at all. Not one output of a
32768-element surface moves, down to the incidental 537 that the no-B model always
coincides on. Driven alone, **`0x501C` stops the DPU writing**: "task never wrote its
rows" over four retries, no fault, nothing in dmesg.

The register needs a matching BS config to write again, and the matching config depends
on the value. The epilogue's `0x114` is re-enabled by `0x4050` bits[19:16] = 2, and the
float path's `0x100` by `0x4044` = 2.

Under the epilogue arrangement, the BS stage does not read the coefficient group's `B`.
The count that `+B*sum(x)` explains falls from every output to 1423. Exactly one output
in four matches the model with no B term at all. The counts are 8192/32768, 7200/28800
and 512/2048 on three different shapes, a quarter each time.

So these registers do not select an
epilogue *op*, as the three captures showed. They select where the BS stage takes its
second operand from. The quarter is what a buffer packed for the inline positions looks
like when read as the other kind of stream.

The float path's refusal to write is the same mechanism, and it makes a third BS
arrangement visible:

| arrangement | `0x4044` | `0x4050` | `0x501C` |
|---|---|---|---|
| plain int8 conv | 1 | `0x80011111` | `0x710` |
| int8 conv + an epilogue stage | 0 | `0x80021111` | `0x114` |
| vendor float conv | 2 | `0x00021111` | `0x100` |

This library packs its A/B/C group for the first, which is bit-exact at both precisions.
Decoding the other two would say what the vendor's epilogue and float paths buy. Neither
is needed to compute.

## Depthwise convolution

`gen_conv2d_dw_int8_rk3576()` computes bit-exactly. Its register program reproduces the
vendor across 88 captured task programs with no field left open. That is not sufficient,
because two of the path's inputs are buffers rather than registers. A `.rknn` capture
carries a register program. The coefficient group and the weight cube are memory the
registers point at, and a capture says nothing about either. Both differ from the direct
path's and both are silent when wrong, so both must be read off the part.

### The envelope

The depthwise path is bit-exact against a CPU model over the whole surface, with every
shape at `maxdiff 0`. The gate is `rk3576_conv_gate dw` at 35 shapes, gap-free with no
cold-start-wall retries. [HW sweep, H96 MAX M9]

| axis | covered |
|---|---|
| channels | 8, 16, 24, 32, 48, 64, 72, 80, 96, 112, 128, 144, 176, 256 |
| kernel | 1, 3, 5, 7 |
| stride | 1 and 2 |
| padding | SAME and VALID |
| plane | 16×16 to 112×112, plus 15×18, 17×19 and 19×19 |
| row window | 1, 2, 3 and 6 tasks |

A depthwise task costs about 1.1-1.2 ms of NPU wall, whatever the channel count, the
kernel or the plane. That is the same per-submit dispatch floor the direct path pays,
with no compute term visible at these shapes. A row window costs that per task.

The awkward numbers in that table are its purpose. Channel counts that are not multiples
of 32 are where this path's granules stop agreeing:

- The weight cube rounds to 16.
- The CBUF allocation takes a 16-group count of 3 mod 4 one group further.
- The 2-bit group field in `0x4050` wraps.

A gate that rounds every depthwise count up to 32 exercises none of that. Planes whose
`ow*oh` is not a multiple of four are where the padded output surface stride below
shows. Channel counts in multiples of 32 and planes in multiples of four hide both.

### The depthwise coefficient group

The depthwise coefficient group covers 8 output channels in 48 bytes and carries no `B`.
The direct path's group is 64 bytes for the same 8 channels: `A` (int32 bias) at
`(oc%8)*4`, `B` (weight zero point) at `+32`, `C` (int16 multiplier) at `+48`. The
depthwise layout is:

```
A[oc]  int32  at (oc%8)*4
C[oc]  int16  at 32 + (oc%8)*2
```

Channel `oc` sits in group `oc/8`, and the group stride is 48.
`rocket_rk3576_pack_coeff_dw()` and `rocket_rk3576_coeff_bytes_dw()` implement it.
[HW sweep, H96 MAX M9]

Handed the 64-byte group, the depthwise path reads `A` and `C` at indices that drift
apart at different rates. There, `A` strides 4 bytes per channel and `C` strides 2. Past the
first eight channels, each is read out of a different group. The BS stage computes
`(acc + A) * C`, so the surface comes back as follows:

- Most channels multiply a bias by a zero and never reach DDR at all.
- A few channels, whose `C` lands in the `A` region, square their own bias.
- One group's `C` multipliers are read as the next group's biases.

The result is a correctly sized, mostly empty surface with no fault to catch it. That is
the same signature a wrong geometry register gives, so it reads as a datapath defect
rather than a packing one.

### The B term is added: `acc + B*sum(x)`

Every vendor capture carries `B = 0`, so a capture shows the field's position and width
but not its sign convention. Driving `B` on the direct path against a CPU model settles
it. The candidate models use `sum(x)`, the sum of the input elements the output
contracted. The model `acc + B*sum(x)` explains every output of every direct shape in
the conv gate's envelope group. It does so at `B` = 1, 40, 127, -37 and -128.
[HW sweep, H96 MAX M9, measured 2026-07-27]

The models `-B*sum(x)` and an inert `B` explain only the few percent that coincide. That
share is larger at small `|B|`, because the correction often rounds to the same int8.
The variable `ROCKET_G_WZP` in `tests/rk3576_conv_gate.c` drives it and scores all three
models. [HW sweep, H96 MAX M9, measured 2026-07-27]

**A weight zero point is programmed negated.** An asymmetric weight is
`w_true = w_stored - wzp`, and its correction is `-wzp*sum(x)`. So pass `B = -wzp` to
`rocket_rk3576_pack_coeff_asym()`, whose parameter is named `b_term` for exactly that
reason. A reversed sign gives a plausible surface with a bias-shaped error and nothing
that faults.

The depthwise group has no slot for a weight zero point, so **an asymmetric depthwise
weight must be folded into the bias**. There is no `_asym` form for it.

### The weight cube

The depthwise weight cube has two layouts, one per precision, which share a block
structure and nothing else.

The float cube is decoded, not inferred. Its captures carry a unique weight value per
(channel, tap), at C = 24, 32, 48, 64 and 128 and at k = 3 and 5:

```
slot(c, kh, kw) = (c/32)*32*KH*KW + (kh*KW + kw)*held + (c%32)
held            = min(32, C - (c/32)*32)
```

The layout has four properties:

- A weight occupies a 16-bit slot. That is why one geometry's float and int8 cubes are
  the same size. It is also why `WEIGHT_BYTES` carries a factor of 2 that is not the
  element size.
- Channels group by 32, each group a contiguous block.
- Inside a group the order is tap-major, kh outer.
- A trailing partial group is dense. Its tap stride is the channels it holds, not 32,
  so C=48 is 432 slots and not 576.

`rocket_rk3576_weight_dw()` implements this layout, and `dw/named/mknamed.py --decode`
re-derives it from the committed captures without a board. The naming must be carried
in float weights. Per-channel weight quantization normalizes each channel by its own
maximum, which erases any naming carried in magnitude. The fp16 format represents
integers exactly to 2048, so a unique-integer ramp survives the float path untouched.
The ramp is findable in the file by its exact value set.

The int8 cube is **not the float cube's low byte**, and it is not the RK3588's
group-of-64 single-byte packing either. The probe mode `rk3576_conv_gate dwmap` reads it
off the part. That mode drives an impulse feature against a cube that is zero but for
one byte, and reports which output that byte reaches. Over a C=32 k=3 cube, every one of
the 576 bytes reaches exactly one (channel, tap): [HW sweep, H96 MAX M9]

```
byte(c, kh, kw) = (c/64)*64*KH*KW*2 + (kh*KW + kw)*held*2 + 4*((c%64)/2) + (c%2)
held            = min(64, round16(C - (c/64)*64))
```

Against the float cube, the int8 cube differs in three ways:

- The channel group is 64, where the float cube's is 32.
- Inside a tap block, the channel a byte carries is `2*(b/4) + (b%2)`. So channel `c`
  owns two bytes per tap, at `4*(c/2) + (c%2)` and two further on. **Both bytes are live
  and both contribute.** A weight written into both is added twice, so the packer writes
  the first and leaves the second at zero.
- A trailing partial group strides by what it holds rounded up to 16, where the float
  cube's is dense in the raw count. That makes the whole cube exactly
  `round16(C)*KH*KW*2` bytes.

`rocket_rk3576_weight_dw_int8()` implements this layout. It returns a byte offset where
the float entry point returns a slot. **An int8 weight written as a 16-bit value into the
float slot puts its sign extension into the byte that belongs to the next channel.**
That is a silent -1 weight on a neighbor, not padding.

Two shapes hide the group size, and they are the ones that pass first. At a single group
the two layouts coincide. At `k=1` there is one tap, so the group base and the tap
stride cannot be told apart. **C=64 k=3 separates 64 from 32.** At that shape, `dwmap`
predicts 576 of 576 (channel, tap) positions at 64 and 64 of 576 at 32. C=24 and C=72
separate the round16 partial group from the raw one.

### The depthwise output surface stride

The depthwise path pads the output surface stride to four elements. Register `0x401C` is
the distance in elements from one 16-channel output group to the next. The depthwise
path programs `round4(ow*oh_full)` there, where the direct path programs `ow*oh_full`.

It is a DDR stride, so **a caller must size the output BO and de-scatter with it**,
through `rocket_rk3576_out_surf_elems(ow, oh_full, dw)`. A de-scatter that assumes the
plane lands every group past the first up to four elements early. That reads as "the
first 16 channels are exact and the rest are noise": at C=32 exactly half the surface,
at C=64 exactly a quarter.

The padding is invisible at any plane whose `ow*oh_full` is already a multiple of four,
which is most planes. Planes of 16×16 and 112×112 pass, and 15×18, 17×19 and 19×19 do
not. The plane term in `0x40B8` is four of these same rounded surfaces.

### The depthwise probe modes

`rk3576_conv_gate` carries five modes. Each has a `direct` control, which runs the same
reading on the path known to be bit-exact. The control is load-bearing: a mode that
cannot reproduce the direct map is not measuring what it claims to.

| mode | what it reads |
|---|---|
| `dwmap` | which output each weight byte reaches, one submit per byte |
| `dwbias` | which coefficient slot each channel read |
| `dwcoeff` | the inverse: which output bytes each coefficient position moves |
| `dwout` | the raw output BO, undescattered |
| `-l` | the shape table, without running it |

Two properties make these modes decode rather than score, and any successor needs both:

- **`dwcoeff` measures its own baseline.** A comparison of each position against the
  packed constant is a delta only where the part agrees that the surface is flat. On a
  path whose surface is not flat, every position reads as "changed" and the probe says
  nothing. The mode keeps one unprobed submit as a reference image, which makes a
  per-position delta meaningful whatever the baseline looks like. There is no need to
  flatten the hardware first.
- **`dwcoeff` and `dwout` read raw bytes.** Scoring a depthwise surface through the
  direct path's de-scatter hides exactly the permutation these modes are looking for.

`ROCKET_G_DWOUT_OUTSCALE` divides the requant down, so a lane at the int8 clip names its
magnitude instead of reading as "large". The variables `ROCKET_G_DWOUT_BASE`, `_STEP`,
`_CBASE` and `_CSTEP` drive the `A` and `C` ramps independently. The two ramps separate
"this lane read the wrong bias" from "this lane read its multiplier out of the bias
region". With `A` constant at 5 and `C` at 1, a lane reading a real `C` answers 5. A
lane reading `A` as its own `C` answers 25.

### Manufactured depthwise captures

Captures built to order separate the axes that the found captures confound, and they
close four depthwise formulas. The manufactured set covers 22 channel counts from 8 to
256, strides 1 and 2, kernels 1, 3, 5 and 7, and rectangular planes.
[source-confirmed, RKNN-Toolkit2 rk3576 depthwise builds]

The path has two channel granules, and they are different numbers:

- The weight granule is `C` rounded up to 16. `WEIGHT_BYTES` (`0x101C`) is
  `round16(C)*kh*kw*2`, and `WEIGHT_ELEMS` (`0x1020`) is `round16(C)*kh*kw`. The
  weight-bytes-per-kernel term (`0x1030` high) is `kh*kw*round16(C)/8`, confirmed at
  k=1, 3, 5 and 7. The found captures are all k=3 and cannot tell `kh*kw*C/8` from
  `kh*kw*4`.
- The feature granule is `C` rounded up to 16. If the resulting 16-channel group count
  is 3 mod 4, the granule rounds up one group further. So 48 rounds to 64 and 112 to 128,
  while 80 and 144 stay put. It sizes the CBUF entry counts (`0x1028` high, `0x103C`
  high, `0x1044` low). Read as hardware, channels are fetched in blocks of 64, and a
  trailing partial block holds one, two or four sixteens, never three.

The depthwise field in `0x4050` tracks the channel count, not the stride. Bits[9:8] are
the 16-channel group count minus one, modulo 4, computed on the *unrounded* count. So
C=48 carries 2 there, while its feature allocation is 64's. The found captures cannot
attribute the field, because every C=32 program in them is stride 1 and every C=64 one
is stride 2. Manufactured captures at both strides show it flat in the stride and
stepping with C, wrapping at 80, 144 and 208.

An emitter that writes the C=32 word unconditionally hands the BS stage, above C=32, a
word the vendor never uses. **That returns a wholly untouched output BO**: no surface at
all, no fault, no dmesg line.

The direct path's `0x4050` is the even-count word at every count this emitter programs.
The emitter programs a direct conv's output channels as a whole number of 32-channel
groups (`rocket_rk3576_pad_oc`), because a partial group computes wrong. So the
programmed 16-channel atom count is always even. The `gahingwoo/mesa-rk3576` fork clears
bit 8 only when `DIV_ROUND_UP(oc, 16)` is odd, and it programs `oc` verbatim to get
there. At every count programmed here, its rule gives this emitter's `0x80011111`.

Real `oc` 8-112, padded, are exact through the emitter and the tiling entry [HW sweep, H96
MAX M9, rocket 1.6.0, 2026-09-30]. Written alone into those even-count programs, the
odd-count word `0x80011011` computes 19 of 21 shapes wrong, most at the backstop,
consistent with that rule. Two are exact, `oc` 112 at stride 2 and `oc` 48 windowed, and
the rule does not explain them. Whether the odd-count word lets an unpadded partial group
compute is not measured.

The plane term in `0x40B8` is a whole destination surface, and its multiplier is flat in
the kernel size. The register is `4*surface - ow*oh_task` on the depthwise path and
`2*surface - ow*oh_task` on the direct one, where `surface` is `0x401C`. A capture set
that is all k=3 cannot tell the 4 from `kh+1`. At k=1, 5 and 7 it does not move.

Register `0x401C` rounds up to four elements on the depthwise path and not on the direct
one. The rounding is invisible at every plane whose `ow*oh_full` is already a multiple
of four. That covers every capture on both paths and every hardware sweep the formula
was fitted against. Rectangular and odd depthwise planes show it: the vendor programs
272, 324 and 364 for `ow*oh_full` of 270, 323 and 361.
[source-confirmed, HW sweep, H96 MAX M9]

The direct path's answer is a hardware result rather than a transcription, because no
direct capture has a plane that separates it. A VALID-padded oc=128 conv at
`ow*oh_full = 105` is bit-exact with 105 programmed and wrong with 108. Carrying the
depthwise rounding onto the direct path costs that shape and nothing else in the
67-shape gate. One shape in the whole envelope is sensitive to it.
[source-confirmed, HW sweep, H96 MAX M9]

Register `0x118C` is `iw-1` in both halves. On a square plane the low half also reads as
the full plane height, so square captures cannot tell the two apart. All 142 non-ARGB
programs carry `iw-1` twice, with no exception. The fact is not depthwise-specific. It
holds on the direct path too, where square planes hide it for the same reason.

The low half of `0x1024` is not a depthwise channel count. It is the same raw `1` at
every count from 8 to 256. An independent on-hardware sweep of the field reaches the
same conclusion from the other direction.

## fp16: the datapath, and the contraction width that bounds it

An fp16 direct convolution computes bit-faithfully on this part and delivers every
output channel. The library drives it with `gen_conv2d_fp16_rk3576()` for one task and
`rocket_rk3576_plan_ic()` for an arbitrary input-channel count. **One bound remains: the
contraction is sixteen input channels wide.** The split is the way around it.

Two apparent gaps are one fact seen from two sides. One is an output writer that spends
four bytes on a two-byte element and reaches only `oc/2` channels. The other is a
feature surface index that caps a task at 8 input channels. The DPU's output element
stride is `16/ic` words. At `ic = 8` that is two words per element, which gives the
"four bytes on a two-byte element" reading. At `ic = 16` it is exactly one word, and the
surface is the plain native cube.

Everything in the rest of this section is
[source-confirmed, RKNN-Toolkit2 rk3576 float build, HW sweep, H96 MAX M9, measured 2026-07-26].

### Capturing the vendor float program

This library's float program is a transcription of the vendor's. Every `.rknn` in the
vendor-capture set is int8, and a zero precision field is invisible. It says nothing
about where the field is, how wide it is, or whether this part carries one there at
all. An int8 capture therefore cannot check a float field inferred from the RK3588's
packing, and one such inference is wrong for this part.

RKNN-Toolkit2 emits a float program for this part. With `do_quantization=False` on a
float ONNX, the toolkit leaves the weights float and the vendor compiler picks the float
datapath. The script `tests/data/rk3576-vendor-capture/float/mkfloat.py` rebuilds the
captures and carries the dependency pins. The pins are particular. The toolkit imports
`onnx.mapping`, which onnx 1.16 removed, and `pkg_resources`, which setuptools 81
removed, and neither failure names itself.

Two things make the resulting diff small enough to read:

- **Only seven registers move with `ic`** across the whole 139-word program: `0x101C`,
  `0x1020`, `0x1028`, `0x1030`, `0x103C`, `0x1044` and `0x107C`. All seven scale
  linearly, and the fp16 program emitted from the RK3588 encoding matches six of them.
- **Fourteen registers differ from the fp16 program emitted from the RK3588 encoding**,
  and they are the same fourteen at every geometry. Three of them are load-bearing. Each
  of the other eleven can be dropped with the datapath still exact.

### The float-mode registers

Three registers carry the float mode. Each is constant across every geometry the vendor
emits (`ic` 8-64, `oc` 32/64, `k` 1/3, planes 16 and 32). Each is load-bearing on its
own. With any one of them at its integer value, the contraction reads every feature
surface twice and skips the odd ones. **A single-register sweep cannot find this**,
because the fix is a register set.

| register | integer value | float value | what is load-bearing |
|---|---|---|---|
| CNA `0x100C` | `0x00000000` | `0x00200120` | all of `bit21 \| proc<<7 \| in<<4` |
| DPU `0x4038` | `0x00120080` | `0x00100092` | the low half, `0x0092` |
| DPU `0x4050` | `0x80011111` | `0x00021111` | bit 31 clear |

Each register has a detail of its own:

- **CNA `0x100C` bit 21 is a float enable** that the integer programs never set. The two
  precision fields do not imply it. With both precisions right and bit 21 clear, the
  contraction still doubles, exactly as it does with bit 21 set and a precision wrong.
- **DPU `0x4038`'s high half is free, but it must be non-zero**, or the DPU writes
  nothing at all. Both `0x0012` and `0x0010` compute.
- **DPU `0x4050`'s low nibbles do not enter here.** The values `0x00011111`,
  `0x00021111` and `0x00020000` all compute. The value `0x80021111` returns `+inf`,
  because the accumulator has summed the doubled reads. The nibble is the write-extent
  field described below, and bit 31 is the float bit.

### PROC_PRECISION

PROC_PRECISION is the operand width. **CNA `0x100C`'s `proc_precision` field carries
fp16 (`2`), not fp32.** The RK3588's field semantics suggest that the field names the
multiply-accumulate datapath's width, so that a float convolution accumulating in fp32
wants `5` there. That reading is wrong for this part: programming `5` with every other
float field right reads each feature surface twice.

A probe that is uniform in the channel axis cannot catch this, because there a doubled
read is indistinguishable from a scale. See Traps.

CORE `0x3018` does not follow the CNA word, and the two divide the work:

- **CNA `0x100C`'s `in_precision` pins the operand width class, not its type.** Values
  `1`, `2` and `3`, the three 2-byte codes, all compute. Value `0` (a 1-byte element)
  writes an entirely zero surface. So an fp16 and a bf16 program are indistinguishable
  in this field.
- **CORE `0x3018` pins the operand type.** Value `2` is fp16 and `3` is bf16, and each
  returns a wrong surface on the other's bytes. Fed bf16 bytes, `3` contracts
  bit-exactly in the matmul-form program, the case that
  [rk3576.md](rk3576.md) §"The precision fork" records.

### The float narrowing and the output width

The output is fp16 only with the float narrowing enabled. The epilogue is a float path,
and its result is an fp32 word. Two fields decide what reaches DDR:

- **DPU `0x40B0` bit 16 is `fp32tofp16_en`**, at the RK3588's own bit position. Without
  it, the DPU writes the fp32 word as-is.
- **DPU `0x4010` bits [31:29] select a width, not a conversion.** Value `5` writes the
  whole 32-bit word, and `2` and `3` write its low and high halves.

The two compose as follows:

| narrowing | width | what reaches DDR |
|---|---|---|
| on | `2` | true fp16 (`0x5400` for 64.0) |
| on | `3` | the top half of the fp32 word, which is bfloat16 |
| off | `5` | fp32 |
| off | `2` | the fp32 word's low mantissa bits, **zero for every value a small integer test pattern can produce** |

**An fp16 program emitted from the RK3588 encoding programs the last combination.** It
writes a full, correctly sized, entirely zero surface from a datapath that is running,
which reads as a dead MAC.

### The direct-path float weight cube

**The float weight cube groups both channel axes by 16**, where the int8 weight cube
groups both by 32 (`weight_conv_int8`). Otherwise the reorder is Mesa's
`oc1, ic1, kh, kw, oc2, ic2`, and `rocket_rk3576_weight_conv_fp16()` is the index. The
feature cube is C2 = 8 fp16 lanes in the same 16-byte atom. A bijective decode pins it,
not a score (C2 = 4 and 16 both fail).

Each group needs a shape chosen so that it is observable at all, and the output group is
the harder of the two:

- **The ic group is 16.** It is exact where 4, 8 and 32 are not.
- **The oc group is 16.** It is exact at `ic = 16` with `k` 3 and 5. **It cannot be
  pinned at `k = 1`**, because the kernel index sits between the group and the
  `(oc2, ic2)` pair. At `k = 1` the index collapses to `(oc/G)*G + oc%G = oc`, and every
  group size is byte-identical. A sweep run only at `k = 1` reports all four candidates
  as correct.

This is the concrete form of the general trap below: a reading of 8 here comes from a
shape where the field is algebraically dead.

A kernel that ramps over `ic` does not settle the ic group, and neither does any score.
That probe sums over the channel axis, so it is invariant under every permutation of it.
Use realistic weights at a shape where the group is live.

### The fp16 output map

The fp16 output map is the plain native cube. At the 16-channel contraction the writer
has no defect: 8 channels to a 16-byte atom, one atom per pixel, and channel groups as
contiguous planes. Every programmed channel is present exactly once. For output channel
`c` at pixel `p`, in 16-bit words:

```
word = (ow*oh*8)*(c/8) + 8*p + (c%8)
```

The map is measured, not fitted. `rk3576_fp16_sweep map` drives a probe whose every
output lane carries a unique name, and decodes the surface. The decode is a bijection
onto the whole surface. Every word names the (channel, pixel) this map predicts, with
none left over and none undecodable.

In the library, `rocket_rk3576_fp16_out_index()` is that map. Its companion
`rocket_rk3576_fp16_accumulate()` de-scatters one surface through it into a row-major
fp32 accumulator, which is what an `ic` split needs. The host-side `regcmd_rk3576_gate`
checks the bijection, so an off-by-one in it does not wait for a board.

The map is what changes when `ic` does, which is the single clearest reading of the
bound. At the same `oc = 16` and a 4×4 plane:

| `ic` | slot of channel `c` within its atom | channels reaching DDR |
|---|---|---|
| 8 | `2*(c%4)`, odd words zero | `oc/2` |
| 16 | `c%8` | **all of them** |
| 32 | `(c/2)%8`, odd channels lost | `oc/2` |

### The contraction width

The contraction is sixteen input channels wide, and `gen_conv2d_fp16_rk3576()`
**refuses** any other `ic` rather than warning about it. A wrong count writes a full,
correctly sized, wrong surface with nothing to fault on. So a caller must not be able to
reach it silently. The unchecked bring-up entry beside it is `gen_conv2d_rk3576_prec()`,
which the sweep's own modes drive.

Probe 10 reads the multiplicity straight off the part. One input channel carries 1.0,
every weight lane is 1.0, and the output is that channel's read count. The variable
`ROCKET_FS_TAP` moves the tapped channel. At `ic = 32`, the pair `TAP=0` and `TAP=16`
separates the two failure modes in one run. Without the float fields they read 2.0 and
0.0, and with them both read 1.0.

With the fields right, the read count is 1.0 at `ic` 16, 32, 64 and 128 and at every
tap, so the reads are unbounded. What bounds a task is the writer's element stride
above.

### The fp16 envelope

A single fp16 task is bit-faithful against the CPU model (`worst relative error 0`) at
`ic = 16`, and every programmed output channel lands. The envelope covers:

| axis | covered |
|---|---|
| `k` | 1, 3 and 5 |
| plane | 8×8 through 56×56 |
| stride | 1 and 2 |
| `oc` | 16 through 64 |

### An arbitrary `ic` via a 16-channel split

Splitting `ic` into slices of 16 and summing the partial surfaces on the host
reproduces the model exactly at any input channel count. It is bit-exact,
`worst relative error 0`, at `ic` 16, 32, 64 and 128 crossed with `k` 1, 3 and 5.

`rocket_rk3576_plan_ic()` is the split. It sits beside `rocket_rk3576_plan_rows()` and
follows the same pattern: it lays the slices out, and the caller emits one conv per
entry. The slice is fixed at 16 rather than as wide as the CBUF allows. A wider slice is
a wrong task, not a cheaper one. It plans the `ic` axis only. A plane whose
16-channel slice still overflows the CBUF is refused, with a pointer at the row
planner, rather than run.

The slices cost nothing to address. At C2 = 8 the feature cube's channel groups are
contiguous planes of `iw*ih_full` 16-byte atoms. Slice `k` is therefore the same BO at a
base offset. An atom stays 16 bytes when C2 halves, so the stride is independent of the
element size.

Only the weight cube is rebuilt per slice. Each slice is its own convolution with its
own group count, so a slice's cube is not a sub-cube of the whole conv's
(`rocket_rk3576_fp16_pack_slice_weights()`). Size that cube by the groups.

The bias belongs to one slice only. Every slice adds the whole `A` term, so a
coefficient buffer carrying a bias handed to all of them lands it `ic/16` times.

On-chip accumulation is unblocked and small. The DPU eltwise stage does this job on the
RK3588 (`ROCKET_KACC`), and here it would remove `ic/16` readbacks. Nothing stands in its
way. At this contraction width, the partial it would read back is a plain, dense,
complete fp16 cube, not a surface carrying the writer's defect.

On-chip accumulation is not the largest win on this path. It is worth 7-13% of the wall
[HW sweep, H96 MAX M9, measured 2026-07-28, `oc` 32, 28×28 k3, `ic` 16-128,
`tests/rk3576_fp16_split_cost.c` with `ROCKET_RK3576_FP16_PROF=1`]:

| phase | share of the entry's wall |
|---|---|
| submit | 77-87% |
| readback (what on-chip accumulation removes) | 7-13% |
| weight repack | 4-7% |
| output stamp | 2-4% |

The sixteen-channel contraction sets the slice count, whatever sums the partials. So
the submits stay and only the readback goes. The EW operand DMA that the DPU would issue
instead is not free, so the net saving is below that band. Read the table as a bound.

A slice costs the poll floor plus `dpu_grace_us`, in full. The same sweep records zero
poisoning retries at every shape, so a slice's ~1.4 ms is not the hazard being cleared.
A wide-output task raises no DPU completion, so `0012` retires it on `PC_DONE` plus the
whole blind settle. Halving the grace moves the wall by very nearly the grace, once per
slice [HW sweep, `ic` 128, 8 slices, five runs each, ranges disjoint]:

| `dpu_grace_us` | 8-slice wall | per slice |
|---|---|---|
| 500 (default) | 15.39-16.16 ms, median 15.89 | ~1.99 ms |
| 250 | 12.95-13.97 ms, median 13.29 | ~1.66 ms |

That is 300 us of a 250 us cut landing per slice, and the convolution stays bit-exact at
every value tried down to 150.

The default cannot collect that lever. The same number is the deadline on the
DPU-completion wait for every job that does raise one. A narrow-output matmul fails 10
runs of 10 at 200 us (see the grace table).

An fp16 conv pays the grace in full only because its tasks raise no DPU completion at
all. What would collect the 1.18x is a way to say which class a task is in. Userspace
knows the class when it emits the program, and the driver cannot see it at `PC_DONE`
time. Lowering the shared default trades the fp16 path's wall time against the
correctness of every other path.

### Cost of an fp16 conv

The fp16 convolution runs as the `ic/16` split. The table prices it at the submits the
split spends [HW sweep, H96 MAX M9, measured 2026-07-26, `oc` 64, third of three runs].

| shape | submits | NPU | host pack + de-scatter | total |
|---|---|---|---|---|
| 32×32 ic16 k3 | 1 | 1.51 ms | 2.52 ms | 4.04 ms |
| 32×32 ic64 k3 | 4 | 5.57 ms | 3.36 ms | 8.92 ms |
| 56×56 ic64 k1 | 4 | 6.00 ms | 24.20 ms | 30.20 ms |
| 56×56 ic64 k3 | 4 | 6.21 ms | 8.62 ms | 14.83 ms |
| 56×56 ic128 k3 | 8 | 11.93 ms | 15.93 ms | 27.86 ms |

The NPU wall is flat in the work and linear in the submits, at about 1.4-1.5 ms per
submit whatever the plane and the kernel. At 56×56, `k = 3` does nine times the
multiply-accumulates of `k = 1` and costs the same 6 ms. No compute term is visible at
these shapes. The wall is entirely the per-submit dispatch floor, the same 1.1-1.3 ms
that the int8 path pays once for its whole `ic` (`rk3576_conv_gate envelope`). At the
8-channel split the same shapes cost 8 and 16 submits. Doubling the contraction width
halves the submit count and the NPU wall with it.

Only the NPU column is a measurement. The host column is noisy on this thermally
limited board, where the same shape varies by a factor of two run to run. The 56×56
`k = 1` row is the clearest case of it. The NPU column is stable to within 6% across
pacing gaps of 0, 20 and 50 ms.

Both planners take the precision that the conv is emitted at:
`rocket_rk3576_cbuf_f_prec` and `rocket_rk3576_max_task_rows_prec`. The shorter names
are the same planners at int8. The feature plane's granule cost is per byte, so a
2-byte element doubles it. An allowance planned at the int8 rate is sized for half an
fp16 plane, and that plane computes wrong with a full surface written. The weight side
of the trade stays at the int8 model on purpose, as an upper bound on the float slice.
Over-estimating refuses early instead of corrupting.

### The write-extent field and the byte-placement words

The low nibbles of `0x4050` encode the written extent. The field encodes 1, 2 and 4
bytes per element as `{0, 1, 3}`: the direct int8 word carries `1` and the depthwise
word `3`. That is the same `size_e = bytes-1` relation that the RK3588 pairs with its
surface-advance multiplier. The RK3576's surface advance follows the same multiplier
(`ow*(2*oh_full - oh)` direct, `ow*(4*oh_full - oh)` depthwise). The field moves the
written *extent* only, never the atom. It is not what the float mode needs from this
register: bit 31 is.

Two DPU words do move the output, and they look like the float mode's answer without
being it:

| register | effect |
|---|---|
| `0x40D0` low 16 | a per-byte write enable for the 16-byte atom: clearing bit *n* leaves byte *n* untouched |
| `0x40D0` bits [19:16], `0x40CC` bits [3:0] | shift the written data by whole bytes within the atom |

### Coefficient formats in float mode

`A` (bias) is consumed as fp32. Packed as its fp32 bit pattern for `bias[c] = c+1` and
read back at width `5`, it returns `3f800000 40000000 40400000 40800000`, exactly 1.0,
2.0, 3.0 and 4.0. `C` (multiplier) is consumed as **fp16**, so the int8 default `C = 1`
is the denormal 6e-8 and underflows the whole surface to empty. The fp16 value `1.0` is
`0x3C00`.

Asymmetric weights are a packing question, not an encoding one. `B` in the coefficient
group is a per-output-channel int16 beside the bias `A` and the multiplier `C`. Driving a
weight zero point through it needs no register change, and
`rocket_rk3576_pack_coeff_asym()` takes one. The DPU adds the term, so a weight zero
point is programmed negated. The measurement is under "The B term is added" above.

### The harness

`tests/rk3576_fp16_sweep.c` scores every candidate against a CPU model of the same
convolution. The failure mode here is a full, correctly sized, *wrong* surface, which no
amount of looking at the output reveals. The harness reports an untouched buffer, an
all-zero surface and a wrong surface as three separate results. An untouched buffer is a
dead DMA, and an all-zero surface is an underflowed or gated epilogue. Only a wrong
surface is an arithmetic or layout error.

Its probes turn "wrong" into "wrong here". Each is built so that its answer does not
depend on what it is not measuring:

| probe | what it drives | what a correct part returns |
|---|---|---|
| 1 | uniform feature and kernel, each value its own knob (`_PIN`, `_PW`) | `ic*k*k * in * w`, and a result that tracks a side's *high byte* names the side being read as int8 |
| 2 | one non-zero element on each side | one non-zero output lane, at a position |
| 3 | the pixel ramp on every input channel | `y*iw+x+1` in every lane, whatever the lane map |
| 5 | uniform feature, kernel *c* weighted *c+1* | `c+1` in every lane, whatever the lane map |
| 6 | input channel *c* carries *c+1*, kernel taps ic=0 | `1`. Any other value names the channels the part paired with lane 0 |
| 7 | uniform feature, kernel ramps over ic | `1+...+ic`. Any other value counts the weight lanes it walked |
| 8 | unique naming: every lane carries `ow*oh*c + p + 1` | itself, the output map, decoded rather than guessed |
| 9 | unique feature, output channel *c* taps input channel *c* | the input cube handed back through the datapath |
| 10 | one input channel at 1.0, every weight lane 1.0 | `1`. The value is that channel's read count |

Probes 3 and 5 are uniform by design in the axis they do not name. They read the output
map without first knowing the lane map. Probes 6 and 7 do the reverse.

Probes 8, 9 and 10 exist because that uniformity is also the others' blind spot. Probes
1, 3, 5, 6 and 7 each pass against a datapath that reduces the wrong input channels. A
probe set built only from them reports an exact conv that is not one. Probes 8 and 9
name lanes instead of scoring them, so the pass is a *bijection*, not a match count.
Probe 10 measures the reduction directly. Run `map` before believing any fp16 layout claim.

| knob | what it does |
|---|---|
| `map` (argv) | drive probe 8 and decode the surface, reporting where each lane landed and how many copies |
| `ROCKET_FS_OUT_LAYOUT` | `0` the native output cube, `1` the library's `rocket_rk3576_fp16_out_index()` |
| `ROCKET_FS_TAP` | move the single live weight lane off ic=0. `-1` taps every input channel (probe 3) |

| knob | what it does |
|---|---|
| `int8` (argv) | the control, the known-good path through the same harness |
| `icsplit` (argv) | the library's `ic` split, scored end to end and timed |
| `f32out` (argv) | the fp16 datapath read back as the full 32-bit epilogue word |
| `bitsweep` (argv) | one register walked bit by bit or over `ROCKET_FS_SWEEP_VALS` |
| `ROCKET_FS_BIAS` | `1` integer bias probe, `2` the same value as fp32 bits |
| `ROCKET_FS_CMUL` | the `C` multiplier. `15360` is fp16 1.0 |
| `ROCKET_FS_SET` | whole-register overrides for the `manual` candidate |
| `ROCKET_FS_C2` / `_WOC` / `_WIC` / `_OUT_C2` | the cube geometry, each axis on its own |
| `ROCKET_FS_DUMP` / `_DUMP_OFF` / `_SCAN` | raw words at an offset, and the written extent and every non-zero lane |

**Run the `int8` control before believing any fp16 result.** It exercises the same
scatter, submit and de-scatter on a path this part is known to compute. So it fails only
when the harness is wrong. It must report `2048/2048`.

**Read the raw surface, not only the score.** Scoring reads the buffer *through* the
layout under test, so it cannot show a layout that disagrees with it.

### Traps

- **A register sweep must parse hex.** The call `atoi("0x100c")` returns 0. A sweep that
  silently patches register 0 reports that every value changed nothing. A correct sweep
  of an inert register reports exactly the same. The mistake can survive a full pass, as
  it did over `0x100C` before the reading that mattered was found.
- **`feature_data()` and `weight_conv_*()` take 1-based channel and kernel indices.** A
  0-based call drives the group remainder negative and misplaces the whole cube. The
  result is a wrong surface rather than an absent one, so it reads as an encoding error.
- **An integer test pattern cannot see a low-16-bit truncation.** Every value that a
  small integer convolution produces has zero low mantissa bits in fp32. Reading an fp32
  result through a 16-bit output width therefore returns a *uniformly zero* surface. That
  is the same signature as a gated epilogue or a dead MAC. Read at width `5`, or drive
  values whose low mantissa bits are non-zero.
- **A bias probe fixes only the channel-to-lane mapping.** Every pixel carries the same
  value per channel, so nothing in it constrains the pixel mapping.
- **A probe that is uniform in one axis cannot validate the reduction.** A uniform
  feature reports how many lanes were read, never which. A weight held constant over
  `ic` makes the contraction invariant under any permutation of the channel axis. A sum
  over `ic` is permutation-invariant however it is driven. A datapath that reduces the
  wrong input channels passes all of them. Measure the multiplicity (probe 10) and
  decode a unique-naming probe (`map`) before calling any float datapath exact.
- **Size a weight cube by its group, not by `ic`.** A partial input-channel group still
  occupies a whole one, so `ic = 8` at an ic group of 16 needs a cube for 16. An
  allocation taken from `ic` alone under-allocates. At `k = 1` the overrun stays inside
  the BO's page and computes correctly anyway. It surfaces only at a kernel large enough
  to run past the page, where it reads as "this part cannot do `k = 3`". Size the buffer
  from `ceil(ic/group)*group`.
- **A cube-geometry sweep needs a shape where the field is observable.** The float
  weight cube's output-channel group is algebraically invisible at `k = 1`. The kernel
  index sits between the group and the `(oc2, ic2)` pair, so at `k = 1` the index
  collapses to `(oc/G)*G + oc%G = oc`. Every candidate then emits the same bytes. A sweep
  at `k = 1` reports the field settled while testing nothing, which is how it fitted the
  wrong value 8. Before believing a geometry knob, check that two candidates produce
  different buffers.
- **Score only what the map covers.** Reading a surface through the layout under test
  cannot reveal a layout that disagrees with it. A partial match invites fitting a shift
  to three atoms. Decode a unique-naming probe instead: the answer is a bijection or it
  is not.
- **Pace probe loops.** Back-to-back submits return untouched surfaces at a rate that
  depends on the gap. At 200 ms and above, every submit writes. Treat one no-write as a
  measurement to repeat, and re-submit rather than record it.
- **The bias belongs to one slice of an `ic` split.** Every slice runs the whole epilogue
  and adds the whole `A` term. A coefficient buffer carrying a bias, handed to all of
  them, lands it `ic/16` times. The error is a per-channel constant on an otherwise exact
  surface, which is the shape a wrong `B` sign convention also has.
- **A slice's weight cube is not a sub-cube of the whole conv's.** Each slice is its own
  convolution, so its group count follows the slice rather than the total. Slicing the
  full cube by byte range hands the part a correctly sized cube with the kernel positions
  shuffled.
- **One register at a time cannot find a mode.** The float datapath needs three
  registers together. Each of the three looks inert while either of the others is at its
  integer value. A single-register sweep of any one of them reports it dead, and a
  leave-one-out over the working set is what separates them. Two of the three are also
  unreachable by a single-bit sweep from the integer word.
- **A vendor capture is cheaper than a sweep, and one can be manufactured.** A capture
  set that is all int8 carries no float fields, but a float capture can still exist.
  RKNN-Toolkit2 emits one for this part after about an hour of dependency pinning, and
  one diff of it settled what bit-level sweeps had not. The same lever settled four
  depthwise register formulas and the depthwise weight cube. Before sweeping for
  anything, ask which ONNX input makes the vendor compiler emit it.
- **A found capture set is a set of confounds.** The vendor compiles real models, so
  the axes that co-vary in real models co-vary in every capture. In this part's
  captures, every C=32 depthwise program was stride 1 and every C=64 one stride 2. Every
  depthwise kernel was 3, and every plane was square.

  Three register formulas fitted through those confounds were all wrong: `0x4050`'s
  channel field, the surface rounding in `0x401C`/`0x40B8`, and `0x118C`. Register
  `0x118C` is `iw-1` in both halves, and it reads as the plane height whenever
  `iw == ih`. When a capture set cannot vary an axis, manufacture one that does before
  fitting anything to it.
- **A degenerate helper program is not a convolution.** A model on a plane whose size
  is not a power of two makes the compiler append a 1×1×1 program beside the conv. That
  program carries a DPU_RDMA config that no conv carries. A conv emitter never produces
  it, and it reads as a spurious gate failure.
- **Timing an `ic` split at a probe pacing gap measures the gap.** The gap knob
  `ROCKET_FS_GAP_MS` exists to make a no-write rare, not to make a number meaningful.
  Time at `0`, and read the retry count that the harness reports alongside.

## `0x5024` is the DPU shift word, and zero is not a safe value for it

`BS_BASE_ADDR1` (`0x5024`) is a live operand base, not a spare word. The DPU
reads one 32-bit word through it per task and right-shifts the accumulator by two
independent 6-bit fields:

```
bits[5:0]   right shift applied where the result is NON-NEGATIVE
bits[13:8]  right shift applied where the result is NEGATIVE
```

Both are exact powers of two over the whole range. A word of `0x0202` takes an
accumulator of +64 to +16 and one of -64 to -16. A field of 8 or more flushes that sign
to zero. The two sides are independent. The field selection follows the sign of
the **accumulator**, not the sign of the weight, so swapping the feature sign swaps which
output channels move. [HW sweep, H96]

The shift applies to the BS product, and it rounds half to even [HW sweep, H96 MAX M9,
`tests/rk3576_coeff_c.c shift`, 2026-09-23]. The stage computes `((acc + A)*C) >> s`. Take
`C = 24576`, a shift of 14 and an accumulator of 160000, which is not a multiple of `2^14`. The
part returns the multiply-then-shift value rather than `(acc >> 14)*C`, and the product is held
wide, as the coefficient buffer above records.

An exact half rounds to the even side. With `C = 1` and a shift of 1, 32 odd values of A over
both signs round this way. Of five candidate rules, only half to even fits every channel. So a
per-channel C carries a fixed-point gain with the shift as its binary point.

Every capture stores 0 here, for the same reason every capture stores 0 for the
feature, weight, output and bias bases. It is an address that the vendor runtime
patches at load time, and a `.rknn`-derived program is the *stored* register set.
Leaving it at zero is not benign, because **IOVA 0 is a real buffer on a mainline
`rocket` stack**. The per-fd address space bump-starts at 0, so whichever BO the
caller allocates first is addressable as 0. The DPU then takes both shift amounts
out of that buffer's first four bytes, which are normally the head of the feature
cube. Point it at 64 zeroed bytes: `rocket_rk3576_coeff_bytes()` reserves them
past the A/B/C groups, and the emitter addresses them.

The fault presents as data-dependent arithmetic, not as addressing, because the shift
depends on two bytes of *feature data*. A conv is bit-exact for one feature tensor and
attenuated by an arbitrary power of two for another. It does so reproducibly, with no
fault and no dmesg.

It also hides from the obvious probes. A uniform feature fill puts the same byte in both
fields, so the result is either right or uniformly wrong. An identity weight set reads
one input channel per output channel and cannot show a contribution that the model
expects to be zero. Any feature whose first two bytes have zero low-6-bits, such as a
fill of 64 or of 0, computes perfectly. The symptom follows the
*feature cube's channel 1* only because that channel's first byte is byte 1 of the buffer.

Recognize it by its pattern. The result is exact for some feature tensors and a clean
power-of-two too small for others. The factor does not change with the programmed
requant scale, which places the fault upstream of the OUT_CVT and downstream of the MAC.

## The cold-start wall

On an unpatched `rocket`, only the first job of each NPU power session computes. The
table submits the same job N times and counts how many wrote [HW sweep, H96]:

| Gap between jobs | Result |
|---|---|
| 0 ms (back to back, one fd) | job 1 writes, and jobs 2-8 leave the output untouched |
| 100 ms | all 8 write, byte-identical |
| 300 ms | all 8 write, byte-identical |
| separate processes | every one writes |

The NPU runtime-suspends 50 ms after its last job. So the unit is the NPU power
session, not the fd, not the process, and not chained-versus-independent submits.
Independent per-op submits inside one power session fail exactly as a chained task
does, and any gap past the autosuspend delay re-arms the PC.

An external report on a different userspace (mesa/rocket + Teflon) matches this. It
read the behavior as the part arming its weight loader once per power session, and
found that independent per-op submits cannot bypass it. The same behavior on this
project's own encoder and submit path is what places it in the driver.

### The `PC_TASK_CON` field width

The wall's cause is the `PC_TASK_CON` field width. The register packs `TASK_NUMBER` in
the low bits, with three control bits directly above it. The field is 12 bits wide on
the RK3588 and **16 on the RK3576**, so all three control bits move up by four:

| | `TASK_NUMBER` | `TASK_PP_EN` | `TASK_COUNT_CLEAR` | `TASK_LAST_LAYER_CLEAR` |
|---|---|---|---|---|
| RK3588 | `BIT[11:0]` | `BIT(12)` | `BIT(13)` | `BIT(14)` |
| RK3576 | `BIT[15:0]` | `BIT(16)` | `BIT(17)` | `BIT(18)` |

The RK3588 register description marks the top control bit reserved. On the RK3576 it is
`task_last_layer_clear`, and it belongs on every submit alongside the count clear.
Chaoyi Chen of Rockchip gave the layout on the `linux-rockchip` list
([message](https://lore.kernel.org/all/4f300b78-d96d-4d98-8819-dc292b0c9b97@rock-chips.com/)),
which makes the naming authoritative rather than inferred. Shifting the whole triple
gives the same word independently, and both derivations write `0x70001`
[source-confirmed].

`rocket` builds the word from the RK3588 field accessors unconditionally, giving
`0x7001`. On this part that asks the PC for a task count of 28673, with all three
control bits above the register's defined fields. The PC starts that 28673-task
program, runs the one task it was handed, and is left mid-stream. Nothing else in the
session starts. A power cycle resets the PC, and that is the whole of why a gap re-arms
it.

Building the word from a per-SoC width clears the wall, and leaves the RK3588 word
bit-identical at 12 bits. Back to back with no gap, 1 of 8 submits writes without the
fix and 8 of 8 with it. A longer soak writes 32 of 32, byte-identical to each other and
bit-exact against the CPU model. A row-windowed 112×112 k3 conv is bit-exact over the
whole surface with no inter-task gap. The patch is `rk3576/npu/0008` in the `rk3576-npu`
profile. [HW sweep, H96 MAX M9, measured 2026-07-25]

Sweeping the control field at each width isolates the cause to `TASK_NUMBER` alone:

| word written | `TASK_NUMBER` seen (16-bit field) | wrote |
|---|---|---|
| `0x70001`, `0x60001`, `0x50001`, `0x30001`, `0x20001`, `0x10001`, `0x00001` | 1 | 8 of 8 each |
| `0x07001` (the RK3588 word) | 28673 | 1 of 8 |
| `0x06001`, `0x04001`, `0x02001`, `0x01001` | 24577, 16385, 8193, 4097 | 0-1 of 8 |

So `TASK_COUNT_CLEAR` is not the mechanism. With the field correctly placed, the part
works with every control bit clear. With it misplaced, no control-bit value tried helps.

Of the other two per-SoC PC parameters in the vendor config, the IRQ-locked
`PC_DATA_ADDR` write of `pc_dma_ctrl` is not needed. Neither is the RK3576-only
`state_init` hook. The other parameter, `pc_task_status_offset`, is real and
load-bearing. The register at `0x48` is the live task counter that a chained stream's
completion is read from (see "`PC_DONE` is per task" below).

Nothing a regcmd can write clears the wall, so the fix is a driver patch. Each of these leaves
the part at one job per power session [HW sweep, H96 MAX M9]:

- Sweeping the CNA/CORE/DPU/RDMA `S_POINTER` value over the whole ping-pong field
  (`POINTER`, both `PP_EN`s, `PP_MODE`, both `PP_CLEAR`s)
- Writing `PC_TASK_CON` from inside the stream
- Replaying the vendor's state-init sequence as regcmd writes

Two of those results are informative in themselves. Forcing `POINTER=1` makes every
job write nothing, so the bit is live and selects a producer register group. Job 2 still
fails with `POINTER` held at 0, so the consumer is not advancing behind the writer, which
rules the ping-pong groups out. [HW sweep, H96 MAX M9]

**Read back, the `POINTER` field is not the driver's to write while `PP_MODE` is set.**
A second party reports from a Radxa ROCK 4D [linux-rockchip RFC v4 4/6 cover and code
comments, Jiaxing Hu, 2026-08-03, their measurement, not reproduced here]:

- Bit 0 written as 0 reads back as 1, on every job, for the rest of the session.
- Flipping it per submit, in the direct register writes *and* in all four `S_POINTER`
  entries of the regcmd, moves neither the readback nor the result.
- Selecting a bank the way the vendor's `state_init` does, with the `PP` bits clear,
  stops the units arming at all.
- Pulsing `POINTER_PP_CLEAR`, with or without `EXECUTER_PP_CLEAR`, moves nothing.
- A 20 KB read snapshot of `pc`, `cna`, `core`, `dpu` and `rdma` differs in exactly one
  word, `OPERATION_ENABLE`. The snapshot is taken at the same point in a job that
  computed and in one that did not.

So the field is hardware-owned under `PP_MODE`. That agrees with the sweep above, which
finds no `S_POINTER` value that changes a second submit's fate. It disagrees with one half
of that sweep. Forcing `POINTER=1` from inside the regcmd kills every job here, where they
report that flipping it changes nothing. Two candidate reasons remain, and neither is
settled:

- The writes go in from different places: a regcmd the PC is fetching, against a
  slave-mode register write.
- The posted series carries no equivalent of `0008`. It still emits mainline's
  `PC_TASK_CON_TASK_NUMBER(1)` against the RK3588's 12-bit accessor.

The posting does not show whether the tree the experiments ran on carried such a fix. Their
public repo has had its own 16-bit fix since at least 2026-07-26, so do not state which.

That caveat is a method trap, not a register fact. On a driver that still programs the
RK3588's 12-bit `PC_TASK_CON` word, the second submit of every power session writes
nothing. The reason has nothing to do with the register under test. Every arm of a
ping-pong sweep then reads "no change", and the ledger is uninformative by construction
however exhaustive it is.

Before attributing anything to a configuration difference between job 1 and job 2,
establish that job 2 can compute *at all*. Stamp the output BO with a sentinel first. A
repeated configuration that reads a stale surface is byte-exact for the wrong reason.
That is exactly the shape that "one configuration is byte exact forever, a different one
computes nothing" produces. A fresh BO's zeros cannot separate "never ran" from "ran and
wrote zeros" (see the write guard in [rk3576.md](rk3576.md)). Fill it through a
`PREP_BO`/`FINI_BO` bracket rather than a bare `memset`, or the dirty lines race the DMA.

The vendor `rknpu` config carries three per-SoC parameters that mainline `rocket`
hardcodes at the RK3588 value. Only the first is load-bearing:

| | RK3588 | RK3576 |
|---|---|---|
| `pc_task_number_bits` | 12 | **16** |
| `pc_task_status_offset` | `0x3c` | `0x48` |
| `pc_dma_ctrl` | 0 | **1** (writes `PC_DATA_ADDR` under the IRQ lock) |

The RK3576 also has a per-SoC `state_init` hook that no other Rockchip NPU in that
driver has. It runs at every power-on with the PC in slave mode. It seeds CNA `0x1024`
with bit 31 in both ping-pong register groups, then clears the pointer
(`S_POINTER = 0x1e`). The `rocket` driver has no analog, and the slave-mode entry cannot
be replayed from a regcmd the PC is fetching.

Placing the hook in `rocket` is easy if it is ever wanted. The driver's
`rocket_job_hw_submit()` already opens with `PC BASE_ADDRESS = 0x1`, which is slave
mode, and the vendor's CNA writes drop straight into that window. To match the vendor
at power-on, the place is the runtime-resume callback.

Neither the hook nor `pc_dma_ctrl` is needed, and the two are ruled out jointly. They
are not needed for correct back-to-back submits, because the task-number width accounts
for the wall on its own. Nor are they the cause of the wrong answers that two jobs in
flight on the two cores compute. That workload is the one that would reach the second
pointer group.

The joint test seeds both groups at every power-on for each core, and holds a
device-global lock across the whole register-programming sequence. The concurrent cell
stays at 100% of calls wrong, against a 97.9% control. The seeding is counted, so the
negative is about the sequence rather than about a knob that never fired. Applying the
two together is the point. A one-at-a-time pass cannot see a condition of two, and
since the whole delta does nothing, no subset can. [HW sweep, H96 MAX M9,
`tests/rk3576_core_pair.c pair`, see [rk3576.md](rk3576.md) for what the mechanism is
narrowed to instead]

### Working against an unpatched driver

This subsection applies to a `rocket` without `0008`. A probe on a stock module hits all
of it.

**`autosuspend_delay_ms` is the only userspace lever, and it is unreliable.** Writing 0
to `/sys/.../27700000.npu/power/autosuspend_delay_ms` makes the driver drop the power
domains straight after each job. A back-to-back loop then climbs from 1 job in N to
about half: 8 of 16, and 6 of 6 on a shorter run. It is a race, not a fix: a job that
arrives before the autosuspend work runs gets no power cycle and is walled. Anything
that must be correct has to check, per task, that its own output landed, and resubmit.
The `rk3576_first_light` test does this when it runs a row split.

A gap of ~1 s makes results stable and repeatable. At ~0.4 s a probe still occasionally
comes back with the output BO wholly untouched. Read a single "the DPU did not write at
all" as the wall, not as a register result, and re-run before believing it.

**The wall has two signatures, and the second one is the dangerous one.** Either the
output BO comes back wholly untouched, or the DPU writes a full surface carrying only
the bias. In the second case the MAC contribution is missing and the surface reads all
zero. Both look exactly like a wrong register program. The second is also what a `C=0`
coefficient buffer produces, so three unrelated causes converge on one appearance.

The `rk3576_first_light` test retries past the autosuspend on both signatures. It
treats an all-zero surface as the wall only when the CPU model predicts a non-zero one.
A sweep that reports its results without that check measures the wall roughly half the
time at multi-second gaps.

## The per-submit floor is the driver's completion poll

Every cost table on this part is a submit-count table, and the per-submit cost is not
the silicon's. On this revision `PC_DONE` is read-only in `INTERRUPT_MASK`, so it cannot
be routed to the GIC. The RK3588 takes an interrupt, and `patches/rk3576/npu/0006`
polls the bit on an hrtimer instead. The period is the floor.

`patches/rk3576/npu/0009` makes the period a module parameter. Patch `0012` sets it to
50 us and changes what the poll retires on. The `rk3576_submit_floor` test submits an
8×64×32 int8 matmul 300 times, a shape chosen so that dispatch dominates compute. Through
it the floor is 439 us/submit against 1654 at the stock 1 ms, 3.77x.
[HW sweep, H96 MAX M9, measured 2026-07-28]

**Do not read a gate's total wall time as the floor.** The conv gate's 102 shapes take
about 5.9 s, unchanged, at every period from 1000 us down to 150 us. Host packing and
the gate's CPU model dominate that gate, not submits. It looks like a flat, negative
result while it measures something else. Time a single small shape, which is what
`rk3576_submit_floor` is for.

### Chained row tasks

A drm task carries one PC program, whatever the stream holds. The obvious way to
amortize the floor is to batch. A row-windowed convolution is one submit per window,
and a tile's row tasks are independent by construction. Each writes its own rows of the
same surface from its own window of the same feature cube. The weight and coefficient
buffers stay unchanged across them. Concatenating their programs into one regcmd stream
is arithmetically sound.

**It does not run.** The driver programs `PC_TASK_CON` with `TASK_NUMBER = 1` per drm
task descriptor, so the program counter executes the first task in the stream and stops.
A per-task write check names the earliest missing task, not only that one is missing.
At 3 tasks and at 6, task 0 lands and task 1 is the first missing. That holds on every
one of eight attempts with the power domain confirmed cycled between them, so it is
deterministic and not the poisoning. Across the conv gate that is 27 shapes, all
multi-window or multi-group plans, and no single-task one
[HW sweep, H96 MAX M9, 2026-07-28].

Collecting the floor takes a kernel patch and a joint layout contract, and both exist.
Submitting n drm task descriptors is not enough on its own. The driver already runs
those back to back (`rocket_job_hw_submit()` re-arms on `next_task_idx` from the
completion path), but each through its own completion poll. The poll is the floor, so
that form saves the ioctl and the fence round trip and not the thing worth saving. What
collects the floor is the RK3588 series' `086` arrangement, ported as
`patches/rk3576/npu/0015` and `0016`:

- Userspace lays the n programs out contiguously at one even-word stride. It rewrites
  each trailer's inert `OP_NONE` filler into a `PC_BASE_ADDRESS` write pointing at the
  next (`src/rocket_chain.c`, shared with the RK3588). This part's emitter ends every
  program with the same `[OP_NONE, PC_REGISTER_AMOUNTS, OP_40, OP_ENABLE]` trailer that
  the rewrite claims.
- Userspace submits them as n drm tasks with `DRM_ROCKET_JOB_BATCHED`.
- The driver programs task 0 only, sets `TASK_NUMBER = n`, and advances `next_task_idx`
  straight to the end. The PC streams all n from one kick, and the job retires on the
  single completion that `TASK_NUMBER` gates.

The `TASK_NUMBER` bound is per-SoC, so `0016` takes it from `rocket_soc_data` (16 bits
here, 12 on the RK3588) rather than from the RK3588 field mask.

Chaining meets the correctness bar, and it saves one completion poll per row task
removed. With chaining on, the convolution library gate reads 156 passed, 7 refused as
required and 0 failed. That is identical to the one-submit-per-task path. A
concatenated stream in one drm task fails 27 shapes of the same gate. A 32×32 k3 plane
forced into 8 row windows goes 2.8 ms -> 1.0 ms, and a 224×224 k3 s1 convolution goes
21.3 ms -> 19.1 ms. [HW sweep, H96 MAX M9, 2026-07-29]

**Do not look for this in a gate's total wall time.** Over the whole convolution library
gate it is 6643 ms -> 6463 ms, about 3%. Forcing `ROCKET_RK3576_MAX_ROWS` down to 8 or 4
does not enlarge it, because most shapes in that gate plan into one or two row tasks.
The gate's wall is host packing and the per-call power-cycle guard, not submits. The
lever is worth `(n-1) * 439 us` per call and nothing else: it pays on the deep planes
and is invisible on the shallow ones.

Chaining is default-on in the int8 convolution path of `librocketnpu`, and
`ROCKET_RK3576_BATCH_TASKS=0` turns it off. The same layout carries a run of
cube-linked layers as one kick (see the cross-layer kick in [rk3576.md](rk3576.md)).

Both trailer fields describe the next segment. The driver programs `PC_BASE_ADDRESS`
and `PC_REGISTER_AMOUNTS` from task 0 alone. So a chain of programs that differ in
length must write its successor's count, not its own. A uniform row-task chain cannot
distinguish the two.

The `rocket_batched_submit_supported()` check refuses to self-chain against a kernel
that does not honor the flag. It reads `/sys/module/rocket/parameters/rocket_batch_submit`,
else the advertised DRM interface version, which must be >= 1.1. That check is
load-bearing, because a chained layout run down the per-task path runs task 0 and
stalls to the job timeout.

The fp16 `ic` split cannot chain this way as the path stands. Its slices repack the
same weight BO, then read back and accumulate into the host's buffer between submits.
That host work between them means they are not one stream. Chaining them needs
per-slice weight buffers, and either per-slice output regions or on-chip accumulation.
That is the same restructuring the on-chip-accumulation item wants, not a free
consequence of this patch.

### `PC_DONE` and the DPU completion bits

`PC_DONE` is the wrong retire signal on this part, and the DPU's own completion is the
right one. `PC_DONE` means the program counter finished issuing, not that the DPU's write
DMA has drained. The driver tears the IOMMU mapping down and signals the fence as soon as
it sees `PC_DONE`. A long poll period hides both actions behind its own detection lag,
and with them two independent failures:

- **The per-job IOMMU detach races the write drain.** Detaching on completion faults
  writes that are still in flight, which leaves a *write* page fault active:
  `RK_MMU_STATUS` `0x2b` = `PAGING_ENABLED|PAGE_FAULT_ACTIVE|IDLE|PAGE_FAULT_IS_WRITE`.
  The fault blocks the next `enable_stall`, which the IOMMU reports as
  `Enable stall request timed out`. The companion status `0x1d` carries `STALL_ACTIVE`,
  so the stall did arrive, just after the poll gave up. The NPU then raises
  `DMA_READ_ERROR|DMA_WRITE_ERROR` and the job comes back short. Patch
  `patches/rk3576/npu/0011` removes this outright by keeping the domain attached across
  jobs.
- **The fence is signaled before the writes are visible.** Patch `0012` closes it.

`RAW_STATUS` carries a signal that retires when the writes have landed. The RK3588 takes
`DPU_0` (`0x100`) and `DPU_1` (`0x200`) as its completion interrupt. On this part they
set tens to hundreds of microseconds after `PC_DONE`. A log line taken at the moment the
poll retires reads:

```
raw@retire=0x20000001 -> 0x30000155 after 82us
```

The word first shows `PC_DONE_1` with the CNA feature bit and no DPU bit, then `DPU_0`
82 us later. Requiring the DPU bit is the fix, and it is what lets the poll period drop:

| poll period | 500 us | 250 us | 125 us | 50 us | 25 us |
|---|---|---|---|---|---|
| us/submit, retire on `PC_DONE` | 1065 | 731* | 535* | 433* | n/a |
| us/submit, retire on the DPU bit | 1073 | 739 | 558 | 439 | 398 |
| int8 matmul gate, retire on `PC_DONE` | 43/43 | 42/43* | n/a | 36/43* | n/a |
| int8 matmul gate, retire on the DPU bit | 43/43 | 43/43 | 43/43 | 43/43 | 43/43 |

`*` marks an incorrect result. [HW sweep, H96 MAX M9, 3 runs per correctness point,
measured 2026-07-28]

The failure is a partial surface, and the missing bytes are a contiguous tail. The
reproducer is `n-tiled-deep`, the largest output surface in the gate at 32×2048×3072. At
`poll_interval_us=250`, retiring on `PC_DONE`, it fails 3 runs of 3, with ~95-98k of
98304 elements exact. A snapshot of the output BO is compared after a settle and a
re-sync. It shows the last 2048 bytes of the 65536-byte surface arriving after the fence
retired (`changed=2048 first=63488 last=65535`), so the data is not lost but drains
late. The drain reaches 343 us for that surface and ~82-98 us for the conv gate's
smaller ones, so it scales with bytes in flight.

Not every task raises a DPU completion, so the wait has to be bounded. A task whose DPU
output element is wider than one byte completes CNA and CORE and never sets a DPU bit at
all. The int32 writer and fp16 output are both in that class. Their status word holds at
`RAW_STATUS = 0x30000055` (`PC_DONE_0|PC_DONE_1|CORE_0|CNA_CSC_0|CNA_WEIGHT_0|CNA_FEATURE_0`)
for milliseconds, with nothing further arriving. Patch `0012` gives these tasks
`rocket.dpu_grace_us` and retires them on `PC_DONE`, as the unpatched driver does. That
grace is load-bearing: those tasks still need a blind settle, and its threshold is the
same ~250 us the settle experiment found:

| `dpu_grace_us` | 100 | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|
| int8 matmul gate | 40/43 | 43/43 | 43/43 | 43/43 | 43/43 |
| through the wide int32 writer | 41/43 | 43/43 | 43/43 | 43/43 | 43/43 |

The grace is a deadline, not a settle, and that is what fixes its default. It bounds the
wait for a signal most jobs do raise. So it also covers the drain of every ordinary
narrow-output job. Lowering it retires those jobs on `PC_DONE`, exactly as the unpatched
driver does. The two roles pull opposite ways, and neither is optional:

| `dpu_grace_us` | 200 | 250 | 300 | 350 | 400 | 500 | 800 |
|---|---|---|---|---|---|---|---|
| `n-tiled-deep`, 10 runs | 10 fail | 0 | 0 | 0 | 0 | 0 | 0 |
| `fc-rgb-224-k7`, 5 runs | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The cliff is sharp and deterministic: every run fails at 200 and none at 250. So 250 is
1.25x the measured failure point, not a margin above it. The gates cannot separate 250
from 500, because nothing in them drains further. The case for a margin rests on the
drain itself. The drain is measured per gate, with the poll retiring on the DPU bit and
the grace set past every wait.

The instrument is a module parameter set on the board's out-of-tree debug tree. Its
`dbg_dpu_wait_max_us` and `dbg_dpu_waits` hold the maximum wait and the count over jobs
that retired on the DPU bit. Jobs that raise no DPU bit time out into `dbg_grace_hits`
instead. The instrument is diagnostic and deliberately kept out of the patch series, so a
reimage removes it along with everything else:

| gate | worst drain | jobs reaching the DPU bit | jobs raising none |
|---|---|---|---|
| `rk3576_conv_gate all` | 99 us | 15 | 0 |
| `rk3576_conv_lib_gate` | 745 us | 30 | 31 |
| `rk3576_matmul_gate` | 290 us | 9 | 58 |
| through the wide int32 writer | 343 us | 10 | 54 |
| `rk3576_argb_ic1` | 46 us | 2 | 0 |

The 745 us is one shape, `fc-rgb-224-k7` (the int8 stem at 224×224 k7 s2), and it
reproduces to within 3 us over five runs. So the shipped 500 us default is already below
the worst drain on the part, and the stem still passes at 200. A drain has to be covered
only when it belongs to the last submit before the host reads. The submits that follow a
row-split job's earlier tasks cover those tasks' drains. That exposed drain is what the
cliff measures, and it is not separately observable. The raw drain is therefore the only
available bound on it, and it reaches 745 us at a 392 KiB surface.

The default is 500 us, 2.5x the measured failure point. The drain scales with bytes in
flight, and the library can emit surfaces larger than any gate shape. The fp16
win from lowering the default (1.18x, below) is a cost paid only by the tasks that raise
no DPU completion. Userspace can name that class, and the driver cannot see it at
`PC_DONE` time. Collecting that win means separating the two classes at submit time, not
lowering the one number that also bounds the other class's correctness.
[HW sweep, H96 MAX M9, measured 2026-07-28]

Patch `patches/rk3576/npu/0017` makes that separation, and it collects the win without
touching the deadline. It adds `DRM_ROCKET_JOB_NO_DPU_DONE` to `drm_rocket_job.flags`. A
job carrying the flag waits `dpu_blind_us` (default 250) past `PC_DONE` instead of
`dpu_grace_us`. A job without it is bit-for-bit unaffected. The hint is advisory by
construction: the poll still retires the moment a DPU completion arrives, whatever the
flag says. A wrong hint therefore costs time and never correctness.

`librocketnpu` sets the flag on exactly the classes the poison probe already names wide.
Those are the fp16 direct conv, the fp16 first conv and the int32 matmul writer. The
measured shape is an fp16 convolution at `ic=128 oc=32` 28×28 k3, eight wide-output
slices, where the settle is the dominant cost:

| `dpu_blind_us` | 250 | 500 (hint neutralized) | 3000 |
|---|---|---|---|
| eight-slice wall | 13.67 ms | 15.20 ms | 36.96 ms |

The hint takes 10% off that path, and every narrow-output job keeps its full 500 us
deadline. [HW sweep, H96 MAX M9, measured 2026-07-29]

A negative A/B on a knob like this is a claim about the wiring first. One silently failed
edit is enough to leave the poll comparing against `dpu_grace_us`. The flag then reaches
the driver and changes nothing. The A/B shows no effect at any `dpu_blind_us`, which
reads as "the lever is not real".

Driving the other knob separates the two readings. In that state, `dpu_grace_us`
500 -> 3000 moved the same wall 15.31 -> 36.14 ms. The tasks were still paying the grace,
so the hint was not being applied. Before believing that a per-class knob does nothing,
show that the class reaches it. An out-of-range value that fails to move the wall is the
cheap version of that check.

The tasks that raise no DPU completion are the same set of programs that carries the
poisoning, and the two are separate hazards. With the DPU bit as the retire condition,
the poison probe's `scope` map is unchanged. The fp16 first conv, the fp16 direct conv
and both int32 writers poison all seven kinds. The int8 first conv, int8 direct conv and
int8 depthwise poison none.

**A poisoned submit raises its DPU completion normally while writing nothing.** So the
missing DPU bit is not a poisoning detector, and the poisoning is not "the DPU never
ran". The poison probe's `pair A int8d`, at `ROCKET_PP_REPS` of 4 and 8, gives grace hits
from the A side only, scaling exactly with the rep count.

With A as `fp16fc` the hits are 4 and 8, one submit a rep. With A as `i32` they are 8 and
16, two submits a rep. The poisoned `int8d` submits on the B side contribute zero at
every rep count. The sentinel scan is the only way to see a poisoned surface.
[HW sweep, H96 MAX M9, measured 2026-07-28]

Measured at a long poll period, the DPU bits look redundant: `DPU_0` is already set on
the same poll tick as `PC_DONE`. At 500 us of detection lag the drain has always finished
by the time the poll looks, so the two signals cannot be told apart. At 50 us they
separate cleanly.

### Per-task `PC_DONE` and the `PC_TASK_STATUS` counter

`PC_DONE` is per task, and `PC_TASK_STATUS` is the live task counter. The bits
`PC_DONE_0` and `PC_DONE_1` are two alternating per-task pulses, not one whole-kick
completion. On a `TASK_NUMBER = n` kick the two bits swap as the program counter retires
each program. The first of them appears a few tens of microseconds into a stream that
can run for milliseconds. It says nothing about the stream being over.

The bits are also not latched across tasks. Mid-stream the whole `RAW_STATUS` word cycles
through `0x10000000`, `0x20000000`, `0x30000000` and `0x00000000`. The per-task block
completion bits (`0x155` for one ping-pong group, `0x2aa` for the other) appear only
briefly and are gone again by the next task. Only the last task's block bits survive,
because no task follows to clear them. [HW sweep, H96 MAX M9]

`PC_TASK_STATUS` says where the kick is. The register is at `0x48` on this part and at
`0x3c` on the RK3588. The vendor `rknpu` config carries the delta as
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

Mid-stream at least one of the two halves is non-zero. So **`PC_DONE` set together with a
zero `PC_TASK_STATUS` means the whole kick is over, and nothing else does.** That is the
signal a driver needs to place a completion wait on a chained stream. A trace of the
35-program cross-layer kick of a MobileNetV1-224, at a 10 us poll:

```
t=   12 us   PC_DONE_0,  ts=0x00000001   task 1 of 35 in flight
t=   31 us   PC_DONE_1,  ts=0x00000001
 ...         the two bits alternating, once per task
t= 1911 us   PC_DONE_1,  ts=0x00220000   task 35 of 35 in flight
t= 1921 us   0x300002aa, ts=0x00000000   the DPU's own completion; the kick is over
```

**`PC_OPERATION_ENABLE` is a self-clearing go bit, not a busy flag.** It reads back 0 at
every poll of a kick that is still running [HW sweep]. The vendor driver writes `1` and
then `0` to it back to back (`rknpu_job.c`) [source-confirmed]. A completion test of the
form `OPERATION_ENABLE == 0 || PC_DONE` is therefore vacuously true on this part from the
first poll onward. Any wait built on that test is anchored to the first poll.

### The half-started job runtime-PM pin

A half-started job pins the NPU runtime-active, and then only the int32 path breaks.
Patch `patches/rk3576/npu/0010` fixes it. The bug is in mainline `rocket`, so a kernel
without 0010 has it, on the RK3588 as well. The state it produces is the most misleading
one the part reaches.

`rocket_job_run()` takes a runtime-PM reference and then attaches the IOMMU group. Both
of its early returns leave the job half-started. Each of two independent leaks is enough
on its own to pin the device runtime-`active`, with `power/runtime_suspended_time`
frozen. The pin holds even with nothing running and `power/control` at `auto`:

- **The usage counter ratchets.** `pm_runtime_get_sync()` keeps its reference when the
  resume fails, and the IOMMU path has already resumed successfully. At that point
  `core->in_flight_job` is still `NULL`. The completion path and the timeout path put the
  reference only when a job is in flight, so neither balances it.
- **The scheduler never gets its credit back.** Nothing will ever signal the fence handed
  to the scheduler, so the job stays pending. While a credit is outstanding,
  `rocket_device_runtime_suspend()` returns `-EBUSY`.

**The usage-counter half outlives the driver.** The reference is on the platform device,
which the OF core owns, so `rmmod`/`insmod` does not clear it. A freshly probed device
comes back `active` and stays there until a reboot.

The pin breaks exactly one thing. The power domain's cycling clears the int32 output
writer's poisoning of the next submit. On a device that cannot suspend, that poisoning
never clears, and every i32 shape writes nothing (`0/256`, `0/32`, `0/224`). Everything
else still works perfectly: the conv gate is 102/102, the library gate 111/1, and the
plain int8 matmul shapes all pass. The signature therefore reads as "the int32 path has
a bug" rather than "the machine is in a bad state". It will absorb a debugging session.

Without 0010 the state is also self-sustaining. One failed attach times out at 500 ms,
and the timeout path calls `rocket_core_reset()`. That reset takes the IOMMU down, so the
next attach fails for real. One injected failure produced seven, and leaked seven
references. With 0010 the injected failure is the only one. The driver retires the job
with its error, the caller's next submit attaches and computes normally, and the device
suspends.

**Check `power/runtime_status` before believing any int32 result.** A board in this state
makes every other experiment lie. A poll-period sweep run through it reports the matmul
failing at every period. That is the signature of a failure the poll has nothing to do
with. [HW sweep, H96 MAX M9, measured 2026-07-27]

### A per-job core reset is not the way to clear the poisoning

`rocket_core_reset()` is a `reset_control` bulk assert, `udelay(10)`, then a deassert,
and the driver runs it on a job timeout. Calling it at job start is not a way to drop the
power-cycle idle that an int32-output job forces on the next submit. Asserted before
every job, it takes the IOMMU down with it:

```
rk_iommu 27702000.iommu: Error during raw reset. MMU_DTE_ADDR is not functioning
rocket 27700000.npu: NPU job timed out: RAW_STATUS=0x30000000 MASK=0x800fffff OP_EN=0x00000000
```

`RAW_STATUS` carries both `PC_DONE` bits, so the job ran, but nothing reached DDR
because the page-table pointer was gone. The matmul gate falls from 43 shapes passing to
2, and the conv gate to 0. The damage outlives the setting: the part stays wedged until
the module is reloaded. The driver's own timeout path resets inside `drm_sched_stop()`,
an IOMMU detach and a re-attach, and the bare call lacks that surrounding re-init. The
vendor `rknpu` driver never resets per job either.
[HW sweep, H96 MAX M9, measured 2026-07-27]

The vendor's recovery reset covers the CBUF, and `rocket`'s does not. `rknpu_soft_reset()`
runs on a job timeout or on request, never per job. It sleeps 100 ms, then asserts all four
of the node's resets for 10 us: `SRST_A_RKNN0`, `SRST_A_RKNN1`, `SRST_A_RKNN_CBUF` and
`SRST_H_RKNN_CBUF`. It then detaches and re-attaches the IOMMU and re-runs its `state_init`
[source-confirmed: BSP `rknpu_reset.c` and `rk3576.dtsi`, unchanged in `develop-6.12`].

`rocket`'s binding names the core's `srst_a` alone, so under `rocket` only a power cycle
resets the CBUF. Whether a CBUF reset inside the timeout path's detach and re-attach
clears the poisoning is untested. If the latched state lives in the CBUF, it clears without
cycling the domain [expected].

## The IOMMU wedge

An earlier run can leave the NPU wedged, and it does not recover on its own. The
signature in `dmesg` is:

```
rk_iommu 27702000.iommu: Enable stall request timed out, status: 0x0000..
rocket 27700000.npu: NPU job timed out: RAW_STATUS=0x30000000 MASK=0x800fffff OP_EN=0x00000000
rk_iommu 27702000.iommu: Error during raw reset. MMU_DTE_ADDR is not functioning
```

After that sequence, every submit times out and every output BO comes back untouched.
That reads exactly like "the encoder writes nothing", and it will send a register hunt
down a false trail. The rail, the clock and the driver binding all still look healthy in this state,
so they do not discriminate. Check `dmesg` for the stall timeout before trusting any
negative result on this part. [HW sweep, H96]

Reloading the module clears the compute path, and it is worth trying before a reboot.
The reload `rmmod rocket; sleep 8; insmod rocket.ko` re-runs the IOMMU attach, and the
conv gates pass again immediately. The settle matters: a reload with a two-second gap
left the part still failing, where an eight-second one fixed it. This holds for two
independently induced wedges, one from a 100 us poll period and one from a per-job core
reset.

**On a kernel without `patches/rk3576/npu/0010`, a reload does not clear everything.** A
leaked runtime-PM reference lives on the platform device, so a reload leaves it in place.
The part then computes convolutions perfectly while every int32 matmul writes nothing, as
the half-started job section above describes. Check `power/runtime_status` after a
reload. If it does not return to `suspended`, reboot.
[HW sweep, H96 MAX M9, measured 2026-07-27]

With 0010 and 0011 the wedge is much harder to reach in the first place. The
`Enable stall request timed out` and `MMU_DTE_ADDR is not functioning` sequence above is
the per-job IOMMU detach racing the DPU's write drain. Patch 0011 stops detaching per
job. Patch 0010 stops one failed attach from cascading through the timeout path's core
reset into the next attach.

## A rejected submit oopses the kernel, and a refusing generator is what emits one

`rocket_ioctl_submit_job()` allocates its `rocket_job` with `kzalloc`. It assigns
`rjob->domain` only after the task copy and both BO lookups have succeeded, while
`rocket_job_cleanup()` puts that domain unconditionally. Every rejection ahead of the
assignment therefore unwinds into `rocket_iommu_domain_put(NULL)`, which dereferences
the NULL to reach the kref:

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

The device node `/dev/accel/accel0` is group `render`, so the trigger is unprivileged.
**This is mainline code.** It is identical in a pristine v7.1 tree and present since the
submit ioctl was added, and the RK3576 series does not introduce it. Patch
`patches/rk3576/npu/0013` fixes it by taking the domain reference at construction, so a
job owns one from the moment it exists. Patch `patches/rocket/085` carries the
complementary NULL guard in the put. The two touch different files and compose.

The test `tests/uapi_submit_errpath_rocket` fires one malformed submit per site from a
forked child. Before the fix 0 of 5 sites returned to userspace, and after it 5 of 5 did
[HW sweep, H96 MAX M9, 7.1.3].

The client that emitted one on this part was this project's own uAPI selftest, and the
mechanism is a refusing generator. `gen_matmul_fp16()` emits the RK3588 geometry encoding
and refuses on this part by construction. That leaves `matmul_params_t.task_count` at 0.
The selftest's deadline canary built its task descriptors from those unset parameters and
submitted `regcmd_count == 0`. **A generator that refuses leaves its output parameters
untouched, and a caller that submits anyway hands the kernel a malformed job.** Check the
generate step before building a submit from it.

A single Oops does not brick the part: an ordinary client and the full 46-shape matmul
gate both still run afterwards. Several Oopses do. After three, the file-close path
faults again inside the IOMMU layer (`iommu_map_nosync`, `iommu_unmap`,
`iommu_domain_free` from `rocket_postclose`). The process then becomes unkillable, and
the kernel prints `Fixing recursive fault but reboot is needed!`. Do not keep running
gates on a kernel that has taken an Oops. The results after it are not about the NPU.

The warning at `rocket_job.c:639` in `rocket_job_irq_handler()` is a different thing, and
it is not a driver defect. It is mainline's
`WARN_ON(raw_status & PC_INTERRUPT_RAW_STATUS_DMA_WRITE_ERROR)`, the driver reporting that
the NPU raised a DMA write error. Seen on core 1's IRQ during a two-core run, it is the
kernel-log fingerprint of the concurrent-job corruption, not a separate bug to chase.

A client-supplied bad address does not reach that `WARN_ON` on this part. The field
`drm_rocket_task.regcmd` is a raw IOVA. The driver's `rocket_job_hw_submit()` writes it
into the PC block's `BASE_ADDRESS`, with no check that the job's BOs mapped it. A client
can therefore aim the instruction fetch at unmapped memory. Through a real program with a
rewritten output address, it can aim the DPU's write DMA there too.

The probe `tests/uapi_regcmd_fault_rocket` tries both, as its `read` and `write` modes.
Both surface as `rk_iommu: Enable stall request timed out` and nothing else: 0 WARNINGs,
taint unchanged, and the matmul gate 46/46 immediately afterwards. The RK3576's IOMMU
absorbs a bad address as a stall, rather than the NPU's PC raising a DMA-error bit. A
`WARN_ON` on a hardware error condition taints the kernel, and panics it under
`panic_on_warn`. So the `WARN_ON`-as-unprivileged-DoS concern is real upstream in
principle, but it is not demonstrated to be client-triggerable here.
[HW sweep, H96 MAX M9, 2026-07-28]

**A job whose program faults still retires cleanly.** The submit returns 0, `PREP_BO`
returns 0, the output BO is untouched, and the driver tells userspace nothing. That is
the same "the encoder writes nothing" false trail the IOMMU-wedge section warns about,
arriving from the driver rather than from the encoder. An untouched output BO is
therefore not evidence about an encoding.

On the RK3588 the same probe finds a client-triggerable warning somewhere else. So the
question "can a client taint the kernel through the NPU" has a different answer from the
one about that particular `WARN_ON`. On the RK3588 both modes end in the job timeout,
and the reset behind it walks into the IOMMU core:

```
rk_iommu fdaca000.iommu: Enable stall request timed out, status: 0x2b
WARNING: drivers/iommu/iommu.c:157 at __iommu_group_set_core_domain
 iommu_detach_group / rocket_reset.part.0 [rocket] / rocket_job_timedout [rocket]
```

The NPU is still faulting when `rocket_reset()` detaches the group, so the IOMMU's stall
and disable-paging requests time out and the detach WARNs. Patch `patches/rocket/083`
(keep-the-domain-attached-across-jobs) is applied on that board and does not cover it.
Keeping the domain attached across jobs says nothing about a reset that detaches
explicitly. So on both parts the reachable path is the reset, not the IRQ handler. The
parts differ only in whether the detach WARNs. [HW sweep, Turing RK1, 2026-07-28]

## A BO in an in-flight job must survive the file that made it

The per-context IOVA allocator (`drm_mm` + `mm_lock`) lives in `struct rocket_file_priv`,
and `rocket_postclose()` tears it down and frees it. Only `rocket_gem_bo_free()` removes a
BO's IOVA node. The drm_sched free worker drops a job's BO references asynchronously, and
it can run after the owning file has closed. A client that submits and closes without
waiting therefore leaves `rocket_gem_bo_free()` taking `bo->driver_priv->mm_lock` and
calling `drm_mm_remove_node()` on memory `rocket_postclose()` already freed. The device
node `/dev/accel/accel0` is group `render`, so the trigger is unprivileged. The defect is
in mainline code that the RK3576 enablement does not touch.

It reproduces on demand. It is rare in ordinary use only because an ordinary caller waits
for its job and so never sets the race up. The test `tests/uapi_bo_lifetime_rocket` opens
a fresh fd per iteration, submits, and closes immediately. It warns on 16 of 16
iterations, and left running longer, the same client reaches the dereference:

```
Internal error: Oops: 0000000096000004 [#1]  SMP
Workqueue: 27700000.npu drm_sched_free_job_work
pc : add_hole+0x34/0x15c
lr : drm_mm_remove_node+0x1e8/0x380
 rocket_gem_bo_free [rocket] / drm_gem_object_free / rocket_job_cleanup [rocket] /
 rocket_job_free [rocket] / drm_sched_free_job_work
```

Patch `patches/rk3576/npu/0014` anchors the allocator to `struct rocket_iommu_domain`,
which is refcounted and outlives the file, every mapped BO and the attached core. With it,
0 of 16 iterations warn, on the same kernel with only that patch changed. It is the RK3588
series' `084` ported, and it applies to this series with offsets only.
[HW sweep, H96 MAX M9, 2026-07-28]

**Match the symbol, not drm_mm's message text.** A 7.1 kernel prints
`warning: drivers/gpu/drm/drm_mm.c:965 at drm_mm_takedown+0x28/0x38`. The older
"allocator still has nodes" wording does not appear at all. A detector keyed on the
message therefore reads clean while the defect fires on every iteration.

**The verdict is the kernel log, not the exit status.** This defect fails no syscall. A
run can pass every ioctl and still be the failing side of the A/B.

## The matmul as a 1×1 convolution

A matmul is a 1×1 convolution over these blocks, exactly as it is on the RK3588. The A
rows are the conv's spatial pixels, K is the input-channel axis, and N is the
output-channel axis. The register program is the convolution's, unchanged. The conv gate
runs a real matmul shape without calling it one. Its `w-e4608-k1` is `ic=4608`, `oc=128`
on a 4×2 VALID plane, which is M=8, K=4608, N=128 in one submit. The whole of the matmul
work is therefore in the userspace layer:

- The operand scatter
- The de-scatter
- A tiling planner
- A dispatch point

The entry is `rocket_matmul_int8_rk3576()`, and the gate is `tests/rk3576_matmul_gate.c`.
The probe `tests/rk3576_matmul_probe.c` is the instrument that read the envelope below.

The mapping, for anything driving the emitter directly:

| operand | layout |
|---|---|
| A[M,K] | the feature cube, channel `k`, pixel `m` at `(m/iw, m%iw)`, C2 = 16 int8 lanes |
| B[N,K] | `weight_conv_int8(N, K, 1, 1, n+1, k+1, 1, 1)` |
| C[M,N] | the int8 output cube, `(n/16)*surf_elems*16 + 16*(y*iw + x) + n%16` |

The plane is free. Any `(iw, ih)` with `iw*ih == M` carries the M rows, and all of them
compute, so the choice is a tiling one. The plane costs granules: a feature row takes
`ceil(iw*K/64)` of them against the task's 4096. The pixels one task holds therefore come
out at `262144/K` however the plane is cut, provided `iw*K` is a whole number of
granules. When it is not, every row rounds up and the waste is real. Take the widest
divisor of M that divides evenly.

### The M axis

The M axis carries no constraint. Every M tried (1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 16, 17,
31, 32 and 64) is bit-exact at every factorization into a plane, from `1xM` through
`Mx1`. Here `M=1` is simply correct, which inverts the RK3588's answer. There, rows are
the conv's spatial height, a height under 4 mis-computes, `M%4` is the real bound, and
software pads `M==1` to 4. Whether it is worth running at M=1 is a separate question,
which the dispatch floor answers. [HW sweep, H96 MAX M9]

### Matmul precision

int8 is the matmul precision, and the margin is large. The two precisions contract at
very different rates. One int8 task takes `ic*kh*kw <= 4608`, so K = 4608 lands in one
submit. One fp16 task contracts exactly sixteen input channels, so the same K costs 288
submits. The table measures each shape at both precisions, with the pacing outside the
timing:

| shape | int8 submits | int8 ms | fp16 submits | fp16 ms | ratio |
|---|---|---|---|---|---|
| 16×512×64 | 1 | 1.4 | 32 | 42.9 | 30x |
| 32×1024×128 | 1 | 1.3 | 64 | 85.2 | 64x |
| 128×2048×128 | 1 | 1.2 | 128 | 171.3 | 142x |
| 56×4608×128 | 1 | 1.3 | 288 | 386.1 | 299x |

On the RK3588 fp16 wins and int8 buys only RAM. Here it is the other way round. The
reason is the contraction width, not the arithmetic: the fp16 one is 16, while the int8
one is 4608. [HW sweep, H96 MAX M9, measured 2026-07-27]

### Throughput and the N axis

Throughput is MACs per submit, and N is the only free axis. A submit costs about 0.44 ms
whatever it carries. The feature budget caps `M*K` and the resident weight slice caps K,
so N is the only axis left to spend. At the feature-budget cap, N buys throughput almost
linearly:

| N | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 2560 |
|---|---|---|---|---|---|---|---|---|
| GOP/s | 12 | 24 | 47 | 93 | 190 | 367 | 719 | 1000 |

The table is at `M*K = 262144`. The figure is flat in how that product is split:
1024×256, 256×1024 and 64×4096 all land within a few percent of each other.
[HW sweep, H96 MAX M9, measured 2026-07-27]

### Output-channel bounds

Two output-channel bounds apply, and neither is a convolution's problem. Past either one,
the trailing output channels do not reach DDR. The surface is full and correctly sized,
its leading channels are bit-exact, and nothing faults. The planner `r76_plan_cbuf`
refuses a shape past either bound:

- **Output channels <= 2944.** 2944 computes 3/3 at every K tried. 3072 is intermittent
  (1/3 at K=256, 2/3 at K=1024), and 4096 never computes.
- **The whole weight cube <= 6 MiB.** The cube at `ic=4096, oc=1536` is 6 MiB and
  computes 3/3. At `ic=4608, oc=1536` it is 6.75 MiB and computes 0/3. At
  `ic=4096, oc=2048` it is 8 MiB and loses about a quarter of its channels.

The second bound is stated over the whole cube rather than per kernel tap, on purpose. It
was measured at `k=1`, the matmul's own kernel. Stated this way, it refuses earlier at
`k>1` than anything measured there, which is the safe direction. No convolution shape in
the gate comes near either bound: the largest is 516 KiB at oc=128.

**Repeat before believing a boundary here.** A single pass per point produced a
clean-looking N ceiling that moved when the same points were re-run. The shape
`1024x256x3072` failed in one pass and passed in the next. Three repeats plus a
classification of the failure (all-sentinel is a dead submit, a clean prefix is a
capacity bound, wrong values in a fully written surface are arithmetic) separate the
bound from the noise. [HW sweep, H96 MAX M9]

### The int32 output

On an integer program the DPU emits its raw 32-bit accumulator, but only the first eight
channels of every thirty-two reach DDR. The generator `gen_conv2d_int8_rk3576_i32out()`
puts `precision_int32` in DPU `0x4010`'s output-width field and pins OUT_CVT to exact
unity. The words that come back are genuine accumulators, laid out on the RK3588's int32
cube map in 32-bit words:

```
word = (c/4) * ow*oh_full * 4 + 4 * (y*ow + x) + (c%4)
```

The map is decoded, not assumed. The probe gives every accumulator a distinct value, so
each value names exactly one word. Every position read that way fits the map with no
misses. The decode covers `oc` 4, 8, 16, 32, 64 and 96, `K` 16 through 256, and `M` 4
and 8.

The writer keeps the int8 surface's byte budget whatever the element width is. It writes
`ceil(oc/16)` contiguous blocks of `ow*oh_full` 16-byte atoms, exactly the int8 surface.
At four bytes an element, each block carries four channels where an int8 block carries
sixteen. Block `j` holds channels `32*(j/2) + 4*(j%2) .. +3`. What reaches DDR is
therefore the first eight output channels of every thirty-two, and nothing else is
touched.

That is one fact, not two. At `oc <= 8` every channel is delivered and the surface is
complete (`oc = 4` and `oc = 8` decode 32/32 and 64/64 with no waste). Past it the yield
is `oc/4`. A sweep run only at `oc = 32` cannot tell the block count from the
delivered-channel rule, and it reads as "the extent is `ceil(oc/16)` groups". The shapes
at `oc = 16` and `oc = 8` separate them, and both write two blocks.

The way round it is the weight cube, not a register. Program a multiple of the output
channels, and put real channel `n` in a slot the writer delivers, leaving the rest zero.
Every real channel then lands in a delivered slot, and the surface reads back as a plain
cube. The entry `rocket_matmul_int8_rk3576_i32()` does that, and it is bit-exact against
a CPU int32 model. On this narrow writer the multiple is four and the slot is
`32*(n/8) + n%8`, which collapses the map above to the plain int32 cube.

The cost is the output-channel axis, spent over. The bytes are not wasted: the budget
comes out at exactly the int32 surface. But the resident weight slice, the N tile and the
2944-channel bound are all functions of the programmed `oc`. The multiple is therefore
paid in MACs per submit. That cost follows from the measured budget rather than from the
encoding, so no register buys it back. Setting `PROC_PRECISION` halves the multiple, and
that is the writer the library uses.

### The wide writer

The wide writer delivers the first eight output channels of every sixteen, at half the
cost. A real channel costs two programmed ones rather than four.

`PROC_PRECISION` doubles the byte budget. DPU `0x4010`'s low field (`[2:0]`) is the DPU's
own operand width. Driving it from int8 to int32 makes the writer emit two 16-byte atoms
per (16-channel block, pixel) instead of one. Nothing else moves: the operands stay int8
everywhere, the arithmetic is bit-identical, and the CNA and CORE programs are
byte-unchanged. The generator `gen_conv2d_int8_rk3576_i32out_wide()` emits it.

Two atoms is the ceiling, swept across all eight values of the field at
`M=8, K=32, N=32`:

| `0x4010[2:0]` | extent | what came back |
|---|---|---|
| 0 int8, 6 int4 | 64 words | one atom per (block, pixel), the 4x rule |
| 1 int16, 4 int32 | 128 words | two atoms, the 2x rule, values correct |
| 2 fp16, 3 bf16, 5 | 128 words | two atoms, and the operands reinterpreted: no accumulator survives |
| 7 tf32 | n/a | every attempt was a dead submit |

The map works in 32-channel super-groups, each `4*A` atoms long, where `A` is the
surface's pixel count `ow*oh_full`. Inside one super-group the writer emits a single
linear stream indexed by `s = 2*p + j`. In that index `p` is the pixel and `j` the
16-channel block. The writer cuts the stream into runs of `A` atoms. The run also carries
the lane group `L = (c%16)/4`, which is the slower axis:

```
atom = 4*A*(c/32) + A*(2*(s/A) + L) + s%A       s = 2*p + j
word = 4*atom + c%4                             delivered iff c%16 < 8
```

Because it is a stream, nothing rounds: an odd `A` simply cuts it at an odd place, and
`A = 1` works as readily as `A = 16`. The map is decoded, not fitted. The operands were
drawn so that every accumulator in the tile is distinct. Each written word therefore names
exactly one (channel, pixel), or the map is not a map. It was read off at `oc`
8/16/24/32/40/48/64/96/128/192 and at pixel counts 4/5/6/7/8/12/13/16/21/33.

**Two bounds apply, and both fail silently.** The bounds, and the zero atoms found
inside them:

- **`oc` must be a multiple of 32**, because a partial super-group is not a truncated
  one. At `oc = 16 (mod 32)` the trailing group holds one 16-channel block instead of
  two, and packs at one atom per pixel with no stream. At an `oc` that is not a multiple
  of 16 at all, the delivered channel set rotates with the pixel. It then has no block
  form to write down. The case read off is 24. Take the count from
  `rocket_rk3576_pad_oc()`, and leave the padding zero.
- **A task's surface is bounded at 8 KiB, and the bound is over the plane.** Stated in
  the programmed channel count, a row task is correct while
  `iw * oh_full * oc_prog < 4096`. A wide task writes `oc_prog/8` atoms a pixel at
  16 bytes each. That product is therefore the output surface in half-bytes, and the
  bound is 8192 bytes, or 512 atoms, per task. The entry
  `rocket_matmul_int8_rk3576_i32()` enforces it by splitting the row plan until every task
  fits. A row task here is a standalone 1×1 convolution with its own surface, so honoring
  the bound costs submits and nothing else.

  **`iw` is in it, and leaving it out is a wrong answer rather than a lost bound.** A row
  split can only shorten `oh_full`. It cannot narrow `iw`, and the plane chooser takes the
  widest divisor of M it can. On a shape whose plane comes out `Mx1`, every task then
  satisfies a cap that divides only by `oc_prog`. The surface is still M times over the
  bound.

  Forced onto eleven shapes whose planner plane is `Mx1`, the wide writer was wrong
  on ten. At `iw = 1`, where the two readings coincide, all eleven are exact and every
  zero run disappears. A plane too wide for even a one-row task cannot be split into one.
  Such a shape cannot use the wide writer at all, and the entry refuses it.
  [HW sweep, H96 MAX M9]

  The cap was lifted (`ROCKET_RK3576_I32_WIDE_SURF`, `tests/rk3576_i32_height_bound.c`)
  for a sweep at `K = 32` and `ow = 1`. At every one of the eight programmed counts, the
  first wrong height is the smallest `oh_full` with `oh_full * oc_prog >= 4096`:

  | `oc_prog` | 64 | 128 | 192 | 256 | 320 | 384 | 448 | 512 |
  |---|---|---|---|---|---|---|---|---|
  | first wrong `oh_full` | 64 | 32 | 22 | 16 | 13 | 11 | 10 | 8 |

  Predicted and observed agree in all eight. [HW sweep, H96 MAX M9]

- **K relaxes the bound, and only ever outward.** The same `oc = 128` that is wrong from
  `oh_full = 16` at `K <= 128` is wrong at only 31, 32 and 40 at `K = 512`. It is exact
  to `oh_full = 40` at `K = 1024` and `K = 2048`. The 8 KiB bound is therefore the
  worst-case floor, reached once the contraction is short enough. Enforcing it
  unconditionally is what makes the path correct at every K.

  **That is also what hides `iw`.** A sweep of the plane width at `K = 512` finds `ow = 2`
  and `ow = 4` exact across the whole range in which `ow = 1` fails. That reads as "the
  plane width is not in the bound". The correct reading is "at this K the bound is far
  outside the geometries tried". At `K = 256` the same doubling breaks it:
  `oc_prog = 512`, `oh = 7` is exact at `ow = 1` and wrong at `ow = 2`. Sweep the width at
  a short contraction, or the axis is invisible.

- **Both writers drop atoms, and the drop tracks the memory system.** Inside the bound the
  failure is rare rather than gone. A row task comes back with a few atoms holding zero
  while the rest of the surface is exact. It is not the wide writer's alone. The narrow
  writer does it too, and without a per-atom check it does so silently. At
  `M=128 K=256 N=2048` it returned a handful of elements out of 262144 reading zero on six
  calls in twelve.

  The rate is a property of the memory system rather than of the program. Specifically,
  it tracks host DDR traffic, not the host merely being busy. Four load threads at
  `M=128 K=256 N=2048`, 30 reps and 120 row tasks per cell:

  | host load                          | row tasks that dropped an atom |
  |------------------------------------|-------------------------------|
  | quiet                              | 5.8%   (7/120)                |
  | integer spin, no memory traffic    | 11.7%  (14/120)               |
  | memcpy over 16 KiB, L1-resident    | 9.2%   (11/120)               |
  | streaming memcpy over 32 MiB, DDR  | **51.7%** (62/120)            |

  The two controls hold the CPUs equally busy and the cache hierarchy equally hot while
  leaving DDR alone. At this sample size they are not distinguishable from quiet
  (7, 11 and 14 of 120, against a binomial sd of about 3). Streaming to DDR is 5-9x all
  three and far outside that. The effect is therefore specific to traffic reaching DDR,
  and CPU occupancy and scheduling latency are ruled out as the mechanism. The count of
  atoms never emitted at all, the wide-output poisoning, barely moves across the same
  contrast, which makes the two hazards separate. [HW sweep, H96 MAX M9,
  2026-07-28, `tests/rk3576_i32_zero_run.c load`]

  The drop does not reach the conv entries or the narrow int8 matmul. At the load that
  takes the int32 writers to 51.7%, `rk3576_conv_lib_gate` is 156 passed / 7 refused /
  0 failed. It reads the same under DDR pressure, under the L1 control and quiet. Under
  DDR pressure `rk3576_matmul_gate` is 46/46. [HW sweep, H96 MAX M9, 2026-07-28,
  `tests/hostload.c`]

  One 155/156 flake appeared under the *spin* control, which touches no memory. The same
  single-case flake also appears with no load at all. It is therefore a pre-existing
  intermittent in that gate, not a load effect, and it is not pinned to a case. The conv
  entries' own "did it write" check is per row task, and cannot see a task that wrote all
  but a few atoms. This is therefore a measured negative rather than a covered one.
  [HW sweep, H96 MAX M9, 2026-07-28, `tests/hostload.c`]

- **Every wrong element comes back zero**, past the bound and inside it alike. Over the
  whole map, eight channel counts, 57 heights each, not one wrong element aliases
  another element's correct value and not one holds anything else. The writer emits empty
  atoms rather than misplacing full ones.

- **A dropped atom cannot be told from a zero accumulator by looking at the surface**, so
  do not try. The bytes are identical, and repetition does not separate them either. The
  drop is correlated with the memory system. It repeats often enough to leak a wrong
  answer through a "came back the same twice, so it is data" rule. What separates them is
  the arithmetic.

  The library stamps a sentinel, so "never emitted" is a fact per atom. It redoes the
  task once, which clears most drops. From the second attempt it computes the zero atoms'
  accumulators on the CPU over that K slice. It keeps the surface only if they really are
  zero. At the worst shape all 240 reps were exact, 120 of them under host load. The
  all-zero-data case (every row of A zeroed) passed on both writers.

**A probe that drives every output channel finds a boundary the library never meets.**
With all sixteen channels of a group live the map stops fitting well before the bound
above, at `oc = 192` from `A = 48` on a flat plane. Under the scatter, the undelivered
half of each sixteen is zero. The same geometries are then bit-exact through the library
at `A` 48, 64 and 128. A boundary found with every channel live is therefore a boundary of
the probe. Setting `ROCKET_MP_SCATTER=1` drives the probe the way the library drives the
part.

#### Cost and the writer choice

The N tile is bounded by the programmed output channels, so halving the multiplier
doubles the tile and halves the submits. On this path a submit costs an idle rather than
1.4 ms. Measured through `rk3576_matmul_gate`:

| shape | 4x writer | 2x writer |
|---|---|---|
| M=64 K=1024 N=2048 | 4 submits, 686 ms | 2 submits, 339 ms |
| M=32 K=1024 N=4096 | 8 submits, 1319 ms | 4 submits, 675 ms |

Below the tile cap the two cost the same, so a shape whose N fits one tile shows no gain.

The planner picks the writer per shape, and `ROCKET_RK3576_I32_OC_MULT` forces either.
The whole benefit is submits and the whole cost is submits, and both are computable
before anything is submitted. The function `r76_i32_submit_count()` counts K slices x
N tiles x row tasks for each writer. The surface cap splits the wide writer's tasks
further. The cheaper writer wins, and a tie goes to the narrow one.

**With the bound stated over the plane, the wide writer does not win.** The cap costs it
exactly the submits its wider N tile was reached for. It costs them on the N-heavy shapes
where the tile would have paid. Over 315 natural shapes, M 4-256, K 64-2048 and
N 32-2048, the chooser took the narrow writer on every one. [HW sweep, H96 MAX M9]

Figures that show the wide writer ahead (8 submits to 4 at `i32-n4096`, 4 to 2 at
`i32-n2048`) were measured against the rows-only cap. That cap let those shapes run one
task per tile at many times the surface the writer can hold. The wide path is kept for two
reasons. It is the instrument that decoded the 32-bit writer's map, and a shape outside
that sweep can still reach it. The default path does not use it. [HW sweep, H96 MAX M9]

#### Zero atoms and their detection

Sparse data does not cost either writer its retry budget. With every row of `A` zero and
the bias zero with them (`ROCKET_MG_ZERO_ROWS=1`), both writers pass the whole int32
gate. A whole-task check, "did this task write anything at all", answers "no" on every
legitimately zero surface. The narrow path asks a per-atom question against a sentinel
instead. A zero atom that survives one redo is settled against the operands rather than
against the surface.

**Both writers intermittently emit zeros**, which is what the detector exists for. On the
wide writer the block is contiguous and one super-group wide. On the narrow writer it is a
handful of atoms. It is the same defect, and it is not rare: see the memory-pressure
measurement above.

**It does not drop writes.** With the output BO stamped with a sentinel before the tasks
run, the bad atoms come back zero rather than holding the sentinel. The stamp is
bracketed by `PREP_BO` and `FINI_BO`, so no dirty line is left to race the DPU's DMA. It
is verified to reach DDR before any submit. The writer reaches the bad atoms, and the data
in them is wrong. On a fresh BO, which arrives zeroed, an unwritten atom and a zeroed one
are indistinguishable, and a wrong zero reads as a dropped write.

Two other causes are excluded. It is not a readback race: a second fence and a second
read return byte-identical contents. It is not the poisoning below either. The
poisoning's signature is an *empty* region, and cycling the power domain clears it. This
defect is indifferent to the idle ahead of it, and host memory traffic moves its rate
fourfold while leaving the poisoning's count alone.

The corrupt block reads most clearly in the wide writer's own coordinates. The writer
emits two atoms per stream position `s`, one per lane group, so its emission order is
`(s, L)` ascending. The corrupt block is contiguous in *that* order and in no other. In
address order it shows up as two separate holes, and it lies inside a single super-group.

**Its extent does not generalize across shapes**, which is why no detector keys on it. On
one shape, every failure over 40 runs ended at `s = 2A-2`, two emissions short of that
group's last. Other shapes break that pattern: logged at debug level across shapes
(`r76_i32_run_log()`), `i32-tall-m` and `i32-m100` corrupt three emissions in mid-stream.
They sit at `s = 1039..1040` of 0..2047 and at `s = 15..16` of 0..199. Both recur at a
fixed address, so a tail-keyed detector would miss all of it. [HW sweep, H96 MAX M9,
25 runs per shape]

The library therefore detects and settles the defect rather than avoiding it, and the
settling is exact rather than heuristic. The signal is two adjacent zero emissions on the
wide path, or any zero atom on the narrow one. The library redoes the task once, which
clears most of it. From the second attempt it computes the zero atoms' accumulators on the
CPU over that K slice. It keeps the surface only if they really are zero. The functions
are `r76_i32_wide_suspect()` and `r76_i32_zeros_are_data()` in `rocket_matmul_rk3576.c`.

The redo needs no idle in front of it, which is further evidence this is not the
poisoning. The sentinel's other reading, an atom that still holds it, *is* the poisoning,
and that redo keeps its idle.

#### Registers eliminated

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
| CORE `0x3020` (dataout_channel) x4 alone | **0 words** | killed the write outright. It is coupled to the others, not inert |
| all five oc carriers x4 together | **256**, the full surface | the byte budget scales, the delivered channels do not, still 8 |

The delivered-channel rule therefore survives every oc register on the part. Scaling them
together buys address extent and nothing else, which places the rule in the datapath
rather than in a geometry field. [HW sweep, H96 MAX M9, measured 2026-07-27]

### The dense writer: every accumulator as an int32, and no poisoning

Eight DPU words make the int8 direct program write every accumulator as a raw int32.
Emitted together over it, one task delivers all `M*N` accumulators exactly, in `M*N` words,
with no bias added. Seven are the `charsiu` project's wide output stage, read from vendor
streams (`src/job.c`), and its `0x40B8` is theirs too, fitted there at one row:

| register | int8 direct | dense int32 |
|---|---|---|
| `0x4010` | `0` | `0xa0000002`, `out` 5 and `proc` 2 |
| `0x4030` low half | `0x0710` | `0x0310` |
| `0x4038` | `0x00120080` | `0x00000053` |
| `0x4044` | `1` | `2` |
| `0x4050` | `0x80011111` | `0x00023333` |
| `0x40AC` / `0x40B0` / `0x40B4` | the requant | `0` / `1` / `0` |
| `0x40B8` | `2*S - ow*oh_task` | `3*S` |

`S` is the task's pixel count `ow*oh`, and `0x401C` stays at `S`. [HW sweep, H96 MAX M9,
rocket 1.6.0, `tests/rk3576_matmul_probe out32`, 2026-09-30]

The map is the wide writer's stream with all four lane groups delivered. With `A` the
pixel count, `g = c/32`, `j = (c%32)/16`, `L = (c%16)/4` and `s = 2p + j`:

```
atom = 8A*g + 4A*(s/A) + A*L + s%A
word = 4*atom + c%4
```

It decodes as a bijection, every accumulator distinct, at `N` 32, 64, 96 and 128. The planes
run from 3×2 to 12×10, `M` 4-120, and `K` 32-4096 on random operands. The programmed `oc` is
a whole number of 32-channel groups, the int8 program's own rule. A 16-channel trailing group
computes wrong past `K` 32. One task writes a 38 KiB surface exactly at `K` 32, so the wide
writer's 8 KiB bound does not hold there. A row window's `0x40B8` is not measured.

**Doubling `0x401C` and `0x40B8` instead is exact at 32 channels and overlaps past them.** The
four quad planes then sit twice as far apart, so one 32-channel group lands whole, and the next
group's start lands one plane in. A count read off an overlapping map is a write race: the same
registers read 160 and 224 of 256 in two runs.

It does not poison: see [rk3576.md](rk3576.md) §"The poisoning is one hazard".

### Poisoning after an i32out job

**After an i32out job, the next submit of any kind writes nothing, until a power cycle
clears the state.** The job that does not write **completes normally**. It raises no
fault, no IOMMU error and no timeout, takes the usual per-submit time, and leaves its
output buffer untouched.

If the caller zeroed that buffer, it reads as an all-zero surface. If the caller reused
one, it reads as the previous result, byte for byte. Either way it looks like an
arithmetic failure, and it is not one. It crosses processes: the plain int8 matmul in
another program is as exposed as the path that caused it. The state lives in the NPU and
not in the fd.

**What clears it is the driver's runtime-PM autosuspend cycling the NPU power domain,
not elapsed time.** Three measurements say so, and together they close it:

- With `on` written to the device's `power/control`, runtime suspend never fires. No gap
  then clears the state: 100 ms, 300 ms and 600 ms each write nothing.
- The working gap tracks `power/autosuspend_delay_ms` one for one. At a 5 ms delay a
  5 ms gap already gives 12 writes of 12. At 50 ms it takes 80-120 ms for the same.
- `power/runtime_suspended_time` advances across the gap, so a suspend did happen.

At stock settings, 50 to 100 ms of idle clears the state. That is the file's stock value
of 50 plus the suspend/resume round trip, and nothing about the silicon. The cost of the
workaround is therefore settable: lowering the delay shortens the path proportionally.
The matmul gate's int32 half runs in 553 ms at the stock 50 and in 195 ms at 5. A
kernel-side reset of the DPU at job start would remove it outright.

**It is one hazard and every wide output carries it.** It is measured with a canary
(`tests/rk3576_poison_probe.c`). The probe submits the program under test, then a program
known to write and known not to poison, and scores the canary. A dead program and a
poisoning one therefore stay distinguishable.

Over seven program kinds the split is exact. The fp16 first conv, the fp16 direct conv
and both int32 writers each poison every one of the seven. The int8 first conv, the int8
direct conv and the int8 depthwise poison none. Every program under test wrote its own
surface in every cell, so no cell is a dead program reading as a clean one. One output
byte never carries it, and a wider output always does.

**A sacrificial submit does not absorb it.** After one int32 job, six consecutive int8
jobs at a zero gap all wrote nothing. The state persists until the domain cycles, so the
guard cannot be a cheap dummy job.

**The cycle clears it about 87% of the time, so the number of redos makes the guard
reliable, not a better cycle.** Over twelve fp16 conv gate runs the redo fired on 358 row
tasks. The attempt that failed *next* was attempt 2 for 45 of them, 3 for 10, 4 for 2,
5 for 1 and 6 for none. Every one recovered. Every cycle in between was confirmed to have
taken the domain to `suspended` first, so the guard was never merely unobserved.
[HW sweep, H96 MAX M9, 2026-07-28]

A cap of four attempts leaves about 0.6% of retried tasks failing outright. That is
exactly the rate the conv gate saw as a "1-in-156 intermittent". At eight the failure is
unobservable (15 of 15 fp16 runs and 3 of 3 full runs clean, deepest retry attempt 4).
Cap the redos at eight. [HW sweep, H96 MAX M9, 2026-07-28]

**A task that reads unwritten really is unwritten**: the fence is not signaling ahead
of the writes on this path. Asking again after a 2 ms settle before the redo rescued
0 of 67 such tasks, while the power cycle behind it recovered all 67. The re-check
costs 14% of the fp16 wall and buys nothing (`ROCKET_RK3576_DRAIN_US`, default 0).

#### Registers that carry it

DPU `0x4010` (the DATA_FORMAT word, `out<<29 | in<<26 | proc`) carries it on the int32
path, and no single register carries it in general. The i32out program differs from the
int8 direct conv in only four registers: `0x4010`, `0x40AC`, `0x40B0` and `0x40B4`.
Zeroing `0x4010` alone stops the poisoning with the program still writing, where the
other three leave it. But the sufficient condition is joint, not one register:

| program | `0x4010` forced to | poisons? |
|---|---|---|
| int8 direct | `0` (its own) | no |
| int8 direct | `out`=1, 2 or 3, any `proc` | no |
| int8 direct | `out`=4 or 5 | yes |
| i32out | `0` | no |
| i32out | any non-zero, including `out`=1..3 and `proc` alone | yes |
| fp16 first conv | each of its 19-register delta, one at a time | yes, all 19 |

The same `out` value is therefore clean on one program and poisoning on another, and a
leave-one-out over the whole fp16 first-conv delta revives nothing. The mechanism is
undecoded. Its scope and its price are decoded. The next job programs `0x4010` back to
all-int8 and still comes back empty, so this is latched datapath state rather than a
stale register.

Nothing about the program or the buffers changes it. Each of these was tried, and none
helps:

- Fresh buffers per submit
- A different feature address
- A different output address
- Packing every K slice up front, so no buffer is rewritten between submits

The entry `rocket_matmul_int8_rk3576_i32()` sizes its idle from the driver's autosuspend
delay (twice it plus 20 ms). It idles between its own submits and once more on the way
out, and retries any submit whose own region came back untouched.

**Check that a job wrote per task, not per tile.** A row-split tile issues several
submits into one output buffer. One poisoned submit among them leaves its own rows empty
while its siblings are full. A check over the whole tile then reads "something was
written" and hands the hole to the caller.

The hole shows up as a result that is exactly one row task short, intermittently, with no
fault and normal timing. Both writers need the per-task check. It was found on the
narrow writer, which ran clean for a session under gate shapes that never split a tile
into row tasks.

#### The guard's cost

The cost is settable from the driver, and the law is twice the delay. A 5×7 sweep sets
`power/autosuspend_delay_ms` against the inter-submit gap. It drives a chain of 20 fp16
first-conv submits and counts how many wrote:

| `autosuspend_delay_ms` \ gap ms | 0 | 5 | 10 | 20 | 40 | 60 | 100 |
|---|---|---|---|---|---|---|---|
| 50 (stock) | 1 | 1 | 1 | 1 | 1 | 9 | **20** |
| 20 | 1 | 1 | 1 | 9 | 19 | 18 | **20** |
| 10 | 1 | 1 | 11 | 18 | **20** | 20 | 19 |
| 5 | 1 | 7 | **20** | 18 | 20 | 19 | 20 |
| 1 | 1 | 19 | 19 | 18 | 19 | **20** | 20 |

So `rocket_rk3576_power_idle()` reads the delay and sizes the guard at twice it plus a
small margin.

**Lowering the driver's delay system-wide is not free, and a back-to-back chain cannot
see the cost.** A resume is about 230 us. Lowering the delay makes every job whose
inter-job gap falls between the new delay and the old one pay it. Measured with
`ROCKET_SF_GAP_MS` on `rk3576_submit_floor`, us/submit:

| inter-submit gap | 0 | 5 | 10 | 16 | 20 | 33 | 60 | 100 |
|---|---|---|---|---|---|---|---|---|
| delay 50 ms | 1062 | 1123 | 1130 | 1124 | 1136 | 1139 | 1283 | 1399 |
| delay 1 ms | 1059 | 1319 | 1334 | **1372** | 1352 | 1368 | 1366 | 1378 |

At a zero gap the two are identical, because the autosuspend timer never fires. That is
why the plain floor probe reads the change as free. They converge again past 60 ms, where
both suspend anyway. In between, the low delay costs ~20% a job. **A 60 Hz frame interval
of 16.7 ms sits squarely in that window**, and that is exactly the pipeline mainline's
50 ms was chosen for.

So drive the transition rather than waiting for it. The library's
`rocket_rk3576_power_idle()` writes a zero delay, polls `runtime_status` until it reads
`suspended`, and puts the caller's delay back. The guard then costs one real power cycle
instead of a worst-case idle. The steady-state policy every other consumer of the NPU
sees is unchanged.

The call needs write access to the sysfs file. It is probed once, and without that access
the call falls back to the plain idle. Setting `ROCKET_RK3576_PM_KICK=0` turns it off. A
kill during the few milliseconds in which the delay sits at zero leaves it at zero. That
state is self-healing. Zero is not a value a system sets deliberately, so it is read as an
interrupted forced transition and restored to mainline's 50.

Measured at the stock 50 ms delay, every gate stays bit-exact (conv 102, conv-lib 154 + 8
refusals, matmul 43 on both writers, refusal 15, regcmd 194, both fp16 sweeps). The fp16
first conv's 224×224 k3 s2 runs in 13.75 ms against 226. Its whole gate group runs in
2.3-15.6 ms against 107-334, and the int32 matmul gate in 9.9 s against 16.2. That beats
lowering the global delay to 1 ms, which reached only 21.6 ms on the same shape and taxed
every gapped workload to get there. [HW sweep, H96 MAX M9, measured 2026-07-27]

#### The driver side

The completion-visibility race is closed by `patches/rk3576/npu/0012`, which retires the
poll on the DPU's own completion bit rather than on `PC_DONE`. The reproducer that found
it, and the shape of the fix, are under "The per-submit floor is the driver's completion
poll" above. The module parameter `poll_interval_us` is live, so the period still sweeps
from sysfs with no rebuild. That is how the sweep against correctness rather than wall
time was run, and how a candidate fix is tested.

The kernel-side fix that would remove the poisoning idle entirely is identified but not
built. The driver already has `rocket_core_reset()` (a `reset_control` assert,
`udelay(10)`, deassert), and calling it at job start is the shape of the fix. The vendor
`rknpu` driver is not the model here. It soft-resets only on a timeout or an abort, never
per job.

The per-submit floor on this part is the driver's poll, not the silicon. PC_DONE is not
routable to the GIC on the RK3576, so `patches/rk3576/npu/0006` polls it where the RK3588
takes an interrupt. Every RK3576 cost table is a submit-count table because of that.

#### Related traps

**The hazard reaches the plain int8 matmul too.** The poisoning is not confined to the
int32 entry that creates it. It outlives the call and the process, so
`rocket_matmul_int8_rk3576()` inherits it from whatever ran before. Its first submit then
comes back untouched while the caller reads a correctly sized, entirely stale tile.
[HW sweep, H96 MAX M9, measured 2026-07-27]

Without the stamp, one full matmul-gate run in twenty failed its very first shape this way
in a soak, at 1 element of 256 correct. The entry stamps its output surface and redoes any
row task whose region is still the stamp. With the stamp, 20 further gate runs were clean.
[HW sweep, H96 MAX M9, measured 2026-07-27]

**Do not zero an output buffer from the CPU before a submit.** A bracketed stamp is a
different thing, and is worth having. A bare `memset` leaves dirty cache lines that race
the DPU's DMA, and the writeback lands on top of the result. The surface then comes back
all zeros, intermittently, and on the first submit as readily as a later one.

A fill bracketed by `PREP_BO` and `FINI_BO` is written back before the submit and leaves
nothing dirty. Both matmul entries use one. It makes "this was never written" a property
of the surface instead of an inference from its value. A freshly allocated BO arrives
zeroed, which cannot distinguish an unwritten word from a legitimate zero. A guard band
past the surface needs 64-byte alignment, so that no line it dirties is shared with a
surface byte.

The int32 sweep also **wedged the IOMMU** once (stall timeout plus a
`rocket_job_irq_handler` warn at `rocket_job.c:528`), and only a reboot cleared it. Do not
read that as an overrun. The measured extent sits well inside the BO that was allocated,
so the mechanism is unestablished. Treating it as a buffer-size problem is a false trail.
[HW sweep, H96 MAX M9, measured 2026-07-27]

### The instrument

The probe `tests/rk3576_matmul_probe.c` has these modes:

- `shape`: the mapping against a CPU model
- `m`: the M axis, every factorization
- `cost`: both precisions on one shape
- `peak`: MACs per submit by N
- `ncap`: the N boundary, with repeats and a failure classification
- `out32`: the 32-bit output width, decoded

Its knobs are `ROCKET_MP_I32`, `ROCKET_MP_IW`, `ROCKET_MP_KS` / `_NS` (comma lists),
`ROCKET_MP_REPS`, `ROCKET_MP_GAP_MS`, `ROCKET_MP_OSLACK` and `ROCKET_MP_VERBOSE`.

Two traps apply to measuring with it. **`rocket_submit_matmul()` returns before the job
does**,
so timing it without fencing on the output BO reports 0.0 ms. **Pace between runs, not
just between tasks, and keep the pacing outside the measurement.** An unpaced job
completes without writing at all, which reads as an arithmetic failure and not as a
scheduling one.

## The DPU LUT

The LUT's table is reachable on this part: a manufactured capture drives it. Every convolution
capture leaves the LUT bank (`0x4100`-`0x4194`) zero, which is why `npu_regcmd_rk3576.c`
writes the whole bank as zeros. That zero bank is a property of the captures, not of the
silicon. An ONNX model carrying a nonlinear activation, compiled for `rk3576`, emits the
table load verbatim (`tests/data/rk3576-vendor-capture/lut/mklut.py`, decoded by
`decode_lut.py`).

A nonlinear activation is a separate program, not a fused epilogue. The compiler emits two
programs per conv-plus-activation: one that is DPU + DPU_RDMA only, with no CNA and no CORE,
and one ordinary convolution. The DPU-only program drives a 1×1×16 dummy cube
(`0x401C = 1`, `0x4030 = 0x000f0f00`, `0x40B0 = 0x00010001`). Its work is loading the table,
not computing. The convolution that follows it computes, and it is configured to use the
table. Against a bare-conv control, whose whole LUT bank is zero, it writes
`LUT_CFG 0x4108 = 0x02000006`, `LUT_INFO 0x410C = 0x00050500` and all four of
`LE_START`/`LE_END`/`LO_START`/`LO_END` (`0x4110`-`0x411C`) = `0xffffc000`.

The table is written through a two-register window that auto-increments. The
`LUT_ACCESS_CFG` register (`0x4100`) selects the table and the start offset. Every
subsequent write to `LUT_ACCESS_DATA` (`0x4104`) stores one entry and advances. The load
program writes `0x4108 = 0x00ff0000` first, then:

| `0x4100` | entries | what |
|---|---|---|
| `0x00020000` | 513 | one table |
| `0x00030000` | 513 | the other |

Each table holds 513 = 2^9 + 1 entries, so it is 512 intervals with both endpoints, the
shape a linear-interpolating LUT wants. The two tables are the same LE/LO pair the NVDLA
SDP documents.

The two tables are the lower and upper halves of one monotone curve, joined at the value
the function takes at the split. The entries are Q15 of the function's output. For
`Sigmoid` the entries run `0x3b -> 0x4000` in the `0x00020000` table and `0x4000 -> 0x7fc4`
in the `0x00030000` one, and `0x4000/0x8000` is exactly 0.5. For `Tanh` they run
`-0x7f63 -> 0` and `0 -> 0x7f63`, symmetric about zero as it must be.

The table is uniform in the input, and each function's span in its own units is the
compiler's choice. The span is recoverable from the first difference and the function's
own derivative at the join. Sigmoid's `d = 100` at the join with `sigmoid' = 0.25` gives a
step of 0.0122 and a half-span of 6.25. Tanh's `d = 193` with `tanh' = 1` gives a step of
0.00589 and a half-span of 3.02. Both agree with the endpoint values to three digits.

The two spans differ because the compiler puts the table where each function saturates.
It expresses that choice by scaling the value the datapath hands the LUT, not by moving
the LUT's own window.

The register pairs `0x4188`/`0x418C` and `0x4190`/`0x4194` are the output clamps: the value
used below the table and the value used above it. Each is replicated across 16-bit lanes
and equals the table's first or last entry. Four functions verify it at once, each
matching its own table's endpoints exactly:

| function | below the table | above the table |
|---|---|---|
| sigmoid | `0x003b` | `0x7fc4` |
| tanh | `0x809d` | `0x7f63` |
| softplus | `0x0094` | `0x7fff` |
| swish | `0xff9d` | `0x7fff` |

Two register groups, `0x4150`/`0x4154` with `0x4160` and `0x4170`/`0x4174` with `0x4184`,
are the underflow and overflow linear-extrapolation slopes (a scale and a shift each). The
four functions' tails predict their own values:

- `tanh` saturates both ends and carries zero in all four.
- `sigmoid` carries a small lower slope and zero above.
- `swish` carries zero below and about 1.0 above.
- `softplus` carries a small lower slope and about 1.0 above.

That is exactly the pair of tails each function has. [Manufactured capture]

On a LUT case, `0x40AC`/`0x40B0`/`0x40B4` are the LUT's output requant, not its input map.
They are the DPU's ordinary output converter, which the piecewise activations and the bare
conv write too. On a fused nonlinear activation they carry Q15 to the op's own
quantization exactly. [Manufactured capture]

Sigmoid's output is `[0, 1]` at zero point -128, so its gain is `255/32768 = 7.7820e-3`,
and the registers read `0x7f81 >> 22 = 7.7815e-3`. Tanh's output is `[-1, 1]` at zero
point 0, so its gain is `127.5/32768 = 3.8910e-3`, against `0x7f81 >> 23 = 3.8907e-3`.
The multiplier is the same with one more shift, and `0x40AC` is -128 for sigmoid and 0 for
tanh. Five digits on two functions is not a coincidence, and it means these three
registers say nothing about which interval an input lands on. [Manufactured capture]

The input map is `index = (value - LE_START) / 2^sel`. The probe
`tests/rk3576_lut_probe.c` reads it off the part with this project's own table-load
program and a table whose entries encode their own index. A convolution drives the table.
Its accumulator is one feature byte, and its per-output-channel `A` and `C` place that
byte anywhere in the datapath:

| register | field | value the vendor writes | what it does |
|---|---|---|---|
| `0x4110`-`0x411C` | `LE_START`, four lanes | `0xffffc000` = -16384 | where the LE table starts |
| `0x4140`-`0x414C` | `LO_END`, four lanes | `0x00004000` = +16384 | where the LO table ends |
| `0x410C` | `LUT_INFO`, two index selects | `0x00050500` | the index step, `2^5 = 32` |

The `0x00020000` table (LE) covers `[LE_START, 0]` and the `0x00030000` table (LO)
covers `[0, LO_END)`, 512 intervals each. All three registers are live. The map is
bit-explained at `sel` 4, 5 and 6, and at an asymmetric `LE_START = -8192` with
`LO_END = +16384`. Each case runs 8192 samples per span, with a maximum disagreement of one
output count (`rk3576_lut_probe gate`, 4 of 4).

The vendor never moves these registers, because it places the value inside the fixed
window with the BS stage's own per-channel `C` instead. That is why no diff of vendor
programs can produce this map. At fp16 the scale rides the coefficient buffer rather than
a register. That is also why sigmoid and tanh emit identical programs there, with tables
spanning 6.25 and 3.02.

The hardware interpolates linearly between entries, at the full resolution of the step.
The probe walks a step table one datapath unit at a time across the interval it steps in.
That gives 33 distinct output runs over 32 units, rising linearly, not two.
[HW sweep, H96 MAX M9]

The domain is half-open at the top. An input of `value == LO_END` takes the overflow
clamp, not the last table entry. An input of `value == LE_START` is in the domain and
reads `LE[0]`. Every disagreement in the first run of the gate, at four different `C` and
all four spans, sat on that one value and nowhere else.

**BN_CFG `0x4060` is a trap: the vendor's value hides the input map.** Every vendor
activation carries `0x20` there, the BN stage active, against `0x903` in its own
bare-conv control. Transcribed, that value makes the LUT return the value at the table
join for every input from -2^21 to +2^21. The surface is perfectly constant and tracks
the table faithfully as the table changes. It therefore reads as a working LUT with an
input map that is nowhere in the registers.

The cause is that BN then multiplies by an operand register no vendor program writes, and
this register file is not cleared between jobs. Left at the generator's own `0x903`, the
BN stage is bypassed, the LUT reads the BS output directly, and the map appears.

The four registers the vendor's own programs leave alone (`0x4040`, `0x4054`, `0x4064`,
`0x4068`) were each driven, and none of them moves it. Where this part keeps
`BN_MUL_OPERAND` is therefore open. It does not need to be found: the BS stage's
per-output-channel `C` is a per-channel scale, where a BN multiply would be one global
one.

The registers that turn the LUT on come from a diff against the bare-conv control of the
same capture set. A diff across activations hides them:

| register | control | LUT | |
|---|---|---|---|
| `0x407C` | `0x010041C1` | `0x01004140` | EW_CFG: `EW_LUT_BYPASS` (bit 7) and `EW_BYPASS` (bit 0) clear, so the LUT is the EW stage |
| `0x4108` | 0 | `0x02000006` | LUT_CFG |
| `0x410C` | 0 | `0x00050500` | LUT_INFO |
| `0x4110`-`0x411C` | 0 | `0xffffc000` | LE_START |
| `0x4140`-`0x414C` | 0 | `0x00004000` | LO_END |
| `0x4188`-`0x4194` | 0 | the tails | the two clamps |
| `0x5028` | 0 | `0x1A` | DPU_RDMA NRDMA_CFG |
| `0x1004`/`0x3004`/`0x4004`/`0x5004` | `0x0E` | `0x30` | every block's S_POINTER |
| `0x4060` | `0x903` | `0x20` | BN_CFG, **do not transcribe this one** |

That table is what `gen_conv2d_int8_rk3576()` emits from `conv_params_t.lut`
(`lut_rk3576_t`: `le_start`, `lo_end`, `sel`, and the two clamps). The generator writes
the whole bank either way, with zeros when the field is NULL. This register file is not
cleared between jobs, so the full write keeps a stale window from the previous job out of
this one. With `lut` NULL, every program is byte-identical to the ones emitted before the
field existed, which `regcmd_rk3576_gate` asserts.

The generator packs the two clamps as the vendor packs them. The first register of each
pair carries the value alone in its high half, and the second carries it doubled across
both halves. For the low clamp, `0x4188` = `lo << 16` and `0x418C` = `(lo << 16) | lo`.
The high clamp takes the same form at `0x4190`/`0x4194`. The generator leaves `0x4184` at
zero.

The EW field layout is the RK3588's and transfers unchanged (`EW_LUT_BYPASS` at bit 7,
`EW_BYPASS` at bit 0), which is what the two values decode to. The register offset does
not transfer: `0x4070` on the RK3588, `0x407C` here.

`gen_lut_load_rk3576()` emits the table-load program, 1121 words as the vendor's is:

- The 1×1×16 dummy cube.
- The `0x4100 = 0` / `0x4104 = 0` / `0x4108 = 0x00ff0000` preamble.
- The two 513-entry bursts.
- The four-word trailer.

The trailer's `PC_OPERATION_ENABLE` is **`0x18`**, DPU and DPU_RDMA, neither the
convolution's `0x1D` nor the pool's `0x60`. Its dummy cube writes nothing at any address,
so the load program is not its own positive control. What says the table loaded is that a
consuming convolution's readout tracks the table.

Put the load and the use in one job. The table lives in the LUT RAM, and a separate
submit can take a runtime-PM cycle in between.

## The PPU pooling program

Pooling on this part is its own NPU program: an independent 31-write program, 23 PPU
writes and 8 PPU_RDMA. Nothing in any found capture drives the pooling engine. Manufactured
captures do (`tests/data/rk3576-vendor-capture/pool/mkpool.py`). A `MaxPool` or
`AveragePool` behind a convolution emits that program, and a bare pool emits exactly the
same program with no convolution beside it. It works in the same NC1HWC2 cube the
convolution path already packs, so it needs no new host packing.

The vendor compiler lowers `GlobalAveragePool` to convolutions, not to pooling. Every
program in that capture is CNA + CORE + DPU, the same lowering the RK3588 uses for a
reduce. The compiler lowers `GlobalMaxPool` to pooling, in two cascaded PPU passes (k7 s7,
then k3 s3 over what is left).

A sweep over method, kernel, stride, plane, channel count and padding gives the meaning of
the 23 PPU writes:

| reg | meaning |
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
| `0x6040` | four pad nibbles, `right\|left\|bottom\|top` |
| `0x6044`-`0x6050` | pad values: `-128` (as `0x0007ff80`, sign-extended in a 19-bit field) for max, and the input zero point times 1, 2, 3, 4 for average |
| `0x607C`, `0x6084` | `round4(ow*oh) * 16`, the destination surface stride per 16-channel group, the same round-to-four the convolution's `0x401C` takes |
| `0x6054`, `0x6058`, `0x605C`, `0x6070`, `0x60DC` | zero in every capture |

The input extent is what the windows consume, not the plane. The registers
`0x600C`/`0x6010` carry `(ow-1)*sx + kw` and `(oh-1)*sy + kh`, clamped to the plane. A
15×18 plane pooled k2 s2 therefore programs 18×14 and not 18×15: the last row, which no
window reaches, is not described. Every non-square and odd-plane case in the sweep agrees,
and a plane-height reading does not.

The eight PPU_RDMA writes are the source side, fitted over the same sweep:

| reg | meaning |
|---|---|
| `0x7004` | `0x0e`, the same enable word |
| `0x700C` / `0x7010` | input width / height consumed, minus 1, the PPU's own `0x600C`/`0x6010` again |
| `0x7014` | input channels minus 1, rounded up to 16 |
| `0x701C` | source base address |
| `0x7024` | DDR line stride in bytes, `iw*16`, the full plane, not the consumed extent |
| `0x7028` | channel-group surface stride, `round4(iw*ih)*16` |
| `0x7030` | `0x40` in every capture |

The strides are the only place the full plane appears. The line stride `0x7024` is what
separates them from the consumed extent: a 19-wide plane pooled k2 s2 consumes 18 and
strides 19.

The consumed extent excludes the leading pad, so it is
`min(plane, (ow-1)*stride + k - pad_start)` and not the windows' raw span. A VALID k3 s2
over 16 programs 15. A SAME k3 s2 over the same 16 programs 16, where its windows span 17.

The end pads sit in the low nibbles of `0x6040`. A SAME k3 s2 over 16 needs
`pad_right = pad_bottom = 1` and nothing on the leading edges. The capture carries
`0x0011`, so the low half is right and bottom. Every other captured pad is symmetric and
cannot tell the two orders apart. Which of the low pair is right and which is bottom is
open, and the two are equal in every shape emitted so far.

The destination base address is `0x6070`, and no capture can show it. Like every other base
on this part, the vendor runtime patches it at load time, and the stored program carries
zero. The address was read off the part by driving, in turn, each PPU register that reads
zero in every capture. Those five are 0x6054, 0x6058, 0x605C, 0x6070, 0x60DC. Each run
asked only whether the output BO moved off its sentinel, and only 0x6070 writes.

**`PC_OPERATION_ENABLE` is a per-block bitmap, and this is the trap.** A convolution
enables its blocks with `0x1D`. The vendor's pooling program enables PPU and PPU_RDMA with
`0x60`, and the two bit sets are disjoint. A pool carrying the convolution's trailer is a
fully configured PPU that is never started. The job completes in the usual per-submit
time, faults nothing and writes nothing. By inspection, that is indistinguishable from a
wrong destination address.

One whole register sweep came back empty on this fault. Reading the raw capture word by
word, not through the block classifier, is what found it. **Transcribe the whole program,
trailer included.**

The average rounds half to even, the same rule the DPU's `OUT_CVT` was measured to use.
Against round-half-away-from-zero, a k2 average disagrees on one output in eight. The
divisor is the window (`0x6038`/`0x603C` are `1/kw` and `1/kh`), not the tap count. Mode
`0x18` excludes the pad from the divisor.

With those fields, `gen_pool_rk3576()` computes bit-exactly against a CPU model over 11
shapes in `tests/rk3576_pool_probe.c`. The shapes cover max and average, kernel 2/3/5,
stride 1/2/3, non-square and odd planes, 8/32/64 channels, and padded and unpadded input.
[HW sweep, manufactured capture, H96 MAX M9, 2026-07-29]

`rocket_pool_int8_rk3576()` is the library entry over it. It takes row-major `[C][IH][IW]`
in and out, and owns the cube, the sentinel, the submit and the de-scatter, the same shape
the convolution entries have. It is a separate entry rather than a dispatch from the
RK3588's `rocket_pool_int8()`. The RK3588 entry truncates the average toward zero and this
one rounds half to even, and two roundings are two functions. It takes an input zero
point, which the RK3588 entry does not, because the average path pads with it.

Whether the average is exactly rounded is a function of the window size. The function
`rocket_pool_int8_rk3576_exact()` answers that for a descriptor without running it. The
reciprocals are truncated, `0x10000/kw` and `0x10000/kh`, so the computed quotient is low
by `|sum| * (1/n - rw*rh)`. The closest a quotient of an integer by `n` comes to a half is
`1/(2n)`. The error is far under that boundary at k2/k3/k5, and is not guaranteed to be
under it at larger windows.

A global 7×7 average over 1024 channels, a MobileNet's pooling layer, measures exact
anyway. It also matches TFLite's `AveragePool` exactly there. An odd window has no tie, so
half-to-even and TFLite's half-up cannot differ. The int8 rebase is exact because `128*49`
is an integer multiple of the divisor. [HW sweep, H96 MAX M9]

PPU_RDMA honors its source surface stride verbatim, including a value that is not a
multiple of four. Every vendor pooling program stores `round4(iw*ih) * 16` in
`R76_PPUR_SRC_SURF`. A direct convolution's output surface stride is `ow*oh` exactly, 49
against 52 at a 7×7 plane. Whether the PPU takes the unpadded stride therefore decides
whether a pool can read a convolution's surface as its feature cube without a copy. The
PPU does take it, bit-exactly at 49, 25 and 9 elements per channel group
(`rk3576_pool_probe lib`, both strides, `ROCKET_RK3576_POOL_PACK_SRC=1` forcing the packed
one). The result holds over max and average, at zero points either side of zero
[HW sweep, H96 MAX M9, 2026-07-31].

A wrong stride there would compute a full and plausible surface rather than faulting.
That is why the stride is a gate rather than an assumption. The field
`pool_params_rk3576_t.src_surf_elems` carries it, and 0 keeps the vendor's `round4`.

**A pooling job raises no DPU completion, and the driver has to be told.** A pool enables
PPU and PPU_RDMA and no DPU stage at all, the same per-block bitmap that makes
`PC_OPERATION_ENABLE` a trap. The `DPU_0`/`DPU_1` bits that `patches/rk3576/npu/0012`
retires a job on can never set for it. The job falls through to the `dpu_grace_us`
deadline, paid in full on every submit. Measured through the library entry, the wait
tracks the parameter count for count, 641 us at 500, 389 at 250 and 213 at 100. That
tracking is what says the job is not retiring on a completion at all
[HW sweep, H96 MAX M9, 2026-07-31].

The PPU's own bits are two positions over in the same register: `PPU_0` at `0x400` and
`PPU_1` at `0x800`, against the DPU's `0x100`/`0x200`. The patch `patches/rk3576/npu/0021`
adds `DRM_ROCKET_JOB_PPU_DONE` to select them per job, and the wait becomes 71 us. The
selection is per job rather than per driver. A job's class is invisible at `PC_DONE` time,
and obvious to the userspace that wrote `PC_OPERATION_ENABLE`.
[HW sweep, H96 MAX M9, 2026-07-31]

**Gate the flag on the interface version (>= 1.3).** The submit ioctl rejects a flag word
it does not recognize, so an older kernel fails the submit rather than ignoring the bit.
Do not set the flag on a stream that mixes DPU and PPU programs. There, an interior
program's PPU bit would retire the job while a later DPU write was still draining.

### The unprivileged double free, and the crashes it was mistaken for

**`rocket_copy_tasks()` frees the submit job's task array twice.** It allocates
`rjob->tasks`, and on any failure inside its copy loop takes a `fail:` path that
`kvfree()`s it and returns, without clearing the pointer. Its only caller,
`rocket_ioctl_submit_job()`, then unwinds through `rocket_job_put()` ->
`rocket_job_cleanup()`, which frees `job->tasks` unconditionally. The same allocation
goes back to the allocator twice. The defect is in mainline v7.1, unchanged by this
series, and the RK3588 reaches the same path. [source-confirmed:
`drivers/accel/rocket/rocket_job.c` at v7.1]

Two failures reach it, and both are ordinary userspace input:

| failure | how a client gets there |
|---|---|
| `copy_struct_from_user()` fails | a bad task pointer, or a trailing field this kernel does not know (`-E2BIG`) |
| `task.regcmd_count == 0` | a task that asks for no registers |

The second is the one that matters in practice. **A generator that refuses leaves its
output parameters untouched.** A caller that submits anyway hands the kernel a task with
`regcmd_count` still at its zeroed value. That is the exact shape `librocketnpu`'s own
deadline canary once submitted.

`/dev/accel/accel0` is group `render`, so this is an unprivileged double free of a
kmalloc-16 object (a `struct rocket_task` is 16 bytes), one of the hottest caches in the
kernel. The freelist then hands one object to two owners, and **the damage surfaces far
from the double free**.

**Two faults look like defects in their own subsystems, and neither is.** They were chased
as subsystem defects for a session and a half. Both are this corruption landing
downstream:

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
the map side are the same event seen twice. The free-side fault reads as "an sgt torn
down while still published", and it is not one. Instrumenting `rocket_gem_bo_free()`
caught two live BOs naming one `sg_table`: two owners of one slab object, not a lifetime
race.

`slub_debug=FZPU slab_nomerge` on the kernel command line settles it in one run, and the
result is deterministic. It gives six "BUG kmalloc-16: Object already free" reports over
three runs of `uapi_submit_errpath_rocket`, and zero on the fixed module. The reports name
`rocket_ioctl_submit()` as the allocation site and `rocket_job_cleanup()` as the second
free. The fix is `patches/rk3576/npu/0020`, which gives the array a single owner and
deletes the `fail:` free. That free can go because `rocket_job_cleanup()` already runs on
every path out of the ioctl.
[HW sweep, H96 MAX M9, 2026-07-29]

The defect is in mainline, and the RK3588 reaches it identically. The same instrument
reproduces it on a Turing RK1 at 7.1.1. A module carrying `patches/rocket/081`-`086` gives
six "Object already free" reports over three runs of `uapi_submit_errpath_rocket`. The
same module plus `087` gives zero over three runs. That is two reports per run on both
parts, at the same allocation and second-free sites. [HW sweep, Turing RK1, 2026-07-29]

On the fixed module, 83 of 83 `ctest` entries pass under `slub_debug` with the taint word
unmoved. Patch `087` is `0020`'s one-line deletion against the RK3588 series.
[HW sweep, Turing RK1, 2026-07-29]

A board whose image ships no kernel headers can still take an A/B comparison. The RK1's
`linux-image` deb carries no build tree, so `make -C /lib/modules/$(uname -r)/build` has
nothing to use. The only thing a `modules_prepare` tree lacks is `CONFIG_MODVERSIONS=y`,
and the CRCs it needs are already on the board. A module built by that kernel carries them
in its own `__versions` section, 64-byte entries of an 8-byte CRC and a 56-byte name.

Read the CRCs back off the shipped `rocket.ko` and write them out as a `Module.symvers`.
The owning module per symbol comes from `/proc/kallsyms`, which an unprivileged reader gets
with the addresses zeroed and the `[module]` tags intact. An out-of-tree build against a
stock kernel.org tarball at the same version then loads with a matching vermagic. The
script is `tools/mksymvers.py`. The tree needs the running `/boot/config-*` and a
`localversion` file carrying the Debian suffix, so that `UTS_RELEASE` comes out identical.

**The gate that triggers it exits 0.** The test `uapi_submit_errpath_rocket` asserts that
five rejection sites return an errno, and they do. An exit status cannot see the slab
corruption on the way. Every gate ran green over it for two sessions. **A test that
drives rejection paths can be corrupting the kernel while it passes.** Read the kernel
log, and run the rejection tests under `slub_debug` at least once.

**A reproducer that never reproduces is evidence about the reproducer.** Neither 300
iterations of `uapi_bo_lifetime_rocket` nor eighteen rounds of the convolution, matmul and
pooling gates fired, with core 1 bound and unbound. They stay silent because not one of
them submits a job the kernel rejects. The crash tracked the presence of
`uapi_submit_errpath_rocket` in the sequence, not the volume of BO churn. It fired in
round 1 on both runs that included it, and in no round of the two that did not.

The crash does surface roughly once per few minutes of heavy BO churn, whichever gate is
running. That timing is right, and it does not make the churn the cause. The cause runs the
other way: one gate plants the corruption, and whichever gate runs next collects it.

A second, smaller mainline defect in the same source is a leak rather than a crash. The
function `rocket_job_open()` allocates the `drm_gpu_scheduler` array, and
`rocket_job_close()` frees `entity->sched_list`. But `drm_sched_entity_init()` stores that
pointer only for `num_sched_list > 1`, and `drm_sched_entity_select_rq()` clears it again
once the entity has settled on a runqueue. On a single-core device, which is the supported
RK3576 configuration, the close is `kfree(NULL)`. The array then leaks on every
open/close, unprivileged and unbounded. [source-confirmed, HW sweep, H96 MAX M9]

The patch `patches/rk3576/npu/0019` keeps the driver's own pointer and frees it after
`drm_sched_entity_destroy()` rather than before. It also checks the allocation, which
mainline leaves unchecked, and frees it on the init-failure path too.
[source-confirmed, HW sweep, H96 MAX M9]

## The DPU's elementwise stage

The DPU's elementwise stage computes on one operand and requantizes the result. An
elementwise op on this part is a program of its own: DPU + DPU_RDMA only, no CNA and no
CORE, 89 writes, `PC_OPERATION_ENABLE` `0x18`. It has the same shape as the LUT table
load, but this one computes. It is not fused into the producing convolution's epilogue, so
in the vendor's hands a residual block pays a program for its add. The lowering this
driver ships does not pay it, as the residual-add section below describes.

What it computes, measured on silicon:

```
out = sat8( ( ((ew + EW_CVT_OFFSET) * EW_CVT_SCALE >> EW_CVT_SHIFT)
                                    * OUT_CVT_SCALE >> OUT_CVT_SHIFT ) + OUT_CVT_OFFSET )
```

The hardware adds `OUT_CVT_OFFSET` after the final shift, in output counts. The vendor's
own Add program carries an offset of -1. Replayed verbatim, it is exact on 8192 of 8192
elements under that placement. With the offset added before the scale, every element is
off by one [HW sweep, H96 MAX M9, 2026-09-26, `tests/rk3576_add_probe.c ga54b`].

The one-operand model is bit-exact over channel counts 16-320 and planes 1×1 to 28×28
(`tests/rk3576_add_probe.c gate`, 10 shapes). The operand is the NC1HWC2 cube the
convolution path already packs. The final shift rounds half to even, the same rule the
direct path's OUT_CVT uses, established here independently. Against a round-half-up
reference, every disagreement was an exact tie: 25% of elements at a gain of one half,
and none at unity. [HW sweep, H96 MAX M9, 2026-07-31]

The registers that differ from a convolution's DPU half:

| register | add | conv | what it is |
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

`MUL` is the same program, differing in four registers: `0x407C` `0x810F4094`, `0x4044`
`0x2`, `0x4050` `0x00020000` and `0x501C` `0x2`. No gate covers its arithmetic.

### The search for a second operand

The elementwise stage takes exactly one operand through this register set, and the
negative is closed. Six sweeps cover the whole interface:

- **Every register the program leaves at zero, tried as a second base at every output
  gain.** Of seventeen candidates crossed with the full 32-rung OUT_CVT ladder, nothing
  reaches the output except `0x5024`, which injects a constant. That register is the DPU
  shift word, a live per-task operand and not a spare. Crossing the two axes is what makes
  this conclusive. A placement sweep at one gain is blind to an operand that sits
  somewhere unexpected and enters the accumulator unscaled. So is a gain sweep at one
  placement.
- **Every register the program never writes at all, appended to the program one at a
  time.** This is the complement of every other sweep here, and it is not an empty set.
  The register file is not cleared between jobs on this part, so a base the vendor
  programs in an earlier task would still be standing. Twenty-nine candidates over the
  DPU and DPU_RDMA blocks, at two gains, show that nothing carries an operand.

  One register stops the write: `0x5068`, the RK3588's `RDMA_WEIGHT`, which holds the
  four read DMAs' arbiter weights. At any value that is not a weight, the job writes
  nothing and retires at the driver's 125 ms backstop. **Its value then stands**: the
  shipped program stalled on 8 of 8 jobs after it, until the NPU's power domains had been
  off for minutes. Written alone behind a power cycle, `0x5070` and `0x5074` write
  exactly, so their stalls were `0x5068`'s standing value [inferred].
- **The main DMA feed at every gain from 2^14 down to 2^-31.** The program is configured
  to read one. DPU `0x400C` bit 0 is the NVDLA feature-mode flying bit, and the add sets
  it. DPU_RDMA `0x5044` bit 4 is `MRDMA_DISABLE` in the same lineage's map, and the add
  clears it. The feed contributes exactly zero at all 64 rungs, with the elementwise
  operand live as a control at every one.
- **A joint grid over `EW_CFG` and `BRDMA_CFG`, and a second over `FEATURE_MODE` and
  `ERDMA_CFG`.** The second drives the `COMB_USE` field that the RK3588's own
  K-accumulation uses to combine two feeds. In both grids, only the captured word writes
  at all, and every other combination leaves the surface untouched.
- **Destination accumulation, with DPU `0x40C0` swept over the captured word and 28
  single-bit variants of it.** Each variant ran twice over differently pre-filled
  destinations, which classifies it. It always overwrites. Two bits (`0x00010000`,
  `0x00020000`) saturate the output, and none accumulates. The RK3588's map puts
  `SURFACE_ADD` at this offset, and on this part it is not an accumulate mode.
- **What the program reads, rather than what it was handed.** The probe walks a single
  non-zero 16-byte atom over an operand buffer eight cubes long, with the base pointed at
  the middle. The addressed cube's 32 atoms map one to one onto the output, and nothing
  outside it moves anything. So the two operands are not one allocation at a fixed offset
  either, which is how the RK3588's K-accumulation feeds its pair.
- **The candidates written together, with the main feed at a second operand.** A condition
  of two has no sufficient singleton, so all three went in at once, with a second operand,
  B, at DPU_RDMA `0x5018`. Register `0x5068` took the RK3588's weight word `0x01010101`.
  The other two took B's address, 0, or the plane's area and width. One arm adds the
  RV1106 vendor Add's `0x5044` word, and each arm ran over one A and two non-periodic B
  fills at two gains. Every arm that writes moves 0 elements between the B fills and
  matches the one-operand model [HW sweep, H96 MAX M9, 2026-09-26, `ga54`].
- **The vendor's own program, fed the way its container says.** Each `.rknn` carries a
  relocation table naming the program words the runtime patches with each tensor's
  address. In `bare_add_c32_16` the first operand goes to DPU_RDMA `0x5018`, the second to
  `0x5038` and the output to DPU `0x4018`, and `subrev` swaps the two operands. The probe
  replayed it verbatim with both bases patched, as one task and as its two slots. It wrote
  the one-operand surface on 8192 of 8192 elements and ignored B [HW sweep, same]. This
  replay cannot see anything else the vendor runtime patches or submits, and the vendor's
  Add has not been run on this silicon.

The manufactured captures say what the vendor's program does have, which is what makes
the negative worth stating precisely. A `Sub` compiles to this same program with the
operand converter's scale negated, and a `Sub` with its operands swapped negates it too.
The subtrahend is therefore always the elementwise cube. The other operand rides a feed
whose weight is fixed at +1, since no register carries a second scale.

Both operands also share one quantization scale. Two graph inputs calibrated 64x apart
still compile to one converter. Its OUT_CVT gain comes out as the ratio of a single shared
input scale to the output's.

### The residual add on the convolution datapath

This driver lowers the residual add onto the convolution datapath instead. The lowering
concatenates the two operands along the channel axis. It convolves the result with a 1×1
kernel of two diagonal blocks, `W[o][o] = w1` and `W[o][C+o] = w2`, zero elsewhere:

```
out[o] = requant( w1 * (a[o] - a_zp) + w2 * (b[o] - b_zp) )
```

The lowering is bit-exact against a CPU model of the part's own arithmetic over nine
MobileNetV2 and ResNet-18 residual shapes (channels 24-512, planes 7×7 to 56×56). At every
one it is within one count of an exact float residual add (`tests/rk3576_residual_add.c`).
Two of its properties are better than the vendor's elementwise program, not merely equal
to it:

- **The two operands can carry different scales.** The ratio rides in the weights as
  `w2/w1`, so the pair only has to be representable as two int8s, about one part in 127.
  The vendor's one operand converter forces its compiler to quantize both operands to a
  common scale.
- **The two zero points ride in the bias, exactly.** The datapath has one input zero
  point, but `w2 * (a_zp - b_zp)` is a per-output-channel constant, which is what the
  bias is.

The lowering also fuses, measured rather than argued. A block's last convolution takes `C`
more input channels and an identity block at the center tap of its kernel. The skip is
then part of a convolution the network was already paying for. The add therefore costs no
program at all, against the one program the vendor pays.

This fusion is the hardware's own idiom: the vendor compiler folds `Add(Conv(x), x)` into
exactly this shape. That folding is why the graph is useless as a capture of an add. The
fused form is bit-exact at MobileNetV2's project convolution for every residual width
(1×1, `ic` 168 to 1120) and at ResNet-18's second convolution (3×3, `ic` 128 to 512).

**The fusion stops at the weight-slice rule**, `ic*kh*kw <= 4608`. A 1×1 project
convolution is nowhere near it, and a 3×3 reaches it exactly at `C = 256`
(`512*9 = 4608`) and is refused at `C = 512`. ResNet-18's widest stage therefore keeps
its add as a standalone program, which the same rule caps at `C = 2304`. The standalone form's weight cube is
`2C*C` bytes, 512 KiB at that stage, paid once as resident weights rather than per
inference. Transiently, a standalone add measures 0.6-3.3 ms across the shape table.
[HW sweep, H96 MAX M9, 2026-07-31]

### Traps in the elementwise program

**Every base left at the capture's stored zero reads IOVA 0, which is a real buffer.**
Per-fd IOVA starts at zero on this stack, so the first BO a probe allocates is what those
bases read. With the operands allocated first, the surface comes back a constant and the
arithmetic looks broken. Allocate a guard BO first, so that no operand or output buffer
sits at IOVA 0, and the operand passes through exactly. A zero base is not a disabled
one.

**DPU `0x40D0` must be the captured `0x0040FFFF` verbatim.** Read as a clamp pair and
written as `0x00407F80`, it leaves exactly half of every 16-channel group unwritten,
silently. That is a write-coverage failure, not a wrong value, and it reads like a
channel-budget property of the writer. The clamps are `OUT_CLAMP_MIN`/`MAX`
(`0x40A4`/`0x40A8`), and the no-clamp pair is `INT32_MIN`/`INT32_MAX`.

`OUT_CVT_SCALE` is a **signed** 16-bit field on both parts: `32768` reads back as `-32768`
and flips the output's sign. The usable maximum is 32767
([../encodings/out-cvt-converter.md](../encodings/out-cvt-converter.md)).

### Manufacturing the capture

The script `tests/data/rk3576-vendor-capture/add/mkadd.py` builds the captures, and
`decode_add.py` reads them. **`Add(Conv(x), x)` is not a capture of an add.** The vendor
compiler folds an identity skip into the convolution's own kernel, at the center tap of the
diagonal. Six such ONNX graphs, over four channel counts and three planes, compiled to a
register program indistinguishable from a plain convolution.

The graph `Add(Conv_a(x), Conv_b(x))` folds too, and so does a per-channel broadcast
operand. A graph `Add(Conv(x), k)` with `k` of shape `[1, C, 1, 1]` compiles to a plain
convolution with `k` in its bias. A capture of the op needs an operand the compiler cannot
reach. Each of these three emits it, identically:

- A second graph input.
- A constant tensor of the same shape.
- A real MobileNetV2 bottleneck whose skip crosses three convolutions.

**Two operands at the same scale are unattributable.** When both operands calibrate to the
same range, every register that scales an operand looks the same for both. The two
converters then cannot be assigned. Give the inputs deliberately asymmetric calibration
amplitudes, and the assignment falls out. That method, with `Sub` compiled in both operand
orders, is what names the elementwise cube as the *second* operand. It also pins the
first's weight at unity.

A capture's container says how many tasks an op costs. Each `.rknn` carries a table of
40-byte task records,
`[index][PC_OPERATION_ENABLE][slot size][int mask][int clear] [write count][…][program offset]`.
Matching `write count` against the program stream is what identifies them. Read the table
before concluding anything about an op's structure from the program list alone. A single
`Conv` emits four task records over four distinct program slots at two different row
geometries, so program count is not op count.
