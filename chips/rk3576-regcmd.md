# RK3576 int8 conv register encoding, as an emitter

The RK3576 NPU runs the same IP family as the RK3588 through the same mainline
`rocket` driver, but its CNA/CORE/DPU blocks use a different geometry-register
encoding at the same block bases. This sheet is the encoding written out as
something you can emit, plus what running it on real silicon settles and what it
does not. The SoC-level parameter sheet (identity, integration, clocks, power) is
[rk3576.md](rk3576.md); this is the register layer.

Provenance: RKNN-Toolkit2 register programs for known conv geometries, from two
capture sets under
[../../rocket-userspace/tests/data/rk3576-vendor-capture/](../../rocket-userspace/tests/data/rk3576-vendor-capture/).

Found captures, from Ga Hing Woo's bring-up repo, real convolutions, including a
MobileNet-shaped stem:

| Capture | Geometry | What only it can show |
|---|---|---|
| `conv2d_rk3576.rknn` | ic 16, oc 128, k5 s2, 80x80 -> 40x40 | the normal `in_ch>4` path at k != 3; a single task over the whole plane |
| `dw_rk3576.rknn` | depthwise c32, k3 s1, 112x112 | the depthwise words, and a task windowed to 91 input rows |
| `conv64_rk3576.rknn` / `conv0_rk3576.rknn` | ic 3, oc 32, k3 s2 | the first-conv ARGB sub-encoding, at two image sizes |
| `iso_bias` / `iso_scale` / `iso_sum` | the `conv2d` geometry plus one appended op | what a DPU epilogue stage moves |

Manufactured captures, in `dw/` and `dw/named/`. **A vendor capture can be built to
order**, and that is the single most useful thing to know about this part: an ONNX
compiled for `rk3576` emits the register program for whatever geometry is asked for,
so a question that a bit sweep answers slowly and ambiguously can be answered by a
diff. `group=C` gives the depthwise path at any channel count, kernel, stride and
plane; `do_quantization=False` on a float ONNX gives the float datapath. `dw/mkdw.py`
and `dw/named/mknamed.py` rebuild both sets and carry the dependency pins.

The found depthwise captures confound almost everything: every C=32 program in them
is stride 1 and every C=64 one is stride 2, every one is k=3, every one is a multiple
of 32 channels, and every one is a square plane. Manufactured ones at 22 channel
counts from 8 to 256, both strides, kernels 1/3/5/7 and rectangular planes separate
all of it; see "Depthwise" below, where four register formulas that had been fitted
or guessed are now transcribed.

The emitter built from them is
[npu_regcmd_rk3576.c](../../rocket-userspace/src/npu_regcmd_rk3576.c); the gate
that diffs it against the captures register-for-register is
[regcmd_rk3576_gate.c](../../rocket-userspace/tests/regcmd_rk3576_gate.c). It
reproduces every checked register of every captured program, 110 conv programs,
88 of them depthwise, with no open field left.

Register fidelity is not correctness, and depthwise is where the two part company.
An emitter matching the vendor register for register across 88 depthwise programs
still computed nothing, because what it was wrong about were the buffers the
registers point at, a capture carries a register program and says nothing about the
memory it addresses. Both of that path's buffer layouts had to be read off the part
(see "Depthwise" below), and that is the general lesson rather than a depthwise one.
The correctness gate is
[rk3576_conv_gate.c](../../rocket-userspace/tests/rk3576_conv_gate.c), a shape table
swept in one process, each entry compared against a CPU model bit-exactly over the
whole surface, covering the envelope, the row window, the channel-group jump and the
weight-slice boundary, and the depthwise envelope beside them. It runs gap-free on a
`rocket` carrying the per-SoC `PC_TASK_CON` width, and reports every cold-start-wall
retry it takes, so a run that prints none is a run the wall did not touch. 102 shapes
pass in 4.6 s.

## Four corrections to the published map

Each of these is load-bearing: an emitter that takes the published reading
silently mis-programs the part.

**The output-height field is not halved.** It is the number of output rows *this
task* writes, minus one. The halving comes from reading a capture whose task
covered half its plane: the same conv captured as a single task writes the full
`oh-1` (`conv2d`: CORE `0x301c` hi = `0x27` = 39 for a 40-row output; the ARGB
`conv0` full-plane task writes `0x6f` = 111 for 112 rows, and only its
half-height slices write 55). Emit `(oh_task-1)<<16 | (ow-1)` and the same
`oh_task-1` at DPU `0x4024` / `0x4034` and RDMA `0x5010`.

**The feature-data address is `0x1088` on both datapaths**, not `0x1070` on the
ARGB path. `0x1070` reads zero in every capture, including the one program that
provably needs a non-zero feature address. The `conv0` capture's fourth task
reads input rows 111.. of a 224-wide 3-channel image, and `0x1088` carries
exactly `111 * 224 * 3 = 0x12360`. `0x1070` is zero in that same program. The
address registers that do coincide with the RK3588 are the weight base
(`0x1110`) and nothing else.

**The pad word `0x1080` is closed form**, not open. It carries four byte fields:

```
0x1080 = pad_right<<24 | pad_bottom<<16 | pad_left<<8 | pad_top
```

**The CBUF entries word is `ceil(iw*ic/64)`, not `max(iw/4, ic/2)`.** The two
agree on every capture the published map was fitted to, which is why it carries
the second form, but they part company as soon as `ic/2` exceeds `iw/4`, and the
depthwise capture is exactly that case: `iw`=112, `ic`=32 gives 56 for the
granule count and 28 for the max form, and the capture holds **56** at both
`0x103c` hi and `0x1044` lo. Read it as what it is, the count of 64-byte CBUF
granules in one feature row. The distinction is invisible at the shapes the map
was derived from and load-bearing for any narrow-and-deep tile (a 1x1 pointwise
over a small plane sits squarely in the diverging region).

where the leading pads are the configured padding and the trailing two are the
pad *actually consumed* by this task's last window,
`max(0, (out-1)*stride + k - (pad_lead + in_extent))`. It reproduces all four
captures exactly, including the two that make it unambiguous: `conv2d`'s
asymmetric SAME padding (l1 r2 t1 b2 -> `0x02020101`) and the depthwise task
whose row window stops short of the image bottom, so its bottom pad is 0 while
its right pad is 1 (`0x01000101`). The published rule, a lookup
keyed on stride and depthwise, predicts `0x00000101` for `conv2d` and is wrong.

**So there is no configured trailing pad, and that is how an asymmetric pad is
expressed**: the trailing fields are a function of the output extent and the
leading pad, so a caller who wants TFLite's SAME at an even plane and stride two,
`pad_before = 0`, `pad_after = 1`, asks for a leading pad of zero and an
output extent one larger than the symmetric formula gives, and nothing has to be
materialised in the feature buffer. `rocket_conv2d_desc.oh/.ow` name that
extent, zero meaning the symmetric derivation. The bound is the kernel: a
trailing pad of `k` or more is an output row whose whole window is pad. Six
shapes of the int8 correctness envelope carry a zero leading pad against a
consumed trailing one, direct and depthwise, and all are bit-exact, the float
path has never been run through that geometry and refuses it.
[HW sweep, H96 MAX M9]

## Registers the map left unnamed, and their RK3588 equivalents

Four of the offsets recorded as "RK3576-only, no RK3588 counterpart" are the same
registers the RK3588 has, moved:

| RK3576 | Function | RK3588 |
|---|---|---|
| `0x1048` | CVT control word, `data_sign<<3 \| cvt_type<<1 \| cvt_bypass`, identical packing | `CNA_CVT_CON0` `0x104C` |
| `0x104C` / `0x1050` | the four CVT scales, **two per register** (`scale1<<16 \| scale0`, `scale3<<16 \| scale2`) | `CNA_CVT_CON1..4`, one each |
| `0x1054`-`0x105C` | the CVT offsets (non-zero only on the ARGB path, where they carry the per-channel -128) | folded into the same CON1..4 |
| `0x108C` | burst lengths, `weight_burst<<16 \| data_burst`, `0x000F000F` | `CNA_DMA_CON0` `0x1078` |
| `0x103C` hi, `0x1044` lo | CBUF data entries, `ceil(iw*ic/64)`, written twice | `CNA_CBUF_CON1` `0x1044` |
| `0x1084` | the CNA border pad constant | `CNA_PAD_CON1` `0x1184` |
| `0x5024` | base of the DPU shift word, a per-task operand, not a spare (see below) | the register fields `BS_MUL_CFG.BS_MUL_SHIFT_VALUE` `0x4048` and `DATA_FORMAT.BS_MUL_SHIFT_VALUE_NEG` `0x4010` |

The identification is by value, not by analogy: the int8 conv's CVT word is
`0x0b` in both encodings, and the burst word is `0x000F000F` in both, for the
same reason.

**The kernel word has a closed form.** `0x1024` hi is `((kh-1)<<8) | (kw-1)`,
`0x0202` at k=3, `0x0404` at k=5 (the `conv2d` capture is the only one that shows
a second k), `0x0000` at k=1. Published as a two-entry lookup; it is one
expression.

**The OUT_CVT triple is `0x40AC` / `0x40B0` / `0x40B4`**: offset, scale, shift,
three consecutive registers, the same requant the RK3588 puts at
`0x4080`-`0x4088`. This was inferred from position (a small signed value ahead of
a scale and a shift) and then **confirmed on hardware by register sweep**: forcing
the offset to 16 adds exactly 16 to every output channel, forcing the scale to 0
zeroes the surface, and raising the shift attenuates a saturated result into
range. [HW sweep, H96 MAX M9]

## Still open

- **On-chip accumulation across the fp16 `ic` split.** The DPU eltwise stage does this
  job on the RK3588 (`ROCKET_KACC`) and would remove `ic/16` readbacks here. It used to
  be blocked behind the output writer; at the 16-channel contraction the partial it
  would read back is a plain, dense, complete fp16 cube, so nothing blocks it now.
- **The ARGB first conv's INT8 weight cube.** The float cube is decoded and the fp16
  first conv computes on it; the int8 one is not, and it is not the same object, the
  depthwise path already showed this part's int8 and float cubes differ. A quantized
  ARGB capture does take the path, but its weights do not survive quantization as a
  findable value set, so this one likely has to be read off the part the way the
  depthwise int8 cube was. `rocket_conv2d_int8_rk3576()` owns `ic <= 4` through its own
  packed-image path.
- **The vendor's float BS arrangement.** `0x501C` and `0x4044` together select a BS
  operand arrangement this library's coefficient group is not packed for; ours is
  bit-exact and theirs is not, so ours is what is emitted. Decoding the vendor's would
  say what its epilogue buys; see the section above for the leave-one-out result.
- **What the vendor's other two BS arrangements buy.** `0x501C` is decoded as the BS
  operand-source register (see the epilogue section) and this library packs for the
  arrangement that is bit-exact; what the epilogue and float arrangements read instead
  is not. The one-in-four survival rate under the epilogue one is the lead.
- **`0x1060` is probably `CVT_OFFSET3`.** `0x1054`-`0x105C` are the first three CVT
  offsets and there are four CVT scales, so the fourth offset almost certainly sits
  here. Every capture leaves it zero, including the ARGB ones, which use only three
  channels, so nothing pins it, and it stays in the address-placement shotgun list
  rather than being claimed.
- **`0x1064`.** The one genuinely live vendor dump (an `rknpu` kernel capture
  taken at submit, as opposed to the ten programs decoded out of `.rknn` files)
  carries `0x777` here where every stored program carries 0, so the vendor
  runtime patches this register at load time, the way it patches the addresses.
  It is the offset the RK3588 calls `CNA_FC_CON1`. Forcing it to `0x777` on our
  path changes nothing (nor does `0x1060` or `0x1074`), and the RK3588 runs its
  own matmul with this register at 0, so it is not required for the direct-conv
  datapath. Worth re-testing if a fully-connected mode is ever driven. The wider
  point is the methodological one: a `.rknn`-derived program is the *stored*
  register set, and any value the runtime patches in is invisible in it.

## What running it establishes

Measured on an H96 MAX M9 (RK3576, mainline 7.1.3), driving the emitter above
through `librocketnpu` with an int8 1x1 conv, IC=OC=32, 8x8, symmetric
quantization. [HW, H96]

**The encoder is the difference between no output and an output surface.** With
the RK3588 register program this part writes nothing at all, the output BO comes
back untouched. With the RK3576 program the DPU writes the full, correctly-sized
surface: 2048 bytes for a 32-channel 8x8 int8 output, in the same NC1HWC2 cube
the RK3588 uses (C2=16, surface stride `ow*oh` in 16-byte atoms, which is why
`0x401c` holds `ow*oh` and not `ow*oh*16`).

**The part computes convolutions, and the emitter is what makes it do so.** A
k=3 SAME 32x32 int8 conv, IC=OC=32, driven with a dense random weight set (all
9216 weights non-zero) over a feature tensor spanning the full signed int8 range
reproduces the CPU model **bit-exactly across the whole surface, 32768/32768,
max |diff| 0**, and the same conv unpadded is 28800/28800. That single result
carries most of the encoding: the CNA feature and weight DMA, the CBUF staging,
the CSC, the CMAC, the BS bias add, the OUT_CVT requant, the output geometry and
the writer are all being programmed correctly. It also settles the weight cube
independently, a dense random set of 9216 weights cannot match bit-exactly under
a permuted layout, so the RK3576 takes the **same weight cube as the RK3588**
(`weight_conv_int8`, oc-group 32 / ic-group 32) at IC=OC=32.

**The envelope is every geometry whose channel counts are programmed as multiples
of 32 and whose plane fits the CBUF budget.** Across IC and OC from 8 to 128,
planes from 8x8 to 128x64, k1/k3/k5, stride 1 and 2, VALID and SAME, against
feature tensors that vary on all three axes, the emitter reproduces the CPU model
bit-exactly, the whole surface, every time, repeatable across separate power
sessions. Both boundaries below are properties of how the operands are described,
not of the datapath, and both are now closed by construction: the channel counts
come from `rocket_rk3576_pad_ic()` / `_pad_oc()`, and the CBUF budget is the
`rocket_hw_rk3576` machine-parameter profile.

## Both channel counts must be programmed as multiples of 32

The register `ic` and `oc` must each be a multiple of 32, the group
`weight_conv_int8` pads its two channel axes to, with the feature cube, the output
BO and the coefficient buffer sized to the padded counts. Convolutions whose
channel counts are not multiples of 32 compute wrong otherwise, at every geometry.
No capture shows this, because every captured geometry already satisfies it.
[HW sweep, H96 MAX M9]

On the input axis, ic of 8, 16, 17 and 48 are wrong at every kernel and plane size
while 32, 64 and 96 are exact. ic=17 fails despite spanning two whole C2=16
surfaces, so the unit is the 32-channel MAC group and not the surface. Padding the
feature cube alone does not help, every read past `ic` already lands in mapped
zeros, so it is the register value that matters, and passing the padded count with
the cube zero-filled to match is bit-exact.

On the output axis a partial group trips **two different mechanisms**, which is why
it presents two ways and why a sweep at one kernel size reads as "oc is
unconstrained":

- At **k=1** the DPU writes only output **row 0** of the trailing group and leaves
  the rest of the surface untouched. oc 8, 16, 40 and 48 fail this way; 24, 32 and
  64 are exact. The rule that fits k=1 alone is surface parity, `ceil(oc/16)` must
  be even, which is why oc=24 passes there.
- At **k>1** the weights come out wrong instead, and oc=24 and oc=56 fail even
  though their surface count is even. `WEIGHT_BYTES` (`0x101C`) is `ic*oc*kh*kw`,
  which describes a cube tighter than the padded one the caller supplies, and in
  the `[kh][kw][oc2=32][ic2=32]` cube that truncation drops whole `(kh,kw)` planes
  rather than trimming `oc` inside each. At k=1 there is only one such plane, so
  the same shortfall lands harmlessly on the unused `oc2` slots, which is exactly
  why k=1 is the tolerant case.

Rounding `oc` up to 32 satisfies both. The trap to recognize is the first
mechanism: a partially-written surface reads as an output-geometry defect, not a
channel one.

The vendor does not pad. Its `conv2d` capture is ic=16 with weight bytes
`ic*oc*kh*kw` = the tight size, so it packs a 16-channel weight group where
`weight_conv_int8` pads to 32, and our emitter matches that program 131/131, so
the difference is in the cube, not the registers. Padding is how the RK3588 cube
expresses the same conv; a partial-group weight packing would be the other way, at
half the weight bytes for such a layer.

## The CBUF budget is programmable, and `0x1040` is what programs it

A conv whose feature plane does not fit computes wrong with the DPU still writing a
full surface, no IOMMU fault and nothing in dmesg. The corruption is graded: one row
past the budget about half the surface is still bit-exact, and it degrades with the
depth of the overflow until almost nothing is (at ic=32, iw=16: 4096 granules is
exact, 4104 gives 132001/262656, 4160 gives 1090/266240). Nothing else on this part
presents that way, which makes it the instrument, grow a known-good conv until it
breaks.

The budget is one scalar in the CNA's granule unit, and it is **set by the register
the published map carries as the constant `0x10000000`**:

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

Widths of 16, 32 and 64 all break at the same granule totals, 4096 at F=0, 6144 at
F=2048, so the limit is the granule count and not rows, width or channels
individually. The data budget does not exceed **6144 granules (384 KiB)**: F=3056
still measures 6144. [HW sweep, H96 MAX M9]

**Only single-bit F values deliver their face value.** Each power of two measures
exactly 4096+F, but a combination delivers less than the sum of its bits, and the two
combinations tested both land on 5152 regardless of which second bit was set. So the
field is not simply an integer the CNA adds, and an emitter that computes an arbitrary
F programs a budget the hardware does not honour, which corrupts silently, since a
plane over its allowance still writes a full surface. Use the measured rungs
(0, 256, 512, 1024, 2048) and round a deficit up to one. The cost is a little weight
headroom the plane did not need. [HW sweep, H96 MAX M9]

**And 256 and 512 deliver conditionally.** Where they do not, each delivers **4096
granules, the F=0 budget**, so a task the planner puts on one of them overruns its
allowance and writes a full, correctly sized surface with a wrong tail: every output row
past `4096 / entries`. 0, 1024 and 2048 are unaffected at every footprint tried on either
path; the vendor's own windowed depthwise capture is a k=3 program at F=1024.

**The failure is a band, not a ceiling, and that is what makes it hard to see.** The
allowance is a ladder and the planner takes the smallest rung that covers the window, so
as the window grows the surface is exact (F=0 still covers it), then wrong across the
band of windows that select an unhonoured rung, then exact again (the next rung up
delivers). Walked one task per height at 160x160 ic = oc = 32 k3 depthwise, 80 granules a
row: exact 45-51, wrong 52-57, exact 58-76. **A probe that bisects the window reports
whichever edge it walks into**, a bisection is justified by "a smaller window is never
worse", which is what a capacity bound means and this is not one. Read the map.

**The condition differs by path, and on the depthwise one no footprint threshold fits.**
Forcing each rung under one fixed window that F=0 does not buy:

| path | kernel | oc | resident footprint | F=256 / F=512 |
|---|---|---|---|---|
| direct | 1x1 | 32 | 1024 B = 16 granules | deliver |
| direct | 1x1 | 64 | 2048 B = 32 granules | fall back |
| depthwise | 1x1 | 32 | 64 B = 1 granule | deliver |
| depthwise | 1x1 | 256 | 512 B = 8 granules | deliver |
| depthwise | 1x1 | 1024 | 2048 B = 32 granules | deliver |
| depthwise | 2x2 | 32 | 256 B = 4 granules | fall back |
| depthwise | 3x3 | 32 | 576 B = 9 granules | fall back |
| depthwise | 5x5 | 32 | 1600 B = 25 granules | fall back |

Depthwise 4 granules is dead where 32 is live, which refutes a threshold in both
directions. What survives the eight cells is the tap count, single-tap delivers at three
channel counts spanning 32x, multi-tap never does, and a 64-channel-group footprint
(`R76_DW_W_GROUP_INT8`) against the direct path's 16-granule threshold fits k1, k3 and k5
and is refuted by the 2x2 cell, which is why that cell is in the probe. So the emitter
declines the low rungs on the depthwise path rather than gating them on a fitted quantity;
it costs nothing, since the fallback rung is strictly larger and always live.
[HW sweep, H96 MAX M9, `tests/rk3576_conv_lib_gate.c rowmap`]

**The direct path's own condition is the resident weight footprint, under 1 KiB**, and
the rest of this section is that measurement.

**The kernel is not the axis, and a square-kernel sweep cannot say so.** The first
characterisation held `ic` at 32 and moved the kernel, at one granule total of 4352,
across five plane widths chosen to hold it constant, k=1 exact everywhere and k=3 and
k=5 wrong everywhere:

| plane | entries/row | rows | k=1 | k=3 | k=5 |
|---|---|---|---|---|---|
| 16 x 544 | 8 | 544 | exact | wrong | n/a |
| 32 x 272 | 16 | 272 | exact | wrong | n/a |
| 64 x 136 | 32 | 136 | exact | wrong | wrong |
| 128 x 68 | 64 | 68 | exact | wrong | n/a |
| 272 x 32 | 136 | 32 | exact | wrong | n/a |

That reads as "the rung needs `kh == 1`" and it is what shipped. **Crossing the axes says
otherwise**: at the same 4352-granule total and the same k1x1, the same (row size, row
count) is bit-exact at `ic` 32 and wrong at `ic` 64 and 128, 68 rows of 64 granules,
wrong from output row 64 in both, with the F=0 control at 64 rows exact and the same
plane forced under the boundary exact.

| 4352 granules, k1x1 | entries/row | rows | result |
|---|---|---|---|
| iw 16, ic 32 | 8 | 544 | exact |
| iw 32, ic 32 | 16 | 272 | exact |
| iw 64, ic 32 | 32 | 136 | exact |
| iw 128, ic 32 | 64 | 68 | exact |
| iw 64, **ic 64** | 64 | 68 | **wrong from row 64** |
| iw 32, **ic 128** | 64 | 68 | **wrong from row 64** |

So `ic` at a fixed kernel and the kernel at a fixed `ic` move the same quantity,
`32*ic*kh*kw`, the resident weight slice, which is what `r76_weight_slice_cap()` is
already stated over. Live at 16 granules (`ic` 32, k=1); dead at 32 (`ic` 64), 64
(`ic` 128), 144 (`ic` 32, k=3) and 400 (`ic` 32, k=5).

**What the rung delivers backs out to the row** from the surviving prefix: the last
correct output row is in every case the one fed by input row `4096 / entries`, which is
what makes 4096 a measurement rather than a reading of a graded corruption.

**The quantity is one output-channel group's slice, not the whole resident cube.** Every
cell that had ever reached a rung carried `oc` 32, one group, where the two are the same
number, so the emitter charges the cube, the smaller envelope. Holding the slice at the
measured-live 16 granules and raising the group count separates them: at `oc` 64 and 96,
cubes of 32 and 48 granules, **F=256 and F=512 both still deliver**. So the shipped rule is
conservative rather than wrong, and it stays that way because relaxing it buys nothing, the
rung it declines to use is replaced by a strictly larger one that also delivers, at the same
CBUF and with no extra submit. [HW sweep, H96 MAX M9, `rk3576_conv_lib_gate rowmap`]

**The threshold between 16 and 32 granules is still bracketed, not measured**, and the
harness cannot narrow it: `ic` is padded to a multiple of 32 on the direct path, so
`32*ic*kh*kw` moves in 1024-byte steps at every shape that can be built with a square kernel
and a whole `ic`. A non-square kernel is what would land between them.

Everything past the threshold rounds up to 1024. The cost is the next rung, which is the same
CBUF and no extra submits, or, when the weight path leaves no room for 1024, a shorter row
window, one more task and not a wrong answer.

Nothing in the correctness envelope reached this before a whole network did: every
direct shape in the table sits at F=0, and so did every depthwise one until a MobileNet
asked for a 3x3 over a 112-row plane. And nothing reaches the ic axis today either, a
direct rung is programmed only where the plane is 4097-4608 granules and the weights are
under 1 KiB, and the matmul's own row planner is past that at every K it runs. **The
packed-image path keeps the direct rule and has never been driven at a rung**: the widest
stem in the corpus, Inception V3's 299x299, stages 5681 granules and lands on F=2048, and a
224x224 one sits at F=0, so that axis is unverified rather than verified. [HW sweep, H96
MAX M9, `tests/rk3576_conv_sym.c rung` and `rk3576_conv_lib_gate rowmap`, with
`tests/rk3576_conv_lib_gate.c` groups `rung256` and `dwbig` for the original kernel reading]

The two values the captures carry are therefore two points on that scale, not a
constant and a variant: `0x10000000` (F=0) buys 4096 granules and `0x14000000`
(F=1024) buys 5120. That is what the vendor's windowed depthwise program needs,
112 wide at ic=32 is 56 granules per row, and its 91-row window is 5096 granules,
which overflows 4096 and fits 5120 with 24 to spare.



**The low bits are the RK3588 field layout, and the vendor leaves them zero.** Bits
0-13, `DATA_BANK[0:3]`, `WEIGHT_BANK[4:7]`, `FC_DATA_BANK[8:10]`, `DATA_REUSE[12]`,
`WEIGHT_REUSE[13]` per Mesa's `registers.xml`, are live: setting any single one of
them on top of a working program corrupts the conv. So the bank *fields* did not
move in the re-pack; what moved is that this part also takes a granule allowance in
bits[16:27], which is reserved on the RK3588. Bit 28 is required (a `0x1040` of zero
corrupts), bit 29 corrupts, and bits 14, 15, 30 and 31 are don't-care. [HW sweep]

**The data and weight sides share one pool, and F trades between them.** The weight
path stages per output-channel group, so its resident slice is `32 * ic * kh * kw`
bytes rather than the whole cube, which is why a 441 KiB weight cube computes fine.
Sizing that slice against F shows the trade directly: a 150 KiB slice is bit-exact at
F=0 and breaks at F=2048, with the ceiling falling as the data side grows.

| F | data allowance | weight slice ceiling | sum |
|---|---|---|---|
| 0 | 4096 gr = 256 KiB | 175 KiB exact, 200 KiB breaks | ~445 KiB |
| 1024 | 5120 gr = 320 KiB | 125 KiB exact, 150 KiB breaks | ~455 KiB |
| 2048 | 6144 gr = 384 KiB | below 75 KiB | ~450 KiB |

Each +1024 granules of data costs the weight path about the 64 KiB the data side
gained, and all three pairs sum to **roughly 448 KiB**: 14 banks of the RK3588's
32 KiB, of which the captures' default program takes 8 for data. The bank *count* is
an inference from that bank size; what is measured is the granule budget, the trade,
and the total. Reading the pool as 14 banks also explains the wedge: F at or past
~3060 leaves the weight path nothing, and the part then writes no output at all at
any plane size, which looks exactly like a wrong geometry encoder. [HW sweep]

Two consequences. `rocket_hw_rk3576`'s 8 x 32 KiB is the **default data allocation**,
not the physical CBUF, raising F buys up to 1.5x the feature capacity, at the cost of
weight-slice headroom that must then be respected. And a register the published map
records as a constant is a tuning knob, so an emitter that copies the constant
inherits the vendor's choice for a full-plane task rather than making its own.

The boundary is sharp enough to plan against, and the pool figure survives a direct
test of it: a slice of exactly 192 KiB (6 banks, what the model leaves beside F=0)
computes bit-exactly at ic=1536 k=2, and 196 KiB at ic=1568 breaks. [HW sweep]

**The emitter plans F per task**, in `rocket_rk3576_cbuf_f()`, the lowest live rung
whose budget covers the plane (256 and 512 counting as live on the direct path only where
the resident weight cube is at most 16 granules, and never on the depthwise one, per
above), refusing the task when the plane needs more than the data cap
or when the rung would starve the weight path, because the recourse (a shorter row
window, an ic split) is the caller's to choose. `rocket_rk3576_max_task_rows()` is the
tiler-facing half: the tallest window one task can carry at the highest rung the weight
slice leaves room for. Planning reproduces both captured words, so the register-fidelity
gate stays byte-identical, and it turns every plane between 4096 and 6144 granules from
silent corruption into an exact result, validated on the part at 4160, 4800, 5600 and
6144 granules, and at 6144 across three widths. `ROCKET_RK3576_CBUF_F` forces F and
bypasses both checks, which is how the allowance was characterised.

## The weight slice caps how many output-channel groups compute

The pool arithmetic above is characterised at one output-channel group, and it is not
sufficient on its own. A conv driving several groups loses the trailing ones well
before the slice reaches what the pool leaves it, and it loses them **one at a time**
as the slice grows: the leading groups come back bit-exact and the rest wrong, with a
full surface written, no fault and nothing in dmesg. It reads as an output-channel
defect rather than a capacity one, which is why it was first recorded as a limit
"that is not slice size".

It is the slice. The governing quantity is the resident weight slice
`32 * ic * kh * kw`, one output-channel group, and the same slice behaves
identically whichever `(ic, kh, kw)` produces it: `ic=512 k=3` and `ic=4608 k=1` both
give 144 KiB and both compute all four groups, while `ic=192 k=5` and `ic=544 k=3`
both sit near 150 KiB and both lose two. Sweeping `ic` at k=1 moves the slice in
1 KiB steps, which is what resolves the boundary at all; at k=3 the step is 9 KiB and
at k=5 it is 25, and the coarse sampling is what made the loss look like a jump from
four groups straight to two.

Measured at F=0 on a 4x2 plane at oc=128 (four groups), and cross-checked at k=3 and
k=5. Each row is the number of groups that come back bit-exact:

| slice | 144 KiB | 145 | 146 | 147 | 148 | 152 | 156 | 162 |
|---|---|---|---|---|---|---|---|---|
| groups exact | 4 | 3 | 3 | 3 | 3 | 2 | 2 | 1 |

So the usable rule is **`ic*kh*kw <= 4608`** (a 144 KiB slice), at which every output
channel count computes. Past it the emitter refuses rather than running, in
`rocket_rk3576_cbuf_f()`, against a small measured table keyed on the group count
(144 KiB at four groups or more, 148 at three, 156 at two); the recourse is an ic
split, which is the caller's to choose. A single group is left to the pool check
alone; it already lands on the measured single-group boundary (175 KiB computes,
200 KiB does not, and the pool reaches its limit at 192 KiB), and a graded
multi-group loss is by definition not a single-group effect.

What the shape of the loss suggests, and what the points do not settle: the boundary
falls as the group count rises, roughly as though each additional group costs a small
fixed staging allowance on top of the slice, which fits a weight path that stages the
next group while computing the current one. Two free parameters against six points is
not a mechanism, so the emitter carries the measured table rather than a formula.
[HW sweep, H96 MAX M9, measured 2026-07-25]

### The group count is a caller's choice, so the cap is one too

The table is read above as a bound on a shape. It is not; it is a bound on a
program, and the group count in it is what one program drives rather than what the
convolution needs. Split the output channels across submits and the group count per
submit falls with them, which raises the slice the part will take. That turns the
table upside down: it becomes a planner, mapping the slice a shape needs to the most
output channels one submit may carry.

```
slice <= 144 KiB   ->  no constraint      (four groups or more)
slice <= 148 KiB   ->  oc tile 96         (three groups)
slice <= 156 KiB   ->  oc tile 64         (two groups)
otherwise          ->  oc tile 32         (one group; the pool check governs)
```

`rocket_conv2d_int8_rk3576()` plans exactly that and costs one submit per tile. It
lifts fourteen of the emitter's refusals, and the two that reach furthest are the
ones the four-group rule puts well out of range: `ic=576 k=3` (162 KiB) and
`ic=5184 k=1` (162 KiB) are both bit-exact at four tiles of 32 channels. So
`ic*kh*kw <= 4608` is the single-program rule and not the part's; through the
library the bound is the CBUF pool at one group, and the rule is
**`32*ic*kh*kw <= 175 KiB`**, i.e. `ic*kh*kw <= 5600`.

One refusal survives the split and it is the one with nowhere left to go:
`ic=256 k=5` needs 200 KiB, which is past the pool at a single group, and the only
recourse there is an input-channel split that the on-chip requant forecloses, int8
partials cannot be summed without quantizing each one. That shape belongs on the
int32-output writer.

Held by `tests/rk3576_conv_lib_gate.c`, which drives the emitter gate's own shape
table through the library entries: 111 pass, the one refusal is asserted in both
directions, and every computed shape is bit-exact against a CPU model.
[HW sweep, H96 MAX M9, measured 2026-07-27]

The cube size is not the constraint, which the same sweep confirms: `ic=448 k=3
oc=128` has a 126 KiB slice under a 504 KiB cube and is exact on all four groups,
while `ic=256 k=5 oc=32` has a 200 KiB slice under a 200 KiB cube and does not fit at
all. Neither is group count on its own, `oc=256` at `ic=64` is eight groups and
exact.

**Everything downstream of the MAC works, and the bias path is bit-exact.** A
bias-only probe, features and weights zero, a per-channel bias, reproduces the
CPU model exactly across all 32 output channels (2048/2048 bytes, max |diff| 0),
and the OUT_CVT sweep moves it exactly as offset/scale/shift should. Getting
there required the coefficient-buffer layout below.

**The OUT_CVT requant rounds to nearest; it does not truncate.** A border whose
exact value is -10.0006 comes back as -10, where an arithmetic shift gives -11.
Adding half an LSB before the shift, `(acc*scale + (1<<(shift-1))) >> shift`,
reproduces every pixel of the probes above exactly, and without it every
otherwise-correct surface is off by one wherever the fraction crosses a half.
[HW sweep, H96]

**All four operand DMAs fire and reach their programmed addresses.** Pointing the
weight base (`0x1110`), the feature base (`0x1088`), the output base (`0x4018`)
or the bias base (`0x5020`) at an unmapped IOVA makes that job fault, each with
its own IOMMU status word, while an untouched run is clean. The output and bias
bases are the positive controls: they are known to be read and written, so a
"no fault" reading elsewhere would have been evidence and a fault everywhere is
the confirmation that no operand loader is silently idle. The fault signature is
two `rk_iommu ... Enable stall request timed out` lines. [HW sweep, H96]

**That fault is not as recoverable as the next job makes it look.** The job
immediately after it computes normally, which is what the "recoverable" reading
was based on, but a session that ends with these faults can leave the part in a
state where the weight loader never arms again: every later job completes cleanly,
with no timeout and no new dmesg line, and writes a bias-only surface. It survived
7 hours of idle and a full `rmmod rocket` / `modprobe rocket` cycle, and only a
reboot cleared it. Budget a reboot after any unmapped-IOVA probing, and re-run a
known-good conv before trusting a negative result taken afterwards. [HW, H96]

**The feature strides are confirmed one axis at a time.** A uniform feature fill
proves nothing about addressing, every read that lands inside the buffer returns
the same byte, so each axis was varied with the other two held flat. Varying
along rows, along columns, and across channel *groups* each reproduces the CPU
model bit-exactly, which confirms in turn the line stride (`0x1090` = `iw*4`, in
4-byte units), the C2=16 channel atom inside a row, and the CBUF surface/group
stride (`0x1094`/`0x1098` = `iw*ih`, in 16-byte units). [HW sweep, H96]

**The bias/coefficient buffer at `BS_BASE_ADDR` (`0x5020`) is not a flat per-OC
int32 array.** It is the structure the vendor's own coefficient buffer uses:
**groups of 64 bytes covering 8 output channels each**, holding

| field | type | offset in group |
|---|---|---|
| `A[oc]`, per-channel bias term | int32 | `(oc%8)*4` |
| `B[oc]`, weight-zero-point correction | int16 | `32 + (oc%8)*2` |
| `C[oc]`, per-channel multiplier | int16 | `48 + (oc%8)*2` |

so `oc` lives at group `oc/8`. Handing the part a flat int32 array instead makes
it read that array *as* this structure: only the first 16 channels get a term at
all, on alternating channels, at 1024x the intended magnitude, an artifact, not
a 2-byte operand DMA.

**The BS stage adds `A` and then multiplies by `C`**: the surface is
`(acc + A[oc] + B[oc]*sum(x)) * C[oc]`. Read off the part against a known accumulator
and a known bias with `C` at 1 on the even channels and 2 on the odd:
`(acc + A)*C` explains 32 of 32 channels and `acc*C + A` explains only the 16 where `C`
is 1. So a bias quantized in the accumulator domain rides the per-channel gain for free
and must **not** be pre-divided by it; getting the order backwards scales the bias by
the wrong channel gain, which is a plausible surface rather than a fault.
[HW sweep, H96, `tests/rk3576_coeff_c.c`]

**That product is int32 and saturates.** Walking `(acc + A)*C` across `2^31` at a fixed
accumulator, every inexact cell implies the same ceiling, 2.147e9 to 2.158e9 against
`2^31` = 2.1475e9, and none of them wraps. So `|(acc + A)*C| <= INT32_MAX` is a bound a
planner can stay inside rather than a cliff, and it is what caps how much precision a
per-channel gain can carry: `C[oc] <= INT32_MAX / max|acc + A|`, which falls as the
layer's fan-in grows. [HW sweep, H96]

**`C` is genuinely per channel, and it gates the whole BS stage.** Every one of 32
channels reads its own `C` at 32 distinct values. At `C=1` the datapath is
bit-exact and `C=4` scales by exactly 4, so it is a live linear per-channel
multiplier, but at `C=0` the DPU writes a full, correctly sized, entirely
**empty** surface no matter what the CNA and the MAC did, for a conv with no bias
at all. An all-zero coefficient buffer is what any caller who does not know about
this layout hands over, and the resulting empty surface is indistinguishable by
inspection from a wrong geometry encoder; it is the single most expensive trap on
this part. Pack the buffer with `rocket_rk3576_pack_coeff()` rather than by hand.
`B` is pinned to 0, which is what symmetric quantization wants; it is unvalidated
against a non-zero weight zero point, and nothing observed so far needs it.
[HW sweep, H96]

**How `C` is read depends on the precision, so `C = 1` is not portable across it.** The
integer multiplier above is the int8 program. On a float program the same field is read
as fp16, where the integer `1` is the denormal 6e-8 and underflows the surface to empty,
the identical signature, from arithmetic rather than from a gate. fp16 `1.0` is `0x3C00`.
See [the fp16 datapath](#fp16-the-datapath-and-the-contraction-width-that-bounds-it).

**The feature domain is signed int8 and needs no centering.** A k=1 conv whose
output plane sweeps every int8 value once reproduces the CPU model over the whole
range: one flat run from -128 to 127, 8192/8192, max |diff| 0, with the probe
feeding raw signed bytes and no `+0x80` applied anywhere. The DPU epilogue is
exact for negative accumulators independently: a bias-only probe driven with
negative per-channel biases, which the MAC never touches, is bit-exact on all 32
channels. **The datapath is not uint8-centered**, and a feature tensor carrying
negatives is not wrong for it. A run that says otherwise was made against an
all-zero coefficient buffer, where the surface is empty whatever the features are. [HW sweep, H96]

**The padded border is exact.** A k=3 SAME 32x32 conv with dense random weights
reproduces the CPU model over the whole surface, ring included, 32768/32768, max
|diff| 0, as does the pad probe driving the border with a -128 pad tap against
zero features. `0x1084`, the pad word `0x1080` and the window geometry are all
right. That border defect is the same coefficient-buffer
artifact. [HW sweep, H96]

## The row window, and the three registers it closes

A plane over its allowance is not a slow conv, it is a wrong one, so the recourse the
allowance planner names has to exist. It is a split by input rows: each task reads a
row window of the full plane and writes the output rows that window supports, with the
full plane still described alongside the window. `rocket_rk3576_plan_rows()` lays the
sequence out and the emitter takes the window in `conv_params_t` `ih`/`oh` against
`ih_full`/`oh_full`.

The caller's whole job per task is two byte offsets, and both are plain row strides,
`iy0*iw*16` into the feature cube and `oy0*ow*16` into the output. The cubes are
NC1HWC2 with a 16-byte channel atom and the CNA takes the DDR group stride from the
full plane (`0x1094` = `iw*ih_full` by default), so one base plus a row offset addresses
that row of every channel group. `0x1094` is a quantity the emitter fills rather than a
derivation the hardware repeats: **the part honours any value at or above the plane**,
which is what lets a consumer read a producer's surface whose groups are further apart
than its own plane would put them, bit-exact at `+3`, `+16` and `+64` elements over five
geometries, one row task and ten, against a control that lays the same padded buffer out
without setting the register and differs every time. `0x1098` is not read as a second DDR
stride: the padded cases are bit-exact with it left at `round4(iw*fetch_rows)`.
[HW sweep, H96 MAX M9, `tests/rk3576_surf_stride.c`] The vendor's sliced capture is the direct evidence on the
feature side: its fourth task reads input rows 111.. and `0x1088` carries exactly that
row offset.

Driving it settles the three registers that only take a second value on a windowed
program. All three were measured with an `oc=64` conv, two output-channel groups, so
group 1 is exposed, cut into 32-, 16- and 8-row windows of the same 64-row plane.
[HW sweep, H96 MAX M9]

**`0x40B8` is the channel-group jump: `ow * (2*oh_full - oh_task)`, and that form is
now swept rather than fitted.** It was pinned at a single output-channel count, which
left open whether it held for a conv driving many groups at a wide kernel. It does:
the form is bit-exact across 2, 4, 8 and 16 output-channel groups, k1/k3/k5, stride 1
and 2, VALID and SAME, wide, tall and ragged planes, ic 32/64/128, and 3 to 16
windows, the `surface` group of the conv gate. The description below is what the
sweep confirms. [HW sweep, H96 MAX M9, measured 2026-07-25]

A full
destination surface plus the rows of it this task does not write. The writer walks the
task's rows and then adds this to reach the same rows of the next group, so a windowed
task has to be told about the rows it skipped or every group past the first lands
short, which is why an `oc=32` split is exact while `oc=64` is half wrong, and why
the defect reads as an output-channel fault rather than a windowing one. Only this
form is bit-exact at all three window sizes; `ow*oh_full`, `ow*oh_task`, `2*ow*oh_full`
and 0 each corrupt the whole surface. It reduces to `ow*oh_full` when the task is the
whole plane, which is what both full-plane captures carry, the register reads like a
constant until a window separates the two terms.

**`0x1018` carries one live bit, and it is not the byte pair.** Bit 30 is required:
clearing it corrupts the whole surface. The low 16 bits are don't-care on the direct
int8 path, `0x0404`, `0x0505`, `0x040b` and `0x0000` are all bit-exact, windowed and
un-windowed alike. The windowed value does not stop the DPU on an un-windowed task;
that reading is the cold-start wall, which is the trap this sheet warns about. Read a
single "the DPU did not write" as the wall and re-run before believing it.

**`0x1038` likewise carries one live bit, bit 31.** `0x07`, `0x010e` and `0x10` are all
bit-exact; `0x80000010` corrupts the surface.

The emitter still reproduces the vendor's values for all three, so the
register-fidelity gate stays byte-identical to the captures.

**What the split computes.** Bit-exact against the CPU model over the whole surface at
every shape tried: 112x112 and 224x224 at ic=32 (the geometries with no single-task
plan at all, 6272 and 25088 granules against a 6144 cap), and forced-cap splits from
2 to 16 windows across ic/oc of 32/64/128, k1/k3/k5, stride 1 and 2, VALID and SAME.
The planner spreads the output rows evenly over the fewest tasks that fit rather than
taking greedy maximum windows, because a greedy pass leaves a ragged tail, a 112-row
plane at a 109-row cap comes out 109+2+1 instead of two windows of 56, and every extra
task costs a submit.

**What a window costs.** With the cold-start wall closed, the same 64x64 ic64 oc64 k3
SAME output cut into progressively more windows and submitted with no inter-task gap:
[HW, H96 MAX M9, measured 2026-07-25]

| windows | 2 | 4 | 5 | 11 | 32 |
|---|---|---|---|---|---|
| wall time | 2.5 ms | 4.8 ms | 6.0 ms | 13.0 ms | 38.7 ms |

Flat at ~1.2 ms per window, so a split is linear in windows and the fewest-that-fit rule
is the right one. Almost all of that is submit-and-wait: emitting a window's regcmd on
the host is 0.02-0.05 ms, and a single-task conv costs 1.2 ms whether its output is 2 KB
or 1.6 MB. The floor is the driver's **completion poll**, not the silicon; this part has
no maskable completion IRQ, so `rocket` polls on an hrtimer, and shortening that interval
moves the per-task cost with it almost exactly. What makes it safe to shorten is retiring
on the DPU's own completion rather than on `PC_DONE`, which fires before the DPU's writes
have drained: `patches/rk3576/npu/0012` does that and takes the period to 50 us, and 0011
removes the IOMMU half of the hazard. See "The per-submit floor is the driver's
completion poll" below. The per-task figures in this section predate 0012 and are ~1.2 ms
where the current floor is ~0.44 ms.

## `ih_full` is not optional on a single-task plan

`rocket_rk3576_plan_rows()` can return one task whose row window is shorter than the
plane, and a caller that sets `ih_full`/`oh_full` only when the plan splits then
mis-programs the DDR channel-group stride. This is ordinary geometry, not an edge
case: any stride greater than 1 whose output does not consume the plane leaves
trailing input rows unread, so a 32x32 k1 s2 VALID conv plans one task over 31 of its
32 rows. With `ih_full` left at 0 the emitter takes the group stride from the window,
`0x1094` comes out `iw*31` instead of `iw*32`, and every channel group past the first
reads at the wrong offset.

The symptom is a full, correctly sized surface bearing no relation to the input, no
fault, no dmesg line, and identical in appearance to a broken geometry encoder. It is
worth stating because the un-windowed path in `rk3576_first_light` sets the window
directly and never trips it, so the defect only appears once a caller starts routing
every conv through the planner, which is what a tiler does. Set `ih_full`/`oh_full`
from the plane on every task. [HW, H96 MAX M9]

## The first-conv ARGB sub-encoding

A convolution whose input is a packed image, 3 or 4 interleaved bytes per pixel
rather than an NC1HWC2 cube, runs on its own CNA datapath. It is not the normal
program at a small channel count; about sixteen registers are packed differently.

It is the only way a vision model's stem runs on this part at all: the normal path
needs `ic` a multiple of 32 (see below), and an image is 3. Twelve captured programs
pin it, at two image sizes (224x224 and 64x64) and two row splits.

The RK3588 has the same datapath, Mesa drives it as `CNA_CONV_CON1` `NONALIGN_DMA |
GROUP_LINE_OFF | ARGB_IN(8)` for a 1-channel input, and `0x100C` is one of the few
geometry registers the RK3576 does not re-pack: `GROUP_LINE_OFF` is bit 29 and
`ARGB_IN` is bits[15:12] on both parts. The captured `ARGB_IN` of `0xA` sits exactly
where Mesa's 1-channel `0x8` does, one step per extra image channel, so the field is
`8 | (image_channels - 1)`. `CONV_MODE` is 6, a third value beside direct (0) and
depthwise (1).

**What the datapath does.** The CNA reads the packed row straight out of DDR and the
CVT, bypassed on every other layer, expands each pixel to four int8 lanes while
applying a per-channel scale and offset. The kernel's horizontal extent is then
folded into the channel axis: the conv the MAC sees is `kh x 1` over `4*kw` channels.
That is why these programs carry two disagreeing channel counts, which is the
sub-encoding's clearest signature in a capture.

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
| `0x1084` pad const | `zp - 0x80`, already centred | raw `zp + 0x80`, one byte per channel |
| `0x1090` line stride | `iw*4` (a cube row in 4-byte words) | `iw*ic/16` (the packed row in granules) |
| `0x1078` hi | `iw-1` | `iw*ic/16 - 1` |
| `0x1094`, `0x1098` | plane / task surface strides | both `line_stride * ih` |
| `0x118C` | `(iw-1)<<16 \| (ih_full-1)` | `(entries-1)<<16 \| (entries-2)` |
| CORE `0x3018` | `0x10000001` | `0x10000081` |

Everything past the CNA is the direct path's, including `0x40B8`'s channel-group
jump, so this is a CNA sub-encoding and not a second pipeline.

Three things follow that a caller has to act on. The feature buffer is a packed
image, `iw*ic` bytes per row, so a row-window plan's feature offsets are in those
units. The pixels are raw uint8 and the converter does the centring, which is why
`0x1084` pads in the raw byte domain, the pad is inserted before the CVT runs. And
`iw` must be a multiple of 16, because both the DDR row stride and the CBUF row are
counted in 16-byte granules.

The `+1` on the CBUF row is the fold's lookahead: the row is the image row expanded
to 4 lanes per pixel (`iw*4` bytes, `iw/16` granules) plus one granule, because the
last output column reads `kw` pixels and reaches past the granule its own pixel sits
in. Both captured widths carry exactly `+1`.

**`0x118C` is not a plane extent here**, which is worth stating separately because a
normal-path decode of an ARGB capture reads a nonsense full-plane height out of it.
An ARGB program carries no plane height anywhere: the packed image is a single
surface, so `0x1094` and `0x1098` both hold the task's own rows and there is no
channel-group stride to describe.

**The one geometry inference.** Every ARGB capture is `kw=3`, so `4*kw` and a
constant 12 fit the programmed channel count equally, as do several readings of the
48-element weight kernel. The mechanism above, 4 lanes per pixel column, `kw`
columns, is what makes `4*kw` the reading rather than a curve fit, but a first conv
at another kernel width is unvalidated.

### The ARGB weight cube

The captures carry register programs, not weight BOs, so what the found programs pin
is only the cube's size and stride, `oc * kh * round16(4*kw)` bytes, one 16-byte row
per kernel row per output channel, and nothing about its byte order. The **float**
cube's layout is decoded, from captures manufactured with weights unique per position
(`tests/data/rk3576-vendor-capture/argb/mkargb.py`):

```
slot(oc, c, kh, kw) = (oc/16) * (KH*KW*64)
                    + kh * (KW*64)
                    + kw * 64
                    + (oc%16) * 4
                    + c
```

A weight in a **sixteen-bit slot**; **four lanes** per (output channel, tap), lane `c`
carrying image channel `c` with any lane past `ic` left don't-care; output channels
**interleaved in groups of sixteen** inside one tap; the tap axis kh-outer. The 64 is
16 output channels x 4 lanes. Note `4*kw` is **not** rounded up to 16 here, so this
layout's element count is not the byte size the register program declares, at
`oc=16 k=3` the cube is 576 halfwords where the declared size is 768 bytes.
Reproduced at `ic` 1/3/4, `k` 1/3/5/7 and `oc` 16/32/48/64, at four widths, three
heights and both strides. `rocket_rk3576_weight_argb_fp16()` is the packer.
[source-confirmed, RKNN-Toolkit2 rk3576 float build]

**The INT8 cube is not this one.** A quantized ARGB capture does take the path,
`0x100C` reads `0x2000a006`, exactly the found value, but its weights do not survive
quantization as a findable value set, so this cube was read off the part instead, with
an impulse image and a one-byte cube (`rk3576_conv_gate fcmap`). Every live byte then
names its output channel, both taps and its lane at once, and the answer is a
different object from the float cube in every axis but the lanes:

```
byte(oc, c, kh, kw) = (oc/32) * (KH * R * 32)
                    + kh * (32 * R)
                    + (oc%32) * R
                    + kw * 4
                    + c              R = round16(4*KW)
```

A weight in one byte; output channels grouped by **thirty-two**, not sixteen; the tap
row outside that group where the float cube puts the whole tap axis outermost; and the
tap column folded into the same `R`-byte row as the four lanes, `4*KW` of it live and
the rest padding the DMA still fetches. Lane `c` carries image channel `c`.

A bijection over every live byte at `oc` 32 and 64, `k` 3/5/7 and `ic` 3 and 4, 1152,
2304, 3200, 6272 and 864 live bytes, each landing on exactly one output position of
one channel, none left over and none doubled, and the map is translation-invariant
(the impulse moved three and six pixels reads the same). The 32-channel group is
observable only above one group: at `oc=32` a flat `oc*R` fits equally, and `oc=64` is
what separates them. `R` is separated from a constant 16 by `k=5` and `k=7`, where it
is 32. `rocket_rk3576_weight_argb_int8()` is the packer. [HW sweep, H96 MAX M9]

**The converter's offset is inert, and the packed byte is a plain signed int8.** The
table above lists `0x1054`-`0x105C` as the uint8 zero point the CVT subtracts, which is
what the datapath's description says, but every ARGB capture is zero point 0, so
nothing ever exercised those registers. Driven on the part they do nothing: an image
written at `raw = s + (zp + 0x80)` comes back with the raw byte read as a **signed
int8** and no subtraction at all, `0x80` reading -128, `0xC0` reading -64 and `0xFF`
reading -1. So a caller writes two's complement and folds the input zero point into
the coefficient group's `A` term exactly as the direct path does; the border constant
still has to be the stored zero point, so that a pad tap's true value is zero.
[HW sweep, H96 MAX M9]

### Four geometry bounds the int8 first conv adds

None of them appears in any capture, because every captured first conv is a 3x3
stride-2 SAME convolution and satisfies all four by construction. Three are silent
when violated, which is why they are refused at the library entry rather than left to
compute.

**The left pad must be non-zero.** At `pad_left = 0` the DPU writes nothing at all, an
untouched surface, not a wrong one, at every plane, stride, kernel and channel count
tried. `CNA_PAD_CON0` (`0x1080`) bits[15:8] decide it alone: forcing `0x0100` into a
zero-pad program makes that same program write, and forcing `0x0000` into a working one
stops it, while the `pad_top` field in bits[7:0] does neither. It is the reason a
TFLite-style SAME 3x3 stride-2 stem does not run and an ONNX-style symmetric one does:
TFLite puts the odd pad byte on the trailing edge and leaves `pad_left` at 0.

**The output width must be `iw/stride`.** Anything else writes a full, correctly sized
surface that is sheared, the tap a weight byte lands on drifts one output column per
output row, exactly as a row-stride mismatch does, and it shears for a narrow `ow` and
an over-padded wide one alike.

**The output width must also be a multiple of 16**, and that is not implied by `iw`
being one. At `ow` 24 and 56, both from an `iw` that is a multiple of 16, output row 0
is exact and every row after it is wrong; `ow` 16, 32, 48, 64, 80, 96 and 112 are all
exact. Taken with the rule above it means `iw` must be a multiple of `16*stride`. The
direct path carries no such rule; this one comes with the channel fold.

**One image channel is programmed as two**, and what forces that is the feature DMA's
row width rather than the mode word. `0x1078` bits[31:16] carry `line_stride - 1`, which
is right from `ic=2` up; at `ic=1` the DPU writes nothing at all, an untouched surface,
not a wrong one. Raising that field alone revives the write, at every plane, and nothing
is exact, because the DMA then reads past the packed row: at `iw=64` (4 granules) the
field must reach 4, at `iw=128` (8 granules) it must reach 8, and the best exactness any
value reaches is about 40% of the surface. So the row is widened instead of the register,
a second interleaved channel of zero samples against zero weights, and `ic=1` is then
bit-exact at 64x64, 128x128 and 224x224 with the same envelope as `ic=2`, refusing `k=1`,
`k=5` and `oc=16` exactly where two channels do. The arithmetic is untouched by the pad:
the MAC term is zero because the weight is, `sum_w` is unchanged so the coefficient `A`
is, and the asymmetric `B` multiplies a sum of raw samples that gains only zeros, so
the zero-point fold keeps the caller's tap count. The cost is one byte per pixel of host
packing and a doubled feature read. The mode word's `ARGB_IN` nibble is `8 | (ic-1)`, so
`ic=1` is also the one value leaving its low bits clear, but forcing that nibble to any
of the working values leaves the surface untouched: it is not what gates the write.
`tests/rk3576_argb_ic1.c` holds it, with the register sweep that found it.
[HW sweep, H96 MAX M9, measured 2026-07-28]

The output-channel count follows the direct path's multiple-of-32 rule (`oc=16` writes
nothing) but not the float first conv's 32-channel per-program cap: one int8 program
delivers 64 output channels. Above that the caller splits, as on the float path.

`rocket_conv2d_int8_rk3576()` owns all of it, the packed image, the cube, the row
window, the output-channel split, the zero-point fold and the de-scatter, and is
bit-exact against a CPU model at twenty shapes, including the 224x224 stride-2 stem at
`oc` 32 and 64 and at `k` 3 and 7 (`rk3576_conv_lib_gate`, the `fq` group, which also
asserts all seven refusals in both directions). [HW sweep, H96 MAX M9]

**A capture is an oracle for a whole program, and running one verbatim is what broke
this open.** Our emitter reproduced twelve int8 ARGB captures register for register and
still wrote nothing at every geometry we tried, which read as a dead datapath. Taking
the captured op stream out of the golden table, patching only the five address
registers and submitting it (`tests/rk3576_fc_vendor.c`) showed the vendor's own
program writing its whole surface, so the mode worked and the extrapolation away from
the captured geometry was what did not. Every bound above came out of walking from that
geometry back toward a small one, one axis at a time.

**One correction to the transcribed program.** The low field of `0x100C` is not
precision-independent: the found int8 ARGB programs carry `0x2000a006` and the float
ones `0x2020a122`, so it is **6 at int8 and 2 at fp16**: bit 2 clears on the float
path while `GROUP_LINE_OFF` and `ARGB_IN` stay where they are. `ARGB_IN` tracks the
channel count as `8 | (ic-1)`: `0x8` at 1 channel, `0xA` at 3 and `0xB` at 4,
confirming the reading that Mesa's 1-channel `0x8` is the same field.

### The fp16 first conv is a different program, not the int8 one at another precision

Six fields move with the precision, and the first two are the shape of the datapath
rather than a width. Manufactured captures separate every one of them: `iw` at 16, 32
and 64, `ih` at 16, 32 and 48, both strides, `k` 1/3/5/7 and `oc` 16/32/48/64, each
varying one axis against the rest. [source-confirmed, RKNN-Toolkit2 rk3576 float build]

| | int8 | fp16 |
|---|---|---|
| mode nibble (`0x100C` low) | 6 | 2 |
| programmed channels (`0x1028` low) | `4*kw`, the kernel width folded into the channel axis | `ic`, the image's own count |
| CBUF granules per row (`0x103C` hi) | `iw*4/64 + 1` | `iw*4*2/64`, no `+1` |
| `0x1044` low | the granule count | `iw` |
| `0x118C` | `(entries-1, entries-2)` | `(entries-1, entries-1)` |
| DDR row stride (`0x1090`) | `iw*ic/16` | `iw*ic*2/16` |
| CVT | runs: Q14 unity, the uint8 zero point as a per-channel offset | **bypassed**, offsets zero |

**There is no channel fold at fp16.** The int8 path folds the kernel's width into the
channel axis, 4 lanes per pixel column, `kw` columns, so 12 at `kw=3` against a feature
DMA of 3, and the float path does not: the programmed count is the image's, and the
taps stay on the kernel axis. The `+1` granule goes with the fold, because it is the
fold's lookahead (the last output column of a row reads `kw` pixels, which reach past
its own granule), and a float program at `k` 1, 3, 5 and 7 carries the same entry count
at every kernel size. The decoded weight cube says the same thing independently: it
carries explicit `kh` and `kw` axes with four lanes inside a tap, where a folded cube
would carry the columns inside the lanes.

**The input is an fp16 packed image**: 3 or 4 interleaved halfwords per pixel, which
is why the row stride takes the element size. The CVT is bypassed rather than
configured, since a float image is already in the value domain the MAC wants; the
per-channel truncates and Q14 scales stay in `0x1048` under the bypass, where they are
inert.

**Weight bytes are the same two registers on both paths, read correctly.** `0x1020` is
the bytes one output channel occupies, the element count scaled by the element size,
and `0x1030`'s high half is twice the raw element count. They coincide at fp16 and
differ by the factor of two at int8, which is why one can be mistaken for the other. The
element count per output channel is `ic*kh*kw` on the direct path, `4*kh*kw` on the float
first conv (dense, no round-up of the lane group to 16) and `kh*round16(4*kw)` on the
int8 one. Depthwise is its own case and must not be scaled: a depthwise weight occupies
a 16-bit slot whatever the precision, so its count is already the float one.

### Thirty-two output channels per first-conv program

One first-conv program delivers **32 output channels and no more**. At `oc` 48, 64 and
96 exactly 32 whole channels come back bit-exact and the rest of the surface is never
written, a contiguous prefix, not the interleave the int32 writer's byte budget gives,
while `oc` 24 and 32 are complete. [HW sweep, H96 MAX M9]

This is not a register the emitter gets wrong. The program matches the vendor's own
`oc=48` and `oc=64` captures register for register, and the weight cube reproduces those
captures too, so the vendor's compiler emits single programs the part does not fully
execute. The recourse is the one the direct path already uses for its weight slice:
split the output channels, one submit per tile of 32, each an independent convolution
over its own channels. `rocket_conv2d_fp16_rk3576()` does it, and the output cube's
channel groups are contiguous planes so a tile is the same BO at a plane offset.

The row window composes with it on the other axis. `rocket_rk3576_plan_rows_prec()` is
the planner at a precision: it changes the CBUF entry count per row, the row cap the
allowance affords, and, on this path, the feature offsets, since a float packed image
is `ic` interleaved halfwords per pixel where an int8 one is bytes. A 224x224 plane is
6272 granules against a 6144 ceiling and has no single-task plan; with the window it is
bit-exact at `k` 3 and 7 and at `oc` 32 and 64, the two splits running together.

### The vendor's float BS arrangement is not this library's

Three DPU_RDMA words carry a different constant in every vendor float program. Two of
them, `0x5034` at `0x1` and `0x5044` at `0x40010050`, against the integer path's `0x41`
and `0x40000010`, are reproduced and are bit-exact on the part.

The third is the case that shows register fidelity is not the goal. **`0x501C`
(BRDMA_CFG), the BS operand reader, reads `0x100` in every vendor float program against
the integer path's `0x710`, and emitting the vendor's value makes the DPU write nothing
at all**, an untouched surface, no fault, nothing in dmesg. It does not stand alone:
paired with the BS ALU config those same programs carry (`0x4044 = 2`, where the integer
path uses 1) the writer runs again, but the arithmetic is then wrong on about a tenth of
the surface (1866/2048, worst relative error 0.016). `0x4044 = 2` on its own, with
`0x501C` at `0x710`, is bit-exact. So the vendor's float epilogue is a different BS
*arrangement* rather than a different constant, and this library's A/B/C coefficient
group is packed for the integer one. Keep `0x710` until the arrangement is decoded as a
whole. [HW sweep, H96 MAX M9; leave-one-out over the six float-path registers]

Two more float-path corrections, both bit-exact on the part: CORE `0x3018` bit 0 is an
**int8 marker** that every float program clears, and DPU `0x4010`'s middle field (the
DPU's input width) stays **zero** on the float path where only the output width and the
processing precision move.

### The compiler takes the packed path at 1, 3 and 4 channels, and not at 2

A 2-channel float conv is compiled down the direct path, no `GROUP_LINE_OFF`, no
`ARGB_IN`, where 1, 3 and 4 all take the packed one. That is a toolkit preference and
not a hardware bound: the packed path computes bit-exactly at two channels on the part
(`rk3576_conv_lib_gate fc-2ch-k3`), so the library keeps taking it for every count at or
below four. Worth knowing when reading a capture set: a channel count with no ARGB
capture in it is not evidence the part refuses that count.

## CBUF row reuse across a windowed sequence

The emitter's own windows refetch: each task fetches its whole window against a CBUF
base of zero, which is self-consistent and bit-exact on the part. The vendor's
programs do something else, and twelve captured continuation tasks pin it exactly.

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

and the feature address (`0x1088`) points at the first new row rather than at the
window start. The vendor's `dw` capture is the clean case: its second task reads
input rows 89..111 of a 112-row plane, retains 2, and carries `0x1088` = row 91.

`retained` is the window arithmetic, the previous window's end minus this window's
start, which at these geometries equals `kh - stride_y`. When it is zero the vendor
does not continue the base, it resets to the sequence's origin, which is what a k=1
continuation carries.

`rocket_rk3576_plan_rows()` fills `retained` and `cbuf_resident` on every task and
`gen_conv2d_int8_rk3576_reuse()` emits it. It is opt-in, and the reason is worth
stating: it saves `entries * retained` granules of feature DMA per continuation task,
under a tenth of the window's traffic at the captured geometries, against a window
cost of ~1.2 ms that is the driver's completion poll and not the fetch. So it does
not shorten a windowed conv measurably. What it buys is that the vendor's split
programs become a complete oracle: seven registers the gate's diff cannot otherwise
reach are checked exactly across thirteen
continuation tasks, at three kernel sizes, both strides, both channel counts and all
three CBUF origins. UNVALIDATED on silicon: a wrong base reads resident rows that are
not there and corrupts silently, so the refetching path stays the default.

**Three-task sequences are not pinned.** Every captured split is two tasks, so that
`cbuf_resident` accumulates over a longer sequence is the natural extension of the
same arithmetic and nothing more.

## The same graph, compiled three times

Each `.rknn` capture holds its graph compiled three times, and the difference is
never the geometry. Two of the three are the same per-layer task split at a
different CBUF origin, 0 and 7168 granules, added to every base in the table above.
The third splits the whole chain instead of each layer: it runs all five layers over
the top half of the image and then all five over the bottom half, from a CBUF origin
of 6144, and it is the one that sets `0x1014` bit 28 and drops the early duplicate
`0x1038` preamble write (138 register writes rather than 139).

Reading the third as the two-core compile, 7168 granules being one core's whole
CBUF pool, an image-plane split being the standard way to use two, fits, and is not
evidence. `0x1014` bit 28 sits where the RK3588 puts `NN_MODE`, which is consistent
with a mode bit and says nothing about which mode. All of it is decodable off-device
and none of it is decided by these captures.

For an emitter none of this matters: the origins, the format low bits (`0x1018`,
`0x1038`) and the allowance field are the vendor allocator's per-compile choices,
and a hardware sweep already found only `0x1018` bit 30 and `0x1038` bit 31 live on
the int8 path.

## The DPU epilogue: what an appended stage moves

Three captures are the same conv as `conv2d_rk3576.rknn` with one extra op appended,
a bias add, a scale, an eltwise sum, which makes them a controlled experiment the
vendor ran for us. The delta against the plain conv is three registers, and it is the
SAME three, with the same values, for all three ops:

| Register | plain conv | conv + an epilogue stage |
|---|---|---|
| `0x4044` BS ALU operand | `1` | `0` |
| `0x4050` BS config | `0x80011111` | `0x80021111` (bits[19:16] 1 -> 2) |
| `0x501C` BRDMA config | `0x710` | `0x114` |

That the three ops are indistinguishable here is the finding. These registers do not
encode which epilogue op runs; a per-output-channel bias, a per-channel scale and a
per-channel sum all lower onto one BS-stage configuration, and what separates them is
the coefficient buffer's contents and the requant triple (`0x40AC`-`0x40B4`), which
is the only other thing that differs between the three captures.

Neither `0x4050` nor `0x501C` decodes under the RK3588's field map at these offsets,
`0x80011111` sets bit 12, which is reserved in the RK3588's `BS_CFG`, and the
RK3576's `BRDMA_CFG` is wider than the RK3588's 5-bit field, so both are
transcribed constants here rather than decomposed. The reading that fits is that the
BS stage takes its second operand from `DPU_RDMA` rather than from the inline path,
which is the same stage and the same buffer as the asymmetric-weight `B` term.

### What the three registers do: `0x501C` moves the BS operand source

Driven against a conv carrying a known non-zero `B` (`rk3576_conv_gate` with
`ROCKET_G_WZP=40`, which reports how many outputs each candidate sign convention
explains) and taken one register at a time, the bundle separates cleanly.
[HW sweep, H96 MAX M9]

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

**`0x501C` is the operand-source register and the other two are inert without it.**
`0x4044` and `0x4050` change nothing at all on the plain path, not one output of a
32768-element surface moves, down to the incidental 537 the no-B model always
coincides on, while `0x501C` alone stops the DPU writing: "task never wrote its rows"
over four retries, no fault, nothing in dmesg. It needs a matching BS config to write
again, and which one depends on the value: the epilogue's `0x114` is re-enabled by
`0x4050` bits[19:16] = 2, and the float path's `0x100` by `0x4044` = 2.

**Under the epilogue arrangement the coefficient group's `B` is not read.**
`+B*sum(x)` falls from every output to 1423, and **exactly one output in four** matches
the model with no B term at all, 8192/32768, 7200/28800 and 512/2048 on three
different shapes, a quarter each time. So these registers do not select an epilogue
*op*, which the three captures already showed; they select where the BS stage takes its
second operand from, and the quarter is what a buffer packed for the inline positions
looks like when read as the other kind of stream.

That is the same mechanism as the float path's refusal to write, and it makes three BS
arrangements visible rather than two:

| arrangement | `0x4044` | `0x4050` | `0x501C` |
|---|---|---|---|
| plain int8 conv | 1 | `0x80011111` | `0x710` |
| int8 conv + an epilogue stage | 0 | `0x80021111` | `0x114` |
| vendor float conv | 2 | `0x00021111` | `0x100` |

This library packs its A/B/C group for the first, which is bit-exact at both
precisions. Decoding the other two would say what the vendor's epilogue and float
paths buy; neither is needed to compute.

## Depthwise: the envelope, and the two buffers that are not the direct path's

`gen_conv2d_dw_int8_rk3576()` computes bit-exactly. Its register program reproduces
the vendor across 88 captured task programs with no field left open, and that was
never sufficient, because the two things it was still wrong about are buffers rather
than registers. A `.rknn` capture carries a register program; the coefficient group
and the weight cube are memory the registers point at, and a capture says nothing
about either. Both differ from the direct path's, both are silent when wrong, and
both had to be read off the part.

### The envelope

Bit-exact against a CPU model over the whole surface, every shape `maxdiff 0`, at
`rk3576_conv_gate dw`, 35 shapes, gap-free with no cold-start-wall retries.
[HW sweep, H96 MAX M9]

| axis | covered |
|---|---|
| channels | 8, 16, 24, 32, 48, 64, 72, 80, 96, 112, 128, 144, 176, 256 |
| kernel | 1, 3, 5, 7 |
| stride | 1 and 2 |
| padding | SAME and VALID |
| plane | 16x16 to 112x112, plus 15x18, 17x19 and 19x19 |
| row window | 1, 2, 3 and 6 tasks |

A depthwise task costs about 1.1-1.2 ms of NPU wall whatever the channel count, the
kernel or the plane, the same per-submit dispatch floor the direct path pays, with no
compute term visible at these shapes. A row window costs that per task.

The awkward numbers in that table are the point of it. **Channel counts that are not
multiples of 32** are where this path's granules stop agreeing, the weight cube
rounds to 16, the CBUF allocation takes a 16-group count of 3 mod 4 one group
further, and `0x4050`'s 2-bit group field wraps, and a gate that rounds every
depthwise count up to 32, as this one used to, exercises none of that. **Planes whose
`ow*oh` is not a multiple of four** are where the padded output surface stride below
shows. Both were invisible at every shape the path was checked at before.

### The coefficient group is 48 bytes and has no B

The direct path's group is 64 bytes for 8 output channels: `A` (int32 bias) at
`(oc%8)*4`, `B` (weight zero point) at `+32`, `C` (int16 multiplier) at `+48`. The
depthwise group covers the same 8 channels in **48 bytes** and carries no `B` at all:

```
A[oc]  int32  at (oc%8)*4
C[oc]  int16  at 32 + (oc%8)*2
```

with `oc` in group `oc/8` and the group stride 48. `rocket_rk3576_pack_coeff_dw()`
and `rocket_rk3576_coeff_bytes_dw()`. [HW sweep, H96 MAX M9]

Hand the depthwise path the 64-byte group and the two indices **drift apart at
different rates**, `A` strides 4 bytes per channel and `C` strides 2, so past the
first eight channels each is read out of a different group. The BS stage computes
`(acc + A) * C`, so what comes back is: most channels multiplying a bias by a zero
and never reaching DDR at all, a few whose `C` lands in the `A` region squaring their
own bias, and one group's `C` multipliers read as the next group's biases. That is a
correctly sized, mostly empty surface with no fault to catch it, the same signature
a wrong geometry register gives, which is why it read for so long as a datapath
defect rather than a packing one.

### The B term is added: `acc + B*sum(x)`

Every vendor capture carries `B = 0`, so the field's existence was read off its position
and width rather than off a program that uses it, and the sign convention was open. It is
not now. Driving `B` on the direct path against a CPU model, `acc + B*sum(x)`, where
`sum(x)` is the sum of the input elements the output contracted, explains **every**
output of **every** direct shape in the conv gate's envelope group, at `B` = 1, 40, 127,
-37 and -128. `-B*sum(x)` and an inert `B` explain only the few percent that coincide,
which at small `|B|` is larger simply because the correction often rounds to the same
int8. `ROCKET_G_WZP` in `tests/rk3576_conv_gate.c` drives it and scores all three models.
[HW sweep, H96 MAX M9, measured 2026-07-27]

**So a weight zero point is programmed negated.** An asymmetric weight is
`w_true = w_stored - wzp` and its correction is `-wzp*sum(x)`, so pass `B = -wzp` to
`rocket_rk3576_pack_coeff_asym()`, whose parameter is named `b_term` for exactly that
reason. Getting it backwards gives a plausible surface with a bias-shaped error and
nothing that faults.

There is nowhere in the DEPTHWISE group to put a weight zero point, so **an asymmetric
depthwise weight has to be folded into the bias**; there is no `_asym` form.

### The weight cube

Two layouts, one per precision, sharing a block structure and nothing else.

The float cube is decoded, not inferred, from captures whose weights carry a unique
value per (channel, tap), at C = 24, 32, 48, 64 and 128 and at k = 3 and 5:

```
slot(c, kh, kw) = (c/32)*32*KH*KW + (kh*KW + kw)*held + (c%32)
held            = min(32, C - (c/32)*32)
```

- a weight occupies a **16-bit slot**, which is why one geometry's float and int8
  cubes are the same size, and why `WEIGHT_BYTES` carries a factor of 2 that is not
  the element size;
- channels group by **32**, each group a contiguous block;
- inside a group the order is **tap-major**, kh outer;
- a trailing partial group is **dense**; its tap stride is the channels it holds,
  not 32, so C=48 is 432 slots and not 576.

This is `rocket_rk3576_weight_dw()`; `dw/named/mknamed.py --decode` re-derives it from
the committed captures without a board. The naming has to be carried in float weights:
per-channel weight quantization normalizes each channel by its own maximum, which
erases any naming carried in magnitude, while fp16 represents integers exactly to 2048
so a unique-integer ramp survives the float path untouched and is findable in the file
by its exact value set.

The INT8 cube is **not that cube's low byte**, and it is not the RK3588's
group-of-64 single-byte packing either. Read off the part with `rk3576_conv_gate
dwmap`, which drives an impulse feature against a cube that is zero but for one byte
and reports which output that byte reaches, over a C=32 k=3 cube every one of the 576
bytes reaches exactly one (channel, tap): [HW sweep, H96 MAX M9]

```
byte(c, kh, kw) = (c/64)*64*KH*KW*2 + (kh*KW + kw)*held*2 + 4*((c%64)/2) + (c%2)
held            = min(64, round16(C - (c/64)*64))
```

- the channel group is **64**, where the float cube's is 32;
- inside a tap block the channel a byte carries is `2*(b/4) + (b%2)`, so channel `c`
  owns two bytes per tap, at `4*(c/2) + (c%2)` and two further on, and **both are live
  and both contribute**, a weight written into both is added twice, so the packer
  writes the first and leaves the second at zero;
- a trailing partial group strides by what it holds **rounded up to 16**, where the
  float cube's is dense in the raw count. That is what makes the whole cube exactly
  `round16(C)*KH*KW*2` bytes.

This is `rocket_rk3576_weight_dw_int8()`, and it returns a byte offset where the float
entry point returns a slot. Writing an int8 weight as a 16-bit value into the float
slot puts its **sign extension into the byte that belongs to the next channel**: a
silent -1 weight on a neighbour rather than padding.

Two shapes hide the group size, and they are the ones that pass first: at a single
group the two layouts coincide, and at `k=1` there is one tap, so the group base and
the tap stride cannot be told apart. **C=64 k=3 is what separates 64 from 32**; there
`dwmap` predicts 576 of 576 (channel, tap) positions at 64 and 64 of 576 at 32. C=24
and C=72 are what separate the round16 partial group from the raw one.

### The output surface stride is padded to four

`0x401C` is the distance in elements from one 16-channel output group to the next, and
the depthwise path programs `round4(ow*oh_full)` where the direct path programs
`ow*oh_full`. It is a DDR stride, so **a caller has to size the output BO and
de-scatter with it**, `rocket_rk3576_out_surf_elems(ow, oh_full, dw)`. Assuming the
plane instead lands every group past the first up to four elements early, which reads
as "the first 16 channels are exact and the rest are noise": at C=32 exactly half the
surface, at C=64 exactly a quarter.

Invisible at any plane whose `ow*oh_full` is already a multiple of four, which is most
of them and was all of them, 16x16 and 112x112 pass, 15x18, 17x19 and 19x19 do not.
`0x40B8`'s plane term is four of these same rounded surfaces.

### The probes, and what each can and cannot see

`rk3576_conv_gate` carries five modes. Each has a `direct` control, and the control is
load-bearing: it runs the same reading on the path that is known bit-exact, so a mode
that cannot reproduce the direct map is not measuring what it claims to.

| mode | what it reads |
|---|---|
| `dwmap` | which output each weight byte reaches, one submit per byte |
| `dwbias` | which coefficient slot each channel read |
| `dwcoeff` | the inverse: which output bytes each coefficient position moves |
| `dwout` | the raw output BO, undescattered |
| `-l` | the shape table, without running it |

Two properties are what made them decode rather than score, and both are worth keeping
in any successor:

- **`dwcoeff` measures its own baseline.** It used to compare each position against
  the constant it had packed, which is a delta only where the part agrees the surface
  is flat. On a path whose surface is not flat every position reads as "changed" and
  the probe says nothing. One unprobed submit kept as a reference image makes a
  per-position delta meaningful whatever the baseline looks like; there is no need to
  flatten the hardware first.
- **`dwcoeff` and `dwout` read raw bytes.** Scoring a depthwise surface through the
  direct path's de-scatter hides exactly the permutation these modes are looking for.

`ROCKET_G_DWOUT_OUTSCALE` divides the requant down so a lane at the int8 clip names
its magnitude instead of reading as "large", and `ROCKET_G_DWOUT_BASE` / `_STEP` /
`_CBASE` / `_CSTEP` drive the `A` and `C` ramps independently. That pair is what
separates "this lane read the wrong bias" from "this lane read its multiplier out of
the bias region": with `A` constant at 5 and `C` at 1, a lane reading a real `C`
answers 5 and one reading `A` as its own `C` answers 25.

### Manufactured captures, and the four formulas they closed

Every depthwise question that was open here was open because the found captures
confound the axes. Building captures to order separates them, at 22 channel counts
from 8 to 256, strides 1 and 2, kernels 1/3/5/7 and rectangular planes.
[source-confirmed, RKNN-Toolkit2 rk3576 depthwise builds]

**Two channel granules, and they are different numbers.**

- The weight granule is `C` rounded up to 16. `WEIGHT_BYTES` (`0x101C`) is
  `round16(C)*kh*kw*2`, `WEIGHT_ELEMS` (`0x1020`) is `round16(C)*kh*kw`, and the
  weight-bytes-per-kernel term (`0x1030` high) is `kh*kw*round16(C)/8`, the last
  confirmed at k=1, 3, 5 and 7, where the found captures were all k=3 and could not
  tell `kh*kw*C/8` from `kh*kw*4`.
- The feature granule is `C` rounded up to 16 and then, if the resulting 16-channel
  group count is 3 mod 4, up one group further: 48 rounds to 64 and 112 to 128, while
  80 and 144 stay put. It sizes the CBUF entry counts (`0x1028` high, `0x103C` high,
  `0x1044` low). Read as hardware, channels are fetched in blocks of 64 and a
  trailing partial block holds one, two or four sixteens, never three.

**`0x4050`'s depthwise field tracks the channel count, not the stride.** Bits[9:8]
are the 16-channel group count minus one, modulo 4, computed on the *unrounded*
count, C=48 carries 2 there while its feature allocation is 64's. The found captures
could not attribute the field because every C=32 program in them is stride 1 and
every C=64 one is stride 2; manufactured ones at both strides show it flat in the
stride and stepping with C, wrapping at 80, 144 and 208. Emit the C=32 word
unconditionally and the BS stage above C=32 gets a word the vendor never uses, which
returns **a wholly untouched output BO**: no surface at all, no fault, no dmesg
line.

**`0x40B8`'s plane term is a whole destination surface, and the multiplier is flat in
the kernel size.** The register is `4*surface - ow*oh_task` on the depthwise path and
`2*surface - ow*oh_task` on the direct one, where `surface` is `0x401C`. The 4 was
indistinguishable from `kh+1` while every depthwise capture was k=3; at k=1, 5 and 7
it does not move.

**`0x401C` rounds up to four elements on the DEPTHWISE path and not on the direct
one.** The rounding is invisible at every plane whose `ow*oh_full` is already a
multiple of four, which is every capture on both paths and every hardware sweep this
had been fitted against. Rectangular and odd depthwise planes show it: the vendor
programs 272, 324 and 364 for `ow*oh_full` of 270, 323 and 361. The direct path's
answer is a hardware result rather than a transcription, because no direct capture
has a plane that separates it, a VALID-padded oc=128 conv at `ow*oh_full = 105` is
bit-exact with 105 programmed and wrong with 108. Carrying the depthwise rounding
onto the direct path costs that shape and nothing else in the 67-shape gate, which is
worth knowing: one shape in the whole envelope is sensitive to it.
[source-confirmed + HW sweep, H96 MAX M9]

**`0x118C` is `iw-1` in both halves.** The low half read as the full plane height for
as long as every capture was a square plane. 142 non-ARGB programs carry `iw-1`
twice, with no exception. This one is not depthwise-specific; it was wrong on the
direct path too, and invisible there for the same reason.

`0x1024`'s low half is still not a depthwise channel count: it is the same raw `1` at
every count from 8 to 256, which is what an on-hardware sweep of the field concluded
independently from the other direction.

## fp16: the datapath, and the contraction width that bounds it

An fp16 direct convolution computes bit-faithfully on this part and delivers every
output channel, and the library drives it: `gen_conv2d_fp16_rk3576()` for one task,
`rocket_rk3576_plan_ic()` for an arbitrary input-channel count. **One bound remains,
and it is a single one rather than the two this once read as: the contraction is
sixteen input channels wide.** The split is the way around it.

Two apparent gaps are one fact seen from two sides: an output writer that spends
four bytes on a two-byte element and reaches only `oc/2` channels, and a feature
surface index that caps a task at 8 input channels. The DPU's output element stride is `16/ic` words. At `ic = 8` that is two
words per element, which is where the "four bytes on a two-byte element" reading came
from; at `ic = 16` it is exactly one and the surface is the plain native cube.

Everything below is [source-confirmed, RKNN-Toolkit2 rk3576 float build; HW sweep,
H96 MAX M9, measured 2026-07-26].

### The float program is a transcription, and here is how to get one

Every `.rknn` in the vendor-capture set is int8, and a zero precision field is
invisible; it says nothing about where the field is, how wide it is, or whether this
part carries one there at all. That is why the float fields were inferred from the
RK3588's packing for as long as they were, and why one of them was wrong.

**RKNN-Toolkit2 will emit a float program for this part.** `do_quantization=False`
on a float ONNX leaves the weights float and the vendor compiler picks the float
datapath; `tests/data/rk3576-vendor-capture/float/mkfloat.py` rebuilds the captures
and carries the dependency pins, which are particular (onnx 1.16 removed
`onnx.mapping` and setuptools 81 removed `pkg_resources`; the toolkit imports both,
and neither failure names itself).

Two things make the resulting diff small enough to read:

- **Only seven registers move with `ic`** across the whole 139-word program,
  `0x101C`, `0x1020`, `0x1028`, `0x1030`, `0x103C`, `0x1044`, `0x107C`, and six of
  them our emitter already matched. All seven scale linearly.
- **Fourteen registers differ from what we emitted**, and they are the same fourteen
  at every geometry. Three of them are load-bearing; the other eleven can each be
  dropped with the datapath still exact.

### Three registers carry the float mode

Each is constant across every geometry the vendor emits (`ic` 8-64, `oc` 32/64, `k`
1/3, planes 16 and 32), and each is load-bearing **on its own**: with any one of them
at its integer value the contraction reads every feature surface twice and skips the
odd ones. That is why no single-register sweep ever found this, the fix is a
register *set*, and every sweep to date moved one register.

| register | integer value | float value | what is load-bearing |
|---|---|---|---|
| CNA `0x100C` | `0x00000000` | `0x00200120` | all of `bit21 \| proc<<7 \| in<<4` |
| DPU `0x4038` | `0x00120080` | `0x00100092` | the low half, `0x0092` |
| DPU `0x4050` | `0x80011111` | `0x00021111` | **bit 31 clear** |

- **CNA `0x100C` bit 21 is a float enable** the integer programs never set. It is not
  implied by the two precision fields: with both precisions right and bit 21 clear the
  contraction still doubles, exactly as it does with bit 21 set and a precision wrong.
- **DPU `0x4038`'s high half is free**: `0x0012` computes as well as `0x0010`, but
  must be non-zero, or the DPU writes nothing at all.
- **DPU `0x4050`'s low nibbles do not enter here.** `0x00011111`, `0x00021111` and
  `0x00020000` all compute; `0x80021111` returns `+inf`, the accumulator having summed
  the doubled reads. The nibble is the write-extent field described below; bit 31 is
  the float bit.

### PROC_PRECISION is the operand width

**CNA `0x100C`'s `proc_precision` field carries fp16 (`2`), not fp32.** An earlier
reading had it naming the width of the multiply-accumulate datapath, so that a float
convolution accumulating in fp32 wanted `5` there. That was inferred from the RK3588's
field semantics and it is wrong for this part: programming `5` with every other float
field right reads each feature surface twice.

The reading survived as long as it did because the probes that endorsed it were
uniform in the channel axis, where a doubled read is indistinguishable from a scale;
see Traps.

CORE `0x3018` still does not follow the CNA word, and the division of labour is the
one already established:

- **CNA `0x100C`'s `in_precision` pins the operand width class, not its type.** Values
  `1`, `2` and `3`, the three 2-byte codes, all compute; `0` (a 1-byte element)
  writes an entirely zero surface. So an fp16 and a bf16 program are indistinguishable
  in this field.
- **CORE `0x3018` is what pins the operand type.** Only `2` computes; `3` (bf16)
  returns a wrong surface against fp16 data.

### The output is fp16 only with the float narrowing enabled

The epilogue is a float path and its result is an fp32 word. What reaches DDR is decided
by two things:

- **DPU `0x40B0` bit 16 is `fp32tofp16_en`**, at the RK3588's own bit position. Without
  it the fp32 word is written as-is.
- **DPU `0x4010` bits [31:29] select a width, not a conversion.** `5` writes the whole
  32-bit word; `2` and `3` write its low and high halves.

Composed: narrowing on plus width `2` writes true fp16 (`0x5400` for 64.0); narrowing on
plus width `3` writes the top half of the fp32 word, which is **bfloat16**; narrowing off
plus width `5` writes fp32, and narrowing off plus width `2` writes the fp32 word's low
mantissa bits, **zero for every value a small integer test pattern can produce**.

That last combination is what an fp16 program emitted from the RK3588 encoding produced,
and it is why the MAC was read as dead for so long: a full, correctly sized, entirely
zero surface, from a datapath that was running.

### The float weight cube groups both channel axes by 16

The int8 weight cube groups both axes by 32 (`weight_conv_int8`). **The float cube
groups both by 16.** The reorder is otherwise Mesa's `oc1, ic1, kh, kw, oc2, ic2`;
`rocket_rk3576_weight_conv_fp16()` is the index. The feature cube is C2 = 8 fp16 lanes
in the same 16-byte atom, pinned by a bijective decode rather than by a score (C2 = 4
and 16 both fail).

Each group needs a shape chosen so that it is observable at all, and the output group
is the harder of the two:

- **ic group = 16** is exact where 4, 8 and 32 are not.
- **oc group = 16** is exact at `ic = 16` with `k` 3 and 5. It **cannot be pinned at
  `k = 1`**: the kernel index sits between the group and the `(oc2, ic2)` pair, so at
  `k = 1` the index collapses to `(oc/G)*G + oc%G = oc` and every group size is
  byte-identical. A sweep run only at `k = 1` reports all four candidates as correct.

That is the concrete form of the general trap below: an earlier reading of 8 here came
from a shape where the field was algebraically dead.

A kernel that ramps over `ic` does **not** settle the ic group, and neither does any
score: that probe sums over the channel axis, so it is invariant under every permutation
of it. Use realistic weights at a shape where the group is live.

### The output map is the plain native cube

At the 16-channel contraction the writer has no defect: 8 channels to a 16-byte atom,
one atom per pixel, channel groups as contiguous planes, **every programmed channel
present exactly once**. For output channel `c` at pixel `p`, in 16-bit words:

```
word = (ow*oh*8)*(c/8) + 8*p + (c%8)
```

That is measured, not fitted: `rk3576_fp16_sweep map` drives a probe whose every output
lane carries a unique name and decodes the surface, and the decode is a **bijection onto
the whole surface**, every word names the (channel, pixel) this map predicts, with none
left over and none undecodable. `rocket_rk3576_fp16_out_index()` is that map and
`rocket_rk3576_fp16_accumulate()` de-scatters one surface through it into a row-major
fp32 accumulator, which is what an `ic` split needs. `regcmd_rk3576_gate` checks the
bijection host-side, so an off-by-one in it does not wait for a board.

**The map is what changes when `ic` does**, which is the single clearest reading of the
bound. At the same `oc = 16`, 4x4 plane:

| `ic` | slot of channel `c` within its atom | channels reaching DDR |
|---|---|---|
| 8 | `2*(c%4)`, odd words zero | `oc/2` |
| 16 | `c%8` | **all of them** |
| 32 | `(c/2)%8`, odd channels lost | `oc/2` |

### The contraction is sixteen input channels wide

`gen_conv2d_fp16_rk3576()` **refuses** any other `ic` rather than warning about it. A
wrong count writes a full, correctly sized, wrong surface with nothing to fault on, so
a caller must not be able to reach it silently; `gen_conv2d_rk3576_prec()` is the
unchecked bring-up entry beside it, and it is what the sweep's own modes drive.

Read the multiplicity straight off the part with probe 10: one input channel carries
1.0, every weight lane is 1.0, and the output *is* that channel's read count.
`ROCKET_FS_TAP` moves the tapped channel, and the pair `TAP=0` / `TAP=16` at `ic = 32`
separates the two failure modes in one run, before the float fields, they read 2.0 and
0.0; after, both read 1.0.

With the fields right the read count is 1.0 at `ic` 16, 32, 64 and 128 and at every
tap, so the *reads* are unbounded. What bounds a task is the writer's element stride
above.

### The fp16 envelope that computes

A single fp16 task is bit-faithful against the CPU model, `worst relative error 0`,
at **`ic = 16`**, for `k` 1, 3 and 5, planes 8x8 through 56x56, stride 1 and 2, `oc` 16
through 64, and every programmed output channel lands.

### An arbitrary `ic` via a 16-channel split

Splitting `ic` into slices of 16 and summing the partial surfaces on the host
reproduces the model exactly at any input channel count, **bit-exact, `worst relative
error 0`, at `ic` 16, 32, 64 and 128** crossed with `k` 1, 3 and 5.

`rocket_rk3576_plan_ic()` is the split, beside `rocket_rk3576_plan_rows()` and on the
same pattern: it lays the slices out and the caller emits one conv per entry. The slice
is fixed at 16 rather than as wide as the CBUF allows, a wider slice would not be a
cheaper task but a wrong one. It plans the `ic` axis only; a plane whose 16-channel
slice still overflows the CBUF is refused with a pointer at the row planner rather than
run.

The slices cost nothing to address: at C2 = 8 the feature cube's channel groups are
contiguous planes of `iw*ih_full` 16-byte atoms, so slice `k` is the same BO at a base
offset, and an atom stays 16 bytes when C2 halves, so the stride is element-size
independent. Only the weight cube is rebuilt per slice; each slice is its own
convolution and so has its own group count, which is why a slice's cube is not a
sub-cube of the whole conv's (`rocket_rk3576_fp16_pack_slice_weights()`). Size that
cube by the groups.

The bias belongs to one slice only. Every slice adds the whole `A` term, so a
coefficient buffer carrying a bias handed to all of them lands it `ic/16` times.

**On-chip accumulation is unblocked and small.** The DPU eltwise stage does this job on
the RK3588 (`ROCKET_KACC`) and would remove `ic/16` readbacks here; nothing stands in
its way, because at this contraction width the partial it would read back is a plain,
dense, complete fp16 cube rather than a surface carrying the writer's defect.

What it is worth is **7-13% of the wall**, not the largest win on this path
[HW sweep, H96 MAX M9, measured 2026-07-28, `oc` 32, 28x28 k3, `ic` 16-128,
`tests/rk3576_fp16_split_cost.c` with `ROCKET_RK3576_FP16_PROF=1`]:

| phase | share of the entry's wall |
|---|---|
| submit | 77-87% |
| readback (what on-chip accumulation removes) | 7-13% |
| weight repack | 4-7% |
| output stamp | 2-4% |

The slice count is set by the sixteen-channel contraction whatever sums the partials,
so the submits stay and only the readback goes, and the EW operand DMA the DPU would
issue instead is not free, so the net is below that band. Read the table as a bound.

**A slice is the poll floor plus `dpu_grace_us`, in full.** The same sweep records zero
poisoning retries at every shape, so a slice's ~1.4 ms is not the hazard being cleared:
a wide-output task raises no DPU completion, so `0012` retires it on `PC_DONE` plus the
whole blind settle. Halving the grace moves the wall by very nearly the grace, once per
slice [HW sweep, `ic` 128, 8 slices, five runs each, ranges disjoint]:

| `dpu_grace_us` | 8-slice wall | per slice |
|---|---|---|
| 500 (default) | 15.39-16.16 ms, median 15.89 | ~1.99 ms |
| 250 | 12.95-13.97 ms, median 13.29 | ~1.66 ms |

That is 300 us of a 250 us cut landing per slice, and the convolution stays bit-exact at
every value tried down to 150.

**That lever is not available through the default**, because the same number is the
deadline on the DPU-completion wait for every job that does raise one, and a narrow-output
matmul fails 10 runs of 10 at 200 us (see the grace table). An fp16 conv pays the grace
in full only because its tasks raise no DPU completion at all, so what would collect the
1.18x is a way to say which class a task is in, which userspace knows when it emits the
program and the driver cannot see at `PC_DONE` time. Lowering the shared default trades
the fp16 path's wall time against the correctness of every other path.

### What an fp16 conv costs

Priced against the `ic/16` submits the split spends, which is what the convolution is
here today. [HW, H96 MAX M9, measured 2026-07-26, `oc` 64, third of three runs]

| shape | submits | NPU | host pack + de-scatter | total |
|---|---|---|---|---|
| 32x32 ic16 k3 | 1 | 1.51 ms | 2.52 ms | 4.04 ms |
| 32x32 ic64 k3 | 4 | 5.57 ms | 3.36 ms | 8.92 ms |
| 56x56 ic64 k1 | 4 | 6.00 ms | 24.20 ms | 30.20 ms |
| 56x56 ic64 k3 | 4 | 6.21 ms | 8.62 ms | 14.83 ms |
| 56x56 ic128 k3 | 8 | 11.93 ms | 15.93 ms | 27.86 ms |

**The NPU wall is flat in the work and linear in the submits: about 1.4-1.5 ms per
submit, whatever the plane and the kernel.** `k = 3` at 56x56 does nine times the
multiply-accumulates of `k = 1` and costs the same 6 ms. There is no compute term
visible at these shapes; it is entirely the per-submit dispatch floor, and that floor
is the same 1.1-1.3 ms the int8 path pays once for its whole `ic`
(`rk3576_conv_gate envelope`). Doubling the contraction width halved the submit count
and the NPU wall with it: the same shapes cost 8 and 16 submits at the 8-channel split.

Only the NPU column is a measurement. The host column is noisy on this thermally
limited board, the same shape varies by a factor of two run to run, and the 56x56
`k = 1` row is the clearest case of it. The NPU column is stable across pacing gaps of
0, 20 and 50 ms to within 6%.

Both planners take the precision the conv will be emitted at
(`rocket_rk3576_cbuf_f_prec`, `rocket_rk3576_max_task_rows_prec`; the shorter names are
those at int8). The feature plane's granule cost is per byte, so a 2-byte element
doubles it, and an allowance planned at the int8 rate would be sized for half an fp16
plane, which computes wrong with a full surface written. The weight side of the trade
stays at the int8 model on purpose, as an upper bound on the float slice: over-estimating
refuses early instead of corrupting.

### The write-extent field, and two DPU words that look like the answer

`0x4050`'s low nibbles encode the written extent: 1 / 2 / 4 bytes per element as
`{0, 1, 3}`, the direct int8 word carrying `1` and the depthwise word `3`, the same
`size_e = bytes-1` relation the RK3588 pairs with its surface-advance multiplier, which
the RK3576 already tracks (`ow*(2*oh_full - oh)` direct, `ow*(4*oh_full - oh)`
depthwise). It moves the written *extent* only, never the atom, and it is not what the
float mode needs from this register, bit 31 is.

Two DPU words do move the output and are worth naming, because they look like the
answer and are not:

| register | what it actually does |
|---|---|
| `0x40D0` low 16 | a per-byte write enable for the 16-byte atom, clearing bit *n* leaves byte *n* untouched |
| `0x40D0` bits [19:16], `0x40CC` bits [3:0] | shift the written data by whole bytes within the atom |

### The knobs that settle the rest

`A` (bias) is consumed as **fp32**: packed as its fp32 bit pattern and read back at width
`5` it returns `3f800000 40000000 40400000 40800000` for `bias[c] = c+1`, exactly
1.0/2.0/3.0/4.0. `C` (multiplier) is consumed as **fp16**, so the int8 default `C = 1` is
the denormal 6e-8 and underflows the whole surface to empty. fp16 `1.0` is `0x3C00`.

**Asymmetric weights are a packing question, not an encoding one.** `B` in the
coefficient group is a per-output-channel int16 sitting beside the bias `A` and the
multiplier `C`, and driving a weight zero point through it needs no register change,
`rocket_rk3576_pack_coeff_asym()` takes one. Its sign convention is settled: the DPU
adds the term, so a weight zero point is programmed negated. See "The B term is ADDED"
above for the measurement.

### The harness

`tests/rk3576_fp16_sweep.c` scores every candidate against a CPU model of the same
convolution, because the failure mode here is a full, correctly sized, *wrong* surface,
which no amount of looking at the output will reveal. It reports an untouched buffer, an
all-zero surface and a wrong surface as three separate results: the first is a dead DMA,
the second is an underflowed or gated epilogue, and only the third is an arithmetic or
layout error.

Its probes are what turn "wrong" into "wrong here". Each is built so its answer does not
depend on the thing it is not measuring:

| probe | what it drives | what a correct part returns |
|---|---|---|
| 1 | uniform feature and kernel, each value its own knob (`_PIN`, `_PW`) | `ic*k*k * in * w`, and a result that tracks a side's *high byte* names the side being read as int8 |
| 2 | one non-zero element on each side | one non-zero output lane, at a position |
| 3 | the pixel ramp on **every** input channel | `y*iw+x+1` in every lane, whatever the lane map |
| 5 | uniform feature, kernel *c* weighted *c+1* | `c+1` in every lane, whatever the lane map |
| 6 | input channel *c* carries *c+1*, kernel taps ic=0 | `1`, anything else names the channels the part paired with lane 0 |
| 7 | uniform feature, kernel ramps over ic | `1+...+ic`, anything else counts the weight lanes it walked |
| 8 | unique naming: every lane carries `ow*oh*c + p + 1` | itself, the output map, decoded rather than guessed |
| 9 | unique feature, output channel *c* taps input channel *c* | the input cube handed back through the datapath |
| 10 | one input channel at 1.0, every weight lane 1.0 | `1`, the value **is** that channel's read count |

Probes 3 and 5 are deliberately *uniform in the axis they do not name*, so they read the
output map without first knowing the lane map. Probes 6 and 7 do the reverse.

Probes 8, 9 and 10 exist because that uniformity is also their blind spot. Each of 1, 3,
5, 6 and 7 passes against a datapath that reduces the wrong input channels, so a probe
set built only from them reports an exact conv that is not one. 8 and 9 name lanes
instead of scoring them, a *bijection* is the pass, not a match count, and 10 measures
the reduction directly. Run `map` before believing any fp16 layout claim.

| knob | what it does |
|---|---|
| `map` (argv) | drive probe 8 and decode the surface; reports where each lane landed and how many copies |
| `ROCKET_FS_OUT_LAYOUT` | `0` the native output cube, `1` the library's `rocket_rk3576_fp16_out_index()` |
| `ROCKET_FS_TAP` | move the single live weight lane off ic=0; `-1` taps every input channel (probe 3) |

| knob | what it does |
|---|---|
| `int8` (argv) | the control, the known-good path through the same harness |
| `icsplit` (argv) | the library's `ic` split, scored end to end and timed |
| `f32out` (argv) | the fp16 datapath read back as the full 32-bit epilogue word |
| `bitsweep` (argv) | one register walked bit by bit or over `ROCKET_FS_SWEEP_VALS` |
| `ROCKET_FS_BIAS` | `1` integer bias probe, `2` the same value as fp32 bits |
| `ROCKET_FS_CMUL` | the `C` multiplier; `15360` is fp16 1.0 |
| `ROCKET_FS_SET` | whole-register overrides for the `manual` candidate |
| `ROCKET_FS_C2` / `_WOC` / `_WIC` / `_OUT_C2` | the cube geometry, each axis on its own |
| `ROCKET_FS_DUMP` / `_DUMP_OFF` / `_SCAN` | raw words at an offset; the written extent and every non-zero lane |

**Run the `int8` control before believing any fp16 result.** It exercises the same
scatter, submit and de-scatter over the path this part is known to compute, so it fails
only when the harness is wrong. It must report `2048/2048`.

**Read the raw surface, not only the score.** Scoring reads the buffer *through* the
layout under test, so it cannot show a layout that disagrees with it.

### Traps

- **A register sweep must parse hex.** `atoi("0x100c")` is 0, and a sweep that silently
  patches register 0 reports that every value changed nothing, which is exactly what a
  correct sweep of an inert register looks like. This cost a full pass over `0x100C`
  before the reading that mattered was found.
- **`feature_data()` and `weight_conv_*()` take 1-based channel and kernel indices.** A
  0-based call drives the group remainder negative and misplaces the whole cube. The
  result is a wrong surface rather than an absent one, so it reads as an encoding error.
- **An integer test pattern cannot see a low-16-bit truncation.** Every value a small
  integer convolution produces has zero low mantissa bits in fp32, so reading an fp32
  result through a 16-bit output width returns a *uniformly zero* surface, the same
  signature as a gated epilogue or a dead MAC. Read at width `5`, or drive values whose
  low mantissa bits are non-zero.
- **A bias probe fixes only the channel-to-lane mapping.** Every pixel carries the same
  value per channel, so nothing in it constrains the pixel mapping.
- **A probe that is uniform in one axis cannot validate the reduction.** A uniform
  feature reports how many lanes were read, never which; a weight held constant over
  `ic` makes the contraction invariant under any permutation of the channel axis; and a
  sum over `ic` is permutation-invariant however it is driven. A datapath that reduces
  the wrong input channels passes all of them. Measure the multiplicity (probe 10) and
  decode a unique-naming probe (`map`) before calling any float datapath exact.
- **Size a weight cube by its group, not by `ic`.** A partial input-channel group still
  occupies a whole one, so `ic = 8` at an ic group of 16 needs a cube for 16. An
  allocation taken from `ic` alone under-allocates, and at `k = 1` the overrun stays
  inside the BO's page and computes correctly anyway; it only surfaces at a kernel large
  enough to run past the page, where it reads as "this part cannot do `k = 3`". Size the
  buffer from `ceil(ic/group)*group`.
- **A cube-geometry sweep needs a shape where the field is observable.** The float
  weight cube's output-channel group is algebraically invisible at `k = 1`, the kernel
  index sits between the group and the `(oc2, ic2)` pair, so the index collapses to
  `(oc/G)*G + oc%G = oc` and every candidate emits the same bytes. A sweep at `k = 1`
  reports the field settled while testing nothing, and that is where the wrong value 8
  came from. Before believing a geometry knob, check that two candidates actually
  produce different buffers.
- **Score only what the map covers.** Reading a surface through the layout under test
  cannot reveal a layout that disagrees with it, and a partial match invites fitting a
  shift to three atoms. Decode a unique-naming probe instead: the answer is a bijection
  or it is not.
- **Pace probe loops.** Back-to-back submits return untouched surfaces at a rate that
  depends on the gap; at 200 ms and above every submit writes. Treat one no-write as a
  measurement to repeat, and re-submit rather than record it.
- **The bias belongs to one slice of an `ic` split.** Every slice runs the whole epilogue
  and adds the whole `A` term, so a coefficient buffer carrying a bias handed to all of
  them lands it `ic/16` times. The error is a per-channel constant on an otherwise exact
  surface, which is the shape a wrong `B` sign convention also has.
- **A slice's weight cube is not a sub-cube of the whole conv's.** Each slice is its own
  convolution, so its group count follows the slice rather than the total. Slicing the
  full cube by byte range hands the part a correctly sized cube with the kernel positions
  shuffled.
- **One register at a time cannot find a mode.** The float datapath needs three
  registers together, and each of the three looks inert while either of the others is at
  its integer value, a leave-one-out over the working set is what separates them, and a
  single-register sweep of any one of them reports it dead. Two of the three are also
  unreachable by a single-bit sweep from the integer word.
- **A vendor capture is cheaper than a sweep, and one can be manufactured.** The float
  fields went un-transcribed for as long as they did because every capture on hand was
  int8, not because no float capture could exist. RKNN-Toolkit2 emits one for this part
  in about an hour of dependency pinning, and it settled in one diff what bit-level
  sweeps had not. The same lever settled four depthwise register formulas and the
  depthwise weight cube. Before sweeping for anything, ask what ONNX would make the
  vendor compiler emit it.
- **A found capture set is a set of confounds.** The vendor compiles real models, so
  the axes that co-vary in real models co-vary in every capture: on this part every
  C=32 depthwise program was stride 1 and every C=64 one stride 2, every depthwise
  kernel was 3, and every plane was square. Three register formulas were fitted
  through those confounds and all three were wrong, `0x4050`'s channel field, the
  surface rounding in `0x401C`/`0x40B8`, and `0x118C`, which is `iw-1` in both halves
  and had read as the plane height for as long as `iw == ih`. When a capture set
  cannot vary an axis, manufacture one that does before fitting anything to it.
- **A degenerate helper program is not a convolution.** A model on a plane whose size
  is not a power of two makes the compiler append a 1x1x1 program beside the conv,
  with a DPU_RDMA config no conv carries. It is not something a conv emitter produces
  and it reads as a spurious gate failure.
- **Timing an `ic` split at a probe pacing gap measures the gap.** `ROCKET_FS_GAP_MS` is
  there so a no-write is rare, not so a number is meaningful; time at `0` and read the
  retry count the harness reports alongside.

## `0x5024` is the DPU shift word, and zero is not a safe value for it

`BS_BASE_ADDR1` (`0x5024`) is a live operand base, not a spare word. The DPU
reads one 32-bit word through it per task and right-shifts the accumulator by two
independent 6-bit fields:

```
bits[5:0]   right shift applied where the result is NON-NEGATIVE
bits[13:8]  right shift applied where the result is NEGATIVE
```

Both are exact powers of two over the whole range: a word of `0x0202` takes an
accumulator of +64 to +16 and one of -64 to -16, and a field of 8 or more flushes
that sign to zero. The two sides are genuinely independent, the field selection
follows the sign of the **accumulator**, not the sign of the weight, so swapping
the feature sign swaps which output channels move. [HW sweep, H96]

Every capture stores 0 here, for the same reason every capture stores 0 for the
feature, weight, output and bias bases: it is an address the vendor runtime
patches at load time, and a `.rknn`-derived program is the *stored* register set.
Leaving it at zero is not benign, because **IOVA 0 is a real buffer on a mainline
`rocket` stack**, the per-fd address space bump-starts at 0, so whichever BO the
caller allocates first is addressable as 0. The DPU then takes both shift amounts
out of that buffer's first four bytes, which is normally the head of the feature
cube. Point it at 64 zeroed bytes; `rocket_rk3576_coeff_bytes()` reserves them
past the A/B/C groups and the emitter addresses them.

This is worth stating as a trap rather than as a fix, because of how it presents.
The shift depends on two bytes of *feature data*, so it looks like a data-dependent
arithmetic fault rather than an addressing one: a conv is bit-exact for one feature
tensor and attenuated by an arbitrary power of two for another, reproducibly, with
no fault and no dmesg. It also hides from the obvious probes. A uniform feature
fill puts the same byte in both fields, so it is either right or uniformly wrong;
an identity weight set reads one input channel per output channel and cannot show
a contribution that should have been zero; and any feature whose first two bytes
happen to have zero low-6-bits, a fill of 64, or of 0, computes perfectly. The
symptom moved with the *feature cube's channel 1* only because that channel's
first byte is byte 1 of the buffer.

The pattern to recognize: a result that is exact for some feature tensors and a
clean power-of-two too small for others, with the factor unchanged by the
programmed requant scale. That last part is what places it upstream of the
OUT_CVT and downstream of the MAC.

## The cold-start wall

On an unpatched `rocket`, only the **first** job of each NPU power session computes.
Submitting the same job N times and counting how many wrote: [HW, H96]

| Gap between jobs | Result |
|---|---|
| 0 ms (back to back, one fd) | job 1 writes; jobs 2-8 leave the output **untouched** |
| 100 ms | all 8 write, byte-identical |
| 300 ms | all 8 write, byte-identical |
| separate processes | every one writes |

The NPU runtime-suspends 50 ms after its last job, so the unit is the **NPU power
session**, not the fd, not the process, and not chained-versus-independent
submits: independent per-op submits inside one power session fail exactly as a
chained task would, and any gap past the autosuspend delay re-arms.

This matches an external report on a different userspace (mesa/rocket + Teflon), which
read the behavior as the part arming its weight loader once per power session and found
independent per-op submits could not bypass it. Reproducing it on our own encoder and our
own submit path is what placed it in the driver.

### The cause is the `PC_TASK_CON` field width

`PC_TASK_CON` packs `TASK_NUMBER` in the low bits with three control bits directly above
it. The field is 12 bits wide on the RK3588 and **16 on the RK3576**, so all three move
up by four:

| | `TASK_NUMBER` | `TASK_PP_EN` | `TASK_COUNT_CLEAR` | `TASK_LAST_LAYER_CLEAR` |
|---|---|---|---|---|
| RK3588 | `BIT[11:0]` | `BIT(12)` | `BIT(13)` | `BIT(14)` |
| RK3576 | `BIT[15:0]` | `BIT(16)` | `BIT(17)` | `BIT(18)` |

The RK3588 register description marks the top control reserved; on the RK3576 it is
`task_last_layer_clear`, and it belongs on every submit alongside the count clear.
Chaoyi Chen of Rockchip gave the layout on the `linux-rockchip` list
([message](https://lore.kernel.org/all/4f300b78-d96d-4d98-8819-dc292b0c9b97@rock-chips.com/)),
which is what makes the naming authoritative rather than inferred, the word itself was
already fixed here by shifting the whole triple, and `0x70001` is what both derivations
write. **[source-confirmed]**

`rocket` builds the word
from the RK3588 field accessors unconditionally, giving `0x7001`, so on this part it
asks the PC for a **task count of 28673** with all three control bits landing above the
register's defined fields. The PC starts that 28673-task program, runs the one task it
was handed and is left mid-stream; nothing else in the session starts. A power cycle
resets the PC, which is the whole of why a gap re-arms it.

Building the word from a per-SoC width instead clears it, and leaves the RK3588 word
bit-identical at 12 bits. Back-to-back with no gap: **1 of 8 submits wrote before, 8 of 8
after**, and 32 of 32 on a longer soak, byte-identical to each other and bit-exact
against the CPU model. A row-windowed 112x112 k3 conv is bit-exact over the whole surface
with no inter-task gap. The patch is `rk3576/npu/0008` in the `rk3576-npu` profile.
[HW sweep, H96 MAX M9, measured 2026-07-25]

Sweeping the control field at each width isolates the cause to `TASK_NUMBER` alone:

| word written | `TASK_NUMBER` seen (16-bit field) | wrote |
|---|---|---|
| `0x70001`, `0x60001`, `0x50001`, `0x30001`, `0x20001`, `0x10001`, `0x00001` | 1 | 8 of 8 each |
| `0x07001` (the RK3588 word) | 28673 | 1 of 8 |
| `0x06001`, `0x04001`, `0x02001`, `0x01001` | 24577, 16385, 8193, 4097 | 0-1 of 8 |

So `TASK_COUNT_CLEAR` is not the mechanism, with the field correctly placed the part
works with every control bit clear, and with it misplaced no control-bit value tried
helps. Of the other two per-SoC PC parameters in the vendor config, `pc_dma_ctrl`'s
IRQ-locked `PC_DATA_ADDR` write is not needed and neither is the RK3576-only
`state_init` hook; `pc_task_status_offset` is real and load-bearing, because the
register at `0x48` is the live task counter a chained stream's completion has to be
read from (see "`PC_DONE` is per TASK" below).

Nothing a regcmd can write clears it, which is why the fix had to be a driver patch.
Sweeping the CNA/CORE/DPU/RDMA `S_POINTER` value over the whole ping-pong field
(`POINTER`, both `PP_EN`s, `PP_MODE`, both `PP_CLEAR`s), writing `PC_TASK_CON` from
inside the stream, and replaying the vendor's state-init sequence as regcmd writes all
leave it at one job per power session. Two of those are informative in themselves:
forcing `POINTER=1` makes **every** job write nothing, so the bit is live and selects a
producer register group, and the fact that job 2 still fails with `POINTER` held at 0
means the consumer is not advancing behind us, which rules the ping-pong groups out.
[HW sweep, H96 MAX M9]

**Read back, the `POINTER` field is not the driver's to write while `PP_MODE` is set.**
A second party reports, on a Radxa ROCK 4D: bit 0 written as 0 reads back as 1, on every
job, for the rest of the session; flipping it per submit, in the direct register writes
*and* in all four `S_POINTER` entries of the regcmd, moves neither the readback nor the
result; selecting a bank the way the vendor's `state_init` does, with the `PP` bits clear,
stops the units arming at all; and pulsing `POINTER_PP_CLEAR`, with or without
`EXECUTER_PP_CLEAR`, moves nothing. A 20 KB read snapshot of `pc`, `cna`, `core`, `dpu`
and `rdma` taken at the same point in a job that computed and one that did not differs in
exactly one word, `OPERATION_ENABLE`. [linux-rockchip RFC v4 4/6 cover and code comments,
Jiaxing Hu, 2026-08-03; their measurement, not reproduced here]

So the field is hardware-owned under `PP_MODE`, which is consistent with the sweep above
finding no `S_POINTER` value that changes a second submit's fate. It does **disagree** with
one half of it: forcing `POINTER=1` from inside the regcmd kills every job here, where they
report flipping it changes nothing. Two candidate reasons, neither settled, the writes go
in from different places (a regcmd the PC is fetching against a slave-mode register write),
and the **posted** series carries no equivalent of `0008`, still emitting mainline's
`PC_TASK_CON_TASK_NUMBER(1)` against the RK3588's 12-bit accessor. Whether the tree the
experiments actually ran on carried it is not knowable from the posting, and their public
repo has had its own 16-bit fix since at least 2026-07-26, so do not state which.

That caveat is what to carry forward, because it is a **method** trap rather than a
register fact: on a driver still programming the RK3588's 12-bit `PC_TASK_CON` word, the
second submit of every power session writes nothing for a reason that has nothing to do
with the register under test. Every arm of a ping-pong sweep then reads "no change", and
the ledger is uninformative by construction however exhaustive it is. Establish that job 2
can compute *at all* before attributing anything to a configuration difference between job
1 and job 2, and stamp the output BO with a sentinel first, since a repeated configuration
reading a stale surface is byte-exact for the wrong reason, which is exactly the shape "one
configuration is byte exact forever, a different one computes nothing" produces. A fresh
BO's zeros cannot separate "never ran" from "ran and wrote zeros"; see the write guard in
[rk3576.md](rk3576.md), and fill through a `PREP_BO`/`FINI_BO` bracket rather than a bare
`memset`, or the dirty lines race the DMA.

For reference, the three per-SoC parameters the vendor `rknpu` config carries and
mainline `rocket` hardcodes at the RK3588 value; only the first is load-bearing:

| | RK3588 | RK3576 |
|---|---|---|
| `pc_task_number_bits` | 12 | **16** |
| `pc_task_status_offset` | `0x3c` | `0x48` |
| `pc_dma_ctrl` | 0 | **1** (writes `PC_DATA_ADDR` under the IRQ lock) |

The RK3576 also has a per-SoC `state_init` hook that no other Rockchip NPU in that
driver has, run at every power-on with the PC in slave mode: it seeds CNA `0x1024`
with bit 31 in **both** ping-pong register groups and then clears the pointer
(`S_POINTER = 0x1e`). `rocket` has no analog, and the slave-mode entry cannot be
replayed from a regcmd the PC is fetching. Placing it is easy if it is ever wanted:
`rocket_job_hw_submit()` already opens with `PC BASE_ADDRESS = 0x1`, which is slave
mode, and the vendor's CNA writes drop straight into that window; at power-on, matching
the vendor, the place is the runtime-resume callback.

**Neither it nor `pc_dma_ctrl` is needed, and the two are ruled out jointly.** They are
not needed for correct back-to-back submits, the task-number width accounts for the
wall on its own. They are also not what makes two jobs in flight on the two cores
compute wrong answers, which is the one workload that would reach the second pointer
group: seeding both groups at every power-on for each core and holding a device-global
lock across the whole register-programming sequence leaves the concurrent cell at 100%
of calls wrong against a 97.9% control, with the seeding counted so the negative is
about the sequence rather than about a knob that never fired. Applying them together
is the point, a one-at-a-time pass cannot see a condition of two, and since the whole
delta does nothing, no subset can. [HW sweep, H96 MAX M9,
`tests/rk3576_core_pair.c pair`; see [rk3576.md](rk3576.md) for what the mechanism is
narrowed to instead]

### Working against an unpatched driver

Everything below applies to a `rocket` without `0008`, and is worth keeping because a
probe on a stock module hits all of it.

**`autosuspend_delay_ms` is the only userspace lever, and it is unreliable.** Writing 0
to `/sys/.../27700000.npu/power/autosuspend_delay_ms` makes the driver drop the power
domains straight after each job, so a back-to-back loop climbs from 1 job in N to
about half, 8 of 16, and 6 of 6 on a shorter run. It is a race, not a fix: when the
next job arrives before the autosuspend work runs, no power cycle happens and that job
is walled. Anything that must be correct has to check per task that its own output
landed and resubmit, which is what `rk3576_first_light` does when it runs a row split.

A gap of ~1 s is enough to make results stable and repeatable; at ~0.4 s a probe
still occasionally comes back with the output BO wholly untouched. Read a single
"the DPU did not write at all" as the wall, not as a register result, re-run
before believing it.

**The wall has two signatures, and the second one is the dangerous one.** Either
the output BO comes back wholly untouched, or the DPU writes a full surface
carrying only the bias, so the MAC contribution is missing and the surface reads
all zero. Both look exactly like a wrong register program, and the second is also
what a `C=0` coefficient buffer produces, so three unrelated causes converge on
one appearance. `rk3576_first_light` retries past the autosuspend on both
signatures, treating an all-zero surface as the wall only when the CPU model says
it should have been non-zero; a sweep that reports its results without that check
is measuring the wall roughly half the time at multi-second gaps.

## The per-submit floor is the driver's completion poll

Every cost table on this part is a submit-count table, and the per-submit cost is not
the silicon. `PC_DONE` is read-only in `INTERRUPT_MASK` on this revision, so it cannot
be routed to the GIC, and `patches/rk3576/npu/0006` polls it on an hrtimer instead of
taking an interrupt as the RK3588 does. The period is the floor.

`patches/rk3576/npu/0009` makes the period a module parameter; `0012` sets it to
**50 us** and changes what the poll retires on. Measured through `rk3576_submit_floor`,
an 8x64x32 int8 matmul submitted 300 times, the shape chosen so dispatch dominates
compute, the floor is **439 us/submit against 1654 at the stock 1 ms**, 3.77x.
[HW sweep, H96 MAX M9, measured 2026-07-28]

**Do not read a gate's total wall time as the floor.** The conv gate's 102 shapes take
about 5.9 s at every period from 1000 us down to 150 us, unchanged, because that gate is
dominated by host packing and its CPU model rather than by submits. It looks like a
flat, negative result and it is measuring something else. Time a single small shape;
that is what `rk3576_submit_floor` is for.

### A drm task carries one PC program, whatever the stream holds

The obvious way to amortize the floor is to batch: a row-windowed convolution is one
submit per window, and a tile's row tasks are independent by construction; each writes
its own rows of the same surface from its own window of the same feature cube, with the
weight and coefficient buffers unchanged across them. Concatenating their programs into
one regcmd stream is arithmetically sound.

**It does not run.** The driver programs `PC_TASK_CON` with `TASK_NUMBER = 1` per drm
task descriptor, so the program counter executes the first task in the stream and stops.
Measured with a per-task write check that names the earliest task missing rather than
only that one was: at 3 tasks and at 6, **task 0 lands and task 1 is the first missing,
on every one of eight attempts with the power domain confirmed cycled between them**,
deterministic, and so not the poisoning. Across the conv gate that is 27 shapes, every
one a multi-window or multi-group plan, and none of the single-task ones.
[HW sweep, H96 MAX M9, 2026-07-28]

**Collecting it takes a kernel patch and a joint layout contract, and both now exist.**
Submitting n drm task descriptors is not enough on its own: the driver already runs those
back to back (`rocket_job_hw_submit()` re-arms on `next_task_idx` from the completion
path) but each through its own completion poll, and the poll is the floor, so that form
saves the ioctl and the fence round trip and not the thing worth saving. What collects the
floor is the RK3588 series' `086` arrangement, ported as `patches/rk3576/npu/0015` and
`0016`:

- userspace lays the n programs out contiguously at one even-word stride and rewrites each
  trailer's inert `OP_NONE` filler into a `PC_BASE_ADDRESS` write pointing at the next
  (`src/rocket_chain.c`, shared with the RK3588; this part's emitter ends every program
  with the same `[OP_NONE, PC_REGISTER_AMOUNTS, OP_40, OP_ENABLE]` trailer that rewrite
  claims);
- it submits them as n drm tasks with `DRM_ROCKET_JOB_BATCHED`;
- the driver programs task 0 only, sets `TASK_NUMBER = n`, and advances `next_task_idx`
  straight to the end, so the PC streams all n from one kick and the job retires on the
  single completion that `TASK_NUMBER` gates.

The `TASK_NUMBER` bound is per-SoC, so `0016` takes it from `rocket_soc_data` (16 bits
here, 12 on the RK3588) rather than from the RK3588 field mask.

**The correctness bar is met and the saving is one completion poll per row task removed.**
The convolution library gate is 156 passed / 7 refused as required / 0 failed with
chaining on, identical to the one-submit-per-task path, where a concatenated stream in one
drm task failed 27 shapes. A 32x32 k3 plane forced into 8 row windows goes **2.8 ms ->
1.0 ms** and a 224x224 k3 s1 convolution **21.3 ms -> 19.1 ms**.
[HW sweep, H96 MAX M9, 2026-07-29]

**Do not look for this in a gate's total wall time.** Over the whole convolution library
gate it is 6643 ms -> 6463 ms, about 3%, and forcing `ROCKET_RK3576_MAX_ROWS` down to 8 or
4 does not enlarge it, because most shapes in that gate plan into one or two row tasks
and the gate's wall is host packing and the per-call power-cycle guard, not submits. The
lever is worth `(n-1) * 439 us` per call and nothing else; it pays on the deep planes and
is invisible on the shallow ones.

It is default-on in `librocketnpu`'s int8 convolution path (`ROCKET_RK3576_BATCH_TASKS=0` turns it off), and the same layout carries a run of cube-linked layers as one kick; see the cross-layer kick in [rk3576.md](rk3576.md). Note that both trailer fields describe the next segment: the driver programs `PC_BASE_ADDRESS` and `PC_REGISTER_AMOUNTS` from task 0 alone, so a chain of programs that differ in length must write its successor's count and not its own, which a uniform row-task chain cannot distinguish.
`rocket_batched_submit_supported()` refuses to self-chain against a kernel that does not
honour the flag (it reads
`/sys/module/rocket/parameters/rocket_batch_submit`, else the advertised DRM interface
version, which must be >= 1.1). That check is load-bearing rather than tidy, since a
chained layout run down the per-task path runs task 0 and stalls to the job timeout.

**The fp16 `ic` split is not reachable this way as the path stands.** Its slices repack
the same weight BO and read back and accumulate into the host's buffer between submits, so
there is host work between them and they are not one stream. Chaining them means per-slice
weight buffers and either per-slice output regions or on-chip accumulation, the same
restructuring the on-chip-accumulation item wants, not a free consequence of this patch.

### `PC_DONE` is the wrong signal; the DPU's own completion is the right one

`PC_DONE` means the **program counter** finished issuing, not that the DPU's write DMA
has drained. The driver both tears the IOMMU mapping down and signals the fence as soon
as it sees it, and a long period hides both behind its own detection lag. Two
independent failures come out from under it:

- **The per-job IOMMU detach races the write drain.** Detaching on completion faults
  writes that are still in flight, which leaves a *write* page fault active,
  `RK_MMU_STATUS` `0x2b` = `PAGING_ENABLED|PAGE_FAULT_ACTIVE|IDLE|PAGE_FAULT_IS_WRITE`.
  That fault blocks the next `enable_stall`, which the IOMMU reports as `Enable stall
  request timed out`; the companion status `0x1d` carries `STALL_ACTIVE`, so the stall
  did arrive, just after the poll gave up. The NPU then raises
  `DMA_READ_ERROR|DMA_WRITE_ERROR` and the job comes back short.
  `patches/rk3576/npu/0011` removes this outright by keeping the domain attached across
  jobs.
- **The fence is signalled before the writes are visible.** Closed by `0012`.

**`RAW_STATUS` carries a signal that retires when the writes have landed.** `DPU_0`
(`0x100`) and `DPU_1` (`0x200`) are what the RK3588 takes as its completion interrupt,
and on this part they set **tens to hundreds of microseconds after `PC_DONE`**. Logged
at the moment the poll retires:

```
raw@retire=0x20000001 -> 0x30000155 after 82us
```

`PC_DONE_1` with the CNA feature bit and no DPU bit, then `DPU_0` 82 us later. Requiring
the DPU bit is the fix, and it is what lets the period drop:

| poll period | 500 us | 250 us | 125 us | 50 us | 25 us |
|---|---|---|---|---|---|
| us/submit, retire on `PC_DONE` | 1065 | 731* | 535* | 433* | n/a |
| us/submit, retire on the DPU bit | 1073 | 739 | 558 | **439** | 398 |
| int8 matmul gate, retire on `PC_DONE` | 43/43 | 42/43* | n/a | 36/43* | n/a |
| int8 matmul gate, retire on the DPU bit | 43/43 | 43/43 | 43/43 | **43/43** | 43/43 |

`*` incorrect. [HW sweep, H96 MAX M9, 3 runs per correctness point, measured 2026-07-28]

**The failure is a partial surface, and the missing bytes are a contiguous tail.** The
reproducer is the largest output surface in the gate, `n-tiled-deep`, 32x2048x3072,
which at `poll_interval_us=250` on `PC_DONE` fails 3 runs of 3 at ~95-98k of 98304
elements exact. Snapshotting the output BO, settling, re-syncing and comparing shows
**the last 2048 bytes of the 65536-byte surface arriving after the fence retired**
(`changed=2048 first=63488 last=65535`): the data is not lost, it drains late. The drain
reaches **343 us** for that surface and is ~82-98 us for the conv gate's smaller ones,
so it scales with bytes in flight.

**Not every task raises a DPU completion, so the wait has to be bounded.** A task whose
DPU output element is **wider than one byte**: the int32 writer, and fp16 output,
completes CNA and CORE and never sets a DPU bit at all: `RAW_STATUS = 0x30000055`
(`PC_DONE_0|PC_DONE_1|CORE_0|CNA_CSC_0|CNA_WEIGHT_0|CNA_FEATURE_0`), held for
milliseconds with nothing further arriving. `0012` gives them `rocket.dpu_grace_us` and
retires them on `PC_DONE` as before. That grace is load-bearing, not a formality; those
tasks still need a blind settle, and its threshold is the same ~250 us the settle
experiment found:

| `dpu_grace_us` | 100 | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|
| int8 matmul gate | 40/43 | 43/43 | 43/43 | 43/43 | 43/43 |
| through the wide int32 writer | 41/43 | 43/43 | 43/43 | 43/43 | 43/43 |

**The grace is a deadline, not a settle, and that is what fixes its default.** It bounds
the wait for a signal most jobs do raise, so it is also what covers the drain of every
ordinary narrow-output job, lowering it retires those on `PC_DONE` exactly as the
unpatched driver did. The two roles pull opposite ways and neither is optional:

| `dpu_grace_us` | 200 | 250 | 300 | 350 | 400 | 500 | 800 |
|---|---|---|---|---|---|---|---|
| `n-tiled-deep`, 10 runs | **10 fail** | 0 | 0 | 0 | 0 | 0 | 0 |
| `fc-rgb-224-k7`, 5 runs | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The cliff is sharp and deterministic, every run fails at 200, none at 250, so **250 is
1.25x the measured failure point, not a margin above it**, and the gates cannot separate
250 from 500 because nothing in them drains further. What argues for the margin is the
drain itself, measured per gate with the poll retiring on the DPU bit and the grace set
past every wait. The instrument is a module parameter on the board's out-of-tree debug
tree, `dbg_dpu_wait_max_us` and `dbg_dpu_waits`, the maximum and count over jobs that
retired on the DPU bit, with the jobs that raise none timing out into `dbg_grace_hits`
instead. It is diagnostic and deliberately not in the patch series, so a reimage takes it
with everything else:

| gate | worst drain | jobs reaching the DPU bit | jobs raising none |
|---|---|---|---|
| `rk3576_conv_gate all` | 99 us | 15 | 0 |
| `rk3576_conv_lib_gate` | **745 us** | 30 | 31 |
| `rk3576_matmul_gate` | 290 us | 9 | 58 |
| through the wide int32 writer | 343 us | 10 | 54 |
| `rk3576_argb_ic1` | 46 us | 2 | 0 |

The 745 us is one shape, `fc-rgb-224-k7`, the int8 stem at 224x224 k7 s2, and it
reproduces to within 3 us over five runs. So **the shipped 500 us default is already
below the worst drain on the part** and the stem still passes at 200: a drain only has to
be covered when it is the last submit before the host reads, and a row-split job's
earlier tasks are covered by the submits that follow them. That exposed drain is what the
cliff measures, and it is not separately observable, so the raw drain is the only
available bound on it and it reaches 745 us at a 392 KiB surface.

**The default stays 500 us.** It is 2.5x the measured failure point, the drain scales
with bytes in flight and the library can emit surfaces larger than any gate shape, and
the fp16 win from lowering it (1.18x, below) is a cost paid only by the tasks that raise
no DPU completion, a class userspace can name and the driver cannot see at `PC_DONE`
time. Collecting that win means separating the two classes at submit time, not lowering
the one number that also bounds the other class's correctness.
[HW sweep, H96 MAX M9, measured 2026-07-28]

**That separation is `patches/rk3576/npu/0017`, and it collects the win without touching
the deadline.** `drm_rocket_job.flags` gains `DRM_ROCKET_JOB_NO_DPU_DONE`; a job carrying
it waits `dpu_blind_us` (default 250) past `PC_DONE` instead of `dpu_grace_us`, and a job
without it is bit-for-bit unaffected. The hint is **advisory by construction**: the poll
still retires the moment a DPU completion arrives, whatever the flag says, so a wrong
hint costs time and never correctness. `librocketnpu` sets it on exactly the classes the
poison probe already names wide: the fp16 direct conv, the fp16 first conv and the int32
matmul writer. On an fp16 convolution at `ic=128 oc=32` 28x28 k3, eight wide-output
slices, the shape where the settle is the dominant cost:

| `dpu_blind_us` | 250 | 500 (hint neutralised) | 3000 |
|---|---|---|---|
| eight-slice wall | **13.67 ms** | 15.20 ms | 36.96 ms |

10% off that path with every narrow-output job keeping its full 500 us deadline.
[HW sweep, H96 MAX M9, measured 2026-07-29]

**A negative A/B on a knob like this is a claim about the wiring first.** The first run of
that measurement showed no effect at any `dpu_blind_us`, which reads as "the lever is not
real", but a single silently-failed edit had left the poll comparing against
`dpu_grace_us`, so the flag reached the driver and changed nothing. What separated the two
readings was **driving the other knob**: `dpu_grace_us` 500 -> 3000 moved the same wall
15.31 -> 36.14 ms, which proves those tasks were still paying the grace and so that the
hint was not being applied. Before believing that a per-class knob does nothing, show that
the class is reaching it, an out-of-range value that fails to move the wall is the cheap
version of that check.

This is the same set of programs that carries the poisoning, and the two are **separate
hazards**: with the DPU bit as the retire condition the poison probe's `scope` map is
unchanged (fp16 first conv, fp16 direct conv and both int32 writers poison all seven
kinds; the int8 first conv, int8 direct conv and int8 depthwise poison none).

**And a poisoned submit raises its DPU completion normally while writing nothing**: so
the missing DPU bit is not a poisoning detector, and the poisoning is not "the DPU never
ran". `pair A int8d` at `ROCKET_PP_REPS` of 4 and 8 gives grace hits from the A side
only, scaling exactly with the rep count (`fp16fc` 4 and 8, one submit a rep; `i32` 8 and
16, two), while the poisoned `int8d` submits on the B side contribute zero at every rep
count. The sentinel scan remains the only way to see a poisoned surface.
[HW sweep, H96 MAX M9, measured 2026-07-28]

An earlier reading of the DPU bits as a closed negative, "`DPU_0` is already set on the
same poll tick as `PC_DONE`", was an artifact of the period it was measured at. At
500 us of detection lag the drain has always finished by the time the poll looks, so the
two signals cannot be told apart; at 50 us they separate cleanly.

### `PC_DONE` is per task, and `PC_TASK_STATUS` is the live task counter

`PC_DONE_0`/`PC_DONE_1` are **two alternating per-task pulses, not one whole-kick
completion**. On a `TASK_NUMBER = n` kick the two bits swap as the program counter
retires each program, so the first of them appears a few tens of microseconds into a
stream that may run for milliseconds and says nothing about the stream being over. They
are also not latched across tasks: mid-stream the whole `RAW_STATUS` word cycles through
`0x10000000`, `0x20000000`, `0x30000000` and `0x00000000`, and the per-task block
completion bits (`0x155` for one ping-pong group, `0x2aa` for the other) appear only
briefly and are gone again by the next task. Only the last task's block bits survive,
because no task follows to clear them. [HW sweep, H96 MAX M9]

**`PC_TASK_STATUS` is where the kick is.** The register is at **`0x48`** on this part
(`0x3c` on the RK3588, the vendor `rknpu` config carries the delta as
`pc_task_status_offset` and reads the register as its "task counter"), and it holds two
16-bit counters, **both modulo the programmed `TASK_NUMBER`**:

| bits | field |
|---|---|
| 15:0 | tasks **started** |
| 31:16 | tasks **completed** |

While task `k` (0-based) of `n` is in flight it reads `(k << 16) | (k + 1)`; while the
last one runs, `started` has wrapped and it reads `(n - 1) << 16`; once every task has
retired both wrap and it reads **0**, which is also what it reads before the first task
starts. Confirmed on kicks of 2, 3, 5, 8, 9, 35 and 90 tasks. The RK3588's reported
`0x0000f000` at IRQ time fits the same model: `& 0xfff == 0` is the completed count
having wrapped, i.e. the whole job done. [HW sweep, H96 MAX M9]

Mid-stream at least one of the two halves is non-zero, so **`PC_DONE` set together with a
zero `PC_TASK_STATUS` means the whole kick is over, and nothing else does**, the signal
a driver needs to place a completion wait on a chained stream. The 35-program cross-layer
kick of a MobileNetV1-224, traced at a 10 us poll:

```
t=   12 us   PC_DONE_0,  ts=0x00000001   task 1 of 35 in flight
t=   31 us   PC_DONE_1,  ts=0x00000001
 ...         the two bits alternating, once per task
t= 1911 us   PC_DONE_1,  ts=0x00220000   task 35 of 35 in flight
t= 1921 us   0x300002aa, ts=0x00000000   the DPU's own completion; the kick is over
```

**`PC_OPERATION_ENABLE` is a self-clearing go bit, not a busy flag.** It reads back **0
at every poll of a kick that is still running** [HW sweep], and the vendor driver writes
`1` and then `0` to it back to back (`rknpu_job.c`) [source-confirmed]. So a completion
test of the form `OPERATION_ENABLE == 0 || PC_DONE` is *vacuously true* on this part from
the first poll onward, which is what any wait built on it is really anchored to.

### A half-started job pins the NPU runtime-active, and then only the int32 path breaks

Fixed by `patches/rk3576/npu/0010`. Recorded because the state it produces is the most
misleading one the part reaches, and because the bug is in mainline `rocket`, a kernel
without 0010 still has it, on the RK3588 as well.

`rocket_job_run()` takes a runtime-PM reference and then attaches the IOMMU group. Both
of its early returns leave the job half-started, and **each of two independent leaks is
enough on its own** to pin the device runtime-`active` with `power/
runtime_suspended_time` frozen, even with nothing running and `power/control` at `auto`:

- **The usage counter ratchets.** `pm_runtime_get_sync()` keeps its reference when the
  resume fails, and the IOMMU path has already resumed successfully. `core->
  in_flight_job` is still `NULL`, so the completion path and the timeout path, both of
  which only put when a job is in flight, never balance it.
- **The scheduler never gets its credit back.** The fence handed to the scheduler is one
  nothing will ever signal, so the job stays pending, and
  `rocket_device_runtime_suspend()` returns `-EBUSY` while a credit is outstanding.

**The usage-counter half outlives the driver.** The reference is on the platform device,
which the of core owns, so `rmmod`/`insmod` does not clear it: a freshly probed device
comes back `active` and stays there until a reboot.

What that breaks is exactly one thing. The int32 output writer's poisoning of the next
submit is cleared by the power domain cycling, so on a device that cannot suspend it
never clears: every i32 shape writes nothing, at `0/256`, `0/32`, `0/224`. Everything
else still works perfectly, the conv gate is 102/102, the library gate 111/1, and the
plain int8 matmul shapes all pass. The signature therefore reads as "the int32 path has
a bug" rather than "the machine is in a bad state", and it will absorb a debugging
session.

It is also self-sustaining without 0010. One failed attach times out at 500 ms, and the
timeout path calls `rocket_core_reset()`, which takes the IOMMU down, so the next
attach fails for real. One injected failure produced seven, and leaked seven references.
With 0010 the injected failure is the only one: the job is retired with its error, the
caller's next submit attaches and computes normally, and the device suspends.

**Check `power/runtime_status` before believing any int32 result.** A board in this state
makes every other experiment lie: a poll-period sweep run through it reports the matmul
failing at every period, which is what it looks like when the poll has nothing to do
with it. [HW, H96 MAX M9, measured 2026-07-27]

### A per-job core reset is not the way to clear the poisoning

`rocket_core_reset()`, a `reset_control` bulk assert, `udelay(10)`, deassert, is what
the driver already does on a job timeout, and calling it at job start looked like a way
to drop the power-cycle idle an int32-output job forces on the next submit. It is not.
Asserted before every job it takes the **IOMMU** down with it:

```
rk_iommu 27702000.iommu: Error during raw reset. MMU_DTE_ADDR is not functioning
rocket 27700000.npu: NPU job timed out: RAW_STATUS=0x30000000 MASK=0x800fffff OP_EN=0x00000000
```

`RAW_STATUS` carries both `PC_DONE` bits, so the job ran; nothing reached DDR because the
page-table pointer was gone. The matmul gate falls from 43 shapes passing to 2, the conv
gate to 0, and the damage outlives the setting, the part stays wedged until the module
is reloaded. The driver's own timeout path resets inside `drm_sched_stop()`, an IOMMU
detach and a re-attach, and it is that surrounding re-init the bare call is missing. The
vendor `rknpu` driver never resets per job either. [HW, H96 MAX M9, measured 2026-07-27]

## The IOMMU wedge

The NPU can be left wedged by a previous session, and it does not recover on its
own. The signature in `dmesg` is

```
rk_iommu 27702000.iommu: Enable stall request timed out, status: 0x0000..
rocket 27700000.npu: NPU job timed out: RAW_STATUS=0x30000000 MASK=0x800fffff OP_EN=0x00000000
rk_iommu 27702000.iommu: Error during raw reset. MMU_DTE_ADDR is not functioning
```

after which every submit times out and every output BO comes back untouched,
which reads exactly like "the encoder writes nothing" and will send a register
hunt down a false trail. The rail, the clock and the driver binding all still
look healthy while this is true, so they do not discriminate. Check `dmesg` for
the stall timeout before trusting any negative result on this part. [HW, H96]

**Reloading the module clears the compute path**, which is worth trying before rebooting:
`rmmod rocket; sleep 8; insmod rocket.ko` re-runs the IOMMU attach and the conv gates
pass again immediately. The settle matters, a reload with a two-second gap left the part
still failing where an eight-second one fixed it. Confirmed against two independently
induced wedges, a 100 us poll period and a per-job core reset.

**It does not clear everything on a kernel without `patches/rk3576/npu/0010`.** A reload
leaves a leaked runtime-PM reference in place, because that lives on the platform device,
and the part then computes convolutions perfectly while every int32 matmul writes nothing;
see the section above. Check `power/runtime_status` after a reload; if it does not
return to `suspended`, reboot. [HW, H96 MAX M9, measured 2026-07-27]

**With 0010 and 0011 the wedge is much harder to reach in the first place.** The
`Enable stall request timed out` / `MMU_DTE_ADDR is not functioning` sequence above is
the per-job IOMMU detach racing the DPU's write drain; 0011 stops detaching per job, and
0010 stops one failed attach from cascading through the timeout path's core reset into
the next one.

## A rejected submit oopses the kernel, and a refusing generator is what emits one

`rocket_ioctl_submit_job()` allocates its `rocket_job` with `kzalloc` and assigns
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
`task_struct_size` below `sizeof(struct drm_rocket_task)`, an unreadable
`drm_rocket_job.tasks` pointer, `drm_rocket_task.regcmd_count == 0`, and either BO
handle not resolving. `/dev/accel/accel0` is group `render`, so the trigger is
unprivileged. **This is mainline code**: identical in a pristine v7.1 tree, present
since the submit ioctl was added, not something the RK3576 series introduces.
`patches/rk3576/npu/0013` fixes it by taking the domain reference at construction, so
a job owns one from the moment it exists; `patches/rocket/085` carries the
complementary NULL guard in the put. The two touch different files and compose.
Measured with `tests/uapi_submit_errpath_rocket`, which fires one malformed submit per
site from a forked child: **0 of 5 sites returned to userspace before, 5 of 5 after**
[HW, H96 MAX M9, 7.1.3].

**The client that emitted one here was our own uAPI selftest, and the mechanism is a
refusing generator.** `gen_matmul_fp16()` emits the RK3588 geometry encoding and
refuses on this part by construction, which leaves `matmul_params_t.task_count` at 0;
the deadline canary then built its task descriptors from those unset parameters and
submitted `regcmd_count == 0`. So the general lesson is not about the canary: **a
generator that refuses leaves its output parameters untouched, and a caller that
submits anyway hands the kernel a malformed job.** Check the generate before building
a submit from it.

A single Oops does not brick the part, an ordinary client and the full 46-shape
matmul gate both still run afterwards. Several do: after three, the file-close path
faults again inside the IOMMU layer (`iommu_map_nosync`, `iommu_unmap`,
`iommu_domain_free` from `rocket_postclose`), the process becomes unkillable, and the
kernel prints `Fixing recursive fault but reboot is needed!`. Do not keep running
gates on a kernel that has taken one, the results after it are not about the NPU.

**`rocket_job.c:639` in `rocket_job_irq_handler()` is a different thing and not a
driver defect.** It is mainline's `WARN_ON(raw_status &
PC_INTERRUPT_RAW_STATUS_DMA_WRITE_ERROR)`, the driver reporting that the NPU raised a
DMA write error. Seen on core 1's IRQ during a two-core run, it is the kernel-log
fingerprint of the concurrent-job corruption, not a separate bug to chase.

**A client-supplied bad address does not reach that `WARN_ON` on this part.**
`drm_rocket_task.regcmd` is a raw IOVA that `rocket_job_hw_submit()` writes into the PC
block's `BASE_ADDRESS` with no check that the job's BOs mapped it, so a client can aim
either the instruction fetch or, through a real program with a rewritten output address,
the DPU's write DMA at unmapped memory. Both were tried
(`tests/uapi_regcmd_fault_rocket`, `read` and `write`) and both surface as
`rk_iommu: Enable stall request timed out` and nothing else: **0 WARNINGs, taint
unchanged, and the matmul gate 46/46 immediately afterwards**. The RK3576's IOMMU
absorbs a bad address as a stall rather than the NPU's PC raising a DMA-error bit. So
the `WARN_ON`-as-unprivileged-DoS concern is real upstream in principle, a `WARN_ON`
on a hardware error condition taints and panics under `panic_on_warn`, but it is **not
demonstrated to be client-triggerable here**. [HW, H96 MAX M9, 2026-07-28]

**A job whose program faults still retires cleanly.** The submit returns 0, `PREP_BO`
returns 0, the output BO is untouched, and userspace is told nothing. That is the same
"the encoder writes nothing" false trail the IOMMU-wedge section warns about, arriving
from the driver rather than from the encoder, so an untouched output BO is not evidence
about an encoding.

**On the RK3588 the same probe finds a client-triggerable warning somewhere else**, so
the question "can a client taint the kernel through the NPU" has a different answer from
the question about that particular `WARN_ON`. There, both modes end in the job timeout,
and the reset behind it walks into the IOMMU core:

```
rk_iommu fdaca000.iommu: Enable stall request timed out, status: 0x2b
WARNING: drivers/iommu/iommu.c:157 at __iommu_group_set_core_domain
 iommu_detach_group / rocket_reset.part.0 [rocket] / rocket_job_timedout [rocket]
```

The NPU is still faulting when `rocket_reset()` detaches the group, so the IOMMU's stall
and disable-paging requests time out and the detach WARNs. `patches/rocket/083`
(keep-the-domain-attached-across-jobs) is applied on that board and does not cover it,
keeping the domain attached across *jobs* says nothing about a reset that detaches
explicitly. So on both parts the reachable path is the reset, not the IRQ handler; the
parts differ only in whether the detach WARNs. [HW, Turing RK1, 2026-07-28]

## A BO in an in-flight job must survive the file that made it

The per-context IOVA allocator (`drm_mm` + `mm_lock`) lives in `struct
rocket_file_priv` and is torn down and freed in `rocket_postclose()`. A BO's IOVA node
is removed only in `rocket_gem_bo_free()`, and a job's BO references are dropped
asynchronously by the drm_sched free worker, which can run after the owning file has
closed. A client that submits and closes without waiting therefore leaves
`rocket_gem_bo_free()` taking `bo->driver_priv->mm_lock` and calling
`drm_mm_remove_node()` on memory `rocket_postclose()` already freed. `/dev/accel/accel0`
is group `render`, so it is unprivileged, and it is mainline code the RK3576 enablement
does not touch.

It reproduces on demand, the reason it looked rare is that an ordinary caller waits for
its job and so never sets the race up. `tests/uapi_bo_lifetime_rocket` opens a fresh fd
per iteration, submits, and closes immediately: **16 of 16 iterations warn**, and left
running longer the same client reaches the dereference:

```
Internal error: Oops: 0000000096000004 [#1]  SMP
Workqueue: 27700000.npu drm_sched_free_job_work
pc : add_hole+0x34/0x15c
lr : drm_mm_remove_node+0x1e8/0x380
 rocket_gem_bo_free [rocket] / drm_gem_object_free / rocket_job_cleanup [rocket] /
 rocket_job_free [rocket] / drm_sched_free_job_work
```

`patches/rk3576/npu/0014` anchors the allocator to `struct rocket_iommu_domain`, which
is refcounted and outlives the file, every mapped BO and the attached core: **0 of 16
after**, same kernel, only that patch changed. It is the RK3588 series' `084` ported,
and applies to this series with offsets only. [HW, H96 MAX M9, 2026-07-28]

**Match the symbol, not drm_mm's message text.** A 7.1 kernel prints `warning:
drivers/gpu/drm/drm_mm.c:965 at drm_mm_takedown+0x28/0x38`; the older "allocator still
has nodes" wording does not appear at all, so a detector keyed on the message reads
clean while the defect fires on every iteration.

**The verdict is the kernel log, not the exit status.** This defect fails no syscall, a
run can pass every ioctl and still be the failing side of the A/B.

## The matmul: the same 1x1 convolution, and the envelope it plans inside

A matmul is a 1x1 convolution over these blocks, exactly as it is on the RK3588: the A
rows are the conv's spatial pixels, K is the input-channel axis, N the output-channel
axis. Nothing about the register program is new, the conv gate already ran a real
matmul shape without calling it one (`w-e4608-k1` is `ic=4608`, `oc=128` on a 4x2 VALID
plane, which is M=8, K=4608, N=128 in one submit), so the whole of the work is the
userspace layer: the operand scatter, the de-scatter, a tiling planner and a dispatch
point. `rocket_matmul_int8_rk3576()` is the entry, `tests/rk3576_matmul_gate.c` the gate,
and `tests/rk3576_matmul_probe.c` the instrument the envelope below was read with.

The mapping, for anything driving the emitter directly:

| operand | layout |
|---|---|
| A[M,K] | the feature cube, channel `k`, pixel `m` at `(m/iw, m%iw)`, C2 = 16 int8 lanes |
| B[N,K] | `weight_conv_int8(N, K, 1, 1, n+1, k+1, 1, 1)` |
| C[M,N] | the int8 output cube, `(n/16)*surf_elems*16 + 16*(y*iw + x) + n%16` |

The **plane is free**. Any `(iw, ih)` with `iw*ih == M` carries the M rows and all of
them compute; the choice is a tiling one. What it costs is granules, a feature row is
`ceil(iw*K/64)` of them against the task's 4096, so the pixels one task holds come out
at `262144/K` however the plane is cut, provided `iw*K` is a whole number of granules.
When it is not, every row rounds up and the waste is real, so take the widest divisor of
M that divides evenly.

### The M axis carries no constraint

M = 1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 16, 17, 31, 32 and 64 are each bit-exact, at every
factorization into a plane, `1xM`, `Mx1` and everything between. This is the RK3588's
answer inverted: there rows are the conv's spatial height, a height under 4 mis-computes,
`M%4` is the real bound and software pads `M==1` to 4. Here `M=1` is simply correct.
Whether it is *worth* running at M=1 is a separate question the dispatch floor answers.
[HW sweep, H96 MAX M9]

### int8 is the matmul precision, and the margin is large

The two precisions contract at wildly different rates. One int8 task takes
`ic*kh*kw <= 4608`, so K = 4608 lands in one submit; one fp16 task contracts exactly
sixteen input channels, so the same K costs 288 submits. Measured on the same shape at
both precisions, with the pacing outside the timing:

| shape | int8 submits | int8 ms | fp16 submits | fp16 ms | ratio |
|---|---|---|---|---|---|
| 16x512x64 | 1 | 1.4 | 32 | 42.9 | 30x |
| 32x1024x128 | 1 | 1.3 | 64 | 85.2 | 64x |
| 128x2048x128 | 1 | 1.2 | 128 | 171.3 | 142x |
| 56x4608x128 | 1 | 1.3 | 288 | 386.1 | 299x |

On the RK3588 fp16 wins and int8 buys only RAM. Here it is the other way round, and the
reason is not the arithmetic; it is that the fp16 contraction width is 16 while the int8
one is 4608. [HW sweep, H96 MAX M9, measured 2026-07-27]

### Throughput is MACs per submit, and N is the only free axis

A submit costs about 0.44 ms whatever it carries. `M*K` is capped by the feature budget
and K by the resident weight slice, so the only axis left to spend is N. At the
feature-budget cap it buys throughput almost linearly:

| N | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 2560 |
|---|---|---|---|---|---|---|---|---|
| GOP/s | 12 | 24 | 47 | 93 | 190 | 367 | 719 | 1000 |

(at `M*K = 262144`; the figure is flat in how that product is split, so 1024x256,
256x1024 and 64x4096 all land within a few percent of each other.)
[HW sweep, H96 MAX M9, measured 2026-07-27]

### Two output-channel bounds, and neither is a convolution's problem

Past either one the trailing output channels do not reach DDR: the surface is full and
correctly sized, its leading channels are bit-exact, and nothing faults. Both are now
refusals in `r76_plan_cbuf` rather than hopes.

- **oc <= 2944.** 2944 computes 3/3 at every K tried; 3072 is intermittent (1/3 at
  K=256, 2/3 at K=1024) and 4096 never computes.
- **the whole weight cube <= 6 MiB.** `ic=4096, oc=1536` is 6 MiB and computes 3/3;
  `ic=4608, oc=1536` is 6.75 MiB and computes 0/3; `ic=4096, oc=2048` is 8 MiB and loses
  about a quarter of its channels.

The second is stated as the whole cube rather than per kernel tap deliberately: it was
measured at `k=1`, which is the matmul's own kernel, so phrasing it this way makes it
refuse earlier at `k>1` than anything measured there. That is the safe direction, and no
convolution shape in the gate comes near either bound, the largest is 516 KiB at oc=128.

**Repeat before believing a boundary here.** The first pass at this sweep ran each point
once and produced a clean-looking N ceiling that moved when the same points were re-run:
`1024x256x3072` failed in one pass and passed in the next. Three repeats plus a
classification of the failure; all-sentinel is a dead submit, a clean prefix is a
capacity bound, wrong values in a fully written surface is arithmetic, is what separated
the real bound from the noise. [HW sweep, H96 MAX M9]

### The int32 output: the first eight channels of every thirty-two

The DPU will emit its raw 32-bit accumulator on an integer program.
`gen_conv2d_int8_rk3576_i32out()` puts `precision_int32` in DPU `0x4010`'s output-width
field and pins OUT_CVT to exact unity, and the words that come back are genuine
accumulators laid out on the RK3588's int32 cube map, in 32-bit words:

```
word = (c/4) * ow*oh_full * 4 + 4 * (y*ow + x) + (c%4)
```

Decoded rather than assumed. The probe gives every accumulator a distinct value, so each
value names exactly one word, and every position read that way fits the map with no
misses, at `oc` 4, 8, 16, 32, 64 and 96, at `K` 16 through 256, and at `M` 4 and 8.

**The writer keeps the INT8 surface's byte budget whatever the element width is.** It
writes `ceil(oc/16)` contiguous blocks of `ow*oh_full` 16-byte atoms, the int8 surface,
exactly, and at four bytes an element each block carries four channels where an int8
block carries sixteen. Block `j` holds channels `32*(j/2) + 4*(j%2) .. +3`, so what
reaches DDR is

> **the first eight output channels of every thirty-two**, and nothing else is touched.

That is one fact, not two. At `oc <= 8` every channel is delivered and the surface is
complete (`oc = 4` and `oc = 8` decode 32/32 and 64/64 with no waste); past it the yield
is `oc/4`. The earlier reading of "the extent is `ceil(oc/16)` groups" came from a sweep
run only at `oc = 32`, where the block count and the delivered-channel rule cannot be
told apart, `oc = 16` and `oc = 8` separate them, and both write two blocks.

**The way round it is the weight cube, not a register.** Program a multiple of the
output channels and put real channel `n` in a slot the writer delivers, leaving the rest
zero. Every real channel then lands in a delivered slot, and the surface reads back as a
plain cube. `rocket_matmul_int8_rk3576_i32()` is that, and it is bit-exact against a CPU
int32 model. On this narrow writer the multiple is four and the slot is
`32*(n/8) + n%8`, which collapses the map above to the plain int32 cube.

**What it costs is the output-channel axis, spent over.** The bytes are not wasted, the
budget comes out exactly the int32 surface, but the resident weight slice, the N tile
and the 2944-channel bound are all functions of the programmed `oc`, so the multiple is
paid in MACs per submit. That follows from the measured budget rather than from the
encoding, so no register will buy it back, but `PROC_PRECISION` halves the multiple, and
that is the writer the library uses.

### The wide writer: the first eight of every sixteen, at half the cost

**`PROC_PRECISION` doubles the byte budget.** DPU `0x4010`'s low field (`[2:0]`) is the
DPU's own operand width, and driving it from int8 to int32 makes the writer emit two
16-byte atoms per (16-channel block, pixel) instead of one. The delivered set becomes
**the first eight output channels of every sixteen**, so a real channel costs two
programmed ones rather than four. Nothing else moves: the operands stay int8 everywhere,
the arithmetic is bit-identical, and the CNA and CORE programs are byte-unchanged.
`gen_conv2d_int8_rk3576_i32out_wide()` emits it.

**Two atoms is the ceiling**, swept across all eight values of the field at
`M=8, K=32, N=32`:

| `0x4010[2:0]` | extent | what came back |
|---|---|---|
| 0 int8, 6 int4 | 64 words | one atom per (block, pixel), the 4x rule |
| 1 int16, 4 int32 | 128 words | two atoms, the 2x rule, values correct |
| 2 fp16, 3 bf16, 5 | 128 words | two atoms, and the operands reinterpreted: no accumulator survives |
| 7 tf32 | n/a | every attempt was a dead submit |

**The map.** Work in 32-channel super-groups, each `4*A` atoms long, where `A` is the
surface's pixel count `ow*oh_full`. Inside one super-group the writer emits a single
linear stream indexed by `s = 2*p + j`, where `p` is the pixel and `j` the 16-channel
block, and cuts that stream into runs of `A` atoms. The run also carries the lane group
`L = (c%16)/4`, which is the slower axis:

```
atom = 4*A*(c/32) + A*(2*(s/A) + L) + s%A       s = 2*p + j
word = 4*atom + c%4                             delivered iff c%16 < 8
```

Because it is a stream, nothing rounds: an odd `A` simply cuts it at an odd place, and
`A = 1` works as readily as `A = 16`. Decoded, not fitted, the operands were drawn so
every accumulator in the tile is distinct, so each written word names exactly one
(channel, pixel) or the map is not a map. Read off at `oc` 8/16/24/32/40/48/64/96/128/192
and at pixel counts 4/5/6/7/8/12/13/16/21/33.

**Two bounds, and both fail silently.**

- **`oc` must be a multiple of 32**, because a partial super-group is not a truncated
  one. At `oc = 16 (mod 32)` the trailing group holds one 16-channel block instead of two
  and packs at one atom per pixel with no stream. At an `oc` that is not a multiple of 16
  at all, 24 is the case read off, the delivered channel set rotates with the pixel and
  there is no block form to write down. `rocket_rk3576_pad_oc()` gives the count; leave
  the padding zero.
- **A task's surface is bounded at 8 KiB, and the bound is over the plane.** State it in
  the programmed channel count: a row task is correct while `iw * oh_full * oc_prog <
  4096`. A wide task writes `oc_prog/8` atoms a pixel at 16 bytes each, so that product
  is the output surface in half-bytes and the bound is **8192 bytes, or 512 atoms, per
  task**. That is what `rocket_matmul_int8_rk3576_i32()` enforces, splitting the row plan
  until every task fits, a row task here is a standalone 1x1 convolution with its own
  surface, so honouring it costs submits and nothing else.

  **`iw` is in it, and leaving it out is a wrong answer rather than a lost bound.** A row
  split can only shorten `oh_full`; it cannot narrow `iw`, and the plane chooser takes the
  widest divisor of M it can, so on a shape whose plane comes out `Mx1` a cap that
  divides only by `oc_prog` is satisfied by every task while the surface is M times over
  it. Forced onto eleven shapes whose planner plane is `Mx1`, the wide writer was wrong on
  ten; at `iw = 1`, where the two readings coincide, all eleven are exact and every zero
  run disappears. A plane too wide for even a one-row task cannot be split into one, so
  such a shape cannot use the wide writer at all and the entry refuses it.
  [HW sweep, H96 MAX M9]

  Swept at `K = 32` with the cap lifted (`ROCKET_RK3576_I32_WIDE_SURF`,
  `tests/rk3576_i32_height_bound.c`) at `ow = 1`, the first wrong height is the smallest
  `oh_full` with `oh_full * oc_prog >= 4096` at **every one of the eight programmed
  counts**:

  | `oc_prog` | 64 | 128 | 192 | 256 | 320 | 384 | 448 | 512 |
  |---|---|---|---|---|---|---|---|---|
  | first wrong `oh_full` | 64 | 32 | 22 | 16 | 13 | 11 | 10 | 8 |

  Predicted and observed agree in all eight. [HW sweep, H96 MAX M9]

- **K relaxes the bound, and only ever outward.** The same `oc = 128` that is wrong from
  `oh_full = 16` at `K <= 128` is wrong at only 31, 32 and 40 at `K = 512`, and exact to
  `oh_full = 40` at `K = 1024` and `K = 2048`. So 8 KiB is the worst-case floor, reached
  once the contraction is short enough, and enforcing it unconditionally is what makes the
  path correct at every K.

  **That is also what hides `iw`.** A sweep of the plane width at `K = 512` finds `ow = 2`
  and `ow = 4` exact across the whole range in which `ow = 1` fails, which reads as "the
  plane width is not in the bound" and is really "at this K the bound is far outside the
  geometries tried". At `K = 256` the same doubling breaks it: `oc_prog = 512`, `oh = 7`
  is exact at `ow = 1` and wrong at `ow = 2`. Sweep the width at a short contraction, or
  the axis is invisible.

- **Both writers drop atoms, and the drop tracks the memory system.** Inside the bound the
  failure is not gone, only rare: a row task comes back with a few atoms holding zero
  while the rest of the surface is exact. It is not the wide writer's alone, the narrow
  one does it too, and had no per-atom check because its surface was believed never to
  drop: `M=128 K=256 N=2048` returned a handful of elements out of 262144 reading zero on
  six calls in twelve, silently.

  The rate is a property of the memory system rather than of the program, and it is host
  DDR traffic specifically, not the host merely being busy. Four load threads at
  `M=128 K=256 N=2048`, 30 reps and 120 row tasks per cell:

  | host load                          | row tasks that dropped an atom |
  |------------------------------------|-------------------------------|
  | quiet                              | 5.8%   (7/120)                |
  | integer spin, no memory traffic    | 11.7%  (14/120)               |
  | memcpy over 16 KiB, L1-resident    | 9.2%   (11/120)               |
  | streaming memcpy over 32 MiB, DDR  | **51.7%** (62/120)            |

  The two controls hold the CPUs equally busy and the cache hierarchy equally hot while
  leaving DDR alone, and at this sample size they are not distinguishable from quiet
  (7, 11 and 14 of 120; a binomial sd is about 3). Streaming to DDR is 5-9x all three and
  far outside that. So the effect is specific to traffic reaching DDR, and CPU occupancy
  and scheduling latency are ruled out as the mechanism. The count of atoms never emitted
  at all, the wide-output poisoning, barely moves across the same contrast, which is
  what makes the two hazards separate. [HW sweep, H96 MAX M9, 2026-07-28,
  `tests/rk3576_i32_zero_run.c load`]

  **It does not reach the conv entries or the narrow int8 matmul.** At the load that takes
  the int32 writers to 51.7%, `rk3576_conv_lib_gate` is 156 passed / 7 refused / 0 failed
  under DDR pressure, under the L1 control and quiet alike, and `rk3576_matmul_gate` is
  46/46 under DDR pressure. (One 155/156 flake appeared under the *spin* control, which
  touches no memory, and the same single-case flake also appears with no load at all, so
  it is a pre-existing intermittent in that gate, not a load effect. It has not been
  pinned to a case.) The conv entries' own "did it write" check is per row task and would
  not see a task that wrote all but a few atoms, so this is a measured negative rather
  than a covered one. [HW, H96 MAX M9, 2026-07-28, `tests/hostload.c`]

- **Every wrong element comes back zero**, past the bound and inside it alike. Over the
  whole map, eight channel counts, 57 heights each, not one wrong element aliases
  another element's correct value and not one holds anything else. The writer emits empty
  atoms rather than misplacing full ones.

- **A dropped atom cannot be told from a zero accumulator by looking at the surface**, so
  do not try. The bytes are identical, and repetition does not separate them either: the
  drop is correlated with the memory system and repeats often enough to leak a wrong
  answer through a "came back the same twice, so it is data" rule. What separates them is
  the arithmetic. The library stamps a sentinel, so "never emitted" is a fact per atom;
  redoes the task once, which clears most drops; and from the second attempt computes the
  zero atoms' accumulators on the CPU over that K slice, keeping the surface only if they
  really are zero. Measured at the worst shape, 240 reps, 120 of them under host load,
  every one exact, with the all-zero-data case (every row of A zeroed) passing on both
  writers.

**A probe that drives every output channel finds a boundary the library never meets.**
With all sixteen channels of a group live the map stops fitting well before the bound
above, at `oc = 192` from `A = 48` on a flat plane. Under the scatter, where the
undelivered half of each sixteen is zero, the same geometries are bit-exact through the
library at `A` 48, 64 and 128. So a boundary found with every channel live is a boundary
of the probe: `ROCKET_MP_SCATTER=1` drives the probe the way the library drives the part.

**What it buys.** The N tile is bounded by the programmed output channels, so halving the
multiplier doubles the tile and halves the submits, and on this path a submit costs an
idle rather than 1.4 ms. Measured through `rk3576_matmul_gate`:

| shape | 4x writer | 2x writer |
|---|---|---|
| M=64 K=1024 N=2048 | 4 submits, 686 ms | 2 submits, 339 ms |
| M=32 K=1024 N=4096 | 8 submits, 1319 ms | 4 submits, 675 ms |

Below the tile cap the two cost the same, so a shape whose N fits one tile shows no gain.

**The planner picks the writer per shape, and `ROCKET_RK3576_I32_OC_MULT` forces either.**
The whole benefit is submits, the whole cost is submits, and both are computable before
anything is submitted: `r76_i32_submit_count()` counts K slices x N tiles x row tasks for
each writer, with the surface cap splitting the wide one's tasks further, and the
cheaper wins, ties to the narrow one.

**Once the bound is stated over the plane, the wide writer stops winning.** The cap costs
it exactly the submits its wider N tile was reached for, and it costs them on the N-heavy
shapes where the tile would have paid: swept over 315 natural shapes, M 4-256, K 64-2048,
N 32-2048, the chooser took the narrow writer on **every one**. Earlier figures showing
it ahead (8 submits to 4 at `i32-n4096`, 4 to 2 at `i32-n2048`) were measured against the
rows-only cap, which let those shapes run one task per tile at many times the surface the
writer can hold. The wide path is kept because it is the instrument that decoded the
32-bit writer's map and because a shape outside that sweep may still reach it, not because
the default path uses it. [HW sweep, H96 MAX M9]

**Sparse data does not cost either writer its retry budget.** With every row of `A` zero
and the bias zero with them (`ROCKET_MG_ZERO_ROWS=1`), both writers pass the whole int32
gate. A legitimately zero surface used to answer "no" to the narrow path's "did this task
write anything at all" every time; it now answers a per-atom question against a sentinel,
and a zero atom that survives one redo is settled against the operands rather than against
the surface.

**Both writers intermittently emit zeros**, which is what the detector exists for. On the
wide one the block is contiguous and one super-group wide; on the narrow one it is a
handful of atoms. It is the same defect and it is not rare; see the memory-pressure
measurement above.

**It does not drop writes.** That was the earlier reading and it was wrong. Stamp the
output BO with a sentinel before the tasks run, bracketed by `PREP_BO` and `FINI_BO`, so
no dirty line is left to race the DPU's DMA, and verified to reach DDR before any submit,
and the bad atoms come back **zero rather than holding the sentinel**. The writer
reaches them; the data is wrong. On a fresh BO, which arrives zeroed, the two are
indistinguishable, which is what made a dropped write look like the answer.

**What it is not.** Not a readback race: a second fence and a second read return
byte-identical contents. Not the poisoning below, whose signature is an *empty* region and
which the power domain cycling clears; this one is indifferent to the idle ahead of it,
and host memory traffic moves its rate fourfold while leaving the poisoning's count alone.

**Its shape, in the wide writer's own coordinates.** The writer emits two atoms per stream
position `s`, one per lane group, so its emission order is `(s, L)` ascending. The corrupt
block is contiguous in *that* order and in no other, in address order it shows up as two
separate holes, and it lies inside a single super-group.

**Its extent does not generalise across shapes**, which is why no detector keys on it. On
the shape it was first measured on every failure over 40 runs ended at `s = 2A-2`, two
emissions short of that group's last. That does not hold: logged at debug level across
shapes (`r76_i32_run_log()`), `i32-tall-m` and `i32-m100` corrupt **three emissions in
mid-stream**, at `s = 1039..1040` of 0..2047 and `s = 15..16` of 0..199. Both recur at a
fixed address, so a tail-keyed detector would miss all of it. [HW sweep, H96 MAX M9,
25 runs per shape]

**So it is detected and settled, not avoided**, and the settling is exact rather than
heuristic. Two adjacent zero emissions on the wide path, or any zero atom on the narrow
one, is the signal; the task is redone once, which clears most of it; and from the second
attempt the zero atoms' accumulators are computed on the CPU over that K slice, so the
surface is kept only if they really are zero. `r76_i32_wide_suspect()` and
`r76_i32_zeros_are_data()` in `rocket_matmul_rk3576.c`. The redo needs **no idle in front
of it**, which is further evidence this is not the poisoning; the sentinel's other
reading, an atom that still holds it, *is* the poisoning and that redo keeps its idle.

**What has been eliminated.** At `M=8, N=32` on an 8x1 plane the baseline is 64 words:

| driven | extent | what it did |
|---|---|---|
| nothing (widths 0, 4, 5) | 64 words, last at 63 | the baseline; identical at all three widths |
| `0x402C` and `0x4030` high half, oc x4 | 64, last at 63 | nothing at all |
| `0x401C` (DST_SURF) doubled | 64, last at **95** | spaced the blocks apart; it is the block stride |
| `0x40B8` (SURFACE_ADD) doubled | 64, last at 63 | inert at `ih=1`, where the task window is the plane |
| `0x401C` **and** `0x40B8` doubled | **0 words** | the DPU wrote nothing, so the two are coupled and `0x40B8` is live |
| `0x4030` low half 0x710 -> 0x310 | 64, last at 63 | nothing (the 7-vs-3 the RK3588 calls `size_e`) |
| `0x4030` low half -> 0xF10 | 64, last at 63 | corrupted the values without moving the extent |
| CNA `0x1024` (kernel word \| oc-1) x4 | 64, last at 63 | nothing |
| DPU_RDMA `0x5014` (CUBE_CHANNEL) x4 | 64, last at 63 | nothing |
| CORE `0x3020` (dataout_channel) x4 **alone** | **0 words** | killed the write outright; it is coupled to the others, not inert |
| all five oc carriers x4 together | **256**, the full surface | the byte budget scales, the delivered channels do not, still 8 |

So the delivered-channel rule survives every oc register on the part. Scaling them
together buys address extent and nothing else, which is what says the rule is in the
datapath rather than in a geometry field. [HW sweep, H96 MAX M9, measured 2026-07-27]

### An i32out job poisons the next submit, and a power cycle is what clears it

**After an i32out job, the next submit of any kind writes nothing.** The job that does
not write **completes normally**: no fault, no IOMMU error, no timeout, the usual
the usual per-submit time, and leaves its output buffer untouched. If the caller zeroed that buffer it
reads as an all-zero surface; if the caller reused one it reads as the previous result,
byte for byte. Either way it looks like an arithmetic failure, and it is not one. It
crosses processes: the plain int8 matmul in another program is as exposed as the path
that caused it, because the state lives in the NPU and not in the fd.

**What clears it is the driver's runtime-PM autosuspend cycling the NPU power domain,
not elapsed time.** Three measurements say so, and together they close it:

- Write `on` to the device's `power/control`, so runtime suspend never fires, and **no
  gap clears it at all**, 100 ms, 300 ms and 600 ms each write nothing.
- The working gap tracks `power/autosuspend_delay_ms` one for one. At a 5 ms delay a
  5 ms gap already gives 12 writes of 12; at 50 ms it takes 80-120 ms for the same.
- `power/runtime_suspended_time` advances across the gap, so a suspend really did happen.

So the "50 to 100 ms of idle" this first read as is that file's stock value of 50 plus
the suspend/resume round trip, and nothing about the silicon. The cost of the workaround
is therefore **settable**: lowering the delay shortens the path proportionally, the
matmul gate's int32 half runs in 553 ms at the stock 50 and 195 ms at 5, and a
kernel-side reset of the DPU at job start would remove it outright.

**It is one hazard and every wide output carries it.** Measured with a canary, submit
the program under test, then a program known to write and known not to poison, and score
the canary, so that a dead program and a poisoning one stay distinguishable
(`tests/rk3576_poison_probe.c`). Over seven program kinds the split is exact: the fp16
first conv, the fp16 direct conv and both int32 writers each poison every one of the
seven; the int8 first conv, the int8 direct conv and the int8 depthwise poison none.
Every program under test wrote its own surface in every cell, so no cell is a dead
program reading as a clean one. **One output byte never carries it, wider always does.**

**A sacrificial submit does not absorb it.** After one int32 job, six consecutive int8
jobs at a zero gap all wrote nothing, so the state persists until the domain cycles and
the guard cannot be a cheap dummy job.

**The cycle clears it about 87% of the time, so what makes the guard reliable is the
number of redos, not a better cycle.** Over twelve fp16 conv gate runs the redo fired on
358 row tasks; the attempt that failed *next* was attempt 2 for 45 of them, 3 for 10, 4
for 2, 5 for 1 and 6 for none. Every one recovered, and every cycle in between was
confirmed to have taken the domain to `suspended` first, the guard was never merely
unobserved. At a cap of four attempts that leaves about 0.6% of retried tasks failing
outright, which is exactly the rate the conv gate saw as a "1-in-156 intermittent"; at
eight it is unobservable (15 of 15 fp16 runs and 3 of 3 full runs clean, deepest retry
attempt 4). Cap the redos at eight. [HW sweep, H96 MAX M9, 2026-07-28]

**A task that reads unwritten really is unwritten**: the fence is not signalling ahead
of the writes on this path. Asking again after a 2 ms settle before the redo rescued
**0 of 67** such tasks, while the power cycle behind it recovered all 67. The re-check
costs 14% of the fp16 wall and buys nothing (`ROCKET_RK3576_DRAIN_US`, default 0).

**DPU `0x4010` (the DATA_FORMAT word, `out<<29 | in<<26 | proc`) carries it on the int32
path, and no single register carries it in general.** The i32out program differs from the
int8 direct conv in only four registers, `0x4010`, `0x40AC`, `0x40B0`, `0x40B4`, and
zeroing `0x4010` alone stops the poisoning with the program still writing, where the
other three leave it. But the sufficient condition is joint, not one register:

| program | `0x4010` forced to | poisons? |
|---|---|---|
| int8 direct | `0` (its own) | no |
| int8 direct | `out`=1, 2 or 3, any `proc` | no |
| int8 direct | `out`=4 or 5 | yes |
| i32out | `0` | no |
| i32out | any non-zero, including `out`=1..3 and `proc` alone | yes |
| fp16 first conv | each of its 19-register delta, one at a time | yes, all 19 |

So the same `out` value is clean on one program and poisoning on another, and a
leave-one-out over the whole fp16 first-conv delta revives nothing. The mechanism is
undecoded; what is decoded is its scope and its price. The next job programs `0x4010`
back to all-int8 and still comes back empty, so this is latched datapath state rather
than a stale register.

Nothing about the program or the buffers changes it. Fresh buffers per submit, a
different feature address, a different output address, and packing every K slice up front
so no buffer is rewritten between submits were each tried, and none helps.
`rocket_matmul_int8_rk3576_i32()` sizes its idle from the driver's autosuspend delay
(twice it plus 20 ms), idles between its own submits and once more on the way out, and
retries any submit whose own region came back untouched.

**Check that a job wrote per task, not per tile.** A row-split tile issues several
submits into one output buffer, so one poisoned submit among them leaves its own rows
empty while its siblings are full, and a check over the whole tile then reads "something
was written" and hands the hole to the caller. It shows up as a result that is exactly
one row task short, intermittently, with no fault and normal timing. Both writers need
this; it was found on the narrow one, which had run clean for a session because no gate
shape until then split a tile into row tasks.

**The cost is settable from the driver, and the law is twice the delay.** Over a 5x7
sweep of `power/autosuspend_delay_ms` against the inter-submit gap, driving a chain of 20
fp16 first-conv submits and counting how many wrote:

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
see the cost.** A resume is about 230 us, and lowering the delay makes every job whose
inter-job gap falls between the new delay and the old one pay it. Measured with
`ROCKET_SF_GAP_MS` on `rk3576_submit_floor`, us/submit:

| inter-submit gap | 0 | 5 | 10 | 16 | 20 | 33 | 60 | 100 |
|---|---|---|---|---|---|---|---|---|
| delay 50 ms | 1062 | 1123 | 1130 | 1124 | 1136 | 1139 | 1283 | 1399 |
| delay 1 ms | 1059 | 1319 | 1334 | **1372** | 1352 | 1368 | 1366 | 1378 |

At a zero gap the two are identical, the autosuspend timer never fires, which is why the
plain floor probe reads the change as free. They converge again past 60 ms, where both
suspend anyway. In between, the low delay costs ~20% a job, and **a 60 Hz frame interval
of 16.7 ms sits squarely in that window**, which is exactly the pipeline mainline's 50 ms
was chosen for.

**So drive the transition rather than waiting for it.** `rocket_rk3576_power_idle()`
writes a zero delay, polls `runtime_status` until it actually reads `suspended`, and puts
the caller's delay back. The guard then costs one real power cycle instead of a worst-case
idle, and the steady-state policy every other consumer of the NPU sees is unchanged. It
needs write access to the sysfs file, so it is probed once and falls back to the plain
idle without it; `ROCKET_RK3576_PM_KICK=0` turns it off. The few-millisecond window in
which a kill would leave the delay at zero is self-healing, zero is not a value a system
sets deliberately, so it is read as an interrupted kick and restored to mainline's 50.

At the stock 50 ms delay, with every gate still bit-exact (conv 102, conv-lib 154 + 8
refusals, matmul 43 on both writers, refusal 15, regcmd 194, both fp16 sweeps): the fp16
first conv's 224x224 k3 s2 runs in **13.75 ms against 226**, its whole gate group in
2.3-15.6 ms against 107-334, and the int32 matmul gate in 9.9 s against 16.2. That beats
lowering the global delay to 1 ms, which reached only 21.6 ms on the same shape and taxed
every gapped workload to get there. [HW sweep, H96 MAX M9, measured 2026-07-27]

**The completion-visibility race is closed by `patches/rk3576/npu/0012`**, which retires
the poll on the DPU's own completion bit rather than on `PC_DONE`; the reproducer that
found it, and the shape of the fix, are under "The per-submit floor is the driver's
completion poll" above. `poll_interval_us` is a live module parameter, so the period still
sweeps from sysfs with no rebuild, which is how the sweep against correctness rather than
wall time was run, and how a candidate fix is tested.

**The kernel-side fix that would remove the poisoning idle entirely is identified but not
built.** `rocket_core_reset()` already exists in the driver (a `reset_control` assert,
`udelay(10)`, deassert); calling it at job start is the shape of it. The vendor `rknpu`
driver is not the model here; it soft-resets only on a timeout or an abort, never per
job. Note also that the **per-submit floor on this part is the driver's poll, not the
silicon**: PC_DONE is not routable to the GIC on the RK3576, so
`patches/rk3576/npu/0006` polls it where the RK3588 takes an interrupt. Every RK3576 cost
table is a submit-count table because of that.

**The hazard reaches the plain int8 matmul too, and that path used to pass it straight
through.** The poisoning is not confined to the int32 entry that creates it: it outlives
the call and the process, so `rocket_matmul_int8_rk3576()` inherits it from whatever ran
before, and its first submit comes back untouched while the caller reads a correctly
sized, entirely stale tile. Measured in a soak, one full matmul-gate run in twenty failed
its very first shape this way, at 1 element of 256 correct. That entry now stamps its
output surface and redoes any row task whose region is still the stamp; 20 further gate
runs were clean. [HW sweep, H96 MAX M9, measured 2026-07-27]

**Do not zero an output buffer from the CPU before a submit**: but a bracketed stamp is
a different thing and is worth having. A bare `memset` leaves dirty cache lines that race
the DPU's DMA, and the writeback lands on top of the result, so the surface comes back all
zeros, intermittently and on the first submit as readily as a later one. A fill bracketed
by `PREP_BO` and `FINI_BO` is written back before the submit and leaves nothing dirty;
both matmul entries use one, and it is what makes "this was never written" a property of
the surface instead of an inference from its value. A freshly allocated BO arrives zeroed,
which cannot distinguish an unwritten word from a legitimate zero; a guard band past the
surface wants 64-byte alignment so no line it dirties is shared with a surface byte.

The int32 sweep also **wedged the IOMMU** once (stall timeout plus a
`rocket_job_irq_handler` warn at `rocket_job.c:528`), reboot-only. Do not read that as an
overrun: the measured extent sits well inside the BO that was allocated, so the mechanism
is unestablished and treating it as a buffer-size problem would be a false trail.
[HW sweep, H96 MAX M9, measured 2026-07-27]

### The instrument

`tests/rk3576_matmul_probe.c`, modes `shape` (the mapping vs a CPU model), `m` (the M
axis, every factorization), `cost` (both precisions on one shape), `peak` (MACs per
submit by N), `ncap` (the N boundary, with repeats and a failure classification) and
`out32` (the 32-bit output width, decoded). Knobs: `ROCKET_MP_I32`, `ROCKET_MP_IW`,
`ROCKET_MP_KS` / `_NS` (comma lists), `ROCKET_MP_REPS`, `ROCKET_MP_GAP_MS`,
`ROCKET_MP_OSLACK`, `ROCKET_MP_VERBOSE`.

Two traps it cost to learn. **`rocket_submit_matmul()` returns before the job does**, so
timing it without fencing on the output BO reports 0.0 ms. And **pace between runs, not
just between tasks, and keep the pacing outside the measurement**, an unpaced job
completes without writing at all, which reads as an arithmetic failure and not as a
scheduling one.

## The DPU LUT: the table is reachable, and the protocol is a two-register window

Every convolution capture of this part leaves the LUT bank (`0x4100`-`0x4194`) zero, which
is why `npu_regcmd_rk3576.c` writes the whole bank as zeros. That is a property of the
captures, not of the silicon: **a manufactured capture drives the LUT**. An ONNX carrying a
nonlinear activation, compiled for `rk3576`, emits the table load verbatim
(`tests/data/rk3576-vendor-capture/lut/mklut.py`, decoded by `decode_lut.py`).

**A nonlinear activation is a separate program, not a fused epilogue.** The compiler emits
two programs per conv-plus-activation: one that is DPU + DPU_RDMA only, with no CNA and no
CORE, and one ordinary convolution. The DPU-only one drives a 1x1x16 dummy cube
(`0x401C = 1`, `0x4030 = 0x000f0f00`, `0x40B0 = 0x00010001`); its work is loading the
table, not computing. The convolution that follows it is the one that computes, and it is
configured to use the table: against a bare-conv control, whose whole LUT bank is zero, it
writes `LUT_CFG 0x4108 = 0x02000006`, `LUT_INFO 0x410C = 0x00050500` and all four of
`LE_START`/`LE_END`/`LO_START`/`LO_END` (`0x4110`-`0x411C`) = `0xffffc000`.

**The table is written through a two-register window that auto-increments.**
`LUT_ACCESS_CFG` (`0x4100`) selects the table and the start offset; every subsequent write
to `LUT_ACCESS_DATA` (`0x4104`) stores one entry and advances. In the load program
`0x4108 = 0x00ff0000` first, then:

| `0x4100` | entries | what |
|---|---|---|
| `0x00020000` | 513 | one table |
| `0x00030000` | 513 | the other |

513 = 2^9 + 1, so each table is 512 intervals with both endpoints, the shape a
linear-interpolating LUT wants, and the same LE/LO pair the NVDLA SDP documents.

**The two tables are the lower and upper halves of one monotone curve**, joined at the
value the function takes at the split, and the entries are **Q15 of the function's
output**. `Sigmoid` runs `0x3b -> 0x4000` in the `0x00020000` table and `0x4000 ->
0x7fc4` in the `0x00030000` one, and `0x4000/0x8000` is exactly 0.5. `Tanh` runs
`-0x7f63 -> 0` and `0 -> 0x7f63`, symmetric about zero as it must be.

**The table is uniform in the input, and each function's span in its own units is the
compiler's choice.** It is recoverable from the first difference and the function's own
derivative at the join: sigmoid's `d = 100` at the join with `sigmoid' = 0.25` gives a
step of 0.0122 and a half-span of 6.25; tanh's `d = 193` with `tanh' = 1` gives 0.00589
and 3.02. Both agree with the endpoint values to three digits. The two differ because
the compiler puts the table where each function saturates, and it expresses that choice
by scaling the value the datapath hands the LUT, not by moving the LUT's own window.

**`0x4188`/`0x418C` and `0x4190`/`0x4194` are the output clamps**: the value used below
the table and above it, replicated across 16-bit lanes, and equal to the tables' first
and last entries. Verified on four functions at once: sigmoid `0x003b`/`0x7fc4`, tanh
`0x809d`/`0x7f63`, softplus `0x0094`/`0x7fff`, swish `0xff9d`/`0x7fff`, each matching its
own table's endpoints exactly.

**`0x4150`/`0x4154` with `0x4160`, and `0x4170`/`0x4174` with `0x4184`, are the
underflow and overflow linear-extrapolation slopes** (a scale and a shift each), and the
four functions' tails predict their own values: `tanh` saturates both ends and carries
zero in all four; `sigmoid` carries a small lower slope and zero above; `swish` carries
zero below and about 1.0 above; `softplus` carries a small lower slope and about 1.0
above. That is exactly the pair of tails each function has. [Manufactured capture]

**`0x40AC`/`0x40B0`/`0x40B4` on a LUT case are the LUT's output requant**, not its input
map. They are the DPU's ordinary output converter, the piecewise activations and the bare
conv write them too, and on a fused nonlinear activation they carry Q15 to the op's own
quantization exactly: sigmoid's output is `[0, 1]` at zero point -128, so its gain is
`255/32768 = 7.7820e-3` and the registers read `0x7f81 >> 22 = 7.7815e-3`; tanh's is
`[-1, 1]` at zero point 0, so `127.5/32768 = 3.8910e-3` against `0x7f81 >> 23 =
3.8907e-3`. Same multiplier, one more shift, and `0x40AC` is -128 for sigmoid and 0 for
tanh. Five digits on two functions is not a coincidence, and it means these three say
nothing about which interval an input lands on. [Manufactured capture]

**The input map is `index = (value - LE_START) / 2^sel`.** Read off the part with our
own table-load program and a table whose entries encode their own index
(`tests/rk3576_lut_probe.c`), against a convolution whose accumulator is one feature
byte and whose per-output-channel `A` and `C` place that byte anywhere in the datapath:

| register | field | value the vendor writes | what it does |
|---|---|---|---|
| `0x4110`-`0x411C` | `LE_START`, four lanes | `0xffffc000` = -16384 | where the LE table starts |
| `0x4140`-`0x414C` | `LO_END`, four lanes | `0x00004000` = +16384 | where the LO table ends |
| `0x410C` | `LUT_INFO`, two index selects | `0x00050500` | the index step, `2^5 = 32` |

The `0x00020000` table (LE) covers `[LE_START, 0]` and the `0x00030000` table (LO)
covers `[0, LO_END)`, 512 intervals each. **All three registers are live**: the map is
bit-explained at `sel` 4, 5 and 6 and at an asymmetric `LE_START = -8192` with
`LO_END = +16384`, over 8192 samples per span, max disagreement one output count
(`rk3576_lut_probe gate`, 4 of 4). The vendor never moves them because it places the
value inside the fixed window with the BS stage's own per-channel `C` instead, which
is why no diff of vendor programs could ever produce this, and why at fp16, where the
scale rides the coefficient buffer rather than a register, sigmoid and tanh emit
identical programs with tables spanning 6.25 and 3.02.

**The hardware interpolates linearly between entries, at the full resolution of the
step.** A step table walked one datapath unit at a time across the interval it steps in
gives 33 distinct output runs over 32 units, rising linearly, not two.
[HW sweep, H96 MAX M9]

**The domain is half-open at the top.** `value == LO_END` takes the overflow clamp, not
the last table entry; `value == LE_START` is in domain and reads `LE[0]`. Every
disagreement in the first run of the gate, at four different `C`, at all four spans,
sat on that one value and nowhere else.

**BN_CFG `0x4060` is a trap, and it is why the map looked absent.** Every vendor
activation carries `0x20` there, the BN stage active, against `0x903` in its own
bare-conv control. Transcribe it and the LUT returns the value at the table join for
every input from -2^21 to +2^21, a perfectly constant surface that tracks the table
faithfully as the table is changed, so it reads as a working LUT with an input map that
is nowhere in the registers. What is actually happening is that BN then multiplies by an
operand register **no vendor program writes**, and this register file is not cleared
between jobs. Left at the generator's own `0x903` the BN stage is bypassed, the LUT reads
the BS output directly, and the map appears. The four registers the vendor's own programs
leave alone (`0x4040`, `0x4054`, `0x4064`, `0x4068`) were each driven and none of them
moves it, so where this part keeps `BN_MUL_OPERAND` is open, and it does not need to be
found: the BS stage's per-output-channel `C` is a per-channel scale where a BN multiply
would be one global one.

**What turns the LUT on**, diffed against the bare-conv control of the same capture set
rather than across activations, which is what hid it:

| register | control | LUT | |
|---|---|---|---|
| `0x407C` | `0x010041C1` | `0x01004140` | EW_CFG: `EW_LUT_BYPASS` (bit 7) and `EW_BYPASS` (bit 0) clear, so the LUT is the **EW** stage |
| `0x4108` | 0 | `0x02000006` | LUT_CFG |
| `0x410C` | 0 | `0x00050500` | LUT_INFO |
| `0x4110`-`0x411C` | 0 | `0xffffc000` | LE_START |
| `0x4140`-`0x414C` | 0 | `0x00004000` | LO_END |
| `0x4188`-`0x4194` | 0 | the tails | the two clamps |
| `0x5028` | 0 | `0x1A` | DPU_RDMA NRDMA_CFG |
| `0x1004`/`0x3004`/`0x4004`/`0x5004` | `0x0E` | `0x30` | every block's S_POINTER |
| `0x4060` | `0x903` | `0x20` | BN_CFG, **do not transcribe this one** |

That table is what `gen_conv2d_int8_rk3576()` emits from `conv_params_t.lut`
(`lut_rk3576_t`: `le_start`, `lo_end`, `sel`, and the two clamps). The whole bank is
written either way, with zeros when the field is NULL, so a stale window from the previous
job cannot survive into this one; this register file is not cleared between jobs. With
`lut` NULL every program is byte-identical to the ones emitted before the field existed,
which `regcmd_rk3576_gate` asserts.

**The two clamps are packed as the vendor packs them**: the value alone in the high half
of the first register of each pair and doubled across the second, `0x4188` = `lo << 16`,
`0x418C` = `(lo << 16) | lo`, and the same at `0x4190`/`0x4194` for the high clamp.
`0x4184` stays zero.

The EW field layout is the RK3588's and transfers unchanged (`EW_LUT_BYPASS` at bit 7,
`EW_BYPASS` at bit 0), which is what the two values decode to; the register offset does
not (`0x4070` there, `0x407C` here).

**`gen_lut_load_rk3576()` emits the table-load program**, 1121 words as the vendor's is:
the 1x1x16 dummy cube, the `0x4100 = 0` / `0x4104 = 0` / `0x4108 = 0x00ff0000` preamble,
the two 513-entry bursts, and the four-word trailer whose `PC_OPERATION_ENABLE` is
**`0x18`**, DPU and DPU_RDMA, neither the convolution's `0x1D` nor the pool's `0x60`.
Its dummy cube writes nothing at any address, so it is not its own positive control; what
says the table loaded is that a consuming convolution's readout tracks the table.

The load and the use want to be one job. The table lives in the LUT RAM and a separate
submit can take a runtime-PM cycle in between.

## The PPU: pooling is a 31-write program, and it is standalone

Nothing in any found capture drives the pooling engine. Manufactured ones do
(`tests/data/rk3576-vendor-capture/pool/mkpool.py`): a `MaxPool` or `AveragePool` behind a
convolution emits **an independent 31-write program, 23 PPU writes and 8 PPU_RDMA**, and a
bare pool emits exactly that program with no convolution beside it. So pooling is its own
NPU program, in the same NC1HWC2 cube the convolution path already packs, and it needs no
new host packing.

**`GlobalAveragePool` is not pooling.** It compiles to convolutions, every program in that
capture is CNA + CORE + DPU, the same lowering the RK3588 uses for a reduce. `GlobalMaxPool`
is pooling, in two cascaded PPU passes (k7 s7 then k3 s3 over what is left).

The 23 PPU writes, read off a sweep over method, kernel, stride, plane, channel count and
padding:

| reg | meaning |
|---|---|
| `0x6004` | `0x0e`, constant (the same enable word CNA/CORE/DPU take at their `+4`) |
| `0x600C` | input width consumed, minus 1 |
| `0x6010` | input height consumed, minus 1 |
| `0x6014` | input channels minus 1, **rounded up to 16** |
| `0x6018` | output width minus 1 |
| `0x601C` | output height minus 1 |
| `0x6020` | output channels minus 1, same rounding |
| `0x6024` | mode: `0x11` max, `0x10` average, `0x18` average with the pad excluded from the divisor |
| `0x6034` | `(sy-1)<<20 | (sx-1)<<16 | (kh-1)<<8 | (kw-1)` |
| `0x6038` / `0x603C` | `1/kw`, `1/kh` in Q16, `0x8000` at k2, `0x5555` at k3; zero for max |
| `0x6040` | four pad nibbles, `right|left|bottom|top` |
| `0x6044`-`0x6050` | pad values: `-128` (as `0x0007ff80`, sign-extended in a 19-bit field) for max; the input zero point times 1, 2, 3, 4 for average |
| `0x607C`, `0x6084` | `round4(ow*oh) * 16`, the destination surface stride per 16-channel group, the SAME round-to-four the convolution's `0x401C` takes |
| `0x6054`, `0x6058`, `0x605C`, `0x6070`, `0x60DC` | zero in every capture |

**The input extent is what the windows consume, not the plane.** `0x600C`/`0x6010` carry
`(ow-1)*sx + kw` and `(oh-1)*sy + kh` clamped to the plane, so a 15x18 plane pooled k2 s2
programs 18x14 and not 18x15, the last row no window reaches is simply not described.
Every non-square and odd-plane case in the sweep agrees, and a plane-height reading does
not.

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

The strides are the only place the full plane appears, and `0x7024` is what separates
them from the consumed extent: a 19-wide plane pooled k2 s2 consumes 18 and strides 19.

**The consumed extent excludes the leading pad**, so it is
`min(plane, (ow-1)*stride + k - pad_start)` and not the windows' raw span. A VALID k3 s2
over 16 programs 15; a SAME k3 s2 over the same 16 programs 16, where its windows span
17.

**The end pads sit in the low nibbles of `0x6040`.** a SAME k3 s2 over 16 needs
`pad_right = pad_bottom = 1` and nothing on the leading edges, and the capture carries
`0x0011`, so the low half is right and bottom. Every other captured pad is symmetric
and cannot tell the two orders apart; which of the low pair is right and which is bottom
is still open, and equal in every shape emitted so far.

**The destination base address is `0x6070`**, and no capture could have said so: like
every other base on this part the vendor runtime patches it at load time and the stored
program carries zero. It was read off the part by driving each of the five PPU registers
that read zero in every capture in turn and asking only whether the output BO moved off
its sentinel, 0x6054, 0x6058, 0x605C, 0x6070, 0x60DC, and only 0x6070 writes.

**`PC_OPERATION_ENABLE` is a per-block bitmap, and this is the trap.** A convolution
enables its blocks with `0x1D`; the vendor's pooling program enables PPU and PPU_RDMA
with `0x60`, and the two bit sets are disjoint. A pool carrying the convolution's
trailer is a fully configured PPU that is never started: the job completes in the usual
per-submit time, faults nothing, and writes nothing, indistinguishable by inspection
from a wrong destination address, and it cost a whole register sweep that came back
empty before the raw capture was read word by word instead of through the block
classifier. **Transcribe the whole program, trailer included.**

**The average rounds half to even**, the same rule the DPU's `OUT_CVT` was measured to
use. Against round-half-away-from-zero a k2 average disagrees on one output in eight.
The divisor is the window (`0x6038`/`0x603C` are `1/kw` and `1/kh`), not the tap count;
excluding the pad from it is what mode `0x18` is for.

With those, `gen_pool_rk3576()` computes bit-exactly against a CPU model over max and
average, kernel 2/3/5, stride 1/2/3, non-square and odd planes, 8/32/64 channels, padded
and not, 11 shapes, `tests/rk3576_pool_probe.c`.
[Manufactured capture + HW sweep, H96 MAX M9, 2026-07-29]

**`rocket_pool_int8_rk3576()` is the library entry over it**: row-major `[C][IH][IW]` in
and out, owning the cube, the sentinel, the submit and the de-scatter, the same shape the
convolution entries have. It is a separate entry rather than a dispatch from the RK3588's
`rocket_pool_int8()`, because that one truncates the average toward zero and this one
rounds half to even: two roundings are two functions. It takes an input zero point, which
the RK3588 entry does not, because the average path pads with it.

**Whether the average is exactly rounded is a function of the window size**, and
`rocket_pool_int8_rk3576_exact()` answers that for a descriptor without running it. The
reciprocals are truncated, `0x10000/kw` and `0x10000/kh`, so the computed quotient is
low by `|sum| * (1/n - rw*rh)`, and against the closest a quotient of an integer by `n`
comes to a half, `1/(2n)`, that error is far under the boundary at k2/k3/k5 and is not
guaranteed to be at larger windows. A global 7x7 average over 1024 channels, a
MobileNet's pooling layer, measures exact anyway, and matches TFLite's `AveragePool`
exactly there as well: an odd window has no tie, so half-to-even and TFLite's half-up
cannot differ, and the int8 rebase is exact because `128*49` is an integer multiple of the
divisor. [HW sweep, H96 MAX M9]

**PPU_RDMA's source surface stride is honoured verbatim, including a value that is not a
multiple of four.** Every vendor pooling program stores `round4(iw*ih) * 16` in
`R76_PPUR_SRC_SURF`, and a direct convolution's output surface stride is `ow*oh` exactly,
49 against 52 at a 7x7 plane, so whether the PPU takes the unpadded one is the whole of
whether a pool can read a convolution's surface as its feature cube without a copy. It
does: bit-exact at 49, 25 and 9 elements per channel group, over max and average, at zero
points either side of zero (`rk3576_pool_probe lib`, both strides,
`ROCKET_RK3576_POOL_PACK_SRC=1` forcing the packed one). A wrong stride there would compute
a full and plausible surface rather than faulting, which is why it is a gate rather than an
assumption. `pool_params_rk3576_t.src_surf_elems` carries it; 0 keeps the vendor's `round4`.
[HW sweep, H96 MAX M9, 2026-07-31]

**A pooling job raises no DPU completion, and the driver has to be told.** The same
per-block bitmap that makes `PC_OPERATION_ENABLE` a trap makes this one: a pool enables PPU
and PPU_RDMA and no DPU stage at all, so the `DPU_0`/`DPU_1` bits
`patches/rk3576/npu/0012` retires a job on can never set for it and the job falls through
to the `dpu_grace_us` deadline, paid in full on every submit. Measured through the library
entry, the wait tracks the parameter count for count, 641 us at 500, 389 at 250, 213 at
100, which is what says it is not retiring on a completion at all. The PPU's own bits are
two positions over in the same register (`PPU_0` `0x400`, `PPU_1` `0x800`, against the DPU's
`0x100`/`0x200`); `patches/rk3576/npu/0021` adds `DRM_ROCKET_JOB_PPU_DONE` to select them
per job, and the wait becomes **71 us**. Per job rather than per driver because which class
a job is in is invisible at `PC_DONE` time and obvious to the userspace that wrote
`PC_OPERATION_ENABLE`. **Gate the flag on the interface version (>= 1.3)**: the submit ioctl
rejects a flag word it does not recognize, so an older kernel fails the submit rather than
ignoring the bit. And do not set it on a stream that mixes DPU and PPU programs, an
interior program's PPU bit would retire the job while a later DPU write was still draining.
[HW sweep, H96 MAX M9, 2026-07-31]

### The unprivileged double free, and the crashes it was mistaken for

**`rocket_copy_tasks()` frees the submit job's task array twice.** It allocates
`rjob->tasks`, and on any failure inside its copy loop takes a `fail:` path that
`kvfree()`s it and returns, **without clearing the pointer**. Its only caller,
`rocket_ioctl_submit_job()`, then unwinds through `rocket_job_put()` ->
`rocket_job_cleanup()`, which frees `job->tasks` unconditionally. The same allocation
goes back to the allocator twice. Mainline v7.1, unchanged by this series, and the
RK3588 reaches the same path. [source-confirmed: `drivers/accel/rocket/rocket_job.c` at
v7.1]

Two failures reach it, and both are ordinary userspace input:

| failure | how a client gets there |
|---|---|
| `copy_struct_from_user()` fails | a bad task pointer, or a trailing field this kernel does not know (`-E2BIG`) |
| `task.regcmd_count == 0` | a task that asks for no registers |

The second is the one that matters in practice. **A generator that refuses leaves its
output parameters untouched**, so a caller that submits anyway hands the kernel a task
with `regcmd_count` still at its zeroed value, the exact shape this library's own
deadline canary once submitted.

`/dev/accel/accel0` is group `render`, so this is an unprivileged double free of a
**kmalloc-16** object (a `struct rocket_task` is 16 bytes), one of the hottest caches in
the kernel. The freelist then hands one object to two owners, and **the damage surfaces
nowhere near here**.

**Two faults were chased for a session and a half as if they were defects in their own
subsystems, and neither is.** Both are this corruption landing downstream:

```
dma_direct_unmap_sg <- dma_unmap_sg_attrs <- drm_gem_shmem_release
  <- drm_gem_shmem_free <- rocket_gem_bo_free        (GEM_CLOSE, file release,
                                                      and the drm_sched free worker)

__pi_memcpy_generic <- swiotlb_bounce <- swiotlb_tbl_map_single <- swiotlb_map
  <- dma_direct_map_sg <- dma_map_sgtable
  <- drm_gem_shmem_get_pages_sgt <- rocket_ioctl_create_bo
```

The first walks a scatter-gather list whose entries are garbage; the second bounces
through swiotlb only because a corrupt `sg_phys()` exceeds the DMA mask, and then
`phys_to_virt()`s a wild address. **A corrupt sg table on the free side and a corrupt
one on the map side are the same event seen twice**, and the free-side reading, "an sgt
torn down while still published", was wrong. Instrumenting `rocket_gem_bo_free()`
caught two live BOs naming one `sg_table`, which is not a lifetime race at all but two
owners of one slab object.

**`slub_debug=FZPU slab_nomerge` on the kernel command line settles it in one run**, and
it is deterministic: **six "BUG kmalloc-16: Object already free" reports over three runs
of `uapi_submit_errpath_rocket`, and zero on the fixed module**, with
`rocket_ioctl_submit()` named as the allocation site and `rocket_job_cleanup()` as the
second free. `patches/rk3576/npu/0020` is the fix, give the array a single owner and
delete the `fail:` free, since `rocket_job_cleanup()` already runs on every path out of
the ioctl. [HW sweep, H96 MAX M9, 2026-07-29]

**It is mainline, and the RK3588 reaches it identically.** Reproduced on a Turing RK1 at
7.1.1 with the same instrument: **six "Object already free" reports over three runs of
`uapi_submit_errpath_rocket` on a module carrying `patches/rocket/081`-`086`, and zero
over three runs on the same module plus `087`**, two per run on both parts, the same
allocation and second-free sites, and 83 of 83 `ctest` entries passing on the fixed module
under `slub_debug` with the taint word unmoved. `087` is `0020`'s one-line deletion against
the RK3588 series. [HW sweep, Turing RK1, 2026-07-29]

**A board whose image ships no kernel headers can still take an A/B.** The RK1's
`linux-image` deb carries no build tree, so there is nothing for `make -C
/lib/modules/$(uname -r)/build` to use, but `CONFIG_MODVERSIONS=y` is the only thing a
`modules_prepare` tree lacks, and the CRCs it needs are already on the board: a module
built by that kernel carries them in its own `__versions` section, 64-byte entries of an
8-byte CRC and a 56-byte name. Read them back off the shipped `rocket.ko`, write them out
as a `Module.symvers` (the owning module per symbol comes from `/proc/kallsyms`, which an
unprivileged reader gets with the addresses zeroed and the `[module]` tags intact), and an
out-of-tree build against a stock kernel.org tarball at the same version loads with a
matching vermagic. `tools/mksymvers.py` is the script; the tree needs the running
`/boot/config-*` and a `localversion` file carrying the Debian suffix so `UTS_RELEASE`
comes out identical.

**The gate that triggers it exits 0.** `uapi_submit_errpath_rocket` asserts that five
rejection sites return an errno, and they do; corrupting the slab on the way is not
something an exit status can see. That is why every gate ran green over it for two
sessions, and it is the general point: **a test that drives rejection paths is a test
that can be corrupting the kernel while passing.** Read the kernel log, and run the
rejection tests under `slub_debug` at least once.

**A reproducer that never reproduces is evidence about the reproducer.** 300 iterations
of `uapi_bo_lifetime_rocket` and eighteen rounds of the convolution, matmul and pooling
gates, with core 1 bound and unbound, never fired, because **not one of them submits a
job the kernel rejects**. The crash tracked the presence of `uapi_submit_errpath_rocket`
in the sequence, not the volume of BO churn: it fired in round 1 on both runs that
included it, and in no round of the two that did not. The earlier "roughly once per few
minutes of heavy BO churn, and it does not care which gate is running" reading had the
timing right and the cause backwards, the corruption is planted by one gate and
collected by whichever runs next.

**A second, smaller mainline defect came out of the same reading**, and it is a leak
rather than a crash. `rocket_job_open()` allocates the `drm_gpu_scheduler` array and
`rocket_job_close()` frees `entity->sched_list`, but `drm_sched_entity_init()` stores
that pointer only for `num_sched_list > 1` and `drm_sched_entity_select_rq()` clears it
again once the entity has settled on a runqueue. On a single-core device, which is the
supported RK3576 configuration, the close is `kfree(NULL)` and the array leaks on every
open/close, unprivileged and unbounded. `patches/rk3576/npu/0019` keeps the driver's own
pointer, frees it after `drm_sched_entity_destroy()` rather than before, checks the
allocation (it was unchecked) and frees it on the init-failure path too.
[source-confirmed + HW sweep, H96 MAX M9]

## The DPU's elementwise stage: one operand, requantized

An elementwise op on this part is a program of its own, **DPU + DPU_RDMA only, no CNA
and no CORE, 89 writes, `PC_OPERATION_ENABLE` `0x18`**, the same shape as the LUT table
load, but this one computes. It is **not** fused into the producing convolution's
epilogue: in the vendor's hands a residual block pays a program for its add. (The
lowering this driver ships does not; see below.)

What it computes, measured on silicon:

```
out = sat8( ((ew + EW_CVT_OFFSET) * EW_CVT_SCALE >> EW_CVT_SHIFT)
                                  * OUT_CVT_SCALE >> OUT_CVT_SHIFT )
```

**One operand.** Bit-exact over channel counts 16-320 and planes 1x1 to 28x28, in the
NC1HWC2 cube the convolution path already packs (`tests/rk3576_add_probe.c gate`, 10
shapes). The final shift **rounds half to even**, the same rule the direct path's
OUT_CVT uses, established here independently: against a round-half-up reference every
disagreement was an exact tie, 25% of elements at a gain of one half and none at unity.
[HW sweep, H96 MAX M9, 2026-07-31]

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

`MUL` is the same program in four registers (`0x407C` `0x810F4094`, `0x4044` `0x2`,
`0x4050` `0x00020000`, `0x501C` `0x2`); its arithmetic is not gated.

### A second operand is not reachable through this register set, and the search is exhaustive

The elementwise stage takes **exactly one operand**, and that is now a closed negative
rather than a lead not yet chased. Six sweeps cover the whole interface:

- **Every register the program leaves at zero, tried as a second base at every output
  gain.** Seventeen candidates crossed with the full 32-rung OUT_CVT ladder. Nothing
  reaches the output at any of them except `0x5024`, which injects a constant; it is
  the DPU shift word, a live per-task operand and not a spare. Crossing the two axes is
  what makes this conclusive: a placement sweep at one gain and a gain sweep at one
  placement are both blind to an operand that sits somewhere unexpected and enters the
  accumulator unscaled.
- **Every register the program never writes at all**, appended to the program one at a
  time, the complement of every other sweep here, and not an empty set, since the
  register file is not cleared between jobs on this part, so a base the vendor programs
  in an earlier task would still be standing. Twenty-nine candidates over the DPU and
  DPU_RDMA blocks, at two gains: nothing carries an operand, and three (`0x5068`,
  `0x5070`, `0x5074`) stop the write entirely.
- **The main DMA feed at every gain from 2^14 down to 2^-31.** The program is configured
  to read one, DPU `0x400C` bit 0 is the NVDLA feature-mode flying bit and the add sets
  it, DPU_RDMA `0x5044` bit 4 is `MRDMA_DISABLE` in the same lineage's map and the add
  clears it, and it contributes exactly zero at all 64 rungs, with the elementwise
  operand live as a control at every one.
- **A joint grid over `EW_CFG` and `BRDMA_CFG`**, and a second over `FEATURE_MODE` and
  `ERDMA_CFG` driving the `COMB_USE` field the RK3588's own K-accumulation uses to
  combine two feeds. In both, **only the captured word writes at all**: every other
  combination leaves the surface untouched.
- **Destination accumulation, with DPU `0x40C0` swept** over the captured word and 28
  single-bit variants of it, classified by running each twice over differently
  pre-filled destinations. It always overwrites. Two bits (`0x00010000`, `0x00020000`)
  saturate the output; none accumulates. `SURFACE_ADD` in the RK3588's map sits at this
  offset, and on this part it is not an accumulate mode.
- **What the program reads, rather than what it was handed.** A single non-zero 16-byte
  atom walked over an operand buffer eight cubes long, with the base pointed at the
  middle: the addressed cube's 32 atoms map one to one onto the output and **nothing
  outside it moves anything**. So the two operands are not one allocation at a fixed
  offset either, which is how the RK3588's K-accumulation feeds its pair.

What the manufactured captures say the vendor's program does have, which is what makes
the negative worth stating precisely: **`Sub` compiles to this same program with the
operand converter's scale negated, and `Sub` with its operands swapped negates it too**,
so the subtrahend is always the elementwise cube and the other operand rides a feed
whose weight is fixed at +1, since no register carries a second scale. And **both
operands share one quantization scale**: two graph inputs calibrated 64x apart still
compile to one converter, with the OUT_CVT gain coming out as the ratio of a single
shared input scale to the output's.

### The residual add is lowered onto the convolution datapath instead

Concatenate the two operands along the channel axis and convolve with a 1x1 kernel of
two diagonal blocks, `W[o][o] = w1`, `W[o][C+o] = w2`, zero elsewhere:

```
out[o] = requant( w1 * (a[o] - a_zp) + w2 * (b[o] - b_zp) )
```

Bit-exact against a CPU model of the part's own arithmetic over nine MobileNetV2 and
ResNet-18 residual shapes (channels 24-512, planes 7x7 to 56x56) and within **one count**
of an exact float residual add at every one (`tests/rk3576_residual_add.c`). Two of its
properties are better than the vendor's elementwise program, not merely equal to it:

- **The two operands may carry different scales.** The ratio rides in the weights as
  `w2/w1`, so the pair only has to be representable as two int8s, about one part in 127,
  where the vendor's one operand converter forces its compiler to quantize both
  operands to a common scale.
- **The two zero points ride in the bias, exactly.** The datapath has one input zero
  point, but `w2 * (a_zp - b_zp)` is a per-output-channel constant, which is what the
  bias is.

And **it fuses**, measured rather than argued. A block's last convolution takes `C` more
input channels and an identity block at the **centre tap** of its kernel, and the skip is
then part of a convolution the network was already paying for, so the add costs **no
program at all**, against the one program the vendor pays. That is the hardware's own
idiom: the vendor compiler folds `Add(Conv(x), x)` into exactly this shape, which is why
that graph is useless as a capture of an add. Bit-exact at MobileNetV2's project
convolution for every residual width (1x1, `ic` 168 to 1120) and at ResNet-18's second
convolution (3x3, `ic` 128 to 512).

**Where the fusion stops is the weight-slice rule**, `ic*kh*kw <= 4608`. A 1x1 project
convolution is nowhere near it. A 3x3 reaches it exactly at `C = 256` (`512*9 = 4608`) and
is refused at `C = 512`, so ResNet-18's widest stage keeps its add as a standalone
program, which the same rule caps at `C = 2304`. The standalone form's weight cube is
`2C*C` bytes, 512 KiB at that stage, paid once as resident weights rather than per
inference; transiently a standalone add measures 0.6-3.3 ms across the shape table.
[HW sweep, H96 MAX M9, 2026-07-31]

### Two traps, each of which cost a round of the probe

**Every base left at the capture's stored zero reads IOVA 0, which is a real buffer**,
per-fd IOVA starts at zero on this stack, so the first BO a probe allocates is what
those bases read. With the operands allocated first the surface came back a constant
and the arithmetic looked broken; allocating a **guard BO first** so nothing of ours
sits at IOVA 0 made the operand pass through exactly. A zero base is not a disabled one.

**DPU `0x40D0` must be the captured `0x0040FFFF` verbatim.** Reading it as a clamp pair
and writing `0x00407F80` left **exactly half of every 16-channel group unwritten**,
silently, a write-coverage failure, not a wrong value, and one that reads like a
channel-budget property of the writer. The clamps are `OUT_CLAMP_MIN`/`MAX`
(`0x40A4`/`0x40A8`); the no-clamp pair is `INT32_MIN`/`INT32_MAX`.

`OUT_CVT_SCALE` is a **signed** 16-bit field: `32768` reads back as `-32768` and flips
the output's sign, so the usable maximum is 32767.

### Manufacturing the capture: what the compiler folds away

The captures are built by `tests/data/rk3576-vendor-capture/add/mkadd.py` and read by
`decode_add.py`. **`Add(Conv(x), x)` is not a capture of an add**: the vendor compiler
folds an identity skip into the convolution's own kernel, the centre tap of the
diagonal, so six such ONNX graphs, over four channel counts and three planes, compiled
to a register program indistinguishable from a plain convolution. `Add(Conv_a(x),
Conv_b(x))` folds too, and so does a **per-channel broadcast** operand, `Add(Conv(x),
k)` with `k` of shape `[1, C, 1, 1]` compiles to a plain convolution with `k` in its
bias. A capture of the op needs an operand the compiler cannot reach: a second graph
input, a constant tensor of the same shape, or a real MobileNetV2 bottleneck whose skip
crosses three convolutions. Those three all emit it, identically.

**Two operands at the same scale are unattributable.** Every register that scales an
operand looks the same for both when both calibrate to the same range, so the two
converters cannot be assigned. Give the inputs deliberately asymmetric calibration
amplitudes and the assignment falls out; that, and compiling `Sub` in both operand
orders, is what named the elementwise cube as the *second* operand and pinned the
first's weight at unity.

**A capture's container says how many tasks an op costs.** Each `.rknn` carries a table
of 40-byte task records, `[index][PC_OPERATION_ENABLE][slot size][int mask][int clear]
[write count][…][program offset]`, and `write count` matched against the program stream
is what identifies them. It is worth reading before concluding anything about an op's
structure from the program list alone: a single `Conv` emits four task records over four
distinct program slots at two different row geometries, so program count is not op count.
