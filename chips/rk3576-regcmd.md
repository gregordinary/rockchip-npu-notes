# RK3576 — the int8 conv register encoding, as an emitter

The RK3576 NPU runs the same IP family as the RK3588 through the same mainline
`rocket` driver, but its CNA/CORE/DPU blocks use a different geometry-register
encoding at the same block bases. This sheet is the encoding written out as
something you can emit, plus what running it on real silicon settles and what it
does not. The SoC-level parameter sheet (identity, integration, clocks, power) is
[rk3576.md](rk3576.md); this is the register layer.

Provenance: RKNN-Toolkit2 register programs for known conv geometries, from two
capture sets under
[../../rocket-userspace/tests/data/rk3576-vendor-capture/](../../rocket-userspace/tests/data/rk3576-vendor-capture/).

FOUND captures, from Ga Hing Woo's bring-up repo — real convolutions, including a
MobileNet-shaped stem:

| Capture | Geometry | What only it can show |
|---|---|---|
| `conv2d_rk3576.rknn` | ic 16, oc 128, k5 s2, 80x80 -> 40x40 | the normal `in_ch>4` path at k != 3; a single task over the whole plane |
| `dw_rk3576.rknn` | depthwise c32, k3 s1, 112x112 | the depthwise words, and a task windowed to 91 input rows |
| `conv64_rk3576.rknn` / `conv0_rk3576.rknn` | ic 3, oc 32, k3 s2 | the first-conv ARGB sub-encoding, at two image sizes |
| `iso_bias` / `iso_scale` / `iso_sum` | the `conv2d` geometry plus one appended op | what a DPU epilogue stage moves |

MANUFACTURED captures, in `dw/` and `dw/named/`. **A vendor capture can be built to
order**, and that is the single most useful thing to know about this part: an ONNX
compiled for `rk3576` emits the register program for whatever geometry is asked for,
so a question that a bit sweep answers slowly and ambiguously can be answered by a
diff. `group=C` gives the depthwise path at any channel count, kernel, stride and
plane; `do_quantization=False` on a float ONNX gives the float datapath. `dw/mkdw.py`
and `dw/named/mknamed.py` rebuild both sets and carry the dependency pins.

The found depthwise captures confound almost everything: every C=32 program in them
is stride 1 and every C=64 one is stride 2, every one is k=3, every one is a multiple
of 32 channels, and every one is a SQUARE plane. Manufactured ones at 22 channel
counts from 8 to 256, both strides, kernels 1/3/5/7 and rectangular planes separate
all of it — see "Depthwise" below, where four register formulas that had been fitted
or guessed are now transcribed.

The emitter built from them is
[npu_regcmd_rk3576.c](../../rocket-userspace/src/npu_regcmd_rk3576.c); the gate
that diffs it against the captures register-for-register is
[regcmd_rk3576_gate.c](../../rocket-userspace/tests/regcmd_rk3576_gate.c). It
reproduces every checked register of every captured program — 110 conv programs,
88 of them depthwise, with no open field left.

Register fidelity is not correctness, and depthwise is where the two part company.
An emitter matching the vendor register for register across 88 depthwise programs
still computed nothing, because what it was wrong about were the BUFFERS the
registers point at — a capture carries a register program and says nothing about the
memory it addresses. Both of that path's buffer layouts had to be read off the part
(see "Depthwise" below), and that is the general lesson rather than a depthwise one.
The correctness gate is
[rk3576_conv_gate.c](../../rocket-userspace/tests/rk3576_conv_gate.c) — a shape table
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
ARGB path. `0x1070` reads zero in every capture — including the one program that
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
the second form, but they part company as soon as `ic/2` exceeds `iw/4` — and the
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
its right pad is 1 (`0x01000101`). The previously published rule — a lookup
keyed on stride and depthwise — predicts `0x00000101` for `conv2d` and is wrong.

## Registers the map left unnamed, and their RK3588 equivalents

Four of the offsets recorded as "RK3576-only, no RK3588 counterpart" are the same
registers the RK3588 has, moved:

| RK3576 | Function | RK3588 |
|---|---|---|
| `0x1048` | CVT control word — `data_sign<<3 \| cvt_type<<1 \| cvt_bypass`, identical packing | `CNA_CVT_CON0` `0x104C` |
| `0x104C` / `0x1050` | the four CVT scales, **two per register** (`scale1<<16 \| scale0`, `scale3<<16 \| scale2`) | `CNA_CVT_CON1..4`, one each |
| `0x1054`-`0x105C` | the CVT offsets (non-zero only on the ARGB path, where they carry the per-channel -128) | folded into the same CON1..4 |
| `0x108C` | burst lengths — `weight_burst<<16 \| data_burst`, `0x000F000F` | `CNA_DMA_CON0` `0x1078` |
| `0x103C` hi, `0x1044` lo | CBUF data entries, `ceil(iw*ic/64)` — written twice | `CNA_CBUF_CON1` `0x1044` |
| `0x1084` | the CNA border pad constant | `CNA_PAD_CON1` `0x1184` |
| `0x5024` | base of the DPU shift word — a per-task operand, not a spare (see below) | the register fields `BS_MUL_CFG.BS_MUL_SHIFT_VALUE` `0x4048` and `DATA_FORMAT.BS_MUL_SHIFT_VALUE_NEG` `0x4010` |

The identification is by value, not by analogy: the int8 conv's CVT word is
`0x0b` in both encodings, and the burst word is `0x000F000F` in both, for the
same reason.

**The kernel word has a closed form.** `0x1024` hi is `((kh-1)<<8) | (kw-1)` —
`0x0202` at k=3, `0x0404` at k=5 (the `conv2d` capture is the only one that shows
a second k), `0x0000` at k=1. Published as a two-entry lookup; it is one
expression.

**The OUT_CVT triple is `0x40AC` / `0x40B0` / `0x40B4`** — offset, scale, shift,
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
- **The ARGB weight cube's byte layout.** The register program is transcribed and
  gated against all twelve captures; the cube those registers point at is not, and
  it is the whole of what a running first conv still needs. See the ARGB section.
  Reach for a manufactured capture with NAMED weights before sweeping: a 3-channel
  float ONNX makes the compiler pick this sub-encoding (check that it did — `0x100C`
  carries `R76_ARGB_CONV_MODE`), and that is how the depthwise cube was decoded.
- **What the DPU epilogue delta actually enables.** Three registers, identical for
  a bias, a scale and an eltwise sum, so they do not encode which. One sweep run
  settles it; the recipe is in the epilogue section.
- **`0x1060` is probably `CVT_OFFSET3`.** `0x1054`-`0x105C` are the first three CVT
  offsets and there are four CVT scales, so the fourth offset almost certainly sits
  here. Every capture leaves it zero, including the ARGB ones — which use only three
  channels — so nothing pins it, and it stays in the address-placement shotgun list
  rather than being claimed.
- **`0x1064`.** The one genuinely live vendor dump (an `rknpu` kernel capture
  taken at submit, as opposed to the ten programs decoded out of `.rknn` files)
  carries `0x777` here where every stored program carries 0 — so the vendor
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
the RK3588 register program this part writes nothing at all — the output BO comes
back untouched. With the RK3576 program the DPU writes the full, correctly-sized
surface: 2048 bytes for a 32-channel 8x8 int8 output, in the same NC1HWC2 cube
the RK3588 uses (C2=16, surface stride `ow*oh` in 16-byte atoms, which is why
`0x401c` holds `ow*oh` and not `ow*oh*16`).

**The part computes convolutions, and the emitter is what makes it do so.** A
k=3 SAME 32x32 int8 conv, IC=OC=32, driven with a dense random weight set (all
9216 weights non-zero) over a feature tensor spanning the full signed int8 range
reproduces the CPU model **bit-exactly across the whole surface — 32768/32768,
max |diff| 0** — and the same conv unpadded is 28800/28800. That single result
carries most of the encoding: the CNA feature and weight DMA, the CBUF staging,
the CSC, the CMAC, the BS bias add, the OUT_CVT requant, the output geometry and
the writer are all being programmed correctly. It also settles the weight cube
independently — a dense random set of 9216 weights cannot match bit-exactly under
a permuted layout — so the RK3576 takes the **same weight cube as the RK3588**
(`weight_conv_int8`, oc-group 32 / ic-group 32) at IC=OC=32.

**The envelope is every geometry whose channel counts are programmed as multiples
of 32 and whose plane fits the CBUF budget.** Across IC and OC from 8 to 128,
planes from 8x8 to 128x64, k1/k3/k5, stride 1 and 2, VALID and SAME, against
feature tensors that vary on all three axes, the emitter reproduces the CPU model
bit-exactly — the whole surface, every time, repeatable across separate power
sessions. Both boundaries below are properties of how the operands are described,
not of the datapath, and both are now closed by construction: the channel counts
come from `rocket_rk3576_pad_ic()` / `_pad_oc()`, and the CBUF budget is the
`rocket_hw_rk3576` machine-parameter profile.

## Both channel counts must be programmed as multiples of 32

The register `ic` and `oc` must each be a multiple of 32 — the group
`weight_conv_int8` pads its two channel axes to — with the feature cube, the output
BO and the coefficient buffer sized to the padded counts. Convolutions whose
channel counts are not multiples of 32 compute wrong otherwise, at every geometry.
No capture shows this, because every captured geometry already satisfies it.
[HW sweep, H96 MAX M9]

On the input axis, ic of 8, 16, 17 and 48 are wrong at every kernel and plane size
while 32, 64 and 96 are exact. ic=17 fails despite spanning two whole C2=16
surfaces, so the unit is the 32-channel MAC group and not the surface. Padding the
feature cube alone does not help — every read past `ic` already lands in mapped
zeros — so it is the register value that matters, and passing the padded count with
the cube zero-filled to match is bit-exact.

On the output axis a partial group trips **two different mechanisms**, which is why
it presents two ways and why a sweep at one kernel size reads as "oc is
unconstrained":

- At **k=1** the DPU writes only output **row 0** of the trailing group and leaves
  the rest of the surface untouched. oc 8, 16, 40 and 48 fail this way; 24, 32 and
  64 are exact. The rule that fits k=1 alone is surface parity — `ceil(oc/16)` must
  be even — which is why oc=24 passes there.
- At **k>1** the weights come out wrong instead, and oc=24 and oc=56 fail even
  though their surface count is even. `WEIGHT_BYTES` (`0x101C`) is `ic*oc*kh*kw`,
  which describes a cube tighter than the padded one the caller supplies, and in
  the `[kh][kw][oc2=32][ic2=32]` cube that truncation drops whole `(kh,kw)` planes
  rather than trimming `oc` inside each. At k=1 there is only one such plane, so
  the same shortfall lands harmlessly on the unused `oc2` slots — which is exactly
  why k=1 is the tolerant case.

Rounding `oc` up to 32 satisfies both. The trap to recognise is the first
mechanism: a partially-written surface reads as an output-geometry defect, not a
channel one.

The vendor does not pad. Its `conv2d` capture is ic=16 with weight bytes
`ic*oc*kh*kw` = the tight size, so it packs a 16-channel weight group where
`weight_conv_int8` pads to 32 — and our emitter matches that program 131/131, so
the difference is in the cube, not the registers. Padding is how the RK3588 cube
expresses the same conv; a partial-group weight packing would be the other way, at
half the weight bytes for such a layer.

## The CBUF budget is programmable, and `0x1040` is what programs it

A conv whose feature plane does not fit computes wrong with the DPU still writing a
full surface, no IOMMU fault and nothing in dmesg. The corruption is graded: one row
past the budget about half the surface is still bit-exact, and it degrades with the
depth of the overflow until almost nothing is (at ic=32, iw=16: 4096 granules is
exact, 4104 gives 132001/262656, 4160 gives 1090/266240). Nothing else on this part
presents that way, which makes it the instrument — grow a known-good conv until it
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
| 1280 (256\|1024) | 5152 | **no** — 5376 expected |
| 1536 (512\|1024) | 5152 | **no** — 5632 expected |

Widths of 16, 32 and 64 all break at the same granule totals — 4096 at F=0, 6144 at
F=2048 — so the limit is the granule count and not rows, width or channels
individually. The data budget does not exceed **6144 granules (384 KiB)**: F=3056
still measures 6144. [HW sweep, H96 MAX M9]

**Only single-bit F values deliver their face value.** Each power of two measures
exactly 4096+F, but a combination delivers less than the sum of its bits, and the two
combinations tested both land on 5152 regardless of which second bit was set. So the
field is not simply an integer the CNA adds, and an emitter that computes an arbitrary
F programs a budget the hardware does not honour — which corrupts silently, since a
plane over its allowance still writes a full surface. Use the measured rungs
(0, 256, 512, 1024, 2048) and round a deficit up to one. The cost is a little weight
headroom the plane did not need. [HW sweep, H96 MAX M9]

The two values the captures carry are therefore two points on that scale, not a
constant and a variant: `0x10000000` (F=0) buys 4096 granules and `0x14000000`
(F=1024) buys 5120. That is what the vendor's windowed depthwise program needs —
112 wide at ic=32 is 56 granules per row, and its 91-row window is 5096 granules,
which overflows 4096 and fits 5120 with 24 to spare.



**The low bits are the RK3588 field layout, and the vendor leaves them zero.** Bits
0-13 — `DATA_BANK[0:3]`, `WEIGHT_BANK[4:7]`, `FC_DATA_BANK[8:10]`, `DATA_REUSE[12]`,
`WEIGHT_REUSE[13]` per Mesa's `registers.xml` — are live: setting any single one of
them on top of a working program corrupts the conv. So the bank *fields* did not
move in the re-pack; what moved is that this part also takes a granule allowance in
bits[16:27], which is RESERVED on the RK3588. Bit 28 is required (a `0x1040` of zero
corrupts), bit 29 corrupts, and bits 14, 15, 30 and 31 are don't-care. [HW sweep]

**The data and weight sides share one pool, and F trades between them.** The weight
path stages per output-channel group, so its resident slice is `32 * ic * kh * kw`
bytes rather than the whole cube — which is why a 441 KiB weight cube computes fine.
Sizing that slice against F shows the trade directly: a 150 KiB slice is bit-exact at
F=0 and breaks at F=2048, with the ceiling falling as the data side grows.

| F | data allowance | weight slice ceiling | sum |
|---|---|---|---|
| 0 | 4096 gr = 256 KiB | 175 KiB exact, 200 KiB breaks | ~445 KiB |
| 1024 | 5120 gr = 320 KiB | 125 KiB exact, 150 KiB breaks | ~455 KiB |
| 2048 | 6144 gr = 384 KiB | below 75 KiB | ~450 KiB |

Each +1024 granules of data costs the weight path about the 64 KiB the data side
gained, and all three pairs sum to **roughly 448 KiB** — 14 banks of the RK3588's
32 KiB, of which the captures' default program takes 8 for data. The bank *count* is
an inference from that bank size; what is measured is the granule budget, the trade,
and the total. Reading the pool as 14 banks also explains the wedge: F at or past
~3060 leaves the weight path nothing, and the part then writes no output at all at
any plane size, which looks exactly like a wrong geometry encoder. [HW sweep]

Two consequences. `rocket_hw_rk3576`'s 8 x 32 KiB is the **default data allocation**,
not the physical CBUF — raising F buys up to 1.5x the feature capacity, at the cost of
weight-slice headroom that must then be respected. And a register the published map
records as a constant is a tuning knob, so an emitter that copies the constant
inherits the vendor's choice for a full-plane task rather than making its own.

The boundary is sharp enough to plan against, and the pool figure survives a direct
test of it: a slice of exactly 192 KiB (6 banks, what the model leaves beside F=0)
computes bit-exactly at ic=1536 k=2, and 196 KiB at ic=1568 breaks. [HW sweep]

**The emitter plans F per task**, in `rocket_rk3576_cbuf_f()` — the lowest rung whose
budget covers the plane, refusing the task when the plane needs more than the data cap
or when the rung would starve the weight path, because the recourse (a shorter row
window, an ic split) is the caller's to choose. `rocket_rk3576_max_task_rows()` is the
tiler-facing half: the tallest window one task can carry at the highest rung the weight
slice leaves room for. Planning reproduces both captured words, so the register-fidelity
gate stays byte-identical, and it turns every plane between 4096 and 6144 granules from
silent corruption into an exact result — validated on the part at 4160, 4800, 5600 and
6144 granules, and at 6144 across three widths. `ROCKET_RK3576_CBUF_F` forces F and
bypasses both checks, which is how the allowance was characterised.

## The weight slice caps how many output-channel groups compute

The pool arithmetic above is characterised at ONE output-channel group, and it is not
sufficient on its own. A conv driving several groups loses the trailing ones well
before the slice reaches what the pool leaves it, and it loses them **one at a time**
as the slice grows: the leading groups come back bit-exact and the rest wrong, with a
full surface written, no fault and nothing in dmesg. It reads as an output-channel
defect rather than a capacity one, which is why it was first recorded as a limit
"that is not slice size".

It is the slice. The governing quantity is the resident weight slice
`32 * ic * kh * kw` — one output-channel group — and the same slice behaves
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
alone — it already lands on the measured single-group boundary (175 KiB computes,
200 KiB does not, and the pool reaches its limit at 192 KiB), and a graded
multi-group loss is by definition not a single-group effect.

What the shape of the loss suggests, and what the points do not settle: the boundary
falls as the group count rises, roughly as though each additional group costs a small
fixed staging allowance on top of the slice, which fits a weight path that stages the
next group while computing the current one. Two free parameters against six points is
not a mechanism, so the emitter carries the measured table rather than a formula.
[HW sweep, H96 MAX M9, measured 2026-07-25]

### The group count is a caller's choice, so the cap is one too

The table is read above as a bound on a shape. It is not — it is a bound on a
PROGRAM, and the group count in it is what one program drives rather than what the
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
recourse there is an input-channel split that the on-chip requant forecloses — int8
partials cannot be summed without quantizing each one. That shape belongs on the
int32-output writer.

Held by `tests/rk3576_conv_lib_gate.c`, which drives the emitter gate's own shape
table through the library entries: 111 pass, the one refusal is asserted in both
directions, and every computed shape is bit-exact against a CPU model.
[HW sweep, H96 MAX M9, measured 2026-07-27]

The cube size is not the constraint, which the same sweep confirms: `ic=448 k=3
oc=128` has a 126 KiB slice under a 504 KiB cube and is exact on all four groups,
while `ic=256 k=5 oc=32` has a 200 KiB slice under a 200 KiB cube and does not fit at
all. Neither is group count on its own — `oc=256` at `ic=64` is eight groups and
exact.

**Everything downstream of the MAC works, and the bias path is bit-exact.** A
bias-only probe — features and weights zero, a per-channel bias — reproduces the
CPU model exactly across all 32 output channels (2048/2048 bytes, max |diff| 0),
and the OUT_CVT sweep moves it exactly as offset/scale/shift should. Getting
there required the coefficient-buffer layout below.

**The OUT_CVT requant rounds to nearest; it does not truncate.** A border whose
exact value is -10.0006 comes back as -10, where an arithmetic shift gives -11.
Adding half an LSB before the shift — `(acc*scale + (1<<(shift-1))) >> shift` —
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
was based on — but a session that ends with these faults can leave the part in a
state where the weight loader never arms again: every later job completes cleanly,
with no timeout and no new dmesg line, and writes a bias-only surface. It survived
7 hours of idle and a full `rmmod rocket` / `modprobe rocket` cycle, and only a
reboot cleared it. Budget a reboot after any unmapped-IOVA probing, and re-run a
known-good conv before trusting a negative result taken afterwards. [HW, H96]

**The feature strides are confirmed one axis at a time.** A uniform feature fill
proves nothing about addressing — every read that lands inside the buffer returns
the same byte — so each axis was varied with the other two held flat. Varying
along rows, along columns, and across channel *groups* each reproduces the CPU
model bit-exactly, which confirms in turn the line stride (`0x1090` = `iw*4`, in
4-byte units), the C2=16 channel atom inside a row, and the CBUF surface/group
stride (`0x1094`/`0x1098` = `iw*ih`, in 16-byte units). [HW sweep, H96]

**The bias/coefficient buffer at `BS_BASE_ADDR` (`0x5020`) is not a flat per-OC
int32 array.** It is the structure the vendor's own coefficient buffer uses:
**groups of 64 bytes covering 8 output channels each**, holding

| field | type | offset in group |
|---|---|---|
| `A[oc]` — per-channel bias term | int32 | `(oc%8)*4` |
| `B[oc]` — weight-zero-point correction | int16 | `32 + (oc%8)*2` |
| `C[oc]` — per-channel multiplier | int16 | `48 + (oc%8)*2` |

so `oc` lives at group `oc/8`. Handing the part a flat int32 array instead makes
it read that array *as* this structure: only the first 16 channels get a term at
all, on alternating channels, at 1024x the intended magnitude — an artifact, not
a 2-byte operand DMA.

**`C` gates the whole BS stage, not just the bias.** At `C=1` the datapath is
bit-exact and `C=4` scales by exactly 4, so it is a live linear per-channel
multiplier — but at `C=0` the DPU writes a full, correctly sized, entirely
**empty** surface no matter what the CNA and the MAC did, for a conv with no bias
at all. An all-zero coefficient buffer is what any caller who does not know about
this layout hands over, and the resulting empty surface is indistinguishable by
inspection from a wrong geometry encoder — it is the single most expensive trap on
this part. Pack the buffer with `rocket_rk3576_pack_coeff()` rather than by hand.
`B` is pinned to 0, which is what symmetric quantization wants; it is unvalidated
against a non-zero weight zero point, and nothing observed so far needs it.
[HW sweep, H96]

**How `C` is read depends on the precision, so `C = 1` is not portable across it.** The
integer multiplier above is the int8 program. On a float program the same field is read
as fp16, where the integer `1` is the denormal 6e-8 and underflows the surface to empty —
the identical signature, from arithmetic rather than from a gate. fp16 `1.0` is `0x3C00`.
See [the fp16 section](#fp16-the-arithmetic-computes-the-output-packing-does-not).

**The feature domain is signed int8 and needs no centering.** A k=1 conv whose
output plane sweeps every int8 value once reproduces the CPU model over the whole
range — one flat run from -128 to 127, 8192/8192, max |diff| 0 — with the probe
feeding raw signed bytes and no `+0x80` applied anywhere. The DPU epilogue is
exact for negative accumulators independently: a bias-only probe driven with
negative per-channel biases, which the MAC never touches, is bit-exact on all 32
channels. The earlier reading — that the datapath is uint8-centered and a feature
tensor carrying negatives is therefore wrong — does not survive; those runs were
made against an all-zero coefficient buffer, where the surface is empty whatever
the features are. [HW sweep, H96]

**The padded border is exact.** A k=3 SAME 32x32 conv with dense random weights
reproduces the CPU model over the whole surface, ring included — 32768/32768, max
|diff| 0 — as does the pad probe driving the border with a -128 pad tap against
zero features. `0x1084`, the pad word `0x1080` and the window geometry are all
right. The previously reported border defect was the same coefficient-buffer
artifact. [HW sweep, H96]

## The row window, and the three registers it closes

A plane over its allowance is not a slow conv, it is a wrong one, so the recourse the
allowance planner names has to exist. It is a split by INPUT ROWS: each task reads a
row window of the full plane and writes the output rows that window supports, with the
full plane still described alongside the window. `rocket_rk3576_plan_rows()` lays the
sequence out and the emitter takes the window in `conv_params_t` `ih`/`oh` against
`ih_full`/`oh_full`.

The caller's whole job per task is two byte offsets, and both are plain row strides —
`iy0*iw*16` into the feature cube and `oy0*ow*16` into the output. The cubes are
NC1HWC2 with a 16-byte channel atom and the CNA takes the DDR group stride from the
FULL plane (`0x1094` = `iw*ih_full`), so one base plus a row offset addresses that row
of EVERY channel group. The vendor's sliced capture is the direct evidence on the
feature side: its fourth task reads input rows 111.. and `0x1088` carries exactly that
row offset.

Driving it settles the three registers that only take a second value on a windowed
program. All three were measured with an `oc=64` conv — two output-channel groups, so
group 1 is exposed — cut into 32-, 16- and 8-row windows of the same 64-row plane.
[HW sweep, H96 MAX M9]

**`0x40B8` is the channel-group jump: `ow * (2*oh_full - oh_task)`, and that form is
now swept rather than fitted.** It was pinned at a single output-channel count, which
left open whether it held for a conv driving many groups at a wide kernel. It does:
the form is bit-exact across 2, 4, 8 and 16 output-channel groups, k1/k3/k5, stride 1
and 2, VALID and SAME, wide, tall and ragged planes, ic 32/64/128, and 3 to 16
windows — the `surface` group of the conv gate. The description below is what the
sweep confirms. [HW sweep, H96 MAX M9, measured 2026-07-25]

A full
destination surface PLUS the rows of it this task does not write. The writer walks the
task's rows and then adds this to reach the same rows of the next group, so a windowed
task has to be told about the rows it skipped or every group past the first lands
short — which is why an `oc=32` split is exact while `oc=64` is half wrong, and why
the defect reads as an output-channel fault rather than a windowing one. Only this
form is bit-exact at all three window sizes; `ow*oh_full`, `ow*oh_task`, `2*ow*oh_full`
and 0 each corrupt the whole surface. It reduces to `ow*oh_full` when the task is the
whole plane, which is what both full-plane captures carry — the register reads like a
constant until a window separates the two terms.

**`0x1018` carries one live bit, and it is not the byte pair.** Bit 30 is required:
clearing it corrupts the whole surface. The low 16 bits are don't-care on the direct
int8 path — `0x0404`, `0x0505`, `0x040b` and `0x0000` are all bit-exact, windowed and
un-windowed alike. An earlier reading had the windowed value stopping the DPU on an
un-windowed task; that was the cold-start wall, which is the trap this sheet warns
about — read a single "the DPU did not write" as the wall and re-run before believing
it.

**`0x1038` likewise carries one live bit, bit 31.** `0x07`, `0x010e` and `0x10` are all
bit-exact; `0x80000010` corrupts the surface.

The emitter still reproduces the vendor's values for all three, so the
register-fidelity gate stays byte-identical to the captures.

**What the split computes.** Bit-exact against the CPU model over the whole surface at
every shape tried: 112x112 and 224x224 at ic=32 (the geometries with no single-task
plan at all — 6272 and 25088 granules against a 6144 cap), and forced-cap splits from
2 to 16 windows across ic/oc of 32/64/128, k1/k3/k5, stride 1 and 2, VALID and SAME.
The planner spreads the output rows evenly over the fewest tasks that fit rather than
taking greedy maximum windows, because a greedy pass leaves a ragged tail — a 112-row
plane at a 109-row cap comes out 109+2+1 instead of two windows of 56 — and every extra
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
or 1.6 MB. The floor is the driver's **1 ms completion poll**, not the silicon — this
part has no maskable completion IRQ, so `rocket` polls `PC_DONE` on a 1 ms hrtimer, and
shortening that interval moves the per-task cost with it almost exactly (0.65 ms at 500
us, 0.42 at 200 us, 0.30 at 100 us). It is not a free lever: below ~200 us a multi-task
run raises a DMA-error interrupt and wedges the IOMMU, so completion is being declared
before the DPU's writes have retired. A real fix wants a completion signal rather than a
faster poll.

## `ih_full` is not optional on a single-task plan

`rocket_rk3576_plan_rows()` can return ONE task whose row window is shorter than the
plane, and a caller that sets `ih_full`/`oh_full` only when the plan splits then
mis-programs the DDR channel-group stride. This is ordinary geometry, not an edge
case: any stride greater than 1 whose output does not consume the plane leaves
trailing input rows unread, so a 32x32 k1 s2 VALID conv plans one task over 31 of its
32 rows. With `ih_full` left at 0 the emitter takes the group stride from the window,
`0x1094` comes out `iw*31` instead of `iw*32`, and every channel group past the first
reads at the wrong offset.

The symptom is a full, correctly sized surface bearing no relation to the input — no
fault, no dmesg line, and identical in appearance to a broken geometry encoder. It is
worth stating because the un-windowed path in `rk3576_first_light` sets the window
directly and never trips it, so the defect only appears once a caller starts routing
every conv through the planner, which is what a tiler does. Set `ih_full`/`oh_full`
from the plane on every task. [HW, H96 MAX M9]

## The first-conv ARGB sub-encoding

A convolution whose input is a PACKED IMAGE — 3 or 4 interleaved bytes per pixel
rather than an NC1HWC2 cube — runs on its own CNA datapath. It is not the normal
program at a small channel count; about sixteen registers are packed differently.

It is the only way a vision model's stem runs on this part at all: the normal path
needs `ic` a multiple of 32 (see below), and an image is 3. Twelve captured programs
pin it, at two image sizes (224x224 and 64x64) and two row splits.

The RK3588 has the same datapath — Mesa drives it as `CNA_CONV_CON1` `NONALIGN_DMA |
GROUP_LINE_OFF | ARGB_IN(8)` for a 1-channel input — and `0x100C` is one of the few
geometry registers the RK3576 does NOT re-pack: `GROUP_LINE_OFF` is bit 29 and
`ARGB_IN` is bits[15:12] on both parts. The captured `ARGB_IN` of `0xA` sits exactly
where Mesa's 1-channel `0x8` does, one step per extra image channel, so the field is
`8 | (image_channels - 1)`. `CONV_MODE` is 6, a third value beside direct (0) and
depthwise (1).

**What the datapath does.** The CNA reads the packed row straight out of DDR and the
CVT — bypassed on every other layer — expands each pixel to FOUR int8 lanes while
applying a per-channel scale and offset. The kernel's horizontal extent is then
folded into the channel axis: the conv the MAC sees is `kh x 1` over `4*kw` channels.
That is why these programs carry two disagreeing channel counts, which is the
sub-encoding's clearest signature in a capture.

| Register | Normal path | ARGB path |
|---|---|---|
| `0x100C` | `proc<<7 \| in<<4 \| conv_mode` | `GROUP_LINE_OFF \| ARGB_IN(8\|(ic-1))<<12 \| 6` |
| `0x1020` weight elems | `ic*kh*kw` | `kh * round16(4*kw)` — a 16-byte row per kernel row |
| `0x101C` weight bytes | `ic*oc*kh*kw` | `oc * kh * round16(4*kw)` |
| `0x1030` hi | `ic*kh*kw*2` | `weight_elems * 2` (same rule, folded count) |
| `0x1028` lo | `ic-1` | `4*kw - 1` — the FOLDED count |
| `0x107C` | `ic-1` | `image_channels - 1` — the DMA's real count |
| `0x1044` lo, `0x103C` hi | `ceil(iw*ic/64)` | `iw*4/64 + 1` |
| `0x1048` CVT | `0x0B`, bypassed | truncate 14 per live channel, bypass CLEAR |
| `0x104C`/`0x1050` | scales = 1 | `0x4000` (Q14 unity) per live channel |
| `0x1054`-`0x105C` | 0 | the uint8 zero point, negated, per live channel |
| `0x1084` pad const | `zp - 0x80`, already centred | raw `zp + 0x80`, one byte per channel |
| `0x1090` line stride | `iw*4` (a cube row in 4-byte words) | `iw*ic/16` (the packed row in granules) |
| `0x1078` hi | `iw-1` | `iw*ic/16 - 1` |
| `0x1094`, `0x1098` | plane / task surface strides | both `line_stride * ih` |
| `0x118C` | `(iw-1)<<16 \| (ih_full-1)` | `(entries-1)<<16 \| (entries-2)` |
| CORE `0x3018` | `0x10000001` | `0x10000081` |

Everything past the CNA is the direct path's, including `0x40B8`'s channel-group
jump — so this is a CNA sub-encoding and not a second pipeline.

Three things follow that a caller has to act on. The feature buffer is a packed
image, `iw*ic` bytes per row, so a row-window plan's feature offsets are in those
units. The pixels are raw uint8 and the converter does the centring, which is why
`0x1084` pads in the raw byte domain — the pad is inserted before the CVT runs. And
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
48-element weight kernel. The mechanism above — 4 lanes per pixel column, `kw`
columns — is what makes `4*kw` the reading rather than a curve fit, but a first conv
at another kernel width is unvalidated.

**What is NOT transcribed, and it is what stands between this and a running first
conv: the weight cube's byte layout.** The captures carry register programs, not
weight BOs. What they pin is the cube's size and stride — `oc * kh * round16(4*kw)`
bytes, one 16-byte row per kernel row per output channel, `4*kw` of those 16 bytes
live — and nothing about its byte order. Packing it is a bring-up item on the board.

## CBUF row reuse across a windowed sequence

The emitter's own windows refetch: each task fetches its whole window against a CBUF
base of zero, which is self-consistent and bit-exact on the part. The vendor's
programs do something else, and twelve captured continuation tasks pin it exactly.

Consecutive windows share `kh - stride_y` input rows. The vendor keeps those resident
and moves the CBUF write pointer instead of refetching them. Seven registers carry it:

| Register | On a continuation task |
|---|---|
| `0x1028` hi | `entries * (ih - retained)` — the granules this task FILLS, not its window |
| `0x1078` lo | `ih - retained - 1` |
| `0x1098` | `iw * (ih - retained)` |
| `0x103C` lo | CBUF granule where the task's WINDOW begins — where the retained rows sit |
| `0x1040` lo | CBUF granule where FETCHING resumes = the window base plus the retained rows |
| `0x1018` bit 31 | set when the task retains rows |
| `0x1038` bit 31 | set on every task past the first, retaining or not |

and the FEATURE ADDRESS (`0x1088`) points at the first NEW row rather than at the
window start. The vendor's `dw` capture is the clean case: its second task reads
input rows 89..111 of a 112-row plane, retains 2, and carries `0x1088` = row 91.

`retained` is the window arithmetic — the previous window's end minus this window's
start — which at these geometries equals `kh - stride_y`. When it is zero the vendor
does not continue the base, it RESETS to the sequence's origin, which is what a k=1
continuation carries.

`rocket_rk3576_plan_rows()` fills `retained` and `cbuf_resident` on every task and
`gen_conv2d_int8_rk3576_reuse()` emits it. It is opt-in, and the reason is worth
stating: it saves `entries * retained` granules of feature DMA per continuation task,
under a tenth of the window's traffic at the captured geometries, against a window
cost of ~1.2 ms that is the driver's completion poll and not the fetch. So it does
not shorten a windowed conv measurably. What it buys is that the vendor's split
programs become a COMPLETE oracle — those seven registers were previously reported
and excluded from the gate's diff, and are now checked exactly across thirteen
continuation tasks, at three kernel sizes, both strides, both channel counts and all
three CBUF origins. UNVALIDATED ON SILICON: a wrong base reads resident rows that are
not there and corrupts silently, so the refetching path stays the default.

**Three-task sequences are not pinned.** Every captured split is two tasks, so that
`cbuf_resident` accumulates over a longer sequence is the natural extension of the
same arithmetic and nothing more.

## The same graph, compiled three times

Each `.rknn` capture holds its graph compiled three times, and the difference is
never the geometry. Two of the three are the same per-layer task split at a
different CBUF ORIGIN — 0 and 7168 granules, added to every base in the table above.
The third splits the whole chain instead of each layer: it runs all five layers over
the top half of the image and then all five over the bottom half, from a CBUF origin
of 6144, and it is the one that sets `0x1014` bit 28 and drops the early duplicate
`0x1038` preamble write (138 register writes rather than 139).

Reading the third as the two-core compile — 7168 granules being one core's whole
CBUF pool, an image-plane split being the standard way to use two — fits, and is not
evidence. `0x1014` bit 28 sits where the RK3588 puts `NN_MODE`, which is consistent
with a mode bit and says nothing about which mode. All of it is decodable off-device
and none of it is decided by these captures.

For an emitter none of this matters: the origins, the format low bits (`0x1018`,
`0x1038`) and the allowance field are the vendor allocator's per-compile choices,
and a hardware sweep already found only `0x1018` bit 30 and `0x1038` bit 31 live on
the int8 path.

## The DPU epilogue: what an appended stage moves

Three captures are the same conv as `conv2d_rk3576.rknn` with one extra op appended
— a bias add, a scale, an eltwise sum — which makes them a controlled experiment the
vendor ran for us. The delta against the plain conv is three registers, and it is the
SAME three, with the same values, for all three ops:

| Register | plain conv | conv + an epilogue stage |
|---|---|---|
| `0x4044` BS ALU operand | `1` | `0` |
| `0x4050` BS config | `0x80011111` | `0x80021111` (bits[19:16] 1 -> 2) |
| `0x501C` BRDMA config | `0x710` | `0x114` |

That the three ops are indistinguishable here is the finding. These registers do not
encode WHICH epilogue op runs; a per-output-channel bias, a per-channel scale and a
per-channel sum all lower onto one BS-stage configuration, and what separates them is
the coefficient buffer's contents and the requant triple (`0x40AC`-`0x40B4`), which
is the only other thing that differs between the three captures.

Neither `0x4050` nor `0x501C` decodes under the RK3588's field map at these offsets
— `0x80011111` sets bit 12, which is reserved in the RK3588's `BS_CFG`, and the
RK3576's `BRDMA_CFG` is wider than the RK3588's 5-bit field — so both are
transcribed constants here rather than decomposed. The reading that fits is that the
BS stage takes its second operand from `DPU_RDMA` rather than from the inline path,
which is the same stage and the same buffer as the asymmetric-weight `B` term.

One run settles what it does, with no emitter change:

```
ROCKET_RK3576_SET="0x4044=0,0x4050=0x80021111,0x501c=0x114"
```

against a conv whose coefficient buffer carries a known non-zero `B`.

## Depthwise: the envelope, and the two buffers that are not the direct path's

`gen_conv2d_dw_int8_rk3576()` computes bit-exactly. Its register program reproduces
the vendor across 88 captured task programs with no field left open — and that was
never sufficient, because the two things it was still wrong about are BUFFERS rather
than registers. A `.rknn` capture carries a register program; the coefficient group
and the weight cube are memory the registers point at, and a capture says nothing
about either. Both differ from the direct path's, both are silent when wrong, and
both had to be read off the part.

### The envelope

Bit-exact against a CPU model over the whole surface — every shape `maxdiff 0` — at
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
kernel or the plane — the same per-submit dispatch floor the direct path pays, with no
compute term visible at these shapes. A row window costs that per task.

The awkward numbers in that table are the point of it. **Channel counts that are not
multiples of 32** are where this path's granules stop agreeing — the weight cube
rounds to 16, the CBUF allocation takes a 16-group count of 3 mod 4 one group
further, and `0x4050`'s 2-bit group field wraps — and a gate that rounds every
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
different rates** — `A` strides 4 bytes per channel and `C` strides 2 — so past the
first eight channels each is read out of a different group. The BS stage computes
`(acc + A) * C`, so what comes back is: most channels multiplying a bias by a zero
and never reaching DDR at all, a few whose `C` lands in the `A` region squaring their
own bias, and one group's `C` multipliers read as the next group's biases. That is a
correctly sized, mostly empty surface with no fault to catch it — the same signature
a wrong geometry register gives, which is why it read for so long as a datapath
defect rather than a packing one.

### The B term is ADDED: `acc + B*sum(x)`

Every vendor capture carries `B = 0`, so the field's existence was read off its position
and width rather than off a program that uses it, and the sign convention was open. It is
not now. Driving `B` on the direct path against a CPU model, `acc + B*sum(x)` — where
`sum(x)` is the sum of the input elements the output contracted — explains **every**
output of **every** direct shape in the conv gate's envelope group, at `B` = 1, 40, 127,
-37 and -128. `-B*sum(x)` and an inert `B` explain only the few percent that coincide,
which at small `|B|` is larger simply because the correction often rounds to the same
int8. `ROCKET_G_WZP` in `tests/rk3576_conv_gate.c` drives it and scores all three models.
[HW sweep, H96 MAX M9, measured 2026-07-27]

**So a weight zero point is programmed NEGATED.** An asymmetric weight is
`w_true = w_stored - wzp` and its correction is `-wzp*sum(x)`, so pass `B = -wzp` to
`rocket_rk3576_pack_coeff_asym()`, whose parameter is named `b_term` for exactly that
reason. Getting it backwards gives a plausible surface with a bias-shaped error and
nothing that faults.

There is nowhere in the DEPTHWISE group to put a weight zero point, so **an asymmetric
depthwise weight has to be folded into the bias**; there is no `_asym` form.

### The weight cube

Two layouts, one per precision, sharing a block structure and nothing else.

The FLOAT cube is decoded — not inferred — from captures whose weights carry a unique
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
- a trailing partial group is **dense** — its tap stride is the channels it holds,
  not 32, so C=48 is 432 slots and not 576.

This is `rocket_rk3576_weight_dw()`; `dw/named/mknamed.py --decode` re-derives it from
the committed captures without a board. The naming has to be carried in FLOAT weights:
per-channel weight quantization normalizes each channel by its own maximum, which
erases any naming carried in magnitude, while fp16 represents integers exactly to 2048
so a unique-integer ramp survives the float path untouched and is findable in the file
by its exact value set.

The INT8 cube is **not that cube's low byte**, and it is not the RK3588's
group-of-64 single-byte packing either. Read off the part with `rk3576_conv_gate
dwmap`, which drives an impulse feature against a cube that is zero but for one byte
and reports which output that byte reaches — over a C=32 k=3 cube every one of the 576
bytes reaches exactly one (channel, tap): [HW sweep, H96 MAX M9]

```
byte(c, kh, kw) = (c/64)*64*KH*KW*2 + (kh*KW + kw)*held*2 + 4*((c%64)/2) + (c%2)
held            = min(64, round16(C - (c/64)*64))
```

- the channel group is **64**, where the float cube's is 32;
- inside a tap block the channel a byte carries is `2*(b/4) + (b%2)`, so channel `c`
  owns two bytes per tap, at `4*(c/2) + (c%2)` and two further on, and **both are live
  and both contribute** — a weight written into both is added twice, so the packer
  writes the first and leaves the second at zero;
- a trailing partial group strides by what it holds **rounded up to 16**, where the
  float cube's is dense in the raw count. That is what makes the whole cube exactly
  `round16(C)*KH*KW*2` bytes.

This is `rocket_rk3576_weight_dw_int8()`, and it returns a BYTE offset where the float
entry point returns a slot. Writing an int8 weight as a 16-bit value into the float
slot puts its **sign extension into the byte that belongs to the next channel** — a
silent -1 weight on a neighbour rather than padding.

Two shapes hide the group size, and they are the ones that pass first: at a single
group the two layouts coincide, and at `k=1` there is one tap, so the group base and
the tap stride cannot be told apart. **C=64 k=3 is what separates 64 from 32** — there
`dwmap` predicts 576 of 576 (channel, tap) positions at 64 and 64 of 576 at 32. C=24
and C=72 are what separate the round16 partial group from the raw one.

### The output surface stride is padded to four

`0x401C` is the distance in elements from one 16-channel output group to the next, and
the depthwise path programs `round4(ow*oh_full)` where the direct path programs
`ow*oh_full`. It is a DDR stride, so **a caller has to size the output BO and
de-scatter with it** — `rocket_rk3576_out_surf_elems(ow, oh_full, dw)`. Assuming the
plane instead lands every group past the first up to four elements early, which reads
as "the first 16 channels are exact and the rest are noise": at C=32 exactly half the
surface, at C=64 exactly a quarter.

Invisible at any plane whose `ow*oh_full` is already a multiple of four, which is most
of them and was all of them — 16x16 and 112x112 pass, 15x18, 17x19 and 19x19 do not.
`0x40B8`'s plane term is four of these same rounded surfaces.

### The probes, and what each can and cannot see

`rk3576_conv_gate` carries five modes. Each has a `direct` control, and the control is
load-bearing: it runs the same reading on the path that is known bit-exact, so a mode
that cannot reproduce the direct map is not measuring what it claims to.

| mode | what it reads |
|---|---|
| `dwmap` | which output each WEIGHT byte reaches, one submit per byte |
| `dwbias` | which coefficient slot each CHANNEL read |
| `dwcoeff` | the inverse: which output BYTES each coefficient position moves |
| `dwout` | the raw output BO, undescattered |
| `-l` | the shape table, without running it |

Two properties are what made them decode rather than score, and both are worth keeping
in any successor:

- **`dwcoeff` measures its own baseline.** It used to compare each position against
  the constant it had packed, which is a delta only where the part agrees the surface
  is flat. On a path whose surface is not flat every position reads as "changed" and
  the probe says nothing. One unprobed submit kept as a reference image makes a
  per-position delta meaningful whatever the baseline looks like — there is no need to
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

- The WEIGHT granule is `C` rounded up to 16. `WEIGHT_BYTES` (`0x101C`) is
  `round16(C)*kh*kw*2`, `WEIGHT_ELEMS` (`0x1020`) is `round16(C)*kh*kw`, and the
  weight-bytes-per-kernel term (`0x1030` high) is `kh*kw*round16(C)/8` — the last
  confirmed at k=1, 3, 5 and 7, where the found captures were all k=3 and could not
  tell `kh*kw*C/8` from `kh*kw*4`.
- The FEATURE granule is `C` rounded up to 16 and then, if the resulting 16-channel
  group count is 3 mod 4, up one group further: 48 rounds to 64 and 112 to 128, while
  80 and 144 stay put. It sizes the CBUF entry counts (`0x1028` high, `0x103C` high,
  `0x1044` low). Read as hardware, channels are fetched in blocks of 64 and a
  trailing partial block holds one, two or four sixteens — never three.

**`0x4050`'s depthwise field tracks the CHANNEL COUNT, not the stride.** Bits[9:8]
are the 16-channel group count minus one, modulo 4, computed on the *unrounded*
count — C=48 carries 2 there while its feature allocation is 64's. The found captures
could not attribute the field because every C=32 program in them is stride 1 and
every C=64 one is stride 2; manufactured ones at both strides show it flat in the
stride and stepping with C, wrapping at 80, 144 and 208. Emit the C=32 word
unconditionally and the BS stage above C=32 gets a word the vendor never uses, which
returns **a wholly untouched output BO** — no surface at all, no fault, no dmesg
line.

**`0x40B8`'s plane term is a whole DESTINATION SURFACE, and the multiplier is flat in
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
has a plane that separates it — a VALID-padded oc=128 conv at `ow*oh_full = 105` is
bit-exact with 105 programmed and wrong with 108. Carrying the depthwise rounding
onto the direct path costs that shape and nothing else in the 67-shape gate, which is
worth knowing: one shape in the whole envelope is sensitive to it.
[source-confirmed + HW sweep, H96 MAX M9]

**`0x118C` is `iw-1` in BOTH halves.** The low half read as the full plane height for
as long as every capture was a square plane. 142 non-ARGB programs carry `iw-1`
twice, with no exception. This one is not depthwise-specific — it was wrong on the
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

What used to be described here as two independent gaps — an output writer that spent
four bytes on a two-byte element and reached only `oc/2` channels, and a feature
surface index that capped a task at 8 input channels — are one fact seen from two
sides. The DPU's output element stride is `16/ic` words. At `ic = 8` that is two
words per element, which is where the "four bytes on a two-byte element" reading came
from; at `ic = 16` it is exactly one and the surface is the plain native cube.

Everything below is [source-confirmed, RKNN-Toolkit2 rk3576 float build; HW sweep,
H96 MAX M9, measured 2026-07-26].

### The float program is a transcription, and here is how to get one

Every `.rknn` in the vendor-capture set is int8, and a zero precision field is
invisible — it says nothing about where the field is, how wide it is, or whether this
part carries one there at all. That is why the float fields were inferred from the
RK3588's packing for as long as they were, and why one of them was wrong.

**RKNN-Toolkit2 will emit a float program for this part.** `do_quantization=False`
on a FLOAT ONNX leaves the weights float and the vendor compiler picks the float
datapath; `tests/data/rk3576-vendor-capture/float/mkfloat.py` rebuilds the captures
and carries the dependency pins, which are particular (onnx 1.16 removed
`onnx.mapping` and setuptools 81 removed `pkg_resources`; the toolkit imports both,
and neither failure names itself).

Two things make the resulting diff small enough to read:

- **Only seven registers move with `ic`** across the whole 139-word program —
  `0x101C`, `0x1020`, `0x1028`, `0x1030`, `0x103C`, `0x1044`, `0x107C` — and six of
  them our emitter already matched. All seven scale linearly.
- **Fourteen registers differ from what we emitted**, and they are the same fourteen
  at every geometry. Three of them are load-bearing; the other eleven can each be
  dropped with the datapath still exact.

### Three registers carry the float mode

Each is constant across every geometry the vendor emits (`ic` 8-64, `oc` 32/64, `k`
1/3, planes 16 and 32), and each is load-bearing **on its own**: with any one of them
at its integer value the contraction reads every feature surface twice and skips the
odd ones. That is why no single-register sweep ever found this — the fix is a
register *set*, and every sweep to date moved one register.

| register | integer value | float value | what is load-bearing |
|---|---|---|---|
| CNA `0x100C` | `0x00000000` | `0x00200120` | all of `bit21 \| proc<<7 \| in<<4` |
| DPU `0x4038` | `0x00120080` | `0x00100092` | the low half, `0x0092` |
| DPU `0x4050` | `0x80011111` | `0x00021111` | **bit 31 clear** |

- **CNA `0x100C` bit 21 is a float enable** the integer programs never set. It is not
  implied by the two precision fields: with both precisions right and bit 21 clear the
  contraction still doubles, exactly as it does with bit 21 set and a precision wrong.
- **DPU `0x4038`'s high half is free** — `0x0012` computes as well as `0x0010` — but
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
uniform in the channel axis, where a doubled read is indistinguishable from a scale —
see Traps.

CORE `0x3018` still does not follow the CNA word, and the division of labour is the
one already established:

- **CNA `0x100C`'s `in_precision` pins the operand WIDTH CLASS, not its type.** Values
  `1`, `2` and `3` — the three 2-byte codes — all compute; `0` (a 1-byte element)
  writes an entirely zero surface. So an fp16 and a bf16 program are indistinguishable
  in this field.
- **CORE `0x3018` is what pins the operand TYPE.** Only `2` computes; `3` (bf16)
  returns a wrong surface against fp16 data.

### The output is fp16 only with the float narrowing enabled

The epilogue is a float path and its result is an fp32 word. What reaches DDR is decided
by two things:

- **DPU `0x40B0` bit 16 is `fp32tofp16_en`**, at the RK3588's own bit position. Without
  it the fp32 word is written as-is.
- **DPU `0x4010` bits [31:29] select a WIDTH, not a conversion.** `5` writes the whole
  32-bit word; `2` and `3` write its low and high halves.

Composed: narrowing on plus width `2` writes true fp16 (`0x5400` for 64.0); narrowing on
plus width `3` writes the top half of the fp32 word, which is **bfloat16**; narrowing off
plus width `5` writes fp32, and narrowing off plus width `2` writes the fp32 word's low
mantissa bits — **zero for every value a small integer test pattern can produce**.

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
the whole surface** — every word names the (channel, pixel) this map predicts, with none
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
wrong count writes a full, correctly sized, WRONG surface with nothing to fault on, so
a caller must not be able to reach it silently; `gen_conv2d_rk3576_prec()` is the
unchecked bring-up entry beside it, and it is what the sweep's own modes drive.

Read the multiplicity straight off the part with probe 10: one input channel carries
1.0, every weight lane is 1.0, and the output *is* that channel's read count.
`ROCKET_FS_TAP` moves the tapped channel, and the pair `TAP=0` / `TAP=16` at `ic = 32`
separates the two failure modes in one run — before the float fields, they read 2.0 and
0.0; after, both read 1.0.

With the fields right the read count is 1.0 at `ic` 16, 32, 64 and 128 and at every
tap, so the *reads* are unbounded. What bounds a task is the writer's element stride
above.

### The fp16 envelope that computes

A single fp16 task is bit-faithful against the CPU model — `worst relative error 0` —
at **`ic = 16`**, for `k` 1, 3 and 5, planes 8x8 through 56x56, stride 1 and 2, `oc` 16
through 64, and every programmed output channel lands.

### An arbitrary `ic` via a 16-channel split

Splitting `ic` into slices of 16 and summing the partial surfaces on the host
reproduces the model exactly at any input channel count — **bit-exact, `worst relative
error 0`, at `ic` 16, 32, 64 and 128** crossed with `k` 1, 3 and 5.

`rocket_rk3576_plan_ic()` is the split, beside `rocket_rk3576_plan_rows()` and on the
same pattern: it lays the slices out and the caller emits one conv per entry. The slice
is FIXED at 16 rather than as wide as the CBUF allows — a wider slice would not be a
cheaper task but a wrong one. It plans the `ic` axis only; a plane whose 16-channel
slice still overflows the CBUF is refused with a pointer at the row planner rather than
run.

The slices cost nothing to address: at C2 = 8 the feature cube's channel groups are
contiguous planes of `iw*ih_full` 16-byte atoms, so slice `k` is the same BO at a base
offset, and an atom stays 16 bytes when C2 halves, so the stride is element-size
independent. Only the weight cube is rebuilt per slice — each slice is its own
convolution and so has its own group count, which is why a slice's cube is NOT a
sub-cube of the whole conv's (`rocket_rk3576_fp16_pack_slice_weights()`). Size that
cube by the GROUPS.

The bias belongs to one slice only. Every slice adds the whole `A` term, so a
coefficient buffer carrying a bias handed to all of them lands it `ic/16` times.

**On-chip accumulation is no longer blocked.** The DPU eltwise stage does this job on
the RK3588 (`ROCKET_KACC`) and would remove `ic/16` readbacks here; the reason not to
was that the partial it would read back carried the writer's defect into every step,
and at this contraction width that partial is a plain, dense, complete fp16 cube. It is
the open lever on this path rather than a defect to compose.

### What an fp16 conv costs

Priced against the `ic/16` submits the split spends, which is what the convolution IS
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
visible at these shapes — it is entirely the per-submit dispatch floor, and that floor
is the same 1.1-1.3 ms the int8 path pays ONCE for its whole `ic`
(`rk3576_conv_gate envelope`). Doubling the contraction width halved the submit count
and the NPU wall with it: the same shapes cost 8 and 16 submits at the 8-channel split.

Only the NPU column is a measurement. The host column is noisy on this thermally
limited board — the same shape varies by a factor of two run to run — and the 56x56
`k = 1` row is the clearest case of it. The NPU column is stable across pacing gaps of
0, 20 and 50 ms to within 6%.

Both planners take the precision the conv will be emitted at
(`rocket_rk3576_cbuf_f_prec`, `rocket_rk3576_max_task_rows_prec`; the shorter names are
those at int8). The feature plane's granule cost is per BYTE, so a 2-byte element
doubles it, and an allowance planned at the int8 rate would be sized for half an fp16
plane — which computes wrong with a full surface written. The WEIGHT side of the trade
stays at the int8 model on purpose, as an upper bound on the float slice: over-estimating
refuses early instead of corrupting.

### The write-extent field, and two DPU words that look like the answer

`0x4050`'s low nibbles encode the written extent: 1 / 2 / 4 bytes per element as
`{0, 1, 3}`, the direct int8 word carrying `1` and the depthwise word `3` — the same
`size_e = bytes-1` relation the RK3588 pairs with its surface-advance multiplier, which
the RK3576 already tracks (`ow*(2*oh_full - oh)` direct, `ow*(4*oh_full - oh)`
depthwise). It moves the written *extent* only, never the atom, and it is not what the
float mode needs from this register — bit 31 is.

Two DPU words do move the output and are worth naming, because they look like the
answer and are not:

| register | what it actually does |
|---|---|
| `0x40D0` low 16 | a per-byte write enable for the 16-byte atom — clearing bit *n* leaves byte *n* untouched |
| `0x40D0` bits [19:16], `0x40CC` bits [3:0] | shift the written data by whole bytes within the atom |

### The knobs that settle the rest

`A` (bias) is consumed as **fp32**: packed as its fp32 bit pattern and read back at width
`5` it returns `3f800000 40000000 40400000 40800000` for `bias[c] = c+1`, exactly
1.0/2.0/3.0/4.0. `C` (multiplier) is consumed as **fp16**, so the int8 default `C = 1` is
the denormal 6e-8 and underflows the whole surface to empty. fp16 `1.0` is `0x3C00`.

**Asymmetric weights are a packing question, not an encoding one.** `B` in the
coefficient group is a per-output-channel int16 sitting beside the bias `A` and the
multiplier `C`, and driving a weight zero point through it needs no register change —
`rocket_rk3576_pack_coeff_asym()` takes one. What is unknown is the **sign
convention**: the accumulator correction is either `-B*sum(x)` or `+B*sum(x)`, and
nothing in a capture that always writes `B = 0` distinguishes them. A wrong sign is a
bias-shaped error on a plausible-looking surface, so drive it at a zero point far from
zero, where the two cannot be confused, and against a CPU model.

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
| 1 | uniform feature and kernel, each value its own knob (`_PIN`, `_PW`) | `ic*k*k * in * w` — and a result that tracks a side's *high byte* names the side being read as int8 |
| 2 | one non-zero element on each side | one non-zero output lane, at a position |
| 3 | the pixel ramp on **every** input channel | `y*iw+x+1` in every lane, whatever the lane map |
| 5 | uniform feature, kernel *c* weighted *c+1* | `c+1` in every lane, whatever the lane map |
| 6 | input channel *c* carries *c+1*, kernel taps ic=0 | `1` — anything else names the channels the part paired with lane 0 |
| 7 | uniform feature, kernel ramps over ic | `1+...+ic` — anything else counts the weight lanes it walked |
| 8 | unique naming: every lane carries `ow*oh*c + p + 1` | itself — the output map, decoded rather than guessed |
| 9 | unique feature, output channel *c* taps input channel *c* | the input cube handed back through the datapath |
| 10 | one input channel at 1.0, every weight lane 1.0 | `1` — the value **is** that channel's read count |

Probes 3 and 5 are deliberately *uniform in the axis they do not name*, so they read the
output map without first knowing the lane map. Probes 6 and 7 do the reverse.

Probes 8, 9 and 10 exist because that uniformity is also their blind spot. Each of 1, 3,
5, 6 and 7 passes against a datapath that reduces the wrong input channels, so a probe
set built only from them reports an exact conv that is not one. 8 and 9 name lanes
instead of scoring them — a *bijection* is the pass, not a match count — and 10 measures
the reduction directly. Run `map` before believing any fp16 layout claim.

| knob | what it does |
|---|---|
| `map` (argv) | drive probe 8 and DECODE the surface; reports where each lane landed and how many copies |
| `ROCKET_FS_OUT_LAYOUT` | `0` the native output cube, `1` the library's `rocket_rk3576_fp16_out_index()` |
| `ROCKET_FS_TAP` | move the single live weight lane off ic=0; `-1` taps EVERY input channel (probe 3) |

| knob | what it does |
|---|---|
| `int8` (argv) | the control — the known-good path through the same harness |
| `icsplit` (argv) | the LIBRARY's `ic` split, scored end to end and timed |
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
  patches register 0 reports that every value changed nothing — which is exactly what a
  correct sweep of an inert register looks like. This cost a full pass over `0x100C`
  before the reading that mattered was found.
- **`feature_data()` and `weight_conv_*()` take 1-based channel and kernel indices.** A
  0-based call drives the group remainder negative and misplaces the whole cube. The
  result is a wrong surface rather than an absent one, so it reads as an encoding error.
- **An integer test pattern cannot see a low-16-bit truncation.** Every value a small
  integer convolution produces has zero low mantissa bits in fp32, so reading an fp32
  result through a 16-bit output width returns a *uniformly zero* surface — the same
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
- **Size a weight cube by its GROUP, not by `ic`.** A partial input-channel group still
  occupies a whole one, so `ic = 8` at an ic group of 16 needs a cube for 16. An
  allocation taken from `ic` alone under-allocates, and at `k = 1` the overrun stays
  inside the BO's page and computes correctly anyway — it only surfaces at a kernel large
  enough to run past the page, where it reads as "this part cannot do `k = 3`". Size the
  buffer from `ceil(ic/group)*group`.
- **A cube-geometry sweep needs a shape where the field is observable.** The float
  weight cube's output-channel group is algebraically invisible at `k = 1` — the kernel
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
- **The bias belongs to ONE slice of an `ic` split.** Every slice runs the whole epilogue
  and adds the whole `A` term, so a coefficient buffer carrying a bias handed to all of
  them lands it `ic/16` times. The error is a per-channel constant on an otherwise exact
  surface, which is the shape a wrong `B` sign convention also has.
- **A slice's weight cube is not a sub-cube of the whole conv's.** Each slice is its own
  convolution, so its group count follows the slice rather than the total. Slicing the
  full cube by byte range hands the part a correctly sized cube with the kernel positions
  shuffled.
- **One register at a time cannot find a mode.** The float datapath needs THREE
  registers together, and each of the three looks inert while either of the others is at
  its integer value — a leave-one-out over the working set is what separates them, and a
  single-register sweep of any one of them reports it dead. Two of the three are also
  unreachable by a single-BIT sweep from the integer word.
- **A vendor capture is cheaper than a sweep, and one can be manufactured.** The float
  fields went un-transcribed for as long as they did because every capture on hand was
  int8 — not because no float capture could exist. RKNN-Toolkit2 emits one for this part
  in about an hour of dependency pinning, and it settled in one diff what bit-level
  sweeps had not. The same lever settled four depthwise register formulas and the
  depthwise weight cube. Before sweeping for anything, ask what ONNX would make the
  vendor compiler emit it.
- **A found capture set is a set of CONFOUNDS.** The vendor compiles real models, so
  the axes that co-vary in real models co-vary in every capture: on this part every
  C=32 depthwise program was stride 1 and every C=64 one stride 2, every depthwise
  kernel was 3, and every plane was SQUARE. Three register formulas were fitted
  through those confounds and all three were wrong — `0x4050`'s channel field, the
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
that sign to zero. The two sides are genuinely independent — the field selection
follows the sign of the **accumulator**, not the sign of the weight, so swapping
the feature sign swaps which output channels move. [HW sweep, H96]

Every capture stores 0 here, for the same reason every capture stores 0 for the
feature, weight, output and bias bases: it is an address the vendor runtime
patches at load time, and a `.rknn`-derived program is the *stored* register set.
Leaving it at zero is not benign, because **IOVA 0 is a real buffer on a mainline
`rocket` stack** — the per-fd address space bump-starts at 0, so whichever BO the
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
happen to have zero low-6-bits — a fill of 64, or of 0 — computes perfectly. The
symptom moved with the *feature cube's channel 1* only because that channel's
first byte is byte 1 of the buffer.

The pattern to recognise: a result that is exact for some feature tensors and a
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
read the behaviour as the part arming its weight loader once per power session and found
independent per-op submits could not bypass it. Reproducing it on our own encoder and our
own submit path is what placed it in the driver.

### The cause is the `PC_TASK_CON` field width

`PC_TASK_CON` packs `TASK_NUMBER` in the low bits with the control bits directly above
it — `TASK_PP_EN`, then `TASK_COUNT_CLEAR`, then a bit the TRM marks reserved. The
field is 12 bits wide on the RK3588 and **16 on the RK3576**. `rocket` builds the word
from the RK3588 field accessors unconditionally, giving `0x7001`, so on this part it
asks the PC for a **task count of 28673** with all three control bits landing above the
register's defined fields. The PC starts that 28673-task program, runs the one task it
was handed and is left mid-stream; nothing else in the session starts. A power cycle
resets the PC, which is the whole of why a gap re-arms it.

Building the word from a per-SoC width instead clears it, and leaves the RK3588 word
bit-identical at 12 bits. Back-to-back with no gap: **1 of 8 submits wrote before, 8 of 8
after**, and 32 of 32 on a longer soak — byte-identical to each other and bit-exact
against the CPU model. A row-windowed 112x112 k3 conv is bit-exact over the whole surface
with no inter-task gap. The patch is `rk3576/npu/0008` in the `rk3576-npu` profile.
[HW sweep, H96 MAX M9, measured 2026-07-25]

Sweeping the control field at each width isolates the cause to `TASK_NUMBER` alone:

| word written | `TASK_NUMBER` seen (16-bit field) | wrote |
|---|---|---|
| `0x70001`, `0x60001`, `0x50001`, `0x30001`, `0x20001`, `0x10001`, `0x00001` | 1 | 8 of 8 each |
| `0x07001` (the RK3588 word) | 28673 | 1 of 8 |
| `0x06001`, `0x04001`, `0x02001`, `0x01001` | 24577, 16385, 8193, 4097 | 0-1 of 8 |

So `TASK_COUNT_CLEAR` is not the mechanism — with the field correctly placed the part
works with every control bit clear, and with it misplaced no control-bit value tried
helps. The other two per-SoC PC parameters in the vendor config are inert here:
`rocket` never reads `PC_TASK_STATUS`, so the `0x3c`/`0x48` offset delta cannot bite,
and the part runs correctly without `pc_dma_ctrl`'s IRQ-locked `PC_DATA_ADDR` write.
The RK3576-only `state_init` hook is likewise not required.

Nothing a regcmd can write clears it, which is why the fix had to be a driver patch.
Sweeping the CNA/CORE/DPU/RDMA `S_POINTER` value over the whole ping-pong field
(`POINTER`, both `PP_EN`s, `PP_MODE`, both `PP_CLEAR`s), writing `PC_TASK_CON` from
inside the stream, and replaying the vendor's state-init sequence as regcmd writes all
leave it at one job per power session. Two of those are informative in themselves:
forcing `POINTER=1` makes **every** job write nothing, so the bit is live and selects a
producer register group — and the fact that job 2 still fails with `POINTER` held at 0
means the consumer is not advancing behind us, which rules the ping-pong groups out.
[HW sweep, H96 MAX M9]

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
replayed from a regcmd the PC is fetching. It is not needed for correct back-to-back
submits — the task-number width accounts for the wall on its own — so `rocket` does not
carry it. If a future symptom does need it, `rocket_job_hw_submit()` already opens with
`PC BASE_ADDRESS = 0x1`, which is slave mode, and the vendor's CNA writes drop straight
into that window.

### Working against an unpatched driver

Everything below applies to a `rocket` without `0008`, and is worth keeping because a
probe on a stock module hits all of it.

**`autosuspend_delay_ms` is the only userspace lever, and it is unreliable.** Writing 0
to `/sys/.../27700000.npu/power/autosuspend_delay_ms` makes the driver drop the power
domains straight after each job, so a back-to-back loop climbs from 1 job in N to
about half — 8 of 16, and 6 of 6 on a shorter run. It is a race, not a fix: when the
next job arrives before the autosuspend work runs, no power cycle happens and that job
is walled. Anything that must be correct has to check per task that its own output
landed and resubmit, which is what `rk3576_first_light` does when it runs a row split.

A gap of ~1 s is enough to make results stable and repeatable; at ~0.4 s a probe
still occasionally comes back with the output BO wholly untouched. Read a single
"the DPU did not write at all" as the wall, not as a register result — re-run
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

Measured by the wall time of a one-task 8x8 conv, five repetitions per point, with the
period exposed as a module parameter so one build sweeps it:

| poll period | 1000 us | 500 us | 250 us | 200 us | 150 us | 100 us |
|---|---|---|---|---|---|---|
| per-submit wall | 1.2 ms | 0.65 ms | 0.45 ms | 0.4 ms | 0.32 ms | wedges |

So the floor tracks the period one for one down to about 150 us, where roughly 0.15 ms
of non-poll cost is left — the ioctl round trips, the job setup and the completion
readback. Shortening the period from the stock 1 ms to 150 us takes the floor from
1.2 ms to 0.32 ms, which is **3.75x off every submit-bound cost on this part**.

**Do not read a gate's total wall time as the floor.** The conv gate's 102 shapes take
about 5.9 s at every period from 1000 us down to 150 us, unchanged, because that gate is
dominated by host packing and its CPU model rather than by submits. It looks like a
flat, negative result and it is measuring something else. Time a single small shape.

**What stops it being a free 3.75x is the IOMMU, and the boundary is load-dependent
rather than a threshold.** At 100 us it wedges outright — `Enable stall request timed
out`, then every submit times out with its output untouched. At 250 us, from a clean
boot, a convolution workload is untouched (102 emitter-gate shapes and 111 library-gate
shapes all bit-exact, no dmesg) while the matmul gate provokes four stall lines and
loses three of its 43 shapes. So the conv path tolerates a short period and the matmul
path, which drives the DPU's 32-bit output writer, does not. Whether that is the poll
rate itself or the stale-page-fault clear
(`0005-iommu-rockchip-clear-stale-page-faults-before-stall`) not covering a job that
completes this fast is not decoded, which is why `patches/rk3576/npu/0009` ships the
period as a parameter at the stock default rather than shortening it.
[HW sweep, H96 MAX M9, measured 2026-07-27]

### A wedge leaks a runtime-PM reference, and then ONLY the int32 path is broken

This is the most misleading state the part reaches, and it cost most of a session.

After an IOMMU wedge the NPU device can be left pinned runtime-`active`: `power/
runtime_status` reads `active` forever and `power/runtime_suspended_time` stops
advancing, even with nothing running and `power/control` at `auto`. **The leaked
reference lives on the platform device, not on the driver**, so `rmmod` and `insmod` do
not clear it — a freshly probed device comes back `active` and stays there.

What that breaks is exactly one thing. The int32 output writer's poisoning of the next
submit is cleared by the power domain cycling, so with runtime suspend disabled it never
clears: every i32 shape writes nothing, at `0/256`, `0/32`, `0/224`. Everything else
still works perfectly — the conv gate is 102/102, the library gate 111/1, and the plain
int8 matmul shapes all pass. The signature therefore reads as "the int32 path has a bug"
rather than "the machine is in a bad state", and it will absorb a debugging session.

**Check `power/runtime_status` before believing any int32 result.** A board in this state
also makes every other experiment lie: a poll-period sweep run through it reports the
matmul failing at every period, which is what it looks like when the poll has nothing to
do with it. Only a reboot clears the leak. A `rmmod`/`insmod` — with a settle of a few
seconds before the `insmod`, or the re-probe does not take — does clear the plainer
wedge, so it is worth trying first, but confirm `runtime_status` returns to `suspended`
before trusting the result. [HW, H96 MAX M9, measured 2026-07-27]

### A per-job core reset is not the way to clear the poisoning

`rocket_core_reset()` — a `reset_control` bulk assert, `udelay(10)`, deassert — is what
the driver already does on a job timeout, and calling it at job start looked like a way
to drop the power-cycle idle an int32-output job forces on the next submit. It is not.
Asserted before every job it takes the **IOMMU** down with it:

```
rk_iommu 27702000.iommu: Error during raw reset. MMU_DTE_ADDR is not functioning
rocket 27700000.npu: NPU job timed out: RAW_STATUS=0x30000000 MASK=0x800fffff OP_EN=0x00000000
```

`RAW_STATUS` carries both `PC_DONE` bits, so the job ran; nothing reached DDR because the
page-table pointer was gone. The matmul gate falls from 43 shapes passing to 2, the conv
gate to 0, and the damage outlives the setting — the part stays wedged until the module
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

after which every submit times out and every output BO comes back untouched —
which reads exactly like "the encoder writes nothing" and will send a register
hunt down a false trail. The rail, the clock and the driver binding all still
look healthy while this is true, so they do not discriminate. Check `dmesg` for
the stall timeout before trusting any negative result on this part. [HW, H96]

**Reloading the module clears the compute path**, which is worth trying before rebooting:
`rmmod rocket; sleep 8; insmod rocket.ko` re-runs the IOMMU attach and the conv gates
pass again immediately. The settle matters — a reload with a two-second gap left the part
still failing where an eight-second one fixed it. Confirmed against two independently
induced wedges, a 100 us poll period and a per-job core reset.

**It does not clear everything.** A reload leaves any leaked runtime-PM reference in
place, because that lives on the platform device, and the part then computes convolutions
perfectly while every int32 matmul writes nothing — see the section above. Check
`power/runtime_status` after a reload; if it does not return to `suspended`, reboot.
[HW, H96 MAX M9, measured 2026-07-27]

## The matmul: the same 1x1 convolution, and the envelope it plans inside

A matmul is a 1x1 convolution over these blocks, exactly as it is on the RK3588: the A
rows are the conv's spatial pixels, K is the input-channel axis, N the output-channel
axis. Nothing about the register program is new — the conv gate already ran a real
matmul shape without calling it one (`w-e4608-k1` is `ic=4608`, `oc=128` on a 4x2 VALID
plane, which is M=8, K=4608, N=128 in one submit) — so the whole of the work is the
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
them compute; the choice is a tiling one. What it costs is granules — a feature row is
`ceil(iw*K/64)` of them against the task's 4096 — so the pixels one task holds come out
at `262144/K` however the plane is cut, provided `iw*K` is a whole number of granules.
When it is not, every row rounds up and the waste is real, so take the widest divisor of
M that divides evenly.

### The M axis carries no constraint

M = 1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 16, 17, 31, 32 and 64 are each bit-exact, at every
factorization into a plane — `1xM`, `Mx1` and everything between. This is the RK3588's
answer inverted: there rows are the conv's spatial height, a height under 4 mis-computes,
`M%4` is the real bound and software pads `M==1` to 4. Here `M=1` is simply correct.
Whether it is *worth* running at M=1 is a separate question the dispatch floor answers.
[HW sweep, H96 MAX M9]

### int8 is the matmul precision, and the margin is large

The two precisions contract at wildly different rates. One int8 task takes
`ic*kh*kw <= 4608`, so K = 4608 lands in ONE submit; one fp16 task contracts exactly
sixteen input channels, so the same K costs 288 submits. Measured on the same shape at
both precisions, with the pacing outside the timing:

| shape | int8 submits | int8 ms | fp16 submits | fp16 ms | ratio |
|---|---|---|---|---|---|
| 16x512x64 | 1 | 1.4 | 32 | 42.9 | 30x |
| 32x1024x128 | 1 | 1.3 | 64 | 85.2 | 64x |
| 128x2048x128 | 1 | 1.2 | 128 | 171.3 | 142x |
| 56x4608x128 | 1 | 1.3 | 288 | 386.1 | 299x |

On the RK3588 fp16 wins and int8 buys only RAM. Here it is the other way round, and the
reason is not the arithmetic — it is that the fp16 contraction width is 16 while the int8
one is 4608. [HW sweep, H96 MAX M9, measured 2026-07-27]

### Throughput is MACs per submit, and N is the only free axis

A submit costs about 1.4 ms whatever it carries. `M*K` is capped by the feature budget
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
refuse EARLIER at `k>1` than anything measured there. That is the safe direction, and no
convolution shape in the gate comes near either bound — the largest is 516 KiB at oc=128.

**Repeat before believing a boundary here.** The first pass at this sweep ran each point
once and produced a clean-looking N ceiling that moved when the same points were re-run:
`1024x256x3072` failed in one pass and passed in the next. Three repeats plus a
classification of the failure — all-sentinel is a dead submit, a clean prefix is a
capacity bound, wrong values in a fully written surface is arithmetic — is what separated
the real bound from the noise. [HW sweep, H96 MAX M9]

### The int32 output: the first eight channels of every thirty-two

The DPU will emit its raw 32-bit accumulator on an integer program.
`gen_conv2d_int8_rk3576_i32out()` puts `precision_int32` in DPU `0x4010`'s output-WIDTH
field and pins OUT_CVT to exact unity, and the words that come back are genuine
accumulators laid out on the RK3588's int32 cube map, in 32-bit words:

```
word = (c/4) * ow*oh_full * 4 + 4 * (y*ow + x) + (c%4)
```

Decoded rather than assumed. The probe gives every accumulator a distinct value, so each
value names exactly one word, and every position read that way fits the map with no
misses — at `oc` 4, 8, 16, 32, 64 and 96, at `K` 16 through 256, and at `M` 4 and 8.

**The writer keeps the INT8 surface's byte budget whatever the element width is.** It
writes `ceil(oc/16)` contiguous blocks of `ow*oh_full` 16-byte atoms — the int8 surface,
exactly — and at four bytes an element each block carries four channels where an int8
block carries sixteen. Block `j` holds channels `32*(j/2) + 4*(j%2) .. +3`, so what
reaches DDR is

> **the first EIGHT output channels of every THIRTY-TWO**, and nothing else is touched.

That is one fact, not two. At `oc <= 8` every channel is delivered and the surface is
complete (`oc = 4` and `oc = 8` decode 32/32 and 64/64 with no waste); past it the yield
is `oc/4`. The earlier reading of "the extent is `ceil(oc/16)` groups" came from a sweep
run only at `oc = 32`, where the block count and the delivered-channel rule cannot be
told apart — `oc = 16` and `oc = 8` separate them, and both write two blocks.

**The way round it is the weight cube, not a register.** Program a multiple of the
output channels and put real channel `n` in a slot the writer delivers, leaving the rest
zero. Every real channel then lands in a delivered slot, and the surface reads back as a
plain cube. `rocket_matmul_int8_rk3576_i32()` is that, and it is bit-exact against a CPU
int32 model. On this narrow writer the multiple is four and the slot is
`32*(n/8) + n%8`, which collapses the map above to the plain int32 cube.

**What it costs is the output-channel axis, spent over.** The bytes are not wasted — the
budget comes out exactly the int32 surface — but the resident weight slice, the N tile
and the 2944-channel bound are all functions of the PROGRAMMED `oc`, so the multiple is
paid in MACs per submit. That follows from the measured budget rather than from the
encoding, so no register will buy it back — but `PROC_PRECISION` halves the multiple, and
that is the writer the library uses.

### The wide writer: the first eight of every sixteen, at half the cost

**`PROC_PRECISION` doubles the byte budget.** DPU `0x4010`'s low field (`[2:0]`) is the
DPU's own operand width, and driving it from int8 to int32 makes the writer emit TWO
16-byte atoms per (16-channel block, pixel) instead of one. The delivered set becomes
**the first eight output channels of every SIXTEEN**, so a real channel costs two
programmed ones rather than four. Nothing else moves: the operands stay int8 everywhere,
the arithmetic is bit-identical, and the CNA and CORE programs are byte-unchanged.
`gen_conv2d_int8_rk3576_i32out_wide()` emits it.

**Two atoms is the ceiling**, swept across all eight values of the field at
`M=8, K=32, N=32`:

| `0x4010[2:0]` | extent | what came back |
|---|---|---|
| 0 int8, 6 int4 | 64 words | one atom per (block, pixel) — the 4x rule |
| 1 int16, 4 int32 | 128 words | two atoms — the 2x rule, values correct |
| 2 fp16, 3 bf16, 5 | 128 words | two atoms, and the operands reinterpreted: no accumulator survives |
| 7 tf32 | — | every attempt was a dead submit |

**The map.** Work in 32-channel SUPER-GROUPS, each `4*A` atoms long, where `A` is the
surface's pixel count `ow*oh_full`. Inside one super-group the writer emits a single
linear STREAM indexed by `s = 2*p + j`, where `p` is the pixel and `j` the 16-channel
block, and cuts that stream into runs of `A` atoms. The run also carries the lane group
`L = (c%16)/4`, which is the slower axis:

```
atom = 4*A*(c/32) + A*(2*(s/A) + L) + s%A       s = 2*p + j
word = 4*atom + c%4                             delivered iff c%16 < 8
```

Because it is a stream, nothing rounds: an odd `A` simply cuts it at an odd place, and
`A = 1` works as readily as `A = 16`. Decoded, not fitted — the operands were drawn so
every accumulator in the tile is distinct, so each written word names exactly one
(channel, pixel) or the map is not a map. Read off at `oc` 8/16/24/32/40/48/64/96/128/192
and at pixel counts 4/5/6/7/8/12/13/16/21/33.

**Two bounds, and both fail silently.**

- **`oc` must be a multiple of 32**, because a partial super-group is not a truncated
  one. At `oc = 16 (mod 32)` the trailing group holds one 16-channel block instead of two
  and packs at one atom per pixel with no stream. At an `oc` that is not a multiple of 16
  at all — 24 is the case read off — the delivered channel set ROTATES with the pixel and
  there is no block form to write down. `rocket_rk3576_pad_oc()` gives the count; leave
  the padding zero.
- **`oh_full * oc < 4096`.** Past it the delivered set drifts with the pixel and a tile
  comes back part right. The boundary is measured at both ends — at `oc = 192` it holds
  at `oh_full = 21` and breaks at 22, at `oc = 64` it holds at 48 and breaks at 64 — and
  the mechanism behind it is NOT decoded. A row task on this path is a standalone 1x1
  convolution with its own surface, so honouring it costs submits and nothing else:
  `rocket_matmul_int8_rk3576_i32()` splits the row plan further until every task fits.

**A probe that drives every output channel finds a boundary the library never meets.**
With all sixteen channels of a group live the map stops fitting well before the bound
above — at `oc = 192` from `A = 48` on a flat plane. Under the SCATTER, where the
undelivered half of each sixteen is zero, the same geometries are bit-exact through the
library at `A` 48, 64 and 128. So a boundary found with every channel live is a boundary
of the probe: `ROCKET_MP_SCATTER=1` drives the probe the way the library drives the part.

**What it buys.** The N tile is bounded by the PROGRAMMED output channels, so halving the
multiplier doubles the tile and halves the submits — and on this path a submit costs an
idle rather than 1.4 ms. Measured through `rk3576_matmul_gate`:

| shape | 4x writer | 2x writer |
|---|---|---|
| M=64 K=1024 N=2048 | 4 submits, 686 ms | 2 submits, 339 ms |
| M=32 K=1024 N=4096 | 8 submits, 1319 ms | 4 submits, 675 ms |

Below the tile cap the two cost the same, so a shape whose N fits one tile shows no gain.

**It is OPT-IN — `ROCKET_RK3576_I32_OC_MULT=2` — because it intermittently emits zeros.**
On about one run in ten of the shape that shows it most, the wide writer emits a
contiguous block of one 32-channel super-group's stream with zero data while the rest of
the surface is exact.

**It does not drop writes.** That was the earlier reading and it was wrong. Stamp the
output BO with a sentinel before the tasks run — bracketed by `PREP_BO` and `FINI_BO`, so
no dirty line is left to race the DPU's DMA, and verified to reach DDR before any submit
— and the bad atoms come back **zero rather than holding the sentinel**. The writer
reaches them; the data is wrong. On a fresh BO, which arrives zeroed, the two are
indistinguishable, which is what made a dropped write look like the answer.

**What it is not.** Not a readback race: a second fence and a second read return
byte-identical contents. Not the poisoning below, whose signature is an *empty* region
and which the power domain cycling clears — this one is indifferent to the idle ahead of
it, at 3 failures in 30 with no idle against 6 in 30 at 800 ms. An earlier reading of
that sweep, 7 in 20 against 0 in 20, did not survive a larger sample.

**Its shape, in the writer's own coordinates.** The writer emits two atoms per stream
position `s`, one per lane group, so its emission order is `(s, L)` ascending. The
corrupt block is contiguous in *that* order and in no other — in address order it shows
up as two separate holes — it lies inside a single super-group, and it always ends at
`s = 2A-2`, two emissions short of that group's last. Its start varies run to run, from
two emissions to most of the group. Every failure over 40 runs of one shape had that
shape.

**So it is detected and redone, not avoided.** Two adjacent emissions coming back zero is
the smallest corruption seen and is implausible as arithmetic — eight output channels at
one pixel, all exactly zero — so it is the signal, and the row task is resubmitted.
`r76_i32_wide_suspect()` in `rocket_matmul_rk3576.c` is that check. The redo
needs **no idle in front of it**, which is further evidence this is not the poisoning; the
sentinel's other reading, an atom that still holds it, *is* the poisoning and that redo
keeps its idle. Over 40 runs of the worst shape: 0 failures, 3 runs redoing one task.

The narrow writer is also the negative control for the wide one: a result that differs
between the two says one of the output maps is wrong and takes the arithmetic out of the
question. [HW sweep, H96 MAX M9, measured 2026-07-27]

**What has been eliminated.** At `M=8, N=32` on an 8x1 plane the baseline is 64 words:

| driven | extent | what it did |
|---|---|---|
| nothing (widths 0, 4, 5) | 64 words, last at 63 | the baseline; identical at all three widths |
| `0x402C` and `0x4030` high half, oc x4 | 64, last at 63 | nothing at all |
| `0x401C` (DST_SURF) doubled | 64, last at **95** | spaced the blocks apart — it is the block STRIDE |
| `0x40B8` (SURFACE_ADD) doubled | 64, last at 63 | inert at `ih=1`, where the task window is the plane |
| `0x401C` **and** `0x40B8` doubled | **0 words** | the DPU wrote nothing, so the two are coupled and `0x40B8` is live |
| `0x4030` low half 0x710 -> 0x310 | 64, last at 63 | nothing (the 7-vs-3 the RK3588 calls `size_e`) |
| `0x4030` low half -> 0xF10 | 64, last at 63 | corrupted the VALUES without moving the extent |
| CNA `0x1024` (kernel word \| oc-1) x4 | 64, last at 63 | nothing |
| DPU_RDMA `0x5014` (CUBE_CHANNEL) x4 | 64, last at 63 | nothing |
| CORE `0x3020` (dataout_channel) x4 **alone** | **0 words** | killed the write outright — it is coupled to the others, not inert |
| all five oc carriers x4 together | **256**, the full surface | the byte budget scales, the delivered channels do NOT — still 8 |

So the delivered-channel rule survives every oc register on the part. Scaling them
together buys address extent and nothing else, which is what says the rule is in the
datapath rather than in a geometry field. [HW sweep, H96 MAX M9, measured 2026-07-27]

### An i32out job poisons the next submit, and a POWER CYCLE is what clears it

**After an i32out job, the next submit of ANY kind writes nothing.** The job that does
not write **completes normally** — no fault, no IOMMU error, no timeout, the usual
~1.4 ms — and leaves its output buffer untouched. If the caller zeroed that buffer it
reads as an all-zero surface; if the caller reused one it reads as the PREVIOUS result,
byte for byte. Either way it looks like an arithmetic failure, and it is not one. It
crosses processes: the plain int8 matmul in another program is as exposed as the path
that caused it, because the state lives in the NPU and not in the fd.

**What clears it is the driver's runtime-PM autosuspend cycling the NPU power domain,
not elapsed time.** Three measurements say so, and together they close it:

- Write `on` to the device's `power/control`, so runtime suspend never fires, and **no
  gap clears it at all** — 100 ms, 300 ms and 600 ms each write nothing.
- The working gap tracks `power/autosuspend_delay_ms` one for one. At a 5 ms delay a
  5 ms gap already gives 12 writes of 12; at 50 ms it takes 80-120 ms for the same.
- `power/runtime_suspended_time` advances across the gap, so a suspend really did happen.

So the "50 to 100 ms of idle" this first read as is that file's stock value of 50 plus
the suspend/resume round trip, and nothing about the silicon. The cost of the workaround
is therefore **settable**: lowering the delay shortens the path proportionally — the
matmul gate's int32 half runs in 553 ms at the stock 50 and 195 ms at 5 — and a
kernel-side reset of the DPU at job start would remove it outright.

**What poisons is DPU `0x4010`, the DATA_FORMAT word.** Driving its output WIDTH field to
int32 does it; driving its PROC_PRECISION field to int32 does it separately; pinning
OUT_CVT to unity, which is the other thing an i32out program changes, does not; and no
plain int8 job does. The next job programs `0x4010` back to all-int8 and still comes back
empty, so this is latched datapath state rather than a stale register.

Nothing about the program or the buffers changes it. Fresh buffers per submit, a
different feature address, a different output address, and packing every K slice up front
so no buffer is rewritten between submits were each tried, and none helps.
`rocket_matmul_int8_rk3576_i32()` sizes its idle from the driver's autosuspend delay
(twice it plus 20 ms), idles between its own submits and once more on the way out, and
retries any submit whose own region came back untouched.

**Check that a job wrote PER TASK, not per tile.** A row-split tile issues several
submits into one output buffer, so one poisoned submit among them leaves its own rows
empty while its siblings are full — and a check over the whole tile then reads "something
was written" and hands the hole to the caller. It shows up as a result that is exactly
one row task short, intermittently, with no fault and normal timing. Both writers need
this; it was found on the narrow one, which had run clean for a session because no gate
shape until then split a tile into row tasks.

**The cost is settable from the driver, and it is worth about 1.7x.** The idle is sized
from `power/autosuspend_delay_ms` (twice it plus 20 ms), so lowering that file shortens
the whole int32 path proportionally. Measured over the matmul gate's int32 half, three
repetitions each, correct at every value:

| `autosuspend_delay_ms` | int32 gate half | conv gate (does not idle) |
|---|---|---|
| 50 (stock) | 10.7 / 10.7 / 11.1 s | 5.1 s |
| 20 | 7.8 s | 4.7 s |
| 10 | 7.3 s | 4.7 s |
| 5 | 6.5 / 6.7 / 6.5 s | 4.6 s |

Nothing else measurably pays for it: the conv gate, which never idles, is unchanged or
slightly faster at the low delay. The domain cycles more often under any NPU load, so
this is a system-wide setting rather than a library one — but on a box doing NPU work it
is close to free. [HW sweep, H96 MAX M9, measured 2026-07-27]

**The kernel-side fix that would remove the idle entirely is identified but not built.**
`rocket_core_reset()` already exists in the driver (a `reset_control` assert, `udelay(10)`,
deassert); calling it at job start is the shape of it. The vendor `rknpu` driver is not
the model here — it soft-resets only on a timeout or an abort, never per job. Note also
that the **~1.4 ms per-submit floor on this part is the driver's poll, not the silicon**:
PC_DONE is not routable to the GIC on the RK3576, so `patches/rk3576/npu/0006` polls it
on a 1 ms hrtimer where the RK3588 takes an interrupt. Every RK3576 cost table is a
submit-count table because of that.

**The hazard reaches the PLAIN int8 matmul too, and that path used to pass it straight
through.** The poisoning is not confined to the int32 entry that creates it: it outlives
the call and the process, so `rocket_matmul_int8_rk3576()` inherits it from whatever ran
before, and its first submit comes back untouched while the caller reads a correctly
sized, entirely stale tile. Measured in a soak, one full matmul-gate run in twenty failed
its very first shape this way, at 1 element of 256 correct. That entry now stamps its
output surface and redoes any row task whose region is still the stamp; 20 further gate
runs were clean. [HW sweep, H96 MAX M9, measured 2026-07-27]

**Do not zero an output buffer from the CPU before a submit** — but a bracketed stamp is
a different thing and is worth having. A bare `memset` leaves dirty cache lines that race
the DPU's DMA, and the writeback lands on top of the result, so the surface comes back all
zeros, intermittently and on the first submit as readily as a later one. A fill bracketed
by `PREP_BO` and `FINI_BO` is written back before the submit and leaves nothing dirty;
both matmul entries use one, and it is what makes "this was never written" a property of
the surface instead of an inference from its value. A freshly allocated BO arrives zeroed,
which cannot distinguish an unwritten word from a legitimate zero; a guard band past the
surface wants 64-byte alignment so no line it dirties is shared with a surface byte.

The int32 sweep also **wedged the IOMMU** once (stall timeout plus a
`rocket_job_irq_handler` WARN at `rocket_job.c:528`), reboot-only. Do not read that as an
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
just between tasks, and keep the pacing outside the measurement** — an unpaced job
completes without writing at all, which reads as an arithmetic failure and not as a
scheduling one.
