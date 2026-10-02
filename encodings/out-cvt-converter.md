# DPU OUT_CVT, the output converter (int32 accumulator -> output)

The last stage of the DPU before write-back is the output converter, from the NVDLA SDP
lineage. NVDLA documents it as `y = sat((x − offset) * scale >> shift)`. On the RK3588's
integer path the offset is added after the shift instead (below). Three registers drive
it:

| reg | addr | fields |
|---|---|---|
| `DPU_OUT_CVT_OFFSET` | `0x4080` | signed 32-bit offset, added after the shift on the integer path: the output zero point |
| `DPU_OUT_CVT_SCALE`  | `0x4084` | `[15:0]` scale (multiplier), signed 16-bit. `[16]` `FP32TOFP16_EN` |
| `DPU_OUT_CVT_SHIFT`  | `0x4088` | `[5:0]` integer shift. `[19:12]` `minus_exp`. `[30]` `CVT_ROUND`, the tie rule. `[31]` `cvt_type` |

## Operating modes (cvt_type)

### Integer convert

This mode is the matmul and conv accumulator path. Acting on the raw int32 CACC
accumulator, the converter is purely integer:

```
out = (float_or_int)( round_half_to_even( (acc_i32 * SCALE) >> SHIFT ) )
```

- `SCALE` is a signed 16-bit integer multiplier, not fp16 and not fixed-point. The ratio
  classifier confirms it on hardware: `SCALE=2 -> ×2`, `SCALE=256 -> ×256`, exactly. Bit 15
  is its sign: `0x8000` multiplies by -32768 on both parts, 64 of 64 elements at two scales
  each [HW sweep, `tests/requant_edge_probe.c`, 2026-09-27]. A requant multiplier therefore
  has 15 bits.
- `SHIFT` is an integer right-shift in the integer domain, applied before any float cast.
  It rounds to nearest (see the tie rule below).
- `minus_exp` (bits[19:12]) and `cvt_type` are no-ops on the integer accumulator path.
  They matter only on the LUT/EW float datapath (below).
- The BN-MUL operand (`DPU_BN_MUL_CFG[31:16]`) is likewise an integer multiply with the
  operand read as uint16 (`0x3800 -> ×14336`), redundant with `SCALE` here.

This is the QNNPACK requantization form (15-bit multiplier + shift + zero-point offset ->
int8/int16). `gen_conv2d_int8_fill(int8_out=1)` uses it to emit requantized int8, within
one of TFLite at a rounding boundary ([depthwise-conv.md](../depthwise-conv.md)).

With an int8 output the order is `sat8(round_half_to_even(acc * SCALE >> SHIFT) + OFFSET)`: the
offset lands after the rounding shift, and saturation comes after the offset. The int8
depthwise path is bit-exact against that model at output zero points from -128 to 127, at two
scales [HW sweep, RK1, measured 2026-09-23]. Under the NVDLA order, subtracting before the
multiply, the zero point scales down to a fraction of one.

### The `CVT_ROUND` tie rule

`acc*SCALE >> SHIFT` rounds to nearest, and bit 30 of `OUT_CVT_SHIFT` decides where an
exact half goes:

| `CVT_ROUND` | Tie rule | 0.5, 1.5, -0.5, -1.5 go to |
|---|---|---|
| 0 | half to even, QNNPACK's *precise* requantization | 0, 2, 0, -2 |
| 1 | half away from zero, the NVDLA documentation's rule and TFLite's | 1, 2, -1, -2 |

The rule is measured on both parts. Each run leaves exactly one of five candidate rules
(half up, half away, half even, truncation, floor) consistent with every element [HW sweep,
`tests/requant_round_probe.c`, 2026-09-23]:

- **RK3576**, bit 30 of `0x40B4`: 160 tie elements per setting at two shifts, both signs,
  through `rocket_matmul_int8_rk3576` [H96 MAX M9].
- **RK3588**, bit 30 of `0x4088`: 128 tie elements per setting at shifts 1 and 3, both
  signs [Turing RK1 at 600 MHz]. The arm runs through the int8 matmul's integer convert
  (`ROCKET_INT8_DEQ=1`, `CVTTYPE=0`, `SCALE=1`), whose fp32 output carries the rounded
  value exactly.

**Every entry in this tree writes 0, and so do the vendor's programs.** None of the 206
`OUT_CVT_SHIFT` writes in the RK3576 vendor-capture corpus sets bit 30. So the shipped rule
is half to even. Neither setting is the round-half-up that `(x + half) >> shift` gives, so
a CPU model that spells the requant that way disagrees with the part at a tie.
`tests/requant_model.h` is the one model and carries the half-to-even rule.
`ROCKET_OUT_CVT_ROUND=1` sets the bit on every integer `OUT_CVT_SHIFT` write, for the
probe.

Reaching a tie needs a deliberately chosen scale. The derivation below ends in `+1`, so
`MUL` is odd for every round scale including every power of two. An odd multiplier moves an
exact half off the tie in the outward direction, so the rounder never sees one. So no
ordinary gate reaches the case. `requant_round_probe` does: it picks a scale whose top 14
mantissa bits are all ones, where the `+1` carries out and `MUL` is exactly `2^14`. It
asserts the rule on both parts.

The two rules differ on a small share of elements. With `MUL` odd, ties are one
accumulator residue in `2^SHIFT`, and the two rules differ on half of those: about
`2^-(SHIFT+1)` of a surface. At a typical `SHIFT` of 15-20 that is nothing in a small gate
case and tens of elements in a large prefill. Those elements are one count each, sparse,
and present in every configuration. They are the standing noise that makes single-element
int8 disagreements unreadable.

### Float-affine convert

This mode is the LUT-activation path. When the converter's input is already a Q-format
value in the float/EW datapath (for example a LUT output `q`), `cvt_type=1` selects
`out = (q + offset) * 2^-minus_exp`, narrowed to fp16 by `FP32TOFP16_EN`. See
[dpu-lut-activation.md](dpu-lut-activation.md). `q` is not the raw integer accumulator,
which is why `minus_exp` is inert on the plain matmul path.

## Derivation of MUL and SHIFT from a float scale

A float requant scale becomes the pair by QNNPACK's derivation. `MUL` is the scale's implicit
bit and its top 14 mantissa bits, plus one, and `SHIFT` is 141 minus the biased exponent, the
register value. The gain `MUL / 2^SHIFT` then lies in `[scale, scale * (1 + 2^-14)]` at every
scale. Where the `+1` carries out of the 15 bits, `MUL` is `0x4000` at one less shift, the same
gain.

**Mesa's copy of the derivation drops that carry** [source-confirmed, `rkt_regcmd.c`]. It masks
`(bits >> 9) & 0x7fff`, which takes the exponent's low bit as the multiplier's bit 14, and
forces bit 14 afterwards. The two agree except where mantissa bits [22:9] are all set:

| Biased exponent | Mesa's `MUL` | Gain the part applies |
|---|---|---|
| even | `0x4000` at the exponent's shift | half the scale |
| odd | `0x8000` | the scale negated |

Both rows are measured on both parts [HW sweep, RK3588 and RK3576, two scales per row,
`tests/requant_edge_probe.c`, 2026-09-27]. The arm is a per-tensor int8-out entry with zero
operands, every element scored against `rint(acc * scale)`. One scale in 16384 is on an edge.
No per-tensor scale in nine detector and classifier `.tflite` models sits on one [host census,
2026-09-27].

A host model built on the same derivation agrees with the part at the wrong gain. Only a
comparison against the float scale sees the defect. rocket-userspace derives every pair
through one function, `npu_out_cvt_pair()`, which keeps the carry.

## Output datatype and cube on the int8 (int32-acc) datapath

| out_precision | writer geom (size_e / surf_add) | cube C2 | notes |
|---|---|---|---|
| int32 | `7` / `stride*8` | 4 | the default raw-accumulator readback |
| fp32  | `7` / `stride*8` | 4 | bit-exact cast of the int32 acc (`ROCKET_INT8_FP32_OUT`) |
| fp16  | `3` / `stride*2` | 8 | small single-tile shapes only, range-limited `\|acc\|<=2048` |
| int8, direct, `QD_EN` 1, OUT_CVT requant | `1` / `stride*2` | 16 | Mesa's direct program. Bit-exact to a host model at whole 32-kernel groups |

- **fp32 cast + integer scale** fold cleanly and generally (any shape, bit-exact).
- **fp16** halves the output readback, with two limits. Its writer geometry is correct only
  for small single-tile shapes (wrong or strided values at, for example, 64×128×64), and
  the int32 accumulator must fit fp16. So it does not help the large-K LLM readback. It is
  not shipped.

### Direct int8-output program

The direct program writes int8 through the requant. Mesa's direct int8-output writer takes
the int32-raw direct program's CNA and CORE [source-confirmed, `rkt_regcmd.c`, `rkt_task.c`]. It
sets `CORE_MISC_CFG.QD_EN` 1, `DPU_DATA_FORMAT` 0, `size_e` 1 and `SURF_ADD` = OH*OW*2. The per-OC
int32 bias goes in the BS stage through BRDMA, and the OUT_CVT requantizes. Its int8 C2=16 cube
matches a host model of TFLite's accumulator and this requant on every element [HW sweep, RK1,
`tests/conv_i8out_probe`]. The shapes are kernels 1, 3 and 5, stride 1 and 2, CNA pad
0-2, IC 32-96, OC 32-160 and a 64×60×56 plane.

The depthwise writer's geometry (`size_e` 3, `SURF_ADD` x4) times out on this program.

**A direct program whose OC ends part way through a 32-kernel group does not complete.** At OC 16
and 48, packed as Mesa packs a partial group, each task outlasts the driver's watchdog. That group
is unwritten while the whole groups are exact. Programmed at whole groups with zero kernels in the
padding, it completes. `rocket_conv2d_int8_q` in rocket-userspace does that. Through
`tflite-rocket` it runs MobileDet and SSD MobileNet v2 1.19-1.27x faster than the int32-raw route
[HW sweep, RK1, pinned, four rotated passes].

## Consequence for int8 dequant

A fractional W8A8 dequant scale (`acc * a_scale[m] * b_scale[n]`, both < 1) cannot fold
into OUT_CVT to produce a fractional float. `(acc*scale)>>shift` always yields an
integer-valued float (the fraction is truncated). So:

- The host per-row × per-channel dequant stays.
- The int8 output-readback lever is bigger-Kt (fewer K-partials to read), not OUT_CVT.
- OUT_CVT is the right tool for int8->int8/int16 requant (integer output), and for the
  fp32 cast and the per-tensor integer gain.

The gate is `tests/matmul_int8_dequant_rocket.c`. Related:
[precision-field.md](precision-field.md), [size-e-quirk.md](size-e-quirk.md), and
[k-accumulation.md](k-accumulation.md) (int8 EW K-accumulation: unestablished, and not a
speed lever).

## Per-channel (per-axis) requant on the BS multiplier

The output requant carries one scale and one shift per task. A per-output-channel scale
therefore goes on the BS stage's multiplier, which reads a per-channel operand. The
measurement is on the RK3588's int8-out depthwise program at two shapes, one a stride-2
plane [HW sweep, RK1, 600 MHz, 2026-09-27, `tests/dw_perc_probe.c`]. The probe scores every
element of every arm against each candidate model. The unpatched program is exact before
and after the arms.

| Word | Value | What it does |
|---|---|---|
| `DPU_RDMA_BRDMA_CFG` `0x501C` | `BRDMA_DATA_USE` 7 (`0xE`) | BRDMA reads a 64-byte group per 8 output channels: int32 `A[8]` at 0, int16 `B[8]` at 32, int16 `C[8]` at 48. Data use 1 reads a dense int32 bias alone |
| `DPU_BS_CFG` `0x4040` | `0x20140` | the ALU adds `A` from DMA, the multiplier is live (`BS_MUL_BYPASS` clear) |
| `DPU_BS_MUL_CFG` `0x4048` | `(s << 8) \| 1` | `BS_MUL_SRC` reads `C` per channel. `[13:8]` shifts a **non-negative** product |
| `DPU_DATA_FORMAT` `0x4010` | `[9:4]` = s | `BS_MUL_SHIFT_VALUE_NEG` shifts a **negative** product |

The stage computes:

```
v = sat32( rne( ((acc + A[c]) * C[c]) >> (p >= 0 ? s : s_neg) ) ),   p = (acc + A[c]) * C[c]
```

`v` then goes through the unchanged OUT_CVT. Each clause is separated by an arm that only
it explains:

- **The add comes before the multiply.** `(acc + A)*C` explains all 9216 elements, and
  `acc*C + A` explains 1-3 thousand.
- **The product is held wide under a shift.** At `s` 20, 6625 of 9216 products are past 2^31.
  Every element reads the wide product, and saturating first explains 2624.
- **At `s` 0 the product saturates at int32.** With 7413 of 9216 products past 2^31, only the
  saturating model explains every element.
- **The shift rounds half to even**, like the OUT_CVT's. On small operands at `s` 8 with an
  identity OUT_CVT, half-to-even explains 9216 elements and half-up 9174.
- **The two shift fields split by sign.** With `[13:8]` at 14 and `[9:4]` at 6, and the reverse,
  the sign-split model explains every element on both shapes. Either field alone explains only
  its own sign's elements (4866 + 4350 = 9216). Set both, or negative products come back at the
  wrong scale.
- **An output zero point at the rail hides the negative half.** At -128 every negative result
  saturates to the same byte. A program with the negative shift wrong passes there.
- **`B` is a per-channel CPEND.** With `DPU_BS_OW_CFG.OW_SRC` (bit 0) set, `(acc + A +
  B[c]*S) * C` explains every element, `S` the window sum of the CNA's input, border pad
  included. With `OW_SRC` clear, `B` is not read. The per-tensor form is `DPU_BS_OW_OP`
  ([depthwise-conv.md](../depthwise-conv.md)). The direct int8-out program computes the same
  model for both, `S` summing every programmed input channel [HW sweep, RK1].

The RK3576's BS stage computes the same epilogue, held wide and rounded half to even. Its shift
is in the per-task operand at `0x5024` and is split by sign there too. The arm sets `C` 16384
and fields 14 and 12, in both orders [HW sweep, H96, `tests/rk3576_coeff_c shiftsign`, agent
run]. The sign-split model explains 32 of 32 channels, and either field alone explains 16-17.
Its depthwise group is 48 bytes with no `B`
([../chips/rk3576-regcmd.md](../chips/rk3576-regcmd.md)).

### Cost of a per-channel scale

TFLite's per-axis requant is
`out = zp + MultiplyByQuantizedMultiplier(acc, mult[c], shift[c])`: a 31-bit multiplier and
a shift per channel. Here a task has one OUT_CVT gain `G` and one BS shift `s`. Channel `c`
reaches its scale `M[c]` through `C[c] = round(M[c] * 2^s / G)`, an int16, and resolves its
gain to `0.5/C[c]`. So the task's scale spread sets `C` for its smallest channel.

The planner puts a task's largest channel at `C` = 32767. It takes the smallest `s` whose product
fits int32 at every channel's accumulator bound, which keeps `G` as fine as that bound allows.
The result is within one count of TFLite where a value sits near a rounding boundary, as the
per-tensor requant is. At six gate shapes the device matches a host model of that arithmetic on
every element [HW sweep, RK1, `tests/conv_dw_int8_perc_runtime.c`]. It differs from TFLite's
per-channel arithmetic by one on 0.017-0.035% of elements, never by more.

A channel far below its task's largest scale gets a small `C`. The library's
`rocket_conv2d_dw_int8_perc` sorts channels by scale and ends a task where the next channel's
`C` would fall under 2048. A wide spread then costs tasks rather than resolution. A channel with
an all-zero filter has its bias for an accumulator at every output, so its plane is one value.

Through `tflite-rocket`, EfficientDet-Lite0's 76 per-axis depthwise layers run on this entry
[HW sweep, RK1, 600 MHz, governor pinned, A76 cores, four rotated passes, 2026-09-27]. The
model runs 1.095-1.104x warm, 133 -> 121 ms. COCO mAP over 100 images is 0.3019 against the
CPU's 0.2996. The depthwise term falls from 43.2 to 34.7 ms of the op sum, and most of that is
the delegate's host stages around the call. The library call itself moves 28.6 -> 26.8 ms. Most
of these layers are small enough that the per-call floor sets their cost.

### BN stage per-channel multiplier

The BN stage's per-channel multiplier is not established. One configuration hung every task:
`BN_MUL_SRC`, `NRDMA_DATA_USE` 7 over the same 64-byte layout, the BN ALU reading from DMA. All
8 tasks over two shapes ran to the 500 ms watchdog and wrote nothing, and the tasks after them
were exact [HW sweep, RK1, 2026-09-27]. Other data-use values, the BN ALU bypassed, and another
operand layout are unsearched. No reference program found here enables NRDMA. A live BN
multiplier would carry a per-channel power of two beside the BS mantissa [expected].

### CNA per-channel converter

The per-channel converter in the `0x1xxx` domain (`CVT_CON0` `0x104C` `CVT_TRUNCATE_0..3`,
`CVT_CON5` `0x1180` `PER_CHANNEL_CVT_EN`) is the CNA input path. It normalizes features
and weights as they stream into CBUF, and is not an output requant.

### Consequence for int8 activations resident between conv ops

An on-chip requant, per-tensor or per-channel, differs from TFLite's by at most one per op.
Chained, it drifts the way TFLite's own optimized kernels drift from its reference ones. In
a detector, 12-14% of the elements end two or more counts off either way
([depthwise-conv.md](../depthwise-conv.md) §"The int8 domain"). So the lever is gated per op
by that bound, and along a chain by an accuracy measure such as COCO mAP.
