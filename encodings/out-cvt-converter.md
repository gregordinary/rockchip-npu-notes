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

- `SCALE` is a signed 16-bit integer multiplier, not fp16 and not fixed-point:
  `SCALE=2 -> ×2`, `SCALE=256 -> ×256`, exactly [HW sweep]. Bit 15 is its sign: `0x8000`
  multiplies by -32768 on the RK3588 and the RK3576 [HW sweep 2026-09-27]. A requant
  multiplier therefore has 15 bits.
- `SHIFT` is an integer right-shift in the integer domain, applied before any float cast.
  It rounds to nearest (see the tie rule below).
- `minus_exp` (bits[19:12]) and `cvt_type` are no-ops on the integer accumulator path.
  They matter only on the LUT/EW float datapath (below).
- The BN-MUL operand (`DPU_BN_MUL_CFG[31:16]`) is likewise an integer multiply with the
  operand read as uint16 (`0x3800 -> ×14336`), redundant with `SCALE` here.

The integer convert is the QNNPACK requantization form (15-bit multiplier + shift +
zero-point offset -> int8/int16).

With an int8 output the order is `sat8(round_half_to_even(acc * SCALE >> SHIFT) + OFFSET)`:
the offset is added after the rounding shift, and saturation comes after the offset. The int8
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

The rule is measured on both SoCs [HW sweep 2026-09-23]. The bit is bit 30 of `0x4088` on the
RK3588, and bit 30 of `0x40B4` on the RK3576.

The vendor's programs clear the bit: none of the 206 `OUT_CVT_SHIFT` writes in the RK3576
vendor-capture corpus sets bit 30. The rule in effect is therefore half to even.

> [!NOTE]
> Neither setting is the round-half-up that `(x + half) >> shift` gives. A CPU model that
> spells the requant that way disagrees with the hardware at a tie.

Ties are rare under the derivation below. It ends in `+1`, so `MUL` is odd for every round
scale, including every power of two. An odd multiplier moves each exact half of
`acc * scale` outward, off the tie, so the rounder never sees one. A tie needs a deliberately
chosen scale, one whose top 14 mantissa bits are all ones. There the `+1` carries out, and
`MUL` is exactly `2^14`.

The two rules differ on a small share of elements. With `MUL` odd, ties of `acc * MUL` are one
accumulator residue in `2^SHIFT`. The two rules differ on half of those, about
`2^-(SHIFT+1)` of an output. At a typical `SHIFT` of 15-20 that is nothing in a small test
case and tens of elements in a large prefill. Those elements are one count each, sparse,
and present in every configuration. They are the standing noise that makes single-element
int8 disagreements unreadable.

**Not characterized:** the tie rule of the EW stage's input converter (`EW_CVT`), a separate
stage. One INT16 `EW_CVT` probe on an RK3588 returned `[2, 5, 8, 11, 14, 17, 20, 23]` where
half to even gives `[2, 6, 8, 12, 14, 18, 20, 24]`, consistent with ties rounding down. The
board crashed later in the same probe sequence [external: allbilly/rk3588 at `e13d99f`].

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

> [!CAUTION]
> Erratum E6. Mesa's copy of the derivation drops that carry [source-confirmed: Mesa
> `rkt_regcmd.c`]. Where mantissa bits [22:9] are all set, the hardware applies half the scale
> or the scale negated (table below). When software derives the scale, it must keep the carry.

Mesa masks `(bits >> 9) & 0x7fff`, which takes the exponent's low bit as the multiplier's bit
14, and forces bit 14 afterwards. The two derivations agree except where mantissa bits [22:9]
are all set:

| Biased exponent | Mesa's `MUL` | Gain the hardware applies |
|---|---|---|
| even | `0x4000` at the exponent's shift | half the scale |
| odd | `0x8000` | the scale negated |

Both rows are measured on the RK3588 and the RK3576 [HW sweep 2026-09-27]. One scale in 16384
is on an edge. No per-tensor scale in nine detector and classifier `.tflite` models sits on one
[host-computed 2026-09-27].

> [!NOTE]
> A host model built on the same derivation agrees with the hardware at the wrong gain. Only a
> comparison against the float scale detects the defect.

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
  the int32 accumulator must fit fp16. The fp16 output therefore does not reduce the large-K
  LLM readback.

### Direct int8-output program

**Conditions:** RK3588 (Turing RK1). Kernels 1, 3 and 5, stride 1 and 2, CNA pad 0-2, IC
32-96, OC 32-160 and a 64×60×56 plane.

The direct program writes int8 through the requant. Mesa's direct int8-output writer takes
the int32-raw direct program's CNA and CORE [source-confirmed: Mesa `rkt_regcmd.c`,
`rkt_task.c`]. It sets `CORE_MISC_CFG.QD_EN` 1, `DPU_DATA_FORMAT` 0, `size_e` 1 and
`SURF_ADD` = OH*OW*2. The per-OC int32 bias goes in the BS stage through BRDMA, and the
OUT_CVT requantizes. Its int8 C2=16 cube matches a host model of TFLite's accumulator and this
requant on every element [HW sweep, RK1].

The depthwise writer's geometry (`size_e` 3, `SURF_ADD` x4) times out on this program.

> [!CAUTION]
> A direct program whose OC ends part way through a 32-kernel group does not complete. At OC
> 16 and 48, packed as Mesa packs a partial group, each task outlasts the driver's watchdog.
> That group is unwritten while the whole groups are exact. Software must program whole
> groups, with zero kernels in the padding, and the program then completes.

## Consequence for int8 dequant

A fractional W8A8 dequant scale (`acc * a_scale[m] * b_scale[n]`, both < 1) cannot fold
into OUT_CVT to produce a fractional float. `(acc*scale)>>shift` always yields an
integer-valued float (the shift rounds the fraction off). The consequences:

- The host per-row × per-channel dequant stays.
- The way to reduce int8 output readback is a bigger Kt (fewer K-partials to read), not
  OUT_CVT.
- OUT_CVT is the right tool for int8->int8/int16 requant (integer output), and for the
  fp32 cast and the per-tensor integer gain.

Related: [precision-field.md](precision-field.md), [size-e-quirk.md](size-e-quirk.md), and
[k-accumulation.md](k-accumulation.md) (int8 EW K-accumulation: unestablished, and not a
speed optimization).

## Per-channel (per-axis) requant on the BS multiplier

**Conditions:** RK3588 (Turing RK1), 600 MHz, 2026-09-27. The int8-out depthwise program, at
two shapes, one a stride-2 plane.

The output requant has one scale and one shift per task. A per-output-channel scale therefore
goes on the BS stage's multiplier, which reads a per-channel operand [HW sweep]:

| Register | Value | Function |
|---|---|---|
| `DPU_RDMA_BRDMA_CFG` `0x501C` | `BRDMA_DATA_USE` 7 (`0xE`) | BRDMA reads a 64-byte group per 8 output channels: int32 `A[8]` at 0, int16 `B[8]` at 32, int16 `C[8]` at 48. Data use 1 reads a dense int32 bias alone |
| `DPU_BS_CFG` `0x4040` | `0x20140` | the ALU adds `A` from DMA, the multiplier is live (`BS_MUL_BYPASS` clear) |
| `DPU_BS_MUL_CFG` `0x4048` | `(s << 8) \| 1` | `BS_MUL_SRC` reads `C` per channel. `[13:8]` shifts a **non-negative** product |
| `DPU_DATA_FORMAT` `0x4010` | `[9:4]` = s | `BS_MUL_SHIFT_VALUE_NEG` shifts a **negative** product |

The stage computes:

```
v = sat32( rne( ((acc + A[c]) * C[c]) >> (p >= 0 ? s : s_neg) ) ),   p = (acc + A[c]) * C[c]
```

`v` then goes through the unchanged OUT_CVT. The stage behaves as follows [HW sweep]:

- The add comes before the multiply.
- The product is held wide under a shift.
- At `s` 0 the product saturates at int32.
- The shift rounds half to even, like the OUT_CVT's.
- The two shift fields split by sign. `[13:8]` shifts a non-negative product, and `[9:4]` a
  negative one.
- `B` is a per-channel CPEND. With `DPU_BS_OW_CFG.OW_SRC` (bit 0) set, the stage computes
  `(acc + A + B[c]*S) * C`, `S` the window sum of the CNA's input, border pad included. With
  `OW_SRC` clear, `B` is not read. The per-tensor form is `DPU_BS_OW_OP`
  ([depthwise-conv.md](../depthwise-conv.md)). The direct int8-out program computes the same
  model for both, `S` summing every programmed input channel [HW sweep, RK1].

> [!CAUTION]
> Erratum E6. The BS stage's shift is split by sign across two fields. With only
> `BS_MUL_CFG[13:8]` set, negative products come back at the wrong scale. Software must
> program both shift fields: `BS_MUL_CFG[13:8]` for non-negative products and
> `DATA_FORMAT[9:4]` for negative ones. On the RK3576, both fields are in the per-task operand
> at `0x5024`.

> [!NOTE]
> An output zero point at the rail hides the negative half. At -128 every negative result
> saturates to the same byte, so a program with the negative shift wrong passes there.

The RK3576's BS stage computes the same epilogue, held wide and rounded half to even. Its shift
is in the per-task operand at `0x5024` and is split by sign there too [HW sweep, H96]. Its
depthwise group is 48 bytes with no `B`
([../chips/rk3576-regcmd.md](../chips/rk3576-regcmd.md)).

### Cost of a per-channel scale

TFLite's per-axis requant is
`out = zp + MultiplyByQuantizedMultiplier(acc, mult[c], shift[c])`: a 31-bit multiplier and
a shift per channel. Here a task has one OUT_CVT gain `G` and one BS shift `s`. Channel `c`
reaches its scale `M[c]` through `C[c] = round(M[c] * 2^s / G)`, an int16, and resolves its
gain to `0.5/C[c]`. The task's scale spread therefore sets `C` for its smallest channel.

A channel far below its task's largest scale gets a small `C`. A channel with an all-zero
filter has its bias for an accumulator at every output, so its plane is one value.

### BN stage per-channel multiplier

The BN stage's per-channel multiplier is not established.

> [!CAUTION]
> One configuration hangs every task: `BN_MUL_SRC`, `NRDMA_DATA_USE` 7 over the same 64-byte
> layout, the BN ALU reading from DMA. Each task runs to the driver's watchdog and writes
> nothing, and the tasks after it are exact [HW sweep, RK1, 2026-09-27].

**Not characterized:** other data-use values, the BN ALU bypassed, and another operand layout.

A live BN multiplier is expected to provide a per-channel power of two beside the BS mantissa
[expected].

### CNA per-channel converter

The per-channel converter in the `0x1xxx` domain (`CVT_CON0` `0x104C` `CVT_TRUNCATE_0..3`,
`CVT_CON5` `0x1180` `PER_CHANNEL_CVT_EN`) is the CNA input path. It normalizes features
and weights as they stream into CBUF, and is not an output requant.

### Consequence for int8 activations resident between conv ops

An on-chip requant, per-tensor or per-channel, differs from TFLite's by at most one per op.
Chained, it drifts the way TFLite's own optimized kernels drift from its reference ones. In
a detector, 12-14% of the elements end two or more counts off either way
([depthwise-conv.md](../depthwise-conv.md) §"The int8 domain"). Keeping int8 activations
resident is therefore judged per op by that bound, and along a chain by an accuracy measure
such as COCO mAP.

## Software support

### Requant entries

`gen_conv2d_int8_fill(int8_out=1)` uses the integer convert to emit requantized int8, within
one of TFLite at a rounding boundary ([depthwise-conv.md](../depthwise-conv.md)).

Every entry in rocket-userspace writes `CVT_ROUND` 0, so its rule is half to even. Its CPU
requant model has the half-to-even rule. `ROCKET_OUT_CVT_ROUND=1` sets the bit on every
integer `OUT_CVT_SHIFT` write, to test the tie rule.

rocket-userspace derives every pair through one function, `npu_out_cvt_pair()`, which keeps the
carry.

The fp16 output of the int8 datapath is not available in rocket-userspace.

### Direct int8-output program

`rocket_conv2d_int8_q` in rocket-userspace programs whole groups, with zero kernels in the
padding. Through `tflite-rocket` it runs MobileDet and SSD MobileNet v2 1.19-1.27x faster than
the int32-raw path [HW sweep, RK1].

### Per-axis depthwise entry

The planner puts a task's largest channel at `C` = 32767. It takes the smallest `s` whose
product fits int32 at every channel's accumulator bound, which keeps `G` as fine as that bound
allows. The result is within one count of TFLite where a value sits near a rounding boundary,
as the per-tensor requant is. At six test shapes the device matches a host model of that
arithmetic on every element [HW sweep, RK1]. It differs from TFLite's per-channel arithmetic by
one on 0.017-0.035% of elements, never by more.

The library's `rocket_conv2d_dw_int8_perc` sorts channels by scale, and ends a task where the
next channel's `C` would fall under 2048. A wide spread then costs tasks rather than
resolution.

Through `tflite-rocket`, EfficientDet-Lite0's 76 per-axis depthwise layers run on this entry
[HW sweep, RK1, 600 MHz, 2026-09-27]. The model runs 1.095-1.104x warm, 133 -> 121 ms. COCO
mAP over 100 images is 0.3019 against the CPU's 0.2996. The depthwise term falls from 43.2 to
34.7 ms of the op sum, and most of that is the delegate's host stages around the call. The
library call itself moves 28.6 -> 26.8 ms. Most of these layers are small enough that the fixed
per-call overhead sets their cost.

## Evidence

### Integer convert

A ratio classifier confirms the `SCALE` multiplier on hardware. The sign bit is measured on
the RK3588 and the RK3576, 64 of 64 elements at two scales each, with
`tests/requant_edge_probe.c` (2026-09-27).

### The `CVT_ROUND` tie rule

The test is `tests/requant_round_probe.c` (2026-09-23). Each run leaves exactly one of five
candidate rules (half up, half away, half even, truncation, floor) consistent with every
element. The runs per SoC:

- **RK3576**: 160 tie elements per setting at two shifts, both signs, through
  `rocket_matmul_int8_rk3576` [H96 MAX M9].
- **RK3588**: 128 tie elements per setting at shifts 1 and 3, both signs [Turing RK1 at
  600 MHz]. The test runs through the int8 matmul's integer convert
  (`ROCKET_INT8_DEQ=1`, `CVTTYPE=0`, `SCALE=1`), whose fp32 output holds the rounded value
  exactly.

No ordinary test reaches a tie. `requant_round_probe` picks a scale whose top 14 mantissa bits
are all ones, and asserts the rule on both SoCs.

### Derivation of MUL and SHIFT from a float scale

Each row is measured at two scales, with `tests/requant_edge_probe.c` (2026-09-27). The test is
a per-tensor int8-out entry with zero operands, every element scored against
`rint(acc * scale)`. The `.tflite` check is a census of the models' per-tensor scales.

### Direct int8-output program

The measurement is `tests/conv_i8out_probe`.

### Consequence for int8 dequant

The test is `tests/matmul_int8_dequant_rocket.c`.

### Per-channel (per-axis) requant on the BS multiplier

`tests/dw_perc_probe.c` scores every element of every test case against each candidate model.
The unpatched program is exact before and after the test cases. Each clause is separated by a
test case that only it explains:

- **The add comes before the multiply.** `(acc + A)*C` explains all 9216 elements, and
  `acc*C + A` explains 1-3 thousand.
- **The product is held wide under a shift.** At `s` 20, 6625 of 9216 products are past 2^31.
  Every element matches the wide product, and saturating first explains 2624.
- **At `s` 0 the product saturates at int32.** With 7413 of 9216 products past 2^31, only the
  saturating model explains every element.
- **The shift rounds half to even.** On small operands at `s` 8 with an identity OUT_CVT,
  half-to-even explains 9216 elements and half-up 9174.
- **The two shift fields split by sign.** With `[13:8]` at 14 and `[9:4]` at 6, and the
  reverse, the sign-split model explains every element on both shapes. Either field alone
  explains only its own sign's elements (4866 + 4350 = 9216).
- **`B` is a per-channel CPEND.** With `OW_SRC` set, `(acc + A + B[c]*S) * C` explains every
  element.

The RK3576 test sets `C` 16384 and fields 14 and 12, in both orders
(`tests/rk3576_coeff_c shiftsign`). The sign-split model explains 32 of 32 channels, and either
field alone explains 16-17.

### BN stage per-channel multiplier

All 8 tasks over two shapes ran to the 500 ms watchdog and wrote nothing. No reference program
found here enables NRDMA.

### Software support

The CPU model is `tests/requant_model.h`. The per-axis test is
`tests/conv_dw_int8_perc_runtime.c`. The MobileDet and SSD MobileNet v2 measurement is pinned,
over four rotated passes. The EfficientDet-Lite0 measurement has the governor pinned, on the
A76 cores, over four rotated passes.
