# DPU OUT_CVT, the output converter (int32 accumulator -> output)

The last stage of the DPU before write-back is the output converter, from the NVDLA SDP
lineage. NVDLA documents it as `y = sat((x − offset) * scale >> shift)`. On the RK3588's
integer path `cvt_type` selects where the offset enters (below). Three registers drive
it:

| reg | addr | fields |
|---|---|---|
| `DPU_OUT_CVT_OFFSET` | `0x4080` | signed 32-bit offset, the output zero point. `cvt_type` selects where it enters (§"Integer convert") |
| `DPU_OUT_CVT_SCALE`  | `0x4084` | `[15:0]` scale (multiplier), signed 16-bit. `[16]` `FP32TOFP16_EN` |
| `DPU_OUT_CVT_SHIFT`  | `0x4088` | `[11:0]` integer shift. `[19:12]` `minus_exp`. `[30]` `CVT_ROUND`, the tie rule. `[31]` `cvt_type`, the order of the multiply and the offset: 0 multiply first, 1 offset first [TRM] |

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
- `minus_exp` (bits[19:12]) is a no-op on the integer accumulator path. It matters only on
  the LUT/EW float datapath (below).
- The BN-MUL operand (`DPU_BN_MUL_CFG[31:16]`) is likewise an integer multiply with the
  operand read as uint16 (`0x3800 -> ×14336`), redundant with `SCALE` here.

The integer convert is the QNNPACK requantization form (15-bit multiplier + shift +
zero-point offset -> int8/int16).

`cvt_type` selects where the offset enters [HW sweep 2026-11-02]. At an offset of 0 the two
settings compute the same value:

| `cvt_type` | Integer convert with an offset |
|---|---|
| 0 | `round_half_to_even(acc * SCALE / 2^SHIFT + OFFSET)`: the offset is added at the output scale, after the multiply and the shift and before the rounding |
| 1 | `round_half_to_even((acc + OFFSET) * SCALE / 2^SHIFT)`: the offset is added before the multiply, the order the TRM gives |

**Conditions:** RK3588 (Turing RK1), `rocket` 1.3.0, NPU clock 928 MHz, 2026-11-02. An int8
matmul through the integer convert, fp32 output, `CVT_ROUND` 0.

At `cvt_type` 0, an offset added before the rounding and one added after it differ only at a
tie. The converter adds the offset to the truncated quotient, and then decides the tie
[HW sweep 2026-11-07]:

| `CVT_ROUND` | A tie of `acc * SCALE / 2^SHIFT` goes to | Equivalent model |
|---|---|---|
| 0, half to even | The even neighbor of the quotient plus the offset | `round_half_to_even(acc * SCALE / 2^SHIFT + OFFSET)` |
| 1, half away from zero | The neighbor away from zero by the accumulator's sign | `round_half_away(acc * SCALE / 2^SHIFT) + OFFSET` |

**Conditions:** RK3588 (Turing RK1), `rocket` 1.3.0, NPU clock 928 MHz, 2026-11-07. The direct
int8-output convolution program at `MUL` 2^14 and `SHIFT` 15 and 17, output zero points from -5
to 100, under each `CVT_ROUND` setting.

At `CVT_ROUND` 0 the two orders therefore differ at a tie with an odd offset. At `CVT_ROUND` 1
they differ at a tie where the accumulator and the offset-included sum have opposite signs.

With an int8 output, saturation comes after the offset, and the int8 writer follows the table
above. The int8 depthwise path is bit-exact against `sat8(round_half_to_even(acc * SCALE >>
SHIFT) + OFFSET)` at output zero points from -128 to 127, at two scales [HW sweep, RK1, measured
2026-09-23]. That model rounds before it adds the offset, so at `CVT_ROUND` 0 it is one count
off at a tie with an odd zero point. With an odd `MUL`, a tie is one accumulator residue in
`2^SHIFT` (§"The `CVT_ROUND` tie rule"), about `2^-SHIFT` of the outputs. Under the NVDLA
order, subtracting before the multiply, the zero point scales down to a fraction of one.

> [!NOTE]
> At the default `CVT_ROUND` 0, the int8 writer adds the zero point before it rounds. A CPU
> model that rounds first is one count off at a tie with an odd zero point.

### The `CVT_ROUND` tie rule

`acc*SCALE >> SHIFT` rounds to nearest, and bit 30 of `OUT_CVT_SHIFT` decides where an
exact half goes:

| `CVT_ROUND` | Tie rule | 0.5, 1.5, -0.5, -1.5 go to |
|---|---|---|
| 0 | half to even, QNNPACK's *precise* requantization | 0, 2, 0, -2 |
| 1 | half away from zero, the NVDLA documentation's rule and TFLite's | 1, 2, -1, -2 |

The rule is measured on both SoCs [HW sweep 2026-09-23]. The bit is bit 30 of `0x4088` on the
RK3588, and bit 30 of `0x40B4` on the RK3576. The TRM gives the two settings as "If the integer
is odd, carry 1" and "Carry 1 no matter what the integer is" [TRM].

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

### fp16 narrowing

`FP32TOFP16_EN`, bit 16 of `OUT_CVT_SCALE`, narrows the converter's fp32 result to fp16 [TRM].
The fp16 matmul writes its fp16 output through it.

**Conditions:** RK3588 (Turing RK1), `rocket` 1.3.0, NPU clock 928 MHz, 2026-11-02. One fp16
matmul task with fp16 output, integer operands, results from ±1000 to ±1000000.

The narrowing rounds to nearest with ties to even, and a result past the fp16 range is written
as infinity [HW sweep 2026-11-02]:

| fp32 result | fp16 written |
|---|---|
| 65472 | 65472 (`0x7BFE`) |
| 65488, a tie | 65472 (`0x7BFE`), the even neighbor |
| 65500 to 65519 | 65504 (`0x7BFF`) |
| 65520 and above | +inf (`0x7C00`) |

A negative result is written with the same magnitude and the sign bit set (`0xFBFE`, `0xFBFF`,
`0xFC00`).

> [!CAUTION]
> An fp16 output of magnitude 65520 or more is written as ±inf, and one from 65504 to 65519 as
> ±65504. Software that can produce a result past 65504 must bound its operands, or treat both
> ±65504 and ±inf as an overflow.

The EW stage's fp16 output narrows by the same rule. An EW add or multiply on a convolution
feed writes ±inf from 65520 and 65472 at the 65488 tie [HW sweep 2026-11-07].

Nothing narrows before the output. A single fp16 product past 65504 enters the sum at full
width, as the arithmetic model in [chips/rk3588.md](../chips/rk3588.md) §"fp16 matmul
arithmetic" has it. A product past the range that later products bring back writes the exact
in-range sum [HW sweep 2026-11-07]. A lone product of 65520 or more writes ±inf, as a sum does.
The batched matmul entry, which the flash attention's scores come from, writes the
same bits.

**Not characterized:** the fp16 output of the LUT path. Also the softmax path of one
flash-attention run whose output stayed finite at raw scores past the fp16 range (Evidence).

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

Mesa masks `(bits >> 9) & 0x7fff`, which takes the exponent's low bit as the multiplier's
bit 14. It adds the `+1`, and sets bit 14 only where the sum is below `2^14`. The two
derivations agree except where mantissa bits [22:9] are all set:

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

**Conditions:** RK3588 (Turing RK1), NPU clock ~816 MHz, 2026-09-27. The int8-out depthwise
program, at two shapes, one a stride-2 plane. The direct int8-out program at two 3×3 shapes, 64
to 64 and 32 to 160 channels, for the multiplier, the split shift and `B`.

The output requant has one scale and one shift per task. A per-output-channel scale therefore
goes on the BS stage's multiplier, which reads a per-channel operand [HW sweep]:

| Register | Value | Function |
|---|---|---|
| `DPU_RDMA_BRDMA_CFG` `0x501C` | `BRDMA_DATA_USE` 7 (`0xE`) | BRDMA reads a 64-byte group per 8 output channels: int32 `A[8]` at 0, int16 `B[8]` at 32, int16 `C[8]` at 48. Data use 1 reads a dense int32 bias alone |
| `DPU_BS_CFG` `0x4040` | `0x20140` | the ALU adds `A` from DMA, the multiplier is live (`BS_MUL_BYPASS` clear) |
| `DPU_BS_MUL_CFG` `0x4048` | `(s << 8) \| 1` | `BS_MUL_SRC` reads `C` per channel. `[13:8]` shifts a **non-negative** product |
| `DPU_DATA_FORMAT` `0x4010` | `[9:4]` = s | `BS_MUL_SHIFT_VALUE_NEG` shifts a **negative** product. The TRM names it the BS shift value for negative data [TRM] |

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

### TFLite's two rounding points

TFLite's integer requant rounds twice. Its fixed-point multiply rounds `acc · M0 / 2^31` to an
integer, where `M0` is a 31-bit multiplier. Its right shift by `e` bits then rounds again
([../depthwise-conv.md](../depthwise-conv.md) §"The int8 domain"). The OUT_CVT rounds once. A
requant agrees with TFLite on more accumulators only when it reproduces both rounding points
[host-computed 2026-09-27]:

| Requant | Accumulators that differ from TFLite's | Largest share in one layer |
|---|---:|---:|
| OUT_CVT alone, `CVT_ROUND` 0 | 1.00% | 24.5% |
| OUT_CVT alone, `CVT_ROUND` 1 | 1.00% | 24.5% |
| BS `C` and OUT_CVT `MUL` as one ~30-bit multiplier, rounded once | 1.16% | 24.7% |
| BS `C = round(M0 / 2^16)` at BS shift 15, then OUT_CVT `MUL` 2^14 at `SHIFT` 14 + `e`, `CVT_ROUND` 1 | 0.13% | 0.44% |

**Conditions:** a host model of each requant over the 255 per-tensor scales of five ImageNet
classifier `.tflite` models. Each scale takes 400000 accumulators, uniform over the range that
does not saturate the output.

The last row rounds at the BS shift where TFLite's multiply rounds, and at the OUT_CVT where its
shift rounds. It still differs on 0.13%, because `C` keeps 15 of `M0`'s 31 bits. The one-rounding
forms differ most where TFLite's right shift is small. At a right shift of 1 they differ on
about a quarter of a layer's accumulators, and at 2 on an eighth.

**Not characterized:** the two-point requant on a device.

### Consequence for int8 activations resident between conv ops

An on-chip requant, per-tensor or per-channel, differs from TFLite's by at most one per op.
Chained, it drifts the way TFLite's own optimized kernels drift from its reference ones. In
a detector, 12-14% of the elements end two or more counts off either way
([depthwise-conv.md](../depthwise-conv.md) §"The int8 domain"). Keeping int8 activations
resident is therefore judged per op by that bound, and along a chain by an accuracy measure
such as COCO mAP.

## Software support

### Requant entries

`gen_conv2d_int8` and `gen_conv2d_dw_int8` with `int8_out=1` use the integer convert to emit
requantized int8, within
one of TFLite at a rounding boundary ([depthwise-conv.md](../depthwise-conv.md)).

Every entry in rocket-userspace writes `CVT_ROUND` 0, so its rule is half to even. Its CPU
requant model has the half-to-even rule. `ROCKET_OUT_CVT_ROUND=1` sets the bit on every
integer `OUT_CVT_SHIFT` write, to test the tie rule.

rocket-userspace derives every pair through one function, `npu_out_cvt_pair()`, which keeps the
carry.

The fp16 output of the int8 datapath is not available in rocket-userspace.

### Direct int8-output program

**Conditions:** RK3588 (Turing RK1), NPU clock ~816 MHz (the 600 MHz request), 2026-09-27.
`tflite-rocket` with `native_int8`, warm.

`rocket_conv2d_int8_q` in rocket-userspace is the direct int8-out entry for a per-tensor weight
scale. It programs whole groups, with zero kernels in the padding. On MobileDet's direct
convolutions, 0.037% of its outputs differ from TFLite's requant by one count, and none by more
[HW sweep].

Through `tflite-rocket`, the entry runs two detectors faster than the int32-raw path with its
host requant [HW sweep]:

| CPU configuration | MobileDet | SSD MobileNet v2 |
|---|---:|---:|
| Governor `performance`, process on the A76 cores | 1.22-1.27x | 1.19-1.20x |
| Governor `ondemand`, `scaling_min_freq` 1.2 GHz, no CPU affinity | 1.205-1.221x | 1.165-1.169x |

On `ondemand`, MobileDet takes 148 ms against the int32-raw path's 180 ms.

For a 1×1 convolution, the other path is the resident int8 matmul with a host requant. The
entry is faster than it at every output plane of 400 pixels or more. At 100 pixels the matmul
is faster on 8 of 12 convolutions [HW sweep]. `tflite-rocket` therefore sends a 1×1 to the
entry from 256 pixels (`pw_i8out_min_m`). On those 1×1s, 0.050% of MobileDet's outputs and
0.074% of SSD MobileNet v2's differ from TFLite's by one count, and none by more. Splitting a
layer's output channels across a pool's workers measures 0.99-1.03x the unsplit speed [HW
sweep].

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
[HW sweep, RK1, ~816 MHz, 2026-09-27]. The model runs 1.095-1.104x warm, 133 -> 121 ms, against
those layers on the fp16 depthwise path. COCO
mAP over 100 images is 0.3019 against the CPU's 0.2996. The depthwise term falls from 43.2 to
34.7 ms of the op sum, and most of that is the delegate's host stages around the call. The
library call itself moves 28.6 -> 26.8 ms. Most of these layers are small enough that the fixed
per-call overhead sets their cost.

### Per-axis direct entry

`rocket_conv2d_int8_q_perc` in rocket-userspace runs a per-output-channel weight scale on the
direct int8-out program. Its planner is the depthwise entry's, which sorts the output channels
into scale classes. Each class runs as tasks of whole 32-kernel groups, with its own BS shift
and OUT_CVT gain. At eight shapes the device matches an independent host model of that
arithmetic on every element [HW sweep, RK1, ~816 MHz, 2026-09-27].

Against TFLite's per-channel arithmetic, 0.156% of the outputs of EfficientDet-Lite0's 102
convolutions differ by one count, and none by more [HW sweep, RK1, ~816 MHz, 2026-09-27]. Over
those convolutions and the 1×1s of §"Direct int8-output program", the largest share in one
layer is 0.41%, at a class head.

## Evidence

### Integer convert

A ratio classifier confirms the `SCALE` multiplier on hardware. The sign bit is measured on
the RK3588 and the RK3576, 64 of 64 elements at two scales each, with
`tests/requant_edge_probe.c` (2026-09-27).

The offset placement ran through the int8 matmul's integer convert (`ROCKET_INT8_DEQ=1`, fp32
output) at M 16, K 64 and N 32. Operands were in [−8, 7] and accumulators from −587 to 571. It used
17 settings of (`SCALE`, `SHIFT`, `OFFSET`), from (1, 0, 0) to (9, 5, 40000), among them odd
offsets at shifts 1 to 6, at each `cvt_type`. Every element was scored against five offset
placements (none, after the rounding, before the shift unscaled, before the multiply, and at
the output scale before the rounding), each under three rounding rules (half to even, rounding
down, half away from zero).

At `cvt_type` 0 only the output-scale placement with ties to even is exact at all 17
settings. At `cvt_type` 1 only the offset before the multiply, with ties to even, is.
The odd offsets at a tie separate the two placements at `cvt_type` 0.

The tie order on the int8 writer was measured with `tests/requant_zp_order_probe.c` through
`rocket_conv2d_int8_q`. Every input sits at its zero point, so each output channel's
accumulator is its bias. The scale's top 14 mantissa bits are all ones, so the derivation gives
`MUL` exactly 2^14. The requant is then `acc >> e`, for e 1 and 3. Of 64 channels, 32 sit on
ties, odd multiples of 2^(e-1), and 32 off them.

Each call wrote 1024 outputs at one zero point. The zero points were -5, -1, 0, 1, 2, 3, 7 and
100, one call each. At `CVT_ROUND` 0 the offset-first model explains all 1024 outputs of every
call. At each odd zero point, all 512 ties disagree with the round-first model. At `CVT_ROUND` 1
the round-first model explains every output. No job timed out.

The offset-first model with ties away from zero misses 16-112 ties at the odd zero points. It
also misses 32 ties at zero point 2 and 256 at zero point 100. Those are the ties whose
accumulator is negative while the offset-included sum is not.

### The `CVT_ROUND` tie rule

The test is `tests/requant_round_probe.c` (2026-09-23). Each run leaves exactly one of five
candidate rules (half up, half away, half even, truncation, floor) consistent with every
element. The runs per SoC:

- **RK3576**: 160 tie elements per setting at two shifts, both signs, through
  `rocket_matmul_int8_rk3576` [H96 MAX M9].
- **RK3588**: 128 tie elements per setting at shifts 1 and 3, both signs [Turing RK1 at
  ~816 MHz]. The test runs through the int8 matmul's integer convert
  (`ROCKET_INT8_DEQ=1`, `CVTTYPE=0`, `SCALE=1`), whose fp32 output holds the rounded value
  exactly.

No ordinary test reaches a tie. `requant_round_probe` picks a scale whose top 14 mantissa bits
are all ones, and asserts the rule on both SoCs.

### fp16 narrowing

The table comes from one `gen_gemm_fp16` task with `fp16_out` = 1, M 4, K 512, N 16. Each
weight column holds integers of at most 2048, which are exact in fp16. Its sum is one
target: 1000, 2048, 65472, 65488, 65500, 65504, 65505, 65519, 65520, 65535, 65536, 70000,
100000, 135000, 262144 or 1000000. The activation rows are +1, −1, +1 and −1. Every sum is an
integer below 2^24, so the fp32 accumulation is exact, and the raw fp16 bits were read from an
output stamped `0xAA`. All 64 elements equal the IEEE round-to-nearest-even conversion of
their sum.

An earlier measurement, ggml-rocket's `test-rocket-fa` with `ROCKET_FA_QSCALE=0` (RK1,
~816 MHz, 2026-10-09), leaves a raw `q·k` of 60000 to 135000 on the flash attention's fp16
scores. The softmax that read the scores returned no non-finite output, where an inf score
gives NaN in the host softmax. It differed from a double reference by up to 1.0. That run recorded the
softmax output, not the value of an overflowed score, and not which of the flash attention's
softmax paths computed it. It was read as a converter that saturates at ±65504. Neither the
one-task measurement nor the product measurement below shows that: both score entries write ±inf
from 65520.

The EW measurement is `tests/fp16_ew_overflow_probe.c` through `rocket_ew_add_fp16` and
`rocket_ew_mul_fp16`, the convolution-fed EW program, on 256 elements each. The add walked
65472 + 8k for k 0-127 and its negation. The multiply used operand pairs whose products are
65472, 65504, 65520 (45 × 1456 and 91 × 720) and 65536, with their negations. No product of two
fp16 values equals the 65488 tie. Every output matches IEEE round-to-nearest-even narrowing, and
saturation at 65504 explains none of the results from 65520 up.

The product measurement is `tests/fp16_mm_product_overflow_probe.c` through
`rocket_matmul_fp16` at M 4, K 128 and N 64, one K pass. It ran on the RK1 at 928 MHz under
`rocket` 1.3.0, on 2026-11-07. The activation rows hold ones and a few large values, with the
signs + − + −. The large values are 256, 45, 625, 375, 1000, 100, 200, 2047 and 999. Some weight
columns select one product of 65472 to 1000000, some two or three in-range products summing past
65504, and some many small ones. The cancellations start with a product past 65504 and end in
range, such as 70000 − 10000 and 1000000 − 999000.

Every element is scored against five models. The first holds the products and their sum exact
and narrows the sum once at the output. The second clamps that output to ±65504. The other three
clamp each product to ±65504, make each product of 65520 or more ±inf, or round each product to
fp16. The first explains 256 of 256 elements and the others 208 to 236. The cancellation
1000000 − 999000 reads 1000, where a clamped product gives 0 and an infinite one NaN.

`rocket_matmul_fp16_batch` on the same operands as two items writes the same bits.

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

The direct-program test is `tests/conv_i8out_probe` in its per-channel mode, at 64 to 64 and 32
to 160 channels. Every element is scored against the split model, the CPEND model and the
multiplier left out. Its cases are these:

- An identity multiplier
- `C` from 8192 to 32767 at shift 14
- The shift split 14 and 6
- `B` with `OW_SRC` clear and set

### BN stage per-channel multiplier

All 8 tasks over two shapes ran to the 500 ms watchdog and wrote nothing. No reference program
found here enables NRDMA.

### TFLite's two rounding points

The model reproduces TFLite's reference integer arithmetic, both roundings half away from zero,
then the clamp. The OUT_CVT rows derive `MUL` and `SHIFT` as the library does. The ~30-bit row
searches `C` from 16384 to 32767 for the pair closest to the scale. The scales are those of
Inception v1 and v3, MobileNet v1 and v2, and ResNet-18. A real activation concentrates nearer
the zero point than a uniform accumulator, so the uniform range weights each layer's extremes
more than inference does. The per-layer medians are 0.27% for both OUT_CVT rows, 0.39% for the
~30-bit row and 0.11% for the two-point row.

### Software support

The CPU model is `tests/requant_model.h`. The per-axis test is
`tests/conv_dw_int8_perc_runtime.c`. The direct per-axis test is
`tests/conv2d_int8_q_perc_rocket.c`, through the fd, context and pool entries and the host
model. It also scores TFLite's per-channel arithmetic, failing on any difference past one.

The `performance` speedups for MobileDet and SSD MobileNet v2 are from four passes with the
configuration order rotated, and the `ondemand` ones from two. The 1×1 comparison ran 39
convolutions on both programs, 27 at output planes of 400 pixels or more and 12 at 100 pixels.
Each of EfficientDet-Lite0's 102 convolutions ran on TFLite's own input. The per-axis depthwise
speedup on EfficientDet-Lite0 has the governor pinned, on the A76 cores, over four rotated
passes.
