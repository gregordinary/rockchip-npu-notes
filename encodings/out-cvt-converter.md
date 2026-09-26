# DPU OUT_CVT, the output converter (int32 accumulator -> output)

The last stage of the DPU before write-back is the **output converter**, from the NVDLA SDP
lineage. NVDLA documents it as `y = sat((x − offset) * scale >> shift)`. On the RK3588's
integer path the offset is added after the shift instead (below). It is driven by three
registers:

| reg | addr | fields |
|---|---|---|
| `DPU_OUT_CVT_OFFSET` | `0x4080` | signed 32-bit offset, added after the shift on the integer path: the output zero point |
| `DPU_OUT_CVT_SCALE`  | `0x4084` | `[15:0]` uint16 scale (multiplier); `[16]` `FP32TOFP16_EN` |
| `DPU_OUT_CVT_SHIFT`  | `0x4088` | `[5:0]` integer shift; `[19:12]` `minus_exp`; `[30]` `CVT_ROUND`, the tie rule; `[31]` `cvt_type` |

## Two operating modes (cvt_type)

**Integer convert (the matmul/conv accumulator path).** Acting on the raw int32 CACC
accumulator, the converter is purely integer:

```
out = (float_or_int)( round_half_to_even( (acc_i32 * SCALE) >> SHIFT ) )
```

- `SCALE` is a **uint16 integer multiplier**, *not* fp16 and *not* fixed-point. HW-confirmed by
  the ratio classifier: `SCALE=2 -> ×2`, `SCALE=256 -> ×256`, exactly.
- `SHIFT` is an **integer right-shift in the integer domain, applied before any float cast**.
  It rounds to nearest; see the tie rule below.
- `minus_exp` (bits[19:12]) and `cvt_type` are **no-ops on the integer accumulator path**;
  they only matter on the LUT/EW float datapath (below).
- The **BN-MUL operand** (`DPU_BN_MUL_CFG[31:16]`) is likewise an integer multiply with the
  operand read as **uint16** (`0x3800 -> ×14336`), redundant with `SCALE` here.

This is exactly the QNNPACK **requantization** form (15-bit multiplier + shift + zero-point
offset -> int8/int16). `gen_conv2d_int8_fill(int8_out=1)` uses it to emit requantized int8,
within one of TFLite at a rounding boundary ([depthwise-conv.md](../depthwise-conv.md)).

With an int8 output the order is `sat8(round_half_to_even(acc * SCALE >> SHIFT) + OFFSET)`: the
offset lands after the rounding shift, and saturation comes after the offset. The int8
depthwise path is bit-exact against that model at output zero points from -128 to 127, at two
scales [HW sweep, RK1, measured 2026-09-23]. Under the NVDLA order, subtracting before the
multiply, the zero point scales down to a fraction of one.

### The tie rule is `CVT_ROUND`

`acc*SCALE >> SHIFT` rounds to nearest, and **bit 30 of `OUT_CVT_SHIFT` decides where an exact
half goes**:

| `CVT_ROUND` | Tie rule | 0.5, 1.5, -0.5, -1.5 go to |
|---|---|---|
| 0 | half to even, QNNPACK's *precise* requantization | 0, 2, 0, -2 |
| 1 | half away from zero, the NVDLA documentation's rule and TFLite's | 1, 2, -1, -2 |

Measured on both parts, each run leaving exactly one of five candidate rules (half up, half
away, half even, truncation, floor) consistent with every element [HW sweep,
`tests/requant_round_probe.c`, 2026-09-23]:

- **RK3576**, bit 30 of `0x40B4`: 160 tie elements per setting at two shifts, both signs,
  through `rocket_matmul_int8_rk3576` [H96 MAX M9].
- **RK3588**, bit 30 of `0x4088`: 128 tie elements per setting at shifts 1 and 3, both signs,
  through the int8 matmul's integer convert (`ROCKET_INT8_DEQ=1`, `CVTTYPE=0`, `SCALE=1`),
  whose fp32 output carries the rounded value exactly [Turing RK1 at 600 MHz].

**Every entry in this tree writes 0, and so do the vendor's programs**: none of the 206
`OUT_CVT_SHIFT` writes in the RK3576 vendor-capture corpus sets bit 30. So the shipped rule is
half to even, and neither setting is the round-half-**up** that `(x + half) >> shift` gives
and that every CPU model in this tree used to spell. `tests/requant_model.h` is the one model
and carries the half-to-even rule. `ROCKET_OUT_CVT_ROUND=1` sets the bit on every integer
`OUT_CVT_SHIFT` write, for the probe.

**Reaching a tie needs a deliberately chosen scale.** The derivation ends in `+1`
(`MUL = ((bits>>9) & 0x7fff) + 1`, bit 14 forced), so `MUL` is **odd** for every round scale
including every power of two, and an odd multiplier moves an exact half off the tie in the
outward direction, so the rounder never sees one. So no ordinary gate reaches the case.
`requant_round_probe` does: it picks a scale whose top 14 mantissa bits are all ones under an
even exponent field, which makes `MUL` exactly `2^14`, and it asserts the rule on both parts.

**How often it matters in practice.** With `MUL` odd, ties are one accumulator residue in
`2^SHIFT`, and the two rules differ on half of those: about `2^-(SHIFT+1)` of a surface.
At a typical `SHIFT` of 15-20 that is nothing in a small gate case and tens of elements in a
large prefill: one count each, sparse, present in every configuration. Exactly the standing
noise that has made single-element int8 disagreements unreadable.

**Float-affine convert (the LUT-activation path).** When the converter's *input* is already a
Q-format value in the float/EW datapath (e.g. a LUT output `q`), `cvt_type=1` selects
`out = (q + offset) * 2^-minus_exp`, narrowed to fp16 by `FP32TOFP16_EN`. See
[dpu-lut-activation.md](dpu-lut-activation.md). The keystone is that `q` is *not* the raw
integer accumulator, which is why `minus_exp` is inert on the plain matmul path.

## Output dtype / cube on the int8 (int32-acc) datapath

| out_precision | writer geom (size_e / surf_add) | cube C2 | notes |
|---|---|---|---|
| int32 | `7` / `stride*8` | 4 | the default raw-accumulator readback |
| fp32  | `7` / `stride*8` | 4 | **bit-exact cast** of the int32 acc (`ROCKET_INT8_FP32_OUT`) |
| fp16  | `3` / `stride*2` | 8 | small single-tile shapes only; **range-limited** `\|acc\|<=2048` |

- **fp32 cast + integer scale** fold cleanly and generally (any shape, bit-exact).
- **fp16** halves the output readback but its writer geometry is only correct for small
  single-tile shapes (wrong/strided values at e.g. 64×128×64) **and** the int32 accumulator
  must fit fp16, so it does **not** help the large-K LLM readback. Not shipped.

## Consequence for int8 dequant (the negative result)

A **fractional** W8A8 dequant scale (`acc * a_scale[m] * b_scale[n]`, both < 1) **cannot fold
into OUT_CVT to produce a fractional float**: `(acc*scale)>>shift` always yields an
integer-valued float (the fraction is truncated). So:
- the host per-row × per-channel dequant **stays**;
- the int8 output-readback lever is **bigger-Kt** (fewer K-partials to read), not OUT_CVT;
- OUT_CVT *is* the right tool for int8->int8/int16 **requant** (integer output) and for the
  fp32 cast / per-tensor integer gain.

Gate: `tests/matmul_int8_dequant_rocket.c`. Related: [precision-field.md](precision-field.md),
[size-e-quirk.md](size-e-quirk.md), [k-accumulation.md](k-accumulation.md) (int8 EW K-accum dead).

## Per-channel (per-axis) requant: multiplier yes, shift no

The DPU output requant stage carries a **per-channel multiplier but only a single per-stage
shift**, so it cannot reproduce TFLite's per-axis int8 requant bit-exactly. [source-confirmed]
(Mesa `registers.xml`, the `0x40xx` DPU domain):

| reg | addr | per-channel? | role |
|---|---|---|---|
| `DPU_BS_MUL_CFG` | `0x4048` | **mul: yes** (`BS_MUL_SRC=1` reads the operand from a `[C]` cube) | per-channel multiply |
| `DPU_BN_MUL_CFG` | `0x4068` | **mul: yes** (`BN_MUL_SRC=1` per-channel `[C]` operand) | per-channel multiply |
| `BS/BN_MUL_SHIFT_VALUE` (+`_NEG`) | in-reg `[13:8]` / `[5:0]` | **shift: no**, one register value per stage | per-stage right-shift |
| `DPU_OUT_CVT_SHIFT` | `0x4088` | **shift: no**, one global value | final requant shift |

So a per-output-channel **scale** is expressible (the BS/BN MUL operand cube, the `[C]` broadcast
that [sdp-stage-precision.md](sdp-stage-precision.md) also notes), but a per-output-channel
**shift** is not: every channel truncates by the same `SHIFT`. TFLite per-axis requant is
`out[oc] = MultiplyByQuantizedMultiplier(acc[oc], mult_q31[oc], shift[oc]) + zp` with a per-OC
multiplier **and** per-OC shift in gemmlowp's Q31 doubling-high-mul form, which the NVDLA
uint16-mul + single-shift integer requant above cannot match channel-for-channel.

The per-channel converter the chip *does* have is the **CNA input** path (`CVT_CON0` `0x104C`
`CVT_TRUNCATE_0..3`, `CVT_CON5` `0x1180` `PER_CHANNEL_CVT_EN`) in the `0x1xxx` domain. It
normalizes input features/weights as they stream into CBUF, **not** the output requant. Don't
mistake it for a per-OC output requant.

**Consequence.** A per-tensor on-chip requant matches CPU TFLite's accumulator exactly and
its output within one at a rounding boundary: 11 of 4096 outputs of a real int8 depthwise layer
differ by one, none by more [HW sweep, RK1], [depthwise-conv.md](../depthwise-conv.md). A
per-axis requant also carries the single-shift limit above, so it cannot match TFLite
channel-for-channel. Both bound **keeping int8 activations resident between conv ops** (the
delegate's NCHW-resident int8 inter-op lever). An on-chip inter-op requant differs from TFLite's
by at most one per op per-tensor, and how that compounds across a chain of ops is unmeasured.
Per-axis, the lever is gated by an accuracy measure such as COCO mAP rather than bit-exactly.
A native per-channel int8 depthwise is not built: COCO mAP parity showed the fp16 depthwise
costs ~0 accuracy on the detectors measured.
