# DPU OUT_CVT — the output converter (int32 accumulator → output)

The last stage of the DPU before write-back is the **output converter** (NVDLA SDP lineage:
`y = sat((x − offset) * scale >> shift)`). It is driven by three registers:

| reg | addr | fields |
|---|---|---|
| `DPU_OUT_CVT_OFFSET` | `0x4080` | signed 32-bit pre-subtract offset (int8-out: `out_zp − 0x80`) |
| `DPU_OUT_CVT_SCALE`  | `0x4084` | `[15:0]` uint16 scale (multiplier); `[16]` `FP32TOFP16_EN` |
| `DPU_OUT_CVT_SHIFT`  | `0x4088` | `[5:0]` integer shift; `[19:12]` `minus_exp`; `[31]` `cvt_type` |

## Two operating modes (cvt_type)

**Integer convert (the matmul/conv accumulator path).** Acting on the raw int32 CACC
accumulator, the converter is purely integer:

```
out = (float_or_int)( (acc_i32 * SCALE) >> SHIFT )       (arithmetic >>, truncates toward −∞)
```

- `SCALE` is a **uint16 integer multiplier** — *not* fp16, *not* fixed-point. HW-confirmed by
  the ratio classifier: `SCALE=2 → ×2`, `SCALE=256 → ×256`, exactly.
- `SHIFT` is an **integer right-shift in the integer domain, applied before any float cast**,
  so it **truncates** (the `cv≠0` floor-rounding signature: `acc>>1` of an odd `acc`).
- `minus_exp` (bits[19:12]) and `cvt_type` are **no-ops on the integer accumulator path** —
  they only matter on the LUT/EW float datapath (below).
- The **BN-MUL operand** (`DPU_BN_MUL_CFG[31:16]`) is likewise an integer multiply with the
  operand read as **uint16** (`0x3800 → ×14336`), redundant with `SCALE` here.

This is exactly the QNNPACK **requantization** form (15-bit multiplier + shift + zero-point
offset → int8/int16). `gen_conv2d_int8_fill(int8_out=1)` uses it to emit requantized int8
bit-exact vs Teflon.

**Float-affine convert (the LUT-activation path).** When the converter's *input* is already a
Q-format value in the float/EW datapath (e.g. a LUT output `q`), `cvt_type=1` selects
`out = (q + offset) * 2^-minus_exp`, narrowed to fp16 by `FP32TOFP16_EN`. See
[dpu-lut-activation.md](dpu-lut-activation.md). The keystone is that `q` is *not* the raw
integer accumulator — which is why `minus_exp` is inert on the plain matmul path.

## Output dtype / cube on the int8 (int32-acc) datapath

| out_precision | writer geom (size_e / surf_add) | cube C2 | notes |
|---|---|---|---|
| int32 | `7` / `stride*8` | 4 | the default raw-accumulator readback |
| fp32  | `7` / `stride*8` | 4 | **bit-exact cast** of the int32 acc (`ROCKET_INT8_FP32_OUT`) |
| fp16  | `3` / `stride*2` | 8 | small single-tile shapes only; **range-limited** `|acc|≤2048` |

- **fp32 cast + integer SCALE** fold cleanly and generally (any shape, bit-exact).
- **fp16** halves the output readback but its writer geometry is only correct for small
  single-tile shapes (wrong/strided values at e.g. 64×128×64) **and** the int32 accumulator
  must fit fp16 — so it does **not** help the large-K LLM readback. Not shipped.

## Consequence for int8 dequant (the negative result)

A **fractional** W8A8 dequant scale (`acc * a_scale[m] * b_scale[n]`, both < 1) **cannot fold
into OUT_CVT to produce a fractional float** — `(acc*scale)>>shift` always yields an
integer-valued float (the fraction is truncated). So:
- the host per-row × per-channel dequant **stays**;
- the int8 output-readback lever is **bigger-Kt** (fewer K-partials to read), not OUT_CVT;
- OUT_CVT *is* the right tool for int8→int8/int16 **requant** (integer output) and for the
  fp32 cast / per-tensor integer gain.

Gate: `tests/matmul_int8_dequant_rocket.c`. Related: [precision-field.md](precision-field.md),
[size-e-quirk.md](size-e-quirk.md), [k-accumulation.md](k-accumulation.md) (int8 EW K-accum dead).

## Per-channel (per-axis) requant — multiplier yes, shift no

The DPU output requant stage carries a **per-channel multiplier but only a single per-stage
shift**, so it cannot reproduce TFLite's per-axis int8 requant bit-exactly. [source-confirmed]
(Mesa `registers.xml`, the `0x40xx` DPU domain):

| reg | addr | per-channel? | role |
|---|---|---|---|
| `DPU_BS_MUL_CFG` | `0x4048` | **mul: yes** (`BS_MUL_SRC=1` reads the operand from a `[C]` cube) | per-channel multiply |
| `DPU_BN_MUL_CFG` | `0x4068` | **mul: yes** (`BN_MUL_SRC=1` per-channel `[C]` operand) | per-channel multiply |
| `BS/BN_MUL_SHIFT_VALUE` (+`_NEG`) | in-reg `[13:8]` / `[5:0]` | **shift: no** — one register value per stage | per-stage right-shift |
| `DPU_OUT_CVT_SHIFT` | `0x4088` | **shift: no** — one global value | final requant shift |

So a per-output-channel **scale** is expressible (the BS/BN MUL operand cube, the `[C]` broadcast
that [sdp-stage-precision.md](sdp-stage-precision.md) also notes), but a per-output-channel
**shift** is not — every channel truncates by the same `SHIFT`. TFLite per-axis requant is
`out[oc] = MultiplyByQuantizedMultiplier(acc[oc], mult_q31[oc], shift[oc]) + zp` with a per-OC
multiplier **and** per-OC shift in gemmlowp's Q31 doubling-high-mul form — which the NVDLA
uint16-mul + single-shift integer requant above cannot match channel-for-channel.

The per-channel converter the chip *does* have is the **CNA input** path (`CVT_CON0` `0x104C`
`CVT_TRUNCATE_0..3`, `CVT_CON5` `0x1180` `PER_CHANNEL_CVT_EN`) in the `0x1xxx` domain — it
normalizes input features/weights as they stream into CBUF, **not** the output requant. Don't
mistake it for a per-OC output requant.

**Consequence.** On-chip int8 requant matches **Teflon** (which uses this same NVDLA
single-shift form, and is itself **per-tensor** only) but diverges from CPU TFLite by up to
**~143** on full-range int8 — the measured Teflon-vs-CPU gap [HW sweep],
[depthwise-conv.md](../depthwise-conv.md). A native per-channel int8 depthwise was declined on
that basis (COCO mAP parity showed the fp16-approx depthwise costs ~0 accuracy). The same
ceiling governs **keeping int8 activations resident between conv ops** (the delegate's
NCHW-resident int8 inter-op lever): doing the inter-op requant on-chip would drift from the
int8 reference unboundedly across ops, so that lever is **mAP-gated (an accuracy decision),
not bit-exact-gateable** — which is why it sits with the calibration-accuracy cluster, not the
on-device bit-exact gates.
