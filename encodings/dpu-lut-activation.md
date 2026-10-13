# DPU LUT: on-NPU elementwise activation (NVDLA SDP)

The RK3588 DPU contains an NVDLA-style SDP [LUT](../GLOSSARY.md) block (registers
`0x4100..0x412C`), which the matmul and conv paths leave fully bypassed. The LUT computes an
arbitrary single-variable function `f(x)` on a feature cube entirely on the NPU. It gives the
`rocket` stack a nonlinear activation on the chip. Clearing `EW_LUT_BYPASS` in `EW_CFG`
runs it, `LUT_CFG` selects its table mux, and `LUT_INFO` sets its table step.

Run standalone, the op is a DPU pass, with no CNA or CORE and no conv. Only the DPU and
DPU_RDMA blocks run, in flying mode. MRDMA reads the input cube straight from DRAM, the
pipeline applies `BN-mul -> LUT -> OUT_CVT`, and the DPU writes the output cube.

## Support status

| Function | Method | Status |
|---|---|---|
| Sigmoid, HardSigmoid | Unit `[0,1]` table, standalone fp16 | Supported. max_abs 0.00146 and 0.00049 against the fp16 CPU reference [HW sweep 2026-06-22] |
| tanh, SiLU, HardSwish over the knee `[-3,3]` | Single-pass table with the affine OUT_CVT | Supported [HW sweep]. The signed-output kinds spike at x~0 (§"LE/LO boundary spike at x~0") |
| sqrt, rsqrt, reciprocal, log, exp | Shifted single table over a positive domain | Supported [HW sweep] |
| HardSwish, SiLU | LUT gate, multiply on the host | Supported (default) |
| HardSwish, SiLU | LUT gate, multiply on the NPU | Supported (opt-in) |
| GELU | Two-pass `x·Φ(x)` | Supported (default) |
| GELU | One table over the whole curve | Not supported. It mis-decodes (§"Two-pass GELU, `x·Φ(x)`") |
| EW ADD, SUB, MUL, MAX, MIN | Identity-conv main feed | Supported, bit-exact |
| EW MUL | DPU-only program, flying MRDMA main | Not supported. Bit-exact on hardware, and the library's entry programs two fields wrong (§"Software support") |
| Clip, Div | EW MAX and MIN with a constant operand, and `a·reciprocal(b)` | Supported |
| LeakyReLU | Two-table LE/LO with a host repair of the x~0 band | Supported. alpha=0 is not supported: use the native DPU ReLU |
| Softplus, Mish, Abs, ELU, SELU, PReLU | §"More activation kinds" | Supported |
| conv -> activation | Single-pass epilogue in the conv task | Supported, for inputs that stay in the curved region |
| matmul -> GELU | Single-pass epilogue | Not supported for wide inputs |
| Functions over a wide dynamic range | Log-domain LUT | Not supported |

## Pipeline

```
in (fp16, DRAM)
  → MRDMA (flying, MRDMA_FP16TOFP32_EN=1)         x as fp32
  → BN stage: multiply by BN_MUL_OPERAND          index = x * index_scale
  → EW stage LUT (EW_LUT_BYPASS=0):               g = LUT(index)   (Q0.15)
  → OUT_CVT: g * 2^-MINUS_EXP → fp16              f(x)
  → WDMA → out (fp16, DRAM)
```

Because the op is elementwise, the host feeds a flat fp16 vector and reads a flat fp16
vector back. The cube dims (C2=8 channels, width = n/8, height 1) only partition the data,
and the read and write strides are identical, so `out[i]=f(in[i])` holds however the data is
tiled. `n` must be a multiple of 8 (the C2 atom).

## The two tables (NVDLA LE/LO hybrid)

There are two 513-entry tables, uploaded through the LUT access port. `DPU_LUT_ACCESS_CFG`
selects `ACCESS_TYPE=1`=write + `TABLE_ID`. `DPU_LUT_ACCESS_DATA` writes one entry,
auto-incrementing the address. The tables are:

- **LE** (`TABLE_ID 0`): the negative branch, covering input index
  `[LE_START, 0]`. `LE_START = 0xffffc000` = −16384.
- **LO** (`TABLE_ID 1`): the positive branch, covering `[0, LO_END]`.
  `LO_END = 0x00004000` = 16384.

`LUT_INFO.{LE,LO}_INDEX_SELECT = 5` sets the table step to `2^5 = 32` index units, so each
table spans `16384/32 = 512` segments (513 endpoints). The hardware linearly interpolates
between adjacent entries. Outside the table range it extrapolates with a slope
(`LUT_LE_SLOPE_SCALE/SHIFT`), so a saturating function like sigmoid is correct well past
the table edge.

`LUT_CFG = HYBRID_PRIORITY(1)|OFLOW_PRIORITY(1)|LO_LE_MUX(2)` selects the LE+LO hybrid mux.
The two tables hold samples on a uniform grid in the input domain with
`step = 32/index_scale`:

```
LE[i] = quant( f( -((512-i)*step) ) )   inputs −range .. 0     (i = 0..512)
LO[i] = quant( f(    i*step       ) )   inputs 0 .. +range     (i = 0..512)
quant(y) = clamp(round(y * 32768), 0, 32767)        # unsigned Q0.15, [0,1] output
```

> [!CAUTION]
> The tables do not persist from one submit to the next. Under `rocket` on an RK3588, a submit
> that wrote no table entries produced unrelated values [external: allbilly/rk3588 at
> `e13d99f`, not reproduced here]. It did so with the table pointer cleared, re-armed or left
> untouched.
> Software must write all 1,026 entries in every submit that reads them. Reuse between the
> tasks of one submit is not characterized.

## The geometry constants (sigmoid)

These constants place the LUT over the input range and map its Q0.15 output back to fp16
[HW sweep]. Software computes the table contents from the activation itself (`quant(f(x))`).

| field | value | meaning |
|---|---|---|
| `BN_MUL_OPERAND` | `0x6912` | fp16(2596), the index_scale. It maps `x` -> index, so the table covers `x ∈ ±16384/2596 ~ ±6.31` |
| `BN_ALU_CFG` | `0x80000000` | BN ALU bias word |
| `BN_CFG` | `BN_ALU_ALGO(2)\|BN_RELU_BYPASS(1)` | BN multiply active, no relu |
| `OUT_CVT_SCALE` | `FP32TOFP16_EN(1)\|1` | |
| `OUT_CVT_SHIFT` | `CVT_TYPE(1)\|MINUS_EXP(15)` | Q0.15 -> fp16 = `g · 2^-15` |
| `OUT_CVT_OFFSET` | `1` | rounding bias (~1 LSB, negligible) |
| `LE_SLOPE_SCALE / SHIFT` | `23107 / 22` | underflow/overflow extrapolation slope |
| `EW_CFG` | `EW_RELU_BYPASS(1)\|EW_OP_CVT_BYPASS(1)` | LUT runs (`EW_LUT_BYPASS` left clear, bit7=0) |

HardSigmoid reuses every one of these constants. Only the table content changes
(`clip(x/6+0.5,0,1)`), because its output is also in `[0,1]` on the same grid. Any
`[0,1]`-codomain `f` drops in the same way.

The register fields follow Mesa `registers.xml` and `librocketnpu`'s `include/npu_hw.h`. See
also [../README.md](../README.md), [precision-field.md](precision-field.md), and
[SOURCES.md](../SOURCES.md) for the allbilly encoding reference.

## Task framing

The op uses the same task framing as `gen_matmul_task`, rather than the rknpu-style framing.
The program sets `DPU_S_POINTER`/`DPU_RDMA_S_POINTER` to `0xE`, then writes the register
content, then the trailer `OP_NONE · OP_REG_PC(PC_REGISTER_AMOUNTS) · OP_40 · OP_ENABLE`.
**The enable word is `0x18`**, bits 3 and 4 (DPU and DPU_RDMA), not the matmul's `0x1D`.
No CNA or CORE participates, so bit 0 (CNA) and bit 2 (CORE) are clear
([regcmd-task-model.md](regcmd-task-model.md) §"`PC_OPERATION_ENABLE` as a block bitmap").
The trailer target encoding is `librocketnpu`'s `OP_REG_DPU = BLOCK_DPU|PC_OP_01`
(`DPU 0x1000 -> 0x1001`).

## The NVDLA hybrid LUT (LE/LO)

The DPU LUT is the NVDLA SDP hybrid pair. NVDLA calls the LE table X (linear or exponential
index) and the LO table Y (linear only). In its typical configuration, a raw table covers the
whole input range and a density table covers a small range at a higher sample rate. Either
table can take either role. NVDLA recommends X as the raw table for LRN, and as the density
table for sigmoid and tanh [source-confirmed: NVDLA IAS, lut-programming].

The programming in §"The two tables (NVDLA LE/LO hybrid)" uses neither role. LE and LO cover
the negative and the positive half of the index range at the same step. `LUT_LO_LE_MUX` and the
`Priority / OverflowPriority / UnderflowPriority` registers select between them.
Out-of-table inputs extrapolate linearly: `LUT[0]+(X-START)·UFLOW_SCALE/SHIFT` (underflow,
`DPU_LUT_LE_SLOPE_*`) and `LUT[N]+(X-END)·OFLOW_SCALE/SHIFT` (overflow,
`DPU_LUT_LO_SLOPE_*`, emitted by `gen_lut_activation_fp16`). NVDLA v1 sizes LE=65 / LO=257
entries, and the RK3588 uses 513 each (`lut[1026]`).

Five stats counters (`XHitNum/YHitNum/UnderflowNum/OverflowNum/PriorityNum`) report
per-layer selection, and are useful diagnostics.

> [!CAUTION]
> Reading some NPU counter pages hard-locks the SoC. Read these counters only with the safe
> pattern in [../perf/hw-byte-counters.md](../perf/hw-byte-counters.md).

Refs: NVDLA IAS [lut-programming](https://nvdla.org/hw/v1/ias/lut-programming.html) /
[unit_description](https://nvdla.org/hw/v1/ias/unit_description.html) /
[programming_guide](https://nvdla.org/hw/v1/ias/programming_guide.html).

## The BN-ALU bias

The BN-ALU bias works in the index domain. The index follows `index = x·BN_MUL + BN_ALU` (MUL
then ALU), and `BN_ALU` is an fp32 bias in the index (post-scale) domain [HW sweep]. Mapping
the whole domain `[x_lo,x_hi]` onto the positive index half needs this bias. Software must
program `BN_ALU = fp32(-x_lo·scale)`. A pre-scale `(x+B)` add, `fp32(-x_lo)`, is
systematically wrong.

The `0x80000000`=-0.0 that the standalone op uses is the no-op case. The bias relocates the
table but does not cure the x~0 spike, because the mux selects on `sign(x)`
(§"LE/LO boundary spike at x~0").

## Signed / wide output: the affine OUT_CVT

The `[0,1]` path above is the special case `OUT_CVT_OFFSET=1, MINUS_EXP=15, SCALE=1`. The
output converter is a general affine, signed, pre-shift map [HW sweep]:

```
out_fp16 = (q_interp + OUT_CVT_OFFSET) * 2^-MINUS_EXP * OUT_CVT_SCALE     (FP32TOFP16_EN)
```

- `OUT_CVT_OFFSET` is signed and added before the shift. The `+1` of the [0,1] path is the
  same field as a Q-domain rounding bias, and Mesa's int8 requant uses it as a signed
  `ozp-0x80`. A tanh run on hardware confirms it: store `(tanh(x)+1)/2` in Q0.15 and decode
  `tanh = (q - 16384) * 2^-14`, i.e. `OFFSET = -16384 (0xFFFFC000)`, `MINUS_EXP = 14`. The
  result is max_abs 0.0034.
- **Bias trick for any bounded range `[lo, hi]`:** store `g = (f(x)-lo)/S`
  (`S=2^Sexp >= hi-lo`), then `MINUS_EXP = 15-Sexp`, `offset = round(lo·32768/S)`,
  `scale=1`. It drives single-pass tanh (S=2), SiLU (S=32, X=±16), and HardSwish over the
  knee [-3,3] (S=4), all validated on hardware [HW sweep] (`build_lut_affine` in
  `rocket_activation.c`). The bias trick is the single-pass activation, with no gate and no
  EW multiply.

## Positive-domain kinds: sqrt / rsqrt / reciprocal

The DPU LUT computes the reciprocal family (`x>0`: `√x`, `1/√x`, `1/x`) [HW sweep]. These
kinds use the shifted single-table mode. It maps the whole positive domain `[x_lo,x_hi]` onto
the LO (positive index) half via `index = (x − x_lo)·scale`, `scale = 16384/(x_hi−x_lo)`
(BN-MUL), `BN-ALU = −x_lo·scale` (fp32, post-scale). Because `x` never reaches 0, the LE/LO
sign mux never fires, so these kinds have none of the x~0 glitch of the symmetric kinds. The
OUT_CVT is the same affine bias-trick (`out_lo=0`, `S=2^Sexp >= max f`).

Uniform-grid accuracy is domain-bounded. The 513-entry LO table samples `x` uniformly, so
for these steep functions the worst error is at the low end. The low-end relative
interpolation error scales as `~ (Δ/x_lo)²·(c/8)` with `Δ=(x_hi−x_lo)/512`. A domain ratio
`x_hi/x_lo ≲ 128` placed away from 0 therefore keeps it near 1% [expected]. Inputs outside `[x_lo,x_hi]`
clamp to the edge value.

The inputs measured span 100-200x, each inside its kind's default table domain
(§"Activations"). Over them the max-relative error is sqrt 0.85%, rsqrt 0.44%, reciprocal 1.0%
(means 0.05-0.12%) [HW sweep]. A wide dynamic range needs a log-domain LUT
(`1/x = exp(−ln x)` is straight in log x), which is not built.

These kinds give on-NPU `Div` (`rocket_ew_div_fp16 = a·reciprocal(b)`, HW 0.35%). They are
the math core of RMSNorm/LayerNorm (`rsqrt(mean(x²)+ε)`) and of the softmax denominator, the
normalization primitives of the LLM and Whisper blocks.

## LOG, a signed-output positive-domain kind

`ln(x)` (`ROCKET_ACTIVATION_LOG`, `x>0`) joins the positive-domain shifted-table family. It is
the natural inverse of EXP, and the per-element log for log-probabilities, NLL and
cross-entropy. It is the first `act_shifted_domain` kind with a signed output: `log(x)<0` for
`x<1`. Unlike sqrt, rsqrt, reciprocal and exp (all `out_lo=0`), it sets
`out_lo=log(x_lo)` < 0. The OUT_CVT offset `lround(out_lo·32768/S)` is then negative and
decodes the signed range.

The generic positive-domain path contains the signed-output machinery. It is the same
negative-`OUT_CVT_OFFSET` decode that tanh and ELU use (§"Signed / wide output: the affine
OUT_CVT"), and LOG exercises it with `out_lo≠0`. LOG adds no hardware path. The default
domain `[0.25,32]` with `S=8` covers `[log .25, log 32]~[-1.39,3.47]`.

For log the right error metric is absolute, not relative. `log` crosses zero at `x=1`,
where relative error is ill-defined (`÷0`). Log is also consumed additively
(`log(ab)=log a+log b`), so absolute error is what propagates. On a uniform grid the worst
error is at the steep small-x end (`d²/dx² log = −1/x²`). Measured on hardware over
`[0.3,30]`: max_abs 0.0066 (@x~0.33), mean 0.0007, on par with the reciprocal family
[HW sweep]. As for the other kinds, a wide dynamic range needs a companded or log-domain grid,
because the uniform grid is domain-bounded.

The signed output decodes via the negative offset. A standalone log that included `x<=0`
would spike at x~0 (§"LE/LO boundary spike at x~0"). Its domain is `x>0` (interior to the LO
half), so it is glitch-free like the rest of the positive-domain family. The per-row `log`
inside LogSoftmax runs on the host (M scalars, exact). The LOG LUT is for a large tensor of
logs, not the softmax denominator (see [whisper-encoder.md](whisper-encoder.md)).

## EXP, the softmax numerator

`exp(x)` joins the shifted-single-table family (`ROCKET_ACTIVATION_EXP`,
`act_shifted_domain`). Its default domain is `[-16,0]`, the softmax case: after the
mandatory row-max subtraction the input is in `(-∞,0]` and the output in `(0,1]`
(`out_lo=0`, `S=1`). Unlike the symmetric kinds, the domain can include `x<=0` with no x~0
spike, because the output decode is unsigned (§"Affected kinds"). The BN-ALU bias maps the
whole domain onto the positive index half, so EXP works on the standalone flying path
(unlike `build_lut_affine` GELU).

The relative interpolation error of exp on a uniform grid is ~constant `Δ²/8` (~1e-4 over
512 intervals) because `f''/f = 1`, a good fit for a uniform LUT. Over `[-16,0]` it measures
max_abs 5.5e-4 [HW sweep]. The softmax-sum end-to-end check (row-max subtracted, T up to 512,
score spread to 20) measures sum_rel <=0.04%, max|Δp| <=6e-5.

EXP's deep tail, `x<=-11`, quantizes its table entries to `q=0`, which decodes to ~4.0.
Software must clamp every shifted-table entry to `q>=1` (§"q=0 table-entry decode fault").

## conv -> activation fusion

The single-pass LUT epilogue also runs fused into a conv. A direct fp16 conv post-processes
its own result with `f(x)` in the same NPU job (`out = f(conv(x))`, no second round-trip).
The fusion is the validated fp16-out conv plus the BN-mul -> EW-LUT -> affine-OUT_CVT
epilogue. The conv's output geometry, `size_e=1` and NC1HWC2 readback are all unchanged.
Because the SDP stages sit downstream of the DPU main feed, the epilogue's register values
are byte-identical to the standalone op's. Only the input source differs: the conv CACC
accumulator instead of a flying MRDMA stream.

The fused output is not byte-identical to the standalone op's. It matches the standalone
`gen_lut_activation_fp16` applied to the same conv output to <= 0.0039 (the LUT quant step),
across 1×1/3×3/stride-2/tiled shapes [HW sweep].

## Large `n` and tiling

The LUT op lays the flat vector out as a cube of `cols = n/8` width positions (C2=8 fp16
each). `DPU_DATA_CUBE_WIDTH` is 13 bits, so the hard maximum for one task is `n <= 65528`. A
transformer's `[M,I]` activation cube is millions of elements, so software must tile it. Each
chunk must stay below the 13-bit maximum (§"13-bit cube-dimension maximum").

## Two-operand EW (HardSwish/SiLU multiply)

`x · gate(x)` (HardSwish, SiLU) needs an elementwise multiply of two buffers. Two programs
compute it bit-exactly. The one `librocketnpu` uses feeds the EW main through an identity
conv, described below.

A DPU-only program computes it too, with no CNA or CORE [HW sweep, RK3588, both drivers]. It
takes a flying MRDMA main and the operand on the ERDMA. That program requires three fields:
`ORIG_CHANNEL`, `COMB_USE` bit 0 and the EW converter bypass. They are in
[sdp-stage-precision.md](sdp-stage-precision.md) §"A two-buffer EW op needs no conv main
feed".

Mesa drives a second EW operand only as the residual of a conv (`add_tensor`), with the conv as
the main feed [source-confirmed, [../teflon-add-capture/](../teflon-add-capture/)]. That
program is Mesa's choice, and the datapath does not require it.

### The identity-conv main feed

An identity conv supplies the main feed. The program reuses the exact fp16 K-accumulation
eltwise path (`gen_matmul_fp16` `accumulate=1`) with an identity weight matrix. The conv
reproduces operand `A` into CACC as the EW main, and the EW unit multiplies it by the ERDMA
operand `B`:

| field | eltwise-ADD (K-accumulation) | eltwise-MUL |
|---|---|---|
| `DPU_EW_CFG` (`0x4070`) | `0x108202C0` `EW_ALU_ALGO(2)`,`EW_OP_TYPE(0)` | **`0x108003C4`** `EW_ALU_ALGO(0)`,**`EW_OP_TYPE(1)`**,`EW_OP_CVT_BYPASS(1)` |
| `DPU_RDMA_ERDMA_CFG` (`0x5034`) | `0x40000008` | `0x40000008` (same) |
| `FEATURE_MODE_CFG` (`0x5044`) | `…\|COMB_USE(5)` | same (op-independent) |
| `SRC_BASE`/`EW_BASE` | operand / operand+1surf | same |

Only `DPU_EW_CFG` changes: clear the ALU algo and set `EW_OP_TYPE(1)`, the fp16
eltwise-multiply word. The operand DMA transport (ERDMA + `COMB_USE(5)` + the `MAX(M,12)`
surface-stride minimum) is identical to K-accumulation. Software must therefore keep
**M >= 12**. Below that minimum, the upper channel surfaces mis-stride. ADD reproduces `A+B`
and MUL gives `A*B` bit-exactly, both max_abs 0 [HW sweep].

### Subtract

Subtract (`rocket_ew_sub_fp16`) needs no new regcmd: `a-b == a+(-b)`, and fp16 negation is
an exact sign-bit flip. SUB packs the operand negated and runs the ADD datapath unchanged,
bit-for-bit the add result. Add, sub and mul measure max_abs 0 for n up to 40000, across an
M-tile crossing [HW sweep]. Subtract covers the ONNX/TFLite `SUB` op on the flat-vector EW
path.

### Max and Min

The EW ALU's `EW_ALU_ALGO` field reaches MAX and MIN, not just sum (add). The field is
`DPU_EW_CFG` bits [17:16], so `2<<16 = 0x20000` is the `EW_ALU_ALGO(2)=SUM` bit in the
`0x108202C0` add word. The NVDLA SDP X/Y ALU algo encoding holds: 0 = MAX, 1 = MIN, 2 = sum.
On the same conv-main EW datapath as add (identity-conv main + ERDMA operand + `COMB_USE(5)`),
only the `DPU_EW_CFG` word changes:

| op | `DPU_EW_CFG` | `EW_ALU_ALGO` |
|---|---|---|
| ADD | `0x108202C0` | 2 (sum) |
| MAX | `0x108002C0` | 0 |
| MIN | `0x108102C0` | 1 |

MAX and MIN only select one of the two fp16 operands, so they are bit-exact (n to 40000:
max_abs 0) [HW sweep]. They cover TFLite/ONNX `Maximum`/`Minimum` and ReLU = `max(x,0)`. With
a constant operand they cover Clip = `min(max(x,lo),hi)` (`rocket_clip_fp16`, two passes) and
the bounded-ReLU family. They also build PReLU with a per-channel slope (§"More activation
kinds").

## Two-pass GELU, `x·Φ(x)`

The accurate on-NPU GELU is the 2-pass form, exactly like SiLU: `GELU(x) = x·Φ(x)`, where
`Φ(x) = 0.5(1+erf(x/√2))` is the Gaussian CDF. The CDF is a monotone `[0,1]` function, so
it uses the clean unit-LUT geometry (`build_lut_unit`, the same one sigmoid uses,
index_scale 2596). It is free of the flat-region spike: Φ has no exactly-flat run, and its
saturating tails use the unit-LUT `le_slope` extrapolation (like sigmoid).

`rocket_activation_fp16(GELU)` takes this path: the gate `ROCKET_ACTIVATION_GELU_GATE` = Φ,
then the EW-mul by x. Validated on hardware: cos=1.000000, max_abs 0.0016 vs true erf-GELU
over `[-12,12]`, including the flat tails [HW sweep]. The two-pass GELU puts the Whisper
encoder block's GELU on the NPU (block cos=1.000000). SiLU is likewise 2-pass and clean
standalone.

> [!CAUTION]
> A single LUT covering the whole GELU curve mis-decodes on the standalone flying path (cos
> ~0.05). It also fails at FFN scale when fused into a conv, through the flat-tail mux spike
> (§"LUT flat-region mux spike"). Use the 2-pass `x·Φ(x)` (cos = 1.000000).

### Fused single-pass matmul->GELU

A fused single-pass matmul->GELU does not work for wide inputs. Lowering `C = GELU(A·Bᵀ)`
onto a 1×1 conv with the single-pass GELU epilogue fails at FFN scale: cos~0.04,
`max_abs=128`. The epilogue is `build_lut_affine`, the same table the conv->act fusion
uses. The cause is the flat-tail mux spike (§"LUT flat-region mux spike"), hit en masse
because a real fc1 output spans the flat negative region (`GELU(x)~0` for `x≲-3`). The tanh
kind fares better (cos 0.955) only because its curved range is narrower.

> [!CAUTION]
> Single-pass LUT fusion (conv->act or a hypothetical matmul->act) is only safe for inputs
> that stay in the curved region. The reliable on-NPU GELU/SiLU is the 2-pass `x·gate(x)`.

A fused matmul->act (`rocket_matmul_fp16_act`) is therefore not viable for wide inputs.

## More activation kinds

All reuse the mechanisms above, with no new regcmd primitive. Each is tested on hardware
against double-precision math.

| kind | mechanism | x~0 | accuracy (HW) |
|---|---|---|---|
| **Softplus** `log(1+e^x)` | shifted single-table (EXP path), `out_lo=0` | clean | max_rel 0.14% |
| **Mish** `x·tanh(softplus(x))` (YOLOv4/v7) | 2-pass: `[0,1]` gate (build_lut_unit) + EW-mul | clean | max_rel 0.06% |
| **Abs** `\|x\|` | symmetric shifted single-table (kink on the middle sample j=256) | clean | max_abs 1e-3 (the q>=1 clamp at x=0) |
| **ELU** `x>=0?x:α(e^x−1)` | symmetric shifted single-table + host x~0 repair (signed) | repaired | max_rel 0.37% |
| **SELU** `λ·ELU_α` (fixed α,λ) | as ELU | repaired | max_rel 0.6% |
| **PReLU** per-channel `α_c` | no LUT: `max(x,α_c·x)` (α∈[0,1]) or `relu+α·min` (general), via EW max/scale | n/a | bit-exact |

PReLU's per-channel slope uses the EW path, not the LUT, because one LUT table cannot hold
a per-channel parameter. The per-channel scale is a row-broadcast `ew_mul` (channel = row
in a `[C,S]` layout), followed by `ew_max`. PReLU therefore inherits the EW bit-exactness
and has no x~0 glitch. Both the α∈[0,1] max path and the general path for α outside it are
bit-exact [HW sweep].

## Software support

`librocketnpu` runs the standalone op through the runtime `rocket_activation_fp16` and the
generator `gen_lut_activation_fp16`. Its `build_lut_unit` builds the `[0,1]`-codomain tables,
`build_lut_affine` the single-pass affine tables, and `build_lut_shifted` the shifted
single-table kinds.

| Option | Effect |
|---|---|
| `ROCKET_ACT_NPU_MUL=1` | Runs HardSwish and SiLU fully on the NPU through `rocket_ew_mul_fp16`, the identity-conv multiply. Off by default |
| `ROCKET_ACT_EXPERIMENTAL=1` | Enables the flying `gen_ew_mul_fp16` |
| `ROCKET_ACT_WIDE_LUT` | Forces the single-pass GELU, for reverse engineering |
| `ROCKET_LUT_ENABLE`, `ROCKET_LUT_SPTR` | Override the enable word and the `S_POINTER` value from the environment, for bring-up |
| `ROCKET_LUT_SHIFT` | Selects the shifted single-table path. Unconditional for the positive-domain kinds |
| `ROCKET_LUT_QFLOOR` | The minimum table entry, default 1 (Erratum E5) |
| `ROCKET_LUT_XLO`, `ROCKET_LUT_XHI` | Tune the positive domain to the data |
| `ROCKET_LEAKY_NOREPAIR`, `ROCKET_ELU_NOREPAIR` | Disable the host repair of the x~0 band |

### Activations

HardSwish and SiLU default to gate-on-NPU-LUT + multiply-on-host, which takes the
transcendental off the CPU (HardSwish max_abs 0.001, SiLU 0.012). Host-mul stays the default
because a standalone EW-mul is a second NPU round-trip. A single pass that fuses the multiply
into the producing conv (conv->`LUT(y)`->mul-by-main) avoids that round-trip, and the
identity-conv multiply establishes the EW mechanism it needs.

The positive-domain default table domains (`act_shifted_domain`) are sqrt `[0.25,64]`, rsqrt
`[0.5,64]` and reciprocal `[0.25,32]`, ratios of 256x, 128x and 128x.

`rocket_leaky_relu_fp16` (LE = `alpha*x`, LO = `x`) and `rocket_elu_fp16` / `rocket_selu_fp16`
repair the x~0 band on the host. The leaky path does not support alpha=0 (plain ReLU). Its
all-zero negative branch is a flat run that trips the flat-region spike across the whole LE
table. Use the native DPU ReLU.

### Large `n`

`run_dpu_lut` tiles over a per-op cap, `DPU_LUT_MAXN = 32768` (`cols = 4096`). Every caller
(SiLU, sigmoid, tanh, GELU, leaky, sqrt, rsqrt, reciprocal) therefore works at any `n`.
Without tiling the op `gen`-fails (`-2`) and returns garbage for `n > 65528`. The LUT, pool
and reduce entries refuse past 13 bits, and the regcmd emitters refuse any CNA or DPU extent
past its field.

### Elementwise ops

The EW selector is `matmul_params_t.ew_op` (2=MAX, 3=MIN, 0 = the legacy add/`ew_mul`
path). The entries are `rocket_ew_sub_fp16`, `rocket_ew_max_fp16`, `rocket_ew_min_fp16`,
`rocket_clip_fp16` and `rocket_ew_div_fp16`.

`gen_ew_mul_fp16`, the retained flying multiply, programs two of the DPU-only program's
fields wrong. It writes `COMB_USE(5)`, whose bit 0 stops the program completing. It also
writes `DATA_CUBE_CHANNEL` with `ORIG_CHANNEL` 0, which makes every lane but lane 0 of each
atom wrong (zero, on this program). It stays behind `ROCKET_ACT_EXPERIMENTAL=1`, with the two
fields still wrong.

### conv -> activation fusion

`npu_dpu_desc.lut_en` + `lut_ep` (a `lut_epilogue_t`) drive the fusion inside
`gen_conv2d_task`. Default-off is byte-identical: the regcmd adds exactly +1026 ops, the LE/LO
upload, and nothing else. `rocket_conv2d_act_fp16` (rocket_conv.h) wires it for
SiLU/tanh/GELU. Away from the x~0 band, the fused conv->tanh output is within 0.0039 of the
true function [HW sweep]. Inside the band it spikes, as the standalone op does (§"LE/LO
boundary spike at x~0").

### Frontends

- **Detection:** HardSigmoid + HardSwish (with the EW-mul) move the MobileNetV3/MobileDet
  activation blocks off the CPU. `tflite-rocket` claims `HARD_SWISH` and `LOGISTIC` nodes and
  runs them on the LUT under its opt-in `act_npu` option. Its default is the exact host
  kernel. Video rate needs the single-pass conv->hardswish fusion.
- **LLM:** the same LUT computes the GELU/SiLU gates for fused FFN / Whisper blocks. SiLU/GELU
  have the on-NPU EW-mul. The affine OUT_CVT also gives a single-pass LUT, which the
  flat-region and x~0 spikes limit.
- **DPU post-processing:** `librocketnpu` drives the LUT, a DPU post-processing block, beyond
  the matmul/conv requant.

## Known issues

The four issues are listed in pipeline order. The input cube's geometry comes first, then the
LE/LO table mux, then the output converter's decode of a table entry.

### 13-bit cube-dimension maximum

> [!CAUTION]
> Erratum E9. A LUT chunk at the exact 13-bit maximum width (`cols = 8191`, `n = 65528`)
> corrupts a thin tail of cube positions [HW sweep]. Software must keep every 13-bit
> cube-dimension field below 8191.

A 65528-element SiLU chunk mis-computed ~54 elements (constant count, data-independent,
absent at small `n`), so cosine dipped to 0.9990. Tiling well under the maximum at
`DPU_LUT_MAXN = 32768` (`cols = 4096`) is bit-clean (geglu/FFN cos = 1.000000, 0 misses at
every size). The hazard is in the same family as the CBUF-bank-slack over-read: positions at
the edge of a register's range are unsafe.

The 13-bit maximum applies to every cube dimension, not to the LUT only [source-confirmed].
SHARD (EuroMLSys '26) reports the same register limit as an `0xe010 "REGTASK Overflow"` that
fires for any operand index > 8191. Large transposes are among the cases, independent of the
LUT path. Treat 8191 as a hard silicon limit on every 13-bit cube-dimension field
(`DPU_DATA_CUBE_WIDTH` / `HEIGHT` / `CHANNEL` and the PPU/RDMA mirrors), and tile below it.
The CNA's input width and height are narrower still, 11 bits, which is the binding limit for
a convolution ([../matmul-as-conv.md](../matmul-as-conv.md) §"Per-tile geometry limits").

### LUT flat-region mux spike

> [!CAUTION]
> In flat or asymptotic regions (zero derivative), the LE/LO overflow mux mis-toggles at
> register-saturation boundaries [HW sweep]. It produces a config-independent garbage spike,
> a discrete glitch rather than a slope error. Software must bypass the LUT for constant
> regions: use the 2-pass gate and EW multiply, or the host.

HardSwish (exactly 0 for x<=-3) trips it whether the flat run is in-table (wide table ->
+128) or pushed into underflow extrapolation (knee table -> +16). Only the curved knee
`[-3,3]` is clean. Smooth activations (sigmoid/tanh/SiLU) never have an exactly-flat run, so
they are correct away from the table boundary. The saturating tail of tanh uses the tuned
underflow slope `23107>>22`, not slope 0. NVDLA's IAS gives no flat-region recipe.

### LE/LO boundary spike at x~0

> [!CAUTION]
> A second, distinct mux glitch sits at the LE/LO table boundary (`index = x·index_scale = 0`,
> that is `x~0`) [HW sweep]. When an input falls within ~0.0015 of exactly 0, the hybrid mux
> mis-toggles and emits a discrete garbage spike (`+128`).

The glitch is a property of the LUT itself, not of the conv->activation fusion. The
standalone flying op and the fused epilogue spike at the identical elements.

> [!NOTE]
> The band is razor-thin (~±R/8192), so a sparse-linspace test steps over it. Dense random
> conv outputs (N=2.6e4-5e5) hit it a handful of times. When validating a single-pass kind,
> sample densely.

#### Mechanism

The mux selects on `sign(x)`, not on the table index. The hybrid mux picks LE-vs-LO on
`sign(x·BN_MUL) = sign(x)`, the pre-ALU value. The BN-ALU bias therefore relocates the
table address but not the mux decision. No index trick (shift, lower-quarter, asymmetric, a
different scale) avoids it: the spike is always at the input value 0, never at a fixed index
[HW sweep]. A true single-table mode (mux disabled) is expected to avoid it [expected], but
re-exposes the flat-region spike for functions with a flat run.

#### Affected kinds

Only signed-output kinds spike. The spike tracks the signed output decode (a negative
`OUT_CVT_OFFSET`), not merely a domain straddling 0. Under a dense `[-0.02,0.02]` step-1e-5
input sweep, the shifted-single-table kinds split cleanly [HW sweep]:

- **Unsigned `[0,hi]` output (`OUT_CVT_OFFSET >= 0`): x~0-clean.** Softplus (`out_lo=0`),
  Abs (symmetric `[-R,R]`, `out_lo=0`), and the 2-pass Mish gate (`[0,1]`, offset `+1`)
  show no spike (worst |Δ| 7e-4 / 1e-3 / 8e-6, interpolation and quantization only). They
  are in the same camp as sigmoid/exp/sqrt/rsqrt/recip.
- **Signed output (`OUT_CVT_OFFSET < 0`): x~0 spike.** The kinds are tanh/SiLU/GELU and
  ELU/SELU (`out_lo=-λα<0`). They spike at x~0 with a discrete ~64-128 over a ~±5e-4 band
  (ELU/SELU 97/4001 in the sweep). Of these, tanh is otherwise clean (no flat run), so x~0
  is its only glitch.

If you can frame the output decode with `out_lo >= 0` (a non-negative offset), the kind is
x~0-clean. A genuinely signed output keeps the spike.

#### Mitigation: host band-repair

The spike is a razor-thin band at x~0, and the readback already streams every output element.
Software can therefore patch that band on the host with the exact value, which is the
cheapest robust repair (§"Software support").

The repair only works where the producer can patch the band. The standalone activation
can. The fused-in-conv epilogue cannot, so a robust FFN SiLU/GELU needs the 2-pass path or a
reverse-engineered single-table mode.

### q=0 table-entry decode fault

> [!CAUTION]
> Erratum E5. A zero-valued LUT table entry trips a decode fault in the output converter: it
> emits a constant ~4.0, not 0 [HW sweep]. Software must clamp every shifted-table entry to
> `q>=1`.

EXP shows it. The output is correct down to `x~-10` (table entry `q=lround(f·32768)>=1`) and
jumps to ~4.0 for `x<=-11`, exactly where `exp(x)<1.5e-5` quantizes the entry to `q=0`. For
EXP the fault inflates the softmax sum: the whole deep tail decodes to ~4 each.

The clamped value decodes to ~3e-5 (rounds to ~0 on readback), which is correct for the
tail. The sqrt, rsqrt and reciprocal kinds never produce a q=0 entry over their domains, so
the clamp is a no-op for them. The same fault likely also sits in the sigmoid/tanh LE deep
tail (`sigmoid(-16)~0 -> q=0`) [expected]. Those paths do not run that deep in a model, and
the case is not measured.

## Evidence

### Support status

The hardware validation of 2026-06-22 covers fp16 Sigmoid and HardSigmoid. Both generalize
from the 128-element reference cube to 1024+ (`tests/activation_lut_rocket.c`).

### The geometry constants (sigmoid)

A BNALU sweep pinned the `BN`/OUT_CVT operand format.

### The BN-ALU bias

`BN_ALU=0x46000000=fp32(8192)` gives tanh ACC 0.0007, and the pre-scale guess `fp32(-x_lo)`
gives ACC~2.

### Signed / wide output: the affine OUT_CVT

The single-pass tanh, SiLU and HardSwish tables are validated by `tests/lut_tanh_rocket.c`.

### Positive-domain kinds: sqrt / rsqrt / reciprocal

The reciprocal family and LOG are validated by `tests/recip_rsqrt_rocket.c`. Its sweeps are
1024 log-spaced inputs over sqrt `[0.30,60]` (200x), rsqrt `[0.60,60]` (100x) and reciprocal
`[0.30,30]` (100x), each inside the kind's default table domain. Errors are relative to the
double-precision function.

### EXP, the softmax numerator

`tests/exp_lut_rocket.c` validates EXP on hardware, including the softmax-sum check.

### conv -> activation fusion

The measurement is `tests/conv_act_rocket.c` on four convolutions, as input channels to output
channels at an input size:

- 1×1, 64 to 64, 8×8
- 3×3, 32 to 16, 8×8
- 3×3 stride 2, 64 to 32, 10×12
- 3×3 tiled, 64 to 16, 40×40

Against the standalone op on the same conv output, the fused output measures 0.0005 for tanh,
0.0039 for SiLU and 0.0020-0.0039 for GELU. Against the true function, the fused tanh
measures 0.0039 on the three shapes where it has no x~0 spike. The tiled shape has three, and
measures 8.0001 [HW sweep, RK3588 (Turing RK1), 2026-09-23].

### Two-operand EW (HardSwish/SiLU multiply)

The DPU-only program's result is from `tests/ew_int_probe.c`. The identity-conv ADD and MUL are
from `tests/ew_mul_rocket.c`. The `ew_mul_rocket` runtime check sweeps add, sub and mul. MAX and
MIN are from `tests/ew_minmax_rocket.c`.

A `COMB_USE` sweep with `ORIG_CHANNEL` held at 0 finds no working value, because
`ORIG_CHANNEL` 0 leaves every lane but lane 0 of each atom wrong.

### Two-pass GELU, `x·Φ(x)`

The GELU measurement is `tests/gelu_rocket.c`.

### More activation kinds

Each kind has a CTest test against double-precision math. PReLU is `tests/prelu_rocket.c`.

### Software support

The HardSwish and SiLU default accuracy is from `tests/activation_lut_rocket.c`.

### Known issues

The cube-dimension test is `tests/ffn_rocket.c` at `n = 65536` / `100000`.

The x~0 mechanism comes from LeakyReLU, which has a sharp kink exactly at 0. Re-mapping x=0 to
a different index moves the spike to follow x=0. A repair-off scan of 16385 inputs uniformly
across [-16,16] finds exactly one spike, at `x=0.000000` (`+128`), for every scale. The dense
sweep of the affected kinds is `tests/x0_glitch_probe.c`.

The host band-repair is validated by `tests/leaky_relu_rocket.c` (alpha ∈
{0.01,0.1,0.125,0.2, 0.25,0.5} all `bad=0`, sweep `max_abs <= 0.002`, x~0 band exact),
`tests/elu_rocket.c` and `tests/softplus_mish_rocket.c`.

The sqrt, rsqrt and reciprocal tests pass with the `q>=1` clamp on.
