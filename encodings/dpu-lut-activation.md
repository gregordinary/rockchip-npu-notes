# DPU LUT — on-NPU elementwise activation (NVDLA SDP)

The RK3588 DPU carries an NVDLA-style SDP **LUT** block (registers `0x4100..0x412C`)
that the matmul/conv paths leave fully bypassed. It computes an arbitrary
single-variable function `f(x)` on a feature cube entirely on the NPU — the first
**nonlinear activation** we run on the FOSS rocket stack. HW-validated 2026-06-22:
fp16 **Sigmoid** (max_abs 0.00146 vs fp16 CPU ref) and **HardSigmoid** (0.00049),
generalizing from the 128-element reference cube to 1024+ (`tests/activation_lut_rocket.c`,
runtime `rocket_activation_fp16`, generator `gen_lut_activation_fp16`).

This is a *standalone DPU pass* — no CNA/CORE (no conv). Only the DPU + DPU_RDMA
blocks run, in **flying mode**: MRDMA reads the input cube straight from DRAM,
the pipeline applies `BN-mul -> LUT -> OUT_CVT`, the DPU writes the output cube.

## Pipeline

```
in (fp16, DRAM)
  → MRDMA (flying, MRDMA_FP16TOFP32_EN=1)         x as fp32
  → BN stage: multiply by BN_MUL_OPERAND          index = x * index_scale
  → EW stage LUT (EW_LUT_BYPASS=0):               g = LUT(index)   (Q0.15)
  → OUT_CVT: g * 2^-MINUS_EXP → fp16              f(x)
  → WDMA → out (fp16, DRAM)
```

Because the op is elementwise, the host feeds a **flat** fp16 vector and reads a
flat fp16 vector back: the cube dims (C2=8 channels, width = n/8, height 1) only
partition the data, and the read/write strides are identical, so `out[i]=f(in[i])`
holds regardless of how the data is tiled. `n` must be a multiple of 8 (the C2 atom).

## The two tables (NVDLA LE/LO hybrid)

There are **two 513-entry tables**, uploaded through the LUT access port
(`DPU_LUT_ACCESS_CFG` selects `ACCESS_TYPE=1`=write + `TABLE_ID`; `DPU_LUT_ACCESS_DATA`
writes one entry, auto-incrementing the address):

- **LE** (`TABLE_ID 0`) — the "linear"/negative branch, covering input index
  `[LE_START, 0]`. `LE_START = 0xffffc000` = −16384.
- **LO** (`TABLE_ID 1`) — the positive branch, covering `[0, LO_END]`.
  `LO_END = 0x00004000` = 16384.

`LUT_INFO.{LE,LO}_INDEX_SELECT = 5` ⇒ table step = `2^5 = 32` index units → each
table spans `16384/32 = 512` segments (513 endpoints). The hardware **linearly
interpolates** between adjacent entries, and **extrapolates with a slope**
(`LUT_LE_SLOPE_SCALE/SHIFT`) outside the table range — so a saturating function
like sigmoid is correct well past the table edge.

`LUT_CFG = HYBRID_PRIORITY(1)|OFLOW_PRIORITY(1)|LO_LE_MUX(2)` selects the LE+LO
hybrid mux. The two tables are sampled on a uniform grid in the **input** domain
with `step = 32/index_scale`:

```
LE[i] = quant( f( -((512-i)*step) ) )   inputs −range .. 0     (i = 0..512)
LO[i] = quant( f(    i*step       ) )   inputs 0 .. +range     (i = 0..512)
quant(y) = clamp(round(y * 32768), 0, 32767)        # unsigned Q0.15, [0,1] output
```

## The geometry constants (sigmoid) [HW-verified]

These place the LUT over the input range and map its Q0.15 output back to fp16. The table
*contents* are computed from the activation itself (`quant(f(x))`, the runtime's `build_lut_unit`);
the geometry below is HW-verified — the activation is bit-exact on device, and the `BN`/OUT_CVT
operand format was pinned by a BNALU sweep.

| field | value | meaning |
|---|---|---|
| `BN_MUL_OPERAND` | `0x6912` | fp16(2596) — the **index_scale**; maps `x` → index, so the table covers `x ∈ ±16384/2596 ≈ ±6.31` |
| `BN_ALU_CFG` | `0x80000000` | BN ALU bias word |
| `BN_CFG` | `BN_ALU_ALGO(2)\|BN_RELU_BYPASS(1)` | BN multiply active, no relu |
| `OUT_CVT_SCALE` | `FP32TOFP16_EN(1)\|1` | |
| `OUT_CVT_SHIFT` | `CVT_TYPE(1)\|MINUS_EXP(15)` | Q0.15 → fp16 = `g · 2^-15` |
| `OUT_CVT_OFFSET` | `1` | rounding bias (≈ 1 LSB, negligible) |
| `LE_SLOPE_SCALE / SHIFT` | `23107 / 22` | underflow/overflow extrapolation slope |
| `EW_CFG` | `EW_RELU_BYPASS(1)\|EW_OP_CVT_BYPASS(1)` | **LUT runs** (`EW_LUT_BYPASS` left clear, bit7=0) |

**HardSigmoid reuses every one of these constants** — only the table content
changes (`clip(x/6+0.5,0,1)`), because its output is also in `[0,1]` on the same
grid. Any `[0,1]`-codomain `f` drops in the same way (the runtime's `build_lut_unit`).

## Framing (rocket task shape)

The op uses the **same task framing as `gen_matmul_task`** (the rocket-proven
shape), not the older rknpu-style framing: arm `DPU_S_POINTER`/`DPU_RDMA_S_POINTER`
with `0xE`, then the register content, then the trailer
`OP_NONE · OP_REG_PC(PC_REGISTER_AMOUNTS) · OP_40 · OP_ENABLE`. The **enable word
is `0x18`** = `RESERVED_0(12)` (DPU + DPU_RDMA block bits, `OP_EN=0`) — *not* the
matmul's `0x1D`, because no CNA/CORE participate. (`ROCKET_LUT_ENABLE` /
`ROCKET_LUT_SPTR` env-override these for bring-up.) The trailer target encoding is our
`OP_REG_DPU = BLOCK_DPU|PC_OP_01` (`DPU 0x1000 → 0x1001`).

## Large `n` — tiling, and the max-width corruption (QUIRK 3)

The LUT op carries the flat vector as a cube of `cols = n/8` width positions (C2=8 fp16 each);
`DPU_DATA_CUBE_WIDTH` is **13 bits**, so the hard ceiling is `n ≤ 65528`. A transformer's `[M,I]`
activation cube is millions of elements, so `run_dpu_lut` **tiles** over a per-op cap, and
every caller — SiLU/sigmoid/tanh/GELU/leaky/sqrt/rsqrt/reciprocal — works at any `n`.
Without tiling the op `gen`-fails (`-2`) and returns garbage for `n > 65528`.

**NPU FACT — do NOT ride the maximum width.** A chunk at the *exact* 13-bit max (`cols = 8191`,
`n = 65528`) **corrupts a thin tail of cube positions** — HW-confirmed: a 65528-element SiLU chunk
mis-computed **~54** elements (constant count, data-independent, absent at small `n`), so cosine
dipped to 0.9990. Tiling **well under** the ceiling at `DPU_LUT_MAXN = 32768` (`cols = 4096`) is
**bit-clean** (geglu/FFN cos = 1.000000, 0 misses at every size). Same family as the
CBUF-bank-slack over-read and the sub-4 PPU intermediate: edge-of-register-range positions are
unsafe; stay off the ceiling. (Gate: `tests/ffn_rocket.c` at `n = 65536` / `100000`.)

**The 13-bit ceiling is general, not LUT-only.** [source-confirmed] The same register
limit is reported externally (SHARD, EuroMLSys '26) as an `0xe010 "REGTASK Overflow"`
that fires for **any operand index > 8191** — e.g. large transposes — independent of the
LUT path. Treat 8191 as a hard silicon ceiling on every 13-bit cube-dimension field
(`DPU_DATA_CUBE_WIDTH` / `HEIGHT` / `CHANNEL` and the PPU/RDMA mirrors), and tile below
it; the codebase masks these fields to `& 0x1FFF` and range-checks `> 0x1FFF`.

**Single-pass GELU is unreliable.** A single LUT covering the whole GELU curve mis-decodes on
the standalone flying path (cos ≈ 0.05) and fails at FFN scale even when fused into a conv (the
flat-tail mux spike, QUIRK 1 — see the GELU section below). The accurate on-NPU GELU is the
**2-pass `x·Φ(x)`** route that `rocket_activation_fp16(GELU)` uses (cos = 1.000000); SiLU is
likewise 2-pass and clean standalone.

## Two-operand EW (HardSwish/SiLU multiply) — needs a conv main feed

`x · gate(x)` (HardSwish, SiLU) needs an **elementwise multiply of two buffers**.
The **flying** attempt (`gen_ew_mul_fp16`: EW MUL sub-unit `EW_OP_TYPE=1`,
`EW_OP_SRC=1`, per-pixel `EW_DATA_MODE`/`ERDMA_DATA_MODE`, 16-bit `EDATA_SIZE`) writes
**all-zero** — the ERDMA operand reads 0. Sweeping `COMB_USE` 0..7 +
`SURF_NOTCH`/`EW_SURF_NOTCH` did nothing.

**The Teflon RE settled why** (capture + Mesa `rkt_regcmd.c` source — see
[../teflon-add-capture/](../teflon-add-capture/)). Mesa only ever drives a second EW
operand as the **residual of a conv** (`add_tensor`), and in *both* the plain-conv
and the residual cases the DPU **main feed is the conv / CACC**:

- plain conv: `DPU_RDMA_FEATURE_MODE_CFG = BURST(15) | MRDMA_DISABLE(1)`, ERDMA off.
- conv + residual: `... | COMB_USE(5)` (MRDMA **enabled**), and **both** `SRC_BASE`
  (MRDMA) **and** `EW_BASE` (ERDMA) point at the operand tensor;
  `EW_CFG = EW_ALU_ALGO(2=add) | EW_OP_SRC(1) | EW_DATA_MODE(1) | EDATA_SIZE(1)`.

So MRDMA is either OFF (plain) or repurposed via `COMB_USE(5)` to deliver the
*operand* — it is **never** simultaneously a flying *main* and an operand feed. A
two-buffer EW needs MRDMA for the operand, which leaves no main → operand reads 0.
**A fully-on-NPU two-buffer EW multiply therefore requires a conv as the main feed.**

**An identity conv supplies the main feed.** Reusing the
exact fp16 K-accum eltwise path (`gen_matmul_fp16` `accumulate=1`) but with an
identity weight matrix, the conv reproduces operand `A` into CACC as the EW main, and
the EW unit **multiplies** it by the ERDMA operand `B`:

| field | eltwise-ADD (K-accum) | eltwise-MUL |
|---|---|---|
| `DPU_EW_CFG` (`0x4070`) | `0x108202C0` `EW_ALU_ALGO(2)`,`EW_OP_TYPE(0)` | **`0x108003C4`** `EW_ALU_ALGO(0)`,**`EW_OP_TYPE(1)`**,`EW_OP_CVT_BYPASS(1)` |
| `DPU_RDMA_ERDMA_CFG` (`0x5034`) | `0x40000008` | `0x40000008` (same) |
| `FEATURE_MODE_CFG` (`0x5044`) | `…\|COMB_USE(5)` | same (op-independent) |
| `SRC_BASE`/`EW_BASE` | operand / operand+1surf | same |

i.e. **only `DPU_EW_CFG` changes** (clear the ALU algo, set `EW_OP_TYPE(1)` — the
fp16 eltwise-multiply word). The operand DMA transport (ERDMA +
`COMB_USE(5)` + the `MAX(M,12)` surface-stride floor) is identical to K-accum, so the
caller must keep **M ≥ 12** (below the floor the upper channel surfaces mis-stride).
HW-validated **bit-exact** (`tests/ew_mul_rocket.c`: ADD reproduces `A+B`, MUL gives
`A*B`, both max_abs 0). The flying `gen_ew_mul_fp16` stays behind
`ROCKET_ACT_EXPERIMENTAL=1` as the negative record.

**Subtract** (`rocket_ew_sub_fp16`) needs no new regcmd: `a-b == a+(-b)`,
and fp16 negation is an exact sign-bit flip, so SUB packs the operand negated and runs
the **ADD** datapath unchanged — bit-for-bit the add result. The `ew_mul_rocket` runtime
check now sweeps add/sub/mul (n up to 40000, M-tile crossing), all max_abs 0. Covers the
ONNX/TFLite `SUB` op on the flat-vector EW path.

**Max / Min** (`rocket_ew_max_fp16` / `rocket_ew_min_fp16`) — **NPU FACT: the EW
ALU's `EW_ALU_ALGO` field reaches MAX and MIN, not just SUM(add).** The field is `DPU_EW_CFG`
**bits [17:16]** (so `2<<16 = 0x20000` is the `EW_ALU_ALGO(2)=SUM` bit in the `0x108202C0` add
word), and the NVDLA SDP X/Y ALU algo encoding holds: **0 = MAX, 1 = MIN, 2 = SUM**. So on the
*identical* conv-main EW datapath as add (identity-conv main + ERDMA operand + `COMB_USE(5)`),
only the `DPU_EW_CFG` word changes:

| op | `DPU_EW_CFG` | `EW_ALU_ALGO` |
|---|---|---|
| ADD | `0x108202C0` | 2 (SUM) |
| MAX | `0x108002C0` | 0 |
| MIN | `0x108102C0` | 1 |

Plumbed as `matmul_params_t.ew_op` (2=MAX, 3=MIN; 0 = the legacy add/`ew_mul` path). MAX/MIN
merely *select* one of the two fp16 operands, so they are **bit-exact** (`tests/ew_minmax_rocket.c`,
n to 40000: max_abs 0). Covers TFLite/ONNX `Maximum`/`Minimum`, **ReLU** = `max(x,0)`, and (with a
constant operand) **Clip** = `min(max(x,lo),hi)` (`rocket_clip_fp16`, two passes) and the bounded-
ReLU family. They also build **PReLU** with a per-channel slope — see below.

**Today** HardSwish/SiLU default to **gate-on-NPU-LUT + multiply-on-host** (the
transcendental is off the CPU; `tests/activation_lut_rocket.c` max_abs 0.001 / 0.012),
or run **fully on the NPU** (`ROCKET_ACT_NPU_MUL=1` → `rocket_ew_mul_fp16`, the
identity-conv mul). Host-mul stays default because a standalone EW-mul is a second NPU
round-trip; the perf path is **fusing** the mul into the producing conv (the
conv→`LUT(y)`→mul-by-main single pass), for which this proves the EW mechanism.

## Signed / wide output — the affine OUT_CVT

The `[0,1]` path above is the special case `OUT_CVT_OFFSET=1, MINUS_EXP=15, SCALE=1`. The
output converter is a general **affine, signed, pre-shift** map (HW-confirmed):

```
out_fp16 = (q_interp + OUT_CVT_OFFSET) * 2^-MINUS_EXP * OUT_CVT_SCALE     (FP32TOFP16_EN)
```

- `OUT_CVT_OFFSET` is **signed** and added **before** the shift (the `+1` of the [0,1]
  path is the same field as a Q-domain rounding bias; Mesa's int8 requant uses it as a
  signed `ozp-0x80`). Confirmed by **tanh** on the first HW run: store `(tanh(x)+1)/2` in
  Q0.15 and decode `tanh = (q - 16384) * 2^-14`, i.e. `OFFSET = -16384 (0xFFFFC000)`,
  `MINUS_EXP = 14`. max_abs 0.0034.
- **Bias trick for any bounded range `[lo, hi]`:** store `g = (f(x)-lo)/S` (`S=2^Sexp ≥
  hi-lo`), then `MINUS_EXP = 15-Sexp`, `OFFSET = round(lo·32768/S)`, `SCALE=1`. Drives
  single-pass **tanh** (S=2), **SiLU** (S=32, X=±16), and **HardSwish over the knee
  [-3,3]** (S=4) — all HW-validated (`tests/lut_tanh_rocket.c`, `build_lut_affine` in
  `rocket_activation.c`). This is the **single-pass** activation (no gate+EW-mul).

## Positive-domain kinds: sqrt / rsqrt / reciprocal

**The DPU LUT computes the reciprocal family** (`x>0`: `√x`, `1/√x`, `1/x`) — HW-validated
(`tests/recip_rsqrt_rocket.c`). The realisation is the **shifted single-table** mode (the
`ROCKET_LUT_SHIFT` path, here unconditional for these kinds): map the whole positive domain
`[x_lo,x_hi]` onto the LO (positive index) half via `index = (x − x_lo)·scale`,
`scale = 16384/(x_hi−x_lo)` (BN-MUL), `BN-ALU = −x_lo·scale` (fp32, post-scale). Because `x`
never reaches 0, the **LE/LO sign mux never fires** — none of the x≈0 glitch that dogs the
symmetric kinds. The OUT_CVT is the same affine bias-trick (`out_lo=0`, `S=2^Sexp ≥ max f`).

**NPU FACT — uniform-grid accuracy is domain-bounded.** The 513-entry LO table samples `x`
**uniformly**, so for these steep functions the worst error is at the low end. The low-end
relative interpolation error scales as `≈ (Δ/x_lo)²·(c/8)` with `Δ=(x_hi−x_lo)/512`, so a
domain *ratio* `x_hi/x_lo ≲ 128` placed away from 0 keeps it ~1%. Measured on HW over a
~100–200× domain: **sqrt 0.85%, rsqrt 0.44%, reciprocal 1.0%** max-relative (means
0.05–0.12%). Inputs outside `[x_lo,x_hi]` clamp to the edge value. Defaults
(`act_positive_domain`): sqrt `[0.25,64]`, rsqrt `[0.5,64]`, reciprocal `[0.25,32]`; tune to
the data with `ROCKET_LUT_XLO/XHI`. A genuinely wide dynamic range would want a **log-domain**
LUT (`1/x = exp(−ln x)` is straight in log x) — the obvious follow-on, not yet built.

These unlock **on-NPU `Div`** (`rocket_ew_div_fp16 = a·reciprocal(b)`, HW 0.35%) and are the
math core of **RMSNorm/LayerNorm** (`rsqrt(mean(x²)+ε)`) and the **softmax denominator** —
the normalization primitives for the LLM/Whisper FFN-fusion work.

## LOG — the first signed-output positive-domain kind

`ln(x)` (`ROCKET_ACTIVATION_LOG`, `x>0`) joins the positive-domain shifted-table family — the
natural inverse of EXP, the per-element log for log-probabilities / NLL / cross-entropy
(`tests/recip_rsqrt_rocket.c`). It is the **first `act_shifted_domain` kind with a SIGNED output**:
`log(x)<0` for `x<1`, so unlike sqrt/rsqrt/reciprocal/exp (all `out_lo=0`) it sets `out_lo=log(x_lo)`
**< 0**, and the **OUT_CVT offset** `lround(out_lo·32768/S)` (now negative) decodes the signed range.
This proves the generic positive-domain path already carried the signed-output machinery (the same
negative-`OUT_CVT_OFFSET` decode RE'd for tanh/ELU in "Signed/wide output" above) — it just had never
been exercised with `out_lo≠0`. No new HW path; default domain `[0.25,32]`, `S=8` covers
`[log .25, log 32]≈[-1.39,3.47]`.

**NPU FACT — for LOG the right error metric is ABSOLUTE, not relative.** `log` crosses zero at
`x=1`, where relative error is ill-defined (`÷0`); and log is consumed *additively*
(`log(ab)=log a+log b`), so absolute error is what propagates. Uniform-grid ⇒ worst at the steep
small-x end (`d²/dx² log = −1/x²`). Measured on HW over `[0.3,30]`: **max_abs 0.0066 (@x≈0.33),
mean 0.0007** — on par with the reciprocal family. (Same caveat as the others: a genuinely wide
dynamic range wants a companded/log-domain grid; the uniform grid is domain-bounded.) The signed
output decodes via the negative offset, so per QUIRK 2 a standalone LOG that included `x≤0` would
spike at x≈0 — but its domain is `x>0` (interior to the LO half), so it is glitch-free like the rest
of the positive-domain family. NOTE: the per-row `log` inside **LogSoftmax** is HOST (M scalars,
exact) — the LOG LUT is for a *large* tensor of logs, not the softmax denominator (see
[whisper-encoder.md](whisper-encoder.md)).

## EXP — the softmax numerator

`exp(x)` joins the shifted-single-table family (`ROCKET_ACTIVATION_EXP`, `act_shifted_domain`),
default domain **`[-16,0]`** — the softmax case: after the mandatory row-max subtraction the input
is in `(-∞,0]`, output `(0,1]` (`out_lo=0`, `S=1`). Unlike the symmetric kinds the domain may
include `x≤0` and still avoid the LE/LO sign mux, because the BN-ALU bias maps the whole domain onto
the positive INDEX half — so **EXP works on the STANDALONE flying path** (unlike `build_lut_affine`
GELU). exp's relative interpolation error on a uniform grid is ~**constant** `Δ²/8` (≈1e-4 over 512
cells) because `f''/f = 1` — a good fit for a uniform LUT. HW-validated `tests/exp_lut_rocket.c`:
sweep `[-16,0]` max_abs 5.5e-4, and the **softmax-sum end-to-end** (row-max subtracted, T up to 512,
score spread to 20) sum_rel ≤0.04%, max|Δp| ≤6e-5.

### QUIRK 4 — a q=0 LUT table entry mis-decodes to a garbage ~4.0 (not 0) [HW sweep]

A **zero-valued LUT table entry** trips a decode fault in the output converter: it emits a
constant **~4.0**, not 0. EXP surfaces it — the output is correct down to `x≈-10` (table entry
`q=lround(f·32768)≥1`) and jumps to ~4.0 for `x≤-11`, exactly where `exp(x)<1.5e-5` quantizes the
entry to `q=0`. For EXP this blew up the softmax sum (the whole deep
tail read ~4 each). **Fix: floor every shifted-table entry to `q≥1`** (`build_lut_shifted`,
`ROCKET_LUT_QFLOOR`, default 1) — the floored value decodes to ~3e-5 (rounds to ~0 on readback),
correct for the tail. sqrt/rsqrt/reciprocal never produce a q=0 entry over their domains, so the
floor is a no-op for them (re-gated green). This likely also lurks in the sigmoid/tanh **LE** deep
tail (`sigmoid(-16)≈0 → q=0`) but those paths aren't ridden that deep in-model — flagged, not chased.

## conv → activation fusion — the LUT epilogue inside `gen_conv2d_task`

The single-pass LUT epilogue runs not only as a standalone flying op but **fused into a
conv**: a DIRECT fp16 conv post-processes its own result with `f(x)` in the *same* NPU job
(`out = f(conv(x))`, no second round-trip). The fusion = the validated **fp16-OUT conv**
(output geometry, `size_e=1`, NC1HWC2 readback all unchanged) **plus the BN-mul → EW-LUT →
affine-OUT_CVT epilogue** — identical registers to the standalone op, only the input source
differs (the conv **CACC accumulator** instead of a flying MRDMA stream). Because the SDP
stages sit downstream of the DPU main feed, the conv-fed epilogue is byte-identical, and it
is **provably correct**: HW shows the fused result matches the standalone `gen_lut_activation_fp16`
applied to the same conv output to ≤ 0.0039 (the LUT quant step) across 1×1/3×3/stride-2/tiled
shapes. Driven by `npu_dpu_desc.lut_en` + `lut_ep` (a `lut_epilogue_t`); default-off is
byte-identical (the regcmd adds **exactly +1026** ops — the LE/LO upload — and nothing else).
`rocket_conv2d_act_fp16` (rocket_conv.h) wires it for SiLU/tanh/GELU; **conv→tanh is bit-accurate**
vs the true function.

## The NVDLA hybrid-LUT (LE/LO) + the flat-region mux quirk

The DPU LUT is the NVDLA SDP hybrid pair: **LE = X/raw** table (full range) + **LO =
Y/density** table (high-res small range), selected by `LUT_LO_LE_MUX` + the `Priority /
OverflowPriority / UnderflowPriority` registers. Out-of-table inputs extrapolate linearly:
`LUT[0]+(X-START)·UFLOW_SCALE/SHIFT` (underflow, `DPU_LUT_LE_SLOPE_*`) and
`LUT[N]+(X-END)·OFLOW_SCALE/SHIFT` (overflow, `DPU_LUT_LO_SLOPE_*`, now emitted by
`gen_lut_activation_fp16`). NVDLA v1 sizes LE=65 / LO=257 entries; **RK3588 uses 513 each**
(`lut[1026]`). Five stats counters (`XHitNum/YHitNum/UnderflowNum/OverflowNum/PriorityNum`)
report per-layer selection — useful diagnostics, but treat any counter read with the
box-safe pattern (some NPU counter pages hard-lock the SoC — see
[../perf/hw-byte-counters.md](../perf/hw-byte-counters.md)).

**QUIRK (HW-observed):** in **flat / asymptotic** regions (zero derivative) the LE/LO
overflow mux **mis-toggles** at register-saturation boundaries, producing a config-
independent garbage spike (a discrete glitch, not a slope error). HardSwish (exactly 0 for
x≤-3) trips it whether the flat run is **in-table** (wide table → +128) or pushed into
**underflow extrapolation** (knee table → +16); only the curved knee `[-3,3]` is clean.
Smooth activations (sigmoid/tanh/SiLU) never have an exactly-flat run, so they're fine
*away from the table join* (tanh's saturating tail uses the tuned underflow slope `23107>>22`,
NOT slope 0). NVDLA's IAS gives no flat-region recipe → the practical answer for HardSwish is
the **2-pass gate+EW-mul** (or host) path, i.e. "bypass the LUT for constant regions."

**QUIRK 2 — the x≈0 LE/LO boundary spike [HW sweep].** A second, distinct mux glitch sits at the
**LE↔LO table join** (`index = x·index_scale = 0`, i.e. `x≈0`): when an input lands within ~0.0015
of exactly 0, the hybrid mux mis-toggles and emits a discrete garbage spike (`+128`). It is a
property of the **LUT itself, not the conv→activation fusion** — the standalone flying op and the
fused epilogue spike at the identical elements. The band is razor-thin (~±R/8192), so a
sparse-linspace gate steps over it; dense random conv outputs (N=2.6e4–5e5) hit it a handful of
times. Sample densely when validating a single-pass kind.

**Mechanism: the mux selects on `sign(x)`, not the table index.** The hybrid mux picks LE-vs-LO on
`sign(x·BN_MUL) = sign(x)` — the *pre*-ALU value — so the BN-ALU bias relocates the table *address*
but not the mux *decision*. No index trick (shift, lower-quarter, asymmetric, a different scale) can
dodge it. Decisive experiment (LeakyReLU, a sharp kink exactly at 0): re-mapping x=0 to a different
index **moves the spike to follow x=0** — it is always at the input value 0, never at a fixed index.
A repair-off scan of 16385 inputs uniformly across [-16,16] finds **exactly one** spike, at
`x=0.000000` (`+128`), for every scale. A true single-table mode (mux disabled) would avoid it, but
re-exposes the flat-region quirk (QUIRK 1) for functions with a flat run.

**Which kinds spike: signed-output only.** The spike tracks the **signed output decode** (a negative
`OUT_CVT_OFFSET`), not merely a domain straddling 0. Under a dense `[-0.02,0.02]` step-1e-5 probe
(`tests/x0_glitch_probe.c`), shifted-single-table kinds split cleanly:
- **Unsigned `[0,hi]` output (`OUT_CVT_OFFSET ≥ 0`) — x≈0-clean.** Softplus (`out_lo=0`), Abs
  (symmetric `[-R,R]`, `out_lo=0`), and the 2-pass Mish gate (`[0,1]`, offset `+1`) show no spike
  (worst |Δ| 7e-4 / 1e-3 / 8e-6 — just interp/quant). Same camp as sigmoid/exp/sqrt/rsqrt/recip.
- **Signed output (`OUT_CVT_OFFSET < 0`) — x≈0 spike.** tanh/SiLU/GELU and ELU/SELU (`out_lo=-λα<0`)
  spike at x≈0 (a discrete ~64–128 over a ~±5e-4 band; ELU/SELU 97/4001 in the probe). tanh is
  otherwise clean (no flat run), so x≈0 is its *only* glitch.

So **if you can frame the output decode with `out_lo ≥ 0` (non-negative offset), the kind is
x≈0-clean**; a genuinely-signed output keeps the spike.

**Mitigation — host band-repair.** Because the spike is a razor-thin band at x≈0 and the runtime
already streams every output element on readback, the cheapest robust fix is a **host patch of that
band** with the exact value — done in `rocket_leaky_relu_fp16` (LE = `alpha*x`, LO = `x`;
`ROCKET_LEAKY_NOREPAIR` disables it) and `rocket_elu_fp16` / `rocket_selu_fp16`
(`ROCKET_ELU_NOREPAIR`). HW-validated: `tests/leaky_relu_rocket.c` (alpha ∈ {0.01,0.1,0.125,0.2,
0.25,0.5} all `bad=0`, sweep `max_abs ≤ 0.002`, x≈0 band exact), `tests/elu_rocket.c`,
`tests/softplus_mish_rocket.c`. The repair only works where the **producer can patch the band**: the
standalone activation can; the fused-in-conv epilogue cannot, so robust FFN SiLU/GELU wants the
2-pass path or a single-table-mode RE. **alpha=0 (plain ReLU) is not supported on the leaky path** —
its all-zero negative branch is a flat run that trips QUIRK 1 across the whole LE table; use the
native DPU ReLU.

**BN-ALU bias works in the index domain.** Mapping the whole domain `[x_lo,x_hi]` onto the positive
index half (`ROCKET_LUT_SHIFT`, `build_lut_shifted`) needs a BN bias, and establishes how the BN ALU
bias works: `index = x·BN_MUL + BN_ALU` (**MUL then ALU**), `BN_ALU` an **fp32 bias in the *index*
(post-scale) domain** — not a pre-scale `(x+B)` add. HW-confirmed by sweep
(`BN_ALU=0x46000000=fp32(8192)` → tanh ACC 0.0007; the pre-scale guess `fp32(-x_lo)` is
systematically wrong, ACC≈2). So `BN_ALU = fp32(-x_lo·scale)` (the `0x80000000`=-0.0 the standalone
uses is the no-op case). This relocates the table but does **not** cure x≈0 (the mux is `sign(x)`).

Refs: NVDLA IAS [lut-programming](https://nvdla.org/hw/v1/ias/lut-programming.html) /
[unit_description](https://nvdla.org/hw/v1/ias/unit_description.html) /
[programming_guide](https://nvdla.org/hw/v1/ias/programming_guide.html).

## More activation kinds

All reuse the mechanisms above — no new regcmd primitive. Each has a CTest gate vs
double-precision math.

| kind | mechanism | x≈0 | accuracy (HW) |
|---|---|---|---|
| **Softplus** `log(1+e^x)` | shifted single-table (EXP path), `out_lo=0` | clean | max_rel 0.14% |
| **Mish** `x·tanh(softplus(x))` (YOLOv4/v7) | 2-pass: `[0,1]` gate (build_lut_unit) + EW-mul | clean | max_rel 0.06% |
| **Abs** `\|x\|` | **symmetric** shifted single-table (kink on the middle sample j=256) | clean | max_abs 1e-3 (the q≥1 floor at x=0) |
| **ELU** `x≥0?x:α(e^x−1)` | symmetric shifted single-table + **host x≈0 repair** (signed) | repaired | max_rel 0.37% |
| **SELU** `λ·ELU_α` (fixed α,λ) | as ELU | repaired | max_rel 0.6% |
| **PReLU** per-channel `α_c` | **no LUT** — `max(x,α_c·x)` (α∈[0,1]) or `relu+α·min` (general), via EW max/scale | n/a | bit-exact |

PReLU's per-channel slope rides the EW path, not the LUT (one LUT table can't hold a per-channel
parameter): the per-channel scale is a row-broadcast `ew_mul` (channel = row in a `[C,S]` layout),
then `ew_max` — so it inherits the EW bit-exactness and has **no** x≈0 glitch. `tests/prelu_rocket.c`
(α∈[0,1] max-path + α-outside general path) all bit-exact.

## GELU — the 2-pass `x·Φ(x)` route

The accurate on-NPU GELU is the **2-pass** route, exactly like SiLU: `GELU(x) = x·Φ(x)` where
`Φ(x) = 0.5(1+erf(x/√2))` is the Gaussian CDF — a **monotone `[0,1]`** function, so it rides the
**clean unit-LUT geometry** (`build_lut_unit`, the same one sigmoid uses, index_scale 2596) and is
free of QUIRK 1: Φ has no exactly-flat run, and its saturating tails use the unit-LUT `le_slope`
extrapolation (like sigmoid). `rocket_activation_fp16(GELU)` routes here (gate `ROCKET_ACTIVATION_
GELU_GATE` = Φ, then the EW-mul by x); `ROCKET_ACT_WIDE_LUT` forces the single-pass path for RE.
HW-validated **cos=1.000000, max_abs 0.0016 vs true erf-GELU over `[-12,12]`** (`tests/gelu_rocket.c`),
INCLUDING the flat tails — and it makes the Whisper encoder block fully on-NPU (cos=1.000000).

**Negative result — a fused single-pass matmul→GELU does not work for wide inputs.** Lowering
`C = GELU(A·Bᵀ)` onto a 1×1 conv with the single-pass GELU epilogue (`build_lut_affine`, the same
table the conv→act fusion uses) **fails at FFN scale**: cos≈0.04, `max_abs=128` — the QUIRK-1
flat-tail mux spike, hit en masse because a real fc1 output spans the flat negative region
(`GELU(x)≈0` for `x≲-3`). tanh fares better (cos 0.955) only because its curved range is narrower.
**Single-pass LUT fusion (conv→act or a hypothetical matmul→act) is only safe for inputs that stay
in the curved region; the durable on-NPU GELU/SiLU is the 2-pass `x·gate(x)`.** So a fused
matmul→act (`rocket_matmul_fp16_act`) is not viable for wide inputs.

## Why this matters

- **Detection:** HardSigmoid + HardSwish (the EW-mul now lands) un-spill the modern
  MobileNetV3/MobileDet activation blocks from the CPU — pending the delegate
  `map_activation` wiring + the single-pass conv→hardswish fusion for video rate.
- **LLM:** the same LUT does GELU/SiLU gates for fused FFN / Whisper blocks;
  SiLU/GELU now have the on-NPU EW-mul (or a wider-Q single-pass LUT once the OUT_CVT
  affine form for non-`[0,1]` ranges is RE'd).
- It is the first DPU **post-processing** block we drive beyond the matmul/conv
  requant — the BS/BN/EW/LUT/OUT_CVT machinery is now partially mapped here.

Register fields per Mesa `registers.xml` and our `include/npu_hw.h`; the LUT geometry is
HW-verified (above). See also [../README.md](../README.md), [precision-field.md](precision-field.md),
and [SOURCES.md](../SOURCES.md) for the allbilly encoding reference.
