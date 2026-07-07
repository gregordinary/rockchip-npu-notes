# The precision field (CNA / CORE / DPU)

The NPU's datatype is selected by a **3-bit precision field**, set independently for
the input, the processing/MAC stage, and the output (in the CNA `CONV_CON1` and the
DPU in/proc/out precision registers). The values:

| dtype | precision value | how established |
|---|---:|---|
| **int8** | `0` | [HW sweep] |
| **int16** | `1` | **[HW sweep, this project]** |
| **fp16** | `2` | [HW sweep] |
| **int4** | `6` | **[HW sweep, this project]** |
| **int32** | `4` | [HW sweep] |
| **fp32** | `5` | [HW sweep] |
| **bf16** | `3` | **[HW sweep, this project]** |
| **tf32** | `7` | **[HW sweep, this project — CNA/CORE only]** |

int8=0 and fp16=2 are the baseline datatypes the whole stack runs; int16=1, int4=6,
bf16=3, and tf32=7 were each established by hardware sweep (classifying every output
element as bit-exact / saturated / unwritten-sentinel); int32=4 and fp32=5 are
output-only precisions confirmed by the working fp32-out writer. int16=1 also matches
the NVDLA heritage. The datatype matrix is **complete**.

## int4 = precision value 6

int4 is the readback "escape": its output is int16, and it packs 4× denser than fp16, so
it can reach single-pass K. The encoding is **not** documented anywhere. The element
bit-width is *precision-driven* (a field value), not a separate datapath — as the int8
work established — so int4 is "pick the right precision value + nibble-pack the operands."

A staged standalone gate (`matmul_int4_rocket.c`) classifies each output element as
**bit-exact / saturated / unwritten-sentinel (0xAAAA)**. Sweeping the candidate precision
values on a tiny shape (`M=4 K=32 N=64`, kept small so the int16 output can't overflow and
the compare is exact):

- **`precision=3` → saturated.** The NPU misread the int4 nibbles as if wider (int8-
  like), so values pinned to the int16 max — wrong but "the engine ran."
- **`precision=7` → wrote nothing.** The output stayed at the 0xAAAA sentinel — the
  engine did not accept this precision for this path.
- **`precision=6` → bit-exact.** The only value where the first columns matched the
  int64 CPU reference exactly.

So **int4 = precision value 6** (for both input and processing; output is int16,
precision=1). The nibble packing is the same as int8's, just reinterpreted 2-per-byte.
Two *geometry* details matter beyond the encoding: the int16 output stride (`size_e`, see
[size-e-quirk.md](size-e-quirk.md)) and the int4 weight N-group of 64 (see
[tile-layouts.md](tile-layouts.md)). With those correct, **int4×int4→int16 matmul is
bit-exact on all tested shapes** (M∈{1,4,8,64,128}, K∈{32,64,256}, N∈{64,128,256},
including M=1 GEMV).

## int16 = precision value 1 (and the rung that has no native matmul output)

The same staged gate (`matmul_int16_rocket.c`) sweeps `precision` 1..7 on
`M=4 K=32 N=64`:

- **`precision=1` → correct int16×int16 dot products** (bit-exact vs the int64 CPU
  reference for the elements the engine wrote). The only value that computes int16.
- **`2/3/4/5/7` → wrong**, **`6` → garbage** (that's int4, which fills the buffer
  with nibble-misread values).

So **int16 = precision value 1** (input + processing), confirming the NVDLA lineage
by HW. But int16 has a wrinkle the other dtypes don't — see
[output-transpose-int16.md](output-transpose-int16.md): there is **no native int16
matmul *output*.** The int16 conv engine
computes correctly, but its output writer can only emit a single int32 tile
(iteration broken) or a full int16-**saturating** transposed buffer (`tp_org_en`),
never full-iteration int32. Full-precision int16 is therefore done by **int8 byte
decomposition** (4 int8 matmuls; `rocket_matmul_int16_exact`), not a native path.

## bf16 = precision value 3, tf32 = 7 (HW sweep)

bf16 and tf32 were established by the same hardware sweep as int4/int16 — sweeping the
precision value on a small shape and classifying the output. On a bf16-formatted matmul
only `3` produces a correct fp32 result; on a tf32 (raw-fp32-input) matmul only `7` (at
the front of the pipe) does. Both then verify end-to-end (below), and bf16 runs at the
same MAC rate as fp16 (same 2-byte operand).

| dtype | precision value | source |
|---|---:|---|
| **bf16** | `3` | **[HW sweep, this project]** |
| **tf32** | `7` | **[HW sweep, this project — CNA in/proc only]** |

The CNA/CORE in/proc precision values and the DPU output precision values **differ in
the upper slots**: at the front of the pipe `7 = tf32` (`4/5` unused); the DPU output
stage has `4 = int32, 5 = fp32` and no tf32 slot — which is why tf32 must ride the fp32
output (HW-confirmed below: setting the DPU stage to 7 writes nothing). Consistent
across both, `0..3 = int8/int16/fp16/bf16` and `6 = int4`.

**Caveat — MAC-capable ≠ usable matmul output path.** int16 computes correct products
in the MAC array yet has **no** native matmul output (its int32 writer iterates only one
tile). So bf16 still needs the standalone gate. The outlook is **much** better than int16, though: bf16 accumulates
to **fp32**, so its output reuses the fp16 path's *proven* fp32-out writer
(`out_precision=5`, `size_e=3`, `surf×4`, output cube **C2=4**) — the exact writer
that fully iterates M×N every prefill. A host regcmd diff confirms `gen_matmul_bf16`
== `gen_matmul_fp16`'s fp32-out program with **only** the in/proc precision word
changed 2→3 (3 words: `CNA_CONV_CON1`, `CORE_MISC_CFG`, `DPU_DATA_FORMAT`).

The one thing int4's `3 saturates / 7 writes nothing` does **not** tell us: that was
int4-**nibble** data fed at those precisions (wrong-format input). bf16 at precision
3 needs bf16-formatted 2-byte operands.

**HW-confirmed (2026-06-18): bf16 matmul works at precision 3.** The gate
(`matmul_bf16_rocket`) passes across the shape ladder (M∈{1,4,64,256}, N∈{64..256},
K∈{32..4096}, incl. M=1 GEMV): the DPU writes a full M×N fp32 result that tracks the fp32
reference at **max_rel ~1e-6** — i.e. **exact bf16 products + fp32 accumulate**, no
accumulator lossiness. A BIG-range run
(|values|~1e5, products ~1.9e10, well past fp16's 65504 ceiling) also passes, which
is the whole point: bf16 carries the range fp16 can't, so the per-row activation
scaling can be dropped. The tiled path (`rocket_matmul_bf16`) is bit-clean to
512×15360×3840. So bf16 is the OPPOSITE of int16: a fully usable native matmul
datatype, because its fp32 output rides the already-proven fp16 fp32-out writer.

## tf32 = precision value 7 (the first 4-byte-input matmul — HW-confirmed)

Encoding HW-established (precision `7` on the CNA/CORE stage; tf32 runs at about half the
fp16/bf16 rate — measured ~37 GOP/s on big-Gemma below). tf32 is 1 sign + 8-bit exponent (fp32 range) + 10-bit mantissa
(fp16 precision) in a **4-byte fp32 container** — the first 4-byte-input matmul
(fp16/bf16/int16 2-byte, int8 1-byte, int4 ½-byte; int32/fp32 are only ever outputs). You
feed raw fp32; the MAC rounds to a 10-bit mantissa, multiplies, and **accumulates in
fp32**. HW-confirmed genuine NVIDIA-style tf32: a random matmul tracks a tf32-rounded
reference to **max rel ~1.5e-7** while differing from a full-fp32 ref by ~8e-4 (the
10-bit-mantissa gap), and a |values|>65504 run passes (fp32 range).

**Final geometry (HW-confirmed). The non-obvious part: for a 4-byte element the weight
K-group is 16, not 32 — the N-group stays 16, but a 4-byte input halves the K-group:**

| param | value | note |
|---|---|---|
| precision CNA in/proc, CORE proc | **7** | tf32 (front of pipe only) |
| precision DPU in/proc/out | **5** | fp32 — DPU enum has no tf32; rides the fp32 accumulator |
| feature cube **C2** | **4** | 16-byte CBUF atom / 4 (confirmed) |
| weight tile | **(N/16, K/16, 16, 16)** | N-group 16 (== fp16), **K-group 16 (HALVED from fp16's 32)**; still a 1024-byte tile (16·16·4) |
| **data_entries** | **K/16** | = number of K-groups (cf. fp16 K/32 at KG=32) |
| output | fp32 cube C2=4, size_e=3, surf×4 | the proven fp16/bf16 fp32-out writer |

`gen_matmul_tf32` (precision per-stage, element 4 B, data_entries K/16) + `weight_tf32`
(N/16,K/16,16,16) + the standalone gate `matmul_tf32_rocket` (raw fp32 in, dual-reference
precision characterization, structured operand patterns) — all pass. Single-task CBUF
limit (4-byte doubles feature bytes): M·K·4 ≤ 11 banks (360448 B) and K ≤ 8192; big shapes
tile.

**Tiled path HW-validated (2026-06-18).** `rocket_matmul_tf32` / `rocket_matmul_plan_tf32`
(`rocket_matmul.c`) is a clone of the bf16 tiled path with the 4-byte geometry — `float`
slots K-aligned to 16, **RAW fp32 scatter (no truncation; the HW rounds the mantissa)**,
banks ×4, Kt ≤ 8192 — and the sample-verified test `matmul_tf32_tiled_rocket.c`. The index
helpers (`feat_idx_tf32` C2=4, `wt_idx_tf32` (N/16,K/16,16,16)) were host-diffed bit-exact
against `weight_tf32` / `feature_data(C2=4)` before HW. All shapes PASS at norm_err ~1e-7,
**including `4 48 64` (K=48 = %16-not-%32)** — the case the single-task gate could never
reach (its `main()` required K%32, so K∈{32,64,128} are all %32). That confirms the K-group
really is 16 on hardware and the K%16 plan alignment is correct (no K%32 fallback). Big-Gemma
`512×3840×4096` runs at ~37 GOP/s.

**Finding the geometry.** tf32=7 is a front-of-pipe-only code: with all stages set to 7
the DPU writes nothing (its out enum is 0..6), so DPU in/proc/out must be fp32 (5) and only
CNA/CORE carry 7 (CORE=5 → 1e25 garbage). Structured operand patterns (`ROCKET_TF32_PAT`:
ones / k-ramp / n-col / m-row / K-impulse) localize a wrong weight tile.

The trap is the weight K-group. At a single-K-group shape (K=32) the weight index is
row-major for *any* N-group/K-group, so K=32 cannot test the weight layout at all: a
half-rate `(N/8, K/32, 8, 32)` guess shows only N/2 distinct output channels and a
K-misalignment there (insensitive to every host knob), which looks like a structural
half-rate hardware limit — and matches the "256×3" rate — when it is just the wrong K-group
splitting the contraction across two output lanes. Testing at K=64/128 distinguishes them
and confirms **`(N/16, K/16, 16, 16)`**. **Rule: never RE a weight tile at a single-K-group
shape; test at K ≥ 2× the candidate K-group.**

## Key principle: precision is a field, not a datapath

The reason int8 and int4 were "weeks, not a driver rewrite": the emit layer
(`gen_matmul_task`) is **datatype-agnostic** — it writes proc/in/out precision,
data_sign, and the cvt_* fields straight from descriptor structs. The dtype lives
only in the descriptor *setup* (which precision value, which element bytes, which
tile layout, which output cube). Adding a datatype = a new precision value + the
sub-byte math + the nibble/tile layout, reusing the same register path.

The matmul output type per input dtype (HW-established):
**int8×int8→int32, int4×int4→int16, fp16×fp16→fp32.** Even the int8 matmul
outputs raw int32 — nobody does on-device integer requant inside the matmul. The
scales/dequant are a host-side concern.
