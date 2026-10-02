# The precision field (CNA / CORE / DPU)

A 3-bit precision field selects the NPU's datatype. The field is set independently for the
input, the processing (MAC) stage, and the output, in the CNA `CONV_CON1` and the DPU
in/proc/out precision registers. The values:

| datatype | precision value | how established |
|---|---:|---|
| int8 | `0` | [HW sweep] |
| int16 | `1` | [HW sweep, this project] |
| fp16 | `2` | [HW sweep] |
| int4 | `6` | [HW sweep, this project] |
| int32 | `4` | [HW sweep] |
| fp32 | `5` | [HW sweep] |
| bf16 | `3` | [HW sweep, this project] |
| tf32 | `7` | [HW sweep, this project, CNA/CORE only] |

int8=0 and fp16=2 are the baseline datatypes the whole stack runs. A hardware sweep
established each of int16=1, int4=6, bf16=3, and tf32=7. The sweep classifies every output
element as bit-exact, saturated, or unwritten sentinel. int32=4 and fp32=5 are output-only
precisions, confirmed by the working fp32-out writer. int16=1 also matches the NVDLA
heritage. The datatype matrix is complete.

## int4 (precision value 6)

int4 is the readback "escape". Its output is int16, and it packs 4x denser than fp16, so
it can reach single-pass K. The encoding is not documented anywhere. The element bit-width
is precision-driven: it follows a field value and has no separate datapath, as the int8
work established. So int4 is "pick the right precision value and nibble-pack the
operands".

A staged standalone gate (`matmul_int4_rocket.c`) classifies each output element as
bit-exact, saturated, or unwritten sentinel (0xAAAA). The gate sweeps the candidate
precision values on a tiny shape (`M=4 K=32 N=64`). The shape is small so that the int16
output cannot overflow and the compare is exact:

- **`precision=3`** saturates. The NPU misreads the int4 nibbles as a wider type
  (int8-like), so values pin to the int16 max. The result is wrong, but "the engine ran".
- **`precision=7`** writes nothing. The output stays at the 0xAAAA sentinel. The engine
  does not accept this precision for this path.
- **`precision=6`** is bit-exact. It is the only value where the first columns match the
  int64 CPU reference exactly.

So int4 is precision value 6 for both input and processing. The output is int16,
precision=1. The nibble packing is the same as int8's, reinterpreted 2-per-byte. Two
geometry details matter beyond the encoding: the int16 output stride (`size_e`, see
[size-e-quirk.md](size-e-quirk.md)) and the int4 weight N-group of 64 (see
[tile-layouts.md](tile-layouts.md)). With those correct, the int4×int4->int16 matmul is
bit-exact on all tested shapes (M∈{1,4,8,64,128}, K∈{32,64,256}, N∈{64,128,256},
including M=1 GEMV).

## int16 (precision value 1)

The same staged gate (`matmul_int16_rocket.c`) sweeps `precision` 1..7 on
`M=4 K=32 N=64`:

- **`precision=1`** computes correct int16×int16 dot products, bit-exact against the
  int64 CPU reference for the elements the engine wrote. It is the only value that
  computes int16.
- **`2/3/4/5/7`** are wrong, and **`6`** is garbage. Value 6 is int4, which fills the
  buffer with nibble-misread values.

So int16 is precision value 1 for input and processing, which confirms the NVDLA lineage
on hardware. Its int32 output takes fp16 -> fp32's output geometry, `size_e` 3, and
saturates. int8's `size_e` 7 hangs the task after one tile. See
[output-transpose-int16.md](output-transpose-int16.md).

## bf16 (precision value 3) and tf32 (precision value 7)

The hardware sweep that established int4 and int16 also established bf16 and tf32
[HW sweep]. It sweeps the precision value on a small shape and classifies the output. On a
bf16-formatted matmul only `3` produces a correct fp32 result. On a tf32 (raw-fp32-input)
matmul only `7` does, in the CNA/CORE in/proc stages. Both then verify end-to-end (below).
bf16 runs at the same MAC rate as fp16 (same 2-byte operand).

| datatype | precision value | source |
|---|---:|---|
| bf16 | `3` | [HW sweep, this project] |
| tf32 | `7` | [HW sweep, this project, CNA in/proc only] |

The CNA/CORE in/proc precision values and the DPU output precision values differ in the
upper slots. In the CNA/CORE in/proc stages, `7 = tf32` and `4/5` are unused. The DPU
output stage has `4 = int32, 5 = fp32` and no tf32 slot, so tf32 must use the fp32 output.
Setting the DPU stage to 7 writes nothing (hardware-confirmed below). Both sets agree on
`0..3 = int8/int16/fp16/bf16` and `6 = int4`.

MAC-capable is not the same as a usable matmul output path. int16 computes correct
products in the MAC array, and its int32 writer saturates and takes fp16's output geometry,
not int8's ([output-transpose-int16.md](output-transpose-int16.md)). So bf16 still needs the
standalone gate. bf16 accumulates to fp32, so its output reuses the fp16 path's proven
fp32-out writer. That
writer is `out_precision=5`, `size_e=3`, `surf×4`, output cube C2=4, and it fully iterates
M×N on every prefill.

A host regcmd diff confirms that `gen_matmul_bf16` == `gen_matmul_fp16`'s fp32-out program
with only the in/proc precision word changed 2->3. The change is 3 words:
`CNA_CONV_CON1`, `CORE_MISC_CFG`, `DPU_DATA_FORMAT`.

The int4 sweep's `3 saturates / 7 writes nothing` result does not rule out precisions 3
and 7. That sweep fed int4 nibble data at those precisions, which is wrong-format input.
bf16 at precision 3 needs bf16-formatted 2-byte operands.

The bf16 matmul works at precision 3 [HW sweep 2026-06-18]. The gate
(`matmul_bf16_rocket`) passes across the shape ladder (M∈{1,4,64,256}, N∈{64..256},
K∈{32..4096}, including M=1 GEMV). The DPU writes a full M×N fp32 result that tracks the
fp32 reference at max_rel ~1e-6. That is exact bf16 products with an fp32 accumulate and
no accumulator lossiness.

A big-range run (|values|~1e5, products ~1.9e10, well past fp16's 65504 ceiling) also
passes. That range is the purpose of bf16: it carries the range that fp16 cannot, so a
caller can drop the per-row activation scaling. The tiled path (`rocket_matmul_bf16`) is
bit-clean to 512×15360×3840. So bf16 is a fully usable native matmul datatype, because its
fp32 output uses the proven fp16 fp32-out writer.

## tf32 (precision value 7)

The tf32 encoding is established on hardware: precision `7` on the CNA/CORE stage. tf32
runs at about half the fp16/bf16 rate, measured ~37 GOP/s on big-Gemma (below).

tf32 is 1 sign, an 8-bit exponent (fp32 range) and a 10-bit mantissa (fp16 precision) in
a 4-byte fp32 container. It is the first 4-byte-input matmul. fp16, bf16 and int16 inputs
are 2-byte, int8 is 1-byte and int4 is ½-byte. int32 and fp32 are only ever outputs. You
feed raw fp32. The MAC rounds to a 10-bit mantissa, multiplies, and accumulates in fp32.

The hardware confirms genuine NVIDIA-style tf32. A random matmul tracks a tf32-rounded
reference to max rel ~1.5e-7 and differs from a full-fp32 reference by ~8e-4 (the
10-bit-mantissa gap). A |values|>65504 run passes (fp32 range).

### Geometry

The geometry below is hardware-confirmed. **For a 4-byte element the weight K-group is 16,
not 32.** The N-group stays 16, but a 4-byte input halves the K-group:

| param | value | note |
|---|---|---|
| precision CNA in/proc, CORE proc | 7 | tf32 (CNA/CORE only) |
| precision DPU in/proc/out | 5 | fp32. The DPU enum has no tf32, so tf32 uses the fp32 accumulator |
| feature cube C2 | 4 | 16-byte CBUF atom / 4 (confirmed) |
| weight tile | (N/16, K/16, 16, 16) | N-group 16 (== fp16), K-group 16 (halved from fp16's 32). Still a 1024-byte tile (16·16·4) |
| data_entries | K/16 | = number of K-groups (cf. fp16 K/32 at KG=32) |
| output | fp32 cube C2=4, size_e=3, surf×4 | the proven fp16/bf16 fp32-out writer |

`gen_matmul_tf32` (precision per-stage, element 4 B, data_entries K/16), `weight_tf32`
(N/16,K/16,16,16) and the standalone gate `matmul_tf32_rocket` all pass. The gate takes
raw fp32 in, characterizes precision against two references, and uses structured operand
patterns. The single-task CBUF limit is M·K·4 <= 11 banks (360448 B) and K <= 8192,
because a 4-byte element doubles the feature bytes. Big shapes tile.

### Tiled path

The tiled path is validated on hardware [HW sweep 2026-06-18]. `rocket_matmul_tf32` and
`rocket_matmul_plan_tf32` (`rocket_matmul.c`) are a clone of the bf16 tiled path with the
4-byte geometry:

- `float` slots K-aligned to 16
- Raw fp32 scatter with no truncation, because the hardware rounds the mantissa
- Banks ×4
- Kt <= 8192
- The sample-verified test `matmul_tf32_tiled_rocket.c`

A host diff showed the index helpers (`feat_idx_tf32` C2=4, `wt_idx_tf32`
(N/16,K/16,16,16)) bit-exact against `weight_tf32` and `feature_data(C2=4)` before the
hardware run. All shapes pass at norm_err ~1e-7, including `4 48 64` (K=48 =
%16-not-%32). The single-task gate could not reach that case: its `main()` required K%32,
so K∈{32,64,128} are all %32. The pass confirms that the K-group is 16 on hardware and
that the K%16 plan alignment is correct (no K%32 fallback). Big-Gemma `512×3840×4096`
runs at ~37 GOP/s.

### Stage assignment and the weight K-group trap

tf32=7 is a code for the CNA/CORE stages only. With all stages set to 7 the DPU writes
nothing (its out enum is 0..6). So DPU in/proc/out must be fp32 (5), and only CNA/CORE
carry 7 (CORE=5 -> 1e25 garbage). Structured operand patterns localize a wrong weight tile
(`ROCKET_TF32_PAT`: ones, k-ramp, n-col, m-row, K-impulse).

The trap is the weight K-group. At a single-K-group shape (K=32) the weight index is
row-major for any N-group and K-group, so K=32 cannot test the weight layout. A half-rate
`(N/8, K/32, 8, 32)` guess shows only N/2 distinct output channels and a K-misalignment
there, insensitive to every host knob. That looks like a structural half-rate hardware
limit, and it matches the "256×3" rate. The cause is the wrong K-group, which splits the
contraction across two output lanes.

Testing at K=64/128 distinguishes the two and confirms `(N/16, K/16, 16, 16)`. **Never
reverse-engineer a weight tile at a single-K-group shape. Test at K >= 2x the candidate
K-group.**

## Precision as a descriptor field

Precision is a field, not a datapath. The emit layer (`gen_matmul_task`) is
datatype-agnostic: it writes proc/in/out precision, data_sign, and the cvt_* fields
straight from descriptor structs. That is why int8 and int4 were "weeks, not a driver
rewrite". The datatype lives only in the descriptor setup (which precision value, which
element bytes, which tile layout, which output cube). Adding a datatype takes a new
precision value, the sub-byte math, and the nibble and tile layout. It reuses the same
register path.

The matmul output type per input datatype is established on hardware:
int8×int8->int32, int4×int4->int16, fp16×fp16->fp32. Even the int8 matmul outputs raw
int32. The RK3588 matmul entries do no on-device integer requant. The scales and the
dequantization are a host-side concern there. The RK3576's per-column int8 matmul requantizes
on chip ([../chips/rk3576.md](../chips/rk3576.md)).
