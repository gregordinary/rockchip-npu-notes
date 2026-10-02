# The `size_e` quirk of the int8 and int4 outputs (DPU write-out)

A `size_e` field and a surface multiplier control the DPU's output-surface stride. For
floating-point outputs the field takes its natural value: the stride encodes `bytes - 1`:

- fp16 output (2 bytes) -> `size_e = 1`
- fp32 output (4 bytes) -> `size_e = 3`

**The int8 and int4 conv outputs do not follow that rule.** Both stride as if
`size_e = 7`, whatever the byte width, while int16's int32 output takes the natural value:

| matmul | output | true bytes | natural `size_e` | actual `size_e` |
|---|---|---:|---:|---:|
| int8×int8 | int32 | 4 | 3 | **7** |
| int4×int4 | int16 | 2 | 1 | **7** |
| int16×int16 | int32 | 4 | 3 | 3 *(natural)* |
| fp16×fp16 | fp32 | 4 | 3 | 3 *(natural)* |

So the int8 and int4 datapaths write their output strided as if each element were 8 bytes,
even though int32 is 4 and int16 is 2. The accompanying surface multiplier is 8
(`SURF_MULT=8`) for both.

int16×int16 does not share the quirk. Its int32 output takes the natural value, as fp16's
fp32 does: `size_e` 3 with the surface add × 4. It is bit-exact at N from 48 to 256
[HW sweep, `tests/int16_native_probe`]. With `size_e` 7 and × 8, an int16 task writes row
0's first sixteen channels and never completes. The kernel's watchdog retires it
([output-transpose-int16.md](output-transpose-int16.md)). So the quirk belongs to the int8
and int4 input paths, and is not a rule for every integer output.

## Sweep evidence

The integer `size_e=7` looks like a bug against the fp16 rule, which predicts 3/×4 for a
4-byte int32. `size_e=7` is correct. **Do not "fix" it to the natural byte width.** Both
integer widths confirm this independently [HW sweep]:

- **int8 (int32 output)**: `ROCKET_INT8_SIZE_E=3 ROCKET_INT8_SURF_MULT=4` leaves every
  output column past the first surface as the `0xAA` sentinel. The surface stride halves,
  so most of the output is never written. Only `size_e=7`/`surf×8` writes the full output.
- **int4 (int16 output)**: at precision=6, `size_e=1` writes only the first 16 N-columns
  (17-64 stay at the `0xAAAA` sentinel). `size_e=3` writes 32 columns. `size_e=7` writes
  all 64 columns, bit-exact. `SURF_MULT` is irrelevant once `size_e=7`. So the int16
  (int4-path) output strides with the same `size_e=7` quirk as the int32 (int8-path)
  output.

## Mental model

The int8 and int4 outputs are a reinterpret of the same physical conv write. The DPU
casts its wide CACC accumulator and writes it with a fixed surface geometry, `size_e=7`,
the 8-byte-per-element stride. That holds whatever number of those bytes the output type
occupies. The float outputs and int16's int32 use the natural byte-width stride. The split
follows the input width (one byte or less against two) and not the output type
[hypothesis: read off which paths take 7, no mechanism isolated]. No source documents it.

**Trap:** a wrong `size_e` is invisible unless the output is large enough to span multiple
surfaces. A tiny shape with one surface "passes" whatever the stride, so a wrong `size_e`
shows only once N exceeds one surface. Always test N past one surface.
