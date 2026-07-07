# The `size_e` integer-output quirk (DPU write-out)

The DPU's output-surface stride is controlled by a `size_e` field (and a surface
multiplier). For **floating-point** outputs it behaves naturally — the stride encodes
`bytes - 1`:

- fp16 output (2 bytes) → `size_e = 1`
- fp32 output (4 bytes) → `size_e = 3`

For **integer-conv outputs it does not.** Both integer output widths stride as if
`size_e = 7`, regardless of the actual byte width:

| matmul | output | true bytes | natural `size_e` | **actual `size_e`** |
|---|---|---:|---:|---:|
| int8×int8 | int32 | 4 | 3 | **7** |
| int4×int4 | int16 | 2 | 1 | **7** |
| fp16×fp16 | fp32 | 4 | 3 | 3 *(natural)* |

So the integer datapath writes its output strided as if each element were 8 bytes,
even though int32 is 4 and int16 is 2. The accompanying surface multiplier is 8
(`SURF_MULT=8`) for both.

**int16×int16:** the native int16 conv carries the same `size_e=7`/`SURF_MULT=8`
integer-output geometry, but for int16 `size_e` is genuinely N-derived (it counts the 8-channel groups in a row per output line, minus 1 → `N/8 - 1`, which is 7 for N=64).
It is moot in practice: int16 has no usable native int32 output at all (the writer
only emits one int32 tile or a transposed 8/16-bit buffer — see
[output-transpose-int16.md](output-transpose-int16.md)), so `size_e` never gates a
working int16 int32 output the way it did for int8/int4.

## The sweep evidence

The integer `size_e=7` looks like a bug against the fp16 rule (which predicts 3/×4 for a
4-byte int32), but `size_e=7` is correct — do **not** "fix" it to the natural byte width.
Both integer widths confirm this independently [HW sweep]:

**int8 (int32 output):** `ROCKET_INT8_SIZE_E=3 ROCKET_INT8_SURF_MULT=4` leaves every output
column past the first surface as the `0xAA` sentinel — the surface stride halves, so most of
the output is never written. Only `size_e=7`/`surf×8` writes the full output.

**int4 (int16 output):** at precision=6, `size_e=1` writes only the first 16 N-columns
(17–64 stay at the `0xAAAA` sentinel); `size_e=3` → 32 cols; **`size_e=7` → all 64 cols,
bit-exact.** `SURF_MULT` is irrelevant once `size_e=7`. So the int16 (int4-path) output
strides with the *same* `size_e=7` quirk as the int32 (int8-path) output.

## Mental model

The integer outputs are a **reinterpret of the same physical conv write** — the DPU
casts its wide CACC accumulator and writes it with a fixed integer-output surface
geometry (`size_e=7`, the 8-byte-per-element stride), independent of how many of those
bytes the chosen output type actually occupies. The float outputs use the natural
byte-width stride. There is no documented reason; it is simply what the hardware does,
confirmed identically for int32 and int16.

**Trap:** this is invisible unless the output is large enough to span multiple surfaces.
A tiny shape with one surface "passes" regardless of stride, so a wrong `size_e` only
shows once N exceeds one surface — always test N past one surface.
