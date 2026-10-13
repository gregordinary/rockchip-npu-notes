# The `size_e` quirk of the int8 and int4 outputs (DPU write-out)

Two values set the DPU's output-surface stride: the `size_e` field,
`DPU_BS_OW_CFG.SIZE_E_{0,1,2}` (`0x4050`), and a surface multiplier. The multiplier scales the
surface add, `DPU_SURFACE_ADD` (`0x40C0`) [source-confirmed: Mesa `registers.xml`]. For a
floating-point output of a direct convolution or a matmul on a floating-point input, `size_e`
takes its natural value, the output's `bytes - 1`. The int8 matmul's int32 output and the int4
matmul's int16 output instead take `size_e = 7`, whatever the byte width. Each strides as if
its elements were 8 bytes (Erratum E2). A depthwise convolution does not follow the natural
value either: its fp16 output takes `size_e = 3`.

## Programming requirements

Software must program `size_e` by the input type and the convolution mode [HW sweep]:

- **int8 input (int32 output):** `size_e = 7`, with the surface add × 8.
- **int4 input (int16 output):** `size_e = 7`.
- **int16 input (int32 output):** the natural `size_e = 3`, with the surface add × 4.
- **Floating-point input and output, direct convolution or matmul:** the natural value. An
  fp16 output (2 bytes) takes `size_e = 1`, and an fp32 output (4 bytes) takes `size_e = 3`.
- **fp16 depthwise convolution, fp16 output:** `size_e = 3`, with the surface add × 4
  ([../depthwise-conv.md](../depthwise-conv.md) §"Registers").

The fp32 and fp16 outputs of the int8 datapath do not take the natural values either. Their
values are in [out-cvt-converter.md](out-cvt-converter.md) §"Output datatype and cube on the
int8 (int32-acc) datapath".

| matmul | output | true bytes | natural `size_e` | actual `size_e` |
|---|---|---:|---:|---:|
| int8×int8 | int32 | 4 | 3 | **7** |
| int4×int4 | int16 | 2 | 1 | **7** |
| int16×int16 | int32 | 4 | 3 | 3 *(natural)* |
| fp16×fp16 | fp32 | 4 | 3 | 3 *(natural)* |

The int16 path is bit-exact at N from 48 to 256 with `size_e` 3 [HW sweep].

> [!CAUTION]
> An int16 task programmed with the int8 geometry, `size_e` 7 and the surface add × 8, never
> completes. It writes row 0's first sixteen channels. The kernel's watchdog retires it
> ([output-transpose-int16.md](output-transpose-int16.md)).

> [!NOTE]
> A wrong `size_e` has no visible effect on an output that fits in one surface. A test shape
> with one surface passes whatever the stride, and a wrong `size_e` shows only once N exceeds
> one surface. Test at an N that spans more than one surface.

## Software support

The generators program the surface multiplier as 8 (`SURF_MULT=8`) for both the int8 and the
int4 paths.

## Known issues

### Erratum E2: most of an int8 or int4 output not written

- **Description:** The int8 and int4 datapaths write their output strided as if each element
  were 8 bytes, even though int32 is 4 bytes and int16 is 2. The fp16 rule predicts
  `size_e` 3 with × 4 for a 4-byte int32 output, and that value is wrong here. Among the
  raw-accumulator outputs, the split follows the input width, one byte or less against two,
  and not the output type [hypothesis]. Input width alone does not decide it: the int8
  datapath's requantized int8 output takes `size_e` 1, and its fp16 output 3
  ([out-cvt-converter.md](out-cvt-converter.md)). A working model: the int8 and int4 outputs are a reinterpret of the same
  physical convolution write. The DPU casts its wide CACC accumulator and writes it with a
  fixed 8-byte-per-element stride, `size_e=7`, whatever the output type's byte width
  [hypothesis].
- **Conditions:** RK3588 under the `rocket` driver. An int8×int8 matmul (int32 output) or an
  int4×int4 matmul (int16 output), programmed with the natural `size_e` of its output. N
  spans more than one surface.
- **Symptom:** Most of the output is never written [HW sweep]. At `size_e` 3 with × 4, an
  int8 output's surface stride halves, and every output column past the first surface is
  left unwritten. An int4 output at `size_e=1` has only its first 16 N-columns written.
- **Workaround:** Software must program `size_e=7` for the int32 output of an int8 input and
  for the int16 output of an int4 input.
- **Status:** Open. No source documents it.

## Evidence

### Programming requirements

The int16 measurement is `tests/int16_native_probe`.

### Known issues

Both integer widths confirm `size_e=7` independently [HW sweep]:

- **int8 (int32 output)**: `ROCKET_INT8_SIZE_E=3 ROCKET_INT8_SURF_MULT=4` leaves every
  output column past the first surface as the `0xAA` sentinel. The surface stride halves,
  so most of the output is never written. Only `size_e=7`/`surf×8` writes the full output.
- **int4 (int16 output)**: at precision=6, `size_e=1` writes only the first 16 N-columns
  (17-64 stay at the `0xAAAA` sentinel). `size_e=3` writes 32 columns. `size_e=7` writes
  all 64 columns, bit-exact. `SURF_MULT` is irrelevant once `size_e=7` (4, 8 and 16 pass at
  M=4). The int16 (int4-path) output therefore strides with the same `size_e=7` quirk as the
  int32 (int8-path) output.

The input-width split is read off which paths take 7. No mechanism is isolated.
