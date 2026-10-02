# Datatype capability matrix

The RK3588 NPU supports a full datatype menu, selected by a 3-bit precision field set
independently for the input, the MAC stage, and the output. Every datatype below has a
working, hardware-validated matmul. This table is the canonical summary. The per-column
detail is in the linked encoding and performance notes.

| datatype | precision field | input width | output (accumulate) | native matmul | on-NPU K-accumulation | MAC rate † | primary use |
|---|---:|---|---|---|---|---:|---|
| int4 | 6 | 4-bit | int16 | yes | no | 4x | smallest weights (~¼ of fp16). W4A4 + Hadamard |
| int8 | 0 | 8-bit | int32 | yes | no | 2x | smaller weights. W8A8 + Hadamard |
| int16 | 1 | 16-bit | int32 ‡ | yes | no | (1x) | exact integer reference |
| fp16 | 2 | 16-bit | fp32 | yes | yes | 1x | default. Best throughput and coherence |
| bf16 | 3 | 16-bit | fp32 | yes | no | 1x | fp32 range at fp16 cost. Drops activation scaling |
| tf32 | 7 § | 32-bit | fp32 | yes | no | ½x | fp32 range with 10-bit precision. Half-rate |

**† The MAC rate does not translate to wall-clock speed.** fp16 is the baseline. int8 and
int4 nominally multiply 2x and 4x faster in the array, and tf32 is half-rate. At the
current operating point (resident weights, multicore, 600 MHz) the resident matmul
measures ~460 GOP/s across fp16, int8, and int4 alike. The path is bound by DMA and
per-job dispatch, not by the MAC array. So the 2x and 4x integer advantages do not express
as throughput.

Quantization's payoff on this hardware is memory footprint, not prefill speed: a smaller
model fits in RAM and in the per-fd address window. See
[perf/not-mac-bound.md](perf/not-mac-bound.md).

**‡ int16's int32 output saturates.** When its operand ranges keep the dot product inside
int32, a task is exact. An int64-exact result over full-range operands takes four int8
matmuls instead. See
[encodings/output-transpose-int16.md](encodings/output-transpose-int16.md).

**§** tf32 uses precision 7 at the input stages of the pipeline (CNA/CORE). The output
stage has no tf32 code, so it uses the fp32 accumulator (precision 5).

## Output containers

int32 (field 4) and fp32 (field 5) are output formats only, not matmul input datatypes.
The native input->output pairings are:

- `int4->int16`
- `int8->int32`
- `int16->int32`
- `fp16/bf16/tf32->fp32`

The matmul entries leave integer requant and dequant to the host. The DPU's output converter can
requantize on chip, and the int8 convolution entries use it
([encodings/out-cvt-converter.md](encodings/out-cvt-converter.md)).

## Scope of the menu

The datatype menu is the matmul capability, not a whole-model quant recipe. int4, int16,
bf16 and tf32 are native matmul types, not necessarily graph-quantization options. A
quantized LLM still runs per-tensor activation scales, and their interaction with
activation outliers is the root cause of LLM gibberish. That is why the W8A8/W4A4 path
adds a Hadamard rotation, a stronger mitigation than plain range-clipping or a per-layer
fp16 hybrid fallback. [TUNING.md](TUNING.md) has the flags.

## Per-datatype notes

- **fp16.** This is the workhorse, and the only datatype with on-NPU K-accumulation. The
  DPU eltwise unit adds K-tile partials in fp16, so readback is `∝ M·N` instead of
  `∝ M·N·nKt`. It has the cleanest numerical behavior, and is the default for LLM prefill
  and the Whisper encoder.
- **bf16.** It has the same MAC rate and operand size as fp16 but fp32 dynamic range, so a
  caller can drop per-row activation scaling entirely. Its fp32 output reuses fp16's
  proven output writer. It is token-identical to fp16 on tested models.
- **int8 and int4.** They give smaller weights, for fitting larger models in memory. Both
  need a Hadamard rotation to tame activation outliers (without it, quantized LLM output
  is incoherent). int4's denser packing can reach single-pass K (no K-tile readback).
  Neither is faster than fp16 at the current operating point (see the † note).
- **int16.** It is present for completeness and as an exact integer reference. Its native
  int32 output saturates, so the exact path decomposes each operand into bytes and runs int8
  matmuls ([encodings/output-transpose-int16.md](encodings/output-transpose-int16.md)).
- **tf32.** It has a genuine 10-bit mantissa and fp32 range, and is the only 4-byte input
  path. It runs at half the MAC rate of fp16 and bf16, which already cover its use cases,
  so it is the lowest-value datatype in practice.

## Related notes

- Precision-field encodings and how each was established: [encodings/precision-field.md](encodings/precision-field.md)
- Per-datatype tile and cube layouts: [encodings/tile-layouts.md](encodings/tile-layouts.md)
- The int8 and int4 output stride quirk: [encodings/size-e-quirk.md](encodings/size-e-quirk.md)
- What the eltwise unit can accumulate: [encodings/k-accumulation.md](encodings/k-accumulation.md)
- Why MAC advantages do not become speed: [perf/not-mac-bound.md](perf/not-mac-bound.md)
