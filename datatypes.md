# Datatype capability matrix

The RK3588 NPU supports a full datatype menu, selected by a 3-bit precision field set
independently for the input, the MAC stage, and the output. Every datatype below has a
working, hardware-validated matmul. This table is the canonical summary; the per-column
detail lives in the linked encoding and performance notes.

| dtype | precision field | input width | output (accumulate) | native matmul | on-NPU K-accum | MAC rate † | primary use |
|---|---:|---|---|---|---|---:|---|
| int4 | 6 | 4-bit | int16 | yes | no | 4× | smallest weights (~¼ of fp16); W4A4 + Hadamard |
| int8 | 0 | 8-bit | int32 | yes | no | 2× | smaller weights; W8A8 + Hadamard |
| int16 | 1 | 16-bit | — | via int8 byte-decomposition ‡ | no | (1×) | exact integer reference |
| fp16 | 2 | 16-bit | fp32 | yes | **yes** | 1× | default — best throughput and coherence |
| bf16 | 3 | 16-bit | fp32 | yes | no | 1× | fp32 range at fp16 cost; drops activation scaling |
| tf32 | 7 § | 32-bit | fp32 | yes | no | ½× | fp32 range with 10-bit precision; half-rate |

**† The MAC rate does not translate to wall-clock speed.** fp16 is the baseline;
int8 and int4 nominally multiply 2×/4× faster in the array, and tf32 is half-rate.
But at the current operating point — resident weights, multicore, 600 MHz — resident
matmul measures **~460 GOP/s across fp16, int8, and int4 alike**, because the path is
bound by DMA and per-job dispatch, not by the MAC array. The 2×/4× integer advantages
do not express as throughput. **Quantization's payoff on this hardware is memory
footprint, not prefill speed** (a smaller model fits in RAM and in the per-fd address
window; see [perf/not-mac-bound.md](perf/not-mac-bound.md)).

**‡** int16 computes correct products in the MAC array but has **no native matmul
output writer**, so full-precision int16 is realized as four int8 matmuls (bit-exact).
See [encodings/output-transpose-int16.md](encodings/output-transpose-int16.md).

**§** tf32 uses precision 7 at the front of the pipe (CNA/CORE); the output stage has
no tf32 code, so it rides the fp32 accumulator (precision 5).

**Output containers.** int32 (field 4) and fp32 (field 5) are output formats only, not
matmul input datatypes. The native input→output pairings are
`int4→int16`, `int8→int32`, `fp16/bf16/tf32→fp32`; integer requant and dequant are
host-side concerns, not done inside the matmul.

**The datatype menu is the *matmul* capability, not a whole-model quant recipe.** int4 /
int16 / bf16 / tf32 are native matmul types, not necessarily graph-quantization options — a
quantized LLM still runs per-tensor activation scales, and the interaction of those with
activation outliers is the LLM gibberish root-cause. That is why the W8A8/W4A4 path adds a
**Hadamard rotation** (a stronger mitigation than plain range-clipping or a per-layer fp16
hybrid fallback). See [encodings/k-accumulation.md](encodings/k-accumulation.md).

## Per-datatype notes

- **fp16** — the workhorse. The only datatype with on-NPU K-accumulation (the DPU
  eltwise unit adds K-tile partials in fp16, so readback is `∝ M·N` instead of
  `∝ M·N·nKt`). Cleanest numerical behaviour; the default for LLM prefill and the
  Whisper encoder.
- **bf16** — same MAC rate and operand size as fp16 but with fp32 dynamic range, so
  per-row activation scaling can be dropped entirely. Its fp32 output reuses fp16's
  proven output writer. Token-identical to fp16 on tested models.
- **int8 / int4** — smaller weights for fitting larger models in memory. Both need a
  Hadamard rotation to tame activation outliers (without it, quantized LLM output is
  incoherent). int4's denser packing can reach single-pass K (no K-tile readback).
  Neither is faster than fp16 at the current operating point — see the † note.
- **int16** — present for completeness and exact integer reference; realized by byte
  decomposition rather than a native output path.
- **tf32** — genuine 10-bit-mantissa, fp32-range, the only 4-byte input path. Half the
  MAC rate of fp16/bf16, which already cover its use cases, so it is the lowest-value
  rung in practice.

## Where the detail lives

- Precision-field encodings and how each was established: [encodings/precision-field.md](encodings/precision-field.md)
- Per-dtype tile/cube layouts: [encodings/tile-layouts.md](encodings/tile-layouts.md)
- The integer-output stride quirk: [encodings/size-e-quirk.md](encodings/size-e-quirk.md)
- What the eltwise unit can accumulate: [encodings/k-accumulation.md](encodings/k-accumulation.md)
- Why MAC advantages don't become speed: [perf/not-mac-bound.md](perf/not-mac-bound.md)
