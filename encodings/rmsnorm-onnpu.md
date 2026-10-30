# RMSNorm on the NPU (+ the per-row broadcast scale)

RMSNorm computes `out[m][h] = x[m][h] / sqrt(mean_h(x[m][h]²) + eps) · weight[h]`. On the NPU
it is built from the DPU's EW multiply and the [feature-axis reduce](feature-reduce.md), with
the per-row rsqrt on the host. Its final step, a per-row broadcast scale, is a reusable
operation of its own.

## Constraints

### The fp16-square overflow and the power-of-2 prescale

The reduce consumes an fp16 square cube, but `x² > 65504` (fp16 max) once `|x| > 256`, and
transformer residual streams have outlier channels well past that. The guard is a
power-of-two prescale:

1. Scan `amax = max|x|` on the host (x is already host-resident).
2. Pick `k = max(0, ceil(log2(amax/223)))`.
3. Prescale `xs = x · 2⁻ᵏ`, so `(amax·2⁻ᵏ)² < ~50000` stays in fp16 range. A power of two
   is exact, with no rounding.
4. Square `xs` and reduce to `ms_scaled`.
5. Recover the true mean-square on the host as `ms = ms_scaled · 4ᵏ` (also exact).

In the common `|x| <= 223` case, `k=0` and there is no copy: `x` is squared directly. At
amp=1000 (`|x|->1000`, `x²->1e6`), the prescaled result is bit-accurate against the fp64
oracle [HW sweep].

### The per-row rsqrt

The rsqrt stays on the host, not on the DPU LUT. The rsqrt is over the M per-row scalars
only, which are already on the host as fp32 after the reduce read-back. Host `1/sqrtf` is
exact and free.

Sending the scalars back to the NPU for the DPU rsqrt LUT would add a round trip for M tiny
values. It would also meet the rsqrt LUT's domain limit. `ms` spans many decades across rows
and layers, and a uniform-grid LUT cannot cover that range at good accuracy. The DPU rsqrt
LUT is for a large-tensor rsqrt, not for this M-vector.

## Lowering

The O(M·H) work runs on the NPU, and the O(M) tail on the host:

| step | where | why |
|------|-------|-----|
| `sq = x ⊙ x` | NPU (DPU ew_mul) | O(M·H) |
| `ms[m] = mean_h sq` | NPU ([feature reduce](feature-reduce.md), fp32 accum) | O(M·H) contraction |
| `r[m] = 1/sqrt(ms[m]+eps)` | host, fp32 | O(M), M tiny scalars |
| `out = x ⊙ (r ⊗ weight)` | NPU (ew_mul) | O(M·H) |

`librocketnpu` implements this as `rocket_rmsnorm_fp16`, plus the reusable
`rocket_scale_rows_fp16`, in `src/rocket_norm.c`, declared in `include/rocket_norm.h`.

### Per-row broadcast scale (`rocket_scale_rows_fp16`)

The entry computes `out[m][n] = in[m][n] · r[m]`, a per-row fp32 scalar broadcast over the
columns. It materializes the per-row scalar across the columns (a pure host fill, no
arithmetic) and reuses the DPU ew_mul, so it inherits that path bit-for-bit.

The in-block RMSNorm contracts to this form. The per-column weight folds into the next
matmul's weight, `W'[n,h] = weight[h]·W[n,h]` (static, once at load). The per-row 1/rms
folds here as a post-matmul per-row scale. In a model, the norm is therefore the first three
steps of §"Lowering", which compute `r[m]`. A weight rescale and a row scale replace the last
step. The standalone normed tensor (`rocket_rmsnorm_fp16`, with `r ⊗ weight`
pre-combined into one ew_mul) is the form the hardware test checks.

Software support: no `librocketnpu` entry implements the in-model fold, and no frontend runs
RMSNorm on the NPU. `rocket_rmsnorm_fp16` returns the normed tensor rather than `r[m]`, and no
entry folds `weight[h]` into a matmul's weight. ggml-rocket does not claim `GGML_OP_RMS_NORM`,
and neither ort-rocket nor tflite-rocket has an RMSNorm handler.

## Accuracy

**Conditions:** RK3588, NPU clock ~816 MHz, kernel 7.1.0-1, against the fp64 oracle.

`scale_rows` measures max_rel ~1e-3 (fp16 ulp) [HW sweep]. RMSNorm measures max_rel <= 3.5e-3
(the benign per-row fp16 square-rounding) [HW sweep] across:

- The M-tile boundary (M=256)
- Gemma hidden (H=3840)
- Small-M / H%32≠0
- The amp=1000 overflow-prescale case

## Performance

RMSNorm is a memory-bound elementwise op plus a reduce. Standalone on the NPU it is bound by
its submits. The flat `ew_mul` tiles 32,640 elements (1020 rows of 32) per submit, so a
`[512,3840]` square is ~60 submits. For an isolated norm, the host A76's single memory pass is
faster.

The NPU path's use is in composition. When the norm sits between two NPU matmuls (FFN or
attention), running it on the NPU keeps the activation in the NC1HWC2
[cube layout](../GLOSSARY.md). It avoids the de-tile->host->re-pack round trip, the dominant
host cost of an NPU matmul ([../perf/not-mac-bound.md](../perf/not-mac-bound.md)). These
primitives are built to be fused into a resident block, not to beat the host standalone.

## Evidence

### Accuracy

The hardware test is `tests/rmsnorm_rocket.c` (CTest `rmsnorm_rocket`). All of its cases
measured `bad=0`.
