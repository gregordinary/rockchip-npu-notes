# RMSNorm on the NPU (+ the per-row broadcast scale)

RMSNorm computes `out[m][h] = x[m][h] / sqrt(mean_h(x[m][h]²) + eps) · weight[h]`.
`librocketnpu` implements it as `rocket_rmsnorm_fp16` plus the reusable
`rocket_scale_rows_fp16` (`src/rocket_norm.c`, `include/rocket_norm.h`). The hardware gate
is `tests/rmsnorm_rocket.c` (CTest `rmsnorm_rocket`). It builds on the
[feature-axis reduce](feature-reduce.md).

## The cost model

RMSNorm is a memory-bound elementwise+reduce. Standalone on the NPU it is submit-bound: the
flat `ew_mul` tiles 32,640 elements (1020 rows of 32) per submit, so a `[512,3840]` square is
~60 submits. For an isolated norm the host A76's single memory pass wins.

The on-NPU value is compositional. When the norm sits between two NPU matmuls
(FFN/attention), running it on-device keeps the activation in the NC1HWC2 cube. That avoids
the de-tile->host->re-pack layout round-trip that dominates the not-mac-bound budget. These
primitives therefore exist to be fused into a resident block, not to beat the host
standalone. Quantization for prefill speed reaches the same conclusion: DMA-bound, no win.

## The work split: O(M·H) on the NPU, the O(M) tail on the host

| step | where | why |
|------|-------|-----|
| `sq = x ⊙ x` | NPU (DPU ew_mul) | O(M·H) |
| `ms[m] = mean_h sq` | NPU ([feature reduce](feature-reduce.md), fp32 accum) | O(M·H) contraction |
| `r[m] = 1/sqrt(ms[m]+eps)` | host, fp32 | O(M), M tiny scalars |
| `out = x ⊙ (r ⊗ weight)` | NPU (ew_mul) | O(M·H) |

The rsqrt stays on the host, not on the DPU LUT. The rsqrt is over the M per-row scalars
only (already on the host as fp32 after the reduce read-back). Host `1/sqrtf` is exact and
free.

Sending the scalars back to the NPU for the DPU rsqrt LUT would add a round-trip for M tiny
values. It would also hit the rsqrt-LUT domain problem. `ms` spans many decades across rows
and layers, and a uniform-grid LUT cannot cover that range at good accuracy. The DPU rsqrt
LUT is for large-tensor rsqrt, not this M-vector, so the per-row rsqrt does not go on the
LUT at all.

## The fp16-square overflow and the power-of-2 prescale

The reduce consumes an fp16 square cube, but `x² > 65504` (fp16 max) once `|x| > 256`, and
transformer residual streams have outlier channels well past that. The guard is a
power-of-two prescale:

1. Scan `amax = max|x|` on the host (x is already host-resident).
2. Pick `k = max(0, ceil(log2(amax/223)))`.
3. Prescale `xs = x · 2⁻ᵏ`, so `(amax·2⁻ᵏ)² < ~50000` stays in fp16 range. A power of two
   is exact, with no rounding.
4. Square `xs` and reduce to `ms_scaled`.
5. Recover the true mean-square on the host as `ms = ms_scaled · 4ᵏ` (also exact).

In the common `|x| <= 223` case, `k=0` and there is no copy: `x` is squared directly. The
prescale is validated on hardware at amp=1000 (`|x|->1000`, `x²->1e6`): bit-accurate vs the
fp64 oracle.

## Per-row broadcast scale (`rocket_scale_rows_fp16`)

The entry computes `out[m][n] = in[m][n] · r[m]`, a per-row fp32 scalar broadcast over the
columns. It materializes the per-row scalar across the columns (a pure host fill, no
arithmetic) and reuses the DPU ew_mul, so it inherits that path bit-for-bit.

The in-block RMSNorm contracts to this form. The per-column weight folds into the next
matmul's weight, `W'[n,h] = weight[h]·W[n,h]` (static, once at load). The per-row 1/rms
folds here as a post-matmul per-row scale. The standalone normed tensor
(`rocket_rmsnorm_fp16`, with `r ⊗ weight` pre-combined into one ew_mul) is therefore
gate-grade, while the in-model path is just a weight-rescale + a row-scale.

The optimization target is to fold the per-row scale further into the matmul's activation
pack. The scatter already touches every element, so the fold removes even the
materialize+ew_mul.

## HW result (gate `rmsnorm_rocket`, 600 MHz, kernel 7.1.0-1)

Against the fp64 oracle, `scale_rows` reads max_rel ~1e-3 (fp16 ulp). RMSNorm reads
max_rel <= 3.5e-3 (the benign per-row fp16 square-rounding) across:

- The M-tile boundary (M=256)
- Gemma hidden (H=3840)
- Small-M / H%32≠0
- The amp=1000 overflow-prescale case

All cases read `bad=0`.
