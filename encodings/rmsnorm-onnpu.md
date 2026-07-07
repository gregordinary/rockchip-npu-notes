# RMSNorm on the NPU (+ the per-row broadcast scale)

`out[m][h] = x[m][h] / sqrt(mean_h(x[m][h]²) + eps) · weight[h]`. Implemented
`rocket_rmsnorm_fp16` + the reusable `rocket_scale_rows_fp16` (`src/rocket_norm.c`,
`include/rocket_norm.h`); HW gate `tests/rmsnorm_rocket.c` (CTest `rmsnorm_rocket`). Built on
the [feature-axis reduce](feature-reduce.md).

## The cost model — submit-bound standalone, compositional on-NPU value

RMSNorm is a memory-bound elementwise+reduce. Standalone on the NPU it is **submit-bound**: the
flat `ew_mul` tiles ~1020 rows/submit, so a `[512,3840]` square is ~60 submits — for an *isolated*
norm the host A76's single memory pass wins. The on-NPU value is **compositional**: when the norm
sits between two NPU matmuls (FFN/attention), running it on-device keeps the activation in the
NC1HWC2 cube and avoids the de-tile→host→re-pack layout round-trip that dominates the not-mac-bound
budget. So these primitives exist to be **fused into a resident block**, not to beat
the host standalone. (Same conclusion as quantization for prefill speed: DMA-bound, no win.)

## The work split — O(M·H) on the NPU, the O(M) tail exact on the host

| step | where | why |
|------|-------|-----|
| `sq = x ⊙ x` | NPU (DPU ew_mul) | O(M·H) |
| `ms[m] = mean_h sq` | NPU ([feature reduce](feature-reduce.md), fp32 accum) | O(M·H) contraction |
| `r[m] = 1/sqrt(ms[m]+eps)` | **host**, fp32 | O(M) — M tiny scalars |
| `out = x ⊙ (r ⊗ weight)` | NPU (ew_mul) | O(M·H) |

**NPU FACT — the rsqrt stays on the host, not the DPU LUT.** The rsqrt is over the M per-row
scalars only (already on the host as fp32 after the reduce read-back). Sending them *back* to the
NPU for the DPU rsqrt LUT would (a) add a round-trip for M tiny values and (b) hit the **rsqrt-LUT
domain problem**: `ms` spans many decades across rows/layers, and a uniform-grid LUT can't cover
that range at good accuracy. Host `1/sqrtf` is exact and free. The DPU rsqrt LUT is for
**large-tensor** rsqrt, not this M-vector — so the per-row rsqrt does not go on the LUT at all.

## NPU FACT — fp16-square overflow needs a power-of-2 prescale

The reduce consumes a **fp16** square cube, but `x² > 65504` (fp16 max) once `|x| > 256` — and
transformer residual streams have outlier channels well past that. Guard: scan `amax = max|x|` on
the host (x is already host-resident), pick `k = max(0, ceil(log2(amax/223)))`, prescale
`xs = x · 2⁻ᵏ` (a power of two ⇒ **exact**, no rounding) so `(amax·2⁻ᵏ)² < ~50000` stays in fp16
range, square `xs`, reduce → `ms_scaled`, and recover the true mean-square on the host as
`ms = ms_scaled · 4ᵏ` (also exact). `k=0` for the common `|x| ≤ 223` case (no copy — `x` is squared
directly). **HW-validated at amp=1000** (`|x|→1000`, `x²→1e6`): bit-accurate vs the fp64 oracle.

## Per-row broadcast scale (`rocket_scale_rows_fp16`) — the FFN/attention post-scale

`out[m][n] = in[m][n] · r[m]` (a per-row fp32 scalar broadcast over the columns). Realized by
**materializing** the per-row scalar across the columns (a pure host fill, no arithmetic) and
reusing the DPU ew_mul, so it inherits that path bit-for-bit. This is the form the in-block
RMSNorm contracts to: the per-column **weight folds into the next matmul's weight** `W'[n,h] =
weight[h]·W[n,h]` (static, once at load), and the per-row **1/rms folds here** as a post-matmul
per-row scale — so the *standalone* normed tensor (`rocket_rmsnorm_fp16`, with `r ⊗ weight`
pre-combined into one ew_mul) is gate-grade, while the in-model path is just a weight-rescale + a
row-scale. **Optimization target:** the per-row scale folds further into the matmul's activation
**pack** (the scatter already touches every element), removing even the materialize+ew_mul.

## HW result (gate `rmsnorm_rocket`, 600 MHz, kernel 7.1.0-1)

vs the fp64 oracle: `scale_rows` max_rel ~1e-3 (fp16 ulp); RMSNorm max_rel ≤ 3.5e-3 (the benign
per-row fp16 square-rounding) across the M-tile boundary (M=256), Gemma hidden (H=3840), small-M /
H%32≠0, and the amp=1000 overflow-prescale case. All `bad=0`.
