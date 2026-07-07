# Feature-axis reduce (sum / mean) + cumsum (prefix sum) — the ones-vector / triangular matmul

Reduce over the **hidden/feature axis, per row**: `out[m] = sum_h x[m][h]` (or the mean). This
is the contraction every transformer normalization needs — RMSNorm/LayerNorm reduce over the
hidden axis, softmax's denominator reduces over the sequence axis — and the one the
[PPU spatial reduce](ppu-reduce-mean.md) **cannot** supply. Implemented:
`rocket_reduce_feature_fp16` (`src/rocket_reduce.c`, `include/rocket_reduce.h`); HW gate
`tests/reduce_feature_rocket.c` (CTest `reduce_feature_rocket`).

## NPU FACT — the PPU cannot reduce the feature (channel) axis

The PPU is a **pooling** engine ([ppu-pooling-not-detile.md](../perf/ppu-pooling-not-detile.md)):
it reduces the spatial axes `[H,W]` **within** a channel and writes the same channel count back
(NC1HWC2 → NC1HWC2, C1 unchanged). There is **no register that contracts across C** — pooling
is per-channel by construction. So a feature-axis reduce is **not** a pool; the PPU reduce
(`rocket_global_avgpool_fp16`) and this feature reduce are orthogonal, covering the two different
axes a transformer contracts.

### The layout-trick alternative, and why it loses

You *could* lay the feature vector on the spatial axis — cube `[C=M, H=1, W=Hfeat]` — and run the
existing PPU avg-pool over `W` to get a per-row mean. Rejected:
- the PPU path needs `Hfeat` **16-smooth** (prime factors ≤16); the ones-matmul accepts **any** H,
- `rocket_global_avgpool_plan` rejects `H=1` (`nh≠nw`); it would need generalizing,
- each pass divides by a **fp16 reciprocal** `fp16(65536/k)` (~1e-3 quant/pass), vs. the matmul's
  genuine fp32 accumulation — and the sum-of-squares variance term wants every bit.

## The mechanism — reduce over K is a matmul against a ones vector

The matmul computes `C[m,n] = sum_k A[m,k]·B[n,k]`. With `B = ones`, the K-sum **is** the reduce:

```
out[M,1] = x[M,H] · ones[1,H]^T          out[m] = sum_h x[m,h]·1 = sum_h x[m,h]
```

So **no new regcmd** — it reuses `rocket_matmul_fp16_f32out` (the fp32-output path) and
inherits its genuine fp32 K-accumulation (K-partials summed in fp32/fp64, not fp16-narrowed per
tile). Mechanics:
- **N padded to 16** (the matmul's N-group). The ones weight is `[16, Kpad]`; 16 identical output
  columns are computed and **column 0** is read back. The redundant 15 columns are a tiny fixed
  cost on the K-dominated shape (no extra DRAM weight traffic beyond the `16×Kpad` ones vector).
- **K (=H) zero-padded to %32, M to %4.** The ones weight is `1` over the real `H` columns and `0`
  over the K-pad, so padded columns contribute 0; padded rows produce an ignored sum. When the
  input already meets `H%32==0 && M%4==0` (the common LLM case, e.g. Gemma H=3840) the input buffer
  is passed **directly** as A — no host staging copy.
- **fp32 output.** A bare sum stays in fp16 range, but the main consumer squares first
  (sum-of-squares for variance) where the running sum easily exceeds fp16 — so the reduce returns
  `float[M]`. The fp16 **input** elements must still be finite; square/scale upstream (see RMSNorm).

## HW result (gate `reduce_feature_rocket`, 600 MHz, kernel 7.1.0-1)

Essentially **bit-exact** vs the fp64 oracle: `max_rel ≤ 1.4e-7` (fp32 noise floor), mostly
`max_abs = 0`, across the M-tile boundary (M=256/260), realistic Gemma widths (H=3840/2048), the
K%32≠0 / M%4≠0 padding cases (H=48→64, M=5→8), and large magnitudes (amp=40, sum ~4400) — for both
sum and mean.

## Follow-ons

- Folds away in the FFN/QKV: RMSNorm's per-column weight `w[h]` folds into the **next** matmul's
  weight (static, once); the per-row `1/rms` folds as a post-matmul per-row scale — so the
  *standalone* reduce here is the gate-grade primitive, and the in-block version contracts to a
  weight-rescale + output-scale (no separate reduce op). See RMSNorm.
- The 16-wide N can carry up to 16 *different weighted* reductions of the **same** A (different
  weight columns: `sum(x)`, `sum(w⊙x)`, …) instead of 16 copies of the ones column. It does not give
  `sum(x²)` — that is a reduction of a *different operand* (`x²`), so LayerNorm's mean+variance share
  one job by **stacking rows** (A = `[x ; x⊙x]`, 2M rows, ones weight → `sum(x)` then `sum(x²)`).

## Cumsum (prefix sum) — the same matmul, ones-COLUMN widened to a triangular MATRIX

A **cumulative sum is a matmul by a triangular ones matrix.** The feature reduce above is the
N=1 special case: a single all-ones *column* sums all of K. Widen that one column to the full
triangle and **every prefix appears as its own output column** — one matmul produces the whole
scan. `rocket_cumsum_fp16` (`src/rocket_reduce.c`), gate `tests/cumsum_rocket.c`.

```
out[M,N] = in[M,N] · L^T          L[n][k] = 1  iff input column k is in prefix n
```

In the matmul's `C[m,n] = sum_k A[m,k]·B[n,k]` convention, B (=L) is the `[N,N]` weight and the
prefix-membership rule picks the triangle (and so the variant):

| variant            | `L[n][k] = 1` when | triangle              |
|--------------------|--------------------|-----------------------|
| inclusive forward  | `k <= n`           | lower, incl. diagonal |
| exclusive forward  | `k <  n`           | strictly lower        |
| reverse, inclusive | `k >= n`           | upper, incl. diagonal |
| reverse, exclusive | `k >  n`           | strictly upper        |

So **no new regcmd** — it reuses `rocket_matmul_fp16_f32out` exactly as the feature reduce does,
inheriting the genuine fp32 K-accumulation (a long prefix sums many terms — the fp32 accumulator
avoids the per-tile fp16 narrowing the plain fp16 path would apply). Mechanics mirror the reduce:
**K(=N) padded to %32, the output-column count N to %16, M to %4**; the triangular weight is set
over the real `[N,N]` block and the pad rows/cols are 0 (contribute nothing); the fp32 result is
narrowed to fp16 on read-back. The `[N,N]` weight is `N²` fp16 (N=1500 → 4.5 MB) — the matmul
tiles it like any other weight.

### HW result (gate `cumsum_rocket`, 600 MHz, kernel 7.1.0-1)

**Bit-exact** vs the fp64 prefix-sum oracle — `max_abs = 0` on every shape and all four variants,
across the M-tile boundary (M=256/260), N%32≠0 (N=100/48), realistic widths (768), and a long
**T=1500** prefix. The expected accuracy was fp16-rounding (fp32-accum then a single fp16 narrow),
but the sums of fp16 terms here are exactly representable in the fp32 accumulator, so fp32 == fp64
and the narrow matches the oracle's. **NPU FACT: no long-prefix fp16 degradation appears** at these
magnitudes (worst prefix ~211); if a much larger/longer prefix ever overflowed fp16, the RMSNorm
power-of-2 **prescale** trick (square/scale into range, recover after) is the lever — same as the
reduce's sum-of-squares path. The gate's own self-check re-derives every prefix with an independent
O(N²) fp64 recompute (catches an off-by-one prefix boundary, which would differ by a whole element).

### Why this matters / consumers

Cumsum is the running-total scan in beam search, CTC alignment, autoregressive/causal masking,
segment offsets, and any "prefix" index math — TFLite/ONNX `CumSum`. It is the second op (after the
[cross-entropy](whisper-encoder.md) logsumexp) shown to fall out of the **reduce-as-matmul** family
for free: the contraction axis is the only axis the matmul can sum over, so anything expressible as
"a (possibly structured) linear combination along the last axis" — full reduce, weighted reduce,
prefix scan — is one ones-/triangular-weight matmul with no new regcmd.
