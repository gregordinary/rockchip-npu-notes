# Feature-axis reduce and cumsum: the ones-vector and triangular matmul

This op reduces over the hidden (feature) axis, per row: `out[m] = sum_h x[m][h]` (or the
mean). It is the contraction every transformer normalization needs. RMSNorm and LayerNorm
reduce over the hidden axis, and softmax's denominator reduces over the sequence axis. The
[PPU spatial reduce](ppu-reduce-mean.md) cannot supply it. `librocketnpu` implements it as
`rocket_reduce_feature_fp16` (`src/rocket_reduce.c`, `include/rocket_reduce.h`). The
hardware gate is `tests/reduce_feature_rocket.c` (CTest `reduce_feature_rocket`).

## The PPU and the feature (channel) axis

The PPU is a pooling engine ([ppu-pooling-not-detile.md](../perf/ppu-pooling-not-detile.md)).
It reduces the spatial axes `[H,W]` within a channel and writes the same channel count back
(NC1HWC2 -> NC1HWC2, C1 unchanged). No register contracts across C: pooling is per-channel
by construction. A feature-axis reduce is therefore not a pool. The PPU reduce
(`rocket_global_avgpool_fp16`) and this feature reduce are orthogonal, and cover the two
different axes a transformer contracts.

### The layout-trick alternative

A feature vector can be laid on the spatial axis, cube `[C=M, H=1, W=Hfeat]`, with the
existing PPU avg-pool run over `W` to get a per-row mean. That route loses to the
ones-matmul for these reasons:

- The PPU path needs `Hfeat` 16-smooth (prime factors <=16). The ones-matmul accepts any H.
- `rocket_global_avgpool_plan` rejects `H=1` (`nh≠nw`), so it would need generalizing.
- Each pass divides by an fp16 reciprocal `fp16(65536/k)` (~1e-3 quant/pass), against the
  matmul's genuine fp32 accumulation. The sum-of-squares variance term wants every bit.

## Reduce over K as a matmul against a ones vector

The matmul computes `C[m,n] = sum_k A[m,k]·B[n,k]`. With `B = ones`, the K-sum is the reduce:

```
out[M,1] = x[M,H] · ones[1,H]^T          out[m] = sum_h x[m,h]·1 = sum_h x[m,h]
```

The reduce needs no new regcmd. It reuses `rocket_matmul_fp16_f32out` (the fp32-output path)
and inherits its genuine fp32 K-accumulation (K-partials summed in fp32/fp64, not
fp16-narrowed per tile). The mechanics:

- **N padded to 16** (the matmul's N-group). The ones weight is `[16, Kpad]`. The matmul
  computes 16 identical output columns, and the entry reads back column 0. The redundant 15
  columns are a tiny fixed cost on the K-dominated shape (no extra DRAM weight traffic beyond
  the `16xKpad` ones vector).
- **K (=H) zero-padded to %32, M to %4.** The ones weight is `1` over the real `H` columns
  and `0` over the K-pad, so padded columns contribute 0. Padded rows produce an ignored
  sum. When the input already meets `H%32==0 && M%4==0` (the common LLM case, e.g. Gemma
  H=3840), the entry passes the input buffer directly as A, with no host staging copy.
- **fp32 output.** A bare sum stays in fp16 range, but the main consumer squares first
  (sum-of-squares for variance), where the running sum easily exceeds fp16. The reduce
  therefore returns `float[M]`. The fp16 input elements must still be finite, so square and
  scale upstream (see RMSNorm).

## HW result (gate `reduce_feature_rocket`, 600 MHz, kernel 7.1.0-1)

The result is essentially bit-exact vs the fp64 oracle: `max_rel <= 1.4e-7` (fp32 noise
floor), mostly `max_abs = 0`, for both sum and mean. The gate covers:

- The M-tile boundary (M=256/260)
- Realistic Gemma widths (H=3840/2048)
- The K%32≠0 / M%4≠0 padding cases (H=48->64, M=5->8)
- Large magnitudes (amp=40, sum ~4400)

## Follow-ons

- **The reduce folds away in the FFN/QKV.** RMSNorm's per-column weight `w[h]` folds into
  the next matmul's weight (static, once). The per-row `1/rms` folds as a post-matmul per-row
  scale. The standalone reduce here is therefore the gate-grade primitive, and the in-block
  version contracts to a weight-rescale + output-scale (no separate reduce op). See RMSNorm.
- **The 16-wide N can carry up to 16 different weighted reductions of the same A** (different
  weight columns: `sum(x)`, `sum(w⊙x)`, …) instead of 16 copies of the ones column. It does
  not give `sum(x²)`: that is a reduction of a different operand (`x²`). LayerNorm's
  mean+variance therefore share one job by stacking rows (A = `[x ; x⊙x]`, 2M rows, ones
  weight -> `sum(x)` then `sum(x²)`).

## Cumsum (prefix sum): the ones column widened to a triangular matrix

A cumulative sum is a matmul by a triangular ones matrix. The feature reduce above is the
N=1 special case: a single all-ones column sums the whole K axis. With that one column
widened to the full triangle, every prefix appears as its own output column, and one matmul
produces the whole scan. The entry is `rocket_cumsum_fp16` (`src/rocket_reduce.c`), and its
gate is `tests/cumsum_rocket.c`.

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

Cumsum needs no new regcmd either. It reuses `rocket_matmul_fp16_f32out` exactly as the
feature reduce does, and inherits the genuine fp32 K-accumulation. A long prefix sums many
terms, and the fp32 accumulator avoids the per-tile fp16 narrowing the plain fp16 path would
apply.

The mechanics mirror the reduce. K(=N) is padded to %32, the output-column count N to %16,
and M to %4. The triangular weight is set over the real `[N,N]` block, and the pad rows and
columns are 0 (they contribute nothing). The entry narrows the fp32 result to fp16 on
read-back. The `[N,N]` weight is `N²` fp16 (N=1500 -> 4.5 MB), and the matmul tiles it like
any other weight.

### HW result (gate `cumsum_rocket`, 600 MHz, kernel 7.1.0-1)

The result is bit-exact vs the fp64 prefix-sum oracle: `max_abs = 0` on every shape and all
four variants. The gate covers:

- The M-tile boundary (M=256/260)
- N%32≠0 (N=100/48)
- Realistic widths (768)
- A long T=1500 prefix

The accuracy to expect in general is fp16-rounding (fp32-accum then a single fp16 narrow).
Here the sums of fp16 terms are exactly representable in the fp32 accumulator, so fp32 ==
fp64 and the narrow matches the oracle's. No long-prefix fp16 degradation appears at these
magnitudes (worst prefix ~211). If a much larger or longer prefix overflows fp16, the
RMSNorm power-of-2 prescale trick (square/scale into range, recover after) is the lever,
the same as the reduce's sum-of-squares path.

The gate's own self-check re-derives every prefix with an independent O(N²) fp64 recompute.
That catches an off-by-one prefix boundary, which would differ by a whole element.

### Consumers

Cumsum is TFLite/ONNX `CumSum`, the running-total scan in:

- Beam search
- CTC alignment
- Autoregressive and causal masking
- Segment offsets
- Any "prefix" index math

Cumsum and the [cross-entropy](whisper-encoder.md) logsumexp both belong to the
reduce-as-matmul family. The contraction axis is the only axis the matmul can sum over.
Anything expressible as "a (possibly structured) linear combination along the last axis" is
therefore one ones-weight or triangular-weight matmul with no new regcmd. That covers the
full reduce, the weighted reduce and the prefix scan.
