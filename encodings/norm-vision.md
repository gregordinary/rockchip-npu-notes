# Vision normalization on the NPU (BatchNorm / GroupNorm / InstanceNorm / L2-Normalize)

The vision normalizations run on the NPU as the LayerNorm composition, with a different
grouping of the channels. Each one squares the input in fp16 and reduces it over the feature
axis with a ones-vector matmul. The host computes the mean, the variance and the rsqrt, and the
DPU's EW multiply and add apply the folded affine. The ops use only these validated
primitives and need no further register programming. The primitives are described in
[feature-reduce.md](feature-reduce.md), [rmsnorm-onnpu.md](rmsnorm-onnpu.md) and
[dpu-lut-activation.md](dpu-lut-activation.md) (the EW mul and add).

## Constraints

- **The reduce is a ones-vector matmul with fp32 accumulation**
  ([feature-reduce.md](feature-reduce.md)). GroupNorm and InstanceNorm reduce across the
  group's channels, so the matmul reduce is the only path. The PPU pools spatial `[H,W]`
  within a channel and never crosses channels. GroupNorm stacks `[x ; x⊙x]` into `2·(N·G)`
  rows for one reduce job, as LayerNorm does.
- **The fp16 square needs an overflow prescale** ([rmsnorm-onnpu.md](rmsnorm-onnpu.md)).
  `x²` overflows fp16 (max ~65504) once `|x|>256`. Past `|x|>~223` (223²~49729), the input
  is prescaled `x·2^-k` before squaring (exact, power-of-2), and the variance is recovered as
  `·4^k` on the host. The mean(x) branch uses `x` directly. The prescale is validated at
  amp=1000 (`maxv~2448`), where the `var` statistic still recovers.
- **The O(rows) mean, variance and rsqrt stay on the host**, in exact fp32, as for the
  transformer norms. Sending the per-group scalars to the DPU rsqrt LUT would add a round trip
  and meet the LUT's domain limit ([rmsnorm-onnpu.md](rmsnorm-onnpu.md)). The variance can
  span decades across groups and layers.

## Lowering

The four ops differ only in the reduce axis and the affine broadcast. Each is the LayerNorm
sequence (square -> stacked feature-reduce -> host mean/var/rsqrt -> fold the affine to
`x⊙A + B`) over a different grouping of the channels-major `[N,C,P]` tensor. `P = H·W` is the
spatial count per channel, and `P=1` for a pure `[N,C]`. A (batch, group) block is contiguous
in `[N,C,P]`, so reshaping to `[rows, cols]` for the feature-axis reduce is a pure view, with
no reorder.

| op | `librocketnpu` entry | rows reduced over | cols per row | affine broadcast |
|----|----|-------------------|--------------|------------------|
| **BatchNorm** (inference) | `rocket_batchnorm_fp16` | n/a (no reduce, uses stored running `mean`/`var[C]`) | n/a | per-channel `x·s[c]+b[c]` |
| **GroupNorm** (G groups) | `rocket_groupnorm_fp16` | `N·G` = one per `(n,group)` | `(C/G)·P` | per-channel (varies within a group's row) -> full broadcast `A`/`B` |
| **InstanceNorm** | `rocket_instancenorm_fp16` | `N·C` = one per `(n,c)` | `P` | = GroupNorm with `G=C`, so the affine is per-row |
| **L2-Normalize** | `rocket_l2norm_fp16` | `M` rows | `H` | per-row scale `1/sqrt(Σx²+eps)` (no mean subtract) |

GroupNorm with `G=C` is InstanceNorm, and with `G=1` it is LayerNorm over CHW, so the same
GroupNorm code computes all three. BatchNorm is the case with no reduce. Inference BN is a
per-channel affine: two EW passes (`tmp = x⊙A`, `out = tmp+B`) over host-materialized
broadcast tensors, the same shape as the LayerNorm affine fold.

The entries are implemented in `src/rocket_normvision.c` and declared in
`include/rocket_normvision.h`.

## Accuracy

**Conditions:** RK3588 (Turing RK1), fp16, the 16 shapes listed in §"Evidence", 2026-06-23.

The four ops are bit-faithful on hardware [HW sweep]. `max_abs` is pure fp16 affine rounding,
for example `0.002` on `O(1)` GroupNorm output and `0.0078` on `O(13)` BatchNorm output.
`max_rel` is large only where the reference value is near zero.

## Performance

Standalone, the family is bound by its submits, like every norm on the NPU. For an
isolated op, a single host memory pass is faster, because the broadcast `A`/`B`
materialization and the EW passes take several submits. The NPU path has two uses:

- **Operator coverage.** A delegate keeps a BatchNorm, GroupNorm or L2-Norm node on the NPU.
  Moving it to the CPU would split an otherwise contiguous NPU partition.
- **Fusion in the cube layout.** Once the activation stays in the NC1HWC2
  [cube layout](../GLOSSARY.md) between two NPU convolutions or matmuls, the norm folds in
  without a de-tile->host->re-pack round trip. That round trip is the dominant host cost of an
  NPU matmul ([../perf/not-mac-bound.md](../perf/not-mac-bound.md)).

## Evidence

### Lowering

The hardware test is `tests/norm_vision_rocket.c` (CTest `norm_vision_rocket`). It runs the
GroupNorm code at both `G=C` and `G=1`.

### Accuracy

The test scores each shape with a combined `rel AND abs` tolerance, as the LayerNorm test does,
so a near-zero reference value does not fail on `max_rel`. All 16 shapes measured `bad=0` on
the RK1 (2026-06-23):

- BatchNorm (P>1 / P=1 / C%32≠0 / large-|x|)
- GroupNorm (4-group 14², G=1, 32-group, P=1, large-|x|)
- InstanceNorm (G=C)
- L2-Normalize (row-tile boundary, H%32≠0, large-|x|)

The full CTest suite on that date passed 35/35, with no regressions.
