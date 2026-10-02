# Vision normalization on the NPU (BatchNorm / GroupNorm / InstanceNorm / L2-Normalize)

This is the vision normalization family. `librocketnpu` implements it as:

- `rocket_batchnorm_fp16`
- `rocket_groupnorm_fp16`
- `rocket_instancenorm_fp16`
- `rocket_l2norm_fp16`

The sources are `src/rocket_normvision.c` and `include/rocket_normvision.h`, and the
hardware gate is `tests/norm_vision_rocket.c` (CTest `norm_vision_rocket`). The family is op
coverage, built entirely by composing validated primitives: no new regcmd, no new hardware
path, no new NPU facts. This note records the lowering (which axis each op reduces, how its
affine broadcasts), so the mapping is not re-derived. For the underlying datapaths see
[feature-reduce.md](feature-reduce.md), [rmsnorm-onnpu.md](rmsnorm-onnpu.md) and
[dpu-lut-activation.md](dpu-lut-activation.md) (the EW mul/add).

## The reduce axis and the affine broadcast

These four ops differ only in the reduce axis and the affine broadcast. All four are the
LayerNorm machinery (square -> stacked feature-reduce -> host mean/var/rsqrt -> fold the
affine to `x⊙A + B`) with a different grouping of the channels-major `[N,C,P]` tensor.
`P = H·W` is the spatial count per channel, and `P=1` for a pure `[N,C]`. What makes it
cheap is that a (batch, group) block is contiguous in `[N,C,P]`. Reshaping to
`[rows, cols]` for the feature-axis reduce is therefore a pure view, with no reorder.

| op | rows reduced over | cols per row | affine broadcast |
|----|-------------------|--------------|------------------|
| **BatchNorm** (inference) | n/a (no reduce, uses stored running `mean`/`var[C]`) | n/a | per-channel `x·s[c]+b[c]` |
| **GroupNorm** (G groups) | `N·G` = one per `(n,group)` | `(C/G)·P` | per-channel (varies within a group's row) -> full broadcast `A`/`B` |
| **InstanceNorm** | `N·C` = one per `(n,c)` | `P` | = GroupNorm with `G=C`, so the affine is per-row |
| **L2-Normalize** | `M` rows | `H` | per-row scale `1/sqrt(Σx²+eps)` (no mean subtract) |

So `G=C` -> InstanceNorm and `G=1` -> LayerNorm-over-CHW fall out of the same GroupNorm
code, and the gate exercises both. BatchNorm is the degenerate "no reduce" case. Inference
BN is a per-channel affine: two EW passes (`tmp = x⊙A`, `out = tmp+B`) over
host-materialized broadcast tensors, the same shape as the LayerNorm affine fold.

## Constraints inherited from the reused primitives

- **The reduce is a ones-vector matmul, fp32-accumulate** ([feature-reduce.md](feature-reduce.md)).
  The PPU cannot help: it pools spatial `[H,W]` within a channel and never crosses channels.
  GroupNorm and InstanceNorm reduce across the group's channels, so the matmul-reduce is the
  only path. GroupNorm stacks `[x ; x⊙x]` into `2·(N·G)` rows for one reduce job (the
  LayerNorm trick).
- **fp16-square overflow prescale** ([rmsnorm-onnpu.md](rmsnorm-onnpu.md)). With `|x|>~223`,
  `x²` overflows fp16 (max ~65504, 223²~49729). Prescale `x·2^-k` before squaring (exact,
  power-of-2), and recover the variance as `·4^k` on the host. The mean(x) branch uses `x`
  directly. Validated at amp=1000 (`maxv~2448`, the `var` stat still recovers).
- **The O(rows) mean/var/rsqrt tail stays on the host**, exact fp32, on the same reasoning as the
  transformer norms. Sending the per-group scalars to the DPU rsqrt LUT would add a
  round-trip and hit the LUT-domain problem. The variance can span decades across groups and
  layers.

## Precision and gate result

The family is validated bit-faithful on hardware: `max_abs` is pure fp16 affine rounding
(e.g. `0.002` on `O(1)` GroupNorm output, `0.0078` on `O(13)` BatchNorm output). `max_rel`
is large only where the reference value is near zero. The gate uses a combined `rel AND abs`
tolerance (the LayerNorm-gate discipline), so a near-zero ref does not false-fail. All 16
shapes read `bad=0`:

- BatchNorm (P>1 / P=1 / C%32≠0 / large-|x|)
- GroupNorm (4-group 14², G=1, 32-group, P=1, large-|x|)
- InstanceNorm (G=C)
- L2-Normalize (row-tile boundary, H%32≠0, large-|x|)

The full suite is 35/35 green, with no regressions.

## Cost and use

Like every norm here, the family is submit-bound standalone. A host single memory pass wins
for an isolated op, since the broadcast `A`/`B` materialization plus the EW passes are
several submits. The value is in two places:

- **Op coverage.** A delegate need not spill a BatchNorm/GroupNorm/L2-Norm node to the CPU
  and break an otherwise-contiguous NPU partition.
- **The cube-resident fusion substrate.** Once the activation stays in the NC1HWC2 cube
  between two NPU convs or matmuls, the norm folds in without a de-tile->host->re-pack
  round-trip. That round-trip is the dominant not-mac-bound cost.
