# Gated-MLP FFN block on the NPU (GeGLU / SwiGLU)

This is the transformer FFN (`gate = x·Wg^T`, `up = x·Wu^T`, `prod = act(gate)⊙up`,
`out = prod·Wd^T`), assembled on the NPU. `librocketnpu` implements it as `rocket_ffn_fp16`
plus the gated-activation core `rocket_geglu_fp16` (`src/rocket_ffn.c`,
`include/rocket_ffn.h`). The hardware gate is `tests/ffn_rocket.c` (CTest `ffn_rocket`). It
builds on the [reduce](feature-reduce.md), [RMSNorm](rmsnorm-onnpu.md) and
[LUT activation](dpu-lut-activation.md) primitives.

## The gated activation

Three of the four ops are plain matmul (validated on hardware). The only new computation is
the gated activation `prod = act(gate) ⊙ up`, so that is the reusable primitive
(`rocket_geglu_fp16`), and the block wraps it between the projections. The geglu is
`rocket_activation_fp16(kind, gate)` followed by `rocket_ew_mul_fp16(act_gate, up)`, both on
the NPU. The activation is the 2-pass SiLU: sigmoid LUT then multiply, with no x~0 glitch.

## HW result (gate `ffn_rocket`, 600 MHz, kernel 7.1.0-1)

The result is cos = 1.000000 with 0 coarse misses, vs an fp64 oracle. It holds for geglu (n
up to 100000) and the full FFN block (up to M=128, H=2048, I=1024, Gemma-ish). Cosine is the
pass metric. A multi-op fp16+LUT block perturbs magnitude (~1% SiLU LUT) but not direction,
and a layout or readback corruption collapses cosine. A standalone GELU with the flat-tail
mux spike shows cos = 0.05.

Two hardware constraints bind here, both documented in
[dpu-lut-activation.md](dpu-lut-activation.md): the LUT max-width corruption (QUIRK 3) and
the standalone-GELU gap. For the SiLU domain, gate logits outside the LUT band (`|x| ≳ 12`)
saturate. **Keep the activation in range** (post-norm logits usually are).

## Cube-resident fusion

The `rocket_ffn_fp16` composition above uses host handoff between ops. Each matmul reads
back its output (NEON de-tile -> row-major), and the activation and ew ops re-scatter into
their own cubes. That is correct but pays the de-tile->host->re-pack round-trip on every op.

The cube-resident variant `rocket_ffn_fp16_fused` keeps the `[M,I]` intermediates
cube-resident between `matmul -> act -> ⊙ -> matmul`. The gate and up projections leave full
output cubes (no de-tile). The gated activation runs element-wise over the cube bytes, and
the down matmul reads the product cube directly. Only `x` is packed in and only `out` is
read back. The non-gated encoder MLP (`fc2(act(fc1·x + b1))`) is `rocket_mlp_fp16_fused`. It
adds the `+b1` on the cube (`mm_scatter_bias_cube` + a flat ew-add), since bias is
per-column, not flat.

The mechanism is proven. The down-matmul's input feature cube (`K=I`) and the geglu output
cube (`channels=I`) share a layout. An fp16 matmul's narrowed output cube and the fp16 input
feature cube are the identical `feat_idx` C2=8 layout. Feeding one matmul's output BO
straight into the next (same IOVA, zero host touch) is therefore bit-faithful. Element-wise
ops (the geglu act + ⊙) preserve the cube, and pad lanes stay 0 (`act(0)·0 = 0`). The build
pieces:

- **Full-cube output + matched tiling** (the core): `mm_compute_kacc_cube` leaves the complete
  KACC output in a caller BO in canonical tile order (no de-tile). `mm_plan_init_pin` pins
  `Nt(gate/up) == Kt(down)` to a shared `T`=256 (a multiple of 32), so the cubes alias
  tile-for-tile. The path is single-batch only (`nMt*nNt <= 64`), and larger shapes fall
  back to the host-handoff path. See [cross-op-chaining.md](cross-op-chaining.md).
- **Gate/up fusion** (`rocket_matmul_fp16_stream_fused`) is available but not used by the fused
  FFN. A fused `[M,2I]` cube interleaves gate/up by N-tile, so the on-cube `act⊙up` would
  need a strided pair-up. Two separate cubes (identical layout) keep the activation and mul
  flat.
- **matmul -> activation epilogue** (`gen_conv2d_task` / `conv_params_t.act`) is an alternative
  to a separate cube op. The fused FFN instead reuses the standalone 2-pass activation over
  the cube (accurate, no x~0 glitch).

### Validation

`tests/ffn_fused_rocket.c` reads cos 1.0 vs the fp64 oracle and vs the host-handoff
`rocket_ffn_fp16` [HW sweep]. The max_abs is ~1 fp16 ULP: the only numeric difference is the
down-matmul K-tiling. The gate covers multi-M/N/K-tile shapes, including the SigLIP fc
geometry. The encoder MLP is covered by `encoder_block_rocket` (cos 1.0) + `siglip_rocket`
(mean-layer cos 0.999984, against 0.999983 on the host-handoff path).

### Payoff by regime

The payoff is regime-dependent [HW sweep, 600 MHz]. For the LLM the round-trips are already
cheap (NEON de-tile + KACC). The FFN-internal subset that cross-op chaining removes is
therefore a ~2-3% ceiling, which is not worth it. The regime where it pays is the transform-bound
encoder (SigLIP/Whisper ~80% transform-to-compute). On the SigLIP simple-path encode the
cube-resident MLP cuts `packA` −20%, `read` −24%, total `pack` −30% (with a `wait` rise from
the pinned-`Kt` K-fragmentation). The per-regime split and the measured table are in
[cross-op-chaining.md](cross-op-chaining.md).
