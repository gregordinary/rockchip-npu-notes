# Gated FFN block on the NPU (GeGLU / SwiGLU)

The transformer FFN (`gate = x·Wg^T`, `up = x·Wu^T`, `prod = act(gate)⊙up`,
`out = prod·Wd^T`) runs on the NPU as three matmuls and a gated activation, a DPU LUT pass
followed by an EW multiply. It builds on the [reduce](feature-reduce.md),
[RMSNorm](rmsnorm-onnpu.md) and [LUT activation](dpu-lut-activation.md) primitives. The block
runs in two forms. One hands off through the host between ops. The other is cube-resident: it
keeps the intermediates in the NPU's [cube layout](../GLOSSARY.md) between the matmuls.

## Constraints

Two hardware constraints apply, both documented in
[dpu-lut-activation.md](dpu-lut-activation.md): the LUT's maximum-width corruption (known
issue E9 in [../README.md](../README.md)) and the standalone GELU's flat-tail mux spike.

> [!CAUTION]
> For the SiLU domain, gate logits outside the LUT band (`|x| ≳ 12`) saturate. Software must
> keep the activation in range. Post-norm logits usually are.

The cube-resident path is single-batch only (`nMt*nNt <= 64`). Larger shapes fall back to the
host-handoff path.

## Lowering

### The gated activation

Three of the four ops are plain matmul, validated on hardware. The only other computation is
the gated activation `prod = act(gate) ⊙ up`, so that is the reusable primitive
(`rocket_geglu_fp16`), and the block wraps it between the projections. The geglu is
`rocket_activation_fp16(kind, gate)` followed by `rocket_ew_mul_fp16(act_gate, up)`, both on
the NPU. The activation is the 2-pass SiLU: sigmoid LUT then multiply, with no x~0 glitch.

`librocketnpu` implements the block as `rocket_ffn_fp16` plus the gated-activation core
`rocket_geglu_fp16`, in `src/rocket_ffn.c`, declared in `include/rocket_ffn.h`.

### Cube-resident fusion

The `rocket_ffn_fp16` composition hands off through the host between ops. Each matmul reads
back its output (NEON de-tile -> row-major), and the activation and ew ops re-scatter into
their own cubes. The result is correct, and every op costs a de-tile->host->re-pack round
trip.

The cube-resident variant `rocket_ffn_fp16_fused` keeps the `[M,I]` intermediates
cube-resident between `matmul -> act -> ⊙ -> matmul`. The gate and up projections leave full
output cubes (no de-tile). The gated activation runs element-wise over the cube bytes, and
the down matmul reads the product cube directly. Only `x` is packed in and only `out` is
read back. The non-gated encoder FFN (`fc2(act(fc1·x + b1))`) is `rocket_mlp_fp16_fused`. It
adds the `+b1` on the cube (`mm_scatter_bias_cube` + a flat ew-add), since bias is
per-column, not flat.

The down-matmul's input feature cube (`K=I`) and the geglu output cube (`channels=I`) share a
layout. An fp16 matmul's narrowed output cube and the fp16 input feature cube are the
identical `feat_idx` C2=8 layout. Feeding one matmul's output BO straight into the next (same
IOVA, zero host touch) is therefore bit-faithful. Element-wise ops (the geglu act + ⊙)
preserve the cube, and pad lanes stay 0 (`act(0)·0 = 0`). The path is built from three
pieces:

- **Full-cube output and matched tiling.** `mm_compute_kacc_cube` leaves the complete KACC
  output in a caller BO in canonical tile order (no de-tile). `mm_plan_init_pin` pins
  `Nt(gate/up) == Kt(down)` to a shared `T`=256 (a multiple of 32), so the cubes alias
  tile-for-tile. See [cross-op-chaining.md](cross-op-chaining.md).
- **Gate/up fusion** (`rocket_matmul_fp16_stream_fused`) is available, and the fused FFN does
  not use it. A fused `[M,2I]` cube interleaves gate/up by N-tile, so the on-cube `act⊙up`
  would need a strided pair-up. Two separate cubes (identical layout) keep the activation and
  mul flat.
- **matmul -> activation epilogue** (`gen_conv2d_task` / `conv_params_t.act`) is an
  alternative to a separate cube op. The fused FFN instead reuses the standalone 2-pass
  activation over the cube (accurate, no x~0 glitch).

## Accuracy

**Conditions:** RK3588. The host-handoff block at 600 MHz and kernel 7.1.0-1. The geglu at n
up to 100000, and the full FFN block up to M=128, H=2048, I=1024 (Gemma-ish). The
cube-resident FFN over shapes of multiple M, N and K tiles, including the SigLIP fc geometry.

The host-handoff block measures cos = 1.000000 with 0 coarse misses against an fp64 oracle
[HW sweep]. Its fp16 and LUT steps perturb magnitude (~1% from the SiLU LUT) but not
direction.

The cube-resident FFN measures cos 1.0 against the fp64 oracle and against the host-handoff
`rocket_ffn_fp16` [HW sweep]. Its max_abs is ~1 fp16 ULP: the only numeric difference is the
down-matmul K-tiling. The cube-resident encoder FFN measures cos 1.0 in the encoder block
[HW sweep]. In the SigLIP encoder it measures mean-layer cos 0.999984, against 0.999983 on the
host-handoff path [HW sweep].

## Performance

**Conditions:** RK3588, 600 MHz.

The cube-resident path's gain depends on the workload [HW sweep]. For the LLM, the round
trips are already cheap (NEON de-tile + KACC), and the FFN-internal subset that cross-op
chaining removes does not justify the path. In the transform-bound encoder (SigLIP/Whisper
~80% transform-to-compute), the cube-resident encoder FFN reduces the input re-tile and output de-tile
time, with a `wait` rise from the pinned-`Kt` K-fragmentation. The per-workload split and the
measured table are in [cross-op-chaining.md](cross-op-chaining.md).

## Evidence

### Accuracy

The host-handoff block's hardware test is `tests/ffn_rocket.c` (CTest `ffn_rocket`). Cosine is
its pass metric, because a layout or readback corruption collapses cosine. A standalone GELU
with the flat-tail mux spike shows cos = 0.05.

The cube-resident FFN's test is `tests/ffn_fused_rocket.c`. The cube-resident encoder FFN is
covered by `encoder_block_rocket` and `siglip_rocket`.
