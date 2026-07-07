# Gated-MLP FFN block on the NPU (GeGLU / SwiGLU)

The transformer FFN — `gate = x·Wg^T`, `up = x·Wu^T`, `prod = act(gate)⊙up`, `out = prod·Wd^T` —
assembled on the NPU. Implemented `rocket_ffn_fp16` + the gated-activation core
`rocket_geglu_fp16` (`src/rocket_ffn.c`, `include/rocket_ffn.h`); HW gate `tests/ffn_rocket.c`
(CTest `ffn_rocket`). Built on the [reduce](feature-reduce.md) /
[RMSNorm](rmsnorm-onnpu.md) / [LUT activation](dpu-lut-activation.md) primitives.

## What is new here

Three of the four ops are plain matmul (HW-validated). The **only new computation is the
gated activation `prod = act(gate) ⊙ up`** — so that is the reusable primitive (`rocket_geglu_fp16`),
and the block wraps it between the projections. The geglu = `rocket_activation_fp16(kind,
gate)` (the 2-pass SiLU: sigmoid LUT then multiply, no x≈0 glitch) followed by
`rocket_ew_mul_fp16(act_gate, up)` — both on the NPU.

## HW result (gate `ffn_rocket`, 600 MHz, kernel 7.1.0-1)

**cos = 1.000000, 0 coarse misses** for geglu (n up to 100000) and the full FFN block (up to
M=128, H=2048, I=1024, Gemma-ish), vs an fp64 oracle. Cosine is the pass metric: a multi-op
fp16+LUT block perturbs magnitude (~1% SiLU LUT) but not direction; a layout/readback corruption
collapses cosine (a standalone GELU with the flat-tail mux spike shows cos = 0.05). Two HW
constraints bind here, both documented in [dpu-lut-activation.md](dpu-lut-activation.md): the
**LUT max-width corruption** (QUIRK 3) and the **standalone-GELU gap**. **SiLU domain:** gate
logits outside the LUT band (`|x| ≳ 12`) saturate; keep the activation in range (post-norm logits
usually are).

## Cube-resident fusion — built; the payoff is regime-dependent

The `rocket_ffn_fp16` composition above uses **host handoff** between ops: each matmul reads
back its output (NEON de-tile → row-major), the activation/ew re-scatter into their own cubes.
That is correct but pays the de-tile→host→re-pack round-trip on every op. The cube-resident
variant `rocket_ffn_fp16_fused` keeps the `[M,I]` intermediates **cube-resident** between
`matmul → act → ⊙ → matmul`: the gate/up projections leave full output cubes (no de-tile), the
gated activation runs element-wise over the cube bytes, and the down matmul reads the product
cube directly. Only `x` is packed in and only `out` is read back. The non-gated encoder MLP
(`fc2(act(fc1·x + b1))`) is `rocket_mlp_fp16_fused`, with the `+b1` added on the cube
(`mm_scatter_bias_cube` + a flat ew-add) since bias is per-column, not flat.

The mechanism is proven and the open layout question here ("the down-matmul's input feature
cube (`K=I`) and the geglu output cube (`channels=I`) must share a layout") is **resolved: they
do** — an fp16 matmul's narrowed output cube and the fp16 input feature cube are the identical
`feat_idx` C2=8 layout, so feeding one matmul's output BO straight into the next (same IOVA,
zero host touch) is bit-faithful. Element-wise ops (the geglu act + ⊙) preserve the cube; pad
lanes stay 0 (`act(0)·0 = 0`). The build pieces:

- **Full-cube output + matched tiling** (the core): `mm_compute_kacc_cube` leaves the complete
  KACC output in a caller BO in canonical tile order (no de-tile); `mm_plan_init_pin` pins
  `Nt(gate/up) == Kt(down)` to a shared `T`=256 (a multiple of 32) so the cubes alias
  tile-for-tile. Single-batch only (`nMt*nNt ≤ 64`); larger shapes fall back to the
  host-handoff path. See [cross-op-chaining.md](cross-op-chaining.md).
- **Gate/up fusion** (`rocket_matmul_fp16_stream_fused`) is available but NOT used by the fused
  FFN — a fused `[M,2I]` cube interleaves gate/up by N-tile, so the on-cube `act⊙up` would need
  a strided pair-up; two separate cubes (identical layout) keep the activation/mul flat.
- **matmul → activation epilogue** (`gen_conv2d_task` / `conv_params_t.act`) is an alternative
  to a separate cube op; the fused FFN instead reuses the standalone 2-pass activation over the
  cube (accurate; no x≈0 glitch).

**Validation** [HW sweep]: `tests/ffn_fused_rocket.c` — cos 1.0 vs the fp64 oracle and vs the
host-handoff `rocket_ffn_fp16`, max_abs ~1 fp16 ULP (the only numeric difference is the
down-matmul K-tiling), across multi-M/N/K-tile shapes incl the SigLIP fc geometry. The encoder
MLP rides `encoder_block_rocket` (cos 1.0) + `siglip_rocket` (mean-layer cos 0.999984,
unchanged).

**The payoff is regime-dependent** [HW sweep, 600 MHz]: for the LLM the round-trips are already
cheap (NEON de-tile + KACC), so the FFN-internal subset cross-op removes is a **~2–3% ceiling**
— not worth it. The regime where it pays is the **transform-bound encoder** (SigLIP/Whisper
~80% transform-to-compute): on the SigLIP simple-path encode the cube-resident MLP cut `packA`
−20%, `read` −24%, total `pack` −30% (with a `wait` rise from the pinned-`Kt` K-fragmentation).
Per-regime split + the measured table in [cross-op-chaining.md](cross-op-chaining.md).
