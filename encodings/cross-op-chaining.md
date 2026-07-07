# Cross-op chaining — one fp16 matmul's output cube IS the next one's input

Two distinct things are called "chaining" in this stack; keep them apart:

- **Same-op instance chaining** (one HW kick over many tasks): the tiles of one matmul,
  or the per-head QK/AV of one flash-attention, laid contiguously with a
  `PC_BASE_ADDRESS` trailer so the PC runs them as one job. The mechanism and its
  fp16-only restriction are in [regcmd-task-model.md](regcmd-task-model.md)
  §contiguous-chaining.
- **Cross-op chaining** (this note): feeding the *output* of one graph op straight into
  the *input* of the next — a different op — with the host never de-tiling and re-tiling
  the intermediate.

## The enabling fact

For an fp16 matmul whose output is the default fp16-narrowed cube (`fp32tofp16=1`), the
**output cube and the input feature cube are the same layout** — both `feat_idx`, channel
atom C2=8, `(dim/8, M, 8)` (see [tile-layouts.md](tile-layouts.md); in `rocket_matmul.c`
the one `feat_idx` helper is commented "input/output cube"). So for a chain
`D = (X·W1^T)·W2^T`, the intermediate `C1 = X·W1^T` — once the DPU has written it into an
output BO — is *already* in the exact byte layout the second matmul's CNA wants for its
input. The host does not need to de-tile `C1` to row-major and re-scatter it; the second
matmul can read the first's output BO at the same IOVA.

This does **not** contradict the "no on-chip layout conversion" rule
([../matmul-as-conv.md](../matmul-as-conv.md)): that rule is about the **row-major ↔ cube
host boundary** — the host must scatter the model's row-major activations in and de-scatter
the row-major result out. It says nothing about **cube → cube** between two NPU ops, and for
fp16 the two cubes coincide.

**HW-proven [HW sweep]** (gate `tests/crossop_chain_rocket.c`, single-tile, 600 MHz): run
matmul A, then run matmul B with its input BO **aliased to A's output BO** (same handle, same
IOVA, zero host touch of the intermediate), and compare to the host-round-trip reference.

1. **Byte equivalence** — A's raw output BO is byte-identical to `C1` de-tiled then re-packed
   as B's input (`memcmp`, 0/4096 lanes differ). The DPU writes exactly what the CNA reads.
2. **Aliased compute** — B reading A's output BO directly is **bit-exact** to the host
   round-trip (`nbad=0`, sentinel-clean), cosine 0.999999 vs an fp64 oracle.

## fp16-only, and for a second reason

Like the one-kick chaining, this is fp16-only — but the cause here is the **output-cube vs
input-cube dtype mismatch**, independent of the CACC-clears-per-kick reason. Only fp16's
narrowed output is C2=8, matching the fp16 input C2=8:

| op | output cube | next-op input cube | alias? |
|---|---|---|---|
| fp16×fp16 (`fp32tofp16=1`) | fp16, C2=8 | fp16, C2=8 | **yes** |
| int8×int8 | int32, C2=4 | int8, C2=16 | no |
| int4×int4 | int16, C2=8 | int4, C2=32 | no |
| bf16×bf16 | fp32, C2=4 | bf16, C2=8 | no |
| fp16 (`fp32tofp16=0`) | fp32, C2=4 | fp16, C2=8 | no |

(The int dtypes would also need a host requant between ops, and bf16 has no narrowed output
cube. fp16-narrowed is the one self-aliasing case.)

## Element-wise ops preserve the cube

An activation or binary EW op (`rocket_activation_fp16`, `rocket_ew_mul_fp16`) is **element-
wise**, so it commutes with any bijective reindex: applying it lane-by-lane to a cube's bytes
yields the same cube with the function applied — the layout is preserved. So a
`matmul → act → ⊙ → matmul` chain (the FFN) can keep the `[M,I]` intermediate cube-resident
the whole way. Pad lanes stay correct: `SiLU(0)=0`, `0·anything=0`.

## Multi-tile and the full-cube output — built

The single-tile alias is a pure BO swap. Real shapes (FFN `I` is thousands) are multi-tile;
two things bind, and both are handled by `mm_compute_kacc_cube` (`rocket_matmul.c`) + the
pinned planner `mm_plan_init_pin`:

- **Matched tiling.** A's output N-tiling must equal B's input K-tiling: pin
  `Nt(A) == Kt(B)`, a multiple of 32 so the output slot (`rup(Mt,4)·rup(Nt,16)`) equals the
  input slot (`rup(Mt,4)·rup(Kt,32)`). `mm_plan_init_pin(M,K,N, pin_Nt, pin_Kt)` fixes the
  pinned dim and shrinks only the free dims to fit the CBUF; the fused driver pins both sides
  to the shared tile `T` (256). The slots then match tile-for-tile, and `feat_idx` uses the
  tile's real row count `Mtile`, so a ragged last M-tile (e.g. Whisper's `M=1500`) still
  aligns lane-for-lane.
- **The output BO holds the *whole* cube.** The default `mm_compute` rolls its output BO one
  `BATCH` of tiles at a time and accumulates the K-partials in **host row-major** `acc`
  (`detile_accum_f16`), so a multi-tile output never exists as one complete cube. The KACC
  path leaves the full result in a cube BO; `mm_compute_kacc_cube` does exactly that —
  forces DATA-reuse tile order (so tile `gi` lands at the canonical `(mi*nNt+ni)*out_slot`
  offset the consumer expects), K-accumulates on the NPU into the caller's `cube` BO, and
  **skips the final de-tile**, leaving the cube CPU-visible for the next op. It requires the
  result to be one NPU batch (`nMt*nNt ≤ 64`); larger shapes return `ROCKET_E_TILING` and the
  caller falls back to the host-handoff path.

An on-cube **bias** epilogue (the non-gated encoder FFN's `+bf1`) is the one piece that is
not flat element-wise — bias depends on the output column. `mm_scatter_bias_cube` builds a
bias cube once (the per-channel bias broadcast over rows, in the same canonical layout), and a
flat `rocket_ew_add_fp16` of the output cube + the bias cube adds it on-cube.

**Tradeoff — K-fragmentation.** Pinning the consumer's `Kt` to the producer's `Nt` (=`T`=256)
forces the consumer to accumulate K in tiles of `T`, which is usually *smaller* than the `Kt`
it would maximise on its own → more, smaller K-passes → more NPU `wait`/`submit`. In the
transform-bound encoder this is more than repaid by the `packA`+`read` it removes (below), but
it is not free; a larger `T` (still `%32`, still single-batch) would reduce the fragmentation.

## Where it pays — measured [HW sweep, 600 MHz, KACC, resident weights]

The intermediate de-tile/re-tile this removes is only worth removing where it is a real share
of wall. `ROCKET_MM_PROFILE` aggregated over a whole forward pass:

| workload | input re-tile `packA` | output de-tile `read` | compute `wait` | transforms / compute |
|---|---:|---:|---:|---:|
| LLM prefill (Qwen3.5-0.8B, pp512) | 289 ms | 758 ms | 12768 ms | ~8% |
| SigLIP-B/16 encoder | 6213 ms | 1919 ms | 10080 ms | ~81% |

The LLM's matmuls are large and **compute-bound**; NEON de-tile ([../perf/](../perf/))
+ KACC already make the round-trip cheap, and cross-op chaining can only remove the FFN-
internal subset → a ~2–3% ceiling, not worth the build at this operating point. The
**encoder is the opposite** — many smaller ops, each paying a full activation re-scatter, and
it runs GELU on-NPU (more ops to chain): ~10× more transform-bound, so keeping the encoder
block's intermediates cube-resident is the regime where this lever pays. Bottleneck-
conditional, not a permanent property: a higher clock, a layout that defeats the NEON
de-tile, or smaller LLM matmuls all shift the balance.

## Measured on the encoder [HW sweep, 600 MHz, performance governor]

The cube-resident encoder FFN (`rocket_mlp_fp16_fused`, wired into `rocket_encoder_block_fp16`)
A/B'd over the SigLIP-B/16 simple-path encode (12 blocks), `ROCKET_MM_PROFILE` aggregate, vs
the host-handoff path (`ROCKET_ENCODER_NOFUSE=1`):

| term | host-handoff | cube-resident | Δ |
|---|---:|---:|---:|
| `packA` (input re-tile) | 490 ms | 393 ms | **−20%** |
| `read` (output de-tile) | 362 ms | 274 ms | **−24%** |
| `pack` total | 697 ms | 485 ms | **−30%** |
| `wait` (NPU compute) | 1028 ms | 1171 ms | +14% |

The transform-bound terms drop as predicted (the fc1 output de-tile and the fc2 input scatter
are removed per block). The `wait` rises because the matched tiling pins fc2's `Kt` to the
producer's `Nt`=256, fragmenting fc2's K-accumulation into more, smaller passes (the tradeoff
above). Fidelity is unchanged (SigLIP mean-layer cos 0.999984, identical to the host-handoff
0.999983). Net-positive in this transform-bound regime; a larger shared tile would shrink the
`wait` penalty.

## Cube-chaining is single-fd — it loses to multicore on a resident path [HW sweep 2026-06-29]

The win above is on the **simple** SigLIP path, where every matmul is single-fd anyway (weights are
re-packed per call), so removing the intermediate round-trip is free upside. It does **not** carry to
a **resident** encoder whose projections are prepacked **multicore**. A cube must be one contiguous BO
on one fd (the consumer matmul reads the producer's cube at one IOVA), so a cube-resident
`fc1 → +bias → GELU → fc2` chain is **inherently single-fd** — it cannot fan fc1/fc2 across the 3 NPU
cores. A prepacked cube-fused MLP (fc1/fc2 weights packed once into the `Nt==Kt` cube-pinned layout,
bias pre-scattered once) is **~15 % slower** end-to-end than the multicore host-handoff FFN on the
resident SigLIP encode (the forfeited 3-core matmul outweighs the saved host GELU +
intermediate de-tile/re-tile), cosine still 0.999987. **So cube-chaining and multicore parallelism are
mutually exclusive; pick multicore whenever the per-op matmul is large enough to fan (the resident
encoder FFN is).** Cube-chaining wins only where the consumer would be single-fd regardless: the
simple-path encoder, or a chain whose ops are too small to fan. The resident SigLIP speed lever is the
opposite axis — **multicore the attention** ([siglip-encoder.md](siglip-encoder.md), 1.44–1.51×).

See [ffn-block.md](ffn-block.md) (the FFN composition + the resident-cube handoff this
resolves) and [siglip-encoder.md](siglip-encoder.md) (the transform-bound encoder floor).
