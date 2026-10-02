# Asymmetric Mt>Nt tiling

For a shape that tiles both N and K, an asymmetric Mt>Nt tile runs the NPU datapath markedly
faster than the symmetric maximum tile. The matmul planner (`rocket_matmul_plan`) caps Mt and
Nt at `MAX_TILE` (256 on the RK3588) and then maximizes Kt to fill the CBUF. That picks a
symmetric Mt=Nt=256 tile, which fixes Kt at the symmetric fit of 384. The profile caps the
tile at 256, not 384, so Kt can reach 384.

Halving Nt to 128 while keeping Mt=256 frees CBUF, so Kt grows to 512. The knob is
`ROCKET_MM_ASYM`. It is on by default, and `=0` opts out.

## Measurement

The A/B compares the default symmetric plan against the asymmetric plan
(`Mt=256, Nt=128, Kt grown`). The arms ran interleaved and warm on an idle RK1
[HW sweep 2026-06-30, 600 MHz, warm median]:

| shape (M×K×N) | symmetric -> asym | warm Δ |
|---|---|---|
| 512×2048×2048 | Nt 256->128, Kt 384->512 | +15.8% |
| 512×2048×4096 | 256->128 | +13.8% |
| 2048×4096×4096 | 256->128 | +9.2% |
| 1024×4096×4096 | 256->128 | +7-9% |
| 512×8192×2048 | 256->128 | +8.1% |
| 1024×8192×8192 | 256->128 | +7.2% |
| 512×15360×3840 (Gemma FFN-down) | 256->128 (nKt 40->30) | +5-7% |
| 512×4096×4096 | 256->128 | +6% |
| 512×3840×15360 (Gemma FFN-up) | 256->128 | +2% |
| 512×3840×4096 | 256->128 | +2% |
| 512×4096×256 (N<=cap) | no-op (N not tiled) | ±0 |
| 512×256×4096 (nKt=1) | no-op (K not tiled) | ±0 |

The asymmetric plan is a win or a wash on all 10 tabled shapes where it fires (+2% to +16%,
biggest on moderate-K). It is never a regression, and it is an exact no-op where the guard
stops it from firing. The wins cover the square and FFN shapes that dominate LLM prefill.

## Submit and wait profile

The gain is datapath efficiency, not fence count. The asymmetric tile raises `submit` slightly
(Nt halved -> nNt doubles -> more tiles) and cuts `wait` ~10% (e.g. 1024²: wait 2047->1833 ms
over 43 batches). The wait reduction dominates. Chaining the K-fences is a separate, marginal
lever (see [k-accumulation.md](../encodings/k-accumulation.md) §ki-fence). The NPU
compute/readback pipeline runs the Mt=256/Nt=128/Kt=512 tile faster than the
Mt=256/Nt=256/Kt=384 tile.

Two compounding effects are the plausible cause [hypothesis]:

- The deeper Kt=512 K-reduction does more MAC per CBUF fill, which raises utilization.
- The taller-than-wide tile (more output rows per weight-kernel pass) amortizes the weight
  load better in the conv-as-matmul datapath.

Halving Mt instead of Nt is worse (Mt=128/Nt=256 measured slower than Mt=256/Nt=128). So the
asymmetry direction matters: keep the input-feature height (Mt) full.

## The heuristic and its guard

`ROCKET_MM_ASYM=1` halves Nt to `MAX_TILE/2` only when all of the conditions below hold.
Otherwise the default symmetric plan stands, byte-identical. The conditions are:

- No explicit `ROCKET_MM_NT` override is set, and Nt is still at the cap (N > MAX_TILE, so N
  is tiled).
- The symmetric plan K-tiles (nKt > 1). If K fits one pass (nKt=1), a bigger Kt is moot and
  the extra N-tiles are pure loss, so the heuristic must not fire. The no-op is confirmed on
  small-K and small-N shapes (N <= 256 or K <= Kt).

The result is bit-exact, because tiling never changes the result. The gate is
`matmul_correctness_matrix`, run under both settings: `matmul_correctness_matrix_asym` is
`ROCKET_MM_ASYM=1` and `matmul_correctness_matrix_sym` is `ROCKET_MM_ASYM=0`. The gate reads
cos = 1.000000 at 512×4096×4096. The heuristic composes with KACC. The A/B ran with KACC on,
the default, so the win adds to the KACC gain.

## End-to-end measurement

A wider A/B ran warm pp2048 through ggml-rocket/llama.cpp, ASYM=0 vs 1 [HW sweep]:

| model | ASYM=0 | ASYM=1 | Δ |
|---|---|---|---|
| Qwen3.5-0.8B-F16 (6 reps) [2026-06-30] | 103.35 ± 0.23 | 106.01 ± 0.13 | +2.6% |
| Qwen3.5-9B-F16 (3 reps) [2026-07-01] | 22.67 | 24.83 | +9.5% |
| Gemma-4-12B-F16 (3 reps) [2026-07-01] | 14.22 | 15.03 | +5.7% |
| Qwen3.5-9B-Q4_K, ub2048, resident=auto (3 reps) [2026-07-01] | 25.82 | 26.16 | +1.3% |

The standalone +2-16% matmul win dilutes across the full prefill (attention, norms, host
pack/readback). It still lands as a clean whole-model gain: large on F16 (+5.7-9.5% at 9B/12B)
and about noise level on quantized models. Quant prefill is dequant-bound, so the
matmul-datapath lever has little to move. The result is a win or a wash on every model, never
a regression. That cleared the default-on bar. The heuristic is default-on as of 2026-07-01
(`asym_on()` defaults to 1, and `ROCKET_MM_ASYM=0` forces the symmetric plan).

The probes are `tests/matmul_kacc_chain_bench.c` (warm A/B with `ROCKET_MM_PROFILE` for the
submit/wait split) and `tests/matmul_correctness_matrix_rocket.c` (bit-exactness).
