# Asymmetric Mt>Nt tiling

For a shape that tiles both N and K, an asymmetric Mt>Nt tile runs the NPU datapath markedly
faster than the symmetric maximum tile. The gain is +2% to +16% on the shapes below
[HW sweep].

The matmul planner (`rocket_matmul_plan`) caps Mt and Nt at `MAX_TILE` (256 on the RK3588)
and then maximizes Kt to fill the CBUF. That rule picks a symmetric Mt=Nt=256 tile, which fixes
Kt at the symmetric fit of 384. The profile caps the tile at 256 rather than 384, so that Kt
can reach 384. Halving Nt to 128 while keeping Mt=256 frees CBUF, so Kt grows to 512. The
option is `ROCKET_MM_ASYM`. It is on by default (`asym_on()` defaults to 1), and
`ROCKET_MM_ASYM=0` forces the symmetric plan.

**Conditions:** RK3588 (RK1), NPU clock 600 MHz, KACC on, warm median, idle board,
2026-06-30. The asymmetric plan is `Mt=256, Nt=128, Kt grown`.

**Not characterized:** which of the two proposed mechanisms produces the gain (§"Cause").

| shape (M×K×N) | symmetric -> asymmetric | warm Δ |
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

The asymmetric plan gains or ties on all 10 tabled shapes where it fires, and gains most where
K is moderate. It never regresses, and it is an exact no-op where the guard stops it from
firing. The gains cover the square and FFN shapes that dominate LLM prefill.

## Cause

The gain is datapath efficiency, not fence count. The asymmetric tile raises `submit` slightly
(Nt halved -> nNt doubles -> more tiles) and cuts `wait` ~10% (e.g. 1024²: wait 2047->1833 ms
over 43 batches). The wait reduction dominates. Chaining the K-fences is a separate optimization
with a marginal gain (see [k-accumulation.md](../encodings/k-accumulation.md) §ki-fence). The
NPU compute and readback pipeline runs the Mt=256/Nt=128/Kt=512 tile faster than the
Mt=256/Nt=256/Kt=384 tile.

The likely cause is two compounding effects [hypothesis]:

- The deeper Kt=512 K-reduction does more MAC per CBUF fill, which raises utilization.
- The taller-than-wide tile (more output rows per weight-kernel pass) amortizes the weight
  load better in the conv-as-matmul datapath.

Halving Mt instead of Nt is worse: Mt=128/Nt=256 measured slower than Mt=256/Nt=128. The
direction of the asymmetry therefore matters. Keep the input-feature height (Mt) full.

## Heuristic and guard

`ROCKET_MM_ASYM=1` halves Nt to `MAX_TILE/2` only under all of the conditions below.
Otherwise the planner keeps the default symmetric plan, byte-identical. The conditions are:

- No explicit `ROCKET_MM_NT` override is set, and Nt is still at the cap (N > MAX_TILE, so N
  is tiled).
- The symmetric plan K-tiles (nKt > 1). If K fits one pass (nKt=1), a bigger Kt is moot and
  the extra N-tiles are pure loss, so the heuristic must not fire. The no-op is confirmed on
  small-K and small-N shapes (N <= 256 or K <= Kt).

The result is bit-exact, because tiling never changes the result. The heuristic composes with
KACC. The measurements ran with KACC on, the default, so the gain adds to the KACC gain.

## End-to-end measurement

In a whole-model prefill the gain is smaller [HW sweep]:

**Conditions:** warm pp2048 through ggml-rocket and llama.cpp, `ROCKET_MM_ASYM` 0 against 1.

| model | ASYM=0 | ASYM=1 | Δ |
|---|---|---|---|
| Qwen3.5-0.8B-F16 [2026-06-30] | 103.35 ± 0.23 | 106.01 ± 0.13 | +2.6% |
| Qwen3.5-9B-F16 [2026-07-01] | 22.67 | 24.83 | +9.5% |
| Gemma-4-12B-F16 [2026-07-01] | 14.22 | 15.03 | +5.7% |
| Qwen3.5-9B-Q4_K, ub2048, resident=auto [2026-07-01] | 25.82 | 26.16 | +1.3% |

The standalone +2-16% matmul gain is diluted across the full prefill: attention, norms, and the
host pack and readback. It remains a whole-model gain, large on F16 (+5.7-9.5% at 9B/12B) and
about noise level on quantized models. Quantized prefill is bound by dequantization, so the
matmul datapath has little to change there. The result is a gain or a tie on every model,
never a regression.

## Evidence

### Heuristic and guard

The bit-exactness test is `matmul_correctness_matrix`, run under both settings:
`matmul_correctness_matrix_asym` is `ROCKET_MM_ASYM=1` and `matmul_correctness_matrix_sym` is
`ROCKET_MM_ASYM=0`. It measures cos = 1.000000 at 512×4096×4096.

### Measurement method

The two plans ran interleaved and warm. The tests are `tests/matmul_kacc_chain_bench.c` (the
warm comparison, with `ROCKET_MM_PROFILE` for the submit and wait split) and
`tests/matmul_correctness_matrix_rocket.c` (bit-exactness).

### End-to-end measurement

Qwen3.5-0.8B-F16 is 6 repetitions, and the other three models 3 each. The end-to-end result
met the bar for a default, and the heuristic is on by default as of 2026-07-01.
