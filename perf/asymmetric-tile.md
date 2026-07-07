# Asymmetric Mt>Nt tiling beats the symmetric max-tile heuristic

The matmul planner (`rocket_matmul_plan`) caps Mt and Nt at `MAX_TILE` (256 on RK3588) and then
maximizes Kt to fill the CBUF. That picks a **symmetric** Mt=Nt=256 tile, which fixes Kt at the
symmetric fit (384 — the profile deliberately caps the tile at 256, not 384, so Kt can reach 384).
For a shape that tiles **both** N and K, that symmetric choice is **not** optimal: halving Nt to
128 while keeping Mt=256 frees CBUF so Kt grows to 512, and the resulting **asymmetric Mt>Nt tile
runs the NPU datapath markedly faster** — `ROCKET_MM_ASYM`, **default on** (`=0` opts out).

## The measurement [HW sweep 2026-06-30, 600 MHz, warm median]

A/B of the default symmetric plan vs the asymmetric plan (`Mt=256, Nt=128, Kt grown`), interleaved,
warm, on an idle RK1:

| shape (M×K×N) | symmetric → asym | warm Δ |
|---|---|---|
| 512×2048×2048 | Nt 256→128, Kt 384→512 | **+15.8%** |
| 512×2048×4096 | 256→128 | **+13.8%** |
| 2048×4096×4096 | 256→128 | **+9.2%** |
| 1024×4096×4096 | 256→128 | **+7–9%** |
| 512×8192×2048 | 256→128 | **+8.1%** |
| 1024×8192×8192 | 256→128 | **+7.2%** |
| 512×15360×3840 (Gemma FFN-down) | 256→128 (nKt 40→30) | **+5–7%** |
| 512×4096×4096 | 256→128 | **+6%** |
| 512×3840×15360 (Gemma FFN-up) | 256→128 | +2% |
| 512×3840×4096 | 256→128 | +2% |
| 512×4096×256 (N≤cap) | **no-op** (N not tiled) | ±0 |
| 512×256×4096 (nKt=1) | **no-op** (K not tiled) | ±0 |

**Win-or-wash on all 12+ shapes where it fires (+2% to +16%, biggest on moderate-K), never a
regression, and an exact no-op where the guard says it shouldn't fire.** The wins cover the square /
FFN shapes that dominate LLM prefill.

## Why it wins — datapath efficiency, not fence count

The profiler is unambiguous: the asymmetric tile **raises `submit` slightly** (Nt halved → nNt
doubles → more tiles) but **cuts `wait` ~10%** (e.g. 1024²: wait 2047→1833 ms over 43 batches),
and the wait reduction dominates. So this is **not** a fence-count win (chaining the K-fences is a
*separate*, marginal lever — see [k-accumulation.md](../encodings/k-accumulation.md) §ki-fence) — it
is the NPU **compute/readback pipeline running the Mt=256/Nt=128/Kt=512 tile faster than
Mt=256/Nt=256/Kt=384**. Two compounding effects, plausibly [hypothesis]: (1) the deeper Kt=512
K-reduction does more MAC per CBUF fill, raising utilization; (2) the taller-than-wide tile
(more output rows per weight-kernel pass) amortizes the weight load better in the conv-as-matmul
datapath. Halving **Mt** instead of Nt is *worse* (Mt=128/Nt=256 measured slower than
Mt=256/Nt=128), so the asymmetry direction matters — keep the input-feature height (Mt) full.

## The heuristic and its guard

`ROCKET_MM_ASYM=1` halves Nt to `MAX_TILE/2` **only** when all hold (else the default symmetric plan
stands, byte-identical):
- no explicit `ROCKET_MM_NT` override, and Nt is still at the cap (N > MAX_TILE — N is actually tiled);
- the symmetric plan **K-tiles** (nKt > 1). If K fits one pass (nKt=1) a bigger Kt is moot and the
  extra N-tiles are pure loss, so the heuristic must not fire — confirmed no-op on small-K / small-N
  shapes (N ≤ 256 or K ≤ Kt).

Bit-exact: tiling never changes the result; gated by `matmul_correctness_matrix` under **both**
settings (`matmul_correctness_matrix_asym` = `ROCKET_MM_ASYM=1`, `matmul_correctness_matrix_sym` =
`ROCKET_MM_ASYM=0`; cos = 1.000000 at 512×4096×4096). It composes with KACC (the A/B ran with KACC
on, default); the win is on top of KACC.

**End-to-end, wider A/B [HW sweep], warm pp2048 through ggml-rocket/llama.cpp, ASYM=0 vs 1:**

| model | ASYM=0 | ASYM=1 | Δ |
|---|---|---|---|
| Qwen3.5-0.8B-F16 (6 reps) [2026-06-30] | 103.35 ± 0.23 | 106.01 ± 0.13 | **+2.6%** |
| Qwen3.5-9B-F16 (3 reps) [2026-07-01] | 22.67 | 24.83 | **+9.5%** |
| Gemma-4-12B-F16 (3 reps) [2026-07-01] | 14.22 | 15.03 | **+5.7%** |
| Qwen3.5-9B-Q4_K, ub2048, resident=auto (3 reps) [2026-07-01] | 25.82 | 26.16 | **+1.3%** |

The standalone +6–15% matmul win dilutes across the full prefill (attention, norms, host
pack/readback) but lands as a clean whole-model gain: **large on F16 (+6–9% at 9B/12B), ~noise on
quantized** (quant prefill is dequant-bound, so the matmul-datapath lever has little to move) —
**win-or-wash on every model, never a regression.** That cleared the default-on bar: **shipped
default-on 2026-07-01** (`asym_on()` defaults to 1; `ROCKET_MM_ASYM=0` forces the symmetric plan).

Probes: `tests/matmul_kacc_chain_bench.c` (warm A/B with `ROCKET_MM_PROFILE` for the submit/wait
split), `tests/matmul_correctness_matrix_rocket.c` (bit-exactness).
