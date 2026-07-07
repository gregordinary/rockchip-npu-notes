<!-- Raw llama-bench + llama-perplexity output backing perf/benchmarks.md (Gemma-4-12B-it).
     The clk=200 in each per-run header is an idle sample taken between the discarded
     warmup and the measured run; the NPU rides to 600 MHz under load (module loaded with
     rocket_npu_clk_hz=600000000, confirmed at 600 MHz by sampling during a run). Warm
     medians, llama-bench -r 2 plus a discarded warmup. Quants at -b 2048 -ub 2048; F16 at
     defaults. GGUFs from unsloth/gemma-4-12b-it-GGUF (base model google/gemma-4-12b-it);
     F16 derived from the published BF16; Q4_K_M quantized locally from BF16 (llama-quantize).
     Gemma-4-12B-it is an instruction/reasoning-tuned model, so the absolute wikitext PPL is
     inflated and not meaningful (~630); only the NPU-CPU delta is informative. See
     ../benchmarks.md Method. Generator: bench-llm.sh + run_sweep_gemma4.sh (2026-07-02). -->

== gemma-4-12B F16  Thu Jul  2 16:38:46 UTC 2026 ==
### gemma-4-12B F16  [cpu]  16:43:27  clk=200 MHz
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | CPU        |       8 |           pp512 |          4.82 ± 0.00 |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | CPU        |       8 |          pp1024 |          4.78 ± 0.01 |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | CPU        |       8 |          pp2048 |          4.63 ± 0.00 |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | CPU        |       8 |            tg64 |          0.94 ± 0.00 |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | CPU        |       8 |    pp2048+tg128 |          3.61 ± 0.00 |

build: 7d2b45b4f (9568)

### gemma-4-12B F16  [npu]  17:52:49  clk=200 MHz
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |           pp512 |         17.42 ± 0.04 |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp1024 |         16.09 ± 0.17 |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         14.98 ± 0.04 |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |            tg64 |          0.94 ± 0.00 |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |    pp2048+tg128 |          7.37 ± 0.00 |

build: 7d2b45b4f (9568)

== gemma-4-12B Q8_0  Thu Jul  2 18:18:55 UTC 2026 ==
### gemma-4-12B Q8_0  [cpu]  18:23:26  clk=200 MHz
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| gemma4 ?B Q8_0                 |  11.78 GiB |    11.91 B | CPU        |       8 |     2048 |           pp512 |          4.52 ± 0.01 |
| gemma4 ?B Q8_0                 |  11.78 GiB |    11.91 B | CPU        |       8 |     2048 |          pp1024 |          4.35 ± 0.01 |
| gemma4 ?B Q8_0                 |  11.78 GiB |    11.91 B | CPU        |       8 |     2048 |          pp2048 |          4.16 ± 0.02 |
| gemma4 ?B Q8_0                 |  11.78 GiB |    11.91 B | CPU        |       8 |     2048 |            tg64 |          1.75 ± 0.01 |
| gemma4 ?B Q8_0                 |  11.78 GiB |    11.91 B | CPU        |       8 |     2048 |    pp2048+tg128 |          3.70 ± 0.00 |

build: 7d2b45b4f (9568)

### gemma-4-12B Q8_0  [npu]  19:36:14  clk=200 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gemma4 ?B Q8_0                 |  11.78 GiB |    11.91 B | ROCKET     |  -1 |     2048 |           pp512 |         11.41 ± 0.05 |
| gemma4 ?B Q8_0                 |  11.78 GiB |    11.91 B | ROCKET     |  -1 |     2048 |          pp1024 |         13.53 ± 0.02 |
| gemma4 ?B Q8_0                 |  11.78 GiB |    11.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         13.43 ± 0.01 |
| gemma4 ?B Q8_0                 |  11.78 GiB |    11.91 B | ROCKET     |  -1 |     2048 |            tg64 |          1.59 ± 0.00 |
| gemma4 ?B Q8_0                 |  11.78 GiB |    11.91 B | ROCKET     |  -1 |     2048 |    pp2048+tg128 |          8.72 ± 0.00 |

build: 7d2b45b4f (9568)

== gemma-4-12B Q4_K_M  Thu Jul  2 20:02:21 UTC 2026 ==
### gemma-4-12B Q4_K_M  [cpu]  20:06:49  clk=200 MHz
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | CPU        |       8 |     2048 |           pp512 |          4.23 ± 0.00 |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | CPU        |       8 |     2048 |          pp1024 |          4.15 ± 0.01 |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | CPU        |       8 |     2048 |          pp2048 |          4.02 ± 0.01 |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | CPU        |       8 |     2048 |            tg64 |          2.44 ± 0.04 |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | CPU        |       8 |     2048 |    pp2048+tg128 |          3.76 ± 0.00 |

build: 7d2b45b4f (9568)

### gemma-4-12B Q4_K_M  [npu]  21:21:01  clk=200 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |     2048 |           pp512 |         11.20 ± 0.10 |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |     2048 |          pp1024 |         13.19 ± 0.06 |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         13.28 ± 0.03 |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |     2048 |            tg64 |          2.50 ± 0.02 |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |     2048 |    pp2048+tg128 |          9.61 ± 0.04 |

build: 7d2b45b4f (9568)

== Gemma-4-12B faithfulness (wikitext test, -c 512, 12 chunks; same GGUF CPU vs NPU) ==
Absolute PPL is inflated/not-meaningful (Gemma-4-12B-it is an instruct/reasoning model on raw
wikitext); only the NPU-CPU delta is informative. Per-run stderr is +/- ~65 (~+/-10%).
### PPL F16    [cpu]  Final estimate: PPL = 630.9975 +/- 62.98025
### PPL F16    [npu]  Final estimate: PPL = 625.8672 +/- 62.55655   (delta -0.81%)
### PPL Q8_0   [cpu]  Final estimate: PPL = 639.9257 +/- 64.14430
### PPL Q8_0   [npu]  Final estimate: PPL = 652.1588 +/- 65.44256   (delta +1.91%)
### PPL Q4_K_M [cpu]  Final estimate: PPL = 670.6969 +/- 67.18099
### PPL Q4_K_M [npu]  Final estimate: PPL = 675.4451 +/- 68.12533   (delta +0.71%)
