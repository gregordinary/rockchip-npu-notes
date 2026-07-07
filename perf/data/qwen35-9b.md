<!-- Raw llama-bench + llama-perplexity output backing perf/benchmarks.md (Qwen3.5-9B).
     The clk=200 in each per-run header is an idle sample taken between the discarded
     warmup and the measured run; the NPU rides to 600 MHz under load (module loaded with
     rocket_npu_clk_hz=600000000, confirmed at 600 MHz by sampling during a run). Warm
     medians, llama-bench -r 2 plus a discarded warmup. Quants at -b 2048 -ub 2048; F16 at
     defaults. IQ4_XS carried as a gap-finder (importance-matrix 4-bit). See ../benchmarks.md
     Method. Generator: bench-llm.sh + run_sweep_qwen35.sh (2026-07-02). -->

== qwen35-9B F16  Thu Jul  2 05:48:09 UTC 2026 ==
### qwen35-9B F16  [cpu]  05:50:42  clk=200 MHz
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35 9B F16                  |  16.68 GiB |     8.95 B | CPU        |       8 |           pp512 |          7.25 ± 0.01 |
| qwen35 9B F16                  |  16.68 GiB |     8.95 B | CPU        |       8 |          pp1024 |          7.14 ± 0.01 |
| qwen35 9B F16                  |  16.68 GiB |     8.95 B | CPU        |       8 |          pp2048 |          7.08 ± 0.00 |
| qwen35 9B F16                  |  16.68 GiB |     8.95 B | CPU        |       8 |            tg64 |          1.35 ± 0.00 |
| qwen35 9B F16                  |  16.68 GiB |     8.95 B | CPU        |       8 |    pp2048+tg128 |          5.59 ± 0.00 |

build: 7d2b45b4f (9568)

### qwen35-9B F16  [npu]  06:36:12  clk=200 MHz
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B F16                  |  16.68 GiB |     8.95 B | ROCKET     |  -1 |           pp512 |         25.86 ± 0.07 |
| qwen35 9B F16                  |  16.68 GiB |     8.95 B | ROCKET     |  -1 |          pp1024 |         25.14 ± 0.18 |
| qwen35 9B F16                  |  16.68 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         24.85 ± 0.05 |
| qwen35 9B F16                  |  16.68 GiB |     8.95 B | ROCKET     |  -1 |            tg64 |          1.34 ± 0.01 |
| qwen35 9B F16                  |  16.68 GiB |     8.95 B | ROCKET     |  -1 |    pp2048+tg128 |         11.96 ± 0.02 |

build: 7d2b45b4f (9568)

== qwen35-9B Q4_K_M  Thu Jul  2 06:52:33 UTC 2026 ==
### qwen35-9B Q4_K_M  [cpu]  06:55:29  clk=200 MHz
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | CPU        |       8 |     2048 |           pp512 |          6.25 ± 0.01 |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | CPU        |       8 |     2048 |          pp1024 |          6.21 ± 0.00 |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | CPU        |       8 |     2048 |          pp2048 |          6.15 ± 0.01 |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | CPU        |       8 |     2048 |            tg64 |          3.60 ± 0.01 |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | CPU        |       8 |     2048 |    pp2048+tg128 |          5.83 ± 0.01 |

build: 7d2b45b4f (9568)

### qwen35-9B Q4_K_M  [npu]  07:44:13  clk=200 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |           pp512 |         16.68 ± 0.22 |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp1024 |         21.31 ± 0.19 |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         22.88 ± 0.01 |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |            tg64 |          3.59 ± 0.00 |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |    pp2048+tg128 |         16.77 ± 0.09 |

build: 7d2b45b4f (9568)

== qwen35-9B IQ4_XS  Thu Jul  2 07:59:12 UTC 2026 ==
### qwen35-9B IQ4_XS  [cpu]  08:01:40  clk=200 MHz
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| qwen35 9B IQ4_XS - 4.25 bpw    |   4.80 GiB |     8.95 B | CPU        |       8 |     2048 |           pp512 |          7.44 ± 0.01 |
| qwen35 9B IQ4_XS - 4.25 bpw    |   4.80 GiB |     8.95 B | CPU        |       8 |     2048 |          pp1024 |          7.37 ± 0.01 |
| qwen35 9B IQ4_XS - 4.25 bpw    |   4.80 GiB |     8.95 B | CPU        |       8 |     2048 |          pp2048 |          7.29 ± 0.00 |
| qwen35 9B IQ4_XS - 4.25 bpw    |   4.80 GiB |     8.95 B | CPU        |       8 |     2048 |            tg64 |          4.08 ± 0.02 |
| qwen35 9B IQ4_XS - 4.25 bpw    |   4.80 GiB |     8.95 B | CPU        |       8 |     2048 |    pp2048+tg128 |          6.88 ± 0.00 |

build: 7d2b45b4f (9568)

### qwen35-9B IQ4_XS  [npu]  08:43:03  clk=200 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B IQ4_XS - 4.25 bpw    |   4.80 GiB |     8.95 B | ROCKET     |  -1 |     2048 |           pp512 |         16.33 ± 0.03 |
| qwen35 9B IQ4_XS - 4.25 bpw    |   4.80 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp1024 |         20.87 ± 0.16 |
| qwen35 9B IQ4_XS - 4.25 bpw    |   4.80 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         22.49 ± 0.27 |
| qwen35 9B IQ4_XS - 4.25 bpw    |   4.80 GiB |     8.95 B | ROCKET     |  -1 |     2048 |            tg64 |          4.09 ± 0.02 |
| qwen35 9B IQ4_XS - 4.25 bpw    |   4.80 GiB |     8.95 B | ROCKET     |  -1 |     2048 |    pp2048+tg128 |         17.01 ± 0.08 |

build: 7d2b45b4f (9568)

== Qwen3.5-9B faithfulness (wikitext test, -c 512, 12 chunks; same GGUF CPU vs NPU) ==
### PPL F16    [cpu]  Final estimate: PPL = 9.0637 +/- 0.42669
### PPL F16    [npu]  Final estimate: PPL = 9.0683 +/- 0.42690
### PPL Q4_K_M [cpu]  Final estimate: PPL = 9.2358 +/- 0.43512
### PPL Q4_K_M [npu]  Final estimate: PPL = 9.2294 +/- 0.43525
### PPL IQ4_XS [cpu]  Final estimate: PPL = 9.3502 +/- 0.44260
### PPL IQ4_XS [npu]  Final estimate: PPL = 9.3411 +/- 0.44262
