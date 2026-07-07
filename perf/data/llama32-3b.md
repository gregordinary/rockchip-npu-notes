<!-- Raw llama-bench + llama-perplexity output backing perf/benchmarks.md (Llama-3.2-3B-Instruct).
     The clk=200 in each per-run header is an idle sample taken between the discarded warmup and
     the measured run; the NPU rides to 600 MHz under load (module loaded with
     rocket_npu_clk_hz=600000000). Warm medians, llama-bench -r 2 plus a discarded warmup. Quants
     at -b 2048 -ub 2048; F16 at defaults. F16 is unsloth's published F16 GGUF (derived from the
     BF16); no local convert this pass. llama.cpp maps Llama-3.2-3B onto the `llama` arch (shown as
     "llama 3B" in the rows). See ../benchmarks.md Method. -->

== llama32-3b F16  Fri Jul  3 12:19:12 UTC 2026 ==
### llama32-3b F16  [cpu]  12:20:10  clk=200 MHz
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| llama 3B F16                   |   5.98 GiB |     3.21 B | CPU        |       8 |           pp512 |         18.23 ± 0.00 |
| llama 3B F16                   |   5.98 GiB |     3.21 B | CPU        |       8 |          pp1024 |         17.74 ± 0.01 |
| llama 3B F16                   |   5.98 GiB |     3.21 B | CPU        |       8 |          pp2048 |         17.29 ± 0.01 |
| llama 3B F16                   |   5.98 GiB |     3.21 B | CPU        |       8 |            tg64 |          3.15 ± 0.00 |
| llama 3B F16                   |   5.98 GiB |     3.21 B | CPU        |       8 |    pp2048+tg128 |         12.44 ± 0.01 |

build: 7d2b45b4f (9568)

### llama32-3b F16  [npu]  12:39:16  clk=200 MHz
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B F16                   |   5.98 GiB |     3.21 B | ROCKET     |  -1 |           pp512 |         61.79 ± 0.21 |
| llama 3B F16                   |   5.98 GiB |     3.21 B | ROCKET     |  -1 |          pp1024 |         52.91 ± 0.23 |
| llama 3B F16                   |   5.98 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         45.54 ± 0.03 |
| llama 3B F16                   |   5.98 GiB |     3.21 B | ROCKET     |  -1 |            tg64 |          3.27 ± 0.04 |
| llama 3B F16                   |   5.98 GiB |     3.21 B | ROCKET     |  -1 |    pp2048+tg128 |         21.06 ± 0.04 |

build: 7d2b45b4f (9568)

== llama32-3b Q8_0  Fri Jul  3 12:47:51 UTC 2026 ==
### llama32-3b Q8_0  [cpu]  12:48:54  clk=200 MHz
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| llama 3B Q8_0                  |   3.18 GiB |     3.21 B | CPU        |       8 |     2048 |           pp512 |         17.05 ± 0.02 |
| llama 3B Q8_0                  |   3.18 GiB |     3.21 B | CPU        |       8 |     2048 |          pp1024 |         16.48 ± 0.13 |
| llama 3B Q8_0                  |   3.18 GiB |     3.21 B | CPU        |       8 |     2048 |          pp2048 |         15.72 ± 0.02 |
| llama 3B Q8_0                  |   3.18 GiB |     3.21 B | CPU        |       8 |     2048 |            tg64 |          5.62 ± 0.04 |
| llama 3B Q8_0                  |   3.18 GiB |     3.21 B | CPU        |       8 |     2048 |    pp2048+tg128 |         12.80 ± 0.01 |

build: 7d2b45b4f (9568)

### llama32-3b Q8_0  [npu]  13:08:51  clk=200 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama 3B Q8_0                  |   3.18 GiB |     3.21 B | ROCKET     |  -1 |     2048 |           pp512 |         34.19 ± 0.22 |
| llama 3B Q8_0                  |   3.18 GiB |     3.21 B | ROCKET     |  -1 |     2048 |          pp1024 |         36.31 ± 0.02 |
| llama 3B Q8_0                  |   3.18 GiB |     3.21 B | ROCKET     |  -1 |     2048 |          pp2048 |         34.11 ± 0.04 |
| llama 3B Q8_0                  |   3.18 GiB |     3.21 B | ROCKET     |  -1 |     2048 |            tg64 |          5.47 ± 0.00 |
| llama 3B Q8_0                  |   3.18 GiB |     3.21 B | ROCKET     |  -1 |     2048 |    pp2048+tg128 |         21.89 ± 0.03 |

build: 7d2b45b4f (9568)

== llama32-3b Q4_K_M  Fri Jul  3 13:18:48 UTC 2026 ==
### llama32-3b Q4_K_M  [cpu]  13:19:53  clk=200 MHz
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | CPU        |       8 |     2048 |           pp512 |         16.71 ± 0.01 |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | CPU        |       8 |     2048 |          pp1024 |         16.34 ± 0.01 |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | CPU        |       8 |     2048 |          pp2048 |         15.49 ± 0.02 |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | CPU        |       8 |     2048 |            tg64 |          7.88 ± 0.00 |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | CPU        |       8 |     2048 |    pp2048+tg128 |         12.94 ± 0.02 |

build: 7d2b45b4f (9568)

### llama32-3b Q4_K_M  [npu]  13:39:50  clk=200 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |     2048 |           pp512 |         33.60 ± 0.10 |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |     2048 |          pp1024 |         36.66 ± 0.46 |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |     2048 |          pp2048 |         34.46 ± 0.02 |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |     2048 |            tg64 |          7.93 ± 0.05 |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |     2048 |    pp2048+tg128 |         22.34 ± 0.03 |

build: 7d2b45b4f (9568)

== Llama-3.2-3B faithfulness (wikitext test, -c 512, 12 chunks; same GGUF CPU vs NPU) ==
### PPL F16    [cpu]  Final estimate: PPL = 12.6656 +/- 0.64642
### PPL F16    [npu]  Final estimate: PPL = 12.6682 +/- 0.64665
### PPL Q4_K_M [cpu]  Final estimate: PPL = 12.9554 +/- 0.66220
### PPL Q4_K_M [npu]  Final estimate: PPL = 12.8978 +/- 0.65899
