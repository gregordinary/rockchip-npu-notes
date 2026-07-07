<!-- Raw llama-bench + llama-perplexity output backing perf/benchmarks.md (Phi-4-mini-instruct).
     The clk=200 in each per-run header is an idle sample taken between the discarded warmup and
     the measured run; the NPU rides to 600 MHz under load (module loaded with
     rocket_npu_clk_hz=600000000). Warm medians, llama-bench -r 2 plus a discarded warmup. Quants
     at -b 2048 -ub 2048; F16 at defaults. F16 derived from the published BF16 via llama-quantize.
     llama.cpp maps Phi-4-mini onto the `phi3` arch (shown as "phi3 3B" in the rows). See
     ../benchmarks.md Method. -->

== phi4-mini F16  Thu Jul  2 11:44:42 UTC 2026 ==
### phi4-mini F16  [cpu]  11:45:49  clk=200 MHz
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| phi3 3B F16                    |   7.15 GiB |     3.84 B | CPU        |       8 |           pp512 |         16.55 ± 0.03 |
| phi3 3B F16                    |   7.15 GiB |     3.84 B | CPU        |       8 |          pp1024 |         15.96 ± 0.00 |
| phi3 3B F16                    |   7.15 GiB |     3.84 B | CPU        |       8 |          pp2048 |         15.07 ± 0.02 |
| phi3 3B F16                    |   7.15 GiB |     3.84 B | CPU        |       8 |            tg64 |          2.86 ± 0.02 |
| phi3 3B F16                    |   7.15 GiB |     3.84 B | CPU        |       8 |    pp2048+tg128 |         10.71 ± 0.01 |

build: 7d2b45b4f (9568)

### phi4-mini F16  [npu]  12:07:39  clk=200 MHz
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B F16                    |   7.15 GiB |     3.84 B | ROCKET     |  -1 |           pp512 |         55.44 ± 0.33 |
| phi3 3B F16                    |   7.15 GiB |     3.84 B | ROCKET     |  -1 |          pp1024 |         47.86 ± 0.05 |
| phi3 3B F16                    |   7.15 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         39.79 ± 0.31 |
| phi3 3B F16                    |   7.15 GiB |     3.84 B | ROCKET     |  -1 |            tg64 |          2.84 ± 0.02 |
| phi3 3B F16                    |   7.15 GiB |     3.84 B | ROCKET     |  -1 |    pp2048+tg128 |         18.35 ± 0.00 |

build: 7d2b45b4f (9568)

== phi4-mini Q8_0  Thu Jul  2 12:17:25 UTC 2026 ==
### phi4-mini Q8_0  [cpu]  12:18:41  clk=200 MHz
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| phi3 3B Q8_0                   |   3.80 GiB |     3.84 B | CPU        |       8 |     2048 |           pp512 |         15.20 ± 0.07 |
| phi3 3B Q8_0                   |   3.80 GiB |     3.84 B | CPU        |       8 |     2048 |          pp1024 |         14.81 ± 0.09 |
| phi3 3B Q8_0                   |   3.80 GiB |     3.84 B | CPU        |       8 |     2048 |          pp2048 |         13.90 ± 0.05 |
| phi3 3B Q8_0                   |   3.80 GiB |     3.84 B | CPU        |       8 |     2048 |            tg64 |          4.86 ± 0.01 |
| phi3 3B Q8_0                   |   3.80 GiB |     3.84 B | CPU        |       8 |     2048 |    pp2048+tg128 |         11.31 ± 0.02 |

build: 7d2b45b4f (9568)

### phi4-mini Q8_0  [npu]  12:41:05  clk=200 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| phi3 3B Q8_0                   |   3.80 GiB |     3.84 B | ROCKET     |  -1 |     2048 |           pp512 |         37.26 ± 0.15 |
| phi3 3B Q8_0                   |   3.80 GiB |     3.84 B | ROCKET     |  -1 |     2048 |          pp1024 |         38.77 ± 0.23 |
| phi3 3B Q8_0                   |   3.80 GiB |     3.84 B | ROCKET     |  -1 |     2048 |          pp2048 |         34.91 ± 0.05 |
| phi3 3B Q8_0                   |   3.80 GiB |     3.84 B | ROCKET     |  -1 |     2048 |            tg64 |          4.82 ± 0.00 |
| phi3 3B Q8_0                   |   3.80 GiB |     3.84 B | ROCKET     |  -1 |     2048 |    pp2048+tg128 |         20.89 ± 0.03 |

build: 7d2b45b4f (9568)

== phi4-mini Q4_K_M  Thu Jul  2 12:51:03 UTC 2026 ==
### phi4-mini Q4_K_M  [cpu]  12:52:23  clk=200 MHz
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | CPU        |       8 |     2048 |           pp512 |         13.85 ± 0.01 |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | CPU        |       8 |     2048 |          pp1024 |         13.55 ± 0.01 |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | CPU        |       8 |     2048 |          pp2048 |         12.89 ± 0.03 |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | CPU        |       8 |     2048 |            tg64 |          7.04 ± 0.02 |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | CPU        |       8 |     2048 |    pp2048+tg128 |         10.95 ± 0.02 |

build: 7d2b45b4f (9568)

### phi4-mini Q4_K_M  [npu]  13:16:06  clk=200 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |     2048 |           pp512 |         36.70 ± 0.01 |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |     2048 |          pp1024 |         37.29 ± 0.12 |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |     2048 |          pp2048 |         33.42 ± 0.04 |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |     2048 |            tg64 |          6.95 ± 0.39 |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |     2048 |    pp2048+tg128 |         21.43 ± 0.01 |

build: 7d2b45b4f (9568)

== Phi-4-mini faithfulness (wikitext test, -c 512, 12 chunks; same GGUF CPU vs NPU) ==
### PPL F16    [cpu]  Final estimate: PPL = 10.9284 +/- 0.52686
### PPL F16    [npu]  Final estimate: PPL = 10.9152 +/- 0.52607
### PPL Q4_K_M [cpu]  Final estimate: PPL = 11.6977 +/- 0.57780
### PPL Q4_K_M [npu]  Final estimate: PPL = 11.5905 +/- 0.57144
