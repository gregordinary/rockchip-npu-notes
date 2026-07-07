<!-- Raw llama-bench + llama-perplexity output backing perf/benchmarks.md (Ministral-3-3B-Instruct-2512).
     The clk=200 in each per-run header is an idle sample taken between the discarded warmup and
     the measured run; the NPU rides to 600 MHz under load (module loaded with
     rocket_npu_clk_hz=600000000). Warm medians, llama-bench -r 2 plus a discarded warmup. Quants
     at -b 2048 -ub 2048; F16 at defaults. F16 derived from the published BF16 via llama-quantize.
     llama.cpp maps Ministral-3-3B onto the `mistral3` arch (shown as "mistral3 3B" in the rows).
     See ../benchmarks.md Method. -->

== ministral3-3b F16  Fri Jul  3 16:53:33 UTC 2026 ==
### ministral3-3b F16  [cpu]  16:54:35  clk=200 MHz
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| mistral3 3B F16                |   6.39 GiB |     3.43 B | CPU        |       8 |           pp512 |         18.02 ± 0.01 |
| mistral3 3B F16                |   6.39 GiB |     3.43 B | CPU        |       8 |          pp1024 |         17.39 ± 0.01 |
| mistral3 3B F16                |   6.39 GiB |     3.43 B | CPU        |       8 |          pp2048 |         16.26 ± 0.01 |
| mistral3 3B F16                |   6.39 GiB |     3.43 B | CPU        |       8 |            tg64 |          3.01 ± 0.01 |
| mistral3 3B F16                |   6.39 GiB |     3.43 B | CPU        |       8 |    pp2048+tg128 |         11.51 ± 0.00 |

build: 7d2b45b4f (9568)

### ministral3-3b F16  [npu]  17:14:48  clk=200 MHz
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B F16                |   6.39 GiB |     3.43 B | ROCKET     |  -1 |           pp512 |         57.62 ± 0.20 |
| mistral3 3B F16                |   6.39 GiB |     3.43 B | ROCKET     |  -1 |          pp1024 |         48.57 ± 0.18 |
| mistral3 3B F16                |   6.39 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         39.78 ± 0.02 |
| mistral3 3B F16                |   6.39 GiB |     3.43 B | ROCKET     |  -1 |            tg64 |          3.00 ± 0.00 |
| mistral3 3B F16                |   6.39 GiB |     3.43 B | ROCKET     |  -1 |    pp2048+tg128 |         19.10 ± 0.03 |

build: 7d2b45b4f (9568)

== ministral3-3b Q8_0  Fri Jul  3 17:24:20 UTC 2026 ==
### ministral3-3b Q8_0  [cpu]  17:25:30  clk=200 MHz
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| mistral3 3B Q8_0               |   3.39 GiB |     3.43 B | CPU        |       8 |     2048 |           pp512 |         16.16 ± 0.01 |
| mistral3 3B Q8_0               |   3.39 GiB |     3.43 B | CPU        |       8 |     2048 |          pp1024 |         15.54 ± 0.02 |
| mistral3 3B Q8_0               |   3.39 GiB |     3.43 B | CPU        |       8 |     2048 |          pp2048 |         14.50 ± 0.02 |
| mistral3 3B Q8_0               |   3.39 GiB |     3.43 B | CPU        |       8 |     2048 |            tg64 |          5.39 ± 0.03 |
| mistral3 3B Q8_0               |   3.39 GiB |     3.43 B | CPU        |       8 |     2048 |    pp2048+tg128 |         11.82 ± 0.00 |

build: 7d2b45b4f (9568)

### ministral3-3b Q8_0  [npu]  17:46:57  clk=200 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 3B Q8_0               |   3.39 GiB |     3.43 B | ROCKET     |  -1 |     2048 |           pp512 |         32.94 ± 0.38 |
| mistral3 3B Q8_0               |   3.39 GiB |     3.43 B | ROCKET     |  -1 |     2048 |          pp1024 |         35.14 ± 0.28 |
| mistral3 3B Q8_0               |   3.39 GiB |     3.43 B | ROCKET     |  -1 |     2048 |          pp2048 |         30.78 ± 0.12 |
| mistral3 3B Q8_0               |   3.39 GiB |     3.43 B | ROCKET     |  -1 |     2048 |            tg64 |          5.53 ± 0.00 |
| mistral3 3B Q8_0               |   3.39 GiB |     3.43 B | ROCKET     |  -1 |     2048 |    pp2048+tg128 |         20.06 ± 0.02 |

build: 7d2b45b4f (9568)

== ministral3-3b Q4_K_M  Fri Jul  3 17:57:45 UTC 2026 ==
### ministral3-3b Q4_K_M  [cpu]  17:58:55  clk=200 MHz
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | CPU        |       8 |     2048 |           pp512 |         15.55 ± 0.00 |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | CPU        |       8 |     2048 |          pp1024 |         15.10 ± 0.06 |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | CPU        |       8 |     2048 |          pp2048 |         14.32 ± 0.00 |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | CPU        |       8 |     2048 |            tg64 |          7.59 ± 0.06 |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | CPU        |       8 |     2048 |    pp2048+tg128 |         12.04 ± 0.02 |

build: 7d2b45b4f (9568)

### ministral3-3b Q4_K_M  [npu]  18:20:25  clk=200 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |     2048 |           pp512 |         32.96 ± 0.18 |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |     2048 |          pp1024 |         34.67 ± 0.29 |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |     2048 |          pp2048 |         30.54 ± 0.02 |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |     2048 |            tg64 |          7.66 ± 0.03 |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |     2048 |    pp2048+tg128 |         20.86 ± 0.04 |

build: 7d2b45b4f (9568)

== Ministral-3-3B faithfulness (wikitext test, -c 512, 12 chunks; same GGUF CPU vs NPU) ==
### PPL F16    [cpu]  Final estimate: PPL = 9.9014 +/- 0.46258
### PPL F16    [npu]  Final estimate: PPL = 9.8979 +/- 0.46231
### PPL Q4_K_M [cpu]  Final estimate: PPL = 10.1022 +/- 0.47437
### PPL Q4_K_M [npu]  Final estimate: PPL = 10.0918 +/- 0.47366
