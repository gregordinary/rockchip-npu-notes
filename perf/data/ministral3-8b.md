<!-- Raw llama-bench output backing perf/benchmarks.md (Ministral-3-8B-Instruct-2512).
     The clk=200 in each per-run header is an idle sample taken between the discarded
     warmup and the measured run; the NPU rides to 600 MHz under load (confirmed by
     sampling the clock during a run). See ../benchmarks.md Method. -->

== mistral3-8B F16  Wed Jul  1 22:40:57 UTC 2026 ==
### mistral3-8B F16  [cpu]  22:43:29  clk=200 MHz
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| mistral3 8B F16                |  15.81 GiB |     8.49 B | CPU        |       8 |           pp512 |          7.22 ± 0.00 |
| mistral3 8B F16                |  15.81 GiB |     8.49 B | CPU        |       8 |          pp1024 |          7.03 ± 0.01 |
| mistral3 8B F16                |  15.81 GiB |     8.49 B | CPU        |       8 |          pp2048 |          6.84 ± 0.00 |
| mistral3 8B F16                |  15.81 GiB |     8.49 B | CPU        |       8 |            tg64 |          1.37 ± 0.01 |
| mistral3 8B F16                |  15.81 GiB |     8.49 B | CPU        |       8 |    pp2048+tg128 |          5.20 ± 0.00 |

build: 7d2b45b4f (9568)

### mistral3-8B F16  [npu]  23:30:41  clk=200 MHz
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 8B F16                |  15.81 GiB |     8.49 B | ROCKET     |  -1 |           pp512 |         27.39 ± 0.13 |
| mistral3 8B F16                |  15.81 GiB |     8.49 B | ROCKET     |  -1 |          pp1024 |         24.39 ± 0.17 |
| mistral3 8B F16                |  15.81 GiB |     8.49 B | ROCKET     |  -1 |          pp2048 |         21.44 ± 0.10 |
| mistral3 8B F16                |  15.81 GiB |     8.49 B | ROCKET     |  -1 |            tg64 |          1.43 ± 0.00 |
| mistral3 8B F16                |  15.81 GiB |     8.49 B | ROCKET     |  -1 |    pp2048+tg128 |         10.29 ± 0.02 |

build: 7d2b45b4f (9568)

== mistral3-8B Q8_0  Wed Jul  1 23:48:46 UTC 2026 ==
### mistral3-8B Q8_0  [cpu]  23:51:39  clk=200 MHz
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| mistral3 8B Q8_0               |   8.40 GiB |     8.49 B | CPU        |       8 |     2048 |           pp512 |          6.91 ± 0.02 |
| mistral3 8B Q8_0               |   8.40 GiB |     8.49 B | CPU        |       8 |     2048 |          pp1024 |          6.72 ± 0.01 |
| mistral3 8B Q8_0               |   8.40 GiB |     8.49 B | CPU        |       8 |     2048 |          pp2048 |          6.37 ± 0.00 |
| mistral3 8B Q8_0               |   8.40 GiB |     8.49 B | CPU        |       8 |     2048 |            tg64 |          2.46 ± 0.03 |
| mistral3 8B Q8_0               |   8.40 GiB |     8.49 B | CPU        |       8 |     2048 |    pp2048+tg128 |          5.44 ± 0.01 |

build: 7d2b45b4f (9568)

### mistral3-8B Q8_0  [npu]  00:39:49  clk=200 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 8B Q8_0               |   8.40 GiB |     8.49 B | ROCKET     |  -1 |     2048 |           pp512 |         17.44 ± 0.13 |
| mistral3 8B Q8_0               |   8.40 GiB |     8.49 B | ROCKET     |  -1 |     2048 |          pp1024 |         19.32 ± 0.05 |
| mistral3 8B Q8_0               |   8.40 GiB |     8.49 B | ROCKET     |  -1 |     2048 |          pp2048 |         17.67 ± 0.07 |
| mistral3 8B Q8_0               |   8.40 GiB |     8.49 B | ROCKET     |  -1 |     2048 |            tg64 |          2.47 ± 0.00 |
| mistral3 8B Q8_0               |   8.40 GiB |     8.49 B | ROCKET     |  -1 |     2048 |    pp2048+tg128 |         11.29 ± 0.02 |

build: 7d2b45b4f (9568)

== mistral3-8B Q4_K_M  Thu Jul  2 00:59:05 UTC 2026 ==
### mistral3-8B Q4_K_M  [cpu]  01:01:54  clk=200 MHz
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | CPU        |       8 |     2048 |           pp512 |          6.46 ± 0.00 |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | CPU        |       8 |     2048 |          pp1024 |          6.37 ± 0.00 |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | CPU        |       8 |     2048 |          pp2048 |          6.01 ± 0.23 |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | CPU        |       8 |     2048 |            tg64 |          3.76 ± 0.01 |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | CPU        |       8 |     2048 |    pp2048+tg128 |          5.55 ± 0.01 |

build: 7d2b45b4f (9568)

### mistral3-8B Q4_K_M  [npu]  01:51:07  clk=200 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | ROCKET     |  -1 |     2048 |           pp512 |         17.00 ± 0.19 |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | ROCKET     |  -1 |     2048 |          pp1024 |         19.12 ± 0.06 |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | ROCKET     |  -1 |     2048 |          pp2048 |         17.41 ± 0.06 |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | ROCKET     |  -1 |     2048 |            tg64 |          3.83 ± 0.00 |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | ROCKET     |  -1 |     2048 |    pp2048+tg128 |         12.26 ± 0.01 |

build: 7d2b45b4f (9568)

