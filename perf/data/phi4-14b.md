<!-- Raw llama-bench + llama-perplexity output backing perf/benchmarks.md (Phi-4, 14B).
     The clk=200 in each per-run header is an idle sample taken between the discarded warmup and
     the measured run; the NPU rides to 600 MHz under load (module loaded with
     rocket_npu_clk_hz=600000000). Warm medians, llama-bench -r 2 plus a discarded warmup. Quants
     at -b 2048 -ub 2048. F16 is intentionally absent: the F16 GGUF (29.3 GB) does not fit the
     31 GB board (29 GB free, no swap), so Q8_0 + Q4_K_M are the precisions that run. GGUFs from
     `unsloth/phi-4-GGUF` (base model microsoft/phi-4); no local convert. llama.cpp maps the 14 B
     Phi-4 onto the `llama` arch (shown as "llama 13B" in the rows) — a different architecture from
     Phi-4-mini's `phi3`. See ../benchmarks.md Method. -->

== phi4-14b Q8_0  Sat Jul  4 05:54:47 UTC 2026 ==
### phi4-14b Q8_0  [cpu]  05:59:22  clk=200 MHz
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| llama 13B Q8_0                 |  14.51 GiB |    14.66 B | CPU        |       8 |     2048 |           pp512 |          3.83 ± 0.02 |
| llama 13B Q8_0                 |  14.51 GiB |    14.66 B | CPU        |       8 |     2048 |          pp1024 |          3.70 ± 0.01 |
| llama 13B Q8_0                 |  14.51 GiB |    14.66 B | CPU        |       8 |     2048 |          pp2048 |          3.60 ± 0.00 |
| llama 13B Q8_0                 |  14.51 GiB |    14.66 B | CPU        |       8 |     2048 |            tg64 |          1.43 ± 0.00 |
| llama 13B Q8_0                 |  14.51 GiB |    14.66 B | CPU        |       8 |     2048 |    pp2048+tg128 |          3.14 ± 0.00 |

build: 7d2b45b4f (9568)

### phi4-14b Q8_0  [npu]  07:24:22  clk=200 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama 13B Q8_0                 |  14.51 GiB |    14.66 B | ROCKET     |  -1 |     2048 |           pp512 |         10.61 ± 0.00 |
| llama 13B Q8_0                 |  14.51 GiB |    14.66 B | ROCKET     |  -1 |     2048 |          pp1024 |         12.13 ± 0.00 |
| llama 13B Q8_0                 |  14.51 GiB |    14.66 B | ROCKET     |  -1 |     2048 |          pp2048 |         12.20 ± 0.00 |
| llama 13B Q8_0                 |  14.51 GiB |    14.66 B | ROCKET     |  -1 |     2048 |            tg64 |          1.42 ± 0.00 |
| llama 13B Q8_0                 |  14.51 GiB |    14.66 B | ROCKET     |  -1 |     2048 |    pp2048+tg128 |          7.48 ± 0.01 |

build: 7d2b45b4f (9568)

== phi4-14b Q4_K_M  Sat Jul  4 07:53:38 UTC 2026 ==
### phi4-14b Q4_K_M  [cpu]  07:58:35  clk=200 MHz
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | CPU        |       8 |     2048 |           pp512 |          3.53 ± 0.01 |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | CPU        |       8 |     2048 |          pp1024 |          3.49 ± 0.01 |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | CPU        |       8 |     2048 |          pp2048 |          3.40 ± 0.01 |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | CPU        |       8 |     2048 |            tg64 |          2.17 ± 0.01 |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | CPU        |       8 |     2048 |    pp2048+tg128 |          3.13 ± 0.00 |

build: 7d2b45b4f (9568)

### phi4-14b Q4_K_M  [npu]  09:26:34  clk=200 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | ROCKET     |  -1 |     2048 |           pp512 |         10.31 ± 0.01 |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | ROCKET     |  -1 |     2048 |          pp1024 |         11.93 ± 0.03 |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | ROCKET     |  -1 |     2048 |          pp2048 |         11.79 ± 0.00 |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | ROCKET     |  -1 |     2048 |            tg64 |          2.19 ± 0.01 |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | ROCKET     |  -1 |     2048 |    pp2048+tg128 |          8.29 ± 0.00 |

build: 7d2b45b4f (9568)

== Phi-4 (14B) faithfulness (wikitext test, -c 512, 12 chunks; same GGUF CPU vs NPU) ==
### PPL Q8_0   [cpu]  Final estimate: PPL = 6.6391 +/- 0.28545
### PPL Q8_0   [npu]  Final estimate: PPL = 6.6330 +/- 0.28504
### PPL Q4_K_M [cpu]  Final estimate: PPL = 6.7688 +/- 0.29233
### PPL Q4_K_M [npu]  Final estimate: PPL = 6.7479 +/- 0.29115
