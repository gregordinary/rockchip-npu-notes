<!-- Raw llama-bench + llama-perplexity output backing perf/benchmarks.md (Qwen3.6-27B). The
     clk=200 in each per-run header is an idle sample taken between the discarded warmup and the
     measured run; the NPU rides to 600 MHz under load (module loaded with
     rocket_npu_clk_hz=600000000). Warm medians. Q4_K_M ONLY: at 27B the F16 GGUF (~54 GB) and
     Q8_0 (~29 GB) do not fit the 31 GB board; Q4_K_M (15.92 GiB) fits with headroom. Quant and PPL
     both at -b 2048 -ub 2048. Two methodology deviations forced by the 27B CPU baseline being
     ~1.8 t/s (a full -r 2 CPU pass is ~2.8 h): the CPU baseline runs -r 1 (its prefill variance is
     +/-0.01 t/s, the BENCHMARKS_TODO-sanctioned CPU-rep trim); the NPU headline stays -r 2.
     GGUF from unsloth/Qwen3.6-27B-MTP-GGUF; arch qwen35 (hybrid Gated-DeltaNet linear-attention +
     SSM scan: the DeltaNet/SSM-scan layers are CPU-only ops and stay there, the FFN + attention
     projections offload). Unlike the instruct/reasoning models in this record, the absolute
     wikitext PPL (~7.2) is in the normal range; the NPU-CPU delta is the faithfulness measure.
     See ../benchmarks.md Method. Generator: run_sweep_qwen36_v2.sh (bench) + run_ppl_qwen36.sh
     (PPL), 2026-07-03. The first sweep's PPL phase died on a set -u collapsed-local bug (the bench
     phases were unaffected); the PPL was re-run from the fixed run_ppl_qwen36.sh. -->

== Qwen3.6-27B Q4_K_M  Fri Jul  3 03:27:38 UTC 2026 ==
### Qwen3.6-27B Q4_K_M  [cpu]  03:37:17  clk=200 MHz  (-r 1)
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | CPU        |       8 |     2048 |           pp512 |          1.79 ± 0.00 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | CPU        |       8 |     2048 |          pp1024 |          1.79 ± 0.00 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | CPU        |       8 |     2048 |          pp2048 |          1.78 ± 0.00 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | CPU        |       8 |     2048 |            tg64 |          1.10 ± 0.00 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | CPU        |       8 |     2048 |    pp2048+tg128 |          1.70 ± 0.00 |

build: 7d2b45b4f (9568)

### Qwen3.6-27B Q4_K_M  [npu]  05:29:18  clk=200 MHz  (-r 2)
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | ROCKET     |  -1 |     2048 |           pp512 |          5.38 ± 0.02 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | ROCKET     |  -1 |     2048 |          pp1024 |          6.78 ± 0.02 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | ROCKET     |  -1 |     2048 |          pp2048 |          7.78 ± 0.01 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | ROCKET     |  -1 |     2048 |            tg64 |          1.11 ± 0.00 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | ROCKET     |  -1 |     2048 |    pp2048+tg128 |          5.57 ± 0.02 |

build: 7d2b45b4f (9568)

Prefill CPU->NPU: pp512 3.0x / pp1024 3.8x / pp2048 4.4x (the largest NPU prefill win in the record;
the win rises with M as the Q4_K_M per-microbatch dequant amortizes and the very slow 27B CPU
baseline is lapped harder). Decode NPU ~= CPU (1.10/1.11 t/s, off-NPU, 27B bandwidth-bound).

== Qwen3.6-27B faithfulness (wikitext test, -c 512, 12 chunks, -b 2048 -ub 2048; same GGUF CPU vs NPU) ==
Absolute PPL (~7.2) is in the normal range (this MTP variant models raw wikitext like a base model,
unlike the instruct/reasoning models whose absolute PPL is inflated); the NPU-CPU delta is the
faithfulness measure. Per-run stderr +/- 0.32.
### PPL Q4_K_M [cpu]  Final estimate: PPL = 7.2188 +/- 0.32293
### PPL Q4_K_M [npu]  Final estimate: PPL = 7.1997 +/- 0.32180   (delta -0.26%)
