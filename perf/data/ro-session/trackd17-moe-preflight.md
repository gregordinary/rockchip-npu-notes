<!-- MoE residency pre-flight: WHICH LIMIT DOES IT NAME?   PROMPT='-p 2048 -n 0 -r 1 -b 2048 -ub 2048'
     Two MoE models, gpt-oss-20b MXFP4 and Qwen3-30B-A3B Q4_K_M, at ROCKET_N_THREADS default (5)
     and 8, everything else default.
     so=61f02a02345f3b2e27cef45de8ade0c2   board: RK1, 600 MHz, governor performance, no swap

     WHY -r 1. This is a readout of placement and of the announced limit, not a timing run. The
     pp2048 figures below are single reps and MUST NOT be quoted as measurements; they are here
     only because the harness prints them.

     WHAT IT ANSWERS. `ggml-rocket/API.md` prescribes raising ROCKET_N_THREADS as the exit from an
     exhausted NPU IOVA window on this route. The pre-flight prints which budget stopped it, and
     the two have opposite exits, so the reason string gates the question. gpt-oss is run TWICE at
     the default because a first-decline reason is a race when two limits coincide.
-->

[2026-09-02T02:45:13+00:00] probe start  so=61f02a02345f3b2e27cef45de8ade0c2
[2026-09-02T02:45:14+00:00] === gptoss_nt5_a  nt='default'  MemAvail=31713876kB
[2026-09-02T02:48:27+00:00] gptoss_nt5_a rc=0 wall=193s
--- moe/resident lines ---
[moe-int8] residency pre-flight: 24517MB RAM budget, 19200MB NPU IOVA across 5 worker fds
[moe-int8] resident budget reached after 62 expert stacks (24140MB RAM, 15693MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
[rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24517MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)
[moe-int8] ingesting experts to int8: 256 done, 2065MB resident, 5s elapsed
[moe-int8] ingesting experts to int8: 512 done, 4130MB resident, 10s elapsed
[moe-int8] ingesting experts to int8: 768 done, 6195MB resident, 16s elapsed
[moe-int8] ingesting experts to int8: 1024 done, 8260MB resident, 21s elapsed
[moe-int8] ingesting experts to int8: 1280 done, 10325MB resident, 26s elapsed
[moe-int8] ingesting experts to int8: 1536 done, 12390MB resident, 31s elapsed
[moe-int8] experts exercised: 1642 resident on the NPU (13245MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
[moe-int8] one-time ingest: 33.8s total (9.4s GGUF->int8 decode, 24.4s NPU-BO pack) for 1642 experts = 21ms each
--- result ---
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.36 ± 0.00 |

[2026-09-02T02:48:29+00:00] === gptoss_nt5_b  nt='default'  MemAvail=31714400kB
[2026-09-02T02:51:42+00:00] gptoss_nt5_b rc=0 wall=193s
--- moe/resident lines ---
[moe-int8] residency pre-flight: 24525MB RAM budget, 19200MB NPU IOVA across 5 worker fds
[moe-int8] resident budget reached after 62 expert stacks (24140MB RAM, 15693MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
[rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24525MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)
[moe-int8] ingesting experts to int8: 256 done, 2065MB resident, 5s elapsed
[moe-int8] ingesting experts to int8: 512 done, 4130MB resident, 10s elapsed
[moe-int8] ingesting experts to int8: 768 done, 6195MB resident, 16s elapsed
[moe-int8] ingesting experts to int8: 1024 done, 8260MB resident, 21s elapsed
[moe-int8] ingesting experts to int8: 1280 done, 10325MB resident, 26s elapsed
[moe-int8] ingesting experts to int8: 1536 done, 12390MB resident, 31s elapsed
[moe-int8] experts exercised: 1642 resident on the NPU (13245MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
[moe-int8] one-time ingest: 34.0s total (9.5s GGUF->int8 decode, 24.5s NPU-BO pack) for 1642 experts = 21ms each
--- result ---
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.20 ± 0.00 |

[2026-09-02T02:51:44+00:00] === gptoss_nt8  nt='8'  MemAvail=31717016kB
[2026-09-02T02:55:10+00:00] gptoss_nt8 rc=0 wall=206s
--- moe/resident lines ---
[moe-int8] residency pre-flight: 24526MB RAM budget, 30720MB NPU IOVA across 8 worker fds
[moe-int8] resident budget reached after 62 expert stacks (24140MB RAM, 15693MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
[rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24526MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)
[moe-int8] ingesting experts to int8: 256 done, 2089MB resident, 5s elapsed
[moe-int8] ingesting experts to int8: 512 done, 4178MB resident, 11s elapsed
[moe-int8] ingesting experts to int8: 768 done, 6267MB resident, 16s elapsed
[moe-int8] ingesting experts to int8: 1024 done, 8356MB resident, 21s elapsed
[moe-int8] ingesting experts to int8: 1280 done, 10445MB resident, 26s elapsed
[moe-int8] ingesting experts to int8: 1536 done, 12534MB resident, 32s elapsed
[moe-int8] experts exercised: 1648 resident on the NPU (13447MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
[moe-int8] one-time ingest: 37.2s total (11.7s GGUF->int8 decode, 25.5s NPU-BO pack) for 1648 experts = 23ms each
--- result ---
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         24.54 ± 0.00 |

[2026-09-02T02:55:13+00:00] === qwen330_nt5  nt='default'  MemAvail=31631256kB
[2026-09-02T03:01:29+00:00] qwen330_nt5 rc=0 wall=376s
--- moe/resident lines ---
--- result ---
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | ROCKET     |  -1 |     2048 |          pp2048 |         11.04 ± 0.00 |

[2026-09-02T03:01:36+00:00] === qwen330_nt8  nt='8'  MemAvail=31114836kB
[2026-09-02T03:09:35+00:00] qwen330_nt8 rc=0 wall=479s
--- moe/resident lines ---
--- result ---
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | ROCKET     |  -1 |     2048 |          pp2048 |          7.25 ± 0.00 |

[2026-09-02T03:09:35+00:00] ALL-DONE
