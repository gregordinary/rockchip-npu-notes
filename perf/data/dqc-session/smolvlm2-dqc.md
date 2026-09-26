<!-- smolvlm2-dqc  class=dqc  fp16-resident-fits=4096  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=<data>/smolvlm2/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf (1112602656 bytes)
     note: control on the one resolved -ub LOSS (0.941x)
-->
== smolvlm2-dqc  Mon Aug 31 02:05:48 UTC 2026 ==
### smolvlm2-dqc [stock] pass 1  env=''  args=''  02:06:03  clk=600 MHz  MemAvail=31478276 kB
<!--PRED 1	stock	memavail_kb=31478276	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5056,6215,6021,4865,3507,3826,3261,2911,2777,2219,4857-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |          pp2048 |         52.10 ± 0.20 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	52.10-->

### smolvlm2-dqc [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  02:08:57  clk=600 MHz  MemAvail=31536352 kB
<!--PRED 1	ub2048	memavail_kb=31536352	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4845,5775,5970,4950,3896,3804,3280,2902,2840,2196,4864-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |     2048 |          pp2048 |         49.01 ± 0.06 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	49.01-->

### smolvlm2-dqc [dqc512] pass 1  env='ROCKET_DEQUANT_CACHE_MB=4096'  args=''  02:12:01  clk=600 MHz  MemAvail=31225984 kB
<!--PRED 1	dqc512	memavail_kb=31225984	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4722,3027,2413,1718,3518,3819,3293,2931,2708,2162,4882-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |          pp2048 |         64.52 ± 0.34 |

build: 171974745 (10558)
<!--DATA 1	dqc512	pp2048	64.52-->
    [dq-cache] 165 weights held (2976MB): 2475 dequants skipped, 165 one-time fills, 0 still-streaming calls

### smolvlm2-dqc [dqc2048] pass 1  env='ROCKET_DEQUANT_CACHE_MB=4096'  args='-b 2048 -ub 2048'  02:14:27  clk=600 MHz  MemAvail=31298504 kB
<!--PRED 1	dqc2048	memavail_kb=31298212	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4761,2047,2731,1691,3263,3754,3308,2935,2825,2165,4876-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |     2048 |          pp2048 |         51.31 ± 0.10 |

build: 171974745 (10558)
<!--DATA 1	dqc2048	pp2048	51.31-->
    [dq-cache] 165 weights held (2976MB): 495 dequants skipped, 165 one-time fills, 0 still-streaming calls

### smolvlm2-dqc [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  02:17:28  clk=600 MHz  MemAvail=31526424 kB
<!--PRED 2	ub2048	memavail_kb=31526652	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3096,5754,6293,4918,3805,3787,3284,2927,2853,2167,4872-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |     2048 |          pp2048 |         48.92 ± 0.05 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	48.92-->

### smolvlm2-dqc [dqc512] pass 2  env='ROCKET_DEQUANT_CACHE_MB=4096'  args=''  02:20:32  clk=600 MHz  MemAvail=31166036 kB
<!--PRED 2	dqc512	memavail_kb=31166036	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4371,1964,2514,1655,3293,3809,3285,2926,2708,2153,4881-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |          pp2048 |         65.99 ± 0.19 |

build: 171974745 (10558)
<!--DATA 2	dqc512	pp2048	65.99-->
    [dq-cache] 165 weights held (2976MB): 2475 dequants skipped, 165 one-time fills, 0 still-streaming calls

### smolvlm2-dqc [dqc2048] pass 2  env='ROCKET_DEQUANT_CACHE_MB=4096'  args='-b 2048 -ub 2048'  02:22:55  clk=600 MHz  MemAvail=31328108 kB
<!--PRED 2	dqc2048	memavail_kb=31328108	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4908,2704,2418,1737,3226,3748,3293,2926,2932,2141,4869-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |     2048 |          pp2048 |         51.77 ± 0.14 |

build: 171974745 (10558)
<!--DATA 2	dqc2048	pp2048	51.77-->
    [dq-cache] 165 weights held (2976MB): 495 dequants skipped, 165 one-time fills, 0 still-streaming calls

### smolvlm2-dqc [stock] pass 2  env=''  args=''  02:25:54  clk=600 MHz  MemAvail=31509868 kB
<!--PRED 2	stock	memavail_kb=31509868	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5384,6222,5814,4626,3532,3788,3272,2909,2802,2187,4877-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |          pp2048 |         51.08 ± 0.21 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	51.08-->

### smolvlm2-dqc [dqc512] pass 3  env='ROCKET_DEQUANT_CACHE_MB=4096'  args=''  02:28:50  clk=600 MHz  MemAvail=31163656 kB
<!--PRED 3	dqc512	memavail_kb=31163836	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5802,5979,2343,1741,3357,3794,3320,2938,2708,2127,4878-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |          pp2048 |         65.57 ± 0.13 |

build: 171974745 (10558)
<!--DATA 3	dqc512	pp2048	65.57-->
    [dq-cache] 165 weights held (2976MB): 2475 dequants skipped, 165 one-time fills, 0 still-streaming calls

### smolvlm2-dqc [dqc2048] pass 3  env='ROCKET_DEQUANT_CACHE_MB=4096'  args='-b 2048 -ub 2048'  02:31:14  clk=600 MHz  MemAvail=31323184 kB
<!--PRED 3	dqc2048	memavail_kb=31323196	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4705,5578,2003,1644,3392,3728,3287,2921,2856,2156,4877-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |     2048 |          pp2048 |         51.60 ± 0.01 |

build: 171974745 (10558)
<!--DATA 3	dqc2048	pp2048	51.60-->
    [dq-cache] 165 weights held (2976MB): 495 dequants skipped, 165 one-time fills, 0 still-streaming calls

### smolvlm2-dqc [stock] pass 3  env=''  args=''  02:34:13  clk=600 MHz  MemAvail=31554556 kB
<!--PRED 3	stock	memavail_kb=31555148	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5945,5828,6383,4848,3776,3813,3281,2918,2878,2155,4875-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |          pp2048 |         51.72 ± 0.18 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	51.72-->

### smolvlm2-dqc [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  02:37:08  clk=600 MHz  MemAvail=31482448 kB
<!--PRED 3	ub2048	memavail_kb=31482448	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5249,6309,6398,4271,3566,3762,3305,2924,2784,2164,4885-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |     2048 |          pp2048 |         48.63 ± 0.18 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	48.63-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 51.63 | 3 | 51.08 | 52.10 | -- | -- |
| pp2048 | ub2048 | 48.85 | 3 | 48.63 | 49.01 | 0.946x | 0.941 0.958 0.940 |
| pp2048 | dqc512 | 65.36 | 3 | 64.52 | 65.99 | 1.266x | 1.238 1.292 1.268 |
| pp2048 | dqc2048 | 51.56 | 3 | 51.31 | 51.77 | 0.999x | 0.985 1.014 0.998 |

