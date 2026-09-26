<!-- smolvlm2-dqc  class=dqc  fp16-resident-fits=4096  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=<data>/smolvlm2/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf (1112602656 bytes)
     note: control on the one resolved -ub LOSS (0.941x)
-->
== smolvlm2-dqc  Sun Aug 30 23:32:42 UTC 2026 ==
### smolvlm2-dqc [stock] pass 1  env=''  args=''  23:33:06  clk=600 MHz  MemAvail=31338700 kB
<!--PRED 1	stock	memavail_kb=31338432	anonhuge_kb=0	hugepagesz_kb=2048	buddy=1759,5410,5852,5114,4094,3907,3352,3165,3162,2388,4464-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |          pp2048 |         39.74 ± 0.51 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	39.74-->

### smolvlm2-dqc [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  23:36:58  clk=600 MHz  MemAvail=31315552 kB
<!--PRED 1	ub2048	memavail_kb=31314488	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6712,5188,6308,5201,3541,3701,3355,2992,2846,2364,4578-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |     2048 |          pp2048 |         44.42 ± 0.23 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	44.42-->

### smolvlm2-dqc [dqc512] pass 1  env='ROCKET_DEQUANT_CACHE_MB=4096'  args=''  23:40:25  clk=600 MHz  MemAvail=31254740 kB
<!--PRED 1	dqc512	memavail_kb=31254740	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4643,5721,6157,5176,3843,3865,3560,3259,2949,2091,4622-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |          pp2048 |         53.88 ± 0.26 |

build: 171974745 (10558)
<!--DATA 1	dqc512	pp2048	53.88-->
    [dq-cache] 165 weights held (2976MB): 2475 dequants skipped, 165 one-time fills, 0 still-streaming calls

### smolvlm2-dqc [dqc2048] pass 1  env='ROCKET_DEQUANT_CACHE_MB=4096'  args='-b 2048 -ub 2048'  23:43:24  clk=600 MHz  MemAvail=31269424 kB
<!--PRED 1	dqc2048	memavail_kb=31269364	anonhuge_kb=0	hugepagesz_kb=2048	buddy=11454,7517,6851,2005,3778,3909,3576,3722,3204,1929,4594-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |     2048 |          pp2048 |         47.75 ± 0.12 |

build: 171974745 (10558)
<!--DATA 1	dqc2048	pp2048	47.75-->
    [dq-cache] 165 weights held (2976MB): 495 dequants skipped, 165 one-time fills, 0 still-streaming calls

### smolvlm2-dqc [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  23:46:46  clk=600 MHz  MemAvail=31363580 kB
<!--PRED 2	ub2048	memavail_kb=31363260	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6939,8329,7175,5042,3979,3701,3267,3220,3208,2078,4605-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |     2048 |          pp2048 |         44.62 ± 0.20 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	44.62-->

### smolvlm2-dqc [dqc512] pass 2  env='ROCKET_DEQUANT_CACHE_MB=4096'  args=''  23:50:13  clk=600 MHz  MemAvail=31264892 kB
<!--PRED 2	dqc512	memavail_kb=31264892	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8086,8351,6753,2865,4204,4001,3381,3146,3061,1999,4665-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |          pp2048 |         54.74 ± 0.23 |

build: 171974745 (10558)
<!--DATA 2	dqc512	pp2048	54.74-->
    [dq-cache] 165 weights held (2976MB): 2475 dequants skipped, 165 one-time fills, 0 still-streaming calls

### smolvlm2-dqc [dqc2048] pass 2  env='ROCKET_DEQUANT_CACHE_MB=4096'  args='-b 2048 -ub 2048'  23:53:10  clk=600 MHz  MemAvail=31279820 kB
<!--PRED 2	dqc2048	memavail_kb=31279364	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10077,9561,6369,2603,3763,4189,3478,3530,3259,1877,4626-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |     2048 |          pp2048 |         47.87 ± 0.07 |

build: 171974745 (10558)
<!--DATA 2	dqc2048	pp2048	47.87-->
    [dq-cache] 165 weights held (2976MB): 495 dequants skipped, 165 one-time fills, 0 still-streaming calls

### smolvlm2-dqc [stock] pass 2  env=''  args=''  23:56:30  clk=600 MHz  MemAvail=31390952 kB
<!--PRED 2	stock	memavail_kb=31390952	anonhuge_kb=0	hugepagesz_kb=2048	buddy=12393,8046,7400,5614,4033,3874,3145,3248,3190,2070,4608-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |          pp2048 |         40.42 ± 0.03 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	40.42-->

### smolvlm2-dqc [dqc512] pass 3  env='ROCKET_DEQUANT_CACHE_MB=4096'  args=''  00:00:16  clk=600 MHz  MemAvail=31379372 kB
<!--PRED 3	dqc512	memavail_kb=31379276	anonhuge_kb=0	hugepagesz_kb=2048	buddy=12944,8235,7092,5887,3938,4218,3437,3411,3033,1969,4644-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |          pp2048 |         54.54 ± 0.22 |

build: 171974745 (10558)
<!--DATA 3	dqc512	pp2048	54.54-->
    [dq-cache] 165 weights held (2976MB): 2475 dequants skipped, 165 one-time fills, 0 still-streaming calls

### smolvlm2-dqc [dqc2048] pass 3  env='ROCKET_DEQUANT_CACHE_MB=4096'  args='-b 2048 -ub 2048'  00:03:13  clk=600 MHz  MemAvail=31273100 kB
<!--PRED 3	dqc2048	memavail_kb=31273544	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6543,7006,6939,3889,3843,4365,3556,3397,3268,1854,4634-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |     2048 |          pp2048 |         48.16 ± 0.18 |

build: 171974745 (10558)
<!--DATA 3	dqc2048	pp2048	48.16-->
    [dq-cache] 165 weights held (2976MB): 495 dequants skipped, 165 one-time fills, 0 still-streaming calls

### smolvlm2-dqc [stock] pass 3  env=''  args=''  00:06:34  clk=600 MHz  MemAvail=31341400 kB
<!--PRED 3	stock	memavail_kb=31341248	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9959,8270,6770,5400,3983,3765,3138,3369,3362,2024,4571-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |          pp2048 |         39.29 ± 0.36 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	39.29-->

### smolvlm2-dqc [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  00:10:28  clk=600 MHz  MemAvail=31367168 kB
<!--PRED 3	ub2048	memavail_kb=31367168	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9841,8593,6833,4993,3873,3926,3245,3213,2967,2155,4623-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |     2048 |          pp2048 |         44.78 ± 0.26 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	44.78-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 39.82 | 3 | 39.29 | 40.42 | -- | -- |
| pp2048 | ub2048 | 44.61 | 3 | 44.42 | 44.78 | 1.120x | 1.118 1.104 1.140 |
| pp2048 | dqc512 | 54.39 | 3 | 53.88 | 54.74 | 1.366x | 1.356 1.354 1.388 |
| pp2048 | dqc2048 | 47.93 | 3 | 47.75 | 48.16 | 1.204x | 1.202 1.184 1.226 |

