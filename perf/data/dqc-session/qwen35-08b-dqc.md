<!-- qwen35-08b-dqc  class=dqc  fp16-resident-fits=1536  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=<data>/qwen35/Qwen3.5-0.8B-Q4_K_M.gguf (532517120 bytes)
     note: dense -ub mechanism control; proves the dqc arms cheaply
-->
== qwen35-08b-dqc  Mon Aug 31 00:29:51 UTC 2026 ==
### qwen35-08b-dqc [stock] pass 1  env=''  args=''  00:30:05  clk=600 MHz  MemAvail=31596544 kB
<!--PRED 1	stock	memavail_kb=31596792	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5684,6614,6236,5704,4864,3884,3314,3464,3174,2202,4832-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |          pp2048 |         99.75 ± 0.50 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	99.75-->

### qwen35-08b-dqc [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  00:31:42  clk=600 MHz  MemAvail=31537248 kB
<!--PRED 1	ub2048	memavail_kb=31537248	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5988,5969,6453,5925,4690,3806,3325,3311,3168,2204,4838-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |     2048 |          pp2048 |        112.36 ± 0.07 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	112.36-->

### qwen35-08b-dqc [dqc512] pass 1  env='ROCKET_DEQUANT_CACHE_MB=1536'  args=''  00:33:10  clk=600 MHz  MemAvail=31242144 kB
<!--PRED 1	dqc512	memavail_kb=31242144	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5887,6275,4569,2154,3508,3799,3278,3232,3141,2207,4841-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        114.55 ± 0.17 |

build: 171974745 (10558)
<!--DATA 1	dqc512	pp2048	114.55-->
    [dq-cache] 150 weights held (948MB): 2250 dequants skipped, 150 one-time fills, 0 still-streaming calls

### qwen35-08b-dqc [dqc2048] pass 1  env='ROCKET_DEQUANT_CACHE_MB=1536'  args='-b 2048 -ub 2048'  00:34:39  clk=600 MHz  MemAvail=31407768 kB
<!--PRED 1	dqc2048	memavail_kb=31408152	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6199,6884,5920,3641,3284,3809,3335,3368,3144,2193,4852-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |     2048 |          pp2048 |        116.78 ± 0.20 |

build: 171974745 (10558)
<!--DATA 1	dqc2048	pp2048	116.78-->
    [dq-cache] 150 weights held (948MB): 450 dequants skipped, 150 one-time fills, 0 still-streaming calls

### qwen35-08b-dqc [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  00:36:05  clk=600 MHz  MemAvail=31488072 kB
<!--PRED 2	ub2048	memavail_kb=31487820	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5769,6012,6268,5243,4089,3790,3298,3319,3116,2201,4859-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |     2048 |          pp2048 |        110.57 ± 0.37 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	110.57-->

### qwen35-08b-dqc [dqc512] pass 2  env='ROCKET_DEQUANT_CACHE_MB=1536'  args=''  00:37:34  clk=600 MHz  MemAvail=31385596 kB
<!--PRED 2	dqc512	memavail_kb=31385596	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5363,5907,5890,5068,4049,3794,3296,3210,3089,2194,4862-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        126.81 ± 0.45 |

build: 171974745 (10558)
<!--DATA 2	dqc512	pp2048	126.81-->
    [dq-cache] 150 weights held (948MB): 2250 dequants skipped, 150 one-time fills, 0 still-streaming calls

### qwen35-08b-dqc [dqc2048] pass 2  env='ROCKET_DEQUANT_CACHE_MB=1536'  args='-b 2048 -ub 2048'  00:38:55  clk=600 MHz  MemAvail=31312220 kB
<!--PRED 2	dqc2048	memavail_kb=31312220	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4356,6520,6330,4254,2935,3798,3328,3205,3085,2197,4862-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |     2048 |          pp2048 |        116.70 ± 0.68 |

build: 171974745 (10558)
<!--DATA 2	dqc2048	pp2048	116.70-->
    [dq-cache] 150 weights held (948MB): 450 dequants skipped, 150 one-time fills, 0 still-streaming calls

### qwen35-08b-dqc [stock] pass 2  env=''  args=''  00:40:21  clk=600 MHz  MemAvail=31570920 kB
<!--PRED 2	stock	memavail_kb=31570920	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5862,6525,6457,5395,4731,3815,3304,3375,3110,2194,4863-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        106.32 ± 0.59 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	106.32-->

### qwen35-08b-dqc [dqc512] pass 3  env='ROCKET_DEQUANT_CACHE_MB=1536'  args=''  00:41:54  clk=600 MHz  MemAvail=31347044 kB
<!--PRED 3	dqc512	memavail_kb=31347044	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6395,6568,6443,5503,4424,3889,3328,3063,3027,2200,4863-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        117.73 ± 0.41 |

build: 171974745 (10558)
<!--DATA 3	dqc512	pp2048	117.73-->
    [dq-cache] 150 weights held (948MB): 2250 dequants skipped, 150 one-time fills, 0 still-streaming calls

### qwen35-08b-dqc [dqc2048] pass 3  env='ROCKET_DEQUANT_CACHE_MB=1536'  args='-b 2048 -ub 2048'  00:43:20  clk=600 MHz  MemAvail=31266044 kB
<!--PRED 3	dqc2048	memavail_kb=31266044	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6344,6289,5138,1661,2362,3739,3350,3328,3134,2185,4864-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |     2048 |          pp2048 |        115.27 ± 0.21 |

build: 171974745 (10558)
<!--DATA 3	dqc2048	pp2048	115.27-->
    [dq-cache] 150 weights held (948MB): 450 dequants skipped, 150 one-time fills, 0 still-streaming calls

### qwen35-08b-dqc [stock] pass 3  env=''  args=''  00:44:47  clk=600 MHz  MemAvail=31551276 kB
<!--PRED 3	stock	memavail_kb=31551276	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6309,6368,6102,5474,4565,3832,3294,3334,3115,2189,4868-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        108.14 ± 0.41 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	108.14-->

### qwen35-08b-dqc [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  00:46:17  clk=600 MHz  MemAvail=31484048 kB
<!--PRED 3	ub2048	memavail_kb=31484056	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5761,6508,5954,5440,4322,3794,3304,3196,3114,2202,4868-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |     2048 |          pp2048 |        111.61 ± 0.97 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	111.61-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 104.74 | 3 | 99.75 | 108.14 | -- | -- |
| pp2048 | ub2048 | 111.51 | 3 | 110.57 | 112.36 | 1.066x | 1.126 1.040 1.032 |
| pp2048 | dqc512 | 119.70 | 3 | 114.55 | 126.81 | 1.143x | 1.148 1.193 1.089 |
| pp2048 | dqc2048 | 116.25 | 3 | 115.27 | 116.78 | 1.111x | 1.171 1.098 1.066 |

