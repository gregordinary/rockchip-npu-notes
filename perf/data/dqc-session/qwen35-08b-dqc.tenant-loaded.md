<!-- qwen35-08b-dqc  class=dqc  fp16-resident-fits=1536  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=<data>/qwen35/Qwen3.5-0.8B-Q4_K_M.gguf (532517120 bytes)
     note: dense -ub mechanism control; proves the dqc arms cheaply
-->
== qwen35-08b-dqc  Sun Aug 30 21:33:43 UTC 2026 ==
### qwen35-08b-dqc [stock] pass 1  env=''  args=''  21:34:03  clk=600 MHz  MemAvail=31338836 kB
<!--PRED 1	stock	memavail_kb=31338468	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5334,6515,6383,6405,4841,4595,4513,4051,3319,2287,4500-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |          pp2048 |         65.65 ± 0.35 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	65.65-->

### qwen35-08b-dqc [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  21:36:28  clk=600 MHz  MemAvail=31307488 kB
<!--PRED 1	ub2048	memavail_kb=31307228	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6205,5532,5998,5897,4610,4524,4241,3854,3221,2351,4539-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |     2048 |          pp2048 |         97.25 ± 1.22 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	97.25-->

### qwen35-08b-dqc [dqc512] pass 1  env='ROCKET_DEQUANT_CACHE_MB=1536'  args=''  21:38:13  clk=600 MHz  MemAvail=31180384 kB
<!--PRED 1	dqc512	memavail_kb=31180384	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6991,5471,5624,4776,3732,4647,4314,3800,3087,2357,4560-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |          pp2048 |         85.24 ± 0.35 |

build: 171974745 (10558)
<!--DATA 1	dqc512	pp2048	85.24-->
    [dq-cache] 150 weights held (948MB): 2250 dequants skipped, 150 one-time fills, 0 still-streaming calls

### qwen35-08b-dqc [dqc2048] pass 1  env='ROCKET_DEQUANT_CACHE_MB=1536'  args='-b 2048 -ub 2048'  21:40:10  clk=600 MHz  MemAvail=31163540 kB
<!--PRED 1	dqc2048	memavail_kb=31163652	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6650,6341,5740,5047,3281,4751,4342,3843,3065,2312,4577-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |     2048 |          pp2048 |        104.68 ± 0.41 |

build: 171974745 (10558)
<!--DATA 1	dqc2048	pp2048	104.68-->
    [dq-cache] 150 weights held (948MB): 450 dequants skipped, 150 one-time fills, 0 still-streaming calls

### qwen35-08b-dqc [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  21:41:52  clk=600 MHz  MemAvail=31324444 kB
<!--PRED 2	ub2048	memavail_kb=31324716	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6946,6638,6550,5952,4910,4405,4129,3852,3143,2306,4586-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |     2048 |          pp2048 |         97.73 ± 1.00 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	97.73-->

### qwen35-08b-dqc [dqc512] pass 2  env='ROCKET_DEQUANT_CACHE_MB=1536'  args=''  21:43:36  clk=600 MHz  MemAvail=31171244 kB
<!--PRED 2	dqc512	memavail_kb=31171244	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6470,5862,6363,4153,4150,4594,4238,3809,3023,2313,4597-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |          pp2048 |         85.54 ± 0.42 |

build: 171974745 (10558)
<!--DATA 2	dqc512	pp2048	85.54-->
    [dq-cache] 150 weights held (948MB): 2250 dequants skipped, 150 one-time fills, 0 still-streaming calls

### qwen35-08b-dqc [dqc2048] pass 2  env='ROCKET_DEQUANT_CACHE_MB=1536'  args='-b 2048 -ub 2048'  21:45:33  clk=600 MHz  MemAvail=31254236 kB
<!--PRED 2	dqc2048	memavail_kb=31254236	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6949,6799,6287,5963,3984,4708,4321,3829,3049,2277,4604-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |     2048 |          pp2048 |        104.40 ± 0.62 |

build: 171974745 (10558)
<!--DATA 2	dqc2048	pp2048	104.40-->
    [dq-cache] 150 weights held (948MB): 450 dequants skipped, 150 one-time fills, 0 still-streaming calls

### qwen35-08b-dqc [stock] pass 2  env=''  args=''  21:47:16  clk=600 MHz  MemAvail=31311108 kB
<!--PRED 2	stock	memavail_kb=31311108	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7335,6410,6632,5908,4864,4244,4158,3849,3126,2271,4608-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |          pp2048 |         65.71 ± 0.69 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	65.71-->

### qwen35-08b-dqc [dqc512] pass 3  env='ROCKET_DEQUANT_CACHE_MB=1536'  args=''  21:49:40  clk=600 MHz  MemAvail=31234064 kB
<!--PRED 3	dqc512	memavail_kb=31234064	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6145,6975,6723,5444,3911,4734,4312,3709,3064,2283,4611-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |          pp2048 |         84.72 ± 1.14 |

build: 171974745 (10558)
<!--DATA 3	dqc512	pp2048	84.72-->
    [dq-cache] 150 weights held (948MB): 2250 dequants skipped, 150 one-time fills, 0 still-streaming calls

### qwen35-08b-dqc [dqc2048] pass 3  env='ROCKET_DEQUANT_CACHE_MB=1536'  args='-b 2048 -ub 2048'  21:51:38  clk=600 MHz  MemAvail=31280844 kB
<!--PRED 3	dqc2048	memavail_kb=31280884	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6775,6021,6790,5744,4568,4880,4321,3767,3048,2256,4616-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |     2048 |          pp2048 |        104.88 ± 0.83 |

build: 171974745 (10558)
<!--DATA 3	dqc2048	pp2048	104.88-->
    [dq-cache] 150 weights held (948MB): 450 dequants skipped, 150 one-time fills, 0 still-streaming calls

### qwen35-08b-dqc [stock] pass 3  env=''  args=''  21:53:19  clk=600 MHz  MemAvail=31301636 kB
<!--PRED 3	stock	memavail_kb=31301636	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7103,6509,6472,5801,4705,4265,4179,3829,3114,2257,4620-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |          pp2048 |         65.76 ± 0.75 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	65.76-->

### qwen35-08b-dqc [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  21:55:44  clk=600 MHz  MemAvail=31295904 kB
<!--PRED 3	ub2048	memavail_kb=31295904	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7767,6187,6460,5980,4458,4273,4210,3673,3119,2288,4624-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |     2048 |          pp2048 |         97.53 ± 0.54 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	97.53-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 65.71 | 3 | 65.65 | 65.76 | -- | -- |
| pp2048 | ub2048 | 97.50 | 3 | 97.25 | 97.73 | 1.484x | 1.481 1.487 1.483 |
| pp2048 | dqc512 | 85.17 | 3 | 84.72 | 85.54 | 1.296x | 1.298 1.302 1.288 |
| pp2048 | dqc2048 | 104.65 | 3 | 104.40 | 104.88 | 1.593x | 1.595 1.589 1.595 |

