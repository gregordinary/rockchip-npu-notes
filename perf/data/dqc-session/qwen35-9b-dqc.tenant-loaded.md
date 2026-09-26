<!-- qwen35-9b-dqc  class=dqc  fp16-resident-fits=15360  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=<data>/qwen35/Qwen3.5-9B-Q4_K_M.gguf (5680522464 bytes)
     note: control on the largest measured -ub lever (1.424x); fp16 image 13184 MB
-->
== qwen35-9b-dqc  Sun Aug 30 21:59:00 UTC 2026 ==
### qwen35-9b-dqc [stock] pass 1  env=''  args=''  22:00:22  clk=600 MHz  MemAvail=31323656 kB
<!--PRED 1	stock	memavail_kb=31323716	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7323,6764,6337,6151,4477,3817,3498,3101,2461,827,4257-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         15.13 ± 0.14 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	15.13-->

### qwen35-9b-dqc [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  22:10:50  clk=600 MHz  MemAvail=31334156 kB
<!--PRED 1	ub2048	memavail_kb=31333852	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7010,6447,6869,6158,4390,3768,3451,3084,2446,835,4321-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         23.66 ± 0.20 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	23.66-->

### qwen35-9b-dqc [dqc512] pass 1  env='ROCKET_DEQUANT_CACHE_MB=15360'  args=''  22:17:54  clk=600 MHz  MemAvail=31335496 kB
<!--PRED 1	dqc512	memavail_kb=31335496	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6910,6507,6266,6049,4848,3826,3482,3141,2522,910,4250-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         24.58 ± 0.09 |

build: 171974745 (10558)
<!--DATA 1	dqc512	pp2048	24.58-->
    [dq-cache] 200 weights held (13184MB): 3000 dequants skipped, 200 one-time fills, 0 still-streaming calls

### qwen35-9b-dqc [dqc2048] pass 1  env='ROCKET_DEQUANT_CACHE_MB=15360'  args='-b 2048 -ub 2048'  22:25:02  clk=600 MHz  MemAvail=31333336 kB
<!--PRED 1	dqc2048	memavail_kb=31333044	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7165,6479,6467,5980,4776,3743,3431,3109,2482,866,4292-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.40 ± 0.15 |

build: 171974745 (10558)
<!--DATA 1	dqc2048	pp2048	28.40-->
    [dq-cache] 200 weights held (13184MB): 600 dequants skipped, 200 one-time fills, 0 still-streaming calls

### qwen35-9b-dqc [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  22:31:37  clk=600 MHz  MemAvail=31336552 kB
<!--PRED 2	ub2048	memavail_kb=31336552	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6458,6754,6139,6135,4697,3800,3433,3032,2429,830,4333-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         23.78 ± 0.10 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	23.78-->

### qwen35-9b-dqc [dqc512] pass 2  env='ROCKET_DEQUANT_CACHE_MB=15360'  args=''  22:38:40  clk=600 MHz  MemAvail=31341092 kB
<!--PRED 2	dqc512	memavail_kb=31341244	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6543,6419,6476,6049,4767,3813,3444,3081,2465,860,4302-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         25.26 ± 0.66 |

build: 171974745 (10558)
<!--DATA 2	dqc512	pp2048	25.26-->
    [dq-cache] 200 weights held (13184MB): 3000 dequants skipped, 200 one-time fills, 0 still-streaming calls

### qwen35-9b-dqc [dqc2048] pass 2  env='ROCKET_DEQUANT_CACHE_MB=15360'  args='-b 2048 -ub 2048'  22:45:43  clk=600 MHz  MemAvail=31457028 kB
<!--PRED 2	dqc2048	memavail_kb=31456736	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6621,6373,6372,6026,4981,3917,3374,3118,2519,907,4285-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.14 ± 0.31 |

build: 171974745 (10558)
<!--DATA 2	dqc2048	pp2048	28.14-->
    [dq-cache] 200 weights held (13184MB): 600 dequants skipped, 200 one-time fills, 0 still-streaming calls

### qwen35-9b-dqc [stock] pass 2  env=''  args=''  22:52:16  clk=600 MHz  MemAvail=31403856 kB
<!--PRED 2	stock	memavail_kb=31403704	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9521,8981,8127,6878,4694,3786,3377,2955,2453,868,4307-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         15.91 ± 0.04 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	15.91-->

### qwen35-9b-dqc [dqc512] pass 3  env='ROCKET_DEQUANT_CACHE_MB=15360'  args=''  23:02:09  clk=600 MHz  MemAvail=31392600 kB
<!--PRED 3	dqc512	memavail_kb=31395480	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10743,9406,8828,7754,5640,4297,3768,3365,2740,1101,3999-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         24.52 ± 0.11 |

build: 171974745 (10558)
<!--DATA 3	dqc512	pp2048	24.52-->
    [dq-cache] 200 weights held (13184MB): 3000 dequants skipped, 200 one-time fills, 0 still-streaming calls

### qwen35-9b-dqc [dqc2048] pass 3  env='ROCKET_DEQUANT_CACHE_MB=15360'  args='-b 2048 -ub 2048'  23:09:17  clk=600 MHz  MemAvail=31443724 kB
<!--PRED 3	dqc2048	memavail_kb=31444180	anonhuge_kb=0	hugepagesz_kb=2048	buddy=11048,9563,8885,7834,5795,4212,3679,3354,2723,1072,4035-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         27.84 ± 0.02 |

build: 171974745 (10558)
<!--DATA 3	dqc2048	pp2048	27.84-->
    [dq-cache] 200 weights held (13184MB): 600 dequants skipped, 200 one-time fills, 0 still-streaming calls

### qwen35-9b-dqc [stock] pass 3  env=''  args=''  23:15:52  clk=600 MHz  MemAvail=31413516 kB
<!--PRED 3	stock	memavail_kb=31413516	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6762,8049,8357,7214,4637,3869,3432,2893,2456,868,4312-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         16.01 ± 0.09 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	16.01-->

### qwen35-9b-dqc [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  23:25:47  clk=600 MHz  MemAvail=31427304 kB
<!--PRED 3	ub2048	memavail_kb=31427304	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8444,10674,9354,6721,4972,3720,3314,2989,2441,857,4313-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         24.35 ± 0.19 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	24.35-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 15.68 | 3 | 15.13 | 16.01 | -- | -- |
| pp2048 | ub2048 | 23.93 | 3 | 23.66 | 24.35 | 1.526x | 1.564 1.495 1.521 |
| pp2048 | dqc512 | 24.79 | 3 | 24.52 | 25.26 | 1.581x | 1.625 1.588 1.532 |
| pp2048 | dqc2048 | 28.13 | 3 | 27.84 | 28.40 | 1.795x | 1.877 1.769 1.739 |

