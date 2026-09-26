<!-- phi4mini-dqc  class=dqc  fp16-resident-fits=7168  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=<data>/phi4mini/Phi-4-mini-instruct-Q4_K_M.gguf (2491874272 bytes)
     note: fp16 image 6000 MB
-->
== phi4mini-dqc  Mon Aug 31 13:50:59 UTC 2026 ==
### phi4mini-dqc [stock] pass 1  env=''  args=''  13:51:28  clk=600 MHz  MemAvail=31587068 kB
<!--PRED 1	stock	memavail_kb=31587372	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5058,6410,6393,5673,4706,3503,3144,2768,2269,1570,5010-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         35.86 ± 0.25 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	35.86-->

### phi4mini-dqc [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  13:55:47  clk=600 MHz  MemAvail=31586316 kB
<!--PRED 1	ub2048	memavail_kb=31586336	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5613,6332,6069,5530,4599,3541,3177,2769,2258,1571,5013-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |     2048 |          pp2048 |         40.50 ± 0.11 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	40.50-->

### phi4mini-dqc [dqc512] pass 1  env='ROCKET_DEQUANT_CACHE_MB=7168'  args=''  13:59:40  clk=600 MHz  MemAvail=31413044 kB
<!--PRED 1	dqc512	memavail_kb=31413220	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4745,6231,5700,5442,3567,2982,3161,2794,2275,1591,4991-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         49.77 ± 0.14 |

build: 171974745 (10558)
<!--DATA 1	dqc512	pp2048	49.77-->
    [dq-cache] 126 weights held (6000MB): 1890 dequants skipped, 126 one-time fills, 0 still-streaming calls

### phi4mini-dqc [dqc2048] pass 1  env='ROCKET_DEQUANT_CACHE_MB=7168'  args='-b 2048 -ub 2048'  14:03:01  clk=600 MHz  MemAvail=31292224 kB
<!--PRED 1	dqc2048	memavail_kb=31292224	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5266,6394,5340,5065,3262,3089,3167,2784,2268,1559,4985-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |     2048 |          pp2048 |         42.54 ± 0.11 |

build: 171974745 (10558)
<!--DATA 1	dqc2048	pp2048	42.54-->
    [dq-cache] 126 weights held (6000MB): 378 dequants skipped, 126 one-time fills, 0 still-streaming calls

### phi4mini-dqc [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  14:06:52  clk=600 MHz  MemAvail=31567712 kB
<!--PRED 2	ub2048	memavail_kb=31567712	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4439,6412,5489,5372,4674,3559,3162,2781,2271,1598,4994-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |     2048 |          pp2048 |         39.70 ± 0.07 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	39.70-->

### phi4mini-dqc [dqc512] pass 2  env='ROCKET_DEQUANT_CACHE_MB=7168'  args=''  14:10:50  clk=600 MHz  MemAvail=31370192 kB
<!--PRED 2	dqc512	memavail_kb=31370192	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3805,6452,6123,5570,3327,3012,3138,2790,2268,1573,4994-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         49.15 ± 0.12 |

build: 171974745 (10558)
<!--DATA 2	dqc512	pp2048	49.15-->
    [dq-cache] 126 weights held (6000MB): 1890 dequants skipped, 126 one-time fills, 0 still-streaming calls

### phi4mini-dqc [dqc2048] pass 2  env='ROCKET_DEQUANT_CACHE_MB=7168'  args='-b 2048 -ub 2048'  14:14:13  clk=600 MHz  MemAvail=31427216 kB
<!--PRED 2	dqc2048	memavail_kb=31427280	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5338,5991,6224,5051,3980,3003,3156,2783,2273,1602,4985-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |     2048 |          pp2048 |         42.74 ± 0.05 |

build: 171974745 (10558)
<!--DATA 2	dqc2048	pp2048	42.74-->
    [dq-cache] 126 weights held (6000MB): 378 dequants skipped, 126 one-time fills, 0 still-streaming calls

### phi4mini-dqc [stock] pass 2  env=''  args=''  14:18:03  clk=600 MHz  MemAvail=31586048 kB
<!--PRED 2	stock	memavail_kb=31586048	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4602,6180,6087,5438,4681,3500,3140,2782,2267,1589,5004-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         36.16 ± 0.11 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	36.16-->

### phi4mini-dqc [dqc512] pass 3  env='ROCKET_DEQUANT_CACHE_MB=7168'  args=''  14:22:19  clk=600 MHz  MemAvail=31362616 kB
<!--PRED 3	dqc512	memavail_kb=31362616	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3682,6113,6192,4991,3197,2919,3160,2787,2270,1568,5003-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         49.44 ± 0.07 |

build: 171974745 (10558)
<!--DATA 3	dqc512	pp2048	49.44-->
    [dq-cache] 126 weights held (6000MB): 1890 dequants skipped, 126 one-time fills, 0 still-streaming calls

### phi4mini-dqc [dqc2048] pass 3  env='ROCKET_DEQUANT_CACHE_MB=7168'  args='-b 2048 -ub 2048'  14:25:41  clk=600 MHz  MemAvail=31390376 kB
<!--PRED 3	dqc2048	memavail_kb=31390412	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4145,5812,6368,5598,3532,3010,3169,2791,2272,1596,4980-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |     2048 |          pp2048 |         42.83 ± 0.03 |

build: 171974745 (10558)
<!--DATA 3	dqc2048	pp2048	42.83-->
    [dq-cache] 126 weights held (6000MB): 378 dequants skipped, 126 one-time fills, 0 still-streaming calls

### phi4mini-dqc [stock] pass 3  env=''  args=''  14:29:31  clk=600 MHz  MemAvail=31564652 kB
<!--PRED 3	stock	memavail_kb=31564652	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6228,6149,5754,5622,4451,3517,3131,2782,2268,1603,4993-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         36.07 ± 0.16 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	36.07-->

### phi4mini-dqc [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  14:33:49  clk=600 MHz  MemAvail=31580884 kB
<!--PRED 3	ub2048	memavail_kb=31580888	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3025,6267,6233,5677,4601,3484,3155,2789,2269,1576,5007-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |     2048 |          pp2048 |         39.47 ± 0.08 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	39.47-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 36.03 | 3 | 35.86 | 36.16 | -- | -- |
| pp2048 | ub2048 | 39.89 | 3 | 39.47 | 40.50 | 1.107x | 1.129 1.098 1.094 |
| pp2048 | dqc512 | 49.45 | 3 | 49.15 | 49.77 | 1.373x | 1.388 1.359 1.371 |
| pp2048 | dqc2048 | 42.70 | 3 | 42.54 | 42.83 | 1.185x | 1.186 1.182 1.187 |

