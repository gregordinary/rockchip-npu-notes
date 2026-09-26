<!-- ministral3-3b-dqc  class=dqc  fp16-resident-fits=6656  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=<data>/ministral3-3b/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf (2146497824 bytes)
     note: fp16 image 5610 MB
-->
== ministral3-3b-dqc  Mon Aug 31 13:01:47 UTC 2026 ==
### ministral3-3b-dqc [stock] pass 1  env=''  args=''  13:02:16  clk=600 MHz  MemAvail=31579484 kB
<!--PRED 1	stock	memavail_kb=31579612	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4191,6092,6288,5765,4292,3536,3155,2775,2260,1723,5022-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         34.72 ± 0.19 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	34.72-->

### ministral3-3b-dqc [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  13:06:42  clk=600 MHz  MemAvail=31593004 kB
<!--PRED 1	ub2048	memavail_kb=31593132	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5531,6580,6338,5795,4347,3537,3172,2774,2259,1719,5023-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |     2048 |          pp2048 |         35.93 ± 0.09 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	35.93-->

### ministral3-3b-dqc [dqc512] pass 1  env='ROCKET_DEQUANT_CACHE_MB=6656'  args=''  13:11:00  clk=600 MHz  MemAvail=31325096 kB
<!--PRED 1	dqc512	memavail_kb=31325096	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5467,6435,5938,5146,3023,3026,3162,2778,2267,1712,5002-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         46.25 ± 0.13 |

build: 171974745 (10558)
<!--DATA 1	dqc512	pp2048	46.25-->
    [dq-cache] 179 weights held (5610MB): 2685 dequants skipped, 179 one-time fills, 0 still-streaming calls

### ministral3-3b-dqc [dqc2048] pass 1  env='ROCKET_DEQUANT_CACHE_MB=6656'  args='-b 2048 -ub 2048'  13:14:33  clk=600 MHz  MemAvail=31397640 kB
<!--PRED 1	dqc2048	memavail_kb=31397640	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4725,6445,6311,5712,3268,3028,3182,2803,2281,1757,4981-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |     2048 |          pp2048 |         38.69 ± 0.10 |

build: 171974745 (10558)
<!--DATA 1	dqc2048	pp2048	38.69-->
    [dq-cache] 179 weights held (5610MB): 537 dequants skipped, 179 one-time fills, 0 still-streaming calls

### ministral3-3b-dqc [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  13:18:42  clk=600 MHz  MemAvail=31591148 kB
<!--PRED 2	ub2048	memavail_kb=31591148	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5985,6459,6376,5818,4232,3572,3140,2766,2264,1810,4979-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |     2048 |          pp2048 |         36.10 ± 0.14 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	36.10-->

### ministral3-3b-dqc [dqc512] pass 2  env='ROCKET_DEQUANT_CACHE_MB=6656'  args=''  13:22:59  clk=600 MHz  MemAvail=31309952 kB
<!--PRED 2	dqc512	memavail_kb=31309952	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6323,6264,5860,5442,3520,3170,3201,2826,2330,1717,4957-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         46.70 ± 0.21 |

build: 171974745 (10558)
<!--DATA 2	dqc512	pp2048	46.70-->
    [dq-cache] 179 weights held (5610MB): 2685 dequants skipped, 179 one-time fills, 0 still-streaming calls

### ministral3-3b-dqc [dqc2048] pass 2  env='ROCKET_DEQUANT_CACHE_MB=6656'  args='-b 2048 -ub 2048'  13:26:30  clk=600 MHz  MemAvail=31324532 kB
<!--PRED 2	dqc2048	memavail_kb=31324676	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5010,6225,6021,5462,3899,2998,3163,2775,2270,1723,4982-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |     2048 |          pp2048 |         39.00 ± 0.02 |

build: 171974745 (10558)
<!--DATA 2	dqc2048	pp2048	39.00-->
    [dq-cache] 179 weights held (5610MB): 537 dequants skipped, 179 one-time fills, 0 still-streaming calls

### ministral3-3b-dqc [stock] pass 2  env=''  args=''  13:30:38  clk=600 MHz  MemAvail=31596412 kB
<!--PRED 2	stock	memavail_kb=31596644	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6040,6258,6334,5680,4418,3594,3162,2782,2270,1815,4971-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         34.77 ± 0.22 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	34.77-->

### ministral3-3b-dqc [dqc512] pass 3  env='ROCKET_DEQUANT_CACHE_MB=6656'  args=''  13:35:02  clk=600 MHz  MemAvail=31501300 kB
<!--PRED 3	dqc512	memavail_kb=31501456	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5982,6244,6151,5655,4442,3385,3189,2793,2279,1770,4972-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         46.47 ± 0.19 |

build: 171974745 (10558)
<!--DATA 3	dqc512	pp2048	46.47-->
    [dq-cache] 179 weights held (5610MB): 2685 dequants skipped, 179 one-time fills, 0 still-streaming calls

### ministral3-3b-dqc [dqc2048] pass 3  env='ROCKET_DEQUANT_CACHE_MB=6656'  args='-b 2048 -ub 2048'  13:38:34  clk=600 MHz  MemAvail=31297392 kB
<!--PRED 3	dqc2048	memavail_kb=31297392	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5594,6251,6026,5419,3412,2911,3182,2790,2271,1757,4965-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |     2048 |          pp2048 |         38.76 ± 0.04 |

build: 171974745 (10558)
<!--DATA 3	dqc2048	pp2048	38.76-->
    [dq-cache] 179 weights held (5610MB): 537 dequants skipped, 179 one-time fills, 0 still-streaming calls

### ministral3-3b-dqc [stock] pass 3  env=''  args=''  13:42:43  clk=600 MHz  MemAvail=31590256 kB
<!--PRED 3	stock	memavail_kb=31590256	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5962,6670,6354,5703,4338,3591,3137,2773,2268,1787,4987-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         34.72 ± 0.11 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	34.72-->

### ministral3-3b-dqc [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  13:47:08  clk=600 MHz  MemAvail=31583284 kB
<!--PRED 3	ub2048	memavail_kb=31583284	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3245,6538,6414,5429,4617,3505,3157,2783,2264,1721,5019-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |     2048 |          pp2048 |         35.84 ± 0.21 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	35.84-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 34.74 | 3 | 34.72 | 34.77 | -- | -- |
| pp2048 | ub2048 | 35.96 | 3 | 35.84 | 36.10 | 1.035x | 1.035 1.038 1.032 |
| pp2048 | dqc512 | 46.47 | 3 | 46.25 | 46.70 | 1.338x | 1.332 1.343 1.338 |
| pp2048 | dqc2048 | 38.82 | 3 | 38.69 | 39.00 | 1.117x | 1.114 1.122 1.116 |

