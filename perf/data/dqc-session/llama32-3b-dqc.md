<!-- llama32-3b-dqc  class=dqc  fp16-resident-fits=6144  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=<data>/llama32-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf (2019377600 bytes)
     note: does the sub-4B class carry a negative residue? fp16 image 5232 MB
-->
== llama32-3b-dqc  Mon Aug 31 12:19:04 UTC 2026 ==
### llama32-3b-dqc [stock] pass 1  env=''  args=''  12:19:30  clk=600 MHz  MemAvail=31554652 kB
<!--PRED 1	stock	memavail_kb=31555184	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3394,5954,6083,5445,4248,3758,3191,2750,2265,1791,5010-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         39.19 ± 0.05 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	39.19-->

### llama32-3b-dqc [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  12:23:28  clk=600 MHz  MemAvail=31583708 kB
<!--PRED 1	ub2048	memavail_kb=31583708	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3770,6340,6371,5767,4218,3680,3196,2756,2268,1778,5019-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |     2048 |          pp2048 |         42.52 ± 0.06 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	42.52-->

### llama32-3b-dqc [dqc512] pass 1  env='ROCKET_DEQUANT_CACHE_MB=6144'  args=''  12:27:08  clk=600 MHz  MemAvail=31394412 kB
<!--PRED 1	dqc512	memavail_kb=31394412	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3534,6689,6409,5749,1931,3605,3220,2760,2270,1764,5016-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         54.88 ± 0.17 |

build: 171974745 (10558)
<!--DATA 1	dqc512	pp2048	54.88-->
    [dq-cache] 193 weights held (5232MB): 2895 dequants skipped, 193 one-time fills, 0 still-streaming calls

### llama32-3b-dqc [dqc2048] pass 1  env='ROCKET_DEQUANT_CACHE_MB=6144'  args='-b 2048 -ub 2048'  12:30:10  clk=600 MHz  MemAvail=31447936 kB
<!--PRED 1	dqc2048	memavail_kb=31447936	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4456,6175,6117,5736,3896,3704,3202,2765,2268,1761,4999-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |     2048 |          pp2048 |         46.43 ± 0.04 |

build: 171974745 (10558)
<!--DATA 1	dqc2048	pp2048	46.43-->
    [dq-cache] 193 weights held (5232MB): 579 dequants skipped, 193 one-time fills, 0 still-streaming calls

### llama32-3b-dqc [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  12:33:41  clk=600 MHz  MemAvail=31566328 kB
<!--PRED 2	ub2048	memavail_kb=31566328	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4899,6107,6210,5417,4177,3745,3179,2778,2267,1815,4997-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |     2048 |          pp2048 |         42.41 ± 0.17 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	42.41-->

### llama32-3b-dqc [dqc512] pass 2  env='ROCKET_DEQUANT_CACHE_MB=6144'  args=''  12:37:22  clk=600 MHz  MemAvail=31443388 kB
<!--PRED 2	dqc512	memavail_kb=31443388	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4762,6511,6437,5705,2692,3685,3170,2779,2268,1788,5002-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         52.79 ± 0.22 |

build: 171974745 (10558)
<!--DATA 2	dqc512	pp2048	52.79-->
    [dq-cache] 193 weights held (5232MB): 2895 dequants skipped, 193 one-time fills, 0 still-streaming calls

### llama32-3b-dqc [dqc2048] pass 2  env='ROCKET_DEQUANT_CACHE_MB=6144'  args='-b 2048 -ub 2048'  12:40:30  clk=600 MHz  MemAvail=31366408 kB
<!--PRED 2	dqc2048	memavail_kb=31366936	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4848,6321,5981,4562,1699,3622,3204,2784,2280,1838,4981-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |     2048 |          pp2048 |         46.40 ± 0.16 |

build: 171974745 (10558)
<!--DATA 2	dqc2048	pp2048	46.40-->
    [dq-cache] 193 weights held (5232MB): 579 dequants skipped, 193 one-time fills, 0 still-streaming calls

### llama32-3b-dqc [stock] pass 2  env=''  args=''  12:44:01  clk=600 MHz  MemAvail=31588440 kB
<!--PRED 2	stock	memavail_kb=31588440	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6316,6604,5974,5799,4056,3708,3184,2779,2263,1860,4979-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         38.95 ± 0.12 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	38.95-->

### llama32-3b-dqc [dqc512] pass 3  env='ROCKET_DEQUANT_CACHE_MB=6144'  args=''  12:47:59  clk=600 MHz  MemAvail=31337584 kB
<!--PRED 3	dqc512	memavail_kb=31337584	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5787,6319,5874,5816,2256,3588,3194,2777,2269,1796,4981-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         53.43 ± 0.09 |

build: 171974745 (10558)
<!--DATA 3	dqc512	pp2048	53.43-->
    [dq-cache] 193 weights held (5232MB): 2895 dequants skipped, 193 one-time fills, 0 still-streaming calls

### llama32-3b-dqc [dqc2048] pass 3  env='ROCKET_DEQUANT_CACHE_MB=6144'  args='-b 2048 -ub 2048'  12:51:06  clk=600 MHz  MemAvail=31486248 kB
<!--PRED 3	dqc2048	memavail_kb=31486328	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4738,6480,5903,5847,3071,3680,3238,2801,2283,1834,4973-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |     2048 |          pp2048 |         46.16 ± 0.03 |

build: 171974745 (10558)
<!--DATA 3	dqc2048	pp2048	46.16-->
    [dq-cache] 193 weights held (5232MB): 579 dequants skipped, 193 one-time fills, 0 still-streaming calls

### llama32-3b-dqc [stock] pass 3  env=''  args=''  12:54:38  clk=600 MHz  MemAvail=31594580 kB
<!--PRED 3	stock	memavail_kb=31594580	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4005,6362,6220,5673,4475,3673,3155,2770,2264,1821,5000-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         39.51 ± 0.09 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	39.51-->

### llama32-3b-dqc [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  12:58:34  clk=600 MHz  MemAvail=31575912 kB
<!--PRED 3	ub2048	memavail_kb=31575912	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5795,6300,6161,5562,4207,3606,3178,2775,2265,1779,5020-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |     2048 |          pp2048 |         42.76 ± 0.10 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	42.76-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 39.22 | 3 | 38.95 | 39.51 | -- | -- |
| pp2048 | ub2048 | 42.56 | 3 | 42.41 | 42.76 | 1.085x | 1.085 1.089 1.082 |
| pp2048 | dqc512 | 53.70 | 3 | 52.79 | 54.88 | 1.369x | 1.400 1.355 1.352 |
| pp2048 | dqc2048 | 46.33 | 3 | 46.16 | 46.43 | 1.181x | 1.185 1.191 1.168 |

