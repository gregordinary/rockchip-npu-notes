<!-- qwen35-9b-dqc  class=dqc  fp16-resident-fits=15360  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=<data>/qwen35/Qwen3.5-9B-Q4_K_M.gguf (5680522464 bytes)
     note: control on the largest measured -ub lever (1.424x); fp16 image 13184 MB
-->
== qwen35-9b-dqc  Mon Aug 31 00:47:32 UTC 2026 ==
### qwen35-9b-dqc [stock] pass 1  env=''  args=''  00:48:37  clk=600 MHz  MemAvail=31606648 kB
<!--PRED 1	stock	memavail_kb=31606648	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6966,6642,5989,5705,4838,3709,3318,2960,2376,866,4540-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         19.04 ± 0.10 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	19.04-->

### qwen35-9b-dqc [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  00:56:54  clk=600 MHz  MemAvail=31602676 kB
<!--PRED 1	ub2048	memavail_kb=31602896	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6024,6423,6134,5774,4997,3693,3274,2948,2381,863,4542-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         27.10 ± 0.04 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	27.10-->

### qwen35-9b-dqc [dqc512] pass 1  env='ROCKET_DEQUANT_CACHE_MB=15360'  args=''  01:03:01  clk=600 MHz  MemAvail=31576296 kB
<!--PRED 1	dqc512	memavail_kb=31576460	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5926,6801,6239,5798,4970,3670,3318,2929,2403,876,4523-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         29.61 ± 0.22 |

build: 171974745 (10558)
<!--DATA 1	dqc512	pp2048	29.61-->
    [dq-cache] 200 weights held (13184MB): 3000 dequants skipped, 200 one-time fills, 0 still-streaming calls

### qwen35-9b-dqc [dqc2048] pass 1  env='ROCKET_DEQUANT_CACHE_MB=15360'  args='-b 2048 -ub 2048'  01:08:55  clk=600 MHz  MemAvail=31594936 kB
<!--PRED 1	dqc2048	memavail_kb=31594984	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6330,7003,6330,5915,4997,3661,3310,2948,2388,866,4532-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         30.52 ± 0.09 |

build: 171974745 (10558)
<!--DATA 1	dqc2048	pp2048	30.52-->
    [dq-cache] 200 weights held (13184MB): 600 dequants skipped, 200 one-time fills, 0 still-streaming calls

### qwen35-9b-dqc [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  01:14:45  clk=600 MHz  MemAvail=31563112 kB
<!--PRED 2	ub2048	memavail_kb=31563112	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5921,6585,6072,5707,4926,3775,3342,2912,2385,862,4531-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         27.18 ± 0.13 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	27.18-->

### qwen35-9b-dqc [dqc512] pass 2  env='ROCKET_DEQUANT_CACHE_MB=15360'  args=''  01:20:50  clk=600 MHz  MemAvail=31549100 kB
<!--PRED 2	dqc512	memavail_kb=31549100	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5900,6512,6232,5630,5008,3695,3365,2915,2405,880,4513-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         29.59 ± 0.07 |

build: 171974745 (10558)
<!--DATA 2	dqc512	pp2048	29.59-->
    [dq-cache] 200 weights held (13184MB): 3000 dequants skipped, 200 one-time fills, 0 still-streaming calls

### qwen35-9b-dqc [dqc2048] pass 2  env='ROCKET_DEQUANT_CACHE_MB=15360'  args='-b 2048 -ub 2048'  01:26:45  clk=600 MHz  MemAvail=31574780 kB
<!--PRED 2	dqc2048	memavail_kb=31574780	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5125,6622,6212,5776,5078,3783,3312,2941,2397,884,4515-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         30.73 ± 0.18 |

build: 171974745 (10558)
<!--DATA 2	dqc2048	pp2048	30.73-->
    [dq-cache] 200 weights held (13184MB): 600 dequants skipped, 200 one-time fills, 0 still-streaming calls

### qwen35-9b-dqc [stock] pass 2  env=''  args=''  01:32:34  clk=600 MHz  MemAvail=31582420 kB
<!--PRED 2	stock	memavail_kb=31582420	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5945,6763,6399,5801,5075,3771,3315,2908,2384,867,4531-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         19.11 ± 0.18 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	19.11-->

### qwen35-9b-dqc [dqc512] pass 3  env='ROCKET_DEQUANT_CACHE_MB=15360'  args=''  01:40:46  clk=600 MHz  MemAvail=31531352 kB
<!--PRED 3	dqc512	memavail_kb=31531352	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5062,6819,6497,5887,4826,3737,3285,2938,2393,885,4510-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         29.54 ± 0.09 |

build: 171974745 (10558)
<!--DATA 3	dqc512	pp2048	29.54-->
    [dq-cache] 200 weights held (13184MB): 3000 dequants skipped, 200 one-time fills, 0 still-streaming calls

### qwen35-9b-dqc [dqc2048] pass 3  env='ROCKET_DEQUANT_CACHE_MB=15360'  args='-b 2048 -ub 2048'  01:46:40  clk=600 MHz  MemAvail=31552872 kB
<!--PRED 3	dqc2048	memavail_kb=31552872	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5508,6954,6297,5809,5149,3663,3262,2944,2388,875,4520-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         30.84 ± 0.06 |

build: 171974745 (10558)
<!--DATA 3	dqc2048	pp2048	30.84-->
    [dq-cache] 200 weights held (13184MB): 600 dequants skipped, 200 one-time fills, 0 still-streaming calls

### qwen35-9b-dqc [stock] pass 3  env=''  args=''  01:52:28  clk=600 MHz  MemAvail=31554084 kB
<!--PRED 3	stock	memavail_kb=31554084	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6112,6639,6329,5734,5079,3673,3287,2913,2378,862,4533-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         19.03 ± 0.10 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	19.03-->

### qwen35-9b-dqc [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  02:00:46  clk=600 MHz  MemAvail=31600460 kB
<!--PRED 3	ub2048	memavail_kb=31600460	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6697,6869,6386,5906,5088,3770,3253,2915,2376,860,4541-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         27.50 ± 0.09 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	27.50-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 19.06 | 3 | 19.03 | 19.11 | -- | -- |
| pp2048 | ub2048 | 27.26 | 3 | 27.10 | 27.50 | 1.430x | 1.423 1.422 1.445 |
| pp2048 | dqc512 | 29.58 | 3 | 29.54 | 29.61 | 1.552x | 1.555 1.548 1.552 |
| pp2048 | dqc2048 | 30.70 | 3 | 30.52 | 30.84 | 1.611x | 1.603 1.608 1.621 |

