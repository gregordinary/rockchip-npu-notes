== phi4mini-qres512  Mon Aug 31 15:26:12 UTC 2026 ==
### phi4mini-qres512 [stock] pass 1  env=''  args=''  15:26:41  clk=600 MHz  MemAvail=31561884 kB
<!--PRED 1	stock	memavail_kb=31562292	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5060,6390,6662,5940,5008,4213,3719,3200,2428,1647,4806-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         35.99 ± 0.06 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	35.99-->

### phi4mini-qres512 [qres512] pass 1  env='ROCKET_QUANT_RESIDENT=auto'  args=''  15:31:05  clk=600 MHz  MemAvail=31092500 kB
<!--PRED 1	qres512	memavail_kb=31092728	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5697,6112,6196,5299,4687,4075,3712,3193,2447,1447,4804-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         54.40 ± 0.22 |

build: 171974745 (10558)
<!--DATA 1	qres512	pp2048	54.40-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (6000MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20813MB (MemAvailable 30349MB - reserve 9535MB, no swap)

### phi4mini-qres512 [qres512] pass 2  env='ROCKET_QUANT_RESIDENT=auto'  args=''  15:34:26  clk=600 MHz  MemAvail=31310704 kB
<!--PRED 2	qres512	memavail_kb=31310956	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5835,6364,6450,5666,5032,4555,4005,3361,2481,1572,4722-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         54.37 ± 0.05 |

build: 171974745 (10558)
<!--DATA 2	qres512	pp2048	54.37-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (6000MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21056MB (MemAvailable 30592MB - reserve 9535MB, no swap)

### phi4mini-qres512 [stock] pass 2  env=''  args=''  15:37:42  clk=600 MHz  MemAvail=31594580 kB
<!--PRED 2	stock	memavail_kb=31594580	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6337,6912,6746,6089,5142,4320,3844,3248,2452,1683,4766-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         35.77 ± 0.06 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	35.77-->

### phi4mini-qres512 [stock] pass 3  env=''  args=''  15:42:02  clk=600 MHz  MemAvail=31601176 kB
<!--PRED 3	stock	memavail_kb=31601176	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6398,6558,6084,5799,4763,3838,3496,3013,2367,1646,4885-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         36.06 ± 0.07 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	36.06-->

### phi4mini-qres512 [qres512] pass 3  env='ROCKET_QUANT_RESIDENT=auto'  args=''  15:46:25  clk=600 MHz  MemAvail=31202432 kB
<!--PRED 3	qres512	memavail_kb=31202432	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6007,6828,6283,5755,4622,3976,3669,3133,2435,1492,4820-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         53.78 ± 0.16 |

build: 171974745 (10558)
<!--DATA 3	qres512	pp2048	53.78-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (6000MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20932MB (MemAvailable 30468MB - reserve 9535MB, no swap)

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 35.94 | 3 | 35.77 | 36.06 | -- | -- |
| pp2048 | qres512 | 54.18 | 3 | 53.78 | 54.40 | 1.508x | 1.512 1.520 1.491 |

