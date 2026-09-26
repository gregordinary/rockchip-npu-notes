== llama32-3b-qres512  Mon Aug 31 14:40:59 UTC 2026 ==
### llama32-3b-qres512 [stock] pass 1  env=''  args=''  14:41:27  clk=600 MHz  MemAvail=31549388 kB
<!--PRED 1	stock	memavail_kb=31549872	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4394,6276,6006,5599,3745,3606,3156,2802,2264,1822,4998-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         39.84 ± 0.05 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	39.84-->

### llama32-3b-qres512 [qres512] pass 1  env='ROCKET_QUANT_RESIDENT=auto'  args=''  14:45:25  clk=600 MHz  MemAvail=31249264 kB
<!--PRED 1	qres512	memavail_kb=31249264	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4980,6043,5951,5273,4497,3929,3556,3081,2338,1730,4874-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         57.77 ± 0.35 |

build: 171974745 (10558)
<!--DATA 1	qres512	pp2048	57.77-->
    [f16-resident] weights offered to the resident route: 193 resident on the NPU (5232MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20974MB (MemAvailable 30510MB - reserve 9535MB, no swap)

### llama32-3b-qres512 [qres512] pass 2  env='ROCKET_QUANT_RESIDENT=auto'  args=''  14:48:31  clk=600 MHz  MemAvail=31331588 kB
<!--PRED 2	qres512	memavail_kb=31331652	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5242,7412,7633,6480,5752,4921,4346,3538,2394,1680,4729-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         57.73 ± 0.11 |

build: 171974745 (10558)
<!--DATA 2	qres512	pp2048	57.73-->
    [f16-resident] weights offered to the resident route: 193 resident on the NPU (5232MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21061MB (MemAvailable 30597MB - reserve 9535MB, no swap)

### llama32-3b-qres512 [stock] pass 2  env=''  args=''  14:51:34  clk=600 MHz  MemAvail=31539600 kB
<!--PRED 2	stock	memavail_kb=31539796	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5420,7643,7019,6833,5369,4757,4130,3387,2399,1773,4774-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         39.11 ± 0.27 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	39.11-->

### llama32-3b-qres512 [stock] pass 3  env=''  args=''  14:55:32  clk=600 MHz  MemAvail=31559916 kB
<!--PRED 3	stock	memavail_kb=31560040	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6090,6454,5958,5702,4759,4063,3586,3097,2351,1793,4897-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         38.66 ± 0.14 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	38.66-->

### llama32-3b-qres512 [qres512] pass 3  env='ROCKET_QUANT_RESIDENT=auto'  args=''  14:59:37  clk=600 MHz  MemAvail=31250408 kB
<!--PRED 3	qres512	memavail_kb=31250408	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4896,6665,6384,5606,4814,4224,3858,3266,2384,1681,4825-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         57.67 ± 0.33 |

build: 171974745 (10558)
<!--DATA 3	qres512	pp2048	57.67-->
    [f16-resident] weights offered to the resident route: 193 resident on the NPU (5232MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20995MB (MemAvailable 30530MB - reserve 9535MB, no swap)

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 39.20 | 3 | 38.66 | 39.84 | -- | -- |
| pp2048 | qres512 | 57.72 | 3 | 57.67 | 57.77 | 1.473x | 1.450 1.476 1.492 |

