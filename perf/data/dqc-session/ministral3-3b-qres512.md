== ministral3-3b-qres512  Mon Aug 31 15:02:13 UTC 2026 ==
### ministral3-3b-qres512 [stock] pass 1  env=''  args=''  15:02:42  clk=600 MHz  MemAvail=31579420 kB
<!--PRED 1	stock	memavail_kb=31579724	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5446,6973,7344,6305,5103,4364,3875,3278,2366,1766,4816-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         34.36 ± 0.19 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	34.36-->

### ministral3-3b-qres512 [qres512] pass 1  env='ROCKET_QUANT_RESIDENT=auto'  args=''  15:07:15  clk=600 MHz  MemAvail=31382876 kB
<!--PRED 1	qres512	memavail_kb=31382876	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6211,6918,6708,6132,5075,4339,3831,3232,2420,1724,4788-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         50.08 ± 0.16 |

build: 171974745 (10558)
<!--DATA 1	qres512	pp2048	50.08-->
    [f16-resident] weights offered to the resident route: 179 resident on the NPU (5610MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21033MB (MemAvailable 30569MB - reserve 9535MB, no swap)

### ministral3-3b-qres512 [qres512] pass 2  env='ROCKET_QUANT_RESIDENT=auto'  args=''  15:10:47  clk=600 MHz  MemAvail=31208240 kB
<!--PRED 2	qres512	memavail_kb=31208128	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5613,6098,6261,5475,4825,4366,3895,3286,2514,1673,4749-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         50.48 ± 0.14 |

build: 171974745 (10558)
<!--DATA 2	qres512	pp2048	50.48-->
    [f16-resident] weights offered to the resident route: 179 resident on the NPU (5610MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20937MB (MemAvailable 30473MB - reserve 9535MB, no swap)

### ministral3-3b-qres512 [stock] pass 2  env=''  args=''  15:14:14  clk=600 MHz  MemAvail=31579044 kB
<!--PRED 2	stock	memavail_kb=31579044	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6101,6529,6623,6079,4895,4066,3671,3164,2456,1787,4827-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         34.51 ± 0.16 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	34.51-->

### ministral3-3b-qres512 [stock] pass 3  env=''  args=''  15:18:42  clk=600 MHz  MemAvail=31592100 kB
<!--PRED 3	stock	memavail_kb=31592352	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3795,6684,6512,5976,4558,3842,3465,3009,2360,1761,4915-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         34.92 ± 0.12 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	34.92-->

### ministral3-3b-qres512 [qres512] pass 3  env='ROCKET_QUANT_RESIDENT=auto'  args=''  15:23:11  clk=600 MHz  MemAvail=31395560 kB
<!--PRED 3	qres512	memavail_kb=31395560	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7266,6738,6518,5843,4695,4085,3694,3143,2401,1722,4833-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         49.43 ± 0.24 |

build: 171974745 (10558)
<!--DATA 3	qres512	pp2048	49.43-->
    [f16-resident] weights offered to the resident route: 179 resident on the NPU (5610MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21115MB (MemAvailable 30650MB - reserve 9535MB, no swap)

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 34.60 | 3 | 34.36 | 34.92 | -- | -- |
| pp2048 | qres512 | 50.00 | 3 | 49.43 | 50.48 | 1.445x | 1.458 1.463 1.416 |

