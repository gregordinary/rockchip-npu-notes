== smolvlm2-qres512  Mon Aug 31 02:43:38 UTC 2026 ==
### smolvlm2-qres512 [stock] pass 1  env=''  args=''  02:43:53  clk=600 MHz  MemAvail=31517920 kB
<!--PRED 1	stock	memavail_kb=31518392	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3708,5926,6218,4773,3544,3760,3283,2916,2802,2160,4892-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |          pp2048 |         51.76 ± 0.16 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	51.76-->

### smolvlm2-qres512 [qres512] pass 1  env='ROCKET_QUANT_RESIDENT=auto'  args=''  02:46:51  clk=600 MHz  MemAvail=31456000 kB
<!--PRED 1	qres512	memavail_kb=31456000	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5161,6647,6054,5363,4351,4091,3554,2991,2487,2167,4895-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |          pp2048 |         69.68 ± 0.31 |

build: 171974745 (10558)
<!--DATA 1	qres512	pp2048	69.68-->
    [f16-resident] weights offered to the resident route: 165 resident on the NPU (2976MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21181MB (MemAvailable 30717MB - reserve 9535MB, no swap)

### smolvlm2-qres512 [qres512] pass 2  env='ROCKET_QUANT_RESIDENT=auto'  args=''  02:49:15  clk=600 MHz  MemAvail=31341868 kB
<!--PRED 2	qres512	memavail_kb=31341868	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6509,6554,6430,5735,4866,4475,3772,3051,2195,2128,4913-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |          pp2048 |         68.83 ± 0.18 |

build: 171974745 (10558)
<!--DATA 2	qres512	pp2048	68.83-->
    [f16-resident] weights offered to the resident route: 165 resident on the NPU (2976MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21070MB (MemAvailable 30605MB - reserve 9535MB, no swap)

### smolvlm2-qres512 [stock] pass 2  env=''  args=''  02:51:38  clk=600 MHz  MemAvail=31472784 kB
<!--PRED 2	stock	memavail_kb=31472784	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6051,6555,6515,3463,4685,4344,3699,2960,2457,2122,4924-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |          pp2048 |         51.48 ± 0.10 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	51.48-->

### smolvlm2-qres512 [stock] pass 3  env=''  args=''  02:54:34  clk=600 MHz  MemAvail=31537632 kB
<!--PRED 3	stock	memavail_kb=31537952	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5491,6413,6256,4980,3760,3982,3448,2927,2615,2130,4932-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |          pp2048 |         51.34 ± 0.36 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	51.34-->

### smolvlm2-qres512 [qres512] pass 3  env='ROCKET_QUANT_RESIDENT=auto'  args=''  02:57:33  clk=600 MHz  MemAvail=31275668 kB
<!--PRED 3	qres512	memavail_kb=31275668	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5300,6335,6102,5374,3678,4190,3611,2975,2310,2103,4934-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |          pp2048 |         69.48 ± 0.45 |

build: 171974745 (10558)
<!--DATA 3	qres512	pp2048	69.48-->
    [f16-resident] weights offered to the resident route: 165 resident on the NPU (2976MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21003MB (MemAvailable 30539MB - reserve 9535MB, no swap)

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 51.53 | 3 | 51.34 | 51.76 | -- | -- |
| pp2048 | qres512 | 69.33 | 3 | 68.83 | 69.68 | 1.346x | 1.346 1.337 1.353 |

