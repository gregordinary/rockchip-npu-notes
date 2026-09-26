<!-- ministral3-8b-qres  class=qres  fp16-resident-fits=1  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=<data>/ministral3/Ministral-3-8B-Instruct-2512-Q4_K_M.gguf (5198386720 bytes)
     note: second model of the 8-9 B class; the class rule must not turn on one model
-->
== ministral3-8b-qres  Mon Aug 31 19:53:20 UTC 2026 ==
### ministral3-8b-qres [stock] pass 1  env=''  args=''  19:54:20  clk=600 MHz  MemAvail=31581664 kB
<!--PRED 1	stock	memavail_kb=31581684	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7092,7180,6660,6576,4657,4355,3851,3350,2716,1072,4344-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | ROCKET     |  -1 |          pp2048 |         17.91 ± 0.03 |

build: 171974745 (10558)
<!--RO 1	stock	wall_s=460	busy=2978,2584,2551,2521,30281,25684,24252,25210	busy_tot=116061	busy_little_share=0.0916	a55_cpu_cycles=218299334166	a55_inst_retired=123164283215	context_switches=8918320	a76_cpu_cycles=2268657612017	a76_l3d_cache_refill=18358985606	a76_l2d_cache_refill=7801306513	cpu_migrations=30410	page_faults=727300	a76_dtlb_walk=792984683	a76_mem_access=589913462843	a76_inst_retired=4635942881803	a76_l1d_cache_refill=14790877465	a55_inst_share=0.0259	a76_ipc=2.043	l2ref_pki=1.683	l3ref_pki=3.960	l1dref_pki=3.190	memacc_pki=127.25	dtlbw_pki=0.1711	pmu_enabled=100.0	pmu_cpu_s=3672.3	who=at_s=6,rss_pg=1430350,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=91136	present_frac=0.8757	l2color_cv=0.0026	l3color_cv=0.0077	contig_frac=0.9109	mean_run=11.0	vapa16=0.0058	gib_regions=28	anon_pg=38579	anon_l2cv=0.0041	anon_contig=0.9180	anon_run=11.9	anon_gib=28	file_pg=24576	file_l2cv=0.0008	file_contig=0.9944	file_run=132.1	file_gib=27	other_pg=16650	other_l2cv=0.0152	other_contig=0.7713	other_run=4.3	other_gib=28	ro_cost_ms=130-->
<!--DATA 1	stock	pp2048	17.91-->

### ministral3-8b-qres [qres512] pass 1  env='ROCKET_QUANT_RESIDENT=auto'  args=''  20:03:11  clk=600 MHz  MemAvail=31129736 kB
<!--PRED 1	qres512	memavail_kb=31129152	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5371,6208,6063,5609,4849,4450,3963,3490,2816,1053,4201-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | ROCKET     |  -1 |          pp2048 |         29.19 ± 0.13 |

build: 171974745 (10558)
<!--RO 1	qres512	wall_s=319	busy=2673,2508,2490,2471,13523,9329,9783,10996	busy_tot=53773	busy_little_share=0.1886	a55_cpu_cycles=200591198644	a55_inst_retired=117067537047	context_switches=8837532	a76_cpu_cycles=962721411429	a76_l3d_cache_refill=11179611987	a76_l2d_cache_refill=4609175887	cpu_migrations=30810	page_faults=11397733	a76_dtlb_walk=531787950	a76_mem_access=289631433151	a76_inst_retired=1549618183112	a76_l1d_cache_refill=10159311112	a55_inst_share=0.0702	a76_ipc=1.610	l2ref_pki=2.974	l3ref_pki=7.214	l1dref_pki=6.556	memacc_pki=186.91	dtlbw_pki=0.3432	pmu_enabled=100.0	pmu_cpu_s=2541.4	who=at_s=6,rss_pg=1432398,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=68608	present_frac=0.8448	l2color_cv=0.0021	l3color_cv=0.0048	contig_frac=0.8762	mean_run=8.0	vapa16=0.0477	gib_regions=27	anon_pg=14420	anon_l2cv=0.0067	anon_contig=0.8375	anon_run=6.1	anon_gib=26	file_pg=24576	file_l2cv=0.0009	file_contig=0.9954	file_run=151.7	file_gib=27	other_pg=18962	other_l2cv=0.0041	other_contig=0.7510	other_run=4.0	other_gib=27	ro_cost_ms=70-->
<!--DATA 1	qres512	pp2048	29.19-->
    [f16-resident] weights offered to the resident route: 235 resident on the NPU (13808MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20826MB (MemAvailable 30362MB - reserve 9535MB, no swap)

### ministral3-8b-qres [qres512] pass 2  env='ROCKET_QUANT_RESIDENT=auto'  args=''  20:09:42  clk=600 MHz  MemAvail=31224708 kB
<!--PRED 2	qres512	memavail_kb=31225028	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6804,7735,7381,7182,5386,4983,4420,3898,3092,1337,3887-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | ROCKET     |  -1 |          pp2048 |         29.25 ± 0.18 |

build: 171974745 (10558)
<!--RO 2	qres512	wall_s=317	busy=2725,2464,2465,2455,13406,9421,9712,11095	busy_tot=53743	busy_little_share=0.1881	a55_cpu_cycles=200164564344	a55_inst_retired=117159673022	context_switches=8827715	a76_cpu_cycles=966368593638	a76_l3d_cache_refill=11152305751	a76_l2d_cache_refill=4641872172	cpu_migrations=30492	page_faults=11368310	a76_dtlb_walk=529549157	a76_mem_access=289181281634	a76_inst_retired=1547836338292	a76_l1d_cache_refill=10167878968	a55_inst_share=0.0704	a76_ipc=1.602	l2ref_pki=2.999	l3ref_pki=7.205	l1dref_pki=6.569	memacc_pki=186.83	dtlbw_pki=0.3421	pmu_enabled=100.0	pmu_cpu_s=2534.0	who=at_s=6,rss_pg=1432393,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=68608	present_frac=0.8443	l2color_cv=0.0028	l3color_cv=0.0131	contig_frac=0.8855	mean_run=8.6	vapa16=0.0552	gib_regions=28	anon_pg=14610	anon_l2cv=0.0068	anon_contig=0.8595	anon_run=7.0	anon_gib=28	file_pg=24576	file_l2cv=0.0006	file_contig=0.9951	file_run=146.3	file_gib=27	other_pg=18742	other_l2cv=0.0086	other_contig=0.7620	other_run=4.2	other_gib=28	ro_cost_ms=84-->
<!--DATA 2	qres512	pp2048	29.25-->
    [f16-resident] weights offered to the resident route: 235 resident on the NPU (13808MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20915MB (MemAvailable 30451MB - reserve 9535MB, no swap)

### ministral3-8b-qres [stock] pass 2  env=''  args=''  20:15:58  clk=600 MHz  MemAvail=31596844 kB
<!--PRED 2	stock	memavail_kb=31596844	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6267,6952,6416,6450,5074,4423,3922,3486,2849,1096,4277-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | ROCKET     |  -1 |          pp2048 |         17.73 ± 0.05 |

build: 171974745 (10558)
<!--RO 2	stock	wall_s=465	busy=3069,2555,2531,2525,30456,26080,24101,24957	busy_tot=116274	busy_little_share=0.0919	a55_cpu_cycles=217503256591	a55_inst_retired=122948699878	context_switches=8912101	a76_cpu_cycles=2267982746846	a76_l3d_cache_refill=18404519742	a76_l2d_cache_refill=7758630177	cpu_migrations=29034	page_faults=687270	a76_dtlb_walk=795557201	a76_mem_access=589291154925	a76_inst_retired=4633311160176	a76_l1d_cache_refill=14820083844	a55_inst_share=0.0258	a76_ipc=2.043	l2ref_pki=1.675	l3ref_pki=3.972	l1dref_pki=3.199	memacc_pki=127.19	dtlbw_pki=0.1717	pmu_enabled=100.0	pmu_cpu_s=3711.3	who=at_s=6,rss_pg=1430356,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=91136	present_frac=0.8761	l2color_cv=0.0024	l3color_cv=0.0088	contig_frac=0.9389	mean_run=15.9	vapa16=0.0109	gib_regions=28	anon_pg=38422	anon_l2cv=0.0026	anon_contig=0.9683	anon_run=29.7	anon_gib=28	file_pg=24576	file_l2cv=0.0005	file_contig=0.9958	file_run=163.8	file_gib=26	other_pg=16850	other_l2cv=0.0120	other_contig=0.7886	other_run=4.7	other_gib=28	ro_cost_ms=123-->
<!--DATA 2	stock	pp2048	17.73-->

### ministral3-8b-qres [stock] pass 3  env=''  args=''  20:24:43  clk=600 MHz  MemAvail=31595088 kB
<!--PRED 3	stock	memavail_kb=31595088	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4766,6454,6106,5894,4680,3912,3512,3095,2516,742,4640-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | ROCKET     |  -1 |          pp2048 |         17.83 ± 0.01 |

build: 171974745 (10558)
<!--RO 3	stock	wall_s=462	busy=3076,2546,2499,2485,30245,25738,24239,25056	busy_tot=115884	busy_little_share=0.0915	a55_cpu_cycles=217188456118	a55_inst_retired=123180061630	context_switches=8893544	a76_cpu_cycles=2264607367918	a76_l3d_cache_refill=18262887170	a76_l2d_cache_refill=7795456107	cpu_migrations=29453	page_faults=688157	a76_dtlb_walk=784841225	a76_mem_access=589375132568	a76_inst_retired=4633071170794	a76_l1d_cache_refill=14789663386	a55_inst_share=0.0259	a76_ipc=2.046	l2ref_pki=1.683	l3ref_pki=3.942	l1dref_pki=3.192	memacc_pki=127.21	dtlbw_pki=0.1694	pmu_enabled=100.0	pmu_cpu_s=3691.3	who=at_s=6,rss_pg=1430355,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=91136	present_frac=0.8748	l2color_cv=0.0026	l3color_cv=0.0055	contig_frac=0.9221	mean_run=12.5	vapa16=0.0892	gib_regions=28	anon_pg=38515	anon_l2cv=0.0023	anon_contig=0.9569	anon_run=22.2	anon_gib=28	file_pg=24576	file_l2cv=0.0010	file_contig=0.9951	file_run=147.2	file_gib=26	other_pg=16631	other_l2cv=0.0125	other_contig=0.7334	other_run=3.7	other_gib=28	ro_cost_ms=129-->
<!--DATA 3	stock	pp2048	17.83-->

### ministral3-8b-qres [qres512] pass 3  env='ROCKET_QUANT_RESIDENT=auto'  args=''  20:33:36  clk=600 MHz  MemAvail=31380488 kB
<!--PRED 3	qres512	memavail_kb=31380488	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6603,6261,6213,6049,4866,4198,3758,3346,2727,920,4384-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | ROCKET     |  -1 |          pp2048 |         29.85 ± 0.06 |

build: 171974745 (10558)
<!--RO 3	qres512	wall_s=313	busy=2710,2455,2506,2444,13447,9461,9845,11018	busy_tot=53886	busy_little_share=0.1877	a55_cpu_cycles=200498509242	a55_inst_retired=117087561235	context_switches=8807050	a76_cpu_cycles=968272147953	a76_l3d_cache_refill=11155026185	a76_l2d_cache_refill=4608617219	cpu_migrations=29959	page_faults=11385217	a76_dtlb_walk=531382183	a76_mem_access=289362752147	a76_inst_retired=1548746592321	a76_l1d_cache_refill=10184728971	a55_inst_share=0.0703	a76_ipc=1.599	l2ref_pki=2.976	l3ref_pki=7.203	l1dref_pki=6.576	memacc_pki=186.84	dtlbw_pki=0.3431	pmu_enabled=100.0	pmu_cpu_s=2492.0	who=at_s=6,rss_pg=1432398,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=93184	present_frac=0.8825	l2color_cv=0.0016	l3color_cv=0.0080	contig_frac=0.6313	mean_run=2.7	vapa16=0.0168	gib_regions=27	anon_pg=38799	anon_l2cv=0.0016	anon_contig=0.4632	anon_run=1.9	anon_gib=24	file_pg=24576	file_l2cv=0.0011	file_contig=0.9941	file_run=128.0	file_gib=27	other_pg=18864	other_l2cv=0.0090	other_contig=0.5045	other_run=2.0	other_gib=27	ro_cost_ms=134-->
<!--DATA 3	qres512	pp2048	29.85-->
    [f16-resident] weights offered to the resident route: 235 resident on the NPU (13808MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21068MB (MemAvailable 30604MB - reserve 9535MB, no swap)

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 17.82 | 3 | 17.73 | 17.91 | -- | -- |
| pp2048 | qres512 | 29.43 | 3 | 29.19 | 29.85 | 1.651x | 1.630 1.650 1.674 |

