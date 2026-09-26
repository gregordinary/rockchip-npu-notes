<!-- qwen35-9b quant-resident, pinning intervention  TESTS='-p 2048 -n 0 -r 3'  PASSES=3
     gguf=<data>/qwen35/Qwen3.5-9B-Q4_K_M.gguf (5680522464 bytes)
     every arm carries ROCKET_QUANT_RESIDENT=auto at the default -ub (the unstacked recipe)
     arms: unpinned | pin76 = PIN_MASK=0xf0 | pin76t4 = PIN_MASK=0xf0 -t 4
-->
== qwen35-9b-qres-pin  Tue Sep  1 00:25:29 UTC 2026 ==
### qwen35-9b-qres-pin [unpinned] pass 1  env='ROCKET_QUANT_RESIDENT=auto'  args=''  00:26:44  clk=600 MHz  MemAvail=31372328 kB
<!--PRED 1	unpinned	memavail_kb=31372580	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6997,7461,6642,6210,5524,4570,3745,3178,2694,946,4256-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         33.30 ± 0.17 |

build: 171974745 (10558)
<!--RO 1	unpinned	wall_s=282	busy=5370,5245,5240,5153,8742,7721,7832,9431	busy_tot=54734	busy_little_share=0.3838	a55_cpu_cycles=381408179439	a55_inst_retired=208066018708	context_switches=7896519	a76_cpu_cycles=753068726143	a76_l3d_cache_refill=9050623339	a76_l2d_cache_refill=5238108023	cpu_migrations=42946	page_faults=10970155	a76_dtlb_walk=2012876597	a76_mem_access=379323712443	a76_inst_retired=1215869974722	a76_l1d_cache_refill=9495908612	a55_inst_share=0.1461	a76_ipc=1.615	l2ref_pki=4.308	l3ref_pki=7.444	l1dref_pki=7.810	memacc_pki=311.98	dtlbw_pki=1.6555	pmu_enabled=100.0	pmu_cpu_s=2251.2	who=at_s=6,rss_pg=1485063,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=90112	present_frac=0.7340	l2color_cv=0.0026	l3color_cv=0.0107	contig_frac=0.7001	mean_run=3.3	vapa16=0.0179	gib_regions=28	anon_pg=20432	anon_l2cv=0.0021	anon_contig=0.4028	anon_run=1.7	anon_gib=25	file_pg=24576	file_l2cv=0.0003	file_contig=0.9961	file_run=171.9	file_gib=27	other_pg=21135	other_l2cv=0.0092	other_contig=0.6431	other_run=2.8	other_gib=28	ro_cost_ms=77-->
<!--DATA 1	unpinned	pp2048	33.30-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21006MB (MemAvailable 30541MB - reserve 9535MB, no swap)

### qwen35-9b-qres-pin [pin76] pass 1  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args=''  00:32:39  clk=600 MHz  MemAvail=31270648 kB
<!--PRED 1	pin76	memavail_kb=31270648	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7374,7049,6573,6518,5465,4728,4089,3565,2946,1181,3975-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         36.55 ± 0.32 |

build: 171974745 (10558)
<!--RO 1	pin76	wall_s=260	busy=245,193,263,262,9683,8276,8085,8971	busy_tot=35978	busy_little_share=0.0268	a55_cpu_cycles=31924616772	a55_inst_retired=8078496525	context_switches=8012173	a76_cpu_cycles=773845200610	a76_l3d_cache_refill=9580918135	a76_l2d_cache_refill=5460731089	cpu_migrations=53966	page_faults=10966312	a76_dtlb_walk=2054953645	a76_mem_access=426126343965	a76_inst_retired=1280573513647	a76_l1d_cache_refill=11160678391	a55_inst_share=0.0063	a76_ipc=1.655	l2ref_pki=4.264	l3ref_pki=7.482	l1dref_pki=8.715	memacc_pki=332.76	dtlbw_pki=1.6047	pmu_enabled=100.0	pmu_cpu_s=2079.8	who=at_s=6,rss_pg=1501442,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6874	l2color_cv=0.0048	l3color_cv=0.0156	contig_frac=0.7929	mean_run=4.8	vapa16=0.0247	gib_regions=28	anon_pg=4614	anon_l2cv=0.0221	anon_contig=0.7704	anon_run=4.3	anon_gib=27	file_pg=24576	file_l2cv=0.0017	file_contig=0.9894	file_run=79.8	file_gib=28	other_pg=21493	other_l2cv=0.0110	other_contig=0.5730	other_run=2.3	other_gib=28	ro_cost_ms=75-->
<!--DATA 1	pin76	pp2048	36.55-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20928MB (MemAvailable 30463MB - reserve 9535MB, no swap)

### qwen35-9b-qres-pin [pin76t4] pass 1  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-t 4'  00:38:11  clk=600 MHz  MemAvail=31203848 kB
<!--PRED 1	pin76t4	memavail_kb=31203556	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7316,6330,6648,5992,5422,4949,4344,3718,3084,1279,3839-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         36.27 ± 0.11 |

build: 171974745 (10558)
<!--RO 1	pin76t4	wall_s=263	busy=267,243,216,200,9334,8886,8660,9534	busy_tot=37340	busy_little_share=0.0248	a55_cpu_cycles=30608683650	a55_inst_retired=7704342260	context_switches=7767911	a76_cpu_cycles=801383895341	a76_l3d_cache_refill=10011312588	a76_l2d_cache_refill=5344634547	cpu_migrations=20241	page_faults=10961625	a76_dtlb_walk=2033360908	a76_mem_access=429470016162	a76_inst_retired=1305137778300	a76_l1d_cache_refill=11027655834	a55_inst_share=0.0059	a76_ipc=1.629	l2ref_pki=4.095	l3ref_pki=7.671	l1dref_pki=8.449	memacc_pki=329.06	dtlbw_pki=1.5580	pmu_enabled=100.0	pmu_cpu_s=2094.5	who=at_s=6,rss_pg=1501419,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6812	l2color_cv=0.0040	l3color_cv=0.0114	contig_frac=0.8272	mean_run=5.7	vapa16=0.0451	gib_regions=28	anon_pg=4608	anon_l2cv=0.0055	anon_contig=0.6149	anon_run=2.6	anon_gib=16	file_pg=24576	file_l2cv=0.0019	file_contig=0.9944	file_run=132.8	file_gib=27	other_pg=21037	other_l2cv=0.0104	other_contig=0.6784	other_run=3.1	other_gib=28	ro_cost_ms=70-->
<!--DATA 1	pin76t4	pp2048	36.27-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20852MB (MemAvailable 30387MB - reserve 9535MB, no swap)

### qwen35-9b-qres-pin [pin76] pass 2  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args=''  00:43:46  clk=600 MHz  MemAvail=31243040 kB
<!--PRED 2	pin76	memavail_kb=31243040	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6682,7521,7242,6713,5329,5029,4430,3830,3106,1309,3798-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         34.85 ± 0.04 |

build: 171974745 (10558)
<!--RO 2	pin76	wall_s=272	busy=264,186,252,248,9157,9296,8275,9009	busy_tot=36687	busy_little_share=0.0259	a55_cpu_cycles=32061230609	a55_inst_retired=8052444418	context_switches=8018196	a76_cpu_cycles=789213208315	a76_l3d_cache_refill=10233076579	a76_l2d_cache_refill=5467305698	cpu_migrations=55523	page_faults=10968594	a76_dtlb_walk=1641045279	a76_mem_access=426024288534	a76_inst_retired=1281106645053	a76_l1d_cache_refill=11130291326	a55_inst_share=0.0062	a76_ipc=1.623	l2ref_pki=4.268	l3ref_pki=7.988	l1dref_pki=8.688	memacc_pki=332.54	dtlbw_pki=1.2810	pmu_enabled=100.0	pmu_cpu_s=2167.5	who=at_s=6,rss_pg=1485325,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6764	l2color_cv=0.0023	l3color_cv=0.0087	contig_frac=0.5834	mean_run=2.4	vapa16=0.0339	gib_regions=28	anon_pg=4455	anon_l2cv=0.0052	anon_contig=0.3311	anon_run=1.5	anon_gib=24	file_pg=24576	file_l2cv=0.0002	file_contig=0.9856	file_run=61.3	file_gib=27	other_pg=20840	other_l2cv=0.0048	other_contig=0.1629	other_run=1.2	other_gib=28	ro_cost_ms=75-->
<!--DATA 2	pin76	pp2048	34.85-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20956MB (MemAvailable 30492MB - reserve 9535MB, no swap)

### qwen35-9b-qres-pin [pin76t4] pass 2  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-t 4'  00:49:29  clk=600 MHz  MemAvail=31451860 kB
<!--PRED 2	pin76t4	memavail_kb=31451860	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6597,7411,7183,6096,5408,4909,4292,3692,3012,1239,3941-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         36.29 ± 0.28 |

build: 171974745 (10558)
<!--RO 2	pin76t4	wall_s=262	busy=263,241,246,231,9096,9084,8580,9559	busy_tot=37300	busy_little_share=0.0263	a55_cpu_cycles=31408886117	a55_inst_retired=8046309602	context_switches=7765383	a76_cpu_cycles=798533423403	a76_l3d_cache_refill=9939965083	a76_l2d_cache_refill=5338561477	cpu_migrations=20493	page_faults=10966842	a76_dtlb_walk=2026176136	a76_mem_access=429398320208	a76_inst_retired=1305044523954	a76_l1d_cache_refill=11053199607	a55_inst_share=0.0061	a76_ipc=1.634	l2ref_pki=4.091	l3ref_pki=7.617	l1dref_pki=8.470	memacc_pki=329.03	dtlbw_pki=1.5526	pmu_enabled=100.0	pmu_cpu_s=2094.1	who=at_s=6,rss_pg=1501419,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6848	l2color_cv=0.0047	l3color_cv=0.0131	contig_frac=0.8313	mean_run=5.9	vapa16=0.0139	gib_regions=28	anon_pg=4608	anon_l2cv=0.0054	anon_contig=0.8121	anon_run=5.3	anon_gib=17	file_pg=24576	file_l2cv=0.0005	file_contig=0.9956	file_run=158.6	file_gib=26	other_pg=21304	other_l2cv=0.0103	other_contig=0.6458	other_run=2.8	other_gib=28	ro_cost_ms=60-->
<!--DATA 2	pin76t4	pp2048	36.29-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21088MB (MemAvailable 30624MB - reserve 9535MB, no swap)

### qwen35-9b-qres-pin [unpinned] pass 2  env='ROCKET_QUANT_RESIDENT=auto'  args=''  00:55:08  clk=600 MHz  MemAvail=31167664 kB
<!--PRED 2	unpinned	memavail_kb=31167664	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6719,7709,7385,6449,5605,5284,4659,4050,3298,1428,3619-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         34.49 ± 0.07 |

build: 171974745 (10558)
<!--RO 2	unpinned	wall_s=274	busy=5386,5215,5256,5180,8557,7812,8244,8784	busy_tot=54434	busy_little_share=0.3865	a55_cpu_cycles=381895675494	a55_inst_retired=208311872056	context_switches=7885361	a76_cpu_cycles=741799352790	a76_l3d_cache_refill=8126384893	a76_l2d_cache_refill=5225220678	cpu_migrations=43742	page_faults=10965972	a76_dtlb_walk=1945158935	a76_mem_access=380052772382	a76_inst_retired=1215710350968	a76_l1d_cache_refill=9669561712	a55_inst_share=0.1463	a76_ipc=1.639	l2ref_pki=4.298	l3ref_pki=6.684	l1dref_pki=7.954	memacc_pki=312.62	dtlbw_pki=1.6000	pmu_enabled=100.0	pmu_cpu_s=2190.0	who=at_s=6,rss_pg=1485064,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=90112	present_frac=0.7357	l2color_cv=0.0031	l3color_cv=0.0098	contig_frac=0.5125	mean_run=2.0	vapa16=0.0332	gib_regions=28	anon_pg=20686	anon_l2cv=0.0044	anon_contig=0.1943	anon_run=1.2	anon_gib=28	file_pg=24576	file_l2cv=0.0017	file_contig=0.9885	file_run=74.7	file_gib=28	other_pg=21032	other_l2cv=0.0093	other_contig=0.2692	other_run=1.4	other_gib=28	ro_cost_ms=66-->
<!--DATA 2	unpinned	pp2048	34.49-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20892MB (MemAvailable 30428MB - reserve 9535MB, no swap)

### qwen35-9b-qres-pin [pin76t4] pass 3  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-t 4'  01:00:54  clk=600 MHz  MemAvail=31214192 kB
<!--PRED 3	pin76t4	memavail_kb=31214192	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5538,6695,6505,5967,5265,4978,4377,3792,3118,1293,3818-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         36.02 ± 0.15 |

build: 171974745 (10558)
<!--RO 3	pin76t4	wall_s=264	busy=241,218,218,258,9368,8801,8944,9322	busy_tot=37370	busy_little_share=0.0250	a55_cpu_cycles=31599795137	a55_inst_retired=8128012210	context_switches=7771098	a76_cpu_cycles=799373341737	a76_l3d_cache_refill=9761744990	a76_l2d_cache_refill=5346734593	cpu_migrations=20594	page_faults=10969131	a76_dtlb_walk=1915151419	a76_mem_access=429368774632	a76_inst_retired=1304550429871	a76_l1d_cache_refill=11011697315	a55_inst_share=0.0062	a76_ipc=1.632	l2ref_pki=4.099	l3ref_pki=7.483	l1dref_pki=8.441	memacc_pki=329.13	dtlbw_pki=1.4681	pmu_enabled=100.0	pmu_cpu_s=2106.4	who=at_s=6,rss_pg=1501418,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6795	l2color_cv=0.0020	l3color_cv=0.0135	contig_frac=0.5952	mean_run=2.5	vapa16=0.3480	gib_regions=28	anon_pg=4608	anon_l2cv=0.0195	anon_contig=0.6445	anon_run=2.8	anon_gib=26	file_pg=24576	file_l2cv=0.0005	file_contig=0.9958	file_run=162.8	file_gib=27	other_pg=20912	other_l2cv=0.0041	other_contig=0.1134	other_run=1.1	other_gib=27	ro_cost_ms=71-->
<!--DATA 3	pin76t4	pp2048	36.02-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20942MB (MemAvailable 30477MB - reserve 9535MB, no swap)

### qwen35-9b-qres-pin [unpinned] pass 3  env='ROCKET_QUANT_RESIDENT=auto'  args=''  01:06:34  clk=600 MHz  MemAvail=31363328 kB
<!--PRED 3	unpinned	memavail_kb=31363524	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7362,7228,7040,6526,5573,4758,4121,3545,2920,1152,4014-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         34.46 ± 0.21 |

build: 171974745 (10558)
<!--RO 3	unpinned	wall_s=274	busy=5297,5253,5232,5121,8459,7756,7750,9280	busy_tot=54148	busy_little_share=0.3860	a55_cpu_cycles=380094165938	a55_inst_retired=207787079481	context_switches=7893620	a76_cpu_cycles=740729517316	a76_l3d_cache_refill=8173121422	a76_l2d_cache_refill=5188010700	cpu_migrations=43472	page_faults=10969133	a76_dtlb_walk=2004725842	a76_mem_access=379573919954	a76_inst_retired=1216071462545	a76_l1d_cache_refill=9588178987	a55_inst_share=0.1459	a76_ipc=1.642	l2ref_pki=4.266	l3ref_pki=6.721	l1dref_pki=7.885	memacc_pki=312.13	dtlbw_pki=1.6485	pmu_enabled=100.0	pmu_cpu_s=2192.0	who=at_s=6,rss_pg=1485805,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=90112	present_frac=0.5563	l2color_cv=0.0033	l3color_cv=0.0136	contig_frac=0.8420	mean_run=6.3	vapa16=0.0155	gib_regions=28	anon_pg=4608	anon_l2cv=0.0079	anon_contig=0.9261	anon_run=13.2	anon_gib=26	file_pg=24576	file_l2cv=0.0007	file_contig=0.9959	file_run=166.1	file_gib=27	other_pg=20948	other_l2cv=0.0079	other_contig=0.6430	other_run=2.8	other_gib=28	ro_cost_ms=75-->
<!--DATA 3	unpinned	pp2048	34.46-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21009MB (MemAvailable 30545MB - reserve 9535MB, no swap)

### qwen35-9b-qres-pin [pin76] pass 3  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args=''  01:12:21  clk=600 MHz  MemAvail=31096000 kB
<!--PRED 3	pin76	memavail_kb=31096000	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7915,7365,6642,6524,5499,4774,4167,3589,2963,1168,3923-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         36.82 ± 0.23 |

build: 171974745 (10558)
<!--RO 3	pin76	wall_s=259	busy=261,261,268,200,9145,8254,8898,8462	busy_tot=35749	busy_little_share=0.0277	a55_cpu_cycles=31921768935	a55_inst_retired=8097768832	context_switches=8012558	a76_cpu_cycles=767871819509	a76_l3d_cache_refill=9381636467	a76_l2d_cache_refill=5403725254	cpu_migrations=52963	page_faults=10966017	a76_dtlb_walk=1964523271	a76_mem_access=425985086521	a76_inst_retired=1280522932077	a76_l1d_cache_refill=11182755256	a55_inst_share=0.0063	a76_ipc=1.668	l2ref_pki=4.220	l3ref_pki=7.326	l1dref_pki=8.733	memacc_pki=332.66	dtlbw_pki=1.5342	pmu_enabled=100.0	pmu_cpu_s=2064.4	who=at_s=6,rss_pg=1501442,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6849	l2color_cv=0.0019	l3color_cv=0.0112	contig_frac=0.6865	mean_run=3.2	vapa16=0.0363	gib_regions=28	anon_pg=4608	anon_l2cv=0.0115	anon_contig=0.5445	anon_run=2.2	anon_gib=27	file_pg=24576	file_l2cv=0.0005	file_contig=0.9822	file_run=50.7	file_gib=26	other_pg=21310	other_l2cv=0.0041	other_contig=0.3761	other_run=1.6	other_gib=28	ro_cost_ms=70-->
<!--DATA 3	pin76	pp2048	36.82-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20812MB (MemAvailable 30348MB - reserve 9535MB, no swap)

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [unpinned]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | unpinned | 34.08 | 3 | 33.30 | 34.49 | -- | -- |
| pp2048 | pin76 | 36.07 | 3 | 34.85 | 36.82 | 1.059x | 1.098 1.010 1.068 |
| pp2048 | pin76t4 | 36.19 | 3 | 36.02 | 36.29 | 1.062x | 1.089 1.052 1.045 |

