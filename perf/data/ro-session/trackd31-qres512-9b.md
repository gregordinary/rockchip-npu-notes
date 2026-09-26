<!-- qwen35-9b Q4_K_M, UNSTACKED quant residency (ROCKET_QUANT_RESIDENT=auto at the default -ub)
     TESTS='-p 2048 -n 0'  PASSES=3  (-r lives in each arm's args)
     gguf=<data>/qwen35/Qwen3.5-9B-Q4_K_M.gguf (5680522464 bytes)
     six PINNED arms (PIN_MASK=0xf0 -t 4) at -r 1/2/4: busy_tot = intercept + (r+1)*slope
       phi_u = 1 - slope_qres512/slope_stock ; ingest = intercept_qres512 - intercept_stock
     two UNPINNED arms at -r 2: K_u = qres512_u_r2/stock_u_r2 paired within a pass; a_u = 1 - 1/K_u
     2x2 at -r 2: interaction = (qres512_r2/stock_r2) / (qres512_u_r2/stock_u_r2)
     PREDICTION 2026-09-03e: phi_u = 0.675 (0.667-0.682), ingest 59.7 core-s (56-63),
       K_u 1.752 (1.72-1.79, reproduction band), a/phi 0.636 (0.62-0.65), q 4.18 (4.0-4.35);
       the third point sits +0.023 in a ABOVE the nested line a = 0.602 phi.
       RIVAL A phi_u = 0.6645 (same removed work as stacked). RIVAL B phi_u > 0.69.
       RIVAL C K_u outside 1.72-1.79 (board state moved).
     CONTROL: every qres512 arm 200 weights / 13184 MB; every stock arm zero
-->
== qwen35-9b-qres512-third-point  Thu Sep  3 16:57:59 UTC 2026 ==
### qwen35-9b-qres512-third-point [stock_r1] pass 1  env='PIN_MASK=0xf0'  args='-t 4 -r 1'  16:59:00  clk=600 MHz  MemAvail=31433972 kB
<!--PRED 1	stock_r1	memavail_kb=31434048	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9610,11880,10366,7774,6560,5862,4786,4082,4110,3538,2362-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.88 ± 0.00 |

build: 171974745 (10558)
<!--RO 1	stock_r1	wall_s=199	busy=121,137,122,122,12444,12267,11180,11273	busy_tot=47666	busy_little_share=0.0105	a55_cpu_cycles=18478108877	a55_inst_retired=4210536389	context_switches=3917024	a76_cpu_cycles=1020920004957	a76_l3d_cache_refill=8568738181	a76_l2d_cache_refill=4229265600	cpu_migrations=9716	page_faults=582653	a76_dtlb_walk=941676177	a76_mem_access=357373050691	a76_inst_retired=2123020297306	a76_l1d_cache_refill=7818417699	a55_inst_share=0.0020	a76_ipc=2.080	l2ref_pki=1.992	l3ref_pki=4.036	l1dref_pki=3.683	memacc_pki=168.33	dtlbw_pki=0.4436	pmu_enabled=100.0	pmu_cpu_s=1586.6	who=at_s=6,rss_pg=1509887,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6971	l2color_cv=0.0091	l3color_cv=0.0111	contig_frac=0.8109	mean_run=5.2	vapa16=0.2136	gib_regions=28	anon_pg=23042	anon_l2cv=0.0012	anon_contig=0.8274	anon_run=5.7	anon_gib=27	file_pg=24576	file_l2cv=0.0023	file_contig=0.9777	file_run=41.4	file_gib=27	other_pg=20912	other_l2cv=0.0286	other_contig=0.5965	other_run=2.5	other_gib=28	ro_cost_ms=105-->
<!--DATA 1	stock_r1	pp2048	20.88-->

### qwen35-9b-qres512-third-point [stock_r2] pass 1  env='PIN_MASK=0xf0'  args='-t 4 -r 2'  17:03:45  clk=600 MHz  MemAvail=31444004 kB
<!--PRED 1	stock_r2	memavail_kb=31444004	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9498,12224,10653,7603,6418,5753,4854,4055,4066,3516,2391-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         21.05 ± 0.03 |

build: 171974745 (10558)
<!--RO 1	stock_r2	wall_s=295	busy=153,176,186,124,18619,18471,16563,16823	busy_tot=71115	busy_little_share=0.0090	a55_cpu_cycles=25231449962	a55_inst_retired=5232094894	context_switches=5876848	a76_cpu_cycles=1527847675594	a76_l3d_cache_refill=12414489273	a76_l2d_cache_refill=6353548712	cpu_migrations=13762	page_faults=579426	a76_dtlb_walk=1721903564	a76_mem_access=534585411521	a76_inst_retired=3179077994599	a76_l1d_cache_refill=11778219599	a55_inst_share=0.0016	a76_ipc=2.081	l2ref_pki=1.999	l3ref_pki=3.905	l1dref_pki=3.705	memacc_pki=168.16	dtlbw_pki=0.5416	pmu_enabled=100.0	pmu_cpu_s=2353.0	who=at_s=6,rss_pg=1509890,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6966	l2color_cv=0.0080	l3color_cv=0.0148	contig_frac=0.8297	mean_run=5.8	vapa16=0.0236	gib_regions=28	anon_pg=23041	anon_l2cv=0.0019	anon_contig=0.8596	anon_run=7.0	anon_gib=27	file_pg=24576	file_l2cv=0.0021	file_contig=0.9758	file_run=38.3	file_gib=28	other_pg=20866	other_l2cv=0.0263	other_contig=0.6244	other_run=2.7	other_gib=28	ro_cost_ms=110-->
<!--DATA 1	stock_r2	pp2048	21.05-->

### qwen35-9b-qres512-third-point [stock_r4] pass 1  env='PIN_MASK=0xf0'  args='-t 4 -r 4'  17:10:58  clk=600 MHz  MemAvail=31460668 kB
<!--PRED 1	stock_r4	memavail_kb=31460668	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9300,12301,10160,7567,6386,5698,4862,4056,4065,3502,2405-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.86 ± 0.03 |

build: 171974745 (10558)
<!--RO 1	stock_r4	wall_s=494	busy=280,202,276,223,30791,30670,27522,28351	busy_tot=118315	busy_little_share=0.0083	a55_cpu_cycles=41648648332	a55_inst_retired=8435580195	context_switches=9795214	a76_cpu_cycles=2541905102198	a76_l3d_cache_refill=21176997539	a76_l2d_cache_refill=10438469781	cpu_migrations=23329	page_faults=599628	a76_dtlb_walk=2362504003	a76_mem_access=889004150476	a76_inst_retired=5292942213221	a76_l1d_cache_refill=19493339748	a55_inst_share=0.0016	a76_ipc=2.082	l2ref_pki=1.972	l3ref_pki=4.001	l1dref_pki=3.683	memacc_pki=167.96	dtlbw_pki=0.4463	pmu_enabled=100.0	pmu_cpu_s=3946.1	who=at_s=6,rss_pg=1509886,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6975	l2color_cv=0.0093	l3color_cv=0.0141	contig_frac=0.8589	mean_run=7.0	vapa16=0.0874	gib_regions=28	anon_pg=23041	anon_l2cv=0.0015	anon_contig=0.8792	anon_run=8.2	anon_gib=27	file_pg=24576	file_l2cv=0.0045	file_contig=0.9785	file_run=42.7	file_gib=28	other_pg=20952	other_l2cv=0.0338	other_contig=0.6961	other_run=3.3	other_gib=28	ro_cost_ms=99-->
<!--DATA 1	stock_r4	pp2048	20.86-->

### qwen35-9b-qres512-third-point [qres512_r1] pass 1  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-t 4 -r 1'  17:20:23  clk=600 MHz  MemAvail=31131728 kB
<!--PRED 1	qres512_r1	memavail_kb=31131728	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10189,11267,11364,8082,6941,6148,5245,4308,3998,3719,2147-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         37.24 ± 0.00 |

build: 171974745 (10558)
<!--RO 1	qres512_r1	wall_s=147	busy=106,43,100,123,5278,4903,6113,4856	busy_tot=21522	busy_little_share=0.0173	a55_cpu_cycles=13607556639	a55_inst_retired=3044091700	context_switches=3871910	a76_cpu_cycles=463028063728	a76_l3d_cache_refill=5589783761	a76_l2d_cache_refill=2834032357	cpu_migrations=9675	page_faults=10903719	a76_dtlb_walk=1038046863	a76_mem_access=235968147500	a76_inst_retired=790071016717	a76_l1d_cache_refill=5672472360	a55_inst_share=0.0038	a76_ipc=1.706	l2ref_pki=3.587	l3ref_pki=7.075	l1dref_pki=7.180	memacc_pki=298.67	dtlbw_pki=1.3139	pmu_enabled=100.0	pmu_cpu_s=1168.9	who=at_s=6,rss_pg=1501415,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6864	l2color_cv=0.0048	l3color_cv=0.0171	contig_frac=0.7726	mean_run=4.4	vapa16=0.0256	gib_regions=28	anon_pg=4608	anon_l2cv=0.0089	anon_contig=0.6456	anon_run=2.8	anon_gib=26	file_pg=24576	file_l2cv=0.0030	file_contig=0.9810	file_run=47.7	file_gib=27	other_pg=21421	other_l2cv=0.0128	other_contig=0.5609	other_run=2.3	other_gib=28	ro_cost_ms=64-->
<!--DATA 1	qres512_r1	pp2048	37.24-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20778MB (MemAvailable 30313MB - reserve 9535MB, no swap)

### qwen35-9b-qres512-third-point [qres512_r2] pass 1  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-t 4 -r 2'  17:24:17  clk=600 MHz  MemAvail=31303828 kB
<!--PRED 1	qres512_r2	memavail_kb=31303708	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9441,13014,11495,8720,7676,6895,5914,4854,4568,4036,1736-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         36.29 ± 0.08 |

build: 171974745 (10558)
<!--RO 1	qres512_r2	wall_s=206	busy=166,134,124,149,7136,6556,7924,7213	busy_tot=29402	busy_little_share=0.0195	a55_cpu_cycles=20937443692	a55_inst_retired=4717039126	context_switches=5823834	a76_cpu_cycles=632620967457	a76_l3d_cache_refill=7745636261	a76_l2d_cache_refill=4083324973	cpu_migrations=14146	page_faults=10917398	a76_dtlb_walk=1536306253	a76_mem_access=332736326299	a76_inst_retired=1047814507173	a76_l1d_cache_refill=8216561315	a55_inst_share=0.0045	a76_ipc=1.656	l2ref_pki=3.897	l3ref_pki=7.392	l1dref_pki=7.842	memacc_pki=317.55	dtlbw_pki=1.4662	pmu_enabled=100.0	pmu_cpu_s=1644.2	who=at_s=6,rss_pg=1495048,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=90112	present_frac=0.5551	l2color_cv=0.0020	l3color_cv=0.0103	contig_frac=0.6897	mean_run=3.2	vapa16=0.0211	gib_regions=28	anon_pg=4608	anon_l2cv=0.0291	anon_contig=0.3810	anon_run=1.6	anon_gib=28	file_pg=24576	file_l2cv=0.0062	file_contig=0.9661	file_run=27.9	file_gib=28	other_pg=20841	other_l2cv=0.0031	other_contig=0.4319	other_run=1.8	other_gib=26	ro_cost_ms=69-->
<!--DATA 1	qres512_r2	pp2048	36.29-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21035MB (MemAvailable 30570MB - reserve 9535MB, no swap)

### qwen35-9b-qres512-third-point [qres512_r4] pass 1  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-t 4 -r 4'  17:29:42  clk=600 MHz  MemAvail=31452916 kB
<!--PRED 1	qres512_r4	memavail_kb=31452916	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10612,13168,11344,8637,7617,6958,5929,4909,4764,4047,1709-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         38.11 ± 0.16 |

build: 171974745 (10558)
<!--RO 1	qres512_r4	wall_s=305	busy=238,237,241,239,10894,10559,11112,11021	busy_tot=44541	busy_little_share=0.0214	a55_cpu_cycles=33486783113	a55_inst_retired=7668006922	context_switches=9618051	a76_cpu_cycles=961944839709	a76_l3d_cache_refill=11874063079	a76_l2d_cache_refill=6607525120	cpu_migrations=23791	page_faults=10934980	a76_dtlb_walk=2529912121	a76_mem_access=525623419720	a76_inst_retired=1560507395269	a76_l1d_cache_refill=13347239516	a55_inst_share=0.0049	a76_ipc=1.622	l2ref_pki=4.234	l3ref_pki=7.609	l1dref_pki=8.553	memacc_pki=336.83	dtlbw_pki=1.6212	pmu_enabled=100.0	pmu_cpu_s=2437.0	who=at_s=6,rss_pg=1501419,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6858	l2color_cv=0.0092	l3color_cv=0.0198	contig_frac=0.7793	mean_run=4.5	vapa16=0.0275	gib_regions=28	anon_pg=4608	anon_l2cv=0.0144	anon_contig=0.8434	anon_run=6.3	anon_gib=27	file_pg=24576	file_l2cv=0.0008	file_contig=0.9795	file_run=44.5	file_gib=27	other_pg=21376	other_l2cv=0.0218	other_contig=0.5352	other_run=2.1	other_gib=28	ro_cost_ms=69-->
<!--DATA 1	qres512_r4	pp2048	38.11-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21074MB (MemAvailable 30610MB - reserve 9535MB, no swap)

### qwen35-9b-qres512-third-point [stock_u_r2] pass 1  env=''  args='-r 2'  17:36:21  clk=600 MHz  MemAvail=31488160 kB
<!--PRED 1	stock_u_r2	memavail_kb=31488160	anonhuge_kb=0	hugepagesz_kb=2048	buddy=11365,12264,12373,8167,7053,6624,5672,4656,4618,3924,1883-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         19.05 ± 0.07 |

build: 171974745 (10558)
<!--RO 1	stock_u_r2	wall_s=324	busy=5119,4140,4097,3982,18605,16601,16664,17218	busy_tot=86426	busy_little_share=0.2006	a55_cpu_cycles=319305597540	a55_inst_retired=167495338109	context_switches=5978502	a76_cpu_cycles=1492905026654	a76_l3d_cache_refill=11835966413	a76_l2d_cache_refill=6185096572	cpu_migrations=30447	page_faults=581871	a76_dtlb_walk=1656740473	a76_mem_access=493857686339	a76_inst_retired=3099861496588	a76_l1d_cache_refill=10310762198	a55_inst_share=0.0513	a76_ipc=2.076	l2ref_pki=1.995	l3ref_pki=3.818	l1dref_pki=3.326	memacc_pki=159.32	dtlbw_pki=0.5345	pmu_enabled=100.0	pmu_cpu_s=2591.0	who=at_s=6,rss_pg=1509913,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6966	l2color_cv=0.0087	l3color_cv=0.0157	contig_frac=0.8427	mean_run=6.3	vapa16=0.0150	gib_regions=28	anon_pg=23041	anon_l2cv=0.0015	anon_contig=0.9250	anon_run=13.0	anon_gib=27	file_pg=24576	file_l2cv=0.0013	file_contig=0.9800	file_run=45.7	file_gib=27	other_pg=20858	other_l2cv=0.0285	other_contig=0.5900	other_run=2.4	other_gib=28	ro_cost_ms=81-->
<!--DATA 1	stock_u_r2	pp2048	19.05-->

### qwen35-9b-qres512-third-point [qres512_u_r2] pass 1  env='ROCKET_QUANT_RESIDENT=auto'  args='-r 2'  17:43:19  clk=600 MHz  MemAvail=31464112 kB
<!--PRED 1	qres512_u_r2	memavail_kb=31464112	anonhuge_kb=0	hugepagesz_kb=2048	buddy=11128,12352,11535,7942,6896,6368,5431,4478,4364,3785,2063-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         33.70 ± 0.03 |

build: 171974745 (10558)
<!--RO 1	qres512_u_r2	wall_s=219	busy=3995,3990,3951,3871,6845,5821,6744,7029	busy_tot=42246	busy_little_share=0.3742	a55_cpu_cycles=286644061379	a55_inst_retired=156162466567	context_switches=5905882	a76_cpu_cycles=593085950563	a76_l3d_cache_refill=6882258367	a76_l2d_cache_refill=4017657616	cpu_migrations=31581	page_faults=10918315	a76_dtlb_walk=1514631915	a76_mem_access=294734792127	a76_inst_retired=978621840481	a76_l1d_cache_refill=7248005968	a55_inst_share=0.1376	a76_ipc=1.650	l2ref_pki=4.105	l3ref_pki=7.033	l1dref_pki=7.406	memacc_pki=301.17	dtlbw_pki=1.5477	pmu_enabled=100.0	pmu_cpu_s=1747.3	who=at_s=6,rss_pg=1485065,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=90112	present_frac=0.7359	l2color_cv=0.0060	l3color_cv=0.0204	contig_frac=0.7144	mean_run=3.5	vapa16=0.0296	gib_regions=28	anon_pg=20782	anon_l2cv=0.0010	anon_contig=0.5973	anon_run=2.5	anon_gib=26	file_pg=24576	file_l2cv=0.0020	file_contig=0.9806	file_run=46.8	file_gib=27	other_pg=20957	other_l2cv=0.0198	other_contig=0.5185	other_run=2.1	other_gib=28	ro_cost_ms=66-->
<!--DATA 1	qres512_u_r2	pp2048	33.70-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21087MB (MemAvailable 30622MB - reserve 9535MB, no swap)

### qwen35-9b-qres512-third-point [stock_r2] pass 2  env='PIN_MASK=0xf0'  args='-t 4 -r 2'  17:48:23  clk=600 MHz  MemAvail=31443264 kB
<!--PRED 2	stock_r2	memavail_kb=31443264	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8706,12639,11703,7926,6629,6369,5445,4448,4409,3813,2041-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.97 ± 0.05 |

build: 171974745 (10558)
<!--RO 2	stock_r2	wall_s=297	busy=169,142,184,143,18486,18395,16575,16961	busy_tot=71055	busy_little_share=0.0090	a55_cpu_cycles=25919864047	a55_inst_retired=5546203349	context_switches=5869510	a76_cpu_cycles=1526610887832	a76_l3d_cache_refill=12743442952	a76_l2d_cache_refill=6357930381	cpu_migrations=14011	page_faults=587451	a76_dtlb_walk=1419407454	a76_mem_access=534547527269	a76_inst_retired=3179557027612	a76_l1d_cache_refill=11713359413	a55_inst_share=0.0017	a76_ipc=2.083	l2ref_pki=2.000	l3ref_pki=4.008	l1dref_pki=3.684	memacc_pki=168.12	dtlbw_pki=0.4464	pmu_enabled=100.0	pmu_cpu_s=2366.0	who=at_s=6,rss_pg=1509886,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6989	l2color_cv=0.0084	l3color_cv=0.0138	contig_frac=0.8265	mean_run=5.7	vapa16=0.0304	gib_regions=28	anon_pg=23041	anon_l2cv=0.0029	anon_contig=0.8606	anon_run=7.1	anon_gib=27	file_pg=24576	file_l2cv=0.0049	file_contig=0.9806	file_run=46.8	file_gib=27	other_pg=21084	other_l2cv=0.0251	other_contig=0.6095	other_run=2.6	other_gib=28	ro_cost_ms=110-->
<!--DATA 2	stock_r2	pp2048	20.97-->

### qwen35-9b-qres512-third-point [stock_r4] pass 2  env='PIN_MASK=0xf0'  args='-t 4 -r 4'  17:55:37  clk=600 MHz  MemAvail=31467116 kB
<!--PRED 2	stock_r4	memavail_kb=31467116	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9255,12805,10649,7859,6432,6068,5279,4322,4306,3711,2165-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.88 ± 0.07 |

build: 171974745 (10558)
<!--RO 2	stock_r4	wall_s=493	busy=205,279,251,250,30916,30513,27422,28026	busy_tot=117862	busy_little_share=0.0084	a55_cpu_cycles=41220325450	a55_inst_retired=8419762319	context_switches=9804628	a76_cpu_cycles=2534017416432	a76_l3d_cache_refill=20735869790	a76_l2d_cache_refill=10521391734	cpu_migrations=23125	page_faults=598811	a76_dtlb_walk=2246629601	a76_mem_access=889181538027	a76_inst_retired=5292645654515	a76_l1d_cache_refill=19604644298	a55_inst_share=0.0016	a76_ipc=2.089	l2ref_pki=1.988	l3ref_pki=3.918	l1dref_pki=3.704	memacc_pki=168.00	dtlbw_pki=0.4245	pmu_enabled=100.0	pmu_cpu_s=3941.3	who=at_s=6,rss_pg=1509887,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6983	l2color_cv=0.0057	l3color_cv=0.0114	contig_frac=0.8596	mean_run=7.0	vapa16=0.1765	gib_regions=28	anon_pg=23041	anon_l2cv=0.0026	anon_contig=0.8794	anon_run=8.2	anon_gib=27	file_pg=24576	file_l2cv=0.0029	file_contig=0.9708	file_run=32.2	file_gib=28	other_pg=21033	other_l2cv=0.0191	other_contig=0.7080	other_run=3.4	other_gib=28	ro_cost_ms=89-->
<!--DATA 2	stock_r4	pp2048	20.88-->

### qwen35-9b-qres512-third-point [qres512_r1] pass 2  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-t 4 -r 1'  18:05:03  clk=600 MHz  MemAvail=31243184 kB
<!--PRED 2	qres512_r1	memavail_kb=31243184	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9918,11979,10725,8443,7151,6456,5574,4597,4319,3883,1941-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         36.77 ± 0.00 |

build: 171974745 (10558)
<!--RO 2	qres512_r1	wall_s=148	busy=109,80,109,87,5074,5092,6206,4809	busy_tot=21566	busy_little_share=0.0179	a55_cpu_cycles=14030207593	a55_inst_retired=3109480597	context_switches=3880496	a76_cpu_cycles=463928378295	a76_l3d_cache_refill=5531490243	a76_l2d_cache_refill=2831507783	cpu_migrations=9210	page_faults=10905001	a76_dtlb_walk=932970559	a76_mem_access=235992369482	a76_inst_retired=790178887338	a76_l1d_cache_refill=5663514695	a55_inst_share=0.0039	a76_ipc=1.703	l2ref_pki=3.583	l3ref_pki=7.000	l1dref_pki=7.167	memacc_pki=298.66	dtlbw_pki=1.1807	pmu_enabled=100.0	pmu_cpu_s=1180.0	who=at_s=6,rss_pg=1501419,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6780	l2color_cv=0.0069	l3color_cv=0.0181	contig_frac=0.7605	mean_run=4.1	vapa16=0.0215	gib_regions=28	anon_pg=4608	anon_l2cv=0.0289	anon_contig=0.4175	anon_run=1.7	anon_gib=27	file_pg=24576	file_l2cv=0.0046	file_contig=0.9685	file_run=30.0	file_gib=27	other_pg=20802	other_l2cv=0.0106	other_contig=0.5906	other_run=2.4	other_gib=28	ro_cost_ms=71-->
<!--DATA 2	qres512_r1	pp2048	36.77-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20969MB (MemAvailable 30505MB - reserve 9535MB, no swap)

### qwen35-9b-qres512-third-point [qres512_r2] pass 2  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-t 4 -r 2'  18:08:58  clk=600 MHz  MemAvail=31351108 kB
<!--PRED 2	qres512_r2	memavail_kb=31351108	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10724,12072,11384,8237,7165,6506,5592,4580,4391,3876,1951-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         36.25 ± 0.12 |

build: 171974745 (10558)
<!--RO 2	qres512_r2	wall_s=206	busy=126,162,104,173,6922,6963,7802,7103	busy_tot=29355	busy_little_share=0.0192	a55_cpu_cycles=20551703626	a55_inst_retired=4647556986	context_switches=5824843	a76_cpu_cycles=630926749515	a76_l3d_cache_refill=7752047778	a76_l2d_cache_refill=4063128611	cpu_migrations=14178	page_faults=10914217	a76_dtlb_walk=1241197322	a76_mem_access=332612908309	a76_inst_retired=1047491022088	a76_l1d_cache_refill=8212313406	a55_inst_share=0.0044	a76_ipc=1.660	l2ref_pki=3.879	l3ref_pki=7.401	l1dref_pki=7.840	memacc_pki=317.53	dtlbw_pki=1.1849	pmu_enabled=100.0	pmu_cpu_s=1641.7	who=at_s=6,rss_pg=1501419,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6870	l2color_cv=0.0092	l3color_cv=0.0238	contig_frac=0.7611	mean_run=4.2	vapa16=0.0174	gib_regions=28	anon_pg=4608	anon_l2cv=0.0051	anon_contig=0.7052	anon_run=3.4	anon_gib=26	file_pg=24576	file_l2cv=0.0048	file_contig=0.9714	file_run=32.8	file_gib=28	other_pg=21464	other_l2cv=0.0196	other_contig=0.5324	other_run=2.1	other_gib=28	ro_cost_ms=70-->
<!--DATA 2	qres512_r2	pp2048	36.25-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21007MB (MemAvailable 30543MB - reserve 9535MB, no swap)

### qwen35-9b-qres512-third-point [qres512_r4] pass 2  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-t 4 -r 4'  18:14:21  clk=600 MHz  MemAvail=31455440 kB
<!--PRED 2	qres512_r4	memavail_kb=31455440	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9475,12520,11926,8615,7528,6907,5933,4885,4717,4033,1734-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         37.29 ± 0.21 |

build: 171974745 (10558)
<!--RO 2	qres512_r4	wall_s=311	busy=260,176,248,262,11080,10840,11172,10766	busy_tot=44804	busy_little_share=0.0211	a55_cpu_cycles=33650781985	a55_inst_retired=7547903172	context_switches=9632252	a76_cpu_cycles=966518790226	a76_l3d_cache_refill=11982755335	a76_l2d_cache_refill=6616095638	cpu_migrations=22901	page_faults=10932471	a76_dtlb_walk=2264243568	a76_mem_access=525682370817	a76_inst_retired=1560755870125	a76_l1d_cache_refill=13369074381	a55_inst_share=0.0048	a76_ipc=1.615	l2ref_pki=4.239	l3ref_pki=7.678	l1dref_pki=8.566	memacc_pki=336.81	dtlbw_pki=1.4507	pmu_enabled=100.0	pmu_cpu_s=2483.9	who=at_s=6,rss_pg=1501418,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6857	l2color_cv=0.0082	l3color_cv=0.0154	contig_frac=0.7618	mean_run=4.2	vapa16=0.0353	gib_regions=28	anon_pg=4608	anon_l2cv=0.0056	anon_contig=0.7591	anon_run=4.1	anon_gib=26	file_pg=24576	file_l2cv=0.0040	file_contig=0.9761	file_run=38.8	file_gib=27	other_pg=21372	other_l2cv=0.0180	other_contig=0.5159	other_run=2.1	other_gib=28	ro_cost_ms=73-->
<!--DATA 2	qres512_r4	pp2048	37.29-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21077MB (MemAvailable 30612MB - reserve 9535MB, no swap)

### qwen35-9b-qres512-third-point [stock_u_r2] pass 2  env=''  args='-r 2'  18:21:07  clk=600 MHz  MemAvail=31469112 kB
<!--PRED 2	stock_u_r2	memavail_kb=31469112	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8845,11755,11911,7962,6846,6444,5519,4553,4462,3859,1987-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         19.22 ± 0.07 |

build: 171974745 (10558)
<!--RO 2	stock_u_r2	wall_s=322	busy=5125,4044,4073,3885,18631,16839,16627,17174	busy_tot=86398	busy_little_share=0.1982	a55_cpu_cycles=315382667371	a55_inst_retired=165695339389	context_switches=5973031	a76_cpu_cycles=1494502970180	a76_l3d_cache_refill=11932464956	a76_l2d_cache_refill=6181504375	cpu_migrations=30184	page_faults=583099	a76_dtlb_walk=1661955106	a76_mem_access=494209368172	a76_inst_retired=3101400606626	a76_l1d_cache_refill=10327412147	a55_inst_share=0.0507	a76_ipc=2.075	l2ref_pki=1.993	l3ref_pki=3.847	l1dref_pki=3.330	memacc_pki=159.35	dtlbw_pki=0.5359	pmu_enabled=100.0	pmu_cpu_s=2573.8	who=at_s=6,rss_pg=1509914,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6971	l2color_cv=0.0070	l3color_cv=0.0142	contig_frac=0.8490	mean_run=6.5	vapa16=0.0720	gib_regions=28	anon_pg=23042	anon_l2cv=0.0038	anon_contig=0.9267	anon_run=13.3	anon_gib=28	file_pg=24576	file_l2cv=0.0030	file_contig=0.9676	file_run=29.2	file_gib=27	other_pg=20912	other_l2cv=0.0224	other_contig=0.6239	other_run=2.6	other_gib=28	ro_cost_ms=68-->
<!--DATA 2	stock_u_r2	pp2048	19.22-->

### qwen35-9b-qres512-third-point [qres512_u_r2] pass 2  env='ROCKET_QUANT_RESIDENT=auto'  args='-r 2'  18:28:02  clk=600 MHz  MemAvail=31464116 kB
<!--PRED 2	qres512_u_r2	memavail_kb=31464116	anonhuge_kb=0	hugepagesz_kb=2048	buddy=11171,11993,11708,8160,7113,6484,5572,4568,4498,3896,1945-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         33.00 ± 0.03 |

build: 171974745 (10558)
<!--RO 2	qres512_u_r2	wall_s=222	busy=4095,3952,3945,3890,6969,6129,6640,6961	busy_tot=42581	busy_little_share=0.3730	a55_cpu_cycles=288413773527	a55_inst_retired=156707363812	context_switches=5923144	a76_cpu_cycles=596112483177	a76_l3d_cache_refill=7026738757	a76_l2d_cache_refill=3996531729	cpu_migrations=32337	page_faults=10915975	a76_dtlb_walk=1511806906	a76_mem_access=294617333082	a76_inst_retired=978789006224	a76_l1d_cache_refill=7237497364	a55_inst_share=0.1380	a76_ipc=1.642	l2ref_pki=4.083	l3ref_pki=7.179	l1dref_pki=7.394	memacc_pki=301.00	dtlbw_pki=1.5446	pmu_enabled=100.0	pmu_cpu_s=1772.0	who=at_s=6,rss_pg=1509635,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6825	l2color_cv=0.0101	l3color_cv=0.0213	contig_frac=0.7663	mean_run=4.3	vapa16=0.0193	gib_regions=28	anon_pg=4608	anon_l2cv=0.0149	anon_contig=0.7506	anon_run=4.0	anon_gib=26	file_pg=24576	file_l2cv=0.0047	file_contig=0.9783	file_run=42.3	file_gib=28	other_pg=21136	other_l2cv=0.0206	other_contig=0.5233	other_run=2.1	other_gib=28	ro_cost_ms=82-->
<!--DATA 2	qres512_u_r2	pp2048	33.00-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21076MB (MemAvailable 30612MB - reserve 9535MB, no swap)

### qwen35-9b-qres512-third-point [stock_r1] pass 2  env='PIN_MASK=0xf0'  args='-t 4 -r 1'  18:32:44  clk=600 MHz  MemAvail=31427784 kB
<!--PRED 2	stock_r1	memavail_kb=31427992	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9134,11612,11140,7951,6767,6399,5472,4492,4402,3829,2024-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         21.05 ± 0.00 |

build: 171974745 (10558)
<!--RO 2	stock_r1	wall_s=198	busy=127,100,130,95,12473,12174,11163,11247	busy_tot=47509	busy_little_share=0.0095	a55_cpu_cycles=17153310843	a55_inst_retired=3653477406	context_switches=3924769	a76_cpu_cycles=1018744367178	a76_l3d_cache_refill=8291799432	a76_l2d_cache_refill=4243680090	cpu_migrations=9389	page_faults=571045	a76_dtlb_walk=1143009463	a76_mem_access=357295490420	a76_inst_retired=2122388627611	a76_l1d_cache_refill=7845684258	a55_inst_share=0.0017	a76_ipc=2.083	l2ref_pki=1.999	l3ref_pki=3.907	l1dref_pki=3.697	memacc_pki=168.35	dtlbw_pki=0.5385	pmu_enabled=100.0	pmu_cpu_s=1578.7	who=at_s=6,rss_pg=1509893,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6961	l2color_cv=0.0050	l3color_cv=0.0119	contig_frac=0.8226	mean_run=5.6	vapa16=0.0142	gib_regions=28	anon_pg=23041	anon_l2cv=0.0024	anon_contig=0.8528	anon_run=6.7	anon_gib=28	file_pg=24576	file_l2cv=0.0018	file_contig=0.9706	file_run=32.0	file_gib=28	other_pg=20810	other_l2cv=0.0155	other_contig=0.6143	other_run=2.6	other_gib=28	ro_cost_ms=111-->
<!--DATA 2	stock_r1	pp2048	21.05-->

### qwen35-9b-qres512-third-point [stock_r4] pass 3  env='PIN_MASK=0xf0'  args='-t 4 -r 4'  18:38:19  clk=600 MHz  MemAvail=31464880 kB
<!--PRED 3	stock_r4	memavail_kb=31464880	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10017,12616,10470,7667,6459,6009,5226,4316,4295,3701,2180-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.76 ± 0.03 |

build: 171974745 (10558)
<!--RO 3	stock_r4	wall_s=496	busy=271,251,254,254,30822,30802,27683,28013	busy_tot=118350	busy_little_share=0.0087	a55_cpu_cycles=41670305834	a55_inst_retired=8562468431	context_switches=9802499	a76_cpu_cycles=2544996182595	a76_l3d_cache_refill=21406542345	a76_l2d_cache_refill=10556049269	cpu_migrations=22793	page_faults=601534	a76_dtlb_walk=2856596875	a76_mem_access=889010806881	a76_inst_retired=5293025801812	a76_l1d_cache_refill=19515599509	a55_inst_share=0.0016	a76_ipc=2.080	l2ref_pki=1.994	l3ref_pki=4.044	l1dref_pki=3.687	memacc_pki=167.96	dtlbw_pki=0.5397	pmu_enabled=100.0	pmu_cpu_s=3963.8	who=at_s=6,rss_pg=1509892,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6965	l2color_cv=0.0074	l3color_cv=0.0136	contig_frac=0.8328	mean_run=5.9	vapa16=0.0357	gib_regions=28	anon_pg=23041	anon_l2cv=0.0018	anon_contig=0.9027	anon_run=10.1	anon_gib=27	file_pg=24576	file_l2cv=0.0062	file_contig=0.9762	file_run=38.9	file_gib=28	other_pg=20850	other_l2cv=0.0252	other_contig=0.5864	other_run=2.4	other_gib=28	ro_cost_ms=85-->
<!--DATA 3	stock_r4	pp2048	20.76-->

### qwen35-9b-qres512-third-point [qres512_r1] pass 3  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-t 4 -r 1'  18:47:48  clk=600 MHz  MemAvail=31258948 kB
<!--PRED 3	qres512_r1	memavail_kb=31258948	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10290,12864,10800,8231,7145,6512,5680,4700,4347,3934,1891-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         36.81 ± 0.00 |

build: 171974745 (10558)
<!--RO 3	qres512_r1	wall_s=148	busy=115,82,117,90,4948,5306,6233,4698	busy_tot=21589	busy_little_share=0.0187	a55_cpu_cycles=14379073752	a55_inst_retired=3315419233	context_switches=3880833	a76_cpu_cycles=463927829947	a76_l3d_cache_refill=5588766781	a76_l2d_cache_refill=2826217241	cpu_migrations=9095	page_faults=10909075	a76_dtlb_walk=842869990	a76_mem_access=236006525724	a76_inst_retired=790260130219	a76_l1d_cache_refill=5664115320	a55_inst_share=0.0042	a76_ipc=1.703	l2ref_pki=3.576	l3ref_pki=7.072	l1dref_pki=7.167	memacc_pki=298.64	dtlbw_pki=1.0666	pmu_enabled=100.0	pmu_cpu_s=1181.5	who=at_s=6,rss_pg=1501414,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6780	l2color_cv=0.0116	l3color_cv=0.0211	contig_frac=0.7819	mean_run=4.6	vapa16=0.0315	gib_regions=28	anon_pg=4608	anon_l2cv=0.0157	anon_contig=0.7358	anon_run=3.8	anon_gib=27	file_pg=24576	file_l2cv=0.0049	file_contig=0.9768	file_run=39.9	file_gib=27	other_pg=20805	other_l2cv=0.0293	other_contig=0.5618	other_run=2.3	other_gib=28	ro_cost_ms=76-->
<!--DATA 3	qres512_r1	pp2048	36.81-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20933MB (MemAvailable 30469MB - reserve 9535MB, no swap)

### qwen35-9b-qres512-third-point [qres512_r2] pass 3  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-t 4 -r 2'  18:51:44  clk=600 MHz  MemAvail=31269664 kB
<!--PRED 3	qres512_r2	memavail_kb=31269664	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10052,12425,11848,8545,7402,6746,5785,4780,4476,4002,1794-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         36.39 ± 0.22 |

build: 171974745 (10558)
<!--RO 3	qres512_r2	wall_s=206	busy=161,123,174,157,6931,6938,7387,7398	busy_tot=29269	busy_little_share=0.0210	a55_cpu_cycles=21041113016	a55_inst_retired=4706719936	context_switches=5817041	a76_cpu_cycles=629853728305	a76_l3d_cache_refill=7582414538	a76_l2d_cache_refill=4083833153	cpu_migrations=14324	page_faults=10915455	a76_dtlb_walk=1538907123	a76_mem_access=332527211892	a76_inst_retired=1046988326529	a76_l1d_cache_refill=8215785009	a55_inst_share=0.0045	a76_ipc=1.662	l2ref_pki=3.901	l3ref_pki=7.242	l1dref_pki=7.847	memacc_pki=317.60	dtlbw_pki=1.4698	pmu_enabled=100.0	pmu_cpu_s=1643.5	who=at_s=6,rss_pg=1501419,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6816	l2color_cv=0.0046	l3color_cv=0.0179	contig_frac=0.7655	mean_run=4.2	vapa16=0.0154	gib_regions=28	anon_pg=4608	anon_l2cv=0.0168	anon_contig=0.8367	anon_run=6.1	anon_gib=25	file_pg=24576	file_l2cv=0.0034	file_contig=0.9786	file_run=43.0	file_gib=28	other_pg=21066	other_l2cv=0.0098	other_contig=0.5011	other_run=2.0	other_gib=28	ro_cost_ms=65-->
<!--DATA 3	qres512_r2	pp2048	36.39-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20925MB (MemAvailable 30460MB - reserve 9535MB, no swap)

### qwen35-9b-qres512-third-point [qres512_r4] pass 3  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-t 4 -r 4'  18:57:07  clk=600 MHz  MemAvail=31457228 kB
<!--PRED 3	qres512_r4	memavail_kb=31457228	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9218,12943,11849,8481,7470,6892,5888,4849,4666,4012,1767-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         37.12 ± 0.14 |

build: 171974745 (10558)
<!--RO 3	qres512_r4	wall_s=312	busy=270,229,196,270,11007,9780,11505,11491	busy_tot=44748	busy_little_share=0.0216	a55_cpu_cycles=34297127565	a55_inst_retired=7629942759	context_switches=9662951	a76_cpu_cycles=965885883160	a76_l3d_cache_refill=12090979851	a76_l2d_cache_refill=6594174599	cpu_migrations=23828	page_faults=10932863	a76_dtlb_walk=2528804949	a76_mem_access=525862631695	a76_inst_retired=1561387915811	a76_l1d_cache_refill=13392784202	a55_inst_share=0.0049	a76_ipc=1.617	l2ref_pki=4.223	l3ref_pki=7.744	l1dref_pki=8.577	memacc_pki=336.79	dtlbw_pki=1.6196	pmu_enabled=100.0	pmu_cpu_s=2497.1	who=at_s=6,rss_pg=1497730,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6810	l2color_cv=0.0116	l3color_cv=0.0284	contig_frac=0.7819	mean_run=4.6	vapa16=0.0464	gib_regions=28	anon_pg=4608	anon_l2cv=0.0055	anon_contig=0.8424	anon_run=6.3	anon_gib=23	file_pg=24576	file_l2cv=0.0025	file_contig=0.9803	file_run=46.4	file_gib=27	other_pg=21022	other_l2cv=0.0276	other_contig=0.5366	other_run=2.2	other_gib=28	ro_cost_ms=74-->
<!--DATA 3	qres512_r4	pp2048	37.12-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21095MB (MemAvailable 30630MB - reserve 9535MB, no swap)

### qwen35-9b-qres512-third-point [stock_u_r2] pass 3  env=''  args='-r 2'  19:03:54  clk=600 MHz  MemAvail=31484560 kB
<!--PRED 3	stock_u_r2	memavail_kb=31484560	anonhuge_kb=0	hugepagesz_kb=2048	buddy=11073,12191,11092,7543,6445,6157,5227,4289,4311,3677,2190-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         19.02 ± 0.07 |

build: 171974745 (10558)
<!--RO 3	stock_u_r2	wall_s=327	busy=5365,4167,4127,3952,18715,16787,16596,16941	busy_tot=86650	busy_little_share=0.2032	a55_cpu_cycles=324412460559	a55_inst_retired=168754234848	context_switches=5970055	a76_cpu_cycles=1491181802235	a76_l3d_cache_refill=11806374766	a76_l2d_cache_refill=6165347947	cpu_migrations=29756	page_faults=582119	a76_dtlb_walk=1651158054	a76_mem_access=493413712285	a76_inst_retired=3097691977120	a76_l1d_cache_refill=10244348404	a55_inst_share=0.0517	a76_ipc=2.077	l2ref_pki=1.990	l3ref_pki=3.811	l1dref_pki=3.307	memacc_pki=159.28	dtlbw_pki=0.5330	pmu_enabled=100.0	pmu_cpu_s=2604.4	who=at_s=6,rss_pg=1509914,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6982	l2color_cv=0.0065	l3color_cv=0.0163	contig_frac=0.8497	mean_run=6.6	vapa16=0.0372	gib_regions=28	anon_pg=22873	anon_l2cv=0.0012	anon_contig=0.9415	anon_run=16.6	anon_gib=28	file_pg=24576	file_l2cv=0.0059	file_contig=0.9759	file_run=38.5	file_gib=27	other_pg=21188	other_l2cv=0.0157	other_contig=0.6041	other_run=2.5	other_gib=28	ro_cost_ms=67-->
<!--DATA 3	stock_u_r2	pp2048	19.02-->

### qwen35-9b-qres512-third-point [qres512_u_r2] pass 3  env='ROCKET_QUANT_RESIDENT=auto'  args='-r 2'  19:10:54  clk=600 MHz  MemAvail=31464140 kB
<!--PRED 3	qres512_u_r2	memavail_kb=31464140	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10616,12900,12065,8345,7213,6682,5726,4716,4540,3938,1872-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         33.91 ± 0.15 |

build: 171974745 (10558)
<!--RO 3	qres512_u_r2	wall_s=218	busy=3951,3888,3911,3838,6552,6040,7200,7036	busy_tot=42416	busy_little_share=0.3675	a55_cpu_cycles=283046874296	a55_inst_retired=154405612074	context_switches=5910670	a76_cpu_cycles=595729676262	a76_l3d_cache_refill=6979263678	a76_l2d_cache_refill=4007851366	cpu_migrations=32006	page_faults=10912413	a76_dtlb_walk=1516688684	a76_mem_access=295350390519	a76_inst_retired=981506977110	a76_l1d_cache_refill=7275315244	a55_inst_share=0.1359	a76_ipc=1.648	l2ref_pki=4.083	l3ref_pki=7.111	l1dref_pki=7.412	memacc_pki=300.92	dtlbw_pki=1.5453	pmu_enabled=100.0	pmu_cpu_s=1737.2	who=at_s=6,rss_pg=1485065,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=90112	present_frac=0.7334	l2color_cv=0.0120	l3color_cv=0.0166	contig_frac=0.8318	mean_run=5.9	vapa16=0.0222	gib_regions=28	anon_pg=20880	anon_l2cv=0.0041	anon_contig=0.9810	anon_run=47.9	anon_gib=28	file_pg=24576	file_l2cv=0.0029	file_contig=0.9765	file_run=39.4	file_gib=27	other_pg=20633	other_l2cv=0.0375	other_contig=0.5082	other_run=2.0	other_gib=28	ro_cost_ms=76-->
<!--DATA 3	qres512_u_r2	pp2048	33.91-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21079MB (MemAvailable 30614MB - reserve 9535MB, no swap)

### qwen35-9b-qres512-third-point [stock_r1] pass 3  env='PIN_MASK=0xf0'  args='-t 4 -r 1'  19:15:31  clk=600 MHz  MemAvail=31438776 kB
<!--PRED 3	stock_r1	memavail_kb=31438776	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9183,12162,11437,7997,6686,6416,5496,4505,4406,3836,2016-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         21.02 ± 0.00 |

build: 171974745 (10558)
<!--RO 3	stock_r1	wall_s=198	busy=90,141,78,98,12523,12157,10947,11337	busy_tot=47371	busy_little_share=0.0086	a55_cpu_cycles=16652129708	a55_inst_retired=3421949910	context_switches=3924690	a76_cpu_cycles=1018677121226	a76_l3d_cache_refill=8120892310	a76_l2d_cache_refill=4301224783	cpu_migrations=9333	page_faults=567188	a76_dtlb_walk=1147984883	a76_mem_access=357389917042	a76_inst_retired=2122478084038	a76_l1d_cache_refill=7857276391	a55_inst_share=0.0016	a76_ipc=2.084	l2ref_pki=2.027	l3ref_pki=3.826	l1dref_pki=3.702	memacc_pki=168.38	dtlbw_pki=0.5409	pmu_enabled=100.0	pmu_cpu_s=1572.5	who=at_s=6,rss_pg=1509886,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6993	l2color_cv=0.0058	l3color_cv=0.0124	contig_frac=0.8487	mean_run=6.5	vapa16=0.0116	gib_regions=28	anon_pg=23041	anon_l2cv=0.0033	anon_contig=0.8614	anon_run=7.1	anon_gib=26	file_pg=24576	file_l2cv=0.0042	file_contig=0.9812	file_run=48.3	file_gib=27	other_pg=21127	other_l2cv=0.0220	other_contig=0.6807	other_run=3.1	other_gib=28	ro_cost_ms=127-->
<!--DATA 3	stock_r1	pp2048	21.02-->

### qwen35-9b-qres512-third-point [stock_r2] pass 3  env='PIN_MASK=0xf0'  args='-t 4 -r 2'  19:20:14  clk=600 MHz  MemAvail=31453280 kB
<!--PRED 3	stock_r2	memavail_kb=31452988	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8854,11785,11288,7595,6317,5868,5092,4170,4151,3596,2298-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.64 ± 0.04 |

build: 171974745 (10558)
<!--RO 3	stock_r2	wall_s=300	busy=149,164,146,156,18670,18443,16522,16974	busy_tot=71224	busy_little_share=0.0086	a55_cpu_cycles=25125234307	a55_inst_retired=5200433707	context_switches=5877645	a76_cpu_cycles=1528749826941	a76_l3d_cache_refill=12783682583	a76_l2d_cache_refill=6301937245	cpu_migrations=13910	page_faults=580108	a76_dtlb_walk=1714876250	a76_mem_access=534493050021	a76_inst_retired=3179215162696	a76_l1d_cache_refill=11631990925	a55_inst_share=0.0016	a76_ipc=2.080	l2ref_pki=1.982	l3ref_pki=4.021	l1dref_pki=3.659	memacc_pki=168.12	dtlbw_pki=0.5394	pmu_enabled=100.0	pmu_cpu_s=2395.9	who=at_s=6,rss_pg=1509887,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6973	l2color_cv=0.0118	l3color_cv=0.0177	contig_frac=0.7949	mean_run=4.8	vapa16=0.0307	gib_regions=28	anon_pg=23041	anon_l2cv=0.0044	anon_contig=0.7735	anon_run=4.4	anon_gib=28	file_pg=24576	file_l2cv=0.0014	file_contig=0.9790	file_run=43.7	file_gib=27	other_pg=20927	other_l2cv=0.0335	other_contig=0.6023	other_run=2.5	other_gib=28	ro_cost_ms=90-->
<!--DATA 3	stock_r2	pp2048	20.64-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock_r1]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock_r1 | 20.98 | 3 | 20.88 | 21.05 | -- | -- |
| pp2048 | stock_r2 | 20.89 | 3 | 20.64 | 21.05 | 0.995x | 1.008 0.996 0.982 |
| pp2048 | stock_r4 | 20.83 | 3 | 20.76 | 20.88 | 0.993x | 0.999 0.992 0.988 |
| pp2048 | qres512_r1 | 36.94 | 3 | 36.77 | 37.24 | 1.761x | 1.784 1.747 1.751 |
| pp2048 | qres512_r2 | 36.31 | 3 | 36.25 | 36.39 | 1.730x | 1.738 1.722 1.731 |
| pp2048 | qres512_r4 | 37.51 | 3 | 37.12 | 38.11 | 1.788x | 1.825 1.771 1.766 |
| pp2048 | stock_u_r2 | 19.10 | 3 | 19.02 | 19.22 | 0.910x | 0.912 0.913 0.905 |
| pp2048 | qres512_u_r2 | 33.54 | 3 | 33.00 | 33.91 | 1.598x | 1.614 1.568 1.613 |

