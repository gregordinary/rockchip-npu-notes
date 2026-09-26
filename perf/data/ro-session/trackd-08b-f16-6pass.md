<!-- qwen35-08b-f16  class=f16  fp16-resident-fits=1  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=<data>/qwen35/Qwen3.5-0.8B-F16.gguf (1516744736 bytes)
-->
== qwen35-08b-f16  Mon Aug 31 21:42:05 UTC 2026 ==
### qwen35-08b-f16 [f16-stock] pass 1  env=''  args=''  21:42:19  clk=600 MHz  MemAvail=31463660 kB
<!--PRED 1	f16-stock	memavail_kb=31463660	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5514,6630,5936,5316,4364,4099,3537,3203,2677,2559,4525-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        131.23 ± 0.35 |

build: 171974745 (10558)
<!--RO 1	f16-stock	wall_s=66	busy=2122,2156,2151,2128,2837,2337,2806,2710	busy_tot=19247	busy_little_share=0.4446	a55_cpu_cycles=152757788045	a55_inst_retired=93982283370	context_switches=925053	a76_cpu_cycles=235125739607	a76_l3d_cache_refill=2947053493	a76_l2d_cache_refill=2093782614	cpu_migrations=28219	page_faults=975298	a76_dtlb_walk=924355003	a76_mem_access=128314937419	a76_inst_retired=367280639286	a76_l1d_cache_refill=3893714560	a55_inst_share=0.2037	a76_ipc=1.562	l2ref_pki=5.701	l3ref_pki=8.024	l1dref_pki=10.601	memacc_pki=349.36	dtlbw_pki=2.5168	pmu_enabled=100.0	pmu_cpu_s=520.8	who=at_s=6,rss_pg=426527,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6454	l2color_cv=0.0038	l3color_cv=0.0253	contig_frac=0.8888	mean_run=8.8	vapa16=0.0378	gib_regions=28	anon_pg=2055	anon_l2cv=0.0792	anon_contig=0.5659	anon_run=2.3	anon_gib=28	file_pg=24576	file_l2cv=0.0012	file_contig=0.9887	file_run=75.4	file_gib=28	other_pg=20956	other_l2cv=0.0058	other_contig=0.8033	other_run=5.0	other_gib=28	ro_cost_ms=65-->
<!--DATA 1	f16-stock	pp2048	131.23-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16 [f16-res] pass 1  env='ROCKET_F16_RESIDENT=auto'  args=''  21:43:39  clk=600 MHz  MemAvail=31295892 kB
<!--PRED 1	f16-res	memavail_kb=31295892	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5869,6022,5718,5019,3259,3603,3080,2863,2384,2287,4803-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        117.82 ± 0.05 |

build: 171974745 (10558)
<!--RO 1	f16-res	wall_s=74	busy=2180,2197,2206,2153,2751,2475,3279,2796	busy_tot=20037	busy_little_share=0.4360	a55_cpu_cycles=156743825780	a55_inst_retired=95924342969	context_switches=925900	a76_cpu_cycles=247603641575	a76_l3d_cache_refill=3693502070	a76_l2d_cache_refill=2063640265	cpu_migrations=28658	page_faults=1069005	a76_dtlb_walk=1099900077	a76_mem_access=126953834533	a76_inst_retired=365239616188	a76_l1d_cache_refill=3879044353	a55_inst_share=0.2080	a76_ipc=1.475	l2ref_pki=5.650	l3ref_pki=10.113	l1dref_pki=10.621	memacc_pki=347.59	dtlbw_pki=3.0114	pmu_enabled=100.0	pmu_cpu_s=580.7	who=at_s=6,rss_pg=426148,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6406	l2color_cv=0.0032	l3color_cv=0.0146	contig_frac=0.8654	mean_run=7.3	vapa16=0.0507	gib_regions=28	anon_pg=2048	anon_l2cv=0.0533	anon_contig=0.5895	anon_run=2.4	anon_gib=28	file_pg=24576	file_l2cv=0.0012	file_contig=0.9879	file_run=71.4	file_gib=28	other_pg=20605	other_l2cv=0.0069	other_contig=0.7467	other_run=3.9	other_gib=27	ro_cost_ms=50-->
<!--DATA 1	f16-res	pp2048	117.82-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 20936MB (MemAvailable 30472MB - reserve 9535MB, no swap)

### qwen35-08b-f16 [f16-res] pass 2  env='ROCKET_F16_RESIDENT=auto'  args=''  21:45:07  clk=600 MHz  MemAvail=31313628 kB
<!--PRED 2	f16-res	memavail_kb=31313628	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6036,5730,5630,4997,3408,3680,3074,2751,2307,2113,4924-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        117.40 ± 0.13 |

build: 171974745 (10558)
<!--RO 2	f16-res	wall_s=73	busy=2169,2206,2224,2213,2812,2474,2929,3063	busy_tot=20090	busy_little_share=0.4386	a55_cpu_cycles=158156042733	a55_inst_retired=97079461774	context_switches=922595	a76_cpu_cycles=246468991034	a76_l3d_cache_refill=3685100752	a76_l2d_cache_refill=2060276199	cpu_migrations=27882	page_faults=1070107	a76_dtlb_walk=1107132580	a76_mem_access=126163001276	a76_inst_retired=362174084279	a76_l1d_cache_refill=3871691665	a55_inst_share=0.2114	a76_ipc=1.469	l2ref_pki=5.689	l3ref_pki=10.175	l1dref_pki=10.690	memacc_pki=348.35	dtlbw_pki=3.0569	pmu_enabled=100.0	pmu_cpu_s=581.2	who=at_s=6,rss_pg=426152,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6434	l2color_cv=0.0032	l3color_cv=0.0111	contig_frac=0.8948	mean_run=9.3	vapa16=0.0113	gib_regions=28	anon_pg=2055	anon_l2cv=0.0374	anon_contig=0.6156	anon_run=2.6	anon_gib=28	file_pg=24576	file_l2cv=0.0013	file_contig=0.9786	file_run=42.8	file_gib=28	other_pg=20809	other_l2cv=0.0050	other_contig=0.8234	other_run=5.6	other_gib=28	ro_cost_ms=57-->
<!--DATA 2	f16-res	pp2048	117.40-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 20984MB (MemAvailable 30520MB - reserve 9535MB, no swap)

### qwen35-08b-f16 [f16-stock] pass 2  env=''  args=''  21:46:33  clk=600 MHz  MemAvail=31393420 kB
<!--PRED 2	f16-stock	memavail_kb=31393420	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4277,6657,4741,2914,4519,3690,3035,2759,2295,2091,4961-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        115.56 ± 0.15 |

build: 171974745 (10558)
<!--RO 2	f16-stock	wall_s=74	busy=2214,2212,2240,2211,2912,2659,2923,2860	busy_tot=20231	busy_little_share=0.4388	a55_cpu_cycles=158899124852	a55_inst_retired=97184764991	context_switches=927019	a76_cpu_cycles=249122579105	a76_l3d_cache_refill=3742317260	a76_l2d_cache_refill=2101263592	cpu_migrations=27880	page_faults=976161	a76_dtlb_walk=1106880949	a76_mem_access=126390146044	a76_inst_retired=363115629747	a76_l1d_cache_refill=3912936444	a55_inst_share=0.2111	a76_ipc=1.458	l2ref_pki=5.787	l3ref_pki=10.306	l1dref_pki=10.776	memacc_pki=348.07	dtlbw_pki=3.0483	pmu_enabled=100.0	pmu_cpu_s=586.9	who=at_s=6,rss_pg=426147,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6397	l2color_cv=0.0028	l3color_cv=0.0127	contig_frac=0.8969	mean_run=9.5	vapa16=0.0162	gib_regions=28	anon_pg=2055	anon_l2cv=0.0577	anon_contig=0.6137	anon_run=2.6	anon_gib=25	file_pg=24576	file_l2cv=0.0003	file_contig=0.9913	file_run=93.8	file_gib=25	other_pg=20536	other_l2cv=0.0075	other_contig=0.8122	other_run=5.3	other_gib=28	ro_cost_ms=69-->
<!--DATA 2	f16-stock	pp2048	115.56-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16 [f16-stock] pass 3  env=''  args=''  21:48:01  clk=600 MHz  MemAvail=31336696 kB
<!--PRED 3	f16-stock	memavail_kb=31336696	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5515,6018,5533,2350,3977,3558,2944,2775,2284,2060,4983-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        126.57 ± 0.16 |

build: 171974745 (10558)
<!--RO 3	f16-stock	wall_s=69	busy=2144,2170,2218,2132,2915,2384,2725,2758	busy_tot=19446	busy_little_share=0.4455	a55_cpu_cycles=154643925611	a55_inst_retired=94947924286	context_switches=924261	a76_cpu_cycles=237897367093	a76_l3d_cache_refill=3073962516	a76_l2d_cache_refill=2094586149	cpu_migrations=28355	page_faults=975248	a76_dtlb_walk=962638914	a76_mem_access=127699406846	a76_inst_retired=365886355684	a76_l1d_cache_refill=3910634433	a55_inst_share=0.2060	a76_ipc=1.538	l2ref_pki=5.725	l3ref_pki=8.401	l1dref_pki=10.688	memacc_pki=349.01	dtlbw_pki=2.6310	pmu_enabled=100.0	pmu_cpu_s=539.2	who=at_s=6,rss_pg=426527,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6445	l2color_cv=0.0028	l3color_cv=0.0092	contig_frac=0.9134	mean_run=11.3	vapa16=0.0939	gib_regions=28	anon_pg=2055	anon_l2cv=0.0396	anon_contig=0.3561	anon_run=1.6	anon_gib=28	file_pg=24576	file_l2cv=0.0007	file_contig=0.9915	file_run=95.6	file_gib=26	other_pg=20890	other_l2cv=0.0054	other_contig=0.8764	other_run=8.0	other_gib=28	ro_cost_ms=50-->
<!--DATA 3	f16-stock	pp2048	126.57-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16 [f16-res] pass 3  env='ROCKET_F16_RESIDENT=auto'  args=''  21:49:23  clk=600 MHz  MemAvail=31404980 kB
<!--PRED 3	f16-res	memavail_kb=31404980	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5075,6032,5741,4868,3846,3340,3122,2745,2282,2039,4992-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        117.54 ± 0.56 |

build: 171974745 (10558)
<!--RO 3	f16-res	wall_s=74	busy=2175,2197,2233,2228,2805,2370,2860,3116	busy_tot=19984	busy_little_share=0.4420	a55_cpu_cycles=158132638434	a55_inst_retired=96746387993	context_switches=926441	a76_cpu_cycles=246705567784	a76_l3d_cache_refill=3666712976	a76_l2d_cache_refill=2059082166	cpu_migrations=28393	page_faults=1070096	a76_dtlb_walk=640164251	a76_mem_access=126157277496	a76_inst_retired=362608651276	a76_l1d_cache_refill=3885351091	a55_inst_share=0.2106	a76_ipc=1.470	l2ref_pki=5.679	l3ref_pki=10.112	l1dref_pki=10.715	memacc_pki=347.92	dtlbw_pki=1.7654	pmu_enabled=100.0	pmu_cpu_s=581.2	who=at_s=6,rss_pg=426147,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6420	l2color_cv=0.0054	l3color_cv=0.0215	contig_frac=0.6616	mean_run=2.9	vapa16=0.0216	gib_regions=28	anon_pg=2048	anon_l2cv=0.0437	anon_contig=0.7617	anon_run=4.2	anon_gib=25	file_pg=24576	file_l2cv=0.0010	file_contig=0.9906	file_run=88.1	file_gib=27	other_pg=20710	other_l2cv=0.0137	other_contig=0.2612	other_run=1.4	other_gib=28	ro_cost_ms=55-->
<!--DATA 3	f16-res	pp2048	117.54-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21129MB (MemAvailable 30665MB - reserve 9535MB, no swap)

### qwen35-08b-f16 [f16-res] pass 4  env='ROCKET_F16_RESIDENT=auto'  args=''  21:50:50  clk=600 MHz  MemAvail=31349552 kB
<!--PRED 4	f16-res	memavail_kb=31349552	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5380,6646,5798,4060,3829,3660,3126,2647,2268,2025,4996-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        130.27 ± 0.41 |

build: 171974745 (10558)
<!--RO 4	f16-res	wall_s=66	busy=2127,2164,2155,2140,2803,2395,2786,2638	busy_tot=19208	busy_little_share=0.4470	a55_cpu_cycles=153740275797	a55_inst_retired=94663577521	context_switches=925352	a76_cpu_cycles=233716966903	a76_l3d_cache_refill=2936853460	a76_l2d_cache_refill=2062441999	cpu_migrations=28662	page_faults=1070186	a76_dtlb_walk=923233311	a76_mem_access=127817599340	a76_inst_retired=366701390655	a76_l1d_cache_refill=3874611840	a55_inst_share=0.2052	a76_ipc=1.569	l2ref_pki=5.624	l3ref_pki=8.009	l1dref_pki=10.566	memacc_pki=348.56	dtlbw_pki=2.5177	pmu_enabled=100.0	pmu_cpu_s=526.8	who=at_s=6,rss_pg=426144,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6461	l2color_cv=0.0054	l3color_cv=0.0104	contig_frac=0.7582	mean_run=4.1	vapa16=0.0318	gib_regions=28	anon_pg=2055	anon_l2cv=0.0404	anon_contig=0.6439	anon_run=2.8	anon_gib=27	file_pg=24576	file_l2cv=0.0009	file_contig=0.9797	file_run=44.9	file_gib=27	other_pg=21007	other_l2cv=0.0093	other_contig=0.5102	other_run=2.0	other_gib=28	ro_cost_ms=54-->
<!--DATA 4	f16-res	pp2048	130.27-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21043MB (MemAvailable 30578MB - reserve 9535MB, no swap)

### qwen35-08b-f16 [f16-stock] pass 4  env=''  args=''  21:52:10  clk=600 MHz  MemAvail=31429992 kB
<!--PRED 4	f16-stock	memavail_kb=31429992	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5899,6597,6166,3459,4540,3781,3068,2651,2303,2027,4997-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        116.81 ± 0.45 |

build: 171974745 (10558)
<!--RO 4	f16-stock	wall_s=73	busy=2156,2190,2236,2241,2658,2732,2962,2921	busy_tot=20096	busy_little_share=0.4390	a55_cpu_cycles=158465694134	a55_inst_retired=97116445389	context_switches=921958	a76_cpu_cycles=248173202275	a76_l3d_cache_refill=3747007104	a76_l2d_cache_refill=2096646777	cpu_migrations=27231	page_faults=979095	a76_dtlb_walk=1108869228	a76_mem_access=126287928621	a76_inst_retired=362106985899	a76_l1d_cache_refill=3908102734	a55_inst_share=0.2115	a76_ipc=1.459	l2ref_pki=5.790	l3ref_pki=10.348	l1dref_pki=10.793	memacc_pki=348.76	dtlbw_pki=3.0623	pmu_enabled=100.0	pmu_cpu_s=582.6	who=at_s=6,rss_pg=426147,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6426	l2color_cv=0.0091	l3color_cv=0.0218	contig_frac=0.8723	mean_run=7.7	vapa16=0.0305	gib_regions=28	anon_pg=2048	anon_l2cv=0.0509	anon_contig=0.5612	anon_run=2.3	anon_gib=28	file_pg=24576	file_l2cv=0.0012	file_contig=0.9908	file_run=89.7	file_gib=27	other_pg=20756	other_l2cv=0.0223	other_contig=0.7627	other_run=4.2	other_gib=28	ro_cost_ms=51-->
<!--DATA 4	f16-stock	pp2048	116.81-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16 [f16-stock] pass 5  env=''  args=''  21:53:37  clk=600 MHz  MemAvail=31405268 kB
<!--PRED 5	f16-stock	memavail_kb=31404976	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5455,6778,5898,3470,4416,3653,2912,2733,2262,2032,5005-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        128.42 ± 0.46 |

build: 171974745 (10558)
<!--RO 5	f16-stock	wall_s=67	busy=2140,2161,2170,2100,2746,2372,2963,2669	busy_tot=19321	busy_little_share=0.4436	a55_cpu_cycles=153547165290	a55_inst_retired=94484481841	context_switches=923391	a76_cpu_cycles=236730731300	a76_l3d_cache_refill=3066211146	a76_l2d_cache_refill=2093305208	cpu_migrations=28193	page_faults=975227	a76_dtlb_walk=561723704	a76_mem_access=128111217563	a76_inst_retired=367091450705	a76_l1d_cache_refill=3893675905	a55_inst_share=0.2047	a76_ipc=1.551	l2ref_pki=5.702	l3ref_pki=8.353	l1dref_pki=10.607	memacc_pki=348.99	dtlbw_pki=1.5302	pmu_enabled=100.0	pmu_cpu_s=530.7	who=at_s=6,rss_pg=426533,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6423	l2color_cv=0.0063	l3color_cv=0.0213	contig_frac=0.8530	mean_run=6.7	vapa16=0.0238	gib_regions=28	anon_pg=2048	anon_l2cv=0.0696	anon_contig=0.5685	anon_run=2.3	anon_gib=28	file_pg=24576	file_l2cv=0.0018	file_contig=0.9905	file_run=87.1	file_gib=27	other_pg=20731	other_l2cv=0.0137	other_contig=0.7180	other_run=3.5	other_gib=28	ro_cost_ms=73-->
<!--DATA 5	f16-stock	pp2048	128.42-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16 [f16-res] pass 5  env='ROCKET_F16_RESIDENT=auto'  args=''  21:54:58  clk=600 MHz  MemAvail=31392332 kB
<!--PRED 5	f16-res	memavail_kb=31392332	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5378,6677,6099,5644,3957,3440,3055,2664,2259,2015,5007-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        117.23 ± 0.27 |

build: 171974745 (10558)
<!--RO 5	f16-res	wall_s=74	busy=2236,2208,2255,2210,2981,2696,2620,2913	busy_tot=20119	busy_little_share=0.4428	a55_cpu_cycles=159146217328	a55_inst_retired=97193638405	context_switches=923699	a76_cpu_cycles=246470408577	a76_l3d_cache_refill=3684802695	a76_l2d_cache_refill=2056562896	cpu_migrations=27662	page_faults=1069829	a76_dtlb_walk=1107505679	a76_mem_access=126183644615	a76_inst_retired=362427791406	a76_l1d_cache_refill=3876844662	a55_inst_share=0.2115	a76_ipc=1.470	l2ref_pki=5.674	l3ref_pki=10.167	l1dref_pki=10.697	memacc_pki=348.16	dtlbw_pki=3.0558	pmu_enabled=100.0	pmu_cpu_s=582.4	who=at_s=6,rss_pg=426150,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6415	l2color_cv=0.0059	l3color_cv=0.0139	contig_frac=0.8736	mean_run=7.8	vapa16=0.0950	gib_regions=28	anon_pg=1968	anon_l2cv=0.0602	anon_contig=0.6083	anon_run=2.5	anon_gib=28	file_pg=24576	file_l2cv=0.0008	file_contig=0.9913	file_run=93.8	file_gib=25	other_pg=20756	other_l2cv=0.0116	other_contig=0.7595	other_run=4.1	other_gib=28	ro_cost_ms=57-->
<!--DATA 5	f16-res	pp2048	117.23-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21028MB (MemAvailable 30563MB - reserve 9535MB, no swap)

### qwen35-08b-f16 [f16-res] pass 6  env='ROCKET_F16_RESIDENT=auto'  args=''  21:56:26  clk=600 MHz  MemAvail=31287852 kB
<!--PRED 6	f16-res	memavail_kb=31287852	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5886,5630,5723,3317,3040,3578,3130,2638,2267,2018,5008-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        118.30 ± 0.05 |

build: 171974745 (10558)
<!--RO 6	f16-res	wall_s=72	busy=2182,2186,2211,2212,2829,2515,2888,2906	busy_tot=19929	busy_little_share=0.4411	a55_cpu_cycles=157401099781	a55_inst_retired=96435391936	context_switches=924646	a76_cpu_cycles=245438558147	a76_l3d_cache_refill=3619876917	a76_l2d_cache_refill=2035161233	cpu_migrations=28474	page_faults=1069346	a76_dtlb_walk=1074165154	a76_mem_access=126585986854	a76_inst_retired=363784253711	a76_l1d_cache_refill=3874913387	a55_inst_share=0.2095	a76_ipc=1.482	l2ref_pki=5.594	l3ref_pki=9.951	l1dref_pki=10.652	memacc_pki=347.97	dtlbw_pki=2.9528	pmu_enabled=100.0	pmu_cpu_s=576.7	who=at_s=6,rss_pg=426147,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6424	l2color_cv=0.0027	l3color_cv=0.0079	contig_frac=0.9221	mean_run=12.5	vapa16=0.0189	gib_regions=28	anon_pg=1984	anon_l2cv=0.0484	anon_contig=0.6918	anon_run=3.2	anon_gib=28	file_pg=24576	file_l2cv=0.0009	file_contig=0.9908	file_run=89.7	file_gib=24	other_pg=20800	other_l2cv=0.0051	other_contig=0.8629	other_run=7.2	other_gib=28	ro_cost_ms=50-->
<!--DATA 6	f16-res	pp2048	118.30-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 20947MB (MemAvailable 30482MB - reserve 9535MB, no swap)

### qwen35-08b-f16 [f16-stock] pass 6  env=''  args=''  21:57:52  clk=600 MHz  MemAvail=31406312 kB
<!--PRED 6	f16-stock	memavail_kb=31406312	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6065,6404,5727,3359,4268,3920,3037,2631,2267,2025,5008-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        122.03 ± 0.37 |

build: 171974745 (10558)
<!--RO 6	f16-stock	wall_s=71	busy=2177,2184,2217,2194,2914,2546,2833,2698	busy_tot=19763	busy_little_share=0.4439	a55_cpu_cycles=157492774619	a55_inst_retired=96638045232	context_switches=926409	a76_cpu_cycles=242703773593	a76_l3d_cache_refill=3404226468	a76_l2d_cache_refill=2092650381	cpu_migrations=28285	page_faults=975686	a76_dtlb_walk=976671525	a76_mem_access=126938539424	a76_inst_retired=364612552466	a76_l1d_cache_refill=3922098279	a55_inst_share=0.2095	a76_ipc=1.502	l2ref_pki=5.739	l3ref_pki=9.337	l1dref_pki=10.757	memacc_pki=348.15	dtlbw_pki=2.6787	pmu_enabled=100.0	pmu_cpu_s=557.9	who=at_s=6,rss_pg=426527,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6398	l2color_cv=0.0050	l3color_cv=0.0098	contig_frac=0.8741	mean_run=7.8	vapa16=0.0285	gib_regions=27	anon_pg=2048	anon_l2cv=0.0361	anon_contig=0.5778	anon_run=2.4	anon_gib=27	file_pg=24576	file_l2cv=0.0013	file_contig=0.9878	file_run=70.6	file_gib=26	other_pg=20545	other_l2cv=0.0113	other_contig=0.7677	other_run=4.3	other_gib=27	ro_cost_ms=49-->
<!--DATA 6	f16-stock	pp2048	122.03-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

#### summary: per-arm mean over 6 passes, ratios paired within a pass against [f16-stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | f16-stock | 123.44 | 6 | 115.56 | 131.23 | -- | -- |
| pp2048 | f16-res | 119.76 | 6 | 117.23 | 130.27 | 0.973x | 0.898 1.016 0.929 1.115 0.913 0.969 |

