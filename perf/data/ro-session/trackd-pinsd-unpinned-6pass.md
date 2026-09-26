<!-- qwen35-08b-f16 protocol campaign 'unpinned'  TESTS='-p 2048 -n 0 -r 3'  PASSES=6
     gguf=<data>/qwen35/Qwen3.5-0.8B-F16.gguf (1516744736 bytes)
     arms: stock | f16res = ROCKET_F16_RESIDENT=auto
     campaign 'pinned' applies PIN_MASK=0xf0 -t 4 to BOTH arms, so pinning is not a
     difference WITHIN a campaign; what is compared across the two campaigns is the
     per-pass paired-ratio sd, NOT the level and NOT the arm spread.
-->
== qwen35-08b-f16-unpinned  Tue Sep  1 04:21:58 UTC 2026 ==
### qwen35-08b-f16-unpinned [stock] pass 1  env=''  args=''  04:22:13  clk=600 MHz  MemAvail=31432024 kB
<!--PRED 1	stock	memavail_kb=31432024	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7567,6495,7643,4953,3639,4022,3299,2874,2902,2448,4604-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        127.81 ± 0.50 |

build: 171974745 (10558)
<!--RO 1	stock	wall_s=67	busy=2163,2174,2196,2132,2713,2364,2784,2910	busy_tot=19436	busy_little_share=0.4458	a55_cpu_cycles=154549772036	a55_inst_retired=94964923612	context_switches=925827	a76_cpu_cycles=237357896562	a76_l3d_cache_refill=3071880129	a76_l2d_cache_refill=2108146435	cpu_migrations=28174	page_faults=976167	a76_dtlb_walk=953335275	a76_mem_access=127771981710	a76_inst_retired=366021641194	a76_l1d_cache_refill=3905131063	a55_inst_share=0.2060	a76_ipc=1.542	l2ref_pki=5.760	l3ref_pki=8.393	l1dref_pki=10.669	memacc_pki=349.08	dtlbw_pki=2.6046	pmu_enabled=100.0	pmu_cpu_s=532.8	who=at_s=6,rss_pg=426549,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6397	l2color_cv=0.0568	l3color_cv=0.0675	contig_frac=0.8110	mean_run=5.2	vapa16=0.0115	gib_regions=28	anon_pg=2055	anon_l2cv=0.2404	anon_contig=0.3020	anon_run=1.4	anon_gib=28	file_pg=24576	file_l2cv=0.0151	file_contig=0.9793	file_run=44.3	file_gib=28	other_pg=20532	other_l2cv=0.1237	other_contig=0.6604	other_run=2.9	other_gib=28	ro_cost_ms=50-->
<!--DATA 1	stock	pp2048	127.81-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-unpinned [f16res] pass 1  env='ROCKET_F16_RESIDENT=auto'  args=''  04:23:35  clk=600 MHz  MemAvail=31277800 kB
<!--PRED 1	f16res	memavail_kb=31277800	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5199,5945,5836,3559,2929,3886,2964,2747,2389,2311,4836-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        116.53 ± 0.33 |

build: 171974745 (10558)
<!--RO 1	f16res	wall_s=74	busy=2220,2221,2264,2224,2797,2521,2811,3045	busy_tot=20103	busy_little_share=0.4442	a55_cpu_cycles=159124512645	a55_inst_retired=97292185550	context_switches=924637	a76_cpu_cycles=246864853426	a76_l3d_cache_refill=3698148908	a76_l2d_cache_refill=2063847481	cpu_migrations=27899	page_faults=1076082	a76_dtlb_walk=1114298243	a76_mem_access=126309909699	a76_inst_retired=362671170990	a76_l1d_cache_refill=3870479874	a55_inst_share=0.2115	a76_ipc=1.469	l2ref_pki=5.691	l3ref_pki=10.197	l1dref_pki=10.672	memacc_pki=348.28	dtlbw_pki=3.0725	pmu_enabled=100.0	pmu_cpu_s=585.3	who=at_s=6,rss_pg=426158,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6470	l2color_cv=0.0049	l3color_cv=0.0120	contig_frac=0.8543	mean_run=6.8	vapa16=0.0389	gib_regions=28	anon_pg=2055	anon_l2cv=0.0820	anon_contig=0.5337	anon_run=2.1	anon_gib=25	file_pg=24576	file_l2cv=0.0014	file_contig=0.9861	file_run=63.0	file_gib=28	other_pg=21068	other_l2cv=0.0108	other_contig=0.7318	other_run=3.7	other_gib=27	ro_cost_ms=52-->
<!--DATA 1	f16res	pp2048	116.53-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 20908MB (MemAvailable 30444MB - reserve 9535MB, no swap)

### qwen35-08b-f16-unpinned [f16res] pass 2  env='ROCKET_F16_RESIDENT=auto'  args=''  04:25:02  clk=600 MHz  MemAvail=31325220 kB
<!--PRED 2	f16res	memavail_kb=31325220	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5685,6614,5971,4185,3613,3915,2983,2665,2304,2236,4896-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        125.62 ± 0.83 |

build: 171974745 (10558)
<!--RO 2	f16res	wall_s=69	busy=2159,2184,2204,2155,2646,2513,2928,2661	busy_tot=19450	busy_little_share=0.4474	a55_cpu_cycles=155693559810	a55_inst_retired=95633610221	context_switches=923693	a76_cpu_cycles=237961961951	a76_l3d_cache_refill=3212360210	a76_l2d_cache_refill=2057641007	cpu_migrations=28452	page_faults=1068385	a76_dtlb_walk=1003702400	a76_mem_access=127167939782	a76_inst_retired=364843697148	a76_l1d_cache_refill=3865029939	a55_inst_share=0.2077	a76_ipc=1.533	l2ref_pki=5.640	l3ref_pki=8.805	l1dref_pki=10.594	memacc_pki=348.55	dtlbw_pki=2.7510	pmu_enabled=100.0	pmu_cpu_s=544.3	who=at_s=6,rss_pg=426162,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6418	l2color_cv=0.0045	l3color_cv=0.0256	contig_frac=0.7483	mean_run=3.9	vapa16=0.0163	gib_regions=28	anon_pg=2055	anon_l2cv=0.0286	anon_contig=0.7156	anon_run=3.5	anon_gib=27	file_pg=24576	file_l2cv=0.0006	file_contig=0.9903	file_run=85.6	file_gib=28	other_pg=20691	other_l2cv=0.0094	other_contig=0.4642	other_run=1.9	other_gib=28	ro_cost_ms=54-->
<!--DATA 2	f16res	pp2048	125.62-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21004MB (MemAvailable 30539MB - reserve 9535MB, no swap)

### qwen35-08b-f16-unpinned [stock] pass 2  env=''  args=''  04:26:24  clk=600 MHz  MemAvail=31442232 kB
<!--PRED 2	stock	memavail_kb=31442232	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4687,6686,6173,3588,4344,3939,3037,2668,2338,2224,4911-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        116.31 ± 0.13 |

build: 171974745 (10558)
<!--RO 2	stock	wall_s=73	busy=2137,2204,2246,2235,2933,2558,2859,2903	busy_tot=20075	busy_little_share=0.4395	a55_cpu_cycles=158291060918	a55_inst_retired=96910091236	context_switches=926865	a76_cpu_cycles=249348933301	a76_l3d_cache_refill=3742796273	a76_l2d_cache_refill=2095163492	cpu_migrations=28826	page_faults=982316	a76_dtlb_walk=1099294621	a76_mem_access=126756440997	a76_inst_retired=364593637701	a76_l1d_cache_refill=3907977997	a55_inst_share=0.2100	a76_ipc=1.462	l2ref_pki=5.747	l3ref_pki=10.266	l1dref_pki=10.719	memacc_pki=347.66	dtlbw_pki=3.0151	pmu_enabled=100.0	pmu_cpu_s=584.3	who=at_s=6,rss_pg=426162,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6419	l2color_cv=0.0057	l3color_cv=0.0167	contig_frac=0.8637	mean_run=7.2	vapa16=0.0303	gib_regions=28	anon_pg=2055	anon_l2cv=0.0370	anon_contig=0.5576	anon_run=2.3	anon_gib=26	file_pg=24576	file_l2cv=0.0010	file_contig=0.9785	file_run=42.7	file_gib=28	other_pg=20695	other_l2cv=0.0117	other_contig=0.7577	other_run=4.1	other_gib=28	ro_cost_ms=53-->
<!--DATA 2	stock	pp2048	116.31-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-unpinned [stock] pass 3  env=''  args=''  04:27:51  clk=600 MHz  MemAvail=31412192 kB
<!--PRED 3	stock	memavail_kb=31412192	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5665,5754,5677,4994,3589,3864,2972,2707,2311,2191,4932-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        116.36 ± 0.47 |

build: 171974745 (10558)
<!--RO 3	stock	wall_s=74	busy=2185,2194,2238,2177,2813,2346,2956,3135	busy_tot=20044	busy_little_share=0.4387	a55_cpu_cycles=157198308721	a55_inst_retired=96321584009	context_switches=927532	a76_cpu_cycles=249693584892	a76_l3d_cache_refill=3746307841	a76_l2d_cache_refill=2098144294	cpu_migrations=28318	page_faults=975126	a76_dtlb_walk=627722620	a76_mem_access=126725404703	a76_inst_retired=364060497926	a76_l1d_cache_refill=3911362442	a55_inst_share=0.2092	a76_ipc=1.458	l2ref_pki=5.763	l3ref_pki=10.290	l1dref_pki=10.744	memacc_pki=348.09	dtlbw_pki=1.7242	pmu_enabled=100.0	pmu_cpu_s=584.1	who=at_s=6,rss_pg=426162,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6467	l2color_cv=0.0044	l3color_cv=0.0135	contig_frac=0.8673	mean_run=7.4	vapa16=0.0098	gib_regions=28	anon_pg=2055	anon_l2cv=0.0585	anon_contig=0.6273	anon_run=2.7	anon_gib=28	file_pg=24576	file_l2cv=0.0009	file_contig=0.9905	file_run=87.1	file_gib=27	other_pg=21052	other_l2cv=0.0059	other_contig=0.7468	other_run=3.9	other_gib=28	ro_cost_ms=56-->
<!--DATA 3	stock	pp2048	116.36-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-unpinned [f16res] pass 3  env='ROCKET_F16_RESIDENT=auto'  args=''  04:29:18  clk=600 MHz  MemAvail=31327500 kB
<!--PRED 3	f16res	memavail_kb=31327500	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6024,6368,6142,4157,3082,3987,2864,2698,2309,2159,4942-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        129.39 ± 0.67 |

build: 171974745 (10558)
<!--RO 3	f16res	wall_s=66	busy=2126,2137,2186,2120,2728,2270,2941,2653	busy_tot=19161	busy_little_share=0.4472	a55_cpu_cycles=153312762637	a55_inst_retired=94307996087	context_switches=925043	a76_cpu_cycles=234530657441	a76_l3d_cache_refill=2967961405	a76_l2d_cache_refill=2064346448	cpu_migrations=28478	page_faults=1070074	a76_dtlb_walk=946362698	a76_mem_access=127880634969	a76_inst_retired=366176191813	a76_l1d_cache_refill=3882949835	a55_inst_share=0.2048	a76_ipc=1.561	l2ref_pki=5.638	l3ref_pki=8.105	l1dref_pki=10.604	memacc_pki=349.23	dtlbw_pki=2.5844	pmu_enabled=100.0	pmu_cpu_s=529.7	who=at_s=6,rss_pg=426162,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6439	l2color_cv=0.0028	l3color_cv=0.0104	contig_frac=0.8315	mean_run=5.9	vapa16=0.0189	gib_regions=28	anon_pg=2051	anon_l2cv=0.0499	anon_contig=0.3697	anon_run=1.6	anon_gib=27	file_pg=24576	file_l2cv=0.0007	file_contig=0.9906	file_run=88.4	file_gib=26	other_pg=20844	other_l2cv=0.0072	other_contig=0.6893	other_run=3.2	other_gib=28	ro_cost_ms=53-->
<!--DATA 3	f16res	pp2048	129.39-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 20964MB (MemAvailable 30500MB - reserve 9535MB, no swap)

### qwen35-08b-f16-unpinned [f16res] pass 4  env='ROCKET_F16_RESIDENT=auto'  args=''  04:30:40  clk=600 MHz  MemAvail=31312608 kB
<!--PRED 4	f16res	memavail_kb=31312608	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5171,6298,6138,5119,3006,3937,2952,2622,2283,2159,4946-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        129.25 ± 0.31 |

build: 171974745 (10558)
<!--RO 4	f16res	wall_s=67	busy=2150,2176,2177,2134,2899,2523,2629,2615	busy_tot=19303	busy_little_share=0.4474	a55_cpu_cycles=154688173557	a55_inst_retired=95136001054	context_switches=927497	a76_cpu_cycles=235074790193	a76_l3d_cache_refill=2970401626	a76_l2d_cache_refill=2062415149	cpu_migrations=29014	page_faults=1077091	a76_dtlb_walk=945975787	a76_mem_access=127867343864	a76_inst_retired=366726577340	a76_l1d_cache_refill=3867067155	a55_inst_share=0.2060	a76_ipc=1.560	l2ref_pki=5.624	l3ref_pki=8.100	l1dref_pki=10.545	memacc_pki=348.67	dtlbw_pki=2.5795	pmu_enabled=100.0	pmu_cpu_s=531.1	who=at_s=6,rss_pg=426162,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6411	l2color_cv=0.0059	l3color_cv=0.0114	contig_frac=0.8742	mean_run=7.8	vapa16=0.0104	gib_regions=28	anon_pg=2048	anon_l2cv=0.0543	anon_contig=0.3679	anon_run=1.6	anon_gib=28	file_pg=24576	file_l2cv=0.0017	file_contig=0.9810	file_run=47.9	file_gib=28	other_pg=20642	other_l2cv=0.0109	other_contig=0.7972	other_run=4.9	other_gib=28	ro_cost_ms=57-->
<!--DATA 4	f16res	pp2048	129.25-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 20969MB (MemAvailable 30505MB - reserve 9535MB, no swap)

### qwen35-08b-f16-unpinned [stock] pass 4  env=''  args=''  04:32:01  clk=600 MHz  MemAvail=31486304 kB
<!--PRED 4	stock	memavail_kb=31488860	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5988,6853,6249,5837,3632,3948,3079,2645,2315,2161,4951-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        117.75 ± 0.23 |

build: 171974745 (10558)
<!--RO 4	stock	wall_s=73	busy=2147,2200,2230,2226,3006,2337,2944,2982	busy_tot=20072	busy_little_share=0.4386	a55_cpu_cycles=157575527240	a55_inst_retired=96468567920	context_switches=924064	a76_cpu_cycles=247241050302	a76_l3d_cache_refill=3628783483	a76_l2d_cache_refill=2067073954	cpu_migrations=27641	page_faults=976936	a76_dtlb_walk=1005194690	a76_mem_access=126339082305	a76_inst_retired=362402672789	a76_l1d_cache_refill=3917915574	a55_inst_share=0.2102	a76_ipc=1.466	l2ref_pki=5.704	l3ref_pki=10.013	l1dref_pki=10.811	memacc_pki=348.62	dtlbw_pki=2.7737	pmu_enabled=100.0	pmu_cpu_s=577.4	who=at_s=6,rss_pg=426161,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6459	l2color_cv=0.0033	l3color_cv=0.0190	contig_frac=0.7922	mean_run=4.8	vapa16=0.0498	gib_regions=28	anon_pg=2055	anon_l2cv=0.0287	anon_contig=0.6898	anon_run=3.2	anon_gib=25	file_pg=24576	file_l2cv=0.0003	file_contig=0.9909	file_run=90.4	file_gib=26	other_pg=20988	other_l2cv=0.0069	other_contig=0.5694	other_run=2.3	other_gib=28	ro_cost_ms=56-->
<!--DATA 4	stock	pp2048	117.75-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-unpinned [stock] pass 5  env=''  args=''  04:33:28  clk=600 MHz  MemAvail=31340264 kB
<!--PRED 5	stock	memavail_kb=31340264	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6109,5971,5950,3564,3531,3874,2909,2664,2321,2151,4951-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        123.15 ± 0.37 |

build: 171974745 (10558)
<!--RO 5	stock	wall_s=69	busy=2185,2210,2198,2196,2806,2581,2785,2746	busy_tot=19707	busy_little_share=0.4460	a55_cpu_cycles=156896642933	a55_inst_retired=96426100793	context_switches=925742	a76_cpu_cycles=240668785378	a76_l3d_cache_refill=3247564450	a76_l2d_cache_refill=2099341943	cpu_migrations=27976	page_faults=974741	a76_dtlb_walk=979722193	a76_mem_access=127102502560	a76_inst_retired=363831251334	a76_l1d_cache_refill=3913676719	a55_inst_share=0.2095	a76_ipc=1.512	l2ref_pki=5.770	l3ref_pki=8.926	l1dref_pki=10.757	memacc_pki=349.34	dtlbw_pki=2.6928	pmu_enabled=100.0	pmu_cpu_s=552.7	who=at_s=6,rss_pg=426549,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6467	l2color_cv=0.0062	l3color_cv=0.0213	contig_frac=0.6989	mean_run=3.3	vapa16=0.0202	gib_regions=28	anon_pg=2051	anon_l2cv=0.1101	anon_contig=0.4861	anon_run=1.9	anon_gib=26	file_pg=24576	file_l2cv=0.0007	file_contig=0.9905	file_run=87.5	file_gib=28	other_pg=21051	other_l2cv=0.0086	other_contig=0.3792	other_run=1.6	other_gib=28	ro_cost_ms=53-->
<!--DATA 5	stock	pp2048	123.15-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-unpinned [f16res] pass 5  env='ROCKET_F16_RESIDENT=auto'  args=''  04:34:51  clk=600 MHz  MemAvail=31428168 kB
<!--PRED 5	f16res	memavail_kb=31428168	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5121,6377,6280,5738,4050,3775,2956,2664,2304,2142,4955-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        131.85 ± 0.54 |

build: 171974745 (10558)
<!--RO 5	f16res	wall_s=65	busy=2090,2131,2142,2095,2683,2355,2806,2731	busy_tot=19033	busy_little_share=0.4444	a55_cpu_cycles=151561974177	a55_inst_retired=93441824437	context_switches=927934	a76_cpu_cycles=234107806147	a76_l3d_cache_refill=2931484749	a76_l2d_cache_refill=2066104011	cpu_migrations=28722	page_faults=1070708	a76_dtlb_walk=935343401	a76_mem_access=128569124364	a76_inst_retired=368622453794	a76_l1d_cache_refill=3881305859	a55_inst_share=0.2022	a76_ipc=1.575	l2ref_pki=5.605	l3ref_pki=7.953	l1dref_pki=10.529	memacc_pki=348.78	dtlbw_pki=2.5374	pmu_enabled=100.0	pmu_cpu_s=520.8	who=at_s=6,rss_pg=426549,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6401	l2color_cv=0.0073	l3color_cv=0.0208	contig_frac=0.8396	mean_run=6.2	vapa16=0.0214	gib_regions=28	anon_pg=2051	anon_l2cv=0.0487	anon_contig=0.3286	anon_run=1.5	anon_gib=27	file_pg=24576	file_l2cv=0.0008	file_contig=0.9902	file_run=85.0	file_gib=27	other_pg=20563	other_l2cv=0.0159	other_contig=0.7106	other_run=3.4	other_gib=28	ro_cost_ms=50-->
<!--DATA 5	f16res	pp2048	131.85-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21070MB (MemAvailable 30605MB - reserve 9535MB, no swap)

### qwen35-08b-f16-unpinned [f16res] pass 6  env='ROCKET_F16_RESIDENT=auto'  args=''  04:36:11  clk=600 MHz  MemAvail=31295024 kB
<!--PRED 6	f16res	memavail_kb=31295024	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5764,6726,5821,4904,3310,3764,2975,2609,2261,2142,4958-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        129.33 ± 0.45 |

build: 171974745 (10558)
<!--RO 6	f16res	wall_s=67	busy=2164,2160,2211,2148,2825,2274,2892,2592	busy_tot=19266	busy_little_share=0.4507	a55_cpu_cycles=154928057460	a55_inst_retired=95094527133	context_switches=925735	a76_cpu_cycles=234502907391	a76_l3d_cache_refill=2970028258	a76_l2d_cache_refill=2063374469	cpu_migrations=28463	page_faults=1069690	a76_dtlb_walk=971299184	a76_mem_access=127604742756	a76_inst_retired=365356749953	a76_l1d_cache_refill=3868972629	a55_inst_share=0.2065	a76_ipc=1.558	l2ref_pki=5.648	l3ref_pki=8.129	l1dref_pki=10.590	memacc_pki=349.26	dtlbw_pki=2.6585	pmu_enabled=100.0	pmu_cpu_s=530.3	who=at_s=6,rss_pg=426162,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6456	l2color_cv=0.0032	l3color_cv=0.0161	contig_frac=0.8030	mean_run=5.0	vapa16=0.0175	gib_regions=28	anon_pg=2051	anon_l2cv=0.0314	anon_contig=0.7218	anon_run=3.6	anon_gib=26	file_pg=24576	file_l2cv=0.0012	file_contig=0.9901	file_run=84.2	file_gib=28	other_pg=20974	other_l2cv=0.0062	other_contig=0.5917	other_run=2.4	other_gib=28	ro_cost_ms=51-->
<!--DATA 6	f16res	pp2048	129.33-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 20971MB (MemAvailable 30507MB - reserve 9535MB, no swap)

### qwen35-08b-f16-unpinned [stock] pass 6  env=''  args=''  04:37:31  clk=600 MHz  MemAvail=31386008 kB
<!--PRED 6	stock	memavail_kb=31386008	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5247,6655,5626,3518,3847,3946,3013,2606,2330,2144,4958-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        116.42 ± 0.36 |

build: 171974745 (10558)
<!--RO 6	stock	wall_s=73	busy=2212,2219,2270,2216,2918,2586,2893,2949	busy_tot=20263	busy_little_share=0.4401	a55_cpu_cycles=159286215610	a55_inst_retired=97302581294	context_switches=922040	a76_cpu_cycles=248446405618	a76_l3d_cache_refill=3732086955	a76_l2d_cache_refill=2093283791	cpu_migrations=27908	page_faults=976905	a76_dtlb_walk=637348917	a76_mem_access=126191848311	a76_inst_retired=361888168509	a76_l1d_cache_refill=3919940541	a55_inst_share=0.2119	a76_ipc=1.457	l2ref_pki=5.784	l3ref_pki=10.313	l1dref_pki=10.832	memacc_pki=348.70	dtlbw_pki=1.7612	pmu_enabled=100.0	pmu_cpu_s=584.2	who=at_s=6,rss_pg=426160,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6460	l2color_cv=0.0031	l3color_cv=0.0140	contig_frac=0.8402	mean_run=6.2	vapa16=0.0086	gib_regions=28	anon_pg=2048	anon_l2cv=0.0355	anon_contig=0.5978	anon_run=2.5	anon_gib=26	file_pg=24576	file_l2cv=0.0017	file_contig=0.9867	file_run=65.7	file_gib=25	other_pg=21005	other_l2cv=0.0071	other_contig=0.6923	other_run=3.2	other_gib=28	ro_cost_ms=53-->
<!--DATA 6	stock	pp2048	116.42-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

#### summary: per-arm mean over 6 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 119.63 | 6 | 116.31 | 127.81 | -- | -- |
| pp2048 | f16res | 127.00 | 6 | 116.53 | 131.85 | 1.064x | 0.912 1.080 1.112 1.098 1.071 1.111 |

