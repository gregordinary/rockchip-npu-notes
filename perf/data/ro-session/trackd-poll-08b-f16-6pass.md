<!-- qwen35-08b-f16 poll-vs-pin 2x2  TESTS='-p 2048 -n 0 -r 3'  PASSES=6
     gguf=<data>/qwen35/Qwen3.5-0.8B-F16.gguf (1516744736 bytes)
     arms: unpinned (base) | poll0 = --poll 0 | pin76t4 = PIN_MASK=0xf0 -t 4
           pinpoll = both
-->
== qwen35-08b-f16-poll  Tue Sep  1 01:17:14 UTC 2026 ==
### qwen35-08b-f16-poll [unpinned] pass 1  env=''  args=''  01:17:29  clk=600 MHz  MemAvail=31486468 kB
<!--PRED 1	unpinned	memavail_kb=31486796	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6789,7101,6411,6321,5047,4508,3960,3481,3108,2786,4212-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        117.40 ± 0.11 |

build: 171974745 (10558)
<!--RO 1	unpinned	wall_s=73	busy=2183,2220,2217,2218,2836,2339,3279,2826	busy_tot=20118	busy_little_share=0.4393	a55_cpu_cycles=158144908384	a55_inst_retired=96815516436	context_switches=929959	a76_cpu_cycles=248824030253	a76_l3d_cache_refill=3601182704	a76_l2d_cache_refill=2065687699	cpu_migrations=29022	page_faults=992998	a76_dtlb_walk=1083592982	a76_mem_access=127233888481	a76_inst_retired=365322016046	a76_l1d_cache_refill=3920952489	a55_inst_share=0.2095	a76_ipc=1.468	l2ref_pki=5.654	l3ref_pki=9.858	l1dref_pki=10.733	memacc_pki=348.28	dtlbw_pki=2.9661	pmu_enabled=100.0	pmu_cpu_s=578.9	who=at_s=6,rss_pg=426146,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6420	l2color_cv=0.0056	l3color_cv=0.0135	contig_frac=0.8585	mean_run=7.0	vapa16=0.0119	gib_regions=28	anon_pg=2000	anon_l2cv=0.0716	anon_contig=0.3373	anon_run=1.5	anon_gib=28	file_pg=24576	file_l2cv=0.0007	file_contig=0.9889	file_run=76.8	file_gib=25	other_pg=20756	other_l2cv=0.0096	other_contig=0.7543	other_run=4.0	other_gib=28	ro_cost_ms=60-->
<!--DATA 1	unpinned	pp2048	117.40-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [poll0] pass 1  env=''  args='--poll 0'  01:18:56  clk=600 MHz  MemAvail=31351840 kB
<!--PRED 1	poll0	memavail_kb=31351840	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6064,5850,5862,4106,3606,4102,3268,3039,2550,2408,4664-->
| model                          |       size |     params | backend    | ngl |       poll |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          0 |          pp2048 |        129.01 ± 0.48 |

build: 171974745 (10558)
<!--RO 1	poll0	wall_s=67	busy=2152,2192,2170,2135,3009,2343,2650,2612	busy_tot=19263	busy_little_share=0.4490	a55_cpu_cycles=154887712379	a55_inst_retired=95042967043	context_switches=928387	a76_cpu_cycles=235414722765	a76_l3d_cache_refill=2909756611	a76_l2d_cache_refill=2090408393	cpu_migrations=29150	page_faults=986580	a76_dtlb_walk=948260302	a76_mem_access=128275552221	a76_inst_retired=367971645885	a76_l1d_cache_refill=3900706476	a55_inst_share=0.2053	a76_ipc=1.563	l2ref_pki=5.681	l3ref_pki=7.908	l1dref_pki=10.601	memacc_pki=348.60	dtlbw_pki=2.5770	pmu_enabled=100.0	pmu_cpu_s=528.5	who=at_s=6,rss_pg=426533,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6410	l2color_cv=0.0050	l3color_cv=0.0192	contig_frac=0.8605	mean_run=7.1	vapa16=0.0680	gib_regions=28	anon_pg=2048	anon_l2cv=0.0541	anon_contig=0.4525	anon_run=1.8	anon_gib=26	file_pg=24576	file_l2cv=0.0010	file_contig=0.9884	file_run=73.8	file_gib=27	other_pg=20638	other_l2cv=0.0106	other_contig=0.7488	other_run=4.0	other_gib=28	ro_cost_ms=74-->
<!--DATA 1	poll0	pp2048	129.01-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [pin76t4] pass 1  env='PIN_MASK=0xf0'  args='-t 4'  01:20:15  clk=600 MHz  MemAvail=31315680 kB
<!--PRED 1	pin76t4	memavail_kb=31315680	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4744,4891,3483,2988,4050,3611,3049,2869,2348,2080,4935-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |       4 |          pp2048 |        135.41 ± 0.17 |

build: 171974745 (10558)
<!--RO 1	pin76t4	wall_s=64	busy=90,73,63,81,3092,2911,2637,3313	busy_tot=12260	busy_little_share=0.0250	a55_cpu_cycles=9864546668	a55_inst_retired=2434138540	context_switches=839601	a76_cpu_cycles=257023194050	a76_l3d_cache_refill=3796720516	a76_l2d_cache_refill=2246888094	cpu_migrations=13409	page_faults=995855	a76_dtlb_walk=615512914	a76_mem_access=145341517560	a76_inst_retired=382711471525	a76_l1d_cache_refill=4962396695	a55_inst_share=0.0063	a76_ipc=1.489	l2ref_pki=5.871	l3ref_pki=9.921	l1dref_pki=12.966	memacc_pki=379.77	dtlbw_pki=1.6083	pmu_enabled=100.0	pmu_cpu_s=504.5	who=at_s=6,rss_pg=426209,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6491	l2color_cv=0.0043	l3color_cv=0.0080	contig_frac=0.8874	mean_run=8.7	vapa16=0.0166	gib_regions=28	anon_pg=2048	anon_l2cv=0.0333	anon_contig=0.5602	anon_run=2.3	anon_gib=24	file_pg=24576	file_l2cv=0.0010	file_contig=0.9912	file_run=93.4	file_gib=24	other_pg=21234	other_l2cv=0.0105	other_contig=0.7988	other_run=4.9	other_gib=28	ro_cost_ms=64-->
<!--DATA 1	pin76t4	pp2048	135.41-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [pinpoll] pass 1  env='PIN_MASK=0xf0'  args='-t 4 --poll 0'  01:21:31  clk=600 MHz  MemAvail=31367292 kB
<!--PRED 1	pinpoll	memavail_kb=31367292	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3251,6631,3544,2941,4681,3573,3026,2811,2324,2042,4971-->
| model                          |       size |     params | backend    | ngl | threads |       poll |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ---------: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |       4 |          0 |          pp2048 |        133.18 ± 0.16 |

build: 171974745 (10558)
<!--RO 1	pinpoll	wall_s=65	busy=73,66,57,75,3211,2967,3036,2879	busy_tot=12364	busy_little_share=0.0219	a55_cpu_cycles=9102528960	a55_inst_retired=2060879932	context_switches=840182	a76_cpu_cycles=259765245085	a76_l3d_cache_refill=3947385350	a76_l2d_cache_refill=2290820295	cpu_migrations=13184	page_faults=986045	a76_dtlb_walk=1128091581	a76_mem_access=145358378043	a76_inst_retired=382778145864	a76_l1d_cache_refill=4958905500	a55_inst_share=0.0054	a76_ipc=1.474	l2ref_pki=5.985	l3ref_pki=10.312	l1dref_pki=12.955	memacc_pki=379.75	dtlbw_pki=2.9471	pmu_enabled=100.0	pmu_cpu_s=512.5	who=at_s=7,rss_pg=426209,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6496	l2color_cv=0.0046	l3color_cv=0.0236	contig_frac=0.8717	mean_run=7.7	vapa16=0.0063	gib_regions=28	anon_pg=2048	anon_l2cv=0.0207	anon_contig=0.5807	anon_run=2.4	anon_gib=27	file_pg=24576	file_l2cv=0.0005	file_contig=0.9907	file_run=89.0	file_gib=25	other_pg=21268	other_l2cv=0.0105	other_contig=0.7622	other_run=4.2	other_gib=28	ro_cost_ms=76-->
<!--DATA 1	pinpoll	pp2048	133.18-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [poll0] pass 2  env=''  args='--poll 0'  01:22:50  clk=600 MHz  MemAvail=31329584 kB
<!--PRED 2	poll0	memavail_kb=31329584	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5708,5761,5809,2898,3714,3787,3035,2755,2306,2016,4984-->
| model                          |       size |     params | backend    | ngl |       poll |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          0 |          pp2048 |        116.24 ± 0.23 |

build: 171974745 (10558)
<!--RO 2	poll0	wall_s=74	busy=2189,2216,2249,2197,2865,2535,3043,2884	busy_tot=20178	busy_little_share=0.4386	a55_cpu_cycles=158010552745	a55_inst_retired=96727697256	context_switches=927645	a76_cpu_cycles=248932101194	a76_l3d_cache_refill=3677754047	a76_l2d_cache_refill=2092919131	cpu_migrations=28138	page_faults=985002	a76_dtlb_walk=1113565785	a76_mem_access=127125476409	a76_inst_retired=365180423640	a76_l1d_cache_refill=3906550831	a55_inst_share=0.2094	a76_ipc=1.467	l2ref_pki=5.731	l3ref_pki=10.071	l1dref_pki=10.698	memacc_pki=348.12	dtlbw_pki=3.0494	pmu_enabled=100.0	pmu_cpu_s=584.4	who=at_s=6,rss_pg=426147,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6446	l2color_cv=0.0029	l3color_cv=0.0163	contig_frac=0.8781	mean_run=8.1	vapa16=0.0297	gib_regions=28	anon_pg=2055	anon_l2cv=0.0799	anon_contig=0.5990	anon_run=2.5	anon_gib=28	file_pg=24576	file_l2cv=0.0008	file_contig=0.9903	file_run=85.6	file_gib=25	other_pg=20894	other_l2cv=0.0056	other_contig=0.7735	other_run=4.4	other_gib=28	ro_cost_ms=75-->
<!--DATA 2	poll0	pp2048	116.24-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [pin76t4] pass 2  env='PIN_MASK=0xf0'  args='-t 4'  01:24:16  clk=600 MHz  MemAvail=31353444 kB
<!--PRED 2	pin76t4	memavail_kb=31353444	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4837,6700,4012,3152,4329,3823,2962,2743,2299,1998,5000-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |       4 |          pp2048 |        134.27 ± 0.18 |

build: 171974745 (10558)
<!--RO 2	pin76t4	wall_s=65	busy=42,72,78,67,3046,3133,2952,2788	busy_tot=12178	busy_little_share=0.0213	a55_cpu_cycles=9198560056	a55_inst_retired=2127730606	context_switches=836953	a76_cpu_cycles=256480257151	a76_l3d_cache_refill=3883799903	a76_l2d_cache_refill=2267036885	cpu_migrations=12718	page_faults=985997	a76_dtlb_walk=1121300067	a76_mem_access=145281495749	a76_inst_retired=382762893270	a76_l1d_cache_refill=4949327688	a55_inst_share=0.0055	a76_ipc=1.492	l2ref_pki=5.923	l3ref_pki=10.147	l1dref_pki=12.931	memacc_pki=379.56	dtlbw_pki=2.9295	pmu_enabled=100.0	pmu_cpu_s=508.1	who=at_s=6,rss_pg=426218,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6426	l2color_cv=0.0018	l3color_cv=0.0300	contig_frac=0.8620	mean_run=7.2	vapa16=0.0460	gib_regions=28	anon_pg=2048	anon_l2cv=0.0219	anon_contig=0.7187	anon_run=3.5	anon_gib=13	file_pg=24576	file_l2cv=0.0007	file_contig=0.9909	file_run=91.0	file_gib=26	other_pg=20756	other_l2cv=0.0047	other_contig=0.7233	other_run=3.6	other_gib=28	ro_cost_ms=65-->
<!--DATA 2	pin76t4	pp2048	134.27-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [pinpoll] pass 2  env='PIN_MASK=0xf0'  args='-t 4 --poll 0'  01:25:33  clk=600 MHz  MemAvail=31452084 kB
<!--PRED 2	pinpoll	memavail_kb=31452084	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4954,5787,5842,5124,4360,3827,3022,2724,2297,1985,5009-->
| model                          |       size |     params | backend    | ngl | threads |       poll |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ---------: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |       4 |          0 |          pp2048 |        133.88 ± 0.02 |

build: 171974745 (10558)
<!--RO 2	pinpoll	wall_s=65	busy=77,71,58,75,2959,3027,3242,2779	busy_tot=12288	busy_little_share=0.0229	a55_cpu_cycles=9371092650	a55_inst_retired=2206947363	context_switches=836806	a76_cpu_cycles=258798094849	a76_l3d_cache_refill=3921700653	a76_l2d_cache_refill=2284359545	cpu_migrations=13487	page_faults=988632	a76_dtlb_walk=1123393845	a76_mem_access=145096056369	a76_inst_retired=381658546995	a76_l1d_cache_refill=4938743489	a55_inst_share=0.0057	a76_ipc=1.475	l2ref_pki=5.985	l3ref_pki=10.275	l1dref_pki=12.940	memacc_pki=380.17	dtlbw_pki=2.9435	pmu_enabled=100.0	pmu_cpu_s=509.4	who=at_s=6,rss_pg=426209,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6387	l2color_cv=0.0018	l3color_cv=0.0045	contig_frac=0.8868	mean_run=8.7	vapa16=0.0068	gib_regions=28	anon_pg=2048	anon_l2cv=0.0247	anon_contig=0.5499	anon_run=2.2	anon_gib=17	file_pg=24576	file_l2cv=0.0016	file_contig=0.9894	file_run=79.8	file_gib=27	other_pg=20469	other_l2cv=0.0035	other_contig=0.7973	other_run=4.9	other_gib=27	ro_cost_ms=57-->
<!--DATA 2	pinpoll	pp2048	133.88-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [unpinned] pass 2  env=''  args=''  01:26:51  clk=600 MHz  MemAvail=31319524 kB
<!--PRED 2	unpinned	memavail_kb=31319524	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6121,5841,4610,2949,3680,3860,2978,2734,2300,1972,5013-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        116.81 ± 0.46 |

build: 171974745 (10558)
<!--RO 2	unpinned	wall_s=73	busy=2174,2228,2223,2221,2818,2572,3065,2854	busy_tot=20155	busy_little_share=0.4389	a55_cpu_cycles=158303893567	a55_inst_retired=96955094848	context_switches=926408	a76_cpu_cycles=249000633432	a76_l3d_cache_refill=3715584620	a76_l2d_cache_refill=2079971944	cpu_migrations=28244	page_faults=981212	a76_dtlb_walk=1106263066	a76_mem_access=126783873597	a76_inst_retired=364477019180	a76_l1d_cache_refill=3911403849	a55_inst_share=0.2101	a76_ipc=1.464	l2ref_pki=5.707	l3ref_pki=10.194	l1dref_pki=10.732	memacc_pki=347.85	dtlbw_pki=3.0352	pmu_enabled=100.0	pmu_cpu_s=580.5	who=at_s=6,rss_pg=426142,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6436	l2color_cv=0.0041	l3color_cv=0.0108	contig_frac=0.8941	mean_run=9.3	vapa16=0.0201	gib_regions=28	anon_pg=2048	anon_l2cv=0.0615	anon_contig=0.7529	anon_run=4.0	anon_gib=27	file_pg=24576	file_l2cv=0.0008	file_contig=0.9914	file_run=94.9	file_gib=27	other_pg=20828	other_l2cv=0.0065	other_contig=0.7932	other_run=4.8	other_gib=28	ro_cost_ms=55-->
<!--DATA 2	unpinned	pp2048	116.81-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [pin76t4] pass 3  env='PIN_MASK=0xf0'  args='-t 4'  01:28:17  clk=600 MHz  MemAvail=31365880 kB
<!--PRED 3	pin76t4	memavail_kb=31370200	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6091,6685,5898,3566,3767,3770,2968,2729,2293,1971,5019-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |       4 |          pp2048 |        133.59 ± 0.11 |

build: 171974745 (10558)
<!--RO 3	pin76t4	wall_s=64	busy=81,75,80,76,2979,2776,3449,2839	busy_tot=12355	busy_little_share=0.0253	a55_cpu_cycles=9794620304	a55_inst_retired=2374985937	context_switches=839152	a76_cpu_cycles=258470264990	a76_l3d_cache_refill=3873229782	a76_l2d_cache_refill=2280666426	cpu_migrations=13215	page_faults=992940	a76_dtlb_walk=648101204	a76_mem_access=145452406977	a76_inst_retired=383357251366	a76_l1d_cache_refill=4962104257	a55_inst_share=0.0062	a76_ipc=1.483	l2ref_pki=5.949	l3ref_pki=10.103	l1dref_pki=12.944	memacc_pki=379.42	dtlbw_pki=1.6906	pmu_enabled=100.0	pmu_cpu_s=510.4	who=at_s=6,rss_pg=426211,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6426	l2color_cv=0.0038	l3color_cv=0.0129	contig_frac=0.8608	mean_run=7.1	vapa16=0.0140	gib_regions=28	anon_pg=2048	anon_l2cv=0.0456	anon_contig=0.5470	anon_run=2.2	anon_gib=25	file_pg=24576	file_l2cv=0.0007	file_contig=0.9911	file_run=92.4	file_gib=26	other_pg=20756	other_l2cv=0.0095	other_contig=0.7374	other_run=3.8	other_gib=28	ro_cost_ms=74-->
<!--DATA 3	pin76t4	pp2048	133.59-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [pinpoll] pass 3  env='PIN_MASK=0xf0'  args='-t 4 --poll 0'  01:29:33  clk=600 MHz  MemAvail=31365240 kB
<!--PRED 3	pinpoll	memavail_kb=31365240	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6187,6146,6087,3321,3891,3853,2927,2717,2290,1967,5023-->
| model                          |       size |     params | backend    | ngl | threads |       poll |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ---------: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |       4 |          0 |          pp2048 |        134.18 ± 0.20 |

build: 171974745 (10558)
<!--RO 3	pinpoll	wall_s=65	busy=70,94,110,55,3068,2689,3349,2912	busy_tot=12347	busy_little_share=0.0266	a55_cpu_cycles=10450364223	a55_inst_retired=2731199804	context_switches=839375	a76_cpu_cycles=259319677283	a76_l3d_cache_refill=3937426357	a76_l2d_cache_refill=2281094224	cpu_migrations=13289	page_faults=998233	a76_dtlb_walk=1117888911	a76_mem_access=145148347645	a76_inst_retired=382001970705	a76_l1d_cache_refill=4950984421	a55_inst_share=0.0071	a76_ipc=1.473	l2ref_pki=5.971	l3ref_pki=10.307	l1dref_pki=12.961	memacc_pki=379.97	dtlbw_pki=2.9264	pmu_enabled=100.0	pmu_cpu_s=508.7	who=at_s=6,rss_pg=426214,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6496	l2color_cv=0.0036	l3color_cv=0.0094	contig_frac=0.8286	mean_run=5.8	vapa16=0.0193	gib_regions=28	anon_pg=2048	anon_l2cv=0.0228	anon_contig=0.5607	anon_run=2.3	anon_gib=19	file_pg=24576	file_l2cv=0.0024	file_contig=0.9808	file_run=47.3	file_gib=27	other_pg=21268	other_l2cv=0.0080	other_contig=0.6784	other_run=3.1	other_gib=28	ro_cost_ms=78-->
<!--DATA 3	pinpoll	pp2048	134.18-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [unpinned] pass 3  env=''  args=''  01:30:52  clk=600 MHz  MemAvail=31345980 kB
<!--PRED 3	unpinned	memavail_kb=31345980	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6340,6005,5909,3925,3792,3910,2975,2676,2291,1951,5024-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        119.94 ± 0.72 |

build: 171974745 (10558)
<!--RO 3	unpinned	wall_s=71	busy=2180,2220,2239,2236,2823,2838,2757,2679	busy_tot=19972	busy_little_share=0.4444	a55_cpu_cycles=158944924053	a55_inst_retired=97268797799	context_switches=931688	a76_cpu_cycles=245567146432	a76_l3d_cache_refill=3522340519	a76_l2d_cache_refill=2064949171	cpu_migrations=28998	page_faults=1001147	a76_dtlb_walk=1019989086	a76_mem_access=127238255356	a76_inst_retired=365756769546	a76_l1d_cache_refill=3911025741	a55_inst_share=0.2101	a76_ipc=1.489	l2ref_pki=5.646	l3ref_pki=9.630	l1dref_pki=10.693	memacc_pki=347.88	dtlbw_pki=2.7887	pmu_enabled=100.0	pmu_cpu_s=567.7	who=at_s=6,rss_pg=426147,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6448	l2color_cv=0.0044	l3color_cv=0.0218	contig_frac=0.6473	mean_run=2.8	vapa16=0.0278	gib_regions=28	anon_pg=2048	anon_l2cv=0.0272	anon_contig=0.7236	anon_run=3.6	anon_gib=27	file_pg=24576	file_l2cv=0.0008	file_contig=0.9909	file_run=91.0	file_gib=25	other_pg=20915	other_l2cv=0.0107	other_contig=0.2359	other_run=1.3	other_gib=28	ro_cost_ms=62-->
<!--DATA 3	unpinned	pp2048	119.94-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [poll0] pass 3  env=''  args='--poll 0'  01:32:17  clk=600 MHz  MemAvail=31527656 kB
<!--PRED 3	poll0	memavail_kb=31527656	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6247,6794,6451,5860,4967,3820,3003,2708,2319,1938,5027-->
| model                          |       size |     params | backend    | ngl |       poll |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          0 |          pp2048 |        116.55 ± 0.39 |

build: 171974745 (10558)
<!--RO 3	poll0	wall_s=74	busy=2188,2251,2228,2223,2785,2834,2774,2906	busy_tot=20189	busy_little_share=0.4403	a55_cpu_cycles=159404967413	a55_inst_retired=97724321453	context_switches=930149	a76_cpu_cycles=249187486433	a76_l3d_cache_refill=3685670073	a76_l2d_cache_refill=2098482275	cpu_migrations=28769	page_faults=1007897	a76_dtlb_walk=645971575	a76_mem_access=127115635049	a76_inst_retired=365689866268	a76_l1d_cache_refill=3918923992	a55_inst_share=0.2109	a76_ipc=1.468	l2ref_pki=5.738	l3ref_pki=10.079	l1dref_pki=10.717	memacc_pki=347.61	dtlbw_pki=1.7664	pmu_enabled=100.0	pmu_cpu_s=582.5	who=at_s=6,rss_pg=426147,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6434	l2color_cv=0.0052	l3color_cv=0.0171	contig_frac=0.8321	mean_run=5.9	vapa16=0.0167	gib_regions=28	anon_pg=1968	anon_l2cv=0.0567	anon_contig=0.6088	anon_run=2.5	anon_gib=28	file_pg=24576	file_l2cv=0.0002	file_contig=0.9906	file_run=88.4	file_gib=28	other_pg=20890	other_l2cv=0.0097	other_contig=0.6666	other_run=3.0	other_gib=28	ro_cost_ms=56-->
<!--DATA 3	poll0	pp2048	116.55-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [pinpoll] pass 4  env='PIN_MASK=0xf0'  args='-t 4 --poll 0'  01:33:43  clk=600 MHz  MemAvail=31322312 kB
<!--PRED 4	pinpoll	memavail_kb=31322312	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5079,5442,3995,3546,4476,3776,2878,2710,2288,1949,5028-->
| model                          |       size |     params | backend    | ngl | threads |       poll |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ---------: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |       4 |          0 |          pp2048 |        133.71 ± 0.12 |

build: 171974745 (10558)
<!--RO 4	pinpoll	wall_s=65	busy=69,62,76,83,2967,3055,3204,2783	busy_tot=12299	busy_little_share=0.0236	a55_cpu_cycles=9901636749	a55_inst_retired=2417958250	context_switches=839169	a76_cpu_cycles=258808998033	a76_l3d_cache_refill=3918034605	a76_l2d_cache_refill=2280117703	cpu_migrations=13064	page_faults=993191	a76_dtlb_walk=1090438112	a76_mem_access=145315408022	a76_inst_retired=382653968492	a76_l1d_cache_refill=4945324266	a55_inst_share=0.0063	a76_ipc=1.479	l2ref_pki=5.959	l3ref_pki=10.239	l1dref_pki=12.924	memacc_pki=379.76	dtlbw_pki=2.8497	pmu_enabled=100.0	pmu_cpu_s=510.2	who=at_s=6,rss_pg=426219,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6500	l2color_cv=0.0054	l3color_cv=0.0163	contig_frac=0.8796	mean_run=8.2	vapa16=0.0193	gib_regions=28	anon_pg=2048	anon_l2cv=0.0265	anon_contig=0.7109	anon_run=3.4	anon_gib=13	file_pg=24576	file_l2cv=0.0009	file_contig=0.9907	file_run=88.7	file_gib=26	other_pg=21297	other_l2cv=0.0128	other_contig=0.7677	other_run=4.3	other_gib=28	ro_cost_ms=67-->
<!--DATA 4	pinpoll	pp2048	133.71-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [unpinned] pass 4  env=''  args=''  01:35:02  clk=600 MHz  MemAvail=31388532 kB
<!--PRED 4	unpinned	memavail_kb=31388536	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6554,6871,5929,3829,4399,3925,2850,2701,2291,1937,5034-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        116.21 ± 0.32 |

build: 171974745 (10558)
<!--RO 4	unpinned	wall_s=73	busy=2220,2248,2259,2231,2998,2434,2846,2998	busy_tot=20234	busy_little_share=0.4427	a55_cpu_cycles=159984330104	a55_inst_retired=97872095346	context_switches=927650	a76_cpu_cycles=249334566119	a76_l3d_cache_refill=3748578734	a76_l2d_cache_refill=2094192428	cpu_migrations=28895	page_faults=995555	a76_dtlb_walk=1110798661	a76_mem_access=126594038086	a76_inst_retired=363787374179	a76_l1d_cache_refill=3903201723	a55_inst_share=0.2120	a76_ipc=1.459	l2ref_pki=5.757	l3ref_pki=10.304	l1dref_pki=10.729	memacc_pki=347.99	dtlbw_pki=3.0534	pmu_enabled=100.0	pmu_cpu_s=584.5	who=at_s=6,rss_pg=426147,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6437	l2color_cv=0.0055	l3color_cv=0.0185	contig_frac=0.8730	mean_run=7.8	vapa16=0.0154	gib_regions=28	anon_pg=2055	anon_l2cv=0.0374	anon_contig=0.6771	anon_run=3.1	anon_gib=24	file_pg=24576	file_l2cv=0.0005	file_contig=0.9909	file_run=90.7	file_gib=26	other_pg=20829	other_l2cv=0.0137	other_contig=0.7531	other_run=4.0	other_gib=28	ro_cost_ms=67-->
<!--DATA 4	unpinned	pp2048	116.21-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [poll0] pass 4  env=''  args='--poll 0'  01:36:30  clk=600 MHz  MemAvail=31357204 kB
<!--PRED 4	poll0	memavail_kb=31357204	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5037,5966,5816,4250,3843,3774,2927,2704,2269,1946,5036-->
| model                          |       size |     params | backend    | ngl |       poll |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          0 |          pp2048 |        125.18 ± 0.90 |

build: 171974745 (10558)
<!--RO 4	poll0	wall_s=68	busy=2181,2172,2197,2149,2940,2321,2769,2868	busy_tot=19597	busy_little_share=0.4439	a55_cpu_cycles=155648423138	a55_inst_retired=95655325435	context_switches=929831	a76_cpu_cycles=240781706742	a76_l3d_cache_refill=3136332275	a76_l2d_cache_refill=2102546579	cpu_migrations=29378	page_faults=1003039	a76_dtlb_walk=955871354	a76_mem_access=128582012211	a76_inst_retired=369610067782	a76_l1d_cache_refill=3919699091	a55_inst_share=0.2056	a76_ipc=1.535	l2ref_pki=5.689	l3ref_pki=8.486	l1dref_pki=10.605	memacc_pki=347.89	dtlbw_pki=2.5862	pmu_enabled=100.0	pmu_cpu_s=544.1	who=at_s=6,rss_pg=426523,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6446	l2color_cv=0.0060	l3color_cv=0.0167	contig_frac=0.6656	mean_run=3.0	vapa16=0.0433	gib_regions=28	anon_pg=2048	anon_l2cv=0.1094	anon_contig=0.5088	anon_run=2.0	anon_gib=28	file_pg=24576	file_l2cv=0.0012	file_contig=0.9834	file_run=53.9	file_gib=27	other_pg=20900	other_l2cv=0.0070	other_contig=0.3073	other_run=1.4	other_gib=28	ro_cost_ms=57-->
<!--DATA 4	poll0	pp2048	125.18-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [pin76t4] pass 4  env='PIN_MASK=0xf0'  args='-t 4'  01:37:50  clk=600 MHz  MemAvail=31328748 kB
<!--PRED 4	pin76t4	memavail_kb=31328748	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6371,7138,6233,3315,3669,3983,2859,2663,2279,1936,5039-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |       4 |          pp2048 |        133.77 ± 0.28 |

build: 171974745 (10558)
<!--RO 4	pin76t4	wall_s=64	busy=87,86,82,95,3156,2841,3211,2823	busy_tot=12381	busy_little_share=0.0283	a55_cpu_cycles=10941155620	a55_inst_retired=3060883052	context_switches=841208	a76_cpu_cycles=258911970903	a76_l3d_cache_refill=3897197690	a76_l2d_cache_refill=2295882397	cpu_migrations=13746	page_faults=1004384	a76_dtlb_walk=1122007790	a76_mem_access=145247417363	a76_inst_retired=382464769483	a76_l1d_cache_refill=4964536204	a55_inst_share=0.0079	a76_ipc=1.477	l2ref_pki=6.003	l3ref_pki=10.190	l1dref_pki=12.980	memacc_pki=379.77	dtlbw_pki=2.9336	pmu_enabled=100.0	pmu_cpu_s=509.6	who=at_s=6,rss_pg=426211,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6442	l2color_cv=0.0035	l3color_cv=0.0185	contig_frac=0.9031	mean_run=10.1	vapa16=0.0160	gib_regions=28	anon_pg=2048	anon_l2cv=0.0496	anon_contig=0.3312	anon_run=1.5	anon_gib=28	file_pg=24576	file_l2cv=0.0008	file_contig=0.9904	file_run=86.8	file_gib=26	other_pg=20868	other_l2cv=0.0069	other_contig=0.8563	other_run=6.9	other_gib=28	ro_cost_ms=114-->
<!--DATA 4	pin76t4	pp2048	133.77-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [unpinned] pass 5  env=''  args=''  01:39:08  clk=600 MHz  MemAvail=31366060 kB
<!--PRED 5	unpinned	memavail_kb=31366292	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6263,6839,4503,3207,4472,3887,2922,2681,2272,1943,5039-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        115.87 ± 0.29 |

build: 171974745 (10558)
<!--RO 5	unpinned	wall_s=74	busy=2198,2204,2252,2264,2881,2448,3135,2832	busy_tot=20214	busy_little_share=0.4412	a55_cpu_cycles=159514451092	a55_inst_retired=97636438260	context_switches=924457	a76_cpu_cycles=249639656449	a76_l3d_cache_refill=3762644224	a76_l2d_cache_refill=2101935988	cpu_migrations=28045	page_faults=995735	a76_dtlb_walk=1111844798	a76_mem_access=126775948507	a76_inst_retired=364727979995	a76_l1d_cache_refill=3916504243	a55_inst_share=0.2112	a76_ipc=1.461	l2ref_pki=5.763	l3ref_pki=10.316	l1dref_pki=10.738	memacc_pki=347.59	dtlbw_pki=3.0484	pmu_enabled=100.0	pmu_cpu_s=586.5	who=at_s=6,rss_pg=426148,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6423	l2color_cv=0.0039	l3color_cv=0.0198	contig_frac=0.8472	mean_run=6.5	vapa16=0.0121	gib_regions=28	anon_pg=2048	anon_l2cv=0.0553	anon_contig=0.5851	anon_run=2.4	anon_gib=27	file_pg=24576	file_l2cv=0.0010	file_contig=0.9909	file_run=90.7	file_gib=26	other_pg=20733	other_l2cv=0.0109	other_contig=0.7028	other_run=3.3	other_gib=28	ro_cost_ms=57-->
<!--DATA 5	unpinned	pp2048	115.87-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [poll0] pass 5  env=''  args='--poll 0'  01:40:36  clk=600 MHz  MemAvail=31377008 kB
<!--PRED 5	poll0	memavail_kb=31377008	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6186,7034,4774,3213,4450,3901,2913,2678,2265,1949,5040-->
| model                          |       size |     params | backend    | ngl |       poll |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          0 |          pp2048 |        118.35 ± 0.44 |

build: 171974745 (10558)
<!--RO 5	poll0	wall_s=72	busy=2199,2241,2240,2185,2934,2381,2921,2848	busy_tot=19949	busy_little_share=0.4444	a55_cpu_cycles=159110411011	a55_inst_retired=97424341678	context_switches=928339	a76_cpu_cycles=246466227950	a76_l3d_cache_refill=3552790168	a76_l2d_cache_refill=2039969807	cpu_migrations=29338	page_faults=1000496	a76_dtlb_walk=1055043949	a76_mem_access=127180906474	a76_inst_retired=365717294636	a76_l1d_cache_refill=3905701838	a55_inst_share=0.2104	a76_ipc=1.484	l2ref_pki=5.578	l3ref_pki=9.715	l1dref_pki=10.680	memacc_pki=347.76	dtlbw_pki=2.8849	pmu_enabled=100.0	pmu_cpu_s=573.9	who=at_s=6,rss_pg=426155,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6417	l2color_cv=0.0062	l3color_cv=0.0201	contig_frac=0.8546	mean_run=6.8	vapa16=0.0119	gib_regions=28	anon_pg=2055	anon_l2cv=0.0369	anon_contig=0.5741	anon_run=2.3	anon_gib=27	file_pg=24576	file_l2cv=0.0031	file_contig=0.9816	file_run=49.2	file_gib=27	other_pg=20681	other_l2cv=0.0120	other_contig=0.7316	other_run=3.7	other_gib=28	ro_cost_ms=51-->
<!--DATA 5	poll0	pp2048	118.35-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [pin76t4] pass 5  env='PIN_MASK=0xf0'  args='-t 4'  01:42:02  clk=600 MHz  MemAvail=31469312 kB
<!--PRED 5	pin76t4	memavail_kb=31469312	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4935,5910,5852,4967,4232,3882,3029,2695,2268,1950,5042-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |       4 |          pp2048 |        149.71 ± 0.13 |

build: 171974745 (10558)
<!--RO 5	pin76t4	wall_s=57	busy=90,68,83,94,3015,2648,2750,2936	busy_tot=11684	busy_little_share=0.0287	a55_cpu_cycles=10370822677	a55_inst_retired=2642547168	context_switches=838842	a76_cpu_cycles=244256504757	a76_l3d_cache_refill=3162406408	a76_l2d_cache_refill=2285872995	cpu_migrations=13290	page_faults=996996	a76_dtlb_walk=954472066	a76_mem_access=145293858837	a76_inst_retired=382362635107	a76_l1d_cache_refill=4947183893	a55_inst_share=0.0069	a76_ipc=1.565	l2ref_pki=5.978	l3ref_pki=8.271	l1dref_pki=12.938	memacc_pki=379.99	dtlbw_pki=2.4962	pmu_enabled=100.0	pmu_cpu_s=457.8	who=at_s=18,rss_pg=464131,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6488	l2color_cv=0.0261	l3color_cv=0.0541	contig_frac=0.8157	mean_run=5.4	vapa16=0.0246	gib_regions=29	anon_pg=2048	anon_l2cv=0.0229	anon_contig=0.3019	anon_run=1.4	anon_gib=18	file_pg=24576	file_l2cv=0.0009	file_contig=0.9909	file_run=90.4	file_gib=26	other_pg=21214	other_l2cv=0.0590	other_contig=0.6622	other_run=2.9	other_gib=29	ro_cost_ms=77-->
<!--DATA 5	pin76t4	pp2048	149.71-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [pinpoll] pass 5  env='PIN_MASK=0xf0'  args='-t 4 --poll 0'  01:43:12  clk=600 MHz  MemAvail=31334084 kB
<!--PRED 5	pinpoll	memavail_kb=31334084	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5265,6008,3711,3641,4292,3941,2811,2653,2297,1938,5042-->
| model                          |       size |     params | backend    | ngl | threads |       poll |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ---------: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |       4 |          0 |          pp2048 |        136.23 ± 0.07 |

build: 171974745 (10558)
<!--RO 5	pinpoll	wall_s=63	busy=87,73,73,95,2987,2926,3166,2851	busy_tot=12258	busy_little_share=0.0268	a55_cpu_cycles=10423011330	a55_inst_retired=2726500003	context_switches=838290	a76_cpu_cycles=256393290269	a76_l3d_cache_refill=3809985244	a76_l2d_cache_refill=2234259043	cpu_migrations=13103	page_faults=1000372	a76_dtlb_walk=1066531161	a76_mem_access=145269995017	a76_inst_retired=382351406000	a76_l1d_cache_refill=4949655444	a55_inst_share=0.0071	a76_ipc=1.491	l2ref_pki=5.843	l3ref_pki=9.965	l1dref_pki=12.945	memacc_pki=379.94	dtlbw_pki=2.7894	pmu_enabled=100.0	pmu_cpu_s=501.6	who=at_s=6,rss_pg=426211,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6426	l2color_cv=0.0039	l3color_cv=0.0197	contig_frac=0.8735	mean_run=7.8	vapa16=0.4963	gib_regions=28	anon_pg=2048	anon_l2cv=0.0413	anon_contig=0.5245	anon_run=2.1	anon_gib=28	file_pg=24576	file_l2cv=0.0007	file_contig=0.9866	file_run=65.2	file_gib=27	other_pg=20756	other_l2cv=0.0074	other_contig=0.7741	other_run=4.4	other_gib=28	ro_cost_ms=72-->
<!--DATA 5	pinpoll	pp2048	136.23-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [poll0] pass 6  env=''  args='--poll 0'  01:44:29  clk=600 MHz  MemAvail=31354092 kB
<!--PRED 6	poll0	memavail_kb=31354092	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5296,5851,5817,4514,3504,3995,2913,2643,2272,1934,5045-->
| model                          |       size |     params | backend    | ngl |       poll |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          0 |          pp2048 |        116.26 ± 0.41 |

build: 171974745 (10558)
<!--RO 6	poll0	wall_s=74	busy=2181,2211,2252,2218,2903,2466,3060,2927	busy_tot=20218	busy_little_share=0.4383	a55_cpu_cycles=159358760850	a55_inst_retired=97347234642	context_switches=928568	a76_cpu_cycles=248937869750	a76_l3d_cache_refill=3681054684	a76_l2d_cache_refill=2097999280	cpu_migrations=29120	page_faults=1002714	a76_dtlb_walk=1107799796	a76_mem_access=126957151586	a76_inst_retired=365331757884	a76_l1d_cache_refill=3931090546	a55_inst_share=0.2104	a76_ipc=1.468	l2ref_pki=5.743	l3ref_pki=10.076	l1dref_pki=10.760	memacc_pki=347.51	dtlbw_pki=3.0323	pmu_enabled=100.0	pmu_cpu_s=583.8	who=at_s=7,rss_pg=426145,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6458	l2color_cv=0.0031	l3color_cv=0.0094	contig_frac=0.8458	mean_run=6.4	vapa16=0.0856	gib_regions=28	anon_pg=2055	anon_l2cv=0.0433	anon_contig=0.5868	anon_run=2.4	anon_gib=28	file_pg=24576	file_l2cv=0.0009	file_contig=0.9900	file_run=83.9	file_gib=25	other_pg=20981	other_l2cv=0.0072	other_contig=0.7023	other_run=3.3	other_gib=28	ro_cost_ms=58-->
<!--DATA 6	poll0	pp2048	116.26-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [pin76t4] pass 6  env='PIN_MASK=0xf0'  args='-t 4'  01:45:55  clk=600 MHz  MemAvail=31457952 kB
<!--PRED 6	pin76t4	memavail_kb=31458232	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6175,5918,5888,5198,4519,3956,2912,2657,2255,1945,5047-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |       4 |          pp2048 |        144.04 ± 0.19 |

build: 171974745 (10558)
<!--RO 6	pin76t4	wall_s=60	busy=96,71,86,69,2958,2678,3035,2821	busy_tot=11814	busy_little_share=0.0273	a55_cpu_cycles=10346258834	a55_inst_retired=2603075598	context_switches=839318	a76_cpu_cycles=248146036810	a76_l3d_cache_refill=3358756804	a76_l2d_cache_refill=2280417423	cpu_migrations=13220	page_faults=997124	a76_dtlb_walk=963961354	a76_mem_access=145254286588	a76_inst_retired=381813874037	a76_l1d_cache_refill=4938685455	a55_inst_share=0.0068	a76_ipc=1.539	l2ref_pki=5.973	l3ref_pki=8.797	l1dref_pki=12.935	memacc_pki=380.43	dtlbw_pki=2.5247	pmu_enabled=100.0	pmu_cpu_s=474.7	who=at_s=18,rss_pg=464124,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6500	l2color_cv=0.0212	l3color_cv=0.0428	contig_frac=0.7969	mean_run=4.9	vapa16=0.0775	gib_regions=29	anon_pg=2048	anon_l2cv=0.0258	anon_contig=0.3033	anon_run=1.4	anon_gib=23	file_pg=24576	file_l2cv=0.0009	file_contig=0.9899	file_run=83.3	file_gib=27	other_pg=21301	other_l2cv=0.0469	other_contig=0.6216	other_run=2.6	other_gib=29	ro_cost_ms=51-->
<!--DATA 6	pin76t4	pp2048	144.04-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [pinpoll] pass 6  env='PIN_MASK=0xf0'  args='-t 4 --poll 0'  01:47:08  clk=600 MHz  MemAvail=31341260 kB
<!--PRED 6	pinpoll	memavail_kb=31341260	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7071,5671,3899,3916,4228,4040,2798,2626,2269,1940,5047-->
| model                          |       size |     params | backend    | ngl | threads |       poll |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ---------: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |       4 |          0 |          pp2048 |        147.00 ± 0.10 |

build: 171974745 (10558)
<!--RO 6	pinpoll	wall_s=59	busy=100,79,62,100,3004,2548,2934,3029	busy_tot=11856	busy_little_share=0.0288	a55_cpu_cycles=10478035059	a55_inst_retired=2688051836	context_switches=838951	a76_cpu_cycles=247002668492	a76_l3d_cache_refill=3291473658	a76_l2d_cache_refill=2290125897	cpu_migrations=13045	page_faults=996252	a76_dtlb_walk=996562987	a76_mem_access=145200624928	a76_inst_retired=381967346340	a76_l1d_cache_refill=4929836348	a55_inst_share=0.0070	a76_ipc=1.546	l2ref_pki=5.996	l3ref_pki=8.617	l1dref_pki=12.906	memacc_pki=380.14	dtlbw_pki=2.6090	pmu_enabled=100.0	pmu_cpu_s=465.8	who=at_s=18,rss_pg=464124,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6491	l2color_cv=0.0228	l3color_cv=0.0491	contig_frac=0.8923	mean_run=9.1	vapa16=0.0120	gib_regions=29	anon_pg=2048	anon_l2cv=0.0403	anon_contig=0.3346	anon_run=1.5	anon_gib=24	file_pg=24576	file_l2cv=0.0015	file_contig=0.9890	file_run=77.0	file_gib=25	other_pg=21234	other_l2cv=0.0511	other_contig=0.8342	other_run=6.0	other_gib=29	ro_cost_ms=74-->
<!--DATA 6	pinpoll	pp2048	147.00-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16-poll [unpinned] pass 6  env=''  args=''  01:48:20  clk=600 MHz  MemAvail=31372944 kB
<!--PRED 6	unpinned	memavail_kb=31372944	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5746,6977,6199,3788,3808,3971,2876,2645,2255,1945,5048-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        127.64 ± 0.45 |

build: 171974745 (10558)
<!--RO 6	unpinned	wall_s=68	busy=2136,2182,2196,2158,2701,2501,2855,2812	busy_tot=19541	busy_little_share=0.4438	a55_cpu_cycles=154971413350	a55_inst_retired=95371021475	context_switches=930149	a76_cpu_cycles=238881512097	a76_l3d_cache_refill=3104853320	a76_l2d_cache_refill=2102751895	cpu_migrations=28684	page_faults=999145	a76_dtlb_walk=925787119	a76_mem_access=128305812267	a76_inst_retired=368234987797	a76_l1d_cache_refill=3910275449	a55_inst_share=0.2057	a76_ipc=1.541	l2ref_pki=5.710	l3ref_pki=8.432	l1dref_pki=10.619	memacc_pki=348.43	dtlbw_pki=2.5141	pmu_enabled=100.0	pmu_cpu_s=534.4	who=at_s=6,rss_pg=426527,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6456	l2color_cv=0.0068	l3color_cv=0.0118	contig_frac=0.8919	mean_run=9.1	vapa16=0.0814	gib_regions=28	anon_pg=2055	anon_l2cv=0.0531	anon_contig=0.5946	anon_run=2.5	anon_gib=27	file_pg=24576	file_l2cv=0.0029	file_contig=0.9791	file_run=43.8	file_gib=27	other_pg=20968	other_l2cv=0.0124	other_contig=0.8189	other_run=5.5	other_gib=28	ro_cost_ms=57-->
<!--DATA 6	unpinned	pp2048	127.64-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

#### summary: per-arm mean over 6 passes, ratios paired within a pass against [unpinned]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | unpinned | 118.98 | 6 | 115.87 | 127.64 | -- | -- |
| pp2048 | poll0 | 120.27 | 6 | 116.24 | 129.01 | 1.013x | 1.099 0.995 0.972 1.077 1.021 0.911 |
| pp2048 | pin76t4 | 138.47 | 6 | 133.59 | 149.71 | 1.165x | 1.153 1.149 1.114 1.151 1.292 1.128 |
| pp2048 | pinpoll | 136.36 | 6 | 133.18 | 147.00 | 1.146x | 1.134 1.146 1.119 1.151 1.176 1.152 |

