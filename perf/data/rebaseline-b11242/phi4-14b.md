== phi4-14b  Tue Sep 29 05:12:35 UTC 2026  cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000 ==
### phi4-14b [cpu-rp0] pass 1  env=''  args='-b 2048 -ub 2048 --repack 0'  05:17:49  clk=200 MHz  MemAvail=31820064 kB
<!--PRED 1	cpu-rp0	memavail_kb=31820312	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3570,2915,2971,2465,1766,1354,1157,992,875,758,4784	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --: | --------------: | -------------------: |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | CPU        |       8 |     2048 |   0 |          pp2048 |          3.39 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 1	cpu-rp0	wall_s=1212	busy=117947,115089,103761,50052,119523,119797,119963,119800	busy_tot=865932	busy_little_share=0.4467	a55_cpu_cycles=6806237753541	a55_inst_retired=4182410376837	context_switches=376282	a76_cpu_cycles=10000802908316	a76_l3d_cache_refill=32409729703	a76_l2d_cache_refill=6223873156	cpu_migrations=21565	page_faults=373971	a76_dtlb_walk=702943307	a76_mem_access=7379047281380	a76_inst_retired=28861747621878	a76_l1d_cache_refill=58599956488	a55_inst_share=0.1266	a76_ipc=2.886	l2ref_pki=0.216	l3ref_pki=1.123	l1dref_pki=2.030	memacc_pki=255.67	dtlbw_pki=0.0244	pmu_enabled=100.0	pmu_cpu_s=9691.3	who=at_s=6,rss_pg=2335532,settled=2	pfn_zero_frac=0.0000	maps=2	sampled=49152	present_frac=0.7188	l2color_cv=0.0013	l3color_cv=0.0041	contig_frac=0.9525	mean_run=20.3	vapa16=0.0830	gib_regions=26	anon_pg=10752	anon_l2cv=0.0038	anon_contig=0.8479	anon_run=6.5	anon_gib=14	file_pg=24576	file_l2cv=0.0014	file_contig=0.9983	file_run=273.1	file_gib=25	ro_cost_ms=131-->
<!--DATA 1	cpu-rp0	pp2048	3.39-->

### phi4-14b [cpu-rp1] pass 1  env=''  args='-b 2048 -ub 2048 --repack 1'  05:41:31  clk=200 MHz  MemAvail=31800756 kB
<!--PRED 1	cpu-rp1	memavail_kb=31800172	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3422,3386,2684,1862,1619,1267,1097,941,836,720,4828	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | CPU        |       8 |     2048 |          pp2048 |          5.46 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 1	cpu-rp1	wall_s=763	busy=66748,66465,59483,39774,72483,72098,73255,71913	busy_tot=522219	busy_little_share=0.4452	a55_cpu_cycles=4079730367577	a55_inst_retired=2419886219053	context_switches=197880	a76_cpu_cycles=6059969693958	a76_l3d_cache_refill=78639492036	a76_l2d_cache_refill=15772193306	cpu_migrations=18327	page_faults=2415398	a76_dtlb_walk=257712097	a76_mem_access=5507886672172	a76_inst_retired=12852083633336	a76_l1d_cache_refill=85828299541	a55_inst_share=0.1585	a76_ipc=2.121	l2ref_pki=1.227	l3ref_pki=6.119	l1dref_pki=6.678	memacc_pki=428.56	dtlbw_pki=0.0201	pmu_enabled=100.0	pmu_cpu_s=6091.9	who=at_s=14,rss_pg=4337671,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.9614	l2color_cv=0.0004	l3color_cv=0.0023	contig_frac=0.9912	mean_run=93.4	vapa16=0.1667	gib_regions=28	anon_pg=46308	anon_l2cv=0.0005	anon_contig=0.9872	anon_run=67.7	anon_gib=27	file_pg=24576	file_l2cv=0.0005	file_contig=0.9989	file_run=327.7	file_gib=23	ro_cost_ms=91-->
<!--DATA 1	cpu-rp1	pp2048	5.46-->

### phi4-14b [npu] pass 1  env=''  args='-b 2048 -ub 2048 --repack 0'  05:55:53  clk=200 MHz  MemAvail=31822804 kB
<!--PRED 1	npu	memavail_kb=31823136	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3647,3081,2877,2191,1543,1282,1096,914,818,716,4841	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | ngl | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | ROCKET     |  -1 |     2048 |   0 |          pp2048 |         16.63 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 1	npu	wall_s=250	busy=957,963,973,956,12180,9091,10415,9895	busy_tot=45430	busy_little_share=0.0847	a55_cpu_cycles=80256819484	a55_inst_retired=33193807906	context_switches=7068611	a76_cpu_cycles=905512116299	a76_l3d_cache_refill=10980023977	a76_l2d_cache_refill=4161113399	cpu_migrations=6177	page_faults=1581317	a76_dtlb_walk=508987385	a76_mem_access=191742041795	a76_inst_retired=1339384613996	a76_l1d_cache_refill=9033032418	a55_inst_share=0.0242	a76_ipc=1.479	l2ref_pki=3.107	l3ref_pki=8.198	l1dref_pki=6.744	memacc_pki=143.16	dtlbw_pki=0.3800	pmu_enabled=100.0	pmu_cpu_s=1997.8	who=at_s=10,rss_pg=2941205,settled=2	pfn_zero_frac=0.0000	maps=23	sampled=472064	present_frac=0.5738	l2color_cv=0.0019	l3color_cv=0.0039	contig_frac=0.9446	mean_run=17.4	vapa16=0.0344	gib_regions=29	accel_pg=0	accel_l2cv=0.0000	accel_contig=0.0000	accel_run=0.0	accel_gib=0	anon_pg=233339	anon_l2cv=0.0021	anon_contig=0.9666	anon_run=28.3	anon_gib=28	file_pg=24576	file_l2cv=0.0013	file_contig=0.9957	file_run=159.6	file_gib=27	other_pg=12974	other_l2cv=0.0111	other_contig=0.4506	other_run=1.8	other_gib=28	ro_cost_ms=323-->
<!--DATA 1	npu	pp2048	16.63-->

### phi4-14b [cpu-rp1] pass 2  env=''  args='-b 2048 -ub 2048 --repack 1'  06:03:34  clk=200 MHz  MemAvail=31823120 kB
<!--PRED 2	cpu-rp1	memavail_kb=31822528	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3150,3236,2932,2335,1527,1276,1063,911,773,658,4883	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | CPU        |       8 |     2048 |          pp2048 |          5.44 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 2	cpu-rp1	wall_s=764	busy=67216,65852,58145,41132,72284,73117,72473,72053	busy_tot=522272	busy_little_share=0.4449	a55_cpu_cycles=4078065449352	a55_inst_retired=2418954551282	context_switches=205782	a76_cpu_cycles=6064390253387	a76_l3d_cache_refill=78774594972	a76_l2d_cache_refill=15782848318	cpu_migrations=18757	page_faults=2446853	a76_dtlb_walk=259247800	a76_mem_access=5509683311693	a76_inst_retired=12854201889550	a76_l1d_cache_refill=85917911560	a55_inst_share=0.1584	a76_ipc=2.120	l2ref_pki=1.228	l3ref_pki=6.128	l1dref_pki=6.684	memacc_pki=428.63	dtlbw_pki=0.0202	pmu_enabled=100.0	pmu_cpu_s=6111.0	who=at_s=14,rss_pg=4337815,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.9628	l2color_cv=0.0006	l3color_cv=0.0021	contig_frac=0.9879	mean_run=67.5	vapa16=0.2096	gib_regions=28	anon_pg=46409	anon_l2cv=0.0009	anon_contig=0.9822	anon_run=47.6	anon_gib=27	file_pg=24576	file_l2cv=0.0003	file_contig=0.9988	file_run=319.2	file_gib=23	ro_cost_ms=258-->
<!--DATA 2	cpu-rp1	pp2048	5.44-->

### phi4-14b [npu] pass 2  env=''  args='-b 2048 -ub 2048 --repack 0'  06:17:59  clk=200 MHz  MemAvail=31817156 kB
<!--PRED 2	npu	memavail_kb=31816272	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3128,3134,2733,2212,1548,1191,1042,887,737,650,4903	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | ngl | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | ROCKET     |  -1 |     2048 |   0 |          pp2048 |         16.73 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 2	npu	wall_s=249	busy=977,972,947,959,12140,9279,9823,10135	busy_tot=45232	busy_little_share=0.0852	a55_cpu_cycles=80581414525	a55_inst_retired=33555654046	context_switches=7068296	a76_cpu_cycles=901548689001	a76_l3d_cache_refill=10999773251	a76_l2d_cache_refill=4156790524	cpu_migrations=6368	page_faults=1586650	a76_dtlb_walk=506089061	a76_mem_access=191696536589	a76_inst_retired=1339103486725	a76_l1d_cache_refill=9010418095	a55_inst_share=0.0244	a76_ipc=1.485	l2ref_pki=3.104	l3ref_pki=8.214	l1dref_pki=6.729	memacc_pki=143.15	dtlbw_pki=0.3779	pmu_enabled=100.0	pmu_cpu_s=1987.3	who=at_s=10,rss_pg=2941205,settled=2	pfn_zero_frac=0.0000	maps=23	sampled=472064	present_frac=0.5765	l2color_cv=0.0021	l3color_cv=0.0046	contig_frac=0.9463	mean_run=18.0	vapa16=0.0452	gib_regions=29	accel_pg=0	accel_l2cv=0.0000	accel_contig=0.0000	accel_run=0.0	accel_gib=0	anon_pg=234715	anon_l2cv=0.0024	anon_contig=0.9666	anon_run=28.3	anon_gib=29	file_pg=24576	file_l2cv=0.0006	file_contig=0.9981	file_run=258.7	file_gib=25	other_pg=12875	other_l2cv=0.0080	other_contig=0.4775	other_run=1.9	other_gib=27	ro_cost_ms=363-->
<!--DATA 2	npu	pp2048	16.73-->

### phi4-14b [cpu-rp0] pass 2  env=''  args='-b 2048 -ub 2048 --repack 0'  06:27:21  clk=200 MHz  MemAvail=31804184 kB
<!--PRED 2	cpu-rp0	memavail_kb=31804140	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3976,2760,3019,2062,1494,1221,1009,872,733,602,4929	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --: | --------------: | -------------------: |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | CPU        |       8 |     2048 |   0 |          pp2048 |          3.38 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 2	cpu-rp0	wall_s=1215	busy=117947,116119,105224,42687,119626,119997,120312,120079	busy_tot=861991	busy_little_share=0.4431	a55_cpu_cycles=6721193463864	a55_inst_retired=4131328074517	context_switches=399156	a76_cpu_cycles=10020508015471	a76_l3d_cache_refill=32492503804	a76_l2d_cache_refill=6259931244	cpu_migrations=22500	page_faults=431769	a76_dtlb_walk=707838526	a76_mem_access=7393783555765	a76_inst_retired=28915211065726	a76_l1d_cache_refill=58818633168	a55_inst_share=0.1250	a76_ipc=2.886	l2ref_pki=0.216	l3ref_pki=1.124	l1dref_pki=2.034	memacc_pki=255.71	dtlbw_pki=0.0245	pmu_enabled=100.0	pmu_cpu_s=9712.2	who=at_s=6,rss_pg=2335787,settled=2	pfn_zero_frac=0.0000	maps=2	sampled=49152	present_frac=0.7188	l2color_cv=0.0028	l3color_cv=0.0072	contig_frac=0.9400	mean_run=16.2	vapa16=0.0100	gib_regions=28	anon_pg=10752	anon_l2cv=0.0070	anon_contig=0.8314	anon_run=5.9	anon_gib=20	file_pg=24576	file_l2cv=0.0024	file_contig=0.9876	file_run=69.6	file_gib=28	ro_cost_ms=72-->
<!--DATA 2	cpu-rp0	pp2048	3.38-->

### phi4-14b [npu] pass 3  env=''  args='-b 2048 -ub 2048 --repack 0'  06:49:16  clk=200 MHz  MemAvail=31811468 kB
<!--PRED 3	npu	memavail_kb=31810396	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3078,3615,3202,2204,1417,1151,1014,867,718,612,4930	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | ngl | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | ROCKET     |  -1 |     2048 |   0 |          pp2048 |         16.65 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 3	npu	wall_s=250	busy=976,944,970,948,11987,9137,10420,10061	busy_tot=45443	busy_little_share=0.0845	a55_cpu_cycles=79830065084	a55_inst_retired=33109862346	context_switches=7069498	a76_cpu_cycles=906773659060	a76_l3d_cache_refill=11021009240	a76_l2d_cache_refill=4135916182	cpu_migrations=6229	page_faults=1581301	a76_dtlb_walk=507246768	a76_mem_access=191768933396	a76_inst_retired=1339528583058	a76_l1d_cache_refill=8984049376	a55_inst_share=0.0241	a76_ipc=1.477	l2ref_pki=3.088	l3ref_pki=8.228	l1dref_pki=6.707	memacc_pki=143.16	dtlbw_pki=0.3787	pmu_enabled=100.0	pmu_cpu_s=1997.1	who=at_s=10,rss_pg=2941076,settled=2	pfn_zero_frac=0.0000	maps=23	sampled=472576	present_frac=0.5729	l2color_cv=0.0018	l3color_cv=0.0047	contig_frac=0.9463	mean_run=17.9	vapa16=0.0224	gib_regions=29	accel_pg=0	accel_l2cv=0.0000	accel_contig=0.0000	accel_run=0.0	accel_gib=0	anon_pg=233219	anon_l2cv=0.0020	anon_contig=0.9668	anon_run=28.4	anon_gib=27	file_pg=24576	file_l2cv=0.0038	file_contig=0.9859	file_run=62.4	file_gib=28	other_pg=12939	other_l2cv=0.0154	other_contig=0.5005	other_run=2.0	other_gib=28	ro_cost_ms=335-->
<!--DATA 3	npu	pp2048	16.65-->

### phi4-14b [cpu-rp0] pass 3  env=''  args='-b 2048 -ub 2048 --repack 0'  06:58:40  clk=200 MHz  MemAvail=31817520 kB
<!--PRED 3	cpu-rp0	memavail_kb=31816916	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3047,3224,2759,2316,1654,1206,1029,858,711,584,4944	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --: | --------------: | -------------------: |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | CPU        |       8 |     2048 |   0 |          pp2048 |          3.37 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 3	cpu-rp0	wall_s=1216	busy=118754,112698,109825,47022,119773,119363,120011,119898	busy_tot=867344	busy_little_share=0.4477	a55_cpu_cycles=6835465622214	a55_inst_retired=4199306090241	context_switches=380496	a76_cpu_cycles=10002138817204	a76_l3d_cache_refill=32511818420	a76_l2d_cache_refill=6243427298	cpu_migrations=21706	page_faults=385979	a76_dtlb_walk=672168786	a76_mem_access=7377739672879	a76_inst_retired=28845852239448	a76_l1d_cache_refill=58750632352	a55_inst_share=0.1271	a76_ipc=2.884	l2ref_pki=0.216	l3ref_pki=1.127	l1dref_pki=2.037	memacc_pki=255.76	dtlbw_pki=0.0233	pmu_enabled=100.0	pmu_cpu_s=9714.4	who=at_s=6,rss_pg=2335747,settled=2	pfn_zero_frac=0.0000	maps=2	sampled=49152	present_frac=0.7188	l2color_cv=0.0016	l3color_cv=0.0070	contig_frac=0.9513	mean_run=19.8	vapa16=0.0079	gib_regions=28	anon_pg=10752	anon_l2cv=0.0043	anon_contig=0.8488	anon_run=6.5	anon_gib=21	file_pg=24576	file_l2cv=0.0008	file_contig=0.9961	file_run=171.9	file_gib=23	ro_cost_ms=44-->
<!--DATA 3	cpu-rp0	pp2048	3.37-->

### phi4-14b [cpu-rp1] pass 3  env=''  args='-b 2048 -ub 2048 --repack 1'  07:22:26  clk=200 MHz  MemAvail=31799820 kB
<!--PRED 3	cpu-rp1	memavail_kb=31799820	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3464,3522,2590,2099,1532,1168,1010,862,720,590,4939	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | CPU        |       8 |     2048 |          pp2048 |          5.46 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 3	cpu-rp1	wall_s=762	busy=66448,66968,61403,38483,73066,72373,72291,71846	busy_tot=522878	busy_little_share=0.4462	a55_cpu_cycles=4093624407421	a55_inst_retired=2432112290384	context_switches=194108	a76_cpu_cycles=6055618514423	a76_l3d_cache_refill=78812508647	a76_l2d_cache_refill=15756099295	cpu_migrations=18357	page_faults=2407047	a76_dtlb_walk=256873005	a76_mem_access=5502453712031	a76_inst_retired=12839337690821	a76_l1d_cache_refill=85710341407	a55_inst_share=0.1593	a76_ipc=2.120	l2ref_pki=1.227	l3ref_pki=6.138	l1dref_pki=6.676	memacc_pki=428.56	dtlbw_pki=0.0200	pmu_enabled=100.0	pmu_cpu_s=6090.3	who=at_s=14,rss_pg=4338283,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.9609	l2color_cv=0.0008	l3color_cv=0.0023	contig_frac=0.9895	mean_run=77.9	vapa16=0.3414	gib_regions=28	anon_pg=46270	anon_l2cv=0.0011	anon_contig=0.9845	anon_run=55.3	anon_gib=28	file_pg=24576	file_l2cv=0.0004	file_contig=0.9990	file_run=336.7	file_gib=22	ro_cost_ms=260-->
<!--DATA 3	cpu-rp1	pp2048	5.46-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [cpu-rp0]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | cpu-rp0 | 3.38 | 3 | 3.37 | 3.39 | -- | -- |
| pp2048 | cpu-rp1 | 5.45 | 3 | 5.44 | 5.46 | 1.613x | 1.611 1.609 1.620 |
| pp2048 | npu | 16.67 | 3 | 16.63 | 16.73 | 4.932x | 4.906 4.950 4.941 |

