== deepseek-v2-lite  Tue Sep 29 03:41:13 UTC 2026  cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000 ==
### deepseek-v2-lite [cpu-rp0] pass 1  env=''  args='-b 2048 -ub 2048 --repack 0'  03:42:22  clk=200 MHz  MemAvail=31800096 kB
<!--PRED 1	cpu-rp0	memavail_kb=31799804	anonhuge_kb=0	hugepagesz_kb=2048	buddy=2897,1868,1593,1201,760,541,389,289,195,151,5096	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |       8 |     2048 |   0 |          pp2048 |         18.74 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 1	cpu-rp0	wall_s=220	busy=20709,20480,17989,8013,20451,21118,21193,21155	busy_tot=151108	busy_little_share=0.4447	a55_cpu_cycles=1183633986354	a55_inst_retired=795044229215	context_switches=99759	a76_cpu_cycles=1758056444003	a76_l3d_cache_refill=3281837805	a76_l2d_cache_refill=1689971006	cpu_migrations=11303	page_faults=282153	a76_dtlb_walk=178151538	a76_mem_access=1389956532753	a76_inst_retired=5212903723580	a76_l1d_cache_refill=6098757468	a55_inst_share=0.1323	a76_ipc=2.965	l2ref_pki=0.324	l3ref_pki=0.630	l1dref_pki=1.170	memacc_pki=266.64	dtlbw_pki=0.0342	pmu_enabled=100.0	pmu_cpu_s=1757.0	who=at_s=6,rss_pg=2765001,settled=2	pfn_zero_frac=0.0000	maps=2	sampled=49152	present_frac=0.7604	l2color_cv=0.0027	l3color_cv=0.0080	contig_frac=0.9162	mean_run=11.7	vapa16=0.1160	gib_regions=27	anon_pg=12799	anon_l2cv=0.0076	anon_contig=0.7613	anon_run=4.2	anon_gib=23	file_pg=24576	file_l2cv=0.0011	file_contig=0.9969	file_run=199.8	file_gib=26	ro_cost_ms=76-->
<!--DATA 1	cpu-rp0	pp2048	18.74-->

### deepseek-v2-lite [cpu-rp1] pass 1  env=''  args='-b 2048 -ub 2048 --repack 1'  03:47:11  clk=200 MHz  MemAvail=31831872 kB
<!--PRED 1	cpu-rp1	memavail_kb=31831872	anonhuge_kb=0	hugepagesz_kb=2048	buddy=2135,2392,2209,1300,918,511,371,275,191,138,5109	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |       8 |     2048 |          pp2048 |         23.62 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 1	cpu-rp1	wall_s=184	busy=13403,13065,12376,8593,15693,16461,15573,15274	busy_tot=110438	busy_little_share=0.4295	a55_cpu_cycles=831080544504	a55_inst_retired=594603132445	context_switches=68787	a76_cpu_cycles=1327924088407	a76_l3d_cache_refill=3245510653	a76_l2d_cache_refill=1376837917	cpu_migrations=12593	page_faults=2320486	a76_dtlb_walk=141760729	a76_mem_access=1143972723690	a76_inst_retired=3407066790518	a76_l1d_cache_refill=10187268162	a55_inst_share=0.1486	a76_ipc=2.566	l2ref_pki=0.404	l3ref_pki=0.953	l1dref_pki=2.990	memacc_pki=335.76	dtlbw_pki=0.0416	pmu_enabled=100.0	pmu_cpu_s=1465.5	who=at_s=14,rss_pg=4784607,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6913	l2color_cv=0.0017	l3color_cv=0.0048	contig_frac=0.9772	mean_run=40.3	vapa16=0.0067	gib_regions=27	anon_pg=26486	anon_l2cv=0.0033	anon_contig=0.9572	anon_run=22.3	anon_gib=23	file_pg=24484	file_l2cv=0.0003	file_contig=0.9988	file_run=313.9	file_gib=22	ro_cost_ms=187-->
<!--DATA 1	cpu-rp1	pp2048	23.62-->

### deepseek-v2-lite [npu] pass 1  env=''  args='-b 2048 -ub 2048 --repack 0'  03:51:12  clk=200 MHz  MemAvail=31837912 kB
<!--PRED 1	npu	memavail_kb=31837376	anonhuge_kb=0	hugepagesz_kb=2048	buddy=2691,2615,2182,1620,920,612,453,335,235,154,5072	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | ngl | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |   0 |          pp2048 |         34.26 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 1	npu	wall_s=156	busy=1330,1218,940,942,8396,6056,7278,5235	busy_tot=31395	busy_little_share=0.1411	a55_cpu_cycles=91030109913	a55_inst_retired=46674050301	context_switches=2412188	a76_cpu_cycles=584798100204	a76_l3d_cache_refill=6979235215	a76_l2d_cache_refill=3023438026	cpu_migrations=52921	page_faults=15034679	a76_dtlb_walk=294677660	a76_mem_access=265908271600	a76_inst_retired=938266402671	a76_l1d_cache_refill=8652183516	a55_inst_share=0.0474	a76_ipc=1.604	l2ref_pki=3.222	l3ref_pki=7.438	l1dref_pki=9.221	memacc_pki=283.40	dtlbw_pki=0.3141	pmu_enabled=100.0	pmu_cpu_s=1243.6	who=at_s=8,rss_pg=2905450,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=97280	present_frac=0.8366	l2color_cv=0.0038	l3color_cv=0.0098	contig_frac=0.9151	mean_run=11.5	vapa16=0.0543	gib_regions=27	anon_pg=33735	anon_l2cv=0.0012	anon_contig=0.8610	anon_run=7.1	anon_gib=18	file_pg=24576	file_l2cv=0.0012	file_contig=0.9973	file_run=215.6	file_gib=23	other_pg=23076	other_l2cv=0.0133	other_contig=0.9068	other_run=10.5	other_gib=21	ro_cost_ms=114-->
<!--DATA 1	npu	pp2048	34.26-->
    [moe-int8] experts exercised: 4746 resident on the NPU (14855MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24880MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [rocket] MoE native-quant experts ON: q4_K -> int8, group=512 (nKt=4), resident budget 24880MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### deepseek-v2-lite [cpu-rp1] pass 2  env=''  args='-b 2048 -ub 2048 --repack 1'  03:54:57  clk=200 MHz  MemAvail=31820668 kB
<!--PRED 2	cpu-rp1	memavail_kb=31820084	anonhuge_kb=0	hugepagesz_kb=2048	buddy=2830,2695,2665,1952,1591,1235,1009,802,608,454,4697	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |       8 |     2048 |          pp2048 |         23.68 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 2	cpu-rp1	wall_s=184	busy=13147,13261,12172,8233,15649,15715,16509,15253	busy_tot=109939	busy_little_share=0.4258	a55_cpu_cycles=820969017575	a55_inst_retired=586240625992	context_switches=72511	a76_cpu_cycles=1331205939611	a76_l3d_cache_refill=3263058818	a76_l2d_cache_refill=1377937238	cpu_migrations=12792	page_faults=2323608	a76_dtlb_walk=144044128	a76_mem_access=1146527556090	a76_inst_retired=3415568773589	a76_l1d_cache_refill=10206577039	a55_inst_share=0.1465	a76_ipc=2.566	l2ref_pki=0.403	l3ref_pki=0.955	l1dref_pki=2.988	memacc_pki=335.68	dtlbw_pki=0.0422	pmu_enabled=100.0	pmu_cpu_s=1465.5	who=at_s=14,rss_pg=4784695,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6916	l2color_cv=0.0012	l3color_cv=0.0037	contig_frac=0.9746	mean_run=36.5	vapa16=0.0047	gib_regions=28	anon_pg=26648	anon_l2cv=0.0025	anon_contig=0.9524	anon_run=20.1	anon_gib=25	file_pg=24340	file_l2cv=0.0003	file_contig=0.9990	file_run=333.4	file_gib=23	ro_cost_ms=71-->
<!--DATA 2	cpu-rp1	pp2048	23.68-->

### deepseek-v2-lite [npu] pass 2  env=''  args='-b 2048 -ub 2048 --repack 0'  03:58:58  clk=200 MHz  MemAvail=31836092 kB
<!--PRED 2	npu	memavail_kb=31835516	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3050,3056,2626,2030,1239,1036,826,673,541,454,4755	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | ngl | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |   0 |          pp2048 |         34.08 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 2	npu	wall_s=157	busy=1267,1154,994,963,8479,6072,7227,5283	busy_tot=31439	busy_little_share=0.1393	a55_cpu_cycles=90122747850	a55_inst_retired=46444779676	context_switches=2412448	a76_cpu_cycles=587199024886	a76_l3d_cache_refill=7012985125	a76_l2d_cache_refill=3033164566	cpu_migrations=53511	page_faults=15034777	a76_dtlb_walk=298846285	a76_mem_access=266202222833	a76_inst_retired=939445094073	a76_l1d_cache_refill=8666359747	a55_inst_share=0.0471	a76_ipc=1.600	l2ref_pki=3.229	l3ref_pki=7.465	l1dref_pki=9.225	memacc_pki=283.36	dtlbw_pki=0.3181	pmu_enabled=100.0	pmu_cpu_s=1250.2	who=at_s=8,rss_pg=2905478,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=97280	present_frac=0.8462	l2color_cv=0.0033	l3color_cv=0.0089	contig_frac=0.9110	mean_run=11.0	vapa16=0.0206	gib_regions=29	anon_pg=33735	anon_l2cv=0.0012	anon_contig=0.8611	anon_run=7.1	anon_gib=21	file_pg=24576	file_l2cv=0.0016	file_contig=0.9975	file_run=223.4	file_gib=24	other_pg=24009	other_l2cv=0.0122	other_contig=0.8926	other_run=9.1	other_gib=27	ro_cost_ms=103-->
<!--DATA 2	npu	pp2048	34.08-->
    [moe-int8] experts exercised: 4746 resident on the NPU (14855MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24877MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [rocket] MoE native-quant experts ON: q4_K -> int8, group=512 (nKt=4), resident budget 24877MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### deepseek-v2-lite [cpu-rp0] pass 2  env=''  args='-b 2048 -ub 2048 --repack 0'  04:02:45  clk=200 MHz  MemAvail=31824940 kB
<!--PRED 2	cpu-rp0	memavail_kb=31824452	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3278,3200,2446,2275,1662,1313,1080,857,662,499,4644	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |       8 |     2048 |   0 |          pp2048 |         18.86 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 2	cpu-rp0	wall_s=221	busy=20621,20464,17795,8772,20315,21159,21196,21192	busy_tot=151514	busy_little_share=0.4465	a55_cpu_cycles=1190916412041	a55_inst_retired=800231041579	context_switches=96036	a76_cpu_cycles=1756042280279	a76_l3d_cache_refill=3228484738	a76_l2d_cache_refill=1683478935	cpu_migrations=10816	page_faults=275252	a76_dtlb_walk=185450386	a76_mem_access=1388535738859	a76_inst_retired=5207312225098	a76_l1d_cache_refill=6084878660	a55_inst_share=0.1332	a76_ipc=2.965	l2ref_pki=0.323	l3ref_pki=0.620	l1dref_pki=1.169	memacc_pki=266.65	dtlbw_pki=0.0356	pmu_enabled=100.0	pmu_cpu_s=1759.6	who=at_s=6,rss_pg=2765074,settled=2	pfn_zero_frac=0.0000	maps=2	sampled=49152	present_frac=0.7604	l2color_cv=0.0011	l3color_cv=0.0055	contig_frac=0.9190	mean_run=12.1	vapa16=0.0050	gib_regions=26	anon_pg=12800	anon_l2cv=0.0029	anon_contig=0.7739	anon_run=4.4	anon_gib=14	file_pg=24576	file_l2cv=0.0008	file_contig=0.9945	file_run=135.0	file_gib=25	ro_cost_ms=77-->
<!--DATA 2	cpu-rp0	pp2048	18.86-->

### deepseek-v2-lite [npu] pass 3  env=''  args='-b 2048 -ub 2048 --repack 0'  04:07:23  clk=200 MHz  MemAvail=31827372 kB
<!--PRED 3	npu	memavail_kb=31827372	anonhuge_kb=0	hugepagesz_kb=2048	buddy=2806,3026,2478,2109,1446,1164,922,754,630,501,4684	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | ngl | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |   0 |          pp2048 |         33.88 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 3	npu	wall_s=157	busy=1353,1139,966,987,8517,6028,7309,5197	busy_tot=31496	busy_little_share=0.1411	a55_cpu_cycles=91414404091	a55_inst_retired=46887182393	context_switches=2414321	a76_cpu_cycles=586382293901	a76_l3d_cache_refill=6971576634	a76_l2d_cache_refill=3024369317	cpu_migrations=52740	page_faults=15034651	a76_dtlb_walk=294366137	a76_mem_access=265859422829	a76_inst_retired=938471000215	a76_l1d_cache_refill=8662381430	a55_inst_share=0.0476	a76_ipc=1.600	l2ref_pki=3.223	l3ref_pki=7.429	l1dref_pki=9.230	memacc_pki=283.29	dtlbw_pki=0.3137	pmu_enabled=100.0	pmu_cpu_s=1251.1	who=at_s=8,rss_pg=2905553,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=97280	present_frac=0.8531	l2color_cv=0.0038	l3color_cv=0.0088	contig_frac=0.9125	mean_run=11.2	vapa16=0.0296	gib_regions=29	anon_pg=34199	anon_l2cv=0.0014	anon_contig=0.8619	anon_run=7.2	anon_gib=21	file_pg=24576	file_l2cv=0.0005	file_contig=0.9986	file_run=299.7	file_gib=24	other_pg=24216	other_l2cv=0.0135	other_contig=0.8967	other_run=9.5	other_gib=24	ro_cost_ms=124-->
<!--DATA 3	npu	pp2048	33.88-->
    [moe-int8] experts exercised: 4746 resident on the NPU (14855MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24880MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [rocket] MoE native-quant experts ON: q4_K -> int8, group=512 (nKt=4), resident budget 24880MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### deepseek-v2-lite [cpu-rp0] pass 3  env=''  args='-b 2048 -ub 2048 --repack 0'  04:11:10  clk=200 MHz  MemAvail=31782704 kB
<!--PRED 3	cpu-rp0	memavail_kb=31781828	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3783,2983,2527,2095,1704,1474,1214,988,794,569,4536	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |       8 |     2048 |   0 |          pp2048 |         18.74 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 3	cpu-rp0	wall_s=220	busy=20935,20107,18333,7763,20702,21056,21184,21045	busy_tot=151125	busy_little_share=0.4443	a55_cpu_cycles=1182199303026	a55_inst_retired=793016941947	context_switches=102530	a76_cpu_cycles=1758897918913	a76_l3d_cache_refill=3203040818	a76_l2d_cache_refill=1697244482	cpu_migrations=11538	page_faults=295969	a76_dtlb_walk=169788547	a76_mem_access=1390459514676	a76_inst_retired=5215873263288	a76_l1d_cache_refill=6090832225	a55_inst_share=0.1320	a76_ipc=2.965	l2ref_pki=0.325	l3ref_pki=0.614	l1dref_pki=1.168	memacc_pki=266.58	dtlbw_pki=0.0326	pmu_enabled=100.0	pmu_cpu_s=1760.3	who=at_s=6,rss_pg=2765027,settled=2	pfn_zero_frac=0.0000	maps=2	sampled=49152	present_frac=0.7604	l2color_cv=0.0029	l3color_cv=0.0090	contig_frac=0.9152	mean_run=11.5	vapa16=0.2828	gib_regions=26	anon_pg=12800	anon_l2cv=0.0092	anon_contig=0.7578	anon_run=4.1	anon_gib=24	file_pg=24576	file_l2cv=0.0011	file_contig=0.9971	file_run=206.5	file_gib=25	ro_cost_ms=46-->
<!--DATA 3	cpu-rp0	pp2048	18.74-->

### deepseek-v2-lite [cpu-rp1] pass 3  env=''  args='-b 2048 -ub 2048 --repack 1'  04:15:59  clk=200 MHz  MemAvail=31787876 kB
<!--PRED 3	cpu-rp1	memavail_kb=31787584	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3170,3228,1921,1937,1504,1296,1078,885,743,559,4589	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |       8 |     2048 |          pp2048 |         23.67 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 3	cpu-rp1	wall_s=184	busy=13324,13298,12099,8183,15828,15523,16478,15274	busy_tot=110007	busy_little_share=0.4264	a55_cpu_cycles=822825583836	a55_inst_retired=587406559299	context_switches=71648	a76_cpu_cycles=1330367930521	a76_l3d_cache_refill=3265319511	a76_l2d_cache_refill=1366068660	cpu_migrations=13120	page_faults=2322494	a76_dtlb_walk=157853660	a76_mem_access=1146232920308	a76_inst_retired=3414613864559	a76_l1d_cache_refill=10204490220	a55_inst_share=0.1468	a76_ipc=2.567	l2ref_pki=0.400	l3ref_pki=0.956	l1dref_pki=2.988	memacc_pki=335.68	dtlbw_pki=0.0462	pmu_enabled=100.0	pmu_cpu_s=1469.0	who=at_s=14,rss_pg=4784695,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6917	l2color_cv=0.0010	l3color_cv=0.0042	contig_frac=0.9743	mean_run=36.0	vapa16=0.1458	gib_regions=28	anon_pg=26703	anon_l2cv=0.0017	anon_contig=0.9518	anon_run=19.9	anon_gib=25	file_pg=24292	file_l2cv=0.0003	file_contig=0.9989	file_run=323.9	file_gib=20	ro_cost_ms=167-->
<!--DATA 3	cpu-rp1	pp2048	23.67-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [cpu-rp0]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | cpu-rp0 | 18.78 | 3 | 18.74 | 18.86 | -- | -- |
| pp2048 | cpu-rp1 | 23.66 | 3 | 23.62 | 23.68 | 1.260x | 1.260 1.256 1.263 |
| pp2048 | npu | 34.07 | 3 | 33.88 | 34.26 | 1.814x | 1.828 1.807 1.808 |

