<!-- qwen35-9b Q4_K_M, pinning x -b 2048 -ub 2048 (a NON-residency knob) 2x2
     TESTS='-p 2048 -n 0 -r 3'  PASSES=6
     gguf=<data>/qwen35/Qwen3.5-9B-Q4_K_M.gguf (5680522464 bytes)
     stock_unpin | stock_pin = PIN_MASK=0xf0 -t 4
     ub_unpin    = -b 2048 -ub 2048   (the published 1.424x arm)
     ub_pin      = the same + PIN_MASK=0xf0 -t 4
     interaction = (ub_pin/stock_pin) / (ub_unpin/stock_unpin), paired within a pass
     no arm places resident weights; an arm that reports any is not this contrast
-->
== qwen35-9b-ub-pin2x2  Wed Sep  2 04:54:50 UTC 2026 ==
### qwen35-9b-ub-pin2x2 [stock_unpin] pass 1  env=''  args=''  04:55:56  clk=600 MHz  MemAvail=31640816 kB
<!--PRED 1	stock_unpin	memavail_kb=31640984	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9018,10484,11215,11257,8234,7601,6675,5603,4531,2496,2390-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         19.12 ± 0.04 |

build: 171974745 (10558)
<!--RO 1	stock_unpin	wall_s=431	busy=6924,5609,5468,5349,25089,22380,21992,22749	busy_tot=115560	busy_little_share=0.2021	a55_cpu_cycles=429384348512	a55_inst_retired=224283609189	context_switches=7970081	a76_cpu_cycles=1991487777172	a76_l3d_cache_refill=15888706837	a76_l2d_cache_refill=8243821345	cpu_migrations=41248	page_faults=609128	a76_dtlb_walk=1819460442	a76_mem_access=657746260635	a76_inst_retired=4131674512576	a76_l1d_cache_refill=13664551462	a55_inst_share=0.0515	a76_ipc=2.075	l2ref_pki=1.995	l3ref_pki=3.846	l1dref_pki=3.307	memacc_pki=159.20	dtlbw_pki=0.4404	pmu_enabled=100.0	pmu_cpu_s=3448.0	who=at_s=6,rss_pg=1509853,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6976	l2color_cv=0.0047	l3color_cv=0.0088	contig_frac=0.8940	mean_run=9.3	vapa16=0.0246	gib_regions=26	anon_pg=23049	anon_l2cv=0.0023	anon_contig=0.9456	anon_run=17.7	anon_gib=24	file_pg=24576	file_l2cv=0.0010	file_contig=0.9938	file_run=122.9	file_gib=25	other_pg=20951	other_l2cv=0.0140	other_contig=0.7201	other_run=3.6	other_gib=26	ro_cost_ms=68-->
<!--DATA 1	stock_unpin	pp2048	19.12-->

### qwen35-9b-ub-pin2x2 [stock_pin] pass 1  env='PIN_MASK=0xf0'  args='-t 4'  05:04:07  clk=600 MHz  MemAvail=31611760 kB
<!--PRED 1	stock_pin	memavail_kb=31611760	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8981,11247,10684,10630,7991,7467,6539,5519,4481,2500,2426-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.55 ± 0.03 |

build: 171974745 (10558)
<!--RO 1	stock_pin	wall_s=401	busy=897,293,238,214,25075,24415,22487,23061	busy_tot=96680	busy_little_share=0.0170	a55_cpu_cycles=50069557750	a55_inst_retired=14251199510	context_switches=7849747	a76_cpu_cycles=2048816517071	a76_l3d_cache_refill=16744191416	a76_l2d_cache_refill=8421905709	cpu_migrations=21332	page_faults=590760	a76_dtlb_walk=2275268971	a76_mem_access=710918312982	a76_inst_retired=4232741307545	a76_l1d_cache_refill=15441433870	a55_inst_share=0.0034	a76_ipc=2.066	l2ref_pki=1.990	l3ref_pki=3.956	l1dref_pki=3.648	memacc_pki=167.96	dtlbw_pki=0.5375	pmu_enabled=100.0	pmu_cpu_s=3204.9	who=at_s=6,rss_pg=1509828,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6838	l2color_cv=0.0035	l3color_cv=0.0088	contig_frac=0.8714	mean_run=7.7	vapa16=0.0093	gib_regions=27	anon_pg=22019	anon_l2cv=0.0025	anon_contig=0.9115	anon_run=11.1	anon_gib=24	file_pg=24576	file_l2cv=0.0008	file_contig=0.9945	file_run=135.0	file_gib=26	other_pg=20629	other_l2cv=0.0112	other_contig=0.6819	other_run=3.1	other_gib=26	ro_cost_ms=110-->
<!--DATA 1	stock_pin	pp2048	20.55-->

### qwen35-9b-ub-pin2x2 [ub_unpin] pass 1  env=''  args='-b 2048 -ub 2048'  05:11:54  clk=600 MHz  MemAvail=31660560 kB
<!--PRED 1	ub_unpin	memavail_kb=31660560	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9083,10663,11230,10970,7682,7372,6462,5459,4455,2515,2455-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         26.87 ± 0.08 |

build: 171974745 (10558)
<!--RO 1	ub_unpin	wall_s=309	busy=4432,4192,4090,3720,12203,10994,11122,11102	busy_tot=61855	busy_little_share=0.2657	a55_cpu_cycles=297287591758	a55_inst_retired=144896537436	context_switches=7268533	a76_cpu_cycles=995552696689	a76_l3d_cache_refill=11471791310	a76_l2d_cache_refill=5689662009	cpu_migrations=16626	page_faults=1078366	a76_dtlb_walk=2170805638	a76_mem_access=419431888257	a76_inst_retired=1761614843792	a76_l1d_cache_refill=10238151553	a55_inst_share=0.0760	a76_ipc=1.769	l2ref_pki=3.230	l3ref_pki=6.512	l1dref_pki=5.812	memacc_pki=238.10	dtlbw_pki=1.2323	pmu_enabled=100.0	pmu_cpu_s=2462.3	who=at_s=8,rss_pg=1651218,settled=2	pfn_zero_frac=0.0000	maps=5	sampled=116224	present_frac=0.7633	l2color_cv=0.0026	l3color_cv=0.0063	contig_frac=0.8103	mean_run=5.2	vapa16=0.0252	gib_regions=26	anon_pg=46628	anon_l2cv=0.0005	anon_contig=0.7558	anon_run=4.1	anon_gib=25	file_pg=24576	file_l2cv=0.0013	file_contig=0.9942	file_run=129.3	file_gib=24	other_pg=17515	other_l2cv=0.0141	other_contig=0.6972	other_run=3.3	other_gib=26	ro_cost_ms=102-->
<!--DATA 1	ub_unpin	pp2048	26.87-->

### qwen35-9b-ub-pin2x2 [ub_pin] pass 1  env='PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4'  05:18:03  clk=600 MHz  MemAvail=31615260 kB
<!--PRED 1	ub_pin	memavail_kb=31614968	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8711,11076,10367,10439,8113,7299,6453,5444,4439,2523,2449-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         28.62 ± 0.05 |

build: 171974745 (10558)
<!--RO 1	ub_pin	wall_s=290	busy=196,106,118,124,12441,11341,12043,12135	busy_tot=48504	busy_little_share=0.0112	a55_cpu_cycles=20218092839	a55_inst_retired=4489306523	context_switches=7212530	a76_cpu_cycles=1038496278608	a76_l3d_cache_refill=11986936357	a76_l2d_cache_refill=5729060032	cpu_migrations=5600	page_faults=1073560	a76_dtlb_walk=2782022576	a76_mem_access=465269040933	a76_inst_retired=1868400854923	a76_l1d_cache_refill=11369141110	a55_inst_share=0.0024	a76_ipc=1.799	l2ref_pki=3.066	l3ref_pki=6.416	l1dref_pki=6.085	memacc_pki=249.02	dtlbw_pki=1.4890	pmu_enabled=100.0	pmu_cpu_s=2314.4	who=at_s=8,rss_pg=1651197,settled=2	pfn_zero_frac=0.0000	maps=5	sampled=116224	present_frac=0.7594	l2color_cv=0.0029	l3color_cv=0.0156	contig_frac=0.7995	mean_run=4.9	vapa16=0.0372	gib_regions=27	anon_pg=46155	anon_l2cv=0.0003	anon_contig=0.7370	anon_run=3.8	anon_gib=25	file_pg=24576	file_l2cv=0.0007	file_contig=0.9917	file_run=97.5	file_gib=26	other_pg=17534	other_l2cv=0.0145	other_contig=0.6949	other_run=3.3	other_gib=26	ro_cost_ms=132-->
<!--DATA 1	ub_pin	pp2048	28.62-->

### qwen35-9b-ub-pin2x2 [stock_pin] pass 2  env='PIN_MASK=0xf0'  args='-t 4'  05:23:53  clk=600 MHz  MemAvail=31618516 kB
<!--PRED 2	stock_pin	memavail_kb=31618516	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9078,10677,11019,10356,7594,7181,6386,5393,4412,2526,2476-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.74 ± 0.03 |

build: 171974745 (10558)
<!--RO 2	stock_pin	wall_s=398	busy=684,305,287,253,24891,24704,22258,22943	busy_tot=96325	busy_little_share=0.0159	a55_cpu_cycles=48616560722	a55_inst_retired=13284922269	context_switches=7842424	a76_cpu_cycles=2042972509882	a76_l3d_cache_refill=16904349960	a76_l2d_cache_refill=8400627021	cpu_migrations=20976	page_faults=616753	a76_dtlb_walk=1880771820	a76_mem_access=711287318422	a76_inst_retired=4234152112703	a76_l1d_cache_refill=15415172913	a55_inst_share=0.0031	a76_ipc=2.073	l2ref_pki=1.984	l3ref_pki=3.992	l1dref_pki=3.641	memacc_pki=167.99	dtlbw_pki=0.4442	pmu_enabled=100.0	pmu_cpu_s=3178.9	who=at_s=6,rss_pg=1509820,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6991	l2color_cv=0.0062	l3color_cv=0.0097	contig_frac=0.8436	mean_run=6.3	vapa16=0.0410	gib_regions=25	anon_pg=23042	anon_l2cv=0.0008	anon_contig=0.9302	anon_run=14.0	anon_gib=24	file_pg=24576	file_l2cv=0.0009	file_contig=0.9949	file_run=141.2	file_gib=25	other_pg=21109	other_l2cv=0.0210	other_contig=0.5730	other_run=2.3	other_gib=25	ro_cost_ms=117-->
<!--DATA 2	stock_pin	pp2048	20.74-->

### qwen35-9b-ub-pin2x2 [ub_unpin] pass 2  env=''  args='-b 2048 -ub 2048'  05:31:36  clk=600 MHz  MemAvail=31669048 kB
<!--PRED 2	ub_unpin	memavail_kb=31669284	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9787,10921,11011,10590,7623,7149,6367,5356,4382,2536,2494-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         26.78 ± 0.07 |

build: 171974745 (10558)
<!--RO 2	ub_unpin	wall_s=309	busy=4484,4148,4066,3748,12303,10628,11436,10941	busy_tot=61754	busy_little_share=0.2663	a55_cpu_cycles=297542999184	a55_inst_retired=144881789091	context_switches=7269616	a76_cpu_cycles=994312959253	a76_l3d_cache_refill=11463891006	a76_l2d_cache_refill=5673871433	cpu_migrations=16643	page_faults=1073673	a76_dtlb_walk=2163908248	a76_mem_access=419371916445	a76_inst_retired=1761937558036	a76_l1d_cache_refill=10212642401	a55_inst_share=0.0760	a76_ipc=1.772	l2ref_pki=3.220	l3ref_pki=6.506	l1dref_pki=5.796	memacc_pki=238.02	dtlbw_pki=1.2281	pmu_enabled=100.0	pmu_cpu_s=2466.9	who=at_s=8,rss_pg=1651209,settled=2	pfn_zero_frac=0.0000	maps=5	sampled=116224	present_frac=0.7626	l2color_cv=0.0035	l3color_cv=0.0066	contig_frac=0.7746	mean_run=4.4	vapa16=0.0185	gib_regions=27	anon_pg=46596	anon_l2cv=0.0008	anon_contig=0.6908	anon_run=3.2	anon_gib=24	file_pg=24576	file_l2cv=0.0019	file_contig=0.9868	file_run=66.2	file_gib=27	other_pg=17460	other_l2cv=0.0155	other_contig=0.6993	other_run=3.3	other_gib=26	ro_cost_ms=162-->
<!--DATA 2	ub_unpin	pp2048	26.78-->

### qwen35-9b-ub-pin2x2 [ub_pin] pass 2  env='PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4'  05:37:45  clk=600 MHz  MemAvail=31622824 kB
<!--PRED 2	ub_pin	memavail_kb=31622824	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8737,10428,10712,10253,7798,7128,6361,5351,4379,2534,2489-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         28.50 ± 0.10 |

build: 171974745 (10558)
<!--RO 2	ub_pin	wall_s=291	busy=147,110,127,107,12551,11306,12453,11781	busy_tot=48582	busy_little_share=0.0101	a55_cpu_cycles=19258242878	a55_inst_retired=4243179152	context_switches=7218976	a76_cpu_cycles=1041982239833	a76_l3d_cache_refill=12042556564	a76_l2d_cache_refill=5771178139	cpu_migrations=5732	page_faults=1071214	a76_dtlb_walk=3734211070	a76_mem_access=465179863598	a76_inst_retired=1868181482936	a76_l1d_cache_refill=11385778744	a55_inst_share=0.0023	a76_ipc=1.793	l2ref_pki=3.089	l3ref_pki=6.446	l1dref_pki=6.095	memacc_pki=249.00	dtlbw_pki=1.9988	pmu_enabled=100.0	pmu_cpu_s=2322.0	who=at_s=9,rss_pg=1651197,settled=2	pfn_zero_frac=0.0000	maps=5	sampled=116224	present_frac=0.7689	l2color_cv=0.0023	l3color_cv=0.0097	contig_frac=0.7993	mean_run=4.9	vapa16=0.0567	gib_regions=27	anon_pg=46988	anon_l2cv=0.0012	anon_contig=0.7295	anon_run=3.7	anon_gib=26	file_pg=24576	file_l2cv=0.0021	file_contig=0.9918	file_run=98.3	file_gib=26	other_pg=17798	other_l2cv=0.0115	other_contig=0.7179	other_run=3.5	other_gib=26	ro_cost_ms=148-->
<!--DATA 2	ub_pin	pp2048	28.50-->

### qwen35-9b-ub-pin2x2 [stock_unpin] pass 2  env=''  args=''  05:43:42  clk=600 MHz  MemAvail=31629160 kB
<!--PRED 2	stock_unpin	memavail_kb=31629208	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8817,10775,10623,10482,7788,7116,6341,5338,4373,2539,2491-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         19.30 ± 0.03 |

build: 171974745 (10558)
<!--RO 2	stock_unpin	wall_s=427	busy=6847,5655,5574,5330,24618,22392,21774,22524	busy_tot=114714	busy_little_share=0.2040	a55_cpu_cycles=431226265406	a55_inst_retired=227001219996	context_switches=7979234	a76_cpu_cycles=1975765725531	a76_l3d_cache_refill=14934742479	a76_l2d_cache_refill=8242837576	cpu_migrations=40199	page_faults=600118	a76_dtlb_walk=2182121684	a76_mem_access=656787435246	a76_inst_retired=4129131424386	a76_l1d_cache_refill=13743640991	a55_inst_share=0.0521	a76_ipc=2.090	l2ref_pki=1.996	l3ref_pki=3.617	l1dref_pki=3.328	memacc_pki=159.06	dtlbw_pki=0.5285	pmu_enabled=100.0	pmu_cpu_s=3411.1	who=at_s=6,rss_pg=1509848,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6851	l2color_cv=0.0029	l3color_cv=0.0124	contig_frac=0.8602	mean_run=7.1	vapa16=0.0049	gib_regions=27	anon_pg=22019	anon_l2cv=0.0033	anon_contig=0.8894	anon_run=8.9	anon_gib=25	file_pg=24576	file_l2cv=0.0005	file_contig=0.9938	file_run=122.3	file_gib=26	other_pg=20750	other_l2cv=0.0091	other_contig=0.6711	other_run=3.0	other_gib=26	ro_cost_ms=113-->
<!--DATA 2	stock_unpin	pp2048	19.30-->

### qwen35-9b-ub-pin2x2 [ub_unpin] pass 3  env=''  args='-b 2048 -ub 2048'  05:51:54  clk=600 MHz  MemAvail=31655176 kB
<!--PRED 3	ub_unpin	memavail_kb=31655176	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8030,11227,11000,10669,7560,7120,6338,5325,4363,2540,2501-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         27.02 ± 0.21 |

build: 171974745 (10558)
<!--RO 3	ub_unpin	wall_s=308	busy=4460,4132,3953,3652,12234,10548,11179,11448	busy_tot=61606	busy_little_share=0.2629	a55_cpu_cycles=292820322360	a55_inst_retired=143232204874	context_switches=7258298	a76_cpu_cycles=995026182432	a76_l3d_cache_refill=11508052175	a76_l2d_cache_refill=5681674962	cpu_migrations=16780	page_faults=1073904	a76_dtlb_walk=2178895749	a76_mem_access=420009850465	a76_inst_retired=1763501752359	a76_l1d_cache_refill=10236056180	a55_inst_share=0.0751	a76_ipc=1.772	l2ref_pki=3.222	l3ref_pki=6.526	l1dref_pki=5.804	memacc_pki=238.17	dtlbw_pki=1.2356	pmu_enabled=100.0	pmu_cpu_s=2452.1	who=at_s=9,rss_pg=1651232,settled=2	pfn_zero_frac=0.0000	maps=5	sampled=116224	present_frac=0.7659	l2color_cv=0.0047	l3color_cv=0.0072	contig_frac=0.7870	mean_run=4.7	vapa16=0.0281	gib_regions=26	anon_pg=46612	anon_l2cv=0.0008	anon_contig=0.7053	anon_run=3.4	anon_gib=25	file_pg=24576	file_l2cv=0.0009	file_contig=0.9947	file_run=138.1	file_gib=24	other_pg=17823	other_l2cv=0.0254	other_contig=0.7142	other_run=3.5	other_gib=26	ro_cost_ms=132-->
<!--DATA 3	ub_unpin	pp2048	27.02-->

### qwen35-9b-ub-pin2x2 [ub_pin] pass 3  env='PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4'  05:58:02  clk=600 MHz  MemAvail=31624524 kB
<!--PRED 3	ub_pin	memavail_kb=31624504	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8641,10566,10705,10217,7592,7100,6333,5324,4351,2545,2500-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         28.53 ± 0.13 |

build: 171974745 (10558)
<!--RO 3	ub_pin	wall_s=291	busy=173,126,128,111,12581,10907,12376,12144	busy_tot=48546	busy_little_share=0.0111	a55_cpu_cycles=19635331649	a55_inst_retired=4255876772	context_switches=7221164	a76_cpu_cycles=1038839093207	a76_l3d_cache_refill=11995596765	a76_l2d_cache_refill=5661275774	cpu_migrations=5795	page_faults=1070760	a76_dtlb_walk=3736393986	a76_mem_access=465091301360	a76_inst_retired=1867969619032	a76_l1d_cache_refill=11373200275	a55_inst_share=0.0023	a76_ipc=1.798	l2ref_pki=3.031	l3ref_pki=6.422	l1dref_pki=6.089	memacc_pki=248.98	dtlbw_pki=2.0002	pmu_enabled=100.0	pmu_cpu_s=2320.1	who=at_s=8,rss_pg=1651207,settled=2	pfn_zero_frac=0.0000	maps=5	sampled=116224	present_frac=0.7592	l2color_cv=0.0040	l3color_cv=0.0094	contig_frac=0.8417	mean_run=6.3	vapa16=0.1879	gib_regions=26	anon_pg=46251	anon_l2cv=0.0042	anon_contig=0.7987	anon_run=4.9	anon_gib=25	file_pg=24576	file_l2cv=0.0007	file_contig=0.9947	file_run=138.1	file_gib=26	other_pg=17409	other_l2cv=0.0131	other_contig=0.7398	other_run=3.8	other_gib=25	ro_cost_ms=116-->
<!--DATA 3	ub_pin	pp2048	28.53-->

### qwen35-9b-ub-pin2x2 [stock_unpin] pass 3  env=''  args=''  06:03:58  clk=600 MHz  MemAvail=31664084 kB
<!--PRED 3	stock_unpin	memavail_kb=31664084	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9985,11199,10828,10516,7714,7077,6310,5299,4340,2538,2514-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         19.09 ± 0.17 |

build: 171974745 (10558)
<!--RO 3	stock_unpin	wall_s=431	busy=6908,5604,5449,5270,24956,22338,21881,22958	busy_tot=115364	busy_little_share=0.2014	a55_cpu_cycles=427757785051	a55_inst_retired=223723514782	context_switches=7970511	a76_cpu_cycles=1990278709171	a76_l3d_cache_refill=15896667769	a76_l2d_cache_refill=8232492568	cpu_migrations=40631	page_faults=594034	a76_dtlb_walk=1832753512	a76_mem_access=656975406550	a76_inst_retired=4131129847122	a76_l1d_cache_refill=13691509589	a55_inst_share=0.0514	a76_ipc=2.076	l2ref_pki=1.993	l3ref_pki=3.848	l1dref_pki=3.314	memacc_pki=159.03	dtlbw_pki=0.4436	pmu_enabled=100.0	pmu_cpu_s=3447.1	who=at_s=6,rss_pg=1509858,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6991	l2color_cv=0.0050	l3color_cv=0.0126	contig_frac=0.8953	mean_run=9.4	vapa16=0.0833	gib_regions=27	anon_pg=23042	anon_l2cv=0.0027	anon_contig=0.9359	anon_run=15.1	anon_gib=24	file_pg=24576	file_l2cv=0.0027	file_contig=0.9833	file_run=53.7	file_gib=26	other_pg=21105	other_l2cv=0.0146	other_contig=0.7484	other_run=3.9	other_gib=26	ro_cost_ms=92-->
<!--DATA 3	stock_unpin	pp2048	19.09-->

### qwen35-9b-ub-pin2x2 [stock_pin] pass 3  env='PIN_MASK=0xf0'  args='-t 4'  06:12:09  clk=600 MHz  MemAvail=31617860 kB
<!--PRED 3	stock_pin	memavail_kb=31617860	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8600,10320,10585,10315,7560,7070,6309,5301,4332,2541,2511-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.54 ± 0.07 |

build: 171974745 (10558)
<!--RO 3	stock_pin	wall_s=402	busy=709,346,224,224,24896,24545,22417,22932	busy_tot=96293	busy_little_share=0.0156	a55_cpu_cycles=47401377466	a55_inst_retired=13270509386	context_switches=7850952	a76_cpu_cycles=2045437401947	a76_l3d_cache_refill=16924386134	a76_l2d_cache_refill=8442908336	cpu_migrations=21612	page_faults=588134	a76_dtlb_walk=2277623287	a76_mem_access=711073205193	a76_inst_retired=4233140937119	a76_l1d_cache_refill=15486350396	a55_inst_share=0.0031	a76_ipc=2.070	l2ref_pki=1.994	l3ref_pki=3.998	l1dref_pki=3.658	memacc_pki=167.98	dtlbw_pki=0.5380	pmu_enabled=100.0	pmu_cpu_s=3205.2	who=at_s=6,rss_pg=1509830,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6958	l2color_cv=0.0032	l3color_cv=0.0131	contig_frac=0.8254	mean_run=5.7	vapa16=0.0509	gib_regions=26	anon_pg=23041	anon_l2cv=0.0014	anon_contig=0.8956	anon_run=9.4	anon_gib=22	file_pg=24576	file_l2cv=0.0009	file_contig=0.9927	file_run=108.3	file_gib=25	other_pg=20787	other_l2cv=0.0100	other_contig=0.5496	other_run=2.2	other_gib=25	ro_cost_ms=109-->
<!--DATA 3	stock_pin	pp2048	20.54-->

### qwen35-9b-ub-pin2x2 [ub_pin] pass 4  env='PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4'  06:19:51  clk=600 MHz  MemAvail=31628084 kB
<!--PRED 4	ub_pin	memavail_kb=31628084	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8400,10365,10540,10440,7509,7080,6295,5303,4328,2539,2516-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         28.52 ± 0.12 |

build: 171974745 (10558)
<!--RO 4	ub_pin	wall_s=291	busy=154,146,116,110,12549,11015,12071,12227	busy_tot=48388	busy_little_share=0.0109	a55_cpu_cycles=19380096459	a55_inst_retired=4246833387	context_switches=7219970	a76_cpu_cycles=1036669917544	a76_l3d_cache_refill=11966533241	a76_l2d_cache_refill=5662544107	cpu_migrations=5815	page_faults=1069019	a76_dtlb_walk=3737139442	a76_mem_access=465120392091	a76_inst_retired=1868061507340	a76_l1d_cache_refill=11364083665	a55_inst_share=0.0023	a76_ipc=1.802	l2ref_pki=3.031	l3ref_pki=6.406	l1dref_pki=6.083	memacc_pki=248.99	dtlbw_pki=2.0005	pmu_enabled=100.0	pmu_cpu_s=2324.0	who=at_s=8,rss_pg=1651208,settled=2	pfn_zero_frac=0.0000	maps=5	sampled=116224	present_frac=0.7629	l2color_cv=0.0028	l3color_cv=0.0076	contig_frac=0.8409	mean_run=6.2	vapa16=0.1082	gib_regions=26	anon_pg=46292	anon_l2cv=0.0017	anon_contig=0.8073	anon_run=5.1	anon_gib=25	file_pg=24576	file_l2cv=0.0012	file_contig=0.9859	file_run=62.4	file_gib=26	other_pg=17804	other_l2cv=0.0142	other_contig=0.7280	other_run=3.7	other_gib=25	ro_cost_ms=136-->
<!--DATA 4	ub_pin	pp2048	28.52-->

### qwen35-9b-ub-pin2x2 [stock_unpin] pass 4  env=''  args=''  06:25:49  clk=600 MHz  MemAvail=31631840 kB
<!--PRED 4	stock_unpin	memavail_kb=31631840	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9123,10819,10682,10190,7688,7065,6279,5290,4315,2543,2516-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         19.09 ± 0.06 |

build: 171974745 (10558)
<!--RO 4	stock_unpin	wall_s=432	busy=7110,5554,5603,5275,24753,22419,22056,22505	busy_tot=115275	busy_little_share=0.2042	a55_cpu_cycles=433690043666	a55_inst_retired=227702121748	context_switches=7972366	a76_cpu_cycles=1983226569699	a76_l3d_cache_refill=15263605269	a76_l2d_cache_refill=8181720772	cpu_migrations=39631	page_faults=589566	a76_dtlb_walk=2195874173	a76_mem_access=656101912305	a76_inst_retired=4126705025328	a76_l1d_cache_refill=13682201845	a55_inst_share=0.0523	a76_ipc=2.081	l2ref_pki=1.983	l3ref_pki=3.699	l1dref_pki=3.316	memacc_pki=158.99	dtlbw_pki=0.5321	pmu_enabled=100.0	pmu_cpu_s=3453.9	who=at_s=6,rss_pg=1509858,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6983	l2color_cv=0.0036	l3color_cv=0.0128	contig_frac=0.8988	mean_run=9.7	vapa16=0.0391	gib_regions=27	anon_pg=23042	anon_l2cv=0.0071	anon_contig=0.9208	anon_run=12.3	anon_gib=25	file_pg=24576	file_l2cv=0.0010	file_contig=0.9947	file_run=138.8	file_gib=25	other_pg=21025	other_l2cv=0.0122	other_contig=0.7625	other_run=4.2	other_gib=26	ro_cost_ms=96-->
<!--DATA 4	stock_unpin	pp2048	19.09-->

### qwen35-9b-ub-pin2x2 [stock_pin] pass 4  env='PIN_MASK=0xf0'  args='-t 4'  06:34:01  clk=600 MHz  MemAvail=31621032 kB
<!--PRED 4	stock_pin	memavail_kb=31621032	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10107,10543,10556,10329,7495,7069,6277,5285,4314,2530,2525-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.71 ± 0.04 |

build: 171974745 (10558)
<!--RO 4	stock_pin	wall_s=399	busy=793,298,220,219,25091,24305,22194,22879	busy_tot=95999	busy_little_share=0.0159	a55_cpu_cycles=47484078700	a55_inst_retired=12888874738	context_switches=7833277	a76_cpu_cycles=2041115853427	a76_l3d_cache_refill=16366468075	a76_l2d_cache_refill=8520897340	cpu_migrations=21051	page_faults=591108	a76_dtlb_walk=2279904863	a76_mem_access=711208838145	a76_inst_retired=4233563629106	a76_l1d_cache_refill=15539191176	a55_inst_share=0.0030	a76_ipc=2.074	l2ref_pki=2.013	l3ref_pki=3.866	l1dref_pki=3.670	memacc_pki=167.99	dtlbw_pki=0.5385	pmu_enabled=100.0	pmu_cpu_s=3180.1	who=at_s=7,rss_pg=1509830,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6998	l2color_cv=0.0036	l3color_cv=0.0113	contig_frac=0.8583	mean_run=7.0	vapa16=0.0292	gib_regions=27	anon_pg=23041	anon_l2cv=0.0017	anon_contig=0.8776	anon_run=8.1	anon_gib=24	file_pg=24576	file_l2cv=0.0016	file_contig=0.9936	file_run=119.3	file_gib=26	other_pg=21177	other_l2cv=0.0118	other_contig=0.6801	other_run=3.1	other_gib=26	ro_cost_ms=130-->
<!--DATA 4	stock_pin	pp2048	20.71-->

### qwen35-9b-ub-pin2x2 [ub_unpin] pass 4  env=''  args='-b 2048 -ub 2048'  06:41:46  clk=600 MHz  MemAvail=31665524 kB
<!--PRED 4	ub_unpin	memavail_kb=31665524	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8195,10948,10769,10599,7565,7025,6271,5272,4306,2534,2536-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         27.05 ± 0.10 |

build: 171974745 (10558)
<!--RO 4	ub_unpin	wall_s=306	busy=4496,4197,3965,3681,12104,11147,11013,11226	busy_tot=61829	busy_little_share=0.2643	a55_cpu_cycles=295056262520	a55_inst_retired=143960213963	context_switches=7261888	a76_cpu_cycles=995962885403	a76_l3d_cache_refill=11479074185	a76_l2d_cache_refill=5674102757	cpu_migrations=16706	page_faults=1074192	a76_dtlb_walk=2176098428	a76_mem_access=420030653886	a76_inst_retired=1762754616185	a76_l1d_cache_refill=10261751975	a55_inst_share=0.0755	a76_ipc=1.770	l2ref_pki=3.219	l3ref_pki=6.512	l1dref_pki=5.821	memacc_pki=238.28	dtlbw_pki=1.2345	pmu_enabled=100.0	pmu_cpu_s=2447.6	who=at_s=8,rss_pg=1651232,settled=2	pfn_zero_frac=0.0000	maps=5	sampled=116224	present_frac=0.7660	l2color_cv=0.0018	l3color_cv=0.0075	contig_frac=0.8438	mean_run=6.3	vapa16=0.0630	gib_regions=27	anon_pg=46789	anon_l2cv=0.0008	anon_contig=0.8111	anon_run=5.3	anon_gib=22	file_pg=24576	file_l2cv=0.0049	file_contig=0.9864	file_run=64.3	file_gib=26	other_pg=17668	other_l2cv=0.0113	other_contig=0.7320	other_run=3.7	other_gib=26	ro_cost_ms=161-->
<!--DATA 4	ub_unpin	pp2048	27.05-->

### qwen35-9b-ub-pin2x2 [stock_unpin] pass 5  env=''  args=''  06:47:58  clk=600 MHz  MemAvail=31636868 kB
<!--PRED 5	stock_unpin	memavail_kb=31636868	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8990,10897,10536,10233,7703,7045,6299,5288,4314,2543,2519-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         18.94 ± 0.07 |

build: 171974745 (10558)
<!--RO 5	stock_unpin	wall_s=435	busy=7040,5567,5496,5247,24952,22245,22251,22732	busy_tot=115530	busy_little_share=0.2021	a55_cpu_cycles=431178136747	a55_inst_retired=225088173244	context_switches=7973895	a76_cpu_cycles=1991818661065	a76_l3d_cache_refill=15556859106	a76_l2d_cache_refill=8221572731	cpu_migrations=39721	page_faults=608673	a76_dtlb_walk=2209483629	a76_mem_access=657538845587	a76_inst_retired=4129734693233	a76_l1d_cache_refill=13799158389	a55_inst_share=0.0517	a76_ipc=2.073	l2ref_pki=1.991	l3ref_pki=3.767	l1dref_pki=3.341	memacc_pki=159.22	dtlbw_pki=0.5350	pmu_enabled=100.0	pmu_cpu_s=3474.5	who=at_s=6,rss_pg=1509858,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6952	l2color_cv=0.0049	l3color_cv=0.0121	contig_frac=0.8713	mean_run=7.7	vapa16=0.0233	gib_regions=26	anon_pg=23024	anon_l2cv=0.0018	anon_contig=0.8773	anon_run=8.0	anon_gib=25	file_pg=24576	file_l2cv=0.0012	file_contig=0.9852	file_run=59.7	file_gib=25	other_pg=20738	other_l2cv=0.0158	other_contig=0.7295	other_run=3.7	other_gib=26	ro_cost_ms=67-->
<!--DATA 5	stock_unpin	pp2048	18.94-->

### qwen35-9b-ub-pin2x2 [stock_pin] pass 5  env='PIN_MASK=0xf0'  args='-t 4'  06:56:13  clk=600 MHz  MemAvail=31612944 kB
<!--PRED 5	stock_pin	memavail_kb=31612944	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7575,10738,10760,10186,7332,7040,6267,5276,4315,2534,2526-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.94 ± 0.08 |

build: 171974745 (10558)
<!--RO 5	stock_pin	wall_s=394	busy=682,312,236,247,24900,24116,22037,23008	busy_tot=95538	busy_little_share=0.0155	a55_cpu_cycles=46767606257	a55_inst_retired=12471819074	context_switches=7851453	a76_cpu_cycles=2030402993723	a76_l3d_cache_refill=15986663446	a76_l2d_cache_refill=8502862840	cpu_migrations=21176	page_faults=587059	a76_dtlb_walk=2273485878	a76_mem_access=711322136205	a76_inst_retired=4233537813158	a76_l1d_cache_refill=15550089060	a55_inst_share=0.0029	a76_ipc=2.085	l2ref_pki=2.008	l3ref_pki=3.776	l1dref_pki=3.673	memacc_pki=168.02	dtlbw_pki=0.5370	pmu_enabled=100.0	pmu_cpu_s=3147.8	who=at_s=6,rss_pg=1509832,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6954	l2color_cv=0.0032	l3color_cv=0.0087	contig_frac=0.8505	mean_run=6.6	vapa16=0.0112	gib_regions=26	anon_pg=23041	anon_l2cv=0.0031	anon_contig=0.8798	anon_run=8.2	anon_gib=25	file_pg=24576	file_l2cv=0.0004	file_contig=0.9949	file_run=141.2	file_gib=25	other_pg=20739	other_l2cv=0.0118	other_contig=0.6470	other_run=2.8	other_gib=26	ro_cost_ms=139-->
<!--DATA 5	stock_pin	pp2048	20.94-->

### qwen35-9b-ub-pin2x2 [ub_unpin] pass 5  env=''  args='-b 2048 -ub 2048'  07:03:52  clk=600 MHz  MemAvail=31623868 kB
<!--PRED 5	ub_unpin	memavail_kb=31623868	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8197,10762,10463,10299,7613,7054,6269,5280,4307,2532,2528-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         26.95 ± 0.15 |

build: 171974745 (10558)
<!--RO 5	ub_unpin	wall_s=307	busy=4461,4167,4027,3722,12332,10776,10889,11465	busy_tot=61839	busy_little_share=0.2648	a55_cpu_cycles=296219642488	a55_inst_retired=144743300498	context_switches=7265923	a76_cpu_cycles=995195224376	a76_l3d_cache_refill=11485097918	a76_l2d_cache_refill=5664958091	cpu_migrations=16938	page_faults=1074032	a76_dtlb_walk=2184385120	a76_mem_access=419553338698	a76_inst_retired=1761913443328	a76_l1d_cache_refill=10198184652	a55_inst_share=0.0759	a76_ipc=1.770	l2ref_pki=3.215	l3ref_pki=6.519	l1dref_pki=5.788	memacc_pki=238.12	dtlbw_pki=1.2398	pmu_enabled=100.0	pmu_cpu_s=2455.5	who=at_s=8,rss_pg=1651232,settled=2	pfn_zero_frac=0.0000	maps=5	sampled=116224	present_frac=0.7652	l2color_cv=0.0023	l3color_cv=0.0070	contig_frac=0.7926	mean_run=4.8	vapa16=0.0258	gib_regions=26	anon_pg=46614	anon_l2cv=0.0037	anon_contig=0.7101	anon_run=3.4	anon_gib=25	file_pg=24576	file_l2cv=0.0013	file_contig=0.9934	file_run=116.5	file_gib=25	other_pg=17749	other_l2cv=0.0111	other_contig=0.7313	other_run=3.7	other_gib=26	ro_cost_ms=141-->
<!--DATA 5	ub_unpin	pp2048	26.95-->

### qwen35-9b-ub-pin2x2 [ub_pin] pass 5  env='PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4'  07:10:00  clk=600 MHz  MemAvail=31617500 kB
<!--PRED 5	ub_pin	memavail_kb=31617500	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8559,10265,10914,10235,7490,7005,6259,5265,4296,2533,2534-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         28.42 ± 0.11 |

build: 171974745 (10558)
<!--RO 5	ub_pin	wall_s=292	busy=175,122,129,110,12740,11080,12150,12139	busy_tot=48645	busy_little_share=0.0110	a55_cpu_cycles=20096720537	a55_inst_retired=4339767601	context_switches=7224102	a76_cpu_cycles=1041932697138	a76_l3d_cache_refill=11971951063	a76_l2d_cache_refill=5677636076	cpu_migrations=5803	page_faults=1070829	a76_dtlb_walk=3723758459	a76_mem_access=465225019657	a76_inst_retired=1868381490532	a76_l1d_cache_refill=11391619103	a55_inst_share=0.0023	a76_ipc=1.793	l2ref_pki=3.039	l3ref_pki=6.408	l1dref_pki=6.097	memacc_pki=249.00	dtlbw_pki=1.9930	pmu_enabled=100.0	pmu_cpu_s=2331.8	who=at_s=8,rss_pg=1651207,settled=2	pfn_zero_frac=0.0000	maps=5	sampled=116224	present_frac=0.7621	l2color_cv=0.0021	l3color_cv=0.0067	contig_frac=0.8275	mean_run=5.7	vapa16=0.1239	gib_regions=27	anon_pg=46251	anon_l2cv=0.0009	anon_contig=0.7760	anon_run=4.4	anon_gib=25	file_pg=24576	file_l2cv=0.0009	file_contig=0.9934	file_run=116.5	file_gib=25	other_pg=17746	other_l2cv=0.0102	other_contig=0.7319	other_run=3.7	other_gib=27	ro_cost_ms=141-->
<!--DATA 5	ub_pin	pp2048	28.42-->

### qwen35-9b-ub-pin2x2 [stock_pin] pass 6  env='PIN_MASK=0xf0'  args='-t 4'  07:15:52  clk=600 MHz  MemAvail=31620356 kB
<!--PRED 6	stock_pin	memavail_kb=31620356	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7910,11005,10560,10444,7386,7005,6264,5271,4304,2531,2531-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.38 ± 0.09 |

build: 171974745 (10558)
<!--RO 6	stock_pin	wall_s=406	busy=952,306,257,209,25171,24463,22693,22756	busy_tot=96807	busy_little_share=0.0178	a55_cpu_cycles=50949432859	a55_inst_retired=14757768590	context_switches=7848019	a76_cpu_cycles=2052639817674	a76_l3d_cache_refill=17175142446	a76_l2d_cache_refill=8410316383	cpu_migrations=21016	page_faults=587965	a76_dtlb_walk=2272440902	a76_mem_access=711019384888	a76_inst_retired=4233313610727	a76_l1d_cache_refill=15431927300	a55_inst_share=0.0035	a76_ipc=2.062	l2ref_pki=1.987	l3ref_pki=4.057	l1dref_pki=3.645	memacc_pki=167.96	dtlbw_pki=0.5368	pmu_enabled=100.0	pmu_cpu_s=3236.8	who=at_s=6,rss_pg=1509832,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6954	l2color_cv=0.0042	l3color_cv=0.0100	contig_frac=0.8963	mean_run=9.5	vapa16=0.0075	gib_regions=27	anon_pg=23042	anon_l2cv=0.0041	anon_contig=0.9566	anon_run=22.0	anon_gib=25	file_pg=24576	file_l2cv=0.0021	file_contig=0.9892	file_run=78.5	file_gib=25	other_pg=20746	other_l2cv=0.0108	other_contig=0.7192	other_run=3.5	other_gib=27	ro_cost_ms=115-->
<!--DATA 6	stock_pin	pp2048	20.38-->

### qwen35-9b-ub-pin2x2 [ub_unpin] pass 6  env=''  args='-b 2048 -ub 2048'  07:23:42  clk=600 MHz  MemAvail=31668704 kB
<!--PRED 6	ub_unpin	memavail_kb=31668996	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7992,10800,10856,10650,7706,7035,6243,5256,4300,2526,2543-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         27.27 ± 0.05 |

build: 171974745 (10558)
<!--RO 6	ub_unpin	wall_s=304	busy=4490,4112,3935,3663,12445,10342,11186,11339	busy_tot=61512	busy_little_share=0.2634	a55_cpu_cycles=293469211256	a55_inst_retired=143107262737	context_switches=7255265	a76_cpu_cycles=993021691347	a76_l3d_cache_refill=11461035854	a76_l2d_cache_refill=5658837735	cpu_migrations=17123	page_faults=1074214	a76_dtlb_walk=2151930371	a76_mem_access=420078770160	a76_inst_retired=1763045865536	a76_l1d_cache_refill=10188418659	a55_inst_share=0.0751	a76_ipc=1.775	l2ref_pki=3.210	l3ref_pki=6.501	l1dref_pki=5.779	memacc_pki=238.27	dtlbw_pki=1.2206	pmu_enabled=100.0	pmu_cpu_s=2425.4	who=at_s=8,rss_pg=1651232,settled=2	pfn_zero_frac=0.0000	maps=5	sampled=116224	present_frac=0.7620	l2color_cv=0.0028	l3color_cv=0.0090	contig_frac=0.8059	mean_run=5.1	vapa16=0.1034	gib_regions=27	anon_pg=46134	anon_l2cv=0.0004	anon_contig=0.7542	anon_run=4.0	anon_gib=24	file_pg=24576	file_l2cv=0.0031	file_contig=0.9843	file_run=56.9	file_gib=25	other_pg=17847	other_l2cv=0.0131	other_contig=0.6938	other_run=3.3	other_gib=26	ro_cost_ms=132-->
<!--DATA 6	ub_unpin	pp2048	27.27-->

### qwen35-9b-ub-pin2x2 [ub_pin] pass 6  env='PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4'  07:29:46  clk=600 MHz  MemAvail=31622092 kB
<!--PRED 6	ub_pin	memavail_kb=31622092	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7828,10521,10703,10427,7461,6993,6263,5271,4306,2536,2530-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         28.60 ± 0.25 |

build: 171974745 (10558)
<!--RO 6	ub_pin	wall_s=289	busy=167,137,121,103,12685,11083,12270,12030	busy_tot=48596	busy_little_share=0.0109	a55_cpu_cycles=19648607830	a55_inst_retired=4243209904	context_switches=7212396	a76_cpu_cycles=1041452330877	a76_l3d_cache_refill=12050267648	a76_l2d_cache_refill=5657451844	cpu_migrations=5657	page_faults=1069741	a76_dtlb_walk=3736932052	a76_mem_access=465237299446	a76_inst_retired=1868421280370	a76_l1d_cache_refill=11407772871	a55_inst_share=0.0023	a76_ipc=1.794	l2ref_pki=3.028	l3ref_pki=6.449	l1dref_pki=6.106	memacc_pki=249.00	dtlbw_pki=2.0000	pmu_enabled=100.0	pmu_cpu_s=2313.2	who=at_s=8,rss_pg=1651206,settled=2	pfn_zero_frac=0.0000	maps=5	sampled=116224	present_frac=0.7683	l2color_cv=0.0050	l3color_cv=0.0100	contig_frac=0.8424	mean_run=6.3	vapa16=0.0239	gib_regions=27	anon_pg=46988	anon_l2cv=0.0005	anon_contig=0.8209	anon_run=5.5	anon_gib=24	file_pg=24576	file_l2cv=0.0013	file_contig=0.9938	file_run=122.3	file_gib=25	other_pg=17736	other_l2cv=0.0254	other_contig=0.6893	other_run=3.2	other_gib=26	ro_cost_ms=114-->
<!--DATA 6	ub_pin	pp2048	28.60-->

### qwen35-9b-ub-pin2x2 [stock_unpin] pass 6  env=''  args=''  07:35:42  clk=600 MHz  MemAvail=31666564 kB
<!--PRED 6	stock_unpin	memavail_kb=31666564	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9250,10734,11061,10557,7490,6966,6243,5251,4300,2531,2543-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         18.98 ± 0.09 |

build: 171974745 (10558)
<!--RO 6	stock_unpin	wall_s=434	busy=7076,5582,5479,5216,25103,22121,21958,22981	busy_tot=115516	busy_little_share=0.2022	a55_cpu_cycles=428999238380	a55_inst_retired=223997459643	context_switches=7964779	a76_cpu_cycles=1991545981066	a76_l3d_cache_refill=15979664636	a76_l2d_cache_refill=8245378553	cpu_migrations=41015	page_faults=589733	a76_dtlb_walk=2202377865	a76_mem_access=657284235598	a76_inst_retired=4130720697327	a76_l1d_cache_refill=13719806552	a55_inst_share=0.0514	a76_ipc=2.074	l2ref_pki=1.996	l3ref_pki=3.868	l1dref_pki=3.321	memacc_pki=159.12	dtlbw_pki=0.5332	pmu_enabled=100.0	pmu_cpu_s=3466.7	who=at_s=6,rss_pg=1509858,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6953	l2color_cv=0.0038	l3color_cv=0.0116	contig_frac=0.8886	mean_run=8.8	vapa16=0.0062	gib_regions=26	anon_pg=23042	anon_l2cv=0.0015	anon_contig=0.9433	anon_run=17.1	anon_gib=25	file_pg=24576	file_l2cv=0.0009	file_contig=0.9843	file_run=56.8	file_gib=25	other_pg=20730	other_l2cv=0.0127	other_contig=0.7143	other_run=3.5	other_gib=26	ro_cost_ms=66-->
<!--DATA 6	stock_unpin	pp2048	18.98-->

#### summary: per-arm mean over 6 passes, ratios paired within a pass against [stock_unpin]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock_unpin | 19.09 | 6 | 18.94 | 19.30 | -- | -- |
| pp2048 | stock_pin | 20.64 | 6 | 20.38 | 20.94 | 1.082x | 1.075 1.075 1.076 1.085 1.106 1.074 |
| pp2048 | ub_unpin | 26.99 | 6 | 26.78 | 27.27 | 1.414x | 1.405 1.388 1.415 1.417 1.423 1.437 |
| pp2048 | ub_pin | 28.53 | 6 | 28.42 | 28.62 | 1.495x | 1.497 1.477 1.494 1.494 1.501 1.507 |

