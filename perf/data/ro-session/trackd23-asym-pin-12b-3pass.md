<!-- gemma4-12b F16, pinning x ROCKET_MM_ASYM (a DEVICE-tiling knob) 2x2
     TESTS='-p 2048 -n 0 -r 3'  PASSES=3
     gguf=<data>/gemma4/gemma-4-12b-it-F16.gguf (23832065184 bytes)
     asym0_unpin = ROCKET_MM_ASYM=0, the knob DECLINED; it is the base of every ratio
     stock_*     = the shipping default, ROCKET_MM_ASYM absent
     *_pin       = PIN_MASK=0xf0 -t 4, matching all three earlier interaction cells
     interaction = (stock_pin/asym0_pin) / (stock_unpin/asym0_unpin), paired within a pass
     no arm places resident weights; an arm that reports any is not this contrast
-->
== gemma4-12b-asym-pin2x2  Wed Sep  2 18:29:35 UTC 2026 ==
### gemma4-12b-asym-pin2x2 [asym0_unpin] pass 1  env='ROCKET_MM_ASYM=0'  args=''  18:31:10  clk=600 MHz  MemAvail=31444636 kB
<!--PRED 1	asym0_unpin	memavail_kb=31444924	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5625,3308,7076,6533,5677,5135,4347,3821,2525,91,266-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         18.11 ± 0.04 |

build: 171974745 (10558)
<!--RO 1	asym0_unpin	wall_s=457	busy=5563,5538,5556,5458,19093,13196,13409,14754	busy_tot=82567	busy_little_share=0.2678	a55_cpu_cycles=423971480550	a55_inst_retired=266240273343	context_switches=9923733	a76_cpu_cycles=1328135071004	a76_l3d_cache_refill=20063324772	a76_l2d_cache_refill=10787042090	cpu_migrations=40521	page_faults=1002591	a76_dtlb_walk=1132567787	a76_mem_access=339572701099	a76_inst_retired=1592334613989	a76_l1d_cache_refill=23101802784	a55_inst_share=0.1432	a76_ipc=1.199	l2ref_pki=6.774	l3ref_pki=12.600	l1dref_pki=14.508	memacc_pki=213.25	dtlbw_pki=0.7113	pmu_enabled=100.0	pmu_cpu_s=3649.3	who=at_s=8,rss_pg=6044641,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8127	l2color_cv=0.0115	l3color_cv=0.0173	contig_frac=0.9401	mean_run=16.2	vapa16=0.1786	gib_regions=32	anon_pg=12303	anon_l2cv=0.0026	anon_contig=0.9836	anon_run=54.2	anon_gib=15	file_pg=24576	file_l2cv=0.0041	file_contig=0.9937	file_run=121.1	file_gib=28	other_pg=23040	other_l2cv=0.0279	other_contig=0.8597	other_run=7.0	other_gib=32	ro_cost_ms=79-->
<!--DATA 1	asym0_unpin	pp2048	18.11-->

### gemma4-12b-asym-pin2x2 [asym0_pin] pass 1  env='ROCKET_MM_ASYM=0 PIN_MASK=0xf0'  args='-t 4'  18:40:14  clk=600 MHz  MemAvail=31501128 kB
<!--PRED 1	asym0_pin	memavail_kb=31501128	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3627,6670,10214,7115,5831,5263,4162,3809,2523,116,253-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         19.55 ± 0.09 |

build: 171974745 (10558)
<!--RO 1	asym0_pin	wall_s=424	busy=369,368,388,374,19978,12961,13159,16699	busy_tot=64296	busy_little_share=0.0233	a55_cpu_cycles=57504795780	a55_inst_retired=12233762376	context_switches=9741572	a76_cpu_cycles=1374016876381	a76_l3d_cache_refill=20941847269	a76_l2d_cache_refill=10900340847	cpu_migrations=25719	page_faults=993137	a76_dtlb_walk=1142303751	a76_mem_access=364437847660	a76_inst_retired=1709665197587	a76_l1d_cache_refill=24663318576	a55_inst_share=0.0071	a76_ipc=1.244	l2ref_pki=6.376	l3ref_pki=12.249	l1dref_pki=14.426	memacc_pki=213.16	dtlbw_pki=0.6681	pmu_enabled=100.0	pmu_cpu_s=3386.4	who=at_s=8,rss_pg=6044319,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8246	l2color_cv=0.0085	l3color_cv=0.0209	contig_frac=0.9035	mean_run=10.2	vapa16=0.0155	gib_regions=32	anon_pg=12925	anon_l2cv=0.0005	anon_contig=0.9981	anon_run=253.4	anon_gib=4	file_pg=24576	file_l2cv=0.0031	file_contig=0.9938	file_run=122.3	file_gib=30	other_pg=23298	other_l2cv=0.0237	other_contig=0.7557	other_run=4.1	other_gib=32	ro_cost_ms=92-->
<!--DATA 1	asym0_pin	pp2048	19.55-->

### gemma4-12b-asym-pin2x2 [stock_unpin] pass 1  env=''  args=''  18:48:49  clk=600 MHz  MemAvail=31400624 kB
<!--PRED 1	stock_unpin	memavail_kb=31400488	anonhuge_kb=0	hugepagesz_kb=2048	buddy=2352,3651,7849,7227,6063,5220,3943,3810,2525,115,255-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         20.04 ± 0.05 |

build: 171974745 (10558)
<!--RO 1	stock_unpin	wall_s=414	busy=5487,5471,5410,5447,18314,12175,12559,16036	busy_tot=80899	busy_little_share=0.2697	a55_cpu_cycles=418596368204	a55_inst_retired=261430286773	context_switches=13124895	a76_cpu_cycles=1310621669191	a76_l3d_cache_refill=19125948308	a76_l2d_cache_refill=9220797354	cpu_migrations=40551	page_faults=953390	a76_dtlb_walk=1066532791	a76_mem_access=344748640627	a76_inst_retired=1596079585874	a76_l1d_cache_refill=24336134310	a55_inst_share=0.1407	a76_ipc=1.218	l2ref_pki=5.777	l3ref_pki=11.983	l1dref_pki=15.247	memacc_pki=216.00	dtlbw_pki=0.6682	pmu_enabled=100.0	pmu_cpu_s=3306.6	who=at_s=8,rss_pg=6045724,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8173	l2color_cv=0.0121	l3color_cv=0.0165	contig_frac=0.9086	mean_run=10.7	vapa16=0.0940	gib_regions=32	anon_pg=12303	anon_l2cv=0.0019	anon_contig=0.9793	anon_run=43.9	anon_gib=25	file_pg=24576	file_l2cv=0.0025	file_contig=0.9960	file_run=167.2	file_gib=30	other_pg=23382	other_l2cv=0.0336	other_contig=0.7796	other_run=4.5	other_gib=32	ro_cost_ms=62-->
<!--DATA 1	stock_unpin	pp2048	20.04-->

### gemma4-12b-asym-pin2x2 [stock_pin] pass 1  env='PIN_MASK=0xf0'  args='-t 4'  18:57:07  clk=600 MHz  MemAvail=31490420 kB
<!--PRED 1	stock_pin	memavail_kb=31490420	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3872,5782,10372,8382,6221,5251,3826,3811,2520,156,236-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         20.86 ± 0.10 |

build: 171974745 (10558)
<!--RO 1	stock_pin	wall_s=398	busy=307,341,350,327,18946,12601,13144,16563	busy_tot=62579	busy_little_share=0.0212	a55_cpu_cycles=54429865645	a55_inst_retired=10875165225	context_switches=12962337	a76_cpu_cycles=1352063174988	a76_l3d_cache_refill=19995180122	a76_l2d_cache_refill=9460338263	cpu_migrations=26422	page_faults=956227	a76_dtlb_walk=1099397031	a76_mem_access=369476283171	a76_inst_retired=1712314914215	a76_l1d_cache_refill=22417196190	a55_inst_share=0.0063	a76_ipc=1.266	l2ref_pki=5.525	l3ref_pki=11.677	l1dref_pki=13.092	memacc_pki=215.78	dtlbw_pki=0.6421	pmu_enabled=100.0	pmu_cpu_s=3176.3	who=at_s=8,rss_pg=6044869,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8338	l2color_cv=0.0102	l3color_cv=0.0146	contig_frac=0.9289	mean_run=13.7	vapa16=0.0467	gib_regions=32	anon_pg=12932	anon_l2cv=0.0009	anon_contig=0.9982	anon_run=263.9	anon_gib=4	file_pg=24576	file_l2cv=0.0026	file_contig=0.9869	file_run=66.4	file_gib=30	other_pg=23970	other_l2cv=0.0249	other_contig=0.8321	other_run=5.9	other_gib=32	ro_cost_ms=83-->
<!--DATA 1	stock_pin	pp2048	20.86-->

### gemma4-12b-asym-pin2x2 [asym0_pin] pass 2  env='ROCKET_MM_ASYM=0 PIN_MASK=0xf0'  args='-t 4'  19:05:14  clk=600 MHz  MemAvail=31486480 kB
<!--PRED 2	asym0_pin	memavail_kb=31486480	anonhuge_kb=0	hugepagesz_kb=2048	buddy=2883,6264,9645,7341,5941,5314,4084,3815,2530,155,230-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         19.43 ± 0.03 |

build: 171974745 (10558)
<!--RO 2	asym0_pin	wall_s=427	busy=384,301,364,443,19495,12881,13020,17429	busy_tot=64317	busy_little_share=0.0232	a55_cpu_cycles=56913833055	a55_inst_retired=12061411964	context_switches=9747687	a76_cpu_cycles=1375757145307	a76_l3d_cache_refill=20988808164	a76_l2d_cache_refill=10931872779	cpu_migrations=25145	page_faults=986425	a76_dtlb_walk=1166888569	a76_mem_access=364669927443	a76_inst_retired=1710659189562	a76_l1d_cache_refill=24684504055	a55_inst_share=0.0070	a76_ipc=1.243	l2ref_pki=6.390	l3ref_pki=12.269	l1dref_pki=14.430	memacc_pki=213.18	dtlbw_pki=0.6821	pmu_enabled=100.0	pmu_cpu_s=3410.2	who=at_s=8,rss_pg=6044325,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8234	l2color_cv=0.0045	l3color_cv=0.0106	contig_frac=0.9295	mean_run=13.8	vapa16=0.0311	gib_regions=32	anon_pg=13054	anon_l2cv=0.0134	anon_contig=0.9843	anon_run=56.5	anon_gib=31	file_pg=24576	file_l2cv=0.0011	file_contig=0.9960	file_run=168.3	file_gib=28	other_pg=23077	other_l2cv=0.0092	other_contig=0.8278	other_run=5.7	other_gib=32	ro_cost_ms=97-->
<!--DATA 2	asym0_pin	pp2048	19.43-->

### gemma4-12b-asym-pin2x2 [stock_unpin] pass 2  env=''  args=''  19:13:51  clk=600 MHz  MemAvail=31427940 kB
<!--PRED 2	stock_unpin	memavail_kb=31427940	anonhuge_kb=0	hugepagesz_kb=2048	buddy=2679,4284,7810,6974,6010,5221,4048,3792,2514,144,247-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         19.81 ± 0.00 |

build: 171974745 (10558)
<!--RO 2	stock_unpin	wall_s=419	busy=5477,5488,5475,5444,18532,12269,12680,15679	busy_tot=81044	busy_little_share=0.2700	a55_cpu_cycles=418862162152	a55_inst_retired=261443915846	context_switches=13125393	a76_cpu_cycles=1314219099387	a76_l3d_cache_refill=19106865028	a76_l2d_cache_refill=9254613266	cpu_migrations=40319	page_faults=955365	a76_dtlb_walk=1068180816	a76_mem_access=344650172765	a76_inst_retired=1595844861875	a76_l1d_cache_refill=24322091961	a55_inst_share=0.1408	a76_ipc=1.214	l2ref_pki=5.799	l3ref_pki=11.973	l1dref_pki=15.241	memacc_pki=215.97	dtlbw_pki=0.6694	pmu_enabled=100.0	pmu_cpu_s=3346.3	who=at_s=8,rss_pg=6045724,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8183	l2color_cv=0.0076	l3color_cv=0.0132	contig_frac=0.8785	mean_run=8.1	vapa16=0.0369	gib_regions=32	anon_pg=12303	anon_l2cv=0.0135	anon_contig=0.8931	anon_run=9.2	anon_gib=32	file_pg=24576	file_l2cv=0.0008	file_contig=0.9941	file_run=127.3	file_gib=31	other_pg=23456	other_l2cv=0.0178	other_contig=0.7497	other_run=4.0	other_gib=32	ro_cost_ms=86-->
<!--DATA 2	stock_unpin	pp2048	19.81-->

### gemma4-12b-asym-pin2x2 [stock_pin] pass 2  env='PIN_MASK=0xf0'  args='-t 4'  19:22:14  clk=600 MHz  MemAvail=31493684 kB
<!--PRED 2	stock_pin	memavail_kb=31494068	anonhuge_kb=0	hugepagesz_kb=2048	buddy=2681,6205,9095,6765,5893,5247,4158,3784,2512,167,240-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         21.12 ± 0.07 |

build: 171974745 (10558)
<!--RO 2	stock_pin	wall_s=394	busy=392,291,344,310,18973,12565,13089,16462	busy_tot=62426	busy_little_share=0.0214	a55_cpu_cycles=54924779070	a55_inst_retired=10952731983	context_switches=12952875	a76_cpu_cycles=1350937138066	a76_l3d_cache_refill=19928532560	a76_l2d_cache_refill=9437611041	cpu_migrations=25665	page_faults=957347	a76_dtlb_walk=1090434206	a76_mem_access=369250624049	a76_inst_retired=1711642772946	a76_l1d_cache_refill=22352282047	a55_inst_share=0.0064	a76_ipc=1.267	l2ref_pki=5.514	l3ref_pki=11.643	l1dref_pki=13.059	memacc_pki=215.73	dtlbw_pki=0.6371	pmu_enabled=100.0	pmu_cpu_s=3140.4	who=at_s=8,rss_pg=6044868,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8265	l2color_cv=0.0088	l3color_cv=0.0175	contig_frac=0.9092	mean_run=10.8	vapa16=0.0085	gib_regions=32	anon_pg=12932	anon_l2cv=0.0146	anon_contig=0.9193	anon_run=12.1	anon_gib=30	file_pg=24576	file_l2cv=0.0039	file_contig=0.9868	file_run=66.1	file_gib=31	other_pg=23428	other_l2cv=0.0183	other_contig=0.8222	other_run=5.6	other_gib=32	ro_cost_ms=85-->
<!--DATA 2	stock_pin	pp2048	21.12-->

### gemma4-12b-asym-pin2x2 [asym0_unpin] pass 2  env='ROCKET_MM_ASYM=0'  args=''  19:30:20  clk=600 MHz  MemAvail=31524200 kB
<!--PRED 2	asym0_unpin	memavail_kb=31524200	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7336,5696,9333,7721,5881,5314,4040,3781,2502,159,247-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         18.02 ± 0.06 |

build: 171974745 (10558)
<!--RO 2	asym0_unpin	wall_s=460	busy=5499,5504,5508,5447,19240,12405,13073,16225	busy_tot=82901	busy_little_share=0.2649	a55_cpu_cycles=421237609496	a55_inst_retired=264155328324	context_switches=9917704	a76_cpu_cycles=1337330323571	a76_l3d_cache_refill=20120474683	a76_l2d_cache_refill=10740635361	cpu_migrations=39498	page_faults=985003	a76_dtlb_walk=1146864368	a76_mem_access=339730136379	a76_inst_retired=1593145560083	a76_l1d_cache_refill=23162597559	a55_inst_share=0.1422	a76_ipc=1.191	l2ref_pki=6.742	l3ref_pki=12.629	l1dref_pki=14.539	memacc_pki=213.24	dtlbw_pki=0.7199	pmu_enabled=100.0	pmu_cpu_s=3674.2	who=at_s=8,rss_pg=6044641,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8127	l2color_cv=0.0054	l3color_cv=0.0164	contig_frac=0.9073	mean_run=10.6	vapa16=0.1821	gib_regions=32	anon_pg=12303	anon_l2cv=0.0027	anon_contig=0.9734	anon_run=34.9	anon_gib=29	file_pg=24576	file_l2cv=0.0007	file_contig=0.9961	file_run=170.7	file_gib=30	other_pg=23040	other_l2cv=0.0142	other_contig=0.7772	other_run=4.5	other_gib=32	ro_cost_ms=86-->
<!--DATA 2	asym0_unpin	pp2048	18.02-->

### gemma4-12b-asym-pin2x2 [stock_unpin] pass 3  env=''  args=''  19:39:30  clk=600 MHz  MemAvail=31402884 kB
<!--PRED 3	stock_unpin	memavail_kb=31402884	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6935,4054,8226,7015,5993,5229,3849,3785,2510,155,243-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         20.00 ± 0.03 |

build: 171974745 (10558)
<!--RO 3	stock_unpin	wall_s=415	busy=5415,5523,5497,5448,18645,12158,12498,15741	busy_tot=80925	busy_little_share=0.2704	a55_cpu_cycles=417790713022	a55_inst_retired=261014570761	context_switches=13121745	a76_cpu_cycles=1311037794174	a76_l3d_cache_refill=19123714115	a76_l2d_cache_refill=9260358840	cpu_migrations=40080	page_faults=950126	a76_dtlb_walk=1064578056	a76_mem_access=345082332170	a76_inst_retired=1597690307747	a76_l1d_cache_refill=24310213208	a55_inst_share=0.1404	a76_ipc=1.219	l2ref_pki=5.796	l3ref_pki=11.970	l1dref_pki=15.216	memacc_pki=215.99	dtlbw_pki=0.6663	pmu_enabled=100.0	pmu_cpu_s=3314.8	who=at_s=8,rss_pg=6045725,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8168	l2color_cv=0.0062	l3color_cv=0.0122	contig_frac=0.9301	mean_run=13.9	vapa16=0.0087	gib_regions=32	anon_pg=12299	anon_l2cv=0.0094	anon_contig=0.9641	anon_run=26.3	anon_gib=32	file_pg=24576	file_l2cv=0.0039	file_contig=0.9943	file_run=130.0	file_gib=30	other_pg=23348	other_l2cv=0.0162	other_contig=0.8448	other_run=6.4	other_gib=32	ro_cost_ms=61-->
<!--DATA 3	stock_unpin	pp2048	20.00-->

### gemma4-12b-asym-pin2x2 [stock_pin] pass 3  env='PIN_MASK=0xf0'  args='-t 4'  19:47:49  clk=600 MHz  MemAvail=31490428 kB
<!--PRED 3	stock_pin	memavail_kb=31490428	anonhuge_kb=0	hugepagesz_kb=2048	buddy=2555,6487,9897,7786,6066,5322,3830,3790,2506,165,244-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         21.12 ± 0.08 |

build: 171974745 (10558)
<!--RO 3	stock_pin	wall_s=393	busy=323,379,379,293,18814,12603,12592,17242	busy_tot=62625	busy_little_share=0.0219	a55_cpu_cycles=54318553242	a55_inst_retired=10942639482	context_switches=12968623	a76_cpu_cycles=1351289124952	a76_l3d_cache_refill=19973496361	a76_l2d_cache_refill=9434116045	cpu_migrations=25575	page_faults=958641	a76_dtlb_walk=1106948913	a76_mem_access=369501452865	a76_inst_retired=1712828196941	a76_l1d_cache_refill=22319944496	a55_inst_share=0.0063	a76_ipc=1.268	l2ref_pki=5.508	l3ref_pki=11.661	l1dref_pki=13.031	memacc_pki=215.73	dtlbw_pki=0.6463	pmu_enabled=100.0	pmu_cpu_s=3140.0	who=at_s=8,rss_pg=6044867,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8309	l2color_cv=0.0069	l3color_cv=0.0149	contig_frac=0.9300	mean_run=13.9	vapa16=0.0724	gib_regions=32	anon_pg=12932	anon_l2cv=0.0060	anon_contig=0.9843	anon_run=56.7	anon_gib=31	file_pg=24576	file_l2cv=0.0039	file_contig=0.9905	file_run=87.5	file_gib=30	other_pg=23751	other_l2cv=0.0176	other_contig=0.8378	other_run=6.1	other_gib=32	ro_cost_ms=96-->
<!--DATA 3	stock_pin	pp2048	21.12-->

### gemma4-12b-asym-pin2x2 [asym0_unpin] pass 3  env='ROCKET_MM_ASYM=0'  args=''  19:55:57  clk=600 MHz  MemAvail=31454032 kB
<!--PRED 3	asym0_unpin	memavail_kb=31454344	anonhuge_kb=0	hugepagesz_kb=2048	buddy=2314,5020,9055,7058,6125,5370,3846,3787,2505,161,247-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         18.40 ± 0.04 |

build: 171974745 (10558)
<!--RO 3	asym0_unpin	wall_s=451	busy=5516,5502,5550,5503,19156,12516,13159,15956	busy_tot=82858	busy_little_share=0.2664	a55_cpu_cycles=422438366814	a55_inst_retired=264435138764	context_switches=9908894	a76_cpu_cycles=1332472880161	a76_l3d_cache_refill=20110838985	a76_l2d_cache_refill=10742842317	cpu_migrations=40121	page_faults=985971	a76_dtlb_walk=1136733244	a76_mem_access=339825572941	a76_inst_retired=1593977964609	a76_l1d_cache_refill=23111875975	a55_inst_share=0.1423	a76_ipc=1.196	l2ref_pki=6.740	l3ref_pki=12.617	l1dref_pki=14.499	memacc_pki=213.19	dtlbw_pki=0.7131	pmu_enabled=100.0	pmu_cpu_s=3600.7	who=at_s=8,rss_pg=6044642,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8223	l2color_cv=0.0050	l3color_cv=0.0108	contig_frac=0.9219	mean_run=12.5	vapa16=0.1750	gib_regions=32	anon_pg=12303	anon_l2cv=0.0071	anon_contig=0.9301	anon_run=13.9	anon_gib=32	file_pg=24576	file_l2cv=0.0019	file_contig=0.9861	file_run=63.2	file_gib=31	other_pg=23746	other_l2cv=0.0119	other_contig=0.8511	other_run=6.6	other_gib=32	ro_cost_ms=65-->
<!--DATA 3	asym0_unpin	pp2048	18.40-->

### gemma4-12b-asym-pin2x2 [asym0_pin] pass 3  env='ROCKET_MM_ASYM=0 PIN_MASK=0xf0'  args='-t 4'  20:04:55  clk=600 MHz  MemAvail=31395572 kB
<!--PRED 3	asym0_pin	memavail_kb=31395572	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4274,2632,7658,7027,6110,5351,3819,3769,2503,164,245-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         19.17 ± 0.01 |

build: 171974745 (10558)
<!--RO 3	asym0_pin	wall_s=433	busy=378,400,373,378,19708,12974,13171,17196	busy_tot=64578	busy_little_share=0.0237	a55_cpu_cycles=58015619118	a55_inst_retired=12318732999	context_switches=9757862	a76_cpu_cycles=1381610354264	a76_l3d_cache_refill=21006587941	a76_l2d_cache_refill=10967632489	cpu_migrations=25505	page_faults=990791	a76_dtlb_walk=1167346872	a76_mem_access=364656499315	a76_inst_retired=1710612982910	a76_l1d_cache_refill=24713160871	a55_inst_share=0.0071	a76_ipc=1.238	l2ref_pki=6.412	l3ref_pki=12.280	l1dref_pki=14.447	memacc_pki=213.17	dtlbw_pki=0.6824	pmu_enabled=100.0	pmu_cpu_s=3458.0	who=at_s=8,rss_pg=6044326,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8247	l2color_cv=0.0043	l3color_cv=0.0114	contig_frac=0.9340	mean_run=14.7	vapa16=0.0052	gib_regions=32	anon_pg=12932	anon_l2cv=0.0047	anon_contig=0.9456	anon_run=17.8	anon_gib=30	file_pg=24576	file_l2cv=0.0043	file_contig=0.9928	file_run=109.7	file_gib=30	other_pg=23298	other_l2cv=0.0092	other_contig=0.8655	other_run=7.3	other_gib=32	ro_cost_ms=91-->
<!--DATA 3	asym0_pin	pp2048	19.17-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [asym0_unpin]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | asym0_unpin | 18.18 | 3 | 18.02 | 18.40 | -- | -- |
| pp2048 | asym0_pin | 19.38 | 3 | 19.17 | 19.55 | 1.067x | 1.080 1.078 1.042 |
| pp2048 | stock_unpin | 19.95 | 3 | 19.81 | 20.04 | 1.098x | 1.107 1.099 1.087 |
| pp2048 | stock_pin | 21.03 | 3 | 20.86 | 21.12 | 1.157x | 1.152 1.172 1.148 |

