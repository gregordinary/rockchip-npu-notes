<!-- gemma4-12b F16, what full residency buys  TESTS='-p 2048 -n 0 -r 3'  PASSES=3
     gguf=<data>/gemma4/gemma-4-12b-it-F16.gguf (23832065184 bytes)
     so=61f02a02345f3b2e27cef45de8ade0c2
     res87    = ROCKET_F16_RESIDENT=auto                        expect 286 of 328 (18078MB)
     res87_t8 = + ROCKET_N_THREADS=8                            expect 286 of 328 (thread control)
     res100   = + ROCKET_QUANT_RESIDENT_RESERVE_MB=6144         expect 328 of 328 (20790MB, 0 streamed)
     All arms UNPINNED, matching every other ratio in the matrix. Pinning interacts with
     the residency knob on this model by -1.5pp against a 6.1pp knob, so a ratio of this
     size is read unpinned and the interaction is a separate row.
-->
== gemma4-12b-f16-res100  Tue Sep  1 20:59:53 UTC 2026 ==
### gemma4-12b-f16-res100 [res87] pass 1  env='ROCKET_F16_RESIDENT=auto'  args=''  21:03:38  clk=600 MHz  MemAvail=31417024 kB
<!--PRED 1	res87	memavail_kb=31418196	anonhuge_kb=0	hugepagesz_kb=2048	buddy=43059,28907,22143,23157,5497,5240,4600,3801,2682,37,2-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         21.48 ± 0.42 |

build: 171974745 (10558)
<!--RO 1	res87	wall_s=532	busy=6201,5851,5734,5753,16241,10532,11907,12961	busy_tot=75180	busy_little_share=0.3131	a55_cpu_cycles=459849953750	a55_inst_retired=282878920058	context_switches=18608131	a76_cpu_cycles=1194426837473	a76_l3d_cache_refill=14420624464	a76_l2d_cache_refill=7625738704	cpu_migrations=47452	page_faults=14037716	a76_dtlb_walk=894035092	a76_mem_access=377455314002	a76_inst_retired=1715796421441	a76_l1d_cache_refill=22382715098	a55_inst_share=0.1415	a76_ipc=1.437	l2ref_pki=4.444	l3ref_pki=8.405	l1dref_pki=13.045	memacc_pki=219.99	dtlbw_pki=0.5211	pmu_enabled=100.0	pmu_cpu_s=4247.6	who=at_s=8,rss_pg=6037407,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8138	l2color_cv=0.1264	l3color_cv=0.1358	contig_frac=0.7259	mean_run=3.6	vapa16=0.0749	gib_regions=32	anon_pg=12303	anon_l2cv=0.0223	anon_contig=0.9653	anon_run=27.2	anon_gib=24	file_pg=24576	file_l2cv=0.0489	file_contig=0.6727	file_run=3.0	file_gib=32	other_pg=23123	other_l2cv=0.3483	other_contig=0.6550	other_run=2.9	other_gib=29	ro_cost_ms=82-->
<!--DATA 1	res87	pp2048	21.48-->
    [f16-resident] admission first declined at 18078MB resident: MemAvailable 9457MB fell below the 9535MB reserve floor
    [f16-resident] weights offered to the resident route: 286 resident on the NPU (18078MB), 42 streamed via the per-call pack -- 87% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21022MB (MemAvailable 30558MB - reserve 9535MB, no swap)

### gemma4-12b-f16-res100 [res87_t8] pass 1  env='ROCKET_F16_RESIDENT=auto ROCKET_N_THREADS=8'  args=''  21:16:32  clk=600 MHz  MemAvail=31435688 kB
<!--PRED 1	res87_t8	memavail_kb=31435688	anonhuge_kb=0	hugepagesz_kb=2048	buddy=32878,19666,18811,14703,5792,5192,4619,3699,3155,39,2-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         21.01 ± 0.95 |

build: 171974745 (10558)
<!--RO 1	res87_t8	wall_s=527	busy=7306,6312,6126,6086,14376,13800,14940,14544	busy_tot=83490	busy_little_share=0.3094	a55_cpu_cycles=511573914180	a55_inst_retired=306727358418	context_switches=19651611	a76_cpu_cycles=1330970093000	a76_l3d_cache_refill=15514912993	a76_l2d_cache_refill=8933485954	cpu_migrations=80656	page_faults=14339960	a76_dtlb_walk=1004153923	a76_mem_access=397851079049	a76_inst_retired=1751322453745	a76_l1d_cache_refill=20027367077	a55_inst_share=0.1490	a76_ipc=1.316	l2ref_pki=5.101	l3ref_pki=8.859	l1dref_pki=11.436	memacc_pki=227.17	dtlbw_pki=0.5734	pmu_enabled=100.0	pmu_cpu_s=4209.8	who=at_s=8,rss_pg=6037404,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8195	l2color_cv=0.0962	l3color_cv=0.1105	contig_frac=0.7348	mean_run=3.7	vapa16=0.1027	gib_regions=32	anon_pg=12303	anon_l2cv=0.0704	anon_contig=0.8941	anon_run=9.3	anon_gib=28	file_pg=24576	file_l2cv=0.0637	file_contig=0.6747	file_run=3.1	file_gib=32	other_pg=23539	other_l2cv=0.2580	other_contig=0.7143	other_run=3.5	other_gib=29	ro_cost_ms=93-->
<!--DATA 1	res87_t8	pp2048	21.01-->
    [f16-resident] admission first declined at 17730MB resident: MemAvailable 9465MB fell below the 9535MB reserve floor
    [f16-resident] weights offered to the resident route: 281 resident on the NPU (17730MB), 47 streamed via the per-call pack -- 86% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 20999MB (MemAvailable 30534MB - reserve 9535MB, no swap)

### gemma4-12b-f16-res100 [res100] pass 1  env='ROCKET_F16_RESIDENT=auto ROCKET_N_THREADS=8 ROCKET_QUANT_RESIDENT_RESERVE_MB=6144'  args=''  21:29:43  clk=600 MHz  MemAvail=31341032 kB
<!--PRED 1	res100	memavail_kb=31341032	anonhuge_kb=0	hugepagesz_kb=2048	buddy=22968,24723,21510,17018,5832,5479,4801,3951,2752,27,2-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         21.31 ± 1.00 |

build: 171974745 (10558)
<!--RO 1	res100	wall_s=551	busy=7240,6346,6229,6074,13272,12898,14825,14022	busy_tot=80906	busy_little_share=0.3200	a55_cpu_cycles=512377018462	a55_inst_retired=307411100835	context_switches=18196913	a76_cpu_cycles=1251219204196	a76_l3d_cache_refill=14236154881	a76_l2d_cache_refill=7193946241	cpu_migrations=75197	page_faults=14673447	a76_dtlb_walk=887235045	a76_mem_access=391764837492	a76_inst_retired=1736169662724	a76_l1d_cache_refill=18957983735	a55_inst_share=0.1504	a76_ipc=1.388	l2ref_pki=4.144	l3ref_pki=8.200	l1dref_pki=10.919	memacc_pki=225.65	dtlbw_pki=0.5110	pmu_enabled=100.0	pmu_cpu_s=4400.0	who=at_s=8,rss_pg=6037396,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8176	l2color_cv=0.3394	l3color_cv=0.3687	contig_frac=0.6555	mean_run=2.9	vapa16=0.0805	gib_regions=32	anon_pg=12303	anon_l2cv=0.0282	anon_contig=0.8471	anon_run=6.5	anon_gib=32	file_pg=24576	file_l2cv=0.0810	file_contig=0.7377	file_run=3.8	file_gib=32	other_pg=23403	other_l2cv=0.8595	other_contig=0.4683	other_run=1.9	other_gib=29	ro_cost_ms=61-->
<!--DATA 1	res100	pp2048	21.31-->
    [f16-resident] weights offered to the resident route: 328 resident on the NPU (20790MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 24373MB (MemAvailable 30517MB - reserve 6144MB, no swap)

### gemma4-12b-f16-res100 [res87_t8] pass 2  env='ROCKET_F16_RESIDENT=auto ROCKET_N_THREADS=8'  args=''  21:42:59  clk=600 MHz  MemAvail=31256108 kB
<!--PRED 2	res87_t8	memavail_kb=31256464	anonhuge_kb=0	hugepagesz_kb=2048	buddy=14490,14107,19661,13617,5967,5603,4895,4063,2852,14,1-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         20.71 ± 0.81 |

build: 171974745 (10558)
<!--RO 2	res87_t8	wall_s=534	busy=7330,6377,6181,6252,14430,13556,15105,14633	busy_tot=83864	busy_little_share=0.3117	a55_cpu_cycles=517236624280	a55_inst_retired=307837071974	context_switches=19999931	a76_cpu_cycles=1327169513757	a76_l3d_cache_refill=15625180240	a76_l2d_cache_refill=9229648440	cpu_migrations=71606	page_faults=14508294	a76_dtlb_walk=941018757	a76_mem_access=389881122174	a76_inst_retired=1729313449704	a76_l1d_cache_refill=19906798764	a55_inst_share=0.1511	a76_ipc=1.303	l2ref_pki=5.337	l3ref_pki=9.035	l1dref_pki=11.511	memacc_pki=225.45	dtlbw_pki=0.5442	pmu_enabled=100.0	pmu_cpu_s=4267.6	who=at_s=8,rss_pg=6037361,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8190	l2color_cv=0.1041	l3color_cv=0.1170	contig_frac=0.7194	mean_run=3.5	vapa16=0.0423	gib_regions=32	anon_pg=12296	anon_l2cv=0.1397	anon_contig=0.7587	anon_run=4.1	anon_gib=32	file_pg=24576	file_l2cv=0.0645	file_contig=0.6899	file_run=3.2	file_gib=32	other_pg=23510	other_l2cv=0.2576	other_contig=0.7297	other_run=3.7	other_gib=32	ro_cost_ms=60-->
<!--DATA 2	res87_t8	pp2048	20.71-->
    [f16-resident] admission first declined at 17730MB resident: MemAvailable 9462MB fell below the 9535MB reserve floor
    [f16-resident] weights offered to the resident route: 281 resident on the NPU (17730MB), 47 streamed via the per-call pack -- 86% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 20877MB (MemAvailable 30413MB - reserve 9535MB, no swap)

### gemma4-12b-f16-res100 [res100] pass 2  env='ROCKET_F16_RESIDENT=auto ROCKET_N_THREADS=8 ROCKET_QUANT_RESIDENT_RESERVE_MB=6144'  args=''  21:55:40  clk=600 MHz  MemAvail=31403832 kB
<!--PRED 2	res100	memavail_kb=31403540	anonhuge_kb=0	hugepagesz_kb=2048	buddy=34549,16693,15218,14311,5619,5249,4548,3764,2441,51,187-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         21.62 ± 0.50 |

build: 171974745 (10558)
<!--RO 2	res100	wall_s=639	busy=7553,6539,6189,6047,13840,13558,13921,14691	busy_tot=82338	busy_little_share=0.3198	a55_cpu_cycles=522168153139	a55_inst_retired=311432624714	context_switches=19328602	a76_cpu_cycles=1282506124092	a76_l3d_cache_refill=14332174580	a76_l2d_cache_refill=7786495557	cpu_migrations=82495	page_faults=15219248	a76_dtlb_walk=908625783	a76_mem_access=403049729342	a76_inst_retired=1770487102499	a76_l1d_cache_refill=19061982307	a55_inst_share=0.1496	a76_ipc=1.380	l2ref_pki=4.398	l3ref_pki=8.095	l1dref_pki=10.767	memacc_pki=227.65	dtlbw_pki=0.5132	pmu_enabled=100.0	pmu_cpu_s=5108.6	who=at_s=8,rss_pg=6037351,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8194	l2color_cv=0.1965	l3color_cv=0.2170	contig_frac=0.6751	mean_run=3.1	vapa16=0.0461	gib_regions=32	anon_pg=12296	anon_l2cv=0.0324	anon_contig=0.9412	anon_run=16.4	anon_gib=32	file_pg=24576	file_l2cv=0.1014	file_contig=0.7915	file_run=4.8	file_gib=31	other_pg=23543	other_l2cv=0.5254	other_contig=0.4146	other_run=1.7	other_gib=32	ro_cost_ms=60-->
<!--DATA 2	res100	pp2048	21.62-->
    [f16-resident] weights offered to the resident route: 328 resident on the NPU (20790MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 24430MB (MemAvailable 30574MB - reserve 6144MB, no swap)

### gemma4-12b-f16-res100 [res87] pass 2  env='ROCKET_F16_RESIDENT=auto'  args=''  22:10:05  clk=600 MHz  MemAvail=31214960 kB
<!--PRED 2	res87	memavail_kb=31214960	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5686,12000,11898,11013,6122,5746,5007,4078,2754,134,2-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         20.87 ± 0.92 |

build: 171974745 (10558)
<!--RO 2	res87	wall_s=536	busy=6223,6016,5948,5916,16218,11414,12895,13069	busy_tot=77699	busy_little_share=0.3102	a55_cpu_cycles=475804854680	a55_inst_retired=285775553147	context_switches=19105537	a76_cpu_cycles=1239493637083	a76_l3d_cache_refill=14822119270	a76_l2d_cache_refill=9768538502	cpu_migrations=52345	page_faults=14270724	a76_dtlb_walk=868376458	a76_mem_access=385353438578	a76_inst_retired=1740571186027	a76_l1d_cache_refill=23667394055	a55_inst_share=0.1410	a76_ipc=1.404	l2ref_pki=5.612	l3ref_pki=8.516	l1dref_pki=13.597	memacc_pki=221.39	dtlbw_pki=0.4989	pmu_enabled=100.0	pmu_cpu_s=4280.8	who=at_s=8,rss_pg=6037384,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8136	l2color_cv=0.0442	l3color_cv=0.0513	contig_frac=0.7345	mean_run=3.7	vapa16=0.1819	gib_regions=32	anon_pg=12303	anon_l2cv=0.0361	anon_contig=0.8024	anon_run=5.0	anon_gib=32	file_pg=24576	file_l2cv=0.0982	file_contig=0.6618	file_run=2.9	file_gib=32	other_pg=23108	other_l2cv=0.0883	other_contig=0.7756	other_run=4.4	other_gib=32	ro_cost_ms=64-->
<!--DATA 2	res87	pp2048	20.87-->
    [f16-resident] admission first declined at 18078MB resident: MemAvailable 9447MB fell below the 9535MB reserve floor
    [f16-resident] weights offered to the resident route: 286 resident on the NPU (18078MB), 42 streamed via the per-call pack -- 87% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 20864MB (MemAvailable 30399MB - reserve 9535MB, no swap)

### gemma4-12b-f16-res100 [res100] pass 3  env='ROCKET_F16_RESIDENT=auto ROCKET_N_THREADS=8 ROCKET_QUANT_RESIDENT_RESERVE_MB=6144'  args=''  22:22:53  clk=600 MHz  MemAvail=31265404 kB
<!--PRED 3	res100	memavail_kb=31265404	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8487,12002,15002,11510,5917,5558,4817,3877,2475,124,117-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         21.20 ± 0.97 |

build: 171974745 (10558)
<!--RO 3	res100	wall_s=556	busy=7449,6490,6209,6123,14124,13017,13871,14048	busy_tot=81331	busy_little_share=0.3230	a55_cpu_cycles=517651703724	a55_inst_retired=309303710508	context_switches=18244792	a76_cpu_cycles=1256358665544	a76_l3d_cache_refill=14471467335	a76_l2d_cache_refill=7774880304	cpu_migrations=72159	page_faults=14699417	a76_dtlb_walk=886590499	a76_mem_access=396025470268	a76_inst_retired=1748581552483	a76_l1d_cache_refill=18850600939	a55_inst_share=0.1503	a76_ipc=1.392	l2ref_pki=4.446	l3ref_pki=8.276	l1dref_pki=10.781	memacc_pki=226.48	dtlbw_pki=0.5070	pmu_enabled=100.0	pmu_cpu_s=4439.3	who=at_s=8,rss_pg=6037384,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8192	l2color_cv=0.0499	l3color_cv=0.0575	contig_frac=0.8247	mean_run=5.6	vapa16=0.0335	gib_regions=32	anon_pg=12289	anon_l2cv=0.0226	anon_contig=0.9703	anon_run=31.5	anon_gib=24	file_pg=24576	file_l2cv=0.0611	file_contig=0.7883	file_run=4.7	file_gib=32	other_pg=23531	other_l2cv=0.1412	other_contig=0.7866	other_run=4.6	other_gib=32	ro_cost_ms=84-->
<!--DATA 3	res100	pp2048	21.20-->
    [f16-resident] weights offered to the resident route: 328 resident on the NPU (20790MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 24248MB (MemAvailable 30392MB - reserve 6144MB, no swap)

### gemma4-12b-f16-res100 [res87] pass 3  env='ROCKET_F16_RESIDENT=auto'  args=''  22:36:10  clk=600 MHz  MemAvail=31205172 kB
<!--PRED 3	res87	memavail_kb=31205532	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10708,14606,16477,13188,6150,5815,5016,4098,2782,14,2-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         21.20 ± 1.00 |

build: 171974745 (10558)
<!--RO 3	res87	wall_s=547	busy=6277,5883,5849,5828,15640,11368,12206,13582	busy_tot=76633	busy_little_share=0.3111	a55_cpu_cycles=469658801978	a55_inst_retired=286365106256	context_switches=19293236	a76_cpu_cycles=1217073050812	a76_l3d_cache_refill=14560546561	a76_l2d_cache_refill=7959348994	cpu_migrations=48607	page_faults=14367420	a76_dtlb_walk=864934933	a76_mem_access=389858386727	a76_inst_retired=1750012416657	a76_l1d_cache_refill=22761610990	a55_inst_share=0.1406	a76_ipc=1.438	l2ref_pki=4.548	l3ref_pki=8.320	l1dref_pki=13.007	memacc_pki=222.77	dtlbw_pki=0.4942	pmu_enabled=100.0	pmu_cpu_s=4365.2	who=at_s=8,rss_pg=6037389,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8133	l2color_cv=0.3136	l3color_cv=0.3449	contig_frac=0.6045	mean_run=2.5	vapa16=0.0461	gib_regions=32	anon_pg=12303	anon_l2cv=0.0119	anon_contig=0.7560	anon_run=4.1	anon_gib=32	file_pg=24576	file_l2cv=0.0759	file_contig=0.6560	file_run=2.9	file_gib=32	other_pg=23081	other_l2cv=0.8540	other_contig=0.4690	other_run=1.9	other_gib=32	ro_cost_ms=86-->
<!--DATA 3	res87	pp2048	21.20-->
    [f16-resident] admission first declined at 18078MB resident: MemAvailable 9443MB fell below the 9535MB reserve floor
    [f16-resident] weights offered to the resident route: 286 resident on the NPU (18078MB), 42 streamed via the per-call pack -- 87% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 20880MB (MemAvailable 30415MB - reserve 9535MB, no swap)

### gemma4-12b-f16-res100 [res87_t8] pass 3  env='ROCKET_F16_RESIDENT=auto ROCKET_N_THREADS=8'  args=''  22:49:14  clk=600 MHz  MemAvail=31424324 kB
<!--PRED 3	res87_t8	memavail_kb=31424324	anonhuge_kb=0	hugepagesz_kb=2048	buddy=32223,15578,12979,12107,5792,5825,4926,4028,3024,37,6-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         21.02 ± 0.81 |

build: 171974745 (10558)
<!--RO 3	res87_t8	wall_s=518	busy=7263,6282,6133,6033,14058,13622,14704,14951	busy_tot=83046	busy_little_share=0.3096	a55_cpu_cycles=510306352931	a55_inst_retired=306022119331	context_switches=19778836	a76_cpu_cycles=1321097512981	a76_l3d_cache_refill=15453778578	a76_l2d_cache_refill=8455551260	cpu_migrations=77671	page_faults=14410977	a76_dtlb_walk=955324013	a76_mem_access=394889301935	a76_inst_retired=1744076462155	a76_l1d_cache_refill=19869832671	a55_inst_share=0.1493	a76_ipc=1.320	l2ref_pki=4.848	l3ref_pki=8.861	l1dref_pki=11.393	memacc_pki=226.42	dtlbw_pki=0.5478	pmu_enabled=100.0	pmu_cpu_s=4138.4	who=at_s=8,rss_pg=6037410,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8166	l2color_cv=0.1680	l3color_cv=0.1879	contig_frac=0.7548	mean_run=4.1	vapa16=0.0334	gib_regions=32	anon_pg=12296	anon_l2cv=0.0075	anon_contig=0.9636	anon_run=26.0	anon_gib=29	file_pg=24576	file_l2cv=0.0785	file_contig=0.6572	file_run=2.9	file_gib=32	other_pg=23333	other_l2cv=0.3963	other_contig=0.7476	other_run=3.9	other_gib=29	ro_cost_ms=86-->
<!--DATA 3	res87_t8	pp2048	21.02-->
    [f16-resident] admission first declined at 17730MB resident: MemAvailable 9461MB fell below the 9535MB reserve floor
    [f16-resident] weights offered to the resident route: 281 resident on the NPU (17730MB), 47 streamed via the per-call pack -- 86% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21004MB (MemAvailable 30540MB - reserve 9535MB, no swap)

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [res87]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | res87 | 21.18 | 3 | 20.87 | 21.48 | -- | -- |
| pp2048 | res87_t8 | 20.91 | 3 | 20.71 | 21.02 | 0.987x | 0.978 0.992 0.992 |
| pp2048 | res100 | 21.38 | 3 | 21.20 | 21.62 | 1.009x | 0.992 1.036 1.000 |

