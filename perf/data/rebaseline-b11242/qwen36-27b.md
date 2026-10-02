== qwen36-27b  Tue Sep 29 07:35:08 UTC 2026  cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000 ==
### qwen36-27b [cpu-rp0] pass 1  env=''  args='-b 2048 -ub 2048 --repack 0'  07:45:24  clk=200 MHz  MemAvail=31808852 kB
<!--PRED 1	cpu-rp0	memavail_kb=31808560	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4189,3557,2592,2156,1883,1337,991,863,731,592,2965	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | CPU        |       8 |     2048 |   0 |          pp2048 |          1.75 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 1	cpu-rp0	wall_s=2338	busy=223257,219764,196897,96995,226607,227066,227240,227349	busy_tot=1645175	busy_little_share=0.4479	a55_cpu_cycles=12974032871317	a55_inst_retired=8043242721920	context_switches=731746	a76_cpu_cycles=18970391465423	a76_l3d_cache_refill=64043747397	a76_l2d_cache_refill=12693905487	cpu_migrations=43385	page_faults=400299	a76_dtlb_walk=5087393490	a76_mem_access=13767485938852	a76_inst_retired=53636811113481	a76_l1d_cache_refill=134527416216	a55_inst_share=0.1304	a76_ipc=2.827	l2ref_pki=0.237	l3ref_pki=1.194	l1dref_pki=2.508	memacc_pki=256.68	dtlbw_pki=0.0948	pmu_enabled=100.0	pmu_cpu_s=18702.9	who=at_s=6,rss_pg=4260188,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=92160	present_frac=0.6652	l2color_cv=0.0057	l3color_cv=0.0150	contig_frac=0.9134	mean_run=11.3	vapa16=0.0746	gib_regions=28	anon_pg=18752	anon_l2cv=0.0052	anon_contig=0.9412	anon_run=16.5	anon_gib=28	file_pg=24576	file_l2cv=0.0009	file_contig=0.9984	file_run=282.5	file_gib=26	other_pg=17981	other_l2cv=0.0148	other_contig=0.7681	other_run=4.3	other_gib=28	ro_cost_ms=63-->
<!--DATA 1	cpu-rp0	pp2048	1.75-->

### qwen36-27b [cpu-rp1] pass 1  env=''  args='-b 2048 -ub 2048 --repack 1'  08:31:06  clk=200 MHz  MemAvail=31790768 kB
<!--PRED 1	cpu-rp1	memavail_kb=31789936	anonhuge_kb=0	hugepagesz_kb=2048	buddy=26355,19098,19296,16868,1663,1019,787,721,636,545,3067	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | CPU        |       8 |     2048 |          pp2048 |          2.99 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 1	cpu-rp1	wall_s=1395	busy=118204,116438,100642,80142,131264,128772,128717,127559	busy_tot=931738	busy_little_share=0.4459	a55_cpu_cycles=7289141288478	a55_inst_retired=4265520365529	context_switches=385693	a76_cpu_cycles=10802569294970	a76_l3d_cache_refill=140782787680	a76_l2d_cache_refill=29735995740	cpu_migrations=34369	page_faults=4269106	a76_dtlb_walk=4261634452	a76_mem_access=9441246042046	a76_inst_retired=22529626129916	a76_l1d_cache_refill=164041661044	a55_inst_share=0.1592	a76_ipc=2.086	l2ref_pki=1.320	l3ref_pki=6.249	l1dref_pki=7.281	memacc_pki=419.06	dtlbw_pki=0.1892	pmu_enabled=100.0	pmu_cpu_s=11150.1	who=at_s=22,rss_pg=7596021,settled=2	pfn_zero_frac=0.0000	maps=5	sampled=117248	present_frac=0.8001	l2color_cv=0.1687	l3color_cv=0.1903	contig_frac=0.8063	mean_run=5.1	vapa16=0.0592	gib_regions=32	anon_pg=53249	anon_l2cv=0.0319	anon_contig=0.9715	anon_run=32.8	anon_gib=32	file_pg=22147	file_l2cv=0.0117	file_contig=0.9965	file_run=97.1	file_gib=27	other_pg=18409	other_l2cv=0.7964	other_contig=0.1007	other_run=1.1	other_gib=28	ro_cost_ms=288-->
<!--DATA 1	cpu-rp1	pp2048	2.99-->

### qwen36-27b [npu] pass 1  env=''  args='-b 2048 -ub 2048 --repack 0'  08:57:39  clk=200 MHz  MemAvail=31817400 kB
<!--PRED 1	npu	memavail_kb=31817256	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4959,3775,3430,2330,700,426,313,280,222,192,3456	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | ngl | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | ROCKET     |  -1 |     2048 |   0 |          pp2048 |          9.58 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 1	npu	wall_s=435	busy=5553,5433,5290,4654,17780,17756,16960,16781	busy_tot=90207	busy_little_share=0.2320	a55_cpu_cycles=380944095952	a55_inst_retired=171839816925	context_switches=12153669	a76_cpu_cycles=1516480134331	a76_l3d_cache_refill=17026508324	a76_l2d_cache_refill=7438119157	cpu_migrations=18203	page_faults=1794305	a76_dtlb_walk=4645623631	a76_mem_access=625598407079	a76_inst_retired=2693982765273	a76_l1d_cache_refill=14070822029	a55_inst_share=0.0600	a76_ipc=1.776	l2ref_pki=2.761	l3ref_pki=6.320	l1dref_pki=5.223	memacc_pki=232.22	dtlbw_pki=1.7244	pmu_enabled=100.0	pmu_cpu_s=3470.7	who=at_s=10,rss_pg=4503249,settled=2	pfn_zero_frac=0.0000	maps=12	sampled=245760	present_frac=0.5029	l2color_cv=0.0185	l3color_cv=0.0252	contig_frac=0.8528	mean_run=6.7	vapa16=0.1159	gib_regions=28	accel_pg=0	accel_l2cv=0.0000	accel_contig=0.0000	accel_run=0.0	accel_gib=0	anon_pg=75920	anon_l2cv=0.0016	anon_contig=0.8495	anon_run=6.6	anon_gib=28	file_pg=24576	file_l2cv=0.0013	file_contig=0.9968	file_run=195.0	file_gib=26	other_pg=23097	other_l2cv=0.0983	other_contig=0.7102	other_run=3.4	other_gib=28	ro_cost_ms=157-->
<!--DATA 1	npu	pp2048	9.58-->

### qwen36-27b [cpu-rp1] pass 2  env=''  args='-b 2048 -ub 2048 --repack 1'  09:11:39  clk=200 MHz  MemAvail=31809768 kB
<!--PRED 2	cpu-rp1	memavail_kb=31808800	anonhuge_kb=0	hugepagesz_kb=2048	buddy=25687,22804,16566,15429,1034,463,340,293,245,200,3466	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | CPU        |       8 |     2048 |          pp2048 |          2.99 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 2	cpu-rp1	wall_s=1394	busy=118371,116841,103839,74439,129728,129048,130065,128141	busy_tot=930472	busy_little_share=0.4444	a55_cpu_cycles=7252835513851	a55_inst_retired=4250945973594	context_switches=381851	a76_cpu_cycles=10813094114467	a76_l3d_cache_refill=140830108218	a76_l2d_cache_refill=29809234919	cpu_migrations=34418	page_faults=4238240	a76_dtlb_walk=4273718384	a76_mem_access=9446683669487	a76_inst_retired=22542136528701	a76_l1d_cache_refill=164118995367	a55_inst_share=0.1587	a76_ipc=2.085	l2ref_pki=1.322	l3ref_pki=6.247	l1dref_pki=7.281	memacc_pki=419.07	dtlbw_pki=0.1896	pmu_enabled=100.0	pmu_cpu_s=11148.5	who=at_s=22,rss_pg=7598439,settled=2	pfn_zero_frac=0.0000	maps=5	sampled=116736	present_frac=0.8047	l2color_cv=0.2288	l3color_cv=0.2505	contig_frac=0.8225	mean_run=5.5	vapa16=0.0434	gib_regions=32	anon_pg=53560	anon_l2cv=0.0269	anon_contig=0.9787	anon_run=42.8	anon_gib=32	file_pg=22104	file_l2cv=0.0176	file_contig=0.9919	file_run=64.6	file_gib=27	other_pg=18269	other_l2cv=1.1234	other_contig=0.1608	other_run=1.2	other_gib=28	ro_cost_ms=134-->
<!--DATA 2	cpu-rp1	pp2048	2.99-->

### qwen36-27b [npu] pass 2  env=''  args='-b 2048 -ub 2048 --repack 0'  09:38:14  clk=200 MHz  MemAvail=31801640 kB
<!--PRED 2	npu	memavail_kb=31801220	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4398,3866,3763,1766,727,480,337,310,245,204,3437	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | ngl | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | ROCKET     |  -1 |     2048 |   0 |          pp2048 |          9.61 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 2	npu	wall_s=434	busy=5724,5328,5183,4877,17759,17420,16700,17592	busy_tot=90583	busy_little_share=0.2331	a55_cpu_cycles=384278348668	a55_inst_retired=173632441874	context_switches=12141023	a76_cpu_cycles=1517306600386	a76_l3d_cache_refill=16958542175	a76_l2d_cache_refill=7452874704	cpu_migrations=18210	page_faults=1797537	a76_dtlb_walk=4297737568	a76_mem_access=624517923131	a76_inst_retired=2692299004585	a76_l1d_cache_refill=14061602088	a55_inst_share=0.0606	a76_ipc=1.774	l2ref_pki=2.768	l3ref_pki=6.299	l1dref_pki=5.223	memacc_pki=231.96	dtlbw_pki=1.5963	pmu_enabled=100.0	pmu_cpu_s=3466.7	who=at_s=10,rss_pg=4503250,settled=2	pfn_zero_frac=0.0000	maps=12	sampled=245760	present_frac=0.5031	l2color_cv=0.0116	l3color_cv=0.0165	contig_frac=0.8484	mean_run=6.5	vapa16=0.0375	gib_regions=28	accel_pg=0	accel_l2cv=0.0000	accel_contig=0.0000	accel_run=0.0	accel_gib=0	anon_pg=75733	anon_l2cv=0.0036	anon_contig=0.9023	anon_run=10.0	anon_gib=28	file_pg=24576	file_l2cv=0.0017	file_contig=0.9965	file_run=182.0	file_gib=28	other_pg=23343	other_l2cv=0.0617	other_contig=0.5177	other_run=2.1	other_gib=28	ro_cost_ms=161-->
<!--DATA 2	npu	pp2048	9.61-->

### qwen36-27b [cpu-rp0] pass 2  env=''  args='-b 2048 -ub 2048 --repack 0'  09:55:43  clk=200 MHz  MemAvail=31808536 kB
<!--PRED 2	cpu-rp0	memavail_kb=31808536	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3142,3547,2378,2688,916,461,367,317,269,218,3421	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | CPU        |       8 |     2048 |   0 |          pp2048 |          1.75 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 2	cpu-rp0	wall_s=2339	busy=223359,215844,198052,92710,226019,227748,228257,227681	busy_tot=1639670	busy_little_share=0.4452	a55_cpu_cycles=12842242116077	a55_inst_retired=7975532092022	context_switches=748876	a76_cpu_cycles=18993195052542	a76_l3d_cache_refill=64111650108	a76_l2d_cache_refill=12744554746	cpu_migrations=44109	page_faults=410981	a76_dtlb_walk=5153185258	a76_mem_access=13787355027675	a76_inst_retired=53704243771276	a76_l1d_cache_refill=135016816738	a55_inst_share=0.1293	a76_ipc=2.828	l2ref_pki=0.237	l3ref_pki=1.194	l1dref_pki=2.514	memacc_pki=256.73	dtlbw_pki=0.0960	pmu_enabled=100.0	pmu_cpu_s=18706.2	who=at_s=6,rss_pg=4264770,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=92160	present_frac=0.6785	l2color_cv=0.0057	l3color_cv=0.0165	contig_frac=0.8874	mean_run=8.7	vapa16=0.0722	gib_regions=28	anon_pg=19726	anon_l2cv=0.0057	anon_contig=0.9434	anon_run=17.1	anon_gib=28	file_pg=24576	file_l2cv=0.0027	file_contig=0.9913	file_run=93.8	file_gib=28	other_pg=18231	other_l2cv=0.0197	other_contig=0.6868	other_run=3.2	other_gib=28	ro_cost_ms=75-->
<!--DATA 2	cpu-rp0	pp2048	1.75-->

#### summary: per-arm mean over 2 passes, ratios paired within a pass against [cpu-rp0]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | cpu-rp0 | 1.75 | 2 | 1.75 | 1.75 | -- | -- |
| pp2048 | cpu-rp1 | 2.99 | 2 | 2.99 | 2.99 | 1.709x | 1.709 1.709 |
| pp2048 | npu | 9.59 | 2 | 9.58 | 9.61 | 5.483x | 5.474 5.491 |

