== gpt-oss-20b  Tue Sep 29 04:19:03 UTC 2026  cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000 ==
### gpt-oss-20b [cpu-rp0] pass 1  env=''  args='-b 2048 -ub 2048 --repack 0'  04:20:45  clk=200 MHz  MemAvail=31824664 kB
<!--PRED 1	cpu-rp0	memavail_kb=31823876	anonhuge_kb=0	hugepagesz_kb=2048	buddy=2992,2790,2790,1968,1741,1530,1134,849,724,544,4181	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CPU        |       8 |     2048 |   0 |          pp2048 |         11.94 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 1	cpu-rp0	wall_s=345	busy=30452,28531,27554,14580,32059,32957,32682,32667	busy_tot=231482	busy_little_share=0.4368	a55_cpu_cycles=1779663527630	a55_inst_retired=1260261703124	context_switches=115489	a76_cpu_cycles=2729144421766	a76_l3d_cache_refill=8474939954	a76_l2d_cache_refill=4826458286	cpu_migrations=10911	page_faults=187116	a76_dtlb_walk=526952894	a76_mem_access=1646940283719	a76_inst_retired=8438923957473	a76_l1d_cache_refill=8381422122	a55_inst_share=0.1299	a76_ipc=3.092	l2ref_pki=0.572	l3ref_pki=1.004	l1dref_pki=0.993	memacc_pki=195.16	dtlbw_pki=0.0624	pmu_enabled=100.0	pmu_cpu_s=2749.7	who=at_s=6,rss_pg=3067201,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=72192	present_frac=0.7171	l2color_cv=0.0058	l3color_cv=0.0128	contig_frac=0.8478	mean_run=6.4	vapa16=0.0092	gib_regions=28	anon_pg=4344	anon_l2cv=0.0224	anon_contig=0.4405	anon_run=1.8	anon_gib=26	file_pg=24576	file_l2cv=0.0019	file_contig=0.9933	file_run=115.9	file_gib=28	other_pg=22850	other_l2cv=0.0134	other_contig=0.7670	other_run=4.3	other_gib=26	ro_cost_ms=187-->
<!--DATA 1	cpu-rp0	pp2048	11.94-->

### gpt-oss-20b [cpu-rp1] pass 1  env=''  args='-b 2048 -ub 2048 --repack 1'  04:28:03  clk=200 MHz  MemAvail=31816944 kB
<!--PRED 1	cpu-rp1	memavail_kb=31816944	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3274,3069,2445,1690,1564,1516,1132,831,706,544,4192	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CPU        |       8 |     2048 |          pp2048 |         15.85 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 1	cpu-rp1	wall_s=267	busy=16351,16029,14204,11076,24104,24209,24457,23427	busy_tot=153857	busy_little_share=0.3748	a55_cpu_cycles=1011743553345	a55_inst_retired=863391611606	context_switches=69982	a76_cpu_cycles=2015272008786	a76_l3d_cache_refill=3306958866	a76_l2d_cache_refill=1310012800	cpu_migrations=10042	page_faults=2959531	a76_dtlb_walk=100767297	a76_mem_access=986479554350	a76_inst_retired=6019984604220	a76_l1d_cache_refill=7239565873	a55_inst_share=0.1254	a76_ipc=2.987	l2ref_pki=0.218	l3ref_pki=0.549	l1dref_pki=1.203	memacc_pki=163.87	dtlbw_pki=0.0167	pmu_enabled=100.0	pmu_cpu_s=2133.5	who=at_s=12,rss_pg=5746488,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=96768	present_frac=0.8211	l2color_cv=0.0040	l3color_cv=0.0068	contig_frac=0.9058	mean_run=10.4	vapa16=0.0078	gib_regions=28	anon_pg=32271	anon_l2cv=0.0025	anon_contig=0.9199	anon_run=12.2	anon_gib=28	file_pg=24576	file_l2cv=0.0002	file_contig=0.9991	file_run=351.1	file_gib=24	other_pg=22609	other_l2cv=0.0136	other_contig=0.7843	other_run=4.6	other_gib=26	ro_cost_ms=96-->
<!--DATA 1	cpu-rp1	pp2048	15.85-->

### gpt-oss-20b [npu] pass 1  env=''  args='-b 2048 -ub 2048 --repack 0'  04:34:03  clk=200 MHz  MemAvail=31805240 kB
<!--PRED 1	npu	memavail_kb=31804548	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3993,3794,4033,2970,2224,1645,1228,1061,947,760,3953	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | ngl | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |   0 |          pp2048 |         30.00 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 1	npu	wall_s=173	busy=5941,5502,5522,4269,11606,10846,10090,9400	busy_tot=63176	busy_little_share=0.3361	a55_cpu_cycles=382143268177	a55_inst_retired=262398725182	context_switches=1585695	a76_cpu_cycles=895996105397	a76_l3d_cache_refill=5830327421	a76_l2d_cache_refill=2535414375	cpu_migrations=24590	page_faults=7488711	a76_dtlb_walk=266562906	a76_mem_access=562542960651	a76_inst_retired=2105656790656	a76_l1d_cache_refill=9632966697	a55_inst_share=0.1108	a76_ipc=2.350	l2ref_pki=1.204	l3ref_pki=2.769	l1dref_pki=4.575	memacc_pki=267.16	dtlbw_pki=0.1266	pmu_enabled=100.0	pmu_cpu_s=1377.6	who=at_s=8,rss_pg=3156880,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.7518	l2color_cv=0.0063	l3color_cv=0.0195	contig_frac=0.8830	mean_run=8.4	vapa16=0.0111	gib_regions=29	anon_pg=6662	anon_l2cv=0.0160	anon_contig=0.8130	anon_run=5.3	anon_gib=24	file_pg=24576	file_l2cv=0.0012	file_contig=0.9949	file_run=142.1	file_gib=27	other_pg=24193	other_l2cv=0.0147	other_contig=0.7886	other_run=4.7	other_gib=28	ro_cost_ms=101-->
<!--DATA 1	npu	pp2048	30.00-->
    [moe-int8] experts exercised: 1668 resident on the NPU (13454MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24787MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM of which 8471MB is GGUF source, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24787MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gpt-oss-20b [cpu-rp1] pass 2  env=''  args='-b 2048 -ub 2048 --repack 1'  04:38:28  clk=200 MHz  MemAvail=31821400 kB
<!--PRED 2	cpu-rp1	memavail_kb=31821036	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3319,3249,2771,2192,1670,1594,1321,963,848,721,4031	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CPU        |       8 |     2048 |          pp2048 |         15.95 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 2	cpu-rp1	wall_s=266	busy=16041,15763,14394,11720,24161,23773,24540,23649	busy_tot=154041	busy_little_share=0.3760	a55_cpu_cycles=1016361036129	a55_inst_retired=867040477826	context_switches=69048	a76_cpu_cycles=2013816566470	a76_l3d_cache_refill=3325001415	a76_l2d_cache_refill=1307929263	cpu_migrations=9822	page_faults=2975238	a76_dtlb_walk=101729306	a76_mem_access=986180068847	a76_inst_retired=6016403479308	a76_l1d_cache_refill=7273767376	a55_inst_share=0.1260	a76_ipc=2.988	l2ref_pki=0.217	l3ref_pki=0.553	l1dref_pki=1.209	memacc_pki=163.92	dtlbw_pki=0.0169	pmu_enabled=100.0	pmu_cpu_s=2122.5	who=at_s=12,rss_pg=5746456,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=97280	present_frac=0.8189	l2color_cv=0.0038	l3color_cv=0.0073	contig_frac=0.8970	mean_run=9.5	vapa16=0.0069	gib_regions=28	anon_pg=32284	anon_l2cv=0.0082	anon_contig=0.9144	anon_run=11.4	anon_gib=28	file_pg=24576	file_l2cv=0.0002	file_contig=0.9988	file_run=319.2	file_gib=21	other_pg=22804	other_l2cv=0.0073	other_contig=0.7626	other_run=4.2	other_gib=27	ro_cost_ms=294-->
<!--DATA 2	cpu-rp1	pp2048	15.95-->

### gpt-oss-20b [npu] pass 2  env=''  args='-b 2048 -ub 2048 --repack 0'  04:44:27  clk=200 MHz  MemAvail=31802308 kB
<!--PRED 2	npu	memavail_kb=31801456	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3551,3547,3891,3199,2266,1791,1352,1135,992,867,3865	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | ngl | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |   0 |          pp2048 |         30.22 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 2	npu	wall_s=172	busy=6023,5681,5482,3926,11272,10729,10131,9869	busy_tot=63113	busy_little_share=0.3345	a55_cpu_cycles=380479974224	a55_inst_retired=261802213346	context_switches=1584344	a76_cpu_cycles=896651859894	a76_l3d_cache_refill=5820014370	a76_l2d_cache_refill=2523849663	cpu_migrations=24621	page_faults=7485904	a76_dtlb_walk=264246996	a76_mem_access=562546603666	a76_inst_retired=2106749095420	a76_l1d_cache_refill=9614896103	a55_inst_share=0.1105	a76_ipc=2.350	l2ref_pki=1.198	l3ref_pki=2.763	l1dref_pki=4.564	memacc_pki=267.02	dtlbw_pki=0.1254	pmu_enabled=100.0	pmu_cpu_s=1373.9	who=at_s=8,rss_pg=3156859,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.7506	l2color_cv=0.0092	l3color_cv=0.0184	contig_frac=0.8961	mean_run=9.5	vapa16=0.0661	gib_regions=29	anon_pg=6656	anon_l2cv=0.0210	anon_contig=0.8064	anon_run=5.1	anon_gib=25	file_pg=24576	file_l2cv=0.0012	file_contig=0.9960	file_run=169.5	file_gib=27	other_pg=24110	other_l2cv=0.0189	other_contig=0.8191	other_run=5.5	other_gib=29	ro_cost_ms=90-->
<!--DATA 2	npu	pp2048	30.22-->
    [moe-int8] experts exercised: 1668 resident on the NPU (13454MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24776MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM of which 8471MB is GGUF source, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24776MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gpt-oss-20b [cpu-rp0] pass 2  env=''  args='-b 2048 -ub 2048 --repack 0'  04:49:01  clk=200 MHz  MemAvail=31824236 kB
<!--PRED 2	cpu-rp0	memavail_kb=31823360	anonhuge_kb=0	hugepagesz_kb=2048	buddy=2959,3141,2809,2558,1657,1665,1258,994,867,750,4008	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CPU        |       8 |     2048 |   0 |          pp2048 |         11.99 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 2	cpu-rp0	wall_s=343	busy=29904,29731,27051,15715,32560,32410,32679,32441	busy_tot=232491	busy_little_share=0.4405	a55_cpu_cycles=1802629300430	a55_inst_retired=1278448686018	context_switches=109625	a76_cpu_cycles=2724237250427	a76_l3d_cache_refill=8439780094	a76_l2d_cache_refill=4868774212	cpu_migrations=10727	page_faults=197202	a76_dtlb_walk=524993800	a76_mem_access=1643122908485	a76_inst_retired=8420595242787	a76_l1d_cache_refill=8376712865	a55_inst_share=0.1318	a76_ipc=3.091	l2ref_pki=0.578	l3ref_pki=1.002	l1dref_pki=0.995	memacc_pki=195.13	dtlbw_pki=0.0623	pmu_enabled=100.0	pmu_cpu_s=2737.4	who=at_s=6,rss_pg=3068143,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=72704	present_frac=0.7147	l2color_cv=0.0052	l3color_cv=0.0159	contig_frac=0.8563	mean_run=6.8	vapa16=0.0158	gib_regions=28	anon_pg=4399	anon_l2cv=0.0267	anon_contig=0.4062	anon_run=1.7	anon_gib=24	file_pg=24576	file_l2cv=0.0004	file_contig=0.9985	file_run=285.8	file_gib=23	other_pg=22988	other_l2cv=0.0107	other_contig=0.7888	other_run=4.7	other_gib=28	ro_cost_ms=188-->
<!--DATA 2	cpu-rp0	pp2048	11.99-->

### gpt-oss-20b [npu] pass 3  env=''  args='-b 2048 -ub 2048 --repack 0'  04:56:17  clk=200 MHz  MemAvail=31797464 kB
<!--PRED 3	npu	memavail_kb=31797908	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3370,3163,3403,3178,2228,1832,1405,1184,1025,900,3832	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | ngl | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |   0 |          pp2048 |         30.37 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 3	npu	wall_s=172	busy=5920,5542,5477,4240,11250,10387,10452,9710	busy_tot=62978	busy_little_share=0.3363	a55_cpu_cycles=381501031353	a55_inst_retired=263107576366	context_switches=1583042	a76_cpu_cycles=893671879423	a76_l3d_cache_refill=5839020599	a76_l2d_cache_refill=2534857974	cpu_migrations=24742	page_faults=7487516	a76_dtlb_walk=264616268	a76_mem_access=562510123396	a76_inst_retired=2105512983629	a76_l1d_cache_refill=9620609322	a55_inst_share=0.1111	a76_ipc=2.356	l2ref_pki=1.204	l3ref_pki=2.773	l1dref_pki=4.569	memacc_pki=267.16	dtlbw_pki=0.1257	pmu_enabled=100.0	pmu_cpu_s=1371.4	who=at_s=8,rss_pg=3156935,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.7520	l2color_cv=0.0071	l3color_cv=0.0174	contig_frac=0.9003	mean_run=9.9	vapa16=0.0592	gib_regions=29	anon_pg=6656	anon_l2cv=0.0090	anon_contig=0.8005	anon_run=5.0	anon_gib=22	file_pg=24576	file_l2cv=0.0010	file_contig=0.9967	file_run=192.0	file_gib=24	other_pg=24208	other_l2cv=0.0161	other_contig=0.8299	other_run=5.8	other_gib=29	ro_cost_ms=101-->
<!--DATA 3	npu	pp2048	30.37-->
    [moe-int8] experts exercised: 1668 resident on the NPU (13454MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24770MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM of which 8471MB is GGUF source, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24770MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gpt-oss-20b [cpu-rp0] pass 3  env=''  args='-b 2048 -ub 2048 --repack 0'  05:00:50  clk=200 MHz  MemAvail=31808276 kB
<!--PRED 3	cpu-rp0	memavail_kb=31807944	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3229,3250,2212,2451,1895,1746,1340,1061,943,824,3931	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CPU        |       8 |     2048 |   0 |          pp2048 |         11.98 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 3	cpu-rp0	wall_s=344	busy=29789,30031,27875,13770,32282,32712,32688,32581	busy_tot=231728	busy_little_share=0.4379	a55_cpu_cycles=1786691826940	a55_inst_retired=1264973595288	context_switches=117029	a76_cpu_cycles=2728181056052	a76_l3d_cache_refill=8533531962	a76_l2d_cache_refill=4885523511	cpu_migrations=10991	page_faults=200663	a76_dtlb_walk=526967539	a76_mem_access=1645524826635	a76_inst_retired=8434455451029	a76_l1d_cache_refill=8386483516	a55_inst_share=0.1304	a76_ipc=3.092	l2ref_pki=0.579	l3ref_pki=1.012	l1dref_pki=0.994	memacc_pki=195.10	dtlbw_pki=0.0625	pmu_enabled=100.0	pmu_cpu_s=2746.6	who=at_s=6,rss_pg=3067719,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=72192	present_frac=0.7111	l2color_cv=0.0045	l3color_cv=0.0128	contig_frac=0.8521	mean_run=6.6	vapa16=0.0094	gib_regions=28	anon_pg=4332	anon_l2cv=0.0245	anon_contig=0.4397	anon_run=1.8	anon_gib=26	file_pg=24576	file_l2cv=0.0018	file_contig=0.9924	file_run=104.6	file_gib=27	other_pg=22426	other_l2cv=0.0084	other_contig=0.7764	other_run=4.4	other_gib=27	ro_cost_ms=83-->
<!--DATA 3	cpu-rp0	pp2048	11.98-->

### gpt-oss-20b [cpu-rp1] pass 3  env=''  args='-b 2048 -ub 2048 --repack 1'  05:08:07  clk=200 MHz  MemAvail=31815860 kB
<!--PRED 3	cpu-rp1	memavail_kb=31815860	anonhuge_kb=0	hugepagesz_kb=2048	buddy=2879,2806,2935,2375,1725,1576,1354,1021,899,782,3976	pinned=1	cpufreq=policy0:performance:1008000-1800000,policy4:performance:1200000-2400000,policy6:performance:1200000-2400000-->
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CPU        |       8 |     2048 |          pp2048 |         15.93 ± 0.00 |

build: 526c43b8f (11242)
<!--RO 3	cpu-rp1	wall_s=268	busy=16075,16155,14725,10866,24097,24023,23476,24577	busy_tot=153994	busy_little_share=0.3755	a55_cpu_cycles=1014493021494	a55_inst_retired=863990693387	context_switches=71592	a76_cpu_cycles=2014593799500	a76_l3d_cache_refill=3285904785	a76_l2d_cache_refill=1318678477	cpu_migrations=9765	page_faults=2969374	a76_dtlb_walk=99640065	a76_mem_access=985505811549	a76_inst_retired=6019359350540	a76_l1d_cache_refill=7263651817	a55_inst_share=0.1255	a76_ipc=2.988	l2ref_pki=0.219	l3ref_pki=0.546	l1dref_pki=1.207	memacc_pki=163.72	dtlbw_pki=0.0166	pmu_enabled=100.0	pmu_cpu_s=2141.2	who=at_s=12,rss_pg=5746407,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=97280	present_frac=0.8182	l2color_cv=0.0036	l3color_cv=0.0097	contig_frac=0.9080	mean_run=10.7	vapa16=0.0476	gib_regions=28	anon_pg=32278	anon_l2cv=0.0077	anon_contig=0.9170	anon_run=11.8	anon_gib=28	file_pg=24576	file_l2cv=0.0002	file_contig=0.9994	file_run=390.1	file_gib=21	other_pg=22738	other_l2cv=0.0114	other_contig=0.7964	other_run=4.9	other_gib=27	ro_cost_ms=293-->
<!--DATA 3	cpu-rp1	pp2048	15.93-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [cpu-rp0]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | cpu-rp0 | 11.97 | 3 | 11.94 | 11.99 | -- | -- |
| pp2048 | cpu-rp1 | 15.91 | 3 | 15.85 | 15.95 | 1.329x | 1.327 1.330 1.330 |
| pp2048 | npu | 30.20 | 3 | 30.00 | 30.37 | 2.523x | 2.513 2.520 2.535 |

