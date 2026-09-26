<!-- gpt-oss-20b MXFP4, the ROCKET_N_THREADS curve between the two measured ends
     TESTS='-p 2048 -n 0 -r 3'  PASSES=3  args='-b 2048 -ub 2048' on every arm
     gguf=<data>/gpt-oss-20b/gpt-oss-20b-mxfp4.gguf (12109566560 bytes)
     every arm is RAM-bound: the pre-flight announces Bound by RAM at 63 stacks at 5 and 8
     ratios are paired WITHIN a pass against nt5; arm order rotated by one each pass
-->
== gptoss-20b-nthreads-shape  Wed Sep  2 16:54:30 UTC 2026 ==
### gptoss-20b-nthreads-shape [nt5] pass 1  env=''  args='-b 2048 -ub 2048'  16:56:04  clk=600 MHz  MemAvail=31616856 kB
<!--PRED 1	nt5	memavail_kb=31617164	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10315,10044,10608,10137,8409,7313,6476,5478,4443,2430,944-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         29.09 ± 0.17 |

build: 171974745 (10558)
<!--RO 1	nt5	wall_s=319	busy=13028,12060,11807,9368,22728,20055,20119,19060	busy_tot=128225	busy_little_share=0.3608	a55_cpu_cycles=828816886376	a55_inst_retired=573830586997	context_switches=3144238	a76_cpu_cycles=1746556370946	a76_l3d_cache_refill=10376308399	a76_l2d_cache_refill=5236295921	cpu_migrations=46484	page_faults=7701884	a76_dtlb_walk=731285578	a76_mem_access=1060318401642	a76_inst_retired=4186141274075	a76_l1d_cache_refill=18689046720	a55_inst_share=0.1206	a76_ipc=2.397	l2ref_pki=1.251	l3ref_pki=2.479	l1dref_pki=4.465	memacc_pki=253.29	dtlbw_pki=0.1747	pmu_enabled=100.0	pmu_cpu_s=2547.1	who=at_s=8,rss_pg=3155550,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.7516	l2color_cv=0.0109	l3color_cv=0.0235	contig_frac=0.8904	mean_run=9.0	vapa16=0.0145	gib_regions=26	anon_pg=6656	anon_l2cv=0.0183	anon_contig=0.8031	anon_run=5.0	anon_gib=25	file_pg=24576	file_l2cv=0.0011	file_contig=0.9956	file_run=157.5	file_gib=25	other_pg=24185	other_l2cv=0.0229	other_contig=0.8074	other_run=5.1	other_gib=26	ro_cost_ms=66-->
<!--DATA 1	nt5	pp2048	29.09-->
    [moe-int8] experts exercised: 1710 resident on the NPU (13793MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24594MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24594MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gptoss-20b-nthreads-shape [nt6] pass 1  env='ROCKET_N_THREADS=6'  args='-b 2048 -ub 2048'  17:02:59  clk=600 MHz  MemAvail=31640460 kB
<!--PRED 1	nt6	memavail_kb=31640680	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8802,10748,11167,10667,8705,7455,6642,5658,4469,2612,803-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         29.68 ± 0.22 |

build: 171974745 (10558)
<!--RO 1	nt6	wall_s=315	busy=12656,11909,11722,8513,23319,22515,20455,19633	busy_tot=130722	busy_little_share=0.3427	a55_cpu_cycles=804274053273	a55_inst_retired=554760842313	context_switches=2904409	a76_cpu_cycles=1825053055459	a76_l3d_cache_refill=10866282389	a76_l2d_cache_refill=5430853559	cpu_migrations=54780	page_faults=8294535	a76_dtlb_walk=982934396	a76_mem_access=1080886072714	a76_inst_retired=4313454410026	a76_l1d_cache_refill=19424997482	a55_inst_share=0.1140	a76_ipc=2.363	l2ref_pki=1.259	l3ref_pki=2.519	l1dref_pki=4.503	memacc_pki=250.58	dtlbw_pki=0.2279	pmu_enabled=100.0	pmu_cpu_s=2511.6	who=at_s=6,rss_pg=3153646,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.7504	l2color_cv=0.0107	l3color_cv=0.0221	contig_frac=0.8879	mean_run=8.8	vapa16=0.0257	gib_regions=29	anon_pg=6658	anon_l2cv=0.0101	anon_contig=0.8078	anon_run=5.2	anon_gib=24	file_pg=24576	file_l2cv=0.0009	file_contig=0.9969	file_run=198.2	file_gib=23	other_pg=24090	other_l2cv=0.0241	other_contig=0.7989	other_run=4.9	other_gib=28	ro_cost_ms=96-->
<!--DATA 1	nt6	pp2048	29.68-->
    [moe-int8] experts exercised: 1713 resident on the NPU (14774MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24617MB RAM budget, 23040MB NPU IOVA across 6 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24617MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gptoss-20b-nthreads-shape [nt7] pass 1  env='ROCKET_N_THREADS=7'  args='-b 2048 -ub 2048'  17:09:50  clk=600 MHz  MemAvail=31645488 kB
<!--PRED 1	nt7	memavail_kb=31645488	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9922,10688,11384,10875,8477,7493,6656,5690,4428,2599,816-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         29.46 ± 0.36 |

build: 171974745 (10558)
<!--RO 1	nt7	wall_s=316	busy=13007,11598,11757,9966,22447,22305,22808,20310	busy_tot=134198	busy_little_share=0.3452	a55_cpu_cycles=831441708292	a55_inst_retired=570860950633	context_switches=3247309	a76_cpu_cycles=1865247643312	a76_l3d_cache_refill=10773517943	a76_l2d_cache_refill=5588270165	cpu_migrations=62611	page_faults=8384053	a76_dtlb_walk=981626874	a76_mem_access=1092581971102	a76_inst_retired=4343122060014	a76_l1d_cache_refill=20003919212	a55_inst_share=0.1162	a76_ipc=2.328	l2ref_pki=1.287	l3ref_pki=2.481	l1dref_pki=4.606	memacc_pki=251.57	dtlbw_pki=0.2260	pmu_enabled=100.0	pmu_cpu_s=2519.8	who=at_s=6,rss_pg=3153698,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.7480	l2color_cv=0.0085	l3color_cv=0.0202	contig_frac=0.8872	mean_run=8.7	vapa16=0.0090	gib_regions=28	anon_pg=6663	anon_l2cv=0.0124	anon_contig=0.8076	anon_run=5.2	anon_gib=25	file_pg=24576	file_l2cv=0.0004	file_contig=0.9972	file_run=210.1	file_gib=25	other_pg=23909	other_l2cv=0.0192	other_contig=0.7962	other_run=4.9	other_gib=27	ro_cost_ms=86-->
<!--DATA 1	nt7	pp2048	29.46-->
    [moe-int8] experts exercised: 1710 resident on the NPU (14802MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24622MB RAM budget, 26880MB NPU IOVA across 7 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24622MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gptoss-20b-nthreads-shape [nt8] pass 1  env='ROCKET_N_THREADS=8'  args='-b 2048 -ub 2048'  17:16:41  clk=600 MHz  MemAvail=31568268 kB
<!--PRED 1	nt8	memavail_kb=31568524	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10925,10182,12289,11520,8945,7884,6940,5890,4460,2524,755-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         29.55 ± 0.35 |

build: 171974745 (10558)
<!--RO 1	nt8	wall_s=314	busy=13026,12288,11867,9409,23173,22680,22170,21179	busy_tot=135792	busy_little_share=0.3431	a55_cpu_cycles=838869961274	a55_inst_retired=573837561158	context_switches=3402277	a76_cpu_cycles=1891769735128	a76_l3d_cache_refill=10747587341	a76_l2d_cache_refill=5630620857	cpu_migrations=71939	page_faults=8036443	a76_dtlb_walk=986863048	a76_mem_access=1108137824169	a76_inst_retired=4377304998377	a76_l1d_cache_refill=19593595179	a55_inst_share=0.1159	a76_ipc=2.314	l2ref_pki=1.286	l3ref_pki=2.455	l1dref_pki=4.476	memacc_pki=253.16	dtlbw_pki=0.2254	pmu_enabled=100.0	pmu_cpu_s=2506.1	who=at_s=8,rss_pg=3156946,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.7475	l2color_cv=0.0116	l3color_cv=0.0185	contig_frac=0.8933	mean_run=9.2	vapa16=0.0981	gib_regions=29	anon_pg=6656	anon_l2cv=0.0117	anon_contig=0.7918	anon_run=4.8	anon_gib=27	file_pg=24576	file_l2cv=0.0009	file_contig=0.9960	file_run=168.3	file_gib=25	other_pg=23876	other_l2cv=0.0252	other_contig=0.8159	other_run=5.4	other_gib=29	ro_cost_ms=101-->
<!--DATA 1	nt8	pp2048	29.55-->
    [moe-int8] experts exercised: 1713 resident on the NPU (13978MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24562MB RAM budget, 30720MB NPU IOVA across 8 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24562MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gptoss-20b-nthreads-shape [nt6] pass 2  env='ROCKET_N_THREADS=6'  args='-b 2048 -ub 2048'  17:23:31  clk=600 MHz  MemAvail=31647440 kB
<!--PRED 2	nt6	memavail_kb=31647440	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9834,10202,10938,10798,8553,7510,6560,5573,4448,2591,837-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         29.41 ± 0.28 |

build: 171974745 (10558)
<!--RO 2	nt6	wall_s=317	busy=13019,12348,10971,9369,22295,22579,21075,19746	busy_tot=131402	busy_little_share=0.3478	a55_cpu_cycles=819831555404	a55_inst_retired=565973948300	context_switches=2898826	a76_cpu_cycles=1821683043710	a76_l3d_cache_refill=10847893922	a76_l2d_cache_refill=5425062527	cpu_migrations=54809	page_faults=8289231	a76_dtlb_walk=981807027	a76_mem_access=1078571454540	a76_inst_retired=4301789907676	a76_l1d_cache_refill=19406008188	a55_inst_share=0.1163	a76_ipc=2.361	l2ref_pki=1.261	l3ref_pki=2.522	l1dref_pki=4.511	memacc_pki=250.73	dtlbw_pki=0.2282	pmu_enabled=100.0	pmu_cpu_s=2526.5	who=at_s=6,rss_pg=3153661,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.7470	l2color_cv=0.0078	l3color_cv=0.0181	contig_frac=0.8769	mean_run=8.0	vapa16=0.0462	gib_regions=28	anon_pg=6656	anon_l2cv=0.0104	anon_contig=0.8091	anon_run=5.2	anon_gib=25	file_pg=24576	file_l2cv=0.0036	file_contig=0.9945	file_run=135.0	file_gib=25	other_pg=23841	other_l2cv=0.0175	other_contig=0.7746	other_run=4.4	other_gib=27	ro_cost_ms=80-->
<!--DATA 2	nt6	pp2048	29.41-->
    [moe-int8] experts exercised: 1713 resident on the NPU (14774MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24630MB RAM budget, 23040MB NPU IOVA across 6 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24630MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gptoss-20b-nthreads-shape [nt7] pass 2  env='ROCKET_N_THREADS=7'  args='-b 2048 -ub 2048'  17:30:24  clk=600 MHz  MemAvail=31638292 kB
<!--PRED 2	nt7	memavail_kb=31638292	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9017,11038,11638,11152,8687,7657,6683,5718,4418,2635,782-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         29.38 ± 0.31 |

build: 171974745 (10558)
<!--RO 2	nt7	wall_s=317	busy=13046,12091,11479,9294,22399,22207,22761,20394	busy_tot=133671	busy_little_share=0.3435	a55_cpu_cycles=825083454140	a55_inst_retired=564970547985	context_switches=3245271	a76_cpu_cycles=1865922798295	a76_l3d_cache_refill=10787109872	a76_l2d_cache_refill=5595318514	cpu_migrations=62597	page_faults=8382861	a76_dtlb_walk=984810313	a76_mem_access=1094201286815	a76_inst_retired=4348910170277	a76_l1d_cache_refill=20006735987	a55_inst_share=0.1150	a76_ipc=2.331	l2ref_pki=1.287	l3ref_pki=2.480	l1dref_pki=4.600	memacc_pki=251.60	dtlbw_pki=0.2264	pmu_enabled=100.0	pmu_cpu_s=2530.2	who=at_s=6,rss_pg=3153698,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.7467	l2color_cv=0.0093	l3color_cv=0.0159	contig_frac=0.8868	mean_run=8.7	vapa16=0.3848	gib_regions=28	anon_pg=6591	anon_l2cv=0.0112	anon_contig=0.8060	anon_run=5.1	anon_gib=24	file_pg=24576	file_l2cv=0.0004	file_contig=0.9967	file_run=190.5	file_gib=25	other_pg=23885	other_l2cv=0.0213	other_contig=0.7961	other_run=4.9	other_gib=28	ro_cost_ms=78-->
<!--DATA 2	nt7	pp2048	29.38-->
    [moe-int8] experts exercised: 1710 resident on the NPU (14802MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24615MB RAM budget, 26880MB NPU IOVA across 7 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24615MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gptoss-20b-nthreads-shape [nt8] pass 2  env='ROCKET_N_THREADS=8'  args='-b 2048 -ub 2048'  17:37:16  clk=600 MHz  MemAvail=31588472 kB
<!--PRED 2	nt8	memavail_kb=31588472	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9937,10322,13055,12184,8882,7851,6925,5839,4450,2526,763-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         29.63 ± 0.30 |

build: 171974745 (10558)
<!--RO 2	nt8	wall_s=313	busy=12866,12063,11794,9248,23652,22301,22104,21133	busy_tot=135161	busy_little_share=0.3401	a55_cpu_cycles=827580199168	a55_inst_retired=566164784549	context_switches=3399550	a76_cpu_cycles=1892122457458	a76_l3d_cache_refill=10749328677	a76_l2d_cache_refill=5633512457	cpu_migrations=72664	page_faults=8037067	a76_dtlb_walk=989839286	a76_mem_access=1109700692941	a76_inst_retired=4385587759517	a76_l1d_cache_refill=19612485971	a55_inst_share=0.1143	a76_ipc=2.318	l2ref_pki=1.285	l3ref_pki=2.451	l1dref_pki=4.472	memacc_pki=253.03	dtlbw_pki=0.2257	pmu_enabled=100.0	pmu_cpu_s=2501.1	who=at_s=6,rss_pg=3156586,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.7482	l2color_cv=0.0105	l3color_cv=0.0220	contig_frac=0.8956	mean_run=9.4	vapa16=0.0136	gib_regions=29	anon_pg=6610	anon_l2cv=0.0105	anon_contig=0.8025	anon_run=5.0	anon_gib=26	file_pg=24576	file_l2cv=0.0014	file_contig=0.9956	file_run=158.6	file_gib=25	other_pg=23976	other_l2cv=0.0238	other_contig=0.8187	other_run=5.5	other_gib=28	ro_cost_ms=69-->
<!--DATA 2	nt8	pp2048	29.63-->
    [moe-int8] experts exercised: 1713 resident on the NPU (13978MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24588MB RAM budget, 30720MB NPU IOVA across 8 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24588MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gptoss-20b-nthreads-shape [nt5] pass 2  env=''  args='-b 2048 -ub 2048'  17:44:05  clk=600 MHz  MemAvail=31629380 kB
<!--PRED 2	nt5	memavail_kb=31629088	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8225,10631,11683,11379,8520,7634,6713,5577,4442,2478,870-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.91 ± 0.21 |

build: 171974745 (10558)
<!--RO 2	nt5	wall_s=320	busy=13082,12071,11215,9746,23174,20279,19842,18962	busy_tot=128371	busy_little_share=0.3592	a55_cpu_cycles=826619946335	a55_inst_retired=571124572517	context_switches=3145342	a76_cpu_cycles=1750391160287	a76_l3d_cache_refill=10413844995	a76_l2d_cache_refill=5272856951	cpu_migrations=46197	page_faults=7690459	a76_dtlb_walk=735599650	a76_mem_access=1060875119691	a76_inst_retired=4188365903779	a76_l1d_cache_refill=18725022147	a55_inst_share=0.1200	a76_ipc=2.393	l2ref_pki=1.259	l3ref_pki=2.486	l1dref_pki=4.471	memacc_pki=253.29	dtlbw_pki=0.1756	pmu_enabled=100.0	pmu_cpu_s=2559.0	who=at_s=8,rss_pg=3155550,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.7525	l2color_cv=0.0120	l3color_cv=0.0239	contig_frac=0.8799	mean_run=8.2	vapa16=0.0778	gib_regions=29	anon_pg=6656	anon_l2cv=0.0140	anon_contig=0.8109	anon_run=5.2	anon_gib=26	file_pg=24576	file_l2cv=0.0023	file_contig=0.9909	file_run=91.0	file_gib=26	other_pg=24245	other_l2cv=0.0281	other_contig=0.7864	other_run=4.6	other_gib=29	ro_cost_ms=91-->
<!--DATA 2	nt5	pp2048	28.91-->
    [moe-int8] experts exercised: 1710 resident on the NPU (13793MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24607MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24607MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gptoss-20b-nthreads-shape [nt7] pass 3  env='ROCKET_N_THREADS=7'  args='-b 2048 -ub 2048'  17:51:01  clk=600 MHz  MemAvail=31635620 kB
<!--PRED 3	nt7	memavail_kb=31635620	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8909,10995,11264,10992,8676,7657,6772,5766,4556,2548,782-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         29.39 ± 0.44 |

build: 171974745 (10558)
<!--RO 3	nt7	wall_s=317	busy=12856,12529,11948,8324,23185,22294,22415,20291	busy_tot=133842	busy_little_share=0.3411	a55_cpu_cycles=820548747731	a55_inst_retired=561877796888	context_switches=3255095	a76_cpu_cycles=1871970450826	a76_l3d_cache_refill=10741321424	a76_l2d_cache_refill=5592095348	cpu_migrations=62573	page_faults=8381849	a76_dtlb_walk=982841032	a76_mem_access=1094026748420	a76_inst_retired=4351970638316	a76_l1d_cache_refill=19999785312	a55_inst_share=0.1143	a76_ipc=2.325	l2ref_pki=1.285	l3ref_pki=2.468	l1dref_pki=4.596	memacc_pki=251.39	dtlbw_pki=0.2258	pmu_enabled=100.0	pmu_cpu_s=2528.8	who=at_s=6,rss_pg=3153696,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.7433	l2color_cv=0.0063	l3color_cv=0.0170	contig_frac=0.8851	mean_run=8.6	vapa16=0.0502	gib_regions=29	anon_pg=6658	anon_l2cv=0.0156	anon_contig=0.7947	anon_run=4.8	anon_gib=25	file_pg=24576	file_l2cv=0.0010	file_contig=0.9965	file_run=183.4	file_gib=25	other_pg=23571	other_l2cv=0.0147	other_contig=0.7945	other_run=4.8	other_gib=28	ro_cost_ms=72-->
<!--DATA 3	nt7	pp2048	29.39-->
    [moe-int8] experts exercised: 1710 resident on the NPU (14802MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24610MB RAM budget, 26880MB NPU IOVA across 7 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24610MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gptoss-20b-nthreads-shape [nt8] pass 3  env='ROCKET_N_THREADS=8'  args='-b 2048 -ub 2048'  17:57:53  clk=600 MHz  MemAvail=31607496 kB
<!--PRED 3	nt8	memavail_kb=31607496	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9583,10403,12330,11423,9008,7810,6855,5815,4440,2533,783-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         29.22 ± 0.27 |

build: 171974745 (10558)
<!--RO 3	nt8	wall_s=316	busy=13115,12419,11485,9118,23431,22471,22149,21123	busy_tot=135311	busy_little_share=0.3410	a55_cpu_cycles=831754982181	a55_inst_retired=565295731641	context_switches=3407021	a76_cpu_cycles=1893468079105	a76_l3d_cache_refill=10750280965	a76_l2d_cache_refill=5634868515	cpu_migrations=72524	page_faults=8042162	a76_dtlb_walk=982908012	a76_mem_access=1109525143981	a76_inst_retired=4386306840961	a76_l1d_cache_refill=19597588992	a55_inst_share=0.1142	a76_ipc=2.317	l2ref_pki=1.285	l3ref_pki=2.451	l1dref_pki=4.468	memacc_pki=252.95	dtlbw_pki=0.2241	pmu_enabled=100.0	pmu_cpu_s=2523.1	who=at_s=6,rss_pg=3156561,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.7480	l2color_cv=0.0074	l3color_cv=0.0189	contig_frac=0.8767	mean_run=8.0	vapa16=0.3550	gib_regions=28	anon_pg=6656	anon_l2cv=0.0152	anon_contig=0.7894	anon_run=4.7	anon_gib=26	file_pg=24576	file_l2cv=0.0008	file_contig=0.9950	file_run=143.7	file_gib=25	other_pg=23913	other_l2cv=0.0171	other_contig=0.7794	other_run=4.5	other_gib=27	ro_cost_ms=69-->
<!--DATA 3	nt8	pp2048	29.22-->
    [moe-int8] experts exercised: 1713 resident on the NPU (13978MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24583MB RAM budget, 30720MB NPU IOVA across 8 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24583MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gptoss-20b-nthreads-shape [nt5] pass 3  env=''  args='-b 2048 -ub 2048'  18:04:44  clk=600 MHz  MemAvail=31633196 kB
<!--PRED 3	nt5	memavail_kb=31633196	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9980,11176,11619,11338,8420,7637,6685,5616,4466,2432,884-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.82 ± 0.45 |

build: 171974745 (10558)
<!--RO 3	nt5	wall_s=321	busy=13306,11961,11020,10604,22605,20019,20377,19061	busy_tot=128953	busy_little_share=0.3636	a55_cpu_cycles=838740351837	a55_inst_retired=579812185324	context_switches=3145148	a76_cpu_cycles=1746003299127	a76_l3d_cache_refill=10387233799	a76_l2d_cache_refill=5251695525	cpu_migrations=46624	page_faults=7690333	a76_dtlb_walk=728378041	a76_mem_access=1058993655255	a76_inst_retired=4179664520140	a76_l1d_cache_refill=18688703225	a55_inst_share=0.1218	a76_ipc=2.394	l2ref_pki=1.256	l3ref_pki=2.485	l1dref_pki=4.471	memacc_pki=253.37	dtlbw_pki=0.1743	pmu_enabled=100.0	pmu_cpu_s=2562.8	who=at_s=8,rss_pg=3155540,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.7473	l2color_cv=0.0085	l3color_cv=0.0217	contig_frac=0.8894	mean_run=8.9	vapa16=0.0238	gib_regions=29	anon_pg=6663	anon_l2cv=0.0199	anon_contig=0.7985	anon_run=4.9	anon_gib=25	file_pg=24576	file_l2cv=0.0005	file_contig=0.9973	file_run=215.6	file_gib=26	other_pg=23860	other_l2cv=0.0179	other_contig=0.8036	other_run=5.0	other_gib=29	ro_cost_ms=96-->
<!--DATA 3	nt5	pp2048	28.82-->
    [moe-int8] experts exercised: 1710 resident on the NPU (13793MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24615MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24615MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gptoss-20b-nthreads-shape [nt6] pass 3  env='ROCKET_N_THREADS=6'  args='-b 2048 -ub 2048'  18:11:42  clk=600 MHz  MemAvail=31650804 kB
<!--PRED 3	nt6	memavail_kb=31650512	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9705,10712,11237,11006,8524,7562,6713,5694,4573,2564,791-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         29.66 ± 0.20 |

build: 171974745 (10558)
<!--RO 3	nt6	wall_s=315	busy=12879,12208,11813,8396,23021,22478,20198,20107	busy_tot=131100	busy_little_share=0.3455	a55_cpu_cycles=812114142993	a55_inst_retired=561191282300	context_switches=2895545	a76_cpu_cycles=1822187027876	a76_l3d_cache_refill=10859340845	a76_l2d_cache_refill=5437066988	cpu_migrations=54719	page_faults=8289043	a76_dtlb_walk=973810044	a76_mem_access=1079968377121	a76_inst_retired=4307067355033	a76_l1d_cache_refill=19420417703	a55_inst_share=0.1153	a76_ipc=2.364	l2ref_pki=1.262	l3ref_pki=2.521	l1dref_pki=4.509	memacc_pki=250.74	dtlbw_pki=0.2261	pmu_enabled=100.0	pmu_cpu_s=2513.3	who=at_s=6,rss_pg=3153660,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.7445	l2color_cv=0.0089	l3color_cv=0.0182	contig_frac=0.8910	mean_run=9.0	vapa16=0.0961	gib_regions=29	anon_pg=6539	anon_l2cv=0.0171	anon_contig=0.8008	anon_run=5.0	anon_gib=26	file_pg=24576	file_l2cv=0.0005	file_contig=0.9965	file_run=182.0	file_gib=23	other_pg=23777	other_l2cv=0.0189	other_contig=0.8068	other_run=5.1	other_gib=29	ro_cost_ms=82-->
<!--DATA 3	nt6	pp2048	29.66-->
    [moe-int8] experts exercised: 1713 resident on the NPU (14774MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24632MB RAM budget, 23040MB NPU IOVA across 6 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24632MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [nt5]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | nt5 | 28.94 | 3 | 28.82 | 29.09 | -- | -- |
| pp2048 | nt6 | 29.58 | 3 | 29.41 | 29.68 | 1.022x | 1.020 1.017 1.029 |
| pp2048 | nt7 | 29.41 | 3 | 29.38 | 29.46 | 1.016x | 1.013 1.016 1.020 |
| pp2048 | nt8 | 29.47 | 3 | 29.22 | 29.63 | 1.018x | 1.016 1.025 1.014 |

