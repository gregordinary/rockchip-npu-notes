<!-- gpt-oss-20b MXFP4, ROCKET_N_THREADS ladder across the pre-flight binder crossover
     TESTS='-p 2048 -n 0 -r 3'  PASSES=3  args='-b 2048 -ub 2048' on every arm
     gguf=<data>/gpt-oss-20b/gpt-oss-20b-mxfp4.gguf (12109566560 bytes)
     both arms are RAM-bound: the pre-flight announces Bound by RAM at 62 stacks either way
     ratios are paired WITHIN a pass against nt5
-->
== gptoss-20b-nthreads-ladder  Wed Sep  2 04:13:35 UTC 2026 ==
### gptoss-20b-nthreads-ladder [nt5] pass 1  env=''  args='-b 2048 -ub 2048'  04:15:09  clk=600 MHz  MemAvail=31647252 kB
<!--PRED 1	nt5	memavail_kb=31647268	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9108,11201,11673,11470,8035,7554,6654,5708,4517,2425,878-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.76 ± 0.21 |

build: 171974745 (10558)
<!--RO 1	nt5	wall_s=322	busy=13116,12397,11607,8909,22795,20166,19964,19124	busy_tot=128078	busy_little_share=0.3594	a55_cpu_cycles=824591605792	a55_inst_retired=568508801365	context_switches=3147954	a76_cpu_cycles=1749992221739	a76_l3d_cache_refill=10397633014	a76_l2d_cache_refill=5325989033	cpu_migrations=46361	page_faults=7716703	a76_dtlb_walk=730538918	a76_mem_access=1062066397852	a76_inst_retired=4193830994203	a76_l1d_cache_refill=18692569649	a55_inst_share=0.1194	a76_ipc=2.396	l2ref_pki=1.270	l3ref_pki=2.479	l1dref_pki=4.457	memacc_pki=253.24	dtlbw_pki=0.1742	pmu_enabled=100.0	pmu_cpu_s=2570.8	who=at_s=8,rss_pg=3154963,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.7516	l2color_cv=0.0124	l3color_cv=0.0259	contig_frac=0.8798	mean_run=8.2	vapa16=0.0597	gib_regions=29	anon_pg=6656	anon_l2cv=0.0223	anon_contig=0.8040	anon_run=5.1	anon_gib=26	file_pg=24576	file_l2cv=0.0033	file_contig=0.9948	file_run=139.6	file_gib=25	other_pg=24180	other_l2cv=0.0279	other_contig=0.7839	other_run=4.6	other_gib=29	ro_cost_ms=90-->
<!--DATA 1	nt5	pp2048	28.76-->
    [moe-int8] experts exercised: 1710 resident on the NPU (13793MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24621MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24621MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gptoss-20b-nthreads-ladder [nt8] pass 1  env='ROCKET_N_THREADS=8'  args='-b 2048 -ub 2048'  04:22:06  clk=600 MHz  MemAvail=31625352 kB
<!--PRED 1	nt8	memavail_kb=31625352	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9101,11478,12124,11127,8481,7726,6859,5862,4646,2491,770-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         29.59 ± 0.37 |

build: 171974745 (10558)
<!--RO 1	nt8	wall_s=314	busy=13040,12618,11652,9384,23021,22616,22504,21081	busy_tot=135916	busy_little_share=0.3436	a55_cpu_cycles=839788134520	a55_inst_retired=574752288752	context_switches=3399897	a76_cpu_cycles=1892322898087	a76_l3d_cache_refill=10719211421	a76_l2d_cache_refill=5619893703	cpu_migrations=72237	page_faults=8035766	a76_dtlb_walk=989993658	a76_mem_access=1107738027867	a76_inst_retired=4376471569875	a76_l1d_cache_refill=19594342446	a55_inst_share=0.1161	a76_ipc=2.313	l2ref_pki=1.284	l3ref_pki=2.449	l1dref_pki=4.477	memacc_pki=253.11	dtlbw_pki=0.2262	pmu_enabled=100.0	pmu_cpu_s=2505.6	who=at_s=6,rss_pg=3156549,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.7508	l2color_cv=0.0079	l3color_cv=0.0194	contig_frac=0.8657	mean_run=7.3	vapa16=0.0162	gib_regions=29	anon_pg=6656	anon_l2cv=0.0083	anon_contig=0.7992	anon_run=4.9	anon_gib=25	file_pg=24576	file_l2cv=0.0021	file_contig=0.9949	file_run=142.1	file_gib=26	other_pg=24126	other_l2cv=0.0177	other_contig=0.7523	other_run=4.0	other_gib=29	ro_cost_ms=84-->
<!--DATA 1	nt8	pp2048	29.59-->
    [moe-int8] experts exercised: 1713 resident on the NPU (13978MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24622MB RAM budget, 30720MB NPU IOVA across 8 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24622MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gptoss-20b-nthreads-ladder [nt8] pass 2  env='ROCKET_N_THREADS=8'  args='-b 2048 -ub 2048'  04:28:55  clk=600 MHz  MemAvail=31610428 kB
<!--PRED 2	nt8	memavail_kb=31610428	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10686,11259,13157,12352,8568,7945,6913,5721,4615,2484,769-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         29.72 ± 0.36 |

build: 171974745 (10558)
<!--RO 2	nt8	wall_s=312	busy=12631,12575,11975,9114,23025,23100,22094,21071	busy_tot=135585	busy_little_share=0.3414	a55_cpu_cycles=833094537742	a55_inst_retired=570247969051	context_switches=3398671	a76_cpu_cycles=1892800190999	a76_l3d_cache_refill=10746962866	a76_l2d_cache_refill=5622774598	cpu_migrations=72920	page_faults=8040606	a76_dtlb_walk=995686270	a76_mem_access=1109099801227	a76_inst_retired=4381983266926	a76_l1d_cache_refill=19612350795	a55_inst_share=0.1151	a76_ipc=2.315	l2ref_pki=1.283	l3ref_pki=2.453	l1dref_pki=4.476	memacc_pki=253.10	dtlbw_pki=0.2272	pmu_enabled=100.0	pmu_cpu_s=2496.0	who=at_s=6,rss_pg=3156534,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.7484	l2color_cv=0.0078	l3color_cv=0.0180	contig_frac=0.8872	mean_run=8.7	vapa16=0.0152	gib_regions=27	anon_pg=6656	anon_l2cv=0.0092	anon_contig=0.7963	anon_run=4.9	anon_gib=26	file_pg=24576	file_l2cv=0.0020	file_contig=0.9915	file_run=96.0	file_gib=24	other_pg=23947	other_l2cv=0.0183	other_contig=0.8054	other_run=5.1	other_gib=26	ro_cost_ms=83-->
<!--DATA 2	nt8	pp2048	29.72-->
    [moe-int8] experts exercised: 1713 resident on the NPU (13978MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24597MB RAM budget, 30720MB NPU IOVA across 8 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24597MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gptoss-20b-nthreads-ladder [nt5] pass 2  env=''  args='-b 2048 -ub 2048'  04:35:43  clk=600 MHz  MemAvail=31646104 kB
<!--PRED 2	nt5	memavail_kb=31646104	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9284,11476,11927,11513,8519,7670,6742,5655,4631,2352,882-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.89 ± 0.33 |

build: 171974745 (10558)
<!--RO 2	nt5	wall_s=321	busy=13197,12376,12043,8543,22833,19796,20446,18994	busy_tot=128228	busy_little_share=0.3600	a55_cpu_cycles=826912652446	a55_inst_retired=571341896456	context_switches=3145362	a76_cpu_cycles=1747323947685	a76_l3d_cache_refill=10368824739	a76_l2d_cache_refill=5299941342	cpu_migrations=46497	page_faults=7692422	a76_dtlb_walk=729771744	a76_mem_access=1061028956350	a76_inst_retired=4188617189232	a76_l1d_cache_refill=18649110411	a55_inst_share=0.1200	a76_ipc=2.397	l2ref_pki=1.265	l3ref_pki=2.475	l1dref_pki=4.452	memacc_pki=253.31	dtlbw_pki=0.1742	pmu_enabled=100.0	pmu_cpu_s=2560.5	who=at_s=8,rss_pg=3155550,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.7472	l2color_cv=0.0074	l3color_cv=0.0239	contig_frac=0.8860	mean_run=8.6	vapa16=0.0391	gib_regions=29	anon_pg=6656	anon_l2cv=0.0168	anon_contig=0.8051	anon_run=5.1	anon_gib=26	file_pg=24576	file_l2cv=0.0012	file_contig=0.9951	file_run=146.3	file_gib=24	other_pg=23861	other_l2cv=0.0190	other_contig=0.7961	other_run=4.9	other_gib=28	ro_cost_ms=86-->
<!--DATA 2	nt5	pp2048	28.89-->
    [moe-int8] experts exercised: 1710 resident on the NPU (13793MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24621MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24621MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gptoss-20b-nthreads-ladder [nt5] pass 3  env=''  args='-b 2048 -ub 2048'  04:42:39  clk=600 MHz  MemAvail=31647824 kB
<!--PRED 3	nt5	memavail_kb=31647824	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8516,10700,11070,11032,8563,7474,6642,5672,4572,2408,888-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.75 ± 0.13 |

build: 171974745 (10558)
<!--RO 3	nt5	wall_s=321	busy=13149,12265,11835,9241,22601,19397,20419,19636	busy_tot=128543	busy_little_share=0.3617	a55_cpu_cycles=831227282370	a55_inst_retired=573819533639	context_switches=3139554	a76_cpu_cycles=1747405809379	a76_l3d_cache_refill=10377766484	a76_l2d_cache_refill=5322060860	cpu_migrations=46645	page_faults=7688723	a76_dtlb_walk=733053970	a76_mem_access=1060686622377	a76_inst_retired=4185530284289	a76_l1d_cache_refill=18687084197	a55_inst_share=0.1206	a76_ipc=2.395	l2ref_pki=1.272	l3ref_pki=2.479	l1dref_pki=4.465	memacc_pki=253.42	dtlbw_pki=0.1751	pmu_enabled=100.0	pmu_cpu_s=2565.8	who=at_s=8,rss_pg=3155547,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.7475	l2color_cv=0.0077	l3color_cv=0.0203	contig_frac=0.8866	mean_run=8.7	vapa16=0.0103	gib_regions=28	anon_pg=6658	anon_l2cv=0.0135	anon_contig=0.7989	anon_run=4.9	anon_gib=26	file_pg=24576	file_l2cv=0.0014	file_contig=0.9969	file_run=199.8	file_gib=25	other_pg=23876	other_l2cv=0.0171	other_contig=0.7975	other_run=4.9	other_gib=27	ro_cost_ms=68-->
<!--DATA 3	nt5	pp2048	28.75-->
    [moe-int8] experts exercised: 1710 resident on the NPU (13793MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24617MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24617MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gptoss-20b-nthreads-ladder [nt8] pass 3  env='ROCKET_N_THREADS=8'  args='-b 2048 -ub 2048'  04:49:35  clk=600 MHz  MemAvail=31620620 kB
<!--PRED 3	nt8	memavail_kb=31620848	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8663,11408,12968,12104,8830,7824,6890,5858,4627,2455,771-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         29.46 ± 0.34 |

build: 171974745 (10558)
<!--RO 3	nt8	wall_s=315	busy=12954,11967,11435,8772,23170,22601,22668,21076	busy_tot=134643	busy_little_share=0.3352	a55_cpu_cycles=813395136866	a55_inst_retired=552372681168	context_switches=3404740	a76_cpu_cycles=1897477970395	a76_l3d_cache_refill=10776505447	a76_l2d_cache_refill=5639258828	cpu_migrations=72567	page_faults=8033344	a76_dtlb_walk=990782786	a76_mem_access=1112227755915	a76_inst_retired=4399256301840	a76_l1d_cache_refill=19610525635	a55_inst_share=0.1116	a76_ipc=2.318	l2ref_pki=1.282	l3ref_pki=2.450	l1dref_pki=4.458	memacc_pki=252.82	dtlbw_pki=0.2252	pmu_enabled=100.0	pmu_cpu_s=2511.4	who=at_s=8,rss_pg=3156918,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.7489	l2color_cv=0.0095	l3color_cv=0.0236	contig_frac=0.8853	mean_run=8.6	vapa16=0.0334	gib_regions=28	anon_pg=6656	anon_l2cv=0.0334	anon_contig=0.8063	anon_run=5.1	anon_gib=25	file_pg=24576	file_l2cv=0.0011	file_contig=0.9949	file_run=141.2	file_gib=26	other_pg=23985	other_l2cv=0.0179	other_contig=0.7949	other_run=4.8	other_gib=28	ro_cost_ms=94-->
<!--DATA 3	nt8	pp2048	29.46-->
    [moe-int8] experts exercised: 1713 resident on the NPU (13978MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24595MB RAM budget, 30720MB NPU IOVA across 8 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24595MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [nt5]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | nt5 | 28.80 | 3 | 28.75 | 28.89 | -- | -- |
| pp2048 | nt8 | 29.59 | 3 | 29.46 | 29.72 | 1.027x | 1.029 1.029 1.025 |

