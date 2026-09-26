<!-- qwen35-9b Q4_K_M, pinning x quant-residency 2x2  TESTS='-p 2048 -n 0 -r 3'  PASSES=6
     gguf=<data>/qwen35/Qwen3.5-9B-Q4_K_M.gguf (5680522464 bytes)
     stock_unpin | stock_pin  = PIN_MASK=0xf0 -t 4
     qres_unpin  = ROCKET_QUANT_RESIDENT=auto -b 2048 -ub 2048   (the published 1.661x arm)
     qres_pin    = the same + PIN_MASK=0xf0 -t 4
     interaction = (qres_pin/stock_pin) / (qres_unpin/stock_unpin), paired within a pass
-->
== qwen35-9b-qres-pin2x2  Tue Sep  1 23:04:22 UTC 2026 ==
### qwen35-9b-qres-pin2x2 [stock_unpin] pass 1  env=''  args=''  23:05:29  clk=600 MHz  MemAvail=31507532 kB
<!--PRED 1	stock_unpin	memavail_kb=31507732	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7007,10097,10404,7054,5986,5504,4783,3965,2674,2777,3176-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         19.13 ± 0.19 |

build: 171974745 (10558)
<!--RO 1	stock_unpin	wall_s=432	busy=6935,5467,5431,5265,24646,22268,22106,22811	busy_tot=114929	busy_little_share=0.2010	a55_cpu_cycles=426088690358	a55_inst_retired=223735948196	context_switches=7960275	a76_cpu_cycles=1988654743547	a76_l3d_cache_refill=15953940185	a76_l2d_cache_refill=8220604424	cpu_migrations=40573	page_faults=598257	a76_dtlb_walk=2212143564	a76_mem_access=657216305617	a76_inst_retired=4128817877739	a76_l1d_cache_refill=13563915150	a55_inst_share=0.0514	a76_ipc=2.076	l2ref_pki=1.991	l3ref_pki=3.864	l1dref_pki=3.285	memacc_pki=159.18	dtlbw_pki=0.5358	pmu_enabled=100.0	pmu_cpu_s=3453.4	who=at_s=6,rss_pg=1509857,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6961	l2color_cv=0.0086	l3color_cv=0.0155	contig_frac=0.8272	mean_run=5.7	vapa16=0.0265	gib_regions=28	anon_pg=23007	anon_l2cv=0.0045	anon_contig=0.9045	anon_run=10.3	anon_gib=22	file_pg=24576	file_l2cv=0.0030	file_contig=0.9829	file_run=52.6	file_gib=27	other_pg=20842	other_l2cv=0.0260	other_contig=0.5582	other_run=2.3	other_gib=28	ro_cost_ms=125-->
<!--DATA 1	stock_unpin	pp2048	19.13-->

### qwen35-9b-qres-pin2x2 [stock_pin] pass 1  env='PIN_MASK=0xf0'  args='-t 4'  23:13:41  clk=600 MHz  MemAvail=31504000 kB
<!--PRED 1	stock_pin	memavail_kb=31504000	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7559,9238,10483,7101,5977,5236,4723,3904,2639,2533,3328-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         21.16 ± 0.04 |

build: 171974745 (10558)
<!--RO 1	stock_pin	wall_s=390	busy=205,232,213,206,24696,24066,22078,22399	busy_tot=94095	busy_little_share=0.0091	a55_cpu_cycles=34186303939	a55_inst_retired=7303738409	context_switches=7830671	a76_cpu_cycles=2025411868046	a76_l3d_cache_refill=16117789132	a76_l2d_cache_refill=8548943612	cpu_migrations=18169	page_faults=598424	a76_dtlb_walk=2290080068	a76_mem_access=711783026833	a76_inst_retired=4234865138235	a76_l1d_cache_refill=15705421410	a55_inst_share=0.0017	a76_ipc=2.091	l2ref_pki=2.019	l3ref_pki=3.806	l1dref_pki=3.709	memacc_pki=168.08	dtlbw_pki=0.5408	pmu_enabled=100.0	pmu_cpu_s=3116.1	who=at_s=6,rss_pg=1509833,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6966	l2color_cv=0.0034	l3color_cv=0.0091	contig_frac=0.8573	mean_run=6.9	vapa16=0.0103	gib_regions=28	anon_pg=23041	anon_l2cv=0.0023	anon_contig=0.8698	anon_run=7.6	anon_gib=23	file_pg=24576	file_l2cv=0.0023	file_contig=0.9893	file_run=79.0	file_gib=27	other_pg=20866	other_l2cv=0.0095	other_contig=0.6881	other_run=3.2	other_gib=28	ro_cost_ms=99-->
<!--DATA 1	stock_pin	pp2048	21.16-->

### qwen35-9b-qres-pin2x2 [qres_unpin] pass 1  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  23:21:26  clk=600 MHz  MemAvail=31278680 kB
<!--PRED 1	qres_unpin	memavail_kb=31278972	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9226,10081,10811,8316,6515,5881,5246,4332,2965,2612,3023-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         31.75 ± 0.09 |

build: 171974745 (10558)
<!--RO 1	qres_unpin	wall_s=296	busy=4254,4178,3996,3740,9313,8173,9524,8065	busy_tot=51243	busy_little_share=0.3155	a55_cpu_cycles=292010416186	a55_inst_retired=143020691399	context_switches=7251928	a76_cpu_cycles=777431286399	a76_l3d_cache_refill=10489512787	a76_l2d_cache_refill=5098585487	cpu_migrations=16774	page_faults=11423253	a76_dtlb_walk=2138529062	a76_mem_access=379367782018	a76_inst_retired=1231871676692	a76_l1d_cache_refill=9453608151	a55_inst_share=0.1040	a76_ipc=1.585	l2ref_pki=4.139	l3ref_pki=8.515	l1dref_pki=7.674	memacc_pki=307.96	dtlbw_pki=1.7360	pmu_enabled=100.0	pmu_cpu_s=2359.0	who=at_s=10,rss_pg=1642302,settled=2	pfn_zero_frac=0.0000	maps=6	sampled=132096	present_frac=0.6206	l2color_cv=0.0031	l3color_cv=0.0098	contig_frac=0.6010	mean_run=2.5	vapa16=0.0672	gib_regions=28	anon_pg=39919	anon_l2cv=0.0009	anon_contig=0.3770	anon_run=1.6	anon_gib=23	file_pg=24576	file_l2cv=0.0029	file_contig=0.9882	file_run=72.9	file_gib=27	other_pg=17483	other_l2cv=0.0169	other_contig=0.5683	other_run=2.3	other_gib=28	ro_cost_ms=133-->
<!--DATA 1	qres_unpin	pp2048	31.75-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20923MB (MemAvailable 30459MB - reserve 9535MB, no swap)

### qwen35-9b-qres-pin2x2 [qres_pin] pass 1  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4'  23:27:34  clk=600 MHz  MemAvail=31261532 kB
<!--PRED 1	qres_pin	memavail_kb=31261532	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7940,9781,10234,8723,6722,6158,5352,4479,3096,2678,2917-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         33.68 ± 0.20 |

build: 171974745 (10558)
<!--RO 1	qres_pin	wall_s=281	busy=104,87,164,109,9158,9954,9101,9300	busy_tot=37977	busy_little_share=0.0122	a55_cpu_cycles=17611385329	a55_inst_retired=3664248168	context_switches=7197811	a76_cpu_cycles=820093429963	a76_l3d_cache_refill=11163049852	a76_l2d_cache_refill=5196423650	cpu_migrations=5071	page_faults=11418144	a76_dtlb_walk=2765413576	a76_mem_access=424678025267	a76_inst_retired=1336872050837	a76_l1d_cache_refill=10608898741	a55_inst_share=0.0027	a76_ipc=1.630	l2ref_pki=3.887	l3ref_pki=8.350	l1dref_pki=7.936	memacc_pki=317.67	dtlbw_pki=2.0686	pmu_enabled=100.0	pmu_cpu_s=2245.6	who=at_s=10,rss_pg=1626631,settled=2	pfn_zero_frac=0.0000	maps=6	sampled=132096	present_frac=0.8042	l2color_cv=0.0027	l3color_cv=0.0101	contig_frac=0.6486	mean_run=2.8	vapa16=0.2825	gib_regions=28	anon_pg=63960	anon_l2cv=0.0020	anon_contig=0.5158	anon_run=2.1	anon_gib=27	file_pg=24576	file_l2cv=0.0031	file_contig=0.9762	file_run=38.9	file_gib=27	other_pg=17699	other_l2cv=0.0171	other_contig=0.6734	other_run=3.0	other_gib=28	ro_cost_ms=122-->
<!--DATA 1	qres_pin	pp2048	33.68-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20896MB (MemAvailable 30431MB - reserve 9535MB, no swap)

### qwen35-9b-qres-pin2x2 [stock_pin] pass 2  env='PIN_MASK=0xf0'  args='-t 4'  23:33:15  clk=600 MHz  MemAvail=31493016 kB
<!--PRED 2	stock_pin	memavail_kb=31493016	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8672,8952,9831,8538,6641,6237,5451,4552,3115,2870,2860-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.82 ± 0.06 |

build: 171974745 (10558)
<!--RO 2	stock_pin	wall_s=396	busy=224,224,228,176,24790,24559,21991,22594	busy_tot=94786	busy_little_share=0.0090	a55_cpu_cycles=33863408062	a55_inst_retired=7188535585	context_switches=7847371	a76_cpu_cycles=2036894269586	a76_l3d_cache_refill=16647812665	a76_l2d_cache_refill=8442157795	cpu_migrations=18649	page_faults=598000	a76_dtlb_walk=2292585184	a76_mem_access=712130648157	a76_inst_retired=4236125147793	a76_l1d_cache_refill=15677973565	a55_inst_share=0.0017	a76_ipc=2.080	l2ref_pki=1.993	l3ref_pki=3.930	l1dref_pki=3.701	memacc_pki=168.11	dtlbw_pki=0.5412	pmu_enabled=100.0	pmu_cpu_s=3165.2	who=at_s=6,rss_pg=1509840,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6993	l2color_cv=0.0058	l3color_cv=0.0169	contig_frac=0.8857	mean_run=8.6	vapa16=0.0128	gib_regions=28	anon_pg=23041	anon_l2cv=0.0031	anon_contig=0.8664	anon_run=7.4	anon_gib=25	file_pg=24576	file_l2cv=0.0017	file_contig=0.9894	file_run=79.5	file_gib=27	other_pg=21127	other_l2cv=0.0201	other_contig=0.7860	other_run=4.6	other_gib=28	ro_cost_ms=90-->
<!--DATA 2	stock_pin	pp2048	20.82-->

### qwen35-9b-qres-pin2x2 [qres_unpin] pass 2  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  23:41:06  clk=600 MHz  MemAvail=31278320 kB
<!--PRED 2	qres_unpin	memavail_kb=31278320	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6450,8719,7711,7261,6397,5894,5149,4350,3015,2622,3034-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         31.82 ± 0.20 |

build: 171974745 (10558)
<!--RO 2	qres_unpin	wall_s=295	busy=4376,4086,4039,3729,9212,8580,8685,8517	busy_tot=51224	busy_little_share=0.3168	a55_cpu_cycles=292850832721	a55_inst_retired=143877874158	context_switches=7236477	a76_cpu_cycles=774620394025	a76_l3d_cache_refill=10463770705	a76_l2d_cache_refill=5099149953	cpu_migrations=16851	page_faults=11415477	a76_dtlb_walk=2150482832	a76_mem_access=378650243044	a76_inst_retired=1230984788754	a76_l1d_cache_refill=9424883554	a55_inst_share=0.1046	a76_ipc=1.589	l2ref_pki=4.142	l3ref_pki=8.500	l1dref_pki=7.656	memacc_pki=307.60	dtlbw_pki=1.7470	pmu_enabled=100.0	pmu_cpu_s=2354.0	who=at_s=10,rss_pg=1650460,settled=2	pfn_zero_frac=0.0000	maps=6	sampled=132096	present_frac=0.6198	l2color_cv=0.0028	l3color_cv=0.0147	contig_frac=0.4057	mean_run=1.7	vapa16=0.0343	gib_regions=28	anon_pg=39639	anon_l2cv=0.0009	anon_contig=0.2049	anon_run=1.3	anon_gib=24	file_pg=24576	file_l2cv=0.0011	file_contig=0.9893	file_run=79.3	file_gib=27	other_pg=17661	other_l2cv=0.0158	other_contig=0.0443	other_run=1.0	other_gib=28	ro_cost_ms=107-->
<!--DATA 2	qres_unpin	pp2048	31.82-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20992MB (MemAvailable 30528MB - reserve 9535MB, no swap)

### qwen35-9b-qres-pin2x2 [qres_pin] pass 2  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4'  23:47:13  clk=600 MHz  MemAvail=31111176 kB
<!--PRED 2	qres_pin	memavail_kb=31111176	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8400,8723,9479,8547,6354,5845,5077,4241,2947,2449,3098-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         33.55 ± 0.07 |

build: 171974745 (10558)
<!--RO 2	qres_pin	wall_s=282	busy=123,111,83,93,9362,9836,9756,8643	busy_tot=38007	busy_little_share=0.0108	a55_cpu_cycles=16663550275	a55_inst_retired=3368066630	context_switches=7196738	a76_cpu_cycles=820741866925	a76_l3d_cache_refill=11073797150	a76_l2d_cache_refill=5151922206	cpu_migrations=4977	page_faults=11413131	a76_dtlb_walk=3687431270	a76_mem_access=424759702003	a76_inst_retired=1337204563163	a76_l1d_cache_refill=10633705558	a55_inst_share=0.0025	a76_ipc=1.629	l2ref_pki=3.853	l3ref_pki=8.281	l1dref_pki=7.952	memacc_pki=317.65	dtlbw_pki=2.7576	pmu_enabled=100.0	pmu_cpu_s=2249.6	who=at_s=10,rss_pg=1626633,settled=2	pfn_zero_frac=0.0000	maps=6	sampled=132096	present_frac=0.8087	l2color_cv=0.0056	l3color_cv=0.0132	contig_frac=0.5230	mean_run=2.1	vapa16=0.0593	gib_regions=28	anon_pg=64440	anon_l2cv=0.0005	anon_contig=0.3638	anon_run=1.6	anon_gib=25	file_pg=24576	file_l2cv=0.0017	file_contig=0.9885	file_run=74.5	file_gib=28	other_pg=17816	other_l2cv=0.0319	other_contig=0.4566	other_run=1.8	other_gib=28	ro_cost_ms=143-->
<!--DATA 2	qres_pin	pp2048	33.55-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20796MB (MemAvailable 30331MB - reserve 9535MB, no swap)

### qwen35-9b-qres-pin2x2 [stock_unpin] pass 2  env=''  args=''  23:53:00  clk=600 MHz  MemAvail=31557548 kB
<!--PRED 2	stock_unpin	memavail_kb=31557548	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7997,9435,9819,8559,6223,5799,5057,4246,2940,2716,3077-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         18.89 ± 0.10 |

build: 171974745 (10558)
<!--RO 2	stock_unpin	wall_s=436	busy=7267,5583,5422,5207,25034,22415,21975,22484	busy_tot=115387	busy_little_share=0.2035	a55_cpu_cycles=431196843981	a55_inst_retired=224644357042	context_switches=7968078	a76_cpu_cycles=1988909096100	a76_l3d_cache_refill=15811521803	a76_l2d_cache_refill=8220161493	cpu_migrations=40958	page_faults=593248	a76_dtlb_walk=2014432852	a76_mem_access=657140165735	a76_inst_retired=4128544770383	a76_l1d_cache_refill=13521708322	a55_inst_share=0.0516	a76_ipc=2.076	l2ref_pki=1.991	l3ref_pki=3.830	l1dref_pki=3.275	memacc_pki=159.17	dtlbw_pki=0.4879	pmu_enabled=100.0	pmu_cpu_s=3482.1	who=at_s=6,rss_pg=1509868,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6978	l2color_cv=0.0067	l3color_cv=0.0189	contig_frac=0.8885	mean_run=8.8	vapa16=0.0236	gib_regions=28	anon_pg=23041	anon_l2cv=0.0018	anon_contig=0.9239	anon_run=12.8	anon_gib=24	file_pg=24576	file_l2cv=0.0016	file_contig=0.9896	file_run=81.1	file_gib=27	other_pg=20984	other_l2cv=0.0233	other_contig=0.7311	other_run=3.7	other_gib=28	ro_cost_ms=83-->
<!--DATA 2	stock_unpin	pp2048	18.89-->

### qwen35-9b-qres-pin2x2 [qres_unpin] pass 3  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  00:01:32  clk=600 MHz  MemAvail=31407440 kB
<!--PRED 3	qres_unpin	memavail_kb=31407440	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8165,10205,10159,8611,6659,6090,5406,4514,3128,2760,2897-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         31.62 ± 0.33 |

build: 171974745 (10558)
<!--RO 3	qres_unpin	wall_s=296	busy=4319,4152,3864,3685,8940,8765,9268,8127	busy_tot=51120	busy_little_share=0.3134	a55_cpu_cycles=289420732412	a55_inst_retired=141327082124	context_switches=7247477	a76_cpu_cycles=776009210022	a76_l3d_cache_refill=10416046692	a76_l2d_cache_refill=5104745967	cpu_migrations=16776	page_faults=11420075	a76_dtlb_walk=2126588766	a76_mem_access=380313891414	a76_inst_retired=1233674692052	a76_l1d_cache_refill=9438943351	a55_inst_share=0.1028	a76_ipc=1.590	l2ref_pki=4.138	l3ref_pki=8.443	l1dref_pki=7.651	memacc_pki=308.28	dtlbw_pki=1.7238	pmu_enabled=100.0	pmu_cpu_s=2363.9	who=at_s=10,rss_pg=1651230,settled=2	pfn_zero_frac=0.0000	maps=6	sampled=132096	present_frac=0.6185	l2color_cv=0.0026	l3color_cv=0.0090	contig_frac=0.6350	mean_run=2.7	vapa16=0.0260	gib_regions=28	anon_pg=39455	anon_l2cv=0.0012	anon_contig=0.4271	anon_run=1.7	anon_gib=23	file_pg=24576	file_l2cv=0.0006	file_contig=0.9904	file_run=86.8	file_gib=28	other_pg=17676	other_l2cv=0.0133	other_contig=0.6050	other_run=2.5	other_gib=28	ro_cost_ms=110-->
<!--DATA 3	qres_unpin	pp2048	31.62-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21064MB (MemAvailable 30599MB - reserve 9535MB, no swap)

### qwen35-9b-qres-pin2x2 [qres_pin] pass 3  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4'  00:07:39  clk=600 MHz  MemAvail=31135200 kB
<!--PRED 3	qres_pin	memavail_kb=31135200	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8701,10418,9401,8040,6977,6458,5606,4683,3277,2714,2774-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         33.51 ± 0.14 |

build: 171974745 (10558)
<!--RO 3	qres_pin	wall_s=283	busy=129,132,114,75,9231,10282,9244,8893	busy_tot=38100	busy_little_share=0.0118	a55_cpu_cycles=17162720105	a55_inst_retired=3566358480	context_switches=7202515	a76_cpu_cycles=821647349169	a76_l3d_cache_refill=11068573981	a76_l2d_cache_refill=5733064785	cpu_migrations=5228	page_faults=11416530	a76_dtlb_walk=3690110157	a76_mem_access=424806678815	a76_inst_retired=1337616328694	a76_l1d_cache_refill=10612365084	a55_inst_share=0.0027	a76_ipc=1.628	l2ref_pki=4.286	l3ref_pki=8.275	l1dref_pki=7.934	memacc_pki=317.58	dtlbw_pki=2.7587	pmu_enabled=100.0	pmu_cpu_s=2251.0	who=at_s=10,rss_pg=1626635,settled=2	pfn_zero_frac=0.0000	maps=6	sampled=132096	present_frac=0.8039	l2color_cv=0.0013	l3color_cv=0.0045	contig_frac=0.5460	mean_run=2.2	vapa16=0.0357	gib_regions=28	anon_pg=63960	anon_l2cv=0.0019	anon_contig=0.4058	anon_run=1.7	anon_gib=28	file_pg=24576	file_l2cv=0.0011	file_contig=0.9705	file_run=31.9	file_gib=27	other_pg=17657	other_l2cv=0.0046	other_contig=0.4626	other_run=1.9	other_gib=23	ro_cost_ms=151-->
<!--DATA 3	qres_pin	pp2048	33.51-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20850MB (MemAvailable 30386MB - reserve 9535MB, no swap)

### qwen35-9b-qres-pin2x2 [stock_unpin] pass 3  env=''  args=''  00:13:27  clk=600 MHz  MemAvail=31557236 kB
<!--PRED 3	stock_unpin	memavail_kb=31557236	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8835,9539,10015,8472,6123,5803,5048,4218,2935,2696,3094-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         19.03 ± 0.20 |

build: 171974745 (10558)
<!--RO 3	stock_unpin	wall_s=433	busy=6872,5417,5523,5315,24900,22376,22301,22631	busy_tot=115335	busy_little_share=0.2005	a55_cpu_cycles=425095743831	a55_inst_retired=223452610849	context_switches=7978222	a76_cpu_cycles=1991005993247	a76_l3d_cache_refill=16036505910	a76_l2d_cache_refill=8216850784	cpu_migrations=40928	page_faults=596695	a76_dtlb_walk=1839670786	a76_mem_access=657265491024	a76_inst_retired=4129743456685	a76_l1d_cache_refill=13596864469	a55_inst_share=0.0513	a76_ipc=2.074	l2ref_pki=1.990	l3ref_pki=3.883	l1dref_pki=3.292	memacc_pki=159.15	dtlbw_pki=0.4455	pmu_enabled=100.0	pmu_cpu_s=3461.2	who=at_s=6,rss_pg=1509874,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6963	l2color_cv=0.0054	l3color_cv=0.0118	contig_frac=0.8658	mean_run=7.4	vapa16=0.0148	gib_regions=28	anon_pg=23041	anon_l2cv=0.0030	anon_contig=0.9090	anon_run=10.8	anon_gib=26	file_pg=24576	file_l2cv=0.0036	file_contig=0.9875	file_run=69.4	file_gib=28	other_pg=20834	other_l2cv=0.0204	other_contig=0.6745	other_run=3.1	other_gib=28	ro_cost_ms=107-->
<!--DATA 3	stock_unpin	pp2048	19.03-->

### qwen35-9b-qres-pin2x2 [stock_pin] pass 3  env='PIN_MASK=0xf0'  args='-t 4'  00:21:40  clk=600 MHz  MemAvail=31501556 kB
<!--PRED 3	stock_pin	memavail_kb=31501556	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7547,8835,9234,8176,6058,5442,4889,4060,2771,2541,3249-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.82 ± 0.04 |

build: 171974745 (10558)
<!--RO 3	stock_pin	wall_s=396	busy=199,214,201,211,24830,24384,21999,22767	busy_tot=94805	busy_little_share=0.0087	a55_cpu_cycles=33198969005	a55_inst_retired=6867887868	context_switches=7848102	a76_cpu_cycles=2035702578212	a76_l3d_cache_refill=16625743964	a76_l2d_cache_refill=8479246210	cpu_migrations=18559	page_faults=592083	a76_dtlb_walk=2146901806	a76_mem_access=711982847554	a76_inst_retired=4236132965020	a76_l1d_cache_refill=15632200770	a55_inst_share=0.0016	a76_ipc=2.081	l2ref_pki=2.002	l3ref_pki=3.925	l1dref_pki=3.690	memacc_pki=168.07	dtlbw_pki=0.5068	pmu_enabled=100.0	pmu_cpu_s=3164.5	who=at_s=6,rss_pg=1509852,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6993	l2color_cv=0.0032	l3color_cv=0.0118	contig_frac=0.8827	mean_run=8.4	vapa16=0.0201	gib_regions=28	anon_pg=23041	anon_l2cv=0.0015	anon_contig=0.8666	anon_run=7.4	anon_gib=26	file_pg=24576	file_l2cv=0.0017	file_contig=0.9881	file_run=72.3	file_gib=28	other_pg=21123	other_l2cv=0.0102	other_contig=0.7777	other_run=4.5	other_gib=28	ro_cost_ms=79-->
<!--DATA 3	stock_pin	pp2048	20.82-->

### qwen35-9b-qres-pin2x2 [qres_pin] pass 4  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4'  00:29:29  clk=600 MHz  MemAvail=31089192 kB
<!--PRED 4	qres_pin	memavail_kb=31089192	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7228,8737,8033,7453,6492,5821,5185,4294,2978,2479,3070-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         33.73 ± 0.09 |

build: 171974745 (10558)
<!--RO 4	qres_pin	wall_s=280	busy=121,112,137,66,9292,9367,9772,9108	busy_tot=37975	busy_little_share=0.0115	a55_cpu_cycles=16891627742	a55_inst_retired=3337015539	context_switches=7198671	a76_cpu_cycles=818948338272	a76_l3d_cache_refill=11092250095	a76_l2d_cache_refill=5146278683	cpu_migrations=5048	page_faults=11411265	a76_dtlb_walk=3301764480	a76_mem_access=424724702760	a76_inst_retired=1337074540736	a76_l1d_cache_refill=10645963917	a55_inst_share=0.0025	a76_ipc=1.633	l2ref_pki=3.849	l3ref_pki=8.296	l1dref_pki=7.962	memacc_pki=317.65	dtlbw_pki=2.4694	pmu_enabled=100.0	pmu_cpu_s=2235.7	who=at_s=10,rss_pg=1626648,settled=2	pfn_zero_frac=0.0000	maps=6	sampled=133120	present_frac=0.8080	l2color_cv=0.0026	l3color_cv=0.0103	contig_frac=0.4976	mean_run=2.0	vapa16=0.1630	gib_regions=28	anon_pg=65238	anon_l2cv=0.0004	anon_contig=0.4083	anon_run=1.7	anon_gib=25	file_pg=24576	file_l2cv=0.0013	file_contig=0.9883	file_run=73.1	file_gib=28	other_pg=17748	other_l2cv=0.0141	other_contig=0.1462	other_run=1.2	other_gib=28	ro_cost_ms=122-->
<!--DATA 4	qres_pin	pp2048	33.73-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20821MB (MemAvailable 30356MB - reserve 9535MB, no swap)

### qwen35-9b-qres-pin2x2 [stock_unpin] pass 4  env=''  args=''  00:35:14  clk=600 MHz  MemAvail=31531812 kB
<!--PRED 4	stock_unpin	memavail_kb=31531812	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7071,9982,10408,8565,6401,5940,5257,4386,3033,2770,2980-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         19.12 ± 0.04 |

build: 171974745 (10558)
<!--RO 4	stock_unpin	wall_s=431	busy=6852,5624,5390,5289,24907,22104,22177,22610	busy_tot=114953	busy_little_share=0.2014	a55_cpu_cycles=426469787544	a55_inst_retired=223305231784	context_switches=7952536	a76_cpu_cycles=1984846286717	a76_l3d_cache_refill=15807487040	a76_l2d_cache_refill=8228675757	cpu_migrations=40178	page_faults=591695	a76_dtlb_walk=2211623760	a76_mem_access=657038768797	a76_inst_retired=4129175096595	a76_l1d_cache_refill=13577616552	a55_inst_share=0.0513	a76_ipc=2.080	l2ref_pki=1.993	l3ref_pki=3.828	l1dref_pki=3.288	memacc_pki=159.12	dtlbw_pki=0.5356	pmu_enabled=100.0	pmu_cpu_s=3441.1	who=at_s=6,rss_pg=1509878,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6993	l2color_cv=0.0053	l3color_cv=0.0137	contig_frac=0.8871	mean_run=8.7	vapa16=0.1527	gib_regions=28	anon_pg=23041	anon_l2cv=0.0040	anon_contig=0.9549	anon_run=21.3	anon_gib=28	file_pg=24576	file_l2cv=0.0041	file_contig=0.9874	file_run=68.6	file_gib=28	other_pg=21126	other_l2cv=0.0167	other_contig=0.6965	other_run=3.3	other_gib=28	ro_cost_ms=95-->
<!--DATA 4	stock_unpin	pp2048	19.12-->

### qwen35-9b-qres-pin2x2 [stock_pin] pass 4  env='PIN_MASK=0xf0'  args='-t 4'  00:43:24  clk=600 MHz  MemAvail=31500408 kB
<!--PRED 4	stock_pin	memavail_kb=31500408	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7866,8032,9279,8072,6017,5432,4880,4071,2776,2533,3253-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.82 ± 0.05 |

build: 171974745 (10558)
<!--RO 4	stock_pin	wall_s=397	busy=200,206,212,205,24824,24379,22237,22522	busy_tot=94785	busy_little_share=0.0087	a55_cpu_cycles=33657077258	a55_inst_retired=7103045053	context_switches=7832492	a76_cpu_cycles=2037614242779	a76_l3d_cache_refill=17194712678	a76_l2d_cache_refill=8405949387	cpu_migrations=18662	page_faults=594051	a76_dtlb_walk=2284508429	a76_mem_access=711633040897	a76_inst_retired=4235905761585	a76_l1d_cache_refill=15512284630	a55_inst_share=0.0017	a76_ipc=2.079	l2ref_pki=1.984	l3ref_pki=4.059	l1dref_pki=3.662	memacc_pki=168.00	dtlbw_pki=0.5393	pmu_enabled=100.0	pmu_cpu_s=3164.3	who=at_s=6,rss_pg=1509853,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6968	l2color_cv=0.0028	l3color_cv=0.0148	contig_frac=0.8568	mean_run=6.9	vapa16=0.0134	gib_regions=28	anon_pg=23042	anon_l2cv=0.0028	anon_contig=0.9242	anon_run=12.9	anon_gib=26	file_pg=24576	file_l2cv=0.0017	file_contig=0.9885	file_run=74.5	file_gib=27	other_pg=20881	other_l2cv=0.0078	other_contig=0.6274	other_run=2.7	other_gib=28	ro_cost_ms=115-->
<!--DATA 4	stock_pin	pp2048	20.82-->

### qwen35-9b-qres-pin2x2 [qres_unpin] pass 4  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  00:51:16  clk=600 MHz  MemAvail=31437016 kB
<!--PRED 4	qres_unpin	memavail_kb=31437016	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8263,9890,10011,8901,6695,6031,5348,4453,3112,2719,2941-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         31.64 ± 0.04 |

build: 171974745 (10558)
<!--RO 4	qres_unpin	wall_s=296	busy=4316,4103,3852,3765,8946,8813,9049,8179	busy_tot=51023	busy_little_share=0.3143	a55_cpu_cycles=289838265174	a55_inst_retired=141994523471	context_switches=7252572	a76_cpu_cycles=773803886341	a76_l3d_cache_refill=10318103991	a76_l2d_cache_refill=5104296062	cpu_migrations=17017	page_faults=11420845	a76_dtlb_walk=2147945684	a76_mem_access=379849749128	a76_inst_retired=1232743700541	a76_l1d_cache_refill=9421113638	a55_inst_share=0.1033	a76_ipc=1.593	l2ref_pki=4.141	l3ref_pki=8.370	l1dref_pki=7.642	memacc_pki=308.13	dtlbw_pki=1.7424	pmu_enabled=100.0	pmu_cpu_s=2361.8	who=at_s=10,rss_pg=1651244,settled=2	pfn_zero_frac=0.0000	maps=5	sampled=107520	present_frac=0.7574	l2color_cv=0.0033	l3color_cv=0.0056	contig_frac=0.5789	mean_run=2.4	vapa16=0.0413	gib_regions=28	anon_pg=39310	anon_l2cv=0.0009	anon_contig=0.3449	anon_run=1.5	anon_gib=24	file_pg=24576	file_l2cv=0.0012	file_contig=0.9866	file_run=65.2	file_gib=27	other_pg=17545	other_l2cv=0.0157	other_contig=0.5322	other_run=2.1	other_gib=28	ro_cost_ms=126-->
<!--DATA 4	qres_unpin	pp2048	31.64-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21124MB (MemAvailable 30660MB - reserve 9535MB, no swap)

### qwen35-9b-qres-pin2x2 [stock_unpin] pass 5  env=''  args=''  00:57:18  clk=600 MHz  MemAvail=31510292 kB
<!--PRED 5	stock_unpin	memavail_kb=31510000	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8395,9349,9362,7935,6121,5600,5014,4185,2955,2662,3114-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         18.74 ± 0.10 |

build: 171974745 (10558)
<!--RO 5	stock_unpin	wall_s=438	busy=7594,5438,5438,5217,24954,22206,22168,22369	busy_tot=115384	busy_little_share=0.2053	a55_cpu_cycles=434946258341	a55_inst_retired=226177703066	context_switches=7973516	a76_cpu_cycles=1984261299590	a76_l3d_cache_refill=15714374668	a76_l2d_cache_refill=8162091353	cpu_migrations=40096	page_faults=590907	a76_dtlb_walk=2198042267	a76_mem_access=656661498720	a76_inst_retired=4126252503906	a76_l1d_cache_refill=13454569394	a55_inst_share=0.0520	a76_ipc=2.079	l2ref_pki=1.978	l3ref_pki=3.808	l1dref_pki=3.261	memacc_pki=159.14	dtlbw_pki=0.5327	pmu_enabled=100.0	pmu_cpu_s=3501.5	who=at_s=6,rss_pg=1509878,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6992	l2color_cv=0.0050	l3color_cv=0.0115	contig_frac=0.8441	mean_run=6.3	vapa16=0.0097	gib_regions=28	anon_pg=23041	anon_l2cv=0.0019	anon_contig=0.8100	anon_run=5.2	anon_gib=28	file_pg=24576	file_l2cv=0.0020	file_contig=0.9769	file_run=40.0	file_gib=27	other_pg=21118	other_l2cv=0.0166	other_contig=0.7267	other_run=3.6	other_gib=28	ro_cost_ms=67-->
<!--DATA 5	stock_unpin	pp2048	18.74-->

### qwen35-9b-qres-pin2x2 [stock_pin] pass 5  env='PIN_MASK=0xf0'  args='-t 4'  01:05:36  clk=600 MHz  MemAvail=31505312 kB
<!--PRED 5	stock_pin	memavail_kb=31505312	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7396,8916,9371,8037,5962,5235,4816,3998,2754,2486,3300-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.78 ± 0.06 |

build: 171974745 (10558)
<!--RO 5	stock_pin	wall_s=397	busy=200,193,222,197,24649,24557,22081,22457	busy_tot=94556	busy_little_share=0.0086	a55_cpu_cycles=33221261063	a55_inst_retired=6916087265	context_switches=7842311	a76_cpu_cycles=2031117901074	a76_l3d_cache_refill=16828288305	a76_l2d_cache_refill=8445960931	cpu_migrations=18686	page_faults=592210	a76_dtlb_walk=2239724606	a76_mem_access=711716559876	a76_inst_retired=4236006023371	a76_l1d_cache_refill=15600276572	a55_inst_share=0.0016	a76_ipc=2.086	l2ref_pki=1.994	l3ref_pki=3.973	l1dref_pki=3.683	memacc_pki=168.02	dtlbw_pki=0.5287	pmu_enabled=100.0	pmu_cpu_s=3170.5	who=at_s=6,rss_pg=1509856,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6992	l2color_cv=0.0040	l3color_cv=0.0107	contig_frac=0.8326	mean_run=5.9	vapa16=0.3683	gib_regions=28	anon_pg=23041	anon_l2cv=0.0051	anon_contig=0.7886	anon_run=4.7	anon_gib=28	file_pg=24576	file_l2cv=0.0019	file_contig=0.9901	file_run=84.2	file_gib=28	other_pg=21113	other_l2cv=0.0137	other_contig=0.6973	other_run=3.3	other_gib=28	ro_cost_ms=96-->
<!--DATA 5	stock_pin	pp2048	20.78-->

### qwen35-9b-qres-pin2x2 [qres_unpin] pass 5  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  01:13:29  clk=600 MHz  MemAvail=31385700 kB
<!--PRED 5	qres_unpin	memavail_kb=31385472	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8209,9370,9875,7903,6283,5627,5042,4168,2905,2531,3157-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         31.32 ± 0.12 |

build: 171974745 (10558)
<!--RO 5	qres_unpin	wall_s=298	busy=4310,4148,3948,3676,9033,8700,9153,8114	busy_tot=51082	busy_little_share=0.3148	a55_cpu_cycles=290133423466	a55_inst_retired=142083412557	context_switches=7266764	a76_cpu_cycles=774310003197	a76_l3d_cache_refill=10411505122	a76_l2d_cache_refill=5096305896	cpu_migrations=16788	page_faults=11417245	a76_dtlb_walk=2148338503	a76_mem_access=379700594985	a76_inst_retired=1232682331075	a76_l1d_cache_refill=9445833268	a55_inst_share=0.1034	a76_ipc=1.592	l2ref_pki=4.134	l3ref_pki=8.446	l1dref_pki=7.663	memacc_pki=308.03	dtlbw_pki=1.7428	pmu_enabled=100.0	pmu_cpu_s=2379.2	who=at_s=10,rss_pg=1648331,settled=2	pfn_zero_frac=0.0000	maps=6	sampled=132096	present_frac=0.6204	l2color_cv=0.0035	l3color_cv=0.0105	contig_frac=0.4462	mean_run=1.8	vapa16=0.0478	gib_regions=28	anon_pg=39654	anon_l2cv=0.0008	anon_contig=0.2419	anon_run=1.3	anon_gib=28	file_pg=24576	file_l2cv=0.0010	file_contig=0.9876	file_run=69.6	file_gib=26	other_pg=17716	other_l2cv=0.0171	other_contig=0.1526	other_run=1.2	other_gib=28	ro_cost_ms=118-->
<!--DATA 5	qres_unpin	pp2048	31.32-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21092MB (MemAvailable 30628MB - reserve 9535MB, no swap)

### qwen35-9b-qres-pin2x2 [qres_pin] pass 5  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4'  01:19:38  clk=600 MHz  MemAvail=31423528 kB
<!--PRED 5	qres_pin	memavail_kb=31423528	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5755,9982,10240,8262,6763,6148,5399,4557,3183,2797,2866-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         33.55 ± 0.20 |

build: 171974745 (10558)
<!--RO 5	qres_pin	wall_s=282	busy=121,108,86,117,9234,10096,9471,8710	busy_tot=37943	busy_little_share=0.0114	a55_cpu_cycles=17073292417	a55_inst_retired=3487371118	context_switches=7209495	a76_cpu_cycles=819191738883	a76_l3d_cache_refill=11013086087	a76_l2d_cache_refill=5133073683	cpu_migrations=5175	page_faults=11414281	a76_dtlb_walk=3701920759	a76_mem_access=424720948831	a76_inst_retired=1337176158570	a76_l1d_cache_refill=10600852993	a55_inst_share=0.0026	a76_ipc=1.632	l2ref_pki=3.839	l3ref_pki=8.236	l1dref_pki=7.928	memacc_pki=317.63	dtlbw_pki=2.7685	pmu_enabled=100.0	pmu_cpu_s=2251.1	who=at_s=10,rss_pg=1626650,settled=2	pfn_zero_frac=0.0000	maps=6	sampled=132096	present_frac=0.8063	l2color_cv=0.0039	l3color_cv=0.0090	contig_frac=0.6094	mean_run=2.6	vapa16=0.0147	gib_regions=28	anon_pg=64184	anon_l2cv=0.0010	anon_contig=0.4948	anon_run=2.0	anon_gib=27	file_pg=24576	file_l2cv=0.0024	file_contig=0.9896	file_run=81.1	file_gib=28	other_pg=17749	other_l2cv=0.0249	other_contig=0.4976	other_run=2.0	other_gib=28	ro_cost_ms=120-->
<!--DATA 5	qres_pin	pp2048	33.55-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21090MB (MemAvailable 30626MB - reserve 9535MB, no swap)

### qwen35-9b-qres-pin2x2 [stock_pin] pass 6  env='PIN_MASK=0xf0'  args='-t 4'  01:25:20  clk=600 MHz  MemAvail=31502560 kB
<!--PRED 6	stock_pin	memavail_kb=31502560	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6601,9201,9348,8273,6230,5845,5155,4290,2970,2697,3059-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.87 ± 0.06 |

build: 171974745 (10558)
<!--RO 6	stock_pin	wall_s=396	busy=200,235,239,149,24658,24367,22361,22607	busy_tot=94816	busy_little_share=0.0087	a55_cpu_cycles=33338465452	a55_inst_retired=7020151844	context_switches=7844291	a76_cpu_cycles=2033641058250	a76_l3d_cache_refill=17104314018	a76_l2d_cache_refill=8402711352	cpu_migrations=19000	page_faults=593700	a76_dtlb_walk=1892099601	a76_mem_access=711868133747	a76_inst_retired=4236797533595	a76_l1d_cache_refill=15542213900	a55_inst_share=0.0017	a76_ipc=2.083	l2ref_pki=1.983	l3ref_pki=4.037	l1dref_pki=3.668	memacc_pki=168.02	dtlbw_pki=0.4466	pmu_enabled=100.0	pmu_cpu_s=3157.5	who=at_s=6,rss_pg=1509856,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6977	l2color_cv=0.0050	l3color_cv=0.0122	contig_frac=0.8479	mean_run=6.5	vapa16=0.0385	gib_regions=28	anon_pg=23041	anon_l2cv=0.0008	anon_contig=0.9091	anon_run=10.8	anon_gib=26	file_pg=24576	file_l2cv=0.0011	file_contig=0.9885	file_run=74.5	file_gib=27	other_pg=20974	other_l2cv=0.0165	other_contig=0.6161	other_run=2.6	other_gib=28	ro_cost_ms=92-->
<!--DATA 6	stock_pin	pp2048	20.87-->

### qwen35-9b-qres-pin2x2 [qres_unpin] pass 6  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  01:33:12  clk=600 MHz  MemAvail=31410488 kB
<!--PRED 6	qres_unpin	memavail_kb=31410488	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8742,9568,9780,8127,6378,5798,5131,4331,3036,2587,3067-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         31.69 ± 0.19 |

build: 171974745 (10558)
<!--RO 6	qres_unpin	wall_s=296	busy=4339,4107,4052,3827,9016,8184,9395,8446	busy_tot=51366	busy_little_share=0.3178	a55_cpu_cycles=294525181284	a55_inst_retired=144365955567	context_switches=7241501	a76_cpu_cycles=775743092075	a76_l3d_cache_refill=10380004672	a76_l2d_cache_refill=5113082055	cpu_migrations=16466	page_faults=11419390	a76_dtlb_walk=2147279647	a76_mem_access=378676134456	a76_inst_retired=1230141047869	a76_l1d_cache_refill=9437354912	a55_inst_share=0.1050	a76_ipc=1.586	l2ref_pki=4.157	l3ref_pki=8.438	l1dref_pki=7.672	memacc_pki=307.83	dtlbw_pki=1.7456	pmu_enabled=100.0	pmu_cpu_s=2362.4	who=at_s=10,rss_pg=1626666,settled=2	pfn_zero_frac=0.0000	maps=6	sampled=132096	present_frac=0.6867	l2color_cv=0.0040	l3color_cv=0.0101	contig_frac=0.4926	mean_run=2.0	vapa16=0.0261	gib_regions=28	anon_pg=48330	anon_l2cv=0.0013	anon_contig=0.1816	anon_run=1.2	anon_gib=26	file_pg=24576	file_l2cv=0.0020	file_contig=0.9798	file_run=45.2	file_gib=27	other_pg=17810	other_l2cv=0.0202	other_contig=0.6641	other_run=3.0	other_gib=28	ro_cost_ms=131-->
<!--DATA 6	qres_unpin	pp2048	31.69-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21039MB (MemAvailable 30574MB - reserve 9535MB, no swap)

### qwen35-9b-qres-pin2x2 [qres_pin] pass 6  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4'  01:39:20  clk=600 MHz  MemAvail=31109564 kB
<!--PRED 6	qres_pin	memavail_kb=31109704	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7730,8156,8176,7871,6802,6157,5388,4521,3152,2631,2897-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         33.47 ± 0.28 |

build: 171974745 (10558)
<!--RO 6	qres_pin	wall_s=282	busy=83,137,107,118,9369,9891,9456,8953	busy_tot=38114	busy_little_share=0.0117	a55_cpu_cycles=17358796752	a55_inst_retired=3577300526	context_switches=7200323	a76_cpu_cycles=820592699914	a76_l3d_cache_refill=11022828164	a76_l2d_cache_refill=5127819056	cpu_migrations=5063	page_faults=11414843	a76_dtlb_walk=3703185648	a76_mem_access=424651760709	a76_inst_retired=1336736669400	a76_l1d_cache_refill=10597554598	a55_inst_share=0.0027	a76_ipc=1.629	l2ref_pki=3.836	l3ref_pki=8.246	l1dref_pki=7.928	memacc_pki=317.68	dtlbw_pki=2.7703	pmu_enabled=100.0	pmu_cpu_s=2250.8	who=at_s=10,rss_pg=1626650,settled=2	pfn_zero_frac=0.0000	maps=6	sampled=132096	present_frac=0.8078	l2color_cv=0.0009	l3color_cv=0.0061	contig_frac=0.5425	mean_run=2.2	vapa16=0.0595	gib_regions=28	anon_pg=64344	anon_l2cv=0.0012	anon_contig=0.4072	anon_run=1.7	anon_gib=28	file_pg=24576	file_l2cv=0.0051	file_contig=0.9835	file_run=54.4	file_gib=27	other_pg=17792	other_l2cv=0.0043	other_contig=0.4226	other_run=1.7	other_gib=26	ro_cost_ms=123-->
<!--DATA 6	qres_pin	pp2048	33.47-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20826MB (MemAvailable 30362MB - reserve 9535MB, no swap)

### qwen35-9b-qres-pin2x2 [stock_unpin] pass 6  env=''  args=''  01:45:07  clk=600 MHz  MemAvail=31512188 kB
<!--PRED 6	stock_unpin	memavail_kb=31512188	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9009,10176,10691,8297,6595,6122,5385,4490,3155,2822,2889-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         18.94 ± 0.06 |

build: 171974745 (10558)
<!--RO 6	stock_unpin	wall_s=435	busy=6995,5437,5592,5352,24929,22155,22024,22866	busy_tot=115350	busy_little_share=0.2027	a55_cpu_cycles=429518487809	a55_inst_retired=224833564826	context_switches=7979458	a76_cpu_cycles=1988423876053	a76_l3d_cache_refill=15931249112	a76_l2d_cache_refill=8185688030	cpu_migrations=40657	page_faults=591781	a76_dtlb_walk=2209409175	a76_mem_access=656659086392	a76_inst_retired=4127389418370	a76_l1d_cache_refill=13475588289	a55_inst_share=0.0517	a76_ipc=2.076	l2ref_pki=1.983	l3ref_pki=3.860	l1dref_pki=3.265	memacc_pki=159.10	dtlbw_pki=0.5353	pmu_enabled=100.0	pmu_cpu_s=3473.9	who=at_s=6,rss_pg=1509874,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6968	l2color_cv=0.0058	l3color_cv=0.0175	contig_frac=0.8766	mean_run=8.0	vapa16=0.0106	gib_regions=28	anon_pg=23007	anon_l2cv=0.0038	anon_contig=0.9725	anon_run=33.9	anon_gib=28	file_pg=24576	file_l2cv=0.0019	file_contig=0.9805	file_run=46.7	file_gib=28	other_pg=20919	other_l2cv=0.0198	other_contig=0.6490	other_run=2.8	other_gib=28	ro_cost_ms=67-->
<!--DATA 6	stock_unpin	pp2048	18.94-->

#### summary: per-arm mean over 6 passes, ratios paired within a pass against [stock_unpin]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock_unpin | 18.97 | 6 | 18.74 | 19.13 | -- | -- |
| pp2048 | stock_pin | 20.88 | 6 | 20.78 | 21.16 | 1.100x | 1.106 1.102 1.094 1.089 1.109 1.102 |
| pp2048 | qres_unpin | 31.64 | 6 | 31.32 | 31.82 | 1.668x | 1.660 1.684 1.662 1.655 1.671 1.673 |
| pp2048 | qres_pin | 33.58 | 6 | 33.47 | 33.73 | 1.770x | 1.761 1.776 1.761 1.764 1.790 1.767 |

