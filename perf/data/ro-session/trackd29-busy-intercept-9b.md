<!-- qwen35-9b Q4_K_M, busy_tot against rep count, PINNED arms only
     TESTS='-p 2048 -n 0'  PASSES=3  (-r lives in each arm's args)
     gguf=<data>/qwen35/Qwen3.5-9B-Q4_K_M.gguf (5680522464 bytes)
     stock_rN = true stock; qres_rN = ROCKET_QUANT_RESIDENT=auto -b 2048 -ub 2048
     (that compound IS the published 1.661x arm and IS trackd16's qres cell)
     all arms PIN_MASK=0xf0 -t 4; phi is defined at fixed pinning
     busy_tot = intercept + (r+1)*slope; llama-bench runs one internal warm-up plus r reps
     phi = 1 - slope_qres/slope_stock ; ingest = intercept_qres - intercept_stock
     PREDICTION 2026-09-03c: per-MB multiple 1.13-1.25 of the f16 route's 3.98 ms/MB,
       ingest 59-66 core-s over 13184 MB, phi 0.664-0.671, host share 0.596-0.603
       RIVALS: phi 0.622 (read-side), 0.627 (0.73 cores), 0.600 (no contamination)
       exact agreement of the two published rows would need a multiple of 1.40
     IMPORT CHECK: trackd16's PINNED ratio is 1.608x; a disagreement voids the a import
     CONTROL: every qres arm must place 200 weights / 13184 MB; every stock arm zero
-->
<!-- RESULT, three rotated passes, 18 of 18 DATA rows, 0 failed arms, pfn_zero_frac=0.0000 and
     pmu_enabled=100 on all eighteen. PLACEMENT CONTROL PASSES: 9 of 9 qres arms place the
     identical 200 weights / 13184 MB at 100% resident, 0 of 9 stock arms place anything, so the
     regression fits one configuration rather than three.
     IMPORT CHECK PASSES: this campaign's PINNED ratio is 1.6068 against trackd16's 1.608x,
     0.07% apart, so importing a = 0.4003 from that campaign is licensed.
       stock = 261 + 23624*(r+1) jiffies      wall = 0.6 + 98.9*(r+1) s
       qres  = 6228 +  7927*(r+1) jiffies     wall = 37.9 + 61.0*(r+1) s
       phi    = 1 - 7927/23624 = 0.6645 +- 0.0015  (per pass 0.664 / 0.662 / 0.667)
       ingest = 59.7 core-seconds                  (per pass 61.4 / 56.1 / 61.5)
       A76-only gives phi 0.6658 and the same 59.7 core-s, 0.2% apart, as it must in a pinned pair
     PREDICTION 2026-09-03c HIT ON ALL FIVE NUMBERS, AND ALL FOUR RIVALS ARE REFUTED.
       per resident MB   predicted 1.13-1.25x of the f16 route's 3.98 ms -> 4.53 ms = 1.14x
       ingest            predicted 59-66 core-seconds                    -> 59.7
       phi               predicted 0.664-0.671                           -> 0.6645
       host share        predicted 0.596-0.603                           -> 0.6024
       the 13% gap       predicted to close to about 2%, not to zero     -> 1.9%
       RIVAL read-side dominated, phi 0.622                  REFUTED
       RIVAL "the ingest runs at the 12B's 0.73 cores", 0.627 REFUTED -- it runs at 1.60 cores here
       RIVAL no contamination at all, phi 0.600               REFUTED
       RIVAL D, decode at the -ub knob's rate, multiple 1.80, phi 0.705  REFUTED
     PER-MB TRANSFERS AND CORES-OVER-A-WINDOW DOES NOT. The f16 route's ingest runs at 0.73 cores
     over 94.9 s and this one at 1.60 cores over 37.3 s, a factor of 2.2 apart, while the two
     agree to 14% per resident MB. The decode's own cost is 59.7 - 52.5 = 7.2 core-seconds over
     13184 MB, inside the 6.6-13.2 first estimate -- so the amendment that raised it to ~42
     core-seconds from the -ub knob's wall was the part that was wrong, and the -ub anchor
     conflates the dequant with the per-call pack it also removes.
     THE TWO 9B ROWS NOW AGREE TO 1.9%: 0.6024 from quant residency against 0.591 from
     -b 2048 -ub 2048. They are NESTED, so this is a two-point collinearity test through the
     origin, not two independent knobs.
-->
== qwen35-9b-busy-intercept  Thu Sep  3 04:09:28 UTC 2026 ==
### qwen35-9b-busy-intercept [stock_r1] pass 1  env='PIN_MASK=0xf0'  args='-t 4 -r 1'  04:10:28  clk=600 MHz  MemAvail=31430584 kB
<!--PRED 1	stock_r1	memavail_kb=31436276	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9009,12017,10691,7528,6436,5588,4846,3978,4171,3393,2442-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         21.22 ± 0.00 |

build: 171974745 (10558)
<!--RO 1	stock_r1	wall_s=196	busy=132,81,102,118,12527,12027,10964,11338	busy_tot=47289	busy_little_share=0.0092	a55_cpu_cycles=17097804238	a55_inst_retired=3601862251	context_switches=3924758	a76_cpu_cycles=1014217050719	a76_l3d_cache_refill=8007350314	a76_l2d_cache_refill=4291098357	cpu_migrations=9374	page_faults=569179	a76_dtlb_walk=1147327297	a76_mem_access=357444811060	a76_inst_retired=2122647894104	a76_l1d_cache_refill=7859834770	a55_inst_share=0.0017	a76_ipc=2.093	l2ref_pki=2.022	l3ref_pki=3.772	l1dref_pki=3.703	memacc_pki=168.40	dtlbw_pki=0.5405	pmu_enabled=100.0	pmu_cpu_s=1560.2	who=at_s=6,rss_pg=1509891,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6957	l2color_cv=0.0077	l3color_cv=0.0161	contig_frac=0.8532	mean_run=6.7	vapa16=0.2722	gib_regions=28	anon_pg=23041	anon_l2cv=0.0025	anon_contig=0.8549	anon_run=6.8	anon_gib=26	file_pg=24576	file_l2cv=0.0062	file_contig=0.9753	file_run=37.6	file_gib=27	other_pg=20774	other_l2cv=0.0249	other_contig=0.7067	other_run=3.4	other_gib=28	ro_cost_ms=91-->
<!--DATA 1	stock_r1	pp2048	21.22-->

### qwen35-9b-busy-intercept [stock_r2] pass 1  env='PIN_MASK=0xf0'  args='-t 4 -r 2'  04:15:09  clk=600 MHz  MemAvail=31457428 kB
<!--PRED 1	stock_r2	memavail_kb=31457136	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7291,12011,10603,7468,6389,5634,4882,3978,4103,3436,2442-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.79 ± 0.03 |

build: 171974745 (10558)
<!--RO 1	stock_r2	wall_s=299	busy=124,168,147,176,18625,18543,16609,16822	busy_tot=71214	busy_little_share=0.0086	a55_cpu_cycles=24627670084	a55_inst_retired=5010742895	context_switches=5878766	a76_cpu_cycles=1527870362446	a76_l3d_cache_refill=12867743562	a76_l2d_cache_refill=6342062437	cpu_migrations=13652	page_faults=575976	a76_dtlb_walk=1716185693	a76_mem_access=534410976603	a76_inst_retired=3179015201465	a76_l1d_cache_refill=11733983171	a55_inst_share=0.0016	a76_ipc=2.081	l2ref_pki=1.995	l3ref_pki=4.048	l1dref_pki=3.691	memacc_pki=168.11	dtlbw_pki=0.5398	pmu_enabled=100.0	pmu_cpu_s=2384.0	who=at_s=6,rss_pg=1509890,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6983	l2color_cv=0.0079	l3color_cv=0.0159	contig_frac=0.8450	mean_run=6.4	vapa16=0.0136	gib_regions=28	anon_pg=23042	anon_l2cv=0.0014	anon_contig=0.9186	anon_run=12.0	anon_gib=27	file_pg=24576	file_l2cv=0.0017	file_contig=0.9805	file_run=46.7	file_gib=27	other_pg=21026	other_l2cv=0.0267	other_contig=0.6057	other_run=2.5	other_gib=28	ro_cost_ms=107-->
<!--DATA 1	stock_r2	pp2048	20.79-->

### qwen35-9b-busy-intercept [stock_r4] pass 1  env='PIN_MASK=0xf0'  args='-t 4 -r 4'  04:22:25  clk=600 MHz  MemAvail=31463216 kB
<!--PRED 1	stock_r4	memavail_kb=31463216	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9629,11770,10044,7482,6350,5616,4894,3990,4074,3452,2442-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.76 ± 0.02 |

build: 171974745 (10558)
<!--RO 1	stock_r4	wall_s=496	busy=175,239,292,263,30812,30507,27620,28688	busy_tot=118596	busy_little_share=0.0082	a55_cpu_cycles=40630531657	a55_inst_retired=8233120735	context_switches=9810426	a76_cpu_cycles=2544043897263	a76_l3d_cache_refill=21354966054	a76_l2d_cache_refill=10494564858	cpu_migrations=23069	page_faults=594993	a76_dtlb_walk=2368277454	a76_mem_access=889028048312	a76_inst_retired=5293223795665	a76_l1d_cache_refill=19444841048	a55_inst_share=0.0016	a76_ipc=2.081	l2ref_pki=1.983	l3ref_pki=4.034	l1dref_pki=3.674	memacc_pki=167.96	dtlbw_pki=0.4474	pmu_enabled=100.0	pmu_cpu_s=3960.5	who=at_s=6,rss_pg=1509890,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6890	l2color_cv=0.0054	l3color_cv=0.0170	contig_frac=0.8469	mean_run=6.5	vapa16=0.0921	gib_regions=28	anon_pg=22019	anon_l2cv=0.0065	anon_contig=0.8798	anon_run=8.2	anon_gib=27	file_pg=24576	file_l2cv=0.0024	file_contig=0.9675	file_run=29.1	file_gib=27	other_pg=21132	other_l2cv=0.0184	other_contig=0.6725	other_run=3.0	other_gib=28	ro_cost_ms=79-->
<!--DATA 1	stock_r4	pp2048	20.76-->

### qwen35-9b-busy-intercept [qres_r1] pass 1  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4 -r 1'  04:31:52  clk=600 MHz  MemAvail=31226720 kB
<!--PRED 1	qres_r1	memavail_kb=31226720	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6999,8351,8934,7973,6876,6121,5257,4275,4138,3691,2176-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         33.61 ± 0.00 |

build: 171974745 (10558)
<!--RO 1	qres_r1	wall_s=160	busy=52,56,49,56,5168,6449,5634,4527	busy_tot=21991	busy_little_share=0.0097	a55_cpu_cycles=8818992251	a55_inst_retired=1790756791	context_switches=3600538	a76_cpu_cycles=474074934911	a76_l3d_cache_refill=6177677458	a76_l2d_cache_refill=2734268520	cpu_migrations=2629	page_faults=11403730	a76_dtlb_walk=1870691812	a76_mem_access=234465043718	a76_inst_retired=808872794512	a76_l1d_cache_refill=5565117062	a55_inst_share=0.0022	a76_ipc=1.706	l2ref_pki=3.380	l3ref_pki=7.637	l1dref_pki=6.880	memacc_pki=289.87	dtlbw_pki=2.3127	pmu_enabled=100.0	pmu_cpu_s=1274.4	who=at_s=10,rss_pg=1626685,settled=2	pfn_zero_frac=0.0000	maps=6	sampled=132096	present_frac=0.8072	l2color_cv=0.0062	l3color_cv=0.0093	contig_frac=0.5451	mean_run=2.2	vapa16=0.0868	gib_regions=28	anon_pg=64278	anon_l2cv=0.0105	anon_contig=0.3147	anon_run=1.5	anon_gib=28	file_pg=24576	file_l2cv=0.0005	file_contig=0.9813	file_run=48.6	file_gib=28	other_pg=17774	other_l2cv=0.0035	other_contig=0.7751	other_run=4.4	other_gib=26	ro_cost_ms=129-->
<!--DATA 1	qres_r1	pp2048	33.61-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20904MB (MemAvailable 30440MB - reserve 9535MB, no swap)

### qwen35-9b-busy-intercept [qres_r2] pass 1  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4 -r 2'  04:36:00  clk=600 MHz  MemAvail=31354752 kB
<!--PRED 1	qres_r2	memavail_kb=31354752	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10212,12799,10993,8121,7086,6382,5460,4458,4349,3720,2072-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         33.45 ± 0.06 |

build: 171974745 (10558)
<!--RO 1	qres_r2	wall_s=221	busy=75,81,69,82,7130,7627,7923,6838	busy_tot=29825	busy_little_share=0.0103	a55_cpu_cycles=12428125572	a55_inst_retired=2403785356	context_switches=5409734	a76_cpu_cycles=642933039598	a76_l3d_cache_refill=8588807740	a76_l2d_cache_refill=3915775054	cpu_migrations=3922	page_faults=11404416	a76_dtlb_walk=2782852584	a76_mem_access=329614339618	a76_inst_retired=1072904889682	a76_l1d_cache_refill=8087219179	a55_inst_share=0.0022	a76_ipc=1.669	l2ref_pki=3.650	l3ref_pki=8.005	l1dref_pki=7.538	memacc_pki=307.22	dtlbw_pki=2.5938	pmu_enabled=100.0	pmu_cpu_s=1760.8	who=at_s=10,rss_pg=1626685,settled=2	pfn_zero_frac=0.0000	maps=6	sampled=132096	present_frac=0.8105	l2color_cv=0.0078	l3color_cv=0.0111	contig_frac=0.6156	mean_run=2.6	vapa16=0.0349	gib_regions=28	anon_pg=64874	anon_l2cv=0.0006	anon_contig=0.4886	anon_run=2.0	anon_gib=27	file_pg=24576	file_l2cv=0.0078	file_contig=0.9792	file_run=44.0	file_gib=27	other_pg=17610	other_l2cv=0.0461	other_contig=0.5756	other_run=2.3	other_gib=28	ro_cost_ms=142-->
<!--DATA 1	qres_r2	pp2048	33.45-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21001MB (MemAvailable 30536MB - reserve 9535MB, no swap)

### qwen35-9b-busy-intercept [qres_r4] pass 1  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4 -r 4'  04:41:38  clk=600 MHz  MemAvail=31462524 kB
<!--PRED 1	qres_r4	memavail_kb=31462528	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10376,13106,11905,8581,7541,6937,5924,4840,4774,3924,1781-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         33.42 ± 0.19 |

build: 171974745 (10558)
<!--RO 1	qres_r4	wall_s=344	busy=109,114,148,128,11415,11401,11570,11007	busy_tot=45892	busy_little_share=0.0109	a55_cpu_cycles=20638766431	a55_inst_retired=4033140605	context_switches=9009386	a76_cpu_cycles=991530255446	a76_l3d_cache_refill=13477892047	a76_l2d_cache_refill=6315208722	cpu_migrations=6300	page_faults=11414049	a76_dtlb_walk=4617967377	a76_mem_access=520269849176	a76_inst_retired=1602205834420	a76_l1d_cache_refill=13081097621	a55_inst_share=0.0025	a76_ipc=1.616	l2ref_pki=3.942	l3ref_pki=8.412	l1dref_pki=8.164	memacc_pki=324.72	dtlbw_pki=2.8823	pmu_enabled=100.0	pmu_cpu_s=2745.5	who=at_s=10,rss_pg=1626685,settled=2	pfn_zero_frac=0.0000	maps=6	sampled=132096	present_frac=0.8091	l2color_cv=0.0062	l3color_cv=0.0101	contig_frac=0.5135	mean_run=2.1	vapa16=0.0495	gib_regions=28	anon_pg=64686	anon_l2cv=0.0005	anon_contig=0.3063	anon_run=1.4	anon_gib=27	file_pg=24576	file_l2cv=0.0041	file_contig=0.9724	file_run=33.9	file_gib=28	other_pg=17620	other_l2cv=0.0348	other_contig=0.6341	other_run=2.7	other_gib=28	ro_cost_ms=122-->
<!--DATA 1	qres_r4	pp2048	33.42-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21077MB (MemAvailable 30612MB - reserve 9535MB, no swap)

### qwen35-9b-busy-intercept [stock_r2] pass 2  env='PIN_MASK=0xf0'  args='-t 4 -r 2'  04:48:47  clk=600 MHz  MemAvail=31448612 kB
<!--PRED 2	stock_r2	memavail_kb=31448612	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10759,13091,11811,8361,7115,6741,5751,4715,4650,3913,1855-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.71 ± 0.08 |

build: 171974745 (10558)
<!--RO 2	stock_r2	wall_s=300	busy=110,150,153,176,18484,18477,16699,17004	busy_tot=71253	busy_little_share=0.0083	a55_cpu_cycles=24546791174	a55_inst_retired=4962309912	context_switches=5886371	a76_cpu_cycles=1528107518215	a76_l3d_cache_refill=12817560778	a76_l2d_cache_refill=6320371923	cpu_migrations=14008	page_faults=576614	a76_dtlb_walk=1715341657	a76_mem_access=534633780228	a76_inst_retired=3179734747981	a76_l1d_cache_refill=11722794150	a55_inst_share=0.0016	a76_ipc=2.081	l2ref_pki=1.988	l3ref_pki=4.031	l1dref_pki=3.687	memacc_pki=168.14	dtlbw_pki=0.5395	pmu_enabled=100.0	pmu_cpu_s=2388.7	who=at_s=6,rss_pg=1509893,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6958	l2color_cv=0.0105	l3color_cv=0.0168	contig_frac=0.8510	mean_run=6.6	vapa16=0.3579	gib_regions=28	anon_pg=23041	anon_l2cv=0.0023	anon_contig=0.8802	anon_run=8.2	anon_gib=26	file_pg=24576	file_l2cv=0.0017	file_contig=0.9774	file_run=40.8	file_gib=28	other_pg=20786	other_l2cv=0.0333	other_contig=0.6691	other_run=3.0	other_gib=28	ro_cost_ms=90-->
<!--DATA 2	stock_r2	pp2048	20.71-->

### qwen35-9b-busy-intercept [stock_r4] pass 2  env='PIN_MASK=0xf0'  args='-t 4 -r 4'  04:56:03  clk=600 MHz  MemAvail=31467304 kB
<!--PRED 2	stock_r4	memavail_kb=31467304	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9821,12544,11480,8155,6733,6411,5457,4469,4437,3743,2068-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.86 ± 0.06 |

build: 171974745 (10558)
<!--RO 2	stock_r4	wall_s=494	busy=235,282,245,208,30739,30707,27669,28088	busy_tot=118173	busy_little_share=0.0082	a55_cpu_cycles=40629713813	a55_inst_retired=8211428319	context_switches=9796567	a76_cpu_cycles=2541898716047	a76_l3d_cache_refill=21259404311	a76_l2d_cache_refill=10526486737	cpu_migrations=23076	page_faults=594423	a76_dtlb_walk=2355079840	a76_mem_access=888996310452	a76_inst_retired=5293140214682	a76_l1d_cache_refill=19393517340	a55_inst_share=0.0015	a76_ipc=2.082	l2ref_pki=1.989	l3ref_pki=4.016	l1dref_pki=3.664	memacc_pki=167.95	dtlbw_pki=0.4449	pmu_enabled=100.0	pmu_cpu_s=3945.1	who=at_s=7,rss_pg=1509893,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6988	l2color_cv=0.0058	l3color_cv=0.0181	contig_frac=0.8723	mean_run=7.7	vapa16=0.1690	gib_regions=28	anon_pg=23041	anon_l2cv=0.0012	anon_contig=0.9083	anon_run=10.7	anon_gib=27	file_pg=24576	file_l2cv=0.0039	file_contig=0.9809	file_run=47.5	file_gib=28	other_pg=21082	other_l2cv=0.0173	other_contig=0.7063	other_run=3.4	other_gib=28	ro_cost_ms=95-->
<!--DATA 2	stock_r4	pp2048	20.86-->

### qwen35-9b-busy-intercept [qres_r1] pass 2  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4 -r 1'  05:05:30  clk=600 MHz  MemAvail=31338844 kB
<!--PRED 2	qres_r1	memavail_kb=31338844	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10721,12008,10964,8285,6997,6415,5532,4530,4311,3759,2045-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         33.73 ± 0.00 |

build: 171974745 (10558)
<!--RO 2	qres_r1	wall_s=159	busy=80,49,39,51,5159,6137,5939,4652	busy_tot=22106	busy_little_share=0.0099	a55_cpu_cycles=8842270376	a55_inst_retired=1732840321	context_switches=3601381	a76_cpu_cycles=474940354718	a76_l3d_cache_refill=6069944527	a76_l2d_cache_refill=2731444011	cpu_migrations=2563	page_faults=11401627	a76_dtlb_walk=1839656580	a76_mem_access=234315281389	a76_inst_retired=808146581396	a76_l1d_cache_refill=5578784100	a55_inst_share=0.0021	a76_ipc=1.702	l2ref_pki=3.380	l3ref_pki=7.511	l1dref_pki=6.903	memacc_pki=289.94	dtlbw_pki=2.2764	pmu_enabled=100.0	pmu_cpu_s=1267.8	who=at_s=10,rss_pg=1626686,settled=2	pfn_zero_frac=0.0000	maps=6	sampled=132096	present_frac=0.8048	l2color_cv=0.0058	l3color_cv=0.0080	contig_frac=0.4724	mean_run=1.9	vapa16=0.0360	gib_regions=28	anon_pg=64200	anon_l2cv=0.0009	anon_contig=0.2643	anon_run=1.4	anon_gib=27	file_pg=24576	file_l2cv=0.0021	file_contig=0.9785	file_run=42.7	file_gib=28	other_pg=17537	other_l2cv=0.0364	other_contig=0.5252	other_run=2.1	other_gib=28	ro_cost_ms=122-->
<!--DATA 2	qres_r1	pp2048	33.73-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21007MB (MemAvailable 30543MB - reserve 9535MB, no swap)

### qwen35-9b-busy-intercept [qres_r2] pass 2  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4 -r 2'  05:09:37  clk=600 MHz  MemAvail=31400508 kB
<!--PRED 2	qres_r2	memavail_kb=31400764	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10558,13639,10608,8707,7709,7032,6037,4962,4772,3991,1708-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         33.47 ± 0.00 |

build: 171974745 (10558)
<!--RO 2	qres_r2	wall_s=221	busy=83,75,87,76,7269,8143,7713,6585	busy_tot=30031	busy_little_share=0.0107	a55_cpu_cycles=12895888642	a55_inst_retired=2532193276	context_switches=5401740	a76_cpu_cycles=647629924030	a76_l3d_cache_refill=8552345560	a76_l2d_cache_refill=3930746036	cpu_migrations=3753	page_faults=11406339	a76_dtlb_walk=2782827899	a76_mem_access=329739970963	a76_inst_retired=1073310325810	a76_l1d_cache_refill=8068553900	a55_inst_share=0.0024	a76_ipc=1.657	l2ref_pki=3.662	l3ref_pki=7.968	l1dref_pki=7.517	memacc_pki=307.22	dtlbw_pki=2.5928	pmu_enabled=100.0	pmu_cpu_s=1764.3	who=at_s=10,rss_pg=1626686,settled=2	pfn_zero_frac=0.0000	maps=6	sampled=132096	present_frac=0.8086	l2color_cv=0.0044	l3color_cv=0.0095	contig_frac=0.5606	mean_run=2.3	vapa16=0.0196	gib_regions=28	anon_pg=64464	anon_l2cv=0.0013	anon_contig=0.4751	anon_run=1.9	anon_gib=28	file_pg=24576	file_l2cv=0.0018	file_contig=0.9766	file_run=39.6	file_gib=27	other_pg=17770	other_l2cv=0.0291	other_contig=0.2950	other_run=1.4	other_gib=28	ro_cost_ms=126-->
<!--DATA 2	qres_r2	pp2048	33.47-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21082MB (MemAvailable 30617MB - reserve 9535MB, no swap)

### qwen35-9b-busy-intercept [qres_r4] pass 2  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4 -r 4'  05:15:15  clk=600 MHz  MemAvail=31456536 kB
<!--PRED 2	qres_r4	memavail_kb=31456536	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9329,13209,12022,8555,7536,6962,5932,4919,4818,3980,1730-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         33.66 ± 0.16 |

build: 171974745 (10558)
<!--RO 2	qres_r4	wall_s=342	busy=149,109,145,105,11254,12108,11394,10690	busy_tot=45954	busy_little_share=0.0111	a55_cpu_cycles=20813506097	a55_inst_retired=4091642140	context_switches=8995072	a76_cpu_cycles=992843631010	a76_l3d_cache_refill=13463585090	a76_l2d_cache_refill=6342455387	cpu_migrations=5990	page_faults=11412293	a76_dtlb_walk=4619243301	a76_mem_access=520138165513	a76_inst_retired=1601928035742	a76_l1d_cache_refill=13097949465	a55_inst_share=0.0025	a76_ipc=1.613	l2ref_pki=3.959	l3ref_pki=8.405	l1dref_pki=8.176	memacc_pki=324.70	dtlbw_pki=2.8836	pmu_enabled=100.0	pmu_cpu_s=2730.2	who=at_s=10,rss_pg=1626686,settled=2	pfn_zero_frac=0.0000	maps=6	sampled=132096	present_frac=0.8082	l2color_cv=0.0038	l3color_cv=0.0072	contig_frac=0.5956	mean_run=2.5	vapa16=0.0459	gib_regions=28	anon_pg=64456	anon_l2cv=0.0008	anon_contig=0.4402	anon_run=1.8	anon_gib=27	file_pg=24576	file_l2cv=0.0037	file_contig=0.9785	file_run=42.7	file_gib=27	other_pg=17728	other_l2cv=0.0215	other_contig=0.6300	other_run=2.7	other_gib=28	ro_cost_ms=160-->
<!--DATA 2	qres_r4	pp2048	33.66-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21079MB (MemAvailable 30615MB - reserve 9535MB, no swap)

### qwen35-9b-busy-intercept [stock_r1] pass 2  env='PIN_MASK=0xf0'  args='-t 4 -r 1'  05:21:57  clk=600 MHz  MemAvail=31436480 kB
<!--PRED 2	stock_r1	memavail_kb=31436480	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9201,12413,11535,8397,7276,6850,5827,4809,4696,3888,1834-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.91 ± 0.00 |

build: 171974745 (10558)
<!--RO 2	stock_r1	wall_s=198	busy=87,87,122,110,12325,12442,11129,11267	busy_tot=47569	busy_little_share=0.0085	a55_cpu_cycles=16681562058	a55_inst_retired=3412862161	context_switches=3921139	a76_cpu_cycles=1019703256538	a76_l3d_cache_refill=8516990642	a76_l2d_cache_refill=4230312256	cpu_migrations=9337	page_faults=568305	a76_dtlb_walk=1143831955	a76_mem_access=357261992776	a76_inst_retired=2122596738880	a76_l1d_cache_refill=7803468717	a55_inst_share=0.0016	a76_ipc=2.082	l2ref_pki=1.993	l3ref_pki=4.013	l1dref_pki=3.676	memacc_pki=168.31	dtlbw_pki=0.5389	pmu_enabled=100.0	pmu_cpu_s=1584.6	who=at_s=6,rss_pg=1509893,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6977	l2color_cv=0.0046	l3color_cv=0.0148	contig_frac=0.8462	mean_run=6.4	vapa16=0.0604	gib_regions=28	anon_pg=23041	anon_l2cv=0.0021	anon_contig=0.9022	anon_run=10.0	anon_gib=26	file_pg=24576	file_l2cv=0.0031	file_contig=0.9775	file_run=40.9	file_gib=27	other_pg=20973	other_l2cv=0.0174	other_contig=0.6308	other_run=2.7	other_gib=28	ro_cost_ms=80-->
<!--DATA 2	stock_r1	pp2048	20.91-->

### qwen35-9b-busy-intercept [stock_r4] pass 3  env='PIN_MASK=0xf0'  args='-t 4 -r 4'  05:27:33  clk=600 MHz  MemAvail=31470072 kB
<!--PRED 3	stock_r4	memavail_kb=31469488	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9505,12468,11571,8190,6816,6471,5569,4578,4545,3780,1999-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.83 ± 0.07 |

build: 171974745 (10558)
<!--RO 3	stock_r4	wall_s=494	busy=215,248,248,237,30814,30589,27551,28383	busy_tot=118285	busy_little_share=0.0080	a55_cpu_cycles=40924126108	a55_inst_retired=8222279073	context_switches=9795443	a76_cpu_cycles=2541881208002	a76_l3d_cache_refill=21268759028	a76_l2d_cache_refill=10518415056	cpu_migrations=22993	page_faults=596002	a76_dtlb_walk=2853438765	a76_mem_access=889157864382	a76_inst_retired=5293484836779	a76_l1d_cache_refill=19465400018	a55_inst_share=0.0016	a76_ipc=2.083	l2ref_pki=1.987	l3ref_pki=4.018	l1dref_pki=3.677	memacc_pki=167.97	dtlbw_pki=0.5390	pmu_enabled=100.0	pmu_cpu_s=3950.3	who=at_s=6,rss_pg=1509886,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6982	l2color_cv=0.0099	l3color_cv=0.0181	contig_frac=0.8685	mean_run=7.5	vapa16=0.0286	gib_regions=28	anon_pg=23041	anon_l2cv=0.0017	anon_contig=0.8762	anon_run=8.0	anon_gib=26	file_pg=24576	file_l2cv=0.0057	file_contig=0.9814	file_run=48.7	file_gib=27	other_pg=21021	other_l2cv=0.0317	other_contig=0.7279	other_run=3.7	other_gib=28	ro_cost_ms=96-->
<!--DATA 3	stock_r4	pp2048	20.83-->

### qwen35-9b-busy-intercept [qres_r1] pass 3  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4 -r 1'  05:36:59  clk=600 MHz  MemAvail=31058696 kB
<!--PRED 3	qres_r1	memavail_kb=31058696	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9715,10482,9368,8275,7171,6611,5598,4561,4188,3804,1978-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         33.45 ± 0.00 |

build: 171974745 (10558)
<!--RO 3	qres_r1	wall_s=160	busy=67,46,72,60,5347,6205,5347,5066	busy_tot=22210	busy_little_share=0.0110	a55_cpu_cycles=8829867390	a55_inst_retired=1755672929	context_switches=3601437	a76_cpu_cycles=476459301761	a76_l3d_cache_refill=6146409651	a76_l2d_cache_refill=2719134363	cpu_migrations=2522	page_faults=11401380	a76_dtlb_walk=1872222320	a76_mem_access=234337842803	a76_inst_retired=808438729042	a76_l1d_cache_refill=5555314031	a55_inst_share=0.0022	a76_ipc=1.697	l2ref_pki=3.363	l3ref_pki=7.603	l1dref_pki=6.872	memacc_pki=289.86	dtlbw_pki=2.3158	pmu_enabled=100.0	pmu_cpu_s=1272.6	who=at_s=10,rss_pg=1626687,settled=2	pfn_zero_frac=0.0000	maps=6	sampled=132096	present_frac=0.8047	l2color_cv=0.0030	l3color_cv=0.0107	contig_frac=0.3739	mean_run=1.6	vapa16=0.0415	gib_regions=28	anon_pg=64312	anon_l2cv=0.0050	anon_contig=0.2156	anon_run=1.3	anon_gib=28	file_pg=24576	file_l2cv=0.0022	file_contig=0.9690	file_run=30.4	file_gib=28	other_pg=17406	other_l2cv=0.0030	other_contig=0.1182	other_run=1.1	other_gib=27	ro_cost_ms=121-->
<!--DATA 3	qres_r1	pp2048	33.45-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20798MB (MemAvailable 30333MB - reserve 9535MB, no swap)

### qwen35-9b-busy-intercept [qres_r2] pass 3  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4 -r 2'  05:41:06  clk=600 MHz  MemAvail=31303628 kB
<!--PRED 3	qres_r2	memavail_kb=31303632	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10423,12412,12187,8472,7430,6835,5875,4801,4589,3952,1788-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         33.45 ± 0.02 |

build: 171974745 (10558)
<!--RO 3	qres_r2	wall_s=221	busy=48,85,111,85,7416,7786,7650,6896	busy_tot=30077	busy_little_share=0.0109	a55_cpu_cycles=13034302826	a55_inst_retired=2628822470	context_switches=5407234	a76_cpu_cycles=648547638624	a76_l3d_cache_refill=8612219129	a76_l2d_cache_refill=3908234434	cpu_migrations=3799	page_faults=11407544	a76_dtlb_walk=2787408940	a76_mem_access=329623564765	a76_inst_retired=1073063263076	a76_l1d_cache_refill=8057904954	a55_inst_share=0.0024	a76_ipc=1.655	l2ref_pki=3.642	l3ref_pki=8.026	l1dref_pki=7.509	memacc_pki=307.18	dtlbw_pki=2.5976	pmu_enabled=100.0	pmu_cpu_s=1765.1	who=at_s=10,rss_pg=1626686,settled=2	pfn_zero_frac=0.0000	maps=6	sampled=132096	present_frac=0.8053	l2color_cv=0.0010	l3color_cv=0.0031	contig_frac=0.5566	mean_run=2.2	vapa16=0.2375	gib_regions=28	anon_pg=64282	anon_l2cv=0.0016	anon_contig=0.4115	anon_run=1.7	anon_gib=28	file_pg=24576	file_l2cv=0.0054	file_contig=0.9759	file_run=38.4	file_gib=28	other_pg=17517	other_l2cv=0.0031	other_contig=0.5009	other_run=2.0	other_gib=27	ro_cost_ms=141-->
<!--DATA 3	qres_r2	pp2048	33.45-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21017MB (MemAvailable 30552MB - reserve 9535MB, no swap)

### qwen35-9b-busy-intercept [qres_r4] pass 3  env='ROCKET_QUANT_RESIDENT=auto PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4 -r 4'  05:46:46  clk=600 MHz  MemAvail=31452296 kB
<!--PRED 3	qres_r4	memavail_kb=31452296	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10033,13250,11882,8693,7640,6975,5982,4904,4802,3942,1747-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         33.63 ± 0.08 |

build: 171974745 (10558)
<!--RO 3	qres_r4	wall_s=342	busy=145,101,114,138,11319,11395,11010,11549	busy_tot=45771	busy_little_share=0.0109	a55_cpu_cycles=20373348492	a55_inst_retired=4066850359	context_switches=9003095	a76_cpu_cycles=990011684401	a76_l3d_cache_refill=13475531596	a76_l2d_cache_refill=6308824388	cpu_migrations=6135	page_faults=11411088	a76_dtlb_walk=4625725854	a76_mem_access=520170270913	a76_inst_retired=1602030613370	a76_l1d_cache_refill=13087848746	a55_inst_share=0.0025	a76_ipc=1.618	l2ref_pki=3.938	l3ref_pki=8.412	l1dref_pki=8.170	memacc_pki=324.69	dtlbw_pki=2.8874	pmu_enabled=100.0	pmu_cpu_s=2731.8	who=at_s=10,rss_pg=1626686,settled=2	pfn_zero_frac=0.0000	maps=6	sampled=132096	present_frac=0.8033	l2color_cv=0.0038	l3color_cv=0.0075	contig_frac=0.4480	mean_run=1.8	vapa16=0.0680	gib_regions=28	anon_pg=63976	anon_l2cv=0.0013	anon_contig=0.2004	anon_run=1.3	anon_gib=27	file_pg=24576	file_l2cv=0.0041	file_contig=0.9804	file_run=46.5	file_gib=27	other_pg=17559	other_l2cv=0.0229	other_contig=0.6047	other_run=2.5	other_gib=28	ro_cost_ms=163-->
<!--DATA 3	qres_r4	pp2048	33.63-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21083MB (MemAvailable 30618MB - reserve 9535MB, no swap)

### qwen35-9b-busy-intercept [stock_r1] pass 3  env='PIN_MASK=0xf0'  args='-t 4 -r 1'  05:53:28  clk=600 MHz  MemAvail=31445276 kB
<!--PRED 3	stock_r1	memavail_kb=31445276	anonhuge_kb=0	hugepagesz_kb=2048	buddy=11205,12016,11404,8449,7111,6666,5737,4688,4623,3873,1890-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.99 ± 0.00 |

build: 171974745 (10558)
<!--RO 3	stock_r1	wall_s=198	busy=115,95,115,78,12465,12236,10921,11463	busy_tot=47488	busy_little_share=0.0085	a55_cpu_cycles=16638891990	a55_inst_retired=3416116270	context_switches=3921335	a76_cpu_cycles=1019002626987	a76_l3d_cache_refill=8232751461	a76_l2d_cache_refill=4244458177	cpu_migrations=9437	page_faults=567239	a76_dtlb_walk=1147149641	a76_mem_access=357336853336	a76_inst_retired=2122304677681	a76_l1d_cache_refill=7830810551	a55_inst_share=0.0016	a76_ipc=2.083	l2ref_pki=2.000	l3ref_pki=3.879	l1dref_pki=3.690	memacc_pki=168.37	dtlbw_pki=0.5405	pmu_enabled=100.0	pmu_cpu_s=1581.6	who=at_s=6,rss_pg=1509893,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6975	l2color_cv=0.0074	l3color_cv=0.0124	contig_frac=0.8476	mean_run=6.5	vapa16=0.0200	gib_regions=28	anon_pg=23041	anon_l2cv=0.0017	anon_contig=0.8815	anon_run=8.3	anon_gib=27	file_pg=24576	file_l2cv=0.0045	file_contig=0.9779	file_run=41.7	file_gib=28	other_pg=20947	other_l2cv=0.0232	other_contig=0.6575	other_run=2.9	other_gib=28	ro_cost_ms=108-->
<!--DATA 3	stock_r1	pp2048	20.99-->

### qwen35-9b-busy-intercept [stock_r2] pass 3  env='PIN_MASK=0xf0'  args='-t 4 -r 2'  05:58:12  clk=600 MHz  MemAvail=31451256 kB
<!--PRED 3	stock_r2	memavail_kb=31451256	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9728,12794,10976,7994,6635,6295,5436,4433,4416,3747,2081-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |       4 |          pp2048 |         20.80 ± 0.02 |

build: 171974745 (10558)
<!--RO 3	stock_r2	wall_s=298	busy=147,172,145,127,18510,18548,16721,16837	busy_tot=71207	busy_little_share=0.0083	a55_cpu_cycles=24761040243	a55_inst_retired=5124029779	context_switches=5876731	a76_cpu_cycles=1528672897872	a76_l3d_cache_refill=12867604740	a76_l2d_cache_refill=6337088332	cpu_migrations=13876	page_faults=577039	a76_dtlb_walk=1715474130	a76_mem_access=534534716370	a76_inst_retired=3179735917182	a76_l1d_cache_refill=11712226185	a55_inst_share=0.0016	a76_ipc=2.080	l2ref_pki=1.993	l3ref_pki=4.047	l1dref_pki=3.683	memacc_pki=168.11	dtlbw_pki=0.5395	pmu_enabled=100.0	pmu_cpu_s=2380.2	who=at_s=6,rss_pg=1509892,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6989	l2color_cv=0.0044	l3color_cv=0.0134	contig_frac=0.8313	mean_run=5.9	vapa16=0.0308	gib_regions=28	anon_pg=23041	anon_l2cv=0.0014	anon_contig=0.8950	anon_run=9.4	anon_gib=27	file_pg=24576	file_l2cv=0.0039	file_contig=0.9796	file_run=44.8	file_gib=27	other_pg=21092	other_l2cv=0.0182	other_contig=0.5887	other_run=2.4	other_gib=28	ro_cost_ms=146-->
<!--DATA 3	stock_r2	pp2048	20.80-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock_r1]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock_r1 | 21.04 | 3 | 20.91 | 21.22 | -- | -- |
| pp2048 | stock_r2 | 20.77 | 3 | 20.71 | 20.80 | 0.987x | 0.980 0.990 0.991 |
| pp2048 | stock_r4 | 20.82 | 3 | 20.76 | 20.86 | 0.989x | 0.978 0.998 0.992 |
| pp2048 | qres_r1 | 33.60 | 3 | 33.45 | 33.73 | 1.597x | 1.584 1.613 1.594 |
| pp2048 | qres_r2 | 33.46 | 3 | 33.45 | 33.47 | 1.590x | 1.576 1.601 1.594 |
| pp2048 | qres_r4 | 33.57 | 3 | 33.42 | 33.66 | 1.596x | 1.575 1.610 1.602 |

