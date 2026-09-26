<!-- gemma4-12b F16, busy_tot against rep count, PINNED arms only
     TESTS='-p 2048 -n 0'  PASSES=3  (-r lives in each arm's args)
     gguf=<data>/gemma4/gemma-4-12b-it-F16.gguf (23832065184 bytes)
     stream_rN = the shipping default; res_rN = ROCKET_F16_RESIDENT=auto
     all arms PIN_MASK=0xf0 -t 4; phi is defined at fixed pinning
     busy_tot = intercept + (r+1)*slope; llama-bench runs one internal warm-up plus r reps
     phi = 1 - slope_res/slope_stream ; ingest = intercept_res - intercept_stream
     prediction 2026-09-02e: phi=0.28 (0.16-0.40), ingest=120 core-s (60-190)
     CONTROL: every res arm must place the SAME weight count; every stream arm zero
-->
<!-- RESULT, fitted over all three passes:
       stream: busy_tot = 15967 + 15540*r jiffies   wall_s = 101.2 + 97.1*r
       res:    busy_tot = 19638 + 12010*r jiffies   wall_s = 199.3 + 100.4*r
     Both busy_tot lines linear to under 0.4% residual; the resident arm's WALL is not
     (its non-timed portion moves 65-143 s across the three rep counts).
       phi    = 1 - 12010/15540 = 0.227 +- 0.018   (per-pass 0.196 / 0.258 / 0.227)
       ingest = 72 core-seconds = 0.73 cores over the 98 s of extra fixed wall
     Prediction 2026-09-02e HIT on both bands, low in each. The contaminated busy_tot ratio
     gave 0.087; the interaction requires 0.280 +- 0.060, so the measured value is 0.85 se
     from what the model needs and 7.8 se from the contaminated one. The cell's residual
     against the forced host-share model moves from -3.5 se to -0.9 se.
     CONTROL PASSED: 9 of 9 res arms at 286 weights / 18078 MB, 0 of 9 stream arms placing.
     18 of 18 DATA rows, 0 failed arms, pfn_zero_frac=0.0000 and pmu_enabled=100 on all 18.
-->
== gemma4-12b-busy-intercept  Wed Sep  2 22:35:51 UTC 2026 ==
### gemma4-12b-busy-intercept [stream_r1] pass 1  env='PIN_MASK=0xf0'  args='-t 4 -r 1'  22:37:15  clk=600 MHz  MemAvail=31530944 kB
<!--PRED 1	stream_r1	memavail_kb=31530952	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3945,4820,9646,8142,5766,5141,4085,3748,2491,174,251-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         21.01 ± 0.00 |

build: 171974745 (10558)
<!--RO 1	stream_r1	wall_s=200	busy=167,199,223,173,9417,7032,6514,7792	busy_tot=31517	busy_little_share=0.0242	a55_cpu_cycles=29222327534	a55_inst_retired=6129449761	context_switches=6483638	a76_cpu_cycles=679837843498	a76_l3d_cache_refill=10032753629	a76_l2d_cache_refill=4739487892	cpu_migrations=13195	page_faults=940882	a76_dtlb_walk=551403472	a76_mem_access=187988498165	a76_inst_retired=866458248451	a76_l1d_cache_refill=11214864956	a55_inst_share=0.0070	a76_ipc=1.275	l2ref_pki=5.470	l3ref_pki=11.579	l1dref_pki=12.943	memacc_pki=216.96	dtlbw_pki=0.6364	pmu_enabled=100.0	pmu_cpu_s=1594.9	who=at_s=8,rss_pg=6044868,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8338	l2color_cv=0.0250	l3color_cv=0.0407	contig_frac=0.8913	mean_run=9.0	vapa16=0.2080	gib_regions=32	anon_pg=12932	anon_l2cv=0.0222	anon_contig=0.9583	anon_run=22.9	anon_gib=29	file_pg=24576	file_l2cv=0.0019	file_contig=0.9933	file_run=115.9	file_gib=29	other_pg=23970	other_l2cv=0.0582	other_contig=0.7505	other_run=4.0	other_gib=32	ro_cost_ms=84-->
<!--DATA 1	stream_r1	pp2048	21.01-->

### gemma4-12b-busy-intercept [stream_r2] pass 1  env='PIN_MASK=0xf0'  args='-t 4 -r 2'  22:42:28  clk=600 MHz  MemAvail=31520288 kB
<!--PRED 1	stream_r2	memavail_kb=31520288	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7587,8128,9371,8059,5777,5126,3859,3749,2501,181,248-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         21.35 ± 0.06 |

build: 171974745 (10558)
<!--RO 1	stream_r2	wall_s=293	busy=230,259,292,281,14406,9403,10037,12055	busy_tot=46963	busy_little_share=0.0226	a55_cpu_cycles=42182660344	a55_inst_retired=8559682886	context_switches=9719800	a76_cpu_cycles=1012628189642	a76_l3d_cache_refill=14976721661	a76_l2d_cache_refill=7069215541	cpu_migrations=19609	page_faults=946985	a76_dtlb_walk=820774181	a76_mem_access=278554956533	a76_inst_retired=1288617964774	a76_l1d_cache_refill=16786331602	a55_inst_share=0.0066	a76_ipc=1.273	l2ref_pki=5.486	l3ref_pki=11.622	l1dref_pki=13.027	memacc_pki=216.17	dtlbw_pki=0.6369	pmu_enabled=100.0	pmu_cpu_s=2339.2	who=at_s=8,rss_pg=6044867,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8297	l2color_cv=0.0059	l3color_cv=0.0184	contig_frac=0.9351	mean_run=15.0	vapa16=0.0407	gib_regions=32	anon_pg=13054	anon_l2cv=0.0011	anon_contig=0.9982	anon_run=256.0	anon_gib=4	file_pg=24576	file_l2cv=0.0015	file_contig=0.9961	file_run=170.7	file_gib=27	other_pg=23541	other_l2cv=0.0157	other_contig=0.8365	other_run=6.0	other_gib=32	ro_cost_ms=87-->
<!--DATA 1	stream_r2	pp2048	21.35-->

### gemma4-12b-busy-intercept [stream_r4] pass 1  env='PIN_MASK=0xf0'  args='-t 4 -r 4'  22:50:10  clk=600 MHz  MemAvail=31598684 kB
<!--PRED 1	stream_r4	memavail_kb=31598684	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4538,8500,10060,8679,5855,5220,3972,3728,2488,192,251-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         21.19 ± 0.02 |

build: 171974745 (10558)
<!--RO 1	stream_r4	wall_s=489	busy=414,404,467,462,23552,15720,16134,21074	busy_tot=78227	busy_little_share=0.0223	a55_cpu_cycles=70200551593	a55_inst_retired=14057494185	context_switches=16186998	a76_cpu_cycles=1686898788260	a76_l3d_cache_refill=24847637287	a76_l2d_cache_refill=11744245857	cpu_migrations=32103	page_faults=979635	a76_dtlb_walk=1337941807	a76_mem_access=459989233293	a76_inst_retired=2134558710101	a76_l1d_cache_refill=27846193637	a55_inst_share=0.0065	a76_ipc=1.265	l2ref_pki=5.502	l3ref_pki=11.641	l1dref_pki=13.045	memacc_pki=215.50	dtlbw_pki=0.6268	pmu_enabled=100.0	pmu_cpu_s=3904.6	who=at_s=8,rss_pg=6044868,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8279	l2color_cv=0.0047	l3color_cv=0.0142	contig_frac=0.9266	mean_run=13.3	vapa16=0.0303	gib_regions=32	anon_pg=12932	anon_l2cv=0.0010	anon_contig=0.9978	anon_run=235.1	anon_gib=4	file_pg=24576	file_l2cv=0.0025	file_contig=0.9855	file_run=61.0	file_gib=30	other_pg=23535	other_l2cv=0.0128	other_contig=0.8260	other_run=5.7	other_gib=32	ro_cost_ms=85-->
<!--DATA 1	stream_r4	pp2048	21.19-->

### gemma4-12b-busy-intercept [res_r1] pass 1  env='ROCKET_F16_RESIDENT=auto PIN_MASK=0xf0'  args='-t 4 -r 1'  23:02:36  clk=600 MHz  MemAvail=31384704 kB
<!--PRED 1	res_r1	memavail_kb=31384404	anonhuge_kb=0	hugepagesz_kb=2048	buddy=49096,57639,69901,57209,6191,4212,3454,2988,1292,14,0-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         23.42 ± 0.00 |

build: 171974745 (10558)
<!--RO 1	res_r1	wall_s=247	busy=461,202,213,176,9025,6325,6715,7418	busy_tot=30535	busy_little_share=0.0345	a55_cpu_cycles=33241051074	a55_inst_retired=9851688979	context_switches=8893861	a76_cpu_cycles=675037606468	a76_l3d_cache_refill=8498313962	a76_l2d_cache_refill=5091728805	cpu_migrations=16153	page_faults=12590388	a76_dtlb_walk=517436827	a76_mem_access=222665092694	a76_inst_retired=985959359103	a76_l1d_cache_refill=12979105766	a55_inst_share=0.0099	a76_ipc=1.461	l2ref_pki=5.164	l3ref_pki=8.619	l1dref_pki=13.164	memacc_pki=225.84	dtlbw_pki=0.5248	pmu_enabled=100.0	pmu_cpu_s=1966.2	who=at_s=8,rss_pg=6028946,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8217	l2color_cv=0.0746	l3color_cv=0.0835	contig_frac=0.7794	mean_run=4.5	vapa16=0.0813	gib_regions=32	anon_pg=12420	anon_l2cv=0.2522	anon_contig=0.7350	anon_run=3.8	anon_gib=31	file_pg=24576	file_l2cv=0.0525	file_contig=0.7361	file_run=3.8	file_gib=32	other_pg=23586	other_l2cv=0.1209	other_contig=0.8480	other_run=6.5	other_gib=32	ro_cost_ms=81-->
<!--DATA 1	res_r1	pp2048	23.42-->
    [f16-resident] admission first declined at 18078MB resident: the NPU IOVA window filled (raise ROCKET_N_THREADS for more fds)
    [f16-resident] weights offered to the resident route: 286 resident on the NPU (18078MB), 42 streamed via the per-call pack -- 87% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21015MB (MemAvailable 30550MB - reserve 9535MB, no swap)

### gemma4-12b-busy-intercept [res_r2] pass 1  env='ROCKET_F16_RESIDENT=auto PIN_MASK=0xf0'  args='-t 4 -r 2'  23:11:37  clk=600 MHz  MemAvail=31601228 kB
<!--PRED 1	res_r2	memavail_kb=31601520	anonhuge_kb=0	hugepagesz_kb=2048	buddy=60625,44661,34643,49575,5504,4500,3942,3392,1992,33,1-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         21.93 ± 2.25 |

build: 171974745 (10558)
<!--RO 1	res_r2	wall_s=345	busy=293,263,243,346,12389,9263,9370,10703	busy_tot=42870	busy_little_share=0.0267	a55_cpu_cycles=41396071602	a55_inst_retired=9375428551	context_switches=12610370	a76_cpu_cycles=950774816175	a76_l3d_cache_refill=11902832416	a76_l2d_cache_refill=6091896032	cpu_migrations=21660	page_faults=12826557	a76_dtlb_walk=745023535	a76_mem_access=311487958020	a76_inst_retired=1410212333388	a76_l1d_cache_refill=18746994350	a55_inst_share=0.0066	a76_ipc=1.483	l2ref_pki=4.320	l3ref_pki=8.440	l1dref_pki=13.294	memacc_pki=220.88	dtlbw_pki=0.5283	pmu_enabled=100.0	pmu_cpu_s=2752.0	who=at_s=8,rss_pg=6036798,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8095	l2color_cv=0.0733	l3color_cv=0.0836	contig_frac=0.7196	mean_run=3.5	vapa16=0.0511	gib_regions=32	anon_pg=12420	anon_l2cv=0.0412	anon_contig=0.9331	anon_run=14.5	anon_gib=32	file_pg=24576	file_l2cv=0.0460	file_contig=0.7391	file_run=3.8	file_gib=32	other_pg=22685	other_l2cv=0.1722	other_contig=0.5817	other_run=2.4	other_gib=29	ro_cost_ms=81-->
<!--DATA 1	res_r2	pp2048	21.93-->
    [f16-resident] admission first declined at 18078MB resident: MemAvailable 9444MB fell below the 9535MB reserve floor
    [f16-resident] weights offered to the resident route: 286 resident on the NPU (18078MB), 42 streamed via the per-call pack -- 87% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21156MB (MemAvailable 30691MB - reserve 9535MB, no swap)

### gemma4-12b-busy-intercept [res_r4] pass 1  env='ROCKET_F16_RESIDENT=auto PIN_MASK=0xf0'  args='-t 4 -r 4'  23:23:05  clk=600 MHz  MemAvail=31422344 kB
<!--PRED 1	res_r4	memavail_kb=31422344	anonhuge_kb=0	hugepagesz_kb=2048	buddy=24612,14215,15943,11897,6030,5851,4971,4114,2363,78,131-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         22.14 ± 1.91 |

build: 171974745 (10558)
<!--RO 1	res_r4	wall_s=606	busy=414,468,466,451,19693,14217,15432,16927	busy_tot=68068	busy_little_share=0.0264	a55_cpu_cycles=66393570056	a55_inst_retired=14680249197	context_switches=22035515	a76_cpu_cycles=1522306888604	a76_l3d_cache_refill=18726964157	a76_l2d_cache_refill=9747451617	cpu_migrations=37132	page_faults=14236046	a76_dtlb_walk=1095441651	a76_mem_access=495522978882	a76_inst_retired=2266050911316	a76_l1d_cache_refill=30842732192	a55_inst_share=0.0064	a76_ipc=1.489	l2ref_pki=4.302	l3ref_pki=8.264	l1dref_pki=13.611	memacc_pki=218.67	dtlbw_pki=0.4834	pmu_enabled=100.0	pmu_cpu_s=4843.7	who=at_s=8,rss_pg=6036977,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8101	l2color_cv=0.1174	l3color_cv=0.1336	contig_frac=0.7273	mean_run=3.6	vapa16=0.0459	gib_regions=32	anon_pg=12478	anon_l2cv=0.0167	anon_contig=0.9820	anon_run=49.9	anon_gib=32	file_pg=24576	file_l2cv=0.0483	file_contig=0.6570	file_run=2.9	file_gib=32	other_pg=22672	other_l2cv=0.2853	other_contig=0.6633	other_run=3.0	other_gib=32	ro_cost_ms=69-->
<!--DATA 1	res_r4	pp2048	22.14-->
    [f16-resident] admission first declined at 18078MB resident: MemAvailable 9415MB fell below the 9535MB reserve floor
    [f16-resident] weights offered to the resident route: 286 resident on the NPU (18078MB), 42 streamed via the per-call pack -- 87% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21028MB (MemAvailable 30563MB - reserve 9535MB, no swap)

### gemma4-12b-busy-intercept [stream_r2] pass 2  env='PIN_MASK=0xf0'  args='-t 4 -r 2'  23:35:02  clk=600 MHz  MemAvail=31282224 kB
<!--PRED 2	stream_r2	memavail_kb=31282224	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7316,5339,7710,7064,6267,5369,5064,4228,2551,207,75-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         21.31 ± 0.05 |

build: 171974745 (10558)
<!--RO 2	stream_r2	wall_s=294	busy=292,290,229,205,14214,9762,9822,11992	busy_tot=46806	busy_little_share=0.0217	a55_cpu_cycles=40993317538	a55_inst_retired=8231898116	context_switches=9720773	a76_cpu_cycles=1009891818374	a76_l3d_cache_refill=15020732696	a76_l2d_cache_refill=7075609818	cpu_migrations=19507	page_faults=945652	a76_dtlb_walk=821384441	a76_mem_access=278777385992	a76_inst_retired=1289623079112	a76_l1d_cache_refill=16796339895	a55_inst_share=0.0063	a76_ipc=1.277	l2ref_pki=5.487	l3ref_pki=11.647	l1dref_pki=13.024	memacc_pki=216.17	dtlbw_pki=0.6369	pmu_enabled=100.0	pmu_cpu_s=2343.1	who=at_s=8,rss_pg=6044847,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8280	l2color_cv=0.0130	l3color_cv=0.0192	contig_frac=0.8211	mean_run=5.5	vapa16=0.0280	gib_regions=32	anon_pg=12932	anon_l2cv=0.0010	anon_contig=0.9983	anon_run=269.4	anon_gib=4	file_pg=24576	file_l2cv=0.0046	file_contig=0.9905	file_run=87.1	file_gib=30	other_pg=23538	other_l2cv=0.0375	other_contig=0.5469	other_run=2.2	other_gib=32	ro_cost_ms=66-->
<!--DATA 2	stream_r2	pp2048	21.31-->

### gemma4-12b-busy-intercept [stream_r4] pass 2  env='PIN_MASK=0xf0'  args='-t 4 -r 4'  23:42:44  clk=600 MHz  MemAvail=31465520 kB
<!--PRED 2	stream_r4	memavail_kb=31465520	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7121,9092,11860,7731,6074,5441,4697,4000,2304,157,230-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         21.11 ± 0.05 |

build: 171974745 (10558)
<!--RO 2	stream_r4	wall_s=490	busy=506,446,384,425,23845,15493,16698,20438	busy_tot=78235	busy_little_share=0.0225	a55_cpu_cycles=69018694749	a55_inst_retired=13773445333	context_switches=16198807	a76_cpu_cycles=1685535286227	a76_l3d_cache_refill=24924839504	a76_l2d_cache_refill=11776517509	cpu_migrations=32304	page_faults=971752	a76_dtlb_walk=1371048764	a76_mem_access=460018224192	a76_inst_retired=2134700392176	a76_l1d_cache_refill=27933117720	a55_inst_share=0.0064	a76_ipc=1.266	l2ref_pki=5.517	l3ref_pki=11.676	l1dref_pki=13.085	memacc_pki=215.50	dtlbw_pki=0.6423	pmu_enabled=100.0	pmu_cpu_s=3918.8	who=at_s=8,rss_pg=6044859,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8228	l2color_cv=0.0084	l3color_cv=0.0190	contig_frac=0.8921	mean_run=9.1	vapa16=0.0218	gib_regions=32	anon_pg=12932	anon_l2cv=0.0012	anon_contig=0.9978	anon_run=235.1	anon_gib=4	file_pg=24576	file_l2cv=0.0050	file_contig=0.9916	file_run=96.4	file_gib=31	other_pg=23158	other_l2cv=0.0213	other_contig=0.7274	other_run=3.6	other_gib=32	ro_cost_ms=88-->
<!--DATA 2	stream_r4	pp2048	21.11-->

### gemma4-12b-busy-intercept [res_r1] pass 2  env='ROCKET_F16_RESIDENT=auto PIN_MASK=0xf0'  args='-t 4 -r 1'  23:55:17  clk=600 MHz  MemAvail=31179776 kB
<!--PRED 2	res_r1	memavail_kb=31179776	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5450,10455,12823,11447,6267,5699,4861,4052,2341,69,139-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         20.09 ± 0.00 |

build: 171974745 (10558)
<!--RO 2	res_r1	wall_s=275	busy=206,175,191,129,9335,7091,7645,7619	busy_tot=32391	busy_little_share=0.0216	a55_cpu_cycles=26525512431	a55_inst_retired=5949157201	context_switches=9775928	a76_cpu_cycles=729890031203	a76_l3d_cache_refill=8684414361	a76_l2d_cache_refill=7167749607	cpu_migrations=15530	page_faults=13004808	a76_dtlb_walk=482991019	a76_mem_access=235643049135	a76_inst_retired=1030367907093	a76_l1d_cache_refill=14050576612	a55_inst_share=0.0057	a76_ipc=1.412	l2ref_pki=6.956	l3ref_pki=8.428	l1dref_pki=13.636	memacc_pki=228.70	dtlbw_pki=0.4688	pmu_enabled=100.0	pmu_cpu_s=2198.9	who=at_s=8,rss_pg=6036963,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8214	l2color_cv=0.0307	l3color_cv=0.0363	contig_frac=0.7374	mean_run=3.8	vapa16=0.0561	gib_regions=32	anon_pg=12420	anon_l2cv=0.0113	anon_contig=0.9163	anon_run=11.7	anon_gib=32	file_pg=24576	file_l2cv=0.0756	file_contig=0.6701	file_run=3.0	file_gib=32	other_pg=23563	other_l2cv=0.0139	other_contig=0.7134	other_run=3.5	other_gib=32	ro_cost_ms=96-->
<!--DATA 2	res_r1	pp2048	20.09-->
    [f16-resident] admission first declined at 18078MB resident: MemAvailable 9411MB fell below the 9535MB reserve floor
    [f16-resident] weights offered to the resident route: 286 resident on the NPU (18078MB), 42 streamed via the per-call pack -- 87% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 20819MB (MemAvailable 30355MB - reserve 9535MB, no swap)

### gemma4-12b-busy-intercept [res_r2] pass 2  env='ROCKET_F16_RESIDENT=auto PIN_MASK=0xf0'  args='-t 4 -r 2'  00:04:34  clk=600 MHz  MemAvail=31440964 kB
<!--PRED 2	res_r2	memavail_kb=31440964	anonhuge_kb=0	hugepagesz_kb=2048	buddy=40388,20866,19503,16703,6427,6038,5177,4214,2522,24,6-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         22.09 ± 1.94 |

build: 171974745 (10558)
<!--RO 2	res_r2	wall_s=438	busy=299,216,271,307,12035,9977,10847,10090	busy_tot=44042	busy_little_share=0.0248	a55_cpu_cycles=41071585881	a55_inst_retired=9395842524	context_switches=15890417	a76_cpu_cycles=1001839721191	a76_l3d_cache_refill=12022497712	a76_l2d_cache_refill=6330208067	cpu_migrations=21839	page_faults=14363192	a76_dtlb_walk=696208760	a76_mem_access=333922268359	a76_inst_retired=1475055841186	a76_l1d_cache_refill=19448163891	a55_inst_share=0.0063	a76_ipc=1.472	l2ref_pki=4.292	l3ref_pki=8.151	l1dref_pki=13.185	memacc_pki=226.38	dtlbw_pki=0.4720	pmu_enabled=100.0	pmu_cpu_s=3500.6	who=at_s=8,rss_pg=6036963,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8165	l2color_cv=0.2837	l3color_cv=0.3063	contig_frac=0.6508	mean_run=2.9	vapa16=0.0553	gib_regions=32	anon_pg=12420	anon_l2cv=0.0350	anon_contig=0.9601	anon_run=23.9	anon_gib=32	file_pg=24576	file_l2cv=0.0897	file_contig=0.6574	file_run=2.9	file_gib=32	other_pg=23205	other_l2cv=0.7196	other_contig=0.4782	other_run=1.9	other_gib=29	ro_cost_ms=99-->
<!--DATA 2	res_r2	pp2048	22.09-->
    [f16-resident] admission first declined at 18078MB resident: MemAvailable 9408MB fell below the 9535MB reserve floor
    [f16-resident] weights offered to the resident route: 286 resident on the NPU (18078MB), 42 streamed via the per-call pack -- 87% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21040MB (MemAvailable 30575MB - reserve 9535MB, no swap)

### gemma4-12b-busy-intercept [res_r4] pass 2  env='ROCKET_F16_RESIDENT=auto PIN_MASK=0xf0'  args='-t 4 -r 4'  00:17:39  clk=600 MHz  MemAvail=31458348 kB
<!--PRED 2	res_r4	memavail_kb=31458348	anonhuge_kb=0	hugepagesz_kb=2048	buddy=30490,19035,18445,14824,6415,5967,5066,4090,2324,201,28-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         22.64 ± 1.81 |

build: 171974745 (10558)
<!--RO 2	res_r4	wall_s=588	busy=469,409,397,407,19097,14586,15217,16523	busy_tot=67105	busy_little_share=0.0251	a55_cpu_cycles=64342755904	a55_inst_retired=13959007686	context_switches=21981797	a76_cpu_cycles=1502328147927	a76_l3d_cache_refill=18801247007	a76_l2d_cache_refill=10541169540	cpu_migrations=36933	page_faults=14201631	a76_dtlb_walk=1103005218	a76_mem_access=491768762407	a76_inst_retired=2257546747201	a76_l1d_cache_refill=30731115724	a55_inst_share=0.0061	a76_ipc=1.503	l2ref_pki=4.669	l3ref_pki=8.328	l1dref_pki=13.613	memacc_pki=217.83	dtlbw_pki=0.4886	pmu_enabled=100.0	pmu_cpu_s=4701.4	who=at_s=8,rss_pg=6036969,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8214	l2color_cv=0.1142	l3color_cv=0.1231	contig_frac=0.7509	mean_run=4.0	vapa16=0.0949	gib_regions=32	anon_pg=12420	anon_l2cv=0.0016	anon_contig=0.9909	anon_run=90.0	anon_gib=24	file_pg=24576	file_l2cv=0.0807	file_contig=0.6519	file_run=2.9	file_gib=32	other_pg=23563	other_l2cv=0.3292	other_contig=0.7277	other_run=3.7	other_gib=32	ro_cost_ms=96-->
<!--DATA 2	res_r4	pp2048	22.64-->
    [f16-resident] admission first declined at 18078MB resident: MemAvailable 9403MB fell below the 9535MB reserve floor
    [f16-resident] weights offered to the resident route: 286 resident on the NPU (18078MB), 42 streamed via the per-call pack -- 87% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21011MB (MemAvailable 30547MB - reserve 9535MB, no swap)

### gemma4-12b-busy-intercept [stream_r1] pass 2  env='PIN_MASK=0xf0'  args='-t 4 -r 1'  00:28:50  clk=600 MHz  MemAvail=31254836 kB
<!--PRED 2	stream_r1	memavail_kb=31255424	anonhuge_kb=0	hugepagesz_kb=2048	buddy=2107,3753,7786,7335,6177,5554,5039,4102,2304,183,161-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         20.99 ± 0.00 |

build: 171974745 (10558)
<!--RO 2	stream_r1	wall_s=200	busy=187,145,178,166,9669,6270,6412,8499	busy_tot=31526	busy_little_share=0.0214	a55_cpu_cycles=27837094647	a55_inst_retired=5596600768	context_switches=6480321	a76_cpu_cycles=681125476514	a76_l3d_cache_refill=10030160710	a76_l2d_cache_refill=4797765445	cpu_migrations=13215	page_faults=927510	a76_dtlb_walk=551808932	a76_mem_access=187880984672	a76_inst_retired=865848952234	a76_l1d_cache_refill=11232335161	a55_inst_share=0.0064	a76_ipc=1.271	l2ref_pki=5.541	l3ref_pki=11.584	l1dref_pki=12.973	memacc_pki=216.99	dtlbw_pki=0.6373	pmu_enabled=100.0	pmu_cpu_s=1596.7	who=at_s=8,rss_pg=6044838,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8288	l2color_cv=0.0092	l3color_cv=0.0137	contig_frac=0.9090	mean_run=10.8	vapa16=0.0065	gib_regions=32	anon_pg=12942	anon_l2cv=0.0091	anon_contig=0.9298	anon_run=13.9	anon_gib=31	file_pg=24576	file_l2cv=0.0003	file_contig=0.9951	file_run=146.3	file_gib=31	other_pg=23589	other_l2cv=0.0207	other_contig=0.8079	other_run=5.2	other_gib=32	ro_cost_ms=90-->
<!--DATA 2	stream_r1	pp2048	20.99-->

### gemma4-12b-busy-intercept [stream_r4] pass 3  env='PIN_MASK=0xf0'  args='-t 4 -r 4'  00:34:58  clk=600 MHz  MemAvail=31454116 kB
<!--PRED 3	stream_r4	memavail_kb=31454572	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6924,9468,12758,7578,6282,5554,4883,4013,2251,158,217-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         21.05 ± 0.07 |

build: 171974745 (10558)
<!--RO 3	stream_r4	wall_s=491	busy=475,423,352,428,23726,15360,16567,20679	busy_tot=78010	busy_little_share=0.0215	a55_cpu_cycles=68642175948	a55_inst_retired=13762936660	context_switches=16201977	a76_cpu_cycles=1683833841241	a76_l3d_cache_refill=24850451342	a76_l2d_cache_refill=11765230603	cpu_migrations=32824	page_faults=973069	a76_dtlb_walk=1381761809	a76_mem_access=460172102245	a76_inst_retired=2135165185716	a76_l1d_cache_refill=27888394460	a55_inst_share=0.0064	a76_ipc=1.268	l2ref_pki=5.510	l3ref_pki=11.639	l1dref_pki=13.061	memacc_pki=215.52	dtlbw_pki=0.6471	pmu_enabled=100.0	pmu_cpu_s=3929.2	who=at_s=8,rss_pg=6044856,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8303	l2color_cv=0.0100	l3color_cv=0.0171	contig_frac=0.9174	mean_run=11.8	vapa16=0.0237	gib_regions=32	anon_pg=12932	anon_l2cv=0.0009	anon_contig=0.9981	anon_run=253.6	anon_gib=4	file_pg=24576	file_l2cv=0.0039	file_contig=0.9910	file_run=91.4	file_gib=30	other_pg=23709	other_l2cv=0.0252	other_contig=0.7971	other_run=4.9	other_gib=32	ro_cost_ms=87-->
<!--DATA 3	stream_r4	pp2048	21.05-->

### gemma4-12b-busy-intercept [res_r1] pass 3  env='ROCKET_F16_RESIDENT=auto PIN_MASK=0xf0'  args='-t 4 -r 1'  00:47:36  clk=600 MHz  MemAvail=31329532 kB
<!--PRED 3	res_r1	memavail_kb=31329532	anonhuge_kb=0	hugepagesz_kb=2048	buddy=28471,10363,13411,11707,6472,5939,5060,4115,2289,57,137-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         20.59 ± 0.00 |

build: 171974745 (10558)
<!--RO 3	res_r1	wall_s=347	busy=171,186,204,209,8865,7830,7236,7291	busy_tot=31992	busy_little_share=0.0241	a55_cpu_cycles=28284451219	a55_inst_retired=6475788284	context_switches=12567049	a76_cpu_cycles=738071754298	a76_l3d_cache_refill=8541247332	a76_l2d_cache_refill=4672839454	cpu_migrations=15592	page_faults=14301154	a76_dtlb_walk=488915792	a76_mem_access=251895444083	a76_inst_retired=1076326714372	a76_l1d_cache_refill=13486951045	a55_inst_share=0.0060	a76_ipc=1.458	l2ref_pki=4.341	l3ref_pki=7.936	l1dref_pki=12.531	memacc_pki=234.03	dtlbw_pki=0.4542	pmu_enabled=100.0	pmu_cpu_s=2762.7	who=at_s=8,rss_pg=6036983,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8206	l2color_cv=0.0795	l3color_cv=0.0923	contig_frac=0.7518	mean_run=4.0	vapa16=0.0672	gib_regions=32	anon_pg=12349	anon_l2cv=0.0007	anon_contig=0.9972	anon_run=209.3	anon_gib=4	file_pg=24576	file_l2cv=0.0875	file_contig=0.6558	file_run=2.9	file_gib=32	other_pg=23575	other_l2cv=0.2492	other_contig=0.7233	other_run=3.6	other_gib=32	ro_cost_ms=85-->
<!--DATA 3	res_r1	pp2048	20.59-->
    [f16-resident] admission first declined at 18078MB resident: MemAvailable 9399MB fell below the 9535MB reserve floor
    [f16-resident] weights offered to the resident route: 286 resident on the NPU (18078MB), 42 streamed via the per-call pack -- 87% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 20909MB (MemAvailable 30444MB - reserve 9535MB, no swap)

### gemma4-12b-busy-intercept [res_r2] pass 3  env='ROCKET_F16_RESIDENT=auto PIN_MASK=0xf0'  args='-t 4 -r 2'  00:58:18  clk=600 MHz  MemAvail=31432784 kB
<!--PRED 3	res_r2	memavail_kb=31432856	anonhuge_kb=0	hugepagesz_kb=2048	buddy=35366,20526,20627,13533,6877,6282,5297,4249,2471,48,4-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         21.99 ± 2.06 |

build: 171974745 (10558)
<!--RO 3	res_r2	wall_s=463	busy=254,289,267,275,12362,9635,9788,11229	busy_tot=44099	busy_little_share=0.0246	a55_cpu_cycles=40390876831	a55_inst_retired=8968093009	context_switches=16059749	a76_cpu_cycles=1006686979982	a76_l3d_cache_refill=12043809339	a76_l2d_cache_refill=6482480260	cpu_migrations=22691	page_faults=14436981	a76_dtlb_walk=700725093	a76_mem_access=334491961780	a76_inst_retired=1476841653631	a76_l1d_cache_refill=19370811492	a55_inst_share=0.0060	a76_ipc=1.467	l2ref_pki=4.389	l3ref_pki=8.155	l1dref_pki=13.116	memacc_pki=226.49	dtlbw_pki=0.4745	pmu_enabled=100.0	pmu_cpu_s=3693.5	who=at_s=8,rss_pg=6036988,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8217	l2color_cv=0.1268	l3color_cv=0.1425	contig_frac=0.7225	mean_run=3.6	vapa16=0.0502	gib_regions=31	anon_pg=12420	anon_l2cv=0.0113	anon_contig=0.9746	anon_run=36.5	anon_gib=27	file_pg=24576	file_l2cv=0.1617	file_contig=0.6545	file_run=2.9	file_gib=31	other_pg=23586	other_l2cv=0.2674	other_contig=0.6606	other_run=2.9	other_gib=29	ro_cost_ms=65-->
<!--DATA 3	res_r2	pp2048	21.99-->
    [f16-resident] admission first declined at 18078MB resident: MemAvailable 9397MB fell below the 9535MB reserve floor
    [f16-resident] weights offered to the resident route: 286 resident on the NPU (18078MB), 42 streamed via the per-call pack -- 87% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 20977MB (MemAvailable 30512MB - reserve 9535MB, no swap)

### gemma4-12b-busy-intercept [res_r4] pass 3  env='ROCKET_F16_RESIDENT=auto PIN_MASK=0xf0'  args='-t 4 -r 4'  01:11:38  clk=600 MHz  MemAvail=31470428 kB
<!--PRED 3	res_r4	memavail_kb=31470428	anonhuge_kb=0	hugepagesz_kb=2048	buddy=34302,17879,15899,12597,6995,6481,5491,4402,2511,31,4-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         23.67 ± 0.13 |

build: 171974745 (10558)
<!--RO 3	res_r4	wall_s=594	busy=423,426,406,466,19465,14379,15263,17018	busy_tot=67846	busy_little_share=0.0254	a55_cpu_cycles=65417869681	a55_inst_retired=14361629996	context_switches=21396378	a76_cpu_cycles=1506686950155	a76_l3d_cache_refill=18774827704	a76_l2d_cache_refill=9234447469	cpu_migrations=37274	page_faults=13898027	a76_dtlb_walk=1072264821	a76_mem_access=494011689355	a76_inst_retired=2261067879042	a76_l1d_cache_refill=30687945019	a55_inst_share=0.0063	a76_ipc=1.501	l2ref_pki=4.084	l3ref_pki=8.304	l1dref_pki=13.572	memacc_pki=218.49	dtlbw_pki=0.4742	pmu_enabled=100.0	pmu_cpu_s=4745.3	who=at_s=8,rss_pg=6037041,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8214	l2color_cv=0.0797	l3color_cv=0.0921	contig_frac=0.7389	mean_run=3.8	vapa16=0.0657	gib_regions=32	anon_pg=12420	anon_l2cv=0.0075	anon_contig=0.9736	anon_run=35.3	anon_gib=30	file_pg=24576	file_l2cv=0.0883	file_contig=0.6799	file_run=3.1	file_gib=32	other_pg=23565	other_l2cv=0.2553	other_contig=0.6766	other_run=3.1	other_gib=29	ro_cost_ms=87-->
<!--DATA 3	res_r4	pp2048	23.67-->
    [f16-resident] admission first declined at 18078MB resident: MemAvailable 9396MB fell below the 9535MB reserve floor
    [f16-resident] weights offered to the resident route: 286 resident on the NPU (18078MB), 42 streamed via the per-call pack -- 87% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21008MB (MemAvailable 30543MB - reserve 9535MB, no swap)

### gemma4-12b-busy-intercept [stream_r1] pass 3  env='PIN_MASK=0xf0'  args='-t 4 -r 1'  01:22:56  clk=600 MHz  MemAvail=31445420 kB
<!--PRED 3	stream_r1	memavail_kb=31445120	anonhuge_kb=0	hugepagesz_kb=2048	buddy=12018,13106,12836,8701,7291,6527,5684,4617,2625,16,0-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         21.26 ± 0.00 |

build: 171974745 (10558)
<!--RO 3	stream_r1	wall_s=197	busy=163,143,198,211,9434,6280,6628,8591	busy_tot=31648	busy_little_share=0.0226	a55_cpu_cycles=27555453273	a55_inst_retired=5585417542	context_switches=6475815	a76_cpu_cycles=681049092860	a76_l3d_cache_refill=10036824070	a76_l2d_cache_refill=4728574220	cpu_migrations=12987	page_faults=930829	a76_dtlb_walk=549604842	a76_mem_access=187827237174	a76_inst_retired=865464980246	a76_l1d_cache_refill=11241446497	a55_inst_share=0.0064	a76_ipc=1.271	l2ref_pki=5.464	l3ref_pki=11.597	l1dref_pki=12.989	memacc_pki=217.02	dtlbw_pki=0.6350	pmu_enabled=100.0	pmu_cpu_s=1575.6	who=at_s=8,rss_pg=6044851,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8277	l2color_cv=0.0270	l3color_cv=0.0348	contig_frac=0.8970	mean_run=9.5	vapa16=0.0419	gib_regions=32	anon_pg=13022	anon_l2cv=0.0020	anon_contig=0.9890	anon_run=76.6	anon_gib=27	file_pg=24576	file_l2cv=0.0057	file_contig=0.9844	file_run=57.0	file_gib=30	other_pg=23424	other_l2cv=0.0671	other_contig=0.7541	other_run=4.0	other_gib=29	ro_cost_ms=84-->
<!--DATA 3	stream_r1	pp2048	21.26-->

### gemma4-12b-busy-intercept [stream_r2] pass 3  env='PIN_MASK=0xf0'  args='-t 4 -r 2'  01:28:05  clk=600 MHz  MemAvail=31378028 kB
<!--PRED 3	stream_r2	memavail_kb=31378028	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8611,9780,12999,7607,6570,5344,4937,4091,2094,205,202-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         21.21 ± 0.02 |

build: 171974745 (10558)
<!--RO 3	stream_r2	wall_s=296	busy=270,278,275,301,14201,9698,9814,12289	busy_tot=47126	busy_little_share=0.0239	a55_cpu_cycles=43151526378	a55_inst_retired=9029401843	context_switches=9724897	a76_cpu_cycles=1018109061022	a76_l3d_cache_refill=15033401917	a76_l2d_cache_refill=7079718125	cpu_migrations=19273	page_faults=965297	a76_dtlb_walk=827733174	a76_mem_access=278942658082	a76_inst_retired=1290326653933	a76_l1d_cache_refill=16768994585	a55_inst_share=0.0069	a76_ipc=1.267	l2ref_pki=5.487	l3ref_pki=11.651	l1dref_pki=12.996	memacc_pki=216.18	dtlbw_pki=0.6415	pmu_enabled=100.0	pmu_cpu_s=2357.4	who=at_s=8,rss_pg=6044860,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8340	l2color_cv=0.0065	l3color_cv=0.0205	contig_frac=0.9128	mean_run=11.2	vapa16=0.0162	gib_regions=32	anon_pg=12942	anon_l2cv=0.0014	anon_contig=0.9982	anon_run=258.8	anon_gib=4	file_pg=24576	file_l2cv=0.0003	file_contig=0.9951	file_run=146.3	file_gib=30	other_pg=23968	other_l2cv=0.0167	other_contig=0.7823	other_run=4.6	other_gib=32	ro_cost_ms=85-->
<!--DATA 3	stream_r2	pp2048	21.21-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stream_r1]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stream_r1 | 21.09 | 3 | 20.99 | 21.26 | -- | -- |
| pp2048 | stream_r2 | 21.29 | 3 | 21.21 | 21.35 | 1.010x | 1.016 1.015 0.998 |
| pp2048 | stream_r4 | 21.12 | 3 | 21.05 | 21.19 | 1.001x | 1.009 1.006 0.990 |
| pp2048 | res_r1 | 21.37 | 3 | 20.09 | 23.42 | 1.013x | 1.115 0.957 0.968 |
| pp2048 | res_r2 | 22.00 | 3 | 21.93 | 22.09 | 1.044x | 1.044 1.052 1.034 |
| pp2048 | res_r4 | 22.82 | 3 | 22.14 | 23.67 | 1.082x | 1.054 1.079 1.113 |

