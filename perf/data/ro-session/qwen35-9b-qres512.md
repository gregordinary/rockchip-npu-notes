== qwen35-9b-qres512  Mon Aug 31 19:09:43 UTC 2026 ==
### qwen35-9b-qres512 [stock] pass 1  env=''  args=''  19:10:47  clk=600 MHz  MemAvail=31602936 kB
<!--PRED 1	stock	memavail_kb=31603308	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7471,6601,5828,5431,4527,3652,3224,2829,2283,466,4787-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         19.19 ± 0.02 |

build: 171974745 (10558)
<!--RO 1	stock	busy=6733,5499,5356,5337,24703,22382,22102,22677	busy_tot=114789	busy_little_share=0.1997	a55_cpu_cycles=423261024763	a55_inst_retired=222259818988	context_switches=7976035	a76_cpu_cycles=1987418058180	a76_l3d_cache_refill=15962979110	a76_l2d_cache_refill=8246146370	cpu_migrations=41472	page_faults=640952	a76_dtlb_walk=2205793381	a76_mem_access=658247950844	a76_inst_retired=4132646583505	a76_l1d_cache_refill=13870577033	a55_inst_share=0.0510	a76_ipc=2.079	l2ref_pki=1.995	l3ref_pki=3.863	l1dref_pki=3.356	memacc_pki=159.28	dtlbw_pki=0.5337	pmu_enabled=100.0	pmu_secs=3428.6	who=at_s=6,rss_pg=1509913,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6979	l2color_cv=0.0114	l3color_cv=0.0145	contig_frac=0.8824	mean_run=8.4	vapa16=0.1756	gib_regions=28	anon_pg=22904	anon_l2cv=0.0012	anon_contig=0.9199	anon_run=12.2	anon_gib=23	file_pg=24576	file_l2cv=0.0011	file_contig=0.9955	file_run=155.5	file_gib=26	other_pg=21123	other_l2cv=0.0368	other_contig=0.7102	other_run=3.4	other_gib=28	ro_cost_ms=93-->
<!--DATA 1	stock	pp2048	19.19-->

### qwen35-9b-qres512 [qres512] pass 1  env='ROCKET_QUANT_RESIDENT=auto'  args=''  19:19:12  clk=600 MHz  MemAvail=31428268 kB
<!--PRED 1	qres512	memavail_kb=31428268	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6955,6798,5986,5461,4484,4198,3618,3210,2617,833,4388-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         33.53 ± 0.21 |

build: 171974745 (10558)
<!--RO 1	qres512	busy=5370,5174,5298,5155,8466,7617,8158,9292	busy_tot=54530	busy_little_share=0.3851	a55_cpu_cycles=380894612176	a55_inst_retired=207529678383	context_switches=7898346	a76_cpu_cycles=749317056513	a76_l3d_cache_refill=8824246410	a76_l2d_cache_refill=5217919927	cpu_migrations=43711	page_faults=10954113	a76_dtlb_walk=1998058948	a76_mem_access=379165339449	a76_inst_retired=1214764797759	a76_l1d_cache_refill=9509960743	a55_inst_share=0.1459	a76_ipc=1.621	l2ref_pki=4.295	l3ref_pki=7.264	l1dref_pki=7.829	memacc_pki=312.13	dtlbw_pki=1.6448	pmu_enabled=100.0	pmu_secs=2241.1	who=at_s=6,rss_pg=1485057,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.6819	l2color_cv=0.0042	l3color_cv=0.0084	contig_frac=0.8335	mean_run=5.9	vapa16=0.1218	gib_regions=28	anon_pg=4608	anon_l2cv=0.0085	anon_contig=0.9202	anon_run=12.3	anon_gib=25	file_pg=24576	file_l2cv=0.0013	file_contig=0.9922	file_run=102.4	file_gib=28	other_pg=21092	other_l2cv=0.0103	other_contig=0.6296	other_run=2.7	other_gib=28	ro_cost_ms=78-->
<!--DATA 1	qres512	pp2048	33.53-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21085MB (MemAvailable 30621MB - reserve 9535MB, no swap)

### qwen35-9b-qres512 [qres512] pass 2  env='ROCKET_QUANT_RESIDENT=auto'  args=''  19:25:08  clk=600 MHz  MemAvail=31409584 kB
<!--PRED 2	qres512	memavail_kb=31409584	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7849,7948,7355,6466,5666,4647,4073,3585,2944,1140,4024-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         33.03 ± 0.19 |

build: 171974745 (10558)
<!--RO 2	qres512	busy=5339,5271,5244,5197,8697,7764,8476,8714	busy_tot=54702	busy_little_share=0.3848	a55_cpu_cycles=382688425701	a55_inst_retired=208113224928	context_switches=7901492	a76_cpu_cycles=752127634965	a76_l3d_cache_refill=8841707090	a76_l2d_cache_refill=5170432804	cpu_migrations=43461	page_faults=10955975	a76_dtlb_walk=1999452646	a76_mem_access=379275898508	a76_inst_retired=1215076976717	a76_l1d_cache_refill=9488698120	a55_inst_share=0.1462	a76_ipc=1.616	l2ref_pki=4.255	l3ref_pki=7.277	l1dref_pki=7.809	memacc_pki=312.14	dtlbw_pki=1.6455	pmu_enabled=100.0	pmu_secs=2266.7	who=at_s=6,rss_pg=1485063,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=90112	present_frac=0.7383	l2color_cv=0.0035	l3color_cv=0.0122	contig_frac=0.8202	mean_run=5.5	vapa16=0.0284	gib_regions=28	anon_pg=20838	anon_l2cv=0.0025	anon_contig=0.8582	anon_run=7.0	anon_gib=27	file_pg=24576	file_l2cv=0.0011	file_contig=0.9912	file_run=93.4	file_gib=28	other_pg=21113	other_l2cv=0.0117	other_contig=0.5836	other_run=2.4	other_gib=28	ro_cost_ms=65-->
<!--DATA 2	qres512	pp2048	33.03-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21060MB (MemAvailable 30595MB - reserve 9535MB, no swap)

### qwen35-9b-qres512 [stock] pass 2  env=''  args=''  19:30:57  clk=600 MHz  MemAvail=31600768 kB
<!--PRED 2	stock	memavail_kb=31600768	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7319,7312,6728,6132,5339,4325,3830,3373,2746,962,4273-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         19.02 ± 0.23 |

build: 171974745 (10558)
<!--RO 2	stock	busy=7212,5791,5587,5413,24982,22317,22105,22515	busy_tot=115922	busy_little_share=0.2071	a55_cpu_cycles=441251566703	a55_inst_retired=229915864735	context_switches=7980780	a76_cpu_cycles=1987648069768	a76_l3d_cache_refill=15875631699	a76_l2d_cache_refill=8215401511	cpu_migrations=41876	page_faults=696461	a76_dtlb_walk=1815396249	a76_mem_access=657010822971	a76_inst_retired=4128403789453	a76_l1d_cache_refill=13788611811	a55_inst_share=0.0528	a76_ipc=2.077	l2ref_pki=1.990	l3ref_pki=3.845	l1dref_pki=3.340	memacc_pki=159.14	dtlbw_pki=0.4397	pmu_enabled=100.0	pmu_secs=3467.3	who=at_s=6,rss_pg=1509917,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6979	l2color_cv=0.0070	l3color_cv=0.0118	contig_frac=0.8944	mean_run=9.3	vapa16=0.1758	gib_regions=28	anon_pg=22952	anon_l2cv=0.0018	anon_contig=0.9173	anon_run=11.8	anon_gib=24	file_pg=24576	file_l2cv=0.0005	file_contig=0.9954	file_run=151.7	file_gib=27	other_pg=21080	other_l2cv=0.0216	other_contig=0.7517	other_run=4.0	other_gib=28	ro_cost_ms=80-->
<!--DATA 2	stock	pp2048	19.02-->

### qwen35-9b-qres512 [stock] pass 3  env=''  args=''  19:39:16  clk=600 MHz  MemAvail=31594984 kB
<!--PRED 3	stock	memavail_kb=31594984	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7453,7038,6492,5856,5173,3822,3509,3076,2485,705,4544-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         19.08 ± 0.16 |

build: 171974745 (10558)
<!--RO 3	stock	busy=6703,5664,5480,5408,24811,22639,22027,22774	busy_tot=115506	busy_little_share=0.2013	a55_cpu_cycles=430842526717	a55_inst_retired=226136176915	context_switches=7974083	a76_cpu_cycles=1992490255557	a76_l3d_cache_refill=15907148428	a76_l2d_cache_refill=8236504896	cpu_migrations=41905	page_faults=703146	a76_dtlb_walk=2211697346	a76_mem_access=658236061003	a76_inst_retired=4132780867121	a76_l1d_cache_refill=13778961124	a55_inst_share=0.0519	a76_ipc=2.074	l2ref_pki=1.993	l3ref_pki=3.849	l1dref_pki=3.334	memacc_pki=159.27	dtlbw_pki=0.5352	pmu_enabled=100.0	pmu_secs=3446.5	who=at_s=6,rss_pg=1509917,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.6945	l2color_cv=0.0044	l3color_cv=0.0122	contig_frac=0.9002	mean_run=9.8	vapa16=0.2324	gib_regions=28	anon_pg=23041	anon_l2cv=0.0013	anon_contig=0.9171	anon_run=11.8	anon_gib=26	file_pg=24576	file_l2cv=0.0012	file_contig=0.9959	file_run=166.1	file_gib=26	other_pg=20659	other_l2cv=0.0140	other_contig=0.7676	other_run=4.3	other_gib=28	ro_cost_ms=67-->
<!--DATA 3	stock	pp2048	19.08-->

### qwen35-9b-qres512 [qres512] pass 3  env='ROCKET_QUANT_RESIDENT=auto'  args=''  19:47:43  clk=600 MHz  MemAvail=31289472 kB
<!--PRED 3	qres512	memavail_kb=31289472	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7215,7487,6820,6108,4830,4244,3695,3287,2645,886,4289-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         33.80 ± 0.23 |

build: 171974745 (10558)
<!--RO 3	qres512	busy=5431,5223,5244,5202,8949,7374,8374,8942	busy_tot=54739	busy_little_share=0.3855	a55_cpu_cycles=382895079648	a55_inst_retired=208930528919	context_switches=7898251	a76_cpu_cycles=751622134698	a76_l3d_cache_refill=9015518032	a76_l2d_cache_refill=5236413971	cpu_migrations=44358	page_faults=10994327	a76_dtlb_walk=2000301069	a76_mem_access=379647365244	a76_inst_retired=1216262313280	a76_l1d_cache_refill=9525286692	a55_inst_share=0.1466	a76_ipc=1.618	l2ref_pki=4.305	l3ref_pki=7.412	l1dref_pki=7.832	memacc_pki=312.14	dtlbw_pki=1.6446	pmu_enabled=100.0	pmu_secs=2226.1	who=at_s=6,rss_pg=1485064,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=90112	present_frac=0.7349	l2color_cv=0.0045	l3color_cv=0.0120	contig_frac=0.6490	mean_run=2.8	vapa16=0.0414	gib_regions=28	anon_pg=20798	anon_l2cv=0.0028	anon_contig=0.2077	anon_run=1.3	anon_gib=27	file_pg=24576	file_l2cv=0.0012	file_contig=0.9951	file_run=147.2	file_gib=27	other_pg=20853	other_l2cv=0.0139	other_contig=0.6811	other_run=3.1	other_gib=28	ro_cost_ms=76-->
<!--DATA 3	qres512	pp2048	33.80-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20921MB (MemAvailable 30457MB - reserve 9535MB, no swap)

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 19.10 | 3 | 19.02 | 19.19 | -- | -- |
| pp2048 | qres512 | 33.45 | 3 | 33.03 | 33.80 | 1.752x | 1.747 1.737 1.771 |

