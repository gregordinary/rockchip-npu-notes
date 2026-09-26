<!-- gemma4-12b F16, PINNED (PIN_MASK=0xf0 -t 4), streamed: the FA handler's critical-path interval
     TESTS='-p 2048 -n 0 -r 2'  PASSES=1  both arms -v so GGML_LOG_INFO reaches stderr
     gguf=<data>/gemma4/gemma-4-12b-it-F16.gguf (23832065184 bytes)
     fa_plain = the published pin76t4 arm; fa_probe = the same + ROCKET_FA_TIMING=1
     share = (gather+compute+scatter) / ((r+1)*2048/t_s)
     PREDICTION 2026-09-03d: gather+scatter 2.5-3.5% of wall, compute 8-17%, total 11-20%;
       probe overhead < 1% of t/s. RULE: total < 12% closes a vectorised-exp build below 4%.
       RIVAL: gather+scatter > 5% -> the kernel's unnamed 6.77%/5.91% entries are the gather's.
     CONTROL: both arms streamed (0 resident lines)
-->
== gemma4-12b-f16-fatiming  Thu Sep  3 16:45:20 UTC 2026 ==
### gemma4-12b-f16-fatiming [fa_plain] pass 1  env='PIN_MASK=0xf0'  args='-t 4 -v'  16:46:42  clk=600 MHz  MemAvail=31247560 kB
<!--PRED 1	fa_plain	memavail_kb=31248060	anonhuge_kb=0	hugepagesz_kb=2048	buddy=1746,3643,7658,7278,6092,5440,4993,4083,2100,127,238-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         21.24 ± 0.02 |

build: 171974745 (10558)
<!--RO 1	fa_plain	wall_s=294	busy=322,195,284,250,14036,9296,10242,12324	busy_tot=46949	busy_little_share=0.0224	a55_cpu_cycles=41730581525	a55_inst_retired=8446400270	context_switches=9712937	a76_cpu_cycles=1016041197234	a76_l3d_cache_refill=14985557911	a76_l2d_cache_refill=7111087059	cpu_migrations=19360	page_faults=944341	a76_dtlb_walk=822580441	a76_mem_access=278577031007	a76_inst_retired=1288561174673	a76_l1d_cache_refill=20079385471	a55_inst_share=0.0065	a76_ipc=1.268	l2ref_pki=5.519	l3ref_pki=11.630	l1dref_pki=15.583	memacc_pki=216.19	dtlbw_pki=0.6384	pmu_enabled=100.0	pmu_cpu_s=2350.7	who=at_s=8,rss_pg=6044936,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8340	l2color_cv=0.0073	l3color_cv=0.0135	contig_frac=0.9323	mean_run=14.4	vapa16=0.0458	gib_regions=32	anon_pg=12942	anon_l2cv=0.0020	anon_contig=0.9359	anon_run=15.1	anon_gib=31	file_pg=24576	file_l2cv=0.0003	file_contig=0.9953	file_run=150.8	file_gib=29	other_pg=23970	other_l2cv=0.0190	other_contig=0.8656	other_run=7.3	other_gib=32	ro_cost_ms=81-->
<!--DATA 1	fa_plain	pp2048	21.24-->

### gemma4-12b-f16-fatiming [fa_probe] pass 1  env='ROCKET_FA_TIMING=1 PIN_MASK=0xf0'  args='-t 4 -v'  16:53:00  clk=600 MHz  MemAvail=31315656 kB
<!--PRED 1	fa_probe	memavail_kb=31315656	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7295,7768,9053,7584,6424,5476,4773,4073,2093,172,220-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         20.93 ± 0.05 |

build: 171974745 (10558)
<!--RO 1	fa_probe	wall_s=299	busy=277,288,232,239,14154,9428,9497,13149	busy_tot=47264	busy_little_share=0.0219	a55_cpu_cycles=41752278104	a55_inst_retired=8268169519	context_switches=9732480	a76_cpu_cycles=1018320256468	a76_l3d_cache_refill=15040881782	a76_l2d_cache_refill=7089693755	cpu_migrations=19341	page_faults=943256	a76_dtlb_walk=822811260	a76_mem_access=279213445193	a76_inst_retired=1291827843288	a76_l1d_cache_refill=20138605305	a55_inst_share=0.0064	a76_ipc=1.269	l2ref_pki=5.488	l3ref_pki=11.643	l1dref_pki=15.589	memacc_pki=216.14	dtlbw_pki=0.6369	pmu_enabled=100.0	pmu_cpu_s=2383.6	who=at_s=8,rss_pg=6044936,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8226	l2color_cv=0.0054	l3color_cv=0.0134	contig_frac=0.9134	mean_run=11.3	vapa16=0.0180	gib_regions=32	anon_pg=12932	anon_l2cv=0.0078	anon_contig=0.9483	anon_run=18.7	anon_gib=29	file_pg=24576	file_l2cv=0.0020	file_contig=0.9873	file_run=68.3	file_gib=31	other_pg=23140	other_l2cv=0.0167	other_contig=0.8154	other_run=5.4	other_gib=32	ro_cost_ms=85-->
<!--DATA 1	fa_probe	pp2048	20.93-->

#### summary: per-arm mean over 1 passes, ratios paired within a pass against [fa_plain]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | fa_plain | 21.24 | 1 | 21.24 | 21.24 | -- | -- |
| pp2048 | fa_probe | 20.93 | 1 | 20.93 | 20.93 | 0.985x | 0.985 |

