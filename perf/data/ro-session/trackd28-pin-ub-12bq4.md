<!-- gemma4-12b Q4_K_M, pinning x -b 2048 -ub 2048 -- the phi-gap cell
     TESTS='-p 2048 -n 0 -r 2'  PASSES=3
     gguf=<data>/gemma4/gemma-4-12b-it-Q4_K_M.gguf (7381382304 bytes)
     stock_unpin | stock_pin = PIN_MASK=0xf0 -t 4
     ub_unpin    = -b 2048 -ub 2048   (the published 1.257x arm, Q4_K_M row)
     ub_pin      = the same + PIN_MASK=0xf0 -t 4
     phi         = 1 - busy_tot(ub_pin)/busy_tot(stock_pin), the two PINNED arms
     a           = 1 - 1/K, K the UNPINNED paired ratio; H/t = a/phi; R = t*(1 - a/phi)
     interaction = (ub_pin/stock_pin) / (ub_unpin/stock_unpin), paired within a pass
     PREDICTION 2026-09-03b: R = 76 s (68-85), H/t = 0.520, phi = 0.393 (0.34-0.46)
       RIVAL A host share tracks the MODEL (0.255) -> phi = 0.802
       RIVAL B host share tracks the QUANT CLASS (0.591) -> phi = 0.346
     CONTROL: no arm may place resident weights; one that does is not this contrast
-->
<!-- RESULT, three rotated passes, 12 of 12 DATA rows, 0 failed arms, pfn_zero_frac=0.0000 and
     pmu_enabled=100 on all twelve, and the placement control PASSES: not one arm reports a
     resident weight, which is what makes this phi uncontaminated with no rep-count regression.
       stock_unpin 12.82 t/s   K unpinned 1.2618 +- 0.0042 (published 1.257x) -> a = 0.2075
       pin gain    1.0471 +- 0.0035 -> b = 0.0450          K pinned 1.2550 +- 0.0030
       phi (A76, pinned pair) = 0.4650 +- 0.0005    phi (all 8 cores) = 0.4655 +- 0.0004
       H/t = a/phi = 0.446        R = t*(1 - a/phi) = 88.5 s      t = 159.7 s
     PREDICTION 2026-09-03b MISSED ON ALL THREE NUMBERS, NARROWLY, AND BOTH RIVALS ARE REFUTED.
       R      predicted 76 s (band 68-85)  -> 88.5 s, 4% above the band
       H/t    predicted 0.520              -> 0.446
       phi    predicted 0.393 (0.34-0.46)  -> 0.4650, 0.005 outside the band
       refined bracket 0.386-0.447         -> 0.4650, above it
       RIVAL A (host share tracks the MODEL, 0.255)      -> phi 0.802, REFUTED
       RIVAL B (host share tracks the QUANT CLASS, 0.591)-> phi 0.346, REFUTED
     WHAT THE MISS SAYS. The premise was that the DEVICE term R is a property of the model, since
     both GGUFs run the same fp16 GEMMs. It is not: R reads 76.4 s on the F16 unit and 88.5 s
     here, 16% apart. That is the finding, and it is why every derived number came out low.
     THE HOST SHARE IS A PROPERTY OF NEITHER THE MODEL NOR THE QUANT CLASS: 0.255 (12B F16),
     0.446 (12B Q4_K_M), 0.591-0.602 (9B Q4_K_M). Three units, three values.
     THE KNOB REACHES THE PER-CALL PACK AS WELL AS THE DEQUANT. The pre-registered bracket was
     0.386 for dequant alone and 0.447 for dequant plus three quarters of the pack; 0.4650 is
     above both, so the knob reaches the pack and rather more besides.
     RULE 147 REFINED. Pinned phi 0.4650, unpinned all-8 0.4655 (+0.0%), unpinned A76 0.4729
     (-1.7%). Here the ALL-CORES reading is the faithful one, the opposite of the 9B cells --
     and busy_little_share moves only 0.1401 -> 0.1520 across this knob, 1.2 pp against the 9B's
     6.4 and 11.5. The bias tracks how much the knob shifts work between clusters, which the
     readout already prints. Read that column, do not apply a fixed cluster rule.
     INTERACTION, RECORDED AND NOT READ. 0.9946 +- 0.0045 against the model's 0.9849 at this phi
     and a null of 1.0000: +2.2 se from the model and -1.2 se from the null, which is the
     straddle the item predicted at three passes. It is now the largest residual of six cells.
-->
== gemma4-12b-q4-ub-pin2x2  Thu Sep  3 02:26:52 UTC 2026 ==
### gemma4-12b-q4-ub-pin2x2 [stock_unpin] pass 1  env=''  args=''  02:28:24  clk=600 MHz  MemAvail=31480004 kB
<!--PRED 1	stock_unpin	memavail_kb=31480004	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10461,11271,13775,7892,6443,5817,4961,4064,2672,3457,2355-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         12.90 ± 0.05 |

build: 171974745 (10558)
<!--RO 1	stock_unpin	wall_s=481	busy=4811,4260,4244,4208,29572,26070,25256,26222	busy_tot=124643	busy_little_share=0.1406	a55_cpu_cycles=335471262502	a55_inst_retired=208282004926	context_switches=9920368	a76_cpu_cycles=2312012752537	a76_l3d_cache_refill=18057649265	a76_l2d_cache_refill=7632091209	cpu_migrations=34505	page_faults=952267	a76_dtlb_walk=867477494	a76_mem_access=616205874346	a76_inst_retired=4909779311251	a76_l1d_cache_refill=22107298310	a55_inst_share=0.0407	a76_ipc=2.124	l2ref_pki=1.554	l3ref_pki=3.678	l1dref_pki=4.503	memacc_pki=125.51	dtlbw_pki=0.1767	pmu_enabled=100.0	pmu_cpu_s=3834.6	who=at_s=9,rss_pg=2050735,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.8571	l2color_cv=0.0066	l3color_cv=0.0146	contig_frac=0.9245	mean_run=12.9	vapa16=0.0106	gib_regions=29	anon_pg=36489	anon_l2cv=0.0005	anon_contig=0.9641	anon_run=26.4	anon_gib=28	file_pg=24576	file_l2cv=0.0051	file_contig=0.9824	file_run=51.3	file_gib=28	other_pg=23191	other_l2cv=0.0215	other_contig=0.8009	other_run=5.0	other_gib=29	ro_cost_ms=133-->
<!--DATA 1	stock_unpin	pp2048	12.90-->

### gemma4-12b-q4-ub-pin2x2 [stock_pin] pass 1  env='PIN_MASK=0xf0'  args='-t 4'  02:37:50  clk=600 MHz  MemAvail=31428284 kB
<!--PRED 1	stock_pin	memavail_kb=31427992	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8465,12598,12807,7493,6510,5682,4915,4042,2637,3404,2392-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         13.42 ± 0.03 |

build: 171974745 (10558)
<!--RO 1	stock_pin	wall_s=462	busy=297,295,254,253,29730,26338,25373,26716	busy_tot=109256	busy_little_share=0.0101	a55_cpu_cycles=44100739953	a55_inst_retired=8942514437	context_switches=9797556	a76_cpu_cycles=2338572724532	a76_l3d_cache_refill=18749553598	a76_l2d_cache_refill=7789223385	cpu_migrations=23404	page_faults=917873	a76_dtlb_walk=891145659	a76_mem_access=634364237854	a76_inst_retired=4999415736009	a76_l1d_cache_refill=23954195083	a55_inst_share=0.0018	a76_ipc=2.138	l2ref_pki=1.558	l3ref_pki=3.750	l1dref_pki=4.791	memacc_pki=126.89	dtlbw_pki=0.1782	pmu_enabled=100.0	pmu_cpu_s=3689.8	who=at_s=8,rss_pg=2049879,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.8716	l2color_cv=0.0057	l3color_cv=0.0127	contig_frac=0.9224	mean_run=12.6	vapa16=0.0172	gib_regions=29	anon_pg=37125	anon_l2cv=0.0029	anon_contig=0.9717	anon_run=33.1	anon_gib=28	file_pg=24576	file_l2cv=0.0027	file_contig=0.9802	file_run=46.0	file_gib=28	other_pg=23976	other_l2cv=0.0218	other_contig=0.7869	other_run=4.7	other_gib=29	ro_cost_ms=168-->
<!--DATA 1	stock_pin	pp2048	13.42-->

### gemma4-12b-q4-ub-pin2x2 [ub_unpin] pass 1  env=''  args='-b 2048 -ub 2048'  02:47:03  clk=600 MHz  MemAvail=31446884 kB
<!--PRED 1	ub_unpin	memavail_kb=31446300	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9107,12258,13638,7534,6495,5677,4924,4032,2614,3388,2409-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         16.17 ± 0.02 |

build: 171974745 (10558)
<!--RO 1	ub_unpin	wall_s=385	busy=2633,2519,2497,2467,17579,12539,13232,13437	busy_tot=66903	busy_little_share=0.1512	a55_cpu_cycles=192282239664	a55_inst_retired=115294763656	context_switches=9237205	a76_cpu_cycles=1236900677104	a76_l3d_cache_refill=14047565382	a76_l2d_cache_refill=5062177396	cpu_migrations=14627	page_faults=1668533	a76_dtlb_walk=657855985	a76_mem_access=321838248168	a76_inst_retired=2265572545599	a76_l1d_cache_refill=11218307383	a55_inst_share=0.0484	a76_ipc=1.832	l2ref_pki=2.234	l3ref_pki=6.200	l1dref_pki=4.952	memacc_pki=142.06	dtlbw_pki=0.2904	pmu_enabled=100.0	pmu_cpu_s=3080.2	who=at_s=10,rss_pg=2413430,settled=2	pfn_zero_frac=0.0000	maps=7	sampled=156672	present_frac=0.8088	l2color_cv=0.0027	l3color_cv=0.0070	contig_frac=0.9474	mean_run=18.3	vapa16=0.0092	gib_regions=29	anon_pg=87755	anon_l2cv=0.0008	anon_contig=0.9736	anon_run=35.3	anon_gib=27	file_pg=24576	file_l2cv=0.0057	file_contig=0.9807	file_run=47.1	file_gib=28	other_pg=14379	other_l2cv=0.0254	other_contig=0.7301	other_run=3.7	other_gib=29	ro_cost_ms=225-->
<!--DATA 1	ub_unpin	pp2048	16.17-->

### gemma4-12b-q4-ub-pin2x2 [ub_pin] pass 1  env='PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4'  02:54:53  clk=600 MHz  MemAvail=31430620 kB
<!--PRED 1	ub_pin	memavail_kb=31430620	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6967,12110,13189,7595,6385,5719,4910,4008,2592,3383,2421-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         16.87 ± 0.01 |

build: 171974745 (10558)
<!--RO 1	ub_pin	wall_s=369	busy=152,143,122,129,17580,13025,13498,13658	busy_tot=58307	busy_little_share=0.0094	a55_cpu_cycles=24494753751	a55_inst_retired=4472330643	context_switches=9202425	a76_cpu_cycles=1256984379476	a76_l3d_cache_refill=14619252807	a76_l2d_cache_refill=5170415807	cpu_migrations=6341	page_faults=1686772	a76_dtlb_walk=684368825	a76_mem_access=335564661822	a76_inst_retired=2349175809862	a76_l1d_cache_refill=11496544156	a55_inst_share=0.0019	a76_ipc=1.869	l2ref_pki=2.201	l3ref_pki=6.223	l1dref_pki=4.894	memacc_pki=142.84	dtlbw_pki=0.2913	pmu_enabled=100.0	pmu_cpu_s=2952.0	who=at_s=10,rss_pg=2413399,settled=2	pfn_zero_frac=0.0000	maps=7	sampled=156672	present_frac=0.8075	l2color_cv=0.0041	l3color_cv=0.0083	contig_frac=0.9380	mean_run=15.6	vapa16=0.2124	gib_regions=29	anon_pg=88178	anon_l2cv=0.0004	anon_contig=0.9543	anon_run=21.0	anon_gib=28	file_pg=24576	file_l2cv=0.0022	file_contig=0.9819	file_run=50.1	file_gib=27	other_pg=13757	other_l2cv=0.0395	other_contig=0.7555	other_run=4.0	other_gib=29	ro_cost_ms=149-->
<!--DATA 1	ub_pin	pp2048	16.87-->

### gemma4-12b-q4-ub-pin2x2 [stock_pin] pass 2  env='PIN_MASK=0xf0'  args='-t 4'  03:02:28  clk=600 MHz  MemAvail=31434896 kB
<!--PRED 2	stock_pin	memavail_kb=31434896	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7171,12163,13261,7487,6592,5700,4888,4005,2517,3407,2427-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         13.45 ± 0.02 |

build: 171974745 (10558)
<!--RO 2	stock_pin	wall_s=460	busy=265,286,284,281,29544,27189,24984,26027	busy_tot=108860	busy_little_share=0.0103	a55_cpu_cycles=45043700719	a55_inst_retired=9297227541	context_switches=9795310	a76_cpu_cycles=2334646678619	a76_l3d_cache_refill=18798165633	a76_l2d_cache_refill=7777462203	cpu_migrations=22820	page_faults=915814	a76_dtlb_walk=878454008	a76_mem_access=634434327143	a76_inst_retired=4999652714937	a76_l1d_cache_refill=23989188375	a55_inst_share=0.0019	a76_ipc=2.142	l2ref_pki=1.556	l3ref_pki=3.760	l1dref_pki=4.798	memacc_pki=126.90	dtlbw_pki=0.1757	pmu_enabled=100.0	pmu_cpu_s=3678.3	who=at_s=8,rss_pg=2049881,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.8697	l2color_cv=0.0081	l3color_cv=0.0137	contig_frac=0.9370	mean_run=15.4	vapa16=0.0119	gib_regions=29	anon_pg=37413	anon_l2cv=0.0060	anon_contig=0.9937	anon_run=120.3	anon_gib=27	file_pg=24576	file_l2cv=0.0033	file_contig=0.9765	file_run=39.4	file_gib=28	other_pg=23503	other_l2cv=0.0226	other_contig=0.8054	other_run=5.1	other_gib=29	ro_cost_ms=148-->
<!--DATA 2	stock_pin	pp2048	13.45-->

### gemma4-12b-q4-ub-pin2x2 [ub_unpin] pass 2  env=''  args='-b 2048 -ub 2048'  03:11:39  clk=600 MHz  MemAvail=31495880 kB
<!--PRED 2	ub_unpin	memavail_kb=31495880	anonhuge_kb=0	hugepagesz_kb=2048	buddy=10234,12442,13339,7461,6517,5696,4892,4007,2553,3400,2430-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         16.22 ± 0.00 |

build: 171974745 (10558)
<!--RO 2	ub_unpin	wall_s=385	busy=2618,2499,2537,2473,17598,12417,13423,13142	busy_tot=66707	busy_little_share=0.1518	a55_cpu_cycles=192446270693	a55_inst_retired=115625626047	context_switches=9234660	a76_cpu_cycles=1234976479911	a76_l3d_cache_refill=14061863722	a76_l2d_cache_refill=5063829122	cpu_migrations=14476	page_faults=1671029	a76_dtlb_walk=663000556	a76_mem_access=322058910926	a76_inst_retired=2265911924878	a76_l1d_cache_refill=11222202162	a55_inst_share=0.0486	a76_ipc=1.835	l2ref_pki=2.235	l3ref_pki=6.206	l1dref_pki=4.953	memacc_pki=142.13	dtlbw_pki=0.2926	pmu_enabled=100.0	pmu_cpu_s=3074.8	who=at_s=10,rss_pg=2413430,settled=2	pfn_zero_frac=0.0000	maps=8	sampled=181248	present_frac=0.8359	l2color_cv=0.0029	l3color_cv=0.0066	contig_frac=0.9309	mean_run=14.1	vapa16=0.0462	gib_regions=29	anon_pg=112956	anon_l2cv=0.0005	anon_contig=0.9406	anon_run=16.3	anon_gib=27	file_pg=24576	file_l2cv=0.0028	file_contig=0.9837	file_run=54.7	file_gib=28	other_pg=13965	other_l2cv=0.0317	other_contig=0.7588	other_run=4.1	other_gib=29	ro_cost_ms=241-->
<!--DATA 2	ub_unpin	pp2048	16.22-->

### gemma4-12b-q4-ub-pin2x2 [ub_pin] pass 2  env='PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4'  03:19:28  clk=600 MHz  MemAvail=31433592 kB
<!--PRED 2	ub_pin	memavail_kb=31433592	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8033,12311,13022,7476,6508,5659,4889,4001,2477,3415,2431-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         16.80 ± 0.00 |

build: 171974745 (10558)
<!--RO 2	ub_pin	wall_s=371	busy=129,103,135,171,17534,13031,13600,13552	busy_tot=58255	busy_little_share=0.0092	a55_cpu_cycles=24594993599	a55_inst_retired=4383191105	context_switches=9201087	a76_cpu_cycles=1257599309986	a76_l3d_cache_refill=14627987358	a76_l2d_cache_refill=5185681379	cpu_migrations=6219	page_faults=1684607	a76_dtlb_walk=682964600	a76_mem_access=335604921197	a76_inst_retired=2349360331094	a76_l1d_cache_refill=11490054741	a55_inst_share=0.0019	a76_ipc=1.868	l2ref_pki=2.207	l3ref_pki=6.226	l1dref_pki=4.891	memacc_pki=142.85	dtlbw_pki=0.2907	pmu_enabled=100.0	pmu_cpu_s=2963.9	who=at_s=10,rss_pg=2413394,settled=2	pfn_zero_frac=0.0000	maps=9	sampled=203264	present_frac=0.8448	l2color_cv=0.0070	l3color_cv=0.0144	contig_frac=0.9212	mean_run=12.4	vapa16=0.0596	gib_regions=29	anon_pg=133139	anon_l2cv=0.0071	anon_contig=0.9248	anon_run=13.0	anon_gib=29	file_pg=24576	file_l2cv=0.0035	file_contig=0.9822	file_run=50.8	file_gib=27	other_pg=13997	other_l2cv=0.0217	other_contig=0.7804	other_run=4.5	other_gib=29	ro_cost_ms=183-->
<!--DATA 2	ub_pin	pp2048	16.80-->

### gemma4-12b-q4-ub-pin2x2 [stock_unpin] pass 2  env=''  args=''  03:27:10  clk=600 MHz  MemAvail=31445612 kB
<!--PRED 2	stock_unpin	memavail_kb=31445612	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8926,11498,13539,7998,6451,5718,4917,4005,2450,3415,2432-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         12.82 ± 0.01 |

build: 171974745 (10558)
<!--RO 2	stock_unpin	wall_s=484	busy=4962,4177,4182,4123,29571,25591,25560,26627	busy_tot=124793	busy_little_share=0.1398	a55_cpu_cycles=335188623946	a55_inst_retired=208347099453	context_switches=9924850	a76_cpu_cycles=2311049850731	a76_l3d_cache_refill=18021890846	a76_l2d_cache_refill=7626903825	cpu_migrations=35438	page_faults=912118	a76_dtlb_walk=846004086	a76_mem_access=613996901926	a76_inst_retired=4901873576016	a76_l1d_cache_refill=22140256875	a55_inst_share=0.0408	a76_ipc=2.121	l2ref_pki=1.556	l3ref_pki=3.677	l1dref_pki=4.517	memacc_pki=125.26	dtlbw_pki=0.1726	pmu_enabled=100.0	pmu_cpu_s=3862.3	who=at_s=8,rss_pg=2050740,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.8622	l2color_cv=0.0049	l3color_cv=0.0123	contig_frac=0.9105	mean_run=10.9	vapa16=0.1430	gib_regions=29	anon_pg=36496	anon_l2cv=0.0020	anon_contig=0.9691	anon_run=30.5	anon_gib=26	file_pg=24576	file_l2cv=0.0015	file_contig=0.9765	file_run=39.3	file_gib=28	other_pg=23687	other_l2cv=0.0169	other_contig=0.7516	other_run=4.0	other_gib=29	ro_cost_ms=116-->
<!--DATA 2	stock_unpin	pp2048	12.82-->

### gemma4-12b-q4-ub-pin2x2 [ub_unpin] pass 3  env=''  args='-b 2048 -ub 2048'  03:36:44  clk=600 MHz  MemAvail=31482204 kB
<!--PRED 3	ub_unpin	memavail_kb=31482204	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9516,12231,13282,7719,6540,5656,4885,3985,2504,3412,2435-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         16.15 ± 0.06 |

build: 171974745 (10558)
<!--RO 3	ub_unpin	wall_s=386	busy=2664,2558,2520,2435,17447,12622,13018,13213	busy_tot=66477	busy_little_share=0.1531	a55_cpu_cycles=192334735294	a55_inst_retired=115476992051	context_switches=9247723	a76_cpu_cycles=1229410159796	a76_l3d_cache_refill=14029742481	a76_l2d_cache_refill=5038229603	cpu_migrations=14708	page_faults=1672219	a76_dtlb_walk=666386966	a76_mem_access=321783994634	a76_inst_retired=2265270198993	a76_l1d_cache_refill=11220790953	a55_inst_share=0.0485	a76_ipc=1.843	l2ref_pki=2.224	l3ref_pki=6.193	l1dref_pki=4.953	memacc_pki=142.05	dtlbw_pki=0.2942	pmu_enabled=100.0	pmu_cpu_s=3083.9	who=at_s=10,rss_pg=2413430,settled=2	pfn_zero_frac=0.0000	maps=7	sampled=156672	present_frac=0.8059	l2color_cv=0.0033	l3color_cv=0.0064	contig_frac=0.9029	mean_run=10.1	vapa16=0.0086	gib_regions=29	anon_pg=87908	anon_l2cv=0.0006	anon_contig=0.9190	anon_run=12.1	anon_gib=27	file_pg=24576	file_l2cv=0.0009	file_contig=0.9859	file_run=62.2	file_gib=27	other_pg=13780	other_l2cv=0.0291	other_contig=0.6518	other_run=2.9	other_gib=29	ro_cost_ms=209-->
<!--DATA 3	ub_unpin	pp2048	16.15-->

### gemma4-12b-q4-ub-pin2x2 [ub_pin] pass 3  env='PIN_MASK=0xf0'  args='-b 2048 -ub 2048 -t 4'  03:44:35  clk=600 MHz  MemAvail=31432304 kB
<!--PRED 3	ub_pin	memavail_kb=31432464	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8511,12321,13145,7607,6507,5682,4870,3976,2488,3404,2435-->
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |       4 |     2048 |          pp2048 |         16.88 ± 0.04 |

build: 171974745 (10558)
<!--RO 3	ub_pin	wall_s=369	busy=126,152,108,128,17463,13069,13714,13466	busy_tot=58226	busy_little_share=0.0088	a55_cpu_cycles=24187829544	a55_inst_retired=4387441772	context_switches=9189000	a76_cpu_cycles=1258343400255	a76_l3d_cache_refill=14640774451	a76_l2d_cache_refill=5179924354	cpu_migrations=6094	page_faults=1684847	a76_dtlb_walk=677450178	a76_mem_access=335507640267	a76_inst_retired=2349176553323	a76_l1d_cache_refill=11494734117	a55_inst_share=0.0019	a76_ipc=1.867	l2ref_pki=2.205	l3ref_pki=6.232	l1dref_pki=4.893	memacc_pki=142.82	dtlbw_pki=0.2884	pmu_enabled=100.0	pmu_cpu_s=2949.0	who=at_s=10,rss_pg=2413401,settled=2	pfn_zero_frac=0.0000	maps=8	sampled=181248	present_frac=0.8301	l2color_cv=0.0033	l3color_cv=0.0067	contig_frac=0.9279	mean_run=13.5	vapa16=0.0231	gib_regions=29	anon_pg=112133	anon_l2cv=0.0006	anon_contig=0.9375	anon_run=15.5	anon_gib=27	file_pg=24576	file_l2cv=0.0017	file_contig=0.9849	file_run=58.8	file_gib=27	other_pg=13748	other_l2cv=0.0346	other_contig=0.7473	other_run=3.9	other_gib=29	ro_cost_ms=185-->
<!--DATA 3	ub_pin	pp2048	16.88-->

### gemma4-12b-q4-ub-pin2x2 [stock_unpin] pass 3  env=''  args=''  03:52:16  clk=600 MHz  MemAvail=31479232 kB
<!--PRED 3	stock_unpin	memavail_kb=31479416	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7985,12058,13300,7576,6497,5697,4881,3986,2507,3414,2435-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         12.75 ± 0.02 |

build: 171974745 (10558)
<!--RO 3	stock_unpin	wall_s=485	busy=4865,4229,4202,4186,29653,25537,25375,26867	busy_tot=124914	busy_little_share=0.1400	a55_cpu_cycles=333876672391	a55_inst_retired=207679382906	context_switches=9925611	a76_cpu_cycles=2311580327962	a76_l3d_cache_refill=18020864988	a76_l2d_cache_refill=7631533227	cpu_migrations=34826	page_faults=909047	a76_dtlb_walk=861625192	a76_mem_access=613984232049	a76_inst_retired=4902393399302	a76_l1d_cache_refill=22114984266	a55_inst_share=0.0406	a76_ipc=2.121	l2ref_pki=1.557	l3ref_pki=3.676	l1dref_pki=4.511	memacc_pki=125.24	dtlbw_pki=0.1758	pmu_enabled=100.0	pmu_cpu_s=3873.9	who=at_s=8,rss_pg=2050733,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.8573	l2color_cv=0.0073	l3color_cv=0.0148	contig_frac=0.9242	mean_run=12.9	vapa16=0.0203	gib_regions=29	anon_pg=36496	anon_l2cv=0.0016	anon_contig=0.9783	anon_run=42.2	anon_gib=27	file_pg=24576	file_l2cv=0.0026	file_contig=0.9841	file_run=56.0	file_gib=27	other_pg=23207	other_l2cv=0.0255	other_contig=0.7758	other_run=4.4	other_gib=29	ro_cost_ms=79-->
<!--DATA 3	stock_unpin	pp2048	12.75-->

### gemma4-12b-q4-ub-pin2x2 [stock_pin] pass 3  env='PIN_MASK=0xf0'  args='-t 4'  04:01:46  clk=600 MHz  MemAvail=31448428 kB
<!--PRED 3	stock_pin	memavail_kb=31447844	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8299,12694,13544,7401,6478,5581,4884,3987,2472,3409,2441-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         13.41 ± 0.00 |

build: 171974745 (10558)
<!--RO 3	stock_pin	wall_s=462	busy=300,242,253,293,29562,26919,25159,26185	busy_tot=108913	busy_little_share=0.0100	a55_cpu_cycles=43717522649	a55_inst_retired=8734752275	context_switches=9794372	a76_cpu_cycles=2336758620459	a76_l3d_cache_refill=18771830363	a76_l2d_cache_refill=7787435293	cpu_migrations=22472	page_faults=916002	a76_dtlb_walk=879316641	a76_mem_access=634350104789	a76_inst_retired=4999520211268	a76_l1d_cache_refill=23962725761	a55_inst_share=0.0017	a76_ipc=2.140	l2ref_pki=1.558	l3ref_pki=3.755	l1dref_pki=4.793	memacc_pki=126.88	dtlbw_pki=0.1759	pmu_enabled=100.0	pmu_cpu_s=3689.7	who=at_s=8,rss_pg=2049874,settled=2	pfn_zero_frac=0.0000	maps=4	sampled=98304	present_frac=0.8671	l2color_cv=0.0042	l3color_cv=0.0102	contig_frac=0.9261	mean_run=13.2	vapa16=0.0281	gib_regions=29	anon_pg=37239	anon_l2cv=0.0023	anon_contig=0.9657	anon_run=27.6	anon_gib=27	file_pg=24576	file_l2cv=0.0037	file_contig=0.9845	file_run=57.4	file_gib=27	other_pg=23424	other_l2cv=0.0182	other_contig=0.8018	other_run=5.0	other_gib=29	ro_cost_ms=150-->
<!--DATA 3	stock_pin	pp2048	13.41-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock_unpin]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock_unpin | 12.82 | 3 | 12.75 | 12.90 | -- | -- |
| pp2048 | stock_pin | 13.43 | 3 | 13.41 | 13.45 | 1.047x | 1.040 1.049 1.052 |
| pp2048 | ub_unpin | 16.18 | 3 | 16.15 | 16.22 | 1.262x | 1.253 1.265 1.267 |
| pp2048 | ub_pin | 16.85 | 3 | 16.80 | 16.88 | 1.314x | 1.308 1.310 1.324 |

