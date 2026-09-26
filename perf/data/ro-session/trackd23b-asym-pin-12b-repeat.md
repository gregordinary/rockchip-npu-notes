<!-- gemma4-12b F16, pinning x ROCKET_MM_ASYM (a DEVICE-tiling knob) 2x2
     TESTS='-p 2048 -n 0 -r 3'  PASSES=3
     gguf=<data>/gemma4/gemma-4-12b-it-F16.gguf (23832065184 bytes)
     asym0_unpin = ROCKET_MM_ASYM=0, the knob DECLINED; it is the base of every ratio
     stock_*     = the shipping default, ROCKET_MM_ASYM absent
     *_pin       = PIN_MASK=0xf0 -t 4, matching all three earlier interaction cells
     interaction = (stock_pin/asym0_pin) / (stock_unpin/asym0_unpin), paired within a pass
     no arm places resident weights; an arm that reports any is not this contrast
-->
== gemma4-12b-asym-pin2x2  Wed Sep  2 20:14:24 UTC 2026 ==
### gemma4-12b-asym-pin2x2 [asym0_unpin] pass 1  env='ROCKET_MM_ASYM=0'  args=''  20:15:59  clk=600 MHz  MemAvail=31444632 kB
<!--PRED 1	asym0_unpin	memavail_kb=31444772	anonhuge_kb=0	hugepagesz_kb=2048	buddy=2274,5077,9490,7201,6094,5334,3791,3763,2498,160,252-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         18.52 ± 0.08 |

build: 171974745 (10558)
<!--RO 1	asym0_unpin	wall_s=448	busy=5602,5548,5564,5568,19090,13095,12829,15298	busy_tot=82594	busy_little_share=0.2698	a55_cpu_cycles=424451574120	a55_inst_retired=266678381772	context_switches=9900068	a76_cpu_cycles=1326905377054	a76_l3d_cache_refill=20076654341	a76_l2d_cache_refill=10718345632	cpu_migrations=40465	page_faults=995885	a76_dtlb_walk=1145249460	a76_mem_access=339411239274	a76_inst_retired=1591437902163	a76_l1d_cache_refill=23113097680	a55_inst_share=0.1435	a76_ipc=1.199	l2ref_pki=6.735	l3ref_pki=12.615	l1dref_pki=14.523	memacc_pki=213.27	dtlbw_pki=0.7196	pmu_enabled=100.0	pmu_cpu_s=3577.2	who=at_s=8,rss_pg=6044635,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8214	l2color_cv=0.0158	l3color_cv=0.0197	contig_frac=0.9364	mean_run=15.3	vapa16=0.1069	gib_regions=32	anon_pg=12303	anon_l2cv=0.0067	anon_contig=0.9783	anon_run=42.1	anon_gib=31	file_pg=24576	file_l2cv=0.0053	file_contig=0.9935	file_run=118.7	file_gib=32	other_pg=23680	other_l2cv=0.0337	other_contig=0.8553	other_run=6.8	other_gib=32	ro_cost_ms=83-->
<!--DATA 1	asym0_unpin	pp2048	18.52-->

### gemma4-12b-asym-pin2x2 [asym0_pin] pass 1  env='ROCKET_MM_ASYM=0 PIN_MASK=0xf0'  args='-t 4'  20:24:55  clk=600 MHz  MemAvail=31407568 kB
<!--PRED 1	asym0_pin	memavail_kb=31407568	anonhuge_kb=0	hugepagesz_kb=2048	buddy=2072,4183,7395,6757,5726,5239,3994,3766,2502,175,243-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         19.41 ± 0.08 |

build: 171974745 (10558)
<!--RO 1	asym0_pin	wall_s=427	busy=431,382,413,277,19889,12914,13656,16152	busy_tot=64114	busy_little_share=0.0234	a55_cpu_cycles=57358414318	a55_inst_retired=12086130504	context_switches=9738335	a76_cpu_cycles=1370089827780	a76_l3d_cache_refill=21045129421	a76_l2d_cache_refill=10958340072	cpu_migrations=25205	page_faults=987572	a76_dtlb_walk=1151683918	a76_mem_access=364301906335	a76_inst_retired=1710338210473	a76_l1d_cache_refill=28936462011	a55_inst_share=0.0070	a76_ipc=1.248	l2ref_pki=6.407	l3ref_pki=12.305	l1dref_pki=16.919	memacc_pki=213.00	dtlbw_pki=0.6734	pmu_enabled=100.0	pmu_cpu_s=3414.1	who=at_s=8,rss_pg=6044317,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8213	l2color_cv=0.0108	l3color_cv=0.0167	contig_frac=0.9010	mean_run=9.9	vapa16=0.0935	gib_regions=32	anon_pg=12932	anon_l2cv=0.0071	anon_contig=0.9849	anon_run=58.5	anon_gib=31	file_pg=24576	file_l2cv=0.0089	file_contig=0.9928	file_run=109.7	file_gib=32	other_pg=23043	other_l2cv=0.0179	other_contig=0.7560	other_run=4.1	other_gib=32	ro_cost_ms=93-->
<!--DATA 1	asym0_pin	pp2048	19.41-->

### gemma4-12b-asym-pin2x2 [stock_unpin] pass 1  env=''  args=''  20:33:32  clk=600 MHz  MemAvail=31405460 kB
<!--PRED 1	stock_unpin	memavail_kb=31405460	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3093,3607,7102,6854,5896,5265,3970,3773,2498,177,240-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         19.84 ± 0.08 |

build: 171974745 (10558)
<!--RO 1	stock_unpin	wall_s=418	busy=5460,5520,5452,5376,18431,12272,13001,15475	busy_tot=80987	busy_little_share=0.2693	a55_cpu_cycles=417706487644	a55_inst_retired=260820083444	context_switches=13120639	a76_cpu_cycles=1310782234839	a76_l3d_cache_refill=19108564000	a76_l2d_cache_refill=9239277712	cpu_migrations=40337	page_faults=952056	a76_dtlb_walk=1062059011	a76_mem_access=345291732243	a76_inst_retired=1597976692179	a76_l1d_cache_refill=24361818941	a55_inst_share=0.1403	a76_ipc=1.219	l2ref_pki=5.782	l3ref_pki=11.958	l1dref_pki=15.245	memacc_pki=216.08	dtlbw_pki=0.6646	pmu_enabled=100.0	pmu_cpu_s=3338.6	who=at_s=8,rss_pg=6045726,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8163	l2color_cv=0.0111	l3color_cv=0.0179	contig_frac=0.8600	mean_run=7.1	vapa16=0.0187	gib_regions=32	anon_pg=12303	anon_l2cv=0.0091	anon_contig=0.9624	anon_run=25.2	anon_gib=32	file_pg=24576	file_l2cv=0.0016	file_contig=0.9923	file_run=103.3	file_gib=31	other_pg=23303	other_l2cv=0.0278	other_contig=0.6665	other_run=3.0	other_gib=32	ro_cost_ms=70-->
<!--DATA 1	stock_unpin	pp2048	19.84-->

### gemma4-12b-asym-pin2x2 [stock_pin] pass 1  env='PIN_MASK=0xf0'  args='-t 4'  20:41:55  clk=600 MHz  MemAvail=31372596 kB
<!--PRED 1	stock_pin	memavail_kb=31372596	anonhuge_kb=0	hugepagesz_kb=2048	buddy=2466,3810,7286,6984,5972,5154,3814,3759,2497,180,243-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         21.19 ± 0.08 |

build: 171974745 (10558)
<!--RO 1	stock_pin	wall_s=392	busy=325,362,342,350,18814,12776,12671,16929	busy_tot=62569	busy_little_share=0.0220	a55_cpu_cycles=54115637196	a55_inst_retired=10974441407	context_switches=12966342	a76_cpu_cycles=1353330405997	a76_l3d_cache_refill=20002600194	a76_l2d_cache_refill=9493831798	cpu_migrations=26240	page_faults=959478	a76_dtlb_walk=1091443783	a76_mem_access=369785981322	a76_inst_retired=1713744707361	a76_l1d_cache_refill=22421290693	a55_inst_share=0.0064	a76_ipc=1.266	l2ref_pki=5.540	l3ref_pki=11.672	l1dref_pki=13.083	memacc_pki=215.78	dtlbw_pki=0.6369	pmu_enabled=100.0	pmu_cpu_s=3128.0	who=at_s=8,rss_pg=6044870,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8260	l2color_cv=0.0056	l3color_cv=0.0141	contig_frac=0.9210	mean_run=12.4	vapa16=0.1013	gib_regions=32	anon_pg=12861	anon_l2cv=0.0208	anon_contig=0.9490	anon_run=18.9	anon_gib=30	file_pg=24576	file_l2cv=0.0023	file_contig=0.9946	file_run=135.8	file_gib=31	other_pg=23460	other_l2cv=0.0203	other_contig=0.8285	other_run=5.8	other_gib=32	ro_cost_ms=86-->
<!--DATA 1	stock_pin	pp2048	21.19-->

### gemma4-12b-asym-pin2x2 [asym0_pin] pass 2  env='ROCKET_MM_ASYM=0 PIN_MASK=0xf0'  args='-t 4'  20:49:55  clk=600 MHz  MemAvail=31423900 kB
<!--PRED 2	asym0_pin	memavail_kb=31423900	anonhuge_kb=0	hugepagesz_kb=2048	buddy=3023,5132,8131,6854,5952,5180,3895,3754,2494,175,249-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         19.12 ± 0.04 |

build: 171974745 (10558)
<!--RO 2	asym0_pin	wall_s=434	busy=441,379,316,342,19551,13873,13296,15937	busy_tot=64135	busy_little_share=0.0230	a55_cpu_cycles=57489175067	a55_inst_retired=12232624840	context_switches=9763533	a76_cpu_cycles=1374926539243	a76_l3d_cache_refill=21017803762	a76_l2d_cache_refill=10950531634	cpu_migrations=25984	page_faults=988423	a76_dtlb_walk=1163349922	a76_mem_access=364512539996	a76_inst_retired=1710487802461	a76_l1d_cache_refill=29013693458	a55_inst_share=0.0071	a76_ipc=1.244	l2ref_pki=6.402	l3ref_pki=12.288	l1dref_pki=16.962	memacc_pki=213.10	dtlbw_pki=0.6801	pmu_enabled=100.0	pmu_cpu_s=3459.7	who=at_s=8,rss_pg=6044318,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8220	l2color_cv=0.0107	l3color_cv=0.0181	contig_frac=0.8729	mean_run=7.8	vapa16=0.0666	gib_regions=32	anon_pg=12932	anon_l2cv=0.0007	anon_contig=0.9985	anon_run=281.1	anon_gib=4	file_pg=24576	file_l2cv=0.0088	file_contig=0.9628	file_run=25.6	file_gib=32	other_pg=23095	other_l2cv=0.0316	other_contig=0.7070	other_run=3.4	other_gib=32	ro_cost_ms=110-->
<!--DATA 2	asym0_pin	pp2048	19.12-->

### gemma4-12b-asym-pin2x2 [stock_unpin] pass 2  env=''  args=''  20:58:38  clk=600 MHz  MemAvail=31533372 kB
<!--PRED 2	stock_unpin	memavail_kb=31532496	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6757,6783,9944,8240,6080,5262,3794,3763,2500,183,246-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         20.01 ± 0.06 |

build: 171974745 (10558)
<!--RO 2	stock_unpin	wall_s=415	busy=5394,5426,5440,5411,18281,12146,12963,15701	busy_tot=80762	busy_little_share=0.2683	a55_cpu_cycles=416926791736	a55_inst_retired=260331758917	context_switches=13118324	a76_cpu_cycles=1312468895844	a76_l3d_cache_refill=19072413277	a76_l2d_cache_refill=9220283821	cpu_migrations=40508	page_faults=956648	a76_dtlb_walk=1073056172	a76_mem_access=345357943907	a76_inst_retired=1598854717460	a76_l1d_cache_refill=24286186300	a55_inst_share=0.1400	a76_ipc=1.218	l2ref_pki=5.767	l3ref_pki=11.929	l1dref_pki=15.190	memacc_pki=216.00	dtlbw_pki=0.6711	pmu_enabled=100.0	pmu_cpu_s=3312.1	who=at_s=8,rss_pg=6045721,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8123	l2color_cv=0.0126	l3color_cv=0.0185	contig_frac=0.9241	mean_run=12.8	vapa16=0.0106	gib_regions=32	anon_pg=12303	anon_l2cv=0.0029	anon_contig=0.9761	anon_run=38.4	anon_gib=32	file_pg=24576	file_l2cv=0.0028	file_contig=0.9863	file_run=64.0	file_gib=32	other_pg=23008	other_l2cv=0.0307	other_contig=0.8299	other_run=5.8	other_gib=32	ro_cost_ms=69-->
<!--DATA 2	stock_unpin	pp2048	20.01-->

### gemma4-12b-asym-pin2x2 [stock_pin] pass 2  env='PIN_MASK=0xf0'  args='-t 4'  21:06:57  clk=600 MHz  MemAvail=31496208 kB
<!--PRED 2	stock_pin	memavail_kb=31495908	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4738,5374,9714,7047,6112,5220,3905,3760,2495,160,258-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         21.07 ± 0.05 |

build: 171974745 (10558)
<!--RO 2	stock_pin	wall_s=394	busy=409,289,406,291,19111,12436,13304,16217	busy_tot=62463	busy_little_share=0.0223	a55_cpu_cycles=54861934265	a55_inst_retired=10955847312	context_switches=12955097	a76_cpu_cycles=1348037763178	a76_l3d_cache_refill=19962383483	a76_l2d_cache_refill=9422483094	cpu_migrations=25745	page_faults=955421	a76_dtlb_walk=1102278151	a76_mem_access=369468605720	a76_inst_retired=1712252889366	a76_l1d_cache_refill=22385505673	a55_inst_share=0.0064	a76_ipc=1.270	l2ref_pki=5.503	l3ref_pki=11.659	l1dref_pki=13.074	memacc_pki=215.78	dtlbw_pki=0.6438	pmu_enabled=100.0	pmu_cpu_s=3148.5	who=at_s=8,rss_pg=6044863,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8338	l2color_cv=0.0077	l3color_cv=0.0165	contig_frac=0.8921	mean_run=9.1	vapa16=0.1236	gib_regions=32	anon_pg=12932	anon_l2cv=0.0045	anon_contig=0.9587	anon_run=23.1	anon_gib=27	file_pg=24576	file_l2cv=0.0015	file_contig=0.9918	file_run=99.1	file_gib=31	other_pg=23967	other_l2cv=0.0206	other_contig=0.7540	other_run=4.0	other_gib=32	ro_cost_ms=93-->
<!--DATA 2	stock_pin	pp2048	21.07-->

### gemma4-12b-asym-pin2x2 [asym0_unpin] pass 2  env='ROCKET_MM_ASYM=0'  args=''  21:15:06  clk=600 MHz  MemAvail=31571424 kB
<!--PRED 2	asym0_unpin	memavail_kb=31570840	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8397,9600,9908,8627,6130,5278,3798,3747,2491,175,251-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         18.90 ± 0.07 |

build: 171974745 (10558)
<!--RO 2	asym0_unpin	wall_s=439	busy=5478,5518,5479,5423,18971,12575,12967,15918	busy_tot=82329	busy_little_share=0.2660	a55_cpu_cycles=419869347877	a55_inst_retired=263553053737	context_switches=9871692	a76_cpu_cycles=1327163150020	a76_l3d_cache_refill=20137519373	a76_l2d_cache_refill=10719669065	cpu_migrations=39686	page_faults=986215	a76_dtlb_walk=1125380873	a76_mem_access=339518117834	a76_inst_retired=1592943550960	a76_l1d_cache_refill=23110954467	a55_inst_share=0.1420	a76_ipc=1.200	l2ref_pki=6.729	l3ref_pki=12.642	l1dref_pki=14.508	memacc_pki=213.14	dtlbw_pki=0.7065	pmu_enabled=100.0	pmu_cpu_s=3506.1	who=at_s=8,rss_pg=6044640,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8203	l2color_cv=0.0092	l3color_cv=0.0151	contig_frac=0.9073	mean_run=10.6	vapa16=0.0248	gib_regions=32	anon_pg=12303	anon_l2cv=0.0023	anon_contig=0.9807	anon_run=46.8	anon_gib=20	file_pg=24576	file_l2cv=0.0006	file_contig=0.9915	file_run=95.6	file_gib=31	other_pg=23599	other_l2cv=0.0228	other_contig=0.7814	other_run=4.5	other_gib=32	ro_cost_ms=59-->
<!--DATA 2	asym0_unpin	pp2048	18.90-->

### gemma4-12b-asym-pin2x2 [stock_unpin] pass 3  env=''  args=''  21:23:55  clk=600 MHz  MemAvail=31396236 kB
<!--PRED 3	stock_unpin	memavail_kb=31396236	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4236,6136,8040,7065,6189,5120,3698,3761,2494,166,251-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         20.06 ± 0.09 |

build: 171974745 (10558)
<!--RO 3	stock_unpin	wall_s=414	busy=5509,5444,5446,5342,18406,12020,12320,16393	busy_tot=80880	busy_little_share=0.2688	a55_cpu_cycles=418295581454	a55_inst_retired=262054099671	context_switches=13129623	a76_cpu_cycles=1311940709452	a76_l3d_cache_refill=19117189968	a76_l2d_cache_refill=9239253705	cpu_migrations=40781	page_faults=953643	a76_dtlb_walk=1068482164	a76_mem_access=345178523754	a76_inst_retired=1597368895372	a76_l1d_cache_refill=24304105884	a55_inst_share=0.1409	a76_ipc=1.218	l2ref_pki=5.784	l3ref_pki=11.968	l1dref_pki=15.215	memacc_pki=216.09	dtlbw_pki=0.6689	pmu_enabled=100.0	pmu_cpu_s=3305.3	who=at_s=8,rss_pg=6045719,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8123	l2color_cv=0.0051	l3color_cv=0.0136	contig_frac=0.9565	mean_run=22.0	vapa16=0.0693	gib_regions=32	anon_pg=12303	anon_l2cv=0.0043	anon_contig=0.9787	anon_run=42.7	anon_gib=29	file_pg=24576	file_l2cv=0.0003	file_contig=0.9967	file_run=192.0	file_gib=29	other_pg=23008	other_l2cv=0.0137	other_contig=0.9016	other_run=10.0	other_gib=32	ro_cost_ms=77-->
<!--DATA 3	stock_unpin	pp2048	20.06-->

### gemma4-12b-asym-pin2x2 [stock_pin] pass 3  env='PIN_MASK=0xf0'  args='-t 4'  21:32:13  clk=600 MHz  MemAvail=31406620 kB
<!--PRED 3	stock_pin	memavail_kb=31406620	anonhuge_kb=0	hugepagesz_kb=2048	buddy=2010,3732,7373,6658,5922,5182,3947,3753,2491,219,229-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         21.31 ± 0.02 |

build: 171974745 (10558)
<!--RO 3	stock_pin	wall_s=390	busy=389,322,319,317,18978,12518,12573,16929	busy_tot=62345	busy_little_share=0.0216	a55_cpu_cycles=55140301965	a55_inst_retired=10990756919	context_switches=12948566	a76_cpu_cycles=1351834697247	a76_l3d_cache_refill=19937827263	a76_l2d_cache_refill=9406386455	cpu_migrations=26200	page_faults=954168	a76_dtlb_walk=1099400532	a76_mem_access=369661262583	a76_inst_retired=1712963924603	a76_l1d_cache_refill=22381334509	a55_inst_share=0.0064	a76_ipc=1.267	l2ref_pki=5.491	l3ref_pki=11.639	l1dref_pki=13.066	memacc_pki=215.80	dtlbw_pki=0.6418	pmu_enabled=100.0	pmu_cpu_s=3114.2	who=at_s=8,rss_pg=6044870,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8226	l2color_cv=0.0091	l3color_cv=0.0193	contig_frac=0.7951	mean_run=4.8	vapa16=0.0306	gib_regions=32	anon_pg=12932	anon_l2cv=0.0089	anon_contig=0.9584	anon_run=23.0	anon_gib=31	file_pg=24576	file_l2cv=0.0026	file_contig=0.9923	file_run=103.7	file_gib=30	other_pg=23140	other_l2cv=0.0214	other_contig=0.4943	other_run=2.0	other_gib=32	ro_cost_ms=68-->
<!--DATA 3	stock_pin	pp2048	21.31-->

### gemma4-12b-asym-pin2x2 [asym0_unpin] pass 3  env='ROCKET_MM_ASYM=0'  args=''  21:40:17  clk=600 MHz  MemAvail=31510128 kB
<!--PRED 3	asym0_unpin	memavail_kb=31510128	anonhuge_kb=0	hugepagesz_kb=2048	buddy=2093,5224,9367,8285,6034,5257,3860,3752,2487,185,249-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         18.37 ± 0.07 |

build: 171974745 (10558)
<!--RO 3	asym0_unpin	wall_s=452	busy=5483,5488,5517,5529,19321,12539,13195,15472	busy_tot=82544	busy_little_share=0.2667	a55_cpu_cycles=421212868034	a55_inst_retired=264146338715	context_switches=9913206	a76_cpu_cycles=1330219592404	a76_l3d_cache_refill=20114695005	a76_l2d_cache_refill=10717207853	cpu_migrations=39586	page_faults=986760	a76_dtlb_walk=1134786124	a76_mem_access=339426618410	a76_inst_retired=1593121999325	a76_l1d_cache_refill=23025245978	a55_inst_share=0.1422	a76_ipc=1.198	l2ref_pki=6.727	l3ref_pki=12.626	l1dref_pki=14.453	memacc_pki=213.06	dtlbw_pki=0.7123	pmu_enabled=100.0	pmu_cpu_s=3605.8	who=at_s=8,rss_pg=6044634,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8226	l2color_cv=0.0234	l3color_cv=0.0416	contig_frac=0.9250	mean_run=13.0	vapa16=0.0352	gib_regions=32	anon_pg=12303	anon_l2cv=0.0055	anon_contig=0.9795	anon_run=44.3	anon_gib=28	file_pg=24576	file_l2cv=0.0014	file_contig=0.9962	file_run=174.3	file_gib=28	other_pg=23770	other_l2cv=0.0604	other_contig=0.8231	other_run=5.6	other_gib=32	ro_cost_ms=83-->
<!--DATA 3	asym0_unpin	pp2048	18.37-->

### gemma4-12b-asym-pin2x2 [asym0_pin] pass 3  env='ROCKET_MM_ASYM=0 PIN_MASK=0xf0'  args='-t 4'  21:49:18  clk=600 MHz  MemAvail=31493976 kB
<!--PRED 3	asym0_pin	memavail_kb=31493684	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5101,6168,9260,7040,5937,5286,3949,3751,2492,173,251-->
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |       4 |          pp2048 |         19.48 ± 0.04 |

build: 171974745 (10558)
<!--RO 3	asym0_pin	wall_s=426	busy=320,334,358,408,19616,13393,13495,15968	busy_tot=63892	busy_little_share=0.0222	a55_cpu_cycles=56245177887	a55_inst_retired=11951149329	context_switches=9753856	a76_cpu_cycles=1370914511114	a76_l3d_cache_refill=21020949565	a76_l2d_cache_refill=10896920817	cpu_migrations=25840	page_faults=985580	a76_dtlb_walk=1172669584	a76_mem_access=364494920936	a76_inst_retired=1710466103822	a76_l1d_cache_refill=29044461521	a55_inst_share=0.0069	a76_ipc=1.248	l2ref_pki=6.371	l3ref_pki=12.290	l1dref_pki=16.980	memacc_pki=213.10	dtlbw_pki=0.6856	pmu_enabled=100.0	pmu_cpu_s=3403.0	who=at_s=8,rss_pg=6044318,settled=2	pfn_zero_frac=0.0000	maps=3	sampled=73728	present_frac=0.8241	l2color_cv=0.0063	l3color_cv=0.0125	contig_frac=0.9197	mean_run=12.2	vapa16=0.0056	gib_regions=32	anon_pg=12932	anon_l2cv=0.0045	anon_contig=0.9444	anon_run=17.4	anon_gib=32	file_pg=24576	file_l2cv=0.0025	file_contig=0.9920	file_run=101.1	file_gib=29	other_pg=23252	other_l2cv=0.0160	other_contig=0.8294	other_run=5.8	other_gib=32	ro_cost_ms=72-->
<!--DATA 3	asym0_pin	pp2048	19.48-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [asym0_unpin]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | asym0_unpin | 18.60 | 3 | 18.37 | 18.90 | -- | -- |
| pp2048 | asym0_pin | 19.34 | 3 | 19.12 | 19.48 | 1.040x | 1.048 1.012 1.060 |
| pp2048 | stock_unpin | 19.97 | 3 | 19.84 | 20.06 | 1.074x | 1.071 1.059 1.092 |
| pp2048 | stock_pin | 21.19 | 3 | 21.07 | 21.31 | 1.140x | 1.144 1.115 1.160 |

