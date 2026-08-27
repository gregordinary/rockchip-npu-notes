<!-- Raw llama-bench output backing perf/benchmarks.md (gpt-oss-20b, MXFP4, MoE), and the
     first archived data for that model -- the block it backs was previously summarized
     inline with no raw run behind it.

     One board, one session, clock PINNED at 600 MHz throughout (power/control=on for all three
     NPU domains, restored to auto after). Nothing here is compared against a number from a
     previous session: the CPU and NPU-default baselines were re-measured alongside the thing
     under test. RK1 (RK3588), 31 GiB, kernel 7.1.1, llama.cpp a646006f0 (9932), 8 CPU threads.

     -b 2048 -ub 2048 throughout -- for the moe_fp16 route because its per-expert dequant is
     exactly what a smaller micro-batch multiplies, and for moe_native because (a) the DENSE
     MXFP4 weights are not in the expert cache and still re-dequantize per micro-batch and
     (b) each expert receives only n_tokens*n_used/n_expert rows (64 at -ub 512 vs 256 at
     -ub 2048) while the per-expert dispatch/gather/scatter/padding around the GEMM stays flat.
     The native route was NOT measured at -ub 512; do not quote the fp16 route's -ub 512
     collapse as if it applied to it.

     It is also why the NPU-default baseline here reads LOWER than the 13.11 recorded in earlier
     notes: that figure was measured at the default -ub 512. The like-for-like -ub 2048 number
     has always been ~11 (Jul 3 raw: 11.31; this run: 10.99). Comparing a -ub 512 baseline
     against a -ub 2048 result is the trap this file exists to close.

     Configs:
       cpu             no backend loaded
       npu_default     GGML_BACKEND_PATH set, ROCKET_MOE unset -- dense graph on the NPU,
                       routed experts on the CPU
       moe_fp16        ROCKET_MOE=1 ROCKET_MOE_NATIVE=0 -- experts on the NPU via the fp16
                       route (weight dequantized to fp16 on the host EVERY micro-batch)
       moe_native      ROCKET_MOE=1 -- experts ingested ONCE to resident int8 codes on the NPU
     [HW sweep, 600 MHz, 2026-07-14].
-->

# gpt-oss-20b (MXFP4, MoE) — raw

## The bench matrix

```
### cpu  2026-07-14T14:39:35Z  clk=600 MHz
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CPU        |       8 |     2048 |           pp512 |         13.09 ± 0.02 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CPU        |       8 |     2048 |          pp2048 |         12.40 ± 0.08 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CPU        |       8 |     2048 |           tg128 |          7.13 ± 0.06 |
build: a646006f0 (9932)
### cpu wall 654s
### npu_default  2026-07-14T14:50:29Z  clk=600 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         14.09 ± 0.06 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         10.99 ± 0.04 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           tg128 |          7.16 ± 0.04 |
build: a646006f0 (9932)
### npu_default wall 709s
### moe_fp16  2026-07-14T15:02:18Z  clk=600 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |          4.59 ± 0.02 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         10.18 ± 0.08 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           tg128 |          7.14 ± 0.02 |
build: a646006f0 (9932)
### moe_fp16 wall 984s
### moe_native  2026-07-14T15:18:42Z  clk=600 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         12.19 ± 0.22 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |          6.11 ± 0.10 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           tg128 |          6.30 ± 0.83 |
build: a646006f0 (9932)
### moe_native wall 1262s
### moe_native_nt  2026-07-14T15:39:44Z  clk=600 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         17.29 ± 0.76 |
### moe_native_nt wall 235s
```

## The three fixes, and the final numbers

Between the matrix above and the run below, three defects were found and fixed. Every one
of them was invisible until the per-phase instrumentation was added, and none of them made
anything FAIL -- they just quietly cost throughput, or quietly computed the wrong answer.

  1. Resident-weight TILE PADDING (driver). The N-tile defaulted to MAX_TILE=256 while each
     worker plans on a 576-wide slice, so 3 tiles stored 768 columns to hold 576. Fixed by
     taking the smallest tile that still reaches the same tile COUNT (192): same dispatch,
     same DMA. 10.70 -> 8.07 MiB per resident expert; residency 82% -> 99%.

  2. M-BUCKET RATCHET (ggml-rocket). The adaptive granule doubled whenever the distinct-slot
     set neared the driver table -- but that set never shrinks, so the test could never
     re-pass and the granule slammed to its 4096 ceiling on the first overflow. 88.3% of
     every expert GEMM was padding. Fixed with a fixed 2-per-octave ladder: 20.5% padded.

  3. ATTENTION SINKS (ggml-rocket). supports_op never checked src[4], so gpt-oss (which
     carries a learned per-head sink logit on every layer) took the FLASH_ATTN offload and
     got a softmax with no sink term -- a silently WRONG attention, past the n_kv floor of
     1024. Fixed by declining. Declining is also +26% at pp2048, which is why the bug
     presented as a performance regression.

```
== final  2026-07-14T15:59:36Z  clk=600 MHz ==
### npu_default_fixed
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         14.11 ± 0.13 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp1024 |         14.29 ± 0.02 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         13.89 ± 0.03 |
### moe_native_fixed
[moe-int8] resident budget reached at 1747 experts (14092MB on the NPU, 21433MB charged incl. the GGUF source) -- the remaining experts stream via dequant->fp16 (raise ROCKET_MOE_CACHE_MB, or ROCKET_N_THREADS for more per-fd IOVA)
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         17.57 ± 0.64 |
[moe-int8] experts exercised: 1747 resident on the NPU (14092MB), 14 streamed via dequant->fp16 -- 99% of the per-micro-batch dequant removed
[moe-int8] one-time ingest: 72.9s total (20.6s GGUF->int8 decode, 52.3s NPU-BO pack) for 1747 experts = 42ms each
[moe-int8] resident budget reached at 1732 experts (13971MB on the NPU, 21249MB charged incl. the GGUF source) -- the remaining experts stream via dequant->fp16 (raise ROCKET_MOE_CACHE_MB, or ROCKET_N_THREADS for more per-fd IOVA)
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp1024 |         24.38 ± 1.03 |
[moe-int8] experts exercised: 1732 resident on the NPU (13971MB), 23 streamed via dequant->fp16 -- 99% of the per-micro-batch dequant removed
[moe-int8] one-time ingest: 71.3s total (20.6s GGUF->int8 decode, 50.8s NPU-BO pack) for 1732 experts = 41ms each
[moe-int8] resident budget reached at 1724 experts (13906MB on the NPU, 21151MB charged incl. the GGUF source) -- the remaining experts stream via dequant->fp16 (raise ROCKET_MOE_CACHE_MB, or ROCKET_N_THREADS for more per-fd IOVA)
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         26.78 ± 0.05 |
[moe-int8] experts exercised: 1724 resident on the NPU (13906MB), 94 streamed via dequant->fp16 -- 95% of the per-micro-batch dequant removed
[moe-int8] one-time ingest: 69.9s total (20.8s GGUF->int8 decode, 49.1s NPU-BO pack) for 1724 experts = 41ms each
ROCKET MoE native total(ms): gather=10912 act_quant=10384 gemm=233791 scatter=7935 | fp16_streamed=9600  (621 ops; 14718/14889 expert GEMMs native = 99%; 20.5% padded rows; gemm=89% of the native route)
== done  2026-07-14T16:23:23Z ==
```

## The one-time ingest

```
[rocket] quantized prefill is dequant-bound at this micro-batch; run with -b 2048 -ub 2048 for ~2x (the default -ub 512 ~halves it)
[rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 21483MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)
[moe-int8] ingesting experts to int8: 256 done, 2740MB resident, 11s elapsed
[moe-int8] ingesting experts to int8: 512 done, 5480MB resident, 22s elapsed
[moe-int8] ingesting experts to int8: 768 done, 8220MB resident, 33s elapsed
[moe-int8] ingesting experts to int8: 1024 done, 10960MB resident, 44s elapsed
[moe-int8] ingesting experts to int8: 1280 done, 13700MB resident, 55s elapsed
[moe-int8] resident budget reached at 1441 experts (15423MB on the NPU, 21478MB charged incl. the GGUF source) -- the remaining experts stream via dequant->fp16 (raise ROCKET_MOE_CACHE_MB, or ROCKET_N_THREADS for more per-fd IOVA)
[moe-int8] experts exercised: 1441 resident on the NPU (15423MB), 260 streamed via dequant->fp16 -- 85% of the per-micro-batch dequant removed
[moe-int8] one-time ingest: 62.4s total (16.9s GGUF->int8 decode, 45.5s NPU-BO pack) for 1441 experts = 43ms each
```

## Faithfulness

```
== faithfulness  2026-07-14T16:59:08Z  clk=600 MHz ==
[cpu] 1632 bytes
[npu_default] 1633 bytes
[moe_native] 1631 bytes
[npu_fa_on] 1633 bytes
== greedy diff vs the CPU reference ==
  npu_default: DIVERGES from CPU
      31c31
      < The user pasted a long text. They didn't ask a question. They might want a summary, or a rewrite, or analysis. The prompt: "The design of a neural processing unit..." repeated. They might want a summary
      ---
      > The user pasted a long text. They didn't ask a question. They might want a summary, or a rewrite, or analysis. The prompt: "The design of a neural processing unit is dominated by the movement of data..."
  moe_native: DIVERGES from CPU
      31c31
      < The user pasted a long text. They didn't ask a question. They might want a summary, or a rewrite, or analysis. The prompt: "The design of a neural processing unit..." repeated. They might want a summary
      ---
      > The user pasted a long text. They didn't ask a question. They might want a summary, or a critique, or a rewrite. The prompt: "The design of a neural processing unit is dominated by the movement of data
  npu_fa_on: DIVERGES from CPU
      31c31
      < The user pasted a long text. They didn't ask a question. They might want a summary, or a rewrite, or analysis. The prompt: "The design of a neural processing unit..." repeated. They might want a summary
      ---
      > The user pasted a long text. They didn't ask a question. They might want a summary, or a rewrite, or analysis. The prompt: "The design of a neural processing unit is dominated by the movement of data..."
== per-matmul cosine vs the fp64 CPU reference (real weights, real activations) ==
ROCKET MoE native-quant cosine vs CPU fp64 reference (real weights, real activations): mean=0.999822 min=0.999395 over 24 expert GEMMs
ROCKET MoE native-quant cosine vs CPU fp64 reference (real weights, real activations): mean=0.999821 min=0.998980 over 48 expert GEMMs
[moe-int8] experts exercised: 1740 resident on the NPU (14035MB), 246 streamed via dequant->fp16 -- 88% of the per-micro-batch dequant removed
ROCKET MoE native-quant cosine vs CPU fp64 reference (real weights, real activations): mean=0.999821 min=0.998980 over 50 expert GEMMs
```

---

# The default-on flip — raw (2026-08-27)

<!-- One board, one session, clock PINNED at 600 MHz (power/control=on for all three NPU
     domains), CPU governor `performance` on all three clusters. RK1 (RK3588), 31 GiB,
     kernel 7.2.0-1, llama.cpp 171974745 (b10558), rocket 1.3.0, rocket-userspace b6e364a.

     THREE BINARIES, and the boundary matters -- do not read rows across it:
       [A] the pre-flight as first written. MoE budget from the SHARED auto reserve
           (max(6 GiB, 30% RAM)) -> 21.2 GB, 54 of 72 stacks.
       [B] + the GGUF double-count removed from the MoE budget (6 GiB floor only)
           -> 24.6 GB, 63 of 72 stacks.
       [C] + the per-expert row floor (M_e >= ROCKET_MOE_M_BUCKET), which is what
           DeepSeek needed. gpt-oss is unaffected by [C] (M_e = 64 and 256, both clear it).
     [A] is kept because it is the run that isolates the PLACEMENT question at a matched
     budget -- default and forced against the same 21.2 GB -- which [B] and [C] no longer do.

     Arms: cpu (no backend), ROCKET_MOE=0 (dense graph on the NPU, experts on the CPU),
     default (ROCKET_MOE unset -> AUTO), ROCKET_MOE=1 (FORCED, claims everything).
     ROCKET_KACC=1 throughout. -r 2, and llama-bench's warmup is a full prompt run, so the
     one-time expert ingest lands there and the reported t/s is clean.
-->

```
===== PASS 1 =====
### cpu-p1  23:38:14  clk=600 MHz
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CPU        |       8 |     2048 |           pp512 |         11.97 ± 0.01 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CPU        |       8 |     2048 |          pp1024 |         12.05 ± 0.12 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CPU        |       8 |     2048 |          pp2048 |         11.76 ± 0.00 |

build: 171974745 (10558)

### npu-moe-off-p1  23:53:23  clk=600 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         14.07 ± 0.05 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp1024 |         14.22 ± 0.02 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         13.72 ± 0.10 |

build: 171974745 (10558)

### npu-default-p1  00:06:20  clk=600 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         20.72 ± 0.11 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp1024 |         24.39 ± 0.46 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         24.87 ± 0.12 |

build: 171974745 (10558)
    [moe-int8] resident budget reached after 54 expert stacks (21025MB RAM, 13668MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [moe-int8] experts exercised: 1473 resident on the NPU (11881MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] one-time ingest: 30.8s total (8.5s GGUF->int8 decode, 22.3s NPU-BO pack) for 1473 experts = 21ms each

### npu-moe-forced-p1  00:15:14  clk=600 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         24.63 ± 0.00 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp1024 |         32.26 ± 0.63 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         32.67 ± 0.08 |

build: 171974745 (10558)
    [moe-int8] resident budget reached at 1729 experts (13946MB on the NPU, 21212MB charged incl. the GGUF source) -- the remaining experts stream via dequant->fp16 (raise ROCKET_MOE_CACHE_MB, or ROCKET_N_THREADS for more per-fd IOVA)
    [moe-int8] experts exercised: 1729 resident on the NPU (13946MB), 170 streamed via dequant->fp16 -- 91% of the per-micro-batch dequant removed
    [moe-int8] one-time ingest: 35.2s total (10.1s GGUF->int8 decode, 25.1s NPU-BO pack) for 1729 experts = 20ms each

===== PASS 2 =====
### cpu-p2  00:22:47  clk=600 MHz
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CPU        |       8 |     2048 |           pp512 |         12.29 ± 0.02 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CPU        |       8 |     2048 |          pp1024 |         12.15 ± 0.02 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CPU        |       8 |     2048 |          pp2048 |         11.79 ± 0.06 |

build: 171974745 (10558)

### npu-moe-off-p2  00:37:48  clk=600 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         14.13 ± 0.05 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp1024 |         14.17 ± 0.04 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         13.68 ± 0.10 |

build: 171974745 (10558)

### npu-default-p2  00:50:47  clk=600 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         20.85 ± 0.28 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp1024 |         24.38 ± 0.40 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         24.79 ± 0.31 |

build: 171974745 (10558)
    [moe-int8] resident budget reached after 54 expert stacks (21025MB RAM, 13668MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [moe-int8] experts exercised: 1473 resident on the NPU (11881MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] one-time ingest: 30.9s total (8.6s GGUF->int8 decode, 22.3s NPU-BO pack) for 1473 experts = 21ms each

### npu-moe-forced-p2  00:59:41  clk=600 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         24.39 ± 0.03 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp1024 |         31.89 ± 0.53 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         33.03 ± 0.14 |

build: 171974745 (10558)
    [moe-int8] resident budget reached at 1731 experts (13962MB on the NPU, 21237MB charged incl. the GGUF source) -- the remaining experts stream via dequant->fp16 (raise ROCKET_MOE_CACHE_MB, or ROCKET_N_THREADS for more per-fd IOVA)
    [moe-int8] experts exercised: 1731 resident on the NPU (13962MB), 168 streamed via dequant->fp16 -- 91% of the per-micro-batch dequant removed
    [moe-int8] one-time ingest: 35.2s total (10.1s GGUF->int8 decode, 25.2s NPU-BO pack) for 1731 experts = 20ms each

DONE
===== (1) budget raised to what the forced arm committed =====
### big-default-26g  01:09:09
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         23.36 ± 0.28 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         30.35 ± 0.11 |

build: 171974745 (10558)
    [moe-int8] experts exercised: 1701 resident on the NPU (13720MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 26000MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [moe-int8] resident budget reached after 66 expert stacks (25697MB RAM, 16706MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [moe-int8] experts exercised: 1731 resident on the NPU (13962MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed

===== (2) a board too small to hold the stack =====
### small-moeoff  01:14:45
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         14.08 ± 0.06 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         13.62 ± 0.01 |

build: 171974745 (10558)

### small-default  01:24:08
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         17.02 ± 0.22 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         18.24 ± 0.12 |

build: 171974745 (10558)
    [moe-int8] experts exercised: 876 resident on the NPU (7066MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 12000MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [moe-int8] resident budget reached after 30 expert stacks (11680MB RAM, 7593MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [moe-int8] experts exercised: 891 resident on the NPU (7187MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed

### small-forced  01:31:53
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         13.63 ± 0.23 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         24.92 ± 0.32 |

build: 171974745 (10558)
    [moe-int8] only 53% resident -- below ~95% this route is typically a net LOSS at short prefill (a streamed expert's dequant does not shrink with the row count). Under the default the pre-flight reserves a stack before claiming its op, so reaching here means a limit it cannot see ahead -- an exhausted NPU IOVA window (raise ROCKET_N_THREADS), or ROCKET_MOE=1, which claims the op without reserving anything. Raise ROCKET_MOE_CACHE_MB if the RAM is there, or set ROCKET_MOE=0 to leave the experts on the CPU.
    [moe-int8] resident budget reached at 978 experts (7888MB on the NPU, 11998MB charged incl. the GGUF source) -- the remaining experts stream via dequant->fp16 (raise ROCKET_MOE_CACHE_MB, or ROCKET_N_THREADS for more per-fd IOVA)
    [moe-int8] experts exercised: 978 resident on the NPU (7888MB), 894 streamed via dequant->fp16 -- 52% of the per-micro-batch dequant removed
    [moe-int8] only 52% resident -- below ~95% this route is typically a net LOSS at short prefill (a streamed expert's dequant does not shrink with the row count). Under the default the pre-flight reserves a stack before claiming its op, so reaching here means a limit it cannot see ahead -- an exhausted NPU IOVA window (raise ROCKET_N_THREADS), or ROCKET_MOE=1, which claims the op without reserving anything. Raise ROCKET_MOE_CACHE_MB if the RAM is there, or set ROCKET_MOE=0 to leave the experts on the CPU.

DONE
===== gpt-oss-20b, -b 2048 -ub 2048 =====
### g-cpu  01:44:01
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CPU        |       8 |     2048 |           pp512 |         12.17 ± 0.00 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CPU        |       8 |     2048 |          pp2048 |         11.80 ± 0.04 |

build: 171974745 (10558)

### g-moeoff  01:54:50
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         14.08 ± 0.04 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         13.63 ± 0.03 |

build: 171974745 (10558)

### g-default  02:04:14
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         22.82 ± 0.32 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.66 ± 0.10 |

build: 171974745 (10558)
    [moe-int8] residency pre-flight: 24638MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [moe-int8] experts exercised: 1656 resident on the NPU (13357MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed

### g-forced  02:10:02
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         24.81 ± 0.61 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         34.01 ± 0.07 |

build: 171974745 (10558)
    [moe-int8] experts exercised: 1845 resident on the NPU (14882MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] experts exercised: 1869 resident on the NPU (15076MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed

===== gpt-oss-20b, llama.cpp DEFAULT micro-batch (-ub 512) =====
### g512-moeoff  02:15:18
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |           pp512 |         14.13 ± 0.06 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |          pp2048 |         13.53 ± 0.01 |

build: 171974745 (10558)

### g512-default  02:24:45
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |           pp512 |         22.69 ± 0.56 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |          pp2048 |         22.79 ± 0.06 |

build: 171974745 (10558)
    [moe-int8] residency pre-flight: 24645MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [moe-int8] experts exercised: 1656 resident on the NPU (13357MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed

===== DeepSeek-V2-Lite Q4_K_M (MLA + MoE), -ub 2048 =====
### d-cpu  02:31:30
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |       8 |     2048 |           pp512 |         19.54 ± 0.03 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |       8 |     2048 |          pp2048 |         18.50 ± 0.02 |

build: 171974745 (10558)

### d-moeoff  02:38:33
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |           pp512 |         26.04 ± 0.11 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |          pp2048 |         21.63 ± 0.02 |

build: 171974745 (10558)

### d-default  02:44:20
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |           pp512 |         19.09 ± 1.04 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |          pp2048 |         27.68 ± 0.50 |

build: 171974745 (10558)
    [moe-int8] experts exercised: 4719 resident on the NPU (14771MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24688MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [moe-int8] experts exercised: 4767 resident on the NPU (14921MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed

DONE
```

## The row floor, the drift discriminator, and the shipped binary (2026-08-27)

<!-- Continues the block above. Binary [C] = [B] + the per-expert row floor
     (M_e >= ROCKET_MOE_M_BUCKET); [D] = [C] + the cached bucket-knob read, which is the
     SHIPPED source and is semantically inert.

     READ THE moeoff CONTROL COLUMN FIRST. `d-default-fixed` / `g-default-fixed` is a single
     rep each, and g-default-fixed came in at 20.52 / 26.88 against 22.82 / 28.66 on [B] --
     a 10% "regression" from a change that cannot move gpt-oss at all (it clears the new floor
     at both ends, and the log shows identical placement: 63 stacks, 1656 experts, 0 streamed).
     `moe-disc` is the arm that settled it: ROCKET_MOE=0, which the change cannot touch,
     interleaved with the default twice so neither owns a position in the sequence. It read
     13.63 / 13.63 at pp2048 on [C] against 13.63 on [B] and 13.61 on [D] -- so the board did
     not drift and the build is congruent, and the 20.52 was a low sample of an arm whose own
     pp512 spread is ~10%. Repeated on [C] and [D] it reads 22.35 / 21.91 / 21.97 / 22.29.

     THE LESSON, because it nearly went into the record the other way: a single run of the
     OFFLOADED arm is not a measurement of it, and the control is what says so. Six runs of
     the default span 20.5-22.8 at pp512. Published figures are the mean of those six.
     See [[rebuild-rerolls-cache-congruence]], [[small-red-sample-proves-no-more-than-green]],
     [[isolation-run-is-the-measurement]].

     NPU die temperature is in each header (npu=<millidegC>); the board sat at 44-61 C
     throughout, well under the ~85 C throttle, and the CPU governor stayed `performance`
     with scmi_clk_npu pinned at 600 MHz for every run in this file.
-->

```
### moeoff-r1  03:16:12  npu=45307
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         14.12 ± 0.07 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         13.63 ± 0.07 |

build: 171974745 (10558)

### default-r1  03:25:36  npu=61000
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         22.35 ± 0.47 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.13 ± 0.26 |

build: 171974745 (10558)
    [moe-int8] experts exercised: 1656 resident on the NPU (13357MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed

### moeoff-r2  03:31:29  npu=55461
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         14.05 ± 0.07 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         13.63 ± 0.10 |

build: 171974745 (10558)

### default-r2  03:40:53  npu=61000
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         21.91 ± 0.33 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.31 ± 0.04 |

build: 171974745 (10558)
    [moe-int8] experts exercised: 1656 resident on the NPU (13357MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed

DONE
### d-default-fixed  02:55:04
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |           pp512 |         25.88 ± 0.24 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.13 ± 0.17 |

build: 171974745 (10558)
    [moe-int8] experts exercised: 4767 resident on the NPU (14921MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed

### g-default-fixed  03:01:53
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         20.52 ± 0.77 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         26.88 ± 0.20 |

build: 171974745 (10558)
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [moe-int8] experts exercised: 1656 resident on the NPU (13357MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed

### g512-default-fixed  03:08:17
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |           pp512 |         22.08 ± 0.35 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |          pp2048 |         22.36 ± 0.12 |

build: 171974745 (10558)
    [moe-int8] resident budget reached after 63 expert stacks (24529MB RAM, 15946MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [moe-int8] experts exercised: 1656 resident on the NPU (13357MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed

DONE
### ship-moeoff  03:49:53  npu=44384
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         14.10 ± 0.04 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         13.61 ± 0.08 |

build: 171974745 (10558)

### ship-default1  03:59:17  npu=61000
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         21.97 ± 0.36 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.34 ± 0.04 |

build: 171974745 (10558)
    [moe-int8] experts exercised: 1656 resident on the NPU (13357MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed

### ship-default2  04:05:10  npu=55461
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |           pp512 |         22.29 ± 0.25 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.30 ± 0.02 |

build: 171974745 (10558)
    [moe-int8] experts exercised: 1656 resident on the NPU (13357MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed

### ship-ub512  04:11:01  npu=54538
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |           pp512 |         21.74 ± 0.69 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |          pp2048 |         22.40 ± 0.08 |

build: 171974745 (10558)
    [moe-int8] experts exercised: 1656 resident on the NPU (13357MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed

### ship-dseek  04:17:53  npu=53615
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |           pp512 |         25.85 ± 0.12 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |          pp2048 |         27.62 ± 0.67 |

build: 171974745 (10558)
    [moe-int8] experts exercised: 4767 resident on the NPU (14921MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed

DONE
```

## The accept-boundary map (2026-08-27)

gpt-oss's five cells of that map are filed with DeepSeek-V2-Lite's, in
[deepseek-v2-lite.md](deepseek-v2-lite.md), because it is one experiment and the defect it found
lives on that model. What the gpt-oss half says on its own: the offload wins at **every** prefill
length measured, rising monotonically with the per-expert row count — 1.64x at `M_e`=64 to 2.08x at
`M_e`=256 — and its `ROCKET_MOE=0` control read **13.64** at pp2048, against 13.61-13.63 on every
earlier binary and repetition. Its placement is unchanged by the per-dispatch work floor added
after that map: the tile granule binds first on this architecture at every reachable shape.

## The ROCKET_MOE_CACHE_MB ladder (2026-08-27)

<!-- What the residency pre-flight costs on a board that nearly fits, and how much of it comes back
     from the budget knob alone. Run behind a `drop_caches`, because the budget is MemAvailable
     minus 6 GiB read ONCE at the first supports_op and it had drifted 24.6 -> 20.7 GB over the
     session -- an experiment that VARIES that budget cannot have its baseline sliding under it.
     MemAvailable 31.7 GB after the drop, 30.8 GB by the first supports_op. RK1 @ 600 MHz pinned,
     governor performance, -b 2048 -ub 2048, -r 3, one discarded warm-up process, same binary
     throughout. EVERY arm reported 0 streamed. -->

```
arm                       budget    stacks    pp512    pp2048     vs default
ROCKET_MOE=0                   -         -    13.95     13.54              -
default (AUTO)          24672 MB        63    21.96     26.51              -
CACHE_MB=26000          26000 MB        66    22.90     27.85   +4.3% / +5.1%
CACHE_MB=28000          28000 MB        71    23.79     30.25   +8.3% / +14.1%
ROCKET_MOE=1 (ceiling)      none    72 all    23.34     31.27   +6.3% / +18.0%

within-process spread:  default +/-1.36 / +/-0.17,  cache28000 +/-1.86 / +/-0.39,
                        forced +/-2.06 / +/-0.38,   ROCKET_MOE=0 +/-0.02 / +/-0.05
```

**`CACHE_MB=28000` reaches 71 of 72 stacks and recovers 79% of the pp2048 ceiling while exceeding
the forced arm outright at pp512 — with the pre-flight's sign guarantee intact**, which
`ROCKET_MOE=1` gives up. So the pre-flight's cost is mostly recoverable by documentation rather
than by code, and the number is 28000 rather than the 26000 an earlier note pointed at.

Two caveats that belong with it. A 28000 MB budget leaves only ~2.8 GB of the headroom the 6 GiB
auto reserve exists for (KV cache, activations, slack) -- it ran clean here at pp512-pp2048 on a
31 GiB board with nothing streamed, but it is a knob for a known working set, not a new default.
And **the pp512 column is unreliable**: `cache28000` reading above `forced` there sits inside a
+/-1.4 to +/-2.1 spread and is not a real inversion. The `ROCKET_MOE=0` control's own spread is
+/-0.02 to +/-0.05, two orders of magnitude tighter, which is why it is the arm that detects drift.

**The control moved slightly this session and it is worth recording rather than smoothing.** It
read 13.54 at pp2048 here against 13.61-13.64 on every earlier binary and repetition -- about
-0.7%, taken immediately after a `drop_caches` had emptied the page cache. Small, but it widens the
range that has been quoted as the congruence check to **13.54-13.64**.

## Per-matmul cosine on the SHIPPED placement (2026-08-27)

<!-- The greedy-match leg raised a question it could not answer: every arm carrying the expert
     offload diverged from the CPU reference at word 10 where the experts-on-CPU arm diverged at
     word 16. ROCKET_MOE_COSINE=1 answers it directly -- one expert per MUL_MAT_ID op, rotating
     across every layer, projection and expert, recomputed on the CPU in fp64 from the undecoded
     GGUF blocks, on REAL weights and REAL activations. Run under the SHIPPED default (63 of 72
     stacks), not the archived harness's ROCKET_MOE=1. -->

```
ROCKET MoE native-quant cosine vs CPU fp64 reference (real weights, real activations):
  mean=0.999822  min=0.999402  over 24 expert GEMMs
  mean=0.999838  min=0.999402  over 48 expert GEMMs
  mean=0.999815  min=0.998976  over 55 expert GEMMs      <- final
```

**0.999815 / 0.998976 on the shipped placement against 0.999821 / 0.998980 on the archived
`ROCKET_MOE=1` one.** The two placements are numerically the same: narrowing the gate to 63 of 72
stacks changed WHAT is placed, not HOW it computes.

That also settles the greedy question. A 0.9998 cosine is excellent and is still enough to flip an
argmax wherever the top two logits are close, at a position nothing controls -- so an arm carrying
the offload diverging six words earlier than one without it is the expected consequence of a small
bounded per-op error, not evidence of a defect. The greedy comparison has no resolution here; the
cosine and the differential PPL are what carry this route's faithfulness.
