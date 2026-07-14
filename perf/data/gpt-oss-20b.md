<!-- Raw llama-bench output backing perf/benchmarks.md (gpt-oss-20b, MXFP4, MoE), and the
     first archived data for that model -- the block it backs was previously summarized
     inline with no raw run behind it.

     One board, one session, clock PINNED at 600 MHz throughout (power/control=on for all three
     NPU domains, restored to auto after). Nothing here is compared against a number from a
     previous session: the CPU and NPU-default baselines were re-measured alongside the thing
     under test. RK1 (RK3588), 31 GiB, kernel 7.1.1, llama.cpp a646006f0 (9932), 8 CPU threads.

     -b 2048 -ub 2048 throughout. That setting is MANDATORY for a quantized MoE (a quantized
     GGUF re-dequantizes per micro-batch, so the llama.cpp default -ub 512 quadruples the
     expert dequant tax), and it is also why the NPU-default baseline here reads LOWER than the
     13.11 recorded in earlier notes: that figure was measured at the default -ub 512. The
     like-for-like -ub 2048 number has always been ~11 (Jul 3 raw: 11.31; this run: 10.99).
     Comparing a -ub 512 baseline against a -ub 2048 result is the trap this file exists to
     close.

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
