<!-- Raw llama-bench + llama-perplexity output backing perf/benchmarks.md (DeepSeek-V2-Lite). The
     clk=200 in each per-run header is an idle sample taken between the discarded warmup and the
     measured run; the NPU rides to 600 MHz under load (module loaded with
     rocket_npu_clk_hz=600000000). Warm medians, llama-bench -r 2 plus a discarded warmup. Q4_K_M
     ONLY: at 15.71B the F16 GGUF (~31 GB) does not fit the 31 GB board; Q4_K_M (9.65 GiB) fits with
     headroom. Quant and PPL both at -b 2048 -ub 2048. GGUF from mradermacher/DeepSeek-V2-Lite-GGUF
     (the base model deepseek-ai/DeepSeek-V2-Lite; converted 2024-05, MLA tensors in the pre-split
     attn_kv_a_mqa/attn_kv_b form, which build 9568 reconstructs -- the FLASH_ATTN op shapes are
     identical either way). arch deepseek2: MLA attention (kv_lora_rank 512, key_length 192 = 128
     nope + 64 rope, value_length 128, 16 heads) + MoE (64 routed + 2 shared experts, 6 routed
     active per token, 27 blocks, block 0 dense). ~2.4B of 15.71B params active per token.

     A COMBINED gap-finder -- it stacks the two attention/expert paths that run on the CPU here:
       1. MLA attention: the FA gate accepts DK != DV (DeepSeek's DK=192 = 128 nope + 64 rope,
          DV=128), so the FLASH_ATTN_EXT primitive is bit-faithful for MLA. The DeepSeek DL-backend
          FA path is not yet exercised on-device, so in this bench attention ran on the CPU -- the
          FA engagement diagnostic below shows no "ROCKET FA total" line under -fa auto or -fa 1.
       2. MoE routed FFN: GGML_OP_MUL_MAT_ID has an opt-in handler (ROCKET_MOE=1), but offloading
          quantized experts is dequant-bound and a net loss, so the 6 active routed experts'
          gate/up/down matmuls stay on the (faster) CPU by default.
     What DOES reach the NPU: the large MLA projections (q_a/q_b/kv_a/kv_b), the 2 always-on SHARED
     experts' gate/up/down, and lm_head -- all ordinary static-weight MUL_MAT. Those dense GEMMs are
     substantial (bigger than gpt-oss's GQA projections + no shared expert), so the NPU prefill win
     is modest-but-real (1.18-1.26x) and LARGER than gpt-oss's ~1.04x, even with attention and the
     routed experts on the CPU. Unlike the instruct/reasoning models in this record,
     the absolute wikitext PPL (~8.2) is in the normal range (base model); the NPU-CPU delta is the
     faithfulness measure. See ../benchmarks.md Method. Generator: run_sweep_deepseek.sh, 2026-07-03. -->

== FA engagement diagnostic (pp2048, r1; a "ROCKET FA total" line == FA offloaded to the NPU) ==
--- -fa auto ---
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |          pp2048 |         23.89 ± 0.00 |
--- -fa 1 (flash attention FORCED ON) ---
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |   1 |          pp2048 |         23.92 ± 0.00 |
NO "ROCKET FA total" line printed under either -fa auto or -fa 1 -> the FLASH_ATTN_EXT op is built
by llama.cpp (fa=1 column present, no error) but did not engage the NPU FA path in this backend
build -> attention ran on the CPU for this bench. (The FA gate itself accepts DK != DV and is
bit-faithful; the DeepSeek end-to-end offload is not yet wired/validated on-device.)

== DeepSeek-V2-Lite Q4_K_M  Fri Jul  3 19:40:28 UTC 2026 ==
### DeepSeek-V2-Lite Q4_K_M  [cpu]  19:41:22  clk=200 MHz
| model                          |       size |     params | backend    | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |       8 |     2048 |           pp512 |         20.37 ± 0.01 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |       8 |     2048 |          pp1024 |         19.92 ± 0.05 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |       8 |     2048 |          pp2048 |         19.01 ± 0.02 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |       8 |     2048 |            tg64 |          7.73 ± 0.02 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | CPU        |       8 |     2048 |    pp2048+tg128 |         15.25 ± 0.00 |

build: 7d2b45b4f (9568)

### DeepSeek-V2-Lite Q4_K_M  [npu]  19:58:15  clk=200 MHz
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |           pp512 |         24.04 ± 0.01 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |          pp1024 |         24.85 ± 0.00 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |          pp2048 |         23.87 ± 0.06 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |            tg64 |          7.77 ± 0.03 |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |    pp2048+tg128 |         18.12 ± 0.03 |

build: 7d2b45b4f (9568)

Prefill CPU->NPU: pp512 1.18x / pp1024 1.25x / pp2048 1.26x. The NPU prefill is flat ~24 t/s across
the curve; the CPU baseline declines with M (20.4->19.0), so the win rises modestly. Larger than
gpt-oss's ~1.04x (the MLA projections + 2 shared experts are substantial dense MUL_MAT that offload),
far below the dense models' 3x+ (attention and the routed experts -- the bulk of the graph -- stay on
the CPU). Decode NPU ~= CPU (7.73/7.77 t/s, off-NPU, MoE ~2.4B active). The combined pp2048+tg128
point (CPU 15.25, NPU 18.12) implies the 128 tokens decoded after a 2048-tok prompt stream at ~3.7 t/s
on both backends -- about half the tg64-from-empty rate, MLA decode cost growing with the filled latent
KV cache -- so a long-prompt turn's stream is slower than tg64.

== DeepSeek-V2-Lite faithfulness (wikitext test, -c 512, 12 chunks, -b 2048 -ub 2048; same GGUF CPU vs NPU) ==
Absolute PPL (~8.2) is in the normal range (base model, not an instruct/reasoning model whose absolute
PPL is inflated); the NPU-CPU delta is the faithfulness measure. Per-run stderr +/- 0.38.
### PPL Q4_K_M [cpu]  Final estimate: PPL = 8.2444 +/- 0.37733
### PPL Q4_K_M [npu]  Final estimate: PPL = 8.2232 +/- 0.37601   (delta -0.26%)

## The native-quant expert route, and the per-expert row floor (2026-08-27)

<!-- This is the model that showed a MoE placement failure mode residency cannot see.
     RK1 (RK3588), 31 GiB, kernel 7.2.0-1, llama.cpp 171974745 (b10558), rocket 1.3.0,
     clock PINNED at 600 MHz, CPU governor `performance`, -b 2048 -ub 2048, -r 2,
     ROCKET_KACC=1. Arms: cpu (no backend), ROCKET_MOE=0 (dense graph on the NPU, experts on
     the CPU), default (ROCKET_MOE unset).

     The pp512 default row BEFORE the row floor is 19.09 against 26.04 with the experts on
     the CPU -- a 27% regression -- and the teardown line for that same run reads
     "4767 resident on the NPU, 0 streamed, 100% of the per-micro-batch dequant removed".
     Residency was perfect and irrelevant. DeepSeek routes 6 of 64 experts, so 512 tokens give
     each one ~48 rows against a 64-row tile granule; gpt-oss routes 4 of 32 and gets 64.
     With the floor (M_e >= ROCKET_MOE_M_BUCKET) pp512 is declined and returns to the
     experts-on-CPU number, while pp2048 (M_e ~= 192) keeps its win.
-->

```
### d-cpu (no backend)
| deepseek2 16B Q4_K - Medium | 9.65 GiB | 15.71 B | CPU    | 8  | 2048 |  pp512 | 19.54 +/- 0.03 |
| deepseek2 16B Q4_K - Medium | 9.65 GiB | 15.71 B | CPU    | 8  | 2048 | pp2048 | 18.50 +/- 0.02 |

### d-moeoff (ROCKET_MOE=0)
| deepseek2 16B Q4_K - Medium | 9.65 GiB | 15.71 B | ROCKET | -1 | 2048 |  pp512 | 26.04 +/- 0.11 |
| deepseek2 16B Q4_K - Medium | 9.65 GiB | 15.71 B | ROCKET | -1 | 2048 | pp2048 | 21.63 +/- 0.02 |

### d-default, BEFORE the per-expert row floor  -- pp512 is the regression
| deepseek2 16B Q4_K - Medium | 9.65 GiB | 15.71 B | ROCKET | -1 | 2048 |  pp512 | 19.09 +/- 1.04 |
| deepseek2 16B Q4_K - Medium | 9.65 GiB | 15.71 B | ROCKET | -1 | 2048 | pp2048 | 27.68 +/- 0.50 |
    [moe-int8] experts exercised: 4767 resident on the NPU (14921MB), 0 streamed
               via dequant->fp16 -- 100% of the per-micro-batch dequant removed

### d-default-fixed, WITH the row floor (M_e=48 at pp512 -> declined; M_e=192 at pp2048 -> offloaded)
| deepseek2 16B Q4_K - Medium | 9.65 GiB | 15.71 B | ROCKET | -1 | 2048 |  pp512 | 25.88 +/- 0.24 |
| deepseek2 16B Q4_K - Medium | 9.65 GiB | 15.71 B | ROCKET | -1 | 2048 | pp2048 | 28.13 +/- 0.17 |

### ship-dseek, the shipped binary, second rep
| deepseek2 16B Q4_K - Medium | 9.65 GiB | 15.71 B | ROCKET | -1 | 2048 |  pp512 | 25.85 +/- 0.12 |
| deepseek2 16B Q4_K - Medium | 9.65 GiB | 15.71 B | ROCKET | -1 | 2048 | pp2048 | 27.62 +/- 0.67 |

build: 171974745 (10558)
```

## The accept-boundary map, and the row floor's replacement (2026-08-27)

<!-- ONE experiment across BOTH MoE models, filed here because the defect it found lives on this
     one. RK1 (RK3588), 31 GiB, kernel 7.2.0-1, rocket 1.3.0, llama.cpp 171974745 (b10558), NPU
     clock PINNED at 600 MHz, CPU governor `performance` on cpu0/4/6, -b 2048 -ub 2048, -r 3 with
     a discarded warm-up process per model, ROCKET_KACC=1. Arms INTERLEAVED per point so clock and
     thermal drift are charged to both equally. Every offloaded cell reported 100% resident and 0
     streamed, so nothing here is a residency effect.

     The question was where the per-expert row floor belongs; it had been placed at the tile
     granule (64) by mechanism and never measured as the edge. The answer is that a row floor
     cannot be placed at all -- M_e = 96 is 1.77x on gpt-oss and 0.94x on DeepSeek. Analysis and
     the replacement gate are in ../benchmarks.md, "The accept boundary". -->

```
                     M_e   work/dispatch   ROCKET_MOE=0        default            ratio
DeepSeek-V2-Lite Q4_K_M, 6-of-64, expert GEMM 2048x1408
  pp512               48        1.38e8    25.94 +/- 0.09   26.03 +/- 0.09   1.003  (declined)
  pp768               72        2.08e8    26.18 +/- 0.15   25.19 +/- 0.13   0.962
  pp1024              96        2.77e8    23.00 +/- 0.06   21.53 +/- 1.43   0.936
  pp1536             144        4.15e8    22.32 +/- 0.05   27.82 +/- 0.20   1.246
  pp2048             192        5.54e8    21.60 +/- 0.02   28.37 +/- 0.10   1.313
gpt-oss-20b MXFP4, 4-of-32, expert GEMM 2880x2880
  pp512               64        5.31e8    14.08 +/- 0.07   23.08 +/- 1.45   1.639
  pp768               96        7.96e8    14.22 +/- 0.07   25.13 +/- 1.27   1.767
  pp1024             128        1.06e9    14.10 +/- 0.10   26.92 +/- 0.73   1.909
  pp1536             192        1.59e9    13.88 +/- 0.06   28.38 +/- 0.21   2.045
  pp2048             256        2.12e9    13.64 +/- 0.02   28.30 +/- 0.30   2.075

build: 171974745 (10558)
```

Three things the raw rows carry that the ratio column does not:

- **The pp512 cell is the harness's own null control.** The shipped floor declines it, so both arms
  are the same placement, and they agree to 0.3%. A map whose null cell did not read 1.00 would be
  measuring the board, not the gate.
- **The control arm is congruent across sessions and builds.** gpt-oss `ROCKET_MOE=0` at pp2048 reads
  **13.64** here, against 13.61-13.63 on every earlier binary and repetition. That is what says a
  rebuild has not moved the floor under the comparison.
- **The offloaded arm's within-process spread is an order of magnitude wider than the control's**
  (+/- 1.45 against +/- 0.07 at gpt-oss pp512, +/- 1.43 against +/- 0.06 at DeepSeek pp1024). A
  single run of the offloaded arm is not a measurement of it; the two sub-parity DeepSeek cells were
  repeated three process-pairs each before anything was written down.

### The boundary cells, repeated

The map ran ONE adjacent pair per cell. That is adequate for gpt-oss, whose ratios sit an order of
magnitude outside the offloaded arm's spread, and NOT adequate for the three DeepSeek cells that sit
within a few percent of 1.00 — this arm varies ~15% run to run here. Three more process-pairs each,
alternating, on the same binary:

```
p=768   M_e= 72   208 MMAC/dispatch
  ROCKET_MOE=0   26.30 +/- 0.13   26.36 +/- 0.18   26.39 +/- 0.03      mean 26.35
  default        23.89 +/- 1.32   25.10 +/- 0.82   25.60 +/- 0.19      mean 24.86
  ratios                  0.908            0.952            0.970      mean 0.944   (map pair: 0.962)
p=1024  M_e= 96   277 MMAC/dispatch
  ROCKET_MOE=0   23.09 +/- 0.05   23.05 +/- 0.10   23.06 +/- 0.01      mean 23.07
  default        23.53 +/- 1.62   24.68 +/- 0.42   24.81 +/- 0.43      mean 24.34
  ratios                  1.019            1.071            1.076      mean 1.055   (map pair: 0.936)
p=1536  M_e=144   415 MMAC/dispatch
  ROCKET_MOE=0   22.34 +/- 0.06   22.31 +/- 0.00   22.33 +/- 0.09      mean 22.33
  default        26.59 +/- 0.67   27.13 +/- 1.13   27.89 +/- 0.25      mean 27.20
  ratios                  1.190            1.216            1.249      mean 1.218   (map pair: 1.246)

cross-check, ROCKET_MOE=1 at p=768:  25.55 +/- 0.16
```

### The pivotal cell, eight pairs

`M_e`=96 decides where the threshold goes and nothing else does, so it got four more pairs before
anything was rebuilt:

```
p=1024  M_e= 96   277 MMAC/dispatch, pairs 4-7
  ROCKET_MOE=0   23.03 +/- 0.08   23.07 +/- 0.04   23.04 +/- 0.07   23.06 +/- 0.08
  default        24.68 +/- 0.44   24.82 +/- 0.50   24.69 +/- 0.42   24.72 +/- 0.29
  ratios                  1.072            1.076            1.072            1.072
```

All eight pairs, in the order taken: **0.936 1.019 1.071 1.076 1.072 1.076 1.072 1.072**.
Pooled mean **1.049** (sd 0.049); over the seven repeat-harness pairs alone, **1.065** (sd 0.021).
The first pair is 6.3 sd below the other seven and it is the map's — but every one of the eight ran
the identical placement, and a different point in the board's allocation history predicts the fresh
run being FASTER, not slower. There is no mechanistic ground to drop it, so the pooled figure is the
one quoted.

An accept criterion was written down BEFORE these four pairs ran — accept a cell only if its mean
over >= 4 pairs reaches 1.05 — precisely because 1.049 against 1.065 is the kind of split that
invites picking after the fact. What makes the pooled side the right one rather than a coin toss is
the **ingest break-even**: the ~32 s expert ingest is charged only if the gate accepts, an offload at
ratio `r` saves `1 - 1/r` of prefill wall, and so a 1.049x cell does not repay its own admission
until **~16 600 tokens** of prefill at that micro-batch size (against ~4 800 at `M_e`=144's 1.22x and
~2 100 at gpt-oss's 1.64x). A marginal cell costs a short session more than it saves.

Two things this settles that the map could not:

- **The map's `M_e`=96 pair was the outlier, not the signal.** It read 0.936 where three repeats read
  1.019-1.076, and all four runs were **placement-identical** (4767 resident, 0 streamed, 14921 MB,
  the same ingest profile) — so nothing but variance separates them and the low one cannot be
  discarded on mechanism either. `M_e`=72 is the cell that is genuinely under 1.00: four pairs,
  every one of them, max 0.970.
- **The deficit at `M_e`=72 belongs to the ROUTE, not to the gate.** AUTO admits every DeepSeek stack
  (no "budget reached" line at any prefill length), so at `M_e`=72 the default and `ROCKET_MOE=1` are
  the same placement — and the forced arm reads **25.55**, inside the default's 23.89-25.60 range and
  likewise under the 26.35 control. Both arms lose there; the gate is not mis-accounting anything.

The offloaded arm's spread is the reason all of this needed repeating: at `M_e`=96 it ranges
**21.5-24.8 t/s** across runs while its control sits at **23.05-23.09**. Absolute t/s is not
comparable across runs here — only adjacent-pair ratios are, which is what both harnesses take.

Per-expert ingest, five samples a model on the same binary, which corrects a figure that had been
carried as a per-expert rate:

```
gpt-oss   ~1656 experts, 13357 MB resident:  35.7-37.0 s  (9.5-10.4 s MXFP4->int8 decode, 26.0-27.1 s NPU-BO pack)
DeepSeek  ~4746 experts, 14855 MB resident:  31.9-32.9 s  (5.0- 5.5 s Q4_K  ->int8 decode, 26.9-27.5 s NPU-BO pack)
```

The pack term is the **same ~27 s on both** across a 2.9x difference in expert count, because both
hold ~13-15 GB: it is bytes-bound at **~505-545 MB/s**, and "42 ms per expert" / "21 ms per expert"
are that one rate divided by two different expert sizes.

### After the fix, on the rebuilt binary

A cell the gate DECLINES is the same code path as `ROCKET_MOE=0`, so it must read ~1.00 and emit no
residency line at all -- nothing is ingested. Two process-pairs per cell:

```
                      before                  after (2 pairs)     experts ingested
p=768   M_e= 72   0.944 (4 pairs)          0.995   1.000          none -- declined
p=1024  M_e= 96   1.049 (8 pairs)          0.993   1.002          none -- declined
p=1536  M_e=144   1.218 (4 pairs)          1.163   1.237          4773 resident, 0 streamed
p=2048  M_e=192   1.313 (1 pair)           1.307                  4773 resident, 0 streamed

gpt-oss, the no-regression arm (its binding floor is the granule, never the work floor):
p=512   M_e= 64   1.639 (1 pair)           1.642   1.645          1656 resident, 0 streamed

Re-taken after `drop_caches` restored MemAvailable to 30.3 GB, so the pre-flight budget is 24742 MB
and no stack is declined -- the like-for-like check the first post-fix p=1536 cell was not:
p=1536  M_e=144   1.218 (4 pairs)          1.256 (28.07 / 22.34)  78 of 78 stacks
p=2048  M_e=192   1.313 (1 pair)           1.307 (28.20 / 21.57)  78 of 78 stacks

control arms across the rebuild:  26.35 -> 26.35/26.29   23.07 -> 23.12/23.03
                                  22.33 -> 22.31/22.37   21.60 -> 21.56/21.58
```

Every control is within **0.2%** of its pre-fix value, which is what says a rebuild that re-rolls
cache congruence did not move the floor under the comparison.

**THE ONE OUTLIER IS A PROPERTY OF THE PRE-FLIGHT, NOT NOISE.** The first p=1536 cell reads 1.163
because its process saw a RAM budget of **20716 MB** and admitted **70 of 78** expert stacks, where
every pre-fix cell saw ~24.6 GB and took all 78. The budget is `MemAvailable - 6 GiB`, read **once**,
at the first `supports_op` -- so hours of multi-GB allocation churn move it, and the same model on
the same board gets a different placement depending on what ran before it. The second iteration, at
a recovered budget, reads 1.237.

Two consequences worth carrying forward. For measurement: **any experiment that varies this budget
must drop the page cache first and record `MemAvailable`**, or its baseline slides under it. For the
product: a long-lived server that builds a fresh `llama_context` late in its life can silently place
fewer stacks on the NPU than the same process would have at startup -- the teardown line
(`ggml_backend_rocket_moe_stats`) is the only thing that reports it.

## Differential perplexity on the SHIPPED placement (2026-08-27)

<!-- Every archived greedy-match and differential-PPL row was taken under ROCKET_MOE=1, which is a
     different placement from what ships. This gates what users get. -c 2048 rather than the
     archived -c 512 for one reason: at -c 512 DeepSeek's M_e is 48 and the default DECLINES, so
     the gate would test only the declined path. -c 2048 puts M_e = 192 and the expert route is
     ACTIVE -- verified in the teardown line, not assumed. Absolute PPL is therefore NOT comparable
     to the archived 8.24; a longer context lowers it. The differential is the gate. -->

```
wikitext test, -c 2048 -b 2048 -ub 2048 --chunks 8, same GGUF, RK1 @ 600 MHz pinned
  cpu                    PPL = 5.3063 +/- 0.13142
  npu, ROCKET_MOE=0      PPL = 5.2906 +/- 0.13108
  npu, shipped default   PPL = 5.2842 +/- 0.13081
      [moe-int8] residency pre-flight: 24393MB RAM budget, 19200MB NPU IOVA across 5 worker fds
      [moe-int8] experts exercised: 4800 resident on the NPU (15025MB), 0 streamed -- 100% removed
```

**Take the PAIRED per-chunk difference, not the finals.** The absolute error bar is +/- 0.131, or
+/- 2.5%, and would resolve nothing; pairing on the same chunks cancels the chunk-to-chunk variance
that both arms share and gives an se around 0.10% in PPL terms.

```
                          per-chunk NLL delta                    PPL ratio
  ROCKET_MOE=0 vs cpu   -0.00296 +/- 0.00079  (3.74 se, 8/8 neg)   0.99704  (-0.296%)
  default      vs cpu   -0.00417 +/- 0.00104  (4.01 se)            0.99584  (-0.416%)
  default vs ROCKET_MOE=0  -0.00121 +/- 0.00101  (1.20 se, mixed)  0.99879  (-0.121%)
```

**The last row is the one that isolates the MoE route.** Both NPU arms carry the same dense fp16
offload, so differencing them cancels it and leaves the experts: **-0.12% +/- 0.10%,
indistinguishable from zero and in the favourable direction.** Comparing the default to the CPU
instead would credit the expert route with the dense path's own -0.30%.

The dense NPU path's -0.296% is itself real rather than noise -- 3.74 se, and every one of the eight
chunks negative -- and it is the known fp16-prefill difference, small and favourable.
