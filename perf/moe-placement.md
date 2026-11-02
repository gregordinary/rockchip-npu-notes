# MoE expert placement

`ggml-rocket` places the routed-expert GEMMs (`MUL_MAT_ID`) of a [MoE](../GLOSSARY.md) model on
the NPU by default, and holds each placed expert resident as int8 codes. Placement applies two
tests. The first is the work one expert dispatch performs, `M_e · K · N`, against a threshold of
340 mega-MACs (`ROCKET_MOE_MIN_WORK`). The second is the [pre-flight](../GLOSSARY.md), which
reserves a whole [expert stack](../GLOSSARY.md) against the RAM and IOVA budgets before its first
[ingest](../GLOSSARY.md). A stack the pre-flight cannot reserve stays on the CPU.

A threshold on the per-expert row count `M_e` cannot separate wins from losses across
architectures, but the work threshold does. Against the experts-on-CPU configuration, gpt-oss-20b
measures 1.64x at `M_e` = 64 and DeepSeek-V2-Lite 0.94x at `M_e` = 72 [HW sweep 2026-08-27]. On
Qwen3-30B-A3B, a third architecture, the threshold declines the experts at `-ub` 2048 and below.
Forcing them onto the NPU there measures 0.800x the experts-on-CPU configuration [HW sweep
2026-08-30]. On that expert-dominated model the automatic budget also places more stacks than the
board can hold, and a smaller `ROCKET_MOE_CACHE_MB` is faster [HW sweep 2026-08-28].

**Conditions:** RK3588 on a Turing RK1 (31 GiB), NPU clock ~816 MHz (the 600 MHz request) pinned,
CPU governor `performance`, the native-quant expert path, `-b 2048 -ub 2048` unless a section
states otherwise. Each section states its model and its date.

**Not characterized:** the accept-boundary map and the budget series at the 900 MHz request
(928 MHz).

## The accept boundary

The MoE placement threshold keeps a small expert GEMM on the CPU. It is a threshold on the work per
dispatch, `M_e · K · N`. A threshold on the per-expert row count, `M_e` =
`n_tokens · n_used / n_expert`, does not separate wins from losses across architectures [HW sweep
2026-08-27]. One lower bound on `M_e` is the tile granule, 64 rows, the smallest tile the resident
path can run. That bound comes from the mechanism, and the measured sign change is not at it.

The map below plots `M_e` against `default ÷ ROCKET_MOE=0` over `-p 512,768,1024,1536,2048` on both
MoE models.

| | `M_e` | work per dispatch | `ROCKET_MOE=0` | default | ratio |
|---|---:|---:|---:|---:|---:|
| **gpt-oss-20b**, 4-of-32, expert GEMM 2880×2880 | 64 | 5.31e8 | 14.08 | 23.08 | **1.64x** |
| | 96 | 7.96e8 | 14.22 | 25.13 | **1.77x** |
| | 128 | 1.06e9 | 14.10 | 26.92 | **1.91x** |
| | 192 | 1.59e9 | 13.88 | 28.38 | **2.05x** |
| | 256 | 2.12e9 | 13.64 | 28.30 | **2.08x** |
| **DeepSeek-V2-Lite**, 6-of-64, expert GEMM 2048×1408 | 48 | 1.38e8 | 25.94 | 26.03 | 1.00 (declined) |
| | 72 | 2.08e8 | 26.35 | 24.86 | **0.94x** |
| | 96 | 2.77e8 | 23.06 | 24.18 | 1.049x *(marginal)* |
| | 144 | 4.15e8 | 22.33 | 27.20 | **1.22x** |
| | 192 | 5.54e8 | 21.60 | 28.37 | **1.31x** |

**Conditions:** gpt-oss-20b MXFP4 and DeepSeek-V2-Lite Q4_K_M, 2026-08-27. Every offloaded test
point is 100% resident with 0 streamed, so no ratio here is a residency effect.

**Not characterized:** the `M_e` = 72 and 96 test points on the 2026-10-09 build. Its readback and
activation-pack changes reduce the expert call's cost on gpt-oss's shape
([benchmarks.md](benchmarks.md) §"gpt-oss-20b native-quant expert path and results"). The
threshold is fitted on the 2026-08-27 measurements.

Against the experts-on-CPU configuration, the gpt-oss model measures 1.64x at `M_e` = 64, while
DeepSeek is 5-6% slower at `M_e` = 72. Any row threshold low enough to admit the first admits the
second, and no row threshold separates them. The loss at `M_e` = 72 belongs to the expert path,
not to the threshold. Auto placement admits every DeepSeek stack at every prefill length, so
`ROCKET_MOE=1` is the same placement there. It measures 25.55 against the same 26.35 control.

The threshold amortizes a per-dispatch cost, incurred once per `(op, expert-with-rows)`:

- One gather
- One per-`(row, K-group)` activation quantize
- One submit
- One fence
- One scatter

One dispatch performs `M_e · K · N` MACs. For the NPU to beat the CPU on that expert,
`fixed_dispatch < M_e · K · N · (1/rate_cpu − 1/rate_npu)` must hold. The condition is a threshold
on the product, and `M_e` stands in for it only while `K · N` is held constant. Across these two
models `K · N` is not constant. The gpt-oss expert GEMM is 2.88x larger than DeepSeek's at the same
row count, which accounts for the whole of the discrepancy above.

Ordered by that product, the test points of both architectures separate with no overlap:

- The one clear loss is at 2.08e8 MACs.
- The boundary test point at 2.77e8 is a marginal 1.049x (0.94-1.08 over its repeats), which does
  not repay its own ingest.
- Every test point with a material gain is at or over 4.15e8.

The default threshold is the geometric midpoint of that last gap, 340 mega-MACs.

The threshold's position rests on DeepSeek-V2-Lite's bracket alone. The gpt-oss model contributes
nothing to where it sits. Its smallest reachable test point, 5.31e8, is 1.56x the 3.4e8
threshold and above the bracket's upper end of 4.15e8 [host-computed]. The threshold's form is
derived and its value is fitted on two architectures, so it is a measured boundary rather than a
bound. The form does not contain `rate_cpu`, and the two models' CPU kernels differ (MXFP4
against Q4_K). The gpt-oss model at 5.31e8 therefore wins 1.64x where DeepSeek at 5.54e8 wins
1.31x, with the same work per dispatch and a different margin.

The threshold demands a material gain rather than `> 1.00`, because of the one-time expert ingest.
The ingest takes ~18 s per `llama_context` on DeepSeek and ~21 s on gpt-oss [HW sweep 2026-10-09,
RK1, ~816 MHz, governor `ondemand`]. The ingest is charged only on an accepted offload, so the
threshold is the one place where it can be avoided. An offload at ratio `r` saves `1 − 1/r` of
prefill wall-clock time, so the ingest breaks even after:

| test point | ratio | saves | breaks even after |
|---|---:|---:|---:|
| DeepSeek `M_e` = 96 | 1.05x | 4.8% | ~7 700 tokens of prefill at that shape |
| DeepSeek `M_e` = 144 | 1.22x | 18.0% | ~2 300 tokens |
| DeepSeek `M_e` = 192 | 1.31x | 23.7% | ~1 800 tokens |
| gpt-oss `M_e` = 64 | 1.64x | 39.0% | ~1 100 tokens |

**Conditions:** the ratios are the map's 2026-08-27 test points.

Each break-even count is that test point's count at the 2026-08 ingest, scaled by the model's
ingest ratio [host-computed]. The ingest ratio, current pack against earlier pack, is 0.47x on
DeepSeek and 0.51x on gpt-oss.

A test point a few percent above parity therefore costs a short-lived context more than it saves.
The threshold is placed above 1.00 for that reason.

## Qwen3-30B-A3B, the decline side

On Qwen3-30B-A3B, a third MoE architecture, the default declines the experts at `-ub` 2048 and
below. Forcing them onto the NPU with `ROCKET_MOE=1` measures 0.800x the experts-on-CPU
configuration at `-ub` 2048 [HW sweep 2026-08-30]. The decision is right on this model, and right
for the reason the threshold was derived from: the work per dispatch.

**Conditions:** Qwen3-30B-A3B Q4_K_M, pp2048, 2026-08-30.

**Not characterized:** the threshold's position. At 201 MMAC the test point is far enough below
340 that it tests the threshold's sign, not its position. A model between 277 and 415 MMAC is what
would move it.

Qwen3-30B-A3B has 128 experts, 8 active, `n_embd` 2048 and `moe_intermediate` 768. Its `K · N`
is therefore 1.573e6, 5.3x smaller than gpt-oss's and 2.7x smaller than DeepSeek's. At `-ub` 2048
it reaches `M_e` = 128 and 201 mega-MACs, under the threshold. Clearing 340 needs `M_e` >= 217,
which is more than 3472 tokens in one micro-batch. The default therefore declines its experts at
`-ub` 2048 and below, and accepts them at `-ub` 4096, which the RK1 runs.

A 2026-08-27 series holds `ROCKET_MOE=1`'s residency matched across two micro-batch sizes, which
separates the work per dispatch from residency:

| test point | work/dispatch | configuration | t/s | ratio |
|---|---:|---|---:|---:|
| `ub` 2048, `M_e` 128 | 201 MMAC | `ROCKET_MOE=0` | 13.38 / 13.32 / 13.31 | n/a |
| | | **default** | 13.32 / 13.34 | **1.000x** |
| | | `ROCKET_MOE=1` | 11.31 / 10.98 / 11.08 | **0.834x** |
| `ub` 1024, `M_e` 64 | 101 MMAC | `ROCKET_MOE=0` | 14.62 | n/a |
| | | `ROCKET_MOE=1` | 9.66 | **0.661x** |

**Conditions:** `-b` and `-ub` as shown, 2026-08-27. The ratios compare within this series.

The default measures 1.000x the `ROCKET_MOE=0` configuration and prints no MoE line at all. Its
ingest is zero.

The conclusion does not depend on residency. Experts make up 29 of this model's 30.5 B parameters,
so it cannot be held resident on a 31 GB board at any budget. Under `ROCKET_MOE=1` the int8 ingest
stops at 9539 of 15277 experts, 62.4%, and the rest streams. Two results separate the work per
dispatch from residency:

- At matched residency, halving the work per dispatch lowers the ratio from 0.834 to 0.661. The
  `ub` 1024 test point holds residency at 62.5% against the `ub` 2048 test point's 62.4%, 9548
  experts against 9539. Work per dispatch alone moves the result by 0.17, and it behaves exactly as
  the cost model predicts.
- Raising residency to 70.8% with `ROCKET_MOE_CACHE_MB` raises the ratio only to 0.863. The
  forced configuration loses at every residency this board can reach.

On a third architecture the threshold changes placement. A 128-expert, 8-used model with a
2048×768 expert is at 201 MMAC at `M_e` = 128, which a row threshold accepts and this one
declines.

## Qwen3-30B-A3B, the accept side and the budget

At `-ub` 4096, Qwen3-30B-A3B reaches `M_e` = 256 and 403 mega-MACs, and the default accepts its
experts. It is the first test point this model reaches above the 340 MMAC threshold. The default
measures 1.016x the `ROCKET_MOE=0` configuration, and `ROCKET_MOE=1` measures 0.964x [HW sweep
2026-08-28]. At only 1.19x the threshold, the test point tests its position rather than clearing it
comfortably.

**Conditions:** Qwen3-30B-A3B Q4_K_M, `-p 4096 -b 4096 -ub 4096`, 2026-08-28.

**Not characterized:** the threshold's position apart from a residency term, and any other
architecture or operating point.

The result covers one architecture at one operating point. The clock, the governor and `rate_npu`
are held fixed, on one model. Residency cannot reach 100% on this model on this board. Only a model
that both falls near 340 MMAC and fits resident can separate the position from residency.

At 403 MMAC this test point is 1.19x the threshold. The tightest accept-side test point this
architecture reaches is `-ub` 3584 -> `M_e` 224 -> 352 MMAC, 1.035x the threshold. A run there
would test the position rather than the sign.

`M_e` is set by the micro-batch, so `-ub` is the only option that moves the test point. The
prompt length `-p` needs only to be large enough to fill it. A 31 GiB board runs
`-b 4096 -ub 4096` with the 17.28 GiB model loaded.

| configuration | t/s, three processes | spread | paired ratios | mean | placement |
|---|---|---:|---|---:|---|
| `ROCKET_MOE=0` | 10.29 / 10.32 / 10.31 | 0.3% | n/a | n/a | n/a |
| **default** | 10.40 / 10.50 / 10.50 | 1.0% | 1.011 / 1.017 / 1.018 | **1.016x** | 79 stacks, 0 streamed |
| `ROCKET_MOE=1` | 9.80 / 9.93 / 10.09 | 2.9% | 0.952 / 0.962 / 0.979 | **0.964x** | 62.6% resident, 5703 streamed |

The two configurations place the experts differently. The default reserves whole stacks and streams
nothing: it holds 79 and leaves the rest on the CPU. Its placement is the same in all three
processes: 8818 experts, 14877 MB, 0 streamed. With `ROCKET_MOE=1`, every expert is claimed and
37% of them stream. Published figures for this model elsewhere are `ROCKET_MOE=1` figures, which
do not describe the default.

The three `ROCKET_MOE=1` test points of the 2026-08-27 and 2026-08-28 series are residency-matched
(62.6% against 62.4%). They trace the ratio against work per dispatch alone: 101 MMAC -> 0.661,
201 -> 0.834, 403 -> 0.964. The curve is saturating rather than linear. A cost model predicts the third point out of sample to 0.4%. The
model is `ratio = A / (B + (P/ub)·k)`, with one term fixed per prefill and one proportional to the
dispatch count. Fitted on the two lower test points, it gives `k` = 0.314A and `B` = 0.885A, and
predicts 0.960 here against 0.964 measured.

> [!NOTE]
> Where the ingest is charged moves this test point by 12%. The `llama-bench` warm-up incurs most of
> the one-time expert ingest, and the timed repetitions incur only the experts it never routed to.
> With the warm-up off, the same test point measures 0.934x instead of 1.016. A figure quoted from
> this test point must state which charging it used.

### The auto budget and `ROCKET_MOE_CACHE_MB`

The automatic budget over-places, and on this model the over-placement costs the whole offload.
The option `ROCKET_MOE_CACHE_MB` sets the pre-flight's budget directly. On the default
configuration it changes only the number of admitted stacks, with zero streaming at every budget.
The ladder below therefore has one regime and one varying quantity [HW sweep 2026-08-28].

**Conditions:** as above. Ratios are against the pooled `ROCKET_MOE=0` mean of six processes. The
`n` column is the number of processes, and the rep sd column the spread between repetitions.

The ingest and repaid-after columns assume an NPU-BO pack of ~505 MB/s [HW sweep 2026-08-28]. The
pack measured on gpt-oss runs ~1.3 GB/s [HW sweep 2026-10-09]. At that rate the pack term of each
ingest is ~0.35x as large, so both columns overstate the cost [host-computed].

| budget | stacks | resident | IOVA | ingest | t/s | n | ratio | rep sd | ingest repaid after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 0 | n/a | n/a | n/a | 10.32 | 6 | 1.000x | ±0.01 | n/a |
| 12000 MB | 38 | 7813 MB | 7296 | 20.2 s | 10.65 | 1 | 1.032x | ±0.00 | ~6 700 tokens |
| **18000 MB** | **58** | 11430 MB | 11136 | 30.9 s | **10.76** | 3 | **1.043x** | ±0.03-0.08 | ~7 400 |
| **21000 MB** | **67** | 13167 MB | 12864 | 36.7 s | **10.83** | 4 | **1.049x** | ±0.04-0.08 | ~7 700 |
| auto (24425) | 79 | 14877 MB | 15168 | 43.2 s | 10.47 | 3 | 1.014x | ±0.6 | ~34 000 |
| 28000 MB | 90 | 16716 MB | 17280 | 48.5 s | 10.31 | 1 | 0.999x | ±1.02 | never |

The ratio rises to a plateau and then falls, and the automatic budget is past the fall. The
18000 MB and 21000 MB budgets, 58 and 67 stacks, are one plateau, pooled at 1.046x over seven
processes. Between 67 and 79 stacks the ratio falls to 1.014.

The 12000 MB and 28000 MB rows are one process each. A single process on this board can sit ~10%
off the level its own configuration repeats at [HW sweep 2026-08-28]. The 28000 MB row's 0.999x
therefore shows one process at parity, not that the offload gives nothing at 90 stacks.

**Not characterized:** the depth of the fall past 79 stacks.

The ladder's repaid-after column uses the 2026-08 pack rate. At that rate the test points of the
break-even table (§"The accept boundary") break even after ~16 300 tokens (1.05x), ~4 800 (1.22x)
and ~3 800 (1.31x). Against those counts, the automatic budget's admission repays after ~34 000 tokens, about twice the 1.05x
test point the threshold was fitted to exclude. The plateau repays after ~7 500, between the 1.05x
and 1.22x test points. The recommended budget is therefore a range, 18000-21000 MB, rather than a
value that has to be hit exactly.

### Budget units and the charge factor

The `ROCKET_MOE_CACHE_MB` budget is in code-relative units, so a value does not transfer to another
quant or another board. The budget bounds the admission charge rather than RAM. The charge is the
int8 codes plus the per-(channel, K-group) scales plus that expert's GGUF source bytes. A budget
therefore gives strictly less residency than it names. Across the ladder:

| budget | resident int8 | budget / resident |
|---:|---:|---:|
| 18000 MB | 11430 MB | 1.58 |
| 21000 MB | 13167 MB | 1.60 |
| 24425 MB | 14877 MB | 1.64 |
| 28000 MB | 16716 MB | 1.68 |

The charge's own form predicts the ratio. Per expert, the charge is `N·K` code bytes +
`N·(K/group)·4` scale bytes + the source stride `nb[2]`. Against the codes alone:

```
charge / codes  =  1 + 4/group + source_bits_per_weight/8
```

This GGUF is 18556686912 bytes for 30.5 B parameters, which is 4.87 bits/weight. At `K`=2048
the group auto-picks to 512, giving 1.617. The scales term is the small one (0.8% here), and the
source term sets the ratio.

The pre-flight's log gives the factor directly, and that is the number to compare against. The
budget-over-resident column above only estimates it, since a budget is what was requested. The
pre-flight prints both halves of the charge:

```
[moe-int8] resident budget reached after 79 expert stacks (24425MB RAM, 15168MB IOVA)
```

RAM is the charge and IOVA is the int8 codes, so their ratio is `1 + 4/group + bits/8` with nothing
inferred. Measured that way, on two models and two quant formats:

| model | quant | charged / codes | derived | agreement |
|---|---|---:|---:|---:|
| Qwen3-30B-A3B | Q4_K_M, group 512 | 24425 / 15168 = **1.610** | 1.617 | 0.4% |
| gpt-oss-20b | MXFP4, group 576 | 21433 / 14092 = **1.521** | 1.538 | 1.1% |

The first row is [HW sweep 2026-08-28, RK1]. The second row comes from four archived runs
([data/gpt-oss-20b.md](data/gpt-oss-20b.md)), whose six such lines measure 1.5209-1.5211. The
derived value is ~1% high in both cases, so the form is a good sizing rule and not an exact one.
Two caveats on its input explain the direction. The whole file's average, 4.87 bits/weight, stands
in for the expert tensors' own rate, and a Q4_K_M GGUF mixes types.

The same physical placement therefore needs a different budget for a different quant of the same
model. The factor is about 2.07 for Q8_0 (8.5 bits/weight) and 1.54 for IQ4_XS (4.25)
[expected: derived from bits/weight, not measured].

To size a budget before a run, derive the factor. For an exact value, read the factor off the
pre-flight line. The derivation transfers across quants and boards, and the log fixes the factor
for one model. Take the ratio from a line whose stack count is limited by RAM, which says "budget
reached ... Bound by RAM". A run at a pinned `ROCKET_MOE_CACHE_MB` stops at the pin rather than at
the natural budget.

### Over-placement mechanism

The automatic budget over-places because of an omission in the charge [hypothesis]. The pre-flight
charges each placed expert as int8 codes plus that expert's GGUF source bytes. It does not charge
two things. The first is the rest of the expert GGUF, the unplaced experts, which the CPU reads
from the same mmap every micro-batch. The second is the placed experts' own source pages, which the
ingest faulted moments earlier. Charging the whole expert GGUF against the anonymous int8 set gives:

| stacks | anonymous int8 | + whole expert GGUF (16.3-17.8 GB) | vs 31744 MB total |
|---:|---:|---:|---|
| 38 | 7813 | 24.1-25.6 GB | under by 6-7.6 |
| 58 | 11430 | 27.7-29.2 GB | under by 2.5-4 |
| 67 | 13167 | 29.5-31.0 GB | under by 0.8-2.3 |
| **79** | 14877 | **31.2-32.7 GB** | **crosses 31.7** |
| 90 | 16716 | 33.0-34.5 GB | over by 1.3-2.8 |

The crossing is at 79 stacks, which is where the ratio falls. The expert GGUF's own size is bounded
rather than known, because the stacks are not equal-sized. The pre-flight's per-stack charge puts
it at 16.3 GB at low placement and 17.8 GB at high placement. The fit is therefore one of ordering
and magnitude, not a coincidence to 0.1%. The 6 GiB reserve exists to absorb exactly this, and the
uncharged remainder here is 7-12 GB.

The fit needs the placed experts' source pages to stay resident after ingest. Without them the hot
set is flat across the five budgets and crosses nothing.

**Not characterized:** which pages the kernel retains, and for how long.

The factor also reconciles the RAM arithmetic with the recommended budget. On this board,
`MemTotal` minus the whole expert GGUF minus the reserve is 31.7 − 17.3 = 14.4 GB, far from
21000 MB. Converted through the measured factor, with a runtime-headroom term, it gives
`1.610 × (31.7 − 17.3 − 1.2)` = 21.3 GB. The form is consistent and is not an independent
prediction: the headroom term is the one free parameter, chosen to match the measured plateau.

The mechanism predicts its own scope, and the prediction holds where it can be checked. The defect
needs the experts to dominate the model and the GGUF to be large against RAM. The gpt-oss-20b model
is neither (11.27 GiB). There, `ROCKET_MOE_CACHE_MB=28000` measured +8.3% / +14.1% with no
degradation. The gpt-oss ladder is one `-r 3` process per configuration, so the two deltas have the
same n=1 caveat as this ladder's ends. They show a direction rather than a magnitude.

The two ladders jointly establish the sign flip between model classes, a comparison of directions
that the single-process caveat does not affect. A `ROCKET_MOE_CACHE_MB` recommendation is therefore
scoped to the model class, not to the board size.

## The induced-scarcity ladder

On gpt-oss-20b, a ballast of `B` GiB of touched anonymous memory is held beside the run. Prefill
throughput falls as `B` rises, with no turn at any step [HW sweep 2026-08-28]. Every step
reports 100% resident, 0 streamed, and no OOM line in `dmesg`. Fewer stacks are placed, and the
rest stay on the CPU, which the code calls "a partial offload and not a loss".

**Conditions:** gpt-oss-20b MXFP4, `-p 2048 -n 0 -r 1 -b 2048 -ub 2048`, one process per step,
2026-08-28. The page cache is dropped
and `compact_memory` run before every step.

| `B` | MemAvailable | pre-flight budget | stacks placed | resident | streamed | pp2048 t/s | vs `B`=0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30.2 GB | 24598 MB | 63 | 13454 MB | **0** | 28.93 | 1.000 |
| 4 | 26.2 | 20488 | 52 | 11317 | **0** | 24.04 | 0.831 |
| 8 | 22.2 | 16401 | 42 | 9486 | **0** | 20.89 | 0.722 |
| 12 | 18.2 | 12290 | 31 | 7348 | **0** | 18.16 | 0.628 |
| 16 | 14.2 | 8225 | 21 | 5372 | **0** | 15.36 | 0.531 |

Throughput falls monotonically and very nearly linearly in the number of placed stacks
(`t/s = 8.32 + 0.315 x stacks`, residuals within ±0.8 t/s over a 15-29 range).

The ladder's variable and the pre-flight's input are the same quantity. The pre-flight budget is
`MemAvailable - 6 GiB`, and the budget column falls by exactly 4096 MB per 4 GiB step. Ballast
therefore makes the pre-flight more conservative, never less. The memory the ballast removes is
removed from the budget first.

> [!NOTE]
> An induced-scarcity ladder of this shape cannot test whether the pre-flight over-places. The
> pre-flight subtracts the ballast from its budget first, so the expert path never holds more than
> the board can honor. Only memory that disappears after the budget is frozen reaches that state.

The hot-set account of int8 codes, the whole GGUF and the ballast over-counts a file-backed GGUF.
At `B` = 16 the accounted hot set is 5372 MB of int8 codes + 11.27 GiB of GGUF + 16 GiB of
ballast. The total, 32.5 GB, exceeds the 31.7 GB of RAM. Nothing streamed, nothing was killed,
and the linear fit did not bend. At least some of what that account calls hot is therefore
reclaimed under pressure. The GGUF, file-backed unlike the ballast, is the only candidate
[hypothesis: not separated from a smaller-than-assumed activation footprint].

The charge model treats each placed expert's source bytes as unreclaimable because decode reads
them from the mmap every token. During a `-n 0` prefill they are not touched after ingest.

The ladder does not refute the charge omission itself. The uncharged remainder is still real and
still largest at low placement, and the ladder never reaches a state where it takes effect.

> [!CAUTION]
> The pre-flight resolves its budget once and keeps it. Memory that another process takes after
> that is still counted as available, so the expert path reserves stacks against RAM that is gone.
> In the run below the kernel OOM killer fired, and the run still exited with status 0 and a
> plausible throughput. Nothing in its output says a process was killed.

A second run, with the same model and flags, allocates the 16 GiB of ballast after the pre-flight
line prints. It reaches the over-placed state [HW sweep 2026-08-28, RK1]:

| | budget frozen at | MemAvailable when the ingest ran | stacks reserved | outcome |
|---|---:|---:|---|---|
| ballast **before** (`B`=16 step above) | 8225 MB | 14.2 GB | 21 | 100% resident, 0 streamed, no OOM |
| ballast **after** the freeze | **24608 MB** | **11.1 GB** | **63** | kernel **OOM-killer fired**, 91% resident, 144 streamed, pp2048 25.53 |

The pre-flight resolved the budget at 24608 MB against an idle board, and kept it while 18 GB of
the board disappeared. It therefore went on reserving 63 stacks and 24529 MB against 11.1 GB of
available RAM. The board recovered only because the OOM killer picked the ballast, 16.8 GB of
anonymous RSS, the largest badness score on the board. A competing allocation smaller than the
inference process would have made `llama-bench` the victim instead.

The two rows are a controlled pair. At essentially the same free RAM (14.2 against 11.1 GB), the
pre-flight reserved 8225 MB in one and 24608 MB in the other. The only difference is when the
memory went away: the pre-flight has no run-time `MemAvailable` check.

## RAM and IOVA budgets

The charge factor (§"Budget units and the charge factor") also decides which budget binds, and on
this path it is never the IOVA window. The pre-flight holds two budgets: RAM, and `ROCKET_N_THREADS x 3840 MB` of NPU IOVA. It charges each
stack `codes + scales + source` against the first. It charges `codes` alone against the second.
The window therefore binds first only where

    RAM budget / IOVA budget  >  1 + 4/group + source_bits_per_weight/8

which is the same factor. On `gpt-oss-20b` MXFP4 the right side is 1.538, and the left is 1.28 at
the default five workers. RAM binds, by a margin no worker count can close, because raising
`ROCKET_N_THREADS` grows only the denominator. At 5 and at 8 workers the admitted set is the same,
and the pre-flight reports `Bound by RAM` [HW sweep 2026-09-02].

The window cannot bind on this path at or above the default worker count. The threshold is set by
the source rather than by the board. At five workers it would need `MemAvailable` above 35.7 GB,
and each further worker adds about 5.9 GB to that. On a 32 GiB board the inequality needs a source
under about 2.2 bits per weight, which no quantization this path ingests approaches. MXFP4, the
densest measured, is 4.25. The option that `Bound by RAM` names, `ROCKET_MOE_CACHE_MB`, is the only
one this path's line ever asks for.

> [!NOTE]
> A missing pre-flight line does not mean the pre-flight passed. Auto
> placement checks the tile granule and the per-dispatch work threshold before it calls the
> pre-flight [source-confirmed, `ggml-rocket.cpp`]. If the line is missing, raise `-ub` before
> concluding anything about RAM or the window.

`Qwen3-30B-A3B` Q4_K_M at `-b 2048 -ub 2048` has 128 experts with 8 used. A 2048-token micro-batch
therefore gives `M_e` = 128, and the work threshold declines it first. Two runs at that shape, at 5
workers and at 8, emit no `[moe-int8]` line at all [HW sweep 2026-09-02, RK1]. The same model
reaches the pre-flight at `-ub 4096`, where `M_e` is 256.

## Evidence

### The accept boundary

The map's test points are `-r 3` each, on one board and one `.so` [HW sweep 2026-08-27, RK1].

Each test point's pair count follows from its distance from 1.00. The offloaded configuration
varies ~15% run to run on DeepSeek. A test point a few percent from parity therefore needs
repeats, and one at 1.6-2.1x does not. The pair counts:

- `M_e` = 72 and 144 are means of four adjacent pairs.
- `M_e` = 96 is a mean of eight, because it is the test point that decides where the threshold
  goes.
- `M_e` = 192 is a mean of two, taken either side of the rebuild.
- Every gpt-oss test point is a single pair. Its margin from parity, 0.6-1.1, is 4-7x the same
  spread [host-computed].

At `M_e` = 96, one of the eight pairs measures 0.936 and the other seven measure 1.019-1.076. All
eight are placement-identical (4767 resident, 0 streamed, the same ingest profile). Nothing but
variance separates them, and there is no mechanistic ground to drop the low one. Pooled, the test
point is 1.049, and over the seven repeats alone it is 1.065. Either way it is marginal.

The `M_e` = 48 row is the null test point. The default threshold declines it, so both
configurations are the same placement, and they agree to 0.3%. The agreement shows that the map
measures the threshold and not run-to-run drift.

The DeepSeek `M_e` = 72 test point has four pairs, every one under 1.00 and none above 0.970.

The ingest ratio in the break-even table compares the earlier pack with the current one in one
session. At the 2026-08 ingest (~32 s on DeepSeek and ~36 s on gpt-oss, a pack of ~505-545 MB/s)
the four test points break even after ~16 300, ~4 800, ~3 800 and ~2 100 tokens. The table scales
those counts by 0.47x and 0.51x.

#### Excluded causes

Bucket padding is the alternative explanation that the map sets aside. The bucket ladder rounds
`M_e` up (64, 96, 128, 192, 256…). DeepSeek's points therefore alternate 33% padding (72 -> 96,
144 -> 192) with none (96 -> 96, 192 -> 192). Padding predicts a sawtooth, in which each step onto
an unpadded bucket is steeper than the step off one.

The measured curve is monotone and saturating instead. Its per-row slopes across `M_e` 72 -> 96 ->
144 -> 192 are 0.0042, 0.0037 and 0.0018. One step agrees with the sawtooth and the next
contradicts it, while a falling slope is exactly what amortizing a fixed per-dispatch cost looks
like. Padding is capped at ~33% by construction and is not the term that flips the sign.

### Qwen3-30B-A3B, the decline side

The 2026-08-30 figure is the Qwen3-30B-A3B unit of the tuning series
([data/tuning-matrix.md](data/tuning-matrix.md)), on llama.cpp b10558 at `-p 2048 -n 0 -r 3`. It
ran three passes with the configuration order rotated. Each configuration started from a memory
reset (`drop_caches` and `compact_memory`) and a discarded warm-up process. `ROCKET_MOE=1` measures
10.71 t/s against 13.39 for `ROCKET_MOE=0`, with per-pass ratios of 0.804, 0.790 and 0.804
[host-computed]. Under `ROCKET_MOE=1`, 9427-9437 experts are resident and 5842-5849 stream, 62%. The
default places nothing in any pass and measures 13.38.

The 2026-08-27 series' test points are [HW sweep 2026-08-27, RK1]. Each is `-r 3`, over three
process-pairs with the configurations alternated. The warm-up process is discarded and the page
cache dropped first.

The default's 1.000x is the null test point, and shows that the threshold is what is being
measured. The `ROCKET_MOE=0` control's spread across three processes is 0.5%, and `ROCKET_MOE=1`'s
is 3.0%. The 17% deficit is far outside both, and the per-pair ratios barely move: 0.845 / 0.824 /
0.832.

Residency confounds the decline-side result. A partially resident configuration is strictly worse
than a resident one. A `ROCKET_MOE=1` result above 1.05 would therefore have falsified the
threshold outright. One result below 1.05 cannot by itself separate "the work per dispatch is too
small" from "it cannot be held". The residency-matched pair and the residency axis separate them.

The residency axis in full: `ROCKET_MOE_CACHE_MB` raises residency 62.4% -> 65.7% -> 70.8%, and the
ratio moves 0.834 -> 0.856 -> 0.863. The 8.3 points of residency give 0.029 of ratio. Extrapolated
linearly, full residency would measure 0.965x. Reaching parity would need 110% residency
[extrapolated from two points, 29 points beyond the data, indicative, not measured]. On that trend
`ROCKET_MOE=1` loses at every residency at all.

### Qwen3-30B-A3B, the accept side and the budget

The accept-side test point is [HW sweep 2026-08-28, RK1]. It is `-r 2`, over three
process-triples (configuration order a balanced Latin square). The warm-up process is discarded and
the page cache dropped before every test point. A single-test-point run first showed the
configuration is reachable. The default's placement line is identical in all three processes,
which dropping the page cache before every test point gives.

The cost model's `P` term is required: `-ub` 4096 needs `-p` >= 4096, so the prefill work doubles
while the per-prefill constants do not. A `P`-fixed extrapolation of the same two points gives
~1.007, wrong by 5%.

#### The auto budget and `ROCKET_MOE_CACHE_MB`

The pooled `ROCKET_MOE=0` control measures 10.29 / 10.32 / 10.31 / 10.35 / 10.39 / 10.27 across six
processes spanning seven hours. Its 1.2% spread is non-monotone, so it is the configuration's own
spread rather than drift.

The two middle budgets are not separable: 10.76 against 10.83 is 0.65% apart, against
configurations whose own spreads are 1.1% and 1.6%. The plateau is seven processes over two
budgets and the automatic budget is three, but 12000 and 28000 are n=1.

The fall's existence rests on the plateau-to-automatic step, which has the depth. Its depth past 79
stacks does not. Per-repetition spread grows monotonically with placement across the whole ladder
(±0.00 -> ±1.02), a memory-pressure signature rather than anything about the offload. The two
single-process budgets are the least trustworthy for the same reason: they were taken where the
spread is widest.

#### Over-placement mechanism

The curve is itself the evidence that the placed experts' source pages stay resident. Suppose they
were promptly reclaimable after ingest, since they are read once and this is a prefill-only run.
The hot set would then be `anonymous + unplaced GGUF only`, which computes to 20.6 / 21.7 / 22.4 /
22.2 / 22.4 GB across the five budgets. That curve is flat, crosses nothing and explains nothing. A
curve exists, so those pages are not promptly reclaimed.

### The induced-scarcity ladder

The ladder tests the MoE budget hypothesis: that the per-expert admission charge covers only the
placed experts' GGUF source bytes. The rest of a still-mapped expert GGUF is then charged to nobody,
so the pre-flight over-places once the true hot set approaches RAM. The hypothesis has an arithmetic
fit to five budgets on one model, and no second model on this board reaches the regime unaided.

The ladder induces the scarcity on a second model. It holds `B` GiB of touched anonymous memory
beside the run and walks `B` up. The account predicts the turn once `25.3 + B` exceeds ~31.7, which
is at `B` ~ 6-7 GB and not at `B` = 4. The predicted collapse into the 0.42-0.97x partial-residency
regime does not happen. Neither does the earlier turn that the "any memory pressure at all"
alternative predicts.

The ladder ran one process per step (`perf/data/moe-ballast-ladder.sh`).

### RAM and IOVA budgets

Three runs show the RAM budget binding on gpt-oss-20b, two at the default 5 workers and one at 8.
The window grows 1.6x between them, from 19200 to 30720 MB. The admitted set is byte-identical
across all three, at 62 stacks, 24140 MB RAM and 15693 MB IOVA. All three announce `Bound by RAM`
[HW sweep 2026-09-02, RK1, `-b 2048 -ub 2048 -p 2048`, raw in
[data/ro-session/trackd17-moe-preflight.md](data/ro-session/trackd17-moe-preflight.md)].

Every pre-flight decline recorded for this model says the same. That is 14 occurrences over five
distinct budget and placement points, from an induced 11.7 GB budget to a pinned 25.7 GB one. None
names the window.

One archived gpt-oss line measures 1.3926 at 10.70 MB/expert against the others' 8.07. That is a
different expert geometry, not a disagreement.

### Revision history

| Quantity | Earlier figure | Its conditions | Current figure |
|---|---|---|---|
| Qwen3-30B-A3B, `ROCKET_MOE=1` against `ROCKET_MOE=0` at `-ub` 2048 | 0.834x, 62.4% resident | 2026-08-27, three process pairs with the configurations alternated, the page cache dropped first | 0.800x, 62% resident, 2026-08-30, three rotated passes with a memory reset before every configuration, both at ~816 MHz. The 2026-08-27 series stays in the reading path as the matched-residency comparison with `-ub` 1024 |
