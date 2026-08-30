<!-- Raw bench-tuning-matrix.sh output: the default-vs-tuned flag axis, one section per unit.
     Every arm runs on the NPU -- this table's question is "what does following the tuning guide
     buy", which is a ratio of two NPU arms, so there is no CPU arm here (CPUARM=1 adds one; the
     per-model CPU reference is already in the sibling files). MODE=headline, so pp2048 only.

     Method: RK1 (RK3588), NPU pinned 600 MHz, CPU governor `performance` on all three policies
     (cur = max, 1.8 / 2.4 / 2.4 GHz), package 41-59 C across the landed units, board otherwise idle. Each arm
     resets the board's memory (drop_caches + compact_memory) and then runs a discarded warm-up
     process, both outside the timed region, and the unit is repeated over PASSES interleaved
     passes with the ARM ORDER ROTATED, ratios paired within a pass. llama-bench -r 3 after its own
     internal warm-up. Stock is spelled as an ABSENCE of settings, not as flags that match today's
     defaults, so a default moving underneath this file changes the measurement instead of hiding
     in it.

     THE PASSES ARE NOT OPTIONAL, and this file's first unit is why. Measured once per arm, unit 1
     read stock 108.46 and `-b 2048 -ub 2048` 110.92 -- a 1.023x lever. Measured over three
     interleaved passes the same lever is 1.130x: the single stock reading was one of this board's
     occasional ~10% high excursions, and it understated the effect SIXFOLD. Read the per-pass
     ratio column before quoting any row; a set that straddles 1.00 has not resolved.

     Each residency arm carries its own [f16-resident] outcome line, so a row that reads as no
     gain can be told from a row whose residency was declined. Read it before quoting a zero.
-->

# The default-vs-tuned matrix

Units landed so far, pp2048 t/s, each ratio paired within a pass against that unit's own stock arm.

| unit | class | stock | tuned arm | t/s | paired ratio | per-pass | placement |
|---|---|---:|---|---:|---:|---|---|
| `qwen35-08b` | quant | 99.09 | `-b 2048 -ub 2048` | 111.93 | **1.130x** | 1.131 1.138 1.120 | — |
| `qwen35-08b` | quant | 99.09 | `+ ROCKET_QUANT_RESIDENT=auto` | 116.86 | **1.179x** | 1.186 1.183 1.169 | 150 resident, 0 streamed |
| `qwen35-08b-f16` | f16 | 120.08 | `ROCKET_F16_RESIDENT=auto` | 122.35 | 1.022x, **unresolved** | 1.109 0.894 1.059 1.093 1.007 1.046 0.944 | 126 → 150 resident, 0 streamed |
| `llama32-3b-f16` | f16 | 56.01 | `ROCKET_F16_RESIDENT=auto` | 60.54 | **1.081x** | 1.082 1.073 1.088 | none → 193 resident (5232 MB), 0 streamed |
| `ministral3-3b-f16` | f16 | 48.28 | `ROCKET_F16_RESIDENT=auto` | 52.04 | **1.078x** | 1.082 1.074 1.077 | none → 179 resident (5610 MB), 0 streamed |
| `phi4mini-f16` | f16 | 50.19 | `ROCKET_F16_RESIDENT=auto` | 54.40 | **1.084x** | 1.091 1.083 1.078 | none → 126 resident (6000 MB), 0 streamed |
| `llama32-3b` | quant | 38.96 | `-b 2048 -ub 2048` | 42.40 | **1.089x** | 1.076 1.098 1.091 | — |
| `llama32-3b` | quant | 38.96 | `+ ROCKET_QUANT_RESIDENT=auto` | 47.21 | **1.212x** | 1.201 1.227 1.208 | 193 resident (5232 MB), 0 streamed |
| `ministral3-3b` | quant | 34.82 | `-b 2048 -ub 2048` | 36.15 | **1.038x** | 1.033 1.045 1.037 | — |
| `ministral3-3b` | quant | 34.82 | `+ ROCKET_QUANT_RESIDENT=auto` | 39.41 | **1.132x** | 1.121 1.138 1.137 | 179 resident (5610 MB), 0 streamed |
| `phi4mini` | quant | 36.02 | `-b 2048 -ub 2048` | 39.38 | **1.093x** | 1.103 1.082 1.095 | — |
| `phi4mini` | quant | 36.02 | `+ ROCKET_QUANT_RESIDENT=auto` | 43.64 | **1.211x** | 1.219 1.207 1.208 | 126 resident (6000 MB), 0 streamed |
| `smolvlm2` | quant | 51.96 | `-b 2048 -ub 2048` | 48.88 | **0.941x** | 0.931 0.938 0.953 | — |
| `smolvlm2` | quant | 51.96 | `+ ROCKET_QUANT_RESIDENT=auto` | 52.04 | 1.002x, **flat** | 0.999 0.996 1.010 | 165 resident (2976 MB), 0 streamed |
| `qwen35-9b` | quant | 19.14 | `-b 2048 -ub 2048` | 27.26 | **1.424x** | 1.408 1.428 1.437 | — |
| `qwen35-9b` | quant | 19.14 | `+ ROCKET_QUANT_RESIDENT=auto` | 31.78 | **1.661x** | 1.639 1.661 1.681 | 200 resident (13184 MB), 0 streamed |
| `ministral3-8b` | quant | 17.79 | `-b 2048 -ub 2048` | 20.92 | **1.176x** | 1.191 1.182 1.155 | — |
| `ministral3-8b` | quant | 17.79 | `+ ROCKET_QUANT_RESIDENT=auto` | 23.56 | **1.325x** | 1.341 1.337 1.295 | 235 resident (13808 MB), 0 streamed |
| `gemma4-12b` | quant | 12.87 | `-b 2048 -ub 2048` | 16.17 | **1.257x** | 1.269 1.252 1.251 | — |
| `gemma4-12b` | quant | 12.87 | `+ ROCKET_QUANT_RESIDENT=auto` | 17.01 | **1.322x** | 1.338 1.315 1.313 | **239-243 resident (15018-15255 MB), 85-89 streamed — 73-74%** |
| `phi4-14b` | quant | 10.91 | `-b 2048 -ub 2048` | 14.27 | **1.308x** | 1.308 1.282 1.335 | — (no F16 staged; fp16 resident does not fit) |
| `qwen36-27b` | quant | 6.06 | `-b 2048 -ub 2048` | 9.28 | **1.532x** | 1.544 1.525 1.527 | — (fp16 resident is ~54 GiB) |
| `gpt-oss-20b` | moe | 22.77 | `-b 2048 -ub 2048` | 28.57 | **1.254x** | 1.258 1.255 1.251 | 1683 experts (13575 MB), 0 streamed |
| `gpt-oss-20b` | moe | 22.77 | `ROCKET_MOE=1` (FORCED) | 34.18 | **1.501x** | 1.489 1.511 1.503 | 1947 experts (15705 MB), 0 streamed |
| `gpt-oss-20b` | moe | 22.77 | `ROCKET_MOE=0` (control) | 13.67 | **0.600x** | 0.596 0.602 0.603 | experts on the CPU |
| `deepseek-v2-lite` | moe | 21.28 | `-b 2048 -ub 2048` | 28.15 | **1.323x** | 1.321 1.324 1.324 | 4779 experts (14959 MB), 0 streamed |
| `deepseek-v2-lite` | moe | 21.28 | `ROCKET_MOE=1` (FORCED) | 28.35 | **1.332x** | 1.330 1.333 1.334 | 4779 experts (14959 MB), 0 streamed |
| `deepseek-v2-lite` | moe | 21.28 | `ROCKET_MOE=0` (control) | 21.61 | **1.016x** | 1.015 1.018 1.014 | experts on the CPU |
| `qwen3-30b-a3b` | moe | 14.74 | `-b 2048 -ub 2048` | 13.38 | **0.908x** | 0.906 0.909 0.908 | **AUTO declines — nothing placed** |
| `qwen3-30b-a3b` | moe | 14.74 | `ROCKET_MOE=1` (FORCED) | 10.71 | **0.726x** | 0.729 0.719 0.730 | 9427-9437 resident (15907-15923 MB), ~5845 streamed — **62%** |
| `qwen3-30b-a3b` | moe | 14.74 | `ROCKET_MOE=0` (control) | 13.39 | **0.908x** | 0.907 0.910 0.908 | experts on the CPU |

Residency's own increment over the `-ub` arm on unit 1 is **1.044x**; it is the second lever, not a
substitute for the first.

**Unit 2 did not resolve, at seven passes**, and the row is kept rather than dropped because the
reason is informative. The per-pass ratios span 0.894-1.109 around a mean of 1.022, so the effect is
inside this board's per-process spread and n=7 is not enough to place it. It is **not** a declined
residency: the teardown line reports 0 streamed on every pass, and it reports the knob doing exactly
what it claims — **126** weights resident under stock, **150** under `auto`. That is the structural
reason to expect ~1.00 here rather than a measurement to chase: the default already residents every
`K<=2048` weight, and on a 0.8B model with `K`=1024 the only weights the knob adds are the 24
`ffn_down` at `K`=3072. A model whose attention and FFN are mostly `K>2048` is where this arm has
something to move.

**The matrix is complete: all 17 units are landed.** The order and the cost model are in `bench-tuning-matrix.sh --list`, and the decomposition is in the project's own open-work tracker, which is not part of this repo. A unit is listed here only once its arms have all run and its
per-pass ratios agree in sign, so a missing unit means not measured or not resolved, never
measured-and-flat.

## What each f16 unit's knob actually moves, read without a timed run

Unit 2 cost seven passes to land unresolved for a reason its own teardown line states, so the same
line was read for every f16 unit before any of them got board time — one warm-up-sized process per
arm, `ROCKET_LOG_STDERR=1` with stderr kept, `-p 512` to clear the `ROCKET_MIN_M` floor
(`perf/data/f16-resident-readout.sh`, ~11 minutes for all four)
[HW sweep 2026-08-28, RK1, 600 MHz]:

| unit | stock | `ROCKET_F16_RESIDENT=auto` | what the knob moves |
|---|---|---|---|
| `qwen35-08b-f16` | 126 resident, 780 MB | 150 resident, 948 MB | +24 weights, +168 MB |
| `llama32-3b-f16` | **no route at all** | 193 resident, 5232 MB | the whole route, off → on |
| `ministral3-3b-f16` | **no route at all** | 179 resident, 5610 MB | the whole route, off → on |
| `phi4mini-f16` | **no route at all** | 126 resident, 6000 MB | the whole route, off → on |

**"No route at all" is not a small number, and it is not a refusal to investigate.** The three
3B-class models print **no `[f16-resident]` line**, which by that line's own condition means nothing
was offered to the resident route — neither resident nor streamed. The mechanism is one clause in
the prepack gate: the default admits a weight only at `K<=2048`, and `ROCKET_F16_RESIDENT` is what
lifts that for all `K`. Every one of these models has `K`=3072, so the default residents **zero**
weights and the knob turns the entire route on.

**So the risk was a property of `K`, not of the f16 class.** Unit 2 was the one model in the set
whose `K` sits *below* the default's threshold, which is exactly why its knob had almost nothing to
add — and it is the only f16 unit for which that is true. The other three carry the largest knob in
the matrix rather than the smallest.

## What the whole residency route is worth: 1.08x, flat across the class

The three 3B-class f16 units ran on 2026-08-29 and all three resolved at three passes
[HW sweep, RK1, 600 MHz, governor `performance`]:

| unit | knob turns on | stock t/s | resident t/s | paired ratio | per-pass |
|---|---|---:|---:|---:|---|
| `llama32-3b-f16` | 193 weights, 5232 MB | 56.01 | 60.54 | **1.081x** | 1.082 1.073 1.088 |
| `ministral3-3b-f16` | 179 weights, 5610 MB | 48.28 | 52.04 | **1.078x** | 1.082 1.074 1.077 |
| `phi4mini-f16` | 126 weights, 6000 MB | 50.19 | 54.40 | **1.084x** | 1.091 1.083 1.078 |

**The ratio does not follow what the knob places.** Weight count spans 1.53x across the three
(126 to 193) and resident footprint 1.15x (5232 to 6000 MB), while the ratio moves 0.6%
(1.078 to 1.084). So this is quotable as one number for the class rather than as a per-model
projection, and a model's weight count is not a way to predict it.

**Do not read the per-GEMM figure as a tuning expectation.** Removing the same weight scatter is
worth 2.3x of wall *on a matmul* (`perf/device-vs-host-split.md`, 127.2 → 55.4 ms at
1024x3840x4096), and the DDR PMU charges that scatter 8.3x the bytes the analytical model counts
for it (`perf/hw-byte-counters.md` §5.2). End to end on a real model the same route buys **1.08x**.
Both numbers are right; they measure different denominators, and only the second is what a user
setting the flag gets.

**The per-process spread is a property of the model, not a constant of the board.** These six arms
repeat across processes to **0.3-1.4%** (stock 55.52-56.26, 48.14-48.45, 50.06-50.37; resident
60.37-60.84, 51.96-52.10, 54.19-54.71). The 0.8B f16 of unit 2, on the same board with the same
protocol, spans **11%** and did not resolve at seven passes. Three passes are therefore enough for
some units and not others for a reason that is visible in the unit rather than in the board, which
narrows what the open spread item has to explain: not "why is this board noisy" but "why is that
model noisy on it".

## What the `-ub 2048` lever is worth per model, and why none of them reaches the published 2.1x

Every quant unit measured under this protocol reads below the ~2.1x that
`quant-prefill-microbatch.md` records for the same flag, including the 9B that figure was taken on
[HW sweep, RK1, 600 MHz, three rotated passes]:

| unit | params | stock (`-ub 512`) | `-b 2048 -ub 2048` | paired ratio | per-pass |
|---|---:|---:|---:|---:|---|
| `qwen35-08b` | 0.75 B | 99.09 | 111.93 | **1.130x** | 1.131 1.138 1.120 |
| `llama32-3b` | 3.21 B | 38.96 | 42.40 | **1.089x** | 1.076 1.098 1.091 |
| `ministral3-3b` | 3.43 B | 34.82 | 36.15 | **1.038x** | 1.033 1.045 1.037 |
| `phi4mini` | 3.84 B | 36.02 | 39.38 | **1.093x** | 1.103 1.082 1.095 |
| `smolvlm2` | 1.81 B | 51.96 | 48.88 | **0.941x** | 0.931 0.938 0.953 |
| `ministral3-8b` | 8.49 B | 17.79 | 20.92 | **1.176x** | 1.191 1.182 1.155 |
| `qwen35-9b` | 8.95 B | 19.14 | 27.26 | **1.424x** | 1.408 1.428 1.437 |
| `gemma4-12b` | 11.91 B | 12.87 | 16.17 | **1.257x** | 1.269 1.252 1.251 |
| `phi4-14b` | 14.66 B | 10.91 | 14.27 | **1.308x** | 1.308 1.282 1.335 |
| `qwen36-27b` | 27.32 B | 6.06 | 9.28 | **1.532x** | 1.544 1.525 1.527 |

All ten resolved at three passes with per-pass ratios agreeing to 2%, so none is a straddling set
being read as settled. The spread within a unit is ~2%; the gap against 2.1x is 0.7-1.2x, so it is
not something this instrument's noise could produce.

**The published figure is not wrong for what it measured; the baseline underneath it moved.** The
2.1x ladder was taken 2026-06-28 on the 9B and 27B. Three host-cost reductions have since become
**default-on**, and by their own documented shape every one of them helps the `-ub 512` arm more
than the `-ub 2048` arm: the threaded streaming dequant (**+33% at `-ub 512` against +19% at
`-ub 2048`**), the persistent dequant worker pool, and the reused `B16`/`C16` context scratch. The
threading A/B published in that same note already shows the consequence — on the 9B the lever falls
from **15.9/7.5 = 2.12x serial to 18.8/10.0 = 1.88x threaded** — and the pool and the scratch reuse
push it the same way. A lever that exists to amortize a fixed per-micro-batch cost shrinks whenever
that cost is cut, so the flag buying less is what success at reducing it looks like.

**BOTH explanations are true, and unit 10 is what separates them.** `qwen35-9b` — the same model
the published 2.1x was measured on — reads **1.424x** under the current build and protocol, three
per-pass ratios agreeing to 2% and a stock arm spanning 1.8%, so it resolved. That number refutes
the two clean stories at once:

- **Size is a real term after all.** The lever is 0.94-1.13x from 0.75 to 3.84 B and **1.176x /
  1.424x** on the two 8-9 B models. Whatever the small models are doing, the large ones are doing
  something different — though **size is not the whole story either**: `ministral3-8b` (8.49 B) and
  `qwen35-9b` (8.95 B) are 5% apart in parameters and **1.21x apart in lever**, so the model matters
  as much as the size does at this end of the range.
- **And the baseline did move, on the very model that carries the claim.** 2.098x published
  (8.2 -> 17.2 t/s, 2026-06-28) against **1.424x** today (19.14 -> 27.26). The published figure is
  not reproducible on its own model.

**The mechanism is confirmed by the asymmetry, which is the part that was predicted.** Between the
two measurements the **stock `-ub 512` arm got 2.33x faster** (8.2 -> 19.14) while the **tuned
`-ub 2048` arm got 1.59x faster** (17.2 -> 27.26). The three default-on host-cost reductions were
documented as helping the small-`-ub` arm more, and that is exactly the shape the two arms moved in;
the lever shrank because the baseline it is measured against rose faster than it did.

**What the five small models could not see was the range, not the count.** Reading 0.941 / 1.038 /
1.089 / 1.093 / 1.130 across 0.75-3.84 B, this file concluded that parameter count predicts nothing.
Six points now say it does — but only past ~4 B, outside the interval those five span. **A corpus
that covers a narrow range of a variable cannot rule that variable out, however many points it
holds**, and adding a sixth model inside 0.75-3.84 B would not have found this. The superseded
readings are kept below because the sequence is the lesson.

**Superseded, in order.** At three units the values were monotone decreasing in parameter count
(1.130x at 0.75 B, 1.089x at 3.21 B, 1.038x at 3.43 B) and that read as a size term running
*opposite* to the published claim. `phi4mini` at 3.84 B read 1.093x and broke the ordering, which
read as no size term at all. `qwen35-9b` at 8.95 B says there is one and it runs the *same* way the
published claim assumed — just to 1.42x rather than 2.1x.

**So what should be quoted.** The `-ub 2048` lever is **model-dependent over 0.94-1.53x** across
0.75-27.32 B, and **the split is at ~4 B**: every model above 8 B reads **1.18-1.53x** and every
model under 4 B reads **0.94-1.13x**. Within the large group parameter count predicts nothing in
detail — 8.49 / 8.95 / 11.91 / 14.66 / 27.32 B read 1.176 / 1.424 / 1.257 / 1.308 / 1.532x, so the
ordering is not clean even though the largest model does now carry the largest lever. **No measured
model reaches the published ~2x.**

## Both published figures fell by the same factor, and that is what confirms the mechanism

The two models the published `-ub` figures were taken on have now both been re-measured under this
protocol, and they did not merely both fall — **they fell by the same multiplier**
[HW sweep 2026-08-29, RK1, 600 MHz, three rotated passes each]:

| model | published (2026-06-28) | measured now | decay |
|---|---:|---:|---:|
| Qwen3.5-9B `Q4_K` | 2.098x (8.2 -> 17.2 t/s) | **1.424x** (19.14 -> 27.26) | **0.679** |
| Qwen3.6-27B `Q4_K` | 2.250x (2.4 -> 5.4 t/s) | **1.532x** (6.06 -> 9.28) | **0.681** |

**Scaling the 27B's published 2.25x by the 9B's decay factor predicts 1.527x against 1.532x
measured — 0.3%.** The two models were not fitted to each other; the factor was measured on one and
carried to the other across a 3x difference in parameters.

The per-arm moves say the same thing. Between the two measurements:

| model | stock (`-ub 512`) arm | tuned (`-ub 2048`) arm | asymmetry |
|---|---:|---:|---:|
| 9B | 2.334x faster | 1.585x faster | **1.473** |
| 27B | 2.525x faster | 1.719x faster | **1.469** |

**Both arms got faster on both models, the stock arm by ~1.47x more in each case, and that ratio is
the same to 0.3%.** That is the shape three default-on host-cost reductions predict — they cut a
fixed per-micro-batch cost, which is a larger share of the `-ub 512` arm's wall than of the
`-ub 2048` arm's — and it is now a quantitative agreement across two models rather than a direction.

**What this does not establish.** The shared factor is two points, and both are Qwen `Q4_K` models
measured in one session against a ladder taken in another; it is not shown to hold across
architectures, and the sub-4 B models have no published ladder to decay from, so nothing here says
where their 0.94-1.13x came from. The mechanism's *direction* is documented in
`quant-prefill-microbatch.md`; what is new is that its *size* is consistent between two models
[HW sweep, n=2].


## The recommended quant config is not a win on every model

`smolvlm2` is the first unit where following the tuning guide end to end buys **nothing**, and the
row is the reason this matrix exists rather than a projection from the flag defaults
[HW sweep 2026-08-29, RK1, 600 MHz, three rotated passes]:

| arm | t/s | against stock | per-pass |
|---|---:|---:|---|
| stock | 51.96 | — | — |
| `-b 2048 -ub 2048` | 48.88 | **0.941x** | 0.931 0.938 0.953 |
| `+ ROCKET_QUANT_RESIDENT=auto` | 52.04 | 1.002x | 0.999 0.996 1.010 |

The `-ub` arm is a **resolved loss** — three per-pass ratios below 1.00 agreeing to 2%. Residency
then recovers it (52.04 against 48.88 is **1.065x**, its own largest increment anywhere in the
matrix) and lands the full recommended stack at **parity**: 1.002x, per-pass straddling 1.00, which
is flat rather than a win too small to see.

**This is not a declined residency, and the teardown line is what proves it.** The arm placed
**165 weights, 2976 MB, 0 streamed — 100% resident** on all three passes. The route did everything
it claims and the model gained nothing from it, which is a different fact from a route that refused
and looked flat. Only the outcome line separates those two, and on the four units above it the same
line accompanies gains of 1.13-1.21x.

**What it costs the guide.** `TUNING.md` recommends the quant pair as a class recipe; on this model
the first half is negative, the second half only cancels it, and a user following the guide ends
where they started while paying 2976 MB of resident footprint for it. That is a recipe cell, not a
projection, and it is why the remaining quant units are worth their board time even where the
expected answer is "yes, tune it".

**The mechanism is open.** Size does not explain it — 1.81 B sits between `qwen35-08b` (0.75 B,
1.130x) and `llama32-3b` (3.21 B, 1.089x), both wins. Nothing measured here says why a larger
micro-batch costs this model wall, and the honest state is one resolved negative with no cause,
not a cause fitted to one point.

## `ROCKET_MOE=1` is not the default made explicit, and the matrix's control arm is what shows it

The `moe` unit's four arms exist so the expert route's contribution can be separated from the
`-ub` lever's. On `gpt-oss-20b` they separate cleanly, and one pair that was expected to be
identical is not [HW sweep 2026-08-29, RK1, 600 MHz, three rotated passes, all four resolved]:

| arm | t/s | against stock | per-pass | experts placed |
|---|---:|---:|---|---|
| stock | 22.77 | — | — | 1683 (13575 MB), 0 streamed |
| `-b 2048 -ub 2048` | 28.57 | **1.254x** | 1.258 1.255 1.251 | 1683 (13575 MB), 0 streamed |
| `ROCKET_MOE=1` + same args | 34.18 | **1.501x** | 1.489 1.511 1.503 | **1947 (15705 MB)**, 0 streamed |
| `ROCKET_MOE=0` + same args | 13.67 | **0.600x** | 0.596 0.602 0.603 | experts on the CPU |

**The `ub2048` and `moe1` arms are the same command line apart from `ROCKET_MOE=1`, and they are
1.197x apart.** The placement lines say why and they are perfectly reproducible: the default places
**1683** experts across 3 of 3 passes and the forced arm **1947** across 3 of 3, a difference of 264
expert stacks and 2130 MB.

**That is the flag working as designed, not a defect.** `ROCKET_MOE` has **three** states, not two:
unset is **AUTO** (claim an expert op only where the offload is measured to win and the pre-flight
can reserve the whole stack), `1` is **FORCED** (claim every `MUL_MAT_ID` the handler can compute,
including stacks AUTO declines, streaming what does not fit), and `0` is **OFF**. AUTO and FORCED
are different placements by construction.

**So "`ub2048` is the same configuration as `moe1` because placement is default-on" is wrong**, and
it is worth correcting wherever it is repeated: it would license dropping one of the four arms as
redundant, and the two arms it calls identical differ by 1.20x here. Placement being default-on
makes `stock` and `ub2048` differ only in `-ub` — which they do, both at 1683 experts — but it says
nothing about AUTO against FORCED.

**The forced-vs-auto gap reproduces its documented size.** `ggml-rocket.cpp` records FORCED at
**+18-21% at pp2048** on this model, measured twice; this is a third measurement at **+19.7%**
(1.501 / 1.254), now over three rotated passes rather than single processes.

**The mechanism control lands at 0.600x, not the ~0.41x the triage expected.** `ROCKET_MOE=0` puts
the experts back on the CPU and costs 40% of prefill, so AUTO's expert route is worth **1.667x**
over experts-on-CPU on this model. The triage's ~0.41x appears to have come from the **0.42x** in
`ggml-rocket.cpp`'s comment, which measures something else entirely — the abandoned fp16
*streaming* expert route against the CPU (5.33 against 12.56 t/s), not AUTO's resident-int8 route
against stock [inference from the two figures' agreement to 0.01, not established]. The handover's
own timing cell for this model implied ~0.55x, which is near this measurement and far from 0.41x,
so the two figures in that basis cell already disagreed.

## On a MoE model the `-ub` flag can be buying something else entirely

The four-arm design pays for itself on `deepseek-v2-lite`, because the arm that looks like the
`-ub` lever is mostly not [HW sweep 2026-08-29/30, RK1, 600 MHz, three rotated passes, all four
resolved, per-pass ratios agreeing to 0.4%]:

| arm | t/s | against stock | experts placed |
|---|---:|---:|---|
| stock (`-ub 512`) | 21.28 | — | **none — 0 of 3 passes place anything** |
| `-b 2048 -ub 2048` | 28.15 | **1.323x** | 4779 (14959 MB), 0 streamed |
| `ROCKET_MOE=1` | 28.35 | **1.332x** | 4779 (14959 MB), 0 streamed |
| `ROCKET_MOE=0` | 21.61 | **1.016x** | experts on the CPU |

**The stock arm never offloads a single expert**, on any of its three passes, because at `-ub 512`
this architecture's per-dispatch work sits below the offload floor. That makes the four arms
decompose cleanly, and the decomposition is the result:

| what | ratio | how |
|---|---:|---|
| the `-ub` lever proper (dequant amortized, experts on CPU both sides) | **1.016x** | `moe0` / `stock` |
| crossing the MoE offload floor (at `-ub 2048` both sides) | **1.303x** | `ub2048` / `moe0` |
| what the flag appears to buy | 1.323x | `ub2048` / `stock` |

**So on this model `-b 2048 -ub 2048` is worth 1.32x and essentially none of it is the
micro-batch dequant amortization the flag is documented for — 1.30 of it is that the larger
micro-batch crosses the expert-offload floor.** The dequant term, isolated, is **1.6%**.

**The contrast with `gpt-oss-20b` is what makes this readable rather than a curiosity.** That model
places 1683 experts at `-ub 512` already, so its stock arm is offloading and its 1.254x *is* the
dequant lever; its expert route, isolated the same way, is worth **2.09x** (`ub2048` / `moe0`).
Two MoE models, the same four arms, and the same flag buying two different things:

| model | `-ub` lever proper | expert route | offloads at `-ub 512`? |
|---|---:|---:|---|
| `gpt-oss-20b` | 1.254x | **2.09x** | yes, 1683 experts |
| `deepseek-v2-lite` | **1.016x** | 1.303x | **no** |

This reproduces, with placement counts rather than inference, the known statement that gpt-oss
offloads from `-ub 512` up while DeepSeek-V2-Lite needs ~1250+ tokens in a micro-batch.

**It also warns against a decomposition this matrix cannot do for the dense models.** Nine dense
units report one `-ub` number each with no mechanism control beside it, and nothing in those rows
says how much of each is dequant amortization versus some other threshold being crossed. The MoE
units have a control arm; the dense units do not, and the 0.94-1.53x range should be read as "what
the flag buys", not "what the dequant costs".

**`ROCKET_MOE=1` adds almost nothing here** — 1.332x against 1.323x, and identical placement (4779
experts, 14959 MB) on both arms. Unlike gpt-oss, where FORCED placed 264 more expert stacks and
bought 1.197x, this model's AUTO route already takes everything FORCED would, so the two states
coincide. **That is a per-model property and not a general one**: the same two arms differ by 20%
on one model and 0.7% on the other.

## The largest model in the matrix is the one where every tuned arm loses

`qwen3-30b-a3b` closes the matrix and inverts its headline. All four arms resolved, per-pass ratios
agreeing to 0.4%, and **all three tuned arms are below stock** [HW sweep 2026-08-30, RK1, 600 MHz,
three rotated passes]:

| arm | t/s | against stock | placed |
|---|---:|---:|---|
| stock | 14.74 | — | nothing |
| `-b 2048 -ub 2048` | 13.38 | **0.908x** | **nothing — AUTO declines** |
| `ROCKET_MOE=1` | 10.71 | **0.726x** | 9427-9437 resident, ~5845 streamed — **62%** |
| `ROCKET_MOE=0` | 13.39 | **0.908x** | experts on the CPU |

**`ub2048` and `moe0` agree to three digits (0.908x against 0.908x), and that agreement is the
finding**: on this model AUTO places nothing at all, on 3 of 3 passes, so the default arm and the
experts-on-CPU control are the *same configuration reached two ways*. 30.53 B with 29 of them
experts does not fit this board's budget, and the route declines rather than half-placing.

**So `-b 2048 -ub 2048` costs 9.2% here**, with experts on the CPU on both sides — the second model
in the matrix where the flag is a resolved loss, after `smolvlm2`'s 0.941x.

**`ROCKET_MOE=1` is the expensive mistake this model exists to demonstrate.** Forced, it places 62%
and streams the rest, and costs **0.800x against the experts-on-CPU baseline** (10.71 / 13.39) —
i.e. a user who reasons "more offload must be faster" and sets the flag pays 20% for it. That is
the loss AUTO's pre-flight exists to avoid, and this is the first time the whole chain has been
measured end to end on the model that triggers it: **AUTO declining is worth 1.25x over FORCED
here.** It is also a third measurement of that gap — the archived figure is 0.834x at 62.4%
residency against **0.800x at 62%** now, over rotated passes rather than three pairs.

**What this does to the size reading.** The 1.18-1.53x band for models above 8 B was drawn from five
**dense** models. This one is 30.53 B by parameter count and reads 0.908x, so **the band is a
dense-model band and total parameter count is the wrong axis for a MoE model** — only ~3 B are
active per token here, and the per-micro-batch dequant is over the active set. Whether the active
count is the right axis is **not established** [hypothesis]: it would put this model with the 3 B
dense group at 1.038-1.093x, and it reads below all of them. Read the MoE units from their own rows.

**The three MoE units do not share a story**, which is the last thing the four-arm design bought:

| model | `-ub` arm | AUTO places? | FORCED vs AUTO | expert route worth |
|---|---:|---|---:|---|
| `gpt-oss-20b` | 1.254x | yes, 1683 experts | **+19.7%** | **2.09x** |
| `deepseek-v2-lite` | 1.323x | only past the floor | +0.7% | 1.303x |
| `qwen3-30b-a3b` | **0.908x** | **no** | **-20.0%** | n/a — declined |

## Does a 12B+ fp16 resident fit beside its GGUF? Partly, and the reserve floor is what stops it

This was the matrix's one named open question, posed three ways — fits, refuses, or refuses for a
stated reason. The answer is **none of them: it fits partially, and the limit that stops it is the
runtime reserve floor rather than the resident-weight budget** [HW sweep 2026-08-29, RK1, 600 MHz,
three rotated passes, `gemma4-12b` = 11.91 B, 6.86 GiB `Q4_K_M`]:

```
[f16-resident] weights offered to the resident route: 239 resident on the NPU (15018MB),
               89 streamed via the per-call pack -- 73% resident
[f16-resident] admission first declined at 15018MB resident: MemAvailable 9460MB fell below
               the 9535MB reserve floor
```

**`gemma4-12b` is the first unit in this matrix that does not read "0 streamed".** Every unit above
it places 100% or nothing; this one places **73-74%** and streams the rest through the per-call
pack. So the 12B+ question has a third answer that the two-outcome framing did not have a slot for,
and partial residency is a real operating state rather than an edge case.

**The budget was never the binding constraint.** The pre-flight resolved a **21037-21087 MB**
resident-weight budget and the route stopped at **15018-15255 MB**, well under it. What stopped it
is the other check: as the route places cubes, `MemAvailable` falls, and admission halts when it
would cross the **9535 MB reserve floor**. The budget is computed once per process from the
`MemAvailable` at start; the floor is tested as placement proceeds. On a model this size the second
binds first, and only the outcome line distinguishes them.

**Placement is therefore not deterministic on this model.** The three passes reached **239 / 240 /
243** weights at **15018 / 15131 / 15255 MB** — the count tracks whatever `MemAvailable` happened to
be, so two runs of the same command place different tensor sets. Every unit above `gemma4-12b`
reports byte-identical placement across its passes; this is the first that does not, and a
before/after comparison on this model has to read the outcome line per arm rather than assume the
arms are comparable.

**73% residency still buys 1.322x**, against 1.325x for `ministral3-8b` at 100%. One point each, so
this does not measure how the ratio falls with residency — but it does say a partial placement is
worth having rather than something to refuse, which is the decision a user faces.

This is the same runtime-`MemAvailable` mechanism filed as an open safety item, observed
here doing its job: the floor held, nothing was over-committed, and no OOM killer fired. What that
item is about is the case where the memory disappears *after* the budget freezes, which this is not.

## The two residency routes place identically on the same model

`llama32-3b`'s `ROCKET_QUANT_RESIDENT=auto` arm reports **193 weights resident, 5232 MB, 0
streamed** on all three passes — the same count and the same footprint as `llama32-3b-f16`'s
`ROCKET_F16_RESIDENT=auto` arm above. The two routes reach it from different files (a 1.87 GiB
`Q4_K_M` GGUF dequantized once, against a 6.4 GiB F16 one packed directly) and converge on the
identical placement, which is the expected behaviour of a shared fp16 prepack and a cheap check
that the quant route is not silently placing a different tensor set.

The speeds do not converge, and should not: the same 193 resident weights run **47.21** t/s from
the quant GGUF against **60.54** from the F16 one. Residency removes the per-micro-batch dequant,
not the one-time decode, and the quant arm still carries what the F16 arm never pays.

## The passive predictors: what carries them and what does not

`llama32-3b` is the **first unit whose rows carry the `<!--PRED-->` predictor lines** — nine of
them, one per (pass, arm). The five units above it carry none. The predictors record
`MemAvailable`, `AnonHugePages`/`Hugepagesize` and the whole `/proc/buddyinfo` free-page vector,
read once at the start of each timed run, and they exist so the per-process spread can be attacked
from rows already taken rather than from a campaign of its own.

`anonhuge_kb=0` in every line here is a **positive statement, not a failed read**: this board's
7.2.0-1-arm64 kernel is built `# CONFIG_TRANSPARENT_HUGEPAGE is not set`, so there are no
transparent huge pages to record. A board or kernel with THP on makes the column live.

**So the retrospective analysis starts at unit 3, not at unit 1**, and the five earlier units are
not candidates for it. Their ratios are unaffected — the predictors are two file reads taken
between the discarded warm-up and the timed run, and they touch neither arm's workload — but a
correlation of the buddyinfo tail against t/s has six arms to work with rather than eighteen, and
the two units with the widest spread on record (unit 2, ~11%) are among those that carry nothing.

## Raw output

<!-- qwen35-08b  class=quant  fp16-resident-fits=1  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=/path/to/models/qwen35/Qwen3.5-0.8B-Q4_K_M.gguf (532517120 bytes)
     note: smallest; proves the harness before anything expensive
-->
== qwen35-08b  Fri Aug 28 19:38:33 UTC 2026 ==
### qwen35-08b [stock] pass 1  env=''  args=''  19:38:47  clk=600 MHz  MemAvail=31514320 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |          pp2048 |         98.26 ± 0.42 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	98.26-->

### qwen35-08b [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  19:40:26  clk=600 MHz  MemAvail=31493932 kB
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |     2048 |          pp2048 |        111.12 ± 0.81 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	111.12-->

### qwen35-08b [qresident] pass 1  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  19:41:55  clk=600 MHz  MemAvail=31326728 kB
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |     2048 |          pp2048 |        116.53 ± 0.26 |

build: 171974745 (10558)
<!--DATA 1	qresident	pp2048	116.53-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20956MB (MemAvailable 30491MB - reserve 9535MB, no swap)

### qwen35-08b [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  19:43:23  clk=600 MHz  MemAvail=31614560 kB
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |     2048 |          pp2048 |        111.98 ± 1.06 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	111.98-->

### qwen35-08b [qresident] pass 2  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  19:44:51  clk=600 MHz  MemAvail=31438240 kB
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |     2048 |          pp2048 |        116.40 ± 0.82 |

build: 171974745 (10558)
<!--DATA 2	qresident	pp2048	116.40-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21080MB (MemAvailable 30616MB - reserve 9535MB, no swap)

### qwen35-08b [stock] pass 2  env=''  args=''  19:46:20  clk=600 MHz  MemAvail=31626668 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |          pp2048 |         98.41 ± 0.84 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	98.41-->

### qwen35-08b [qresident] pass 3  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  19:47:59  clk=600 MHz  MemAvail=31599344 kB
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |     2048 |          pp2048 |        117.65 ± 0.42 |

build: 171974745 (10558)
<!--DATA 3	qresident	pp2048	117.65-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21243MB (MemAvailable 30778MB - reserve 9535MB, no swap)

### qwen35-08b [stock] pass 3  env=''  args=''  19:49:26  clk=600 MHz  MemAvail=31600948 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        100.59 ± 0.39 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	100.59-->

### qwen35-08b [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  19:51:03  clk=600 MHz  MemAvail=31593152 kB
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | ROCKET     |  -1 |     2048 |          pp2048 |        112.70 ± 0.72 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	112.70-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio |
|---|---|---:|---:|---:|---:|---:|
| pp2048 | stock | 99.09 | 3 | 98.26 | 100.59 | -- |
| pp2048 | ub2048 | 111.93 | 3 | 111.12 | 112.70 | 1.130x |
| pp2048 | qresident | 116.86 | 3 | 116.40 | 117.65 | 1.179x |


<!-- qwen35-08b-f16  class=f16  fp16-resident-fits=1  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=/path/to/models/qwen35/Qwen3.5-0.8B-F16.gguf (1516744736 bytes)
-->
== qwen35-08b-f16  Fri Aug 28 19:53:41 UTC 2026 ==
### qwen35-08b-f16 [f16-stock] pass 1  env=''  args=''  19:53:55  clk=600 MHz  MemAvail=31497524 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        117.02 ± 0.05 |

build: 171974745 (10558)
<!--DATA 1	f16-stock	pp2048	117.02-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16 [f16-res] pass 1  env='ROCKET_F16_RESIDENT=auto'  args=''  19:55:22  clk=600 MHz  MemAvail=31408136 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        129.79 ± 1.26 |

build: 171974745 (10558)
<!--DATA 1	f16-res	pp2048	129.79-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21033MB (MemAvailable 30569MB - reserve 9535MB, no swap)

### qwen35-08b-f16 [f16-res] pass 2  env='ROCKET_F16_RESIDENT=auto'  args=''  19:56:42  clk=600 MHz  MemAvail=31477020 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        117.61 ± 0.17 |

build: 171974745 (10558)
<!--DATA 2	f16-res	pp2048	117.61-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21118MB (MemAvailable 30654MB - reserve 9535MB, no swap)

### qwen35-08b-f16 [f16-stock] pass 2  env=''  args=''  19:58:08  clk=600 MHz  MemAvail=31359748 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        131.61 ± 0.58 |

build: 171974745 (10558)
<!--DATA 2	f16-stock	pp2048	131.61-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16 [f16-stock] pass 3  env=''  args=''  19:59:27  clk=600 MHz  MemAvail=31483724 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        116.91 ± 0.39 |

build: 171974745 (10558)
<!--DATA 3	f16-stock	pp2048	116.91-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16 [f16-res] pass 3  env='ROCKET_F16_RESIDENT=auto'  args=''  20:00:54  clk=600 MHz  MemAvail=31538096 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        123.76 ± 0.27 |

build: 171974745 (10558)
<!--DATA 3	f16-res	pp2048	123.76-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21201MB (MemAvailable 30736MB - reserve 9535MB, no swap)

### qwen35-08b-f16 [f16-res] pass 4  env='ROCKET_F16_RESIDENT=auto'  args=''  20:02:18  clk=600 MHz  MemAvail=31594212 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        126.94 ± 0.63 |

build: 171974745 (10558)
<!--DATA 4	f16-res	pp2048	126.94-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21240MB (MemAvailable 30776MB - reserve 9535MB, no swap)

### qwen35-08b-f16 [f16-stock] pass 4  env=''  args=''  20:03:39  clk=600 MHz  MemAvail=31467540 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        116.10 ± 0.46 |

build: 171974745 (10558)
<!--DATA 4	f16-stock	pp2048	116.10-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16 [f16-stock] pass 5  env=''  args=''  20:05:06  clk=600 MHz  MemAvail=31450468 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        116.20 ± 0.09 |

build: 171974745 (10558)
<!--DATA 5	f16-stock	pp2048	116.20-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16 [f16-res] pass 5  env='ROCKET_F16_RESIDENT=auto'  args=''  20:06:34  clk=600 MHz  MemAvail=31359824 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        117.04 ± 0.34 |

build: 171974745 (10558)
<!--DATA 5	f16-res	pp2048	117.04-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21086MB (MemAvailable 30622MB - reserve 9535MB, no swap)

### qwen35-08b-f16 [f16-res] pass 6  env='ROCKET_F16_RESIDENT=auto'  args=''  20:08:01  clk=600 MHz  MemAvail=31359284 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        124.12 ± 0.79 |

build: 171974745 (10558)
<!--DATA 6	f16-res	pp2048	124.12-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21084MB (MemAvailable 30619MB - reserve 9535MB, no swap)

### qwen35-08b-f16 [f16-stock] pass 6  env=''  args=''  20:09:24  clk=600 MHz  MemAvail=31329956 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        118.63 ± 0.30 |

build: 171974745 (10558)
<!--DATA 6	f16-stock	pp2048	118.63-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16 [f16-stock] pass 7  env=''  args=''  20:10:49  clk=600 MHz  MemAvail=31468724 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        124.09 ± 0.22 |

build: 171974745 (10558)
<!--DATA 7	f16-stock	pp2048	124.09-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (780MB), 0 streamed via the per-call pack -- 100% resident

### qwen35-08b-f16 [f16-res] pass 7  env='ROCKET_F16_RESIDENT=auto'  args=''  20:12:11  clk=600 MHz  MemAvail=31443964 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B F16                |   1.40 GiB |   752.39 M | ROCKET     |  -1 |          pp2048 |        117.20 ± 0.66 |

build: 171974745 (10558)
<!--DATA 7	f16-res	pp2048	117.20-->
    [f16-resident] weights offered to the resident route: 150 resident on the NPU (948MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21069MB (MemAvailable 30605MB - reserve 9535MB, no swap)

#### summary: per-arm mean over 7 passes, ratios paired within a pass against [f16-stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | f16-stock | 120.08 | 7 | 116.10 | 131.61 | -- | -- |
| pp2048 | f16-res | 122.35 | 7 | 117.04 | 129.79 | 1.022x | 1.109 0.894 1.059 1.093 1.007 1.046 0.944 |


<!-- llama32-3b-f16  class=f16  fp16-resident-fits=1  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=/path/to/models/llama32-3b/Llama-3.2-3B-Instruct-F16.gguf (6433687616 bytes)
-->
== llama32-3b-f16  Sat Aug 29 13:46:48 UTC 2026 ==
### llama32-3b-f16 [f16-stock] pass 1  env=''  args=''  13:47:13  clk=600 MHz  MemAvail=31499492 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B F16                   |   5.98 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         56.24 ± 0.04 |

build: 171974745 (10558)
<!--DATA 1	f16-stock	pp2048	56.24-->

### llama32-3b-f16 [f16-res] pass 1  env='ROCKET_F16_RESIDENT=auto'  args=''  13:50:13  clk=600 MHz  MemAvail=31414088 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B F16                   |   5.98 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         60.84 ± 0.37 |

build: 171974745 (10558)
<!--DATA 1	f16-res	pp2048	60.84-->
    [f16-resident] weights offered to the resident route: 193 resident on the NPU (5232MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21142MB (MemAvailable 30678MB - reserve 9535MB, no swap)

### llama32-3b-f16 [f16-res] pass 2  env='ROCKET_F16_RESIDENT=auto'  args=''  13:53:10  clk=600 MHz  MemAvail=31451812 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B F16                   |   5.98 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         60.37 ± 0.07 |

build: 171974745 (10558)
<!--DATA 2	f16-res	pp2048	60.37-->
    [f16-resident] weights offered to the resident route: 193 resident on the NPU (5232MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21117MB (MemAvailable 30652MB - reserve 9535MB, no swap)

### llama32-3b-f16 [f16-stock] pass 2  env=''  args=''  13:56:01  clk=600 MHz  MemAvail=31483860 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B F16                   |   5.98 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         56.26 ± 0.12 |

build: 171974745 (10558)
<!--DATA 2	f16-stock	pp2048	56.26-->

### llama32-3b-f16 [f16-stock] pass 3  env=''  args=''  13:58:55  clk=600 MHz  MemAvail=31530960 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B F16                   |   5.98 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         55.52 ± 0.05 |

build: 171974745 (10558)
<!--DATA 3	f16-stock	pp2048	55.52-->

### llama32-3b-f16 [f16-res] pass 3  env='ROCKET_F16_RESIDENT=auto'  args=''  14:01:57  clk=600 MHz  MemAvail=31471804 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B F16                   |   5.98 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         60.41 ± 0.05 |

build: 171974745 (10558)
<!--DATA 3	f16-res	pp2048	60.41-->
    [f16-resident] weights offered to the resident route: 193 resident on the NPU (5232MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21167MB (MemAvailable 30702MB - reserve 9535MB, no swap)

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [f16-stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | f16-stock | 56.01 | 3 | 55.52 | 56.26 | -- | -- |
| pp2048 | f16-res | 60.54 | 3 | 60.37 | 60.84 | 1.081x | 1.082 1.073 1.088 |


<!-- ministral3-3b-f16  class=f16  fp16-resident-fits=1  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=/path/to/models/ministral3-3b/Ministral-3-3B-Instruct-2512-F16.gguf (6866220032 bytes)
-->
== ministral3-3b-f16  Sat Aug 29 14:05:02 UTC 2026 ==
### ministral3-3b-f16 [f16-stock] pass 1  env=''  args=''  14:05:30  clk=600 MHz  MemAvail=31497496 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B F16                |   6.39 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         48.14 ± 0.21 |

build: 171974745 (10558)
<!--DATA 1	f16-stock	pp2048	48.14-->

### ministral3-3b-f16 [f16-res] pass 1  env='ROCKET_F16_RESIDENT=auto'  args=''  14:08:57  clk=600 MHz  MemAvail=31454248 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B F16                |   6.39 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         52.10 ± 0.19 |

build: 171974745 (10558)
<!--DATA 1	f16-res	pp2048	52.10-->
    [f16-resident] weights offered to the resident route: 179 resident on the NPU (5610MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21103MB (MemAvailable 30638MB - reserve 9535MB, no swap)

### ministral3-3b-f16 [f16-res] pass 2  env='ROCKET_F16_RESIDENT=auto'  args=''  14:12:19  clk=600 MHz  MemAvail=31428816 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B F16                |   6.39 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         52.05 ± 0.04 |

build: 171974745 (10558)
<!--DATA 2	f16-res	pp2048	52.05-->
    [f16-resident] weights offered to the resident route: 179 resident on the NPU (5610MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21079MB (MemAvailable 30614MB - reserve 9535MB, no swap)

### ministral3-3b-f16 [f16-stock] pass 2  env=''  args=''  14:15:35  clk=600 MHz  MemAvail=31511100 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B F16                |   6.39 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         48.45 ± 0.08 |

build: 171974745 (10558)
<!--DATA 2	f16-stock	pp2048	48.45-->

### ministral3-3b-f16 [f16-stock] pass 3  env=''  args=''  14:18:54  clk=600 MHz  MemAvail=31542752 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B F16                |   6.39 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         48.26 ± 0.08 |

build: 171974745 (10558)
<!--DATA 3	f16-stock	pp2048	48.26-->

### ministral3-3b-f16 [f16-res] pass 3  env='ROCKET_F16_RESIDENT=auto'  args=''  14:22:20  clk=600 MHz  MemAvail=31435396 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B F16                |   6.39 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         51.96 ± 0.13 |

build: 171974745 (10558)
<!--DATA 3	f16-res	pp2048	51.96-->
    [f16-resident] weights offered to the resident route: 179 resident on the NPU (5610MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21085MB (MemAvailable 30621MB - reserve 9535MB, no swap)

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [f16-stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | f16-stock | 48.28 | 3 | 48.14 | 48.45 | -- | -- |
| pp2048 | f16-res | 52.04 | 3 | 51.96 | 52.10 | 1.078x | 1.082 1.074 1.077 |


<!-- phi4mini-f16  class=f16  fp16-resident-fits=1  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=/path/to/models/phi4mini/Phi-4-mini-instruct-F16.gguf (7680694240 bytes)
-->
== phi4mini-f16  Sat Aug 29 14:25:51 UTC 2026 ==
### phi4mini-f16 [f16-stock] pass 1  env=''  args=''  14:26:20  clk=600 MHz  MemAvail=31497408 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B F16                    |   7.15 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         50.13 ± 0.05 |

build: 171974745 (10558)
<!--DATA 1	f16-stock	pp2048	50.13-->

### phi4mini-f16 [f16-res] pass 1  env='ROCKET_F16_RESIDENT=auto'  args=''  14:29:43  clk=600 MHz  MemAvail=31440536 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B F16                    |   7.15 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         54.71 ± 0.16 |

build: 171974745 (10558)
<!--DATA 1	f16-res	pp2048	54.71-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (6000MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21166MB (MemAvailable 30702MB - reserve 9535MB, no swap)

### phi4mini-f16 [f16-res] pass 2  env='ROCKET_F16_RESIDENT=auto'  args=''  14:33:00  clk=600 MHz  MemAvail=31393112 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B F16                    |   7.15 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         54.19 ± 0.11 |

build: 171974745 (10558)
<!--DATA 2	f16-res	pp2048	54.19-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (6000MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21106MB (MemAvailable 30642MB - reserve 9535MB, no swap)

### phi4mini-f16 [f16-stock] pass 2  env=''  args=''  14:36:12  clk=600 MHz  MemAvail=31470604 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B F16                    |   7.15 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         50.06 ± 0.10 |

build: 171974745 (10558)
<!--DATA 2	f16-stock	pp2048	50.06-->

### phi4mini-f16 [f16-stock] pass 3  env=''  args=''  14:39:28  clk=600 MHz  MemAvail=31492892 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B F16                    |   7.15 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         50.37 ± 0.19 |

build: 171974745 (10558)
<!--DATA 3	f16-stock	pp2048	50.37-->

### phi4mini-f16 [f16-res] pass 3  env='ROCKET_F16_RESIDENT=auto'  args=''  14:42:49  clk=600 MHz  MemAvail=31447296 kB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B F16                    |   7.15 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         54.30 ± 0.03 |

build: 171974745 (10558)
<!--DATA 3	f16-res	pp2048	54.30-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (6000MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21096MB (MemAvailable 30632MB - reserve 9535MB, no swap)

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [f16-stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | f16-stock | 50.19 | 3 | 50.06 | 50.37 | -- | -- |
| pp2048 | f16-res | 54.40 | 3 | 54.19 | 54.71 | 1.084x | 1.091 1.083 1.078 |


<!-- llama32-3b  class=quant  fp16-resident-fits=1  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=/path/to/models/llama32-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf (2019377600 bytes)
-->
== llama32-3b  Sat Aug 29 15:36:39 UTC 2026 ==
### llama32-3b [stock] pass 1  env=''  args=''  15:37:06  clk=600 MHz  MemAvail=31498496 kB
<!--PRED 1	stock	memavail_kb=31498496	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6549,6470,7789,5451,4285,4059,3599,3081,2390,2161,4690-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         39.24 ± 0.20 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	39.24-->

### llama32-3b [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  15:41:04  clk=600 MHz  MemAvail=31515788 kB
<!--PRED 1	ub2048	memavail_kb=31515788	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6182,6255,6298,5975,4494,3883,3456,2957,2299,1857,4899-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |     2048 |          pp2048 |         42.23 ± 0.16 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	42.23-->

### llama32-3b [qresident] pass 1  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  15:44:51  clk=600 MHz  MemAvail=31263480 kB
<!--PRED 1	qresident	memavail_kb=31263732	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5131,6165,6424,6286,4927,4266,3723,3169,2317,1579,4907-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |     2048 |          pp2048 |         47.13 ± 0.09 |

build: 171974745 (10558)
<!--DATA 1	qresident	pp2048	47.13-->
    [f16-resident] weights offered to the resident route: 193 resident on the NPU (5232MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20963MB (MemAvailable 30499MB - reserve 9535MB, no swap)

### llama32-3b [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  15:48:26  clk=600 MHz  MemAvail=31487532 kB
<!--PRED 2	ub2048	memavail_kb=31487532	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5186,6999,6483,6355,4599,4306,3785,3171,2344,1755,4865-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |     2048 |          pp2048 |         42.47 ± 0.04 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	42.47-->

### llama32-3b [qresident] pass 2  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  15:52:12  clk=600 MHz  MemAvail=31356300 kB
<!--PRED 2	qresident	memavail_kb=31356416	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5909,6640,6365,6630,5067,4436,3920,3300,2321,1529,4914-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |     2048 |          pp2048 |         47.45 ± 0.09 |

build: 171974745 (10558)
<!--DATA 2	qresident	pp2048	47.45-->
    [f16-resident] weights offered to the resident route: 193 resident on the NPU (5232MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21031MB (MemAvailable 30567MB - reserve 9535MB, no swap)

### llama32-3b [stock] pass 2  env=''  args=''  15:55:46  clk=600 MHz  MemAvail=31477252 kB
<!--PRED 2	stock	memavail_kb=31477252	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4408,6494,6500,6009,4777,4299,3746,3155,2303,1723,4895-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         38.68 ± 0.03 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	38.68-->

### llama32-3b [qresident] pass 3  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  15:59:50  clk=600 MHz  MemAvail=31239300 kB
<!--PRED 3	qresident	memavail_kb=31239300	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5229,6018,6221,5606,4845,4394,3886,3262,2305,1476,4938-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |     2048 |          pp2048 |         47.05 ± 0.04 |

build: 171974745 (10558)
<!--DATA 3	qresident	pp2048	47.05-->
    [f16-resident] weights offered to the resident route: 193 resident on the NPU (5232MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20967MB (MemAvailable 30503MB - reserve 9535MB, no swap)

### llama32-3b [stock] pass 3  env=''  args=''  16:03:26  clk=600 MHz  MemAvail=31517624 kB
<!--PRED 3	stock	memavail_kb=31517624	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6014,7223,6651,6623,5037,4488,3963,3303,2345,1665,4873-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |          pp2048 |         38.95 ± 0.11 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	38.95-->

### llama32-3b [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  16:07:25  clk=600 MHz  MemAvail=31486648 kB
<!--PRED 3	ub2048	memavail_kb=31486648	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5804,6603,6120,6139,4467,4063,3603,3025,2246,1705,4957-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ROCKET     |  -1 |     2048 |          pp2048 |         42.51 ± 0.11 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	42.51-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 38.96 | 3 | 38.68 | 39.24 | -- | -- |
| pp2048 | ub2048 | 42.40 | 3 | 42.23 | 42.51 | 1.089x | 1.076 1.098 1.091 |
| pp2048 | qresident | 47.21 | 3 | 47.05 | 47.45 | 1.212x | 1.201 1.227 1.208 |


<!-- ministral3-3b  class=quant  fp16-resident-fits=1  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=/path/to/models/ministral3-3b/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf (2146497824 bytes)
-->
== ministral3-3b  Sat Aug 29 16:11:16 UTC 2026 ==
### ministral3-3b [stock] pass 1  env=''  args=''  16:11:45  clk=600 MHz  MemAvail=31518384 kB
<!--PRED 1	stock	memavail_kb=31518488	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5491,6392,6187,5844,4373,3694,3347,2882,2198,1689,5003-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         35.04 ± 0.03 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	35.04-->

### ministral3-3b [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  16:16:10  clk=600 MHz  MemAvail=31515860 kB
<!--PRED 1	ub2048	memavail_kb=31515860	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5348,6726,6313,5847,4222,3575,3277,2832,2159,1703,5021-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |     2048 |          pp2048 |         36.20 ± 0.04 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	36.20-->

### ministral3-3b [qresident] pass 1  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  16:20:31  clk=600 MHz  MemAvail=31148484 kB
<!--PRED 1	qresident	memavail_kb=31148484	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5178,6909,6700,6229,4896,4104,3636,3069,2283,1485,4926-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |     2048 |          pp2048 |         39.29 ± 0.08 |

build: 171974745 (10558)
<!--DATA 1	qresident	pp2048	39.29-->
    [f16-resident] weights offered to the resident route: 179 resident on the NPU (5610MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20825MB (MemAvailable 30360MB - reserve 9535MB, no swap)

### ministral3-3b [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  16:24:45  clk=600 MHz  MemAvail=31506436 kB
<!--PRED 2	ub2048	memavail_kb=31506436	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6541,6948,6651,6318,4781,4190,3721,3108,2308,1738,4867-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |     2048 |          pp2048 |         36.26 ± 0.19 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	36.26-->

### ministral3-3b [qresident] pass 2  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  16:29:07  clk=600 MHz  MemAvail=31311388 kB
<!--PRED 2	qresident	memavail_kb=31311388	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6055,6616,6148,5241,4544,4063,3629,3091,2313,1564,4932-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |     2048 |          pp2048 |         39.50 ± 0.06 |

build: 171974745 (10558)
<!--DATA 2	qresident	pp2048	39.50-->
    [f16-resident] weights offered to the resident route: 179 resident on the NPU (5610MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20976MB (MemAvailable 30511MB - reserve 9535MB, no swap)

### ministral3-3b [stock] pass 2  env=''  args=''  16:33:19  clk=600 MHz  MemAvail=31502040 kB
<!--PRED 2	stock	memavail_kb=31502040	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5301,6739,6675,6032,4752,4066,3613,3098,2329,1717,4888-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         34.70 ± 0.09 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	34.70-->

### ministral3-3b [qresident] pass 3  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  16:37:49  clk=600 MHz  MemAvail=31205724 kB
<!--PRED 3	qresident	memavail_kb=31205724	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5822,5982,6069,5497,4826,4238,3763,3138,2284,1478,4933-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |     2048 |          pp2048 |         39.45 ± 0.14 |

build: 171974745 (10558)
<!--DATA 3	qresident	pp2048	39.45-->
    [f16-resident] weights offered to the resident route: 179 resident on the NPU (5610MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20932MB (MemAvailable 30467MB - reserve 9535MB, no swap)

### ministral3-3b [stock] pass 3  env=''  args=''  16:42:02  clk=600 MHz  MemAvail=31517824 kB
<!--PRED 3	stock	memavail_kb=31517824	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5064,6708,6516,6307,4936,4083,3699,3114,2330,1754,4861-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |          pp2048 |         34.71 ± 0.10 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	34.71-->

### ministral3-3b [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  16:46:29  clk=600 MHz  MemAvail=31531864 kB
<!--PRED 3	ub2048	memavail_kb=31531864	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6102,6412,6167,5934,4710,3693,3385,2907,2228,1712,4976-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 3B Q4_K - Medium      |   1.99 GiB |     3.43 B | ROCKET     |  -1 |     2048 |          pp2048 |         35.99 ± 0.04 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	35.99-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 34.82 | 3 | 34.70 | 35.04 | -- | -- |
| pp2048 | ub2048 | 36.15 | 3 | 35.99 | 36.26 | 1.038x | 1.033 1.045 1.037 |
| pp2048 | qresident | 39.41 | 3 | 39.29 | 39.50 | 1.132x | 1.121 1.138 1.137 |


<!-- phi4mini  class=quant  fp16-resident-fits=1  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=/path/to/models/phi4mini/Phi-4-mini-instruct-Q4_K_M.gguf (2491874272 bytes)
-->
== phi4mini  Sat Aug 29 16:50:37 UTC 2026 ==
### phi4mini [stock] pass 1  env=''  args=''  16:51:07  clk=600 MHz  MemAvail=31509744 kB
<!--PRED 1	stock	memavail_kb=31510000	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6714,6685,6407,5659,4468,3589,3248,2846,2197,1567,4991-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         35.96 ± 0.10 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	35.96-->

### phi4mini [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  16:55:25  clk=600 MHz  MemAvail=31506508 kB
<!--PRED 1	ub2048	memavail_kb=31506472	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5384,6663,6197,5758,4416,3530,3248,2830,2188,1547,5009-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |     2048 |          pp2048 |         39.65 ± 0.05 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	39.65-->

### phi4mini [qresident] pass 1  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  16:59:29  clk=600 MHz  MemAvail=31147616 kB
<!--PRED 1	qresident	memavail_kb=31147616	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6165,5785,5762,5177,4532,3973,3540,3070,2293,1390,4917-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |     2048 |          pp2048 |         43.85 ± 0.04 |

build: 171974745 (10558)
<!--DATA 1	qresident	pp2048	43.85-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (6000MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20885MB (MemAvailable 30421MB - reserve 9535MB, no swap)

### phi4mini [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  17:03:22  clk=600 MHz  MemAvail=31502064 kB
<!--PRED 2	ub2048	memavail_kb=31502064	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5665,6406,6331,6099,4709,3848,3487,2974,2275,1625,4897-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |     2048 |          pp2048 |         39.01 ± 0.04 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	39.01-->

### phi4mini [qresident] pass 2  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  17:07:28  clk=600 MHz  MemAvail=31011632 kB
<!--PRED 2	qresident	memavail_kb=31011632	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6213,5975,5825,5242,4611,4039,3621,3117,2357,1417,4839-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |     2048 |          pp2048 |         43.52 ± 0.05 |

build: 171974745 (10558)
<!--DATA 2	qresident	pp2048	43.52-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (6000MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20746MB (MemAvailable 30281MB - reserve 9535MB, no swap)

### phi4mini [stock] pass 2  env=''  args=''  17:11:23  clk=600 MHz  MemAvail=31520684 kB
<!--PRED 2	stock	memavail_kb=31520684	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6089,5760,6891,6158,4919,4195,3729,3158,2339,1584,4852-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         36.07 ± 0.02 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	36.07-->

### phi4mini [qresident] pass 3  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  17:15:46  clk=600 MHz  MemAvail=30790512 kB
<!--PRED 3	qresident	memavail_kb=30790512	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6826,6428,6205,5523,4823,4261,3846,3247,2371,1410,4739-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |     2048 |          pp2048 |         43.54 ± 0.06 |

build: 171974745 (10558)
<!--DATA 3	qresident	pp2048	43.54-->
    [f16-resident] weights offered to the resident route: 126 resident on the NPU (6000MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20525MB (MemAvailable 30061MB - reserve 9535MB, no swap)

### phi4mini [stock] pass 3  env=''  args=''  17:19:40  clk=600 MHz  MemAvail=31523868 kB
<!--PRED 3	stock	memavail_kb=31523868	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5792,6703,6697,5958,4724,4103,3631,3095,2342,1599,4865-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |          pp2048 |         36.04 ± 0.12 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	36.04-->

### phi4mini [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  17:23:58  clk=600 MHz  MemAvail=31494720 kB
<!--PRED 3	ub2048	memavail_kb=31495196	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5240,6573,5853,5703,4823,3842,3491,2995,2271,1544,4937-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | ROCKET     |  -1 |     2048 |          pp2048 |         39.48 ± 0.05 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	39.48-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 36.02 | 3 | 35.96 | 36.07 | -- | -- |
| pp2048 | ub2048 | 39.38 | 3 | 39.01 | 39.65 | 1.093x | 1.103 1.082 1.095 |
| pp2048 | qresident | 43.64 | 3 | 43.52 | 43.85 | 1.211x | 1.219 1.207 1.208 |


<!-- smolvlm2  class=quant  fp16-resident-fits=1  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=/path/to/models/smolvlm2/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf (1112602656 bytes)
     note: text rows only; the vision path is a separate item
-->
== smolvlm2  Sat Aug 29 17:28:01 UTC 2026 ==
### smolvlm2 [stock] pass 1  env=''  args=''  17:28:17  clk=600 MHz  MemAvail=31472200 kB
<!--PRED 1	stock	memavail_kb=31472276	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4362,6225,5712,4877,4040,3748,3343,2888,2718,1994,4970-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |          pp2048 |         51.98 ± 0.10 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	51.98-->

### smolvlm2 [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  17:31:11  clk=600 MHz  MemAvail=31425144 kB
<!--PRED 1	ub2048	memavail_kb=31425144	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4584,4249,5291,4698,3603,3669,3262,2815,2544,1997,5031-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |     2048 |          pp2048 |         48.41 ± 0.06 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	48.41-->

### smolvlm2 [qresident] pass 1  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  17:34:20  clk=600 MHz  MemAvail=31353080 kB
<!--PRED 1	qresident	memavail_kb=31353080	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4769,5800,5857,5329,4456,4074,3553,2869,2186,1994,5043-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |     2048 |          pp2048 |         51.91 ± 0.03 |

build: 171974745 (10558)
<!--DATA 1	qresident	pp2048	51.91-->
    [f16-resident] weights offered to the resident route: 165 resident on the NPU (2976MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21083MB (MemAvailable 30618MB - reserve 9535MB, no swap)

### smolvlm2 [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  17:37:22  clk=600 MHz  MemAvail=31430180 kB
<!--PRED 2	ub2048	memavail_kb=31430180	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5796,6624,6640,3013,4566,4168,3593,2897,2456,1924,5031-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |     2048 |          pp2048 |         49.09 ± 0.09 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	49.09-->

### smolvlm2 [qresident] pass 2  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  17:40:29  clk=600 MHz  MemAvail=31281016 kB
<!--PRED 2	qresident	memavail_kb=31281024	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5225,6237,6130,5368,4621,4105,3547,2852,2193,1951,5041-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |     2048 |          pp2048 |         52.14 ± 0.05 |

build: 171974745 (10558)
<!--DATA 2	qresident	pp2048	52.14-->
    [f16-resident] weights offered to the resident route: 165 resident on the NPU (2976MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21011MB (MemAvailable 30547MB - reserve 9535MB, no swap)

### smolvlm2 [stock] pass 2  env=''  args=''  17:43:31  clk=600 MHz  MemAvail=31468384 kB
<!--PRED 2	stock	memavail_kb=31468384	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5357,5838,5504,4983,4169,3975,3443,2883,2485,1893,5064-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |          pp2048 |         52.34 ± 0.25 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	52.34-->

### smolvlm2 [qresident] pass 3  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  17:46:27  clk=600 MHz  MemAvail=31243132 kB
<!--PRED 3	qresident	memavail_kb=31243276	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5388,5806,5961,5453,4477,4228,3607,2870,2068,1912,5076-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |     2048 |          pp2048 |         52.06 ± 0.05 |

build: 171974745 (10558)
<!--DATA 3	qresident	pp2048	52.06-->
    [f16-resident] weights offered to the resident route: 165 resident on the NPU (2976MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20972MB (MemAvailable 30507MB - reserve 9535MB, no swap)

### smolvlm2 [stock] pass 3  env=''  args=''  17:49:29  clk=600 MHz  MemAvail=31468072 kB
<!--PRED 3	stock	memavail_kb=31468364	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5332,5898,6547,5238,4264,4136,3551,2870,2395,1908,5060-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |          pp2048 |         51.56 ± 0.25 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	51.56-->

### smolvlm2 [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  17:52:23  clk=600 MHz  MemAvail=31423616 kB
<!--PRED 3	ub2048	memavail_kb=31423616	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4434,6400,5919,3139,3649,3861,3378,2832,2408,1939,5083-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama ?B Q4_K - Medium         |   1.03 GiB |     1.81 B | ROCKET     |  -1 |     2048 |          pp2048 |         49.13 ± 0.05 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	49.13-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 51.96 | 3 | 51.56 | 52.34 | -- | -- |
| pp2048 | ub2048 | 48.88 | 3 | 48.41 | 49.13 | 0.941x | 0.931 0.938 0.953 |
| pp2048 | qresident | 52.04 | 3 | 51.91 | 52.14 | 1.002x | 0.999 0.996 1.010 |


<!-- qwen35-9b  class=quant  fp16-resident-fits=1  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=/path/to/models/qwen35/Qwen3.5-9B-Q4_K_M.gguf (5680522464 bytes)
     note: 5.29+16.69 GiB resident: fits, but not with room
-->
== qwen35-9b  Sat Aug 29 17:55:44 UTC 2026 ==
### qwen35-9b [stock] pass 1  env=''  args=''  17:56:49  clk=600 MHz  MemAvail=31488252 kB
<!--PRED 1	stock	memavail_kb=31488372	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5157,6404,5743,5548,4672,3591,3242,2781,2064,1296,4404-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         19.30 ± 0.20 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	19.30-->

### qwen35-9b [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  18:05:01  clk=600 MHz  MemAvail=31490004 kB
<!--PRED 1	ub2048	memavail_kb=31490004	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4815,6619,5758,5341,4787,3574,3211,2802,2079,1294,4402-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         27.17 ± 0.08 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	27.17-->

### qwen35-9b [qresident] pass 1  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  18:11:21  clk=600 MHz  MemAvail=31393352 kB
<!--PRED 1	qresident	memavail_kb=31393556	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7235,7225,6712,6093,4965,4133,3656,3151,2429,1583,4041-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         31.64 ± 0.29 |

build: 171974745 (10558)
<!--DATA 1	qresident	pp2048	31.64-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21083MB (MemAvailable 30618MB - reserve 9535MB, no swap)

### qwen35-9b [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  18:17:21  clk=600 MHz  MemAvail=31524168 kB
<!--PRED 2	ub2048	memavail_kb=31524168	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5046,6884,6483,5940,5365,3916,3541,3070,2352,1532,4141-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         27.38 ± 0.15 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	27.38-->

### qwen35-9b [qresident] pass 2  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  18:23:40  clk=600 MHz  MemAvail=31279260 kB
<!--PRED 2	qresident	memavail_kb=31279260	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6416,6611,6299,5487,4928,4254,3742,3316,2583,1706,3892-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         31.85 ± 0.05 |

build: 171974745 (10558)
<!--DATA 2	qresident	pp2048	31.85-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20999MB (MemAvailable 30534MB - reserve 9535MB, no swap)

### qwen35-9b [stock] pass 2  env=''  args=''  18:29:39  clk=600 MHz  MemAvail=31483056 kB
<!--PRED 2	stock	memavail_kb=31483284	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5871,6304,6507,6131,5496,4347,3912,3411,2633,1731,3878-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         19.17 ± 0.09 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	19.17-->

### qwen35-9b [qresident] pass 3  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  18:38:04  clk=600 MHz  MemAvail=31253896 kB
<!--PRED 3	qresident	memavail_kb=31253896	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5788,7442,6785,6399,5571,4358,3963,3432,2680,1756,3785-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         31.86 ± 0.08 |

build: 171974745 (10558)
<!--DATA 3	qresident	pp2048	31.86-->
    [f16-resident] weights offered to the resident route: 200 resident on the NPU (13184MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20906MB (MemAvailable 30442MB - reserve 9535MB, no swap)

### qwen35-9b [stock] pass 3  env=''  args=''  18:44:03  clk=600 MHz  MemAvail=31520580 kB
<!--PRED 3	stock	memavail_kb=31520580	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6454,6720,6488,5951,5130,4037,3672,3213,2470,1637,4030-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |          pp2048 |         18.95 ± 0.08 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	18.95-->

### qwen35-9b [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  18:52:22  clk=600 MHz  MemAvail=31524236 kB
<!--PRED 3	ub2048	memavail_kb=31524236	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6397,6616,6128,5736,5118,3751,3472,3050,2357,1528,4159-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | ROCKET     |  -1 |     2048 |          pp2048 |         27.23 ± 0.11 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	27.23-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 19.14 | 3 | 18.95 | 19.30 | -- | -- |
| pp2048 | ub2048 | 27.26 | 3 | 27.17 | 27.38 | 1.424x | 1.408 1.428 1.437 |
| pp2048 | qresident | 31.78 | 3 | 31.64 | 31.86 | 1.661x | 1.639 1.661 1.681 |


<!-- ministral3-8b  class=quant  fp16-resident-fits=1  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=/path/to/models/ministral3/Ministral-3-8B-Instruct-2512-Q4_K_M.gguf (5198386720 bytes)
-->
== ministral3-8b  Sat Aug 29 18:58:19 UTC 2026 ==
### ministral3-8b [stock] pass 1  env=''  args=''  18:59:19  clk=600 MHz  MemAvail=31532748 kB
<!--PRED 1	stock	memavail_kb=31532748	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4729,6645,6116,5843,4471,3708,3380,2940,2276,1464,4360-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | ROCKET     |  -1 |          pp2048 |         17.58 ± 0.05 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	17.58-->

### ministral3-8b [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  19:08:06  clk=600 MHz  MemAvail=31503780 kB
<!--PRED 1	ub2048	memavail_kb=31504036	anonhuge_kb=0	hugepagesz_kb=2048	buddy=4430,6385,6012,5704,4506,3523,3283,2873,2212,1408,4419-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | ROCKET     |  -1 |     2048 |          pp2048 |         20.94 ± 0.03 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	20.94-->

### ministral3-8b [qresident] pass 1  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  19:15:52  clk=600 MHz  MemAvail=30971832 kB
<!--PRED 1	qresident	memavail_kb=30971832	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6094,6001,5729,5354,4759,4213,3820,3348,2599,1687,3938-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | ROCKET     |  -1 |     2048 |          pp2048 |         23.58 ± 0.05 |

build: 171974745 (10558)
<!--DATA 1	qresident	pp2048	23.58-->
    [f16-resident] weights offered to the resident route: 235 resident on the NPU (13808MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20664MB (MemAvailable 30199MB - reserve 9535MB, no swap)

### ministral3-8b [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  19:23:17  clk=600 MHz  MemAvail=31530088 kB
<!--PRED 2	ub2048	memavail_kb=31530088	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6346,6920,6661,6392,5168,4327,3938,3456,2718,1747,3970-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | ROCKET     |  -1 |     2048 |          pp2048 |         20.87 ± 0.04 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	20.87-->

### ministral3-8b [qresident] pass 2  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  19:31:05  clk=600 MHz  MemAvail=31435036 kB
<!--PRED 2	qresident	memavail_kb=31435384	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6640,6326,6413,6345,5260,4491,4104,3603,2814,1835,3845-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | ROCKET     |  -1 |     2048 |          pp2048 |         23.60 ± 0.05 |

build: 171974745 (10558)
<!--DATA 2	qresident	pp2048	23.60-->
    [f16-resident] weights offered to the resident route: 235 resident on the NPU (13808MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21130MB (MemAvailable 30665MB - reserve 9535MB, no swap)

### ministral3-8b [stock] pass 2  env=''  args=''  19:38:30  clk=600 MHz  MemAvail=31520712 kB
<!--PRED 2	stock	memavail_kb=31520712	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6017,7287,6814,6596,5137,4414,4036,3542,2767,1807,3903-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | ROCKET     |  -1 |          pp2048 |         17.65 ± 0.07 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	17.65-->

### ministral3-8b [qresident] pass 3  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  19:47:27  clk=600 MHz  MemAvail=31222328 kB
<!--PRED 3	qresident	memavail_kb=31222328	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7400,7128,6600,6032,5022,4504,4112,3587,2813,1860,3785-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | ROCKET     |  -1 |     2048 |          pp2048 |         23.50 ± 0.03 |

build: 171974745 (10558)
<!--DATA 3	qresident	pp2048	23.50-->
    [f16-resident] weights offered to the resident route: 235 resident on the NPU (13808MB), 0 streamed via the per-call pack -- 100% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 20892MB (MemAvailable 30428MB - reserve 9535MB, no swap)

### ministral3-8b [stock] pass 3  env=''  args=''  19:54:54  clk=600 MHz  MemAvail=31506764 kB
<!--PRED 3	stock	memavail_kb=31506916	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6885,6625,6681,6387,4801,4181,3866,3386,2660,1738,4006-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | ROCKET     |  -1 |          pp2048 |         18.14 ± 0.08 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	18.14-->

### ministral3-8b [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  20:03:28  clk=600 MHz  MemAvail=31526996 kB
<!--PRED 3	ub2048	memavail_kb=31527776	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6772,6730,6230,5965,4698,3784,3518,3064,2425,1555,4243-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| mistral3 8B Q4_K - Medium      |   4.83 GiB |     8.49 B | ROCKET     |  -1 |     2048 |          pp2048 |         20.95 ± 0.01 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	20.95-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 17.79 | 3 | 17.58 | 18.14 | -- | -- |
| pp2048 | ub2048 | 20.92 | 3 | 20.87 | 20.95 | 1.176x | 1.191 1.182 1.155 |
| pp2048 | qresident | 23.56 | 3 | 23.50 | 23.60 | 1.325x | 1.341 1.337 1.295 |


<!-- gemma4-12b  class=quant  fp16-resident-fits=1  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=/path/to/models/gemma4/gemma-4-12b-it-Q4_K_M.gguf (7381382304 bytes)
     note: THE 12B+ QUESTION: 6.87+22.20 GiB. May refuse; the refusal is the datum
-->
== gemma4-12b  Sat Aug 29 20:10:26 UTC 2026 ==
### gemma4-12b [stock] pass 1  env=''  args=''  20:11:56  clk=600 MHz  MemAvail=31526540 kB
<!--PRED 1	stock	memavail_kb=31526540	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6534,5919,6147,5444,5189,3985,3349,2960,2317,1472,3805-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         12.77 ± 0.02 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	12.77-->

### gemma4-12b [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  20:24:11  clk=600 MHz  MemAvail=31522848 kB
<!--PRED 1	ub2048	memavail_kb=31522784	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6731,6196,6291,5510,5165,3823,3129,2842,2216,1409,3894-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         16.20 ± 0.02 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	16.20-->

### gemma4-12b [qresident] pass 1  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  20:34:34  clk=600 MHz  MemAvail=31411076 kB
<!--PRED 1	qresident	memavail_kb=31411616	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6484,6404,6125,5483,5146,3957,3315,2887,2241,1453,3818-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         17.08 ± 0.02 |

build: 171974745 (10558)
<!--DATA 1	qresident	pp2048	17.08-->
    [f16-resident] admission first declined at 15255MB resident: MemAvailable 9508MB fell below the 9535MB reserve floor
    [f16-resident] weights offered to the resident route: 243 resident on the NPU (15255MB), 85 streamed via the per-call pack -- 74% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21037MB (MemAvailable 30572MB - reserve 9535MB, no swap)

### gemma4-12b [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  20:44:49  clk=600 MHz  MemAvail=31478072 kB
<!--PRED 2	ub2048	memavail_kb=31478200	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5413,4984,5771,5486,5142,4140,3314,2938,2302,1428,3824-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         16.12 ± 0.02 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	16.12-->

### gemma4-12b [qresident] pass 2  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  20:55:14  clk=600 MHz  MemAvail=31426264 kB
<!--PRED 2	qresident	memavail_kb=31426264	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5711,5772,6010,5659,5287,4211,3396,2972,2339,1424,3786-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         16.94 ± 0.19 |

build: 171974745 (10558)
<!--DATA 2	qresident	pp2048	16.94-->
    [f16-resident] admission first declined at 15018MB resident: MemAvailable 9460MB fell below the 9535MB reserve floor
    [f16-resident] weights offered to the resident route: 239 resident on the NPU (15018MB), 89 streamed via the per-call pack -- 73% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21052MB (MemAvailable 30587MB - reserve 9535MB, no swap)

### gemma4-12b [stock] pass 2  env=''  args=''  21:05:31  clk=600 MHz  MemAvail=31523416 kB
<!--PRED 2	stock	memavail_kb=31523280	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6157,6181,6374,5801,5201,3812,3156,2789,2193,1394,3909-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         12.88 ± 0.01 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	12.88-->

### gemma4-12b [qresident] pass 3  env='ROCKET_QUANT_RESIDENT=auto'  args='-b 2048 -ub 2048'  21:18:02  clk=600 MHz  MemAvail=31487684 kB
<!--PRED 3	qresident	memavail_kb=31487684	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7621,6176,6461,5826,5386,3964,3204,2849,2254,1383,3870-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         17.00 ± 0.02 |

build: 171974745 (10558)
<!--DATA 3	qresident	pp2048	17.00-->
    [f16-resident] admission first declined at 15131MB resident: MemAvailable 9514MB fell below the 9535MB reserve floor
    [f16-resident] weights offered to the resident route: 240 resident on the NPU (15131MB), 88 streamed via the per-call pack -- 73% resident
    [rocket] ROCKET_QUANT_RESIDENT=auto -> resident budget 21087MB (MemAvailable 30623MB - reserve 9535MB, no swap)

### gemma4-12b [stock] pass 3  env=''  args=''  21:28:19  clk=600 MHz  MemAvail=31524376 kB
<!--PRED 3	stock	memavail_kb=31524376	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6009,6040,6236,5652,5276,3839,3154,2813,2218,1404,3895-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         12.95 ± 0.05 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	12.95-->

### gemma4-12b [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  21:40:26  clk=600 MHz  MemAvail=31515916 kB
<!--PRED 3	ub2048	memavail_kb=31515916	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5744,6383,6100,5370,5082,4036,3077,2788,2197,1382,3917-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gemma4 ?B Q4_K - Medium        |   6.86 GiB |    11.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         16.20 ± 0.03 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	16.20-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 12.87 | 3 | 12.77 | 12.95 | -- | -- |
| pp2048 | ub2048 | 16.17 | 3 | 16.12 | 16.20 | 1.257x | 1.269 1.252 1.251 |
| pp2048 | qresident | 17.01 | 3 | 16.94 | 17.08 | 1.322x | 1.338 1.315 1.313 |


<!-- phi4-14b  class=quant  fp16-resident-fits=0  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=/path/to/models/phi4/phi-4-Q4_K_M.gguf (8890306112 bytes)
     note: no F16 staged; fp16 resident ~29 GiB beside the GGUF, does not fit
-->
== phi4-14b  Sat Aug 29 21:49:27 UTC 2026 ==
### phi4-14b [stock] pass 1  env=''  args=''  21:51:07  clk=600 MHz  MemAvail=31494960 kB
<!--PRED 1	stock	memavail_kb=31495072	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5926,5603,5916,5035,4211,3735,3118,2753,2182,1358,3596-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | ROCKET     |  -1 |          pp2048 |         10.85 ± 0.03 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	10.85-->

### phi4-14b [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  22:05:23  clk=600 MHz  MemAvail=31526364 kB
<!--PRED 1	ub2048	memavail_kb=31526364	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7023,6462,6077,4808,4197,3731,3064,2733,2170,1349,3616-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | ROCKET     |  -1 |     2048 |          pp2048 |         14.19 ± 0.03 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	14.19-->

### phi4-14b [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  22:16:43  clk=600 MHz  MemAvail=31514104 kB
<!--PRED 2	ub2048	memavail_kb=31514104	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6759,5349,5944,5020,4203,3737,3081,2702,2155,1334,3628-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | ROCKET     |  -1 |     2048 |          pp2048 |         14.10 ± 0.00 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	14.10-->

### phi4-14b [stock] pass 2  env=''  args=''  22:28:08  clk=600 MHz  MemAvail=31495804 kB
<!--PRED 2	stock	memavail_kb=31495804	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5669,5376,6087,5120,4193,3685,3074,2713,2156,1340,3621-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | ROCKET     |  -1 |          pp2048 |         11.00 ± 0.00 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	11.00-->

### phi4-14b [stock] pass 3  env=''  args=''  22:42:14  clk=600 MHz  MemAvail=31514144 kB
<!--PRED 3	stock	memavail_kb=31514144	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5727,6252,6081,5026,4272,3745,3029,2712,2153,1330,3630-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | ROCKET     |  -1 |          pp2048 |         10.88 ± 0.03 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	10.88-->

### phi4-14b [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  22:56:29  clk=600 MHz  MemAvail=31508128 kB
<!--PRED 3	ub2048	memavail_kb=31508128	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5606,5816,6385,5055,4267,3770,2956,2734,2149,1338,3626-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| llama 13B Q4_K - Medium        |   8.28 GiB |    14.66 B | ROCKET     |  -1 |     2048 |          pp2048 |         14.53 ± 0.02 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	14.53-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 10.91 | 3 | 10.85 | 11.00 | -- | -- |
| pp2048 | ub2048 | 14.27 | 3 | 14.10 | 14.53 | 1.308x | 1.308 1.282 1.335 |


<!-- qwen36-27b  class=quant  fp16-resident-fits=0  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=/path/to/models/qwen36/Qwen3.6-27B-Q4_K_M.gguf (17106773120 bytes)
     note: -ub lever only; fp16 resident is ~54 GiB
-->
== qwen36-27b  Sat Aug 29 23:06:26 UTC 2026 ==
### qwen36-27b [stock] pass 1  env=''  args=''  23:09:45  clk=600 MHz  MemAvail=31509416 kB
<!--PRED 1	stock	memavail_kb=31509628	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6705,6368,5943,5383,4855,3794,2981,2684,2144,1326,1664-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | ROCKET     |  -1 |          pp2048 |          6.05 ± 0.01 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	6.05-->

### qwen36-27b [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  23:35:40  clk=600 MHz  MemAvail=31484704 kB
<!--PRED 1	ub2048	memavail_kb=31484704	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7745,6016,6266,5574,4712,3739,3063,2615,2147,1331,1659-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | ROCKET     |  -1 |     2048 |          pp2048 |          9.34 ± 0.06 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	9.34-->

### qwen36-27b [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  23:53:42  clk=600 MHz  MemAvail=31514008 kB
<!--PRED 2	ub2048	memavail_kb=31514008	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7538,6733,6123,5576,4831,3756,3095,2638,2124,1329,1665-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | ROCKET     |  -1 |     2048 |          pp2048 |          9.27 ± 0.03 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	9.27-->

### qwen36-27b [stock] pass 2  env=''  args=''  00:11:52  clk=600 MHz  MemAvail=31517012 kB
<!--PRED 2	stock	memavail_kb=31517012	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6752,6703,6175,5628,4743,3768,3125,2643,2116,1332,1665-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | ROCKET     |  -1 |          pp2048 |          6.08 ± 0.01 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	6.08-->

### qwen36-27b [stock] pass 3  env=''  args=''  00:37:40  clk=600 MHz  MemAvail=31506676 kB
<!--PRED 3	stock	memavail_kb=31506676	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7654,6377,6327,5774,4844,3811,3091,2602,2107,1326,1670-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | ROCKET     |  -1 |          pp2048 |          6.05 ± 0.01 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	6.05-->

### qwen36-27b [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  01:03:37  clk=600 MHz  MemAvail=31476452 kB
<!--PRED 3	ub2048	memavail_kb=31476452	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7301,6492,6671,5997,4759,3832,3127,2594,2129,1335,1649-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | ROCKET     |  -1 |     2048 |          pp2048 |          9.24 ± 0.02 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	9.24-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 6.06 | 3 | 6.05 | 6.08 | -- | -- |
| pp2048 | ub2048 | 9.28 | 3 | 9.24 | 9.34 | 1.532x | 1.544 1.525 1.527 |


<!-- gpt-oss-20b  class=moe  fp16-resident-fits=0  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=/path/to/models/gpt-oss-20b/gpt-oss-20b-mxfp4.gguf (12109566560 bytes)
-->
== gpt-oss-20b  Sun Aug 30 01:19:09 UTC 2026 ==
### gpt-oss-20b [stock] pass 1  env=''  args=''  01:20:44  clk=600 MHz  MemAvail=31512068 kB
<!--PRED 1	stock	memavail_kb=31512068	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8319,6492,6778,5900,5341,4090,3410,2822,2302,1493,2664-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |          pp2048 |         22.86 ± 0.19 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	22.86-->
    [moe-int8] experts exercised: 1683 resident on the NPU (13575MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24496MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [moe-int8] resident budget reached after 62 expert stacks (24140MB RAM, 15693MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24496MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gpt-oss-20b [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  01:28:55  clk=600 MHz  MemAvail=31511596 kB
<!--PRED 1	ub2048	memavail_kb=31511596	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6908,6136,7076,5912,5579,4221,3549,3071,2482,1653,2493-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.75 ± 0.18 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	28.75-->
    [moe-int8] experts exercised: 1683 resident on the NPU (13575MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24505MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [moe-int8] resident budget reached after 62 expert stacks (24140MB RAM, 15693MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24505MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gpt-oss-20b [moe1] pass 1  env='ROCKET_MOE=1'  args='-b 2048 -ub 2048'  01:35:51  clk=600 MHz  MemAvail=31472080 kB
<!--PRED 1	moe1	memavail_kb=31472080	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8166,7718,9155,7997,7548,6204,5373,4778,3637,1989,1577-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         34.04 ± 0.48 |

build: 171974745 (10558)
<!--DATA 1	moe1	pp2048	34.04-->
    [moe-int8] experts exercised: 1947 resident on the NPU (15705MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24465MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gpt-oss-20b [moe0] pass 1  env='ROCKET_MOE=0'  args='-b 2048 -ub 2048'  01:42:03  clk=600 MHz  MemAvail=31510304 kB
<!--PRED 1	moe0	memavail_kb=31510364	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7423,8019,8440,7895,7629,6500,5682,4898,3850,2029,1472-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         13.63 ± 0.15 |

build: 171974745 (10558)
<!--DATA 1	moe0	pp2048	13.63-->

### gpt-oss-20b [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  01:53:42  clk=600 MHz  MemAvail=31513068 kB
<!--PRED 2	ub2048	memavail_kb=31513324	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7393,6410,7321,6457,5884,4749,4051,3517,2832,1616,2309-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.53 ± 0.31 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	28.53-->
    [moe-int8] experts exercised: 1683 resident on the NPU (13575MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24502MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [moe-int8] resident budget reached after 62 expert stacks (24140MB RAM, 15693MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24502MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gpt-oss-20b [moe1] pass 2  env='ROCKET_MOE=1'  args='-b 2048 -ub 2048'  02:00:40  clk=600 MHz  MemAvail=31425688 kB
<!--PRED 2	moe1	memavail_kb=31425688	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8299,7764,9029,8023,7550,6317,5536,4862,3750,1963,1526-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         34.36 ± 0.58 |

build: 171974745 (10558)
<!--DATA 2	moe1	pp2048	34.36-->
    [moe-int8] experts exercised: 1947 resident on the NPU (15705MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24414MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gpt-oss-20b [moe0] pass 2  env='ROCKET_MOE=0'  args='-b 2048 -ub 2048'  02:06:50  clk=600 MHz  MemAvail=31507532 kB
<!--PRED 2	moe0	memavail_kb=31507532	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6320,7834,7884,7271,6992,5868,5168,4476,3439,2024,1700-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         13.69 ± 0.06 |

build: 171974745 (10558)
<!--DATA 2	moe0	pp2048	13.69-->

### gpt-oss-20b [stock] pass 2  env=''  args=''  02:18:26  clk=600 MHz  MemAvail=31508360 kB
<!--PRED 2	stock	memavail_kb=31508360	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8005,6112,7198,6402,5649,4459,3892,3387,2700,1719,2329-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |          pp2048 |         22.74 ± 0.18 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	22.74-->
    [moe-int8] experts exercised: 1683 resident on the NPU (13575MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24493MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [moe-int8] resident budget reached after 62 expert stacks (24140MB RAM, 15693MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24493MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gpt-oss-20b [moe1] pass 3  env='ROCKET_MOE=1'  args='-b 2048 -ub 2048'  02:26:39  clk=600 MHz  MemAvail=31477280 kB
<!--PRED 3	moe1	memavail_kb=31476988	anonhuge_kb=0	hugepagesz_kb=2048	buddy=9625,8018,10226,8817,7962,6644,5784,5055,3645,1817,1569-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         34.14 ± 0.46 |

build: 171974745 (10558)
<!--DATA 3	moe1	pp2048	34.14-->
    [moe-int8] experts exercised: 1947 resident on the NPU (15705MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24470MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gpt-oss-20b [moe0] pass 3  env='ROCKET_MOE=0'  args='-b 2048 -ub 2048'  02:32:50  clk=600 MHz  MemAvail=31505772 kB
<!--PRED 3	moe0	memavail_kb=31505712	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7551,7670,8114,7490,7342,6282,5496,4796,3606,2010,1582-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         13.70 ± 0.05 |

build: 171974745 (10558)
<!--DATA 3	moe0	pp2048	13.70-->

### gpt-oss-20b [stock] pass 3  env=''  args=''  02:44:26  clk=600 MHz  MemAvail=31506780 kB
<!--PRED 3	stock	memavail_kb=31506680	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7347,7303,7836,6641,6287,5118,4483,3916,3069,1844,2034-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |          pp2048 |         22.72 ± 0.18 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	22.72-->
    [moe-int8] experts exercised: 1683 resident on the NPU (13575MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24488MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [moe-int8] resident budget reached after 62 expert stacks (24140MB RAM, 15693MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24488MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### gpt-oss-20b [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  02:52:39  clk=600 MHz  MemAvail=31507984 kB
<!--PRED 3	ub2048	memavail_kb=31507984	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7304,6072,7431,6451,5804,4558,3985,3492,2741,1784,2261-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.42 ± 0.21 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	28.42-->
    [moe-int8] experts exercised: 1683 resident on the NPU (13575MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24492MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [moe-int8] resident budget reached after 62 expert stacks (24140MB RAM, 15693MB IOVA) -- the rest of the experts stay on the CPU, which is a partial offload and not a loss. Bound by RAM: raise ROCKET_MOE_CACHE_MB if it is there.
    [rocket] MoE native-quant experts ON: mxfp4 -> int8, group=576 (nKt=5), resident budget 24492MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 22.77 | 3 | 22.72 | 22.86 | -- | -- |
| pp2048 | ub2048 | 28.57 | 3 | 28.42 | 28.75 | 1.254x | 1.258 1.255 1.251 |
| pp2048 | moe1 | 34.18 | 3 | 34.04 | 34.36 | 1.501x | 1.489 1.511 1.503 |
| pp2048 | moe0 | 13.67 | 3 | 13.63 | 13.70 | 0.600x | 0.596 0.602 0.603 |


<!-- deepseek-v2-lite  class=moe  fp16-resident-fits=0  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=/path/to/models/deepseek-v2-lite/DeepSeek-V2-Lite.Q4_K_M.gguf (10364416736 bytes)
-->
== deepseek-v2-lite  Sun Aug 30 02:58:38 UTC 2026 ==
### deepseek-v2-lite [stock] pass 1  env=''  args=''  02:59:36  clk=600 MHz  MemAvail=31518028 kB
<!--PRED 1	stock	memavail_kb=31518124	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7110,5814,6068,5227,4317,3809,3430,3001,2428,1595,3011-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |          pp2048 |         21.31 ± 0.02 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	21.31-->

### deepseek-v2-lite [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  03:07:01  clk=600 MHz  MemAvail=31517244 kB
<!--PRED 1	ub2048	memavail_kb=31517244	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6565,6484,6603,5558,4466,3361,3370,2971,2397,1564,3049-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.14 ± 0.16 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	28.14-->
    [moe-int8] experts exercised: 4779 resident on the NPU (14959MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24571MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [rocket] MoE native-quant experts ON: q4_K -> int8, group=512 (nKt=4), resident budget 24571MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### deepseek-v2-lite [moe1] pass 1  env='ROCKET_MOE=1'  args='-b 2048 -ub 2048'  03:14:08  clk=600 MHz  MemAvail=31511532 kB
<!--PRED 1	moe1	memavail_kb=31511736	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5911,5803,6429,5842,4972,4237,3802,3239,2584,1632,2870-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.35 ± 0.06 |

build: 171974745 (10558)
<!--DATA 1	moe1	pp2048	28.35-->
    [moe-int8] experts exercised: 4779 resident on the NPU (14959MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [rocket] MoE native-quant experts ON: q4_K -> int8, group=512 (nKt=4), resident budget 24573MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### deepseek-v2-lite [moe0] pass 1  env='ROCKET_MOE=0'  args='-b 2048 -ub 2048'  03:20:29  clk=600 MHz  MemAvail=31483012 kB
<!--PRED 1	moe0	memavail_kb=31483012	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6075,5642,6152,5677,4622,3981,3679,3166,2474,1449,3015-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |          pp2048 |         21.64 ± 0.03 |

build: 171974745 (10558)
<!--DATA 1	moe0	pp2048	21.64-->

### deepseek-v2-lite [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  03:27:48  clk=600 MHz  MemAvail=31480924 kB
<!--PRED 2	ub2048	memavail_kb=31480924	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6505,6893,6587,6039,4904,4240,3202,3087,2433,1440,3050-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.15 ± 0.14 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	28.15-->
    [moe-int8] experts exercised: 4779 resident on the NPU (14959MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24539MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [rocket] MoE native-quant experts ON: q4_K -> int8, group=512 (nKt=4), resident budget 24539MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### deepseek-v2-lite [moe1] pass 2  env='ROCKET_MOE=1'  args='-b 2048 -ub 2048'  03:34:55  clk=600 MHz  MemAvail=31516920 kB
<!--PRED 2	moe1	memavail_kb=31516920	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6626,6431,6547,5875,5098,4369,3950,3361,2695,1568,2842-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.33 ± 0.09 |

build: 171974745 (10558)
<!--DATA 2	moe1	pp2048	28.33-->
    [moe-int8] experts exercised: 4779 resident on the NPU (14959MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [rocket] MoE native-quant experts ON: q4_K -> int8, group=512 (nKt=4), resident budget 24578MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### deepseek-v2-lite [moe0] pass 2  env='ROCKET_MOE=0'  args='-b 2048 -ub 2048'  03:41:17  clk=600 MHz  MemAvail=31506024 kB
<!--PRED 2	moe0	memavail_kb=31506024	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5131,6198,6348,5874,4896,4069,3808,3298,2579,1438,2967-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |          pp2048 |         21.64 ± 0.05 |

build: 171974745 (10558)
<!--DATA 2	moe0	pp2048	21.64-->

### deepseek-v2-lite [stock] pass 2  env=''  args=''  03:48:36  clk=600 MHz  MemAvail=31494276 kB
<!--PRED 2	stock	memavail_kb=31494276	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6349,6291,6699,5878,4833,4170,3510,3226,2522,1395,3021-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |          pp2048 |         21.26 ± 0.02 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	21.26-->

### deepseek-v2-lite [moe1] pass 3  env='ROCKET_MOE=1'  args='-b 2048 -ub 2048'  03:56:43  clk=600 MHz  MemAvail=31513888 kB
<!--PRED 3	moe1	memavail_kb=31513644	anonhuge_kb=0	hugepagesz_kb=2048	buddy=7707,6806,7238,6438,5144,4151,3826,3321,2654,1511,2890-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.37 ± 0.13 |

build: 171974745 (10558)
<!--DATA 3	moe1	pp2048	28.37-->
    [moe-int8] experts exercised: 4779 resident on the NPU (14959MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [rocket] MoE native-quant experts ON: q4_K -> int8, group=512 (nKt=4), resident budget 24568MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### deepseek-v2-lite [moe0] pass 3  env='ROCKET_MOE=0'  args='-b 2048 -ub 2048'  04:03:05  clk=600 MHz  MemAvail=31484112 kB
<!--PRED 3	moe0	memavail_kb=31484112	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6237,6107,6475,5909,4994,4278,3947,3377,2644,1476,2898-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |          pp2048 |         21.56 ± 0.01 |

build: 171974745 (10558)
<!--DATA 3	moe0	pp2048	21.56-->

### deepseek-v2-lite [stock] pass 3  env=''  args=''  04:10:26  clk=600 MHz  MemAvail=31513020 kB
<!--PRED 3	stock	memavail_kb=31513020	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5908,6893,6484,5884,4689,4165,3623,3247,2565,1430,2990-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |          pp2048 |         21.27 ± 0.01 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	21.27-->

### deepseek-v2-lite [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  04:17:50  clk=600 MHz  MemAvail=31521800 kB
<!--PRED 3	ub2048	memavail_kb=31521800	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6034,7122,6760,5851,4847,3580,3620,3158,2516,1411,3041-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| deepseek2 16B Q4_K - Medium    |   9.65 GiB |    15.71 B | ROCKET     |  -1 |     2048 |          pp2048 |         28.17 ± 0.06 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	28.17-->
    [moe-int8] experts exercised: 4779 resident on the NPU (14959MB), 0 streamed via dequant->fp16 -- 100% of the per-micro-batch dequant removed
    [moe-int8] residency pre-flight: 24581MB RAM budget, 19200MB NPU IOVA across 5 worker fds
    [rocket] MoE native-quant experts ON: q4_K -> int8, group=512 (nKt=4), resident budget 24581MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 21.28 | 3 | 21.26 | 21.31 | -- | -- |
| pp2048 | ub2048 | 28.15 | 3 | 28.14 | 28.17 | 1.323x | 1.321 1.324 1.324 |
| pp2048 | moe1 | 28.35 | 3 | 28.33 | 28.37 | 1.332x | 1.330 1.333 1.334 |
| pp2048 | moe0 | 21.61 | 3 | 21.56 | 21.64 | 1.016x | 1.015 1.018 1.014 |


<!-- qwen3-30b-a3b  class=moe  fp16-resident-fits=0  MODE=headline  TESTS='-p 2048 -n 0 -r 3'
     gguf=/path/to/models/qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf (18556686912 bytes)
     note: 29 of 30.5 B are experts: cannot be held resident here
-->
== qwen3-30b-a3b  Sun Aug 30 07:52:16 UTC 2026 ==
### qwen3-30b-a3b [stock] pass 1  env=''  args=''  07:53:41  clk=600 MHz  MemAvail=31500844 kB
<!--PRED 1	stock	memavail_kb=31501432	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6300,6060,6509,5723,4521,4104,3765,3235,2495,1309,1109-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | ROCKET     |  -1 |          pp2048 |         14.75 ± 0.01 |

build: 171974745 (10558)
<!--DATA 1	stock	pp2048	14.75-->

### qwen3-30b-a3b [ub2048] pass 1  env=''  args='-b 2048 -ub 2048'  08:04:26  clk=600 MHz  MemAvail=31500956 kB
<!--PRED 1	ub2048	memavail_kb=31500956	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6931,5566,6277,5854,4348,4050,3699,3182,2472,1299,1135-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | ROCKET     |  -1 |     2048 |          pp2048 |         13.36 ± 0.02 |

build: 171974745 (10558)
<!--DATA 1	ub2048	pp2048	13.36-->

### qwen3-30b-a3b [moe1] pass 1  env='ROCKET_MOE=1'  args='-b 2048 -ub 2048'  08:18:17  clk=600 MHz  MemAvail=31399128 kB
<!--PRED 1	moe1	memavail_kb=31398836	anonhuge_kb=0	hugepagesz_kb=2048	buddy=24787,31587,67134,44791,4583,4160,3763,3213,2574,1520,1413-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | ROCKET     |  -1 |     2048 |          pp2048 |         10.76 ± 0.07 |

build: 171974745 (10558)
<!--DATA 1	moe1	pp2048	10.76-->
    [moe-int8] experts exercised: 9435 resident on the NPU (15920MB), 5847 streamed via dequant->fp16 -- 62% of the per-micro-batch dequant removed
    [moe-int8] resident budget reached at 9435 experts (15920MB on the NPU, 24453MB charged incl. the GGUF source) -- the remaining experts stream via dequant->fp16 (raise ROCKET_MOE_CACHE_MB, or ROCKET_N_THREADS for more per-fd IOVA)
    [rocket] MoE native-quant experts ON: q4_K -> int8, group=512 (nKt=4), resident budget 24454MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### qwen3-30b-a3b [moe0] pass 1  env='ROCKET_MOE=0'  args='-b 2048 -ub 2048'  08:33:07  clk=600 MHz  MemAvail=31482248 kB
<!--PRED 1	moe0	memavail_kb=31482248	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6534,8003,7773,6847,5648,5205,4472,3599,2709,1975,587-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | ROCKET     |  -1 |     2048 |          pp2048 |         13.38 ± 0.02 |

build: 171974745 (10558)
<!--DATA 1	moe0	pp2048	13.38-->

### qwen3-30b-a3b [ub2048] pass 2  env=''  args='-b 2048 -ub 2048'  08:44:49  clk=600 MHz  MemAvail=31486496 kB
<!--PRED 2	ub2048	memavail_kb=31486496	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6813,6646,6193,5804,4726,4379,3998,3391,2682,1761,814-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | ROCKET     |  -1 |     2048 |          pp2048 |         13.38 ± 0.01 |

build: 171974745 (10558)
<!--DATA 2	ub2048	pp2048	13.38-->

### qwen3-30b-a3b [moe1] pass 2  env='ROCKET_MOE=1'  args='-b 2048 -ub 2048'  08:58:38  clk=600 MHz  MemAvail=31317036 kB
<!--PRED 2	moe1	memavail_kb=31317036	anonhuge_kb=0	hugepagesz_kb=2048	buddy=23246,34251,75759,54008,5774,5269,4557,3744,2762,1731,960-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | ROCKET     |  -1 |     2048 |          pp2048 |         10.58 ± 0.43 |

build: 171974745 (10558)
<!--DATA 2	moe1	pp2048	10.58-->
    [moe-int8] experts exercised: 9427 resident on the NPU (15907MB), 5849 streamed via dequant->fp16 -- 62% of the per-micro-batch dequant removed
    [moe-int8] resident budget reached at 9427 experts (15907MB on the NPU, 24433MB charged incl. the GGUF source) -- the remaining experts stream via dequant->fp16 (raise ROCKET_MOE_CACHE_MB, or ROCKET_N_THREADS for more per-fd IOVA)
    [rocket] MoE native-quant experts ON: q4_K -> int8, group=512 (nKt=4), resident budget 24433MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### qwen3-30b-a3b [moe0] pass 2  env='ROCKET_MOE=0'  args='-b 2048 -ub 2048'  09:13:38  clk=600 MHz  MemAvail=31483760 kB
<!--PRED 2	moe0	memavail_kb=31483760	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6572,6812,7321,6647,5200,4745,4272,3630,2862,1967,587-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | ROCKET     |  -1 |     2048 |          pp2048 |         13.39 ± 0.00 |

build: 171974745 (10558)
<!--DATA 2	moe0	pp2048	13.39-->

### qwen3-30b-a3b [stock] pass 2  env=''  args=''  09:25:20  clk=600 MHz  MemAvail=31493612 kB
<!--PRED 2	stock	memavail_kb=31493320	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6151,6102,6807,5706,4647,4211,3917,3429,2743,1747,812-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | ROCKET     |  -1 |          pp2048 |         14.72 ± 0.00 |

build: 171974745 (10558)
<!--DATA 2	stock	pp2048	14.72-->

### qwen3-30b-a3b [moe1] pass 3  env='ROCKET_MOE=1'  args='-b 2048 -ub 2048'  09:38:14  clk=600 MHz  MemAvail=31360488 kB
<!--PRED 3	moe1	memavail_kb=31360488	anonhuge_kb=0	hugepagesz_kb=2048	buddy=13618,20811,44123,48435,5195,4779,4143,3472,2742,1847,1205-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | ROCKET     |  -1 |     2048 |          pp2048 |         10.78 ± 0.30 |

build: 171974745 (10558)
<!--DATA 3	moe1	pp2048	10.78-->
    [moe-int8] experts exercised: 9437 resident on the NPU (15923MB), 5842 streamed via dequant->fp16 -- 62% of the per-micro-batch dequant removed
    [moe-int8] resident budget reached at 9437 experts (15923MB on the NPU, 24458MB charged incl. the GGUF source) -- the remaining experts stream via dequant->fp16 (raise ROCKET_MOE_CACHE_MB, or ROCKET_N_THREADS for more per-fd IOVA)
    [rocket] MoE native-quant experts ON: q4_K -> int8, group=512 (nKt=4), resident budget 24459MB (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)

### qwen3-30b-a3b [moe0] pass 3  env='ROCKET_MOE=0'  args='-b 2048 -ub 2048'  09:53:09  clk=600 MHz  MemAvail=31482580 kB
<!--PRED 3	moe0	memavail_kb=31482580	anonhuge_kb=0	hugepagesz_kb=2048	buddy=8027,7645,8454,7959,5884,4889,4184,3619,2874,2080,503-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | ROCKET     |  -1 |     2048 |          pp2048 |         13.40 ± 0.01 |

build: 171974745 (10558)
<!--DATA 3	moe0	pp2048	13.40-->

### qwen3-30b-a3b [stock] pass 3  env=''  args=''  10:04:50  clk=600 MHz  MemAvail=31481928 kB
<!--PRED 3	stock	memavail_kb=31481928	anonhuge_kb=0	hugepagesz_kb=2048	buddy=6178,7236,7077,6852,5312,4714,4149,3558,2856,1604,784-->
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | ROCKET     |  -1 |          pp2048 |         14.76 ± 0.00 |

build: 171974745 (10558)
<!--DATA 3	stock	pp2048	14.76-->

### qwen3-30b-a3b [ub2048] pass 3  env=''  args='-b 2048 -ub 2048'  10:15:34  clk=600 MHz  MemAvail=31486412 kB
<!--PRED 3	ub2048	memavail_kb=31486732	anonhuge_kb=0	hugepagesz_kb=2048	buddy=5668,6948,6815,6004,4760,4453,4018,3471,2787,1610,845-->
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | ROCKET     |  -1 |     2048 |          pp2048 |         13.40 ± 0.01 |

build: 171974745 (10558)
<!--DATA 3	ub2048	pp2048	13.40-->

#### summary: per-arm mean over 3 passes, ratios paired within a pass against [stock]
| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |
|---|---|---:|---:|---:|---:|---:|---|
| pp2048 | stock | 14.74 | 3 | 14.72 | 14.76 | -- | -- |
| pp2048 | ub2048 | 13.38 | 3 | 13.36 | 13.40 | 0.908x | 0.906 0.909 0.908 |
| pp2048 | moe1 | 10.71 | 3 | 10.58 | 10.78 | 0.726x | 0.729 0.719 0.730 |
| pp2048 | moe0 | 13.39 | 3 | 13.38 | 13.40 | 0.908x | 0.907 0.910 0.908 |

