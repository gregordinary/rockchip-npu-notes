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

## What each f16 unit's knob moves, read without a timed run

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
against stock [inference from the two figures' agreement to 0.01, not established]. The plan's
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

**The dense units now have the same control** — see "The dense `-ub` lever decomposed" below. The
dense rows in the main table remain "what the flag buys"; the dqc units are what turned three of
them into "what the dequant costs".

**`ROCKET_MOE=1` adds almost nothing here** — 1.332x against 1.323x, and identical placement (4779
experts, 14959 MB) on both arms. Unlike gpt-oss, where FORCED placed 264 more expert stacks and
bought 1.197x, this model's AUTO route already takes everything FORCED would, so the two states
coincide. **That is a per-model property and not a general one**: the same two arms differ by 20%
on one model and 0.7% on the other.

## The dense `-ub` lever decomposed: the dqc mechanism control

The dense analogue of the MoE units' `ROCKET_MOE=0` arm now exists: **`ROCKET_DEQUANT_CACHE_MB`**
holds each streaming quant weight's dequantized fp16 form host-side (dequant once per process),
leaving the per-call pack, the submit and the placement exactly the shipped path. Four arms per
unit — stock, `ub2048`, and the same pair under the cache — so `dqc2048/dqc512` paired within a
pass is the `-ub` lever with the dequant term removed, and `full / control` is the dequant
component. Greedy output under the cache is byte-identical to the streaming path, and every arm's
`[dq-cache]` teardown line reported full engagement (150 / 200 / 165 weights, 0 still-streaming)
[HW sweep 2026-08-31, RK1, 600 MHz, governor `performance`, three rotated passes, 0 failed arms]:

| unit | full `-ub` lever | dequant component | non-dequant residue (control lever) | per-pass control |
|---|---:|---:|---:|---|
| `qwen35-9b` | **1.430x** | **1.378x** | **1.038x** | 1.031 1.039 1.044 |
| `smolvlm2` | **0.946x** | **1.199x** | **0.789x** | 0.796 0.785 0.787 |
| `qwen35-08b` | 1.066x | ~1.10x | **unresolved** | 1.020 0.920 0.979 |
| `llama32-3b` | 1.085x | 1.258x | **0.863x** | 0.846 0.879 0.864 |
| `ministral3-3b` | 1.035x | 1.239x | **0.835x** | 0.837 0.835 0.834 |
| `phi4mini` | 1.107x | 1.282x | **0.864x** | 0.855 0.870 0.866 |

The last three landed 2026-08-31 in a second campaign (stock arms replicating the matrix epoch
to <= 1.3%, idle audited, dq-cache fully engaged on every arm, raw files in `dqc-session/`).
**Every sub-4B model measured — four of four — carries a real 14-21% non-dequant loss at
`-ub 2048` that a larger dequant win masks**, and the 9B's 1.038x residue is the outlier, not
the rule. The dequant term removed at the DEFAULT `-ub` (dqc512-vs-stock) reads **1.369x /
1.338x / 1.373x** on the three 3B-class units — well above their stacked-recipe measurements
(1.212x / 1.132x / 1.211x) — which is what put the unstacked-residency question on the board
for the whole class rather than for smolvlm2 alone.

- **On the 9B the documented mechanism is the mechanism.** 1.378 of the 1.430 is dequant
  amortization — ~90% of the lever in log terms — and the residue is 1.038x. The stock arm
  replicated the matrix row across the campaign (19.06 against 19.14 t/s), so the two epochs are
  directly comparable.
- **On `smolvlm2` the flag's loss is NOT a dequant effect, and dequant amortization is what was
  hiding most of it.** With the dequant term removed, `-ub 2048` loses **0.789x** — a real 21%
  non-dequant cost at the larger micro-batch (mechanism open) — and the shipped 0.946x is that
  loss with a 1.199x dequant win partly masking it. The unit's best arm is `dqc512` at **1.266x**:
  on this model the win is removing the dequant at the DEFAULT `-ub`, not raising `-ub`.
- **The 0.8B cannot resolve its control lever.** Its per-pass control ratios straddle 1.00
  (0.920-1.020) and its stock arm spans 8.4% across processes — the same small-model per-process
  spread unit 2 showed. Reported unresolved rather than averaged.

**The actionable form of the smolvlm2 finding is residency at the DEFAULT `-ub`, and it was
hiding behind the arm stacking.** The matrix's `qresident` arm stacks on `-b 2048 -ub 2048`, so on
this model it measured residency on top of the loss and read 1.002x flat. Unstacked
[HW sweep 2026-08-31, RK1, 600 MHz, three rotated passes, 165 resident (2976 MB), 0 streamed]:
`ROCKET_QUANT_RESIDENT=auto` at stock `-ub 512` reads **1.346x** (1.346 1.337 1.353) — above the
dqc512 control's 1.266x, as removing the per-call pack on top of the dequant should be. On a
model whose `-ub` residue is a loss, the recipe inverts: residency INSTEAD OF `-ub 2048`, not on
top of it.

**The inversion is now measured across the whole sub-4B class, and it holds on every unit**
[HW sweep 2026-08-31, RK1, 600 MHz, three rotated passes per unit, all arms 100% resident /
0 streamed, stock arms replicating the matrix epoch to <= 0.7%; raw files
`dqc-session/*-qres512.md`]:

| unit | qres512-vs-stock | dqc512 control | stacked recipe (matrix) | unstacked over stacked |
|---|---:|---:|---:|---:|
| `llama32-3b` | **1.472x** (1.450-1.492) | 1.369x | 1.212x | +21% |
| `ministral3-3b` | **1.445x** (1.416-1.463) | 1.338x | 1.132x | +28% |
| `phi4mini` | **1.508x** (1.491-1.520) | 1.373x | 1.211x | +25% |
| `smolvlm2` | **1.346x** (1.337-1.353) | 1.266x | 1.002x | +34% |
| `ministral3-8b` | **1.651x** (1.630-1.674) | — | 1.325x | +25% |
| `qwen35-9b` | **1.752x** (1.737-1.771) | — | 1.661x | +5.5% |
| `gemma4-12b` | **1.368x** (1.319-1.402) | — | 1.322x | +3.5% |

Each sub-4B unit's qres512 sits above its own dqc512 control (pack removal on top of the
dequant, the predicted ordering), and 21-34% above the stacked recipe the guide shipped.

**Seven of seven models prefer the unstacked form, and that is what licenses a class rule.**
The set spans 1.8-11.9 B (`llama-bench` parameter counts), both signs of the non-dequant
residue, and both full and partial residency, and no model measured both ways prefers stacking.
So the recipe for a quantized GGUF is **`ROCKET_QUANT_RESIDENT=auto` at the DEFAULT `-ub`
whenever the fp16 image fits resident, wholly or partly**, and `-b 2048 -ub 2048` is the lever
only for a model that cannot go resident at all. **One model with a resident arm is still
outside that statement**: `qwen35-08b` cannot resolve a ratio of this size at any affordable
pass count in its class. The rule was deliberately not re-cut on the 9B alone, because one
model's ratio does not read across: the same lesson the MoE work floor taught.

`gemma4-12b` is the only row here whose arms did NOT run at the same residency, and the reason
is the finding below. Its +3.5% is the narrowest margin in the set.

**The 9B row is the class edge, and it settles a question the sub-4B rows could not.** It is
the only measured model whose non-dequant `-ub` residue is a WIN (1.038x), so it is the case
where the residue argument predicts stacking should hold. It does not: unstacked reads
**1.752x** against the stacked **1.661x**, resolved the same way on all three passes, with the
stock arm replicating the matrix epoch to 0.2% and every pass 100% resident
[HW sweep 2026-08-31, RK1, 600 MHz, three rotated passes, raw in
`ro-session/qwen35-9b-qres512.md`]. So the earlier expectation recorded here — that stacking
barely matters on a model with a winning residue — was wrong by 5.5%, and in the direction of
unstacked.

### The unstacked recipe also buys RESIDENCY on a partly-placed model, which is a third mechanism

`gemma4-12b` was expected to be the case where the class rule breaks, because it places only
part of its weights and the streamed remainder still pays the per-micro-batch dequant that the
unstacked form exists to avoid. It does not break, and the reason is a term the six
fully-placed models could not show [HW sweep 2026-08-31, RK1, 600 MHz, three rotated passes,
raw in `ro-session/gemma4-12b-qres.md`]:

| arm | `-ub` | weights resident | MB resident | pp2048 |
|---|---|---:|---:|---:|
| stacked (matrix epoch) | 2048 | 239 / 240 / 243 of 328 | 15018-15255 | 1.322x |
| unstacked | default | 282 / 284 / 272 of 328 | 17077-17853 | **1.368x** |

**The unstacked arm places 9-13 points more of the model.** Both arms stop at the same place,
the runtime's 9535 MB `MemAvailable` reserve floor rather than the 21129 MB resident-weight
budget, and the unstacked arm reaches that floor about 2.6-2.8 GB later. The difference is the
activation and compute buffers, which `-b 2048 -ub 2048` grows about 4x. On a model where the
budget never binds, that RAM is spent on the micro-batch instead of on weights.

So on a partly-placed model the two arms are not the same placement with a different
micro-batch. Raising `-ub` costs residency directly, and a comparison here has to read the
`[f16-resident]` outcome line per arm rather than assume the arms are comparable.

**This unit's per-pass spread is the widest of the seven, and placement is the candidate.** Its
ratios are 1.402 / 1.382 / 1.319, a 6.3% spread against 1-3% on the fully-placed units, and the
pass with the lowest ratio is also the pass with the lowest placement (83% against 86% and
87%). Three points cannot establish that relation — a monotone column over three points scores
a degenerate rank correlation — so this is a candidate and not a measurement [hypothesis]. What
it does say is that a partly-placed model's ratio carries a placement term that a fully-placed
one does not, and a future unit on this model wants more passes than three.

**The mechanism is the per-call pack, and it explains why the residue alone under-predicts.**
Write `k` for the factor by which unstacked beats what a residue correction alone gives,
`k = (unstacked/stock) x residue / (stacked/stock)`. It is above 1 on every model measured
because residency removes the per-call pack and upload as well as the dequant, and at the
default `-ub` there are four times as many micro-batch calls for it to remove. **`k` is not a
constant: it rises with model size** — 1.048 / 1.060 / 1.066 / 1.076 on the four sub-4B units
and **1.095** on the 9B. A band built on a flat `k` fitted at one size will read low at a
larger one, which is how this cell's own registered band missed on the high side.

**The smolvlm2 non-dequant residue is bounded by a profiled A/B of the dqc pair, and half of
it is the output de-tile** [HW sweep 2026-08-31, RK1, 600 MHz, one profiled process per arm,
`ROCKET_MM_PROFILE=1 ROCKET_FA_TIMING=1`, both arms under the cache; the profiled arms
reproduce the campaign ratio to 0.1% (65.14/51.48 = 0.790 against 0.789), raw in
`dqc-session/smolvlm2.dqc{512,2048}-prof.{out,err}`]:

| driver bucket | dqc512 | dqc2048 | delta |
|---|---:|---:|---:|
| read (readback + C de-tile) | 28.2 s | 46.3 s | **+18.0 s** |
| wait | 154.2 s | 189.9 s | **+35.7 s** |
| pack (packA + packB) | 57.8 s | 43.8 s | **-14.1 s** |
| everything else (gen/sync/submit/pack_act/unpack_out/dequant) | 14.3 s | 13.8 s | ~0 |

The accounted delta (+39.7 s per process) closes against the measured wall loss (~33.5 s)
within the profiler's own overhead. Two of the three candidates are now placed:

- **The C de-tile at [2048,N] is real and is ~half the residue.** `read` grows 64% at
  byte-identical output elements (5083M both arms), and per job-batch it is 1.76 -> 10.9 ms —
  the de-tile is superlinear in M on this model's N geometry.
- **The attention-chunk-across-the-FA-gate candidate is dead as framed**: `ROCKET_FA_TIMING`
  printed nothing on either arm because llama-bench runs `-fa 0`, so the whole campaign —
  including the 0.789x residue itself — contains **no FLASH_ATTN ops at all**. Scope: a
  deployment running `-fa` on is a different configuration this table does not price.
- **The remainder sits in `wait`** (+23% at identical MAC work, 9.6 -> 44.6 ms per batch),
  net of the pack saving — consistent with the fewer, larger batches overlapping less host
  work with NPU time, which is the scheduler-shaped candidate [hypothesis, not isolated].

**The dequant share is a live term, not a constant.** The weight_dequant profile bucket, read in
the same session on an idle board, is 2.47 / 33.9 / 7.6 s per `-ub 512` pass (08b / 9B /
smolvlm2) against walls of 20.7 / 107 / 39.4 s, and a bucket-removal model on those covariates
predicted the measured `dqc2048`-vs-`ub2048` ratios to 0.5% on all three units and smolvlm2's
control lever to 1%. The same buckets read under a one-core tenant (below) were 1.5-2.1x larger.

**The accidental load experiment: a single busy core flips the flag's sign on smolvlm2.** The
first take of this campaign unknowingly ran behind a leaked spinner process holding one core at
100%. Those runs are kept (`*-dqc.tenant-loaded.md`) as a labeled measurement of host-load bias,
because the contrast is a finding [HW, same board, same binary, tenant present throughout]:

| unit | `-ub` lever, idle | `-ub` lever, one-core tenant | control lever idle -> loaded |
|---|---:|---:|---|
| `qwen35-08b` | 1.066x | **1.484x** | unresolved -> 1.229 |
| `qwen35-9b` | 1.430x | 1.526x | 1.038 -> 1.135 |
| `smolvlm2` | **0.946x** | **1.120x** | 0.789 -> 0.882 |

A tenant steals exactly the resource the stock arm uses four times as often (the A76-pinned
dequant pool), so it inflates the flag on every unit and takes smolvlm2's loss to a win. The
practical reading for the tuning guide: the `-ub 2048` recipe's value GROWS on a loaded host, and
a benchmark taken with any background load overstates it for an idle deployment.

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
experts, and every expert dispatch at `-ub` 2048 carries 201 MMAC, under the 340 MMAC work floor
(`ROCKET_MOE_MIN_WORK`), so the route declines rather than half-placing.

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
identical placement, which is the expected behavior of a shared fp16 prepack and a cheap check
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

### The predictors do not predict the spread, and the re-run says what would

The correlation has now been run, and unit 2 re-run under the predictor-carrying harness
[HW sweep 2026-08-31, RK1, 600 MHz, governor `performance`; raw rows in
`qwen35-08b-f16-rerun.md`, two 6-pass campaigns, 24 predictor-carrying arms].

**The recorded covariates cannot carry the spread.** Over 177 joined (PRED, t/s) rows — the
matrix's 153 plus the re-run's 24 — the loud groups swing 8-13% in t/s while their recorded
pre-state is statistically flat: MemAvailable varies under 0.9%, the buddyinfo high-order tail
under 1.2%, and the within-group correlations are weak with inconsistent signs (|r| <= 0.5,
both directions across groups). The board's post-reset state is also uniformly unfragmented
everywhere — 85-89% of free pages sit at order >= 8 in every group — so the `compact_memory`
half of the reset is doing its job and fragmentation *as buddyinfo records it* does not vary
enough between processes to explain anything. Scope: this closes the recorded covariates as
predictors of the mode; it says nothing about axes the PRED line does not record (physical
page placement of the BOs and mmap, thread placement).

**The spread replicated in both campaigns, and its shape narrows the mechanism.** Each arm
spans 11.7-12.9% across 12 processes, structured as a tight floor near 116.5 t/s (repeats to
0.3%) with fast excursions to 129-132 (+12%) and a few intermediate levels. The mode is not
noise around a mean — it is which level a process lands on. In campaign 1 the level flipped
between the two processes of a pass (per-pass ratios 0.93-1.13); in campaign 2 adjacent
processes shared it (all six ratios 0.99-1.01). Same board, same binary, same hour. A
mechanism that survives those two patterns is one decided per process at startup by something
the allocator hands the process — where the working set physically lands — with whatever
history-dependence made campaign 2's processes correlate. That candidate class (BO/mmap
physical placement, cache congruence, thread placement) needs a per-process **readout**, not
more rows of these predictors.

**That readout now exists and every timed arm carries it**: a `<!--RO-->` line beside the
`<!--DATA-->` row, holding both cluster PMUs across the timed region, a per-CPU jiffy delta,
and one bounded `pagemap` sample of where the process's pages landed. What it can and cannot
score, and the positive controls it had to pass first, are in
[../per-process-readout.md](../per-process-readout.md); [ro-join.py](ro-join.py) joins those
lines back to the t/s rows and marks a covariate FLAT when its own range is too small for a
correlation over it to mean anything.

### The L3 column is a within-arm covariate, the level is thread placement, and one intervention separates them

The readout has now been taken on the unit it was built for [HW sweep 2026-08-31, RK1, 600 MHz,
governor `performance`, `qwen35-08b-f16`, six rotated passes over two arms, memory reset before
every arm, `pfn_zero_frac`=0.0000 and `pmu_enabled`=100 on all twelve processes; raw rows in
`ro-session/trackd-08b-f16-6pass.md`]. Both arms reproduced the mode at the depth the unit was
picked for: `f16-stock` spans 115.56-131.23 t/s (12.4%) and `f16-res` 117.23-130.27 (11.1%).

**One covariate tracks the wall, and it sits at the L3-to-DRAM boundary.** Pooled over all twelve
processes, `l3ref_pki` has a 23.1% range and a rank correlation of **-0.972** against t/s, and the
ordering is close to monotone: 8.02 at the fastest process, 10.35 at the slowest, with the one
intermediate wall (122.03 t/s) carrying an intermediate 9.34. Everything upstream of the L3 is
flat. `memacc_pki` varies **0.51%**, `l1dref_pki` 1.8%, `l2ref_pki` 3.4% at rho -0.273, and
`a76_inst_retired` 1.4%. So the same instruction stream issues the same memory accesses and takes
the same L1 and L2 misses. What differs between a fast process and a slow one is the fraction of
those L2 misses the L3 satisfies.

| t/s | arm | wall s | `l3ref_pki` | L3 refills | GB read | `l2ref_pki` | `memacc_pki` | `a55_inst_share` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 131.23 | stock | 66 | 8.02 | 2.95 G | 188.6 | 5.70 | 349.4 | 0.2037 |
| 130.27 | res | 66 | 8.01 | 2.94 G | 188.0 | 5.62 | 348.6 | 0.2052 |
| 128.42 | stock | 67 | 8.35 | 3.07 G | 196.2 | 5.70 | 349.0 | 0.2047 |
| 126.57 | stock | 69 | 8.40 | 3.07 G | 196.7 | 5.73 | 349.0 | 0.2060 |
| 122.03 | stock | 71 | 9.34 | 3.40 G | 217.9 | 5.74 | 348.2 | 0.2095 |
| 118.30 | res | 72 | 9.95 | 3.62 G | 231.7 | 5.59 | 348.0 | 0.2095 |
| 117.82 | res | 74 | 10.11 | 3.69 G | 236.4 | 5.65 | 347.6 | 0.2080 |
| 117.54 | res | 74 | 10.11 | 3.67 G | 234.7 | 5.68 | 347.9 | 0.2106 |
| 117.40 | res | 73 | 10.18 | 3.69 G | 235.8 | 5.69 | 348.4 | 0.2114 |
| 117.23 | res | 74 | 10.17 | 3.69 G | 235.8 | 5.67 | 348.2 | 0.2115 |
| 116.81 | stock | 73 | 10.35 | 3.75 G | 239.8 | 5.79 | 348.8 | 0.2115 |
| 115.56 | stock | 74 | 10.31 | 3.74 G | 239.5 | 5.79 | 348.1 | 0.2111 |

**The extra traffic is the right size for the extra wall.** At 64 B per refill the fastest process
reads 188.6 GB in 66 s and the slowest 239.5 GB in 74 s. The extra 50.9 GB against the extra 8 s is
a marginal **6.4 GB/s**, while the average over each run is only 2.9-3.2 GB/s, so this is not a
bandwidth ceiling being reached. It reads as extra latency-bound refill stall. That is an
arithmetic consistency check rather than an attribution: `rockchip_ddr` differencing, which is
calibrated to 0.4% as a between-arm difference, is the instrument that would put a measured DRAM
figure against it, and it has not been run on these arms.

**The placement columns had real range here and did not correlate, which is a stronger null than a
flat one.** `contig_frac` ranges 29.8% at rho -0.119, `mean_run` 123% at -0.154, `l3color_cv` 125%
at -0.091 and `l2color_cv` 128% at -0.070. Page placement varied between these processes, by a lot,
and the wall did not follow it. The guarded reading of a *flat* placement column, that the reset
leaves nothing varying, does not apply: something was varying, and it was not what decided the
level.

**Scope of that null.** `l3color_cv` is a coefficient of variation over color bins of the process's
own sampled pages, at `present_frac` ~0.64, and the L3 is shared with everything else on the SoC.
So it refutes a coloring effect that this statistic can see, over this process's own pages. It
does not refute L3 set pressure arriving from outside the process.

**A second covariate separates the two levels perfectly, and it is too small to be the carrier by
itself.** `a55_inst_share` reads 0.2037-0.2060 on the four fastest processes and 0.2080-0.2115 on
the other eight, with no overlap and a pooled rho of -0.930. Total instructions are constant to
0.4%, so what moves is placement rather than work: the slow processes retire 3.4% more instructions
on the A55s and 1.4% fewer on the A76s, about 5 G instructions, near 1.1% of the total, landing on
the little cluster instead of the big one. At a 3.7% range that column cannot be an additive term
carrying a 12% wall, and the detection floor says so.

Those two columns co-vary across all twelve processes, and an observational design cannot order
them. That is what the next section intervenes on.

### Pinning to the A76s is worth 1.05-1.13x on prefill, and the gain tracks the little cluster's instruction share

The knob is `taskset`, not `ROCKET_CPU_AFFINITY`. `librocketnpu` already pins its own pack and
readback workers to the max-frequency cluster, all four A76s on this part, so the instructions the
readout sees on the little cluster belong to llama.cpp's own ggml threads. Three arms, six rotated
passes, everything else held [HW sweep 2026-09-01, RK1, 600 MHz, governor `performance`,
`qwen35-08b-f16`, memory reset before every arm, 0 failed arms, `pfn_zero_frac`=0.0000 and
`pmu_enabled`=100 on all eighteen processes; raw rows in
`ro-session/trackd-pin-08b-f16-6pass.md`]. `pin76` is `taskset 0xf0` at llama-bench's default
`-t 8`, so eight threads share four cores. `pin76t4` adds `-t 4`, which holds threads-per-core at
the one the unpinned arm has.

| arm | mean t/s | range | spread | paired ratio | A55 inst | A76 inst | L3 refills | wall |
|---|---:|---|---:|---:|---:|---:|---|---:|
| unpinned | 119.93 | 116.16-124.91 | 7.3% | -- | 9.66e10 | 3.654e11 | 3.19-3.76e9 | 71.3 s |
| `pin76` | 132.25 | 127.60-142.24 | 11.0% | **1.103x** | 0.22e10 | 3.666e11 | 3.48-4.20e9 | 65.5 s |
| `pin76t4` | 134.93 | 133.36-139.32 | 4.5% | **1.126x** | 0.20e10 | 3.827e11 | 3.57-3.90e9 | 64.2 s |

The paired ratios carry a per-pass standard deviation of 0.035 and 0.044 over six passes, so both
resolve above 1.00 by about seven standard errors. This is the largest single-flag effect measured
on this unit, and no line of the stack changed to get it.

**The NPU half is identical in every arm, so the whole effect is host-side.** All eighteen
processes report the same engagement line: 126 weights resident on the NPU at 780 MB, 0 streamed.
This unit admits the resident route on stock because its K is inside the default gate, so the
device sees the same program and the same buffers in all three arms.

**And this is not the governor effect.** Every arm ran with all three `cpufreq` policies pinned
to `performance`, so the A76 cluster never parked, and the penalty described in
[../cpu-governor-and-offload.md](../cpu-governor-and-offload.md) is not available to explain any
of it. The 10-13% sits on top of a pinned governor.

**The L3 column inverts across the intervention while surviving inside it.** Within each arm the
absolute `a76_l3d_cache_refill` still tracks the wall at rho **-0.943**, over a real range of 8.5%
to 17.3%, and it does so in the two arms where the little cluster is idle. Pooled over all
eighteen processes its rank correlation is **+0.172**: the faster arm is the one taking more
refills, 4.00e9 against 3.50e9, at a higher rate per second as well as in total. So the column is a
within-arm lottery covariate and not a cross-arm predictor of the level, and the twelve-row
observational design could not have shown that. **The hypothesis that A55 migration drives the
within-arm L3 variation is refuted**: `a55_inst_share` falls 40x under the intervention and the
within-arm correlation is unchanged.

**The mode is not what pinning removes.** `pin76` holds an 11.0% spread, wider than the unpinned
arm's own 7.3% in this campaign, with per-pass ratios from 1.040 to 1.139. `pin76t4` read 4.5%
here and **11.9%** in the second campaign below, on the same arm at the same shape, so the narrow
draw was luck and pinning does not reliably tighten the arm. What the residual lottery is remains
open. The two `-t 8` arms did share a pass-level component the `-t 4` arm did not, their walls
ranking together at rho **+0.829** across the six passes against +0.029 [one campaign, one unit].

**The little cluster's instructions do not migrate, they vanish.** Pinning removes 9.66e10 A55
instructions and 88 A55 core-seconds, and the A76 instruction count does not move to absorb them:
3.666e11 against 3.654e11, +0.3%. Total instructions fall **20%**, from 4.620e11 to 3.688e11, for
the same tokens. The A76 side becomes denser in memory as well, `mem_access` rising 12% and
`l1d_cache_refill` 16% over an unchanged instruction count, which is the arithmetic behind the
higher refill totals. Whether the vanished instructions were a synchronization tax or redundant
work is **not settled here**: the PMU half is system-wide and attributes nothing to a thread, so
this instrument cannot decompose an instruction count by which thread retired it.

**The standing negative does not survive its own cell.** It recorded whole-process
`taskset 0xf0` as no win on Gemma-4-12B F16, with prefill flat and decode 34% down. Decode is
untouched by every arm in every campaign here, all of which are `-n 0`, and giving up four cores
for a bandwidth-bound decode remains a real cost. Prefill is the half that disagreed, and the
third campaign below re-measures that exact model under the rotated-pass protocol: prefill is
**not** flat there. The recorded cell had no raw evidence file and no recorded protocol anywhere
in this workspace, and this board does not settle a sign at one process per arm.

**A second model, at the other end of the matrix and under the recipe the guide recommends.**
`qwen35-9b` Q4_K_M with `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, three rotated passes,
every arm 200 weights resident at 13184 MB and 0 streamed [HW sweep 2026-09-01, RK1, 600 MHz,
governor `performance`, 0 failed arms, both guards clean on all nine; raw in
`ro-session/trackd-pin-9b-qres-3pass.md`]. `pin76` reads **1.059x** (per-pass 1.098 / 1.010 /
1.068) and `pin76t4` **1.062x** (1.089 / 1.052 / 1.045), with all six ratios above 1.00. So the
lever reads across, and it is **smaller at the large end**. At three passes only `pin76t4`
resolves comfortably, at 4.4 standard errors against `pin76`'s 2.3.

**The two models' instruction accounting differs, and a fixed poll budget explains both.** On the
9B the A76 count DOES rise when pinned, 12.159e11 to 12.807e11, absorbing about a third of the
2.00e11 A55 instructions removed. On the 0.8B none of it was absorbed. The unpinned
`a55_inst_share` is also lower on the bigger model, 0.146 against 0.209. ggml's worker spins
about 6.5e6 `yield` rounds before sleeping on a mutex, roughly 7-9 ms
[source-confirmed, `ggml/src/ggml-cpu/ggml-cpu.c`], so a short NPU op is spun through end to end
while a long one exhausts the budget and the thread sleeps. That predicts a smaller lever on the
larger model, which is what both numbers show [hypothesis, and `--poll` is the arm that tests it].

**A symbol profile of the little cluster refutes the spin half of that story.** `perf record -a -e
armv8_cortex_a55/inst_retired/` over one unpinned `qwen35-9b` quant-resident arm, 97524 samples and
none lost, attributes **97.20%** of the A55 instructions to `llama-bench` itself [HW readout
2026-09-02, RK1, `-p 2048 -n 0 -r 3`, arm at 200 resident and 0 streamed; raw in
`ro-session/trackd18-a55-symbols.md`]. The top of the histogram is graph work, not waiting:

| share | symbol |
|---:|---|
| 20.79% | `ggml_compute_forward_gated_delta_net` |
| 20.57% | `ggml_vec_dot_f32` |
| 17.31% | `ggml_compute_forward_ssm_conv` |
| 7.10% | `ggml_compute_forward_rms_norm_mul_fused` |
| 5.48% | `ggml_vec_swiglu_f32` |
| 3.70% | `ggml_vec_silu_f32` |
| 3.43% | `tinyBLAS_Q0_ARM<block_q8_0>::gemm<3,3>` |

`librocketnpu`'s own host symbols come to **0.43%** together (`rocket_pack_activations` 0.27%,
`rocket_unpack_output` 0.16%), which confirms from the other side that the driver's workers are
not what runs there.

**`ggml_barrier` appears nowhere in the histogram, and that is guaranteed by construction rather
than measured.** `libggml-cpu.so` imports `GOMP_barrier`, `GOMP_parallel` and `GOMP_single_start`
[verified on the board, `nm -D`], so this is an OpenMP build, and under `GGML_USE_OPENMP`
`ggml_barrier` is `#pragma omp barrier` [source-confirmed, `ggml/src/ggml-cpu/ggml-cpu.c`]. It
lowers into libgomp and can never be a leaf symbol. **An absent barrier symbol is not evidence
about spinning in this build.**

**The 6.4% perf left unresolved is the OpenMP runtime.** Six addresses clustered at `0x2244x` and
`0x2277x` were read here as `ggml_gated_linear_attn` and `ggml_rwkv_wkv7` in
`libggml-base.so.0.20.2`, by offset and by model context. The same six carry **16.02%** on
`gemma4-12b` F16, a plain transformer with neither op, where a `--sort dso` view names
`libgomp.so.1.0.0` for exactly that total [HW readout 2026-09-02, RK1; raw in
`ro-session/trackd22-a55-symbols-12b.md`]. The `libggml-base` symbols at those offsets are graph
CONSTRUCTORS called once per node per build, not kernels -- the kernel is the histogram's own
20.79% entry. **Offsets agreeing across two shared objects are not attribution**, and the context
that picked between them was true and still selected the wrong object.

**So the term `taskset 0xf0` moves is mostly real graph work with a runtime component, and
`--poll` was never able to reach either.** On this model the two largest entries are the hybrid
attention layers, which have no NPU handler at all and run on the CPU by construction. `--poll 0`
measured 1.013x because the whole polling machinery -- `threadpool->poll`,
`ggml_graph_compute_poll_for_work`, the hybrid poll-then-sleep loop -- sits inside
`#ifndef GGML_USE_OPENMP` and is not compiled into this build [source-confirmed]. The wait policy
that does govern this barrier is libgomp's, through `OMP_WAIT_POLICY` and `GOMP_SPINCOUNT`, and it
has both a spin budget and a sleep fallback.

### The same histogram on a plain transformer, and the runtime term it exposes

`gemma4-12b` F16 unpinned at pp2048, the published 1.046x `pin76` arm, 153932 samples and none
lost, 95.44% in `llama-bench`, the arm reading 19.85 t/s against a published 19.94 mean
[HW readout 2026-09-02, RK1; raw in `ro-session/trackd22-a55-symbols-12b.md`].

| share | symbol |
|---:|---|
| 34.44% | `ggml_compute_forward_glu` |
| 14.38% | `ggml_compute_forward_rms_norm_mul_fused` |
| 12.32% | `ggml_compute_forward_flash_attn_ext_tiled` |
| 6.43% | `ggml_cpu_fp32_to_fp16` |
| 5.10% | `ggml_compute_forward_mul` |
| 1.46% | `ggml_compute_forward_rope_flt<float>` |
| **16.02%** | **`libgomp.so.1.0.0`, six unresolved addresses in two tight runs** |

This is the out-of-sample model the 9B histogram could not stand in for: every large op here has an
NPU handler, so what is left on the CPU is glue, and the glue is 74% of the little cluster. The
driver's own host symbols total **0.91%** on a model that streams every weight and pays the
per-call pack on every call, because the workers are pinned to the big cores [source-confirmed,
`rocket_affinity.c`].

**A cap of 0.7% was derived here and it does not hold.** The derivation was `16.02% of the little
cluster's 4.6% of wall`, taking pinning's 1.046x as the whole A55 contribution. Both halves leak.
The 16.02% is a share of A55 `inst_retired` read as a share of A55 TIME, and a spin loop is
exactly where those diverge. And the 4.6% caps only the cluster pinning REMOVES, while
`OMP_WAIT_POLICY` changes barrier behavior on all eight cores -- so under the recommended pinned
configuration the capped half is zero and the whole term sits outside the cap. **A cap read on one
cluster does not cap a term that lives on all of them.**

#### The A76 histogram, which is what that cap needed

`gemma4-12b` F16 PINNED (`taskset 0xf0`, `-t 4`), otherwise the same arm,
`perf record -a -e armv8_cortex_a76/inst_retired/ -F 499`: **368879 samples, 0 lost**, the arm at
21.19 t/s against the published pinned 21.11, 97.21% inside `llama-bench`
[HW readout 2026-09-02, RK1; raw in `ro-session/trackd24-a76-symbols-12b.md`].

| share of A76 `inst_retired` | object |
|---:|---|
| 32.08% | `libggml-cpu.so` |
| **27.92%** | **`libggml-rocket.so`** |
| 17.13% | `[kernel.kallsyms]` |
| 14.83% | `libm.so.6` |
| 3.05% | `libc.so.6` |
| 2.53% | `libggml-base.so` |
| **2.19%** | **`libgomp.so.1.0.0`** |

**libgomp is 2.19% here, and it is the same four addresses.** `0x22440`, `0x2244c`, `0x2276c` and
`0x22778` carry all of it -- the region the A55 capture resolved to libgomp by its `--sort dso`
view after an earlier session mis-attributed it to `libggml-base` graph constructors. A second
cluster resolving the same region to the same object confirms that correction independently.

**And the term does not shrink under pinning.** Against the whole instruction stream it is flat:
16.02% of A55 instructions at an unpinned `a55_inst_share` of 0.141 is **2.26%**, and 2.19% of A76
instructions at a pinned share of 0.007 is **2.17%**. So the cluster-asymmetry story -- symmetric
pinned A76s should spin less than a mixed cluster pair -- is not what this measures. **What should
be quoted is the instruction share, not a wall cap**: converting it needs a cycles-based capture,
and not all of libgomp is spin, since `GOMP_parallel` and `GOMP_single_start` are real work
distribution that `OMP_WAIT_POLICY` does not remove. The term is small on both clusters, and
`OMP_WAIT_POLICY=passive` is still not worth a wall campaign -- for a reason the earlier cap did
not establish.

**The driver's own host symbols are 27.92% here against 0.91% on the A55s**, which measures
directly the inference the A55 capture could only make: the pack runs on the big cores because the
workers are pinned there. And the attention path dominates the big cores -- `expf` 14.07%,
`ggml_compute_forward_flash_attn_ext_tiled` 12.00%, `fa_mask_scores` 6.38%, `host_softmax_rows`
4.02%, `ggml_backend_rocket_flash_attn` 2.96% and `expf@plt` 0.44%, **39.9% between them**, with a
scalar libm `expf` the single largest symbol in the profile. That is an INSTRUCTION share and so a
lever candidate rather than a cap; pricing it needs the host term's share of wall, which this arm
did not measure. For scale, the weight pack and output unpack that residency and the micro-batch
knobs remove total 9.6% in the same profile.

### The pinning interaction is not a residency effect, and within one model it scales with the knob

The two measured interaction cells were both residency knobs, and the mechanism offered for them --
residency has already removed the A76 host pack work that pinning was accelerating -- predicts a
null on a knob that places nothing. **It does not happen.** `-b 2048 -ub 2048` alone on
`qwen35-9b` Q4_K_M, which places nothing at all, reads an interaction of **0.9775** (per-pass
0.9910 / 0.9903 / 0.9813 / 0.9719 / 0.9538 / 0.9767, sd 0.0138, se 0.0056, **4.0 standard errors
below 1.00** and below it in **6 of 6** passes) [HW sweep 2026-09-02, RK1, 600 MHz, six rotated
passes, 24 of 24 `<!--DATA-->` rows, 0 failed arms, `pfn_zero_frac`=0.0000 and `pmu_enabled`=100 on
all 24, and no arm reports a resident weight; raw in `ro-session/trackd20-pin-ub-2x2-6pass.md`].

| arm | mean t/s | range | paired ratio | per-pass ratios |
|---|---:|---|---:|---|
| `stock_unpin` | 19.09 | 18.94-19.30 | -- | -- |
| `stock_pin` | 20.64 | 20.38-20.94 | 1.082x | 1.075 1.075 1.076 1.085 1.106 1.074 |
| `ub_unpin` | 26.99 | 26.78-27.27 | **1.414x** | 1.405 1.388 1.415 1.417 1.423 1.437 |
| `ub_pin` | 28.53 | 28.42-28.62 | 1.495x | 1.497 1.477 1.494 1.494 1.501 1.507 |

**So every knob measured so far interacts negatively, residency or not.** With three cells a
structure appears that two could not show:

| cell | knob | interaction | pin-gain loss | loss / knob |
|---|---:|---:|---:|---:|
| `gemma4-12b` F16 residency | 6.1 pp | 0.9854 | 1.55 pp | **0.239** |
| `qwen35-9b` `-b 2048 -ub 2048` | 41.4 pp | 0.9775 | 2.41 pp | **0.0583** |
| `qwen35-9b` quant residency | 66.8 pp | 0.9646 | 3.89 pp | **0.0583** |

**On one model the pin-gain loss is 5.83% of the knob's size for both knobs, agreeing to four
decimals**, across a micro-batch knob and a residency knob 1.6x apart in size. `gemma4-12b` F16 is
4.1x that. So the loss looks proportional to the knob **within** a model, with a per-model
constant, rather than constant in either absolute or relative terms. **Two points do not establish
a proportionality**, and the three cells share something the form does not name: every one of these
knobs removes host work. The subsection below tests a knob that does not, and the form does not
survive it.

A third knob on `qwen35-9b` cannot be the test, and the arithmetic says so without a pass. Every
knob left on that model acts on the matmul datapath while its prefill is dequant-bound: `MM_ASYM`
measures **1.3%** there [perf/asymmetric-tile.md], and less at the stock micro-batch, so the form
predicts a 0.08 pp deficit against a 1.4% per-pass spread. **The two knobs that ARE large on that
model are large for one reason, both removing the per-micro-batch dequant**, so a large knob of a
different kind does not exist there.

**What this does to the published column.** Every ratio in this file is an unpinned ratio and is
correct as labelled. A reader who also takes `taskset 0xf0` should expect **1.38x, not 1.41x** from
`-b 2048 -ub 2048` on this model, and 1.61x rather than 1.67x from the residency recipe.

#### A device-tiling knob does not interact, which separates the knob's SIZE from the host work it removes

The form above is that the pin-gain loss is a fixed fraction of the knob's size within a model.
Every cell it was fitted on carries a knob that removes HOST work: two residency knobs and a
micro-batch knob. `ROCKET_MM_ASYM` does not -- it halves Nt so the CBUF fill does more MAC per
pass, at unchanged K-accumulation and output volume -- and on `gemma4-12b` F16 the interaction is
**1.0047** (se 0.0108, per-pass 0.9642 / 0.9888 / 1.0136 / 1.0191 / 1.0409 / 1.0018, below 1.00 in
2 of 6) [HW sweep 2026-09-02, RK1, 600 MHz, two three-pass campaigns pooled, 24 of 24
`<!--DATA-->` rows, 0 failed arms, `pfn_zero_frac`=0.0000 and `pmu_enabled`=100 on all 24, no arm
placing a resident weight; raw in `ro-session/trackd23-asym-pin-12b-3pass.md` and
`ro-session/trackd23b-asym-pin-12b-repeat.md`].

| arm | mean t/s | range | paired ratio | sd | per-pass ratios |
|---|---:|---|---:|---:|---|
| `asym0_unpin` | 18.39 | 18.02-18.90 | -- | -- | -- |
| `asym0_pin` | 19.36 | 19.12-19.55 | 1.0533x | 0.0255 | 1.0795 1.0782 1.0418 1.0481 1.0116 1.0604 |
| `stock_unpin` | 19.96 | 19.81-20.06 | **1.0858x** | 0.0179 | 1.1066 1.0993 1.0870 1.0713 1.0587 1.0920 |
| `stock_pin` | 21.11 | 20.86-21.31 | 1.1485x | 0.0192 | 1.1518 1.1720 1.1478 1.1442 1.1148 1.1600 |

The knob is **8.58 pp** here, larger than the 5.7% recorded from three reps in 2026-07, and it is
the shipping default, so `asym0_*` is the arm that opts out. At that size the form predicts
**0.9805**. The measurement sits **2.24 standard errors above it** and 0.44 below 1.00, which
**disfavours the form and is consistent with the registered rival** -- the recorded mechanism,
that pinning accelerates host work and every knob measured so far removes some of it, predicts a
null for a knob that removes little.

**So the knob's SIZE and the host work it removes were confounded in every earlier cell**, and
this is the first one where they part. On this evidence the loss tracks the host work rather than
the size, and the 0.239 and 0.0583 constants are properties of what those knobs removed rather
than of the models. **The separation is 2.2 se and does not close the question**: at this
contrast's 2.64% per-pass spread, three standard errors on the form-versus-null gap needs **16.6
passes**, and six were bought. What is settled is narrower and still useful -- **the interaction
is not resolved as negative here**, where all three earlier cells were, so a reader who pins
should not discount a device-tiling knob the way they discount a host-work one.

**Budget an interaction's passes from an interaction.** This cell was budgeted at three passes
from the unit's 0.6-0.9% paired-ratio sd, which understated the requirement threefold. An
interaction is a ratio of two ratios and carries all four arms' variance; its own per-pass sd is
**2.64%**, close to what four independent arms of these spreads would give, so the within-pass
pairing that tightens a single ratio buys the interaction almost nothing. The one prior
interaction sd, 1.38% on `qwen35-9b`, is itself 1.9x smaller than this one.

**The knob-off arms are the noisy ones**, 4.8% and 2.2% against 1.3% and 2.1% for the shipping
default, in both pinned and unpinned pairs. No mechanism is offered [hypothesis], and it is why
`pin_base` is the widest ingredient at 2.55%.

**A bigger knob on this model was proposed as the cheap test, and the proposal named the wrong
GGUF.** The 1.257x figure for `-b 2048 -ub 2048` is the `gemma4-12b` **Q4_K_M** row of the flag
table above -- 12.87 t/s stock against this F16 unit's 19.96 -- and the whole `-ub` table is the
quant class. The mechanism recorded for that knob is amortising the per-micro-batch dequant, which
an F16 GGUF does not do. **A published ratio is keyed by the GGUF, not by the model name**, and
`gemma4-12b` names two units here. The `-ub` knob's size on the F16 unit is unmeasured.

The proportionality question is in any case superseded by the subsection below, which derives the
interaction instead of fitting it.

#### The interaction is forced by an additive-time model, and the per-model constants were a curve fit

The form above -- a pin-gain loss proportional to the knob's size, with one constant per model --
was fitted to cells that a two-term cost model predicts outright. Write a prefill's time as a HOST
term the CPU owns plus a REST that CPU affinity cannot touch. Let pinning cut the host term by
some factor, and let the knob remove a fraction of it. Then with

- `a` = 1 - 1/K, K the knob's unpinned paired ratio,
- `b` = 1 - 1/P, P the base pin gain,
- `phi` the fraction of HOST core-seconds the knob removes,

the interaction is

```
    I = (1 - a)(1 - b) / (1 - a - b + b*phi)
```

with no free parameter. The derivation allows the knob to cut the rest as well as the host term --
only `phi`, the HOST fraction, survives into the result -- so **the knob's size and the kind of
work it removes are separated in the algebra**, which is what three campaigns were being bought to
do. `a` and `b` come from the `t/s` rows; `phi` comes from `busy_tot` in the two PINNED arms of
the same 2x2, where all host work is on the A76s. Scored per pass, paired within a pass:

| cell | K | phi | I predicted | I measured | residual | non-timed wall, base vs knob |
|---|---:|---:|---:|---:|---:|---|
| `qwen35-9b` x `-b 2048 -ub 2048` | 1.414 | 0.496 | 0.9772 | 0.9775 | **+0.1 se** | 2.5 s vs 3.6 s |
| `gemma4-12b` F16 x `MM_ASYM`, 6 passes | 1.086 | 0.025 | 1.0022 | 1.0206 | **+1.5 se** | 5.4 s vs 5.5 s |
| `gemma4-12b` F16 x `MM_ASYM`, first 3 | 1.098 | 0.029 | 1.0046 | 0.9889 | **-1.0 se** | 5.3 s vs 5.5 s |
| `qwen35-9b` x quant residency | 1.667 | 0.6645 | 0.9577 | 0.9646 | **+2.0 se** | 2.4 s vs 37.3 s |
| `gemma4-12b` F16 x f16 residency | 1.061 | 0.227 | 0.9885 | 0.9854 | **-0.9 se** | 5.4 s vs 158.8 s |
| `gemma4-12b` Q4_K_M x `-b 2048 -ub 2048` | 1.262 | 0.465 | 0.9849 | 0.9946 | **+2.1 se** | 4.9 s vs 5.6 s |
| `qwen35-9b` x unstacked quant residency, `-r 2` | 1.756 | 0.6727 | 0.9617 | 0.9901 | **+3.0 se** | 2.6 s vs 39 s |
| `gemma4-12b` F16 x `ROCKET_FLASH_ATTN=0` (the knob ADDS host work) | 0.957 | -0.513 | 1.0306 | 0.9690 | **-6.8 se** | 5.3 s vs 5.4 s |

Both residency rows carry the uncontaminated `phi` from the rep-count regressions below (0.6645
and 0.227); scored with the raw ratio the 9B row read -1.1 se and the 12B row -3.5 se. **The seven residuals sum to a chi-square of 22 on seven degrees of freedom** (p about 0.002), with
the four largest host knobs all positive (+2.0, +2.1, +3.0 and +0.1): the model predicts a larger
pin-gain loss than is measured where the knob removes the most host work, by 1-3 pp. **One mechanism
for that is already refuted**: if the removed work's own pinning speedup were what mattered, `phi`
should be weighted by `(1-1/g_r)/(1-1/g)` with `g_r` = the removed core-seconds unpinned over pinned;
on the one clean cell where that differs from `g` (the 9B micro-batch cell, `g_r` 1.123 against `g`
1.198) the variant moves the residual from +0.1 to **-3.3 se**, and on the 12B Q4_K_M cell `g_r` = `g`
and nothing moves [`interaction-refit.py`]. What is left is a term the model does not carry, and
seven cells over three units cannot say which.

**THE EIGHTH CELL IS THE FIRST WHOSE KNOB ADDS HOST WORK, AND THE MODEL GETS ITS SIGN WRONG.**
`ROCKET_FLASH_ATTN=0` on `gemma4-12b` F16 moves the above-gate attention onto the CPU backend
(`flash_attn_ext_tiled` at `-t` threads): `a` = -0.045, `phi` = **-0.513 +- 0.002** -- CPU attention
adds 51% of host core-seconds -- and the model, with both signs flipped, predicts I = 1.031: the
CPU-attention arm should gain MORE from pinning, since it has more host work for pinning to speed
up. It gains less: **I = 0.9690 +- 0.0090, 6.8 se below the model and 3.4 se below the null**,
below 1 in 3 of 3 passes [HW sweep 2026-09-03, RK1, 600 MHz, `performance`, three rotated passes,
12 of 12 `<!--DATA-->` rows, 0 failed arms, `pfn_zero_frac` 0.0000 and `pmu_enabled` 100 on all,
no arm placing a resident weight; raw in `ro-session/trackd32-fa-pin-2x2-12b.md`, driver
`perf/data/trackd32-fa-pin-2x2-12b.sh`; registered 2026-09-03, sign REFUTED]. **The model
carries pinning as a speedup `g` of host work and has no core-count term.** Every earlier cell's host
work is the backend's five-worker pool plus the dispatch thread, which fits in four A76s, so
`taskset 0xf0 -t 4` only ever moved it onto faster cores. CPU attention is an eight-thread
throughput term, and for it the same pinning is four cores instead of eight: a slowdown the
model cannot express with a `g` above 1. **So the model is a model of knobs that remove work from
the worker pool**, seven cells wide, and a knob that adds parallel CPU-backend work is outside it.

**The same cell prices the FA offload at pp2048 today: 1.045 +- 0.004 unpinned and 1.079 +- 0.009
pinned** (FA-on over FA-off, paired within a pass), against the 1.02 the gate's own comment records
from 2026-06-28. The `MIN_KV` = 1024 gate stands. And the swap's own exchange figure, `a/phi` =
0.088, says what a swap's ratio means: removing half the host CPU by computing attention on the part
buys 4.5% of wall, because the work it removes was eight-way parallel and mostly off the critical
path -- which is the number the previous plan proposed to multiply the attention path's share
by, and why it cannot be.

**The `phi` the model wants IS the core-seconds fraction, and the "host wall" reading of the
derivation is what the data refute.** The derivation above writes the host term as a wall `H` that
pinning divides by `g` and the knob removes a fraction of; read literally, that fraction is of host
WALL, and the number fed in is a fraction of host CORE-SECONDS, which coincide only where the removed
work converts to wall at the host average. Closing the wall form needs `H`, and `b = (H/t)(1 - 1/g)`
gives it from the pin gain and `g` = `busy_tot` unpinned over pinned of the base arms, so
`phi_wall = a(1 - 1/g)/b`. Re-fitted that way over the same six cells, with no board time
[`perf/data/interaction-refit.py` over the stored `<!--RO-->` and `<!--DATA-->` rows of trackd16, 20,
23, 23b, 28 and trackd-res12b]:

| cell | `g` | `phi` (core-s) | `phi_wall` | I, core-s `phi` | I, `phi_wall` | measured |
|---|---:|---:|---:|---:|---:|---:|
| `qwen35-9b` x `-b 2048 -ub 2048` | 1.198 | 0.496 | 0.641 | 0.9771 (+0.1 se) | 0.9613 (**+2.9 se**) | 0.9775 |
| `qwen35-9b` x quant residency | 1.217 | 0.6645 | 0.784 | 0.9577 (+2.0 se) | 0.9396 (**+7.2 se**) | 0.9646 |
| `gemma4-12b` F16 x `MM_ASYM`, 3 | 1.285 | 0.029 | 0.317 | 1.0044 (-1.1 se) | 0.9837 (+0.4 se) | 0.9889 |
| `gemma4-12b` F16 x `MM_ASYM`, 3b | 1.288 | 0.025 | 0.400 | 1.0019 (+1.7 se) | 0.9860 (**+3.1 se**) | 1.0206 |
| `gemma4-12b` F16 x f16 residency | 1.300 | 0.227 | 0.220 | 0.9885 (-0.9 se) | 0.9890 (-1.1 se) | 0.9854 |
| `gemma4-12b` Q4_K_M x `-b 2048 -ub 2048` | 1.145 | 0.465 | 0.583 | 0.9849 (+2.1 se) | 0.9782 (**+3.6 se**) | 0.9946 |

The wall form is worse on five of six cells and by 3-7 se on four, all in one direction: every
`phi_wall` above its core-seconds `phi` over-predicts the pin-gain loss. The null `I` = 1, which is
what the substitution `phi_wall` = `a` collapses to, is rejected at -4.0, -10.2 and -4.3 se on the
three cells with the largest knobs. So the quantity the interaction tracks is the host
CORE-SECONDS a knob removes, which is what a throughput reading of the host term predicts --
pinning buys a fixed fraction of every host core-second's wall cost, and removing a fraction of
those core-seconds removes that fraction of the pin gain -- and not the additive host-wall picture
the derivation was told in. The number fed in was right and the story around it was wrong; the
correction the previous plan proposed ("a `phi` in the right unit") is refuted, and its
prediction that the re-fit would leave four of five original cells within 0.5 se and pull the
+2.2 se cell to 0.8-1.8 se missed on both counts [registered 2026-09-03].

**The fit is not one that could not fail.** On the two `qwen35-9b` cells the `b*phi` term carries
the whole effect: drop it and the same algebra predicts **1.072** and **1.048** where the
measurements are 0.965 and 0.978. It swings 10.7 pp on one cell and lands at 0.1 se.

**What it settles.** The deficit is carried by `b*phi` -- the base pin gain times the host work the
knob removes -- and the knob's size enters only through the `(1-a)` terms. `MM_ASYM`'s measured
`phi` of 0.025-0.029 forces `I` = 1.002-1.005 against 1.0047 measured, so **the loss tracks the
host work and not the size**, and the 0.0583 and 0.239 constants are what `b*phi/(K-1)` happens to
equal on those knobs rather than properties of the models. The device-knob cell resolves the kind
question at +0.2 se, not at the 16.6 passes a form-versus-null contrast was priced at.

**The one miss was the instrument, and measuring it closes the model.** `bench-llm.sh` takes
`wall_s` and both `busy_tot` snapshots around the whole `llama-bench` invocation, and a residency
arm pays its one-time ingest inside that bracket: the shell warm-up that runs before the readout
opens is a SEPARATE PROCESS, so the timed process places its weights again. The driver comment
claiming that warm-up "pays this arm's one-time residency ingest OUTSIDE the measured run" is true
of the `t/s` reps and false of every `<!--RO-->` column. The residual tracks the last column
monotonically: the two arms adding no non-timed wall sit at +0.1 and +1.5 se, the one adding 35 s
at -1.1, the one adding 153 s at -3.5. Inverting the model on the two contaminated cells gives an
implied setup cost of **0.49 and 0.74 cores** over their non-timed windows, both physically
sensible; on the two clean cells the same inversion divides noise by a four-second window and
returns nonsense, which is the control.

##### Separating the ingest from the per-rep host work, by regressing on the rep count

`busy_tot = intercept + (r+1)*slope`, because `llama-bench` runs one internal warm-up plus `r`
reps. The **slope** is the per-rep host work and gives an uncontaminated `phi`; the **intercept**
is the model load plus the ingest. `gemma4-12b` F16, PINNED throughout (`PIN_MASK=0xf0 -t 4`,
since `phi` is defined at fixed pinning), streamed against `ROCKET_F16_RESIDENT=auto`, at `-r 1`,
`-r 2` and `-r 4`, three rotated passes [HW sweep 2026-09-02, RK1, 600 MHz, governor
`performance`; 18 of 18 `<!--DATA-->` rows, 0 failed arms, `pfn_zero_frac`=0.0000 and
`pmu_enabled`=100 on all eighteen; raw in `ro-session/trackd25-busy-intercept-12b.md`]:

| configuration | `busy_tot` fit (jiffies) | `wall_s` fit | one rep at the mean `t/s` |
|---|---|---|---|
| streamed | 15967 + **15540**`*r` | 101.2 + 97.1`*r` | 96.8 s |
| `ROCKET_F16_RESIDENT=auto` | 19638 + **12010**`*r` | 199.3 + 100.4`*r` | 92.8 s |

Both `busy_tot` lines are linear to under **0.4%** residual, and the streamed arms reproduce
across passes to 0.03-0.4% at every rep count. The streamed `wall_s` slope of 97.1 s reproduces
2048/21.1 = 96.8 s exactly and its intercept implies one warm-up plus a 5 s model load, which is
the 5.4 s of non-timed wall the earlier campaigns show for a streamed arm. **The resident arm's
wall is NOT linear the same way** -- its non-timed portion moves 65-143 s across the three rep
counts -- while its `busy_tot` is. CPU work scales with reps; wall absorbs a variable wait. That
is the reason to regress the counter and not the clock.

- **`phi` = 1 - 12010/15540 = 0.227 +- 0.018** (per-pass 0.196 / 0.258 / 0.227), against the
  **0.087** the raw `busy_tot` ratio gives and the **0.280 +- 0.060** the interaction requires --
  **0.85 se** from the required value and 7.8 se from the contaminated one.
- **The ingest is 72 core-seconds**, from the intercept difference at one warm-up. Over the 98 s
  of extra fixed wall the resident arms carry that is **0.73 cores**, against the **0.74 cores**
  the model's inversion implied from a different campaign at a different rep count. Two
  independent routes to the same rate.
- **The cell's residual moves from -3.5 se to -0.9 se**, so the forced model fits all five 2x2
  cells within 1.5 se.
- **The control passes**: 9 of 9 resident arms place the identical 286 weights / 18078 MB and 0 of
  9 streamed arms report any, so the regression fits one configuration rather than three.
- **Residency removes 22.7% of the per-prefill host CPU** on this unit and is worth 1.061x of
  `t/s`. Those are the two halves of what the knob buys, measured separately for the first time.

##### The model returns the HOST TERM'S SHARE OF PREFILL WALL, which nothing here had measured

`a = (H/t)*phi` by construction, so **`H/t = a/phi`** -- a knob's paired ratio divided by the
fraction of host CPU it removes is the host term's share of the wall. That is the denominator every
host-side cap in this workspace has been missing, and it costs two quantities both already
collected.

| unit | knob | `a` | `phi` | **`a/phi`** | `q` = `C/(t*a/phi)` |
|---|---|---:|---:|---:|---:|
| `qwen35-9b` Q4_K_M | quant residency, `phi` off the rep-count regression | 0.4003 | 0.6645 | **0.602** | 4.46 |
| `qwen35-9b` Q4_K_M | `-b 2048 -ub 2048` | 0.2928 | 0.4958 | **0.591** | 4.55 |
| `qwen35-9b` Q4_K_M | unstacked quant residency (`ROCKET_QUANT_RESIDENT=auto` at the default `-ub`), `phi` off the rep-count regression, `a` from the same campaign | 0.4306 | 0.6727 | **0.640** | 4.17 |
| `gemma4-12b` Q4_K_M | `-b 2048 -ub 2048` | 0.2075 | 0.4650 | **0.446** | 5.83 |
| `gemma4-12b` F16 | f16 residency, `phi` off the rep-count regression | 0.0578 | 0.227 | **0.255** | 7.73 |

**`a/phi` IS NOT A SHARE OF THE WALL, AND THE PINNED ARM PROVES IT.** `phi` is a fraction of host
CORE-SECONDS and `a` is a fraction of WALL, so `a/phi` = `C/(q*t)` where `C` is the host CPU per
prefill and `q` is core-seconds removed per second of wall gained. Reading it as "the host term's
share of prefill wall" assumes `q` is the host work's parallelism. It is not, and one cell settles
it: on `gemma4-12b` F16 **pinned to four cores**, residency removes **35.3 core-seconds** of host
CPU per prefill and **4.22 +- 0.26 s** of wall, so **`q` = 8.35 +- 0.52 -- 8.4 se above the four
cores the arm is allowed**. No four-core arm retires more than four core-seconds of critical-path
work per second, so **at least 52% of the removed host CPU was overlapped with the device**, which
is what the worker pool is designed to do ("3 cores + 2 to fill pack/read idle bubbles"
[source-confirmed, `ggml-rocket.cpp`]). The unpinned pair does not fire the test (`q` = 7.73
against eight cores), which is why three sessions read the quantity as a share.

**What survives, and it is the useful half.** `q` is a MARGINAL EXCHANGE RATE: how many core-seconds
of host work must be deleted to buy one second of wall, and it is legitimately knob-dependent
because different work overlaps the device by different amounts. **The cap arithmetic is unchanged**
-- a term's cap is (its share of host core-seconds) x `a/phi` -- but it now carries a stated
condition: **the `q` used must be the term's own.** Where the knob IS the term's knob it is, and
the pack cap lands at 5.12% against a measured 5.79%. Where it is imported, say so.

**And the `MM_ASYM` classifier survives with a better reason.** Its `a/phi` of 3.09 was called
"impossible for a share"; under the exchange-rate reading nothing is impossible about it, and what
it says is that the knob bought wall while removing almost no host CPU -- which is precisely a
device-side knob. Same verdict, sound derivation.

**`a/phi` IS A PROPERTY OF NEITHER THE MODEL NOR THE QUANT CLASS.** Three units give 0.255, 0.446
and 0.591-0.602, and the two `gemma4-12b` rows are the same model with two GGUFs. Under the
exchange-rate reading that is expected rather than surprising: `q` runs 4.46 to 7.73 across the
four cells, so the same host core-second buys 1.7x more wall on the 9B than on the 12B F16 unit.
**The prediction that the device term `R` transfers between two GGUFs of one model is a registered
miss** [registered 2026-09-03] and `R` = `t*(1 - a/phi)` is now known not to be a device term
at all, so its 76.4-versus-88.5 s gap is a restatement of the `q` gap rather than a second finding.

**The two 9B rows agree to 1.9%**, down from the 13% the contaminated `phi` gave. They are
**nested** -- the residency arm is the micro-batch arm with residency on top -- so this is a
two-point collinearity test through the origin rather than two independent knobs, and under the
exchange-rate reading what it says is that both knobs remove work with the same `q`.

**The 9B's residency `phi` is measured uncontaminated at 0.6645 +- 0.0015** by the same rep-count
regression the 12B needed, against the **0.598** the raw ratio gave, and that moves its host share
from 0.669 to **0.602** -- **1.9% from the micro-batch row**, where the contaminated pair sat 13%
apart [three passes, 18 of 18 rows, placement 9 of 9 at 200 weights / 13184 MB;
`ro-session/trackd29-busy-intercept-9b.md`]. `g`, pinning's speedup of host work, agreed to 1% on
the contaminated pair and is unchanged by this.

**THE TWO 9B KNOBS ARE NESTED, SO THE CHECK IS WEAKER THAN TWO INDEPENDENT ONES.** The residency
arm IS `ROCKET_QUANT_RESIDENT=auto` **on top of** `-b 2048 -ub 2048` [`ro-session/trackd16-pin9b-2x2-6pass.md`,
header: "the published 1.661x arm"], so both rows carry the micro-batch component and a device-side
term in it would inflate them together where their agreement could not see it. What the pair
constitutes is a **two-point collinearity test through the origin in (`phi`, `a`)**, with no
residual degree of freedom -- and the "incremental knob" reading, residency given `-ub`, is
algebraically the same equation rather than a third point:
`a_inc/phi_inc = (a_q - a_u)/(phi_q - phi_u)` equals `a_u/phi_u` exactly when `a_q/phi_q` does. A
third, non-nested knob on this unit is what would give the first genuine residual, and it now exists.

**THE NON-NESTED THIRD POINT SITS ABOVE THE NESTED LINE AT 4 se, AND ITS `phi` WAS DERIVABLE FROM THE
OTHER TWO CELLS TO 0.3%.** Unstacked quant residency -- `ROCKET_QUANT_RESIDENT=auto` at the DEFAULT
`-ub`, the recipe this file recommends -- measured with its own unpinned `a` rather than an imported
one [HW sweep 2026-09-03, RK1, 600 MHz, `performance`; six pinned arms at `-r 1/2/4` for the
rep-count regression plus two unpinned arms at `-r 2`, three rotated passes, 24 of 24 `<!--DATA-->`
rows, 0 failed arms, `pfn_zero_frac` 0.0000 and `pmu_enabled` 100 on all, 12 of 12 resident arms
at 200 weights / 13184 MB and 0 of 12 stock arms placing anything; raw in
`ro-session/trackd31-qres512-9b.md`, driver `perf/data/trackd31-qres512-9b.sh`]:

- **`phi_u` = 0.6727 +- 0.0015** (per pass 0.6749 / 0.6697 / 0.6734; A76-only 0.6782). Stock at
  `-ub 512` runs four micro-batches per 2048-token prefill and re-dequantizes and re-packs every
  weight in each, so write `D` for those four passes and `R` for whatever else `-ub 2048` changes in
  host work: the `-ub` cell removes 0.75 D + R = 0.4958 C_s and the stacked cell D + R = 0.6645 C_s,
  so **D = 0.675 C_s and R = -0.010 C_s**, and unstacked residency removes D alone. The prediction
  was 0.675 [registered 2026-09-03]; the rival "the same dequant and pack as the stacked knob"
  (0.6645) is 5.5 se away. **Four dequant+pack passes are two thirds of this unit's stock host CPU,
  and `-ub 2048` adds about 1% of host work on top of what it removes.**
- **Ingest 57.3 +- 1.5 core-seconds** = **4.35 ms per resident MB**, against 4.53 on the stacked
  route (4% apart) and 3.98 on the f16 route. Per-MB transfers a second time.
- **`K_u` = 1.7563 +- 0.0201** unpinned (the published unstacked row read 1.752), so `a` = 0.4306,
  **`a/phi` = 0.640** and **`q` = 4.17** against the nested pair's 4.46-4.55. On the line
  a = 0.602 phi the third point would sit at a = 0.405; it sits at 0.4306, **+0.026 in `a`, 4 se**
  -- so the two published 9B rows' agreement was real and their shared `-ub` component carries a wall
  COST of about 2.6% of the stock wall on the resident route, which is the 1.756/1.661 margin by
  which unstacked beats stacked here, now with a mechanism: it is not that stacking removes less host
  work (it removes slightly more), it is that the larger micro-batch costs wall elsewhere.
- **The pinned/unpinned `-r 2` pairs are a seventh interaction cell**: P = 1.0937, I = **0.9901 +-
  0.0093**, against the model's 0.9617 at this `phi` -- **+3.0 se**, the largest residual of the seven
  and positive like the other large host knobs (the table above).

##### `phi` is a core-seconds fraction over heterogeneous cores, so read it on the A76s

The derivation pairs an UNPINNED `a` with a PINNED `phi`, which is licensed only if `phi` is the
same in both pinning states. Over all eight cores it is not [6 rotated passes each,
`ro-session/trackd20-pin-ub-2x2-6pass.md` and `trackd16-pin9b-2x2-6pass.md`]:

| cell | `phi`, all 8 cores | `phi`, A76s only |
|---|---|---|
| `qwen35-9b` x `-b 2048 -ub 2048` (places nothing, so uncontaminated) | 0.4647 unpinned, 0.4958 pinned, **+6.7%** | 0.5063 unpinned, 0.4930 pinned, **-2.6%** |
| `qwen35-9b` x quant residency (carries the ingest) | 0.5558 unpinned, 0.5983 pinned, **+7.6%** | 0.6188 unpinned, 0.5994 pinned, **-3.1%** |

The mechanism is in a column the readout already prints, and a third cell says it is the column to
read rather than a fixed cluster rule:

| cell | `busy_little_share`, base -> knob | all-cores bias, pinned against unpinned |
|---|---|---:|
| `gemma4-12b` Q4_K_M x `-b 2048 -ub 2048` | 0.140 -> 0.152, **1.2 pp** | **+0.0%** |
| `qwen35-9b` x `-b 2048 -ub 2048` | 0.202 -> 0.266, **6.4 pp** | **+6.7%** |
| `qwen35-9b` x quant residency | 0.201 -> 0.316, **11.5 pp** | **+7.6%** |

**The bias tracks how far the knob shifts work between clusters, and where it does not shift, the
all-cores reading is faithful** -- on the 12B Q4_K_M cell it is the A76-only reading that sits 1.7%
away instead. So the rule is to read `busy_little_share` in both arms, not to apply a cluster
rule blind. **The pinned arms are ~99% A76 either way** (pinned all-core and pinned A76-only `phi`
agree to 0.6%, 0.2% and 0.1% on the three cells), **so the published estimator is already using the
invariant quantity and every number in the table above stands.** What this names is the trap:
**computing `phi` from an UNPINNED arm's `busy_tot` can break the derivation by 7%**, in the
direction that inflates the host share, and the readout says in advance whether it will. Taking
the worst cell as the floor, **`H/t` is good to about 3%, not to three digits** -- which is what
makes the 13% gap between the 9B's two knobs a real target and its residual 1.9% a closure.

**AND THE HOST SHARE IS A SHARE OF WHICHEVER WALL `a` WAS MEASURED ON.** Evaluated all-unpinned,
all-pinned, and as the mixed estimator above, the same two cells read **0.720 / 0.632 / 0.669**
and **0.630 / 0.558 / 0.591**. The campaign numbers in this file are unpinned, so the unpinned
share is the one that prices a lever on them and the pinned one prices a lever on a pinned run.
**The ratio between the two knobs is invariant to the choice** (1.132 / 1.133 / 1.143).

**It only holds for a knob that removes HOST work alone.** `MM_ASYM` gives `a/phi` = **3.09**,
which is impossible for a share, and that is the tell rather than a defect: `a = a_H + a_R` and a
device-tiling knob has `a_R` > 0. **A ratio above 1 here says the knob touched the device**, which
makes this a cheap classifier as well as a measurement.

**What it prices, now that the other half is measured.** A host-side lever on this unit is capped
at 0.255 times its share of host TIME, and host time is A76 CYCLES rather than retired
instructions. Recording both events in one capture gives the conversion
[`ro-session/trackd27-a76-ipc-12b.md`], and it inverts the ranking the instruction histogram gave:

| term | share of non-idle A76 cycles | share of instructions | IPC | cap on prefill wall |
|---|---:|---:|---:|---:|
| host weight pack, `mm_pack_weights` + `_seg` | **23.05%** | 4.59% | **0.26** | **5.88%** |
| attention path, 6 symbols | **22.93%** | 40.68% | 2.35 | **5.85%** |
| `libc` string and memory routines | 9.70% | 2.93% | 0.40 | 2.47% |
| `expf` alone | 7.72% | 14.13% | 2.42 | 1.97% |
| output unpack, 2 symbols | 0.97% | 1.76% | 2.41 | 0.25% |
| `libgomp`, four addresses | 0.93% | 2.26% | **3.20** | **0.24%** |

**The cap is validated by a lever already in this file.** Residency removes 87% of that weight
pack and is worth **1.0614x**, which is 5.79% of the streamed wall; the cycles cap over the same
two symbols is **5.12%**, so the enumerated pair accounts for 88% of a measured gain and the
remainder is kernel-side per-call BO work. The instruction cap over the same pair is **1.02%**,
**5.7x below a gain that is on record**, which is arithmetically impossible for a cap. **Find a
lever that already removes part of a term and check the cap exceeds its measured gain, before
quoting any share of a profile as a cap.**

**The attention row's cap is now measured directly and the imported rate under-stated it 3.9x.**
Every FA knob is a swap -- attention is computed on the CPU instead -- so no knob can produce the
term's own exchange rate. What can be read is the handler's critical-path interval: it runs on the
single backend dispatch thread while the scheduler waits, and `ROCKET_FA_TIMING=1` brackets it
(gather of the strided Q/K/V/mask views into dense fp16, the worker fan-out `compute`, the F32
scatter). On the same pinned streamed arm, one process, `-p 2048 -n 0 -r 2`, over 432 offloaded ops
at `n_kv` 1024-2048 [HW readout 2026-09-03, RK1; `ro-session/trackd30-fa-timing-12b.md`;
driver `perf/data/trackd30-fa-timing-12b.sh`; `llama-bench -v` on both arms, because the probe's
summary is a `GGML_LOG_INFO` line that the bench otherwise swallows]:

| segment | per 2048-token prefill | share of the 97.8 s pinned prefill wall | threading |
|---|---:|---:|---|
| gather | 3.46 s | **3.5%** | single dispatch thread |
| compute (batched QK on the NPU, host mask + softmax on the workers, batched AV) | 16.99 s | **17.4%** | five workers, device interleaved |
| scatter | 1.98 s | **2.0%** | single dispatch thread |
| handler total | 22.43 s | **22.9%** | |

The probe arm read 20.93 t/s against the plain arm's 21.24, one process each, inside the
per-process lottery. **The single-threaded gather and scatter are 5.6% of the pinned wall, and their
exchange rate is 1 by construction**: a core-second on the dispatch thread while everything waits
is a second of wall, so threading them over `k` cores is capped at 5.6% x (1 - 1/k), with nothing
imported.

**THAT LEVER IS BUILT AND SPENT, AND IT PAID 89% OF ITS CAP.** `ROCKET_FA_THREADS=k` splits all
five walks over the host pool, and at `k`=4 on this unit `G_k` falls 5.80 -> 3.17 -> 1.87 s per
prefill at 1 / 2 / 4 workers, a wall ratio of **1.0231** and **1.0389** [HW sweep 2026-09-07,
three arms x three passes, rotated and paired within a pass, 9 of 9 rows;
`ro-session/trackd33-fathreads-12b.md`]. Two instruments agree on the size: the interval says
4.07% of a 96.4 s prefill and llama-bench's own throughput says 3.75%, closing to 0.32 percentage
points, which is what an exchange rate of 1 predicts. **Fitting `G_k` = `A`/`k` + `B` to the first
two points gives `A` = 5.25 s parallel and `B` = 0.55 s fixed, and predicts the third to 0.7%** --
so the ceiling is `B` = 0.57% of wall, four workers already take 89% of what any number would, and
**there is no further lever in these walks**. The compute segment does not move across the arms
(16.76 s, spread 2.1%), and `G_1` re-measured on the newer `.so` is 5.80 s rather than the 5.44
above, so quote the ratio rather than the absolute. That is 1.8x what the handler symbol's own 1.92% of cycles (3.0 core-s) would give, so
2.4 s of the 5.4 sit outside the symbol -- in the kernel's two unnamed entries (6.77% + 5.91%) or
in `libc`'s 9.70% at IPC 0.40, which is where a strided copy lands, and leaf attribution cannot say
which. **The compute interval bounds the host softmax's wall at 17.4%** and cannot split it from the
device's QK/AV time; `expf` at 7.72% of cycles is 12 core-s per prefill inside it, so the vectorised-exp
question stays open with a cap somewhere between the borrowed 1.97% and the compute interval, and
the next instrument is a caller split rather than another share. The interval is of the PINNED wall
and one shape; the below-gate CPU attention (`flash_attn_ext_tiled`, 6.72% of cycles at `-t 4`) is
outside the bracket.

**The leftover 13% checks from the other end.** Residency reaches 87% of the pack, so the streamed
remainder is capped at **0.76%** of wall, against the **0.9%** the same remainder reads when it is
derived from measured walls (42 weights, 2712 of 20790 MB). Two routes to a term a profile bucket
had priced at over a quarter of the wall.

##### The dequant is half as parallel as the rest of the host half

Two `busy_tot` columns and one measured host share give the core count each part of the host half
actually uses, on one model across two GGUFs at `-p 2048` unpinned [`ro-session/trackd-res12b-2x2-3pass.md`
and `trackd28-pinub12bq4/`, stock arms]:

| unit | host CPU per 2048-token prefill | prefill wall |
|---|---:|---:|
| `gemma4-12b` F16 | 201.9 core-s | 102.49 s |
| `gemma4-12b` Q4_K_M | 415.5 core-s | 158.76 s |

The difference between the units is **213.5 core-seconds over 56.3 s of extra wall**, so a
core-second of the quant route's added host work buys wall at **3.79 core-seconds per second** --
against **7.73** for the whole of the F16 unit's host CPU measured the same way. **A quant
prefill's added work converts to wall more than twice as efficiently as the work it is added to**,
either because it is less overlapped with the device or because it is less parallel, which this
pair cannot separate. It is consistent with the quant unit's `a76_ipc` reading 2.12 against the
F16 unit's 1.27: what it adds is arithmetic.
[HW readout 2026-09-03, RK1; the F16 side is three passes, the Q4_K_M side one.]

**AND TWO UNITS IN THE FLAG TABLE ALREADY PUT `a` BELOW ZERO.** `smolvlm2` reads 0.941x on
`-b 2048 -ub 2048` and `qwen3-30b-a3b` reads 0.908x, so `a` < 0 and `a/phi` cannot be a share
there at all. That is the same classifier firing from the other end of the range, and it bounds
the derivation's generality without a run.

**What this does NOT settle.** `q` on any model outside these two, one shape. The `phi` gap between
0.09 and 0.465 still has no uncontaminated cell in it, and every other `<!--RO-->` column of a
residency arm -- `a55_inst_share`, `a76_ipc`, every `_pki` density -- is computed over the same
bracket and mixes the ingest with steady state in the same way.

### `ROCKET_N_THREADS` is a gain on the MoE route and a loss on the dense f16 one, and the gain is a step at six

#### Six workers buy nothing on the dense f16 route at `-p 2048`, and the limit is why

The `-p 512` ladder reads six as the peak on this route: 293 weights against five's 286, with the
route declining on the NPU IOVA window and a real `ROCKET_CREATE_BO` ENOSPC beside it
[ro-session/trackd19-f16-window-ladder.md]. **That does not transfer to the shape every campaign
number in this file is taken at.** At `-p 2048` the KV cache lowers MemAvailable and the RESERVE
FLOOR binds instead, and a worker count buys file descriptors, which relieve the window and do
nothing to a memory floor [HW readout 2026-09-03, RK1, 600 MHz, `gemma4-12b` F16,
`ROCKET_F16_RESIDENT=auto`, `-p 2048 -n 0 -r 1`, all four arms rc=0; raw in
`ro-session/trackd26-f16-window-p2048.md`]:

| arm | placed | MB | MemAvailable at the decline | limit named |
|---|---:|---:|---:|---|
| `t5_a` | 286 | 18078 | 9401 MB | reserve floor (9535 MB) |
| `t5_b` | 286 | 18078 | 9395 MB | reserve floor |
| `t6_a` | **284** | 17853 | 9527 MB | reserve floor |
| `t6_b` | **284** | 17853 | 9532 MB | reserve floor |

Both configurations reproduce to the byte. **Six is 2 weights WORSE here**, not better: more
workers cost a little more memory, so the floor catches marginally sooner. So the recommendation
of six on the dense f16 route rests on the `-p 512` ladder alone. Its wall was never measured on
this route and cannot be -- the inferred cap is 0.15% of prefill, below this instrument's
resolution -- so the cap is the result and a campaign is not the follow-up.

**And read the reason string as a race, not a diagnosis, whenever the byte count matches on both
sides.** The `trackd25` regression's PINNED `-t 4` residency arms, same model and same `-p 2048`,
named the **IOVA window** at the identical 18078 MB where these default-thread arms name the
floor. The two limits have converged at that point; which one is announced moves with
MemAvailable. That is why the readout runs `t5` twice.


The knob's only measured price was **0.9873x** of prefill wall on `gemma4-12b` F16 at pp2048, with
`a55_inst_share` rising 0.1410 to 0.1498. On `gpt-oss-20b` MXFP4 the same step measures
**1.0274x**, per-pass 1.0289 / 1.0287 / 1.0247, se 0.0014, **20.1 standard errors above 1.00** and
above it in 3 of 3 passes [HW sweep 2026-09-02, RK1, 600 MHz, `-b 2048 -ub 2048`, three rotated
passes, 6 of 6 `<!--DATA-->` rows, 0 failed arms, `pfn_zero_frac`=0.0000 and `pmu_enabled`=100 on
all six; raw in `ro-session/trackd17c-moe-nthreads-cost.md`].

| arm | mean t/s | range | placement | `a55_inst_share` |
|---|---:|---|---|---:|
| nt5 | 28.80 | 28.75-28.89 | 63 stacks, 1710 experts, 13793 MB, 0 streamed | 0.1194-0.1206 |
| nt8 | 29.59 | 29.46-29.72 | 63 stacks, 1713 experts, 13978 MB, 0 streamed | 0.1116-0.1161 |

**The pre-flight admits the identical 63 stacks in all six rows**, so the contrast is the worker
count and nothing else. And **the mechanism inverts with the sign**: the little cluster's share
FALLS here, with no overlap between the arms' three-pass ranges, where on the dense route it rose.
More fds fan the expert GEMMs wider, which moves host work onto the big cores rather than
oversubscribing them.

**So a number measured on one route is not a prior for another**, and this is the row that shows
even the sign failing to carry. **One process would have reported the opposite**: a single `-r 1`
pair read 24.54 against 28.36 t/s, a 0.87x, which three rotated passes turned into 1.027x.

**The shape between five and eight is a step at the first worker above the default**, not a ramp
and not a step at the count that balances the cores. Four arms, ratios paired within a pass
against nt5, three rotated passes [HW sweep 2026-09-02, RK1, 600 MHz, `-b 2048 -ub 2048`, 12 of 12
`<!--DATA-->` rows, 0 failed arms, `pfn_zero_frac`=0.0000 and `pmu_enabled`=100 on all twelve, and
the pre-flight admits an identical 63 stacks / 24529 MB RAM / 15946 MB IOVA and announces `Bound
by RAM` in every one; raw in `ro-session/trackd21-moe-nthreads-shape.md`].

| arm | mean t/s | paired ratio | se | per-pass ratios | `a55_inst_share` | `a76_ipc` | migrations |
|---|---:|---:|---:|---|---|---:|---:|
| nt5 | 28.94 | -- | -- | -- | 0.1200-0.1218 | 2.395 | 4.64e4 |
| nt6 | 29.58 | **1.0222x** | 0.0036 | 1.0203 1.0173 1.0291 | 0.1140-0.1163 | 2.363 | 5.48e4 |
| nt7 | 29.41 | **1.0163x** | 0.0020 | 1.0127 1.0163 1.0198 | 0.1143-0.1162 | 2.328 | 6.26e4 |
| nt8 | 29.47 | **1.0182x** | 0.0034 | 1.0158 1.0249 1.0139 | 0.1142-0.1159 | 2.316 | 7.24e4 |

All three are resolved above 1.00, at 6.2, 8.2 and 5.4 standard errors, and **none is separated
from another**. Paired within a pass, nt6 over nt8 is 1.0040 at 0.6 se, nt7 over nt8 is 0.9981 at
0.5 se, and nt6 over nt7 is 1.0059 at 2.4 se, which is weak across three comparisons. **The whole
of the available gain is bought by the first worker added**, and the three counts above the
default are one level.

**The readout column carries the same shape.** `a55_inst_share` falls from 0.1200-0.1218 to
0.1140-0.1163 at six and does not move again: nt5's three-pass range overlaps none of the other
three, and those three overlap each other completely. The cost of oversubscription does keep
growing, monotonically in the worker count -- `cpu_migrations` 4.64e4 to 7.24e4, `a76_ipc` 2.395
to 2.316. A saturating benefit against a monotone cost is what a flat curve looks like, and it is
a mechanism the wall alone could not have named [hypothesis].

**The five-to-eight endpoint did not reproduce.** The same cell read 1.0274x (se 0.0014) the day
before and 1.0182x (se 0.0034) here, 2.5 standard errors of the combined error apart, on a ratio
that stood 20 se above 1.00 inside its own campaign. What differs is that two arms now sit between
the paired ones, so a pair's members are separated by up to three arms and half an hour instead of
being adjacent [hypothesis]. The sign is unaffected and the size is not: **quote this step as
1.6-2.2%, not as 2.7%.**

**What this settles about the default.** Six is the only value with a measured gain on one route
and no measured loss on the other. It buys the MoE route's whole two percent, and on the dense f16
route it is the placement peak, 293 weights against five's 286, whose wall difference is 0.15% of
prefill and below this instrument's resolution. Eight buys nothing further on the MoE route and
costs 0.9873x on the dense one. **Seven is worse than six on the MoE route and worse than five on
the f16 one**, and it is a setting a reader following "raise it" passes through. This is one model
per route on one board, and the mechanism is a worker count against a big-core count, so none of
it ports to a part with a different one.

**A third model, and it is the one the standing negative was recorded on.** `gemma4-12b` F16,
the 22.18 GiB GGUF, three rotated passes [HW sweep 2026-09-01, RK1, 600 MHz, governor
`performance`, memory reset before every arm, 0 failed arms, 9 of 9 `<!--DATA-->` rows,
`pfn_zero_frac`=0.0000 and `pmu_enabled`=100 on all nine; raw in
`ro-session/trackd-pin12b-f16-3pass.md`].

| arm | mean t/s | range | spread | paired ratio | ratio sd | A55 inst | A76 inst | L3 refills |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| unpinned | 19.94 | 19.85-20.04 | 1.0% | -- | -- | 2.621e11 | 1.596e12 | 1.910e10 |
| `pin76` | 20.85 | 20.78-20.93 | 0.7% | **1.046x** | 0.9% | 0.109e11 | 1.679e12 | 2.033e10 |
| `pin76t4` | 21.17 | 21.12-21.26 | 0.7% | **1.062x** | 0.6% | 0.106e11 | 1.710e12 | 1.996e10 |

`pin76` is the faithful reproduction of the recorded arm, plain `taskset 0xf0` at llama-bench's
default `-t 8`. It reads 1.046x, nine standard errors above 1.00, with all three per-pass ratios
above it, and `pin76t4` reads 1.062x at seventeen.

**And the mechanism offered for the disagreement is refuted rather than confirmed.** The reason to
expect a null here was that this model streams its weights and is bandwidth-bound, so four cores
would issue fewer outstanding misses than eight. The streaming half is real and now measured
rather than inferred: the stock arm residents **zero** weights, read from a warm-up-sized process
with `ROCKET_LOG_STDERR=1` and stderr kept, because an absent `[f16-resident]` line is not a zero
on its own. The conclusion drawn from it is wrong. Streamed against resident is not what separates
a cell that gains from one that does not, because **no measured cell fails to gain**.

**What does order the three is a column the readout already emits.**

| model | config | resident | unpinned `a55_inst_share` | `pin76` | `pin76t4` |
|---|---|---|---:|---:|---:|
| `qwen35-08b-f16` | f16 stock | 126 weights, 780 MB | 0.209 | **1.103x** | **1.126x** |
| `qwen35-9b` | quant-resident | 200 weights, 13184 MB | 0.146 | **1.059x** | **1.062x** |
| `gemma4-12b` F16 | streamed | 0 weights | 0.141 | **1.046x** | **1.062x** |

The unpinned `a55_inst_share` rank-orders the `pin76` gain exactly across all three and the
`pin76t4` gain without inversion, and the wall returned is between a third and a half of the share
(0.49, 0.40, 0.33). So the lever is predictable per model from one warm-up-sized process instead of
from a campaign: read the share first, and expect roughly 33-49% of it back. **Three points, and
the share co-varies with model size**, so this orders the models without separating those two; it
is a usable predictor, not a mechanism. Under either pinned arm the share collapses to 0.0053-0.0064
on all three, a near-constant residual on a system-wide instrument that attributes nothing to a
thread.

**The L3 inversion replicates on the third model.** Absolute `a76_l3d_cache_refill` is higher in
the faster pinned arm on every model measured: 3.50e9 to 4.00e9 on the 0.8B, 8.45e9 to 9.73e9 on
the 9B, and 1.910e10 to 2.033e10 here. The pooled-positive direction is not one campaign's
accident, and the retired reading of the twelve observational rows stays retired.

**The pass-count bound is a property of the unit, not of the class, and it runs the counterintuitive
way.** This unit's per-pass paired-ratio sd is **0.6-0.9%** against the 0.8B class's ~6.5%, and its
arm spreads are 0.7-1.0% against 7.3-11.0%. A 12B F16 pass costs about five times the wall of a
0.8B one and the per-process lottery does not grow with it, so **the slow model resolves a small
knob better than the fast one does**: three passes here resolve a 4.6% lever at nine standard
errors, where six passes on the 0.8B resolve a 10.3% one at seven. Budget passes from the unit's
own ratio sd, not from a class bound imported from a different unit -- including from the same
model's quant unit, whose 6.3% is eight times this one's.

**This board CAN hold a 12B F16 mostly resident, which was believed impossible.** With
`ROCKET_F16_RESIDENT=auto` the same GGUF residents **286 of 328 weights at 18078 MB, 87% resident**,
42 streamed via the per-call pack [HW readout 2026-09-01, RK1, `f16-resident-readout.sh`]. Admission
stops at 18078 MB against a 21127 MB budget and the teardown names the reason: **the NPU IOVA window
filled**, not RAM and not the reserve floor. So "Gemma-4-12B F16 cannot be held resident on this
board" is retired. It also means the arm that separates streamed from resident without changing the
model is available on this one: the same three arms with `ROCKET_F16_RESIDENT=auto` added is a 2x2
in pinning and residency on a single GGUF.

**One consequence for the published numbers.** The CPU-versus-NPU multiples in
[../benchmarks.md](../benchmarks.md) take both arms unpinned, and pinning moves the two arms
opposite ways: it adds 10-13% to an offloading prefill and takes bandwidth away from a CPU-only
decode. So those multiples are conservative for the NPU rather than flattering. That is an
inference from these arms and not a measured re-run [expected].

**And a spread claim that did not replicate, recorded because it was nearly written up as a
lead.** `pin76t4`'s own arm spread read 4.5% in the first campaign against the unpinned arm's
7.3%, which looked like a protocol lever for the 0.8B class's pass-count bound. The second
campaign read **11.9%** on the same arm at the same shape, driven by one 1.292 pass. **Pinning
does not reliably tighten the arm.** Two draws of six is what it took to see that, which is the
same pass-count bound talking.

**Pinning BOTH arms halves the paired ratio's spread, and the reason is not that the arms get
quieter.** The 6.5% per-pass paired-ratio sd on the 0.8B class is what bounds every knob in this
matrix, so the same two-arm unit was run twice, six passes each: once with both arms unpinned and
once with `PIN_MASK=0xf0 -t 4` applied to both, which makes pinning a constant within a campaign
rather than a difference between arms. The unit is `qwen35-08b-f16` stock against
`ROCKET_F16_RESIDENT=auto`, the one that straddles [HW sweep 2026-09-01, RK1, 600 MHz, governor
`performance`, memory reset before every arm, 0 failed arms, 12 of 12 `<!--DATA-->` rows per
campaign, both guards clean on all twenty-four, and every arm at its expected split -- 126 resident
stock and 150 under the knob, 0 streamed, twelve of each; raw in
`ro-session/trackd-pinsd-unpinned-6pass.md` and `trackd-pinsd-pinned-6pass.md`].

| campaign | ratio mean | ratio sd | ratio se | stock cv | `f16res` cv | within-pass r |
|---|---:|---:|---:|---:|---:|---:|
| both arms unpinned | 1.064 | **7.2%** | 2.9% | 4.0% | 4.3% | **-0.634** |
| both arms pinned | 0.999 | **3.3%** | 1.3% | 4.8% | 4.5% | **+0.748** |

**The arms are equally noisy in both campaigns.** Each arm's own coefficient of variation sits at
4.0-4.8% either way, so the pinned campaign did not produce steadier processes. What changed is the
sign of the correlation between the two arms inside a pass. Against the 5.9% that two independent
arms of those coefficients would give, the unpinned pairing delivers **7.2%** -- pairing within a
pass makes that campaign WORSE than not pairing at all -- while the pinned pairing delivers 3.3%
against an independent 6.6%, removing 3.4 points.

**The unpinned structure is bimodal within a pass, and it is the recorded two-level mode.** The
twelve unpinned processes fall into a slow group at 116.3-117.8 t/s and a fast one at 123.2-131.8
with a clear gap, and **every one of the six passes holds exactly one arm from each** (under an
independent assignment of six slow processes to twelve slots that happens about 7% of the time).
Those two levels reproduce the modes recorded from the observational rows, 128.74 and 118.13, to
about 1%. The pinned campaign shows 2 of 6 and no gap. **Position within a pass does not explain
it**: first-over-second reads +3.6% unpinned and -1.3% pinned, 1.0 and 1.1 standard errors, neither
resolved, so the arm-order rotation is working and is not the cause.

**The consequence for anything measured this way.** When exactly one arm per pass draws the fast
mode, the paired ratio reports which arm won the draw rather than what the knob did, and no number
of passes fixes a pairing that is anti-correlated -- it converges on the average of the draw. That
is why this unit stayed unresolved across thirteen unpinned passes at about 1.04x, and why under
pinning it resolves in six as a **null**: 0.999, two standard errors 0.972-1.025. **Pinning both
arms is a protocol lever worth roughly four times the passes** for a given resolution on this unit.

**Which published rows this puts at risk, audited over this file's own raw rows: one, and it is
already reported unresolved.** The within-pass correlation was recomputed for every unit here that
carries per-pass `<!--DATA-->` rows. **Most of that audit is uninformative and says so**: every unit
except one ran at three passes, and a correlation over three points is as degenerate as the rank
correlation `ro-join.py` already refuses to quote under `--rho-n`. What the audit can read without a
correlation is the ratio sd.

| unit class | ratio sd | knob |
|---|---:|---|
| `qwen35-08b-f16`, the affected unit | **7.7%** over seven passes and **7.2%** over this session's six | 1.02-1.06x |
| every other unit in this file | **0.1-2.1%** | 1.04-1.66x |

**A ratio sd of 0.1-2.1% cannot host a mode that moves one arm by 9%**, so the bimodal draw that
damaged the 0.8B F16 unit is not present in the units carrying the published knobs, whatever their
correlation sign. On the affected unit the anti-correlation **replicates across two independent
campaigns**, -0.57 over the older seven passes and -0.634 over this session's six. So the defect is
confined to the one unit whose knob is smaller than its own spread, which this file already reports
as unresolved rather than as a number.

**And it changes the question, which is the cost of taking it.** A knob measured with both arms
pinned is the knob's value UNDER pinning, and this matrix is published unpinned. On this unit the
two readings differ -- about 1.04x unpinned against a resolved 1.00x pinned -- but the unpinned
estimate straddles, so whether that is a true interaction between the two host-side levers or the
unpinned figure being noise around 1.00 is **not separated here**. What is settled is narrower and
still useful: the pinned protocol answers this unit's question in six passes and the unpinned one
does not answer it in thirteen. One unit, one knob, six passes per campaign, and neither
correlation resolves on its own at that n.

**No flag captures the lever, and the arm that looked like it would is closed.** ggml's worker
polls `1024*128*poll` rounds of `yield` before sleeping, and llama.cpp's `--poll` sets that
budget, so `--poll 0` looked like the same win with no cores given up. It is not
[HW sweep 2026-09-01, RK1, `qwen35-08b-f16`, six rotated passes over four arms, 0 failed arms,
both guards clean on all twenty-four; raw in `ro-session/trackd-poll-08b-f16-6pass.md`]:

| arm | mean t/s | paired ratio | se | A55 instructions | context switches |
|---|---:|---:|---:|---:|---:|
| unpinned | 118.98 | -- | -- | 9.699e10 | 928 385 |
| `--poll 0` | 120.27 | **1.013x** | 0.028 | 9.665e10 | 928 820 |
| `taskset 0xf0 -t 4` | 138.47 | **1.165x** | 0.026 | 0.254e10 | 839 179 |
| both | 136.36 | **1.146x** | 0.008 | 0.247e10 | 838 796 |

`--poll 0` moves the wall 0.44 standard errors, the A55 instruction count 0.35%, and the context
switches 0.05%. Nothing traded spinning for sleeping. **The flag is not inert** -- `llama-bench`
sets `tpp.poll`, builds the pool with `ggml_threadpool_new_fn` and attaches it
[source-confirmed, `tools/llama-bench/llama-bench.cpp`]. **It reaches no code at all in this
build.** `libggml-cpu.so` imports `GOMP_barrier`, `GOMP_parallel` and `GOMP_single_start`
[verified on the board, `nm -D`], and `threadpool->poll` with its whole poll-then-sleep loop sits
inside `#ifndef GGML_USE_OPENMP` [source-confirmed, `ggml/src/ggml-cpu/ggml-cpu.c`]. The barrier
is `#pragma omp barrier` in the same build, so the spin lives in libgomp and its wait policy is
`OMP_WAIT_POLICY` and `GOMP_SPINCOUNT`, which do have a spin budget and a sleep fallback. **A
runtime knob does reach it**, and its cap is 0.7% of prefill wall [expected], derived from the
pinning arm and the 16.02% libgomp share of the little cluster above. That is under the spread of
every unit that could host the arm, so it is a knob to know about and not one to campaign on.

**The residency configuration varies with the model held fixed, and the two host-side levers
overlap.** The three cells above differ in residency and in model size at once, so neither
separates them. A 2x2 in pinning and residency on ONE GGUF does. `gemma4-12b` F16, `-p 2048 -n 0
-r 3`, three passes [HW sweep 2026-09-01, RK1, 600 MHz, governor `performance`, memory reset
before every arm, arm order rotated per pass, ratios paired within a pass, 12 of 12
`<!--DATA-->` rows, `pfn_zero_frac`=0.0000 and `pmu_enabled`=100 on all twelve, and every
resident arm at an identical 286 of 328 weights (18078 MB, 87%); raw in
`ro-session/trackd-res12b-2x2-3pass.md`].

| arm | mean t/s | range | spread |
|---|---:|---|---:|
| `stream_unpin` | 19.98 | 19.92-20.07 | 0.8% |
| `stream_pin` | 21.27 | 21.14-21.52 | 1.8% |
| `res_unpin` | 21.21 | 21.14-21.25 | 0.5% |
| `res_pin` | 22.25 | 22.00-22.58 | 2.6% |

| lever | paired ratio | se |
|---|---:|---:|
| pinning alone, streamed pair | **1.0645** | 0.0040 |
| pinning alone, resident pair | 1.0490 | 0.0071 |
| residency alone, unpinned pair | **1.0614** | 0.0018 |
| residency alone, pinned pair | 1.0459 | 0.0031 |
| both together | **1.1134** | 0.0061 |

**The two levers are sub-additive, and the interaction is resolved.** Independent levers would
multiply to 1.1299. The measured pair reads 1.1134, a shortfall of 1.65 pp. The interaction
factor is 0.9854 with a per-pass se of 0.34 pp, 4.3 standard errors below 1.00, and it holds the
same sign in all three passes. So roughly 1.5 pp of each lever is the same win. Both remove A76
pack and readback work, so the second one applied finds less of it left [hypothesis].

**This answers the caveat every ratio in this file carries.** The matrix takes both arms
unpinned, which is sound only if pinning does not interact with the knob under test. On this unit
it does interact. The interaction is 1.5 pp against a knob of 6.1 pp, large enough to resolve and
small enough to leave a published ratio of that size standing. That is one knob on one model, and
nothing here bounds a larger or a smaller one.

**The `a55_inst_share` predictor does not survive the out-of-sample test, and the arm was
underpowered to test it.** The three-model table orders the `pin76` gain by the unpinned share.
This arm varies the configuration with the model held fixed, and the sign goes the wrong way.
`res_unpin` carries the HIGHER share, 0.14193 against 0.14090, so the predictor calls for the
larger resident gain. The resident gain is 1.55 pp SMALLER. **Read the size of the contrast
before reading the refutation.** The predictor was fitted over shares spanning 0.141 to 0.209,
and this arm moves the share by 0.001. That is about 1.5% of the fitted range. So the column has
no resolving power at this scale, which is not the same as failing across the range it was fitted
on. A test with real power varies the share by a large fraction of 0.068, and on this evidence
that means varying the model.

**The 87% residency ceiling is TWO limits at once, and neither fix alone moves it by a single
weight.** `ROCKET_F16_RESIDENT=auto` on this model places 286 of 328 weights (18078 MB). Three
limits can turn a weight away and `build_resident` checks them in order -- byte budget, RAM floor,
then the pack itself (IOVA). `[f16-resident] admission first declined` names whichever fired
first, and **which one that is depends on the PREFILL LENGTH as well as the worker count**. At
`-p 2048` it is the RAM floor in 16 of 16 arms, at both 5 and 8 workers. At `-p 512` and 5 workers
it is the IOVA window in 5 of 5. At `-p 512` and 8 workers it is the RAM floor again. No cell
disagrees with itself [HW readouts 2026-09-01/02, RK1; raw in
`ro-session/trackd13-iova-admission-readout.md`, `ro-session/trackd19-f16-window-ladder.md`,
`ro-session/trackd15-res100-value.md`, `ro-session/trackd-res12b-2x2-3pass.md`]. The mechanism is
that a longer prefill's KV cache lowers `MemAvailable` and the RAM floor is checked BEFORE the
pack is attempted, so the floor fires first at `-p 2048` while at `-p 512` admission reaches the
pack and the window fills. So read the string at the shape the campaign will run.

| arm | resident budget | placed | first-decline reason |
|---|---:|---:|---|
| `auto_default` | 21095 MB | 286 | IOVA window filled |
| `ROCKET_N_THREADS=8` | 21099 MB | 286 | reserve floor, 9469 MB against 9535 MB |
| reserve 7168 MB | 23466 MB | 286 | IOVA window filled |
| reserve 6144 MB | 24494 MB | 286 | IOVA window filled |

**But "the fd fix moves nothing" was tested at one worker count, and the curve is not monotonic.**
Laddering `ROCKET_N_THREADS` at the same shape and a near-identical budget (21028-21226 MB) gives
[HW readouts 2026-09-02, RK1, `-p 512 -n 0 -r 1`; raw in
`ro-session/trackd19-f16-window-ladder.md` and `ro-session/trackd19b-f16-window-repeat.md`]:

| workers | placed | resident | share | first-decline reason | readings |
|---:|---:|---:|---:|---|---|
| 5 | 286 | 18078 MB | 87% | **IOVA window filled** | 5, all identical |
| 6 | **293** | **18506 MB** | **89%** | reserve floor | 3, byte-identical |
| 7 | 274 | 17302 MB | 84% | reserve floor | 2, byte-identical |
| 8 | 287 | 18191 MB | 88% | reserve floor | 1 here, 286 in trackd13 |

**Placement is deterministic per worker count, to the byte, across independent processes.** So the
non-monotonicity is real and not the per-process lottery. Six is the best of the four and seven is
**worse than the default**, which is the setting a reader following "raise it" would pass through.

**What moves is the NON-WEIGHT half of the budget.** The floor fires at a fixed `MemAvailable`, and
the total consumed before it fires is near-constant at 21.3-21.5 GB in every arm. The bytes per
placed weight are also constant at 63.15-63.38 MB, so weight padding is not it. The remainder is:
2898-2947 MB at six workers, **4008 MB at seven**, 3160 MB at eight. Seven spends about 1.1 GB more
on everything that is not a weight, and the floor takes it out of the weights.

**The likely mechanism is the per-shape scratch, not the weights** [hypothesis]. Each resident
weight is split across every fd by columns, with the slice rounded up to a multiple of 16
[source-confirmed, `rocket_fanout_nstep`, `rkw_weights_pack`], and a per-shape scratch is allocated
per worker beside it. Seven has the worst rounding of the four on every shape this model carries:
`3840` allocates 1.0208 of itself at seven workers against 1.0000 at five, six and eight. That
predicts seven as the outlier, which it is, and does not predict six above eight, which it is.

**Both fixes together reach 100%.** `ROCKET_N_THREADS=8` with
`ROCKET_QUANT_RESIDENT_RESERVE_MB=6144` places **328 of 328 weights, 20790 MB, 0 streamed**,
reproduced twice with no decline line at all [HW readout 2026-09-01, RK1; raw in
`ro-session/trackd13b-iova-joint-readout.md`]. The two limits mask each other. The RAM floor is
checked BEFORE the pack is attempted, so raising fds alone lets the floor fire at the same point,
and lowering the floor alone runs into the fd count. Only lifting both admits the last 42 weights.

**The memory cost is real and this is not a recommended default.** The joint arm ran with
MemFree at **261 MB** and MemAvailable at **6542 MB** against its own 6144 MB floor, on a board
with **no swap**, where dmesg already carries three OOM kills from earlier work. It completed
twice at `-p 512 -n 0 -r 1`. A longer prefill carries a larger KV cache into that headroom, and
nothing here bounds that. Treat full residency on this model as a measured capability, not as a
setting to adopt.

### What the last 13% is worth, and why the profiler could not say

**The cap on full residency is 1.009x, and it comes from two measured walls rather than from a
profile bucket** [HW readout 2026-09-01, RK1, 600 MHz, `gemma4-12b` F16 at `-p 2048`, 87%
resident; raw in `ro-session/trackd14-packcap-readout.md`]. Residency over 87% of this model's
weight bytes is a measured 1.0614x, worth 5.94 s of a 102.50 s streamed prefill. The 42 weights
left over are 2712 of 20790 MB. At a per-byte-constant pack cost that is 0.89 s of the 96.56 s
resident prefill, so `res100/res87` is capped at **1.0093x**, under 1% of prefill wall.

**`ROCKET_MM_PROFILE` gives a number 33x larger, and it is the wrong number.** Its `packB` bucket
reads 88259 ms over the run's three prefills, 29.4 s each against a 102.7 s prefill wall, which
would price the streamed remainder's per-call weight pack at over a quarter of the wall. The
buckets are `clock_gettime` intervals taken on worker threads, and `rocket_pin_worker_based` pins
worker *i* to big core *i* mod the big-core count [source-confirmed, `rocket_affinity.c`]. The
RK3588 has four A76s, so at the default `ROCKET_N_THREADS=5` two workers already share a core
while llama.cpp's own eight compute threads run on the same eight. **The buckets sum to 854 s
against 308 s of prefill wall, 2.77x**, and the offloaded matmuls are only part of that prefill.
So a bucket is an upper bound on its term's thread-time, not a share of the wall. The single-fd
microbenchmark splits in [../not-mac-bound.md](../not-mac-bound.md) are unaffected: one thread,
nothing beside it, bucket equals wall.

**"0 streamed" does not mean the per-call weight pack is gone.** The 100%-resident arm reports
328 of 328 and 0 streamed, and its `packB` is still **12011 ms**. The `[f16-resident]` ledger's
denominator is the weights OFFERED to the residency route; a matmul that was never a candidate
(below the min-M gate, or with no stable weight name) still takes the streaming path and still
pays a per-call scatter. Read the ledger as a statement about the route, not about the profile.

**The memory headroom at `-p 2048` is measured, and the arm is safe.** The 100%-resident
configuration reproduced its placement a third time at this shape -- 328 of 328, 20790 MB, 0
streamed -- with **MemAvailable bottoming at 5521 MB** and MemFree at 260 MB, and no OOM kill
[HW readout 2026-09-01, RK1; raw in `ro-session/trackd14b-res100-probe.md`]. The `-p 512`
readouts left 6542 MB, so the four-times-larger KV cache costs about 1.0 GB of that headroom.
MemFree is not the column to watch: it reads 260 MB in the 87% arm too, because the page cache
absorbs the GGUF.

**The two profiled arms are not a ratio.** They differ in `ROCKET_N_THREADS` as well as in
placement (5 against 8), which moves the job-batch count 21120 to 33792 and the `wait` bucket
655 s to 1116 s. That is why the timed campaign carries a thread-count control at matched
placement rather than two arms.

### Full residency does not pay on this model, and the recipe is two opposite effects that cancel

[HW sweep 2026-09-01, RK1, 600 MHz, governor `performance`, `gemma4-12b` F16 at `-p 2048 -n 0
-r 3`, three rotated passes, memory reset before every arm, ratios paired within a pass, 9 of 9
`<!--DATA-->` rows, 0 failed arms, `pfn_zero_frac`=0.0000 and `pmu_enabled`=100.0 on all nine, and
each arm's placement identical in all three passes; raw in `ro-session/trackd15-res100-value.md`.]

| arm | env added to `ROCKET_F16_RESIDENT=auto` | placed | mean t/s | range |
|---|---|---:|---:|---|
| `res87` | -- | 286 of 328 (18078 MB) | 21.18 | 20.87-21.48 |
| `res87_t8` | `ROCKET_N_THREADS=8` | 281 of 328 (17730 MB) | 20.91 | 20.71-21.02 |
| `res100` | + `ROCKET_QUANT_RESIDENT_RESERVE_MB=6144` | 328 of 328 (20790 MB) | 21.38 | 21.20-21.62 |

| contrast | what it isolates | paired ratio | se | per-pass |
|---|---|---:|---:|---|
| `res87_t8/res87` | the worker count, placement near-matched | **0.9873** | 0.0046 | 0.9781 0.9923 0.9915 |
| `res100/res87_t8` | the last 47 weights, worker count held | **1.0223** | 0.0110 | 1.0143 1.0439 1.0086 |
| `res100/res87` | the whole 100%-residency recipe | 1.0093 | 0.0135 | 0.9921 1.0359 1.0000 |

**The whole recipe straddles 1.00 and is not resolved.** It sits 0.7 se from no change, and the
per-pass ratios include one below 1.00 and one exactly at it. Resolving a 0.9% effect against this
contrast's 2.34% per-pass sd needs about **25 passes**, roughly eighteen hours. **The memory cost
settles it without them**: 2712 MB more held resident on a board with no swap, for an effect three
passes cannot separate from zero. Full residency on this model is a measured capability and not a
setting to adopt.

**The two components are each larger than their sum, and only the control shows it.** Residency
proper is worth **1.0223x** with the worker count held fixed, 2.0 se above 1.00. The worker count
the recipe requires costs **0.9873x**, 2.8 se below 1.00 and the same sign in all three passes.
They nearly cancel. A two-arm campaign reads the 1.0093 alone and charges the worker-count loss to
residency, which is the reading the item was queued with.

**The residency half is consistent with its cap.** The cap for 47 weights is 1.0105x, and the
measured 1.0223x carries a 2se band of 1.0003-1.0442 that contains it. So the byte-scaling
derivation is not contradicted, and it is also not confirmed at this pass count.

### `ROCKET_N_THREADS` is not a free supply of fds

It is prescribed as the exit from an exhausted NPU IOVA window, for the MoE route as well as this
one. On this unit going 5 to 8 costs in **two** independent ways, and both are measured above.

- **Throughput**: 0.9873x, resolved, same sign in three passes. The mechanism is visible in the
  readout. `rocket_pin_worker_based` pins worker *i* to big core *i* mod the big-core count
  [source-confirmed, `rocket_affinity.c`], so on this four-A76 part eight workers put two on every
  big core and displace llama.cpp's own threads onto the little cluster. `a55_inst_share` rises
  from **0.1410** to **0.1498**, with no overlap between the two arms' three-pass ranges.
- **Placement**: at the default reserve it places **fewer** weights, 281 against 286, reproduced
  in all three passes of each arm. More worker fds reach the `MemAvailable` floor sooner, and that
  floor is checked before the pack is attempted. So where the binding limit is RAM rather than
  IOVA, the knob moves placement the wrong way.

Raise it to open an IOVA window, and read the placement and the wall afterwards rather than
assuming the knob is free.

### Pinning interacts with the matrix's largest knob too, and there it costs 5.9 pp

Every ratio in this file takes both arms unpinned, which is sound only if pinning does not
interact with the knob under test. The `gemma4-12b` F16 2x2 above resolved a 0.9854 interaction
against a 6.1 pp knob and left the published ratio standing. **A knob eleven times larger
interacts more, and the sign is now resolved on both units.** `qwen35-9b` Q4_K_M, the matrix's
largest resolved cell, stock against `ROCKET_QUANT_RESIDENT=auto -b 2048 -ub 2048`, with each arm
run pinned and unpinned [HW sweep 2026-09-01/02, RK1, 600 MHz, governor `performance`, `-p 2048
-n 0 -r 3`, **six** rotated passes, memory reset before every arm, ratios paired within a pass,
24 of 24 `<!--DATA-->` rows, 0 failed arms, `pfn_zero_frac`=0.0000 and `pmu_enabled`=100.0 on all
24, and all twelve resident-arm lines at 200 of 200 (13184 MB), 0 streamed; raw in
`ro-session/trackd16-pin9b-2x2-6pass.md`].

| quantity | mean | se | per-pass |
|---|---:|---:|---|
| the knob, both arms UNPINNED | **1.6675x** | 0.0044 | 1.6597 1.6845 1.6616 1.6548 1.6713 1.6732 |
| the knob, both arms PINNED | **1.6085x** | 0.0040 | 1.5917 1.6114 1.6095 1.6201 1.6145 1.6037 |
| **interaction** | **0.9646** | 0.0035 | 0.9590 0.9566 0.9687 0.9790 0.9660 0.9585 |
| pinning on the stock arm | 1.1003x | 0.0031 | -- |
| pinning on the resident arm | 1.0614x | 0.0026 | -- |

**The unpinned arm replicates the published cell to 0.39%** (1.6675 against 1.6611), and the
stock arm replicates the matrix epoch to 0.05%, so this is the same cell and not a re-cut.

**The interaction is 10.2 se below 1.00 and below it in all six passes.** It costs **5.9 pp of a
66.8 pp knob**. So the published 1.661x is correct as the unpinned number it is labelled, and a
reader who follows the guide's *other* recommendation -- `taskset 0xf0` for a prefill-heavy run --
should expect this knob to add **1.61x, not 1.67x**, on top of the pinning it already has. Taking
both levers from stock reads **1.7699x** (se 0.0047) against **1.8348x** if they were independent.

**The mechanism is the same one the 12B showed, and its size is not.** Pinning buys 1.1003x on the
stock arm and 1.0614x on the resident one: residency has already removed the A76 pack work that
pinning was accelerating, so the second lever applied finds less of it left. The residual 6.1% is
host work residency does not touch.

| unit | knob | interaction | pinning on stock | pinning on resident | pin gain lost |
|---|---:|---:|---:|---:|---:|
| `gemma4-12b` F16 | 1.061x | 0.9854 (4.3 se) | 1.0645 | 1.0490 | 1.55 pp |
| `qwen35-9b` Q4_K_M | 1.668x | 0.9646 (10.2 se) | 1.1003 | 1.0614 | 3.89 pp |

**Two units do not give a law, and this pair actively refuses one**: neither the interaction
factor nor the absolute pin-gain loss is constant between them, so neither can be applied to a
third knob by arithmetic. What is settled is the **sign**, on both units, at 4.3 and 10.2 se and
in 3 of 3 and 6 of 6 passes. **What this does not settle**: the knob here spans residency *and*
`-b 2048 -ub 2048` together, because that is what the published cell is, so the interaction is
with the pair rather than with residency alone. And both units place their weights fully or
mostly; a knob that is not a residency knob at all is untested.

**What this does not settle.** The lever is one model at one shape. Neither half of the readout
reads IOVA, so a placement effect on the NPU side of the IOMMU is outside what any of these
columns could have found. And every campaign number in this file was taken unpinned, so the
absolute levels here are unpinned levels. The paired ratios are not disturbed by that as long as
pinning does not interact with the knob under test. On the one unit where that is measured it DOES interact, by 1.5 pp
against a 6.1 pp knob -- resolved, and too small to overturn a published ratio of that size.

**The pass-count bound for this unit class, stated once**: the empirical per-pass paired-ratio
sd on the re-run is ~6.5%, so at three passes the ratio's se is ~3.8% and a knob under ~8%
cannot resolve; a 2% knob needs ~40 passes at 2 sigma. That is why the 0.8B units land
unresolved and why their rows are reported as straddles rather than means. The knob's own
per-pass sign agreement (campaign 2: six ratios within 1.1% of 1.00) shows the *pairing*
works when the modes co-occur; the budget rule is per unit class, not per knob.

## Raw output

The dqc campaign's raw unit files (clean and tenant-loaded, plus the driver log with its
per-stage idle audits) are the standalone files in [dqc-session/](dqc-session/), pulled from the
board with md5 verified at both ends; they are not duplicated inline here.

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

