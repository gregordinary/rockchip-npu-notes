<!-- gemma4-12b F16 streamed, PINNED (PIN_MASK=0xf0, -t 4), governor performance on all three
     policies, NPU 600 MHz, one job in flight. llama-bench -p 2048 -n 0 -r 2 -v, three arms
     (ROCKET_FA_THREADS 1 / 2 / 4) x three passes, arm order rotated each pass, ratios paired
     within a pass. 9 of 9 rows. Measured 2026-09-07, RK1, rocket 1.3.0 on kernel 7.2.0-1-arm64,
     llama.cpp b10558, libggml-rocket.so md5 0c716831d47055d1a99749b79fc10bf5.
     Driver: perf/data/trackd33-fa-threads-12b.sh (md5 0f82b5bf9a76a7a03f23752e9a6cc62b).
     Raw: trackd33-fathreads-12b.log and the two .err files beside it, md5 verified at both ends.
     Scored with perf/data/fa-threads-score.py. -->

# Threading the attention handler's host walks is worth 1.039x of the pinned prefill wall

The `FLASH_ATTN_EXT` handler's gather and scatter run on the ggml dispatch thread while the
scheduler waits, so their interval is wall and their exchange rate is 1 by construction. That
capped threading them over `k` workers at `5.6% x (1 - 1/k)` with nothing imported.
`ROCKET_FA_THREADS=k` splits all five walks -- the Q convert, the K copy, the V transpose, the
mask copy and the output scatter -- over the process-wide host pool, which is already pinned to
the big cluster.

`G_k` is gather plus scatter seconds per 2048-token prefill, read from the `ROCKET FA total` line
rather than from the wall, because the interval is what the prediction's band is written in.

| `ROCKET_FA_THREADS` | gather | scatter | `G_k` | `G_k`/`G_1` | per-pass | t/s | `t_1`/`t_k` |
|---|---:|---:|---:|---:|---|---:|---:|
| 1 | 3.88 s | 1.92 s | **5.80 s** | -- | -- | 21.25 | -- |
| 2 | 2.11 s | 1.06 s | **3.17 s** | **0.547** | 0.547 0.547 0.547 | 21.74 | **1.0231** |
| 4 | 1.34 s | 0.54 s | **1.87 s** | **0.323** | 0.324 0.324 0.320 | 22.07 | **1.0389** |

**Prediction 2026-09-07 HIT on both registered numbers.** `r` = 0.323 against a band of
0.29-0.44, and the wall ratio 1.0389 against 1.032-1.041. The per-pass `r` spread is 0.4%, so the
result does not rest on the pass count.

## The two readings agree, which is the check that matters

The interval says the saving is `G_1` - `G_4` = 3.92 s of a 96.4 s prefill, **4.07% of wall**. The
wall itself says **3.75%** (`t_1`/`t_4` = 1.0389). Those are separate instruments -- a bracket
inside the handler and llama-bench's own throughput -- and they close to 0.32 percentage points.
A bracket that did not correspond to wall would show up here as a gap, and the earlier reading
that this term's `q` is 1 is what predicts there should be none.

Against the cap of 5.6% x (1 - 1/4) = **4.20%**, the realised 3.75% is **89% of it**.

## Four workers is most of what any number of workers could buy

Fitting `G_k` = `A`/`k` + `B` to the `k`=1 and `k`=2 points gives a parallel part `A` = **5.25 s**
and a fixed part `B` = **0.55 s**, 9.4% of `G_1`. That predicts `G_4` = 1.86 s against a measured
1.87, **0.7% out**, so the third arm is a test of the model rather than an input to it.

The consequence is the ceiling: as `k` grows, `G` tends to `B` = 0.55 s, **0.57% of wall**. Four
workers already take 89% of everything more workers could take, and the pool is four wide on this
part anyway. **There is no further lever here**, and a request above the pool size is clamped and
reads `k`=4's interval.

## Controls

- **One graph in every arm**: 432 FA ops, `n_kv` [1024..2048], in all nine. A ratio between arms
  running different graphs is not a ratio of the same thing.
- **The compute segment did not move**: 16.76 s per prefill over all nine arms, spread 2.1%, no
  trend across `k`. The knob reaches the host walks and nothing else.
- **Every arm streamed.** No `[f16-resident]` line in any arm, as the 24 GB F16 model on a 31 GB
  board places nothing under the reserve floor.
- **Correctness is measured, not argued.** `ROCKET_FA_CHECKSUM=1` hashes every byte the handler
  writes; one hash across `k` = 1, 2, 3, 4, 5 on ministral3-3b and across `k` = 1, 4 on this unit
  [`trackd33-fa-checksum-3b.md`]. `k`=3 is the arm that carries it, dividing no walk length
  evenly.

## What this does not show

`G_1` re-measured here is **5.80 s**, 6.01% of wall, where the 2026-09-03 reading was 5.44 s and
5.6% [`trackd30-fa-timing-12b.md`]. The term is 6.6% larger on the newer `.so` and the ratio is
what the band was written in, so the ratio is what is scored; a session quoting the absolute
should quote this one and say which build.

The compute segment's 17.4% of wall is untouched, and the below-gate CPU attention is outside the
bracket. **This is the PINNED wall**, one unit and one shape: the campaign matrix is unpinned, so
this ratio does not go into it without being re-read there. And the four workers here share four
A76s with whatever the CPU backend's threads are doing between splits, which is not separated.
