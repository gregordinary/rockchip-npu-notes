<!-- ROCKET_FA_PROFILE read on the model, gemma4-12b F16 streamed, RK3588 RK1 600 MHz, governor
     performance on all three policies, PIN_MASK=0xf0 with -t 4, llama-bench -p 2048 -n 0 -r 2 -v,
     two arms x three passes rotated, ratios paired within a pass, memory reset before every arm.
     Measured 2026-09-07. libggml-rocket.so 138de09d05f235f8c50c7d9d7df3eaa5 against
     librocketnpu.a 1fd95700abdad3143afdd04c6458b5f1 (rocket-userspace as of 2026-09-07 plus the
     max-range line). Raw: trackd36-faprofile-12b.md, trackd36-faprofile-12b.log.
     Every arm streamed, every arm 432 FA ops, n_kv [1024..2048]. -->

# The host softmax is a third of the flash-attention range, so a vectorised exp is capped at 6% of the prefill wall

The compute segment of the FA offload is 17.4% of the pinned prefill wall and mixes the device's
QK/AV with the host mask and softmax. `expf` is 7.72% of A76 cycles, and its borrowed-rate cap of
1.97% is a floor while the 17.4% is the ceiling -- a bracket nearly ten times wide.
`ROCKET_FA_PROFILE` closes it to one number.

## The read

`max-range` is the largest single head range's own buckets: one thread, one interval, and the
range the dispatch thread waited for. Three prof_on passes, in milliseconds:

| pass | gather | qk | mask | softmax | av | scatter | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.8 | 80.1 | 21.1 | 82.9 | 44.2 | 0.4 | 243.5 |
| 2 | 13.1 | 78.9 | 17.8 | 85.2 | 46.1 | 0.4 | 241.3 |
| 3 | 15.3 | 77.3 | 21.1 | 83.5 | 46.3 | 0.4 | 244.0 |

`softmax` / `max-range` is **0.3405 / 0.3531 / 0.3422**, mean **0.3453**, a spread of 1.8%.

The wall it multiplies is this run's own, not an imported constant. The compute segment reads
16.86 / 16.63 / 16.84 s per prefill against walls of 96.47 / 96.56 / 96.74 s, so it is **17.48% /
17.23% / 17.41%** of the pinned prefill wall, mean **17.37%** -- an independent reproduction of the
published 17.4% on a build twelve commits newer, which is worth having given that `G_1` moved 6.6%
across a similar gap.

**So deleting `expf` from this datapath is capped at 6.00% of the pinned prefill wall**
(5.95 / 6.08 / 5.96 per pass). The same arithmetic prices the mask walk at **1.4%** and the
driver's own operand gather at **1.0%**.

## What the ordering says

Inside the range: qk+av **0.511**, softmax **0.345**, mask **0.087**, gather **0.059**, scatter
**0.002**. The device pair is the larger half, but only as a pair -- **the host softmax is the
largest single bucket in the range**, above QK's 0.32 and AV's 0.19. A range on this shape is
about half device wait and half host arithmetic, and nearly all of the host half is one loop.

## The two lines disagree, and that is what the second one is for

The bucket sums over all 2160 ranges read `softmax` at **0.415 / 0.419 / 0.409** of the summed
buckets -- 1.2x the max-range figure -- and they put host work **above** device work (0.566 against
0.434) where the max range puts it below (0.489 against 0.511). **The two lines invert the
host/device ordering.** The sums are added over concurrent workers and count time a worker spent
descheduled, and the host buckets are the ones that inflates: a device bucket is a fence wait,
which elapses whether or not the thread is on a core. Read as a share, the sums would have
overpriced this lever by a fifth and misordered the halves. The max-range line is the claim.

## The instrument is free on the real workload too

The prof_on arm reads **1.003x** of prof_off (per pass 1.005 / 1.000 / 1.005), which is inside
prof_off's own 0.76% pass range and in the direction that cannot be a cost. The compute segment
does not move across the arms (50.58 / 49.90 / 50.53 s against 50.30 / 49.60 / 50.27 s over 432
ops), every arm streamed, and every arm ran 432 FA ops over the same `n_kv` range. The driver's own
gate had already shown the same on both FA paths [trackd37-fa-prof-validate.md].

## What this does not settle

**A bucket counts descheduled time, so 6.00% is an upper bound and not a size.** The five FA
workers run on four A76s with llama.cpp's own threads beside them, so a host bucket is inflated by
exactly the amount the two lines disagree by.

**The bucket is the softmax LOOP, not `expf`.** It also carries the row maximum pass, the
normalize pass and the fp16 conversions, so the part a vectorised exp could reach is a fraction of
the 6.00%, and a replacement exp has its own cost. **What the 6.00% bounds is deleting the whole
loop**, which nothing can do.

**`expf` is called from both sides of the seam.** The 7.72% of A76 cycles is not all the driver's,
and this profile measures only the driver's. The two are consistent -- 0.345 of a 17.37% segment is
6.0% of wall, and the borrowed-rate reading of the driver's own two symbols was 1.6% -- but they
are not the same quantity, and the gap between them is what descheduling and the loop's other
passes occupy.
