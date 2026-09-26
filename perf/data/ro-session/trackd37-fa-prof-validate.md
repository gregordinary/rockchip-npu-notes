<!-- ROCKET_FA_PROFILE validated before use, RK3588 RK1, 600 MHz, governor performance on all
     three policies, power/control=on, /dev/accel/accel0 otherwise free. The subject is the
     driver's own gate `tests/flash_attn_rocket`, run through both FA paths (ROCKET_FA_CHAIN=1
     and =0), knob ON and OFF, three passes with the knob order rotated between them so a board
     drift cannot sit on one arm. Measured 2026-09-07 on rocket-userspace 8d17646 plus the
     max-range line, built on the board; librocketnpu.a 1fd95700abdad3143afdd04c6458b5f1.
     Raw: trackd37-fa-prof-validate.log (e9897f13be485e2d593bbe72c1281972). -->

# The FA profile costs nothing it measures, and on the gate's shapes the range is device wait

`ROCKET_FA_PROFILE=1` splits one flash-attention head range into gather / QK / mask / softmax /
AV / scatter. It shipped without a device read, and an instrument that perturbs its own subject
is the failure mode, so the wall it does not move is the first thing to establish.

## The instrument is free

Twelve runs, paired within a pass. The ratio is prof-on over prof-off.

| path | pass 1 | pass 2 | pass 3 | mean |
|---|---:|---:|---:|---:|
| `ROCKET_FA_CHAIN=1` | 0.99968 | 1.00080 | 0.99958 | **1.00002** |
| `ROCKET_FA_CHAIN=0` | 0.99944 | 1.00035 | 0.99927 | **0.99969** |

Walls are 99.72-99.82 s chained and 100.72-100.87 s per-head. The effect is under 0.04% on both
paths, it is smaller than each path's own pass spread, and **its sign is not the same on the two
paths** -- which is what no effect looks like, and what a real cost would not do. Two timer reads
per bucket per head against a range of hundreds of milliseconds is the reason.

**The knob does not change what is computed.** All twelve runs report 38 PASS and 0 FAIL, and the
largest case (`T=128 n_kv=2048 dh=256 dv=256 nh=16 nkvh=8`) reads `cos=1.000000 max_abs=3.052e-05`
in every one of them, knob on and off alike.

## What the gate's own shapes say

The `max-range` line is the largest single range's own buckets -- one thread, one interval, the
range the dispatch thread waited for. Six instrumented runs:

| arm | gather | qk | mask | softmax | av | scatter | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| chain=1, pass 1 | 11.4 | 714.4 | 48.6 | 111.4 | 750.5 | 6.3 | 1642.7 |
| chain=1, pass 2 | 11.2 | 706.1 | 48.8 | 111.2 | 738.2 | 6.2 | 1621.6 |
| chain=1, pass 3 | 11.3 | 712.4 | 48.7 | 111.3 | 747.8 | 6.3 | 1637.8 |
| chain=0, pass 1 | 11.7 | 712.7 | 48.3 | 110.6 | 745.5 | 6.3 | 1635.2 |
| chain=0, pass 2 | 11.9 | 709.6 | 48.2 | 110.7 | 744.2 | 6.4 | 1631.0 |
| chain=0, pass 3 | 11.7 | 695.6 | 48.1 | 110.6 | 738.0 | 6.3 | 1610.3 |

`softmax` / `max-range` is **0.068** and moves by 0.0004 over six runs; QK plus AV is **0.892**.
The host terms together are 10.8% of the range, and the softmax is two thirds of that.

**This is the driver gate's geometry, not a model's**, and it is quoted here only as the
instrument's output on a shape whose answer is stable. The gate sweeps small `T` against `n_kv`
up to 2048, so its ratio of device work to host softmax is not the ratio a 2048-token prefill
presents, and the number that prices a vectorised exp has to come from the model.

## What this does not show

The buckets count descheduled time as well as working time, so each is an upper bound on its
term. And the wall control above says the instrument does not perturb this gate, which runs the
handler with nothing else on the board; it is not evidence about a frontend whose own compute
threads share the four A76s.
