<!-- The FA handler's output checksum moved between runs that differed in nothing, RK3588 RK1,
     600 MHz, governor performance, power/control=on, board otherwise idle. gemma4-12b F16,
     llama-bench -p 2048 -n 0 -r 1 -t 4 -v under taskset 0xf0, ROCKET_KACC=1, page cache dropped
     before every run, one process per run. Measured 2026-09-07 across two ggml-rocket builds
     (libggml-rocket.so c66ef3e8b67437779daf0e73595e5ba9 and 138de09d05f235f8c50c7d9d7df3eaa5)
     against librocketnpu.a 1fd95700abdad3143afdd04c6458b5f1.
     Raw: trackd37-fa-cksum-gate.log, trackd37-fa-cksum-repro.log, trackd38-fa-cksum-localize.log. -->

# The FA output hash is not reproducible run to run, and that retires the bit-identity gate's evidence

`ROCKET_FA_CHECKSUM=1` hashes every byte the `FLASH_ATTN_EXT` handler writes and prints one number
at exit. It was built to make `ROCKET_FA_THREADS`'s bit-identical claim a measurement, and it
compares one run against one run. **That is only a gate on the chunk count if the handler is
reproducible at a FIXED chunk count, and on this unit it is not.**

## What was measured

Thirteen runs of one model at one shape, differing only in `ROCKET_FA_THREADS` where noted. Every
run reports `288 ops, 2818572288 bytes`, so every run ran the same graph and hashed the same
surface.

| campaign | `.so` | runs | `k` | distinct hashes |
|---|---|---:|---|---|
| the gate | `c66ef3e8` | 2 | 1, 4 | `345a87c8...` at `k`=1, `8852206b...` at `k`=4 |
| reproduction | `c66ef3e8` | 5 | 1,4,1,4,1 | `8852206b...` x4, `345a87c8...` once (a `k`=1 run) |
| localization | `138de09d` | 6 | 1 | `8852206b...` x6 |

**The gate's red result was not about `k`.** Its `k`=1 arm drew `345a87c8...`, but the
reproduction ran `k`=1 three more times and got the majority hash twice -- so two runs at the SAME
chunk count disagree, and the disagreement the gate reported carries no information about the
split. Both odd draws landed on `k`=1 arms, which is what a rate of about one in four looks like
when five of the seven runs on that build were `k`=1.

**The majority hash is the one the previous build recorded**, `8852206bf3ccda62` over the same 288
ops and 2818572288 bytes, so the stable value has now been reproduced across three builds of
`libggml-rocket.so` and two of `librocketnpu.a`. Only the driver's profiling counters and a log
argument separate those driver builds, and neither reaches the arithmetic.

## What is not established

**No op has been named.** `ROCKET_FA_CHECKSUM=2` records a hash per op with its shape beside it,
so a differing run diffs to an op index; six runs under it are identical in all 288 lines and the
event did not occur. So the two candidates a single number cannot separate are both still open:
one op wrong on some geometry every time it appears, or any op wrong occasionally.

**The rate is not established and neither is its build dependence.** Two events in seven runs on
one `.so` and none in six on the next is compatible with a rare event that did not recur -- at a
one-in-four rate, six clean runs have probability 0.18 -- and with a build-specific one. Nothing
here separates them, and the second reading would be the more surprising, since the two `.so`
builds differ only by the per-op recording added for this question.

**The magnitude is unknown.** A hash moves on one bit as readily as on a whole surface, so
nothing here says whether the affected run was slightly or badly wrong, nor whether the majority
hash is the correct one. The next instrument is a value comparison, not another hash.

## What it means for what has been published

The claim that threading the host walks is bit-identical at every chunk count rests on one hash
across `k` = 1, 2, 3, 4, 5 on `ministral3-3b` and one across `k` = 1, 4 on this 12B unit. Those
runs agreed, and on this evidence agreeing is what most runs do whatever the split is doing.
**The claim is neither confirmed nor refuted; the evidence for it is retired.** Re-establishing
it needs the comparison run at one `k` first -- enough repeats to bound the same-`k` disagreement
rate -- and only then across `k`, with more repeats than the disagreement rate.

The same reasoning reaches one line further. `llama-perplexity` was read as showing the stack is
deterministic across processes because two `ROCKET_FLASH_ATTN=1` processes agreed to every printed
digit. Perplexity is a mean over every logit and prints four decimals; it is not sensitive enough
to see a surface that moves this rarely, so that agreement is evidence about the printed digits
and not about determinism.

## The trap this leaves

**A gate that compares one run to one run measures the difference between its arms PLUS whatever
the subject does on its own, and the two are not separable from inside it.** The fix is cheap and
is the same one a paired A/B uses for time: run the null arm against itself first. Where the
observable is an equality rather than a magnitude, that means repeating one arm and counting how
often it agrees with itself -- a step no correctness gate in this tree currently takes.
