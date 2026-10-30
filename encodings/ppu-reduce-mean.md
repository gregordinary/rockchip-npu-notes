# GlobalAveragePool / Mean / ReduceMean on the PPU (multi-pass reduction)

The spatial mean is the reduction over `[H,W]` per channel,
`out[c] = mean_{h,w} in[c][h][w]`. It is the squeeze of every SE block and of most classifier
and detection heads (TFLite `MEAN` over axes `[1,2]`, ONNX `GlobalAveragePool` /
`ReduceMean`). It builds on the [PPU pooling engine](ppu-pooling.md). A mean larger than one
PPU window runs as a chain of small non-overlapping average pools, one PPU pass each.
`PPU_POOLING_KERNEL_CFG` sets each pass's window and stride, and
`PPU_RECIP_KERNEL_{WIDTH,HEIGHT}` its divisor. ReduceMax and ReduceMin run the same chain
with the max and min methods.

## Support status

| Reduction | Shape | Status |
|---|---|---|
| Mean, max, min | H, W <= 16 | Supported: one PPU pool |
| Mean, max, min | Both axes 16-smooth (every prime factor <= 16), with the same factor count | Supported: a chain of PPU pools |
| Mean, max, min | Any other shape | Supported: an exact host reduction |
| Sum | Any | Not supported as a native pass |

## The single-pool window limit

The PPU `POOLING_KERNEL_CFG` kernel and stride fields are 4-bit (value−1), so a single pool
window is capped at 16×16 ([ppu-pooling.md](ppu-pooling.md)). A global pool over, say,
56×56 cannot run as one pool. A large or global pool must be lowered into many small ops.
The reproducible rknn-toolkit2 capture harness shows the same: the `gavgpool_c16` /
`avgpool_4x4` cases decompose into many ops rather than one PPU pool (see
[../ppu-rknn-capture/README.md](../ppu-rknn-capture/README.md)).

## The decomposition: telescoping non-overlapping average pools

A global mean factors into a chain of small non-overlapping average pools, telescoped:

```
mean over (k1·k2·…·kn) equal groups  ==  (…((mean_k1) mean_k2) …) mean_kn
```

The average of equal-sized group-averages is the grand average. Each pass tiles the current
extent exactly (`kernel | size`, `stride = kernel`, `pad = 0`), so the product of the
per-pass divisors is exactly `H·W`. The library factors each axis into kernels in `[2,16]`,
and a pass pools `(kh_i, kw_i)`.

The PPU's per-axis reciprocal `fp16(65536/k)` applies the divisor (there is no hardware
divider). A symmetric kernel uses the exact validated per-axis value. An asymmetric kernel
(e.g. a 14×10 final pass of a 28×20 map) splits the divisor geometrically across the two
axes. The product `recip_h·recip_w·2⁻³²` then still equals `1/(kh·kw)`.

Intermediates stay resident in the NC1HWC2 cube between passes (input scattered once, the
1×1xC result de-tiled once). A pass's output cube is a standard contiguous NC1HWC2
cube, the conv/pool feature layout. Each following pass reads it directly as its input, so
the chain is layout-consistent by construction.

## PPU-written sub-4 intermediates

**Conditions:** RK3588, `rocket` 1.3.0, NPU clock ~816 MHz, 2026-09-26.

**Not characterized:** a chain past two passes, an intermediate one row or one column wide,
int8 pooling, and padded or overlapping windows.

A two-pass pooling chain whose intermediate is 2×2 or 3×3 gives the correct result [HW sweep].
Its second pass matches the same pass run over a CPU-scattered copy of the intermediate, on
every channel. The chain's first pooling pass is exact against a CPU model, and so is the
whole chain [HW sweep]. The chains measured, at C 64 and 130, in MAX and AVG, are:

- 8×8 through 4 then 2
- 6×6 through 3 then 2
- 12×12 through 4 then 3
- 16×16 through 8 then 2

With the kernel's PPU completion class set ([pool-completion.md](../perf/pool-completion.md)),
each pass retires in 0.03-0.2 ms. Without it, each job runs to the 500 ms watchdog and a core
reset, and still leaves the correct values [HW sweep].

> [!CAUTION]
> A run on a driver with no PPU completion class read zero or garbage on about half the
> channels of a 2×2 chain. On that driver every pool job retired by timeout. The cause of the
> wrong values is not established. A difference in that driver's timeout handling is the
> proposed cause [hypothesis].

## Accuracy

**Conditions:** RK3588 (RK1), NPU clock ~816 MHz.

The mean is checked against an fp64 reference on these cases [HW sweep]:

| Case | Shapes |
|---|---|
| Single-pass | 7×7, 14×14, C=130/512 |
| Two-pass square | 28×28, 32×32, 56×56, 64×64 |
| Two-pass equal-count rectangle | 28×20, asymmetric 14×10 final pass |
| C not a multiple of 8 | 130 |
| Host fallback | 56×8, 17×17, 19×19 |

The first four cases measure all `bad=0`, `max_abs <= 0.001`. The host fallback is exact. The
per-pass error is fp16 rounding + ~1e-3 reciprocal quant.

## ReduceMax / ReduceMin (GlobalMaxPool / GlobalMinPool)

The PPU pool engine has native `POOL_METHOD_MAX=1` / `MIN=2` (the AVG path is `0`). The
spatial ReduceMax and ReduceMin therefore reuse the mean's telescoping chain and its
smallest-first factor order. Only the per-pass reciprocal is dropped.

Max and min are exact through the multi-pass chain, and average is not. MAX and MIN are
idempotent: `max(max(block₁), max(block₂)) = max(all)` regardless of how the spatial extent
is grouped into passes. There is no reciprocal or divide, so no fp16 rounding enters. The
decomposed NPU result is therefore bit-exact against the host reduction [HW sweep].

The "equal factor count per axis" constraint is therefore not needed for numerical
correctness: a max or min reduce would be correct under any grouping.

## Software support

`librocketnpu` implements the mean as `rocket_global_avgpool_fp16` (`src/rocket_reduce.c`,
`include/rocket_reduce.h`). `rocket_global_maxpool_fp16` / `rocket_global_minpool_fp16`
(same file/header) are the spatial ReduceMax / ReduceMin. They share the mean's
decidability check (`rocket_global_avgpool_plan`) and factor order.

The cube layout is the conv feature cube (`feature_data`, C2=8), so the input packB and the
output de-tile reuse the conv path. Single-pass global pooling (H,W <= 16) is one PPU
average pool.

### Keeping every chained intermediate >= 4

The library applies the axis factors smallest-kernel-first. The running quotient after pass
*i* is then the product of the remaining (larger) factors, hence >= the largest factor. For
any axis > 16 the largest factor is >= 4 (the greedy <=16 decomposition always grabs a
chunk >= 4, verified for every 16-smooth size 18..), so no intermediate spatial dim is
ever < 4.

The NPU path additionally requires H and W to have the same factor count. Every pass then
pools both axes, and neither axis collapses to 1, which would create a height-1/width-1
intermediate in the tail passes. The rule holds for every square map (`H==W`, the usual
GlobalAvgPool case) and equal-count rectangles. Both rules keep the library off the two
shapes that are not characterized: a 1-wide intermediate and a chain past two passes. The
shared plan keeps the equal-count rule for max and min for the same reason.

### Decomposability and fallback

`rocket_global_avgpool_plan(C,H,W)` returns 0 (NPU) iff both axes are 16-smooth (every
prime factor <= 16, covering all powers of two and 7,14,28,56,49,98,…) and have equal
factor count. A non-16-smooth axis (prime factor 17,19,23,…) or an unequal factor count
(e.g. 56×8) takes an exact host reduction (`rocket_global_avgpool_ref_fp16`): always the
correct answer, on the CPU.

### ReduceSum

ReduceSum-over-spatial is intentionally not a native pass. The avg reciprocal field encodes
`fp16(65536/k)` and cannot represent ×1 (`k=1` -> 65536 = inf in fp16), so a sum-pool would
need a different encoding. `mean · H · W` on the host is the trivial wrapper.

## Evidence

### PPU-written sub-4 intermediates

The chains are `tests/ppu_sub4_chain_probe`, run beside a 4×4-intermediate control. The
completion-class comparison is the probe's flag and no-flag configurations on `rocket` 1.3.0.

The wrong values come from one earlier run on a driver with no PPU completion class. On
`rocket` 1.3.0, the probe's no-flag configuration reaches the same timeout retirement path.
It does not reproduce the wrong values, so their cause is not established. Neither that driver
nor that probe is available to test the hypothesis.

The probe's method could not have found a failure in these cases:

- A chain past two passes
- An intermediate one row or one column wide
- int8 pooling
- Padded and overlapping windows

### Accuracy

`tests/reduce_mean_rocket.c` (CTest `reduce_mean_rocket`) runs on the RK1 at ~816 MHz and holds
these checks:

- Factor-axis unit checks (16-smooth detect, products in `[2,16]`)
- An off-device schedule + cube-layout self-check (true-division telescoping == direct mean)
- On-hardware runs vs the fp64 oracle, in the Accuracy table

### ReduceMax / ReduceMin

The test fails on `out[c] != ref[c]`. `tests/reduce_mean_rocket.c` tests both: `max`/`min`
`exact bad=0` across single + 2-pass + the 28×20 rectangle + C=512 + C%8≠0 + the host
fallbacks.
