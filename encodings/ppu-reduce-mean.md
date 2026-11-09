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
| Mean, max, min | H, W <= 8 | Supported: one PPU pool |
| Mean, max, min | Both axes 8-smooth (every prime factor <= 7), with the same factor count, and every pass within its window's output-width allowance | Supported: a chain of PPU pools |
| Mean, max, min | Any other shape | Supported: an exact host reduction |
| Sum | Any | Not supported as a native pass |

## The single-pool window limit

The PPU `POOLING_KERNEL_CFG` kernel and stride fields are 4-bit (value−1), so a single pool
window is capped at 16×16. On the RK3588 a window past 8 on either axis computes incorrect
output from one or two output columns ([ppu-pooling.md](ppu-pooling.md) §"Output-width
allowance"). A pass therefore uses windows to 8×8. A global pool over, say, 56×56 cannot run as one
pool. A large or global pool must be lowered into many small ops.
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
per-pass divisors is exactly `H·W`. The library factors each axis into kernels in `[2,8]`,
and a pass pools `(kh_i, kw_i)`.

The PPU's per-axis reciprocal `fp16(65536/k)` applies the divisor (there is no hardware
divider). A symmetric kernel uses the exact validated per-axis value. An asymmetric kernel
(e.g. a 7×5 final pass of a 28×20 map) splits the divisor geometrically across the two
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
> channels of a 2×2 chain. On that driver every pool job retired by timeout. A difference in
> that driver's timeout handling is the proposed cause of the wrong values [hypothesis].

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

### Passes past the output-width allowance

**Conditions:** RK3588 (Turing RK1), `rocket` 1.3.0, NPU clock 928 MHz, 2026-11-02. Square
maps, C 1 to 64, mean, max and min.

Each pass is a pool, so each is bound by the PPU's output-width allowance for its window
([ppu-pooling.md](ppu-pooling.md) §"Output-width allowance"). A chain whose first pass runs past
it returns incorrect mean, max and min, and its jobs retire at the driver's timeout
[HW sweep 2026-11-02]:

| Map | Passes (window, output width) | Result |
|---|---|---|
| 64×64 | 4×4 to 16 wide, then 16×16 to 1 | Max and min exact, mean within 0.003 |
| 125×125 | 5×5 to 25 wide, then 5×5 to 5, then 5×5 to 1 | Max and min exact, mean within 6.1e-5 |
| 343×343 | 7×7 to 49 wide, then 7×7 to 7, then 7×7 to 1 | Max and min exact, mean within 2.3e-5 |
| 216×216 | 2×2 to 108 wide, then 9×9 to 12, then 12×12 to 1 | Incorrect |
| 256×256 | 16×16 to 16 wide, then 16×16 to 1 | Incorrect |
| 288×288 | 2×2 to 144 wide, then 9×9, then 16×16 | Incorrect |
| 512×512 | 2×2 to 256 wide, then 16×16, then 16×16 | Incorrect |

A chain of three passes is exact when every pass is within the allowance. Each incorrect map
has a pass past it: the pass that opens the chain at 256, 288 and 512, and the 9×9 pass at 216. The mean's
error is the fp16 reciprocal's rounding, so its absolute size follows the input's magnitude.

> [!CAUTION]
> A pass past its window's output-width allowance writes a complete, incorrect cube, and the
> passes after it reduce it into a plausible incorrect result. Software must keep every pass
> of a chain within the allowance, and use windows to 8×8.

### Wide maps within the allowance

**Conditions:** RK3588 (Turing RK1), `rocket` 1.3.0, NPU clock 928 MHz, 2026-11-06. C 8 and 16,
mean, max and min, the library's pass order.

With windows to 8×8 and every pass within its allowance, the maps above compute correctly
[HW sweep 2026-11-06]:

| Map | Passes (window, output width) | Result |
|---|---|---|
| 216×216 | 8×8 to 27 wide, then 3×3 to 9, 3, 1 | Max and min exact, mean within 0.004 |
| 256×256 | 4×4 to 64 wide, then 8×8 to 8, then 8×8 to 1 | Max and min exact, mean within 0.001 |
| 288×288 | 6×6 to 48 wide, then 6×6 to 8, then 8×8 to 1 | Max and min exact, mean within 0.004 |
| 480×480 | 8×8 to 60 wide, then 6×6, 5×5 and 2×2 | Max and min exact, mean within 0.002 |
| 512×512 | 8×8 to 64 wide, then 8×8 to 8, then 8×8 to 1 | Max, min and mean equal to the reference |

A map no pass order fits, such as 1024×1024 (its first pass is past the allowance at 2×2 or
8×8), takes the host reduction.

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
decidability check (`rocket_global_avgpool_plan`) and pass order
(`rocket_global_pool_passes`).

### Pass order and chained intermediates

The library applies the axis factors smallest-kernel-first. The running quotient after pass
*i* is then the product of the remaining (larger) factors. It is therefore >= the largest
factor, which is >= 3 for any axis > 8. That order can put a pass past its allowance: 216
opens with 3×3 to 72, and 480 with 2×2 to 240.

The library then applies the factors largest-first, which narrows the opening pass most. The intermediates are then >= the last
factor, so 2 or more, and never 1 wide. The plan checks each pass's output width against the
allowance for its window (`rocket_pool_fp16_max_ow`) before it admits the map.

The NPU path additionally requires H and W to have the same factor count. Every pass then
pools both axes, and neither axis collapses to 1, which would create a height-1/width-1
intermediate in the tail passes. The rule holds for every square map (`H==W`, the usual
GlobalAvgPool case) and equal-count rectangles. It keeps the library off a 1-wide intermediate,
which no measurement reached. The shared plan keeps the equal-count rule for max and min for
the same reason. An intermediate below 4 in a chain past two passes is measured at 216 (3) and
480 (2), each exact (§"Wide maps within the allowance").

### Decomposability and fallback

`rocket_global_avgpool_plan(C,H,W)` returns 0 (NPU) under four conditions:

- Both axes are 8-smooth: every prime factor is <= 7, which covers all powers of two and
  7,14,28,56,49,98,112,224,….
- The two axes have equal factor counts.
- Every pass is within its allowance in one of the two orders.
- `C`, `H` and `W` are each at most 8192.

Otherwise it returns a negative `rocket_status`. `ROCKET_E_TILING` covers a non-8-smooth axis
(prime factor 11,13,17,…), an unequal factor count (e.g. 56×8) and a map neither order fits
(e.g. 1024×1024). `ROCKET_E_UNSUPPORTED` covers a dimension past 8192. Every such shape
takes an exact host reduction (`rocket_global_avgpool_ref_fp16`): always the correct answer,
on the CPU.

### ReduceSum

ReduceSum-over-spatial is intentionally not a native pass. The avg reciprocal field encodes
`fp16(65536/k)` and cannot represent ×1 (`k=1` -> 65536 = inf in fp16), so a sum-pool would
need a different encoding. Software computes a spatial sum as the mean times `H · W` on the
host.

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

- Factor-axis unit checks (8-smooth detect, products in `[2,8]`)
- Every pass of the library's schedule within its window's allowance
- An off-device schedule + cube-layout self-check (true-division telescoping == direct mean)
- On-hardware runs vs the fp64 oracle, in the Accuracy table

### Passes past the output-width allowance

The probe ran each map through the three library entries against an fp64 reference, one
process a map. At 64, 256, 288 and 512 the inputs were hashed values in [0.5, 4] for the mean
and [−4, 4] for max and min. At 125, 216 and 343 they were in [−4, 4] for all three, C 8 to 64,
on a freshly booted board. At 216, 256, 288 and 512 nearly every channel differs, and the
max at 216 returns NaN. Every chain at 256 and 512, all but one at 288, and two at 216 retired
at the 500 ms timeout. The chains at 125 and 343 completed normally.

### Wide maps within the allowance

The 216 to 512 maps are rows of `tests/reduce_mean_rocket.c`. It runs the library's own pass
schedule (`rocket_global_pool_passes`) in its host self-check, then the three entries on the
NPU. The kernel log had no job timeout. Its inputs are a per-channel offset in [0.5, 2.5]
plus noise in [−1, 1]. The mean's tolerance is 0.004 + 0.4% of the reference. One run each.

### ReduceMax / ReduceMin

The test fails on `out[c] != ref[c]`. `tests/reduce_mean_rocket.c` tests both: `max`/`min`
`exact bad=0` across single + 2-pass + the 28×20 rectangle + C=512 + C%8≠0 + the host
fallbacks.
