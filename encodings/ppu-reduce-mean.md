# GlobalAveragePool / Mean / ReduceMean on the PPU (multi-pass reduction)

Spatial mean over `[H,W]` per channel — `out[c] = mean_{h,w} in[c][h][w]` — the squeeze of
every SE block and most classifier / detection heads (TFLite `MEAN` over axes `[1,2]`,
ONNX `GlobalAveragePool` / `ReduceMean`). Built on the [PPU pooling engine](ppu-pooling.md).
Implemented: `rocket_global_avgpool_fp16` (`src/rocket_reduce.c`,
`include/rocket_reduce.h`), HW gate `tests/reduce_mean_rocket.c` (CTest `reduce_mean_rocket`).

## Why a global mean is not one average pool

The PPU `POOLING_KERNEL_CFG` kernel **and** stride fields are **4-bit (value−1)** ⇒ a single
pool window is capped at **16×16** ([ppu-pooling.md](ppu-pooling.md)). A global pool over,
say, 56×56 cannot run as one pool — a large/global pool must be lowered into *many small
ops* (which the reproducible rknn-toolkit2 capture harness also shows: the `gavgpool_c16` /
`avgpool_4x4` cases decompose into many ops rather than one clean PPU pool; see
[../ppu-rknn-capture/README.md](../ppu-rknn-capture/README.md)).

## The decomposition: telescoping non-overlapping average pools

A global mean factors into a **chain** of small non-overlapping average pools, telescoped:

```
mean over (k1·k2·…·kn) equal groups  ==  (…((mean_k1) mean_k2) …) mean_kn
```

— the average of equal-sized group-averages **is** the grand average. Each pass tiles the
current extent exactly (`kernel | size`, `stride = kernel`, `pad = 0`), so the product of the
per-pass divisors is exactly `H·W`. Each axis is factored into kernels in `[2,16]`; a pass
pools `(kh_i, kw_i)`. The divisor is applied via the PPU's per-axis reciprocal `fp16(65536/k)`
(no hardware divider); a symmetric kernel uses the exact validated per-axis value, an
asymmetric kernel (e.g. a 14×10 final pass of a 28×20 map) splits the divisor geometrically
across the two axes so the product `recip_h·recip_w·2⁻³²` still equals `1/(kh·kw)`.

Intermediates stay **resident in the NC1HWC2 cube** between passes (input scattered once, the
1×1×C result de-scattered once). A pass’s output cube is a standard contiguous NC1HWC2 cube,
read directly as the next pass’s input — the conv/pool feature layout, so the chain is
layout-consistent by construction.

## NPU FACT — a PPU-written sub-4 intermediate is mis-read by the next chained pass

A genuine NPU↔NPU chaining quirk, isolated empirically. The next implementer chaining PPU passes
will hit it: a sub-4 spatial cube written by the PPU looks correct (a CPU tight-read of it is
bit-exact) but the next PPU_RDMA reads it wrong. Factor smallest-kernel-first to keep every
intermediate ≥ 4 (mitigation below).

| case | result |
|---|---|
| standalone pool, **CPU-scattered** 2×2 input → 1×1 | **exact** (PASS) |
| standalone pool, 4×4 input → **CPU-read** 2×2 output | **exact** (PASS) — the PPU *writes* a 2×2 cube correctly |
| chained pool, **PPU-written** 2×2 intermediate → next pass reads it | **WRONG** (≈half the channels read 0 / garbage) |
| chained pool, **PPU-written** 4×4 intermediate → next pass reads it | correct |

So: the PPU *writes* a sub-4 cube correctly (a CPU tight-read of it is bit-exact), and the
PPU_RDMA *reads* a CPU-scattered sub-4 cube correctly — but the PPU_RDMA does **not** correctly
read a sub-4 cube **that the PPU itself just wrote** in a back-to-back job (even with a full
PREP_BO fence/wait between the two). The break is between 2 and 4 (4×4 is fine), matching the
"`datain_height < 4`" class seen in the conv tiler. Most likely a device→device write-then-read
hazard that only bites very small (fast-draining) WDMA transfers, or a sub-4 output-write
granularity the subsequent RDMA misinterprets; the exact register-level cause is **not yet
RE'd** (a standalone probe would force descending-order factors + bracket the two jobs).

### Mitigation: keep every chained intermediate ≥ 4

Apply the axis factors **smallest-kernel-first**. Then the running quotient after pass *i*
equals the product of the *remaining (larger)* factors, hence ≥ the largest factor. For any
axis > 16 the largest factor is ≥ 4 (the greedy ≤16 decomposition always grabs a chunk ≥ 4 —
verified for every 16-smooth size 18..), so **no intermediate spatial dim is ever < 4**. The
NPU path additionally requires H and W to have the **same factor count** (so every pass pools
both axes and neither axis collapses to 1 — which would create a height-1/width-1 intermediate
in the tail passes). That holds for every **square** map (`H==W`, the usual GlobalAvgPool case)
and equal-count rectangles.

## Decomposability / fallback

`rocket_global_avgpool_plan(C,H,W)` returns 0 (NPU) iff both axes are **16-smooth** (every
prime factor ≤ 16 — covers all powers of two and 7,14,28,56,49,98,…) **and** have equal factor
count. A non-16-smooth axis (prime factor 17,19,23,…) or an unequal factor count
(e.g. 56×8) takes an **exact host reduction** (`rocket_global_avgpool_ref_fp16`) — always the
correct answer, just on the CPU.

## HW validation

`tests/reduce_mean_rocket.c` (CTest `reduce_mean_rocket`), on the RK1 @600 MHz:

- factor-axis unit checks (16-smooth detect; products in `[2,16]`);
- an off-device schedule + cube-layout self-check (true-division telescoping == direct mean);
- on-HW vs the fp64 oracle: **single-pass** (7×7, 14×14, C=130/512), **two-pass** square
  (28×28, 32×32, 56×56, 64×64), **two-pass equal-count rectangle** (28×20, asymmetric 14×10
  final pass), C not a multiple of 8 (130) — all `bad=0`, `max_abs ≤ 0.001`; **host fallback**
  (56×8, 17×17, 19×19) exact. Per-pass error is fp16 rounding + ~1e-3 reciprocal quant.

Cube layout = the conv feature cube (`feature_data`, C2=8), so input packB / output de-tile
reuse the conv path. Single-pass global pooling (H,W ≤ 16) is just one PPU average pool.

## ReduceMax / ReduceMin (GlobalMaxPool / GlobalMinPool) — the idempotent siblings

`rocket_global_maxpool_fp16` / `rocket_global_minpool_fp16` (same file/header) are the spatial
ReduceMax / ReduceMin. The PPU pool engine has native `POOL_METHOD_MAX=1` / `MIN=2` (the AVG path
is `0`), so they reuse the **exact same telescoping engine, decidability** (`rocket_global_avgpool_plan`)
and **sub-4-chained-intermediate mitigation** as the mean — only the per-pass reciprocal is dropped.

**NPU FACT — max/min are EXACT through the multi-pass chain (avg is not).** MAX and MIN are
*idempotent*: `max(max(block₁), max(block₂)) = max(all)` regardless of how the spatial extent is
grouped into passes, and there is no reciprocal/divide, so no fp16 rounding enters. The decomposed
NPU result is therefore **bit-exact** vs the host reduction (`out[c] != ref[c]` ⇒ fail), a stronger
property than the mean's `max_abs ≤ 0.001`. (This also means the "equal factor count per axis"
constraint is only needed for the sub-4 chained-cube quirk, not for numerical correctness — a max/min
reduce would be correct under any grouping; the shared plan is kept for the quirk-safety.) Both are
gated in `tests/reduce_mean_rocket.c`: `max`/`min` `exact bad=0` across single + 2-pass + the 28×20
rectangle + C=512 + C%8≠0 + the host fallbacks.

ReduceSum-over-spatial is intentionally **not** a native pass: the avg reciprocal field encodes
`fp16(65536/k)` and can't represent ×1 (`k=1` → 65536 = inf in fp16), so a sum-pool would need a
different encoding; `mean · H · W` on the host is the trivial wrapper.
