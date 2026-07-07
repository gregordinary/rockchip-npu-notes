# The RGA 2D engine cannot accelerate the output de-tile

The Rockchip **RGA** (2D raster graphics accelerator — a separate IP block from the NPU,
`/dev/rga`) **cannot** offload the NPU's output-cube → row-major de-tile (the gather-bound
readback, `rocket_matmul.c` `detile_store_f16` / `detile_accum_f16`). The A76 NEON de-tile
is irreducible against this candidate too — the companion negative to
[ppu-pooling-not-detile.md](ppu-pooling-not-detile.md) (the on-chip PPU can't de-tile) and
[hw-byte-counters.md](hw-byte-counters.md) (no HW byte counters). RGA *can* do the move
**bit-exactly**, but every way of doing it is **slower than the CPU**, for structural reasons
that bound all RGA schemes at once. [HW sweep 2026-06-29, RK1 7.1.1, RGA driver v1.3.10]

## The de-tile as a 2D blit

The fp16 output cube is C2=8 (`feat_idx`): for a fixed row `h`, group `g=(nn-1)/8` is **8
contiguous fp16** at `slot[g·H·8 + 8·(h-1) + f]`, landing at 8 contiguous row-major columns
`C[(m0+h-1)·N + n0 + g·8 + f]`. Per output tile that is **`ng=Nt/8` strided blits**, each an
**8-wide × Mt-tall** copy (src row-stride 8, dst row-stride N). fp16 is moved as
`RK_FORMAT_RGB_565` (16bpp, 1 fp16 = 1 px): src format == dst format, no scale ⇒ a
byte-preserving copy, so the bit pattern is irrelevant and equality is byte-exactness.

A probe (`importbuffer_fd` from `/dev/dma_heap/default_cma_region`, `improcess` per group)
confirms RGA produces **byte-identical** output to the NEON `detile_store_f16` and the scalar
`feat_idx` gather. The mechanism works. The problem is entirely cost and reach.

## Why every RGA scheme loses

**The throughput ceiling already loses to the CPU.** RGA's *best case* — a single **wide**
full-image copy (width ≥ 68 so it can use the fast RGA3 cores, one submit, no per-tile
overhead) — moves 4 MiB at **5.8 GB/s** (1.45 ms), vs a single-threaded CPU `memcpy` at
**13.4 GB/s** (0.63 ms). RGA peaks at **~½ the bandwidth of one CPU core** (and < ⅓ of
LPDDR's ~17 GB/s), while the NEON de-tile already fans across **3** A76 cores. So even an
idealized batched RGA de-tile cannot beat the CPU — this single number bounds **all** RGA
schemes, batched or not, and makes further RGA de-tile measurement unnecessary.

**The de-tile is latency/index-bound, not bandwidth-bound.** Per
[not-mac-bound.md](not-mac-bound.md) the host de-tile runs at ~5 % of LPDDR — the cost is the
strided gather pattern + fp16 handling, not raw bytes. A DMA engine attacks *bandwidth*, which
is not the bottleneck. This is the principle that also subsumes the generic **PL330 DMAC**
(linear/2D-strided copy, lower throughput than RGA, same shared LPDDR): moving the de-tile to
any DMA engine doesn't address the binding constraint.

**The actual de-tile is RGA's worst case, measured.** The intrinsic blit is **8 px wide**
(the C2=8 group), and RGA is built for wide rasters. Per-group sync de-tile of a tiny tile
measured **7.5 ms for 32 blits ≈ 0.23 ms/blit of pure per-submit latency — 1155× slower than
NEON** (0.0065 ms) for the same output. Batching into one submit removes the per-submit term
but leaves the narrow-blit inefficiency, still under the 5.8 GB/s wide ceiling above.

## Structural reach blockers (independent of speed)

Even if RGA were fast, the de-tile buffers can't reach it on RK3588:

- **RGA3 (the 2 fast cores): minimum input width 68 px** (`input_range = {{68,2},...}` in the
  driver `rga_hw_config.c`). The 8-wide C2=8 group is far below it → RGA3 **cannot** do the
  de-tile at all. Only the single RGA2 core can — no 3-core fan-out (which the NEON path has).
- **RGA2: 32-bit DMA, "only support under 4G memory"** (`RGA_MMU`; rejects any buffer with a
  page ≥ 4 GiB). On a 31 GiB box `malloc` lands high → rejected; buffers must come from a low
  (CMA) dma-heap. But the NPU output BO is allocated by `rocket` (no DMA32 flag in
  `drm_rocket_create_bo`) and the row-major destination is a ggml activation tensor (high
  memory) — neither can be forced into the scarce ~322 MiB CMA pool at LLM-prefill scale.
  Bouncing through CMA would re-read the whole cube, paying the readback the offload was meant
  to remove.
- **Image dimension cap ~8176** and the narrow-blit hostility compound the above.

## Driver context

The vendor `/dev/rga` stack is required for any of this and is already live on the RK1: the
out-of-tree multicore RGA driver (rockchip develop-6.6, forward-ported to kernel 7.1) provides
`/dev/rga` + `librga.so.2`; the mainline V4L2 `rockchip-rga` path speaks a different ABI that
librga does not. The live driver reports **v1.3.10, 3 schedulers (2× RGA3 + 1× RGA2)**. The
forward-port's user-page import path (`follow_pfnmap_start`, ≥6.12) is exactly the code a
virtual-address RGA de-tile would exercise. None of this changes the verdict — RGA is the
wrong tool for this move.

## Verdict

The de-tile-offload frontier (RGA, and by the bandwidth/latency argument the PL330 DMAC) is
**closed**: RGA's throughput ceiling is below one CPU core, the de-tile isn't bandwidth-bound,
and the real buffers can't reach the only core that could do the narrow blit. The NEON de-tile
stays the de-tile path; the readback lever is fewer/
bigger NPU jobs (the dispatch floor), not a different copy engine.
