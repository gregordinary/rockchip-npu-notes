# The RGA 2D engine cannot accelerate the output de-tile

The Rockchip RGA (2D raster graphics accelerator, a separate IP block from the NPU,
`/dev/rga`) cannot offload the NPU's output-cube -> row-major de-tile. That de-tile is the
gather-bound readback, `detile_store_f16` and `detile_accum_f16` in `rocket_matmul.c`. The
A76 NEON de-tile is irreducible against this candidate too. This note is the companion
negative to [ppu-pooling-not-detile.md](ppu-pooling-not-detile.md) (the on-chip PPU cannot
de-tile) and [hw-byte-counters.md](hw-byte-counters.md) (no on-NPU DMA byte counters).

The RGA can do the move bit-exactly, but every way of doing it is slower than the CPU. The
reasons are structural, and they bound all RGA schemes at once
[HW sweep 2026-06-29, RK1 7.1.1, RGA driver v1.3.10].

## The de-tile as a 2D blit

The fp16 output cube is C2=8 (`feat_idx`). For a fixed row `h`, group `g=(nn-1)/8` is 8
contiguous fp16 values at `slot[g·H·8 + 8·(h-1) + f]`. They land at 8 contiguous row-major
columns `C[(m0+h-1)·N + n0 + g·8 + f]`. Per output tile that is `ng=Nt/8` strided blits,
each an 8-wide × Mt-tall copy (src row-stride 8, dst row-stride N).

The blit moves fp16 as `RK_FORMAT_RGB_565` (16bpp, 1 fp16 = 1 px). The source format
equals the destination format and there is no scale, so the copy is byte-preserving. The
bit pattern is therefore irrelevant, and equality is byte-exactness.

A probe (`importbuffer_fd` from `/dev/dma_heap/default_cma_region`, `improcess` per group)
confirms that the RGA produces output byte-identical to the NEON `detile_store_f16` and to
the scalar `feat_idx` gather. The mechanism works. The problem is entirely cost and reach.

## Cost against the CPU

The RGA's throughput ceiling already loses to the CPU. The RGA's best case is a single wide
full-image copy (width >= 68 so it can use the fast RGA3 cores, one submit, no per-tile
overhead). That copy moves 4 MiB at 5.8 GB/s (1.45 ms), against a single-threaded CPU
`memcpy` at 13.4 GB/s (0.63 ms). The RGA peaks at ~½ the bandwidth of one CPU core (and
< ⅓ of LPDDR's ~17 GB/s), while the NEON de-tile already fans across 3 A76 cores.

So even an idealized batched RGA de-tile cannot beat the CPU. This single number bounds
all RGA schemes, batched or not, and makes further RGA de-tile measurement unnecessary.

The de-tile is latency/index-bound, not bandwidth-bound. Per
[not-mac-bound.md](not-mac-bound.md), the host de-tile runs at ~5 % of LPDDR. The cost is
the strided gather pattern plus fp16 handling, not raw bytes. A DMA engine attacks
bandwidth, which is not the bottleneck. The same principle covers the generic PL330 DMAC
(linear/2D-strided copy, lower throughput than RGA, same shared LPDDR). Moving the de-tile
to any DMA engine does not address the binding constraint.

The de-tile itself is the RGA's worst case, and the probe measured it. The intrinsic blit
is 8 px wide (the C2=8 group), and the RGA is built for wide rasters. A per-group sync
de-tile of a tiny tile measured 7.5 ms for 32 blits. That is ~0.23 ms/blit of pure
per-submit latency, 1155x slower than NEON (0.0065 ms) for the same output. Batching into
one submit removes the per-submit term but leaves the narrow-blit inefficiency, still under
the 5.8 GB/s wide-copy ceiling.

## Buffer reach

The reach limits are independent of speed. Even if the RGA were fast, the de-tile buffers
cannot reach it on the RK3588:

- **RGA3** (the 2 fast cores) has a minimum input width of 68 px
  (`input_range = {{68,2},...}` in the driver's `rga_hw_config.c`). The 8-wide C2=8 group
  is far below it, so RGA3 cannot do the de-tile at all. Only the single RGA2 core can,
  with no 3-core fan-out (which the NEON path has).
- **RGA2** has 32-bit DMA, "only support under 4G memory" (`RGA_MMU` rejects any buffer
  with a page >= 4 GiB). On a 31 GiB box, `malloc` lands high and the buffer is rejected,
  so buffers must come from a low (CMA) dma-heap. `rocket` allocates the NPU output BO, and
  `drm_rocket_create_bo` has no DMA32 flag. The row-major destination is a ggml activation
  tensor in high memory. Neither can be forced into the scarce ~322 MiB CMA pool at
  LLM-prefill scale. Bouncing through CMA would re-read the whole cube, paying the readback
  the offload was meant to remove.
- **Image dimensions** are capped at ~8176. That cap and the narrow-blit inefficiency
  compound the two limits above.

## Driver context

Any RGA de-tile requires the vendor `/dev/rga` stack, which is live on the RK1. The
out-of-tree multicore RGA driver (rockchip develop-6.6, forward-ported to kernel 7.1)
provides `/dev/rga` and `librga.so.2`. The mainline V4L2 `rockchip-rga` path speaks a
different ABI, which librga does not speak. The live driver reports v1.3.10 and 3
schedulers (2x RGA3 + 1x RGA2). The forward-port's user-page import path
(`follow_pfnmap_start`, >=6.12) is exactly the code that a virtual-address RGA de-tile
would exercise.

None of this changes the verdict: the RGA is the wrong tool for this move.

## The de-tile-offload frontier

The de-tile-offload frontier is closed for the RGA, and by the bandwidth/latency argument
for the PL330 DMAC. The RGA's throughput ceiling is below one CPU core, and the de-tile is
not bandwidth-bound. The real buffers cannot reach the only core that can do the narrow
blit. The NEON de-tile stays the de-tile path. The readback lever is fewer, bigger NPU jobs
(the dispatch floor), not a different copy engine.
