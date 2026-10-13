# RGA 2D engine and the output de-tile

The Rockchip RGA (2D raster graphics accelerator, a separate IP block from the NPU,
`/dev/rga`) performs the NPU's output-cube -> row-major de-tile bit-exactly, and every way of
doing it is slower than the CPU [HW sweep]. That de-tile is the gather-bound readback,
`detile_store_f16` and `detile_accum_f16` in `rocket_matmul.c`. The A76 NEON de-tile remains
the de-tile path, and the readback cost falls with fewer, larger NPU jobs, which reduce the
dispatch overhead. The on-chip PPU cannot de-tile either
([ppu-pooling-not-detile.md](ppu-pooling-not-detile.md)), and the NPU has no DMA byte counters
([hw-byte-counters.md](hw-byte-counters.md)).

**Conditions:** RK3588 (RK1), kernel 7.1.1, RGA driver v1.3.10, 2026-06-29.

**Not characterized:** a batched RGA de-tile, and the PL330 DMAC. The wide-copy ceiling in
§"Cost against the CPU" bounds both.

| Copy | Engine | Time | Throughput |
|---|---|---:|---:|
| 4 MiB wide full-image copy, the RGA's best case | RGA | 1.45 ms | 5.8 GB/s |
| 4 MiB `memcpy` | one CPU core | 0.63 ms | 13.4 GB/s |
| De-tile of a tiny tile, 32 blits, one sync per group | RGA | 7.5 ms | |
| The same de-tile | A76 NEON | 0.0065 ms | |

## The de-tile as a 2D blit

The fp16 output cube is C2=8 (`feat_idx`). For a fixed row `h`, group `g=(nn-1)/8` is 8
contiguous fp16 values at `slot[g·H·8 + 8·(h-1) + f]`. They go to 8 contiguous row-major
columns `C[(m0+h-1)·N + n0 + g·8 + f]`. Per output tile that is `ng=Nt/8` strided blits,
each an 8-wide × Mt-tall copy (src row-stride 8, dst row-stride N).

The blit moves fp16 as `RK_FORMAT_RGB_565` (16bpp, 1 fp16 = 1 px). The source format
equals the destination format and there is no scale, so the copy is byte-preserving. The
bit pattern is therefore irrelevant, and equality is byte-exactness.

The RGA produces output byte-identical to the NEON `detile_store_f16` and to the scalar
`feat_idx` gather [HW sweep]. Cost and reach, not correctness, rule it out.

## Cost against the CPU

The RGA's throughput ceiling already loses to the CPU. Its best case is a single wide
full-image copy (width >= 68 so it can use the fast RGA3 cores, one submit, no per-tile
overhead). That copy runs at ~½ the bandwidth of one CPU core's `memcpy` (and < ⅓ of LPDDR's
~17 GB/s), while the NEON de-tile already fans across 3 A76 cores. Even an idealized batched
RGA de-tile therefore cannot beat the CPU, and the wide-copy figure bounds all RGA schemes,
batched or not.

The de-tile is latency/index-bound, not bandwidth-bound. Per
[not-mac-bound.md](not-mac-bound.md), the host de-tile runs at ~5 % of LPDDR. The cost is
the strided gather pattern plus fp16 handling, not raw bytes. A DMA engine attacks
bandwidth, which is not the bottleneck. The same principle covers the generic PL330 DMAC
(linear/2D-strided copy, lower throughput than RGA, same shared LPDDR). Moving the de-tile
to any DMA engine does not address the binding constraint.

The de-tile itself is the RGA's worst case. The intrinsic blit is 8 px wide (the C2=8 group),
and the RGA is built for wide rasters. The per-group sync de-tile in the table is ~0.23 ms/blit
of pure per-submit latency, 1155x slower than NEON for the same output. Batching into one
submit removes the per-submit term but leaves the narrow-blit inefficiency, still under the
5.8 GB/s wide-copy ceiling.

## Buffer reach

The reach limits are independent of speed. Even if the RGA were fast, the de-tile buffers
cannot reach it on the RK3588:

- **RGA3** (the 2 fast cores) has a minimum input width of 68 px
  (`input_range = {{68,2},...}` in the driver's `rga_hw_config.c`). The 8-wide C2=8 group
  is far below it, so RGA3 cannot do the de-tile at all. Only the single RGA2 core can,
  with no 3-core fan-out (which the NEON path has).
- **RGA2** has 32-bit DMA, "only support under 4G memory" (`RGA_MMU` rejects any buffer
  with a page >= 4 GiB). On a 31 GiB board, `malloc` places buffers high and the buffer is
  rejected, so buffers must come from a low (CMA) dma-heap. `rocket` allocates the NPU output
  BO, and `drm_rocket_create_bo` has no DMA32 flag. The row-major destination is a ggml
  activation tensor in high memory. Neither can be forced into the scarce ~322 MiB CMA pool at
  LLM-prefill scale. Bouncing through CMA would re-read the whole cube, repeating the readback
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

## Evidence

### The de-tile as a 2D blit

The byte-exactness test imports its buffers with `importbuffer_fd` from
`/dev/dma_heap/default_cma_region`, and calls `improcess` per group.

### Cost against the CPU

The wide-copy ceiling bounds every RGA scheme, batched or not, so no further RGA de-tile
scheme was measured.
