# On-chip and system SRAM on the NPU (NBUF)

The RK3588 NPU reaches SRAM the same way it reaches DDR: through an IOVA in its per-fd
IOMMU paging domain. The NPU has no dedicated NPU↔SRAM bus, no special register, and no
"NBUF" hardware engine. "Using SRAM" means mapping a SRAM physical region into the NPU
domain with `iommu_map()` and addressing it by IOVA like any BO.

## NPU memory addressing

Every NPU memory access (CNA/CDMA feature and weight reads, WDMA output writes, regcmd
fetch) is by IOVA, and the per-NPU IOMMU translates it. The NPU is agnostic to whether a
PTE points at LPDDR or at SRAM. The regcmd has no "memory type" field. So SRAM is purely a
faster physical backing store. You choose it at map time, not in the compute description.

## The vendor (BSP rknpu) mechanism

The BSP `rknpu` driver backs a BO with SRAM as follows:

- `rknpu_find_sram_resource()` parses DT `rockchip,sram` -> `of_address_to_resource()` ->
  `devm_ioremap()` (CPU view) -> PAGE_SIZE-chunk bitmap allocator (`rknpu_mm.c`).
- A SRAM-backed ("cache") gem does `iommu_map(domain, iova, sram_phys+off, size, prot)`
  (`rknpu_gem.c`, `RKNPU_CACHE_SRAM`). The returned IOVA is the NPU address.
- `RKNN_INTERNAL_MEM_TYPE=sram`, `RKNN_WEIGHT_MEM_TYPE=sram` and `TRY_ALLOC_SRAM` are
  userspace policy (which BO to back with SRAM), gated by `CONFIG_ROCKCHIP_RKNPU_SRAM`.
  `RKNN_QUERY_MEM_SIZE` reports the sizes (`total_sram_size`, `free_sram_size`).
- The vendor's SRAM knobs back two memory classes, Internal (layer intermediates) and
  Weight, sized auto or `=sram#KB`, with a per-layer SramHit predictor. On a streaming-CNN
  example, SRAM serves ~60 % of internal read+write traffic (6.7 MB of 11.1 MB/frame). The
  saved cost is the re-read and re-write of feature maps to DDR every layer.

## RK3588 physical SRAM map

The SRAM regions on the RK3588:

- **`syssram`** is at `0xff001000`, 956 KB (`0xef000`), 4K-aligned, `mmio-sram`. The stock
  dtsi partitions all of it to the video decoders (`rkvdec0` 480 KB + `rkvdec1` 476 KB).
  The live Turing RK1 is the same (`codec-sram@0/@78000`, `rkvdec` bound). NPU use requires
  repartitioning away from HW video decode.
- **`0xfd600000`** is 1 MB (`fd600000-fd6fffff`), board-specific `mmio-sram`, with no DT
  consumers (apparently free). It is a candidate target but unidentified (could be
  firmware-owned), and its NPU-IOMMU reachability is unconfirmed.

## Mainline `rocket`

Mainline `rocket` has no SRAM support. The BO path is shmem(DDR)-pages-only:
`rocket_ioctl_create_bo` -> `iommu_map_sgtable` of DDR pages into the per-fd domain, and
it returns IOVA `mm.start`. Domains are per-fd (`iommu_paging_domain_alloc`, 4 GB aperture
each). Adding SRAM means porting the BSP resource discovery and bitmap allocator, plus a
`CREATE_BO` SRAM flag that maps the SRAM physical region instead of pages. The port is
clean and moderate in size, and the hardware does not block it.

## SRAM as a performance lever

SRAM is a weak performance lever for this workload (see
[not-mac-bound.md](not-mac-bound.md)). Prefill pack and readback are A76-NEON gather-bound
(~5 % of LPDDR bandwidth), not bandwidth/latency-bound. SRAM speeds DMA, not the CPU
gather, so it cannot move the dominant cost. The capacity (<=956 KB / 1 MB) is far below
tile sizes (a 512×4096 fp16 tile = 4 MB). The only plausible win is the dispatch-latency
small-op regime (decode GEMV, detection 1×1s), which is unmeasured.

The BSP's ~60 % SramHit is a streaming-CNN pattern: the intermediates are re-read from DDR
every layer. This project's prefill keeps weights IOVA-resident, and keeps on-NPU encoder
intermediates cube-resident between matmuls. So no per-layer DDR round-trip remains for
SRAM to absorb. What remains is the NEON gather, which SRAM does not touch.
