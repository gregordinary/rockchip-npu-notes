# On-chip and system SRAM on the NPU (NBUF)

The RK3588 NPU reaches SRAM the same way it reaches DDR: through an IOVA in its IOMMU
domain. The NPU has no dedicated NPU↔SRAM bus, no special register, and no "NBUF" hardware
engine. "Using SRAM" means mapping a SRAM physical region into the NPU domain with
`iommu_map()` and addressing it by IOVA like any BO. For LLM prefill, SRAM is expected to give
little speedup, because it speeds DMA and the dominant cost is the host gather [expected].

**Conditions:** RK3588, the BSP `rknpu` driver 0.9.8 and mainline `rocket` as sources, the
device tree of a Turing RK1.

**Not characterized:** SRAM in the dispatch-latency small-op regime (decode GEMV, detection
1×1s), and the owner and NPU-IOMMU reachability of the region at `0xfd600000`.

## Cause

SRAM is a weak performance option for prefill (see [not-mac-bound.md](not-mac-bound.md)).
Prefill pack and readback are A76-NEON gather-bound (~5 % of LPDDR bandwidth), not
bandwidth/latency-bound. SRAM speeds DMA, not the CPU gather, so it cannot move the dominant
cost. The capacity (<=956 KB / 1 MB) is far below tile sizes (a 512×4096 fp16 tile = 4 MB).
The only plausible gain is in the dispatch-latency small-op regime (decode GEMV, detection
1×1s).

The vendor's ~60 % SramHit prediction (§"BSP `rknpu` SRAM mechanism") is a streaming-CNN
pattern: the intermediates are re-read from DDR every layer. `librocketnpu`'s prefill keeps
weights IOVA-resident, and keeps on-NPU encoder intermediates cube-resident between matmuls. No
per-layer DDR round-trip therefore remains for SRAM to absorb. The NEON gather remains, and
SRAM does not touch it.

## NPU memory addressing

Every NPU memory access (CNA/CDMA feature and weight reads, WDMA output writes, regcmd
fetch) is by IOVA, and the per-NPU IOMMU translates it. The NPU is agnostic to whether a
PTE points at LPDDR or at SRAM. The regcmd has no "memory type" field. SRAM is therefore
purely a faster physical backing store, chosen at map time, not in the compute description.

## BSP `rknpu` SRAM mechanism

The BSP `rknpu` driver (0.9.8) backs a BO with SRAM as follows [source-confirmed,
`rknpu_drv.c` and `rknpu_gem.c`]:

- The SRAM pool comes from the device tree, not from the driver's per-SoC configuration.
  `rknpu_find_sram_resource()` parses the NPU node's `rockchip,sram` phandle ->
  `of_address_to_resource()` -> `devm_ioremap()` (CPU view) -> PAGE_SIZE-chunk bitmap
  allocator (`rknpu_mm.c`). The driver builds the pool on any SoC whose NPU node has the
  phandle. The kernel must be built with `CONFIG_ROCKCHIP_RKNPU_SRAM` and `CONFIG_NO_GKI`, and
  the IOMMU must be enabled.
- A buffer created with the `RKNPU_MEM_TRY_ALLOC_SRAM` flag (bit 8) is SRAM-backed where the
  pool has room. The driver allocates the SRAM from the pool and maps it with
  `iommu_map(domain, iova, sram_phys+off, size, prot)` (`rknpu_gem_alloc_buf_with_cache()`,
  cache type `RKNPU_CACHE_SRAM`). The returned IOVA is the NPU address.
- NBUF is a second, separate region. Its physical address and size are fixed in the per-SoC
  configuration (`nbuf_phyaddr`, `nbuf_size`). The region is 1 MB on the RK3576, 256 KB on the
  RK3562 and 512 KB on the RV1126B, and the RK3588 has none. A buffer requests it with
  `RKNPU_MEM_TRY_ALLOC_NBUF` (bit 9), and the driver maps it the same way (cache type
  `RKNPU_CACHE_NBUF`).
- `RKNN_INTERNAL_MEM_TYPE=sram` and `RKNN_WEIGHT_MEM_TYPE=sram` are the vendor runtime's policy:
  which BO to request SRAM for. `RKNN_QUERY_MEM_SIZE` reports the sizes (`total_sram_size`,
  `free_sram_size`).
- The vendor's SRAM settings back two memory classes, Internal (layer intermediates) and
  Weight, sized auto or `=sram#KB`, with a per-layer SramHit predictor. On a streaming-CNN
  example, the vendor runtime predicts that SRAM serves ~60 % of internal read+write traffic
  (6.7 MB of 11.1 MB/frame) [source-confirmed, Rockchip's RK3588 NPU SRAM usage document]. The
  saved cost is the re-read and re-write of feature maps to DDR every layer.

## RK3588 physical SRAM map

The SRAM regions on the RK3588:

- **`syssram`** is at `0xff001000`, 956 KB (`0xef000`), 4K-aligned, `mmio-sram`. The stock
  dtsi partitions all of it to the video decoders (`rkvdec0` 480 KB + `rkvdec1` 476 KB).
  The live Turing RK1 is the same (`codec-sram@0/@78000`, `rkvdec` bound). NPU use requires
  repartitioning away from HW video decode.
- **`0xfd600000`** is 1 MB (`fd600000-fd6fffff`), an `mmio-sram` node (`system_sram1`) in
  mainline's `rk3588-base.dtsi`, with no DT consumers (apparently free). The BSP's
  `rk3588s.dtsi` has no node for it [source-confirmed]. It is a candidate target but
  unidentified (could be firmware-owned), and its NPU-IOMMU reachability is unconfirmed.

## Mainline `rocket`

Mainline `rocket` has no SRAM support. The BO path is shmem(DDR)-pages-only:
`rocket_ioctl_create_bo` -> `iommu_map_sgtable` of DDR pages into the per-fd domain, and
it returns IOVA `mm.start`. Domains are per-fd (`iommu_paging_domain_alloc`, 4 GB aperture
each). Adding SRAM means porting the BSP resource discovery and bitmap allocator, plus a
`CREATE_BO` SRAM flag that maps the SRAM physical region instead of pages. Nothing in the
hardware blocks the port, and it is expected to be moderate in size [expected].
