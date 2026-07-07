# On-chip / system SRAM and the NPU (NBUF) — what it is, how it's reached

**One-line NPU fact:** the RK3588 NPU reaches SRAM the *same way it reaches DDR* — via an
IOVA in its per-fd IOMMU paging domain. There is **no dedicated NPU↔SRAM bus, no special
register, and no "NBUF" hardware engine** on the NPU. "Using SRAM" = `iommu_map()` a SRAM
*physical* region into the NPU domain and address it by IOVA like any BO.

## How the NPU addresses memory
Every NPU memory access (CNA/CDMA feature/weight reads, WDMA output writes, regcmd fetch) is
by **IOVA**, translated by the per-NPU IOMMU. The NPU is agnostic to whether a PTE points at
LPDDR or at SRAM — there is no "memory type" field in the regcmd. So SRAM is purely a faster
*physical* backing store; you choose it at map time, not in the compute description.

## The vendor (BSP rknpu) mechanism — the reference implementation
- `rknpu_find_sram_resource()` parses DT `rockchip,sram` → `of_address_to_resource()` →
  `devm_ioremap()` (CPU view) → PAGE_SIZE-chunk **bitmap allocator** (`rknpu_mm.c`).
- A SRAM-backed ("cache") GEM does `iommu_map(domain, iova, sram_phys+off, size, prot)`
  (`rknpu_gem.c`, `RKNPU_CACHE_SRAM`). The returned IOVA is the NPU address.
- `RKNN_INTERNAL_MEM_TYPE=sram` / `RKNN_WEIGHT_MEM_TYPE=sram` / `TRY_ALLOC_SRAM` are
  **userspace policy** (which BO to back with SRAM), gated by `CONFIG_ROCKCHIP_RKNPU_SRAM`.
  Query via `RKNN_QUERY_MEM_SIZE` (`total_sram_size`/`free_sram_size`).
- The vendor's SRAM knobs back two memory classes — **Internal** (layer intermediates) and
  **Weight** — sized auto or `=sram#KB`, with a per-layer **SramHit** predictor. On a
  streaming-CNN example ~60 % of internal read+write traffic is served from SRAM (6.7 MB of
  11.1 MB/frame) — the saved cost is the re-read/re-write of feature maps to DDR every layer.

## RK3588 physical SRAM map
- **`syssram` = `0xff001000`, 956 KB** (`0xef000`), 4K-aligned, `mmio-sram`. In the stock
  dtsi **fully partitioned to the video decoders** (`rkvdec0` 480 KB + `rkvdec1` 476 KB);
  on the live Turing RK1 the same (`codec-sram@0/@78000`, `rkvdec` bound). NPU use requires
  repartitioning away from HW video decode.
- **`0xfd600000`, 1 MB** (`fd600000-fd6fffff`) — board-specific `mmio-sram`, **no DT
  consumers (apparently free)**. Candidate target but unidentified (could be firmware-owned);
  unconfirmed NPU-IOMMU reachability.

## Mainline `rocket`
**No SRAM support** — BO path is shmem(DDR)-pages-only (`rocket_ioctl_create_bo` →
`iommu_map_sgtable` of DDR pages into the per-fd domain, returns IOVA `mm.start`). Domains are
**per-fd** (`iommu_paging_domain_alloc`, 4 GB aperture each). Adding SRAM = port the BSP
resource-discovery + bitmap allocator + a `CREATE_BO` SRAM flag that maps the SRAM phys
instead of pages. Clean, moderate; not HW-blocked.

## Why it's a weak perf lever here (see not-mac-bound.md)
Prefill pack+readback is **A76-NEON gather-bound** (~5 % of LPDDR bandwidth), not bandwidth/
latency-bound — SRAM speeds DMA, not the CPU gather, so it can't move the dominant cost.
Capacity (≤956 KB / 1 MB) is far below tile sizes (a 512×4096 fp16 tile = 4 MB). The only
plausible win is the **dispatch-latency small-op regime** (decode GEMV, detection 1×1s),
unmeasured.

The BSP's ~60 % SramHit (above) is a **streaming-CNN** pattern — intermediates re-read
from DDR every layer. Our prefill keeps weights IOVA-resident and on-NPU encoder
intermediates cube-resident between matmuls, so there is no per-layer DDR round-trip for
SRAM to absorb; what remains is the NEON gather, which SRAM does not touch.
