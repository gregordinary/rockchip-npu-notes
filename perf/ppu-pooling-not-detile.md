# The PPU is a pooling engine, not a de-tile engine

The PPU cannot write a row-major output layout on-chip, so it cannot replace the A76 NEON
output de-tile (~76 ms of the prefill wall). It is the NPU's pooling processor, the NVDLA
PDP analog, and not a layout or reshape engine. The register set is reverse-engineered
from Mesa's `rocket/registers.xml`. The register pages are `0x6000` (PPU) and `0x7000`
(PPU_RDMA), in `include/npu_hw.h` [source-confirmed].

## The PPU register set

| register | function |
|---|---|
| `PPU_DATA_CUBE_IN_*` / `OUT_*` | input and output cube dims (W/H/C) |
| `PPU_OPERATION_MODE_CFG` | `POOLING_METHOD`, `FLYING_MODE`, `INDEX_EN`, `USE_CNT` |
| `PPU_POOLING_KERNEL_CFG` | pooling kernel W/H + stride W/H |
| `PPU_RECIP_KERNEL_WIDTH/HEIGHT` | `1/kW`, `1/kH`, the average-pool reciprocal |
| `PPU_POOLING_PADDING_CFG` + `PADDING_VALUE_*` | pad top/bottom/left/right + pad value |
| `PPU_DST_BASE_ADDR` / `DST_SURF_STRIDE` / `MISC_CTRL` | pooled cube write-out |
| `PPU_RDMA_*` | the read-DMA that feeds the pooling unit |

That is the complete set. The map has no `TRANSPOSE`, `RESHAPE`, `PERMUTE`, `SPLIT`,
`MERGE`, or `CONTRACT` register. The NPU has no NVDLA-RUBIK functionality, and no RUBIK
register page exists (Mesa `registers.xml`, `librocketnpu`'s `npu_hw.h`). In NVDLA, layout
reshape is a dedicated engine (RUBIK), separate from pooling (PDP). The RK3588's rocket NPU
ships the PDP (PPU) but not RUBIK.

## Layout conversion

The PPU reads a cube (NC1HWC2) and writes a cube (NC1HWC2). Pooling reduces the spatial
dims but does not move the C2 channel-atom into the row-major position. A 1×1 identity
"pool" copies the cube. It does not transpose `[N/C2][M][C2] -> [M][N]`. So the PPU cannot
produce the row-major `C[M,N]` that the host wants, and it cannot replace the host output
de-tile (`detile_accum_f16`, NEON).

The NPU has no engine that reorders between row-major and the cube layout in either
direction. A matmul avoids the de-tile another way. Its DPU can write C row-major
directly, by strides ([matmul-as-conv.md](../matmul-as-conv.md) §"Layouts"). That write is
not a reorder of a cube.

The absence of RUBIK is also why the framework layout ops (`TRANSPOSE`, `PAD`, `SLICE`,
`SPLIT`, `RESHAPE`, `CONCAT`) have no on-NPU route. No register program permutes, crops
or extends a tensor, so they are host byte-copies. The tflite-rocket delegate claims them
anyway, as exact host kernels. The claim does not offload compute. It keeps a real graph's
`conv -> layout-op -> conv` in one delegated partition, where the graph would otherwise go
to a CPU node and back.

## Pooling on the NPU

The PPU is a fully decoded, idle pooling engine, and on-NPU `MaxPool` and `AveragePool`
are HW-validated. The path is `gen_pool_fp16` plus `rocket_pool_fp16`, a standalone job
fed by PPU_RDMA. Average pooling uses the `RECIP_KERNEL_*` reciprocal, fp16(65536/k). Max
pooling uses the `-inf` pad fill. The full register program and the reciprocal format are
HW-validated: MAX is bit-exact against the CPU, and AVG is within fp16-recip tolerance.

The related material:

- [ppu-pooling.md](../encodings/ppu-pooling.md): the encoding
- [ppu-rknn-capture/](../ppu-rknn-capture/): the reproducible capture harness
- `tests/pool_fp16_rocket.c`: the gate
- `pool_npu`: the delegate opt-in, gated in `convert_test`

On-NPU pooling is a prerequisite for keeping a partition's intermediates resident in cube
layout, so that adjacent NPU ops skip the host round-trip. Absent that residency, the
delegate keeps pooling on the host by default. The NPU path costs a second round-trip plus
an NHWC↔cube transpose.

## References

The sources are Mesa `rocket/registers.xml`, `librocketnpu`'s `include/npu_hw.h`, and
[NVDLA hwarch](https://nvdla.org/hw/v1/hwarch.html) (PDP against RUBIK). Related:
[not-mac-bound.md](not-mac-bound.md) (readback is the bottleneck) and the NEON de-tile.
