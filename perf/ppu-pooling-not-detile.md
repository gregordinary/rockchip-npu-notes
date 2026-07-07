# The PPU is a pooling engine, not a de-tile engine

The PPU **cannot** write a row-major output layout on-chip, so it **cannot** replace the
A76 NEON output de-tile (~76 ms of the prefill wall). It is the NPU's **pooling** processor
— the NVDLA **PDP** analog — not a layout/reshape engine. RE'd from Mesa's
`rocket/registers.xml`; register page `0x6000` (PPU) / `0x7000` (PPU_RDMA), in
`include/npu_hw.h`. [source-confirmed]

## The PPU register set

| register | function |
|---|---|
| `PPU_DATA_CUBE_IN_*` / `OUT_*` | input/output cube dims (W/H/C) |
| `PPU_OPERATION_MODE_CFG` | `POOLING_METHOD`, `FLYING_MODE`, `INDEX_EN`, `USE_CNT` |
| `PPU_POOLING_KERNEL_CFG` | pooling kernel W/H + stride W/H |
| `PPU_RECIP_KERNEL_WIDTH/HEIGHT` | `1/kW`, `1/kH` — the **average-pool reciprocal** |
| `PPU_POOLING_PADDING_CFG` + `PADDING_VALUE_*` | pad top/bottom/left/right + pad value |
| `PPU_DST_BASE_ADDR` / `DST_SURF_STRIDE` / `MISC_CTRL` | pooled **cube** write-out |
| `PPU_RDMA_*` | the read-DMA that feeds the pooling unit |

That is the complete set. There is **no** `TRANSPOSE`, `RESHAPE`, `PERMUTE`, `SPLIT`,
`MERGE`, or `CONTRACT` register in the map — i.e. **no NVDLA-RUBIK** functionality, and no
RUBIK register page exists (Mesa `registers.xml` / our `npu_hw.h`). In NVDLA, layout reshape
is a *dedicated* engine (RUBIK) separate from pooling (PDP); RK3588's rocket NPU ships the
PDP (PPU) but **not** RUBIK.

## Why that kills the de-tile idea

The PPU reads a cube (NC1HWC2) and writes a cube (NC1HWC2) — pooling reduces the spatial
dims but does **not** move the C2 channel-atom into the row-major position. A 1×1 identity
"pool" just copies the cube; it does not transpose `[N/C2][M][C2] → [M][N]`. So the PPU
cannot produce the row-major `C[M,N]` the host wants, and cannot replace the NEON de-tile.

So the host output de-tile (`detile_accum_f16`, NEON) is **irreducible** on this HW.
Combined with the input side (the host `packB` scatter is irreducible — see
[matmul-as-conv.md](../matmul-as-conv.md)), **"no on-chip layout conversion" holds for both
directions**: the NPU has no engine that reorders between row-major and the cube layout.

The same no-RUBIK fact is why the framework layout ops — `TRANSPOSE`, `PAD`, `SLICE`,
`SPLIT` (and `RESHAPE`, `CONCAT`) — have **no on-NPU route**: there is no register program
that permutes/crops/extends a tensor, so they are host byte-copies. In the tflite-rocket
delegate they are claimed anyway (exact host kernels), not to offload compute but to keep a
real graph's `conv → layout-op → conv` in one delegated partition instead of bouncing to a
CPU node and back.

## The silver lining — pooling can run on the NPU

The PPU is a fully-decoded, idle pooling engine, and on-NPU `MaxPool` / `AveragePool` is
HW-validated: `gen_pool_fp16` + `rocket_pool_fp16` (PPU_RDMA-fed standalone job), average
via the `RECIP_KERNEL_*` reciprocal = **fp16(65536/k)**, max via the `-inf` pad fill. The
full register program and reciprocal format are HW-validated: MAX is bit-exact vs CPU,
AVG within fp16-recip tolerance. See [../encodings/ppu-pooling.md](../encodings/ppu-pooling.md)
(encoding), [../ppu-rknn-capture/](../ppu-rknn-capture/) (reproducible capture harness),
gate `tests/pool_fp16_rocket.c`, delegate opt-in `pool_npu` (gated in `convert_test`).

It is a prerequisite for keeping a partition's intermediates resident in cube layout so
adjacent NPU ops skip the host round-trip. Absent that, the delegate keeps pooling on the
host by default (the NPU path is a 2nd round-trip + NHWC↔cube transpose).

## References

Mesa `rocket/registers.xml`, our `include/npu_hw.h`, [NVDLA hwarch](https://nvdla.org/hw/v1/hwarch.html)
(PDP vs RUBIK). Related: [not-mac-bound.md](not-mac-bound.md) (readback is the bottleneck),
the NEON de-tile.
