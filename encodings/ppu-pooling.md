# On-NPU pooling via the PPU (MaxPool / AveragePool)

The PPU ("Planar Processing Unit") is the RK3588 NPU's **pooling** engine (NVDLA **PDP**
analog). It is not a de-tile engine ([ppu-pooling-not-detile.md](../perf/ppu-pooling-not-detile.md)),
but it does run pooling on the NPU, HW-validated. A pool is a **self-contained PPU + PPU_RDMA
job** (no CNA/CORE/DPU, no weights): PPU_RDMA reads the input NC1HWC2 cube, the PPU reduces each
kernel window per channel (max or average) and writes the output cube in the **same** NC1HWC2
layout.

Implemented: `gen_pool_fp16` (`src/npu_regcmd.c`, params `include/npu_pool.h`), runtime
`rocket_pool_fp16` (`src/rocket_pool.c`, `include/rocket_pool.h`), HW gate
`tests/pool_fp16_rocket.c`. Delegate opt-in `pool_npu` in `tflite-rocket`.

## Ground truth

The full PPU pooling program below is **HW-validated** — `gen_pool_fp16` runs bit-exact
for MAX and within fp16-recip tolerance for AVG (see HW validation, below). It was
developed with a reproducible rknn-toolkit2 capture harness
([ppu-rknn-capture/](../ppu-rknn-capture/)), including the average-reciprocal
`fp16(65536/k)` format.

## The PPU pooling program (fp16, C2=8)

Every geometry field is **(value − 1)**; cube strides are **bytes** (16-aligned). One
`NPUOP(target, value, reg)` per line; PPU target = `BLOCK_PPU|0x01` = `0x4001`, PPU_RDMA =
`0x8001`.

| reg | value |
|---|---|
| `PPU_S_POINTER` / `PPU_RDMA_S_POINTER` | `0xE` (POINTER_PP_MODE\|EXECUTER_PP_EN\|POINTER_PP_EN, == DPU) |
| `PPU_DATA_CUBE_IN_{WIDTH,HEIGHT,CHANNEL}` | iw−1, ih−1, c−1 |
| `PPU_DATA_CUBE_OUT_{WIDTH,HEIGHT,CHANNEL}` | ow−1, oh−1, c−1 |
| `PPU_OPERATION_MODE_CFG` | `FLYING_MODE`(bit4) \| `POOLING_METHOD`(bits[1:0]): **max=1, avg=0** |
| `PPU_POOLING_KERNEL_CFG` | (kw−1) \| (kh−1)<<8 \| (sx−1)<<16 \| (sy−1)<<20 |
| `PPU_RECIP_KERNEL_{WIDTH,HEIGHT}` | avg: **fp16(65536/k)**; max: 0 |
| `PPU_POOLING_PADDING_CFG` | L \| T<<4 \| R<<8 \| B<<12 (pad counts, **not** −1; 3-bit each) |
| `PPU_PADDING_VALUE_1_CFG` | 0 (avg) / `0xFC00` = −inf fp16 (max, when padded) |
| `PPU_DST_BASE_ADDR` | output IOVA (raw; field [31:4], BO page-aligned) |
| `PPU_DST_SURF_STRIDE` | oh·ow·C2·2 bytes |
| `PPU_DATA_FORMAT` | (oh·ow·C2·2) \| `PROC_PRECISION`(2) — INDEX_ADD[31:4] mirrors the out surf stride |
| `PPU_MISC_CTRL` | `BURST_LEN`(3) |
| `PPU_RDMA_CUBE_IN_{WIDTH,HEIGHT,CHANNEL}` | iw−1, ih−1, c−1 |
| `PPU_RDMA_SRC_BASE_ADDR` | input IOVA |
| `PPU_RDMA_SRC_LINE_STRIDE` | iw·C2·2 bytes |
| `PPU_RDMA_SRC_SURF_STRIDE` | ih·iw·C2·2 bytes |
| `PPU_RDMA_DATA_FORMAT` | `IN_PRECISION`(2) = fp16 |
| PC trailer | `OP_NONE` / `PC_REGISTER_AMOUNTS`=0 / `OP_40` / **`PC_OPERATION_ENABLE`=0x60** |

- **Enable mask `0x60`** = `PPU_OP_EN`(bit5) \| `PPU_RDMA_OP_EN`(bit6). This is the global
  block-participation mask written via target `0x81` (matmul/conv = `0x1D` =
  CNA\|CORE\|DPU\|DPU_RDMA). It carries **no bit0**: bit0 is CNA_OP_EN (unused for a pool), not a
  task-start bit — the `0x60` mask alone fires both blocks. The vendor emits **no** per-block
  `OPERATION_ENABLE` (0x6008/0x7008).
- **Flying vs standalone:** `OPERATION_MODE_CFG.FLYING_MODE=1` **and** PPU_RDMA
  fully armed, while `PPU_DATA_FORMAT.DPU_FLYIN=0`. So a standalone pool is a self-contained
  PPU job **fed by PPU_RDMA** (FLYING_MODE=1 regardless; an epilogue-of-conv would be
  DPU_FLYIN=1). No MRDMA trap — CNA/CORE/DPU are never enabled.

## Average reciprocal — `fp16(65536 / k)` per axis

The PPU has no divider; it multiplies the window sum by a per-axis reciprocal:
`avg = sum · recip_w · recip_h · 2⁻³² = sum/(kw·kh)`. The field holds the **fp16 bit
pattern of 65536/k** (`ppu_recip_kernel_fp16`). Verified: k=2→`0x7800`, k=3→`0x7555`,
asymmetric 2×3 per-axis. ~3-4 sig-fig precision (avg carries ~0.02-0.05% recip-quant
error — bit-identical to the vendor, same recip). **k ≥ 2** (k=1 → 65536 overflows fp16).

## Semantics / caveats

- All geometry fields (kernel, stride, dims) are **value − 1**. Pad counts are not.
- **Average divides by KH·KW** (count-include-pad = TRUE). TFLite AVERAGE_POOL_2D divides by
  the *valid* count, so a **padded average diverges at the border** — the delegate routes
  average to the NPU **only when VALID** (pad=0). MAX with any pad is fine (the −inf pad
  fill never wins ≡ clamp-to-image).
- MAX is **bit-exact** vs CPU; AVG within a small tolerance (recip + fp16 rounding).
- Single job, per-channel (no channel/spatial tiling yet — pooling has no weights so CBUF
  pressure is low; large spatial is a follow-on if needed).

## HW validation

`tests/pool_fp16_rocket.c` (CTest `pool_fp16_rocket`): max & avg, k=2/3/7, stride 1/2,
C=8/16/24 (single- and multi-C-plane), global pooling, padded max — **all PASS** on the RK1
(@600 MHz), MAX bit-exact, AVG ≤ 0.002. `tflite-rocket` `convert_test` NPU-PPU path
(`pool_npu`) bit-exact incl. max 3×3 SAME + ReLU. Cube layout = the conv feature cube
(`feature_data`, C2=8), so input packB / output de-tile reuse the conv path.

End-to-end: `libtflite_rocket.so` (the external delegate, `pool_npu` branch) **builds + runs
on the RK1** — `tflite-rocket/tools/run_pool_delegate.py` loads it via `tf.lite` on float pool
models; all match the CPU interpreter and `profile=1` reports `pool (npu-ppu)`. Build needs the
TFLite C-API headers (`-DTFLITE_DIR`, version-matched: sparse-clone the TF tag's
`tensorflow/lite/{core/c,c}` + `tensorflow/compiler/mlir/lite/core/c`) and the rocketnpu
install must include the internal headers `npu_dpu.h`/`npu_cna.h`/`npu_hw.h` (added to
`ROCKETNPU_PUBLIC_HEADERS`).

## int8 / uint8 pooling — no native int8 PPU precision (route through fp16)

**NPU FACT.** The PPU has **no native int8 pooling mode** [HW sweep]. A pool job
with `PPU_DATA_FORMAT.PROC_PRECISION = int8 (0)` and `PPU_RDMA_DATA_FORMAT.IN_PRECISION = 0` over a
packed **int8 C2=16** cube does not pool in int8 — the HW reads the byte stream as **fp16** and
emits garbage (small, near-constant values = two int8 bytes mis-read as one fp16). Measured on the
RK1 2026-06-22: native-int8 max/avg over C2=16 cubes failed every shape (`maxd` ≈ 100+ vs int8
golden). Every pool therefore uses `PROC_PRECISION(2)` / `IN_PRECISION(2)` = **fp16**, even inside
int8 models.

**Therefore int8/uint8 pooling routes through the fp16 PPU path** (`rocket_pool_int8` /
`rocket_pool_uint8`, `src/rocket_pool.c`): every int8 (−128..127) and uint8 (0..255) value is
**exactly representable in fp16**, so lifting the feature to fp16, running the fp16 PPU job, and
narrowing back is:
- **MAX: bit-exact** (fp16 max of exact integers == int max; round-trip lossless).
- **AVG: the fp16(65536/k) recip**, then round-to-nearest int8 (matches the fp16 path).
- **uint8**: recentered by −128 before pooling (MAX is shift-invariant; `avg(x−128)+128==avg(x)`)
  so the fp16 domain stays small; output clamped to [0,255].

`gen_pool_fp16` / the regcmd are unchanged (no int8 precision plumbing — it doesn't work). The
standalone int8 pool is **not** a perf win over host pooling (it adds int8↔fp16 conversion around
the same fp16 job); its value is (a) a cube-resident int8/uint8 pooling primitive for the
fused-partition path, and (b) this documented negative. HW gate `tests/pool_int8_rocket.c`
(CTest `pool_int8_rocket`, 13/13): int8 & uint8 MAX (single/multi-C-plane, stride, global,
same-pad, C-not-%16) bit-exact; int8 AVG within ±1 ULP.
