# On-NPU pooling via the PPU (MaxPool / AveragePool)

The PPU ("Planar Processing Unit") is the RK3588 NPU's pooling engine (NVDLA PDP analog).
It is not a de-tile engine ([ppu-pooling-not-detile.md](../perf/ppu-pooling-not-detile.md)).
It runs pooling on the NPU, validated on hardware. A pool is a self-contained PPU +
PPU_RDMA job (no CNA/CORE/DPU, no weights). PPU_RDMA reads the input NC1HWC2 cube, and the
PPU reduces each kernel window per channel (max or average). It writes the output cube in
the same NC1HWC2 layout.

The generator is `gen_pool_fp16` (`src/npu_regcmd.c`, params `include/npu_pool.h`). The
runtime entry is `rocket_pool_fp16` (`src/rocket_pool.c`, `include/rocket_pool.h`), and the
hardware gate is `tests/pool_fp16_rocket.c`. The delegate opt-in is `pool_npu` in
`tflite-rocket`.

## Ground truth

The full PPU pooling program below is validated on hardware: `gen_pool_fp16` runs bit-exact
for MAX and within fp16-recip tolerance for AVG (see HW validation, below). The program
comes from a reproducible rknn-toolkit2 capture harness
([ppu-rknn-capture/](../ppu-rknn-capture/)), including the average-reciprocal
`fp16(65536/k)` format.

## The PPU pooling program (fp16, C2=8)

Every geometry field is **(value − 1)**, and cube strides are bytes (16-aligned). Each line
is one `NPUOP(target, value, reg)`. The PPU target is `BLOCK_PPU|0x01` = `0x4001`, and PPU_RDMA
is `0x8001`.

| reg | value |
|---|---|
| `PPU_S_POINTER` / `PPU_RDMA_S_POINTER` | `0xE` (POINTER_PP_MODE\|EXECUTER_PP_EN\|POINTER_PP_EN, == DPU) |
| `PPU_DATA_CUBE_IN_{WIDTH,HEIGHT,CHANNEL}` | iw−1, ih−1, c−1 |
| `PPU_DATA_CUBE_OUT_{WIDTH,HEIGHT,CHANNEL}` | ow−1, oh−1, c−1 |
| `PPU_OPERATION_MODE_CFG` | `FLYING_MODE`(bit4) \| `POOLING_METHOD`(bits[1:0]): max=1, avg=0 |
| `PPU_POOLING_KERNEL_CFG` | (kw−1) \| (kh−1)<<8 \| (sx−1)<<16 \| (sy−1)<<20 |
| `PPU_RECIP_KERNEL_{WIDTH,HEIGHT}` | avg: fp16(65536/k), max: 0 |
| `PPU_POOLING_PADDING_CFG` | L \| T<<4 \| R<<8 \| B<<12 (pad counts, **not** −1, 3-bit each) |
| `PPU_PADDING_VALUE_1_CFG` | 0 (avg) / `0xFC00` = −inf fp16 (max, when padded) |
| `PPU_DST_BASE_ADDR` | output IOVA (raw, field [31:4], BO page-aligned) |
| `PPU_DST_SURF_STRIDE` | oh·ow·C2·2 bytes |
| `PPU_DATA_FORMAT` | (oh·ow·C2·2) \| `PROC_PRECISION`(2). INDEX_ADD[31:4] mirrors the out surf stride |
| `PPU_MISC_CTRL` | `BURST_LEN`(3) |
| `PPU_RDMA_CUBE_IN_{WIDTH,HEIGHT,CHANNEL}` | iw−1, ih−1, c−1 |
| `PPU_RDMA_SRC_BASE_ADDR` | input IOVA |
| `PPU_RDMA_SRC_LINE_STRIDE` | iw·C2·2 bytes |
| `PPU_RDMA_SRC_SURF_STRIDE` | ih·iw·C2·2 bytes |
| `PPU_RDMA_DATA_FORMAT` | `IN_PRECISION`(2) = fp16 |
| PC trailer | `OP_NONE` / `PC_REGISTER_AMOUNTS`=0 / `OP_40` / `PC_OPERATION_ENABLE`=0x60 |

Two of these fields carry more than their values:

- **Enable mask `0x60`** = `PPU_OP_EN`(bit5) \| `PPU_RDMA_OP_EN`(bit6). This is the global
  block-participation mask written via target `0x81` (matmul/conv = `0x1D` =
  CNA\|CORE\|DPU\|DPU_RDMA). It carries **no bit0**: bit0 is CNA_OP_EN (unused for a pool), not a
  task-start bit, and the `0x60` mask alone fires both blocks. The vendor emits no per-block
  `OPERATION_ENABLE` (0x6008/0x7008).
- **Flying vs standalone:** a standalone pool sets `OPERATION_MODE_CFG.FLYING_MODE=1` with
  PPU_RDMA fully armed, and `PPU_DATA_FORMAT.DPU_FLYIN=0`. A standalone pool is therefore a
  self-contained PPU job fed by PPU_RDMA (FLYING_MODE=1 regardless, while an
  epilogue-of-conv would be DPU_FLYIN=1). There is no MRDMA trap, because CNA/CORE/DPU are
  never enabled.

## Average reciprocal: `fp16(65536 / k)` per axis

The PPU has no divider. It multiplies the window sum by a per-axis reciprocal:
`avg = sum · recip_w · recip_h · 2⁻³² = sum/(kw·kh)`. The field holds the fp16 bit pattern
of 65536/k (`ppu_recip_kernel_fp16`). Verified values: k=2->`0x7800`, k=3->`0x7555`, and
asymmetric 2×3 per-axis. The precision is ~3-4 sig-fig (avg carries ~0.02-0.05% recip-quant
error, bit-identical to the vendor, same recip). **k >= 2** is required (k=1 -> 65536
overflows fp16).

## Semantics and caveats

- All geometry fields (kernel, stride, dims) are value − 1. Pad counts are not.
- The kernel and stride fields are 4 bits each, so a window is at most 16×16 and a stride at
  most 16 [source-confirmed, Mesa `registers.xml`]. `gen_pool_fp16` refuses past them.
- **Average divides by kh·kw** (count-include-pad = true). TFLite AVERAGE_POOL_2D divides by
  the valid count, so a **padded average diverges at the border**. The delegate therefore
  routes average to the NPU only when VALID (pad=0). MAX with any pad is fine (the −inf pad
  fill never wins, which is clamp-to-image).
- MAX is bit-exact vs CPU, and AVG is within a small tolerance (recip + fp16 rounding).
- A pool is a single job, per-channel, with no channel or spatial tiling yet. Pooling has no
  weights, so CBUF pressure is low, and large spatial is a follow-on if needed.

## HW validation

`tests/pool_fp16_rocket.c` (CTest `pool_fp16_rocket`) covers these cases:

- Max and avg
- k=2/3/7
- Stride 1/2
- C=8/16/24 (single- and multi-C-plane)
- Global pooling
- Padded max

All pass on the RK1 (@600 MHz), with MAX bit-exact and AVG <= 0.002. The `tflite-rocket`
`convert_test` NPU-PPU path (`pool_npu`) is bit-exact, including max 3×3 SAME + ReLU. The
cube layout is the conv feature cube (`feature_data`, C2=8), so the input packB and the
output de-tile reuse the conv path.

End-to-end, `libtflite_rocket.so` (the external delegate, `pool_npu` branch) builds and runs
on the RK1. `tflite-rocket/tools/run_pool_delegate.py` loads it via `tf.lite` on float pool
models. All match the CPU interpreter, and `profile=1` reports `pool (npu-ppu)`. The build needs
the TFLite C-API headers (`-DTFLITE_DIR`, version-matched: sparse-clone the TF tag's
`tensorflow/lite/{core/c,c}` + `tensorflow/compiler/mlir/lite/core/c`). The rocketnpu install
must also include the internal headers `npu_dpu.h`/`npu_cna.h`/`npu_hw.h` (added to
`ROCKETNPU_PUBLIC_HEADERS`).

## int8 / uint8 pooling: the PPU pools natively in int8

The PPU has a native int8 pooling precision, and MAX, MIN and AVG are all bit-exact on it
[HW sweep, Turing RK1, 2026-09-20, `tests/pool_int8_native_probe.c`]. The configuration is a
pair of precision fields plus two encodings that are datatype-dependent and easy to miss:

| what | register | int8 value | the fp16 path's value |
|---|---|---|---|
| processing precision | `PPU_DATA_FORMAT[2:0]` (`0x6084`) | 0 | 2 |
| input storage width | `PPU_RDMA_DATA_FORMAT[1:0]` (`0x7030`) | 1 | 2 |
| average reciprocal | `PPU_RECIP_KERNEL_W/H` (`0x6038`/`0x603C`) | integer Q16 `0x10000/k` | `fp16(65536/k)` bits |
| MAX pad fill | `PPU_PADDING_VALUE_1_CFG` (`0x6044`) | `0x0007FF80` (−128, sign-extended in the 19-bit field) | `0xFC00` (fp16 −inf) |

The feature cube is the packed int8 C2=16 cube, one byte an element, with the line and
surface strides in bytes as usual.

### The precision fields and the storage-width trap

`PROC_PRECISION` is the datatype selector and is the load-bearing one. Every cell with
`PROC_PRECISION = 0` and a non-zero storage width is exact. Every cell at 1 (int16) or 2
(fp16) over an int8 cube is wrong.

`IN_PRECISION` is a storage width: the TRM enumerates `0x7030[1:0]` as `2'd0: 4bit; 2'd1: 8bit;
2'd2: 16bit; 2'd3: 32bit` [TRM, RK3588 Part1]. It does not behave as a strict element-width
selector here. Program against the measured map rather than the enumeration:

- `IN_PRECISION = 0` (4-bit) is wrong on every method. This is the value to avoid.
- `1` (8-bit) is exact on MAX, MIN and AVG. **Use 1**: it is the value the enumeration calls
  for.
- `2` is also exact on all three, which is why nothing forces the issue on a MAX-only corpus.
- `3` is exact on MAX and MIN but fails on AVG, and its failure is not stable run to run.
  Do not read "1, 2 and 3 are equivalent" off a max-pooling gate.

### The integer average: reciprocal and rounding

**The average's reciprocal must be the integer one.** With `fp16(65536/k)` in the register the
int8 average is wrong by up to 44 on the shapes measured, so this is not a tolerance question.
The integer Q16 `0x10000/k` is exact, including on an asymmetric 2×3 window whose per-axis
reciprocal does not divide 65536.

The average's rounding is the RK3576's model, unchanged. The RK3588 PPU reproduces
`rocket_pool_rk3576.c`'s `r76p_avg_round` bit for bit. It rounds half away from zero,
except that an even window whose remainder is exactly half steps back toward zero. When the Q16
reciprocal is inexact, that step always happens. When it is exact, the step happens only for an
odd quotient, which is round-half-to-even.

A naive round-half-away model disagrees with the hardware on 20% of outputs at ±1, while the
chip model is exact. So **score an integer average against that model, not against a plain
rounded division**. This is a machine-parameter-independent algebra that ported between the two
parts without change.

### Gating an integer pool

**A wrong precision pair here computes a full, correctly sized, entirely plausible surface.**
The failing cells in the map return values in range, not faults. One of them
returns an all-zero surface that a gate scoring only "did it write" would pass.

Gate an integer pool against a CPU model of the chip's own arithmetic, along the method axis
(max, min and avg). Include at least one padded and one asymmetric window. Both matter
because the pad fill and the reciprocal are separately datatype-dependent, and each corrupts
only part of the surface.

### Measured envelope

The measured envelope [HW sweep, Turing RK1, 2026-09-20, identical across three full runs, the
exact-cell set stable]:

| Method | Shapes | Result |
|---|---|---|
| MAX | 2×2 s2, 3×3 s1, asymmetric 2×3 s1, C=32 (two channel groups), and a padded 2×2 | bit-exact |
| MIN | 2×2 s2 | bit-exact |
| AVG | 2×2 s2, 4×4 s4, asymmetric 2×3 s1 | bit-exact against the chip rounding model |

Run-to-run variation exists only in the invalid cells (`PROC_PRECISION = 1` over an int8
cube). That is what an undefined combination reading a byte stream at the wrong width looks
like.

## The shipping int8 and uint8 entries

`rocket_pool_int8` / `rocket_pool_uint8` (`src/rocket_pool.c`) route through the fp16 PPU
path, and are correct. Every int8 (−128..127) and uint8 (0..255) value is exactly
representable in fp16. So lifting the feature to fp16, running the fp16 job and narrowing back
gives MAX bit-exact and AVG within ±1 ULP. The uint8 entry recenters by −128 before pooling
(MAX is shift-invariant, and `avg(x−128)+128 == avg(x)`) and clamps to [0,255].

The native path is a measured capability that is not plumbed. What it is worth is the
int8/fp16 conversion on either side of the job, and a cube-resident int8 primitive for the
fused-partition path. It is not the pool itself, which is not a win over host pooling either
way. `gen_pool_fp16` carries no int8 precision plumbing. A native entry would be a second
generator or a precision parameter on that one.

The hardware gate `tests/pool_int8_rocket.c` (CTest `pool_int8_rocket`) covers the shipping
fp16-routed path: int8 and uint8 MAX (single/multi-C-plane, stride, global, same-pad,
C-not-%16) bit-exact, int8 AVG within ±1 ULP. `tests/pool_int8_native_probe.c` is the
precision map above.
