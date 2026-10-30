# On-NPU pooling via the PPU (MaxPool / AveragePool)

The PPU ("Planar Processing Unit") is the RK3588 NPU's pooling block, the counterpart of the
NVDLA PDP. A pool is a self-contained PPU + PPU_RDMA job (no CNA/CORE/DPU, no weights).
PPU_RDMA reads the input NC1HWC2 cube, and the PPU reduces each kernel window per channel. It
writes the output cube in the same NC1HWC2 layout. `PPU_OPERATION_MODE_CFG` selects the
method, and `PPU_POOLING_KERNEL_CFG` the window and the stride. The PPU cannot de-tile
([ppu-pooling-not-detile.md](../perf/ppu-pooling-not-detile.md)).

## Support status

| Pool | Datatype | Status |
|---|---|---|
| MAX, AVG | fp16 | Supported |
| MAX, AVG | int8, uint8, computed on the fp16 program | Supported |
| MAX, MIN, AVG | int8, native | Bit-exact on hardware. No library entry |
| A window past 16×16, or a stride past 16 | Any | Not supported. The fields are 4 bits |
| AVG with padding, divided by the valid count as TFLite defines it | Any | Not supported. The PPU divides by `kh·kw` |

## Registers

The program below comes from a reproducible rknn-toolkit2 capture harness
([ppu-rknn-capture/](../ppu-rknn-capture/)), including the average-reciprocal
`fp16(65536/k)` format. It is the full PPU pooling program, and it is validated on hardware
[HW sweep].

### The PPU pooling program (fp16, C2=8)

Every geometry field (kernel, stride, dims) is **(value − 1)**, and cube strides are bytes
(16-aligned). Pad counts are not. Each line is one `NPUOP(target, value, reg)`. The PPU target
is `BLOCK_PPU|0x01` = `0x4001`, and PPU_RDMA is `0x8001`.

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

Two of these fields need more than their values:

- **Enable mask `0x60`** = `PPU_OP_EN`(bit5) \| `PPU_RDMA_OP_EN`(bit6). It is the global
  block-participation mask written via target `0x81` (matmul/conv = `0x1D` =
  CNA\|CORE\|DPU\|DPU_RDMA). It has **no bit0**: bit0 is CNA_OP_EN (unused for a pool), not a
  task-start bit, and the `0x60` mask alone starts both blocks. The vendor emits no per-block
  `OPERATION_ENABLE` (0x6008/0x7008).
- **Flying vs standalone:** a standalone pool sets `OPERATION_MODE_CFG.FLYING_MODE=1` with
  PPU_RDMA fully armed, and `PPU_DATA_FORMAT.DPU_FLYIN=0`. A standalone pool is therefore a
  self-contained PPU job fed by PPU_RDMA. `FLYING_MODE` = 1 is the TRM's "outside" source (0 is
  the DPU) [TRM]. A pool fed by the DPU as a convolution's epilogue would take `FLYING_MODE` = 0,
  with `DPU_FLYIN` set [expected]. No capture or hardware test covers that encoding. There is
  no MRDMA trap, because CNA/CORE/DPU are
  never enabled.

### Average reciprocal: `fp16(65536 / k)` per axis

The PPU has no divider. It multiplies the window sum by a per-axis reciprocal:
`avg = sum · recip_w · recip_h · 2⁻³² = sum/(kw·kh)`. The field holds the fp16 bit pattern
of 65536/k. Verified values: k=2->`0x7800`, k=3->`0x7555`, and asymmetric 2×3 per-axis.

## Programming requirements

The PPU's fields set these limits:

- The kernel and stride fields are 4 bits each, so a window is at most 16×16 and a stride at
  most 16 [source-confirmed, Mesa `registers.xml`].
- An average requires **k >= 2** (k=1 -> 65536 overflows fp16).

> [!CAUTION]
> An average divides by kh·kw (count-include-pad = true). TFLite AVERAGE_POOL_2D divides by
> the valid count, so a padded average diverges from TFLite's at the border. MAX with any pad
> matches, because the −inf pad fill never wins, which is clamp-to-image.

## Accuracy

**Conditions:** RK3588 (RK1), NPU clock ~816 MHz.

MAX is bit-exact against the CPU, and AVG is within a small tolerance from the reciprocal and
fp16 rounding, <= 0.002 [HW sweep]. The reciprocal's precision is ~3-4 significant figures.
An average has ~0.02-0.05% reciprocal-quantization error, bit-identical to the vendor's, which
uses the same reciprocal.

## int8 / uint8 pooling: the PPU pools natively in int8

**Conditions:** RK3588 (Turing RK1), 2026-09-20.

The PPU has a native int8 pooling precision, and MAX, MIN and AVG are all bit-exact on it
[HW sweep, 2026-09-20]. Two precision fields select it, and two encodings are
datatype-dependent and easy to miss:

| what | register | int8 value | the fp16 path's value |
|---|---|---|---|
| processing precision | `PPU_DATA_FORMAT[2:0]` (`0x6084`) | 0 | 2 |
| input storage width | `PPU_RDMA_DATA_FORMAT[1:0]` (`0x7030`) | 1 | 2 |
| average reciprocal | `PPU_RECIP_KERNEL_W/H` (`0x6038`/`0x603C`) | integer Q16 `0x10000/k` | `fp16(65536/k)` bits |
| MAX pad fill | `PPU_PADDING_VALUE_1_CFG` (`0x6044`) | `0x0007FF80` (−128, sign-extended in the 19-bit field) | `0xFC00` (fp16 −inf) |

The feature cube is the packed int8 C2=16 cube, one byte an element, with the line and
surface strides in bytes as usual.

### The precision fields and the storage-width trap

`PROC_PRECISION` is the datatype selector, and software must program it to 0 for an int8
cube [HW sweep]. Every test point with `PROC_PRECISION = 0` and a non-zero storage width is
exact. Every test point at 1 (int16) or 2 (fp16) over an int8 cube is wrong.

`IN_PRECISION` is a storage width: the TRM enumerates `0x7030[1:0]` as `2'd0: 4bit; 2'd1: 8bit;
2'd2: 16bit; 2'd3: 32bit` [TRM, RK3588 Part1]. It does not behave as a strict element-width
selector. Software must program 1 (8-bit), the value the enumeration calls for. The measured
map [HW sweep]:

- `IN_PRECISION = 0` (4-bit) is wrong on every method.
- `1` (8-bit) is exact on MAX, MIN and AVG.
- `2` is also exact on all three.
- `3` is exact on MAX and MIN but fails on AVG, and its failure is not stable run to run.

> [!CAUTION]
> `IN_PRECISION` values 1, 2 and 3 all pass a max-pooling test, and 3 fails on an average. A
> test of MAX alone does not show that they differ.

### The integer average: reciprocal and rounding

Software must program the integer reciprocal for an int8 average [HW sweep]. The integer Q16
`0x10000/k` is exact, including on an asymmetric 2×3 window whose per-axis reciprocal does
not divide 65536.

> [!CAUTION]
> With `fp16(65536/k)` in the register, the int8 average is wrong by up to 44 on the shapes
> measured.

The PPU's integer average rounds half away from zero, except that an even window whose
remainder is exactly half steps back toward zero [HW sweep]. When the Q16 reciprocal is
inexact, that step always happens. When it is exact, the step happens only for an odd
quotient, which is round-half-to-even. The RK3576's PPU rounds the same way: the rule does not
depend on machine parameters, and it is unchanged between the two SoCs.

> [!NOTE]
> A naive round-half-away model disagrees with the hardware on 20% of outputs at ±1. Score
> an integer average against the rule above, not against a plain rounded division.

### Testing an integer pool

> [!CAUTION]
> A wrong precision pair gives output of the correct size with values in range, and no error
> is reported. A wrong pair can return an all-zero output on one run and incorrect values on
> the next. A test that checks only whether output was written passes both.

Test an integer pool against a CPU model of the PPU's own arithmetic, along the method axis
(max, min and avg). Include at least one padded and one asymmetric window. The pad fill and
the reciprocal are separately datatype-dependent, and each corrupts only part of the output.

### Measured shapes

The native int8 path is exact on these shapes [HW sweep, 2026-09-20]:

| Method | Shapes | Result |
|---|---|---|
| MAX | 2×2 s2, 3×3 s1, asymmetric 2×3 s1, C=32 (two channel groups), and a padded 2×2 | bit-exact |
| MIN | 2×2 s2 | bit-exact |
| AVG | 2×2 s2, 4×4 s4, asymmetric 2×3 s1 | bit-exact against the chip rounding model |

## Software support

The fp16 generator is `gen_pool_fp16` (`src/npu_regcmd.c`, params `include/npu_pool.h`). The
runtime entry is `rocket_pool_fp16` (`src/rocket_pool.c`, `include/rocket_pool.h`).
`ppu_recip_kernel_fp16` computes the average reciprocal. `gen_pool_fp16` refuses a window or
stride past the 4-bit fields. A pool is a single job, per-channel, with no channel or spatial
tiling. Pooling has no weights, so CBUF pressure is low.

The cube layout is the conv feature cube (`feature_data`, C2=8), so the input packB and the
output de-tile reuse the conv path.

The `tflite-rocket` delegate's opt-in is `pool_npu`. The delegate sends an average to the NPU
only for a VALID pool (pad=0).

### int8 and uint8 entries

`rocket_pool_int8` / `rocket_pool_uint8` (`src/rocket_pool.c`) run on the fp16 PPU path, and
are correct. Every int8 (−128..127) and uint8 (0..255) value is exactly representable in fp16.
Lifting the feature to fp16, running the fp16 job and narrowing back therefore gives MAX
bit-exact and AVG within ±1 ULP. The uint8 entry recenters by −128 before pooling (MAX is
shift-invariant, and `avg(x−128)+128 == avg(x)`) and clamps to [0,255].

The native int8 path has no library entry: `gen_pool_fp16` has no int8 precision parameter.
A native entry would remove the int8/fp16 conversion on either side of the job, and would
give a cube-resident int8 primitive for the fused-partition path. The pool itself is not
faster than host pooling at either precision.

## Evidence

### Accuracy

`tests/pool_fp16_rocket.c` (CTest `pool_fp16_rocket`) covers these cases:

- Max and avg
- k=2/3/7
- Stride 1/2
- C=8/16/24 (single- and multi-C-plane)
- Global pooling
- Padded max

All pass on the RK1 (NPU clock ~816 MHz). The `tflite-rocket` `convert_test` NPU-PPU path
(`pool_npu`) is bit-exact, including max 3×3 SAME + ReLU.

End-to-end, `libtflite_rocket.so` (the external delegate, `pool_npu` branch) builds and runs
on the RK1. `tflite-rocket/tools/run_pool_delegate.py` loads it via `tf.lite` on float pool
models. All match the CPU interpreter, and `profile=1` reports `pool (npu-ppu)`. The build needs
the TFLite C-API headers (`-DTFLITE_DIR`, version-matched: sparse-clone the TF tag's
`tensorflow/lite/{core/c,c}` + `tensorflow/compiler/mlir/lite/core/c`). The rocketnpu install
must also include the internal headers `npu_dpu.h`/`npu_cna.h`/`npu_hw.h` (added to
`ROCKETNPU_PUBLIC_HEADERS`).

### int8 / uint8 pooling

The precision map and the measured shapes are `tests/pool_int8_native_probe.c`, on the
Turing RK1. Three full runs gave the same set of exact test points. Failing test points vary
between runs: the `PROC_PRECISION = 1` points over an int8 cube, and the `IN_PRECISION = 3`
average. The pattern fits an undefined combination that reads a byte stream at the wrong width
[hypothesis].

The comparison of the three runs listed only its first three differing test points, all at
`PROC_PRECISION = 1`. The `IN_PRECISION = 3` average differed between two earlier runs on the
same input, in which every other `PROC_PRECISION = 0` test point gave the same result. It
returned all zeros in one of them and other wrong values in the other.

The rounding rule is `librocketnpu`'s RK3576 model: the RK3588 PPU reproduces
`rocket_pool_rk3576.c`'s `r76p_avg_round` bit for bit, while the naive round-half-away model
disagrees.

### Software support

The hardware test for the int8 and uint8 entries is `tests/pool_int8_rocket.c` (CTest
`pool_int8_rocket`). It covers int8 and uint8 MAX (single/multi-C-plane, stride, global,
same-pad, C-not-%16) bit-exact, int8 AVG within ±1 ULP. `tests/pool_int8_native_probe.c` is
the precision map above.
