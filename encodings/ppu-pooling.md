# On-NPU pooling via the PPU (MaxPool / AveragePool)

The PPU ("Planar Processing Unit") is the RK3588 NPU's pooling block, the counterpart of the
NVDLA PDP. A pool is a self-contained PPU + PPU_RDMA task (no CNA/CORE/DPU, no weights).
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
| A window past 16×16, a stride past 16, or a pad past 7 | Any | Not supported. The kernel and stride fields are 4 bits, and the pad fields 3 |
| An output wider than its window's allowance (§"Output-width allowance") | Any | Not supported in one task. The PPU writes incorrect output. Software runs it as column bands |
| A window past 8×8, or a stride past the window | Any | Not supported. The PPU writes incorrect output at an output width of 8 or less |
| AVG with padding, divided by the valid count as TFLite defines it | Any | Not supported. The PPU divides by `kh·kw` |

## Registers

The program below comes from a reproducible rknn-toolkit2 capture harness
([ppu-rknn-capture/](../ppu-rknn-capture/)), including the average-reciprocal
`fp16(65536/k)` format. It is the full PPU pooling program, and it is validated on hardware
[HW sweep].

### The PPU pooling program (fp16, C2=8)

Every geometry field (kernel, stride, dims) is **(value − 1)**, and cube strides are bytes
(16-aligned). Pad counts are not. Each line is one `REGCMD(target, reg, value)`. The PPU target
is `NPU_TARGET_PPU|0x01` = `0x4001`, and PPU_RDMA is `0x8001`.

| reg | value |
|---|---|
| `PPU_S_POINTER` / `PPU_RDMA_S_POINTER` | `0xE` (POINTER_PP_MODE\|EXECUTER_PP_EN\|POINTER_PP_EN, == DPU) |
| `PPU_DATA_CUBE_IN_{WIDTH,HEIGHT,CHANNEL}` | iw−1, ih−1, c−1 |
| `PPU_DATA_CUBE_OUT_{WIDTH,HEIGHT,CHANNEL}` | ow−1, oh−1, c−1 |
| `PPU_OPERATION_MODE_CFG` | `FLYING_MODE`(bit4) \| `POOLING_METHOD`(bits[1:0]): max=1, avg=0, min=2 [TRM] |
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
| PC trailer | `TARGET_NONE` / `PC_REGISTER_AMOUNTS`=0 / `TARGET_PRE_ENABLE` / `PC_OPERATION_ENABLE`=0x60 |

Two of these fields need more than their values:

- **Enable mask `0x60`** = `PPU_OP_EN`(bit5) \| `PPU_RDMA_OP_EN`(bit6), the block bitmap of
  [regcmd-task-model.md](regcmd-task-model.md) §"`PC_OPERATION_ENABLE` as a block bitmap". It
  starts both blocks with bit 0 clear. The vendor emits no per-block `OPERATION_ENABLE`
  (0x6008/0x7008).
- **Flying vs standalone:** a standalone pool sets `OPERATION_MODE_CFG.FLYING_MODE=1` with
  PPU_RDMA fully armed, and `PPU_DATA_FORMAT.DPU_FLYIN=0`. A standalone pool is therefore a
  self-contained PPU task fed by PPU_RDMA. `FLYING_MODE` = 1 is the TRM's "outside" source (0 is
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
- The pad fields are 3 bits each, so a pad is at most 7 on each side [source-confirmed, Mesa
  `registers.xml`].
- An average requires **k >= 2** (k=1 -> 65536 overflows fp16).

> [!NOTE]
> A pooling program raises no DPU interrupt, and mainline `rocket` enables only the DPU
> interrupts. Each pool submit there retires at the 500 ms job timeout with a core reset, and
> its output is correct ([../perf/pool-completion.md](../perf/pool-completion.md)).

> [!CAUTION]
> An average divides by kh·kw (count-include-pad = true). TFLite AVERAGE_POOL_2D divides by
> the valid count, so a padded average diverges from TFLite's at the border. MAX with any pad
> matches, because the −inf pad fill never wins, which is clamp-to-image.

### Output-width allowance

**Conditions:** RK3588 (Turing RK1), `rocket` 1.3.0, NPU clock 928 MHz, 2026-11-02. fp16, 16
channels, four output rows, no pad, MAX. AVG at 2×2 stride 2 and 3×3 stride 1 gives the same
limits.

A pooling task has an allowance on its output width. Past it the task writes incorrect output
across the whole output [HW sweep 2026-11-02]. Most windows past it also raise no completion,
and the job retires at the driver's timeout. Repeated timeouts wedge the NPU
([../chips/rk3588.md](../chips/rk3588.md) Erratum E13), so a sweep past the allowance must bound
how many configurations can time out. The allowance depends on the kernel height and the
vertical stride. A 3×1 window has the 3×3 window's allowance, and a 1×3 window has none up to an
output width of 256:

| Window (kh×kw) | Stride (sy×sx) | Exact up to output width | Incorrect from |
|---|---|---:|---:|
| 2×2 | 2×2 | 128 | 129 |
| 2×2 | 2×1 | 65, the widest measured | |
| 2×2, 2×1 | 1×1 | 64 | 65 |
| 2×2 | 1×2 | 64 | 65 |
| 3×3, 3×1 | 1×1 | 32 | 33 |
| 3×3 | 2×2 and 3×3 | 64 | 65 |
| 4×4 | 1×1 | 32 | 33 |
| 4×4 | 2×2 and 4×4 | 64 | 65 |
| 5×5 and 8×8 | 1×1 | 16 | 17 |
| 5×5, 6×6, 7×7 and 8×8 | Equal to the window | 64 | 65 |
| 1×2, 1×3 | 1×1 | 256, the widest measured | |
| 1×16 | 1×16 | 1 | 2 |
| 12×12, 16×16 and 16×1 | Equal to the window | | 1 |
| 16×16 | 1×1 | | 1 |
| 2×2 | 3×3, 2×3 and 3×2 | | 8 or less |

At one output row and a stride equal to the window, 10×10 to 16×16 windows are exact at
width 1 [HW sweep 2026-11-02, one run]. They are incorrect from a width of 2. A 9×9 window at stride 9
was incorrect at a 1×1 output in the same run.

A stride past the window on either axis is not a narrower allowance. A 2×2 window at stride
3×3, 2×3 or 3×2 writes zeros at an output width of 8, and the job retires at the driver's
timeout. It is exact only at a 1×1 output [HW sweep 2026-11-06]. A 1×1 window at a horizontal
stride of 2 was exact at width 8 [HW sweep 2026-11-06, one run].

Every measured cell fits one rule for windows to 8×8 at a stride up to the window. The
allowance follows `o = ceil(kh / sy)`, the output rows one input row feeds: 64 at `o` 1-2,
32 at `o` 3-4 and 16 at `o` 5-8. The 2×2 window at stride 2 has 128, and a one-row window to
1×3 at stride 1 has 256 or more. Each cell below is exact at the rule's width, MAX, and AVG
where marked [HW sweep 2026-11-06]. The cells vary the kernel width, the horizontal stride and
the pad. Each ran at the rule's width only, so its allowance can be wider:

| Window (kh×kw) | Stride (sy×sx) | Pad | Exact at output width |
|---|---|---|---:|
| 6×6 (MAX, AVG), 7×7 | 1×1 | 0 | 16 |
| 8×2 | 1×1 | 0 | 16 |
| 5×5 | 1×1 | 2 | 16 |
| 5×5, 6×6, 7×7, 8×8 (MAX, AVG) | 2×2 | 0 | 32 |
| 8×8 | 3×3 | 0 | 32 |
| 4×2 | 1×1 | 0 | 32 |
| 3×3 | 1×2 | 0 | 32 |
| 3×3 | 1×1 | 1 | 32 |
| 3×3 | 2×2 | 1 | 56 |
| 4×4, 5×5, 6×6 (MAX, AVG) | 3×3 | 0 | 64 |
| 7×7, 8×8 | 4×4 | 0 | 64 |
| 2×8 (MAX, AVG) | 1×1 | 0 | 64 |
| 1×5, 1×8 | 1×1 | 0 | 64 |
| 1×8 | 1×8 | 0 | 64 |

> [!CAUTION]
> A pool past its allowance produces a complete, correctly sized, incorrect output. Software
> must keep each pooling task's output width within the allowance for its window and vertical
> stride. A wider pool must run as column bands, each band its own task (§"Column bands").

The RK3576's PPU has an allowance of the same kind with different values
([../chips/rk3576.md](../chips/rk3576.md) §"The pooling output-width allowance").

**Not characterized:** a mechanism for the rule above, and windows from 9×9 to 16×16 at more
than one output width or row. Also a stride past the window at widths 2 to 7, a pad past 2,
and the native int8 program.

### Column bands

**Conditions:** RK3588 (Turing RK1), `rocket` 1.3.0, NPU clock 928 MHz, 2026-11-06. fp16.

A pool wider than its allowance computes correctly as column bands [HW sweep 2026-11-06]. Each
band produces a run of output columns within the allowance and reads the input columns its
windows reach. It takes the left pad only at the plane's left edge, and the right pad only where
its windows reach past the right edge. Each band's input is its own dense cube, and the bands
run as the tasks of one job with the PPU completion. These shapes are exact, MAX bit-exact and
AVG within fp16 rounding:

| Window, stride, pad | Plane (H×W) | Output width | Bands |
|---|---|---:|---:|
| 2×2 s2 | 4×512 | 256 | 2 |
| 3×3 s1, AVG | 4×130 | 128 | 4 |
| 7×7 s7, AVG | 7×896 | 128 | 2 |
| 5×5 s1 pad 2 (YOLO's SPPF) | 20×20 | 20 | 2 |
| 3×3 s2 pad 1 | 8×224 | 112 | 2 |
| 3×3 s1 pad 1, all-negative input | 6×80 | 80 | 3 |

### Cube dimensions at the field maximum

**Conditions:** as above.

The PPU's height and channel axes compute at 8190, 8191 and 8192, the last of which fills the
13-bit field (8191). A 1×1 MAX and a 2×2 stride-2 MAX are exact at each, and a 2×2 stride-1 AVG
is within fp16 rounding (0.002) [HW sweep 2026-11-02]. A 1×1 MAX along the width is exact at input widths 8190 and 8191. At an
input width of 8192 it was incorrect in two runs. Each ran after earlier jobs had retired at
the timeout, so the width axis at 8192 is not characterized. Erratum E9, the 13-bit cube-dimension
limit, does not apply to the PPU's height and channel axes.

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
- `3` is exact on MAX and MIN. On AVG it is wrong, and the wrong output differed between runs.

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
stride past the 4-bit fields. A pool is a single job, per-channel, with no channel tiling.
Pooling has no weights, so CBUF pressure is low.

`rocket_pool_fp16_max_ow` returns the rule's allowance for a window (§"Output-width
allowance"). It returns 0 for a window past 8×8 or a stride past the window.
`rocket_pool_fp16` runs an output wider than the allowance as up to 256 evenly sized column
bands in one job (`rocket_pool_fp16_bands`, §"Column bands"). `rocket_pool_fp16_plan` refuses
a window the allowance gives 0, and a pool needing more bands. The int8 and uint8 entries run
on the plan, and so does a frontend that asks it, so both decline those pools.

The cube layout is the conv feature cube (`feature_index`, C2=8), so the input packB and the
output de-tile reuse the conv path.

The `tflite-rocket` delegate's opt-in is `pool_npu`. The delegate sends an average to the NPU
only for a VALID pool (pad=0).

### int8 and uint8 entries

`rocket_pool_int8` / `rocket_pool_uint8` (`src/rocket_pool.c`) run on the fp16 PPU path, and
are correct. Every int8 (−128..127) and uint8 (0..255) value is exactly representable in fp16.
The entries lift the feature to fp16, run the fp16 job, round the result to nearest and clamp
it to the output type. MAX is then bit-exact, and AVG within ±1 ULP of the exactly rounded
average. The uint8 entry recenters by −128 before pooling (MAX is shift-invariant, and
`avg(x−128)+128 == avg(x)`) and clamps to [0,255]. The PPU's native integer rounding (§"The
integer average: reciprocal and rounding") does not apply to these entries.

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

### Output-width allowance

A probe ran `rocket_pool_fp16` at a ladder of output widths for each window. The ladder was 8,
14, 16, 17, 24, 30, 32, 33, 40, 48, 56, 63-65, 96, 112, 127-129, 192 and 256. For the larger
windows it was 1-8, 12, 15-17, 24, 31-33, 48 and 63-65. Inputs were hashed uniform values in
[−4, 4] over an output stamped with NaN. Every element was compared with `rocket_pool_ref_fp16`,
and a difference past `0.004 + 0.004·|ref|` was a failure. An AVG element within it differs by
fp16 rounding.

In each run the first failure fell at a width that depends on the window. The windows
measured after a failure were again exact below their own limit. Past the allowance the
2×2 stride-2 window returned incorrect values in milliseconds. The other windows returned
incorrect values at the 500 ms job timeout.

The one-row measurements ran last, after more than 500 timed-out jobs, and the NPU stopped
completing jobs at the end of that run. Only their exact results are recorded above, with the
9×9 result, which was the first failure of that run.

A second run on a freshly booted board, one process a configuration, reproduced each boundary
it tested. A 1×3 window was exact at 254. The boundaries, last exact and first incorrect:

- 2×2 stride 2: 128 and 129, MAX and AVG
- 3×3 stride 1: 32 and 33
- 5×5 stride 1: 16 and 17
- 5×5 stride 5: 64 and 65

Each exact configuration ran again after a failing one and was again exact. A failure
therefore leaves no state for the next process. YOLO's SPPF pool, 5×5 at stride 1 and pad 2 over a 20×20 plane, is past
the allowance: 10744 of 12800 elements were incorrect. The run had four job timeouts.

The rule's cells and the stride-past-window cases ran on 2026-11-06 through `rocket_pool_fp16`,
one process a configuration. Each ran exactly at the rule's width, so as one task, and the
kernel log was read after each. The inputs were hashed small integers in [−6, 6] over a
NaN-stamped output (`tests/pool_fp16_rocket`, single-shape mode), C 16, four or more output
rows.

MAX was compared exactly and AVG within 0.06. The AVG cells differed by at most 0.004.

The sequence stopped at its first failure, the 2×2 window at stride 3 at width 64. Four
further runs bounded it.

Width 1 was exact. Width 8 was incorrect, with a job timeout, at
strides 3×3, 2×3 and 3×2. The 1×1 window at stride 2 was exact at width 8. The cells after
the stop then ran in fresh processes. The session had four job timeouts, each a stride past
the window.

The method could not have found a narrower allowance below a cell's tested width, because
each cell ran at one width. It also ran each cell once.

The column-band shapes are rows of `tests/pool_fp16_rocket` (CTest `pool_fp16_rocket`),
which also checks on the host that the bands reassemble the reference exactly. The board ran
the whole test with no job timeout in the kernel log.

### Cube dimensions at the field maximum

The width and height cases ran at C 16, with a 2-row input for the width and a 2-column input
for the height. The channel cases ran on a 4×4 plane. Each ran the windows above. The
height and channel cases pass at every dimension. The width cases with a 2×2 window are
past the output-width allowance at every dimension measured.

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
