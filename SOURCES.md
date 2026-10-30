# External sources

This list gives the external documents, source trees and publications on the RKNPU that these
notes cite, grouped by topic. Each entry opens with the source and where to find it, the
version or date it was read at, and what it covers. The findings below each opening use the
evidence tags that the [README](README.md) defines, and state where a source disagrees with
these notes.

## Regcmd and hardware sources

### Mesa `rocket` (Teflon) driver

Mesa's Teflon driver for `rocket` (`src/gallium/drivers/rocket/` in Mesa), read at `31e2daea`
on 2026-08-18. It is a working in-tree FOSS TFLite delegate that emits convolution regcmd for
the RK3588 NPU, and it is ground truth for the regcmd format. It is INT8-only, so it does not
cover the fp16 and int4 paths. Its files divide the format this way:

| File | Contents |
|---|---|
| `rkt_regcmd.c` | `fill_first_regcmd`, the validated CNA→CORE→DPU->DPU_RDMA sequence. Also the `add_tensor` eltwise geometry, the enable mask and the MRDMA-disable block |
| `registers.xml` | Every register address and field |
| `rkt_coefs.c` | Weight packing, the WEIGHT_ATOMIC_SIZE=32 reorder, the CACC 48-bit accumulator note |
| `rkt_ml.c` | Feature packing, FEATURE_ATOMIC_SIZE=16 |
| `rkt_task.c` | NVDLA-style tiling and split |

Two correctness fixes entered the driver on 2026-06-27, both found against MobileNetV2.

The first fix gives the fused conv+add requant constants a closed form, in place of tables of
magic floats captured from one model. Without it, any scale absent from the table falls back to
`add_scale = 0` and drops the residual silently. The form is
`EW scale = addition_scale / (input_scale * weights_scale)` and
`output scale = (input_scale * weights_scale) / output_scale`. With it, the output path is
identical to the non-add case, and the fused conv's `output_scale`/`output_zero_point` describe
the ADD output tensor. `librocketnpu` derives its requant and has no such tables. The fix is an
independent statement of the same algebra, to check `librocketnpu`'s derivation against.

The second fix concerns weight packing, which must pad input channels to `FEATURE_ATOMIC_SIZE`
(16), not to `MAX2(ch, 16)`. `fill_task()` tells the hardware to read
`align(MAX2(ch, 16), 16)` channels, while `rkt_fill_weights()` packed only `MAX2(ch, 16)`.
Any conv whose input channel count is not a multiple of 16 then had a misaligned weight stride
that corrupted every output channel (mean error 68 -> 0.5 once fixed). MobileNetV2's
24-channel blocks are the example. The padding channels contain the weight zero-point, so they
contribute zero.

Its output requant derivation drops a carry at one mantissa pattern. The copy at
`31e2daea` masks `(bits >> 9) & 0x7fff`, adds one and forces bit 14. Where mantissa bits
[22:9] are all set, the `+1` carries out. The DPU then applies half the scale at an even
exponent, and at an odd one `MUL` `0x8000`, which it reads as -32768. One scale in 16384 is
affected [HW sweep, both parts]. The measurement, and a derivation that keeps the carry, are
in [encodings/out-cvt-converter.md](encodings/out-cvt-converter.md).

### RK3588 TRM and datasheets

Rockchip's technical reference manual and datasheets for the RK3588. "Rockchip RK3588 TRM
V1.0-Part1-20220309" has the NPU register chapter: `RKNN_pc_*` `0x0xxx`, `RKNN_cna_*`
`0x1xxx`, CORE `0x3xxx`, DPU `0x4xxx`, DPU_RDMA `0x5xxx`. `pdftotext` extracts it.

These notes take most register facts from hardware sweeps and the Mesa driver, and tag a fact
the TRM states `[TRM]`. Where a TRM statement contradicts a sweep, the sweep stands. The RK3576 TRM (Part1/Part2
V1.2) has no NPU register chapter: it contains the memory map, the interrupt table and the CRU
clock tree ([chips/rk3576.md](chips/rk3576.md) §"Documentary sources"). The chip sheet cites it
for SoC integration, and [chips/rk3576-regcmd.md](chips/rk3576-regcmd.md) takes the register
encoding from vendor RKNN-Toolkit2 register programs instead.

### allbilly/npu

A higher-level op generator by allbilly (conv1d/2d, matmul, activations, LUTs) on the same Mesa
regcmd encoding, chiefly `include/rknnops.h`. It covers more operations than Mesa, and is the
reference for operations beyond matmul.

Its `float16_alu_op(ALU_ALGO_ADD)` encodings are the source of the fp16 DPU-EW
K-accumulation encoding. Its int-EW `EW_OP_TYPE` bit, moved alone, makes a working integer
ALU return garbage ([encodings/k-accumulation.md](encodings/k-accumulation.md)).

### allbilly/rk3588

The Python successor to allbilly/npu, by the same author, with the field triage read at
`e13d99f` on 2026-09-20. It covers decoded vendor convolution programs for the RK3588 and a field-by-field
triage of the RK3588 register set. Its register definitions are Mesa's
(`experimental/registers.xml`, `include/rkt_registers.h`). Neither allbilly repository has a
license file, so treat both as readable facts rather than as code or data to vendor.

#### Vendor convolution programs

`conv_expt/capture_harness/decoded/` has 83 vendor RK3588 conv register programs decoded to
named CNA/CORE/DPU fields: multi-task, ic 16-1280, planes to 150, oc 12-1024, pointwise and
depthwise. They are a vendor reference for the RK3588 conv geometry words, as
`tests/data/rk3576-vendor-capture/` is for the RK3576. The address registers hold one
session's IOVAs, so only the geometry is comparable.

The captures come from a gdb harness on the vendor BSP runtime
(`experimental/rknn/trace_librknnc_*.gdb`,
`conv_expt/capture_harness/rknn_prefix_capture.gdb`, which patches the rknpu submit
struct down to a one-task prefix). The harness needs a vendor-BSP board and captures only
what the vendor compiler chose to emit. It is therefore a weaker instrument than the offline
compile-and-decode used for the RK3576 captures. It is RK3588 only, and nothing in it
addresses the RK3576 encoding.

#### Register-field triage

`examples/expt/` has an exhaustive field-by-field triage of the RK3588 NPU register set, with
executable probes. `trm_register_matrix.md` classifies every non-reserved field
in Mesa's `registers.xml` (199 register records, 512 field occurrences) across PC, CNA,
CORE, DPU, DPU_RDMA, PPU, PPU_RDMA, DDMA, SDMA and GLOBAL. Its status vocabulary
separates local silicon results from captures, from other projects' reports, and from
TRM-only definitions. A QUARANTINED class holds the sequences after which they saw a crash.

`trm_register_findings.md` has the results. `elementwise_int.py`, `pooling.py` and
`conv_simple.py` are the probes, each with an offline `--validate` mode that checks the
decoded streams without opening the device. Their board is an Orange Pi RK3588 on mainline
rocket. Of the results below, only those with an [HW sweep] tag are measured for these notes.

#### Integer EW ALU

The per-element EW ALU has integer modes. They measure same-format integer MAX, MIN,
ADD, MINUS, ABS and NEG for INT8, INT16 and INT32, across widths 1/2/3 and partial and
complete C1WC2 surfaces. They document saturation and `INT32_MIN` behavior, and measure MUL
for INT8/INT16 (its INT32 second operand is signed 16-bit). Their streams set the precision as
a matched set across DPU `0x4010` and `RDMA_FEATURE_MODE_CFG`, with the EW converter bypassed.

These notes measure the same ALU exact under that whole set [HW sweep]. The bundle and the
per-op table are in [encodings/sdp-stage-precision.md](encodings/sdp-stage-precision.md) §"The
EW stage in integer mode". ABS, NEG and the partial-surface cases are measured only by them.

#### PPU_RDMA input precision

`PPU_RDMA_DATA_FORMAT.IN_PRECISION` is a storage width rather than a datatype. The TRM
enumerates `0x7030[1:0]` as 4/8/16/32-bit, so a value of 0 asks for 4-bit input, and an
int8 pool programmed that way returns garbage. They pass signed INT8/INT16 maximum and
minimum, including adversarial and all-negative vectors, and integer average with Q16
`round(65536/k)` reciprocals in place of the FP16-encoded ones.

These notes reproduce native int8 pooling on that reading [HW sweep, Turing RK1,
2026-09-20]. The program, and the `IN_PRECISION` values 2 and 3 that their result does not
include, are in [encodings/ppu-pooling.md](encodings/ppu-pooling.md) §"The precision fields and
the storage-width trap".

#### Other PPU and DPU results

Their other capability results are not measured for these notes. On the PPU:

- `USE_CNT` is not an exclude-padding switch. All eight encodings return the include-pad
  answer, so `count_include_pad=False` has no register path in that recipe.
- `INDEX_EN` writes only six bits, `(row & 7) << 3 | (column & 7)`. It is a local pool-window
  coordinate, not a general ArgMax.
- Padding with stride adds a non-local vertical phase.
- PPU INT32/FP32 processing precision submits and writes zero on the external-RDMA path.
- Kernel 16 and padding 7 each pass alone and fail combined.
- Multi-pass 8×8 -> 4×4 -> 2×2 -> 1×1 average, maximum and minimum pass through NPU-written
  intermediates with no host repair, separately fenced. This corroborates
  [encodings/ppu-reduce-mean.md](encodings/ppu-reduce-mean.md).
- Explicit PPU_RDMA line and surface strides work, while a nonzero `NOTCH_ADDR` corrupts. The
  notch is not an additive line pitch there.

On the DPU:

- Scalar operands through `EW_OP_SRC=0` plus `EW_OP_VALUE_n` with `ERDMA_DISABLE=1` remove the
  scratch buffer and its read entirely.
- BS and BN each fuse a configured ALU and MUL, `RELUX_EN` and `MUL_PRELU`, up to five scalar
  operations in one task.
- Main + BRDMA + NRDMA + ERDMA gives an exact four-input sum in one task.
- FP16 comparison results write exact INT16 0/1, and INT16 EW results write sign-preserving
  external INT32.

#### State between tasks and submits

Three of their failures involve state left between tasks or submits. A two-task CNA spatial
split with `WEIGHT_REUSE` produced exact INT8 output, but the first following known-good DPU
add returned a stale `48576`. Only a second add recovered, and they therefore never submitted
`DATA_REUSE` speculatively. Their `EW_CVT`/`OUT_CVT` probes are QUARANTINED after a reported
crash with no retained kernel fault record.

> [!CAUTION]
> In their sweep, LUT reuse across separate rocket submissions fails. A second task without
> table writes produced unrelated values for clear, rearm and untouched pointer states.
> Software must reload all 1,026 entries per standalone submit. Reuse within one submit is not
> characterized.

#### EW_CVT rounding

Their one rounding result is on a different converter stage from the requant. A controlled
`OUT_CVT` FP16-to-INT16 probe behaved as expected. A following INT16 `EW_CVT` probe returned
`[2, 5, 8, 11, 14, 17, 20, 23]` where nearest-even predicts
`[2, 6, 8, 12, 14, 18, 20, 24]`, consistent with half-way values rounding downward. The
result does not contradict the requant rule: the RK3588's `OUT_CVT` rounds ties to even at
`0x4088` [HW sweep] ([encodings/out-cvt-converter.md](encodings/out-cvt-converter.md)).
`EW_CVT` is a separate stage, and its tie rule on the RK3588 is not characterized. Their
result comes from the sequence that crashed, so it is not established as that stage's rule.

#### Their reading of these notes

They cross-check against these notes as read at `e0c7213`. They report that the notes find no
native deconvolution mode, which is out of date:
[encodings/conv-transpose.md](encodings/conv-transpose.md) records `CNA_CONV_CON1[16]`
`DECONV` live on the RK3588. They also note, correctly, that the notes repository does not include
the `tests/` and `src/` trees that hold the tests these notes cite.

### RKNN-Toolkit2

The vendor's proprietary compile-and-run stack (`github.com/airockchip/rknn-toolkit2`).
`librocketnpu` drives the same NPU through mainline `rocket` instead. RKNN3 (targeting
RK1820 / RK3572) is a different NPU generation, and these notes do not cover it.

Its offline compiler output is the reverse-engineering input for the PPU pooling family, which
the FOSS Mesa/Teflon path never emits. A 1-op ONNX pool compiled to a `.rknn` and decoded for
the PPU / PPU_RDMA register page gives the exact `RECIP_KERNEL = fp16(65536/k)` reciprocal
format. The method is in [ppu-rknn-capture/](ppu-rknn-capture/), and no vendor artifacts are
redistributed.

The same offline output is the source of the RK3576 register encoding. The RK3576 TRM has no
NPU register chapter, so the formulas in [chips/rk3576-regcmd.md](chips/rk3576-regcmd.md) are
transcribed from RKNN-Toolkit2 register programs compiled for `rk3576` at known geometries.

### RKNN-Toolkit2 SDK documentation

The vendor's documentation for RKNN-Toolkit2 (`github.com/airockchip/rknn-toolkit2/doc`,
V2.3.2), not for RKNN3. It is an external cross-reference for the vendor's own statements,
rather than reverse-engineering input.

User Guide §3.5.4's high-performance layout table covers the tile-layout matrix for the A, B and C
operands. It also states that A and B must share a datatype (cf.
[encodings/tile-layouts.md](encodings/tile-layouts.md)). Its §6 covers the quant path
(INT8-only, per-channel weights / per-tensor activations, range solvers, hybrid FP16
fallback, cf. [datatypes.md](datatypes.md)), and §5.3.3 the multi-core split op list and
the IRQ-affinity tip (cf. [perf/iova-and-multicore.md](perf/iova-and-multicore.md)).

The runtime header `rknn_api.h` has the perf/mem query structs, and `rknn_mem_size` reports
allocations. Per-frame bytes are an analytical string in `rknn_perf_detail`, not a hardware
counter. [perf/hw-byte-counters.md](perf/hw-byte-counters.md) covers the hardware counters
instead. The `OP_Support` / Compiler-Operator-List docs are the vendor's operator-coverage
reference.

### RKNN-Toolkit2 issue #163

A YOLOv8s compile for `rk3588` under toolkit 2.1.0 and 2.2.0
(`github.com/airockchip/rknn-toolkit2/issues/163`). Its log prints the compiler's `REGTASK`
field-width check in full:

```
REGTASK: The bit width of field value exceeds the limit, target: v2, offset: 0x500c, shift = 0, limit: 0x1fff, value: 0x20cf
```

Each line names a register offset, the field's low bit and its maximum. Each error the check
prints therefore gives one field's position and width, as Rockchip's compiler holds them. The
log's three fields match Mesa's `registers.xml` [source-confirmed]: DPU_RDMA `0x500C` `WIDTH`
[12:0], and DPU `0x4038` `NOTCH_ADDR_0` [12:0] and `NOTCH_ADDR_1` [28:16]. `v2` is the
target the runtime reports as "RKNPU v2" for this model.

The layer is the DFL head's 1×1 convolution on a cube 4 high, 8400 wide and 16 channels deep.
A 640×640 YOLOv8 or YOLO11 head has 8400 anchors (80² + 40² + 20²), over the 13-bit
maximum of 8191. The width field holds `0x20cf`, 8400 - 1, and each notch field `0x419f`,
2×8400 - 1.

The same log shows two more compiler rules. When height × width exceeds 16384, a Transpose
falls back to the CPU. The three-core split refuses this convolution and runs it on one core.

> [!CAUTION]
> The compile does not stop at a `REGTASK` error. The exported model returned an all-zero
> output on the device under both toolkit versions.

Rockchip's own YOLO export (`airockchip/ultralytics_yolo11`, `RKOPT_README.md`) moves the
DFL and the post-process out of the model and onto the CPU. It gives two reasons: the
post-process quantizes poorly, and the DFL is slow on the NPU. Its outputs stay per scale
(`[1,64,80,80]` and the like), so no 8400-anchor tensor reaches the NPU either. The 13-bit
limit itself is documented in [encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md).

### `rknpu-reverse-engineering`

An early-stage, ASR/TTS-focused reverse-engineering repository (phhusson / Tomeu lineage) on
the BSP `rknpu`/`/dev/dri/card1` path, not rocket. Mesa `registers.xml` plus the Teflon decode
supersede its register encodings: they are more complete, and target the rocket path.
`rknpu-ioctl.h` is the BSP uAPI, which `librocketnpu` does not use. Three of its artifacts are
still useful:

| Artifact | Content | Use |
|---|---|---|
| `hello2.c` | The in-core IRQ/block-completion bitmap (CNA/CORE/DPU/PPU/DMA-err, two reg-banks per block = `RKNPU_JOB_PINGPONG`) | Recorded in [perf/iova-and-multicore.md](perf/iova-and-multicore.md). It confirms the PPU is a real separately-completing block |
| `instrs.h` | A hand-assembled plain conv (`0x0201`=CNA `0x10xx`, `0x0801`=CORE `0x30xx`, `0x1001`=DPU `0x40xx`, for example `DPU_EW_CFG 0x4070=0x383` plain-conv bypass) | Confirms the block and register format these notes use. It adds no new information |
| `analysis` + `mess/dump-*` | The raw hex dumps of the 6 gem BOs from real models (weights / gem2 64-bit regcmd stream / gem3 10-word task list with "jump-to-next-task" / working / input / output) | A reference for the multi-task chaining structure (task persistence, and the fixed per-dispatch overhead) |

For the rocket path, a targeted Teflon capture gives more than decoding these raw BSP dumps.
Its dumps show no integer K-accumulation, and that absence does not establish a float-only
EW ALU (see [encodings/k-accumulation.md](encodings/k-accumulation.md)).

### Rockchip hardware design guides

Rockchip's board-design guides, `Rockchip_RK3588_Hardware Design Guide_V1.4_EN.pdf` and
`3576_hardware_design_guide.pdf` (V1.1, 2024-05). They cover the platform limits that each
SoC's NPU figures depend on. They have no register content, and do not cover the regcmd
interface or the NPU's internals. `pdftotext -layout` extracts both cleanly.

The RK3576 differs from the RK3588 in each of these limits:

- The DRAM bus width: 32-bit / 2 channels vs 64-bit / 4 channels, at an identical 2112 MHz
  PHY clock, so exactly half the bandwidth.
- The NPU power rails: the RK3588 has a separate `VDD_NPU_MEM`, and the RK3576 has none. The
  RK3576's CBUF and other NPU arrays are on the logic rail.
- The peak operating point: 0.800 V / 4 A / 3.20 W vs 0.850 V / 4 A / 3.40 W, both at
  1000 MHz.
- The package thermal resistance: θJA 15.84 vs 8.7 C/W.

The DRAM bus width is the mechanism behind the RK3576's DDR-traffic-driven atom drop and its
long DPU write drain. It is also the mechanism behind the limit on its host cube packing. Both
guides are indexed in [chips/rk3588.md](chips/rk3588.md) and [chips/rk3576.md](chips/rk3576.md).

### 6.6 BSP kernel `rknpu` driver

The vendor kernel driver in Rockchip's 6.6 BSP kernel, with the HW performance counters and
the devfreq/OPP table. The rockchip-linux `develop-6.12` branch has the same driver, 0.9.8
(read at `470f9dccbdc4`, 2026-06-29). The two copies differ only in kernel-API porting, so the 6.6
copy is current for hardware behavior.

`rknpu_devfreq.c` shows that 200 MHz is the literal `POWER_DOWN_FREQ`. It also shows that the
vendor sets the NPU clock only while the power domain is active (`!pm_runtime_active` ->
refuse). The two IOVA allocators that [perf/iova-and-multicore.md](perf/iova-and-multicore.md)
describes are the same in both copies.

> [!CAUTION]
> Its non-blocking submit path has a unit error. `rknpu_job_timeout_clean()` compares a
> running job's age in microseconds against a timeout in milliseconds. A non-blocking submit
> therefore soft-resets a job that has run past `timeout` microseconds, 1000x early. The
> blocking path multiplies by 1000 and is correct.

What the driver says about the RK3576 is in [chips/rk3576.md](chips/rk3576.md): the soft
reset's CBUF resets, the per-task completion mask, and the SRAM read margin.

### RK3588 BL31 / ATF binary

The secure firmware that controls the NPU clock (`rk3588_bl31_v1.51.elf`, from rockchip
`rkbin`), read with strings plus `radare2`/`aarch64` `objdump`. It covers the NPU clock's
SCMI interface and its PVTPLL.

The NPU is SCMI clock id 6, set in EL3 through `rockchip_opteed_clk_set_rate`. A PVTPLL clocks
it, with a min and max from per-chip OTP (`adjust npu pvtpll by otp: min=.. max=..`). The blob's
clock descriptor has no static rate table, where upstream TF-A's source for the same clock has
one ([perf/clock.md](perf/clock.md)). The same reading finds no voltage coupling in the
firmware. The binary explains why setting the rate cold hangs EL3, and why the maximum rate is an OTP
value ([perf/clock.md](perf/clock.md)).

### Mainline `rocket` driver source and RK1 serial boot log

The upstream `rocket` driver, `drivers/accel/rocket/` (android-mainline, and a v7.1 build),
read from `torvalds/linux` on 2026-09-20 with mainline past v7.2, and a Turing RK1 serial boot
log. Stock upstream has no NPU clock handling (`clk_bulk_*` only), so the `clk_set_rate` ramp
is the out-of-tree `rocket-clk` patch, not upstream. The serial boot log confirms the SCMI
handshake (`SCMI Protocol v2.0 'rockchip:'`) and the `quirk_clock_rates_triplet_out_of_spec`
rockchip clock quirk.

#### Upstream fixes after v7.1

Upstream has moved past v7.1 and has merged fixes that overlap the out-of-tree patch series
(the `patches` repository). Three of them are in `rocket_job.c`.

`70e6a33d` (2026-07-01, Shuvam Pandey), "accel/rocket: initialize job domain before
cleanup paths", fixes a NULL domain in cleanup. `rocket_ioctl_submit_job()` assigns
`job->domain` only after task copying and BO lookups, while `rocket_job_cleanup()` puts it
unconditionally. A failure before that assignment then cleans up a job with a NULL domain. The
commit takes the per-file reference before the first error path, the same bug and the same fix
approach as `patches/rk3576/npu/0013`. It also clears `rjob->tasks` after freeing it in
`rocket_copy_tasks()`, so the common cleanup path cannot double-free the task array. One
upstream commit therefore covers two out-of-tree patches.

`a85402bf` (2026-05-24, Muhammad Bilal) fixes a NULL dereference and an integer overflow in
`rocket_job_push()`'s `kvmalloc_array()` of the combined in/out BO array. It overlaps the
NULL guard in the `patches/rocket/085` uAPI work.

`9b2dedad` (2026-06-10, ZhaoJinming) fixes error-path handling in `rocket_job_run()`: a
`dma_fence` reference leak, and an unsignaled fence returned to the scheduler. It also fixes
`pm_runtime_get_sync()` leaking its reference on failure, which keeps the NPU from suspending.
That leak is the mechanism behind the half-started-job runtime-PM pin, which these notes record
as a mainline bug that affects the RK3588 too.

At least three out-of-tree patches now have upstream equivalents, so a rebase onto a current
mainline will conflict or silently duplicate. To test a patch's application, fetch the base for
its kernel version from `raw.githubusercontent.com/torvalds/linux/<tag>/drivers/accel/rocket/`.

#### DVFS series

An upstream DVFS series for the RK3588 NPU is in review. It is Igor Paunovic's "accel/rocket:
DVFS for the RK3588 NPU", v2, patchwork series 1171089, 2026-09-22, 11 patches on drm-misc-next,
read 2026-09-23. Patch 4 keeps core slots stable across unbind and rebind. Patches 6-11 are the
DVFS core, and add:

- A binding, and the `rk3588-opp.dtsi` table on all three cores with `opp-shared`
- A boot-rate restore with a `.shutdown` hook
- devfreq `simple_ondemand` at a 50 ms window
- A cooling device and an 85 C passive trip

The author conceded five review findings on patch 9, and a v3 waits on review.

The series sets the clock in the runtime-PM callbacks, so it replaces the out-of-tree clock
and voltage patches rather than combining with them. Adopted as posted, it changes the
operating point in two ways. The rail at the 600 MHz request drops from 800 mV to the table's
700 mV, where the clock measures 638 MHz against ~816 MHz [HW sweep 2026-10-19]. The rate
becomes governor-driven, with a window shorter than the host gaps inside one Whisper encode
[expected]. Its
statement about powering an island above the boot rate is in [perf/clock.md](perf/clock.md).

### drivercraft/rk3588-clk

A Rust `no_std` RK3588 CRU clock library (MIT, for bare-metal / U-Boot) that sets the NPU clock
by direct CRU register writes (`npu_set_clk` / `npu_get_clk` + `ACLK/HCLK/PCLK_NPU0..2` gates,
`pll.rs` / `clksel.rs` / `gate.rs` / `constant.rs`). It covers the CRU NPU clock tree at
register level: the PLL configuration, the clksel mux and the gate bits.

It cannot replace the `rocket-clk` patch at runtime: on mainline Linux, BL31 controls the NPU
clock (SCMI id 6 plus a per-OTP PVTPLL, see the BL31 entry). This library has no PVTPLL, no
voltage handling, and no SCMI-conflict guard. Its direct register writes would conflict with
EL3 and skip the f/V coupling the patch depends on. It is an MIT-licensed register-level
cross-reference for annotating or extending the clock patch, not a mechanism to adopt
([perf/clock.md](perf/clock.md)).

### LKML: "[RFC PATCH v4 0/9] accel: rocket: Add RK3568 NPU support"

An RFC series by Midgy BALON (2026-06-13, v2 at
[lkml.iu.edu/2605.3/10672.html](https://lkml.iu.edu/2605.3/10672.html), base v7.1-rc6). It
adds the RK3568 to the upstream `rocket` driver through a per-SoC `rocket_soc_data`, which
derives the DMA width and core count from match data. As an RFC it asks for design feedback,
and is not for merge.

The RK3568 NPU is a single NVDLA-derived core (0.8 TOPS), and its register layout matches the
RK3588's. The match corroborates that the NVDLA-derived IP is common across Rockchip SoCs, and
`librocketnpu` is expected to drive most of it too [expected]. End to end, the series is blocked
on the Mesa/Teflon userspace, which still emits RK3588-tuned configuration. It is also blocked
on a hardware issue in the MAC/output stage (below). Beyond Teflon's conv+add, `librocketnpu`
has the full-datatype matmul, general/DW/int8 conv, LUT activation and an on-NPU EW multiply.

It gives the RK3568's NPU AXI master and IOMMU as 32-bit and the RK3588's as 40-bit. The
RK3588's 4 GB per-fd limit is therefore the regcmd address field's, not the bus's
([perf/iova-and-multicore.md](perf/iova-and-multicore.md) §"Per-fd IOVA windows"). The
RK3568's 32-bit DTE needs `GFP_DMA32` page tables, relying on Simon Xue's per-device-ops
series for the `rockchip,iommu` ops.

Its patch 5 keeps the IOMMU domain attached across same-context jobs, where stock `rocket`
attaches and detaches it on every job. What that costs on the RK3588, and the out-of-tree
patch that does the same there, are in
[perf/iova-and-multicore.md](perf/iova-and-multicore.md) §"Per-job IOMMU cost".

The author reads the NPU's DMA byte counters ("the NPU reads the full input and weight tensors
per its DMA counters"). The counters exist and are readable on the RK3568, where reading the
`0x2xxx` page hard-locks the RK3588. The RK3568 reads them through the vendor's
`rknpu_old_top_amount` offsets (`0x8010`-`0x803c`), and the RK3588's vendor configuration disables
the amount counters altogether ([hw-byte-counters.md](perf/hw-byte-counters.md) §1).

The MAC/output stage never completes on the RK3568, even on a byte-exact replay of the
vendor command list. The v2 cover letter attributes it to a Mesa-side regcmd issue, or to a part
of the bring-up not yet found. A byte-exact replay that fails the same way points instead to
bring-up: PVTPLL, power or NoC de-idle [hypothesis].

The author asks for pointers on it, and [clock.md](perf/clock.md) covers the
RK3588's BL31, PVTPLL and clock. Patch 3 starts the PVTPLL compute clock through SCMI, which
corroborates the PVTPLL finding in these notes. Patch 9 wires vdd_npu as the power-domain
`domain-supply` (`need_regulator`), so genpd manages the rail. Patch 9 is the upstream-idiomatic
alternative to the driver-held-regulator f/V coupling of the out-of-tree patches
([clock.md](perf/clock.md)).

The v2 cover letter gives the per-sub-unit `OPERATION_ENABLE` offset as `0x_00c` on the
RK3568, where the RK3588's map has it at `0x_008`. Its evidence is BSP captures of two conv
shapes that write nothing at `0x_008`, and multi-bit values at `0x_00c` (CMAC `0x1`, ACCU `0x0`,
DPU `0x108`, DPU_RDMA `0x13f`). The v4 cover letter does not restate the claim.

The evidence does not separate the two SoCs, because no program these notes use writes the
per-block registers. Mesa at `31e2daea` and `librocketnpu` start a task's blocks from the PC's
bitmap at `0x0008` ([encodings/regcmd-task-model.md](encodings/regcmd-task-model.md)
§"`PC_OPERATION_ENABLE` as a block bitmap"). The vendor programs write none either: the
RK3576's convolution program writes nothing at `0x_008`, and the RK3588's pooling program
nothing at `0x6008`/`0x7008` ([encodings/ppu-pooling.md](encodings/ppu-pooling.md)).

On the RK3588's map `0x500C` is DPU_RDMA's cube width. Its `0x13f` is 320 minus 1, the output
width of the YOLOv5s first layer, a 6×6 stride-2 convolution over a 640-wide input
[host-computed]. The `0x_00c` values are probably geometry and mode words rather than enables
[hypothesis].

### NetVar1337/linux-rk3576-rocket

"[PATCH RFC 0/4] accel/rocket: add support for the RK3576" (VoidChecksum / Markus Kvam,
2026-06-11, against `torvalds/master`): an independent mainline-targeted implementation of
gahingwoo's bring-up (the next entry), which it credits throughout. It has the binding, a
clocks-by-name fix, per-SoC match data with PC_DONE polling, and the `rk3576.dtsi` core
nodes. It is compile-tested only: the author has no RK3576 hardware, and asks for testing
reports. It covers what `patches/rk3576/npu/0001`, `0006` and `0007` do in the out-of-tree
series, and nothing else.

Patch 3 states that the `PC_DONE` bits "are read-only in `INTERRUPT_MASK`, so completion
cannot be routed to the GIC". It builds a 1 ms hrtimer poll on that premise. The read-only
result concerns `PC_DONE` only. The DPU pair, `DPU_0`/`DPU_1` (bits 8-9), masks normally, and
its interrupt reaches the GIC [HW sweep, see [chips/rk3576.md](chips/rk3576.md)].

Polling adds latency. Retiring on the DPU bit at a 50 us period, in place of `PC_DONE` at
500 us, reduces the minimum submit latency from 1065 to 439 us. Their 1 ms period is slower
again. Two task classes raise no DPU completion (pooling, and the narrow and wide int32
writers), so a poll or a grace period must remain as the fallback for those.

Its patch 2 is the same fix as `patches/rocket/089` (`clk_bulk_data.id` never set, so
all four entries resolve to the node's first clock). It was found independently, with a
thinner rationale, and neither posting has been merged.

Its device tree places core 1 correctly: `0x27708000`, with the IOMMU at `0x2770a000`. It is
cross-checked against the vendor BSP DT, which matches the live silicon. The `0x27710000`
placement that these notes record as wrong is in the gahingwoo series, not this one. Two RK3576
RFCs exist, so a reference to "the RFC" must say which.

Its stated open items (the NPU power-domain chain status never asserting after power-off,
whether the RKNN BIU resets belong to the power domain, and the boot firmware's orphaned
IOMMU page fault) are the ones `patches/rk3576/npu/0002`-`0005` already address.

### gahingwoo, "Mainlining the RK3576 NPU"

A blog (`blog.gahingwoo.com/posts/rk3576-npu-mainline/`, moved from
`gahingwoo.github.io/posts/rk3576-npu-mainline/`) and repository
(`github.com/gahingwoo/linux-rk3576-npu`: `notes/provenance.md`, `notes/rk3576-npu-values.md`,
`extract/extract-npu-values.sh`, read at `82cdbe9`, 2026-09-24). It records a mainline-`rocket`
bring-up of the RK3576 (2-core, 16 CBUF banks vs the RK3588's 12), with its kernel series and
board logs.

#### Repository contents

Its `board-logs/` is the current record, and the `charsiu` notebook is not: numbered rounds
run through r421 at 2026-09-24, where `charsiu`'s `docs/lab-notebook.md` stops 2026-09-12.
Read the board logs for anything after that date. Its README has the current state only, and
`HISTORY.md` the history.

The repository reaches well past the register maps:

- A 28-patch kernel series
- A Mesa fork with an RK3576 Teflon conv2d
- A `replay/` capture-and-replay harness that runs the same regcmd through both `rknpu` and
  `rocket`
- `FINDINGS.md`, a 2900-line chronological RE log that keeps its own reversed verdicts

Its main register documents:

- The CNA and CORE/DPU maps
- `predict.py`, a closed form
- `vendor_regcmd_full.txt`, a complete 139-entry vendor register program (CNA + CORE + DPU +
  RDMA) that an RK3576 regcmd encoder can be diffed against off-device
- `FINDINGS-FLOATSURFACE.md`
- `MATMUL-PIPELINE-ANALYSIS.md`, which confirms RK3576 matmul is the same CNA->CSC->CMAC
  pipeline in FC mode (no separate GEMM unit)

`MATMUL-PIPELINE-ANALYSIS.md` also reports a per-power-session "cold-start consume-arm"
stopping point, which it attributes to the hardware. Reproduced on `librocketnpu`'s regcmd encoder and
submit path, the stopping point is in the driver
([chips/rk3576-regcmd.md](chips/rk3576-regcmd.md)).

#### Capture method and platform script

Its capture method builds a 1-conv ONNX and converts it with `rknn-toolkit2`. It then walks the
`.rknn` for the 64-bit command words and decodes them per unit. The method is an alternative
to the Mesa-Teflon capture that can expose ops Teflon never emits, for example pooling.

`extract-npu-values.sh` derives the platform constants (power-domain / clock / reset IDs, GRF
base, PVTPLL, OPP table, per-core MMIO bases, IRQs, QoS) by grepping the kernel DT-bindings +
TF-A BL31. It adapts to the RK3588 with s/rk3576/rk3588/.

#### Agreement with these notes

Five of its findings agree with findings in these notes, each reached independently:

| Their finding | What it confirms here |
|---|---|
| IOMMU attach-once / detach-on-power-down, not per-job | The keep-attached patch ([iova-and-multicore.md](perf/iova-and-multicore.md)) |
| Ping-pong producer/consumer register groups (`S_POINTER`): the executer reads the consumer group, and misalignment -> stale geometry -> zero/garbage output. The groups need per-job re-init | They are the mechanism that decides where a delta regcmd task lands ([regcmd-task-model.md](encodings/regcmd-task-model.md)) |
| Requant is a right-shift whose magnitude decides the output (vendor 26-bit vs a wrong 14-bit -> saturation to black/white) | The per-scale QNNPACK shift in the conv int8-out path ([out-cvt-converter.md](encodings/out-cvt-converter.md)) |
| Per-channel zero-point correction in a weight-buffer tail (8-OC groups, 64 B = 8×32-bit + 8×16-bit + 8×16-bit, the 16-bit holding `128 − weight_zp`, term `(128−wt_zp)·input_sum`) | The term CPEND adds on chip ([depthwise-conv.md](depthwise-conv.md) §"`DPU_BS_OW_OP` is the CPEND operand"), and the term a host box-sum adds where a program cannot reach CPEND |
| The `dt_wr`/`dt_rd`/`wt_rd` byte counters are readable on RK3576 | The [hw-byte-counters.md](perf/hw-byte-counters.md) table predicts exactly that (rk3576 config wires `0x2234/38/3c`, while rk3588 nulls them and that page hard-locks). It leaves the RK3588 result unchanged, and confirms the asymmetry between the two SoCs |

Its register map does not transfer to the RK3588 as it stands. The RK3576 map is shifted and
re-packed, and its clock and power tree differ.

#### Two jobs in flight (board log r419)

Board log r419 compares four ways to put two jobs on the hardware:

- A: one job, one ioctl
- B: two jobs, one ioctl, one fd
- C: two jobs, two ioctls, one fd
- D: two jobs, two ioctls, two fds

| k, n | A (us) | B (us) | C (us) | D (us) |
|---|---|---|---|---|
| 1024, 256 | 49.77 | 95.05 | 94.74 | 52.43 |
| 1024, 1024 | 133.64 | 263.05 | 263.30 | 135.66 |
| 2048, 2048 | 489.63 | 943.00 | 937.84 | 471.88 |

**Conditions:** ROCK 4D, NPU 594 MHz, board log r419 (2026-09-15). All four configurations run
back to back in one process, at 200 repetitions each.

One fd serializes two jobs whatever the ioctl count. Two fds run them side by side, and the
pair costs what one job costs, at every shape where the configurations separate. On their
board the second core runs a second job at no added cost, and they report no corruption from
it. At k=64 n=32, where the arithmetic is negligible, a first reading said the exact opposite
(B beating D by 2.5x) and was noise.

Their 594 MHz is the operating point where the undervolt table in
[chips/rk3576.md](chips/rk3576.md) says two-core corruption vanishes. The two-core runs in
these notes were at 786 MHz. Two claims rest on those runs: that a second core gives no
throughput, and that two jobs in flight produce wrong output. Board log r419 is a second,
independent reason not to treat either as a property of the silicon.

**Not characterized:** two jobs in flight at 594 MHz on the boards these notes use.

#### Cold-start stopping point and task chaining

`CHAINED-CMAC-STOPPING-POINT.md` is the falsification-ledger writeup of the same
per-power-session stopping point, parked 2026-07-10. It concludes that the consume-arm is
internal cold-start sequencer state reachable only from vendor RTL.

They independently found and fixed the 16-bit `pc_task_number_bits` (`WRITEL-AUDIT.md`, and
patch 0028 writes `(0x7 << 16) | task_count`), as the out-of-tree series does. Both stacks run
n-task jobs in one hardware kick: the out-of-tree series through `DRM_ROCKET_JOB_BATCHED`
(`patches/rk3576/npu/0015`-`0016`), and theirs in `charsiu` at 32 chained tasks per job.
`FINDINGS.md`'s "even task 0 computes nothing when task_number=29" describes that log's own
submit path, not a hardware bound.

Every experiment in the parked ledger varies the job that comes out empty, never the job
before it. The ledger therefore cannot detect the next-submit corruption a wide output causes,
which is a property of the preceding submit. `charsiu`'s harness does not share the limitation:
its bisects judge a following job run in a separate process (see its entry).

#### Upstream series

Their upstream series is at v14, posted 2026-09-24 with 15 patches against next-20260914
([v14](https://lore.kernel.org/all/20260924102135.92217-1-gahing@gahingwoo.com/)). At v13
(2026-09-15) it had an Acked-by from Conor Dooley on both dt-bindings and a Reviewed-by
from Abel Vesa on both pmdomain patches. It also had a Tested-by from Igor Paunovic on
each of the three reset-race patches
([v13](https://lore.kernel.org/all/20260915104328.45901-1-gahing@gahingwoo.com/)). The v13
posting partly exists to put a retraction into its commit messages. Before reading an older
posting's case for the reset-race patches, see the withdrawn induced-reset claim in the
`charsiu` entry.

Ulf Hansson has offered to take v14's three pmdomain patches (7, 9 and 10) through his tree.
The series has review tags from Krzysztof Kozlowski, Conor Dooley, Heiko Stuebner and
Abel Vesa. Patch 14/15 assigns the NPU 594 MHz at the SoC level, citing Rockchip's table as
asking 725 mV of the 500 and 600 MHz steps. The 725 mV is the table's default row: bins
`L0`-`L1` ask 737.5 mV, which still leaves 594 MHz at 750 mV inside it at every bin. The v12
and v13 postings give the two-core fault as "13 to 20 wrong rows", which the v14 posting
withdraws as unsourced. The board log's figure is 11 to 25 wrong words a pass.

Board log r420 places the power-on settle requirement in the domain rather than the rail. The
measurement is recorded in [chips/rk3576.md](chips/rk3576.md), under the power-on settle, and
its run count in [chips/rk3576-validation.md](chips/rk3576-validation.md) §"Power-on settle".

Board log r421 boots the v14 series as sent, and both cores probe. Every numerics check matches
the previous kernel to the last digit, perplexity included. Decode measures 2.8% lower from one
unpinned reading per kernel, which the log declines to call a regression.

The reading that follows was done against v7, posted 2026-08-12 with 10 patches as
`accel/rocket: RK3576 NPU (RKNN) enablement`
([cover](https://patchwork.kernel.org/project/linux-rockchip/cover/20260812094106.1391698-1-gahing@gahingwoo.com/)).
The v7 posting retracts the "the completion interrupt never reaches the GIC" premise of v3-v6.
It attributes the whole of that premise to the `PC_TASK_CON` width. Polling, the hrtimer and
every alternative completion path are therefore gone, and the RK3576 retires on the DPU
interrupt like the RK3588. The open question this reopens is in
[chips/rk3576-regcmd.md](chips/rk3576-regcmd.md).

Two of its patches match out-of-tree patches. Patch `01/10` is `patches/rocket/090` hunk for
hunk. Patch `08/10`'s `PC_TASK_CON` word is `0x70001`, the same value `rk3576/npu/0008` writes
by shifting the triple. Their evidence is one convolution submitted three times, and
their Teflon userspace is conv2d-only. Neither the pooling programs nor the wide-output
writers, which raise no DPU completion at all, are reachable from it.

### gahingwoo `charsiu`

An open LLM runtime for the RK3576 on mainline `rocket` (`github.com/gahingwoo/charsiu`,
GPL-2.0-or-later, first commit 2026-08-14). It takes the same design as `rocket-userspace` +
`ggml-rocket` on the RK3576, and names both, with these notes, as its starting point. It covers
the vendor's dispatch plan read from `.rkllm` files, a cost model of its NPU path, and the
RK3576 output-stage and CBUF registers. It does not cover rotation, the DPU LUT or the PPU.

**Conditions:** Every number in this entry is theirs unless the text says it was taken for
these notes. They measure on a ROCK 4D, on their own kernel series. Their records run from
2026-08-14 (their v7-lineage kernel) to 2026-09-14, and a passage that rests on one dated record
gives its date or round. None is reproduced on the RK3576 board these notes use.

#### Repository state

At `dev` 524e10a (read 2026-09-18) the branches are `dev` and `stable`. The `main` branch is
stale, and its README still prints withdrawn figures. The notebook, `docs/lab-notebook.md`,
stops 2026-09-12, and rounds r393 onward are in `board-logs/` in the driver repository, not
this one. For a question the notebook does not answer, read the board logs. Their multimodal
side runs SmolVLM, CLIP and Whisper on the same matmul primitive.

#### Vendor dispatch plan

A vendor `.rkllm` contains the register-command streams the closed stack submits. Their
reader, `tools/rkllm_regcmd.py`, reads the whole dispatch plan out of one offline, with no
board and no vendor runtime.

Its first reading (`docs/vendor-dispatch.md`, Llama-3.2-1B-Instruct-w4a16) finds 13,224
streams over 1,061 distinct shapes, 8,808 convolutions + 4,416 DPU-only. Precision goes by
role. Int4 projections are batched at M = 16 to 80 (2,816 of 3,328, 85%), and fp16
attention runs at M=32-48 against 128 precompiled KV-length buckets (one per 32 tokens of
context). The LM head is int8, as forty 2048×8160 pieces. Every projection is split across
the two cores by output channel.

The FFN down-projection is split on both axes at per-piece K=4096, inside the 4608 slice
bound measured for these notes. The split is the vendor's own way around a K that does not fit
one slice (the shape class `librocketnpu`'s matmul entry refuses at K>4608). Every int4
dispatch is K=2048 (2,688, 81%) or K=4096 (640, 19%): the vendor never hands one dispatch a K
of 1024.

#### Withdrawn reader figures

> [!NOTE]
> Its author withdraws two figures from older versions of the reader. Do not quote "21,532
> streams / 12,724 DPU-only" or "int4 projections all at M=1 (3,752 dispatches)".

Both were defects in `tools/rkllm_regcmd.py`, found and fixed by its author (`c179704`
2026-08-28, `80bd4d2` 2026-09-01). The same defects affect the reads of two more models that
were run for these notes with the unfixed reader (below).

The M histogram read the wrong register. It took M from `0x102c`, the row count,
where the correct register is `0x1034`, the pixel count. An int4 projection is emitted as a
one-row image M pixels wide, so its row count is 1 whatever M is. By contrast, fp16
attention is emitted as an M-row image whose rows equal its pixels in 4,940 of 4,940
streams. The wrong reading was therefore correct on everything it could be checked against,
and returned 1 for every int4 op in the file. The vendor batches, and 80 is both its widest
and its most common width.

The stream census over-counted DPU-only runs threefold. Target `0x0401` was missing
from the reader's table, and an unknown target ends a run, cutting each affected op into
three. The fix takes 12,724 DPU-only streams to 4,416.

The census fix also removes a taxonomy. The "six DPU-only program kinds", grouped by `0x4010`
and `0x4050`, were the fragments containing whichever registers fell on each side of a cut.
Regrouped, there is exactly one kind, with both registers reading 0. Their file keeps the old
table, marked as a record of the bug rather than of the model, and offers no replacement
taxonomy. Assume none.

#### Further model reads

**Conditions:** Read for these notes on 2026-08-18, offline, with the reader as it stood then,
before either fix.

A second model, Qwen3-0.6B-Base-rk3576-w4a16-grq v1.2.3 (725 MB, HF `MichaelAndrewFischer`,
2026-08-18), separates model shape from runtime policy. The reader finds 19,380 convolution
streams (its DPU-only count of 4,160, and the 23,540 total, are pre-fix readings). The
M=32-128 program counts are nearly identical to Llama-3.2-1B's (1108/820/732/676 against
1108/856/728/672). The KV-bucketed attention-program population is therefore converter
policy, not model shape. Qwen3's FFN down-projection (K=3072, inside the 4608 slice bound
measured for these notes) dispatches whole with only the two-core output-channel split. The
vendor K-splits only when forced past its slice bound.

A third model determines the split policy and the combine mechanism
(Qwen3-1.7B-Base-rk3576-w4a16-grq v1.2.3, 1.6 GB, same HF author, read 2026-08-18,
offline, both files re-fetched). The reader finds 22,740 convolution streams (its DPU-only
count of 6,064, and the 28,804 total, are pre-fix readings). The 1.7B's FFN-down (K=6144)
never dispatches whole: every instance is a 4096-piece plus a 2048-piece, oc-halved
(`blk.N.ffn_down.weight_rkllm_spilt_0/1` in the file's own tensor names). The vendor's
per-dispatch K bound is therefore exactly 4096, in greedy chunks. It is the one rule that fits
Llama-1B's 8192 = 4096+4096, this 6144 = 4096+2048 (not 3072+3072), and the 0.6B's 3072
whole.

The combine runs on the NPU: a dedicated DPU-only elementwise program, one per split pair.
At M=1 each of the 1344 `[4096-piece][2048-piece]` pairs is followed by exactly one. Prefill
sites have the same program in pixel-bucket variants. Its primary operand comes from memory
(`0x400c = 5` where a conv has `0x40000004`). Its second comes through DPU_RDMA (`0x5xxx`
words live, including the exact `0x5034 = 4000004c` / `0x5044 = 000280a1` values of the w4a16
next-job hazard in [chips/rk3576.md](chips/rk3576.md)). Both are at the pair's output
geometry (oc-half × M pixels).

Its `0x4010` input width code is 5, the 32-bit code the coefficient-A fp32 readback
anchored, where the attention-interior EW program has 2. The partials are therefore float,
separately scaled (`ffn_down` is the only projection with a `_C_secondary` coefficient group,
one multiplier set per K-piece). Nothing in the mechanism touches int32: no integer GEMM in
any of the three read models exceeds K=4096 (the int8 LM head is K=2048, whole).

The split pieces' conv programs are register-identical to never-split projections outside
pure geometry. The combine is invisible to a per-register diff, and was read from the program
sequence and counts. The program type is not exclusively the combine: ~16 instances in each
file follow a `2176x32` fp16 op. A count test alone would misattribute them, and adjacency
separates the uses.

The two stream classes past the reader's decode are identified. The weight-bits-0 streams are
chained no-weight-fetch delta task programs. A prefill M bucket is pixel-chunked (0.6B:
53+53+22 = 128, 1.7B: 40+40+16 = 96), and the last chunk restates neither the weight
registers nor the full geometry. The reader's bits formula misparses those chunks, 228 on
the 0.6B and 340 on the 1.7B. Every one is adjacent to same-`ic` full programs whose pixel
counts it completes. The 1.7B's 672-count "bits 4096" class is the same thing with a weight
fetch: trailing K-pieces as delta programs.

The defects do not reach two results. The attention-interior EW population is identical across
the two Qwen3 files (4120+24 of kind `0x4010=40000002`, KV-bucketed). Neither Qwen3 file has
any norm tensor, so the decode path's norm/residual/activation work is off-NPU in converter
v1.2.3, whatever the stream census says. The Llama file's converter version is unstamped.

#### Status of claims from these reads

The claims derived from the 2026-08-18 reads sort by whether the defects can reach them:

| Status | Claim | Reason |
|---|---|---|
| Unaffected | Everything read from the convolution streams' geometry and weight registers. That covers the per-dispatch K bound of 4,096 in greedy chunks across three models, the oc-halving, and `ffn_down`'s `_C_secondary` per-piece scales | The conv count is identical either side of the `0x0401` fix (8,808 both ways), so conv parsing never fragmented |
| Unaffected | The KV-bucketed attention population. The M=32-128 program counts matching across the two Qwen3 files, and hence "the KV-bucket population is converter policy, not model shape", stand | fp16 streams are the case where the row and pixel counts agree |
| Withdrawn | "M again in {1, 32, 64, 96, 128} with M=1 dominating (14,280)" | It is the row-count artifact in its pure form. It says nothing about how the vendor batches those models |
| Not established | The DPU-only stream counts, and with them the argument that the DPU-only share collapse is converter policy rather than architecture | That argument's main quantitative support was Llama's dominant per-projection EW kind at 8,268 streams, and the fix removes 8,308 Llama DPU-only streams. The population and the correction are the same size to within half a percent, so it was very largely the bug |

A residual collapse survives arithmetically, 33% against 18-21%, rather than the 59%
against 18-21% claimed. It has not been re-read, and the "one kind, both registers 0"
regrouping leaves the kind-based half of the argument without terms.

#### Cost model

Two of its results corroborate findings in these notes. M=1/2/3 exact through the open driver
(seven 1×1 convs at projection shapes) independently corroborates the no-M-constraint fact in
`rocket_matmul_rk3576.c`. Its 32-task chained jobs at ~26.3 us/task marginal (~172
us/submit removed) corroborate the one-kick mechanism that `patches/rk3576/npu/0015`-`0016`
implement.

Their cost model is a fixed cost (2026-08-31), refitted over five stages on the busier core.
It is `us a call = 128.7 + 36.8*tasks + 110.0*MB`, with RMS 9 us on a 540 mean, inside 2.4% at
every stage. Per token that is 11.5 ms of call, 7.4 ms of task and 29.1 ms of weights. Of the
hardware path, 39% is dispatch, not bandwidth [their HW sweep, ROCK 4D].

The author has since withdrawn the 39% and the fit's 36.8 us task coefficient (charsiu
`docs/paper-evidence.md` §8, round 412, read at `cf16d2e`). Their round 152 measured a job at
16.85 us + 4.81 us a task, and the per-call term later fell to 43-51 us.

The fixed-cost model replaces their earlier weight-fetch fit,
`us/task = 26.3 + weight_MB * 84.3`. Prefer the later fit to the 84.3 line: both are theirs,
and the later one is fitted on more shapes.

Two independent measurements agree that DRAM does not limit that board. Eight reader threads
pulling 11.93 GB/s alongside a decode change the decode by 0.1%, and the decode changes them
by 0.1%.

One of their calls issues one submit per core and waits on both. Every "us a submit" that
tree had printed was therefore half a call's latency. The fixed term they published as 112 us
is 224 per call. Their totals were right while the per-unit number was out by two.

Two of their submit-shape options were read nowhere (`CHARSIU_NPU_MAXTASK`) or read only for
an unrelated threshold (`CHARSIU_NPU_NOCHAIN`). Every round that set them therefore measured
its own baseline twice.

#### Decode

Their decode claims are not measured for these notes. They claim that NPU decode is viable on
the RK3576, and that the vendor runs M=1 decode (~13 tok/s on that board and model). Their
arithmetic projects 11.8 tok/s at int8 and 22.7 at int4 for Llama-3.2-1B. Their stated deciding
measurement is one projection, NPU against four A72 cores, at M=1 and M=32. The claim challenges
the decode-on-CPU default taken over from the RK3588 [their measurement + projection, untested
here].

**Conditions:** ROCK 4D, `librkllmrt` 1.3.0 run on that same board rather than quoted,
Llama-3.2-1B, NPU 594 MHz on both sides unless stated, CPU pinned. The decode figures are board
log r390 (2026-09-12), and the prefill figures r393 (2026-09-13).

Decode is 1.39x the vendor at 594 MHz, and 1.46x with both sides at 786 MHz. Their decode
scales with the NPU clock and the vendor's does not, so the multiple depends on the condition
rather than on either runtime. Prefill is ahead at every prompt length from 27 to 602 of
their own tokens, and level at 852.

The vendor's chat template adds a constant 33 tokens at every length, so each row is the same
input text. The margin a row has to clear is measured, not assumed. Twenty readings of one
configuration at 302 tokens on one boot, changing nothing, spanned 2.5% of their median. The
852 row at +2.1% is therefore called level rather than a win. The vendor's decode is inert to
the NPU clock: 32.4% more clock changes it by -0.9% [their HW sweep].

Their NPU decode does not contradict the decode-on-CPU rule, because M=1 is not efficient on
the RK3576 either. At K=1024 int8 their per-output-channel cost runs 0.111 us at m=1 and
0.202 at m=80. Eighty times the rows cost 1.8 times as much, because the weights are
read once for the batch. Per channel and row, the cost is 0.0025 us at m=80 against 0.111 at
m=1, a factor of 44.

Their decode gain comes from removing dispatch cost, not from the NPU being efficient at GEMV.
It is bounded by a host side with one A72 cluster, where the RK3588 has four A76s. Whether
decode belongs on the CPU therefore depends on the host, and its failing on the RK3576 says
nothing about the NPU. The conclusion is consistent with the platform limits in
[chips/rk3576.md](chips/rk3576.md).

#### The w4a16 output stage

Their w4a16 output stage multiplies the two operands' fp16 bit patterns as signed integers.
Their probes fit the output as `((int16)fp16bits(w) * (int16)fp16bits(a)) >> 16`, 18/18
measured points exact. The fit is consistent with a partially-set float mode, which on the
RK3576 is three registers moving together ([chips/rk3576-regcmd.md](chips/rk3576-regcmd.md))
[hypothesis, unverified against their stream].

Their int4 matmul is built on that arithmetic. It packs its operands for the stage, and is
byte-exact against a CPU reference at K 32 to 224 from rounds 280-281 (2026-08-20,
`docs/lab-notebook.md` §"An int4 matmul that computes"). It is their default weight format,
at the perplexity cost in §"Coverage gaps".

A w4a16 job with the vendor's values in RDMA `0x5034`/`0x5044` leaves the next job timing out.
The hazard is in [chips/rk3576.md](chips/rk3576.md) §"An externally reported hazard on the
w4a16 RDMA path", beside the next-submit corruption a wide output causes.

#### Quantization evidence

Their quantization evidence has a matched-magnitude noise configuration inside the harness
rather than beside it. Unstructured error at the same per-tensor magnitude costs +60.00%
perplexity, where the vendor's actual quantization costs +13.32%. The vendor row therefore
sits four and a half times further from noise than from the reference.

The noise configuration caught a defect. An earlier reconstruction scored ppl 1701 at 18.3%
weight error, where Gaussian noise of the same magnitude scored 32.10. A reconstruction 53x
worse than noise of its own size is a broken reconstruction, not a quantization result.

> [!NOTE]
> The 1701/32.10 pair is a fault signature, not a finding about quantization. Do not quote it
> as one.

They also find that a genuine reduction in Frobenius weight error changes quality in the
opposite direction on two models of eight. Weight error therefore cannot predict even the sign
of a quality change.

#### Requant derivation

Their requant derivation is an independent cross-check on the per-column ramp in these notes.
Their `src/job.c` derives a 15-bit multiplier with an implicit leading 1, and takes the shift
from the float's exponent bits. It rounds to nearest by incrementing on the bit below the
field, and renormalizes on carry. The output stage is
`clamp(max(requant, 0) + offset, -128, 127)`, with the clamp at zero applied before the
offset. That clamp alone would cost the negative half of the range, and a "lift" recovers it
exactly: `+128/mult` folded into coefficient A, with `offset = out_zp - 0x80`. The lift is
exact because `max(rq + 128, 0) + (out_zp - 128) = rq + out_zp` for every `rq >= -128`.

Their DPU requant registers are single, not per-channel: `0x40ac` offset, `0x40b0` scale,
`0x40b4` shift, one triple per program. Per-channel behavior comes only from the coefficient
table (coefficients A, B and C) and the fp16 scale table. The per-column entry in
`librocketnpu` has to do the same. The single triple is also a second derivation of the field
widths the `worst_rel_err` ramp is bounded by.

#### Output-width registers

Their output-width bundle is six registers. Their option `CHARSIU_WIDE8` forces the w4a16
output stage onto an int8 weight job, one register at a time:

| Bit | Register |
|---|---|
| bit0 | `0x4010` |
| bit1 | `0x4030` low |
| bit2 | `0x4038` |
| bit3 | `0x4044` |
| bit4 | `0x4050` |
| bit5 | The `0x40ac`/`0x40b0`/`0x40b4` identity requant |

> [!CAUTION]
> In their sweep, bit0 and bit4 must be set together. Either alone hangs the NPU.

Bit3 and bit5 are not needed at all (`0x37` and `0x1f` both give the exact four-byte
accumulator). They run a four-byte output on every dispatch, across processes, and report no
next-submit corruption. They also report that the width has exactly two values, 1 and 4, with
no 2. Nothing in this bundle therefore selects an fp16 output width.

The whole stage is measured for these notes. Its `0x4010`, `0x4038` and `0x4050` clear the
next-submit corruption that the regcmd encoder's partial output stages cause
([chips/rk3576.md](chips/rk3576.md#a-whole-output-stage-that-does-not-poison)). The int32
matmul entry in `librocketnpu` runs on the whole stage
([chips/rk3576-regcmd.md](chips/rk3576-regcmd.md) §"The dense writer").

#### CBUF window

Their CBUF window is six registers derived from a start bank. They read it from a 2026-09-04
census of the vendor's Llama-3.2-1B `.rkllm`, where every op appears twice. On that reading the
CBUF is two mirror-image windows of 7 banks inside 14, and their runtime gives each core its own
(`charsiu_job.cbuf_window = di`). The six registers, and why `ROCKET_RK3576_CBUF_BIAS` moves a
window's base without its extent, are in
[chips/rk3576-validation.md](chips/rk3576-validation.md) §"The six-register CBUF window".

#### Two cores in flight and the NPU supply

They attribute two-cores-in-flight corruption to an undervolt, and the two-core runs in these
notes were at the operating point they name as unsafe. On their board the corruption appears
at 786 MHz and 750 mV and vanishes at 594 MHz or at 800 mV. Their test was the element probe at
width 24 and KMAX 1024. Their sweep, and how it bears on the two-core corruption these notes
record, are in [chips/rk3576-validation.md](chips/rk3576-validation.md) §"Undervolt".

Their `overlap_safe()` serializes unless the rail (from sysfs) and the clock (from debugfs) sit
inside the vendor's table at its lowest bin. At 800 mV they overlap batched calls, and nine
models match the serialized configuration with TTFT 845/1183/3963/2891 ms against
1037/1565/5073/3604 [their HW sweep, ROCK 4D, 2026-09-04].

#### Reusable tools and techniques

None of these depends on their hardware:

| Item | What it does |
|---|---|
| `tools/check_consistency.py` | States evidence-pack rules as conditions rather than verdicts, requires supersession to be mutual, and self-tests each rule against a document that violates it |
| `src/sentinel.h` | Puts one sentinel word per row rather than per element. It is cheap enough to leave enabled in production, and still distinguishes wrote-nothing from wrote-all from stopped-midway |
| Their `/dev/cpu_dma_latency` hold | Gives +24% decode by forbidding deep idle, because `rk3576.dtsi` gives `CPU_SLEEP` a 250 us exit latency |
| Their spin-then-sleep fence wait | Exploits that `timeout_ns` is an absolute deadline, so a value of 1 makes `PREP_BO` a non-blocking poll. The poll wins 81% of waits at a 200 us budget |
| `tests/mel_cross.py` | Sets its tolerance by measurement rather than by a constant: run the reference in f64 and f32 and fail only above `max(TOL, 2*(f32 - f64))` |

#### Coverage gaps

They have no rotation work: a full-tree grep for Hadamard, QuaRot or SpinQuant returns only
RoPE. They found the diagonal AWQ variant and leave it off by default, and their int4 has a
perplexity cost of +102% (Qwen3-0.6B, NPU 594 MHz, their README of 2026-09-14). The
Hadamard-plus-per-column-requant results in these notes bear directly on that loss. They emit
the DPU LUT block `0x4100`-`0x4194` as all zeros with no comment, where `librocketnpu` has the
input map and a working activation. They have no pooling or PPU work.

#### Withdrawn reset claim

> [!NOTE]
> One claim of theirs is withdrawn by the person who measured it. Do not use it.

Their README reported that RK3588 testing had reproduced a silent failure, once in 102
induced resets. The failure was an inference that signaled success while its output buffer
was never written, on the unpatched configuration only: a differential silent-corruption
signal.

On 2026-09-12 he found the error in his own reports. The script kept every inference's
scorer output per round and never aggregated it, so the summaries scored only the one
inference after the forced autosuspend. Aggregated, that constant-`0x80` buffer is in nearly
every run, on every configuration, on all three dates. It is not a differential signal: it is
what a job canceled by a reset looks like from userspace, whatever made it miss its deadline.
The mechanism is `drm_sched_start()` completing detached jobs with `-ECANCELED`.

Their v13 series puts the correction into the commit messages. The case for those three
patches rests on the source analysis rather than the protocol.

### gahingwoo `kiln`

The vendor RKLLM/RKNN stack run on a mainline kernel, 7.1.3+ (`github.com/gahingwoo/kiln`,
GPL-2.0). It has the GPL `rknpu` driver built out of tree plus a small kernel patch set, an
installer, and an OpenAI-compatible server. `charsiu` measures against it. It also has the
live-capture harness (`capture/rknpu-regcmd-dump.patch`) that complements the offline `.rkllm`
reader.

Its ten kernel patches apply to 7.1.7 in series order with zero fuzz under
`git apply --check` (read 2026-08-26 against a prepared 7.1.7 tree). They touch only
`pm-domains.c`, `rockchip-iommu.c`, `rk3576.dtsi` and `rk3576-rock-4d.dts`, so the stated
7.1.3 target is not an upper limit there.

Two of its kernel-side statements bear on the RK3576 NPU, read against the out-of-tree device
tree (2026-08-18).

The first, "two IOMMUs / four MMU banks, mainline drives one", is about the vendor `rknpu`
aggregate node. That node is one device listing both iommus, with the second left unattached,
so the second core reads IOVAs as physical addresses. The statement does not transfer to
`rocket` as it stands. The RK3576 NPU is two iommu instances of two banks each (`rknn_mmu_0`
at `0x27702000`+`0x27702100`, `rknn_mmu_1` at `0x2770a000`+`0x2770a100`, in the
`patches/rk3576/npu/0007` DTS), and each core device has its own. Mainline `rockchip-iommu`
iterates every bank of an attached instance, so single-core `rocket` already manages both banks
of core 0. The mechanism is therefore not a candidate for the two-jobs-in-flight corruption
there.

The second is in their `0007`/`0008` iommu patches (authored by Jiaxing Hu, the RK3576 RFC
author), which name a mechanism whose symptom appears in the H96's logs. A bank left by
firmware with `PAGE_FAULT_ACTIVE & !STALL_ACTIVE & IDLE` silently drops `CMD_ENABLE_STALL`
and delays the other banks past the poll timeout. The recurring
`rk_iommu ... Enable stall request timed out` on the H96 during long probe runs matches that
failure path exactly [symptom match, the orphaned-fault status bits unverified on that board].
Mainline fixes the same condition from v7.3-rc1 [source-confirmed, from the commit]. Commit
`b10d5920cafa` acknowledges the stale fault with `CMD_PAGE_FAULT_DONE` before enabling stall,
where `0007`/`0008` skip the bank. The commit has no `Fixes:` or stable tag.

### gahingwoo `mesa-rk3576`

Their Teflon/`rocket` RK3576 driver as a maintained Mesa fork
(`github.com/gahingwoo/mesa-rk3576`, branches `rk3576` and `rk3576-clean`, on Mesa 25.3.0,
read at `1f0de93`, 2026-08-31). It has the conv paths, depthwise in 64-channel groups, CBUF
allocation and reuse, and a `ROCKET_REG_SET` late-override option. It is a second independent
RK3576 regcmd encoder to diff against, for the RK3576 what Mesa's `rkt_regcmd.c` is for the RK3588.

It has a field-level decode of DPU `0x4050`, which corroborates the RK3576 regcmd encoder in
`librocketnpu` on one axis and diverges on another. The decode is fitted against a corpus of 94
vendor `.rknn` compiled offline (oc 4-1024, ic 3-1024, 1×1 to 224×224, k 1/3/5, stride 1 and
2).

On depthwise the two agree, independently. Their word is
`0x00013033 | ((DIV_ROUND_UP(oc,16) - 1) & 3) << 8`, and `npu_regcmd_rk3576.c`'s
`r76_dw_bs_cfg()` computes the same base constant and the same field. Their stated reason
is that depthwise pads output channels to 64. The last bank therefore holds one to four
16-channel atoms, and the field is that count minus one. The comment in `npu_regcmd_rk3576.c`
gives the same derivation, and it holds on all 13 depthwise models, three of them predicted
rather than fitted. Upstream Mesa hardcodes `0x00013133`, the 32-channel case, so every
depthwise outside oc 32/96/160 is misconfigured there.

On the direct (non-depthwise) path the word is oc-dependent for them. They select `0x80011111`
when `DIV_ROUND_UP(oc, 16)` is even and `0x80011011` when it is odd. The two differ in bit 8,
which is inside the field the depthwise path uses for its atom count. They also report that an
`oc % 32` form is wrong at oc 56, the one shape that times out on their board under it. That
form was fitted on multiples of 16 only, which is every count the vendor sweep covered. They
report that the vendor toolkit picks the parity answer at that count.

In `librocketnpu`, `R76_DPU_BS_CFG_DIRECT` is `0x80011111` unconditionally, and the regcmd encoder pads
a direct conv's output channels to whole 32-channel groups. Its atom count is therefore always
even, and their rule gives the same word at every count it programs. The odd-count word's test
on the padded programs is in [chips/rk3576-regcmd.md](chips/rk3576-regcmd.md) §"Depthwise
geometry registers".

Their single-field sweep of the five unexplained `0x4050` fields (round 260, at oc 128 and
oc 88, each field moved to upstream's value alone) gives:

| Field | Result |
|---|---|
| `RGP_CNTER` | Wrong output with no hang |
| `RESERVED_0` | Correct output, but the job times out |
| `SIZE_E_0` | Wrong output and a hang |
| `OW_SRC` | Wrong output and a hang |
| `SIZE_E_1` | Inert at both shapes. They later resolved it as the depthwise flag, 0 in all 81 regular models and 1 in all 13 depthwise ones |

> [!NOTE]
> `0x4050` is not a single-purpose register. Four of these five fields change the result,
> and the width bits sit in a word whose other fields encode the program's own geometry. A
> wide-output stage writes the whole word: the dense writer in
> [chips/rk3576-regcmd.md](chips/rk3576-regcmd.md) §"The dense writer" programs `0x00023333`
> with the rest of `charsiu`'s output stage.

### Chaoyi Chen (Rockchip), `PC_TASK_CON` bit assignment

A Rockchip engineer's reply on `linux-rockchip` giving the RK3576 field layout of
`PC_TASK_CON`, at
[lore.kernel.org](https://lore.kernel.org/all/4f300b78-d96d-4d98-8819-dc292b0c9b97@rock-chips.com/).
It includes the name of the control the RK3588 description marks reserved:
`BIT(18) task_last_layer_clear`. It is the only authoritative statement of that register
available. Everything else these notes state about the register was read off the silicon, see
[chips/rk3576-regcmd.md](chips/rk3576-regcmd.md).

## Userspace stacks

### `johanvdb/librocket`

A FOSS userspace fp16 matmul, run as a 1×1 conv on mainline `rocket`, written "for GGML
projects". It has a working single-task fp16 matmul and includes a kernel NULL-deref iommu
patch. It combines Jasbir Matharu's `rk3588-npu` register headers with the Mesa regcmd format.

It has no tiling, quantization, multicore or int8. `librocketnpu` built its kernel-access
layer from it, and adds those and a rewritten shim. Its `rocket_interface.c` passes
`timeout_ns` raw, an absolute-deadline bug that `librocketnpu` fixes.

### `johanvdb/ggml` `rocket-backend`

A ggml backend skeleton. `*-matmul.cpp` returns -1 (CPU fallback), and `*-dequant.cpp` is
unimplemented. It covers the structure of a backend and has no working model path.

### mtx512 / jas-hacks `rocket-userspace`

The original RKNN-reverse-engineering repository, with its blog (jas-hacks.blogspot.com,
"RK3588 reverse engineering RKNN"). Its `gen_matmul_task` covers the register configuration of
a matmul-as-1×1-conv over the NVDLA CNA/CORE/DPU blocks. It targets the proprietary rknpu
ioctls (5.10 BSP). Driving it through `rocket` requires swapping the shim and adding the
DPU_RDMA block it omits.

### `fukumori/iwagumi`

A ggml and GGUF runtime for the RK3588 on the vendor `rknpu` ioctl path (Apache-2.0, a single
commit, read 2026-08-18). It runs prefill on the NPU and decode on the CPU, at fp16 and W8A8
with an opt-in W4A4 behind quality checks. The split is the one these notes record, applied to
the vendor driver. It names kernel 6.1.115 with `rknpu` 0.9.8 as its known-good configuration.

### elepedus/laya-rk3588-turingpi-rk1

The Laya decision model (`convaiinnovations/laya`) on a Turing RK1 through the vendor stack,
read 2026-09-27. Rust services drive `librknnrt` 2.3.2 and OpenCL, on Ubuntu 22.04 with kernel
5.10.160 and `rknpu` 0.9.2. The model is a ModernBERT-large encoder (d 1024, 28 layers, 16
heads, a 128-token sliding window with global attention every third layer) under a two-layer
decision head. A multilingual variant has the mmBERT-base shape (d 768, 22 layers).

Their figures, none reproduced here: 0.12-0.26 s for a short request and 2.25 s for a typed
639-token one. RKNN's static shapes cost them 7 fixed-length buckets times 3 checkpoints,
~13 GB of `.rknn` files. The fp16 drift grows from 0.0055 to 0.36 mean error over 22 layers.
Three cores measure 1.57x over one. Their CPU comparison is PyTorch in eager mode, a weak
baseline.

`rknpu-submit` refuses `rknpu` 0.9.2 at open. The vendor's memory-create ioctl grew from 40
to 48 bytes at 0.9.6, and the open provider accepts only the later form. The same model
through `ort-rocket` is in [perf/benchmarks.md](perf/benchmarks.md) §"ONNX Runtime
(ort-rocket)". Their banded sliding-window attention, query tiles with a key halo, is the form
`librocketnpu`'s ModernBERT encoder takes for its windowed layers.

### Mesa Teflon on `rocket`

Upstream Mesa's Teflon TFLite delegate, the public FOSS-rocket baseline, as packaged by
rpardini/mesa-teflon-etnaviv-rocket-docker and documented in the BredOS wiki (`NPU/rocket.md`).
It needs kernel >= 6.18 with `CONFIG_DRM_ACCEL_ROCKET`, uses `/dev/accel/accel0`, and loads
`libteflon.so` through `tflite.load_delegate`.

It supports quantized uint8 CNNs only: conv + EW-add + fused ReLU, single-core, with no SiLU
and no transformer. AVGPOOL, RESHAPE and SOFTMAX fall back to the CPU. It runs MobileNetV1 in
~13-17 ms (~3-4x over CPU ~48 ms). `rocket-userspace`/`tflite-rocket` run 3-core per-fd,
fp16/int8/int4, SiLU/GELU and transformer blocks, and MobileDet.

The rpardini repository also includes per-board NPU-regulator DT patches (CM3588-NAS,
NanoPC-T6, Turing RK1). The `patches/rocket` clock/volt coupling depends on that voltage
wiring [source-confirmed].

### `llama.cpp` ggml NPU-backend discussion

An upstream attempt at an RK3588 (and Tenstorrent) ggml backend (ggml-org/llama.cpp#8111). It
stalled, and the author moved to Tenstorrent for its open stack. It covers the device-buffer
problems of an NPU backend for ggml, and supports the design `ggml-rocket` takes.

The attempt hit four problems: managing device buffers, needing the datatype and layout at
allocation time, no offset-subbuffering, and the per-fd 4 GB IOVA limit. `ggml-rocket`
avoids all four through the BLAS-backend model. It uses host (CPU) buffers, offloads only
`MUL_MAT` through `graph_compute`, and packs and DMAs inside `rocket-userspace` per call.

The guidance slaren gives there is the API surface `ggml-rocket` does not need: `set_tensor`
passes `tensor->type`, and `get_alloc_size` / `get_max_size` / `get_alignment` serve
device-buffer backends. The 4 GB-window / no-subbuffer constraint applies, and
`rocket-userspace` IOVA management handles it (`rocket_prepacked_int8.c` escape check),
not the ggml buffer-type [source-confirmed].

### poad42/opennpu_rk3588

An ONNX/JAX-to-RK3588 compiler and runtime (MIT, first commit 2026-08-02, read at `983282e`),
whose PJRT plugin makes the NPU a first-class JAX device (`jax.devices()` -> `[npu:0]`). It
drives the vendor `rknpu` BSP driver on a 6.1 vendor kernel (`/dev/dri/card1`, with
`MEM_CREATE`/`MEM_MAP`/`MEM_SYNC`/`SUBMIT`/`ACTION`).

Its runtime, ioctl layer and kernel patches do not transfer to the mainline `rocket` path, and
its register encoding does. It has the same author as poad42/smolvlm_rk3588_full_npu_native,
which has its own entry. These notes cite its docs, `cna_matmul.c`, `lm_forward.c` and the four
kernel patches.

#### Register program

The repository has two register-program paths. The formula path, `pjrt_c/cna_matmul.c`, emits
the CNA descriptor as 112 `uint64` entries computed from the M, K and N dimensions. It adds a
weight cache that preloads all 144 GPT-2 matrices at init. The template path replays captured
vendor programs, `matmul_tmpl.h` (1.0 MB), `tanh_tmpl.h` (384 KB) and `relu_tmpl.h`, with the
DMA addresses patched. The templates limit their op table to "MatMul supports 5 GPT-2 shapes", and only the
formula path states a register encoding.

`docs/ARCHITECTURE.md` says the templates were "captured once from vendor toolkits via ioctl
tracing". The README's provenance section says no proprietary code was involved.

The formula path agrees register for register with `librocketnpu`'s matmul program
[source-confirmed, read from `cna_matmul.c`]. The entry encoding is
`(op << 48) | (val << 16) | reg`, and the block tags are `0x0201` for CNA, `0x0801` for CORE
and `0x1001` for DPU. The CNA registers are `0x100C`, `0x1010`, `0x1014`, `0x1020`, `0x1024`,
`0x1030`, `0x1040`, `0x1070`, `0x1084`, `0x1088` and `0x1110`, with DPU `0x4020` as the
destination base. Their `0x1040` leaves bits [10:8] zero, an independent instance of the
`FC_DATA_BANK` = 0 rule in [encodings/cbuf-reuse.md](encodings/cbuf-reuse.md). They write the
`FC_DATA_SIZE0/1` pair as `(1<<16)|M` and `K`, as `npu_regcmd.c` does.

#### Machine parameters

Two of its machine parameters match these notes: CBUF = 12 banks × 32768 B, and `nbuf_size=0`
with `CONFIG_ROCKCHIP_RKNPU_SRAM` unset. Its guide also states "M=1, 2, 3 produce completely
wrong results", and `npu_cna_cache_run_m()` pads to `M4 = 4` before every submit. Its matmul
program writes the feature surface stride as `4 × ((M/4) − 1)`, clamped at 0, which is correct
only at `M%4==0` ([matmul-as-conv.md](matmul-as-conv.md) §"Feature surface stride"). Its M
result therefore reproduces Erratum E3 and is not a second observation of a hardware limit.

Their host weight scatter (`weight_fp16_off`, 16 outputs per group, 32 inputs per group) and
feature scatter (`feature_data_off`, C2=8 fp16 input, C2=4 fp32 output) state the native cube
layout independently. Their scattering on the host at all corroborates the finding that no
register converts a layout. The README's "reads W directly as fp16, no reordering" describes
the absence of quantization: the weights are still tiled.

Its CBUF bank split allocates no slack bank: `fd_banks = ceil(M·K·2 / 32768)`,
`wt_banks = 12 - fd_banks`. The split is correct for its datatype. The formula path's feature
is fp16 (`0x100C` precision 2), and an fp16 feature cube keeps the exact-fit sizing. The
feature-DMA overread past the allocation affects int8 only [HW sweep]
([encodings/cbuf-bank-slack.md](encodings/cbuf-bank-slack.md)).

#### Entry encoding in the companion report

The companion report, *OpenNPU v1.0* (amohan.dev, 2026-08-18), mislabels the entry fields
[source-confirmed, against Mesa's `target` enum and `rkt_regcmd.c`]. It describes an entry as
a 16-bit register, a 16-bit value and a 32-bit tag. `codegen_synthesize.py` packs that layout
(`'<HHI'`). It fills the tag's low half with the value's upper 16 bits for every register, not
only for DMA addresses.

Those bytes are the 32-bit value field of the `(op << 48) | (val << 16) | reg` encoding. A
16-bit value written from the report's prose therefore truncates any wider field. The tag's
high half is the block target: `CORE0_ID` (`0x1001`) tags the DPU `0x4xxx` registers and
`CORE1_ID` (`0x2001`) the DPU_RDMA `0x5xxx` ones. The report's claim that the three cores are
addressed as `0x1001`, `0x2001` and `0x3001` misreads that field.

#### Stated limits

Three of its stated impossibilities are bounds of its own harness:

| Their statement | What bounds it |
|---|---|
| "Multi-core model-parallel matmul is impossible" | A position-locked 28 KB template block and a ~3.5 MB scratch DMA window. `librocketnpu` runs 3-core per-fd |
| "K > 768 requires tiling" | Their bank split's own consequence: `cna_matmul.c` clamps `Kt` to 768 and accumulates the tiles on the host |
| "w8a8 … 16% error, 256 levels insufficient for 768-wide" | The per-tensor-scale error that a Hadamard rotation plus a per-output-channel requant removes |

Their DMA limit belongs to that path. `ARCHITECTURE.md` fits
`t = 0.08 ms + bytes / ~1000 MB/s` and calls 1 GB/s "the NPU internal DMA engine's hardware
limit". The README's results table states 3.0 GB/s. The two figures disagree, and both are far
below what `librocketnpu`'s register programs sustain.

#### GPT-2 decode

It reports GPT-2 124M at 28 ms/token (36 tok/s), KV-cached, tabled as "NPU 1.4x". Taken at face
value, it is a second challenge to the decode-on-CPU default, after the RK3576 vendor's M=1
decode. It is the weaker of the two on record.

Its comparison column is RKLLM at 33 ms/token, which is Rockchip's NPU runtime rather than a CPU
measurement. The timing (84 matmuls × 0.32 ms = 21 ms/token) holds only at 3 GB/s. Their own
1 GB/s model puts a 236 ms minimum under GPT-2's 144 matmuls. `lm_forward.c` keeps the KV cache,
attention, softmax, LayerNorm and GELU on the CPU. It offloads only the four per-layer
projections, at a padded M=4 [their claim, internally inconsistent, untested here].

#### PJRT plugin and kernel patches

It has a working PJRT plugin (`pjrt_c_api.h` v0.112 pinned to jaxlib 0.10.2), a frontend shape
that none of `ggml-rocket`, `tflite-rocket` or `ort-rocket` covers. Its four kernel patches add:

- Batch submit
- An IOMMU lock-free domain fast path
- A `kmem_cache` for job structs
- Fence fds

Three of them independently reach, on the vendor driver, the per-submit overhead reductions in
`patches/rocket`. The patches are plain `diff -u` against `drivers/rknpu/` on the vendor
`develop-6.1` tree, and none of them applies to `rocket`. What transfers is that a second
effort, on the other driver, converged on batch submit and an IOMMU domain fast path.

The batch-submit and fast-path patches have four defects [read from the series]:

| Where | Defect | Consequence |
|---|---|---|
| `rknpu_batch_submit()`'s `err_iommu` unwind | Puts the domain once per allocated job rather than once per acquired one | A failure at job k of n underflows the refcount by n-k |
| Its `core = ffs(core_mask) - 1` | Takes only the lowest set bit while `run_count` still counts the whole mask | A multi-core mask queues on one core, decrements once and never commits: a silent hang. Nothing validates the mask |
| `sd->job` | Published under `irq_lock`, then committed to hardware with no lock held | A window in which the IRQ handler sees a job the hardware has not been programmed for. It is the same driver-state-versus-hardware-state divergence as the RK3576 two-jobs-in-flight hazard |
| 0002's fast path | Reads the refcount and then increments it non-atomically where `atomic_inc_not_zero()` is the primitive, with `iommu_domain_id` read outside the atomic besides | Safe only because one domain is ever used |

There are two smaller defects. `rknpu_batch_submit` gets no prototype in any patched header,
though 0003 calls it. The README's `patch -p1` from `drivers/rknpu/` cannot apply paths rooted
at `a/drivers/rknpu/`. Their 0003 also contains a `dma_buf` leak fix on fd close (bare
`vunmap` to `dma_buf_vunmap` + `unmap_attachment` + `detach`). The leak is the same BO-lifetime
class as `patches/rocket` 0013/0014, present in the vendor driver too.

#### Bus QoS

The repository programs SoC-level AXI QoS for the NPU, a device-tree change
[their measurement, untested here]. `docs/ref/QOS_TEST_RESULTS.md` notes that the vendor driver
leaves `bw_priority_addr = 0x0`, so the NPU's bus QoS is never programmed at all. The repository
sets `priority-init=7` / `mode-init=0` on all five NPU QoS nodes through the `pm_domains`
driver's own DT path:

| Node | Address |
|---|---|
| `qos_npu0_mwr` | `0xfdf72000` |
| `qos_npu0_mro` | `0xfdf72200` |
| `qos_npu1` | `0xfdf70000` |
| `qos_npu2` | `0xfdf71000` |
| `qos_mcu_npu` | `0xfdf72400` |

Their measurement: 2x on ~200 KB matmuls (`[64x768x64]` 0.658 -> 0.322 ms, `[64x64x768]`
0.416 -> 0.228 ms) and nothing past 1 MB (1.00x, 0.96x, 1.00x on three shapes). GPT-2's 144
large matmuls are unchanged at 235 vs 237 ms. The setting sits upstream of the driver entirely,
and unlike the rest of that repository it is expected to port to `rocket` unchanged
[expected]. It is a different layer from the MCIF per-client 2-bit QoS at `0x8000` in
[nvdla-lineage.md](nvdla-lineage.md), and neither layer has been measured here.

Their write-up reads the null result at 1 MB as proof that the limit is the internal DMA
engine's ("definitively proves"). A null result at one priority bounds bus arbitration, and does
not isolate the engine. The 1 GB/s limit it argues for is also the figure their own README
contradicts at 3.0 GB/s.

#### Counter page

`docs/ref/NPU_REGISTER_INVESTIGATION.md` is a second probe of the counter page, by a kernel
module's `ioremap(0xfdab0000)` on core 0 under the vendor driver. It reports the opposite
severity on each page from the probes under `rocket`. `0x2210`-`0x223c` gives a contained
DECERR oops, and `0x8000`-`0x803c` hangs the bus until the watchdog reboots. Both probes are
kernel-side `ioremap` and `readl`, so the split is not obviously an access-path artifact.

Their report reads both pages as unimplemented counters. The comparison, and the power-state
hypothesis that would reconcile the two, are in
[perf/hw-byte-counters.md](perf/hw-byte-counters.md) §"An independent probe under the vendor
driver".

#### Register facts

The same document reports two register reads. `VERSION`, at PC offset `0x0`, reads
`0x46495245`, ASCII "FIRE". Mesa's `registers.xml` names that register `VERSION`, and the
vendor driver reads it for `RKNPU_GET_HW_VERSION`.

It also reports that `0x1004` bit 4 is not writable on the RK3588: `0x1e` written there reads
back `0x0e`. It reads the bit as an internal-memory enable that the RK3576's `state_init` sets.

The reading is wrong. `0x1004` is the CNA's `S_POINTER`. Its bit 4 is `pointer_pp_clear`, a
write-1-to-clear action that returns the register-group pointer to group 0 [TRM, RK3588 Part
1, `RKNN_cna_s_pointer`]. A readback of 0 after a write of 1 is consistent with that
definition. The RK3576's `state_init` writes `0x1e` to clear the pointer
([chips/rk3576-regcmd.md](chips/rk3576-regcmd.md) §"The `PC_TASK_CON` field width"). The
readback therefore says nothing about whether the two SoCs integrate different NPU revisions.

The field table is in [encodings/regcmd-task-model.md](encodings/regcmd-task-model.md)
§"Does register state persist across tasks (can later tasks send a delta)?".

#### SigLIP

Its vision figures, on the vendor driver, are for comparison and do not transfer. SigLIP ViT
runs at 995 ms/image on one core against a ~2500 ms CPU baseline. Across three cores it runs at
370 ms/image with six or more images in flight, and one image alone takes 1181 ms. Its cosine
is 0.9999 against the HuggingFace reference. All 48 matmuls run on the NPU, with attention /
GELU / LayerNorm on the CPU under OpenMP. The full block on the FOSS path reaches cosine
0.999998 ([encodings/siglip-encoder.md](encodings/siglip-encoder.md)).

### dnhkng/open-rknpu

An int8 compiler from ONNX to regcmd, in Python, and a libc-only runtime for the RV1103/RV1106
NPU (`rockchip,rv1106-rknpu`), MIT, first commit 2026-09-11, read at `0e0afcd`. It drives the vendor `rknpu` driver v0.8.2 with no IOMMU, on a
Luckfox Pico Mini B at 420 MHz. The vendor toolkit served only as a black-box oracle.

It compiles dense, depthwise and transposed int8 convolution, 2×2 pooling and elementwise joins.
It also runs LUT sigmoid and tanh and whole small DAGs. Every claim is tied to a board ledger
of 1,799 models checked byte for byte against a Python integer reference.
`docs/investigation-log.md` is the chronological record, failed hypotheses included.

#### Register map

Its register map is a third point between the RK3588 and the RK3576 [source-confirmed, read from
`src/open_rknpu/register_profile.py` and the emitters]. CORE, DPU and DPU_RDMA sit at the
RK3588's offsets with the RK3588's fields. The CNA geometry words, the feature and weight bases
and the pad value at `0x1184` also follow the RK3588. A few CNA words follow the RK3576 instead.
`0x103C` and `0x1044` contain the data entries and the input width, and `0x118C` is the input
width minus one in both halves.

The pads at `0x1068` are byte-packed, where the RK3588's are nibbles. The `0x118C` formula
matches the RK3576's in [chips/rk3576-regcmd.md](chips/rk3576-regcmd.md). A CORE or DPU result
of theirs therefore ports to the RK3588 more directly than an RK3576 result does.

#### Results reproduced on the RK3588 and the RK3576

Three of their results reproduce here. Each is written up in the note that documents it.

The deconvolution mode runs a whole ConvTranspose in one task once the output extent is
programmed. Their transposed emitter sets the deconv stride fields to `s-1` and keeps the
input geometry undilated. It programs the output geometry to the transposed extent and pads by
`k-1-p`. With those settings it runs 8×8 -> 17×17 at pad 2 in one task, and about 90 models
are byte-exact on their board.

Their stride fields are live without `CONV_CON1[16]`. On the RK3588, with the bit set, the same
program is bit-exact up to 64×64 -> 128×128 [HW sweep, Turing RK1, 2026-09-23]. See
[encodings/conv-transpose.md](encodings/conv-transpose.md).

`OUT_CVT_SHIFT[30]` selects the tie rule. Their field experiment sets bit 30 in an elementwise
add, and 1,438 predicted bytes move from half-to-even to half-away-from-zero. Both settings
reproduce on the RK3588 and on the RK3576 [HW sweep, 2026-09-23]. See
[encodings/out-cvt-converter.md](encodings/out-cvt-converter.md). Their `quantization.md` says
the conv path rounds half away. Their code and tests say half to even, and the board agrees with
the code.

The BS stage shifts a product held wider than int32. Their per-channel epilogue is
`round_even(((acc + A) * C) >> 14)`. A is the bias, and a Q14 multiplier C and the negated
weight zero point share its 32-byte record for four channels. The record is the RK3576's A, B
and C coefficient record at a different grouping.

The epilogue is exact on their board at an accumulator of 2.08e9 with `C = 16384`. On the
RK3576 the `0x5024` shift word behaves the same way: wide, multiply before shift, ties to even
[HW sweep, H96 MAX M9, 2026-09-23]. See [chips/rk3576-regcmd.md](chips/rk3576-regcmd.md).

#### Results consistent with these notes

Four more of their results agree with these notes:

- The elementwise engine's second operand read needs its surface notch. Without it, a compact
  operand hangs.
- The second operand read is linear, one atom per output pixel, with no spatial broadcast.
- `BS_OW_CFG.OW_SRC = 1` hangs.
- The 2×2 average rounds half to even.

Their LUT setup words (`0x410C = 0x50500`, `LE_START = 0xffffc000`, `LO_END = 0x4000`)
are the ones the RK3576 vendor programs contain. They are also the RK3588 values in
[encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md) §"The two tables (NVDLA LE/LO
hybrid)", where `0x50500` is `LUT_INFO` with both index selects at 5.

Their LUT's negative half advances by a whole-bit gain that follows the stem's weight scale,
which they fit and did not decode. It has the shape of a sign-dependent shift ahead of the
table [hypothesis].

#### RV1106-specific limits

Three of its properties are specific to the RV1106 and its kernel, and do not transfer:

- The RV1106's channel limits: 511 weight parts per task, and 8192 output channels from a 9-bit
  block index
- Its per-task costs at 420 MHz
- The runtime's completion scheme for a vendor kernel built without fence support: a
  non-blocking submit drained by a blocking barrier job

### happyme531/rknpu2_matmul_open and open_rknpu2_lm_infer

Two AGPL-3.0 userspace projects for the RK3588 that program the NPU directly with their own
register programs, on the vendor `rknpu` driver. `rknpu2_matmul_open` is a matmul library.
`open_rknpu2_lm_infer` runs LLM inference on it, decode included. Read 2026-10-21. Neither was
run on `rocket` for these notes.

These notes take five facts from them:

| Fact | Where it is used |
|---|---|
| fp16 tasks that stream a weight cube through the CBUF to N 4096, with the weight banks sized for one 32-kernel group [source-confirmed] | The streamed form, reproduced on `rocket` ([matmul-as-conv.md](matmul-as-conv.md) §"CBUF weight streaming") |
| fp16 device time of ~2.6 TFLOPS over three cores at K 1024 and N 4096, at 1 GHz [source-confirmed] | The highest matmul rate reported for this silicon ([perf/not-mac-bound.md](perf/not-mac-bound.md) §"Streamed weights") |
| M 1-8 laid along the CNA width at height 1, with a line stride of 0x200 and a surface stride of -3M [source-confirmed] | Reproduced on `rocket`. The -3M is the feature surface stride `width × (H − 4)` at width M and height 1 ([matmul-as-conv.md](matmul-as-conv.md) §"Feature surface stride") |
| DCOMP as a lossless entropy codec over dense int8 weights, with its register values and stream layout [source-confirmed] | The leading description of the compressed format, and the compression it gives on real weights ([encodings/cna-dcomp-weight-decompression.md](encodings/cna-dcomp-weight-decompression.md)) |
| Decode on the NPU with per-channel int4 and int8 weights, at an implied ~24 GB/s over three cores, with the DDR pinned at 2112 MHz and one task run across all three cores [source-confirmed] | The decode bound ([perf/decode-gemv.md](perf/decode-gemv.md) §"Weight-streaming bandwidth on `rocket`") |

The vendor driver runs one task across the three cores, which `rocket` does not offer. The
decode rate therefore does not transfer to `rocket` as it stands.

### Seeed-Solution/openvoicestream

Seeed's streaming ASR and TTS product stack for Jetson, the RK3588, the RK3576 and the
Raspberry Pi, with its RK engine `suharvest/rkvoice-stream` and `suharvest/rknn-matmul-parallel`
(MIT, read at `9521320`, `5978d04` and the 2026-04-12 head). On Rockchip it runs the vendor RKNN and RKLLM runtimes and has no register-level
material. Four of its measurements bear on this silicon.

#### Whisper encoder and decoder

A whole-graph Whisper encoder runs faster than `librocketnpu`'s matmul offload, by 1.70x on one
board rather than the ~2.5x their figure implies. `rknn_model_zoo`'s whisper-base encoder is
fp16 at a 20 s window on three cores. By their measurement it takes ~250 ms on their RK3588 at
an NPU clock of 1 GHz, and about the same on their RK3576.

Their model file, run with their runtime version (`rknnlite` 2.3.0) on all three cores of a
vendor-kernel RK1 at 1000 MHz, takes 344 ms [HW sweep 2026-09-26]. Their figure is not
reproduced. The same-board comparison and what bounds the RKNN graph are in
[encodings/whisper-encoder.md](encodings/whisper-encoder.md) §"In-model fused integration".

Their vendor default also ran the decoder on the NPU with no KV cache, a 12-slot window. A CPU
decoder with a cache took their RK3588 from RTF 0.149 to 0.061 and cut long-form WER. The
whisper.cpp and ggml-rocket combination already makes that split.

#### Per-op decode and batch prefill

Per-op NPU decode is 5x slower than a compiled graph. The rknn-matmul-parallel engine runs
Qwen3-ASR-0.6B's decoder (28 layers, d 1024) as 196 `rknn_matmul_run` calls a token on the
RK3576. The decoder costs 225 ms a token at fp16 and 146 at int8, against RKLLM's 43. Their own
breakdown is matmul 91, weight rebind 91, lm_head 35 and CPU 8 ms.

Batch prefill at M 60 costs 17 ms a token against 202 row by row. The row-by-row cost is the
dispatch-bound decode of [perf/not-mac-bound.md](perf/not-mac-bound.md), reached through the
vendor API. The repo's "~16 ms/token" headline and their writeup's 21-29 ms disagree with that
table.

#### Datatypes and CPU idle state

Their "the RK3576 NPU has no bf16" describes the toolkit's datatype menu. Under a raw register
program the RK3576 contracts bf16 bit-exactly, see [chips/rk3576.md](chips/rk3576.md)
§"The precision fork".

Their RK3576 performance test disables the CPU's deep idle state without measuring it. On the
RK3588 that state costs an offloaded prefill a few percent, see
[perf/cpu-governor-and-offload.md](perf/cpu-governor-and-offload.md).

#### Vendor-runtime defects

They report these defects for a reader on the RKNN path, and none is reproduced here:

| Runtime | Behavior |
|---|---|
| RKLLM and RKNN | They share IOMMU domain 0 by default, and `rknn_run` hangs once RKLLM has loaded. RKLLM's `base_domain_id = 1` separates them (rknn-llm#437), and an RKNN model context cannot leave domain 0 |
| librknnrt 2.3.x int4 matmul | Loses about half of every positive value, a nibble-order defect (rknn-toolkit2 PR #412) that they measured at cosine 0.06 |
| librknnrt 2.3.2 | Segfaults in `rknn_run` on RK3576 Linux, and 2.3.0 does not |
| The RK3576's `rknn_matmul_api` | Offers fp16 operands only with an fp32 output |

#### Kokoro TTS

The RKNN compiler refuses a layer past the 13-bit cube-dimension limit, or falls back to the
CPU. Kokoro's vocoder at 13,080 samples fails its `REGTASK` check with
`limit: 0x1fff, value: 0x3318`. The limit is in
[encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md) §"13-bit cube-dimension
maximum".

Where the compiler does place Kokoro's generator on the RK3588, at their 16-token bucket, it is
slower than the CPU. It takes 3464 ms at fp16 and 2656 at int8, against 1819 ms in ONNX
Runtime.

### boundarybitlabs/rkwhisper

`boundarybitlabs/rkwhisper` and `boundarybitlabs/rkwhisper-models` (read 2026-10-10, with the
models' v0.1.0 release): a Rust stack built on the vendor runtimes, with bindings for
`librknnrt` and RKLLM, a Whisper daemon and an OpenAI-compatible server. Its `rknn-inspect` prints
the runtime's per-op profile. It has no register-level material. Two of its Whisper export
techniques bear on this silicon.

#### STFT as a polyphase convolution

Whisper's STFT runs on the NPU as a polyphase convolution [source-confirmed, their
`MELSPEC.md` and export script]. The STFT is a stride-160 conv with a 400-tap kernel over one
channel of 480,400 padded samples, past the 13-bit cube limit of
[encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md). With `g = gcd(400, 160)`
= 80, the host moves sample `80t + c` to channel `c`, column `t`. The conv becomes a 5-tap,
stride-2 conv over 80 channels and 6005 columns, with the weight permuted to match. The fold
is exact for any kernel and stride with a common factor.

On this silicon the fold also fills the channel lanes a one-channel input leaves empty. It
shrinks a kernel wider than the RK3576's 5-bit kernel field
([chips/rk3576-regcmd.md](chips/rk3576-regcmd.md#the-geometry-fields-are-narrower-than-their-register-halves)).
They clamp the STFT's real and imaginary outputs to ±255.89 before squaring, since a larger
square leaves fp16's range.

#### Decoder as one graph per token

The decoder runs on the NPU as one graph per token [source-confirmed, their exporter and
daemon]. The graph takes the token, a static 448-slot self-attention cache and every layer's
cross-attention keys and values. It returns the logits and the new token's keys and values.
Three mask inputs update the cache by multiply and add, so the graph has no scatter. One mask
restricts attention to the filled slots, one selects the slot to write, and one keeps the rest.

Their daemon re-sends every input through `rknn_inputs_set` at each step: 12 MB for tiny, 24 MB
for base and 72 MB for small [host-computed from their model spec]. Most of it is
cross-attention K/V, constant within a window. The daemon also primes the prompt one token per
step. None of their timings is reproduced here.

## Quantization and coherence references

### r/RockchipNPU

Field reports that int8/int4 LLMs on RK3588 need INT8_HADAMARD / INT4_HADAMARD for coherence
(e.g. gpt-oss-20b). They agree with the finding here that a Hadamard rotation is mandatory.

### `rk-llama.cpp` issue #9

Q8_0 and Q4_0 prefill, and decode, on the proprietary rknpu2 stack, independently confirming
the direction here. It reports "Q8_0 maps well to RKNPU W8A8" -> Q8_0 prefill +200-400%, Q4_0
maps poorly, and decode is memory-bound, as in the CPU-decode split here.

Their Q8 gain is operand bandwidth at a lower fixed dispatch overhead, not a capability the
open path lacks. RKLLM quantizes to W8A8 (8-bit weights and activations), halving the bytes
moved for both operands on a path that is dispatch/DMA-bound. Their own Gemma int8 runs at
40-58% NPU utilization, so their stack is not MAC-bound either.

Their gain does not come from native on-device int32 K-accumulation. The DPU-EW adds int32
exactly under the whole precision bundle, and a K-accumulation built on it is not measured
(see [encodings/k-accumulation.md](encodings/k-accumulation.md)). See
[perf/not-mac-bound.md](perf/not-mac-bound.md) §"Vendor-stack int8 prefill".

## Outlier-aware quantization

### QuaRot, SpinQuant and SmoothQuant

QuaRot and SpinQuant (Hadamard-rotation quantization) and SmoothQuant: the principled fixes
for incoherent int8/int4 output driven by activation outliers. `ggml-rocket` uses a Kronecker
Hadamard (H_{2^k} ⊗ H_60 via Paley for Gemma's non-power-of-2 K). Measured perplexity with
W8A8 + Hadamard tracks fp16, see [encodings/k-accumulation.md](encodings/k-accumulation.md).

### AWQ

Activation-aware weight quantization
([mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)), with per-group scales plus
protection for the ~1% salient channels (largest activation magnitude). The repository includes
the reusable scale-search, not just weights. It addresses int4 quality from the weight side
(W4A16 as published), and complements a Hadamard rotation on the activation side.

### Outlier Suppression+

Wei et al. 2023 (arXiv:2304.09145, ModelTC/Outlier_Suppression_Plus): a per-channel shift
(which removes asymmetric activation outliers) plus a per-channel scale, both migrated into
adjacent layers (free at inference). It is near-FP at 8/6-bit. It complements a Hadamard
rotation, and the asymmetric shift fits the asymmetric uint8 weights of the detection models
`tflite-rocket` runs.

## Transformer softmax and exp on NPUs

### Attention Distribution-Aware Softmax for NPU-Accelerated On-Device Inference of LLMs

Sadheerthan et al., *Electronics* 2026, 15, 1312: a LUT approximation of `exp` for softmax on
NPUs. It reports softmax as the NPU transformer bottleneck, because NPUs lack a native `exp`
unit.

Its approximation is distribution-aware and variable-degree (PSO-learned non-uniform segments
snapped to a 128-bin grid). It cuts exp-kernel cycles-per-call ~18.5% vs uniform Degree-4. It
optimizes the exp kernel's compute, while the fused-encoder softmax here is bound by host↔NPU
transfers. A slower host `expf` softmax ran 3x faster than the on-NPU LUT path (see
[encodings/whisper-encoder.md](encodings/whisper-encoder.md) §in-model). Its quant-domain,
clamp-`[-20,0]`, in-domain row-max recipe is the design reference for a fully-on-NPU resident
softmax (no host round-trip).

### THU-MIG/yolov10 issue #115

A YOLOv10-on-RK3588 latency report
([THU-MIG/yolov10 #115](https://github.com/THU-MIG/yolov10/issues/115)). YOLOv10s is ~2.7x
slower than YOLOv6/DAMO-YOLO on the RK3588 (32 vs 12 ms @416) despite fewer params. Its
attention/PSA blocks map poorly to the NPU. It is a detection datapoint on the same theme as
the softmax entry: attention and softmax run poorly on this NPU.

### LSH9832/edgeyolo

EdgeYOLO (anchor-free, YOLOX-style) on the RK3588 through the proprietary RKNN runtime, in
`cpp/rknn`. Decode and NMS sit inside the `RKNN::YOLO` class, not in a reusable,
backend-agnostic postprocessor.

Its one transferable datapoint is an activation cost. The LeakyReLU variant (Tiny-LRELU) runs
65 FPS vs 24 FPS for the SiLU variant at 384×640, int8, 3-core. The author attributes the gap
to "SiLU activation layers", a ~2.7x activation-driven cost, the same theme as the YOLOv10 PSA
datapoint. For throughput in a YOLO-family graph, prefer LReLU over SiLU. The number is an
RKNN-path reference, not the DPU-LUT path here
([encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md)).

## Decode and multi-instance throughput

### Hummingbird+

Li et al., *"Hummingbird+: Advancing FPGA-based LLM Deployment from Research Prototype to Edge
Product"*, FPGA '26, doi:10.1145/3748173.3779189. It is a dedicated FPGA GEMV engine (Zynq
UltraScale, 140 DSPs, <1K LUTs, 272 GOPs) for edge LLM decode. It identifies no GEMV
optimization that applies to fixed silicon.

Its speedups (DSP pre-adder operand packing, DDR LUT-mux elimination, DOT/AXPY mode switch,
W4/KV8 dual precision) are reconfigurable-datapath microarchitecture. They have no analogue in
the RK3588's fixed convolution pipeline. What transfers is bandwidth reduction at the model
and format level (4-bit weights + 8-bit KV, MoE), already available in stock llama.cpp. It
also cites RKNN-LLM at "~10 token/s on a 3B on RK3588," so vendor on-NPU decode is
bandwidth-bound too, and does not beat the CPU. See [perf/decode-gemv.md](perf/decode-gemv.md).

### Accelerating OpenPangu Inference on NPU via Speculative Decoding

Dai et al. 2026 (arXiv:2603.03383, wujing215/OpenPangu7B-with-Medusa): speculative decoding on
an NPU. It combines a Medusa multi-head draft (no separate draft model), static tree attention
and zero-copy retrieval, to fit the NPU's static-graph execution. It reports 1.35x on short
sequences, and long sequences are memory-bandwidth-bound. It is the closest prior art for
speculative decoding on an NPU, and its static-tree design is the component that ports to a
fixed-graph NPU.

### `leafqycc/rknn-multi-threaded` and HN 48527630

Two working RK3588 multi-instance inference pools: the `leafqycc/rknn-multi-threaded`
repository, and the one alebal123bal describes in HN 48527630. Each is a thread pool + queue + one RKNN context per core,
round-robin, with a preallocated buffer pool and RGA preprocessing. The reported gains are
2.6x (YOLOv5s) / 31->46 FPS (full isp->RGA->YOLOv8n pipeline). They are a reference
architecture for a multi-instance throughput pool (Frigate multi-camera).

### johnjiamzhong/AlertGateway on zhihu

"我以为 NPU 推理到了硬件极限，结果发现是 CPU 频率在拖后腿" (zhihu, johnjiamzhong/AlertGateway,
RK3588S, RKNN path, `zhuanlan.zhihu.com/p/2051444846548857552`): a proprietary-path write-up
on CPU-governor effects. It attributes a "~40 ms is the NPU hardware limit" YOLOv8s number to
the CPU governor, not the NPU. `rknn_run` includes CPU-side submit + IRQ-wait, so it swings
59 -> 35 ms (−41%) from the CPU governor alone (NPU clock fixed at 1 GHz). Pinning both the
CPU and the NPU to `performance` removes the jitter.

It independently corroborates the CPU-side submit and dispatch overhead here, see
[perf/not-mac-bound.md](perf/not-mac-bound.md) §"Per-submit overhead reductions" and
[perf/clock.md](perf/clock.md). It covers method only, with no FOSS-path numbers (it never
leaves rknn).

## The NVDLA ancestor

### NVDLA hardware manual v1

The NVDLA hardware manual ([nvdla.org](http://nvdla.org/)): the only authoritative prose about
this datapath, since Rockchip publishes a register list and no semantics.
[nvdla-lineage.md](nvdla-lineage.md) works through which of the questions here it answers,
which it answers wrongly for this silicon, and which it cannot reach. It also has a per-page
reading list.

The structure transfers (blocks, the ping-pong register file, CBUF banking and ports, CACC's
within-one-layer accumulate, the MCIF/SRAMIF arbiters). The arithmetic details do not always
transfer: it specifies round-half-away-from-zero, and this silicon rounds half to even.
Several fields this silicon depends on (`PC`, `DECONV`, the CBUF granule allowance, the
per-sign shift word) are Rockchip additions it never had, so its silence about them is not
evidence.

## Compression

### NVDLA weight-compression format

NVDLA's weight-compression format ([nvdla.org/hw/format.html](https://nvdla.org/hw/format.html)):
CWT (packed non-zero weights) + WMB (1 bit/element mask, 128-B aligned) + WGS (per-kernel-group
byte count). The kernel group is 32 (int8) / 16 (fp16/int16), which matches the weight-tile
groups here. CDMA decompresses into CBUF, skipping zero MACs. It compresses sparse (pruned)
weights, and gives no gain on dense weights.

It documents NVDLA's format, not the RK3588's. By the lineage it is one candidate layout of the
RK3588 CNA's DCOMP [expected]. The DPU is NVDLA's SDP, so the CNA is likely NVDLA's CC. An
external userspace drives DCOMP with a different, lossless codec over dense weights
(§"happyme531/rknpu2_matmul_open and open_rknpu2_lm_infer"). That codec is the leading
description, and this format the alternative. No compressed RK3588 capture exists to confirm
either
([encodings/cna-dcomp-weight-decompression.md](encodings/cna-dcomp-weight-decompression.md)).

## Related RK3588 NPU work

### SHARD

Mohan et al., *"SHARD: A Compatibility Framework for Deploying Transformer Models on Edge
NPUs"*, EuroMLSys '26, doi:10.1145/3805621.3807618, also the amohan.dev blog. It deploys the
SigLIP-B/16 vision encoder (93 M params, the SmolVLM-256M front-end) on the RK3588 through
constraint-driven graph rewrites on rknn-toolkit2. Its methods work around the vendor
toolchain: graph sharding, GELU-approx and LayerNorm-decompose legalization, and fusion
barriers. A regcmd path needs none of them. One finding transfers.

#### REGTASK width limit

The `0xe010 "REGTASK Overflow"` is an undocumented 13-bit instruction-register width limit:
operand indices > 8191 fail [source-confirmed]. The blog reads `0xe010` as a memory overflow.
The compiler's own message is a per-field width check. The 13-bit limit, and the corruption at
it on this silicon, are in [encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md)
§"13-bit cube-dimension maximum".

#### Toolkit limits

Its other limits are the toolkit's. The 32 KB figure comes from synthetic ONNX graphs run
through RKNN: a 32 KB tensor passes and a 32.1 KB one fails. The figure bounds what the vendor
toolchain accepts, and does not isolate a hardware buffer.

RKNN's Transpose limit is 16 KiB. Its fallback warning names 8192 elements in an fp16 build and
16384 in an int8 one [source-confirmed: rknn-toolkit2 2.3.2 build log].

#### Table 1 on the same silicon

**Conditions:** Turing RK1, `rknpu` 0.9.8, rknn-toolkit2 2.3.2.

Its Table 1 does not reproduce as labeled. Each row maps to a run in SHARD's repository
[HW sweep 2026-10-03]. Its 8.7x over RKNN divides a whole caption generate by its own encoder.
On the same silicon its pack takes 3.0x the wall-clock time of the toolkit's own fp16 encoder
build. The row-by-row mapping, the measurements and the attribution of the pack's fidelity loss
are in
[perf/data/siglip-encoder.md](perf/data/siglip-encoder.md) §"SHARD's Table 1 rows".

The paper's finding that AWQ fails "because the error stems from activation outliers, not
weight sensitivity" agrees with the Hadamard activation-rotation choice.

### poad42/smolvlm_rk3588_full_npu_native

SHARD's own code, by the same author, read at `fe3eefc`, so it does not corroborate SHARD
independently. It contains the conversion and runtime pipeline on the proprietary
`rknn-toolkit2` and RKLLM bindings (no stated OSS license on the main code), and the board logs
behind the paper. None of the code transfers to the rocket path.

Its files say what each Table 1 row measured:

| Table 1 figure | File | What it measured |
|---|---|---|
| 19.63 s | `sbc_results_runs/ablation_tiling_run.txt` | A 50-token `generate()` with the language model on the CPU |
| 1.40 s | `BASELINE_RESULTS`'s `monolithic_encoder_offload.rknn` | An fp16 build that file scores at 0.99 |
| int8 0.02 | `export_broken_int8.py` | A calibration on uniform noise with one planted 1000.0 element |

SHARD's `disable_rules` turn off the toolkit's SDPA fusion, the rule that keeps the score
transpose out of the graph. Its per-layer "vanilla" sweep adds a score transpose (the
`score_transpose` probe) to force the CPU fallback. Two of its scripts do not parse at that
commit, and the reproduction is in [perf/data/siglip-rknn/](perf/data/siglip-rknn/).

### clehaxze.tw gemlog

"Benchmarking RK3588 NPU matrix multiplication performance", episodes 1-2 (2023), with the
measurements in the 2024-02-14 post. It measures the same silicon through RKNN, an upper
reference: fp16 ~900 GFLOPS peak, int8 ~2x fp16, int4 ~4x (in MAC terms), and K-spilling past
~1024. It gives the per-cycle MACs as 2048 int4 / 1024 int8 / 512 fp16. It notes that RKNN
matmul lacks multi-core (the rocket path runs 3-core via per-fd).

### Martin Chang, "porting LLM to RK3588"

An RWKV-on-RK3588 talk on K reduction in the convolution, and the prefill/decode split. The
conv reduces over K in one pass up to the CBUF limit ("Max K=2048 @ FP16" in one pass), and
nobody used the DPU eltwise for K-accumulation. `librocketnpu`'s tiling is built on the
one-pass reduction.

The talk also states the decode split: the NPU is good at MatMul and bad at GEMV (M=1). CPU
GGML wins decode (NPU 83 ms vs CPU 61 ms/token) because llama.cpp's NEON GEMV is already
bandwidth-saturating. See [perf/decode-gemv.md](perf/decode-gemv.md).

### kevbuh/rk3588

Notes and datasheets stating int4/int8/int16/fp16/bf16/tf32 support on the silicon. They
corroborate the precision menu.

### widgetii/orangepi5plus-npu

An independent FOSS RE of the RK3588 NPU on the same `rocket` driver. It replays the vendor's
register programs from `.rknn` binaries rather than generating them. It also includes a
standalone driver library, Mesa optimization patches and a QEMU model of the NPU. Its research
report counts kernel calls per inference on both stacks. RKNN makes 63 IOCTLs and 1 submit,
and Mesa's Teflon driver over `rocket` makes 634 and 10.
[perf/iova-and-multicore.md](perf/iova-and-multicore.md) cites that count.

### t-firefly ROC-RK3588S NPU wiki

Vendor RKNN usage only, with no multi-core/SRAM/register/driver detail.

### sagi21805/matmul-npu

A C++/OpenCV matmul wrapper over the proprietary RKNN toolkit2 (fp16/int8 in -> fp16/int8/fp32
out). It exposes no regcmd/ioctl/tiling detail.

### `llama.cpp#722`

7B-Q4 on an RK3588 (NanoPi R6s) at ~98 s/token (2023), an early CPU-only, microSD/RAM-bound
outlier (not representative of current llama.cpp CPU perf). It shows a 7B LLM on the RK3588's
CPU alone as impractical, the case for the NPU-prefill / CPU-decode split.

## Evidence

### Regcmd and hardware sources

#### allbilly/npu and allbilly/rk3588

The integer EW ALU and the `PPU_RDMA_DATA_FORMAT.IN_PRECISION` results replace two negatives
these notes once recorded. Both were field-semantics errors of the same shape, a reading that
is correct on everything it could be checked against. The first int32 K-accumulation probes
here moved allbilly/npu's int-EW `EW_OP_TYPE` bit alone, varying the ALU algorithm alone. They
could not rule an integer EW path out. The sweep that found the ALU exact under the
whole precision set is `tests/ew_int_probe.c`.

#### gahingwoo `charsiu`

The DPU-only share collapse argued from the 2026-08-18 reads does not survive the reader fix,
and these notes no longer use it. It read 59% of Llama's streams against 18-21% on both Qwen3
files, and attributed the difference to converter policy. That rested on Llama's dominant
per-projection EW kind (`a0000002`/`00023333`, 8268 streams) having no counterpart in either
file. That kind was the fragment population the missing `0x0401` target manufactured, to
within half a percent of the 8,308 streams the fix removes.

The regrouped census has one DPU-only kind with both registers reading 0, so the contrast it
was built on has no terms left. Qwen3 needs more norm work than Llama (QK-norm), which is why architecture was a poor
explanation for the disappearance. With the Llama file's converter version unstamped, even
the surviving half is argued from that direction rather than read off a version field.

Their earlier cost fit (above) gave 11.9 GB/s, M nearly free, and a second core ~5% worse at
these shapes. It was the weight-fetch-bound reading that the platform limits in these notes
predict. They refuted it themselves on 2026-08-31. A 9.4 GB/s average over stages running
6.67 to 15.60 GB/s is not a bandwidth limit. Such a limit does not have a 2.3x spread across
shapes, and a fixed cost does. And `gate+up` moves 253.8 MB in 18.59 ms, 13.65 GB/s with its
own dispatch still inside it.

The entry as first read, on 2026-08-18 from their README of 2026-08-15, recorded w4a16 and int4
as their open defect: "w4a16/int4 does not compute". Their int4 matmul computed from rounds
280-281 (2026-08-20), and §"The w4a16 output stage" states the current reading.

#### gahingwoo `kiln`

Applied out of series order against an unpatched tree, five of its patches report offsets and
two report fuzz. That is a property of that test, not of the series.
