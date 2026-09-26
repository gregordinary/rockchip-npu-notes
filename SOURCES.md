# Sources: external references for FOSS RK3588 NPU work

External resources others can consult for RK3588 NPU work, with one line each on **why it
matters**. Each entry names its upstream so you can find it yourself. These are context and
cross-references. The facts in these notes are established by HW sweep and the FOSS Mesa
driver (see the [README](README.md) evidence tags).

## The authoritative regcmd / hardware sources

- **Mesa `rocket` (Teflon) driver** (`src/gallium/drivers/rocket/` in Mesa).
  The single most useful source: a *working*, in-tree FOSS driver that emits real
  conv regcmd for this exact hardware. `rkt_regcmd.c` (`fill_first_regcmd` = the
  validated CNA→CORE→DPU->DPU-RDMA sequence, the `add_tensor` eltwise geometry, the
  enable mask, the MRDMA-disable block), `registers.xml` (every register address +
  field), `rkt_coefs.c` (weight packing, the WEIGHT_ATOMIC_SIZE=32 reorder, the
  CACC 48-bit accumulator note), `rkt_ml.c` (feature packing, FEATURE_ATOMIC_SIZE=16),
  `rkt_task.c` (NVDLA-style tiling/split). INT8-only (TFLite delegate), so it does
  not show the fp16/int4 paths, but it is ground truth for the format.

  **Two correctness fixes landed 2026-06-27** (read at `31e2daea`, 2026-08-18),
  both found against MobileNetV2 and both worth reading rather than just noting:

  **The fused conv+add requant constants now have a closed form**, replacing tables of magic
  floats captured from one model. Any scale absent from the table had been falling back to
  `add_scale = 0`, silently dropping the residual. The form is
  `EW scale = addition_scale / (input_scale * weights_scale)` and
  `output scale = (input_scale * weights_scale) / output_scale`. After it, the output path
  is identical to the non-add case, and the fused conv's `output_scale`/`output_zero_point`
  describe the ADD output tensor. We do not carry those tables: our requant is derived. This
  is therefore an independent statement of the same algebra to check ours against, not a
  defect here.

  **Weight packing must pad input channels to `FEATURE_ATOMIC_SIZE` (16), not to
  `MAX2(ch, 16)`.** `fill_task()` tells the hardware to read
  `align(MAX2(ch, 16), 16)` channels, while `rkt_fill_weights()` packed only `MAX2(ch, 16)`.
  So any conv whose input channel count is not a multiple of 16 had a misaligned weight stride
  that corrupted **every** output channel (mean error 68 -> 0.5 once fixed). MobileNetV2's
  24-channel blocks are the example. The padding channels carry the weight zero-point so they
  contribute zero. Worth checking against our own int8 conv packer for input-channel counts
  that are not a multiple of the ic group.

- **RK3588 TRM + datasheets**, an external Rockchip reference others can consult. "Rockchip RK3588 TRM
  V1.0-Part1-20220309" holds the NPU register chapter, `RKNN_pc_*` `0x0xxx`, `RKNN_cna_*`
  `0x1xxx`, CORE `0x3xxx`, DPU `0x4xxx`, DPU_RDMA `0x5xxx`, and `pdftotext` works on it. The
  register facts these notes rely on are established independently by HW sweep plus the Mesa
  driver, not derived from it. That is why a TRM statement that contradicts a sweep loses. The
  RK3576 TRM (Part1/Part2 V1.2) is on disk beside it but is **not** cited by
  [chips/rk3576-regcmd.md](chips/rk3576-regcmd.md). That part's register map was established
  from other sources, so treat the RK3576 TRM as unmined rather than as agreeing.

- **allbilly/npu** (`allbilly-npu`), esp.
  `include/rknnops.h`. A higher-level op generator (conv1d/2d, matmul, activations,
  LUTs) using the same Mesa regcmd encoding. Its `float16_alu_op(ALU_ALGO_ADD)`
  encodings are what cracked the **fp16 DPU-EW K-accumulation**. Its int-EW
  `EW_OP_TYPE` bit is what we tested (and ruled out) for int32 K-accum. Broader op
  coverage than Mesa, the reference for going beyond matmul.

- **allbilly/rk3588** (`allbilly-rk3588`): the same author's Python successor to the
  above, and a different kind of source. It carries no register definitions of its own
  (`experimental/registers.xml` and `include/rkt_registers.h` are Mesa's). But
  `conv_expt/capture_harness/decoded/` holds **83 vendor RK3588 conv register programs
  decoded to named CNA/CORE/DPU fields**, multi-task, ic 16-1280, planes to 150, oc
  12-1024, pointwise and depthwise. That is a vendor oracle for the RK3588 conv geometry
  words, of the kind `tests/data/rk3576-vendor-capture/` is for the RK3576. The address
  registers are one session's IOVAs, and only the geometry is comparable.

  Its capture route is a gdb harness on the vendor BSP runtime
  (`experimental/rknn/trace_librknnc_*.gdb`,
  `conv_expt/capture_harness/rknn_prefix_capture.gdb`, which patches the rknpu submit
  struct down to a one-task prefix). That is a weaker instrument than the offline
  compile-and-decode used for the RK3576 captures. It is weaker because it needs a vendor-BSP
  board and captures only what the vendor compiler chose to emit. RK3588 only: nothing in it
  addresses the RK3576 encoding. Neither allbilly repo carries a license file, so treat
  both as readable facts rather than as code or data to vendor.

  **Re-surveyed 2026-09-20 at `e13d99f`.** It has become a second kind of source again: an
  **exhaustive field-by-field triage of the RK3588 NPU register surface**, with executable
  probes, in `examples/expt/`. `trm_register_matrix.md` classifies every non-reserved field
  in Mesa's `registers.xml` (199 register records, 512 field occurrences) across PC, CNA,
  CORE, DPU, DPU_RDMA, PPU, PPU_RDMA, DDMA, SDMA and GLOBAL. Its status vocabulary
  separates local silicon results from captures, from other projects' reports, and from
  TRM-only definitions. A QUARANTINED class holds the sequences after which they saw a crash.

  `trm_register_findings.md` carries the results, `elementwise_int.py`, `pooling.py` and
  `conv_simple.py` the probes, each with an offline `--validate` mode that checks the decoded
  streams without opening the device. Their board is an Orange Pi RK3588 on mainline rocket.
  Nothing below is reproduced here.

  **Two of our documented negatives do not survive it.** Both are field-semantics errors
  of the same shape, a reading that is correct on everything it could be checked against:

  **The per-element EW ALU is not float-only.** They measure same-format integer MAX, MIN,
  ADD, MINUS, ABS and NEG for INT8, INT16 and INT32, across widths 1/2/3 and partial and
  complete C1WC2 surfaces. They document saturation and `INT32_MIN` behavior, and measure MUL
  for INT8/INT16 (its INT32 second operand is signed 16-bit).

  The TRM agrees: DPU `0x4010` enumerates `proc_precision` `3'd4` as Integer 32bit. Their
  streams set the precision as a matched set across DPU `0x4010` and `RDMA_FEATURE_MODE_CFG`
  with the EW converter bypassed. Ours varied the ALU algorithm alone. Corrected in
  [encodings/k-accumulation.md](encodings/k-accumulation.md).

  **`PPU_RDMA_DATA_FORMAT.IN_PRECISION` is a storage width, not a dtype.** The TRM
  enumerates `0x7030[1:0]` as 4/8/16/32-bit, so our native-int8 pooling probe asked for
  4-bit input and got the garbage that implies. They pass signed INT8/INT16 maximum and
  minimum, including adversarial and all-negative vectors, and integer average with Q16
  `round(65536/k)` reciprocals rather than the FP16-encoded ones.

  **Reproduced here** [HW sweep, Turing RK1, 2026-09-20]: MAX, MIN and AVG are bit-exact at
  `PROC_PRECISION = 0` with `IN_PRECISION = 1` and the integer reciprocal. Our negative is
  withdrawn. One refinement their result does not carry: `IN_PRECISION` 2 is exact on all
  three methods, and 3 is exact on MAX/MIN but fails on AVG. So the field is not a strict
  element-width selector on this block, and 1 is the value to program. Corrected in
  [encodings/ppu-pooling.md](encodings/ppu-pooling.md).

  **New capability results worth having**, none measured here. `USE_CNT` is **not** an
  exclude-padding switch: all eight encodings return the include-pad answer, so
  `count_include_pad=False` has no register route in that recipe. `INDEX_EN` writes only six
  bits, `(row & 7) << 3 | (column & 7)`, so it is a local pool-window coordinate and not a
  general ArgMax. Padding with stride adds a non-local vertical phase.

  PPU INT32/FP32 processing precision submits and writes zero on the external-RDMA path.
  Kernel 16 and padding 7 each pass alone and fail combined. Multi-pass 8x8 -> 4x4 -> 2x2 ->
  1x1 average, maximum and minimum pass through NPU-written intermediates with no host repair,
  separately fenced, which corroborates [encodings/ppu-reduce-mean.md](encodings/ppu-reduce-mean.md).
  Explicit PPU_RDMA line and surface strides work while a nonzero `NOTCH_ADDR` corrupts, so
  notch is not an additive line pitch there.

  On the DPU side, scalar operands through `EW_OP_SRC=0` plus `EW_OP_VALUE_n` with
  `ERDMA_DISABLE=1` remove the scratch buffer and its read entirely. BS and BN each fuse a
  configured ALU and MUL, `RELUX_EN` and `MUL_PRELU`, up to five scalar operations in one
  task. And main + BRDMA + NRDMA + ERDMA gives an exact four-input sum in one task. FP16
  comparison results write exact INT16 0/1, and INT16 EW results write sign-preserving
  external INT32.

  **Three state-hygiene failures of theirs are worth reading before trusting a chain here.**
  A two-task CNA spatial split with `WEIGHT_REUSE` computed exact INT8 output, but the first
  following known-good DPU add returned a stale `48576`. Only a second add recovered, and they
  therefore never submitted `DATA_REUSE` speculatively. **LUT reuse across separate rocket
  submissions fails**: a second task without table writes produced unrelated values for
  clear, rearm and untouched pointer states. So all 1,026 entries must be reloaded per
  standalone submit, and same-submit chaining is unproved either way. And their
  `EW_CVT`/`OUT_CVT` probes are QUARANTINED after a reported crash with no retained kernel
  fault record.

  **One quarantined observation bears on an open question of ours**: a controlled `OUT_CVT`
  FP16-to-INT16 probe behaved as expected. But a following INT16 `EW_CVT` probe returned
  `[2, 5, 8, 11, 14, 17, 20, 23]` where nearest-even predicts
  `[2, 6, 8, 12, 14, 18, 20, 24]`, consistent with half-way values rounding downward. Our
  round-half-to-even requant result is measured on the **RK3576**, and the RK3588 is an
  [expected] that this does not confirm. That observation is a different converter stage,
  from a crashed sequence, but it is the only RK3588-side datum either way.

  **They have read these notes** (at `e0c7213`) and cross-check against them explicitly, which
  makes their disagreements useful. Two of their readings of us are stale rather than wrong.
  They report that we find no native deconvolution mode, where
  [encodings/conv-transpose.md](encodings/conv-transpose.md) records `CNA_CONV_CON1[16]`
  `DECONV` live on the RK3588. And they note the notes clone does not carry the `tests/` and
  `src/` those gates live in. That is correct, and is a limit of publishing the notes alone.

- **RKNN-Toolkit2** (`github.com/airockchip/rknn-toolkit2`): the vendor's proprietary
  compile-and-run stack. This project is a mainline alternative to it. Its offline compiler
  output is the reverse-engineering input for the **PPU pooling family**, which the FOSS
  Mesa/Teflon path never emits. A 1-op ONNX pool compiled to a `.rknn` and decoded for the
  PPU / PPU_RDMA register page yields the exact `RECIP_KERNEL = fp16(65536/k)` reciprocal
  format (method in [ppu-rknn-capture/](ppu-rknn-capture/), and no vendor artifacts are
  redistributed). Every other encoding here comes from the FOSS Mesa driver plus HW sweep.
  RKNN3 (targeting RK1820 / RK3572) is a different NPU generation, out of scope.

- **RKNN-Toolkit2 SDK docs** (`github.com/airockchip/rknn-toolkit2/doc`, V2.3.2), the vendor's own documentation, an **external cross-reference**
  rather than RE input. User Guide **§3.5.4**'s high-performance layout table covers the A/B/C
  tile-layout matrix and the same-A/B-dtype-only constraint (cf.
  [encodings/tile-layouts.md](encodings/tile-layouts.md)). Its **§6** covers the quant path
  (INT8-only, per-channel weights / per-tensor activations, range solvers, hybrid FP16
  fallback, cf. [datatypes.md](datatypes.md)), and **§5.3.3** the multi-core split op list and
  the IRQ-affinity tip (cf. [perf/iova-and-multicore.md](perf/iova-and-multicore.md)).

  The runtime header `rknn_api.h` carries the perf/mem query structs, and `rknn_mem_size` is
  allocations. Per-frame bytes are an analytical string in `rknn_perf_detail`, not a hardware
  counter: one more reason [perf/hw-byte-counters.md](perf/hw-byte-counters.md) had to go to
  the silicon. The `OP_Support` / Compiler-Operator-List docs are the op-coverage reference for the
  delegate roadmap. Not the same as RKNN3, which targets RK1820 / RK3572, a different NPU
  generation, out of scope here.

- **RKNN-Toolkit2 issue #163** (`github.com/airockchip/rknn-toolkit2/issues/163`), a YOLOv8s
  compile for `rk3588` under toolkit 2.1.0 and 2.2.0. Its log prints the compiler's `REGTASK`
  check in full:

  ```
  REGTASK: The bit width of field value exceeds the limit, target: v2, offset: 0x500c, shift = 0, limit: 0x1fff, value: 0x20cf
  ```

  Each line names a register offset, the field's low bit and its maximum. So every error the
  check prints is one field's position and width, as Rockchip's compiler holds them. The log's
  three fields match Mesa's `registers.xml` [source-confirmed]: DPU_RDMA `0x500C` `WIDTH`
  [12:0], and DPU `0x4038` `NOTCH_ADDR_0` [12:0] and `NOTCH_ADDR_1` [28:16]. `v2` is the
  target the runtime reports as "RKNPU v2" for this model.

  The layer is the DFL head's 1×1 convolution on a cube 4 high, 8400 wide and 16 channels deep.
  A 640×640 YOLOv8 or YOLO11 head has 8400 anchors (80² + 40² + 20²), over the 13-bit
  ceiling of 8191. The width field holds `0x20cf`, 8400 - 1, and each notch field `0x419f`,
  2×8400 - 1.

  The same log carries two more compiler rules. A Transpose falls back to the CPU when height
  × width exceeds 16384. The three-core split refuses this convolution and runs it on one
  core. The compile does not stop at a `REGTASK` error: the exported model returned an
  all-zero output on the device under both toolkit versions.

  Rockchip's own YOLO export (`airockchip/ultralytics_yolo11`, `RKOPT_README.md`) moves the
  DFL and the post-process out of the model and onto the CPU. It gives two reasons: the
  post-process quantizes poorly, and the DFL is slow on the NPU. Its outputs stay per scale
  (`[1,64,80,80]` and the like), so no 8400-anchor tensor reaches the NPU either. The ceiling
  itself is in [encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md).

- **`rknpu-reverse-engineering`** (phhusson / Tomeu lineage): early-stage,
  STT/TTS-focused, on the **BSP `rknpu`/`/dev/dri/card1` path** (not rocket). Its
  register *encodings* are **superseded by Mesa `registers.xml` + the
  Teflon decode** (more complete, on our actual rocket path), and `rknpu-ioctl.h` is the
  BSP uAPI we don't use.

  **Still-useful artifacts:** (1) `hello2.c`, the **in-core
  IRQ/block-completion bitmap** (CNA/CORE/DPU/**PPU**/DMA-err, two reg-banks per block =
  `RKNPU_JOB_PINGPONG`), now captured in
  [perf/iova-and-multicore.md](perf/iova-and-multicore.md), and confirms the PPU is a real
  separately-completing block. (2) `instrs.h`, a hand-assembled plain conv that
  **confirms our block/register format** (`0x0201`=CNA `0x10xx`, `0x0801`=CORE `0x30xx`,
  `0x1001`=DPU `0x40xx`, for example `DPU_EW_CFG 0x4070=0x383` plain-conv bypass), provenance,
  not new info. (3) The raw hex dumps of the 6 gem BOs
  (weights / **gem2 64-bit instruction stream** / **gem3 10-word task list with
  "jump-to-next-task"** / working / input / output) from real models (`analysis` +
  `mess/dump-*`) are a reference for the **multi-task chaining structure** (task-persistence /
  the dispatch floor). But decoding raw BSP dumps is lower-value than a targeted Teflon
  capture on the rocket path. Its dumps show no integer K-accumulation, and that absence does
  not establish a float-only EW ALU (see [encodings/k-accumulation.md](encodings/k-accumulation.md)).

- **Rockchip Hardware Design Guides**: `Rockchip_RK3588_Hardware Design Guide_V1.4_EN.pdf`
  and `3576_hardware_design_guide.pdf` (V1.1, 2024-05). Board-design documents: no register
  content, nothing about the regcmd interface or the NPU's internals, so they are useless for
  encoding work. What they carry is the **platform envelope** each part's NPU numbers must be
  read against. The RK3576 differs from the RK3588 on every axis of it: the **DRAM bus
  width** (32-bit / 2 channels vs 64-bit / 4 channels, at an identical 2112 MHz PHY clock, so
  exactly half the bandwidth) and the **NPU power rails** (the RK3588 has a separate
  `VDD_NPU_MEM`, while the RK3576 has none, so its CBUF and other NPU arrays sit on the logic
  rail). It also differs in the **peak operating point** (0.800 V / 4 A / 3.20 W vs 0.850 V /
  4 A / 3.40 W, both at 1000 MHz) and the **package thermal resistance** (θJA 15.84 vs
  8.7 C/W).

  The DRAM figure is the load-bearing one. It is the mechanism behind the RK3576's
  DDR-traffic-driven atom drop, its long DPU write drain, and the ceiling on its host cube
  packing. Both are indexed in [chips/rk3588.md](chips/rk3588.md) and
  [chips/rk3576.md](chips/rk3576.md). `pdftotext -layout` extracts both cleanly.

- **6.6 BSP kernel `rknpu` driver**: the vendor kernel driver, with HW performance
  counters and the devfreq/OPP table. Critically for the clock work, `rknpu_devfreq.c`
  shows **200 MHz is the literal `POWER_DOWN_FREQ`**. It also shows that the vendor only ever
  sets the NPU clock while the power domain is active (`!pm_runtime_active` -> refuse).

  rockchip-linux `develop-6.12` carries the same driver, 0.9.8 (read at `470f9dccbdc4`,
  2026-06-29). The two copies differ only in kernel-API porting, so the 6.6 copy is current
  for hardware behavior. The two IOVA allocators that
  [perf/iova-and-multicore.md](perf/iova-and-multicore.md) describes are unchanged.

  Its non-blocking submit path has a unit error: `rknpu_job_timeout_clean()` compares a running
  job's age in microseconds against a timeout in milliseconds. So a non-blocking submit
  soft-resets a job that has run past `timeout` microseconds, 1000x early. The blocking path
  multiplies by 1000 and is correct. What the driver says about the RK3576 is in
  [chips/rk3576.md](chips/rk3576.md): the soft reset's CBUF resets, the per-task completion
  mask, and the SRAM read margin.

- **RK3588 BL31 / ATF binary** (`rk3588_bl31_v1.51.elf`, from rockchip `rkbin`): the
  secure firmware that actually owns the NPU clock. Strings + `radare2`/`aarch64`
  `objdump` confirm the NPU is **SCMI clock id 6**, set in EL3 via
  `rockchip_opteed_clk_set_rate`. It is clocked by a **PVTPLL whose min/max come from
  per-chip OTP** (`adjust npu pvtpll by otp: min=.. max=..`), i.e. no static rate
  table. The same reading confirms **no voltage coupling** in firmware. The source of truth for why cold
  rate-setting wedges EL3 and why the real ceiling is an OTP value. See
  [perf/clock.md](perf/clock.md).

- **Mainline `rocket` driver source + RK1 serial boot log**: `drivers/accel/rocket/`
  (android-mainline vs the local v7.1 build): stock upstream has **no** NPU clock
  handling (`clk_bulk_*` only), so the `clk_set_rate` ramp is the local `rocket-clk`
  patch, not upstream. The serial boot log confirms the SCMI handshake
  (`SCMI Protocol v2.0 'rockchip:'`) and the `quirk_clock_rates_triplet_out_of_spec`
  rockchip clock quirk.

  **Upstream has moved past the v7.1 `rocket` driver and has landed fixes that
  overlap our out-of-tree series** (read from `torvalds/linux` 2026-09-20, with mainline past
  v7.2). Three of them are in `rocket_job.c`:

  **`70e6a33d`** (2026-07-01, Shuvam Pandey) "accel/rocket: initialize job domain before
  cleanup paths": `rocket_ioctl_submit_job()` assigns `job->domain` only after task copying
  and BO lookups, while `rocket_job_cleanup()` puts it unconditionally. So a failure before
  that assignment cleans up a job with a NULL domain. Fixed by taking the per-file reference
  before the first error path. **That is the same bug and the same fix approach as our
  `patches/rk3576/npu/0013`.** The same commit also clears `rjob->tasks` after freeing it
  in `rocket_copy_tasks()`, so the common cleanup path cannot double-free the task array.
  That is a second patch of ours, and one upstream commit covers both.

  **`a85402bf`** (2026-05-24, Muhammad Bilal) NULL dereference and integer overflow in
  `rocket_job_push()`'s `kvmalloc_array()` of the combined in/out BO array. Overlaps the NULL
  guard in the `patches/rocket/085` uAPI work.

  **`9b2dedad`** (2026-06-10, ZhaoJinming) error-path handling in `rocket_job_run()`. It fixes
  a `dma_fence` reference leak and an unsignaled fence returned to the scheduler. It also fixes
  **`pm_runtime_get_sync()` leaking its reference on failure so the NPU cannot suspend**. That
  leak is the mechanism behind the half-started-job runtime-PM pin recorded here as a
  mainline bug affecting the RK3588 too.

  **Consequence for the patch series** (the `patches` repo): at least three patches now have
  upstream equivalents. So a rebase onto a
  current mainline will conflict or silently duplicate. A v7.1 snapshot with
  `081`/`082` already applied cannot be used to test application. It is now also stale as a
  baseline.
  Fetch the real base per kernel version from
  `raw.githubusercontent.com/torvalds/linux/<tag>/drivers/accel/rocket/`.

- **drivercraft/rk3588-clk**: a Rust `no_std` RK3588 CRU clock library (MIT, for bare-metal /
  U-Boot) that sets the NPU clock by **direct CRU register writes** (`npu_set_clk` /
  `npu_get_clk` + `ACLK/HCLK/PCLK_NPU0..2` gates, `pll.rs` / `clksel.rs` / `gate.rs` /
  `constant.rs`). **Not a runtime alternative to the `rocket-clk` patch**: on mainline Linux BL31
  owns the NPU clock via SCMI id 6 + a per-OTP **PVTPLL** (above). This library has **no PVTPLL,
  no voltage handling, and no SCMI-conflict guard**. Direct pokes would fight EL3 and skip the
  f/V coupling the patch depends on. **Useful as** an MIT-licensed, register-level
  cross-reference for the CRU NPU clock tree (PLL config, the clksel mux, gate bits) when
  annotating or extending the clock patch. That is register provenance, not a mechanism to
  adopt. See [perf/clock.md](perf/clock.md).

- **LKML: "[RFC PATCH v4 0/9] accel: rocket: Add RK3568 NPU support"** (Midgy BALON, 2026-06-13,
  v2 at [lkml.iu.edu/2605.3/10672.html](https://lkml.iu.edu/2605.3/10672.html), base v7.1-rc6). An RFC (design
  feedback, not for merge) adding RK3568 to the upstream `rocket` driver via a per-SoC
  `rocket_soc_data` (derive DMA width + core count from match data).
  Project-relevant facts:

  **RK3568 NPU = a single NVDLA-derived core (0.8 TOPS), register layout matches RK3588.**
  That corroborates "same NVDLA IP across RK SoCs", and our `librocketnpu` userspace is
  expected to largely drive it too. End-to-end is blocked on **Mesa/Teflon userspace** (still
  emits RK3588-tuned config) + a HW issue (below), exactly where our richer rocket userspace
  (full dtype matmul, general/DW/int8 conv, LUT activation, on-NPU EW mul vs Teflon's conv+add)
  is an asset.

  **Address width: RK3588 NPU AXI/IOMMU is 40-bit, RK3568 is 32-bit.** So the **4 GB per-fd
  cap on RK3588 is the 32-bit *regcmd address field*, not the bus** (the bus reaches 40-bit),
  as documented in [perf/iova-and-multicore.md](perf/iova-and-multicore.md). RK3568's 32-bit
  DTE needs `GFP_DMA32` page tables (`rockchip,iommu` ops, relying on Simon Xue's
  per-device-ops series).

  **Stock rocket attaches and detaches the IOMMU domain on *every job*** (`iommu_attach_group`
  in `rocket_job_run`, `iommu_detach_group` in `rocket_job_handle_irq`), each toggling the
  rk_iommu stall/reset/paging handshake. **Patch 5 keeps the domain attached across same-context
  jobs.** This is a **per-job dispatch-floor cost on RK3588 too**, a concrete, testable kernel
  lever for our submit-overhead-bound paths (detection 1×1s, KACC's nKt sequential jobs).
  [not-mac-bound.md](perf/not-mac-bound.md).

  **The author reads the NPU's DMA byte counters** ("the NPU reads the full input and weight
  tensors per its DMA counters"), a lead vs our **dead-RK3588-counter** finding (reading the
  `0x2xxx` page hard-locks RK3588). The counters exist + are readable on RK3568, so it is
  possible RK3588's differ by offset/access rather than being absent.
  [hw-byte-counters.md](perf/hw-byte-counters.md).

  **MAC/output stage never completes on RK3568** even on a **byte-exact replay of the vendor
  command list**. That makes it a hardware bring-up issue (PVTPLL/power/NoC de-idle), not a
  regcmd problem. The author asks for pointers, and our deep BL31/PVTPLL/clock RE
  ([clock.md](perf/clock.md)) could help. Patch 3 starts the **PVTPLL compute clock via SCMI**
  (corroborates our PVTPLL finding). Patch 9 wires **vdd_npu as the power-domain
  `domain-supply` (`need_regulator`)** so genpd owns the rail. That is the upstream-idiomatic
  alternative to our driver-held-regulator f/V coupling ([clock.md](perf/clock.md)), relevant
  if we upstream the volt work.

  **OP_ENABLE offset** (from the v2 thread): the per-sub-unit `OPERATION_ENABLE` is `0x_008` on
  RK3588 (what we emit: `0xf008` + per-block `0x1008/0x3008/0x4008…`) vs `0x_00c` on RK3568, a
  regcmd delta for any RK3568 port (not restated in v4's cover letter: verify against Mesa).

- **NetVar1337/linux-rk3576-rocket**: "[PATCH RFC 0/4] accel/rocket: add support for the RK3576"
  (VoidChecksum / Markus Kvam, 2026-06-11, against `torvalds/master`). It carries the
  binding, a clocks-by-name fix, per-SoC match data with PC_DONE polling, and the `rk3576.dtsi`
  core nodes. An independent mainline-targeted implementation of **gahingwoo**'s bring-up
  (below), which it credits throughout. **Compile-tested only: the author has no RK3576
  hardware** and asks for testing reports. It covers what `patches/rk3576/npu/0001`, `0006` and
  `0007` do and nothing else, so it is a subset of the series here. Three things about it are
  still worth knowing:

  **Its central premise is refuted by measurement here.** Patch 3 states that the `PC_DONE`
  bits "are read-only in `INTERRUPT_MASK`, so completion cannot be routed to the GIC". It
  builds a 1 ms hrtimer poll on that premise. That was established about `PC_DONE` and **never
  covered the DPU pair**: `DPU_0`/`DPU_1` (bits 8-9) mask normally and the interrupt reaches the
  GIC [HW sweep, see [chips/rk3576.md](chips/rk3576.md)].

  The poll is a driver choice with a price. Retiring on the DPU bit at a 50 us period instead
  of `PC_DONE` at 500 us took the submit floor from 1065 to 439 us. Their 1 ms period is slower
  again. Two task classes do raise no DPU completion (pooling, and any output element wider
  than one byte), so a poll or a grace still has to survive as the fallback for those.

  **Its patch 2 is the same fix as `patches/rocket/089`** (`clk_bulk_data.id` never set, so
  all four entries resolve to the node's first clock). Independently found, thinner
  rationale, and neither posting has landed.

  **Its device tree has core 1 right**: `0x27708000` with the IOMMU at `0x2770a000`. It is
  cross-checked against the vendor BSP DT, which is what live silicon reads here. The
  `0x27710000` placement recorded in these notes as wrong belongs to the gahingwoo series,
  not this one. With two RK3576 RFCs now in circulation, "the RFC" needs qualifying.

  Its stated open items (the NPU power-domain chain status never asserting after power-off,
  whether the RKNN BIU resets belong to the power domain, and the boot firmware's orphaned
  IOMMU page fault) are the ones `patches/rk3576/npu/0002`-`0005` already address.

- **gahingwoo "Mainlining the RK3576 NPU"** (blog `gahingwoo.github.io/posts/rk3576-npu-mainline/`,
  repo `github.com/gahingwoo/linux-rk3576-npu`: `notes/provenance.md`,
  `notes/rk3576-npu-values.md`, `extract/extract-npu-values.sh`, read at `82cdbe9`, 2026-09-24).

  **Its `board-logs/` is the live record and the `charsiu` notebook is not**: numbered rounds
  through r421 at 2026-09-24, where `charsiu`'s `docs/lab-notebook.md` stops 2026-09-12. Read
  the board logs for anything after that date. Its README carries the current state only, and
  the history it used to carry is in `HISTORY.md`.

  **r419 bears directly on our two-core negative, and it runs at the clock that matters.**
  Four ways to put two jobs on the hardware, 200 repetitions an arm, all four arms back to
  back in one process. The arms are one job/one ioctl (A), two jobs/one ioctl/one fd (B), two
  jobs/two ioctls/one fd (C) and two jobs/two ioctls/two fds (D).

  | k, n | A | B | C | D |
  |---|---|---|---|---|
  | 1024, 256 | 49.77 | 95.05 | 94.74 | 52.43 |
  | 1024, 1024 | 133.64 | 263.05 | 263.30 | 135.66 |
  | 2048, 2048 | 489.63 | 943.00 | 937.84 | 471.88 |

  One fd serializes two jobs whatever the ioctl count. **Two fds run them side by side and the
  pair costs what one job costs**, at every shape where the arms separate. So on their board
  the second core is real and free, and they report no corruption from it. **Their board is a
  ROCK 4D at 594 MHz**, the cell where the undervolt table in
  [chips/rk3576.md](chips/rk3576.md) says two-core corruption vanishes, while ours is at
  786 MHz. That makes r419 a second, independent reason to repeat the two-core run at 594 MHz before
  treating either claim below as a property of the silicon. The claims are "core 1 buys nothing" and
  "two jobs in flight compute wrong answers".

  Their round also carries a methodology warning we hold in our own form. The first reading of
  those arms, taken at k=64 n=32 where the arithmetic is nothing, said the exact opposite
  (B beating D by 2.5x) and was noise.

  A sibling-SoC (RK3576, 2-core, **16 CBUF banks** vs our 12) mainline-`rocket` bring-up.
  **Methodology** worth borrowing: capture the vendor command stream by building a 1-conv ONNX
  and converting it with `rknn-toolkit2`. Then walk the `.rknn` for the 64-bit command words and
  decode per unit (an alternative to our Mesa-Teflon capture that can expose ops Teflon never
  emits, e.g. **pooling**, for the on-NPU PPU work). Also worth borrowing is
  `extract-npu-values.sh`, which auto-derives the platform constants (power-domain / clock /
  reset IDs, GRF base, PVTPLL, OPP table, per-core MMIO bases, IRQs, QoS) by grepping the kernel
  DT-bindings + TF-A BL31. It is adaptable to RK3588 (s/rk3576/rk3588/) to auto-document our
  clock/volt patch provenance.

  **Cross-confirms our findings** (all independently). (1) **IOMMU attach-once /
  detach-on-power-down, not per-job** == our keep-attached patch
  ([iova-and-multicore.md](perf/iova-and-multicore.md)). (2) **Ping-pong producer/consumer
  register groups** (`S_POINTER`): the executer reads the *consumer* group, and misalignment
  -> stale geometry -> zero/garbage output. The groups need per-job re-init, and they are the
  **mechanism that decides** where a delta regcmd task lands
  ([regcmd-task-model.md](encodings/regcmd-task-model.md)). (3) **Requant is a right-shift**
  whose magnitude is load-bearing (vendor 26-bit vs a wrong 14-bit -> saturation to
  black/white), which corroborates our per-scale QNNPACK shift in the conv int8-out path
  ([out-cvt-converter.md](encodings/out-cvt-converter.md), bit-exact vs Teflon).

  (4) **Per-channel zero-point correction in a weight-buffer tail** (8-OC groups, 64 B =
  8×32-bit + 8×16-bit + 8×16-bit, the 16-bit holding `128 − weight_zp`, term
  `(128−wt_zp)·input_sum`) == our Option-D uint8 recenter + box-sum. (5) The
  **`dt_wr`/`dt_rd`/`wt_rd` byte counters are readable on RK3576**, exactly as our
  [hw-byte-counters.md](perf/hw-byte-counters.md) table predicts (rk3576 config wires
  `0x2234/38/3c`, while rk3588 nulls them and that page hard-locks). That **does not reopen our
  RK3588 negative**: it confirms the sibling asymmetry. Net: strong independent validation of
  the shared NVDLA-derived IP, plus two transferable scripts. Little is usable *as-is* (RK3576
  register map is shifted/re-packed, different clock/power tree).

  The load-bearing documents in it are the CNA and CORE/DPU maps, the closed-form `predict.py`,
  `vendor_regcmd_full.txt`, `FINDINGS-FLOATSURFACE.md` and `MATMUL-PIPELINE-ANALYSIS.md`.
  `vendor_regcmd_full.txt` is a **complete 139-entry vendor register program** (CNA + CORE +
  DPU + RDMA) that an RK3576 emitter can be diffed against off-device.
  `MATMUL-PIPELINE-ANALYSIS.md` confirms RK3576 matmul is the same CNA->CSC->CMAC pipeline in
  FC mode (no separate GEMM unit). It also reports the per-power-session "cold-start
  consume-arm" wall it read as a hardware arm. Reproducing that on our own encoder and submit
  path placed it in the driver instead (see [chips/rk3576-regcmd.md](chips/rk3576-regcmd.md)).

  The repo has since grown well past the register maps and is worth re-reading as a whole. It
  now holds a 28-patch kernel series and a Mesa fork carrying an RK3576 Teflon conv2d. It also
  holds a `replay/` capture-and-replay harness that runs the same regcmd through both `rknpu`
  and `rocket`. And `FINDINGS.md` is a 2900-line chronological RE log that keeps its own
  reversed verdicts.

  `CHAINED-CMAC-STOPPING-POINT.md` is the falsification-ledger writeup of the same
  per-power-session wall, parked 2026-07-10. Its conclusion is that the consume-arm is
  internal cold-start sequencer state reachable only from vendor RTL. **They independently
  found and fixed the 16-bit `pc_task_number_bits`** (`WRITEL-AUDIT.md`, and patch 0028 writes
  `(0x7 << 16) | task_count`), so that half is common ground. Both stacks run n-task jobs in
  one hardware kick: ours via `DRM_ROCKET_JOB_BATCHED` (`patches/rk3576/npu/0015`-`0016`),
  and theirs in `charsiu` at 32 chained tasks per job. `FINDINGS.md`'s "even task 0 computes
  nothing when task_number=29" describes that log's own submit path, not a hardware bound.

  One structural blind spot in the parked ledger is worth knowing when reading it. Every
  experiment in it varies the job that comes out empty, never the job before it. So the
  wide-output poisoning, a property of the preceding submit, is invisible to it however
  exhaustive it is. `charsiu`'s harness does not share the blind spot: its bisects judge a
  following job run in a separate process (see its entry below). Blog moved to
  `blog.gahingwoo.com/posts/rk3576-npu-mainline/`. `charsiu` chains multi-task jobs and
  places the OUT_CVT triple at `0x40ac/0x40b0/0x40b4` itself.

  Their upstream series is at **v14, posted 2026-09-24** with 15 patches against next-20260914
  ([v14](https://lore.kernel.org/all/20260924102135.92217-1-gahing@gahingwoo.com/)). At v13
  (2026-09-15) it carried an Acked-by from Conor Dooley on both dt-bindings and a Reviewed-by
  from Abel Vesa on both pmdomain patches. It also carried a Tested-by from Igor Paunovic on
  each of the three reset-race patches
  ([v13](https://lore.kernel.org/all/20260915104328.45901-1-gahing@gahingwoo.com/)). The v13
  posting exists partly to carry a retraction into its commit messages. See the withdrawn
  induced-reset claim in the `charsiu` entry below before reading an older posting's case for
  the reset-race patches.

  Ulf Hansson has offered to take v14's three pmdomain patches (7, 9 and 10) through his tree.
  The series carries review tags from Krzysztof Kozlowski, Conor Dooley, Heiko Stuebner and
  Abel Vesa. Patch 14/15 assigns the NPU 594 MHz at the SoC level, citing Rockchip's table as
  asking 725 mV of the 500 and 600 MHz steps. That is the table's default row: bins `L0`-`L1`
  ask 737.5 mV, which still leaves 594 MHz at 750 mV inside it at every bin. The v12 and v13
  postings give the two-core fault as "13 to 20 wrong rows", which the v14 posting withdraws as
  unsourced. The board log's figure is 11 to 25 wrong words a pass.

  **Board log r420 places the power-on settle requirement in the domain.** With the delay at 0,
  the first NPU domain power-on takes an asynchronous SError. Holding the rail up with
  `regulator-always-on` does not prevent it, which matches what Rockchip said on the v13 thread
  ([chips/rk3576.md](chips/rk3576.md), under the power-on settle). **Board log r421** boots the
  v14 series as sent, and both cores probe. Every numerics check matches the previous kernel to
  the last digit, perplexity included. Decode reads 2.8% lower from one unpinned reading per
  kernel, which the log declines to call a regression.

  **v7 on 2026-08-12**, 10 patches, `accel/rocket: RK3576 NPU (RKNN) enablement`
  ([cover](https://patchwork.kernel.org/project/linux-rockchip/cover/20260812094106.1391698-1-gahing@gahingwoo.com/)),
  is the revision the reading below was done against.
  It retracts the "the completion interrupt never reaches the GIC" premise its v3-v6 carried,
  and attributes the whole of it to the `PC_TASK_CON` width. So polling, the hrtimer and every
  alternative completion path are gone, and the RK3576 retires on the DPU interrupt like the
  RK3588. The open question that reopens for us is in
  [chips/rk3576-regcmd.md](chips/rk3576-regcmd.md).

  Two of their patches land on ours: `01/10` is `patches/rocket/090` hunk for hunk, and
  `08/10`'s `PC_TASK_CON` word is `0x70001`. That is the same value `rk3576/npu/0008` writes by
  shifting the triple. Their evidence is one convolution submitted three times, and their
  Teflon userspace is conv2d-only. So neither the pooling programs nor the wide-output writers
  that raise no DPU completion at all are reachable from it.

- **gahingwoo `charsiu`** (`github.com/gahingwoo/charsiu`, GPL-2.0-or-later, first commit
  2026-08-14). It is an open LLM runtime for the RK3576 on
  mainline `rocket`, the same architectural bet as `rocket-userspace` + `ggml-rocket` on the
  sibling part. It names both plus these notes as its stated starting point. Every number
  below is theirs (ROCK 4D, their v7-lineage kernel, 2026-08-14/15) and none is reproduced on
  our board.

  **The load-bearing instrument is `tools/rkllm_regcmd.py`**: a vendor `.rkllm` carries the
  register-command streams the closed stack submits. The script reads the whole dispatch
  plan out of one offline, with no board and no vendor runtime.

  Its first reading (`docs/vendor-dispatch.md`, Llama-3.2-1B-Instruct-w4a16) finds 13,224
  streams over 1,061 distinct shapes, 8,808 convolutions + 4,416 DPU-only. Precision goes by
  role. Int4 projections are **batched** at M = 16 to 80 (2,816 of 3,328, 85%), and fp16
  attention runs at M=32-48 against 128 precompiled KV-length buckets (one per 32 tokens of
  context). The LM head is int8, as forty 2048x8160 pieces. Every projection is split across
  the two cores by output channel.

  The FFN down-projection is split on both axes at per-piece K=4096, inside the 4608 slice
  bound measured here. That is the vendor's own route around a K that does not fit one slice
  (the shape class our matmul entry refuses at K>4608). Every int4 dispatch is K=2048
  (2,688, 81%) or K=4096 (640, 19%): the vendor never hands one dispatch a K of 1024.

  **Two figures this entry used to carry are withdrawn at the source, and the reason reaches
  our own reader runs.** Do not quote "21,532 streams / 12,724 DPU-only" or "int4 projections
  all at M=1 (3,752 dispatches)". Both were defects in `tools/rkllm_regcmd.py`, found and
  fixed by its author (`c179704` 2026-08-28, `80bd4d2` 2026-09-01).

  **The M histogram read the wrong register.** It took M from `0x102c`, the row count,
  where the correct register is `0x1034`, the pixel count. An int4 projection is emitted as a
  **one-row image M pixels wide**, so its row count is 1 whatever M is. By contrast, fp16
  attention is emitted as an M-row image whose rows equal its pixels in 4,940 of 4,940
  streams. The wrong reading was therefore correct on everything it could be checked against
  and returned 1 for every int4 op in the file. The vendor batches, and 80 is both its widest
  and its most common width.

  **The stream census over-counted DPU-only runs threefold.** Target `0x0401` was missing
  from the reader's table and an unknown target ENDS a run, cutting each affected op into
  three. That takes 12,724 DPU-only streams to 4,416.

  It also takes a taxonomy with it. The "six DPU-only program kinds", grouped by `0x4010` and
  `0x4050`, were the **fragments** carrying whichever registers fell on each side of a cut.
  Re-grouped there is exactly one kind, both registers reading 0. Their file keeps the old
  table marked as a record of the bug rather than of the model. No replacement taxonomy is
  offered there, and none is to be assumed here.

  **What that does to the two model reads recorded below.** Both were run here on
  2026-08-18, against the tool as it stood then, so both predate either fix. Sorting our
  derived claims by whether the defects can reach them:

  **Unaffected.** Everything read from the convolution streams' geometry and weight
  registers. The conv count is identical either side of the `0x0401` fix (8,808 both
  ways), so conv parsing never fragmented. That covers the per-dispatch K bound of
  **4,096 in greedy chunks** across three models, the oc-halving, and `ffn_down`'s
  `_C_secondary` per-piece scales.

  **Unaffected.** The KV-bucketed attention population, because fp16 streams are the case
  where the row and pixel counts agree. The M=32-128 program counts matching across the two
  Qwen3 files, and hence "the KV-bucket population is converter policy, not model shape",
  stand.

  **Withdrawn.** "M again in {1, 32, 64, 96, 128} with M=1 dominating (14,280)" is the
  row-count artifact in its pure form. It says nothing about how the vendor batches those
  models.

  **Needs re-deriving.** The DPU-only stream counts, and with them the argument that the
  DPU-only share collapse is converter policy rather than architecture. That argument's
  quantitative pillar was Llama's dominant per-projection EW kind at 8,268 streams, and
  the fix removes 8,308 Llama DPU-only streams. The population and the correction are the
  same size to within half a percent, so it was very largely the bug.

  A residual collapse survives arithmetically, 33% against 18-21%, rather than the 59%
  against 18-21% claimed. But it has not been re-read, and the "one kind, both registers 0"
  regrouping means the kind-based half of the argument no longer has terms. Re-running the
  fixed reader needs the two Qwen3 `.rkllm` files re-fetched (725 MB and 1.6 GB). Neither is
  on disk here.

  **Where it lands on our findings:** M=1/2/3 exact through the open driver (seven 1x1 convs
  at projection shapes) independently corroborates the no-M-constraint fact in
  `rocket_matmul_rk3576.c`. Its 32-task chained jobs at ~26.3 us/task marginal (~172
  us/submit removed) corroborate the one-kick mechanism `patches/rk3576/npu/0015`-`0016` ship.
  And their fitted cost `us/task = 26.3 + weight_MB * 84.3` (11.9 GB/s, M nearly free, a
  second core ~5% worse at these shapes) was the weight-fetch-bound reading the platform
  envelope here predicts.

  **They have since refuted that reading themselves, and the replacement is a fixed cost**
  (2026-08-31). A 9.4 GB/s average over stages running 6.67 to 15.60 GB/s is not a roof. A
  roof does not have a 2.3x spread across shapes, and a fixed cost does. And `gate+up` moves
  253.8 MB in 18.59 ms, 13.65 GB/s with its own dispatch still inside it.

  Refitted over five stages on the busier core, the cost is
  `us a call = 128.7 + 36.8*tasks + 110.0*MB`, RMS 9 us on a 540 mean and inside 2.4% at every
  stage. Per token that is 11.5 ms of call, 7.4 ms of task and 29.1 ms of weights. So **39% of
  the hardware path is dispatch, not bandwidth** [their HW sweep, ROCK 4D].

  Two independent measurements agree that the DRAM is not the constraint on that board. Eight
  reader threads pulling 11.93 GB/s alongside a decode move it by 0.1%, and the decode moves
  them by 0.1%. Prefer this fit to the 84.3 line above. Both are theirs, and the later one is
  fitted on more shapes.

  **A units trap rides with it, and it is one we can make too.** One of their calls issues one
  submit PER CORE and waits on both. So every "us a submit" that tree had printed was half a
  call's latency, and the fixed term they published as 112 us is really 224. The product was
  right while the per-unit number was out by two. That is the failure mode where a total
  looks checked and the unit is not.

  They also found two submit-shape knobs that were read nowhere (`CHARSIU_NPU_MAXTASK`) or
  read only for an unrelated threshold (`CHARSIU_NPU_NOCHAIN`). So every round that set them
  measured its own baseline twice, the same positive-control gap our own register sweeps have
  hit.

  **What it claims that is unmeasured here:** NPU decode is viable on this part, and the
  vendor ships M=1 decode (~13 tok/s on that board and model). Their arithmetic projects
  11.8 tok/s int8 / 22.7 int4 for Llama-3.2-1B. Their stated deciding measurement is one
  projection, NPU against four A72 cores, at M=1 and M=32. That challenges the decode-on-CPU
  default this stack inherited from the RK3588 [their measurement + projection, untested here].

  **Their open defects, and what bears on them:** w4a16/int4 does not compute. Their probes
  fit the output as `((int16)fp16bits(w) * (int16)fp16bits(a)) >> 16`, 18/18 measured points
  exact, i.e. the fp16 bit patterns multiplied as signed integers. That is consistent with a
  partially-set float mode, which on this part is three registers moving together
  ([chips/rk3576-regcmd.md](chips/rk3576-regcmd.md)) [expected, unverified against their
  stream]. And a w4a16 job carrying the vendor's values in RDMA `0x5034`/`0x5044` leaves the
  next job timing out. That is a next-submit hazard with a different signature from the
  wide-output poisoning, recorded beside it in [chips/rk3576.md](chips/rk3576.md).

  **The reader run here on a second model** (Qwen3-0.6B-Base-rk3576-w4a16-grq v1.2.3,
  725 MB, HF `MichaelAndrewFischer`, 2026-08-18) separates model shape from runtime policy.
  It finds 19,380 convolution streams (its DPU-only count of 4,160, and the 23,540 total, are
  pre-fix readings). The M=32-128 program counts are nearly identical to Llama-3.2-1B's
  (1108/820/732/676 against 1108/856/728/672). So the KV-bucketed attention-program
  population is converter policy, not model shape. Qwen3's FFN down-projection (K=3072,
  inside the 4608 slice bound measured here) dispatches whole with only the two-core
  output-channel split. The vendor K-splits only when forced past its slice bound.

  **A third model read settles the split policy and the combine mechanism**
  (Qwen3-1.7B-Base-rk3576-w4a16-grq v1.2.3, 1.6 GB, same HF author, read 2026-08-18,
  offline, both files re-fetched): 22,740 convolution streams (its DPU-only count of 6,064,
  and the 28,804 total, are pre-fix readings). The 1.7B's FFN-down (K=6144) never dispatches
  whole: every instance is a 4096-piece plus a 2048-piece, oc-halved
  (`blk.N.ffn_down.weight_rkllm_spilt_0/1` in the file's own tensor names). So the vendor's
  per-dispatch K bound is exactly **4096, greedy chunks**. That is the one rule that fits
  Llama-1B's 8192 = 4096+4096, this 6144 = 4096+2048 (not 3072+3072), and the 0.6B's 3072
  whole.

  **The combine is ON-NPU: a dedicated DPU-only elementwise program, one per split pair.**
  At M=1 every `[4096-piece][2048-piece]` pair is followed by exactly one (1344 of 1344).
  Prefill sites carry the same program in pixel-bucket variants. Its primary operand comes
  from memory (`0x400c = 5` where a conv carries `0x40000004`). Its second comes through
  DPU_RDMA (`0x5xxx` words live, including the exact `0x5034 = 4000004c` /
  `0x5044 = 000280a1` values of the w4a16 next-job hazard in
  [chips/rk3576.md](chips/rk3576.md)). Both are at the pair's output geometry (oc-half x M
  pixels).

  Its `0x4010` input width code is **5**, the 32-bit code the coefficient-A fp32 readback
  anchored, where the attention-interior EW program carries 2. So the partials are float,
  separately scaled (`ffn_down` is the only projection with a `_C_secondary` coefficient
  group, one multiplier set per K-piece). And nothing in the mechanism touches int32: no
  integer GEMM in any of the three read models exceeds K=4096 (the int8 LM head is K=2048,
  whole).

  The split pieces' conv programs are register-identical to never-split projections outside
  pure geometry. The combine is invisible to a per-register diff and was read from the
  program sequence and counts. The program type is not exclusively the combine: ~16 instances
  in each file follow a `2176x32` fp16 op. So a count test alone would misattribute, and
  adjacency separated the uses.

  **Both former decode frontiers are read.** The weight-bits-0 streams are chained
  no-weight-fetch delta task programs. A prefill M bucket is pixel-chunked (0.6B:
  53+53+22 = 128, 1.7B: 40+40+16 = 96), and the last chunk restates neither the weight
  registers nor the full geometry. The reader's bits formula misparses those chunks, 228 on
  the 0.6B and 340 on the 1.7B. Every one is adjacent to same-`ic` full programs whose pixel
  counts it completes (the 1.7B's 672-count "bits 4096" class is the same thing with a weight
  fetch: trailing K-pieces as delta programs).

  **The DPU-only share collapse argued from these reads does not survive the reader fix and
  is not to be carried forward.** It read 59% of Llama's streams against 18-21% on both Qwen3
  files, and attributed the difference to converter policy. That rested on Llama's dominant
  per-projection EW kind (`a0000002`/`00023333`, 8268 streams) having no counterpart in either
  file. That kind was the fragment population the missing `0x0401` target manufactured, to
  within half a percent of the 8,308 streams the fix removes. And the regrouped census has one
  DPU-only kind with both registers reading 0, so the contrast it was built on has no terms
  left.

  What is untouched by the defect and still stands: the attention-interior EW population is
  identical across the two Qwen3 files (4120+24 of kind `0x4010=40000002`, KV-bucketed). And
  neither Qwen3 file holds any norm tensor, so the decode path's norm/residual/activation work
  is off-NPU in converter v1.2.3 whatever the stream census says. Qwen3 needs more norm work
  than Llama (QK-norm), which is why architecture was a poor explanation for the
  disappearance. The Llama file's converter version is unstamped, so even the surviving half
  is argued from that direction rather than read off a version field.

  **Re-surveyed 2026-09-18 at `dev` 524e10a.** Branches are now `dev` and `stable`. `main`
  is stale and its README still prints withdrawn figures. `docs/lab-notebook.md` stops
  2026-09-12, and rounds r393 onward live in `board-logs/` in the driver repo, not this one.
  So a question the notebook does not answer is not necessarily unanswered.

  **Their headline against the vendor** (one board, ROCK 4D, `librkllmrt` 1.3.0 run on that
  same board rather than quoted, Llama-3.2-1B, NPU 594 MHz both sides, CPU pinned): decode
  **1.39x** the vendor at 594 MHz and **1.46x** at 786, and the spread between those two is
  itself the finding -- their decode scales with the NPU clock and the vendor's does not, so
  the multiple is a property of the condition rather than of either runtime. Prefill is ahead
  at every prompt length from 27 to 602 of their own tokens and **level at 852**. Two things
  make that table quotable and are worth copying: the vendor's chat template costs a constant
  33 tokens at every length, so each row is the same input text; and the margin a row has to
  clear is **measured, not assumed** -- twenty readings of one arm at 302 tokens on one boot,
  changing nothing, spanned 2.5% of their median, so 852 at +2.1% is called level rather than
  a win. **The vendor's decode is inert to the NPU clock**: 32.4% more clock buys it -0.9%
  [their HW sweep].

  **Why decode-on-NPU there does not contradict the decode-on-CPU rule this stack inherited.**
  M=1 is not efficient on the RK3576 either. At K=1024 int8 their per-output-channel cost runs
  0.111 us at m=1 and 0.202 at m=80 -- eighty times the rows for 1.8 times the cost, because
  the weights are read once for the batch -- which is 0.0025 us a channel a row at m=80
  against 0.111 at m=1, **a factor of 44**. Their decode win comes from removing dispatch
  cost, not from the part being good at GEMV, and it is bounded by a host side with one A72
  cluster. The RK3588 has four A76s. So "decode belongs on the CPU" is a statement about the
  host, and the reason it fails there is not a statement about the NPU. Consistent with the
  platform envelope in [chips/rk3576.md](chips/rk3576.md).

  **The noise arm is the experimental control this stack does not have.** Their quantisation
  evidence carries a matched-magnitude noise arm inside the harness rather than beside it:
  unstructured error at the same per-tensor magnitude costs +60.00% perplexity where the
  vendor's actual quantisation costs +13.32%, so the vendor row sits four and a half times
  further from noise than from the reference. The arm earns its place by having caught a
  defect: an earlier reconstruction scored ppl 1701 at 18.3% weight error where Gaussian noise
  of the same magnitude scored 32.10, and a reconstruction that is 53x worse than noise of its
  own size is a broken reconstruction, not a quantisation result. **The 1701/32.10 pair is a
  fault signature, not a finding about quantisation, and should not be quoted as one.** The
  durable lesson is the one they draw from the other side: a genuine reduction in Frobenius
  weight error changes quality in the opposite direction on two models of eight, so weight
  error cannot predict even the SIGN of a quality change.

  **Their requant derivation is an independent cross-check on our per-column ramp.**
  `src/job.c` derives a 15-bit multiplier with an implicit leading 1 and takes the shift from
  the float's exponent bits, rounds to nearest by incrementing on the bit below the field and
  renormalises on carry; the output stage is `clamp(max(requant, 0) + offset, -128, 127)`,
  with the floor at zero applied BEFORE the offset. That floor would cost the negative half of
  the range, and a "lift" recovers it exactly -- `+128/mult` folded into coefficient A, with
  `offset = out_zp - 0x80` -- since `max(rq + 128, 0) + (out_zp - 128) = rq + out_zp` for every
  `rq >= -128`. **Their DPU requant registers are single, not per-channel**: `0x40ac` offset,
  `0x40b0` scale, `0x40b4` shift, one triple per program, and per-channel behaviour comes only
  from the A/B/C coefficient table and the fp16 scale table. That matches what our per-column
  entry has to do and is a second derivation of the field widths our
  `worst_rel_err` ramp is bounded by.

  **Their output-width bundle is six registers, and it reconciles two of our register-level
  results at once.** `CHARSIU_WIDE8` forces the w4a16 output stage onto an int8 weight job one
  register at a time: bit0 `0x4010`, bit1 `0x4030` low, bit2 `0x4038`, bit3 `0x4044`, bit4
  `0x4050`, bit5 the `0x40ac`/`0x40b0`/`0x40b4` identity requant. Swept, **bit0 and bit4 have
  to be set together -- either alone wedges the NPU** -- while bit3 and bit5 are not needed at
  all (`0x37` and `0x1f` both give the exact four-byte accumulator). They ship a four-byte
  output on every dispatch, across processes, and report no poisoning. Two consequences here:
  our int32 writer moves `0x4010`'s width field without `0x4050`, which is precisely the
  partial bundle that wedges for them; and our fp16 poisoning condition, decoded by joint
  sweep to `0x4038[4]` and `0x4050[17]` together and neither alone, is two members of this same
  bundle. Recorded against the hazard in [chips/rk3576.md](chips/rk3576.md). They also report
  the width has **exactly two values, 1 and 4 -- there is no 2** -- so nothing in this bundle
  selects an fp16 output width.

  **Their CBUF window is six registers derived from a start bank, and two of them are absent
  from our tree.** Read off a 2026-09-04 census of the vendor's Llama-3.2-1B `.rkllm`, where
  every op appears twice and the two copies differ in exactly six registers as a function of
  the start bank S: `0x1018` (low byte `4 + S`), `0x1038` (end bank, `| 0x100` for S > 0),
  `0x103c` (low half `S*0x400`), `0x1040` (`(S*0x400 + 0x1000) << 16 | S*0x400`), `0x2818`
  (`S*0x400 << 16`) and `0x2820` (`S*8`). So the CBUF is two mirror-image windows of 7 banks
  inside 14, and their runtime gives each core its own (`charsiu_job.cbuf_window = di`).
  **`0x2818` and `0x2820` appear nowhere in our tree, neither emitted nor documented**, and our
  `ROCKET_RK3576_CBUF_BIAS` moves two of the six, neither of them the end bank -- so it moves a
  window's base while leaving its extent where it was. Recorded against our no-partition
  negative in [chips/rk3576.md](chips/rk3576.md).

  **They attribute two-cores-in-flight corruption to an undervolt, and our board sits at the
  operating point they name as unsafe.** With both cores in flight, the job on the second core
  wrote one word of a row wrong -- the right value plus 1024, a few bits around bit 10 of the
  accumulator, one row in a few thousand, always the same array position. Four DTBs on one
  board, kernel and binary held, the element probe at width 24 and KMAX 1024, four passes of
  5400 rows each:

  | NPU clock | `vdd_npu_s0` | wrong words a pass |
  |---|---|---|
  | 786 MHz | 750 mV | 11 to 25 |
  | 594 MHz | 750 mV | 0 0 0 0 |
  | 786 MHz | 800 mV | 0 0 0 0 |
  | 786 MHz | 850 mV | 0 0 0 0 |

  Mainline runs the NPU at 786.43 MHz and 750 mV, where the vendor asks 725-800 mV of its
  800 MHz step by chip bin (table in [chips/rk3576.md](chips/rk3576.md)). Their
  `overlap_safe()` serialises unless the rail (from sysfs) and the clock (from debugfs) sit
  inside that table at its lowest bin. At 800 mV they overlap batched calls, and nine
  models match the serialised arm with TTFT 845/1183/3963/2891 ms against 1037/1565/5073/3604
  [their HW sweep, ROCK 4D, 2026-09-04]. **The H96 MAX M9 here sits at 786 MHz / 750 mV**
  [live 2026-09-25], outside the table unless it bins at `L5` or above. Their
  fault is far milder than ours (tens of words a pass against 72-76% of a victim's calls), so
  this is not obviously one mechanism. It is still an untested competing explanation for a
  negative we treat as settled, and the test is cheap.

  **Directly liftable, and none of it needs their hardware.** `tools/check_consistency.py`
  states evidence-pack rules as CONDITIONS rather than verdicts, requires supersession to be
  MUTUAL, and self-tests each rule against a document that violates it. `src/sentinel.h` puts
  one poison word per ROW rather than per cell, which is cheap enough to ship hot and still
  distinguishes wrote-nothing from wrote-all from stopped-midway. Their `/dev/cpu_dma_latency`
  hold buys +24% decode by forbidding deep idle, which is worth it because `rk3576.dtsi` gives
  `CPU_SLEEP` a 250 us exit latency. Their fence waits spin then sleep, exploiting that
  `timeout_ns` is an ABSOLUTE deadline so a value of 1 makes `PREP_BO` a non-blocking poll, and
  the poll wins 81% of waits at a 200 us budget. And `tests/mel_cross.py` sets its tolerance by
  MEASUREMENT -- run the reference in f64 and f32 and fail only above
  `max(TOL, 2*(f32 - f64))` -- rather than by a constant.

  **What this stack still has that they do not**, and what is worth giving back: they have no
  rotation work at all (a full-tree grep for Hadamard, QuaRot or SpinQuant returns only RoPE),
  they found the diagonal AWQ variant and ship it off by default, and their int4 costs +102%
  perplexity, so our Hadamard-plus-per-column-requant results land directly on what they are
  pushing. They emit the DPU LUT block `0x4100`-`0x4194` as all zeros with no comment, where we
  have the input map and a working activation. They have no pooling or PPU work.

  **One claim of theirs is withdrawn by the person who measured it, and we should not inherit
  it.** Their README used to report that RK3588 testing had reproduced, once in 102 induced
  resets, an inference that signalled success while its output buffer was never written, on
  the unpatched arm only -- a differential silent-corruption signal. On 2026-09-12 he found
  the error in his own reports: the script kept every inference's scorer output per round and
  never aggregated it, so the summaries scored only the one inference after the forced
  autosuspend. Aggregated, that constant-`0x80` buffer is in nearly every run, on every arm,
  on all three dates. It is not a differential signal; it is **what a job cancelled by a reset
  looks like from userspace**, whatever made it miss its deadline, and `drm_sched_start()`
  completing detached jobs with `-ECANCELED` is the mechanism. Their v13 series carries the
  correction into the commit messages, and the case for those three patches is the source
  analysis rather than the protocol. Nothing in these notes ever carried the withdrawn version;
  this entry exists so it is not picked up from their README or from an older posting.

- **gahingwoo `kiln`** (`github.com/gahingwoo/kiln`, GPL-2.0). The vendor RKLLM/RKNN stack run
  on a mainline kernel (7.1.3+): the GPL `rknpu` driver built out of tree plus a small kernel
  patch set, an installer, and an OpenAI-compatible server. **Its ten kernel patches apply to
  7.1.7 in series order with zero fuzz under `git apply --check`** (read 2026-08-26 against a
  prepared 7.1.7 tree), touching only `pm-domains.c`, `rockchip-iommu.c`, `rk3576.dtsi` and
  `rk3576-rock-4d.dts`, so the stated 7.1.3 target is not a ceiling there. Applied out of
  series order against an unpatched tree, five report offsets and two report fuzz, which is a
  property of that test and not of the series. `charsiu`'s measuring stick, and
  the live-capture harness (`capture/rknpu-regcmd-dump.patch`) that complements the offline
  `.rkllm` reader. Two of its kernel-side facts land here, read against our own DTS
  (2026-08-18): **(1) The "two IOMMUs / four MMU banks, mainline drives one" claim is about
  the vendor `rknpu` aggregate node**; one device listing both iommus, the second left
  unattached, so the second core reads IOVAs as physical addresses. It does not transfer to
  `rocket` as-is: the RK3576 NPU is two iommu instances of two banks each (`rknn_mmu_0` at
  `0x27702000`+`0x27702100`, `rknn_mmu_1` at `0x2770a000`+`0x2770a100`; our
  `patches/rk3576/npu/0007` DTS), each core device carries its own, and mainline
  `rockchip-iommu` iterates every bank of an attached instance, so single-core `rocket`
  already manages both banks of core 0, and this mechanism is not a candidate for the
  two-jobs-in-flight corruption there. **(2) Their `0007`/`0008` iommu patches (authored by
  Jiaxing Hu, the RK3576 RFC author) name a mechanism whose symptom we log**: a bank left by
  firmware with `PAGE_FAULT_ACTIVE & !STALL_ACTIVE & IDLE` silently drops
  `CMD_ENABLE_STALL` and delays the other banks past the poll timeout, the recurring
  `rk_iommu ... Enable stall request timed out` on the H96 during long probe runs matches
  that failure path exactly [symptom match; the orphaned-fault status bits unverified on our
  board]. Mainline fixes the same condition from v7.3-rc1 [source-confirmed, from the commit].
  `b10d5920cafa` acknowledges the stale fault with `CMD_PAGE_FAULT_DONE` before enabling
  stall, where `0007`/`0008` skip the bank. It carries no `Fixes:` or stable tag.

- **gahingwoo `mesa-rk3576`** (`github.com/gahingwoo/mesa-rk3576`, branches `rk3576` and
  `rk3576-clean`, on Mesa 25.3.0; read at
  `1f0de93`, 2026-08-31). Their Teflon/`rocket` RK3576 driver as a maintained Mesa fork: conv
  paths, depthwise in 64-channel groups, CBUF allocation/reuse, a `ROCKET_REG_SET`
  late-override knob. A second independent RK3576 encoder to diff against, for the RK3576 what
  Mesa's `rkt_regcmd.c` is for the RK3588.

  **Its recent work is a field-level decode of DPU `0x4050`, and it corroborates our emitter
  on one axis while diverging on another.** Against a corpus of 94 vendor `.rknn` compiled
  offline (oc 4-1024, ic 3-1024, 1x1 to 224x224, k 1/3/5, stride 1 and 2):
  - **Depthwise agrees with us, independently.** Their word is
    `0x00013033 | ((DIV_ROUND_UP(oc,16) - 1) & 3) << 8`, and `npu_regcmd_rk3576.c`'s
    `r76_dw_bs_cfg()` computes the same base constant and the same field. Their stated reason
    (depthwise pads output channels to 64, so the last bank holds one to four 16-channel
    atoms and the field is that count minus one) is the derivation behind our own comment, and
    it holds on 13 of 13 depthwise models, three of them predicted rather than fitted. Upstream
    Mesa hardcodes `0x00013133`, the 32-channel case, so every depthwise outside oc 32/96/160
    is misconfigured there -- the same defect our emitter had and fixed.
  - **The direct (non-depthwise) word is oc-dependent for them and unconditional for us.**
    They select `0x80011111` when `DIV_ROUND_UP(oc, 16)` is **even** and `0x80011011` when it
    is **odd**; the two differ in bit 8, which is inside the field the depthwise path uses for
    its atom count. `R76_DPU_BS_CFG_DIRECT` here is `0x80011111` unconditionally. They also
    report that an `oc % 32` form -- fitted on multiples of 16 only, which is every count the
    vendor sweep covered -- is wrong at **oc 56**, the one shape that times out on their board
    under it, and that the vendor toolkit picks the parity answer at that count. Our gates may
    simply never have emitted an odd atom count on the direct path. Open here.
  - **Their single-field sweep of the five unexplained `0x4050` fields** (round 260, at oc 128
    and oc 88, each field moved to upstream's value alone): `RGP_CNTER` wrong output with no
    hang, `RESERVED_0` correct output but the job times out, `SIZE_E_0` and `OW_SRC` both wrong
    and hanging, `SIZE_E_1` inert at both shapes -- later resolved as the **depthwise flag**,
    0 in all 81 regular models and 1 in all 13 depthwise ones. Four of five load-bearing.
    Read this before turning `0x4050` on as part of a wide-output bundle: it is not a
    single-purpose register, and the width bits sit in a word whose other fields encode the
    program's own geometry.

- **Chaoyi Chen (Rockchip), `PC_TASK_CON` bit assignment**,
  at [lore.kernel.org](https://lore.kernel.org/all/4f300b78-d96d-4d98-8819-dc292b0c9b97@rock-chips.com/).
  A vendor engineer's reply on `linux-rockchip` giving the RK3576 field layout, including the
  name of the control the RK3588 description marks reserved: `BIT(18) task_last_layer_clear`.
  The only authoritative statement of that register we have; everything else about it here was
  read off the silicon. See [chips/rk3576-regcmd.md](chips/rk3576-regcmd.md).

## Userspace stacks we learned from

- **johanvdb/librocket**: FOSS userspace fp16 matmul (as 1×1 conv) on mainline
  `rocket`, "for GGML projects." It combined Jasbir Matharu's `rk3588-npu` register
  headers with the Mesa regcmd format, and is the starting point our kernel-access layer
  was built from. Working single-task fp16 matmul; ships a kernel NULL-deref iommu patch.
  **No** tiling/quant/multicore/int8; those, and the rewritten shim, are ours. (Its
  `rocket_interface.c` passes `timeout_ns` raw, an absolute-deadline bug we fixed.)

- **johanvdb/ggml `rocket-backend`**: a ggml backend *skeleton*. `*-matmul.cpp`
  returns -1 (CPU fallback), `*-dequant.cpp` is unimplemented. Scaffolding, not a working
  model path, useful as a structural reference only.

- **mtx512 / jas-hacks `rocket-userspace`**: the original RKNN-reverse-engineering repo
  (blog: jas-hacks.blogspot.com, "RK3588 reverse engineering RKNN"). Its
  `gen_matmul_task` (matmul-as-1×1-conv over NVDLA CNA/CORE/DPU blocks) is the
  most complete register config to start from, but it targets the **proprietary
  rknpu ioctls** (5.10 BSP), so driving it through `rocket` requires swapping the shim
  and adding the DPU-RDMA block it omits.

- **poad42/opennpu_rk3588** (MIT, first commit 2026-08-02, read at `983282e`): an
  ONNX/JAX-to-RK3588 compiler and runtime whose PJRT plugin makes the NPU a first-class JAX device
  (`jax.devices()` -> `[npu:0]`). It drives the **vendor `rknpu` BSP driver** on a 6.1 vendor
  kernel (`/dev/dri/card1`; `MEM_CREATE`/`MEM_MAP`/`MEM_SYNC`/`SUBMIT`/`ACTION`), so its
  runtime, ioctl layer and kernel patches do not transfer to the mainline `rocket` path, the
  register encoding does. Same author as **poad42/smolvlm_rk3588_full_npu_native** below.
  The useful subset is its docs, `cna_matmul.c`, `lm_forward.c` and the four kernel patches.

  **It is two stacks and only one is worth reading.** `pjrt_c/cna_matmul.c` emits the CNA
  descriptor **by formula**, 112 `uint64` entries computed from M/K/N, plus a weight cache
  that preloads all 144 GPT-2 matrices at init. `matmul_tmpl.h` (1.0 MB), `tanh_tmpl.h`
  (384 KB) and `relu_tmpl.h` are **captured vendor programs replayed with the DMA addresses
  patched**, which is why their op table reads "MatMul supports 5 GPT-2 shapes."
  `docs/ARCHITECTURE.md` says the templates were "captured once from vendor toolkits via
  ioctl tracing", while the README's provenance section says no proprietary code was involved.

  **The formula path agrees with ours register for register** [source-confirmed, read from
  `cna_matmul.c`]. Entry encoding `(op << 48) | (val << 16) | reg`, block tags `0x0201` /
  `0x0801` / `0x1001` for CNA / CORE / DPU, and CNA `0x100C`, `0x1010`, `0x1014`, `0x1020`,
  `0x1024`, `0x1030`, `0x1040`, `0x1070`, `0x1084`, `0x1088`, `0x1110` with DPU `0x4020` as
  the destination base. Their `0x1040` leaves bits [10:8] zero, an independent instance of
  the `FC_DATA_BANK` = 0 rule in [encodings/cbuf-reuse.md](encodings/cbuf-reuse.md), and
  they write the `FC_DATA_SIZE0/1` pair as `(1<<16)|M` and `K`, as `npu_regcmd.c` does.
  Three machine facts match. **`M % 4`**: their guide states "M=1, 2, 3 produce completely
  wrong results" and `npu_cna_cache_run_m()` pads to `M4 = 4` before every submit, the same
  trap and the same workaround as [matmul-as-conv.md](matmul-as-conv.md), and a second RK3588
  witness for the row in [chips/porting-patterns.md](chips/porting-patterns.md) where the
  RK3576 carries no M constraint. **CBUF = 12 banks × 32768 B.** And `nbuf_size=0` with
  `CONFIG_ROCKCHIP_RKNPU_SRAM` unset. Their host weight scatter (`weight_fp16_off`, 16
  outputs per group, 32 inputs per group) and feature scatter (`feature_data_off`, C2=8 fp16
  input, C2=4 fp32 output) are a second independent statement of the native cube layout, and
  that they scatter on the host at all corroborates the no-on-chip-layout-conversion finding:
  the README's "reads W directly as fp16, no reordering" describes the absence of
  quantization, not of tiling.

  **The companion report mislabels the entry fields** [source-confirmed, against Mesa's
  `target` enum and `rkt_regcmd.c`]. The report, *OpenNPU v1.0* (amohan.dev, 2026-08-18),
  describes an entry as a 16-bit register, a 16-bit value and a 32-bit tag.
  `codegen_synthesize.py` packs that layout (`'<HHI'`) and fills the tag's low half with the
  value's upper 16 bits for every register, not only for DMA addresses. Those bytes are the
  32-bit value field of the `(op << 48) | (val << 16) | reg` encoding, so a 16-bit value
  written from the prose truncates any wider field. The tag's high half is the block target:
  `CORE0_ID` (`0x1001`) tags the DPU `0x4xxx` registers and `CORE1_ID` (`0x2001`) the DPU_RDMA
  `0x5xxx` ones. The report's claim that the three cores are addressed as `0x1001`, `0x2001`
  and `0x3001` misreads that field.

  **Their bank split allocates no slack bank**: `fd_banks = ceil(M·K·2 / 32768)`,
  `wt_banks = 12 - fd_banks`, so that path is exposed to the feature-DMA overread that reads
  one CBUF bank past the allocation here [expected; not observed on their driver].

  **Three stated impossibilities are bounds of their harness, not of the silicon.**
  "Multi-core model-parallel matmul is impossible" follows from a position-locked 28 KB
  template block and a ~3.5 MB scratch DMA window; this stack runs 3-core per-fd. "K > 768
  requires tiling" is their bank split's own consequence, `cna_matmul.c` clamps `Kt` to 768
  and accumulates the tiles on the host. And "w8a8 … 16% error, 256 levels insufficient for
  768-wide" is the per-tensor-scale wall that a Hadamard rotation plus a per-output-channel
  requant closes.

  **Their DMA ceiling belongs to that path.** `ARCHITECTURE.md` fits
  `t = 0.08 ms + bytes / ~1000 MB/s` and calls 1 GB/s "the NPU internal DMA engine's hardware
  limit"; the README's results table says 3.0 GB/s. The two do not reconcile with each other,
  and both sit far under what the regcmd path here sustains.

  **The decode claim, and why it is the weaker of the two on record.** GPT-2 124M at
  **28 ms/token (36 tok/s)**, KV-cached, tabled as "NPU 1.4x", nominally a second challenge
  to the decode-on-CPU default after the RK3576 vendor's M=1 decode. Its comparison column is
  **RKLLM at 33 ms/token**, which is Rockchip's NPU runtime rather than a CPU measurement; the
  timing (84 matmuls × 0.32 ms = 21 ms/token) closes only at 3 GB/s, where their own 1 GB/s
  model puts a 236 ms floor under GPT-2's 144 matmuls; and `lm_forward.c` keeps the KV cache,
  attention, softmax, LayerNorm and GELU on the CPU, offloading only the four per-layer
  projections, at a padded M=4. [their claim, internally inconsistent, untested here]

  **What it has that nothing else here does:** a working **PJRT plugin** (`pjrt_c_api.h`
  v0.112 pinned to jaxlib 0.10.2), a frontend shape none of `ggml-rocket`, `tflite-rocket` or
  `ort-rocket` covers. Its four kernel patches (batch submit, an IOMMU lock-free domain fast
  path, a `kmem_cache` for job structs, and fence fds) are an independent rediscovery of
  three of the dispatch-floor levers in `patches/rocket`, reached on the other driver.

  **The four patches are levers worth knowing and code worth not copying** [read from the
  series]. They are plain `diff -u` against `drivers/rknpu/` on the vendor `develop-6.1` tree,
  so none of it applies here; what transfers is that a second effort, on the other driver,
  converged on batch submit and an IOMMU domain fast path. Four defects sit in the two that
  matter. `rknpu_batch_submit()`'s `err_iommu` unwind puts the domain once per *allocated* job
  rather than once per *acquired* one, so a failure at job k of n underflows the refcount by
  n-k. Its `core = ffs(core_mask) - 1` takes only the lowest set bit while `run_count` still
  counts the whole mask, so a multi-core mask queues on one core, decrements once and never
  commits -- a silent hang, and nothing validates the mask. It publishes `sd->job` under
  `irq_lock` and then commits to hardware with no lock held, leaving a window in which the IRQ
  handler sees a job the hardware has not been programmed for -- the same driver-state-versus-
  hardware-state divergence as the RK3576 two-jobs-in-flight hazard. And 0002's fast path reads
  the refcount and then increments it non-atomically where `atomic_inc_not_zero()` is the
  primitive, with `iommu_domain_id` read outside the atomic besides; it is safe only because one
  domain is ever used. Two smaller ones: `rknpu_batch_submit` gets no prototype in any patched
  header though 0003 calls it, and the README's `patch -p1` from `drivers/rknpu/` cannot apply
  paths rooted at `a/drivers/rknpu/`. Their 0003 also carries a `dma_buf` leak fix on fd close
  (bare `vunmap` to `dma_buf_vunmap` + `unmap_attachment` + `detach`) -- the same BO-lifetime
  class as `patches/rocket` 0013/0014, present in the vendor driver too.

  **The one lever here we have never tried is SoC-level AXI QoS, and it is a device-tree change**
  [their measurement, untested here]. `docs/ref/QOS_TEST_RESULTS.md` notes the vendor driver
  leaves `bw_priority_addr = 0x0`, so the NPU's bus QoS is never programmed at all, and sets
  `priority-init=7` / `mode-init=0` on all five NPU QoS nodes (`qos_npu0_mwr` `0xfdf72000`,
  `qos_npu0_mro` `0xfdf72200`, `qos_npu1` `0xfdf70000`, `qos_npu2` `0xfdf71000`, `qos_mcu_npu`
  `0xfdf72400`) through the `pm_domains` driver's own DT path. Measured there: **2x on ~200 KB
  matmuls** (`[64x768x64]` 0.658 -> 0.322 ms, `[64x64x768]` 0.416 -> 0.228 ms) and **nothing
  past 1 MB** (1.00x, 0.96x, 1.00x on three shapes), with GPT-2's 144 large matmuls unchanged at
  235 vs 237 ms. This sits upstream of the driver entirely, so unlike the rest of that repo it
  should port to `rocket` unchanged. It is a different layer from the MCIF per-client 2-bit QoS
  at `0x8000` in [nvdla-lineage.md](nvdla-lineage.md), and neither layer has been measured here.
  Their reading of it -- that the null result at 1 MB "definitively proves" the ceiling is the
  internal DMA engine's -- is not supported by the test: a null result at one priority bounds bus
  arbitration, not the engine, and the 1 GB/s ceiling it argues for is the figure their own
  README contradicts at 3.0 GB/s.

  **`docs/ref/NPU_REGISTER_INVESTIGATION.md` is a second witness on the counter page, and it
  disagrees with ours about which offsets are fatal.** Probing core 0 by kernel-module
  `ioremap(0xfdab0000)`, they report `0x2210`-`0x223c` raising a *contained* kernel oops (DECERR,
  board survives) and `0x8000`-`0x803c` **hanging the bus** hard enough for the watchdog to
  reboot -- where [perf/hw-byte-counters.md](perf/hw-byte-counters.md) has the `0x2xxx` read
  hard-locking the SoC and the legacy `0x80xx` offsets reading 0 as mapped DDMA space. Both
  probes are kernel-side `ioremap` + `readl`, so the split is not obviously an access-path
  artifact. Treat **both pages as unsafe to read on either path** until someone re-runs it; the
  cheapest wrong guess here costs a cold power-cycle. What the disagreement does settle is that
  file's own caveat -- an independent DECERR on a different kernel is further evidence the
  `0x2xxx` page is undecoded rather than merely power-gated. Two register facts there are new
  here: `0x1004` bit 4 accepts a write on the RK3576 (`state_init` 0x1e) and **is not writable on
  the RK3588** (0x1e reads back 0x0e), a plausible internal-memory enable and a third independent
  statement that the two parts are different revisions
  ([chips/porting-patterns.md](chips/porting-patterns.md)); and `VERSION` at `0x0` reads
  `0x46495245`, "FIRE".

  Vision numbers, for comparison rather than transfer: SigLIP ViT at 995 ms/image on one core
  against a ~2500 ms CPU baseline, 370 ms/image across three, cosine 0.9999 against the
  HuggingFace reference, with all 48 matmuls on the NPU and attention / GELU / LayerNorm on
  the CPU under OpenMP, against cosine 0.999998 for the full block on the FOSS path
  ([encodings/siglip-encoder.md](encodings/siglip-encoder.md)).

- **dnhkng/open-rknpu** (MIT, first commit 2026-09-11, read at `0e0afcd`): an int8 compiler
  from ONNX to regcmd, in Python, and a libc-only runtime for the **RV1103/RV1106** NPU
  (`rockchip,rv1106-rknpu`). It drives the vendor `rknpu` driver v0.8.2 with no IOMMU, on a
  Luckfox Pico Mini B at 420 MHz. The vendor toolkit served only as a black-box oracle.

  It compiles dense, depthwise and transposed int8 convolution, 2×2 pooling and elementwise joins.
  It also runs LUT sigmoid and tanh and whole small DAGs. Every claim is tied to a board ledger
  of 1,799 models checked byte for byte against a Python integer reference.
  `docs/investigation-log.md` is the chronological record, failed hypotheses included.

  **Its register map is a third point between the two parts here** [source-confirmed, read from
  `src/open_rknpu/register_profile.py` and the emitters]. CORE, DPU and DPU_RDMA sit
  at the RK3588's offsets with the RK3588's fields. So do the CNA geometry words, the feature
  and weight bases and the pad value at `0x1184`. A few CNA words follow the RK3576 instead.
  `0x103C` and `0x1044` carry the data entries and the input width, and `0x118C` is the input
  width minus one in both halves. The pads at `0x1068` are byte-packed, where the RK3588's are
  nibbles.

  The `0x118C` formula matches the RK3576's in [chips/rk3576-regcmd.md](chips/rk3576-regcmd.md).
  So a CORE or DPU result of theirs ports to the RK3588 more directly than an RK3576 result
  does.

  **Three of their results reproduce here.** Each is written up in the note that owns it.

  **The deconvolution mode runs a whole ConvTranspose in one task once the output extent is
  programmed.** Their transposed emitter sets the deconv stride fields to `s-1` and keeps the
  input geometry undilated. It programs the output geometry to the transposed extent and pads by
  `k-1-p`. That gives 8×8 -> 17×17 at pad 2 in one task, and about 90 models byte-exact on
  their board. Their stride fields are live without `CONV_CON1[16]`. On the RK3588, with the bit
  set, the same program is bit-exact up to 64×64 -> 128×128 [HW sweep, Turing RK1, 2026-09-23].
  See [encodings/conv-transpose.md](encodings/conv-transpose.md).

  **`OUT_CVT_SHIFT[30]` selects the tie rule.** Their field experiment sets bit 30 in an
  elementwise add, and 1,438 predicted bytes move from half-to-even to half-away-from-zero. Both
  settings reproduce on the RK3588 and on the RK3576 [HW sweep, 2026-09-23]. See
  [encodings/out-cvt-converter.md](encodings/out-cvt-converter.md). Their `quantization.md`
  says the conv path rounds half away. Their code and tests say half to even, and the board
  agrees with the code.

  **The BS stage shifts a product held wider than int32.** Their per-channel epilogue is
  `round_even(((acc + A) * C) >> 14)`. A is the bias, and a Q14 multiplier C and the negated
  weight zero point share its 32-byte record for four channels. That is the RK3576's A/B/C
  coefficient record at a different grouping. It is exact on their board at an accumulator of
  2.08e9 with `C = 16384`. On the RK3576 the `0x5024` shift word behaves the same way: wide,
  multiply before shift, ties to even [HW sweep, H96 MAX M9, 2026-09-23]. See
  [chips/rk3576-regcmd.md](chips/rk3576-regcmd.md).

  **Consistent with ours, and not new here.** The elementwise engine's second operand read needs
  its surface notch, or a compact operand hangs. That read is linear, one atom per output pixel,
  with no spatial broadcast. `BS_OW_CFG.OW_SRC = 1` hangs, and the 2×2 average rounds half to
  even.

  Their LUT setup words (`0x410C = 0x50500`, `LE_START = 0xffffc000`, `LO_END = 0x4000`)
  are the ones the RK3576 vendor programs carry. Their LUT's negative half advances by a
  whole-bit gain that follows the stem's weight scale, which they fit and did not decode. It has
  the shape of a sign-dependent shift ahead of the table **[expected]**.

  **What does not transfer.** The RV1106's channel walls do not: 511 weight parts per task, and
  8192 output channels from a 9-bit block index. Nor do its per-task costs at 420 MHz. Nor does
  the runtime's completion scheme for a vendor kernel built without fence support, a
  non-blocking submit drained by a blocking barrier job.

- **Seeed-Solution/openvoicestream**, with its RK engine **suharvest/rkvoice-stream** and
  **suharvest/rknn-matmul-parallel** (MIT, read at `9521320`, `5978d04` and the 2026-04-12
  head). It is a streaming ASR and TTS product stack for Jetson, the RK3588, the RK3576 and
  the Raspberry Pi. On Rockchip it runs the vendor RKNN and RKLLM runtimes and carries no
  register-level material. Four of its measurements bear on this silicon.

  **A whole-graph Whisper encoder runs faster than our matmul offload, by 1.70x on one board
  rather than the ~2.5x their figure implies.** `rknn_model_zoo`'s whisper-base encoder, fp16 at
  a 20 s window on three cores, takes ~250 ms on their RK3588 at an NPU clock of 1 GHz by their
  measurement, and about the same on their RK3576. Their model file, run with their runtime
  version (`rknnlite` 2.3.0) on all three cores of a vendor-kernel RK1 at 1000 MHz, takes
  344 ms [HW sweep 2026-09-26]. Their figure is not reproduced. The same-board comparison and
  what bounds the RKNN graph are in
  [encodings/whisper-encoder.md](encodings/whisper-encoder.md) §"In-model fused integration".
  Their vendor default also ran the decoder on the NPU with no KV cache, a 12-slot window. A CPU
  decoder with a cache took their RK3588 from RTF 0.149 to 0.061 and cut long-form WER, the
  split whisper.cpp already makes with ggml-rocket.

  **Per-op NPU decode loses to a compiled graph by 5x.** rknn-matmul-parallel runs
  Qwen3-ASR-0.6B's decoder (28 layers, d 1024) as 196 `rknn_matmul_run` calls a token on the
  RK3576. That costs 225 ms a token at fp16 and 146 at int8, against RKLLM's 43. Their own
  breakdown is matmul 91, weight rebind 91, lm_head 35 and CPU 8 ms.

  Batch prefill at M 60 costs 17 ms a token against 202 row by row. It is the dispatch-bound
  decode of
  [perf/not-mac-bound.md](perf/not-mac-bound.md), reached through the vendor API. The repo's
  "~16 ms/token" headline and their writeup's 21-29 ms do not reconcile with that table.

  **Their "the RK3576 NPU has no bf16" is the toolkit's datatype menu.** Under a raw register
  program the part contracts bf16 bit-exactly, see [chips/rk3576.md](chips/rk3576.md)
  §"The precision fork".

  **Their RK3576 performance gate disables the CPU's deep idle state** without measuring it. On
  the RK3588 that state costs the offloaded arm of a prefill a few percent, see
  [perf/cpu-governor-and-offload.md](perf/cpu-governor-and-offload.md).

  **Their vendor-runtime traps** are for a reader on the RKNN path, and none is reproduced here.
  RKLLM and RKNN share IOMMU domain 0 by default, and `rknn_run` hangs once RKLLM has loaded.
  RKLLM's `base_domain_id = 1` separates them (rknn-llm#437), and an RKNN model context cannot
  leave domain 0. The int4 matmul in librknnrt 2.3.x loses about half of every positive value,
  a nibble-order defect (rknn-toolkit2 PR #412) that they measured at cosine 0.06. Version 2.3.2
  segfaults in `rknn_run` on RK3576 Linux, and 2.3.0 does not. The RK3576's `rknn_matmul_api`
  offers fp16 operands only with an fp32 output.

  **The RKNN compiler refuses a layer past the 13-bit cube-dimension ceiling, or falls back to
  the CPU.** Kokoro's vocoder at 13,080 samples fails the `REGTASK` check with
  `limit: 0x1fff, value: 0x3318`. That is the ceiling in
  [encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md).
  Where it does place Kokoro's generator on the RK3588, at their 16-token bucket, it is slower
  than the CPU. It takes 3464 ms at fp16 and 2656 at int8, against 1819 ms in ONNX Runtime.

- **Mesa Teflon on `rocket`**
  (rpardini/mesa-teflon-etnaviv-rocket-docker; BredOS wiki `NPU/rocket.md`). Upstream
  Mesa's Teflon TFLite delegate is the public FOSS-rocket baseline: kernel >= 6.18
  `CONFIG_DRM_ACCEL_ROCKET`, `/dev/accel/accel0`, `libteflon.so` via
  `tflite.load_delegate`. Envelope: **quantized uint8 CNNs only, conv + EW-add + fused
  ReLU, single-core, no SiLU, no transformer**; AVGPOOL/RESHAPE/SOFTMAX fall to CPU.
  Perf ~13-17 ms MobileNetV1 (~3-4x over CPU ~48 ms). `rocket-userspace`/`tflite-rocket`
  run 3-core per-fd, fp16/int8/int4, SiLU/GELU and transformer blocks, and MobileDet.
  rpardini also
  carries per-board NPU-regulator DT patches (CM3588-NAS, NanoPC-T6, **Turing RK1**),
  the voltage wiring the `patches/rocket` clock/volt coupling depends on.
  [source-confirmed]

- **llama.cpp ggml NPU-backend discussion** (ggml-org/llama.cpp#8111): an upstream
  attempt at an RK3588 (and Tenstorrent) ggml backend that **stalled** (author pivoted to
  Tenstorrent for its open stack). It validates our design: the hard path they hit
  (managing device buffers, needing dtype/layout at *allocation* time, no
  offset-subbuffering, the **per-fd 4 GB IOVA limit**) is exactly what `ggml-rocket`
  sidesteps via the **BLAS-backend model** (host/CPU buffers, offload only `MUL_MAT`
  through `graph_compute`, pack+DMA inside `rocket-userspace` per call). slaren's guidance
  there (`set_tensor` carries `tensor->type`; `get_alloc_size` / `get_max_size` /
  `get_alignment` for device-buffer backends) is the API surface we deliberately don't
  need. The 4 GB-window / no-subbuffer constraint is real and handled one layer down in
  `rocket-userspace` IOVA management (`rocket_prepacked_int8.c` escape check), not at the
  ggml buffer-type. [source-confirmed]

## Quantization / coherence references

- **clehaxze.tw gemlog**: "Benchmarking RK3588 NPU matrix multiplication
  performance" (eps 1-2; the 2024-02-14 post has the hard numbers). Same silicon,
  RKNN-measured (an upper-reference): fp16 ~900 GFLOPS peak, int8 ~2x fp16, int4 ~4x
  (in MAC terms), K-spilling past ~1024.

- **Martin Chang, "porting LLM to RK3588"** (RWKV-on-RK3588 talk). The
  **native-K-reduction** insight: the conv reduces over K
  in one pass up to the CBUF limit ("Max K=2048 @ FP16" in one pass), nobody used
  the DPU eltwise for K-accum. Reframed our tiling. Also the clearest external statement
  of the **decode split**: NPU is good at MatMul, bad at GEMV (M=1); CPU GGML wins decode
  (83 ms vs 61 ms/token) because llama.cpp's NEON GEMV is already bandwidth-saturating.
  See [perf/decode-gemv.md](perf/decode-gemv.md).

- **Hummingbird+** (Li et al., *"Hummingbird+: Advancing FPGA-based LLM Deployment from
  Research Prototype to Edge Product"*, FPGA '26, doi:10.1145/3748173.3779189), a
  dedicated **FPGA** GEMV engine (Zynq UltraScale, 140 DSPs, <1K LUTs, 272 GOPs) for
  edge LLM decode. Consulted on the question "are there GEMV optimizations we are
  missing?" The answer is no for fixed silicon: its speedups (DSP pre-adder operand
  packing, DDR LUT-mux elimination, DOT/AXPY mode switch, W4/KV8 dual precision) are
  reconfigurable-datapath microarchitecture with no analogue in the RK3588's fixed
  convolution pipeline. What transfers is bandwidth-reduction at the model/format level
  (4-bit weights + 8-bit KV, MoE), already available in stock llama.cpp. It also cites
  RKNN-LLM at "~10 token/s on a 3B on RK3588," i.e. even vendor on-NPU decode is
  bandwidth-bound, not a CPU-beating win. See [perf/decode-gemv.md](perf/decode-gemv.md).

- **SHARD** (Mohan et al., *"SHARD: A Compatibility Framework for Deploying Transformer
  Models on Edge NPUs"*, EuroMLSys '26, doi:10.1145/3805621.3807618; also the
  amohan.dev blog), deploys the **SigLIP-B/16 vision encoder** (93 M params, the
  SmolVLM-256M front-end) on RK3588 through **rknn-toolkit2**, so it is a vendor-
  toolchain *workaround* (graph sharding, GELU-approx / LayerNorm-decompose
  legalization, fusion barriers; all RKNN-steering a regcmd path doesn't need). What
  transfers: (1) the `0xe010 "REGTASK Overflow"` is an undocumented **13-bit
  instruction-register width limit, operand indices > 8191 fail** [source-confirmed],
  the *same* 13-bit register class as our `DPU_DATA_CUBE_WIDTH` 8191 corruption [HW
  sweep] (independent cross-validation; a **general operand-index ceiling**, not LUT-
  only; see [encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md)); (2) a
  **32 KB per-op scratchpad** -> <= 16384 fp16 elems/op (their attention tile 256×64),
  the tiling discipline we already follow; (3) **Sandwich λ-scaling** (host pre×0.1 /
  post×10.0) keeps fp16 off a "saturation cliff" (cosine 0.98->0.11 by layer 5 without
  it), and the paper's finding that **AWQ fails "because the error stems from activation
  outliers, not weight sensitivity"** validates the Hadamard activation-rotation choice.
  Numbers (Orange Pi 5 Max): SHARD 2.24 s @ cosine 0.95 vs RKNN-FP16 19.63 s @ 0.64
  (CPU-fallback transpose) vs RKNN-INT8 1.40 s @ 0.02 (collapsed) vs CPU-FP32 30 s, a
  vendor baseline for the ViT-encoder primitives (MHA / LayerNorm / GELU / FFN) the
  rocket stack runs on-NPU.

  The 32 KB figure comes from synthetic ONNX graphs run through RKNN: a 32 KB tensor passes
  and a 32.1 KB one fails. That bounds what the vendor toolchain accepts, and does not isolate
  a hardware buffer. RKNN's Transpose limit on height × width, which the RKNN-Toolkit2 issue
  #163 entry records, is also 16384 fp16 elements, and is the likely source [expected]. The
  blog reads `0xe010` as a memory overflow. The compiler's own message is a per-field width
  check.

- **poad42/smolvlm_rk3588_full_npu_native**: a concrete deployment of the same **SmolVLM-256M**
  front-end on the RK3588 NPU via the **proprietary** `rknn-toolkit2` + RKLLM bindings (no stated
  OSS license on the main code), i.e. SHARD's model on the vendor toolchain, the code does not
  transfer to the rocket path. What corroborates SHARD independently: it splits the vision encoder
  into **24 shards (12 layers × 2 blocks) across NPU cores 0-2**, FP16/INT8 hybrid, and wraps each
  NPU block in input/output scalers ("**Sandwich Quantization**" / InputScaler) to keep fp16 off
  the saturation cliff, the same sandwich-scaling lever SHARD formalizes as λ-scaling, independent
  confirmation that this exact front-end needs activation rescaling to survive NPU quant. It also
  tiles attention into **32×32 blocks with small-chunk transposes** purely to dodge the **RKNN
  compiler's** transpose handling, a constraint the regcmd path does not share. No published
  accuracy or perf numbers; a parallel effort to benchmark against, given our SigLIP-B/16 encoder
  runs the full block on the FOSS path at **cosine 0.999998**
  ([encodings/siglip-encoder.md](encodings/siglip-encoder.md)).

- **r/RockchipNPU thread**: field reports
  that int8/int4 LLMs on RK3588 need **INT8_HADAMARD / INT4_HADAMARD** for coherence
  (e.g. gpt-oss-20b). Validated our Hadamard-is-mandatory finding.

- **rk-llama.cpp issue #9** (the proprietary rknpu2 stack): independently confirms
  the *direction*: "Q8_0 maps well to RKNPU W8A8" -> Q8_0 prefill +200-400%; Q4_0
  maps poorly; decode is memory-bound (matches our CPU-decode split). Their Q8 win is
  **operand bandwidth at a lower dispatch floor**, not a capability the open path lacks:
  RKLLM quantizes to W8A8 (8-bit weights *and* activations), halving the bytes moved for
  both operands on a path that is dispatch/DMA-bound, and their own Gemma int8 runs at
  40-58% NPU utilization, so their stack is not MAC-bound either. The one thing it is
  *not* is native on-device int32 K-accum. Whether the DPU-EW can add int32 at all is
  unestablished (see [encodings/k-accumulation.md](encodings/k-accumulation.md)).
  See [perf/not-mac-bound.md](perf/not-mac-bound.md) §"The proprietary stack's int8 win is bandwidth".

- **kevbuh/rk3588**: notes/datasheets confirming int4/int8/int16/fp16/bf16/tf32
  support on the silicon (used to sanity-check the precision menu).

## Quant-coherence primary research

- **QuaRot / SpinQuant** (Hadamard-rotation quantization) and **SmoothQuant**: the
  principled fixes for activation-outlier-driven int8/int4 gibberish. We use a
  Kronecker Hadamard (H_{2^k} ⊗ H_60 via Paley for Gemma's non-power-of-2 K). See
  [encodings/k-accumulation.md](encodings/k-accumulation.md); measured perplexity with
  W8A8 + Hadamard tracks fp16.

- **AWQ** ([mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)): activation-aware
  *weight* quant: per-group scales + protect the ~1% salient channels (largest activation
  magnitude). Ships the reusable scale-search, not just weights. The weight-side lever for our
  int4 quality (W4A16 as-published; complements Hadamard on the activation side). -> the int4 work.

- **Outlier Suppression+** (arXiv:2304.09145, Wei et al. 2023;
  ModelTC/Outlier_Suppression_Plus), per-channel **shift** (kills *asymmetric* activation
  outliers) + per-channel **scale**, both migrated into adjacent layers (free at inference).
  Near-FP at 8/6-bit. Complements Hadamard; the asymmetric-shift fits our asymmetric uint8
  detection weights. -> the int4 + MMSE range-setting work.

## Transformer softmax / exp on NPUs

- **Attention Distribution-Aware Softmax for NPU-Accelerated On-Device Inference of LLMs**
  (Sadheerthan et al., *Electronics* 2026, 15, 1312).
  Confirms softmax is *the* NPU transformer bottleneck because NPUs lack a native `exp` unit, and
  proposes a distribution-aware, variable-degree LUT approximation of `exp` (PSO-learned non-uniform
  segments snapped to a 128-bin grid), cutting exp-kernel cycles-per-call ~18.5% vs uniform Degree-4.
  **Relevance:** it optimizes the exp *kernel compute*; our fused-encoder softmax is bound by host↔NPU
  *transfers* instead (proven: a slower host `expf` softmax ran 3x faster than the on-NPU LUT path;
  see [encodings/whisper-encoder.md](encodings/whisper-encoder.md) §in-model). Its quant-domain,
  clamp-`[-20,0]`, in-domain row-max recipe is the design reference for a fully-on-NPU **resident**
  softmax (no host round-trip), the lever if Whisper encoder fusion is ever pursued for perf.

- **YOLOv10-on-RK3588 latency** ([THU-MIG/yolov10 #115](https://github.com/THU-MIG/yolov10/issues/115)):
  YOLOv10s is ~2.7x *slower* than YOLOv6/DAMO-YOLO on RK3588 (32 vs 12 ms @416) despite fewer
  params: its attention/PSA blocks are NPU-hostile. A detection-pillar datapoint (same
  attention/softmax-fights-this-NPU theme); relevant if the detection pillar moves toward YOLO.

- **LSH9832/edgeyolo** (`cpp/rknn`): EdgeYOLO (anchor-free, YOLOX-style) deployed on RK3588 via
  the **proprietary RKNN runtime**, with decode + NMS wrapped inside the `RKNN::YOLO` class (not a
  reusable, backend-agnostic postprocessor). One transferable datapoint: the **LeakyReLU** variant
  (Tiny-LRELU) runs **65 FPS** vs **24 FPS** for the **SiLU** variant at 384×640 / int8 / 3-core,
  the author attributing the gap to "SiLU activation layers", a ~2.7x activation-driven cost, the
  same NPU-hostile-op theme as the YOLOv10 PSA datapoint (above). If the detection pillar moves
  toward YOLO-family graphs, prefer LReLU over SiLU for throughput; the number is an RKNN-path
  reference, not our DPU-LUT path ([encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md)).

## NPU generation & throughput (decode / multi-instance)

- **"Accelerating OpenPangu Inference on NPU via Speculative Decoding"**
  (arXiv:2603.03383, Dai et al. 2026; wujing215/OpenPangu7B-with-Medusa), Medusa
  multi-head draft (no separate draft model) + **static tree attention** + zero-copy retrieval to
  fit the NPU's static-graph execution; 1.35x short-seq, long-seq memory-BW-bound. Closest prior
  art for speculative decoding on the NPU; the static-tree design is the part that ports to a fixed-graph NPU.

- **leafqycc/rknn-multi-threaded** + **HN 48527630 (alebal123bal)**: two working RK3588
  multi-instance inference pools: thread pool + queue + one RKNN context per core, round-robin,
  preallocated buffer pool, RGA preprocessing. 2.6x (YOLOv5s) / 31->46 FPS (full isp->RGA->YOLOv8n
  pipeline). The reference architecture for the multi-instance throughput pool (Frigate multi-camera).

- **"我以为 NPU 推理到了硬件极限，结果发现是 CPU 频率在拖后腿"** (zhihu, johnjiamzhong/AlertGateway,
  RK3588S, RKNN path; `zhuanlan.zhihu.com/p/2051444846548857552`), a proprietary-path writeup that
  attributes a "~40 ms is the NPU hardware limit" YOLOv8s number to the **CPU governor**, not the
  NPU: `rknn_run` includes CPU-side submit + IRQ-wait, so it swings 59 -> 35 ms (−41%) from the
  CPU governor alone (NPU clock fixed at 1 GHz), and CPU+NPU both pinned to `performance` collapses
  the jitter. Independent corroboration of our CPU-side submit/dispatch floor; see
  [perf/not-mac-bound.md](perf/not-mac-bound.md) §Dispatch-floor reducers and
  [perf/clock.md](perf/clock.md). Methodology only; no FOSS-path numbers (it never leaves rknn).

## The NVDLA ancestor

- **NVDLA hardware manual v1** ([nvdla.org](http://nvdla.org/)): the only authoritative prose
  about this datapath; Rockchip publishes a register list and no semantics. Which of our
  questions it answers, which it answers *wrongly* for this silicon, and which it cannot reach
  is worked through in [nvdla-lineage.md](nvdla-lineage.md), with a per-page reading list. In
  short: the structure transfers (blocks, the ping-pong register file, CBUF banking and ports,
  CACC's within-one-layer accumulate, the MCIF/SRAMIF arbiters), the arithmetic details do not
  always (it specifies round-half-away-from-zero; the part rounds half to even), and the fields
  that matter most here (`PC`, `DECONV`, the CBUF granule allowance, the per-sign shift word)
  are Rockchip additions it never had, so its silence about them is not evidence.

## Compression

- **NVDLA weight-compression format** ([nvdla.org/hw/format.html](https://nvdla.org/hw/format.html)):
  CWT (packed non-zero weights) + WMB (1 bit/element mask, 128-B aligned) + WGS (per-kernel-group
  byte count). Kernel group 32 (int8) / 16 (fp16/int16), matches our weight-tile groups. CDMA
  decompresses into CBUF, skipping zero MACs. **Sparsity (pruning) compression**, no win on dense
  weights. Probable layout of the RK3588 CNA DCOMP (DPU==NVDLA SDP ⇒ CNA likely==NVDLA CC).

## Related RK3588 NPU work

- **SHARD** (EuroMLSys'26, doi:10.1145/3805621.3807618): RK3588 VLM via constraint-driven
  graph rewrite, 8.7x over RKNN. A design precedent for per-shard precision selection and
  for a per-layer FP16 hybrid; its primitives (native GELU/LayerNorm, CBUF tiling, hybrid
  CPU/NPU) are ones the rocket stack also implements.
- **clehaxze gemlog (2023)**: RK3588 NPU per-cycle MACs (2048 int4 / 1024 int8 / 512 fp16);
  notes that RKNN matmul lacks multi-core (the rocket path runs 3-core via per-fd).
- **t-firefly ROC-RK3588S NPU wiki**: vendor RKNN usage only; no multi-core/SRAM/register/driver
  detail.
- **sagi21805/matmul-npu**: a C++/OpenCV matmul wrapper over the proprietary RKNN toolkit2
  (fp16/int8 in -> fp16/int8/fp32 out); no regcmd/ioctl/tiling detail exposed.
- **llama.cpp#722 (2023)**: 7B-Q4 on an RK3588 (NanoPi R6s) at ~98 s/token: an early CPU-only,
  microSD/RAM-bound outlier (not representative of current llama.cpp CPU perf). The
  motivational datapoint: RK3588 CPU LLM is impractical at 7B, hence the
  NPU-prefill / CPU-decode split.
