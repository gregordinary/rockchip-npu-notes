# Sources — curated external references for FOSS RK3588 NPU work

External resources others can consult for RK3588 NPU work, with one line each on **why it
matters**. Each entry names its upstream so you can find it yourself. These are context and
cross-references; the facts in these notes are established by HW sweep and the FOSS Mesa
driver (see the [README](README.md) evidence tags).

## The authoritative regcmd / hardware sources

- **Mesa `rocket` (Teflon) driver** (`src/gallium/drivers/rocket/` in Mesa).
  The single most useful source: a *working*, in-tree FOSS driver that emits real
  conv regcmd for this exact hardware. `rkt_regcmd.c` (`fill_first_regcmd` = the
  validated CNA→CORE→DPU→DPU-RDMA sequence, the `add_tensor` eltwise geometry, the
  enable mask, the MRDMA-disable block), `registers.xml` (every register address +
  field), `rkt_coefs.c` (weight packing, the WEIGHT_ATOMIC_SIZE=32 reorder, the
  CACC 48-bit accumulator note), `rkt_ml.c` (feature packing, FEATURE_ATOMIC_SIZE=16),
  `rkt_task.c` (NVDLA-style tiling/split). INT8-only (TFLite delegate), so it does
  not show the fp16/int4 paths — but it is ground truth for the format.

- **RK3588 TRM + datasheets**. "Rockchip RK3588 TRM
  V1.0-Part1-20220309" holds the NPU register chapter — `RKNN_pc_*` `0x0xxx`, `RKNN_cna_*`
  `0x1xxx`, CORE `0x3xxx`, DPU `0x4xxx`, DPU_RDMA `0x5xxx` — and `pdftotext` works on it. Listed
  as an external Rockchip reference others may consult: the register facts these notes rely on
  are established independently by HW sweep plus the Mesa driver, not derived from it, which is
  why a TRM statement that contradicts a sweep loses. The RK3576 TRM (Part1/Part2 V1.2) is on
  disk beside it but is **not** cited by [chips/rk3576-regcmd.md](chips/rk3576-regcmd.md) — that
  part's register map was established from other sources, so treat the RK3576 TRM as unmined
  rather than as agreeing.

- **allbilly/npu** (`allbilly-npu`), esp.
  `include/rknnops.h`. A higher-level op generator (conv1d/2d, matmul, activations,
  LUTs) using the same Mesa regcmd encoding. Its `float16_alu_op(ALU_ALGO_ADD)`
  encodings are what cracked the **fp16 DPU-EW K-accumulation**; its int-EW
  `EW_OP_TYPE` bit is what we tested (and ruled out) for int32 K-accum. Broader op
  coverage than Mesa — the reference for going beyond matmul.

- **allbilly/rk3588** (`allbilly-rk3588`) — the same author's Python successor to the
  above, and a different kind of source: it carries no register definitions of its own
  (`experimental/registers.xml` and `include/rkt_registers.h` are Mesa's), but
  `conv_expt/capture_harness/decoded/` holds **83 vendor RK3588 conv register programs
  decoded to named CNA/CORE/DPU fields** — multi-task, ic 16-1280, planes to 150, oc
  12-1024, pointwise and depthwise. That is a vendor oracle for the RK3588 conv geometry
  words of the kind `tests/data/rk3576-vendor-capture/` is for the RK3576; the address
  registers are one session's IOVAs and only the geometry is comparable. Its capture
  route is a gdb harness on the vendor BSP runtime (`experimental/rknn/trace_librknnc_*.gdb`,
  `conv_expt/capture_harness/rknn_prefix_capture.gdb`, which patches the rknpu submit
  struct down to a one-task prefix) — a weaker instrument than the offline
  compile-and-decode used for the RK3576 captures, since it needs a vendor-BSP board and
  captures only what the vendor compiler chose to emit. RK3588 only; nothing in it
  addresses the RK3576 encoding. Neither allbilly repo carries a LICENSE file, so treat
  both as readable facts rather than as code or data to vendor.

- **RKNN-Toolkit2** (`github.com/airockchip/rknn-toolkit2`) — the vendor's proprietary
  compile-and-run stack; this project is a mainline alternative to it. Its offline compiler
  output is the reverse-engineering input for the **PPU pooling family**, which the FOSS
  Mesa/Teflon path never emits: a 1-op ONNX pool compiled to a `.rknn` and decoded for the
  PPU / PPU_RDMA register page yields the exact `RECIP_KERNEL = fp16(65536/k)` reciprocal
  format (method in [ppu-rknn-capture/](ppu-rknn-capture/); no vendor artifacts are
  redistributed). Every other encoding here comes from the FOSS Mesa driver plus HW sweep.
  RKNN3 (targeting RK1820 / RK3572) is a different NPU generation, out of scope.

- **RKNN-Toolkit2 SDK docs** (`github.com/airockchip/rknn-toolkit2/doc`, V2.3.2) — the vendor's own documentation, an **external cross-reference**
  rather than RE input. User Guide **§3.5.4**'s high-performance layout table covers the A/B/C
  tile-layout matrix and the same-A/B-dtype-only constraint (cf.
  [encodings/tile-layouts.md](encodings/tile-layouts.md)); **§6** the quant path (INT8-only,
  per-channel weights / per-tensor activations, range solvers, hybrid FP16 fallback — cf.
  [datatypes.md](datatypes.md)); **§5.3.3** the multi-core split op list and the IRQ-affinity tip
  (cf. [perf/iova-and-multicore.md](perf/iova-and-multicore.md)). The runtime header
  `rknn_api.h` carries the perf/mem query structs — `rknn_mem_size` is allocations, and
  per-frame bytes are an analytical string in `rknn_perf_detail` rather than a hardware counter,
  which is one more reason [perf/hw-byte-counters.md](perf/hw-byte-counters.md) had to go to the
  silicon. The `OP_Support` / Compiler-Operator-List docs are the op-coverage reference for the
  delegate roadmap. Not the same as RKNN3, which targets RK1820 / RK3572 — a different NPU
  generation, out of scope here.

- **`rknpu-reverse-engineering`** (phhusson / Tomeu lineage) — early-stage,
  STT/TTS-focused, on the **BSP `rknpu`/`/dev/dri/card1` path** (not rocket). Its
  register *encodings* are **superseded by Mesa `registers.xml` + the
  Teflon decode** (more complete, on our actual rocket path), and `rknpu-ioctl.h` is the
  BSP uAPI we don't use. **Still-useful artifacts:** (1) `hello2.c` — the **in-core
  IRQ/block-completion bitmap** (CNA/CORE/DPU/**PPU**/DMA-err, two reg-banks per block =
  `RKNPU_JOB_PINGPONG`), now captured in
  [perf/iova-and-multicore.md](perf/iova-and-multicore.md); confirms the PPU is a real
  separately-completing block. (2) `instrs.h` — a hand-assembled plain conv that
  **confirms our block/register format** (`0x0201`=CNA `0x10xx`, `0x0801`=CORE `0x30xx`,
  `0x1001`=DPU `0x40xx`, e.g. `DPU_EW_CFG 0x4070=0x383` plain-conv bypass) — provenance,
  not new info. (3) The `analysis` + `mess/dump-*` raw hex dumps of the 6 gem BOs
  (weights / **gem2 64-bit instruction stream** / **gem3 10-word task list with
  "jump-to-next-task"** / working / input / output) from real models — a reference for the
  **multi-task chaining structure** (task-persistence / the dispatch floor), though
  decoding raw BSP dumps is lower-value than a targeted Teflon capture on the rocket path.
  Independently corroborates the **float-only EW ALU** that kills integer K-accum.

- **Rockchip Hardware Design Guides** — `Rockchip_RK3588_Hardware Design Guide_V1.4_EN.pdf`
  and `3576_hardware_design_guide.pdf` (V1.1, 2024-05). Board-design documents: no register
  content, nothing about the regcmd interface or the NPU's internals, so they are useless for
  encoding work. What they carry is the **platform envelope** each part's NPU numbers must be
  read against, and the RK3576 differs from the RK3588 on every axis of it: the **DRAM bus
  width** (32-bit / 2 channels vs 64-bit / 4 channels, at an identical 2112 MHz PHY clock, so
  exactly half the bandwidth), the **NPU power rails** (the RK3588 has a separate
  `VDD_NPU_MEM`; the RK3576 has none, so its CBUF and other NPU arrays sit on the logic rail),
  the **peak operating point** (0.800 V / 4 A / 3.20 W vs 0.850 V / 4 A / 3.40 W, both at
  1000 MHz) and the **package thermal resistance** (θJA 15.84 vs 8.7 C/W). The DRAM figure is
  the load-bearing one: it is the mechanism behind the RK3576's DDR-traffic-driven atom drop,
  its long DPU write drain, and the ceiling on its host cube packing. Both are indexed in
  [chips/rk3588.md](chips/rk3588.md) and
  [chips/rk3576.md](chips/rk3576.md). `pdftotext -layout` extracts both cleanly.

- **6.6 BSP kernel `rknpu` driver** — the vendor kernel driver: HW performance
  counters, the devfreq/OPP table, and —
  critically for the clock work — `rknpu_devfreq.c` showing **200 MHz is the literal
  `POWER_DOWN_FREQ`** and that the vendor only ever sets the NPU clock while the
  power domain is active (`!pm_runtime_active` → refuse).

- **RK3588 BL31 / ATF binary** (`rk3588_bl31_v1.51.elf`, from rockchip `rkbin`) — the
  secure firmware that actually owns the NPU clock. Strings + `radare2`/`aarch64`
  `objdump` confirm the NPU is **SCMI clock id 6**, set in EL3 via
  `rockchip_opteed_clk_set_rate`, and clocked by a **PVTPLL whose min/max come from
  per-chip OTP** (`adjust npu pvtpll by otp: min=.. max=..`) — i.e. no static rate
  table, and **no voltage coupling** in firmware. The source of truth for why cold
  rate-setting wedges EL3 and why the real ceiling is an OTP value. See
  [perf/clock.md](perf/clock.md).

- **mainline `rocket` driver source + RK1 serial boot log** — `drivers/accel/rocket/`
  (android-mainline vs the local v7.1 build): stock upstream has **no** NPU clock
  handling (`clk_bulk_*` only), so the `clk_set_rate` ramp is the local `rocket-clk`
  patch, not upstream. The serial boot log confirms the SCMI handshake
  (`SCMI Protocol v2.0 'rockchip:'`) and the `quirk_clock_rates_triplet_out_of_spec`
  rockchip clock quirk.

- **drivercraft/rk3588-clk** — a Rust `no_std` RK3588 CRU clock library (MIT, for bare-metal /
  U-Boot) that sets the NPU clock by **direct CRU register writes** (`npu_set_clk` / `npu_get_clk`
  + `ACLK/HCLK/PCLK_NPU0..2` gates; `pll.rs` / `clksel.rs` / `gate.rs` / `constant.rs`). **Not a
  runtime alternative to the `rocket-clk` patch**: on mainline Linux BL31 owns the NPU clock via
  SCMI id 6 + a per-OTP **PVTPLL** (above), and this library has **no PVTPLL, no voltage handling,
  and no SCMI-conflict guard** — direct pokes would fight EL3 and skip the f/V coupling the patch
  depends on. **Useful as** an MIT-licensed, register-level cross-reference for the CRU NPU clock
  tree (PLL config, the clksel mux, gate bits) when annotating or extending the clock patch —
  register provenance, not a mechanism to adopt. See [perf/clock.md](perf/clock.md).

- **LKML: "[RFC PATCH v4 0/9] accel: rocket: Add RK3568 NPU support"** (Midgy BALON, 2026-06-13;
  v2 at [lkml.iu.edu/2605.3/10672.html](https://lkml.iu.edu/2605.3/10672.html); base v7.1-rc6). An RFC (design
  feedback, not for merge) adding RK3568 to the upstream `rocket` driver via a per-SoC
  `rocket_soc_data` (derive DMA width + core count from match data).
  Project-relevant facts:
  - **RK3568 NPU = a single NVDLA-derived core (0.8 TOPS), register layout matches RK3588** —
    corroborates "same NVDLA IP across RK SoCs"; our `librocketnpu` userspace should largely
    drive it too. End-to-end is blocked on **Mesa/Teflon userspace** (still emits RK3588-tuned
    config) + a HW issue (below) — exactly where our richer rocket userspace (full dtype matmul,
    general/DW/int8 conv, LUT activation, on-NPU EW mul vs Teflon's conv+add) is an asset.
  - **Address width: RK3588 NPU AXI/IOMMU is 40-bit; RK3568 is 32-bit.** So the **4 GB per-fd
    cap on RK3588 is the 32-bit *regcmd address field*, not the bus** (the bus reaches 40-bit)
    — as documented in [perf/iova-and-multicore.md](perf/iova-and-multicore.md). RK3568's 32-bit DTE needs
    `GFP_DMA32` page tables (`rockchip,iommu` ops; relies on Simon Xue's per-device-ops series).
  - **Stock rocket attaches and detaches the IOMMU domain on *every job*** (`iommu_attach_group`
    in `rocket_job_run`, `iommu_detach_group` in `rocket_job_handle_irq`) — each toggling the
    rk_iommu stall/reset/paging handshake. **Patch 5 keeps the domain attached across same-context
    jobs.** This is a **per-job dispatch-floor cost on RK3588 too** — a concrete, testable kernel
    lever for our submit-overhead-bound paths (detection 1×1s; KACC's nKt sequential jobs).
    [not-mac-bound.md](perf/not-mac-bound.md).
  - **The author reads the NPU's DMA byte counters** ("the NPU reads the full input and weight
    tensors per its DMA counters") — a lead vs our **dead-RK3588-counter** finding (reading the
    `0x2xxx` page hard-locks RK3588): the counters exist + are readable on RK3568, so RK3588's may
    differ by offset/access, not be absent. [hw-byte-counters.md](perf/hw-byte-counters.md).
  - **MAC/output stage never completes on RK3568** even on a **byte-exact replay of the vendor
    command list** → a hardware bring-up issue (PVTPLL/power/NoC de-idle), not a regcmd problem;
    the author asks for pointers — our deep BL31/PVTPLL/clock RE ([clock.md](perf/clock.md)) could
    help. Patch 3 starts the **PVTPLL compute clock via SCMI** (corroborates our PVTPLL finding);
    patch 9 wires **vdd_npu as the power-domain `domain-supply` (`need_regulator`)** so genpd owns
    the rail — the upstream-idiomatic alternative to our driver-held-regulator f/V coupling
    ([clock.md](perf/clock.md)); relevant if we upstream the volt work.
  - **OP_ENABLE offset** (from the v2 thread): the per-sub-unit `OPERATION_ENABLE` is `0x_008` on
    RK3588 (what we emit: `0xf008` + per-block `0x1008/0x3008/0x4008…`) vs `0x_00c` on RK3568 — a
    regcmd delta for any RK3568 port (not restated in v4's cover letter; verify against Mesa).

- **NetVar1337/linux-rk3576-rocket** — "[PATCH RFC 0/4] accel/rocket: add support for the RK3576"
  (VoidChecksum / Markus Kvam, 2026-06-11, against `torvalds/master`): binding, a
  clocks-by-name fix, per-SoC match data with PC_DONE polling, and the `rk3576.dtsi` core
  nodes. An independent mainline-targeted implementation of **gahingwoo**'s bring-up (below),
  which it credits throughout. **Compile-tested only — the author has no RK3576 hardware** and
  asks for testing reports. It covers what `patches/rk3576/npu/0001`, `0006` and `0007` do and
  nothing else, so it is a subset of the series here; three things about it are still worth
  knowing.
  - **Its central premise is refuted by measurement here.** Patch 3 states that the `PC_DONE`
    bits "are read-only in `INTERRUPT_MASK`, so completion cannot be routed to the GIC", and
    builds a 1 ms hrtimer poll on it. That was established about `PC_DONE` and **never covered
    the DPU pair**: `DPU_0`/`DPU_1` (bits 8-9) mask normally and the interrupt reaches the GIC
    [HW sweep, see [chips/rk3576.md](chips/rk3576.md)]. The poll is a driver choice with a
    price — retiring on the DPU bit at a 50 us period instead of `PC_DONE` at 500 us took the
    submit floor from 1065 to 439 us. Their 1 ms period is slower again. Two task classes do
    raise no DPU completion — pooling, and any output element wider than one byte — so a poll
    or a grace still has to survive as the fallback for those.
  - **Its patch 2 is the same fix as `patches/rocket/089`** (`clk_bulk_data.id` never set, so
    all four entries resolve to the node's first clock). Independently found, thinner
    rationale, and neither posting has landed.
  - **Its device tree has core 1 right**: `0x27708000` with the IOMMU at `0x2770a000`,
    cross-checked against the vendor BSP DT, which is what live silicon reads here. The
    `0x27710000` placement recorded in these notes as wrong belongs to the gahingwoo series,
    not this one — with two RK3576 RFCs now in circulation, "the RFC" needs qualifying.
  Its stated open items — the NPU power-domain chain status never asserting after power-off,
  whether the RKNN BIU resets belong to the power domain, and the boot firmware's orphaned
  IOMMU page fault — are the ones `patches/rk3576/npu/0002`-`0005` already address.

- **gahingwoo "Mainlining the RK3576 NPU"** (blog `gahingwoo.github.io/posts/rk3576-npu-mainline/`
  + repo `github.com/gahingwoo/linux-rk3576-npu`: `notes/provenance.md`, `notes/rk3576-npu-values.md`,
  `extract/extract-npu-values.sh`). A sibling-SoC (RK3576, 2-core, **16 CBUF banks** vs our 12)
  mainline-`rocket` bring-up. **Methodology** worth borrowing: capture the
  vendor command stream by building a 1-conv ONNX → convert with `rknn-toolkit2` → walk the `.rknn`
  for the 64-bit command words → decode per unit (an alternative to our Mesa-Teflon capture that may
  expose ops Teflon never emits — e.g. **pooling**, for the on-NPU PPU work); and an
  `extract-npu-values.sh` that auto-derives the platform constants (power-domain / clock / reset IDs,
  GRF base, PVTPLL, OPP table, per-core MMIO bases, IRQs, QoS) by grepping the kernel DT-bindings +
  TF-A BL31 — adaptable to RK3588 (s/rk3576/rk3588/) to auto-document our clock/volt patch provenance.
  **Cross-confirms our findings** (all independently): (1) **IOMMU attach-once / detach-on-power-down,
  not per-job** == our keep-attached patch ([iova-and-multicore.md](perf/iova-and-multicore.md)); (2) **ping-pong
  producer/consumer register groups** (`S_POINTER`), executer reads the *consumer* group, misalignment
  → stale geometry → zero/garbage output, needs per-job re-init — the **mechanism behind our**
  delta-regcmd negative ([regcmd-task-model.md](encodings/regcmd-task-model.md)); (3) **requant is a
  right-shift** whose magnitude is load-bearing (vendor 26-bit vs a wrong 14-bit → saturation to
  black/white) — corroborates our per-scale QNNPACK shift in the conv int8-out path
  ([out-cvt-converter.md](encodings/out-cvt-converter.md), bit-exact vs Teflon); (4) **per-channel
  zero-point correction in a weight-buffer tail** (8-OC groups, 64 B = 8×32-bit + 8×16-bit + 8×16-bit,
  the 16-bit holding `128 − weight_zp`, term `(128−wt_zp)·input_sum`) == our Option-D uint8 recenter +
  box-sum; (5) the **`dt_wr`/`dt_rd`/`wt_rd` byte counters are readable on
  RK3576** — exactly as our [hw-byte-counters.md](perf/hw-byte-counters.md) table predicts (rk3576
  config wires `0x2234/38/3c`; rk3588 nulls them and that page hard-locks) → **does not reopen our
  RK3588 negative**, it confirms the sibling asymmetry. Net: strong independent validation of the
  shared NVDLA-derived IP, plus two transferable scripts; little is usable *as-is* (RK3576 register
  map is shifted/re-packed, different clock/power tree).
  The load-bearing documents in it are the CNA and CORE/DPU maps and the closed-form `predict.py`;
  `vendor_regcmd_full.txt`, a **complete 139-entry vendor register program** (CNA + CORE + DPU +
  RDMA) that an RK3576 emitter can be diffed against off-device; `FINDINGS-FLOATSURFACE.md`; and
  `MATMUL-PIPELINE-ANALYSIS.md`, which confirms RK3576 matmul is the same CNA→CSC→CMAC pipeline in
  FC mode (no separate GEMM unit) and reports the per-power-session "cold-start consume-arm" wall
  it read as a hardware arm. Reproducing that on our own encoder and submit path placed it in the
  driver instead — see [chips/rk3576-regcmd.md](chips/rk3576-regcmd.md).
  The repo has since grown well past the register maps and is worth re-reading as a whole: a
  28-patch kernel series, a Mesa fork carrying an RK3576 Teflon conv2d, a `replay/`
  capture-and-replay harness that runs the same regcmd through both `rknpu` and `rocket`, and
  `FINDINGS.md` — a 2900-line chronological RE log that keeps its own reversed verdicts.
  `CHAINED-CMAC-STOPPING-POINT.md` is the falsification-ledger writeup of the same
  per-power-session wall, parked 2026-07-10 with the conclusion that the consume-arm is
  internal cold-start sequencer state reachable only from vendor RTL. **They independently
  found and fixed the 16-bit `pc_task_number_bits`** (`WRITEL-AUDIT.md`; patch 0028 writes
  `(0x7 << 16) | task_count`), so that half is common ground — both stacks run n-task jobs in
  one hardware kick (ours via `DRM_ROCKET_JOB_BATCHED`, `patches/rk3576/npu/0015`-`0016`;
  theirs in `charsiu` at 32 chained tasks per job). `FINDINGS.md`'s "even task 0 computes
  nothing when task_number=29" describes that log's own submit path, not a hardware bound.
  One structural blind spot in the parked ledger is worth knowing when reading it: every
  experiment in it varies the job that comes out empty, never the job BEFORE it — so the
  wide-output poisoning, a property of the preceding submit, is invisible to it however
  exhaustive it is. `charsiu`'s harness does not share the blind spot: its bisects judge a
  following job run in a separate process (see its entry below). Blog moved to
  `blog.gahingwoo.com/posts/rk3576-npu-mainline/`. Our draft give-back is
  `../RK3576-REPORT-FOR-GAHINGWOO.md` (private, unsent); its chaining and OUT_CVT sections
  are superseded — `charsiu` chains multi-task jobs and places the OUT_CVT triple at
  `0x40ac/0x40b0/0x40b4` itself — so what remains to send is the `RK3576_CNA_MAP.md`
  corrections, the wide-output poisoning, and the float-mode three-register condition.

  Their upstream series reached **v7 on 2026-08-12**, 10 patches, `accel/rocket: RK3576 NPU
  (RKNN) enablement`
  ([cover](https://patchwork.kernel.org/project/linux-rockchip/cover/20260812094106.1391698-1-gahing@gahingwoo.com/)).
  It retracts the "the completion interrupt never reaches the GIC" premise its v3-v6 carried and
  attributes the whole of it to the `PC_TASK_CON` width, so polling, the hrtimer and every
  alternative completion path are gone and the RK3576 retires on the DPU interrupt like the
  RK3588 — the open question that reopens for us is in
  [chips/rk3576-regcmd.md](chips/rk3576-regcmd.md). Two of their patches land on ours: `01/10`
  is `patches/rocket/090` hunk for hunk, and `08/10`'s `PC_TASK_CON` word is `0x70001`, the same
  value `rk3576/npu/0008` writes by shifting the triple. Their evidence is one convolution
  submitted three times, and their Teflon userspace is conv2d-only, so neither the pooling
  programs nor the wide-output writers that raise no DPU completion at all are reachable from
  it.

- **gahingwoo `charsiu`** (`github.com/gahingwoo/charsiu`, GPL-2.0-or-later, first commit
  2026-08-14). An open LLM runtime for the RK3576 on
  mainline `rocket` — the same architectural bet as `rocket-userspace` + `ggml-rocket` on the
  sibling part, and it names both plus these notes as its stated starting point. Every number
  below is theirs (ROCK 4D, their v7-lineage kernel, 2026-08-14/15) and none is reproduced on
  our board.
  **The load-bearing instrument is `tools/rkllm_regcmd.py`**: a vendor `.rkllm` carries the
  register-command streams the closed stack submits, and the script reads the whole dispatch
  plan out of one offline — no board, no vendor runtime. Its first reading
  (`docs/vendor-dispatch.md`, Llama-3.2-1B-Instruct-w4a16): 21,532 streams over 1,061 distinct
  shapes, 8,808 convolutions + 12,724 DPU-only; precision by role — int4 projections all at
  M=1 (3,752 dispatches), fp16 attention at M=32-48 against 128 precompiled KV-length buckets
  (one per 32 tokens of context), an int8 LM head as forty 2048x8160 pieces; every projection
  split across the two cores by output channel; and the FFN down-projection split on BOTH
  axes at per-piece K=4096 — inside the 4608 slice bound measured here, and the vendor's own
  route around a K that does not fit one slice (the shape class our matmul entry refuses at
  K>4608).
  **Where it lands on our findings:** M=1/2/3 exact through the open driver (seven 1x1 convs
  at projection shapes) independently corroborates the no-M-constraint fact in
  `rocket_matmul_rk3576.c`; 32-task chained jobs at ~26.3 us/task marginal (~172 us/submit
  removed) corroborate the one-kick mechanism `patches/rk3576/npu/0015`-`0016` ship; and
  their fitted cost `us/task = 26.3 + weight_MB * 84.3` (11.9 GB/s, M nearly free, a second
  core ~5% WORSE at these shapes) is the weight-fetch-bound reading the platform envelope
  here predicts.
  **What it claims that is unmeasured here:** NPU decode is viable on this part — the vendor
  ships M=1 decode (~13 tok/s on that board and model), their arithmetic projects 11.8 tok/s
  int8 / 22.7 int4 for Llama-3.2-1B, and their stated deciding measurement is one projection,
  NPU against four A72 cores, at M=1 and M=32. That challenges the decode-on-CPU default this
  stack inherited from the RK3588 [their measurement + projection, untested here].
  **Their open defects, and what bears on them:** w4a16/int4 does not compute — their probes
  fit the output as `((int16)fp16bits(w) * (int16)fp16bits(a)) >> 16`, 18/18 measured points
  exact, i.e. the fp16 bit patterns multiplied as signed integers — consistent with a
  partially-set float mode, which on this part is three registers moving together
  ([chips/rk3576-regcmd.md](chips/rk3576-regcmd.md)) [expected, unverified against their
  stream]. And a w4a16 job carrying the vendor's values in RDMA `0x5034`/`0x5044` leaves the
  next job timing out — a next-submit hazard with a different signature from the wide-output
  poisoning, recorded beside it in [chips/rk3576.md](chips/rk3576.md).
  **The reader run here on a SECOND model** (Qwen3-0.6B-Base-rk3576-w4a16-grq v1.2.3,
  725 MB, HF `MichaelAndrewFischer`, 2026-08-18) separates model shape from runtime policy:
  23,540 streams (19,380 conv + 4,160 DPU-only), M again in {1, 32, 64, 96, 128} with M=1
  dominating (14,280), and the M=32-128 program counts nearly identical to Llama-3.2-1B's
  (1108/820/732/676 against 1108/856/728/672) — so the KV-bucketed attention-program
  population is CONVERTER POLICY, not model shape. Qwen3's FFN down-projection (K=3072,
  inside the 4608 slice bound measured here) dispatches WHOLE with only the two-core
  output-channel split — the vendor K-splits only when forced past its slice bound.
  **A THIRD model read settles the split policy and the combine mechanism**
  (Qwen3-1.7B-Base-rk3576-w4a16-grq v1.2.3, 1.6 GB, same HF author, read 2026-08-18,
  offline, both files re-fetched): 28,804 streams (22,740 conv + 6,064 DPU-only). The
  1.7B's FFN-down (K=6144) never dispatches whole — every instance is a 4096-piece plus a
  2048-piece, oc-halved (`blk.N.ffn_down.weight_rkllm_spilt_0/1` in the file's own tensor
  names) — so the vendor's per-dispatch K bound is exactly **4096, greedy chunks**: the
  one rule that fits Llama-1B's 8192 = 4096+4096, this 6144 = 4096+2048 (not 3072+3072),
  and the 0.6B's 3072 whole.
  **The combine is ON-NPU: a dedicated DPU-only elementwise program, one per split pair.**
  At M=1 every `[4096-piece][2048-piece]` pair is followed by exactly one (1344 of 1344);
  prefill sites carry the same program in pixel-bucket variants. Its primary operand comes
  from memory (`0x400c = 5` where a conv carries `0x40000004`), its second through
  DPU_RDMA (`0x5xxx` words live — including the exact `0x5034 = 4000004c` /
  `0x5044 = 000280a1` values of the w4a16 next-job hazard in
  [chips/rk3576.md](chips/rk3576.md)), both at the pair's output geometry (oc-half x M
  pixels), and its `0x4010` input width code is **5** — the 32-bit code the
  coefficient-A fp32 readback anchored — where the attention-interior EW program carries
  2. So the partials are FLOAT, separately scaled (`ffn_down` is the only projection with
  a `_C_secondary` coefficient group, one multiplier set per K-piece), and nothing in the
  mechanism touches int32: no integer GEMM in any of the three read models exceeds K=4096
  (the int8 LM head is K=2048, whole). The split pieces' conv programs are
  register-identical to never-split projections outside pure geometry — the combine is
  invisible to a per-register diff and was read from the program sequence and counts. The
  program type is not exclusively the combine: ~16 instances in each file follow a
  `2176x32` fp16 op, so a count test alone would misattribute — adjacency separated the
  uses.
  **Both former decode frontiers are read.** The weight-bits-0 streams are chained
  no-weight-fetch DELTA task programs: a prefill M bucket is pixel-chunked (0.6B:
  53+53+22 = 128; 1.7B: 40+40+16 = 96) and the last chunk restates neither the weight
  registers nor the full geometry, which the reader's bits formula misparses — 228 on the
  0.6B, 340 on the 1.7B, every one adjacent to same-`ic` full programs whose pixel counts
  it completes (the 1.7B's 672-count "bits 4096" class is the same thing with a weight
  fetch: trailing K-pieces as delta programs). And the DPU-only share collapse (59% of
  Llama's streams to 18-21% on both Qwen3 files) is CONVERTER/RUNTIME POLICY, not
  architecture: the attention-interior EW population is identical across the two Qwen3
  files (4120+24 of kind `0x4010=40000002`, KV-bucketed), Llama's dominant per-projection
  EW kind (`a0000002`/`00023333`, 8268 streams) has no counterpart in either — the Qwen3
  files' conv programs carry those same two values inline, and the files hold no norm
  tensors at all, so the decode path's norm/residual/activation work is off-NPU in
  v1.2.3. Qwen3 needs MORE norm work than Llama (QK-norm), so architecture cannot explain
  the disappearance; the Llama file's converter version is unstamped, so this is argued
  from that direction, not read off a version field.

- **gahingwoo `kiln`** (`github.com/gahingwoo/kiln`, GPL-2.0). The VENDOR RKLLM/RKNN stack run
  on a mainline kernel (7.1.3+): the GPL `rknpu` driver built out of tree plus a small kernel
  patch set, an installer, and an OpenAI-compatible server. `charsiu`'s measuring stick, and
  the live-capture harness (`capture/rknpu-regcmd-dump.patch`) that complements the offline
  `.rkllm` reader. Two of its kernel-side facts land here, read against our own DTS
  (2026-08-18): **(1) The "two IOMMUs / four MMU banks, mainline drives one" claim is about
  the vendor `rknpu` AGGREGATE node** — one device listing both iommus, the second left
  unattached, so the second core reads IOVAs as physical addresses. It does not transfer to
  `rocket` as-is: the RK3576 NPU is two iommu instances of two banks each (`rknn_mmu_0` at
  `0x27702000`+`0x27702100`, `rknn_mmu_1` at `0x2770a000`+`0x2770a100` — our
  `patches/rk3576/npu/0007` DTS), each core device carries its own, and mainline
  `rockchip-iommu` iterates every bank of an attached instance — so single-core `rocket`
  already manages both banks of core 0, and this mechanism is NOT a candidate for the
  two-jobs-in-flight corruption there. **(2) Their `0007`/`0008` iommu patches (authored by
  Jiaxing Hu, the RK3576 RFC author) name a mechanism whose symptom we log**: a bank left by
  firmware with `PAGE_FAULT_ACTIVE & !STALL_ACTIVE & IDLE` silently drops
  `CMD_ENABLE_STALL` and delays the other banks past the poll timeout — the recurring
  `rk_iommu ... Enable stall request timed out` on the H96 during long probe runs matches
  that failure path exactly [symptom match; the orphaned-fault status bits unverified on our
  board]. The two patches sit in shared `rockchip-iommu` code and are candidates for
  `patches/rk3576/npu` evaluation.

- **gahingwoo `mesa-rk3576`** (`github.com/gahingwoo/mesa-rk3576`, `rk3576` branch on Mesa
  25.3.0). Their Teflon/`rocket` RK3576 driver as a maintained Mesa fork (~104 branch
  commits, active 2026-08): conv paths, depthwise in 64-channel groups, CBUF
  allocation/reuse, DPU `0x4044`/`0x4050` handling, a `ROCKET_REG_SET` late-override knob. A
  second independent RK3576 encoder to diff against beyond the register maps in
  `github.com/gahingwoo/linux-rk3576-npu` — for the RK3576 what Mesa's `rkt_regcmd.c` is for the
  RK3588. Not checked out; fetch on demand.

- **Chaoyi Chen (Rockchip), `PC_TASK_CON` bit assignment** —
  [lore.kernel.org](https://lore.kernel.org/all/4f300b78-d96d-4d98-8819-dc292b0c9b97@rock-chips.com/).
  A vendor engineer's reply on `linux-rockchip` giving the RK3576 field layout, including the
  name of the control the RK3588 description marks reserved: `BIT(18) task_last_layer_clear`.
  The only authoritative statement of that register we have; everything else about it here was
  read off the silicon. See [chips/rk3576-regcmd.md](chips/rk3576-regcmd.md).

## Userspace stacks we learned from

- **johanvdb/librocket** — FOSS userspace fp16 matmul (as 1×1 conv) on mainline
  `rocket`, "for GGML projects." It combined Jasbir Matharu's `rk3588-npu` register
  headers with the Mesa regcmd format, and is the starting point our kernel-access layer
  was built from. Working single-task fp16 matmul; ships a kernel NULL-deref iommu patch.
  **No** tiling/quant/multicore/int8 — those, and the rewritten shim, are ours. (Its
  `rocket_interface.c` passes `timeout_ns` raw, an absolute-deadline bug we fixed.)

- **johanvdb/ggml `rocket-backend`** — a ggml backend *skeleton*. `*-matmul.cpp`
  returns -1 (CPU fallback), `*-dequant.cpp` is TODO. Scaffolding, not a working
  model path — useful as a structural reference only.

- **mtx512 / jas-hacks `rocket-userspace`** — the original RKNN-reverse-engineering repo
  (blog: jas-hacks.blogspot.com, "RK3588 reverse engineering RKNN"). Its
  `gen_matmul_task` (matmul-as-1×1-conv over NVDLA CNA/CORE/DPU blocks) is the
  most complete register config to start from — but it targets the **proprietary
  rknpu ioctls** (5.10 BSP), so driving it through `rocket` requires swapping the shim
  and adding the DPU-RDMA block it omits.

- **poad42/opennpu_rk3588** (MIT, first commit 2026-08-02) — an ONNX/JAX-to-RK3588 compiler
  and runtime whose PJRT plugin makes the NPU a first-class JAX device
  (`jax.devices()` → `[npu:0]`). It drives the **vendor `rknpu` BSP driver** on a 6.1 vendor
  kernel (`/dev/dri/card1`; `MEM_CREATE`/`MEM_MAP`/`MEM_SYNC`/`SUBMIT`/`ACTION`), so its
  runtime, ioctl layer and kernel patches do not transfer to the mainline `rocket` path — the
  register encoding does. Same author as **poad42/smolvlm_rk3588_full_npu_native** below.
  The useful subset is docs, `cna_matmul.c`, `lm_forward.c`, and the four kernel patches.

  **It is two stacks and only one is worth reading.** `pjrt_c/cna_matmul.c` emits the CNA
  descriptor **by formula** — 112 `uint64` entries computed from M/K/N, plus a weight cache
  that preloads all 144 GPT-2 matrices at init. `matmul_tmpl.h` (1.0 MB), `tanh_tmpl.h`
  (384 KB) and `relu_tmpl.h` are **captured vendor programs replayed with the DMA addresses
  patched**, which is why their op table reads "MatMul supports 5 GPT-2 shapes."
  `docs/ARCHITECTURE.md` says the templates were "captured once from vendor toolkits via
  ioctl tracing", while the README's provenance section says no proprietary code was involved.

  **The formula path agrees with ours register for register** [source-confirmed, read from
  `cna_matmul.c`]. Entry encoding `(op << 48) | (val << 16) | reg`, block tags `0x0201` /
  `0x0801` / `0x1001` for CNA / CORE / DPU, and CNA `0x100C`, `0x1010`, `0x1014`, `0x1020`,
  `0x1024`, `0x1030`, `0x1040`, `0x1070`, `0x1084`, `0x1088`, `0x1110` with DPU `0x4020` as
  the destination base. Their `0x1040` leaves bits [10:8] zero — an independent instance of
  the `FC_DATA_BANK` = 0 rule in [encodings/cbuf-reuse.md](encodings/cbuf-reuse.md) — and
  they write the `FC_DATA_SIZE0/1` pair as `(1<<16)|M` and `K`, as `npu_regcmd.c` does.
  Three machine facts match. **`M % 4`**: their guide states "M=1, 2, 3 produce completely
  wrong results" and `npu_cna_cache_run_m()` pads to `M4 = 4` before every submit — the same
  trap and the same workaround as [matmul-as-conv.md](matmul-as-conv.md), and a second RK3588
  witness for the row in [chips/porting-patterns.md](chips/porting-patterns.md) where the
  RK3576 carries no M constraint. **CBUF = 12 banks × 32768 B.** And `nbuf_size=0` with
  `CONFIG_ROCKCHIP_RKNPU_SRAM` unset. Their host weight scatter (`weight_fp16_off`, 16
  outputs per group, 32 inputs per group) and feature scatter (`feature_data_off`, C2=8 fp16
  input, C2=4 fp32 output) are a second independent statement of the native cube layout, and
  that they scatter on the host at all corroborates the no-on-chip-layout-conversion finding:
  the README's "reads W directly as fp16 — no reordering" describes the absence of
  quantization, not of tiling.

  **Their bank split allocates no slack bank** — `fd_banks = ceil(M·K·2 / 32768)`,
  `wt_banks = 12 - fd_banks` — so that path is exposed to the feature-DMA overread that reads
  one CBUF bank past the allocation here [expected; not observed on their driver].

  **Three stated impossibilities are bounds of their harness, not of the silicon.**
  "Multi-core model-parallel matmul is impossible" follows from a position-locked 28 KB
  template block and a ~3.5 MB scratch DMA window; this stack runs 3-core per-fd. "K > 768
  requires tiling" is their bank split's own consequence — `cna_matmul.c` clamps `Kt` to 768
  and accumulates the tiles on the host. And "w8a8 … 16% error, 256 levels insufficient for
  768-wide" is the per-tensor-scale wall that a Hadamard rotation plus a per-output-channel
  requant closes.

  **Their DMA ceiling belongs to that path.** `ARCHITECTURE.md` fits
  `t = 0.08 ms + bytes / ~1000 MB/s` and calls 1 GB/s "the NPU internal DMA engine's hardware
  limit"; the README's results table says 3.0 GB/s. The two do not reconcile with each other,
  and both sit far under what the regcmd path here sustains.

  **The decode claim, and why it is the weaker of the two on record.** GPT-2 124M at
  **28 ms/token (36 tok/s)**, KV-cached, tabled as "NPU 1.4×" — nominally a second challenge
  to the decode-on-CPU default after the RK3576 vendor's M=1 decode. Its comparison column is
  **RKLLM at 33 ms/token**, which is Rockchip's NPU runtime rather than a CPU measurement; the
  timing (84 matmuls × 0.32 ms = 21 ms/token) closes only at 3 GB/s, where their own 1 GB/s
  model puts a 236 ms floor under GPT-2's 144 matmuls; and `lm_forward.c` keeps the KV cache,
  attention, softmax, LayerNorm and GELU on the CPU, offloading only the four per-layer
  projections, at a padded M=4. [their claim, internally inconsistent, untested here]

  **What it has that nothing else here does:** a working **PJRT plugin** (`pjrt_c_api.h`
  v0.112 pinned to jaxlib 0.10.2) — a frontend shape none of `ggml-rocket`, `tflite-rocket` or
  `ort-rocket` covers. Its four kernel patches — batch submit, an IOMMU lock-free domain fast
  path, a `kmem_cache` for job structs, and fence fds — are an independent rediscovery of
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
  the CPU under OpenMP — against cosine 0.999998 for the full block on the FOSS path
  ([encodings/siglip-encoder.md](encodings/siglip-encoder.md)).

- **Mesa Teflon on `rocket`**
  (rpardini/mesa-teflon-etnaviv-rocket-docker; BredOS wiki `NPU/rocket.md`). Upstream
  Mesa's Teflon TFLite delegate is the public FOSS-rocket baseline: kernel ≥ 6.18
  `CONFIG_DRM_ACCEL_ROCKET`, `/dev/accel/accel0`, `libteflon.so` via
  `tflite.load_delegate`. Envelope: **quantized uint8 CNNs only, conv + EW-add + fused
  ReLU, single-core, no SiLU, no transformer**; AVGPOOL/RESHAPE/SOFTMAX fall to CPU.
  Perf ~13–17 ms MobileNetV1 (≈ 3–4× over CPU ~48 ms). `rocket-userspace`/`tflite-rocket`
  run 3-core per-fd, fp16/int8/int4, SiLU/GELU and transformer blocks, and MobileDet.
  rpardini also
  carries per-board NPU-regulator DT patches (CM3588-NAS, NanoPC-T6, **Turing RK1**) —
  the voltage wiring the `patches/rocket` clock/volt coupling depends on.
  [source-confirmed]

- **llama.cpp ggml NPU-backend discussion** (ggml-org/llama.cpp#8111) — an upstream
  attempt at an RK3588 (and Tenstorrent) ggml backend that **stalled** (author pivoted to
  Tenstorrent for its open stack). It validates our design: the hard path they hit —
  managing device buffers, needing dtype/layout at *allocation* time, no
  offset-subbuffering, the **per-fd 4 GB IOVA limit** — is exactly what `ggml-rocket`
  sidesteps via the **BLAS-backend model** (host/CPU buffers, offload only `MUL_MAT`
  through `graph_compute`, pack+DMA inside `rocket-userspace` per call). slaren's guidance
  there (`set_tensor` carries `tensor->type`; `get_alloc_size` / `get_max_size` /
  `get_alignment` for device-buffer backends) is the API surface we deliberately don't
  need. The 4 GB-window / no-subbuffer constraint is real and handled one layer down in
  `rocket-userspace` IOVA management (`rocket_prepacked_int8.c` escape check), not at the
  ggml buffer-type. [source-confirmed]

## Quantization / coherence references

- **clehaxze.tw gemlog** — "Benchmarking RK3588 NPU matrix multiplication
  performance" (eps 1–2; the 2024-02-14 post has the hard numbers). Same silicon,
  RKNN-measured (an upper-reference): fp16 ~900 GFLOPS peak, int8 ~2× fp16, int4 ~4×
  (in MAC terms), K-spilling past ~1024.

- **Martin Chang, "porting LLM to RK3588"** (RWKV-on-RK3588 talk). The
  **native-K-reduction** insight: the conv reduces over K
  in one pass up to the CBUF limit ("Max K=2048 @ FP16" in one pass) — nobody used
  the DPU eltwise for K-accum. Reframed our tiling. Also the clearest external statement
  of the **decode split**: NPU is good at MatMul, bad at GEMV (M=1); CPU GGML wins decode
  (83 ms vs 61 ms/token) because llama.cpp's NEON GEMV is already bandwidth-saturating.
  See [perf/decode-gemv.md](perf/decode-gemv.md).

- **Hummingbird+** (Li et al., *"Hummingbird+: Advancing FPGA-based LLM Deployment from
  Research Prototype to Edge Product"*, FPGA '26, doi:10.1145/3748173.3779189) — a
  dedicated **FPGA** GEMV engine (Zynq UltraScale, 140 DSPs, <1K LUTs, 272 GOPs) for
  edge LLM decode. Consulted on the question "are there GEMV optimizations we are
  missing?" The answer is no for fixed silicon: its speedups (DSP pre-adder operand
  packing, DDR LUT-mux elimination, DOT/AXPY mode switch, W4/KV8 dual precision) are
  reconfigurable-datapath microarchitecture with no analogue in the RK3588's fixed
  convolution pipeline. What transfers is bandwidth-reduction at the model/format level
  (4-bit weights + 8-bit KV, MoE) — already available in stock llama.cpp. It also cites
  RKNN-LLM at "~10 token/s on a 3B on RK3588," i.e. even vendor on-NPU decode is
  bandwidth-bound, not a CPU-beating win. See [perf/decode-gemv.md](perf/decode-gemv.md).

- **SHARD** (Mohan et al., *"SHARD: A Compatibility Framework for Deploying Transformer
  Models on Edge NPUs"*, EuroMLSys '26, doi:10.1145/3805621.3807618; also the
  amohan.dev blog) — deploys the **SigLIP-B/16 vision encoder** (93 M params, the
  SmolVLM-256M front-end) on RK3588 through **rknn-toolkit2**, so it is a vendor-
  toolchain *workaround* (graph sharding, GELU-approx / LayerNorm-decompose
  legalization, fusion barriers — all RKNN-steering a regcmd path doesn't need). What
  transfers: (1) the `0xe010 "REGTASK Overflow"` is an undocumented **13-bit
  instruction-register width limit — operand indices > 8191 fail** [source-confirmed],
  the *same* 13-bit register class as our `DPU_DATA_CUBE_WIDTH` 8191 corruption [HW
  sweep] (independent cross-validation; a **general operand-index ceiling**, not LUT-
  only — see [encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md)); (2) a
  **32 KB per-op scratchpad** → ≤ 16384 fp16 elems/op (their attention tile 256×64),
  the tiling discipline we already follow; (3) **Sandwich λ-scaling** (host pre×0.1 /
  post×10.0) keeps fp16 off a "saturation cliff" (cosine 0.98→0.11 by layer 5 without
  it), and the paper's finding that **AWQ fails "because the error stems from activation
  outliers, not weight sensitivity"** validates the Hadamard activation-rotation choice.
  Numbers (Orange Pi 5 Max): SHARD 2.24 s @ cosine 0.95 vs RKNN-FP16 19.63 s @ 0.64
  (CPU-fallback transpose) vs RKNN-INT8 1.40 s @ 0.02 (collapsed) vs CPU-FP32 30 s — a
  vendor baseline for the ViT-encoder primitives (MHA / LayerNorm / GELU / FFN) the
  rocket stack runs on-NPU.

- **poad42/smolvlm_rk3588_full_npu_native** — a concrete deployment of the same **SmolVLM-256M**
  front-end on the RK3588 NPU via the **proprietary** `rknn-toolkit2` + RKLLM bindings (no stated
  OSS license on the main code), i.e. SHARD's model on the vendor toolchain — the code does not
  transfer to the rocket path. What corroborates SHARD independently: it splits the vision encoder
  into **24 shards (12 layers × 2 blocks) across NPU cores 0–2**, FP16/INT8 hybrid, and wraps each
  NPU block in input/output scalers ("**Sandwich Quantization**" / InputScaler) to keep fp16 off
  the saturation cliff — the same sandwich-scaling lever SHARD formalizes as λ-scaling, independent
  confirmation that this exact front-end needs activation rescaling to survive NPU quant. It also
  tiles attention into **32×32 blocks with small-chunk transposes** purely to dodge the **RKNN
  compiler's** transpose handling — a constraint the regcmd path does not share. No published
  accuracy or perf numbers; a parallel effort to benchmark against, given our SigLIP-B/16 encoder
  runs the full block on the FOSS path at **cosine 0.999998**
  ([encodings/siglip-encoder.md](encodings/siglip-encoder.md)).

- **r/RockchipNPU thread** — field reports
  that int8/int4 LLMs on RK3588 need **INT8_HADAMARD / INT4_HADAMARD** for coherence
  (e.g. gpt-oss-20b). Validated our Hadamard-is-mandatory finding.

- **rk-llama.cpp issue #9** (the proprietary rknpu2 stack) — independently confirms
  the *direction*: "Q8_0 maps well to RKNPU W8A8" → Q8_0 prefill +200–400%; Q4_0
  maps poorly; decode is memory-bound (matches our CPU-decode split). Their Q8 win is
  **operand bandwidth at a lower dispatch floor**, not a capability the open path lacks:
  RKLLM quantizes to W8A8 (8-bit weights *and* activations), halving the bytes moved for
  both operands on a path that is dispatch/DMA-bound — and their own Gemma int8 runs at
  40–58% NPU utilization, so their stack is not MAC-bound either. The one thing it is
  *not* is native on-device int32 K-accum: that is a hardware dead-end for every stack
  (the DPU-EW ALU is float-only — [encodings/k-accumulation.md](encodings/k-accumulation.md)).
  See [perf/not-mac-bound.md](perf/not-mac-bound.md) §"The proprietary stack's int8 win is bandwidth".

- **kevbuh/rk3588** — notes/datasheets confirming int4/int8/int16/fp16/bf16/tf32
  support on the silicon (used to sanity-check the precision menu).

## Quant-coherence primary research

- **QuaRot / SpinQuant** (Hadamard-rotation quantization) and **SmoothQuant** — the
  principled fixes for activation-outlier-driven int8/int4 gibberish. We use a
  Kronecker Hadamard (H_{2^k} ⊗ H_60 via Paley for Gemma's non-power-of-2 K). See
  [encodings/k-accumulation.md](encodings/k-accumulation.md); measured perplexity with
  W8A8 + Hadamard tracks fp16.

- **AWQ** ([mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)) — activation-aware
  *weight* quant: per-group scales + protect the ~1% salient channels (largest activation
  magnitude). Ships the reusable scale-search, not just weights. The weight-side lever for our
  int4 quality (W4A16 as-published; complements Hadamard on the activation side). → the int4 work.

- **Outlier Suppression+** (arXiv:2304.09145, Wei et al. 2023;
  ModelTC/Outlier_Suppression_Plus) — per-channel **shift** (kills *asymmetric* activation
  outliers) + per-channel **scale**, both migrated into adjacent layers (free at inference).
  Near-FP at 8/6-bit. Complements Hadamard; the asymmetric-shift fits our asymmetric uint8
  detection weights. → the int4 + MMSE range-setting work.

## Transformer softmax / exp on NPUs

- **Attention Distribution-Aware Softmax for NPU-Accelerated On-Device Inference of LLMs**
  (Sadheerthan et al., *Electronics* 2026, 15, 1312).
  Confirms softmax is *the* NPU transformer bottleneck because NPUs lack a native `exp` unit, and
  proposes a distribution-aware, variable-degree LUT approximation of `exp` (PSO-learned non-uniform
  segments snapped to a 128-bin grid), cutting exp-kernel cycles-per-call ~18.5% vs uniform Degree-4.
  **Relevance:** it optimizes the exp *kernel compute*; our fused-encoder softmax is bound by host↔NPU
  *transfers* instead (proven: a slower host `expf` softmax ran 3× faster than the on-NPU LUT path —
  see [encodings/whisper-encoder.md](encodings/whisper-encoder.md) §in-model). Its quant-domain,
  clamp-`[-20,0]`, in-domain row-max recipe is the design reference for a fully-on-NPU **resident**
  softmax (no host round-trip) — the lever if Whisper encoder fusion is ever pursued for perf.

- **YOLOv10-on-RK3588 latency** ([THU-MIG/yolov10 #115](https://github.com/THU-MIG/yolov10/issues/115))
  — YOLOv10s is ~2.7× *slower* than YOLOv6/DAMO-YOLO on RK3588 (32 vs 12 ms @416) despite fewer
  params: its attention/PSA blocks are NPU-hostile. A detection-pillar datapoint (same
  attention/softmax-fights-this-NPU theme); relevant if the detection pillar moves toward YOLO.

- **LSH9832/edgeyolo** (`cpp/rknn`) — EdgeYOLO (anchor-free, YOLOX-style) deployed on RK3588 via
  the **proprietary RKNN runtime**, with decode + NMS wrapped inside the `RKNN::YOLO` class (not a
  reusable, backend-agnostic postprocessor). One transferable datapoint: the **LeakyReLU** variant
  (Tiny-LRELU) runs **65 FPS** vs **24 FPS** for the **SiLU** variant at 384×640 / int8 / 3-core,
  the author attributing the gap to "SiLU activation layers" — a ~2.7× activation-driven cost, the
  same NPU-hostile-op theme as the YOLOv10 PSA datapoint (above). If the detection pillar moves
  toward YOLO-family graphs, prefer LReLU over SiLU for throughput; the number is an RKNN-path
  reference, not our DPU-LUT path ([encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md)).

## NPU generation & throughput (decode / multi-instance)

- **"Accelerating OpenPangu Inference on NPU via Speculative Decoding"**
  (arXiv:2603.03383, Dai et al. 2026; wujing215/OpenPangu7B-with-Medusa) — Medusa
  multi-head draft (no separate draft model) + **static tree attention** + zero-copy retrieval to
  fit the NPU's static-graph execution; 1.35× short-seq, long-seq memory-BW-bound. Closest prior
  art for speculative decoding on the NPU; the static-tree design is the part that ports to a fixed-graph NPU.

- **leafqycc/rknn-multi-threaded** + **HN 48527630 (alebal123bal)** — two working RK3588
  multi-instance inference pools: thread pool + queue + one RKNN context per core, round-robin,
  preallocated buffer pool, RGA preprocessing. 2.6× (YOLOv5s) / 31→46 FPS (full ISP→RGA→YOLOv8n
  pipeline). The reference architecture for the multi-instance throughput pool (Frigate multi-camera).

- **"我以为 NPU 推理到了硬件极限，结果发现是 CPU 频率在拖后腿"** (zhihu, johnjiamzhong/AlertGateway,
  RK3588S, RKNN path; `zhuanlan.zhihu.com/p/2051444846548857552`) — a proprietary-path writeup that
  attributes a "~40 ms is the NPU hardware limit" YOLOv8s number to the **CPU governor**, not the
  NPU: `rknn_run` includes CPU-side submit + IRQ-wait, so it swings 59 → 35 ms (−41%) from the
  CPU governor alone (NPU clock fixed at 1 GHz), and CPU+NPU both pinned to `performance` collapses
  the jitter. Independent corroboration of our CPU-side submit/dispatch floor — see
  [perf/not-mac-bound.md](perf/not-mac-bound.md) §Dispatch-floor reducers and
  [perf/clock.md](perf/clock.md). Methodology only; no FOSS-path numbers (it never leaves rknn).

## The NVDLA ancestor

- **NVDLA hardware manual v1** ([nvdla.org](http://nvdla.org/)) — the only authoritative prose
  about this datapath; Rockchip publishes a register list and no semantics. Which of our
  questions it answers, which it answers *wrongly* for this silicon, and which it cannot reach
  is worked through in [nvdla-lineage.md](nvdla-lineage.md), with a per-page reading list. In
  short: the structure transfers (blocks, the ping-pong register file, CBUF banking and ports,
  CACC's within-one-layer accumulate, the MCIF/SRAMIF arbiters), the arithmetic details do not
  always (it specifies round-half-away-from-zero; the part rounds half to even), and the fields
  that matter most here — `PC`, `DECONV`, the CBUF granule allowance, the per-sign shift word —
  are Rockchip additions it never had, so its silence about them is not evidence.

## Compression

- **NVDLA weight-compression format** ([nvdla.org/hw/format.html](https://nvdla.org/hw/format.html))
  — CWT (packed non-zero weights) + WMB (1 bit/element mask, 128-B aligned) + WGS (per-kernel-group
  byte count). Kernel group 32 (int8) / 16 (fp16/int16) — matches our weight-tile groups. CDMA
  decompresses into CBUF, skipping zero MACs. **Sparsity (pruning) compression** — no win on dense
  weights. Probable layout of the RK3588 CNA DCOMP (DPU==NVDLA SDP ⇒ CNA likely==NVDLA CC).

## Related RK3588 NPU work

- **SHARD** (EuroMLSys'26, doi:10.1145/3805621.3807618) — RK3588 VLM via constraint-driven
  graph rewrite, 8.7× over RKNN. A design precedent for per-shard precision selection
  (→ per-layer FP16 hybrid TODO); its primitives (native GELU/LayerNorm, CBUF tiling, hybrid
  CPU/NPU) are ones the rocket stack also implements.
- **clehaxze gemlog (2023)** — RK3588 NPU per-cycle MACs (2048 int4 / 1024 int8 / 512 fp16);
  notes that RKNN matmul lacks multi-core (the rocket path runs 3-core via per-fd).
- **t-firefly ROC-RK3588S NPU wiki** — vendor RKNN usage only; no multi-core/SRAM/register/driver
  detail.
- **sagi21805/matmul-npu** — a C++/OpenCV matmul wrapper over the proprietary RKNN toolkit2
  (fp16/int8 in → fp16/int8/fp32 out); no regcmd/ioctl/tiling detail exposed.
- **llama.cpp#722 (2023)** — 7B-Q4 on an RK3588 (NanoPi R6s) at ~98 s/token: an early CPU-only,
  microSD/RAM-bound outlier (not representative of current llama.cpp CPU perf). The
  motivational datapoint: RK3588 CPU LLM is impractical at 7B, hence the
  NPU-prefill / CPU-decode split.
