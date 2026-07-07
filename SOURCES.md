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

- **allbilly/rk3588** (`allbilly-npu`), esp.
  `include/rknnops.h`. A higher-level op generator (conv1d/2d, matmul, activations,
  LUTs) using the same Mesa regcmd encoding. Its `float16_alu_op(ALU_ALGO_ADD)`
  encodings are what cracked the **fp16 DPU-EW K-accumulation**; its int-EW
  `EW_OP_TYPE` bit is what we tested (and ruled out) for int32 K-accum. Broader op
  coverage than Mesa — the reference for going beyond matmul.

- **RKNN-Toolkit2** (`github.com/airockchip/rknn-toolkit2`) — the vendor's proprietary
  compile-and-run stack; this project is a mainline alternative to it. Its offline compiler
  output is the reverse-engineering input for the **PPU pooling family**, which the FOSS
  Mesa/Teflon path never emits: a 1-op ONNX pool compiled to a `.rknn` and decoded for the
  PPU / PPU_RDMA register page yields the exact `RECIP_KERNEL = fp16(65536/k)` reciprocal
  format (method in [ppu-rknn-capture/](ppu-rknn-capture/); no vendor artifacts are
  redistributed). Every other encoding here comes from the FOSS Mesa driver plus HW sweep.
  RKNN3 (targeting RK1820 / RK3572) is a different NPU generation, out of scope.

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
  Independently corroborates the **≤16-bit EW operand** limit that kills integer K-accum.

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
  maps poorly; decode is memory-bound (matches our CPU-decode split). **The
  *mechanism* behind their Q8 win is inferred, not proven** — our path measures int8
  *slower* than fp16 (the int32-readback wall), and *something* lets theirs avoid it,
  but we have not isolated which of: native on-device int32 K-accum / weight-DMA
  binding (theirs may be weight-bandwidth-bound where ours is not) / on-chip SRAM
  staging of partials (a second memory interface `rocket` doesn't use). So the numbers
  validate direction but do *not* transfer to our readback-bound path, and the causal
  attribution is hypothesis-level — see
  [perf/not-mac-bound.md](perf/not-mac-bound.md) §"int8 is slower than fp16".

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
