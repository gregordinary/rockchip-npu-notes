# Rockchip NPU reverse-engineering notes

## AI disclosure

The documents in rockchip-npu-notes were authored by AI, primarily Claude Code (Opus 4.8). They
were produced as part of a series of side projects, and are published here in case they are of use
to others. Accuracy of information is not guaranteed.

## About rockchip-npu-notes

These are subsystem-organized, project-independent notes on the Rockchip RK3588
NPU as driven through the mainline `rocket` DRM-accel driver.

Most of what is here is IP-inherent, meaning true of the rknpu and NVDLA-derived block on any
Rockchip SoC that carries it. The values that vary per chip are collected in the per-SoC sheets
under [chips/](chips/): machine parameters, the register offset map, and SoC integration. The
RK3588 is the hardware-validated reference, with the RK3576 and RK3566 tracked there.

They were established by reverse-engineering the hardware on a real device, a Turing RK1 with
32 GB on mainline kernel ~7.1. That work built a FOSS inference stack on top of `rocket`: a
userspace matmul library, a ggml backend, an NPU-clock patch, and a TFLite delegate.

Most of what we learned is not specific to any one of those projects. It is facts about the
silicon and how its register-command interface behaves. That is what lives here.

This repository is for someone running their own compute on the RK3588 NPU through `rocket`, or
any raw-regcmd path. It carries observations, insights and details on:

- Precision encodings and native tile layouts.
- The integer-output `size_e` quirk.
- What the DPU eltwise unit can and cannot accumulate.
- The CBUF operand-reuse bits.
- The MRDMA trap that hangs your first job.
- The per-fd IOVA window.
- The clock that boots at 1/5 speed.

For a start-to-finish walkthrough that ties the driver library, the frontends, and the
kernel patches together, see the [guide](guide/).

## How to read this

Every claim is tagged with how it was established:

- **[HW sweep]**: reverse-engineered empirically by sweeping a value/geometry on the
  real NPU and observing bit-exact vs garbage vs nothing-written. The strongest
  evidence: it is what the hardware actually does.
- **[source-confirmed]**: corroborated by reading an authoritative FOSS source: the Mesa
  `rocket`/Teflon driver, the open NVDLA documentation, or the RK3588 register headers we
  build on (`npu_hw.h`, from Jasbir Matharu's `rk3588-npu`). See [SOURCES.md](SOURCES.md).

Where a fact is HW-confirmed *and* matches a source, both tags appear. Negative
results (things that do not work) are documented as carefully as the positive
ones, which were the most expensive to learn.

## Map

| Doc | Subsystem | The fact |
|---|---|---|
| [chips/](chips/) | per-SoC | machine parameters + register offset map + SoC integration, one sheet per chip (RK3588 validated; RK3576 / RK3566 tracked) |
| [hardware-overview.md](hardware-overview.md) | whole NPU | NVDLA lineage, 3 cores, CBUF 12×32 KB, the precision menu, TOPS vs reality |
| [nvdla-lineage.md](nvdla-lineage.md) | whole NPU | what the ancestor's open documentation carries and where it stops: the block map (and the three blocks the RK does not have: RUBIK, CDP, BDMA), the ping-pong register file (`PRODUCER`/`CONSUMER` == `POINTER`/`EXECUTER`, and why the interrupt bits come in `_0`/`_1` pairs), the CBUF's separate feature-write and weight-write ports, CACC's within-one-layer accumulate, the DDMA/SDMA arbitration knobs nothing writes; **and where it is wrong for this silicon**: NVDLA specifies round-half-away-from-zero, the part rounds half to even; plus the Rockchip additions it is silent about (PC, `DECONV`, the per-sign shift word) |
| [datatypes.md](datatypes.md) | whole NPU | the datatype capability matrix: precision field, output type, MAC rate, and use, per dtype |
| [matmul-as-conv.md](matmul-as-conv.md) | CNA/CORE/DPU | how a matmul is run as a 1×1 convolution; tiling; the data flow; the alignment rules + the feature-height-<4 (M==1 GEMV) break |
| [depthwise-conv.md](depthwise-conv.md) | CNA/CORE/DPU | how depthwise differs from a direct conv: `CONV_MODE=3`+`DW_EN`, `weights_kernels=1`, `size_e=3`, surfaces ×2 |
| [encodings/conv-transpose.md](encodings/conv-transpose.md) | CNA (lowering + HW mode) | the shipping ConvTranspose2d lowers to dilate-input + `rot180(Wᵀ)` + a stride-1 forward conv (the 180°-flip derivation + `pad <= d·(K−1)` constraint), paying `s²` zero-MACs, **but the CNA has a real deconvolution mode and it is live**: `CONV_CON1[16]` `DECONV` gates `CONV_CON3` `DECONV_Y/X_STRIDE`, only field values 1/3/7 do anything (power-of-two strides), and the surfaces sparsify to one live element per channel; geometry not yet decoded |
| [encodings/resize-upsample.md](encodings/resize-upsample.md) | CNA (lowering) | nearest/bilinear resize = a depthwise transposed conv with a box/triangle kernel; the triangle's stride-subsample is a partition of unity ⇒ half-pixel 2-tap bilinear; `C%32` |
| [encodings/precision-field.md](encodings/precision-field.md) | CNA/CORE/DPU | the 3-bit precision values for all 6 dtypes (incl. the int4=6 RE and the bf16=3 / tf32=7 float rungs) |
| [encodings/tile-layouts.md](encodings/tile-layouts.md) | CNA / DPU-RDMA | feature cube C2, weight layouts, output cubes, per dtype |
| [encodings/cross-op-chaining.md](encodings/cross-op-chaining.md) | CNA / DPU (cube) | an fp16 matmul's narrowed output cube **is** the next op's input feature cube (both `feat_idx` C2=8) -> feed one op's output BO straight into the next, no host de-tile/re-tile (bit-exact [HW]); fp16-only (int/bf16 output cubes mismatch); multi-tile needs a KACC full-cube output + matched `Nt==Kt`; **pays on the transform-bound encoder (~80% transform/compute), ~2-3% on compute-bound LLM prefill** |
| [encodings/size-e-quirk.md](encodings/size-e-quirk.md) | DPU (write-out) | integer outputs stride as `size_e=7` regardless of byte width |
| [encodings/output-transpose-int16.md](encodings/output-transpose-int16.md) | DPU (write-out) | why int16 has no native matmul output; the byte-decomposition path |
| [encodings/k-accumulation.md](encodings/k-accumulation.md) | DPU-EW / DPU-RDMA | fp16 K-accum works; int8/int16/int32 EW K-accum is HW-dead |
| [encodings/sdp-stage-precision.md](encodings/sdp-stage-precision.md) | DPU SDP (BS/BN/EW) + CORE | the 3 SDP stages' precision: no stage adds a per-element *integer* tensor (BS/BN per-channel int32 broadcast, EW per-element but float-only ALU) -> on-device int32 K-accum impossible = the int8 ceiling; CACC no cross-op accumulate |
| [encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md) | DPU LUT (SDP) | on-NPU activation (NVDLA LE/LO hybrid): sigmoid/hardsigmoid/tanh/SiLU + **GELU (accurate 2-pass `x·Φ(x)`; the single-pass spikes in the flat tail)** + conv->act fusion + **LeakyReLU** + **sqrt/rsqrt/reciprocal/EXP/LOG** (shifted single-table, <1% over ~128x; EXP works standalone; **LOG = the first signed-output positive-domain kind, negative `out_lo` via OUT_CVT offset, absolute-error metric**) + fully-on-NPU EW **mul/add/sub/div**; the x~0 mux glitch = the LE/LO mux selects on `sign(x)`; **QUIRK 1: a flat/saturated in-table run mis-toggles the mux (~128 spike) ⇒ single-pass LUT fusion is curved-region-only, use 2-pass `x·gate(x)`**; **QUIRK 3: riding the exact max width (cols 8191) corrupts ~54 cube positions**; **QUIRK 4: a q=0 LUT table entry mis-decodes to a garbage ~4.0 ⇒ floor every table entry to q>=1** |
| [encodings/whisper-encoder.md](encodings/whisper-encoder.md) | composition | the Whisper/transformer encoder block fully on the NPU (cos=1.000000): EXP LUT, row-wise softmax (host row-max, no on-NPU max-reduce datapath) + **LogSoftmax** (`x−logsumexp`, host `log(s)` like softmax's `1/s`, per-row `ew_sub`) + stable **cross-entropy** (`logsumexp − logits[target]`, the on-NPU logsumexp + a **host gather**, since no hardware gather exists; fp32-grade since the loss skips fp16 output storage), LayerNorm (both reductions in one stacked-row feature-reduce), conv1d (lower with time on the height axis, since IH=1 overflows the feature banks), multi-head self-attention (pad the key count to %32 + mask the pad score columns; the matmul rejects unaligned N/K), 2-pass GELU, the full pre-norm block |
| [encodings/siglip-encoder.md](encodings/siglip-encoder.md) | composition | the **SigLIP-B/16 vision encoder** (SmolVLM-256M front-end) end-to-end on the NPU = patch-embed (im2col->matmul, stride==kernel patchify) + pos + 12x`rocket_encoder_block_fp16` `(L=1024,d=768,12h,d_ff=3072)` + post-LN; **fidelity 0.999998 cosine vs the fp32 HF oracle (SHARD 0.95)**; latency ~2.71 s warm (resident: prepacked GEMMs, multicore head-fanned attention, host softmax + bit-exact GELU LUT; ~2.30 s with all levers, 1.78x), not iso-hardware vs SHARD's 2.24 s; **NPU fact: full-attention softmax is data-movement bound on-NPU (~6.5 s, batching heads doesn't help) -> host threaded softmax ~10x cheaper once scores are de-tiled**; remaining floor = matmul de-tile + the host-softmax score round-trip |
| [encodings/feature-reduce.md](encodings/feature-reduce.md) | CNA/CORE/DPU (matmul) | reduce over the hidden/feature axis (`sum_h x[m,h]`) = a **ones-vector matmul**; the PPU **cannot** reduce the channel axis (it pools spatial `[H,W]` within a channel only); fp32-accumulate, the transformer-norm / softmax contraction. **Cumsum / prefix sum** = the same matmul with the ones-column widened to a **triangular ones matrix** (`out=in·Lᵀ`; incl/excl×fwd/rev; HW bit-exact) ⇒ the reduce-as-matmul family = full + weighted reduce + prefix scan |
| [encodings/rmsnorm-onnpu.md](encodings/rmsnorm-onnpu.md) | composition | RMSNorm = square->feature-reduce->(host rsqrt)->scale; the rsqrt stays on the host (M per-row scalars; LUT-domain otherwise); fp16-square overflow needs a power-of-2 prescale; the per-row broadcast scale primitive |
| [encodings/ffn-block.md](encodings/ffn-block.md) | composition | the gated-MLP FFN (GeGLU/SwiGLU): the only new op vs matmul is `act(gate)⊙up`; cosine-validated; the resident-cube fusion plan (host handoff today) |
| [encodings/norm-vision.md](encodings/norm-vision.md) | composition | vision norms (BatchNorm/GroupNorm/InstanceNorm/L2-Normalize) = the LayerNorm machinery with a different reduce-axis grouping of `[N,C,P]`; a (batch,group) block is contiguous ⇒ the feature-reduce reshape is a pure view; `G=C`->InstanceNorm, `G=1`->LayerNorm-over-CHW, BatchNorm = no-reduce per-channel affine; no new regcmd |
| [encodings/ppu-pooling.md](encodings/ppu-pooling.md) | PPU (PDP) | on-NPU MaxPool / AveragePool: the PPU+PPU_RDMA program, avg `RECIP=fp16(65536/k)`, enable mask 0x60 |
| [encodings/ppu-reduce-mean.md](encodings/ppu-reduce-mean.md) | PPU (PDP) | GlobalAvgPool / Mean over the spatial [H,W] axes via telescoping multi-pass (kernel cap 16); a PPU-written sub-4 intermediate is mis-read by the next chained pass; **GlobalMax/MinPool (ReduceMax/Min) reuse the same engine, idempotent ⇒ no reciprocal, bit-exact through the chain** (cf. feature-reduce.md for the orthogonal channel-axis reduce) |
| [encodings/out-cvt-converter.md](encodings/out-cvt-converter.md) | DPU (write-out) | the output converter `(acc×SCALE)>>SHIFT`, an integer scale and shift; fp32-cast + integer-scale fold, fractional dequant can't; **the tie rounds to even** (banker's, as QNNPACK's precise requantization, not the round-half-away-from-zero NVDLA specifies, nor the round-half-up the CPU models used to spell), and reaching a tie at all needs a scale chosen to make `MUL` a power of two, which is why no gate had ever exercised it |
| [encodings/regcmd-task-model.md](encodings/regcmd-task-model.md) | PC / tasks | task = regcmd + enable; a delta task computes in the register group its own job wrote, two tasks back as shipped; the register file persists across jobs; the CNA/CORE pointers are the kernel's |
| [encodings/cbuf-reuse.md](encodings/cbuf-reuse.md) | CNA (CBUF) | the WEIGHT_REUSE / DATA_REUSE operand-reuse bits |
| [encodings/cbuf-bank-slack.md](encodings/cbuf-bank-slack.md) | CNA (CBUF) | the int8 feature DMA over-reads by one bank; reserve `data_bank = fd_banks+1` |
| [encodings/mrdma-trap.md](encodings/mrdma-trap.md) | DPU-RDMA | the regcmd block you must emit or the job times out |
| [perf/device-vs-host-split.md](perf/device-vs-host-split.md) | whole NPU | how much of a matmul's wall the device actually holds, from the vendor driver's per-submit elapsed time: **67-82% once the weight is resident, 0.9-34% when it is re-packed per call**, and the envelope that decides how the number may be read |
| [perf/not-mac-bound.md](perf/not-mac-bound.md) | whole NPU | the ~460 GOP/s dtype-independent floor at this operating point; quant does not speed up matmul here (bottleneck-conditional, not a silicon law) |
| [perf/quant-prefill-microbatch.md](perf/quant-prefill-microbatch.md) | LLM prefill | quantized-GGUF prefill is **per-micro-batch dequant-bound**: `-ub 2048` ~2x's it, quant *type* is irrelevant to throughput, quant ~0.64x F16; short quant prefills route to CPU (`ROCKET_MIN_M_QUANT`); the F16 NPU prefill win **scales with model size** (0.8B 1.44x -> 9B 3.65x CPU); Qwen3.5/3.6 incl. hybrid-DeltaNet validated |
| [perf/weight-residency-fusion.md](perf/weight-residency-fusion.md) | LLM prefill | `ROCKET_F16_RESIDENT` packs F16 weights **once** (removes per-µbatch packB); projection fusion (Q\|K\|V, gate\|up) removes the redundant packA + collapses submits. Orthogonal, and they **stack**: a fusable group goes resident as one combined-N weight (bit-exact, no driver change); 3B-F16 fusion +5.7% on top of residency, +18.5%/+13.6% pp512/pp2048 stacked over streaming [HW sweep]; separate-projection archs only, F16 only, opt-in |
| [perf/iova-and-multicore.md](perf/iova-and-multicore.md) | kernel/DMA | per-fd 4 GB IOVA window (and how the vendor driver's one shared domain differs); N fds for N cores |
| [perf/clock.md](perf/clock.md) | clock/PM | 200 -> 600 MHz, the cold power-domain gotcha, the 900 MHz hard-lock |
| [perf/per-process-readout.md](perf/per-process-readout.md) | benchmark method | the instrument behind each timed arm's `<!--RO-->` line: which cluster retired the work (both cluster PMUs, plus a free per-CPU jiffy delta) and where its pages landed (cache color, physical contiguity, from one bounded `pagemap` sample). Carries its positive controls, because on a freshly reset board the placement columns are near-constant by construction and a flat column cannot otherwise be told from a broken probe; and the two silent failures (`pfn_zero_frac`, `pmu_enabled`) to check before quoting any of it |
| [perf/cpu-governor-and-offload.md](perf/cpu-governor-and-offload.md) | clock/PM | an offloading process blocks, so a load-sampling CPU governor parks the big cores and the host half of the work runs at the floor: **1.27x on a board whose A76 floor is 1.2 GHz, 3.2x where it is 408 MHz**, while the CPU-only arm is flat; pin the governor and read `scaling_min_freq` before quoting an NPU-versus-CPU number |
| [perf/bo-sync-cost.md](perf/bo-sync-cost.md) | kernel/DMA | `PREP_BO`/`FINI_BO` cache-sync is ∝ BO size; right-size the repeatedly-synced KACC output BO (~+11% resident fp16) |
| [perf/pool-completion.md](perf/pool-completion.md) | PPU / kernel | a pool raises no DPU interrupt and mainline `rocket` waits for one: **~507 ms and a core reset per pool submit** on the RK3588 (answers stay correct); the `PPU_DONE` fix is in the RK3576 patch series and not in `patches/rocket/` |
| [perf/ppu-pooling-not-detile.md](perf/ppu-pooling-not-detile.md) | PPU | the PPU is a pooling engine (PDP), not RUBIK, and cannot de-tile; no on-chip layout conv in either direction |
| [perf/sram-nbuf.md](perf/sram-nbuf.md) | system SRAM / IOMMU | the NPU reaches SRAM via an IOVA like DDR (no NPU↔SRAM bus / no "NBUF" engine); mainline rocket has no SRAM support; syssram owned by the codec; weak lever (gather-bound readback) |
| [perf/attention-offload-crossover.md](perf/attention-offload-crossover.md) | prefill attention | offloading `FLASH_ATTN_EXT` **wins from ~2K** with per-worker QK/AV submit chaining (parity <=1K, 1.45x @8K; ~6K crossover without chaining): CPU attention is super-linear, NPU attention flat once multicored + chained; gate the offload on `n_kv` (~ the sliding-window length), not `n_tokens` |
| [perf/asr-cpu-relief.md](perf/asr-cpu-relief.md) | speech-to-text | ASR measured in **CPU core-seconds**, not throughput: whisper frees 16-50% of the process CPU (rises with model size, largest on short clips because every window pads to 30 s) and the wall ratio understates it; `ROCKET_MIN_M_QUANT`=512 is an LLM-prefill floor that offloads nothing from a Q8_0 CTC encoder under ~60 s, and 128 turns 0.4% into 42%; the fp16 encoder is bit-identical on clean audio and diverges word-level on degraded audio; the whisper weight-repack lever caps at 0.6% of process CPU, and on a Q8_0 host the same lever caps at **12.2% at 10 s falling to 0.8% at 120 s** because the term is fixed per forward pass, reachable only across utterances |
| [perf/asr-streaming-service.md](perf/asr-streaming-service.md) | speech-to-text | `whisper-server` fed fixed-length chunks the way OpenWebRX+ feeds it, billed in CPU core-seconds per second of audio: the shipped NPU arm is 0.58x the CPU; `-nt -ac 1200 -sns` with the temperature ladder off reads **0.70x at better WER** on 20 s chunks and 30 s chunks with `-nt` 0.65x; plain 30 s chunks cost **+13%** (whisper re-windows the mid-word cut at the window's edge); an `-ac` equal to the chunk length loops the decoder at the audio's end and 4 s of headroom fixes it; the server's `-nf` is a no-op; a quantized whisper costs +10%; the unmasked-attention offload is CPU parity at +36% wall |

## The one-paragraph summary

The RK3588 NPU is an NVDLA-derived 3-core accelerator. The `rocket` kernel driver is a generic
register-command submitter (`CREATE_BO` / `SUBMIT` / `PREP_BO`). You can therefore drive matmul
yourself by emitting the same CNA→CORE→DPU regcmd Mesa uses for convolution, because a matmul is
a 1×1 convolution.

**Datatypes.** It natively supports int4, int8, int16, fp16, bf16 and tf32, with int32 and fp32
outputs. We have a working matmul for every one. int16 is the lone exception: it has no native
matmul *output*, so it is done by int8 byte-decomposition.

**Layout.** Weights and activations must be pre-scattered into native tiled layouts on the host,
because the NPU has no on-chip row-major-to-tiled conversion. The integer-output write stride has
a quirk, `size_e=7`.

**Accumulation.** You can accumulate fp16 K-partials on-chip via the DPU eltwise unit, but not
integer ones, because the DPU eltwise ALU is float-only.

**The int8 feature cube has a CBUF gotcha.** Its DMA over-reads by one bank, so give it
`data_bank = fd_banks+1` of slack. fp16 is immune.

**Alignment.** The matmul rows are the conv's spatial height, and a height below 4 mis-computes on
the hardware at every dtype. That is the `M==1` single-vector or GEMV case. So `M%4==0` is the
real constraint, and software pads `M==1` to 4.

**Cores.** You reach the 3 cores by opening 3 or more file descriptors, one scheduling entity per
fd.

**Activations.** The DPU also has an NVDLA LUT unit that computes nonlinear activations on-chip:
sigmoid, tanh, SiLU, GELU, sqrt, rsqrt, reciprocal and exp. Composed with the matmul, that is
enough to run a full transformer or Whisper encoder block on the NPU.

Two LUT gotchas. A table entry of exactly `q=0` mis-decodes to a garbage ~4.0, so floor entries to
`q>=1`. And riding the exact 13-bit max cube width corrupts the tail, so tile under it.

**The most important performance fact.** At the current operating point the matmul is
DMA/dispatch-bound rather than MAC-bound. Quantization therefore buys RAM rather than prefill
speed. That is bottleneck-conditional rather than a permanent silicon law.

**Clock.** It boots at 200 MHz, and can only be raised from inside the driver after the power
domain is up.

## License

The documentation in this repository (the prose notes, tables, and encoding write-ups) is
licensed under [Creative Commons Attribution 4.0 International](LICENSE) (CC-BY-4.0): reuse it
freely with attribution.

The helper scripts under `ppu-rknn-capture/` and the harness under `perf/data/rknn-encoder/`
carry their own `SPDX-License-Identifier: GPL-3.0-or-later` headers. They are licensed
accordingly. The two C programs in `perf/data/rknn-encoder/` link Rockchip's proprietary
`librknnrt`. Each also grants the GPL version 3 section 7 permission to convey it combined with
that library. Third-party captures retain their upstream copyright and license terms, notably
`ppu-rknn-capture/registers.xml` (from the Mesa `rocket`/Teflon driver), credited in
[SOURCES.md](SOURCES.md).
