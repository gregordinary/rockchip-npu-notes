# Rockchip NPU — Reverse-engineering notes

## AI Disclosure

The documents in rockchip-npu-notes were authored by AI, primarily Claude Code (Opus 4.8). These documents were produced as part of a series of side projects and are being published here as they may be of use to efforts by others. Accuracy of information is not guaranteed.

## About rockchip-npu-notes

These are subsystem-organized, project-independent notes on the Rockchip RK3588
NPU as driven through the mainline `rocket` DRM-accel driver.

They were established by reverse-engineering the hardware on a real device (Turing
RK1, 32 GB, mainline kernel ~7.1) while building a FOSS inference stack on top of
`rocket`: a userspace matmul library, a ggml backend, an NPU-clock patch, and a
TFLite delegate. Most of what we learned, though, is not specific to any one of
those projects — it is facts about the silicon and how its register-command
interface behaves. That is what lives here.

If you are trying to run your own compute on the RK3588 NPU through `rocket` (or
any raw-regcmd path), this repository provides observations, insights, and details on precision encodings, native tile layouts, integer-output `size_e` quirk, what the DPU eltwise unit can and cannot accumulate, the CBUF operand-reuse bits, the MRDMA trap that hangs your first job, the per-fd IOVA window, and the clock that boots at 1/5 speed.

For a start-to-finish walkthrough that ties the driver library, the frontends, and the
kernel patches together, see the [guide](guide/).

## How to read this

Every claim is tagged with how it was established:

- **[HW sweep]** — reverse-engineered empirically by sweeping a value/geometry on the
  real NPU and observing bit-exact vs garbage vs nothing-written. The strongest
  evidence: it is what the hardware actually does.
- **[source-confirmed]** — corroborated by reading an authoritative FOSS source: the Mesa
  `rocket`/Teflon driver, the open NVDLA documentation, or the RK3588 register headers we
  build on (`npu_hw.h`, from Jasbir Matharu's `rk3588-npu`). See [SOURCES.md](SOURCES.md).

Where a fact is HW-confirmed *and* matches a source, both tags appear. Negative
results (things that do not work) are documented as carefully as the positive
ones — they were the most expensive to learn.

## Map

| Doc | Subsystem | The fact |
|---|---|---|
| [hardware-overview.md](hardware-overview.md) | whole NPU | NVDLA lineage, 3 cores, CBUF 12×32 KB, the precision menu, TOPS vs reality |
| [datatypes.md](datatypes.md) | whole NPU | the datatype capability matrix — precision field, output type, MAC rate, and use, per dtype |
| [matmul-as-conv.md](matmul-as-conv.md) | CNA/CORE/DPU | how a matmul is run as a 1×1 convolution; tiling; the data flow; the alignment rules + the feature-height-<4 (M==1 GEMV) break |
| [depthwise-conv.md](depthwise-conv.md) | CNA/CORE/DPU | how depthwise differs from a direct conv: `CONV_MODE=3`+`DW_EN`, `weights_kernels=1`, `size_e=3`, surfaces ×2 |
| [encodings/conv-transpose.md](encodings/conv-transpose.md) | CNA (lowering) | ConvTranspose2d/deconv has NO hardware mode — it lowers to dilate-input + `rot180(Wᵀ)` + a stride-1 forward conv; the 180°-flip derivation + `pad ≤ d·(K−1)` constraint |
| [encodings/resize-upsample.md](encodings/resize-upsample.md) | CNA (lowering) | nearest/bilinear resize = a depthwise transposed conv with a box/triangle kernel; the triangle's stride-subsample is a partition of unity ⇒ half-pixel 2-tap bilinear; `C%32` |
| [encodings/precision-field.md](encodings/precision-field.md) | CNA/CORE/DPU | the 3-bit precision values for all 6 dtypes (incl. the int4=6 RE and the bf16=3 / tf32=7 float rungs) |
| [encodings/tile-layouts.md](encodings/tile-layouts.md) | CNA / DPU-RDMA | feature cube C2, weight layouts, output cubes — per dtype |
| [encodings/cross-op-chaining.md](encodings/cross-op-chaining.md) | CNA / DPU (cube) | an fp16 matmul's narrowed output cube **is** the next op's input feature cube (both `feat_idx` C2=8) → feed one op's output BO straight into the next, no host de-tile/re-tile (bit-exact [HW]); fp16-only (int/bf16 output cubes mismatch); multi-tile needs a KACC full-cube output + matched `Nt==Kt`; **pays on the transform-bound encoder (~80% transform/compute), ~2–3% on compute-bound LLM prefill** |
| [encodings/size-e-quirk.md](encodings/size-e-quirk.md) | DPU (write-out) | integer outputs stride as `size_e=7` regardless of byte width |
| [encodings/output-transpose-int16.md](encodings/output-transpose-int16.md) | DPU (write-out) | why int16 has no native matmul output; the byte-decomposition path |
| [encodings/k-accumulation.md](encodings/k-accumulation.md) | DPU-EW / DPU-RDMA | fp16 K-accum works; int8/int16/int32 EW K-accum is HW-dead |
| [encodings/sdp-stage-precision.md](encodings/sdp-stage-precision.md) | DPU SDP (BS/BN/EW) + CORE | the 3 SDP stages' precision: no stage adds a per-element *integer* tensor (BS/BN per-channel int32 broadcast, EW per-element but float-only ALU) → on-device int32 K-accum impossible = the int8 ceiling; CACC no cross-op accumulate |
| [encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md) | DPU LUT (SDP) | on-NPU activation (NVDLA LE/LO hybrid): sigmoid/hardsigmoid/tanh/SiLU + **GELU (accurate 2-pass `x·Φ(x)`; the single-pass spikes in the flat tail)** + conv→act fusion + **LeakyReLU** + **sqrt/rsqrt/reciprocal/EXP/LOG** (shifted single-table, <1% over ~128×; EXP works standalone; **LOG = the first SIGNED-output positive-domain kind, negative `out_lo` via OUT_CVT offset, absolute-error metric**) + fully-on-NPU EW **mul/add/sub/div**; the x≈0 mux glitch = the LE/LO mux selects on `sign(x)`; **QUIRK 1: a flat/saturated in-table run mis-toggles the mux (~128 spike) ⇒ single-pass LUT fusion is curved-region-only, use 2-pass `x·gate(x)`**; **QUIRK 3: riding the EXACT max width (cols 8191) corrupts ~54 cube positions**; **QUIRK 4: a q=0 LUT table entry mis-decodes to a garbage ~4.0 ⇒ floor every table entry to q≥1** |
| [encodings/whisper-encoder.md](encodings/whisper-encoder.md) | composition | the Whisper/transformer encoder block FULLY on the NPU (cos=1.000000): EXP LUT, row-wise softmax (host row-max, no on-NPU max-reduce datapath) + **LogSoftmax** (`x−logsumexp`, host `log(s)` like softmax's `1/s`, per-row `ew_sub`) + stable **cross-entropy** (`logsumexp − logits[target]`, the on-NPU logsumexp + a **host GATHER** — NO HW gather exists; fp32-grade since the loss skips fp16 output storage), LayerNorm (BOTH reductions in ONE stacked-row feature-reduce), conv1d (lower with TIME on the HEIGHT axis — IH=1 overflows the feature banks), multi-head self-attention (pad the key count to %32 + mask the pad score columns; the matmul rejects unaligned N/K), 2-pass GELU, the full pre-norm block |
| [encodings/siglip-encoder.md](encodings/siglip-encoder.md) | composition | the **SigLIP-B/16 vision encoder** (SmolVLM-256M front-end) end-to-end on the NPU = patch-embed (im2col→matmul, stride==kernel patchify) + pos + 12×`rocket_encoder_block_fp16` `(L=1024,d=768,12h,d_ff=3072)` + post-LN; **fidelity 0.999998 cosine vs the fp32 HF oracle (SHARD 0.95)**; latency ~6 s warm (resident: prepacked GEMMs + threaded host softmax/GELU), NOT iso-hardware vs SHARD's 2.24 s; **NPU FACT: full-attention softmax is data-movement bound on-NPU (~6.5 s, batching heads doesn't help) → host threaded softmax ~10× cheaper once scores are de-tiled**; remaining floor = matmul de-tile + non-fused attention |
| [encodings/feature-reduce.md](encodings/feature-reduce.md) | CNA/CORE/DPU (matmul) | reduce over the hidden/feature axis (`sum_h x[m,h]`) = a **ones-vector matmul** — the PPU **cannot** reduce the channel axis (it pools spatial `[H,W]` within a channel only); fp32-accumulate, the transformer-norm / softmax contraction. **Cumsum / prefix sum** = the same matmul with the ones-COLUMN widened to a **triangular ones MATRIX** (`out=in·Lᵀ`; incl/excl×fwd/rev; HW bit-exact) ⇒ the reduce-as-matmul family = full + weighted reduce + prefix scan |
| [encodings/rmsnorm-onnpu.md](encodings/rmsnorm-onnpu.md) | composition | RMSNorm = square→feature-reduce→(host rsqrt)→scale; the rsqrt stays on the HOST (M per-row scalars; LUT-domain otherwise); fp16-square overflow needs a power-of-2 prescale; the per-row broadcast scale primitive |
| [encodings/ffn-block.md](encodings/ffn-block.md) | composition | the gated-MLP FFN (GeGLU/SwiGLU): the only new op vs matmul is `act(gate)⊙up`; cosine-validated; the resident-cube fusion plan (host handoff today) |
| [encodings/norm-vision.md](encodings/norm-vision.md) | composition | vision norms (BatchNorm/GroupNorm/InstanceNorm/L2-Normalize) = the LayerNorm machinery with a different reduce-axis grouping of `[N,C,P]`; a (batch,group) block is contiguous ⇒ the feature-reduce reshape is a pure view; `G=C`→InstanceNorm, `G=1`→LayerNorm-over-CHW, BatchNorm = no-reduce per-channel affine; no new regcmd |
| [encodings/ppu-pooling.md](encodings/ppu-pooling.md) | PPU (PDP) | on-NPU MaxPool / AveragePool: the PPU+PPU_RDMA program, avg `RECIP=fp16(65536/k)`, enable mask 0x60 |
| [encodings/ppu-reduce-mean.md](encodings/ppu-reduce-mean.md) | PPU (PDP) | GlobalAvgPool / Mean over the SPATIAL [H,W] axes via telescoping multi-pass (kernel cap 16); a PPU-written sub-4 intermediate is mis-read by the next chained pass; **GlobalMax/MinPool (ReduceMax/Min) reuse the same engine — idempotent ⇒ no reciprocal, BIT-EXACT through the chain** (cf. feature-reduce.md for the orthogonal channel-axis reduce) |
| [encodings/out-cvt-converter.md](encodings/out-cvt-converter.md) | DPU (write-out) | the output converter `(acc×SCALE)>>SHIFT` — INTEGER scale/shift; fp32-cast + integer-scale fold, fractional dequant can't |
| [encodings/regcmd-task-model.md](encodings/regcmd-task-model.md) | PC / tasks | task = full regcmd + enable; delta regcmd doesn't fire; register file persists globally across jobs |
| [encodings/cbuf-reuse.md](encodings/cbuf-reuse.md) | CNA (CBUF) | the WEIGHT_REUSE / DATA_REUSE operand-reuse bits |
| [encodings/cbuf-bank-slack.md](encodings/cbuf-bank-slack.md) | CNA (CBUF) | the int8 feature DMA over-reads by one bank — reserve `data_bank = fd_banks+1` |
| [encodings/mrdma-trap.md](encodings/mrdma-trap.md) | DPU-RDMA | the regcmd block you must emit or the job times out |
| [perf/not-mac-bound.md](perf/not-mac-bound.md) | whole NPU | the ~460 GOP/s dtype-independent ceiling — quant doesn't speed up matmul |
| [perf/quant-prefill-microbatch.md](perf/quant-prefill-microbatch.md) | LLM prefill | quantized-GGUF prefill is **per-micro-batch dequant-bound** — `-ub 2048` ~2×'s it, quant *type* is irrelevant to throughput, quant ≈ 0.64× F16; short quant prefills route to CPU (`ROCKET_MIN_M_QUANT`); the F16 NPU prefill win **scales with model size** (0.8B 1.44× → 9B 3.65× CPU); Qwen3.5/3.6 incl. hybrid-DeltaNet validated |
| [perf/iova-and-multicore.md](perf/iova-and-multicore.md) | kernel/DMA | per-fd 4 GB IOVA window; N fds for N cores |
| [perf/clock.md](perf/clock.md) | clock/PM | 200 → 600 MHz, the cold power-domain gotcha, the 900 MHz hard-lock |
| [perf/bo-sync-cost.md](perf/bo-sync-cost.md) | kernel/DMA | `PREP_BO`/`FINI_BO` cache-sync is ∝ BO size — right-size the repeatedly-synced KACC output BO (~+11% resident fp16) |
| [perf/ppu-pooling-not-detile.md](perf/ppu-pooling-not-detile.md) | PPU | the PPU is a POOLING engine (PDP), not RUBIK — can't de-tile; no on-chip layout conv in either direction |
| [perf/sram-nbuf.md](perf/sram-nbuf.md) | system SRAM / IOMMU | the NPU reaches SRAM via an IOVA like DDR (no NPU↔SRAM bus / no "NBUF" engine); mainline rocket has no SRAM support; syssram owned by the codec; weak lever (gather-bound readback) |
| [perf/attention-offload-crossover.md](perf/attention-offload-crossover.md) | prefill attention | offloading `FLASH_ATTN_EXT` **wins from ~2K** with per-worker QK/AV submit chaining (parity ≤1K, 1.45× @8K; ~6K crossover without chaining) — CPU attention is super-linear, NPU attention flat once multicored + chained; gate the offload on `n_kv` (≈ the sliding-window length), not `n_tokens` |

## The one-paragraph summary

The RK3588 NPU is an NVDLA-derived 3-core accelerator. The `rocket` kernel driver
is a generic register-command submitter (`CREATE_BO` / `SUBMIT` / `PREP_BO`), so you
can drive matmul yourself by emitting the same CNA→CORE→DPU regcmd Mesa uses for
convolution — a matmul is just a 1×1 convolution. It natively supports int4 / int8 /
int16 / fp16 / bf16 / tf32 (+ int32 / fp32 outputs); we have a working matmul for
every one (int16 is the lone exception — it has no native matmul *output*, so it's
done by int8 byte-decomposition). Weights and activations must be pre-scattered into
native tiled layouts on the host (the NPU has no on-chip row-major→tiled
conversion). The integer-output write stride has a quirk (`size_e=7`). You can
accumulate fp16 K-partials on-chip via the DPU eltwise unit, but not integer
ones — the EW operand DMA is ≤16-bit. The int8 feature cube has a CBUF gotcha — its
DMA over-reads by one bank, so you must give it `data_bank = fd_banks+1` of slack
(fp16 is immune). The matmul rows are the conv's spatial height, and a height below
4 (the `M==1` single-vector / GEMV case) mis-computes on the hardware at every
dtype — so `M%4==0` is the real constraint and software pads `M==1` to 4. You reach
the 3 cores by opening 3+ file descriptors (one scheduling entity per fd). The DPU also has an
NVDLA LUT unit that computes nonlinear activations on-chip (sigmoid/tanh/SiLU/GELU/sqrt/rsqrt/
reciprocal/exp) — enough, composed with the matmul, to run a full transformer/Whisper encoder
block on the NPU; two LUT gotchas: a table entry of exactly `q=0` mis-decodes to a garbage ~4.0
(floor entries to `q≥1`), and riding the exact 13-bit max cube width corrupts the tail (tile under
it). And the most important performance fact: on this hardware the matmul is DMA/dispatch-bound,
not MAC-bound — so quantization buys you RAM, not prefill speed. The clock boots at 200 MHz and
can only be raised
from inside the driver after the power domain is up.

## License

The documentation in this repository — the prose notes, tables, and encoding write-ups — is
licensed under [Creative Commons Attribution 4.0 International](LICENSE) (CC-BY-4.0): reuse it
freely with attribution.

The helper scripts under `ppu-rknn-capture/` carry their own `SPDX-License-Identifier:
GPL-3.0-or-later` headers and are licensed accordingly. Third-party captures retain their upstream
copyright and license terms — notably `ppu-rknn-capture/registers.xml` (from the Mesa
`rocket`/Teflon driver), credited in [SOURCES.md](SOURCES.md).
