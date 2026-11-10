# Rockchip NPU programming notes

These notes describe the hardware blocks and the register-command interface of the Rockchip
NPU, the [RKNPU](GLOSSARY.md). They describe it as programmed through the mainline Linux
`rocket` DRM accel driver. They are for developers who program the NPU directly, through `rocket` or another raw
register-command path. Unless stated otherwise, the notes describe the RK3588.

## AI disclosure

The documents in rockchip-npu-notes were authored by AI, primarily Claude. They were produced as
part of a series of side projects, and are published here in case they are of use to others.
Accuracy of information is not guaranteed.

## Supported SoCs

Most of this content applies to every Rockchip SoC that integrates the RKNPU, an NVDLA-derived
design. Per-SoC differences ([machine parameters](GLOSSARY.md), the register offset map, and SoC
integration) are documented in [chips/](chips/).

| SoC | NPU cores | Status | Sheet |
|---|---:|---|---|
| RK3588 | 3 | Supported. The reference target | [chips/rk3588.md](chips/rk3588.md) |
| RK3576 | 2 | Supported for the operations [chips/README.md](chips/README.md) lists, on `rocket` 1.6.0 | [chips/rk3576.md](chips/rk3576.md), [chips/rk3576-regcmd.md](chips/rk3576-regcmd.md) |
| RK3566 | 1 | Not characterized | [chips/rk3566.md](chips/rk3566.md) |

[chips/porting-patterns.md](chips/porting-patterns.md) lists which RK3588 facts transfer to
another SoC, and [chips/rk3576.md](chips/rk3576.md) has the RK3576's own.

## Architecture overview

The RK3588 NPU is a three-core accelerator derived from NVIDIA's NVDLA. Each core runs a fixed
pipeline of blocks, [CNA](GLOSSARY.md)→[CORE](GLOSSARY.md)→[DPU](GLOSSARY.md), and has its own
384 KB convolution buffer, the [CBUF](GLOSSARY.md) [source-confirmed]. Each block has its own register range [source-confirmed: Mesa
`registers.xml`]:

| Block | Registers | Function | NVDLA equivalent |
|---|---|---|---|
| [PC](GLOSSARY.md) | `0x0xxx` | Fetches a [task](GLOSSARY.md)'s [register program](GLOSSARY.md) from memory and starts the blocks it names | None |
| CNA | `0x1xxx` | Loads the input feature and the weights into the CBUF, and feeds the MAC array | CDMA, CSC |
| CORE | `0x3xxx` | The MAC array and its accumulators | CMAC, CACC |
| DPU | `0x4xxx` | Post-processing: the BS, BN and [EW stages](GLOSSARY.md), the [LUT](GLOSSARY.md), the [output converter](GLOSSARY.md), and the write DMA | SDP |
| [DPU_RDMA](GLOSSARY.md) | `0x5xxx` | The DPU's read DMA. Its channels read the BS, BN and EW stages' operands, and [MRDMA](GLOSSARY.md) can read the DPU's main input from memory in place of the CORE's output | SDP_RDMA |
| [PPU](GLOSSARY.md) | `0x6xxx` | Pooling, with its read DMA at `0x7xxx` | PDP |

The NPU has no matmul primitive. A matmul runs as a 1×1 convolution, on the same program Mesa
emits for a convolution. [hardware-overview.md](hardware-overview.md) describes the blocks.
[nvdla-lineage.md](nvdla-lineage.md) maps them to the NVDLA documentation.

## Programming model

`rocket` is a generic register-command submitter. Userspace builds each operation as a register
program, and the kernel runs whatever CNA→CORE→DPU program it is handed. The uAPI has four
ioctls:

| Ioctl | Function |
|---|---|
| `CREATE_BO` | Allocates a buffer object ([BO](GLOSSARY.md)) and maps it into the file descriptor's [IOVA window](GLOSSARY.md) |
| `SUBMIT` | Queues one or more jobs, each of one or more tasks, and returns without waiting for them |
| `PREP_BO` | Begins CPU access to a BO. On an output BO, it waits for the job that writes it |
| `FINI_BO` | Ends CPU access to a BO |

A task is one register program: a sequence of [regcmds](GLOSSARY.md), each writing one value
to one register. The PC block fetches and applies them, then starts the blocks. A
[submit](GLOSSARY.md), a task and a [kick](GLOSSARY.md) (one hardware start of the PC) are three separate units, and
[encodings/regcmd-task-model.md](encodings/regcmd-task-model.md) describes how they relate.

A matmul computes `C[M×N] = A[M×K] × B[K×N]`, with A the activation, B the weight and C the
output. Run as a 1×1 convolution, M is the spatial height, K the input channels and N the
output channels ([matmul-as-conv.md](matmul-as-conv.md)). The host scatters B into the
[cube layout](GLOSSARY.md) the CNA reads ([encodings/tile-layouts.md](encodings/tile-layouts.md)).

## Capabilities

The facts below apply to the RK3588. Each label links to the note with the detail and the
evidence:

- **[Datatypes](datatypes.md).** The integer types are int4, int8 and int16, and the float types
  are fp16, bf16 and tf32. Each has a native matmul [HW sweep]. int16's int32 output saturates,
  so a matmul that needs int64-exact results over full-range int16 operands is computed as four
  int8 matmuls.
- **[Layouts](matmul-as-conv.md).** No register converts a layout, so software must scatter the
  weights into the cube layout on the host [source-confirmed]. A matmul's activation and output
  need no packing. With the strides programmed for it, the CNA reads A as plain rows, and the
  DPU writes C as plain rows [HW sweep]. The device time of that form grows with the output's row pitch once a
  row passes 4 KiB.
- **[K-accumulation](encodings/k-accumulation.md).** The DPU's EW stage performs fp16 [K-accumulation](GLOSSARY.md)
  on chip, so the host reads each output [tile](GLOSSARY.md) once [HW sweep, source-confirmed].
  The EW ALU also adds
  int8, int16 and int32 exactly when every precision field is set to one integer code
  [HW sweep]. With a convolution as its main feed, it reads its operand as 16-bit elements. No
  tested configuration accumulates an int32 partial on chip [HW sweep].
- **[On-chip requant](encodings/out-cvt-converter.md).** The DPU's output converter performs the [requant](GLOSSARY.md)
  of an int32 accumulator, with an integer multiplier and a shift. Its [BS stage](GLOSSARY.md) multiplies
  per output channel, which applies TFLite's [per-axis](GLOSSARY.md) scales on chip [HW sweep]. An int8
  convolution therefore writes int8 with no host requant.
- **[Cores and address space](perf/iova-and-multicore.md).** Under `rocket`, each open file
  descriptor is one scheduling entity with its own 4 GB IOVA window [HW sweep]. The jobs of one
  file descriptor run one at a time, in order. The scheduler moves a file descriptor to another
  core only when none of its jobs is queued or running. A lone file descriptor that waits on
  each job therefore runs only on cores 0 and 1, alternately [HW sweep, source-confirmed: the
  DRM scheduler]. To use all three cores, software must open three or more file descriptors.
- **[Operations beyond matmul](encodings/dpu-lut-activation.md).** The DPU's LUT computes
  activations such as SiLU and GELU, and functions such as exp and rsqrt [HW sweep]. The PPU runs
  max and average pooling. Composed with the matmul, they run a full Whisper [encoder](GLOSSARY.md)
  block on the NPU ([encodings/whisper-encoder.md](encodings/whisper-encoder.md)). The CNA also has a
  [hardware deconvolution mode](encodings/conv-transpose.md) that computes a ConvTranspose with
  a power-of-two stride in one task.

## Known issues

Each of these produces a wrong result or a stalled job, and no error names the cause. The issue
column links to the note with the mechanism and the evidence.

| ID | Issue | Symptom | Workaround |
|---|---|---|---|
| E1 | [The MRDMA block](encodings/mrdma-trap.md): a plain convolution or matmul with `mrdma_disable` clear in `DPU_RDMA_FEATURE_MODE_CFG` (`0x5044`) | The DPU read DMA waits for data that never arrives. The job times out with its output untouched [HW sweep, source-confirmed] | Software must set `mrdma_disable`, and include the DPU_RDMA bit in `PC_OPERATION_ENABLE` |
| E2 | [Output stride](encodings/size-e-quirk.md): the output of an int8 or int4 matmul (int32 and int16) programmed with its natural `size_e`, with N spanning more than one [surface](GLOSSARY.md) | Most of the output is not written [HW sweep] | Software must program `size_e=7` for the int32 output of an int8 matmul and the int16 output of an int4 matmul. An int16 matmul's int32 output takes the natural `size_e=3`, and hangs at 7 |
| E3 | [Feature surface stride](matmul-as-conv.md): a [feature cube](GLOSSARY.md) whose height is not a multiple of 4, with the surface stride in `CNA_DMA_CON2` computed as `line_stride × (H/4 − 1)` in integer arithmetic | Incorrect output. A matmul programmed this way is incorrect at every M that is not a multiple of 4 [HW sweep] | Software must program the surface stride as `width × (H − 4)`, a 28-bit two's complement below four rows |
| E4 | [CBUF bank slack](encodings/cbuf-bank-slack.md): the int8 feature DMA at some tile geometries | The DMA over-reads by one CBUF bank and returns garbage in the tile's last rows [HW sweep]. The fp16 cube is not affected | Software must reserve `data_bank = fd_banks+1` |
| E5 | [LUT table entries](encodings/dpu-lut-activation.md): a shifted LUT table entry of exactly `q=0` | The entry decodes to ~4.0 rather than 0 [HW sweep] | Software must clamp every shifted-table entry to `q>=1` |
| E6 | [Requant scale and shift sign](encodings/out-cvt-converter.md): the output converter's scale and the BS stage's shift | Bit 15 of the scale is a sign, and Mesa's derivation of the scale drops a carry, which halves or negates one scale in 16384 [HW sweep]. The BS stage's shift is split by sign across two fields, and one field alone scales negative products wrongly | Software must keep the carry when it derives the scale, and must program both shift fields |
| E7 | [Per-core float difference](chips/rk3588.md#erratum-e7-fp16-matmul-output-differs-between-cores-on-output-lane-0): an fp16 matmul with fp32 output, compared across cores | The three cores compute it identically except on output lane 0 of each 16-channel group, where each core rarely writes a result slightly low, on its own set of inputs: 2^-16 low for results from about 2^-4 to 2 on the data characterized [HW sweep]. `rocket` places a job on a core by load, so two runs of one fp16 job can differ on lane 0. The integer paths are core-invariant | None |
| E8 | [Convolution tile limits](matmul-as-conv.md): a convolution task that reads more than 1022 input rows or 2047 input columns, or that programs a pad past 15 | The extent or the pad wraps its field. The task writes a complete output of the wrapped extent, or at the wrong offset, with incorrect values [HW sweep] | Software must keep each task within 1022 input rows by 2047 columns and a pad of 15, and must write a larger pad into the task's input |
| E9 | [Cube-dimension limit](encodings/dpu-lut-activation.md): a LUT chunk wider than 8176 positions, and any other DPU cube dimension of 8191 | The LUT chunk computes every position from 8176 on incorrectly and raises no completion [HW sweep 2026-11-02]. SHARD reports a `REGTASK Overflow` for any operand index past 8191 [source-confirmed] | Software must keep a LUT chunk at or below 8176 positions and every other DPU cube dimension below 8191. The PPU's height and channel axes are exact at 8192 [HW sweep 2026-11-02]. The CNA's input width and height are 11 bits, narrower still |
| E10 | [NPU clock](perf/clock.md): setting the NPU clock rate while the power domain is off | The SCMI firmware hangs [HW sweep] | Software must raise the rate from inside the driver, after runtime PM powers the domain |
| E11 | [Streamed weight banks](matmul-as-conv.md): a matmul task that streams its weights through fewer weight banks than one whole 32-kernel group needs, or through one bank. An int8 task also at exactly that minimum | The task never completes. The watchdog retires it, the fence signals as if it had completed, and the output is not written [HW sweep] | Software must program at least two weight banks, enough banks for one whole 32-kernel group, and for int8 one bank more |
| E12 | [K-accumulation surface stride](encodings/k-accumulation.md): the EW operand's `EW_SURF_STRIDE` and `SURF_NOTCH` floored at 12, `MAX(M, 12)`, with M below 12 and N above 16 | The sum reads the wrong partial surfaces from the third onward, and the output is complete and plausible [HW sweep] | Software must program both fields as the partial's own surface stride, M |
| E13 | [Job timeouts](chips/rk3588.md#erratum-e13-a-core-stops-completing-jobs-after-job-timeouts): a job that times out with its DMA still active, and repeated job timeouts on one core | The core's IOMMU cannot stall, and later jobs on the core intermittently never complete. Then an NPU power domain fails to idle, the next resume takes an external abort, and the core stays `resuming` until a power cycle [HW sweep] | Software must not submit a program that cannot raise its completion, and must stop submitting after an IOMMU stall timeout on an NPU core |

## Performance

**Conditions:** RK3588 on mainline `rocket`. Each item states its NPU clock, and the linked note
gives its full conditions.

- **[Matmul throughput](perf/not-mac-bound.md).** DMA and dispatch bound the matmul, rather than
  the MAC array. At `512×3840×4096`, with [resident weights](GLOSSARY.md) on three cores at
  928 MHz, fp16 measures 947-966 GOP/s. int8 with streamed weights and int4 run 1.14-1.20x that
  [HW sweep 2026-11-02]. Their MAC rates are 2x and 4x fp16's, so quantization reduces
  memory use and raises the matmul's throughput far less than its MAC rate. In a model, resident
  int8 [prefill](GLOSSARY.md) measured slower than fp16.
- **[Weight streaming](perf/not-mac-bound.md).** [CBUF weight streaming](GLOSSARY.md), the fp16
  default, removes most of the whole-tile program's tasks. At 928 MHz it gives 1.07-1.17x on
  prefill over the whole-tile program, on four [GGUFs](GLOSSARY.md) [HW sweep 2026-10-21].
- **[Decode](perf/decode-gemv.md).** Single-row [decode](GLOSSARY.md) is a GEMV, bound by the DDR bandwidth the
  NPU shares with the CPU, so decode stays on the CPU. At 928 MHz the NPU streams resident weights
  at ~16 GB/s over three cores, which only matches the CPU's decode stream [HW sweep 2026-10-21].
  The NPU's fp16 throughput at M=1 is ~82x below its batched-GEMM peak, as an external benchmark
  measured it through the vendor RKNN matmul API.
- **[Clock](perf/clock.md).** The NPU clock boots at 198 MHz, one fifth of its 1 GHz maximum
  [source-confirmed]. Above that it is a PVTPLL whose frequency follows the rail voltage, so a
  requested rate is not the clock. At a 0.80 V rail the 900 MHz request runs at 928 MHz, the
  vendor's 900 MHz pairing. It gives 1.66x on Gemma-4-12B prefill over the boot clock
  [HW sweep 2026-11-02]. On matmul device time 928 MHz gives 1.044x over the 600 MHz request
  (~816 MHz), and 1019 MHz, a 1 GHz request at 0.85 V, 1.083x [HW sweep]. Known issue E10 is
  the hazard in raising it.

## Reading paths

| Task | Start with |
|---|---|
| Run a first register program | [guide/](guide/), then [matmul-as-conv.md](matmul-as-conv.md) and [encodings/regcmd-task-model.md](encodings/regcmd-task-model.md) |
| Port to another SoC | [chips/README.md](chips/README.md), then [chips/porting-patterns.md](chips/porting-patterns.md) |
| Add an operator | [encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md), [encodings/feature-reduce.md](encodings/feature-reduce.md) and [encodings/sdp-stage-precision.md](encodings/sdp-stage-precision.md) |
| Speed up LLM prefill | [TUNING.md](TUNING.md), then [perf/not-mac-bound.md](perf/not-mac-bound.md) |
| Run a model | [MODEL-NOTES.md](MODEL-NOTES.md), then [TUNING.md](TUNING.md) |

## Document index

### Foundations

| Document | Covers |
|---|---|
| [GLOSSARY.md](GLOSSARY.md) | The notes' own terms, the NPU's block and stage names, and the abbreviations of the models the notes run, each with its source |
| [hardware-overview.md](hardware-overview.md) | What the NPU is: its NVDLA lineage, the block pipeline, the cores and CBUF, and its rated and measured throughput |
| [nvdla-lineage.md](nvdla-lineage.md) | What the NVDLA's open documentation says about this NPU, where it is wrong for this silicon, and the Rockchip additions it does not describe |
| [datatypes.md](datatypes.md) | The datatype capability matrix: precision field, output type, MAC rate and use, per datatype |
| [matmul-as-conv.md](matmul-as-conv.md) | How a matmul runs as a 1×1 convolution: the data flow, the layouts, tiling, the alignment rules, and a KxK convolution run as an im2col matmul |
| [depthwise-conv.md](depthwise-conv.md) | How a depthwise convolution differs from a direct one |

### Per-SoC sheets

| Document | Covers |
|---|---|
| [chips/README.md](chips/README.md) | The per-SoC status table, and what varies between SoCs |
| [chips/rk3588.md](chips/rk3588.md) | The RK3588's machine parameters, register map and SoC integration, and the per-core float difference on output lane 0 |
| [chips/rk3576.md](chips/rk3576.md) | The RK3576: supported operations, limits and errata, and the cost of its library entries |
| [chips/rk3576-regcmd.md](chips/rk3576-regcmd.md) | The RK3576 register encoding and its hardware verification |
| [chips/rk3576-hazards.md](chips/rk3576-hazards.md) | The RK3576's wide-output hazard, drain deadline and pooling divisor lag in detail, and its unresolved hazards |
| [chips/rk3566.md](chips/rk3566.md) | The RK3566, from published sources, not yet verified on hardware |
| [chips/porting-patterns.md](chips/porting-patterns.md) | Which facts transfer between NPU revisions, and which invert |

### Encodings

| Document | Block | Covers |
|---|---|---|
| [encodings/precision-field.md](encodings/precision-field.md) | CNA, CORE, DPU | The 3-bit precision values for all six datatypes |
| [encodings/tile-layouts.md](encodings/tile-layouts.md) | CNA, DPU_RDMA | The feature, weight and output cube layouts, per datatype |
| [encodings/size-e-quirk.md](encodings/size-e-quirk.md) | DPU write-out | The `size_e=7` stride of the int8 and int4 outputs |
| [encodings/output-transpose-int16.md](encodings/output-transpose-int16.md) | DPU write-out | int16's saturating int32 writer, the transposed writer, and the byte-decomposition path |
| [encodings/out-cvt-converter.md](encodings/out-cvt-converter.md) | DPU write-out | The integer output converter `(acc×SCALE)>>SHIFT`, its tie-to-even rounding, and the sign bit in its scale |
| [encodings/sdp-stage-precision.md](encodings/sdp-stage-precision.md) | DPU SDP, CORE | The precision of the BS, BN and EW stages, the EW stage's integer mode, and why CACC cannot accumulate across ops |
| [encodings/k-accumulation.md](encodings/k-accumulation.md) | DPU-EW, DPU_RDMA | Summing K-partials on-chip: the supported fp16 path, and what an integer path still needs |
| [encodings/cross-op-chaining.md](encodings/cross-op-chaining.md) | CNA, DPU | Feeding one fp16 op's output cube to the next op with no host [de-tile](GLOSSARY.md) |
| [encodings/regcmd-task-model.md](encodings/regcmd-task-model.md) | PC | What a task is, how a [delta task](GLOSSARY.md) inherits register state, and which pointers the kernel sets |
| [encodings/cbuf-reuse.md](encodings/cbuf-reuse.md) | CNA (CBUF) | The `WEIGHT_REUSE` and `DATA_REUSE` operand-reuse bits |
| [encodings/cbuf-bank-slack.md](encodings/cbuf-bank-slack.md) | CNA (CBUF) | The int8 feature DMA's one-bank over-read |
| [encodings/cna-dcomp-weight-decompression.md](encodings/cna-dcomp-weight-decompression.md) | CNA ([DCOMP](GLOSSARY.md)) | The weight-decompression block, decoded |
| [encodings/mrdma-trap.md](encodings/mrdma-trap.md) | DPU_RDMA | The register block a job must emit to avoid a timeout |
| [encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md) | DPU LUT | Activations and elementary functions through the LUT, the elementwise ops, and the LUT's hazards |
| [encodings/conv-transpose.md](encodings/conv-transpose.md) | CNA | ConvTranspose: the hardware deconvolution mode, the lowering onto a forward convolution, and the resident path |
| [encodings/resize-upsample.md](encodings/resize-upsample.md) | CNA | Nearest and bilinear resize as a depthwise transposed convolution |
| [encodings/ppu-pooling.md](encodings/ppu-pooling.md) | PPU | Max and average pooling |
| [encodings/ppu-reduce-mean.md](encodings/ppu-reduce-mean.md) | PPU | Global average, max and min pooling over the spatial axes |
| [encodings/feature-reduce.md](encodings/feature-reduce.md) | CNA, CORE, DPU | Reducing over the feature axis, and prefix sums, as a matmul against a ones weight |

### Application notes

| Document | Covers |
|---|---|
| [encodings/whisper-encoder.md](encodings/whisper-encoder.md) | A full Whisper encoder block on the NPU, and the operators it is built from |
| [encodings/siglip-encoder.md](encodings/siglip-encoder.md) | The SigLIP-B/16 vision encoder end to end on the NPU |
| [encodings/ffn-block.md](encodings/ffn-block.md) | The gated [FFN](GLOSSARY.md) (GeGLU, SwiGLU) |
| [encodings/rmsnorm-onnpu.md](encodings/rmsnorm-onnpu.md) | RMSNorm from a square, a feature reduce and a host rsqrt |
| [encodings/norm-vision.md](encodings/norm-vision.md) | The vision normalizations, on the LayerNorm machinery |

### Performance

| Document | Covers |
|---|---|
| [perf/not-mac-bound.md](perf/not-mac-bound.md) | Why the whole-tile matmul program measures within ~20% at every datatype, and what CBUF weight streaming gives |
| [perf/device-vs-host-split.md](perf/device-vs-host-split.md) | How much of a matmul's wall-clock time the device takes, with resident and re-packed weights |
| [perf/decode-gemv.md](perf/decode-gemv.md) | Why single-row decode stays on the CPU |
| [perf/asymmetric-tile.md](perf/asymmetric-tile.md) | Why an Mt>Nt tile beats the symmetric maximum |
| [perf/weight-residency-fusion.md](perf/weight-residency-fusion.md) | Packing F16 weights once, and fusing projections that share an input |
| [perf/quant-prefill-microbatch.md](perf/quant-prefill-microbatch.md) | Why quantized-GGUF prefill is bound by per-micro-batch dequantization |
| [perf/moe-placement.md](perf/moe-placement.md) | Which MoE expert GEMMs the default places on the NPU: the work-per-dispatch threshold and the residency budget |
| [perf/attention-offload-crossover.md](perf/attention-offload-crossover.md) | The context length where offloading prefill attention starts to win |
| [perf/rk3576-w8a8-llm.md](perf/rk3576-w8a8-llm.md) | LLM prefill on the RK3576's int8 matmul through llama.cpp: speed, accuracy and the per-column calibration |
| [perf/rk3576-graph.md](perf/rk3576-graph.md) | Whole networks on the RK3576: the graph planner, the TFLite delegate's results, and the terms of a graph's wall-clock time |
| [perf/iova-and-multicore.md](perf/iova-and-multicore.md) | The per-fd 4 GB IOVA window, and one fd per core |
| [perf/bo-sync-cost.md](perf/bo-sync-cost.md) | The `PREP_BO`/`FINI_BO` cache-sync cost, which scales with BO size |
| [perf/pool-completion.md](perf/pool-completion.md) | Why a pool submit on mainline `rocket` waits ~507 ms and resets a core |
| [perf/clock.md](perf/clock.md) | The boot clock, the requested rate against the physical PVTPLL clock, raising it safely, and the gain to 1019 MHz |
| [perf/cpu-governor-and-offload.md](perf/cpu-governor-and-offload.md) | How the CPU governor biases an NPU-against-CPU comparison, what it costs a per-layer offload inside a model, and how to pin it |
| [perf/cpu-repack-baseline.md](perf/cpu-repack-baseline.md) | Why a quantized CPU baseline needs llama.cpp's weight repack, and the NPU's lead over a repacked CPU |
| [perf/hw-byte-counters.md](perf/hw-byte-counters.md) | The NPU's missing DMA byte counters, and the DDR controller PMU that measures board traffic instead |
| [perf/ppu-pooling-not-detile.md](perf/ppu-pooling-not-detile.md) | The PPU as a pooling engine that cannot de-tile |
| [perf/rga-detile.md](perf/rga-detile.md) | Why the RGA engine de-tiles bit-exactly but loses to the CPU |
| [perf/sram-nbuf.md](perf/sram-nbuf.md) | How the NPU reaches system SRAM, and why it gives little speedup |
| [perf/asr-cpu-relief.md](perf/asr-cpu-relief.md) | [ASR](GLOSSARY.md) measured in CPU core-seconds |
| [perf/asr-streaming-service.md](perf/asr-streaming-service.md) | `whisper-server` fed fixed-length chunks, in CPU core-seconds per second of audio |
| [perf/per-process-readout.md](perf/per-process-readout.md) | The per-process instrument behind each timed configuration, and its silent failures |
| [perf/benchmarks.md](perf/benchmarks.md) | The consolidated benchmark record: method and per-model results |
| [perf/data/](perf/data/) | Per-model measurement records and the benchmark scripts |

### Running models

| Document | Covers |
|---|---|
| [MODEL-NOTES.md](MODEL-NOTES.md) | Per-model behavior on the stack: faithfulness checks, sampling settings, quirks |
| [TUNING.md](TUNING.md) | Which options to set for a workload |
| [guide/](guide/) | The start-to-finish walkthrough across the driver library, the frontends and the kernel patches |

### Sources and captures

| Document | Covers |
|---|---|
| [SOURCES.md](SOURCES.md) | Every external source, and what each covers |
| [ppu-rknn-capture/](ppu-rknn-capture/) | The vendor compiler's PPU pooling program, captured and decoded |
| [teflon-add-capture/](teflon-add-capture/) | A Teflon regcmd capture for the elementwise-operand work |
| [whisper-encoder-validation/](whisper-encoder-validation/) | Scripts that reproduce the Whisper encoder validation |

## Conventions

A claim has a tag that says how it was established:

| Tag | Means |
|---|---|
| `[HW sweep]` | Observed on the device by sweeping a value or geometry, and scoring each result as [bit-exact](GLOSSARY.md), garbage, or nothing written. |
| `[source-confirmed]` | Corroborated by an authoritative source, named at the claim: most often the Mesa `rocket` (Teflon) driver, the open NVDLA documentation, the `npu_hw.h` register headers from Jasbir Matharu's `rk3588-npu`, the vendor BSP kernel, or the kernel source and patch series. [SOURCES.md](SOURCES.md) lists them. |
| `[TRM]` | Stated in Rockchip's Technical Reference Manual. |
| `[expected]` | An outcome not yet measured: a prediction, a derived bound, or an estimate. |
| `[hypothesis]` | A proposed cause for a measured effect, not yet isolated. The effect is established, and the explanation is not. |
| `[live]` | Read once from a running board: a clock rate, a regulator state, a device-tree property, a module version. It states the date and the kernel, because a reflash can change it. |
| `[host-computed]` | Computed on the host with no device run: arithmetic on recorded numbers, a planner's output, a census of a model file. The computation is exact, and its inputs have their own tags. An estimate is `[expected]` instead. |

Where a fact is measured and a source agrees, both tags appear. Performance figures state the
SoC, the NPU clock and the tensor shape, in a **Conditions:** line under the section or table
they apply to.

On the RK3588, the NPU clock in a Conditions line is the physical frequency, not the requested
rate. Above 200 MHz the two differ, because the clock is a PVTPLL that follows the rail voltage
([perf/clock.md](perf/clock.md)). On the Turing RK1, ~816 MHz is the 600 MHz request
(`rocket_npu_clk_hz=600000000`) at a 0.80 V rail. The 900 MHz request runs at 928 MHz, the
clock of the RK1 measurements from 2026-10-19 on. A rate named for the vendor kernel is that driver's OPP, whose
physical frequency is not characterized.

A note's record, where it has one, is an Evidence section at its end. It states how each fact
was established, including the fields and values a negative result searched. A hazard is marked as
a Caution callout, and a defect as an erratum entry with its conditions, symptom and workaround.
[GLOSSARY.md](GLOSSARY.md) defines the notes' own terms, the NPU's block and stage names, and the
abbreviations of the models the notes run. It also names the source of each.

## Test platforms

Most findings come from devices running mainline 7.1 and 7.2 kernels:

- A Turing RK1: RK3588, 32 GB
- An H96 MAX M9: RK3576

A second Turing RK1, on Rockchip's 6.1 BSP kernel with the vendor `rknpu` 0.9.8 driver, provides
the vendor-driver results.

The same work produced a FOSS inference stack on `rocket`: a userspace library, frontends for
ggml, TFLite and ONNX Runtime, and kernel patches. The [guide](guide/) is a start-to-finish
walkthrough that ties the driver library, the frontends and the kernel patches together.

## License

The documentation in this repository (the prose notes, tables, and encoding write-ups) is
licensed under [Creative Commons Attribution 4.0 International](LICENSE) (CC-BY-4.0): reuse it
freely with attribution.

The scripts and programs (every `.py`, `.sh` and `.c` file, each with an
`SPDX-License-Identifier: MIT` line) are licensed under the [MIT License](LICENSE-CODE). The C
programs in `perf/data/rknn-encoder/` and `perf/data/rknn-decoder/` link Rockchip's proprietary
`librknnrt`. The generators in `lut-rknn-capture/` and `ppu-rknn-capture/` and the
`perf/data/siglip-rknn/` harness import Rockchip's proprietary rknn-toolkit2 or
rknn-toolkit-lite2. Those come from Rockchip under its own terms. Third-party captures retain
their upstream copyright and license terms, notably `ppu-rknn-capture/registers.xml` (from the
Mesa `rocket`/Teflon driver), credited in [SOURCES.md](SOURCES.md).
