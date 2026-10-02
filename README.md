# Rockchip NPU reverse-engineering notes

## AI disclosure

The documents in rockchip-npu-notes were authored by AI, primarily Claude. They were produced as
part of a series of side projects, and are published here in case they are of use to others.
Accuracy of information is not guaranteed.

## About rockchip-npu-notes

These notes record how the Rockchip NPU behaves when driven through the mainline `rocket`
DRM-accel driver. They describe the silicon and its register-command interface, organized by
subsystem, and are independent of any one project built on them.

Most of the content is IP-inherent: true of the NVDLA-derived rknpu block on any Rockchip SoC
that carries it. What varies per chip lives in the per-SoC sheets under [chips/](chips/):
machine parameters, the register offset map, and SoC integration. The RK3588 is the reference
part and the one most notes describe. The RK3576 also computes on hardware, and the RK3566 is
planned.

The findings come from reverse-engineering on real devices running mainline 7.1 and 7.2 kernels:
a Turing RK1 (RK3588, 32 GB) and an H96 MAX M9 (RK3576). That work also produced a FOSS
inference stack on `rocket`: a userspace library, frontends for ggml, TFLite and ONNX Runtime,
and kernel patches.

The notes serve someone running their own compute on the NPU, through `rocket` or any other raw
register-command path. For a start-to-finish walkthrough that ties the driver library, the
frontends and the kernel patches together, see the [guide](guide/).

## Evidence tags

Claims carry a tag that says how each was established:

| Tag | Means |
|---|---|
| `[HW sweep]` | Observed on the device by sweeping a value or geometry, and scoring each result as bit-exact, garbage, or nothing written. The strongest grade. |
| `[source-confirmed]` | Corroborated by an authoritative FOSS source, named at the claim: the Mesa `rocket` (Teflon) driver, the open NVDLA documentation, or the `npu_hw.h` register headers from Jasbir Matharu's `rk3588-npu`. [SOURCES.md](SOURCES.md) lists them. |
| `[TRM]` | Stated in Rockchip's Technical Reference Manual. |
| `[expected]` | An outcome not yet measured: a prediction, a derived bound, or an estimate. |
| `[hypothesis]` | A proposed cause for a measured effect, not yet isolated. The effect is established, and the explanation is not. |

Where a fact is measured and a source agrees, both tags appear. A performance number carries its
operating point: the part, the clock, and the shape. Negative results are recorded as carefully
as positive ones. Each states what its method searched, so a limit of the method does not read
as a limit of the silicon.

## Key facts

The RK3588 NPU is a three-core accelerator derived from NVIDIA's NVDLA. The `rocket` driver is a
generic register-command submitter (`CREATE_BO`, `SUBMIT`, `PREP_BO`, `FINI_BO`): the kernel runs
whatever CNA→CORE→DPU program userspace hands it. A matmul is a 1×1 convolution, so it runs on
the same program Mesa emits for a convolution.

The facts below are the RK3588's. [chips/porting-patterns.md](chips/porting-patterns.md) says
which of them carry to the RK3576, and [chips/rk3576.md](chips/rk3576.md) has the RK3576's own.
Each label links to the note that owns the fact and its evidence.

### Capabilities

- **[Datatypes](datatypes.md).** The integer types are int4, int8 and int16, and the float types
  are fp16, bf16 and tf32. Each has a working native matmul [HW sweep]. int16's int32 output
  saturates, so an int64-exact int16 matmul takes four int8 matmuls.
- **[Layouts](matmul-as-conv.md).** No register converts a layout, so weights must be scattered
  into the cube layout on the host [source-confirmed]. A matmul's activation and output need no
  packing: a stride-only form reads A and writes C as plain rows [HW sweep]. Its device cost
  grows with the output's row pitch once a row passes 4 KiB.
- **[K-accumulation](encodings/k-accumulation.md).** The DPU eltwise unit sums fp16 K-partials
  on-chip, reading each output tile once [HW sweep, source-confirmed]. The eltwise ALU also adds
  int8, int16 and int32 exactly once every precision field carries one integer code [HW sweep].
  An integer K-accumulation built on it is not measured. It is not a speed lever either, because
  readback does not set the prefill wall.
- **[On-chip requant](encodings/out-cvt-converter.md).** The DPU's output converter requantizes
  an int32 accumulator with an integer multiplier and a shift. Its BS stage multiplies per
  output channel, which carries TFLite's per-axis scales [HW sweep]. An int8 convolution
  therefore writes int8 with no host requant.
- **[Cores and address space](perf/iova-and-multicore.md).** Under `rocket`, each open file
  descriptor is one scheduling entity with its own 4 GB IOVA window [HW sweep]. Reaching all
  three cores takes three or more fds, because one fd's jobs serialize onto one core.
- **[Operations beyond matmul](encodings/dpu-lut-activation.md).** The DPU's LUT computes
  activations such as SiLU and GELU, and functions such as exp and rsqrt [HW sweep]. The PPU runs
  max and average pooling. Composed with the matmul, they run a
  [full Whisper encoder block](encodings/whisper-encoder.md) on the NPU. The CNA also has a
  [hardware deconvolution mode](encodings/conv-transpose.md) that computes a power-of-two-stride
  ConvTranspose in one task.

### Traps

Each of these produces a wrong answer or a stalled job, with no error that names the cause:

- **[The MRDMA block](encodings/mrdma-trap.md).** A plain convolution or matmul must set
  `mrdma_disable` in `DPU_RDMA_FEATURE_MODE_CFG` (`0x5044`). Left at its default, the DPU read
  DMA waits for data that never arrives. The job times out with its output untouched
  [HW sweep, source-confirmed].
- **[Output stride](encodings/size-e-quirk.md).** The int8 and int4 outputs stride as `size_e=7`
  whatever their byte width, and the natural width leaves most of the output unwritten
  [HW sweep]. int16's int32 output takes the natural `size_e=3`, and hangs at 7.
- **[Matmul height](matmul-as-conv.md).** Matmul rows are the convolution's spatial height, and a
  height below 4 mis-computes at every datatype [HW sweep]. `M%4==0` is the real constraint, so
  pad a single-row matmul to 4 rows.
- **[CBUF bank slack](encodings/cbuf-bank-slack.md).** At some tile geometries the int8 feature
  DMA over-reads by one CBUF bank and returns garbage in the tile's last rows [HW sweep].
  Reserve `data_bank = fd_banks+1`. The fp16 cube is immune.
- **[LUT table entries](encodings/dpu-lut-activation.md).** A LUT table entry of exactly `q=0`
  decodes to ~4.0 rather than 0, so floor every entry to `q>=1` [HW sweep].
- **[Requant rounding and sign](encodings/out-cvt-converter.md).** The requant rounds a tie to
  even unless bit 30 of `OUT_CVT_SHIFT` is set, and bit 15 of its scale is a sign [HW sweep].
  Mesa's derivation of the scale drops a carry, which halves or negates one scale in 16384. The
  BS stage's shift is split by sign across two fields, and a program that sets one computes
  negative products at the wrong scale.
- **[Per-core float difference](chips/rk3588.md).** The three cores compute an fp16 matmul
  identically except on output lane 0 of each 16-channel group. There each core rarely writes
  2^-16 low, on its own set of inputs [HW sweep]. `rocket` places a job on a core by load, so
  one fp16 run compared to another compares core draws. The integer paths are core-invariant.
- **[Convolution tile limits](matmul-as-conv.md).** One convolution task takes at most 1022
  input rows by 2047 columns. A wider tile wraps its field and computes a plausible wrong
  surface [HW sweep].
- **[Cube-dimension ceiling](encodings/dpu-lut-activation.md).** Stay under the 13-bit maximum of
  8191 in every cube-dimension field. A LUT chunk at exactly that width corrupts ~54 positions
  [HW sweep], and SHARD reports a `REGTASK Overflow` for any operand index past it
  [source-confirmed]. The CNA's input width and height are 11 bits, narrower still.

### Performance and clock

- **[Matmul throughput](perf/not-mac-bound.md).** At the current operating point (resident
  weights, multicore, 600 MHz), a matmul is bound by DMA and dispatch rather than by the MAC
  array [HW sweep]. fp16, int8 and int4 all land near 460 GOP/s, so quantization buys memory
  rather than prefill speed. This holds while that bottleneck binds, and is not a property of
  the silicon.
- **[Decode](perf/decode-gemv.md).** Single-row decode is a GEMV, bound by the DDR bandwidth the
  NPU shares with the CPU. The NPU is ~82x slower at M=1 than at the batched GEMM it was built
  for, so decode stays on the CPU [HW sweep].
- **[Clock](perf/clock.md).** The NPU clock boots parked at 200 MHz, one fifth of its 1 GHz
  maximum [source-confirmed]. Setting the rate while the power domain is off wedges the
  firmware [HW sweep]. Raise it from inside the driver, after runtime PM powers the domain.
  The operating point is 600 MHz (~1.43x on prefill), and 900 MHz adds no speed.

## Document map

Each file owns one mechanism or one finding. The descriptions below say what each covers, and
the file holds the claims and their evidence.

### Foundations

| Document | Covers |
|---|---|
| [hardware-overview.md](hardware-overview.md) | What the NPU is: its NVDLA lineage, the block pipeline, the cores and CBUF, and rated against measured throughput |
| [nvdla-lineage.md](nvdla-lineage.md) | What the NVDLA's open documentation carries for this part, where it is wrong for this silicon, and the Rockchip additions it does not describe |
| [datatypes.md](datatypes.md) | The datatype capability matrix: precision field, output type, MAC rate and use, per datatype |
| [matmul-as-conv.md](matmul-as-conv.md) | How a matmul runs as a 1×1 convolution: the data flow, the layouts, tiling, and the alignment rules |
| [depthwise-conv.md](depthwise-conv.md) | How a depthwise convolution differs from a direct one |

### Per-SoC sheets

| Document | Covers |
|---|---|
| [chips/README.md](chips/README.md) | The per-SoC status table, and what varies between parts |
| [chips/rk3588.md](chips/rk3588.md) | The RK3588's machine parameters, register map and SoC integration, and the per-core float difference on output lane 0 |
| [chips/rk3576.md](chips/rk3576.md) | The RK3576: what computes, its bounds and hazards, and the graph and frontend results |
| [chips/rk3576-regcmd.md](chips/rk3576-regcmd.md) | The RK3576 register encoding as an emitter, and what running it on silicon settles |
| [chips/rk3566.md](chips/rk3566.md) | The RK3566, from sources ahead of hardware |
| [chips/porting-patterns.md](chips/porting-patterns.md) | Which facts transfer between NPU revisions, and which invert |

### Encodings

| Document | Block | Covers |
|---|---|---|
| [encodings/precision-field.md](encodings/precision-field.md) | CNA, CORE, DPU | The 3-bit precision values for all six datatypes |
| [encodings/tile-layouts.md](encodings/tile-layouts.md) | CNA, DPU-RDMA | The feature, weight and output cube layouts, per datatype |
| [encodings/size-e-quirk.md](encodings/size-e-quirk.md) | DPU write-out | The `size_e=7` stride of the int8 and int4 outputs |
| [encodings/output-transpose-int16.md](encodings/output-transpose-int16.md) | DPU write-out | int16's saturating int32 writer, the transposed writer, and the byte-decomposition path |
| [encodings/out-cvt-converter.md](encodings/out-cvt-converter.md) | DPU write-out | The integer output converter `(acc×SCALE)>>SHIFT`, its tie-to-even rounding, and the sign bit in its scale |
| [encodings/sdp-stage-precision.md](encodings/sdp-stage-precision.md) | DPU SDP, CORE | The precision of the BS, BN and EW stages, the EW stage's integer mode, and why CACC cannot accumulate across ops |
| [encodings/k-accumulation.md](encodings/k-accumulation.md) | DPU-EW, DPU-RDMA | Summing K-partials on-chip: the fp16 path that ships, and what an integer path still needs |
| [encodings/cross-op-chaining.md](encodings/cross-op-chaining.md) | CNA, DPU | Feeding one fp16 op's output cube to the next op with no host de-tile |
| [encodings/regcmd-task-model.md](encodings/regcmd-task-model.md) | PC | What a task is, how a delta task inherits register state, and which pointers the kernel owns |
| [encodings/cbuf-reuse.md](encodings/cbuf-reuse.md) | CNA (CBUF) | The `WEIGHT_REUSE` and `DATA_REUSE` operand-reuse bits |
| [encodings/cbuf-bank-slack.md](encodings/cbuf-bank-slack.md) | CNA (CBUF) | The int8 feature DMA's one-bank over-read |
| [encodings/cna-dcomp-weight-decompression.md](encodings/cna-dcomp-weight-decompression.md) | CNA (DCOMP) | The weight-decompression block, decoded |
| [encodings/mrdma-trap.md](encodings/mrdma-trap.md) | DPU-RDMA | The register block a job must emit to avoid a timeout |
| [encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md) | DPU LUT | Activations and elementary functions through the LUT, the elementwise ops, and the LUT's traps |
| [encodings/conv-transpose.md](encodings/conv-transpose.md) | CNA | ConvTranspose: the hardware deconvolution mode, the lowering onto a forward convolution, and the resident route |
| [encodings/resize-upsample.md](encodings/resize-upsample.md) | CNA | Nearest and bilinear resize as a depthwise transposed convolution |
| [encodings/ppu-pooling.md](encodings/ppu-pooling.md) | PPU | Max and average pooling |
| [encodings/ppu-reduce-mean.md](encodings/ppu-reduce-mean.md) | PPU | Global average, max and min pooling over the spatial axes |
| [encodings/feature-reduce.md](encodings/feature-reduce.md) | CNA, CORE, DPU | Reducing over the feature axis, and prefix sums, as a matmul against a ones weight |

### Compositions

| Document | Covers |
|---|---|
| [encodings/whisper-encoder.md](encodings/whisper-encoder.md) | A full Whisper encoder block on the NPU, and the operators it is built from |
| [encodings/siglip-encoder.md](encodings/siglip-encoder.md) | The SigLIP-B/16 vision encoder end to end on the NPU |
| [encodings/ffn-block.md](encodings/ffn-block.md) | The gated-MLP FFN (GeGLU, SwiGLU) |
| [encodings/rmsnorm-onnpu.md](encodings/rmsnorm-onnpu.md) | RMSNorm from a square, a feature reduce and a host rsqrt |
| [encodings/norm-vision.md](encodings/norm-vision.md) | The vision normalizations, on the LayerNorm machinery |

### Performance

| Document | Covers |
|---|---|
| [perf/not-mac-bound.md](perf/not-mac-bound.md) | Why matmul throughput sits near 460 GOP/s at every datatype |
| [perf/device-vs-host-split.md](perf/device-vs-host-split.md) | How much of a matmul's wall the device holds, resident against re-packed |
| [perf/decode-gemv.md](perf/decode-gemv.md) | Why single-row decode stays on the CPU |
| [perf/asymmetric-tile.md](perf/asymmetric-tile.md) | Why an Mt>Nt tile beats the symmetric maximum |
| [perf/weight-residency-fusion.md](perf/weight-residency-fusion.md) | Packing F16 weights once, and fusing projections that share an input |
| [perf/quant-prefill-microbatch.md](perf/quant-prefill-microbatch.md) | Why quantized-GGUF prefill is bound by per-micro-batch dequantization |
| [perf/attention-offload-crossover.md](perf/attention-offload-crossover.md) | The context length where offloading prefill attention starts to win |
| [perf/iova-and-multicore.md](perf/iova-and-multicore.md) | The per-fd 4 GB IOVA window, and one fd per core |
| [perf/bo-sync-cost.md](perf/bo-sync-cost.md) | The `PREP_BO`/`FINI_BO` cache-sync cost, which scales with BO size |
| [perf/pool-completion.md](perf/pool-completion.md) | Why a pool submit on mainline `rocket` waits ~507 ms and resets a core |
| [perf/clock.md](perf/clock.md) | The 200 MHz boot clock, raising it safely to 600 MHz, and the limits past it |
| [perf/cpu-governor-and-offload.md](perf/cpu-governor-and-offload.md) | How the CPU governor biases an NPU-against-CPU comparison, and how to pin it |
| [perf/cpu-repack-baseline.md](perf/cpu-repack-baseline.md) | Why a quantized CPU baseline needs llama.cpp's weight repack, and the NPU's lead over a repacked CPU |
| [perf/hw-byte-counters.md](perf/hw-byte-counters.md) | The NPU's missing DMA byte counters, and the DDR controller PMU that measures board traffic instead |
| [perf/ppu-pooling-not-detile.md](perf/ppu-pooling-not-detile.md) | The PPU as a pooling engine that cannot de-tile |
| [perf/rga-detile.md](perf/rga-detile.md) | Why the RGA engine de-tiles bit-exactly but loses to the CPU |
| [perf/sram-nbuf.md](perf/sram-nbuf.md) | How the NPU reaches system SRAM, and why it is a weak lever |
| [perf/asr-cpu-relief.md](perf/asr-cpu-relief.md) | Speech-to-text measured in CPU core-seconds |
| [perf/asr-streaming-service.md](perf/asr-streaming-service.md) | `whisper-server` fed fixed-length chunks, in CPU core-seconds per second of audio |
| [perf/per-process-readout.md](perf/per-process-readout.md) | The per-process instrument behind each timed arm, and its silent failures |
| [perf/benchmarks.md](perf/benchmarks.md) | The consolidated benchmark record: method and per-model results |
| [perf/data/](perf/data/) | Per-model measurement records and the benchmark scripts |

### Running models

| Document | Covers |
|---|---|
| [MODEL-NOTES.md](MODEL-NOTES.md) | Per-model behavior on the stack: faithfulness checks, sampling settings, quirks |
| [TUNING.md](TUNING.md) | Which flags to set for a workload |
| [guide/](guide/) | The start-to-finish walkthrough across the driver library, the frontends and the kernel patches |

### Sources and captures

| Document | Covers |
|---|---|
| [SOURCES.md](SOURCES.md) | Every external source, and what each was good for |
| [ppu-rknn-capture/](ppu-rknn-capture/) | The vendor compiler's PPU pooling program, captured and decoded |
| [teflon-add-capture/](teflon-add-capture/) | A Teflon regcmd capture for the elementwise-operand work |
| [whisper-encoder-validation/](whisper-encoder-validation/) | Scripts that reproduce the Whisper encoder validation |

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
