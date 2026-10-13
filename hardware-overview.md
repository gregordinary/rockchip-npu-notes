# RK3588 NPU hardware overview

The RK3588 NPU is a convolution engine derived from NVIDIA's open NVDLA (Deep Learning
Accelerator). It has no matmul primitive, so software runs a matmul as a 1×1 convolution
([matmul-as-conv.md](matmul-as-conv.md)). Software drives it through the mainline `rocket`
driver by submitting register programs (§"Kernel interface").
[nvdla-lineage.md](nvdla-lineage.md) maps each block to its NVDLA counterpart.

Features:

- Three cores, which run independently (§"Cores")
- Per core, a fixed pipeline of blocks, CNA→CORE→DPU (§"Block pipeline")
- Per core, a 384 KB CBUF, which holds a task's input-feature and weight tiles (§"CBUF")
- Six datatypes: int4, int8, int16, fp16, bf16 and tf32 (§"Datatypes")
- 6 TOPS rated by Rockchip, the int8-convolution peak (§"Throughput")

## Block pipeline

The pipeline is a fixed sequence of blocks, CNA→CORE→DPU:

- **CNA**: the feature and weight load into the CBUF, and the sequence controller that feeds
  both to the MAC array (NVDLA CDMA and CSC).
- **CORE**: the MAC array and its accumulator (NVDLA CMAC and CACC). Its register block
  contains the MAC gating register, `MAC_GATING` (`0x300C`) [TRM].
- **DPU**: the post-processing (data-processing) unit, which is the NVDLA SDP.

The DPU has the BS, BN and EW affine stages, an NVDLA-style LUT for nonlinear activations,
and its read DMA, DPU_RDMA. EW is the eltwise sub-unit.

The register bases are [source-confirmed: Mesa `registers.xml`]:

| Block | Register base |
|---|---|
| PC | `0x0xxx` |
| CNA | `0x1xxx` |
| CORE | `0x3xxx` |
| DPU | `0x4xxx` |
| DPU_RDMA | `0x5xxx` |

The DPU computes more than multiply-accumulate. The EW unit runs elementwise add and mul
(residuals, gated activations). The LUT runs sigmoid, tanh, SiLU, GELU, sqrt, rsqrt,
reciprocal and exp on-chip.

The DPU, composed with the matmul, runs RMSNorm, LayerNorm and softmax on the NPU. The PPU
cannot reduce channels, so the feature-axis reduce they need runs as a matmul against a ones
weight ([encodings/feature-reduce.md](encodings/feature-reduce.md)). The host computes only
per-row scalars: a norm's reciprocal square root, and a softmax's row maximum and reciprocal of
its sum. The same composition runs a full transformer or Whisper encoder block on the NPU. See
[encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md) and
[encodings/whisper-encoder.md](encodings/whisper-encoder.md).

### Accumulators

Accumulators are wide and NVDLA-shaped: int8 accumulates in INT34->INT32, int16 in INT48,
and fp16 in FP44/FP48->FP32. The CACC integer accumulator is 48-bit for int16 and 34-bit
for int8 [source-confirmed]. The source is a note in Mesa `rkt_coefs.c`, cited in
[encodings/output-transpose-int16.md](encodings/output-transpose-int16.md).

The accumulator width keeps large-K fp16 from losing range. Incoherent output from a
quantized model therefore comes from activation quantization, not from the accumulator.

### Data layouts

Data lives in DRAM in NVDLA-style cube layouts with an atomic channel block ("C2"). Software
scatters the weights into them on the host. A matmul's activation and output can instead stay
row-major, by programming the strides ([matmul-as-conv.md](matmul-as-conv.md) §"Layouts"). See
[encodings/tile-layouts.md](encodings/tile-layouts.md).

### Index operations

The hardware has no gather, scatter or indexed read. Every block streams contiguous tiles
through fixed DMA, and there is no indexed-fetch datapath. An index op (a class-target gather
for cross-entropy, an embedding lookup, `Gather`/`Slice`-by-index) therefore runs on the host.
It is like the host `1/s` of softmax: M scalar lookups, correct and free. A gather is neither a
contraction (matmul) nor a pool (PPU), so no on-chip block supplies it.

Anything linear along the contraction axis is a matmul against a ones weight, which can be
triangular, and needs no new regcmd. The ones-weight matmul covers a full reduce, a weighted
reduce, and a prefix scan (cumsum). See [encodings/feature-reduce.md](encodings/feature-reduce.md).

## Cores

The NPU has 3 cores (`npu@fdab0000`, `fdac0000`, `fdad0000` in the device tree). They run
concurrently and independently: with one file descriptor per core, throughput is ~3x one
core's [HW sweep].

Under `rocket`, software reaches the cores by opening 3+ file descriptors, rather than by a
core mask in the submit. The driver creates one `drm_sched` per core but one scheduling entity
per fd. An entity stays on one core while it has a job queued or running. One fd's jobs
run one at a time, and a lone fd that waits on each job alternates between cores 0 and 1. N fds
spread across the N cores [HW sweep, source-confirmed]. See
[perf/iova-and-multicore.md](perf/iova-and-multicore.md).

## CBUF

Each core has a CBUF of 12 banks × 32 KB = 384 KB [source-confirmed: Mesa / SHARD]. During
a conv or matmul task the CBUF must hold both the input-feature tile and the weight tile.
That constraint sets the K-tile size. With the input and weight both resident, the banks
bound the contraction depth `Kt` that fits at a given output tile (Mt × Nt).

The size limits that the SHARD paper ([SOURCES.md](SOURCES.md)) reports for its ViT encoder
are rknn-toolkit2's, not the CBUF's. Its `0xe010 "REGTASK Overflow"` is a separate limit, the
13-bit operand index (operand indices > 8191), not a CBUF-capacity error. See
[encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md).

Smaller elements pack more K per bank, so quantization changes the tiling. At Mt=Nt=256:

| datatype | element bytes | Kt @ Mt=Nt=256 |
|---|---:|---:|
| fp16 | 2 | 384 |
| int8 | 1 | 768 |
| int4 | 0.5 | 1536 |

The measured rule is `Kt ∝ 1/element-bytes` [HW sweep]. With its denser pack, int4 can
reach `nKt=1` (single-pass K, no readback K-accumulation) on Gemma's `K=3840` by shrinking
the output tile to 64×64. See [encodings/k-accumulation.md](encodings/k-accumulation.md).

The CBUF also has operand-reuse bits. They let a task reuse a tile already resident from
the previous task on the same core, instead of re-fetching it from DRAM. See
[encodings/cbuf-reuse.md](encodings/cbuf-reuse.md).

> [!CAUTION]
> The int8 feature DMA over-reads by one bank (README known issue E4). Software must declare
> `data_bank = ceil(bytes/bank) + 1` for an int8 feature. The fp16 feature is not affected. See
> [encodings/cbuf-bank-slack.md](encodings/cbuf-bank-slack.md).

> [!NOTE]
> The SoC also has an on-chip SRAM (~956 KB). The vendor `rknpu` driver enables it behind
> `CONFIG_ROCKCHIP_RKNPU_SRAM` and debugfs. Mainline `rocket` does not expose it. See
> [perf/sram-nbuf.md](perf/sram-nbuf.md).

## Datatypes

The NPU supports six datatypes: int4, int8, int16, fp16, bf16 and tf32. The 3-bit precision
field selects the precision per stage (input, processing, output). Every datatype has a
working, hardware-validated matmul. The int16 matmul writes int32 that saturates rather than
wraps.

The full capability table is in [datatypes.md](datatypes.md): precision values, output
types, MAC rates, and the use of each datatype. The encoding detail is in
[encodings/precision-field.md](encodings/precision-field.md).

## Throughput

The theoretical 3-core MAC peak scales with the datatype: ~3 TFLOPS fp16, ~6 TOPS int8 (2x),
~12 TOPS int4 (2x again). Measured RKNN matmul benchmarks (clehaxze, external) reach only
~0.5-1 TFLOPS fp16.

On the `rocket` path, the matmul measures ~460 GOP/s at 600 MHz at fp16, int8 and int4 alike
[HW sweep]. DMA and dispatch bound it, rather than the MAC array
([perf/not-mac-bound.md](perf/not-mac-bound.md)).

## Clock

The NPU compute clock (`scmi_clk_npu`) boots at 200 MHz, 1/5 of the 1 GHz maximum, and stays
there. Mainline `rocket` has no NPU devfreq, and 200 MHz is the vendor's `POWER_DOWN_FREQ`.
Raising the clock gives ~1.43x.

> [!CAUTION]
> The NPU power domain is off at cold boot. Setting the PLL while the domain is off hangs the
> SCMI firmware (README known issue E10). See [perf/clock.md](perf/clock.md).

## Kernel interface

`rocket` is a generic register-command submitter (uAPI in
`include/uapi/drm/rocket_accel.h`). Its ioctls are `CREATE_BO`, `SUBMIT`, `PREP_BO` and
`FINI_BO`. A `drm_rocket_task` is `{regcmd (DMA addr of the register-command buffer),
regcmd_count}`.

The kernel runs any register program it is handed. Mesa Teflon's CNN op set is a limit of
Teflon, in userspace, not of the kernel. Software emits its own register program, and the
kernel DMAs its BOs and starts the blocks. `SUBMIT` is asynchronous, and `PREP_BO` on the
output BO is the fence barrier.

> [!NOTE]
> `PREP_BO`'s `timeout_ns` is an absolute `CLOCK_MONOTONIC` deadline, not a relative timeout.
> Passing a relative value is an easy mistake at bring-up.
