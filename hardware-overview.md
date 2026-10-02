# RK3588 NPU hardware overview

This note describes what the NPU is, at the level you need to drive it yourself through
`rocket`.

## NVDLA lineage

The RK3588 NPU is derived from NVIDIA's open NVDLA (Deep Learning Accelerator). The
lineage explains almost every quirk in these notes.

### Block pipeline

The pipeline is a fixed sequence of blocks, CNA→CORE→DPU:

- **CNA** (Convolution...): the feature and weight load, plus the MAC array.
- **CORE**: the MAC/accumulator core.
- **DPU**: the post-processing/data-processing unit, which is the NVDLA SDP.

The DPU has the BS/BN/EW affine stages, the eltwise (EW) sub-unit, an NVDLA-style LUT for
nonlinear activations, and its read DMA, DPU-RDMA.

The register bases are [source-confirmed: Mesa `registers.xml`]:

| Block | Register base |
|---|---|
| PC | `0x0xxx` |
| CNA | `0x1xxx` |
| CORE | `0x3xxx` |
| DPU | `0x4xxx` |
| DPU-RDMA | `0x5xxx` |

The DPU lets the NPU do more than MACs. The EW unit runs elementwise add and mul
(residuals, gated activations). The LUT runs sigmoid, tanh, SiLU, GELU, sqrt, rsqrt,
reciprocal and exp on-chip.

The DPU composes with the matmul and a host-side feature-axis reduce (the PPU cannot
reduce channels). That is enough to run RMSNorm, LayerNorm and softmax, and a full
transformer or Whisper encoder block on the NPU. See
[encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md) and
[encodings/whisper-encoder.md](encodings/whisper-encoder.md).

### Index operations

The hardware has no gather, scatter or indexed read. Every block streams contiguous tiles
through fixed DMA, and there is no indexed-fetch datapath. So any index op (a class-target
gather for cross-entropy, an embedding lookup, `Gather`/`Slice`-by-index) is a host index,
like the host `1/s` of softmax (M scalar lookups, correct and free). A gather is neither a
contraction (matmul) nor a pool (PPU), so no on-chip block supplies it.

Anything linear along the contraction axis is a matmul against a ones weight, which can be
triangular, and needs no new regcmd. That covers a full reduce, a weighted reduce, and a
prefix scan (cumsum). See [encodings/feature-reduce.md](encodings/feature-reduce.md).

### Convolution engine

The NPU is a convolution engine. It has no matmul primitive, so you express a matmul as a
1×1 convolution (see [matmul-as-conv.md](matmul-as-conv.md)).

### Accumulators

Accumulators are wide and NVDLA-shaped: int8 accumulates in INT34->INT32, int16 in INT48,
and fp16 in FP44/FP48->FP32. The CACC integer accumulator is 48-bit for int16 and 34-bit
for int8 [source-confirmed]. The source is a note in Mesa `rkt_coefs.c`, cited in
[encodings/output-transpose-int16.md](encodings/output-transpose-int16.md).

The accumulator width is why large-K fp16 does not lose range. It is also why the
"gibberish" people see is an activation-quant problem, not an accumulator one.

### Data layouts

Data lives in DRAM in NVDLA-style cube layouts with an atomic channel block ("C2"). The
host pre-scatters weights into them. A matmul's activation and output can instead stay
row-major by strides ([matmul-as-conv.md](matmul-as-conv.md) §"Layouts"). See
[encodings/tile-layouts.md](encodings/tile-layouts.md).

## Three cores

The NPU has 3 cores (`npu@fdab0000`, `fdac0000`, `fdad0000` in the device tree). They
co-work and run independently, which a per-fd HW sweep confirms (~3x concurrent
throughput) [HW sweep].

Under `rocket` you reach them by **opening 3+ file descriptors**, not by a core-mask in
the submit. The driver creates one `drm_sched` per core but one scheduling entity per fd,
and an entity pins to one core while it has queued work. One fd with many jobs serializes
onto one core. N fds spread across the N cores [HW sweep, source-confirmed]. See
[perf/iova-and-multicore.md](perf/iova-and-multicore.md).

## CBUF

Each core has a CBUF of 12 banks × 32 KB = 384 KB [source-confirmed: Mesa / SHARD]. During
a conv or matmul task the CBUF must hold both the input-feature tile and the weight tile.
That constraint sets the K-tile size. With the input and weight both resident, the banks
bound the contraction depth `Kt` that fits at a given output tile (Mt × Nt).

The SHARD ViT effort corroborates this on-chip working-set pressure [source-confirmed].
Its `0xe010 "REGTASK Overflow"` is a separate limit: a 13-bit operand-index ceiling
(operand indices > 8191), not a CBUF-capacity error. See
[encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md).

The concrete `Kt` at Mt=Nt=256 shows why quantization changes tiling. Smaller elements
pack more K per bank:

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

The bank count you declare for the feature has a quirk. **The int8 feature DMA over-reads
by one bank**, so it needs `data_bank = ceil(bytes/bank) + 1`. fp16 is immune. See
[encodings/cbuf-bank-slack.md](encodings/cbuf-bank-slack.md).

> There is also an on-chip SRAM (~956 KB, proprietary path gates it behind
> `CONFIG_ROCKCHIP_RKNPU_SRAM` + debugfs). Mainline `rocket` does not expose it. See
> [perf/sram-nbuf.md](perf/sram-nbuf.md).

## The precision menu

The NPU supports a full datatype matrix (int4, int8, int16, fp16, bf16, tf32). The 3-bit
precision field selects the precision per stage (input, processing, output). Every
datatype has a working, hardware-validated matmul. The int16 matmul writes int32 that
saturates rather than wraps.

The full capability table is in [datatypes.md](datatypes.md): precision values, output
types, MAC rates, and what each datatype is for. The encoding detail is in
[encodings/precision-field.md](encodings/precision-field.md).

## Rated and measured throughput

Rockchip markets the NPU at 6 TOPS, the int8-convolution peak. The theoretical 3-core
MAC peak scales with the datatype: ~3 TFLOPS fp16, ~6 TOPS int8 (2x), ~12 TOPS int4 (2x
again). Measured RKNN matmul benchmarks (clehaxze, external) reach only ~0.5-1 TFLOPS
fp16.

On the FOSS `rocket` path the matmul runs far below the MAC peak. It measures ~460 GOP/s
at 600 MHz across precisions (fp16 ~ int8 ~ int4). That is about 15% of the theoretical
fp16 MAC peak and ~4% of the int4 peak. **At this operating point the matmul is
DMA/dispatch-bound, not MAC-bound**, so the 2x and 4x quant MAC advantages do not express
as speed. This is the single most important performance fact.
[perf/not-mac-bound.md](perf/not-mac-bound.md) owns it and states its scope.

## Boot clock

**The NPU compute clock (`scmi_clk_npu`) boots pinned at 200 MHz** (1/5 of the 1 GHz
max). The cause is that mainline `rocket` has no NPU devfreq, and 200 MHz is the vendor's
`POWER_DOWN_FREQ`. Raising the clock is worth ~1.43x.

**Raising it the wrong way is dangerous.** The power domain is off cold, and setting the
PLL cold wedges the SCMI firmware. See [perf/clock.md](perf/clock.md).

## Kernel interface

`rocket` is a generic register-command submitter (uAPI in
`include/uapi/drm/rocket_accel.h`). Its ioctls are `CREATE_BO`, `SUBMIT`, `PREP_BO` and
`FINI_BO`. A `drm_rocket_task` is `{regcmd (DMA addr of the register-command buffer),
regcmd_count}`.

The kernel is not locked to Mesa-Teflon's CNN op set, which is a userspace limitation.
You emit your own register program, and the kernel DMAs your BOs and fires the blocks.
`SUBMIT` is async, and `PREP_BO` on the output BO is the fence barrier.

**`PREP_BO`'s `timeout_ns` is an absolute `CLOCK_MONOTONIC` deadline, not a relative
timeout.** Passing a relative value is an easy bring-up bug.
