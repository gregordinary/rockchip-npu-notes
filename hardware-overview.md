# Hardware overview — the RK3588 NPU

What the NPU is, at the level you need to drive it yourself through `rocket`.

## Lineage: it is an NVDLA derivative

The RK3588 NPU is derived from NVIDIA's open **NVDLA** (Deep Learning Accelerator).
This is not trivia — it explains almost every quirk in these notes:

- The pipeline is a fixed sequence of blocks: **CNA** (Convolution... feature/weight
  load + the MAC array) → **CORE** (the MAC/accumulator core) → **DPU** (the
  post-processing/data-processing unit, the NVDLA **SDP**: the BS/BN/EW affine stages,
  the eltwise (EW) sub-unit, an NVDLA-style **LUT** for nonlinear activations, and its
  read DMA, **DPU-RDMA**). Register bases: PC `0x0xxx`, CNA `0x1xxx`, CORE `0x3xxx`, DPU
  `0x4xxx`, DPU-RDMA `0x5xxx`. [source-confirmed: Mesa `registers.xml`] The DPU is what lets the NPU do more than MACs:
  the EW unit runs elementwise add/mul (residuals, gated activations) and the LUT runs
  sigmoid/tanh/SiLU/GELU/sqrt/rsqrt/reciprocal/exp on-chip — composed with the matmul
  and a host-side feature-axis reduce (the PPU can't reduce channels), this is enough to
  run RMSNorm/LayerNorm/softmax and a **full transformer/Whisper encoder block on the
  NPU**. See [encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md) and
  [encodings/whisper-encoder.md](encodings/whisper-encoder.md).
- **No hardware gather / scatter / indexed read.** Every block streams contiguous tiles
  through fixed DMA; there is no indexed-fetch datapath. So any index op — a class-target
  gather for cross-entropy, an embedding lookup, `Gather`/`Slice`-by-index — is a **host
  index**, like the host `1/s` of softmax (M scalar lookups, correct and free). A gather is
  neither a contraction (matmul) nor a pool (PPU), so no on-chip block supplies it. Note the
  flip side: anything *linear along the contraction axis* (full reduce, weighted reduce,
  **prefix scan / cumsum**) IS a matmul against a (possibly triangular) ones weight, no new
  regcmd — see [encodings/feature-reduce.md](encodings/feature-reduce.md).
- It is a **convolution** engine. There is no "matmul" primitive — you express
  matmul as a 1×1 convolution (see [matmul-as-conv.md](matmul-as-conv.md)).
- Accumulators are wide and NVDLA-shaped: int8 accumulates in INT34→INT32, int16 in
  INT48, fp16 in FP44/FP48→FP32. The CACC **integer** accumulator is 48-bit (int16) /
  34-bit (int8) (Mesa `rkt_coefs.c` note; cited correctly in
  [encodings/output-transpose-int16.md](encodings/output-transpose-int16.md)).
  [source-confirmed] This is why large-K fp16 does **not** lose range, and why the
  "gibberish" people see is an *activation-quant* problem, not an accumulator one.
- Data lives in DRAM in NVDLA-style **tiled "cube" layouts** with an atomic channel
  block ("C2"); the host must pre-scatter into them (see
  [encodings/tile-layouts.md](encodings/tile-layouts.md)).

## Three cores

There are **3 NPU cores** (`npu@fdab0000`, `fdac0000`, `fdad0000` in the device
tree). They co-work and run independently, which a per-fd HW sweep confirms (~3×
concurrent throughput). [HW sweep]

Under `rocket` you reach them by **opening 3+ file descriptors**, not by a core-mask
in the submit: the driver creates one `drm_sched` per core but one scheduling
*entity* per fd, and an entity pins to one core while it has queued work. One fd with
many jobs serializes onto one core; N fds spread across the N cores. See
[perf/iova-and-multicore.md](perf/iova-and-multicore.md). [HW sweep + source-confirmed]

## CBUF — the on-chip scratchpad

Each core has a **CBUF of 12 banks × 32 KB = 384 KB**. [source-confirmed: Mesa /
SHARD] During a conv/matmul task the CBUF must hold the input-feature tile **and**
the weight tile. This is the constraint that sets the K-tile size: with the input
and weight both resident, the contraction depth `Kt` you can fit at a given output
tile (Mt × Nt) is bounded by the banks. (The SHARD ViT effort [source-confirmed]
corroborates this on-chip working-set pressure; note its `0xe010 "REGTASK Overflow"` is
a *separate* 13-bit operand-index ceiling — operand indices > 8191 — not a CBUF-capacity
error; see [encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md).)

Concrete `Kt` at Mt=Nt=256, which is why quantization changes tiling — smaller
elements pack more K per bank:

| dtype | element bytes | Kt @ Mt=Nt=256 |
|---|---:|---:|
| fp16 | 2 | 384 |
| int8 | 1 | 768 |
| int4 | 0.5 | 1536 |

[HW + measured] `Kt ∝ 1/element-bytes`. int4's denser pack means it can reach
`nKt=1` (single-pass K, no readback K-accumulation) on Gemma's `K=3840` by shrinking
the output tile to 64×64. See [encodings/k-accumulation.md](encodings/k-accumulation.md).

The CBUF also has **operand-reuse bits** that let a tile already resident from the
previous task on the same core be reused instead of re-fetched from DRAM — see
[encodings/cbuf-reuse.md](encodings/cbuf-reuse.md).

The bank *count* you declare for the feature has a quirk: the **int8** feature DMA
over-reads by one bank, so it needs `data_bank = ceil(bytes/bank) + 1` (fp16 is
immune) — see [encodings/cbuf-bank-slack.md](encodings/cbuf-bank-slack.md).

> There is also an on-chip SRAM (~956 KB, proprietary path gates it behind
> `CONFIG_ROCKCHIP_RKNPU_SRAM` + debugfs). Mainline `rocket` does not expose it. See
> [perf/sram-nbuf.md](perf/sram-nbuf.md).

## The precision menu

The NPU supports a full datatype matrix — **int4, int8, int16, fp16, bf16, tf32** — with
the 3-bit precision field selecting per-stage (input / processing / output) precision.
Every datatype has a working, hardware-validated matmul; int16 alone has no native output
and is realized by int8 byte-decomposition. The full capability table — precision values,
output types, MAC rates, and what each datatype is for — is in [datatypes.md](datatypes.md),
with the encoding detail in [encodings/precision-field.md](encodings/precision-field.md).

## TOPS vs reality

Rockchip markets the NPU at **6 TOPS** — that is int4-convolution peak. In matmul terms
the realistic numbers (clehaxze, external RKNN benchmarks) are ~0.5–1 TFLOPS fp16 peak,
~2× for int8, ~2× again for int4 *in MAC throughput*.

**On the FOSS rocket path the matmul does not get anywhere near MAC peak.** It measures
~460 GOP/s at 600 MHz **across precisions** (fp16 ≈ int8 ≈ int4) — about 15% of the
fp16 MAC peak and ~4% of the int4 peak. The matmul is **DMA/dispatch-bound, not
MAC-bound**, so the 2×/4× quant MAC advantages do not express as speed. This is the single
most important performance fact and it has its own doc:
[perf/not-mac-bound.md](perf/not-mac-bound.md).

## The clock boots throttled

The NPU compute clock (`scmi_clk_npu`) boots **pinned at 200 MHz** (1/5 of the
1 GHz max) because there is no NPU devfreq under mainline `rocket` and 200 MHz is the
vendor's `POWER_DOWN_FREQ`. Raising it is worth ~1.43× but is dangerous to do wrong
(the power domain is off cold; setting the PLL cold wedges the SCMI firmware). See
[perf/clock.md](perf/clock.md).

## What the kernel gives you

`rocket` is a generic register-command submitter (uAPI in
`include/uapi/drm/rocket_accel.h`): `CREATE_BO` / `SUBMIT` / `PREP_BO` / `FINI_BO`.
A `drm_rocket_task` is `{regcmd (DMA addr of the register-command buffer),
regcmd_count}`. The kernel is **not** locked to Mesa-Teflon's CNN op set — that is a
userspace limitation. You emit your own register program; the kernel DMAs your BOs
and fires the blocks. `SUBMIT` is async; `PREP_BO` on the output BO is the fence
barrier. (`PREP_BO`'s `timeout_ns` is an **absolute** `CLOCK_MONOTONIC` deadline, not
a relative timeout — an easy bring-up bug.)
