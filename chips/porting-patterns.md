# Porting between Rockchip NPU revisions

An RK3588 fact falls into one of three classes on another [RKNPU](../GLOSSARY.md) revision. The
classes come from the two SoCs driven to bit-exact compute: the RK3588 and the RK3576.

| Class of fact | On another revision | Examples |
|---|---|---|
| Datapath algebra | Applies | How a matmul maps to a 1×1 convolution, the block sequence and bases, the precision fields, the BS/BN/EW/LUT semantics, what the DPU adds, how requant composes, the field conventions |
| Geometry-register encoding | Differs | Register offsets and bit packing, within a block |
| Anything derived from a machine parameter | Re-measure | Which precision is faster, which axis is free, whether a second core helps, where the bottleneck sits, any tile or alignment cap |

## Register offsets across revisions

Of the nineteen CNA geometry registers in the RK3588-to-RK3576 delta table
([rk3576.md](rk3576.md)), two coincide by both offset and meaning: `0x1014`
(stride) and `0x1110` (weight address). Every other register moved, was re-packed, appeared,
or collided.

An RK3588 CNA geometry register applied by offset to a new revision is therefore right about
one time in ten.

> [!CAUTION]
> A wrong geometry register does not fault. The task produces output of the correct size with
> incorrect values, or completes and writes nothing. Both signatures are indistinguishable from
> a dozen other causes ([rk3576-regcmd.md](rk3576-regcmd.md), "The wall has two signatures"),
> so an offset prior is worse than none.

A third revision sits between the two. In dnhkng/open-rknpu's RV1106 emitters, CORE, DPU and
DPU_RDMA are at the RK3588's offsets with the RK3588's fields. So are the CNA geometry words,
the feature and weight bases and the pad value at `0x1184`. A few CNA words follow the RK3576.
`0x103C` and `0x1044` contain the data entries and the input width. The input width minus one
fills both halves of `0x118C`.

The pads at `0x1068` are byte-packed, where the RK3588's are
nibbles [source-confirmed: open-rknpu `register_profile.py` and its emitters, read at
`0e0afcd`]. A CORE or DPU result from the RV1106 therefore ports to the RK3588 more directly
than an RK3576 result does.

The RK3566 is expected to be the opposite case. Per the RK3568 `rocket` RFC, it likely needs a
`rocket_hw_profile`, possibly a few register-value changes, and no regcmd encoder ([rk3566.md](rk3566.md)).

## The encoding delta

Every RK3576 difference from the RK3588 map is one of the six edit operations below. The
re-pack is an edit list, not a transform.

| Edit | Example |
|---|---|
| **Move**, packing preserved | `CNA_CVT_CON0` `0x104C` -> `0x1048`, identical `data_sign<<3 \| cvt_type<<1 \| cvt_bypass`. The OUT_CVT triple `0x4080`-`0x4088` -> `0x40AC`-`0x40B4`, still three consecutive. Output address `0x4020` -> `0x4018`. `CNA_PAD_CON1` `0x1184` -> `0x1084`. |
| **Widen a count field**, with the control bits above it shifted up | `PC_TASK_CON.TASK_NUMBER` 12 bits -> 16. The three control bits above it move up by four, and the RK3588 word `0x7001` becomes a task count of 28673. |
| **Compact**: one register per field -> two fields per register | The four CVT scales: RK3588 `CNA_CVT_CON1..4`, one each -> RK3576 `0x104C`/`0x1050`, `scale1<<16 \| scale0`. Burst lengths likewise fold into `0x108C` as `weight_burst<<16 \| data_burst`. |
| **Add**: live on the new SoC, a gap on the old | `0x1018` precision/ARGB word, `0x101C` total weight bytes, `0x1094`/`0x1098`, `0x118C`. |
| **Collide**: same offset, unrelated meaning | RK3576 `0x1090` is the input line stride. RK3588 `0x1090` is a clock-gating register. The RK3588 writes its line stride at `CNA_DMA_CON1` `0x107C`. |
| **Duplicate**: one quantity at two offsets | CBUF data entries at both `0x103C` hi and `0x1044` lo. |

A single address-translation layer cannot express widened fields, compaction, or collisions,
so each revision needs its own regcmd encoder, not an offset table.

A value that appears twice is a Duplicate, not a transcription error. The SoC expects both,
and software must program both.

## Matching registers by value and by function

Match a register by value and by function, never by offset. The identification method is to
hold the semantic function fixed and search for the offset, using constants as the anchor.
Four properties make it work:

- **Constants are the invariants across a re-pack.** The int8 conv's CVT control word is
  `0x0b` on both SoCs. The burst word is `0x000F000F` on both. When the same magic
  value appears in both streams, it is the same register regardless of where it sits.
- **Consecutive groups stay consecutive.** OUT_CVT is offset/scale/shift in three
  adjacent registers on both SoCs. If you have located one member, the others are
  adjacent.
- **The encoding conventions are common to every RKNPU, so they decode a candidate.** Geometry
  fields are consistently `value - 1`, and pairs are consistently `(hi<<16) | lo`:
  `(in_w-1)<<16 | (in_h-1)`, `(oh_task-1)<<16 | (ow-1)`, `ic-1`, `out_w*out_h - 1`,
  `((kh-1)<<8) | ((kw-1))`. A candidate register whose contents do not decode under
  those conventions is probably not the register you think it is.
- **Transcribe whole programs from a capture, not single fields.** A register spliced from a
  capture into a different program can fail. The vendor's float `0x501C` makes the DPU write
  nothing at all against `librocketnpu`'s coefficient group.

## Facts common to every RKNPU

These hold on both the RK3588 and the RK3576, and are the facts to assume on another revision:

- Matmul is a 1×1 convolution over CNA→CORE→DPU, and the block sequence is the same.
- Block bases within a core: PC at +0x0, CNA at +0x1000, CORE at +0x3000, DPU at
  +0x4000, DPU_RDMA at +0x5000.
- The precision-field encodings and the BS/BN/EW/LUT datapath semantics (the NVDLA SDP
  X1/X2/Y mapping).
- The coefficient algebra. The DPU adds `B*sum(x)`, so a weight zero point is programmed
  negated, and the input zero point folds into the `A` term.
- Requant is `(acc * SCALE) >> SHIFT` with a per-output-channel multiplier and one
  global shift.
- The `value - 1` and `(hi<<16) | lo` field conventions.
- No on-chip layout conversion and no hardware gather. The host packs the cubes.

## Inverted behavior

Nearly every performance fact established on the RK3588 is false on the RK3576, and several
correctness constraints invert outright.

| Axis | RK3588 | RK3576 |
|---|---|---|
| Matmul precision | fp16 is faster. Resident int8 prefill is 0.60x fp16 | int8 is faster. A resident fp16 GEMM costs 2.0-2.2x the resident int8 one. The convolution-form fp16 program contracts 16 input channels a task, and is 30-299x slower |
| M alignment | `M % 4`. `M == 1` is padded to 4 | No constraint. `M = 1` is bit-exact |
| Matmul output | raw int32 readback | int8 through the DPU requant |
| Integer partials | No on-chip integer K-accumulation is implemented. The eltwise ALU adds integers exactly under the whole precision bundle, and a K-accumulation on it is not measured | An int32 output writer exists, so a K split writes exact integer partials out |
| M=1 GEMV | fp16 throughput ~82x below the batched-GEMM peak, through the vendor RKNN API ([decode-gemv.md](../perf/decode-gemv.md)). Decode stays on the host | bit-exact and unconstrained (still submit-bound) |
| Multiple cores | 3 cores, per-fd entities, scheduling implemented and correct | 2 cores. Two jobs in flight at once produce incorrect output on 96-100% of calls |
| Completion | maskable completion interrupt | `PC_DONE` read-only in `INTERRUPT_MASK`, so the driver polls it. The DPU's completion interrupt does reach the GIC |
| HW byte counters | reading the `0x2xxx` page hard-locks the SoC | `dt_wr`/`dt_rd`/`wt_rd` are readable |
| Matmul tile cap | `max_tile` 256, `ngroup` 16 | `max_tile` 2048, `ngroup` 32 |
| Performance limit | DMA/dispatch-bound at this operating point | a per-op call is host-bound: the int8 matmul entry's device `wait` is 13.7-25.6% of a call at M 512-2048 [HW sweep, 2026-08-11]. Four resident ImageNet graphs, each one kick, are 43-66% submit + wait [HW sweep, 2026-08-03] |

> [!CAUTION]
> On the RK3576, a wide output written through a partial output stage leaves the next submit
> of any kind writing nothing. The effect holds across processes, until the power domain
> cycles, and has no RK3588 counterpart. The cause is the partial stage and not the width: the
> whole output stage writes int32 and fp32 and leaves the next submit unaffected
> ([rk3576.md](rk3576.md) §"The poisoning is one hazard, and its cost is a system setting").

### Classifying a fact before porting it

The inversions cluster on the axes where a machine parameter moved: contraction width, tile
cap, core count, completion routing, output writer width. They are neither random nor a
transform. The invariants cluster on the axes that are pure datapath algebra.

Before applying an RK3588 claim to another revision, ask:

> Was this derived from a machine parameter, or from the datapath algebra?

The answer sorts every claim:

- **Algebra** is near-certain to transfer, so assume it.
- **Machine parameter**: re-measure it. These are the facts that inverted between the RK3588
  and the RK3576, because the number they derive from changed.

Much of the RK3576 performance delta follows from one machine parameter, the contraction one
task holds [hypothesis]. The contraction accounts for int8 being the faster matmul precision,
and for the K split past one task's contraction. An int8 task contracts 4608 input channels.
The convolution-form fp16 program contracts 16, which is why an fp16 convolution needs an
`ic/16` submit split. The matmul-form fp16 program contracts the whole of K in one task, inside
a 4096-entry CBUF data window. On a new SoC, look first for one such dominant parameter.

## Software support

`librocketnpu` reads each SoC's machine parameters from one `rocket_hw_profile`, and has one
regcmd encoder per SoC.

## Porting method

These practices apply on both the RK3588 and the RK3576:

- **Manufacture a vendor capture in preference to a sweep.** Compile an ONNX for the target and
  read the register program for the exact geometry you want. Captures you find are confounds:
  the vendor compiles real models, so real models' co-varying axes co-vary in every one.
- **Submit a captured stream verbatim before believing a register-identical emitter.** A
  register-identical program can still differ in geometry.
- **Sweep registers jointly, not one at a time.** A one-at-a-time sweep cannot see a condition
  of two, and two of the RK3576's modes are exactly that.
- **Sentinel-stamp an output BO.** A fresh BO's zeros cannot distinguish unwritten from
  legitimately zero.
- **Score a canary to test whether a program damages something else.** Do not score the
  program under test. Scored alone, a program that writes nothing and a program that corrupts
  the next submit look identical.

> [!CAUTION]
> An unbracketed CPU fill of an output BO before a submit leaves dirty cache lines that race
> the DPU's DMA. Software must bracket the fill in `PREP_BO` and `FINI_BO`.
