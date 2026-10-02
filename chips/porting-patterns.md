# What transfers between Rockchip NPU revisions

Two chips have been driven to bit-exact compute here: the RK3588 and the RK3576. This sheet is
what the pair establishes about porting to a third:

- Which classes of claim carry over as reliable priors.
- Which are coin flips.
- What the encoding delta looks like when you diff two revisions of the same IP.

The short version: **datapath semantics transfer, geometry encodings do not, and
performance laws invert.** The useful prior is not a transform to apply to an RK3588
register but a rule for deciding which RK3588 facts to trust.

## The base rate

Of the nineteen CNA geometry registers in the RK3588-to-RK3576 delta table
([rk3576.md](rk3576.md)), two coincide by both offset and meaning: `0x1014`
(stride) and `0x1110` (weight address). Everything else moved, re-packed, appeared, or
collided.

So carrying an RK3588 CNA geometry register to a new revision by offset is right about one time
in ten. That is worse than no prior at all, because a wrong geometry register does not fault. It
computes silently wrong, or completes and writes nothing. Both signatures are indistinguishable
from a dozen other causes
([rk3576-regcmd.md](rk3576-regcmd.md), "The wall has two signatures").

The RK3566 is expected to be the opposite case. The RK3568 `rocket` RFC reports the same
NVDLA core and a matching register layout as the RK3588 [source-confirmed]. It therefore
likely needs only a `rocket_hw_profile` and no encoder. That expectation is itself a prior
to test rather than to rely on.

## The encoding delta

Every RK3576 difference from the RK3588 map falls into one of six operations. There is no
seventh. In particular there is no bit-reversal, no endianness flip, no field rotation, and no
constant offset delta, so there is nothing to invert or rotate. The re-pack
is an edit list, not a transform.

| Edit | Example |
|---|---|
| **Move**, packing preserved | `CNA_CVT_CON0` `0x104C` -> `0x1048`, identical `data_sign<<3 \| cvt_type<<1 \| cvt_bypass`. The OUT_CVT triple `0x4080`-`0x4088` -> `0x40AC`-`0x40B4`, still three consecutive. Output address `0x4020` -> `0x4018`. `CNA_PAD_CON1` `0x1184` -> `0x1084`. |
| **Widen a count field**; control bits above it shift up | `PC_TASK_CON.TASK_NUMBER` 12 bits -> 16. The three control bits above it move up by four, and the RK3588 word `0x7001` becomes a task count of 28673. |
| **Compact**: one register per field -> two fields per register | The four CVT scales: RK3588 `CNA_CVT_CON1..4`, one each -> RK3576 `0x104C`/`0x1050`, `scale1<<16 \| scale0`. Burst lengths likewise fold into `0x108C` as `weight_burst<<16 \| data_burst`. |
| **Add**: live on the new part, a gap on the old | `0x1018` precision/ARGB word, `0x101C` total weight bytes, `0x1094`/`0x1098`, `0x118C`. |
| **Collide**: same offset, unrelated meaning | RK3576 `0x1090` is the input line stride; RK3588 `0x1090` is a clock-gating register. The RK3588 writes its line stride at `CNA_DMA_CON1` `0x107C`. |
| **Duplicate**: one quantity at two offsets | CBUF data entries at both `0x103C` hi and `0x1044` lo. |

A value appearing twice is a signal, not a transcription error: the part expects both.

## Matching by value and by function

Match a register by value and by function, never by offset. The identification method is to
hold the semantic function fixed and search for the offset, using constants as the anchor.
Four properties make it work:

- **Constants are the invariants across a re-pack.** The int8 conv's CVT control word is
  `0x0b` on both parts. The burst word is `0x000F000F` on both. When the same magic
  value appears in both streams, that is the same register regardless of where it sits.
- **Consecutive groups stay consecutive.** OUT_CVT is offset/scale/shift in three
  adjacent registers on both parts. If you have located one member, the others are
  adjacent.
- **The encoding conventions are IP-wide, so they are a decoder.** Geometry fields are
  consistently `value - 1`, and pairs are consistently `(hi<<16) | lo`:
  `(in_w-1)<<16 | (in_h-1)`, `(oh_task-1)<<16 | (ow-1)`, `ic-1`, `out_w*out_h - 1`,
  `((kh-1)<<8) | ((kw-1))`. A candidate register whose contents do not decode under
  those conventions is probably not the register you think it is.
- **A capture is an oracle for a whole program, not for a register spliced into a
  different one.** The vendor's float `0x501C` makes the DPU write nothing at all
  against this library's coefficient group. Transcribe programs, not fields.

## IP-inherent priors

These held across both parts and are the things worth assuming on a third:

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

## Where the behavior inverts

**Nearly every performance fact established on the RK3588 is false on the RK3576**, and
several correctness constraints invert outright.

| Axis | RK3588 | RK3576 |
|---|---|---|
| Matmul precision | fp16 wins. Resident int8 prefill is 0.60x fp16 | int8 wins. A resident fp16 GEMM costs 2.0-2.2x the resident int8 one. The convolution-form fp16 program contracts 16 input channels a task, and is 30-299x slower |
| M alignment | `M % 4`. `M == 1` is padded to 4 | No constraint. `M = 1` is bit-exact |
| Matmul output | raw int32 readback | int8 through the DPU requant |
| Integer partials | no on-chip integer K-accumulation ships. The eltwise ALU adds integers exactly under the whole precision bundle, and a K-accumulation on it is not measured | an int32 output writer exists, so a K split carries exact integer partials out |
| M=1 GEMV | ~82x slower than CPU. Decode stays on the host | bit-exact and unconstrained (still submit-bound) |
| Multiple cores | 3 cores, per-fd entities, scheduling shipped and correct | 2 cores. **Two jobs in flight at once compute wrong answers**, 96-100% of calls |
| Completion | maskable completion interrupt | `PC_DONE` read-only in `INTERRUPT_MASK`, so the driver polls it. The DPU's completion interrupt does reach the GIC |
| HW byte counters | reading the `0x2xxx` page hard-locks the SoC | `dt_wr`/`dt_rd`/`wt_rd` are readable |
| Matmul tile cap | `max_tile` 256, `ngroup` 16 | `max_tile` 2048, `ngroup` 32 |
| Where the wall is | DMA/dispatch-bound at this operating point | the host cube scatter was most of every wall, not the submit |

One hazard has no RK3588 counterpart at all. On the RK3576, **a wide output written through a
partial output stage leaves the next submit of any kind writing nothing**. That holds across
processes, until the power domain cycles. The cause is the partial stage and not the width: the
whole output stage writes int32 and fp32 and poisons nothing
([rk3576.md](rk3576.md) §"The poisoning is one hazard, and its cost is a system setting").

### The sorting rule

The inversions cluster on the axes where a machine parameter moved: contraction width, tile
cap, core count, completion routing, output writer width. They are neither random nor a
transform. The invariants cluster on the axes that are pure datapath algebra.

So the usable prior is a question to ask of any RK3588 claim before carrying it:

> Was this derived from a machine parameter, or from the datapath algebra?

The answer sorts every claim:

- **Algebra** is near-certain to transfer, so assume it. That is what the DPU adds, how requant
  composes, how a matmul maps to a convolution, and the field conventions.
- **Machine parameter** (which precision wins, which axis is free, whether a second
  core helps, where the bottleneck sits, any tile or alignment cap): **re-measure.**
  These are the ones that inverted. They inverted because the number they derive from
  changed, not because the silicon disagrees about anything.

Much of the RK3576 performance delta follows from one machine parameter, the contraction one
task holds. An int8 task contracts 4608 input channels. The convolution-form fp16 program
contracts 16, which is why an fp16 convolution needs an `ic/16` submit split. The matmul-form
fp16 program contracts the whole of K in one task, inside a 4096-entry CBUF data window.

That parameter accounts for int8 being the faster matmul precision, and for the K split past one
task's contraction.

Treating a single dominant parameter as the root of a cluster of "laws" is a hypothesis worth
holding rather than a result. It is still the right first thing to look for on a new part.

## Consequences for the codebase

- Machine parameters belong in one `rocket_hw_profile` per chip, read by the planners,
  never as bare literals. That mechanism already exists.
- The geometry-register encoder is per-chip, not an offset table. A single
  address-translation layer cannot express widened fields, compaction, or collisions.
- The refusal belongs at the generator, not at the capability mask. A datatype mask
  says which precisions a chip can run, and both parts run int8 and fp16 perfectly well,
  through different encoders. The check that belongs at the seam is on the encoding.

## Method notes

Each of these was learned on one part and paid off on the other. The notes are:

- **A vendor capture can be manufactured, and it beats a sweep.** Compile an ONNX for
  the target and read the register program for the exact geometry you want. Captures
  you find are confounds: the vendor compiles real models, so real models'
  co-varying axes co-vary in every one.
- **Submit a captured stream verbatim before believing a register-identical emitter.**
  Register-identical is not geometry-identical.
- **Joint sweep, not leave-one-out.** A one-at-a-time sweep cannot see a condition of
  two, and two of the RK3576's modes are exactly that.
- **Sentinel-stamp an output BO.** A fresh BO's zeros cannot distinguish unwritten from
  legitimately zero. Never bare-memset an output BO before a submit. Bracket the fill in
  `PREP_BO` and `FINI_BO`, or the dirty lines race the DPU's DMA.
- **When the question is whether a program damages something else, score a canary, never
  the program under test.** If you score only the program under test, a dead program and a
  poisoning program look identical.
