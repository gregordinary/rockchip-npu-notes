# regcmd and task model

A job (one `drm_rocket_submit`) contains one or more tasks, each a
`{regcmd IOVA, regcmd_count}` descriptor. The [PC](../GLOSSARY.md) executes a task's
[regcmd](../GLOSSARY.md), a stream of `NPUOP(op, value, reg)` words, ending in the control
trailer:

```
NPUOP(OP_NONE, 0, 0)
NPUOP(OP_REG_PC, 0, PC_REGISTER_AMOUNTS)   # driver patches PC_DATA_AMOUNT from regcmd_count
NPUOP(OP_40, 0, 0)
NPUOP(OP_ENABLE, 0x1D, PC_OPERATION_ENABLE)   # 0x1D = bits 0,2,3,4 -> fires CNA+CORE+DPU+DPU_RDMA
```

Each task's regcmd opens with `DPU_S_POINTER = 0xE` and `DPU_RDMA_S_POINTER = 0xE`
(`POINTER_PP_MODE | EXECUTER_PP_EN | POINTER_PP_EN`), which enables the NVDLA-style
ping-pong register groups in those two blocks. The kernel sets the CNA and CORE pointers the
same way before every task, writing `0xE | (0x10000000 * core index)` [source-confirmed,
mainline `rocket_job.c`, `rocket_job_hw_submit()`]. A full int8 matmul task is 126 NPUOP words.

Each regcmd word is 64 bits, `(target << 48) | (value << 16) | offset`: a 16-bit target, a
32-bit value and a 16-bit register offset [source-confirmed: Mesa `rkt_regcmd.c`,
`emit_raw()`]. A register write's target is its block's code plus 1 [source-confirmed: Mesa
`registers.xml`, the `target` enum]:

| Block | PC | CNA | CORE | DPU | DPU_RDMA | PPU | PPU_RDMA |
|---|---|---|---|---|---|---|---|
| Target | `0x0101` | `0x0201` | `0x0801` | `0x1001` | `0x2001` | `0x4001` | `0x8001` |
 The trailer's enable word has
target `0x0081`.

## `PC_OPERATION_ENABLE` as a block bitmap

The trailer's last word writes `PC_OPERATION_ENABLE` (offset 0x0008) with target `0x81`. Its
value is a per-block participation bitmap that selects which blocks run [HW sweep, RK3588,
2026-08-24]. The bits are those of the global enable register at `0xF008` [TRM, RK3588 Part 1,
`RKNN_global_operation_enable`] [source-confirmed: Mesa `registers.xml`, `GLOBAL`
`OPERATION_ENABLE`]:

| Bit | Field | Block |
|---:|---|---|
| 0 | `cna_op_en` | CNA |
| 1 | reserved | |
| 2 | `core_op_en` | CORE |
| 3 | `dpu_op_en` | DPU |
| 4 | `dpu_rdma_op_en` | DPU_RDMA |
| 5 | `ppu_op_en` | PPU |
| 6 | `ppu_rdma_op_en` | PPU_RDMA |

A convolution's `0x1D` is bits 0,2,3,4 (CNA/CORE/DPU/DPU_RDMA), a DPU-only program's `0x18` is
bits 3,4 (DPU/DPU_RDMA), and a pool's `0x60` is bits 5,6 (PPU/PPU_RDMA). The convolution and
pool words are disjoint, and `0x60` leaves bit 0 clear. Read a program's trailer as the list of
blocks it fires.

| Source | Published |
|---|---|
| TRM, `RKNN_pc_register_amounts` | A regcmd word with bit 55 set "will set each block's op_en". The example, `64'h0081_0000_007f_0008`, sets it for "CNA, CORE, ..., PPU_RDMA" |
| TRM, `RKNN_global_operation_enable` | The bit layout above |
| TRM, `RKNN_pc_operation_enable` | `31:1 RO reserved`, with a single `op_en` at bit 0 ("1'd1: Enable PC module to fetch register for each task"). This is the PC register the kernel writes over the bus to start a task's fetch [source-confirmed, mainline `rocket_job.c`, `rocket_job_hw_submit()`]. It does not describe the trailer word's value |
| Mesa `registers.xml` | PC `OPERATION_ENABLE`: one `OP_EN`, bits 1-31 `RESERVED_0`. `GLOBAL` `OPERATION_ENABLE`: the bit layout above. Mesa's trailer writes `0x1D` through the PC fields, as `RESERVED_0(14) \| OP_EN(1)` |

> [!NOTE]
> The register self-clears, so polling it returns 0 whatever was written. A readback does not
> show which blocks a trailer enabled.

## Does register state persist across tasks (can later tasks send a delta)?

Register state persists within a register group [HW sweep, RK3588, 2026-09-25]. Each block's
`S_POINTER` selects the group that register writes go to and the group the executer reads. The
register file is not re-latched per task: a task computes with whatever its group holds.

**Conditions:** RK3588, `rocket` 1.3.0, NPU clock ~816 MHz. int8 × int8 -> int32 at 64×256×64 and
128×1024×128. Each job is gapped tasks, one kick a task, on cores 0 and 1.

**Not characterized:** other generators (fp16, convolution, the DPU elementwise stage, the PPU),
self-chained jobs, the RK3576, and core 2.

The pointer fields, one `S_POINTER` per block at `0x1004`, `0x3004`, `0x4004` and `0x6004`
[TRM, RK3588 Part 1, `RKNN_cna_s_pointer` and its siblings]:

| Bit | Field | Meaning |
|---:|---|---|
| 0 | `pointer` | the group the next register writes land in |
| 1 | `pointer_pp_en` | the write pointer toggles after each task |
| 2 | `executer_pp_en` | the executer toggles after each task |
| 3 | `pointer_pp_mode` | 1: the pointer toggles by pointer, 0: by executer |
| 4 | `pointer_pp_clear` | write 1 to clear the write pointer to group 0 |
| 5 | `executer_pp_clear` | write 1 to clear the executer to group 0 |
| 16 | `executer` (RO) | the group the executer reads |

Every field resets to 0. The kernel and the regcmd above program `0xE` in all four blocks. Each
task then writes and executes the group the task before it did not. The group a task executes
is therefore the one written two tasks back.

A delta task writes only its pointers, the three buffer addresses and the 4-word trailer. For an
int8 matmul the addresses are `CNA_FEATURE_DATA_ADDR` (`0x1070`), `CNA_DCOMP_ADDR0` (`0x1110`)
and `DPU_DST_BASE_ADD` (`0x4020`), 9 words against the full task's 126. A delta is correct
wherever its group holds the right geometry:

- With `0xE` pointers, write a job's first two tasks in full and the rest as deltas.
- With ping-pong off, all four `S_POINTER` at `0x30` (both pointers cleared) with the core's
  own bit, a delta follows the task before it.

> [!CAUTION]
> A delta placed one task after a full task executes the other register group, which holds
> whatever its last writer left. After a job that wrote a different geometry there, the delta
> computes that geometry. After an identical 2-task job, it returns a correct-looking answer
> from that job's second task. Test a delta after a job that leaves its group holding a
> different geometry.

The gahingwoo RK3576 mainline-`rocket` bring-up describes the same condition: the geometry is
written where the executer is not ([SOURCES.md](../SOURCES.md)). The BSP's
`RKNPU_JOB_PINGPONG`, "two reg-banks per block", is the same mechanism (the rknpu-RE entry in
[SOURCES.md](../SOURCES.md)).

Software must not write the CNA or CORE `S_POINTER` from a regcmd. Those writes need the
kernel's per-core bit, which a job cannot know, and the TRM marks bits 28 and 29 reserved. The
kernel writes those two pointers.

> [!CAUTION]
> A CNA or CORE `S_POINTER` written without the core's bit, or with another core's, hangs the
> task. It raises no NPU interrupt and leaves its output unwritten, and the kernel's 500 ms
> timeout retires it [HW sweep, RK3588, 2026-09-25].

## Regcmd generation

- A task can be a delta where it executes a register group that its own job wrote with the
  same geometry.
- A delta must not rely on a previous job. The register file persists across jobs and
  processes. The scheduler picks a job's core, so the last task that core ran is unknown. Each
  open fd is one scheduling entity ([iova-and-multicore.md](../perf/iova-and-multicore.md)).
- Caching the full regcmd per `(Mtile,Ktile,Ntile,accumulate,out-precision)` and patching its
  address fields costs no `gen` call, and keeps every task self-contained.
- CBUF persistence is the on-chip analog of register persistence. CBUF data is reused across
  tasks through the explicit `WEIGHT_REUSE`/`DATA_REUSE` bits, which a full regcmd re-asserts
  ([cbuf-reuse.md](cbuf-reuse.md)).

A delta's time saving is not resolved. At 64×256×64 and 128×1024×128, fetching 117 more regcmd
words a task is not a measurable cost [HW sweep, RK3588, 2026-09-25]. The difference sits inside
the run-to-run spread.

## Contiguous chaining (batched submit)

A job's tasks normally run as N separate hardware [kicks](../GLOSSARY.md). The kernel re-arms
`next_task_idx` on each completion IRQ, so the job has one fence at the end but N kicks and N
IRQs. Batched submit collapses them into one kick, in three steps:

1. Lay the tasks' regcmds contiguously (stride = the even-rounded word count).
2. Rewrite each task's trailer to redirect the PC to the next task. The redirect is an
   embedded `PC_BASE_ADDRESS` op that repurposes the inert `OP_NONE` filler at count-4,
   plus the next segment's `PC_REGISTER_AMOUNTS` length.
3. Set `PC_TASK_CON.TASK_NUMBER = N`, so the PC streams all N and fires a single
   completion IRQ.

The `PC_BASE_ADDRESS` redirect is required: without it the PC runs task 0 and stops
[HW sweep]. `TASK_NUMBER = N` ends the chain: the PC retires N tasks and halts, whatever the
trailer chain holds. Clear the final task's forward link back to `OP_NONE` (the chain seal).
With that link left dangling into the slot past the chain, the kick still completes correctly
[HW sweep 2026-06-30].

The chained layout is datatype-independent. Every `gen_*` ends in the same
`[OP_NONE, PC_REGISTER_AMOUNTS, OP_40, OP_ENABLE]` trailer. The matmul tile op count is a
data-independent 126 words for fp16, int8 and int4 alike (even, so stride == count, no gap).

| Datatype | Self-chained in one kick |
|---|---|
| fp16 | Bit-exact at any batch length |
| int8 | Independent int8 × int8 -> int32 matmul tasks are bit-exact against a CPU model [HW sweep, RK1, 2026-09-25] |
| int4 | Not characterized. Its output stage writes int16 |

**Conditions:** int8 row: RK1, `rocket` 1.3.0, NPU clock ~816 MHz, 4 tasks of 64×256×64 and 8 of
128×1024×128.

A chained job raises one NPU interrupt, and a gapped job raises one per task
[HW sweep, RK1, 2026-09-25].

### Chained tasks honor an in-kick data dependency (WDMA -> ERDMA)

In a chained kick, one task's WDMA output is visible to a later task's ERDMA read
[HW sweep 2026-06-30]. The PC's in-order task advance respects both the read-after-write and the
write-after-read between tasks. An fp16 EW K-accumulation chained across ki is byte-identical to
the per-ki fenced path across nKt=2…43. A chain can therefore express a cross-task dependency.

The redirect fires after each task's `OP_ENABLE`. The likely mechanism is that `OP_ENABLE`
retires the task's whole CNA→CORE→DPU->WDMA pipeline before the next task's ERDMA issues
[hypothesis].

A chain can mix `accumulate=0` at ki=0 and `accumulate=1` at ki>0. Both emit the same 126-word
op count, so the uniform stride holds.

> [!CAUTION]
> Software must list a BO that one kick both writes and reads in `out_bo_handles` only. The
> ping-pong buffers are such BOs. With the handle in both the in-list and the out-list, the
> kernel signals completion having executed nothing. It reports no error and no timeout, and the
> output stays zero-pages. The intra-kick read is device-internal. The in-list is only for read-only inputs
> from other jobs.

> [!NOTE]
> A chain of dependent ki-tasks gains little. Serializing them loses the intra-kick pipelining
> that the per-ki path gets from its independent tiles. Only a ki-block with enough independent
> tiles to hide the stalls gives the chain a net gain
> ([k-accumulation.md](k-accumulation.md) §"ki-fence chaining").

### Submit options

| Option | Kicks and IRQs | Correct for | Saves |
|---|---|---|---|
| One ioctl, N gapped tasks | N | All datatypes | The per-job host cost: the submit syscall, the fence wakeup, the IOMMU attach and detach ([iova-and-multicore.md](../perf/iova-and-multicore.md)) |
| Contiguous chaining | One | fp16 and int8. int4 not characterized | The per-job host cost, and also the per-task IRQ and re-kick |

The per-job `DRM_ROCKET_JOB_BATCHED` flag picks each job's layout, so chained and gapped jobs
share one process.

> [!CAUTION]
> A kernel with only the global `rocket_batch_submit` parameter treats every multi-task job as
> chained. A gapped job there streams task 0 into the gap, and garbles or times out after its
> first task.

## Software support

The resident int8 and int4 matmuls in `librocketnpu` submit gapped (`rocket_prepacked_int8.c`,
`_int4.c`).

## Evidence

### `PC_OPERATION_ENABLE` as a block bitmap

Read through the PC register's own field description (one `op_en` at bit 0), a pooling
program's `0x60` would enable nothing and never start, and pools run. The measurement also
covers the other direction. Substituting the convolution's `0x1D` into the pooling generator's
trailer, with nothing else changed, leaves the output buffer untouched. Every element reads
`got=0.0000`, against a reference the same binary reproduces at `max_abs=0.00024` with `0x60`.
Bit 0 is therefore neither sufficient (a conv word starts no pool) nor necessary (a pool word
has it clear and runs), which is consistent only with a bitmap.

The two substitutions test the word as a whole. The block each bit enables comes from the
TRM's `RKNN_global_operation_enable` layout, and Mesa's `GLOBAL` register agrees with it. The
measured words fit that layout. `0x1D` enables the four blocks a convolution uses, `0x18` the
two a DPU-only program uses, and `0x60` the two a pool uses.

### Does register state persist across tasks (can later tasks send a delta)?

The measurement is `tests/regcmd_delta_probe.c`: the configurations below, per-core interrupt
attribution, and the timing. Each row is one job. Every task has its own buffers and is scored
against a CPU model. `F` is a full task, `D` a delta, and `S` a scrub: the full program at half
the output channels. A task that inherits a scrub's configuration writes the first half of its
columns and leaves the rest unwritten.

| Job | Pointers | The delta task |
|---|---|---|
| `F S D` | default, `0xE` | exact, 7 of 7 |
| `S F D` | default, `0xE` | the scrub's half-width output, 7 of 7 |
| `F F D D D D` | default, `0xE` | every delta exact, 7 of 7 jobs |
| `S S F D` | default, `0xE` | the scrub's half-width output, 12 of 12 |
| `F D` | all four `0x30` (ping-pong off, both pointers cleared), with the core's own bit | exact, 7 of 7 |

The `S_POINTER` hang held for 24 of 24 jobs, each on the core whose bit the write lacked. `0x30`
hung only on core 1, and `0x30 | 0x10000000` only on core 0. Core 2 took no job in these runs.

At 64×256×64, two deltas in a 4-task job measure a median 0.2739 ms against 0.2771 ms, submit
to fence (40 interleaved reps). At 128×1024×128 they measure 0.4139 ms against 0.4845 ms (20
reps). Both differences sit inside p10-p90 spreads of 0.24-0.28 ms.

### Contiguous chaining (batched submit)

`tests/chain_layout_rocket.c` checks the layout off-device, for all datatypes.
`tests/int8_chain_probe.c` scores int8 execution per task against a CPU model, and records the
interrupts each job raised. The int8 result also held for a chain whose first task reads only
zeros, so no task computes on its predecessor's accumulator.
`tests/mixed_chain_coexist_rocket.c` runs chained and gapped jobs in one process.

### Chained tasks honor an in-kick data dependency (WDMA -> ERDMA)

The measurement is the fp16 EW K-accumulation chained across ki
(`tests/matmul_kacc_chain_rocket.c`, `ROCKET_KACC_CHAIN`). The whole `[ki][tile]` sequence
(ki-outer) is laid in one chain, with two ping-pong output BOs. Each `ki>0` task EW-adds the
prior ki's partial, which an earlier task in the same kick just wrote. Each ki reads the prior
partial (read-after-write), and the ping-pong reuses a buffer two ki later (write-after-read).

### Excluded causes

A "first task exact, every later task garbage" result for chained integer batches does not
describe the RK3588. That result was scored against `rocket_matmul_int8()`, whose gapped
multi-task jobs ran under the global `rocket_batch_submit` parameter. That parameter garbles
exactly such jobs after their first task. The per-kick accumulator clear offered as its
mechanism was never compared against a neighboring task's answer. The int8 chain measurement
finds no residual from a previous task.
