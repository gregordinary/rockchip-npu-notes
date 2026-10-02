# regcmd and task model

A job (one `drm_rocket_submit`) carries one or more tasks, each a
`{regcmd IOVA, regcmd_count}` descriptor. The PC engine executes a task's regcmd, a
stream of `NPUOP(op, value, reg)` words, ending in the control trailer:

```
NPUOP(OP_NONE, 0, 0)
NPUOP(OP_REG_PC, 0, PC_REGISTER_AMOUNTS)   # driver patches PC_DATA_AMOUNT from regcmd_count
NPUOP(OP_40, 0, 0)
NPUOP(OP_ENABLE, 0x1D, PC_OPERATION_ENABLE)   # 0x1D = bits 0,2,3,4 -> fires PC+CNA+DPU+DPU_RDMA
```

Each task's regcmd opens with `DPU_S_POINTER = 0xE` and `DPU_RDMA_S_POINTER = 0xE`
(`POINTER_PP_MODE | EXECUTER_PP_EN | POINTER_PP_EN`), which arms the NVDLA-style
ping-pong register groups in those two blocks. The kernel arms the CNA and CORE the same
way before every task, writing `0xE | (0x10000000 * core index)` [source-confirmed, mainline
`rocket_job.c`, `rocket_job_hw_submit()`]. A full int8 matmul task is 126 NPUOP words.

## `PC_OPERATION_ENABLE` as a block bitmap

**`PC_OPERATION_ENABLE` is a per-block participation bitmap, and the TRM says otherwise**
[HW sweep, RK3588, 2026-08-24]. The TRM documents `RKNN_pc_operation_enable` (offset
0x0008) as `31:1 RO reserved` with a single `op_en` at bit 0 ("1'd1: Enable PC module to
fetch register for each task"). Mesa's `registers.xml` models it the same way: one `OP_EN`,
bits 1-31 `RESERVED_0`. Both are wrong about the upper bits on this silicon.

The trailer word selects which blocks run. A convolution's `0x1D` is bits 0,2,3,4
(PC/CNA/DPU/DPU_RDMA) and a pool's `0x60` is bits 5,6 (PPU/PPU_RDMA). The two are disjoint,
and `0x60` does not set bit 0 at all. Under the documented reading a pooling program would
enable nothing and never start. Pools run.

The measurement covers both directions. Substituting the convolution's `0x1D` into the
pooling generator's trailer, with nothing else changed, leaves the output buffer untouched.
Every element reads `got=0.0000`, against a reference the same binary reproduces at
`max_abs=0.00024` with `0x60`. So bit 0 is neither sufficient (a conv word starts no pool)
nor necessary (a pool word carries it clear and runs), which is consistent only with a
bitmap. Treat the "reserved" span as undocumented-but-live, and read a program's trailer as
the list of blocks it fires.

Do not read a readback as a contradiction: the register self-clears, so polling it returns 0
whatever was written. What the bits mean and what a readback shows are different questions.

## Does register state persist across tasks (can later tasks send a delta)?

Register state persists within a register group [HW sweep, RK3588, 2026-09-25]. **As
shipped, the group a task lands in is the one written two tasks back.** A delta task writes
only its pointers, the three buffer addresses and the 4-word trailer. For an int8 matmul
the addresses are `CNA_FEATURE_DATA_ADDR` (`0x1070`), `CNA_DCOMP_ADDR0` (`0x1110`) and
`DPU_DST_BASE_ADD` (`0x4020`), 9 words against the full task's 126.

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

Every field resets to 0. With `0xE` in all four blocks, each task writes and executes the group
the task before it did not.

The measurement is on the RK3588 (`rocket` 1.3.0, 600 MHz), int8 × int8 -> int32 at
64×256×64 and 128×1024×128. Each row is one job of gapped tasks, one kick a task, on cores 0 and 1. Every
task has its own buffers and is scored against a CPU model. `F` is a full task, `D` a delta,
and `S` a scrub: the full program at half the output channels. A task that inherits a scrub's
configuration writes the first half of its columns and leaves the rest unwritten.

| Job | Pointers | The delta task |
|---|---|---|
| `F S D` | as shipped | exact, 7 of 7 |
| `S F D` | as shipped | the scrub's half-width surface, 7 of 7 |
| `F F D D D D` | as shipped | every delta exact, 7 of 7 jobs |
| `S S F D` | as shipped | the scrub's half-width surface, 12 of 12 |
| `F D` | all four `0x30` (ping-pong off, both pointers cleared), with the core's own bit | exact, 7 of 7 |

So the register file is not re-latched per task. A task computes against whatever its group
holds, and a delta works wherever that group holds the right geometry. As shipped, write a
job's first two tasks in full and the rest as deltas. With ping-pong off, a delta follows the
task before it.

The measurement does not cover:

- Other generators (fp16, convolution, the DPU elementwise stage, the PPU)
- Self-chained jobs
- The RK3576
- Core 2, which took no job in these runs

**A delta placed one task after a full task lands in the other register group.** That group
holds whatever its last writer left. After a job that wrote a different geometry there, the
delta computes that geometry. After an identical 2-task job, it computes a correct-looking
answer from that job's second task. So test a delta after a job that leaves its group
holding a different geometry.

An RK3576 mainline-`rocket` bring-up hit the same condition and named it: the geometry lands
where the executer is not (gahingwoo, see [SOURCES.md](../SOURCES.md)). The BSP's
`RKNPU_JOB_PINGPONG` "two reg-banks per block" is the same mechanism (see the
[SOURCES.md](../SOURCES.md) rknpu-RE entry).

**A regcmd must not write the CNA or CORE `S_POINTER`.** Those writes need the kernel's
per-core bit, which a job cannot know. The TRM marks bits 28 and 29 reserved. Leave those two
pointers to the kernel.

Written without the bit, or with another core's, the task never completes. It raises no NPU
interrupt and leaves its output unwritten, and the kernel's 500 ms timeout retires it. That
held for 24 of 24 jobs, each on the core whose bit the write lacked. `0x30` hung only on
core 1, and `0x30 | 0x10000000` only on core 0 [HW sweep, RK3588, 2026-09-25].

A delta's saving is not resolved. At 64×256×64, two deltas in a 4-task job read a median
0.2739 ms against 0.2771 ms, submit to fence (40 interleaved reps). At 128×1024×128 they
read 0.4139 ms against 0.4845 ms (20 reps). Both differences sit inside p10-p90 spreads of
0.24-0.28 ms [HW sweep, RK3588, 2026-09-25]. So fetching 117 more regcmd words a task is not
a measurable cost at these shapes.

## Consequences for regcmd generation

- **A task can carry a delta** where it lands in a register group that its own job wrote
  with the same geometry. Nothing measured says that is faster.
- **A delta must not rely on a previous job.** The register file persists across jobs and
  processes. The scheduler picks a job's core, so the last task that core ran is unknown.
- The cheap regcmd optimization is to cache the full regcmd per
  `(Mtile,Ktile,Ntile,accumulate,out-precision)` and patch its address fields. That costs no
  `gen` call, and every task stays self-contained.
- This is the regcmd-side analog of CBUF persistence. On-chip CBUF data is reused across
  tasks through the explicit `WEIGHT_REUSE`/`DATA_REUSE` bits, which a full regcmd re-asserts.

The probe is `tests/regcmd_delta_probe.c` (the arms above, per-core interrupt attribution,
and the timing). Related: [cbuf-reuse.md](cbuf-reuse.md) and
[iova-and-multicore.md](../perf/iova-and-multicore.md) (one fd = one scheduling entity).

## Contiguous chaining (batched submit)

A job's tasks normally run as N separate hardware kicks. The kernel re-arms
`next_task_idx` on each completion IRQ, so the job has one fence at the end but N kicks
and N IRQs. Batched submit collapses them into one kick, in three steps:

1. Lay the tasks' regcmds contiguously (stride = the even-rounded word count).
2. Rewrite each task's trailer to redirect the PC to the next task. The redirect is an
   embedded `PC_BASE_ADDRESS` op that repurposes the inert `OP_NONE` filler at count-4,
   plus the next segment's `PC_REGISTER_AMOUNTS` length.
3. Set `PC_TASK_CON.TASK_NUMBER = N`, so the PC streams all N and fires a single
   completion IRQ.

The `PC_BASE_ADDRESS` redirect is required: without it the PC runs task 0 and stops
[HW sweep]. `TASK_NUMBER = N` is the true stop, not the trailer chain. Clear the final
task's forward link back to `OP_NONE` (the chain seal). With that link left dangling into
the slot past the chain, the kick still completes correctly [HW sweep 2026-06-30]. The PC
retires N tasks and halts regardless.

The chained layout is datatype-independent. Every `gen_*` ends in the same
`[OP_NONE, PC_REGISTER_AMOUNTS, OP_40, OP_ENABLE]` trailer. The matmul tile op count is a
data-independent 126 words for fp16, int8 and int4 alike (even, so stride == count, no
gap). `tests/chain_layout_rocket.c` checks this off-device.

Chaining computes int8 as it computes fp16. Independent int8 × int8 -> int32 matmul tasks,
self-chained into one kick, come back bit-exact against a CPU model. The chained job raises
one NPU interrupt where the gapped one raises one a task. That held for every task at 4
tasks of 64×256×64 and at 8 of 128×1024×128 [HW sweep, RK1, `rocket` 1.3.0, 600 MHz,
`tests/int8_chain_probe`, 2026-09-25]. It also held for a chain whose first task reads only
zeros, so no task lands on its predecessor's accumulator.

fp16 chains any batch length bit-exactly. int4, whose output stage writes int16, has not
been run chained.

A "first task exact, every later task garbage" result for chained integer batches does not
describe the part. That result was scored against `rocket_matmul_int8()`, whose gapped
multi-task jobs ran under the global `rocket_batch_submit` parameter described below. That
parameter garbles exactly such jobs after their first task. The per-kick accumulator clear
offered as its mechanism was never compared against a neighboring task's answer, and the
probe above finds no carried residual.

### Chained tasks honor an in-kick data dependency (WDMA -> ERDMA)

A chained kick serializes its tasks tightly enough that one task's WDMA output is visible
to a later task's ERDMA read [HW sweep 2026-06-30]. It does more than run independent tiles
back-to-back.

The proof is the fp16 EW K-accumulation chained across ki
(`tests/matmul_kacc_chain_rocket.c`, `ROCKET_KACC_CHAIN`). The whole `[ki][tile]` sequence
(ki-outer) is laid in one chain, with two ping-pong output BOs. Each `ki>0` task EW-adds
the prior ki's partial, which an earlier task in the same kick just wrote. The result is
byte-identical to the per-ki fenced path across nKt=2…43. So the PC's in-order task advance
respects both the read-after-write (each ki reads the prior partial) and the
write-after-read (the ping-pong reuses a buffer two ki later).

The redirect fires after each task's `OP_ENABLE`, and `OP_ENABLE` evidently retires the
task's whole CNA→CORE→DPU->WDMA pipeline before the next task's ERDMA issues. This is a
stronger property than "independent tiles chain": the chain can express a cross-task
dependency. Mixing `accumulate=0` ki=0 and `accumulate=1` ki>0 in one chain is fine. Both
emit the same 126-word op count, so the uniform stride holds.

Two traps apply:

1. **A BO that is both written and read inside the kick** (the ping-pong buffers) must be
   listed in `out_bo_handles` only, never in both the in-list and the out-list. A handle in
   both makes the kernel signal completion having executed nothing (no error, no timeout,
   output stays zero-pages). The intra-kick read is device-internal. The in-list is only
   for read-only inputs from other jobs.
2. **The chained form buys little.** Serializing the dependent ki-tasks forfeits the
   intra-kick pipelining that the per-ki path gets from its independent tiles. So it pays
   only when each ki-block carries enough independent tiles to hide the stalls. See
   [k-accumulation.md](k-accumulation.md) §"ki-fence chaining".

### Consequences for the submit path

- **Lever 1, one ioctl and N gapped tasks** (separate kicks) is safe for all datatypes. It
  saves the per-job host cost: the submit syscall, the fence wakeup, the IOMMU attach and
  detach (see [iova-and-multicore.md](../perf/iova-and-multicore.md)).
- **Lever 2, contiguous chaining** (one kick, one IRQ) covers fp16 and int8, with int4
  unmeasured. It also collapses the per-task IRQ and re-kick.
- The per-job `DRM_ROCKET_JOB_BATCHED` flag picks each job's layout, so chained and gapped
  jobs share one process (`tests/mixed_chain_coexist_rocket.c`). A kernel with only the
  global `rocket_batch_submit` parameter treats every multi-task job as chained. A gapped
  job there streams task 0 into the gap and garbles or times out after its first task. The
  resident int8 and int4 matmuls submit gapped (`rocket_prepacked_int8.c`, `_int4.c`).

The probes are `tests/chain_layout_rocket.c` (the layout, off-device, all datatypes) and
`tests/int8_chain_probe.c` (int8 execution, per task against a CPU model, with the
interrupts each job raised).
