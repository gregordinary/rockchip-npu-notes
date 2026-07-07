# regcmd / task model — and why a delta regcmd doesn't work

A **job** (one `drm_rocket_submit`) carries one or more **tasks**, each a
`{regcmd IOVA, regcmd_count}` descriptor. The PC engine executes a task's regcmd — a
stream of `NPUOP(op, value, reg)` words — ending in the control trailer:

```
NPUOP(OP_NONE, 0, 0)
NPUOP(OP_REG_PC, 0, PC_REGISTER_AMOUNTS)   # driver patches PC_DATA_AMOUNT from regcmd_count
NPUOP(OP_40, 0, 0)
NPUOP(OP_ENABLE, 0x1D, PC_OPERATION_ENABLE)   # 0x1D = bits 0,2,3,4 -> fires PC+CNA+DPU+DPU_RDMA
```

Each task also opens with `DPU_S_POINTER = 0xE` / `DPU_RDMA_S_POINTER = 0xE`
(`POINTER_PP_MODE | EXECUTER_PP_EN | POINTER_PP_EN`) = the NVDLA-style **ping-pong dual
register groups**. A full matmul task is ~126 NPUOP words.

## Does register state persist across tasks (can later tasks send a delta)?

**No — not usably.** HW-probed deterministically (`tests/regcmd_persist_rocket.c`, a 2-task
job: task0 full → outA, task1 a 6–8 op delta that rewrites only the changed
addresses + the enable → outB):

- A delta task (just `S_POINTER` + `DPU_DST_BASE_ADD` [+ `CNA_FEATURE_DATA_ADDR` +
  `CNA_DCOMP_ADDR0` for a new tile] + PC trailer) leaves the output **untouched** — the
  compute does not fire. Every run, both `S_POINTER=0` and `0xE`.
- It "passes" **only when a full job ran immediately before it** (even in a *different
  process*): `minimal` alone → NO; `minimal` right after a `full` → YES, deterministically.

**What this reveals:** the NPU **register file is not reset between jobs or processes** — it
globally retains the last-written values. So a delta regcmd can inherit a *prior full job's*
configuration, but that is **unsafe to exploit**: in a real system the "previous job" on a
given core is unpredictable (3 cores, multiple fds, kernel scheduling with no task→core
affinity guarantee), and within a single job task0's config does **not** reliably reach
task1's compute (the enable fires against per-task/per-group state, not the persisted file).

**The mechanism — ping-pong producer/consumer groups.** Each task opens with `S_POINTER=0xE`,
which arms the NVDLA-style **dual register groups**: a *producer* pointer selects which group
the regcmd writes, and a *consumer* pointer tells the executer which group to read. An
independent RK3576 mainline-`rocket` bring-up (gahingwoo, see [SOURCES.md](../SOURCES.md)) hit
the same wall and named the cause: on a job where the written (producer) group is not the one
the executer reads (consumer), "the geometry lands where the executer *isn't*" → stale/empty
config → zero or garbage output, fixed only by **re-initialising the ping-pong pointers every
job**. That is exactly the delta-task failure: a 6–8 op delta writes an *incomplete* producer
group, the executer reads a *different* group, and the compute fires against stale geometry
(or doesn't fire) → output untouched. A *full* regcmd per task is self-consistent because it
(re)writes the whole group its own `S_POINTER` selects. (Corroborates the BSP
`RKNPU_JOB_PINGPONG` "two reg-banks per block" — [SOURCES.md](../SOURCES.md) rknpu-RE entry.)

## Consequence

- **Each task must carry its full, self-contained regcmd.** Delta/partial regcmd is not a
  usable traffic/latency optimization.
- The safe regcmd optimization is to **cache the full regcmd** per
  `(Mtile,Ktile,Ntile,accumulate,out-precision)` and patch only the address fields — same
  126 words, no `gen` cost, every task still self-contained.
- This is the regcmd-side analog of the CBUF-persistence question: on-chip *CBUF*
  data can be reused across tasks via the explicit `WEIGHT_REUSE`/`DATA_REUSE` bits (which
  *are* re-asserted in each task's full regcmd), but the *register configuration* cannot be
  implicitly inherited.

Probe: `tests/regcmd_persist_rocket.c` (`ROCKET_PERSIST_MODE=full|minimal|tile`,
`ROCKET_PERSIST_SPTR=0|0xE`). Related: [cbuf-reuse.md](cbuf-reuse.md),
[iova-and-multicore.md](../perf/iova-and-multicore.md) (one fd = one scheduling entity).

## Contiguous chaining (batched submit) — fp16 only; the integer datapath can't

A job's tasks normally run as **N separate HW kicks** (the kernel re-arms `next_task_idx`
on each completion IRQ — one fence at the end, but N kicks / N IRQs). **Batched submit**
collapses them into **one kick**: lay the tasks' regcmds contiguously (stride = the even-
rounded word count), rewrite each task's trailer to redirect the PC to the next — an embedded
`PC_BASE_ADDRESS` op repurposing the inert `OP_NONE` filler at count-4, plus the next
segment's `PC_REGISTER_AMOUNTS` length — and set `PC_TASK_CON.TASK_NUMBER = N` so the PC
streams all N and fires a single completion IRQ. The `PC_BASE_ADDRESS` redirect is
load-bearing: without it the PC runs task 0 and stops [HW sweep]. **`TASK_NUMBER = N` is the
true stop**, not the trailer chain: the final task's forward link should be cleared back to
`OP_NONE` (the chain seal), but if it is left dangling into the slot past the chain the kick
still completes correctly — the PC retires N tasks and halts regardless [HW sweep 2026-06-30]. The chained **layout is
dtype-independent** — every `gen_*` ends in the same `[OP_NONE, PC_REGISTER_AMOUNTS, OP_40,
OP_ENABLE]` trailer, and the matmul tile op count is a data-independent **126 words** (even,
so stride == count, no gap) for fp16 / int8 / int4 alike (`tests/chain_layout_rocket.c`,
off-device).

**But chaining works end-to-end for fp16 only — the integer datapath garbles** [HW sweep
2026-06-28]. A chained int8 or int4 batch computes the **first task correctly and every
subsequent task as garbage**, identically whether the second tile comes from the M, N, or K
split; fp16 chains any batch length bit-exactly. The layout is byte-identical across dtypes
(ruled out as the cause), so the divergence is in execution: the integer **int32 accumulator
(CACC) clears per HW kick, not per task** — so when several integer tasks run back-to-back in
one kick, task N+1 accumulates onto task N's residual. This is the same accumulator property
behind the no-cross-op-int32-accumulate ceiling (see
[k-accumulation.md](k-accumulation.md), [sdp-stage-precision.md](sdp-stage-precision.md)):
the CACC has no per-task clear the chained stream can invoke. fp16 does not carry that
accumulator across tiles, so it is immune. Re-enabling integer chaining would require a
per-task CACC-clear op in the chained regcmd (not known to exist).

### Chained tasks honor an in-kick data dependency (WDMA → ERDMA)

A chained kick does not just run independent tiles back-to-back — it **serializes them
tightly enough that one task's WDMA output is visible to a later task's ERDMA read** [HW
sweep 2026-06-30]. Proven with the fp16 EW K-accumulation chained **across** ki: lay the
whole `[ki][tile]` sequence (ki-outer) in one chain, ping-pong two output BOs, and let each
`ki>0` task EW-add the prior ki's partial that an *earlier task in the same kick* just wrote
(`tests/matmul_kacc_chain_rocket.c`, `ROCKET_KACC_CHAIN`). The result is **byte-identical** to
the per-ki fenced path across nKt=2…43 — so both the read-after-write (each ki reads the prior
partial) and the write-after-read (the ping-pong reuses a buffer two ki later) are respected
under the PC's in-order task advance. The redirect fires after each task's `OP_ENABLE`, and
`OP_ENABLE` evidently retires the task's whole CNA→CORE→DPU→WDMA pipeline before the next
task's ERDMA issues. This is a stronger property than "independent tiles chain": the chain can
express a genuine cross-task dependency. (Mixing `accumulate=0` ki=0 and `accumulate=1` ki>0
in one chain is fine — both emit the same 126-word op count, so the uniform stride holds.)

Two traps surfaced building it: (1) a BO that is both written and read inside the kick (the
ping-pong buffers) must be listed in **out_bo_handles only**, never in both in- and out-lists —
a handle in both makes the kernel signal completion having executed **nothing** (no error, no
timeout, output stays zero-pages). The intra-kick read is device-internal; the in-list is only
for read-only inputs from *other* jobs. (2) This buys little: serializing the dependent
ki-tasks forfeits the intra-kick pipelining the per-ki path gets from its *independent* tiles,
so it only pays when enough independent tiles ride each ki-block to hide the stalls — see
[k-accumulation.md](k-accumulation.md) §"ki-fence chaining".

**Consequences for the submit path:**
- **Lever 1 — one ioctl, N gapped tasks** (separate kicks, CACC clears per task): safe for
  **all** dtypes; saves the per-*job* host cost (submit syscall, fence wakeup, the IOMMU
  attach/detach — see [iova-and-multicore.md](../perf/iova-and-multicore.md)).
- **Lever 2 — contiguous chaining** (one kick, one IRQ): **fp16 only**; additionally collapses
  the per-task IRQ / re-kick.
- The kernel `rocket_batch_submit` param is **global**: with it on, *every* >1-task job is
  treated as chained, so a gapped int8 job submitted in the same process mismatches the kernel
  (task 0 streams into the gap → timeout/garbage). fp16-chained and int8/int4 work are
  therefore **mutually exclusive per process** under the global param; the upstream-clean fix
  is a **per-`drm_rocket_job` batched flag** so each job picks its own layout. Until then,
  enable the param only for fp16-only workloads; the resident int8/int4 matmul forces gapped
  regardless (`rocket_prepacked_int8.c` / `_int4.c`).

Probe: `tests/chain_layout_rocket.c` (layout, off-device, all dtypes);
`tests/matmul_int8_prepacked_rocket.c M K N W` under `ROCKET_BATCH_SUBMIT=1` + kernel
`rocket_batch_submit=1` reproduces the integer garble (first tile exact, rest wrong).
