# IOVA windows and multicore (kernel / DMA)

Two facts about how `rocket` maps memory and dispatches work that you must design
around.

## IOVA is per-fd, 4 GB each

The regcmd encodes BO addresses as **32-bit fields** (e.g. `weights_dma & 0xFFFFFFFF`),
so every BO referenced by a job must live in the low **4 GB** of its address space.
On **RK3588 this 4 GB is the regcmd-field encoding limit, not the bus** — the RK3588 NPU
AXI/IOMMU is **40-bit** (per the upstream rocket RK3568 RFC, which contrasts RK3568's
"32-bit NPU AXI/IOMMU (vs 40-bit)" — [LKML](https://lkml.iu.edu/2605.3/10672.html)). So
the bus could physically reach >4 GB; the 32-bit *address field* in the command list is
what caps a single context at 4 GB. (On **RK3568** it is instead a true bus limit: the
NPU AXI **and** the IOMMU page-walker/DTE are 32-bit, so its page tables need `GFP_DMA32`.)
But — and this is the useful part — **each fd has its own independent 0…4 GB IOVA
window.** [HW sweep + upstream-source]

A probe (`iova_ceiling_rocket.c`) opening 2 fds saw both independently climb
`va = 0x0 → 0x100000000` (8 GB total; the VAs *repeat* per fd). So:

- **N worker fds give ~N × 4 GB of usable IOVA.** The 5-worker resident context has
  ~20 GB — enough for ~80–90% of a 22 GB Gemma-4-12B F16 resident, and **all** of any
  smaller/quantized model.
- BO allocation is **lazy**: reserving 8 GB of IOVA left RAM flat (~595 MB) — only the
  data you actually pack commits physical RAM.

This is why a "`ROCKET_CACHE_MB=12000` exceeds 32 bits" crash is **not** a 4 GB
wall — it is per-tensor *scratch bloat* (each resident weight carrying its own
compute scratch ≈ 2× the weight bytes). **Sharing scratch** per (worker-fd, shape) and
keeping only the weight tiles per-tensor lifts a 12B F16 model from ~30%-resident to
~80–90%-resident within the same IOVA.

**The resident weight scatter is M-independent.** A weight's scattered tile positions
are fixed by the N-slice split + the K/N tiling (`Nt`, `Kt`) alone — the **M** dimension
only sets how many input/output tiles stream through, never where a weight byte lands.
So if the tiling is planned at a **canonical M** (`MAX_TILE`) instead of the actual row
count, one resident weight serves *every* M with no re-pack: warmup-M packs once, any
later prefill M (down to a single short-prompt tile) reuses it. This holds for fp16,
int8 (int32 K-accum is exact for any K-tiling) and group-wise int4 (the K-tile is already
pinned to `group`, the host fp32 K-accum order is per-`(m,n,group)`), so all three resident
paths are bit-exact across M [HW sweep — `matmul_{int8,int4}_crossm_rocket`, pack M=512
reused at M=512/256/768/64/8]. The cost of *not* doing this is a full model re-pack when
warmup-M ≠ prefill-M — a multi-minute stall on a 12B model for a single short prompt.
A genuine tiling mismatch (e.g. a `ROCKET_MM_*` override changing the tiling between pack
and compute) is detected and returns `-2` so the caller re-packs rather than miscomputing.

## Multicore: N fds for N cores, not a core-mask

There are 3 NPU cores. `rocket` exposes them through the **scheduler topology**, not a
submit flag:

- The driver creates **one `drm_sched` per core** but **one scheduling *entity* per
  fd**, and a DRM entity pins to one core while it has queued work (the in-order
  guarantee).
- So **one fd with many jobs serializes onto a single core.** A first probe that
  submitted N jobs in one submit on one fd scaled 1.00 / 1.99 / 3.07× — i.e. it did
  *not* spread across cores (that 3.07 was an artifact of job batching, not
  multicore).
- **Driving N threads, each with its own fd/entity**, makes the kernel dispatch across
  all 3 cores: measured **39.5 → 84.3 → 116.1 → 120.9 jobs/s** at 1/2/3/4 threads
  (1.0 / 2.13 / 2.94 / 3.06×). [HW sweep; corroborated by Tomeu's blog]

**"One thread above the core count" edges higher** (T=4 > T=3, T=5 is the knee): the
extra worker fills the idle bubbles left by each worker's serial pack→submit→readback
cycle. This is the same pattern as the common "rknnpool" (N worker contexts
round-robin over 3 cores, queue depth > core count).

(The proprietary RKNPU exposes core selection via a `core_mask` SUBMIT field; mainline
`rocket` has none — you get cores by using multiple entities. That's not a limitation,
just a different interface.)

## How we use both

The matmul library splits **N** (output channels) across worker threads, each with its
own fd, each running the unchanged single-fd matmul on a contiguous column slice and
scattering its dense result into the strided output. Resident weights are fanned
across the worker fds so each fd's slice fits its own 4 GB window. Default worker
count is 5 (the measured knee). See the `rocket-userspace` library's `rocket_matmul_mt.c`,
`rocket_prepacked*.c`.

**The native int8/uint8 DIRECT conv uses the same pattern** (`rocket_conv_pool` /
`rocket_conv2d_int8_mt` in `rocket_conv.c`). A conv's tiling already decomposes it into
independent **OC-group × OH-band × OW-band** tiles, each writing a disjoint region of the
output — so they fan across a pool of N worker fds, each with its own resident
`rocket_conv_ctx` (BO pool) + scratch. It is **bit-identical** to the single-fd
`rocket_conv2d_int8` (same tiles, same single jobs, just dispatched on different cores)
and falls back to serial for single-tile convs. The `tflite-rocket` delegate creates one
pool per partition (`nthreads`-sized). Measured: warm MobileDet `native_int8` 560→458 ms
(1.21×), conv bucket 1.46×.

**Caveat — multicore only helps a *multi-tile* conv.** A conv small enough to fit one
CBUF pass is a single job ⇒ a single tile ⇒ no intra-conv fan-out (it stays on one core).
For pointwise-heavy detectors many small 1×1s are single-tile; the matmul path (which
splits the output columns regardless of CBUF-pass fit) is the way to parallelize those.

## Two multicore models

The RKNN `core_mask` API conflates two different things:

1. **Intra-model split** — one inference fanned across the 3 cores. RKNN does this with
   an intra-op `subcore_task[5]` partition inside one submit (the vendor `core_mask`
   `0_1`/`0_1_2`). Multi-fd entities (the N-split above) give the **same outcome** — it is
   shipped for the prefill matmul (5 workers) and the detection conv pool, and the prefill
   readback floor is already post-3-core. The vendor's in-one-submit interface is not on
   mainline `rocket` and is not needed.
2. **Multi-instance / throughput** — N *independent* inferences run concurrently
   (queue-depth > core count) so each context's serial pack→submit→readback bubble is
   filled by another. go-rknnlite measured EfficientNet-Lite0 pool-of-9 → 7.9→1.65 ms
   (**4.8× throughput**, latency unchanged). The payoff is **Frigate multi-camera** (one context/stream),
   and it is wired: the `tflite-rocket` delegate's `rocket.py` sets
   `ROCKET_CPU_AFFINITY` per detector process (one A76 each, `nthreads=1`), so
   Frigate's one-process-per-detector model spreads N independent contexts across the
   big cluster. Measured end-to-end through the delegate: **P=1→4 = 1.00 / 2.17 / 3.11
   / 3.56×** (`tflite-rocket/tools/pool_throughput.py`). Confirmed from the mainline kernel
   `drivers/accel/rocket/rocket_job.c`:
   per-core `drm_sched` + per-fd `sched_entity` initialised with all core schedulers →
   automatic load-balance of concurrent jobs.

## Multi-core behavior: the small-op wall

A layer smaller than the multi-core task-allocation granularity runs on a single core — the
same wall as our **multi-tile-only** caveat above: small pointwise convs don't fan out, which
is why the detection path needs the throughput pool (model 2), not more cores.

The load-bearing per-core knob is IRQ affinity: set CPU/DDR/NPU to max frequency, pin the app
to a CPU big core, **and bind the three NPU interrupts to that big core**
(`/proc/irq/<npu-irq>/smp_affinity_list`). *App* `taskset` alone is a no-op for fp16 prefill
(whole-process `taskset`); the **IRQ-affinity binding is the load-bearing knob**, and on the
submit-overhead-bound path it is a large win. See §IRQ affinity below.

## IRQ affinity: the default routes the NPU completion IRQ onto a *little* A55 core

**Measured 2026-06-22 (7.1.0-1-arm64 @600 MHz, IOMMU keep-attached, `submit_overhead_rocket
8 64 16`, 5×3000 + 4000-iter confirmation).** On this kernel the NPU GIC IRQs are **69 =
`fdab0000.npu`, 70 = `fdac0000.npu`, 71 = `fdad0000.npu`** (GIC 142/143/144 — *not* the
110/111/112 sometimes quoted elsewhere; they're SoC/kernel-specific, always read
`/proc/interrupts`). RK3588 CPU map: **cpu0–3 = A55 little @1.8 GHz, cpu4–7 = A76 big
@2.4 GHz.**

The default IRQ affinity is the all-CPUs mask `0-7`. GICv3 routes a multi-CPU level IRQ to
the **lowest CPU in the mask = cpu0, an A55 little core** — confirmed by watching
`/proc/interrupts`: 8000 submits incremented IRQ 69/70 entirely in the **CPU0** column. So by
default the completion handler **and** the waiter wakeup run at 1.8 GHz. Result (median
µs/submit, dispatch floor only):

| config | IRQ affinity | app taskset | median µs/submit |
|---|---|---|---|
| default | `0-7` (→ services on cpu0, A55) | none | **51–53** |
| default + app pinned | `0-7` | cpu7 | **51** (no change — IRQ still on A55) |
| IRQ on big core, app floats | cpu6 | none | **33.5** |
| IRQ + app co-located on one big core | cpu6 / cpu7 / cpu4 | same big core | **27–28** |
| IRQ + app on *different* big cores | cpu4 | cpu7 | **31** |

Takeaways:
- **Moving the IRQ off the A55 onto any A76 is the dominant win: 51 → ~31 µs (−40%).** It does
  not matter *which* big core (4, 6, 7 all gave ~27–28 µs co-located).
- **Co-locating the waiter on that same big core captures a further ~4 µs** (31 → 27, −47%
  total vs default), because the completion IRQ then wakes a cache-hot, same-cluster thread with
  no cross-cluster IPI.
- **App `taskset` *alone* does nothing** (`0-7` IRQ + app on cpu7 = 51 µs): with the IRQ
  still serviced on the little core, pinning the app cannot help. The IRQ binding is the
  prerequisite; co-location is the bonus.

**Recommended bindings (apply once, root):**
- *Latency / single-stream* (decode GEMV, single-camera detection): pin all 3 NPU IRQs to one
  A76 and run the app there — `for q in 69 70 71; do echo 7 > /proc/irq/$q/smp_affinity_list;
  done; taskset -c 7 <app>`.
- *Throughput pool* (the multi-instance, multi-fd work): spread the 3 IRQs across 3 A76s
  (`69→5 70→6 71→7`) and pin each worker to its core, so completions don't queue behind one
  handler.

This is a **runtime/system-config** lever (no driver or library change), complementary to the
IOMMU keep-attached patch below — that one removes ~20 µs of *kernel* work per submit; this one
removes the *little-core wakeup latency* per submit. Stacked, the floor on this path is ~27 µs
vs the ~54 µs of stock-rocket-on-default-affinity. Helper: `rocket-userspace/tests/irq_affinity_probe.sh`
(A/B harness) and `rocket-userspace/tools/npu_set_irq_affinity.sh` (apply recommended binding).
Pays on every many-small-submit path; flat on a single big tiled prefill matmul (one submit).

## Busy-polling the completion fence: redundant with IRQ affinity

A blocking wait (`PREP_BO` with a real timeout) sleeps the waiter until the kernel signals
the job fence from its threaded completion IRQ, so each wait costs an IRQ delivery plus a
scheduler round-trip to re-run the waiter. `ROCKET_BUSY_POLL=<µs>` instead spins on a
non-blocking completion probe for up to that budget before falling back to the blocking
wait — keeping the waiter runnable so it returns within one probe of the fence signalling.
The probe is `PREP_BO` with a **zero** deadline (the only completion check the mainline
uAPI exposes), so every poll also `dma_sync`s the output BO; the lever lives in
`rocket_bo_prep()` and the A/B is the second loop in `submit_overhead_rocket`.

**It does not skip the IRQ.** The mainline fence is only ever signalled from the kernel IRQ
handler, and userspace has no MMIO view of the PC `INTERRUPT_RAW_STATUS` register, so a
userspace spin can only remove the *waiter-side* wakeup, not the interrupt. Skipping the
IRQ itself would need a kernel-side poll (a `patches/rocket` change).

**Measured 2026-06-29** (7.1.0-1-arm64 @600 MHz, A76 governors pinned `performance` — the
governor must be pinned or the idle A76 parks between submits and swamps the signal;
`submit_overhead_rocket 64 256 512`, 3000-iter, in-process blocking-vs-busy-poll A/B):
[HW sweep]

| IRQ affinity | app core | blocking median | busy-poll Δ median / mean |
|---|---|---|---|
| default `0-7` (A55) | A76 | 109 µs | **−4.3 % / −4.7 %** |
| → cpu7 (A76) | other A76 | 96 µs | −1.8 % / +0.9 % (wash) |
| → cpu7 (A76) | co-located | 97 µs | +0.3 % / −0.9 % (none) |

Busy-poll's *best-case* `min`/`p10` improves ~4–5 µs at every config (the removable wakeup),
but the **median win only materialises when the completion IRQ is on a little A55 core** —
i.e. busy-poll and the IRQ-affinity binding above attack the *same* waiter-wakeup term.
Move the IRQ to an A76 (the existing, free, no-core-cost knob) and busy-poll adds nothing;
and even on the default A55 the IRQ-affinity binding alone (blocking 103→89 µs mean, −14 %)
beats busy-poll's −4.7 % without burning a core to spin. On the tiny `8 64 16` shape (~20 µs
floor) busy-poll is high-variance and not a reliable win at all — at that floor the blocking
waiter barely sleeps, so there is little wakeup to remove.

So `ROCKET_BUSY_POLL` ships **opt-in, default-off**: a single-stream latency fallback for the
case where the completion IRQ is stuck on its default little-core affinity and you cannot
re-bind it. When you can, prefer `npu_set_irq_affinity.sh latency` — it is cheaper (no spun
core) and captures the same term. Never enable it under the throughput pool (it burns a core
that the pool wants for another stream).

## Per-job IOMMU dispatch cost — measured ~15–20 µs (and how to remove it)

Stock `rocket` calls `iommu_attach_group()` in `rocket_job_run()` on **every** `drm_sched`
job and `iommu_detach_group()` in `rocket_job_handle_irq()` on **every** completion (plus on
reset). Each toggles the rk_iommu stall/force-reset/paging handshake. On RK3568 that handshake
*times out* on the idle NPU MMU (the upstream bug fixed by RFC patch 5/9); on **RK3588 it
completes silently but still costs latency on every submitted job**.

**Measured (2026-06-22, 7.1.0-1-arm64 @600 MHz, clean A/B of stock vs patch-5):** the
attach+detach handshake is **~15–20 µs per `drm_sched` job**. Two independent tests agree:
- `rocket-userspace/tests/submit_overhead_rocket.c` (tiny 1-task job, 2000× same fd): **median
  54→34 µs/submit, min 39→23 µs** with keep-attached — i.e. ~20 µs (~38%) is the IOMMU term.
- `tests/multicore_probe` (64-task jobs, 10 reps): **−17 to −18 µs per job** (J=1: 7.68→7.50 ms
  over 10 jobs; J=3: 23.56→23.04 ms over 30 jobs).
- `matmul_tiled_rocket 512 3840 4096` (one big job): **flat** — the cost is **per submit, not
  per task**, so it amortizes to nothing on a single tiled prefill matmul but dominates streams
  of small jobs.

**The lever:** RFC patch 5 keeps the per-context domain attached across same-fd jobs (track
`attached_domain` in `struct rocket_core`, swap only on a context change, `kref`-held, detach at
teardown / after reset). Shipped as `patches/rocket/083-rocket-drv-iommu-keepattach.patch`; CTest
8/8, dmesg clean. This is the **per-job** companion to the **per-tile** dispatch levers (tile
fusion, no-alloc submit) — it cuts the floor itself rather than the job count. It pays on
the **submit-overhead-bound** paths this file is about: decode GEMV, the small detection
convs/1×1s, multi-fd contention, and the throughput pool (model 2). See
[not-mac-bound.md](not-mac-bound.md) §dispatch floor.

## Batched submit: one HW kick for many tasks

One large dispatch-floor lever is the number of HW kicks per inference. Mainline
`rocket` submits **one task per kick**: `rocket_job_run()` programs a single task's regcmd into
`PC_DATA_ADDR`, sets `PC_TASK_CON` `TASK_NUMBER(1)`, and re-arms the next task only on each
completion IRQ (`next_task_idx++` in the IRQ handler). A matmul tiled into `nMt·nNt·nKt` tiles
therefore pays one submit + one completion IRQ + one waiter wakeup **per tile**.

The BSP `rknpu` driver fires a **whole task-list in one kick**, but **not** by DMA-walking a
kernel-built descriptor table — that model was tested on RK3588 and disproven (every variant
times out with no completion IRQ; see batched-submit-findings).
The real mechanism: the N tasks' regcmds are laid **contiguously** in one buffer and
**self-chain** — each task ends in an `OP_ENABLE` and its trailer carries the **next** task's
**address** (an embedded `PC_BASE_ADDRESS` op) and stream length (its `PC_REGISTER_AMOUNTS`
op). The kernel programs only the first task's `PC_DATA_ADDR`/`PC_DATA_AMOUNT`, sets
`PC_TASK_CON.TASK_NUMBER = N`, and kicks once; the PC executes one `OP_ENABLE` per task and
follows each `PC_BASE_ADDRESS` link to the next, and `TASK_NUMBER` gates a single completion
IRQ that fires after the last task (up to `max_submit_number` = 4095/chunk on RK3588). The
`PC_BASE_ADDRESS` redirect is **load-bearing**: contiguous layout with only the amount op (or
a zeroed trailer) runs task 0 and stops — the PC must be told *where* the next task is, not
just how long it is. Do **not** retire on a `PC_TASK_STATUS` read: at IRQ time it reads
`0x0000f000` (`& 0xfff == 0`), not the completed count; rely on the `TASK_NUMBER`-gated single
IRQ. `PC_TASK_DMA_BASE_ADDR` is **not** a kernel-walked task-list (the BSP sets it to the
regcmd buffer in one example and to `0` in another — it is don't-care for the stream); the
`rknpu_task` array exists only so the BSP *kernel* can read per-task fields CPU-side.
[HW sweep + source-confirmed: our `rocket-userspace/src/rocket_matmul.c`]

The end-to-end size of the gap, from an independent FOSS RE of both stacks: the proprietary path
issues **≈63 IOCTLs / 1 submit per inference** where an open replay path issues **≈634 IOCTLs /
10 submits** — ~10× fewer kernel transitions. [source: an independent FOSS RE of both stacks
(orangepi5plus-npu), see [SOURCES.md](../SOURCES.md)]

**The lever (shipped + measured).** Mainline's multi-task jobs now run in one kick rather than
re-arming per IRQ. It is a coordinated change: a **userspace** regcmd-chaining pass
(`rocket-userspace` `mm_pack_regcmd`/`mm_seal_chain`, `ROCKET_BATCH_SUBMIT=1` — packs regcmds
contiguously and links each trailer's `PC_BASE_ADDRESS`+`PC_REGISTER_AMOUNTS` to the next task)
plus a small kernel change (`patches/rocket/085-rocket-drv-batched-submit.patch`, module param
`rocket_batch_submit=1` — set `TASK_NUMBER=N`, advance `next_task_idx` to the end so the stock
IRQ-handler path retires on one completion). Both halves off by default; they must be enabled
together (kernel-on + userspace-gapped → recoverable timeout). It removes `N−1` of every `N`
per-task IRQ round-trips, attacking the `wait` term directly (the CPU-blocked-on-fence share
that dominates prefill, [not-mac-bound.md](not-mac-bound.md)), and is **dtype-independent**.
Measured on `matmul_tiled_rocket 512 3840 4096` (320 tiles → 5 batches of 64, 600 MHz,
`performance`): `wait` ~62 → ~48 ms, GFLOP/s ~94 → ~96 (best 99); cosine 1.000000 vs the
per-task path. Modest here (the matmul is compute/readback-bound at this operating point);
larger on submit-overhead-bound paths. Wired into the `mm_compute` fp16 path so far; the other
matmul submit sites share the helpers and are the mechanical follow-up. Full mechanism + the
disproven kernel-only model: BATCHED_SUBMIT_FINDINGS.md.

**Scope.** It pays where an op decomposes into many *independent* tasks: the prefill matmul's
`nMt·nNt` output tiles, a layer's independent Q/K/V projections, the multi-tile detection convs.
It does **not** collapse a data-dependent chain into one kick — a transformer prefill is
sequential across layers, and the `ROCKET_KACC` K-tiles ping-pong (each reads the prior partial),
so they still fence in order. The open path's realistic ceiling is therefore below the vendor's
"1 submit per inference," which is measured on feed-forward vision CNNs. It composes with the
per-submit levers above: IRQ affinity cuts the wakeup latency *of* a submit, IOMMU keep-attached
cuts the kernel work *in* a submit, batched submit cuts the *number* of submits. The HW substrate
is the per-block register ping-pong the next section describes (two register banks per block), which
lets the PC stage task `i+1`'s registers while task `i` runs.

## In-core interrupt / block-completion bitmap (phhusson RE)

Distinct from the three *per-core* GIC IRQs above: **within** a core, each pipeline block
raises a completion bit in the NPU interrupt-status register. The full 14-bit bitmap
(phhusson's `rknpu-reverse-engineering` `hello2.c`, corroborated against the BSP `rknpu`
driver; see [SOURCES.md](../SOURCES.md)):

| bits | block |
|---|---|
| 0,1 | CNA feature group 0 / 1 |
| 2,3 | CNA weight group 0 / 1 |
| 4,5 | CNA CSC group 0 / 1 |
| 6,7 | CORE group 0 / 1 |
| 8,9 | DPU group 0 / 1 |
| 10,11 | **PPU** group 0 / 1 |
| 12 | DMA read error |
| 13 | DMA write error |


Two facts worth keeping:
- **The PPU is a real, separately-completing block** with its own IRQ bits (10/11) — not a
  phantom in `npu_hw.h`. Relevant to the PPU-as-de-tile-engine probe.
- **"group 0 / 1" = two register banks per block** (the ping-pong the BSP exposes as
  `RKNPU_JOB_PINGPONG`): a block carries two register sets so the next task's registers can be
  staged in the idle bank while the current runs. That is the HW basis for any
  register-staging / task-persistence attack on the dispatch floor. On our
  rocket path the kernel owns IRQ config, so this bitmap is *informational* — the `int_mask`/
  `int_clear` fields live in the BSP `rknpu_task`, not in the regcmd we submit; the `0x14
  PC_REGISTER_AMOUNTS` length and the `OP_ENABLE` block mask we *do* drive are a separate
  mechanism.

## Context-pool throughput (model 2, measured)

**Measured 2026-06-24 (7.1.0-1-arm64 @600 MHz, `performance` governor), `ctx_pool_throughput`.**
The throughput pool — P independent contexts (each its own fd/entity), each running a full
"inference" of small prepacked fp16 matmuls (the 1×1-conv-as-matmul detection unit; each call is
a real A-pack → submit → readback) — was swept over pool depth P. This is the model-2
throughput path (queue depth > core count), distinct from the model-1 `multicore_threads` probe
which packs once and loops a bare submit (pure-NPU, saturates at 3).

**Spreading the pool's contexts across the A76 cores is load-bearing — and it is a deliberate
caller responsibility, not automatic.** The library's auto-affinity pins worker `idx` →
`big[idx % n_big]`; a one-thread context (the natural pool unit) always has `idx == 0`, so left
to itself **every** independent pool context pins to the *same* big core (cpu4) and their host
pack/readback serialize there. [HW sweep]

| workload | pool affinity | best speedup vs P=1 | where |
|---|---|---|---|
| tiny 64×256×256 (submit/readback-bound) | colliding (default, all on cpu4) | **~2.1×** | flat past P=3 |
| tiny 64×256×256 | spread across A76 (`ROCKET_CPU_AFFINITY=off` + caller pins each ctx) | **~3.9×** | **peak P=4** |
| large 512×1024×1024 (compute-bound) | spread across A76 | **~2.7×** | still climbing at P=6 |

Takeaways:
- **The pool beats the 3-core count on submit-bound work** (3.9× at P=4 > 3): with contexts on
  separate cores, one context's host pack/readback overlaps another's NPU compute — exactly the
  "rknnpool" effect. P=4 (= the 4 A76 cores) is the sweet spot; P≥5 oversubscribes the big
  cluster and wobbles.
- **Compute-bound ops gain less** (~2.7×, approaching the 3-core NPU ceiling): there is little
  host bubble to hide, so the cores are the limit.
- **The ceiling is host-core / submit-path, not the IRQ core.** Applying the model-2 IRQ
  binding (3 NPU IRQs → cpu5/6/7) raised the P=1 *baseline* (the −47% submit-floor win) but left
  the aggregate *plateau* unchanged — confirming the colliding-pool cap was the shared host core,
  not IRQ servicing.

**Delegate recipe (the throughput pool):** one fd/context per pool instance, **P ≈ 4** (= the
A76 count), set `ROCKET_CPU_AFFINITY=off` and pin each instance to a *distinct* big core, and
apply the throughput IRQ binding (`npu_set_irq_affinity.sh throughput`). Probe:
`rocket-userspace/tests/ctx_pool_throughput.c`.

## uAPI contracts (pinned by `uapi_selftest_rocket`)

A runtime conformance gate (`rocket-userspace/tests/uapi_selftest_rocket.c`) verifies the
`drm_rocket_*` behaviors the library depends on, so a kernel that drifts (a new-SoC port, a uAPI
revision) fails there with a named diagnostic instead of mis-waiting deep in the matmul path.
Confirmed on 7.1.0-1-arm64 (driver reports version **0.0.0**):

- **`CREATE_BO` IOVA bump-starts at 0x0.** The per-fd IOVA allocator hands out ascending,
  page-aligned addresses **beginning at 0** — so the **first BO on a fresh fd legitimately has
  `dma_address == 0`** (the next allocations follow at 0x1000, 0x2000, …). `dma_address` is
  therefore **not** a validity sentinel; test `handle`/`ptr` instead. (Confirms the
  `iova_ceiling` probe's `va = 0x0 → …` read above is the real allocator base, not an artifact.)
  All BOs are page-aligned and stay in the low 4 GB (the 32-bit regcmd window). [HW sweep]
- **`PREP_BO.timeout_ns` is an ABSOLUTE `CLOCK_MONOTONIC` deadline**, not a duration (the kernel
  runs it through `drm_timeout_abs_to_jiffies()`). Verified live: a raw `PREP_BO` with a
  deadline 1 ms in the past returns **promptly with `-EBUSY`** on an in-flight job (no hang),
  while the shim's relative→absolute conversion lets a generous relative wait complete the job.
  A kernel that regressed this to a relative duration would silently turn every job wait into an
  immediate `-EBUSY` poll — which is what this canary catches. (The shim takes a *relative*
  timeout and converts; see `rocket_bo_prep`.)
- **`FINI_BO` always succeeds** (syncs caches back for the device); a failure is a real error,
  not a routine return.
