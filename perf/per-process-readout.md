# The per-process readout

An instrument, not a finding. It records **where each timed benchmark arm's work ran and where
its pages landed**. This file states what a flat column from it does and does not mean.

Operating point for every number below: RK3588 (Turing RK1, 31 GiB, kernel 7.2.0-1-arm64,
`CONFIG_TRANSPARENT_HUGEPAGE` not set), NPU at 600 MHz. CPU governor `performance` on all
three policies, board audited idle, `drop_caches` + `compact_memory` before each arm.

## What it is for

The same configuration reads **8-13% apart between consecutive processes**, on an idle pinned
board with a memory reset before every arm. The shape is a tight floor with fast excursions
rather than noise around a mean. The phenomenon and the pass-count budget it imposes are in
[data/tuning-matrix.md](data/tuning-matrix.md) §"The predictors do not predict the spread".

What that analysis closed is the *board-level* covariates. Over 177 joined rows, `MemAvailable`
varies under 0.9% and the buddyinfo high-order tail under 1.2%, while the wall swings 8-13%
[HW sweep 2026-08-31]. A snapshot of the board cannot see what the allocator and the scheduler
hand one process. This reads the process instead, during its own timed run.

## What each arm records

`bench-llm.sh` emits one `<!--RO pass arm k=v ...-->` line per timed arm, beside the
`<!--DATA-->` t/s row it belongs to. Three independent sources:

| Source | Columns | Reads |
|---|---|---|
| `perf stat -a`, both cluster PMUs, across exactly the timed region | `a76_*`, `a55_*`, `a55_inst_share`, `a76_ipc`, `l2ref_pki`, `l3ref_pki`, `l1dref_pki`, `memacc_pki`, `dtlbw_pki` | which cluster retired the work, and the A76's cache and TLB traffic per thousand instructions |
| two `/proc/stat` reads | `busy` (per-CPU jiffy delta), `busy_little_share` | the same placement question, independently and for free |
| one bounded `/proc/PID/pagemap` sample | `l2color_cv`, `l3color_cv`, `contig_frac`, `mean_run`, `vapa16`, `gib_regions`, and the same per mapping class | where the process's pages landed physically |

The memory events are normalised per thousand A76 instructions, not per second. An arm that
simply ran more work would otherwise show more refills and read as worse placement.

Column meanings and the reasoning behind the sampling are in the probes themselves.
[data/ro-pagemap.py](data/ro-pagemap.py) carries the placement half. The block above
`ro_line()` in [data/bench-llm.sh](data/bench-llm.sh) carries the rest, and
[data/ro-join.py](data/ro-join.py) joins the annotation lines back to the t/s rows.

**Two silent failures, both of which produce a full and plausible line.** `pfn_zero_frac` is 0
only when the sampler had the privilege to see real page frame numbers. Without it every PFN
reads 0, and every placement statistic is computed over zeros. `pmu_enabled` is 100 only when
every event had its own counter. The A76 list is sized to that part's six programmable counters
exactly, so one added event makes perf multiplex and silently scale every count. Read both
before quoting anything else.

## Cost

The PMU half programs counters and then sleeps for the whole run. The placement half is a
single burst of `/proc` reads once the process has warmed in. It costs **21-58 ms** against
arms of 40 to 500 s, measured and reported per sample as `ro_cost_ms`.

Contiguity is a run-length property, so pages are sampled as evenly spaced windows of 512
consecutive pages rather than strided. Run lengths stay exact inside a window, and the total
read is bounded by the window count however large the process is. That is what keeps the
sampler off `mmap_lock` while the workload is faulting.

`RO=0` disables the whole readout, `RO_PMU=0` and `RO_PM=0` its halves.

## What it can score, and what it cannot

Both halves move on a positive control, and the controls are the reason a flat column can be
read at all. On a board the harness has just reset, physical placement is essentially perfect
and the cache colors essentially uniform. Those columns are therefore near-constant **by
construction**, and a flat column there is a statement about the board rather than a working
instrument.

**Placement** [HW sweep 2026-08-31, RK1, `ro-selftest-p1.sh`]. The same 4 GiB allocation,
sampled twice. Once on a compacted board, and once on a shattered one. The shatter holds every
other 4 KiB page of a 16 GiB private mapping, leaving nothing above order 0 to coalesce:

| cell | order-0 free pages | `contig_frac` | `mean_run` | `l2color_cv` |
|---|---:|---:|---:|---:|
| compacted | 1 959 | 0.9925 | 106.0 | 0.0020 |
| shattered | 1 047 514 | **0.0000** | **1.0** | **0.1665** |

That is a 106x separation on `mean_run` and 83x on `l2color_cv`. The arming check confirms the
control fired before the sample was taken: order-0 free pages rose from 4 424 to 2 008 892.

**Cluster placement** [HW sweep 2026-08-31, RK1, `-p 512`, Qwen3.5-0.8B-Q4_K]. `llama-bench`
forced onto each cluster with `taskset`:

| cell | `busy_little_share` | `a55_inst_share` | pp512 t/s |
|---|---:|---:|---:|
| pinned to the A76s (`4-7`) | 0.0194 | 0.0036 | 123.08 |
| unpinned | 0.4022 | 0.1506 | 116.97 |
| pinned to the A55s (`0-3`) | 0.8621 | 0.5713 | **40.86** |

`pmu_cpu_s` is a duration proxy and not a wall. It is the counters' running time **summed over
the CPUs the event ran on**. A system-wide software event therefore reports about eight times
the elapsed seconds here. As a ratio it tracks the throughput correctly, 0.331 against the
throughput's 0.332 across these two cells. It must not be quoted as an arm's wall. The six rows
taken on 2026-08-31 carry it under the earlier key name `pmu_secs`, same meaning.

**The memory columns** got their contrast on a real workload rather than a synthetic one. It
came from the 9B unstacked-residency cell, on one model and one graph. One arm holds 13 GB of
fp16 weights resident, the other re-dequantises them per micro-batch
[HW sweep 2026-08-31, RK1, three rotated passes per arm]:

| quantity | streaming | resident | ratio |
|---|---:|---:|---:|
| `l2ref_pki` | 1.99 | 4.29 | 2.15x |
| `l3ref_pki` | 3.85 | 7.32 | 1.90x |
| `memacc_pki` | 159.2 | 312.1 | 1.96x |
| A76 instructions retired | 4.13e12 | 1.22e12 | 0.29x |
| A76 L3 refills, absolute | 1.59e10 | 8.89e9 | 0.56x |

**Read that table before reading any `_pki` column, because the normalisation inverts the naive
direction.** Residency removes 71% of the host instructions and 44% of the absolute L3 refills,
which is the expected win. Per instruction it still reads 1.9-2.2x *higher*. What residency
removes is the compute-dense NEON dequant loop, and what it leaves behind is the memory-bound
pack and readback. A `_pki` column is a density, not a total, and a change in the denominator
moves it as readily as a change in the traffic.

**The placement columns do vary between consecutive processes on a quiet board, and the level
does not follow them.** The first six-pass campaign taken with this instrument settles the
caveat that used to stand here [HW sweep 2026-08-31, RK1, `qwen35-08b-f16`, twelve processes].
Across those twelve, `contig_frac` ranged 29.8%, `mean_run` 123%, `l3color_cv` 125% and
`l2color_cv` 128%, while their rank correlations against t/s were -0.119, -0.154, -0.091 and
-0.070. The one covariate that tracks the wall is `l3ref_pki`, at a 23.1% range and rho -0.972.
So a null on a placement column here is a real null and not a detection-floor artifact, at least
on this unit. The analysis is in [data/tuning-matrix.md](data/tuning-matrix.md).

**That L3 correlation is a within-arm one, and it inverts across an intervention.** A second campaign
pinned the same unit to the big cluster and read eighteen processes over three arms
[HW sweep 2026-09-01, RK1, `qwen35-08b-f16`, six passes]. Inside each arm the absolute
`a76_l3d_cache_refill` still tracks the wall at rho -0.943, including in the two arms whose
little cluster is idle. Pooled over all eighteen its rank correlation is **+0.172**, because the
faster arm is the one taking more refills. The column is a covariate of the residual per-process
lottery, and it does not predict the level across a change that moves the level 10-13%.

**Read that as a bound on what any single campaign here can conclude.** A covariate can hold a
near-perfect rank correlation inside every stratum and carry none of the effect between strata.
Only an intervention separates the two, and the observational rows cannot be made to.

**What it cannot score.** Neither half reads IOVA. Mainline `rocket`'s accel node exposes only
`name` under debugfs, with no BO list, and the NPU reaches its pages through the IOMMU. Physical
placement on the device side of that boundary is outside every column here. The PMU half is
system-wide and attributes nothing to a thread. A per-thread claim needs the two placement
columns read together, and cannot be taken off the counters alone.

The PMU half is system-wide, which on an audited-idle board is within a fraction of a percent
of the process. It attributes nothing to a specific thread, and it cannot separate the workload
from a tenant.

Neither half reads IOVA. The mainline `rocket` accel node exposes only `name` under debugfs,
with no BO list, and the NPU reaches its pages through the IOMMU. Where a buffer sits in the
device's address space is not observable from userspace here.

## Two things the controls established on the way

**`taskset` confines this stack in one direction only, and the readout is what shows it.** The
A55-pinned cell above still retired 43% of its instructions on the A76s. `librocketnpu` pins its
own matmul pack and readback workers to the big cluster (`rocket_pin_worker_based`, big set
auto-detected by `cpuinfo_max_freq`, overridable with `ROCKET_CPU_AFFINITY`). A child's own
`pthread_setaffinity_np` overrides the mask it inherited, so pinning the parent at the little
cluster measures "everything except the pack pool moved". The signature is the two placement
columns disagreeing: 86% of the *time* on the little cluster against 57% of the *instructions*.

Pinning the parent at the **big** cluster has no such problem, because the library's own pin
already sits inside that mask. So `taskset 0xf0` is the knob that moves the host application's
threads, and `ROCKET_CPU_AFFINITY` is the one that moves the library's.

**Two plausible ways to fragment memory do not fragment it, and both fail quietly by making the
next allocation MORE contiguous.** Filling the page cache with a large file read leaves the
allocator tidier, because clean page cache is reclaimed in whole high-order blocks (`mean_run`
319 against 122 for the compacted control). Holding every other page of a Python
`mmap.mmap(-1, size)` region does nothing either. That call defaults to `MAP_SHARED`, so with
descriptor -1 it is a shmem object. `MADV_DONTNEED` on shmem drops only the caller's page
tables, and the pages stay in the object.

The check that catches both is the order-0 free count in `/proc/buddyinfo`. Freeing N single
pages must put about N blocks on the order-0 list, and in both failures that count went **down**.
