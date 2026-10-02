# The per-process readout

The per-process readout is an instrument, not a finding. It records where each timed
benchmark arm's work ran and where its pages landed. This file states what a flat column
from it does and does not mean.

The operating point for every number in this file is an RK3588 (Turing RK1, 31 GiB, kernel
7.2.0-1-arm64, `CONFIG_TRANSPARENT_HUGEPAGE` not set) with the NPU at 600 MHz. The CPU
governor is `performance` on all three policies, and the board is audited idle. Each arm
follows a `drop_caches` + `compact_memory` reset.

## Purpose

The same configuration reads 8-13% apart between consecutive processes, on an idle pinned
board with a memory reset before every arm. The shape is a tight floor with fast excursions
rather than noise around a mean. The phenomenon and the pass-count budget it imposes are in
[data/tuning-matrix.md](data/tuning-matrix.md) §"The predictors do not predict the spread".

That analysis closes the board-level covariates. Over 177 joined rows, `MemAvailable`
varies under 0.9% and the buddyinfo high-order tail under 1.2%, while the wall swings 8-13%
[HW sweep 2026-08-31]. A snapshot of the board cannot see what the allocator and the scheduler
hand one process. The readout reads the process instead, during its own timed run.

The memory reset in the operating point is a measured requirement, not hygiene. One unchanged
configuration read 109.59 t/s, then 118.23 and 118.16 after `drop_caches` and `compact_memory`,
then 116.78 with no further reset. The low level had held across ~20 processes and 40
minutes. Within-run spreads were 0.01-0.9% [HW sweep, RK1, Qwen3.5-0.8B `Q4_K`,
`ROCKET_QUANT_RESIDENT=auto`, pp2048, 600 MHz, governor `performance`, 2026-08-28]. The state
could not be induced again on demand, so its trigger is not known. The two halves of the
reset were not separated.

## Columns recorded per arm

`bench-llm.sh` emits one `<!--RO pass arm k=v ...-->` line per timed arm, beside the
`<!--DATA-->` t/s row it belongs to. The line draws on three independent sources:

| Source | Columns | Reads |
|---|---|---|
| `perf stat -a`, both cluster PMUs, across exactly the timed region | `a76_*`, `a55_*`, `a55_inst_share`, `a76_ipc`, `l2ref_pki`, `l3ref_pki`, `l1dref_pki`, `memacc_pki`, `dtlbw_pki` | which cluster retired the work, and the A76's cache and TLB traffic per thousand instructions |
| two `/proc/stat` reads | `busy` (per-CPU jiffy delta), `busy_little_share` | the same placement question, independently and for free |
| one bounded `/proc/PID/pagemap` sample | `l2color_cv`, `l3color_cv`, `contig_frac`, `mean_run`, `vapa16`, `gib_regions`, and the same per mapping class | where the process's pages landed physically |

The memory events are normalized per thousand A76 instructions, not per second. An arm that
simply ran more work would otherwise show more refills and read as worse placement.

Column meanings and the reasoning behind the sampling are in the probes themselves.
[data/ro-pagemap.py](data/ro-pagemap.py) carries the placement half. The block above
`ro_line()` in [data/bench-llm.sh](data/bench-llm.sh) carries the rest, and
[data/ro-join.py](data/ro-join.py) joins the annotation lines back to the t/s rows.

### Silent failures

Two failures are silent, and each produces a full and plausible line. Read `pfn_zero_frac`
and `pmu_enabled` before quoting anything else.

A `pfn_zero_frac` of 0 means the sampler had the privilege to see real page frame numbers.
Without that privilege every PFN reads 0, and every placement statistic is computed over
zeros.

A `pmu_enabled` of 100 means every event had its own counter. The A76 list is sized to that
part's six programmable counters exactly, so one added event makes perf multiplex and
silently scale every count.

## Cost

The PMU half programs counters and then sleeps for the whole run. The placement half is a
single burst of `/proc` reads once the process has warmed in. It costs 21-58 ms against
arms of 40 to 500 s. The cost is measured and reported per sample as `ro_cost_ms`.

Contiguity is a run-length property, so the sampler reads pages as evenly spaced windows of
512 consecutive pages rather than strided. Run lengths stay exact inside a window, and the
window count bounds the total read however large the process is. That bound keeps the
sampler off `mmap_lock` while the workload is faulting.

`RO=0` disables the whole readout, and `RO_PMU=0` and `RO_PM=0` disable its halves.

## Controls and limits

Both halves move on a positive control, and the controls are the reason a flat column can be
read at all. On a board the harness has just reset, physical placement is essentially perfect
and the cache colors essentially uniform. Those columns are therefore near-constant by
construction, and a flat column there is a statement about the board rather than a working
instrument.

### Placement control

The control samples the same 4 GiB allocation twice, once on a compacted board and once on
a shattered one [HW sweep 2026-08-31, RK1, `ro-selftest-p1.sh`]. The shatter holds every
other 4 KiB page of a 16 GiB private mapping, leaving nothing above order 0 to coalesce:

| cell | order-0 free pages | `contig_frac` | `mean_run` | `l2color_cv` |
|---|---:|---:|---:|---:|
| compacted | 1 959 | 0.9925 | 106.0 | 0.0020 |
| shattered | 1 047 514 | 0.0000 | 1.0 | 0.1665 |

That is a 106x separation on `mean_run` and 83x on `l2color_cv`. The arming check confirms the
control fired before the sample was taken: order-0 free pages rose from 4 424 to 2 008 892.

### Cluster placement control

The control forces `llama-bench` onto each cluster with `taskset`
[HW sweep 2026-08-31, RK1, `-p 512`, Qwen3.5-0.8B-Q4_K]:

| cell | `busy_little_share` | `a55_inst_share` | pp512 t/s |
|---|---:|---:|---:|
| pinned to the A76s (`4-7`) | 0.0194 | 0.0036 | 123.08 |
| unpinned | 0.4022 | 0.1506 | 116.97 |
| pinned to the A55s (`0-3`) | 0.8621 | 0.5713 | 40.86 |

`pmu_cpu_s` is a duration proxy and not a wall. It is the counters' running time summed over
the CPUs the event ran on. A system-wide software event therefore reports about eight times
the elapsed seconds here. As a ratio it tracks the throughput correctly, 0.331 against the
throughput's 0.332 across these two cells. It must not be quoted as an arm's wall. The six
rows taken on 2026-08-31 carry it under the key name `pmu_secs`, with the same meaning.

### Memory-column contrast

The memory columns got their contrast on a real workload rather than a synthetic one. The
contrast came from the 9B unstacked-residency cell, on one model and one graph. One arm
holds 13 GB of fp16 weights resident, and the other re-dequantizes them per micro-batch
[HW sweep 2026-08-31, RK1, three rotated passes per arm]:

| quantity | streaming | resident | ratio |
|---|---:|---:|---:|
| `l2ref_pki` | 1.99 | 4.29 | 2.15x |
| `l3ref_pki` | 3.85 | 7.32 | 1.90x |
| `memacc_pki` | 159.2 | 312.1 | 1.96x |
| A76 instructions retired | 4.13e12 | 1.22e12 | 0.29x |
| A76 L3 refills, absolute | 1.59e10 | 8.89e9 | 0.56x |

Read that table before reading any `_pki` column, because **the normalization inverts the
naive direction**. Residency removes 71% of the host instructions and 44% of the absolute L3
refills, which is the expected win. Per instruction it still reads 1.9-2.2x higher. What
residency removes is the compute-dense NEON dequant loop, and what it leaves is the
memory-bound pack and readback. A `_pki` column is a density, not a total, and a change in
the denominator moves it as readily as a change in the traffic.

### Placement columns between consecutive processes

The placement columns vary between consecutive processes on a quiet board, and the level
does not follow them. A six-pass campaign taken with this instrument shows it
[HW sweep 2026-08-31, RK1, `qwen35-08b-f16`, twelve processes]:

| column | range across the twelve processes | rank correlation against t/s |
|---|---:|---:|
| `contig_frac` | 29.8% | -0.119 |
| `mean_run` | 123% | -0.154 |
| `l3color_cv` | 125% | -0.091 |
| `l2color_cv` | 128% | -0.070 |

The one covariate that tracks the wall is `l3ref_pki`, at a 23.1% range and rho -0.972. So
a null on a placement column here is a real null and not a detection-floor artifact, at
least on this unit. The analysis is in [data/tuning-matrix.md](data/tuning-matrix.md).

### The L3 covariate across an intervention

That L3 correlation is a within-arm one, and it inverts across an intervention. A second
campaign pinned the same unit to the big cluster and read eighteen processes over three arms
[HW sweep 2026-09-01, RK1, `qwen35-08b-f16`, six passes]. Inside each arm the absolute
`a76_l3d_cache_refill` still tracks the wall at rho -0.943, including in the two arms whose
little cluster is idle. Pooled over all eighteen, its rank correlation is +0.172, because
the faster arm is the one taking more refills. The column is a covariate of the residual
per-process lottery. It does not predict the level across a change that moves the level
10-13%.

Read that as a bound on what any single campaign here can conclude. A covariate can hold a
near-perfect rank correlation inside every stratum and carry none of the effect between strata.
Only an intervention separates the two, and the observational rows cannot be made to.

### Limits

Neither half reads IOVA. The mainline `rocket` accel node exposes only `name` under debugfs,
with no BO list, and the NPU reaches its pages through the IOMMU. Physical placement on the
device side of the IOMMU is outside every column here. Where a buffer sits in the device's
address space is not observable from userspace here.

The PMU half is system-wide, which on an audited-idle board is within a fraction of a percent
of the process. It attributes nothing to a specific thread, and it cannot separate the workload
from a tenant. A per-thread claim needs the two placement columns read together. The counters
alone cannot support it.

## Working-set checks with `dtlbw_pki`

A working-set check varies the working set on one model. The column `dtlbw_pki` counts A76
data-TLB walks. A check that it follows a process's resident set fails across two models and
passes on one.

Walks per cycle read 5.49e-04 on gpt-oss-20b against 2.50e-03 on Qwen3.5-0.8B F16, 0.22x,
with the larger resident set walking less. The two models differ in graph and in IPC (1.46
against 1.93). Their absolute walk counts are close, 322M against 417M, while their cycles
differ 5.9x. So a per-cycle rate compares two machines. On one model with only the resident
set varying, `ROCKET_MOE=1`'s 15197 MB against none at `ROCKET_MOE=0`, the same quantity
reads 7.88x [HW sweep, RK1, 2026-08-28].

## Findings from the controls

### `taskset` and the library's own pin

`taskset` confines this stack in one direction only, and the readout shows it. The
A55-pinned cell in §"Cluster placement control" still retired 43% of its instructions on the
A76s. The `librocketnpu` library pins its own matmul pack and readback workers to the big
cluster (`rocket_pin_worker_based`, big set auto-detected by `cpuinfo_max_freq`, overridable
with `ROCKET_CPU_AFFINITY`). A child's own `pthread_setaffinity_np` overrides the mask it
inherited, so pinning the parent at the little cluster measures "everything except the pack
pool moved". The signature is the two placement columns disagreeing: 86% of the time on the
little cluster against 57% of the instructions.

Pinning the parent at the big cluster has no such problem, because the library's own pin
already sits inside that mask. So `taskset 0xf0` is the knob that moves the host application's
threads, and `ROCKET_CPU_AFFINITY` is the one that moves the library's.

### Fragmentation methods that do not fragment

Two plausible ways to fragment memory do not fragment it, and both fail quietly by making the
next allocation more contiguous. Filling the page cache with a large file read leaves the
allocator tidier, because the kernel reclaims clean page cache in whole high-order blocks
(`mean_run` 319 against 122 for the compacted control). Holding every other page of a Python
`mmap.mmap(-1, size)` region does nothing either. That call defaults to `MAP_SHARED`, so with
descriptor -1 it is a shmem object. On shmem, `MADV_DONTNEED` drops only the caller's page
tables, and the pages stay in the object.

The check that catches both is the order-0 free count in `/proc/buddyinfo`. Freeing N single
pages must put about N blocks on the order-0 list, and in both failures that count went down.
