# The per-process readout

For each timed benchmark run, the per-process readout records which CPU cluster retired the
work and the A76's cache and TLB traffic. It also records where the process's pages were placed
in physical memory. It reads the process during its own timed run. A snapshot of the board
cannot see what the allocator and the scheduler hand one process. The benchmark harness emits
one `<!--RO pass arm k=v ...-->` line per timed configuration, beside the `<!--DATA-->` t/s row
it belongs to.

**Conditions:** an RK3588 (Turing RK1, 31 GiB, kernel 7.2.0-1-arm64,
`CONFIG_TRANSPARENT_HUGEPAGE` not set), NPU clock ~816 MHz, CPU governor `performance` on all
three policies. The board is audited idle, with a `drop_caches` + `compact_memory` reset before
each configuration.

**Not characterized:** what triggers the low-throughput state that the memory reset clears
(§"Variation between processes"), and which half of the reset clears it.

The line draws on three independent sources:

| Source | Columns | Measures |
|---|---|---|
| `perf stat -a`, both cluster PMUs, across exactly the timed region | `a76_*`, `a55_*`, `a55_inst_share`, `a76_ipc`, `l2ref_pki`, `l3ref_pki`, `l1dref_pki`, `memacc_pki`, `dtlbw_pki` | which cluster retired the work, and the A76's cache and TLB traffic per thousand instructions |
| two `/proc/stat` reads | `busy` (per-CPU jiffy delta), `busy_little_share` | the same placement question, independently and for free |
| one bounded `/proc/PID/pagemap` sample | `l2color_cv`, `l3color_cv`, `contig_frac`, `mean_run`, `vapa16`, `gib_regions`, and the same per mapping class | where the process's pages were placed physically |

The memory events are normalized per thousand A76 instructions, not per second. Without the
normalization, a configuration that simply ran more work would show more refills, and would
look like worse placement. Each script that computes a column documents its meaning and the
reasoning behind its sampling (§"Evidence" lists them).

## Variation between processes

The same configuration measures 8-13% apart between consecutive processes, on an idle board
with the governor pinned and a memory reset before every configuration. The variation is a
narrow band with fast excursions, rather than noise around a mean. It is analyzed, with the
number of passes it requires, in [data/tuning-matrix.md](data/tuning-matrix.md) §"The
predictors do not predict the spread".

Board-level covariates do not account for it. Over 177 joined rows, `MemAvailable` varies under
0.9% and the buddyinfo high-order tail under 1.2%, while wall-clock time varies 8-13%
[HW sweep 2026-08-31].

> [!NOTE]
> The memory reset before each configuration is required. One unchanged configuration
> (Qwen3.5-0.8B `Q4_K`, `ROCKET_QUANT_RESIDENT=auto`, pp2048) measured 109.59 t/s, then 118.23
> and 118.16 after `drop_caches` and `compact_memory`, then 116.78 with no further reset
> [HW sweep 2026-08-28]. The low level had held across ~20 processes and 40 minutes.

## Silent failures

Two failures are silent, and each produces a full and plausible line. Read `pfn_zero_frac`
and `pmu_enabled` before quoting anything else.

A `pfn_zero_frac` of 0 means the sampler had the privilege to see real page frame numbers.
Without that privilege every PFN reads 0, and every placement statistic is computed over
zeros.

A `pmu_enabled` of 100 means every event had its own counter. The A76 list is sized to the
A76's six programmable counters exactly, so one added event makes perf multiplex and silently
scale every count.

## Cost

The PMU half programs counters and then sleeps for the whole run. The placement half is a
single burst of `/proc` reads once the process has warmed in. It costs 21-58 ms against timed
runs of 40 to 500 s. The cost is measured and reported per sample as `ro_cost_ms`.

Contiguity is a run-length property, so the sampler reads pages as evenly spaced windows of
512 consecutive pages rather than strided. Run lengths stay exact inside a window, and the
window count bounds the total read however large the process is. That bound keeps the
sampler off `mmap_lock` while the workload is faulting.

`RO=0` disables the whole readout, and `RO_PMU=0` and `RO_PM=0` disable its halves.

## Column interpretation

On a board the harness has just reset, physical placement is essentially perfect and the cache
colors essentially uniform. The placement columns are therefore near-constant by construction,
and a flat column there describes the board, and does not show that the instrument works. The
positive controls in §"Evidence" show that both halves respond.

### `pmu_cpu_s`

`pmu_cpu_s` is a duration proxy: the counters' running time summed over the CPUs the event ran
on. A system-wide software event therefore reports about eight times the elapsed seconds here.
As a ratio it tracks the throughput correctly [HW sweep 2026-08-31]. Across the two pinned test
points of the cluster placement control (§"Evidence"), it gives 0.331 against the throughput's
0.332. It must not be quoted as a configuration's wall-clock time. The six rows taken on
2026-08-31 record it under the key name `pmu_secs`, with the same meaning.

### `_pki` columns and the instruction count

The contrast for the memory columns comes from the 9B unstacked-residency test point (weights
resident at the default `-ub`), on one model and one graph. One configuration holds 13 GB of fp16 weights resident, and the other
re-dequantizes them per micro-batch [HW sweep 2026-08-31]:

| quantity | streaming | resident | ratio |
|---|---:|---:|---:|
| `l2ref_pki` | 1.99 | 4.29 | 2.15x |
| `l3ref_pki` | 3.85 | 7.32 | 1.90x |
| `memacc_pki` | 159.2 | 312.1 | 1.96x |
| A76 instructions retired | 4.13e12 | 1.22e12 | 0.29x |
| A76 L3 refills, absolute | 1.59e10 | 8.89e9 | 0.56x |

Residency removes 71% of the host instructions and 44% of the absolute L3 refills. Per
instruction, the resident configuration still measures 1.9-2.2x higher. What residency removes
is the compute-dense NEON dequant loop, and what it leaves is the memory-bound pack and
readback.

> [!NOTE]
> A `_pki` column is a density, not a total. A change in the instruction count moves it as
> readily as a change in the traffic. In the table above, the normalization inverts the
> direction of the change in traffic. Read it before reading any `_pki` column.

### Placement columns between consecutive processes

The placement columns vary between consecutive processes on a quiet board, and the throughput
level does not follow them [HW sweep 2026-08-31].

**Conditions:** `qwen35-08b-f16`, twelve processes over six passes.

| column | range across the twelve processes | rank correlation against t/s |
|---|---:|---:|
| `contig_frac` | 29.8% | -0.119 |
| `mean_run` | 123% | -0.154 |
| `l3color_cv` | 125% | -0.091 |
| `l2color_cv` | 128% | -0.070 |

The one covariate that tracks performance is `l3ref_pki`, at a 23.1% range and rho -0.972. A
null on a placement column here is therefore a real null, and not an artifact of the detection
threshold, at least on this model. The analysis is in
[data/tuning-matrix.md](data/tuning-matrix.md).

### The L3 covariate across an intervention

The L3 correlation holds within a configuration, and inverts across an intervention
[HW sweep 2026-09-01].

**Conditions:** `qwen35-08b-f16` pinned to the big cluster, eighteen processes over three
configurations, six passes.

Inside each configuration, the absolute `a76_l3d_cache_refill` still tracks wall-clock time at
rho -0.943, including in the two configurations whose little cluster is idle. Pooled over all
eighteen, its rank correlation is +0.172, because the faster configuration is the one taking
more refills. The column is a covariate of the residual per-process variation. It does not
predict the throughput level across a change that moves the level 10-13%.

> [!NOTE]
> A covariate can hold a near-perfect rank correlation inside every configuration and account
> for none of the difference between configurations. Only an intervention separates the two,
> and observational rows cannot.

### `dtlbw_pki` and the working set

The column `dtlbw_pki` counts A76 data-TLB walks. It follows a process's resident set on one
model, and does not compare across two models [HW sweep 2026-08-28].

On one model with only the resident set varying, `ROCKET_MOE=1`'s 15197 MB against none at
`ROCKET_MOE=0`, walks per cycle measure 7.88x. Across two models, walks per cycle measure
5.49e-04 on gpt-oss-20b against 2.50e-03 on Qwen3.5-0.8B F16, 0.22x, with the larger resident
set walking less. The two models differ in graph and in IPC (1.46 against 1.93). Their absolute
walk counts are close, 322M against 417M, while their cycles differ 5.9x. A per-cycle rate
across the two therefore compares two different workloads.

## Limits

Neither half reads IOVA. The mainline `rocket` accel node exposes only `name` under debugfs,
with no BO list, and the NPU reaches its pages through the IOMMU. Physical placement on the
device side of the IOMMU is outside every column here. Where a buffer sits in the device's
address space is not observable from userspace.

The PMU half is system-wide, which on an audited-idle board is within a fraction of a percent
of the process. It attributes nothing to a specific thread, and it cannot separate the workload
from a tenant. A per-thread claim needs the two placement columns read together. The counters
alone cannot support it.

## `taskset` and the library's own pin

`taskset` confines a process that uses `librocketnpu` in one direction only. Pinned to the A55s
(§"Cluster placement control" in the Evidence), `llama-bench` still retired 43% of its
instructions on the A76s. `librocketnpu` pins its own matmul pack and readback workers to the big cluster
(`rocket_pin_worker_based`, big set auto-detected by `cpuinfo_max_freq`, overridable with
`ROCKET_CPU_AFFINITY`). A child's own `pthread_setaffinity_np` overrides the mask it inherited.
Pinning the parent to the little cluster therefore moves everything except the pack pool. The
readout shows it as the two placement columns disagreeing: 86% of the time on the little
cluster against 57% of the instructions.

Pinning the parent to the big cluster has no such problem, because the library's own pin
already sits inside that mask. `taskset 0xf0` therefore moves the host application's threads,
and `ROCKET_CPU_AFFINITY` moves the library's.

## Fragmenting memory for a placement control

Two plausible ways to fragment memory do not fragment it, and both fail quietly by making the
next allocation more contiguous:

- Filling the page cache with a large file read leaves the allocator tidier, because the kernel
  reclaims clean page cache in whole high-order blocks (`mean_run` 319 against 122 for that
  run's own compacted test point).
- Holding every other page of a Python `mmap.mmap(-1, size)` region does nothing. That call
  defaults to `MAP_SHARED`, so with descriptor -1 it is a shmem object. On shmem,
  `MADV_DONTNEED` drops only the caller's page tables, and the pages stay in the object.

> [!NOTE]
> Check a fragmentation step with the order-0 free count in `/proc/buddyinfo`. Freeing N single
> pages must put about N blocks on the order-0 list, and in both failures above that count went
> down.

## Evidence

### Columns

[data/ro-pagemap.py](data/ro-pagemap.py) computes the placement half. The block above
`ro_line()` in [data/bench-llm.sh](data/bench-llm.sh) computes the rest and emits the line, and
[data/ro-join.py](data/ro-join.py) joins the annotation lines back to the t/s rows.

### Variation between processes

In the reset measurement, within-run spreads were 0.01-0.9% [HW sweep, RK1, ~816 MHz, governor
`performance`, 2026-08-28]. The low state could not be induced again on demand.

### Placement control

The control samples the same 4 GiB allocation twice, once on a compacted board and once on
a shattered one [HW sweep 2026-08-31, RK1]. The script is `ro-selftest-p1.sh`. The shatter
holds every other 4 KiB page of a 16 GiB private mapping, leaving nothing above order 0 to
coalesce:

| test point | order-0 free pages | `contig_frac` | `mean_run` | `l2color_cv` |
|---|---:|---:|---:|---:|
| compacted | 1 959 | 0.9925 | 106.0 | 0.0020 |
| shattered | 1 047 514 | 0.0000 | 1.0 | 0.1665 |

The separation is 106x on `mean_run` and 83x on `l2color_cv`. The arming check confirms the
control fired before the 4 GiB allocation: order-0 free pages rose from 4 424 to 2 008 892. The
table's count is read at the sample, after that allocation.

### Cluster placement control

The control forces `llama-bench` onto each cluster with `taskset`
[HW sweep 2026-08-31, RK1, `-p 512`, Qwen3.5-0.8B-Q4_K]:

| test point | `busy_little_share` | `a55_inst_share` | pp512 t/s |
|---|---:|---:|---:|
| pinned to the A76s (`4-7`) | 0.0194 | 0.0036 | 123.08 |
| unpinned | 0.4022 | 0.1506 | 116.97 |
| pinned to the A55s (`0-3`) | 0.8621 | 0.5713 | 40.86 |

### `_pki` columns and the instruction count

The memory columns got their contrast on a real workload rather than a synthetic one, with
three rotated passes per configuration.
