# Matmul device time and host time

On the vendor `rknpu` driver, each submit returns `hw_elapse_time`, the submit's device time.
It runs from the job's commit, just before the driver writes the register program, to the
completion interrupt [source-confirmed, `rknpu_job.c` in `rknpu` 0.9.8]. It separates a
matmul's device time from its host time, and mainline `rocket` has no equivalent. With the
weight resident, the device takes 67-82% of an fp16 matmul's wall-clock time [HW sweep]. At
1024×3840×4096, packing the weight on every call leaves the device time unchanged, and the
difference in wall-clock time is the per-call weight scatter.

**Conditions:** RK3588, NPU clock 1000 MHz, `rknpu` 0.9.8, fp16 matmul, `ROCKET_KACC=1`, one
worker, no other load on the board, 2026-08-25.

**Not characterized:** runs with several worker fds, and the division of the one-shot path's
fixed per-call overhead into host work and dispatch cost.

"One-shot" is `rocket_matmul_fp16`, which packs the weight every call. "Resident" is
`rocket_matmul_fp16_prepacked`, which scatters the weight once. Times are in ms [HW sweep]:

| M×K×N | submits | one-shot wall-clock | device | device % | resident wall-clock | device | device % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64×256×256 | 1 | 10.26 ms | 0.09 | 0.9% | n/a | n/a | n/a |
| 128×512×512 | 1 | 9.58 ms | 0.60 | 6.2% | n/a | n/a | n/a |
| 256×1024×1024 | 2 | 19.13 ms | 2.58 | 13.5% | 4.72 ms | 3.85 | 81.6% |
| 512×1024×1024 | 2 | 25.53 ms | 4.89 | 19.2% | 7.16 ms | 4.81 | 67.2% |
| 512×2048×2048 | 4 | 37.44 ms | 8.20 | 21.9% | 8.24 ms | 5.74 | 69.7% |
| 512×3840×4096 | 8 | 106.99 ms | 31.70 | 29.6% | 27.76 ms | 21.45 | 77.3% |
| 1024×3840×4096 | 16 | 127.19 ms | 42.90 | 33.7% | 55.42 ms | 42.91 | 77.4% |

## Cause

### Per-call weight scatter

The host share is a property of the one-shot path. At 1024×3840×4096 the device time is the
same to within 0.01 ms on both paths (42.90 against 42.91 ms), while wall-clock time falls from
127.2 to 55.4 ms. The whole 71.8 ms difference is the per-call weight scatter. At
512×1024×1024 the two device times are within 2%. At 256×1024×1024, 512×2048×2048 and
512×3840×4096 the one-shot device time is 0.67x, 1.43x and 1.48x the resident one
[host-computed]. The cause of those three differences is not characterized.

### Fixed per-call overhead

At small shapes almost none of a matmul's wall-clock time is device time. The 64×256×256 shape
takes 0.09 ms of device time in a 10.26 ms call. The 128×512×512 call is shorter than the
smaller one. The one-shot path therefore has a fixed per-call overhead of roughly 9-10 ms,
independent of shape at small shapes.

The instrument cannot divide that overhead further, because the ioctl entry, the BO and IOMMU
setup and the fence plumbing all fall outside its span. At these shapes `wall − dev` is host
work and dispatch cost together, and quoting it as the dispatch overhead overstates what was
measured.

## The `hw_elapse_time` instrument

`rknpu_hw_elapse_total_ns()` (provider `rknpu-submit`) sums `rknpu_submit.hw_elapse_time`
over every submit a process has made since `rocket_submit_counters_reset()`.

The value is a `ktime` pair, a software timestamp rather than a hardware counter
[source-confirmed, `rknpu_job.c` in `rknpu` 0.9.8]. The driver stamps `hw_commit_time` where
it pulls the job off the core's todo list, immediately before it writes the register program.
It takes the delta in the IRQ path. The value therefore contains the register-write burst and the interrupt
latency. It excludes the ioctl entry, the BO and IOMMU setup, the fence plumbing, and every
host-side cost.

The RK3588 has no byte counters to go with it. Its driver configuration leaves `amount_top`
and `amount_core` NULL, and bandwidth QoS is unimplemented there [source-confirmed,
`rknpu_drv.c` in `rknpu` 0.9.8]. The SRAM pool is not in that configuration. The driver takes
it from the device tree's `rockchip,sram` phandle on any part [source-confirmed, `rknpu_drv.c`
in `rknpu` 0.9.8]. The RK1 measured has none [HW sweep].

The value has three limits that it does not show:

- Device time is not MAC time. The span covers whatever the core does between commit and
  interrupt, DMA included, so the instrument cannot separate MAC-bound from DMA-bound. It
  separates *device* from *host*. See [not-mac-bound.md](not-mac-bound.md) for the other axis.
- A multicore job reports one core's span, not their sum: each core's completion overwrites
  the job's elapsed time. The accumulator is also process-wide. A run with several concurrent
  worker fds therefore sums *core* time, which can exceed wall-clock time.
- The driver commits a job of more than `max_submit_number` tasks (4095 on the RK3588) in
  chunks. `rknpu_job_done()` commits each next chunk without re-stamping `hw_commit_time`, so
  the job's contribution spans every chunk from the first commit [source-confirmed,
  `rknpu_job.c` in `rknpu` 0.9.8]. This limit does not apply at the task counts `librocketnpu` emits.

## Usage

Reset the counters immediately before the region to measure, and read the total immediately
after. Every submit overwrites the per-fd `rknpu_last_hw_elapse_ns()`, so that call reports
only the last kick of a many-kick matmul. To compare the total with wall-clock time, run one
worker. Discard a warm-up repetition. The NPU clock drops to its idle rate while the NPU is
idle, so a cold run measures low ([clock.md](clock.md#the-200-mhz-boot-rate)).

## Evidence

### Measurement method

Each test point is the best of five warm repetitions, after a discarded warm-up. The sweep
harnesses are `hwsplit.c` (one-shot) and `hwsplit2.c` (resident). Both build by hand against
the provider and the driver archive, alongside the other board instruments.

### Per-call weight scatter

At 1024×3840×4096 the same device work measures identically through the two host paths, whose
wall-clock times differ by 2.3x. The agreement is the internal check on the instrument. The
other ratios in §"Per-call weight scatter" are the table's device columns divided.
