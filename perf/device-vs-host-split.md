# Matmul device time and host time

On the vendor `rknpu` driver, each submit returns `hw_elapse_time`, the submit's device time.
It runs from the job's commit, just before the driver writes the register program, to the
completion interrupt [source-confirmed, `rknpu_job.c`]. It separates a matmul's device time
from its host time, and mainline `rocket` has no equivalent. With the weight resident, the
device takes 67-82% of an fp16 matmul's wall-clock time [HW sweep]. With the weight packed on
every call, the device time is unchanged, and the difference is the per-call weight scatter.

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

The host share is a property of the one-shot path, and the device work is the same on both
paths. At 1024×3840×4096 the device time is the same to within 0.01 ms on both (42.90 against
42.91 ms), while wall-clock time falls from 127.2 to 55.4 ms. The whole 71.8 ms difference is
the per-call weight scatter.

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
[source-confirmed, `rknpu_job.c`]. The driver stamps `hw_commit_time` where it pulls the job
off the core's todo list, immediately before it writes the register program. It takes the
delta in the IRQ path. The value therefore contains the register-write burst and the interrupt
latency. It excludes the ioctl entry, the BO and IOMMU setup, the fence plumbing, and every
host-side cost.

The RK3588 has no byte counters to go with it. Its driver configuration leaves `amount_top`
and `amount_core` NULL, and SRAM and bandwidth QoS are unimplemented there [source-confirmed].

The value has three limits that it does not show:

- Device time is not MAC time. The span covers whatever the core does between commit and
  interrupt, DMA included, so the instrument cannot separate MAC-bound from DMA-bound. It
  separates *device* from *host*. See [not-mac-bound.md](not-mac-bound.md) for the other axis.
- A multicore job reports one core's span, not their sum: each core's completion overwrites
  the job's elapsed time. The accumulator is also process-wide. A run with several concurrent
  worker fds therefore sums *core* time, which can exceed wall-clock time.
- The driver re-commits a job of more than `max_submit_number` tasks (4095 on the RK3588) in
  chunks. It re-stamps `hw_commit_time` each time, so the job's contribution covers only the
  final chunk. This limit does not apply at the task counts `librocketnpu` emits.

## Usage

Reset the counters immediately before the region to measure, and read the total immediately
after. Every submit overwrites the per-fd `rknpu_last_hw_elapse_ns()`, so that call reports
only the last kick of a many-kick matmul. To compare the total with wall-clock time, run one
worker. Discard a warm-up repetition. The NPU clock drops to its idle rate while the NPU is
idle, so a cold run measures low.

## Evidence

### Measurement method

Each test point is the best of five warm repetitions, after a discarded warm-up. The sweep
harnesses are `hwsplit.c` (one-shot) and `hwsplit2.c` (resident). Both build by hand against
the provider and the driver archive, alongside the other board instruments.

### Per-call weight scatter

The same device work measures identically through the two host paths, whose wall-clock times
differ by 2.3x. The agreement is the internal check on the instrument.
