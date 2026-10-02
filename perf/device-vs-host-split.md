# Splitting a matmul's wall into device time and host time

On the vendor `rknpu` path the driver writes back a per-submit elapsed time. It is the first
way on the RK3588 to say how much of a matmul's wall the device held. Mainline `rocket` has
no equivalent. This note records what that instrument says and, first, what it does not
measure. The envelope decides how to read the numbers.

## The instrument and its envelope

`rknpu_hw_elapse_total_ns()` (provider `rknpu-submit`) sums `rknpu_submit.hw_elapse_time`
over every submit a process has made since `rocket_submit_counters_reset()`.

The value is a `ktime` pair, not a hardware counter [source-confirmed, `rknpu_job.c`]. The
driver stamps `hw_commit_time` where it pulls the job off the core's todo list, immediately
before it writes the register program. It takes the delta in the IRQ path. So the value
contains the register-write burst and the interrupt latency. It excludes the ioctl entry, the
BO and IOMMU setup, the fence plumbing, and every host-side cost.

There are no byte counters on this part to go with it. The RK3588's driver config leaves
`amount_top` and `amount_core` NULL, and SRAM and bandwidth QoS are unimplemented there
[source-confirmed].

Three bounds are invisible in the value itself:

- Device time is not MAC time. The span covers whatever the core does between commit and
  interrupt, DMA included, so this instrument cannot separate MAC-bound from DMA-bound. It
  separates *device* from *host*. See [not-mac-bound.md](not-mac-bound.md) for the other axis.
- A multicore job reports one core's span, not their sum: each core's completion overwrites
  the job's elapsed time. Separately, the accumulator is process-wide. So a run with several
  concurrent worker fds sums *core* time, which can exceed the wall. When you compare the
  number against a wall clock, measure single-worker.
- The driver re-commits a job carrying more than `max_submit_number` tasks (4095 on the
  RK3588) in chunks. It re-stamps `hw_commit_time` each time, so the job's contribution
  covers only the final chunk. This bound is inert at the task counts `librocketnpu` emits.

## Measurements

The operating point is an fp16 matmul on the RK3588 at 1000 MHz, with `ROCKET_KACC=1` and one
worker. Each cell is the best of five warm reps with a warm-up discarded, with the board to
itself [HW sweep, `rknpu` 0.9.8, 2026-08-25]. "One-shot" is `rocket_matmul_fp16`, which packs
the weight every call. "Resident" is `rocket_matmul_fp16_prepacked`, which scatters the
weight once.

| M×K×N | submits | one-shot wall | dev | dev % | resident wall | dev | dev % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64×256×256 | 1 | 10.26 ms | 0.09 | 0.9% | n/a | n/a | n/a |
| 128×512×512 | 1 | 9.58 ms | 0.60 | 6.2% | n/a | n/a | n/a |
| 256×1024×1024 | 2 | 19.13 ms | 2.58 | 13.5% | 4.72 ms | 3.85 | 81.6% |
| 512×1024×1024 | 2 | 25.53 ms | 4.89 | 19.2% | 7.16 ms | 4.81 | 67.2% |
| 512×2048×2048 | 4 | 37.44 ms | 8.20 | 21.9% | 8.24 ms | 5.74 | 69.7% |
| 512×3840×4096 | 8 | 106.99 ms | 31.70 | 29.6% | 27.76 ms | 21.45 | 77.3% |
| 1024×3840×4096 | 16 | 127.19 ms | 42.90 | 33.7% | 55.42 ms | 42.91 | 77.4% |

The host share belongs to the one-shot path, not to the datapath. At 1024×3840×4096 the
device time is the same to two decimals in both arms (42.90 against 42.91 ms) while the wall
falls from 127.2 to 55.4 ms. The whole 71.8 ms difference is the per-call weight scatter. With
the weight resident the device holds 67-82% of the wall. That is the internal check on the
instrument as much as it is the result. The instrument reports the same device work
identically through two host paths that differ by 2.3x in wall time.

Almost none of a small matmul's wall is device time. The 64×256×256 shape spends 0.09 ms of a
10.26 ms wall on the device. The 128×512×512 shape has a shorter wall than the smaller shape.
So the one-shot path carries a per-call floor of roughly 9-10 ms that is independent of shape
at this end.

The instrument cannot split that floor further. The ioctl entry, the BO and IOMMU setup and
the fence plumbing all sit in the excluded region. So `wall − dev` there is host work and
dispatch cost together, and quoting it as a dispatch floor overstates what was measured.

## Usage

The two sweep harnesses behind the table are `hwsplit.c` (one-shot) and `hwsplit2.c`
(resident). Both build by hand against the provider and the driver archive, alongside the
other board instruments.

Reset the counters immediately before the region you want, and read the total immediately
after. Every submit overwrites the per-fd `rknpu_last_hw_elapse_ns()`, so that call reports
only the last kick of a many-kick matmul. If you divide the number by a wall clock, keep the
run single-worker. Take a warm-up rep: the clock parks at idle, so a cold run reads low.
