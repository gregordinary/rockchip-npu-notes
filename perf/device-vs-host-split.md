# Splitting a matmul's wall into device time and host time

On the vendor `rknpu` path the driver writes back a per-submit elapsed time, which is the
first way on RK3588 to say how much of a matmul's wall the device actually held. Mainline
`rocket` has no equivalent. This note is what that instrument says and, first, what it does
not measure — the envelope decides how the numbers may be read.

## The instrument and its envelope

`rknpu_hw_elapse_total_ns()` (provider `rknpu-submit`) sums `rknpu_submit.hw_elapse_time`
over every submit a process has made since `rocket_submit_counters_reset()`.

**It is a `ktime` pair, not a hardware counter** [source-confirmed, `rknpu_job.c`].
`hw_commit_time` is stamped where the job is pulled off the core's todo list, immediately
before the register program is written, and the delta is taken in the IRQ path. So the value
**contains** the register-write burst and the interrupt latency, and **excludes** the ioctl
entry, the BO and IOMMU setup, the fence plumbing, and every host-side cost. There are no
byte counters on this part to go with it — RK3588's driver config leaves `amount_top` and
`amount_core` NULL, and SRAM and bandwidth QoS are unimplemented there [source-confirmed].

Three bounds are invisible in the value itself:

- **Device time is not MAC time.** The span covers whatever the core does between commit and
  interrupt, DMA included, so this instrument cannot separate MAC-bound from DMA-bound. It
  separates *device* from *host*. See [not-mac-bound.md](not-mac-bound.md) for the other axis.
- **A multicore job reports one core's span, not their sum** — each core's completion
  overwrites the job's elapsed time. Separately, because the accumulator is process-wide, a
  run with several concurrent worker fds sums *core* time, which can exceed the wall. Measure
  single-worker when the number is to be compared against a wall clock.
- **A job carrying more than `max_submit_number` tasks (4095 on RK3588) is re-committed in
  chunks**, re-stamping `hw_commit_time` each time, so its contribution covers only the final
  chunk. Inert at the task counts this library emits.

## What it measures

fp16 matmul, RK3588 at 1000 MHz, `ROCKET_KACC=1`, one worker, best of five warm reps with a
warm-up discarded, board to itself [HW sweep, `rknpu` 0.9.8, 2026-08-25]. "one-shot" is
`rocket_matmul_fp16`, which packs the weight every call; "resident" is
`rocket_matmul_fp16_prepacked`, where the weight is scattered once.

| M×K×N | submits | one-shot wall | dev | dev % | resident wall | dev | dev % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64×256×256 | 1 | 10.26 ms | 0.09 | **0.9%** | — | — | — |
| 128×512×512 | 1 | 9.58 ms | 0.60 | 6.2% | — | — | — |
| 256×1024×1024 | 2 | 19.13 ms | 2.58 | 13.5% | 4.72 ms | 3.85 | **81.6%** |
| 512×1024×1024 | 2 | 25.53 ms | 4.89 | 19.2% | 7.16 ms | 4.81 | 67.2% |
| 512×2048×2048 | 4 | 37.44 ms | 8.20 | 21.9% | 8.24 ms | 5.74 | 69.7% |
| 512×3840×4096 | 8 | 106.99 ms | 31.70 | 29.6% | 27.76 ms | 21.45 | 77.3% |
| 1024×3840×4096 | 16 | 127.19 ms | 42.90 | 33.7% | 55.42 ms | 42.91 | **77.4%** |

**The host share belongs to the one-shot path, not to the datapath.** At 1024×3840×4096 the
device time is the same to two decimals in both arms — 42.90 against 42.91 ms — while the wall
falls from 127.2 to 55.4 ms. The whole 71.8 ms difference is the per-call weight scatter, and
with the weight resident the device holds **67–82%** of the wall. That is the internal check on
the instrument as much as it is the result: the same device work is reported identically through
two host paths that differ by 2.3× in wall time.

**A small matmul is almost entirely not the device.** 64×256×256 spends 0.09 ms of a 10.26 ms
wall on the device, and 128×512×512 has a *shorter* wall than the shape below it — so the
one-shot path carries a per-call floor of roughly 9–10 ms that is independent of shape at this
end. What the instrument cannot do is split that floor further: the ioctl entry, BO and IOMMU
setup and fence plumbing all sit in the excluded region, so `wall − dev` there is host work and
dispatch cost together, and quoting it as a dispatch floor overstates what was measured.

## How to use it

The two sweep harnesses behind the table are `hwsplit.c` (one-shot) and `hwsplit2.c` (resident),
built by hand against the provider and the driver archive, alongside the other board instruments.

Reset the counters immediately before the region you want and read the total immediately after;
the per-fd `rknpu_last_hw_elapse_ns()` is overwritten by every submit and reports only the last
kick of a many-kick matmul. Keep the run single-worker if the number is going to be divided by a
wall clock. And take a warm-up rep: the clock parks at idle, so a cold run reads low.
