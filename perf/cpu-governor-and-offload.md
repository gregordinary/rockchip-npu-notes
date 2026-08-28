# An offloading process defeats a load-based CPU governor

A workload that hands its heavy arithmetic to the NPU spends that time blocked, with its
threads off the run queue. A load-sampling CPU governor reads that as idle and drops the big
cores toward their floor, and the half of the work that never left the host, the cube scatter
and gather, then runs at that floor. **The NPU arm pays the penalty and the CPU-only arm does
not**, so an A/B taken under the default governor understates the offload and can read as
a regression [HW sweep 2026-08-25].

Measured with the same TFLite detector (SSDLite-MobileDet, `native_int8=1`) on two RK3588
boards, twelve invokes per arm with the first three discarded, the governor arms interleaved
and the run repeated twice:

| board | CPU governor | NPU delegate (median) | plain CPU TFLite (median) |
|---|---|---|---|
| mainline `rocket` 1.3.0, kernel 7.2 | `performance` | 199.4 ms | 198.0 ms |
| | `ondemand` | **252.3 ms** | 198.2 ms |
| vendor `rknpu` 0.9.8, kernel 6.1 | `performance` | 199.7 ms | 183.4 ms |
| | `ondemand` | **639.4 ms** | 183.3 ms |

The CPU-only arm is flat to a tenth of a millisecond in both governors on both boards: a
multi-threaded XNNPACK inference keeps every core busy, so `ondemand` ramps and stays ramped.
The delegated arm costs **1.27x** on one board and **3.2x** on the other.

## The size of the penalty is the CPU's minimum frequency

The two boards run the same governor with the same tunables (`up_threshold` 95,
`sampling_rate` 6.7-10 ms, `powersave_bias` 0). What differs is the floor the governor is
allowed to fall to on the A76 cluster:

| board | A76 `scaling_min_freq` | A76 `scaling_max_freq` | ratio | measured penalty |
|---|---|---|---|---|
| mainline, kernel 7.2 | 1 200 MHz | 2 400 MHz | 2.0x | 1.27x |
| vendor, kernel 6.1 | 408 MHz | 2 352 MHz | 5.8x | 3.2x |

So this is a platform-configuration effect and **not a property of either driver**: the
mechanism is identical on both, and the board whose A76 may park at 408 MHz pays about
2.5x more for it. A board read right after a delegated run shows the cluster sitting at its
floor.

## What to do

- **Pin the CPU governor before quoting any NPU-versus-CPU number**, and put it back
  afterwards. Reading `scaling_governor` is not enough: read `scaling_min_freq` too, because
  that is what sets the cost.
- **In a deployment, raise the floor rather than the governor.** A detection service does not
  need `performance` on every core; it needs the cores its offloading process runs on not to
  park. `scaling_min_freq`, or an affinity that keeps one busy thread on the cluster, both
  work.
- **The effect grows with the host share of the workload.** A detection inference on this part
  is host cube-gather-bound, which is why it shows here so strongly; a workload whose wall is
  mostly device time would barely notice. See
  [device-vs-host-split.md](device-vs-host-split.md) for how to split one, and
  [not-mac-bound.md](not-mac-bound.md) for why the host share is large on this silicon.
