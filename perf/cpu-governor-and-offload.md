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

## The deep CPU idle state

Every RK3588 core has one deep idle state, `cpu-sleep`, a PSCI power-down. Its sysfs exit
latency is 220 µs and its target residency 1000 µs. The `menu` idle governor picks it for a
predicted idle of 1 ms or more. **Disabling it makes the offloaded arm of a prefill ~3% faster
and leaves the CPU-only arm unchanged** [HW sweep 2026-09-26]. It is the governor's bias again,
smaller.

The setup was the mainline board with `rocket` 1.3.0 at 600 MHz, the governor `performance` on
every policy, and `llama-bench -p 512 -n 0`. The arms ran in balanced ABBA blocks, with the page
cache dropped and memory compacted before each. Each ratio pairs the two arms inside one block.

| model | arm | ABBA blocks | disabled / enabled, per block |
|---|---|---:|---|
| Qwen3.5-0.8B F16 | NPU, `ROCKET_KACC=1` | 4 | 1.029, 1.023, 1.027, 1.032 |
| Qwen3.5-0.8B F16 | CPU only | 2 | 1.005, 1.001 |
| Llama-3.2-3B F16 | NPU, `ROCKET_KACC=1` | 2 | 1.030, 1.040 |

The cost is not the NPU completion's wakeup. On one fd, a job of 16 tasks lasting ~3 ms moves by
+14 µs with the stock IRQ affinity, the sign split across blocks. With the IRQs on an A76 it
moves by -33 µs, at most 1%. Both are the disabled arm minus the enabled one. The busy-polling
waiter moves as much as the blocking one [HW sweep, same board,
`submit_overhead_rocket 256 512 256 400 8000 16`]. So the waiter is not where the state costs
anything.

The likely cause is the host thread pool, idle for over a millisecond while a matmul runs and
paying the exit at each hand-back (inferred, not isolated).

Two runs in one campaign read 13-15% faster in both arms at once. Pairing inside the block
cancelled them, and their cause is not located. An unpaired A/B would have read them as an
effect.

## What to do

- **Pin the CPU governor before quoting any NPU-versus-CPU number**, and put it back
  afterwards. Reading `scaling_governor` is not enough: read `scaling_min_freq` too, because
  that is what sets the cost.
- **Disable `cpu-sleep` for the same comparison**, with `echo 1` into every core's
  `cpuidle/state1/disable`, and put it back afterwards. It is worth ~3% to the NPU arm at
  `pp512` and nothing to the CPU arm. Larger models and quantized ones are untested. In a
  deployment, a service holding a `/dev/cpu_dma_latency` request under 220 µs keeps the state
  off at an idle-power cost **[expected]**.
- **In a deployment, raise the floor rather than the governor.** A detection service does not
  need `performance` on every core; it needs the cores its offloading process runs on not to
  park. `scaling_min_freq`, or an affinity that keeps one busy thread on the cluster, both
  work.
- **The effect grows with the host share of the workload.** A detection inference on this part
  is host cube-gather-bound, which is why it shows here so strongly; a workload whose wall is
  mostly device time would barely notice. See
  [device-vs-host-split.md](device-vs-host-split.md) for how to split one, and
  [not-mac-bound.md](not-mac-bound.md) for why the host share is large on this silicon.
