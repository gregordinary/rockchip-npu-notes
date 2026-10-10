# The CPU governor under an offloading process

A workload that hands its heavy arithmetic to the NPU spends that time blocked, with its
threads off the run queue. A load-sampling CPU governor reads that as idle and drops the big
cores toward their floor. The half of the work that never left the host, the cube scatter and
gather, then runs at that floor. **The NPU arm pays the penalty and the CPU-only arm does
not.** So an A/B taken under the default governor understates the offload and can read as a
regression [HW sweep 2026-08-25].

The measurement ran the same TFLite detector (SSDLite-MobileDet, `native_int8=1`) on two
RK3588 boards. Each arm took twelve invokes, with the first three discarded. The governor arms
were interleaved, and the run was repeated twice:

| board | CPU governor | NPU delegate (median) | plain CPU TFLite (median) |
|---|---|---|---|
| mainline `rocket` 1.3.0, kernel 7.2 | `performance` | 199.4 ms | 198.0 ms |
| | `ondemand` | **252.3 ms** | 198.2 ms |
| vendor `rknpu` 0.9.8, kernel 6.1 | `performance` | 199.7 ms | 183.4 ms |
| | `ondemand` | **639.4 ms** | 183.3 ms |

The CPU-only arm is flat to a tenth of a millisecond in both governors on both boards. A
multi-threaded XNNPACK inference keeps every core busy, so `ondemand` ramps and stays ramped.
The delegated arm costs 1.27x on one board and 3.2x on the other.

## The CPU's minimum frequency and the size of the penalty

The two boards run the same governor with the same tunables (`up_threshold` 95,
`sampling_rate` 6.7-10 ms, `powersave_bias` 0). What differs is the floor the governor is
allowed to fall to on the A76 cluster:

| board | A76 `scaling_min_freq` | A76 `scaling_max_freq` | ratio | measured penalty |
|---|---|---|---|---|
| mainline, kernel 7.2 | 1 200 MHz | 2 400 MHz | 2.0x | 1.27x |
| vendor, kernel 6.1 | 408 MHz | 2 352 MHz | 5.8x | 3.2x |

So this is a platform-configuration effect, and it is not a property of either driver. The
mechanism is identical on both. The board whose A76 can park at 408 MHz pays about 2.5x more
for it. A board read right after a delegated run shows the cluster sitting at its floor.

## The deep CPU idle state

Every RK3588 core has one deep idle state, `cpu-sleep`, a PSCI power-down. Its sysfs exit
latency is 220 µs and its target residency 1000 µs. The `menu` idle governor picks it for a
predicted idle of 1 ms or more. Disabling it makes the offloaded arm of a prefill ~3% faster
and leaves the CPU-only arm unchanged [HW sweep 2026-09-26]. It is the governor's bias again,
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
canceled them, and their cause is not located. An unpaired A/B would have read them as an
effect.

## The DDR frequency governor on a vendor kernel

The vendor kernel also scales DDR, and there the bias runs the other way: **the CPU-only arm
pays and the NPU arm does not.** Its `dmc` devfreq node runs `dmc_ondemand` over 528, 1068, 1560
and 2112 MHz. An NPU job drives DDR to the top rate by itself, and a multi-threaded CPU encoder
does not. Pinning `dmc` to `performance` makes the CPU arm 4.7% faster and removes nearly all of
its spread. The NPU arms move by nothing measurable [HW sweep 2026-09-26].

The setup was the vendor RK1 on `6.1.172-vendor-rk35xx` with `rknpu` 0.9.8 and the NPU pinned at
1000 MHz. The CPU governor was `performance` on every policy, with `taskset -c 4-7`. The workload
was the whisper base.en encoder at a 20 s window. The arms alternated the DDR governor over three
passes, with DDR sampled every 50 ms:

| arm | `dmc_ondemand` (ms) | `dmc` at 2112 MHz (ms) | DDR below 2112 MHz under `dmc_ondemand` |
|---|---:|---:|---|
| CPU only, `whisper-cli -t 4` | 858.0 (847.8-876.3) | 817.3 (817.1-817.5) | 44-58% of samples |
| ggml-rocket drop-in | 577.7 | 574.1 | 12-30% of samples |
| RKNN whole-graph encoder | 344.05 | 343.95 | 2-9% of samples |

Each cell is the median of three passes. The samples carry no timestamps, so which phase of a
run the low ones fall in is not known. The mainline kernel on the same board exposes no DDR
devfreq node, so this bias cannot arise there. DDR runs at the rate the bootloader set.

## LLM prefill

The same bias holds on llama.cpp prefill through ggml-rocket. The NPU arm at pp2048
(`-b 2048 -ub 2048`, llama.cpp b11242) reads 1.130x faster pinned than under `ondemand` on
DeepSeek-V2-Lite `Q4_K_M` and 1.079x on Phi-4 `Q4_K_M`. That is paired over three rotated
passes, each within 0.01 of the mean, on the mainline RK1 with `rocket` 1.3.0 [HW sweep
2026-09-29]. The CPU arm of the same models keeps every core busy and reproduced a campaign that
recorded no governor within 2%. See [cpu-repack-baseline.md](cpu-repack-baseline.md).

## A per-node offload inside a model

A frontend that offloads single layers pays the host's costs once per layer. Three were measured on
the resident im2col convolution, alone and inside ONNX Runtime. They are the governor, the deep idle
state, and the core each call's workers start on.

### The governor and the idle state

An idle gap was slept before each timed call, outside the timing [HW sweep, RK1 at 600 MHz,
`rocket` 1.3.0, two passes, 2026-10-09 UTC]:

| Layer | `ondemand`, no gap / 100 ms gap | `performance` | `performance`, `cpu-sleep` off |
|---|---|---|---|
| 3x3 stride 2, 384×37×37 -> 384 | 3.53 / 5.83 ms | 2.96 / 3.93 | 2.85 / 2.88 |
| 3x3, 96×74×74 -> 64 | 5.86 / 7.39 | 5.41 / 5.83 | 4.81 / 4.89 |
| 3x3, 64×148×148 -> 64 | 11.94 / 13.14 | 10.92 / 11.01 | 10.08 / 9.98 |

The NPU cannot be the cause at these gaps, because each core's runtime PM waits 1000 ms before it
suspends [live, kernel 7.2.9]. `ondemand` slows even the back-to-back call, since its
`up_threshold` is 95. A call whose workers block on the fence reads as below that. The A76s spent
26-66% of such a run at their 1 200 MHz floor.

Pinned, a 100 ms gap still costs up to 0.96 ms on the smallest layer. Turning `cpu-sleep` off
removes that and 1-11% of the back-to-back call. A completion waiter that spins
(`ROCKET_BUSY_POLL=20000`) recovers none of it, a median of -4% of what `cpu-sleep` off buys over
16 shapes [HW sweep]. The state's cost therefore lies in the threads each call starts and wakes,
not in the waiter [hypothesis].

### Worker threads started from a pinned caller

A thread inherits its creator's CPU mask. Depth Anything's encoder pins its calling thread to one
A76. Every per-node call made from that thread then started all of its workers on that core. Each
queued there until it could pin itself to its own. The claimed convs ran 1.27-1.29x their
back-to-back time, and created on their own cores they run 1.01-1.04x. The model's wall moves
0.969-0.981x, and that of pix2pix, whose caller is not pinned, 0.926-0.958x [HW sweep, RK1,
`performance`, one binary, three passes].

A benchmark whose own caller is unpinned cannot show this cost, however its gaps and governor are
set. `librocketnpu` creates every per-call worker on its target core.

### Whole models

These ran pix2pix through ort-rocket at 4 threads with spinning off, before the placement change.
Each ratio pairs two arms inside one of five rotated passes, with memory reset before each arm
[HW sweep, RK1, 2026-10-09 UTC]:

| Arm | `performance` over `ondemand` | `cpu-sleep` off over `performance` |
|---|---:|---:|
| CPU EP | 0.928 | 0.993 |
| ConvTranspose claim only | 0.892 | 0.985 |
| ConvTranspose and Conv claims | 0.884 | 0.951 |

The CPU-only arm moves here, where the busy XNNPACK arm above did not. With ONNX Runtime's spinning
off, its threads block between operators and `ondemand` clocks them down as well. So a spin-off CPU
arm carries the same bias as the NPU arm, smaller. See [../matmul-as-conv.md](../matmul-as-conv.md)
§"A convolution on the resident matmul" for the entry these numbers ran on.

## Governor, idle-state and DDR settings

For an NPU-against-CPU comparison and for a deployment:

- **Pin the CPU governor before quoting any NPU-versus-CPU number**, and put it back
  afterwards. Reading `scaling_governor` is not enough: read `scaling_min_freq` too, because
  that is what sets the cost.
- **Disable `cpu-sleep` for the same comparison**, with `echo 1` into every core's
  `cpuidle/state1/disable`, and put it back afterwards. It is worth ~3% to the NPU arm at
  `pp512` and nothing to the CPU arm. A per-node offload gains 5% (pix2pix through ort-rocket),
  and its spin-off CPU arm 0.7%. Larger models and quantized ones are untested. In a
  deployment, a service holding a `/dev/cpu_dma_latency` request under 220 µs keeps the state
  off at an idle-power cost [expected].
- **On a vendor kernel, pin DDR as well**, with `performance` in `/sys/class/devfreq/dmc/governor`,
  and put it back afterwards. Left on `dmc_ondemand` it slows the CPU-only arm by ~5% and widens
  its spread, which flatters the NPU arm.
- **In a deployment, raise the floor rather than the governor.** A detection service does not
  need `performance` on every core. It needs the cores its offloading process runs on not to
  park. Either `scaling_min_freq` or an affinity that keeps one busy thread on the cluster
  works.

The effect grows with the host share of the workload. A detection inference on this part is
host cube-gather-bound, which is why it shows here so strongly. A workload whose wall is mostly
device time would barely notice. See [device-vs-host-split.md](device-vs-host-split.md) for how
to split one, and [not-mac-bound.md](not-mac-bound.md) for why the host share is large on this
silicon.
