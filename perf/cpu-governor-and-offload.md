# The CPU governor under an offloading process

A process that hands its heavy arithmetic to the NPU is blocked while the NPU runs, with its
threads off the run queue. A load-sampling CPU governor reads that as idle and lowers the big
cores toward their minimum frequency. The half of the work that never left the host, the cube
scatter and gather, then runs at that frequency. The NPU-offloaded configuration slows, but the
CPU-only configuration does not. A comparison taken under the default governor therefore
understates the offload, and can show it as a regression [HW sweep 2026-08-25].

**Conditions:** RK3588 (Turing RK1), the mainline `rocket` driver, NPU clock 928 MHz (the
900 MHz request), A76 `scaling_min_freq` 1 200 MHz. Three TFLite detectors through tflite-rocket
at `native_int8=1`, each latency the median of 30 warm invokes, 2026-10-29. The `performance`
configuration also runs the process on the A76 cores, and the `ondemand` configuration leaves it
unpinned.

**Not characterized:** the governor's effect at 928 MHz apart from that pinning. The `cpu-sleep`
gain on larger models and on quantized ones. The separate effects of the governor and of
`cpu-sleep` on whisper.cpp's decoder.

| Detector | `performance` | `ondemand` | `ondemand` over `performance`, per pass |
|---|---:|---:|---:|
| SSDLite-MobileDet | 51.4 ms | 73.5 ms | 1.29-1.50x |
| SSD MobileNet v2 | 47.1 ms | 64.3 ms | 1.34-1.40x |
| EfficientDet-Lite0 | 94.5 ms | 127.8 ms | 1.34-1.37x |

On this board and on a vendor-kernel board, the CPU-only configuration is flat to a tenth of a
millisecond under both governors [HW sweep 2026-08-25]. A multi-threaded XNNPACK inference keeps
every core busy, so `ondemand` raises the clock and keeps it raised. On the vendor board, `rknpu`
0.9.8 on kernel 6.1, the delegated SSDLite-MobileDet took 199.7 ms on `performance` and 639.4 ms
on `ondemand`, 3.2x. Its CPU interpreter took 183.4 and 183.3 ms.

## Cause

The size of the penalty is set by the minimum frequency the governor is allowed to set on the
A76 cluster. The two boards run the same governor with the same tunables (`up_threshold` 95,
`sampling_rate` 6.7-10 ms, `powersave_bias` 0), and differ in that minimum:

**Conditions:** SSDLite-MobileDet at `native_int8=1` on one delegate build, on both boards,
2026-08-25 [HW sweep].

| board | A76 `scaling_min_freq` | A76 `scaling_max_freq` | ratio | measured penalty |
|---|---|---|---|---|
| mainline, kernel 7.2 | 1 200 MHz | 2 400 MHz | 2.0x | 1.27x |
| vendor, kernel 6.1 | 408 MHz | 2 352 MHz | 5.8x | 3.2x |

The penalty follows the platform's CPU frequency configuration, and the mechanism is identical
under both drivers. A board read right after a delegated run shows the cluster at its minimum
frequency.

## The deep CPU idle state

Every RK3588 core has one deep idle state, `cpu-sleep`, a PSCI power-down. Its sysfs exit
latency is 220 µs and its target residency 1000 µs. The `menu` idle governor picks it for a
predicted idle of 1 ms or more. Disabling it makes offloaded prefill ~3% faster and leaves
CPU-only prefill unchanged [HW sweep 2026-09-26]. The bias has the governor's direction, and
is smaller.

**Conditions:** the mainline board, `rocket` 1.3.0, NPU clock ~816 MHz, CPU governor
`performance` on every policy, `llama-bench -p 512 -n 0`.

| model | configuration | ABBA blocks | disabled / enabled, per block |
|---|---|---:|---|
| Qwen3.5-0.8B F16 | NPU, `ROCKET_KACC=1` | 4 | 1.029, 1.023, 1.027, 1.032 |
| Qwen3.5-0.8B F16 | CPU only | 2 | 1.005, 1.001 |
| Llama-3.2-3B F16 | NPU, `ROCKET_KACC=1` | 2 | 1.030, 1.040 |

The likely cause is the host thread pool, which idles for over a millisecond while a matmul
runs. Each hand-back to the pool then takes the state's exit latency [hypothesis]. The NPU
completion's wakeup does not account for the cost (§"Excluded causes").

## The DDR frequency governor on a vendor kernel

The vendor kernel also scales the DDR frequency, and there the bias runs the other way. The
CPU-only configuration slows, but the NPU configuration does not. Its `dmc` devfreq node runs
`dmc_ondemand` over 528, 1068, 1560 and 2112 MHz. An NPU job drives DDR to the top rate by
itself, and a multi-threaded CPU encoder does not. Pinning `dmc` to `performance` cuts the CPU
configuration's time by 4.7% and removes nearly all of its spread. The NPU configurations move by
nothing measurable [HW sweep 2026-09-26].

**Conditions:** the vendor RK1 on `6.1.172-vendor-rk35xx`, `rknpu` 0.9.8, NPU clock 1000 MHz,
CPU governor `performance` on every policy, `taskset -c 4-7`. The workload is the whisper
base.en encoder at a 20 s window.

| configuration | `dmc_ondemand` (ms) | `dmc` at 2112 MHz (ms) | DDR below 2112 MHz under `dmc_ondemand` |
|---|---:|---:|---|
| CPU only, `whisper-cli -t 4` | 858.0 (847.8-876.3) | 817.3 (817.1-817.5) | 44-58% of samples |
| ggml-rocket drop-in | 577.7 | 574.1 | 12-30% of samples |
| RKNN whole-graph encoder | 344.05 | 343.95 | 2-9% of samples |

The mainline kernel on the same board exposes no DDR devfreq node, so this bias cannot arise
there. DDR runs at the rate the bootloader set.

## LLM prefill

The same bias applies to llama.cpp prefill through ggml-rocket. NPU prefill is 1.130x faster
under `performance` than under `ondemand` on DeepSeek-V2-Lite `Q4_K_M`, and 1.079x faster on
Phi-4 `Q4_K_M` [HW sweep 2026-09-29]. The CPU-only configuration of the same models keeps
every core busy. It matches, within 2%, an earlier measurement series that recorded no governor
([cpu-repack-baseline.md](cpu-repack-baseline.md)).

**Conditions:** the mainline RK1, `rocket` 1.3.0, NPU clock ~816 MHz, pp2048 with
`-b 2048 -ub 2048`, llama.cpp b11242.

The size of the bias depends on the configuration. On Llama-3.2-3B `Q4_K_M` at the default `-ub`,
NPU prefill is only 1.018x faster under `performance` [HW sweep 2026-11-07]. There `ondemand`
holds the A76 cores at 2.4 GHz for ~83% of their busy time, an effective 2.06 GHz against
2.11 GHz pinned. A cycles-per-busy-time readout therefore cannot tell the two governors apart on
such a process.

**Conditions:** the mainline RK1, `rocket` 1.3.0, NPU clock 928 MHz, pp2048 at the default `-b`
and `-ub`, llama.cpp b11539.

## A per-node offload inside a model

A frontend that offloads single layers incurs the host's costs once per layer. Four of them
were measured on the resident im2col convolution, alone and inside ONNX Runtime. They are the
governor, the deep idle state, the core that each call's workers start on, and scratch memory
allocated per call.

### The governor and the idle state

Both the `ondemand` governor and `cpu-sleep` lengthen a single-layer call that follows an idle
gap [HW sweep].

**Conditions:** RK1, NPU clock ~816 MHz, `rocket` 1.3.0, 2026-10-09 UTC. An idle gap is slept
before each timed call, outside the timing.

| Layer | `ondemand`, no gap / 100 ms gap | `performance` | `performance`, `cpu-sleep` off |
|---|---|---|---|
| 3x3 stride 2, 384×37×37 -> 384 | 3.53 / 5.83 ms | 2.96 / 3.93 | 2.85 / 2.88 |
| 3x3, 96×74×74 -> 64 | 5.86 / 7.39 | 5.41 / 5.83 | 4.81 / 4.89 |
| 3x3, 64×148×148 -> 64 | 11.94 / 13.14 | 10.92 / 11.01 | 10.08 / 9.98 |

`ondemand` slows even the back-to-back call, because its `up_threshold` is 95, and a call whose
workers block on the fence reads as below that. The A76s were at their 1 200 MHz minimum
frequency for 26-66% of such a run.

Under `performance`, a 100 ms gap still costs up to 0.96 ms on the smallest layer. Turning
`cpu-sleep` off removes that and 1-11% of the back-to-back call. A completion waiter that spins
(`ROCKET_BUSY_POLL=20000`) recovers none of it. Over 16 shapes, it gives a median of -4% of
the gain from turning `cpu-sleep` off [HW sweep]. The state's cost therefore lies in the
threads each call starts and wakes, not in the waiter [hypothesis].

### Worker threads started from a pinned caller

A thread inherits its creator's CPU affinity mask. Depth Anything's encoder pins its calling
thread to one A76. Every per-node call made from that thread then started all of its workers on
that core. Each worker queued there until it could pin itself to its own core. The claimed
convolutions ran at 1.27-1.29x their back-to-back time, and with each worker created on its own
core they run at 1.01-1.04x. The model's wall-clock time moves 0.969-0.981x, and that of
pix2pix, whose caller is not pinned, 0.926-0.958x [HW sweep].

**Conditions:** RK1, CPU governor `performance`.

A benchmark whose own caller is unpinned cannot show this cost, however its gaps and governor
are set. `librocketnpu` creates every per-call worker on its target core.

### Scratch allocated per call

A buffer that a per-node call allocates and frees each time can return to the kernel between
calls. Every call then faults it in again. Depth Anything's encoder allocated and freed
~17 MiB of scratch per encode. With the `Conv` claim on, the process took 2.8-5.2 k minor faults
an image, and with the claim off none [HW sweep 2026-10-09].

**Conditions:** RK1, NPU clock ~816 MHz, Depth Anything v2 Small through ort-rocket at
`ORT_DISABLE_ALL`, 4 threads on the A76 cores, ONNX Runtime spinning off, 2026-10-09.

Two changes each remove the faults:

- Disabling glibc's heap trim and its dynamic mmap threshold takes the faults to 0. It also
  takes the encoder's added time with the claim on from 29-32 ms to 14-15 ms. An application can
  set `GLIBC_TUNABLES=glibc.malloc.trim_threshold=2147483648:glibc.malloc.mmap_threshold=33554432`.
- Holding the scratch in the library's context, allocated once, takes the faults to 0.
  `librocketnpu`'s SigLIP-family encoder holds it so.

### Whole models

On a whole model, the governor moves the CPU-only configuration as well as the offloaded ones
[HW sweep 2026-10-27].

**Conditions:** pix2pix through ort-rocket, 4 intra-op threads on the A76 cores, both sessions at
`ORT_ENABLE_ALL`, ONNX Runtime spinning off, RK1, NPU clock 928 MHz (the 900 MHz request),
2026-10-27.

| Configuration | `performance` | `ondemand` | `performance` over `ondemand` |
|---|---:|---:|---:|
| CPU EP | 146.6 ms | 158.3 ms | 0.926 |
| ConvTranspose claim only | 76.4 ms | 84.5 ms | 0.904 |
| ConvTranspose and Conv claims | 47.4 ms | 54.3 ms | 0.873 |

The ratio column is the ratio of the medians [host-computed]. Against the CPU EP, both claims
measure 3.09x on `performance` and 2.91x on `ondemand`.

Under `performance`, turning `cpu-sleep` off takes the CPU EP to 0.993x its time [HW sweep
2026-10-09, ~816 MHz]. It takes the ConvTranspose claim alone to 0.985x and both claims to 0.951x.
That measurement predates the per-call worker placement above.

With ONNX Runtime's spinning off, the CPU-only configuration's threads block between operators,
and `ondemand` lowers their clock as well. The XNNPACK configuration of the detection
measurement keeps every core busy and does not move. A CPU-only configuration with spinning off
therefore has the same bias as the NPU configuration, smaller. The entry these numbers ran on
is described in [../matmul-as-conv.md](../matmul-as-conv.md) §"A convolution on the resident
matmul".

## whisper.cpp's CPU decoder

A CPU workload that runs a loop of short steps gains more from the `performance` governor and
`cpu-sleep` off than an NPU one does. On the CPU, whisper.cpp's greedy decoder takes 0.70-0.71x
as long per base step under `performance` with `cpu-sleep` off as under `ondemand`
[HW sweep 2026-10-10].

**Conditions:** whisper.cpp v1.9.4, greedy decoding (`-bs 1 -nf`), 4 threads,
`taskset -c 4-7`. The NPU row at NPU clock ~816 MHz.

| Decoder step | `ondemand` | `performance`, `cpu-sleep` off |
|---|---:|---:|
| CPU, tiny | 5.5 ms | 3.7-3.8 ms |
| CPU, base | 11.6 ms | 8.1-8.3 ms |
| CPU, small | 31.4 ms | 22.4-22.6 ms |
| NPU, base, resident weights | 27.3 ms | 21.1-22.1 ms |

An `ondemand` comparison of the two decoders is therefore biased toward the NPU, the opposite of
the busy XNNPACK case above. Between steps the sampling runs on one thread while the other
three wait, which lets `ondemand` lower the cluster's clock [hypothesis]. The NPU comparison is
in [decode-gemv.md](decode-gemv.md) §"A Whisper decoder step on resident weights".

## Governor, idle-state and DDR settings

For an NPU-against-CPU comparison and for a deployment:

- **Pin the CPU governor before quoting any NPU-versus-CPU number**, and put it back
  afterwards. Reading `scaling_governor` is not enough: read `scaling_min_freq` too, because
  that is what sets the cost.
- **Disable `cpu-sleep` for the same comparison**, with `echo 1` into every core's
  `cpuidle/state1/disable`, and put it back afterwards. It is worth ~3% to the NPU
  configuration at `pp512` and nothing to the CPU-only configuration. A per-node offload gains
  5% (pix2pix through ort-rocket), and its spin-off CPU configuration 0.7%. In a deployment, a
  service holding a `/dev/cpu_dma_latency` request under 220 µs keeps the state off at an
  idle-power cost [expected].
- **On a vendor kernel, pin DDR as well**, with `performance` in `/sys/class/devfreq/dmc/governor`,
  and put it back afterwards. Left on `dmc_ondemand` it slows the CPU-only configuration by ~5%
  and widens its spread, which flatters the NPU configuration.
- **In a deployment, raise the minimum frequency rather than changing the governor.** A
  detection service does not need `performance` on every core. It needs the cores its
  offloading process runs on to stay above their minimum frequency. Either `scaling_min_freq`
  or an affinity that keeps one busy thread on the cluster works.

## Applicability

The effect grows with the host share of the workload. A detection inference on the RK3588 is
bound by the host cube gather, so the effect is large there. A workload whose wall-clock time
is mostly device time is expected to show little of it [expected]. The method for splitting a
workload's time between device and host is in [device-vs-host-split.md](device-vs-host-split.md).
The reason the host share is large on this silicon is in [not-mac-bound.md](not-mac-bound.md).

## Evidence

### Detection measurement

The 928 MHz table is four passes, one process per configuration [HW sweep 2026-10-29]. The
governor order and the model order rotated, and memory was reset before every process. Each ratio
pairs the two governors within a pass.

The two-board measurement took twelve invokes a configuration, with the first three discarded.
The governor configurations were interleaved, and the run was repeated twice [HW sweep
2026-08-25].

### The deep CPU idle state

The configurations ran in balanced ABBA blocks, with the page cache dropped and memory
compacted before each. Each ratio pairs the two configurations inside one block.

Two runs in one measurement series measured 13-15% faster in both configurations at once.
Pairing inside the block canceled them, and their cause is not located. An unpaired A/B would
have read them as an effect.

### The DDR frequency governor on a vendor kernel

The configurations alternated the DDR governor over three passes, with DDR sampled every 50 ms.
Each value is the median of the three passes.
The samples have no timestamps, so which phase of a run the low ones fall in is not known.

### LLM prefill

Each ratio is paired over three rotated passes, each within 0.01 of the mean. The Llama-3.2-3B
ratio is two rotated passes, 25.22-25.23 t/s against 24.76-24.79, each process after a memory
reset and a discarded warm-up. A76 cycles are `perf stat -a` on cores 4-7 over the timed
process. Busy time is the cores' `/proc/stat` user, system, irq and softirq time, and the time at
each frequency is `cpufreq` `time_in_state`.

### A per-node offload inside a model

The single-layer table is two passes. The worker-thread figures are one binary over three
passes. The whole-model table is medians over five passes, with the configuration order and the
governor order rotated, and memory reset before each configuration. Its paired speedups with both
claims span 2.95-3.10x on `performance` and 2.89-2.94x on `ondemand`. The `cpu-sleep` ratios pair
two configurations inside one of five rotated passes.

The heap-trim figures are two rotated passes under `performance`, with the claim off and on. The
context-owned scratch is four rotated passes under `ondemand`. There the faults with the claim on
measured 4.1-5.2 k an image before the change and 0 after. The `performance` passes measured
2.8-3.9 k, the low end of the reported range.

### whisper.cpp's CPU decoder

The `ondemand` figures are single processes, and the `performance` ones are three passes with
the configuration order rotated.

### Revision history

| Measured | Quantity | Value | Conditions |
|---|---|---|---|
| 2026-08-25 | SSDLite-MobileDet on the mainline board, delegate and CPU interpreter | `performance` 199.4 and 198.0 ms, `ondemand` 252.3 and 198.2 ms | `rocket` 1.3.0, kernel 7.2, that date's delegate build |
| 2026-10-09 | pix2pix, `performance` over `ondemand` | CPU EP 0.928, ConvTranspose claim 0.892, both claims 0.884 | ~816 MHz, before the per-call worker placement change |

### Excluded causes

- **The NPU completion's wakeup, for the `cpu-sleep` cost on prefill.** On one fd, a job of 16
  tasks lasting ~3 ms moves by +14 µs with the stock IRQ affinity, the sign split across blocks.
  With the IRQs on an A76 it moves by -33 µs, at most 1%. Both are the `cpu-sleep`-disabled
  configuration minus the enabled one. Measured with
  `submit_overhead_rocket 256 512 256 400 8000 16` on the same board, the busy-polling waiter
  moves as much as the blocking one [HW sweep]. The waiter is therefore not where the state
  costs anything.
- **NPU runtime suspend, for the single-layer gap cost.** The NPU cannot be the cause at these
  gaps, because each core's runtime PM waits 1000 ms before it suspends [live, kernel 7.2.9].
