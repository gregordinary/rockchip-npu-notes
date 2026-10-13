# The NPU clock

The RK3588 NPU compute clock, `scmi_clk_npu`, is shared by all 3 cores. It boots at 200 MHz,
one fifth of the 1 GHz silicon maximum [source-confirmed, HW sweep]. Raising it to 600 MHz
gives ~1.43x on prefill, and 900 MHz gives no further speedup [HW sweep]. Erratum E10
(§"Raising the clock on an unpowered domain") restricts where software can set the rate.

**Conditions:** RK3588 (Turing RK1), mainline `rocket` with the clock patch, NPU clock 200 MHz
against 600 MHz, 2026-06-15, with the matmul's K-accumulation on the host.

**Not characterized:** operation above 600 MHz beyond the 900 MHz throughput reading, and this
chip's OTP PVTPLL ceiling (§"Applicability").

| Workload | 200 MHz | 600 MHz | Speedup |
|---|---:|---:|---:|
| Standalone fp16 `512×3840×4096` | 56.5 GFLOP/s | 75.3 GFLOP/s | 1.33x |
| Gemma pp2048, the clock changed alone | 7.98 t/s | 11.40 t/s | 1.43x |

## Cause

### The 200 MHz boot rate

Mainline `rocket` has no NPU devfreq. The driver never requests a higher OPP, so the SCMI
clock stays at its boot default. The device tree also sets it explicitly: the
`npu@fdab0000` nodes have `assigned-clock-rates = <200000000>` for the `npu` clock.

200 MHz is the vendor's `POWER_DOWN_FREQ`, the rate its driver sets at runtime suspend
[source-confirmed: BSP `rknpu_devfreq.c`]. A powered vendor NPU does not idle there. On a
vendor 6.1.172 kernel with `rknpu` 0.9.8, under the `rknpu_ondemand` governor, the clock runs
at 1000 MHz at idle. The governor's OPPs span 300 to 1000 MHz [HW sweep, vendor RK1].

With no devfreq to raise it, `rocket` leaves the clock at 200 MHz. Under load it stays at
200 MHz through a full prefill [source-confirmed, HW sweep].

The 200 MHz clock inflates the NPU `wait` ~5x. At 200 MHz, ~150 GFLOP/s per core is ~73% of
the 200 MHz fp16 peak, a plausible efficiency, and an implausible ~15% of the 1 GHz peak. The
prefill `wait` at 200 MHz therefore reflects the 200 MHz clock, and does not show that the NPU
is compute-bound at full clock.

### The size of the gain

Prefill gains 1.43x, not ~4.5x. At 600 MHz the NPU `wait` collapses, and the other limits
take over: host readback and per-job dispatch, both independent of the NPU clock. The
matmul's underlying limit is DMA and dispatch rather than compute
([not-mac-bound.md](not-mac-bound.md)). Raising the clock speeds up only the `wait`, which
shrinks with each step.

## Raising the clock on an unpowered domain

The NPU power domains (`PD_NPUTOP/NPU1/NPU2`) are off when the device is idle or cold,
because `rocket` uses runtime PM.

> [!CAUTION]
> Erratum E10: setting the SCMI NPU PLL while its power domain is off hangs the SCMI
> firmware (EL3) [HW sweep]. Software must set the rate inside the driver, after runtime PM
> powers the domain.

Both obvious ways to raise the rate set it with the domain off:

1. **A DT override** (`assigned-clock-rates = 600M`) hangs the boot. It is applied at
   `of_clk_set_defaults`, before the driver probes and powers the domain.
2. **A standalone out-of-tree `clk_set_rate` module** hangs the running system. It sets the
   rate at idle, with the domain off.

The vendor driver corroborates the rule. It sets the NPU clock only during operation, with
the domain powered, through devfreq. Its OPP helper refuses when `!pm_runtime_active`.

An upstream DVFS series for this SoC states the hazard more broadly: an island powered up
above the boot rate never acknowledges (patchwork series 1171089, v2, 2026-09-22). Under the
clock patch, child islands power on at a raised 600 MHz routinely, with no failure recorded.
Two explanations fit [hypothesis]. The hazard is specific to the island that holds the
PVTPLL, or the v1.51 blob serves 600 MHz from a normal PLL. One read of `CLKSEL_CON74` at
600 MHz distinguishes them.

## Setting the clock in `rocket`'s runtime resume

The clock patch, `081-rocket-drv-npu-clk.patch`, sets the rate where Erratum E10 requires:
inside the driver, after `pm_runtime` has powered the domain. It adds:

- A module param `rocket_npu_clk_hz` (default 0 = stock)
- A `clk_set_rate` in `rocket_device_runtime_resume()`, with the domain powered
- A park back to 200 MHz in `rocket_device_runtime_suspend()`, before the domain powers off

The patch builds as a module. The kernel image and the DTB are untouched, so the boot is
never at risk. Recovery is `rmmod` or a reboot. Probe calls `pm_runtime_resume_and_get`, so
a non-zero param is applied during `insmod` with the domain powered. An `insmod` that
returns cleanly therefore shows that the rate was not set with the domain off.

## The 600 MHz operating point

600 MHz is safe: int8 stays bit-exact, and temperatures stay at 48-57 °C with no throttle
[HW sweep]. With the datapath optimizations of [not-mac-bound.md](not-mac-bound.md)
§"Datatype-independent prefill optimizations" on, 600 MHz Gemma pp2048 is ~15 t/s. The
11.40 t/s of the table above predates them. The point is vendor-validated at vdd_npu 0.80 V:
the OPP table puts 300-700 MHz at 0.70 V, 900 MHz at 0.80 V, and 1 GHz at 0.85 V.

### Clock duty cycle under a bursty workload

The clock is at 600 MHz only during bursts, and a bursty prefill samples a median of 200 MHz.
The park to 200 MHz is in runtime suspend, so the clock is at 600 MHz only while the NPU is
held active between submits. A workload with idle gaps lets the domain runtime-suspend. Such
gaps are a few large `-ub 2048` micro-batches with host pack and readback between them, or
the pause between llama-bench repetitions. A 1 Hz sampler of the clock then measures a
maximum of 600 MHz and a median of 200 MHz. A continuously fed workload (many small
`-ub 512` micro-batches) stays at 600 MHz for the whole run [HW sweep].

The clock sampled during a benchmark is therefore a duty cycle. Its maximum is the operating
point, and its median shows how continuously the NPU was fed.

> [!NOTE]
> A colder or less continuously fed session measures low. Compare throughput only between
> runs of one session.

### 900 MHz

900 MHz gives no speedup over 600 MHz. The NPU `wait` stays at 61, and the matmul is bound
entirely by readback and dispatch. The board is also V/f-marginal at 900 MHz, because the
vendor's `set_read_margin` GRF tuning is not applied.

> [!CAUTION]
> With `power/control=on` pinned, an `rmmod` at 900 MHz hard-locks the board, and only a
> power cycle recovers it [HW sweep]. The same pinned `rmmod` at 600 MHz does not.

The cause is not isolated [hypothesis]. The pin defeats the suspend park at 200 MHz, so the
domain powers off and on again at 900 MHz, the condition the park exists to prevent. The
domain powers on at a raised 600 MHz without a hang, so the rate contributes as well, and at
900 MHz the `set_read_margin` tuning is missing. The rail sits at 0.80 V without the voltage
patch, the voltage the vendor OPP table gives 900 MHz.

### The `power/control` pin on a vendor kernel

> [!CAUTION]
> On a vendor 6.1.172 kernel with `rknpu` 0.9.8, one sequence hung the board once. It wrote
> `power/control=on` on the NPU, then set `devfreq/fdab0000.npu` min and max to 300 MHz. The
> board answered ping and nothing else until a power cycle. Which of the two writes causes the
> hang is not isolated [HW sweep, vendor RK1, 2026-09-27].

Pinning the clock through the same devfreq node alone has run without incident
([data/rknn-encoder/README.md](data/rknn-encoder/README.md)).

## Safe operating procedure

With the NPU idle, load the module at 600 MHz with **no `power/control` pin**:
`sudo rmmod rocket; sudo insmod rocket.ko rocket_npu_clk_hz=600000000`.
The clock rises to 600 MHz under load and parks at 200 MHz when idle, so every power cycle of
the domain is safe. Leave the NPU idle before an `rmmod` too. The pin is a measurement aid
only. Never use it for normal operation.

### Measurement discipline

> [!NOTE]
> The clock parks at 200 MHz when idle and rises under load. The first `-r 1` benchmark after
> an `rmmod`, an `insmod`, a reboot or an idle gap is cold and measures ~15% low (~2 t/s). A
> single cold sample after a reload therefore looks like a regression. Run it >=3x back to
> back, and compare only the warm runs (the 2nd and 3rd).

## The per-core park of a shared clock

The three NPU cores share one clock, but the clock patch without its refcount
(§"The refcounted park") parks it per core [HW sweep, source-confirmed]. The clock
`scmi_clk_npu` is a single SCMI clock (id 6) with `fdab0000.npu`, `fdac0000.npu` and
`fdad0000.npu` all as consumers: one rate row, three consumers in `clk_summary`. But
`rocket_device_runtime_suspend()` checks only `rocket_job_is_idle(&rdev->cores[core])`, which
tests this core alone. It then calls `clk_set_rate(npu_clk, ROCKET_NPU_POWER_DOWN_HZ)` on the
shared clock.

> [!CAUTION]
> Without the refcount, a core idle for its 50 ms `autosuspend_delay_ms` drops the shared
> clock to 200 MHz under the other two cores' running jobs.

With work spread across worker fds, the `librocketnpu` default, the cores go idle at different
times, and each sets the one shared rate independently. On a MoE int8 workload (one resident
GEMM per iteration, ~10 ms of host work between them), identical invocations alternate 15 ms
and 69 ms, a 4.8x swing that persists run to run. The sampled `scmi_clk_npu` rate flaps
200 ↔ 600 MHz. The swing is larger than the 3x the clock ratio implies, because a job loses
the rate mid-flight and also waits for the resume latency.

With a single worker fd, only one core cycles. Its own suspend parks the clock and its own
resume raises it, with no other core to interfere.

> [!CAUTION]
> While the clock flaps, a worker-count sweep and a group sweep both return non-monotonic
> results that are entirely an artifact of the flapping. Stop the flapping before a sweep.

### Raising `autosuspend_delay_ms`

Raise `autosuspend_delay_ms` past the longest NPU-idle gap inside the workload. A raised delay
is better than pinning `power/control` on every property in the table below:

```sh
for d in /sys/devices/platform/*.npu; do echo 2000 | sudo tee $d/power/autosuspend_delay_ms; done
...measure...
for d in /sys/devices/platform/*.npu; do echo 50   | sudo tee $d/power/autosuspend_delay_ms; done
```

Variance collapses from 4.8x to ±3%. The domain still parks: after the run it is
`runtime_status=suspended`, with the clock back at 200 MHz. The suspend path still runs, so
the clock is always parked before the domain powers down. The pin `power/control=on` removes
that guarantee, which is one of the candidate causes of the pin's hard-lock at 900 MHz
(§"900 MHz"). A raised delay is expected not to hard-lock the board [expected].

| | idles at 200? | flaps mid-run? | parks before power-down? | safe >600 MHz? |
|---|---|---|---|---|
| `auto`, 50 ms (default) | yes | **yes, 4.8x** | yes | not validated |
| `control=on` (the pin) | **no** | no | **no** | **no, hard-locks** |
| `auto`, raised delay | yes | no | yes | not validated |

The pin still works at 600 MHz and remains usable as a measurement aid. It gives nothing the
raised delay does not, and it is never safe above 600 MHz.

### Cost on a CPU-bound graph

On a CPU-bound graph the flapping costs little, so not every recorded NPU baseline is
depressed by it [HW sweep]:

| `autosuspend_delay_ms` | pp2048 | clock duty cycle (1 Hz sampler) |
|---|---:|---|
| 50 (default) | 10.90 t/s | 18% of samples at 600 MHz |
| 2000 | 11.05 t/s | 61% of samples at 600 MHz |

**Conditions:** gpt-oss `pp2048`, the default NPU placement (projections on the NPU, experts
on the CPU).

Tripling the time at 600 MHz raises throughput 1.4%. The graph is CPU-bound: the expert FFNs
are ~76% of the compute, and they run on the CPU. The NPU waits for most of the wall-clock
time, and idle time at 200 MHz costs nothing.

The two regimes differ:

- **NPU-dense, multi-worker** (a resident-weight matmul loop, the MoE int8 workload): the
  cross-core park is a 4.8x swing in the measurement. Software must stop it before comparing
  two configurations.
- **CPU-bound graph** (the default MoE placement, anything where the NPU mostly waits):
  stopping the park is worth ~1%.

Stopping the park is worth more as more work moves onto the NPU. Offloading the MoE experts,
for instance, turns the graph from CPU-bound to NPU-dense and multi-worker, the regime where
the per-core park costs the most.

### The refcounted park

With the refcounted park, only the last core to suspend parks the shared clock. The clock
patch, `081-rocket-drv-npu-clk.patch`, includes it. The counter `rocket_clk_users` counts
runtime-resumed cores, independently of `rocket_npu_clk_hz`. Changing the sysfs-writable
param while cores are active therefore cannot strand the count. A mutex serializes the
counter against a concurrent resume on another core. The decrement must come after the
`rocket_job_is_idle()` `-EBUSY` return, so that a busy core does not leak the count.

The voltage path stays per core by design, with no refcount. The rail falls only once every
core has lowered its own request, which is exactly what max-aggregation already gives. Tying
it to the last core would strand the earlier cores' votes high, and the rail would never
come down.

| | stock module | with the refcounted park |
|---|---|---|
| 6 repeats | 15.5 / **68.6** / 15.2 / **35.5** / 15.2 / **72.9** ms | 15.5 / 14.6 / 16.0 / 23.5 / 15.1 / 14.9 ms |
| clock through a long multi-core run | flaps 200 ↔ 600 | holds 600: exactly one 600->200 transition, the park at the end |
| parks when idle | yes | yes (invariant preserved: `suspended`, 200 MHz) |

**Conditions:** RK1, the stock 50 ms autosuspend delay, no pin, a resident-int8 matmul over 3
worker fds.

With the refcounted park applied, a multi-core workload does not need the
`autosuspend_delay_ms` workaround above. The workaround still suits a single-core workload
with long host-side gaps, where the park is legitimate and unwanted.

The refcount changes nothing above 600 MHz, and does not address the 900 MHz hard-lock. A
staggered multi-core resume already powers a domain on at a rate a sibling raised, so that
exposure exists without the refcount. The only validated operating point remains 600 MHz.

### The CPU governor

The CPU governor is a second frequency setting that affects NPU measurements, independently of
the NPU clock. The per-submit overhead is CPU-side: the submit `ioctl` plus the blocking wait
on the completion IRQ. On an idle board an `ondemand` or `interactive` governor therefore
under-clocks the A76 cores between submits, and inflates any submit-bound number. An external
RKNN-path measurement shows the effect with the NPU clock fixed
([not-mac-bound.md](not-mac-bound.md) §"Per-submit overhead reductions") [external,
proprietary path].

For a per-submit overhead measurement, pin the CPU cores to `performance` as well. A llama.cpp
prefill needs the pin too. At pp2048 on two `Q4_K_M` models, NPU prefill measures
1.079-1.130x faster pinned than under `ondemand` [HW sweep 2026-09-29]
([cpu-governor-and-offload.md](cpu-governor-and-offload.md) §"LLM prefill").

## Firmware (BL31): rate setting and the OTP ceiling

The secure firmware (BL31 / ATF) controls the NPU compute clock and exposes it to Linux over
SCMI. The kernel does not write it directly, as it does a normal CRU clock. Firmware control
of the clock is the mechanism behind Erratum E10, and it determines where the rate limit is
set.

### SCMI clock id 6

BL31 exposes the NPU clock as `scmi_clk_npu`, SCMI clock id `6` (table order: cpul 0, dsu 1,
cpub01 2, cpub23 3, ddr 4, gpu 5, npu 6, sbus 7). That matches the device tree's
`clocks = <... &scmi_clk SCMI_CLK_NPU ...>`. Linux reaches the clock via `arm-scmi` over the
SMC transport (boot log: `SCMI Protocol v2.0 'rockchip:'`)
[device tree SCMI_CLK_NPU, boot log].

### `clk_set_rate` in EL3

Setting the rate passes through the secure monitor. BL31 applies it via
`rockchip_opteed_clk_set_rate`, so EL3 programs the NPU PLL, not the kernel. Setting the rate
while the NPU power domain is off therefore hangs the firmware, which is Erratum E10. The
runtime-resume path avoids that hang [firmware behavior].

### The PVTPLL and its per-chip OTP bound

The rate is a PVTPLL, bounded by per-chip OTP, and the blob shows no static rate table. At
init BL31 adjusts the NPU PVTPLL against eFUSE/OTP values. The evidence is the format string
`adjust npu pvtpll by otp: min=%uM, max=%uM, length=%u` (same for cpul/cpub01/cpub23/gpu).
The BL31 SCMI clock descriptor contains pointer/ops arrays, not a list of allowed Hz. The
per-silicon NPU ceiling is therefore the OTP `max`, set at the factory, not a value in
firmware or DT [BL31 firmware, per-chip OTP].

Only a debug BL31 emits that OTP `min/max` line, to the secure UART. The release BL31 v1.51 on
the RK1 does not print it at normal verbosity, so it is absent from the U-Boot/Linux serial
log. To read this chip's ceiling, boot a debug BL31 or read the NPU PVTPLL OTP fields
directly.

### The upstream TF-A rate table

Upstream TF-A contains a rate table for this clock. In TF-A v2.12's `rk3588_clk.c`, 600 MHz
is a PVTPLL rate (ring 1, length 17), and only 200 MHz comes from the GPLL. A rate outside
its 9-entry table returns `SCMI_INVALID_PARAMETERS`, which Linux's `clk_change_rate` ignores.
On the PVTPLL path `clk_scmi_npu_get_rate` returns the last request
[source-confirmed: TF-A v2.12 `rk3588_clk.c`, read 2026-09-23].

The RK1 runs the rkbin v1.51 blob, of the same lineage. Whether the blob matches that source
is not verified. If it does, a kernel-log read-back of 600000000 and `clk_summary` report the
request and not the PLL. The witness is CRU `CLKSEL_CON74` bit 0. The absence of a static rate
table, above, is what the blob's descriptor shows, and is not established firmware behavior.

### Voltage and frequency in firmware

Firmware does not couple voltage to frequency. The BL31 SCMI clock path programs only the
PLL, with no regulator/voltage operations for the NPU clock. Linux alone manages `vdd_npu`
(RK8602 PMIC at i2c 0x42, range 0.55-0.95 V). Under the SCMI clock model, which is not an
SCMI perf-domain, nothing raises voltage when frequency rises. A raised rate without a
matching voltage is therefore V/f-marginal [firmware behavior, Android DT `vdd_npu_s0`].

### The rockchip SCMI clock quirk

Linux applies a rockchip clock quirk. The kernel enables
`quirk_clock_rates_triplet_out_of_spec` for this `'rockchip:'` SCMI firmware, which reports
clock rates as a non-standard min/max/step triplet [boot log].

### Clock handling in stock `rocket`

Stock upstream `rocket` has no clock handling at all. In android-mainline
`drivers/accel/rocket/rocket_drv.c`, runtime resume and suspend call only `clk_bulk_*`
(enable/disable). There is no `clk_set_rate`, no module param, and no devfreq/OPP. The
`rocket_npu_clk_hz` ramp comes entirely from the clock patch. An unpatched mainline kernel
leaves the NPU at 200 MHz regardless [android-mainline source, v7.1 source].

### The upstream DVFS series

An upstream series adds DVFS to `rocket` for the RK3588 NPU. It is Igor Paunovic's
"accel/rocket: DVFS for the RK3588 NPU", v2, patchwork series 1171089, 2026-09-22, 11 patches
on drm-misc-next. Patches 6-11 add:

- A binding with the `rk3588-opp.dtsi` table on all three cores under `opp-shared`
- A boot-rate restore with a `.shutdown` hook
- devfreq `simple_ondemand` at a 50 ms window
- A cooling device with an 85 C passive trip

The series sets the clock in the runtime-PM callbacks, so it replaces the clock and voltage
patches rather than combining with them.

As posted it changes the operating point in two ways [expected]. The rail at 600 MHz drops
from 800 mV to the table's 700 mV. The rate follows the governor, whose window is shorter than
the host gaps inside one Whisper encode.

**Not characterized:** the series on hardware.

## Voltage coupling for rates above 600 MHz

Software scales the rail with the clock. The voltage patch
(`patches/rocket/082-rocket-drv-npu-volt.patch`, companion to the clk patch) holds the
`vdd_npu_s0` regulator for the device lifetime. It scales the regulator with the clock from
the same runtime-PM hooks: voltage-up before clock-up on resume, and clock-down before
voltage-down on suspend. The target is a vendor f->V map (300-700->0.70, 800->0.75,
900->0.80, 1000->0.85 V), clamped up to a 0.80 V minimum.

The 0.80 V minimum is required. The rail already sits at 0.80 V, and the regulator framework
aggregates consumers by max. A vote at the minimum therefore holds the rail at its present
voltage at <=600 MHz, which changes nothing there. It also stops this vote from ever pulling
the shared rail down below 0.80 V. A `rocket_npu_uv` µV override exists for >600 MHz
bring-up.

The patch compiles in-tree on mainline kernel 7.1.0-1-arm64, and is validated at 600 MHz
[HW sweep 2026-06-22]. Activate it with
`sudo modprobe rocket rocket_npu_clk_hz=600000000` (no pin).

The coupling is fixed-rate rather than devfreq: it gives f/V safety, and no governor. Dynamic
scaling additionally needs only an OPP/devfreq table.

## Applicability

A rate above 600 MHz raises prefill throughput only where the clock, rather than readback and
dispatch, is the limit. At 900 MHz readback and dispatch remain the limit, and the speedup is
zero (§"900 MHz"). Given [not-mac-bound.md](not-mac-bound.md), fewer and bigger NPU jobs are
the larger prefill optimization, ahead of a higher clock.

900 MHz and 1 GHz need a config change plus a deliberate V/f-and-thermal test, and no new
code. The voltage coupling above supplies the prerequisite. Before any sweep above 600 MHz,
capture this chip's OTP PVTPLL `max` (debug BL31). Watch temperatures during it: the RK3588
(~15 W) has no auto-throttle.

## Evidence

### The 200 MHz boot rate

Under load, 60 samples across a full prefill were all 200 MHz [source-confirmed, HW sweep].

### Clock duty cycle under a bursty workload

The 1 Hz sampler is `rocket-userspace/tools/npu_bench_env.sh`. Sessions also differ in the
other direction: 0.8B F16 pp2048 measured 105 t/s in a cold-ish session against 89 in a warm
back-to-back sweep. The conditions of that pair are not recorded, so it supports comparing only
within one session, and not the direction of the difference.

### Measurement discipline

A single cold sample after a reload looked like a ~14% regression.

### 900 MHz

The `rmmod` hard-lock at 900 MHz under the pin was measured twice, each needing a power
cycle, on 2026-06-15, before the voltage patch.

### The `power/control` pin on a vendor kernel

One command made both writes, and the hang occurred once. The command's output stopped after
the first `power/control` read, so one of those two writes is the cause.

### The per-core park of a shared clock

The MoE int8 workload's swing persisted run to run, indefinitely. A worker-count sweep
identifies the cause: `W=1` is the one stable configuration (31.1 ms unpinned against
28.4 ms pinned), and every multi-worker configuration was wildly unstable.

A worker sweep and a group sweep both came back non-monotonic, and both were 100% artifact
of the flapping. They measured cleanly monotonic once the flapping was stopped.

### Cost on a CPU-bound graph

Each `autosuspend_delay_ms` setting ran 2 repetitions.

### Voltage coupling for rates above 600 MHz

Validated 2026-06-22, with all 4 tests passing at 600 MHz:

- Per-core `vdd->0.80 V`/`clk->600 MHz` in dmesg
- The rail pinned at 0.80 V
- The matmul bit-exact at 80-87 GFLOP/s warm
- The idle park returning clk->200 MHz at 0.80 V before a clean `rmmod`/reload

The 80-87 GFLOP/s is the fp16 `512×3840×4096` matmul on the library of 2026-06-22. The
75.3 GFLOP/s of the clock comparison at the top is the same shape on the library of 2026-06-15,
with host K-accumulation.
