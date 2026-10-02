# The NPU clock

The RK3588 NPU compute clock (`scmi_clk_npu`, shared by all 3 cores) **boots pinned at
200 MHz**, one fifth of the 1 GHz silicon max. Raising it is a ~1.43x win, and it is
dangerous to do the obvious ways. Both naive ways to raise it hang the box. The safe path
is below.

## The 200 MHz boot rate

Mainline `rocket` has no NPU devfreq. The driver never requests a higher OPP, so the SCMI
clock stays at its boot default. The device tree also pins it explicitly: the
`npu@fdab0000` nodes have `assigned-clock-rates = <200000000>` for the `npu` clock.

200 MHz is the vendor's `POWER_DOWN_FREQ`, the rate its driver sets at runtime suspend
[source-confirmed: BSP `rknpu_devfreq.c`]. A powered vendor NPU does not idle there. On a
vendor 6.1.172 kernel with `rknpu` 0.9.8, the clock reads 1000 MHz at idle. That is under the
`rknpu_ondemand` governor with OPPs from 300 to 1000 MHz [HW sweep, vendor RK1].

With no devfreq to ramp it back up, `rocket` leaves the clock parked at 200 MHz. Under load,
60 samples across a full prefill were all 200 MHz [source-confirmed, HW sweep].

The throttled clock inflates the NPU `wait` ~5x. At 200 MHz, ~150 GFLOP/s/core is ~73% of
the 200 MHz fp16 peak, which is a sane efficiency. Against 1 GHz it is an implausible ~15%.
The efficiency is against the throttled clock, not against a fast one. So the prefill
`wait` is not evidence that the NPU is compute-bound at full clock.

## Raising the clock on an unpowered domain

The NPU power domains (`PD_NPUTOP/NPU1/NPU2`) are off when the device is idle or cold
(`rocket` uses runtime PM). **Setting the SCMI NPU PLL on an unpowered domain wedges
the SCMI firmware (EL3).** Both obvious approaches do exactly that:

1. **A DT override** (`assigned-clock-rates = 600M`) hangs the boot. It is applied at
   `of_clk_set_defaults`, before the driver probes and powers the domain.
2. **A standalone out-of-tree `clk_set_rate` module** hangs the live box. It sets the rate
   at idle, with the domain off.

The vendor driver corroborates the rule. It sets the NPU clock only during operation, with
the domain powered, through devfreq. Its OPP helper refuses when `!pm_runtime_active`.

An upstream DVFS series for this part states the hazard more broadly: an island powered up
above the boot rate never acknowledges (patchwork series 1171089, v2, 2026-09-22). Under the
clock patch here, child islands power on at a raised 600 MHz routinely, with no failure
recorded. Two explanations fit [hypothesis]. The hazard is specific to the island that holds
the PVTPLL, or the v1.51 blob serves 600 MHz from a normal PLL. One read of `CLKSEL_CON74` at
600 MHz separates them.

## Setting the clock in `rocket`'s runtime resume

The only safe place to set the clock is inside the driver, after `pm_runtime` has powered
the domain. That is the `rocket-clk/` patch: a module param `rocket_npu_clk_hz`
(default 0 = stock), a `clk_set_rate` in `rocket_device_runtime_resume()` (domain
powered), and a park back to 200 MHz in `rocket_device_runtime_suspend()` (before the
domain powers off).

The patch builds as a module. The kernel image and the DTB are untouched, so the boot is
never at risk. Recovery is `rmmod` or a reboot. Probe calls `pm_runtime_resume_and_get`, so
a non-zero param is applied during `insmod` with the domain powered. An `insmod` that
returns cleanly is therefore itself the proof that the rate was not set cold. See the
`rocket-clk` project.

## The 600 MHz operating point

600 MHz is safe and coherent. The measurements:

- Standalone fp16 `512×3840×4096`: 56.5 -> 75.3 GFLOP/s (1.33x)
- Gemma pp2048, isolating the clock: 7.98 -> 11.40 t/s (1.43x)
- int8: still bit-exact
- Temperatures: 48-57 °C, no throttle

With the datapath levers on, 600 MHz Gemma pp2048 is ~15 t/s. The point is vendor-validated
at vdd_npu 0.80 V: the OPP table puts 300-700 MHz at 0.70 V, 900 MHz at 0.80 V, and 1 GHz at
0.85 V.

### The size of the gain

The gain is 1.43x and not ~4.5x. At 600 MHz the NPU `wait` collapses, and the other floors
take over: host readback and per-job dispatch, both clock-independent. As
[not-mac-bound.md](not-mac-bound.md) shows, the deeper floor is DMA/dispatch, not compute.
Raising the clock speeds only the shrinking `wait`.

### Clock duty cycle under a bursty workload

600 MHz is per-burst rather than pinned, and a bursty prefill samples a median of 200 MHz.
The park to 200 lives in runtime suspend, so the clock is 600 only while the NPU is actively
held between submits. A workload with idle gaps lets the domain runtime-suspend. Such gaps
are a few large `-ub 2048` ubatches with host pack/readback between them, or the pause
between llama-bench reps. A 1 Hz sampler (`rocket-userspace/tools/npu_bench_env.sh`) then
reads max 600 but median 200. A continuously fed workload (many small `-ub 512` ubatches)
stays pinned at 600 the whole run [HW sweep].

So the clock sampled during a bench is a duty cycle, not a constant. Read max as the
operating point and median as how continuously the NPU was fed. Compare t/s A/B only within
one session. A colder or less continuous session reads low: 0.8B F16 pp2048 was 105 t/s
cold-ish vs 89 in a warm back-to-back sweep.

### 900 MHz

900 MHz gives zero extra speedup over 600 MHz: `wait` stays 61, fully
readback/dispatch-bound. It is also V/f-marginal here (the vendor's `set_read_margin` GRF
tuning is unapplied).

**A pinned `rmmod` at 900 MHz hard-locks the box** (measured twice, each needing a
power cycle). The `power/control=on` measurement pin defeats the suspend park at 200, so the
domain cold-powers-on at 900 MHz. That is the exact hang condition the suspend park exists to
prevent.

### The `power/control` pin on a vendor kernel

A vendor kernel can wedge on the same pin. On a vendor 6.1.172 kernel with `rknpu` 0.9.8,
one command wrote `power/control=on` on the NPU and then set `devfreq/fdab0000.npu` min and
max to 300 MHz. The board answered ping and nothing else until a power cycle. The command's
output stopped after the first `power/control` read, so one of those two writes is the cause.
Which one is not isolated [HW sweep, vendor RK1, 2026-09-27, one occurrence]. Pinning the
clock through the same devfreq node alone has run without incident
([data/rknn-encoder/README.md](data/rknn-encoder/README.md)).

## Safe operating procedure

Load at 600 MHz with **no `power/control` pin**, and be idle first:
`sudo rmmod rocket; sudo insmod rocket.ko rocket_npu_clk_hz=600000000`.
The clock rides up to 600 MHz under load and auto-parks at 200 MHz when idle, so every
power cycle is safe. Be idle before `rmmod`. The pin is a measurement crutch. Never use it
for normal operation.

### Measurement discipline

The clock parks at 200 MHz when idle and ramps under load. So the first `-r 1` benchmark
after any `rmmod`, `insmod`, reboot or idle gap is cold and reads ~15% low (~2 t/s). Always
run >=3x back-to-back and compare the warm runs (2nd-3rd). Treat run 1 as a throwaway. A
single cold sample reads like a ~14% regression, which is a trap for anyone benchmarking
after a reload.

## The per-core park of a shared clock

The three NPU cores share one clock, but the clock patch parks it per core
[HW sweep, source-confirmed]. The clock `scmi_clk_npu` is a single SCMI clock (id 6) with
`fdab0000.npu`, `fdac0000.npu` and `fdad0000.npu` all as consumers: one rate row, three
consumers in `clk_summary`. But `rocket_device_runtime_suspend()` gates on
`rocket_job_is_idle(&rdev->cores[core])`, which is this core alone. It then calls
`clk_set_rate(npu_clk, ROCKET_NPU_POWER_DOWN_HZ)` on the shared clock.

Suppose one core goes idle for its 50 ms `autosuspend_delay_ms` while the other two are
still mid-job. **That core's suspend handler then drops the shared clock to 200 MHz under
the running jobs.** With work fanned across worker fds, the default everywhere in this
stack, the cores idle at different times and independently fight over one rate.

The effect was measured on the MoE int8 spike (one resident GEMM per iteration, ~10 ms of
host work between them). Identical invocations alternated 15 ms and 69 ms, a 4.8x swing, run
to run, indefinitely. Sampling `scmi_clk_npu` through the run showed the rate flapping
200 ↔ 600 MHz. That swing is larger than the 3x the clock ratio implies, because jobs get
the rate cut mid-flight and pay resume latency on top.

The worker-count sweep identifies the cause: `W=1` is the one stable configuration (31.1 ms
unpinned vs 28.4 ms pinned). With a single worker only one core cycles. Its own suspend
parks the clock and its own resume raises it, with no other core to interfere. Every
multi-worker config was wildly unstable.

The flapping does not look like noise. It looks like a finding. A worker sweep and a group
sweep both came back non-monotonic, and both were 100% artifact. They read cleanly monotonic
once the flapping was stopped.

### Raising `autosuspend_delay_ms`

Raise `autosuspend_delay_ms` past the longest NPU-idle gap inside the workload. This is the
right knob, and it is strictly better than pinning `power/control`:

```sh
for d in /sys/devices/platform/*.npu; do echo 2000 | sudo tee $d/power/autosuspend_delay_ms; done
...measure...
for d in /sys/devices/platform/*.npu; do echo 50   | sudo tee $d/power/autosuspend_delay_ms; done
```

Variance collapses from 4.8x to ±3%. The domain still parks, verified after the run: it
goes `runtime_status=suspended` with the clock back at 200 MHz. The suspend path still runs,
so the clock is always parked before the domain powers down. The pin `power/control=on`
destroys that invariant. That is why the pin hard-locks the box at 900 MHz (above) and a
raised delay would not.

| | idles at 200? | flaps mid-run? | parks before power-down? | safe >600 MHz? |
|---|---|---|---|---|
| `auto`, 50 ms (default) | yes | **yes, 4.8x** | yes | yes |
| `control=on` (the pin) | **no** | no | **no** | **no, hard-locks** |
| `auto`, raised delay | yes | no | yes | yes |

Prefer the raised delay. The pin still works at 600 MHz and remains a valid measurement
crutch. It buys nothing the delay does not, and it is never safe above 600.

### Cost on a CPU-bound graph

The flapping is dramatic, but the recorded NPU baselines are not all artificially depressed,
and the A/B says so [HW sweep, A/B]. It ran gpt-oss `pp2048`, NPU-default (projections on
the NPU, experts on the CPU), 2 reps:

| `autosuspend_delay_ms` | pp2048 | clock duty cycle (1 Hz sampler) |
|---|---:|---|
| 50 (default) | 10.90 t/s | 18% of samples at 600 MHz |
| 2000 | 11.05 t/s | 61% of samples at 600 MHz |

Tripling the time spent at 600 MHz bought 1.4%. The reason is that this graph is CPU-bound.
The expert FFNs are ~76% of the compute and they run on the CPU. So the NPU is waiting, not
computing, for most of the wall clock, and idle time at 200 MHz is free. Raising the clock
duty cycle of a mostly idle unit buys almost nothing.

So the two regimes differ, and conflating them wastes a day:

- **NPU-dense, multi-worker** (a resident-weight matmul loop, the MoE int8 spike): the
  cross-core park is a 4.8x swing in the measurement. It must be stopped before any A/B is
  trustworthy.
- **CPU-bound graph** (today's MoE default, anything where the NPU mostly waits): stopping
  the park is worth ~1%. It is not a hidden win. Do not go looking for one.

The corollary matters for planning: the clock fix gets more valuable as more work moves onto
the NPU, not less. Offloading the MoE experts, for instance, flips the graph from CPU-bound
to NPU-dense and multi-worker. That is the regime where the per-core park costs the most.

### The refcounted park

Only the last core down parks the shared clock [shipped: `081-rocket-drv-npu-clk.patch`].
The counter `rocket_clk_users` counts runtime-resumed cores, independently of
`rocket_npu_clk_hz`. So flipping the sysfs-writable param while cores are active cannot
strand the count. The counter sits under a mutex that serializes against a concurrent resume
on another core. The decrement sits after the `rocket_job_is_idle()` `-EBUSY` bail, or a
busy core would leak the count.

The voltage path stays per core by design, with no refcount. The rail falls only once every
core has lowered its own request, which is exactly what max-aggregation already gives. Gating
it on the last core would strand the earlier cores' votes high, and the rail would never
come down.

Measured on the RK1 with the stock 50 ms autosuspend and no pin (resident-int8 matmul, 3
worker fds):

| | stock module | with the refcounted park |
|---|---|---|
| 6 repeats | 15.5 / **68.6** / 15.2 / **35.5** / 15.2 / **72.9** ms | 15.5 / 14.6 / 16.0 / 23.5 / 15.1 / 14.9 ms |
| clock through a long multi-core run | flaps 200 ↔ 600 | holds 600: exactly one 600->200 transition, the park at the end |
| parks when idle | yes | yes (invariant preserved: `suspended`, 200 MHz) |

With the refcounted park applied, a multi-core workload does not need the
`autosuspend_delay_ms` workaround above. The workaround is still the right tool for a
single-core harness with long host-side gaps, where the park is legitimate and unwanted.

The refcount changes nothing above 600 MHz. It does not revisit the 900 MHz hard-lock. A
staggered multi-core resume already powers a domain on at a rate a sibling raised, so that
exposure exists without the refcount. The only validated operating point remains 600 MHz.

### The CPU governor

The CPU governor is a second, independent frequency confounder. The per-submit/dispatch
overhead is CPU-side: the submit `ioctl` plus the blocking wait on the completion IRQ. So on
an idle box an `ondemand` or `interactive` governor under-clocks the A76 cores between
submits and inflates any submit-bound number. An external RKNN-path writeup measured a
single `rknn_run` swing of −41% (59 -> 35 ms) from the CPU governor alone, with the NPU
clock fixed. Pinning the NPU governor alone did nothing [external, proprietary path].

So for a dispatch-floor measurement, pin the CPU cores to `performance` as well. For a
prefill throughput measurement (a few large jobs, dominated by NPU `wait`), the CPU governor
matters far less. See [not-mac-bound.md](not-mac-bound.md) §Dispatch-floor reducers.

## Firmware (BL31): rate setting and the OTP ceiling

The secure firmware (BL31 / ATF) owns the NPU compute clock and exposes it to Linux over
SCMI. The kernel does not write it directly, as it does a normal CRU clock. That ownership
is the mechanism behind the power-domain hazard above, and it shows where the true rate
limit is set.

### SCMI clock id 6

BL31 exposes the NPU clock as `scmi_clk_npu`, SCMI clock id `6` (table order: cpul 0, dsu 1,
cpub01 2, cpub23 3, ddr 4, gpu 5, npu 6, sbus 7). That matches the device tree's
`clocks = <... &scmi_clk SCMI_CLK_NPU ...>`. Linux reaches the clock via `arm-scmi` over the
SMC transport (boot log: `SCMI Protocol v2.0 'rockchip:'`)
[device tree SCMI_CLK_NPU, boot log].

### `clk_set_rate` in EL3

Setting the rate routes through the secure monitor. BL31 applies it via
`rockchip_opteed_clk_set_rate`, so EL3 programs the NPU PLL, not the kernel. Setting the rate
while the NPU power domain is off therefore wedges the firmware. This is the EL3 hang that
the runtime-resume path avoids [firmware behavior].

### The PVTPLL and its per-chip OTP bound

The rate is a PVTPLL, bounded by per-chip OTP, and the blob shows no static rate table. At
init BL31 adjusts the NPU PVTPLL against eFUSE/OTP values. The evidence is the format string
`adjust npu pvtpll by otp: min=%uM, max=%uM, length=%u` (same for cpul/cpub01/cpub23/gpu).
The BL31 SCMI clock descriptor contains pointer/ops arrays, not a list of allowed Hz. So the
per-silicon NPU ceiling is the OTP `max`, set at the factory, not a value baked into
firmware or DT [BL31 firmware, per-chip OTP].

Only a debug BL31 emits that OTP `min/max` line, to the secure UART. The release BL31 v1.51
shipped here does not print it at normal verbosity, so it is absent from the U-Boot/Linux
serial log. To read this chip's ceiling, boot a debug BL31 or read the NPU PVTPLL OTP fields
directly.

### The upstream TF-A rate table

Upstream TF-A carries a rate table for this clock. In TF-A v2.12's `rk3588_clk.c`, 600 MHz
is a PVTPLL rate (ring 1, length 17), and only 200 MHz comes from the GPLL. A rate outside
its 9-entry table returns `SCMI_INVALID_PARAMETERS`, which Linux's `clk_change_rate` ignores.
On the PVTPLL path `clk_scmi_npu_get_rate` returns the last request
[source-confirmed: TF-A v2.12 `rk3588_clk.c`, read 2026-09-23].

The board here runs the rkbin v1.51 blob, of the same lineage. Whether the blob matches that
source is not verified. If it does, a kernel-log read-back of 600000000 and `clk_summary`
report the request and not the PLL. The witness is CRU `CLKSEL_CON74` bit 0. So read "no
table" above as what the blob's descriptor shows, and not as established firmware behavior.

### Voltage and frequency in firmware

Firmware does not couple voltage to frequency. The BL31 SCMI clock path programs only the
PLL, with no regulator/voltage operations for the NPU clock. Linux alone manages `vdd_npu`
(RK8602 PMIC at i2c 0x42, range 0.55-0.95 V). Under the SCMI clock model, which is not an
SCMI perf-domain, nothing raises voltage when frequency rises. This is why a raised rate
without a matching voltage is V/f-marginal (the 900 MHz hard-lock above)
[firmware behavior, Android DT `vdd_npu_s0`].

### The rockchip SCMI clock quirk

Linux applies a rockchip clock quirk. The kernel enables
`quirk_clock_rates_triplet_out_of_spec` for this `'rockchip:'` SCMI firmware, which reports
clock rates as a non-standard min/max/step triplet [boot log].

### Clock handling in stock `rocket`

Stock upstream `rocket` has no clock handling at all. In android-mainline
`drivers/accel/rocket/rocket_drv.c`, runtime resume and suspend call only `clk_bulk_*`
(enable/disable). There is no `clk_set_rate`, no module param, and no devfreq/OPP. The
`rocket_npu_clk_hz` ramp is entirely the local `rocket-clk` patch. An unpatched mainline
kernel leaves the NPU at 200 MHz regardless [android-mainline source, local v7.1 tree].

## Voltage coupling for rates above 600 MHz

Software scales the rail with the clock. The voltage patch
(`patches/rocket/082-rocket-drv-npu-volt.patch`, companion to the clk patch) holds the
`vdd_npu_s0` regulator for the device lifetime. It scales the regulator with the clock from
the same runtime-PM hooks: voltage-up before clock-up on resume, and clock-down before
voltage-down on suspend. The target is a vendor f->V map (300-700->0.70, 800->0.75,
900->0.80, 1000->0.85 V), clamped up to a 0.80 V floor.

The floor is load-bearing. The rail already sits at 0.80 V, and the regulator framework
aggregates consumers by max. Voting the floor therefore pins the rail at today's voltage at
<=600 MHz (non-disruptive). It also stops this vote from ever pulling the shared rail down
below 0.80 V. A `rocket_npu_uv` µV override exists for >600 MHz bring-up.

The patch compiles in-tree on mainline kernel 7.1.0-1-arm64. It was validated 2026-06-22,
with all 4 gates passing at 600 MHz (per-core `vdd->0.80 V`/`clk->600 MHz` dmesg, rail
pinned at 0.80 V, matmul bit-exact 80-87 GFLOP/s warm, idle parks clk->200 MHz at 0.80 V,
clean `rmmod`/reload). Activate it with `sudo modprobe rocket rocket_npu_clk_hz=600000000`
(no pin).

This is a fixed-rate coupling, not devfreq. It gives f/V safety, not a governor. If dynamic
scaling is ever wanted, the remaining gap is only an OPP/devfreq table.

## Remaining clock headroom

900 MHz and 1 GHz are a config change plus a deliberate V/f-and-thermal test, not new code.
The voltage coupling above supplies the prerequisite. They are worth revisiting only after
confirming that the dispatch/readback floor, not the clock, is what is left. It currently
is: at 900 MHz the speedup is zero. The vendor's `set_read_margin` GRF tuning is unapplied.
Given [not-mac-bound.md](not-mac-bound.md), the bigger prefill lever is fewer/bigger NPU
jobs, not more MHz.

Before any sweep, capture this chip's OTP PVTPLL `max` (debug BL31). Watch temps during it
(RK3588 ~15 W, no auto-throttle).
