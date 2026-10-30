# The NPU clock

The RK3588 NPU compute clock, `scmi_clk_npu`, is shared by all 3 cores. It boots at 198 MHz,
the 200 MHz rate that firmware serves from the GPLL [source-confirmed, HW sweep]. Every higher
rate comes from a PVTPLL, a ring oscillator whose frequency follows the rail voltage. A
requested rate is therefore not the clock frequency (§"The requested rate and the physical
clock").

At the 0.80 V rail, the 600 MHz request runs at ~816 MHz and gives ~1.43x on prefill over the
boot clock [HW sweep]. The operating point is the 900 MHz request, which runs at 928 MHz at
that rail (§"The operating point"). Erratum E10 (§"Raising the clock on an unpowered domain")
restricts where software can set the rate.

**Conditions:** RK3588 (Turing RK1), mainline `rocket` with the clock patch, NPU clock 198 MHz
against ~816 MHz (the 600 MHz request at a 0.80 V rail), 2026-06-15, with the matmul's
K-accumulation on the host.

**Not characterized:** a clock above 1019 MHz, which needs a rail above the vendor's maximum,
and this chip's OTP PVTPLL ceiling. Sustained operation above 928 MHz, or with the NPU above
49 °C, is not characterized either (§"Applicability").

| Workload | 198 MHz | ~816 MHz | Speedup |
|---|---:|---:|---:|
| Standalone fp16 `512×3840×4096` | 56.5 GFLOP/s | 75.3 GFLOP/s | 1.33x |
| Gemma pp2048, the clock changed alone | 7.98 t/s | 11.40 t/s | 1.43x |

## The requested rate and the physical clock

Above 200 MHz the NPU clock is `clk_npu_pvtpll`, the PVTPLL in the NPU GRF at `0xfd5a2000`
[TRM, source-confirmed: TF-A v2.12 `rk3588_clk.c`, HW sweep]. `CRU_CLKSEL_CON74` bit 0
(`clk_rknn_dsu0_sel`) selects it, and 0 selects the normal-PLL path that `CRU_CLKSEL_CON73`
divides [TRM]. For each requested rate, BL31 programs one ring
length into `NPU_GRF_PVTPLL_CON_H` (`ring_length_sel`) [source-confirmed: TF-A v2.12
`rk3588_clk.c`, HW sweep]. A shorter ring runs faster, and any ring runs faster at a higher
rail voltage. The firmware's table pairs each length with the vendor's voltage for that rate.
A request therefore names its frequency only at that voltage [HW sweep 2026-10-19].

| Request | Ring length | Vendor OPP voltage | Clock at the OPP voltage | Clock at 0.80 V |
|---|---:|---:|---:|---:|
| 200 MHz | GPLL / 6 | n/a | 198 MHz | 198 MHz |
| 300 MHz | 55 | 0.70 V | 331 MHz | 427 MHz |
| 400 MHz | 38 | 0.70 V | Not characterized | 543 MHz |
| 500 MHz | 25 | 0.70 V | 536 MHz | 689 MHz |
| 600 MHz | 17 | 0.70 V | 638 MHz | 818 MHz |
| 700 MHz | 13 | 0.70 V | 705 MHz | 903 MHz |
| 800 MHz | 12 | 0.75 V | 830 MHz | 928 MHz |
| 900 MHz | 12 | 0.80 V | 928 MHz | 928 MHz |
| 1000 MHz | 12 | 0.85 V | 1019 MHz | Not characterized |

At ring length 17, the 600 MHz request, the clock follows the rail:

| Rail | 0.70 V | 0.75 V | 0.80 V | 0.85 V |
|---|---:|---:|---:|---:|
| Clock | 638 MHz | 732 MHz | 818 MHz | 898 MHz |

**Conditions:** Turing RK1, kernel 7.2.9-1-arm64, `rocket` 1.3.0 with the clock and voltage
patches, rkbin BL31 v1.51, 2026-10-19. All three cores are runtime-active and idle, at ~37 °C.
The rail is set through the voltage patch's `rocket_npu_uv`. The fp16 `512×3840×4096` matmul is
bit-exact at every point in both tables.

At its OPP voltages this chip runs 1-10% above each nominal rate. The 800, 900 and 1000 MHz
requests share the shortest ring, so above 928 MHz at 0.80 V only the rail raises the clock.
A request above the rail's rate does not overclock the NPU: at 0.80 V the 900 MHz request runs
at 928 MHz, the vendor's own 900 MHz point.

### The operating point

The clock patch requests 900 MHz, and the voltage patch holds the rail at its 0.80 V boot
voltage (§"Voltage coupling"). The NPU therefore runs at 928 MHz under load. Ring length 12 at
0.80 V is the vendor's 900 MHz operating point and the upstream DVFS series' one
(§"The upstream DVFS series") [source-confirmed: BSP `rk3588s.dtsi`, HW sweep 2026-10-19].

Over 15 minutes of continuous convolution and matmul work, the clock measures 922-928 MHz. The
NPU warms from 40 to 49 °C and levels at 47-48 °C. Every matmul is bit-exact
[HW sweep 2026-10-19]. The `rocket-userspace` test suite passes in full at this request, with
every core parking and resuming at the raised clock between tests [HW sweep 2026-10-19].

**Conditions:** Turing RK1, kernel 7.2.9-1-arm64, `rocket` 1.3.0 with the clock and voltage
patches, `RM` 4 (reset), governor `ondemand`, from ~37 °C idle, 2026-10-19.

> [!NOTE]
> Performance figures in these notes state their NPU clock in their Conditions blocks. Most
> were measured at ~816 MHz, the 600 MHz request at the 0.80 V rail.

### Reading the physical clock

> [!NOTE]
> On the PVTPLL path, BL31's `get_rate` returns the last requested rate
> [source-confirmed: TF-A v2.12 `rk3588_clk.c`]. `clk_get_rate()`, `clk_summary` and a driver's
> read-back log all report the request, not the frequency.

NPU GRF words `0x4c` and `0x50` contain the PVTPLL monitor's count, and the TRM does not list
them. BL31 programs the count window through `NPU_GRF_PVTPLL_CON1` (`cal_cnt`) as 24 cycles of
the 24 MHz reference, 1 µs. The count is therefore the frequency in MHz [HW sweep 2026-10-19].
Matmul device time across the GPLL and PVTPLL paths confirms the unit (§"Evidence").

> [!CAUTION]
> The PVTPLL sits inside the NPU power domains [source-confirmed: the upstream DVFS series,
> §"The upstream DVFS series"]. Software must read the NPU GRF only while every NPU core is
> runtime-active. BL31's writes to it with the domain off hang EL3 (Erratum E10), and a read
> is expected to fault the same way [expected].

`CRU_CLKSEL_CON74` sits in the always-on CRU at `0xfd7c0428` and is readable at any time. It
reads 1 in bit 0 while the NPU runs above 200 MHz, and 0 when the clock is parked [HW sweep].

`rocket-userspace`'s `tools/npu_clock` reads both: the path at any time, and the count only
while every core is held on. Its `tools/npu_bench_env.sh` reports the physical clock after a
benchmark run.

The SRAM read-margin field `RM`, bits [4:2] of `NPU_GRF_MEM_CON0`, `MEM_CON1` and `MEM_CON2`,
resets to 4 [TRM]. `rocket` never writes it, and it reads 4 at every rate from 198 to
1019 MHz [live 2026-10-19]. The vendor driver programs it from the rail voltage at each runtime
resume [source-confirmed: BSP `rknpu_devfreq.c`]. Its table assigns 4 to no NPU operating point
[source-confirmed: BSP `rk3588s.dtsi`]:

| Rail | `RM` |
|---|---:|
| 855 mV and above | 1 |
| 765 mV to below 855 mV | 2 |
| 675 mV to below 765 mV | 3 |
| 495 mV to below 675 mV | 4 |

The vendor driver lowers `RM` only after it raises the rail, and raises `RM` before it lowers
the rail. It moves `RM` to or from 4 only with the clock at or below 500 MHz. When the clock is
higher, the driver first lowers it to 500 MHz [source-confirmed: BSP `rockchip_opp_select.c`]. At
the 0.80 V rail the vendor value is 2.

**Not characterized:** `RM` 2 under `rocket`, and whether the vendor's 500 MHz rule protects
the change itself or bounds the clock at `RM` 4.

## Cause

### The 200 MHz boot rate

Mainline `rocket` has no NPU devfreq. The driver never requests a higher OPP, so the SCMI
clock stays at its boot default. The device tree also sets it explicitly: the
`npu@fdab0000` nodes have `assigned-clock-rates = <200000000>` for the `npu` clock. BL31 serves
that request as the GPLL divided by 6, 198 MHz [source-confirmed: TF-A v2.12, HW sweep].

200 MHz is the vendor's `POWER_DOWN_FREQ`, the rate its driver sets at runtime suspend
[source-confirmed: BSP `rknpu_devfreq.c`]. A powered vendor NPU does not idle there. On a
vendor 6.1.172 kernel with `rknpu` 0.9.8, under the `rknpu_ondemand` governor, the clock runs
at its 1000 MHz OPP at idle. The governor's OPPs span 300 to 1000 MHz [HW sweep, vendor RK1].

With no devfreq to raise it, `rocket` leaves the clock at 198 MHz. Under load it stays there
through a full prefill [source-confirmed, HW sweep].

The boot clock inflates the NPU `wait` ~5x. At 198 MHz, ~150 GFLOP/s per core is ~73% of the
fp16 peak at that clock, a plausible efficiency. Against the 1 GHz peak it is an implausible
~15%.
The prefill `wait` at the boot clock therefore reflects the clock, and does not show that the
NPU is compute-bound at full clock.

### The size of the gain

Prefill gains 1.43x for a 4.1x clock step, 198 to ~816 MHz. At ~816 MHz the NPU `wait`
collapses, and the other limits take over: host readback and per-job dispatch, both
independent of the NPU clock. The matmul's underlying limit is DMA and dispatch rather than
compute ([not-mac-bound.md](not-mac-bound.md)). Raising the clock speeds up only the `wait`,
which shrinks with each step.

## Raising the clock on an unpowered domain

The NPU power domains are off when the device is idle or cold, because `rocket` uses runtime
PM. They nest: `npu` contains `nputop`, which holds core 0 and contains `npu1` and `npu2`, one
for each other core [live: `pm_genpd_summary`, 2026-10-19].

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
clock patch, the child islands `npu1` and `npu2` power on while a sibling holds the raised
PVTPLL clock. That transition is bit-exact and hang-free at every request from 300 to
1000 MHz [HW sweep 2026-10-19]. Under the refcounted park (§"The refcounted park"), the
parent islands power on only from the parked 198 MHz.

**Not characterized:** a parent island powering on with the PVTPLL selected.

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

The param is writable through sysfs, and the next runtime resume applies it. Software must
change it only while every core is runtime-suspended, so that no core runs across the change.

## Operation at the raised clock

The 600 MHz request at 0.80 V (~816 MHz) is safe: int8 stays bit-exact, and temperatures stay
at 48-57 °C with no throttle [HW sweep]. With the datapath optimizations of
[not-mac-bound.md](not-mac-bound.md) §"Datatype-independent prefill optimizations" on, Gemma
pp2048 at ~816 MHz is ~15 t/s. The 11.40 t/s of the table above predates them.

### Clock duty cycle under a bursty workload

The clock is raised only during bursts, and a bursty prefill samples a median at the boot
rate. The park is in runtime suspend, so the clock stays raised only while the NPU is held
active between submits. A workload with idle gaps lets the domain runtime-suspend. Such gaps
are a few large `-ub 2048` micro-batches with host pack and readback between them, or the
pause between llama-bench repetitions. A 1 Hz sampler of the requested rate then measures the
request as its maximum and 200 MHz as its median. A continuously fed workload (many small `-ub 512`
micro-batches) stays at the raised rate for the whole run [HW sweep].

The rate sampled during a benchmark is therefore a duty cycle. Its maximum is the operating
point, and its median shows how continuously the NPU was fed.

> [!NOTE]
> A colder or less continuously fed session measures low. Compare throughput only between
> runs of one session.

### Rates from ~816 to 1019 MHz

Every request from 300 to 1000 MHz is bit-exact on the fp16 matmul, including requests whose
child-island power-ons happen at the raised clock [HW sweep 2026-10-19]. Steady-state matmul
device time falls 1.044x from ~816 to 928 MHz, and 1.083x to 1019 MHz:

| Request (rail) | Clock | Steady device time | Against ~816 MHz |
|---|---:|---:|---:|
| 600 MHz (0.80 V) | ~816 MHz | 25.07 ms | 1.00x |
| 800 MHz (0.80 V) | 928 MHz | 24.02 ms | 1.044x |
| 1000 MHz (0.85 V) | 1019 MHz | 23.15 ms | 1.083x |

**Conditions:** RK1, fp16 `512×3840×4096` through `rocket_matmul_fp16` on two cores, CPU
governor `performance`, 2026-10-19. Device time is the scheduler's job run-to-done interval,
excluding each core's first job after a resume.

The device time fits 7963/f + 15.3 ms, with f in MHz. At ~816 MHz, ~39% of it scales with the
clock [host-computed]. At 0.85 V each core's first job after a resume takes ~2.4 ms longer, the
rail's ramp from 0.80 V [HW sweep 2026-10-19].

> [!CAUTION]
> With `power/control=on` pinned, an `rmmod` at the 900 MHz request (928 MHz) hard-locks the
> board, and only a power cycle recovers it [HW sweep]. The same pinned `rmmod` at the 600 MHz
> request (~816 MHz) does not. Software must unload `rocket` only while every core's
> `power/runtime_status` reads `suspended`.

The pin defeats the suspend park at 200 MHz, so the parent islands power off with the PVTPLL
selected, the condition the park exists to prevent. Both rates come from the PVTPLL at the
same 0.80 V rail, and differ in ring length, 12 against 17. The cause is not isolated
[hypothesis]. A core can also stay runtime-active with no pin, after a job whose submit fails
part-way ([chips/rk3576-regcmd.md](../chips/rk3576-regcmd.md) §"The half-started job runtime-PM
pin"). An `rmmod` then meets the same condition [expected].

### The `power/control` pin on a vendor kernel

> [!CAUTION]
> On a vendor 6.1.172 kernel with `rknpu` 0.9.8, one sequence hung the board once. It wrote
> `power/control=on` on the NPU, then set `devfreq/fdab0000.npu` min and max to 300 MHz. The
> board answered ping and nothing else until a power cycle. Which of the two writes causes the
> hang is not isolated [HW sweep, vendor RK1, 2026-09-27].

Pinning the clock through the same devfreq node alone has run without incident
([data/rknn-encoder/README.md](data/rknn-encoder/README.md)).

## Safe operating procedure

With the NPU idle, load the module with **no `power/control` pin**:
`sudo rmmod rocket; sudo insmod rocket.ko rocket_npu_clk_hz=900000000`.
The clock rises to 928 MHz under load and parks at 198 MHz when idle, so every power cycle of
the domain is safe. To change the rate without a reload, write the sysfs param while every
core is runtime-suspended. The pin is a measurement aid only. Never use it for normal
operation.

Before an `rmmod`, confirm that all three cores are suspended:

```sh
cat /sys/devices/platform/fd{ab,ac,ad}0000.npu/power/runtime_status   # suspended, three times
```

If a core stays `active`, reboot instead of unloading the module.

### Measurement discipline

> [!NOTE]
> The clock parks at 198 MHz when idle and rises under load. The first `-r 1` benchmark after
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
> clock to 198 MHz under the other two cores' running jobs.

With work spread across worker fds, the `librocketnpu` default, the cores go idle at different
times, and each sets the one shared rate independently. On a MoE int8 workload (one resident
GEMM per iteration, ~10 ms of host work between them), identical invocations alternate 15 ms
and 69 ms, a 4.8x swing that persists run to run. The sampled `scmi_clk_npu` request flaps
200 ↔ 600 MHz. The swing is larger than the 4.1x clock ratio (198 against ~816 MHz), because a
job loses the rate mid-flight and also waits for the resume latency.

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
`runtime_status=suspended`, with the clock back at 198 MHz. The suspend path still runs, so
the clock is always parked before the domain powers down. The pin `power/control=on` removes
that guarantee, which is one of the candidate causes of the pin's hard-lock at the 900 MHz
request (§"Rates from ~816 to 1019 MHz").

| | idles at 198 MHz? | flaps mid-run? | parks before power-down? | safe above ~816 MHz? |
|---|---|---|---|---|
| `auto`, 50 ms (default) | yes | **yes, 4.8x** | yes | not validated |
| `control=on` (the pin) | **no** | no | **no** | **no, hard-locks** |
| `auto`, raised delay | yes | no | yes | yes, to 1019 MHz |

**Conditions:** the last column's raised delay is the clock patch's 1000 ms
`rocket_autosuspend_ms`, with the refcounted park, 2026-10-19.

The pin still works at the 600 MHz request and remains usable as a measurement aid. It gives
nothing the raised delay does not, and it is never safe above ~816 MHz.

### Cost on a CPU-bound graph

On a CPU-bound graph the flapping costs little, so not every recorded NPU baseline is
depressed by it [HW sweep]:

| `autosuspend_delay_ms` | pp2048 | clock duty cycle (1 Hz sampler) |
|---|---:|---|
| 50 (default) | 10.90 t/s | 18% of samples at the 600 MHz request |
| 2000 | 11.05 t/s | 61% of samples at the 600 MHz request |

**Conditions:** gpt-oss `pp2048`, the default NPU placement (projections on the NPU, experts
on the CPU), NPU clock ~816 MHz when raised.

Tripling the time at the raised clock raises throughput 1.4%. The graph is CPU-bound: the
expert FFNs are ~76% of the compute, and they run on the CPU. The NPU waits for most of the
wall-clock time, and idle time at the boot clock costs nothing.

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
| requested rate through a long multi-core run | flaps 200 ↔ 600 | holds 600: exactly one 600->200 transition, the park at the end |
| parks when idle | yes | yes (invariant preserved: `suspended`, 198 MHz) |

**Conditions:** RK1, the stock 50 ms autosuspend delay, no pin, a resident-int8 matmul over 3
worker fds, NPU clock ~816 MHz when raised.

With the refcounted park applied, a multi-core workload does not need the
`autosuspend_delay_ms` workaround above. The workaround still suits a single-core workload
with long host-side gaps, where the park is legitimate and unwanted.

The refcount does not address the pinned-`rmmod` hard-lock. A staggered multi-core resume
powers a child island on at a rate a sibling raised, with or without the refcount. The
child-island transition is safe to 1019 MHz (§"Raising the clock on an unpowered domain").

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

### The rate table

BL31 serves the NPU clock from a 9-entry table. Its 200 MHz entry is the GPLL divided by 6.
Every other entry is a PVTPLL ring, at the lengths in §"The requested rate and the physical
clock" [source-confirmed: TF-A v2.12 `rk3588_clk.c`]. The rkbin v1.51 blob on the RK1
programs the same lengths, read back from `NPU_GRF_PVTPLL_CON_H` [HW sweep 2026-10-19]. A rate
outside the table returns `SCMI_INVALID_PARAMETERS`, which Linux's `clk_change_rate` ignores
[source-confirmed: TF-A v2.12].

### The per-chip OTP bound

At init BL31 adjusts the NPU PVTPLL against eFUSE/OTP values. The evidence is the format
string `adjust npu pvtpll by otp: min=%uM, max=%uM, length=%u` (same for
cpul/cpub01/cpub23/gpu). The per-silicon NPU ceiling is therefore the OTP `max`, set at the
factory [BL31 firmware, per-chip OTP].

Only a debug BL31 emits that OTP `min/max` line, to the secure UART. The release BL31 v1.51 on
the RK1 does not print it at normal verbosity, so it is absent from the U-Boot/Linux serial
log. To read this chip's ceiling, boot a debug BL31 or read the NPU PVTPLL OTP fields
directly.

### Voltage and frequency in firmware

Firmware does not couple voltage to frequency. The BL31 SCMI clock path programs only the
PLL, with no regulator/voltage operations for the NPU clock. Linux alone manages `vdd_npu`
(RK8602 PMIC at i2c 0x42, range 0.55-0.95 V) [firmware behavior, Android DT `vdd_npu_s0`].
The PVTPLL couples them the other way: at a fixed ring length, the clock follows the rail
(§"The requested rate and the physical clock").

### The rockchip SCMI clock quirk

Linux applies a rockchip clock quirk. The kernel enables
`quirk_clock_rates_triplet_out_of_spec` for this `'rockchip:'` SCMI firmware, which reports
clock rates as a non-standard min/max/step triplet [boot log].

### Clock handling in stock `rocket`

Stock upstream `rocket` has no clock handling at all. In android-mainline
`drivers/accel/rocket/rocket_drv.c`, runtime resume and suspend call only `clk_bulk_*`
(enable/disable). There is no `clk_set_rate`, no module param, and no devfreq/OPP. The
`rocket_npu_clk_hz` ramp comes entirely from the clock patch. An unpatched mainline kernel
leaves the NPU at 198 MHz regardless [android-mainline source, v7.1 source].

### The upstream DVFS series

An upstream series adds DVFS to `rocket` for the RK3588 NPU. It is Igor Paunovic's
"accel/rocket: DVFS for the RK3588 NPU", v2, patchwork series 1171089, 2026-09-22, 11 patches
on drm-misc-next. Patches 6-11 add:

- A binding with the `rk3588-opp.dtsi` table on all three cores under `opp-shared`
- A boot-rate restore with a `.shutdown` hook
- devfreq `simple_ondemand` at a 50 ms window
- A cooling device with an 85 C passive trip

The series sets the clock in the runtime-PM callbacks, so it replaces the clock and voltage
patches rather than combining with them. Above the boot rate it holds every core
runtime-resumed, so no island powers on or off while the clock is raised [source-confirmed:
the series' `rocket_devfreq.c`].

Its table pairs every rate with the vendor's voltage, 900 MHz at 0.80 V among them
[source-confirmed: the series' `rk3588-opp.dtsi` table]. A rate pinned through the series
therefore runs at the clock that pairing measures on this chip [HW sweep 2026-10-19]. The 900 MHz
OPP runs at 928 MHz, and the 600 MHz OPP at 638 MHz. The series parks the clock at the boot rate when
the last core suspends and in a `.shutdown` hook. It does not program `RM` [source-confirmed: the
series' patches 08 and 09]. The rate follows the governor, whose window is shorter than the host
gaps inside one Whisper encode [expected].

**Not characterized:** the series on hardware.

## Voltage coupling

Software scales the rail with the clock. The voltage patch
(`patches/rocket/082-rocket-drv-npu-volt.patch`, companion to the clk patch) holds the
`vdd_npu_s0` regulator for the device lifetime. It scales the regulator with the clock from
the same runtime-PM hooks: voltage-up before clock-up on resume, and clock-down before
voltage-down on suspend. The target is a vendor f->V map (300-700->0.70, 800->0.75,
900->0.80, 1000->0.85 V), clamped up to a 0.80 V minimum.

The 0.80 V minimum is required. The rail boots at 0.80 V, and the regulator framework
aggregates consumers by max. A vote at the minimum therefore holds the rail at its boot
voltage for requests up to 900 MHz. It also stops this vote from ever pulling the shared rail
below the minimum.

The minimum also sets the clock. With the rail held at 0.80 V, the 600 MHz request runs at
~816 MHz. A `rocket_npu_uv` µV override sets the rail directly.

The patch compiles in-tree on mainline kernel 7.1.0-1-arm64. It is validated at the 600 MHz
request [HW sweep 2026-06-22], and at every request from 300 to 1000 MHz
[HW sweep 2026-10-19]. Activate it with `sudo modprobe rocket rocket_npu_clk_hz=900000000`
(no pin).

The coupling is fixed-rate rather than devfreq: it gives f/V safety, and no governor. Dynamic
scaling additionally needs only an OPP/devfreq table.

## Applicability

A higher clock raises prefill throughput only in the share of device time that scales with
it. On the fp16 matmul that share is ~39% at ~816 MHz, so 1019 MHz gives 1.083x on device
time (§"Rates from ~816 to 1019 MHz"). Wall-clock time gains less, because host work does
not scale with the NPU clock. One fp16 `512×3840×4096` matmul call takes ~105 ms, of which
~25 ms is device time. From ~816 to 928 MHz the call measures 1.00x, inside its ±4% run-to-run
spread [HW sweep 2026-10-19]. Given [not-mac-bound.md](not-mac-bound.md),
fewer and bigger NPU jobs are the larger prefill optimization, ahead of a higher clock.

Requests up to 1000 MHz need no new code: the voltage patch raises the rail to 0.85 V at
1 GHz. A clock above 1019 MHz needs a rail above the vendor's 0.85 V maximum. The RK3588
(~15 W) has no thermal throttle under `rocket`. The vendor caps the NPU at 800 MHz above
85 °C [source-confirmed: BSP `rk3588s.dtsi`]. At 928 MHz under continuous load the NPU levels
at 47-48 °C on a heatsinked RK1 [HW sweep 2026-10-19]. Watch temperatures under sustained
load above 928 MHz, or with less cooling.

## Evidence

### The requested rate and the physical clock

Each point was read with all three cores pinned runtime-active (`power/control=on`), 100
samples of the two monitor words at 2 ms, then released to `auto`. The rate and the rail were
changed through the `rocket_npu_clk_hz` and `rocket_npu_uv` sysfs params only while every core
was runtime-suspended. Each point first ran the fp16 `512×3840×4096` matmul test and compared
it bit-exactly. `CRU_CLKSEL_CON74` was read through `/dev/mem`, which the kernel permits
because `IO_STRICT_DEVMEM` is off.

The count's unit was checked against device time, taken from `drm_sched_job_run` to
`drm_sched_job_done` in a private tracefs instance. From the 200 MHz request (GPLL, 198 MHz) to
the 300 MHz request (427 counted), the median matmul job sped up 2.07x. Over the same step the
median job of a `64×148×148` 3×3 convolution sped up 1.86x, and the summed matmul device time
1.88x. The count predicts 2.16x. The requested rates allow at most 1.52x.

Under load, a convolution loop and a matmul loop ran together for 40 s with the cores pinned.
The count read 814-818 MHz throughout, from 36 to 42.5 °C.

### The operating point

At the 900 MHz request, a `64×148×148` 3×3 convolution loop and an fp16 `512×3840×4096` matmul
loop ran together for 901 s. All three cores were pinned on, and each convolution invocation made
4000 calls. Every 5 s the rail, the `npu-thermal` and `package-thermal` zones and 50 reads of
each monitor word at 1 ms were sampled.

| Quantity | Reading over the 165 samples |
|---|---|
| `0x50` word, per-sample mean | 924.7-927.5 MHz, single reads 922-928, 927.8 idle at the same request |
| `0x4c` word, per-sample mean | 921.8-928.7 MHz, single reads down to 897 |
| Rail | 800000 µV at every sample |
| `npu-thermal` | 49.0 °C peak |
| `package-thermal` | 49.9 °C peak |
| Matmul | 108 of 108 bit-exact against the host reference |
| Kernel log | No warning |

The test suite then ran unpinned at the 900 MHz request, and 126 of 126 tests passed in 791 s.
All three cores ended suspended, and the kernel log held no warning.

### Applicability

The matmul call ran under the `performance` governor at the 600 and 900 MHz requests. Five
passes alternated the order, with three calls a request a pass. Within each pass the median
call at 900 MHz measured 0.993, 1.000, 1.045, 1.040 and 0.992 times the throughput at 600 MHz.
Single calls ranged over 102.0-111.8 ms, and every output was bit-exact.

### Rates from ~816 to 1019 MHz

The governor was pinned to `performance` and restored afterwards. Two passes ran the 600, 800
and 1000 MHz requests in rotated order, two matmul runs per request per pass. Each run's first
job on each core includes the resume and is excluded from the steady sum. That first job
takes ~6.3 ms at a 0.80 V rail and ~8.7 ms at a 0.85 V rail.

One 1000 MHz run measured 19.68 ms and is excluded from the 23.15 ms mean. With it, the gain is
1.125x. A `64×148×148` 3×3 convolution measured too bimodal with the governor pinned,
169-231 µs per job at every rate, to resolve these steps.

### The 200 MHz boot rate

Under load, 60 samples of the requested rate across a full prefill were all 200 MHz
[source-confirmed, HW sweep].

### Clock duty cycle under a bursty workload

The 1 Hz sampler is `rocket-userspace/tools/npu_bench_env.sh`, which reads the requested rate.
Sessions also differ in the other direction: 0.8B F16 pp2048 measured 105 t/s in a cold-ish
session against 89 in a warm back-to-back sweep. The conditions of that pair are not recorded,
so it supports comparing only within one session, and not the direction of the difference.

### Measurement discipline

A single cold sample after a reload looked like a ~14% regression.

### The pinned `rmmod` hard-lock

The `rmmod` hard-lock at the 900 MHz request under the pin was measured twice, each needing a
power cycle, on 2026-06-15, before the voltage patch. The rail was at its 0.80 V boot voltage.

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

### Voltage coupling

Validated 2026-06-22, with all 4 tests passing at the 600 MHz request:

- Per-core `vdd->0.80 V`/`clk->600 MHz` in dmesg
- The rail pinned at 0.80 V
- The matmul bit-exact at 80-87 GFLOP/s warm
- The idle park returning clk->200 MHz at 0.80 V before a clean `rmmod`/reload

The 80-87 GFLOP/s is the fp16 `512×3840×4096` matmul on the library of 2026-06-22. The
75.3 GFLOP/s of the clock comparison at the top is the same shape on the library of 2026-06-15,
with host K-accumulation.

### Revision history

| Date | Superseded statement | Conditions | Current reading |
|---|---|---|---|
| 2026-06-15 | 200 MHz against 600 MHz, 1.43x on prefill | Requested rates, 0.80 V rail | 198 MHz against ~816 MHz |
| 2026-06-15 | 900 MHz gives no speedup over 600 MHz, `wait` 61 against 62 | Requested rates, 0.80 V rail | 928 against ~816 MHz, a 1.13x clock step, 1.044x on matmul device time |
| 2026-09-23 | The blob shows no static rate table | Read from the BL31 descriptor | The blob programs TF-A v2.12's ring lengths |
| 2026-10-19 | The operating point is the 600 MHz request, ~816 MHz | 0.80 V rail, ring length 17 | The 900 MHz request, 928 MHz, ring length 12 |
