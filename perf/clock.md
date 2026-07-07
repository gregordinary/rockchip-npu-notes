# The NPU clock — 200 MHz boot, the power-domain gotcha, the 900 MHz hard-lock

The RK3588 NPU compute clock (`scmi_clk_npu`, shared by all 3 cores) **boots pinned at
200 MHz** — one fifth of the 1 GHz silicon max. Raising it is a real ~1.43× win but is
dangerous to do the obvious ways. Both naive ways to raise it hang the box; the safe
path is below.

## Why it's stuck at 200 MHz

There is **no NPU devfreq under mainline `rocket`** — the driver never requests a
higher OPP, so the SCMI clock stays at its boot default. And the device tree
explicitly pins it: the `npu@fdab0000` nodes have `assigned-clock-rates = <200000000>`
for the `npu` clock. 200 MHz is the vendor's idle **`POWER_DOWN_FREQ`** — the rate it
parks at when idle. With no devfreq to ramp it back
up, `rocket` just leaves it parked there forever. (Confirmed under load: 60 samples
across a full prefill were *all* 200 MHz.) [source-confirmed + HW]

The throttled clock inflates the NPU `wait` ~5×: at 200 MHz, ~150 GFLOP/s/core is ~73%
of the *200 MHz* fp16 peak (a sane efficiency), versus an implausible ~15% against
1 GHz. The efficiency is against the throttled clock, not against a fast one — so the
prefill `wait` is not evidence the NPU is compute-bound at full clock.

## The power-domain gotcha — both naive fixes hang the box

The NPU power domains (`PD_NPUTOP/NPU1/NPU2`) are **off when the device is idle/cold**
(`rocket` uses runtime PM). **Setting the SCMI NPU PLL on an unpowered domain wedges
the SCMI firmware (EL3).** Both obvious approaches do exactly that:

1. **DT override** (`assigned-clock-rates = 600M`) — applied at `of_clk_set_defaults`,
   *before* the driver probes and powers the domain → **hung boot.**
2. **A standalone out-of-tree `clk_set_rate` module** — set the rate at idle
   (domain off) → **hung the live box.**

The vendor proves the rule: it only ever sets the NPU clock *during operation* (domain
powered), via devfreq, and its OPP helper refuses when `!pm_runtime_active`.

## The fix: set the clock from inside `rocket`'s runtime-resume

The only safe place to set the clock is **inside the driver, after `pm_runtime` has
powered the domain.** That is the `rocket-clk/` patch: a module param `rocket_npu_clk_hz`
(default 0 = stock), a `clk_set_rate` in `rocket_device_runtime_resume()` (domain
powered), and a park back to 200 MHz in `rocket_device_runtime_suspend()` (before the
domain powers off). Built as a **module** (kernel image + DTB untouched → boot never
at risk; recovery = `rmmod` or reboot). Because probe calls `pm_runtime_resume_and_get`,
a non-zero param is applied *during `insmod`* with the domain powered — so `insmod`
returning cleanly is itself the "didn't cold-set" proof. See the `rocket-clk` project.

## 600 MHz is the operating point (~1.43×)

- **600 MHz: safe + coherent.** Standalone fp16 `512×3840×4096` 56.5 → 75.3 GFLOP/s
  (1.33×); Gemma pp2048 **7.98 → 11.40 t/s (1.43×)**; int8 still bit-exact; temps
  48–57 °C, no throttle. Vendor-validated at vdd_npu 0.80 V (the OPP table puts 300–700
  MHz at 0.70 V, 900 MHz at 0.80 V, 1 GHz at 0.85 V).

- **Why only 1.43×, not ~4.5×:** at 600 MHz the NPU `wait` collapses and the *other*
  floors take over — host readback and per-job dispatch, both clock-independent (and, as
  [not-mac-bound.md](not-mac-bound.md) shows, the deeper floor is DMA/dispatch, not
  compute). Raising the clock only speeds the shrinking `wait`.

- **600 MHz is per-burst, not pinned — a bursty prefill samples a *median* of 200 MHz.**
  The park-to-200 lives in runtime-*suspend*, so the clock is 600 only while the NPU is
  actively held between submits. A workload with idle gaps — a few large `-ub 2048` ubatches
  with host pack/readback between them, or the pause between llama-bench reps — lets the
  domain runtime-suspend, so a 1 Hz sampler (`rocket-userspace/tools/npu_bench_env.sh`) reads
  *max* 600 but *median* 200; a continuously-fed workload (many small `-ub 512` ubatches)
  stays pinned at 600 the whole run. [HW] So the clock sampled during a bench is a duty-cycle,
  not a constant — read **max** as the operating point and **median** as how continuously the
  NPU was fed, and only compare t/s A/B within one session (a colder/less-continuous session
  reads low: 0.8B F16 pp2048 was 105 t/s cold-ish vs 89 in a warm back-to-back sweep).

- **900 MHz: not worth it.** Gives **zero** extra speedup over 600 MHz (`wait` stays 61
  — fully readback/dispatch-bound), and it is V/f-marginal here (the vendor's
  `set_read_margin` GRF tuning is unapplied). A *pinned* `rmmod` at 900 MHz **hard-locks
  the box** (measured twice, each needing a power-cycle): the `power/control=on`
  measurement pin defeats the suspend park-at-200, so the domain cold-powers-on at
  900 MHz — the exact hang condition the suspend-park exists to prevent.

## Safe operating procedure

Load at 600 MHz with **no `power/control` pin**:
`sudo rmmod rocket; sudo insmod rocket.ko rocket_npu_clk_hz=600000000` (be idle first).
The clock rides up to 600 MHz under load and auto-parks at 200 MHz when idle, so every
power-cycle is safe. Be idle before `rmmod`. The pin was a measurement crutch — never
use it for normal operation.

**Measurement discipline:** because the clock parks at 200 MHz idle and ramps under
load, the **first** `-r 1` benchmark after any `rmmod`/`insmod`/reboot/idle gap is
**cold** and reads ~15% low (≈2 t/s). Always run ≥3× back-to-back and compare the warm
runs (2nd–3rd); treat run 1 as a throwaway. A single cold sample reads like a ~14%
regression — a trap for anyone benchmarking after a reload.

There is a **second, independent** frequency confounder: the **CPU** governor. The
per-submit/dispatch overhead is CPU-side (the submit `ioctl` + the blocking wait on the
completion IRQ), so on an idle box an `ondemand`/`interactive` governor under-clocks the
A76 cores between submits and inflates any submit-bound number. An external RKNN-path
writeup measured a single `rknn_run` swing of **−41%** (59 → 35 ms) from the CPU governor
alone, NPU clock fixed — and pinning the *NPU* governor alone did nothing. [external,
proprietary path] So for a dispatch-floor measurement, pin the CPU cores to `performance`
as well; for a prefill throughput measurement (a few large jobs, dominated by NPU `wait`)
the CPU governor matters far less. See
[not-mac-bound.md](not-mac-bound.md) §Dispatch-floor reducers.

## Firmware (BL31): how the rate is set, and the OTP ceiling

The NPU compute clock is not a normal CRU clock the kernel writes directly — it is
owned by the secure firmware (BL31 / ATF) and exposed to Linux over SCMI. This is the
mechanism behind the power-domain gotcha above, and it pins down where the true rate
limit lives.

- **SCMI clock id 6.** BL31 exposes the NPU clock as `scmi_clk_npu`, SCMI clock id `6`
  (table order: cpul 0, dsu 1, cpub01 2, cpub23 3, ddr 4, gpu 5, **npu 6**, sbus 7).
  Matches the device tree `clocks = <... &scmi_clk SCMI_CLK_NPU ...>`. Linux reaches it
  via `arm-scmi` over the SMC transport (boot log: `SCMI Protocol v2.0 'rockchip:'`).
  [device tree SCMI_CLK_NPU; boot log]

- **`clk_set_rate` runs in EL3.** Setting the rate routes through the secure monitor;
  BL31 applies it via `rockchip_opteed_clk_set_rate`, i.e. the NPU PLL is programmed in
  EL3, not by the kernel. Setting it while the NPU power domain is off therefore wedges
  the firmware (this is the EL3 hang the runtime-resume fix avoids). [firmware behavior]

- **The rate is a PVTPLL, bounded by per-chip OTP — there is no static rate table.**
  At init BL31 adjusts the NPU PVTPLL against eFUSE/OTP values: the format string
  `adjust npu pvtpll by otp: min=%uM, max=%uM, length=%u` (same for cpul/cpub01/cpub23/
  gpu). The BL31 SCMI clock descriptor contains pointer/ops arrays, **not** a list of
  allowed Hz — so the real per-silicon NPU ceiling is the OTP `max`, set at the factory,
  not a value baked into firmware or DT. [BL31 firmware; per-chip OTP]
  - That OTP `min/max` line is emitted only by a **debug** BL31 to the secure UART. The
    release BL31 v1.51 shipped here does not print it at normal verbosity, so it is
    absent from the U-Boot/Linux serial log. To read this chip's ceiling: boot a debug
    BL31, or read the NPU PVTPLL OTP fields directly.

- **Firmware does not couple voltage to frequency.** The BL31 SCMI clock path programs
  only the PLL — there are no regulator/voltage operations for the NPU clock. `vdd_npu`
  (RK8602 PMIC at i2c 0x42, range 0.55–0.95 V) is managed entirely Linux-side, and under
  the SCMI *clock* model (not an SCMI perf-domain) nothing raises voltage when frequency
  rises. This is why a raised rate without a matching voltage is V/f-marginal (the
  900 MHz hard-lock above). [firmware behavior; Android DT `vdd_npu_s0`]

- **Linux applies a rockchip clock quirk.** The kernel enables
  `quirk_clock_rates_triplet_out_of_spec` for this `'rockchip:'` SCMI firmware — the
  firmware reports clock rates as a non-standard min/max/step triplet. [boot log]

- **Stock upstream `rocket` has no clock handling at all.** In android-mainline
  `drivers/accel/rocket/rocket_drv.c`, runtime resume/suspend only call `clk_bulk_*`
  (enable/disable) — there is no `clk_set_rate`, no module param, no devfreq/OPP. The
  `rocket_npu_clk_hz` ramp is entirely the local `rocket-clk` patch; an unpatched
  mainline kernel leaves the NPU at 200 MHz regardless. [android-mainline source; local
  v7.1 tree]

## Voltage coupling — the prerequisite for >600 MHz

Software scales the rail with the clock. The voltage patch
(`patches/rocket/082-rocket-drv-npu-volt.patch`, companion to the clk patch) holds the
`vdd_npu_s0` regulator for the device lifetime and scales it with the clock from the same
runtime-PM hooks: **voltage-up before clock-up** on resume, **clock-down before
voltage-down** on suspend, target = a vendor f→V map (300–700→0.70, 800→0.75, 900→0.80,
1000→0.85 V) clamped up to a **0.80 V floor**. The floor is load-bearing: the regulator
framework aggregates consumers by *max*, the rail already sits at 0.80 V, so voting the
floor **pins it at today's voltage at ≤600 MHz (non-disruptive)** and stops this vote
from ever pulling the shared rail *down* below 0.80 V. A `rocket_npu_uv` µV override exists
for >600 MHz bring-up. Compiled in-tree on mainline kernel **7.1.0-1-arm64**; validated
2026-06-22, all 4 gates pass @600 MHz (per-core `vdd→0.80 V`/`clk→600 MHz` dmesg, rail
pinned at 0.80 V, matmul bit-exact 80–87 GFLOP/s warm, idle parks clk→200 MHz @ 0.80 V, clean
`rmmod`/reload). Activate with `sudo modprobe rocket rocket_npu_clk_hz=600000000` (no pin).

This is a fixed-rate coupling, not devfreq — it gives f/V *safety*, not a governor. The
remaining gap is only an OPP/devfreq table if dynamic scaling is ever wanted.

## The remaining clock headroom (future work)

900 MHz / 1 GHz are a **config + deliberate V/f-and-thermal test** (the voltage coupling
above supplies the prerequisite), not new code — but worth revisiting **only** after
confirming the dispatch/readback floor, not the clock, is what's left (it currently is: at
900 MHz the speedup is zero). The vendor's `set_read_margin` GRF tuning is unapplied. Given
[not-mac-bound.md](not-mac-bound.md), the bigger prefill lever is fewer/bigger NPU jobs, not
more MHz. Capture this chip's OTP PVTPLL `max` (debug BL31) before any sweep, and watch temps
(RK3588 ~15 W, no auto-throttle).
