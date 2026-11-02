# RK3576 hazards

This note details three of the RK3576's hazards: the wide-output hazard (Erratum RK3576-E2),
the direct path's drain deadline (RK3576-E3) and the pooling divisor lag (RK3576-E4). It also
gives the hazards and limits that are not resolved. The errata table in
[rk3576.md](rk3576.md#errata) indexes every RK3576 hazard, and each erratum entry there gives its
conditions, symptom, workaround and status. A hazard that belongs to one datapath is detailed
beside that datapath in [rk3576.md](rk3576.md). The record behind each section here is in
[rk3576-validation.md](rk3576-validation.md), under the same section titles.

## The wide-output hazard

> [!CAUTION]
> A program that writes a DPU output element wider than one byte through a partial output
> stage leaves the next submit unwritten. The next submit can be of any kind and from any
> process. It writes nothing, and completes in the usual time except on `rocket` 1.6.0, where
> the driver retires it at the 125 ms backstop, its job timeout. Submits in between do not clear
> the state.
> Software must let the NPU power domain collapse before the next submit. To detect a victim,
> software must stamp the output BO before the submit and check that each task wrote.

**Conditions:** H96 MAX M9. A result measured on `rocket` 1.6.0 names the module version.

The wide-output hazard is a state of the SoC, not of the process, and only a power-domain
collapse resets it [HW sweep, H96 MAX M9]. A one-byte output never triggers it. The regcmd encoder's
int32 writer is a partial output stage. It moves `0x4010`'s `out` field and leaves the rest of
the stage at its int8 values. A whole output stage writes the same widths without the hazard
(§"The whole output stage").

| Program | Triggers the hazard |
|---|---|
| int8 direct convolution | No |
| int8 depthwise convolution | No |
| int8 first convolution | No |
| fp16 direct convolution | Yes |
| fp16 first convolution | Yes |
| int32 writer (`gen_conv2d_int8_rk3576_i32out()`) | Yes |
| Wide int32 writer | Yes |
| Matmul-form fp16 program, fp32 output | No |
| Dense int32 writer, on a whole output stage | No |
| fp16 convolution with fp32 output, on a whole output stage | No |

In the first seven rows, each wide output is written through the regcmd encoder's partial output
stage. Of those seven programs, each marked Yes leaves all seven kinds unwritten as the next
submit. Each marked No leaves none [HW sweep, H96 MAX M9]. The last three rows were
measured on `rocket` 1.6.0: the matmul-form program on 2026-09-25 and the two whole-stage
programs on 2026-09-30. The fp16 first convolution, unmodified, was measured on 1.6.0 the same
day and still triggers the hazard [HW sweep, H96 MAX M9, 2026-09-30].

The hazard persists on `rocket` 1.6.0 (kernel 7.2.3-1) [HW sweep, H96 MAX M9, measured
2026-09-24 UTC]. The 1.6.0 drain fix cures a lost output tail, not this hazard. The driver logs a
kernel `retiring it` line for each submit it retires at the backstop
[HW sweep, H96 MAX M9, 2026-09-27].

The matmul-form fp16 program writes a flat fp32 output, four bytes an element, and does not
trigger the hazard. It sets DPU `0x4038[4]` and `0x4050[17]`, the pair the regcmd encoder's fp16
convolution triggers the hazard through (§"The register conditions"). It also raises
its own completion, and the driver retires none of its submits at the backstop.
[HW sweep, H96 MAX M9, `rocket` 1.6.0, measured 2026-09-25 UTC]

> [!NOTE]
> A test that runs the wide-output program and the next submit in two processes does not see
> the hazard. Starting a process takes longer than the idle that clears the state
> (§"The clearing idle").

### The register conditions

The regcmd encoder's two wide-output paths trigger the hazard through different DPU register
conditions [HW sweep, H96 MAX M9]:

- **int32 writers:** DPU `0x4010` (`out<<29 | in<<26 | proc`) alone. Forced onto the int8
  direct convolution, `out` values 1-3 are clean and 4-5 trigger the hazard. Restored to its
  int8 value in the int32 writer, it clears the hazard, and the program still writes.
- **fp16 path:** `0x4038[4]` and `0x4050[17]`. With either bit set, the program triggers the
  hazard, and with both clear it does not. No other bit of the two registers that differs from
  the int8 program matters.

The two bits are not sufficient on their own. The matmul-form program and the whole output
stage set both, and neither triggers the hazard.

The narrow int32 writer differs from the int8 direct convolution in four registers, `0x4010`,
`0x40AC`, `0x40B0` and `0x40B4`. On it any non-zero `0x4010` triggers the hazard, `out` 1-3
included. The same `out` value is therefore clean on one program and triggers it on another.
The fp16 first convolution differs from the int8 program in 19 registers. Each of them triggers
the hazard alone, and leaving out any one clears nothing [HW sweep, H96 MAX M9].

> [!CAUTION]
> `0x4038[4]` and `0x4050[17]` belong to the float mode. `0x4038`'s low half must read `0x92`,
> not the direct path's `0x80`. `0x4050[17]` lies in the high-half field that reads 2 at float
> and 1 at int8. With both cleared, the fp16 direct convolution writes its output without
> the hazard, and shapes that matched a CPU model exactly produce wrong output. Software must
> keep both bits set in a float program. [HW sweep, H96 MAX M9]

The state is held inside the NPU, outside the DPU register file. No register write by the next
submit resets it, including `0x4038` and `0x4050` at their int8 values. Changing the buffers or
their addresses does not clear it either. [HW sweep, H96 MAX M9]

> [!NOTE]
> A program that writes nothing can still trigger the hazard. With `0x4038`'s low half at `0x80`
> and the rest of the program float, the program writes nothing, and the submit after it is
> unwritten too. A register sweep must score a known-good submit after each program, not the
> program's own output.

### The clearing idle

The idle that clears the state is twice `power/autosuspend_delay_ms`. The table gives the gap
between wide-output submits at which every submit of a chain writes:

| `autosuspend_delay_ms` | Gap at which every submit writes, ms |
|---|---|
| 50 | ~100 |
| 20 | ~40 |
| 10 | ~20 |
| 5 | ~10 |
| 1 | ~5 |

[HW sweep, H96 MAX M9]

> [!CAUTION]
> With `on` written to the device's `power/control`, runtime suspend never fires, and no idle
> clears the state [HW sweep, H96 MAX M9]. Software that pins the NPU active must clear the
> state some other way, or not run a program that triggers the hazard.

Lowering the delay system-wide has a cost that a back-to-back chain does not show. A resume
costs about 230 us, and every job whose inter-job gap falls between the new delay and the old
one incurs it. At a 1 ms delay that costs ~20% a job, until both delays suspend anyway past
60 ms. At a 16.7 ms gap, one 60 Hz frame, a submit goes from 1124 us to 1372. Mainline's
50 ms delay was chosen for that pipeline. At a zero gap the two delays cost the same, because
the timer never fires.

Driving the power cycle clears the state sooner than waiting for it. Write a zero delay to
`power/autosuspend_delay_ms`, poll `runtime_status` until it reads `suspended`, and restore the
previous delay. The cycle then
costs one real power-down rather than a worst-case idle. The delay every other consumer of the
NPU sees is unchanged.

### LUT and elementwise jobs as victims

A victim, the submit the hazard leaves unwritten, can be a DPU-only program. A LUT job writes
nothing behind a wide-output program, and so does an elementwise program
[HW sweep, H96 MAX M9]. The LUT job is a table load and its consumer, in one submit. Every victim
kind measured writes nothing under the hazard: the seven convolution and matmul kinds, the LUT
job and the elementwise program.

The hazard takes the whole next submit, not only its first task [HW sweep, H96 MAX M9]. Once
it starts, no submit writes until the power domain collapses. An elementwise run of ten submits
inside one active window writes none of them. A single power cycle clears the state for the
rest of the run. [HW sweep, H96 MAX M9, 2026-08-04]

A pooling job does not trigger the hazard, including a padded average pool in either divisor
mode [HW sweep, H96 MAX M9].

> [!NOTE]
> An unwritten output keeps whatever the CPU left in its BO. Pre-filled with `0xAA`, an int8
> output reads back as a constant -86 over the whole output, which looks like a computed wrong
> value. A test that pre-fills an output BO must report how much of the fill remains.

### An externally reported hazard on the w4a16 RDMA path

> [!CAUTION]
> A job that programs the vendor's w4a16 values into DPU_RDMA `0x5034` (`0x4000004c`) or
> `0x5044` (`0x000280a1`), either register alone, leaves the next job timing out. An `rk_iommu`
> reset error appears beside it, and the driver recovers without a power-domain cycle. Software
> that takes a w4a16 stage from vendor register streams must treat this as a separate hazard.

The `charsiu` project reports this hazard on a ROCK 4D with its own kernel (2026-08-15). It has
not been measured on the H96 MAX M9. It differs from the wide-output hazard on both observable
axes. Its victim times out with an IOMMU fault rather than completing unwritten, and its state
clears on driver recovery rather than on a power-domain collapse.

### The whole output stage

A whole output stage writes a wide output without the hazard
[HW sweep, H96 MAX M9, rocket 1.6.0, 2026-09-30]. The stage measured is the `charsiu` project's
wide output stage, which writes a four-byte output on every dispatch. The stage is nine DPU
words, listed in [rk3576-regcmd.md](rk3576-regcmd.md) §"The dense writer". Three of them clear
the hazard on their own, and the other six set what is written. Added to the regcmd encoder's
int32 writer, the three clear the hazard and keep the output width:

| Register | Value | Fields |
|---|---|---|
| `0x4010` | `0xa0000002` | `out`, `in` and `proc` |
| `0x4038` | `0x53` | Bit 4 set |
| `0x4050` | `0x00023333` | Bit 17 set |

All three are required: any one or two of them leave the hazard in place. The program still
writes, at a zero gap.

The whole stage, with `0x40B8` at three surfaces, is a dense int32 writer that does not trigger
the hazard. Submits after it write, and so do its own submits back to back with no idle. It
raises its own completion, so the driver retires none of them at the backstop
[HW sweep, H96 MAX M9]. The register program and its map are in
[rk3576-regcmd.md](rk3576-regcmd.md) §"The dense writer".

The hazard therefore belongs to the regcmd encoder's partial output stage, not to the output width.
The regcmd encoder's int32 writer moves `0x4010`'s `out` field and leaves the rest of the stage at its
int8 values. The whole stage also sets `0x4038[4]` and `0x4050[17]`, the two bits the fp16
convolution triggers the hazard through. Neither that pair nor `0x4010` decides it alone.

On the fp16 slice program, the same stage computes an fp16 convolution with an fp32 output.
Every element matches a CPU model on random small-integer operands at k 1 and 3
[HW sweep, H96 MAX M9, rocket 1.6.0, 2026-09-30]. `0x4038 = 0x53` selects the float mode there,
as the float word's `0x92` low half does. The output is the plain int32 cube: four fp32 lanes to
an atom, one atom per pixel, each quad of channels a plane. This fp32-output convolution does
not trigger the hazard. With the regcmd encoder's float `0x4038` in place of `0x53`, it does.

### Software support

`rocket_rk3576_power_idle()` drives the power cycle that §"The clearing idle" describes. It
needs write access to `power/autosuspend_delay_ms`, probes that access once, and without it falls
back to the plain idle. `ROCKET_RK3576_PM_KICK=0` turns it off. A kill during the few
milliseconds in which the delay sits at zero leaves it at zero. The library reads a zero delay as
an interrupted forced transition, and restores mainline's 50.

At the stock 50 ms delay, `rocket_rk3576_power_idle()` runs the fp16 first convolution's
224×224 k3 s2 in 13.75 ms, against 226 ms waiting for the idle. The int32 matmul test runs in
9.9 s against 16.2 [HW sweep, H96 MAX M9, 2026-07-27]. Every test stays bit-exact.
Lowering the global delay to 1 ms instead reached only 21.6 ms, and slowed every gapped
workload. [HW sweep, H96 MAX M9]

`librocketnpu`'s write guard detects an output left wholly untouched. It then cycles the power
domain with `rocket_rk3576_power_idle()` and redoes the submit, up to eight attempts. It does not
retry a partially written output, because a retry there would hide an encoding defect. The guard
depends on a victim writing nothing, which every victim kind measured does. In the LUT and
elementwise measurements, a single power cycle repaired every untouched output [HW sweep, H96
MAX M9].

On the narrow int32 writer, one power cycle cleared the state about 87% of the time. A
cap of four attempts left about 0.6% of retried tasks failing outright, and at eight none
failed [HW sweep, H96 MAX M9, 2026-07-28]. A task that reads unwritten is unwritten: a re-check
after a short settle recovers no task, and the power cycle behind it recovers them all.

> [!CAUTION]
> A row-split tile issues several submits into one output buffer. When one is left unwritten,
> its rows are empty while its siblings' rows are full. A check over the whole tile then passes
> the hole to the caller. The result is one row task short, intermittently, with no fault and
> normal timing. Software must check each task's output region, not the whole tile.

The activation entry runs behind the guard. At `ROCKET_LOG_LEVEL=debug`, its log line names the
unwritten task and the attempt:
`rocket_act_int8_rk3576: of 1 row task(s) in this submit the first that wrote nothing is 0, on attempt 1; cycling the power domain and redoing it`.

`librocketnpu` emits the whole output stage only as all nine words. It refuses a program in
which it finds fewer, because a partial stage is the hazard's condition.

`rocket_conv2d_fp16_rk3576()` runs its slices on the whole output stage, with no power cycle per
program and none on exit. At k9 over 32 channels on 120×64, a call takes 9.9 ms, and over two
slices 15.1 ms, every element exact [HW sweep, H96 MAX M9]. At 28×28 k3, 128 input channels into
256 output channels run as 64 programs in 47 ms. The packed-image first convolution has not been
run on the stage, and still cycles the power domain.

No `librocketnpu` regcmd encoder emits a w4a16 stage, so none emits the DPU_RDMA values of the
external report.

## The direct path's drain deadline

**Conditions:** H96 MAX M9, 2026-08-05, NPU clock 786 MHz (see [NPU clock
rate](rk3576.md#npu-clock-rate)). `rocket` 1.5.0 at its default `dpu_grace_us` of 500, unless a
row states otherwise. One VALID-padded plane per task.

The RK3576 raises `PC_DONE` when the program counter retires the descriptor, not when the
pipeline behind it has finished. The `rocket` driver can therefore signal a job's fence while
the DPU is still writing. On the direct convolution path, the row limit is this drain deadline,
and no CBUF row allowance applies [HW sweep, H96 MAX M9, 2026-08-05].

> [!CAUTION]
> On `rocket` 1.5.0, a program whose DPU drain outlasts the driver's `dpu_grace_us` loses the
> tail of its output, and nothing faults. The driver signals the job's fence while the core is
> still writing. On the direct convolution path the last rows of the last output-channel group
> are never written. The output has the correct size and a wrong corner, and resembles a CBUF
> row-allowance overrun. The int8 matmul entry loses part of its last task's output the same way
> ([Dropped writes at the drain deadline](rk3576.md#dropped-writes-at-the-drain-deadline)).
> Software must run on a `rocket` module that waits for the writing block, or keep each
> program's drain inside the grace.
>
> On `rocket` 1.6.0 the drain fix is in. `dpu_grace_us` = 0, its default, means wait for the
> writing block, and no loss is measured [HW sweep, H96 MAX M9]. A nonzero `dpu_grace_us`
> restores the deadline, and `ROCKET_JOB_NO_DPU_DONE` fences a job at the blind settle, so
> either brings the loss back ([`dpu_grace_us` on `rocket`
> 1.6.0](rk3576.md#dpu_grace_us-on-rocket-160), [Retirement on the DPU
> completion](rk3576.md#retirement-on-the-dpu-completion)). Software must leave `dpu_grace_us`
> at 0 on 1.6.0, and set the flag only on a program that raises no DPU completion.

One module parameter decides the outcome. At `dpu_grace_us = 500`, a `112x112 k7 ic32`
convolution at four output-channel groups is wrong at every plane height from 45 rows up. At
750 every one of those heights is bit-exact [HW sweep, H96 MAX M9, 2026-08-05]. The shape, the
program and the CBUF `F` are the same at both. The threshold lies between 500 and 750 us at four
groups and at six alike. It is a fixed time, independent of the shape.

Each measured limit falls at about the same amount of work, and no CBUF sum fits them. Each row
is one VALID-padded plane, run as exactly one task:

| groups | oc | tallest exact plane | feature granules | MMACs |
|---:|---:|---:|---:|---:|
| 1 | 32 | never fails to 109, the reachable top | 6104 | 548 |
| 2 | 64 | 78 | 4368 | 766 |
| 3 | 96 | 56 | 3136 | 798 |
| 4 | 128 | 44 | 2464 | 808 |
| 5 | 160 | 36 | 2016 | 798 |
| 6 | 192 | 32 | 1792 | 830 |

The limits span 766-830 MMACs, an 8% band, while the feature footprint at the limit falls 2.4x
across the same rows. One output-channel group never fails at any reachable height, because its
work reaches only 548 MMACs, under the band. The capacity rule `feature + weights <= pool` does
not fit at all. One group is exact at a feature-plus-cube total of 6888 granules, where two
groups fail at 5992.

### Work and output size

The deadline limits the program's execution time, not the size of the output it writes
[HW sweep, H96 MAX M9, 2026-08-05]. The six rows above hold `ic` and `k` fixed, where the MACs
are exactly proportional to the output elements. A work law and an output-bytes law fit them
equally. Five further test points break that proportionality through the kernel, the plane
width and `ic`, and they separate the two laws by factors:

| Test point | ic | oc | iw | k | first wrong height | output there | work there |
|---|---:|---:|---:|---:|---:|---:|---:|
| `k7g2-112` | 32 | 64 | 112 | 7 | 79 | 495 KB | 777 MMAC |
| `k7g3-112` | 32 | 96 | 112 | 7 | 57 | 519 KB | 814 MMAC |
| `k7g4-112` | 32 | 128 | 112 | 7 | 45 | 529 KB | 830 MMAC |
| `k7g5-112` | 32 | 160 | 112 | 7 | 37 | 526 KB | 824 MMAC |
| `k7g6-112` | 32 | 192 | 112 | 7 | 33 | 550 KB | 862 MMAC |
| `k5g4-112` | 32 | 128 | 112 | 5 | 77 | 1009 KB | 807 MMAC |
| `k7g4-224` | 32 | 128 | 224 | 7 | 25 | 530 KB | 831 MMAC |
| `k7g4-56` | 32 | 128 | 56 | 7 | 60, dense from 89 | 346 KB, 531 KB | 542, 833 MMAC |
| `k7ic64-g2` | 64 | 64 | 112 | 7 | 34 | 190 KB | 596 MMAC |
| `k3g4-112` | 32 | 128 | 112 | 3 | none to 109 | 1507 KB exact | 434 MMAC |
| `k7g1-112` | 32 | 32 | 112 | 7 | none to 109 | 349 KB exact | 548 MMAC |

Output size does not set the limit. At the first failures, output spans 190 KB to 1009 KB, a
factor of 5.3, where work spans 542-862 MMACs, a factor of 1.6. The test point `k3g4-112` writes
1507 KB with no error at any reachable height, while `k7ic64-g2` is wrong at 190 KB, eight times
less. At three taps the MACs per output element fall 5.4x and the output grows, and the limit
disappears.

Output rows do not set it either. The test point `k7g4-224` is wrong at 19 output rows. The test
point `k7g4-56` first fails at 54 output rows, intermittently, and fails densely from 83. A row
count fitted on the `112`-wide family, 38 output rows, misses in both directions. It is twice
`k7g4-224`'s count, and 0.46x to 0.70x `k7g4-56`'s. The width moves the limit through the total
work, not through the row count.

Work stands in for the program's execution time, and does so well only where the weight traffic
per output is fixed. The five `112`-wide k7 test points and `k7g4-224` fall in 777-862 MMACs,
and the dense region of `k7g4-56` starts at 833. The test point `k7ic64-g2` fails 27% below
that band, at 596 MMACs. It has twice the resident weight slice per output-channel group (1568
granules against 784), so each output element costs more DMA than a MAC count sees. The
explanation of the residual spread is [hypothesis], and the measured result is the ordering.

At 786 MHz, 800 MMACs in 500 us is about 2050 MACs per cycle, and two cores at that rate make
the RK3576's marketed 6 TOPS. A one-task program's whole compute therefore fits inside the
grace.

At `k7ic64-g2` the failure changes form at height 37, where two mechanisms meet. Below 37 the
wrong elements are a row tail of the last output-channel group, the deadline's form. From 37 up
they are every row of that group (rows 0-30 of 31), and the planner's CBUF `F` moves from 0 to
1024 at the same height.

On a `rocket` module that waits for the writing block, the heights below 37 are exact. Heights
37-45 stay wrong at exactly the same element counts under both modules (104936 at 37, 108489 at
38, 111936 at 39). The second mechanism is therefore deterministic and independent of timing,
and [The weight slice and the feature
allowance](rk3576.md#the-weight-slice-and-the-feature-allowance) describes it. Software support:
`librocketnpu` refuses the heights from 37, because the slice cap's `F`-aware form does not
cover them.

### Failure rate below the refused heights

At heights 33 to 36 on `k7ic64-g2`, below the refused heights, the failures are the drain
deadline [HW sweep, H96 MAX M9, 2026-08-05]. The planner programs `F=0` there. The plane needs
3696 to 4032 of 4096 granules, and the 98 KiB slice is inside the 156 KiB `F=0` cap. These
heights are wrong on `rocket` 1.5.0 at its default grace, and bit-exact under either of two
independent removals of the deadline:

| Module and grace | ih 33 | ih 34 | ih 35 | ih 36 |
|---|---:|---:|---:|---:|
| `rocket` 1.5.0, `dpu_grace_us` 500 | 2 of 30 | 14 of 60 | 26 of 60 | 56 of 60 |
| `rocket` 1.5.0, `dpu_grace_us` 1000 | n/a | 0 of 30 | 0 of 30 | 0 of 30 |
| `rocket` 1.6.0, which waits for the writing block, at its default | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |

> [!NOTE]
> At a fixed height, the failure rate varies widely between identical runs. At ih 35, six runs
> of the same command at the same grace gave 0, 1, 1, 6, 8 and 10 wrong of 10 repetitions. A
> run with no failures does not show that a height is safe. Nor does the range of heights a
> single run marks as failing.

### Extent of the loss

The extent of the loss depends on when the fence falls, along one continuum
[HW sweep, H96 MAX M9, 2026-08-05]. A late cut drops the row tail of one output-channel group:
hundreds to a few thousand wrong elements, `chans 32-47`. An earlier cut leaves the output wrong
across every row and all 64 channels, 150000 to 170000 of 203520 elements. Sweeping
`dpu_grace_us` at `k7ic64-g2` heights 33-36 moves the extent smoothly from one form to the
other:

| `dpu_grace_us` | wrong reps | wrong elements per rep | channel span | row span |
|---:|---:|---:|---|---|
| 500 | 11 of 40 | 1184 - 3808 | `c 32-47` | a 1-3 row suffix |
| 250 | 40 of 40 | 27798 - 68768 | `c 32-63` | from row 5-16 down |
| 100 | 40 of 40 | 84638 - 131008 | `c 0-63` | every row |
| 0 | 40 of 40 | 116364 - 155008 | `c 0-63` | every row |

Grace 0 at ih 36 reaches 155008 elements, inside the severe form's band. The severe form at
grace 500 is therefore the tail of one distribution, not a second event. Of the 131 wrong
repetitions in the sweep, 107 fall between the two forms. At grace 500 the severe form did not
occur once in 240 repetitions on 2026-08-05.

Every wrong element is a dropped write, never a wrong value [HW sweep, H96 MAX M9, 2026-08-05].
Each output is stamped with `0xA5` before the submit. Every wrong element in every wrong
repetition still holds the stamp, and no wrong element holds data. The result holds at every
grace value, including 0, where the driver retires each job as early as it can. A lost drain
therefore costs part of the job's own output, and the core does not write into an unrelated
job's buffer. The result holds for one process issuing sequential convolution submits ~10 ms
apart, at one geometry.

**Not characterized:** a victim in another process, of another job type, or in a graph
submitting back to back.

The loss grows in the order `c 32-47`, then `c 48-63`, then `c 0-31`, never in natural channel
order, at any grace value [HW sweep]. The likely cause is the DPU's DDR write order, in which
32-47 is the last channel group this geometry writes [hypothesis].

> [!NOTE]
> A slice-cap overrun and the drain deadline both leave output unwritten, and the shape of the
> output alone does not separate them. The stamped elements do, by where they fall. A slice-cap
> overrun loses whole trailing output-channel groups, every row of each: `chans 32-63`, exactly
> 50% here. The deadline loses a suffix in write order, at whatever depth the cut fell.

## The pooling divisor lag

A padded average pool intermittently divides an output window by the tap count of the previous
window in raster order. The sum is right, and only the divisor is one output position late. The
predecessor of a row's first window is the previous row's last window. Once the lag starts, it
persists to the last channel group of the program (see "Lag extent and the tail-only check"
below). A failing output is therefore a contiguous suffix of channel groups, wrong at a few
columns of every row and exact everywhere else.

> [!CAUTION]
> A padded average pool in exclude-pad mode can divide an output window by the previous window's
> tap count, intermittently and with no fault. The output is written in full, and is wrong. Software
> must check the positions whose raster predecessor has a different tap count, and redo the submit
> on a mismatch.

The lag is observable only where the pad is excluded, the only mode whose divisor varies with
position. At a constant divisor, the previous window's divisor is this window's. The lag can
therefore occur under every max pool and every count-include-pad average on the RK3576 without
changing the output.

The lag is independent of the wide-output hazard. An int32-output matmul in front leaves its rate
where it was, and the lag reproduces standalone with nothing before it at all. A redo without a
power cycle repairs it, since a domain collapse is neither necessary nor sufficient for this
hazard. Its rate depends on the composition of buffers in the process. The same shape and entry
range between 0% and 95% with no program changing (see "What sets the loud/quiet rate" below).

The lag can show only at positions whose raster predecessor has a different tap count. For a SAME
window, those are two columns of every row, plus a position where the vertical count changes. A
check of them costs three columns: O(rows) per image rather than O(rows × columns). Any occurrence
shows at thousands of them at once.

In a graph the rate is about a third of calls. In Inception V3, `librocketnpu` repairs five to
eleven calls per inference set across the graph's nine padded average pools. A failing output has
30-40% of its observable elements wrong. A run of failures is therefore ordinary, and is not a sign
of another fault.

#### Software support

`librocketnpu` checks the observable positions after every padded average pool and redoes the
submit on a failure, with no power cycle. The check also covers a cube join, which is where the
hazard costs a label. A joined pool reads the producer's output rather than a row-major tensor:
the same values, in the layout the PPU itself fetched them from. The check is the same arithmetic
over a different accessor, and it costs one cache-maintenance bracket the join otherwise never
incurs. Only a padded average pool incurs it.

A cross-layer kick issues the submit instead of the entry, so the kick runs the same check from its
verify bracket (`rocket_pool_int8_rk3576_lag_check()`). It runs before the write check, because
that loop re-stamps as it goes and the pool's producer is earlier in the same chained run. The
check first asks whether an output was written at all. An unwritten output goes to the write check,
which has the power cycle this hazard does not need.

The redos have their own budget of 32, `ROCKET_RK3576_POOL_LAG_ATTEMPTS`, separate from the
wide-output hazard's eight. A lag redo costs a submit, where a wide-output redo costs a power cycle.
A kick also redoes every pool of the chained run at once. Two observable pools in one stream must therefore
both be clean on the same attempt.

### What sets the loud/quiet rate

The divisor lag's rate depends on where the job's buffers fall in the IOVA space, modulo 64 KiB.
A pool fires on a call whose output shows the lag. A loud configuration fires on most calls, and a
quiet one on almost none.

The same geometry, entry and binary, called back to back, range between ~0% and ~95% depending on
which other buffers the process allocated. A BO allocated before the job's buffers, and never
submitted against or touched, moves the rate. The rate follows a map periodic in exactly 16 pages,
64 KiB [HW sweep, H96 MAX M9, 2026-08-05]. A loud band is a run of adjacent loud residues of that
period. `rocket` exposes no way to read an IOVA back. The 64 KiB modulus is therefore measured
through the allocation offset, and the mechanism behind it is unidentified.

The rule has five measured properties [HW sweep, H96 MAX M9, 2026-08-05]:

- The variable is the total page count allocated before the job's own buffers, not the buffer
  count. `rocket_bo_alloc` rounds to 4096, so a byte-valued rule is not even expressible.
- Freeing the earlier allocation restores the unpadded state exactly. It leaves no pressure, no
  fragmentation and no page-table residue behind.
- The period is a constant of the address space, and its phase is absolute. Which residues are loud
  is a function of the geometry and the entry.
- The prepacked cube-in path obeys the same 64 KiB period. It needs both a phase and other pool
  handles present in the process. Those handles need not run. A `rocket_bo_alloc` that no program
  ever names shifts the phase, but cannot create a loud band.
- Only a susceptible pool at its own phase is loud. In a process of fifteen pool configurations,
  two show loud bands, at disjoint phases, and the other thirteen show none at any phase.

What makes a configuration susceptible is not identified, and it is not a gradient in the source
line stride. A source line stride of 128 gives a loud band, and the next value down, 64, gives none
[HW sweep, H96 MAX M9, 2026-08-06]. The phase is not predictable from a byte count, because some
allocation in the bookkeeping is uncounted [HW sweep, H96 MAX M9, 2026-08-05].

> [!NOTE]
> A divisor-lag rate measured for one shape holds only for the buffer composition it was measured
> in. A standalone test of one shape does not give that shape's rate in a graph. A pad sweep run on
> a shape in isolation reads quiet at all sixteen phases.

### Exposure and cost in the model corpus

One model of the seven-model corpus is exposed to this hazard at all. The check runs only where
the pad is excluded from an average. A max pool and a count-include-pad average have a constant
divisor and cannot show the lag. The table counts each model's own pooling calls:

| model | pooling calls an inference | observable (padded exclude-pad average) |
|---|---:|---:|
| Inception V3 | 10 | **9** |
| Inception V1, MobileNet V1, MobileNet V2, ResNet-18 | n/a | **0** |
| EfficientDet-Lite0 | 14 | **0** (all max) |
| SSD MobileNet V2 | 0 | **0** |

Inception V3 sits on a band's partial edge, neither in it nor out of it. Eight of its nine
observable pools fire on 0-9% of calls, the pad sweep's quiet level. Layer 125 (`2048x8x8`, cube in
and cube out) fires on 22.9-33.1%, the rate the pad sweep produces at a band's partial residue. The
graph is reproducibly one phase step from a rate three times higher, and one step from nothing.
Which of the three applies is not predictable from a byte count.

**Conditions:** H96 MAX M9, Inception V3, `bench 100`, `ondemand` governor, 2026-08-06.

The check and its redos together cost Inception V3 1.5 ms of 34.0 ms (4.4%). The redos are ~1
extra submit an inference [HW sweep, H96 MAX M9, 2026-08-06]. The check is 1.04 ms an inference,
incurred on every call the hazard can show on, and a redo is incurred only at the hazard's rate. A
pool call measures 283 us clean and 562 us with one redo.

A setting that moved the graph into a quiet phase would remove the redo half only. The redo half is
~0.5 ms of 34.0 ms, ~1.5% on one model of seven, and zero on the other six [HW sweep, H96 MAX M9,
2026-08-06]. `librocketnpu` keeps the check, because a phase that cannot be predicted cannot be
trusted.

> [!NOTE]
> Profiling with `ROCKET_RK3576_POOL_PROF=1` roughly halves the lag's rate. The per-call host work
> between submits is itself part of the composition the rate depends on. Use the profile for the
> per-redo cost, and an uninstrumented run for the rate.

Every failing element is the previous window's divisor, and none is anything else. The check scores
rather than judges. It counts the observable elements that match the right divisor, those that
match the raster predecessor's, and those that match neither. A zero third count identifies a
failure as this hazard rather than a defect in the accessor.

### Coverage cost per graph

The divisor-lag check costs nothing on four graphs of five. Only a padded average pool has a
position where the divisor varies, and only Inception V3 has one. The other four measure identical
wall-clock times with the check on and off. Inception V3's cost is under "Exposure and cost in
the model corpus".

**Conditions:** H96 MAX M9, `bench 100`, with `ROCKET_RK3576_POOL_LAGCHECK=0` as the control,
2026-08-03.

| graph | wall-clock time, check on or off (ms) |
|---|---:|
| MobileNetV1 | 5.0 |
| MobileNetV2 | 6.5-6.6 |
| ResNet-18 | 6.6 |
| Inception V1 | 7.0-7.1 |

Placing a padded average pool inside a chained kick raises the graph's time and makes it vary from
run to run. The spread is the redos, whose count is a random variable. The placement moves both
halves of the cost, and `librocketnpu` refuses such a pool as a member of a chained run (see "Chained-run refusal for
pools whose divisor can lag" below).

The check and the redos are separate costs that share a loop position and nothing else, and the
check is the larger one. The check is arithmetic plus a cache-maintenance bracket, incurred on
every call the hazard can show on, so it barely moves with the placement. It is host work rather
than NPU time. A redo is incurred at the hazard's rate, and what the placement moves is what each
redo costs. A redo inside a kick re-runs every program of the chained run.

### The pooling path's read interval

The pooling path's `read` interval runs from the post-submit `PREP_BO` to the closing `FINI_BO`. It
contains the fence wait as well as the write check, and the wait is the PPU finishing, not a guard.
The profile under `ROCKET_RK3576_POOL_PROF=1` prints the split as wait / scan / fini.

**Conditions:** H96 MAX M9, Inception V3, `bench 100`, `ondemand` governor, 2026-08-03, per
inference over 15.2 pool calls.

| term | ms | what it is |
|---|---:|---|
| divisor lag check | 5.98 | host arithmetic over the observable positions |
| fence wait | 2.85 | the PPU finishing, plus one invalidate of the output |
| de-tile | 1.44 | the column-split pools' output side |
| sentinel stamp | 1.05 | prep + memset + fini, per call |
| redos | 0.81 | every attempt after the first |
| closing fini | 0.38 | the guard's other half |
| submit ioctl | 0.25 | |
| host cube scatter | 0.07 | |
| sentinel scan | **0.00** | it stops at the first byte the program wrote |

The tail-only check (see "Lag extent and the tail-only check") takes the lag check's figure to
about 1 ms.

The pooling write guard is `stamp + scan + fini` = 1.43 ms [host-computed from the table]. The
fence wait inside the `read` interval is not part of it, and the sentinel scan costs nothing
measurable.

The guard is zero on the other four graphs. Their pools are members of chained runs, so the per-call pooling
entry never runs at all.

### Lag check cost

The lag check is a pass over the source cube and the output, not arithmetic. It touches ~60% of the
cache lines of both, at roughly 0.5 GB/s of effective line traffic. No reordering helps, because
the positions it must score are spread over every row.

`librocketnpu`'s check scores 16 channels at a time rather than one. An atom is sixteen consecutive
channels, so a per-channel walk is sixteen strided passes over the same lines. Two further
optimizations are each byte-identical, and neither trades coverage:

- **The tap loop in NEON.** On a cube the sixteen lanes of an atom are sixteen contiguous bytes, so
  a tap is one 16-byte load and two widening adds.
- **The rounding's three integer divisions, hoisted out of the element loop.** The quotient and the
  two modulos behind `exact_recip` are functions of the output position, not of the channel. They
  are built once per position, and the quotient becomes an exact reciprocal multiply,
  `M = 2^32/n + 1`. The multiply is exact for every window a pool can have, because the argument
  is under 2^16.

### Lag extent and the tail-only check

The lag is a contiguous suffix of channel groups that ends at the last channel group. Its onset
varies, and is near-constant per shape: group 7 of 48, 44 or 108 of 128, 1 or 5 of 18. The end never
varies. The PPU's divisor state therefore slips once and does not resynchronize inside a program.

The onset's constancy holds at one memory-system load. A change in that load moves the onset (see
"Memory traffic moves the onset and the rate, the other way" below). The end does not move: under an
applied DRAM load every occurrence still reaches the last group.

#### Software support

`librocketnpu` scores the last channel group alone. Scoring it detects every occurrence recorded, at
1/ngroups of the cost, and its detection is indistinguishable from the full check's.

> [!NOTE]
> The tail-only check rests on an observation of how the PPU's divisor state fails, not on a
> register. A lag that stopped short of the last group would be missed. It would leave a full-size,
> wrong output that no redo count or passing test shows. `ROCKET_RK3576_POOL_LAG_TAIL=0` restores
> the full scan.

**Conditions:** H96 MAX M9, Inception V3, `bench 100`, 2026-08-04.

On the default `ondemand` governor the tail-only check takes Inception V3's wall-clock time from
38.2 to 33.4 ms (12.6%). With the governor pinned to `performance`, where the configurations'
differing host work cannot move the clock, the saving is 3.53 ms. The saving is zero on the other
four networks [HW sweep, H96 MAX M9, 2026-08-04]. Their pools are members of chained runs, so the per-call
entry never runs.

### Lag rate per layer in Inception V3

The rate is not a function of the geometry. In Inception V3, the four 768×17×17 branch pools have
the same shape, program and channel count. They fire at 49, 0, 3 and 6 percent.

Every one of the nine padded average pools has thirteen to thirty-seven thousand elements whose two
candidate divisors give different values on its data. A pool that does not fire is therefore not an
artifact of rounding.

Run standalone, the same shapes give different rates from those in the graph, and two of the three
differ in opposite directions. The 17×17×768 pool fires on 0 of 60 calls standalone and 49%
in the graph at the same geometry. No function of the geometry fits.

The rate does not depend on the gap between submits, on the source's channel-group stride, or on
the input data.

### Lag rate and submit structure

The rate depends on whether the pool's program runs inside a chained kick or as its own submit
[HW sweep, H96 MAX M9]. The two configurations below have the same handle, the same register
program, the same source cube at the same BO, and the same data.

**Conditions:** H96 MAX M9, Inception V3, `bench 100`, three repeats a configuration.

| layer | shape | inside a kick | its own submit |
|---|---|---:|---:|
| 110 | 1280×8×8 | 52.6 / 35.7 / 46.7 % | **1.0 / 0.0 / 0.0 %** |
| 125 | 2048×8×8 | 51.2 / 82.1 / 61.9 % | 35.7 / 8.2 / 22.9 % |

The graph's own layers show the same direction. Layer 50 reads a source the host scattered
immediately before the submit, and fires at 49%. Layers 62, 74 and 86 have the same shape and the
same program. They read a cube that an earlier, completed submit wrote, and fire at 0-15%.

The two high-rate configurations share a source written shortly before the PPU reads it
[hypothesis]. In one, the preceding program of the same job writes it microseconds earlier. In the
other, the host writes it between the scatter and the submit ioctl. The low-rate configuration is
separated from its producer by a whole fence. The hypothesis fits every measurement above, and does
not explain the standalone probe. Its source is also host-written immediately before, and it fires
on 0 of 60 calls at one shape and 55 of 60 at another.

**Not characterized:** the mechanism. Only the kick against own-submit comparison is measured.

### Memory traffic moves the onset and the rate, the other way

CPU-side DRAM traffic lowers the lag's rate at a loud shape and spreads its onset
[HW sweep, H96 MAX M9, 2026-08-05]. The traffic is applied from the CPU, touching no NPU buffer and
issuing no ioctl. Memory-system pressure is not what separates loud shapes from quiet ones:

- **The quiet shape stays quiet.** `768x17x17` does not reach 0.2% of calls standalone at any load.
  Pressure does not create the hazard.
- **The loud shape moves the other way, with DRAM traffic and not with core occupancy.** At
  `288x35x35` the rate falls from 92.6% idle to 79.2% under 8.3-8.4 GB/s of copy traffic. A spin
  control on the same cores with no memory traffic sits above idle, at 95.6%. Less DRAM traffic
  means more lag.
- **The onset is a run-time property, not a constant of the shape.** Idle, it is a point mass on
  group 2. Loaded, it spreads across groups g0-g6.

The slip is tied to how fast the PPU's feature stream arrives relative to its own divisor sequencing
[hypothesis]. Slowing delivery then both delays the slip and makes it less likely, which fits all
three readings. It is not a decode, and it does not name what separates loud shapes from quiet ones.

**Not characterized:** pressure applied from the NPU side rather than the CPU's, and anything about
the physical BO or IOVA. The measurement covers one SoC, one pooling mode and three shapes.
### Chained-run refusal for pools whose divisor can lag

**Conditions:** H96 MAX M9, Inception V3, 2026-08-03, with the divisor-lag check scoring every
channel group (before the tail-only check).

**Not characterized:** a graph whose lag-prone pool sits in a long chained run at a low lag rate. There
the split would cost two submits and save nothing. The measurements cover two pools on one
graph.

A pool whose divisor can lag must not be a member of a chained run. Software must
give it its own submit. A chained run is one kick inside one submit, so a lag redo inside it is joint: it re-runs
every program of the chained run. Every observable pool in the chained run must then come out clean on the same
attempt. The redo budget is per hazard, and a joint attempt uses it up geometrically.

With Inception V3's two padded average pools inside one 41-layer kick, the per-attempt joint
clean rate falls to about 0.12. The default budget of 32 redos then runs out. The result is 8
hard failures in 1100 inferences, and none with each pool in its own submit [HW sweep, H96 MAX
M9]. A run that exhausts the budget ends with `the surface was never right` and returns
`ROCKET_E_DEVICE`.

`rocket_chain_plan_rk3576()` refuses such a pool as a member of a chained run, rather than raising the budget.
The lag rate itself moves the same way inside a kick, and is the larger half of the failure. A
larger budget would only cover a hazard that the placement creates.

The refusal applies only to pools that can lag. A max pool, an unpadded average pool and a
count-include-pad average pool have no position where the divisor varies. They stay members of
chained runs. A `conv -> pool -> conv` sequence in one stream is unaffected. The refusal reads the
geometry rather than `_lag_observable()`, so that `ROCKET_RK3576_POOL_LAGCHECK=0` isolates the
check's cost without also changing the chained-run structure. `ROCKET_RK3576_CHAIN_POOL_LAG=1` puts
such a pool back in a chained run, for testing.

The refusal costs nothing. Inception V3 splits its 89-129 chained run into 89-109, 114-118 and 119-124.
With the refusal it measures 40.7-41.9 ms, against 46.9-62.8 ms with one such pool back in a
kick: faster, and with the spread gone. The joint redos are most of the variance: with the pool
in the kick, an inference runs 173-293 tasks against 132. Both ranges are measured with the
column-split pools joined cube-in. The other four graphs have no padded average pool, and the
refusal does not change them.

A lower lag rate does not remove the hazard. Software must keep the lag check and the redo
budget for a pool in its own submit.

### Divisor lag on the cube-in path

The divisor lag also occurs when a padded average pool reads a cube that an earlier submit
wrote. It occurs with one submit per layer and no kick.

> [!CAUTION]
> Without a lag check on the cube-in path, the cube-chained Inception V3 disagrees with the
> row-major lowering at the classifier. It differs in 831-963 of 1001 elements. Software must
> apply the divisor lag check on the cube-in path as well as the row-major one.

The cause is the divisor lag, and neither the kick nor the joins. Holding the divisor constant
across positions (`ROCKET_RK3576_NET_AVGNOPAD=0`) removes the disagreement, with every join,
every program and every chained-run boundary unchanged. With the check extended to the cube-in path,
the cube-chained Inception V3 passes its network test.

> [!NOTE]
> The lag is intermittent. A sweep with one run per test point, such as one over the kick size,
> can read as a race. Repeat each test point before reading a non-monotone result as one.

## Unresolved hazards and limits

### Drain deadline

On `rocket` 1.5.0, a program whose DPU drain outlasts `dpu_grace_us` loses its output tail
(RK3576-E3). [The direct path's drain deadline](#the-direct-paths-drain-deadline) states the
hazard and its scope by module version. [The drain deadline and the row
planner](rk3576.md#the-drain-deadline-and-the-row-planner) states the planner's agreement at a
longer grace and the one-task case.

### Pooling state across processes

**Conditions:** H96 MAX M9, a k3 s2 max pool at 16 channels as the next process's first pooling
submits, 2026-08-02.

Besides the state it leaves for the next process (Erratum RK3576-E9), pooling with a unit kernel
on one axis is unstable in its own output.

**Not characterized:** which register or state contains the corruption, and whether a
non-pooling submit in between clears it. Whether that instability and this state are one defect
is not characterized, nor whether any geometry a real graph runs can produce it.

### A pool at the start or end of a chained run

`librocketnpu` refuses a pool at the start or the end of a chained run. A pool that began a chained run
would need the chain to scatter a row-major tensor into its cube. One that ended a chained run would
need its output de-tiled. Both are host work the stream exists to remove, so the limit is
the library's rather than the hardware's. No measured graph needs it: both MobileNets and
ResNet-18 pool in the middle.

**Not characterized:** a pool at the start or the end of a chained run.

## Evidence

The method, controls, excluded causes and run detail behind each section are in
[rk3576-validation.md](rk3576-validation.md), under the same titles. They run from §"The
wide-output hazard" to §"Unresolved hazards and limits".
