# RK3576 validation record

How the facts in [rk3576.md](rk3576.md) were established: the validated configuration, the
sources, and the measurements, controls and excluded causes behind each section. The sheet
states each fact. This record states its evidence, under the sheet's section names.

## Validated configuration

The port is validated on kernel 7.1.6 with the whole RFC series in-tree (`0001`-`0024`, uAPI
interface 1.5). All 57 tests return `rc=0` (57 of 57), the taint is unmoved, and `dmesg` shows
zero WARNINGs. The five graphs run at 4.9 / 6.5 / 6.6 / 7.0 / 33.8 ms, indistinguishable from
the same list on an out-of-tree `0023` module. So `0024` (do not let a stale poll retire the
next job) costs nothing measurable and breaks nothing [HW sweep, H96 MAX M9, 2026-08-04].

The uAPI header assembled from the series' own patch hunks is byte-identical to the one that
kernel installs. That match shows the userspace and the loaded module agree on the interface,
beyond both building. [HW sweep, H96 MAX M9, 2026-08-04]

## Sources

The sheet draws on external reverse-engineering, on the RFC, and on what an H96 MAX M9
confirms live. The external sources are:

- Ga Hing Woo's mainline bring-up (blog + `github.com/gahingwoo/linux-rk3576-npu`, including
  `notes/rk3576-npu-values.md`)
- The RFC series "accel/rocket: RK3576 NPU (RKNN) enablement" (Jiaxing Hu, patchwork
  `linux-rockchip` series 1129921, 8 patches, 2026-07-18)
- The Radxa RK3576 vendor kernel `linux-6.1-stan-rkr5.1`

The gahingwoo repo's documents the sheet relies on are:

- `vendor-capture/RK3576_CNA_MAP.md` and the CORE/DPU map
- `predict.py`
- `FINDINGS-FLOATSURFACE.md`
- `MATMUL-PIPELINE-ANALYSIS.md`
- `vendor_regcmd_full.txt`, a full 139-entry vendor register program

No other public effort covers the two-core case. The one other mainline RK3576 NPU bring-up
(gahingwoo, `linux-rk3576-npu`, ROCK 4D) runs a single core. It stops at the chained-task
defect that `patches/rk3576/npu/0008` closes.

## SoC integration

### Core work state

A 20 s loop took 68.6 M reads of `NPU_GRF_RKNNST` at 292 ns each, and no read had a bit set. The
load was a resident fp16 matmul bench with a minimum device time of 35.9 ms a call at
512×1536×8192 [HW sweep, H96 MAX M9, 2026-09-27].

The control for that negative is the rest of the NPU GRF. Every other documented word read its
TRM reset value, the non-zero ones included. The read therefore decodes, and the zero is the
register's content. The instrument is `rocket-userspace/tools/rk3576-npu-grf.c`. The method does not reach
a bit behind an enable no driver sets, or a state only the NPU MCU raises.

### Power-on settle

The delay at 15 us survived 536 cold power-ons [their HW sweep, gahingwoo board log r420, ROCK
4D, 2026-09-24]. Each failing configuration is one boot, and no value between 0 and 15 us was
tried.

## Clocks, resets, power

### Clocks and resets

The four-clock core 1 node in the upstream RFC device tree reads as an oversight nobody upstream
would have hit. The RFC sets both cores `status = "disabled"`.

The `reset-names` binding was read live. It confirms what the vendor-kernel extraction left
unverified.

### NPU clock rate

At ~786 MHz with no `assigned-clock-rates`, `scmi_clk_npu` / `clk_rknn_dsu0` were sampled once
per second across a 200-submit loop and after it drained [live]. At
`assigned-clock-rates = <594000000>` they read 594000000 Hz at rest and in 91 samples across two
running networks [live 2026-10-02, kernel 7.2.8-1].

The 594 MHz column of the network table is four or five runs a network, measured with
`rk3576_net_gate bench 100` [live 2026-10-02, kernel 7.2.8-1, `rocket` 1.6.0].

### NPU supply rail

With the rail switched by the power domain, a snapshot 12 s into a network's all-passes run found
the domain off and the rail disabled. Across a `bench 100` run, 80 of 81 samples found both on. A
job submitted from rail-off computes, and the seven jobs that follow it match it byte for byte
[live 2026-10-02, kernel 7.2.8-1].

Before the first job of a boot, `npu`, `nputop` and `npu0` had a genpd `active_time` equal to a
49-hour uptime. The `npu1` domain's was 143 s [live 2026-10-02, kernel 7.2.8-1].

## Memory system, NPU rail, and thermal envelope

### DRAM bandwidth

The RK3576 design guide's §2.1.7.1 reads "a 32-bit data bus width, composed of 2 16bits DDR
channels". The RK3588 guide reads "64bits data bus width, composed of 4 16bits DDR channels".

The int32 writers' atom drop was measured under a load ladder. An integer spin (11.7%) and an
L1-resident `memcpy` (9.2%) barely move it, against 51.7% under a host streaming `memcpy` to
DDR. The single-thread `memcpy` figure of 4.4 GB/s is from `build/membench`, 256 MB.

#### Excluded causes

- *Thermal.* The load ladder rules it out as the confound: integer spin heats the CPU at least
  as hard and does not move the rate.

The RK3588's 2112 MHz is the note "Frequency 2112MHz" on the `VDD_DDR_S0` row of the RK3588
Hardware Design Guide V1.4's peak-current table (Table 2-13: 0.870 V, 2.50 A, 2.18 W, measured
at a 100 C junction on Rockchip's own board) [source-confirmed]. `VDD_DDR_S0` is the DDR supply
rail, and the table gives the DDR frequency at which that rail's peak was taken.

### NPU supply rails

A register sweep places the wide-output hazard's state in an NPU-internal memory array. In
that sweep, the canary already writes both of the implicated bits at their int8 values.

### DPU write drain

The drain bullet stated from 2026-07-29 that the 745 us drain of the int8 stem (at 224×224 k7
s2) "is why `dpu_grace_us` has to default to 500". The 745 us exceeds 500 us, so the default
was not set by it. On `rocket` 1.5.0 the 500 us default is 2.5x the measured failure point of
the poll that retires on the DPU bit (a 32×2048×3072 output fails at 200 us and passes at
250 us). Only the drain of the last submit before the host reads has to fit it. The
745 us stem passes at a grace of 200 us [HW sweep, H96 MAX M9, measured 2026-07-28,
rk3576-regcmd.md §"Tasks that raise no DPU completion"]. On `rocket` 1.6.0 the default is 0,
which waits for the writing block.

### NPU thermal throttling

| Reading | Workload | Date |
|---|---|---|
| 44-49 C | Submit loops during convolution bring-up, kernel 7.1.3-1 image | [live 2026-07-25] |
| 40-43 C, every cooling device at state 0 | Sustained test runs | [live 2026-07-29] |

The 2026-07-25 reading entered the sheet with the remark that the board was "limited to
probing, so nothing here is a performance measurement". The thermal measurement of 2026-07-29
found no throttle between idle and the 115 C critical trip. The remark was withdrawn from
the clock section that day. It no longer scopes any timing in the sheet. Each timing carries
its own conditions.

## Two-core operation

### Second-core throughput

A second core is measured against a one-core control, not against one fd. With both cores
bound, two fds give 1.55-1.74x, 1.49-1.68x and 1.38-1.42x over one, which matches or is below
the one-core control in the sheet's table. One fd does not saturate a single core at the ~340 us
per-submit overhead. Filling its dispatch gaps is the whole gain
[HW sweep, H96 MAX M9, 2026-07-28, `tests/rk3576_multicore.c`].

### Two jobs in flight

The condition is simultaneous execution. It is not the second core's existence, not anything a
job inherits from the one before it, and not the mapping. A mutex can hold the pair to one job
in flight, with two fds, both cores bound and jobs alternating cores. Under that mutex every
call is exact: 96 calls a test point at 256×512×2048, 32×1024×512 and 128×1024×1024. The same
two fds driving at once are wrong on 96-100% of calls. The wrong calls cover as much as 94% of
the output, with deltas spanning the whole int8 range
[HW sweep, H96 MAX M9, `tests/rk3576_core_pair.c pair`].

The correspondence with overlap is one to one, and it is read from the driver rather than
inferred. Both cores report runtime-active time for the whole run in every test point, because
both stay resumed throughout. That shows they are powered, not that a test point's jobs were
spread over them. A test point whose jobs all ran on one core is the one-core control under a
different label. [HW sweep, H96 MAX M9]

An instrumented `rocket` counts jobs per core, domain swaps per core, and jobs started while
another core already had one in flight. At 256×512×2048 one library call is one job. The
concurrent test point runs 48 jobs a core with 48 overlap events, every job in a simultaneous
pair, against 96 of 96 calls wrong. Every clean test point records zero overlap.
[HW sweep, H96 MAX M9]

The vulnerable window is the job's own duration, so the contended resource is held for the
whole job rather than for a phase. The stagger test delays one worker past a common barrier,
one call each per repetition. The share of wrong calls is 100% at +0 and +100 us, 91.7% at +250
and 4.2% at +500. None is wrong from +750 us out to +5 ms. The transition is sharp, and the
driver's overlap count reaches zero at the same step. The per-submit overhead is ~340 us, so the
window is the submit [HW sweep, H96 MAX M9, `tests/rk3576_core_pair.c stagger`].

The corruption crosses processes. A library mutex cannot serialize two processes holding one fd
each. They corrupt exactly as two fds in one process do, with 96 of 96 calls wrong. A single
process doing the same total work gets zero wrong.
[HW sweep, H96 MAX M9, `tests/rk3576_core_pair.c xproc`]

### Excluded causes

- *Residual per-core register state.* Anything a program inherits from a previous one would
  corrupt the serial test point too, and the serial test point is clean. That includes the
  differing-weights test point, where the two workers hold different weights. That test point
  is the one that can see a worker being handed the other's operands. The plain matrix gives
  both workers the same operands, so the differing-weights test point is the one that tests
  inputs.
- *The two register banks being one bank aliased twice.* They are distinct hardware. Different
  values written into core 0's and core 1's PC `TASK_DMA_BASE_ADDR` and CNA `S_POINTER` each
  read back as written.
- *The IOMMU domain.* Three results exclude it. Each core has its own MMU (`27702000` and
  `2770a000`) in its own group, so no attach can displace the other's. With every context
  sharing one domain, the counter confirms zero domain swaps in every test point. The
  concurrent test point stays at 96 of 96 wrong. A page-table mix-up would also show in the
  differing-weights serial test point, which is clean across 36 domain swaps a core.
- *The vendor's RK3576-only `state_init` and its `pc_dma_ctrl` PC serialization.* The two were
  applied together as one delta, because a one-at-a-time pass could not have found a condition
  of two. The delta seeds both CNA pointer groups at every power-on and holds a device-global
  lock across the whole register-programming sequence. Under it the concurrent test point is
  100% wrong, against a 97.9% control. The seeding is counted, so this is a negative about the
  vendor's sequence and not about a setting that never took effect. Since the whole delta does
  nothing, no subset can.

### The shared CBUF

The CBUF is the one datapath resource the TRM names as shared ([TRM], in the sheet's list).
The vendor driver is the only Rockchip NPU configuration with a `cache_sgt_init`. Its table is
`block_size_kb = {448, 64, 448, 64}` at `block_offset_kb = {0, 896, 448, 960}`, indexed
`[i * core_num + j]` with `core_num = num_irqs = 2` [source-confirmed: `rknpu_drv.c:123`].
`librocketnpu`'s 448 KiB pool was measured with core 1 unbound and with no knowledge of the
vendor table.

On the RK3588, `CNA_CBUF_CON0` has `DATA_BANK`, `WEIGHT_BANK` and `FC_DATA_BANK` bank selects
over 16 banks. On the RK3576 the same register has the granule allowance F. Both cores staging
into the same banks is a hypothesis with a named mechanism. The 448 KiB coincidence
points both ways. A hardware per-core partition would mean the two cores *cannot* collide, which
would exclude the candidate. One core's addressing reaching 448 KiB of a shared pool would
confirm it.

Two distinctions limit what the vendor table shows. The word `cbuf` appears nowhere in the
vendor driver. The driver implements the partition as an allocator plus an IOMMU mapping
(`rknpu_iommu_map_with_cache_sgt`, `rknpu_remap_pfn_with_cache_sgt`), never as a register
write. Whether a job's buffers are placed in that region is an alloc-flag opt-in
(`rknpu_gem.c`, the `nbuf_size > 0` branch). So `cache_sgt` governs CBUF-as-allocatable-memory,
a different use of the same RAM from CBUF-as-staging. It is not what keeps vendor staging from
colliding: an external RK3576 YOLOv8 benchmark, below, runs two per-core contexts correctly
*without* the SRAM opt-in.

An external benchmark measures two-core concurrency. A reComputer RK3576 YOLOv8 run
(hackster.io/wurengmin) puts two `spawn` processes on separate cores, through
`init_runtime(core_mask=NPU_CORE_0)` against `NPU_CORE_1`. It releases them together through a
barrier, and reports ~2x throughput with plausible detections over 394 frames x 5 repeats x 3
model sizes. Its script has no numerical check. It records output shapes and timings and never
diffs outputs across configurations. It establishes that correct two-core concurrency is
reachable in software on this part, not bit-exactness.

On the CBUF hypothesis, the per-core partition `rocket` lacks would be in the *register
program*. A fix would therefore need the core identity to reach the regcmd encoder, not just a driver
patch [hypothesis].

### CBUF feature staging

The symmetric test points give both workers the same shape, so none of them can separate a
capacity from a control path. The asymmetric sweep holds one worker fixed as a victim
(256×512×2048, 2048 feature granules) and sweeps the other's footprint as an aggressor. It runs
`tests/rk3576_core_asym.c` with 160 victim calls a test point, over two runs
[HW sweep, H96 MAX M9, measured 2026-07-28]:

| aggressor | feature granules | weights | victim calls wrong | output destroyed per wrong call |
|---|---|---|---|---|
| tiny  | 4    | 4 KiB    | 71.9-75.6% | 0.6-0.7% |
| small | 32   | 16 KiB   | 56.2-70.0% | 2.7-3.0% |
| mid   | 256  | 64 KiB   | 22.5-32.5% | 12.9-16.9% |
| large | 1024 | 256 KiB  | 20.6-22.5% | 27.9-38.9% |
| equal | 2048 | 1024 KiB | 95.6-98.8% | 48.5-55.3% |

The trigger is not capacity. An aggressor whose whole feature plane is four granules, 256 bytes
against a 448 KiB pool, corrupts about three quarters of the victim's calls. Nothing is being
exhausted, so the collision is unconditional.

The extent follows the feature plane, not the weight path or the occupancy. Footprint and
submit duration co-vary across those five rows, so two more shapes separate them. The `wideN`
shape (4 granules behind a 128 KiB N-tiled weight cube: a small plane and a long submit)
destroys 5.6-6.5%. The `flatN` shape (a 2048-granule plane against 32 KiB of weights: a large
plane and a short submit) destroys 30.4-34.9%. That is five to six times the damage from four
times less weight and a shorter occupancy, with 512x the feature plane.

The two cores therefore stage their feature planes into the same CBUF, unconditionally, and an
intruder destroys however much plane it staged.

A test point is a measurement only once the driver's overlap counter is non-zero. A small
aggressor running a fixed call count finishes inside the victim's host-side packing. One run of
this sweep recorded zero overlap for every asymmetric test point for that reason. The
aggressor drives continuously for the victim's whole run.

### CBUF placement

Placement does not separate the cores, so the collision is not two jobs writing the same CBUF
granules. The CBUF base is a real, honored address, and moving it is not enough. The field that
moves a job is already decoded and already emitted, so no separate bank-select field remains to
find. The row-reuse path drives the granule offset. Every plain entry point leaves it at zero,
so every job stages from granule 0 (`tests/rk3576_cbuf_base.c`, `ROCKET_RK3576_CBUF_BIAS` /
`rocket_rk3576_set_cbuf_bias()`) [HW sweep, H96 MAX M9, measured 2026-07-28]. Two jobs can be
placed apart only when their planes together fit one window.

Placed apart, the two jobs corrupt exactly as they do at a shared base. The victim is
256×512×2048 at base 0 (2048 granules), against the `tiny` aggressor with its base swept clear
of the victim. The share of victim calls wrong is 78% at base 0, and 68-72% at bases 2048,
3072 and 4032. That is flat, inside the run-to-run scatter, with the driver's overlap counter at 57-60 a
test point throughout.

The bias reaches the emitter on the worker thread. A base past the aggressor's *own* bound
(4156 against a bound of 4092) takes it to 896 of 896 calls self-wrong, where every scored row
leaves it at 2-4%. Without that positive control, "biasing changed nothing" and "the bias never
reached the emitter" are the same table.

A sweep of the only `CNA_CBUF_CON0` bits `librocketnpu` never varies leaves the register fully
accounted for: nothing else in it is a bank base.

No userspace-reachable partition of the CBUF was found, and an address partition alone would not
have helped. The mechanism this leaves is a shared feature-staging datapath rather than shared
storage [hypothesis]. This IP's ancestor gives the CBUF four ports, feature-write and
weight-write among them separately [nvdla.org/hw/v1/hwarch.html]. Contention on one shared
feature-write path predicts all three measurements at once. They are the extent proportional to
the feature plane, the indifference to weight bytes and to occupancy, and the immunity to
placement.

**Not characterized:** whether the two cores' windows sit at the same physical base. Reading the
CBUF SRAM at `0x3fe80000` while a job runs would show it. That reading is a `/dev/mem` poke at
an SRAM in the NPU voltage domain. The domain must be held resumed during the read, or the SoC
hangs. The other test decisive on the mechanism diffs the vendor's live submit program for
one model at `core_mask = NPU_CORE_0` against `NPU_CORE_1`. It needs a vendor-BSP RK3576 with an
rknpu regcmd dump, and the H96 is mainline.

### The six-register CBUF window

An externally read CBUF window is six registers, and `librocketnpu`'s bias moves two of them
[hypothesis, not measured here]. The `charsiu` project derives the vendor's CBUF window from a
start bank S, off a 2026-09-04 census of the Llama-3.2-1B `.rkllm`. In that census every op
appears twice, and the two copies differ in exactly six registers (`src/job.c`,
`charsiu_cbuf_window`):

| register | window 0 | window 1 | as a function of S |
|---|---|---|---|
| `0x1018` | `40000404` | `4000040b` | low byte `4 + S` |
| `0x1038` | `00000007` | `0000010e` | end bank, `\| 0x100` for S > 0 |
| `0x103c` | `xxxx0000` | `xxxx1c00` | low half `S*0x400` |
| `0x1040` | `10000000` | `2c001c00` | `(S*0x400 + 0x1000) << 16 \| S*0x400` |
| `0x2818` | `00000000` | `1c000000` | `S*0x400 << 16` |
| `0x2820` | `00000000` | `00000038` | `S*8` |

On that reading the CBUF is two mirror-image windows of 7 banks inside 14 (`0x3800` entries),
and the `charsiu` runtime hands each core its own (`charsiu_job.cbuf_window = di`). The
registers `0x2818` and `0x2820` appear nowhere in `librocketnpu`, neither emitted nor
documented. The `ROCKET_RK3576_CBUF_BIAS` setting moves only the low halves of `0x103c` and
`0x1040`. It moves two of six, and not the end bank, so it moves a window's base while leaving
its extent at bank 7. That shape predicts exactly the flat corruption measured above.

The placement negative therefore rests on moving two registers of a six-register bundle. The
positive control proves the bias reached the emitter. It does not prove the bias was the whole
unit. Float mode on this part follows the same rule: three registers move together.

**Not characterized:** emitting all six as a function of S, with the two cores given different
S, under `tests/rk3576_core_asym.c`. Until that runs, the shared-feature-write-datapath
hypothesis above is one of two open explanations.

### Undervolt

An undervolt is a competing explanation, and the two-core corruption was measured at the
operating point it names [hypothesis, not measured here]. The `charsiu` project reports the same
class of fault from two cores in flight, and removes it entirely by moving the rail or the
clock. In that fault the job on the second core writes one word of a row wrong, the right value
plus 1024. The error is a few bits around bit 10 of the accumulator. It hits one row in a few
thousand, always at the same array position. Their sweep runs four DTBs on one board, with the
kernel and binary held, in four passes of 5400 rows each:

| NPU clock | `vdd_npu_s0` | wrong words a pass |
|---|---|---|
| 786 MHz | 750 mV | 11 to 25 |
| 594 MHz | 750 mV | 0 0 0 0 |
| 786 MHz | 800 mV | 0 0 0 0 |
| 786 MHz | 850 mV | 0 0 0 0 |

With no `assigned-clock-rates` on its nodes, this NPU runs at 786.43 MHz off the audio PLL, the
rate the boot firmware leaves. It has no OPP, and the rail sits at the U-Boot default of 750 mV.
The upstream series assigns 594 MHz in the SoC device tree. Whether 786 MHz at 750 mV is under
the vendor's voltage depends on the chip's bin, which no mainline kernel reads. The vendor's
800 MHz step asks 800 mV of the lowest bins and 725 mV of the highest (per-bin table under
[Memory system, NPU rail, and thermal envelope](rk3576.md#memory-system-npu-rail-and-thermal-envelope)).

For the two-core runs the H96 MAX M9 read 750 mV, `enabled`, `num_users=1` and 786431952 Hz on
`scmi_clk_npu` (debugfs, root only). Neither NPU node had `assigned-clock-rates`
[live 2026-09-25, kernel 7.2.3-1]. With `assigned-clock-rates = <594000000>` on both nodes, the
same board reads 594000000 Hz [live 2026-10-02, kernel 7.2.8-1]. That image has core 1
`disabled`, so no two-core run exists at 594 MHz here.

How the rail is held is a variable too. The two-core runs had `vdd_npu_s0` pinned by
`regulator-always-on` on the PMIC node, and `num_users=1` was that pin alone. The regulator core
counts an always-on regulator as one user [source-confirmed: `drivers/regulator/core.c`
`set_machine_constraints()`]. An `npu-supply` on a core node is inert, because `rocket` calls no
regulator API.

The upstream series hands the rail to the NPU power domain as `domain-supply`, with `npu-supply`
on both cores and no `regulator-always-on`. The series reports the rail then following the
domain, at `use_count` 0 and disabled at rest. The H96 MAX M9 reads the same under that
arrangement. At rest the rail is `disabled` with `num_users=0`, and with a job in flight it is
`enabled` with `num_users=1` [live 2026-10-02, kernel 7.2.8-1]. "Clocks, resets, power" in the
sheet has the detail.

A pinned rail and a domain-switched one are not the same electrical condition, and neither
project has held that difference constant against the other. The `charsiu` fix is a runtime
check rather than a serialization. Their `overlap_safe()` reads the rail from sysfs and the
clock from debugfs, and checks both against the vendor's table at its lowest bin. At 800 mV they
overlap every batched call, with nine models identical to the serialized configuration.

The `charsiu` fault and the one here are not obviously one mechanism. Their fault is tens of
wrong words a pass. The fault here destroys 71.9-75.6% of a victim's calls and scales with the
aggressor's feature plane, which an undervolt does not straightforwardly predict. But the two
negatives are on the same silicon at the same operating point, and the one here was never run
at any other.

At 594 MHz an external measurement runs two jobs on two fds side by side [their HW sweep,
gahingwoo board log r419, ROCK 4D, 2026-09-15]. Two jobs on one fd take twice one job's time,
and two jobs on two fds take what one job takes:

| Shape | One job | Two jobs on two fds |
|---|---:|---:|
| k 1024 n 256 | 49.77 us | 52.43 us |
| k 1024 n 1024 | 133.64 us | 135.66 us |
| k 2048 n 2048 | 489.63 us | 471.88 us |

Each configuration ran 200 times, all four back to back in one process. The log reports no
corruption from the overlap.

**Not characterized:** `tests/rk3576_core_asym.c` at 594 MHz, or at 750 mV against 800 mV. Either
needs core 1 enabled in the device tree, and a 786 MHz control under the same rail arrangement.

### The per-core `S_POINTER` bit

A third explanation is the per-core bit in `S_POINTER` [expected, untested]. The RK3576 regcmd encoder
writes CNA and CORE `S_POINTER` from the regcmd, as `0xE`, or `0x30` for a LUT program. That
drops the per-core bit, `0x10000000` times the core index, which the vendor `rknpu` driver and
mainline `rocket` write before every task. On the RK3588 the same write hangs every job that
runs off core 0, 24 of 24 ([../encodings/regcmd-task-model.md](../encodings/regcmd-task-model.md)).
On the RK3576 serialized core-1 jobs are exact, so the bit can matter only with both cores in
flight.

**Not characterized:** dropping the two writes, confirming the kernel's value stands, and
re-running `tests/rk3576_core_pair.c pair` with both cores driving.

### Corruption pattern

The corruption is gross, not sparse. Checked against a CPU model of the DPU's integer requant,
it instead reads as "a handful of positions off by exactly +/-2". That model disagrees by one
count on a fraction of a percent of elements in every configuration, one core included. The
disagreement is noise larger than the signal the model was used to find. A solo run on the SoC
is the reference for a concurrency question, because the requant gap cancels against it.

Part of that gap is the tie rule. The convertor rounds ties to even, where the models used for
this check rounded half up ([out-cvt-converter.md](../encodings/out-cvt-converter.md)). A solo
run cancels the model's remaining error, whatever it is, and a model does not.

With unit operands and an identity requant, the delta reads directly in MAC terms. The wrong
elements at the boundary of the damage are off by two or three terms, while the bulk are off by
far more. That pattern is a staging corruption rather than an unwritten output. With each worker
given its own weights, 3.3% of wrong elements hold the *other* worker's correct value, against
0.4% by chance. That is cross-talk but not a clean swap, so the dominant effect is neither job's
arithmetic.

### Shared power domain

The shared power domain was measured on the H96 MAX M9 with jobs placed on both cores
[HW sweep, H96 MAX M9]. The domain collapses only once both cores are runtime-suspended. Either
core can be the active one at any instant.

## Driver and device-tree state

The core 0 and MMU addresses and the interrupts are confirmed on live silicon by `rocket`
binding and computing there. Core 1's address is confirmed by reading it, not by the node
existing. A DT unit-address is only a claim, and the RFC's was wrong.

### Memory-safety fixes

The image that contains `0001`-`0012` and none of `0013`, `0014` or `0020` is the kernel
7.1.3-1 image of the section's Conditions (2026-07). The validated 7.1.6 image of 2026-08-04
contains the whole series, `0001`-`0024`. The Note's "Both patches" was written on 2026-07-28,
when the list held `0013` and `0014`. `0020` was added to it later.

### Register banks

The `/proc/iomem` read is the 7.1.3-1 image's alone. The later images' bank mapping is read from
the source rather than from a board. `rocket_core_init()` calls
`devm_platform_ioremap_resource_byname()` for `pc`, `cna` and `core` only. The RK3576 support
patch `0006` leaves the core with those three, and no later patch of the series through `0034`
adds one.
The completion reads the series adds, `INTERRUPT_RAW_STATUS` and the task-status counter, are
in the `pc` bank. The device-tree nodes of `0007` list all five banks.

## Dispatch path

The library `librocketnpu` builds unmodified on the RK3576 (gcc 15.3 / cmake 4.3, `libdrm-dev`
plus the kernel's own `drm/rocket_accel.h`), and its fp16 matmul test submits successfully. The
dispatch path was established with the RK3588 regcmd encoder driving RK3576 silicon. No `dmesg`
error, job timeout or IOMMU fault appeared across hundreds of submits (M=4/K=32/N=16 and
M=64/K=128/N=64) [live].

The output BO came back untouched, still the 0xAA sentinel, so there was no write at all. The
empty BO was the predicted consequence of the re-packed geometry registers [live]. Compute being
gated on the geometry registers alone is what made the int8 conv regcmd encoder the whole of the bring-up
work.

## CNA register encoding

### The published map

Ga Hing Woo derived `vendor-capture/RK3576_CNA_MAP.md` by converting the same Conv2d to `.rknn`
one parameter at a time and diffing the vendor RKNN regcmd. A `predict.py` then validates it,
reproducing 34/34 geometry / channel / tiling / const registers with zero mismatch across 7
MobileNetV1 captures. The map covers the vendor-toolkit conv path, cross-referenced against
what Mesa emits with the RK3588 map. `librocketnpu` has not exercised the map itself on its
matmul path, or run it on hardware.

From 2026-07-25 the sheet's CNA delta table and its RK3588-gaps subsection took four readings
from the published `RK3576_CNA_MAP.md`. Two sections of rk3576-regcmd.md correct them:
§"Corrections to the published map" and §"Unnamed registers and their RK3588 equivalents".
`librocketnpu`'s regcmd encoder (`src/npu_regcmd_rk3576.c`) programs them in the corrected form:

| Offset | Published reading, as the sheet stated it | Corrected reading |
|---|---|---|
| `0x103C` hi, `0x1044` lo | `surf` = `max(in_w/4, ic/2)`, a "CBUF surface stride" | `ceil(iw*ic/64)`, the CBUF data entries (the depthwise capture: 56 at `iw` 112, `ic` 32) |
| `0x1048` | `0x00071C70`, with Mesa's RK3588 `0x000E38E0` "2x, wrong", read as a CBUF data-entries word | The CVT control word, packed as RK3588 `CNA_CVT_CON0` (`0x104C`). The normal path programs `0x0B`, the converter bypassed. `0x000E38E0` is the ARGB path's own value, truncate 14 in each of three live lanes (`r76_argb_cvt_con0()`) |
| `0x108C` | `0x000F000F` const, RK3576-only | The weight and data burst lengths, RK3588 `CNA_DMA_CON0` (`0x1078`) |
| `0x1080` | The "surf-stride formula", left open | The pad word, `pad_right<<24 \| pad_bottom<<16 \| pad_left<<8 \| pad_top`, in closed form |

### The feature-data address

`0x1070` reads zero in every capture, including the one program that provably needs a non-zero
feature address. That program is a row-slice whose `0x1088` contains exactly that slice's byte
offset. The
feature-data address is therefore `0x1088` on both CNA datapaths. Only the weight address
(`0x1110`) coincides with the RK3588.

## What the RK3576 map reveals about RK3588 gaps

The RK3588 equivalents of the RK3576 line stride and surface stride were found by dumping the
`librocketnpu` RK3588 matmul regcmd on-device [HW sweep, RK1]. The dump shows the equivalents
exist, and that the generator already emits them.

The subsection listed "A CBUF data-entries word (`0x1048`)" among the gaps. It listed `0x1048`
and `0x108c` among the "RK3576-only offsets" with no RK3588 counterpart. It closed with: "The
`0x1048` slot, where Mesa's RK3588 value is 2x the RK3576 one, matters only to a port to the
RK3576 (Mesa's CBUF-entries arithmetic misapplied). It is not an RK3588 defect." Both offsets
are RK3588 registers at a new offset (the table above).

The subsection paired the RK3588 matmul program's surface stride (`CNA_DMA_CON2`, `0x1080`)
with RK3576 `0x103c`, the published map's "CBUF surface stride". `0x103c` is the CBUF data
entries. The RK3576's DDR surface and channel-group stride is `0x1094` [HW sweep, H96 MAX M9,
rk3576-regcmd.md §"The DPU epilogue and the operand DMAs" and §"Window addressing"].

## The CNA's kernel word

Every test point of `rk3576_conv_lib_gate` sets `d.kh = d.kw = k` throughout. Every kernel in
the first four networks is square, as is every vendor capture the register was transcribed
from. The transposed assignment therefore produces plausible output on every one
of those test points. The Inception V3 graph is the strongest available form of the
experiment. `ROCKET_RK3576_KSWAP=1` is kept as the control [HW sweep, H96 MAX M9].

## The symmetry audit on the CNA

Every test point of `rk3576_conv_lib_gate` sets both pairs square, so that gate does not drive
the convolution block's two remaining axes. `tests/rk3576_conv_sym.c` drives them as pairs at
one rectangular plane. A transpose maps each test point of a pair onto the other. A transposed
field therefore fails both halves, and a formula keyed on the wrong axis fails one.

The non-square strides are 12 of 12 test points bit-exact [HW sweep, H96 MAX M9]. Nothing in
the corpus has a non-square stride, so the axis was unverified rather than suspected. The result
is a negative.

The rule that offered the 256 and 512 rungs wherever `kh == 1` came from a measurement that held
`ic` at 32 and moved the kernel. With the planner keyed on the footprint,
`regcmd_rk3576_gate`'s sixteen planner test points and `rk3576_conv_lib_gate`'s 244 are
unmoved.

The probe's own control located the finding. `k1x1` is the configuration that must succeed,
because it is where the rungs were measured to deliver. It failed, so the finding concerns the
control and not the `k1xN` test points the probe was written for. Three walks mapped it:

- The F=0 boundary, walked row by row
- The same granule total, reached through different (row size, row count) factorizations
- The same planes forced under the boundary, as the configuration that must be exact

## Regcmd encoder

The test `tests/regcmd_rk3576_gate.c` diffs the regcmd encoder register for register against the vendor
programs. The test `tests/rk3576_refusal_gate.c` holds the refusal at the RK3588 generators.

The driving logic for the RK3576 conv regcmd encoders (the BO management, the operand packing, and the
`ic` and row planning) is in the library: `src/rocket_conv2d_rk3576.c`, with the planners in `src/npu_regcmd_rk3576.c`.

From 2026-07-25 the sheet said the RK3576 regcmd encoder was "for the int8 conv path". From the
morning of 2026-07-27 it said that "only the test harnesses and `rocket_matmul_rk3576.c` reach"
the RK3576 conv regcmd encoders. It said that `rocket_conv2d_int8()` "and its siblings" refuse on the
RK3576. It also said: "The entries that have their own encoder (`rocket_matmul_int8_rk3576()`
and its int32 sibling) dispatch ahead of it."

From the afternoon of 2026-07-27 the RK3576
convolution is behind the library's public entries. In the current `rocket-userspace`, the
per-chip dispatch point in `src/rocket_conv.c` (`conv_is_rk3576()`, `conv_wrong_encoding()`)
routes `rocket_conv2d_fp16()` and `rocket_conv2d_dw_int8()` to their RK3576 entries and has
`rocket_conv2d_int8()` refuse, naming `rocket_conv2d_int8_rk3576()`. `src/rocket_matmul.c`'s
`mm_wrong_encoding()` does the same for the matmul entries. `include/rocket_hw_profile.h`
lists the regcmd encoder's datapaths. They are int8 direct and depthwise conv, fp16 conv, the
packed-image first conv, the int8, int32 and fp16 matmuls, pooling and the LUT activations.

## `PC_TASK_CON` task-number width

### Back-to-back submits

An int8 conv was submitted back to back with no gap. Without the per-SoC width, 1 of 8 submits
produced output. With it, 8 of 8 did (32 of 32 on a longer soak), byte-identical to each other
and bit-exact against a CPU model. A row-windowed 112×112 k3 conv is bit-exact over the whole
output with no inter-task gap [HW sweep, H96 MAX M9, measured 2026-07-25].

### Control-bit sweep

Sweeping the control field at each width isolates the cause to `TASK_NUMBER` alone. With the
field correctly placed, every control-bit value tried (`0x0`-`0x3`, `0x5`-`0x7`) passes. With it
misplaced, every non-unit task count fails [HW sweep, H96 MAX M9, measured 2026-07-25].

### External report

A different open userspace (mesa/rocket + Teflon) first reported the behavior. There, a chained
`task_number=N` submit produced an empty result for every non-first task, and splitting into
independent single-task submits did not help. That effort concluded the arming was
per-power-session rather than per-submit [external, gahingwoo `MATMUL-PIPELINE-ANALYSIS.md`].
Reproducing it on the `librocketnpu` regcmd encoder and submit path places it in the driver rather than
in the register program. See [rk3576-regcmd.md](rk3576-regcmd.md).

## Convolution datapaths

The device enumerates, and jobs submit and complete. The clock and power behavior are
characterized, and an RK3576 int8 conv regcmd encoder exists and runs on the SoC.

### Direct int8 convolution

With the RK3576 regcmd encoder, a k=3 SAME 32×32 int8 conv at IC=OC=32, driven by a dense random weight
set, reproduces the CPU model bit-exactly. It matches over the whole output (32768/32768,
reproducible). An unpadded conv over a dense feature ramp is bit-exact as well (28800/28800).

The row-window split reproduces the CPU model bit-exactly over the whole output at every shape
tried. Driving the row window determined the three registers that take a second value only on a
windowed program.

The direct path is swept gap-free. With a `rocket` that has the per-SoC `PC_TASK_CON` width, the
whole tested range runs in one NPU power session. It runs with no inter-shape gap and no
cold-start retries. Running in one session removes the doubt the retry compensation attached to
every result taken through it. The sweep passes 102 shapes in 4.6 s, 67 of them direct. It covers:

- The tested range
- The row window as a matrix
- The channel-group jump across group count and kernel size
- The weight-slice boundary, asserted in both directions

The sweep sharpened the two limits the sheet states: the resident weight slice at
`ic*kh*kw <= 4608`, and the row planner's API trap.

The per-window cost of about 1.2 ms is the 2026-07-25 measurement of a 64×64 ic64 oc64 k3 SAME
convolution cut into 2-32 windows. Each window was its own submit, on a `rocket` with the
cold-start fix and without patch `0012`. That `rocket`'s completion poll ran at the stock 1 ms
period. At the 50 us period of `0012` a submit measures 439 us (an 8×64×32 int8 matmul,
2026-07-28) [HW sweep, H96 MAX M9]. Chaining the row tasks into one kick takes a 32×32 plane
in 8 windows from 2.8 to 1.0 ms (2026-07-29) [HW sweep, H96 MAX M9]. See rk3576-regcmd.md
§"Cost per window", §"The per-submit floor is the driver's completion poll" and §"Chained row
tasks".

### Depthwise convolution

The depthwise register program comes from manufactured captures at 22 channel counts. Those
captures fix the four formulas that the found captures' confounds leave open. A corpus of square
captures reads `0x118C` as the plane height.

Both depthwise buffer layouts were read off the RK3576 rather than out of a capture.
[rk3576-regcmd.md](rk3576-regcmd.md) has the probes that decoded them.

### fp16 convolution

The fp16 program is a transcription, not a sweep. Every `.rknn` on hand was int8, so its
precision fields had no oracle and were inferred from the RK3588's field packing. A float capture is
manufactured with `tests/data/rk3576-vendor-capture/float/mkfloat.py`.

The generator `gen_conv2d_fp16_rk3576()` emits the one task for which the program is exact, 16
input channels. The planner `rocket_rk3576_plan_ic()` lays an arbitrary `ic` out as `ic/16` slices.

A single-register sweep cannot find the float mode, because each of its three registers is inert
while either of the others is wrong. Two of the three registers are also out of reach of a
single-*bit* sweep from the integer word. Of the 139-word program, fourteen registers differ from
the integer one, and seven move with `ic`, which is what makes the diff readable.

The host column of the split's cost varies by a factor of two run to run on this board. Only the NPU column is therefore a measurement
[HW sweep, H96 MAX M9, measured 2026-07-26].
Against an 8-channel split, the 16-channel contraction halves both the submit count and the NPU
wall-clock time.

### First convolution on a packed image

The fp16 float weight cube was decoded from manufactured captures. The int8 weight-cube layout
was read off the RK3576 rather than out of a capture. A quantized ARGB capture's weights do not
survive quantization as a findable value set.

No test result on this path is wrong at the input-zero-point bound. Every first-conv shape in
every test has a zero input zero point, and a TFLite MobileNet's does too after the uint8 rebase.
Finding the bound needs a probe that varies the zero point on this path specifically
(`tests/rk3576_argb_pad.c`). [HW sweep, H96 MAX M9]

#### Revision history

| Date | int8 stem | fp16 stem | Conditions |
|---|---|---|---|
| 2026-07-27 | 3.94 ms | 256.7 ms | 3×224×224 k3 s2 oc32, whole entry. The fp16 entry waited out the idle that clears its next-submit hazard before every call, a ~122 ms guard |
| 2026-07-27 | | 226 ms, then 13.75 ms | The same fp16 shape, waiting for the idle, then with the power cycle driven by `rocket_rk3576_power_idle()` at the stock 50 ms autosuspend delay (rk3576-regcmd.md §"Cost of the clearing idle") |
| 2026-08-10 | 3.21 ms | 13.98 ms | Kernel 7.1.7, `rocket` 1.6.0, `dpu_grace_us` 0, governor `ondemand` ("Cost of the library entries"). The current figures |

The sheet stated the first row as "The int8 program takes 3.9 ms against the fp16 program's
257 ms at the same 224×224 stride-2 shape" until 2026-10-12. It now gives the ratio of the
2026-08-10 figures, about a quarter [host-computed: 3.21 / 13.98 = 0.23].

The int8 program was stated as "the one path here with no per-program output-channel limit"
(2026-07-27). The measured delivery is 64 output channels from one program, at `oc` 32 and 64.
`librocketnpu` splits above 64 (`R76_ARGB_INT8_OC_MAX` in `src/rocket_conv2d_rk3576.c`), and
rk3576-regcmd.md §"Geometry bounds of the int8 first conv" states the same split. One program
above 64 output channels has not been run.

### The `CNA_PAD_CON1` pad map

A pass/fail sweep reads `CNA_PAD_CON1` as "live, but no single centering explains it". Such a
sweep asks "is there a value that works" and returns nothing when there is not.

The probe `tests/rk3576_argb_padcon.c` reads the pad tap instead. It programs one output channel,
one unit weight at kernel tap (0,0), `in_zp` 0 and unit scales. At k3 s1 pad 1, the outputs at
(0,0), (0,1) and (1,0) are then the corner, top and left pad samples the MAC saw. The output at
(1,1) is the image pixel, as a positive control.

The same geometry and weights also run on the direct datapath, whose pad constant is measured
correct. That control places a map read here in
the sub-encoding rather than in the instrument. The readout covers 8 programmed values × 3 image
channels × both datapaths [HW sweep, H96 MAX M9].

The emitter's `0x1084 = 0` at `in_zp` 0 is why every test that ever ran this path was exact.

### Pad columns on the packed path

The readout above samples one position, output (1,0), whose tap sees input (0,-1), the row-0 left
pad. In a contiguous packed image that byte lies before the start of the buffer. There, "the
columns feed a constant 0" and "the columns feed whatever is linearly adjacent" predict the same
number. That readout reads neither the trailing edge nor any other row.

The probe `tests/rk3576_argb_padmap.c` reads all four edges at seven positions each. A second
weight placement at tap (k-1,k-1) reaches the bottom row and the right column. The probe computes
`fed = output + in_zp`, so the readout survives a non-zero zero point [HW sweep, H96 MAX M9].

At `in_zp` 0 a dead column feeds the right answer, which is why every test that ever ran this
path was exact.

The register walk takes the packed and direct programs at one geometry (`ROCKET_RK3576_DUMP=1` on
both), where they differ on 17 non-address registers. It sets the packed program's registers one
at a time to the direct program's values. It leaves the columns at 0 in all 11 configurations
whose interior still computes, and 6 configurations break the program. The walk is a one-at-a-time
search and cannot see a condition of two registers.

#### Excluded causes

- *The pad columns feed whatever is linearly adjacent in the buffer.* No column sample varies down
  its edge. The wrong trailing region, which no leading-column story explains, is the right
  column.
- *Another single register.* The walk above leaves the columns at 0 in every configuration whose
  interior still computes.

### Materialized pad columns

The host-repair cost was measured with `tests/rk3576_argb_border_cost.c`, governor pinned. The
four test points of the extension, and the entry at the caller's own geometry, were measured with
`tests/rk3576_argb_extend.c` [HW sweep, H96 MAX M9, `tests/rk3576_argb_extend.c`].

### A plane inside wider rows

The PPU sweep covers 16 test points over four geometries at gaps of +1/+4/+16/+64 elements, plus
12 test points of column offset at 1/3/8 elements. The geometries span max and average pooling,
kernel 2/3, stride 1/2, pad 0/1, square, odd and non-square planes, and 1/2/4 channel groups.
Every test point is bit-identical to a reference over a tight plane whose surplus columns hold a
sentinel byte. Each geometry also runs against a control that tells the group stride and leaves the
pitch derived. The control differs in all four [HW sweep, H96 MAX M9, `tests/rk3576_row_pitch.c`].

A source read of `librocketnpu`'s emitter makes the CNA's `0x1078` width field and `0x1090` "one
quantity in two units". A readout on the RK3576 separates them. It is a 1×1 convolution whose two
live weights copy a 16-bit position code out of the feature buffer. The output at `(y, x)` then
names the atom the DMA fetched. At two geometries, the map departs at exactly the atom where the
row budget is used up. Inflating that one field to the formula the sheet gives makes the map
affine to the last row.

The cheaper form's errors were measured and scored by region. It is wrong on nothing at k1 or at
the stride-2 shape whose grid fits exactly.

The pitched read is bit-exact against a tight-plane reference with the surplus filled with a
sentinel byte. The sweep covers nine geometries at four gaps of +1/+4/+16/+64 elements, plus three
column offsets of 1/3/4 elements [HW sweep, H96 MAX M9, `tests/rk3576_conv_pitch.c`]. The
geometries span kernels 1/3/5, both strides, square and non-square planes, 2 and 4 channel groups,
and one row task and up to ten. Each geometry also runs against a control that leaves the row
stride derived, and the control differs [HW sweep, H96 MAX M9, `tests/rk3576_conv_pitch.c`].

In the pitched row window, the residue between 36 and 40 rows is unpinned. The missing margin has
the size of the retained rows a chained window has, which the arithmetic does not model.

The configurations of the CNA pitch probe are not independent measurements on the RK3576. Run in
sequence, the pitched walk appears to revert to the tight stride after four rows. Run one
configuration per process, it is affine to the last row. The sequence supports a complete,
mechanically plausible and wrong story: "the row stride and the group jump do not compose".

### The materialized stem on a graph

The direct stem is the control configuration.

The governor does not account for the saving. The two configurations differ in host work, so
pinning `performance` is the control the governor bias calls for. The corpus numbers stay on
`ondemand`, which is what every other wall-clock figure on the sheet is measured on.

The per-call figures come from [HW sweep, `tests/rk3576_stem_cost` group E,
`ROCKET_SC_ZP=-128`]. The host compaction is measured with `tests/rk3576_row_pitch cost`.

The network test harness is `rk3576_net_gate`. It takes the materializing form where the next
layer is a pool, and its A/B switch is `ROCKET_RK3576_NET_ARGB=0`.

### TFLite stem through an input shift

The interior result is 394272 of 394272 outputs bit-exact
[HW sweep, H96 MAX M9, `tests/rk3576_argb_pad.c`].

Wired into a real graph, the shift is a loss, which inverts the interior result's reading. The
lowering is in `rk3576_net_gate` behind `ROCKET_RK3576_NET_TFLITE_STEM=1`. The harness runs that
A/B whatever the default is. The border is computed by the same reference everything else is
asserted against, so a comparison against that reference alone would say nothing about it.

A "worth about 4 ms" estimate compares the shift with the widened path transient (10.6-12.2 ms),
a saving resident weights already collect. Two costs were measured between the shift and a win.
The first is the border fix-up, the shift lowering's own cost, which stands. The second was a
per-call BO teardown of 1.0-1.8 ms, incurred while the packed path had no resident handle. The
library removes it. What remains against the TFLite-stem shift is the border fix-up alone
[HW sweep, H96 MAX M9, `tests/rk3576_net_gate.c`, 2026-07-29].

The ~30 ms and ~31.5 ms inference times are MobileNetV1-224, the one network the graph test
ran on 2026-07-29. That run had resident weights, its layers submitted separately, and a
scalar cube transpose: the graph measured 30.1 ms. A NEON transpose took the same graph to 21.2 ms later
that day, and the one-kick graph measures 5.0 ms (2026-08-03) [HW sweep, H96 MAX M9].

### Packed-image path traps

`librocketnpu`'s emitter reproduced twelve int8 ARGB captures register for register and still
wrote nothing at every geometry tried. A leave-one-out over the nineteen-register delta against the
working fp16 program revived none of it, which read as a dead datapath. The captured op stream
from the golden table, submitted with only its five address registers patched
(`tests/rk3576_fc_vendor.c`), writes its whole output. The mode works, and the extrapolation away
from the captured geometry does not.

Every ARGB capture is zero point 0, so no capture exercises `0x1054`-`0x105C`.

The int8 packed-image program has no next-submit hazard: 20 of 20 submits write at a zero gap
[HW sweep, H96 MAX M9].

The fp16 program's write rate against the gap was swept on 2026-07-27, on the `rocket` module
before interface 1.0, with `power/autosuspend_delay_ms` at mainline's 50 ms. With `on` written to
`power/control`, the program wrote once and then never again at any gap, 400 ms included
[HW sweep, H96 MAX M9, 2026-07-27].

"It is the one hazard the RK3576 has on any wide output" was written on 2026-07-27. Three
paths write wide outputs without the hazard [HW sweep, H96 MAX M9, `rocket` 1.6.0]. They are
the matmul-form fp16 program's fp32 output (2026-09-25), and the whole output stage's dense
int32 writer and fp32-output convolution (2026-09-30). See §"The poisoning is one hazard, and
its cost is a system setting".

## Narrow input-channel counts on the direct datapath

The border defect was invisible to the existing tests, because every existing shape had `ic` a
multiple of 32 or `w_zp` zero. The fix is gated by `tests/rk3576_conv_lib_gate.c` group `nic`
[HW sweep, H96 MAX M9, `tests/rk3576_conv_lib_gate.c` `nic`, 2026-07-30].

## Asymmetric padding as an output extent

The six shapes are in `tests/rk3576_conv_shapes.h`. On MobileNetV1-224 the derived trailing pad
matches the model on 30 of 30 compute layers. The reference trap is the network test's per-axis
pass, which builds its own descriptor. Without the extent, its entry computes 55 rows and its
reference computes 56.

## The depthwise weight zero point

The fold result is from `tests/rk3576_conv_lib_gate.c` group `zp`
[HW sweep, H96 MAX M9, `tests/rk3576_conv_lib_gate.c` group `zp`]. The same group gives the
result at both signs, both channel-group boundaries, k=5 and stride 2
[HW sweep, H96 MAX M9, `tests/rk3576_conv_lib_gate.c` group `zp`].

## Per-axis weights on the depthwise path

The `C` ramp result is from `tests/rk3576_perchannel_gate.c`
[HW sweep, H96 MAX M9, `tests/rk3576_perchannel_gate.c`]. The eight-shape result is from the
same test [HW sweep, H96 MAX M9, `tests/rk3576_perchannel_gate.c`].

## The weight slice and the feature allowance

The table covers two shapes, the `F` values 0, 1024 and 2048, and two group counts. It ran one
task per test point at `MAP_REPS=3` [HW sweep, H96 MAX M9, 2026-08-05]. The run's own record
names four `F` values and lists these three. No log of a fourth survives.

### On `rocket` 1.6.0

The run covers the same two shapes at all five `F` values and plane heights 20 and 30, both
inside the `F=0` budget. It runs `tests/rk3576_conv_lib_gate.c rowmap` at `MAP_REPS=3`, and
`ROCKET_LG_MAP_F` forces each value [HW sweep, H96 MAX M9, `rocket` 1.6.0, 594 MHz,
rocket-userspace as of 2026-09-30, 2026-10-12]:

| Shape | `F=0` | `F=256` | `F=512` | `F=1024` | `F=2048` |
|---|---|---|---|---|---|
| `112x112 k7 ic 32 oc 128` | exact, 6 of 6 | height 20: row 0 wrong, 3 of 3, at most 690 elements; height 30: exact | height 20: row 0 wrong, 3 of 3, at most 502 elements; height 30: exact | exact, 6 of 6 | no completion, 6 of 6 |
| `112x112 k7 ic 64 oc 64` | exact, 6 of 6 | exact, 6 of 6 | exact, 6 of 6 | no completion, 6 of 6 | no completion, 6 of 6 |

No completion is a job retired at the 125 ms backstop with nothing written. The entry then
returned `ROCKET_E_DEVICE` after its eight power cycles. These cells account for
all 145 retirements in the kernel log. The wrong row-0 elements hold written values, not the
sentinel, and 2 to 12 of them are zero.

### Excluded causes

- *An `F=1024` value the hardware does not honor.* Forcing `F=1024` on the two-group test point
  is 3 of 3 wrong at all 30 plane heights from 7 to 36. That includes heights needing 784
  granules of a 4096-granule budget, where a value delivering only `4096` would be harmless. At
  every one, `worst` is exactly half the output.
- *Programming an allowance larger than the plane being itself invalid.* The same forced value
  beside a 16-granule slice is exact at 42 of 42 test points over 1, 2 and 3 output-channel
  groups.
- *The `F` value as the governing quantity.* The table of forced `F` values records `F=2048` as
  delivering at every slice tried, and here it does not deliver at all. The job raises no
  `PC_DONE`, the output guard runs eight power cycles, and the entry gives up with `-4`.

## Slope and reserve

The sweep reads the boundary directly rather than bracketing it, and reports the walk as a
map. It forces `F` so that `F` is the only thing that changes. The slope is 1:1 to the
granule at three group counts, and what the group count costs is a fixed reserve, not a
scaling. Each boundary is 3 of 3 wrong at the first KiB over and 0 of 3 at the last KiB
under [HW sweep, `rk3576_conv_lib_gate slicemap`].

The cap table's sweep took the last measured-good sample rather than the boundary. Raising a
cap on an extrapolation is the direction that produces wrong output.

With both slice checks in place, `claimplan` reports 160 shapes, 0 accept-a-refusal and 0
over-refusal.

That no network reaches the bound is measured by the five network tests keeping their
structure across the change. Why no model reaches it is not enumerated layer by layer. The
instrument for the question is `rk3576_conv_lib_gate rowmap`'s `ROCKET_LG_MAP_F`, and the test
points are `k7ic64-g2` and `k7g4-112`.

## Slices at a multiple of 32 KiB

### Walks

The walks move both conditions that every recorded test point shared, the kernel and the
plane. They run at `F=1024` and `oc=128`, where the cap is 80 KiB. Both 96 and 128 KiB are well
above it [HW sweep, H96 MAX M9, `rk3576_conv_lib_gate slicemap` with `ROCKET_LG_SLICE_K`]:

| walk | cap | 81-95 KiB | **96 KiB** | 97-127 KiB | **128 KiB** |
|---|---|---|---|---|---|
| `k=1`, 4×2 plane (the record) | 80 KiB | 1 group lost to 85, 2 from 86 | **0 of 3** | 3 groups | **0 of 3** |
| `k=2`, 4×2 plane | 80 KiB | 1 at 84, 2 at 88 and 92 | **0 of 20** | 3 (20 of 20 each) | **0 of 20** |
| `k=1`, 4×19 plane | 80 KiB | 1 to 85, 2 from 86 | **0 of 10** | 3 to 107, **108-127 do not complete** | **0 of 10** |

The graded group loss steps at the same slice sizes on both lattices. These walks hold one
group count and one allowance. The invariance is measured at one of each, where the slope
table has three group counts and two allowances. A walk that repeats this at `k=3` cannot be
built, because its first 32 KiB multiple is far past the pool.

### Excluded causes

- *The pool edge.* 96 KiB is not the pool edge: at `F=1024` the pool edge is 128 KiB
  [HW sweep].
- *A power-of-two `ic`.* At 96 KiB and `k=1`, `ic` is 3072 [HW sweep].
- *The weight cube.* For the same 96 KiB, `ic` is 3072 at `k=1` and 768 at `k=2`, so nothing
  about the cube survives as an explanation.
- *Extra headroom.* At 4×19, 128 KiB runs and is exact, and the escape jumps clean over a band
  where the program does not execute. No reading of the escape as "a little more headroom"
  survives that.

## The drain deadline and the row planner

The planner's agreement with the SoC on 23 of 23 geometries is measured by
`rk3576_conv_lib_gate rowbound` at `dpu_grace_us = 750`.

### Edge movement

At two groups, the boundary height is wrong on every run inside a long sequential walk and
clean in a short one. The binary and the shape are the same. A movement of 1-4 rows between
runs is this same effect.

The edge is read as a rate, with `ROCKET_LG_MAP_REPS`. Below the height where the failure is 3
of 3, there is a region where it is 1 of 3. The `k7g4-56` test point is wrong once at 60 and
once at 68, then clean to 84. The `k7g4-224` test point is wrong once at 19, then clean to 24.

### Excluded causes

- *The NPU clock.* A reading in which a slower clock lengthens the drain contradicts the fixed
  clock recorded in the sheet. The NPU clock holds 786 MHz for a whole boot, and the NPU has no
  devfreq node.

## The deadline under the write guard

The full-but-wrong outcome shows 0 dropped-to-zero elements in 12 of 12 failing test points
[HW sweep, H96 MAX M9, `rocket` 1.5.0, `dpu_grace_us` 500]. The count is of zero elements, and
the entry had stamped the output with `0xA5`. A zero count therefore cannot see a dropped write
[source-confirmed: rocket-userspace as of 2026-08-02, group `rowlaw` and the default stamp].

Re-scored against the stamp, the same test, run twice at `dpu_grace_us` 500, gives 16 failing
test points (9 and 7). In every one, no wrong element is zero, and every wrong element still
holds the stamp. One point has 7712 of 7712. `k5-ic32-g6`, wrong at every row of its trailing
groups, has 325804 of 325804 [HW sweep, H96 MAX M9, `rocket` 1.6.0, 594 MHz, rocket-userspace
as of 2026-09-30, 2026-10-12]. The current planner refuses `k3-ic64` at 4, 6 and 8 groups one row past their
ceiling. Of the other points, five computed one row past it, and one reached the data cap.

## Output-channel group loss at a large weight slice

### Revision history

On 2026-08-05, after the drain map, the section described "A large-window failure at a small
slice, 49 KiB against that 144 KiB cap". It said that failure "is a different rule, not this
one met from the other end". It also said the loss "depends on the slice rather than the
window". The slice cap was then taken as independent of `F`.

The `F`-aware measurement of the same day charged the slice against what `F` leaves. Its
conditions were `112x112 k7 ic 32 oc 128`, slice 49 KiB, four groups, `F` forced, one task per
test point, the plane held inside the `F=0` budget. It is exact at `F=0` and `F=1024` and wrong
at `F=2048` (in channels 32-127, three groups of four, at every row) [HW sweep, H96 MAX M9,
2026-08-05, "The weight slice and the feature allowance"].

## On the DEPTHWISE path the two low CBUF rungs do not deliver, and the failure is a band

The height walk is `rk3576_conv_lib_gate rowmap`. The forced-allowance table uses one window
that `F=0` does not cover, with `ROCKET_RK3576_CBUF_F` forcing each allowance beneath it. All
ten depthwise `rowbound` test points agree with the planner [HW sweep, H96 MAX M9,
`rk3576_conv_lib_gate rowmap`]. The unchanged windows and task counts are those of the ten
`rowbound` depthwise test points.

### Excluded causes

- *A threshold on the depthwise weight footprint.* On the depthwise path, 4 granules is dead
  where 32 is live, so a threshold on that footprint is refuted in both directions. The 2×2
  test point also refutes a 64-channel-group footprint against the direct path's 16-granule
  threshold.
- *A capacity bound read from `dw-k5-s1-160`.* That test point does not refute one. It needs
  4640 granules, which puts it one allowance higher, so it computes because it asks for more.

### Per-partition comparison

One instrument finds a model layer the per-operation test points miss. It compares every
delegated partition's output against CPU TFLite on the model's own weights and input. That
comparison is a distance rather than a pass.

Under `experimental_preserve_all_tensors`, the
runtime gives every claimed node its own partition. The same run is therefore also the
layout-or-arithmetic discriminator, with no joins and no cross-layer kick in play. The flag is
set on both configurations. Without it, the arena reuses a partition output's bytes and the
control reports hundreds of counts.

## The matmul entry's requant and per-column ramp

### Requant model

The accuracy statements in the sheet's sections before this one are host arithmetic. The test
`tests/rk3576_mm_requant` compares the on-NPU requant against a host model of it, at the entry.
The device matches the model on 0 wrong of 1,396,736 elements, with nothing unwritten. The host
model `tests/requant_model.h` is therefore the matmul path's arithmetic as well as the
convolution's.

The exact-float model's cost as an instrument is two to three orders of magnitude below what it
was used to decide. The accuracy simulations used an exact-float requant with round-to-even, and
it disagrees with the integer requant on 0.014% to 0.21% of elements. Every disagreement is
exactly 1 LSB. The perplexity numbers stand as written.

### Per-column epilogue

The per-column epilogue matches on 0 wrong of 698,368 elements over twelve test points, six
shapes at two column-scale spreads each. One of the test points is forced to eight N tiles, so the
per-tile `(MUL, SHIFT)` re-plan is exercised.

### Shift ramp

The shift ramp read 0 wrong of 1380352 elements over 42 test points [HW sweep, H96 MAX M9, rocket
1.6.0, 2026-09-27]. The median column gain error on the two models' projections is measured with
`tests/rk3576_mm_percol_err`.

### Shift-0 ramp accuracy

The worst-case table is the test's own figure. Its weights come from a full-range generator. The
host model of the planner reproduces the table within 0.92-1.19x on the test points with a column
spread, which are the discriminating ones.

The perplexity re-run models the ramp in place of an exact per-column scale, as `chanout_ramp`
and `hadamard_chanout_ramp` in `tools/rk3576-w8a8-ppl.py`.

### Stability without the rotation

On SmolLM2-1.7B, de-quantizing by the gain the ramp achieved measures 22.209x, against the
requested scale's 1.505x. The same substitution on Qwen2.5-1.5B measures 1.083x against 1.080x.
The blow-up is not a property of the reconstruction. The jitter control in the sheet is the test
that shows it: an uninformative perturbation in layer 0 alone gives the same catastrophe.
[host-computed, arithmetic, 2026-08-10]

The achieved-gain question closes as a wash (the Qwen2.5-1.5B figures in the sheet). A signature
change that returns the achieved gains is therefore neither dangerous nor worth building.

#### Excluded causes

- *An error in the achieved-gain reconstruction.* A per-element map covers the model's own
  layers, on either configuration's own trajectory. In every band of every mapped projection, the
  achieved-gain reconstruction is equal or smaller in error. No `C` sits at either clip, and no
  column is clamped at the minimum.
- *The last decoder layer.* A per-layer magnitude trace localizes the divergence entirely to the
  last decoder layer, and that localization is false. Restoring the requested divisor in that one
  layer leaves 2.24e10. The achieved divisor in that layer alone costs nothing (10.578 against a
  10.577 baseline).
- *A depth threshold.* The depth sweep is non-monotone. The achieved divisor in the first L
  decoder layers gives 3.6e10 at L=1, 10.68 at L=2 and 1.3e10 at L=3.

### Resolution limits

The N-tile and accumulator-bound measurements were taken in the test suite's run on the H96 MAX
M9 under `rocket` 1.6.0, 2026-08-10. The forced-tile test point is the sorted case: the test's
column scales are monotone in the column index.

## The DPU LUT and its input map

The input map is bit-explained against a CPU model at four spans of 8192 samples each, to at most
one output count (`tests/rk3576_lut_probe.c`). [HW sweep, H96 MAX M9]

## The int8 activation op

The test `tests/rk3576_act_gate.c` sweeps 6 kinds × 5 quantizations × 6 shapes, dense over all
256 byte values. The result is 180 of 180 bit-exact against a CPU model of the RK3576's
arithmetic. [HW sweep, H96 MAX M9]

## The activation fused into a convolution epilogue

The 28 fused cases run as `tests/rk3576_act_gate.c fused`. [HW sweep, H96 MAX M9]

## The residual add on the convolution datapath

### The search for a second operand

Two of the six sweeps close the question, because they cross axes. The first crosses every
register the program leaves at zero with all 32 steps of the output gain ladder. A placement
sweep at one gain and a gain sweep at one placement are each blind to one kind of operand. That
operand both sits somewhere unexpected and enters the accumulator unscaled. The second is a
delta probe that walks a single non-zero atom over an operand buffer eight cubes long. It shows
that the program reads exactly the addressed cube and nothing within two cubes either side of
it.

### The lowering

The lowering's accuracy over the nine MobileNetV2 and ResNet-18 residual shapes comes from
`tests/rk3576_residual_add.c`, which passes 12 of 12.

### Fusion into a block's last convolution

The fused form, the arithmetic and refusal of `rocket_residual_fuse_weight_rk3576()`, and the
standalone add's transient cost all come from `tests/rk3576_residual_add.c`
[HW sweep, H96 MAX M9, 2026-07-31]. The standalone add's 0.6-3.3 ms is measured across the
shape table of `tests/rk3576_residual_add.c`.

## Accumulation order

Two runs are identical. The test is `tests/rk3576_mm_fp16_round`. [HW sweep, H96 MAX M9,
`rocket` 1.6.0, 2026-09-25]

### Excluded models

Groups of 16 or 64, or a rounding per product, miss by up to hundreds of ulps at K 2048.

## bf16 operands

The probe is `tests/rk3576_mm_bf16_probe`, one task a shape. Two runs are identical. bf16 bytes
under the unmodified fp16 program come back wrong on every element, so the check can fail. The
fp16 control runs after the bf16 runs. [HW sweep, H96 MAX M9, `rocket` 1.6.0, 2026-09-26]

## Single-task limits

The limits come from `tests/rk3576_mm_fp16_envelope`, 49 shapes, each run as one task past the
tested shapes. The shape one row past the window is wrong the same way on a second run. The
multi-task shapes were run in the test suite on the same board.
[HW sweep, H96 MAX M9, `rocket` 1.6.0, 2026-09-25]

## fp16 matmul entries

On the H96, `tests/rk3576_mm_fp16_gate` runs 23 one-task shapes exact to the single-task limits,
and 6 shapes of 2 to 25 tasks. Its row-major weight control comes back wrong on 64 of 64
[HW sweep, gate run, `rocket` 1.6.0, 2026-09-25]. The test's fills make every sum exact in fp32,
so it cannot see rounding.

## An int8 matmul job can raise no completion, and the shape decides

The partial outputs were scored against the requant model by `tests/rk3576_retire_probe real K`
[HW sweep, H96 MAX M9, `rocket` 1.6.0, kernel 7.2.3, 2026-09-27]. Under a write check alone, the
entry returned 0 at `K`=2240 in three processes of three.

### Revision history

Before the entry planned around the weight-phase rule, and before it narrowed a stalled tile,
`rocket_matmul_int8_rk3576()` scored each job's completion. It refused at all five stall `K`
(2208, 2240, 2272, 2432, 4480 at `M`=512 `N`=1536), after eight retired attempts and about
1.88 s [HW sweep, H96 MAX M9, `rocket` 1.6.0, kernel 7.2.3, 2026-09-27]. The harness was
`tests/rk3576_retire_probe real K`, its outputs scored against the requant model. The narrowing then computed all five after 2-4
re-plans, and the weight-phase plan computes them with no retirement (the sheet's §"Planning
around the weight-phase rule").

## The weight-phase rule

### Stall map

`tests/rk3576_stallk_map` runs one job per row task, with no retry. It scores each 32-channel
group against the host int64 reference. In all, 206 jobs over 54 configurations stop where the
rule puts them. Every job it calls clean is exact.

| Allowance, plane × rows | `K`: first unwritten channel | Clean `K` |
|---|---|---|
| `F` 1024, 64×2, `M` 512, `N` 1536 | 2144: 672, 2208: 384, 2240: 320, 2272: 288, 2336: 224, 2368: 192, 2400: 160, 2432: 160, 2464, 2496, 2528: 128 | 2080, 2112, 2176, 2304, 2560 |
| `F` 0, 32×1, `M` 512, `N` 768 | 4192: 672, 4384: 224, 4416: 192, 4448, 4480: 160, 4512, 4544, 4576: 128 | 4128, 4160, 4224, 4352, 4608 |

Each clean `K` sits exactly on the edge the rule draws. [HW sweep, H96 MAX M9, `rocket` 1.6.0,
kernel 7.2.3, BO pool off, 2026-09-27]

### Allowance, rows and address

The allowance moves the stall, and the rows, `M` and the address do not. At `K`=2240:

- `Mt` 32, 64 and 96 at `F`=0 are clean on four planes, `M` 192 included.
- `Mt` 32 and 64 with `F` forced to 1024 stall at channel 320.
- `Mt` 128 as 32×4, 64×2 and 128×1 stalls at 320, and so does every row task at `M` 1024 and 2048.
- A weight base 1-60 KiB into its buffer leaves `K`=2240 at 320 and `K`=2304 clean, so the phase
  is not the IOVA's.

### Whole 64 KiB units

Forced to `F`=256 and 512, whose areas are 176 and 160 KiB by the granule count, `K` 3328-3968
stall at 32 channels and `K`=4096 is clean. A linear 176 or 160 KiB area would have left 3328,
3584, 3712 and 3840 clean.

## Planning around the weight-phase rule

Each `K` of the matmul plan ran in three processes, exact with no retirement on the first call and
the repeat. [HW sweep, H96 MAX M9, `rocket` 1.6.0, `tests/rk3576_retire_probe`,
`tests/rk3576_mm_requant`, 2026-09-27]

The convolution bands and their planner skip were measured with `tests/rk3576_conv_band_probe` [HW
sweep, H96 MAX M9, `rocket` 1.6.0, 2026-09-27].

## The rule on the other convolution paths

The fp16 group-of-16 stall at `F`=2048, k9 and k11 at oc 32 on a 64×176 plane, left channels 16-31
unwritten in 6 of 6 runs. [HW sweep, H96 MAX M9, `rocket` 1.6.0, `tests/rk3576_conv_band_probe`
`f:` and `d:` configurations, 2026-09-27, the headline configurations re-run]

## Chunk depths for the K-split

Five chunk depths were screened by redo count before use as test configurations, at `M`=512
`N`=1536, four warm reps each with the first discarded. The backstop count was unmoved across the
whole screen [HW sweep, H96 MAX M9, rocket 1.6.0, 2026-08-11].

## The K-split wedge

The wedge ran on `rocket` 1.5.0, kernel 7.1.6, on the int32 entry's narrow writer, then its
default. The board ran kernel 7.1.7 and `rocket` 1.6.0 from 2026-08-10. The dense writer's run
of the same shape is the library's own. It measured 345.7 ms, 48 submits with no redo, and 4096
sampled elements exact [HW sweep, H96 MAX M9, `rocket` 1.6.0, 2026-09-30].

### Software support

The wedge shape returns `ROCKET_E_UNSUPPORTED` in a 64 ms process, 21 MB of which is generating
the operands, and the `dmesg` warning delta is 0. A control shape run immediately afterwards in
the same boot computes at 26.1 ms. [HW sweep, H96 MAX M9, 2026-08-06]

### Small matmuls in the class

`M=32 K=8192 N=128` and `M=16 K=16384 N=64` ran green in the test list across several sessions,
and `M=64 K=8192 N=32` scored exact three times. Every axis that separates them from the wedge
moves together:

| test point | submits | weight cube | per-submit output |
|---|---:|---:|---:|
| `M=32 K=8192 N=128` | 2 | 2304 KiB | 16 KiB |
| `M=16 K=16384 N=64` | 4 | 1152 KiB | 4 KiB |
| `M=64 K=8192 N=32` | 4 | 576 KiB | 8 KiB |
| the wedge | 256 | 4608 KiB | 512 KiB |

All four share `ks`=4608 and a 144 KiB resident slice. The K split stays in the test list through
`rocket_matmul_int8_rk3576_i32()`, called directly.
[pure, `tools/rk3576-planprobe.c cell`, 2026-08-06]

### Class boundary

The pure planner was enumerated over thirty `(M, N)` test points and gives one answer in every
one. [host-computed, pure enumeration, `tools/rk3576-planprobe.c map`, 2026-08-06]

## The single-group band below the K split

The tile ladder is a pure enumeration [host-computed, `tools/rk3576-planprobe.c map`,
2026-08-06]. The wall-clock times were measured with `rk3576_submit_floor` [HW sweep, H96 MAX
M9, 2026-08-06].

## System hang during matmul work

Five whole-board hangs were recorded on `rocket` 1.5.0. None had a serial console behind it.

### First hang

The first hang came during the sgemm A/B, which was pinned to the `performance` governor with
four threads busy. A power event remains a rival reading for it.

### Second hang: the K-split ladder

A ladder over the K split on the 1.5.0 module ran these test points in order, to find which one
hangs the board:

- `M=64 K=8192 N=32` (3 submits)
- `M=128 K=8192 N=256`
- `M=512 K=8192 N=256`
- The wedge itself

It ran one test point per process on `ondemand`, with a single-threaded CPU reference between
test points. The board went down during it.

The ladder's logs survived the power cycle, and they show that it never got past the first test
point. The ladder's output and per-cell logs record `M=64 K=8192 N=32` three times. Every rep
reads `RC=0` and `WRONG=0 of 2048` over the whole output, with wall-clock times of 14.4-16.5 ms.
The boot has zero `WARNING`/`Oops`, and its last NPU message is 45 minutes before the end. The
journal stops abruptly at 04:29:45 UTC on a routine `sudo` line, with no shutdown sequence.

The board therefore hung after a clean repetition of the smallest test point in the ladder,
with no NPU fault of any kind recorded. That hang is neither shape-selective nor announced. It
does not implicate the K-split submit, which had just returned `RC=0`. A reading of this run
from host unreachability alone concludes the opposite, and the logs contradict it.
[HW sweep, H96 MAX M9, n=1 ladder, logs read 2026-08-06, cause unmeasured]

### Third hang: the matmul correctness sweep

A matmul correctness sweep (`tools/rk3576-mm-drain-sweep.sh`) started on a board that had been up
1h56m, at taint 4096 with zero `dmesg` `WARNING`s. That board had just completed a
five-repetition device run cleanly. It went unreachable inside the next four minutes and did not
come back.

The governor was `ondemand`, and the workload was pinned to one core with `taskset -c 4`. No
shape in the sweep's grid takes the K split. Every one is a single-task `K <= 4096` matmul.

The one thing this run introduced that no earlier one had is `dpu_grace_us` = 0 on the matmul
path. The conv-path `rowmap` sweep weakens it as a candidate, because that sweep drove grace to 0
at 10 repetitions a test point with no incident.

A second host on the same subnet, the RK3588 board, answered SSH normally and could not ping the
RK3576, repeatedly, over ten minutes.

The sweep wrote its log under `$HOME`, so that a reboot would not clear it, and nothing reached
it. A filesystem-wide search after the power cycle found no copy of the log at all. The run
therefore died before its first line was flushed, and how far it got is unrecoverable.
[HW sweep, H96 MAX M9, n=1, no serial console, 2026-08-06, cause unmeasured]

### Fourth hang: the same sweep without grace 0

The same sweep ran again on a board up 7h05m, at taint 4096 with zero `WARNING`s, freshly
rebuilt. That board had just returned 58 of 58 test rows and a clean five-repetition matmul
probe. The grace list was `500 4000 250 100`, so 0 was never written. The governor was
`ondemand` on one pinned core, and again no shape had `K >= 6176`.

The board went unreachable much faster. The launching `ssh` failed to deliver an `echo`
scheduled five seconds behind the detached start. That brackets the hang at under about 5 s from
the first process, against the previous four minutes. The first test point in the safe-first
order is `M=128 K=2048 N=2048` at grace 500. Its grace write was a no-op over the value already
live, and it was the one grid shape nothing had run that session. The sweep's log was again
empty, and the RK3588 again showed the network up.

After four hangs, the leading candidate was that one shape. The sweep's `SHAPES` array begins
with `M=128 K=2048 N=2048`. It was therefore the first device work of both the third and the
fourth hang, and the fourth's ~5 s bracket puts the hang inside it. Before the fifth hang that
shape appeared nowhere else in this SoC's recorded corpus. It is not one of the sgemm A/B's shapes,
not one of the six "0 of 5" test points, and not in any test row. It is a single-tile matmul,
the only 1-tile test point in the grid.

The discriminating test runs `rk3576_mm_corr 128 2048 2048 1` alone, in the foreground, as the
first device action after the power cycle, before any loop. A hang means the cause is a shape. A
shape that survives three times is excluded, and leaves the structural reading.

The structural reading: what the three sweep hangs share, and the survivors do not, is a detached
loop (`setsid nohup`) running one test point per process. Each process allocates and tears down
tens of megabytes of BOs. The module has neither the BO-lifetime fix (`0014`) nor the
IOMMU-domain fix (`0013`), and both were written against observed render-group crashes on this
SoC. This reading is [hypothesis], with no console behind it. It is also in tension with the
discipline that produced the structure. That discipline runs one test point per process, so
that a wedge cannot contaminate the next one.

Process churn alone is not sufficient. The 58-row test list is a foreground loop of 58 device
processes, and it completed cleanly on the same boot, minutes before the hang.
[HW sweep, H96 MAX M9, n=1 per hang, no serial console, 2026-08-06, cause unmeasured]

### Fifth hang: one shape in the foreground

`M=128 K=2048 N=2048` ran alone:

- One foreground process, one repetition, one test point
- No detached loop, no module-parameter write, no governor change and no all-core load
- No `K >= 6176` shape

The board had been power-cycled 49 minutes earlier, and its whole parameter block was read
rather than assumed:

| parameter | value |
|---|---|
| module | shipped `rocket` 1.5.0 |
| bound entry | core 0, the only one |
| `poll_interval_us` | 50 |
| `dpu_grace_us` | 500 |
| `dpu_blind_us` | 250 |
| `rocket_batch_submit` | 1 |
| governor | `ondemand` |
| taint | 4096 |
| `dmesg` `WARNING`s | zero |

The two probe files were shipped by tar with `md5sum` matching at both ends, and the tree was
rebuilt at rc=0 first. The command was:

    timeout 120 sudo -E taskset -c 4 ./build/rk3576_mm_corr 128 2048 2048 1

It emitted the library's three initialization lines and nothing else. The `timeout 120` never
fired and the `ssh` never returned. The board therefore left the network before its own 120 s
guard could act, and it took the whole system down, not just the NPU. Six probes over two minutes
from the RK3588 on the same subnet all read down. That RK3588 answered SSH meanwhile and reported
1d8h of uptime.

The pure plan does not separate this shape from the one that survives. Read with no device, the
planner gives this plan: at `K=2048` the CBUF feature budget is `8 * 32768 / 2048` = 128 pixels.
`r76_mm_plane` takes the widest divisor of `M` that fits: `iw`=128 and `ih`=1. That is one row
task, one N tile, one task in one submit. The shape `M=256 K=2048 N=2048` is the most-run test
point on this path and a 10-of-10 survivor. It plans the identical per-task program at `iw`=128,
and differs only in having two tasks instead of one.

On the axes a pure read can see, the difference between the hanging shape and a survivor is one
entry in the task array. That this points at the task-array and submit path rather than at the
regcmd is [hypothesis]. The two shapes also differ in every runtime quantity the plan does not
name.

A single-task submit is ordinary on this entry. In the cost table of the library entries,
`32x1024x1024`, `128x1024x1024` and `256x512x2048` all plan `ih`=1 and all compute. The table as
measured on `rocket` 1.5.0 has `128x1024x1024` at 5.98 ms. What the SoC objects to is therefore
not the task count alone. It is not any bound the library models either, as for the K-split
wedge.
[HW sweep, H96 MAX M9, n=1, no serial console, 2026-08-06, pure plan read from
`src/rocket_matmul_rk3576.c` and `src/rocket_hw_profile.c`, cause still unmeasured]

### Excluded causes

Each candidate below is excluded as a necessary condition, most of them by a later hang that ran
without it:

- *The `performance` governor with a sustained all-core load*, the first hang's rival reading.
  The third, fourth and fifth hangs ran on `ondemand`, pinned to one core.
- *The K-split submit.* The second hang came after a clean `M=64 K=8192 N=32`, and the third,
  fourth and fifth hangs ran no shape with `K >= 6176`.
- *`dpu_grace_us`=0 on the matmul path.* The fourth hang's grace list never wrote 0.
- *The harness structure*: a detached loop, per-process BO churn, a module-parameter write, a
  sustained load, a governor state. The fifth hang had none of them. The 58-row test list is 58
  foreground device processes, and it completes cleanly on the same kind of boot.
- *A network failure rather than a board hang*, for the third to fifth hangs. A second host on
  the same subnet answered SSH and could not reach the RK3576.

### Kernel 7.1.7 and `rocket` 1.6.0

The shape first ran exactly as it did when it hung the board,
`timeout 120 sudo -E taskset -c 4 ./build/rk3576_mm_corr 128 2048 2048 1`, as the only device
action after a passing control. It returned clean in 16.583 ms.

It then ran 18 clean repetitions across 13 processes. The last 12 of them ran as a repeated
one-test-point-per-process loop, the harness structure that hung the board 3 of 3 times on
1.5.0. The board stayed up throughout, with the backstop count unchanged.

The two discriminating test points are clean, and so are the two shapes never run before, which
had been held back as separate device costs. The hang therefore does not reproduce under the
conditions that produced it. That holds over 18 repetitions of the exact shape, in both the
isolated and the looped harness. [HW sweep, H96 MAX M9, kernel 7.1.7 / `rocket` 1.6.0, 9 shapes at the sheet's per-row repetitions, one cell per process, 2026-08-10]

The per-row repetition counts of the 1.6.0 table sum to 42 (1 + 18 + 3 + 3 + 5 + 3 + 3 + 3 + 3)
[host-computed]. The total entered with the table on 2026-08-10 was 39. Which of the two is
wrong is not recoverable from the record, so the sheet states the shapes and the per-row counts
and no total. The run logs on the board decide it.

## The K-split wedge's register program

The program was dumped with no device [pure, dumped from `tools/rk3576-planprobe.c -r`,
2026-08-06].

## Dropped writes at the drain deadline

### `rocket` 1.5.0

At `dpu_grace_us`=4000 the shape is wrong in 0 of 9, interleaved over three alternating passes,
and the failure returns every time the parameter goes back. The other six shapes in the table
are 0 of 5 each. [HW sweep, H96 MAX M9, measured 2026-08-05]

`rk3576_mm_corr` scores every repetition of an arbitrary shape and prints each wrong
repetition's bounding box. Five repetitions of `M=256 K=2048 N=2048` on the 1.5.0 module at
`dpu_grace_us`=500, pinned, in one process, were 5 of 5 wrong. The five agree on structure:

| rep | wrong of 524288 | rows | cols | wrong / bounding box |
|---|---:|---|---|---:|
| 0 | 19840 | 128..199 | 1504..2047 | 0.506 |
| 1 | 2944 | 128..151 | 1824..2047 | 0.548 |
| 2 | 40576 | 128..239 | 1312..2047 | 0.492 |
| 3 | 17664 | 128..195 | 1536..2047 | 0.508 |
| 4 | 24688 | 128..207 | 1440..2047 | 0.508 |

- The pure plan tiles this shape into two tasks of 128 rows. Every wrong element in all five
  repetitions is at row >= 128, the first row of the second task. Rows 0..127 are exact in
  5 of 5.
- The low column edge wanders (1312-1824), and the high edge is 2047 in all five.
- The wrong count is 0.49-0.55 of the bounding box in every repetition. The front between
  correct and lost is therefore diagonal in (row, column), consistent with a write order that
  advances in both axes. It is not consistent with a contiguous suffix of a channel-group-major
  scan, which would put the high row edge at 255.
- 0 elements of 524288 retain the probe's host-side sentinel in any repetition.

[HW sweep, H96 MAX M9, one shape, one process, 5 reps, 2026-08-06]

The probe, `rk3576_mm_corr`, classifies each wrong element by what it holds:

- The library's `0xA5` device stamp (the DPU never wrote the atom)
- Zero
- An arbitrary computed value
- The probe's host sentinel

A second five-repetition run of the same shape and configuration has `computed`, `zero` and
`no-descatter` all 0. It reads `drop == wrong` exactly, ratio 1.0000, in 5 of 5:

| rep | wrong of 524288 | drop | computed | rows | cols | wrong / bounding box |
|---|---:|---:|---:|---|---|---:|
| 0 | 3968 | 3968 | 0 | 128..155 | 1792..2047 | 0.554 |
| 1 | 17664 | 17664 | 0 | 128..195 | 1536..2047 | 0.507 |
| 2 | 19840 | 19840 | 0 | 128..199 | 1504..2047 | 0.507 |
| 3 | 5760 | 5760 | 0 | 128..163 | 1760..2047 | 0.556 |
| 4 | 22272 | 22272 | 0 | 128..203 | 1472..2047 | 0.509 |

Ten repetitions across two runs and two sessions agree on the confinement to the second task and
on the reach to the last column.

Three runs measured `M=256 K=2048 N=2048` at `dpu_grace_us`=500 on `rocket` 1.5.0. The first
lost elements in 12 of 14 repetitions, 768 to 26112 of 524288 [HW sweep, H96 MAX M9]. The second
and third, five repetitions each, lost elements in 10 of 10, 2944 to 40576. The 14x spread the
sheet quotes is over those ten.

### `dpu_grace_us` on `rocket` 1.6.0

The clean grace-0 result is not an instrument failure. Setting grace back to 500 on the same
boot and the same binary brings the loss back at full strength. The probe therefore keeps its
discriminating power on this kernel. The output BO stamp is reported `0xA5` in both
configurations, so drops are scorable throughout. Every wrong element classifies as `drop`, with
`computed`, `zero` and `no-descatter` all 0.

One DMA-error warning appeared during the grace-500 configurations, and none during any grace-0
work.
[HW sweep, A/B, H96 MAX M9, one shape, one process per configuration, 5 + 5 reps, 2026-08-10]

The warning the grace-500 configurations raised was logged as
`WARNING: drivers/accel/rocket/rocket_job.c:895 at rocket_job_irq_handler`, on the kernel 7.1.7
image with `rocket` 1.6.0 in-tree [HW sweep, H96 MAX M9, 2026-08-10]. The line number belongs to
that build.

## Geometry of the lost output

The loss is anchored at row 128 in 10 of 10 repetitions and at column 2047 in 10 of 10.

With the probe's scoring mask at 1.0 at every K, the reference is zero everywhere and the output
BO is stamped. The wrong set is therefore the dropped set, and the fill is the loss's own
geometry, not an artifact of the input pattern. A fill of one half anchored at a corner is what a
straight diagonal cut across a rectangle gives. A bounding box and a count therefore indicate a
triangular front.

The per-row element map is `ROCKET_MM_CORR_ROWMAP=1` on `M=256 K=2048 N=2048`, with grace 500 as
the inducer on the 1.6.0 kernel. It covers three repetitions and 308 wrong rows in total:

| | rep 0 | rep 1 | rep 2 |
|---|---:|---:|---:|
| wrong rows | 108 | 104 | 96 |
| rows whose wrong columns are one contiguous run | 108 | 104 | 93 |
| rows whose run ends at column 2047 | 108 | 104 | 96 |
| rows whose run starts on a multiple of 32 | 108 | 104 | 96 |
| inversions in the per-row count | 1 | 0 | 0 |
| row 128 width -> last wrong row's width | 736 -> 32 | 640 -> 32 | 640 -> 32 |

The count equals the span in 305 of 308 rows. The per-row width has one inversion in 308 rows.
Every run start is 32-aligned in 308 of 308 rows.
[HW sweep, H96 MAX M9, one shape, one process, 3 reps, 308 wrong rows, 2026-08-10]

The extent distribution runs closer to zero than the large repetitions suggest. Over ten
repetitions at one fixed shape and one fixed grace, the loss spans 2944 to 40576 elements,
0.56% to 7.7%.

## Exposure across shapes

With grace 500 as an inducer on a cured 1.6.0 kernel, the whole grid can be run at depth. At five
repetitions a test point, the grid reproduces the confined-looking picture of the earlier tables
exactly. At twenty-five it does not:

| shape | tasks | 5 reps | 25 reps | rate |
|---|---:|---|---|---:|
| `128x2048x2048` | 1 | 3 of 5 | n/a | 60% |
| `256x2048x2048` | 2 | 5 of 5 | n/a | 100% |
| `384x2048x2048` | 3 | 0 of 5 | 4 of 25 | 16% |
| `512x2048x2048` | 4 | 0 of 5 | 5 of 25 | 20% |
| `1024x2048x2048` | 8 | 0 of 5 | 0 of 25 | 0% |
| `256x2048x4096` | 2 | 5 of 5 | n/a | 100% |
| `128x2048x1024` | 1 | 0 of 5 | 0 of 25 | 0% |
| `128x1024x2048` | 1 | 0 of 5 | 0 of 25 | 0% |
| `128x1024x1024` | 1 | 0 of 5 | n/a | 0% |

`M`=384 and `M`=512 both read 0 of 5 and both fail at 25. Confinement to `M`=256 was an artifact
of the repetition count. The same grid, board and binary read as "confined to one test point" at
5 repetitions and as "reaches at least four shapes" at 25. Depth separates the rate reading from
the boundary reading, and argument does not.
[HW sweep, H96 MAX M9, kernel 7.1.7 / `rocket` 1.6.0, grace 500 as inducer, one cell per
process, 2026-08-10]

## The int32 output and the K split

The decode drives operands whose every accumulator is distinct, so that each one names exactly
one word. The height-bound sweep is `tests/rk3576_i32_height_bound.c`.

Over 40 runs of the shape that shows the zeroed block most, there are 0 failures, and 3 of the
runs redo a task. No redo needs an idle in front of it.
[HW sweep, H96 MAX M9, measured 2026-07-27]

## The op library on the RK3576

The library test, `tests/rk3576_conv_lib_gate.c`, drives the emitter test's own shape table
through the library: 111 pass bit-exact, and 1 refuses.

With `ROCKET_LG_RESIDENT=1`, the same test runs the 173 shapes through a pack plus two
prepacked calls. It compares each call to the transient output element for element, and 0
differ. The test compares outputs, not wall-clock time. It makes two calls because the second
is the first to reuse a held cube and a held output buffer. One call would not notice a stale
byte.
[HW sweep, H96 MAX M9, `tests/rk3576_conv_lib_gate.c`]

## Per-submit overhead and the completion poll

The probe `rk3576_submit_floor` runs the 8×64×32 int8 matmul, n=300, under `taskset` to one
A72. The medians are eight runs across three modules, and they reproduce to within 20 us. A
figure of 439 us, taken earlier in the series, does not reproduce on any module now on the
board.

The DPU interrupt was enabled by a diagnostic patch on the 1.6.0 OOT reconstruction, which
set `DPU_0`/`DPU_1` in the mask at submit. The NPU's shared line counted 1 -> 311 across a
300-submit run of the same probe, one interrupt per submit. The patch left retirement to the
poll, so nothing else moved. The per-submit median under it is compared against a control on
the same base. [HW sweep, H96 MAX M9, 2026-08-13]

### Excluded causes

- *The completion not being routable to the GIC.* Before `0008`, the series' premise was "not
  routable to the GIC". That premise is a measurement of `PC_DONE`, and it never covered the
  DPU pair.

## Retirement on the DPU completion

Retiring on `PC_DONE`, the int8 matmul test passes 42/43 shapes at a 250 us period and 36/43
at 50 us. Retiring on the DPU bit, it passes 43/43 at both, and the whole test set is clean.

On `rocket` 1.6.0, the int32 job without `ROCKET_JOB_NO_DPU_DONE` waits out the backstop in 5
of 5 runs per writer. The wide-output hazard's probe reports a `scope` map that is unchanged
under `0012`.

The 1.6.0 fp16 fences come from `tests/rk3576_drain_probe raw`. At k9 over 32 channels on
120×64 the fence comes at 2.07 ms in 10 of 10 runs. An fp16 job submitted after the
wide-output hazard writes nothing and waits out the backstop in 5 of 5.

The 2026-09-30 slice runs are `tests/rk3576_drain_probe raw fp16d:16:32:28:28:3`, 10 reps a
configuration. After the wide-output hazard, the slice fences at 665 us at grace 500 and
waits the backstop at grace 0. Each holds in 6 of 6 runs. Of the two unhazarded slices that
wrote nothing after the 300 ms idle, one retired at the grace-100 deadline and one at the
backstop.

Under `ROCKET_JOB_NO_DPU_DONE`, on the same probe, the tail is lost in 10 of 10 runs at k9 over
32 channels on 120×64. It is lost in 6 of 6 at k5 and k7 on that plane, and at k9 on 64×64.
Every one of those runs returned rc 0.

A slice submitted after the wide-output hazard at `dpu_grace_us` 0 fences 127 ms after the
submit (6 of 6) [HW sweep, H96 MAX M9, `rocket` 1.6.0, `tests/rk3576_drain_probe raw`,
2026-09-30]. The driver's backstop is 125 ms, counted from the kick, which follows the start of
the submit ioctl [source-confirmed: patches `rk3576/npu/0022` and `0025`]. A job retired at the
backstop therefore fences at least 125 ms after its submit.

### Excluded causes

- *A deadline outrunning the fp16 slice program.* The 2026-07-28 sweep read the fp16 slice
  program as raising no DPU completion. On 1.6.0 the program fences 343-350 us after the
  submit. A deadline counted from the submit cannot bill it a whole grace, so that reading
  does not come from a deadline outrunning it. A slice submitted after the wide-output hazard
  does wait out a grace, so the no-completion reading is not a property of the program.
  Whether the 2026-07-28 sweep's module or its submits produced it is not established.

## The settle wait on the fp16 path

### Measurement before `0017`

Without `0017`, every wide-output submit pays the grace in full, which makes it the fp16
path's largest single cost. A slice costs the poll overhead plus the whole blind settle. At
28×28 k3, `ic=128` runs a median 15.89 ms over 8 slices at the 500 us default and 13.29 at 250.
Each configuration is five runs with disjoint ranges, and the difference is 300 us of a 250 us
cut, once per slice.
[HW sweep, measured 2026-07-28, `tests/rk3576_fp16_split_cost.c`]

Three consecutive full test suites pass at 250. Each suite is 102 conv, 156+7 conv-library
and 43 matmul on each int32 writer, plus 15 refusals and the host regcmd test. They cover the
wide-output programs the grace exists for.
[HW sweep, measured 2026-07-28, `tests/rk3576_fp16_split_cost.c`]

### The 2026-08-06 parameter sweep

The sweep drove each parameter with the other held, five interleaved pairs a parameter, on the
1.5.0 module. The 2.6 ms that the 2026-07-28 sweep measured for lowering `dpu_grace_us` is
already taken at the default `dpu_blind_us`, and moving the grace gives this path nothing.
[HW sweep, H96 MAX M9, 2026-08-06, `tests/rk3576_fp16_split_cost.c`]

The 2026-07-28 figures (15.89 and 13.29 ms) were taken on the default CPU governor, and the
2026-08-06 ones with it pinned. The two sets of absolutes therefore do not compare. One
configuration's spread in the 2026-08-06 sweep (9.25-12.17) is wider than the 2.6 ms the
comparison is about.
[HW sweep, H96 MAX M9, 2026-08-06, `tests/rk3576_fp16_split_cost.c`]

## Per-job core reset and the half-started job

With a per-job `rocket_core_reset()`, the matmul test drops from 43 shapes to 2.

With the NPU pinned active by a half-started job, the convolution tests pass 102/102 and
111/1, while every i32 matmul shape writes nothing.
[HW sweep, H96 MAX M9, measured 2026-07-27]

## The PPU's pad nibbles

Every pool in every vendor capture is padded symmetrically, and a symmetric pad cannot
distinguish the map from its transpose. No per-op test drove an asymmetric pool before
ResNet-18's first max pool.

A one-dimensional sweep pins the map (`rk3576_pool_probe pad`). The sweep drives a single row
with a unit kernel on the other axis. It scores the SoC's line against the window grid each
candidate leading pad implies. The grid follows the nibble at [3:0] on x and [7:4] on y, at pads
0, 1 and 2, on both axes. The `rk3576_pool_probe bound` walk runs max k3 s2 with one leading pad
and none trailing, over planes 16 to 112 and channel counts 16 to 128. With the map in the sheet
it is exact in all 40 test points, and with the pairs swapped every one of them is wrong.

## The pooling output-width allowance

### Allowance measurement

The square-kernel table is the largest exact output width and the smallest wrong one, measured
with `tests/rk3576_pool_probe.c rect` (max, c=32, both pads). Every row of that table is square, so
readings keyed on the width and on the height agree on all of them. Holding one half of the kernel
word against the other at one plane separates them. The rectangular rows use `rect` with
`ROCKET_POOLB_KH` / `_KW`, VALID pad, height 8 and c=32.

### Excluded causes

- *An allowance keyed on the kernel width.* Keyed on the width, the formula gives 128 at 3×1, over
  by 4x, and 64 at 4×2, over by 2x. It gives 32 at 1×3 and 16 at 2×4. A width-keyed allowance
  accepts the two rows where it is over, and their output is silently wrong. Both are `kh > kw`
  pooling windows, which no graph in the corpus has.

### PPU destination address readout

The PPU's destination address function was read out from the hardware. A max pool at kernel 1
stride 1 is an identity copy. A cube whose every atom contains its own index therefore turns the
destination map into an output. Scanning the output for each code gives each atom's address. The
probe is `tests/rk3576_pool_probe.c dst`, over two planes, one register at a time and then both
together.

The plane is 5×3, whose 15 elements round up to 16. A `0x607C` of 240 bytes (the plane exactly) and
one of 272 (odd) both place every atom at exactly the programmed jump. At 224, one atom under the
plane, one atom is lost. That loss is the positive control that says the register is live.

### Placed and split pools

The placed pool is checked at planes of 289 and 49, and at a padded average whose divisor varies
with position. Nothing is written outside the slice. A plane of 64 is the control, where the two
strides agree. A stride one element under the plane is refused (`tests/rk3576_pool_probe.c place`).

The PPU's input-side base part way into a row is HW-validated at 16 + 12 test points. The split
input is bit-exact against a CPU model of the RK3576's arithmetic over six test points
(`tests/rk3576_pool_probe.c split`). The pitch and the column offset compose with the split rather
than excluding it. The row-major configuration of the same handle is the one that must succeed, and
the probe asserts that the output side is still refused.

On a graph, the divisor-lag check scores 872240 lagged elements on the split input over 77 firings,
with zero in the "neither" bucket.

### Inception V3

Refusing the split pools' input join is a graph-wide veto. Joining it takes Inception V3 from 51.0
to 42.1 ms at `bench 100`. The result has the same per-layer distance to TFLite, the same maxdiff
and the same top-5. The other four graphs are unmoved to the tenth, and none of them has a
column-split pool.

Host transposes across all three paths go from 14.51 to 6.23 ms (-8.28). The submit bucket is flat,
17.36 to 17.18 (-0.18), across 8.4 fewer submits. The write guard grows 0.7.

The reach is wider than the four pools' own scatter, which goes from 4.21 to 0.20 ms. A join is
decided by passes that run over the whole graph. One refusing participant unwinds a pass entirely.
Joining the split pools' input changes these [HW sweep, H96 MAX M9]:

- The shared-surface pass gains a source (6 -> 7).
- The concatenation-placement pass gains two (12 -> 14).
- The graph gains 6 adjacent joins.
- The graph gains 15 non-adjacent joins (5072 -> 8249 KiB).
- The graph gains 8 layers inside a chained run (40 -> 32 outside).

The column split is bit-exact on every wide shape the test contains, including the 147-wide max pool
and a 147-wide padded average at five slices. With it, Inception V3 goes from 108.9 to 47.8 ms at
`bench 100`, with the other four graphs unmoved [HW sweep, H96 MAX M9,
`tests/rk3576_pool_probe.c lib`].

## The AVG_NOPAD pooling mode

The constant `ROCKET_RK3576_POOL_AVG_NOPAD` (PPU mode `0x18`) was transcribed from the register map
with no caller in the library and no test point in any test. It was then driven on hardware.

The wrong outputs are exactly the border outputs and no others: 4096 of 18496 at 17×17 c=64
(64 border positions × 64 channels), and 3584 of 8192 at 8×8 c=128. They are wrong at input zero
points -128 and +40 and exact at 0, at the identical geometry [HW sweep, H96 MAX M9,
`tests/rk3576_pool_probe.c lib`].

`ROCKET_RK3576_NET_AVGNOPAD=0` restores the count-include-pad divisor on Inception V3, and is
deterministically wrong on all nine of the graph's padded pools in 8 of 8 runs. The mode bit and the
pad-value fix are therefore a separate and correct change.

That configuration was also markedly slower, against the coverage's prediction [HW sweep, H96 MAX
M9, 2026-08-03]. On `rocket` 1.6.0 it is 0.858-0.879x the default configuration's time per
inference, over eight rotated interleaved pairs of `bench 100` with `NET_RESIDENT=1 NET_CUBE=1` and
the governor pinned. It runs 118 layers in 7 kicks over 20 submits, against 94 layers in 16 kicks
over 49. Both configurations return TFLite's top-1, and the write guard redoes no kick [HW sweep,
H96 MAX M9, 594 MHz, rocket-userspace as of 2026-09-30, 2026-10-12]. With the divisor constant over
position the hazard is unobservable, so the check finds nothing and nothing is redone.

## The poisoning is one hazard, and its cost is a system setting

### The program matrix

The instrument is `tests/rk3576_poison_probe.c`. It runs the program under test, then a canary:
a submit of a kind known to write. It scores the canary and lets the program under test compute
nonsense. Asking whether the program under test wrote cannot tell a dead program from a
poisoned one, which is why the canary is the measurement.

The A-poisons-B matrix runs each of seven program kinds before each of the seven. Each pair is
measured against a canary that is known to write. The matrix splits the kinds exactly.

The fp16 first conv, the fp16 direct conv, and both int32 writers poison every one of the
seven. The int8 first conv, the int8 direct conv and the int8 depthwise poison none of them.
Every A wrote its own surface in every test point, so no test point is a dead program reading as
a clean one.
[HW sweep, H96 MAX M9]

The hazard persists on `rocket` 1.6.0. On kernel 7.2.3-1, each of three programs leaves the
int8 canary empty 20 of 20 at a zero gap. They are the int32 writer, the wide int32 writer and
the fp16 direct conv (`pair i32`, `pair i32w` and `pair fp16d` against `int8fc`).
[HW sweep, H96 MAX M9, measured 2026-09-24 UTC]

On 1.6.0 the driver retires the poisoned submit at the 125 ms backstop with its surface
unwritten. That is 161 of the 161 retirements the gate list's three poison rows scored, each one
matched by a kernel `retiring it` line. [HW sweep, H96 MAX M9, `tools/gates-rk3576.sh`,
2026-09-27]

After the matmul-form fp16 program, the int8 direct and first-conv canaries wrote 20 of 20 each
in one process at zero gap. The program itself wrote as well. The controls in the same run
poisoned as usual: `pair i32 int8d` and `pair fp16d int8d` each left the canary empty 20 of 20.
The matmul-form program's pairs logged no retired job, where the other two logged 62.
[HW sweep, H96 MAX M9, `rocket` 1.6.0, `rk3576_poison_probe` kind `mmf16`,
measured 2026-09-25 UTC]

A sacrificial submit does not absorb the state. After one int32 job, six consecutive int8 jobs
at a zero gap all wrote nothing. For each of the four poisoning program kinds, one poisoning job
followed by six submits leaves every position at 0 of 6. The guard therefore has to be an idle
rather than a cheap dummy job.

On `rocket` 1.6.0, the unmodified packed-image fp16 first convolution leaves 0 of 20 canaries
written [HW sweep, H96 MAX M9, 2026-09-30]. The same program on a whole output stage leaves 20 of
20 written.

### The register conditions

Delta debugging (`rk3576_poison_probe heal`) over the register delta between a poisoning program
and its clean counterpart at the same geometry found both conditions. The method applies the
whole delta, confirms that it stops the poisoning, then bisects. A leave-one-out asks whether
any one register is sufficient. Every negative it returns is equally consistent with a condition
of two, so one-at-a-time sweeps found nothing on the fp16 path. The two conditions are:

- **int32 writers.** `gen_conv2d_int8_rk3576_i32out()` differs from the int8 direct conv in four
  registers, and DPU `0x4010` alone carries the hazard. Restoring it stops the poisoning with
  the program still writing. [HW sweep, H96 MAX M9]
- **fp16 path.** The fp16 direct conv writes the same 96 registers as the int8 one, no register
  is exclusive to either, and 11 values differ. Minimized, the condition is DPU `0x4038` and
  `0x4050` together. Each restored alone leaves the canary at 0/8, and both restored take it to
  8/8 with the program still writing. [HW sweep, H96 MAX M9]

An exhaustive truth table over the six differing bits of those two registers narrows the fp16
condition to `0x4038[4]` and `0x4050[17]`. Every combination that clears both is clean, every
combination that leaves either set poisons, and no other bit matters. [HW sweep, H96 MAX M9]

With both bits cleared, the fp16 direct conv still writes its surface and stops poisoning.
Every one of the five fp16 direct-conv shapes that reproduces a CPU model exactly then computes
wrong.

The four truth-table rows that drive `0x4038`'s low half to `0x80`, while the rest of the
program stays float, write nothing themselves. They still leave the canary at 0/6. "It wrote
nothing" is therefore not a proxy for "it was harmless". A register sweep here that scores the
program under test cannot see it.

### The clearing idle

The sheet's table comes from a 5×7 sweep of `autosuspend_delay_ms` against the inter-submit
gap. Each entry is the gap at which a chain of wide-output submits first reaches 20 of 20
written. [HW sweep, H96 MAX M9]

With the driven power cycle at the stock 50 ms delay, every gate stays bit-exact. The fp16
first conv's whole gate group runs in 2.3-15.6 ms against 107-334. The int32 matmul gate runs in
9.9 s against 16.2. [HW sweep, H96 MAX M9]

### LUT and elementwise jobs as victims

The A-poisons-B matrix covers seven convolution and matmul kinds, and a poisoned B in all seven
writes nothing. The library's whole-surface write guard is built on that, so a LUT job was
measured as an eighth victim kind.

Behind an int32-output job, the table-load-plus-consumer pair in `tests/rk3576_lut_probe.c`
comes back with its surface untouched. Every failing span reports `surface untouched 8192/8192`,
and every passing one reports `0/8192`. The rate is 4 of 20 behind one wide-output predecessor,
interleaved. [HW sweep, H96 MAX M9]

Scored without a stamp count, a poisoned LUT job reads as a surface written in full, correctly
sized and constant. Every one of 8192 sampled inputs reads `-86` against a ramp, at
`max |diff| 213`, which looks like the consumer running against a table it never loaded. The
probe's `SENTINEL` is `0xAA`, which it memsets into the output BO before every submit, and
`(int8)0xAA` is `-86`. A job that wrote nothing hands the model back the probe's own fill. A
scorer that only knows "expected vs got" reads that fill as a wrong constant.

The probe reports how much of the surface is still at the stamp, so the instrument separates the
two readings.

Only the first span of a run can show it. The probe's `mode_gate` sleeps 150 ms between spans,
longer than the ~100 ms idle that clears the hazard, so three spans always pass alongside it.

The library's activation entry is a victim at the same rate. Its `0 of 24` was its own guard
firing, not an immunity. In one interleaved run behind the same predecessor, at N=20, the raw
probe fails 4 of 20.

The gate `rk3576_act_gate` fails 0 of 20, with its guard firing in 3 of 20. With no predecessor
at all, the gate fails 0 of 20 and the guard fires in 0 of 20. The two counts, 4 of 20 and 3 of
20, are one rate. [HW sweep, H96 MAX M9]

The guard's line names the poisoned job on the first call of the process, and after it the gate
passes its 180 and 28 cases. [HW sweep, H96 MAX M9]

The gate mode `rk3576_add_probe gate` is a DPU-only program like the LUT pair. Behind
`rk3576_pool_probe avg` (the only pooling mode that submits an int32-output matmul), it reports
nothing written at all ten of its shapes, not only at the first one. The ten submits sit inside
one active window. [HW sweep, H96 MAX M9, 2026-08-04]

The elementwise failure rate is 2 of 6 runs behind that predecessor. It is 0 of 6 behind an idle
and 0 of 6 behind `pool_probe gate` (pooling with no int32 job). That separates the trigger from
the predecessor's kind.

Guarded the same way as the library, the probe is 6 of 6 green in the poisoned position. A single
power cycle repaired 4 of them. One cycle therefore clears the state for the rest of the run,
which is what "all ten fail" and "one redo is enough" say together.
[HW sweep, H96 MAX M9, 2026-08-04]

The probe runs the same guard the library does. It redoes a wholly untouched surface after
`rocket_rk3576_power_idle()`, up to the library's eight attempts, and prints the redo count
beside the span. The guard turns `rk3576_lut_probe`, intermittently red at its list position
without it, into a deterministic gate without hiding the hazard. Behind the predecessor that
puts the unguarded probe at 4 of 20, the guarded probe fails 0 of 20. Its redo column is
non-zero in 7 of 20, and a single power cycle repairs every one of them (7 redos over 7 runs).
[HW sweep, H96 MAX M9]

The pair is one submit of two tasks, and a dead task 0 with a live task 1 would produce exactly
the LUT signature. That reading predicts a rate of zero with the table load duplicated
(`ROCKET_LUT_DOUBLELOAD=1`, job load/load/consumer). Interleaved against its own control, 40
each, the result is 10 of 40 against 14 of 40, not zero. [HW sweep, H96 MAX M9]

Of six `pool_probe` modes, only `rk3576_pool_probe avg` reaches the LUT failure. That reads as
"a padded average pool poisons the next process's first LUT submits". The same mode also submits
an int32-output matmul before half of its pools, and no other mode submits one at all. The
wide-output writer and the `AVG_NOPAD` divisor mode are therefore confounded across every
configuration of that comparison. A predecessor is read for what the whole process did, not
for the program it is named after. [HW sweep, H96 MAX M9]

Three knobs separate them (`ROCKET_POOLA_PRE`, `ROCKET_POOLA_EXCL`, `ROCKET_POOLA_PRE_ONLY`).
With no int32 job in the predecessor the rate is 0 of 48, including the padded average pool alone
and both divisor modes. With one it is 24 of 84, across three different binaries. The int32 job
with no pool at all is 4 of 12. The pooling mode does not move the rate (7/12 with `AVG_NOPAD`
against 5/12 with count-include-pad). [HW sweep, H96 MAX M9]

Putting a pooling or an activation process between the two takes the LUT failures to 0 of 24.
That is not evidence that a submit absorbs the state. A process boundary contains an idle, and
the idle that clears the state is twice `autosuspend_delay_ms`, about 100 ms at the stock 50.
Submits alone do nothing to it (six after one poisoning job, every position 0 of 6).

### The `rk3576_lut_probe load` positive control

The probe's `load` mode, which submits the table-load program alone, reports a false negative.
Its positive control for "the program executed" is the scratch BO moving off its sentinel. That
control reads zero: 0 of the scratch's whole 4096 bytes moves, in three consecutive runs. In the
same state `gate` is exact in all four spans, so the table demonstrably loads. The program runs,
and its dummy cube does not write that BO. [HW sweep, H96 MAX M9]

Only `gate` is in the gate list, so nothing asserts on `load`.

### An externally reported hazard on the w4a16 RDMA path

The source is the gahingwoo/charsiu README (ROCK 4D, their v7-lineage kernel, 2026-08-15, not
reproduced here, see SOURCES.md). Their bisect has the same shape as the one above. It runs one
w4a16 job per register mask, judged by an int8 job in a separate process that is itself long
byte-exact. The two registers differ from their int8-path values in bit 30, in opposite
directions. Setting all four RDMA registers together, the vendor's own combination, also
triggers it.

### A whole output stage that does not poison

The three DPU words take the int8 direct canary from 0 of 12 to 12 of 12
[HW sweep, H96 MAX M9, rocket 1.6.0, 2026-09-30]. The gap is zero, the program still writes,
and every pair and every single one of the three leaves the canary at 0 of 12. They are three
members of the `charsiu` project's wide output stage (`src/job.c`). Delta debugging over that
whole stage found them (`rk3576_poison_probe heal` under `ROCKET_PP_ASET`).

After the dense int32 writer, the int8 direct, int8 first conv, depthwise and fp16 direct
canaries each wrote 20 of 20 at a zero gap. It ran 80 submits back to back with no idle, and
every one wrote. None was retired at the backstop. In the same session the unmodified writer
left the fp16 canary empty 20 of 20, and a chain of it stopped at its first job.

The fp32-output conv was scored by `tests/rk3576_fp16_fp32out_probe`. The operands fill all
sixteen input channels, and the planes are 8×4 and 12×10 at 32 output channels and 5×3 at 16.
The conv leaves the int8 canary clean 20 of 20. Eighty of its own submits ran back to back with
no idle, each written and none retired. With the regcmd encoder's float `0x4038` kept in place of
`0x53`, the canary stays empty 20 of 20.

### Software support

`rocket_conv2d_fp16_rk3576()` on the whole stage was timed by `tests/rk3576_drain_probe gate`,
6 calls a shape. At k9 over 32 channels on 120×64, the comparison figure in that measurement was
115-193 ms a call, and over two slices 227 ms. [HW sweep, same board]

### Excluded causes

- *A sacrificial submit absorbing the state.* Six submits after one poisoning job leave every
  position at 0 of 6, for each of the four poisoning kinds.
- *The state living in the DPU register file.* Neither program writes nine of the DPU registers
  (`0x4000`, `0x4008`, `0x4040`, `0x4054`, `0x4064`, `0x4068`, `0x4098`, `0x40a0`, `0x40c4`).
  When the next job writes those nine explicitly, it stays poisoned. The canary already writes
  `0x4038` and `0x4050` at their int8 values and is poisoned anyway.
- *`0x4038[4]` and `0x4050[17]` as the cause alone.* The matmul-form program sets both and does
  not poison, and a whole output stage that keeps both set cures the int32 writer.
- *The first task of the next submit dying.* Duplicating the LUT table load leaves the rate at
  10 of 40 against 14 of 40.
- *The setup submits of `r76_conv_int8_run()` standing between the predecessor and the LUT job.*
  A sacrificial submit is measured not to absorb this hazard (six after one, every position 0
  of 6), and the guard fires on the LUT job itself. [HW sweep, H96 MAX M9]
- *The padded average pool as the trigger.* With no int32 job in the predecessor the rate is 0
  of 48.

## The direct path's drain deadline

Restoring `dpu_grace_us` to 500 after the run at 750 brings the failure back
[HW sweep, H96 MAX M9, 2026-08-05]. It returns at the same heights with the same signature.

The six-row table comes from `rk3576_conv_lib_gate rowmap`, walked row by row with three runs
per height (618 heights). One output-channel group never reaching the work band is why the k7
test points recorded at one or two groups all agree with one another.

### Work and output size

The five further test points use the same instrument, with one VALID task per height and three
runs each [HW sweep, H96 MAX M9, 2026-08-05]. They separate the two laws by factors, not by
margins, so the contradiction is direct rather than statistical. The test point `k7g4-224` is
wrong 3 of 3 at 19 output rows.

### Refused heights in a walk

The heights above a refusal read as exact. The `k7ic64-g2` test point is wrong from 34 to 45,
and then reports `0/3 wrong` from 46 up. That looks like a band with a clean upper edge, exactly
the depthwise path's unhonored-`F` signature. It is not one: those rows are marked
`(3 refused or skipped)`, and the entry emitted no program for them. An untouched output BO is
not evidence about anything. Read the annotation, not the count.

On a kernel that waits for the writing block, the heights below 37 read 0 of 3. Under the
deadline they read 1 and 3 of 3.

### Failure rate below the refused heights

Both the rate and the form of the failure are wider than 3 repetitions can see. The table's walk
is `rowmap` with `MAP_REPS=10`, one task per test point, with the planner choosing `F`.

The 500-against-1000 comparison is interleaved, in three alternating passes on 2026-08-05. It
reads 40 of 90 wrong at 500 against 0 of 90 at 1000, and the failure returns every time the
grace does. The interleave matters because the rate at a fixed height wanders across identical
walks. A single walk's zero is not evidence, and neither is a single walk's localization.

A walk of 3 repetitions localizes the failure to 34-36, and that range is an artifact of the
repetition count. The failure reaches ih 33 too. At ih 35 the first walk on 2026-08-05 read 0
of 10, and the fourth read 10 of 10.

### Extent of the loss

The sweep is `k7ic64-g2` at ih 33-36, `rowmap` with `MAP_REPS=10` and `MAP_SEQ=1`, every
repetition printed, 131 wrong repetitions [HW sweep, H96 MAX M9, 2026-08-05]. The 107 of 131
that fall between the two forms lie in a range a two-mechanism reading requires to be empty.
At grace 500 the severe form is rare enough that walks of 3 to 10 repetitions read the two ends
as two separate events. Such walks did so on three earlier occasions.

The stamp is `librocketnpu`'s sentinel, `rocket_rk3576_sentinel_on`, on by default. In 131 of
131 wrong repetitions, every wrong element still holds it, and zero elements hold data.

Nothing in this test separates the DPU's write order from another cause of the same channel
ordering, so the write-order reading stays a hypothesis.

## The pooling divisor lag

### Divisor readout

The lag's divisor is read out, not fitted. A wrong-element count cannot say which function the PPU
evaluated, so a failing output is scored against four candidate divisors at once. The positions are
read out as a map. The map shows `.` where the PPU agrees with the model. Elsewhere it shows the tap
count that reproduces what the PPU wrote, solved for rather than proposed. The map below is channel
288 of Inception V3's 17×17×768 branch average, one occurrence:

```
 . . . . . . . . . . . . . . . . .      rows 0-7 exact: the lag had not started
 ...
 . . . . . . . . . . . . . . . . 9      x=16 divided by 9 where the model says 6
 . 6 . . . . . . . . . . . . . . 9      x=1  divided by 6 where the model says 9
 ...
 6 4 . . . . . . . . . . . . . . 6      the last row, where the vertical count changes too
```

Every one of those positions is the divisor of its raster predecessor. That includes the `6` at the
last row's column 0, whose predecessor is the previous row's last column, and which no column-only
lag explains.

### Excluded causes

Over whole occurrences, the candidate divisors fit as follows:

| candidate divisor | fit over whole occurrences |
|---|---|
| count-include-pad | a third |
| a one-column lag | 89% |
| a one-row lag | 22% |
| the previous window's divisor | **18368 of 18368** |

- *The wide-output hazard.* An int32-output matmul in front leaves the rate where it was, and the
  lag reproduces standalone with nothing before it. The standalone probe is
  `tests/rk3576_pool_probe.c avg`, at about 1% of calls at an 8×8 plane.

### Repair

With the check on, 0 of 1200 probe iterations are wrong, and 12 of 12 Inception V3 oracle passes are
clean. Without it, the same test is wrong on 4 of 12. A redo budget of eight was measured to run
out, which is why `ROCKET_RK3576_POOL_LAG_ATTEMPTS` is 32.

### Superseded reading of the rate

One superseded reading called the graph rate a property of the layer. The superseded sentence read
"In a graph the rate is about a third, not 1%, and it is a property of the layer". Standalone, the
probe reproduces the lag on ~1% of calls at an 8×8 plane. The composition result under "What sets
the loud/quiet rate" supersedes the per-layer reading, and the per-graph reading before it.

### Revision history

- **Extent of an occurrence, 2026-08-03.** The first description of the lag read: "Once the lag
  starts, it persists over a long run of channel groups and then stops. A failing surface is
  therefore a contiguous stretch of groups". The extent readout the same day superseded it.
  Every one of 1109 occurrences ends at the last channel group, and so do 656 further
  occurrences under an applied DRAM load (§"Lag extent and the tail-only check" below). The
  sheet states the suffix form.

## What sets the loud/quiet rate

### Composition sweep

The same geometry, entry and binary, called back to back, swing between ~0% and ~95% depending on
which other buffers the process allocated. The test point is `288x35x35`, 3×3 stride 1 SAME, pad
excluded, with the check off, under `tests/rk3576_pool_probe.c onset`:

| the process | `rm-transient` (transient row-major entry) | `pitch128` (prepacked cube-in, source line stride 128) |
|---|---:|---:|
| that test point alone | **51-59 of 60** | 0-2 of 60 |
| fifteen pooling test points, interleaved | **0 of 20 in nine passes** | **19-20 of 20 in nine passes** |
| the same fifteen, order reversed | 0 of 20 | **19-20 of 20** |
| the same fifteen with `pitch128` removed | **19-20 of 20** | n/a |
| beside `rm-prepack` only | 0-1 of 30 | n/a |

### Excluded causes

- *The call's position.* Reversing the order runs `pitch128` first and `rm-transient` last, and each
  keeps its rate.
- *The geometry.* Every test point in the composition table is the same pool.
- *The source cube's offset inside its buffer.* Five whole-group offsets (1, 2, 3, 5 and 8 groups,
  five distinct 4 KiB page phases) measure 0-1 of 60 each, against a control at 1 of 60.
- *A wandering rate.* Three repeats of each composition agree to within a few counts, and the two
  states are saturated rather than adjacent.

### The pad and the 64 KiB period

One untouched buffer moves the rate, and the rule it moves along is decoded. The pad is a BO
allocated before the handles, never submitted against and never touched by either side. Its dial is
its page count. The map is periodic in 16 pages, exactly 64 KiB. The table is `rm-transient` alone
at `288x35x35`, one process per point, N=60, by `ceil(pad/4096) mod 16`:

| pad pages mod 16 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| of 60 | 52-58 | 51-60 | 46-57 | 27-40 | 58-60 | **0-1** | **0-2** | **0-3** | 5-17 | 53-57 | 55-60 | **0-1** | **0-2** | **0-1** | **2-8** | 51-59 |

Every residue is 6 to 10 independent samples drawn from three sweeps. A 4096-step sweep covers
pages 0-100, and a repeat of pages 0-31 reproduces it test point for test point. A sweep of pages
200-231 gives the identical residue map. The period is therefore exact over at least 950 KiB of
offset, and the phase is absolute, not local. Four measurements pin what the pad takes:

- **It is page-quantized to the byte.** A pad of 16384 bytes (4 pages) measures 59-60 of 60, and one
  of 16385 (5 pages) 0 of 60. At 57344 bytes (14 pages) the rate is 5-6, and at 57345 (15 pages) it
  is 53.
- **It is the total page count, not the buffer count.** Sixteen one-page BOs measure the same as one
  16-page BO (55-58 against 55-56). Five one-page BOs measure as one five-page BO (0 against 0), and
  twelve as one twelve-page BO (0-2 against 0-1). Two two-page BOs fall on the loud four-page test
  point (58-60).
- **Freeing the pad restores the unpadded state exactly.** A five-page pad measures 0 of 60 held and
  57-60 of 60 freed. A fourteen-page pad measures 5 held and 55-59 freed.
- **The period is a constant of the address space, and the loud set is a function of the
  geometry.** At `288x25x25` the same 16-page period appears with only residue 2 loud (27-50 of 60).
  Residue 0 is weakly loud (6-13), and every other residue measures 0-1.

The pad therefore occupies live IOVA space below the job's own buffers. [HW sweep, H96 MAX M9,
2026-08-05, `tests/rk3576_pool_probe.c onset`, `ROCKET_POOLO_PAD` / `_PAD_N` / `_PAD_FREE` /
`_ONLY` / `_SKIP` / `_REV`]

### The prepacked cube-in path

The composition is the second half of one rule rather than a second mechanism. A prepacked test
point needs both a phase and company, and neither alone is enough:

- **The phase.** In the fifteen-test-point process, a pad sweep over pages 0-31 gives a map periodic
  in 16 pages. The `pitch128` test point is loud at residues 14, 15 and 0 (39-40 of 40), and partial
  at 1 (10-13). It is <= 5 of 40 at the other twelve. Two sweeps reproduce it test point for test
  point. That three-wide band is `pitch128`'s, and its width is not a constant of the rule. The same
  sixteen-phase sweep measures `rm-transient` loud over eight or nine of the sixteen residues (see
  the full map below).
- **The company.** With `pitch128` set up alone, no pad reaches a loud test point at any residue or
  at any magnitude. It measures 0-4 of 40 across 48 points, at pages 100-115, 366-381 and 700-715.
  Two other pooling test points are enough: `pitch48`+`pitch64`+`pitch128` is loud at pad residues
  10, 11, 12, and the four-test-point `pitch*` set at 8, 9, 10.
- **The company does not have to run.** With all fourteen companions packed, allocated and
  cube-joined but never submitted against, `pitch128` measures 40, 37 and 39 of 40 over three
  interleaved repetitions. Conversely, `rm-transient` is the single test point whose removal leaves
  the rate untouched. It is the one test point that runs 40 programs and allocates nothing that
  outlives them.

A leave-one-out map over the thirteen companions shows the dial is the total page count of what was
allocated first, on the same modulus. Every configuration holds fourteen test points, so a count
rule predicts them all alike. Instead, the outcome is a function of the dropped buffer's own page
count mod 16. It holds with no exceptions across thirteen test points, with three-way agreement in
two residue classes:

| dropped test point's pages mod 16 | test points | `pitch128` after the drop |
|---|---|---|
| 7 | `cube-ctrl` (87), `stride+7` (87), `pitch48` (119) | **39-40 of 40** |
| 9 | `pitch36` (89) | **40 of 40** |
| 0 | `base-g2` (96), `stride+823` (144), `rm-prepack` (no cube BO) | 0-2 of 40 |
| 5, 10, 11, 13, 14, 15 | `base-g3`, `stride+55`, `base-g1`, `base-g8`, `pitch64`, `base-g5` | 0-4 of 40 |

The two maps agree under one fitted parameter: a packed pool handle costs 9 pages mod 16. An
independent configuration confirms that parameter. Dropping `pitch36` (89 pages + one handle ≡ 2
mod 16) moves the four-test-point band from residues 8, 9, 10 to exactly 10, 11, 12.

The band's absolute residue does not reconcile. The three- and four-test-point compositions agree
with each other, and under the same bookkeeping they disagree with the fifteen-test-point one by 4
pages. Some allocation is therefore uncounted. Only the phase's period, its width and how it moves
are measured. [HW sweep, H96 MAX M9, 2026-08-05, `tests/rk3576_pool_probe.c onset`,
`ROCKET_POOLO_PAD` / `_ONLY` / `_SKIP` / `_QUIET`]

### The two loud entries

The two entries follow one rule at disjoint phases, and that is the whole of the "two regimes".
Susceptibility is a real class, and at every phase it belongs to two test points of the fifteen. The
map covers all sixteen residues and all fifteen rows, over two independent sweeps of the same
fifteen-test-point process. Each value is N=40 a test point a phase:

| pad pages mod 16 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pitch128` | **40** | 3-4 | 1-2 | 0-1 | 0-1 | 2 | 2 | 1-2 | 0-2 | 2-3 | 2-3 | 0-1 | 2-3 | 2-3 | **40** | **40** |
| `rm-transient` | 0-1 | 1 | **35-40** | **37-39** | 0 | 0 | 0-1 | **40** | **39-40** | **40** | **38** | **31-36** | 13-21 | **31-33** | 1-2 | 1-2 |
| the other thirteen | \<= 3 | \<= 3 | \<= 3 | \<= 4 | \<= 4 | \<= 3 | \<= 3 | \<= 5 | \<= 4 | \<= 3 | \<= 3 | \<= 2 | \<= 2 | \<= 4 | \<= 3 | \<= 4 |

In one process, at one instant, each entry is loud at its own phase and quiet at the other's. The
two loud sets are disjoint. The transient entry's loudness does not come from allocating per
call, and the prepacked cube-in test point's does not come from its company alone. The phase is
necessary and not sufficient. Across 416 test-point-phase observations (thirteen test points,
sixteen residues, two sweeps), the maximum is 5 of 40, and nothing resembling a band appears
anywhere. Those thirteen test points are therefore not merely unsampled: a test point has to be
susceptible as well as in phase.

The `pitch128` test point (source line stride 128, output 4480) is loud at three residues. The
`pitch64` test point (stride 64, output 2240) is 0-4 of 40 at all sixteen. The two susceptible test
points have different properties. One has the largest stride and output in the set, and the other
is the only entry that allocates and frees its buffers per call.

The wider band of `rm-transient` is its own churn rather than a wider rule [hypothesis]. It
reallocates every call, so the pad selects a distribution of phases rather than one residue. A rate
averaged over that distribution would spread, and would show exactly the partial value the map has
at residue 12 (13-21 of 40 where its neighbors saturate). The spread is untested. The prepacked test
points, whose buffers are allocated once, are the ones whose band is three wide. [HW sweep, H96 MAX
M9, 2026-08-06, `tests/rk3576_pool_probe.c onset`, `ROCKET_POOLO_PAD` at k*4096 for k=0..15,
`_N=20 _PASSES=2`]

### Superseded readings

This result supersedes two readings of the rate: "a property of the layer" and, before it, "a
property of the graph". Both rest on a probe and a graph being different contexts. They are, but
not for any reason to do with the graph. A rate measured for one shape holds for the composition it
was measured in. A coverage decision resting on one cannot be re-scored by re-running the same probe
in a different composition.

## Exposure and cost in the model corpus

### Inception V3's observable pools

Inception V3's nine observable pools were read under `bench 100` over four un-instrumented
repetitions. Eight of them measure 0-9%. Layer 125 (`2048x8x8`) measures 22.9, 24.6, 28.4 and 33.1%.
The pad sweep produces the same rate at the band's partial fourth residue (10-13 of 40). There the
core band saturates at 39-40 of 40, and the twelve quiet residues sit at \<= 5.

Which phase the graph falls on is not predictable from a byte count. The absolute phase is exactly
what does not reconcile under "What sets the loud/quiet rate".

### Check and redo cost

The A/B runs `ROCKET_RK3576_POOL_LAGCHECK` on against off, in three interleaved pairs under
`ondemand` and `bench 100`. It measures 34.0 ms against 32.5 ms (configurations 33.6 / 34.0 / 34.2
and 32.4 / 32.5 / 32.6, disjoint ranges), and 48.0 against 47.0 submits an inference. Check plus
redo is therefore 1.5 ms of 34.0 = 4.4% [HW sweep, H96 MAX M9, 2026-08-06, `rk3576_net_gate bench
100` and `tools/coco_map.py` through the delegate].

The profiler splits the two terms, and totals the check at 1.04 ms an inference over 1247 scorings.
[HW sweep, H96 MAX M9, 2026-08-06, `rk3576_net_gate bench 100` and `tools/coco_map.py` through the
delegate]

### The profiler and the rate

Under `ROCKET_RK3576_POOL_PROF=1`, layer 125 fires 14 of 100, where the same binary un-instrumented
fires 23-33 of 100. The whole graph's redos measure 0.35 an inference against the A/B's ~1.0. The
per-redo cost (283 us clean against 562 us with one redo) is from the profile, at n=86 and n=14.

### Whole-output score

A whole-output score puts only about half the wrong elements on the previous window's divisor at
this plane, and the fraction is a constant. Of the wrong elements, 4304 of 8480, 258392 of 509128
and 532680 of 1049584 match `AVGM_PREV`, 50.75% in all three. The constant holds across
compositions and across totals two orders of magnitude apart, where at `2048x8x8` the match is 18368
of 18368. The check scores only the observable positions (those whose raster predecessor has a
different tap count) and finds 100% there. The two are therefore not in conflict on their face, but
nothing accounts for the constant, and it is unexplained. The remainder are positions that the
`AVGM_PREV` model's row-wrap guess gets wrong, rather than a second mechanism [hypothesis].

### The neither bucket

The count of elements that match neither divisor is zero in every one of ~70 occurrences. They
span the per-layer and the chained path, across six pooling geometries.

## Coverage cost per graph

The four-graph table and Inception V3's 34.4-34.6 ms (check off) against 40.7-41.9 ms (check
on) date from 2026-08-03. They were measured before the tail-only check, with the check scoring
every channel group [HW sweep, H96 MAX M9, two and three `bench 100` runs, `ondemand`]. The sheet's
current Inception V3 cost, 1.5 ms of 34.0 ms (4.4%), is the 2026-08-06 measurement under
§"Exposure and cost in the model corpus".

### Inception V3 before the tail-only check

Inception V3 is 34.4-34.6 ms with the check off and 40.7-41.9 with it on, a factor of 1.2. [HW
sweep, two and three `bench 100` runs, ondemand, 2026-08-03]

The 1.2 factor predates the tail-only check, and "Exposure and cost in the model corpus" gives the
later figure.

The 40.7-41.9 ms figure holds because no pool of the graph sits inside a chained kick. With one
placed there, the same graph runs 52.8-89.5 ms, a mean of 70.6 over eleven runs.

### Check and redos by placement

The chain and pooling profiles split the two costs (`ROCKET_RK3576_INT8_PROF=1`,
`ROCKET_RK3576_POOL_PROF=1`). The chain's `lag check` bucket is its own, out of `verify`, and `redo`
spans submit, verify and stamp together. The wall-clock time combines both terms, so the comparison
reads the profile buckets per term. Per inference, over four `bench 50`/`bench 100` runs:

| | check | redos |
|---|---:|---:|
| every pool its own submit (the default placement) | **8.45 ms** | **2.25** |
| with a pool inside a kick, per-layer calls | 7.0-8.8 ms | 1.5-1.7 |
| with a pool inside a kick, inside the kicks | 1.9-4.4 ms | 1.1-2.9, at 4.8-13.2 ms |

With the column-split pools taking a cube in, the check is 6.82 ms of a 43.1 ms profiled inference
(16%). Before that join it is 8.45 of 51.0 (17%).

The pool-inside-a-kick figures are the configuration the library refuses. The section "Chained-run
refusal for pools whose divisor can lag" gives what the same graph costs without one.

## The pooling path's read interval is three things, and only two of them are the guard

The read-interval table was taken under `rk3576_net_gate bench 100` [HW sweep, H96 MAX M9,
`rk3576_net_gate bench 100`, ondemand, 2026-08-03]. A re-reading at a later wall-clock time, over
15.2 pool calls, has the same terms in kind as the 43.1 ms reading. Every one of them moved.
Charging the whole of `read - lag` to the guard overstates it by 3.5x. With the per-call pooling
entry unused, the profile prints 0 lines over 100 inferences on each of MobileNetV1, MobileNetV2,
ResNet-18 and Inception V1.

### Inception V3's profiled time

The whole of Inception V3's wall-clock time closes to 98%, against a 43.1 ms total with both
profilers on:

| bucket | ms | parts (ms) |
|---|---:|---|
| conv chain | 20.61 | submit 11.96 (of which 10.89 is the fence), verify 3.63, stamp 2.11, feature 1.61, de-tile 1.27 |
| conv per-layer | 8.14 | submit 4.96, plan+gen 1.57, feature 1.31, readback 0.29 |
| pooling | 13.40 | |

With both profilers on, the instrument itself is 1.5-2 ms of that total. The unprofiled time is
40.7-41.9 ms.

### Revision history

- **The first reading, 2026-08-03, over 16 pool calls.** The split under
  `ROCKET_RK3576_POOL_PROF=1`, Inception V3 per inference, `rk3576_net_gate bench 100`,
  `ondemand` [HW sweep, H96 MAX M9, 2026-08-03]:

  | term | ms |
  |---|---:|
  | divisor lag check | 6.82 |
  | fence wait | 3.22 |
  | de-tile | 1.48 |
  | sentinel stamp | 1.09 |
  | closing fini | 0.40 |
  | submit ioctl | 0.25 |
  | host cube scatter | 0.13 |
  | sentinel scan | 0.00 |

  From that reading the write guard was stated as `stamp + scan + fini` = 1.5 ms, against the
  5.1 ms of the `read - lag` interval. The sheet's table is the re-reading taken later the same
  day, over 15.2 pool calls, with a `redos` row added. Its guard sum is 1.43 ms
  [host-computed: 1.05 + 0.00 + 0.38]. The validation text's "overstates it by 3.5x" belongs to
  the first reading.

## Lag check cost

The two optimizations below are worth 30% between them and not a factor, because the check is a
pass over memory. Scoring 16 channels at a time rather than one is worth about 20 ms of the check's
own arithmetic. With that in place, at 13.1 ms an inference:

- The NEON tap loop takes the check from 13.1 to 11.5 ms (-12%).
- Hoisting the rounding's divisions takes it from 11.5 to 10.2 ms (-11%). The reciprocal is built by
  `r76p_div_init`, and is the same function rather than a second one, since `r76p_avg_round()` calls
  through it.

## Lag extent and the tail-only check

### Occurrences

The extent is recorded as each occurrence's reached set rather than its element count, which is
what makes a cheaper check answerable at all. Over 1109 occurrences on six of Inception V3's
geometries (channel counts 256-2048, planes 8×8 to 35×19, both the per-layer and the chained path):

- Every one is contiguous, 1109 of 1109.
- Every one ends at the last channel group, 1109 of 1109.
- No element matches neither divisor, in any of them.

Under a dialed DRAM load, 656 further occurrences all reach the last group.

### Tail-only check

The ondemand result is three interleaved repeats a configuration, at a within-configuration spread
of 0.4. The measured saving is 4.80 ms. The test list measures 39.0 -> 33.7 [HW sweep, H96 MAX M9,
`rk3576_net_gate bench 100`, 2026-08-04].

Detection was compared over three runs of 100 inferences each way. Redos per inference measure
0.92 / 1.85 / 1.30 with the full check against 1.22 / 1.59 / 1.74 with the tail check. The
within-configuration spread is larger than the difference between the configurations. The check is
required either way: `ROCKET_RK3576_POOL_LAGCHECK=0` failed 1 of 2 bench runs. Neither the redo
comparison nor a passing test can see a lag that stops short of the last group. Both count what
the tail check did detect.

The write guard's cheaper forms stay default-off. What separates the tail check from them is that
the evidence bounds the class it would miss: 1109 of 1109 occurrences reach the last group. The
missed class of `GUARD_PER_KICK`, a program dying mid-stream while the last one still writes, cannot
be induced by the instrument at all.

## Lag rate per layer in Inception V3

### Per-layer counts

The table counts Inception V3's nine padded average pools per handle
(`rocket_pool_int8_rk3576_lag_counts()`) over `bench 100`:

| layer | shape | fires / checks | discriminating elements a check |
|---|---|---:|---:|
| 13 | 192×35×35 | 0 / 202 | 13055 |
| 22 | 256×35×35 | 0 / 202 | 17407 |
| 31 | 288×35×35 | 68 / 270 (25%) | 19583 |
| 50 | 768×17×17 | 98 / 199 (49%) | 27642 |
| 62 | 768×17×17 | **0 / 101** | 27648 |
| 74 | 768×17×17 | 3 / 104 (3%) | 27648 |
| 86 | 768×17×17 | 6 / 107 (6%) | 27648 |
| 110 | 1280×8×8 | 226 / 442 (51%) | 23040 |
| 125 | 2048×8×8 | 116 / 217 (54%) | 36864 |

Layers 13, 22 and 62 have never been seen to fire at all. The last column counts the scored
elements whose two candidate divisors would give different values on this call's data. A layer that
never fired because its border windows round the same either way would show a zero there. None
does. An earlier reading held that the rate is a property of the layer, not of the geometry. It
held that the placement sets the rate, not the program, the geometry or the data.

### Standalone against the graph

The `rk3576_pool_probe rate` mode runs the same three shapes standalone, 60 calls each, with the
library's check turned off:

| shape | standalone | in Inception V3 |
|---|---:|---:|
| 768×17×17 | **0 / 60** | 0-49% |
| 288×35×35 | **55 / 60 (92%)** | 7-27% |
| 2048×8×8 | **1 / 60 (1.7%)** | 22-82% |

Under `rate`, the standalone probe is a 0% context for the shape that fires hardest in the graph.
The probe's "~1% of calls" is therefore the rate of a context rather than the rate of the hazard.

### Excluded causes

- *The gap between submits.* `rk3576_pool_probe rate` runs one geometry at a sequence of sleeps and
  reports the rate at each. The high-rate shape (288×35×35) is 95% at 0 ms and 92-98% at 1, 2, 5,
  10 and 20 ms. The 30x difference between this probe's ~1% and the graph's ~30% is not the
  back-to-back submission.
- *The source output's channel-group stride.* `ROCKET_RK3576_POOL_PACK_SRC=1` packs the source cube
  at the plane exactly (what a cube join hands the PPU) instead of `round4`, and the rate does not
  move in any test point.
- *The input data.* The discriminating-element column excludes it.

## Lag rate and submit structure

A placement can differ in the producer, the physical BO and IOVA, the position in the stream, and
what ran immediately before. The one property measured is whether the pool's program runs inside a
chained kick or as its own submit. Both configurations have:

- The same handle.
- The same register program.
- The same source cube, at the same BO.
- The same data.
- A discriminating-element count identical to the digit.

The table holds three `bench 100` repeats a configuration, at ~100-850 checks a test point, and
neither layer's configurations overlap [HW sweep, H96 MAX M9].

## Memory traffic moves the onset and the rate, the other way

Two contexts are loud: a pool inside a kick, and a pool reading an output the host scattered
microseconds earlier. Both have traffic on the memory system while the pooling program runs. Every
quiet one has less of it. The tool `tools/rk3576-memhog.c` applies that traffic from the CPU without
touching an NPU buffer or issuing an ioctl. The script `tools/rk3576-lag-pressure.sh` interleaves it
against `rk3576_pool_probe rate`. A rate that moves with it is therefore about the memory system,
not the submit or the source output [HW sweep, H96 MAX M9, 2026-08-05]:

| configuration (4 threads, `ondemand` unless noted) | applied | `768x17x17` | `288x35x35` | onset at `288x35x35`, of 18 groups |
|---|---|---:|---:|---|
| idle | n/a | **1 / 1080** | 667 / 720 (92.6%) | point mass at **g2**, mean 2.0-2.2 |
| memcpy hog | 8.3-8.4 GB/s | **2 / 1080** | 570 / 720 (79.2%) | **g0-g6, mean 2.8-3.7**, once g15 |
| memcpy hog, governor pinned | 9.7 GB/s | 0 / 300 | 246 / 300 (82.0%) | not instrumented in that run |
| spin control, same cores, no memory | 0.00 GB/s | 0 / 480 | 459 / 480 (95.6%) | **g2-g2, mean exactly 2.0, 8 of 8 passes** |

`768x17x17` is 0-49% in the graph. At `288x35x35` the fall is monotone in hog threads, and
reproduces at both governors and over four interleaved runs. An earlier 5-pass run measures 91.7%
at 0 threads, 83.7% at 1 and 80.0% at 4. Quiet, the onset is exactly 2.0 with no spread at all in
every spin-control pass. The onset is the sharper instrument of the two: its point mass breaks at
n~50, where the rate needs several hundred calls to separate configurations. This measurement
removes one candidate for the loud/quiet axis, and gives a per-occurrence signal.

The hog saturates at one thread (8.9 GB/s at 1, 8.4 at 4), so the dial has one useful step and not
a range.

The tail-only divisor check is complete only while every occurrence reaches the last channel group,
a property established over occurrences drawn at one load. The `rate` mode counts the occurrences
that stop short. It finds 0 of 656 across all three configurations, at both the
shape whose onset moved and the shape whose onset did not.

## Chained-run refusal for pools whose divisor can lag

The hard failures are over 11 `bench 100` runs, of which 3 ended in `the surface was never
right` and returned `ROCKET_E_DEVICE`. With each pool its own submit, 0 of 7 runs failed [HW
sweep, H96 MAX M9].

Each Inception V3 wall range is over three runs, with `ROCKET_RK3576_CHAIN_POOL_LAG=1` putting
the pool back in a kick. The other four graphs run 5.0 / 6.6 / 6.6 / 7.1 ms in one submit
each, byte-identical, and do not move.

At the older join set, before the column-split pools' cube-in join, the test configuration
measured a mean of 70.6 over eleven runs. With the refusal it measured 50.8-51.4 over seven. The
conclusion was the same.

The evidence is two pools on one graph. Consider a graph whose observable pool sits in a
long chained run at a genuinely low rate. The lag rate measured inside a kick is the reason to doubt
one exists.

The refusal's figures (8 hard failures in 1100 inferences, 40.7-41.9 ms against 46.9-62.8 ms,
173-293 tasks against 132) were measured on 2026-08-03, with the divisor-lag check scoring every
channel group. The tail-only check, also of 2026-08-03, came after them.

## Divisor lag on the cube-in path

The section backs the mismatch between Inception V3's cube-chained and row-major
configurations. Without the divisor lag check on its cube-in path, the disagreement at the
classifier occurs on 4 of 5 runs.

The discriminator is one run of the same graph with `ROCKET_RK3576_NET_AVGNOPAD=0`. That leaves
every join, every program and every chained-run boundary exactly where they were, and only stops the
divisor varying with position. It reads 5 of 5 green, against 4 of 5 wrong with it on. Three of
those four failures were in the cube chain pass, which is one submit per layer and has no kick
in it at all.

With the check extended, `rk3576_net_gate cube --net iv3` is green and is in the gate list.

### Excluded causes

- *The global pool reading the final concatenation as its cube.* This localization rests on a
  single knob.
- *The kick rather than the joins.* This localization rests on a single `NET_KICK=0` run that
  happened not to fire a ~30% hazard.
- *A race in the kick size.* The `NET_KICK_MAX` table reads as a race: 8 passes, 20 and 30 fail,
  40 passes, 41 fails. It is a one-run-per-test-point sample of that same rate.

## Unresolved hazards and limits

### Drain deadline

Raising `dpu_grace_us` from 500 to 750 us takes the planner to exact agreement with the RK3576 on
23 of 23 geometries.

The function `rocket_rk3576_max_task_rows()` charges every output-channel group's slice where a
rung still fits. That charge was fitted to the ResNet stem's 48-row measurement, and that
measurement is the deadline. It costs the stem a fifth row task. The one geometry that keeps the
one-slice model, `56x56 k7 ic 64 oc 64`, is honored in full at a grace of 750. So the charge is
[expected] to be unnecessary as well, and that is a prediction.

### Pooling state across processes

| what runs before | runs with a wrong test point |
|---|---:|
| `rk3576_pool_probe pad` | 2 of 30 |
| a wide-output int32 program | 0 of 30 |
| nothing | 0 of 30 |
| nothing, in an earlier run | 0 of 300 |

It is the same question as `pool_probe pad`'s own unstable unit-kernel test points.

The k3 s2 max pool, the 16-channel first test points, the two-submit clearing and the 2-of-30
rate come from the `rk3576_pool_probe bound` sweep. That sweep is recorded under §"Pool write
guard" (in "The pooling layer"), and backs this section as well [HW sweep, H96 MAX M9,
2026-08-02].

## Cost of the library entries

### Measurement method

The table comes from `tests/rk3576_cost_table.c`: three timed repeats after a discarded first,
one process per row. [HW sweep, H96 MAX M9, kernel 7.1.7 / `rocket` 1.6.0 in-tree,
`dpu_grace_us` 0, `ondemand`, 2026-08-10]

### Revision history

No single row's move against the 1.5.0 table is a result. The 1.5.0 table carried the same
sixteen shapes. Twelve of the sixteen NPU rows read lower in the 1.6.0 table, the four matmul
rows by 9-22% and the three int8 stems by 12-16%. But the CPU reference, which no module change
can touch, moves by up to 25% between the two runs (`conv 256->512 k1` 42.0 -> 52.3,
`conv 32->64 k1` 11.4 -> 12.0). That is the table's own run-to-run spread, measured on a column
with no NPU in it.

The spread is wider than most of the NPU deltas, so the two tables are two readings rather than a
change with a direction. The deltas also do not support the drain deadline as their cause. A job
retiring early is a job whose wall-clock time is short, so curing it predicts higher numbers, not
lower ones.

These rows move much less than a graph does under a host-side change, because of what they
measure. Vectorizing the CHW <-> NC1HWC2 transpose takes a resident MobileNet inference
30.1 -> ~21 ms (1.41x) and moves these rows by 0-30%. Every row here is a transient call. It
also pays the BO allocation and teardown, and packs its weight cube from scratch, so the transpose
is a much smaller share of it. The rows with the least other work move most. Under that change,
`depthwise 64 k3 56x56` goes 3.11 -> 2.11, `conv 256->512 k1` 2.02 -> 1.70, and the int8 stem
3.40 -> 3.09.

The completion poll was moved from 500 us on `PC_DONE` to 50 us on the DPU bit. Under that
change, the rows with the least work per submit move most: `conv 32->64 k1 56x56` at
3.12 -> 2.01 and `depthwise 256 k3 14x14` at 1.42 -> 0.96. The rows dominated by host packing or
by a power cycle barely move.

Three statements in the body were revised:

- The depthwise rows' ratio range read 2.1-3.7x on the table of 2026-07-28. The 1.6.0 table of
  2026-08-10 gives 2.1-3.2x, and the body states that range.
- The cost table's submit cost was stated as about 0.44 ms, the 439 us measured on 2026-07-28 with
  patches `0006`, `0009` and `0012` at a 50 us poll. The per-submit overhead re-measured on
  2026-07-31 at 329-331 us on modules through `0020` and 344-348 us through `0021` and `0022`.
  The 439 us figure does not reproduce on any module now on the board. The body states about
  340 us.
- The stem bullet was written against the table of 2026-07-28, where the int8 stem measured
  3.2 ms against the fp16 stem's 13.7. It stated that the fp16 path waits out the `dpu_grace_us`
  settle because it raises no DPU completion. On the 1.6.0 table of 2026-08-10 the two measure
  3.21 and 13.98 ms. On `rocket` 1.6.0 the fp16 programs raise their own DPU completion
  [HW sweep, H96 MAX M9, rocket 1.6.0, 2026-09-27]. At the 2026-08-10 measurement `librocketnpu`
  submitted them under `ROCKET_JOB_NO_DPU_DONE`, so each fenced at the `dpu_blind_us` settle (the
  0017 parameter, default 250 us past `PC_DONE`). The library stopped setting the flag on the
  fp16 programs on 2026-09-27.

### The host cube scatter

Without the block copies, the matmul ties the CPU reference at M=32 (1.0x), depthwise runs
1.4-1.6x, and the 1×1 convolutions 2.2x. [HW sweep, H96 MAX M9]

The knob `ROCKET_RK3576_INT8_PROF=1` found the convolution entries' two weight-side loops. With
both loops per-element, the submit is 1-3% of the wall-clock time on the layers that cost most,
and the two host loops are 90%. [HW sweep, H96 MAX M9, `tests/rk3576_net_gate.c`]

Straightening both moved the 1×1 convolution rows of the cost table as measured on 2026-07-29, before the block transpose, from 5.6x/15.7x to 5.3x/21.7x. The 1.6.0 table of 2026-08-10 has them at 5.0x and 28.0x. [HW sweep, H96 MAX M9, `tests/rk3576_net_gate.c`]

### The int8 matmul entry

The gate `tests/rk3576_matmul_gate.c` holds `rocket_matmul_int8_rk3576()` against a CPU model
over 33 shapes, which include:

- M with no even divisor
- K and N off the round grid
- Several N tiles
- A row-windowed M
- The two alignment refusals
- Eleven that gate the int32 output exactly

- Superseded: the entry once split a K past one task's contraction through the int32-output
  entry and requantized on the host. Since 2026-08-06 it declines a K with no single-task plan,
  `K >= 6176`, and `ROCKET_RK3576_MM_KSPLIT=1` takes the K split for measurement
  (§"The K-split wedge").
- The section stated the M axis, the contraction width, the throughput along N
  (12-1000 GOP/s at `M*K = 262144`, 2026-07-27) and the two output-channel bounds (2944 channels,
  6 MiB weight cube, 2026-07-27). Those statements are consolidated into `rk3576-regcmd.md`
  §"The matmul as a 1×1 convolution", which states each with its conditions.

### The int8 GEMM against a real CPU GEMM, at LLM prefill shapes

The harness is `ggml-rocket/test-rk3576-sgemm-ab.cpp`, with the CPU arms interleaved. Two
figures are derived from its measurements:

- Projection, 2026-08-05 [expected]: a resident-weight path is worth up to the M-independent
  intercept. At `M`=512 `K`=`N`=2048 that would take the call from 28.2 ms to ~17.3, and the
  ratio against the best CPU GEMM from 4.7x to ~7.7x [expected]. The resident per-column entry
  `rocket_matmul_int8_rk3576_perc_wbo()` has since been measured at that shape. With the BO pool
  off it measured 12.5-13.0 ms [HW sweep, H96 MAX M9, `rocket` 1.6.0, 786 MHz, governor
  `performance`, 2026-09-25]. That figure is not paired with the 2026-08-05 transient 28.2 ms, so no ratio is
  derived from the two.
- The model-weighted 2.39x is arithmetic over the 2026-08-05 A/B's measured per-shape ratios and
  a Llama-3.2-1B layer's MAC shares, with the down-projection on the CPU [host-computed]. The
  2.37x that `perf/rk3576-w8a8-llm.md` calls the port's model-weighted cap adds the W8A8 path's
  host rotation and quantization (2026-08-10). That note gives it as moving from 2.44x. The
  two weightings start from different figures, 2.39x and 2.44x, and the difference is not
  attributed.

### Cost at M = 1 and the decode projection

The decode-shaped rows of `rk3576_cost_table` are ten timed repeats after a discarded first, in
one invocation. M = 1 tracks M = 32 within 3-11% over two runs. On single-row test points,
`ROCKET_MM_PROFILE`'s wall minus accounted reads 0 over 11 calls. The teardown term accounts for
the ~1.3 ms that a profile without the teardown bucket leaves unattributed.
[HW sweep, H96 MAX M9, 7.1.7 / `rocket` 1.6.0 in-tree, `ondemand`, 2026-08-18/19]

The weight-fetch marginal's +-60 us/MB comes from the profiler dumping integer milliseconds over
eleven calls. The agreement with the `charsiu` project's 84.3 us/MB is closer than the
instrument.

## The precision fork: what the op library should be founded on here

The charsiu program's fills are period-free, and their sums are exact in any order. The same job
with the wrong weight layout came back wrong on 64 of 64 outputs, so the check can fail.
[HW sweep, H96 MAX M9, 7.2.3-1, `rocket` 1.6.0, 2026-09-24]

Four fp32-out jobs back to back in one process were exact on every rep [HW sweep, same board,
2026-09-24]. After it, `librocketnpu`'s int8 direct program wrote 20 of 20 in one process at a
zero gap. Controls in the same run were corrupted on 20 of 20 [HW sweep, same board,
2026-09-25].

## Cost against the int8 entry

The device-time figures come from `rk3576_mm_fp16_cost`. The paired resident table's two passes
agree within 2%. [HW sweep, H96 MAX M9, `rocket` 1.6.0, 786 MHz, both clusters `performance`,
2026-09-25 UTC]

## Cost at a prefill's M

The resident pair is timed over two processes with the pool off, by `rk3576_mm_fp16_cost
resident`. [HW sweep, H96 MAX M9, `rocket` 1.6.0, 786 MHz, governor `performance`, A72s,
2026-09-25 UTC]

## The int8 route's host terms

The per-shape host terms come from four processes, 540-560 converged calls a shape each. Two of
the processes also time an FCVTN narrowing of the same activation. The resident costs are 30
rounds, one entry a process, measured with `rk3576_mm_fp16_cost resident`. [HW sweep, H96 MAX M9,
`rocket` 1.6.0, 786 MHz, 2026-09-25]

The pooled 4.4 s and the unpooled 6.1 s are arithmetic over measured per-call terms, since the
fp16 path is not built.

## A whole network

The network test is `tests/rk3576_net_gate.c`. Its input is a blob that
`tests/data/rk3576-net/mknet.py` flattens from a per-tensor-quantized `.tflite`. The blob contains
TFLite's output for every intermediate tensor. A wrong layer is therefore named by its own index,
not by a wrong label 27 layers later.

### Revision history

- The headline table's per-call row read 115 ms, the after-figure of the weight-loop
  straightening A/B (146 -> 115 ms) [HW sweep, H96 MAX M9, 2026-07-29]. Later
  the same day the per-call graph measured 98.8 ms, paired against 33.6 ms with resident handles
  on one build [HW sweep, H96 MAX M9, 2026-07-29]. The table now gives 98.8 ms,
  the reading paired with the next step. The 16 ms between the two readings is not attributed.
- The headline table's top row was the 2026-07-31 configuration, 5.1 ms with the pooling layer
  retiring on its own completion and the graph in three submits. With the pooling layer inside
  the stream the graph is one kick in one submit. It measures 4.9 ms (2026-08-02, "A pooling
  layer sits inside the stream") and 4.9-5.0 ms in the guard tables of 2026-08-03. The TFLite delegate's
  5.2 ms (2026-08-04) includes the interpreter's invoke and the partition-boundary transposes,
  0.3-0.7 ms above the network test on the same graphs.

## The graph planner

The planner is the product, separate from the test. The network test is the planner's first
caller.

Measured as a sequence of optimizations, MobileNetV1-224 takes 98.8-115 ms with per-op entries and transient weights (two readings of 2026-07-29), and ~21 with resident weights. It takes 10.4-10.5 with cube layout and 5.0
with one kick. [HW sweep, H96 MAX M9, 2026-08-04]

The move into `graph/` preserves behavior. Every join, refusal bucket, chained-run boundary, submit and
task count and TFLite distance is identical on four graphs. Inception V3 differs only in two
pooling layers' submit counts, which vary run to run on one binary because they are the divisor
lag's redos. [HW sweep, H96 MAX M9, 2026-08-04]

## The TFLite delegate

Each row is the median of 19 runs. The plans are identical to the blob's, join for join and chained
run for chained run. That identity is the check that two independent lowerings of the same five networks are
the same lowering.

On the four graphs where both take the same stem, the delegate's wall-clock time runs 0.3-0.7 ms
above that of `rk3576_net_gate`. That difference holds the interpreter's own invoke, the two
boundary transposes and the harness difference. None of the three is measured apart, so no cap
is quoted for it. ResNet-18 is 8.8 against the network test's 6.6, because the test takes the
packed-image stem there and the delegate does not.

### Revision history

- The table's EfficientDet-Lite0 row read 120.3 ms against the CPU's 122.7, with 231/267 nodes
  claimed, 85/89 joins and 10 kicks. It was measured on kernel 7.1.6 at NPU 786 MHz [HW sweep,
  H96 MAX M9, 2026-08-04]. That run was the shift-0 ramp with all 7 layers that have a clamped
  channel refused. It predates the inert acceptance (§"Inert clamped channels", 234 claimed) and
  the BS shift word.
- The row now gives the 2026-09-27 measurement on the shift word, where no layer is refused. It
  is the 55.5 ms run of §"Per-axis gain on the BS shift word", not a median of 19 runs. Its
  238/267 is the delegate's claimed count on that build, and its joins and kicks were not read.

## Claim-time refusals

A probe model of three layers shows both halves of the row-window case. Its middle convolution
is ic=256 oc=32 8×8 k5, a 200 KiB resident slice that is past the CBUF pool at a single
output-channel group. Without the claim-time check, `rocket_conv2d_int8_prepacked_rk3576`
returns -5 from inside the partition and the whole model dies. With it, 2 of 3 nodes are
claimed, and the refused convolution runs on the CPU between two delegated partitions.
[HW sweep, H96 MAX M9, 2026-08-04]

Nothing in the model corpus reaches any of these bounds, so a test of them must be written rather
than found. This one is `tflite-rocket/tools/mk_claim_probe.py`. The anti-drift control is
`rk3576_conv_lib_gate claimplan`, which needs no device. Over all 160 shapes of the envelope
table, it asserts the pure plan's verdict against `lib_refuse`, the run's own. It finds 0
accept-a-refusal and 0 over-refusal. [HW sweep, H96 MAX M9, 2026-08-04]

## Detector accuracy

The mAP comes from `tflite-rocket/tools/coco_map.py`. On SSD, an element-wise diff against the
CPU disagrees on 4 of 20 class ids and 75 of 80 box coordinates. Of the 20 scores, 19 agree to
0.02.

Both detectors reach CPU parity once two per-axis fixes are in. EfficientDet-Lite0 is the first
model with per-axis weights to reach the RK3576 through a frontend. Without the two fixes it
scores 0.0000. It emits exactly 25 boxes an image, almost all class 1, with scores pinned in
[0.3125, 0.5]. That pattern is a degenerate head, not drift.

The discriminator for what a per-axis defect costs is a per-axis classifier, whose label is an
assertion a detector cannot carry. The script `mkresnet18.py --per-axis` builds one from the
same weights and the same calibration set as the per-tensor ResNet-18. The quantization
granularity is then the only variable. The scorer, `rk3576_net_ab.py --images DIR`, measures
top-1 agreement against the CPU as a rate over 100 real photographs rather than on one image.
The per-axis build measures 0 of 100 against the per-tensor build's 93 of 100. Two independent
defects lie under that reading, and neither is the arithmetic `rk3576_perchannel_gate` asserts.

EfficientDet-Lite0 has two scores on record:

- The table's EfficientDet-Lite0 score, 0.2826, is the shift-0 ramp with
  `ROCKET_RK3576_PC_INERT` at full acceptance [HW sweep, H96 MAX M9, 2026-08-04]. The default
  per-axis path puts the gain in the BS shift word. It scores 0.2818 against the CPU's 0.2823
  [HW sweep, H96 MAX M9, `rocket` 1.6.0, 2026-09-27].

## Inert clamped channels

- The section's measurements (the 7 -> 4 refusals, 99.3 / 90.3 / 82.8 ms, 0.2809 / 0.2832 /
  0.2826 mAP) are the shift-0 ramp of 2026-08-04, before the BS shift word. Under the shift word,
  `ROCKET_RK3576_PC_INERT` is not read. Every channel whose output cannot vary is programmed as
  its byte. Only a live channel below 1/65534 of its task's largest live scale is refused
  (`rocket-userspace/src/rocket_conv2d_rk3576.c`, `r76_pc_shift_on()`).

### Per-op results

In `rk3576_perchannel_gate`, the `pruned` and `dw-pruned` test points measure 0.0 counts of
distance from an exact per-axis requant on the pruned channels, on the SoC. The `pruned-live`
test point, which differs only in the fold those channels hold, refuses. The accepted layers'
non-clamped channels are close to their own gain, so the ramp's resolution does not limit them.
[HW sweep, H96 MAX M9, 2026-08-04]

### Accuracy before the depthwise CBUF allowance fix

Before the depthwise CBUF allowance fix, accepting the three layers lost 0.043 mAP. The loss
was the CBUF allowance setting, not the per-axis arithmetic and not the graph. Accepting all
three scored 0.2380 against the refusal's 0.2809, and accepting only the two depthwise ones
scored 0.2076.

One of that pair is `blocks_0`: 160×160, ic = oc = 32, k3 s1. Its three row windows all
selected F=512, an allowance setting the depthwise path does not honor. The layer came back
wrong on 28 of its 32 live channels, in exactly the output rows past each window's 51st staged
input row. A forced small window closes the whole gap and more, which localizes the cause to
the allowance setting:

| configuration | mAP | vs CPU's 0.2823 |
|---|---:|---:|
| the refusal | 0.2809 | -0.0014 |
| `PC_INERT=dw`, before the allowance fix | 0.2076 | -0.0747 |
| `PC_INERT=dw` + `MAX_ROWS=48`, before the fix | 0.2832 | +0.0009 |
| `PC_INERT=1` + `MAX_ROWS=48`, before the fix | 0.2826 | +0.0003 |
| `MAX_ROWS=48` alone | 0.2809 | -0.0014 |
| **`PC_INERT=dw`, after the allowance fix, no forced cap** | **0.2832** | **+0.0009** |

Two configurations separate the causes. With `MAX_ROWS=48` alone the score reproduces the
refusal's 0.2809, so the cap changes nothing by itself. With the allowance fixed, the
acceptance reaches the same 0.2832 with no cap forced at all. The allowance defect is in the
sheet's section "On the DEPTHWISE path the two low CBUF rungs do not deliver".
[HW sweep, H96 MAX M9, 500 val2017 images per configuration, 2026-08-04]

### Inference-time protocol

The protocol is five interleaved rounds of 20 iterations, with the governor pinned to
`performance` and the first process discarded. Each configuration's four rounds sit inside
0.6 ms, and the CPU configuration is constant at 121.9 ms. On `ondemand` the same ordering
measures 120.2 ms for `PC_INERT=0`, 110.9 for `dw` and ~99 for the default, with more spread.
[HW sweep, H96 MAX M9, 2026-08-04]

SSD contains no clamped channel and is the control. It is unmoved at 0.2617 against the CPU's
0.2631, and at 17.0 ms [HW sweep, H96 MAX M9, 2026-08-04].

## Channel-range splits

A host readout tried the split on all six layers, over every reachable tile (a multiple of the
output cube's 16-channel atom). It asks the planner the library uses, with the range sliced as a
layer of its own [host-computed, readout, 2026-08-04].

## Channel order of a per-axis output

The de-tile hides the sort completely, which is why every per-op test is exact. It is also
why the graph test's per-axis pass passes, since that pass runs each layer as a materializing
per-op call. The cube path is where the sort shows.

## Per-axis gain on the BS shift word

`rk3576_coeff_c shiftsign` matches 32 of 32 channels under the sign-split model, and 16-17
under either field alone [HW sweep, H96 MAX M9].

The per-axis test is 34 of 34 bit-exact against the extended host model [HW sweep, H96 MAX M9,
`rocket` 1.6.0, 2026-09-27, `tests/rk3576_perchannel_gate`].

The refusals' cost and the tiles' cost are each measured by turning the library option for one
of them, over 8 rotated passes. The 55.5 ms run had both CPU clusters pinned to `performance`,
and recorded 0 retirements [HW sweep, H96 MAX M9].

## Distance from TFLite

The graph test asserts the RELU6 equivalence rather than assuming it.

## CBUF allowances, SAME padding and the stem

A per-op test does not reach these three requirements. Every shape in the per-op correctness
table sits at F=0. The first plane large enough to need an allowance setting is
MobileNetV1-224's second depthwise block. That measurement read the failing axis off as the
kernel. On the direct path the axis is the resident weight footprint instead, and a
square-kernel sweep cannot separate the two. The sheet's symmetry audit has the detail.

## Inference time with per-call weight packing

The graph's measurements run under `tests/rk3576_net_gate.c` [HW sweep, H96 MAX M9].

## Resident weight handles

The speedup is measured with `tests/rk3576_net_gate.c` under `ROCKET_RK3576_NET_RESIDENT=1`
[HW sweep, H96 MAX M9, `tests/rk3576_net_gate.c`, `ROCKET_RK3576_NET_RESIDENT=1`]. The bucket
table is per inference, over 20 iterations.

The per-call and resident entries are compared with the same test and option [HW sweep, H96 MAX
M9, `tests/rk3576_net_gate.c`, `ROCKET_RK3576_NET_RESIDENT=1`].

Two per-call readings are recorded in another section:

- See "A whole network" above for the 98.8 ms and 115 ms per-call readings of 2026-07-29.

### Sentinel stamp cost

Pack-time generation is bit-identical over the 173-shape resident A/B and over the whole
network [HW sweep, H96 MAX M9, 2026-07-29].

For the stamp fold, interleaved bench-100 runs of two binaries over five pairs measure 30.2 ms
against 29.7. Four sequential blocks of 100 each way put 29.9-30.6 against 29.8 and below, with
no overlap. With the sentinel off entirely, the same bench measures 28.5 ms.

#### Excluded causes

- *The stamp's cost is its two cache-maintenance ioctls.* That attribution predicts that
  removing one of the two brackets removes half the cost, and it does not.
- *The bucket profile measures the stamp.* The sentinel-off run measures worse in `plan+gen`
  plus `de-scatter` than either stamped arrangement, while being 1.2 ms faster in inference
  time. [HW sweep, H96 MAX M9, 2026-07-29]

## The CHW <-> NC1HWC2 transpose

The vectorized transpose also leaves the 173-shape resident A/B unchanged. The `memcpy` rate is
from `build/membench`, at 256 MB.

## Cube layout between consecutive layers

The stride sweep is bit-exact at +3, +16 and +64 elements over five geometries, at one row task
and at ten. The task-window stride (`0x1098`) stays derived throughout. The control lays the
same padded buffer out without telling the register, and it differs every time [HW sweep, H96
MAX M9, `tests/rk3576_surf_stride.c`].

### What a producer leaves past its live output channels

The direct-producer result holds for 12544 of 12544 padding elements at `oc` 16, 24 and 144 and
at `k` 1 and 3 (`tests/rk3576_pad_channels.c`).

With the B term given to the tail, at `oc=16 w_zp=12` the probe measured 5776 of 12544 padding
elements away from the output zero point. Those channels are not a live output, so the
correction they lose is one nobody applies.

A test whose producers were all symmetric accepted MobileNetV2's layer 2 -> 3 join, which took
that layer to 0.5% agreement with the row-major path. Every isolated per-op case stayed
byte-identical.

Behind a depthwise producer, 12544 of 12544 padding elements still hold the sentinel after the
job. Read back after an inference, the pack-time fill stands at 12544 of 12544 elements at the
output zero point, at `oc` 16, 48 and 144.
[HW sweep, H96 MAX M9, `tests/rk3576_pad_channels.c`, 2026-07-31]

### Join pass

The four combinations of asymmetric pad and cube chain are measured interleaved at `bench 100`.

The join test checks the arithmetic, because a layout change must keep byte-identity. It runs
in five steps:

1. Run the graph row-major.
2. Snapshot every layer.
3. Join the cubes.
4. Run the graph again.
5. Compare every layer the second run still materializes, plus the final logits.

A wrong byte inside a join reaches the next materialization point, so nothing hides in the
middle of a chain. On the measured run, every materialized layer is identical and the logits are
byte-identical.

The submit saving of about 110 us is measured rather than inferred. The 40 tasks and the
assertions are the same in both configurations [HW sweep, H96 MAX M9, 2026-07-29].

## A retired job is redone whatever it wrote

| Check | Result |
|---|---|
| Backstop forced to 1 us | Eight small entry calls and the chained kick refuse, each count equal to its attempts |
| Gate list, 67 rows | 67 pass in 320 s. Scored retirements equal the kernel's `retiring it` count on every row that submits through an entry, 163 in all |
| Gate list, 69 rows, kernel 7.2.8-1, 594 MHz, the rail switched with the power domain (2026-10-02) | 69 pass in 286 s. 165 scored retirements, equal to the per-row counts |
| The three rows for the wide-output hazard | 52, 55 and 54 scored, every one unwritten |
| Five stall `K`, §"An int8 matmul job can raise no completion" | Scored equal to the kernel's count at each. The entry's output is correct where a write check alone returned incorrect output |

The one kernel retirement the list does not score is `uapi_regcmd_fault`'s, a raw uAPI submit
outside every entry.
[HW sweep, H96 MAX M9, `rocket` 1.6.0, `tests/rk3576_retire_probe`, `tools/gates-rk3576.sh`,
2026-09-27]

## Sentinel guard brackets

The fill and bracket terms are measured directly, rather than inferred from an A/B test. The
sections "Bracket cost" and "Fill and bracket terms of the guard" have them. The fold is measured with
`tests/rk3576_net_gate.c`. [HW sweep, H96 MAX M9, `tests/rk3576_net_gate.c`, 2026-07-31]

### Revision history

- The memsets' share of the guard was given as 15-18%, measured 2026-08-02 on the four graphs
  of that day's tables, with a bracket per layer. With the default that names each BO once, the fill is 17-27% of
  the guard ("Fill and bracket terms of the guard", 2026-08-03), and the body states that.

## One bracket per BO

Keyed on the `rocket_bo` pointer, the redundancy measures zero on every graph, and the test
configuration that must show a positive shows none.

MobileNetV1 is the zero and the control. Inception V3's row is that optimization's own A/B, at
the numbers it was measured on. The other four rows are current as of the measurement
[HW sweep, H96 MAX M9, `rk3576_net_gate bench 100`, ondemand, 2026-08-03].

## Bracket cost

The ladder runs 4 KiB to 4 MiB at 400 iterations a step and takes the median [HW sweep, H96 MAX
M9, `tests/rk3576_prep_floor.c`, 2026-08-02].

## Fill and bracket terms of the guard

Inferred from the bracket cost model, the fill comes out 2.1x too large, so it is measured
directly rather than as a residual. The fill rate independently reproduces the 0.113 that
`tests/rk3576_prep_floor.c` measures on a synthetic ladder [HW sweep, H96 MAX M9,
`rk3576_net_gate bench 100`, `INT8_PROF=1`, ondemand, 2026-08-03].

## Narrowing's fill and bracket shares

The two terms are conjunctive from their rates, and the table is the measured split of that
conjunction. [HW sweep, H96 MAX M9, `rk3576_net_gate bench 100`, `INT8_PROF=1`, ondemand,
2026-08-03]

An earlier figure for the saving, 1.2-1.3 ms, was Inception V1's alone. The table it headed
showed 1.2-3.4.

## Narrowed footprint and ranged ioctl

### Both changes together

The narrowed guard and the ranged ioctls together run on the same submits and tasks. Every
layer is byte-identical, and each graph returns TFLite's own top-1. The run is
`rk3576_net_gate bench 100` [HW sweep, H96 MAX M9, `rk3576_net_gate bench 100`, 2026-08-02].

### Guard forms compared

The `ondemand` table of guard forms is `rk3576_net_gate bench 100` at the default that already
names each BO once [HW sweep, H96 MAX M9, `rk3576_net_gate bench 100`, ondemand, 2026-08-03].
In it, the `GUARD_PER_KICK` column equals the guard-off column on every graph that is one kick.

Inception V3's row is re-measured at two runs a configuration, with its fifteenth concatenation
placed. The other four rows are unmoved by that placement and are single runs. The row's
baseline is not the graph's current default: with the divisor-lag check scoring the tail, the
default is 33.4-33.7 rather than 38.5-38.8. The row is a stored `ondemand` table and has not
been re-taken.

Earlier rows for Inception V1 in this group of sections were measured with its stem on the
direct lowering, and with a bracket per layer. That graph's stem runs the packed-image encoding
at the current default.

### Per-kick coverage of the wide-output hazard

The test runs a bare int32-output program before each inference and feeds two alternating
inputs. Each inference is scored byte for byte against its own clean answer. With either guard,
0 of 144 inferences after the injected program came back wrong, at 51-57 kicks redone per 48
inferences. The runs were MobileNetV1 and ResNet-18 with every layer checked, and MobileNetV1
with the last output only. [HW sweep, H96 MAX M9, 7.2.3-1, `rocket` 1.6.0,
`ROCKET_RK3576_NET_POISON` / `_VARY`, measured 2026-09-24 UTC]

With the guard off, 24 of 48 came back wrong. Every kick being stale is consistent with that
count, since two alternating inputs see a stream that never writes again only on every other
inference. [HW sweep, H96 MAX M9, 7.2.3-1, `rocket` 1.6.0, `ROCKET_RK3576_NET_POISON` /
`_VARY`, measured 2026-09-24 UTC]

The injection must be a bare program. The entry `rocket_matmul_int8_rk3576_i32()` power-cycles
the SoC on its way out, so the next caller never meets the hazard it made. Through it, the same
rows tripped the guard on 1 of 144 inferences. An injection that clears its own hazard passes a
guard that does nothing.

The measurement induces the whole-submit failure. It says nothing about a program dying in the
middle of a stream while the last one still writes.

### Guard forms with the governor pinned

The `performance` table takes three interleaved repeats a configuration on Inception V3, at a
within-configuration spread of 0.3 [HW sweep, H96 MAX M9, 2026-08-04]. Under `ondemand` the
combination is 33.50 against 33.43 for the tail alone, at a 4.7 ms spread, so it is not separable
there.

### Coverage and cost of the narrowed check

The "a fifth to a third" is the per-range work as a share of the saving a fixed-cost-only model
predicts [host-computed]. Its inputs are the default guard, with a bracket per layer, against
the narrowed guard on 2026-08-02 [HW sweep, H96 MAX M9, `rk3576_net_gate bench 100`,
2026-08-02]. They measure MobileNetV1 1.99 / 0.59 ms, MobileNetV2 3.23 / 1.09, ResNet-18
2.67 / 0.69, Inception V1 4.58 / 1.47. Narrowed bracket counts 30 / 85 / 51 / 122 (the
2026-08-03 table), at the 2.24 us fixed cost.

The per-range work, narrowed guard less brackets times the fixed cost, is 0.52 /
0.90 / 0.58 / 1.20 ms. The saving the fixed-cost-only model predicts, default guard less the
same, is 1.92 / 3.04 / 2.56 / 4.31 ms. The shares are 27% / 30% / 23% / 28%. The same
arithmetic on the 2026-08-03 table, whose default already names each BO once, gives 24-49%.

## Cross-layer chaining in one kick

### Read-after-write across layers

The ordering is measured with `tests/rk3576_chain_raw.c` [HW sweep, H96 MAX M9,
`tests/rk3576_chain_raw.c`]. The probe runs two 1×1 int8 convolutions where the second's feature
cube is the first's output, byte for byte. It compares the batched job against two references:
the same pair as two submits, and the second program alone on an output nothing wrote. Over five
planes from 2 KB to 387 KB, 40 of 40 iterations gave the two-submit answer.

The probe sweeps the plane because write visibility scales with the bytes in flight. A small
output would look ordered whether or not the SoC orders it.

### Graph comparison

The MobileNetV1-224 test compares the kicked graph against the per-layer cube-chained graph, not
against the row-major one. A comparison with row-major would test both changes at once, and a
wrong byte inside a kick would then read as a cube-chain defect.

### Cost inside a kick

The cap table is from `bench 50` [HW sweep, H96 MAX M9, 2026-07-31, `bench 50`]. The
MobileNetV2 bucket table is from `bench 30` and `bench 100`
[HW sweep, H96 MAX M9, 2026-07-31, `bench 30`/`bench 100`]. Read as a finite difference, chained run 1
of MobileNetV2 puts its cost in its first four layers, which reads as compute.

### Stream length and the completion wait

The 35-program stream fails 3 of 3 at the 500 us default. It is bit-exact 3 of 3 at a longer
deadline, and 3 of 3 at a `poll_interval_us` of 250 us and up.

### Completion on the task-status counter

A deadline scaled by the task count retires the 75-task layer at t=37554 us, 75 x 500 us to the
microsecond. The counter then reads 67 started of 75, and the layer's output is wrong. Under the
counter-anchored wait, 29 tests pass with the taint word unmoved and zero backstop hits, which
confirms that every job class raises `PC_DONE`.

The 6.4-6.5 ms graph and the 348 us per-submit median were measured on 2026-07-31, before the
library changes above the driver took the graph further.

The same 75-task convolution hung the board twice: once against the scaled deadline and once
against patch 0021 alone. Each hang needed a physical power cycle. The board does not come back
on its own and answers no ping, and a full LAN sweep finds no host accepting SSH. [HW sweep, H96
MAX M9]

A sweep on the driver without the counter-anchored wait reported "the per-layer reference failed
at layer 0" at three lengths in a row. The geometry had passed minutes earlier, and the failures
came purely from the IOMMU already being stalled. On a kernel without `0022`, the `sweep` mode
therefore leaves `wide` out and stops a mix at its first non-bit-exact length.

The instrument is `tests/rk3576_chain_len.c`. Every mode takes three footprint knobs:
`ROCKET_CHAINLEN_C` for channels, `_P` for the plane and `_K` for the direct leg's kernel. The
length, the mix and the footprint therefore move independently. Its defaults are per mode, and
they are the shapes the modes were characterized at:

| mode | channels | plane | direct leg |
|---|---|---|---|
| `uniform`, `mixed`, `geom` | 32 | 32×32 | 1×1 |
| `wide` | 128 | 28×28 | 3×3 |

It asserts the kicked answer against the same handles run one submit each.

The completion wait's forms were timed on MobileNetV1-224:

- The 5.0 -> 6.3 ms on MobileNetV1-224 compares two deadlines [HW sweep, H96 MAX M9, recorded
  2026-07-31]. The first is the 500 us deadline that started at the first poll tick (patch
  `0021`). The second is a deadline scaled by the task count, the first form of patch `0022`.
  The counter-anchored wait is the form `0022` took. It measured 6.4-6.5 ms in 5 submits and a
  348 us per-submit median, unchanged from the scaled deadline [HW sweep, H96 MAX M9,
  2026-07-31]. The scaled
  deadline was squashed out of the series, so neither 6.3 nor 6.4-6.5 is a measurement of a
  shipped intermediate.

## The pooling layer

The first of the three findings below is what made the other two worth building.

### Pool completion

The flag-word measurement uses one binary with the flag word as the only variable. It covers the
two tests whose harness helpers submitted with `flags = 0` (`rk3576_row_pitch`,
`rk3576_chain_pool`), in ms:

| test | module | `flags = 0`, ms | `PPU_DONE`, ms |
|---|---|---|---|
| `chain_pool` | 1.5.0 release, `dpu_grace_us` 500 | 149 | 110 |
| `row_pitch` | 1.5.0 release, `dpu_grace_us` 500 | 229 | 186 |
| `chain_pool` | 1.6.0 candidate, wait untimed | 10300 | 104 |
| `row_pitch` | 1.6.0 candidate, wait untimed | 4804 | 185 |
| `pool_gate` | 1.5.0 release | 2461 | 2452 |
| `pool_gate` | 1.6.0 candidate | 4494 | 2452 |

The untimed wait costs patch 0022's 125 ms backstop times the submit count. The 149 and 229 ms
that read like a baseline on 1.5.0 are the 500 us grace per submit itself. On 1.5.0, `pool_gate`
moves 8 ms against a 2452 ms row, ~16 submits at 500 us, and on the 1.6.0 candidate it moves
2033 ms. No 1.5.0 measurement could therefore have pointed at it. The population is every caller
of `gen_pool_rk3576`: the library entry, which has always named its class, plus four raw test
submits in three files (`rk3576_row_pitch.c`, `rk3576_chain_pool.c`, and `rk3576_pool_probe.c`'s
`run_pool` and `dst_arm`). [HW sweep, H96 MAX M9, 2026-08-05]

With all four named, a full-list backstop count is module-independent (28 on 1.5.0 against 29 on
the 1.6.0 candidate). Before they were named, it differed by ~10. The wide-output class
(`NO_DPU_DONE`) is a separate population and was not enumerated. [HW sweep, H96 MAX M9,
2026-08-05]

Two notes apply to this section:

- The module table's second row was measured on an out-of-tree build of the series through patch
  `0025` (interface 1.6, `dpu_grace_us` 0 meaning "wait for the writing block"). It was measured
  on 2026-08-05, before `rocket` 1.6.0 shipped in-tree in kernel 7.1.7. The 125 ms backstop it waits out is the
  one patch `0022` introduced (`JOB_TIMEOUT_MS / 4`), which `0025` extends to every way a job can
  fail to complete.
- The section restated the operation-enable bitmap (`0x1D` for CNA, CORE, DPU and DPU_RDMA,
  `0x60` for PPU and PPU_RDMA), the completion bits (`0x100`/`0x200` DPU, `0x400`/`0x800` PPU)
  and the pool's wait at `dpu_grace_us` 500 / 250 / 100 (641 / 389 / 213 us). That restatement
  is consolidated into `rk3576-regcmd.md` §"Operation enable" and §"Pool completion".

### Cube-in pooling and residency

The source-stride result is `rk3576_pool_probe lib` at both strides, with
`ROCKET_RK3576_POOL_PACK_SRC=1` forcing the packed one.

### Pool write guard

One observation raised the question. Inside a full test-list run, `rk3576_pool_probe bound`
reported 1 wrong test point of 40 once: 128 elements at `c=16` on a 16×16 plane. That is exactly
one output row across all sixteen channels, which is the shape a dropped block of atoms has.

Measured as a rate, it does not reproduce. The `bound` mode ran 150 times quiet and 150 times
under four streaming-memcpy threads, 6000 test points a configuration. Its sweep is max pooling k3
s2 with a leading pad, over planes 16-112 and channels 16-128. It gives 0 wrong test points
in 12000. By the rule of three, that bounds the rate at 0.025% of test points (95%).

The loaded configuration has its positive control from the same run and the same load. There the
int32 wide writer drops atoms on 30.8% of its row tasks. The memory system was therefore under
the pressure the hazard tracks, and the pool did not follow it. The conv entries and the narrow
int8 matmul were clean at this same load.

The event reproduced in the very next full test-list run, and then on demand. The test runs three
configurations of 30 runs each, which differ only in what runs before each `bound` sweep:

| what precedes each sweep | runs with a wrong test point |
|---|---:|
| `rk3576_pool_probe pad` (pooling with a unit kernel on one axis) | **2 of 30** |
| a wide-output int32 matmul (the source of the corrupted next submit) | 0 of 30 |
| nothing (the control) | 0 of 30, plus the 300 above |

Both failures are byte-identical to each other and to what the test-list run showed. They are
the first two test points of the sweep, at `c=16`. The 16×16 plane has 128 wrong elements, and
the 24×24 one has 2302 of 2304 wrong. The 128-element signature is also exactly the original
single observation.

The unit-kernel test points of `rk3576_pool_probe pad` are themselves unstable, which is why that
mode does not assert. The same unit-kernel program corrupts the next process's pooling here. The
two are treated as one question. [HW sweep, H96 MAX M9, 2026-08-02]

#### Excluded causes

- *A partial write slipping past the whole-output check.* The failures follow the preceding
  unit-kernel pool and no other context.
- *The atom drop under memory load.* The pool shows none under the load that drops atoms on the
  int32 wide writer.

### Pool cube output

The PPU stride result is from `rk3576_net_gate`
[HW sweep, H96 MAX M9, `rk3576_net_gate`, 2026-08-02]. The refusals are from the same test
[HW sweep, H96 MAX M9, `rk3576_net_gate`, 2026-08-02].

The section carried one correction:

- Corrected: the section stated that the pooling program derives its destination channel-group
  stride from the output plane. It stated that no register moving it had been decoded, and that
  the cube entries refuse a slice at any other stride. `0x607C` is the destination channel-group
  stride, honored at any value at or above the plane (§"PPU destination and source addressing").
  `rocket_pool_int8_cube_out_at_rk3576()` programs the slice's stride and refuses only one below
  the plane (`rocket-userspace/include/rocket_pool.h`).

## A pooling layer sits inside the stream

The probe runs `conv -> pool -> conv` as one batched job. It compares that job against two
references: the same three as three submits, and each consumer alone on an input nothing wrote.
Over five geometries (planes 16 to 110, channels 32 and 64), 20 of 20 iterations gave the same
pool output and the same output from the convolution after it.
[HW sweep, H96 MAX M9, `tests/rk3576_chain_pool.c`, 2026-08-02]

Both intermediates are read, so a failure would name the boundary it happened at. The sentinel
separates "never ran" from "ran on stale input", which a fresh BO's zeros cannot.

The earlier kick comparison measured 0.145-0.25 ms a submit, when a graph still had tens of
submits.

## Per-axis weights in a chained graph

A per-axis requant's error is a per-layer quantity. What one layer gives up to the single
`OUT_CVT` shift is not what twenty-nine of them give up, and only a network answers that. The
test `rk3576_net_gate peraxis` runs MobileNetV1-224 with every convolution and depthwise
convolution on the per-axis entry, chained.

The per-axis model comes from re-quantizing this model's own weights per output channel. Each
channel moves from the shared grid onto `max|w_c - w_zp| * w_scale / 127`, with its weights and
bias rescaled. The resulting spread is real and wide.

The 26.6-count worst case is the per-op test's reading, and it stands where it was measured. The
output-channel tile option is never taken on this model.

The pass asserts what a graph is for. No layer refuses, and no layer is far enough from an exact
per-axis requant to be a defect rather than the single shift. The compounded chain still returns
the framework's label. The bit-exact assertion against the chip's own arithmetic stays in
`rk3576_perchannel_gate`. There the planner's `C` multipliers and `OUT_CVT` shift are visible to
a model, and in a graph they are not.
[HW sweep, H96 MAX M9, `tests/rk3576_net_gate.c peraxis`]

Neither of those two tests can see the channel order, and both miss it the same way. Every test
point of the per-op test and every layer of that per-axis pass runs a materializing call. No
per-axis shape in the corpus is consumed as a cube. The corpus's per-axis coverage is a synthetic
re-quantization of one graph, run one layer at a time.

## MobileNetV2

MobileNetV2 is the second network run on the RK3576, after MobileNetV1. It uses the same
harness and the same blob format as MobileNetV1: `rk3576_net_gate --net v2`, with the blob
built by the same `mknet.py`.

### Chain fragmentation on MobileNetV2

The cube chain is the measurement to read on this network, because a residual network is where
the join rules first have something to refuse. The harness counts the refusals by reason
rather than reporting the total.

The offset cube, the padding-channel rule and the producer's group stride were in place. The
remaining-refusal table read one refusal on 63 adjacent pairs: the consumer is not a resident
convolution (the classifier, reading the pool) [HW sweep, H96 MAX M9, 2026-08-01]. It read zero
once a pool could leave its output as a cube [HW sweep, H96 MAX M9, 2026-08-02]. The 62 of 63
in the sheet is the first state and the 63 of 63 the second.

### A cube's base is a plain address, so a concatenation costs nothing

The plain-address result covers 34 placements at three geometries, one of them row-split, with
an unaligned group index (`tests/rk3576_offset_cube.c`).

### Inference time by optimization

The table is measured with `rk3576_net_gate` at `bench 100` (`tests/rk3576_net_gate.c`),
warm. The per-axis result at the end of the section is tagged
[HW sweep, H96 MAX M9, `tests/rk3576_net_gate.c`, 2026-07-31].

## ResNet-18

ResNet-18 is the third network run on the RK3576. Three facts in the subsections below come
from it. Two are registers, and neither is reachable from a MobileNet. The third is about how a
harness reads a graph.

### Model construction

No per-tensor quantized ResNet-18 is published. MobileNetV1 and V2 arrive from
google-coral/test_data already quantized per tensor. Every quantized ResNet-18 in circulation is
per-axis or float. The script `tests/data/rk3576-net/mkresnet18.py` therefore assembles one:

- The timm `resnet18.a1_in1k` safetensors checkpoint (torchvision's parameter layout under
  torchvision's names, read with a 30-line parser rather than a framework)
- Batch norm folded into each convolution
- The ImageNet normalization folded into conv1, so the network's input is the raw pixel and
  the quantized input tensor is the identity on bytes
- TFLite post-training quantization with per-channel disabled

The script asserts both the byte-identity input and the per-tensor weights. A quantizer flag
that silently stopped working would produce a model the harness reads wrong, rather than one it
refuses.

### Revision history

| Quantity | Earlier figure | Current figure in the sheet |
|---|---|---|
| ResNet-18 with every join refusal closed, 11 submits | **10.5** ms, the last row of the first ResNet-18 table. "The three refusal classes" quoted the closure as 18.4 -> 10.5 ms and 25 -> 11 submits [HW sweep, H96 MAX M9, 2026-08-02] | 10.6 ms, in the decomposition and pricing tables |
| ResNet-18 as one kick in one submit | 10.5 -> 9.2 ms, with the chained-run finder's two rules replaced | 9.2 ms |

| Quantity | Earlier figure | Current figure in the sheet |
|---|---|---|
| Saving from closing the twelve refusals | 7.9 ms, 18.4 -> 10.5, the 11-submit figure of the first ResNet-18 table [HW sweep, H96 MAX M9, 2026-08-02] | 7.8 ms, 18.4 -> 10.6, the pricing table's own "none" row [host-computed]. Still 1.8x the 4.4 ms per-pair sum |
| Marginal submit, ResNet-18 from 11 submits to 1 | about 0.13 ms, 1.3 ms over 10 submits, from 10.5 -> 9.2 | about 0.14 ms, 1.4 ms over 10 submits, from the decomposition's 10.6 -> 9.2 [host-computed] |

### Per-optimization decomposition

The decomposition table takes 2 repeats per configuration at `bench 100`. The single-join table
takes 100 iterations and 3 repeats per configuration
[HW sweep, H96 MAX M9, `rk3576_net_gate bench 100`].

The iteration-count Note compares `bench 10` against `bench 100`.

The marginal submit at about 0.14 ms sits just below the 0.145-0.25 band. The
MobileNets measure it lower still (0.05-0.10).

The configuration with no cube layout does the most host allocation. It runs a
`calloc`/`free` of a concatenation buffer per add per inference, on top of the ping-pong. One
explanation of its costlier first inferences is glibc's dynamic mmap threshold rising once
those blocks are freed [hypothesis, not measured].

### The three refusal classes

The pricing table runs `rk3576_net_gate` at `bench 100`, with each class's option off and the
other two on.

The per-pair sum in the sheet prices each refused pair at the single-join table's cost for its
size [HW sweep, H96 MAX M9, `rk3576_net_gate`, 2026-08-02].

#### Revision history

| Quantity | Earlier figure | Current figure in the sheet |
|---|---|---|
| The PPU's destination channel-group stride | Derived from the output plane, with no decoded register to move it, so a pool output was placed only in a slice at the PPU's own `round4(ow*oh)` stride and the library refused any other (2026-08-02) | `0x607C`, honored at any value at or above the plane, read at a 5×3 plane at 240 and 272 bytes against a loss at 224 [HW sweep, H96 MAX M9, 2026-08-03]. `rocket_pool_int8_cube_out_at_rk3576()` takes the stride from the slice and refuses one below the plane |

### One-kick inference time by configuration

ResNet-18's one-kick inference time, every reading in the sheet and in this record, by
configuration. All are on the H96 MAX M9 with resident weights, the cube layout and one kick
in one submit, at `bench 100` unless stated:

| Configuration | Reading | Date | Where |
|---|---|---|---|
| Direct-lowering stem, write guard bracketed per layer, 786 MHz | 9.2 ms, 38 tasks | 2026-08-02 | Sheet: the ResNet-18 decomposition, the current figure for this configuration |
| Same | 9.3 ms | 2026-08-02 | The NEON-block comparison's control ("Pooling layers and the host transpose"), and the "What the wall is made of" table before one bracket per BO |
| Same | 9.2 ms | 2026-08-02 | The control of the concatenation-placement comparison ("Concatenation placement") |
| Same | 9.2-9.3 ms | 2026-08-02 | The control of the Inception V1 packed-stem comparison, and the direct-stem row of "The materialized stem on a graph" |
| Same | layer total 9.15, wall-clock 9.1 ms | 2026-08-02 | "Per-layer scoring overhead" |
| Packed-image stem with materialized pad columns, guard bracketed per layer, 786 MHz | 7.4 ms, 33 tasks | 2026-08-02 | Sheet: "The materialized stem on a graph", and the per-layer column of "One bracket per BO" |
| Packed-image stem, one guard bracket per BO, 786 MHz (the default) | 6.6 ms, 33 tasks; 6.5-6.6 in "One bracket per BO" | 2026-08-03 | Sheet: "What the wall is made of", "Guard forms compared", "Coverage cost per graph", and the 786 MHz column of "NPU clock rate" |
| Same, kernel 7.1.6 with the series `0001`-`0024` in-tree | 6.6 ms | 2026-08-04 | This record, "Validated configuration" |
| Same configuration at 594 MHz, kernel 7.2.8-1, `rocket` 1.6.0 | 7.9-8.0 ms | 2026-10-02 | Sheet: "NPU clock rate" [live] |
| `tflite-rocket` delegate, direct-lowering stem, kernel 7.1.6, 786 MHz | 8.8 ms, median of 19 runs | 2026-08-04 | Sheet: "The TFLite delegate" |

The configurations that are not one kick (18.3 ms at 25 submits, 10.6 at 11, 9.9 at 5) are the
rows of the decomposition and are dated with it.

## Operand producers in a graph

Only the chain pass can see a wrong producer. The oracle hands every layer TFLite's golden
input. It scored all 32 layers exact, and the per-layer table was clean while the label was
garbage. The test is the distance, not the operand.

The same harness resolved operand A by producer on its pooling and add branch, and by
position on its convolution branch. The three affected layers are on the convolution branch,
and the two branches were four lines apart. The defect therefore survived in the per-axis pass after
the chain pass was fixed. There it presented as an unrelated top-5 that read like an accuracy
result.

The count of layers whose operand A comes from a producer other than the layer before comes
from `rk3576_net_gate hostchain`. On the two MobileNets every configuration of every pass
agrees for free, and no test there can fail. A wiring defect therefore passes every test on
both MobileNets, and it reached ResNet-18 twice.

## Per-axis requant on ResNet-18

The discriminator is the graph run three ways on the CPU (`rk3576_net_gate hostchain`, no
device required). There, no part of the hardware is involved, and the operand resolution is
the only variable:

| configuration | ResNet-18 top-1 | TFLite's top-1 falls to |
|---|---|---|
| per-tensor, operands by producer, the control | `suit`, TFLite's own | n/a |
| per-axis, operands by producer | `bulletproof vest`, TFLite's own **number 2** | 2 counts behind |
| per-axis, operand A threaded | `nematode`, outside TFLite's top-5 | **120 counts** behind |

The control configuration makes the other two readable. It is the arithmetic every layer of
the chain pass is bit-exact against, so it must return TFLite's top-1, and it does. Two orders
of magnitude in the logit domain then separate the two candidate causes: 2 counts against 120.
That answer is
stronger than either candidate alone, because both effects are real and only one of them is a
defect.

The test asserts top-5 membership, not the top-1. The threshold is not fitted to this image:
it sits between two measured regimes 60x apart. On all three networks,
`rk3576_net_gate peraxis` passes. A green per-axis result covers one image and one network
family.

## Inception V1 and the concatenation topology

With the concatenations built on the host, the earlier graphs join 28 of 28 adjacent pairs
(MobileNetV1) and 63 of 63 (MobileNetV2).

Inception V1 was chosen for the axes it varies rather than for its size. It is the first
network in the test set to have any of them. The pooling had been tested only at k of 2, 3
and 5 before it.

The build is `mknet.py iv1`. The build script `mknet.py` refuses a model where an operand's
quantization differs, because copying it through would need a requant the blob format cannot
contain. It also refuses an operand that does not start on a 16-channel atom.

All 81 compute layers were bit-exact on the first run. The result is about the
three-network test set as much as about this graph. The set's coverage of the arithmetic held
against a graph shaped nothing like it.

## Pooling layers and the host transpose

The NEON block's output is byte-identical: all four `chain` tests pass, per layer against the
CPU model.

The other three graphs measured 4.9, 6.7 and 9.3 ms (MobileNetV1, MobileNetV2, ResNet-18) with
and without the NEON block [HW sweep, H96 MAX M9, 2026-08-02]. ResNet-18 ran with its stem on
the direct lowering and the write guard bracketed per layer.

The 69-submit Inception V1 graph measures 20.0-20.1 ms in the NEON-block comparison. As the
control of the two placement comparisons later the same day, it measures 19.8 (profiled, 19.82)
and 19.9 ms. The readings are separate runs of one configuration [HW sweep, H96 MAX M9, 2026-08-02].

## Per-layer scoring overhead

`rk3576_net_gate bench N` scores every materialized layer against its golden tensor on the
first iteration only. The score `s` and the drift index it fills are read only on that
iteration. Scoring on every iteration adds the pass inside the wall-clock time the bench reports.

The three older graphs are each one kick with one materialized layer, so the per-iteration
scan costs them nothing measurable. Their layer totals and their wall-clock times agree to 0.05 ms
(4.83/4.8, 6.66/6.7, 9.15/9.1), where Inception V1's are 18.78 and 28.9. An instrument's
overhead is a function of the graph. A test set in which it is zero says nothing about a graph
in which it is not. The gap shows only when the per-layer table is summed against the wall-clock time.
The table prints each layer's share, so that sum is always available.

Scoring the first iteration only leaves the assertions unchanged by construction. The `all`,
`chain`, `oracle` and `peraxis` modes each run one iteration, so `it == 0` always holds, and
every test is byte-identical.

## Concatenation placement

Read-after-write between two chained programs is measured at 40/40 (`rk3576_chain_raw.c`).

The configuration `ROCKET_RK3576_NET_CATN_OVERLAP=0` refuses exactly the overlapping case, and
it is the configuration that separates a disagreement there from any other. The tail fix's
configuration is `ROCKET_RK3576_NET_TAIL=0`.

Placement adds a cost as well as removing others: the second write-guard bracket each placed
producer takes, the guard row of the sheet's term table. A profile of the graph taken before the
change contains no such term.

A profile bucket does not isolate a change's gain. Profiled at both ends of the change, the
transposes fall from 5.28 to 0.60 ms, by 4.68. The
configuration that appears to isolate them says 1.8: `ROCKET_RK3576_NET_KICK=0` with
everything joined measures 18.1 ms against 19.9. Turning the kick off also dissolves the
4-layer stem kick into per-layer calls, and adds back its `plan+gen` and its submits. An A/B
that changes two things measures neither. The term-by-term profile is what separates them.

The 46 refusals with host-built concatenations are 46 of the 80 index-adjacent pairs (34 join).
The network test's refusal count names three reasons [HW sweep, H96 MAX M9, 2026-08-02]. It
counts 27 branch outputs that must materialize, and 18 pairs with the concatenation as producer
or consumer. The third reason is one channel-alignment refusal, the 528-channel max pool whose
consumer has a non-zero weight zero point.

The three older graphs measured 4.9, 6.7 and 9.2 ms (MobileNetV1, MobileNetV2, ResNet-18), one
submit each, before and after the placement [HW sweep, H96 MAX M9, `bench 100`, 2026-08-02].
ResNet-18 ran with its stem on the direct lowering.

## Inception V3 and gate symmetry

The test is `tests/rk3576_net_gate.c --net iv3`, and `tests/data/rk3576-net/mknet.py` builds
the blob.

Wall-clock time after each change, in order [HW sweep, three runs a configuration, interleaved]:

| configuration | wall |
|---|---:|
| first run | 108.9 ms |
| the four wide pools on the NPU instead of the host | 47.8 ms |
| the divisor lag checked | 40.7-41.9 ms |
| the fifteenth concatenation placed | 38.4-38.6 ms |
| the divisor-lag check scoring the last channel group alone, `bench 100` | 33.4-33.7 ms |

The 34.4-34.6 ms check-off reading (2026-08-03) predates the fifteenth concatenation's placement,
so it does not bound the 33.4-33.7 ms row. On 2026-08-06 the check and its redos measured 1.5 ms
of 34.0. The check costs time on every call, while the redos are a
couple an inference and are the smaller half.

The 33.4-33.7 ms figure is three interleaved `bench 100` runs at NPU 786 MHz, 2026-08-03, with
the divisor-lag check scoring the last channel group. The divisor-lag cost measurement of
2026-08-06 ("Exposure and cost in the model corpus") measured the same graph at 34.0 ms.

## The join ratio's denominator

Counting index-adjacent pairs understates the ratio, and sending them through the refusal
buckets lets a bucket name a producer whose index-successor it never feeds. The table is
counted by `rk3576_net_gate bench 100` [HW sweep, H96 MAX M9, 2026-08-03]. Joins plus refusals
equal the denominator exactly on all five graphs, which confirms that the set is the right
one.

The refusal table counts the 768×17×17 concatenation's two refusals at 298 KiB, ~0.7 ms at
this graph's own ~2.4 us/KiB for a join. Closing them gains 2.6-3.1 ms, about four times
that, measured against `ROCKET_RK3576_NET_NOJOIN=40` over three interleaved repeats a
configuration:

| configuration | wall, three repeats | submits | chained runs | layers inside a chained run |
|---|---|---:|---:|---:|
| joined | 38.4, 38.6, 38.6 ms | 47.6-48.1 | 16 | 94 |
| `ROCKET_RK3576_NET_NOJOIN=40` | 41.2, 41.2, 41.5 ms | 55.6-56.0 | 18 | 88 |

A refusal table counts pairs, and a placement's reach is its readers. The divisor-lag rate is
read with `ROCKET_RK3576_POOL_PROF=1` [HW sweep]. Neither the five new readers nor the pool's
drop is a pair, and neither is in the refusal table.

Without the dataflow-adjacency check, the join pass rejects a non-edge pair only on a shape
mismatch between producer and consumer, which guarantees nothing.

## Symmetric test axes

Inception V3 was chosen for the axes it varies, and what it yields is defects rather than a
feature. Every earlier test point on the RK3576 is symmetric in the axes this graph breaks.
Two library encodings were wrong in ways no symmetric test point could observe. A third
defect, the divisor lag, is observable only in the mode this graph is the first to need.

The symmetry audit is computable from the test tables, before any hardware runs. For each
packed register field and mode bit an emitter writes, it asks which axis separates the field
from a plausible wrong value. Then it asks whether any test point varies that axis.

The audit's first deliberate run is on the PPU's `0x6034`. Every source that had exercised it
held both pairs square:

- Every pooling test point that had run on the RK3576.
- Every vendor capture.
- Every test point of the test suites.
- Every pool of the five networks.

The CNA's kernel word and the PPU's own pad nibbles had that same standing when each was found
to be transposed. The audit drives the test points as pairs at one rectangular plane. A
transpose maps each test point of a pair onto the other, so both being exact is what says the
map is right.

The register map is as labeled, and the audit finds a defect one level up instead. Under the
width-keyed allowance, four of the eight test points are wrong, and not in pairs. Every test
point with `kh > kw` fails and every one with `kh < kw` passes [HW sweep, H96 MAX M9,
`tests/rk3576_pool_probe.c lib`]. A transposed field does not fail that way. With the
allowance keyed on the kernel height and the vertical stride, all eight are bit-exact. The
other 21 test points of `lib` are bit-exact too [HW sweep, H96 MAX M9, `tests/rk3576_pool_probe.c lib`].

The other axes Inception V3 varies were all exact on the first run.

## Requantization edges

The direct 1×1 convolution over an identity matrix is kept as a control
(`MKNET_REQUANT_DIRECT=1`), not as the lowering.

## Inception V3 chained-run fragmentation by reason

A join is a property of a pair, and one layer breaks a chained run, so the join-refusal table cannot
answer why Inception V3 fragments. `kick_build()` reports it per layer instead. The count is
zero on the other four graphs.

The pool row is marked where the candidate is built rather than read off the plan. Read off the
plan, the table reports the layers stranded around each pool and never the pool itself, which
reads as fragmentation the topology caused.

The table is read from `kick_build()`'s report in nine processes of `bench 100`, on
rocket-userspace as of 2026-09-30 [HW sweep, H96 MAX M9, 2026-10-12]. They ran with `NET_RESIDENT=1
NET_CUBE=1` and the governor pinned. Every process reports the same 27 layers and reasons.

### Revision history

On 2026-08-03, at 64 submits an inference, 40 layers sat outside any chained run [HW sweep, H96 MAX
M9]:

| count | why |
|---:|---|
| 9 | it feeds a skip that reads a row-major tensor, so it must materialize one |
| 9 | it is a padded average pool whose divisor can lag |
| 5 | it leaves no cube, so the layer after it would need a host de-tile |
| 5 | it reads no cube any earlier member of the chained run wrote |
| 12 | it is joined at both ends but is the only program between two breaks |

Two totals followed it the same day. Once the column-split pools read a cube, 32 layers sat
outside a chained run, at 55 submits. Once the fifteenth concatenation was placed, 27 did, at 47.6-48.1
submits. Neither total was recorded by reason.

## What the wall is made of, once the graph work is collected

The section measures the graphs with residency, the cube layout and the cross-layer kick in
place. The profile under `ROCKET_RK3576_INT8_PROF=1` covers the chain as well as the per-call
path, so a kick is not a single opaque number. Every graph is one chain call, so the profile
covers the whole wall rather than a sample of it. The accounting is therefore exact rather than
approximate.

### Host share

The reading "The submit is 1-16% of a layer against a host share of 84-99%" was measured on a
40-submit graph. That graph transposed at every layer boundary and repacked its weights every
inference. Residency, the cube layout and the kick remove all three of those terms.

The figure "the host operand packing is 6-12% of the wall" was read from the table as it stood
before two changes. Those are the change to one bracket per BO and the packed-image stems. There
the feature
scatter was 0.60 of 5.0 ms (MobileNetV1), 0.59 of 6.7 (MobileNetV2) and 0.62 of 10.6
(ResNet-18 in 11 submits) [HW sweep, H96 MAX M9, 2026-08-02]. In the current table the packed
stems take ResNet-18 and Inception V1 to 3% and 1%.

### Revision history

- The section's sentence on Inception V3 read "18 kicks and 21 per-layer convolution calls,
  beside 16 pooling ones". That was the graph of 2026-08-03 at 55 submits an inference, before
  its fifteenth concatenation was placed [HW sweep, H96 MAX M9, 2026-08-03]. The placement took it
  to 16 chained runs and 47.6-48.1 submits (§"The join ratio's denominator"). The split into
  convolution and pooling calls was not recorded after it.

### The write guard

A `ROCKET_RK3576_I32_SENTINEL=0` A/B confirms independently that the write guard is `verify` +
`stamp`:

| | `verify` + `stamp` | `I32_SENTINEL=0` A/B |
|---|---:|---:|
| MobileNetV1 | 2.07 ms | 1.9 ms |
| MobileNetV2 | 3.19 | 3.0 |
| ResNet-18 | 2.10 | 2.1 |
| Inception V1 | 2.83 | 2.8 |

On MobileNetV2 the guard is the first term, at 3.0 ms of 6.7 against the submit's 3.0.

The reduced guard, `ROCKET_RK3576_GUARD_PER_KICK=1`, catches the whole-submit poisoning: 0 of
144 poisoned inferences are wrong under either form.

With ResNet-18 in eleven submits, the flag collected the whole guard on both MobileNets, each
already one kick. On ResNet-18 it collected only 65%, 10.6 -> 8.9 against a guard-off floor
of 8.0. Seven of ResNet-18's submits sat outside any kick, and each paid its own per-call bracket.
The 65% on ResNet-18 is a measurement of the eleven-submit graph, and it does not describe the
one-kick graph.

With every graph in one kick the flag has no shortfall. All four land on the guard-off floor
[HW sweep, H96 MAX M9, `bench 100`, 2026-08-02]:

| | default | `GUARD_PER_KICK=1` | `I32_SENTINEL=0` floor | guard, of the default |
|---|---:|---:|---:|---:|
| MobileNetV1 | 4.8 | **3.0** | 3.0 | 1.8 ms of 4.8 (38%) |
| MobileNetV2 | 6.8 | **3.5** | 3.5 | 3.3 of 6.8 (49%) |
| ResNet-18 | 9.2 | **6.5** | 6.4 | 2.8 of 9.2 (30%) |
| Inception V1 | 10.8 | **6.2** | 6.2 | 4.6 of 10.8 (43%) |

On the graphs of that table the guard is the largest single term on three of the four.

#### Revision history

| Quantity | Earlier figure | Current figure in the sheet |
|---|---|---|
| Stamp per inference: MobileNetV1, MobileNetV2, ResNet-18, Inception V1 | 0.01, 0.39, 1.16 and 2.07 ms, with the write guard bracketed per layer, both 7×7 stems on the direct lowering, at 4.9, 6.8, 9.3 and 10.8 ms and 40, 77, 38 and 77 tasks [HW sweep, H96 MAX M9, `bench 100`, 2026-08-02] | 0.03, 0.26, 0.78 and 1.12 ms, the stamp column of the first table, with one bracket per BO and the packed-image stems [HW sweep, H96 MAX M9, `bench 100`, 2026-08-03] |

The 3.26 ms the placement returned to the guard on Inception V1 is from the 2026-08-02
comparison, with the guard bracketed per layer.

### The submit column

The marginal cost of a submit is priced from the A/Bs that add or remove one.

### The submit bucket's composition

Two instruments split the bucket. The chain's own profile splits its interval into the submit
ioctl, the fence wait and the `FINI_BO`. The second instrument is
`rocket-userspace/tools/rk3576-job-timing-debug.patch`, a debug module rather than a series
patch. It adds five microsecond accumulators to the driver's polled completion path, exposed as
writable module parameters. The table runs on the 0023 module plus the instrument. The driver's
terms are `bench(200) - bench(100)` over 100, so the gate's fixed setup jobs cancel.

The accounting closes to under 1%. The terms sum to the measured bucket, and the driver's
independent `push -> fence signal` total reproduces the caller's own fence wait to within tens
of microseconds:

| | terms sum | measured bucket | driver `push -> fence signal` | caller's fence wait |
|---|---:|---:|---:|---:|
| MobileNetV1 | 2.34 ms | 2.34 ms | 2.21 ms | 2.28 ms |
| MobileNetV2 | 2.86 | 2.86 | 2.73 | 2.72 |
| ResNet-18 | 5.84 | 5.84 | 5.78 | 5.78 |
| Inception V1 | 5.59 | 5.56 | 5.48 | 5.45 |

So this is a decomposition rather than a residual: nothing is attributed to what was not
measured.

The driver's `drain0` case is a job whose completion bit was already up on the tick that first
saw the kick over. It fires on every job of all four graphs.

The 61-81% is each graph's submit bucket (2.34, 2.86, 5.84 and 5.56 ms) against its wall-clock
time with the guard narrowed on the same day. Those times are 3.6, 4.7, 7.2 and 7.7 ms, giving
65%, 61%, 81% and 72% [host-computed]. The narrowed guard was 0.59, 1.09, 0.69 and 1.47 ms of
those times, 16%, 23%, 10% and 19% [HW sweep, H96 MAX M9, `bench 100`, 2026-08-02]. On
MobileNetV2 it is therefore above 20% too. The submit bucket is the largest term on all four.

### The stem's share of the execution

The per-layer configuration (`ROCKET_RK3576_NET_KICK=0`) shows which program the execution is
spent on. Its layers sum with the stem to the kick configuration's total (30 x ~0.12 + 2.05 =
5.65 against 5.75 measured), so the accounting closes here too.

With ResNet-18 in eleven submits, the stem was the largest single layer of any of the three, at
29% of ResNet-18. It took 3.08 ms: 2.13 submit and wait, 0.62 the image scatter, and 0.34
plan+gen on a resident handle.

### The execution cost model

The table runs `tests/rk3576_stem_cost.c`, with resident weights, the driver's `stat_exec_us`
read through the job-timing debug module, and `bench(2N) - bench(N)`. A convolution's execution
has two candidate terms, and a stem is where they are hardest to tell apart. Crossing the two
axes separates them, where sweeping either alone cannot. In group A, a pure feature-DMA bound
predicts a flat column, and a pure MAC bound predicts 49x.

### The packed encoding's bounds on the four graphs

The table reads each stem out of the four `.rnet` blobs. Inception V1's earlier block was the
library's and is closed. The two MobileNet refusals are the library's own messages, read out of a
graph run rather than computed here. The gate `rk3576_net_gate` attempts the packed pack for
every `ic <= 4` convolution and prints what came back, so the count is a measurement.

#### Revision history

| Quantity | Earlier figure | Current figure in the sheet |
|---|---|---|
| ResNet-18's stem on the packed encoding | Refused: "the zero point, on the pad columns", the library's own message from a graph run, and the encoding on one graph of four [HW sweep, H96 MAX M9, 2026-08-02] | Runs with its pad columns materialized, the max pool reading the wider output in place: 7.4 ms against the direct stem's 9.2-9.3, 33 tasks against 38 [HW sweep, H96 MAX M9, 2026-08-02]. The encoding runs on two graphs of four |

### The asymmetric output extent on the packed encoding

The extent is measured with `tests/rk3576_argb_extent.c`. The probe scores by pad region. It
classifies every output position by which pads its window reaches: interior, leading, trailing
or both. That classification separates the two ways the answer could have been no. A derived
pad the sub-encoding does not read shows as an exact interior against a wrong trailing edge. A
mis-programmed geometry is wrong everywhere.

Each test point has two controls: the direct path at the same extent, and the packed path at the
symmetric lead. The direct path is the configuration that must succeed, since it is what makes
the reference trustworthy.

The refusal of an asymmetric extent on this path was the library's.

### A packed program at the head of a chained stream

The packed program chains in 20 of 20 test points over five geometries, with both boundaries
scored [HW sweep, `tests/rk3576_chain_argb.c`]. The consumer's stale form is distinguishable
throughout. The join's cost is the per-join sweep's figure.

### Residency and the vectorized image pack

The residency table's rows are `rk3576_stem_cost` E and F. Neither residency nor the vectorized
pack is encoding work. Both were worth more than the register question. The scalar and
vectorized walls are measured under the same run conditions.

The direct configuration's wall read 3.82-3.86 ms before the interleave landed and 4.53-4.61
after, with nothing about the direct configuration changed. The packed configuration ahead of it
in the same process got 1.2 ms faster. The governor had therefore ramped less by the time the
direct configuration ran. The same configuration reads 4.6-4.8 run alone through the name
filter, and 3.86 as the last row of a full sweep.

### The packed stem in the Inception V1 graph

Wiring the packed stem into `rk3576_net_gate` takes one descriptor field and leaves the rest of
the chained-run structure untouched. Each configuration ran three times, under the `ondemand` governor
[HW sweep, H96 MAX M9, `bench 100`]:

| Inception V1, `bench 100` | direct stem | packed stem | saves |
|---|---:|---:|---:|
| default guard | 10.7-10.8 ms | **8.7** | 2.0-2.1 (19%) |
| `GUARD_NARROW=1` | 7.7 | **5.6** | 2.1 (27%) |
| `GUARD_PER_KICK=1` | 6.2 | **4.1** | 2.1 (34%) |

The other three graphs do not move, at 4.8-4.9 / 6.7 / 9.2-9.3. Each prints the library's own
refusal for its stem, which is what makes "one graph of four" a measurement. Measured
2026-08-02, before the pad columns were materialized.

The term table's ratios come from the profiled run (`ROCKET_RK3576_INT8_PROF=1`) and its walls
from an unprofiled one.

Every layer of the graph scores `hw exact` against the CPU model. The per-op probes all ran at
`w_zp = 0`, so the graph is where that axis first varies.

The packed stem is default-on in the gate. Its control is `ROCKET_RK3576_NET_ARGB=0`, which is
itself a gate: the two lowerings must agree byte for byte and return the same label.

## Register encoding: Captures and the regcmd encoder

This backs the opening of [rk3576-regcmd.md](rk3576-regcmd.md), before its first section.

### Capture sets

The register formulas come from RKNN-Toolkit2 register programs for known conv geometries, from
two capture sets under
[rocket-userspace `tests/data/rk3576-vendor-capture/`](https://github.com/gregordinary/rocket-userspace/blob/main/tests/data/rk3576-vendor-capture/).

The found captures come from Ga Hing Woo's bring-up repo. They are real convolutions, including
a MobileNet-shaped stem:

| Capture | Geometry | What only it can show |
|---|---|---|
| `conv2d_rk3576.rknn` | ic 16, oc 128, k5 s2, 80×80 -> 40×40 | the normal `in_ch>4` path at k != 3, and a single task over the whole plane |
| `dw_rk3576.rknn` | depthwise c32, k3 s1, 112×112 | the depthwise words, and a task windowed to 91 input rows |
| `conv64_rk3576.rknn` / `conv0_rk3576.rknn` | ic 3, oc 32, k3 s2 | the first-conv ARGB sub-encoding, at two image sizes |
| `iso_bias` / `iso_scale` / `iso_sum` | the `conv2d` geometry plus one appended op | what a DPU epilogue stage moves |

The manufactured captures are in `dw/` and `dw/named/`. The scripts `dw/mkdw.py` and
`dw/named/mknamed.py` rebuild both sets and contain the dependency pins.

The found depthwise captures confound almost everything. In them, every C=32 program is
stride 1 and every C=64 one is stride 2. Every one is k=3, every one is a multiple of 32
channels, and every one is a square plane. The manufactured captures separate all of it, at 22
channel counts from 8 to 256, both strides, kernels 1/3/5/7 and rectangular planes. The
sheet's "Depthwise convolution" section transcribes four register formulas from them.

### Register fidelity

[regcmd_rk3576_gate.c](https://github.com/gregordinary/rocket-userspace/blob/main/tests/regcmd_rk3576_gate.c)
diffs the regcmd encoder against the captures register for register. The regcmd encoder reproduces every
checked register of every captured program, 110 conv programs, 88 of them depthwise, with no
open field left.

Register fidelity is not correctness, and the depthwise path is where the two diverge. A regcmd
encoder matching the vendor register for register across 88 depthwise programs still produced
no correct output. What it had wrong were the buffers the registers point at, which the
capture does not contain.

### Correctness test

[rk3576_conv_gate.c](https://github.com/gregordinary/rocket-userspace/blob/main/tests/rk3576_conv_gate.c)
sweeps a shape table in one process. It compares each entry against a CPU model bit-exactly
over the whole output. The table covers the supported range, the row window, the
channel-group jump and the weight-slice boundary, with the depthwise range beside them. The
test runs with no gaps on a `rocket` that has the per-SoC `PC_TASK_CON` width. It reports every
cold-start retry it takes, so a run that prints none had no cold-start failure. All 102 shapes
pass in 4.6 s.

## Register encoding: Corrections to the published map

### Output height

In `conv2d`, CORE `0x301c` hi is `0x27` = 39 for a 40-row output. The ARGB `conv0` full-plane
task writes `0x6f` = 111 for 112 rows, and only its half-height slices write 55. The published
halving comes from a capture whose task covers half its plane. The same conv captured as a
single task writes the full `oh-1`.

### Feature-data address

Register `0x1070` reads zero in every capture, including the one program that provably needs a
non-zero feature address. The `conv0` capture's fourth task reads input rows 111.. of a
224-wide 3-channel image, and `0x1088` contains exactly `111 * 224 * 3 = 0x12360`. Register
`0x1070` is zero in that same program.

### Pad word

The formula reproduces all four captures exactly, including the two that make it unambiguous.
One is `conv2d`'s asymmetric SAME padding (l1 r2 t1 b2 -> `0x02020101`). The other is the
depthwise task whose row window stops short of the image bottom. Its bottom pad is 0 while its
right pad is 1 (`0x01000101`). The published rule, a lookup keyed on stride and depthwise,
predicts `0x00000101` for `conv2d`.

Six shapes of the int8 correctness test's table have a zero leading pad against a consumed
trailing one, direct and depthwise. All six are bit-exact [HW sweep, H96 MAX M9]. The float
path has never been run through that geometry.

### CBUF entries

The two forms agree on every capture the published map was fitted to, which is why the map
gives the second one. At `iw`=112 and `ic`=32, the granule count gives 56 and the max form
gives 28. The capture holds 56 at both `0x103c` hi and `0x1044` lo. The distinction is
invisible at the shapes the map was derived from.

## Register encoding: Unnamed registers and their RK3588 equivalents

The identification is by value, not by analogy. The int8 conv's CVT word is `0x0b` in both
encodings, and the burst word is `0x000F000F` in both, for the same reason.

The `conv2d` capture is the only one that shows a second k for the kernel word. Every capture
is square, so the captures cannot tell the two halves apart. Inception V3's non-square kernels
can. Its 34 layers at 1×7, 7×1, 1×3 and 3×1 are bit-exact with the width in the high byte. With
the halves swapped they are wrong in every element region
[HW sweep, H96 MAX M9, `tests/rk3576_net_gate.c --net iv3`].

The OUT_CVT reading comes from position: a small signed value ahead of a scale and a shift. A
register sweep on hardware confirms it. Forcing the offset to 16 adds exactly 16 to every
output channel, and forcing the scale to 0 zeroes the output. Raising the shift attenuates a
saturated result into range. [HW sweep, H96 MAX M9]

## Register encoding: int8 convolution

The k=3 SAME 32×32 int8 conv reproduces the CPU model bit-exactly across the whole output,
32768/32768, max |diff| 0. The same conv unpadded is 28800/28800.

The same result confirms the weight cube independently. A dense random set of 9216 weights
cannot match bit-exactly under a permuted layout.

The geometry sweep is bit-exact on every run, repeatably across separate power sessions.

## Register encoding: Channel counts as multiples of 32

The regcmd encoder matches the vendor's `conv2d` program 131/131.

## Register encoding: The CBUF budget and `0x1040`

### Graded corruption

At ic=32, iw=16: 4096 granules is exact, 4104 gives 132001/262656, and 4160 gives 1090/266240.
The graded corruption is the instrument for the budget: grow a known-good conv until it
breaks.

### Single-bit F values and conditional delivery

On the depthwise path, 4 granules is dead where 32 is live, which refutes a footprint threshold
in both directions. What survives the eight test points is the tap count.

#### Excluded causes

- *A 64-channel-group footprint.* A footprint taken over a 64-channel group
  (`R76_DW_W_GROUP_INT8`), against the direct path's 16-granule threshold, fits k1, k3 and
  k5. The 2×2 test point refutes it, which is why that test point is in the table.

[HW sweep, H96 MAX M9, `tests/rk3576_conv_lib_gate.c rowmap`]

`librocketnpu` declines F=256 and F=512 on the depthwise path rather than gating them on a
fitted quantity.

### The direct path's weight-footprint condition

A characterization that holds `ic` at 32 and moves the kernel uses one granule total of 4352
and holds it constant across five plane widths. It reads k=1 exact everywhere and k=3 and k=5
wrong everywhere:

| plane | entries/row | rows | k=1 | k=3 | k=5 |
|---|---|---|---|---|---|
| 16×544 | 8 | 544 | exact | wrong | n/a |
| 32×272 | 16 | 272 | exact | wrong | n/a |
| 64×136 | 32 | 136 | exact | wrong | wrong |
| 128×68 | 64 | 68 | exact | wrong | n/a |
| 272×32 | 136 | 32 | exact | wrong | n/a |

#### Excluded causes

- *The kernel height.* The table above suggests the condition is `kh == 1`. Crossing the axes
  refutes that. At the same 4352-granule total and the same 1×1 kernel, the same (row size,
  row count) is bit-exact at `ic` 32 and wrong at `ic` 64 and 128. Both wrong test points are
  68 rows of 64 granules, and both are wrong from output row 64. The F=0 control at 64 rows is
  exact, and so is the same plane forced under the boundary.

| 4352 granules, 1×1 kernel | entries/row | rows | result |
|---|---|---|---|
| iw 16, ic 32 | 8 | 544 | exact |
| iw 32, ic 32 | 16 | 272 | exact |
| iw 64, ic 32 | 32 | 136 | exact |
| iw 128, ic 32 | 64 | 68 | exact |
| iw 64, **ic 64** | 64 | 68 | **wrong from row 64** |
| iw 32, **ic 128** | 64 | 68 | **wrong from row 64** |

`ic` at a fixed kernel and the kernel at a fixed `ic` therefore move the same quantity,
`32*ic*kh*kw`, the resident weight slice. That is the quantity `r76_weight_slice_cap()` is
already stated over.

What F=256 and F=512 give backs out to the row from the surviving prefix. That makes 4096 a
measurement rather than a reading of a graded corruption.

Every test point above that reaches F=256 or F=512 has `oc` 32, one group, where the slice and
the cube are the same number. [HW sweep, H96 MAX M9, `rk3576_conv_lib_gate rowmap`]

Holding the slice at the measured 16 granules and raising the group count separates the two.
[HW sweep, H96 MAX M9, `rk3576_conv_lib_gate rowmap`]

The threshold between 16 and 32 granules is bracketed, not measured, and the harness cannot
narrow it.

The correctness test does not reach this condition on its own, and a whole network does. Every
direct shape in its table sits at F=0. So does every depthwise one, except a MobileNet's 3×3
over a 112-row plane. Nothing reaches the ic axis either.

The packed-image path has never been driven at F=256 or F=512.

[HW sweep, H96 MAX M9, `tests/rk3576_conv_sym.c rung` and `rk3576_conv_lib_gate rowmap`, with
`tests/rk3576_conv_lib_gate.c` groups `rung256` and `dwbig` for the original kernel reading]

### Per-task planning of F

Planning reproduces both captured words, so the register-fidelity test stays byte-identical.
Per-task F is validated on the SoC at 4160, 4800, 5600 and 6144 granules, and at 6144 across
three widths. `ROCKET_RK3576_CBUF_F` is how the allowance was characterized.

## Register encoding: The weight slice and the output-channel group count

Sweeping `ic` at k=1 moves the slice in 1 KiB steps, which is what resolves the boundary at
all. At k=3 the step is 9 KiB and at k=5 it is 25. That coarse sampling makes the loss look
like a jump from four groups straight to two.

The points do not isolate a mechanism. Two free parameters against six points is not a
mechanism, so the regcmd encoder uses the measured table rather than a formula. A graded
multi-group loss is by definition not a single-group effect, so a single group is left to the
pool check. [HW sweep, H96 MAX M9, measured 2026-07-25]

### Output-channel tiling under the group cap

The test `tests/rk3576_conv_lib_gate.c` records the tiling result. It drives the register-fidelity
test's own shape table through the library entries. It passes 111 shapes and asserts the one
refusal in both directions. Every computed shape is bit-exact against a CPU model
[HW sweep, H96 MAX M9, measured 2026-07-27].

Output-channel tiling lifts fourteen of the regcmd encoder's refusals. The two that reach furthest
are `ic=576 k=3` and `ic=5184 k=1`. The same sweep confirms that the cube size is not the
constraint.

## Register encoding: The DPU epilogue and the operand DMAs

The bias-only measurement (features and weights zero, a per-channel bias) reproduces the CPU
model across all 32 output channels: 2048/2048 bytes, max |diff| 0.

The output and bias bases are the positive controls of the DMA test, because they are known to
be read and written. A "no fault" reading elsewhere would therefore have been evidence, and a
fault everywhere confirms that no operand loader is silently idle.

A uniform feature fill proves nothing about addressing, because every read that falls inside
the buffer returns the same byte. The stride sweep therefore varies each axis with the other
two held flat.

### The coefficient buffer

The BS order is read off the SoC against a known accumulator and a known bias. `C` is set to 1
on the even channels and 2 on the odd. There, `(acc + A)*C` explains 32 of 32 channels, and
`acc*C + A` explains only the 16 where `C` is 1 [HW sweep, H96, `tests/rk3576_coeff_c.c`].

The saturation sweep holds the shift word at zero and walks `(acc + A)*C` across `2^31` at a
fixed accumulator. Every inexact test point implies the same ceiling, 2.147e9 to 2.158e9
against `2^31` = 2.1475e9, and none of them wraps [HW sweep, H96].

At a shift of 14 and `C = 16384`, a saturate-first product reads 131071
[HW sweep, H96 MAX M9, `tests/rk3576_coeff_c.c shift`, 2026-09-23].



### The B term is added: `acc + B*sum(x)`

Every vendor capture has `B = 0`, so a capture shows the field's position and width but
not its sign convention. Driving `B` on the direct path against a CPU model determines it. The
model `acc + B*sum(x)` explains every output of every direct shape in the conv gate's envelope
group [HW sweep, H96 MAX M9, measured 2026-07-27].

The models `-B*sum(x)` and an inert `B` explain only the few percent that coincide. That share
is larger at small `|B|`, because the correction often rounds to the same int8. The variable
`ROCKET_G_WZP` in `tests/rk3576_conv_gate.c` drives it and scores all three models.
[HW sweep, H96 MAX M9, measured 2026-07-27]

### Revision history

- The `B` term was recorded as unvalidated against a non-zero weight zero point on 2026-07-25.
  The measurement of 2026-07-27 under "The B term is added: `acc + B*sum(x)`" supersedes it.
  `acc + B*sum(x)` explains every output of every direct shape at `B` = 1, 40, 127, -37 and
  -128 [HW sweep, H96 MAX M9].

### The OUT_CVT tie rule

The half-up model `(acc*scale + (1<<(shift-1))) >> shift` reproduces every pixel of the bias
and OUT_CVT measurements of this section [HW sweep, H96]. Those measurements contain no exact
tie, so they cannot separate half up from half to even. The tie rule is from
`tests/requant_round_probe.c`, 160 tie elements per setting at two shifts and both signs,
through `rocket_matmul_int8_rk3576` [HW sweep, H96 MAX M9, 2026-09-23].

## Register encoding: The feature domain and the padded border

The full-range feature sweep is one flat run from -128 to 127, 8192/8192, max |diff| 0, with
raw signed bytes fed.

### Excluded causes

- *A uint8-centered datapath.* A run in which negative features fail, as though the datapath
  were uint8-centered, was made against an all-zero coefficient buffer. With that buffer the
  output is empty whatever the features are.

The padded border reproduces the CPU model over the whole output, ring included, 32768/32768,
max |diff| 0. So does the pad measurement, which drives the border with a -128 pad tap against
zero features.

## Register encoding: The row window and its three registers

### Window addressing

The vendor's sliced capture is the direct evidence for the row offset on the feature side. Its
fourth task reads input rows 111.. and `0x1088` contains exactly that row offset.

A padded `0x1094` is bit-exact at `+3`, `+16` and `+64` elements over five geometries, with
one row task and with ten. A control that lays the same padded buffer out without setting the
register differs every time [HW sweep, H96 MAX M9, `tests/rk3576_surf_stride.c`]. The padded
cases are bit-exact with `0x1098` left at `round4(iw*fetch_rows)` [HW sweep, H96 MAX M9,
`tests/rk3576_surf_stride.c`].

### The windowed registers

One sweep measures all three registers. It cuts an `oc=64` conv into 32-, 16- and 8-row
windows of the same 64-row plane. The conv's two output-channel groups expose group 1
[HW sweep, H96 MAX M9].

The `0x40B8` form is swept, not fitted, and it holds for a conv driving many groups at a wide
kernel. The coverage stated in the sheet is the `surface`
group of `rk3576_conv_gate`. The sheet's description of the jump is what that group confirms
[HW sweep, H96 MAX M9, measured 2026-07-25].

### Excluded forms of the channel-group jump

Only `ow * (2*oh_full - oh_task)` is bit-exact at all three window sizes. The forms
`ow*oh_full`, `ow*oh_task`, `2*ow*oh_full` and 0 each corrupt the whole output
[HW sweep, H96 MAX M9].

### Cost per window

The cost sweep cuts the same 64×64 ic64 oc64 k3 SAME output into progressively more windows.
It submits them with no inter-task gap, with the cold-start defect fixed [HW sweep, H96 MAX M9,
measured 2026-07-25]. The per-task figures in that section were measured without 0012, at
~1.2 ms.

### Revision history

- 2026-07-28: with `0012` the fixed per-task cost was stated as ~0.44 ms. The per-submit time
  re-measured on 2026-07-31 is 329-348 us at a 50 us period (below).

## Register encoding: The group stride on a single-task plan

The un-windowed path in `rk3576_first_light` sets the window directly and never trips the
defect. It appears only once a caller sends every conv through the planner
[HW sweep, H96 MAX M9].

## Register encoding: The geometry fields are narrower than their register halves

The widths are measured through the public entries, with every element scored
[HW sweep, H96 MAX M9, 2026-09-26, `tests/rk3576_conv_width_probe`]. Every test configuration
had the input and output widths equal, so the probe could not say which width register holds
the 13-bit field. Strides 8 and 16 program 0 and did not run.

After the 8300-wide task, the 8192-wide job (259665 of 262144 elements wrong) and the
2048-wide one (64961 of 65536 wrong) both returned rc 0. A third job retired, and the entry's
guard redid it exactly.

### The fields the vendor compiler checks

One-op graphs sized past a field along one axis at a time make RKNN-Toolkit2 name fields
[`rocket-userspace/tests/data/rk3576-vendor-capture/regtask/mkregtask.py`, toolkits
2.2.0 and 2.3.0, 2026-09-27]. On the RK3588 every field it printed matches Mesa's `registers.xml`:
DPU_RDMA `0x500C` [12:0], DPU `0x4038` [12:0] and DPU `0x403C` [12:0] and [28:16]
[source-confirmed]. The method therefore reads the compiler's widths correctly.

The line-stride result past the printed 14 bits is from
`tests/rk3576_conv_width_probe r4` [HW sweep, H96 MAX M9, 2026-09-27].

## Register encoding: The first-conv ARGB sub-encoding

Twelve captured programs pin the sub-encoding, at two image sizes (224×224 and 64×64) and two
row splits.

### The ARGB weight cube

The found captures contain register programs, not weight BOs, so they pin only the cube's size
and stride. The float cube's layout is decoded from captures manufactured with weights unique
per position (`tests/data/rk3576-vendor-capture/argb/mkargb.py`)
[source-confirmed, RKNN-Toolkit2 rk3576 float build].

A quantized ARGB capture does take the path: `0x100C` reads `0x2000a006`, exactly the found
value. But its weights do not survive quantization as a findable value set. The probe
therefore reads the int8 cube off the SoC instead, with an impulse image and a one-byte cube
(`rk3576_conv_gate fcmap`). Every live byte then names its output channel, both taps and its
lane at once.

The map is a bijection over every live byte at `oc` 32 and 64, `k` 3/5/7 and `ic` 3 and 4.
That covers 1152, 2304, 3200, 6272 and 864 live bytes. Each maps to exactly one output
position of one channel, with none left over and none doubled. The map is also
translation-invariant: the impulse moved three and six pixels reads the same.
[HW sweep, H96 MAX M9]

The 32-channel group is observable only above one group. At `oc=32` a flat `oc*R` fits
equally, and `oc=64` is what separates them. The `k=5` and `k=7` cubes separate `R` from a
constant 16, because `R` is 32 there. [HW sweep, H96 MAX M9]

The fp16 program's declared weight size is read from the manufactured float capture
`f16argb_c3_16_k3_s1_32x16_o32x16`. The capture holds `0x101C` = `0x480` (1152 bytes) and
`0x1020` = `0x48` (72 bytes an output channel) [source-confirmed, RKNN-Toolkit2 rk3576 float
build].

### Geometry bounds of the int8 first conv

The test `tests/rk3576_argb_ic1.c` holds the `ic=1` result, with the register sweep that
found it [HW sweep, H96 MAX M9, measured 2026-07-28].

The twenty-shape library result is the `fq` group of `rk3576_conv_lib_gate`, which also
asserts all seven refusals in both directions [HW sweep, H96 MAX M9].

A capture is an oracle for a whole program, and running one verbatim separates a dead mode
from a wrong extrapolation. `librocketnpu`'s regcmd encoder reproduced twelve int8 ARGB captures
register for register and still wrote nothing at every geometry tried, which reads as a dead
datapath. The test `tests/rk3576_fc_vendor.c` takes the captured op stream out of the golden
table, patches only the five address registers and submits it. The vendor's own program then
writes its whole output. The mode works, and the extrapolation away from the captured
geometry is what fails. Every bound comes from walking from that geometry back toward a small
one, one axis at a time.

### The fp16 first-conv program

Manufactured captures separate every one of the seven fields, each varying one axis against the
rest [source-confirmed, RKNN-Toolkit2 rk3576 float build].

### The vendor's float BS arrangement

The `0x501C` and `0x4044` results are a leave-one-out over the six float-path registers
[HW sweep, H96 MAX M9, leave-one-out over the six float-path registers].

The figure 1866/2048 is the fp16 sweep harness's "within 1%" count
(`tests/rk3576_fp16_sweep.c`), so 182 of 2048 elements are off by more than 1%
[host-computed].

### The vendor compiler's packed-path choice

The two-channel packed path is measured by `rk3576_conv_lib_gate fc-2ch-k3`.

### Revision history

- 2026-07-25, from the found captures, all `kw=3`: the programmed channel count was read as
  `4*kw`. That was an inference from the datapath, since `4*kw` and a constant 12 fit equally,
  as did several readings of the 48-element weight kernel. A first conv at another kernel width
  was then unvalidated. The int8 cube map at `k` 3/5/7 now runs the `4*kw` program bit-exactly
  [HW sweep, H96 MAX M9]. So do the extent sweep at kernels 5 and 7 and the library's 224×224
  stem at `k` 3 and 7 [HW sweep, H96 MAX M9].
- 2026-07-25, from the found captures: the CVT was read as subtracting the uint8 zero point
  held negated in `0x1054`-`0x105C`. The pixels were read as raw uint8 that the converter
  centers. The pad constant was read as programmed in the raw byte domain, because the pad is
  inserted before the CVT. The input-domain sweep of 2026-07-27 supersedes that reading. The
  offsets are inert and the CNA reads each packed byte as a signed int8 [HW sweep, H96 MAX M9].
- 2026-07-26: the int8 weight cube was listed as not decoded. It was read as a different
  object from the float cube, because the depthwise path's int8 and float cubes differ. The cube map of
  2026-07-27 in §"The ARGB weight cube" above supersedes it [HW sweep, H96 MAX M9].
- 2026-07-27: the input-domain Caution read that programming the border constant as the stored
  zero point makes a pad tap's true value zero. The pad-edge readout of 2026-08-02 (the sheet's
  §"Pad columns on the packed path") supersedes it: only the pad rows and corners read the
  constant [HW sweep, H96 MAX M9].

## Register encoding: CBUF row reuse across a windowed sequence

Thirteen captured continuation tasks pin the reuse scheme exactly. With the reuse form emitted,
seven registers that the register-fidelity gate's diff cannot otherwise reach are checked
exactly across thirteen continuation tasks. The tasks span three kernel sizes, both strides,
both channel counts and all three CBUF origins.

## Register encoding: The appended DPU epilogue stage

Three captures are the same conv as `conv2d_rk3576.rknn` with one extra op appended, one each
of a bias add, a scale and an eltwise sum. They make a controlled experiment run by the
vendor's own compiler.

### The BS operand source

The drive runs against a conv with a known non-zero `B`. The harness is `rk3576_conv_gate`
with `ROCKET_G_WZP=40`, which reports how many outputs each candidate sign convention explains
[HW sweep, H96 MAX M9]. Driven alone, `0x501C` gives "task never wrote its rows" over four
retries, no fault and nothing in dmesg.

## Register encoding: Depthwise convolution

### Validated depthwise shapes

The test is `rk3576_conv_gate dw` at 35 shapes, gap-free with no cold-start-wall retries, with
every shape at `maxdiff 0` [HW sweep, H96 MAX M9]. The table's channel counts that are not multiples
of 32 exercise the quantities that diverge there. Its planes whose `ow*oh` is not a multiple
of four do the same.

## Register encoding: Depthwise convolution

### The weight cube

The float cube is decoded, not inferred. Its captures hold a unique weight value per
(channel, tap). `dw/named/mknamed.py --decode` re-derives the layout from the committed
captures without a board.

The naming must be held in float weights. Per-channel weight quantization normalizes each
channel by its own maximum, which erases any naming held in magnitude. The fp16 format
represents integers exactly to 2048, so a unique-integer ramp survives the float path
untouched. The ramp is findable in the file by its exact value set.

The int8 cube is read off the SoC by the probe mode `rk3576_conv_gate dwmap`. That mode drives
an impulse feature against a cube that is zero but for one byte, and reports which output that
byte reaches. Over a C=32 k=3 cube, every one of the 576 bytes reaches exactly one
(channel, tap) [HW sweep, H96 MAX M9].

The two shapes that hide the group size, a single channel group and `k=1`, are the ones that
pass first. At C=64 k=3, `dwmap` predicts 576 of 576 (channel, tap) positions at a group of 64
and 64 of 576 at a group of 32.

### The depthwise probe modes

`rk3576_conv_gate` has five modes. Each has a `direct` control, which runs the same reading on
the path known to be bit-exact. The control is required: a mode that cannot reproduce the
direct map is not measuring what it claims to.

| mode | what it reads |
|---|---|
| `dwmap` | which output each weight byte reaches, one submit per byte |
| `dwbias` | which coefficient slot each channel read |
| `dwcoeff` | the inverse: which output bytes each coefficient position moves |
| `dwout` | the raw output BO, not de-tiled |
| `-l` | the shape table, without running it |

Two properties make these modes decode rather than score, and any successor needs both:

- `dwcoeff` measures its own baseline. A comparison of each position against the packed
  constant is a delta only where the SoC agrees that the output is flat. On a path whose
  output is not flat, every position reads as "changed" and the probe says nothing. The mode
  keeps one unprobed submit as a reference image, which makes a per-position delta meaningful
  whatever the baseline looks like. There is no need to flatten the hardware first.
- `dwcoeff` and `dwout` read raw bytes. Scoring a depthwise output through the direct path's
  de-tile hides exactly the permutation these modes are looking for.

`ROCKET_G_DWOUT_OUTSCALE` divides the requant down, so a lane at the int8 clip names its
magnitude instead of reading as "large". The variables `ROCKET_G_DWOUT_BASE`, `_STEP`,
`_CBASE` and `_CSTEP` drive the `A` and `C` ramps independently. The two ramps separate "this
lane read the wrong bias" from "this lane read its multiplier out of the bias region". With
`A` constant at 5 and `C` at 1, a lane reading a real `C` answers 5. A lane reading `A` as its
own `C` answers 25.

### The depthwise output surface stride

The rounding in `0x401C` is invisible at every plane whose `ow*oh_full` is already a multiple
of four. That covers every capture on both paths, and every hardware sweep the formula was
first fitted against. Rectangular and odd depthwise planes show it.

The direct path's answer is a hardware result rather than a transcription, because no direct
capture has a plane that separates it. Applying the depthwise rounding onto the direct path
costs the `ow*oh_full = 105` shape and nothing else in the 67-shape test. One shape in the
whole tested range is sensitive to it. [source-confirmed, HW sweep, H96 MAX M9]

### Depthwise geometry registers

Captures built to order separate the axes that the found captures confound, and they close
four depthwise formulas. What the found captures cannot separate:

- The found captures are all k=3 and cannot tell `kh*kw*C/8` from `kh*kw*4` for the
  weight-bytes-per-kernel term.
- The found captures cannot attribute the `0x4050` depthwise field, because every C=32
  program in them is stride 1 and every C=64 one is stride 2. Manufactured captures at both
  strides show it flat in the stride and stepping with C.
- A capture set that is all k=3 cannot tell the `0x40B8` multiplier 4 from `kh+1`.
- All 142 non-ARGB programs hold `iw-1` twice in `0x118C`, with no exception. Square
  captures cannot tell the low half from the plane height.
- For the low half of `0x1024`, an independent on-hardware sweep of the field reaches the
  same conclusion from the other direction.

The odd-count `0x4050` word `0x80011011`, written alone into the padded even-count programs,
computes 19 of 21 shapes wrong, most at the backstop. Two are exact, `oc` 112 at stride 2 and
`oc` 48 windowed. The `gahingwoo/mesa-rk3576` rule does not explain them [HW sweep, H96 MAX
M9, rocket 1.6.0, 2026-09-30].

## Register encoding: fp16 direct convolution

Two apparent gaps seen during bring-up are one fact seen from two sides. One is an output
writer that uses four bytes for a two-byte element and reaches only `oc/2` channels. The
other is a feature surface index that caps a task at 8 input channels. Both follow from the
DPU's output element stride of `16/ic` words, which at `ic = 8` gives the "four bytes on a
two-byte element" reading.

### The vendor float program

Every `.rknn` in the vendor-capture set is int8, and a zero precision field is invisible. An
int8 capture therefore cannot check a float field inferred from the RK3588's packing, and one
such inference is wrong for the RK3576 (`proc_precision`, in the sheet's "Precision fields").

The script `tests/data/rk3576-vendor-capture/float/mkfloat.py` rebuilds the float captures and
holds the dependency pins. The pins are particular. Two things make the resulting diff small
enough to read. Seven registers move with `ic`, and fourteen differ from the RK3588-encoding
program at every geometry.

### The float-mode registers

A single-register sweep cannot find the float mode, because the fix is a register set. Each of
the three registers looks inert while either of the others is at its integer value. A
single-register sweep of any one of them reports it dead, and a leave-one-out over the working
set is what separates them. Two of the three are also unreachable by a single-bit sweep from
the integer word.

### The direct-path float weight cube

The int8 weight cube's group of 32 is the harness's `weight_conv_int8`. A bijective decode
pins the feature cube's C2 = 8, not a score.

Each group needs a shape chosen so that it is observable at all, and the output group is the
harder of the two. A sweep run only at `k = 1` reports all four oc-group candidates as
correct. A reading of 8 for the oc group came from such a shape, where the field is
algebraically dead.

The weight-sizing Caution's example of an `ic = 8` slice dates from the 8-channel split of
2026-07-26, before the 16-channel contraction width was found the same day. Every fp16 task
now runs at `ic = 16`. `rocket_rk3576_plan_ic()` refuses an `ic` that is not a multiple of 16,
so an 8-channel convolution is zero-padded to 16. `rocket_rk3576_fp16_slice_weight_bytes()`
sizes the cube by the groups.

### The fp16 output map

The map is measured, not fitted. `rk3576_fp16_sweep map` drives a probe whose every output
lane carries a unique name, and decodes the output. The decode is a bijection onto the whole
output. Every word names the (channel, pixel) the map predicts, with none left over and none
undecodable. The host-side `regcmd_rk3576_gate` checks the bijection in
`rocket_rk3576_fp16_out_index()`, so an off-by-one in it does not wait for a board. The
`ic` table, the map at `ic` 8, 16 and 32, is the clearest reading of the bound.

### The contraction width

Probe 10 reads the multiplicity straight off the SoC. One input channel holds 1.0, every
weight lane is 1.0, and the output is that channel's read count. The variable `ROCKET_FS_TAP`
moves the tapped channel. At `ic = 32`, the pair `TAP=0` and `TAP=16` separates the two
failure modes in one run. Without the float fields they read 2.0 and 0.0, and with them both
read 1.0. With the fields right, the read count is 1.0 at `ic` 16, 32, 64 and 128 and at every
tap.

`gen_conv2d_fp16_rk3576()` refuses any other `ic` rather than warning about it, so a caller
cannot reach a wrong count silently. The sweep's own modes drive the unchecked
`gen_conv2d_rk3576_prec()`.

### Cost of an fp16 conv

The cost table is the third of three runs, with `oc` 64 [HW sweep, H96 MAX M9, measured
2026-07-26]. The int8 path's 1.1-1.3 ms per-submit overhead is from `rk3576_conv_gate
envelope`. At the 8-channel split the same shapes cost 8 and 16 submits. Doubling the
contraction width halves the submit count and the NPU time with it.

The readback share was measured with `tests/rk3576_fp16_split_cost.c` and
`ROCKET_RK3576_FP16_PROF=1` [HW sweep, H96 MAX M9, measured 2026-07-28, `oc` 32, 28×28 k3,
`ic` 16-128]. The table's readback share is a bound on what on-chip accumulation saves.

The grace measurement is five runs each, at `ic` 128 and 8 slices, with the ranges disjoint
[HW sweep]. The same sweep records zero retries for an unwritten next submit at every shape,
so a slice's ~1.4 ms is not that hazard being cleared.

### Coefficient formats in float mode

`A` packed as its fp32 bit pattern for `bias[c] = c+1` and read back at width `5` returns
`3f800000 40000000 40400000 40800000`, exactly 1.0, 2.0, 3.0 and 4.0.

### The harness

`tests/rk3576_fp16_sweep.c` scores every candidate against a CPU model of the same
convolution. The failure mode is a full, correctly sized, *wrong* output, which no amount of
looking at the output reveals. The harness reports an untouched buffer, an all-zero output and
a wrong output as three separate results.

Its probes turn "wrong" into "wrong here". Each is built so that its answer does not depend on
what it is not measuring:

| probe | what it drives | what a correct SoC returns |
|---|---|---|
| 1 | uniform feature and kernel, each value its own knob (`_PIN`, `_PW`) | `ic*k*k * in * w`, and a result that tracks a side's *high byte* names the side being read as int8 |
| 2 | one non-zero element on each side | one non-zero output lane, at a position |
| 3 | the pixel ramp on every input channel | `y*iw+x+1` in every lane, whatever the lane map |
| 5 | uniform feature, kernel *c* weighted *c+1* | `c+1` in every lane, whatever the lane map |
| 6 | input channel *c* carries *c+1*, kernel taps ic=0 | `1`. Any other value names the channels the SoC paired with lane 0 |
| 7 | uniform feature, kernel ramps over ic | `1+...+ic`. Any other value counts the weight lanes it walked |
| 8 | unique naming: every lane carries `ow*oh*c + p + 1` | itself, the output map, decoded rather than guessed |
| 9 | unique feature, output channel *c* taps input channel *c* | the input cube handed back through the datapath |
| 10 | one input channel at 1.0, every weight lane 1.0 | `1`. The value is that channel's read count |

Probes 3 and 5 are uniform by design in the axis they do not name. They read the output map
without first knowing the lane map. Probes 6 and 7 do the reverse.

Probes 8, 9 and 10 exist because that uniformity is also the others' blind spot. Probes 1, 3,
5, 6 and 7 each pass against a datapath that reduces the wrong input channels. A probe set
built only from them reports an exact conv that is not one. Probes 8 and 9 name lanes instead
of scoring them, so the pass is a *bijection*, not a match count. Probe 10 measures the
reduction directly. Run `map` before believing any fp16 layout claim.

| knob | what it does |
|---|---|
| `map` (argv) | drive probe 8 and decode the output, reporting where each lane landed and how many copies |
| `ROCKET_FS_OUT_LAYOUT` | `0` the native output cube, `1` the library's `rocket_rk3576_fp16_out_index()` |
| `ROCKET_FS_TAP` | move the single live weight lane off ic=0. `-1` taps every input channel (probe 3) |

| knob | what it does |
|---|---|
| `int8` (argv) | the control, the known-good path through the same harness |
| `icsplit` (argv) | the library's `ic` split, scored end to end and timed |
| `f32out` (argv) | the fp16 datapath read back as the full 32-bit epilogue word |
| `bitsweep` (argv) | one register walked bit by bit or over `ROCKET_FS_SWEEP_VALS` |
| `ROCKET_FS_BIAS` | `1` integer bias probe, `2` the same value as fp32 bits |
| `ROCKET_FS_CMUL` | the `C` multiplier. `15360` is fp16 1.0 |
| `ROCKET_FS_SET` | whole-register overrides for the `manual` candidate |
| `ROCKET_FS_C2` / `_WOC` / `_WIC` / `_OUT_C2` | the cube geometry, each axis on its own |
| `ROCKET_FS_DUMP` / `_DUMP_OFF` / `_SCAN` | raw words at an offset, and the written extent and every non-zero lane |

Run the `int8` control before believing any fp16 result. It exercises the same scatter, submit
and de-tile on a path the SoC is known to compute. It therefore fails only on a harness
error. It must report `2048/2048`.

Read the raw output, not only the score. Scoring reads the buffer *through* the layout under
test, so it cannot show a layout that disagrees with it.

### Traps

The sheet states as a Caution or a Note each trap that a user of the registers acts on. The
method traps met in bringing up the fp16 datapath are:

- A register sweep must parse hex. The call `atoi("0x100c")` returns 0. A sweep that silently
  patches register 0 reports that every value changed nothing. A correct sweep of an inert
  register reports exactly the same. The mistake survived a full pass over `0x100C` before
  the reading that mattered was found.
- `feature_data()` and `weight_conv_*()` take 1-based channel and kernel indices. A 0-based
  call drives the group remainder negative and misplaces the whole cube. The result is a wrong
  output rather than an absent one, so it reads as an encoding error.
- A bias probe fixes only the channel-to-lane mapping. Every pixel holds the same value per
  channel, so nothing in it constrains the pixel mapping.
- A probe that is uniform in one axis cannot validate the reduction. A uniform feature reports
  how many lanes were read, never which. A weight held constant over `ic` makes the
  contraction invariant under any permutation of the channel axis. A sum over `ic` is
  permutation-invariant however it is driven. A datapath that reduces the wrong input channels
  passes all of them. Measure the multiplicity (probe 10) and decode a unique-naming probe
  (`map`) before calling any float datapath exact.
- A cube-geometry sweep needs a shape where the field is observable. The float weight cube's
  output-channel group is algebraically invisible at `k = 1`. The kernel index sits between
  the group and the `(oc2, ic2)` pair, so at `k = 1` the index collapses to
  `(oc/G)*G + oc%G = oc`. Every candidate then emits the same bytes. A sweep at `k = 1`
  reports the field determined while testing nothing, which is how it fitted the wrong value 8.
  Before believing a geometry knob, check that two candidates produce different buffers.
- Score only what the map covers. Reading an output through the layout under test cannot
  reveal a layout that disagrees with it. A partial match invites fitting a shift to three
  atoms. Decode a unique-naming probe instead: the answer is a bijection or it is not.
- Pace probe loops. Back-to-back fp16 submits return untouched outputs at a rate that
  depends on the gap. At 200 ms and above, every submit writes (2026-07-26, the fp16
  programs, which also trigger the wide-output hazard). Treat one no-write as a measurement to
  repeat, and re-submit rather than record it.
- A vendor capture is cheaper than a sweep, and one can be manufactured. A capture set that is
  all int8 holds no float fields, but a float capture can still exist. RKNN-Toolkit2 emits
  one for the RK3576 after about an hour of dependency pinning, and one diff of it determined
  what bit-level sweeps had not. The same method determined four depthwise register formulas
  and the depthwise weight cube. Before sweeping for anything, ask which ONNX input makes the
  vendor compiler emit it.
- A found capture set is a set of confounds. The vendor compiles real models, so the axes
  that co-vary in real models co-vary in every capture. In the RK3576's captures, every C=32
  depthwise program was stride 1 and every C=64 one stride 2. Every depthwise kernel was 3,
  and every plane was square.

  Three register formulas fitted through those confounds were all wrong: `0x4050`'s channel
  field, the surface rounding in `0x401C`/`0x40B8`, and `0x118C`.
  Register `0x118C` is `iw-1` in both halves, and it reads as the plane height whenever
  `iw == ih`. When a capture set cannot vary an axis, manufacture one that does before fitting
  anything to it.
- A degenerate helper program reads as a spurious test failure in a capture diff.
- Timing an `ic` split at a probe pacing gap measures the gap. The gap knob
  `ROCKET_FS_GAP_MS` exists to make a no-write rare, not to make a number meaningful. Time at
  `0`, and read the retry count that the harness reports alongside.

### DPU grace

On 2026-07-28 the slice's grace-bound cost was read as a class property. In that reading a
wide-output task raises no DPU completion, so `rocket` with patch `0012` retires it on
`PC_DONE` plus the whole grace. An fp16 conv, in the same reading, waits the full grace only
because its tasks raise no DPU completion at all.

Recovering the speedup was then read as needing a way to say which class a task is in. The
grounds were that userspace knows the class when it emits the program, and the driver cannot
see it at `PC_DONE` time. The same reading held that lowering the shared default would trade
the fp16 path's wall-clock time against the correctness of every other path. On `rocket` 1.6.0 the fp16 programs raise the DPU
completion [HW sweep, H96 MAX M9, rocket 1.6.0, `tests/rk3576_drain_probe raw`, 2026-09-27],
which supersedes the class reading.

The ratio of the two medians is 15.89 / 13.29 = 1.196, stated as 1.20x [host-computed]. The
commit that entered the table on 2026-07-28 gave 1.18x, which no recorded figure reproduces.

## Register encoding: DPU shift operand (`0x5024`)

The multiply-before-shift order and the tie rule are from `tests/rk3576_coeff_c.c shift`
[HW sweep, H96 MAX M9, 2026-09-23]. Take `C = 24576`, a shift of 14 and an accumulator of
160000, which is not a multiple of `2^14`. The SoC returns the multiply-then-shift value
rather than `(acc >> 14)*C`.

An exact half rounds to the even side. With `C = 1` and a shift of 1, 32 odd values of A over
both signs round this way. Of five candidate rules, only half to even fits every channel.

## Register encoding: The cold-start wall

The table submits the same job N times and counts how many wrote [HW sweep, H96].

### Excluded causes

- *The SoC arming its weight loader once per power session.* The external report on
  mesa/rocket + Teflon read the behavior this way. The cause is the `PC_TASK_CON` field width,
  in the sheet's section of that name.

## Register encoding: The `PC_TASK_CON` field width

### The width fix

Back to back with no gap, 1 of 8 submits writes without the fix and 8 of 8 with it. A longer
soak writes 32 of 32, byte-identical to each other and bit-exact against the CPU model
[HW sweep, H96 MAX M9, measured 2026-07-25].

### The control-field sweep

Sweeping the control field at each width isolates the cause to `TASK_NUMBER` alone
[HW sweep, H96 MAX M9]:

| word written | `TASK_NUMBER` seen (16-bit field) | wrote |
|---|---|---|
| `0x70001`, `0x60001`, `0x50001`, `0x30001`, `0x20001`, `0x10001`, `0x00001` | 1 | 8 of 8 each |
| `0x07001` (the RK3588 word) | 28673 | 1 of 8 |
| `0x06001`, `0x04001`, `0x02001`, `0x01001` | 24577, 16385, 8193, 4097 | 0-1 of 8 |

### Regcmd writes searched

Each of these leaves the RK3576 at one job per power session [HW sweep, H96 MAX M9]:

- Sweeping the CNA/CORE/DPU/RDMA `S_POINTER` value over the whole ping-pong field
  (`POINTER`, both `PP_EN`s, `PP_MODE`, both `PP_CLEAR`s)
- Writing `PC_TASK_CON` from inside the stream
- Replaying the vendor's state-init sequence as regcmd writes

Two of those results bear on the mechanism. Forcing `POINTER=1` makes every job write nothing,
which shows the bit is live. Job 2 still fails with `POINTER` held at 0, so the consumer is not
advancing behind the writer [HW sweep, H96 MAX M9].

### Excluded causes

- *`TASK_COUNT_CLEAR`.* With the field correctly placed, every job writes with every control
  bit clear. With it misplaced, no control-bit value tried helps. `TASK_COUNT_CLEAR` is
  therefore not the mechanism.
- *The ping-pong register groups.* Job 2 fails with `POINTER` held at 0, which rules them out.
- *The vendor's `state_init` hook and `pc_dma_ctrl`.* Back-to-back submits write without them.
  The joint test of the two against the two-core corruption is under "Two-core operation",
  Excluded causes, with the reason they were applied together. Its test is
  `tests/rk3576_core_pair.c pair`.

### The `POINTER` field under `PP_MODE`

The second party's evidence, from a Radxa ROCK 4D [linux-rockchip RFC v4 4/6 cover and code
comments, Jiaxing Hu, 2026-08-03, their measurement, not reproduced here]:

- Bit 0 written as 0 reads back as 1, on every job, for the rest of the session.
- Flipping it per submit, in the direct register writes *and* in all four `S_POINTER`
  entries of the regcmd, moves neither the readback nor the result.
- Selecting a bank the way the vendor's `state_init` does, with the `PP` bits clear,
  stops the units arming at all.
- Pulsing `POINTER_PP_CLEAR`, with or without `EXECUTER_PP_CLEAR`, moves nothing.
- A 20 KB read snapshot of `pc`, `cna`, `core`, `dpu` and `rdma` differs in exactly one
  word, `OPERATION_ENABLE`. The snapshot is taken at the same point in a job that
  computed and in one that did not.

Forcing `POINTER=1` from inside the regcmd stops every job on the H96 MAX M9, where the report
finds that flipping it changes nothing. Two candidate reasons remain, and neither is
established:

- The writes go in from different places: a regcmd the PC is fetching, against a
  slave-mode register write.
- The posted series has no equivalent of `0008`. It still emits mainline's
  `PC_TASK_CON_TASK_NUMBER(1)` against the RK3588's 12-bit accessor.

The posting does not show whether the tree the experiments ran on had such a fix. The second
party's public repo has had its own 16-bit fix since at least 2026-07-26.

On a driver that still programs the RK3588's 12-bit `PC_TASK_CON` word, the second submit of
every power session writes nothing. Every configuration of a ping-pong sweep then reads "no
change", however exhaustive the sweep.

### The `state_init` hook in `rocket`

Placing the hook in `rocket` is straightforward. The driver's `rocket_job_hw_submit()` already
opens with `PC BASE_ADDRESS = 0x1`, which is slave mode, and the vendor's CNA writes drop
straight into that window. To match the vendor at power-on, the place is the runtime-resume
callback.

## Register encoding: Working against an unpatched driver

With `autosuspend_delay_ms` at 0, a back-to-back loop writes 8 of 16 jobs, and 6 of 6 on a
shorter run.

The `rk3576_first_light` test checks per task that its own output was written, and resubmits,
when it runs a row split. The `rk3576_first_light` test also retries past the autosuspend on
both signatures of the cold-start defect. It treats an all-zero output as the defect only when
the CPU model predicts a non-zero one. A sweep that reports its results without that check
records the defect, not a register result, roughly half the time at multi-second gaps.

## Register encoding: The per-submit floor is the driver's completion poll

The `rk3576_submit_floor` test submits the 8×64×32 int8 matmul 300 times and reports the time
per submit.

The convolution test's 102 shapes take about 5.9 s, unchanged, at every poll period from
1000 us down to 150 us. Host packing and the test's CPU model dominate that test, not submits.
Its total therefore reads as a flat, negative result while it measures something else. A single
small shape is timed instead, which is what `rk3576_submit_floor` is for.

### Chained row tasks

A concatenated regcmd stream in one drm task, checked per task, names the earliest missing
task. At 3 tasks and at 6, task 0 is written and task 1 is the first missing, on every one of
eight attempts. The power domain was confirmed cycled between attempts, so the failure is
deterministic and is not the wide-output hazard. Across the convolution test, 27 shapes fail,
all of them multi-window or multi-group plans, and no single-task one [HW sweep, H96 MAX M9,
2026-07-28].

With chaining on, the convolution library test reads 156 passed, 7 refused as required and 0
failed. That result is identical to the one-submit-per-task path. A concatenated stream in one
drm task fails 27 shapes of the same test [HW sweep, H96 MAX M9, 2026-07-29].

Over the whole convolution library test, chaining moves the total from 6643 ms to 6463 ms,
about 3%. Forcing `ROCKET_RK3576_MAX_ROWS` down to 8 or 4 does not enlarge the saving, because
most shapes in that test plan into one or two row tasks.

### `PC_DONE` and the DPU completion bits

Each correctness point of the poll-period table is 3 runs.

The partial output reproduces on `n-tiled-deep`, the largest output in the int8 matmul test at
32×2048×3072. At `poll_interval_us=250`, retiring on `PC_DONE`, it fails 3 runs of 3, with
~95-98k of 98304 elements exact. A snapshot of the output BO is compared after a settle and a
re-sync. It shows the last 2048 bytes of the 65536-byte output arriving after the fence
retired (`changed=2048 first=63488 last=65535`) [HW sweep, H96 MAX M9, measured 2026-07-28].

The grace's failure point, with the poll retiring on the DPU bit [HW sweep, H96 MAX M9,
measured 2026-07-28]:

| `dpu_grace_us` | 200 | 250 | 300 | 350 | 400 | 500 | 800 |
|---|---|---|---|---|---|---|---|
| `n-tiled-deep`, 10 runs | 10 fail | 0 | 0 | 0 | 0 | 0 | 0 |
| `fc-rgb-224-k7`, 5 runs | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The test suites cannot separate 250 from 500, because nothing in them drains further. The case
for a margin rests on the drain itself. The drain is measured per test, with the poll retiring
on the DPU bit and the grace set past every wait.

The drain instrument is a module parameter set on the board's out-of-tree debug tree. Its
`dbg_dpu_wait_max_us` and `dbg_dpu_waits` hold the maximum wait and the count over jobs that
retired on the DPU bit. Jobs that raise no DPU bit time out into `dbg_grace_hits` instead. The
instrument is diagnostic and kept out of the patch series, so a reimage removes it along with
everything else:

| test | worst drain | jobs reaching the DPU bit | jobs raising none |
|---|---|---|---|
| `rk3576_conv_gate all` | 99 us | 15 | 0 |
| `rk3576_conv_lib_gate` | 745 us | 30 | 31 |
| `rk3576_matmul_gate` | 290 us | 9 | 58 |
| through the wide int32 writer | 343 us | 10 | 54 |
| `rk3576_argb_ic1` | 46 us | 2 | 0 |

The 745 us is one shape, and it reproduces to within 3 us over five runs.

`librocketnpu` sets `DRM_ROCKET_JOB_NO_DPU_DONE` on the classes the poison probe names wide.

A negative A/B on the flag is first a claim about the wiring. One silently failed edit is
enough to leave the poll comparing against `dpu_grace_us`. The flag then reaches the driver and
changes nothing, and the A/B shows no effect at any `dpu_blind_us`. In that state,
`dpu_grace_us` 500 -> 3000 moved the same eight-slice time 15.31 -> 36.14 ms. The tasks were
still waiting out the grace, so the hint was not being applied.

With the DPU bit as the retire condition, the poison probe's `scope` map is unchanged. The
probe's `pair A int8d`, at `ROCKET_PP_REPS` of 4 and 8, gives grace hits from the A side only,
scaling exactly with the rep count. With A as `fp16fc` the hits are 4 and 8, one submit a rep.
With A as `i32` they are 8 and 16, two submits a rep. The hazard prevents the `int8d` submits
on the B side from writing. Those submits contribute zero at every rep count
[HW sweep, H96 MAX M9, measured 2026-07-28].

### The half-started job runtime-PM pin

On a pinned device every i32 shape writes nothing (`0/256`, `0/32`, `0/224`). Everything else
still works: the convolution test is 102/102, the library test 111/1, and the plain int8
matmul shapes all pass. A poll-period sweep run on a pinned board reports the matmul failing at
every period.

Without `0010`, one injected attach failure produced seven, and leaked seven references. With
`0010` the injected failure is the only one [HW sweep, H96 MAX M9, measured 2026-07-27].

### Revision history

- 2026-07-28: at a 50 us poll period a submit of the 8×64×32 int8 matmul cost 439 us (against
  1654 us at the stock 1 ms, 3.77x less) [HW sweep, H96 MAX M9]. The module was a `rocket` with
  patches `0006`, `0009` and `0012`, before interface version 1.0. Re-measured on 2026-07-31,
  pinned to one A72 over eight runs across three modules, the median is 329-331 us through
  `0020` and 344-348 us through `0021` and `0022`. The 439 us does not reproduce on any module
  then on the board. The poll-period table of "`PC_DONE` and the DPU completion bits" is from the
  2026-07-28 series and keeps its own values.
- 2026-07-29: chaining was stated to save `(n-1) * 439 us` per call. It saves one submit's cost
  per row task removed.

### Tasks that raise no DPU completion

#### Revision history

- 2026-07-28: a task whose DPU output element is wider than one byte was recorded as completing
  CNA and CORE and never setting a DPU bit. The int32 writer and the fp16 output were both in
  that class. The module was `rocket` with patch `0012`, before interface version 1.0.

  The 2026-09-27 sweep on `rocket` 1.6.0 supersedes it for the fp16 programs. The input-channel
  slice and the packed-image first conv fence 0.38-2.2 ms after the submit. Every element is
  written and no deadline retirement occurs [HW sweep, H96 MAX M9]. The narrow and wide int32
  writers still raise none. Without the flag each waits the 125 ms backstop, 5 of 5 [HW sweep,
  H96 MAX M9, rocket 1.6.0, 2026-09-27].
- 2026-07-28: the fp16 path's gain from lowering the default (then stated as 1.18x) was read as
  a cost paid only by the tasks that raise no DPU completion. On that reading, collecting it
  needed the two classes separated at submit time [HW sweep, H96 MAX M9].
- 2026-07-29, interface 1.2: `librocketnpu` set `DRM_ROCKET_JOB_NO_DPU_DONE` on the fp16 direct
  conv, the fp16 first conv and the int32 matmul writer. On an fp16 convolution at `ic=128
  oc=32` 28×28 k3, run as eight wide-output slices, the settle was the dominant cost:

  | `dpu_blind_us` | 250 | 500 (hint neutralized) | 3000 |
  |---|---|---|---|
  | eight-slice time | 13.67 ms | 15.20 ms | 36.96 ms |

  The hint took 10% off that path, and every narrow-output job kept its full 500 us deadline
  [HW sweep, H96 MAX M9, measured 2026-07-29]. The flag was then described as costing time and
  never correctness. The 2026-09-27 sweep on `rocket` 1.6.0 supersedes that. Under the flag,
  k9 over 32 channels on 120×64 fenced at 0.74 ms with the last rows of every 8-channel plane
  unwritten, 10 of 10. Without it, the same test point fenced at 2.05 ms with every element
  written [HW sweep, H96 MAX M9, rocket 1.6.0, `tests/rk3576_drain_probe raw`]. The library has submitted the fp16
  programs without the flag since then.

### Completion and the wide-output hazard

The pre-1.0 reading that a victim raises its DPU completion comes from the poison probe's grace
counts. With A as `fp16fc` or `i32`, grace hits came from the A side only. The `int8d` victims
on the B side contributed zero at every rep count [HW sweep, H96 MAX M9, measured 2026-07-28].
On `rocket` 1.6.0 a slice submitted after the hazard fenced at 665 us at grace 500. At grace 0
it waited the 127 ms backstop [HW sweep, H96 MAX M9, rocket 1.6.0, 2026-09-30].
The cause of the difference between the two modules is not isolated.

## Register encoding: Kernel oops on a rejected submit

The test `tests/uapi_submit_errpath_rocket` fires one malformed submit per site from a forked
child. Before the fix 0 of 5 sites returned to userspace, and after it 5 of 5 did [HW sweep,
H96 MAX M9, 7.1.3].

The client that reached the oops on the RK3576 was `librocketnpu`'s uAPI selftest. Its
deadline canary built its task descriptors from the parameters a refused `gen_matmul_fp16()`
left unset.

### Client-supplied addresses

The probe `tests/uapi_regcmd_fault_rocket` tries both bad-address cases, as its `read` and
`write` modes. On the RK3576 both give 0 WARNINGs, the taint unchanged, and the matmul test
46/46 immediately afterwards [HW sweep, H96 MAX M9, 2026-07-28]. On the RK3588 the same probe
reaches the detach warning [HW sweep, Turing RK1, 2026-07-28].

## Register encoding: A BO in an in-flight job must survive the file that made it

The test `tests/uapi_bo_lifetime_rocket` opens a fresh fd per iteration, submits, and closes
immediately. It warns on 16 of 16 iterations, and left running longer, the same client reaches
the dereference. With `0014`, 0 of 16 iterations warn, on the same kernel with only that patch
changed [HW sweep, H96 MAX M9, 2026-07-28]. The verdict is the kernel log: a run can pass every
ioctl and still be the failing side of the A/B.

## Register encoding: The matmul as a 1×1 convolution

The convolution test runs a real matmul shape without calling it one. Its `w-e4608-k1` is
`ic=4608`, `oc=128` on a 4×2 VALID plane: M=8, K=4608, N=128 in one submit. The int8 matmul
test is `tests/rk3576_matmul_gate.c`, and the probe `tests/rk3576_matmul_probe.c` measured the
limits in the sections that follow.

### Throughput and the N axis

The throughput table's conditions stated a submit at about 0.44 ms whatever its work
(2026-07-27). The per-submit time is in "The per-submit floor is the driver's completion
poll".

## Register encoding: Matmul precision

### The matmul probe

The probe `tests/rk3576_matmul_probe.c` measured the limits in the matmul sections that
follow. It has these modes:

- `shape`: the mapping against a CPU model
- `m`: the M axis, every factorization
- `cost`: both precisions on one shape
- `peak`: MACs per submit by N
- `ncap`: the N boundary, with repeats and a failure classification
- `out32`: the 32-bit output width, decoded

Its settings are `ROCKET_MP_I32`, `ROCKET_MP_IW`, `ROCKET_MP_KS` / `_NS` (comma lists),
`ROCKET_MP_REPS`, `ROCKET_MP_GAP_MS`, `ROCKET_MP_OSLACK` and `ROCKET_MP_VERBOSE`.

### The precision table

The table measures each shape at both precisions, with the pacing outside the timing.

## Register encoding: Output-channel bounds

Each point of the bound was run three times [HW sweep, H96 MAX M9]:

- 2944 output channels is correct in 3 of 3 runs at every K tried. 3072 is intermittent (1 of
  3 at K=256, 2 of 3 at K=1024), and 4096 is never correct.
- The weight cube at `ic=4096, oc=1536` is 6 MiB and is correct in 3 of 3 runs. At
  `ic=4608, oc=1536` it is 6.75 MiB and is correct in 0 of 3. At `ic=4096, oc=2048` it is 8 MiB and loses about a quarter of its
  channels.

A single pass per point produced a clean-looking N ceiling that moved when the same points were
run again. The shape `1024x256x3072` failed in one pass and passed in the next. Three repeats
plus a classification of the failure separate the bound from the noise. All-sentinel is a dead
submit, a clean prefix is a capacity bound, and wrong values in a fully written output are
arithmetic. [HW sweep, H96 MAX M9]

## Register encoding: The int32 output

### The map decode

The decode gives every accumulator a distinct value, so each value names exactly one word.
Every position read that way fits the map with no misses. At `oc = 4` and `oc = 8` the decode
reads 32/32 and 64/64, with no waste.

### The block count and the delivered-channel rule

The block count and the delivered-channel rule are one fact. A sweep run only at `oc = 32`
cannot tell the block count from the delivered-channel rule, and it reads as "the extent is
`ceil(oc/16)` groups". The shapes at `oc = 16` and `oc = 8` separate them, and both write two
blocks.

## Register encoding: The wide writer

### The map decode

The map is decoded, not fitted. The operands were drawn so that every accumulator in the tile
is distinct. Each written word therefore names exactly one (channel, pixel), or the map is not
a map. It was read off at `oc` 8/16/24/32/40/48/64/96/128/192 and at pixel counts
4/5/6/7/8/12/13/16/21/33. In the field sweep at `M=8, K=32, N=32`, every attempt at
`0x4010[2:0]` = 7 (tf32) was a dead submit.

### Wide-writer bounds

Forced onto eleven shapes whose planner plane is `Mx1`, the wide writer was wrong on ten. At
`iw = 1`, where the two readings coincide, all eleven are exact and every zero run disappears.
[HW sweep, H96 MAX M9]

The cap was lifted (`ROCKET_RK3576_I32_WIDE_SURF`, `tests/rk3576_i32_height_bound.c`) for a
sweep at `K = 32` and `ow = 1`. At every one of the eight programmed counts, the first wrong
height is the smallest `oh_full` with `oh_full * oc_prog >= 4096`:

| `oc_prog` | 64 | 128 | 192 | 256 | 320 | 384 | 448 | 512 |
|---|---|---|---|---|---|---|---|---|
| first wrong `oh_full` | 64 | 32 | 22 | 16 | 13 | 11 | 10 | 8 |

Predicted and observed agree in all eight. [HW sweep, H96 MAX M9]

The contraction's relaxation of the bound also hides `iw`. A sweep of the plane width at
`K = 512` finds `ow = 2` and `ow = 4` exact across the whole range in which `ow = 1` fails. At
that K the bound is far outside the geometries tried, so the sweep cannot see the plane width.
At `K = 256` the same doubling breaks it: `oc_prog = 512`, `oh = 7` is exact at `ow = 1` and
wrong at `ow = 2`. The plane width shows in the bound only when it is swept at a short
contraction.

The probe that drives every output channel live finds a boundary that the library, which
leaves the undelivered half zero under its scatter, never meets. The same geometries are
bit-exact through the library at `A` 48, 64 and 128. A boundary found with every channel live is
therefore a boundary of the probe. Setting `ROCKET_MP_SCATTER=1` drives the probe the way the
library drives the SoC.

### Zero atoms and their detection

Inside the bound the failure persists. It is not the wide writer's alone. The
narrow writer does it too, and without a per-atom check it does so silently. At
`M=128 K=256 N=2048` the narrow writer returned a handful of elements out of 262144 reading zero
on six calls in twelve. It is the same defect on both writers, and it is not rare: see the
memory-pressure measurement below.

The rate was measured with four load threads at `M=128 K=256 N=2048`. Each test point is 30 reps
and 120 row tasks [HW sweep, H96 MAX M9, 2026-07-28, `tests/rk3576_i32_zero_run.c load`]:

| host load                          | row tasks that dropped an atom |
|------------------------------------|-------------------------------|
| quiet                              | 5.8%   (7/120)                |
| integer spin, no memory traffic    | 11.7%  (14/120)               |
| memcpy over 16 KiB, L1-resident    | 9.2%   (11/120)               |
| streaming memcpy over 32 MiB, DDR  | **51.7%** (62/120)            |

The two controls hold the CPUs equally busy and the cache hierarchy equally hot while leaving
DDR alone. At this sample size they are not distinguishable from quiet (7, 11 and 14 of 120,
against a binomial sd of about 3). Streaming to DDR is 5-9x all three and far outside that.

At the load that takes the int32 writers to 51.7%, `rk3576_conv_lib_gate` is 156 passed / 7
refused / 0 failed. It reads the same under DDR pressure, under the L1 control and quiet. Under
DDR pressure `rk3576_matmul_gate` is 46/46. [HW sweep, H96 MAX M9, 2026-07-28,
`tests/hostload.c`]

One 155/156 flake appeared under the *spin* control, which touches no memory. The same
single-case flake also appears with no load at all. It is therefore a pre-existing intermittent
in that test, not a load effect, and it is not pinned to a case. The conv entries' write check
cannot see a task that wrote all but a few atoms. Their result under load is therefore a
measured negative rather than a covered one.
[HW sweep, H96 MAX M9, 2026-07-28, `tests/hostload.c`]

Every wrong element is zero over the whole map: eight channel counts, 57 heights each.

The sentinel stamp is bracketed by `PREP_BO` and `FINI_BO`, so no dirty line is left to race the
DPU's DMA. It is verified to reach DDR before any submit. The bad atoms come back zero rather
than holding the sentinel, so the writer reaches them and the data in them is wrong. On a fresh
BO a wrong zero reads as a dropped write.

On one shape, every failure over 40 runs ended at `s = 2A-2`, two emissions short of that
group's last. Logged at debug level across shapes (`r76_i32_run_log()`), `i32-tall-m` and
`i32-m100` corrupt three emissions in mid-stream. They sit at `s = 1039..1040` of 0..2047 and at
`s = 15..16` of 0..199. Both recur at a fixed address, so a tail-keyed detector would miss all
of it. [HW sweep, H96 MAX M9, 25 runs per shape]

With the library's detection, all 240 reps at the worst shape were exact, 120 of them under
host load. The all-zero-data case (every row of `A` zeroed) passed on both writers. With every
row of `A` zero and the bias zero with them (`ROCKET_MG_ZERO_ROWS=1`), both writers pass the
whole int32 test.

### Excluded causes

- *A readback race.* A second fence and a second read return byte-identical contents.
- *The wide-output hazard.* Its signature is an *empty* region, and cycling the power domain
  clears it. This defect is indifferent to the idle ahead of it. Host memory traffic moves its
  rate fourfold while leaving the hazard's count alone. Its redo needs no idle in front of it,
  which is further evidence against the hazard.
- *CPU occupancy and scheduling latency.* The integer-spin and L1-resident controls hold the
  CPUs equally busy without raising the rate.

### Cost and the writer choice

Figures that show the wide writer ahead, 8 submits to 4 at `i32-n4096` and 4 to 2 at
`i32-n2048`, were measured against the rows-only cap. That cap let those shapes run one task per
tile at many times the output the writer can hold. Measured through `rk3576_matmul_gate`:

| shape | 4x writer | 2x writer |
|---|---|---|
| M=64 K=1024 N=2048 | 4 submits, 686 ms | 2 submits, 339 ms |
| M=32 K=1024 N=4096 | 8 submits, 1319 ms | 4 submits, 675 ms |

The wide path is kept for two reasons. It is the instrument that decoded the 32-bit writer's
map, and a shape outside the 315-shape sweep can still reach it. [HW sweep, H96 MAX M9]

## Register encoding: The dense writer: every accumulator as an int32, and no poisoning

The dense writer's registers and map were measured with `tests/rk3576_matmul_probe out32`
[HW sweep, H96 MAX M9, rocket 1.6.0, 2026-09-30]. Doubling `0x401C` and `0x40B8` produces an
overlapping map, and a count read off it is a write race. The same registers read 160 and 224
of 256 in two runs.

## Register encoding: The wide-output hazard after an int32-output job

### What clears the state

Three measurements show that the driver's runtime-PM autosuspend clears the state by cycling
the NPU power domain, and that elapsed time does not:

- With `on` written to the device's `power/control`, runtime suspend never fires. No gap then
  clears the state: 100 ms, 300 ms and 600 ms each write nothing.
- The working gap tracks `power/autosuspend_delay_ms` one for one. At a 5 ms delay a 5 ms gap
  already gives 12 writes of 12. At 50 ms it takes 80-120 ms for the same.
- `power/runtime_suspended_time` advances across the gap, so a suspend did happen.

The matmul test's int32 half runs in 553 ms at the stock 50 and in 195 ms at 5.

### The program split

The split is measured with a canary (`tests/rk3576_poison_probe.c`). The probe submits the
program under test and then the canary, and scores the canary. The canary is a program known to
write and known not to leave the next submit unwritten. A dead program and one that sets the
hazard therefore stay distinguishable.

Over seven program kinds the split is exact. Each program that sets the hazard does so before
every one of the seven, and each that does not, before none. Every program under test wrote its
own output in every test point, so no test point is a dead program reading as a clean one.

After one int32 job, six consecutive int8 jobs at a zero gap all wrote nothing. The state
persists until the domain cycles, so the guard cannot be a cheap dummy job.

### Redo depth

Over twelve fp16 conv test runs the redo fired on 358 row tasks. The attempt that failed *next*
was attempt 2 for 45 of them, 3 for 10, 4 for 2, 5 for 1 and 6 for none. Every one recovered.
Every cycle in between was confirmed to have taken the domain to `suspended` first, so the guard
was never merely unobserved. [HW sweep, H96 MAX M9, 2026-07-28]

A cap of four attempts leaves about 0.6% of retried tasks failing outright. The conv test suite
saw the same rate as a "1-in-156 intermittent". At eight the failure is unobservable
(15 of 15 fp16 runs and 3 of 3 full runs clean, deepest retry attempt 4). [HW sweep, H96 MAX M9,
2026-07-28]

Asking again after a 2 ms settle before the redo rescued 0 of 67 unwritten tasks, while the
power cycle behind it recovered all 67. The re-check costs 14% of the fp16 wall-clock time and
recovers nothing (`ROCKET_RK3576_DRAIN_US`, default 0).

### The per-task check

The per-task check was found on the narrow writer. It ran clean under test shapes that never
split a tile into row tasks.

### Excluded causes

Nothing about the buffers changes the state. Each of these was tried, and none helps:

- Fresh buffers per submit
- A different feature address
- A different output address
- Packing every K slice up front, so no buffer is rewritten between submits

### Cost of the clearing idle

A 5×7 sweep sets `power/autosuspend_delay_ms` against the inter-submit gap. It drives a chain of
20 fp16 first-conv submits and counts how many wrote:

| `autosuspend_delay_ms` \ gap ms | 0 | 5 | 10 | 20 | 40 | 60 | 100 |
|---|---|---|---|---|---|---|---|
| 50 (stock) | 1 | 1 | 1 | 1 | 1 | 9 | **20** |
| 20 | 1 | 1 | 1 | 9 | 19 | 18 | **20** |
| 10 | 1 | 1 | 11 | 18 | **20** | 20 | 19 |
| 5 | 1 | 7 | **20** | 18 | 20 | 19 | 20 |
| 1 | 1 | 19 | 19 | 18 | 19 | **20** | 20 |

The resume cost was measured with `ROCKET_SF_GAP_MS` on `rk3576_submit_floor`, in us/submit:

| inter-submit gap | 0 | 5 | 10 | 16 | 20 | 33 | 60 | 100 |
|---|---|---|---|---|---|---|---|---|
| delay 50 ms | 1062 | 1123 | 1130 | 1124 | 1136 | 1139 | 1283 | 1399 |
| delay 1 ms | 1059 | 1319 | 1334 | **1372** | 1352 | 1368 | 1366 | 1378 |

At a zero gap the two are identical, because the autosuspend timer never fires. A plain
per-submit-cost probe, which runs at a zero gap, therefore reads the change as free. The two
converge again past 60 ms, where both suspend anyway.

Measured at the stock 50 ms delay with the driven transition, every test stays bit-exact (conv
102, conv-lib 154 + 8 refusals, matmul 43 on both writers, refusal 15, regcmd 194, both fp16
sweeps). The fp16 first conv's whole test group runs in 2.3-15.6 ms against 107-334.
[HW sweep, H96 MAX M9, measured 2026-07-27]

### Driver completion poll

The live `poll_interval_us` parameter is how the period was swept against correctness rather
than wall-clock time, and how a candidate fix is tested.

The driver has `rocket_core_reset()`, a `reset_control` assert, `udelay(10)`, then a deassert.
The vendor `rknpu` driver soft-resets only on a timeout or an abort, never per job.

### Related hazards

Without the stamp, one full matmul-test run in twenty failed its very first shape this way in a
soak, at 1 element of 256 correct. With the stamp, 20 further test runs were clean. [HW sweep,
H96 MAX M9, measured 2026-07-27]

The int32 sweep's IOMMU wedge showed a stall timeout plus a `rocket_job_irq_handler` warn at
`rocket_job.c:528`. Treating it as a buffer-size problem is a false trail: the measured extent
sits well inside the BO that was allocated.

The int32 sweep's IOMMU wedge was cleared by a reboot on 2026-07-27. That was before the module
reload with an eight-second settle was found, later that day. The reload was found to clear the
IOMMU wedges induced by a 100 us poll period and by a per-job core reset.

## Register encoding: The DPU LUT

### Captures

The script `tests/data/rk3576-vendor-capture/lut/mklut.py` builds the LUT captures, and
`decode_lut.py` decodes them. An ONNX model with a nonlinear activation, compiled for
`rk3576`, emits the table load verbatim. Every convolution capture leaves the LUT bank
(`0x4100`-`0x4194`) zero, which is why `npu_regcmd_rk3576.c` writes the whole bank as zeros.
That zero bank is a property of the captures, not of the silicon.

The control is a bare convolution, whose whole LUT bank is zero. Against it, the convolution
that follows the load program writes these registers:

- `LUT_CFG 0x4108 = 0x02000006`
- `LUT_INFO 0x410C = 0x00050500`
- `LE_START` in all four lanes (`0x4110`-`0x411C`) = `0xffffc000`
- `LO_END` in all four (`0x4140`-`0x414C`) = `0x00004000`

The registers that turn the LUT on come from a diff against the bare-conv control of the
same capture set. A diff across activations hides them.

### Table span

The span is recoverable from the first difference and the function's own derivative at the
table boundary. Sigmoid's `d = 100` at the boundary with `sigmoid' = 0.25` gives a step of 0.0122 and a
half-span of 6.25. Tanh's `d = 193` with `tanh' = 1` gives a step of 0.00589 and a half-span
of 3.02. Both agree with the endpoint values to three digits.

### Output converter

The output converter matches each op's output quantization to five digits on two functions,
sigmoid and tanh. Five digits on two functions is not a coincidence, and it means these three
registers say nothing about which interval an input falls in. [Manufactured capture]

### Input map

The test `tests/rk3576_lut_probe.c` reads the map off the RK3576 with `librocketnpu`'s own
table-load program and a table whose entries encode their own index. A convolution drives the
table. Its accumulator is one feature byte, and its per-output-channel `A` and `C` place that
byte anywhere in the datapath. Each case runs 8192 samples per span, with a maximum
disagreement of one output count (`rk3576_lut_probe gate`, 4 of 4) [HW sweep, H96 MAX M9].

The vendor never moves these registers, which is why no diff of vendor programs can produce
this map.

The first run of the test covered four different `C` and all four spans. Every
disagreement in it sat on the one value `value == LO_END` and nowhere else. That value is the half-open top of
the domain.

### Interpolation

The probe walks a step table one datapath unit at a time across the interval it steps in.
The walk gives 33 distinct output runs over 32 units, rising linearly, not two.
[HW sweep, H96 MAX M9]

### `BN_CFG` and the BN operand

The four registers the vendor's own programs leave alone (`0x4040`, `0x4054`, `0x4064`,
`0x4068`) were each driven, and none of them moves it. Where the RK3576 keeps
`BN_MUL_OPERAND` is therefore open.

### The regcmd encoder

With `lut` NULL, every program `gen_conv2d_int8_rk3576()` emits is byte-identical to the ones
emitted before the field existed, which `regcmd_rk3576_gate` asserts.

## Register encoding: The PPU pooling program

### Captures and the register fit

Nothing in any found capture drives the pooling engine. Manufactured captures do
(`tests/data/rk3576-vendor-capture/pool/mkpool.py`). A sweep over method, kernel, stride,
plane, channel count and padding gives the meaning of the 23 PPU writes. The eight PPU_RDMA
writes are fitted over the same sweep.

For the input extent in `0x600C`/`0x6010`, every non-square and odd-plane case in the sweep
agrees with the consumed-extent reading, and a plane-height reading does not.

For the pad nibbles in `0x6040`, every capture's pool declares a symmetric pad, and a
symmetric programmed pad cannot tell the two orders apart. The SAME k3 s2 capture over 16
declares one on every edge (`pads=[1, 1, 1, 1]` in `mkpool.py`) and consumes no trailing
pad. Its `0x0011` is therefore the leading pads alone.

### The destination base address

The address was read off the RK3576 by driving, in turn, each PPU register that reads zero in
every capture. Those five are 0x6054, 0x6058, 0x605C, 0x6070, 0x60DC. Each run asked only
whether the output BO moved off its sentinel, and only 0x6070 writes. [HW sweep, H96 MAX M9]

### The trailer

One whole register sweep came back empty on the `PC_OPERATION_ENABLE` fault. Reading the raw
capture word by word, not through the block classifier, is what found it.

### Bit-exactness

With those fields, `gen_pool_rk3576()` is bit-exact against a CPU model over 11 shapes in
`tests/rk3576_pool_probe.c` [HW sweep, manufactured capture, H96 MAX M9, 2026-07-29].

### The source surface stride

The unpadded stride was tested with `rk3576_pool_probe lib`, at both strides, with
`ROCKET_RK3576_POOL_PACK_SRC=1` forcing the packed one [HW sweep, H96 MAX M9, 2026-07-31]. A
wrong stride there would compute a full and plausible output rather than faulting. The stride
is therefore a test rather than an assumption.

### Pool completion

The wait was measured through the library entry [HW sweep, H96 MAX M9, 2026-07-31]. It
tracks the parameter count for count, which says the job is not retiring on a completion at
all.

### Revision history

- The low half of `0x6040` was read as the end pads (right and bottom) from the manufactured
  SAME k3 s2 capture over 16, which programs `0x0011`. That reading assumed TFLite's SAME, a
  trailing pad of one and no leading pad. The capture's ONNX graph declares a pad of one on
  every edge (`pads=[1, 1, 1, 1]` in `mkpool.py`), and its last window consumes no trailing
  pad. `0x0011` is therefore the leading pads alone. The one-dimensional sweep
  (`rk3576_pool_probe pad`) ran at pads 0, 1 and 2 on both axes. It measures left in [3:0] and
  top in [7:4] [HW sweep, H96 MAX M9].

## Register encoding: The unprivileged double free, and the crashes it was mistaken for

### The client that reached it

A submit with `regcmd_count` still at its zeroed value is the exact shape `librocketnpu`'s
own deadline canary once submitted.

### Detection

`slub_debug=FZPU slab_nomerge` detects the defect [HW sweep, H96 MAX M9, 2026-07-29]. It
gives six "BUG kmalloc-16: Object already free" reports over three runs of
`uapi_submit_errpath_rocket`, and zero on the fixed module.

The same instrument reproduces it on a Turing RK1 at 7.1.1. A module with
`patches/rocket/081`-`086` gives six "Object already free" reports over three runs of
`uapi_submit_errpath_rocket`. The same module plus `087` gives zero over three runs. Both SoCs
give two reports per run, at the same allocation and second-free sites.
[HW sweep, Turing RK1, 2026-07-29]

On the fixed module, 83 of 83 `ctest` entries pass under `slub_debug` with the taint word
unmoved. [HW sweep, Turing RK1, 2026-07-29]

### The free-side fault

Instrumenting `rocket_gem_bo_free()` caught two live BOs naming one `sg_table`: two owners of
one slab object, not a lifetime race.

### The test that triggers it

The test `uapi_submit_errpath_rocket` asserts that five rejection sites return an errno, and
they do. An exit status cannot see the slab corruption on the way, and every test ran green
over it.

### Excluded causes

- *A lifetime race on the sg table.* The free-side fault reads as "an sgt torn down while
  still published", and it is not one. The instrumented free above shows two owners of one
  slab object.
- *Defects in the DMA mapping and GEM shmem code.* Both faults are the double free's
  corruption reaching downstream.
- *BO churn.* Neither 300 iterations of `uapi_bo_lifetime_rocket` nor eighteen rounds of the
  convolution, matmul and pooling tests fired, with core 1 bound and unbound. They stay silent
  because not one of them submits a job the kernel rejects. The crash tracked the presence of
  `uapi_submit_errpath_rocket` in the sequence, not the volume of BO churn. It fired in round
  1 on both runs that included it, and in no round of the two that did not.

  The crash does surface roughly once per few minutes of heavy BO churn, whichever test is
  running. That timing is right, and it does not make the churn the cause. The cause runs the
  other way: one test plants the corruption, and whichever test runs next collects it.

### Out-of-tree module builds without kernel headers

A board whose image provides no kernel headers can still take an A/B comparison. The RK1's
`linux-image` deb contains no build tree, so `make -C /lib/modules/$(uname -r)/build` has
nothing to use. The only thing a `modules_prepare` tree lacks is `CONFIG_MODVERSIONS=y`,
and the CRCs it needs are already on the board. A module built by that kernel contains them
in its own `__versions` section, 64-byte entries of an 8-byte CRC and a 56-byte name.

Read the CRCs back off the image's `rocket.ko` and write them out as a `Module.symvers`.
The owning module per symbol comes from `/proc/kallsyms`, which an unprivileged reader gets
with the addresses zeroed and the `[module]` tags intact. An out-of-tree build against a
stock kernel.org tarball at the same version then loads with a matching vermagic. The
script is `tools/mksymvers.py`. The tree needs the running `/boot/config-*` and a
`localversion` file with the Debian suffix, so that `UTS_RELEASE` comes out identical.

## Register encoding: The DPU's elementwise stage

### The output offset

The vendor's own Add program has an offset of -1. Replayed verbatim, it is exact on 8192 of
8192 elements with the offset added after the final shift
[HW sweep, H96 MAX M9, 2026-09-26, `tests/rk3576_add_probe.c ga54b`].

### The one-operand model

The model's test is `tests/rk3576_add_probe.c gate`, 10 shapes. The final shift's
half-to-even rounding was established here independently of the direct path. Against a
round-half-up reference, every disagreement was an exact tie: 25% of elements at a gain of one
half, and none at unity. [HW sweep, H96 MAX M9, 2026-07-31]

### The search for a second operand

The negative is closed. Eight sweeps cover the whole interface.

#### Registers left at zero

Every register the program leaves at zero was tried as a second base at every output gain.
Of seventeen candidates crossed with the full 32-step OUT_CVT ladder, nothing reaches the
output except `0x5024`, which injects a constant. That register is the DPU shift word, a live
per-task operand and not a spare. Crossing the two axes is what makes this conclusive. A
placement sweep at one gain is blind to an operand that sits somewhere unexpected and enters
the accumulator unscaled. A gain sweep at one placement is blind to it too.

#### Registers never written

Every register the program never writes at all was appended to the program one at a time.
This sweep is the complement of every other sweep here, and it is not an empty set. The register
file is not cleared between jobs on the RK3576, so a base the vendor programs in an earlier
task would still be standing. Twenty-nine candidates over the DPU and DPU_RDMA blocks, at two
gains, show that nothing holds an operand.

One register stops the write: `0x5068`, the RK3588's `RDMA_WEIGHT`. Its value stands after
the job that writes it. The shipped program stalled on 8 of 8 jobs after it, until the NPU's
power domains had been off for minutes. Written alone behind a power cycle, `0x5070` and
`0x5074` write exactly, so their stalls were `0x5068`'s standing value [inferred].

#### The main feed

The main DMA feed was measured at every gain from 2^14 down to 2^-31. It contributes exactly
zero at all 64 steps, with the elementwise operand live as a control at every one.

#### Mode-field grids

A joint grid over `EW_CFG` and `BRDMA_CFG`, and a second over `FEATURE_MODE` and
`ERDMA_CFG`. The second drives the `COMB_USE` field that the RK3588's own K-accumulation uses
to combine two feeds. In both grids, only the captured word writes at all, and every other
combination leaves the output untouched.

#### Destination accumulation

DPU `0x40C0` was swept over the captured word and 28 single-bit variants of it. Each variant
ran twice over differently pre-filled destinations, which classifies it. It always
overwrites, two bits (`0x00010000`, `0x00020000`) saturate the output, and none accumulates.

#### Addresses read

This sweep measures what the program reads, rather than what it was handed. The probe walks
a single non-zero 16-byte atom over an operand buffer eight cubes long, with the base pointed
at the middle. The addressed cube's 32 atoms map one to one onto the output, and nothing
outside it moves anything.

#### The candidates written together

The candidates went in together, with the main feed at a second operand. A condition of two
has no sufficient singleton, so all three went in at once, with a second operand, B, at
DPU_RDMA `0x5018`. Register `0x5068` took the RK3588's weight word `0x01010101`. The other
two took B's address, 0, or the plane's area and width. One configuration adds the RV1106
vendor Add's `0x5044` word, and each configuration ran over one A and two non-periodic B fills
at two gains. Every configuration that writes moves 0 elements between the B fills and matches
the one-operand model [HW sweep, H96 MAX M9, 2026-09-26], probe mode `ga54`.

#### The vendor's program, fed by its relocation table

Each `.rknn` has a relocation table naming the program words the runtime patches with each
tensor's address. In `bare_add_c32_16` the first operand goes to DPU_RDMA `0x5018`, the second
to `0x5038` and the output to DPU `0x4018`, and `subrev` swaps the two operands. The probe
replayed it verbatim with both bases patched, as one task and as its two slots. It wrote the
one-operand result on 8192 of 8192 elements and ignored B [HW sweep, H96 MAX M9, 2026-09-26].
This replay cannot see anything else the vendor runtime patches or submits, and the vendor's
Add has not been run on this silicon.

### The residual add on the convolution datapath

The lowering's test is `tests/rk3576_residual_add.c`. The vendor compiler's fold of
`Add(Conv(x), x)` into this shape is why that graph is useless as a capture of an add.

The lowering's first form fixed the larger weight at 127, which resolves the operand ratio to
about one part in 127 [rocket-userspace, 2026-07-31]. A search over every weight pair
in [1, 127] replaced it the same day [rocket-userspace, 2026-07-31].

### Vendor lowering of an elementwise add

The script `tests/data/rk3576-vendor-capture/add/mkadd.py` builds the captures, and
`decode_add.py` reads them. `Add(Conv(x), x)` is not a capture of an add. The vendor
compiler folds an identity skip into the convolution's own kernel, at the center tap of the
diagonal. Six such ONNX graphs, over four channel counts and three planes, compiled to a
register program indistinguishable from a plain convolution. A capture of the op needs an
operand the compiler cannot reach.

Two operands at the same scale are unattributable. When both operands calibrate to the
same range, every register that scales an operand looks the same for both. The two
converters then cannot be assigned. With deliberately asymmetric calibration amplitudes on the
inputs, the assignment falls out. That method, with `Sub` compiled in both operand orders, is
what names the elementwise cube as the *second* operand. It also pins the first's weight at
unity.

A capture's container says how many tasks an op costs. Each `.rknn` has a table of 40-byte
task records,
`[index][PC_OPERATION_ENABLE][slot size][int mask][int clear] [write count][…][program offset]`.
Matching `write count` against the program stream is what identifies them. Read the table
before concluding anything about an op's structure from the program list alone. A single
`Conv` emits four task records over four distinct program slots at two different row
geometries, so program count is not op count.

## Register encoding: Uncharacterized registers and functions

- The ARGB first conv's int8 weight cube. A quantized ARGB capture does take the path, but its
  weights do not survive quantization as a findable value set. This cube likely has to be read
  off the SoC, the way the depthwise int8 cube was.
- `0x1060`. Nothing pins it, so `librocketnpu` keeps it in the address-placement shotgun list
  rather than claiming it.
- `0x1064`. The one live vendor dump is an `rknpu` kernel capture taken at submit, not one of
  the ten programs decoded out of `.rknn` files.
