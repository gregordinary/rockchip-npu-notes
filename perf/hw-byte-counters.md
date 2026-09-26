# RK3588 NPU hardware DMA byte counters

**Verdict:** the RK3588 NPU does **not** expose usable hardware DMA *byte* counters
(weight-read / data-read / data-write bytes) through the mainline `rocket` driver. The
vendor's per-core "amount" counter offsets are **undecoded** on rk3588, and *reading* them
raises a bus abort that hard-locks the SoC. The legacy amount offsets alias DDMA
reserved space and read `0` regardless of traffic. **The system-level route this file used
to recommend has now been taken and works**: `rockchip_ddr` is a live `perf` PMU on the
RK1, calibrated here to within 0.4% against a known number of bytes, and §5 is what it
reads. It has no master-id attribution, so it measures the board rather than the NPU.

The motivation was to turn DMA-traffic
levers (CBUF operand reuse, resident weights, quantized readback) from wall-time
inference into bytes moved. (That specific question, does `DATA_REUSE` cut traffic,
was already answered HW-side by a −21% `wait` drop; the counters would have been
corroboration.)

## 1. The counters exist in the IP family, but not in rk3588's config

The NVDLA-derived NPU has DMA "amount" counters. The vendor BSP reads
them on *sibling* SoCs but **disables them on rk3588** (its rk3588 config sets the
top/core amount tables to NULL, and the read helper early-returns *"not supported on this
device"*). The offset tables exist, assigned only to rk3576 / rv1126b / rk356x configs:

| set | clr_all | dt_wr | dt_rd | wt_rd | used by |
|---|---|---|---|---|---|
| `rknpu_top_amount`  | 0x2210 | 0x2234 | 0x2238 | 0x223c | rk3576, rv1126b |
| `rknpu_core_amount` | 0x2410 | 0x2434 | 0x2438 | 0x243c | rk3576 |
| `rknpu_old_top_amount` | 0x8010 | 0x8034 | 0x8038 | 0x803c | rk356x, rv1106, rk3562 |

`pc_data_amount_scale = 2` on rk3588 (raw reads ×2); clear sequence (rk3588
`pc_dma_ctrl=0`) is `WRITE(0x80000101, clr); WRITE(0x00000101, clr)`. All offsets are
relative to a core's MMIO window (the BSP reads them from `base[0]`).

Note: do **not** confuse `0x14 PC_REGISTER_AMOUNTS` (the regcmd fetch length the driver
*writes*) with a byte counter; it is not one.

## 2. Address mapping: where the offsets land, and the `rocket` wrinkle

The BSP maps each rk3588 core as one 64 KB window: core 0 `0xfdab0000`, core 1
`0xfdac0000`, core 2 `0xfdad0000` (`rk3588s.dtsi`). So BSP `base[0]+0x2234` = phys
`0xfdab2234`.

`rocket` instead maps **three named sub-resources** per core (`rk3588-base.dtsi`,
`rocket_core.c`): `pc @ 0xfdab0000`, `cna @ 0xfdab1000`, `core @ 0xfdab3000` (each
0x1000). Its register map (`rocket_registers.h`, and Mesa `registers.xml`) has 10
domains at MMIO bases `0x0000/0x1000/0x3000/0x4000/0x5000/0x6000/0x7000/0x8000/0x9000/
0xa000` (PC/CNA/CORE/DPU/DPU_RDMA/PPU/PPU_RDMA/DDMA/SDMA/GLOBAL).

The decisive fact: **there are zero registers in the `0x2xxx` range in the entire map.**
The `0x22xx`/`0x24xx` amount block sits in a genuine gap between CNA (`0x1xxx`) and CORE
(`0x3000`), a page no domain covers, and which `rocket` does not map. The DDMA block at
`0x8000`, by contrast, *is* a defined, mapped domain.

## 3. Result 1: reading the `0x2000` amount page hard-locks the SoC

A debugfs probe `ioremap`'d core 0's `pc_base + 0x2000` page and read `0x2234` etc.
(domains powered, `pm_runtime`-guarded). Behavior:

- The **clear write** to `0x2210` survived (Device-nGnRE writes are early-acked).
- The **read** of the amount offsets **hard-locked the SoC**: no output, requiring a
  cold power-cycle.

This is the signature of an **unclaimed/undecoded MMIO read**: nothing decodes the
`0x2000` page, so the read raises a synchronous external abort that wedges the box (a
write posts and survives; a read must return data and cannot). Consistent with, and the
reason for, Rockchip's `amount_top = NULL` on rk3588.

> Caveat: "absent in silicon" is the leading explanation but not strictly proven; a
> hard *decode* abort argues for absent over merely power/clock-gated (a gated register
> typically reads 0 rather than aborting). Either way the offsets are unusable and
> unsafe to read via `rocket`.

## 4. Result 2: the `0x8000` DDMA block is readable (safe probe)

A read-only, disarmed-by-default probe of the DDMA domain (`pc_base + 0x8000`, phys
`0xfdab8000`) returned cleanly, confirming the abort theory (mapped domain decodes;
the `0x2000` gap does not). Core 0, NPU idle:

```
0x8030 CFG_STATUS      = 0x00000100   # IDEL (bit 8) = 1  -> DDMA idle
0x8000 CFG_OUTSTANDING = 0x00000fff   # WR_OS_CNT=0x0f, RD_OS_CNT=0xff (outstanding limits)
0x8004 RD_WEIGHT_0     = 0x01010101   # arbitration weights: PDP/DPU/KERNEL/FEATURE
0x8008 WR_WEIGHT_0     = 0x00010101   # WR weights: PDP/DPU
0x800c CFG_ID_ERROR    = 0x00000000   # no DMA ID errors
0x8010 RD_WEIGHT_1     = 0x00000101   # RD weight: PC
0x8034/38/3c (legacy DT_WR/DT_RD/WT_RD) = 0x00000000
```

**Validation:** these are structured values matching the Mesa DDMA bitfield definitions,
and `CFG_STATUS` `IDEL = 1` is *semantically correct* (the NPU was idle at read time),
proving we read real registers, not a floating bus. **Before/after a 320-job
`512×3840×4096` matmul** (real DMA traffic, ~31 MB weights × refetch), every value was
**identical**, including `0x8034/38/3c` still `0` and `CFG_STATUS` back to idle.

So the legacy `0x80xx` amount offsets are **DDMA reserved space** (not defined in the
register map) that **reads 0 regardless of traffic, not counters** on rk3588. The DDMA
block itself only exposes *configuration* (outstanding limits, arbitration weights) and a
coarse `IDEL` status bit, with no bytes-moved counter.

### A second witness disagrees on both pages, and the likely variable is power state

An independent RK3588 probe (poad42/opennpu_rk3588, `docs/ref/NPU_REGISTER_INVESTIGATION.md`;
kernel module, `ioremap(0xfdab0000)`, vendor `rknpu` on a 6.1 BSP kernel) reports the opposite
severities on both pages: `0x2210`-`0x223c` raising a **contained kernel oops** (DECERR, board
survives) where the probe above hard-locked, and `0x8000`-`0x803c` **hanging the bus** until the
watchdog rebooted, where the probe above read it cleanly.

The `0x8000` half is the tractable one, because both probes resolve to the same physical address
(`0xfdab8000`, their BSP 64 KB window at offset 0x8000, this one via `rocket`'s DDMA domain) and
this one is validated semantically rather than merely non-fatal: structured values matching the
Mesa DDMA bitfields, `CFG_STATUS.IDEL` correctly reading idle, and every field unchanged across a
320-job `512x3840x4096` matmul. A floating bus does not produce that. **The variable most likely
to separate the two is the NPU domain's power/clock state at read time**: the probe here is
`pm_runtime`-guarded with domains powered, and a *clock*-gated AXI slave stalls a read forever
rather than returning data, which is the exact signature they describe. [hypothesis: the
reconciliation has not been tested by re-running either probe against the other's power state]

Two things follow for anyone retrying this. **Power state is part of the probe design, not a
detail**; §6's guard exists for this. And the disagreement strengthens rather than weakens §3's
caveat: an independent DECERR from a different kernel on the `0x2xxx` page is further evidence it
is genuinely undecoded rather than power-gated, since a gated slave hangs where an undecoded one
aborts. Until someone re-runs it with power state controlled, treat **both** pages as unsafe on
either path; the cheapest wrong guess costs a cold power-cycle.

## 5. Conclusion, and the system-level route that answers it

- **No HW DMA byte counters via `rocket` on rk3588.** The real `0x22xx`/`0x24xx`
  counters are undecoded (fatal to read); the legacy `0x80xx` offsets are reserved and
  static.
- **Side-result:** the DDMA control/status block *is* safely readable; `CFG_STATUS.IDEL`
  is a coarse "DDMA idle" signal, but that is not byte accounting.
- **Bytes-moved ground truth comes from outside the NPU register space, and it is
  reachable today.** The kernel exposes the DDR controller's PMU as an ordinary `perf`
  uncore PMU, `rockchip_ddr`, and it sidesteps the unmapped-MMIO hazard entirely. What it
  costs is attribution: it counts the board, not a master.

### 5.1 The PMU, and the positive control that makes it quotable

On the RK1, kernel 7.2.0-1-arm64, `/sys/bus/event_source/devices/rockchip_ddr/` publishes
`bytes`, `read-bytes`, `write-bytes`, the same three per channel (`read-bytes0..3` /
`write-bytes0..3`), and `cycles`, each with a `.scale` of `2^-20` and a `.unit` of MB. The
`cycles` event tracks wall time at the memory clock exactly (2.112e10 over 10 s = 2112 MHz),
which says the PMU is running; it does not say the byte events mean bytes.

**They do, to within 0.4%** [HW sweep 2026-08-28, RK1]. `tests/ddr_pmu_cal.c` moves a known
number of bytes past every cache and `tools/ddr-pmu-cal.sh` differences two pass counts, so
that allocation, page faults, the kernel's page zeroing and the idle floor -- identical in
both arms -- cancel:

| known traffic | counted | ratio | the other column |
|---|---|---|---|
| 16384 MiB read (2 GiB buffer, 1 vs 9 read passes, 3 reps) | 16447.4 MiB | **1.0038** | +0.5 MiB written |
| 16384 MiB written (1 vs 9 write passes, 3 reps) | 16372.8 MiB | **0.9993** | +41.5 MiB read, 0.25% |

The idle floor is **4.8-7.3 MB/s** over 10 s, three orders below any figure below. The near-zero
cross-columns are a second check: a read pass charges nothing to writes, and a write pass
charges 0.25% to reads, so the A76's write-streaming mode is engaging and the columns mean
what they are named.

**The differential is not a nicety.** The same tool's single-pass arm counts **4.3 GiB read**
where only 2 GiB is known -- the surplus is the fault and allocation path, which no model of
the workload contains. An absolute system-wide count is not a measurement of a workload; a
difference between two arms of it is.

### 5.2 What one tiled matmul moves, against the analytical model

`tests/bytes_moved_rocket.c` predicts DRAM traffic analytically because no counter existed.
Now it can be checked. `tests/ddr_mm_bytes.c` runs the **single-fd streaming** path -- the one
the model describes -- with no CPU reference, at 5 and 45 reps, 3 reps of each; the difference
is 40 matmuls and nothing else. `512x3840x4096` fp16, KACC and data-reuse on, 600 MHz
[HW sweep 2026-08-28, RK1]:

| arm | model total | measured read | measured write | measured total |
|---|---:|---:|---:|---:|
| default (KACC + data reuse) | 133.50 MB | 287.30 MiB | 188.65 MiB | **475.95 MiB** |
| `ROCKET_REUSE=0` | 249.75 MB | 408.29 | 186.72 | 595.00 |
| `ROCKET_KACC=0` | 161.50 MB | 476.26 | 209.03 | 685.29 |
| prepacked (same arithmetic, weight scatter hoisted) | 103.50 MB | 152.38 | 74.22 | 226.61 |

Three separate results, and they do not all point the same way.

**The data-reuse term is confirmed.** The model says turning `data_reuse` off multiplies the
feature DMA by `nNt`=32, `+116.25 MB` of reads and nothing on the write side. Measured:
**+120.98 MiB read, -1.93 MiB write** -- 4% high on the term and zero on the column the model
says should not move. That is a real validation of the model's tiling and loop order (the
feature tile is reused across the `N` loop), and it is the first time any traffic claim here
has been checked against a counter rather than against wall time.

**The absolute level is 3.57x the model, and the excess is concentrated in the host scatter.**
Hoisting the weight scatter out of the loop -- the prepacked arm, same arithmetic -- removes
**249.34 MiB per call** against a `packB` term of **30.00 MB**: the bus charges that phase
**8.3x** what counting each weight byte once charges it, and it charges it in both
directions (`-134.92` read, `-114.42` write). The model counts an operand's bytes once per
logical movement; a host scatter moves useful chunks smaller than a cache line, so the bus
moves a line per chunk and moves it twice, source in and destination out.
**One term inside that difference is not separated**: `rocket_matmul_fp16` calls
`mm_bos_alloc` per call [source-confirmed, `src/rocket_matmul.c`], so the streaming path also
allocates, kernel-zeroes and frees a ~30 MB weight BO every call where the prepacked path does it
once. If the zeroing is the whole 30 MB, the scatter's own write amplification is 2.8x rather than
3.8x. Separating them needs a pack-only arm, which does not exist.

**The `KACC` sensitivity is wrong.** The model predicts `+28 MB` of writes and no change in
reads when K-accumulation moves off the NPU. Measured: **+20.4 MiB write** (same order) but
**+189.0 MiB read**, where zero was predicted. So the model's account of what the
un-accumulated partials cost is incomplete on the read side, and the `int8` readback floor
that account underwrites -- `nKt * M * N * 4` -- is a lower bound, not the figure.

**What a green result here does NOT show.** `rockchip_ddr` has no master-id filter exposed on
this SoC, so every figure is the whole board's traffic during the run; the differential removes
the fixed load but cannot attribute what remains between the NPU's DMA and the host packing
that same arm causes. That pair happens to be exactly what the bytes-moved model is about, so
it is the right total for this question and the wrong one for "how much did the NPU's DMA
move". A DRAM counter also cannot see traffic served from cache, so every number is a **lower
bound on data touched**. And this is one shape, one dtype, `n`=3 per cell. If master-id
filtering turns out to be reachable on this SoC -- a NOC/MSCH probe, or a `filter` format on
this PMU that is not published in its sysfs -- that is a separate and bigger finding.

### 5.3 What an LLM prefill moves end to end

System-wide over the whole process, memory reset before every arm, `-p 2048 -n 0`
[HW sweep 2026-08-28, RK1, 600 MHz, governor `performance`]. `llama-bench` runs one warm-up
plus the reps, so the token count is `(reps + 1) x 2048`. The counter covers model load and,
on the MoE arm, the expert ingest as well; the load is under 1% of either total.

**Dense, `Qwen3.5-0.8B-F16`, `-r 2`, two passes, means of the pair** (6144 tokens per arm):

| arm | read | write | total | per prefill token | t/s |
|---|---:|---:|---:|---:|---:|
| CPU only | 202736 MiB | 38485 MiB | 241221 MiB | 39.3 MiB | 68.4 |
| NPU (`GGML_BACKEND_PATH`) | 255272 | 102838 | 358110 | 58.3 MiB | 119.5 |

**The offload buys 1.75x the speed by moving 1.48x the bytes.** The write column is where the
arms differ most -- **2.7x** -- which is the host cube scatter and the readback, consistent with
§5.2's finding that the host phases carry the traffic. The CPU arm ran *longer* in wall time and
still moved less, so the difference is not a duration artifact.

**MoE, `gpt-oss-20b-mxfp4`, `-r 1 -b 2048 -ub 2048`, one process per arm** (4096 tokens per
arm). This is the pair the byte counter was wanted for -- the expert route against leaving the
experts on the CPU -- and both arms reproduce this board's published rates for those configs
(`ROCKET_MOE=0` is on record at 13.68-13.72, the placed arms at 24.79-33.03):

| arm | read | write | total | per prefill token | t/s | wall | bytes / wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ROCKET_MOE=0` (experts on the CPU) | 358387 MiB | 123631 MiB | 482018 MiB | 117.7 MiB | 13.67 | 316 s | 1.6 GB/s |
| `ROCKET_MOE=1` (experts placed) | 614269 | 451662 | 1065931 | 260.2 MiB | 33.72 | 174 s | **6.4 GB/s** |

**The expert route buys 2.47x the speed for 2.21x the bytes, and it runs the memory system
about four times harder** -- 6.4 GB/s against 1.6, where a `memcpy` on this board reaches
~13-15 GB/s and streaming reads ~17. So the placed MoE arm is the first workload measured here
that is within sight of a DRAM ceiling rather than an order below it, and the write column
(**3.65x**) is again where the difference sits. Two caveats on the magnitude, neither of which
touches the sign: it is **one process per arm**, which does not settle a percentage on this board
(§"Performance discipline"), and the `ROCKET_MOE=1` total includes the one-time int8 expert
ingest -- ~30 GB of it, about **3%**, so the per-token figure is not an ingest artifact.

Same attribution caveat as §5.2 throughout: this is the board's traffic, not the NPU's.

### 5.4 Repeating it

`perf` is not installed by default. `apt-get install linux-perf` gives **7.1.8-2** against a
**7.2.0-1** kernel and the binary runs anyway -- `perf stat` with sysfs-named events tolerates
that skew where `perf record` does not. `perf_event_paranoid` is **2** and an uncore PMU has no
per-process attribution, so every reading needs `sudo perf stat -a`; use **`sudo -E`** whenever
the command also carries `ROCKET_*` knobs or they do not survive.

## 6. How to read DDMA safely (probe design)

The hazard above (a read can hard-lock the box) shapes the safe probe. It is not carried in
the patch series, since the counters are unreadable and there is nothing to ship, but the design
is recorded here for anyone who retries it: debugfs `rocket_perf/ddma`, **read-only** (no
writes -> cannot corrupt DDMA config; note `0x8010 = RD_WEIGHT_1` is live config),
**core-0 only**, **disarmed unless** loaded with `rocket_ddma_probe=1` (so `cat` while
disarmed touches no hardware), reading the known `CFG_STATUS` first with each read in its
own `seq_printf` (so any abort is attributable to one register). Run with a hardware
watchdog / BMC reset available. The probe maps only the DDMA domain, never the
hard-locking `0x2000` page.

## 7. The RK3576 "DPU bytes-written counter" lead

A mainline-`rocket` **RK3576** bring-up (gahingwoo) reads a per-job "DPU bytes-written
counter" as a conv success metric: conv0 112×112×32 int8: **401408 B** = all 32 channels,
25088 = 2 channels. It is not a new readable register; it is the **same `dt_wr` amount
counter** §1 rules dead, confirmed three independent ways:

- **It is the BSP `dt_wr` ("data write") amount counter.** The vendor BSP sums a
  top + per-core "data write" amount at core-window offsets **0x2234** + **0x2434**, ×2
  scale. rk3576 wires both tables; **rk3588 sets both NULL**, on the same `0x22xx`/`0x24xx`
  family as §1.
- **Even on RK3576 the counter is behavioral**, a BSP/register-RE artifact of the amount
  block, not a separately documented register. So there is no documented "separate readable"
  counter to port to RK3588.
- **On RK3588 it lands in the unmapped, hard-locking page.** 0x2234/0x2434 relative to a
  core's window -> phys `0xfdab2234`/`0xfdab4234`, the `0x2000`-gap pages `rocket` never
  ioremaps (it maps only `pc`/`cna`/`core`): §3's read-fatal region.

No **separate** output-write counter hides in a *mapped* domain either: the only WDMA
registers in the DPU block are `WDMA_SIZE_0/1` (0x4058/0x405C) = output-**shape** config
(channel/height/width the host writes), not a write-back count (Mesa `registers.xml`; our
`npu_hw.h`). This is NVDLA by construction: the SDP (== our DPU) register group's
only perf registers are stall / saturation *events* (`D_PERF_WDMA_WRITE_STALL`,
`D_PERF_OUT_SATURATION`), never byte/element counts; NVDLA byte accounting lives in the
separate amount block (rocket's unmapped `0x2xxx`), not the SDP group.

**The lead does not reopen the negative.** No safe, `rocket`-mapped, per-op output-bytes
counter exists on RK3588; the analytical bytes-moved model (§5) remains the route. No probe
was run: the only candidate register is the known hard-locking page, and reading the mapped
WDMA/DPU registers returns the programmed output *shape*, not traffic.

## References

- `rocket`: `drivers/accel/rocket/rocket_{drv,core}.c`, `rocket_registers.h`; DT
  `rk3588-base.dtsi` (per-core pc/cna/core)
- Register map: Mesa `rocket/registers.xml` (`DDMA` domain @ 0x8000), our `npu_hw.h`
- RK3576 lead (§7): gahingwoo's mainline-`rocket` RK3576 bring-up
  (`https://www.reddit.com/r/embedded/comments/1ub5npg/`)
- Second witness (§4): poad42/opennpu_rk3588 `docs/ref/NPU_REGISTER_INVESTIGATION.md`, vendor
  `rknpu` on a 6.1 BSP kernel, so a different driver and a different power-management path;
  see [SOURCES.md](../SOURCES.md)
