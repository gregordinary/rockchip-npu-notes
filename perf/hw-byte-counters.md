# RK3588 NPU hardware DMA byte counters

The RK3588 NPU exposes no usable hardware DMA byte counters (weight-read, data-read and
data-write bytes) through the mainline `rocket` driver. The vendor's per-core "amount"
counter offsets are undecoded on the RK3588, and **reading them raises a bus abort that
hard-locks the SoC**. The legacy amount offsets alias DDMA reserved space and read `0`
regardless of traffic.

The system-level route works: `rockchip_ddr` is a live `perf` PMU on the RK1, calibrated
here to within 0.4% against a known number of bytes. What it reads is in §5. It has no
master-id attribution, so it measures the board rather than the NPU.

The purpose of a byte counter is to turn DMA-traffic levers (CBUF operand reuse, resident
weights, quantized readback) from wall-time inference into bytes moved. One such question,
whether `DATA_REUSE` cuts traffic, has a hardware answer without one: a −21% `wait` drop.
The counters would have been corroboration.

## 1. The amount counters in the IP family and in the rk3588 config

The NVDLA-derived NPU has DMA "amount" counters. The vendor BSP reads them on sibling SoCs
and disables them on the RK3588. Its rk3588 config sets the top/core amount tables to NULL,
and the read helper returns early with "not supported on this device". The offset tables
exist, assigned only to the rk3576, rv1126b and rk356x configs:

| set | clr_all | dt_wr | dt_rd | wt_rd | used by |
|---|---|---|---|---|---|
| `rknpu_top_amount`  | 0x2210 | 0x2234 | 0x2238 | 0x223c | rk3576, rv1126b |
| `rknpu_core_amount` | 0x2410 | 0x2434 | 0x2438 | 0x243c | rk3576 |
| `rknpu_old_top_amount` | 0x8010 | 0x8034 | 0x8038 | 0x803c | rk356x, rv1106, rk3562 |

In the rk3588 config, `pc_data_amount_scale = 2` (raw reads ×2). The clear sequence (rk3588
`pc_dma_ctrl=0`) is `WRITE(0x80000101, clr); WRITE(0x00000101, clr)`. All offsets are
relative to a core's MMIO window (the BSP reads them from `base[0]`).

Do not confuse `0x14 PC_REGISTER_AMOUNTS` with a byte counter. It is the regcmd fetch length
that the driver writes.

## 2. Address mapping of the amount offsets

The BSP maps each RK3588 core as one 64 KB window: core 0 `0xfdab0000`, core 1
`0xfdac0000`, core 2 `0xfdad0000` (`rk3588s.dtsi`). So BSP `base[0]+0x2234` = phys
`0xfdab2234`.

`rocket` instead maps three named sub-resources per core (`rk3588-base.dtsi`,
`rocket_core.c`): `pc @ 0xfdab0000`, `cna @ 0xfdab1000`, `core @ 0xfdab3000` (each
0x1000). Its register map (`rocket_registers.h`, and Mesa `registers.xml`) has 10
domains:

| domain | MMIO base |
|---|---|
| PC | `0x0000` |
| CNA | `0x1000` |
| CORE | `0x3000` |
| DPU | `0x4000` |
| DPU_RDMA | `0x5000` |
| PPU | `0x6000` |
| PPU_RDMA | `0x7000` |
| DDMA | `0x8000` |
| SDMA | `0x9000` |
| GLOBAL | `0xa000` |

**The entire map has zero registers in the `0x2xxx` range.** The `0x22xx`/`0x24xx` amount
block sits in a gap between CNA (`0x1xxx`) and CORE (`0x3000`). No domain covers that page,
and `rocket` does not map it. The DDMA block at `0x8000`, by contrast, is a defined, mapped
domain.

## 3. A read of the `0x2000` amount page

**Reading the `0x2000` amount page hard-locks the SoC.** A debugfs probe `ioremap`'d core
0's `pc_base + 0x2000` page and read `0x2234` etc. (domains powered,
`pm_runtime`-guarded). It behaved as follows:

- The clear write to `0x2210` survived (Device-nGnRE writes are early-acked).
- The read of the amount offsets hard-locked the SoC: no output, requiring a cold power
  cycle.

This is the signature of an unclaimed or undecoded MMIO read. Nothing decodes the `0x2000`
page, so the read raises a synchronous external abort that wedges the box. A write posts
and survives, and a read must return data and cannot. The abort is consistent with
Rockchip's `amount_top = NULL` on the RK3588, and is the reason for it.

> Caveat: "absent in silicon" is the leading explanation, and it is not strictly proven. A
> hard decode abort argues for absent over merely power/clock-gated, because a gated
> register typically reads 0 rather than aborting. Either way the offsets are unusable and
> unsafe to read via `rocket`.

## 4. A read of the `0x8000` DDMA block

A read-only, disarmed-by-default probe of the DDMA domain (`pc_base + 0x8000`, phys
`0xfdab8000`) returned cleanly. That confirms the abort theory: a mapped domain decodes,
and the `0x2000` gap does not. The read is of core 0 with the NPU idle:

```
0x8030 CFG_STATUS      = 0x00000100   # IDEL (bit 8) = 1  -> DDMA idle
0x8000 CFG_OUTSTANDING = 0x00000fff   # WR_OS_CNT=0x0f, RD_OS_CNT=0xff (outstanding limits)
0x8004 RD_WEIGHT_0     = 0x01010101   # arbitration weights: PDP/DPU/KERNEL/FEATURE
0x8008 WR_WEIGHT_0     = 0x00010101   # WR weights: PDP/DPU
0x800c CFG_ID_ERROR    = 0x00000000   # no DMA ID errors
0x8010 RD_WEIGHT_1     = 0x00000101   # RD weight: PC
0x8034/38/3c (legacy DT_WR/DT_RD/WT_RD) = 0x00000000
```

These are structured values that match the Mesa DDMA bitfield definitions. The `CFG_STATUS`
value `IDEL = 1` is also semantically correct, because the NPU was idle at read time. So the
probe read real registers, not a floating bus. Every value was identical before and after a
320-job `512×3840×4096` matmul (real DMA traffic, ~31 MB weights × refetch). That includes
`0x8034/38/3c` still at `0` and `CFG_STATUS` back at idle.

So on the RK3588 the legacy `0x80xx` amount offsets are DDMA reserved space, not counters.
The register map does not define them, and they read 0 regardless of traffic. The DDMA
block itself exposes only configuration (outstanding limits, arbitration weights) and a
coarse `IDEL` status bit, with no bytes-moved counter.

### A second witness on both pages

An independent RK3588 probe reports the opposite severities on both pages. It is
poad42/opennpu_rk3588 (`docs/ref/NPU_REGISTER_INVESTIGATION.md`): a kernel module using
`ioremap(0xfdab0000)`, with vendor `rknpu` on a 6.1 BSP kernel. On that probe,
`0x2210`-`0x223c` raises a contained kernel oops (DECERR, board survives), where the probe
above hard-locked. And `0x8000`-`0x803c` hangs the bus until the watchdog reboots, where the
probe above read it cleanly.

The `0x8000` half is the tractable one, because both probes resolve to the same physical
address. That address is `0xfdab8000`: their BSP 64 KB window at offset 0x8000, and this
probe's `rocket` DDMA domain. This probe's read is also validated semantically rather than
merely non-fatal. It returned structured values matching the Mesa DDMA bitfields,
`CFG_STATUS.IDEL` correctly reading idle, and every field unchanged across a 320-job
`512×3840×4096` matmul. A floating bus does not produce that.

The variable most likely to separate the two is the NPU domain's power/clock state at read
time. The probe here is `pm_runtime`-guarded with domains powered. A clock-gated AXI slave
stalls a read forever rather than returning data. That is the exact signature they describe
[hypothesis: the reconciliation has not been tested by re-running either probe against the
other's power state].

Two things follow for anyone retrying this. First, power state is part of the probe design,
and §6's guard exists for it. Second, the disagreement strengthens §3's caveat rather than
weakening it. An independent DECERR from a different kernel on the `0x2xxx` page is further
evidence that the page is undecoded rather than power-gated. A gated slave hangs where an
undecoded one aborts.

**Until someone re-runs it with power state controlled, treat both pages as unsafe on either
path.** The cheapest wrong guess costs a cold power cycle.

## 5. Conclusion, and the system-level route that answers it

The three results:

- **No HW DMA byte counters via `rocket` on the RK3588.** The real `0x22xx`/`0x24xx`
  counters are undecoded and fatal to read. The legacy `0x80xx` offsets are reserved and
  static.
- **Side result:** the DDMA control/status block is safely readable. `CFG_STATUS.IDEL` is a
  coarse "DDMA idle" signal, and that is not byte accounting.
- **Bytes-moved ground truth comes from outside the NPU register space, and it is
  reachable.** The kernel exposes the DDR controller's PMU as an ordinary `perf` uncore
  PMU, `rockchip_ddr`, which avoids the unmapped-MMIO hazard entirely. Its cost is
  attribution: it counts the board, not a master.

### 5.1 The `rockchip_ddr` PMU and its calibration

On the RK1, kernel 7.2.0-1-arm64, `/sys/bus/event_source/devices/rockchip_ddr/` publishes
these events, each with a `.scale` of `2^-20` and a `.unit` of MB:

- `bytes`, `read-bytes` and `write-bytes`
- The same three per channel (`read-bytes0..3` / `write-bytes0..3`)
- `cycles`

The `cycles` event tracks wall time at the memory clock exactly (2.112e10 over 10 s =
2112 MHz). That says the PMU is running. It does not say the byte events mean bytes.

The byte events do mean bytes, to within 0.4% [HW sweep 2026-08-28, RK1]. The test
`tests/ddr_pmu_cal.c` moves a known number of bytes past every cache, and
`tools/ddr-pmu-cal.sh` differences two pass counts. Both arms carry the same allocation,
page faults, kernel page zeroing and idle floor, so those cancel:

| known traffic | counted | ratio | the other column |
|---|---|---|---|
| 16384 MiB read (2 GiB buffer, 1 vs 9 read passes, 3 reps) | 16447.4 MiB | 1.0038 | +0.5 MiB written |
| 16384 MiB written (1 vs 9 write passes, 3 reps) | 16372.8 MiB | 0.9993 | +41.5 MiB read, 0.25% |

The idle floor is 4.8-7.3 MB/s over 10 s, three orders below any figure that follows. The
near-zero cross-columns are a second check. A read pass charges nothing to writes, and a
write pass charges 0.25% to reads. So the A76's write-streaming mode is engaging, and the
columns mean what they are named.

**The differential is required.** The same tool's single-pass arm counts 4.3 GiB read where
only 2 GiB is known. The surplus is the fault and allocation path, which no model of the
workload contains. An absolute system-wide count is not a measurement of a workload. A
difference between two arms of it is.

### 5.2 What one tiled matmul moves, against the analytical model

`tests/bytes_moved_rocket.c` predicts DRAM traffic analytically, and the PMU checks that
prediction. The test `tests/ddr_mm_bytes.c` runs the single-fd streaming path, which is the
one the model describes, with no CPU reference. It runs at 5 and 45 reps, 3 reps of each, so
the difference is 40 matmuls and nothing else. The shape is `512×3840×4096` fp16 with KACC
and data-reuse on, at 600 MHz [HW sweep 2026-08-28, RK1]:

| arm | model total | measured read | measured write | measured total |
|---|---:|---:|---:|---:|
| default (KACC + data reuse) | 133.50 MB | 287.30 MiB | 188.65 MiB | 475.95 MiB |
| `ROCKET_REUSE=0` | 249.75 MB | 408.29 | 186.72 | 595.00 |
| `ROCKET_KACC=0` | 161.50 MB | 476.26 | 209.03 | 685.29 |
| prepacked (same arithmetic, weight scatter hoisted) | 103.50 MB | 152.38 | 74.22 | 226.61 |

The table gives three separate results, and they do not all point the same way.

#### The data-reuse term

The measurement confirms the data-reuse term. The model says that turning `data_reuse` off
multiplies the feature DMA by `nNt`=32: `+116.25 MB` of reads and nothing on the write side.
The measured change is +120.98 MiB read and -1.93 MiB write. That is 4% high on the term,
and zero on the column the model says does not move. It validates the model's tiling and
loop order (the feature tile is reused across the `N` loop). It also checks a traffic claim
against a counter rather than against wall time, which no earlier measurement here did.

#### The absolute level

The absolute level is 3.57x the model, and the excess is concentrated in the host scatter.
The prepacked arm hoists the weight scatter out of the loop and keeps the same arithmetic.
It removes 249.34 MiB per call against a `packB` term of 30.00 MB. So the bus charges that
phase 8.3x what counting each weight byte once charges it. It charges it in both directions
(`-134.92` read, `-114.42` write).

The model counts an operand's bytes once per logical movement. A host scatter moves useful
chunks smaller than a cache line. So the bus moves a line per chunk and moves it twice,
source in and destination out.

One term inside that difference is not separated. The entry `rocket_matmul_fp16` calls
`mm_bos_alloc` per call [source-confirmed, `src/rocket_matmul.c`]. So the streaming path
also allocates, kernel-zeroes and frees a ~30 MB weight BO every call, where the prepacked
path does it once. If the zeroing is the whole 30 MB, the scatter's own write amplification
is 2.8x rather than 3.8x. Separating them needs a pack-only arm, which does not exist.

#### The `KACC` sensitivity

The model's `KACC` sensitivity is wrong. The model predicts `+28 MB` of writes and no change
in reads when K-accumulation moves off the NPU. The measurement is +20.4 MiB write (same
order) but +189.0 MiB read, where zero was predicted. So the model's account of what the
un-accumulated partials cost is incomplete on the read side. That account underwrites the
`int8` readback floor, `nKt * M * N * 4`, which is therefore a lower bound, not the figure.

#### Scope of the comparison

`rockchip_ddr` has no master-id filter exposed on this SoC, so every figure is the whole
board's traffic during the run. The differential removes the fixed load. It cannot attribute
what remains between the NPU's DMA and the host packing that the same arm causes. That pair
happens to be exactly what the bytes-moved model is about. So the figure is the right total
for this question, and the wrong one for "how much did the NPU's DMA move".

A DRAM counter also cannot see traffic served from cache, so every number is a lower bound
on data touched. And this is one shape, one datatype, `n`=3 per cell. If master-id filtering
turns out to be reachable on this SoC, that is a separate and bigger finding. The candidate
routes are a NOC/MSCH probe, or a `filter` format on this PMU that its sysfs does not
publish.

### 5.3 What an LLM prefill moves end to end

The counts are system-wide over the whole process, with a memory reset before every arm and
`-p 2048 -n 0` [HW sweep 2026-08-28, RK1, 600 MHz, governor `performance`]. Because
`llama-bench` runs one warm-up plus the reps, the token count is `(reps + 1) x 2048`. The
counter covers model load and, on the MoE arm, the expert ingest as well. The load is under
1% of either total.

#### Dense: `Qwen3.5-0.8B-F16`

The run is `-r 2`, two passes, with the means of the pair (6144 tokens per arm):

| arm | read | write | total | per prefill token | t/s |
|---|---:|---:|---:|---:|---:|
| CPU only | 202736 MiB | 38485 MiB | 241221 MiB | 39.3 MiB | 68.4 |
| NPU (`GGML_BACKEND_PATH`) | 255272 | 102838 | 358110 | 58.3 MiB | 119.5 |

The offload buys 1.75x the speed by moving 1.48x the bytes. The write column is where the
arms differ most, at 2.7x. That is the host cube scatter and the readback, consistent with
§5.2's finding that the host phases carry the traffic. The CPU arm ran longer in wall time
and still moved less, so the difference is not a duration artifact.

#### MoE: `gpt-oss-20b-mxfp4`

The run is `-r 1 -b 2048 -ub 2048`, one process per arm (4096 tokens per arm). This pair,
the expert route against leaving the experts on the CPU, is the one the byte counter was
wanted for. Both arms reproduce this board's published rates for those configs. On record,
`ROCKET_MOE=0` is at 13.68-13.72 and the placed arms are at 24.79-33.03:

| arm | read | write | total | per prefill token | t/s | wall | bytes / wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ROCKET_MOE=0` (experts on the CPU) | 358387 MiB | 123631 MiB | 482018 MiB | 117.7 MiB | 13.67 | 316 s | 1.6 GB/s |
| `ROCKET_MOE=1` (experts placed) | 614269 | 451662 | 1065931 | 260.2 MiB | 33.72 | 174 s | 6.4 GB/s |

The expert route buys 2.47x the speed for 2.21x the bytes, and it runs the memory system
about four times harder: 6.4 GB/s against 1.6. A `memcpy` on this board reaches
~13-15 GB/s, and streaming reads reach ~17. So the placed MoE arm is the first workload
measured here that is within sight of a DRAM ceiling rather than an order below it. The
write column (3.65x) is again where the difference sits.

Two caveats apply to the magnitude, and neither touches the sign. The run is one process per
arm, which does not settle a percentage on this board (§"Performance discipline"). The
`ROCKET_MOE=1` total also includes the one-time int8 expert ingest. That ingest is ~30 GB,
about 3%, so the per-token figure is not an ingest artifact.

The attribution caveat of §5.2 applies throughout: this is the board's traffic, not the
NPU's.

### 5.4 Reproduction

`perf` is not installed by default. Installing it with `apt-get install linux-perf` gives
7.1.8-2 against a 7.2.0-1 kernel, and the binary runs anyway. With sysfs-named events,
`perf stat` tolerates that skew where `perf record` does not. Because `perf_event_paranoid`
is 2 and an uncore PMU has no per-process attribution, every reading needs
`sudo perf stat -a`. Whenever the command also carries `ROCKET_*` knobs, use **`sudo -E`**,
or the knobs do not survive.

## 6. A safe DDMA probe design

The hazard above (a read can hard-lock the box) shapes the safe probe. The patch series does
not carry the probe, since the counters are unreadable and there is nothing to ship. The
design is recorded here for anyone who retries it. The probe is debugfs `rocket_perf/ddma`,
with these properties:

- **Read-only**: with no writes, it cannot corrupt DDMA config. Note that
  `0x8010 = RD_WEIGHT_1` is live config.
- **Core 0 only**
- **Disarmed unless loaded with `rocket_ddma_probe=1`**: a `cat` while disarmed touches no
  hardware.
- **`CFG_STATUS` first**: the probe reads the known `CFG_STATUS` first, with each read in
  its own `seq_printf`, so any abort is attributable to one register.

Run it with a hardware watchdog or a BMC reset available. The probe maps only the DDMA
domain, never the hard-locking `0x2000` page.

## 7. The RK3576 "DPU bytes-written counter" lead

A mainline-`rocket` RK3576 bring-up (gahingwoo) reads a per-job "DPU bytes-written counter"
as a conv success metric. On conv0, 112×112×32 int8, 401408 B = all 32 channels and 25088 =
2 channels. That counter is the same `dt_wr` amount counter that §1 rules out, not a new
readable register. Three independent observations confirm it:

- **It is the BSP `dt_wr` ("data write") amount counter.** The vendor BSP sums a top +
  per-core "data write" amount at core-window offsets 0x2234 + 0x2434, ×2 scale. The rk3576
  config wires both tables and the rk3588 config sets both NULL, on the same
  `0x22xx`/`0x24xx` family as §1.
- **Even on the RK3576 the counter is behavioral.** It is a BSP/register-RE artifact of the
  amount block, not a separately documented register. So there is no documented "separate
  readable" counter to port to the RK3588.
- **On the RK3588 it lands in the unmapped, hard-locking page.** Relative to a core's
  window, 0x2234/0x2434 are phys `0xfdab2234`/`0xfdab4234`. Those are the `0x2000`-gap pages
  that `rocket` never ioremaps (it maps only `pc`/`cna`/`core`), which is §3's read-fatal
  region.

No separate output-write counter exists in a mapped domain either. The only WDMA registers
in the DPU block are `WDMA_SIZE_0/1` (0x4058/0x405C). They are output-shape config, the
channel/height/width that the host writes, and not a write-back count (Mesa `registers.xml`,
`librocketnpu`'s `npu_hw.h`).

This follows NVDLA's structure. The only perf registers of the SDP register group (this IP's
DPU) are stall / saturation events (`D_PERF_WDMA_WRITE_STALL`, `D_PERF_OUT_SATURATION`),
never byte/element counts. NVDLA byte accounting lives in the separate amount block
(`rocket`'s unmapped `0x2xxx`), not in the SDP group.

The lead does not reopen the negative. No safe, `rocket`-mapped, per-op output-bytes counter
exists on the RK3588, and the analytical bytes-moved model (§5) remains the route. No probe
was run. The only candidate register is the known hard-locking page. Reading the mapped
WDMA/DPU registers returns the programmed output shape, not traffic.

## References

- `rocket`: `drivers/accel/rocket/rocket_{drv,core}.c`, `rocket_registers.h`, and the DT
  `rk3588-base.dtsi` (per-core pc/cna/core)
- Register map: Mesa `rocket/registers.xml` (`DDMA` domain @ 0x8000), and `librocketnpu`'s
  `npu_hw.h`
- RK3576 lead (§7): gahingwoo's mainline-`rocket` RK3576 bring-up
  (`https://www.reddit.com/r/embedded/comments/1ub5npg/`)
- Second witness (§4): poad42/opennpu_rk3588 `docs/ref/NPU_REGISTER_INVESTIGATION.md`.
  It runs vendor `rknpu` on a 6.1 BSP kernel, so a
  different driver and a different power-management path. See [SOURCES.md](../SOURCES.md).
