# RK3588 NPU hardware DMA byte counters

The RK3588 NPU exposes no usable hardware DMA byte counters (weight-read, data-read and
data-write bytes) through the mainline `rocket` driver. The vendor's per-core "amount" counter
offsets are undecoded on the RK3588. The legacy amount offsets alias DDMA reserved space, and
read `0` regardless of traffic. The bytes a workload moves are measurable at the DDR controller
instead. Its `perf` PMU, `rockchip_ddr`, is live on the RK1 and calibrated to within 0.4%
against a known number of bytes. It has no master-id attribution, so it measures the board
rather than the NPU. §5 gives its calibration and what it measures.

**Conditions:** RK3588 (RK1) under the mainline `rocket` driver, with the register reads on
core 0.

**Not characterized:** a read of the `0x2000` or the `0x8000` page with the NPU's power state
controlled, and master-id filtering on `rockchip_ddr` (through a NOC/MSCH probe, or a `filter` format that
its sysfs does not publish).

| Source | Location | Result on the RK3588 |
|---|---|---|
| Amount counters | `0x22xx`/`0x24xx` in a core's window | Undecoded. A read hard-locks the SoC (§3) |
| Legacy amount offsets | `0x8034/38/3c` | DDMA reserved space, static at `0` (§4) |
| DDMA control and status | `0x8000` domain | Safely readable. Configuration and a coarse "DDMA idle" bit, `CFG_STATUS.IDEL`, with no byte accounting (§4) |
| `rockchip_ddr` | DDR controller PMU | Bytes to within 0.4%, for the whole board (§5) |

> [!CAUTION]
> A read of the amount offsets raises a bus abort that hard-locks the SoC, and only a cold power
> cycle recovers it. An independent probe under the vendor `rknpu` driver hangs the bus on the
> `0x80xx` offsets instead (§4). Treat both pages as unsafe to read under either driver.

## 1. The amount counters in the RKNPU family and in the rk3588 config

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

> [!NOTE]
> `0x14 PC_REGISTER_AMOUNTS` is the regcmd fetch length that the driver writes, and does not
> count bytes.

## 2. Address mapping of the amount offsets

The BSP maps each RK3588 core as one 64 KB window: core 0 `0xfdab0000`, core 1
`0xfdac0000`, core 2 `0xfdad0000` (`rk3588s.dtsi`). BSP `base[0]+0x2234` is therefore phys
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

The map defines no register in the `0x2xxx` range. The `0x22xx`/`0x24xx` amount block sits in
a gap between CNA (`0x1xxx`) and CORE (`0x3000`). No domain covers that page, and `rocket`
does not map it. The DDMA block at `0x8000`, by contrast, is a defined, mapped domain.

## 3. A read of the `0x2000` amount page

A read of the `0x2000` amount page hard-locks the SoC, and a write to it completes.

**Conditions:** core 0's `pc_base + 0x2000` page mapped with `ioremap` from a debugfs entry,
reading `0x2234` and its neighbors, with the domains powered and the access
`pm_runtime`-guarded.

The page behaves as follows:

- The clear write to `0x2210` completes (Device-nGnRE writes are early-acked).
- The read of the amount offsets hard-locks the SoC: no output, and recovery takes a cold
  power cycle.

The behavior matches an unclaimed or undecoded MMIO read. Nothing decodes the `0x2000` page, so
the read raises a synchronous external abort that hangs the system. A write is posted and
completes, while a read must return data and cannot. The abort is consistent with Rockchip's
`amount_top = NULL` on the RK3588, and is the reason for it.

The likely cause is that the counters are absent in silicon, rather than power- or clock-gated
[hypothesis]. A decode abort argues for absent over gated, because a gated slave stalls a read
rather than returning a decode error (§4). Either way the offsets are unusable and unsafe to
read through `rocket`.

## 4. A read of the `0x8000` DDMA block

The DDMA domain (`pc_base + 0x8000`, phys `0xfdab8000`) reads cleanly. A mapped domain
decodes, and the `0x2000` gap does not, which confirms the abort explanation in §3.

**Conditions:** core 0, NPU idle, a read-only probe that is disarmed by default (§6).

```
0x8030 CFG_STATUS      = 0x00000100   # IDEL (bit 8) = 1  -> DDMA idle
0x8000 CFG_OUTSTANDING = 0x00000fff   # WR_OS_CNT=0x0f, RD_OS_CNT=0xff (outstanding limits)
0x8004 RD_WEIGHT_0     = 0x01010101   # arbitration weights: PDP/DPU/KERNEL/FEATURE
0x8008 WR_WEIGHT_0     = 0x00010101   # WR weights: PDP/DPU
0x800c CFG_ID_ERROR    = 0x00000000   # no DMA ID errors
0x8010 RD_WEIGHT_1     = 0x00000101   # RD weight: PC
0x8034/38/3c (legacy DT_WR/DT_RD/WT_RD) = 0x00000000
```

Every value is identical before and after a `512×3840×4096` matmul of 320 tasks (real DMA traffic,
~31 MB weights × refetch), with `0x8034/38/3c` still at `0` and `CFG_STATUS` back at idle.

On the RK3588, the legacy `0x80xx` amount offsets are therefore DDMA reserved space. The
register map does not define them, and they read 0 regardless of traffic. The DDMA block itself
exposes only configuration (outstanding limits, arbitration weights) and a coarse `IDEL` status
bit, with no bytes-moved counter.

### An independent probe under the vendor driver

An independent RK3588 probe reports the opposite severity on each page. It is
poad42/opennpu_rk3588 (`docs/ref/NPU_REGISTER_INVESTIGATION.md`): a kernel module using
`ioremap(0xfdab0000)`, with vendor `rknpu` on a 6.1 BSP kernel.

| Offsets | §3 and §4, mainline `rocket` | poad42/opennpu_rk3588, vendor `rknpu` |
|---|---|---|
| `0x2210`-`0x223c` | Hard-locks the SoC | A contained kernel oops (DECERR), and the board survives |
| `0x8000`-`0x803c` | Reads cleanly | The bus hangs until the watchdog reboots |

On the `0x8000` page both probes reach the same physical address, `0xfdab8000`: their BSP 64 KB
window at offset 0x8000, and the `rocket` DDMA domain. The §4 read returned register values,
and not a floating bus (§"Evidence").

The likely difference is the NPU domain's power and clock state at read time [hypothesis].
The §3 and §4 reads are `pm_runtime`-guarded, with the domains powered. A clock-gated AXI slave
stalls a read forever rather than returning data, which is the hang the vendor-driver probe
reports. Power state is therefore part of a probe's design, and §6's guard exists for it.

The vendor-driver DECERR on the `0x2xxx` page, from a different kernel, supports §3's
hypothesis that the page is undecoded rather than power-gated. A gated slave hangs where an
undecoded one aborts.

## 5. The DDR controller PMU

The kernel exposes the DDR controller's PMU as an ordinary `perf` uncore PMU, `rockchip_ddr`.
Reading it touches no NPU register, so it avoids the unmapped-MMIO hazard entirely. It counts
the board, not a master. A byte count turns a DMA-traffic optimization (CBUF operand reuse,
resident weights, quantized readback) from an inference from wall-clock time into a count of
bytes moved.

### 5.1 The `rockchip_ddr` PMU and its calibration

On the RK1, kernel 7.2.0-1-arm64, `/sys/bus/event_source/devices/rockchip_ddr/` publishes
these events, each with a `.scale` of `2^-20` and a `.unit` of MB:

- `bytes`, `read-bytes` and `write-bytes`
- The same three per channel (`read-bytes0..3` / `write-bytes0..3`)
- `cycles`

The `cycles` event tracks wall-clock time at the memory clock exactly (2.112e10 over 10 s =
2112 MHz). The byte events count bytes to within 0.4% [HW sweep 2026-08-28, RK1]. The
calibration moves a known number of bytes past every cache, and takes the difference between
two pass counts:

| known traffic | counted | ratio | the other column |
|---|---|---|---|
| 16384 MiB read (2 GiB buffer, 1 vs 9 read passes, 3 reps) | 16447.4 MiB | 1.0038 | +0.5 MiB written |
| 16384 MiB written (1 vs 9 write passes, 3 reps) | 16372.8 MiB | 0.9993 | +41.5 MiB read, 0.25% |

The idle background is 4.8-7.3 MB/s over 10 s, three orders of magnitude below any figure that
follows.

> [!NOTE]
> An absolute `rockchip_ddr` count includes the process's page faults and allocation, which no
> model of the workload contains. A single pass that reads a known 2 GiB counts 4.3 GiB read.
> Measure a workload as the difference between two runs that differ only in the workload.

### 5.2 Bytes moved by a tiled matmul, against the analytical model

The analytical bytes-moved model ([not-mac-bound.md](not-mac-bound.md) §"Bytes-moved model")
predicts the DRAM traffic of a tiled matmul. The PMU measures it on the single-fd streaming
path, which is the path the model describes, with no CPU reference [HW sweep 2026-08-28, RK1].

**Conditions:** `512×3840×4096` fp16, KACC and data reuse on, the planner's default tiling for
the shape (Mt 256, Kt 512, Nt 128: `nMt`=2, `nKt`=8, `nNt`=32), NPU clock 600 MHz. Each figure
is the difference between 45 and 5 repetitions, which is 40 matmuls and nothing else.

| configuration | model total | measured read | measured write | measured total |
|---|---:|---:|---:|---:|
| default (KACC + data reuse) | 133.50 MB | 287.30 MiB | 188.65 MiB | 475.95 MiB |
| `ROCKET_REUSE=0` | 249.75 MB | 408.29 | 186.72 | 595.00 |
| `ROCKET_KACC=0` | 161.50 MB | 476.26 | 209.03 | 685.29 |
| prepacked (same arithmetic, weight scatter hoisted) | 103.50 MB | 152.38 | 74.22 | 226.61 |

#### The data-reuse term

The model's data-reuse term matches the counter. In the model, turning `data_reuse` off
multiplies the feature DMA by `nNt`=32: `+116.25 MB` of reads and nothing on the write side.
The measured change is +120.98 MiB read and -1.93 MiB write. The measurement is 4% high on the
term, and zero on the column the model says does not move. The model's tiling and loop order
are therefore right: the feature tile is reused across the `N` loop. The counter confirms in
bytes the reduction that `DATA_REUSE`'s −21% `wait` drop shows in time
([../encodings/cbuf-reuse.md](../encodings/cbuf-reuse.md)).

#### The absolute level

The absolute level is 3.57x the model, and the excess is concentrated in the host scatter.
The prepacked configuration hoists the weight scatter out of the loop and keeps the same
arithmetic. It removes 249.34 MiB per call against a `packB` term of 30.00 MB. The bus
therefore moves 8.3x the bytes for that phase that the model counts, once per weight byte. The
excess is in both directions (`-134.92` read, `-114.42` write).

The model counts an operand's bytes once per logical movement. A host scatter moves useful
chunks smaller than a cache line. The bus moves a whole line per chunk, and moves it twice:
source in and destination out.

One term inside that difference is not separated. The entry `rocket_matmul_fp16` calls
`mm_bos_alloc` per call [source-confirmed, `src/rocket_matmul.c`]. The streaming path
therefore also allocates, kernel-zeroes and frees a ~30 MB weight BO every call, where the
prepacked path does it once. If the zeroing is the whole 30 MB, the scatter's own write
amplification is 2.8x rather than 3.8x.

#### The `KACC` sensitivity

The model's `KACC` sensitivity is wrong. With K-accumulation off the NPU, the model predicts
`+28 MB` of writes and no change in reads. The measurement is +20.4 MiB write (the same order)
and +189.0 MiB read. The model's account of what the un-accumulated partials cost is therefore
incomplete on the read side. That account underlies the `int8` readback cost,
`nKt * M * N * 4`, which is a lower bound.

#### Attribution

`rockchip_ddr` has no master-id filter exposed on this SoC, so every figure is the whole
board's traffic during the run. The difference between two runs removes the fixed load. It
cannot divide what remains between the NPU's DMA and the host packing that the same
configuration causes. The bytes-moved model covers exactly that pair. The figure is
therefore the right total for checking the model, and the wrong one for "how much did the NPU's
DMA move".

A DRAM counter does not see traffic served from cache, so every number is a lower bound on
data touched.

### 5.3 Bytes moved by an LLM prefill

The counts are system-wide over the whole `llama-bench` process [HW sweep 2026-08-28, RK1].

**Conditions:** NPU clock 600 MHz, CPU governor `performance`, `-p 2048 -n 0`, with a memory
reset before every configuration. `llama-bench` runs one warm-up plus the repetitions, so the
token count is `(reps + 1) x 2048`. The counter covers model load and, in the MoE
configuration, the expert ingest as well. The load is under 1% of either total.

#### Dense: `Qwen3.5-0.8B-F16`

**Conditions:** `-r 2`, two passes, with the means of the pair (6144 tokens per configuration).

| configuration | read | write | total | per prefill token | t/s |
|---|---:|---:|---:|---:|---:|
| CPU only | 202736 MiB | 38485 MiB | 241221 MiB | 39.3 MiB | 68.4 |
| NPU (`GGML_BACKEND_PATH`) | 255272 | 102838 | 358110 | 58.3 MiB | 119.5 |

With NPU offload, prefill is 1.75x faster and moves 1.48x the DRAM bytes. The configurations
differ most in the write column, at 2.7x. The extra writes are the host cube scatter and the
readback, consistent with §5.2's finding that the host phases account for the traffic. The
CPU-only configuration ran longer and still moved less, so the difference is not an effect of
duration.

#### MoE: `gpt-oss-20b-mxfp4`

**Conditions:** `-r 1 -b 2048 -ub 2048`, one process per configuration (4096 tokens per
configuration).

| configuration | read | write | total | per prefill token | t/s | wall-clock | bytes / wall-clock |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ROCKET_MOE=0` (experts on the CPU) | 358387 MiB | 123631 MiB | 482018 MiB | 117.7 MiB | 13.67 | 316 s | 1.6 GB/s |
| `ROCKET_MOE=1` (experts placed) | 614269 | 451662 | 1065931 | 260.2 MiB | 33.72 | 174 s | 6.4 GB/s |

With the experts placed on the NPU, prefill is 2.47x faster and moves 2.21x the bytes. It
runs the memory system about four times harder: 6.4 GB/s against 1.6. A `memcpy` on this board
reaches ~13-15 GB/s, and streaming reads reach ~17. The placed MoE configuration is the first
workload measured here that approaches a DRAM bandwidth limit. The others run an order of
magnitude below it. The write column (3.65x) again holds most of the difference.

The `ROCKET_MOE=1` total includes the one-time int8 expert ingest. That ingest is ~30 GB,
about 3%, so the per-token figure is not an ingest artifact.

As in §5.2, these counts are the board's traffic, not the NPU's.

### 5.4 Reproduction

`perf` is not installed by default. Installing it with `apt-get install linux-perf` gives
7.1.8-2 against a 7.2.0-1 kernel, and the binary runs anyway. With sysfs-named events,
`perf stat` tolerates that skew where `perf record` does not. Because `perf_event_paranoid`
is 2 and an uncore PMU has no per-process attribution, every reading needs
`sudo perf stat -a`. If the command also sets `ROCKET_*` variables, run it under `sudo -E`,
because plain `sudo` drops them.

## 6. A safe DDMA probe design

A read can hard-lock the system, so a probe of the DDMA domain takes the design below. The
`rocket` patch series does not include the probe, because the counters are unreadable. The
probe is debugfs `rocket_perf/ddma`, with these properties:

- **Read-only**: with no writes, it cannot corrupt DDMA config. Note that
  `0x8010 = RD_WEIGHT_1` is live config.
- **Core 0 only**
- **Disarmed unless loaded with `rocket_ddma_probe=1`**: a `cat` while disarmed touches no
  hardware.
- **`CFG_STATUS` first**: the probe reads the known `CFG_STATUS` first, with each read in
  its own `seq_printf`, so any abort is attributable to one register.

Run it with a hardware watchdog or a BMC reset available. The probe maps only the DDMA
domain, never the hard-locking `0x2000` page.

## 7. The RK3576 "DPU bytes-written counter"

A mainline-`rocket` RK3576 bring-up (gahingwoo) reads a per-job "DPU bytes-written counter"
as a measure of convolution success. On conv0, 112×112×32 int8, 401408 B = all 32 channels and
25088 = 2 channels. The counter is the same `dt_wr` amount counter as §1's, and not a separate
readable register. Three independent observations confirm it:

- **It is the BSP `dt_wr` ("data write") amount counter.** The vendor BSP sums a top +
  per-core "data write" amount at core-window offsets 0x2234 + 0x2434, ×2 scale. The rk3576
  config wires both tables and the rk3588 config sets both NULL, on the same
  `0x22xx`/`0x24xx` family as §1.
- **Even on the RK3576 the counter is behavioral.** It is a BSP/register-RE artifact of the
  amount block, not a separately documented register. No documented "separate readable"
  counter exists to port to the RK3588.
- **On the RK3588 its offsets are in the unmapped, hard-locking page.** Relative to a core's
  window, 0x2234/0x2434 are phys `0xfdab2234`/`0xfdab2434`. Those are the `0x2000`-gap pages
  that `rocket` never ioremaps (it maps only `pc`/`cna`/`core`), which is §3's read-fatal
  region.

No separate output-write counter exists in a mapped domain either. The only WDMA registers
in the DPU block are `WDMA_SIZE_0/1` (0x4058/0x405C). They are output-shape config, the
channel/height/width that the host writes, and not a write-back count (Mesa `registers.xml`,
`librocketnpu`'s `npu_hw.h`).

NVDLA has the same structure. The only perf registers of the SDP register group (the RKNPU's
DPU) are stall and saturation events (`D_PERF_WDMA_WRITE_STALL`, `D_PERF_OUT_SATURATION`),
never byte or element counts. NVDLA byte accounting is in the separate amount block
(`rocket`'s unmapped `0x2xxx`), not in the SDP group.

The RK3588 has no safe, `rocket`-mapped, per-op output-bytes counter, and the `rockchip_ddr`
PMU (§5) is the way to measure bytes moved.

## References

- `rocket`: `drivers/accel/rocket/rocket_{drv,core}.c`, `rocket_registers.h`, and the DT
  `rk3588-base.dtsi` (per-core pc/cna/core)
- Register map: Mesa `rocket/registers.xml` (`DDMA` domain @ 0x8000), and `librocketnpu`'s
  `npu_hw.h`
- RK3576 counter (§7): gahingwoo's mainline-`rocket` RK3576 bring-up
  (`https://www.reddit.com/r/embedded/comments/1ub5npg/`)
- Independent probe (§4): poad42/opennpu_rk3588, read at `983282e`,
  `docs/ref/NPU_REGISTER_INVESTIGATION.md`. It runs vendor `rknpu` on a 6.1 BSP kernel, so a
  different driver and a different power-management path. See [SOURCES.md](../SOURCES.md).

## Evidence

### A read of the `0x8000` DDMA block

The values are structured and match the Mesa DDMA bitfield definitions. The `CFG_STATUS` value
`IDEL = 1` is also semantically correct, because the NPU was idle at read time. The probe
therefore read real registers. A floating bus does not produce structured values, a correct
`CFG_STATUS.IDEL`, and every field unchanged across a 320-task `512×3840×4096` matmul.

### The `rockchip_ddr` PMU and its calibration

The calibration test `tests/ddr_pmu_cal.c` moves a known number of bytes past every cache, and
`tools/ddr-pmu-cal.sh` differences two pass counts. Both configurations have the same
allocation, page faults, kernel page zeroing and idle background, so those cancel.

The near-zero other columns are a second check. A read pass adds nothing to writes, and a write
pass adds 0.25% to reads. The A76's write-streaming mode is therefore engaging, and each column
counts the direction it is named for.

### Bytes moved by a tiled matmul

The model is `tests/bytes_moved_rocket.c`. The measurement is `tests/ddr_mm_bytes.c`, run at 5
and 45 repetitions, three runs of each (`n`=3 per test point). It covers one shape and one
datatype.

### Bytes moved by an LLM prefill

The expert path against the experts on the CPU is the comparison the byte counter was wanted
for. Both MoE configurations measure within 2.1% of this board's published rates for them. On
record, `ROCKET_MOE=0` is at 13.68-13.72, and the placed configurations are at 24.79-33.03.

Two caveats apply to the MoE magnitude, and neither touches the sign. The run is one process
per configuration, which does not establish a percentage on this board. Establishing one takes
interleaved passes with the configuration order rotated, and each ratio paired within a pass. The
`ROCKET_MOE=1` total also includes the expert ingest (§5.3).

### The RK3576 "DPU bytes-written counter"

No probe was run for §7. The only candidate register is the known hard-locking page.
