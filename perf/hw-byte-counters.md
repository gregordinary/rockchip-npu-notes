# RK3588 NPU HW DMA byte counters — not available via the FOSS `rocket` path

**Verdict:** the RK3588 NPU does **not** expose usable hardware DMA *byte* counters
(weight-read / data-read / data-write bytes) through the mainline `rocket` driver. The
vendor's per-core "amount" counter offsets are **undecoded** on rk3588 — *reading* them
raises a bus abort that hard-locks the SoC. The legacy amount offsets alias DDMA
reserved space and read `0` regardless of traffic. For bytes-moved ground truth, use a
system-level PMU (DDR/DFI or NOC/MSCH) instead.

The motivation was to turn DMA-traffic
levers (CBUF operand reuse, resident weights, quantized readback) from wall-time
inference into bytes moved. (That specific question — does `DATA_REUSE` cut traffic? —
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
*writes*) with a byte counter — it is not one.

## 2. Address mapping: where the offsets land, and the `rocket` wrinkle

The BSP maps each rk3588 core as one 64 KB window — core 0 `0xfdab0000`, core 1
`0xfdac0000`, core 2 `0xfdad0000` (`rk3588s.dtsi`). So BSP `base[0]+0x2234` = phys
`0xfdab2234`.

`rocket` instead maps **three named sub-resources** per core (`rk3588-base.dtsi`,
`rocket_core.c`): `pc @ 0xfdab0000`, `cna @ 0xfdab1000`, `core @ 0xfdab3000` (each
0x1000). Its register map (`rocket_registers.h`, and Mesa `registers.xml`) has 10
domains at MMIO bases `0x0000/0x1000/0x3000/0x4000/0x5000/0x6000/0x7000/0x8000/0x9000/
0xa000` (PC/CNA/CORE/DPU/DPU_RDMA/PPU/PPU_RDMA/DDMA/SDMA/GLOBAL).

The decisive fact: **there are zero registers in the `0x2xxx` range in the entire map.**
The `0x22xx`/`0x24xx` amount block sits in a genuine gap between CNA (`0x1xxx`) and CORE
(`0x3000`) — a page no domain covers, and which `rocket` does not map. The DDMA block at
`0x8000`, by contrast, *is* a defined, mapped domain.

## 3. Result 1 — reading the `0x2000` amount page hard-locks the SoC

A debugfs probe `ioremap`'d core 0's `pc_base + 0x2000` page and read `0x2234` etc.
(domains powered, `pm_runtime`-guarded). Behavior:

- The **clear write** to `0x2210` survived (Device-nGnRE writes are early-acked).
- The **read** of the amount offsets **hard-locked the SoC** — no output, requiring a
  cold power-cycle.

This is the signature of an **unclaimed/undecoded MMIO read**: nothing decodes the
`0x2000` page, so the read raises a synchronous external abort that wedges the box (a
write posts and survives; a read must return data and cannot). Consistent with — and the
reason for — Rockchip's `amount_top = NULL` on rk3588.

> Caveat: "absent in silicon" is the leading explanation but not strictly proven; a
> hard *decode* abort argues for absent over merely power/clock-gated (a gated register
> typically reads 0 rather than aborting). Either way the offsets are unusable and
> unsafe to read via `rocket`.

## 4. Result 2 — the `0x8000` DDMA block is readable (safe probe)

A read-only, disarmed-by-default probe of the DDMA domain (`pc_base + 0x8000`, phys
`0xfdab8000`) returned cleanly — confirming the abort theory (mapped domain decodes;
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
and `CFG_STATUS` `IDEL = 1` is *semantically correct* (the NPU was idle at read time) —
proving we read real registers, not a floating bus. **Before/after a 320-job
`512×3840×4096` matmul** (real DMA traffic, ~31 MB weights × refetch), every value was
**identical**, including `0x8034/38/3c` still `0` and `CFG_STATUS` back to idle.

So the legacy `0x80xx` amount offsets are **DDMA reserved space** (not defined in the
register map) that **reads 0 regardless of traffic — not counters** on rk3588. The DDMA
block itself only exposes *configuration* (outstanding limits, arbitration weights) and a
coarse `IDEL` status bit — no bytes-moved counter.

## 5. Conclusion & the fallback

- **No HW DMA byte counters via `rocket` on rk3588.** The real `0x22xx`/`0x24xx`
  counters are undecoded (fatal to read); the legacy `0x80xx` offsets are reserved and
  static.
- **Side-result:** the DDMA control/status block *is* safely readable; `CFG_STATUS.IDEL`
  is a coarse "DDMA idle" signal, but that is not byte accounting.
- **For bytes-moved ground truth, go outside the NPU register space:** the RK3588
  **DDR/DFI PMU** or a **NOC/MSCH performance probe** with master-ID filtering measures
  the same physical traffic and sidesteps the unmapped-MMIO hazard entirely. This is the
  recommended next avenue if DMA-byte accounting is needed.

## 6. How to read DDMA safely (probe design)

The hazard above (a read can hard-lock the box) shapes the safe probe
(`patches/rocket/086-rocket-drv-perf-counters.patch`, source
`patches/rocket/perf-probe-v2-safe.c`): debugfs `rocket_perf/ddma`, **read-only** (no
writes → cannot corrupt DDMA config; note `0x8010 = RD_WEIGHT_1` is live config),
**core-0 only**, **disarmed unless** loaded with `rocket_ddma_probe=1` (so `cat` while
disarmed touches no hardware), reading the known `CFG_STATUS` first with each read in its
own `seq_printf` (so any abort is attributable to one register). Run with a hardware
watchdog / BMC reset available. The probe maps only the DDMA domain — never the
hard-locking `0x2000` page.

## 7. The RK3576 "DPU bytes-written counter" lead

A mainline-`rocket` **RK3576** bring-up (gahingwoo) reads a per-job "DPU bytes-written
counter" as a conv success metric — conv0 112×112×32 int8: **401408 B** = all 32 channels,
25088 = 2 channels. It is not a new readable register; it is the **same `dt_wr` amount
counter** §1 rules dead, confirmed three independent ways:

- **It is the BSP `dt_wr` ("data write") amount counter.** The vendor BSP sums a
  top + per-core "data write" amount at core-window offsets **0x2234** + **0x2434**, ×2
  scale. rk3576 wires both tables; **rk3588 sets both NULL** — the same `0x22xx`/`0x24xx`
  family as §1.
- **Even on RK3576 the counter is behavioral** — a BSP/register-RE artifact of the amount
  block, not a separately documented register. So there is no documented "separate readable"
  counter to port to RK3588.
- **On RK3588 it lands in the unmapped, hard-locking page.** 0x2234/0x2434 relative to a
  core's window → phys `0xfdab2234`/`0xfdab4234`, the `0x2000`-gap pages `rocket` never
  ioremaps (it maps only `pc`/`cna`/`core`) — §3's read-fatal region.

No **separate** output-write counter hides in a *mapped* domain either: the only WDMA
registers in the DPU block are `WDMA_SIZE_0/1` (0x4058/0x405C) = output-**shape** config
(channel/height/width the host writes), not a write-back count (Mesa `registers.xml`; our
`npu_hw.h`). This is NVDLA by construction — the SDP (== our DPU) register group's
only perf registers are stall / saturation *events* (`D_PERF_WDMA_WRITE_STALL`,
`D_PERF_OUT_SATURATION`), never byte/element counts; NVDLA byte accounting lives in the
separate amount block (rocket's unmapped `0x2xxx`), not the SDP group.

**The lead does not reopen the negative.** No safe, `rocket`-mapped, per-op output-bytes
counter exists on RK3588; the analytical bytes-moved model (§5) remains the route. No probe
was run — the only candidate register is the known hard-locking page, and reading the mapped
WDMA/DPU registers returns the programmed output *shape*, not traffic.

## References

- `rocket`: `drivers/accel/rocket/rocket_{drv,core}.c`, `rocket_registers.h`; DT
  `rk3588-base.dtsi` (per-core pc/cna/core)
- Register map: Mesa `rocket/registers.xml` (`DDMA` domain @ 0x8000), our `npu_hw.h`
- Probe source: the `rocket` perf-counter probe patch (`perf-probe-v2-safe.c`)
- RK3576 lead (§7): gahingwoo's mainline-`rocket` RK3576 bring-up
  (`https://www.reddit.com/r/embedded/comments/1ub5npg/`)
