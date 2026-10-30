# Pool completion on mainline rocket

On mainline `rocket` without the `DRM_ROCKET_JOB_PPU_DONE` flag, every pool submit on the RK3588
costs ~507 ms and one core reset [HW sweep]. A pooling program raises no DPU interrupt. A
driver that enables only the DPU pair therefore masks off the only completion such a job can
raise. Nothing signals, `drm_sched` retires the job at its 500 ms deadline, and the core is
reset. The output is correct: the hardware finishes in microseconds, and the output BO holds
the right values. Only wall-clock time shows the delay.

`DRM_ROCKET_JOB_PPU_DONE` enables the PPU pair for a job whose last program is a PPU program.
With the flag, the same submits cost microseconds. The flag comes with DRM interface version
1.3, which userspace reads to tell a kernel that honors the flag from one that rejects it.

**Conditions:** Turing RK1 (RK3588), NPU pinned at ~816 MHz, the same binaries in both columns.
Interface 1.1.0 is kernel `7.1.7-1-arm64`. Interface 1.3.0 is kernel `7.2.0-1-arm64` with
`patches/rocket/093-rocket-drv-ppu-completion.patch`, measured 2026-08-24, warm. The two
columns differ in kernel version as well as in the flag.

**Not characterized:** a pool encoded DPU-fed (`FLYING_MODE` 0, the TRM's DPU source, with
`DPU_FLYIN` set [expected]), so that a
DPU completion arrives on a kernel with no patch at all. Only the standalone RDMA-fed program
that the regcmd encoder emits is measured.

| Workload | interface 1.1.0 | interface 1.3.0 |
|---|---|---|
| fp16 pool (11 submits) | 5.867, 5.872 s | 0.038, 0.036 s |
| int8 pool | 6.952 s | 0.041 s |
| Mean reduction (33 submits) | 21.956, 21.993 s | 0.095, 0.088 s |
| Feature reduction, no PPU program | 0.162 s | 0.161 s |

## Cause

### The DPU-only interrupt mask

`PC_OPERATION_ENABLE` is a per-block bitmap, and the two programs are disjoint. A convolution
enables CNA/CORE/DPU/DPU_RDMA with `0x1d`. A pool enables PPU and PPU_RDMA with `0x60`. A pool
is a self-contained PPU job with no DPU stage anywhere in it. PPU_RDMA reads the input cube,
the PPU reduces, and the PPU writes the output cube.

Without the flag, `rocket` enables and tests exactly the DPU pair [source-confirmed]:

- `rocket_job.c`: `rocket_pc_writel(core, INTERRUPT_MASK, PC_INTERRUPT_MASK_DPU_0 | PC_INTERRUPT_MASK_DPU_1)`
- `rocket_job.c`: the handler returns `IRQ_NONE` unless `DPU_0` or `DPU_1` is set in `INTERRUPT_RAW_STATUS`

The PPU's own completion, two bits over in the same register, is therefore masked off and
ignored. `drm_sched` then times the job out at `JOB_TIMEOUT_MS` and resets the core.

The deadline is what a stock driver falls back on. No patch in `patches/rocket/` other than 093
touches `INTERRUPT_MASK` or the handler.

## The cost without the flag

Both userspace pool entries, at one geometry, per call [HW sweep]:

| entry | per call |
|---|---|
| `rocket_pool_fp16()` | 530.6, 533.3, 506.7, 506.6, 506.7, 506.5 ms |
| `rocket_global_avgpool_fp16()` | 506.8, 506.6, 506.8, 533.2, 506.8, 506.5 ms |

**Conditions:** Turing RK1, kernel `7.1.7-1-arm64`, `rocket` DRM interface 1.1.0, NPU pinned
at ~816 MHz. C=8, a 7×7 plane, a 7×7 average window at stride 1, a whole-plane mean either way.

Each submit logs one `NPU job timed out` line in `dmesg`. Both entries return the same value
(1.4229), which is the value the host reference gives. The minimum per-call cost is
`JOB_TIMEOUT_MS` (500, `rocket_job.c`) plus the scheduler's tick. The cost has no shape or size
dependence, because the wait depends on nothing about the program.

On the vendor `rknpu` driver, the same mean-reduction test takes 0.027 s, with the same
reductions, the same library and the same regcmd program. Userspace tells that driver which
block finishes the job, and the driver waits for that one [HW sweep]. The ratio between the two
drivers is the size of what the deadline costs.

## The cost with the flag

The feature reduction is unchanged because it submits no PPU program. Its reductions run on the
matmul datapath.

With the flag, no pool or reduce submit raises an `NPU job timed out` line [HW sweep]. The
completion arrives, and the deadline is never reached.

## The `DRM_ROCKET_JOB_PPU_DONE` flag

`DRM_ROCKET_JOB_PPU_DONE` is bit 2, and it names the class of the job's last program. Software
must set it only on a job whose last program is a PPU program. Setting it on a job whose last
program is a convolution costs that job the same 500 ms deadline.

> [!CAUTION]
> On a chained stream with DPU programs before its final PPU program, the flag retires the job
> early. An interior program's PPU bit would retire the job while a later DPU write was still
> draining. Software must not set the flag on such a job.

The submit ioctl rejects a flag word with an unknown bit rather than ignoring it. Bit 2
therefore fails the submit on a kernel older than 1.3, and software must not set it there.

Bit 1 and version 1.2 stay claimed by the RK3576 series' `DRM_ROCKET_JOB_NO_DPU_DONE`. That
flag names a completion class for a driver that polls, and it has no meaning here. Skipping
both keeps one uAPI across the two SoCs, which is what userspace checks.

The driver latches the mask into the core at submit, so the hard IRQ handler still needs no job
pointer and no lock.

When `rocket_ppu_done_supported()`, a version-only check, reports support, `librocketnpu` asks
for the flag everywhere it submits a pool or a reduce (`rocket_pool.c`, `rocket_reduce.c`,
`rocket_pool_rk3576.c`). The check does not test the SoC, by design. Bit 2 and version 1.3 mean
the same thing on both SoCs, which is what the RK3588 series gains by skipping 1.2.

## Kernels that report below 1.3

Do not submit a pool per inference on such a kernel. There, `rocket_ppu_done_supported()`
returns 0, the flag is never set, and every pool submit incurs the deadline. The `tflite-rocket`
delegate's on-NPU pool is opt-in (`pool_npu`) and off by default. Leave it off against such a
kernel: the host pool kernel costs microseconds against the half second a submit costs.

The RK3576 series includes the same change as `patches/rk3576/npu/0021`. Its prose states that
"RK3588 takes a completion interrupt and does not poll, so it is untouched". The completion
interrupt an unpatched RK3588 takes is DPU-only, so the same class of program cannot be retired
by its completion there. The deadline it falls back on is 500 ms rather than the RK3576's
`dpu_grace_us` (213-641 us measured).

## The `dmesg` timeout count as an instrument

> [!NOTE]
> A before/after `grep -c` of `dmesg` can read zero for a test that timed out on every submit.
> The ring wraps, and once a run adds timeout lines at 2/s, the evictions take matching lines
> out at the same rate. Wall-clock time detects the failing case, and `dmesg` only tells you the
> reason.

The count is a sound instrument for the passing case, where the expected count is zero and a
single line is a finding. Before reading a timeout line as a pool's, attribute it to a test by
running that test alone. A test that abandons jobs by design produces the same message.

## Evidence

### The DPU-only interrupt mask

The pool's `0x60` is the value `gen_pool_fp16` (`src/npu_regcmd.c`) emits.

### The cost without the flag

Each entry took six calls, alternating between the two. The twelve submits logged twelve
`NPU job timed out` lines in `dmesg`. The vendor-driver figure is the same `reduce_mean_rocket`
binary.

### The cost with the flag

The opening table's rows are the tests `pool_fp16_rocket`, `pool_int8_rocket`,
`reduce_mean_rocket` and `reduce_feature_rocket`, with the first run of each discarded.

The wall-clock comparison is the weaker half of the evidence. Patch 093 is built into the
boot2deb `linux-image-7.2.0-1-arm64` kernel, so the kernel moved together with the driver. The
two columns therefore differ by more than the flag. The absence of timeout lines does not
depend on the kernel version.

The whole RK3588 suite is 96 of 96 in 380.46 s against 415.15 s on interface 1.1.0. Those
runs also logged 64 timeout lines, and none of them was a pool. `uapi_bo_lifetime_rocket` built
the RK3576's program on every SoC, and the RK3588 hung on each of its 64 submits until the
watchdog retired it. With each SoC's own program the whole suite logs no timeout [HW sweep,
Turing RK1, 2026-09-24].
