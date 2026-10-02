# Pool completion on mainline rocket

A pooling program on the RK3588 raises no DPU interrupt. A driver that arms only the DPU pair
therefore masks off the only completion such a job can raise. Nothing signals, `drm_sched`
retires the job at its 500 ms deadline, and the core is reset. **Every pool submit costs
~507 ms and one reset** on a driver in that state [HW sweep].

The answer is correct throughout. The hardware finished in microseconds and the output BO holds
the right surface, so nothing in a gate or a model output shows it. Only the wall does.

`DRM_ROCKET_JOB_PPU_DONE` is the fix. It arms the PPU pair for a job whose last program is a
PPU program. The DRM interface version goes to 1.3 with it. Userspace reads that version to
tell a kernel that honors the flag from one that rejects it. With the flag, the same submits
cost microseconds.

## The DPU-only interrupt mask

`PC_OPERATION_ENABLE` is a per-block bitmap, and the two programs are disjoint. A convolution
enables CNA/CORE/DPU/DPU-RDMA with `0x1d`. A pool enables PPU and PPU_RDMA with `0x60`
(`gen_pool_fp16`, `src/npu_regcmd.c`). A pool is a self-contained PPU job with no DPU stage
anywhere in it. PPU_RDMA reads the input cube, the PPU reduces, and the PPU writes the output
cube.

Without the flag, `rocket` arms and tests exactly the DPU pair [source-confirmed]:

- `rocket_job.c`: `rocket_pc_writel(core, INTERRUPT_MASK, PC_INTERRUPT_MASK_DPU_0 | PC_INTERRUPT_MASK_DPU_1)`
- `rocket_job.c`: the handler returns `IRQ_NONE` unless `DPU_0` or `DPU_1` is set in `INTERRUPT_RAW_STATUS`

So the PPU's own completion, two bits over in the same register, is masked off and ignored.
`drm_sched` then times the job out at `JOB_TIMEOUT_MS` and resets the core.

The deadline is what a stock driver falls back on. No patch in `patches/rocket/` other than 093
touches `INTERRUPT_MASK` or the handler.

## The cost without the flag

The board is a Turing RK1 on kernel `7.1.7-1-arm64`, with `rocket` DRM interface 1.1.0 and the
NPU pinned at 600 MHz. Both userspace pool entries ran the same geometry (C=8, a 7×7 plane, a
7×7 average window at stride 1, a whole-plane mean either way). Each took six calls,
alternating [HW sweep]:

| entry | per call |
|---|---|
| `rocket_pool_fp16()` | 530.6, 533.3, 506.7, 506.6, 506.7, 506.5 ms |
| `rocket_global_avgpool_fp16()` | 506.8, 506.6, 506.8, 533.2, 506.8, 506.5 ms |

The twelve submits log twelve `NPU job timed out` lines in `dmesg`. Both entries return the
same value (1.4229), which is the value the host reference gives. The floor is `JOB_TIMEOUT_MS`
(500, `rocket_job.c`) plus the scheduler's tick. The cost has no shape or size dependence,
because the wait depends on nothing about the program.

The same `reduce_mean_rocket` binary takes 0.027 s against the vendor `rknpu` driver. It runs
the same reductions, the same library and the same regcmd program there. The difference is that
userspace tells that driver which block finishes the job, and the driver waits for that one
[HW sweep]. That ratio is the size of what the deadline costs, not a property of either part.

## The cost with the flag

The board and the binaries are the same. The kernel is `7.2.0-1-arm64` carrying
`patches/rocket/093-rocket-drv-ppu-completion.patch`, with `rocket` DRM interface 1.3.0 and the
NPU pinned at 600 MHz. The runs were measured 2026-08-24, warm, with the first run of each
discarded [HW sweep]:

| gate | interface 1.1.0 | interface 1.3.0 |
|---|---|---|
| `pool_fp16_rocket` (11 submits) | 5.867, 5.872 s | 0.038, 0.036 s |
| `pool_int8_rocket` | 6.952 s | 0.041 s |
| `reduce_mean_rocket` (33 submits) | 21.956, 21.993 s | 0.095, 0.088 s |
| `reduce_feature_rocket` | 0.162 s | 0.161 s |

`reduce_feature_rocket` is unchanged because it submits no PPU program. Its reductions run on
the matmul datapath.

The wall is the weaker half of the evidence. 093 ships inside the boot2deb
`linux-image-7.2.0-1-arm64` build, so the kernel moved together with the driver. The two arms
therefore differ by more than the flag. One observation does not depend on the kernel version:
with the flag, no pool or reduce submit raises an `NPU job timed out` line. The completion
arrives, and the deadline is never reached.

The whole RK3588 suite is 96 of 96 in 380.46 s against 415.15 s on interface 1.1.0. Those
runs also logged 64 timeout lines, and none of them was a pool. `uapi_bo_lifetime_rocket` built
the RK3576's program on every part, and the RK3588 hung on each of its 64 submits until the
watchdog retired it. With each part's own program the whole suite logs no timeout [HW sweep,
Turing RK1, 2026-09-24]. Attribute a timeout line before reading it as a pool.

## The `DRM_ROCKET_JOB_PPU_DONE` flag

`DRM_ROCKET_JOB_PPU_DONE` is bit 2, and it names the class of the job's last program. What a
caller must know about it:

- It is a hint that has to be right, not an optimization. Setting it on a job whose last
  program is a convolution costs that job the same 500 ms deadline.
- If a chained stream has DPU programs before its final PPU program, the job **must not** set
  the flag. An interior program's PPU bit would retire the job while a later DPU write was
  still draining.
- The submit ioctl **rejects an unknown flag word** rather than ignoring it, so sending the bit
  to a kernel older than 1.3 fails the submit. "Advisory" is a property of the flag's meaning,
  not of sending it.
- Bit 1 and version 1.2 stay claimed by the RK3576 series' `DRM_ROCKET_JOB_NO_DPU_DONE`. That
  flag names a completion class for a driver that polls, and it has no meaning here. Skipping
  both keeps one uAPI across the two parts, which is what userspace gates on.

The driver latches the mask into the core at submit, so the hard IRQ handler still needs no job
pointer and no lock.

`librocketnpu` asks for the flag everywhere it submits a pool or a reduce (`rocket_pool.c`,
`rocket_reduce.c`, `rocket_pool_rk3576.c`), gated on `rocket_ppu_done_supported()`, a
version-only probe. That probe has no SoC gate, by design. Bit 2 and version 1.3 mean the same
thing on both parts, which is what the RK3588 series gains by skipping 1.2.

## Kernels that report below 1.3

Do not submit a pool per inference on such a kernel. There, `rocket_ppu_done_supported()`
returns 0, the flag is never set, and every pool submit pays the deadline. The `tflite-rocket`
delegate's on-NPU pool is opt-in (`pool_npu`) and off by default. Leave it off against such a
kernel: the host pool kernel costs microseconds against the half second a submit costs.

The RK3576 series carries the same change as `patches/rk3576/npu/0021`. Its prose states that
"RK3588 takes a completion interrupt and does not poll, so it is untouched". The completion
interrupt an unpatched RK3588 takes is DPU-only, so the same class of program is unretirable
there. The deadline it falls back on is 500 ms rather than the RK3576's `dpu_grace_us`
(213-641 us measured).

## The `dmesg` timeout count as an instrument

**A `dmesg` count delta is not an instrument for the failing case.** The ring wraps. Once a run
adds timeout lines at 2/s, the evictions take matching lines out at the same rate. So a
before/after `grep -c` can read zero for a gate that timed out on every submit. Wall time is
the instrument there, and `dmesg` only tells you the reason.

The count is a sound instrument for the passing case, where the expected count is zero and a
single line is a finding. Before reading a line, attribute it to a gate by running that gate
alone. A gate that abandons jobs by design produces the same message.

## Untested alternative

One alternative is untested: encoding the pool DPU-fed (`FLYING_MODE=1` with `DPU_FLYIN=1`), so
that a DPU completion arrives on a kernel with no patch at all. This project measured only the
standalone RDMA-fed program that the encoder emits.
