# IOVA windows and multicore (kernel / DMA)

This note covers two facts about how `rocket` maps memory and dispatches work, both of which
a design must account for. It also covers how the vendor `rknpu` BSP driver differs on the
first of them.

## Per-fd IOVA windows

The regcmd encodes BO addresses as 32-bit fields (e.g. `weights_dma & 0xFFFFFFFF`), so
every BO that a job references must live in the low 4 GB of its address space. **Each fd has
its own independent 0…4 GB IOVA window** [HW sweep, source-confirmed].

On the RK3588 this 4 GB is the regcmd-field encoding limit, not the bus. The RK3588 NPU
AXI/IOMMU is 40-bit, per the upstream rocket RK3568 RFC, which contrasts RK3568's "32-bit
NPU AXI/IOMMU (vs 40-bit)" ([LKML](https://lkml.iu.edu/2605.3/10672.html)). So the bus can
physically reach >4 GB, and the 32-bit address field in the regcmd is what caps a single
context at 4 GB. On the RK3568 the 4 GB is instead a true bus limit. The NPU AXI and the
IOMMU page-walker/DTE are both 32-bit, so its page tables need `GFP_DMA32`.

A probe (`iova_ceiling_rocket.c`) that opened 2 fds saw both independently climb
`va = 0x0 -> 0x100000000`. That is 8 GB total, and the VAs repeat per fd. The consequences:

- **N worker fds give ~N × 4 GB of usable IOVA.** The 5-worker resident context has
  ~20 GB. That is enough for ~80-90% of a 22 GB Gemma-4-12B F16 resident, and for all of any
  smaller or quantized model.
- **BO allocation is lazy.** Reserving 8 GB of IOVA left RAM flat (~595 MB). Only the data
  you actually pack commits physical RAM.
- **Whether that window is the binding limit is a separate question, and usually it is not.**
  The f16 residency route fills it at 5 workers on a 12B F16 at `-p 512`, placing 286 of 328
  weights. At `-p 2048` the same route hits the RAM reserve floor first, at the same placement.
  The MoE route never reaches it at all, because its admission charges the GGUF source against
  RAM and only the int8 codes against IOVA. See `benchmarks.md` for the arithmetic and
  `data/tuning-matrix.md` for the per-shape readings.

## The vendor `rknpu` driver's shared IOVA domain

**Per-fd is a property of `rocket`, not of the silicon. The multi-fd strategy above does not
transfer to the vendor `rknpu` BSP driver.** That driver maps every buffer through one IOMMU
domain shared across the whole process. So a second fd buys no address space, and two
processes spend the same budget. Freshly booted, it serves about the same total: ~3.9 GB in
buffers of any size from 16 MB to 256 MB, against `rocket`'s 4.00. So the two drivers start
equivalent, and they diverge only in how that budget is shared and in what happens next.

The vendor driver also allocates from the top of the domain down. BOs land at `0xffe00000` and
`0xffff0000` on a 6.1.172 vendor kernel with `rknpu` 0.9.8 [HW sweep, vendor RK1]. `rocket`
allocates from 0 upward.

**On that driver, workloads that map through the kernel's generic path consume the window,
and nothing in normal operation gives it back.** The consumer is the mapping route, not
elapsed time. One `llama.cpp` 2048-token prefill (Llama-3.2-3B F16) with
`RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT` clear costs the shared domain 5-11 of its 31 buffers of
128 MB in 153 s. The loss outlives the process. With that flag set, which is the
`librocketnpu` provider's default, the identical run costs zero. Four such arms interleaved
around the leaking ones left all four size counts byte-identical
[HW sweep, RK3588, `rknpu` 0.9.8, 2026-08-25].

Because the domain outlives every process that used it, a reboot is the only reset.
Detaching and re-attaching the device (the driver's own soft reset) re-uses the same domain
object and does not rebuild the allocator.

**The flag exists only from `rknpu` 0.9.7 (2024-04-24), and an older driver takes it and
ignores it** [source-confirmed, `rknpu_ioctl.h` and `rknpu_gem.c` across 0.8.0-0.9.8]. The
flag `RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT` is bit 10, added by that release. The driver
declares `RKNPU_MEM_MASK` and never validates against it. It passes `args->flags` straight to
`rknpu_gem_object_create()` and tests individual bits. So on 0.9.6 and earlier the
allocation takes the leaking generic route while the flag reads as set, with nothing
reported. Read the driver version before crediting the flag with anything.

### The IOVA rcache mechanism

The mechanism is the kernel's IOVA rcache, and the driver reaches it by mixing two
allocators on one domain. The two routes do not differ in size rounding or in placement.
They differ in which allocator they use, and therefore in where a freed range goes. With the
flag set, the driver calls `alloc_iova()` / `free_iova()`, which are the rbtree directly.
With the flag clear, the mapping falls to the generic `dma_map_sg()`, hence
`alloc_iova_fast()` / `free_iova_fast()`, which go through the per-CPU IOVA rcache.

On a free, `free_iova_fast()` parks the range in a per-CPU magazine or the global depot, and
falls through to `free_iova()` only when that fails. **`alloc_iova()` never consults the
rcache**, so address space freed on the generic route becomes unreachable to the driver's own
route. The rcache belongs to the `iova_domain`, which here is one domain shared process-wide.
That is why the loss outlives the process that caused it. Mixing the two allocators on a
single domain is the defect. Either one used consistently is sound.

`iova_rcache_insert()` accepts only sizes up to `2^(IOVA_RANGE_CACHE_MAX_SIZE-1)` = 32 pages =
128 KB. That bound makes the effect reproducible, and it hides the effect from a probe that
frees only larger buffers. The measurement churns 4400 buffers through the generic route and
frees every one. The result [HW sweep, RK3588, `rknpu` 0.9.8, 2026-08-26, fresh domain per
arm, capacity measured on the tight route]:

| churn size | route | 16 MB | 64 MB | 128 MB | 192 MB | consumed |
|---|---|---|---|---|---|---|
| fresh | n/a | 255 | 63 | 31 | 21 | n/a |
| **128 KB** | generic | 221 | 55 | **27** | 18 | **34 / 8 / 4 / 3** |
| 256 KB | generic | 255 | 63 | 31 | 21 | 0 |
| 132 KB (one page over) | generic | 255 | 63 | 31 | 21 | 0 |
| 128 KB | tight | 255 | 63 | 31 | 21 | 0 |

The effect is route-selective, and size-selective at exactly the 32-page boundary. One page
over, it vanishes, while twice the address space at 256 KB costs nothing. Three quantitative
checks agree:

- The 34 x 16 MB lost is 544 MB, against the 540 MB that a single-threaded run can park (one
  CPU's two magazines plus the 32-magazine depot, 4318 ranges x 128 KB).
- A second identical churn costs only 3 more and a third only 1, because the cache is
  already at its ceiling.
- A churn spread over 8 CPUs and all six cached orders costs 10 of 31 buffers of 128 MB.
  One real 153 s prefill costs 11.

The synthetic churn reproduces the workload.

### Probes that cannot see the loss

A probe whose frees are all above the 128 KB rcache bound reads zero loss. The probe
`iova_probe` allocates 16-192 MB, and the mixed-size fragmenter cycles 1-23 MB. Every size
either of them frees is far above the bound, so none of their frees can enter the cache at
any flag value.

A probe that allocates one size also cannot separate the routes at all. On an empty domain
both routes return identical counts and identical addresses at 16, 32, 32.03, 48, 64, 128
and 192 MB. The error path is not the cause either: every one of five prefill runs across
both routes reported zero kernel allocation failures.

### The rcache flush on the generic route

The kernel's safety valve exists and works, and it belongs to the wrong allocator. When
`alloc_iova_fast()` cannot satisfy a request, it flushes every online CPU's magazines and
the whole depot, then retries. So the route that fills the cache can always empty it again.
Allocating to refusal on the generic route repairs the domain outright
[HW sweep, RK3588, `rknpu` 0.9.8, 2026-08-26, one boot, addresses recorded]:

| step | measuring route | 128 MB BOs | address span |
|---|---|---|---|
| fresh | tight | 31 | `0x8000000`-`0xffffffff` |
| fresh | generic | 31 | same |
| after a 128 KB generic churn | tight | **27** | `0x5a00000`-`0xdd9fffff` |
| same domain | **generic** | **31** | `0x8000000`-`0xffffffff` |
| after that generic pass | tight | **31** | recovered |

The 550 MB that the degraded tight route cannot reach sits in one block above `0xdda00000`,
which is the rcache ceiling to the megabyte. It comes back whole. The workload-shaped churn
behaves the same: 8 CPUs across all six cached orders take 255/63/31/21 to 213/53/26/17, and
one generic pass restores 255/63/31/21.

`alloc_iova()` has no such path. The flush helpers `free_cpu_cached_iovas()` and
`free_global_cached_iovas()` are static to `iova.c` and unexported. So a driver allocating
that way cannot ask for the parked space back at all. It returns `-ENOMEM` with hundreds of
megabytes sitting in a cache that it cannot see or drain. That, rather than "mixing
allocators leaks", is the precise defect. The leaking route can always clean up after
itself, and the route that never leaks is the one that pays.

### Measuring the loss

Two traps apply to measuring this. First, a route that repairs the domain and a route that
is blind to the loss return the same number. Separating them takes a third measurement: a
re-read on the route that showed the loss, after the other route has run. Without that step
a repair is recorded as blindness.

Second, if the routes are mixed inside one before/after comparison, two fixed route
properties read as degradation. The generic route allocates size-aligned, so a fresh domain
serves it 15 buffers of 192 MB against the tight route's 21 (192 MB rounds its alignment to
256 MB). The tight route also packs below the generic route's first address. Take both
halves of any comparison on one route.

### The vendor default route

**The leaking route is the vendor default.** The flag `RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT`
is an opt-in bit at buffer creation. A userspace that does not set it, which is the stock
path, takes the generic route and leaks. The `librocketnpu` provider sets the flag by
default, which is the whole reason the tight route is the default there.

Two consequences follow for anyone driving that path. First, a large allocation failure
there is transient rather than structural. A retry at a smaller size is usually served,
which is what `librocketnpu` does for the chained attention path.

Second, **a board's allocation history, not its uptime, is the variable in any
allocation-sensitive measurement**. A board 1 day 4 h into its uptime had carried a full
day of NPU work under the tight route. It still measured the fresh-boot row (255 / 63 / 31 /
21 buffers at 16 / 64 / 128 / 192 MB). One 153 s run on the generic route moved it. Probe
the domain before recording a number as a property of the driver
[HW sweep, RK3588, `rknpu` 0.9.8 vs mainline `rocket` 1.1.0].

## Resident weights in the per-fd window

### Per-tensor scratch

A "`ROCKET_CACHE_MB=12000` exceeds 32 bits" crash is per-tensor scratch bloat, not a 4 GB
wall. Each resident weight carries its own compute scratch, ~2x the weight bytes. Sharing
scratch per (worker-fd, shape) and keeping only the weight tiles per-tensor lifts a 12B F16
model from ~30%-resident to ~80-90%-resident within the same IOVA.

### M-independent resident weight scatter

The resident weight scatter is M-independent. A weight's scattered tile positions are fixed
by the N-slice split and the K/N tiling (`Nt`, `Kt`) alone. The M dimension sets only how
many input and output tiles stream through, never where a weight byte lands. The tiling can
therefore be planned at a canonical M (`MAX_TILE`) instead of the actual row count. One
resident weight then serves every M with no re-pack. Warmup-M packs once, and any later
prefill M (down to a single short-prompt tile) reuses it.

This holds for three resident paths, which are all bit-exact across M [HW sweep,
`matmul_{int8,int4}_crossm_rocket`, pack M=512 reused at M=512/256/768/64/8]:

- fp16
- int8, because int32 K-accumulation is exact for any K-tiling
- Group-wise int4, because the K-tile is already pinned to `group` and the host fp32
  K-accumulation order is per-`(m,n,group)`

When warmup-M ≠ prefill-M, the cost of not doing this is a full model re-pack. That is a
multi-minute stall on a 12B model for a single short prompt. The library detects a genuine
tiling mismatch (e.g. a `ROCKET_MM_*` override changing the tiling between pack and compute)
and returns `-2`, so the caller re-packs rather than miscomputing.

A caller with many short, varied lengths asks for the canonical plan outright:
`rocket_ctx_create_ex(n, ROCKET_CTX_TILING_CANONICAL)`. A small M then keeps the M=256 `Kt` and
`Nt`, so one pack serves every length. The price is at the small end, where the per-M plan would
have picked larger K tiles. It is 1.16-1.42x at M 48-128 [HW sweep, RK1, 600 MHz, 2026-09-27].

### N-tile sizing on a worker slice

The N-tile must fit the slice, not the tile cap, or the tail tile is mostly empty. Because N
is split across the worker fds, each worker plans its tiling on a slice. A slice is not a
multiple of the 256-column tile cap. Defaulting the N-tile to the cap then stores a nearly
empty tail tile.

At the gpt-oss expert shape, `N=2880` over 5 workers gives a 576-wide slice. Three 256-wide
tiles then store 768 columns to hold 576. That is a 35% memory tax on every resident weight,
paid for the process lifetime.

The fix costs nothing. The cap fixes the tile count (`nNt = ceil(N/max_tile)`), and any
tile >= `ceil(N/nNt)` reaches that same count, so take the smallest one. Here that is
`ceil(576/3) = 192`. It gives the same three tiles, hence the same task count, the same
dispatch and the same real DMA, and it stores exactly 576 columns. Measured per resident
gpt-oss expert, the weight goes 10.70 -> 8.07 MiB (1.35x -> 1.02x of the logical `N·K` int8
bytes) [HW sweep].

Round the 32-column alignment up, not down. Rounding down can drop the tile below
`ceil(N/nNt)` and buy an extra tile, which is a real dispatch cost that saves nothing.

Two cautions apply:

- **It is a residency lever.** Residency is the whole product of a native-quant MoE expert
  cache, so the tax falls on the one thing being bought. It is nearly invisible on a dense
  model (one weight per tensor), and it is decisive across thousands of experts.
- **Shrinking `Nt` frees CBUF banks, which can let `Kt` grow.** A grown `Kt` re-tiles K,
  which changes the host K-accumulation order and breaks the resident-vs-one-shot
  bit-exactness. The change is safe on the group-wise planner only, where `Kt` can never
  exceed the quant group (checked: `Kt` is invariant over every `(group, slice)` a real
  weight produces). On the per-channel planner the same change moves `Kt` 640 -> 768, so it
  needs its own gate run.

## Multicore dispatch through worker fds

There are 3 NPU cores. `rocket` exposes them through the scheduler topology, not a submit
flag:

- The driver creates one `drm_sched` per core and one scheduling entity per fd. A DRM entity
  pins to one core while it has queued work (the in-order guarantee).
- So **one fd with many jobs serializes onto a single core.** A probe that submitted N jobs
  in one submit on one fd scaled 1.00 / 1.99 / 3.07x. It did not spread across cores: that
  3.07 was an artifact of job batching, not multicore.
- Driving N threads, each with its own fd and entity, makes the kernel dispatch across all 3
  cores. The measured rate is 39.5 -> 84.3 -> 116.1 -> 120.9 jobs/s at 1/2/3/4 threads
  (1.0 / 2.13 / 2.94 / 3.06x) [HW sweep, corroborated by Tomeu's blog].

One thread above the core count edges higher (T=4 > T=3, T=5 is the knee). The extra worker
fills the idle bubbles left by each worker's serial pack->submit->readback cycle. This is the
same pattern as the common "rknnpool" (N worker contexts round-robin over 3 cores, queue
depth > core count).

The proprietary RKNPU exposes core selection via a `core_mask` submit field. Mainline
`rocket` has none, and you get cores by using multiple entities. That is a different
interface, not a limitation.

## Worker fds in `librocketnpu`

`librocketnpu`'s matmul splits N (output channels) across worker threads, each with its own
fd. Each worker runs the unchanged single-fd matmul on a contiguous column slice and scatters
its dense result into the strided output. The library fans resident weights across the
worker fds, so each fd's slice fits its own 4 GB window. The default worker count is 5 (the
measured knee). See the `rocket-userspace` library's `rocket_matmul_mt.c` and
`rocket_prepacked*.c`.

The native int8/uint8 direct conv uses the same pattern (`rocket_conv_pool` /
`rocket_conv2d_int8_mt` in `rocket_conv.c`). A conv's tiling already decomposes it into
independent OC-group × oh-band × ow-band tiles, each writing a disjoint region of the
output. So the tiles fan across a pool of N worker fds, each with its own resident
`rocket_conv_ctx` (BO pool) and scratch. The pooled conv is bit-identical to the single-fd
`rocket_conv2d_int8` (same tiles, same single jobs, dispatched on different cores). It falls
back to serial for single-tile convs.

The `tflite-rocket` delegate creates one pool per partition (`nthreads`-sized). Measured on
warm MobileDet `native_int8`: 560 -> 458 ms (1.21x), with the conv bucket at 1.46x.

Multicore helps only a multi-tile conv. A conv small enough to fit one CBUF pass is a single
job, hence a single tile, with no intra-conv fan-out. It stays on one core. In
pointwise-heavy detectors many small 1×1s are single-tile. The matmul path, which splits the
output columns regardless of CBUF-pass fit, is the way to parallelize those.

## Multicore models

The RKNN `core_mask` API conflates two different things.

### Model 1: intra-model split

One inference is fanned across the 3 cores. RKNN does this with an intra-op
`subcore_task[5]` partition inside one submit (the vendor `core_mask` `0_1`/`0_1_2`).
Multi-fd entities (the N-split above) give the same outcome. `librocketnpu` ships that for
the prefill matmul (5 workers) and the detection conv pool, and the prefill readback floor
is already post-3-core. Mainline `rocket` lacks the vendor's in-one-submit interface and
does not need it.

### Model 2: multi-instance throughput

N independent inferences run concurrently (queue depth > core count), so another context
fills each context's serial pack->submit->readback bubble. go-rknnlite measured
EfficientNet-Lite0 at pool-of-9: 7.9 -> 1.65 ms (4.8x throughput, latency unchanged).

The payoff is Frigate multi-camera (one context per stream), and it is wired. The
`tflite-rocket` delegate's `rocket.py` sets `ROCKET_CPU_AFFINITY` per detector process (one
A76 each, `nthreads=1`). So Frigate's one-process-per-detector model spreads N independent
contexts across the big cluster. Measured end to end through the delegate, P=1->4 gives
1.00 / 2.17 / 3.11 / 3.56x (`tflite-rocket/tools/pool_throughput.py`).

The mainline kernel's `drivers/accel/rocket/rocket_job.c` confirms the mechanism. A per-core
`drm_sched` plus a per-fd `sched_entity` initialized with all core schedulers gives
automatic load-balance of concurrent jobs.

## Small ops and multicore

A layer smaller than the multicore task-allocation granularity runs on a single core. This
is the same limit as the multi-tile-only caveat above: small pointwise convs do not fan out.
That is why the detection path needs the throughput pool (model 2), not more cores.

The load-bearing per-core knob is IRQ affinity. Set CPU/DDR/NPU to max frequency, pin the
app to a CPU big core, and **bind the three NPU interrupts to that big core**
(`/proc/irq/<npu-irq>/smp_affinity_list`). App `taskset` alone is a no-op for fp16 prefill
(whole-process `taskset`). The IRQ-affinity binding is the knob that matters, and on the
submit-overhead-bound path it is a large win. See §IRQ affinity below.

## IRQ affinity: the default routes the NPU completion IRQ onto a *little* A55 core

The measurement is from 2026-06-22 (7.1.0-1-arm64 at 600 MHz, IOMMU keep-attached,
`submit_overhead_rocket 8 64 16`, 5×3000 + 4000-iter confirmation). On this kernel the NPU
GIC IRQs are 69 = `fdab0000.npu`, 70 = `fdac0000.npu` and 71 = `fdad0000.npu`. Those are GIC
142/143/144, not the 110/111/112 sometimes quoted elsewhere. The numbers are SoC- and
kernel-specific, so always read `/proc/interrupts`. The RK3588 CPU map is cpu0-3 = A55
little at 1.8 GHz and cpu4-7 = A76 big at 2.4 GHz.

The default IRQ affinity is the all-CPUs mask `0-7`. GICv3 routes a multi-CPU level IRQ to
the lowest CPU in the mask, which is cpu0, an A55 little core. Watching `/proc/interrupts`
confirms it: 8000 submits incremented IRQ 69/70 entirely in the CPU0 column. So by default
the completion handler and the waiter wakeup both run at 1.8 GHz. The result, as median
µs/submit for the dispatch floor only:

| config | IRQ affinity | app taskset | median µs/submit |
|---|---|---|---|
| default | `0-7` (-> services on cpu0, A55) | none | 51-53 |
| default + app pinned | `0-7` | cpu7 | 51 (no change, the IRQ is still on A55) |
| IRQ on big core, app floats | cpu6 | none | 33.5 |
| IRQ + app co-located on one big core | cpu6 / cpu7 / cpu4 | same big core | 27-28 |
| IRQ + app on *different* big cores | cpu4 | cpu7 | 31 |

The table shows three things:

- **Moving the IRQ off the A55 onto any A76 is the dominant win**: 51 -> ~31 µs (−40%).
  Which big core does not matter (4, 6, 7 all gave ~27-28 µs co-located).
- **Co-locating the waiter on that same big core captures a further ~4 µs** (31 -> 27, −47%
  total vs default). The completion IRQ then wakes a cache-hot, same-cluster thread with no
  cross-cluster IPI.
- **App `taskset` alone does nothing** (`0-7` IRQ + app on cpu7 = 51 µs). With the IRQ
  still serviced on the little core, pinning the app cannot help. The IRQ binding is the
  prerequisite, and co-location is the bonus.

### Recommended bindings

Apply these once, as root.

**Latency or single-stream** (decode GEMV, single-camera detection): pin all 3 NPU IRQs to
one A76 and run the app there:

```sh
for q in 69 70 71; do echo 7 > /proc/irq/$q/smp_affinity_list; done; taskset -c 7 <app>
```

**Throughput pool** (the multi-instance, multi-fd work): spread the 3 IRQs across 3 A76s
(`69->5 70->6 71->7`) and pin each worker to its core. Completions then do not queue behind
one handler.

This is a runtime and system-config lever, with no driver or library change. It complements
the IOMMU keep-attached patch below. That patch removes ~20 µs of kernel work per submit,
and this lever removes the little-core wakeup latency per submit. Stacked, the floor on this
path is ~27 µs, against the ~54 µs of stock `rocket` on the default affinity.

The helpers are `rocket-userspace/tests/irq_affinity_probe.sh` (A/B harness) and
`rocket-userspace/tools/npu_set_irq_affinity.sh` (applies the recommended binding). The
lever pays on every many-small-submit path, and it is flat on a single big tiled prefill
matmul (one submit).

## Busy-polling the completion fence

Busy-polling is redundant with the IRQ-affinity binding. A blocking wait (`PREP_BO` with a
real timeout) sleeps the waiter until the kernel signals the job fence from its threaded
completion IRQ. So each wait costs an IRQ delivery plus a scheduler round-trip to re-run the
waiter. The knob `ROCKET_BUSY_POLL=<µs>` instead spins on a non-blocking completion probe for
up to that budget, then falls back to the blocking wait. That keeps the waiter runnable, so
it returns within one probe of the fence signalling.

The probe is `PREP_BO` with a zero deadline, the only completion check the mainline uAPI
exposes, so every poll also `dma_sync`s the output BO. The lever lives in
`rocket_bo_prep()`, and the A/B is the second loop in `submit_overhead_rocket`.

Busy-polling does not skip the IRQ. Only the kernel IRQ handler signals the mainline fence,
and userspace has no MMIO view of the PC `INTERRUPT_RAW_STATUS` register. So a userspace spin
can remove only the waiter-side wakeup, not the interrupt. Skipping the IRQ itself would need
a kernel-side poll (a `patches/rocket` change).

The measurement is from 2026-06-29, on 7.1.0-1-arm64 at 600 MHz with the A76 governors
pinned to `performance`. The governor must be pinned, or the idle A76 parks between submits
and swamps the signal. The run is `submit_overhead_rocket 64 256 512`, 3000-iter, an
in-process blocking-vs-busy-poll A/B [HW sweep]:

| IRQ affinity | app core | blocking median | busy-poll Δ median / mean |
|---|---|---|---|
| default `0-7` (A55) | A76 | 109 µs | −4.3 % / −4.7 % |
| -> cpu7 (A76) | other A76 | 96 µs | −1.8 % / +0.9 % (wash) |
| -> cpu7 (A76) | co-located | 97 µs | +0.3 % / −0.9 % (none) |

Busy-poll's best-case `min`/`p10` improves ~4-5 µs at every config (the removable wakeup).
The median win appears only when the completion IRQ is on a little A55 core. So busy-poll
and the IRQ-affinity binding above attack the same waiter-wakeup term. With the IRQ moved to
an A76 (the existing, free, no-core-cost knob), busy-poll adds nothing. Even on the default
A55, the IRQ-affinity binding alone (blocking 103->89 µs mean, −14 %) beats busy-poll's
−4.7 % without burning a core to spin.

On the tiny `8 64 16` shape (~20 µs floor) busy-poll is high-variance and not a reliable win
at all. At that floor the blocking waiter barely sleeps, so there is little wakeup to remove.

So `ROCKET_BUSY_POLL` ships opt-in and default-off. It is a single-stream latency fallback
for the case where the completion IRQ is stuck on its default little-core affinity and you
cannot re-bind it. When you can re-bind it, prefer `npu_set_irq_affinity.sh latency`, which
is cheaper (no spun core) and captures the same term. **Never enable busy-polling under the
throughput pool**: it burns a core that the pool wants for another stream.

## Per-job IOMMU dispatch cost

Stock `rocket` calls `iommu_attach_group()` in `rocket_job_run()` on every `drm_sched` job.
It calls `iommu_detach_group()` in `rocket_job_handle_irq()` on every completion, and on
reset. Each call toggles the rk_iommu stall/force-reset/paging handshake. On the RK3568 that
handshake times out on the idle NPU MMU (the upstream bug fixed by RFC patch 5/9). On the
RK3588 it completes silently and still costs latency on every submitted job.

The attach+detach handshake costs ~15-20 µs per `drm_sched` job. It was measured 2026-06-22
on 7.1.0-1-arm64 at 600 MHz, as a clean A/B of stock vs patch-5. Two independent tests
agree, and a third shows where the cost amortizes:

- `rocket-userspace/tests/submit_overhead_rocket.c` (tiny 1-task job, 2000x same fd): median
  54->34 µs/submit and min 39->23 µs with keep-attached, so ~20 µs (~38%) is the IOMMU term.
- `tests/multicore_probe` (64-task jobs, 10 reps): −17 to −18 µs per job (J=1: 7.68->7.50 ms
  over 10 jobs, J=3: 23.56->23.04 ms over 30 jobs).
- `matmul_tiled_rocket 512 3840 4096` (one big job): flat. The cost is per submit, not per
  task, so it amortizes to nothing on a single tiled prefill matmul and dominates streams of
  small jobs.

The lever is RFC patch 5, which keeps the per-context domain attached across same-fd jobs.
It tracks `attached_domain` in `struct rocket_core`, swaps only on a context change, holds a
`kref`, and detaches at teardown or after reset. It ships as
`patches/rocket/083-rocket-drv-iommu-keepattach.patch`, with CTest 8/8 and a clean dmesg.
This is the per-job companion to the per-tile dispatch levers (tile fusion, no-alloc
submit): it cuts the floor itself rather than the job count.

It pays on the submit-overhead-bound paths this file is about:

- Decode GEMV
- The small detection convs and 1×1s
- Multi-fd contention
- The throughput pool (model 2)

See [not-mac-bound.md](not-mac-bound.md) §dispatch floor.

### Reset before IOMMU detach

Keeping the domain attached across jobs says nothing about `rocket_reset()`, which detaches
explicitly. That detach is the one an unprivileged client can reach. The timeout handler
`rocket_job_timedout()` is its only client-reachable caller. So the group is detached while
the core is still wedged mid-DMA, with an unacknowledged fault in its MMU. In that state
rk_iommu cannot stall a bank. Each faulting job logs two
`rk_iommu … Enable stall request timed out` `dev_err`s, one per MMU bank (the RK3588 NPU MMU
has two).

`rk_iommu_disable()` swallows that error. The same handshake runs again inside
`rk_iommu_enable()` when the IOMMU core puts the group back on its default domain. The RK1's
default domain type is *Translated*, so an rk_iommu attach really does run. There the error
is returned, and `__iommu_group_set_core_domain()` turns it into a `WARN`: a taint, a
backtrace, and a panic under `panic_on_warn`.

`drm_rocket_task.regcmd` is a raw NPU IOVA written into the PC block unvalidated, and
`/dev/accel/accel0` is group `render`. So one field pointed at an unmapped address reaches
all of it.

**Reset the core before detaching**, and the handshake has a quiesced master to stall. The
reset also wipes the MMU page-table base, which is already why the domain must be dropped.
The change is `patches/rocket/088-rocket-drv-reset-before-iommu-detach.patch`, and that
order is also where the upstream RFC puts the detach.

It was measured 2026-07-31 (Turing RK1, 7.1.1, `081`-`087` out-of-tree) over 30
client-requested DMA faults per arm (`tests/uapi_regcmd_fault_rocket`, both modes). Either
module logs the same 30 `NPU job timed out` lines. The stall timeouts are 60 without the
patch and 0 with it, with 83/83 `ctest` and the taint word unmoved [HW sweep]. The `WARN`
itself is intermittent, because it needs the second handshake to fail as well, and it fired
in neither arm. What the A/B measures is the failing handshake that the `WARN` is downstream
of.

The RK3576 reaches the same detach and logs the same stall timeouts, and it does not `WARN`.
The two parts differ in the consequence, not in the trigger.

## Batched submit: one HW kick for many tasks

One large dispatch-floor lever is the number of HW kicks per inference. Mainline `rocket`
submits one task per kick. Its `rocket_job_run()` programs a single task's regcmd into
`PC_DATA_ADDR` and sets `PC_TASK_CON` `TASK_NUMBER(1)`. It re-arms the next task only on each
completion IRQ (`next_task_idx++` in the IRQ handler). A matmul tiled into `nMt·nNt·nKt`
tiles therefore pays one submit, one completion IRQ and one waiter wakeup per tile.

The BSP `rknpu` driver fires a whole task list in one kick. It does not do that by
DMA-walking a kernel-built descriptor table. That model was tested on the RK3588, and every
variant times out with no completion IRQ.

The mechanism is a self-chain
[HW sweep, source-confirmed: `rocket-userspace/src/rocket_matmul.c`]. The N tasks' regcmds
are laid contiguously in one buffer. Each task ends in an `OP_ENABLE`, and its trailer
carries the next task's address (an embedded `PC_BASE_ADDRESS` op) and stream length (its
`PC_REGISTER_AMOUNTS` op). The kernel programs only the first task's
`PC_DATA_ADDR`/`PC_DATA_AMOUNT`, sets `PC_TASK_CON.TASK_NUMBER = N`, and kicks once. The PC
executes one `OP_ENABLE` per task and follows each `PC_BASE_ADDRESS` link to the next. The
`TASK_NUMBER` field gates a single completion IRQ that fires after the last task (up to
`max_submit_number` = 4095/chunk on the RK3588).

The `PC_BASE_ADDRESS` redirect is load-bearing. A contiguous layout with only the amount op,
or with a zeroed trailer, runs task 0 and stops. The PC must be told where the next task is,
not just how long it is.

**Do not retire on a `PC_TASK_STATUS` read.** At IRQ time it reads `0x0000f000`
(`& 0xfff == 0`), not the completed count. Rely on the `TASK_NUMBER`-gated single IRQ.

`PC_TASK_DMA_BASE_ADDR` is not a kernel-walked task list. The BSP sets it to the regcmd
buffer in one example and to `0` in another, so it is don't-care for the stream. The
`rknpu_task` array exists only so the BSP kernel can read per-task fields CPU-side.

An independent FOSS RE of both stacks gives the end-to-end size of the gap. The proprietary
path issues ~63 IOCTLs / 1 submit per inference, where an open replay path issues
~634 IOCTLs / 10 submits. The proprietary path therefore makes ~10x fewer kernel transitions
[source: an independent FOSS RE of both stacks (orangepi5plus-npu), see
[SOURCES.md](../SOURCES.md)].

### The `rocket` lever

A `rocket` multi-task job runs in one kick under two coordinated changes.
Userspace packs the regcmds contiguously and links each trailer's `PC_BASE_ADDRESS` and
`PC_REGISTER_AMOUNTS` to the next task (`rocket-userspace` `mm_pack_regcmd` and `mm_seal_chain`,
`ROCKET_BATCH_SUBMIT=1`). The kernel sets `TASK_NUMBER=N` and advances `next_task_idx` to the
end, so the stock IRQ handler retires on one completion
(`patches/rocket/086-rocket-drv-batched-submit.patch`).

Chaining is chosen per job. `DRM_ROCKET_JOB_BATCHED` in `drm_rocket_job.flags` marks a job whose
regcmds self-chain. The field sits past the original struct, and the kernel copies the struct
with `copy_struct_from_user()`, so stock userspace is unaffected.

The kernel batches a job only where it carries the flag, has more than one task, and its
count fits the 12-bit `TASK_NUMBER`.
The module parameter `rocket_batch_submit` is a master switch. At 1, the default, the kernel
honors the flag. At 0 every job takes the per-task path. A kernel that reports interface 1.1
or later honors the flag, and `rocket_batched_submit_supported()` in `librocketnpu` reads that.
The two halves must agree: a flagged job with a gapped layout times out, recoverably.

The change removes `N−1` of every `N` per-task IRQ round-trips. That is the `wait` term, the
CPU-blocked-on-fence share that dominates prefill ([not-mac-bound.md](not-mac-bound.md)), and it
does not depend on the datatype. Measured on the RK1 at 600 MHz, governor `performance`
[HW sweep]:

| Case | `wait`, per-task -> chained | Throughput |
|---|---|---|
| `matmul_tiled_rocket 512 3840 4096`, 320 tiles in 5 batches of 64 | ~62 -> ~48 ms | ~94 -> ~96 GFLOP/s, best 99 |
| The same, kernel 7.1.1-1, interleaved warm pairs, 2026-06-25 | ~68 -> ~46 ms | ~85 -> ~93 GFLOP/s median |
| Resident weights, 512×4096×4096, 2026-06-29 | 556 -> 510 ms | not recorded |

The chained output has cosine 1.000000 against the per-task path, and the resident case is
numerically identical to its gapped run. The gain is modest because the matmul is compute- and
readback-bound at this operating point. It is larger on a path bound by submit overhead.

Every fp16 matmul submit path chains: the one-shot path, the K-accumulation path that prefill
takes, and the flash-attention batch. The K-accumulation path chains the tiles inside one K
step, and its K steps still fence in order. The resident int8 and int4 matmuls submit gapped
([../encodings/regcmd-task-model.md](../encodings/regcmd-task-model.md)).
`patches/rocket/BATCHED_SUBMIT_FINDINGS.md` holds the full mechanism and the disproven
kernel-only model.

### Scope

Batched submit pays where an op decomposes into many independent tasks. Examples are the
prefill matmul's `nMt·nNt` output tiles, a layer's independent Q/K/V projections, and the
multi-tile detection convs. It does not collapse a data-dependent chain into one kick. A transformer
prefill is sequential across layers. The `ROCKET_KACC` K-tiles ping-pong (each reads the
prior partial), so they still fence in order. The open path's realistic ceiling is therefore
below the vendor's "1 submit per inference", which is measured on feed-forward vision CNNs.

Batched submit composes with the per-submit levers above:

- IRQ affinity cuts the wakeup latency of a submit.
- IOMMU keep-attached cuts the kernel work in a submit.
- Batched submit cuts the number of submits.

The hardware substrate is the per-block register ping-pong that the next section describes
(two register banks per block). It lets the PC stage task `i+1`'s registers while task `i`
runs.

## In-core block-completion interrupt bitmap

Within a core, each pipeline block raises a completion bit in the NPU interrupt-status
register. These bits are distinct from the three per-core GIC IRQs above. The full 14-bit
bitmap is from phhusson's `rknpu-reverse-engineering` `hello2.c`, corroborated against the
BSP `rknpu` driver (see [SOURCES.md](../SOURCES.md)):

| bits | block |
|---|---|
| 0,1 | CNA feature group 0 / 1 |
| 2,3 | CNA weight group 0 / 1 |
| 4,5 | CNA CSC group 0 / 1 |
| 6,7 | CORE group 0 / 1 |
| 8,9 | DPU group 0 / 1 |
| 10,11 | PPU group 0 / 1 |
| 12 | DMA read error |
| 13 | DMA write error |

Two facts follow from the bitmap:

- **The PPU is a real, separately completing block** with its own IRQ bits (10/11), not a
  phantom in `npu_hw.h`. That is relevant to the PPU-as-de-tile-engine probe.
- **"group 0 / 1" means two register banks per block**, the ping-pong that the BSP exposes
  as `RKNPU_JOB_PINGPONG`. A block carries two register sets, so the next task's registers
  can be staged in the idle bank while the current task runs. That is the hardware basis for
  any register-staging or task-persistence attack on the dispatch floor.

On the `rocket` path the kernel owns the IRQ config, so this bitmap is informational. The
`int_mask`/`int_clear` fields live in the BSP `rknpu_task`, not in the regcmd that userspace
submits. The `0x14 PC_REGISTER_AMOUNTS` length and the `OP_ENABLE` block mask, which
userspace does drive, are a separate mechanism.

## Context-pool throughput

The measurement is from 2026-06-24 (7.1.0-1-arm64 at 600 MHz, `performance` governor), with
`ctx_pool_throughput`. The throughput pool is model 2 above: P independent contexts, each
with its own fd and entity. Each context runs a full "inference" of small prepacked fp16
matmuls (the 1×1-conv-as-matmul detection unit). Each call is a real A-pack -> submit ->
readback. Pool depth P is the swept variable. This model-2 throughput path (queue depth > core
count) is distinct from the model-1 `multicore_threads` probe, which packs once and loops a
bare submit (pure-NPU, saturates at 3).

**Spreading the pool's contexts across the A76 cores is load-bearing, and it is the caller's
responsibility, not automatic.** The library's auto-affinity pins worker `idx` to
`big[idx % n_big]`. A one-thread context (the natural pool unit) always has `idx == 0`. So
left to itself, every independent pool context pins to the same big core (cpu4), and their
host pack/readback serialize there [HW sweep].

| workload | pool affinity | best speedup vs P=1 | where |
|---|---|---|---|
| tiny 64×256×256 (submit/readback-bound) | colliding (default, all on cpu4) | ~2.1x | flat past P=3 |
| tiny 64×256×256 | spread across A76 (`ROCKET_CPU_AFFINITY=off` + caller pins each ctx) | ~3.9x | peak P=4 |
| large 512×1024×1024 (compute-bound) | spread across A76 | ~2.7x | still climbing at P=6 |

The sweep shows three things:

- **The pool beats the 3-core count on submit-bound work** (3.9x at P=4 > 3). With contexts
  on separate cores, one context's host pack/readback overlaps another's NPU compute, which
  is the "rknnpool" effect. P=4 (= the 4 A76 cores) is the best depth. P>=5 oversubscribes
  the big cluster and wobbles.
- **Compute-bound ops gain less** (~2.7x, approaching the 3-core NPU ceiling). There is
  little host bubble to hide, so the cores are the limit.
- **The ceiling is the host core and the submit path, not the IRQ core.** Applying the
  model-2 IRQ binding (3 NPU IRQs -> cpu5/6/7) raised the P=1 baseline (the −47%
  submit-floor win) and left the aggregate plateau unchanged. That confirms the
  colliding-pool cap was the shared host core, not IRQ servicing.

### Delegate recipe

For the throughput pool:

- Use one fd/context per pool instance, with P ~4 (= the A76 count).
- Set `ROCKET_CPU_AFFINITY=off` and pin each instance to a distinct big core.
- Apply the throughput IRQ binding (`npu_set_irq_affinity.sh throughput`).

The probe is `rocket-userspace/tests/ctx_pool_throughput.c`.

## uAPI contracts

A runtime conformance gate, `uapi_selftest_rocket`
(`rocket-userspace/tests/uapi_selftest_rocket.c`), pins the `drm_rocket_*` behaviors that the
library depends on. A kernel that drifts (a new-SoC port, a uAPI revision) then fails there
with a named diagnostic instead of mis-waiting deep in the matmul path. The contracts are
confirmed on 7.1.0-1-arm64 (driver reports version 0.0.0):

- **`CREATE_BO` IOVA bump-starts at 0x0.** The per-fd IOVA allocator hands out ascending,
  page-aligned addresses beginning at 0, so the first BO on a fresh fd legitimately has
  `dma_address == 0` (the next allocations follow at 0x1000, 0x2000, …). **`dma_address` is
  therefore not a validity sentinel.** Test `handle`/`ptr` instead. This confirms that the
  `iova_ceiling` probe's `va = 0x0 -> …` read above is the real allocator base, not an
  artifact. All BOs are page-aligned and stay in the low 4 GB (the 32-bit regcmd window)
  [HW sweep].
- **`PREP_BO.timeout_ns` is an absolute `CLOCK_MONOTONIC` deadline**, not a duration (the
  kernel runs it through `drm_timeout_abs_to_jiffies()`). In a live check, a raw
  `PREP_BO` with a deadline 1 ms in the past returns promptly with `-EBUSY` on an in-flight
  job, with no hang. The shim's relative->absolute conversion lets a generous relative wait
  complete the job. A kernel that regressed this to a relative duration would silently turn
  every job wait into an immediate `-EBUSY` poll. This canary catches that. The shim
  takes a relative timeout and converts it (see `rocket_bo_prep`).
- **`FINI_BO` always succeeds** (it syncs caches back for the device). A failure is a real
  error, not a routine return.
