# IOVA windows, multicore dispatch and submit overhead

Under `rocket`, each open file descriptor (fd) maps its BOs into its own 4 GB IOVA window
[HW sweep]. Each fd is also one scheduling entity, whose jobs run one at a time and in order
[HW sweep]. Software reaches more address space, and all three RK3588 cores, by opening one fd
per worker. The vendor `rknpu` BSP driver differs on the first of these: it maps every buffer
through one IOMMU domain per process. Each submit also has a fixed overhead. IRQ affinity and
IOMMU keep-attached reduce it, and batched submit reduces the number of submits.

**Conditions:** RK3588, mainline `rocket`, except where a section names the vendor `rknpu`
driver. A section's own Conditions block gives its kernel, NPU clock and date.

**Not characterized:** N-tile right-sizing on the per-channel planner, the `WARN` that a
client-reachable reset can raise, and a kernel-side completion poll that skips the IRQ.

| Section | Covers |
|---|---|
| [uAPI contracts](#uapi-contracts) | The `CREATE_BO`, `PREP_BO` and `FINI_BO` behaviors that software depends on |
| [Per-fd IOVA windows](#per-fd-iova-windows) | The 4 GB window per fd, and what sets its size |
| [Resident weights in the per-fd window](#resident-weights-in-the-per-fd-window) | Scratch, the M-independent weight scatter, and N-tile sizing |
| [The vendor `rknpu` driver's shared IOVA domain](#the-vendor-rknpu-drivers-shared-iova-domain) | One domain per process, and the address space its generic mapping path loses |
| [Multicore dispatch through worker fds](#multicore-dispatch-through-worker-fds) | One scheduling entity per fd, and one fd per core |
| [Worker fds in `librocketnpu`](#worker-fds-in-librocketnpu) | How the library's matmul and convolution fan out across fds |
| [Multicore models](#multicore-models) | Splitting one inference, against running independent inferences |
| [Small ops and multicore](#small-ops-and-multicore) | Single-tile layers |
| [IRQ affinity](#irq-affinity) | Which CPU services the completion IRQ |
| [Busy-polling the completion fence](#busy-polling-the-completion-fence) | `ROCKET_BUSY_POLL` |
| [Per-job IOMMU cost](#per-job-iommu-cost) | The IOMMU attach and detach on every job, and the reset before detach |
| [Batched submit](#batched-submit) | Many tasks in one kick |
| [In-core block-completion interrupt bitmap](#in-core-block-completion-interrupt-bitmap) | The per-block completion bits |
| [Context-pool throughput](#context-pool-throughput) | Independent contexts spread across the A76 cores |

## uAPI contracts

**Conditions:** kernel 7.1.0-1-arm64, where the driver reports version 0.0.0.

Software that drives `rocket` depends on three behaviors of the `drm_rocket_*` ioctls:

- **`CREATE_BO`.** The per-fd IOVA allocator hands out ascending, page-aligned addresses
  beginning at 0x0, so the first BO on a fresh fd legitimately has `dma_address == 0` (the next
  allocations follow at 0x1000, 0x2000, …). All BOs are page-aligned and stay in the low 4 GB,
  the 32-bit regcmd window [HW sweep]. `dma_address` is therefore not a validity sentinel.
  Software must test `handle` or `ptr` instead.
- **`PREP_BO`.** `PREP_BO.timeout_ns` is an absolute `CLOCK_MONOTONIC` deadline, not a
  duration: the kernel runs it through `drm_timeout_abs_to_jiffies()`. A raw `PREP_BO` with a
  deadline 1 ms in the past returns promptly with `-EBUSY` on an in-flight job, with no hang.
  `librocketnpu`'s `rocket_bo_prep` takes a relative timeout and converts it, so a generous
  relative wait completes the job.
- **`FINI_BO`.** It always succeeds, and syncs caches back for the device. A failure is a real
  error, not a routine return.

## Per-fd IOVA windows

Each fd has its own independent 0…4 GB IOVA window [HW sweep, source-confirmed]. The regcmd
encodes BO addresses as 32-bit fields (e.g. `weights_dma & 0xFFFFFFFF`), so every BO that a job
references must lie in the low 4 GB of its fd's address space.

On the RK3588 the 4 GB limit is the regcmd-field encoding, not the bus. The RK3588 NPU
AXI/IOMMU is 40-bit, per the upstream `rocket` RK3568 RFC, which contrasts RK3568's "32-bit NPU
AXI/IOMMU (vs 40-bit)" ([LKML](https://lkml.iu.edu/2605.3/10672.html)). The bus can physically
reach >4 GB, and the 32-bit address field in the regcmd caps a single context at 4 GB. On the
RK3568 the 4 GB is a bus limit: the NPU AXI and the IOMMU page-walker/DTE are both 32-bit, so
its page tables need `GFP_DMA32`.

Two fds opened together each climb `va = 0x0 -> 0x100000000` independently: 8 GB in total, with
the VAs repeating per fd. The consequences:

- N worker fds give ~N × 4 GB of usable IOVA. The 5-worker resident context has ~20 GB, enough
  for ~80-90% of a 22 GB Gemma-4-12B F16 resident, and for all of any smaller or quantized
  model.
- BO allocation is lazy. Reserving 8 GB of IOVA left RAM flat (~595 MB), and only the data that
  software packs commits physical RAM.
- The window is usually not the binding limit. The f16 residency path fills it at 5 workers on
  a 12B F16 at `-p 512`, placing 286 of 328 weights. At `-p 2048` the same path reaches the RAM
  reserve first, at the same placement. The MoE path never reaches the window, because its
  admission charges the GGUF source against RAM and only the int8 codes against IOVA.
  `benchmarks.md` has the arithmetic, and `data/tuning-matrix.md` the per-shape measurements.

## Resident weights in the per-fd window

### Per-tensor scratch

A "`ROCKET_CACHE_MB=12000` exceeds 32 bits" crash comes from per-tensor scratch, not from the
4 GB window. Each resident weight has its own compute scratch, ~2x the weight bytes. Sharing
scratch per (worker-fd, shape), and keeping only the weight tiles per-tensor, lifts a 12B F16
model from ~30%-resident to ~80-90%-resident within the same IOVA.

### M-independent resident weight scatter

The resident weight scatter is M-independent. A weight's scattered tile positions are fixed by
the N-slice split and the K/N tiling (`Nt`, `Kt`) alone. The M dimension sets only how many
input and output tiles stream through, never where a weight byte is placed. The tiling can
therefore be planned at a canonical M (`MAX_TILE`) instead of the actual row count. One resident
weight then serves every M with no re-pack. Warmup-M packs once, and any later prefill M (down
to a single short-prompt tile) reuses it.

Three resident paths are bit-exact across M, packed at M=512 and reused at
M=512/256/768/64/8 [HW sweep]:

- fp16
- int8, because int32 K-accumulation is exact for any K-tiling
- Group-wise int4, because the K-tile is already pinned to `group` and the host fp32
  K-accumulation order is per-`(m,n,group)`

Without this, a warmup M that differs from the prefill M costs a full model re-pack. On a 12B
model, that re-pack is a multi-minute stall for a single short prompt. `librocketnpu` detects a
genuine tiling mismatch (e.g. a `ROCKET_MM_*` override changing the tiling between pack and
compute) and returns `-2`, so the caller re-packs rather than miscomputing.

A caller with many short, varied lengths asks for the canonical plan outright:
`rocket_ctx_create_ex(n, ROCKET_CTX_TILING_CANONICAL)`. A small M then keeps the M=256 `Kt` and
`Nt`, so one pack serves every length. The cost falls at the small end, where the per-M plan
would have picked larger K tiles. It is 1.16-1.42x at M 48-128 [HW sweep, RK1, ~816 MHz,
2026-09-27].

### N-tile sizing on a worker slice

Software must size the N-tile to the worker's slice, not to the tile cap. N is split across the
worker fds, so each worker plans its tiling on a slice. A slice is not a multiple of the
256-column tile cap. An N-tile defaulted to the cap stores a nearly empty tail tile.

At the gpt-oss expert shape, `N=2880` over 5 workers gives a 576-wide slice. Three 256-wide
tiles then store 768 columns to hold 576: a 35% memory overhead on every resident weight, for the
process lifetime.

The cap fixes the tile count (`nNt = ceil(N/max_tile)`), and any tile >= `ceil(N/nNt)` reaches
that same count. Take the smallest one, here `ceil(576/3) = 192`. The change costs nothing. It
gives the same three tiles, hence the same task count, the same dispatch and the same real DMA.
It stores exactly 576 columns. Measured per resident gpt-oss expert, the weight goes
10.70 -> 8.07 MiB (1.35x -> 1.02x of the logical `N·K` int8 bytes) [HW sweep].

Round the 32-column alignment up, not down. Rounding down can drop the tile below
`ceil(N/nNt)` and add an extra tile, a real dispatch cost that saves nothing.

The overhead falls on residency, which is the whole purpose of a native-quant MoE expert cache.
It is nearly invisible on a dense model (one weight per tensor), and decisive across thousands
of experts.

> [!CAUTION]
> Shrinking `Nt` frees CBUF banks, which can let `Kt` grow. A grown `Kt` re-tiles K, which
> changes the host K-accumulation order and breaks the resident-vs-one-shot bit-exactness. On
> the group-wise planner, where `Kt` can never exceed the quant group, the change is safe. On
> the per-channel planner the same change moves `Kt` 640 -> 768.

## The vendor `rknpu` driver's shared IOVA domain

The vendor `rknpu` BSP driver maps every buffer through one IOMMU domain shared across the whole
process. A second fd adds no address space, and two processes share the same budget. The multi-fd
strategy of §"Per-fd IOVA windows" therefore does not transfer to it. The per-fd window is a
property of `rocket`, not of the silicon. Freshly booted, the vendor driver serves about the same
total: ~3.9 GB in buffers of any size from 16 MB to 256 MB, against `rocket`'s 4.00. The two
drivers start equivalent, and differ only in how that budget is shared and in what happens to it
afterwards.

The vendor driver also allocates from the top of the domain down. BOs are placed at
`0xffe00000` and `0xffff0000` on a 6.1.172 vendor kernel with `rknpu` 0.9.8 [HW sweep, vendor
RK1]. `rocket` allocates from 0 upward.

On that driver, workloads that map through the kernel's generic DMA-mapping path consume the
window, and the driver's own allocator cannot reclaim it. The consumer is the mapping path, not
elapsed time. One `llama.cpp` 2048-token prefill (Llama-3.2-3B F16) with
`RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT` clear costs the shared domain 5-11 of its 31 buffers of
128 MB in 153 s. The loss outlives the process. With that flag set, which is the
`librocketnpu` provider's default, the identical run costs zero [HW sweep, RK3588, `rknpu` 0.9.8,
2026-08-25].

The domain outlives every process that used it, so the loss persists until a reboot or a
generic-path allocation to refusal (§"The rcache flush on the generic path"). Detaching and
re-attaching the device (the driver's own soft reset) re-uses the same domain object and does
not rebuild the allocator.

> [!NOTE]
> `RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT` exists only from `rknpu` 0.9.7 (2024-04-24). An older
> driver accepts the bit and ignores it, and allocates on the generic path with nothing
> reported [source-confirmed, `rknpu_ioctl.h` and `rknpu_gem.c` across 0.8.0-0.9.8]. Software
> must read the driver version before relying on the flag.

The flag is bit 10, added by that release. The driver declares `RKNPU_MEM_MASK` and never
validates against it. The driver passes `args->flags` straight to `rknpu_gem_object_create()`
and tests individual bits. On 0.9.6 and earlier the allocation therefore takes the generic path
while the flag reads as set [source-confirmed].

### The IOVA rcache mechanism

The loss comes from the kernel's IOVA rcache, which the driver reaches by mixing two allocators
on one domain. Which allocator each path uses decides where a freed range goes:

- With the flag set, the driver takes the tight path. It calls `alloc_iova()` / `free_iova()`,
  which are the rbtree directly.
- With the flag clear, the driver takes the generic path. The mapping falls to the generic
  `dma_map_sg()`, hence `alloc_iova_fast()` / `free_iova_fast()`, which go through the per-CPU
  IOVA rcache.

On a free, `free_iova_fast()` parks the range in a per-CPU magazine or the global depot, and
falls through to `free_iova()` only when that fails. `alloc_iova()` never consults the rcache,
so address space freed on the generic path becomes unreachable to the driver's own path. The
rcache belongs to the `iova_domain`, which here is one domain shared process-wide, so the loss
outlives the process that caused it. The loss needs the two allocators mixed on a single
domain. Either one used consistently is sound.

`iova_rcache_insert()` accepts only sizes up to `2^(IOVA_RANGE_CACHE_MAX_SIZE-1)` = 32 pages =
128 KB. The bound makes the effect reproducible, and hides it from a probe that frees only larger
buffers (§"Measuring the loss"). Churning 4400 buffers through the generic path, and freeing every
one, leaves this capacity, in buffers of each size [HW sweep]:

| churn size | path | 16 MB | 64 MB | 128 MB | 192 MB | consumed |
|---|---|---|---|---|---|---|
| fresh | n/a | 255 | 63 | 31 | 21 | n/a |
| **128 KB** | generic | 221 | 55 | **27** | 18 | **34 / 8 / 4 / 3** |
| 256 KB | generic | 255 | 63 | 31 | 21 | 0 |
| 132 KB (one page over) | generic | 255 | 63 | 31 | 21 | 0 |
| 128 KB | tight | 255 | 63 | 31 | 21 | 0 |

**Conditions:** RK3588, `rknpu` 0.9.8, 2026-08-26, a fresh domain per configuration, capacity
measured on the tight path.

The effect is path-selective, and size-selective at exactly the 32-page boundary. One page over,
it vanishes, while twice the address space at 256 KB costs nothing.

### The rcache flush on the generic path

The kernel flushes the rcache only from the generic path's allocator. When `alloc_iova_fast()`
cannot satisfy a request, it flushes every online CPU's magazines and the whole depot, then
retries. The path that fills the cache can therefore always empty it again. Allocating to refusal
on the generic path repairs the domain outright [HW sweep]:

| step | measuring path | 128 MB BOs | address span |
|---|---|---|---|
| fresh | tight | 31 | `0x8000000`-`0xffffffff` |
| fresh | generic | 31 | same |
| after a 128 KB generic churn | tight | **27** | `0x5a00000`-`0xdd9fffff` |
| same domain | **generic** | **31** | `0x8000000`-`0xffffffff` |
| after that generic pass | tight | **31** | recovered |

**Conditions:** RK3588, `rknpu` 0.9.8, 2026-08-26, one boot, addresses recorded.

The 550 MB that the degraded tight path cannot reach sits in one block above `0xdda00000`, which
is the rcache ceiling to the megabyte. It comes back whole. The workload-shaped churn behaves the
same: 8 CPUs across all six cached orders take 255/63/31/21 to 213/53/26/17, and one generic pass
restores 255/63/31/21.

`alloc_iova()` has no such flush. The flush helpers `free_cpu_cached_iovas()` and
`free_global_cached_iovas()` are static to `iova.c` and unexported, so a driver allocating
through `alloc_iova()` cannot ask for the parked space back. The tight path returns `-ENOMEM`
while hundreds of megabytes sit in a cache that it cannot see or drain. The generic path, which
causes the loss, can always recover it, and the tight path, which causes none, is the one that
fails.

### The vendor default path

The generic path is the vendor default. `RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT` is an opt-in bit at
buffer creation. A userspace that does not set it, which is the stock path, takes the generic
path and loses address space. The `librocketnpu` provider sets the flag by default, and so
allocates on the tight path.

A large allocation failure on the generic path is transient rather than structural. A retry at a
smaller size is usually served, which is what `librocketnpu` does for the chained attention path.

> [!NOTE]
> On the vendor driver, a board's allocation history, not its uptime, is the variable in any
> allocation-sensitive measurement. Probe the domain before recording a number as a property of
> the driver [HW sweep, RK3588, `rknpu` 0.9.8 vs mainline `rocket` 1.1.0].

### Measuring the loss

Four properties of the two paths mislead a measurement of the loss:

- A probe whose frees are all above the 128 KB rcache bound measures zero loss. None of its frees
  can enter the cache, at any flag value.
- A probe that allocates one size cannot separate the paths' loss. On an empty domain both
  paths return identical counts and identical addresses at the power-of-two sizes 16, 32, 64
  and 128 MB. At a size that is not a power of two they differ by the fixed alignment property
  in the last item.
- A path that repairs the domain and a path that is blind to the loss return the same number.
  Separating them takes a third measurement: a re-read on the path that showed the loss, after
  the other path has run. Without that step a repair is recorded as blindness.
- Mixed inside one before/after comparison, two fixed path properties read as degradation. The
  generic path allocates size-aligned, so a fresh domain serves it 15 buffers of 192 MB against
  the tight path's 21 (192 MB rounds its alignment to 256 MB). The tight path also packs below
  the generic path's first address.

Take both halves of any comparison on one path.

## Multicore dispatch through worker fds

`rocket` exposes the 3 NPU cores through the scheduler topology, not a submit flag:

- The driver creates one `drm_sched` per core and one scheduling entity per fd. A DRM entity
  stays on one core while it has a job queued or running (the in-order guarantee).
- Once its last job has finished, the next one goes to the core with the lowest score, the
  first in index order on a tie. The entity's own presence counts toward its current core's
  score. A lone fd that waits on each job therefore alternates between cores 0 and 1
  [HW sweep, `rocket` 1.3.0, 2026-09-27, source-confirmed: `drm_sched_entity_select_rq`]. It
  never reaches core 2, and its jobs still run one at a time.
- Driving N threads, each with its own fd and entity, makes the kernel dispatch across all 3
  cores. The measured rate is 39.5 -> 84.3 -> 116.1 -> 120.9 jobs/s at 1/2/3/4 threads
  (1.0 / 2.13 / 2.94 / 3.06x) [HW sweep, corroborated by Tomeu's blog].

Throughput rises one thread past the core count (T=4 > T=3, and T=5 is the knee). The extra
worker fills the idle bubbles left by each worker's serial pack->submit->readback cycle. The
common "rknnpool" uses the same pattern: N worker contexts round-robin over 3 cores, with queue
depth > core count.

The proprietary RKNN stack exposes core selection through a `core_mask` submit field. Mainline
`rocket` has none, and software reaches the cores through multiple entities.

## Worker fds in `librocketnpu`

`librocketnpu`'s matmul splits N (output channels) across worker threads, each with its own fd.
Each worker runs the unchanged single-fd matmul on a contiguous column slice and scatters its
dense result into the strided output. The library fans resident weights across the worker fds,
so each fd's slice fits its own 4 GB window. The default worker count is 5 (the measured knee).
See the `rocket-userspace` library's `rocket_matmul_mt.c` and `rocket_prepacked*.c`.

The native int8/uint8 direct conv uses the same pattern (`rocket_conv_pool` /
`rocket_conv2d_int8_mt` in `rocket_conv.c`). A conv's tiling already decomposes it into
independent OC-group × oh-band × ow-band tiles, each writing a disjoint region of the output.
The tiles therefore fan across a pool of N worker fds, each with its own resident
`rocket_conv_ctx` (BO pool) and scratch. The pooled conv is bit-identical to the single-fd
`rocket_conv2d_int8` (same tiles, same single jobs, dispatched on different cores). It falls back
to serial for single-tile convs.

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
`subcore_task[5]` partition inside one submit (the vendor `core_mask` `0_1`/`0_1_2`). Multi-fd
entities (the N-split above) give the same outcome. `librocketnpu` implements it for the prefill
matmul (5 workers) and the detection conv pool, and the prefill readback cost is already the
post-3-core figure. Mainline `rocket` lacks the vendor's in-one-submit interface and does not
need it.

### Model 2: multi-instance throughput

N independent inferences run concurrently (queue depth > core count), so another context fills
each context's serial pack->submit->readback bubble. go-rknnlite measured EfficientNet-Lite0 at
pool-of-9: 7.9 -> 1.65 ms (4.8x throughput, latency unchanged).

Frigate's multi-camera detection (one context per stream) uses this model, and the
`tflite-rocket` delegate supports it. The delegate's `rocket.py` sets `ROCKET_CPU_AFFINITY` per
detector process (one A76 each, `nthreads=1`), so Frigate's one-process-per-detector model
spreads N independent contexts across the big cluster. Measured end to end through the
delegate, P=1->4 gives 1.00 / 2.17 / 3.11 / 3.56x.

The mainline kernel's `drivers/accel/rocket/rocket_job.c` confirms the mechanism: a per-core
`drm_sched` plus a per-fd `sched_entity` initialized with all core schedulers gives automatic
load-balance of concurrent jobs.

## Small ops and multicore

A layer smaller than the multicore task-allocation granularity runs on a single core. The limit
is the multi-tile-only one in §"Worker fds in `librocketnpu`": small pointwise convs do not fan
out. The detection path therefore needs the throughput pool (model 2), not more cores.

For such a layer, the per-core setting with an effect is IRQ affinity. Set CPU/DDR/NPU to max
frequency, pin the app to a CPU big core, and bind the three NPU interrupts to that big core
(`/proc/irq/<npu-irq>/smp_affinity_list`). App `taskset` alone leaves the per-submit overhead
unchanged. On llama.cpp prefill, whole-process `taskset 0xf0` is a host-side 1.05-1.13x
[HW sweep, RK1, ~816 MHz] ([data/tuning-matrix.md](data/tuning-matrix.md) §"Pinning to the
A76s"). On a path bound by submit overhead, the IRQ binding gives a large reduction
(§"IRQ affinity").

## IRQ affinity

**Conditions:** RK3588 (Turing RK1), kernel 7.2.9-1-arm64, the image's `rocket` 1.3.0, which keeps
the IOMMU domain attached between jobs, NPU clock ~816 MHz, CPU governors pinned to `performance`.
A tiny 1-task job at the `8 64 16` shape, 2026-10-12.

The default IRQ affinity services the NPU completion IRQ on an A55 little core. Binding the IRQ
to an A76, and running the waiter on that core, reduces the per-submit overhead.

On kernels 7.1.0 and 7.2.9 the NPU GIC IRQs are 69 = `fdab0000.npu`, 70 = `fdac0000.npu` and 71 =
`fdad0000.npu`. Their GIC numbers are 142/143/144, not the 110/111/112 sometimes quoted
elsewhere. The numbers are SoC- and kernel-specific, so read `/proc/interrupts` on the target
kernel. The RK3588 CPU map is cpu0-3 = A55 little at 1.8 GHz and cpu4-7 = A76 big at 2.4 GHz.

The default IRQ affinity is the all-CPUs mask `0-7`. GICv3 routes a multi-CPU level IRQ to the
lowest CPU in the mask, which is cpu0, an A55 little core. By default, the completion handler and
the waiter wakeup therefore both run at 1.8 GHz. The median per-submit overhead [HW sweep]:

| config | IRQ affinity | app taskset | blocking median µs/submit | busy-poll median µs/submit |
|---|---|---|---:|---:|
| default | `0-7` (-> services on cpu0, A55) | none | 29.75 | 16.04 |
| IRQ on a big core, app floats | cpu7 | none | 25.67 | 14.88 |
| IRQ + app co-located on one big core | cpu7 | cpu7 | 14.58 | 14.29 |

The table shows two results:

- Co-locating the waiter with the IRQ on one A76 gives most of the reduction, 29.75 -> 14.58 µs.
  Paired within each pass, the ratio is 0.49. Moving the IRQ alone, with the app floating, gives
  29.75 -> 25.67 µs.
- Busy-polling the fence (§"Busy-polling the completion fence") reaches about the co-located
  figure from either of the other bindings, 14.9-16.0 µs.

With the IRQ left on the A55, pinning the app alone left the overhead unchanged on kernel 7.1.0
[HW sweep, 2026-06-22]. The IRQ binding comes first, and co-location adds to it.

**Not characterized:** app pinning alone, and big cores other than cpu7, on this module.

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

`rocket-userspace/tools/npu_set_irq_affinity.sh` applies the recommended binding. The binding
is a runtime system setting, with no driver or library change. It complements the IOMMU
keep-attached patch (§"Per-job IOMMU cost"), which removes ~20 µs of kernel work per submit.
The binding removes the little-core wakeup latency per submit. On a module that keeps the IOMMU
domain attached, the co-located binding measures 14.6 µs per submit, against 29.8 µs at the
default affinity (§"IRQ affinity").
The binding reduces the overhead on every path with many small submits, and it is flat on a
single big tiled prefill matmul (one submit per batch of up to 64 tiles).

## Busy-polling the completion fence

**Conditions:** RK3588, kernel 7.1.0-1-arm64, NPU clock ~816 MHz, A76 governors pinned to
`performance`, a `64 256 512` submit, 2026-06-29.

Busy-polling is redundant with the IRQ-affinity binding. A blocking wait (`PREP_BO` with a real
timeout) sleeps the waiter until the kernel signals the job fence from its threaded completion
IRQ. Each wait therefore costs an IRQ delivery plus a scheduler round-trip to re-run the waiter.
The option `ROCKET_BUSY_POLL=<µs>` instead spins on a non-blocking completion check for up to
that budget, then falls back to the blocking wait. The spinning waiter stays runnable, so it
returns within one check of the fence signalling.

The completion check is `PREP_BO` with a zero deadline (§"uAPI contracts"), the only completion
check that the mainline uAPI exposes. Every poll therefore also `dma_sync`s the output BO. The
option is implemented in `rocket_bo_prep()`.

Busy-polling does not skip the IRQ. Only the kernel IRQ handler signals the mainline fence, and
userspace has no MMIO view of the PC `INTERRUPT_RAW_STATUS` register. A userspace spin can
therefore remove only the waiter-side wakeup, not the interrupt. Skipping the IRQ itself would
need a kernel-side poll (a `patches/rocket` change).

> [!NOTE]
> Pin the A76 governors to `performance` before timing busy-polling. Unpinned, the idle A76
> parks between submits and swamps the signal.

Blocking wait against busy-polling, in one process [HW sweep]:

| IRQ affinity | app core | blocking median | busy-poll Δ median / mean |
|---|---|---|---|
| default `0-7` (A55) | A76 | 109 µs | −4.3 % / −4.7 % |
| -> cpu7 (A76) | other A76 | 96 µs | −1.8 % / +0.9 % (wash) |
| -> cpu7 (A76) | co-located | 97 µs | +0.3 % / −0.9 % (none) |

Busy-poll's best-case `min`/`p10` improves ~4-5 µs at every configuration (the removable
wakeup). The median gain appears only when the completion IRQ is on a little A55 core.
Busy-polling and the IRQ-affinity binding therefore reduce the same waiter-wakeup term. With the
IRQ moved to an A76, which costs no core, busy-polling adds nothing. Even on the default A55, the
IRQ-affinity binding alone (blocking 103->89 µs mean, −14 %) beats busy-poll's −4.7 % without
occupying a core to spin.

On the tiny `8 64 16` shape, busy-polling removes the same term. At the default affinity it
takes the median from 29.75 to 16.04 µs. With the waiter co-located it changes nothing, 14.58
against 14.29 µs [HW sweep, RK1, kernel 7.2.9, `rocket` 1.3.0, 2026-10-12].

`ROCKET_BUSY_POLL` is opt-in and off by default. It is a single-stream latency fallback for a
system where the completion IRQ stays on its default little-core affinity and cannot be
re-bound. Where it can be re-bound, `npu_set_irq_affinity.sh latency` is cheaper (no spinning
core) and reduces the same term.

> [!NOTE]
> Do not enable busy-polling under the throughput pool. The spin occupies a core that the pool
> needs for another stream.

## Per-job IOMMU cost

**Conditions:** RK3588, kernel 7.1.0-1-arm64, NPU clock ~816 MHz, stock `rocket` against
patch-5, 2026-06-22.

Stock `rocket` calls `iommu_attach_group()` in `rocket_job_run()` on every `drm_sched` job. It
calls `iommu_detach_group()` in `rocket_job_handle_irq()` on every completion, and on reset.
Each call toggles the rk_iommu stall/force-reset/paging handshake. On the RK3568 that handshake
times out on the idle NPU MMU (the upstream bug fixed by RFC patch 5/9). On the RK3588 it
completes silently and still costs latency on every submitted job.

The attach+detach handshake costs ~15-20 µs per `drm_sched` job. The cost is per submit, not
per task:

- A tiny 1-task job on one fd: median 54->34 µs/submit and min 39->23 µs with keep-attached, so
  ~20 µs (~38%) is the IOMMU term.
- 64-task jobs: −17 to −18 µs per job (J=1: 7.68->7.50 ms over 10 jobs, J=3: 23.56->23.04 ms
  over 30 jobs).
- A `512×3840×4096` tiled matmul (320 tiles in 5 jobs of 64): flat. The cost amortizes to nothing
  on a single tiled prefill matmul, and dominates streams of small jobs.

RFC patch 5 keeps the per-context domain attached across same-fd jobs. It tracks
`attached_domain` in `struct rocket_core`, swaps only on a context change, holds a `kref`, and
detaches at teardown or after reset. It is available as
`patches/rocket/083-rocket-drv-iommu-keepattach.patch`. The patch is the per-job counterpart of
the per-tile dispatch optimizations (tile fusion, no-alloc submit): it reduces the per-job
overhead itself rather than the job count.

It reduces wall-clock time on the paths bound by per-submit overhead:

- Decode GEMV
- The small detection convs and 1×1s
- Multi-fd contention
- The throughput pool (model 2)

See [not-mac-bound.md](not-mac-bound.md) §"Per-submit overhead reductions".

### Reset before IOMMU detach

**Conditions:** Turing RK1, kernel 7.1.1, patches `081`-`087` out-of-tree, 30 client-requested
DMA faults per configuration, 2026-07-31.

`rocket_reset()` detaches the IOMMU group explicitly, with or without keep-attached. An
unprivileged client can reach this detach: the timeout handler `rocket_job_timedout()` is its
only client-reachable caller. The group is therefore detached
while the core is still wedged mid-DMA, with an unacknowledged fault in its MMU. In that state
rk_iommu cannot stall a bank. Each faulting job logs two
`rk_iommu … Enable stall request timed out` `dev_err`s, one per MMU bank (the RK3588 NPU MMU
has two).

`rk_iommu_disable()` swallows that error. The same handshake runs again inside
`rk_iommu_enable()` when the IOMMU core puts the group back on its default domain. The RK1's
default domain type is *Translated*, so an rk_iommu attach runs. There the error is returned, and
`__iommu_group_set_core_domain()` turns it into a `WARN`: a taint, a backtrace, and a panic
under `panic_on_warn`. The `WARN` is intermittent, because it needs the second handshake to fail
as well.

> [!CAUTION]
> `drm_rocket_task.regcmd` is a raw NPU IOVA written into the PC block unvalidated, and
> `/dev/accel/accel0` is group `render`. One field pointed at an unmapped address reaches all of
> the above: the timeout, the failed stall and the `WARN`.

The driver must reset the core before detaching, so the handshake has a quiesced master to
stall. The reset also wipes the MMU page-table base, which is already why the domain must be
dropped. The change is `patches/rocket/088-rocket-drv-reset-before-iommu-detach.patch`, and that
order is also where the upstream RFC puts the detach.

With the patch, the stall timeouts are 0, against 60 without it, and either module logs the same
30 `NPU job timed out` lines [HW sweep].

The RK3576 reaches the same detach and logs the same stall timeouts, and it does not `WARN`.
The two SoCs differ in the consequence, not in the trigger.

## Batched submit

Batched submit runs many tasks in one hardware kick. Mainline `rocket` submits one task per
kick. Its `rocket_job_run()` programs a single task's regcmd into `PC_DATA_ADDR` and sets
`PC_TASK_CON` `TASK_NUMBER(1)`. It re-arms the next task only on each completion IRQ
(`next_task_idx++` in the IRQ handler). A matmul tiled into `nMt·nNt·nKt` tiles therefore costs
one kick and one completion IRQ per tile. The submit, the job fence and the waiter wakeup are
per job, and `librocketnpu`'s matmul puts up to 64 tiles in a job.

The BSP `rknpu` driver fires a whole task list in one kick, through a self-chain
[HW sweep, source-confirmed: `rocket-userspace/src/rocket_matmul.c`]. The N tasks' regcmds are
laid contiguously in one buffer. Each task ends in an `OP_ENABLE`, and its trailer contains the
next task's address (an embedded `PC_BASE_ADDRESS` op) and stream length (its
`PC_REGISTER_AMOUNTS` op). The kernel programs only the first task's
`PC_DATA_ADDR`/`PC_DATA_AMOUNT`, sets `PC_TASK_CON.TASK_NUMBER = N`, and kicks once. The PC
executes one `OP_ENABLE` per task and follows each `PC_BASE_ADDRESS` link to the next. With
`TASK_NUMBER` set, a single completion IRQ fires after the last task, up to
`max_submit_number` = 4095/chunk on the RK3588.

Software must put the next task's address in each trailer's `PC_BASE_ADDRESS` op, as well as
its length in `PC_REGISTER_AMOUNTS`. A contiguous layout with only the amount op, or with a
zeroed trailer, runs task 0 and stops.

> [!CAUTION]
> `PC_TASK_STATUS` does not hold the completed task count. At IRQ time it reads `0x0000f000`
> (`& 0xfff == 0`). Software must retire a chained job on the single completion IRQ that
> `TASK_NUMBER` sets up, not on a `PC_TASK_STATUS` read.

`PC_TASK_DMA_BASE_ADDR` is not a kernel-walked task list. The BSP sets it to the regcmd buffer
in one example and to `0` in another, so it is don't-care for the stream. The `rknpu_task` array
exists only so the BSP kernel can read per-task fields CPU-side.

An independent FOSS RE of both stacks gives the end-to-end size of the gap. The proprietary path
issues ~63 IOCTLs / 1 submit per inference, where Mesa's Teflon driver on `rocket` issues ~634
IOCTLs / 10 submits. The proprietary path therefore makes ~10x fewer kernel transitions
[source: an independent FOSS RE of both stacks (orangepi5plus-npu), see
[SOURCES.md](../SOURCES.md)].

### The `rocket` implementation

A `rocket` multi-task job runs in one kick under two coordinated changes. Userspace packs the
regcmds contiguously and links each trailer's `PC_BASE_ADDRESS` and
`PC_REGISTER_AMOUNTS` to the next task (`rocket-userspace` `mm_pack_regcmd` and `mm_seal_chain`,
`ROCKET_BATCH_SUBMIT=1`). The kernel sets `TASK_NUMBER=N` and advances `next_task_idx` to the
end, so the stock IRQ handler retires on one completion
(`patches/rocket/086-rocket-drv-batched-submit.patch`).

Chaining is chosen per job. `DRM_ROCKET_JOB_BATCHED` in `drm_rocket_job.flags` marks a job whose
regcmds self-chain. The field sits past the original struct, and the kernel copies the struct
with `copy_struct_from_user()`, so stock userspace is unaffected.

The kernel batches a job only where it has the flag, has more than one task, and its count fits
the 12-bit `TASK_NUMBER`. The module parameter `rocket_batch_submit` is a master switch. At 1,
the default, the kernel honors the flag. At 0 every job takes the per-task path. A kernel that
reports interface 1.1 or later honors the flag, and `rocket_batched_submit_supported()` in
`librocketnpu` reads that. Software must set the flag only on a job whose regcmds self-chain: a
flagged job with a gapped layout times out, recoverably.

The change removes `N−1` of every `N` per-task IRQ round-trips. They fall in the `wait` term,
the CPU-blocked-on-fence share that dominates prefill ([not-mac-bound.md](not-mac-bound.md)),
and the saving does not depend on the datatype. Per-task against chained [HW sweep]:

| Case | `wait`, per-task -> chained | Throughput |
|---|---|---|
| `512×3840×4096` tiled matmul, 320 tiles in 5 batches of 64, kernel 7.1.0-1, 2026-06-24 | ~62 -> ~48 ms | ~94 -> ~96 GFLOP/s, best 99 |
| The same, kernel 7.1.1-1, interleaved warm pairs, 2026-06-25 | ~68 -> ~46 ms | ~85 -> ~93 GFLOP/s median |
| Resident weights, 512×4096×4096, 2026-06-29 | 556 -> 510 ms | not recorded |

**Conditions:** RK1, NPU clock ~816 MHz, governor `performance`. The two `512×3840×4096` rows
use the symmetric tiling of the time (Nt 256, Kt 384: 320 tiles).

The chained output has cosine 1.000000 against the per-task path, and the resident case is
numerically identical to its gapped run. The gain is modest because the matmul is compute- and
readback-bound at this operating point. It is larger on a path bound by submit overhead.

Every fp16 matmul submit path chains: the one-shot path, the K-accumulation path that prefill
takes, and the flash-attention batch. The K-accumulation path chains the tiles inside one K
step, and its K steps still fence in order. The resident int8 and int4 matmuls submit gapped
([../encodings/regcmd-task-model.md](../encodings/regcmd-task-model.md)).

### Scope

Batched submit helps where an op decomposes into many independent tasks.
Examples are the prefill matmul's `nMt·nNt` output tiles, a layer's independent Q/K/V
projections, and the multi-tile detection convs. It does not collapse a data-dependent chain into
one kick. A transformer prefill is sequential across layers. The `ROCKET_KACC` K-tiles ping-pong
(each reads the prior partial), so they still fence in order. The `rocket` path's realistic
ceiling is therefore below the vendor's "1 submit per inference", which is measured on
feed-forward vision CNNs.

Batched submit composes with the per-submit optimizations above:

- IRQ affinity cuts the wakeup latency of a submit.
- IOMMU keep-attached cuts the kernel work in a submit.
- Batched submit cuts the number of submits.

The hardware basis is the per-block register ping-pong that the next section describes (two
register banks per block). It lets the PC stage task `i+1`'s registers while task `i` runs.

## In-core block-completion interrupt bitmap

Within a core, each pipeline block raises a completion bit in the NPU interrupt-status register.
The bits are distinct from the three per-core GIC IRQs above. The full 14-bit bitmap is from
phhusson's `rknpu-reverse-engineering` `hello2.c`, corroborated against the BSP `rknpu` driver
(see [SOURCES.md](../SOURCES.md)):

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

The bitmap shows two things:

- The PPU that `npu_hw.h` defines is a real, separately completing block, with its own IRQ bits
  (10/11). [ppu-pooling-not-detile.md](ppu-pooling-not-detile.md) covers its use as a de-tile
  engine.
- "group 0 / 1" means two register banks per block, the ping-pong that the BSP exposes as
  `RKNPU_JOB_PINGPONG`. With two register sets per block, the next task's registers can be staged
  in the idle bank while the current task runs. The banks are the hardware basis for any
  register-staging or task-persistence approach to the per-submit overhead.

On the `rocket` path the kernel controls the IRQ configuration, so this bitmap is informational.
The `int_mask`/`int_clear` fields are in the BSP `rknpu_task`, not in the regcmd that userspace
submits. The `0x14 PC_REGISTER_AMOUNTS` length and the `OP_ENABLE` block mask, which userspace
does drive, are a separate mechanism.

## Context-pool throughput

**Conditions:** RK3588, kernel 7.1.0-1-arm64, NPU clock ~816 MHz, `performance` governor,
2026-06-24. Each context runs a full "inference" of small prepacked fp16 matmuls (the
1×1-conv-as-matmul detection unit), and each call is a real A-pack -> submit -> readback.

The throughput pool is model 2 above: P independent contexts, each with its own fd and entity.
Pool depth P is the swept variable.

Software must pin each pool context to a distinct A76 core. The library's auto-affinity pins
worker `idx` to `big[idx % n_big]`, and a one-thread context (the natural pool unit) always has
`idx == 0`. Left to the default, every independent pool context pins to the same big core
(cpu4), and their host pack/readback serialize there [HW sweep].

| workload | pool affinity | best speedup vs P=1 | where |
|---|---|---|---|
| tiny 64×256×256 (submit/readback-bound) | colliding (default, all on cpu4) | ~2.1x | flat past P=3 |
| tiny 64×256×256 | spread across A76 (`ROCKET_CPU_AFFINITY=off` + caller pins each ctx) | ~3.9x | peak P=4 |
| large 512×1024×1024 (compute-bound) | spread across A76 | ~2.7x | still climbing at P=6 |

The sweep shows three results:

- The pool exceeds the 3-core count on submit-bound work (3.9x at P=4 > 3). With contexts on
  separate cores, one context's host pack/readback overlaps another's NPU compute, which is the
  "rknnpool" effect. P=4 (= the 4 A76 cores) is the best depth. P>=5 oversubscribes the big
  cluster, and throughput fluctuates.
- Compute-bound ops gain less (~2.7x, approaching the 3-core NPU ceiling). There is little host
  bubble to hide, so the cores are the limit.
- The limit is the host core and the submit path, not the IRQ core. Applying the model-2 IRQ
  binding (3 NPU IRQs -> cpu5/6/7) raised the P=1 baseline (the −47% per-submit overhead
  reduction) and left the aggregate plateau unchanged. The colliding pool's limit is therefore
  the shared host core, not IRQ servicing.

### Delegate recipe

For the throughput pool:

- Use one fd/context per pool instance, with P ~4 (= the A76 count).
- Set `ROCKET_CPU_AFFINITY=off` and pin each instance to a distinct big core.
- Apply the throughput IRQ binding (`npu_set_irq_affinity.sh throughput`).

## Evidence

### uAPI contracts

The runtime conformance test `uapi_selftest_rocket`
(`rocket-userspace/tests/uapi_selftest_rocket.c`) pins the `drm_rocket_*` behaviors that the
library depends on. A kernel that drifts (a new-SoC port, a uAPI revision) then fails there with
a named diagnostic instead of mis-waiting deep in the matmul path. A kernel that regressed
`PREP_BO.timeout_ns` to a relative duration would silently turn every job wait into an immediate
`-EBUSY` poll, and this canary catches that. The past-deadline `PREP_BO` result is from a live
check.

The `CREATE_BO` result confirms that the `iova_ceiling` probe's `va = 0x0 -> …` read in
§"Per-fd IOVA windows" is the real allocator base, not an artifact.

### Per-fd IOVA windows

The two-fd result is from a probe (`iova_ceiling_rocket.c`) that opened 2 fds.

### Resident weights in the per-fd window

The M-independence sweep is `matmul_{int8,int4}_crossm_rocket`, with a pack at M=512 reused at
each M.

The group-wise planner's safety under a shrunk `Nt` is checked: `Kt` is invariant over every
`(group, slice)` a real weight produces.

### The vendor `rknpu` driver's shared IOVA domain

Four runs with the flag set, interleaved around the leaking ones, left all four size counts
byte-identical [HW sweep, RK3588, `rknpu` 0.9.8, 2026-08-25].

Three quantitative checks agree with the rcache mechanism:

- The 34 x 16 MB lost is 544 MB, against the 540 MB that a single-threaded run can park (one
  CPU's two magazines plus the 32-magazine depot, 4318 ranges x 128 KB).
- A second identical churn costs only 3 more and a third only 1, because the cache is already at
  its ceiling.
- A churn spread over 8 CPUs and all six cached orders costs 10 of 31 buffers of 128 MB. One
  real 153 s prefill costs 11.

The synthetic churn reproduces the workload.

The probe `iova_probe` allocates 16-192 MB, and the mixed-size fragmenter cycles 1-23 MB. Every
size either of them frees is far above the 128 KB bound, so neither can see the loss.

A board 1 day 4 h into its uptime had run a full day of NPU work on the tight path. It still
measured the fresh-boot row (255 / 63 / 31 / 21 buffers at 16 / 64 / 128 / 192 MB). One 153 s
run on the generic path moved it [HW sweep, RK3588, `rknpu` 0.9.8 vs mainline `rocket` 1.1.0].

### Multicore dispatch through worker fds

A probe that submitted N jobs in one submit on one fd scaled 1.00 / 1.99 / 3.07x. It did not
spread across cores: that 3.07 was an artifact of job batching, not multicore.

### Multicore models

The end-to-end delegate scaling is from `tflite-rocket/tools/pool_throughput.py`.

### IRQ affinity

The table's run is `submit_overhead_rocket 8 64 16 3000`, one discarded warm-up and seven
rotated passes per configuration, under a lock that excludes other NPU users. The
`/proc/interrupts` delta of every run confirms its routing: cpu0 at the default mask, cpu7 when
bound. Only cores 0 and 1 took jobs. The blocking medians range 29.75-33.54 (default),
25.67-29.46 (IRQ on cpu7) and 14.58-20.13 µs (co-located), and the co-located-to-default ratio
0.435-0.677 over the passes [HW sweep, RK1, 2026-10-12].

Under the as-found `ondemand` governor, five or six passes give a blocking median of 36.2 µs at
the default affinity (25.1-40.6). Co-located, they give 23.3 µs (17.5-23.6). Busy-polling gives 20.9 and
16.3 µs [HW sweep, RK1, 2026-10-12].

The module's keep-attached behavior is read from its disassembly. `iommu_detach_group` is
called from `rocket_job_run` on a domain change, from `rocket_reset` and from
`rocket_core_fini`. The completion path does not call it. Mainline v7.2 `rocket_job.c` detaches in `rocket_job_handle_irq` on
every completion.

#### Revision history

On 2026-06-22, on kernel 7.1.0-1-arm64 at ~816 MHz, the same shape gave these medians [HW sweep]:

| config | IRQ affinity | app taskset | median µs/submit |
|---|---|---|---|
| default | `0-7` (-> services on cpu0, A55) | none | 51-53 |
| default + app pinned | `0-7` | cpu7 | 51 (no change, the IRQ is still on A55) |
| IRQ on big core, app floats | cpu6 | none | 33.5 |
| IRQ + app co-located on one big core | cpu6 / cpu7 / cpu4 | same big core | 27-28 |
| IRQ + app on *different* big cores | cpu4 | cpu7 | 31 |

The run was 5×3000 + a 4000-iteration confirmation, with `rocket-userspace/tests/irq_affinity_probe.sh`
as the A/B harness, and 8000 submits incremented IRQ 69/70 entirely in the CPU0 column. Its
reading was that moving the IRQ off the A55 gave most of the reduction (51 -> ~31 µs), and
co-location a further ~4 µs. Keep-attached plus the binding was then given as ~27 µs, against
~54 µs for stock `rocket` at the default affinity. Its Conditions named IOMMU keep-attached. Its
default row matches the stock module's 54 µs in §"Per-job IOMMU cost", not the 34 µs measured
with keep-attached. Which module ran it is not recorded.

The busy-polling section stated, from a run of the same period, that at the `8 64 16` shape
(~20 µs per submit) busy-polling was high-variance and not a reliable gain.

### Busy-polling the completion fence

The run is `submit_overhead_rocket 64 256 512`, 3000-iter, an in-process blocking-vs-busy-poll
A/B. The A/B is the second loop in `submit_overhead_rocket`.

### Per-job IOMMU cost

The measurement is a clean A/B of stock vs patch-5. Two independent tests agree, and a third
shows where the cost amortizes:

- `rocket-userspace/tests/submit_overhead_rocket.c`: the tiny 1-task job, 2000x on the same fd
- `tests/multicore_probe`: the 64-task jobs, 10 reps
- `matmul_tiled_rocket 512 3840 4096`: the tiled matmul

With the keep-attached patch, CTest passes 8/8 and dmesg is clean.

### Reset before IOMMU detach

The faults are requested through `tests/uapi_regcmd_fault_rocket`, both modes. With the patch,
83/83 `ctest` pass and the taint word is unmoved. The `WARN` fired in neither configuration.
What the A/B measures is the failing handshake that the `WARN` is downstream of.

### Batched submit

The first table row is `matmul_tiled_rocket 512 3840 4096`.
`patches/rocket/BATCHED_SUBMIT_FINDINGS.md` holds the full mechanism and the disproven
kernel-only model.

### Context-pool throughput

The probe is `rocket-userspace/tests/ctx_pool_throughput.c` (`ctx_pool_throughput`). The
model-2 throughput path (queue depth > core count) is distinct from the model-1
`multicore_threads` probe, which packs once and loops a bare submit (pure-NPU, saturates at 3).

### Excluded causes

- The vendor driver's loss does not come from the two paths' size rounding or placement.
- The error path is not the cause either: every one of five prefill runs across both paths
  reported zero kernel allocation failures.
- The BSP `rknpu` driver does not fire a task list in one kick by DMA-walking a kernel-built
  descriptor table. That model was tested on the RK3588, and every variant times out with no
  completion IRQ.

### Revision history

A probe on 2026-08-25 recorded identical counts and identical addresses on both vendor-driver
paths on an empty domain. The sizes were 16, 32, 32.03, 48, 64, 128 and 192 MB. At 32.03, 48
and 192 MB that disagrees with a measurement of 2026-08-26. There a fresh domain served 15
buffers of 192 MB on the generic path, against 21 on the tight one.

It also disagrees with the source. On the generic path, with `RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT` clear,
`rknpu_iommu_dma_alloc_iova()` allocates through `alloc_iova_fast()`. That function aligns each
range to its size rounded up to a power of two [source-confirmed: `rknpu_iommu.c` in `rknpu`
0.9.8, and the kernel's `iova.c`].
