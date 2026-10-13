# PREP_BO / FINI_BO cache-sync cost

The cache-sync cost of a PREP_BO / FINI_BO pair is proportional to the BO's allocated size,
not to the bytes used [source-confirmed: `rocket_gem.c`, HW sweep]. It is a property of the
`rocket` driver and its uAPI, and a reducible part of the prefill dispatch overhead. An
oversized output BO puts ~20% of wall-clock time in `sync`. Sizing the K-accumulation output
BO to its live tiles, rather than to the 64-tile batch, raises resident fp16 matmul throughput
by ~11% [HW sweep].

**Conditions:** RK3588, NPU clock 600 MHz, resident multicore, `512×3840×4096`, fp16 + KACC +
DATA_REUSE, idle board.

**Not characterized:** int4 at a small-shape, many-call workload, and a reducible fixed
per-fence component of the `wait` term.

| output BO | `sync` (Σ over workers) | fp16 wall-clock |
|---|---:|---:|
| `BATCH`-sized | 127 ms | baseline |
| right-sized `nMt·nNt` | 15 ms (−88%) | +~11% |

The `sync` term falls ~8.5x, which tracks the ~8x BO-size reduction and confirms that the cost
is ∝ size.

## Cause

The BOs of `rocket` are cached (write-back) CPU mappings of the DMA memory. Software must
perform cache maintenance each time it hands a BO between the CPU and the NPU:

- `DRM_IOCTL_ROCKET_PREP_BO` runs `dma_sync_sgtable_for_cpu()` on the BO.
- `DRM_IOCTL_ROCKET_FINI_BO` runs `dma_sync_sgtable_for_device()`.

Both walk the BO's entire scatter-gather list and do per-page cache maintenance. Neither
ioctl has an offset or length (`drm_rocket_prep_bo` is just `{handle, timeout}`). Both
therefore always sync the whole BO, even where the NPU touched only a small live sub-region.
The sync cost is ∝ the allocated BO size (page count), not ∝ the bytes used
[source-confirmed: `rocket_gem.c`, HW sweep].

These are the only cache-sync ioctls of mainline `rocket`. A kernel that reports interface 1.5
or later, which takes an out-of-tree patch, adds `DRM_ROCKET_PREP_BO_RANGES` and
`DRM_ROCKET_FINI_BO_RANGES`. They take an array of `{offset, size}` ranges and sync only those
[source-confirmed: `librocketnpu` `rocket_bo_prep_ranges()`]. The measurements in this note
use the whole-BO forms.

The readback de-tile and the pack are memory-bound, not ALU-bound, for the same reason (see
[not-mac-bound.md](not-mac-bound.md)). The sync cost is a separate line in the profile
(`sync`), distinct from `pack`/`read`/`wait`. Part of the dispatch overhead of
[not-mac-bound.md](not-mac-bound.md) is this host-side cache maintenance on an over-allocated,
repeatedly synced output BO, rather than irreducible NPU latency.

### Output BO size on the K-accumulation path

The matmul batches up to `BATCH = 64` tiles into one NPU job (one fence), so the per-job
output/regcmd BOs are sized for 64 tiles. The K-accumulation path (`mm_compute_kacc`, the
default `ROCKET_KACC` operating mode) issues one job per K-tile. Each K-step's DPU
eltwise-add reads the previous step's output (ping-pong), which is a serial dependency. Each
such job writes only `nMt·nNt` output tiles, never the full `BATCH`. For a typical resident
Gemma slice `nMt·nNt = 8`. A `BATCH`-sized KACC output BO (`out_all` + `pong`) is then 8x
oversized: ~8 MB allocated, ~1 MB live, cache-synced four times per K-job, ten K-jobs per
matmul.

## Right-sized output BO

Size the K-accumulation output BO to its live tile count, not to `BATCH`. `librocketnpu` gives
KACC its own right-sized output ping-pong (`okacc0` + `pong`, sized to `nMt·nNt` tiles, not
`BATCH`). It leaves `out_all` at `BATCH` for the rare tiny-M `mm_compute` fallback. The change
is bit-exact. `ROCKET_KACC_BATCHOUT=1` selects the `BATCH` sizing.

## Applicability

The mechanism is datatype-independent, and its effect depends on the operating point. The
right-sized output BO helps only where the weights are resident. Those cases are:

- The standalone benchmark
- The K<=2048 shapes that ggml-rocket holds resident by default
- F16 weights of any K under the opt-in `ROCKET_F16_RESIDENT`
- Whisper's repeated encoder
- A quantized model held fully resident

It has no effect on streaming-bound prefill, and it costs nothing and stays bit-exact.

In a model the effect is ~flat. A profiled `pp512` comparison on Gemma-4-12B F16 shows in-model
`sync` unchanged (20.6 s -> 21.0 s), and pp512 wall-clock time unchanged with no regression.
This fp16 model's K>2048 shapes run the streaming path (re-pack B every call -> `packB`~72 s,
the giant pack mass). The weight-BO re-pack sync therefore dominates in-model `sync`, not the
output BO.

The `packB`~72 s term in the pp512 profile is the streaming re-pack of B (+ its `wt_all`
cache-sync) for K>2048 shapes. Removing it needs resident pre-tiled weights for K>2048.
ggml-rocket holds F16 weights resident by default only at K<=2048, and the opt-in
`ROCKET_F16_RESIDENT` extends residency to every K, where in-model fp16 prefill gains
([weight-residency-fusion.md](weight-residency-fusion.md)). The remaining `wait` term (CPU
blocked on the fence) is NPU-bound (see [not-mac-bound.md](not-mac-bound.md)).

## Sync count across a workload

A right-sized output BO helps a path that syncs that BO many times, per matmul or per second.
The saving is `sync_saved = (size_old − size_new) × (#syncs of that BO)`, counted over the
workload. The measured cases are:

- fp16 KACC gains (§"Right-sized output BO"). It issues one job *per K-tile* and syncs the
  output BO on every one (`~nKt` syncs/matmul). An 8x smaller BO × `nKt` is therefore a large
  cut.
- The int4, int8 and `mm_compute` host-accumulation paths gain nothing at an LLM shape. They
  batch across K and sync the output BO once per job, and at `512×3840×4096` a job fills most
  of the batch. With one sync of a nearly full BO, right-sizing saves ~nothing. The int4
  right-size is bit-exact and measures no benefit there (int4 throughput is noisy
  ~410-580 GOP/s and the change sits inside that band). When `total_tiles < I4_BATCH` (int4's
  denser pack -> ~24 tiles for the shape above), the int4 output BO is oversized, but at that
  shape one sync leaves nothing to cut.
- The resident int8 path gains at detector shapes, even at one sync per job. A detector 1×1
  layer gives each worker a handful of tiles against a 64-slot output BO. A MobileDet invoke
  runs 126 job batches, and each syncs that BO four times. Capping the output and regcmd BOs
  at the call's tile count takes the matmul `sync` phase from 743 to 52 ms over 21 MobileDet
  invokes. MobileDet runs 1.15x and EfficientDet-Lite0 1.11x warm, with outputs
  byte-identical [HW sweep, RK1, 600 MHz, governor pinned, through `tflite-rocket`,
  2026-09-27]. At an LLM shape resident int8 fills `BATCH`, and at detector shapes it does not.
- A grow-only BO shared by calls of many sizes is the same trap. An RK3588 conv context
  that holds one BO per role, grown to its largest job, zeroes and syncs the whole BO on
  every job. With that sizing, MobileDet's int8 direct conv synced 2228 MB of output BO for
  449 MB of output over 21 invokes. One BO per power-of-two size class per role cuts that to
  643 MB, and its output zeroing from 207 to 77 ms. MobileDet then runs 1.08-1.10x warm, with
  outputs byte-identical [HW sweep, RK1, 600 MHz, governor pinned, `rocket` 1.3.0,
  2026-09-27]. A class keeps up to 2x, and a kernel with the ranged forms (interface 1.5 and
  later, `rocket_bo_prep_ranges()`) syncs only the live bytes.
- The flash-attention batch contexts have the same trap, by op. A worker that keeps one
  batched matmul context per op, grown to the largest shape seen, syncs the whole BO around
  every pack. With that sizing, a three-question 58-token batch cost 375 ms after a 726-token
  call and 244 ms alone. With one context per op and per 4x class of rows times keys, it
  costs 241-242 ms either way [HW sweep, RK1, 600 MHz, `rocket` 1.3.0, 2026-09-27].

Count the syncs over the workload. Many small calls against an oversized BO cost the same as
many syncs per matmul.

## Write-combine and uncached BO mappings

A write-combine or uncached BO mapping would make `PREP_BO`'s `dma_sync_sgtable_for_cpu()`
unnecessary for the output read. Userspace cannot select one.

The mainline rocket uAPI does not expose a cache mode to userspace [source-confirmed]. The
`struct drm_rocket_create_bo` argument has only `{size, handle, dma_address, offset}`, with no
flags and no cache-mode field. The kernel maps every BO cached (GEM-SHMEM default). The only
userspace cache control is the `PREP_BO`/`FINI_BO` sync pair. A WC mapping therefore requires a
kernel-module change, not a library option.

A WC mapping is also unlikely to help even then. The dominant readback cost is the A76 NEON
de-tile gather (the cube->row-major scatter, [not-mac-bound.md](not-mac-bound.md)). NEON reads
from *uncached* memory are far slower than reads from cached memory plus one bulk invalidate.
WC therefore trades a cheap bulk `sync` for an expensive per-element uncached read. WC helps
streaming *writes* (the pack side), not the gather-bound *read* side.

Right-sizing (above) reduces the `sync` cost, and the NEON de-tile reduces the readback cost.

## Evidence

### Right-sized output BO

The wall-clock figure is a back-to-back comparison in one binary, three runs: +9.1 / +11.4 /
+12.2%. The toggle `ROCKET_KACC_BATCHOUT=1` selects the `BATCH` sizing in that same binary, so
there is no run-to-run drift. The `sync` reduction is the reproducible result. A best-of-N
comparison at a favorable board state measures as high as +17%, and the drift-controlled ~+11%
is the figure the body states. The cosine correctness matrix passes with the change, which is
one allocation site.

### Applicability

The in-model comparison is profiled and interleaved (B~A).

### Sync count across a workload

The conv BO figures are `ROCKET_CONV_PROFILE` readouts.
