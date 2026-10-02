# PREP_BO / FINI_BO cache-sync cost

The cache-sync cost of a PREP_BO / FINI_BO pair is proportional to the BO's allocated size.
It is a driver/uAPI fact, and a reducible slice of the prefill "dispatch floor". Ignoring it
is an easy way to waste ~20% of wall time.

## The mechanism

The BOs of `rocket` are cached (write-back) CPU mappings of the DMA memory. So every time you
hand a BO between the CPU and the NPU, you must do cache maintenance:

- `DRM_IOCTL_ROCKET_PREP_BO` runs `dma_sync_sgtable_for_cpu()` on the BO.
- `DRM_IOCTL_ROCKET_FINI_BO` runs `dma_sync_sgtable_for_device()`.

Both walk the BO's entire scatter-gather list and do per-page cache maintenance. Neither
ioctl has an offset or length (`drm_rocket_prep_bo` is just `{handle, timeout}`). So they
always sync the whole BO, even if the NPU only touched a small live sub-region. The sync cost
is therefore ∝ the allocated BO size (page count), not ∝ the bytes used
[source-confirmed: `rocket_gem.c`, HW sweep].

This is the same reason the readback de-tile and the pack are memory-bound, not ALU-bound
(see [not-mac-bound.md](not-mac-bound.md)). It is a separate cost line in the profile
(`sync`), distinct from `pack`/`read`/`wait`.

## Output BO size on the K-accumulation path

**Size the K-accumulation output BO to its live tile count, not to `BATCH`.** Sharing
the `BATCH`-sized output BO with the K-accumulation path over-allocates it 8x and inflates
`sync` to ~20% of wall.

The matmul batches up to `BATCH = 64` tiles into one NPU job (one fence), so the per-job
output/regcmd BOs are sized for 64 tiles. The K-accumulation path (`mm_compute_kacc`, the
default `ROCKET_KACC` operating mode) issues one job per K-tile. Each K-step's DPU
eltwise-add reads the previous step's output (ping-pong), which is a serial dependency. Each
such job writes only `nMt·nNt` output tiles, never the full `BATCH`. For a typical resident
Gemma slice `nMt·nNt = 8`. A `BATCH`-sized KACC output BO (`out_all` + `pong`) is then 8x
oversized: ~8 MB allocated, ~1 MB live, cache-synced four times per K-job, ten K-jobs per
matmul.

## Right-sized output BO and measurement

Give KACC its own right-sized output ping-pong (`okacc0` + `pong`, sized to `nMt·nNt` tiles,
not `BATCH`). Leave `out_all` at `BATCH` for the rare tiny-M `mm_compute` fallback. This is
one alloc-site change, and it is bit-exact (the cosine correctness matrix is all green). Set
`ROCKET_KACC_BATCHOUT=1` to select the `BATCH` sizing for an A/B.

The operating point is resident multicore, `512×3840×4096`, fp16 + KACC + DATA_REUSE, 600 MHz,
on an idle box. The hard, reproducible result is the `sync` collapse. The wall translation
comes from a controlled back-to-back A/B in one binary. The toggle `ROCKET_KACC_BATCHOUT=1`
selects the `BATCH` sizing in that same binary, so there is no run-to-run drift:

| output BO | `sync` (Σ over workers) | fp16 wall (back-to-back A/B) |
|---|---:|---:|
| `BATCH`-sized | 127 ms | baseline |
| right-sized `nMt·nNt` | 15 ms (−88%) | +~11% (3 runs: +9.1 / +11.4 / +12.2%) |

The `sync` term collapses ~8.5x, which tracks the ~8x BO-size reduction and confirms that the
cost is ∝ size. A best-of-N comparison at favorable box state reads as high as +17%. The
drift-controlled A/B is ~+11%, and that is the figure to use.

### In-model effect

In-model this lever is ~flat, and that scopes where it applies. A profiled `pp512` A/B
(Gemma-4-12B F16) shows in-model `sync` unchanged (20.6 s -> 21.0 s) and pp512 wall unchanged
(interleaved B~A, no regression). The cause is that this fp16 model's K>2048 shapes run the
streaming path (re-pack B every call -> `packB`~72 s, the giant pack mass). So the weight-BO
re-pack sync dominates in-model `sync`, not the output BO.

The output-BO right-size only helps when weights are resident (the standalone bench, K<=2048
prepacked shapes, Whisper's repeated encoder, or a quantized model held fully resident). So
it is a win at the resident operating point. It is invisible to streaming-bound prefill, and
it is free and correct.

## General rule

A right-sized output BO helps a path that syncs that BO many times, per matmul or per second.
The win is `sync_saved = (size_old − size_new) × (#syncs of that BO)`, counted over the
workload. The measured cases follow:

- fp16 KACC wins (+~11%). It issues one job *per K-tile* and syncs the output BO on every
  one (`~nKt` syncs/matmul). So an 8x smaller BO × `nKt` is a big cut.
- The int4, int8 and `mm_compute` host-accumulation paths do not win at an LLM shape. They
  batch across K and sync the output BO once per job, and at `512×3840×4096` a job fills most
  of the batch. With one sync of a nearly full BO, right-sizing saves ~nothing. The int4
  right-size is bit-exact and measures no benefit there (int4 throughput is noisy
  ~410-580 GOP/s and the change sits inside that band).
- The resident int8 path does win at detector shapes, even at one sync per job. A detector
  1×1 layer gives each worker a handful of tiles against a 64-slot output BO. A MobileDet
  invoke runs 126 job batches, and each syncs that BO four times. Capping the output and
  regcmd BOs at the call's tile count takes the matmul `sync` phase from 743 to 52 ms over 21
  MobileDet invokes. MobileDet runs 1.15x and EfficientDet-Lite0 1.11x warm, with outputs
  byte-identical [HW sweep, RK1, 600 MHz, governor pinned, through `tflite-rocket`,
  2026-09-27]. So count the syncs over the workload: many small calls against an oversized
  BO pay the same cost as many syncs per matmul.
- A grow-only BO shared by calls of many sizes is the same trap. An RK3588 conv context
  that holds one BO per role, grown to its largest job, zeroes and syncs the whole BO on
  every job. With that sizing, MobileDet's int8 direct conv synced 2228 MB of output BO for
  449 MB of output over 21 invokes. One BO per power-of-two size class per role cuts that to
  643 MB and its output zeroing from 207 to 77 ms. MobileDet then runs 1.08-1.10x warm, with
  outputs byte-identical [HW sweep, RK1, 600 MHz, governor pinned, `rocket` 1.3.0,
  `ROCKET_CONV_PROFILE`, 2026-09-27]. A class keeps up to 2x, and a kernel with the ranged
  forms (interface 1.5 and later, `rocket_bo_prep_ranges()`) syncs only the live bytes.
- The flash-attention batch contexts carry the same trap, by op. A worker that keeps one
  batched matmul context per op, grown to the largest shape seen, syncs the whole BO around
  every pack. With that sizing, a three-question 58-token batch cost 375 ms after a 726-token
  call and 244 ms alone. With one context per op and per 4x class of rows times keys, it
  costs 241-242 ms either way [HW sweep, RK1, 600 MHz, `rocket` 1.3.0, 2026-09-27].
- The lever is datatype-independent in mechanism and operating-point-specific in effect. It
  lifts the resident fp16 path (standalone bench, K<=2048 prepacked, Whisper, fully-resident
  quant). It is invisible to streaming prefill (in-model Gemma F16, K>2048), where the
  weight-BO pack-sync dominates.
- The rule refines [not-mac-bound.md](not-mac-bound.md). Part of the "dispatch floor" is
  host-side cache maintenance on an over-allocated, repeatedly-synced output BO, not
  irreducible NPU latency.

## Write-combine and uncached BO mappings

Skipping the sync through a write-combine or uncached BO is not a userspace lever. The
cleanest way to eliminate the output-read `sync` would be to map the BO write-combine /
uncached, so that `PREP_BO`'s `dma_sync_sgtable_for_cpu()` becomes unnecessary.

**The mainline rocket uAPI does not expose this from userspace** [source-confirmed]. The
`struct drm_rocket_create_bo` argument has only `{size, handle, dma_address, offset}`, with no
flags and no cache-mode field. The kernel maps every BO cached (GEM-SHMEM default). The only
userspace cache control is the `PREP_BO`/`FINI_BO` sync pair. So a WC mapping requires a
kernel-module change (the `patches/rocket` patches), not a library knob.

It is also unlikely to pay even then. The dominant readback cost is the A76 NEON de-tile
gather (the cube->row-major scatter, [not-mac-bound.md](not-mac-bound.md)). NEON reads from
*uncached* memory are far slower than reads from cached memory plus one bulk invalidate. So
WC trades a cheap bulk `sync` for an expensive per-element uncached read. WC helps streaming
*writes* (the pack side), not the gather-bound *read* side.

The `sync` lever is right-sizing (above), not WC. The readback lever is the NEON de-tile, not
the mapping mode.

## Open questions

- The int4 resident path does not benefit (single-sync, see above). Its output BO is
  oversized when `total_tiles < I4_BATCH` (int4's denser pack -> ~24 tiles for the shape
  above), but at that shape one sync leaves nothing to cut. At an LLM shape int8 resident
  fills `BATCH`. At detector shapes it does not, and the path right-sizes its output BO
  (above). Whether int4 gains at a small-shape, many-call workload is not measured.
- The in-model `sync` lever is the streaming weight-BO, not the output BO. The `packB`~72 s
  term in the pp512 profile is the streaming re-pack of B (+ its `wt_all` cache-sync) for
  K>2048 shapes. Removing it needs resident pre-tiled weights for K>2048 (residency currently
  gated at K<=2048). That is where in-model fp16 prefill gains.
- The remaining `wait` term (CPU blocked on the fence) is the NPU-bound floor (see
  [not-mac-bound.md](not-mac-bound.md)). Whether it has a reducible fixed-per-fence component
  is a separate microbench.
