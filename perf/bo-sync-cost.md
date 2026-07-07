# PREP_BO / FINI_BO cache-sync cost is proportional to BO size

A driver/uAPI fact that is a real, reducible slice of the prefill "dispatch floor" — and
an easy way to waste ~20% of wall time if you ignore it.

## The mechanism

`rocket`'s BOs are **cached (write-back) CPU mappings** of the DMA memory. So every
time you hand a BO between the CPU and the NPU you must do cache maintenance:

- `DRM_IOCTL_ROCKET_PREP_BO` runs `dma_sync_sgtable_for_cpu()` on the BO,
- `DRM_IOCTL_ROCKET_FINI_BO` runs `dma_sync_sgtable_for_device()`.

Both walk the BO's **entire** scatter-gather list and do per-page cache maintenance.
There is **no offset/length** in the uAPI (`drm_rocket_prep_bo` is just
`{handle, timeout}`) — you always sync the whole BO, even if the NPU only touched a
small live sub-region. So the sync cost is **∝ the allocated BO size (page count)**,
*not* ∝ the bytes actually used. [source: `rocket_gem.c`; HW + measured]

This is the same reason the readback de-tile and the pack are memory-bound, not
ALU-bound (see [not-mac-bound.md](not-mac-bound.md)) — but it is a *separate* cost
line in the profile (`sync`), distinct from `pack`/`read`/`wait`.

## The trap: a BATCH-sized output BO on the K-accumulation path

**Size the K-accumulation output BO to its live tile count, not to `BATCH`** — sharing
the `BATCH`-sized output BO with the K-accumulation path over-allocates it 8× and inflates
`sync` to ~20% of wall.

The matmul batches up to `BATCH = 64` tiles into one NPU job (one fence), so the per-job
output/regcmd BOs are sized for 64 tiles. But the **K-accumulation path**
(`mm_compute_kacc`, the default `ROCKET_KACC` operating mode) issues **one job per
K-tile** — each K-step's DPU eltwise-add reads the previous step's output (ping-pong), a
serial dependency — and each such job writes only `nMt·nNt` output tiles, **never the full
`BATCH`**. For a typical resident Gemma slice `nMt·nNt = 8`. A `BATCH`-sized KACC output BO
(`out_all` + `pong`) is then **8× oversized**: ~8 MB allocated, ~1 MB live, cache-synced
four times per K-job, ten K-jobs per matmul.

## The right-size and the measurement

Give KACC its own **right-sized** output ping-pong (`okacc0` + `pong`, sized to
`nMt·nNt` tiles, not `BATCH`); leave `out_all` at `BATCH` for the rare tiny-M
`mm_compute` fallback. One alloc-site change, bit-exact (the cosine correctness matrix all
green), `ROCKET_KACC_BATCHOUT=1` selects the `BATCH` sizing for A/B.

Resident multicore, `512×3840×4096`, fp16 + KACC + DATA_REUSE, 600 MHz, idle box.
The hard, reproducible result is the **`sync` collapse**; the wall translation is from
a controlled back-to-back A/B (`ROCKET_KACC_BATCHOUT=1` toggles the old sizing in the
same binary, so no run-to-run drift):

| output BO | `sync` (Σ over workers) | fp16 wall (back-to-back A/B) |
|---|---:|---:|
| `BATCH`-sized | 127 ms | baseline |
| right-sized `nMt·nNt` | **15 ms** (−88%) | **+~11%** (3 runs: +9.1 / +11.4 / +12.2%) |

The `sync` term collapses ~8.5×, tracking the ~8× BO-size reduction — confirming the cost
is ∝ size. (A best-of-N comparison at favorable box state reads as high as +17%; the
drift-controlled A/B is ~+11% — use that.)

**In-model this lever is ~flat — and that scopes where it applies.** A profiled
`pp512` A/B (Gemma-4-12B F16) shows in-model `sync` **unchanged** (20.6 s → 21.0 s)
and pp512 wall **unchanged** (interleaved B≈A, no regression). Why: this fp16 model's
K>2048 shapes run the **streaming** path (re-pack B every call → `packB`≈72 s, the
giant pack mass), so in-model `sync` is dominated by the **weight-BO** re-pack sync,
not the output BO. The output-BO right-size only bites when weights are **resident**
(the standalone bench, K≤2048 prepacked shapes, Whisper's repeated encoder, or a
quantized model held fully resident). So: a real win at the **resident operating
point**, invisible to streaming-bound prefill — but free and correct.

## The general rule (and its sharp edge)

**A right-sized output BO only helps a path that syncs that BO *many times per
matmul*.** The win is `sync_saved = (size_old − size_new) × (#syncs of that BO)`. So:

- **fp16 KACC wins (+~11%):** it issues one job *per K-tile* and syncs the output BO
  on every one (`~nKt` syncs/matmul), so an 8× smaller BO × `nKt` is a big cut.
- **int4 / int8 / `mm_compute` host-accum do not win:** they batch across K and sync
  the output BO **once per job**. With one sync, an 8× smaller BO saves ~nothing; the
  int4 right-size is bit-exact but **measures no benefit** (int4 throughput is noisy
  ~410–580 GOP/s and the change sits inside that band — possibly a larger BO even
  DDR-interleaves the single WDMA burst slightly better), so it stays `BATCH`-sized. Do
  not port the right-size to single-sync paths.
- It is **dtype-independent in mechanism** but **operating-point-specific in effect**:
  it lifts the **resident** fp16 path (standalone bench, K≤2048 prepacked, Whisper,
  fully-resident quant), and is invisible to **streaming** prefill (in-model Gemma F16,
  K>2048) where the weight-BO pack-sync dominates.
- It refines [not-mac-bound.md](not-mac-bound.md): part of the "dispatch floor" is
  **not** irreducible NPU latency but host-side cache maintenance on an over-allocated,
  repeatedly-synced output BO.

## Skipping the sync via a write-combine / uncached BO is not a userspace lever

The cleanest way to eliminate the output-read `sync` would be to map the BO **write-combine /
uncached** so `PREP_BO`'s `dma_sync_sgtable_for_cpu()` becomes unnecessary. **The mainline
rocket uAPI does not expose this from userspace** [source-confirmed]: `struct
drm_rocket_create_bo` has only `{size, handle, dma_address, offset}` — **no flags / cache-mode
field** — and the kernel maps every BO cached (GEM-SHMEM default). The only userspace cache
control is the `PREP_BO`/`FINI_BO` sync pair. So a WC mapping requires a **kernel-module change**
(the `patches/rocket` patches), not a library knob.

And it is unlikely to pay even then: the dominant readback cost is the **A76 NEON de-tile
gather** (the cube→row-major scatter, [not-mac-bound.md](not-mac-bound.md)), and NEON reads from
*uncached* memory are far slower than from cached + one bulk invalidate — WC trades a cheap bulk
`sync` for an expensive per-element uncached read. WC helps streaming *writes* (the pack side),
not the gather-bound *read* side. Conclusion: the `sync` lever is right-sizing (above), not WC;
the readback lever is the NEON de-tile, not the mapping mode.

## Open follow-ups

- The int4 resident path does **not** benefit (single-sync; see above). Its output BO is
  oversized when `total_tiles < I4_BATCH` (int4's denser pack → ~24 tiles for the shape
  above), but with one sync there is nothing to cut. int8 resident fills `BATCH` (batches
  across K, no oversizing) and is readback-bound regardless.
- **The real in-model `sync` lever is the streaming weight-BO**, not the output BO:
  `packB`≈72 s in the pp512 profile is the streaming re-pack of B (+ its `wt_all`
  cache-sync) for K>2048 shapes. Killing it = resident pre-tiled weights for K>2048
  (residency currently gated at K≤2048). That is where in-model fp16 prefill
  actually gains.
- The remaining `wait` term (CPU blocked on the fence) is the genuinely NPU-bound
  floor — see [not-mac-bound.md](not-mac-bound.md). Whether it has a reducible
  fixed-per-fence component is a separate microbench.
