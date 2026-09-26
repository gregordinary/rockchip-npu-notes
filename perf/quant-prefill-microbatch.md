# Quantized-GGUF prefill is micro-batch-dequant-bound

A quantized GGUF (`Q4_K` / `IQ4_XS` / `Q8_0` / …) prefills on the NPU through the
dequant->fp16 streaming path. Each weight is dequantized to fp16 and packed into the native
tiles **per micro-batch**, every forward pass. The resident fp16 weight cache is F16-only, so
quant weights re-pack each call. That per-ubatch dequant, not the matmul, sets quantized
prefill throughput. Two consequences follow: a runtime lever (`-ub`) and a routing floor
(`ROCKET_MIN_M_QUANT`). Measured on Qwen3.5-9B / Qwen3.6-27B, 600 MHz.

This refines [not-mac-bound.md](not-mac-bound.md), which shows the *native* int8/int4 paths
losing to fp16 on the int32-readback wall. This is the *GGUF-quant dequant* path instead. Its
loss is a host dequant cost that **amortizes with micro-batch size**, so it is far more
recoverable than the native-int readback wall.

## The micro-batch lever: `-ub 2048` ~doubles quantized prefill

The dequant is paid once per micro-batch, so it amortizes over the ubatch's rows. The
llama.cpp default `-ub 512` re-dequantizes the whole model every 512-row chunk, and a larger
ubatch spreads that fixed cost. **9B Q4_K, pp2048, by `-ub`** [HW sweep, 2026-06-28, 600 MHz]:

| `-ub` | Q4_K_M | IQ4_XS | Q8_0 |
|---|---:|---:|---:|
| 512 (default) | 8.2 | 8.0 | 8.3 |
| 1024 | 12.7 | 12.5 | 13.1 |
| 2048 | **17.2** | **17.2** | **17.4** |

- **`-ub 2048` read ~2.1x over the default in this 2026-06-28 sweep; under the current build it
  is 0.91-1.53x, model-dependent, and negative on two models of eleven** — the per-model rows
  and the mechanism of the fall are in `perf/data/tuning-matrix.md`. The dated table above is
  kept as the measurement it was; do not quote its ratio as current.
- **Quant type is irrelevant to NPU prefill throughput.** Q4_K / IQ4_XS / Q8_0 converge
  within noise at a given `-ub`, because they run the identical fp16 matmul. Only the dequant
  differs, and at `-ub 2048` it is amortized away. Choose the quant on RAM and quality, not NPU
  speed. i-quants carry no NPU penalty, and their thinner *relative* win is only that the
  smaller file gives the CPU baseline a head start.
- **Even amortized, quant ~0.64x F16, unless the dequant is made resident.** F16 stays resident
  at zero dequant, 26.8 t/s @ub2048 on the 9B, where the quants plateau near 17.3. The residual
  per-2048 dequant is still ~35%, and **`-ub` cannot close it**. Only a resident dequant cache,
  dequant and pack once, can. **`ROCKET_QUANT_RESIDENT=1`** is that cache: it dequantizes each
  quantized weight to fp16 **once** and reuses the F16 prepacked path, so prefill pays neither the
  per-µbatch dequant nor the per-call `packB`, which collapses 12440 to 889 ms [HW MM_PROFILE].

  It measures **F16 parity**, at **bit-identical PPL** to streaming [HW, same session]. On the 0.8B
  `Q4_K` at pp2048 resident reads 89.2 t/s against F16's 88.9, where streaming reads 80.7 @ub2048
  and 50.7 @ub512, so 1.50x at the default `-ub`. On the 9B resident reads 24.6 against streaming's
  15.8, a 1.56x that is ~0.92x the 26.8 F16, because the 9B's ~18 GB fp16 footprint fit the 31 GB
  RAM and the IOVA window.

  It trades the quant RAM saving back for the full fp16 resident footprint, which is why it is
  opt-in, and it confirms the 0.64x plateau was a **software** cost rather than silicon. The 0.64x
  echoed the native resident-int8 0.60x of [not-mac-bound.md] by a different mechanism, host
  dequant here against int32 readback there. Unlike the int8 readback wall, this one was
  recoverable, and now is.

## What the `-ub` lever is made of: the dequant component isolated

The flag's documented mechanism — dequant amortization — is now separable from everything else
the micro-batch size moves, because the dense mechanism control exists:
**`ROCKET_DEQUANT_CACHE_MB`** (ggml-rocket) holds each streaming weight's dequantized fp16 form
host-side, dequant once per process, with the per-call pack, submit and placement exactly the
shipped path (greedy output byte-identical; the `[dq-cache]` teardown line reports engagement).
An A/B of `-ub 512` against `-ub 2048` under the cache is the lever with the dequant term
removed. Three units, three rotated passes each, idle board audited
[HW sweep 2026-08-31, RK1, 600 MHz]:

| model | full `-ub` lever | dequant component | non-dequant residue |
|---|---:|---:|---:|
| Qwen3.5-9B `Q4_K` | 1.430x | **1.378x** | 1.038x |
| SmolVLM2-2.2B `Q4_K` | 0.946x | **1.199x** | **0.789x** |
| Qwen3.5-0.8B `Q4_K` | 1.066x | ~1.10x | unresolved (straddles 1.00) |
| Llama-3.2-3B `Q4_K` | 1.085x | 1.258x | **0.863x** |
| Ministral-3-3B `Q4_K` | 1.035x | 1.239x | **0.835x** |
| Phi-4-mini `Q4_K` | 1.107x | 1.282x | **0.864x** |

- **On the 9B the mechanism story is right**: ~90% of the lever (in log terms) is dequant
  amortization.
- **On every measured sub-4B model it is upside down.** The flag nets a small win or a loss,
  because a **14-21% non-dequant cost** of the larger micro-batch sits under a 1.20-1.28x dequant
  win. On SmolVLM2 a profiled A/B of the cache pair places about half that cost in the driver's
  output readback and de-tile. `read` grows 64% at identical output bytes, 1.76 to 10.9 ms per
  job-batch, superlinear in M. The rest is in `wait` [hypothesis, not isolated]. The instrument
  contains no FLASH_ATTN ops, because llama-bench runs `-fa 0`, so attention-chunk shape is not
  part of the measured residue.
- **The right lever is therefore dequant-removal at the DEFAULT `-ub`, and not only below 4 B.**
  Unstacked `ROCKET_QUANT_RESIDENT=auto` beats residency stacked on `-ub 2048` on **six of six**
  models measured both ways. Three rotated passes each, 100% resident on every pass, unstacked
  against stacked: 1.346x/1.002x (SmolVLM2), 1.472x/1.212x (Llama-3.2-3B), 1.445x/1.132x
  (Ministral-3-3B), 1.508x/1.211x (Phi-4-mini), 1.651x/1.325x (Ministral-3-8B), 1.752x/1.661x
  (Qwen3.5-9B).
- **The 9B is why this is not just the sub-4B residue.** Its own non-dequant residue is a WIN
  at 1.038x. That is the case where the residue argument predicts stacking holds, and it still
  gains 5.5% unstacked. The second mechanism is the per-call pack and upload. Residency removes
  that too, and the default `-ub` presents it four times as often. Untested unstacked:
  `gemma4-12b`, which places only 73-74% of its weights.
- **The dequant share is load-sensitive.** The same campaign run behind a single busy core
  (a leaked spinner, discovered later) read the lever at 1.484x / 1.526x / **1.120x** — the
  tenant steals the A76-pinned dequant pool's capacity, which the `-ub 512` arm uses four
  times as often, so host load inflates the flag on every model and flips SmolVLM2's sign.
  A `-ub` recommendation measured on a loaded host overstates the flag for an idle deployment.

The control knob itself is a diagnostic: as a user lever it is dominated by
`ROCKET_QUANT_RESIDENT`, which holds the same fp16 footprint and removes the per-call pack and
upload as well.

## The threading lever: the streaming dequant was serial

The per-µbatch dequant that `-ub` amortizes and `ROCKET_QUANT_RESIDENT` eliminates was, on the
streaming path, **single-threaded**: one core decoded every weight's `[N,K]` rows quant->fp16
before the (already multicore) tile scatter ran. The rows are independent (a private K-float
scratch in, a disjoint fp16 slice out), so fanning them across the A76+A55 recovers a large
share of that cost with **no extra RAM**. That makes it the lever for the RAM-constrained case
the resident fp16 copy can't serve. **9B `Q4_K`, serial vs threaded dequant** [HW A/B, same
session, 2026-06-28]:

| `-ub` | serial | threaded | gain |
|---|---:|---:|---:|
| 512 (default) | 7.5 | 10.0 | **+33%** |
| 2048 | 15.9 | 18.8 | **+19%** |

Threaded is the default (`ROCKET_DEQUANT_THREADS`, auto = `hardware_concurrency` capped at 8;
set 1 for the serial A/B). Greedy output is char-identical serial vs threaded. The gain is
largest at small `-ub`, where the fixed per-µbatch dequant is the biggest share of wall, so it
most helps the llama.cpp default `-ub 512`. This raises the streaming baseline the resident
comparison above is drawn against (9B `Q4_K` @ub2048 ~15.8 -> ~18.8), narrowing
`ROCKET_QUANT_RESIDENT`'s edge over *threaded* streaming to ~1.3x: resident still wins (zero
dequant), by less.

**Fusing the dequant into the scatter is not worth it.** The tempting next step, decoding each
weight straight into the native tile lanes and skipping the row-major fp16 intermediate, removes
only that intermediate's traffic (~35 GB written+read against a ~109 s wall ~ **1.4%** at
`-ub 2048`). The cost was the serial decode *compute*, which the threading above already
parallelizes; the buffer was never the bottleneck. Don't re-chase it.

## The per-call overheads: thread spawn and buffer fault

Threading the decode left two fixed *per-call* host costs that are neither the decode compute
nor the tile scatter, and that `-ub` cannot amortize away:

- **Thread create/join.** The row fan-out spawned a fresh `std::thread` set on every weight,
  every micro-batch. Reusing a persistent worker pool instead is **34-54% faster than per-call
  spawn** for the decode of one Gemma-scale weight [HW microbench, A76, `Q4_K`, 8 workers],
  e.g. a 4096×4096 weight ~50 -> ~25 ms, a 15360×3840 weight ~124 -> ~76 ms. The pool runs the
  identical row chunks, so the result is bit-identical (greedy output unchanged).
- **Buffer alloc + page-fault.** The streaming `mul_mat` allocated its `[N,K]` fp16 dequant
  buffer (and `[Mp,N]` output) fresh per op, so each call `mmap`ed and first-touch-faulted a
  large buffer: **~17-60 ms for the big quantized weight** alone [HW microbench, A76]. Holding
  them as grow-only context scratch keeps the pages resident, removing **~86%** of that alloc
  cost. Both buffers are fully overwritten before they are read, so reuse is bit-identical.

The wall the *threaded* dequant still left was as much per-call dispatch (thread spawn plus
page faults) as decode compute. `ggml-rocket` makes both default: a process-wide dequant pool
(created on the first quantized fan-out, shared across backend instances) and reused `B16`/`C16`
context scratch. Like the threading lever they cut host cost, not bytes or MACs, so they help
most where the dequant is the biggest share of wall (the streaming path at small `-ub` and the
RAM-tight case `ROCKET_QUANT_RESIDENT` cannot serve) and are inert on a resident-fp16 run.
Unlike fusing the scatter, these were worth taking: the cost was real per-call dispatch, not
buffer traffic. (The `C16` reuse also serves the F16 streaming path; the large `B16` is the
quant-only piece.)

## The routing floor: short quant prefills belong on the CPU

Below a few hundred rows the per-pass dequant does not amortize and the offload **loses to
the CPU.** 9B Q4_K, one ubatch each, NPU vs CPU (~6.2) [HW sweep, 2026-06-28]:

| M (rows) | 64 | 128 | 256 | 320 | 384 | 512 |
|---|---:|---:|---:|---:|---:|---:|
| NPU t/s | 1.3 | 2.6 | 4.8 | 5.7 | 6.6 | 8.3 |

Crossover ~ **360 rows** (the 27B's pp128->pp512 extrapolates to ~370). So `ggml-rocket`'s
`supports_op` gates quantized weights on **`ROCKET_MIN_M_QUANT`** (default 512 = the default
micro-batch, just past the crossover with margin): quant prefills below it stay on the CPU,
where they beat a dequant-bound offload, while full ubatches still offload and win. The F16
path keeps the lower `ROCKET_MIN_M`; it wins even at small M (F16 pp128 13.3 = 1.86x CPU).
Native int4/int8 modes re-quantize F16 weights (not `ggml_is_quantized`), so the floor is the
dequant path alone.

## Model-size scaling: the NPU prefill win grows with the model

F16 prefill, NPU vs CPU, best per model [HW sweep, 2026-06-27/28, 600 MHz] (best-case runs from an
early scaling sweep; canonical per-model warm medians are in [benchmarks.md](benchmarks.md), for example
9B pp2048 ~24.9 t/s / 3.5x):

| model | params | best NPU prefill | CPU | NPU× |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B | 0.75 B | 106.9 (F16, pp512) | 74.2 | 1.44x |
| Qwen3.5-9B | 8.95 B | 26.8 (F16, pp2048 @ub2048) | 7.34 | **3.65x** |
| Qwen3.6-27B | 27.3 B | 5.35 (Q4_K @ub2048)\* | 1.80 | 2.97x |

\* 27B F16 ~54 GB > 31 GB RAM -> quant-only; extrapolating quant ~0.64x F16, a fitting F16
would be ~4-4.5x CPU. **The footprint, not the NPU, caps the 27B.** The win grows with size
because the CPU slows faster than the NPU as the model grows (the NPU has idle MAC headroom,
[not-mac-bound.md]). Decode (tg, M=1) stays on the CPU at every size, ~equal either backend
(memory-bound).

## Architecture coverage

The sweep validated two architecture families beyond the original Gemma-4 target, both
**PPL-faithful to the CPU** [HW sweep, 2026-06-28]:

- **Qwen3.5 (conventional dense GQA).** GGUF arch `qwen35`; loads and offloads with no
  changes (n_embd 1024-… , n_ff a multiple of 16, GQA, head_dim 256, clearing the offload
  contract).
- **Qwen3.6-27B (hybrid Gated-DeltaNet).** 48 linear-attention (DeltaNet) layers + 16
  standard-attention, n_ff 17408. `GGML_OP_DELTA_NET` / `GGML_OP_SSM_SCAN` are **CPU-only ops**
  (no rocket handler), so the DeltaNet scan stays on the CPU, but it **does not erode the
  prefill win** (holds at 2.97x): the huge FFN + projections offload and dominate, and linear
  attention is O(L)-cheap. The open coverage frontier is a linear-attention NPU primitive,
  which only matters at extreme context.

## Scope

The `-ub` lever is the *default-path* operating point (resident F16 weights, per-call quant
re-pack, 600 MHz). The 0.64x plateau was a **software** limit (no resident dequant cache), not
silicon, confirmed by `ROCKET_QUANT_RESIDENT=1` lifting it to F16 parity by holding the
dequantized fp16 weights resident (above; distinct from the dtype-independent dispatch floor of
[not-mac-bound.md], which it does *not* beat: it reaches F16, not past it). The resident cache is
bounded by the fp16 footprint vs RAM and the NPU IOVA window, so on a model whose fp16 form
exceeds either it goes partly resident (the rest streams); it is the lever for the
"quantized-on-disk, RAM-to-spare" case, while `-ub 2048` stays the zero-RAM-cost default. The
~360-row crossover scales only weakly with model size (9B and 27B both ~360-370); re-measure for a
very different shape before trusting `ROCKET_MIN_M_QUANT`'s default on it.
