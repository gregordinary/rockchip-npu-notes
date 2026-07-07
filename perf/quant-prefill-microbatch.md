# Quantized-GGUF prefill is micro-batch-dequant-bound

A quantized GGUF (`Q4_K` / `IQ4_XS` / `Q8_0` / …) prefills on the NPU through the
dequant→fp16 streaming path: each weight is dequantized to fp16 and packed into the native
tiles **per micro-batch**, every forward pass (the resident fp16 weight cache is F16-only —
quant weights re-pack each call). That per-ubatch dequant, not the matmul, sets quantized
prefill throughput. Two consequences follow — a runtime lever (`-ub`) and a routing floor
(`ROCKET_MIN_M_QUANT`). Measured on Qwen3.5-9B / Qwen3.6-27B, 600 MHz.

This refines [not-mac-bound.md](not-mac-bound.md): that note shows the *native* int8/int4
paths lose to fp16 on the int32-readback wall; this is the *GGUF-quant dequant* path, whose
loss is a host dequant cost that **amortizes with micro-batch size** — so it is far more
recoverable than the native-int readback wall.

## The micro-batch lever: `-ub 2048` ~doubles quantized prefill

The dequant is paid once per micro-batch, so it amortizes over the ubatch's rows. The
llama.cpp default `-ub 512` re-dequantizes the whole model every 512-row chunk; a larger
ubatch spreads that fixed cost. **9B Q4_K, pp2048, by `-ub`** [HW sweep, 2026-06-28, 600 MHz]:

| `-ub` | Q4_K_M | IQ4_XS | Q8_0 |
|---|---:|---:|---:|
| 512 (default) | 8.2 | 8.0 | 8.3 |
| 1024 | 12.7 | 12.5 | 13.1 |
| 2048 | **17.2** | **17.2** | **17.4** |

- **`-ub 2048` is ~2.1× over the default** — a free runtime knob, no code change. The 27B
  repeats it (Q4_K pp2048 2.4 → 5.4 t/s).
- **Quant type is irrelevant to NPU prefill throughput.** Q4_K / IQ4_XS / Q8_0 converge
  within noise at a given `-ub` — they run the identical fp16 matmul; only the dequant
  differs, and at `-ub 2048` it is amortized away. Choose the quant on RAM / quality, not NPU
  speed. (i-quants carry no NPU penalty; their thinner *relative* win is only that the smaller
  file gives the CPU baseline a head start.)
- **Even amortized, quant ≈ 0.64× F16 — unless the dequant is made resident.** F16 stays resident
  (zero dequant) at 26.8 t/s @ub2048 on the 9B; the quants plateau ~17.3. The residual per-2048
  dequant is still ~35%, and **`-ub` cannot close it** — only a resident dequant cache (dequant→pack
  once) can. That cache now exists: **`ROCKET_QUANT_RESIDENT=1`** dequantizes each quantized weight
  to fp16 **once** and reuses the F16 prepacked path, so prefill pays neither the per-µbatch dequant
  nor the per-call `packB` (it collapses 12440→889 ms [HW MM_PROFILE]). Measured **F16 parity** [HW,
  0.8B `Q4_K` pp2048, same session: resident 89.2 t/s == F16 88.9, vs streaming 80.7 @ub2048 / 50.7
  @ub512 → 1.50× at the default `-ub`; **9B same session: resident 24.6 vs streaming 15.8 = 1.56×,
  ≈0.92× the 26.8 F16** — the 9B's ~18 GB fp16 footprint fit the 31 GB RAM + IOVA window], **bit-identical
  PPL** to streaming. It trades the quant RAM
  saving back for the full fp16 resident footprint (opt-in), and confirms the 0.64× plateau was a
  **software** cost, not silicon. The 0.64× echoed the native resident-int8 0.60× of
  [not-mac-bound.md] by a different mechanism (host dequant here, int32 readback there) — but unlike
  the int8 readback wall, this one was recoverable, and now is.

## The threading lever: the streaming dequant was serial

The per-µbatch dequant that `-ub` amortizes and `ROCKET_QUANT_RESIDENT` eliminates was, on the
streaming path, **single-threaded**: one core decoded every weight's `[N,K]` rows quant→fp16
before the (already multicore) tile scatter ran. The rows are independent — a private K-float
scratch in, a disjoint fp16 slice out — so fanning them across the A76+A55 recovers a large
share of that cost with **no extra RAM**. That makes it the lever for the RAM-constrained case
the resident fp16 copy can't serve. **9B `Q4_K`, serial vs threaded dequant** [HW A/B, same
session, 2026-06-28]:

| `-ub` | serial | threaded | gain |
|---|---:|---:|---:|
| 512 (default) | 7.5 | 10.0 | **+33%** |
| 2048 | 15.9 | 18.8 | **+19%** |

Threaded is the default (`ROCKET_DEQUANT_THREADS`, auto = `hardware_concurrency` capped at 8;
set 1 for the serial A/B). Greedy output is char-identical serial vs threaded. The gain is
largest at small `-ub`, where the fixed per-µbatch dequant is the biggest share of wall — so it
most helps the llama.cpp default `-ub 512`. This raises the streaming baseline the resident
comparison above is drawn against (9B `Q4_K` @ub2048 ~15.8 → ~18.8), narrowing
`ROCKET_QUANT_RESIDENT`'s edge over *threaded* streaming to ~1.3×: resident still wins (zero
dequant), by less.

**Fusing the dequant into the scatter is not worth it.** The tempting next step — decode each
weight straight into the native tile lanes, skipping the row-major fp16 intermediate — removes
only that intermediate's traffic (~35 GB written+read against a ~109 s wall ≈ **1.4%** at
`-ub 2048`). The cost was the serial decode *compute*, which the threading above already
parallelizes; the buffer was never the bottleneck. Don't re-chase it.

## The per-call overheads: thread spawn and buffer fault

Threading the decode left two fixed *per-call* host costs that are neither the decode compute
nor the tile scatter, and that `-ub` cannot amortize away:

- **Thread create/join.** The row fan-out spawned a fresh `std::thread` set on every weight,
  every micro-batch. Reusing a persistent worker pool instead is **34–54% faster than per-call
  spawn** for the decode of one Gemma-scale weight [HW microbench, A76, `Q4_K`, 8 workers] —
  e.g. a 4096×4096 weight ~50 → ~25 ms, a 15360×3840 weight ~124 → ~76 ms. The pool runs the
  identical row chunks, so the result is bit-identical (greedy output unchanged).
- **Buffer alloc + page-fault.** The streaming `mul_mat` allocated its `[N,K]` fp16 dequant
  buffer (and `[Mp,N]` output) fresh per op, so each call `mmap`ed and first-touch-faulted a
  large buffer — **~17–60 ms for the big quantized weight** alone [HW microbench, A76]. Holding
  them as grow-only context scratch keeps the pages resident, removing **~86%** of that alloc
  cost. Both buffers are fully overwritten before they are read, so reuse is bit-identical.

The wall the *threaded* dequant still left was as much per-call dispatch — thread spawn plus
page faults — as decode compute. `ggml-rocket` makes both default: a process-wide dequant pool
(created on the first quantized fan-out, shared across backend instances) and reused `B16`/`C16`
context scratch. Like the threading lever they cut host cost, not bytes or MACs, so they help
most where the dequant is the biggest share of wall — the streaming path at small `-ub` and the
RAM-tight case `ROCKET_QUANT_RESIDENT` can't serve — and are inert on a resident-fp16 run.
Unlike fusing the scatter, these were worth taking: the cost was real per-call dispatch, not
buffer traffic. (The `C16` reuse also serves the F16 streaming path; the large `B16` is the
quant-only piece.)

## The routing floor: short quant prefills belong on the CPU

Below a few hundred rows the per-pass dequant does not amortize and the offload **loses to
the CPU.** 9B Q4_K, one ubatch each, NPU vs CPU (~6.2) [HW sweep, 2026-06-28]:

| M (rows) | 64 | 128 | 256 | 320 | 384 | 512 |
|---|---:|---:|---:|---:|---:|---:|
| NPU t/s | 1.3 | 2.6 | 4.8 | 5.7 | 6.6 | 8.3 |

Crossover ≈ **360 rows** (the 27B's pp128→pp512 extrapolates to ~370). So `ggml-rocket`'s
`supports_op` gates quantized weights on **`ROCKET_MIN_M_QUANT`** (default 512 = the default
micro-batch, just past the crossover with margin): quant prefills below it stay on the CPU,
where they beat a dequant-bound offload, while full ubatches still offload and win. The F16
path keeps the lower `ROCKET_MIN_M` — it wins even at small M (F16 pp128 13.3 = 1.86× CPU).
Native int4/int8 modes re-quantize F16 weights (not `ggml_is_quantized`), so the floor is the
dequant path alone.

## Model-size scaling: the NPU prefill win grows with the model

F16 prefill, NPU vs CPU, best per model [HW sweep, 2026-06-27/28, 600 MHz]:

| model | params | best NPU prefill | CPU | NPU× |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B | 0.75 B | 106.9 (F16, pp512) | 74.2 | 1.44× |
| Qwen3.5-9B | 8.95 B | 26.8 (F16, pp2048 @ub2048) | 7.34 | **3.65×** |
| Qwen3.6-27B | 27.3 B | 5.35 (Q4_K @ub2048)\* | 1.80 | 2.97× |

\* 27B F16 ≈ 54 GB > 31 GB RAM → quant-only; extrapolating quant ≈ 0.64× F16, a fitting F16
would be ~4–4.5× CPU. **The footprint, not the NPU, caps the 27B.** The win grows with size
because the CPU slows faster than the NPU as the model grows (the NPU has idle MAC headroom —
[not-mac-bound.md]). Decode (tg, M=1) stays on the CPU at every size, ~equal either backend
(memory-bound).

## Architecture coverage

The sweep validated two architecture families beyond the original Gemma-4 target, both
**PPL-faithful to the CPU** [HW sweep, 2026-06-28]:

- **Qwen3.5 (conventional dense GQA)** — GGUF arch `qwen35`; loads and offloads with no
  changes (n_embd 1024–… , n_ff a multiple of 16, GQA, head_dim 256 — clears the offload
  contract).
- **Qwen3.6-27B (hybrid Gated-DeltaNet)** — 48 linear-attention (DeltaNet) layers + 16
  standard-attention, n_ff 17408. `GGML_OP_DELTA_NET` / `GGML_OP_SSM_SCAN` are **CPU-only ops**
  (no rocket handler), so the DeltaNet scan stays on the CPU — but it **does not erode the
  prefill win** (holds at 2.97×): the huge FFN + projections offload and dominate, and linear
  attention is O(L)-cheap. The open coverage frontier is a linear-attention NPU primitive,
  which only matters at extreme context.

## Scope

The `-ub` lever is the *default-path* operating point (resident F16 weights, per-call quant
re-pack, 600 MHz). The 0.64× plateau was a **software** limit (no resident dequant cache), not
silicon — confirmed now that `ROCKET_QUANT_RESIDENT=1` lifts it to F16 parity by holding the
dequantized fp16 weights resident (above; distinct from the dtype-independent dispatch floor of
[not-mac-bound.md], which it does *not* beat — it reaches F16, not past it). The resident cache is
bounded by the fp16 footprint vs RAM and the NPU IOVA window, so on a model whose fp16 form
exceeds either it goes partly resident (the rest streams); it is the lever for the
"quantized-on-disk, RAM-to-spare" case, while `-ub 2048` stays the zero-RAM-cost default. The
~360-row crossover scales only weakly with model size (9B and 27B both ~360–370); re-measure for a
very different shape before trusting `ROCKET_MIN_M_QUANT`'s default on it.
