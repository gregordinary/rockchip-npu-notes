# Quantized-GGUF prefill and per-micro-batch dequantization

A quantized GGUF (`Q4_K` / `IQ4_XS` / `Q8_0` / …) prefills on the NPU through the
dequant->fp16 streaming path by default. That path dequantizes each weight to fp16 and packs it
into the native tiles per micro-batch, on every forward pass. The per-micro-batch
dequantization, not the matmul, sets quantized prefill throughput [HW sweep]. Where the
model's fp16 form fits RAM, wholly or partly, the resident dequant cache removes that cost.
`ROCKET_QUANT_RESIDENT=auto` at the default `-ub` is the fastest configuration on seven of
seven models measured (§"Residency at the default `-ub`").

Where the fp16 form does not fit, a larger micro-batch (`-ub 2048`) amortizes the cost instead, by an amount that
depends on the model. The routing threshold (`ROCKET_MIN_M_QUANT`) keeps short quantized
prefills on the CPU.

**Conditions:** RK3588, NPU clock ~816 MHz, Qwen3.5-9B and Qwen3.6-27B unless a section names
other models.

**Not characterized:** the dequant crossover at shapes far from the 9B and the 27B, and where
the rest of the sub-4B non-dequant cost goes. Nor is the quant-to-F16 ratio at `-ub 2048` on a
build with the threaded dequant.

## Cause

The native int8 and int4 paths of [not-mac-bound.md](not-mac-bound.md) lose to fp16 on their
int32 readback. The GGUF dequant path loses to fp16 on a host dequantization cost instead. That
cost amortizes with micro-batch size, so it is far more recoverable than the native-int
readback limit.

## Micro-batch size (`-ub`)

The path dequantizes once per micro-batch, so the cost amortizes over the micro-batch's rows.
The llama.cpp default `-ub 512` re-dequantizes the whole model every 512-row chunk, and a
larger micro-batch spreads that fixed cost. `-ub 2048` measures 0.91-1.53x over the default,
model-dependent, and is negative on two models of eleven [HW sweep 2026-08-29/30].

**Conditions:** eleven models from 0.75 to 30.53 B, on the build of 2026-08-29/30, which threads
the dequant (§"Threaded streaming dequant"). The per-model rows, and the mechanism of the change
from the 2026-06-28 figures (§"Revision history"), are in `perf/data/tuning-matrix.md`.

`-ub` spreads the dequant and cannot remove it. On the serial-dequant build of 2026-06-28, the
9B's three quants measured ~0.64x its resident F16 GGUF at `-ub 2048` [HW sweep 2026-06-28]
(§"Revision history"). Only a resident dequant cache, which
dequantizes and packs once, removes the dequant.

Quant type is irrelevant to NPU prefill throughput [HW sweep 2026-06-28]. Q4_K, IQ4_XS and
Q8_0 converge within noise at a given `-ub`, because they run the identical fp16 matmul and
differ only in the dequant. Choose the quant on RAM and quality, not NPU speed. The i-quants
have no NPU penalty. Their thinner *relative* gain is only that the smaller file gives the CPU
baseline a head start.

### Resident dequant cache (`ROCKET_QUANT_RESIDENT`)

`ROCKET_QUANT_RESIDENT` enables the resident dequant cache. `auto` sizes its budget from free
RAM, and a number above 1 sets the budget in MB. `1` uses the 2 GB default budget, which holds
only part of a larger model. The cache dequantizes each quantized weight to fp16 once, and
reuses the F16 prepacked path. Prefill therefore performs neither the per-micro-batch dequant
nor the per-call `packB`, which cuts 12440 to 889 ms [HW sweep].

The cache measures F16 parity, at bit-identical PPL to streaming [HW sweep]. On the 0.8B
`Q4_K` at pp2048, resident measures 89.2 t/s against F16's 88.9. Streaming measures 80.7
@ub2048 and 50.7 @ub512. The resident run's `-ub` was not recorded. On the 9B, in one
session, resident measures 24.6 against the serial streaming path's 15.8 @ub2048. The 9B's 1.56x is ~0.92x the 26.8 F16, because the
9B's ~18 GB fp16 footprint fit the 31 GB RAM and the IOVA window.

The cache gives up the quant RAM saving for the full fp16 resident footprint, and it is opt-in
for that reason. It shows that the 0.64x plateau is a software cost. The 0.64x resembles the
whole-tile resident-int8 0.61-0.66x of [not-mac-bound.md](not-mac-bound.md), by a different
mechanism: host dequant here, int32 readback there. Both costs are recoverable: the cache
recovers the dequant, and the streamed int8 plan most of the readback.

## Dequant component of `-ub`

`ROCKET_DEQUANT_CACHE_MB` (ggml-rocket) separates the documented mechanism of `-ub`, dequant
amortization, from everything else the micro-batch size changes. It holds each streaming
weight's dequantized fp16 form host-side, with one dequant per process. The per-call pack,
submit and placement are exactly the default path's. Greedy output is byte-identical, and the
`[dq-cache]` teardown line reports engagement.

> [!NOTE]
> `ROCKET_DEQUANT_CACHE_MB` is a diagnostic. For deployment, `ROCKET_QUANT_RESIDENT` does
> better: it holds the same fp16 footprint, and also removes the per-call pack and upload.

Comparing `-ub 512` with `-ub 2048` under the cache gives the `-ub` gain with the dequant term
removed [HW sweep 2026-08-31]:

**Conditions:** RK3588 (RK1), NPU clock ~816 MHz, six `Q4_K` models, idle board.

| model | full `-ub` gain | dequant component | non-dequant remainder |
|---|---:|---:|---:|
| Qwen3.5-9B `Q4_K` | 1.430x | 1.378x | 1.038x |
| SmolVLM2-2.2B `Q4_K` | 0.946x | 1.199x | 0.789x |
| Qwen3.5-0.8B `Q4_K` | 1.066x | ~1.10x | unresolved (straddles 1.00) |
| Llama-3.2-3B `Q4_K` | 1.085x | 1.258x | 0.863x |
| Ministral-3-3B `Q4_K` | 1.035x | 1.239x | 0.835x |
| Phi-4-mini `Q4_K` | 1.107x | 1.282x | 0.864x |

On the 9B the documented mechanism holds: ~90% of the gain (in log terms) is dequant
amortization.

On every measured sub-4B model the result inverts. The flag gives a small gain or a loss,
because the larger micro-batch adds a 14-21% non-dequant cost against a 1.20-1.28x dequant
gain. On SmolVLM2, a profile of the cache pair places about half that cost in the driver's
output readback and de-tile. The `read` phase's total grows 64% at identical output bytes, 28.2
to 46.3 s a process. Per job-batch it grows from 1.76 to 10.9 ms, superlinear in M.

The rest is in `wait` [hypothesis]. The measurement
contains no FLASH_ATTN ops, because llama-bench runs `-fa 0`, so attention-chunk shape is not
part of the measured remainder.

### Residency at the default `-ub`

For models above and below 4 B alike, the better option is therefore dequant removal at the
default `-ub`. `ROCKET_QUANT_RESIDENT=auto` at the default `-ub` beats residency combined with
`-ub 2048` [HW sweep 2026-08-31]. It does so on seven of seven models measured both ways. Every
model is 100% resident on every pass except Gemma-4-12B, which places part of its weights:

**Conditions:** RK3588 (RK1), NPU clock ~816 MHz, pp2048. Each value is the speedup over the
streamed default at the default `-ub`.

| model | residency, default `-ub` | residency with `-ub 2048` |
|---|---:|---:|
| SmolVLM2 | 1.346x | 1.002x |
| Llama-3.2-3B | 1.472x | 1.212x |
| Ministral-3-3B | 1.445x | 1.132x |
| Phi-4-mini | 1.508x | 1.211x |
| Ministral-3-8B | 1.651x | 1.325x |
| Qwen3.5-9B | 1.752x | 1.661x |
| Gemma-4-12B | 1.368x | 1.322x |

The 9B shows a second mechanism beyond the sub-4B remainder. Its own non-dequant remainder is
a gain, at 1.038x, which favors `-ub 2048`, and it still gains 5.5% at the default `-ub`. The
second mechanism is the per-call pack and upload. Residency removes it too, and the default
`-ub` presents it four times as often.

Gemma-4-12B places only part of its weights and still gains 3.5% at the default `-ub`. That
configuration also places more of them, 83-87% against 73-74%, because `-ub 2048` uses RAM for
compute buffers ([../TUNING.md](../TUNING.md) §"Residency in place of `-ub 2048`").

### Load sensitivity

The dequant share depends on host load. The same measurement series ran with a single core
kept busy by another process. It measured the `-ub` gain at 1.484x on Qwen3.5-0.8B, 1.526x on
Qwen3.5-9B and 1.120x on SmolVLM2, against 1.066x, 1.430x and 0.946x on an idle board. The busy
process takes capacity from the A76-pinned dequant pool, which the `-ub 512` configuration
uses four times as often. Host load therefore inflates the flag on every model, and flips
SmolVLM2's sign. A `-ub` recommendation measured on a loaded host overstates the flag for an
idle deployment.

## Threaded streaming dequant

The streaming path's per-micro-batch dequant is the cost that `-ub` amortizes and
`ROCKET_QUANT_RESIDENT` eliminates. Run serially, it decodes every weight's `[N,K]` rows
quant->fp16 on one core before the (already multicore) tile scatter runs. The rows are
independent (a private K-float scratch in, a disjoint fp16 slice out). Fanning them across
cores therefore recovers a large share of that cost, with no extra RAM. Threading is the option
for the RAM-constrained case, which the resident fp16 copy cannot serve. The table is the 9B
`Q4_K`, serial against threaded dequant over eight threads on the A76 and A55 cores
[HW sweep 2026-06-28]:

| `-ub` | serial | threaded | gain |
|---|---:|---:|---:|
| 512 (default) | 7.5 | 10.0 | +33% |
| 2048 | 15.9 | 18.8 | +19% |

Threaded dequant is the default. `ROCKET_DEQUANT_THREADS` unset or 0 selects the A76 core
count, with the workers pinned to the A76 cluster. Where no big cluster is detected, it selects
`hardware_concurrency` capped at 8. 1 selects the serial path. Greedy output is char-identical serial
against threaded. The gain is largest at small `-ub`, where the fixed per-micro-batch dequant is the
largest share of wall-clock time. Threading therefore helps the llama.cpp default `-ub 512`
most.

Threading raises the streaming baseline of the resident comparison above. That comparison's
serial streaming measured 15.8 t/s, and this one's serial and threaded configurations 15.9 and
18.8 (9B `Q4_K` @ub2048). Against *threaded* streaming, `ROCKET_QUANT_RESIDENT`'s gain narrows
to ~1.3x [host-computed across the two sessions: 24.6 / 18.8]. Resident still wins (zero
dequant), by less.

Fusing the dequant into the scatter gains little. A fused step decodes each weight straight
into the native tile lanes, and skips the row-major fp16 intermediate. It removes only that intermediate's traffic:
~35 GB written+read against a ~109 s wall-clock time, ~1.4% at `-ub 2048`. The cost is the
serial decode *compute*, which threading already parallelizes, and the buffer is not the
bottleneck.

## Per-call thread spawn and buffer fault

A threaded decode still has two fixed *per-call* host costs. They are neither the decode
compute nor the tile scatter, and `-ub` cannot amortize them away:

- **Thread create/join.** A per-call row fan-out spawns a fresh `std::thread` set on every
  weight, every micro-batch. A persistent worker pool is 34-54% faster than per-call spawn
  for the decode of one Gemma-scale `Q4_K` weight [HW sweep]. The decode runs on the A76
  cores, with 8 workers. For example, a 4096×4096 weight goes ~50 -> ~25 ms and a 15360×3840
  weight ~124 -> ~76 ms. The pool runs the identical row chunks, so the result is bit-identical
  (greedy output unchanged).
- **Buffer alloc + page-fault.** A streaming `mul_mat` that allocates its `[N,K]` fp16
  dequant buffer (and `[Mp,N]` output) fresh per op `mmap`s and first-touch-faults a large
  buffer on each call. The allocation costs ~17-60 ms on the A76 cores for the big quantized
  weight alone [HW sweep]. Holding both buffers as grow-only context scratch keeps the pages
  resident, which removes ~86% of that alloc cost. Both buffers are fully overwritten before
  they are read, so reuse is bit-identical.

The wall-clock time the *threaded* dequant leaves is as much per-call dispatch (thread spawn
plus page faults) as decode compute. Both fixes are the default in `ggml-rocket`: a
process-wide dequant pool (created on the first quantized fan-out, shared across backend
instances) and reused `B16`/`C16` context scratch. Like threading, they cut host cost, not
bytes or MACs. They therefore help most where the dequant is the largest share of wall-clock
time. Those cases are the streaming path at small `-ub`, and the RAM-tight case that
`ROCKET_QUANT_RESIDENT` cannot serve. They have no effect on a resident-fp16 run.

The `C16` reuse also serves the F16 streaming path, and only the quantized path uses the large
`B16`.

## Routing threshold (`ROCKET_MIN_M_QUANT`)

The `supports_op` of `ggml-rocket` offloads a quantized weight only from `ROCKET_MIN_M_QUANT`
rows, 512 by default, the default micro-batch. Below it a quantized prefill runs on the CPU.
On Qwen3.5-9B the offload measures 0.93x the CPU at 128 rows and 1.46-1.80x at 256-384 rows
[HW sweep 2026-11-02]. The crossover therefore sits between 128 and 256 rows, ~145 by linear
interpolation [host-computed]. The default leaves the 256-511-row gain on the CPU.

**Conditions:** RK3588 (RK1), NPU clock 928 MHz, governor `performance`, llama.cpp b11539.
Qwen3.5-9B `Q4_K_M` streaming its weights, one micro-batch of M rows (`-b` and `-ub` at M),
`ROCKET_MIN_M_QUANT=1`, `--repack 0`. The CPU backend at 8 threads with its weight repack on.
Measured 2026-11-02.

| M (rows) | 128 | 256 | 384 | 512 |
|---|---:|---:|---:|---:|
| NPU, t/s | 9.4 | 15.3-15.5 | 19.0-19.2 | 21.7-22.0 |
| CPU, t/s | 10.1-10.2 | 10.5 | 10.6-10.7 | 10.7 |
| NPU against CPU | 0.93x | 1.46-1.48x | 1.78-1.80x | 2.02-2.05x |

**Not characterized:** the crossover on another model or under `ROCKET_QUANT_RESIDENT`.

> [!NOTE]
> On a workload of 256-511-row quantized prompts, `ROCKET_MIN_M_QUANT=256` takes the gain on
> this model. Measure the crossover for a very different shape before lowering it there.

The F16 path keeps the lower `ROCKET_MIN_M`. It wins even at small M: Qwen3.5-9B F16 measures
3.3-3.4x the CPU at pp128 [HW sweep 2026-11-02]. Native int4/int8 modes re-quantize F16 weights
(not `ggml_is_quantized`), so the threshold applies to the dequant path alone.

## Model-size scaling

At pp2048, seven dense F16 models of 1.8-11.9 B prefill at 4.2-5.8x the CPU backend, lowest on
the smallest. Qwen3.5-9B F16 prefills at 36.1 t/s, 4.9x, and Qwen3.6-27B `Q4_K_M` at 11.5 t/s,
3.8x [HW sweep 2026-11-01]. The canonical per-model rates are in
[benchmarks.md](benchmarks.md).

**Conditions:** RK3588 (RK1), NPU clock 928 MHz, governor `performance`, llama.cpp b11539,
against the CPU backend at 8 threads. The 9B at the default `-b` and `-ub`, the 27B at
`-b 2048 -ub 2048` against a CPU backend with its weight repack on.

The 27B F16 is ~54 GB against 31 GB of RAM, so the 27B runs only quantized. The footprint, not
the NPU, limits it to the quantized path. Decode (tg, M=1) stays on the CPU at every size
([decode-gemv.md](decode-gemv.md)).

## Architecture coverage

The sweep validated two architecture families besides Gemma-4. Both are PPL-faithful to the
CPU [HW sweep 2026-06-28]:

- **Qwen3.5 (conventional dense GQA).** The GGUF arch is `qwen35`. It loads and offloads
  with no changes (n_embd 1024-… , n_ff a multiple of 16, GQA, head_dim 256, meeting the
  offload contract).
- **Qwen3.6-27B (hybrid Gated-DeltaNet).** The model has 48 linear-attention (DeltaNet)
  layers + 16 standard-attention, n_ff 17408. The ops `GGML_OP_DELTA_NET` /
  `GGML_OP_SSM_SCAN` are CPU-only (no rocket handler), so the DeltaNet scan stays on the CPU.
  The prefill gain holds, at 3.8x on 928 MHz (§"Model-size scaling"): the large FFN and
  projections offload and dominate, and linear attention is O(L)-cheap. A linear-attention NPU primitive is not implemented, and it
  matters only at extreme context.

## Applicability

The `-ub` results hold on the streamed quantized path, which dequantizes and re-packs on every
call, at ~816 MHz. The resident dequant cache reaches F16, not past it. It does not beat the
datatype-independent dispatch limit of [not-mac-bound.md](not-mac-bound.md).

The resident cache is bounded by the fp16 footprint against RAM, and by the NPU IOVA window. A
model whose fp16 form exceeds either therefore goes partly resident, and the rest streams.

## Evidence

### Micro-batch size (`-ub`)

The current figure is from rotated passes.

### Revision history

The `-ub 2048` gain over the `-ub 512` default, at pp2048:

| Date | Model | Superseded value | Conditions | Current reading |
|---|---|---|---|---|
| 2026-06-28 | Qwen3.5-9B `Q4_K` | 2.098x (8.2 to 17.2 t/s), quoted as ~2x | ~816 MHz, serial streaming dequant | 1.424x (19.14 to 27.26 t/s), 2026-08-29 |
| 2026-06-28 | Qwen3.6-27B `Q4_K` | 2.250x (2.4 to 5.4 t/s) | ~816 MHz, serial streaming dequant | 1.532x (6.06 to 9.28 t/s), 2026-08-29 |

Both current readings are three rotated passes at ~816 MHz on the build of 2026-08-29/30. Its
threaded dequant, dequant worker pool and reused scratch cut the `-ub 512` configuration's fixed
cost more than the `-ub 2048` one's.

The 2026-06-28 sweep, 9B at pp2048, by `-ub` [HW sweep 2026-06-28, ~816 MHz]:

| `-ub` | Q4_K_M | IQ4_XS | Q8_0 |
|---|---:|---:|---:|
| 512 (default) | 8.2 | 8.0 | 8.3 |
| 1024 | 12.7 | 12.5 | 13.1 |
| 2048 | 17.2 | 17.2 | 17.4 |

The quant-type convergence and the ~0.64x plateau in §"Micro-batch size (`-ub`)" are readings
of this table. The 9B's resident F16 GGUF measured 26.8 t/s at `-ub 2048` in the same sweep, so
the residual per-2048 dequant was ~35% of the quantized prefill. On the build of 2026-07-02 the
9B `Q4_K_M` measured 22.88 t/s at `-ub 2048` [HW sweep 2026-07-02]. The F16 GGUF measured 24.85
at the default `-ub`, so the ratio is 0.92x at unequal `-ub` [host-computed].

The prefill gain by model size is the best run per model of a scaling sweep
[HW sweep 2026-06-27/28, ~816 MHz]. The 27B's CPU backend ran with the weight repack off:

| model | params | best NPU prefill | CPU | speedup |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B | 0.75 B | 106.9 (F16, pp512) | 74.2 | 1.44x |
| Qwen3.5-9B | 8.95 B | 26.8 (F16, pp2048 @ub2048) | 7.34 | 3.65x |
| Qwen3.6-27B | 27.3 B | 5.35 (Q4_K @ub2048) | 1.80 | 2.97x |

The 0.8B row is that model's only speedup figure.

The routing crossover and the small-M F16 figure:

| Date | Superseded reading | Conditions | Current reading |
|---|---|---|---|
| 2026-06-28 | Crossover ~360 rows on Qwen3.5-9B `Q4_K` (NPU 2.6 t/s at 128 rows, 4.8 at 256, 6.6 at 384, 8.3 at 512, against a CPU at ~6.2); the 27B's pp128 to pp512 extrapolated to ~370 | ~816 MHz, serial streaming dequant, CPU without the weight repack | Between 128 and 256 rows, 2026-11-02 |
| 2026-06-28 | F16 pp128 13.3 t/s, 1.86x the CPU | ~816 MHz | Qwen3.5-9B F16 pp128 3.3-3.4x, 2026-11-02 |

### Resident dequant cache

The profile figures in §"Resident dequant cache" are `MM_PROFILE` readouts. The parity and PPL
figures were measured in one session.

### Dequant component of `-ub`

Each model is three rotated passes on an audited idle board. The SmolVLM2 split is a profiled
comparison of the cache pair. Each row of the residency table is three rotated passes.

### Load sensitivity

The busy core was a leaked spinner process.

### Threaded streaming dequant

Serial and threaded ran as a paired comparison in one session, with `ROCKET_DEQUANT_THREADS`
at 1 for the serial configuration.

### Per-call thread spawn and buffer fault

Both figures are from microbenchmarks.

### Routing threshold

The 2026-11-02 series ran two passes with the arm order rotated, each process after a memory
reset. Each NPU arm followed a discarded warm-up process, and every arm ran `llama-bench -r 2`.
Each ratio pairs the arms within one pass. The CPU at `taskset 0xf0 -t 4` measured 9.7-9.8 t/s at every M, 3-10%
below 8 threads.

### Model-size scaling

The figures are one measurement series of two passes, the second reversing the GGUF order and
the configuration order. Each speedup is the mean of the two per-pass ratios. The per-model
rates are in ggml-rocket's README and MODELS.md.
